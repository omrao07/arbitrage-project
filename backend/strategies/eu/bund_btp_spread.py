#!/usr/bin/env python3
"""
bund_btp_spread.py — Analytics for Germany Bund vs. Italy BTP yield spreads

What it does
------------
- Loads wide CSV of sovereign yields (Date × Columns), finds the Bund & BTP columns for a given tenor
- Computes the Bund–BTP spread (IT - DE) in basis points
- Rolling statistics: mean, stdev, z-score
- Rolling hedge ratio (IT ~ a + b*DE) and residual ("idiosyncratic") spread
- Mean-reversion diagnostics on residual (AR(1) phi and half-life)
- Optional DV01-neutral notional ratio using user-provided durations
- Optional multi-tenor curve spread table if several tenors exist (e.g., 2Y/5Y/10Y)
- Stress table for user-specified shocks (e.g., +25/+50/+100 bps BTP widening)

Inputs (CSV)
------------
--yields yields.csv      Wide CSV with first column Date, other columns named like:
                         DE10Y, BUND_10Y, GER_10, IT10Y, BTP_10Y, ITA_10, etc. Decimals OR percentages.
                         Script auto-detects units.

Key Options
-----------
--tenor 10               Tenor to target (string match, e.g., '10', '10Y')
--window 60              Rolling window for stats/hedge
--dur-de 9.0             Bund duration (years) for DV01 neutral ratio (optional)
--dur-it 7.5             BTP duration (years) for DV01 neutral ratio (optional)
--shocks "25,50,100"     Stress widening shocks in bps (applied to IT-DE)
--outdir out_bundbtp     Output directory

Outputs
-------
- spread_ts.csv          Time series with DE_yld, IT_yld, spread_bps, z, hedge_beta, resid_bps
- spread_summary.json    Latest snapshot & long-run stats
- hedge_ratios.csv       Rolling intercept/β and residual stdev
- curve_spreads.csv      If multiple tenors found: per-date spreads for each tenor
- stress_table.csv       P&L-style sensitivities under shocks (bp)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------- I/O helpers --------------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    c0 = df.columns[0]
    if str(c0).lower() in ("date", "time"):
        df[c0] = pd.to_datetime(df[c0])
        df = df.set_index(c0)
    else:
        # Try parse first column as date anyway
        try:
            df[c0] = pd.to_datetime(df[c0])
            df = df.set_index(c0)
        except Exception:
            pass
    return df.sort_index()


def _colname_norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", c.lower())


def detect_columns(cols: List[str], tenor: str) -> Tuple[Optional[str], Optional[str], Dict[str, List[str]]]:
    """
    Heuristics to find DE vs IT for a target tenor.
    Returns (de_col, it_col, grouped_tenors) where grouped_tenors is {tenor: [DEcol, ITcol, ...]} for optional curve table.
    """
    tnorm = _colname_norm(tenor)
    de_alias = ["de", "ger", "bund", "germany"]
    it_alias = ["it", "ita", "btp", "italy"]

    norm_map = {c: _colname_norm(c) for c in cols}

    # group by tenor tokens present in colnames (e.g., 2y,5y,10y)
    tenor_pat = re.compile(r"(?:^|[^0-9])([0-9]{1,2})(?:y|yr|year|years)?$")
    buckets: Dict[str, List[str]] = {}
    for c, cn in norm_map.items():
        # find tenor token at end (or include raw digits)
        m = re.search(r"([0-9]{1,2})y", cn) or re.search(r"([0-9]{1,2})(?![0-9])", cn)
        if m:
            k = m.group(1)
            buckets.setdefault(k, []).append(c)

    # Pick target tenor bucket
    target_bucket = []
    for k, members in buckets.items():
        if k == tnorm or (tnorm.endswith("y") and k == tnorm[:-1]):
            target_bucket = members
            break
    # Fallback: any column containing tenor string
    if not target_bucket:
        target_bucket = [c for c, cn in norm_map.items() if tnorm in cn]

    # Within bucket, pick DE / IT
    def pick(members: List[str], aliases: List[str]) -> Optional[str]:
        # rank by alias hits; prefer those with 'y' suffix or exact tenor
        best, score = None, -1
        for c in members:
            cn = norm_map[c]
            s = 0
            if any(a in cn for a in aliases):
                s += 2
            if tnorm in cn:
                s += 1
            # penalize if it also mentions the other country alias
            if any(a in cn for a in (it_alias if aliases is de_alias else de_alias)):
                s -= 1
            if s > score:
                best, score = c, s
        return best

    de_col = pick(target_bucket or list(cols), de_alias)
    it_col = pick(target_bucket or list(cols), it_alias)

    return de_col, it_col, buckets


def coerce_units(series: pd.Series) -> pd.Series:
    """
    Auto-detect if yields are in % (e.g., 2.34) or decimals (0.0234).
    Decision: if median > 1.5, treat as %, convert to decimals.
    """
    s = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(s.values))
    if np.isfinite(med) and med > 1.5:
        s = s / 100.0
    return s


# -------------- Stats helpers --------------
def rolling_regression(y: pd.Series, x: pd.Series, window: int) -> pd.DataFrame:
    """
    OLS y = a + b x on rolling window.
    Returns DataFrame with columns: alpha, beta, resid_std
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    alpha = pd.Series(index=df.index, dtype=float)
    beta = pd.Series(index=df.index, dtype=float)
    resid_std = pd.Series(index=df.index, dtype=float)

    # Efficient rolling with expanding arrays
    Y = df["y"].values
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    for i in range(window - 1, len(df)):
        sl = slice(i - window + 1, i + 1)
        Xi = X[sl]
        Yi = Y[sl]
        # (X'X)^-1 X'Y
        XtX = Xi.T @ Xi
        try:
            inv = np.linalg.pinv(XtX)
        except Exception:
            inv = np.linalg.pinv(XtX + 1e-12 * np.eye(2))
        b = inv @ (Xi.T @ Yi)
        a_i, b_i = float(b[0]), float(b[1])
        yhat = Xi @ b
        r = Yi - yhat
        s = float(np.sqrt(max(np.sum(r * r) / max(window - 2, 1), 0.0)))
        idx = df.index[i]
        alpha.at[idx] = a_i
        beta.at[idx] = b_i
        resid_std.at[idx] = s

    return pd.DataFrame({"alpha": alpha, "beta": beta, "resid_std": resid_std})


def ar1_phil_hl(series: pd.Series) -> Tuple[float, Optional[float]]:
    """
    AR(1) x_t = phi x_{t-1} + e_t (no intercept) estimated by OLS on overlapping pairs.
    Returns (phi, half_life_days). Half-life = -ln(2) / ln(phi), if 0<phi<1.
    """
    s = series.dropna()
    if len(s) < 30:
        return (np.nan, None)
    x = s.values
    xlag = x[:-1]
    y = x[1:]
    denom = float(np.dot(xlag, xlag))
    if denom <= 1e-12:
        return (np.nan, None)
    phi = float(np.dot(xlag, y) / denom)
    hl = None
    if 0 < phi < 1:
        hl = float(-np.log(2.0) / np.log(phi))
    return (phi, hl)


# -------------- Core --------------
def main():
    ap = argparse.ArgumentParser(description="Bund vs. BTP spread analytics")
    ap.add_argument("--yields", required=True, help="Wide CSV Date × Columns with yields (decimals or %)")
    ap.add_argument("--tenor", default="10", help="Target tenor token, e.g., '10' or '10Y'")
    ap.add_argument("--window", type=int, default=60, help="Rolling window size")
    ap.add_argument("--dur-de", type=float, default=None, help="Bund duration (years) for DV01-neutral ratio")
    ap.add_argument("--dur-it", type=float, default=None, help="BTP duration (years) for DV01-neutral ratio")
    ap.add_argument("--shocks", default="25,50,100", help="Comma list of widening shocks (bps) for stress table")
    ap.add_argument("--outdir", default="out_bundbtp", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = read_wide_csv(args.yields)
    if df.empty or df.shape[1] < 2:
        raise SystemExit("Yields CSV appears empty or lacks sufficient columns.")
    de_col, it_col, tenor_groups = detect_columns(list(df.columns), args.tenor)
    if not de_col or not it_col:
        raise SystemExit(f"Could not detect DE/IT columns for tenor '{args.tenor}'. "
                         f"Found DE={de_col}, IT={it_col}. Columns present: {list(df.columns)[:6]}...")

    de = coerce_units(df[de_col]).rename("DE")
    it = coerce_units(df[it_col]).rename("IT")
    data = pd.concat([de, it], axis=1).dropna()

    # Spread (bps) and rolling stats
    spread = (data["IT"] - data["DE"]) * 1e4  # in bps
    mu = spread.rolling(args.window).mean()
    sd = spread.rolling(args.window).std(ddof=1)
    z = (spread - mu) / (sd + 1e-12)

    # Rolling hedge ratio (IT on DE)
    rr = rolling_regression(data["IT"], data["DE"], args.window)
    # Residual "idiosyncratic" component (in bps)
    # Align rr to data index
    rr_aligned = rr.reindex(data.index)
    yhat = rr_aligned["alpha"] + rr_aligned["beta"] * data["DE"]
    resid_bps = (data["IT"] - yhat) * 1e4

    # Mean reversion diagnostics on residual
    phi, half_life = ar1_phil_hl(resid_bps)

    # DV01-neutral notional ratio (optional)
    dv01_ratio = None
    if args.dur_de and args.dur_it and args.dur_de > 0 and args.dur_it > 0:
        # DV01 ∝ duration; to DV01-neutral a 1 notional short in IT vs long in DE requires ratio = dur_it / dur_de
        dv01_ratio = float(args.dur_it / args.dur_de)

    # Compose time series output
    ts = pd.DataFrame({
        "DE_yield": data["DE"],
        "IT_yield": data["IT"],
        "spread_bps": spread,
        "spread_mu_bps": mu,
        "spread_sd_bps": sd,
        "zscore": z,
        "hedge_alpha": rr_aligned["alpha"],
        "hedge_beta": rr_aligned["beta"],
        "resid_bps": resid_bps,
        "resid_sd_bps": rr_aligned["resid_std"] * 1e4
    })
    ts.to_csv(outdir / "spread_ts.csv", float_format="%.6f")

    # Hedge ratios table (last N rows)
    rr.to_csv(outdir / "hedge_ratios.csv", float_format="%.8f")

    # Optional curve spreads if multiple tenors exist
    curve_out = []
    for k, members in sorted(tenor_groups.items(), key=lambda kv: int(kv[0])):
        # find one DE and one IT per bucket
        dcol, icol, _ = detect_columns(members, k)
        if dcol and icol:
            d = coerce_units(df[dcol])
            i = coerce_units(df[icol])
            sp = (i - d) * 1e4
            tmp = sp.rename(f"spread_{k}Y_bps")
            curve_out.append(tmp)
    if curve_out:
        curve_df = pd.concat(curve_out, axis=1).dropna(how="all")
        curve_df.to_csv(outdir / "curve_spreads.csv", float_format="%.6f")

    # Stress table (bps widening)
    shocks = [int(s.strip()) for s in str(args.shocks).split(",") if s.strip()]
    latest = ts.dropna().iloc[-1] if not ts.dropna().empty else None
    stress_rows = []
    for sh in shocks:
        # P&L per 100 notional, DV01-neutral ratio if provided
        # Pure spread change Δs (bps) × (w_IT - w_DE) with DV01 weights; here we just report spread change impact proxy.
        pnl_spread = -sh  # long spread tightener loses when spread widens; sign is contextual—report Δspread itself and z-shift
        z_new = None
        if latest is not None and np.isfinite(latest["spread_sd_bps"]) and latest["spread_sd_bps"] > 1e-9:
            z_new = float((float(latest["spread_bps"]) + sh - float(latest["spread_mu_bps"])) / float(latest["spread_sd_bps"]))
        stress_rows.append({
            "shock_bps": sh,
            "spread_bps_new": (None if latest is None else float(latest["spread_bps"]) + sh),
            "z_after": z_new,
            "note": "Positive shock means BTP widens vs Bund"
        })
    pd.DataFrame(stress_rows).to_csv(outdir / "stress_table.csv", index=False)

    # Summary JSON
    summary = {
        "tenor_selected": str(args.tenor),
        "de_column": de_col,
        "it_column": it_col,
        "last_date": (None if ts.empty else str(ts.index[-1].date())),
        "last_spread_bps": (None if ts.empty else float(ts["spread_bps"].iloc[-1])),
        "last_zscore": (None if ts.empty else float(ts["zscore"].iloc[-1])),
        "rolling_window": int(args.window),
        "residual_ar1_phi": (None if not np.isfinite(phi) else float(phi)),
        "residual_half_life_days": (None if half_life is None else float(half_life)),
        "dv01_neutral_ratio_IT_over_DE": dv01_ratio,
        "files": ["spread_ts.csv", "hedge_ratios.csv"] + (["curve_spreads.csv"] if curve_out else []) + ["stress_table.csv"],
    }
    (outdir / "spread_summary.json").write_text(json.dumps(summary, indent=2))

    # Console print
    print("== Bund–BTP Spread ==")
    print(f"Tenor: {args.tenor}  DE: {de_col}  IT: {it_col}")
    if summary["last_date"] is not None:
        print(f"Last date: {summary['last_date']}  Spread: {summary['last_spread_bps']:.1f} bps  z: {summary['last_zscore']:.2f}")
    if dv01_ratio is not None:
        print(f"DV01-neutral notional ratio (IT/DE): {dv01_ratio:.3f}")
    if half_life is not None:
        print(f"Residual AR(1) phi: {summary['residual_ar1_phi']:.3f}  Half-life: {half_life:.1f} days")
    print(f"Outputs in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
