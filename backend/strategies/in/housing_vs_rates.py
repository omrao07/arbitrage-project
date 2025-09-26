#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
housing_vs_rates.py — Housing activity vs interest rates (rolling stats, lead–lag, IRFs)
----------------------------------------------------------------------------------------

What this does
==============
Given housing time series (prices, sales, starts, permits, inventory, months’ supply, etc.)
and one or more interest-rate series (mortgage rate, policy rate, 10y yield), this script:

1) Cleans & aligns all series to **monthly** frequency (quarterly data are forward-filled within quarter).
2) Builds core transforms:
   • Δlog for housing variables (approx % m/m) and ΔRate in **basis points**.
   • Z-scores and rolling correlations/betas (short/med/long).
3) Lead–lag tables: Corr(ΔRate_{t−k}, Δlog(Housing)_t) for lags −L..+L.
4) Local Projections (Jordà) impulse responses:
   • For each housing var and horizon h=0..H, regress cumulative log change on a **100 bp rate shock** at t
     with lags of shocks and past Δlog terms → IRF in % per +100 bp.
   • Newey–West (HAC) standard errors with lag=max(h, 6).
5) (Optional) Mortgage affordability index if income & price are supplied.

Inputs (CSV; headers are flexible)
----------------------------------
--housing housing.csv        REQUIRED
  Columns (any subset; case-insensitive):
    date, price_index, sales, starts, permits, inventory, months_supply, income (optional), region (optional)
  You may use your own names; the script guesses sensibly or you can pass --vars.

--rates rates.csv            REQUIRED
  Columns: date, <rate columns...> (e.g., mortgage_rate, policy_rate, yield_10y)
  Use --rate_col to pick the rate used for “ΔRate (bps)”. Others are kept as controls if desired.

--macro macro.csv            OPTIONAL (additional numeric controls, e.g., CPI, unemployment)
  Columns: date, <numeric...> [, region]

Key CLI parameters
------------------
--vars price_index,sales,starts   Which housing vars to analyze (auto-detected if omitted)
--rate_col mortgage_rate          Which column in rates.csv to use as the “shock” rate (auto-pick if omitted)
--use_controls                    If set, include available controls (other rates & macro) in IRFs
--lags 12                         Lags of ΔRate and Δlog(y) in local projections
--horizons 24                     IRF horizons (months)
--windows 6,12,24                 Rolling windows (months) for stats
--region "West"                   Filter to a region (if region present)
--start 2012-01-01  --end 2025-06-01
--outdir out_housing_rates

Outputs
-------
- panel.csv               Monthly aligned panel (levels, Δlogs, ΔRate_bps, z-scores)
- rolling_stats.csv       Rolling corr/beta for each var × window (short/med/long)
- leadlag_corr.csv        Lead–lag correlation table (var × lag)
- lp_irf.csv              Local projection IRFs: response % per +100 bp shock (coef, se, t)
- affordability.csv       (If price & income & mortgage_rate provided) simple index over time
- summary.json            Headline diagnostics (latest stats, strongest leads/lags)
- config.json             Run configuration for reproducibility

Notes
-----
• ΔRate is computed in basis points with auto scale detection:
    if rate median > 1.5 → treat as percent; else decimal → Δ×10000.
• IRF dependent variable at horizon h is cumulative log change: (log y_{t+h} − log y_{t−1}),
  so coefficients are approximate % responses to a **100 bp** shock at t.
• This is research tooling; validate before relying on output.

DISCLAIMER
----------
Not investment advice. No guarantee of stability or causality.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return the first matching column by exact/lower/contains."""
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        if cand.lower() in low: return low[cand.lower()]
    # fuzzy contains
    for cand in cands:
        t = cand.lower()
        for c in df.columns:
            if t in str(c).lower(): return c
    return None

def to_month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def resample_monthly(df: pd.DataFrame, how: str="mean", ffill_q: bool=True) -> pd.DataFrame:
    """Resample to monthly end; forward-fill up to 2 months for quarterly-like series."""
    agg = "mean" if how=="mean" else "sum"
    out = df.set_index("date").resample("M").agg(agg)
    if ffill_q:
        out = out.fillna(method="ffill", limit=2)
    out = out.reset_index()
    return out

def deduce_rate_scale(rate: pd.Series) -> int:
    """Return 100 if series is already in %, else 10000 for decimals."""
    med = np.nanmedian(rate.values.astype(float))
    return 100 if med > 1.5 else 10000

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(6, window//3)).corr(y)

def roll_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta for y ~ a + b x (one regressor)."""
    minp = max(6, window//3)
    x_ = x.astype(float); y_ = y.astype(float)
    mx = x_.rolling(window, min_periods=minp).mean()
    my = y_.rolling(window, min_periods=minp).mean()
    cov = (x_*y_).rolling(window, min_periods=minp).mean() - mx*my
    varx = (x_*x_).rolling(window, min_periods=minp).mean() - mx*mx
    return cov / varx.replace(0, np.nan)

def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(6, window//3)).mean()
    sd = s.rolling(window, min_periods=max(6, window//3)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)

def leadlag_corr_table(flow: pd.Series, ret: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            f = flow.shift(k); r = ret
        else:
            f = flow; r = ret.shift(-k)
        c = f.corr(r)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

def ols_beta_se(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with (X'X)^-1 X'y; return (beta, residuals, XtX_inv)."""
    XtX = X.T @ X
    XtY = X.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ XtY
    resid = y - X @ beta
    return beta, resid, XtX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
    """
    Newey–West HAC (Bartlett kernel) for OLS.
    X: (n×k), resid: (n×1)
    """
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = np.zeros((k, k))
    # S_0
    S += (X * u).T @ (X * u)
    # Lags
    for l in range(1, min(L, n-1)+1):
        w = 1.0 - l/(L+1)
        Gamma_l = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (Gamma_l + Gamma_l.T)
    # Cov(beta) = (X'X)^-1 S (X'X)^-1
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return se

# ----------------------------- loaders -----------------------------

def load_housing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    # standardize common names
    ren = {}
    for raw, std in [
        ("price_index","price_index"), ("prices","price_index"), ("house_price","price_index"),
        ("sales","sales"), ("home_sales","sales"), ("transactions","sales"),
        ("starts","starts"), ("housing_starts","starts"),
        ("permits","permits"),
        ("inventory","inventory"), ("inventories","inventory"),
        ("months_supply","months_supply"), ("mths_supply","months_supply"),
        ("income","income"), ("avg_income","income"), ("income_index","income"),
        ("region","region")
    ]:
        c = ncol(df, raw)
        if c: ren[c] = std
    df = df.rename(columns=ren)
    # keep numeric except date/region
    for c in df.columns:
        if c not in ["date","region"]:
            df[c] = safe_num(df[c])
    return resample_monthly(df, how="mean", ffill_q=True)

def load_rates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    # numericify non-date
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])
    return resample_monthly(df, how="mean", ffill_q=True)

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    for c in df.columns:
        if c not in ["date","region"]:
            df[c] = safe_num(df[c])
    return resample_monthly(df, how="mean", ffill_q=True)


# ----------------------------- core calculations -----------------------------

def build_panel(H: pd.DataFrame, R: pd.DataFrame, M: pd.DataFrame,
                vars_keep: List[str], rate_col: str, region: Optional[str]) -> Tuple[pd.DataFrame, List[str], str, int]:
    """Merge housing + rates (+macro), compute transforms."""
    df = H.copy()
    if region and "region" in df.columns:
        df = df[df["region"].astype(str) == region]
    df = df.drop(columns=[c for c in df.columns if c not in (["date","region"] + vars_keep)], errors="ignore")

    # attach rate(s)
    rates = R.copy()
    r_candidates = [c for c in rates.columns if c != "date"]
    if not r_candidates:
        raise ValueError("rates.csv must include at least one numeric rate column.")
    if not rate_col:
        # heuristic pick
        rc = (ncol(rates, "mortgage_rate", "mortgage", "home_loan", "housing_rate")
              or ncol(rates, "policy_rate", "repo", "ffr")
              or ncol(rates, "yield_10y", "10y", "long_yield")
              or r_candidates[0])
        rate_col = rc
    shock_rate = rate_col
    df = df.merge(rates, on="date", how="left", suffixes=("",""))

    # attach macro (if any)
    if not M.empty:
        on_cols = ["date"]
        if region and "region" in M.columns:
            M = M[M["region"].astype(str)==region]
        df = df.merge(M, on=on_cols, how="left")

    # keep cols
    keep = ["date","region"] if "region" in df.columns else ["date"]
    keep += vars_keep
    keep += [c for c in r_candidates if c not in keep]
    if not M.empty:
        keep += [c for c in M.columns if c not in keep]
    df = df[keep].sort_values("date").drop_duplicates(subset=["date"])

    # Δlog housing, Δrate in bps
    scale = deduce_rate_scale(df[shock_rate])
    df["d_rate_bps"] = df[shock_rate].diff() * (100 if scale==100 else 10000)

    for v in vars_keep:
        df[f"log_{v}"] = np.log(df[v].replace(0, np.nan))
        df[f"dlog_{v}"] = df[f"log_{v}"].diff()

    # z-scores (long window = 24m)
    for v in vars_keep:
        df[f"dlog_{v}_z"] = zscore(df[f"dlog_{v}"], window=24)
    df["d_rate_bps_z"] = zscore(df["d_rate_bps"], window=24)

    df = df.dropna(subset=["d_rate_bps"] + [f"dlog_{v}" for v in vars_keep], how="all")
    return df, vars_keep, shock_rate, scale

def rolling_stats(panel: pd.DataFrame, vars_keep: List[str], windows: List[int]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    for v in vars_keep:
        y = idx[f"dlog_{v}"]
        x = idx["d_rate_bps"]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append(pd.DataFrame({
                "date": idx.index,
                "var": v,
                f"corr_{tag}": roll_corr(x, y, w).values,
                f"beta_{tag}": roll_beta(y, x, w).values
            }).set_index("date"))
    out = (pd.concat(rows, axis=1)
             .groupby(level=0, axis=1)
             .first()
             .reset_index()
             .sort_values("date"))
    return out

def leadlag(panel: pd.DataFrame, vars_keep: List[str], lags: int) -> pd.DataFrame:
    idx = panel.set_index("date")
    out = []
    for v in vars_keep:
        tab = leadlag_corr_table(idx["d_rate_bps"], idx[f"dlog_{v}"], lags)
        tab["var"] = v
        out.append(tab)
    return pd.concat(out, ignore_index=True)

def local_projections(panel: pd.DataFrame, vars_keep: List[str], horizons: int,
                      lags: int, use_controls: bool, shock_rate: str) -> pd.DataFrame:
    """
    IRF via Jordà LP: dep_h = log(y_{t+h}) − log(y_{t−1})  on shock100_t and controls/lags.
    shock100_t = Δrate_bps / 100  (so coef is % response per +100 bp)
    """
    df = panel.copy().set_index("date")
    df["shock100"] = df["d_rate_bps"] / 100.0

    # candidate controls: other rates & macro Δlogs
    control_cols = []
    if use_controls:
        for c in df.columns:
            if c in ["shock100","d_rate_bps","region"] or c.startswith("log_") or c.startswith("dlog_"):
                continue
            if c == shock_rate or c == "date":
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                control_cols.append(c)
    # build Δlog controls for macro price-like series (e.g., CPI, income if level given)
    for c in list(control_cols):
        if c.startswith("d_") or c.startswith("Δ"):
            continue
        # if it varies over time and is positive, create dlog
        s = df[c]
        if s.notna().sum() > 12 and (s > 0).mean() > 0.95:
            df[f"dlog_{c}"] = dlog(s)
            control_cols.append(f"dlog_{c}")

    # also include lags of dependent Δlog and shock
    results = []
    for v in vars_keep:
        ylog = df[f"log_{v}"]
        dlog_y = df[f"dlog_{v}"]
        # regressors basic: constant, shock100_t, lags of shock and dlog_y
        for h in range(0, horizons+1):
            dep = ylog.shift(-h) - ylog.shift(-1)  # cumulative change from t-1 to t+h
            Xparts = [pd.Series(1.0, index=df.index, name="const"),
                      df["shock100"].rename("shock100_t")]
            # lags of shock and Δlog(y)
            for L in range(1, lags+1):
                Xparts.append(df["shock100"].shift(L).rename(f"shock100_l{L}"))
                Xparts.append(dlog_y.shift(L).rename(f"dlog_{v}_l{L}"))
            # controls contemporaneous (t) and optionally their lags (keep simple: only t)
            if use_controls:
                for c in control_cols:
                    if f"dlog_{c}" in df.columns:  # already Δlog
                        Xparts.append(df[f"dlog_{c}"].rename(f"{c}_t"))
                    elif c not in [shock_rate]:
                        Xparts.append(df[c].rename(f"{c}_t"))
            X = pd.concat(Xparts, axis=1)
            XY = pd.concat([dep.rename("dep"), X], axis=1).dropna()
            if XY.shape[0] < max(24, 5*len(X.columns)):
                continue
            yv = XY["dep"].values.reshape(-1,1)
            Xv = XY.drop(columns=["dep"]).values
            beta, resid, XtX_inv = ols_beta_se(Xv, yv)
            # HAC lag: max(h,6)
            se = hac_se(Xv, resid, XtX_inv, L=max(h,6))
            # extract coefficient on shock100_t (contemporaneous)
            names = XY.drop(columns=["dep"]).columns.tolist()
            try:
                i = names.index("shock100_t")
                b = float(beta[i,0]); s = float(se[i]); tstat = b / s if s>0 else np.nan
                results.append({"var": v, "h": h, "coef_pct_per_100bp": b*100.0,  # convert log change to %
                                "se_pct": s*100.0, "t_stat": tstat})
            except ValueError:
                continue
    return pd.DataFrame(results).sort_values(["var","h"])

def affordability_table(panel: pd.DataFrame, price_col: str, mort_rate_col: str,
                        income_col: str="income") -> pd.DataFrame:
    """
    Simple affordability index: (monthly income) / (mortgage P&I on price index).
    Assumptions:
      - Price_index is normalized (any units): we compute payment on normalized principal.
      - 30-year fixed; if rate in %, convert to decimal.
      - Monthly income = income / 12 (if income is an index, index-based ratio is still informative).
    """
    df = panel.copy()
    if price_col not in df.columns or mort_rate_col not in df.columns or income_col not in df.columns:
        return pd.DataFrame()
    rate = safe_num(df[mort_rate_col])
    scale = deduce_rate_scale(rate)
    r_m = (rate / (100.0 if scale==100 else 1.0)) / 12.0  # monthly decimal
    # Avoid zero/negatives
    r_m = r_m.replace(0, np.nan)
    N = 360.0
    # normalized principal: use price_index; if missing, skip
    P = safe_num(df[price_col])
    # Monthly payment factor
    fact = r_m * (1 + r_m)**N / ((1 + r_m)**N - 1)
    pay = P * fact
    inc_m = safe_num(df[income_col]) / 12.0
    aff = inc_m / pay.replace(0, np.nan)
    out = pd.DataFrame({"date": df["date"], "affordability_index": aff})
    return out.dropna()

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    housing: str
    rates: str
    macro: Optional[str]
    vars: Optional[str]
    rate_col: Optional[str]
    use_controls: bool
    lags: int
    horizons: int
    windows: str
    region: Optional[str]
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Housing vs Rates analytics")
    ap.add_argument("--housing", required=True)
    ap.add_argument("--rates", required=True)
    ap.add_argument("--macro", default="")
    ap.add_argument("--vars", default="", help="Comma list of housing vars to analyze (e.g., price_index,sales,starts)")
    ap.add_argument("--rate_col", default="", help="Column in rates.csv to use for ΔRate (bps), e.g., mortgage_rate")
    ap.add_argument("--use_controls", action="store_true", help="Include other rates/macro controls in IRF")
    ap.add_argument("--lags", type=int, default=12, help="LP lags of shock and Δlog(y)")
    ap.add_argument("--horizons", type=int, default=24, help="IRF horizons (months)")
    ap.add_argument("--windows", default="6,12,24", help="Rolling windows (months), comma-separated")
    ap.add_argument("--region", default="", help="Filter to region (if present)")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_housing_rates")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    H = load_housing(args.housing)
    R = load_rates(args.rates)
    M = load_macro(args.macro) if args.macro else pd.DataFrame()

    # Region & date filters
    region = args.region or None
    if args.start:
        for df in [H, R, M]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [H, R, M]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    # Determine vars
    auto_vars = [c for c in H.columns if c not in ["date","region"]]
    # Prefer common housing names
    preferred = [c for c in ["price_index","sales","starts","permits","inventory","months_supply","income"] if c in auto_vars]
    vars_keep = [v.strip() for v in args.vars.split(",") if v.strip()] if args.vars else preferred or auto_vars
    # Exclude 'income' from housing vars (kept for affordability)
    vars_keep = [v for v in vars_keep if v != "income"]

    panel, vars_used, shock_rate, rate_scale = build_panel(H, R, M, vars_keep, args.rate_col or "", region)
    if panel.shape[0] < 36:
        raise ValueError("Insufficient overlapping months after alignment (need ≥36).")

    # Rolling stats
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    roll = rolling_stats(panel, vars_used, windows)
    roll.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag
    ll = leadlag(panel, vars_used, lags=int(args.lags))
    ll.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Local projections IRFs
    irf = local_projections(panel, vars_used, horizons=int(args.horizons),
                            lags=int(args.lags), use_controls=bool(args.use_controls),
                            shock_rate=shock_rate)
    if not irf.empty:
        irf.to_csv(outdir / "lp_irf.csv", index=False)

    # Affordability (optional)
    aff = pd.DataFrame()
    mort_rate_guess = (ncol(panel, "mortgage_rate") or ncol(panel, "home_loan") or shock_rate)
    if "price_index" in panel.columns and "income" in panel.columns and mort_rate_guess in panel.columns:
        aff = affordability_table(panel[["date","price_index","income", mort_rate_guess]].copy(),
                                  price_col="price_index", mort_rate_col=mort_rate_guess, income_col="income")
        if not aff.empty:
            aff.to_csv(outdir / "affordability.csv", index=False)

    # Persist main panel
    panel.to_csv(outdir / "panel.csv", index=False)

    # Summary
    latest = panel.tail(1)
    # strongest lead/lag for each var
    best_ll = {}
    if not ll.empty:
        for v, g in ll.groupby("var"):
            g2 = g.dropna(subset=["corr"])
            if g2.empty: continue
            i = g2["corr"].abs().idxmax()
            row = g2.loc[i]
            best_ll[v] = {"lag": int(row["lag"]), "corr": float(row["corr"])}
    summary = {
        "rows": int(panel.shape[0]),
        "date_range": {"start": str(panel["date"].min().date()), "end": str(panel["date"].max().date())},
        "vars": vars_used,
        "shock_rate": shock_rate,
        "rate_scale": ("percent" if rate_scale==100 else "decimal"),
        "latest": {
            "date": (str(latest["date"].iloc[0].date()) if not latest.empty else None),
            "d_rate_bps": (float(latest["d_rate_bps"].iloc[0]) if not latest.empty else None),
            **{f"dlog_{v}_pct": (float(latest[f"dlog_{v}"].iloc[0])*100 if (f"dlog_{v}" in latest.columns) else None)
               for v in vars_used}
        },
        "leadlag_best": best_ll,
        "irf_note": "IRF coefficients are % response to a +100 bp rate shock at t; HAC SE with lag=max(h,6).",
        "affordability_available": (not aff.empty)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        housing=args.housing, rates=args.rates, macro=(args.macro or None),
        vars=(args.vars or None), rate_col=(args.rate_col or None),
        use_controls=bool(args.use_controls), lags=int(args.lags),
        horizons=int(args.horizons), windows=args.windows, region=(args.region or None),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Housing vs Rates ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Vars: {', '.join(vars_used)}")
    if not irf.empty:
        for v in vars_used[:3]:
            sub = irf[irf["var"]==v]
            if not sub.empty:
                h12 = sub[sub["h"]==12]
                if not h12.empty:
                    print(f"IRF {v}: 12m response to +100bp = {h12['coef_pct_per_100bp'].iloc[0]:+.2f}% "
                          f"(t={h12['t_stat'].iloc[0]:+.2f})")
    if not aff.empty:
        print("Affordability index computed (see affordability.csv).")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
