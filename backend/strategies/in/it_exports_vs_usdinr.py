#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
it_exports_vs_usdinr.py — Indian IT exports vs USD/INR (rolling stats, lead–lag, IRFs & scenarios)
--------------------------------------------------------------------------------------------------

What this does
==============
Given Indian IT/Services exports (USD and/or INR) and USD/INR FX, this script:

1) Cleans & aligns all series to **monthly** frequency (quarterly values are forward-filled within quarter).
2) Builds core transforms:
   • Δlog(exports_USD) and/or Δlog(exports_INR)
   • Δlog(USDINR) (positive = INR depreciation)
3) Rolling diagnostics (short/med/long windows): correlations & betas of exports vs FX.
4) Lead–lag tables: Corr(Δlog FX_{t−k}, Δlog Exports_t) for k ∈ [−L..L].
5) Distributed-lag elasticities:
   Δlog(Exports)_t ~ Σ_{i=0..L} β_i·Δlog(USDINR)_{t−i} + controls_t  (HAC/Newey–West SE)
6) Local Projections (Jordà) IRFs: response of exports to a **+10% INR depreciation** shock.
7) Scenarios: +5% & +10% INR depreciation → implied export growth impact (using cumulative β).

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--exports exports.csv      REQUIRED
  Columns (any subset; extras ignored):
    date,
    it_exports_usd, services_exports_usd, exports_usd, it_services_usd, ...   (any USD measure)
    it_exports_inr, services_exports_inr, exports_inr, ...                    (any INR measure)
    category/segment/state (optional)
  NOTE: If both USD and INR exports exist, the USD measure is preferred for elasticities
        to avoid mechanical valuation effects.

--fx fx.csv                REQUIRED
  Columns:
    date, usdinr (INR per USD)   OR   inrusd/usd_inr_inv (USD per INR) → auto-inverted
  Optional controls (either here or in --macro):
    cpi_exporter, us_pmi_services, global_it_spend, dxy, oil_usd, etc. (any numeric column kept)

--macro macro.csv          OPTIONAL (monthly)
  Columns: date, <any numeric controls>, e.g., us_pmi_services, global_it_spend, world_trade, cpi_services

Key CLI parameters
------------------
--exp_col_usd   it_exports_usd     Pick a specific USD exports column (auto-detected if omitted)
--exp_col_inr   it_exports_inr     Pick a specific INR exports column (auto-detected if omitted)
--fx_col        usdinr             Pick FX column (auto-detected if omitted; auto inverts if needed)
--lags 12                         D-lag/lead–lag horizons (months)
--horizons 18                     IRF horizons (months)
--windows 6,12,24                 Rolling windows (months)
--start 2010-01-01  --end 2025-06-01
--outdir out_it_fx

Outputs
-------
- panel.csv               Monthly aligned panel (levels, Δlogs, z-scores)
- rolling_stats.csv       Rolling corr/beta for each exports measure × window
- leadlag_corr.csv        Lead–lag correlation table (per exports measure)
- dlag_regression.csv     Distributed-lag regression (coef, se, t) for each lag (per measure)
- lp_irf.csv              IRF: % response per **+10%** INR depreciation (coef, se, t by horizon)
- scenario_fx_shocks.csv  Impact of +5%/+10% depreciation using cumulative d-lag β
- summary.json            Headline diagnostics (date range, best lead/lag, scenario impacts)
- config.json             Run configuration

DISCLAIMER
----------
Research tooling; validate before policy, hedging, or trading decisions.
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
    """Return first matching column by exact/lower/contains."""
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        if cand.lower() in low: return low[cand.lower()]
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
    """Resample to month-end; forward-fill up to 2 months (for quarterly)."""
    out = df.set_index("date").resample("M").agg(how)
    if ffill_q:
        out = out.fillna(method="ffill", limit=2)
    return out.reset_index()

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(6, window//3)).corr(y)

def roll_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    minp = max(6, window//3)
    x_ = x.astype(float); y_ = y.astype(float)
    mx = x_.rolling(window, min_periods=minp).mean()
    my = y_.rolling(window, min_periods=minp).mean()
    cov = (x_*y_).rolling(window, min_periods=minp).mean() - mx*my
    varx = (x_*x_).rolling(window, min_periods=minp).mean() - mx*mx
    return cov / varx.replace(0, np.nan)

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

def ols_beta_se(X: np.ndarray, y: np.ndarray):
    XtX = X.T @ X
    XtY = X.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ XtY
    resid = y - X @ beta
    return beta, resid, XtX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
    """Newey–West HAC with Bartlett kernel."""
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1)+1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return se


# ----------------------------- loaders -----------------------------

def load_exports(path: str, exp_col_usd_hint: str="", exp_col_inr_hint: str="") -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    df = pd.read_csv(path)
    dt = (ncol(df, "date") or df.columns[0])
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])

    # Candidate USD / INR export columns
    usd_c = (ncol(df, exp_col_usd_hint) if exp_col_usd_hint else
             ncol(df, "it_exports_usd","services_exports_usd","exports_usd","it_services_usd","it_services_exports_usd","itusd"))
    inr_c = (ncol(df, exp_col_inr_hint) if exp_col_inr_hint else
             ncol(df, "it_exports_inr","services_exports_inr","exports_inr","it_services_inr","itinr"))

    # Keep both if present
    keep = ["date"]
    if usd_c:
        df[usd_c] = safe_num(df[usd_c]); keep.append(usd_c)
    if inr_c:
        df[inr_c] = safe_num(df[inr_c]); keep.append(inr_c)

    if len(keep)==1:
        raise ValueError("exports.csv must contain a USD and/or INR exports column.")

    out = resample_monthly(df[keep], how="mean", ffill_q=True)
    return out, (usd_c if usd_c else None), (inr_c if inr_c else None)

def load_fx(path: str, fx_col_hint: str="") -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    dt = (ncol(df, "date") or df.columns[0])
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])

    c = ncol(df, fx_col_hint) if fx_col_hint else (ncol(df, "usdinr","usd_inr","inr_per_usd","fx_usdinr","rate"))
    if not c:
        # maybe provided as USD per INR
        c = ncol(df, "inrusd","usdperinr","inr_usd","inv_usdinr")
    if not c:
        raise ValueError("fx.csv must contain a USD/INR column (usdinr or inverse).")

    series = safe_num(df[c])
    # Detect if inverse (USD per INR) by value range (< 1.0 typical)
    med = np.nanmedian(series.values.astype(float))
    if med < 1.0:
        usdinr = 1.0 / series
    else:
        usdinr = series
    out = pd.DataFrame({"date": df["date"], "usdinr": usdinr})
    out = resample_monthly(out, how="mean", ffill_q=True)
    return out, "usdinr"

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = (ncol(df, "date") or df.columns[0])
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])
    return resample_monthly(df, how="mean", ffill_q=True)


# ----------------------------- core construction -----------------------------

def build_panel(EX: pd.DataFrame, FX: pd.DataFrame, MAC: pd.DataFrame,
                usd_col: Optional[str], inr_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = EX.merge(FX, on="date", how="inner")
    if not MAC.empty:
        df = df.merge(MAC, on="date", how="left")

    # transforms
    df["dlog_fx"] = dlog(df["usdinr"])  # +ve = INR depreciation
    measures = []
    if usd_col:
        df = df.rename(columns={usd_col:"exports_usd"})
        df["dlog_exports_usd"] = dlog(df["exports_usd"])
        measures.append("exports_usd")
    if inr_col:
        df = df.rename(columns={inr_col:"exports_inr"})
        df["dlog_exports_inr"] = dlog(df["exports_inr"])
        measures.append("exports_inr")

    # z-scores (long window = 24m)
    for m in ["dlog_fx"] + [f"dlog_{m}" for m in ["exports_usd","exports_inr"] if f"dlog_{m}" in df.columns]:
        df[f"{m}_z"] = df[m].rolling(24, min_periods=8).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x, ddof=0) or np.nan))

    df = df.sort_values("date").dropna(subset=["dlog_fx"], how="all")
    return df, measures

def rolling_stats(panel: pd.DataFrame, measures: List[str], windows: List[int]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    for m in measures:
        y = idx[f"dlog_{m}"] if f"dlog_{m}" in idx.columns else None
        if y is None: continue
        x = idx["dlog_fx"]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({
                "measure": m, "window": w, "tag": tag,
                "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan,
                "beta": float(roll_beta(y, x, w).iloc[-1]) if len(idx)>=w else np.nan
            })
    return pd.DataFrame(rows)

def leadlag(panel: pd.DataFrame, measures: List[str], lags: int) -> pd.DataFrame:
    out = []
    for m in measures:
        y = panel[f"dlog_{m}"]
        x = panel["dlog_fx"]
        tab = leadlag_corr_table(x, y, lags)
        tab["measure"] = m
        out.append(tab)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["lag","corr","measure"])

def distributed_lag(panel: pd.DataFrame, measures: List[str], L: int, use_controls: bool=True) -> pd.DataFrame:
    rows = []
    df = panel.copy()
    # controls: any numeric controls (contemporaneous)
    control_cols = []
    if use_controls:
        for c in df.columns:
            if c in ["date","usdinr","dlog_fx"] or c.startswith("dlog_exports") or c.endswith("_z"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                control_cols.append(c)
    for m in measures:
        dep = f"dlog_{m}"
        if dep not in df.columns: continue
        # lags of dlog_fx
        Xparts = [pd.Series(1.0, index=df.index, name="const")]
        for l in range(0, L+1):
            Xparts.append(df["dlog_fx"].shift(l).rename(f"dlog_fx_l{l}"))
        if use_controls and control_cols:
            for c in control_cols:
                Xparts.append(df[c].rename(f"{c}_t"))
        X = pd.concat(Xparts, axis=1)
        XY = pd.concat([df[dep].rename("dep"), X], axis=1).dropna()
        if XY.shape[0] < max(36, 5*X.shape[1]):  # minimal sample
            continue
        yv = XY["dep"].values.reshape(-1,1)
        Xv = XY.drop(columns=["dep"]).values
        beta, resid, XtX_inv = ols_beta_se(Xv, yv)
        se = hac_se(Xv, resid, XtX_inv, L=max(6, L))
        names = XY.drop(columns=["dep"]).columns.tolist()
        for i, nm in enumerate(names):
            rows.append({"measure": m, "var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                         "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "lags": L})
        # cumulative effect of FX (sum of lags)
        idx = [i for i,nm in enumerate(names) if nm.startswith("dlog_fx_l")]
        cum_b = float(beta[idx,0].sum())
        cum_se = float(np.sqrt(np.sum(se[idx]**2)))  # rough
        rows.append({"measure": m, "var": "dlog_fx_cumulative_0..L", "coef": cum_b, "se": cum_se,
                     "t_stat": cum_b/(cum_se if cum_se>0 else np.nan), "lags": L})
    return pd.DataFrame(rows)

def local_projections(panel: pd.DataFrame, measures: List[str], horizons: int, L: int=12) -> pd.DataFrame:
    """
    IRF via Jordà LP: dep_h = log(y_{t+h}) − log(y_{t−1})  on shock10_t and lags/controls.
    shock10_t = Δlog FX / 0.10  → coefficients ×100 = % response to a **+10%** INR depreciation.
    """
    df = panel.copy().set_index("date")
    df["shock10"] = df["dlog_fx"] / 0.10
    # contemporaneous controls (optional): none by default to keep simple; you can augment here.
    results = []
    for m in measures:
        if f"dlog_{m}" not in df.columns or f"log_{m}" in df.columns:
            df[f"log_{m}"] = np.log(df[m].replace(0, np.nan)) if m in df.columns else np.nan
        ylog = df[f"log_{m}"]
        dlog_y = df[f"dlog_{m}"]
        for h in range(0, horizons+1):
            dep = ylog.shift(-h) - ylog.shift(-1)
            Xparts = [pd.Series(1.0, index=df.index, name="const"),
                      df["shock10"].rename("shock10_t")]
            for l in range(1, min(L, 12)+1):
                Xparts.append(df["shock10"].shift(l).rename(f"shock10_l{l}"))
                Xparts.append(dlog_y.shift(l).rename(f"dlog_{m}_l{l}"))
            X = pd.concat(Xparts, axis=1)
            XY = pd.concat([dep.rename("dep"), X], axis=1).dropna()
            if XY.shape[0] < max(36, 5*X.shape[1]):  # ensure enough obs
                continue
            yv = XY["dep"].values.reshape(-1,1)
            Xv = XY.drop(columns=["dep"]).values
            beta, resid, XtX_inv = ols_beta_se(Xv, yv)
            se = hac_se(Xv, resid, XtX_inv, L=max(6, h))
            names = XY.drop(columns=["dep"]).columns.tolist()
            try:
                i = names.index("shock10_t")
                b = float(beta[i,0]); s = float(se[i])
                results.append({"measure": m, "h": h,
                                "coef_pct_per_+10pct_fx": b*100.0,
                                "se_pct": s*100.0,
                                "t_stat": (b/s if s>0 else np.nan)})
            except ValueError:
                continue
    return pd.DataFrame(results).sort_values(["measure","h"])


# ----------------------------- scenarios -----------------------------

def scenario_impacts(dlag: pd.DataFrame, shocks: List[float]=[0.05, 0.10]) -> pd.DataFrame:
    """
    Use cumulative d-lag beta to map FX shocks (e.g., +5%, +10% INR depreciation)
    to export growth (%) = (cum_beta * shock * 100)
    """
    rows = []
    if dlag.empty: return pd.DataFrame(columns=["measure","shock_pct","impact_pct"])
    for m, g in dlag.groupby("measure"):
        cumrow = g[g["var"]=="dlog_fx_cumulative_0..L"]
        if cumrow.empty: continue
        b = float(cumrow["coef"].iloc[0])
        se = float(cumrow["se"].iloc[0]) if cumrow["se"].notna().all() else np.nan
        for sh in shocks:
            rows.append({"measure": m, "shock_pct": sh*100.0, "impact_pct": b*sh*100.0,
                         "cum_beta": b, "cum_se": se})
    return pd.DataFrame(rows)


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    exports: str
    fx: str
    macro: Optional[str]
    exp_col_usd: Optional[str]
    exp_col_inr: Optional[str]
    fx_col: Optional[str]
    lags: int
    horizons: int
    windows: str
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Indian IT exports vs USD/INR analytics")
    ap.add_argument("--exports", required=True)
    ap.add_argument("--fx", required=True)
    ap.add_argument("--macro", default="")
    ap.add_argument("--exp_col_usd", default="", help="Column name for USD exports in exports.csv")
    ap.add_argument("--exp_col_inr", default="", help="Column name for INR exports in exports.csv")
    ap.add_argument("--fx_col", default="", help="Column name for USD/INR rate in fx.csv (auto-detect if omitted)")
    ap.add_argument("--lags", type=int, default=12, help="Distributed-lag/lead–lag horizons (months)")
    ap.add_argument("--horizons", type=int, default=18, help="IRF horizons (months)")
    ap.add_argument("--windows", default="6,12,24", help="Rolling windows (months), comma-separated")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_it_fx")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    EX, exp_usd, exp_inr = load_exports(args.exports, args.exp_col_usd or "", args.exp_col_inr or "")
    FX, fx_col = load_fx(args.fx, args.fx_col or "")
    MAC = load_macro(args.macro) if args.macro else pd.DataFrame()

    # Date filters (before merge to reduce memory)
    if args.start:
        for df in [EX, FX, MAC]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [EX, FX, MAC]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    PANEL, measures = build_panel(EX, FX, MAC, exp_usd, exp_inr)
    if not measures:
        raise ValueError("No exports measure found after alignment.")
    if PANEL.shape[0] < 24:
        raise ValueError("Insufficient overlapping months after alignment (need ≥24).")

    # Persist panel
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Rolling stats
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, measures, windows)
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag
    LL = leadlag(PANEL, measures, lags=int(args.lags))
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Distributed-lag regression
    DLR = distributed_lag(PANEL, measures, L=int(args.lags), use_controls=True)
    if not DLR.empty: DLR.to_csv(outdir / "dlag_regression.csv", index=False)

    # Local projections (IRF)
    IRF = local_projections(PANEL, measures, horizons=int(args.horizons), L=int(args.lags))
    if not IRF.empty: IRF.to_csv(outdir / "lp_irf.csv", index=False)

    # Scenarios
    SCEN = scenario_impacts(DLR, shocks=[0.05, 0.10])
    if not SCEN.empty: SCEN.to_csv(outdir / "scenario_fx_shocks.csv", index=False)

    # Best lead/lag per measure
    best_ll = {}
    if not LL.empty:
        for m, g in LL.dropna(subset=["corr"]).groupby("measure"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[m] = {"lag_months": int(row["lag"]), "corr": float(row["corr"])}

    # Summary
    latest = PANEL.tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "fx_col": fx_col,
        "export_measures": measures,
        "latest": {
            "date": str(latest["date"].date()),
            "usdinr": float(latest["usdinr"]),
            "dlog_fx_pct": float(latest["dlog_fx"]*100) if pd.notna(latest["dlog_fx"]) else None,
            **{f"dlog_{m}_pct": (float(latest[f"dlog_{m}"])*100 if f"dlog_{m}" in PANEL.columns and pd.notna(latest[f"dlog_{m}"]) else None)
               for m in measures}
        },
        "leadlag_best": best_ll,
        "scenarios": SCEN.to_dict(orient="records") if not SCEN.empty else [],
        "notes": "IRF coefficients are % response to a +10% INR depreciation (USD/INR ↑10%)."
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        exports=args.exports, fx=args.fx, macro=(args.macro or None),
        exp_col_usd=(args.exp_col_usd or None), exp_col_inr=(args.exp_col_inr or None),
        fx_col=(args.fx_col or None),
        lags=int(args.lags), horizons=int(args.horizons), windows=args.windows,
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== IT Exports vs USD/INR ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Measures: {', '.join(measures)}")
    if best_ll:
        for m, st in best_ll.items():
            print(f"Lead–lag ({m}): max |corr| at lag {st['lag_months']:+d} → {st['corr']:+.2f}")
    if not SCEN.empty:
        for _, r in SCEN.iterrows():
            print(f"Scenario {r['shock_pct']:.0f}% INR depreciation → {r['measure']}: {r['impact_pct']:+.2f}%")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
