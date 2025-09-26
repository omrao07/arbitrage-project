#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
india_vix_vs_skew.py — India VIX vs options skew (rolling stats, lead–lag, RR25)
---------------------------------------------------------------------------------

What this does
==============
This script relates **India VIX** (index-level implied volatility) to **index options
skew** computed either from a ready-made time series (e.g., RR25/Skew) or directly
from an options chain (per date × strike × expiry implied vols).

It produces:
1) A daily panel with VIX, several skew measures, and their changes
2) Rolling correlations/betas (short/med/long windows)
3) Lead–lag tables Corr(VIX_{t−k}, SKEW_t), k ∈ [−L..L]
4) Extreme regime flags (elevated VIX with steep put skew)
5) Summary JSON + tidy CSVs

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
Option A — prebuilt time series:
--timeseries ts.csv
  Columns (any subset; extras ignored):
    date, vix (or india_vix, nvix), skew (or rr25, risk_reversal_25, put_call_skew, slope)
  You can include multiple skew-like columns; the script will keep all.

Option B — raw option chain (NIFTY/BANKNIFTY index options):
--options options.csv
  Columns (required-ish; extras ignored):
    date, expiry, option_type (C/P), strike, iv, underlying[, risk_free]
  Notes:
    • iv accepted names: iv, implied_vol, sigma, iv_pct (auto %→decimal if needed)
    • option_type values: C/Call, P/Put (case-insensitive)
    • Time-to-expiry is computed from date→expiry (in years).
    • Forward F ≈ S·exp(r·T) if risk_free provided, else F:=S.
    • Near expiry bucket defaults to 7–45 days (override with --min_dte/--max_dte).

CLI examples
------------
# Using prebuilt skew
python india_vix_vs_skew.py --timeseries vix_skew.csv --windows 10,20,60 --lags 10 --outdir out_vix_skew

# Build skew from option chain
python india_vix_vs_skew.py --options option_chain.csv --vix_csv india_vix.csv --vix_col INDIA_VIX \
  --min_dte 7 --max_dte 45 --windows 10,20,60 --lags 10 --outdir out_vix_skew

Outputs
-------
- panel.csv               Per-date panel (VIX, skew measures, Δs, z-scores)
- rolling_stats.csv       Rolling corr/beta of VIX vs each skew (for each window)
- leadlag_corr.csv        Lead–lag table (per skew)
- regimes.csv             Extreme regime flags (e.g., HighVIX+SteepPutSkew)
- summary.json            Headline stats / latest snapshot
- config.json             Run configuration

Skew measures in this script
----------------------------
If built from options:
  • slope_lm: OLS slope of IV vs log-moneyness ln(K/F) for near-dated puts & calls
  • rr25: 25Δ Call IV − 25Δ Put IV (positive if calls richer than puts)
  • bf25: 0.5×(25Δ Call IV + 25Δ Put IV) − ATM50Δ IV (smile curvature)
  • atm50: IV at ≈50Δ (nearest to K≈F)

If provided in timeseries, we keep any columns whose names contain:
  "skew", "rr25", "risk_reversal", "put_call", "slope", "smile", "bf", "butterfly".

DISCLAIMER
----------
Research tooling. Approximations (e.g., forward, delta from IV) can differ from vendor methods.
Validate before trading or risk use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from math import log, sqrt, exp, erf


# ----------------------------- utils -----------------------------

def ensure_dir(path: str) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return first matching column by exact/lower/contains."""
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        lc = cand.lower()
        if lc in low: return low[lc]
    for cand in cands:
        t = cand.lower()
        for c in df.columns:
            if t in str(c).lower(): return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(5, window//2)).mean()
    sd = s.rolling(window, min_periods=max(5, window//2)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)

def d1(F: float, K: float, sigma: float, T: float) -> float:
    # Avoid div by zero
    if sigma is None or sigma <= 0 or T <= 0 or F<=0 or K<=0:
        return np.nan
    return (log(F/K) + 0.5*sigma*sigma*T)/(sigma*sqrt(T))

def ndist(x: float) -> float:
    # standard normal CDF using erf
    return 0.5*(1.0 + erf(x/sqrt(2.0)))

def call_delta(F: float, K: float, sigma: float, T: float) -> float:
    d_1 = d1(F, K, sigma, T)
    if d_1!=d_1: return np.nan
    return ndist(d_1)

def put_delta(F: float, K: float, sigma: float, T: float) -> float:
    cdel = call_delta(F, K, sigma, T)
    return cdel - 1.0 if cdel==cdel else np.nan

def iv_to_decimal(iv: pd.Series) -> pd.Series:
    """Detect if IV is in % and convert to decimal."""
    s = iv.dropna().astype(float)
    if s.empty: return iv
    med = np.nanmedian(s)
    if med > 1.0:  # looks like percent
        return iv.astype(float) / 100.0
    return iv.astype(float)

def leadlag_corr(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            xx = x.shift(k); yy = y
        else:
            xx = x; yy = y.shift(-k)
        c = xx.corr(yy)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(5, window//2)).corr(y)

def roll_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    minp = max(5, window//2)
    x_ = x.astype(float); y_ = y.astype(float)
    mx = x_.rolling(window, min_periods=minp).mean()
    my = y_.rolling(window, min_periods=minp).mean()
    cov = (x_*y_).rolling(window, min_periods=minp).mean() - mx*my
    varx = (x_*x_).rolling(window, min_periods=minp).mean() - mx*mx
    return cov / varx.replace(0, np.nan)


# ----------------------------- loaders -----------------------------

def load_timeseries(path: str, vix_col_hint: str="") -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    # VIX col
    vixc = ncol(df, vix_col_hint) if vix_col_hint else (ncol(df, "india_vix","vix","nvix","nifty_vix"))
    if not vixc:
        raise ValueError("timeseries file: cannot find VIX column (try --vix_col).")
    df = df.rename(columns={vixc:"vix"})
    df["vix"] = safe_num(df["vix"])
    # Keep any skew-like columns
    keep = ["date","vix"]
    skew_like = []
    for c in df.columns:
        cl = str(c).lower()
        if c in keep: continue
        if any(k in cl for k in ["skew","risk_reversal","rr25","put_call","slope","smile","bf","butterfly"]):
            df[c] = safe_num(df[c])
            keep.append(c); skew_like.append(c)
    if not skew_like:
        raise ValueError("timeseries file: no skew-like columns found (e.g., rr25/skew/slope).")
    out = df[keep].dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    return out

def load_vix_only(path: str, vix_col_hint: str="") -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    vixc = ncol(df, vix_col_hint) if vix_col_hint else (ncol(df, "india_vix","vix","nvix","nifty_vix"))
    if not vixc:
        raise ValueError("vix_csv: cannot find VIX column (try --vix_col).")
    df = df.rename(columns={vixc:"vix"})
    df["vix"] = safe_num(df["vix"])
    return df[["date","vix"]].dropna().sort_values("date")

def load_options(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # map cols
    dt  = ncol(df, "date", "trade_date", "timestamp")
    ex  = ncol(df, "expiry", "expiration", "expiry_date")
    typ = ncol(df, "option_type", "type", "cp", "call_put")
    k   = ncol(df, "strike", "strike_price", "k")
    ivc = ncol(df, "iv", "implied_vol", "sigma", "iv_pct")
    spot= ncol(df, "underlying", "spot", "price", "close")
    rf  = ncol(df, "risk_free", "r", "rate", "rf")
    if not all([dt, ex, typ, k, ivc, spot]):
        raise ValueError("options.csv must include date, expiry, option_type, strike, iv, underlying.")
    df = df.rename(columns={dt:"date", ex:"expiry", typ:"cp", k:"strike", ivc:"iv", spot:"spot"})
    if rf: df = df.rename(columns={rf:"r"})
    df["date"]   = to_date(df["date"])
    df["expiry"] = to_date(df["expiry"])
    df["cp"]     = df["cp"].astype(str).str.upper().str[0].map({"C":"C","P":"P"})
    df["strike"] = safe_num(df["strike"])
    df["iv"]     = iv_to_decimal(safe_num(df["iv"]))
    df["spot"]   = safe_num(df["spot"])
    if "r" in df.columns: df["r"] = safe_num(df["r"]).fillna(0.0)
    else: df["r"] = 0.0
    # DTE & T
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    df["T"]   = df["dte"].clip(lower=1).astype(float) / 365.0
    # forward approx
    df["F"] = df["spot"] * np.exp(df["r"] * df["T"])
    # moneyness & deltas
    df["log_moneyness"] = np.log(df["strike"] / df["F"].replace(0, np.nan))
    # row-wise delta using each row's iv
    def row_delta(row):
        if row["cp"]=="C":
            return call_delta(row["F"], row["strike"], row["iv"], row["T"])
        else:
            return put_delta(row["F"], row["strike"], row["iv"], row["T"])
    df["delta"] = df.apply(row_delta, axis=1)
    # clean
    df = df.dropna(subset=["date","cp","iv","strike","F","T"])
    return df.sort_values(["date","expiry","cp","strike"])

# ----------------------------- skew construction from options -----------------------------

def nearest_by_delta(group: pd.DataFrame, target_delta: float, side: str) -> Optional[float]:
    """
    Return IV at strike whose delta is closest to target (side='C' for calls: +delta, 'P' for puts: negative).
    target_delta should be positive e.g., 0.25; we map to sign for puts.
    """
    if group.empty or group["delta"].dropna().empty:
        return np.nan
    if side=="C":
        g = group[group["cp"]=="C"].copy()
        if g.empty: return np.nan
        g["d_err"] = (g["delta"] - target_delta).abs()
    else:
        g = group[group["cp"]=="P"].copy()
        if g.empty: return np.nan
        g["d_err"] = (g["delta"] - (-target_delta)).abs()
    row = g.loc[g["d_err"].idxmin()] if not g.empty else None
    return float(row["iv"]) if row is not None and pd.notna(row["iv"]) else np.nan

def iv_atm50(group: pd.DataFrame) -> float:
    """
    IV near 50Δ (ATM-forward). We find:
      • Call with delta closest to +0.5
      • Put with delta closest to −0.5
    Then average them (or take the available one).
    """
    c = nearest_by_delta(group, 0.50, "C")
    p = nearest_by_delta(group, 0.50, "P")
    vals = [v for v in [c,p] if v==v]
    return float(np.mean(vals)) if vals else np.nan

def smile_slope(group: pd.DataFrame) -> float:
    """
    OLS slope of IV vs log-moneyness ln(K/F) using both calls and puts (near-term).
    """
    dd = group.dropna(subset=["iv","log_moneyness"])
    if dd.shape[0] < 5: return np.nan
    X = dd["log_moneyness"].values
    Y = dd["iv"].values
    X = np.column_stack([np.ones_like(X), X])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ Y.reshape(-1,1))
    return float(beta[1,0])  # slope

def rr25_and_bf(group: pd.DataFrame) -> Tuple[float,float]:
    ivc = nearest_by_delta(group, 0.25, "C")
    ivp = nearest_by_delta(group, 0.25, "P")
    atm  = iv_atm50(group)
    rr25 = (ivc - ivp) if (ivc==ivc and ivp==ivp) else np.nan
    bf25 = (0.5*((ivc if ivc==ivc else 0)+(ivp if ivp==ivp else 0)) - atm) if atm==atm else np.nan
    return rr25, bf25

def build_skew_from_options(OPT: pd.DataFrame, min_dte: int, max_dte: int) -> pd.DataFrame:
    """
    Compute skew measures per DATE using near-term expiry bucket with DTE ∈ [min_dte, max_dte].
    If multiple expiries qualify on a date, pick the one with DTE closest to 30 days (or middle of the range).
    """
    # pick near-term expiry per date
    near = OPT[(OPT["dte"] >= min_dte) & (OPT["dte"] <= max_dte)].copy()
    if near.empty:
        raise ValueError("No options within the specified DTE bucket; adjust --min_dte/--max_dte.")
    # choose best expiry per date (closest to target_dte)
    target = (min_dte + max_dte)/2.0
    pick = (near.groupby(["date","expiry"])["dte"].mean().reset_index()
                 .assign(d_err=lambda x: (x["dte"]-target).abs())
                 .sort_values(["date","d_err"]))
    best = pick.groupby("date").head(1)[["date","expiry"]]
    sub = near.merge(best, on=["date","expiry"], how="inner")
    # compute metrics per date
    rows = []
    for d, G in sub.groupby("date"):
        slope = smile_slope(G)
        rr25, bf25 = rr25_and_bf(G)
        atm = iv_atm50(G)
        rows.append({"date": d, "slope_lm": slope, "rr25": rr25, "bf25": bf25, "atm50": atm})
    out = pd.DataFrame(rows).sort_values("date")
    return out

# ----------------------------- analytics -----------------------------

def enrich_panel(panel: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = panel.copy().sort_values("date")
    # changes
    df["dvix"] = df["vix"].diff()
    for c in df.columns:
        if c in ["date","vix","dvix"]: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[f"d_{c}"] = df[c].diff()
    # z-scores
    L = max(windows) if windows else 60
    df["vix_z"] = zscore(df["vix"], window=L)
    for c in df.columns:
        if c.startswith("rr") or "skew" in c.lower() or "slope" in c.lower():
            df[f"{c}_z"] = zscore(df[c], window=L)
    return df

def rolling_stats(panel: pd.DataFrame, windows: List[int], skew_cols: List[str]) -> pd.DataFrame:
    idx = panel.set_index("date")
    rows = []
    for sk in skew_cols:
        x = idx["vix"]; y = idx[sk]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({
                "skew": sk, "window": w, "tag": tag,
                "corr": float(roll_corr(x,y,w).iloc[-1]) if len(idx)>=w else np.nan,
                "beta(d{sk}/dVIX)".format(sk=sk): float(roll_beta(idx[f"d_{sk}"], idx["dvix"], w).iloc[-1]) if f"d_{sk}" in idx.columns and len(idx)>=w else np.nan
            })
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, max_lag: int, skew_cols: List[str]) -> pd.DataFrame:
    rows = []
    for sk in skew_cols:
        tab = leadlag_corr(panel["vix"], panel[sk], max_lag)
        tab["skew"] = sk
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","skew"])

def detect_regimes(panel: pd.DataFrame, skew_cols: List[str]) -> pd.DataFrame:
    df = panel.copy()
    events = []
    for i in range(len(df)):
        r = df.iloc[i]
        if pd.notna(r.get("vix_z")) and r["vix_z"] >= 2.0:
            for sk in skew_cols:
                val = r.get(sk)
                if sk.lower().startswith("rr"):  # large negative RR = puts >> calls
                    cond = (pd.notna(val) and val < np.nanpercentile(df[sk], 20))
                    if cond:
                        events.append({"date": r["date"], "regime": "HighVIX_DeepPutSkew", "skew": sk, "vix": float(r["vix"]), "skew_val": float(val)})
                elif "slope" in sk.lower():
                    cond = (pd.notna(val) and val < np.nanpercentile(df[sk], 20))
                    if cond:
                        events.append({"date": r["date"], "regime": "HighVIX_NegSlope", "skew": sk, "vix": float(r["vix"]), "skew_val": float(val)})
    return pd.DataFrame(events).sort_values("date")

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    timeseries: Optional[str]
    options: Optional[str]
    vix_csv: Optional[str]
    vix_col: Optional[str]
    min_dte: int
    max_dte: int
    windows: str
    lags: int
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="India VIX vs options skew analytics")
    ap.add_argument("--timeseries", default="", help="CSV with date, vix, and skew columns (rr25/skew/slope/etc.)")
    ap.add_argument("--options", default="", help="CSV option chain (date, expiry, cp, strike, iv, underlying[, r]) to build skew")
    ap.add_argument("--vix_csv", default="", help="If using --options, provide a separate VIX timeseries CSV")
    ap.add_argument("--vix_col", default="", help="Column name for VIX in timeseries/vix_csv")
    ap.add_argument("--min_dte", type=int, default=7, help="Min DTE (days) for near-term bucket when using options")
    ap.add_argument("--max_dte", type=int, default=45, help="Max DTE (days) for near-term bucket when using options")
    ap.add_argument("--windows", default="10,20,60", help="Rolling windows in trading days (comma-separated)")
    ap.add_argument("--lags", type=int, default=10, help="Lead–lag horizon in days")
    ap.add_argument("--start", default="", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", default="", help="End date YYYY-MM-DD")
    ap.add_argument("--outdir", default="out_vix_skew")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load data paths
    PNL = pd.DataFrame()
    skew_cols: List[str] = []

    if args.timeseries:
        TS = load_timeseries(args.timeseries, vix_col_hint=(args.vix_col or ""))
        if args.start: TS = TS[TS["date"] >= pd.to_datetime(args.start)]
        if args.end:   TS = TS[TS["date"] <= pd.to_datetime(args.end)]
        PNL = TS.copy()
        skew_cols = [c for c in TS.columns if c not in ["date","vix"]]
    elif args.options:
        OPT = load_options(args.options)
        if args.start: OPT = OPT[OPT["date"] >= pd.to_datetime(args.start)]
        if args.end:   OPT = OPT[OPT["date"] <= pd.to_datetime(args.end)]
        SKEW = build_skew_from_options(OPT, min_dte=int(args.min_dte), max_dte=int(args.max_dte))
        if not args.vix_csv:
            raise ValueError("When using --options, you must also provide --vix_csv for India VIX.")
        VIX = load_vix_only(args.vix_csv, vix_col_hint=(args.vix_col or ""))
        TS = pd.merge(VIX, SKEW, on="date", how="inner")
        if TS.empty:
            raise ValueError("No overlap between VIX series and constructed skew — check dates.")
        PNL = TS.copy()
        skew_cols = [c for c in TS.columns if c not in ["date","vix"]]
    else:
        raise ValueError("Provide either --timeseries OR --options (with --vix_csv).")

    if PNL.shape[0] < 30:
        raise ValueError("Insufficient overlapping days after filters (need ≥30).")

    # Rolling windows
    windows = [int(w.strip()) for w in args.windows.split(",") if w.strip()]

    # Enrich panel
    PANEL = enrich_panel(PNL, windows)
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Rolling stats
    ROLL = rolling_stats(PANEL, windows, skew_cols)
    if not ROLL.empty:
        ROLL.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag
    LL = leadlag_tables(PANEL, max_lag=int(args.lags), skew_cols=skew_cols)
    if not LL.empty:
        LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Regimes
    REG = detect_regimes(PANEL, skew_cols)
    if not REG.empty:
        REG.to_csv(outdir / "regimes.csv", index=False)

    # Summary
    latest = PANEL.tail(1).iloc[0]
    best_ll = {}
    if not LL.empty:
        for sk, g in LL.dropna(subset=["corr"]).groupby("skew"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[sk] = {"lag": int(row["lag"]), "corr": float(row["corr"])}

    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "rows": int(PANEL.shape[0]),
        "skews": skew_cols,
        "latest": {
            "date": str(latest["date"].date()),
            "vix": float(latest["vix"]) if pd.notna(latest["vix"]) else None,
            **{sk: (float(latest[sk]) if pd.notna(latest[sk]) else None) for sk in skew_cols}
        },
        "leadlag_best": best_ll,
        "regime_events": int(REG.shape[0]) if not REG.empty else 0
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        timeseries=(args.timeseries or None),
        options=(args.options or None),
        vix_csv=(args.vix_csv or None),
        vix_col=(args.vix_col or None),
        min_dte=int(args.min_dte), max_dte=int(args.max_dte),
        windows=args.windows, lags=int(args.lags),
        start=(args.start or None), end=(args.end or None),
        outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== India VIX vs Skew ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Skews: {', '.join(skew_cols)}")
    if best_ll:
        for sk, st in best_ll.items():
            print(f"Lead–lag {sk}: max |corr| at lag {st['lag']:+d} → {st['corr']:+.2f}")
    if summary["regime_events"] > 0:
        print(f"Extreme regime flags written ({summary['regime_events']} rows).")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
