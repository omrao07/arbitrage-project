#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mobile_payments_upi.py — UPI analytics: volume/value, P2P vs P2M, failure spikes, diffusion & nowcast
-----------------------------------------------------------------------------------------------------

What this does
==============
Given UPI data (daily or monthly) and optional macro/events/state splits, this script:

1) Cleans & aligns everything to **monthly** (default) or **daily** frequency.
2) Builds KPIs
   • Txn count/value, AOV (avg value/txn), Δlog & YoY growth
   • P2P vs P2M shares (count & value) if available
   • Failure/success rates, active users/merchants/QR density if present
   • PSP concentration: shares & HHI if columns like psp_* are provided
3) Diagnostics
   • Rolling correlations of Δlog(UPI) vs macro covariates
   • Lead–lag tables Corr(Δlog X_{t−k}, Δlog UPI_t), k ∈ [−L..+L]
   • Event-study around policy/outage dates (e.g., UPI-on-RuPay, interchange tweaks)
4) Diffusion fit (optional)
   • Logistic fit for active users / merchants (K inferred or provided)
5) Nowcast (daily → month-end)
   • If input is daily and the current month is incomplete, scale-to-month to nowcast totals
6) Exports tidy CSVs + JSON summary.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--upi upi.csv            REQUIRED
  Columns (any subset; extras ignored). The script maps by fuzzy names:
    date
    txn_count, transactions, txns
    txn_value_inr, value_inr, value_rs, amount, gmv
    p2p_count, p2p_value, p2m_count, p2m_value
    failure_rate, success_rate, failures, successes
    active_users, users_active, active_handles, merchants, qr, qr_terminals, banks_live
    Any number of **PSP columns**:
      psp_<name>_count, psp_<name>_value  OR psp_<name>_share (0–1 or 0–100)
    (All numeric columns are kept; missing are fine.)

--macro macro.csv        OPTIONAL (monthly or daily)
  Columns: date, <any numeric drivers> e.g., smartphone_shipments, data_price, cpi, fest_days, ...
  (All numeric columns are retained as controls/diagnostics.)

--events events.csv      OPTIONAL (for event-study)
  Columns: date, label

--states states.csv      OPTIONAL (state-level panel)
  Columns: date, state, txn_count[, txn_value_inr, merchants, qr]
  (Produces penetration metrics if present, but not required.)

Key CLI
-------
--freq monthly|daily        Default monthly (daily allowed; YoY uses 365-day lag)
--windows 3,6,12            Rolling windows (periods)
--lags 6                    Lead–lag horizon (periods)
--diffuse_on users|merchants|none   Target for diffusion fit (default users if present)
--capacity 500_000_000      Optional carrying capacity (K) for diffusion fit
--start / --end             Date filters
--outdir out_upi            Output directory
--min_obs 24                Minimum observations for regressions

Outputs
-------
- panel.csv                 Aligned panel (levels, AOV, transforms, shares)
- psp_concentration.csv     PSP shares & HHI by period (if PSP columns present)
- rolling_stats.csv         Rolling corr of Δlog UPI vs macro
- leadlag_corr.csv          Lead–lag tables for each macro variable
- event_study.csv           Average Δlog deviations around events by horizon
- diffusion_fit.csv         Logistic fit results for users/merchants (if available)
- nowcast.csv               Daily→monthly scale-up nowcast (if applicable)
- states_panel.csv          Cleaned state panel & penetration (if provided)
- summary.json              Headline diagnostics & latest snapshot
- config.json               Run configuration

DISCLAIMER
----------
Research tooling with simple approximations; validate before operational/financial use.
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

def to_period_end(s: pd.Series, freq: str) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    if freq.startswith("M"):
        return dt.dt.to_period("M").dt.to_timestamp("M")
    elif freq.startswith("D"):
        return dt.dt.normalize()
    else:
        return dt

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def yoy(s: pd.Series, periods: int) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff(periods)

def resample_df(df: pd.DataFrame, freq: str, sum_cols: List[str], last_cols: List[str]) -> pd.DataFrame:
    """Resample to monthly/daily. Sum transactional flows; last for stocks; mean for everything else."""
    if df.empty: return df
    idx = df.set_index("date").sort_index()
    rule = "M" if freq.startswith("M") else "D"
    agg_map: Dict[str, str] = {}
    for c in idx.columns:
        if c in sum_cols:
            agg_map[c] = "sum"
        elif c in last_cols:
            agg_map[c] = "last"
        else:
            agg_map[c] = "mean" if pd.api.types.is_numeric_dtype(idx[c]) else "first"
    out = idx.resample(rule).agg(agg_map).reset_index()
    return out

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(4, window//2)).corr(y)

def leadlag_corr(flow: pd.Series, ret: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            f = flow.shift(k); r = ret
        else:
            f = flow; r = ret.shift(-k)
        c = f.corr(r)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

def zscore_window(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(6, window//3)).mean()
    sd = s.rolling(window, min_periods=max(6, window//3)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


# ----------------------------- loaders -----------------------------

def load_upi(path: str, freq: str) -> Tuple[pd.DataFrame, Dict[str,str], List[str], List[str]]:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt: "date"})
    df["date"] = to_period_end(df["date"], freq)

    # map primary fields
    cnt = ncol(df, "txn_count","transactions","txns","count")
    val = ncol(df, "txn_value_inr","value_inr","value_rs","amount","gmv","value")
    df = df.rename(columns={cnt or "txn_count":"txn_count", val or "txn_value_inr":"txn_value_inr"})

    # optional splits
    p2p_c = ncol(df, "p2p_count","p2p_txns")
    p2m_c = ncol(df, "p2m_count","p2m_txns","merchant_count")
    p2p_v = ncol(df, "p2p_value","p2p_value_inr")
    p2m_v = ncol(df, "p2m_value","p2m_value_inr","merchant_value")

    # failure/success
    fail_r = ncol(df, "failure_rate","fail_rate")
    succ_r = ncol(df, "success_rate","succ_rate")
    fails  = ncol(df, "failures","failed_txns")
    succs  = ncol(df, "successes","successful_txns")

    # stocks
    users = ncol(df, "active_users","users_active","active_handles","handles_active")
    merch = ncol(df, "merchants","active_merchants")
    qr    = ncol(df, "qr","qr_terminals","qr_codes")
    banks = ncol(df, "banks_live","upi_banks")

    # numerify all numeric-looking cols
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])

    # rename options if present
    if p2p_c: df = df.rename(columns={p2p_c:"p2p_count"})
    if p2m_c: df = df.rename(columns={p2m_c:"p2m_count"})
    if p2p_v: df = df.rename(columns={p2p_v:"p2p_value"})
    if p2m_v: df = df.rename(columns={p2m_v:"p2m_value"})
    if fail_r: df = df.rename(columns={fail_r:"failure_rate"})
    if succ_r: df = df.rename(columns={succ_r:"success_rate"})
    if fails:  df = df.rename(columns={fails:"failures"})
    if succs:  df = df.rename(columns={succs:"successes"})
    if users:  df = df.rename(columns={users:"active_users"})
    if merch:  df = df.rename(columns={merch:"merchants"})
    if qr:     df = df.rename(columns={qr:"qr"})
    if banks:  df = df.rename(columns={banks:"banks_live"})

    # infer rates if counts exist
    if "failure_rate" not in df.columns and {"failures","txn_count"} <= set(df.columns):
        df["failure_rate"] = df["failures"] / df["txn_count"].replace(0, np.nan)
    if "success_rate" not in df.columns and {"successes","txn_count"} <= set(df.columns):
        df["success_rate"] = df["successes"] / df["txn_count"].replace(0, np.nan)

    # AOVs
    df["aov"] = df["txn_value_inr"] / df["txn_count"]
    if "p2p_value" in df.columns and "p2p_count" in df.columns:
        df["aov_p2p"] = df["p2p_value"] / df["p2p_count"]
    if "p2m_value" in df.columns and "p2m_count" in df.columns:
        df["aov_p2m"] = df["p2m_value"] / df["p2m_count"]

    # PSP columns detection
    psp_cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("psp_")]
    # Normalize shares to decimals if looks like percent (>1 median)
    for c in psp_cols:
        if "share" in c.lower():
            med = np.nanmedian(df[c].values.astype(float))
            if med > 1.0: df[c] = df[c] / 100.0

    # columns to sum/last for resampling
    sum_cols = [c for c in ["txn_count","txn_value_inr","p2p_count","p2m_count","p2p_value","p2m_value","failures","successes"] if c in df.columns]
    sum_cols += [c for c in psp_cols if any(x in c.lower() for x in ["_count","_value"])]
    last_cols = [c for c in ["active_users","merchants","qr","banks_live"] if c in df.columns]
    # resample
    out = resample_df(df[["date"] + [c for c in df.columns if c!="date"]], freq=freq, sum_cols=sum_cols, last_cols=last_cols)

    # recompute derived metrics post-resample
    out["aov"] = out["txn_value_inr"] / out["txn_count"]
    if {"p2p_count","txn_count"} <= set(out.columns):
        out["share_p2p_count"] = out["p2p_count"] / out["txn_count"].replace(0, np.nan)
    if {"p2m_count","txn_count"} <= set(out.columns):
        out["share_p2m_count"] = out["p2m_count"] / out["txn_count"].replace(0, np.nan)
    if {"p2p_value","txn_value_inr"} <= set(out.columns):
        out["share_p2p_value"] = out["p2p_value"] / out["txn_value_inr"].replace(0, np.nan)
    if {"p2m_value","txn_value_inr"} <= set(out.columns):
        out["share_p2m_value"] = out["p2m_value"] / out["txn_value_inr"].replace(0, np.nan)

    # transforms (growth)
    yo_periods = 12 if freq.startswith("M") else 365
    for c in ["txn_count","txn_value_inr","aov","p2p_count","p2m_count","p2p_value","p2m_value"]:
        if c in out.columns:
            out[f"dlog_{c}"] = dlog(out[c])
            out[f"yoy_{c}"] = yoy(out[c], periods=yo_periods)

    # failure/success rates as is
    if "failure_rate" in out.columns and out["failure_rate"].dropna().empty:
        out.drop(columns=["failure_rate"], inplace=True)

    mapping = {
        "count": "txn_count",
        "value": "txn_value_inr",
        "p2p_count": "p2p_count" if "p2p_count" in out.columns else None,
        "p2m_count": "p2m_count" if "p2m_count" in out.columns else None,
        "active_users": "active_users" if "active_users" in out.columns else None,
        "merchants": "merchants" if "merchants" in out.columns else None,
        "qr": "qr" if "qr" in out.columns else None,
        "failure_rate": "failure_rate" if "failure_rate" in out.columns else None,
    }
    return out.sort_values("date"), mapping, psp_cols, sum_cols

def load_macro(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt: "date"})
    df["date"] = to_period_end(df["date"], freq)
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])
    # resample monthly mean / daily mean
    out = df.set_index("date").sort_index().resample("M" if freq.startswith("M") else "D").mean().reset_index()
    return out

def load_events(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    lab = ncol(df, "label","event","name") or "label"
    df = df.rename(columns={dt: "date", lab: "label"})
    df["date"] = to_period_end(df["date"], freq)
    df["label"] = df["label"].astype(str)
    return df[["date","label"]].dropna()

def load_states(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    st = ncol(df, "state","region")
    if not st: raise ValueError("states.csv must include a 'state' column.")
    df = df.rename(columns={dt: "date", st: "state"})
    df["date"] = to_period_end(df["date"], freq)
    # map common columns
    cnt = ncol(df, "txn_count","txns","transactions","count")
    val = ncol(df, "txn_value_inr","value_inr","amount","value")
    mer = ncol(df, "merchants","active_merchants")
    qr  = ncol(df, "qr","qr_terminals")
    keep = ["date","state"]
    for (c, nm) in [(cnt,"txn_count"), (val,"txn_value_inr"), (mer,"merchants"), (qr,"qr")]:
        if c: df = df.rename(columns={c: nm}); keep.append(nm)
    for c in keep:
        if c!="date" and c!="state": df[c] = safe_num(df[c])
    out = df[keep].copy()
    return out.sort_values(["state","date"])

# ----------------------------- analytics -----------------------------

def psp_concentration(panel: pd.DataFrame, psp_cols: List[str]) -> pd.DataFrame:
    if not psp_cols: return pd.DataFrame()
    df = panel[["date"] + psp_cols].copy()
    # If value/count columns exist, aggregate to shares by *value* if available, else count; if *_share exist, use as-is.
    val_cols = [c for c in psp_cols if c.endswith("_value")]
    cnt_cols = [c for c in psp_cols if c.endswith("_count")]
    shr_cols = [c for c in psp_cols if "share" in c.lower()]
    out_rows = []
    for _, r in df.iterrows():
        if not pd.isna(r["date"]):
            if val_cols:
                sub = r[val_cols].astype(float)
                tot = float(np.nansum(sub))
                shares = (sub / tot) if tot>0 else sub*0.0
            elif cnt_cols:
                sub = r[cnt_cols].astype(float)
                tot = float(np.nansum(sub))
                shares = (sub / tot) if tot>0 else sub*0.0
            elif shr_cols:
                sub = r[shr_cols].astype(float)
                # ensure decimals
                shares = sub.apply(lambda x: x/100.0 if x>1.0 else x)
            else:
                continue
            hhi = float(np.nansum((shares.values.astype(float))**2))
            top3 = float(np.nansum(np.sort(shares.values.astype(float))[::-1][:3]))
            out_rows.append({"date": r["date"], "psp_hhi": hhi, "psp_top3_share": top3})
    return pd.DataFrame(out_rows).sort_values("date")

def rolling_stats(panel: pd.DataFrame, macro: pd.DataFrame, windows: List[int], target: str="dlog_txn_count") -> pd.DataFrame:
    if panel.empty or macro.empty or target not in panel.columns:
        return pd.DataFrame()
    idx = panel.merge(macro, on="date", how="inner").set_index("date")
    y = idx[target]
    rows = []
    numeric_macros = [c for c in idx.columns if c not in panel.columns and pd.api.types.is_numeric_dtype(idx[c])]
    for m in numeric_macros:
        x = idx[m]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({"macro": m, "window": w, "tag": tag,
                         "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan})
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, macro: pd.DataFrame, lags: int, target: str="dlog_txn_count") -> pd.DataFrame:
    if panel.empty or macro.empty or target not in panel.columns:
        return pd.DataFrame()
    idx = panel.merge(macro, on="date", how="inner").sort_values("date")
    y = idx[target]
    rows = []
    numeric_macros = [c for c in idx.columns if c not in panel.columns and pd.api.types.is_numeric_dtype(idx[c])]
    for m in numeric_macros:
        tab = leadlag_corr(idx[m], y, lags)
        tab["macro"] = m
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","macro"])

def event_study(panel: pd.DataFrame, events: pd.DataFrame, window: int, targets: List[str]) -> pd.DataFrame:
    if events.empty or panel.empty:
        return pd.DataFrame()
    rows = []
    dates = panel["date"].values
    for _, ev in events.iterrows():
        d0 = pd.to_datetime(ev["date"])
        lbl = str(ev["label"])
        # nearest panel date
        if len(dates)==0: continue
        idx = np.argmin(np.abs(dates - np.datetime64(d0)))
        anchor = pd.to_datetime(dates[idx])
        sl = panel[(panel["date"] >= anchor - pd.offsets.DateOffset(months=window)) &
                   (panel["date"] <= anchor + pd.offsets.DateOffset(months=window))]
        for _, r in sl.iterrows():
            h = (r["date"].to_period("M") - anchor.to_period("M")).n if isinstance(r["date"], pd.Timestamp) else 0
            row = {"event": lbl, "event_date": anchor, "h": int(h)}
            for t in targets:
                if t in panel.columns:
                    row[t] = float(r[t]) if pd.notna(r[t]) else np.nan
            rows.append(row)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # average by horizon
    grp_cols = ["event","event_date","h"]
    agg = {t: "mean" for t in targets if t in df.columns}
    out = df.groupby(grp_cols).agg(agg).reset_index()
    return out.sort_values(["event","h"])

def diffusion_fit(series: pd.Series, dates: pd.Series, K: Optional[float]=None) -> Dict:
    """
    Fit logistic y_t = K / (1 + exp(-(a + b*t))). Linearize with logit(y/K):
    ln(y/(K-y)) = a + b*t; choose K as 1.25×max(y) if not provided.
    """
    s = series.dropna()
    if s.shape[0] < 12:
        return {}
    if K is None or not np.isfinite(K) or K <= s.max():
        K = 1.25*float(s.max())
    t = pd.RangeIndex(start=0, stop=len(s), step=1).astype(float).values.reshape(-1,1)
    y = s.values.astype(float).reshape(-1,1)
    # avoid boundary issues
    y = np.clip(y, 1e-6, K-1e-6)
    z = np.log(y/(K - y))
    X = np.column_stack([np.ones_like(t), t])
    XtX = X.T @ X
    XtY = X.T @ z
    beta = np.linalg.pinv(XtX) @ XtY
    a, b = float(beta[0,0]), float(beta[1,0])
    # implied growth rate r ≈ b
    # t_50 (half-saturation): when y=K/2 ⇒ z=0 ⇒ a + b*t = 0 ⇒ t_50 = -a/b
    t50 = -a/b if b!=0 else np.nan
    return {"K": float(K), "a": a, "b": b, "t50_idx": float(t50), "n": int(s.shape[0]),
            "start": str(dates.iloc[0].date()), "end": str(dates.iloc[s.shape[0]-1].date())}

def daily_month_nowcast(daily: pd.DataFrame) -> pd.DataFrame:
    """
    For the last (possibly incomplete) month in daily data, scale YTD-in-month to full month.
    Returns a single-row DataFrame with nowcast for txn_count & txn_value_inr.
    """
    if daily.empty or "txn_count" not in daily.columns: return pd.DataFrame()
    d = daily.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    y, m = int(d["year"].max()), int(d[d["year"]==d["year"].max()]["month"].max())
    cur = d[(d["year"]==y)&(d["month"]==m)].sort_values("date")
    if cur.empty: return pd.DataFrame()
    days_elapsed = cur["date"].dt.day.max()
    days_in_month = int(pd.Period(cur["date"].iloc[0], freq="M").days_in_month)
    scale = days_in_month / max(1, days_elapsed)
    row = {"year": y, "month": m, "days_elapsed": int(days_elapsed), "days_in_month": int(days_in_month), "scale": float(scale)}
    for c in ["txn_count","txn_value_inr","p2m_count","p2m_value","p2p_count","p2p_value"]:
        if c in cur.columns:
            row[f"{c}_month_so_far"] = float(cur[c].sum())
            row[f"{c}_nowcast_full_month"] = float(cur[c].sum() * scale)
    return pd.DataFrame([row])

def add_transforms(panel: pd.DataFrame, freq: str) -> pd.DataFrame:
    yo_periods = 12 if freq.startswith("M") else 365
    df = panel.copy().sort_values("date")
    for c in ["txn_count","txn_value_inr","aov","p2m_count","p2p_count","p2m_value","p2p_value"]:
        if c in df.columns:
            df[f"dlog_{c}"] = dlog(df[c])
            df[f"yoy_{c}"] = yoy(df[c], periods=yo_periods)
    # z-scores for growth
    for c in [col for col in df.columns if col.startswith("dlog_")]:
        df[f"{c}_z"] = zscore_window(df[c], window=24 if freq.startswith("M") else 180)
    return df

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    upi: str
    macro: Optional[str]
    events: Optional[str]
    states: Optional[str]
    freq: str
    windows: str
    lags: int
    diffuse_on: str
    capacity: Optional[float]
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="UPI analytics: volume/value, splits, diffusion & nowcast")
    ap.add_argument("--upi", required=True)
    ap.add_argument("--macro", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--states", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","daily"])
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--diffuse_on", default="auto", choices=["auto","users","merchants","none"])
    ap.add_argument("--capacity", type=float, default=float("nan"), help="Carrying capacity K for diffusion (users/merchants)")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_upi")
    ap.add_argument("--min_obs", type=int, default=24)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq = "M" if args.freq.startswith("m") else "D"

    PANEL, mapping, psp_cols, sum_cols = load_upi(args.upi, freq=freq)
    MACRO = load_macro(args.macro, freq=freq) if args.macro else pd.DataFrame()
    EVENTS = load_events(args.events, freq=freq) if args.events else pd.DataFrame()
    STATES = load_states(args.states, freq=freq) if args.states else pd.DataFrame()

    # Date filters
    if args.start:
        for df in [PANEL, MACRO, STATES]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [PANEL, MACRO, STATES]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    if PANEL.shape[0] < args.min_obs:
        raise ValueError("Insufficient observations after alignment. Provide more history.")

    # Derive transforms & save
    PANEL = add_transforms(PANEL, freq=freq)
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # PSP concentration
    PSP = psp_concentration(PANEL, psp_cols)
    if not PSP.empty:
        PSP.to_csv(outdir / "psp_concentration.csv", index=False)
        PANEL = PANEL.merge(PSP, on="date", how="left")

    # Rolling stats & Lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, MACRO, windows, target="dlog_txn_count") if not MACRO.empty else pd.DataFrame()
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)

    LL = leadlag_tables(PANEL, MACRO, lags=int(args.lags), target="dlog_txn_count") if not MACRO.empty else pd.DataFrame()
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Event study (on growth of count/value)
    ES = pd.DataFrame()
    if not EVENTS.empty:
        targets = [c for c in ["dlog_txn_count","dlog_txn_value_inr","failure_rate"] if c in PANEL.columns]
        ES = event_study(PANEL, EVENTS, window=max(3, int(args.lags)), targets=targets)
        if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Diffusion fit
    DIFF = {}
    diffuse_target = None
    if args.diffuse_on != "none":
        if args.diffuse_on == "auto":
            diffuse_target = "active_users" if mapping.get("active_users") else ("merchants" if mapping.get("merchants") else None)
        elif args.diffuse_on == "users":
            diffuse_target = "active_users" if mapping.get("active_users") else None
        elif args.diffuse_on == "merchants":
            diffuse_target = "merchants" if mapping.get("merchants") else None
        if diffuse_target:
            K = args.capacity if np.isfinite(args.capacity) else None
            DIFF = diffusion_fit(PANEL[diffuse_target], PANEL["date"], K=K)
            if DIFF:
                pd.DataFrame([DIFF]).to_csv(outdir / "diffusion_fit.csv", index=False)

    # Nowcast (daily only)
    NOW = pd.DataFrame()
    if not args.freq.startswith("m"):
        NOW = daily_month_nowcast(PANEL[["date"] + [c for c in ["txn_count","txn_value_inr","p2m_count","p2m_value","p2p_count","p2p_value"] if c in PANEL.columns]].copy())
        if not NOW.empty:
            NOW.to_csv(outdir / "nowcast.csv", index=False)

    # States panel (optional)
    if not STATES.empty:
        # Aggregate to monthly/daily aligned, compute per-capita if state_pop provided? (not required)
        STATES["aov"] = STATES["txn_value_inr"] / STATES["txn_count"] if "txn_value_inr" in STATES.columns and "txn_count" in STATES.columns else np.nan
        STATES.to_csv(outdir / "states_panel.csv", index=False)

    # Best lead/lag summary
    best_ll = {}
    if not LL.empty:
        for m, g in LL.dropna(subset=["corr"]).groupby("macro"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[m] = {"lag": int(row["lag"]), "corr": float(row["corr"])}

    # Summary
    latest = PANEL.dropna(subset=["txn_count"]).tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "freq": "monthly" if freq.startswith("M") else "daily",
        "columns_used": mapping,
        "latest": {
            "date": str(latest["date"].date()),
            "txn_count": float(latest["txn_count"]),
            "txn_value_inr": float(latest.get("txn_value_inr", np.nan)) if pd.notna(latest.get("txn_value_inr", np.nan)) else None,
            "aov_inr": float(latest.get("aov", np.nan)) if pd.notna(latest.get("aov", np.nan)) else None,
            "p2m_share_count": float(latest.get("share_p2m_count", np.nan)) if pd.notna(latest.get("share_p2m_count", np.nan)) else None,
            "failure_rate": float(latest.get("failure_rate", np.nan)) if pd.notna(latest.get("failure_rate", np.nan)) else None,
            "psp_hhi": float(latest.get("psp_hhi", np.nan)) if pd.notna(latest.get("psp_hhi", np.nan)) else None
        },
        "rolling_windows": windows,
        "leadlag_best": best_ll,
        "diffusion": DIFF,
        "nowcast_available": (not NOW.empty),
        "events_count": (int(EVENTS.shape[0]) if not EVENTS.empty else 0)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        upi=args.upi, macro=(args.macro or None), events=(args.events or None), states=(args.states or None),
        freq=args.freq, windows=args.windows, lags=int(args.lags), diffuse_on=args.diffuse_on,
        capacity=(args.capacity if np.isfinite(args.capacity) else None),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== UPI Analytics ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Freq: {summary['freq']}")
    core = f"Count={summary['latest']['txn_count']:.0f}"
    if summary["latest"]["txn_value_inr"] is not None:
        core += f", Value=₹{summary['latest']['txn_value_inr']:.2f}"
    if summary["latest"]["aov_inr"] is not None:
        core += f", AOV=₹{summary['latest']['aov_inr']:.2f}"
    print("Latest:", core)
    if summary["latest"]["failure_rate"] is not None:
        print(f"Failure rate (latest): {summary['latest']['failure_rate']*100:.2f}%")
    if best_ll:
        for m, st in best_ll.items():
            print(f"Lead–lag macro {m}: max |corr| at lag {st['lag']:+d} → {st['corr']:+.2f}")
    if DIFF:
        print(f"Diffusion fit on {'active_users' if mapping.get('active_users') else 'merchants'}: b≈{DIFF['b']:+.4f}, K≈{DIFF['K']:.0f}, t50≈{DIFF['t50_idx']:.1f}")
    if not NOW.empty:
        print(f"Nowcast: scaled by ×{NOW['scale'].iloc[0]:.2f} to full month (days {NOW['days_elapsed'].iloc[0]}/{NOW['days_in_month'].iloc[0]})")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
