#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logistics_gst.py — India logistics activity vs GST collections
--------------------------------------------------------------

What this does
==============
Given timeseries for logistics activity (E-Way Bills, FASTag, freight rate/load indices,
rail freight, diesel), this script:

1) Cleans & aligns all series to **monthly** or **weekly** frequency.
2) Builds transforms: Δlog growth, YoY, z-scores.
3) Runs diagnostics:
   • Rolling correlations between logistics indicators and GST collections
   • Lead–lag tables Corr(Δlog X_{t−k}, Δlog GST_t), k∈[−L..+L]
   • Event-study around policy/event dates (e.g., 2017-07-01 GST rollout, 2018-04 E-Way Bill)
4) Estimates a simple **nowcast/elasticity** model:
   Δlog(GST)_t ~ β1·Δlog(EWB)_t + β2·Δlog(FASTag)_t + β3·Δlog(Freight)_t + β4·Δlog(Diesel)_t + ε_t
   (HAC/Newey–West standard errors)
5) Generates a nowcast for the most recent period(s) where GST is missing but activity is present.

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--gst gst.csv            REQUIRED
  Columns (any subset; extras ignored):
    date, gst_gross, gst_net, gst_total, cgst, sgst, igst, cess
  (The script auto-picks a "best" GST total column.)

--eway eway.csv          OPTIONAL
  Columns: date, eway_bills (or eway, eway_count, ewb)

--fastag fastag.csv      OPTIONAL
  Columns: date, fastag_value (INR), fastag_txns (count), fastag (either ok)

--freight freight.csv    OPTIONAL
  Columns: date, freight_index (TRUCK/Freight rate index), load_factor, ccf, rldi, etc.
          (Any numeric column retained; the one with the longest coverage used as "freight_proxy".)

--rail rail.csv          OPTIONAL
  Columns: date, rail_loading_mt (or rkm/ntkm/tonnage)

--diesel diesel.csv      OPTIONAL
  Columns: date, diesel_price (or diesel, diesel_inr_ltr)

--events events.csv      OPTIONAL (for event study)
  Columns: date, label
  (E.g., 2017-07-01 "GST Rollout"; 2018-04-01 "E-Way Bill national"; 2020-10-01 "E-invoicing phase 1")

Key CLI
-------
--freq monthly|weekly       Default monthly
--windows 3,6,12            Rolling windows (periods)
--lags 6                    Lead–lag horizon (periods)
--event_window 6            Event study window (±periods around event)
--start / --end             Date filters
--outdir out_gst
--min_obs 36                Min observations to run regressions

Outputs
-------
- panel.csv                 Aligned panel (levels & transforms)
- rolling_stats.csv         Rolling corr of Δlog GST vs Δlog indicators
- leadlag_corr.csv          Lead–lag tables for each indicator
- event_study.csv           Average deviations around events by indicator & horizon
- regression_main.csv       OLS elasticities with HAC/Newey–West SEs
- nowcast.csv               Fitted Δlog GST and implied level; fills recent missing months
- summary.json              Headline diagnostics & last-available snapshot
- config.json               Run configuration

DISCLAIMER
----------
Research tooling with simplifying assumptions; validate before operational or policy use.
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
    elif freq.startswith("W"):
        # align to week-end Sunday
        return (dt + pd.offsets.Week(weekday=6)).dt.normalize()
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

def resample_df(df: pd.DataFrame, freq: str, how: str="sum") -> pd.DataFrame:
    if df.empty: return df
    idx = "date"; df = df.set_index(idx).sort_index()
    # For value-like series (GST, FASTag, e-way counts): sum if higher frequency; else mean is fine.
    agg = {c: (how if pd.api.types.is_numeric_dtype(df[c]) else "first") for c in df.columns}
    out = df.resample("M" if freq.startswith("M") else "W-SUN").agg(agg)
    return out.reset_index()

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(3, window//2)).corr(y)

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
    return np.sqrt(np.diag(cov))


# ----------------------------- loaders -----------------------------

def load_gst(path: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt: "date"})
    df["date"] = to_period_end(df["date"], freq)
    # pick GST column
    cand = (ncol(df, "gst_gross","gst_total","gst_net","gross_gst","gst") or
            ncol(df, "total_gst","gst_revenue"))
    if not cand:
        # try components sum
        cgst, sgst, igst, cess = [ncol(df, "cgst"), ncol(df, "sgst"), ncol(df, "igst"), ncol(df, "cess")]
        if not any([cgst, sgst, igst, cess]):
            raise ValueError("gst.csv: cannot find GST total nor components.")
        df["gst_total"] = sum([safe_num(df[c]) for c in [cgst, sgst, igst, cess] if c])
        cand = "gst_total"
    df[cand] = safe_num(df[cand])
    out = df[["date", cand]].rename(columns={cand: "gst_total"})
    out = resample_df(out, freq, how="sum")
    return out

def load_generic(path: str, freq: str, name_map: List[Tuple[str, List[str]]], how: str="sum") -> pd.DataFrame:
    """Load a CSV and extract columns by fuzzy names. Returns wide DF with 'date' + mapped names."""
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt: "date"})
    df["date"] = to_period_end(df["date"], freq)
    keep = ["date"]
    out_ren = {}
    for canon, opts in name_map:
        c = None
        for opt in opts:
            c = ncol(df, opt)
            if c: break
        if c:
            df[c] = safe_num(df[c]); keep.append(c); out_ren[c] = canon
    if keep == ["date"]:
        return pd.DataFrame()
    out = df[keep].rename(columns=out_ren)
    out = resample_df(out, freq, how=how)
    return out

def pick_proxy(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df.empty: return None
    exists = [c for c in candidates if c in df.columns]
    if not exists: return None
    # pick with max non-na count
    counts = {c: df[c].notna().sum() for c in exists}
    return max(counts, key=counts.get)


# ----------------------------- core construction -----------------------------

def build_panel(freq: str,
                GST: pd.DataFrame,
                EWB: pd.DataFrame,
                FAST: pd.DataFrame,
                FRT: pd.DataFrame,
                RAIL: pd.DataFrame,
                DSL: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    base = GST.copy()
    dfs = [EWB, FAST, FRT, RAIL, DSL]
    for d in dfs:
        if not d.empty:
            base = base.merge(d, on="date", how="outer")
    base = base.sort_values("date").drop_duplicates(subset=["date"])
    # compute transforms
    out = base.copy()
    # choose proxies
    mapping = {}
    if "eway_bills" in out.columns:
        mapping["eway_proxy"] = "eway_bills"
    elif "eway" in out.columns:
        mapping["eway_proxy"] = "eway"
    else:
        mapping["eway_proxy"] = pick_proxy(out, ["ewb","eway_count"])

    mapping["fastag_proxy"] = pick_proxy(out, ["fastag_value","fastag_txns","fastag"])
    mapping["freight_proxy"] = pick_proxy(out, ["freight_index","freight","load_factor","rldi","ccf"])
    mapping["rail_proxy"] = pick_proxy(out, ["rail_loading_mt","ntkm","rkm","tonnage"])
    mapping["diesel_proxy"] = pick_proxy(out, ["diesel_price","diesel","diesel_inr_ltr"])

    # Δlog / YoY
    period_yoy = 12 if freq.startswith("M") else 52
    for c in out.columns:
        if c == "date": continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[f"dlog_{c}"] = dlog(out[c])
            out[f"yoy_{c}"] = yoy(out[c], periods=period_yoy)
    # z-scores (rolling 24M/52W)
    L = 24 if freq.startswith("M") else 52
    for c in out.columns:
        if c.startswith("dlog_"):
            out[f"{c}_z"] = out[c].rolling(L, min_periods=max(6, L//4)).apply(
                lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x, ddof=0) or np.nan)
            )
    return out, mapping


# ----------------------------- analytics -----------------------------

def rolling_stats(panel: pd.DataFrame, windows: List[int], mapping: Dict[str,str]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    y = idx.get("dlog_gst_total")
    if y is None: return pd.DataFrame()
    for name, col in mapping.items():
        if not col or f"dlog_{col}" not in idx.columns: continue
        x = idx[f"dlog_{col}"]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({
                "indicator": name.replace("_proxy",""),
                "column": col, "window": w, "tag": tag,
                "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan
            })
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, lags: int, mapping: Dict[str,str]) -> pd.DataFrame:
    rows = []
    y = panel.get("dlog_gst_total")
    if y is None or y.dropna().empty: return pd.DataFrame()
    for name, col in mapping.items():
        if not col or f"dlog_{col}" not in panel.columns: continue
        tab = leadlag_corr(panel[f"dlog_{col}"], y, lags)
        tab["indicator"] = name.replace("_proxy","")
        tab["column"] = col
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","indicator","column"])

def event_study(panel: pd.DataFrame, events: pd.DataFrame, window: int, targets: List[Tuple[str,str]]) -> pd.DataFrame:
    """
    Compute average Δlog deviations around event dates for given (label, column) targets.
    Returns stacked result with horizon h=-W..+W.
    """
    if events.empty: return pd.DataFrame()
    rows = []
    dates = panel["date"].values
    for _, ev in events.iterrows():
        d0 = pd.to_datetime(ev["date"])
        lbl = str(ev.get("label","event"))
        # find nearest panel date
        if len(dates)==0: continue
        # index of closest
        idx = np.argmin(np.abs(dates - np.datetime64(d0)))
        dstar = pd.to_datetime(dates[idx])
        for lab, col in targets:
            c = f"dlog_{col}"
            if c not in panel.columns: continue
            W = window
            sl = panel[(panel["date"] >= dstar - pd.offsets.DateOffset(months=W)) &
                       (panel["date"] <= dstar + pd.offsets.DateOffset(months=W))]
            # compute horizon in periods relative to event anchor
            for _, r in sl.iterrows():
                h = (r["date"].to_period("M") - dstar.to_period("M")).n if isinstance(r["date"], pd.Timestamp) else 0
                rows.append({"event": lbl, "event_date": dstar, "h": int(h), "target": lab, "col": col,
                             "dlog": float(r[c]) if pd.notna(r[c]) else np.nan})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Average by event, target, horizon
    out = (df.groupby(["event","event_date","target","col","h"])["dlog"]
             .mean().reset_index().rename(columns={"dlog":"avg_dlog"}))
    return out.sort_values(["event","target","h"])

def run_regression(panel: pd.DataFrame, mapping: Dict[str,str], L_hac: int, min_obs: int) -> pd.DataFrame:
    """
    Δlog(GST) on contemporaneous Δlog indicators (drop missing).
    """
    y = panel.get("dlog_gst_total")
    if y is None: return pd.DataFrame()
    Xparts = [pd.Series(1.0, index=panel.index, name="const")]
    varnames = []
    for key in ["eway_proxy","fastag_proxy","freight_proxy","diesel_proxy","rail_proxy"]:
        col = mapping.get(key)
        if not col: continue
        c = f"dlog_{col}"
        if c in panel.columns:
            Xparts.append(panel[c].rename(c))
            varnames.append(c)
    if len(Xparts) == 1:
        return pd.DataFrame()
    X = pd.concat(Xparts, axis=1)
    XY = pd.concat([y.rename("dep"), X], axis=1).dropna()
    if XY.shape[0] < max(min_obs, 5*X.shape[1]):
        return pd.DataFrame()
    yv = XY["dep"].values.reshape(-1,1)
    Xv = XY.drop(columns=["dep"]).values
    beta, resid, XtX_inv = ols_beta_se(Xv, yv)
    se = hac_se(Xv, resid, XtX_inv, L=L_hac)
    names = XY.drop(columns=["dep"]).columns.tolist()
    rows = []
    for i, nm in enumerate(names):
        rows.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                     "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                     "n": int(XY.shape[0])})
    return pd.DataFrame(rows)

def build_nowcast(panel: pd.DataFrame, reg: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Use regression to fit Δlog GST where predictors exist; back out fitted GST level.
    If latest GST is missing but predictors are present, generate nowcast.
    """
    if reg.empty or "coef" not in reg.columns: return pd.DataFrame()
    # coefficients
    coefs = {r["var"]: r["coef"] for _, r in reg.iterrows()}
    if "const" not in coefs: coefs["const"] = 0.0
    work = panel.copy().sort_values("date")
    # required predictor names
    preds = [k for k in coefs.keys() if k!="const"]
    for p in preds:
        if p not in work.columns: work[p] = np.nan
    # predict dlog_gst_hat where predictors are available
    X = work[preds]
    work["dlog_gst_hat"] = coefs["const"] + X.mul([coefs.get(c,0.0) for c in preds], axis=1).sum(axis=1, min_count=1)
    # build fitted GST level using recursive application where GST_total available as anchor
    work["gst_fitted"] = np.nan
    # start from first non-NaN gst_total
    anchor_idx = work["gst_total"].first_valid_index()
    if anchor_idx is None:
        return pd.DataFrame()
    work.loc[anchor_idx, "gst_fitted"] = work.loc[anchor_idx, "gst_total"]
    for i in range(anchor_idx+1, len(work)):
        prev = work.loc[i-1, "gst_fitted"]
        dl = work.loc[i, "dlog_gst_hat"]
        if pd.notna(prev) and pd.notna(dl):
            work.loc[i, "gst_fitted"] = prev * float(np.exp(dl))
        else:
            # if we have actual GST, reset anchor; else carry prev
            act = work.loc[i, "gst_total"]
            work.loc[i, "gst_fitted"] = act if pd.notna(act) else prev
    return work[["date","gst_total","dlog_gst_hat","gst_fitted"]]

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    gst: str
    eway: Optional[str]
    fastag: Optional[str]
    freight: Optional[str]
    rail: Optional[str]
    diesel: Optional[str]
    events: Optional[str]
    freq: str
    windows: str
    lags: int
    event_window: int
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Logistics activity vs GST analytics (India)")
    ap.add_argument("--gst", required=True)
    ap.add_argument("--eway", default="")
    ap.add_argument("--fastag", default="")
    ap.add_argument("--freight", default="")
    ap.add_argument("--rail", default="")
    ap.add_argument("--diesel", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","weekly"])
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--event_window", type=int, default=6)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_gst")
    ap.add_argument("--min_obs", type=int, default=36)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq = "M" if args.freq.startswith("m") else "W"

    GST = load_gst(args.gst, freq=freq)

    EWB = load_generic(args.eway, freq=freq, how="sum", name_map=[
        ("eway_bills", ["eway_bills","eway","ewb","eway_count"])
    ]) if args.eway else pd.DataFrame()

    FAST = load_generic(args.fastag, freq=freq, how="sum", name_map=[
        ("fastag_value", ["fastag_value","fastag_rev","fastag_amount","fastag_gmv","toll_value"]),
        ("fastag_txns",  ["fastag_txns","fastag_count","toll_txns","transactions"])
    ]) if args.fastag else pd.DataFrame()

    FRT = load_generic(args.freight, freq=freq, how="mean", name_map=[
        ("freight_index", ["freight_index","freight","truck_rate","rldi","ccf","fti","fti_index","fti_value"]),
        ("load_factor",   ["load_factor","fleet_util","capacity_util"])
    ]) if args.freight else pd.DataFrame()

    RAIL = load_generic(args.rail, freq=freq, how="sum", name_map=[
        ("rail_loading_mt", ["rail_loading_mt","rail_tonnage","tonnage","ntkm","rkm"])
    ]) if args.rail else pd.DataFrame()

    DSL = load_generic(args.diesel, freq=freq, how="mean", name_map=[
        ("diesel_price", ["diesel_price","diesel","diesel_inr_ltr","diesel_inr"])
    ]) if args.diesel else pd.DataFrame()

    EVENTS = pd.DataFrame()
    if args.events:
        ev = pd.read_csv(args.events)
        dt = ncol(ev, "date") or ev.columns[0]
        ev = ev.rename(columns={dt:"date"})
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce").dt.tz_localize(None)
        lab = ncol(ev, "label","event","name") or "label"
        if lab not in ev.columns: ev["label"] = "event"
        EVENTS = ev[["date","label"]].dropna()

    # Date filters
    if args.start:
        for df in [GST, EWB, FAST, FRT, RAIL, DSL]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [GST, EWB, FAST, FRT, RAIL, DSL]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    PANEL, mapping = build_panel(freq=freq, GST=GST, EWB=EWB, FAST=FAST, FRT=FRT, RAIL=RAIL, DSL=DSL)
    if PANEL["gst_total"].notna().sum() < max(12, args.min_obs//2):
        raise ValueError("Insufficient GST observations after alignment.")

    # Persist panel
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Rolling correlations
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, windows, mapping)
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag
    LL = leadlag_tables(PANEL, lags=int(args.lags), mapping=mapping)
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Event study (monthly horizons)
    ES = pd.DataFrame()
    if not EVENTS.empty:
        # choose targets: GST and available proxies
        targets = [("GST", "gst_total")]
        for key, col in mapping.items():
            if col: targets.append((key.replace("_proxy","").upper(), col))
        ES = event_study(PANEL, EVENTS, window=int(args.event_window), targets=targets)
        if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Regression (elasticities)
    REG = run_regression(PANEL, mapping, L_hac=max(4, int(args.lags)), min_obs=int(args.min_obs))
    if not REG.empty: REG.to_csv(outdir / "regression_main.csv", index=False)

    # Nowcast
    NC = build_nowcast(PANEL, REG, freq=freq)
    if not NC.empty: NC.to_csv(outdir / "nowcast.csv", index=False)

    # Best lead/lag summary
    best_ll = {}
    if not LL.empty:
        for ind, g in LL.dropna(subset=["corr"]).groupby("indicator"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[ind] = {"lag": int(row["lag"]), "corr": float(row["corr"]), "column": row["column"]}

    # Summary
    latest = PANEL.dropna(subset=["gst_total"]).tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "freq": "monthly" if freq.startswith("M") else "weekly",
        "columns_used": mapping,
        "latest": {
            "date": str(latest["date"].date()),
            "gst_total": float(latest["gst_total"]),
            **{k: (float(latest[v]) if v and v in PANEL.columns and pd.notna(latest[v]) else None)
               for k, v in mapping.items()}
        },
        "rolling_windows": windows,
        "leadlag_best": best_ll,
        "regression_vars": (REG["var"].tolist() if not REG.empty else []),
        "nowcast_available": (not NC.empty),
        "events_count": (int(EVENTS.shape[0]) if not EVENTS.empty else 0)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        gst=args.gst, eway=(args.eway or None), fastag=(args.fastag or None), freight=(args.freight or None),
        rail=(args.rail or None), diesel=(args.diesel or None), events=(args.events or None),
        freq=args.freq, windows=args.windows, lags=int(args.lags), event_window=int(args.event_window),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Logistics vs GST ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Freq: {summary['freq']}")
    if best_ll:
        for k, st in best_ll.items():
            print(f"Lead–lag {k}: max |corr| at lag {st['lag']:+d} → {st['corr']:+.2f}")
    if not REG.empty:
        disp = ", ".join([f"{r.var}={r.coef:+.2f} (t={r.t_stat:+.2f})" for _, r in REG.iterrows() if r.var!="const"])
        print("Elasticities (Δlog GST on Δlog X):", disp)
    if not NC.empty:
        last = NC.tail(1).iloc[0]
        print(f"Nowcast latest: dlog_gst_hat={last['dlog_gst_hat']:+.3f} → gst_fitted≈{last['gst_fitted']:.0f}")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
