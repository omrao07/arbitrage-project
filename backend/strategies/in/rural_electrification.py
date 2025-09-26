#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rural_electrification.py — Access, reliability, losses & scenarios for rural India
----------------------------------------------------------------------------------

What this does
==============
Given administrative + utility datasets on **rural electrification**, reliability, and
DISCOM health, this script:

1) Cleans & aligns sources to **monthly** or **quarterly** frequency.
2) Builds core metrics:
   • Electrification rate = rural HH electrified / rural HH total
   • Monthly pace & backlog to 100% coverage
   • Supply-hours (rural vs urban), equity gap, SAIFI/SAIDI, outage minutes
   • AT&C loss, billing/collection efficiency, feeder separation %, transformer failures
   • VIIRS night-lights (as an access/usage proxy)
3) Diagnostics:
   • Rolling correlations & lead–lag vs capex, AT&C, outages, night-lights
   • Distributed-lag regressions with HAC (Newey–West) SEs:
       ΔElectrificationRate_t ~ Σ β·drivers_{t−i}
       SAIDI_t ~ Σ β·drivers_{t−i}
4) Reliability Index:
   • Z-score blend of SAIDI↑, SAIFI↑, transformer fail rate↑, equity gap↑ (rural<urban)
   • Early-warning flags for blackout risk (large jumps or > threshold)
5) Scenarios:
   • Universal access in H months: required monthly connections & capex
   • Loss reduction: AT&C −X pp → expected SAIDI change (via regression elasticity)
   • Solarization of ag pumps (feeder load factor ↓L%) → SAIDI improvement (via elasticity)

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--connections connections.csv   REQUIRED
  Columns (any subset):
    date, [state|district|region],
    rural_households_total, rural_households_electrified[, new_connections],
    supply_hours_rural[, supply_hours_urban],
    energy_consumption_kwh (rural total or per-HH)

--reliability reliability.csv   OPTIONAL (utility logs, feeder-level aggregated)
  Columns:
    date[, state|district],
    saidi_minutes[, saifi_count], outage_minutes[, feeder_outages],
    transformer_failures[, customers], feeder_load_factor

--losses losses.csv             OPTIONAL (DISCOM performance/finance)
  Columns:
    date[, state], atc_loss_pct[, billing_eff_pct, collection_eff_pct],
    subsidy_inr[, revenue_gap_inr], capex_inr[, feeder_separation_pct]

--nightlights viirs.csv         OPTIONAL (VIIRS-DNB monthly mean)
  Columns:
    date[, district|state], viirs_dnb_mean

--events events.csv             OPTIONAL (policy/scheme milestones)
  Columns: date, label

Key CLI
-------
--freq monthly|quarterly     Output frequency (default monthly)
--lags 6                     Max lag for lead–lag & regressions
--windows 3,6,12             Rolling windows
--blackout_thresh 1.5        Reliability Index Z-threshold for EWS
--target_months 12           Months to reach 100% electrification in scenario
--cost_per_conn 8000         INR per new connection (capex; user-supplied)
--atc_reduction_pp 5         Scenario: AT&C loss reduction (percentage points)
--solarization_pct 20        Scenario: % drop in feeder load factor
--start / --end              Date filters (YYYY-MM-DD)
--outdir out_power           Output directory
--min_obs 24                 Minimum obs for regressions

Outputs
-------
- panel.csv                  Master panel (levels & transforms)
- region_panel.csv           Region/district panel if provided
- rolling_stats.csv          Rolling corr of ΔElectrificationRate vs drivers
- leadlag_corr.csv           Lead–lag X_{t−k} vs ΔElectrificationRate_t / SAIDI_t
- reg_electrification.csv    D-lag regression for ΔElectrificationRate
- reg_saidi.csv              D-lag regression for SAIDI level
- reliability_index.csv      RI (aggregate & by region) + components
- ews.csv                    Early-warning signals for reliability
- scenarios.csv              Scenario results (universal access, AT&C, solarization)
- summary.json               Headline diagnostics
- config.json                Run configuration

DISCLAIMER
----------
Research tooling with simplifying assumptions. Validate mappings, measurement,
and elasticities before policy or investment decisions.
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
    return (dt.dt.to_period("M").dt.to_timestamp("M") if freq.startswith("M")
            else dt.dt.to_period("Q").dt.to_timestamp("Q"))

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def d(s: pd.Series) -> pd.Series:
    return s.astype(float).diff()

def yoy(s: pd.Series, periods: int) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s) - np.log(s.shift(periods))

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(4, window//2)).corr(y)

def leadlag_corr(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            xx = x.shift(k); yy = y
        else:
            xx = x; yy = y.shift(-k)
        rows.append({"lag": k, "corr": float(xx.corr(yy)) if not (xx.isna().all() or yy.isna().all()) else np.nan})
    return pd.DataFrame(rows)

# OLS + HAC
def ols_beta_se(X: np.ndarray, y: np.ndarray):
    XtX = X.T @ X
    XtY = X.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ XtY
    resid = y - X @ beta
    return beta, resid, XtX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
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

def load_connections(path: str, freq: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], bool]:
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    reg = ncol(df, "district","state","region")
    tot = ncol(df, "rural_households_total","hh_total","households_total")
    ele = ncol(df, "rural_households_electrified","hh_electrified","households_electrified")
    newc= ncol(df, "new_connections","connections_new","new_conn")
    shr = ncol(df, "supply_hours_rural","supply_hours","supply_hours_day_rural","hours_rural")
    shu = ncol(df, "supply_hours_urban","hours_urban")
    en  = ncol(df, "energy_consumption_kwh","consumption_kwh","kwh")
    if not (dt and tot and ele):
        raise ValueError("connections.csv needs date, rural_households_total, rural_households_electrified.")
    df = df.rename(columns={dt:"date", tot:"hh_total", ele:"hh_elec"})
    df["date"] = to_period_end(df["date"], "M" if freq.startswith("m") else "Q")
    for c in ["hh_total","hh_elec", newc, shr, shu, en]:
        if c and c in df.columns:
            df[c if c in ["hh_total","hh_elec"] else c] = safe_num(df[c])
    if newc: df = df.rename(columns={newc:"new_connections"})
    if shr:  df = df.rename(columns={shr:"supply_hours_rural"})
    if shu:  df = df.rename(columns={shu:"supply_hours_urban"})
    if en:   df = df.rename(columns={en:"energy_kwh"})
    if reg:  df = df.rename(columns={reg:"region"})
    # region panel if available
    REG = None
    if reg:
        REG = df.groupby(["date","region"], as_index=False).agg({
            "hh_total":"sum","hh_elec":"sum",
            "new_connections":"sum" if "new_connections" in df.columns else "first",
            "supply_hours_rural":"mean" if "supply_hours_rural" in df.columns else "first",
            "supply_hours_urban":"mean" if "supply_hours_urban" in df.columns else "first",
            "energy_kwh":"sum" if "energy_kwh" in df.columns else "first"
        })
    AGG = df.groupby("date", as_index=False).agg({
        "hh_total":"sum","hh_elec":"sum",
        "new_connections":"sum" if "new_connections" in df.columns else "first",
        "supply_hours_rural":"mean" if "supply_hours_rural" in df.columns else "first",
        "supply_hours_urban":"mean" if "supply_hours_urban" in df.columns else "first",
        "energy_kwh":"sum" if "energy_kwh" in df.columns else "first"
    })
    return AGG.sort_values("date"), (REG.sort_values(["date","region"]) if REG is not None else None), bool(reg)

def load_reliability(path: Optional[str], freq: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not path: return pd.DataFrame(), None
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    reg = ncol(df, "district","state","region")
    sai = ncol(df, "saidi_minutes","saidi","saidi_min")
    saf = ncol(df, "saifi_count","saifi","saifi_events")
    om  = ncol(df, "outage_minutes","outage_min")
    tf  = ncol(df, "transformer_failures","tx_failures")
    cust= ncol(df, "customers","consumers","metered_customers")
    flf = ncol(df, "feeder_load_factor","load_factor")
    if not dt: raise ValueError("reliability.csv needs date.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], "M" if freq.startswith("m") else "Q")
    ren = {}
    if sai: ren[sai] = "saidi"
    if saf: ren[saf] = "saifi"
    if om:  ren[om]  = "outage_minutes"
    if tf:  ren[tf]  = "transformer_failures"
    if cust:ren[cust]= "customers"
    if flf: ren[flf] = "feeder_load_factor"
    df = df.rename(columns=ren)
    for c in [*ren.values()]:
        df[c] = safe_num(df[c])
    if reg: df = df.rename(columns={reg:"region"})
    REG = df.groupby(["date","region"], as_index=False).mean() if reg else None
    AGG = df.groupby("date", as_index=False).mean(numeric_only=True)
    return AGG.sort_values("date"), (REG.sort_values(["date","region"]) if REG is not None else None)

def load_losses(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"}); df["date"] = to_period_end(df["date"], "M" if freq.startswith("m") else "Q")
    maps = [
        (["atc_loss_pct","atc","atc_loss"], "atc_loss_pct"),
        (["billing_eff_pct","billing_efficiency_pct"], "billing_eff_pct"),
        (["collection_eff_pct","collection_efficiency_pct"], "collection_eff_pct"),
        (["subsidy_inr","subsidy"], "subsidy_inr"),
        (["revenue_gap_inr","gap_inr"], "revenue_gap_inr"),
        (["capex_inr","capital_expenditure_inr"], "capex_inr"),
        (["feeder_separation_pct","feeder_sep_pct"], "feeder_separation_pct"),
        (["state","region"], "region")
    ]
    for srcs, out in maps:
        c = None
        for s in srcs:
            c = ncol(df, s) or c
        if c: df.rename(columns={c:out}, inplace=True)
    for c in df.columns:
        if c!="date" and c!="region":
            df[c] = safe_num(df[c])
    return df.sort_values(["date","region"] if "region" in df.columns else "date")

def load_viirs(path: Optional[str], freq: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not path: return pd.DataFrame(), None
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    reg = ncol(df, "district","state","region")
    v   = ncol(df, "viirs_dnb_mean","viirs_mean","night_lights")
    if not (dt and v): raise ValueError("viirs.csv needs date and viirs_dnb_mean.")
    df = df.rename(columns={dt:"date", v:"viirs_dnb_mean"})
    df["date"] = to_period_end(df["date"], "M" if freq.startswith("m") else "Q")
    if reg: df = df.rename(columns={reg:"region"})
    for c in ["viirs_dnb_mean"]:
        df[c] = safe_num(df[c])
    REG = df.groupby(["date","region"], as_index=False)["viirs_dnb_mean"].mean() if "region" in df.columns else None
    AGG = df.groupby("date", as_index=False)["viirs_dnb_mean"].mean()
    return AGG.sort_values("date"), (REG.sort_values(["date","region"]) if REG is not None else None)

def load_events(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    lab = ncol(df, "label","event","name") or "label"
    df = df.rename(columns={dt:"date", lab:"label"})
    df["date"] = to_period_end(df["date"], "M" if freq.startswith("m") else "Q")
    df["label"] = df["label"].astype(str)
    return df[["date","label"]].dropna().sort_values("date")


# ----------------------------- constructions -----------------------------

def build_panels(freq: str,
                 CONN_AGG: pd.DataFrame, CONN_REG: Optional[pd.DataFrame], has_region: bool,
                 REL_AGG: pd.DataFrame, REL_REG: Optional[pd.DataFrame],
                 LOSS: pd.DataFrame, VIIRS_AGG: pd.DataFrame, VIIRS_REG: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Aggregate master
    df = CONN_AGG.copy()
    for d in [REL_AGG, VIIRS_AGG]:
        if not (d is None or d.empty):
            df = df.merge(d, on="date", how="left")
    # region-aware merge for losses (state-level)
    if not LOSS.empty:
        if "region" in LOSS.columns:
            loss_agg = LOSS.groupby("date", as_index=False).mean(numeric_only=True)
        else:
            loss_agg = LOSS
        df = df.merge(loss_agg, on="date", how="left")

    df = df.sort_values("date")
    yo = 12 if freq.startswith("M") else 4

    # Core metrics
    df["elec_rate"] = df["hh_elec"] / df["hh_total"].replace(0, np.nan)
    df["d_elec_rate"] = d(df["elec_rate"])
    df["yoy_elec_rate"] = yoy(df["elec_rate"], yo)
    if "new_connections" in df.columns:
        df["conn_pace"] = df["new_connections"]
    else:
        df["conn_pace"] = df["hh_elec"].diff().clip(lower=0)

    df["backlog_hh"] = (df["hh_total"] - df["hh_elec"]).clip(lower=0)
    df["months_to_100"] = df["backlog_hh"] / df["conn_pace"].replace(0, np.nan)

    # Equity gap (urban - rural supply hours; positive = urban better)
    if "supply_hours_rural" in df.columns:
        if "supply_hours_urban" in df.columns:
            df["equity_gap_hours"] = df["supply_hours_urban"] - df["supply_hours_rural"]
        else:
            df["equity_gap_hours"] = np.nan

    # Reliability transforms
    if "saidi" in df.columns:
        df["d_saidi"] = d(df["saidi"])
    if "saifi" in df.columns:
        df["d_saifi"] = d(df["saifi"])
    if "transformer_failures" in df.columns and "customers" in df.columns:
        df["tx_fail_rate_per_1k"] = (df["transformer_failures"] / df["customers"].replace(0, np.nan)) * 1000.0

    # Losses
    if "atc_loss_pct" in df.columns:
        df["atc_loss_dec"] = df["atc_loss_pct"] / 100.0
        df["d_atc_loss_pp"] = d(df["atc_loss_pct"])

    # VIIRS transforms
    if "viirs_dnb_mean" in df.columns:
        df["dlog_viirs"] = dlog(df["viirs_dnb_mean"])

    # Region panel (if available)
    REG_PANEL = pd.DataFrame()
    if has_region and CONN_REG is not None and not CONN_REG.empty:
        REG_PANEL = CONN_REG.copy()
        for d in [REL_REG, VIIRS_REG]:
            if d is not None and not d.empty:
                REG_PANEL = REG_PANEL.merge(d, on=["date","region"], how="left")
        if not LOSS.empty and "region" in LOSS.columns:
            REG_PANEL = REG_PANEL.merge(LOSS, on=["date","region"], how="left")
        REG_PANEL = REG_PANEL.sort_values(["date","region"])
        REG_PANEL["elec_rate"] = REG_PANEL["hh_elec"] / REG_PANEL["hh_total"].replace(0, np.nan)
        REG_PANEL["d_elec_rate"] = REG_PANEL.groupby("region")["elec_rate"].diff()
        if "supply_hours_rural" in REG_PANEL.columns and "supply_hours_urban" in REG_PANEL.columns:
            REG_PANEL["equity_gap_hours"] = REG_PANEL["supply_hours_urban"] - REG_PANEL["supply_hours_rural"]
        if "transformer_failures" in REG_PANEL.columns and "customers" in REG_PANEL.columns:
            REG_PANEL["tx_fail_rate_per_1k"] = (REG_PANEL["transformer_failures"] / REG_PANEL["customers"].replace(0, np.nan)) * 1000.0
        if "atc_loss_pct" in REG_PANEL.columns:
            REG_PANEL["atc_loss_dec"] = REG_PANEL["atc_loss_pct"] / 100.0
    return df, REG_PANEL

# ----------------------------- diagnostics -----------------------------

def rolling_stats(panel: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    y = idx.get("d_elec_rate")
    if y is None or y.dropna().empty: return pd.DataFrame()
    for key, col in [
        ("capex","capex_inr"),
        ("atc_loss","atc_loss_pct"),
        ("saidi","saidi"),
        ("saifi","saifi"),
        ("equity_gap","equity_gap_hours"),
        ("viirs","dlog_viirs"),
    ]:
        if col in idx.columns:
            x = idx[col]
            for w, tag in zip(windows, ["short","med","long"]):
                rows.append({"driver": key, "column": col, "window": w, "tag": tag,
                             "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan})
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, lags: int) -> pd.DataFrame:
    rows = []
    if "d_elec_rate" in panel.columns:
        y = panel["d_elec_rate"]
        for key in ["capex_inr","atc_loss_pct","saidi","saifi","equity_gap_hours","dlog_viirs"]:
            if key in panel.columns:
                tab = leadlag_corr(panel[key], y, lags)
                tab["driver"] = key; tab["dep"] = "d_elec_rate"
                rows.append(tab)
    if "saidi" in panel.columns:
        y2 = panel["saidi"]
        for key in ["capex_inr","atc_loss_pct","saifi","equity_gap_hours","feeder_load_factor","tx_fail_rate_per_1k"]:
            if key in panel.columns:
                tab = leadlag_corr(panel[key], y2, lags)
                tab["driver"] = key; tab["dep"] = "saidi"
                rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","driver","dep"])

def dlag_regression(panel: pd.DataFrame, dep_col: str, L: int, min_obs: int) -> pd.DataFrame:
    if dep_col not in panel.columns: return pd.DataFrame()
    df = panel.copy()
    dep = dep_col
    Xparts = [pd.Series(1.0, index=df.index, name="const")]
    names = ["const"]
    # choose drivers based on dep
    if dep_col == "d_elec_rate":
        drivers = ["capex_inr","subsidy_inr","atc_loss_pct","saidi","equity_gap_hours","dlog_viirs"]
    else:  # SAIDI
        drivers = ["capex_inr","atc_loss_pct","saifi","feeder_separation_pct","feeder_load_factor","tx_fail_rate_per_1k"]
    avail = []
    for col in drivers:
        if col in df.columns:
            avail.append(col)
            for l in range(0, L+1):
                nm = f"{col}_l{l}"
                Xparts.append(df[col].shift(l).rename(nm)); names.append(nm)
    if not avail: return pd.DataFrame()
    X = pd.concat(Xparts, axis=1)
    XY = pd.concat([df[dep].rename("dep"), X], axis=1).dropna()
    if XY.shape[0] < max(min_obs, 5*X.shape[1]):
        return pd.DataFrame()
    yv = XY["dep"].values.reshape(-1,1)
    Xv = XY.drop(columns=["dep"]).values
    beta, resid, XtX_inv = ols_beta_se(Xv, yv)
    se = hac_se(Xv, resid, XtX_inv, L=max(6, L))
    out = []
    for i, nm in enumerate(names):
        out.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                    "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "lags": L, "dep": dep})
    # cumulative effects per driver
    for col in avail:
        idxs = [i for i, nm in enumerate(names) if nm.startswith(f"{col}_l")]
        if idxs:
            bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
            out.append({"var": f"{col}_cum_0..L", "coef": bsum, "se": ses,
                        "t_stat": bsum/(ses if ses>0 else np.nan), "lags": L, "dep": dep})
    return pd.DataFrame(out)

# ----------------------------- Reliability Index & EWS -----------------------------

def zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = x.mean(); sd = x.std(ddof=0)
    return (x - mu) / (sd if (sd and sd==sd) else np.nan)

def reliability_index(panel: pd.DataFrame, reg_panel: pd.DataFrame, thresh: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    RI = mean_z( +SAIDI, +SAIFI, +tx_fail_rate_per_1k, +equity_gap_hours )
    """
    def compute(df: pd.DataFrame, by_region=False) -> pd.DataFrame:
        cols = {}
        if "saidi" in df.columns: cols["z_saidi"] = zscore(df["saidi"])
        if "saifi" in df.columns: cols["z_saifi"] = zscore(df["saifi"])
        if "tx_fail_rate_per_1k" in df.columns: cols["z_tx"] = zscore(df["tx_fail_rate_per_1k"])
        if "equity_gap_hours" in df.columns: cols["z_eqgap"] = zscore(df["equity_gap_hours"])
        if not cols: return pd.DataFrame()
        Z = pd.DataFrame(cols)
        out = pd.DataFrame({"date": df["date"]})
        if by_region:
            Z = Z.groupby(df["region"]).transform(lambda s: zscore(s))
            out["region"] = df["region"]
        out["RI"] = Z.mean(axis=1, skipna=True)
        return out

    agg = compute(panel, by_region=False)
    ews = pd.DataFrame()
    if not agg.empty:
        agg = agg.sort_values("date")
        agg["d_RI"] = agg["RI"].diff()
        P90 = np.nanpercentile(agg["RI"], 90) if agg["RI"].notna().sum()>=10 else np.nan
        rows = []
        for _, r in agg.iterrows():
            cond1 = pd.notna(r["RI"]) and r["RI"] >= thresh
            cond2 = pd.notna(P90) and pd.notna(r["RI"]) and r["RI"] >= P90
            cond3 = pd.notna(r["d_RI"]) and r["d_RI"] > 0.5
            if cond1 or cond2 or cond3:
                rows.append({"date": r["date"], "RI": r["RI"], "trigger_thresh": cond1, "trigger_p90": cond2, "trigger_momentum": cond3})
        ews = pd.DataFrame(rows).sort_values("date")
    reg = pd.DataFrame()
    if reg_panel is not None and not reg_panel.empty:
        reg = compute(reg_panel, by_region=True).sort_values(["date","region"])
    RI = pd.concat([agg, reg], ignore_index=True, sort=False) if not agg.empty or not reg.empty else pd.DataFrame()
    return RI, ews

# ----------------------------- Scenarios -----------------------------

def scenario_universal_access(panel: pd.DataFrame, target_months: int, cost_per_conn: float) -> Dict:
    last = panel.dropna(subset=["hh_total","hh_elec"]).tail(1)
    if last.empty:
        return {"note":"no household totals/electrified data"}
    r = last.iloc[0]
    backlog = float(max(0.0, r["hh_total"] - r["hh_elec"]))
    pace_curr = float(max(0.0, panel["conn_pace"].tail(6).mean())) if "conn_pace" in panel.columns else np.nan
    pace_req = backlog / max(1, target_months)
    extra = max(0.0, pace_req - (pace_curr if pace_curr==pace_curr else 0.0))
    capex_total = backlog * float(cost_per_conn)
    return {
        "as_of": str(r["date"].date()),
        "backlog_hh": backlog,
        "current_pace_hh_per_m": pace_curr if pace_curr==pace_curr else None,
        "required_pace_hh_per_m": pace_req,
        "extra_pace_needed_hh_per_m": extra,
        "capex_total_inr": capex_total
    }

def extract_cum_coef(reg: pd.DataFrame, label: str) -> Optional[float]:
    if reg.empty: return None
    r = reg[reg["var"] == f"{label}_cum_0..L"]
    return float(r["coef"].iloc[0]) if not r.empty else None

def scenario_losses_solar(reg_saidi: pd.DataFrame, atc_reduction_pp: float, solar_pct: float) -> Dict:
    """
    Use regression elasticities to approximate SAIDI change.
    ΔSAIDI ≈ β_atc * (−atc_reduction_pp) + β_load * (−solar_pct)
    where β_load uses feeder_load_factor cumulative coefficient (per 1.0 change).
    """
    beta_atc = extract_cum_coef(reg_saidi, "atc_loss_pct")
    beta_load= extract_cum_coef(reg_saidi, "feeder_load_factor")
    d_saidi_atc = (beta_atc * (-atc_reduction_pp)) if beta_atc is not None else None
    d_saidi_solar= (beta_load * (-solar_pct/100.0)) if beta_load is not None else None
    eff = {}
    if d_saidi_atc is not None: eff["delta_saidi_atc_minutes"] = d_saidi_atc
    if d_saidi_solar is not None: eff["delta_saidi_solar_minutes"] = d_saidi_solar
    if not eff:
        eff["note"] = "regression elasticities unavailable"
    return eff

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    connections: str
    reliability: Optional[str]
    losses: Optional[str]
    nightlights: Optional[str]
    events: Optional[str]
    freq: str
    lags: int
    windows: str
    blackout_thresh: float
    target_months: int
    cost_per_conn: float
    atc_reduction_pp: float
    solarization_pct: float
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rural electrification: access, reliability, losses & scenarios")
    ap.add_argument("--connections", required=True)
    ap.add_argument("--reliability", default="")
    ap.add_argument("--losses", default="")
    ap.add_argument("--nightlights", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","quarterly"])
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--blackout_thresh", type=float, default=1.5)
    ap.add_argument("--target_months", type=int, default=12)
    ap.add_argument("--cost_per_conn", type=float, default=8000.0)
    ap.add_argument("--atc_reduction_pp", type=float, default=5.0)
    ap.add_argument("--solarization_pct", type=float, default=20.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_power")
    ap.add_argument("--min_obs", type=int, default=24)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq = "M" if args.freq.startswith("m") else "Q"

    CONN_AGG, CONN_REG, has_region = load_connections(args.connections, freq=freq)
    REL_AGG, REL_REG = load_reliability(args.reliability, freq=freq) if args.reliability else (pd.DataFrame(), None)
    LOSS    = load_losses(args.losses, freq=freq) if args.losses else pd.DataFrame()
    VIIRS_AGG, VIIRS_REG = load_viirs(args.nightlights, freq=freq) if args.nightlights else (pd.DataFrame(), None)
    EVTS    = load_events(args.events, freq=freq) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        t0 = pd.to_datetime(args.start)
        for df in [CONN_AGG, CONN_REG, REL_AGG, REL_REG, LOSS, VIIRS_AGG, VIIRS_REG]:
            if df is not None and not df.empty:
                df.drop(df[df["date"] < t0].index, inplace=True)
    if args.end:
        t1 = pd.to_datetime(args.end)
        for df in [CONN_AGG, CONN_REG, REL_AGG, REL_REG, LOSS, VIIRS_AGG, VIIRS_REG]:
            if df is not None and not df.empty:
                df.drop(df[df["date"] > t1].index, inplace=True)

    PANEL, REGION_PANEL = build_panels(freq, CONN_AGG, CONN_REG, has_region, REL_AGG, REL_REG, LOSS, VIIRS_AGG, VIIRS_REG)
    PANEL.to_csv(outdir / "panel.csv", index=False)
    if not REGION_PANEL.empty:
        REGION_PANEL.to_csv(outdir / "region_panel.csv", index=False)

    # Rolling & lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, windows)
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)
    LL = leadlag_tables(PANEL, lags=int(args.lags))
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Regressions
    REG_E = dlag_regression(PANEL, dep_col="d_elec_rate", L=int(args.lags), min_obs=int(args.min_obs))
    if not REG_E.empty: REG_E.to_csv(outdir / "reg_electrification.csv", index=False)
    # SAIDI regression needs SAIDI
    REG_S = dlag_regression(PANEL, dep_col="saidi", L=int(args.lags), min_obs=int(args.min_obs)) if "saidi" in PANEL.columns else pd.DataFrame()
    if not REG_S.empty: REG_S.to_csv(outdir / "reg_saidi.csv", index=False)

    # Reliability Index & EWS
    RI, EWS = reliability_index(PANEL, REGION_PANEL, thresh=float(args.blackout_thresh))
    if not RI.empty: RI.to_csv(outdir / "reliability_index.csv", index=False)
    if not EWS.empty: EWS.to_csv(outdir / "ews.csv", index=False)

    # Scenarios
    scn_access = scenario_universal_access(PANEL, target_months=int(args.target_months), cost_per_conn=float(args.cost_per_conn))
    scn_loss_solar = scenario_losses_solar(REG_S if not REG_S.empty else pd.DataFrame(),
                                           atc_reduction_pp=float(args.atc_reduction_pp),
                                           solar_pct=float(args.solarization_pct))
    SCN_ROWS = []
    if scn_access:
        SCN_ROWS.append({"scenario":"universal_access", **{k:v for k,v in scn_access.items()}})
    if scn_loss_solar:
        SCN_ROWS.append({"scenario":"atc_reduction", **{k:v for k,v in scn_loss_solar.items() if k.startswith("delta_saidi_atc")}})
        SCN_ROWS.append({"scenario":"solarization", **{k:v for k,v in scn_loss_solar.items() if k.startswith("delta_saidi_solar")}})
    SCN = pd.DataFrame(SCN_ROWS)
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    latest = PANEL.tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "freq": args.freq,
        "latest": {
            "date": str(latest["date"].date()),
            "elec_rate": float(latest.get("elec_rate", np.nan)) if pd.notna(latest.get("elec_rate", np.nan)) else None,
            "backlog_hh": float(latest.get("hh_total", np.nan) - latest.get("hh_elec", np.nan)) if pd.notna(latest.get("hh_total", np.nan)) and pd.notna(latest.get("hh_elec", np.nan)) else None,
            "saidi": float(latest.get("saidi", np.nan)) if "saidi" in PANEL.columns and pd.notna(latest.get("saidi", np.nan)) else None,
            "equity_gap_hours": float(latest.get("equity_gap_hours", np.nan)) if "equity_gap_hours" in PANEL.columns and pd.notna(latest.get("equity_gap_hours", np.nan)) else None,
        },
        "rolling_windows": windows,
        "has_region": has_region,
        "ews_latest": (EWS.tail(1).to_dict(orient="records")[0] if not EWS.empty else None),
        "reg_electrification_terms": REG_E["var"].tolist() if not REG_E.empty else [],
        "reg_saidi_terms": REG_S["var"].tolist() if not REG_S.empty else [],
        "scenarios_included": SCN["scenario"].unique().tolist() if not SCN.empty else []
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        connections=args.connections, reliability=(args.reliability or None), losses=(args.losses or None),
        nightlights=(args.nightlights or None), events=(args.events or None),
        freq=args.freq, lags=int(args.lags), windows=args.windows,
        blackout_thresh=float(args.blackout_thresh), target_months=int(args.target_months),
        cost_per_conn=float(args.cost_per_conn), atc_reduction_pp=float(args.atc_reduction_pp),
        solarization_pct=float(args.solarization_pct), start=(args.start or None), end=(args.end or None),
        outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Rural Electrification ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Freq: {summary['freq']}")
    if summary["latest"]["elec_rate"] is not None:
        print(f"Latest electrification rate: {summary['latest']['elec_rate']*100:.1f}% | Backlog HH: {summary['latest']['backlog_hh']:,}")
    if summary["latest"]["saidi"] is not None:
        print(f"Latest SAIDI: {summary['latest']['saidi']:.0f} minutes")
    if summary["latest"]["equity_gap_hours"] is not None:
        print(f"Equity gap (urban−rural supply hours): {summary['latest']['equity_gap_hours']:+.2f}")
    if "universal_access" in summary["scenarios_included"]:
        print("Scenario: universal access computed (see scenarios.csv).")
    if "atc_reduction" in summary["scenarios_included"] or "solarization" in summary["scenarios_included"]:
        print("Scenario: SAIDI deltas from AT&C reduction / solarization estimated.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
import argparse
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path        