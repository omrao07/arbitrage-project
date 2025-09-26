#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hydropower_vs_drought.py — Quantify how drought affects hydropower output
--------------------------------------------------------------------------

What this does
==============
Given hydropower generation (plant- or region-level) and drought/hydrology
indices, this script:

1) Cleans and aggregates everything to **monthly** frequency by a chosen key
   (region, basin, or "ALL").
2) Builds core transforms per group:
   • Capacity factor (CF) if capacities are provided
   • Seasonal anomalies (vs 2015–2019 or user baseline)
   • Δlog of generation/CF (≈ % m/m)
3) Rolling diagnostics:
   • Corr( drought , Δlog(CF) ) and β( Δlog(CF) on drought ) for short/med/long
4) Lead–lag tables:
   • Corr( drought_{t−k} , Δlog(CF)_t ) for k ∈ [−L..L]
5) Distributed-lag regression (local projection style):
   • Δlog(CF)_t  on [drought_{t},…,drought_{t−H}] + controls (precip, temp, inflow, storage)
     → elasticities and HAC (Newey–West) t-stats
6) “Drought events”:
   • Months crossing severity thresholds (e.g., SPI < −1.5 or PDSI < −3)
     and realized CF anomalies around each episode.
7) Scenario stress:
   • Apply user scenarios (severity & duration per group) to estimate expected
     generation impact using fitted distributed-lag coefficients.

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--gen gen.csv                REQUIRED   (plant- or region-level)
  Columns (any subset): date, plant_id, region, basin, generation_mwh
  If daily/weekly, we sum to monthly.

--capacity capacity.csv      OPTIONAL   (needed for capacity factor)
  Columns: plant_id [, region|basin], capacity_mw
  If plant-level gen is provided, we aggregate capacity to chosen key.
  If region-level gen: either provide capacity at same key, or omit CF.

--drought drought.csv        REQUIRED
  Columns: date, <key>, <one or more drought metrics>
  Examples of drought metrics: spi, spei, pdsi, sgi, soil_moist_anom
  <key> should match your --group_key (region|basin). If omitted, we treat as ALL.

--hydrology hydro.csv        OPTIONAL
  Columns: date, <key>, inflow, storage, runoff, discharge, snow, …
  (Any numeric columns are considered candidate controls.)

--climate climate.csv        OPTIONAL
  Columns: date, <key>, precip, tmean, tmax, tmin, snow, …
  (Any numeric columns become candidate controls.)

--scenarios scenarios.csv    OPTIONAL (stress testing)
  Columns: scenario, key, value [, group]
  Examples:
    drought.var=spi
    drought.severity=-2.0          # value to apply (e.g., SPI = -2)
    horizon_months=6
    group=California               # optional: scenario for a specific group
    apply_as="level"               # "level" or "z" (if drought z-score provided)

Key CLI parameters
------------------
--group_key region|basin|ALL   How to aggregate (default: region if present else ALL)
--drought_var spi              Which drought column to consider the "main" shock
--windows 6,12,24              Rolling windows (months) for stats
--lags 12                      Max lead/lag for correlation tables (months)
--h 12                         Horizon for distributed-lag (number of lags of drought)
--controls yes/no              Include hydro/climate controls (default yes)
--base_start 2015-01 --base_end 2019-12   Seasonal baseline period for anomalies
--start YYYY-MM --end YYYY-MM  Date filters
--outdir out_hydro_drought     Output directory

Outputs
-------
- panel_monthly.csv        Per month×group: levels, CF, anomalies, Δlogs, drought & controls
- rolling_stats.csv        Rolling corr & beta between drought and Δlog(CF)
- leadlag_corr.csv         Corr(drought_{t−k}, Δlog(CF)_t) table across lags
- dlm_coeffs.csv           Distributed-lag coefficients (lag 0..H) with HAC SE & t
- events.csv               Drought episodes with CF anomaly impacts
- scenario_results.csv     Expected CF/gen impact under user scenarios (if provided)
- summary.json             Headline diagnostics (date range, latest, best lags, elasticities)
- config.json              Run configuration for reproducibility

Notes
-----
• If capacities are missing, CF is omitted and Δlog(Generation) is used instead.
• Positive drought indexes can mean “wetter” (e.g., SPI) while PDSI more negative is drier.
  Interpret signs accordingly; the model is agnostic and infers from data.
• HAC standard errors use Bartlett kernel with lag = max(lag_count, 6).

DISCLAIMER
----------
Research tool with simplifying assumptions; validate before operational use.
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
    for cand in cands:
        t = cand.lower()
        for c in df.columns:
            if t in str(c).lower(): return c
    return None

def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def month_hours(idx: pd.DatetimeIndex) -> pd.Series:
    # Hours in month: days_in_month * 24
    ser = pd.Series(idx.days_in_month * 24.0, index=idx)
    ser.index = idx
    return ser

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def seasonal_anomaly(series: pd.Series, base_mask: pd.Series) -> pd.Series:
    """Remove monthly seasonal mean computed over baseline period."""
    df = pd.DataFrame({"val": series, "month": series.index.month})
    base = df[base_mask.reindex(series.index, fill_value=False)]
    mavg = base.groupby("month")["val"].mean()
    return df.apply(lambda r: r["val"] - mavg.get(r["month"], np.nan), axis=1).rename(series.name)

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

def leadlag_corr_table(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            xv = x.shift(k); yv = y
        else:
            xv = x; yv = y.shift(-k)
        c = xv.corr(yv)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

def ols_beta(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    XtX = X.T @ X
    XtY = X.T @ y
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ XtY
    resid = y - X @ beta
    return beta, resid, XtX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, L: int) -> np.ndarray:
    """Newey–West HAC SE with Bartlett kernel."""
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1)+1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    return se


# ----------------------------- loaders & aggregation -----------------------------

def load_gen(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month(df["date"])
    gen_c = ncol(df, "generation_mwh", "gen_mwh", "mwh", "generation", "energy_mwh")
    if not gen_c:
        raise ValueError("gen.csv must include a generation column (e.g., generation_mwh).")
    df = df.rename(columns={gen_c:"gen_mwh"})
    # optional identifiers
    if ncol(df, "plant_id"):
        df = df.rename(columns={ncol(df,"plant_id"):"plant_id"})
    if ncol(df, "region"):
        df = df.rename(columns={ncol(df,"region"):"region"})
    if ncol(df, "basin"):
        df = df.rename(columns={ncol(df,"basin"):"basin"})
    df["gen_mwh"] = safe_num(df["gen_mwh"])
    # monthly sum
    grp = []
    for k in ["plant_id","region","basin"]:
        if k in df.columns: grp.append(k)
    df = df.groupby(["date"] + grp, dropna=False)["gen_mwh"].sum(min_count=1).reset_index()
    return df

def load_capacity(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    if ncol(df, "plant_id"):
        df = df.rename(columns={ncol(df,"plant_id"):"plant_id"})
    if ncol(df, "region"):
        df = df.rename(columns={ncol(df,"region"):"region"})
    if ncol(df, "basin"):
        df = df.rename(columns={ncol(df,"basin"):"basin"})
    cap_c = ncol(df, "capacity_mw", "mw", "nameplate_mw")
    if not cap_c:
        raise ValueError("capacity.csv must include capacity_mw (or mw/nameplate_mw).")
    df = df.rename(columns={cap_c:"capacity_mw"})
    df["capacity_mw"] = safe_num(df["capacity_mw"])
    return df

def load_keyed(path: Optional[str], name: str) -> pd.DataFrame:
    """Generic loader for drought/hydro/climate; keeps all numeric columns."""
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month(df["date"])
    if ncol(df, "region"):
        df = df.rename(columns={ncol(df,"region"):"region"})
    if ncol(df, "basin"):
        df = df.rename(columns={ncol(df,"basin"):"basin"})
    # keep numeric features
    keep = ["date"] + [k for k in ["region","basin"] if k in df.columns]
    keep += [c for c in df.columns if c not in keep and pd.api.types.is_numeric_dtype(df[c])]
    df = df[keep]
    # monthly: mean
    grp = ["date"] + [k for k in ["region","basin"] if k in df.columns]
    df = df.groupby(grp, dropna=False).mean(numeric_only=True).reset_index()
    return df

def pick_group_key(gen: pd.DataFrame, group_key: str) -> str:
    if group_key and group_key.upper() != "ALL":
        return group_key.lower()
    # auto: prefer region then basin, else ALL
    for k in ["region","basin"]:
        if k in gen.columns and gen[k].notna().any():
            return k
    return "ALL"

def aggregate_by_key(gen: pd.DataFrame, cap: pd.DataFrame, drought: pd.DataFrame,
                     hydro: pd.DataFrame, climate: pd.DataFrame, key: str) -> pd.DataFrame:
    # aggregate gen to key
    g = gen.copy()
    if key != "ALL":
        if key not in g.columns:
            raise ValueError(f"--group_key={key} but '{key}' not present in gen.csv")
        grp_cols = ["date", key]
        g = g.groupby(grp_cols, dropna=False)["gen_mwh"].sum(min_count=1).reset_index()
    else:
        g[key] = "ALL"
        g = g.groupby(["date", key], dropna=False)["gen_mwh"].sum(min_count=1).reset_index()

    # capacity at key (if provided)
    cap_key = None
    if not cap.empty:
        c = cap.copy()
        if key == "ALL":
            c[key] = "ALL"
            c = c.groupby([key], dropna=False)["capacity_mw"].sum(min_count=1).reset_index()
        else:
            if key not in c.columns and "plant_id" in cap.columns and key in gen.columns:
                # try: map plant → key from gen (last observation)
                map_df = gen.dropna(subset=[key]).drop_duplicates(subset=["plant_id"])[["plant_id", key]]
                c = c.merge(map_df, on="plant_id", how="left")
            if key not in c.columns:
                raise ValueError(f"capacity.csv lacks '{key}' and cannot infer from plant_id.")
            c = c.groupby([key], dropna=False)["capacity_mw"].sum(min_count=1).reset_index()
        cap_key = c

    # join drought/hydro/climate
    out = g.copy()
    for df in [drought, hydro, climate]:
        if df.empty: continue
        D = df.copy()
        if key not in D.columns:
            D[key] = "ALL"
        out = out.merge(D, on=["date", key], how="left")

    # attach capacity (static)
    if cap_key is not None:
        out = out.merge(cap_key, on=[key], how="left")

    return out.rename(columns={key:"group"})


# ----------------------------- core computation -----------------------------

def build_panel(DF: pd.DataFrame, drought_var: str,
                base_start: Optional[str], base_end: Optional[str]) -> Tuple[pd.DataFrame, bool, str]:
    """
    Compute CF (if capacity available), Δlogs, anomalies; ensure drought_var present.
    Returns (panel, use_cf, drought_var_name)
    """
    df = DF.sort_values(["group","date"]).copy()
    # Hours per month & CF
    has_cap = "capacity_mw" in df.columns and df["capacity_mw"].notna().any()
    if has_cap:
        # expand capacity per month per group
        cap_g = df[["group","capacity_mw"]].drop_duplicates(subset=["group"]).set_index("group")["capacity_mw"]
        df["hours"] = month_hours(df["date"])
        df["cf"] = df["gen_mwh"] / (df["hours"] * df["capacity_mw"])
        df["cf"] = df["cf"].clip(lower=0)
        target = "cf"
    else:
        target = "gen_mwh"

    # Baseline seasonal anomaly (2015–2019 by default)
    if base_start:
        bs = pd.to_datetime(base_start).to_period("M").to_timestamp("M")
    else:
        bs = pd.to_datetime("2015-01").to_period("M").to_timestamp("M")
    if base_end:
        be = pd.to_datetime(base_end).to_period("M").to_timestamp("M")
    else:
        be = pd.to_datetime("2019-12").to_period("M").to_timestamp("M")

    out_frames = []
    for g, sub in df.groupby("group"):
        sub = sub.set_index("date").sort_index()
        base_mask = (sub.index >= bs) & (sub.index <= be)
        sub[f"{target}_anom"] = seasonal_anomaly(sub[target], base_mask)
        sub[f"dlog_{target}"] = dlog(sub[target])
        out_frames.append(sub.reset_index())
    panel = pd.concat(out_frames, ignore_index=True)

    # Verify drought var
    dv = ncol(panel, drought_var) or drought_var
    if dv not in panel.columns:
        # try common aliases
        dv = ncol(panel, "spi", "spei", "pdsi", "soil_moist_anom", "sgi")
        if not dv:
            raise ValueError(f"Drought variable '{drought_var}' not found in merged data.")
    panel = panel.rename(columns={dv:"drought"})
    use_cf = (target == "cf")

    return panel, use_cf, ("cf" if use_cf else "gen_mwh")

def rolling_stats(panel: pd.DataFrame, windows: List[int], use_cf: bool) -> pd.DataFrame:
    df = panel.copy().set_index("date")
    y = df["dlog_cf" if use_cf else "dlog_gen_mwh"] if ("dlog_cf" in df.columns or "dlog_gen_mwh" in df.columns) \
        else (dlog(df["cf" if use_cf else "gen_mwh"]))
    rows = []
    for g, sub in df.groupby("group"):
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append(pd.DataFrame({
                "date": sub.index,
                "group": g,
                f"corr_{tag}": roll_corr(sub["drought"], y.loc[sub.index], w).values,
                f"beta_{tag}": roll_beta(y.loc[sub.index], sub["drought"], w).values
            }).set_index("date"))
    out = (pd.concat(rows, axis=0).reset_index().sort_values(["group","date"]))
    return out

def leadlag(panel: pd.DataFrame, lags: int, use_cf: bool) -> pd.DataFrame:
    df = panel.copy().set_index("date")
    y = df["dlog_cf" if use_cf else "dlog_gen_mwh"] if ("dlog_cf" in df.columns or "dlog_gen_mwh" in df.columns) \
        else (dlog(df["cf" if use_cf else "gen_mwh"]))
    rows = []
    for g, sub in df.groupby("group"):
        tab = leadlag_corr_table(sub["drought"], y.loc[sub.index], lags)
        tab["group"] = g
        rows.append(tab)
    return pd.concat(rows, ignore_index=True)

def distributed_lag(panel: pd.DataFrame, H: int, use_controls: bool, use_cf: bool) -> pd.DataFrame:
    """
    Δlog(target)_t = a + Σ_{h=0..H} b_h * drought_{t-h} + controls_t + ε
    HAC SE with L = max(H, 6). Returns one block per group.
    """
    df = panel.copy().set_index("date").sort_index()
    target_col = "dlog_cf" if use_cf else "dlog_gen_mwh"
    if target_col not in df.columns:
        df[target_col] = dlog(df["cf" if use_cf else "gen_mwh"])
    results = []

    # Determine control candidates (contemporaneous levels or anomalies)
    ctrl_cols = []
    if use_controls:
        for c in df.columns:
            if c in ["gen_mwh","cf","drought",target_col,"group","hours","capacity_mw",
                     "cf_anom","gen_mwh_anom","dlog_cf","dlog_gen_mwh"]:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                ctrl_cols.append(c)

    for g, sub in df.groupby("group"):
        X_parts = [pd.Series(1.0, index=sub.index, name="const")]
        # lagged drought
        for h in range(0, H+1):
            X_parts.append(sub["drought"].shift(h).rename(f"drought_l{h}"))
        # controls
        if use_controls and ctrl_cols:
            for c in ctrl_cols:
                X_parts.append(sub[c].rename(f"{c}_t"))

        X = pd.concat(X_parts, axis=1)
        Y = sub[target_col]
        XY = pd.concat([Y.rename("y"), X], axis=1).dropna()
        if XY.shape[0] < max(36, 5 * (H+1)):
            continue
        yv = XY["y"].values.reshape(-1,1)
        Xv = XY.drop(columns=["y"]).values
        beta, resid, XtX_inv = ols_beta(Xv, yv)
        se = hac_se(Xv, resid, XtX_inv, L=max(H, 6))
        names = XY.drop(columns=["y"]).columns.tolist()
        for name, b, s in zip(names, beta.ravel(), se):
            if name.startswith("drought_l"):
                lag = int(name.split("l")[1])
                tstat = (b / s) if s > 0 else np.nan
                results.append({"group": g, "lag": lag, "coef": float(b), "se": float(s), "t_stat": float(tstat)})
    return pd.DataFrame(results).sort_values(["group","lag"])

def detect_events(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Flag drought episodes using common rules:
      SPI/SPEI <= -1.5 OR PDSI <= -3 (if column exists).
    For generic 'drought' column, use <= p10 of its history per group.
    Returns windowed impacts on CF anomalies (t-1..t+3).
    """
    df = panel.copy().set_index("date").sort_index()
    rows = []
    for g, sub in df.groupby("group"):
        x = sub["drought"]
        # threshold
        thr = np.nanpercentile(x, 10)
        ev_idx = sub.index[x <= thr]
        for d in ev_idx:
            win = sub.loc[(sub.index >= d - pd.offsets.MonthEnd(1)) &
                          (sub.index <= d + pd.offsets.MonthEnd(3))]
            if win.empty: continue
            cf_anom = win["cf_anom"] if "cf_anom" in win.columns else win["gen_mwh_anom"]
            rows.append({
                "group": g, "event_date": d,
                "threshold": float(thr),
                "n_win_months": int(win.shape[0]),
                "anom_t": float(cf_anom.loc[d]) if d in cf_anom.index else np.nan,
                "anom_t1": float(cf_anom.shift(-1).loc[d]) if d in cf_anom.index else np.nan,
                "anom_t2": float(cf_anom.shift(-2).loc[d]) if d in cf_anom.index else np.nan,
                "anom_t3": float(cf_anom.shift(-3).loc[d]) if d in cf_anom.index else np.nan
            })
    return pd.DataFrame(rows).sort_values(["group","event_date"])

def run_scenarios(panel: pd.DataFrame, dlm: pd.DataFrame, scenarios: pd.DataFrame,
                  use_cf: bool) -> pd.DataFrame:
    """
    For each scenario, apply severity at lags 0..H and multiply by coefficients for that group.
    Scenario keys:
      drought.var          (ignored here, we already have 'drought')
      drought.severity     (numeric level to apply per month)
      horizon_months       (int; how many months to simulate)
      group                (optional; else applies to all groups)
      apply_as             ("level" or "z") — treated as level either way (user should ensure scale)
    Returns expected Δlog(CF) per month; we also convert to CF % change approximations.
    """
    if dlm.empty or scenarios.empty:
        return pd.DataFrame()
    out_rows = []
    # reshape dlm to wide by lag per group
    H = int(dlm["lag"].max()) if not dlm.empty else 0
    for sc_name, grp in scenarios.groupby("scenario"):
        kv = dict(zip(grp["key"].astype(str), grp["value"]))
        sev = float(kv.get("drought.severity", -2.0))
        horiz = int(kv.get("horizon_months", H+1))
        target_group = kv.get("group", None)

        for g, dsub in dlm.groupby("group"):
            if target_group and str(target_group) != str(g):
                continue
            coefs = dsub.set_index("lag")["coef"]
            # simulate Δlog(target) each month as b0*sev + b1*sev (carry over) ... up to horizon
            # If severity is sustained across horizon months, Δlog each month equals sum of overlapping lags
            for t in range(horiz):
                # lags impacting month t: 0..min(H,t)
                eff = 0.0
                for h in range(0, min(H, t)+1):
                    eff += float(coefs.get(h, 0.0)) * sev
                out_rows.append({"scenario": sc_name, "group": g, "month": t,
                                 "expected_dlog": eff, "assumed_severity": sev})
    out = pd.DataFrame(out_rows)
    if out.empty: return out
    # Convert to % change ~ 100 * Δlog
    out["expected_pct"] = out["expected_dlog"] * 100.0
    # Approximate CF level change from baseline: if current CF available, we provide hint at t=0
    base = (panel.groupby("group")[("cf" if use_cf else "gen_mwh")].last()
                .rename("last_level")).reset_index()
    out = out.merge(base, on="group", how="left")
    return out


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    gen: str
    capacity: Optional[str]
    drought: str
    hydro: Optional[str]
    climate: Optional[str]
    scenarios: Optional[str]
    group_key: str
    drought_var: str
    windows: str
    lags: int
    H: int
    controls: bool
    base_start: Optional[str]
    base_end: Optional[str]
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hydropower vs Drought analytics")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--capacity", default="")
    ap.add_argument("--drought", required=True)
    ap.add_argument("--hydro", default="")
    ap.add_argument("--climate", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--group_key", default="", help="region|basin|ALL (auto if blank)")
    ap.add_argument("--drought_var", default="spi", help="Column to use as main drought variable")
    ap.add_argument("--windows", default="6,12,24", help="Rolling windows (months)")
    ap.add_argument("--lags", type=int, default=12, help="Max lead/lag for correlation table")
    ap.add_argument("--H", type=int, default=12, help="Distributed-lag horizon (lags of drought)")
    ap.add_argument("--controls", default="yes", choices=["yes","no"])
    ap.add_argument("--base_start", default="2015-01")
    ap.add_argument("--base_end", default="2019-12")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_hydro_drought")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    GEN  = load_gen(args.gen)
    CAP  = load_capacity(args.capacity) if args.capacity else pd.DataFrame()
    DRO  = load_keyed(args.drought, "drought")
    HYD  = load_keyed(args.hydro, "hydrology") if args.hydro else pd.DataFrame()
    CLIM = load_keyed(args.climate, "climate") if args.climate else pd.DataFrame()

    # Date filters (apply early)
    if args.start:
        start = pd.to_datetime(args.start).to_period("M").to_timestamp("M")
        for df in [GEN, DRO, HYD, CLIM]:
            if not df.empty: df.drop(df[df["date"] < start].index, inplace=True)
    if args.end:
        end = pd.to_datetime(args.end).to_period("M").to_timestamp("M")
        for df in [GEN, DRO, HYD, CLIM]:
            if not df.empty: df.drop(df[df["date"] > end].index, inplace=True)

    # Group key
    gkey = pick_group_key(GEN, args.group_key)
    DATA = aggregate_by_key(GEN, CAP, DRO, HYD, CLIM, gkey)

    # Build panel
    panel, use_cf, target_name = build_panel(DATA, drought_var=args.drought_var,
                                             base_start=(args.base_start or None),
                                             base_end=(args.base_end or None))
    if panel["date"].nunique() < 24:
        raise ValueError("Insufficient monthly history after alignment (need ≥24 months).")
    panel = panel.sort_values(["group","date"])
    # Save panel
    panel.to_csv(outdir / "panel_monthly.csv", index=False)

    # Rolling stats
    windows = [int(w.strip()) for w in args.windows.split(",") if w.strip()]
    ROLL = rolling_stats(panel, windows, use_cf)
    ROLL.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag
    LL = leadlag(panel, lags=int(args.lags), use_cf=use_cf)
    LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Distributed-lag regression
    DLM = distributed_lag(panel, H=int(args.H), use_controls=(args.controls=="yes"), use_cf=use_cf)
    if not DLM.empty:
        DLM.to_csv(outdir / "dlm_coeffs.csv", index=False)

    # Events
    EV = detect_events(panel)
    if not EV.empty:
        EV.to_csv(outdir / "events.csv", index=False)

    # Scenarios
    SC = pd.read_csv(args.scenarios) if args.scenarios else pd.DataFrame()
    if not SC.empty:
        req = {"scenario","key","value"}
        if not req.issubset(set(SC.columns)):
            raise ValueError("scenarios.csv must have columns: scenario,key,value[,group]")
        SC_RES = run_scenarios(panel, DLM, SC, use_cf=use_cf)
        if not SC_RES.empty:
            SC_RES.to_csv(outdir / "scenario_results.csv", index=False)

    # Summary
    first, last = str(panel["date"].min().date()), str(panel["date"].max().date())
    # Best lead/lag per group
    best_ll = {}
    if not LL.empty:
        for g, sub in LL.groupby("group"):
            sub = sub.dropna(subset=["corr"])
            if sub.empty: continue
            i = sub["corr"].abs().idxmax()
            row = sub.loc[i]
            best_ll[str(g)] = {"lag": int(row["lag"]), "corr": float(row["corr"])}
    # Latest snapshot
    latest = panel.groupby("group").tail(1)
    latest_dict = {}
    for _, r in latest.iterrows():
        latest_dict[str(r["group"])] = {
            "date": str(pd.to_datetime(r["date"]).date()),
            ("cf" if use_cf else "gen_mwh"): (float(r["cf"]) if use_cf else float(r["gen_mwh"])),
            "drought": float(r["drought"]) if pd.notna(r["drought"]) else None
        }

    summary = {
        "date_range": {"start": first, "end": last},
        "group_key": gkey,
        "use_capacity_factor": use_cf,
        "rows": int(panel.shape[0]),
        "groups": int(panel["group"].nunique()),
        "latest_by_group": latest_dict,
        "leadlag_best_by_group": best_ll,
        "dlm": {
            "horizon": int(args.H),
            "groups_estimated": int(DLM["group"].nunique()) if not DLM.empty else 0,
            "note": "Coefficients are effects on Δlog(target) per 1-unit change in drought index."
        },
        "events_detected": int(EV.shape[0]),
        "scenarios": (SC["scenario"].unique().tolist() if not SC.empty else [])
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        gen=args.gen, capacity=(args.capacity or None), drought=args.drought,
        hydro=(args.hydro or None), climate=(args.climate or None),
        scenarios=(args.scenarios or None), group_key=gkey, drought_var=args.drought_var,
        windows=args.windows, lags=int(args.lags), H=int(args.H),
        controls=(args.controls=="yes"), base_start=(args.base_start or None),
        base_end=(args.base_end or None), start=(args.start or None),
        end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Hydropower vs Drought ==")
    print(f"Groups: {summary['groups']} | Sample: {first} → {last} | Use CF: {use_cf}")
    if best_ll:
        for g, info in list(best_ll.items())[:5]:
            print(f"Best lead/lag [{g}]: lag {info['lag']} corr={info['corr']:+.2f}")
    if not DLM.empty:
        for g in DLM["group"].unique()[:3]:
            sub = DLM[DLM["group"]==g]
            b0 = sub[sub["lag"]==0]
            if not b0.empty:
                print(f"DLM {g}: lag0 coef={b0['coef'].iloc[0]:+.3f} (t={b0['t_stat'].iloc[0]:+.2f})")
    print("Outputs in:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
