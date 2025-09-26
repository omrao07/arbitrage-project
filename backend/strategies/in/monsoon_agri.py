#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monsoon_agri.py — India monsoon & agriculture: rainfall/reservoirs/sowing → food inflation & scenarios
-----------------------------------------------------------------------------------------------------

What this does
==============
Given rainfall, reservoir storage, crop sowing, and CPI data (daily/weekly/monthly),
this script builds a clean monthly panel; constructs monsoon health indicators; runs
lead–lag and regression diagnostics; and estimates simple scenario impacts on food CPI.

Core features
-------------
1) Frequency alignment to monthly (from daily/weekly if needed)
2) Rainfall anomalies vs normals (or internal climatology) & spatial dispersion (CV across regions)
3) Kharif (Jun–Sep) and Rabi (Oct–Dec) composites; annual Monsoon Health Index (MHI)
4) Rolling & lead–lag correlations vs food CPI inflation
5) OLS with Newey–West (HAC) SEs: ΔYoY(CPI_Food) on rainfall anomaly, reservoirs, sowing
6) Scenarios: ±10% rainfall deviation → implied 6M food inflation impact using d-lag betas
7) Optional event-study (ENSO/IOD episodes, policy events)

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--rain rainfall.csv        REQUIRED (daily/weekly/monthly; optionally panel by state/district)
  Columns (min): date[, region] + rainfall_amount
  Optional normals in the same file: normal_rainfall (climatology for the period)
  Fuzzy name mapping used: rain, rainfall, precip, mm; normal, norm, climatology

--reservoir reservoirs.csv OPTIONAL (weekly/monthly)
  Columns: date[, region] + storage_pct or live_storage, capacity
  If only live_storage & capacity, storage_pct is computed.

--sowing sowing.csv        OPTIONAL (weekly/monthly; crop area)
  Columns: date[, region] + area_current[, area_normal]
  Fuzzy names: sown, sown_area, acreage, area; normal, normal_area

--cpi cpi.csv              OPTIONAL (monthly CPI)
  Columns: date + cpi_food (preferred) or cpi_cf, cpi_rl_food, or total cpi with food_weight
  Fuzzy names: cpi_food, food_cpi, cpi_fo, cfpi; also headline cpi if nothing else.

--events events.csv        OPTIONAL (event study)
  Columns: date, label

--baseline normals.csv     OPTIONAL (external rainfall normals)
  Columns: month (1..12 or Jan..Dec) + normal_rainfall (per-region optional)

Key CLI
-------
--freq daily|weekly|monthly     Input base frequency for rainfall/sowing/reservoir (default auto)
--region_col state|district     Name of regional column if panel data provided (default auto-detect)
--windows 3,6,12                Rolling windows (months)
--lags 12                       Lead–lag/regression lag horizon (months)
--kharif 6,7,8,9                Kharif months (default 6–9)
--rabi 10,11,12                 Rabi months (default 10–12)
--start / --end                 Date filters (YYYY-MM-DD)
--outdir out_monsoon            Output directory (default)

Outputs
-------
- panel.csv                   Monthly aligned panel (levels & transforms)
- monsoon_health.csv          Yearly composite index (MHI) & components by monsoon-year
- sowing_progress.csv         Monthly sowing progress & anomaly (overall and, if available, by region)
- rolling_stats.csv           Rolling corr of rain/reservoir/sowing vs food CPI YoY
- leadlag_corr.csv            Lead–lag tables Corr(X_{t−k}, CPI_food_yoy_t)
- regression_food.csv         Newey–West OLS: ΔYoY(CPI_food) ~ Σ β·lags(X)
- scenarios_food.csv          Impact of ±10% rainfall deviation (6M horizon) via cumulative betas
- event_study.csv             Average ΔYoY around events (if provided)
- summary.json                Headline diagnostics, latest snapshot, and best lead/lag
- config.json                 Run configuration

DISCLAIMER
----------
For research/monitoring. Approximations (e.g., normals, lag structure) are simplistic;
validate before policy or trading decisions.
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

def to_month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")

def to_period(s: pd.Series, base: str) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    if base.startswith("d"):  # daily
        return dt.dt.normalize()
    elif base.startswith("w"):  # weekly → align to month-end after resample
        return dt.dt.normalize()
    else:
        return dt.dt.to_period("M").dt.to_timestamp("M")

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def yoy_log(s: pd.Series, periods: int=12) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return (np.log(s) - np.log(s.shift(periods)))

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def cv(values: pd.Series) -> float:
    arr = values.astype(float).dropna().values
    if arr.size == 0:
        return np.nan
    m = np.mean(arr)
    sd = np.std(arr, ddof=0)
    return float(sd / m) if m != 0 else np.nan

def resample_to_monthly(df: pd.DataFrame, date_col: str, sum_cols: List[str], mean_cols: List[str]) -> pd.DataFrame:
    idx = df.set_index(date_col).sort_index()
    agg_map = {}
    for c in sum_cols: agg_map[c] = "sum"
    for c in mean_cols: agg_map[c] = "mean"
    out = idx.resample("M").agg(agg_map).reset_index()
    out[date_col] = out[date_col].dt.to_period("M").dt.to_timestamp("M")
    return out

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(4, window//2)).corr(y)

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

def load_rainfall(path: str, base_freq: str="", region_col_hint: str="") -> Tuple[pd.DataFrame, Optional[str], str]:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    # region
    region_col = None
    if region_col_hint and region_col_hint in df.columns:
        region_col = region_col_hint
    else:
        region_col = ncol(df, "state","region","district","subdivision","zone")
    # rainfall and normals
    rain_c = ncol(df, "rainfall","rain","precip","mm","rain_mm")
    norm_c = ncol(df, "normal","norm","climatology","long_term_avg","ltm")
    if not rain_c:
        raise ValueError("rainfall.csv must include a rainfall column (e.g., 'rainfall'/'rain'/'precip').")
    df = df.rename(columns={rain_c:"rain", (norm_c or "normal"):"normal"})
    # parse dates
    base = base_freq.lower().strip()
    df["date"] = to_period(df["date"], base or "monthly")
    # numerify
    df["rain"] = safe_num(df["rain"])
    if "normal" in df.columns:
        df["normal"] = safe_num(df["normal"])
    # If monthly already (freq guessed), okay; else resample
    if region_col:
        # aggregate by region first
        sum_cols = ["rain"]; mean_cols = []
        if "normal" in df.columns: mean_cols = ["normal"]
        g = df.groupby([region_col, df["date"]]).agg({"rain":"sum", **({"normal":"mean"} if "normal" in df.columns else {})}).reset_index()
        g = g.rename(columns={"date":"_date"})
        out = []
        for r, sub in g.groupby(region_col):
            tmp = sub.rename(columns={"_date":"date"})
            if isinstance(tmp["date"].iloc[0], pd.Timestamp) and (tmp["date"].dt.freq is None):
                tmp = tmp  # already monthly daily normalized; resample per below
            # resample to monthly
            tmp = resample_to_monthly(tmp, "date", sum_cols=["rain"], mean_cols=(["normal"] if "normal" in tmp.columns else []))
            tmp[region_col] = r
            out.append(tmp)
        rain_df = pd.concat(out, ignore_index=True)
    else:
        sum_cols = ["rain"]; mean_cols=(["normal"] if "normal" in df.columns else [])
        rain_df = resample_to_monthly(df[["date"]+sum_cols+mean_cols], "date", sum_cols=sum_cols, mean_cols=mean_cols)
    return rain_df, region_col, "rain"

def attach_baseline_normals(RAIN: pd.DataFrame, BASE: Optional[pd.DataFrame], region_col: Optional[str]) -> pd.DataFrame:
    out = RAIN.copy()
    if BASE is None or BASE.empty:
        # internal climatology: compute mean by calendar month (and region if present)
        if region_col:
            out["_month"] = out["date"].dt.month
            clim = out.groupby([region_col,"_month"])["rain"].mean().rename("normal_inferred").reset_index()
            out = out.merge(clim, on=[region_col,"_month"], how="left").drop(columns=["_month"])
            if "normal" not in out.columns: out = out.rename(columns={"normal_inferred":"normal"})
        else:
            out["_month"] = out["date"].dt.month
            clim = out.groupby("_month")["rain"].mean().rename("normal_inferred").reset_index()
            out = out.merge(clim, left_on="_month", right_on="_month", how="left").drop(columns=["_month"])
            if "normal" not in out.columns: out = out.rename(columns={"normal_inferred":"normal"})
    else:
        # external normals: month (+ region optional)
        B = BASE.copy()
        mcol = ncol(B, "month","mon","m")
        if not mcol: raise ValueError("baseline normals require a 'month' column (1..12 or month names).")
        B = B.rename(columns={mcol:"month"})
        # normalize month to 1..12
        if B["month"].dtype == object:
            month_map = {m.lower()[:3]: i for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}
            B["month"] = B["month"].astype(str).str[:3].str.lower().map(month_map)
        B["month"] = safe_num(B["month"]).astype(int)
        normc = ncol(B, "normal","norm","climatology","ltm")
        if not normc: raise ValueError("baseline normals need a normal/climatology column.")
        B = B.rename(columns={normc:"normal"})
        if region_col and region_col in B.columns:
            out["_m"] = out["date"].dt.month
            out = out.merge(B[[region_col,"month","normal"]].rename(columns={"month":"_m"}), on=[region_col,"_m"], how="left").drop(columns=["_m"])
        else:
            out["_m"] = out["date"].dt.month
            out = out.merge(B[["month","normal"]].rename(columns={"month":"_m"}), on=["_m"], how="left").drop(columns=["_m"])
    return out

def load_reservoirs(path: Optional[str], region_col: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    if region_col and region_col in df.columns:
        reg = region_col
    else:
        reg = ncol(df, "state","region","basin","reservoir","dam")
    stor = ncol(df, "storage_pct","storage_percent","stor_pct","percent_full")
    live = ncol(df, "live_storage","live","storage")
    cap  = ncol(df, "capacity","gross_storage","full_reservoir_level","frl")
    df["date"] = to_month_end(df["date"])
    if stor:
        df = df.rename(columns={stor:"storage_pct"})
        df["storage_pct"] = safe_num(df["storage_pct"])
    elif live and cap:
        df = df.rename(columns={live:"live_storage", cap:"capacity"})
        df["storage_pct"] = safe_num(df["live_storage"]) / safe_num(df["capacity"]).replace(0, np.nan)
    else:
        raise ValueError("reservoirs.csv must have storage_pct or (live_storage & capacity).")
    keep = ["date","storage_pct"] + ([reg] if reg else [])
    out = df[keep].copy()
    if reg:
        out = out.groupby([reg,"date"])["storage_pct"].mean().reset_index()
    out = out.groupby("date")["storage_pct"].mean().reset_index()
    return out

def load_sowing(path: Optional[str], region_col: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    if region_col and region_col in df.columns:
        reg = region_col
    else:
        reg = ncol(df, "state","region","district")
    cur = ncol(df, "area_current","sown","sown_area","acreage","area")
    nor = ncol(df, "area_normal","normal","normal_area")
    if not cur: raise ValueError("sowing.csv must contain an area column (e.g., area_current/sown/acreage).")
    df["date"] = to_month_end(df["date"])
    df = df.rename(columns={cur:"area_current"})
    df["area_current"] = safe_num(df["area_current"])
    if nor:
        df = df.rename(columns={nor:"area_normal"})
        df["area_normal"] = safe_num(df["area_normal"])
    keep = ["date","area_current"] + (["area_normal"] if "area_normal" in df.columns else []) + ([reg] if reg else [])
    out = df[keep].copy()
    if reg:
        out = out.groupby([reg,"date"], as_index=False).sum()
    out = out.groupby("date", as_index=False).sum()
    return out

def load_cpi(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    food = ncol(df, "cpi_food","food_cpi","cpi_fo","cfpi","cpi_cf")
    if not food:
        # fallback to headline if necessary
        food = ncol(df, "cpi","cpi_all","headline")
    if not food:
        raise ValueError("cpi.csv must contain CPI food (cpi_food/cfpi) or headline CPI.")
    df = df.rename(columns={food:"cpi_food"})
    df["cpi_food"] = safe_num(df["cpi_food"])
    out = df[["date","cpi_food"]].copy()
    out["cpi_food_yoy"] = yoy_log(out["cpi_food"])
    out["cpi_food_d1y"] = out["cpi_food_yoy"].diff()
    return out


# ----------------------------- constructions -----------------------------

def build_panel(RAIN0: pd.DataFrame, RES: pd.DataFrame, SOW: pd.DataFrame, CPI: pd.DataFrame,
                region_col: Optional[str]) -> pd.DataFrame:
    RAIN = RAIN0.copy()
    # rainfall anomaly
    if "normal" in RAIN.columns:
        RAIN["rain_anom"] = (RAIN["rain"] - RAIN["normal"]) / RAIN["normal"].replace(0, np.nan)
    else:
        # fallback z-score by month
        RAIN["_m"] = RAIN["date"].dt.month
        mu = RAIN.groupby("_m")["rain"].transform("mean")
        sd = RAIN.groupby("_m")["rain"].transform("std").replace(0, np.nan)
        RAIN["rain_anom"] = (RAIN["rain"] - mu) / sd
        RAIN.drop(columns=["_m"], inplace=True)

    # spatial dispersion (CV) if regional granularity existed before collapsing
    # Here RAIN is already aggregated nationally; we approximate dispersion using rolling volatility of anomalies
    RAIN["rain_cv_roll"] = RAIN["rain_anom"].rolling(6, min_periods=3).std(ddof=0)  # proxy

    # merge all
    panel = RAIN[["date","rain","rain_anom","rain_cv_roll"]].merge(RES, on="date", how="left", suffixes=("",""))
    if not SOW.empty:
        panel = panel.merge(SOW, on="date", how="left")
        if "area_normal" in panel.columns:
            panel["sowing_progress"] = panel["area_current"] / panel["area_normal"].replace(0, np.nan)
        else:
            panel["sowing_progress"] = panel["area_current"] / panel["area_current"].shift(52//4 if False else 12)  # rough YoY if no normal
    if not CPI.empty:
        panel = panel.merge(CPI, on="date", how="left")
    # transforms
    panel = panel.sort_values("date").drop_duplicates(subset=["date"])
    panel["storage_pct"] = panel["storage_pct"].clip(upper=5)  # sane cap
    panel["rain_z"] = (panel["rain_anom"] - panel["rain_anom"].rolling(24, min_periods=12).mean()) / panel["rain_anom"].rolling(24, min_periods=12).std(ddof=0)
    panel["storage_z"] = (panel["storage_pct"] - panel["storage_pct"].rolling(24, min_periods=12).mean()) / panel["storage_pct"].rolling(24, min_periods=12).std(ddof=0)
    panel["sowing_z"] = (panel["sowing_progress"] - panel["sowing_progress"].rolling(24, min_periods=12).mean()) / panel["sowing_progress"].rolling(24, min_periods=12).std(ddof=0)
    return panel

def kharif_rabi_composites(panel: pd.DataFrame, kharif_months: List[int], rabi_months: List[int]) -> pd.DataFrame:
    df = panel.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # monsoon year anchored on Jun (if kharif starts in Jun)
    anchor = min(kharif_months) if kharif_months else 6
    df["monsoon_year"] = df["year"] + ((df["month"] >= anchor).astype(int))
    agg = []
    for y, g in df.groupby("monsoon_year"):
        kh = g[g["month"].isin(kharif_months)]
        rb = g[g["month"].isin(rabi_months)]
        rec = {"monsoon_year": int(y)}
        for col, func in [("rain_anom","mean"), ("rain_cv_roll","mean"), ("storage_pct","mean"), ("sowing_progress","mean")]:
            if col in g.columns:
                rec[f"kharif_{col}"] = getattr(kh[col], func)() if not kh.empty else np.nan
                rec[f"rabi_{col}"]   = getattr(rb[col], func)() if not rb.empty else np.nan
        agg.append(rec)
    comp = pd.DataFrame(agg).sort_values("monsoon_year")
    # Monsoon Health Index (weights can be tweaked)
    w = {"rain_anom": 0.5, "rain_cv_roll": -0.2, "storage_pct": 0.2, "sowing_progress": 0.1}
    # standardize components before weighting
    for prefix in ["kharif","rabi"]:
        for col in ["rain_anom","rain_cv_roll","storage_pct","sowing_progress"]:
            c = f"{prefix}_{col}"
            if c in comp.columns:
                comp[f"z_{c}"] = (comp[c] - comp[c].mean()) / comp[c].std(ddof=0)
        comp[f"{prefix}_MHI"] = (
            w["rain_anom"] * comp.get(f"z_{prefix}_rain_anom", pd.Series(index=comp.index)) +
            w["rain_cv_roll"] * comp.get(f"z_{prefix}_rain_cv_roll", pd.Series(index=comp.index)) +
            w["storage_pct"] * comp.get(f"z_{prefix}_storage_pct", pd.Series(index=comp.index)) +
            w["sowing_progress"] * comp.get(f"z_{prefix}_sowing_progress", pd.Series(index=comp.index))
        )
    return comp

def rolling_and_leadlag(panel: pd.DataFrame, windows: List[int], lags: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_roll = []
    if "cpi_food_yoy" in panel.columns:
        y = panel.set_index("date")["cpi_food_yoy"]
        for xname in ["rain_anom","storage_pct","sowing_progress"]:
            if xname not in panel.columns: continue
            x = panel.set_index("date")[xname]
            for w, tag in zip(windows, ["short","med","long"]):
                out_roll.append({"x": xname, "window": w, "tag": tag,
                                 "corr": float(roll_corr(x, y, w).iloc[-1]) if len(panel)>=w else np.nan})
    roll_df = pd.DataFrame(out_roll)

    out_ll = []
    if "cpi_food_yoy" in panel.columns:
        y = panel["cpi_food_yoy"]
        for xname in ["rain_anom","storage_pct","sowing_progress"]:
            if xname not in panel.columns: continue
            tab = leadlag_corr(panel[xname], y, lags)
            tab["x"] = xname
            out_ll.append(tab)
    ll_df = pd.concat(out_ll, ignore_index=True) if out_ll else pd.DataFrame(columns=["lag","corr","x"])
    return roll_df, ll_df

def regression_food(panel: pd.DataFrame, L: int=12) -> pd.DataFrame:
    """
    ΔYoY(CPI_food)_t on lags of rain_anom, storage_pct, sowing_progress (0..L), Newey–West SE.
    """
    if "cpi_food_yoy" not in panel.columns or panel["cpi_food_yoy"].dropna().shape[0] < 24:
        return pd.DataFrame()
    df = panel.copy()
    df["d_cpi_food_yoy"] = df["cpi_food_yoy"].diff()
    dep = "d_cpi_food_yoy"
    # build lag matrix
    Xparts = [pd.Series(1.0, index=df.index, name="const")]
    names = ["const"]
    for v in ["rain_anom","storage_pct","sowing_progress"]:
        if v in df.columns:
            for l in range(0, L+1):
                nm = f"{v}_l{l}"
                Xparts.append(df[v].shift(l).rename(nm))
                names.append(nm)
    X = pd.concat(Xparts, axis=1)
    XY = pd.concat([df[dep].rename("dep"), X], axis=1).dropna()
    if XY.shape[0] < max(36, 5*X.shape[1]):
        return pd.DataFrame()
    yv = XY["dep"].values.reshape(-1,1)
    Xv = XY.drop(columns=["dep"]).values
    beta, resid, XtX_inv = ols_beta_se(Xv, yv)
    se = hac_se(Xv, resid, XtX_inv, L=max(6, L))
    rows = []
    for i, nm in enumerate(names):
        rows.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                     "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "lags": L})
    # cumulative rainfall effect (0..L)
    ridx = [i for i, nm in enumerate(names) if nm.startswith("rain_anom_l")]
    if ridx:
        bsum = float(beta[ridx,0].sum()); sesum = float(np.sqrt(np.sum(se[ridx]**2)))
        rows.append({"var": "rain_anom_cum_0..L", "coef": bsum, "se": sesum, "t_stat": bsum/(sesum if sesum>0 else np.nan), "lags": L})
    return pd.DataFrame(rows)

def scenarios_from_reg(reg: pd.DataFrame, shocks: List[float]=[0.10, -0.10]) -> pd.DataFrame:
    """
    Map rainfall deviation shocks (±10%) to ΔYoY(CPI_food) over next 6 months using cumulative betas.
    We use 0..5 lags sum if available; else 0..L.
    """
    if reg.empty: return pd.DataFrame()
    # Prefer cum betas over first 6 lags
    def cum_beta(prefix: str, max_l: int) -> float:
        filt = reg[reg["var"].str.startswith(prefix)]
        if filt.empty: return np.nan
        # pick up to first 6 lag entries (l0..l5)
        part = []
        for l in range(0, min(6, max_l)+1):
            nm = f"{prefix}{l}"
            r = filt[filt["var"]==nm]
            if not r.empty:
                part.append(float(r["coef"].iloc[0]))
        if part:
            return float(np.sum(part))
        # fallback to explicit cum row
        rc = reg[reg["var"]==f"{prefix[:-1]}cum_0..L"]
        return float(rc["coef"].iloc[0]) if not rc.empty else np.nan

    Lmax = int(reg["lags"].max() if "lags" in reg.columns and reg["lags"].notna().any() else 12)
    b_rain = cum_beta("rain_anom_l", Lmax)
    rows = []
    for sh in shocks:
        rows.append({"shock": f"{sh*100:.0f}%", "variable": "rain_anom", "cum_beta_6m": b_rain, "impact_pct": b_rain*sh*100.0})
    return pd.DataFrame(rows)


# ----------------------------- event study -----------------------------

def event_study(panel: pd.DataFrame, events: pd.DataFrame, window: int=6) -> pd.DataFrame:
    if events.empty or "cpi_food_yoy" not in panel.columns:
        return pd.DataFrame()
    rows = []
    dates = panel["date"].values
    for _, ev in events.iterrows():
        d0 = pd.to_datetime(ev["date"])
        lbl = str(ev.get("label","event"))
        if len(dates)==0: continue
        idx = np.argmin(np.abs(dates - np.datetime64(d0)))
        anchor = pd.to_datetime(dates[idx])
        sl = panel[(panel["date"] >= anchor - pd.offsets.DateOffset(months=window)) &
                   (panel["date"] <= anchor + pd.offsets.DateOffset(months=window))]
        for _, r in sl.iterrows():
            h = (r["date"].to_period("M") - anchor.to_period("M")).n
            rows.append({"event": lbl, "event_date": anchor,
                         "h": int(h),
                         "d_cpi_food_yoy": float(r.get("cpi_food_d1y", np.nan)) if pd.notna(r.get("cpi_food_d1y", np.nan)) else np.nan,
                         "rain_anom": float(r.get("rain_anom", np.nan)) if pd.notna(r.get("rain_anom", np.nan)) else np.nan})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = (df.groupby(["event","event_date","h"])
             .agg({"d_cpi_food_yoy":"mean","rain_anom":"mean"})
             .reset_index()
             .sort_values(["event","h"]))
    return out


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    rain: str
    reservoirs: Optional[str]
    sowing: Optional[str]
    cpi: Optional[str]
    events: Optional[str]
    baseline: Optional[str]
    freq: Optional[str]
    region_col: Optional[str]
    windows: str
    lags: int
    kharif: str
    rabi: str
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Monsoon → Agriculture → Food CPI analytics")
    ap.add_argument("--rain", required=True, help="Rainfall CSV (date[, region], rainfall[, normal])")
    ap.add_argument("--reservoirs", default="")
    ap.add_argument("--sowing", default="")
    ap.add_argument("--cpi", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--baseline", default="", help="Optional normals CSV (month[, region], normal)")
    ap.add_argument("--freq", default="", choices=["","daily","weekly","monthly"], help="Base input freq for rainfall")
    ap.add_argument("--region_col", default="", help="Region column name if present (e.g., 'state')")
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--kharif", default="6,7,8,9")
    ap.add_argument("--rabi", default="10,11,12")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_monsoon")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load rainfall
    RAIN_raw, region_col, rain_col = load_rainfall(args.rain, base_freq=(args.freq or ""), region_col_hint=(args.region_col or ""))
    BASE = pd.read_csv(args.baseline) if args.baseline else pd.DataFrame()
    if not BASE.empty:
        # normalize month column type inside attach
        pass
    RAIN = attach_baseline_normals(RAIN_raw, BASE if not BASE.empty else None, region_col)

    # Date filters applied later in panel merge
    # Reservoirs / Sowing / CPI / Events
    RES = load_reservoirs(args.reservoirs, region_col) if args.reservoirs else pd.DataFrame()
    SOW = load_sowing(args.sowing, region_col) if args.sowing else pd.DataFrame()
    CPI = load_cpi(args.cpi) if args.cpi else pd.DataFrame()

    if args.start:
        for df in [RAIN, RES, SOW, CPI]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [RAIN, RES, SOW, CPI]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    PANEL = build_panel(RAIN, RES, SOW, CPI, region_col)
    if PANEL.shape[0] < 24:
        raise ValueError("Insufficient overlapping months after alignment (need ≥24).")
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Kharif/Rabi composites & MHI
    kharif_months = [int(x.strip()) for x in args.kharif.split(",") if x.strip()]
    rabi_months   = [int(x.strip()) for x in args.rabi.split(",") if x.strip()]
    MHI = kharif_rabi_composites(PANEL, kharif_months, rabi_months)
    if not MHI.empty: MHI.to_csv(outdir / "monsoon_health.csv", index=False)

    # Sowing progress export (if available)
    if "area_current" in PANEL.columns:
        sp = PANEL[["date"] + [c for c in ["area_current","area_normal","sowing_progress"] if c in PANEL.columns]].copy()
        sp.to_csv(outdir / "sowing_progress.csv", index=False)

    # Rolling & lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL, LL = rolling_and_leadlag(PANEL, windows, lags=int(args.lags))
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)
    if not LL.empty:   LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Regression & scenarios
    REG = regression_food(PANEL, L=int(args.lags))
    if not REG.empty: REG.to_csv(outdir / "regression_food.csv", index=False)
    SCN = scenarios_from_reg(REG, shocks=[+0.10, -0.10])
    if not SCN.empty: SCN.to_csv(outdir / "scenarios_food.csv", index=False)

    # Event study
    EVENTS = pd.DataFrame()
    if args.events:
        ev = pd.read_csv(args.events)
        dt = ncol(ev, "date") or ev.columns[0]
        lab = ncol(ev, "label","event","name") or "label"
        ev = ev.rename(columns={dt:"date", lab:"label"})
        ev["date"] = to_month_end(ev["date"])
        EVENTS = ev[["date","label"]].dropna()
    ES = event_study(PANEL, EVENTS, window=max(6, int(args.lags)//2)) if not EVENTS.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Best lead/lag summary
    best_ll = {}
    if not LL.empty:
        for x, g in LL.dropna(subset=["corr"]).groupby("x"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[x] = {"lag": int(row["lag"]), "corr": float(row["corr"])}

    # Latest snapshot & summary
    last = PANEL.tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "latest": {
            "date": str(last["date"].date()),
            "rain_anom": float(last.get("rain_anom", np.nan)) if pd.notna(last.get("rain_anom", np.nan)) else None,
            "storage_pct": float(last.get("storage_pct", np.nan)) if pd.notna(last.get("storage_pct", np.nan)) else None,
            "sowing_progress": float(last.get("sowing_progress", np.nan)) if pd.notna(last.get("sowing_progress", np.nan)) else None,
            "cpi_food_yoy": float(last.get("cpi_food_yoy", np.nan)) if pd.notna(last.get("cpi_food_yoy", np.nan)) else None,
        },
        "rolling_windows": windows,
        "leadlag_best": best_ll,
        "regression_vars": REG["var"].tolist() if not REG.empty else [],
        "scenario_sample": (SCN.to_dict(orient="records") if not SCN.empty else []),
        "mhi_latest": (
            MHI.tail(1).to_dict(orient="records")[0] if not MHI.empty else {}
        )
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        rain=args.rain, reservoirs=(args.reservoirs or None), sowing=(args.sowing or None),
        cpi=(args.cpi or None), events=(args.events or None), baseline=(args.baseline or None),
        freq=(args.freq or None), region_col=(args.region_col or None),
        windows=args.windows, lags=int(args.lags),
        kharif=args.kharif, rabi=args.rabi,
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Monsoon → Agri → Food CPI ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']}")
    if summary["latest"]["rain_anom"] is not None:
        print(f"Latest rain anomaly: {summary['latest']['rain_anom']:+.2f}  | Storage: {summary['latest']['storage_pct']:.2f if summary['latest']['storage_pct'] is not None else float('nan')}")
    if best_ll:
        for x, st in best_ll.items():
            print(f"Lead–lag {x}: max |corr| at lag {st['lag']:+d} → {st['corr']:+.2f}")
    if not SCN.empty:
        for _, r in SCN.iterrows():
            print(f"Scenario {r['shock']} rainfall → ΔYoY(CPI_Food) ≈ {r['impact_pct']:+.2f} pp (6M cum)")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
