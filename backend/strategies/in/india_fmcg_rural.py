#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
india_fmcg_rural.py — India FMCG: Rural vs Urban demand tracker & drivers
-------------------------------------------------------------------------

What this does
==============
Given FMCG sales data split by channel (Rural/Urban) and optional macro inputs
(rainfall/monsoon, CPI, rural wages, distribution metrics), this script:

1) Cleans & aligns everything to **monthly** frequency at national and state levels.
2) Builds KPIs:
   • Volume, Value, ASP (value/volume), Real Volume (deflated), YoY/3MMA growth
   • Price-vs-Volume decomposition: dlog(Value) ≈ dlog(Volume) + dlog(ASP)
   • Rural–Urban gap (%-pt difference) for value & volume growth
3) Driver diagnostics for **Rural** channel:
   • Rolling correlations/betas vs rainfall anomaly, CPI-food, rural wages, diesel
   • Lead–lag: Corr(Rainfall_{t−k}, Δlog(Volume)_t), k ∈ [−L..+L]
   • Elasticities via OLS:
       Δlog(Vol_rural) ~ α + β_P·Δlog(ASP_rural) + β_W·Δlog(Wages)
                         + β_R·Rain_anom  + β_D·Δlog(Distribution)
     with month-of-year fixed effects
4) (Optional) Distribution effects if WD/ND provided: elasticity of volume to WD/ND
5) Simple **forecast & scenarios**:
   • Forecast Rural volume for N months using last estimated elasticities
   • Scenarios: Monsoon −10% anomaly; ASP +5%; WD +10% (where available)
6) Outputs tidy CSVs and a JSON summary.

Inputs (CSV; headers are flexible, case-insensitive)
----------------------------------------------------
--sales sales.csv          REQUIRED (daily/weekly/monthly supported)
  Columns (min): date, value_in_inr, volume_units, channel
  Optional: state (or region), category, asp, price, pack_size, brand, sku

  'channel' should contain Rural/Urban (case-insensitive; fuzzy match ok).
  If 'asp' missing, computed as value_in_inr / volume_units.

--distribution dist.csv    OPTIONAL (monthly recommended)
  Columns: date [, state] [, category], nd (numeric distribution), wd (weighted distribution)

--macro macro.csv          OPTIONAL (monthly)
  Put any of these (name them however; script will guess):
    cpi_food, cpi_rural, wages_rural, diesel_price, rainfall_mm, rainfall_anom, rainfall_z,
    imd_anom, spei, spi
  Non-monthly will be resampled to month.

--start YYYY-MM-DD         OPTIONAL date filter
--end   YYYY-MM-DD         OPTIONAL date filter
--windows 3,6,12           Rolling window lengths in months
--lags 6                   Lead–lag horizon (months)
--forecast 6               Forecast horizon in months
--outdir out_rural         Output directory

Outputs
-------
- panel.csv                Per date×channel (and state, if present): KPIs & transforms
- rural_vs_urban.csv       National rural/urban growth & gap; price/volume decomposition
- rolling_stats.csv        Rolling corr/beta of Rural Δlog(Vol) vs drivers (per window)
- rainfall_leadlag.csv     Lead–lag table (Rural Δlog(Vol) vs rainfall anomaly)
- rural_elasticities.csv   OLS elasticities (coef, se, t) for Rural
- distribution_effects.csv Elasticity of Rural volume to WD/ND (if provided)
- forecast.csv             Rural volume forecast & scenarios
- summary.json             Headline metrics & latest snapshot
- config.json              Run configuration for reproducibility

Notes
-----
• Growth measures use Δlog (approx % m/m). YoY uses 12-month log difference.
• Real volume uses CPI deflation: volume_real ≈ volume / (CPI/CPI_base) if CPI given
  (or value deflation if only value is available).
• Rainfall anomaly sources: rainfall_anom (%) or rainfall_z or IMD anomaly. If only rainfall_mm
  is provided, a within-calendar-month z-score is used.
• Elasticities are national unless --state segmentation is used in your sales.csv; you’ll also
  get state-level panels.

DISCLAIMER
----------
Research tooling; validate with your internal RMS/POS panels before decisioning.
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
    # contains
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

def yoy(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff(12)

def z_by_month(s: pd.Series, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"v": s.values, "m": pd.to_datetime(dates).dt.month.values})
    out = pd.Series(np.nan, index=s.index)
    for m in range(1,13):
        idx = (df["m"] == m)
        vals = df.loc[idx, "v"].astype(float)
        if vals.notna().sum() >= 6:
            mu, sd = vals.mean(), vals.std(ddof=0)
            sd = sd if (sd and np.isfinite(sd)) else np.nan
            out.loc[idx[idx].index] = (vals - mu) / sd if (sd and sd>0) else np.nan
    return out

def roll_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(3, window//2)).corr(y)

def roll_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    minp = max(3, window//2)
    x_ = x.astype(float); y_ = y.astype(float)
    mx = x_.rolling(window, min_periods=minp).mean()
    my = y_.rolling(window, min_periods=minp).mean()
    cov = (x_*y_).rolling(window, min_periods=minp).mean() - mx*my
    varx = (x_*x_).rolling(window, min_periods=minp).mean() - mx*mx
    return cov / varx.replace(0, np.nan)

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

def ols(y: pd.Series, X: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """OLS with (X'X)^-1 X'y; returns (beta, cov_beta) with homoskedastic SE."""
    Y = y.values.reshape(-1,1)
    Xv = X.values
    XtX = Xv.T @ Xv
    XtY = Xv.T @ Y
    XtX_inv = np.linalg.pinv(XtX)
    B = XtX_inv @ XtY
    resid = Y - Xv @ B
    n, k = Xv.shape
    dof = max(1, n - k)
    sigma2 = float((resid.T @ resid) / dof)
    covB = XtX_inv * sigma2
    beta = pd.Series(B.ravel(), index=X.columns, dtype=float)
    cov = pd.DataFrame(covB, index=X.columns, columns=X.columns, dtype=float)
    return beta, cov


# ----------------------------- loaders -----------------------------

def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])

    # map columns
    val_c = ncol(df, "value_in_inr", "value", "sales_value", "revenue", "gmv")
    vol_c = ncol(df, "volume_units", "volume", "qty", "quantity")
    ch_c  = ncol(df, "channel", "channel_type", "rural_urban", "market")
    st_c  = ncol(df, "state", "region")

    if not (val_c and vol_c and ch_c):
        raise ValueError("sales.csv requires 'date', 'value_in_inr' & 'volume_units' & 'channel' (Rural/Urban).")

    df = df.rename(columns={val_c:"value_inr", vol_c:"volume", ch_c:"channel"})
    if st_c: df = df.rename(columns={st_c:"state"})

    # clean channel
    def ch_map(x: str) -> str:
        if not isinstance(x, str): return np.nan
        xl = x.lower()
        if "rur" in xl or "upc" in xl: return "Rural"
        if "urb" in xl or "metro" in xl or "town" in xl: return "Urban"
        return x.title()
    df["channel"] = df["channel"].astype(str).apply(ch_map)

    # ASP
    asp_c = ncol(df, "asp", "price", "avg_price")
    if asp_c: df["asp"] = safe_num(df[asp_c])
    else: df["asp"] = safe_num(df["value_inr"]) / safe_num(df["volume"])

    # aggregate to month × channel × (state)
    group_cols = ["date", "channel"] + (["state"] if "state" in df.columns else [])
    agg = df.groupby(group_cols).agg(
        value_inr=("value_inr","sum"),
        volume=("volume","sum"),
        asp=("asp","mean")
    ).reset_index()

    return agg.sort_values(group_cols)

def load_distribution(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    state_c = ncol(df, "state", "region")
    if state_c: df = df.rename(columns={state_c:"state"})
    nd_c = ncol(df, "nd", "numeric_distribution", "num_dist")
    wd_c = ncol(df, "wd", "weighted_distribution", "wtd_dist", "weighted_dist")
    out = df[["date"] + ([ "state" ] if "state" in df.columns else []) + ([nd_c] if nd_c else []) + ([wd_c] if wd_c else [])].copy()
    if nd_c: out = out.rename(columns={nd_c:"nd"})
    if wd_c: out = out.rename(columns={wd_c:"wd"})
    # enforce numeric
    for c in ["nd","wd"]:
        if c in out.columns:
            out[c] = safe_num(out[c])
    return out

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_month_end(df["date"])
    # Try to map typical macro names
    ren = {}
    for raw, std in [
        ("cpi_food","cpi_food"), ("cpi_rural","cpi_rural"),
        ("wages_rural","wages"), ("rural_wages","wages"),
        ("diesel_price","diesel"), ("diesel","diesel"),
        ("rainfall_anom","rain_anom"), ("rain_anom","rain_anom"),
        ("rainfall_z","rain_z"), ("imd_anom","rain_anom"),
        ("rainfall_mm","rain_mm"), ("spi","spi"), ("spei","spei")
    ]:
        c = ncol(df, raw)
        if c: ren[c] = std
    df = df.rename(columns=ren)
    # numericify
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])
    return df


# ----------------------------- core construction -----------------------------

def build_panel(SALES: pd.DataFrame, DIST: pd.DataFrame, MACRO: pd.DataFrame) -> pd.DataFrame:
    # National panel
    nat = (SALES.groupby(["date","channel"])
                 .agg(value_inr=("value_inr","sum"),
                      volume=("volume","sum"),
                      asp=("asp","mean"))
                 .reset_index())
    nat["state"] = "ALL"
    # State panel (if states exist)
    if "state" in SALES.columns:
        st = SALES.copy()
        st = st[["date","state","channel","value_inr","volume","asp"]]
    else:
        st = nat.copy()

    panel = pd.concat([nat, st], ignore_index=True).drop_duplicates(subset=["date","state","channel"])

    # Merge distribution (state-level, fallback to national if provided as ALL)
    if not DIST.empty:
        if "state" not in DIST.columns:
            tmp = DIST.copy(); tmp["state"] = "ALL"; DIST2 = tmp
        else:
            DIST2 = DIST.copy()
        panel = panel.merge(DIST2, on=["date","state"], how="left")

    # Merge macro (national)
    if not MACRO.empty:
        panel = panel.merge(MACRO, on="date", how="left")

    # Transforms
    panel["dlog_volume"] = panel.groupby(["state","channel"])["volume"].apply(dlog).values
    panel["dlog_value"]  = panel.groupby(["state","channel"])["value_inr"].apply(dlog).values
    panel["dlog_asp"]    = panel.groupby(["state","channel"])["asp"].apply(dlog).values
    panel["yoy_volume"]  = panel.groupby(["state","channel"])["volume"].apply(yoy).values
    panel["yoy_value"]   = panel.groupby(["state","channel"])["value_inr"].apply(yoy).values
    panel["mm3_volume"]  = panel.groupby(["state","channel"])["dlog_volume"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    panel["mm3_value"]   = panel.groupby(["state","channel"])["dlog_value"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # Real volume (deflate by CPI if available)
    if "cpi_rural" in panel.columns or "cpi_food" in panel.columns:
        cpi = panel["cpi_rural"] if "cpi_rural" in panel.columns else panel["cpi_food"]
        base = cpi.dropna().iloc[0] if cpi.dropna().size else np.nan
        if base and base > 0:
            panel["price_index"] = cpi / base
            panel["volume_real"] = panel["volume"] / panel["price_index"]
            panel["dlog_volume_real"] = panel.groupby(["state","channel"])["volume_real"].apply(dlog).values
        else:
            panel["volume_real"] = np.nan
            panel["dlog_volume_real"] = np.nan
    else:
        panel["volume_real"] = np.nan
        panel["dlog_volume_real"] = np.nan

    # Rainfall anomaly construction if missing
    if "rain_anom" not in panel.columns:
        if "rain_z" in panel.columns:
            panel["rain_anom"] = panel["rain_z"]
        elif "rain_mm" in panel.columns:
            panel["rain_anom"] = z_by_month(panel["rain_mm"], panel["date"])
        elif "spi" in panel.columns:
            panel["rain_anom"] = panel["spi"]  # proxy
        else:
            panel["rain_anom"] = np.nan

    # Δlog Wages, Diesel, Distribution
    if "wages" in panel.columns:
        panel["dlog_wages"] = dlog(panel["wages"])
    if "diesel" in panel.columns:
        panel["dlog_diesel"] = dlog(panel["diesel"])
    for c in ["nd","wd"]:
        if c in panel.columns:
            panel[f"dlog_{c}"] = dlog(panel[c])

    return panel.sort_values(["state","channel","date"])


# ----------------------------- analytics -----------------------------

def rural_vs_urban(panel: pd.DataFrame) -> pd.DataFrame:
    """National rural vs urban growth & decomposition."""
    nat = panel[panel["state"]=="ALL"].copy()
    # Pivot to R/U
    pvt = nat.pivot_table(index="date", columns="channel", values=["dlog_value","dlog_volume","dlog_asp","yoy_value","yoy_volume"], aggfunc="first")
    pvt.columns = ["_".join(col).strip() for col in pvt.columns.values]
    # Gaps (Rural - Urban)
    pvt["gap_dlog_value"]  = pvt.get("dlog_value_Rural")  - pvt.get("dlog_value_Urban")
    pvt["gap_dlog_volume"] = pvt.get("dlog_volume_Rural") - pvt.get("dlog_volume_Urban")
    pvt["gap_dlog_asp"]    = pvt.get("dlog_asp_Rural")    - pvt.get("dlog_asp_Urban")
    pvt["gap_yoy_value"]   = pvt.get("yoy_value_Rural")   - pvt.get("yoy_value_Urban")
    pvt["gap_yoy_volume"]  = pvt.get("yoy_volume_Rural")  - pvt.get("yoy_volume_Urban")
    return pvt.reset_index()

def rolling_driver_stats(panel: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Rolling corr/beta of Rural Δlog(Vol) vs drivers."""
    nat_r = panel[(panel["state"]=="ALL") & (panel["channel"]=="Rural")].set_index("date")
    drivers = {
        "rain_anom": nat_r["rain_anom"] if "rain_anom" in nat_r.columns else None,
        "dlog_wages": nat_r["dlog_wages"] if "dlog_wages" in nat_r.columns else None,
        "dlog_diesel": nat_r["dlog_diesel"] if "dlog_diesel" in nat_r.columns else None,
        "dlog_asp": nat_r["dlog_asp"] if "dlog_asp" in nat_r.columns else None,
        "dlog_wd": nat_r["dlog_wd"] if "dlog_wd" in nat_r.columns else None
    }
    y = nat_r["dlog_volume"].copy()
    rows = []
    for w, tag in zip(windows, ["short","med","long"]):
        for name, x in drivers.items():
            if x is None or x.dropna().empty:
                continue
            rows.append({
                "window": w, "tag": tag, "driver": name,
                "corr": float(roll_corr(x, y, w).iloc[-1]) if len(y)>=w else np.nan,
                "beta": float(roll_beta(y, x, w).iloc[-1]) if len(y)>=w else np.nan
            })
    return pd.DataFrame(rows)

def rainfall_leadlag(panel: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    nat_r = panel[(panel["state"]=="ALL") & (panel["channel"]=="Rural")].set_index("date")
    if "rain_anom" not in nat_r.columns or nat_r["rain_anom"].dropna().empty:
        return pd.DataFrame(columns=["lag","corr"])
    return leadlag_corr(nat_r["rain_anom"], nat_r["dlog_volume"], max_lag)

def rural_elasticities(panel: pd.DataFrame) -> pd.DataFrame:
    """OLS elasticities for Rural at national level with month FEs."""
    df = panel[(panel["state"]=="ALL") & (panel["channel"]=="Rural")].copy()
    if df.empty: return pd.DataFrame(columns=["var","coef","std_err","t_stat","n"])
    df["month"] = df["date"].dt.month
    # Build regressors
    X = pd.DataFrame({"const": 1.0}, index=df.index)
    # core regressors
    regs = []
    if "dlog_asp" in df.columns: X["dlog_asp"] = df["dlog_asp"]; regs.append("dlog_asp")
    if "dlog_wages" in df.columns: X["dlog_wages"] = df["dlog_wages"]; regs.append("dlog_wages")
    if "rain_anom" in df.columns: X["rain_anom"] = df["rain_anom"]; regs.append("rain_anom")
    if "dlog_wd" in df.columns: X["dlog_wd"] = df["dlog_wd"]; regs.append("dlog_wd")
    # month fixed effects (drop one)
    dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)
    X = pd.concat([X, dummies], axis=1)
    y = df["dlog_volume"]
    XY = pd.concat([y, X], axis=1).dropna()
    if XY.shape[0] < 36:
        return pd.DataFrame(columns=["var","coef","std_err","t_stat","n"])
    beta, cov = ols(XY["dlog_volume"], XY.drop(columns=["dlog_volume"]))
    se = np.sqrt(np.diag(cov.values))
    out = pd.DataFrame({
        "var": XY.drop(columns=["dlog_volume"]).columns,
        "coef": beta.values,
        "std_err": se,
        "t_stat": beta.values / np.where(se==0, np.nan, se),
    })
    out = out[~out["var"].str.startswith("m_") & (out["var"]!="const")]
    out["n"] = XY.shape[0]
    return out.reset_index(drop=True)

def distribution_effects(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel[(panel["state"]=="ALL") & (panel["channel"]=="Rural")].copy()
    if "wd" not in df.columns and "nd" not in df.columns:
        return pd.DataFrame(columns=["var","coef","std_err","t_stat","n"])
    X = pd.DataFrame({"const": 1.0}, index=df.index)
    regs = []
    if "dlog_wd" in df.columns: X["dlog_wd"] = df["dlog_wd"]; regs.append("dlog_wd")
    if "dlog_nd" in df.columns: X["dlog_nd"] = df["dlog_nd"]; regs.append("dlog_nd")
    X = X.dropna(axis=1, how="any")
    y = df["dlog_volume"]
    XY = pd.concat([y, X], axis=1).dropna()
    if XY.shape[0] < 24 or len(regs)==0:
        return pd.DataFrame(columns=["var","coef","std_err","t_stat","n"])
    beta, cov = ols(XY["dlog_volume"], XY.drop(columns=["dlog_volume"]))
    se = np.sqrt(np.diag(cov.values))
    out = pd.DataFrame({
        "var": XY.drop(columns=["dlog_volume"]).columns,
        "coef": beta.values,
        "std_err": se,
        "t_stat": beta.values / np.where(se==0, np.nan, se),
        "n": XY.shape[0]
    })
    return out[out["var"]!="const"].reset_index(drop=True)

def simple_forecast(panel: pd.DataFrame, elastic: pd.DataFrame, horizon: int=6) -> pd.DataFrame:
    """
    Forecast Rural volume level using last month as base and Δlog model with drivers.
    Δlog Vol_hat = bP*Δlog ASP + bW*Δlog Wages + bR*Rain_anom + bD*Δlog WD
    If a driver is missing forward, assume 0 (flat) for Δlogs and 0 for anomaly.
    """
    nat_r = panel[(panel["state"]=="ALL") & (panel["channel"]=="Rural")].copy().sort_values("date")
    if nat_r.empty: return pd.DataFrame()
    last_row = nat_r.tail(1).iloc[0]
    start_date = last_row["date"] + pd.offsets.MonthEnd(1)

    # pull coefs
    coefs = {r["var"]: r["coef"] for _, r in elastic.iterrows()}
    # driver series
    drv = nat_r[["date","dlog_asp","dlog_wages","rain_anom","dlog_wd"]].copy()
    # build forward dates
    fut_dates = pd.date_range(start=start_date, periods=horizon, freq="M")
    fut = pd.DataFrame({"date": fut_dates})
    # fill with zeros by default (flat)
    for c in ["dlog_asp","dlog_wages","dlog_wd","rain_anom"]:
        if c in drv.columns:
            base = drv[["date", c]].tail(12).copy()
            # forward assume zero change / neutral anomaly
            fut[c] = 0.0 if c!="rain_anom" else 0.0
        else:
            fut[c] = 0.0

    # simulate level path
    vol0 = float(last_row["volume"])
    path = []
    vol = vol0
    for _, r in fut.iterrows():
        dlog_hat = 0.0
        dlog_hat += coefs.get("dlog_asp", 0.0)*r.get("dlog_asp", 0.0)
        dlog_hat += coefs.get("dlog_wages", 0.0)*r.get("dlog_wages", 0.0)
        dlog_hat += coefs.get("rain_anom", 0.0)*r.get("rain_anom", 0.0)
        dlog_hat += coefs.get("dlog_wd", 0.0)*r.get("dlog_wd", 0.0)
        vol = vol * float(np.exp(dlog_hat))
        path.append({"date": r["date"], "volume_hat": vol, "dlog_hat": dlog_hat})
    fc = pd.DataFrame(path)

    # Scenarios
    scen = []
    if "rain_anom" in fut.columns:
        fut_r = fut.copy(); fut_r["rain_anom"] = fut_r["rain_anom"] - 0.10  # −10% anomaly proxy
        vol = vol0; rows=[]
        for _, r in fut_r.iterrows():
            dlog_hat = (coefs.get("dlog_asp",0.0)*r["dlog_asp"] +
                        coefs.get("dlog_wages",0.0)*r["dlog_wages"] +
                        coefs.get("rain_anom",0.0)*r["rain_anom"] +
                        coefs.get("dlog_wd",0.0)*r["dlog_wd"])
            vol = vol * float(np.exp(dlog_hat)); rows.append({"date": r["date"], "volume_hat": vol})
        scen.append(("monsoon_-10pct", pd.DataFrame(rows)))
    if "dlog_asp" in fut.columns:
        fut_p = fut.copy(); fut_p["dlog_asp"] = fut_p["dlog_asp"] + np.log(1.05)  # +5% ASP once
        vol = vol0; rows=[]
        for _, r in fut_p.iterrows():
            dlog_hat = (coefs.get("dlog_asp",0.0)*r["dlog_asp"] +
                        coefs.get("dlog_wages",0.0)*r["dlog_wages"] +
                        coefs.get("rain_anom",0.0)*r["rain_anom"] +
                        coefs.get("dlog_wd",0.0)*r["dlog_wd"])
            vol = vol * float(np.exp(dlog_hat)); rows.append({"date": r["date"], "volume_hat": vol})
        scen.append(("asp_+5pct", pd.DataFrame(rows)))
    if "dlog_wd" in fut.columns:
        fut_d = fut.copy(); fut_d["dlog_wd"] = fut_d["dlog_wd"] + np.log(1.10)  # +10% WD once
        vol = vol0; rows=[]
        for _, r in fut_d.iterrows():
            dlog_hat = (coefs.get("dlog_asp",0.0)*r["dlog_asp"] +
                        coefs.get("dlog_wages",0.0)*r["dlog_wages"] +
                        coefs.get("rain_anom",0.0)*r["rain_anom"] +
                        coefs.get("dlog_wd",0.0)*r["dlog_wd"])
            vol = vol * float(np.exp(dlog_hat)); rows.append({"date": r["date"], "volume_hat": vol})
        scen.append(("wd_+10pct", pd.DataFrame(rows)))

    # join scenarios
    for name, df_s in scen:
        fc = fc.merge(df_s.rename(columns={"volume_hat": f"volume_hat__{name}"}), on="date", how="left")

    return fc


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    sales: str
    distribution: Optional[str]
    macro: Optional[str]
    start: Optional[str]
    end: Optional[str]
    windows: str
    lags: int
    forecast: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="India FMCG — Rural vs Urban demand tracker & drivers")
    ap.add_argument("--sales", required=True)
    ap.add_argument("--distribution", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--forecast", type=int, default=6)
    ap.add_argument("--outdir", default="out_rural")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    SALES = load_sales(args.sales)
    DIST  = load_distribution(args.distribution) if args.distribution else pd.DataFrame()
    MACRO = load_macro(args.macro) if args.macro else pd.DataFrame()

    # Date filters
    if args.start:
        for df in [SALES, DIST, MACRO]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [SALES, DIST, MACRO]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    PANEL = build_panel(SALES, DIST, MACRO)
    if PANEL.groupby(["state","channel"]).size().max() < 12:
        raise ValueError("Need ≥12 months per channel after alignment. Check inputs/date filters.")
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Rural vs Urban
    RVU = rural_vs_urban(PANEL)
    RVU.to_csv(outdir / "rural_vs_urban.csv", index=False)

    # Rolling stats
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_driver_stats(PANEL, windows)
    if not ROLL.empty:
        ROLL.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag rainfall
    LL = rainfall_leadlag(PANEL, max_lag=int(args.lags))
    if not LL.empty:
        LL.to_csv(outdir / "rainfall_leadlag.csv", index=False)

    # Elasticities
    EL = rural_elasticities(PANEL)
    if not EL.empty:
        EL.to_csv(outdir / "rural_elasticities.csv", index=False)

    # Distribution effects
    DE = distribution_effects(PANEL)
    if not DE.empty:
        DE.to_csv(outdir / "distribution_effects.csv", index=False)

    # Forecast
    FC = pd.DataFrame()
    if not EL.empty:
        FC = simple_forecast(PANEL, EL, horizon=int(args.forecast))
        if not FC.empty:
            FC.to_csv(outdir / "forecast.csv", index=False)

    # Summary
    nat = PANEL[(PANEL["state"]=="ALL")]
    latest = nat.groupby("channel").tail(1).set_index("channel")
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "latest": {
            "Rural": {
                "date": str(latest.loc["Rural","date"].date()) if "Rural" in latest.index else None,
                "value_inr": float(latest.loc["Rural","value_inr"]) if "Rural" in latest.index else None,
                "volume": float(latest.loc["Rural","volume"]) if "Rural" in latest.index else None,
                "dlog_value_pct": float(latest.loc["Rural","dlog_value"]*100) if "Rural" in latest.index and pd.notna(latest.loc["Rural","dlog_value"]) else None,
                "dlog_volume_pct": float(latest.loc["Rural","dlog_volume"]*100) if "Rural" in latest.index and pd.notna(latest.loc["Rural","dlog_volume"]) else None
            },
            "Urban": {
                "date": str(latest.loc["Urban","date"].date()) if "Urban" in latest.index else None,
                "value_inr": float(latest.loc["Urban","value_inr"]) if "Urban" in latest.index else None,
                "volume": float(latest.loc["Urban","volume"]) if "Urban" in latest.index else None,
                "dlog_value_pct": float(latest.loc["Urban","dlog_value"]*100) if "Urban" in latest.index and pd.notna(latest.loc["Urban","dlog_value"]) else None,
                "dlog_volume_pct": float(latest.loc["Urban","dlog_volume"]*100) if "Urban" in latest.index and pd.notna(latest.loc["Urban","dlog_volume"]) else None
            }
        },
        "rural_urban_gap_latest_pctpt": {
            "value": (float((latest.loc["Rural","dlog_value"] - latest.loc["Urban","dlog_value"])*100)
                      if {"Rural","Urban"} <= set(latest.index) and pd.notna(latest.loc["Rural","dlog_value"]) and pd.notna(latest.loc["Urban","dlog_value"]) else None),
            "volume": (float((latest.loc["Rural","dlog_volume"] - latest.loc["Urban","dlog_volume"])*100)
                      if {"Rural","Urban"} <= set(latest.index) and pd.notna(latest.loc["Rural","dlog_volume"]) and pd.notna(latest.loc["Urban","dlog_volume"]) else None)
        },
        "elasticities": (EL.set_index("var")[["coef","std_err","t_stat"]].to_dict() if not EL.empty else {}),
        "forecast_horizon_m": int(args.forecast),
        "forecast_available": (not FC.empty)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        sales=args.sales, distribution=(args.distribution or None), macro=(args.macro or None),
        start=(args.start or None), end=(args.end or None), windows=args.windows,
        lags=int(args.lags), forecast=int(args.forecast), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== India FMCG — Rural vs Urban ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']}")
    if summary["latest"]["Rural"]["dlog_value_pct"] is not None:
        print(f"Latest Rural: Value {summary['latest']['Rural']['dlog_value_pct']:+.2f}% m/m, "
              f"Volume {summary['latest']['Rural']['dlog_volume_pct']:+.2f}% m/m")
    if summary["latest"]["Urban"]["dlog_value_pct"] is not None:
        print(f"Latest Urban: Value {summary['latest']['Urban']['dlog_value_pct']:+.2f}% m/m, "
              f"Volume {summary['latest']['Urban']['dlog_volume_pct']:+.2f}% m/m")
    if summary["rural_urban_gap_latest_pctpt"]["value"] is not None:
        print(f"Gap (Rural - Urban) m/m (pp): Value {summary['rural_urban_gap_latest_pctpt']['value']:+.2f}, "
              f"Volume {summary['rural_urban_gap_latest_pctpt']['volume']:+.2f}")
    if not EL.empty:
        print("Elasticities (Rural):", ", ".join([f"{r.var}={r.coef:+.2f}" for _, r in EL.iterrows()]))
    if not FC.empty:
        last_fc = FC.tail(1).iloc[0]
        print(f"Forecast {args.forecast}m ahead: vol_hat={last_fc['volume_hat']:.0f}")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
