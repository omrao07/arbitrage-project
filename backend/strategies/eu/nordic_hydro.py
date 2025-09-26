#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nordic_hydro.py — Reservoirs, inflows, snowpack, hydrobalance & price linkage for the Nordics

What this does
--------------
Given reservoir levels (weekly or daily), inflows, snowpack (optional), hydro generation,
demand, interconnector flows, and prices (Nord Pool areas or System), this script builds:

Core outputs (CSV)
- area_daily.csv            Per-area daily panel: storage_pct/twh, inflow, gen, demand, price, anomalies
- system_daily.csv          System-weighted aggregates (incl. HydroBalance and storage anomalies)
- balance_daily.csv         Compact day-by-day hydrological balance table (system level)
- price_regression.csv      OLS: price ~ storage_pct + inflow + demand (per area & system)
- scenarios_out.csv         Deterministic scenarios (dry/wet/neutral or user-provided scenarios.csv)
- mc_storage_price.csv      Monte Carlo fan (p10/p50/p90) for storage% and price (system)
- summary.json              KPIs (latest storage vs normal, price beta, etc.)
- config.json               Run configuration dump

Inputs (CSV; headers are case-insensitive and flexible)
------------------------------------------------------
--reservoirs reservoirs.csv   REQUIRED (weekly or daily)
  Columns (suggested):
    date, area (NO1..NO5, SE1..SE4, FI, DK1..DK2, SYS), storage_pct (0..100), storage_twh (optional),
    capacity_twh (optional), normal_pct (optional), normal_twh (optional), week(optional)

--inflows inflows.csv         OPTIONAL (daily or weekly; auto-forward-fill to daily)
  Columns:
    date, area, inflow_gwh (or inflow_twh)

--snowpack snow.csv           OPTIONAL (daily/weekly/monthly)
  Columns:
    date, area (or region mapped to area), swe_twh (preferred) or swe_mm (will be kept as auxiliary)

--generation gen.csv          OPTIONAL
  Columns:
    date, area, hydro_gwh (or hydro_mwh)

--demand demand.csv           OPTIONAL
  Columns:
    date, area (or system), demand_gwh (or load_mw → we convert to GWh using 24h if daily)

--prices prices.csv           OPTIONAL
  Columns:
    date, area (or 'system'), price_eur_mwh

--exchange exchange.csv       OPTIONAL (net exports + to outside Nordics; + = export)
  Columns:
    date, area, net_export_gwh

--scenarios scenarios.csv     OPTIONAL (dot-key overrides per 'scenario' id)
  Columns:
    scenario, name, key, value
  Example keys:
    inflow.multiplier = 0.8
    demand.multiplier = 1.02
    price.beta_storage_override = -1.8
    mc.inflow_sd_mult = 1.5
    capacity.area_NO2 = 32.0
    reservoir.min_pct = 5
    reservoir.max_pct = 98

Key options
-----------
--start 2020-01-01
--end   2025-12-31
--system_areas "NO1,NO2,NO3,NO4,NO5,SE1,SE2,SE3,SE4,FI,DK1,DK2"
--scenario baseline
--seed 42
--outdir out_nordic_hydro

Notes
-----
- If storage_twh missing, we use storage_pct * capacity_twh / 100. If capacity_twh missing,
  we compute per-area capacity from the max observed storage_twh or back-solve from the max storage_pct+twh combo.
- Normals: prefer provided normal_twh/normal_pct; else we compute weekly-of-year normals from multi-year history (or 35D rolling mean fallback).
- Monte Carlo is stylized: inflow ~ N(μ, σ) from recent window; generation held near recent mean and clipped to plausible bounds.
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

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.floor("D")

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pct01(x) -> float:
    try:
        v = float(x)
        return v/100.0 if v > 1.5 else v
    except Exception:
        return np.nan

def df_weekly_to_daily(df: pd.DataFrame, date_col: str = "date", group_cols: List[str] = []) -> pd.DataFrame:
    if df.empty: return df
    d = df.copy()
    d[date_col] = to_date(d[date_col])
    d = d.sort_values(date_col)
    pieces = []
    for _, g in d.groupby(group_cols) if group_cols else [(None, d)]:
        g = g.sort_values(date_col).set_index(date_col)
        # forward-fill to daily across full span
        full = pd.DataFrame(index=pd.date_range(g.index.min(), g.index.max(), freq="D"))
        h = pd.concat([full, g], axis=1)
        h = h.ffill().reset_index().rename(columns={"index": date_col})
        if group_cols:
            for c in group_cols:
                h[c] = g.reset_index().iloc[0][c]
        pieces.append(h)
    return pd.concat(pieces, ignore_index=True)

def normalize_area(area: str) -> str:
    if area is None: return ""
    s = str(area).strip().upper().replace("AREA_","")
    mapping = {"SYSTEM":"SYS","SYSTEM PRICE":"SYS","SE":"SE","NO":"NO","FI":"FI","DK":"DK"}
    return mapping.get(s, s)

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ----------------------------- loaders -----------------------------

def load_reservoirs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"area") or ncol(df,"price_area") or "area"): "area",
        (ncol(df,"storage_pct") or ncol(df,"res_pct") or "storage_pct"): "storage_pct",
        (ncol(df,"storage_twh") or ncol(df,"res_twh") or "storage_twh"): "storage_twh",
        (ncol(df,"capacity_twh") or ncol(df,"cap_twh") or "capacity_twh"): "capacity_twh",
        (ncol(df,"normal_pct") or "normal_pct"): "normal_pct",
        (ncol(df,"normal_twh") or "normal_twh"): "normal_twh",
        (ncol(df,"week") or "week"): "week",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["area"] = df["area"].apply(normalize_area)
    for c in ["storage_pct","storage_twh","capacity_twh","normal_pct","normal_twh"]:
        if c in df.columns: df[c] = num(df[c])
    return df.sort_values(["area","date"])

def load_inflows(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"area") or "area"): "area",
           (ncol(df,"inflow_gwh") or ncol(df,"inflow_twh") or "inflow_gwh"): "inflow_gwh"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["area"] = df["area"].apply(normalize_area)
    if "inflow_gwh" in df.columns:
        df["inflow_gwh"] = num(df["inflow_gwh"])
    if ncol(df, "inflow_twh"):
        df["inflow_gwh"] = num(df[ncol(df, "inflow_twh")]) * 1000.0
    return df.sort_values(["area","date"])

def load_snow(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"area") or ncol(df,"region") or "area"): "area",
           (ncol(df,"swe_twh") or "swe_twh"): "swe_twh",
           (ncol(df,"swe_mm") or "swe_mm"): "swe_mm"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["area"] = df["area"].apply(normalize_area)
    for c in ["swe_twh","swe_mm"]:
        if c in df.columns: df[c] = num(df[c])
    return df.sort_values(["area","date"])

def load_generation(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"area") or "area"): "area",
           (ncol(df,"hydro_gwh") or ncol(df,"hydro_mwh") or "hydro_gwh"): "hydro_gwh"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["area"] = df["area"].apply(normalize_area)
    if ncol(df,"hydro_mwh"): df["hydro_gwh"] = num(df[ncol(df,"hydro_mwh")]) / 1000.0
    if "hydro_gwh" in df.columns: df["hydro_gwh"] = num(df["hydro_gwh"])
    return df.sort_values(["area","date"])

def load_demand(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    # Support "load_mw" or "demand_gwh"
    date_c = ncol(df,"date") or df.columns[0]
    area_c = ncol(df,"area") or "area"
    df = df.rename(columns={date_c:"date", area_c:"area"})
    df["date"] = to_date(df["date"])
    df["area"] = df["area"].apply(normalize_area)
    if ncol(df,"demand_gwh"):
        df["demand_gwh"] = num(df[ncol(df,"demand_gwh")])
    elif ncol(df,"load_mw"):
        # daily energy ≈ mean load * 24 / 1000
        df["demand_gwh"] = num(df[ncol(df,"load_mw")]) * 24.0 / 1000.0
    return df[["date","area","demand_gwh"]].sort_values(["area","date"])

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"area") or ncol(df,"price_area") or ncol(df,"system") or "area"): "area",
           (ncol(df,"price_eur_mwh") or ncol(df,"price") or "price_eur_mwh"): "price_eur_mwh"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["area"] = df["area"].apply(normalize_area)
    df["price_eur_mwh"] = num(df["price_eur_mwh"])
    return df.sort_values(["area","date"])

def load_exchange(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"area") or "area"): "area",
           (ncol(df,"net_export_gwh") or ncol(df,"net_exports_gwh") or "net_export_gwh"): "net_export_gwh"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["area"] = df["area"].apply(normalize_area)
    df["net_export_gwh"] = num(df["net_export_gwh"])
    return df.sort_values(["area","date"])

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"): "scenario",
           (ncol(df,"name") or "name"): "name",
           (ncol(df,"key") or "key"): "key",
           (ncol(df,"value") or "value"): "value"}
    return df.rename(columns=ren)


# ----------------------------- core transforms -----------------------------

def backsolve_capacity(area_df: pd.DataFrame) -> float:
    """Infer capacity_twh if missing: prefer provided, else use max storage_twh or (storage_twh / (pct/100))."""
    if not area_df.empty and "capacity_twh" in area_df.columns and area_df["capacity_twh"].notna().any():
        return float(area_df["capacity_twh"].dropna().iloc[0])
    cap_from_twh = area_df["storage_twh"].max(skipna=True) if "storage_twh" in area_df.columns else np.nan
    cap_from_pct = np.nan
    if "storage_twh" in area_df.columns and "storage_pct" in area_df.columns:
        tmp = area_df.dropna(subset=["storage_twh","storage_pct"]).copy()
        if not tmp.empty and (tmp["storage_pct"]>0).any():
            cap_from_pct = float((tmp["storage_twh"] / (tmp["storage_pct"]/100.0)).max())
    # choose reasonable
    candidates = [x for x in [cap_from_twh, cap_from_pct] if np.isfinite(x)]
    return float(max(candidates)) if candidates else np.nan

def compute_storage_fields(res: pd.DataFrame) -> pd.DataFrame:
    df = res.copy()
    out = []
    for a, g in df.groupby("area"):
        cap = backsolve_capacity(g)
        gg = g.copy()
        if "storage_twh" not in gg.columns or gg["storage_twh"].isna().all():
            if np.isfinite(cap) and "storage_pct" in gg.columns:
                gg["storage_twh"] = (num(gg["storage_pct"]) / 100.0) * cap
        gg["capacity_twh"] = gg.get("capacity_twh", np.nan)
        if gg["capacity_twh"].isna().all() and np.isfinite(cap):
            gg["capacity_twh"] = cap
        # weekly→daily if needed
        out.append(gg)
    d = pd.concat(out, ignore_index=True)
    # if weekly, densify to daily
    # Heuristic: if median inter-row gap > 2 days → treat as weekly and expand
    gaps = d.groupby("area")["date"].diff().dt.days
    if gaps.dropna().median() and gaps.dropna().median() > 2:
        d = df_weekly_to_daily(d, "date", ["area"])
    return d.sort_values(["area","date"])

def compute_normals(d: pd.DataFrame) -> pd.DataFrame:
    """Fill normal_twh/normal_pct using provided or history-based weekly-of-year averages."""
    x = d.copy()
    if ("normal_twh" in x.columns and x["normal_twh"].notna().any()) or ("normal_pct" in x.columns and x["normal_pct"].notna().any()):
        return x
    x["week"] = x["date"].dt.isocalendar().week.astype(int)
    normals = (x.groupby(["area","week"], as_index=False)
                 .agg(normal_twh=("storage_twh","mean"),
                      normal_pct=("storage_pct","mean")))
    x = x.merge(normals, on=["area","week"], how="left")
    # fallback: 35D rolling mean per area
    if x["normal_twh"].isna().all() and "storage_twh" in x.columns:
        x["normal_twh"] = (x.groupby("area")["storage_twh"].transform(lambda s: s.rolling(35, min_periods=7).mean()))
    if x["normal_pct"].isna().all() and "storage_pct" in x.columns:
        x["normal_pct"] = (x.groupby("area")["storage_pct"].transform(lambda s: s.rolling(35, min_periods=7).mean()))
    return x

def join_all(res_d: pd.DataFrame,
             infl: pd.DataFrame,
             snow: pd.DataFrame,
             gen: pd.DataFrame,
             dem: pd.DataFrame,
             price: pd.DataFrame,
             exch: pd.DataFrame) -> pd.DataFrame:
    df = res_d.copy()
    for tab, cols in [(infl, ["inflow_gwh"]), (snow, ["swe_twh","swe_mm"]),
                      (gen, ["hydro_gwh"]), (dem, ["demand_gwh"]), (price, ["price_eur_mwh"]),
                      (exch, ["net_export_gwh"])]:
        if tab is None or tab.empty:
            continue
        tmp = tab.copy()
        df = df.merge(tmp, on=["date","area"], how="left")
    # anomalies vs normal
    if "normal_twh" in df.columns and "storage_twh" in df.columns:
        df["storage_dev_twh"] = df["storage_twh"] - df["normal_twh"]
    if "normal_pct" in df.columns and "storage_pct" in df.columns:
        df["storage_dev_pp"] = df["storage_pct"] - df["normal_pct"]
    # rolling sums/means
    if "inflow_gwh" in df.columns:
        df["inflow_7d_gwh"] = df.groupby("area")["inflow_gwh"].transform(lambda s: s.rolling(7, min_periods=3).sum())
    if "hydro_gwh" in df.columns:
        df["hydro_7d_gwh"] = df.groupby("area")["hydro_gwh"].transform(lambda s: s.rolling(7, min_periods=3).sum())
    if "price_eur_mwh" in df.columns:
        df["price_7d"] = df.groupby("area")["price_eur_mwh"].transform(lambda s: s.rolling(7, min_periods=3).mean())
    return df

def system_aggregate(df: pd.DataFrame, system_areas: List[str]) -> pd.DataFrame:
    x = df[df["area"].isin(system_areas)].copy()
    if x.empty: return pd.DataFrame()
    # Energy-weighted where appropriate (capacity as proxy)
    caps = x.groupby("area", as_index=False)["capacity_twh"].max()
    caps["w"] = caps["capacity_twh"] / caps["capacity_twh"].sum()
    x = x.merge(caps[["area","w"]], on="area", how="left")
    agg_cols_sum = ["storage_twh","normal_twh","inflow_gwh","inflow_7d_gwh","hydro_gwh","hydro_7d_gwh","demand_gwh","net_export_gwh"]
    agg_cols_wavg = ["storage_pct","normal_pct","price_eur_mwh","price_7d"]
    g = x.groupby("date").agg({c:"sum" for c in agg_cols_sum if c in x.columns})
    for c in agg_cols_wavg:
        if c in x.columns:
            g[c] = x.groupby("date").apply(lambda s: np.average(s[c], weights=s["w"].fillna(0))).values
    g = g.reset_index()
    # derived
    if {"storage_twh","normal_twh"}.issubset(g.columns):
        g["storage_dev_twh"] = g["storage_twh"] - g["normal_twh"]
    if {"storage_pct","normal_pct"}.issubset(g.columns):
        g["storage_dev_pp"] = g["storage_pct"] - g["normal_pct"]
    g["area"] = "SYS"
    return g

def ols_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit per-area OLS: price ~ storage_pct + inflow_7d + demand_gwh (all standardized).
    """
    out = []
    for a, g in df.groupby("area"):
        gg = g.dropna(subset=["price_eur_mwh"])
        if gg.shape[0] < 30: 
            continue
        Xcols = [c for c in ["storage_pct","inflow_7d_gwh","demand_gwh"] if c in gg.columns]
        if not Xcols: 
            continue
        Z = gg[Xcols].astype(float)
        Z = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-9)
        Y = gg["price_eur_mwh"].values.astype(float)
        X = np.column_stack([np.ones(len(Z))] + [Z[c].values for c in Xcols])
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            yhat = X @ beta
            r2 = 1.0 - np.sum((Y - yhat)**2) / max(1e-12, np.sum((Y - Y.mean())**2))
            row = {"area": a, "n": int(len(Y)), "r2": float(r2), "intercept": float(beta[0])}
            for i, c in enumerate(Xcols, start=1):
                row[f"beta_{c}_std"] = float(beta[i])
            out.append(row)
        except Exception:
            continue
    return pd.DataFrame(out).sort_values("r2", ascending=False)

def scenario_apply(df_sys: pd.DataFrame, betas: pd.DataFrame, scen: Dict[str,str]) -> pd.DataFrame:
    """
    Deterministic scenario: apply multipliers/overrides, estimate price via standardized betas (system row).
    """
    if df_sys.empty or betas.empty:
        return pd.DataFrame()
    b = betas[betas["area"]=="SYS"].iloc[0] if "SYS" in betas["area"].values else betas.iloc[0]
    recent = df_sys.sort_values("date").tail(30).copy()
    # Apply simple multipliers
    inflow_mult = float(scen.get("inflow.multiplier", 1.0))
    demand_mult = float(scen.get("demand.multiplier", 1.0))
    if "inflow_gwh" in recent.columns:
        recent["inflow_7d_gwh"] = recent["inflow_7d_gwh"] * inflow_mult if "inflow_7d_gwh" in recent.columns else recent["inflow_gwh"] * inflow_mult
    if "demand_gwh" in recent.columns:
        recent["demand_gwh"] = recent["demand_gwh"] * demand_mult
    # Standardize on recent window
    Xcols = [c for c in ["storage_pct","inflow_7d_gwh","demand_gwh"] if c in recent.columns]
    Z = recent[Xcols].astype(float)
    Zstd = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-9)
    # Price estimate
    p_base = recent["price_eur_mwh"].mean() if "price_eur_mwh" in recent.columns else np.nan
    beta_storage_override = scen.get("price.beta_storage_override", None)
    betas_map = {c: b.get(f"beta_{c}_std", np.nan) for c in Xcols}
    if beta_storage_override is not None:
        betas_map["storage_pct"] = float(beta_storage_override)
    yhat = p_base + np.sum(Zstd.mean(axis=0).values * np.array([betas_map.get(c, 0.0) for c in Xcols]))
    return pd.DataFrame([{"scenario": "custom", "name": scen.get("name","custom"), "delta_price_eur_mwh": float(yhat - p_base), "new_price_eur_mwh": float(yhat)}])

def mc_paths(df_sys: pd.DataFrame, days: int, seed: Optional[int], scen: Dict[str,str]) -> pd.DataFrame:
    """
    Stylized MC on storage% and price using inflow shocks; generation held near recent mean.
    State update (system):
      storage_twh(t+1) = storage_twh(t) + inflow(t) - hydro_gen(t) - net_exports(t)
      storage_pct = storage_twh / capacity
      price_t ≈ intercept + β_s * z(storage_pct) + β_i * z(inflow_7d) + β_d * z(demand)
    """
    if df_sys.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    hist = df_sys.sort_values("date").copy()
    cap = float(hist["storage_twh"].max()) / max(hist["storage_pct"].max()/100.0, 1e-6) if {"storage_twh","storage_pct"}.issubset(hist.columns) and hist["storage_pct"].max() else np.nan
    if not np.isfinite(cap):
        return pd.DataFrame()
    # recent stats
    recent = hist.tail(120).copy()
    inflow = recent.get("inflow_gwh", pd.Series([0]*len(recent)))
    mu_in, sd_in = float(np.nanmean(inflow)), float(np.nanstd(inflow))
    sd_in *= float(scen.get("mc.inflow_sd_mult", 1.0))
    gen = float(np.nanmean(recent.get("hydro_gwh", pd.Series([0]*len(recent)))))
    dem = float(np.nanmean(recent.get("demand_gwh", pd.Series([0]*len(recent)))))
    exports = float(np.nanmean(recent.get("net_export_gwh", pd.Series([0]*len(recent)))))
    demand_mult = float(scen.get("demand.multiplier", 1.0))
    gen = max(0.0, gen)  # guardrail
    storage0 = float(recent["storage_twh"].iloc[-1])
    price_base = float(np.nanmean(recent.get("price_eur_mwh", pd.Series([0]*len(recent)))))
    # simple betas from recent regression (fallback constants)
    beta_s, beta_i, beta_d = -2.0, -0.3, 0.5
    # try to estimate from recent standardized corr to price
    if {"storage_pct","price_eur_mwh"}.issubset(recent.columns):
        zz = (recent["storage_pct"] - recent["storage_pct"].mean())/(recent["storage_pct"].std(ddof=0)+1e-9)
        pp = recent["price_eur_mwh"]
        try:
            beta_s = float(np.linalg.lstsq(np.column_stack([np.ones(len(zz)), zz]), pp, rcond=None)[0][1])
        except Exception:
            pass
    beta_storage_override = scen.get("price.beta_storage_override", None)
    if beta_storage_override is not None:
        beta_s = float(beta_storage_override)

    N = 200  # paths
    T = days
    paths_pct = np.zeros((N, T))
    paths_price = np.zeros((N, T))
    min_pct = float(scen.get("reservoir.min_pct", 2.0))
    max_pct = float(scen.get("reservoir.max_pct", 98.0))
    for n in range(N):
        st = storage0
        inflow_hist = []
        for t in range(T):
            infl_t = float(rng.normal(mu_in, sd_in))
            inflow_hist.append(infl_t)
            # simple seasonal gen: clip to recent mean; allow mild demand scaling
            gen_t = gen * demand_mult
            # update storage
            st = max(0.0, min(st + infl_t - gen_t - exports, cap * (max_pct/100.0)))
            pct = (st / cap) * 100.0
            # z-features for price
            z_s = (pct - recent["storage_pct"].mean()) / (recent["storage_pct"].std(ddof=0)+1e-9)
            z_i = (pd.Series(inflow_hist[-7:]).sum() if len(inflow_hist)>=7 else np.sum(inflow_hist))  # 7d sum
            # normalize inflow 7d by recent stats
            mu7 = (inflow.rolling(7,min_periods=3).sum()).dropna()
            z_i = (z_i - (mu7.mean() if not mu7.empty else (7*mu_in))) / ((mu7.std(ddof=0)+1e-9) if not mu7.empty else (np.sqrt(7)*sd_in+1e-9))
            z_d = 0.0  # keep demand constant in MC
            price_t = price_base + beta_s*z_s + beta_i*z_i + beta_d*z_d
            # floors/ceilings
            price_t = float(np.clip(price_t, -50, 500))
            paths_pct[n, t] = pct
            paths_price[n, t] = price_t
    # quantiles
    q10 = np.nanpercentile(paths_pct, 10, axis=0)
    q50 = np.nanpercentile(paths_pct, 50, axis=0)
    q90 = np.nanpercentile(paths_pct, 90, axis=0)
    p10 = np.nanpercentile(paths_price, 10, axis=0)
    p50 = np.nanpercentile(paths_price, 50, axis=0)
    p90 = np.nanpercentile(paths_price, 90, axis=0)
    dates = pd.date_range(hist["date"].iloc[-1] + pd.Timedelta(days=1), periods=T, freq="D")
    out = pd.DataFrame({"date": dates,
                        "storage_pct_p10": q10, "storage_pct_p50": q50, "storage_pct_p90": q90,
                        "price_p10": p10, "price_p50": p50, "price_p90": p90})
    return out


# ----------------------------- CLI -----------------------------

@dataclass
class Config:
    reservoirs: str
    inflows: Optional[str]
    snowpack: Optional[str]
    generation: Optional[str]
    demand: Optional[str]
    prices: Optional[str]
    exchange: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    system_areas: List[str]
    scenario: str
    seed: Optional[int]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Nordic hydrobalance & price linkage")
    ap.add_argument("--reservoirs", required=True)
    ap.add_argument("--inflows", default="")
    ap.add_argument("--snowpack", default="")
    ap.add_argument("--generation", default="")
    ap.add_argument("--demand", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--exchange", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--system_areas", default="NO1,NO2,NO3,NO4,NO5,SE1,SE2,SE3,SE4,FI,DK1,DK2")
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="out_nordic_hydro")
    return ap.parse_args()


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    start = pd.to_datetime(args.start); end = pd.to_datetime(args.end)
    system_areas = [s.strip().upper() for s in args.system_areas.split(",") if s.strip()]

    # Load
    res = load_reservoirs(args.reservoirs)
    infl = load_inflows(args.inflows) if args.inflows else pd.DataFrame()
    snow = load_snow(args.snowpack) if args.snowpack else pd.DataFrame()
    gen  = load_generation(args.generation) if args.generation else pd.DataFrame()
    dem  = load_demand(args.demand) if args.demand else pd.DataFrame()
    pric = load_prices(args.prices) if args.prices else pd.DataFrame()
    exch = load_exchange(args.exchange) if args.exchange else pd.DataFrame()
    scen_df = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Compute storage fields & normals, then join all
    res2 = compute_storage_fields(res)
    res2 = compute_normals(res2)
    panel = join_all(res2, infl, snow, gen, dem, pric, exch)
    # Filter window
    panel = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()
    panel = panel.sort_values(["area","date"])

    # Outputs: area_daily
    area_out = panel.copy()
    ensure_cols(area_out, ["storage_pct","storage_twh","normal_pct","normal_twh","storage_dev_pp","storage_dev_twh",
                           "inflow_gwh","hydro_gwh","demand_gwh","net_export_gwh","price_eur_mwh"])
    area_out.to_csv(outdir / "area_daily.csv", index=False)

    # System aggregate
    sys = system_aggregate(area_out, system_areas)
    sys.to_csv(outdir / "system_daily.csv", index=False)

    # Balance compact
    bal_cols = ["date","storage_pct","normal_pct","storage_dev_pp","storage_twh","normal_twh","storage_dev_twh","inflow_gwh","hydro_gwh","demand_gwh","price_eur_mwh"]
    balance = sys[bal_cols].copy() if not sys.empty else pd.DataFrame(columns=bal_cols)
    balance.to_csv(outdir / "balance_daily.csv", index=False)

    # Price regression per area + system
    reg = ols_price(pd.concat([area_out, sys], ignore_index=True))
    reg.to_csv(outdir / "price_regression.csv", index=False)

    # Scenarios
    overrides = {}
    if not scen_df.empty and args.scenario != "baseline":
        sub = scen_df[scen_df["scenario"]==args.scenario]
        overrides = {str(k): str(v) for k, v in zip(sub["key"], sub["value"])}
        overrides["name"] = sub["name"].iloc[0] if "name" in sub.columns and not sub["name"].isna().all() else args.scenario
    scen_out = scenario_apply(sys, reg, overrides) if overrides else pd.DataFrame()
    if not scen_out.empty:
        scen_out.to_csv(outdir / "scenarios_out.csv", index=False)

    # Monte Carlo
    mc = mc_paths(sys, days=90, seed=args.seed, scen=overrides) if not sys.empty else pd.DataFrame()
    if not mc.empty:
        mc.to_csv(outdir / "mc_storage_price.csv", index=False)

    # Summary
    latest = sys["date"].max() if not sys.empty else None
    kpi = {
        "period": {"start": args.start, "end": args.end},
        "areas": int(area_out["area"].nunique()) if not area_out.empty else 0,
        "system_latest_date": str(latest.date()) if latest is not None else None,
        "system_storage_pct_latest": float(sys["storage_pct"].iloc[-1]) if not sys.empty and sys["storage_pct"].notna().any() else None,
        "system_storage_dev_pp_latest": float(sys["storage_dev_pp"].iloc[-1]) if not sys.empty and "storage_dev_pp" in sys.columns else None,
        "system_price_latest": float(sys["price_eur_mwh"].iloc[-1]) if not sys.empty and "price_eur_mwh" in sys.columns else None,
        "regression_top": (reg.head(6).to_dict(orient="records") if not reg.empty else []),
        "scenario": args.scenario if overrides else "baseline",
        "mc_horizon_days": 90 if not mc.empty else 0
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))

    # Config dump
    cfg = asdict(Config(
        reservoirs=args.reservoirs, inflows=args.inflows or None, snowpack=args.snowpack or None,
        generation=args.generation or None, demand=args.demand or None, prices=args.prices or None,
        exchange=args.exchange or None, scenarios=args.scenarios or None, start=args.start, end=args.end,
        system_areas=system_areas, scenario=args.scenario, seed=args.seed, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Nordic Hydro ==")
    if not sys.empty:
        print(f"Latest {sys['date'].iloc[-1]:%Y-%m-%d} | Storage {sys['storage_pct'].iloc[-1]:.1f}% ({sys['storage_dev_pp'].iloc[-1]:+.1f} pp vs normal) | Price {sys.get('price_eur_mwh', pd.Series([np.nan])).iloc[-1]:.1f} €/MWh")
    if not reg.empty:
        row = reg[reg["area"]=="SYS"].head(1) if "SYS" in reg["area"].values else reg.head(1)
        if not row.empty:
            print("Price OLS (system):", row.to_dict(orient="records")[0])
    if not scen_out.empty:
        print("Scenario:", scen_out.to_dict(orient="records")[0])
    if not mc.empty:
        print("MC horizon: 90 days (p50 final storage ≈ %.1f%%, price ≈ %.1f €/MWh)" % (mc["storage_pct_p50"].iloc[-1], mc["price_p50"].iloc[-1]))
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
