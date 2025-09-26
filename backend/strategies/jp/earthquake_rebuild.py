#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
earthquake_rebuild.py — From quake shocks to reconstruction paths, IRFs & materials demand
-----------------------------------------------------------------------------------------

What this does
==============
A research toolkit to analyze **earthquake rebuild cycles** for a given region, linking
events → macro/sector impacts → reconstruction spending paths → materials demand and
capacity gaps. It ingests a quake catalog, macro & sector series, fiscal/insurance
flows, equities prices, and supply-side capacity to:

1) Clean & align a region–date panel (weekly or monthly)
   • Quake shock series (by magnitude or damage), fiscal disbursements, insurance payouts
   • Macro: IP, GDP, retail, employment, CPI (materials), permits/starts, tourism, electricity
   • Supply: cement/steel capacity, port throughput, logistics index

2) Event studies
   • Abnormal returns around quake dates for sector equities (cement, construction, insurers, utilities)
   • CARs around announcement & disbursement dates of fiscal packages (optional)

3) Local projections (Jordà) as impulse responses (IRFs)
   • Δy_{t+h} ~ α_h + β_h * shock_t + Γ_h * controls_t + seasonality + ε_{t+h},  h=0..H
   • HAC/Newey–West SEs; outputs β_h for each variable as IRF curves

4) Reconstruction path model (S-curve)
   • Damages → spending path by category (residential / infrastructure / commercial)
   • Speed + mid-point parameters per category
   • Produces cumulative and period flows; aligns with fiscal & insurance timing

5) Materials demand & capacity gap
   • Translate spending → material volumes via intensities (cement, steel, lumber, asphalt)
   • Compare to local capacity & import slack → estimate deficits & price-pressure proxy

6) Scenarios
   • Speed multipliers, aid scaling, import relaxation, labor bottlenecks
   • Forward projections of spending, material demand and gaps

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--quakes quakes.csv              REQUIRED
  Columns (any subset):
    date, region[, country], magnitude[, Mw], intensity_mmi[, mmi],
    damages_usd[, loss_usd, econ_loss_usd], fatalities (optional)

--macro macro.csv                OPTIONAL (monthly or weekly)
  Columns (any subset):
    date, region[, country],
    ip_index[, industrial_prod], gdp_usd[, gdp], retail_idx, unemployment_rate,
    building_permits[, permits], housing_starts, electricity_demand,
    cpi_construction[, cpi_materials], tourism_arrivals[, arrivals]

--fiscal fiscal.csv              OPTIONAL
  Columns:
    date, region, package_usd[, amount_usd], type[, label]  # announce/disbursement mix acceptable

--insurance insurance.csv        OPTIONAL
  Columns:
    date, region, insured_loss_usd[, payouts_usd], catbond_issuance_usd[, rol_rate]

--prices prices.csv              OPTIONAL (daily)
  Columns:
    date, ticker[, symbol], adj_close[, close], sector[, group]  # sector tagging optional

--bench bench.csv                OPTIONAL (daily)
  Columns:
    date, ticker, adj_close

--supply supply.csv              OPTIONAL
  Columns:
    date, region,
    cement_capacity_tons, steel_capacity_tons, port_throughput_teu,
    logistics_index[, supply_chain_index]

--permits permits.csv            OPTIONAL (if not in macro)
  Columns:
    date, region, building_permits[, housing_starts]

CLI (key)
---------
--region "JP-KANTO"             Target region string to filter (case-insensitive substring match)
--freq monthly|weekly           Output frequency for panel/IRFs
--shock "damage"                "magnitude" or "damage" (how to size the shock)
--event_window 30               Days around quake date for equities event study (if prices provided)
--lp_h 18                       IRF horizon H (months or weeks depending on --freq)
--lags 3                        Lags of controls in local projections
--speed_res 0.20                S-curve speed (k) for residential
--speed_inf 0.12                S-curve speed (k) for infrastructure
--speed_com 0.16                S-curve speed (k) for commercial
--mid_res 6                     S-curve mid-point t0 (periods) for residential
--mid_inf 10                    S-curve mid-point t0 (periods) for infrastructure
--mid_com 8                     S-curve mid-point t0 (periods) for commercial
--share_res 0.45                Damage share allocation to residential
--share_inf 0.40                ... to infrastructure
--share_com 0.15                ... to commercial
--aid_scale 1.0                 Multiplier on fiscal+insurance resources
--import_relax 0.0              Extra import slack (fraction of local capacity, 0–1)
--labor_bottleneck 0.0          Fractional cap on deployable spending per period (0–1 reduces speed)
--cement_int 0.11               Tons of cement per $1k spend (example: 0.11 t / $1k)
--steel_int 0.08                Tons of steel per $1k spend
--lumber_int 0.15               m³ lumber per $1k spend
--start / --end                 Date filters (YYYY-MM-DD)
--outdir out_quake              Output directory
--min_obs 60                    Minimum obs for IRFs per variable

Outputs
-------
- cleaned_quakes.csv            Filtered & normalized quake list for the target region
- panel.csv                     Region–date panel with shocks & controls
- event_study_equities.csv      CARs around quake dates (if prices provided)
- irf_local_projections.csv     β_h and SEs per variable & horizon
- rebuild_path.csv              Period & cumulative reconstruction spending (by category)
- materials_demand.csv          Material volumes & capacity gaps per period
- scenarios.csv                 Scenario-adjusted forward projections (if any deviation from base)
- summary.json                  Headline diagnostics
- config.json                   Echo run configuration

DISCLAIMER
----------
Research tooling with simplifying assumptions. Validate with local knowledge & higher fidelity data.
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

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def to_date(s: pd.Series) -> pd.Series:
    return to_dt(s).dt.date

def eom(d: pd.Series) -> pd.Series:
    dt = to_dt(d)
    return (dt + pd.offsets.MonthEnd(0))

def eow(d: pd.Series) -> pd.Series:
    dt = to_dt(d)
    return (dt + pd.offsets.Week(weekday=6))  # Sunday

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace(0, np.nan)
    return np.log(s) - np.log(s.shift(1))

def winsor(s: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

# OLS + HAC (Newey–West)
def ols_beta_resid(X: np.ndarray, y: np.ndarray):
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XTX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XTX_inv: np.ndarray, L: int) -> np.ndarray:
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1) + 1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XTX_inv @ S @ XTX_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


# ----------------------------- loaders -----------------------------

def load_quakes(path: str, region_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df, "date"); r = ncol(df, "region","country","place","location")
    mw = ncol(df, "magnitude","mw"); mmi = ncol(df, "intensity_mmi","mmi")
    dmg = ncol(df, "damages_usd","loss_usd","econ_loss_usd")
    fat = ncol(df, "fatalities","deaths")
    if not d:
        raise ValueError("quakes.csv must include a date column.")
    df = df.rename(columns={d:"date"})
    if r: df = df.rename(columns={r:"region"})
    if mw: df = df.rename(columns={mw:"magnitude"})
    if mmi: df = df.rename(columns={mmi:"intensity_mmi"})
    if dmg: df = df.rename(columns={dmg:"damages_usd"})
    if fat: df = df.rename(columns={fat:"fatalities"})
    df["date"] = to_dt(df["date"])
    if "region" not in df.columns:
        df["region"] = region_key
    # Filter by region substring (case-insensitive)
    key = str(region_key).lower()
    df = df[df["region"].astype(str).str.lower().str.contains(key)]
    # Normalize damages
    if "damages_usd" in df.columns:
        df["damages_usd"] = safe_num(df["damages_usd"])
    # Heuristic damages if missing
    if "damages_usd" not in df.columns or df["damages_usd"].isna().all():
        # proxy: damages ≈ A * exp(B*(Mw-6)), with cap at very small events
        A, B = 5e7, 1.6
        mag = safe_num(df.get("magnitude", pd.Series(np.nan)))
        df["damages_usd"] = A * np.exp(B * (mag.fillna(6.0) - 6.0))
    # Keep required
    keep = ["date","region","magnitude","intensity_mmi","damages_usd","fatalities"]
    for c in keep:
        if c not in df.columns: df[c] = np.nan
    return df[keep].sort_values("date")

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); r = ncol(df, "region","country")
    if not (d and r): raise ValueError("macro.csv needs date and region/country.")
    df = df.rename(columns={d:"date", r:"region"})
    df["date"] = to_dt(df["date"])
    df["region"] = df["region"].astype(str).str.strip()
    # Optional columns normalized
    ren = {
        ncol(df,"ip_index","industrial_prod","ip"): "ip_index",
        ncol(df,"gdp_usd","gdp"): "gdp_usd",
        ncol(df,"retail_idx","retail_sales","retail"): "retail_idx",
        ncol(df,"unemployment_rate","unemp_rate","u_rate"): "unemployment_rate",
        ncol(df,"building_permits","permits"): "building_permits",
        ncol(df,"housing_starts","starts"): "housing_starts",
        ncol(df,"electricity_demand","electricity","power_demand"): "electricity_demand",
        ncol(df,"cpi_construction","cpi_materials","cpi_const"): "cpi_construction",
        ncol(df,"tourism_arrivals","arrivals","visitors"): "tourism_arrivals"
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    # Numeric
    for c in ["ip_index","gdp_usd","retail_idx","unemployment_rate","building_permits","housing_starts",
              "electricity_demand","cpi_construction","tourism_arrivals"]:
        if c in df.columns: df[c] = safe_num(df[c])
    return df

def load_fiscal(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); r = ncol(df,"region","country"); a = ncol(df,"package_usd","amount_usd","value_usd","value")
    typ = ncol(df,"type","label","event")
    if not (d and r and a): raise ValueError("fiscal.csv needs date, region, package_usd.")
    df = df.rename(columns={d:"date", r:"region", a:"package_usd"})
    if typ: df = df.rename(columns={typ:"type"})
    df["date"] = to_dt(df["date"])
    df["region"] = df["region"].astype(str).str.strip()
    df["package_usd"] = safe_num(df["package_usd"])
    if "type" in df.columns: df["type"] = df["type"].astype(str).str.upper().str.strip()
    return df

def load_insurance(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); r = ncol(df,"region","country")
    ins = ncol(df,"insured_loss_usd","payouts_usd","claims_paid_usd")
    cb = ncol(df,"catbond_issuance_usd","catbond_usd")
    if not (d and r and ins): raise ValueError("insurance.csv needs date, region, insured_loss_usd/payouts_usd.")
    df = df.rename(columns={d:"date", r:"region", ins:"insured_loss_usd"})
    if cb: df = df.rename(columns={cb:"catbond_issuance_usd"})
    df["date"] = to_dt(df["date"])
    df["region"] = df["region"].astype(str).str.strip()
    for c in ["insured_loss_usd","catbond_issuance_usd"]:
        if c in df.columns: df[c] = safe_num(df[c])
    return df

def load_supply(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); r = ncol(df,"region","country")
    if not (d and r): raise ValueError("supply.csv needs date and region.")
    df = df.rename(columns={d:"date", r:"region"})
    df["date"] = to_dt(df["date"])
    df["region"] = df["region"].astype(str).str.strip()
    ren = {
        ncol(df,"cement_capacity_tons","cement_tons","cement_cap"): "cement_capacity_tons",
        ncol(df,"steel_capacity_tons","steel_tons","steel_cap"): "steel_capacity_tons",
        ncol(df,"port_throughput_teu","port_throughput","teu"): "port_throughput_teu",
        ncol(df,"logistics_index","supply_chain_index","logistics"): "logistics_index"
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for c in ["cement_capacity_tons","steel_capacity_tons","port_throughput_teu","logistics_index"]:
        if c in df.columns: df[c] = safe_num(df[c])
    return df

def load_permits(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); r = ncol(df,"region","country")
    bp = ncol(df,"building_permits","permits")
    hs = ncol(df,"housing_starts","starts")
    if not (d and r and (bp or hs)): raise ValueError("permits.csv needs date, region and permits/starts.")
    df = df.rename(columns={d:"date", r:"region"})
    if bp: df = df.rename(columns={bp:"building_permits"})
    if hs: df = df.rename(columns={hs:"housing_starts"})
    df["date"] = to_dt(df["date"]); df["region"] = df["region"].astype(str).str.strip()
    for c in ["building_permits","housing_starts"]:
        if c in df.columns: df[c] = safe_num(df[c])
    return df

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"ticker","symbol"); p = ncol(df,"adj_close","close")
    sec = ncol(df,"sector","group")
    if not (d and t and p): raise ValueError("prices.csv needs date, ticker, adj_close/close.")
    df = df.rename(columns={d:"date", t:"ticker", p:"adj_close"})
    if sec: df = df.rename(columns={sec:"sector"})
    df["date"] = to_dt(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "sector" in df.columns: df["sector"] = df["sector"].astype(str).str.upper().str.strip()
    df["adj_close"] = safe_num(df["adj_close"])
    return df.dropna(subset=["date","ticker","adj_close"]).sort_values(["ticker","date"])

def load_bench(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"ticker","symbol"); p = ncol(df,"adj_close","close")
    if not (d and t and p): raise ValueError("bench.csv needs date, ticker, adj_close.")
    df = df.rename(columns={d:"date", t:"ticker", p:"adj_close"})
    df["date"] = to_dt(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["adj_close"] = safe_num(df["adj_close"])
    return df.dropna().sort_values(["ticker","date"])


# ----------------------------- resampling & panel -----------------------------

def resample_region(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty: return df
    out = []
    by = "date"
    if freq == "monthly":
        rule = "M"
    else:
        rule = "W-SUN"
    for reg, g in df.groupby("region"):
        g = g.sort_values(by).set_index(by)
        agg = {}
        for c in g.columns:
            if c in ["region"]: continue
            if g[c].dtype.kind in "biufc":
                # sums for flows; means for indices
                if "payout" in c or "loss" in c or "package" in c or "damages" in c:
                    agg[c] = "sum"
                else:
                    agg[c] = "mean"
            else:
                agg[c] = "last"
        gg = g.resample(rule).agg(agg)
        gg["region"] = reg
        out.append(gg.reset_index())
    return pd.concat(out, ignore_index=True).sort_values(["region","date"])

def build_panel(quakes: pd.DataFrame, macro: pd.DataFrame, fiscal: pd.DataFrame,
                insurance: pd.DataFrame, supply: pd.DataFrame, freq: str,
                region_key: str, shock: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Prepare quake shocks (daily) → resample
    Q = quakes.copy()
    Q["shock_mag"] = safe_num(Q.get("magnitude", np.nan))
    Q["shock_damage"] = safe_num(Q.get("damages_usd", np.nan))
    # scale damages to billions for stability
    Q["shock_damage_bn"] = Q["shock_damage"] / 1e9
    # Build a daily region dataframe from quake events
    regions = [region_key]
    # Base date span from data
    dt_min = min([x for x in [Q["date"].min() if not Q.empty else None,
                              macro["date"].min() if not macro.empty else None,
                              fiscal["date"].min() if not fiscal.empty else None,
                              insurance["date"].min() if not insurance.empty else None,
                              supply["date"].min() if not supply.empty else None] if x is not None])
    dt_max = max([x for x in [Q["date"].max() if not Q.empty else None,
                              macro["date"].max() if not macro.empty else None,
                              fiscal["date"].max() if not fiscal.empty else None,
                              insurance["date"].max() if not insurance.empty else None,
                              supply["date"].max() if not supply.empty else None] if x is not None])
    if pd.isna(dt_min) or pd.isna(dt_max):
        raise ValueError("Insufficient dates to build panel. Check inputs.")
    daily = pd.DataFrame({"date": pd.date_range(dt_min, dt_max, freq="D")})
    daily["region"] = region_key
    if not Q.empty:
        # Aggregate per day
        qd = (Q.groupby(["date","region"], as_index=False)
                .agg(shock_mag=("shock_mag","max"),
                     shock_damage_bn=("shock_damage_bn","sum")))
        daily = daily.merge(qd, on=["date","region"], how="left")
    daily["shock_mag"] = daily["shock_mag"].fillna(0.0)
    daily["shock_damage_bn"] = daily["shock_damage_bn"].fillna(0.0)
    # Fiscal & insurance timelines (flows sum)
    if not fiscal.empty:
        f = fiscal.groupby(["date","region"], as_index=False)["package_usd"].sum()
        f["fiscal_usd"] = f["package_usd"]; f.drop(columns=["package_usd"], inplace=True)
        daily = daily.merge(f, on=["date","region"], how="left")
    if not insurance.empty:
        ins = insurance.groupby(["date","region"], as_index=False)["insured_loss_usd"].sum()
        daily = daily.merge(ins, on=["date","region"], how="left")
    for c in ["fiscal_usd","insured_loss_usd"]:
        if c in daily.columns: daily[c] = daily[c].fillna(0.0)
    # Resample everything to freq
    daily["region"] = daily["region"].astype(str)
    if freq == "monthly":
        rule = "M"
    else:
        rule = "W-SUN"
    agg = {"shock_mag":"max",
           "shock_damage_bn":"sum",
           "fiscal_usd":"sum" if "fiscal_usd" in daily.columns else "sum",
           "insured_loss_usd":"sum" if "insured_loss_usd" in daily.columns else "sum"}
    P0 = (daily.set_index("date").groupby("region")
                 .resample(rule).agg(agg).reset_index())
    # Attach macro & supply (already periodic, but resample to align)
    MAC = resample_region(macro, freq) if not macro.empty else pd.DataFrame()
    SUP = resample_region(supply, freq) if not supply.empty else pd.DataFrame()
    # Merge
    PANEL = P0.merge(MAC, on=["date","region"], how="left", suffixes=("","_mac"))
    if not SUP.empty:
        PANEL = PANEL.merge(SUP, on=["date","region"], how="left", suffixes=("","_sup"))
    # Shock pick
    if shock.lower().startswith("mag"):
        PANEL["shock"] = PANEL["shock_mag"].fillna(0.0)
    else:
        PANEL["shock"] = PANEL["shock_damage_bn"].fillna(0.0)
    # Controls (changes / growth where applicable)
    if "ip_index" in PANEL.columns: PANEL["d_ip"] = dlog(PANEL["ip_index"])
    if "retail_idx" in PANEL.columns: PANEL["d_retail"] = dlog(PANEL["retail_idx"])
    if "electricity_demand" in PANEL.columns: PANEL["d_elec"] = dlog(PANEL["electricity_demand"])
    if "cpi_construction" in PANEL.columns: PANEL["d_cpi_const"] = dlog(PANEL["cpi_construction"])
    if "tourism_arrivals" in PANEL.columns: PANEL["d_tour"] = dlog(PANEL["tourism_arrivals"])
    if "building_permits" in PANEL.columns: PANEL["d_permits"] = dlog(PANEL["building_permits"])
    if "housing_starts" in PANEL.columns: PANEL["d_starts"] = dlog(PANEL["housing_starts"])
    # Aid resources (fiscal + insurance)
    PANEL["aid_resources_usd"] = PANEL.get("fiscal_usd", 0.0) + PANEL.get("insured_loss_usd", 0.0)
    PANEL = PANEL.sort_values(["region","date"]).reset_index(drop=True)
    return PANEL, daily


# ----------------------------- equities event study -----------------------------

def pivot_prices(PR: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    px = PR.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = px.pct_change().replace([np.inf,-np.inf], np.nan)
    return px, rets

def abnormal_returns(rets: pd.DataFrame, bench_rets: pd.DataFrame) -> pd.DataFrame:
    if bench_rets.empty:
        return rets
    # Use the first bench column as market
    btk = bench_rets.columns[0]
    abn = rets.sub(bench_rets[btk], axis=0)
    return abn

def car_window(series: pd.Series, anchor_dt: pd.Timestamp, pre: int, post: int) -> Optional[float]:
    if pd.isna(anchor_dt): return np.nan
    idx = series.index
    if anchor_dt not in idx:
        pos = idx.searchsorted(anchor_dt)
        if pos >= len(idx): return np.nan
        anchor_dt = idx[pos]
    start = idx.searchsorted(anchor_dt) - pre
    end   = idx.searchsorted(anchor_dt) + post
    start = max(0, start); end = min(len(idx)-1, end)
    window = series.iloc[start:end+1]
    return float(window.sum()) if not window.empty else np.nan

def event_study_equities(quakes: pd.DataFrame, prices: pd.DataFrame, bench: pd.DataFrame, window_days: int) -> pd.DataFrame:
    if prices.empty or quakes.empty:
        return pd.DataFrame()
    PX, RET = pivot_prices(prices)
    BXP, BRT = pivot_prices(bench) if not bench.empty else (pd.DataFrame(index=PX.index), pd.DataFrame(index=PX.index))
    if not BXP.empty:
        BXP = BXP.reindex(PX.index); BRT = BXP.pct_change()
    ABN = abnormal_returns(RET, BRT) if not BRT.empty else RET
    rows = []
    qdates = sorted(pd.to_datetime(quakes["date"]).dt.normalize().unique())
    for d in qdates:
        for t in ABN.columns:
            car = car_window(ABN[t].dropna(), pd.Timestamp(d), window_days, window_days)
            rows.append({"event_date": pd.Timestamp(d).date(), "ticker": t, "CAR": car})
    return pd.DataFrame(rows).sort_values(["event_date","ticker"])


# ----------------------------- local projections (IRFs) -----------------------------

def local_projections(panel: pd.DataFrame, dep_cols: List[str], H: int, lags: int, min_obs: int) -> pd.DataFrame:
    """
    For each dependent variable (in changes), estimate IRF β_h of shock_t on Δy_{t+h}.
    Controls: L lags of Δy and shock; plus contemporaneous aid_resources (scaled).
    """
    if panel.empty: return pd.DataFrame()
    df = panel.copy()
    df = df.sort_values("date")
    out = []
    if "aid_resources_usd" in df.columns:
        df["aid_scaled"] = df["aid_resources_usd"] / (df["aid_resources_usd"].rolling(12, min_periods=3).mean() + 1e-9)
    else:
        df["aid_scaled"] = 0.0
    for dep in dep_cols:
        if dep not in df.columns: 
            continue
        y = df[dep]
        # require enough data
        if y.notna().sum() < max(min_obs, 12):
            continue
        # Build lag matrices once
        lag_block = []
        lag_names = []
        for L in range(1, lags+1):
            lag_block.append(df["shock"].shift(L))
            lag_names.append(f"shock_l{L}")
            if dep in df.columns:
                lag_block.append(y.shift(L))
                lag_names.append(f"{dep}_l{L}")
        LAGS = pd.concat(lag_block, axis=1) if lag_block else pd.DataFrame(index=df.index)
        if not LAGS.empty: LAGS.columns = lag_names
        for h in range(0, H+1):
            # Left-hand side Δy_{t+h}
            yy = y.shift(-h)  # already Δ if dep is a dlog
            Xparts = [pd.Series(1.0, index=df.index, name="const"),
                      df["shock"].rename("shock_t"),
                      df["aid_scaled"].rename("aid_scaled")]
            if not LAGS.empty: Xparts.append(LAGS)
            X = pd.concat(Xparts, axis=1).dropna()
            Y = yy.loc[X.index]
            if X.shape[0] < max(min_obs, 5*X.shape[1]):
                continue
            beta, resid, XTX_inv = ols_beta_resid(X.values, Y.values.reshape(-1,1))
            se = hac_se(X.values, resid, XTX_inv, L=max(6, lags))
            for i, nm in enumerate(X.columns):
                if nm == "shock_t":
                    out.append({
                        "var": dep,
                        "h": int(h),
                        "beta": float(beta[i,0]),
                        "se": float(se[i]),
                        "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                        "n": int(X.shape[0]),
                        "lags": int(lags)
                    })
    return pd.DataFrame(out).sort_values(["var","h"])


# ----------------------------- rebuild S-curves & materials -----------------------------

def logistic_s_curve(t: np.ndarray, k: float, t0: float, total: float) -> np.ndarray:
    """Cumulative share over time following a logistic S-curve."""
    # normalized 0..1
    s = 1.0 / (1.0 + np.exp(-k * (t - t0)))
    s0 = s.min()
    s1 = s.max()
    s = (s - s0) / (s1 - s0 + 1e-12)
    return total * s

def rebuild_paths(quakes_periodic: pd.DataFrame, panel: pd.DataFrame, freq: str,
                  share_res: float, share_inf: float, share_com: float,
                  speed_res: float, speed_inf: float, speed_com: float,
                  mid_res: int, mid_inf: int, mid_com: int,
                  aid_scale: float, labor_bottleneck: float) -> pd.DataFrame:
    """
    Build post-event spending paths per major event and aggregate to region path.
    Uses damages_bn per period as anchor "total" for each event block; if damages are zero, uses aid_resources.
    """
    # Build a time index from panel
    dates = sorted(panel["date"].unique())
    if not dates: return pd.DataFrame()
    idx = pd.Index(dates)
    period = np.arange(len(idx))
    rows = []
    # Aggregate shocks per period
    dmg = panel.set_index("date")["shock_damage_bn"] if "shock_damage_bn" in panel.columns else panel.set_index("date")["shock"]
    dmg = dmg.reindex(idx).fillna(0.0)
    # Aid baseline
    aid = panel.set_index("date")["aid_resources_usd"].reindex(idx).fillna(0.0)
    # For each period where damage occurs, create an S-curve contribution
    total_spend_usd = []
    res_, inf_, com_ = [], [], []
    for i, dt in enumerate(idx):
        total_bn = float(dmg.iloc[i])
        total_usd = total_bn * 1e9
        if total_usd <= 0 and aid.iloc[i] > 0:
            total_usd = float(aid.iloc[i]) * float(aid_scale)
        if total_usd <= 0:
            # no event this period; zero contribution
            res_.append(0.0); inf_.append(0.0); com_.append(0.0)
            total_spend_usd.append(0.0)
            continue
        # Category totals
        tot_res = total_usd * share_res
        tot_inf = total_usd * share_inf
        tot_com = total_usd * share_com
        # Build cumulative paths starting at i
        t = np.arange(len(idx)-i)
        # apply labor bottleneck as slower speed
        sr = max(1e-6, speed_res * (1.0 - labor_bottleneck))
        si = max(1e-6, speed_inf * (1.0 - labor_bottleneck))
        sc = max(1e-6, speed_com * (1.0 - labor_bottleneck))
        cum_r = logistic_s_curve(t, sr, mid_res, tot_res)
        cum_i = logistic_s_curve(t, si, mid_inf, tot_inf)
        cum_c = logistic_s_curve(t, sc, mid_com, tot_com)
        # Convert to per-period flows and add to timelines
        flow_r = np.diff(np.concatenate([[0.0], cum_r]))
        flow_i = np.diff(np.concatenate([[0.0], cum_i]))
        flow_c = np.diff(np.concatenate([[0.0], cum_c]))
        # add to arrays (with alignment)
        while len(res_) < i: res_.append(0.0); inf_.append(0.0); com_.append(0.0); total_spend_usd.append(0.0)
        for j, val in enumerate(flow_r):
            if i+j >= len(idx): break
            res_[i+j] = res_[i+j] + float(val)
        for j, val in enumerate(flow_i):
            if i+j >= len(idx): break
            inf_[i+j] = inf_[i+j] + float(val)
        for j, val in enumerate(flow_c):
            if i+j >= len(idx): break
            com_[i+j] = com_[i+j] + float(val)
    # Ensure arrays length
    while len(res_) < len(idx): res_.append(0.0); inf_.append(0.0); com_.append(0.0)
    total_spend_usd = (np.array(res_) + np.array(inf_) + np.array(com_)).tolist()
    RB = pd.DataFrame({
        "date": idx,
        "spend_res_usd": res_,
        "spend_inf_usd": inf_,
        "spend_com_usd": com_,
    })
    RB["spend_total_usd"] = RB[["spend_res_usd","spend_inf_usd","spend_com_usd"]].sum(axis=1)
    RB["cum_spend_usd"] = RB["spend_total_usd"].cumsum()
    return RB

def materials_from_spend(rebuild: pd.DataFrame, supply: pd.DataFrame,
                         cement_int: float, steel_int: float, lumber_int: float,
                         import_relax: float, freq: str) -> pd.DataFrame:
    if rebuild.empty: return pd.DataFrame()
    out = rebuild.copy()
    k = 1000.0  # $1k unit
    out["cement_tons"] = (out["spend_total_usd"] / k) * cement_int
    out["steel_tons"]  = (out["spend_total_usd"] / k) * steel_int
    out["lumber_m3"]   = (out["spend_total_usd"] / k) * lumber_int
    # Capacity
    CAP = resample_region(supply, freq) if not supply.empty else pd.DataFrame()
    if not CAP.empty:
        cap = CAP[["date","cement_capacity_tons","steel_capacity_tons"]].copy()
        out = out.merge(cap, on="date", how="left")
        # import relaxation adds fraction of capacity as extra slack
        out["cement_available"] = out["cement_capacity_tons"] * (1.0 + float(import_relax))
        out["steel_available"]  = out["steel_capacity_tons"]  * (1.0 + float(import_relax))
        # Gaps
        out["cement_gap_tons"] = out["cement_tons"] - out["cement_available"]
        out["steel_gap_tons"]  = out["steel_tons"]  - out["steel_available"]
        # price-pressure proxy as demand/available
        out["cement_pressure"] = out["cement_tons"] / (out["cement_available"] + 1e-9)
        out["steel_pressure"]  = out["steel_tons"]  / (out["steel_available"] + 1e-9)
    return out


# ----------------------------- scenarios -----------------------------

def scenario_run(panel: pd.DataFrame, rebuild: pd.DataFrame, supply: pd.DataFrame, freq: str,
                 speed_mult: float, aid_scale: float, import_relax: float, labor_bottleneck: float,
                 cement_int: float, steel_int: float, lumber_int: float,
                 share_res: float, share_inf: float, share_com: float,
                 speed_res: float, speed_inf: float, speed_com: float,
                 mid_res: int, mid_inf: int, mid_com: int) -> pd.DataFrame:
    # Recompute rebuild under scenario tweaks
    RB = rebuild_paths(
        quakes_periodic=pd.DataFrame(), panel=panel, freq=freq,
        share_res=share_res, share_inf=share_inf, share_com=share_com,
        speed_res=speed_res*speed_mult, speed_inf=speed_inf*speed_mult, speed_com=speed_com*speed_mult,
        mid_res=mid_res, mid_inf=mid_inf, mid_com=mid_com,
        aid_scale=aid_scale, labor_bottleneck=labor_bottleneck
    )
    MAT = materials_from_spend(RB, supply, cement_int, steel_int, lumber_int, import_relax, freq)
    MAT["scenario_speed_mult"] = speed_mult
    MAT["scenario_aid_scale"] = aid_scale
    MAT["scenario_import_relax"] = import_relax
    MAT["scenario_labor_bottleneck"] = labor_bottleneck
    return MAT


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    quakes: str
    macro: Optional[str]
    fiscal: Optional[str]
    insurance: Optional[str]
    prices: Optional[str]
    bench: Optional[str]
    supply: Optional[str]
    permits: Optional[str]
    region: str
    freq: str
    shock: str
    event_window: int
    lp_h: int
    lags: int
    speed_res: float
    speed_inf: float
    speed_com: float
    mid_res: int
    mid_inf: int
    mid_com: int
    share_res: float
    share_inf: float
    share_com: float
    aid_scale: float
    import_relax: float
    labor_bottleneck: float
    cement_int: float
    steel_int: float
    lumber_int: float
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Earthquake → rebuild: panel, event study, IRFs, materials & scenarios")
    ap.add_argument("--quakes", required=True)
    ap.add_argument("--macro", default="")
    ap.add_argument("--fiscal", default="")
    ap.add_argument("--insurance", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--supply", default="")
    ap.add_argument("--permits", default="")
    ap.add_argument("--region", required=True)
    ap.add_argument("--freq", default="monthly", choices=["monthly","weekly"])
    ap.add_argument("--shock", default="damage", choices=["damage","magnitude"])
    ap.add_argument("--event_window", type=int, default=30)
    ap.add_argument("--lp_h", type=int, default=18)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--speed_res", type=float, default=0.20)
    ap.add_argument("--speed_inf", type=float, default=0.12)
    ap.add_argument("--speed_com", type=float, default=0.16)
    ap.add_argument("--mid_res", type=int, default=6)
    ap.add_argument("--mid_inf", type=int, default=10)
    ap.add_argument("--mid_com", type=int, default=8)
    ap.add_argument("--share_res", type=float, default=0.45)
    ap.add_argument("--share_inf", type=float, default=0.40)
    ap.add_argument("--share_com", type=float, default=0.15)
    ap.add_argument("--aid_scale", type=float, default=1.0)
    ap.add_argument("--import_relax", type=float, default=0.0)
    ap.add_argument("--labor_bottleneck", type=float, default=0.0)
    ap.add_argument("--cement_int", type=float, default=0.11)
    ap.add_argument("--steel_int", type=float, default=0.08)
    ap.add_argument("--lumber_int", type=float, default=0.15)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_quake")
    ap.add_argument("--min_obs", type=int, default=60)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load
    Q = load_quakes(args.quakes, region_key=args.region)
    M = load_macro(args.macro) if args.macro else pd.DataFrame()
    F = load_fiscal(args.fiscal) if args.fiscal else pd.DataFrame()
    I = load_insurance(args.insurance) if args.insurance else pd.DataFrame()
    S = load_supply(args.supply) if args.supply else pd.DataFrame()
    PR = load_prices(args.prices) if args.prices else pd.DataFrame()
    BE = load_bench(args.bench) if args.bench else pd.DataFrame()
    PER = load_permits(args.permits) if args.permits else pd.DataFrame()

    # Merge permits into macro if provided (and missing)
    if not PER.empty:
        if M.empty:
            M = PER.copy()
        else:
            M = M.merge(PER, on=["date","region"], how="outer")

    # Time filters
    if args.start:
        s = to_dt(pd.Series([args.start])).iloc[0]
        for df in [Q,M,F,I,S,PR,BE]:
            if not df.empty:
                df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = to_dt(pd.Series([args.end])).iloc[0]
        for df in [Q,M,F,I,S,PR,BE]:
            if not df.empty:
                df.drop(df[df["date"] > e].index, inplace=True)

    # Cleaned quake log out
    Q.to_csv(outdir / "cleaned_quakes.csv", index=False)

    # Panel
    PANEL, DAILY = build_panel(Q, M, F, I, S, freq=args.freq, region_key=args.region, shock=args.shock)
    if PANEL.empty:
        raise ValueError("Panel is empty after alignment. Check inputs and region filter.")
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Event study (equities)
    ES = event_study_equities(Q, PR, BE, window_days=int(args.event_window)) if not PR.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study_equities.csv", index=False)

    # IRFs via local projections
    dep_cols = [c for c in ["d_ip","d_retail","d_elec","d_cpi_const","d_tour","d_permits","d_starts"] if c in PANEL.columns]
    IRF = local_projections(PANEL, dep_cols=dep_cols, H=int(args.lp_h), lags=int(args.lags), min_obs=int(args.min_obs))
    if not IRF.empty: IRF.to_csv(outdir / "irf_local_projections.csv", index=False)

    # Rebuild paths (base)
    RB = rebuild_paths(
        quakes_periodic=PANEL[["date","shock_damage_bn"]] if "shock_damage_bn" in PANEL.columns else pd.DataFrame(),
        panel=PANEL, freq=args.freq,
        share_res=float(args.share_res), share_inf=float(args.share_inf), share_com=float(args.share_com),
        speed_res=float(args.speed_res), speed_inf=float(args.speed_inf), speed_com=float(args.speed_com),
        mid_res=int(args.mid_res), mid_inf=int(args.mid_inf), mid_com=int(args.mid_com),
        aid_scale=float(args.aid_scale), labor_bottleneck=float(args.labor_bottleneck)
    )
    if not RB.empty: RB.to_csv(outdir / "rebuild_path.csv", index=False)

    # Materials demand & gaps
    MAT = materials_from_spend(
        RB, S, cement_int=float(args.cement_int), steel_int=float(args.steel_int),
        lumber_int=float(args.lumber_int), import_relax=float(args.import_relax), freq=args.freq
    )
    if not MAT.empty: MAT.to_csv(outdir / "materials_demand.csv", index=False)

    # One illustrative scenario (if any tweak vs base provided)
    SCN = pd.DataFrame()
    # run a scenario only if any of these deviates notably from base defaults (heuristic):
    if (args.import_relax != 0.0) or (args.aid_scale != 1.0) or (args.labor_bottleneck != 0.0):
        SCN = scenario_run(
            PANEL, RB, S, args.freq,
            speed_mult=1.0,
            aid_scale=float(args.aid_scale),
            import_relax=float(args.import_relax),
            labor_bottleneck=float(args.labor_bottleneck),
            cement_int=float(args.cement_int), steel_int=float(args.steel_int), lumber_int=float(args.lumber_int),
            share_res=float(args.share_res), share_inf=float(args.share_inf), share_com=float(args.share_com),
            speed_res=float(args.speed_res), speed_inf=float(args.speed_inf), speed_com=float(args.speed_com),
            mid_res=int(args.mid_res), mid_inf=int(args.mid_inf), mid_com=int(args.mid_com)
        )
        if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    key = {
        "region": args.region,
        "freq": args.freq,
        "sample_start": str(PANEL["date"].min().date()),
        "sample_end": str(PANEL["date"].max().date()),
        "n_periods": int(PANEL.shape[0]),
        "has_event_study": not ES.empty,
        "has_irf": not IRF.empty,
        "has_rebuild": not RB.empty,
        "has_materials": not MAT.empty,
        "has_scenarios": not SCN.empty
    }
    summary = {
        "key": key,
        "files": {
            "cleaned_quakes": "cleaned_quakes.csv",
            "panel": "panel.csv",
            "event_study_equities": "event_study_equities.csv" if not ES.empty else None,
            "irf_local_projections": "irf_local_projections.csv" if not IRF.empty else None,
            "rebuild_path": "rebuild_path.csv" if not RB.empty else None,
            "materials_demand": "materials_demand.csv" if not MAT.empty else None,
            "scenarios": "scenarios.csv" if not SCN.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        quakes=args.quakes, macro=(args.macro or None), fiscal=(args.fiscal or None),
        insurance=(args.insurance or None), prices=(args.prices or None), bench=(args.bench or None),
        supply=(args.supply or None), permits=(args.permits or None),
        region=args.region, freq=args.freq, shock=args.shock, event_window=int(args.event_window),
        lp_h=int(args.lp_h), lags=int(args.lags),
        speed_res=float(args.speed_res), speed_inf=float(args.speed_inf), speed_com=float(args.speed_com),
        mid_res=int(args.mid_res), mid_inf=int(args.mid_inf), mid_com=int(args.mid_com),
        share_res=float(args.share_res), share_inf=float(args.share_inf), share_com=float(args.share_com),
        aid_scale=float(args.aid_scale), import_relax=float(args.import_relax),
        labor_bottleneck=float(args.labor_bottleneck),
        cement_int=float(args.cement_int), steel_int=float(args.steel_int), lumber_int=float(args.lumber_int),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Earthquake → Rebuild Toolkit ==")
    print(f"Region: {key['region']} | Freq: {key['freq']} | Sample: {key['sample_start']} → {key['sample_end']}")
    if key["has_event_study"]:
        print(f"Equity event study written (±{args.event_window} days).")
    if key["has_irf"]:
        print(f"IRFs estimated (H={args.lp_h}, lags={args.lags}).")
    if key["has_rebuild"]:
        print("Rebuild path generated (see rebuild_path.csv).")
    if key["has_materials"]:
        print("Materials demand & capacity gaps computed (see materials_demand.csv).")
    if key["has_scenarios"]:
        print("Scenario results written (see scenarios.csv).")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
