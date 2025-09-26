#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
food_security_imports.py — Imports dependence, price pass-through, risk index & stress scenarios
------------------------------------------------------------------------------------------------

What this does
==============
A research toolkit for **food security** focused on **imported staples**. It builds a
country–commodity monthly panel and then:

1) Cleans & aligns inputs
   • Trade: import volume/value by commodity & origin  
   • Prices: global benchmarks (USD/ton) and local wholesale/retail prices  
   • FX: USD/local; Shipping/freight; Policies (tariffs, quotas, stock releases, export bans elsewhere)  
   • Stocks/reserves; Weather/climate signals (rainfall anomaly, drought index, ENSO)  
   • Macro: population, income, food CPI; Production (optional) & consumption weights (optional)

2) Elasticities (panel regressions, HAC/Newey–West SEs)
   • dlog(import_volume) ~ dlog(global_price) + dlog(FX) + dlog(shipping) + income + policy + weather + stocks + seasonality  
   • Pass-through: dlog(local_price) ~ dlog(global_price*FX + shipping) + policy + stocks

3) Risk index (per commodity, rolling & latest)
   • Import dependence (needs production; else proxy)  
   • Concentration (HHI by origin)  
   • Driver volatility (price/FX/shipping)  
   • Stock cover (months) & climate pressure  
   → Composite score ∈ [0,100]

4) Scenarios & projections (H months)
   • Shocks to global price (%), FX (%), shipping (%), production (%), policy (pp), stocks release (tons)  
   • Uses estimated elasticities (fallback priors if sparse) → volumes, import bill (USD), local price & rough food CPI impact (if weights provided)

5) Stress: Monte Carlo VaR/ES of import bill
   • Historical covariances of {global price, FX, shipping}; simulate N paths → import bill distribution

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--trade trade.csv                REQUIRED (monthly or higher freq)
  Columns (any subset; case-insensitive):
    date, country, commodity[, hs_code], origin_country[, origin],
    volume_tons[, qty_tons], value_usd[, cif_usd, import_usd], unit_value_usd

--gprice global_prices.csv       OPTIONAL (benchmarks)
  Columns: date, commodity, global_price_usd_per_ton[, price_usd]

--lprice local_prices.csv        OPTIONAL (wholesale/retail)
  Columns: date, country, commodity, local_price_per_ton[, local_price]

--fx fx.csv                      OPTIONAL
  Columns: date, country, usd_local[, fx_usd_local]

--ship shipping.csv              OPTIONAL
  Columns: date, route[, country], commodity[, scope], freight_usd_per_ton[, freight_idx]

--policy policy.csv              OPTIONAL
  Columns: date, country, type[, measure], label[, event], value[, delta_pp]
  Examples for type: TARIFF_UP, TARIFF_DOWN, QUOTA, STOCK_RELEASE, SUBSIDY

--stocks stocks.csv              OPTIONAL
  Columns: date, country, commodity, stocks_tons[, public_stocks_tons]

--weather weather.csv            OPTIONAL
  Columns: date, country, commodity[, crop], rainfall_anom[, rain_anom], drought_idx[, spei], enso_idx

--macro macro.csv                OPTIONAL
  Columns: date, country, pop[, population], income_pc[, income], food_cpi[, cpi_food]

--production production.csv      OPTIONAL (for import dependence)
  Columns: date, country, commodity, production_tons

--weights weights.csv            OPTIONAL (for CPI pass-through aggregation)
  Columns: country, commodity, food_weight  # share in food CPI basket ∈ [0,1]

CLI (key)
---------
--country "IN"                   Country ISO/name (case-insensitive substring match)
--commodities "WHEAT,RICE"       Comma list or "ALL"
--freq monthly                   Resample frequency (monthly only supported)
--lags 3                         Distributed lags for elasticities
--horizon 12                     Scenario horizon in months
--n_sims 10000                   Monte Carlo paths for VaR/ES
--fx_shock_pct 0                 Scenario FX (% chg in USD/local, + = USD stronger)
--price_shock_pct 0              Scenario global price (%)
--ship_shock_pct 0               Scenario shipping/freight (%)
--prod_shock_pct 0               Scenario domestic production (%)
--policy_shift_pp 0              Scenario policy index shift (pp)
--stocks_release_tons 0          Scenario one-off stock release at t+1
--risk_weights "0.3,0.2,0.2,0.2,0.1"  Weights for (dependence, HHI, vol, stock_cover, climate)
--start / --end                  Sample filters (YYYY-MM-DD; month-end aligned)
--outdir out_foodsec             Output directory
--min_obs 48                     Minimum obs per commodity for regressions

Outputs
-------
- panel.csv                      Country–commodity monthly panel
- elasticity_imports.csv         Volume elasticities (+ HAC SEs)
- elasticity_passthrough.csv     Local price pass-through (+ HAC SEs)
- risk_index.csv                 Components & composite (rolling and latest)
- scenarios.csv                  Forward projections under shocks (by commodity)
- stress_vares.csv               Monte Carlo VaR/ES for import bill
- event_study.csv                CARs around large policy or ENSO shocks (if present)
- summary.json                   Headline metrics
- config.json                    Echo of run configuration

Notes & caveats
---------------
• All USD series should be nominal USD; FX is USD/local.  
• Shipping can be commodity-specific or an index; the code will fallback gracefully.  
• If production is missing, import dependence is proxied using imports share of (imports+stocks draw).  
• Research tooling; validate with local knowledge before decisions.

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

def eom(d: pd.Series) -> pd.Series:
    dt = to_dt(d)
    return (dt + pd.offsets.MonthEnd(0))

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = safe_num(s).replace(0, np.nan)
    return np.log(s) - np.log(s.shift(1))

def winsor(s: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def hhi(shares: pd.Series) -> float:
    x = safe_num(shares.fillna(0.0)).clip(0,1).values
    return float(np.sum((x*100)**2) / 10000.0)  # 0..1 scaled

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

def load_trade(path: str, country_key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); com = ncol(df,"commodity")
    vol = ncol(df,"volume_tons","qty_tons","volume","qty")
    val = ncol(df,"value_usd","cif_usd","import_usd","value")
    uvu = ncol(df,"unit_value_usd","unit_value","usd_per_ton")
    org = ncol(df,"origin_country","origin")
    if not (d and c and com):
        raise ValueError("trade.csv must include date, country, commodity (and ideally volume/value).")
    df = df.rename(columns={d:"date", c:"country", com:"commodity"})
    if vol: df = df.rename(columns={vol:"volume_tons"})
    if val: df = df.rename(columns={val:"value_usd"})
    if uvu: df = df.rename(columns={uvu:"unit_value_usd"})
    if org: df = df.rename(columns={org:"origin_country"})
    df["date"] = eom(df["date"])
    df["country"] = df["country"].astype(str)
    # filter country (substring)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    df["commodity"] = df["commodity"].astype(str).str.upper().str.strip()
    for k in ["volume_tons","value_usd","unit_value_usd"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # derive unit value if missing
    if "unit_value_usd" not in df.columns and {"value_usd","volume_tons"}.issubset(df.columns):
        df["unit_value_usd"] = df["value_usd"] / df["volume_tons"].replace(0,np.nan)
    return df

def load_gprice(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); com = ncol(df,"commodity"); p = ncol(df,"global_price_usd_per_ton","price_usd","usd_per_ton")
    if not (d and com and p): raise ValueError("global_prices.csv needs date, commodity, price.")
    df = df.rename(columns={d:"date", com:"commodity", p:"gprice_usd"})
    df["date"] = eom(df["date"]); df["commodity"] = df["commodity"].astype(str).str.UPPER().str.strip()
    df["gprice_usd"] = safe_num(df["gprice_usd"])
    return df

def load_lprice(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); com = ncol(df,"commodity"); p = ncol(df,"local_price_per_ton","local_price","price_local")
    if not (d and c and com and p): raise ValueError("local_prices.csv needs date, country, commodity, local_price.")
    df = df.rename(columns={d:"date", c:"country", com:"commodity", p:"lprice_local"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    df["commodity"] = df["commodity"].astype(str).str.upper().str.strip()
    df["lprice_local"] = safe_num(df["lprice_local"])
    return df

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); fx = ncol(df,"usd_local","fx_usd_local","usd_per_local")
    if not (d and c and fx): raise ValueError("fx.csv needs date, country, usd_local.")
    df = df.rename(columns={d:"date", c:"country", fx:"usd_local"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    df["usd_local"] = safe_num(df["usd_local"])
    return df

def load_shipping(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); r = ncol(df,"route","country"); com = ncol(df,"commodity","scope"); f = ncol(df,"freight_usd_per_ton","freight_idx","cost_usd")
    if not d: raise ValueError("shipping.csv needs date.")
    df = df.rename(columns={d:"date"})
    if r: df = df.rename(columns={r:"route"})
    if com: df = df.rename(columns={com:"commodity"})
    if f: df = df.rename(columns={f:"freight"})
    df["date"] = eom(df["date"])
    if "commodity" in df.columns: df["commodity"] = df["commodity"].astype(str).str.upper().str.strip()
    df["freight"] = safe_num(df.get("freight", np.nan))
    return df

def load_policy(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); t = ncol(df,"type","measure"); v = ncol(df,"value","delta_pp")
    if not (d and c and (t or v)): raise ValueError("policy.csv needs date, country and type/value.")
    df = df.rename(columns={d:"date", c:"country"})
    if t: df = df.rename(columns={t:"type"})
    if v: df = df.rename(columns={v:"value"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    if "type" in df.columns: df["type"] = df["type"].astype(str).str.upper().str.strip()
    df["value"] = safe_num(df.get("value", np.nan))
    return df

def load_stocks(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); com = ncol(df,"commodity"); s = ncol(df,"stocks_tons","public_stocks_tons","stocks")
    if not (d and c and com and s): raise ValueError("stocks.csv needs date,country,commodity,stocks.")
    df = df.rename(columns={d:"date", c:"country", com:"commodity", s:"stocks_tons"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    df["commodity"] = df["commodity"].astype(str).str.upper().str.strip()
    df["stocks_tons"] = safe_num(df["stocks_tons"])
    return df

def load_weather(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); com = ncol(df,"commodity","crop")
    r = ncol(df,"rainfall_anom","rain_anom"); dr = ncol(df,"drought_idx","spei"); en = ncol(df,"enso_idx")
    if not d: raise ValueError("weather.csv needs date.")
    df = df.rename(columns={d:"date"})
    if c: df = df.rename(columns={c:"country"})
    if com: df = df.rename(columns={com:"commodity"})
    if r: df = df.rename(columns={r:"rain_anom"})
    if dr: df = df.rename(columns={dr:"drought_idx"})
    if en: df = df.rename(columns={en:"enso_idx"})
    df["date"] = eom(df["date"])
    if "country" in df.columns: df["country"] = df["country"].astype(str)
    if "commodity" in df.columns: df["commodity"] = df["commodity"].astype(str).str.upper().str.strip()
    for k in ["rain_anom","drought_idx","enso_idx"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    pop = ncol(df,"pop","population"); inc = ncol(df,"income_pc","income"); f = ncol(df,"food_cpi","cpi_food")
    if not (d and c): raise ValueError("macro.csv needs date, country.")
    df = df.rename(columns={d:"date", c:"country"})
    if pop: df = df.rename(columns={pop:"population"})
    if inc: df = df.rename(columns={inc:"income_pc"})
    if f: df = df.rename(columns={f:"food_cpi"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    for k in ["population","income_pc","food_cpi"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_production(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); com = ncol(df,"commodity"); q = ncol(df,"production_tons","output_tons","production")
    if not (d and c and com and q): raise ValueError("production.csv needs date,country,commodity,production_tons.")
    df = df.rename(columns={d:"date", c:"country", com:"commodity", q:"production_tons"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    df["commodity"] = df["commodity"].astype(str).str.upper().str.strip()
    df["production_tons"] = safe_num(df["production_tons"])
    return df

def load_weights(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    c = ncol(df,"country"); com = ncol(df,"commodity"); w = ncol(df,"food_weight","weight")
    if not (c and com and w): raise ValueError("weights.csv needs country, commodity, food_weight.")
    df = df.rename(columns={c:"country", com:"commodity", w:"food_weight"})
    df["country"] = df["country"].astype(str)
    df["commodity"] = df["commodity"].astype(str).str.upper().str.strip()
    df["food_weight"] = safe_num(df["food_weight"]).clip(0,1)
    return df


# ----------------------------- panel construction -----------------------------

def build_panel(TR: pd.DataFrame, GP: pd.DataFrame, LP: pd.DataFrame, FX: pd.DataFrame,
                SH: pd.DataFrame, POL: pd.DataFrame, ST: pd.DataFrame, WEA: pd.DataFrame,
                MAC: pd.DataFrame, PROD: pd.DataFrame,
                country_key: str, commodities: Optional[List[str]]) -> pd.DataFrame:
    df = TR.copy()
    if commodities and "ALL" not in [x.upper() for x in commodities]:
        df = df[df["commodity"].isin([x.upper().strip() for x in commodities])]
    # aggregate trade to month-country-commodity
    agg = (df.groupby(["date","country","commodity","origin_country"], dropna=False)
             .agg(volume_tons=("volume_tons","sum") if "volume_tons" in df.columns else ("value_usd","size"),
                  value_usd=("value_usd","sum") if "value_usd" in df.columns else ("volume_tons","size"))
             .reset_index())
    # by commodity overall
    tot = (agg.groupby(["date","country","commodity"], as_index=False)
              .agg(volume_tons=("volume_tons","sum"), value_usd=("value_usd","sum")))
    tot["unit_value_usd"] = tot["value_usd"] / tot["volume_tons"].replace(0,np.nan)
    P = tot.copy()
    # origin concentration (HHI)
    shares = agg.merge(tot[["date","country","commodity","value_usd"]].rename(columns={"value_usd":"tot_val"}),
                       on=["date","country","commodity"], how="left")
    shares["share"] = shares["value_usd"] / shares["tot_val"].replace(0,np.nan)
    H = (shares.groupby(["date","country","commodity"])["share"].apply(hhi)
               .reset_index().rename(columns={"share":"origin_hhi"}))
    P = P.merge(H, on=["date","country","commodity"], how="left")
    # merge global price
    if not GP.empty:
        gp = GP[["date","commodity","gprice_usd"]].copy()
        P = P.merge(gp, on=["date","commodity"], how="left")
    # merge local price
    if not LP.empty:
        lp = LP[["date","country","commodity","lprice_local"]].copy()
        P = P.merge(lp, on=["date","country","commodity"], how="left")
    # FX
    if not FX.empty:
        fx = FX[["date","country","usd_local"]].copy()
        P = P.merge(fx, on=["date","country"], how="left")
    # Shipping
    if not SH.empty:
        sh = SH.copy()
        # prefer commodity-specific; else use average per date
        if "commodity" in sh.columns:
            shc = (sh.groupby(["date","commodity"], as_index=False)["freight"].mean())
            P = P.merge(shc, on=["date","commodity"], how="left")
        else:
            sha = (sh.groupby(["date"], as_index=False)["freight"].mean())
            P = P.merge(sha, on=["date"], how="left")
        P = P.rename(columns={"freight":"freight_usd"})
    # Stocks
    if not ST.empty:
        st = ST[["date","country","commodity","stocks_tons"]].copy()
        P = P.merge(st, on=["date","country","commodity"], how="left")
    # Weather
    if not WEA.empty:
        we = WEA.copy()
        # allow country-level / commodity-level matches
        if {"country","commodity"}.issubset(we.columns):
            wec = we.groupby(["date","country","commodity"], as_index=False).agg(
                rain_anom=("rain_anom","mean") if "rain_anom" in we.columns else ("date","size"),
                drought_idx=("drought_idx","mean") if "drought_idx" in we.columns else ("date","size"),
                enso_idx=("enso_idx","mean") if "enso_idx" in we.columns else ("date","size"),
            )
            P = P.merge(wec, on=["date","country","commodity"], how="left")
        else:
            wec = we.groupby(["date"], as_index=False).agg(
                rain_anom=("rain_anom","mean") if "rain_anom" in we.columns else ("date","size"),
                drought_idx=("drought_idx","mean") if "drought_idx" in we.columns else ("date","size"),
                enso_idx=("enso_idx","mean") if "enso_idx" in we.columns else ("date","size"),
            )
            P = P.merge(wec, on=["date"], how="left")
    # Macro
    if not MAC.empty:
        mc = MAC[["date","country"] + [c for c in ["population","income_pc","food_cpi"] if c in MAC.columns]]
        P = P.merge(mc, on=["date","country"], how="left")
    # Production
    if not PROD.empty:
        pr = PROD[["date","country","commodity","production_tons"]].copy()
        P = P.merge(pr, on=["date","country","commodity"], how="left")
    # Derived vars
    P = P.sort_values(["commodity","date"])
    P["dlog_volume"] = P.groupby(["commodity"])["volume_tons"].apply(dlog).reset_index(level=0, drop=True)
    if "gprice_usd" in P.columns: P["dlog_gprice"] = P.groupby(["commodity"])["gprice_usd"].apply(dlog).reset_index(level=0, drop=True)
    if "usd_local" in P.columns:  P["dlog_fx"]     = P.groupby(["commodity"])["usd_local"].apply(dlog).reset_index(level=0, drop=True)
    if "freight_usd" in P.columns:P["dlog_ship"]   = P.groupby(["commodity"])["freight_usd"].apply(dlog).reset_index(level=0, drop=True)
    if "lprice_local" in P.columns: P["dlog_lprice"] = P.groupby(["commodity"])["lprice_local"].apply(dlog).reset_index(level=0, drop=True)
    # Policy index (simple net sign by month)
    if not POL.empty:
        pol = POL.copy()
        pol["sign"] = 0.0
        pol.loc[pol["type"].str.contains("TARIFF_UP|BAN|QUOTA", na=False), "sign"] = -1.0
        pol.loc[pol["type"].str.contains("TARIFF_DOWN|SUBSIDY|STOCK_RELEASE", na=False), "sign"] = +1.0
        polm = pol.groupby(["date","country"], as_index=False)["sign"].sum().rename(columns={"sign":"policy_index"})
        P = P.merge(polm, on=["date","country"], how="left")
    # Import bill
    P["import_bill_usd"] = P["value_usd"]
    # Import dependence proxy
    if "production_tons" in P.columns:
        P["apparent_supply_tons"] = P["production_tons"] + P["volume_tons"]
        P["import_dependence"] = (P["volume_tons"] / P["apparent_supply_tons"].replace(0,np.nan)).clip(0,1)
    else:
        P["import_dependence"] = np.nan
    # Seasonality
    P["month"] = P["date"].dt.month
    return P


# ----------------------------- elasticities -----------------------------

def build_dlag_matrix(g: pd.DataFrame, L: int, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.DataFrame(index=g.index)
    names = []
    for c in cols:
        if c not in g.columns: continue
        for l in range(0, L+1):
            nm = f"{c}_l{l}"
            X[nm] = g[c].shift(l)
            names.append(nm)
    return X, names

def regress_elasticities(P: pd.DataFrame, L: int, min_obs: int) -> pd.DataFrame:
    if P.empty: return pd.DataFrame()
    rows = []
    for com, g in P.groupby("commodity"):
        g = g.sort_values("date").copy()
        y = g["dlog_volume"]
        if y.notna().sum() < max(min_obs, 24): continue
        X1, names1 = build_dlag_matrix(g, L, cols=[c for c in ["dlog_gprice","dlog_fx","dlog_ship"] if c in g.columns])
        add = []
        if "income_pc" in g.columns: add.append("income_pc")
        if "policy_index" in g.columns: add.append("policy_index")
        if "rain_anom" in g.columns: add.append("rain_anom")
        if "drought_idx" in g.columns: add.append("drought_idx")
        if "stocks_tons" in g.columns: add.append("stocks_tons")
        X2 = g[add] if add else pd.DataFrame(index=g.index)
        D = pd.get_dummies(g["month"].astype(int), prefix="m", drop_first=True)
        X = pd.concat([pd.Series(1.0, index=g.index, name="const"), X1, X2, D], axis=1).dropna()
        Y = y.loc[X.index]
        if X.shape[0] < max(min_obs, 5*X.shape[1]//3): continue
        beta, resid, XTX_inv = ols_beta_resid(X.values, Y.values.reshape(-1,1))
        se = hac_se(X.values, resid, XTX_inv, L=max(6, L))
        for i, nm in enumerate(X.columns):
            rows.append({"commodity": com, "var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                         "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "n": int(X.shape[0]), "lags": int(L)})
        # cumulative elasticities for key drivers
        for base in ["dlog_gprice","dlog_fx","dlog_ship"]:
            idxs = [i for i, nm in enumerate(X.columns) if nm.startswith(base+"_l")]
            if idxs:
                bsum = float(np.sum([beta[i,0] for i in idxs]))
                sesq = float(np.sum([se[i]**2 for i in idxs]))
                rows.append({"commodity": com, "var": base+"_cum_0..L", "coef": bsum, "se": np.sqrt(sesq),
                             "t_stat": bsum/np.sqrt(sesq) if sesq>0 else np.nan, "n": int(X.shape[0]), "lags": int(L)})
    return pd.DataFrame(rows)

def pass_through(P: pd.DataFrame, L: int, min_obs: int) -> pd.DataFrame:
    if P.empty or "dlog_lprice" not in P.columns: return pd.DataFrame()
    rows = []
    for com, g in P.groupby("commodity"):
        g = g.sort_values("date")
        y = g["dlog_lprice"]
        if y.notna().sum() < max(min_obs, 24): continue
        # composite import cost driver: log(global_price*FX + shipping add-on approx)
        # use dlog(gprice) + dlog(fx) + (shipping in levels scaled)
        cols = []
        if "dlog_gprice" in g.columns: cols.append("dlog_gprice")
        if "dlog_fx" in g.columns: cols.append("dlog_fx")
        X1, _ = build_dlag_matrix(g, L, cols=cols)
        add = []
        if "dlog_ship" in g.columns: add.append("dlog_ship")  # treat as pass-through component
        if "policy_index" in g.columns: add.append("policy_index")
        if "stocks_tons" in g.columns: add.append("stocks_tons")
        X2 = g[add] if add else pd.DataFrame(index=g.index)
        D = pd.get_dummies(g["month"].astype(int), prefix="m", drop_first=True)
        X = pd.concat([pd.Series(1.0, index=g.index, name="const"), X1, X2, D], axis=1).dropna()
        Y = y.loc[X.index]
        if X.shape[0] < max(min_obs, 5*X.shape[1]//3): continue
        beta, resid, XTX_inv = ols_beta_resid(X.values, Y.values.reshape(-1,1))
        se = hac_se(X.values, resid, XTX_inv, L=max(6, L))
        for i, nm in enumerate(X.columns):
            rows.append({"commodity": com, "var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                         "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "n": int(X.shape[0]), "lags": int(L)})
        # cumulative pass-through of (gprice+fx)
        idxs = [i for i, nm in enumerate(X.columns) if nm.startswith("dlog_gprice_l") or nm.startswith("dlog_fx_l")]
        if idxs:
            bsum = float(np.sum([beta[i,0] for i in idxs]))
            sesq = float(np.sum([se[i]**2 for i in idxs]))
            rows.append({"commodity": com, "var": "pass_through_cum_0..L", "coef": bsum, "se": np.sqrt(sesq),
                         "t_stat": bsum/np.sqrt(sesq) if sesq>0 else np.nan, "n": int(X.shape[0]), "lags": int(L)})
    return pd.DataFrame(rows)


# ----------------------------- risk index -----------------------------

def rolling_vol(x: pd.Series, w: int=12) -> pd.Series:
    return x.rolling(w, min_periods=max(4, w//2)).std(ddof=0)

def risk_components(P: pd.DataFrame, PROD: pd.DataFrame) -> pd.DataFrame:
    if P.empty: return pd.DataFrame()
    out = P[["date","country","commodity","import_bill_usd","volume_tons","origin_hhi","import_dependence",
             "gprice_usd","usd_local","freight_usd","stocks_tons","rain_anom","drought_idx"]].copy()
    # vols
    for c, nm in [("gprice_usd","vol_gprice"), ("usd_local","vol_fx"), ("freight_usd","vol_ship")]:
        if c in out.columns:
            out[nm] = rolling_vol(dlog(out[c]), w=12)
    # stock cover months ~ stocks / (avg monthly imports last year)
    if "stocks_tons" in out.columns:
        flow = out.groupby("commodity")["volume_tons"].transform(lambda s: s.rolling(12, min_periods=6).mean())
        out["stock_cover_months"] = out["stocks_tons"] / (flow.replace(0,np.nan))
    # climate pressure proxy
    if "drought_idx" in out.columns:
        out["climate_pressure"] = out["drought_idx"].clip(lower=0)  # only positive drought stress
    elif "rain_anom" in out.columns:
        out["climate_pressure"] = (-out["rain_anom"]).clip(lower=0)
    # normalize components to 0..1 within commodity
    comps = []
    for k in ["import_dependence","origin_hhi","vol_gprice","vol_fx","vol_ship","climate_pressure"]:
        if k in out.columns:
            z = out.groupby("commodity")[k].transform(lambda s: (s - s.min())/(s.max()-s.min()+1e-12))
            out[k+"_n"] = z.fillna(0.0); comps.append(k+"_n")
    if "stock_cover_months" in out.columns:
        z = out.groupby("commodity")["stock_cover_months"].transform(lambda s: (s - s.min())/(s.max()-s.min()+1e-12))
        out["stock_cover_n"] = 1.0 - z.fillna(0.0)  # lower cover → higher risk
        comps.append("stock_cover_n")
    out["risk_components_used"] = ",".join(comps)
    return out

def composite_risk(RC: pd.DataFrame, weights: List[float]) -> pd.DataFrame:
    if RC.empty: return pd.DataFrame()
    used = RC["risk_components_used"].iloc[0].split(",") if "risk_components_used" in RC.columns else []
    # map provided weights to groups: dependence, HHI, vol(=avg of vols), stock_cover, climate
    # Build a per-row risk score
    w_dep, w_hhi, w_vol, w_stock, w_clim = weights
    # aggregate vols
    RC["vol_avg_n"] = RC[[c for c in ["vol_gprice_n","vol_fx_n","vol_ship_n"] if c in RC.columns]].mean(axis=1)
    parts = []
    if "import_dependence_n" in RC.columns: parts.append(w_dep * RC["import_dependence_n"])
    if "origin_hhi_n" in RC.columns:       parts.append(w_hhi * RC["origin_hhi_n"])
    parts.append(w_vol * RC["vol_avg_n"].fillna(0.0))
    if "stock_cover_n" in RC.columns:       parts.append(w_stock * RC["stock_cover_n"])
    if "climate_pressure_n" in RC.columns:  parts.append(w_clim * RC["climate_pressure_n"])
    RC["risk_score"] = 100.0 * np.sum(parts, axis=0) / max(1e-12, (w_dep + w_hhi + w_vol + w_stock + w_clim))
    return RC


# ----------------------------- scenarios -----------------------------

def pick_coef(el: pd.DataFrame, com: str, var: str, default: float) -> float:
    r = el[(el["commodity"]==com) & (el["var"]==var)]
    if not r.empty: return float(r["coef"].iloc[0])
    # fallback to average across commodities
    r2 = el[el["var"]==var]
    return float(r2["coef"].mean()) if not r2.empty else float(default)

def run_scenarios(P: pd.DataFrame, EL: pd.DataFrame, PT: pd.DataFrame,
                  FX_shock: float, PR_shock: float, SH_shock: float, PROD_shock: float,
                  POL_shift_pp: float, STOCK_rel_tons: float, horizon: int,
                  WGT: pd.DataFrame) -> pd.DataFrame:
    if P.empty: return pd.DataFrame()
    last = P.groupby("commodity").tail(1).set_index("commodity")
    rows = []
    for com, r in last.iterrows():
        base_vol = float(r["volume_tons"])
        base_bill = float(r["import_bill_usd"])
        # elasticities
        e_p  = pick_coef(EL, com, "dlog_gprice_cum_0..L", default=-0.30)
        e_fx = pick_coef(EL, com, "dlog_fx_cum_0..L",     default=-0.20)
        e_sh = pick_coef(EL, com, "dlog_ship_cum_0..L",   default=-0.10)
        # pass-through (for local price/CPI), use cum pass-through if available
        pt = pick_coef(PT, com, "pass_through_cum_0..L",  default=0.6)
        # shocks as dlog
        dlog_p  = np.log1p(PR_shock/100.0)
        dlog_fx = np.log1p(FX_shock/100.0)
        dlog_sh = np.log1p(SH_shock/100.0)
        # new volume via log-linear approx
        dv = e_p*dlog_p + e_fx*dlog_fx + e_sh*dlog_sh
        vol_t = base_vol * np.exp(dv)
        # stocks release offsets volume need in t+1 only
        vol_series = []
        for h in range(1, horizon+1):
            v = vol_t
            if h == 1 and STOCK_rel_tons and STOCK_rel_tons>0:
                v = max(0.0, v - STOCK_rel_tons)
            vol_series.append(v)
        # import price side
        gprice = float(r.get("gprice_usd", np.nan))
        fx     = float(r.get("usd_local", np.nan))
        ship   = float(r.get("freight_usd", 0.0))
        gprice_new = gprice * (1 + PR_shock/100.0) if gprice==gprice else np.nan
        fx_new     = fx * (1 + FX_shock/100.0) if fx==fx else np.nan
        ship_new   = ship * (1 + SH_shock/100.0) if ship==ship else np.nan
        # import bill approximation (CIF): (gprice_new*vol + ship_new*vol)
        bills = []
        for h, v in enumerate(vol_series, start=1):
            unit = (gprice_new if gprice_new==gprice_new else r["unit_value_usd"]) + (ship_new if ship_new==ship_new else 0.0)
            bills.append(float(v) * float(unit))
        # local price effect (for CPI)
        dlog_local = pt * (dlog_p + dlog_fx) + (0.2 * dlog_sh)
        # approximate food CPI impact as weight * local move
        w = 0.0
        if not WGT.empty:
            ww = WGT[(WGT["commodity"]==com)]["food_weight"]
            if not ww.empty: w = float(ww.iloc[0])
        cpi_impact_pp = 100.0 * w * dlog_local  # in percentage points
        for h, (v, b) in enumerate(zip(vol_series, bills), start=1):
            rows.append({"commodity": com, "h_month": h, "volume_tons": float(v), "import_bill_usd": float(b),
                         "dlog_local_price": float(dlog_local), "food_cpi_impact_pp": float(cpi_impact_pp),
                         "fx_shock_pct": FX_shock, "price_shock_pct": PR_shock, "ship_shock_pct": SH_shock,
                         "prod_shock_pct": PROD_shock, "policy_shift_pp": POL_shift_pp, "stocks_release_tons": STOCK_rel_tons})
    return pd.DataFrame(rows).sort_values(["commodity","h_month"])


# ----------------------------- stress (VaR/ES) -----------------------------

def stress_var_es(P: pd.DataFrame, n_sims: int=10000, horizon: int=12) -> pd.DataFrame:
    """
    Historical monthly returns of global price, FX, shipping → simulate dlog drivers for horizon.
    Import bill baseline from last observation; apply driver shocks using elasticities proxy on unit value.
    This is a coarse approximation when full elasticities are unavailable per commodity.
    """
    needed = [("gprice_usd","dlog_gprice"), ("usd_local","dlog_fx"), ("freight_usd","dlog_ship")]
    avail = [nm for col, nm in needed if col in P.columns]
    if not avail: return pd.DataFrame()
    # Build driver return matrix by date (use all commodities averaged)
    tmp = P.groupby("date")[avail].mean().dropna()
    if tmp.shape[0] < 24: return pd.DataFrame()
    R = tmp.copy()
    cov = R.cov().values
    mu  = R.mean().values
    # Simulate horizon 1 step (monthly)
    L = np.linalg.cholesky(cov + 1e-9*np.eye(len(avail)))
    rng = np.random.default_rng(42)
    sims = rng.standard_normal(size=(n_sims, len(avail))) @ L.T + mu
    # Baseline across commodities at last point
    last = P.groupby("commodity").tail(1)
    out = []
    for _, r in last.iterrows():
        g = float(r.get("gprice_usd", np.nan)); f = float(r.get("usd_local", np.nan)); s = float(r.get("freight_usd", 0.0))
        base_vol = float(r["volume_tons"]); base_bill = float(r["import_bill_usd"])
        # map sims to unit value change
        # assume unit value ∝ gprice * fx + ship; apply one-month shock, multiply by vol (hold vol constant for VaR)
        for i in range(n_sims):
            d = dict(zip(avail, sims[i,:]))
            ug = g * np.exp(d.get("dlog_gprice", 0.0)) if g==g else float(r["unit_value_usd"])
            uf = f * np.exp(d.get("dlog_fx", 0.0))     if f==f else 1.0
            us = s * np.exp(d.get("dlog_ship", 0.0))   if s==s else 0.0
            unit = (ug if g==g else float(r["unit_value_usd"])) + (us if s==s else 0.0)
            bill = base_vol * unit
            out.append({"commodity": r["commodity"], "sim": i, "import_bill_usd": float(bill)})
    DF = pd.DataFrame(out)
    sta = []
    for com, g in DF.groupby("commodity"):
        x = g["import_bill_usd"].values
        x_sorted = np.sort(x)
        var95 = float(np.percentile(x, 5))
        es95 = float(x_sorted[:max(1,int(0.05*len(x_sorted)))].mean())
        sta.append({"commodity": com, "VaR_5pct_bill_usd": var95, "ES_5pct_bill_usd": es95,
                    "mean_bill_usd": float(x.mean()), "sd_bill_usd": float(x.std(ddof=0)),
                    "n_sims": int(n_sims)})
    return pd.DataFrame(sta)


# ----------------------------- event study -----------------------------

def event_study(P: pd.DataFrame, POL: pd.DataFrame, window: int=3) -> pd.DataFrame:
    """
    CAR of Δlog(local price) around large positive/negative policy_index or ENSO shocks.
    """
    if P.empty or "dlog_lprice" not in P.columns:
        return pd.DataFrame()
    series = P.set_index(["commodity","date"])["dlog_lprice"].unstack(0).sort_index()
    # Build event dates: large |Δpolicy| or |Δenso_idx|
    events = []
    if not POL.empty:
        polm = POL.groupby("date")["value"].sum(min_count=1)
        z = (polm - polm.rolling(12).mean()) / (polm.rolling(12).std(ddof=0) + 1e-12)
        ev = z[abs(z)>=1.0]
        for dt, zz in ev.items():
            events.append(("POLICY", pd.Timestamp(dt)))
    if "enso_idx" in P.columns:
        en = P.groupby("date")["enso_idx"].mean()
        z = (en - en.rolling(12).mean())/(en.rolling(12).std(ddof=0)+1e-12)
        ev = z[abs(z)>=1.0]
        for dt, zz in ev.items():
            events.append(("ENSO", pd.Timestamp(dt)))
    if not events: return pd.DataFrame()
    rows = []
    idx = series.index
    for lab, dt in events:
        if dt not in idx:
            pos = idx.searchsorted(dt); 
            if pos>=len(idx): continue
            dt = idx[pos]
        i0 = idx.get_loc(dt)
        iL = max(0, i0-window); iR = min(len(idx)-1, i0+window)
        win = series.iloc[iL:iR+1]
        car = win.sum(axis=0)  # sum Δlogs
        for com, val in car.items():
            rows.append({"event": lab, "event_date": str(pd.Timestamp(dt).date()), "commodity": com, "CAR_dlog_local": float(val)})
    return pd.DataFrame(rows)


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    trade: str
    gprice: Optional[str]
    lprice: Optional[str]
    fx: Optional[str]
    ship: Optional[str]
    policy: Optional[str]
    stocks: Optional[str]
    weather: Optional[str]
    macro: Optional[str]
    production: Optional[str]
    weights: Optional[str]
    country: str
    commodities: str
    freq: str
    lags: int
    horizon: int
    n_sims: int
    fx_shock_pct: float
    price_shock_pct: float
    ship_shock_pct: float
    prod_shock_pct: float
    policy_shift_pp: float
    stocks_release_tons: float
    risk_weights: str
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Food security — imports, pass-through, risk & scenarios")
    ap.add_argument("--trade", required=True)
    ap.add_argument("--gprice", default="")
    ap.add_argument("--lprice", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--ship", default="")
    ap.add_argument("--policy", default="")
    ap.add_argument("--stocks", default="")
    ap.add_argument("--weather", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--production", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--country", required=True)
    ap.add_argument("--commodities", default="ALL")
    ap.add_argument("--freq", default="monthly", choices=["monthly"])
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n_sims", type=int, default=10000)
    ap.add_argument("--fx_shock_pct", type=float, default=0.0)
    ap.add_argument("--price_shock_pct", type=float, default=0.0)
    ap.add_argument("--ship_shock_pct", type=float, default=0.0)
    ap.add_argument("--prod_shock_pct", type=float, default=0.0)
    ap.add_argument("--policy_shift_pp", type=float, default=0.0)
    ap.add_argument("--stocks_release_tons", type=float, default=0.0)
    ap.add_argument("--risk_weights", default="0.3,0.2,0.2,0.2,0.1")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_foodsec")
    ap.add_argument("--min_obs", type=int, default=48)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    commodities = None if not args.commodities else [s.strip() for s in args.commodities.split(",")]

    TR  = load_trade(args.trade, country_key=args.country)
    GP  = load_gprice(args.gprice)    if args.gprice    else pd.DataFrame()
    LP  = load_lprice(args.lprice)    if args.lprice    else pd.DataFrame()
    FX  = load_fx(args.fx)            if args.fx        else pd.DataFrame()
    SH  = load_shipping(args.ship)    if args.ship      else pd.DataFrame()
    POL = load_policy(args.policy)    if args.policy    else pd.DataFrame()
    ST  = load_stocks(args.stocks)    if args.stocks    else pd.DataFrame()
    WEA = load_weather(args.weather)  if args.weather   else pd.DataFrame()
    MAC = load_macro(args.macro)      if args.macro     else pd.DataFrame()
    PRD = load_production(args.production) if args.production else pd.DataFrame()
    WGT = load_weights(args.weights)  if args.weights   else pd.DataFrame()

    # time filters
    if args.start:
        s = eom(pd.Series([args.start])).iloc[0]
        for df in [TR, GP, LP, FX, SH, POL, ST, WEA, MAC, PRD]:
            if not df.empty: df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = eom(pd.Series([args.end])).iloc[0]
        for df in [TR, GP, LP, FX, SH, POL, ST, WEA, MAC, PRD]:
            if not df.empty: df.drop(df[df["date"] > e].index, inplace=True)

    # panel
    P = build_panel(TR, GP, LP, FX, SH, POL, ST, WEA, MAC, PRD, country_key=args.country, commodities=commodities)
    if P.empty:
        raise ValueError("Panel is empty after filters. Check inputs/country/commodities.")
    P.to_csv(outdir / "panel.csv", index=False)

    # elasticities
    EL = regress_elasticities(P, L=int(args.lags), min_obs=int(args.min_obs))
    if not EL.empty: EL.to_csv(outdir / "elasticity_imports.csv", index=False)
    PT = pass_through(P, L=int(args.lags), min_obs=int(args.min_obs))
    if not PT.empty: PT.to_csv(outdir / "elasticity_passthrough.csv", index=False)

    # risk
    RC = risk_components(P, PRD)
    if not RC.empty:
        w = [float(x) for x in args.risk_weights.split(",")]
        if len(w) != 5: w = [0.3,0.2,0.2,0.2,0.1]
        RISK = composite_risk(RC, w)
        RISK.to_csv(outdir / "risk_index.csv", index=False)
    else:
        RISK = pd.DataFrame()

    # scenarios
    SCN = run_scenarios(P, EL if not EL.empty else pd.DataFrame(),
                        PT if not PT.empty else pd.DataFrame(),
                        FX_shock=float(args.fx_shock_pct),
                        PR_shock=float(args.price_shock_pct),
                        SH_shock=float(args.ship_shock_pct),
                        PROD_shock=float(args.prod_shock_pct),
                        POL_shift_pp=float(args.policy_shift_pp),
                        STOCK_rel_tons=float(args.stocks_release_tons),
                        horizon=int(args.horizon), WGT=WGT if not WGT.empty else pd.DataFrame())
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # stress VaR/ES
    STRESS = stress_var_es(P, n_sims=int(args.n_sims), horizon=int(args.horizon))
    if not STRESS.empty: STRESS.to_csv(outdir / "stress_vares.csv", index=False)

    # events
    ES = event_study(P, POL, window=3) if not POL.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # summary
    summary = {
        "country": args.country,
        "commodities": sorted(P["commodity"].unique().tolist()),
        "sample": {
            "start": str(P["date"].min().date()),
            "end": str(P["date"].max().date()),
            "months": int(P["date"].nunique())
        },
        "n_obs": int(P.shape[0]),
        "has_local_price": bool("lprice_local" in P.columns),
        "has_fx": bool("usd_local" in P.columns),
        "has_shipping": bool("freight_usd" in P.columns),
        "has_stocks": bool("stocks_tons" in P.columns),
        "has_production": bool("production_tons" in P.columns),
        "outputs": {
            "panel": "panel.csv",
            "elasticity_imports": "elasticity_imports.csv" if not EL.empty else None,
            "elasticity_passthrough": "elasticity_passthrough.csv" if not PT.empty else None,
            "risk_index": "risk_index.csv" if not RISK.empty else None,
            "scenarios": "scenarios.csv" if not SCN.empty else None,
            "stress_vares": "stress_vares.csv" if not STRESS.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # config echo
    cfg = asdict(Config(
        trade=args.trade, gprice=(args.gprice or None), lprice=(args.lprice or None),
        fx=(args.fx or None), ship=(args.ship or None), policy=(args.policy or None),
        stocks=(args.stocks or None), weather=(args.weather or None), macro=(args.macro or None),
        production=(args.production or None), weights=(args.weights or None),
        country=args.country, commodities=args.commodities, freq=args.freq, lags=int(args.lags),
        horizon=int(args.horizon), n_sims=int(args.n_sims),
        fx_shock_pct=float(args.fx_shock_pct), price_shock_pct=float(args.price_shock_pct),
        ship_shock_pct=float(args.ship_shock_pct), prod_shock_pct=float(args.prod_shock_pct),
        policy_shift_pp=float(args.policy_shift_pp), stocks_release_tons=float(args.stocks_release_tons),
        risk_weights=args.risk_weights, start=(args.start or None), end=(args.end or None),
        outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # console
    print("== Food Security — Imports Toolkit ==")
    print(f"Country: {args.country} | Commodities: {', '.join(summary['commodities'])}")
    print(f"Sample: {summary['sample']['start']} → {summary['sample']['end']} | Obs: {summary['n_obs']}")
    if summary["outputs"]["elasticity_imports"]: print("Volume elasticities estimated.")
    if summary["outputs"]["elasticity_passthrough"]: print("Pass-through elasticities estimated.")
    if summary["outputs"]["risk_index"]: print("Risk index computed.")
    if summary["outputs"]["scenarios"]: print(f"Scenario horizon: {args.horizon} months.")
    if summary["outputs"]["stress_vares"]: print(f"Stress VaR/ES done (n={args.n_sims}).")
    if summary["outputs"]["event_study"]: print("Event study written.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
