#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inbound_tourism.py — Drivers, elasticities, events & scenarios for inbound tourism
----------------------------------------------------------------------------------

What this does
==============
Builds a **monthly** destination-country tourism panel and quantifies how arrivals and
receipts respond to core drivers: **FX**, **air capacity**, **visa policy**, **safety/advisories**,
**prices**, **global travel demand**, **weather/seasonality**, and **restrictions**. Then it
runs event studies, estimates elasticities with HAC/Newey–West SEs, and projects scenarios.

It ingests:
- arrivals by origin (optional) and destination
- spend (total or per-visitor)
- FX (USD/local and/or REER)
- air capacity (flights, seats)
- visa policy events
- safety/advisory indices
- prices (hotel/tourism CPI)
- global demand index
- restrictions (e.g., pandemic stringency)
- weather (optional)

Core outputs:
1) Clean monthly **panel.csv** (destination-level, plus concentration metrics by origin)
2) **elasticity_arrivals.csv** and **elasticity_spend.csv** (pass-through to spend/visitor)
3) **event_study.csv** (CARs around visa/safety/air-capacity shocks)
4) **scenarios.csv** (H-month projections for arrivals & receipts under user shocks)
5) **stress_vares.csv** (Monte Carlo VaR/ES on receipts using driver covariances)
6) **summary.json** and **config.json**

DISCLAIMER: Research tooling with simplifying assumptions. Validate locally before decisions.
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
        key = cand.lower()
        for c in df.columns:
            if key in str(c).lower(): return c
    return None

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def eom(s: pd.Series) -> pd.Series:
    return to_dt(s) + pd.offsets.MonthEnd(0)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(x: pd.Series) -> pd.Series:
    x = safe_num(x).replace(0, np.nan)
    return np.log(x) - np.log(x.shift(1))

def winsor(x: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

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

def load_arrivals(path: str, country_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      A_tot: monthly totals per destination (date,country,arrivals)
      A_by:  by-origin arrivals (date,country,origin,arrivals) if available; else empty
    """
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country","destination"); o = ncol(df,"origin_country","origin")
    a = ncol(df,"arrivals","visitors","tourist_arrivals")
    if not (d and c and a):
        raise ValueError("arrivals.csv must include date, country, arrivals (origin optional).")
    df = df.rename(columns={d:"date", c:"country", a:"arrivals"})
    if o: df = df.rename(columns={o:"origin"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    # filter destination country (substring)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    df["arrivals"] = safe_num(df["arrivals"])
    if "origin" in df.columns:
        by = (df.groupby(["date","country","origin"], as_index=False)["arrivals"].sum())
        tot = (by.groupby(["date","country"], as_index=False)["arrivals"].sum())
        by["origin"] = by["origin"].astype(str).str.upper().str.strip()
        return tot, by
    else:
        tot = (df.groupby(["date","country"], as_index=False)["arrivals"].sum())
        return tot, pd.DataFrame()

def load_spend(path: Optional[str], country_key: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); s = ncol(df,"spend_usd","receipts_usd","tourism_receipts_usd")
    pv = ncol(df,"spend_per_visitor_usd","spend_per_visitor","spend_per_tourist_usd")
    a  = ncol(df,"arrivals")  # optional if per-visitor given
    if not (d and c and (s or pv)):
        raise ValueError("spend.csv needs date,country and spend_usd or spend_per_visitor_usd.")
    df = df.rename(columns={d:"date", c:"country"})
    if s: df = df.rename(columns={s:"spend_usd"})
    if pv: df = df.rename(columns={pv:"spend_per_visitor_usd"})
    if a: df = df.rename(columns={a:"arrivals"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    for k in ["spend_usd","spend_per_visitor_usd","arrivals"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # derive per-visitor if not supplied
    if "spend_per_visitor_usd" not in df.columns and {"spend_usd","arrivals"}.issubset(df.columns):
        df["spend_per_visitor_usd"] = df["spend_usd"] / df["arrivals"].replace(0,np.nan)
    return df

def load_fx(path: Optional[str], country_key: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    usd = ncol(df,"usd_local","usd_per_local")
    reer = ncol(df,"reer_index","reer")
    if not (d and c and (usd or reer)):
        raise ValueError("fx.csv needs date,country and usd_local and/or reer_index.")
    df = df.rename(columns={d:"date", c:"country"})
    if usd: df = df.rename(columns={usd:"usd_local"})
    if reer: df = df.rename(columns={reer:"reer_index"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    for k in ["usd_local","reer_index"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_air(path: Optional[str], country_key: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    fl = ncol(df,"flights","departures"); st = ncol(df,"seats","available_seats"); lf = ncol(df,"load_factor","lf")
    if not (d and c and (fl or st)):
        raise ValueError("air.csv needs date,country and flights or seats.")
    df = df.rename(columns={d:"date", c:"country"})
    if fl: df = df.rename(columns={fl:"flights"})
    if st: df = df.rename(columns={st:"seats"})
    if lf: df = df.rename(columns={lf:"load_factor"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    for k in ["flights","seats","load_factor"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_policy(path: Optional[str], country_key: str) -> pd.DataFrame:
    """
    Visa policy events: type ∈ {VISA_EASE, VISA_TIGHTEN, E_VISA, VOA, VISA_FREE, VISA_SUSPEND}
    value: optional (+1 ease, -1 tighten); will be auto-inferred by 'type' if missing.
    """
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); t = ncol(df,"type","event"); v = ncol(df,"value","delta")
    if not (d and c and t):
        raise ValueError("visa.csv needs date,country,type.")
    df = df.rename(columns={d:"date", c:"country", t:"type"})
    if v: df = df.rename(columns={v:"value"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    df["type"] = df["type"].astype(str).str.upper().str.strip()
    if "value" not in df.columns:
        df["value"] = 0.0
        df.loc[df["type"].str.contains("EASE|E_VISA|VOA|FREE"), "value"] = +1.0
        df.loc[df["type"].str.contains("TIGHTEN|SUSPEND"), "value"] = -1.0
    df["value"] = safe_num(df["value"])
    return df

def load_safety(path: Optional[str], country_key: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    s = ncol(df,"safety_index","advisory_score","travel_advisory")
    inc = ncol(df,"incidents","events")
    if not (d and c and (s or inc)): raise ValueError("safety.csv needs date,country and safety_index or incidents.")
    df = df.rename(columns={d:"date", c:"country"})
    if s: df = df.rename(columns={s:"safety_index"})
    if inc: df = df.rename(columns={inc:"incidents"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    for k in ["safety_index","incidents"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_prices(path: Optional[str], country_key: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    h = ncol(df,"hotel_price_idx","hpi"); f = ncol(df,"food_cpi","cpi_food"); t = ncol(df,"transport_cpi","cpi_transport")
    if not (d and c and (h or f or t)): raise ValueError("prices.csv needs date,country and at least one price index.")
    df = df.rename(columns={d:"date", c:"country"})
    if h: df = df.rename(columns={h:"hotel_price_idx"})
    if f: df = df.rename(columns={f:"food_cpi"})
    if t: df = df.rename(columns={t:"transport_cpi"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    for k in ["hotel_price_idx","food_cpi","transport_cpi"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_global(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date")
    g = ncol(df,"global_travel_demand_idx","global_travel_idx","google_travel_idx")
    y = ncol(df,"global_income_proxy","real_gdp_world")
    if not d: raise ValueError("global.csv needs date.")
    df = df.rename(columns={d:"date"})
    if g: df = df.rename(columns={g:"global_travel_idx"})
    if y: df = df.rename(columns={y:"global_income_proxy"})
    df["date"] = eom(df["date"])
    for k in ["global_travel_idx","global_income_proxy"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_restrict(path: Optional[str], country_key: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country"); r = ncol(df,"restrictions_index","stringency_index")
    if not (d and c and r): raise ValueError("restrict.csv needs date,country,restrictions_index.")
    df = df.rename(columns={d:"date", c:"country", r:"restrictions_index"})
    df["date"] = eom(df["date"]); df["country"] = df["country"].astype(str)
    key = str(country_key).lower()
    df = df[df["country"].str.lower().str.contains(key)]
    df["restrictions_index"] = safe_num(df["restrictions_index"])
    return df

def load_weather(path: Optional[str], country_key: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); c = ncol(df,"country")
    r = ncol(df,"rain_anom","rainfall_anom"); t = ncol(df,"temp_anom","temperature_anom")
    if not d: raise ValueError("weather.csv needs date.")
    df = df.rename(columns={d:"date"})
    if c: df = df.rename(columns={c:"country"})
    if r: df = df.rename(columns={r:"rain_anom"})
    if t: df = df.rename(columns={t:"temp_anom"})
    df["date"] = eom(df["date"])
    if "country" in df.columns:
        key = str(country_key).lower()
        df = df[df["country"].astype(str).str.lower().str.contains(key)]
    for k in ["rain_anom","temp_anom"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df


# ----------------------------- panel construction -----------------------------

def origin_concentration(arrivals_by: pd.DataFrame) -> pd.DataFrame:
    if arrivals_by.empty: return pd.DataFrame()
    g = arrivals_by.copy()
    tot = g.groupby(["date","country"], as_index=False)["arrivals"].sum().rename(columns={"arrivals":"tot"})
    g = g.merge(tot, on=["date","country"], how="left")
    g["share"] = g["arrivals"] / g["tot"].replace(0,np.nan)
    def hhi(s: pd.Series) -> float:
        x = safe_num(s).fillna(0).clip(0,1).values
        return float(np.sum((x*100)**2) / 10000.0)  # 0..1
    H = g.groupby(["date","country"], as_index=False)["share"].apply(hhi).rename(columns={"share":"origin_hhi"})
    return H

def build_panel(country_key: str,
                ARR: pd.DataFrame, ARR_BY: pd.DataFrame, SPD: pd.DataFrame,
                FX: pd.DataFrame, AIR: pd.DataFrame, VIS: pd.DataFrame,
                SAF: pd.DataFrame, PRC: pd.DataFrame, GBL: pd.DataFrame,
                RST: pd.DataFrame, WEA: pd.DataFrame) -> pd.DataFrame:
    P = ARR.copy()
    # origin HHI
    H = origin_concentration(ARR_BY)
    if not H.empty:
        P = P.merge(H, on=["date","country"], how="left")
    # Spend
    if not SPD.empty:
        P = P.merge(SPD[["date","country"] + [c for c in ["spend_usd","spend_per_visitor_usd"] if c in SPD.columns]],
                    on=["date","country"], how="left")
    # FX & REER
    if not FX.empty:  P = P.merge(FX, on=["date","country"], how="left")
    # Air capacity
    if not AIR.empty: P = P.merge(AIR, on=["date","country"], how="left")
    # Visa policy → monthly index (sum of event values)
    if not VIS.empty:
        vis_m = VIS.groupby(["date","country"], as_index=False)["value"].sum().rename(columns={"value":"visa_index"})
        P = P.merge(vis_m, on=["date","country"], how="left")
    # Safety/advisory
    if not SAF.empty: P = P.merge(SAF, on=["date","country"], how="left")
    # Prices
    if not PRC.empty: P = P.merge(PRC, on=["date","country"], how="left")
    # Global demand
    if not GBL.empty: P = P.merge(GBL, on=["date"], how="left")
    # Restrictions
    if not RST.empty: P = P.merge(RST, on=["date","country"], how="left")
    # Weather
    if not WEA.empty:
        if "country" in WEA.columns:
            P = P.merge(WEA, on=["date","country"], how="left")
        else:
            P = P.merge(WEA, on=["date"], how="left")
    # Derived
    P = P.sort_values("date")
    P["dlog_arrivals"] = dlog(P["arrivals"])
    if "spend_per_visitor_usd" in P.columns:
        P["dlog_spv"] = dlog(P["spend_per_visitor_usd"])
    if "spend_usd" in P.columns:
        P["dlog_spend"] = dlog(P["spend_usd"])
    if "usd_local" in P.columns:     P["dlog_usd_local"] = dlog(P["usd_local"])      # + = USD↑ (local weaker)
    if "reer_index" in P.columns:    P["dlog_reer"] = dlog(P["reer_index"])          # + = local stronger vs basket
    if "seats" in P.columns:         P["dlog_seats"] = dlog(P["seats"])
    if "flights" in P.columns:       P["dlog_flights"] = dlog(P["flights"])
    if "hotel_price_idx" in P.columns: P["dlog_hotel"] = dlog(P["hotel_price_idx"])
    if "food_cpi" in P.columns:        P["dlog_food"]  = dlog(P["food_cpi"])
    if "transport_cpi" in P.columns:   P["dlog_trans"] = dlog(P["transport_cpi"])
    if "global_travel_idx" in P.columns: P["dlog_global_travel"] = dlog(P["global_travel_idx"])
    # receipts (prefer direct spend_usd, else arrivals * spv)
    P["receipts_usd"] = np.where(
        "spend_usd" in P.columns,
        P.get("spend_usd", np.nan),
        P["arrivals"] * P.get("spend_per_visitor_usd", np.nan)
    )
    # seasonality controls
    P["month"] = P["date"].dt.month
    return P


# ----------------------------- regressions -----------------------------

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

def regress_elasticities_arrivals(P: pd.DataFrame, L: int, min_obs: int) -> pd.DataFrame:
    if P.empty: return pd.DataFrame()
    g = P.sort_values("date").copy()
    y = g["dlog_arrivals"]
    # Driver candidates (only those present)
    drivers = [c for c in [
        "dlog_usd_local",     # USD↑ (local↓) → destination cheaper for USD earners → arrivals ↑ (expected sign +)
        "dlog_reer",          # REER↑ (local stronger) → arrivals ↓ (expected sign -)
        "dlog_seats", "dlog_flights",
        "visa_index",         # easing positive
        "safety_index",       # higher = safer
        "incidents",          # negative driver (levels)
        "dlog_hotel", "dlog_food", "dlog_trans",  # cost-of-stay
        "dlog_global_travel",
        "restrictions_index", # higher = tighter
        "rain_anom", "temp_anom",
        "origin_hhi"          # concentration (risk; level)
    ] if c in g.columns]
    X1, _ = build_dlag_matrix(g, L, cols=[c for c in drivers if c.startswith("dlog_")])
    add_levels = [c for c in drivers if not c.startswith("dlog_")]
    X2 = g[add_levels] if add_levels else pd.DataFrame(index=g.index)
    D = pd.get_dummies(g["month"].astype(int), prefix="m", drop_first=True)
    X = pd.concat([pd.Series(1.0, index=g.index, name="const"), X1, X2, D], axis=1).dropna()
    Y = y.loc[X.index]
    if X.shape[0] < max(min_obs, 5*X.shape[1]//3): return pd.DataFrame()
    beta, resid, XTX_inv = ols_beta_resid(X.values, Y.values.reshape(-1,1))
    se = hac_se(X.values, resid, XTX_inv, L=max(6, L))
    rows = []
    for i, nm in enumerate(X.columns):
        rows.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                     "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                     "n": int(X.shape[0]), "lags": int(L)})
    # cumulative elasticities for key dlog drivers
    for base in ["dlog_usd_local","dlog_reer","dlog_seats","dlog_flights","dlog_hotel","dlog_global_travel"]:
        idxs = [i for i, nm in enumerate(X.columns) if nm.startswith(base+"_l")]
        if idxs:
            bsum = float(np.sum([beta[i,0] for i in idxs]))
            sesq = float(np.sum([se[i]**2 for i in idxs]))
            rows.append({"var": base+"_cum_0..L", "coef": bsum, "se": np.sqrt(sesq),
                         "t_stat": bsum/np.sqrt(sesq) if sesq>0 else np.nan,
                         "n": int(X.shape[0]), "lags": int(L)})
    return pd.DataFrame(rows)

def regress_elasticities_spend(P: pd.DataFrame, L: int, min_obs: int) -> pd.DataFrame:
    if P.empty or "spend_per_visitor_usd" not in P.columns: return pd.DataFrame()
    g = P.sort_values("date").copy()
    y = g["dlog_spv"]
    if y.notna().sum() < max(min_obs, 24): return pd.DataFrame()
    drivers = [c for c in [
        "dlog_usd_local","dlog_reer",            # destination cheapness vs origin
        "dlog_hotel","dlog_food","dlog_trans",   # in-destination prices
        "dlog_global_travel","visa_index","safety_index","restrictions_index"
    ] if c in g.columns]
    X1, _ = build_dlag_matrix(g, L, cols=[c for c in drivers if c.startswith("dlog_")])
    add_levels = [c for c in drivers if not c.startswith("dlog_")]
    X2 = g[add_levels] if add_levels else pd.DataFrame(index=g.index)
    D = pd.get_dummies(g["month"].astype(int), prefix="m", drop_first=True)
    X = pd.concat([pd.Series(1.0, index=g.index, name="const"), X1, X2, D], axis=1).dropna()
    Y = y.loc[X.index]
    if X.shape[0] < max(min_obs, 5*X.shape[1]//3): return pd.DataFrame()
    beta, resid, XTX_inv = ols_beta_resid(X.values, Y.values.reshape(-1,1))
    se = hac_se(X.values, resid, XTX_inv, L=max(6, L))
    rows = []
    for i, nm in enumerate(X.columns):
        rows.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i]),
                     "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                     "n": int(X.shape[0]), "lags": int(L)})
    for base in ["dlog_usd_local","dlog_reer","dlog_hotel"]:
        idxs = [i for i, nm in enumerate(X.columns) if nm.startswith(base+"_l")]
        if idxs:
            bsum = float(np.sum([beta[i,0] for i in idxs]))
            sesq = float(np.sum([se[i]**2 for i in idxs]))
            rows.append({"var": base+"_cum_0..L", "coef": bsum, "se": np.sqrt(sesq),
                         "t_stat": bsum/np.sqrt(sesq) if sesq>0 else np.nan,
                         "n": int(X.shape[0]), "lags": int(L)})
    return pd.DataFrame(rows)


# ----------------------------- event study -----------------------------

def zscore(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w, min_periods=max(6, w//3)).mean()
    sd = s.rolling(w, min_periods=max(6, w//3)).std(ddof=0)
    return (s - mu) / (sd + 1e-12)

def car(series: pd.Series, anchor: pd.Timestamp, pre: int, post: int) -> float:
    idx = series.index
    if anchor not in idx:
        pos = idx.searchsorted(anchor)
        if pos >= len(idx): return np.nan
        anchor = idx[pos]
    i0 = idx.searchsorted(anchor) - pre
    i1 = idx.searchsorted(anchor) + post
    i0 = max(0, i0); i1 = min(len(idx)-1, i1)
    return float(series.iloc[i0:i1+1].sum())

def event_study(P: pd.DataFrame, VIS: pd.DataFrame, AIR: pd.DataFrame, SAF: pd.DataFrame,
                window: int=3, z_window: int=12, z_thr: float=1.0) -> pd.DataFrame:
    if P.empty: return pd.DataFrame()
    ser = P.set_index("date")["dlog_arrivals"].dropna().sort_index()
    events = []
    # Visa events: any entry is an event
    if not VIS.empty:
        for dt in sorted(VIS["date"].unique()):
            events.append(("VISA", pd.Timestamp(dt)))
    # Air capacity shocks: large z in Δlog(seats)
    if not AIR.empty and "dlog_seats" in P.columns:
        z = zscore(P.set_index("date")["dlog_seats"].dropna(), w=z_window)
        ev = z[abs(z)>=z_thr]
        for dt in ev.index:
            events.append(("AIR", pd.Timestamp(dt)))
    # Safety shocks: large change in safety_index (use z)
    if not SAF.empty and "safety_index" in P.columns:
        z = zscore(P.set_index("date")["safety_index"].dropna().pct_change(), w=z_window)
        ev = z[abs(z)>=z_thr]
        for dt in ev.index:
            events.append(("SAFETY", pd.Timestamp(dt)))
    # Compute CARs
    rows = []
    for lab, dt in events:
        rows.append({"event": lab, "event_date": str(dt.date()), "CAR_dlog_arrivals": car(ser, dt, window, window)})
    return pd.DataFrame(rows).sort_values(["event_date","event"])


# ----------------------------- scenarios -----------------------------

def pick_coef(EL: pd.DataFrame, var: str, default: float) -> float:
    r = EL[EL["var"]==var]
    if not r.empty: return float(r["coef"].iloc[0])
    # try cumulative first if base asked differently
    if var.endswith("_cum_0..L"):
        base = var.replace("_cum_0..L","")
        rr = EL[EL["var"]==base+"_cum_0..L"]
        if not rr.empty: return float(rr["coef"].iloc[0])
    return float(default)

def run_scenarios(P: pd.DataFrame, ELA: pd.DataFrame, ELS: pd.DataFrame,
                  fx_pct: float, reer_pct: float, seats_pct: float,
                  visa_shift: float, safety_shift: float, price_pct: float,
                  global_demand_pct: float, restrictions_shift: float,
                  horizon: int) -> pd.DataFrame:
    """
    Apply simple log-linear elasticities to last observation to project arrivals & receipts.
    """
    if P.empty: return pd.DataFrame()
    last = P.tail(1).iloc[0]
    base_arr = float(last["arrivals"])
    base_spv = float(last.get("spend_per_visitor_usd", np.nan))
    base_rec = float(last.get("receipts_usd", base_arr * (base_spv if base_spv==base_spv else 0.0)))
    # arrivals elasticities (cumulative)
    e_usd  = pick_coef(ELA, "dlog_usd_local_cum_0..L", default=+0.30) if "dlog_usd_local" in P.columns else 0.0
    e_reer = pick_coef(ELA, "dlog_reer_cum_0..L",      default=-0.40) if "dlog_reer" in P.columns else 0.0
    e_seat = pick_coef(ELA, "dlog_seats_cum_0..L",     default=+0.50) if "dlog_seats" in P.columns else 0.0
    e_hotel= pick_coef(ELA, "dlog_hotel_cum_0..L",     default=-0.20) if "dlog_hotel" in P.columns else 0.0
    e_glob = pick_coef(ELA, "dlog_global_travel_cum_0..L", default=+0.25) if "dlog_global_travel" in P.columns else 0.0
    b_visa = float(ELA[ELA["var"]=="visa_index"]["coef"].iloc[0]) if "visa_index" in P.columns and not ELA[ELA["var"]=="visa_index"].empty else +0.02
    b_safe = float(ELA[ELA["var"]=="safety_index"]["coef"].iloc[0]) if "safety_index" in P.columns and not ELA[ELA["var"]=="safety_index"].empty else +0.01
    b_rest = float(ELA[ELA["var"]=="restrictions_index"]["coef"].iloc[0]) if "restrictions_index" in P.columns and not ELA[ELA["var"]=="restrictions_index"].empty else -0.02
    # spend/visitor elasticities
    s_usd  = pick_coef(ELS, "dlog_usd_local_cum_0..L", default=+0.10) if not ELS.empty and "dlog_usd_local" in P.columns else 0.0
    s_reer = pick_coef(ELS, "dlog_reer_cum_0..L",      default=-0.15) if not ELS.empty and "dlog_reer" in P.columns else 0.0
    s_hotel= pick_coef(ELS, "dlog_hotel_cum_0..L",     default=+0.20) if not ELS.empty and "dlog_hotel" in P.columns else 0.0

    # convert scenario % to dlog
    d_usd   = np.log1p(fx_pct/100.0)
    d_reer  = np.log1p(reer_pct/100.0)
    d_seats = np.log1p(seats_pct/100.0)
    d_hotel = np.log1p(price_pct/100.0)
    d_glob  = np.log1p(global_demand_pct/100.0)

    dv_arr = e_usd*d_usd + e_reer*d_reer + e_seat*d_seats + e_hotel*d_hotel + e_glob*d_glob \
             + b_visa*visa_shift + b_safe*safety_shift + b_rest*restrictions_shift
    dv_spv = s_usd*d_usd + s_reer*d_reer + s_hotel*d_hotel

    rows = []
    for h in range(1, horizon+1):
        arr = base_arr * np.exp(dv_arr)
        spv = base_spv * np.exp(dv_spv) if base_spv==base_spv else np.nan
        rec = float(arr) * (float(spv) if spv==spv else (base_rec/base_arr if base_arr>0 else 0.0))
        rows.append({
            "h_month": h, "arrivals": float(arr), "spend_per_visitor_usd": float(spv) if spv==spv else np.nan,
            "receipts_usd": float(rec),
            "fx_pct": fx_pct, "reer_pct": reer_pct, "seats_pct": seats_pct,
            "visa_shift": visa_shift, "safety_shift": safety_shift,
            "price_pct": price_pct, "global_demand_pct": global_demand_pct,
            "restrictions_shift": restrictions_shift
        })
    return pd.DataFrame(rows)


# ----------------------------- stress (VaR/ES) -----------------------------

def stress_var_es(P: pd.DataFrame, n_sims: int=10000) -> pd.DataFrame:
    """
    Simple one-month receipts distribution using historical dlog drivers' covariances.
    Holds arrivals elasticity vector from point estimates (fallback priors).
    """
    need = [c for c in ["dlog_usd_local","dlog_reer","dlog_seats","dlog_hotel","dlog_global_travel"] if c in P.columns]
    if not need: return pd.DataFrame()
    R = P[need].dropna()
    if R.shape[0] < 24: return pd.DataFrame()
    mu = R.mean().values; cov = R.cov().values
    L = np.linalg.cholesky(cov + 1e-12*np.eye(len(need)))
    rng = np.random.default_rng(42)
    shocks = rng.standard_normal(size=(n_sims, len(need))) @ L.T + mu
    last = P.tail(1).iloc[0]
    base_arr = float(last["arrivals"])
    base_spv = float(last.get("spend_per_visitor_usd", np.nan))
    base_rec = float(last.get("receipts_usd", base_arr * (base_spv if base_spv==base_spv else 0.0)))

    # Priors for cumulative elasticities if none estimated
    priors = {
        "dlog_usd_local": +0.30, "dlog_reer": -0.40,
        "dlog_seats": +0.50, "dlog_hotel": -0.20, "dlog_global_travel": +0.25
    }
    # build vector in need order
    evec = np.array([priors[c] for c in need])
    recs = []
    for i in range(n_sims):
        dv_arr = float(evec @ shocks[i,:])
        arr = base_arr * np.exp(dv_arr)
        rec = float(arr) * (base_rec/base_arr if base_arr>0 else 0.0)
        recs.append(rec)
    x = np.array(recs)
    x_sorted = np.sort(x)
    var5 = float(np.percentile(x, 5))
    es5 = float(x_sorted[:max(1,int(0.05*len(x_sorted)))].mean())
    return pd.DataFrame([{
        "VaR_5pct_receipts_usd": var5,
        "ES_5pct_receipts_usd": es5,
        "mean_receipts_usd": float(x.mean()),
        "sd_receipts_usd": float(x.std(ddof=0)),
        "n_sims": int(n_sims)
    }])


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    arrivals: str
    spend: Optional[str]
    fx: Optional[str]
    air: Optional[str]
    visa: Optional[str]
    safety: Optional[str]
    prices: Optional[str]
    global_idx: Optional[str]
    restrict: Optional[str]
    weather: Optional[str]
    country: str
    lags: int
    horizon: int
    n_sims: int
    z_window: int
    z_threshold: float
    event_window: int
    fx_pct: float
    reer_pct: float
    seats_pct: float
    visa_shift: float
    safety_shift: float
    price_pct: float
    global_demand_pct: float
    restrictions_shift: float
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inbound tourism — elasticities, events, scenarios")
    ap.add_argument("--arrivals", required=True, help="CSV with date,country[,origin],arrivals")
    ap.add_argument("--spend", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--air", default="")
    ap.add_argument("--visa", default="")
    ap.add_argument("--safety", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--global_idx", default="")
    ap.add_argument("--restrict", default="")
    ap.add_argument("--weather", default="")
    ap.add_argument("--country", required=True)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n_sims", type=int, default=10000)
    ap.add_argument("--z_window", type=int, default=12)
    ap.add_argument("--z_threshold", type=float, default=1.0)
    ap.add_argument("--event_window", type=int, default=3)
    # Scenario shocks
    ap.add_argument("--fx_pct", type=float, default=0.0, help="USD/local % change (+ = USD up)")
    ap.add_argument("--reer_pct", type=float, default=0.0, help="REER % change (+ = local stronger)")
    ap.add_argument("--seats_pct", type=float, default=0.0)
    ap.add_argument("--visa_shift", type=float, default=0.0, help="+1 for meaningful easing, -1 tighten")
    ap.add_argument("--safety_shift", type=float, default=0.0)
    ap.add_argument("--price_pct", type=float, default=0.0, help="Hotel price index % change")
    ap.add_argument("--global_demand_pct", type=float, default=0.0)
    ap.add_argument("--restrictions_shift", type=float, default=0.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_tourism")
    ap.add_argument("--min_obs", type=int, default=48)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    ARR, ARR_BY = load_arrivals(args.arrivals, args.country)
    SPD = load_spend(args.spend, args.country)      if args.spend     else pd.DataFrame()
    FX  = load_fx(args.fx, args.country)            if args.fx        else pd.DataFrame()
    AIR = load_air(args.air, args.country)          if args.air       else pd.DataFrame()
    VIS = load_policy(args.visa, args.country)      if args.visa      else pd.DataFrame()
    SAF = load_safety(args.safety, args.country)    if args.safety    else pd.DataFrame()
    PRC = load_prices(args.prices, args.country)    if args.prices    else pd.DataFrame()
    GBL = load_global(args.global_idx)              if args.global_idx else pd.DataFrame()
    RST = load_restrict(args.restrict, args.country)if args.restrict  else pd.DataFrame()
    WEA = load_weather(args.weather, args.country)  if args.weather   else pd.DataFrame()

    # time filters
    if args.start:
        s = eom(pd.Series([args.start])).iloc[0]
        for df in [ARR, ARR_BY, SPD, FX, AIR, VIS, SAF, PRC, GBL, RST, WEA]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = eom(pd.Series([args.end])).iloc[0]
        for df in [ARR, ARR_BY, SPD, FX, AIR, VIS, SAF, PRC, GBL, RST, WEA]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] > e].index, inplace=True)

    # panel
    P = build_panel(args.country, ARR, ARR_BY, SPD, FX, AIR, VIS, SAF, PRC, GBL, RST, WEA)
    if P.empty:
        raise ValueError("Panel empty after filtering/merges. Check inputs.")
    P.to_csv(outdir / "panel.csv", index=False)

    # elasticities
    ELA = regress_elasticities_arrivals(P, L=int(args.lags), min_obs=int(args.min_obs))
    if not ELA.empty: ELA.to_csv(outdir / "elasticity_arrivals.csv", index=False)
    ELS = regress_elasticities_spend(P, L=int(args.lags), min_obs=int(args.min_obs))
    if not ELS.empty: ELS.to_csv(outdir / "elasticity_spend.csv", index=False)

    # events
    ES = event_study(P, VIS, AIR, SAF, window=int(args.event_window),
                     z_window=int(args.z_window), z_thr=float(args.z_threshold))
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # scenarios
    SCN = run_scenarios(P, ELA if not ELA.empty else pd.DataFrame(),
                        ELS if not ELS.empty else pd.DataFrame(),
                        fx_pct=float(args.fx_pct), reer_pct=float(args.reer_pct),
                        seats_pct=float(args.seats_pct), visa_shift=float(args.visa_shift),
                        safety_shift=float(args.safety_shift), price_pct=float(args.price_pct),
                        global_demand_pct=float(args.global_demand_pct),
                        restrictions_shift=float(args.restrictions_shift),
                        horizon=int(args.horizon))
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # stress
    ST = stress_var_es(P, n_sims=int(args.n_sims))
    if not ST.empty: ST.to_csv(outdir / "stress_vares.csv", index=False)

    # summary
    summary = {
        "country": args.country,
        "sample": {
            "start": str(P["date"].min().date()),
            "end": str(P["date"].max().date()),
            "months": int(P["date"].nunique())
        },
        "n_obs": int(P.shape[0]),
        "has_origin_split": bool(not ARR_BY.empty),
        "has_fx": bool("usd_local" in P.columns or "reer_index" in P.columns),
        "has_air": bool("seats" in P.columns or "flights" in P.columns),
        "has_prices": bool("hotel_price_idx" in P.columns),
        "has_spend": bool("spend_usd" in P.columns or "spend_per_visitor_usd" in P.columns),
        "outputs": {
            "panel": "panel.csv",
            "elasticity_arrivals": "elasticity_arrivals.csv" if not ELA.empty else None,
            "elasticity_spend": "elasticity_spend.csv" if not ELS.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "scenarios": "scenarios.csv" if not SCN.empty else None,
            "stress_vares": "stress_vares.csv" if not ST.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # config echo
    cfg = asdict(Config(
        arrivals=args.arrivals, spend=(args.spend or None), fx=(args.fx or None), air=(args.air or None),
        visa=(args.visa or None), safety=(args.safety or None), prices=(args.prices or None),
        global_idx=(args.global_idx or None), restrict=(args.restrict or None), weather=(args.weather or None),
        country=args.country, lags=int(args.lags), horizon=int(args.horizon), n_sims=int(args.n_sims),
        z_window=int(args.z_window), z_threshold=float(args.z_threshold), event_window=int(args.event_window),
        fx_pct=float(args.fx_pct), reer_pct=float(args.reer_pct), seats_pct=float(args.seats_pct),
        visa_shift=float(args.visa_shift), safety_shift=float(args.safety_shift), price_pct=float(args.price_pct),
        global_demand_pct=float(args.global_demand_pct), restrictions_shift=float(args.restrictions_shift),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # console
    print("== Inbound Tourism Toolkit ==")
    print(f"Country: {args.country} | Sample: {summary['sample']['start']} → {summary['sample']['end']} | Obs: {summary['n_obs']}")
    if summary["outputs"]["elasticity_arrivals"]: print("Arrivals elasticities estimated.")
    if summary["outputs"]["elasticity_spend"]: print("Spend-per-visitor elasticities estimated.")
    if summary["outputs"]["event_study"]: print(f"Event study written (±{args.event_window} months).")
    if summary["outputs"]["scenarios"]: print(f"Scenarios projected ({args.horizon} months).")
    if summary["outputs"]["stress_vares"]: print(f"Stress VaR/ES done (n={args.n_sims}).")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
