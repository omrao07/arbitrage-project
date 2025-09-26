#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rural_credit_cycle.py — Rural credit cycle: growth, stress index, lead–lag & scenarios
--------------------------------------------------------------------------------------

What this does
==============
Given bank/segment/regional datasets for **rural India**, this script:

1) Cleans & aligns data to **monthly** or **quarterly** frequency (end-of-period dating).
2) Builds key transforms:
   • Δlog (m/m) & YoY credit growth, disbursements, deposits
   • NPA/PAR deltas, credit intensity (credit per rural activity proxy)
   • Macro/agri drivers: rainfall deviation, food CPI YoY, rural wages YoY, mandi price YoY, MGNREGA demand growth
3) Diagnostics:
   • Rolling correlations and lead–lag tables between credit growth and drivers
   • Distributed-lag elasticities with HAC (Newey–West) SEs:
       Δlog(Credit)_t ~ Σ β·drivers_{t−i}  (and a separate model for ΔNPA)
4) **Rural Stress Index (RSI)**:
   • Z-score blend of distress signals (signed): −rainfall_dev, +food CPI YoY, +MGNREGA growth,
     −wage YoY, −mandi price YoY, +NPA/ +PAR30; computed aggregate & by region (if available)
   • Early-warning flags when RSI exceeds thresholds / rises rapidly
5) Segment & region contributions (if provided):
   • Contribution of each segment/region to aggregate Δlog credit
6) Scenarios:
   • Monsoon deficit (e.g., rainfall −20% for 3m)
   • Food inflation +200bp YoY for 6m
   • Wage growth −5pp YoY for 6m
   → Uses cumulated regression betas to approximate impact on credit growth and NPA

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--credit credit.csv          REQUIRED
  Columns (any subset):
    date,
    [region|state|district], [segment],
    credit_outstanding_inr[, disbursements_inr],
    npa_pct[, par30_pct],
    [rural_flag]  (optional boolean/yes/no)

--deposits deposits.csv      OPTIONAL  (aggregate or region-level)
  Columns: date[, region] deposits_inr

--rain rainfall.csv          OPTIONAL
  Columns (any subset):
    date[, region], rainfall_mm[, normal_mm][, rainfall_dev_pct]
  (If normal_mm is present, rainfall_dev_pct = (rainfall_mm - normal_mm)/normal_mm.)

--macro macro.csv            OPTIONAL
  Columns (any subset, monthly):
    date,
    cpi_rural[, cpi_food], rural_wage_idx[, wage_avg],
    mandi_price_idx[, mandi_price],
    mgnrega_persondays[, mgnrega_households],
    tractor_sales[, two_wheeler_sales],
    rural_unemployment[, fmcg_rural_index]

--rates rates.csv            OPTIONAL
  Columns: date, repo_rate[, deposit_rate, lending_rate, gsec_10y]

--events events.csv          OPTIONAL
  Columns: date, label

Key CLI
-------
--freq monthly|quarterly     Default monthly
--lags 6                     Max lag for lead–lag & regressions (periods)
--windows 3,6,12             Rolling corr windows (periods)
--rsi_thresh 1.5             Z-score trigger for EWS
--start / --end              Date filters (YYYY-MM-DD)
--outdir out_rural           Output directory
--min_obs 24                 Minimum overlapping periods for regressions

Outputs
-------
- panel.csv                  Aggregate master panel (levels & transforms)
- region_panel.csv           Region-level panel (if region provided)
- segment_contrib.csv        Segment contribution to Δlog credit (if segment provided)
- rolling_stats.csv          Rolling corr of Δlog credit vs drivers
- leadlag_corr.csv           Corr(X_{t−k}, Δlog credit_t)
- dlag_reg_credit.csv        D-lag elasticities for credit growth (coef, HAC se, t)
- dlag_reg_npa.csv           D-lag elasticities for ΔNPA (coef, HAC se, t)
- rsi.csv                    Rural Stress Index (aggregate & by region if available)
- ews.csv                    Early-warning signals (dates, RSI, components)
- scenarios.csv              Shock scenarios → growth/NPA impact (pp)
- summary.json               Headline diagnostics
- config.json                Run configuration

DISCLAIMER
----------
This is research tooling. It uses simplified heuristics for aggregation and standardization,
and assumes monthly data unless specified. Validate with your definitions and robustness checks
before operational or investment decisions.
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
    if freq.startswith("Q"):
        return dt.dt.to_period("Q").dt.to_timestamp("Q")
    return dt.dt.to_period("M").dt.to_timestamp("M")

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def yoy(s: pd.Series, periods: int) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s) - np.log(s.shift(periods))

def d(s: pd.Series) -> pd.Series:
    return s.astype(float).diff()

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

def load_credit(path: str, freq: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], bool, bool]:
    """
    Returns:
      - aggregate panel (by date)
      - region-level panel (if region present)
      - has_region, has_segment flags
    """
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    reg = ncol(df, "region","state","district")
    seg = ncol(df, "segment","product","loan_type")
    cred = ncol(df, "credit_outstanding_inr","credit","loans","advances","net_advances","gross_advances","outstanding_inr")
    disb = ncol(df, "disbursements_inr","disbursed_inr","disb","flow_inr")
    npa  = ncol(df, "npa_pct","gnpa_pct","par90_pct","npa")
    par  = ncol(df, "par30_pct","par_30","par30")
    rur  = ncol(df, "rural_flag","is_rural")
    if not (dt and cred):
        raise ValueError("credit.csv must include date and credit_outstanding_inr (or equivalent).")
    df = df.rename(columns={dt:"date", cred:"credit"})
    df["date"] = to_period_end(df["date"], freq)
    df["credit"] = safe_num(df["credit"])
    if disb: df = df.rename(columns={disb:"disbursements"}); df["disbursements"] = safe_num(df["disbursements"])
    if npa:  df = df.rename(columns={npa:"npa_pct"}); df["npa_pct"] = safe_num(df["npa_pct"])
    if par:  df = df.rename(columns={par:"par30_pct"}); df["par30_pct"] = safe_num(df["par30_pct"])
    if reg:  df = df.rename(columns={reg:"region"})
    if seg:  df = df.rename(columns={seg:"segment"})
    if rur:  df = df.rename(columns={rur:"rural_flag"}); df["rural_flag"] = df["rural_flag"].astype(str).str.lower().isin(["y","yes","true","1","rural"])
    has_region = reg is not None
    has_segment = seg is not None

    # Region panel if present
    region_panel = None
    if has_region:
        region_panel = df.groupby(["date","region"], as_index=False).agg({
            "credit":"sum",
            "disbursements":"sum" if "disbursements" in df.columns else "first",
            "npa_pct":"mean" if "npa_pct" in df.columns else "first",
            "par30_pct":"mean" if "par30_pct" in df.columns else "first"
        })

    # Aggregate across regions/segments
    agg = df.groupby("date", as_index=False).agg({
        "credit":"sum",
        "disbursements":"sum" if "disbursements" in df.columns else "first",
        "npa_pct":"mean" if "npa_pct" in df.columns else "first",
        "par30_pct":"mean" if "par30_pct" in df.columns else "first"
    })

    # Segment contributions need the original df with 'segment'
    df["_has_segment"] = has_segment
    return agg.sort_values("date"), (region_panel.sort_values(["date","region"]) if region_panel is not None else None), has_region, has_segment

def load_deposits(path: Optional[str], freq: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not path: return pd.DataFrame(), None
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    reg = ncol(df, "region","state","district")
    dep = ncol(df, "deposits_inr","deposit_outstanding_inr","deposits")
    if not (dt and dep): raise ValueError("deposits.csv needs date and deposits.")
    df = df.rename(columns={dt:"date", dep:"deposits"})
    df["date"] = to_period_end(df["date"], freq); df["deposits"] = safe_num(df["deposits"])
    if reg: df = df.rename(columns={reg:"region"})
    agg = df.groupby("date", as_index=False)["deposits"].sum()
    regp = df.groupby(["date","region"], as_index=False)["deposits"].sum() if reg else None
    return agg, regp

def load_rain(path: Optional[str], freq: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not path: return pd.DataFrame(), None
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    reg = ncol(df, "region","state","district")
    rmm = ncol(df, "rainfall_mm","rain_mm","rainfall")
    nor = ncol(df, "normal_mm","normal","avg_mm")
    dev = ncol(df, "rainfall_dev_pct","rain_dev","rainfall_deviation_pct")
    if not dt: raise ValueError("rainfall.csv needs date.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], freq)
    if reg: df = df.rename(columns={reg:"region"})
    if rmm: df[rmm] = safe_num(df[rmm])
    if nor: df[nor] = safe_num(df[nor])
    if dev: df = df.rename(columns={dev:"rainfall_dev_pct"}); df["rainfall_dev_pct"] = safe_num(df["rainfall_dev_pct"])
    if rmm and nor and "rainfall_dev_pct" not in df.columns:
        df["rainfall_dev_pct"] = (df[rmm] - df[nor]) / df[nor].replace(0, np.nan) * 100.0
    keep = ["date","rainfall_dev_pct"] + (["region"] if reg else [])
    agg = df.groupby("date", as_index=False)["rainfall_dev_pct"].mean()
    regp = df[keep].dropna() if reg else None
    return agg, regp

def load_macro(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], freq)
    ren = {}
    for pair in [
        ("cpi_rural","cpi_rural"), ("cpi_food","cpi_food"),
        ("rural_wage_idx","rural_wage_idx"), ("wage_avg","wage_avg"),
        ("mandi_price_idx","mandi_price_idx"), ("mandi_price","mandi_price"),
        ("mgnrega_persondays","mgnrega_persondays"), ("mgnrega_households","mgnrega_households"),
        ("tractor_sales","tractor_sales"), ("two_wheeler_sales","two_wheeler_sales"),
        ("rural_unemployment","rural_unemployment"), ("fmcg_rural_index","fmcg_rural_index"),
    ]:
        c = ncol(df, pair[0])
        if c: ren[c] = pair[1]
    df = df.rename(columns=ren)
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    return df.sort_values("date")

def load_rates(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"}); df["date"] = to_period_end(df["date"], freq)
    ren = {}
    for pair in [("repo_rate","repo_rate"), ("deposit_rate","deposit_rate"), ("lending_rate","lending_rate"), ("gsec_10y","gsec_10y")]:
        c = ncol(df, pair[0])
        if c: ren[c] = pair[1]
    df = df.rename(columns=ren)
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    return df.sort_values("date")[["date"] + [c for c in ["repo_rate","deposit_rate","lending_rate","gsec_10y"] if c in df.columns]]

def load_events(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    lab = ncol(df, "label","event","name") or "label"
    df = df.rename(columns={dt:"date", lab:"label"})
    df["date"] = to_period_end(df["date"], freq)
    df["label"] = df["label"].astype(str)
    return df[["date","label"]].dropna()

# ----------------------------- constructions -----------------------------

def build_aggregate_panel(freq: str,
                          CREDIT: pd.DataFrame,
                          DEPOS: pd.DataFrame,
                          RAIN: pd.DataFrame,
                          MACRO: pd.DataFrame,
                          RATES: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    df = CREDIT.copy()
    for d in [DEPOS, RAIN, MACRO, RATES]:
        if d is not None and not d.empty:
            df = df.merge(d, on="date", how="left")
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    yo = 12 if freq.startswith("M") else 4

    # Core transforms
    df["dlog_credit"] = dlog(df["credit"])
    df["yoy_credit"]  = yoy(df["credit"], yo)
    if "disbursements" in df.columns:
        df["dlog_disb"] = dlog(df["disbursements"])
    if "deposits" in df.columns:
        df["dlog_deposits"] = dlog(df["deposits"])
    if "npa_pct" in df.columns:
        df["d_npa_pct"] = d(df["npa_pct"])
    if "par30_pct" in df.columns:
        df["d_par30_pct"] = d(df["par30_pct"])

    # Rainfall deviation: ensure decimal form too
    if "rainfall_dev_pct" in df.columns:
        df["rainfall_dev_dec"] = df["rainfall_dev_pct"] / 100.0

    # Price & wage inflation (YoY)
    if "cpi_food" in df.columns: df["yoy_cpi_food"] = yoy(df["cpi_food"], yo)
    if "cpi_rural" in df.columns: df["yoy_cpi_rural"] = yoy(df["cpi_rural"], yo)
    # Wage growth
    if "rural_wage_idx" in df.columns: df["yoy_wage"] = yoy(df["rural_wage_idx"], yo)
    if "wage_avg" in df.columns and "yoy_wage" not in df.columns: df["yoy_wage"] = yoy(df["wage_avg"], yo)
    # Mandi price YoY
    if "mandi_price_idx" in df.columns: df["yoy_mandi"] = yoy(df["mandi_price_idx"], yo)
    if "mandi_price" in df.columns and "yoy_mandi" not in df.columns: df["yoy_mandi"] = yoy(df["mandi_price"], yo)
    # MGNREGA growth (proxy of stress when ↑)
    if "mgnrega_persondays" in df.columns: df["dlog_mgnrega"] = dlog(df["mgnrega_persondays"])
    if "mgnrega_households" in df.columns and "dlog_mgnrega" not in df.columns: df["dlog_mgnrega"] = dlog(df["mgnrega_households"])
    # Other proxies (optionally used in rolling only)
    for c in ["tractor_sales","two_wheeler_sales","fmcg_rural_index","rural_unemployment"]:
        if c in df.columns:
            if c == "rural_unemployment":
                df["d_"+c] = d(df[c])
            else:
                df["dlog_"+c] = dlog(df[c])

    # Rate deltas
    for c in ["repo_rate","deposit_rate","lending_rate","gsec_10y"]:
        if c in df.columns:
            df[f"d_{c}"] = d(df[c])

    # Driver mapping
    mapping = {
        "rain": "rainfall_dev_dec" if "rainfall_dev_dec" in df.columns else None,
        "food_inf": "yoy_cpi_food" if "yoy_cpi_food" in df.columns else None,
        "wage": "yoy_wage" if "yoy_wage" in df.columns else None,
        "mandi": "yoy_mandi" if "yoy_mandi" in df.columns else None,
        "mgnrega": "dlog_mgnrega" if "dlog_mgnrega" in df.columns else None,
        "deposits": "dlog_deposits" if "dlog_deposits" in df.columns else None,
        "repo": "d_repo_rate" if "d_repo_rate" in df.columns else None
    }
    return df, mapping

def build_region_panel(REG_CREDIT: pd.DataFrame,
                       REG_DEPOS: Optional[pd.DataFrame],
                       REG_RAIN: Optional[pd.DataFrame],
                       freq: str) -> pd.DataFrame:
    if REG_CREDIT is None or REG_CREDIT.empty: return pd.DataFrame()
    df = REG_CREDIT.copy()
    for d in [REG_DEPOS, REG_RAIN]:
        if d is not None and not d.empty:
            df = df.merge(d, on=["date","region"], how="left")
    df = df.sort_values(["date","region"])
    yo = 12 if freq.startswith("M") else 4
    df["dlog_credit"] = df.groupby("region")["credit"].apply(dlog).reset_index(level=0, drop=True)
    df["yoy_credit"]  = df.groupby("region")["credit"].apply(lambda s: yoy(s, yo)).reset_index(level=0, drop=True)
    if "npa_pct" in df.columns:
        df["d_npa_pct"] = df.groupby("region")["npa_pct"].diff()
    if "par30_pct" in df.columns:
        df["d_par30_pct"] = df.groupby("region")["par30_pct"].diff()
    if "rainfall_dev_pct" in df.columns:
        df["rainfall_dev_dec"] = df["rainfall_dev_pct"] / 100.0
    if "deposits" in df.columns:
        df["dlog_deposits"] = df.groupby("region")["deposits"].apply(dlog).reset_index(level=0, drop=True)
    return df

# ----------------------------- contributions -----------------------------

def segment_contributions(raw_credit_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    contrib_{seg,t} = weight_{seg,t-1} * Δlog(credit_{seg,t})
    Requires 'segment' column in raw credit file.
    """
    if raw_credit_df is None or raw_credit_df.empty or "segment" not in raw_credit_df.columns:
        return pd.DataFrame()
    df = raw_credit_df.copy()
    if "date" not in df.columns:
        dt = ncol(df, "date") or df.columns[0]; df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], "M" if freq.startswith("M") else "Q")
    cred = ncol(df, "credit_outstanding_inr","credit","loans","advances","outstanding_inr")
    if not cred: return pd.DataFrame()
    df = df.rename(columns={cred:"credit"})
    df["credit"] = safe_num(df["credit"])
    df = df.groupby(["date","segment"], as_index=False)["credit"].sum().sort_values(["segment","date"])
    df["dlog_seg"] = df.groupby("segment")["credit"].apply(dlog).reset_index(level=0, drop=True)
    tot = df.groupby("date", as_index=False)["credit"].sum().rename(columns={"credit":"total"})
    df = df.merge(tot, on="date", how="left")
    df["w_lag"] = df.groupby("segment")["credit"].shift(1) / df["total"].shift(1)
    df["contrib"] = df["w_lag"] * df["dlog_seg"]
    return df.dropna(subset=["contrib"])[["date","segment","contrib","w_lag","dlog_seg","credit"]].sort_values(["date","segment"])

# ----------------------------- analytics -----------------------------

def rolling_stats(panel: pd.DataFrame, mapping: Dict[str,str], windows: List[int]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    y = idx.get("dlog_credit")
    if y is None: return pd.DataFrame()
    for key in ["rain","food_inf","wage","mandi","mgnrega","deposits","repo"]:
        col = mapping.get(key)
        if not col or col not in idx.columns: continue
        x = idx[col]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({"driver": key, "column": col, "window": w, "tag": tag,
                         "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan})
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, mapping: Dict[str,str], lags: int) -> pd.DataFrame:
    rows = []
    y = panel.get("dlog_credit")
    if y is None or y.dropna().empty: return pd.DataFrame()
    for key in ["rain","food_inf","wage","mandi","mgnrega","deposits","repo"]:
        col = mapping.get(key)
        if not col or col not in panel.columns: continue
        tab = leadlag_corr(panel[col], y, lags)
        tab["driver"] = key
        tab["column"] = col
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","driver","column"])

def dlag_regression(panel: pd.DataFrame, dep_col: str, mapping: Dict[str,str], L: int, min_obs: int) -> pd.DataFrame:
    """
    Generic distributed-lag regression with HAC SEs.
    dep_col: 'dlog_credit' or 'd_npa_pct'
    """
    if dep_col not in panel.columns: return pd.DataFrame()
    df = panel.copy()
    dep = dep_col
    Xparts = [pd.Series(1.0, index=df.index, name="const")]
    names = ["const"]
    use_keys = ["rain","food_inf","wage","mandi","mgnrega","deposits","repo"]
    avail = []
    for key in use_keys:
        col = mapping.get(key)
        if not col or col not in df.columns: continue
        avail.append((key, col))
        for l in range(0, L+1):
            nm = f"{col}_l{l}"
            Xparts.append(df[col].shift(l).rename(nm))
            names.append(nm)
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
    # cumulative effects 0..L for each driver
    for key, col in avail:
        idxs = [i for i, nm in enumerate(names) if nm.startswith(f"{col}_l")]
        if idxs:
            bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
            out.append({"var": f"{col}_cum_0..L", "coef": bsum, "se": ses,
                        "t_stat": bsum/(ses if ses>0 else np.nan), "lags": L, "dep": dep})
    return pd.DataFrame(out)

# ----------------------------- Rural Stress Index (RSI) -----------------------------

def zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    return (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) not in [0, np.nan] else np.nan)

def compute_rsi(panel: pd.DataFrame, region_panel: Optional[pd.DataFrame], thresh: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    RSI = mean of standardized distress components (sign-adjusted):
      components:
        rain_s = -rainfall_dev_dec
        food_s = +yoy_cpi_food
        mgnr_s = +dlog_mgnrega
        wage_s = -yoy_wage
        mandi_s= -yoy_mandi
        npa_s  = +npa_pct (or par30_pct if npa missing)
    Returns rsi_df (aggregate) and ews flags.
    """
    def rsi_from_df(df: pd.DataFrame, by_region: bool=False) -> pd.DataFrame:
        cols = {}
        if "rainfall_dev_dec" in df.columns: cols["rain_s"] = -df["rainfall_dev_dec"]
        if "yoy_cpi_food"   in df.columns: cols["food_s"] = df["yoy_cpi_food"]
        if "dlog_mgnrega"   in df.columns: cols["mgnr_s"] = df["dlog_mgnrega"]
        if "yoy_wage"       in df.columns: cols["wage_s"] = -df["yoy_wage"]
        if "yoy_mandi"      in df.columns: cols["mandi_s"] = -df["yoy_mandi"]
        if "npa_pct"        in df.columns: cols["npa_s"]  = df["npa_pct"]
        elif "par30_pct"    in df.columns: cols["npa_s"]  = df["par30_pct"]
        if not cols: 
            return pd.DataFrame()
        F = pd.DataFrame(cols).copy()
        # Standardize across time (per region if applicable)
        if by_region:
            Z = F.groupby(df["region"]).transform(zscore)
            out = pd.DataFrame({"date": df["date"], "region": df["region"]})
        else:
            Z = F.apply(zscore, axis=0)
            out = pd.DataFrame({"date": df["date"]})
        out["RSI"] = Z.mean(axis=1, skipna=True)
        # keep component z-scores for diagnostics
        for c in Z.columns:
            out[c] = Z[c]
        # aggregate duplicates (if any) by mean
        grp = ["date","region"] if by_region else ["date"]
        return out.groupby(grp, as_index=False).mean()

    agg_rsi = rsi_from_df(panel, by_region=False)
    # Early warning flags (aggregate)
    ews_rows = []
    if not agg_rsi.empty:
        agg_rsi = agg_rsi.sort_values("date")
        # rate of change
        agg_rsi["d_RSI"] = agg_rsi["RSI"].diff()
        # percentile thresholds
        P90 = np.nanpercentile(agg_rsi["RSI"], 90) if agg_rsi["RSI"].notna().sum()>=10 else np.nan
        for i, r in agg_rsi.iterrows():
            cond1 = (pd.notna(r["RSI"]) and r["RSI"] >= thresh)
            cond2 = (pd.notna(r["RSI"]) and pd.notna(P90) and r["RSI"] >= P90)
            cond3 = (pd.notna(r["d_RSI"]) and r["d_RSI"] > 0.5)
            if cond1 or cond2 or cond3:
                # find top 2 contributing components by absolute z
                comps = {k: abs(r.get(k, np.nan)) for k in ["rain_s","food_s","mgnr_s","wage_s","mandi_s","npa_s"] if k in agg_rsi.columns}
                top = sorted([(k,v) for k,v in comps.items() if pd.notna(v)], key=lambda x: x[1], reverse=True)[:2]
                ews_rows.append({"date": r["date"], "RSI": r["RSI"], "trigger_thresh": cond1, "trigger_p90": cond2,
                                 "trigger_momentum": cond3,
                                 "top1": (top[0][0] if len(top)>0 else None), "top2": (top[1][0] if len(top)>1 else None)})
    reg_rsi = pd.DataFrame()
    if region_panel is not None and not region_panel.empty:
        reg_rsi = rsi_from_df(region_panel, by_region=True).sort_values(["date","region"])
    ews = pd.DataFrame(ews_rows).sort_values("date")
    return (pd.concat([agg_rsi, reg_rsi], ignore_index=True, sort=False) if not agg_rsi.empty or not reg_rsi.empty else pd.DataFrame(),
            ews)

# ----------------------------- Scenarios -----------------------------

def extract_cum_coef(reg: pd.DataFrame, label_prefix: str) -> Optional[float]:
    if reg.empty: return None
    r = reg[reg["var"] == f"{label_prefix}_cum_0..L"]
    return float(r["coef"].iloc[0]) if not r.empty else None

def scenarios(reg_credit: pd.DataFrame, reg_npa: pd.DataFrame) -> pd.DataFrame:
    """
    Map shocks to Δlog credit (pp) and ΔNPA (pp), using cumulative betas when available.
      • Monsoon deficit: rainfall_dev_dec = −0.20 for 3 months → impact ≈ β_rain * (−0.20)
      • Food inflation +200bp YoY for 6 months → β_food * 0.02
      • Wage growth −5pp YoY for 6 months → β_wage * (−0.05)
    Notes:
      - Betas are per-period (monthly) effects; we report **monthly** growth impacts in percentage points (×100).
      - ΔNPA impacts are in percentage points as-is.
    """
    rows = []
    b_rain_c = extract_cum_coef(reg_credit, "rainfall_dev_dec")
    b_food_c = extract_cum_coef(reg_credit, "yoy_cpi_food")
    b_wage_c = extract_cum_coef(reg_credit, "yoy_wage")
    b_rain_n = extract_cum_coef(reg_npa, "rainfall_dev_dec") if reg_npa is not None else None
    b_food_n = extract_cum_coef(reg_npa, "yoy_cpi_food") if reg_npa is not None else None
    b_wage_n = extract_cum_coef(reg_npa, "yoy_wage") if reg_npa is not None else None

    # Monsoon deficit
    if b_rain_c is not None:
        shock = -0.20  # −20% vs normal
        rows.append({"scenario":"Monsoon −20% for 3m", "var":"credit_growth_pp", "impact": b_rain_c * shock * 100.0})
    if b_rain_n is not None:
        rows.append({"scenario":"Monsoon −20% for 3m", "var":"npa_delta_pp", "impact": b_rain_n * (-0.20)})

    # Food inflation +200bp YoY
    if b_food_c is not None:
        rows.append({"scenario":"Food CPI +200bp for 6m", "var":"credit_growth_pp", "impact": b_food_c * 0.02 * 100.0})
    if b_food_n is not None:
        rows.append({"scenario":"Food CPI +200bp for 6m", "var":"npa_delta_pp", "impact": b_food_n * 0.02})

    # Wage growth −5pp
    if b_wage_c is not None:
        rows.append({"scenario":"Wage growth −5pp for 6m", "var":"credit_growth_pp", "impact": b_wage_c * (-0.05) * 100.0})
    if b_wage_n is not None:
        rows.append({"scenario":"Wage growth −5pp for 6m", "var":"npa_delta_pp", "impact": b_wage_n * (-0.05)})

    return pd.DataFrame(rows)

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    credit: str
    deposits: Optional[str]
    rainfall: Optional[str]
    macro: Optional[str]
    rates: Optional[str]
    events: Optional[str]
    freq: str
    lags: int
    windows: str
    rsi_thresh: float
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rural credit cycle: growth, stress index, lead–lag & scenarios")
    ap.add_argument("--credit", required=True)
    ap.add_argument("--deposits", default="")
    ap.add_argument("--rainfall", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--rates", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","quarterly"])
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--rsi_thresh", type=float, default=1.5)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_rural")
    ap.add_argument("--min_obs", type=int, default=24)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq = "M" if args.freq.startswith("m") else "Q"

    CREDIT_AGG, REG_CREDIT, has_region, has_segment = load_credit(args.credit, freq=freq)
    DEPOS_AGG, REG_DEPOS = load_deposits(args.deposits, freq=freq) if args.deposits else (pd.DataFrame(), None)
    RAIN_AGG, REG_RAIN   = load_rain(args.rainfall, freq=freq) if args.rainfall else (pd.DataFrame(), None)
    MACRO = load_macro(args.macro, freq=freq) if args.macro else pd.DataFrame()
    RATES = load_rates(args.rates, freq=freq) if args.rates else pd.DataFrame()
    EVENTS= load_events(args.events, freq=freq) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        t0 = pd.to_datetime(args.start)
        for df in [CREDIT_AGG, REG_CREDIT, DEPOS_AGG, REG_DEPOS, RAIN_AGG, REG_RAIN, MACRO, RATES]:
            if df is not None and not df.empty:
                df.drop(df[df["date"] < t0].index, inplace=True)
    if args.end:
        t1 = pd.to_datetime(args.end)
        for df in [CREDIT_AGG, REG_CREDIT, DEPOS_AGG, REG_DEPOS, RAIN_AGG, REG_RAIN, MACRO, RATES]:
            if df is not None and not df.empty:
                df.drop(df[df["date"] > t1].index, inplace=True)

    # Aggregate panel
    PANEL, mapping = build_aggregate_panel(freq, CREDIT_AGG, DEPOS_AGG, RAIN_AGG, MACRO, RATES)
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Region panel (if available)
    REGION_PANEL = build_region_panel(REG_CREDIT, REG_DEPOS, REG_RAIN, freq=freq) if has_region else pd.DataFrame()
    if not REGION_PANEL.empty:
        REGION_PANEL.to_csv(outdir / "region_panel.csv", index=False)

    # Segment contributions (if segment present)
    SEG_CONTRIB = pd.DataFrame()
    if has_segment:
        raw_credit_df = pd.read_csv(args.credit)
        SEG_CONTRIB = segment_contributions(raw_credit_df, freq=freq)
        if not SEG_CONTRIB.empty:
            SEG_CONTRIB.to_csv(outdir / "segment_contrib.csv", index=False)

    # Rolling & lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, mapping, windows)
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)
    LL = leadlag_tables(PANEL, mapping, lags=int(args.lags))
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # D-lag regressions
    REG_CREDIT = dlag_regression(PANEL, dep_col="dlog_credit", mapping=mapping, L=int(args.lags), min_obs=int(args.min_obs))
    if not REG_CREDIT.empty: REG_CREDIT.to_csv(outdir / "dlag_reg_credit.csv", index=False)
    REG_NPA = dlag_regression(PANEL, dep_col="d_npa_pct", mapping=mapping, L=int(args.lags), min_obs=int(args.min_obs)) if "d_npa_pct" in PANEL.columns else pd.DataFrame()
    if not REG_NPA.empty: REG_NPA.to_csv(outdir / "dlag_reg_npa.csv", index=False)

    # RSI & EWS
    RSI, EWS = compute_rsi(PANEL, REGION_PANEL if not REGION_PANEL.empty else None, thresh=float(args.rsi_thresh))
    if not RSI.empty: RSI.to_csv(outdir / "rsi.csv", index=False)
    if not EWS.empty: EWS.to_csv(outdir / "ews.csv", index=False)

    # Scenarios
    SCN = scenarios(REG_CREDIT if not REG_CREDIT.empty else pd.DataFrame(),
                    REG_NPA if not REG_NPA.empty else pd.DataFrame())
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    latest = PANEL.tail(1).iloc[0]
    best_ll = {}
    if not LL.empty:
        for drv, g in LL.dropna(subset=["corr"]).groupby("driver"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[drv] = {"lag": int(row["lag"]), "corr": float(row["corr"]), "column": row["column"]}
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "freq": args.freq,
        "latest": {
            "date": str(latest["date"].date()),
            "credit_inr": float(latest.get("credit", np.nan)),
            "yoy_credit": float(latest.get("yoy_credit", np.nan)) if pd.notna(latest.get("yoy_credit", np.nan)) else None,
            "npa_pct": float(latest.get("npa_pct", np.nan)) if "npa_pct" in PANEL.columns and pd.notna(latest.get("npa_pct", np.nan)) else None,
            "RSI": float(RSI[RSI["date"]==latest["date"]]["RSI"].iloc[0]) if not RSI.empty and (RSI["date"]==latest["date"]).any() else None
        },
        "rolling_windows": windows,
        "leadlag_best": best_ll,
        "reg_credit_terms": REG_CREDIT["var"].tolist() if not REG_CREDIT.empty else [],
        "reg_npa_terms": REG_NPA["var"].tolist() if not REG_NPA.empty else [],
        "has_region": has_region,
        "has_segment": has_segment,
        "ews_latest": (EWS.tail(1).to_dict(orient="records")[0] if not EWS.empty else None)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        credit=args.credit, deposits=(args.deposits or None), rainfall=(args.rainfall or None),
        macro=(args.macro or None), rates=(args.rates or None), events=(args.events or None),
        freq=args.freq, lags=int(args.lags), windows=args.windows, rsi_thresh=float(args.rsi_thresh),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Rural Credit Cycle ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Freq: {summary['freq']}")
    if summary["latest"]["yoy_credit"] is not None:
        print(f"Latest YoY credit: {summary['latest']['yoy_credit']*100:.2f}%")
    if summary["latest"]["RSI"] is not None:
        print(f"Latest RSI: {summary['latest']['RSI']:+.2f} (>{args.rsi_thresh} indicates stress)")
    if best_ll:
        for k, st in best_ll.items():
            print(f"Lead–lag {k}: max |corr| at lag {st['lag']:+d} → {st['corr']:+.2f}")
    if not SCN.empty:
        for _, r in SCN.iterrows():
            print(f"Scenario {r['scenario']} → {r['var']}: {r['impact']:+.2f}")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
