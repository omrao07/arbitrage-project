#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psu_bank_lending.py — India PSU bank lending: growth, mix, rate sensitivity & scenarios
---------------------------------------------------------------------------------------

What this does
==============
Given bank/sector/monthly datasets, this script focuses on **Public Sector Banks (PSBs/PSUs)** and:

1) Cleans & aligns to **monthly** (or **quarterly**) frequency.
2) Builds transforms:
   • Δlog (m/m) and YoY growth for credit & deposits
   • Credit/Deposit (CD) ratio, rate spreads (lending − deposit), real policy rate (repo − CPI)
3) Diagnostics:
   • Rolling correlations (short/med/long) of PSU credit growth vs drivers (policy rate, deposits, yields, macro)
   • Lead–lag tables Corr(X_{t−k}, Δlog Credit_t)
4) Distributed-lag elasticities (Newey–West HAC SEs):
   Δlog(Credit)_t ~ Σ β_repo·ΔRepo_{t−i}
                     + Σ β_depR·ΔDepositRate_{t−i}
                     + Σ β_depG·Δlog(Deposits)_{t−i}
                     + Σ β_y10·Δ10Y_{t−i}
                     + Σ β_macro·Δlog(Macro)_t−i
5) Growth **contributions**:
   • By **bank** (if bank-level credit is provided)
   • By **sector** (if sectoral deployment is provided)
6) Scenarios:
   • Repo +50bp / +100bp
   • Avg deposit rate +50bp
   • Deposit growth shock −5pp YoY
7) Optional event-study around policy/credit events.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--credit credit.csv          REQUIRED
  Either **bank-level**:
    date, bank, credit_outstanding_inr  [, owner|type|category, psu_flag]
  Or **aggregate** (already PSU-filtered or total with PSU flag):
    date, psu_credit_inr  [and optionally total_credit_inr, pvt_credit_inr]
  Fuzzy maps: credit, loans, advances, net_advances, gross_advances

--deposits deposits.csv      OPTIONAL
  Columns: date [, bank] deposit_outstanding_inr
           (aggregate or bank-level like credit)

--rates rates.csv            OPTIONAL
  Columns (any subset):
    date,
    repo_rate, policy_rate,
    mclr, eblr, lending_rate, wald (weighted avg lending rate),
    deposit_rate, card_rate, term_deposit_rate,
    gsec_10y, y10, yield10

--npa npa.csv                OPTIONAL (bank or aggregate)
  Columns: date [, bank] gnpa_pct [, nnpa_pct, prov_cov_pct]

--sector sector.csv          OPTIONAL (sectoral deployment)
  Columns: date, sector, credit_outstanding_inr  [, bank or psu flag]

--macro macro.csv            OPTIONAL
  Columns: date, cpi, wpi, iip, pmi, gva, gdp, etc. (kept if numeric)

--events events.csv          OPTIONAL
  Columns: date, label

Key CLI
-------
--freq monthly|quarterly     Default monthly
--psu_list psu_list.csv      Optional list of PSU bank names (single column 'bank')
--windows 3,6,12             Rolling windows (periods)
--lags 12                    Lag horizon (periods)
--start / --end              Date filters (YYYY-MM-DD)
--outdir out_psu             Output directory
--min_obs 36                 Min observations for regressions

Outputs
-------
- panel.csv                  Master aligned panel (levels & transforms)
- bank_contrib.csv           Per-bank contributions to PSU credit growth (if bank-level input)
- sector_mix.csv             Sector shares; sector_contrib.csv: sector growth contributions (if sector input)
- rolling_stats.csv          Rolling corr of Δlog(credit) vs drivers
- leadlag_corr.csv           Lead–lag Corr(X_{t−k}, Δlog(credit)_t)
- dlag_regression.csv        Distributed-lag elasticities (coef, se, t)
- scenarios.csv              Repo/Deposit-rate/Deposit-growth shocks → growth impact (pp)
- event_study.csv            Average responses around events (if provided)
- summary.json               Headline diagnostics
- config.json                Run configuration

DISCLAIMER
----------
Research/monitoring tool with simplifications. Validate with your own definitions, data hygiene,
and robustness checks before investment, supervisory, or policy decisions.
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

DEFAULT_PSU_NAMES = {
    "STATE BANK OF INDIA", "SBI",
    "PUNJAB NATIONAL BANK", "PNB",
    "BANK OF BARODA", "BOB",
    "CANARA BANK",
    "UNION BANK OF INDIA",
    "BANK OF INDIA",
    "INDIAN BANK",
    "BANK OF MAHARASHTRA",
    "UCO BANK",
    "CENTRAL BANK OF INDIA",
    "PUNJAB & SIND BANK", "PUNJAB AND SIND BANK"
}

def is_psu(name: str, owner: Optional[str]=None, flag: Optional[str]=None, psu_list: Optional[set]=None) -> bool:
    if pd.isna(name) and pd.isna(owner) and pd.isna(flag): return False
    if psu_list and isinstance(name, str) and name.upper() in psu_list: return True
    if isinstance(flag, str) and flag.strip().lower() in {"y","yes","psu","psb","public"}: return True
    if isinstance(owner, str) and any(k in owner.lower() for k in ["public","gov","goi","psu","psb"]): return True
    if isinstance(name, str) and name.upper() in DEFAULT_PSU_NAMES: return True
    return False

def load_psu_list(path: Optional[str]) -> Optional[set]:
    if not path: return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    bcol = ncol(df, "bank","name","bank_name")
    if not bcol: return None
    return set(df[bcol].dropna().astype(str).str.upper().unique().tolist())

def resample_agg(df: pd.DataFrame, date_col: str, freq: str, how: str="sum") -> pd.DataFrame:
    idx = df.set_index(date_col).sort_index()
    rule = "Q" if freq.startswith("Q") else "M"
    agg = {c: (how if pd.api.types.is_numeric_dtype(idx[c]) else "first") for c in idx.columns}
    return idx.resample(rule).agg(agg).reset_index()

def load_credit(path: str, freq: str, psu_names: Optional[set]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    b  = ncol(df, "bank","name","bank_name","entity")
    own = ncol(df, "owner","type","category")
    flg = ncol(df, "psu_flag","psb","is_psu")
    cred = ncol(df, "credit_outstanding_inr","credit","loans","advances","net_advances","gross_advances","total_loans")
    psu_agg = ncol(df, "psu_credit_inr","psu_credit","public_credit")
    tot_agg = ncol(df, "total_credit_inr","credit_total","banking_credit")
    pvt_agg = ncol(df, "pvt_credit_inr","private_credit")
    if not dt: raise ValueError("credit.csv must include a 'date' column.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], freq)
    if b and cred:
        df = df.rename(columns={b:"bank", cred:"credit"})
        df["credit"] = safe_num(df["credit"])
        if own: df = df.rename(columns={own:"owner"})
        if flg: df = df.rename(columns={flg:"psu_flag"})
        # identify psu
        df["_is_psu"] = df.apply(lambda r: is_psu(
            r.get("bank"), r.get("owner"), r.get("psu_flag"), psu_names
        ), axis=1)
        # bank-level panel
        bank_panel = df[["date","bank","credit","_is_psu"]].copy()
        # aggregate to PSU total
        psu = bank_panel[bank_panel["_is_psu"]].groupby("date", as_index=False)["credit"].sum().rename(columns={"credit":"psu_credit"})
        out = psu.copy()
        if tot_agg and tot_agg in df.columns:
            out = out.merge(df.groupby("date", as_index=False)[tot_agg].sum().rename(columns={tot_agg:"total_credit"}), on="date", how="left")
        return resample_agg(out, "date", freq, how="sum"), resample_agg(bank_panel, "date", freq, how="sum")
    elif psu_agg:
        df = df.rename(columns={psu_agg:"psu_credit"})
        df["psu_credit"] = safe_num(df["psu_credit"])
        keep = ["date","psu_credit"]
        if tot_agg: df = df.rename(columns={tot_agg:"total_credit"}); df["total_credit"] = safe_num(df["total_credit"]); keep.append("total_credit")
        if pvt_agg: df = df.rename(columns={pvt_agg:"pvt_credit"}); df["pvt_credit"] = safe_num(df["pvt_credit"]); keep.append("pvt_credit")
        out = resample_agg(df[keep], "date", freq, how="sum")
        return out, None
    else:
        raise ValueError("credit.csv must contain either bank-level credit or a 'psu_credit' aggregate column.")

def load_deposits(path: Optional[str], freq: str, psu_names: Optional[set]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not path: return pd.DataFrame(), None
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    b  = ncol(df, "bank","name","bank_name","entity")
    dep = ncol(df, "deposit_outstanding_inr","deposits","total_deposits")
    own = ncol(df, "owner","type","category")
    flg = ncol(df, "psu_flag","psb","is_psu")
    if not dt or not dep: raise ValueError("deposits.csv must include date and deposits columns.")
    df = df.rename(columns={dt:"date", dep:"deposits"})
    df["date"] = to_period_end(df["date"], freq)
    df["deposits"] = safe_num(df["deposits"])
    if b:
        if own: df = df.rename(columns={own:"owner"})
        if flg: df = df.rename(columns={flg:"psu_flag"})
        df["_is_psu"] = df.apply(lambda r: is_psu(r.get("bank"), r.get("owner"), r.get("psu_flag"), psu_names), axis=1)
        bank_panel = df[["date","bank","deposits","_is_psu"]].copy()
        psu = bank_panel[bank_panel["_is_psu"]].groupby("date", as_index=False)["deposits"].sum().rename(columns={"deposits":"psu_deposits"})
        return resample_agg(psu, "date", freq, how="sum"), resample_agg(bank_panel, "date", freq, how="sum")
    else:
        # aggregate already
        df = df.rename(columns={"deposits":"psu_deposits"})
        return resample_agg(df[["date","psu_deposits"]], "date", freq, how="sum"), None

def load_rates(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], freq)
    rename = {}
    for pair in [
        ("repo_rate","repo_rate"), ("policy_rate","repo_rate"),
        ("mclr","mclr"), ("eblr","eblr"), ("lending_rate","lending_rate"), ("wald","lending_rate"),
        ("deposit_rate","deposit_rate"), ("term_deposit_rate","deposit_rate"), ("card_rate","deposit_rate"),
        ("gsec_10y","gsec_10y"), ("y10","gsec_10y"), ("yield10","gsec_10y")
    ]:
        c = ncol(df, pair[0])
        if c: rename[c] = pair[1]
    df = df.rename(columns=rename)
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    out = df[["date"] + [c for c in ["repo_rate","mclr","eblr","lending_rate","deposit_rate","gsec_10y"] if c in df.columns]]
    return resample_agg(out, "date", freq, how="mean")

def load_npa(path: Optional[str], freq: str, psu_names: Optional[set]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    b  = ncol(df, "bank","name","bank_name")
    gn = ncol(df, "gnpa_pct","gnpa","gross_npa_pct")
    nn = ncol(df, "nnpa_pct","nnpa","net_npa_pct")
    pc = ncol(df, "prov_cov_pct","provision_coverage_pct","pcr")
    if not dt: raise ValueError("npa.csv must include a date column.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], freq)
    if b:
        df = df.rename(columns={b:"bank"})
    for c in [gn, nn, pc]:
        if c and c in df.columns:
            df[c] = safe_num(df[c])
    # Weighted by credit if available in same frame; else simple mean (approx)
    if b and "credit" in df.columns:
        w = safe_num(df["credit"])
    else:
        w = None
    if b:
        df["_is_psu"] = df.apply(lambda r: is_psu(r.get("bank"), None, None, psu_names), axis=1)
        sub = df[df["_is_psu"]].copy()
        agg = sub.groupby("date").agg({
            gn: (lambda x: np.average(x.dropna(), weights=w.loc[x.index].dropna()) if w is not None else np.nanmean(x)),
            nn: (lambda x: np.average(x.dropna(), weights=w.loc[x.index].dropna()) if w is not None else np.nanmean(x)) if nn else 'first',
            pc: (lambda x: np.average(x.dropna(), weights=w.loc[x.index].dropna()) if w is not None else np.nanmean(x)) if pc else 'first'
        }).reset_index()
        agg = agg.rename(columns={gn:"gnpa_pct", nn or "nnpa_pct":"nnpa_pct", pc or "prov_cov_pct":"prov_cov_pct"})
        return resample_agg(agg, "date", freq, how="mean")
    else:
        # already aggregated
        out = pd.DataFrame({"date": df["date"]})
        if gn: out["gnpa_pct"] = safe_num(df[gn])
        if nn: out["nnpa_pct"] = safe_num(df[nn])
        if pc: out["prov_cov_pct"] = safe_num(df[pc])
        return resample_agg(out, "date", freq, how="mean")

def load_sector(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    sc = ncol(df, "sector","segment","category")
    cr = ncol(df, "credit_outstanding_inr","credit","loans","advances")
    if not (dt and sc and cr): raise ValueError("sector.csv needs date, sector, credit columns.")
    df = df.rename(columns={dt:"date", sc:"sector", cr:"credit"})
    df["date"] = to_period_end(df["date"], freq)
    df["credit"] = safe_num(df["credit"])
    out = df.groupby(["date","sector"], as_index=False)["credit"].sum()
    return resample_agg(out, "date", freq, how="sum")

def load_macro(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], freq)
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    return resample_agg(df, "date", freq, how="mean")

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

def build_panel(freq: str,
                CREDIT: pd.DataFrame,
                DEPO: pd.DataFrame,
                RATES: pd.DataFrame,
                NPA: pd.DataFrame,
                MACRO: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    df = CREDIT.copy()
    if not DEPO.empty: df = df.merge(DEPO, on="date", how="left")
    if not RATES.empty: df = df.merge(RATES, on="date", how="left")
    if not NPA.empty:   df = df.merge(NPA, on="date", how="left")
    if not MACRO.empty: df = df.merge(MACRO, on="date", how="left")
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    # Levels
    # credit: psu_credit ; deposits: psu_deposits
    yo = 12 if freq.startswith("M") else 4
    if "psu_credit" not in df.columns:
        raise ValueError("Could not find PSU credit column 'psu_credit' after loading.")
    df["dlog_credit"] = dlog(df["psu_credit"])
    df["yoy_credit"]  = yoy(df["psu_credit"], yo)

    if "psu_deposits" in df.columns:
        df["dlog_deposits"] = dlog(df["psu_deposits"])
        df["yoy_deposits"]  = yoy(df["psu_deposits"], yo)
        df["cd_ratio"] = df["psu_credit"] / df["psu_deposits"].replace(0, np.nan)

    # Rates & spreads
    for c in ["repo_rate","deposit_rate","lending_rate","gsec_10y","mclr","eblr"]:
        if c in df.columns:
            df[f"d_{c}"] = d(df[c])
    if "lending_rate" in df.columns and "deposit_rate" in df.columns:
        df["rate_spread"] = df["lending_rate"] - df["deposit_rate"]
        df["d_rate_spread"] = d(df["rate_spread"])

    # Real policy rate if CPI present
    cpi = ncol(df, "cpi","cpi_all","cpi_combined")
    if cpi and "repo_rate" in df.columns:
        df["real_repo"] = df["repo_rate"] - df[cpi]
        df["d_real_repo"] = d(df["real_repo"])

    # Mapping of drivers for diagnostics/regression
    mapping = {
        "credit": "dlog_credit",
        "deposit_growth": "dlog_deposits" if "dlog_deposits" in df.columns else None,
        "repo_change": "d_repo_rate" if "d_repo_rate" in df.columns else None,
        "dep_rate_change": "d_deposit_rate" if "d_deposit_rate" in df.columns else None,
        "y10_change": "d_gsec_10y" if "d_gsec_10y" in df.columns else None,
        "rate_spread_change": "d_rate_spread" if "d_rate_spread" in df.columns else None
    }
    # Add first macro growth driver, if any
    macro_dlogs = []
    for c in df.columns:
        if c not in ["date"] and pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in {
            "psu_credit","psu_deposits","yoy_credit","yoy_deposits","cd_ratio","repo_rate","deposit_rate",
            "lending_rate","gsec_10y","mclr","eblr","rate_spread","real_repo",
            "d_repo_rate","d_deposit_rate","d_lending_rate","d_gsec_10y","d_mclr","d_eblr","d_rate_spread","d_real_repo",
            "dlog_credit","dlog_deposits"
        }:
            # create dlog if not rate-like (heuristic)
            if "pmi" in c.lower() or "rate" in c.lower() or "yield" in c.lower():
                continue
            df[f"dlog_{c}"] = dlog(df[c])
            macro_dlogs.append(f"dlog_{c}")
    if macro_dlogs:
        mapping["macro"] = macro_dlogs[0]

    return df, mapping

def bank_contributions(bank_panel: Optional[pd.DataFrame], freq: str) -> pd.DataFrame:
    """
    Contribution of each PSU bank to aggregate PSU credit growth (Δlog):
      contrib_{i,t} = weight_{i,t-1} * Δlog(credit_{i,t})
    where weight_{i,t-1} = credit_{i,t-1} / PSU_credit_{t-1}
    """
    if bank_panel is None or bank_panel.empty: return pd.DataFrame()
    df = bank_panel.copy()
    df = df[df["_is_psu"]].copy()
    df = df.sort_values(["bank","date"])
    df["dlog_bank"] = dlog(df["credit"])
    # lagged weights
    tot = df.groupby("date", as_index=False)["credit"].sum().rename(columns={"credit":"psu_credit"})
    df = df.merge(tot, on="date", how="left")
    df["w_lag"] = (df.groupby("bank")["credit"].shift(1) /
                   df.groupby("bank")["psu_credit"].shift(1))
    df["contrib"] = df["w_lag"] * df["dlog_bank"]
    out = df[["date","bank","contrib","dlog_bank","w_lag","credit"]].dropna(subset=["contrib"])
    return resample_agg(out, "date", freq, how="sum")

def sector_mix_and_contrib(sector: pd.DataFrame, psu_total: pd.DataFrame, freq: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if sector.empty: return pd.DataFrame(), pd.DataFrame()
    S = sector.copy().sort_values(["date","sector"])
    # shares
    tot = S.groupby("date", as_index=False)["credit"].sum().rename(columns={"credit":"sector_total"})
    S = S.merge(tot, on="date", how="left")
    S["share"] = S["credit"] / S["sector_total"].replace(0, np.nan)
    mix = S[["date","sector","share","credit"]].copy()
    # contributions vs PSU (if psu_total provided; else vs sector total)
    S["dlog"] = dlog(S.groupby("sector")["credit"].transform(lambda x: x))
    # Use PSU aggregate if present
    if "psu_credit" in psu_total.columns:
        PSU = psu_total[["date","psu_credit"]].copy()
        S = S.merge(PSU, on="date", how="left")
        S["w_lag"] = (S.groupby("sector")["credit"].shift(1) /
                      S["psu_credit"].shift(1))
    else:
        S["w_lag"] = S.groupby("sector")["credit"].shift(1) / S["sector_total"].shift(1)
    S["contrib"] = S["w_lag"] * S["dlog"]
    contrib = S.dropna(subset=["contrib"])[["date","sector","contrib","share"]].copy()
    return (resample_agg(mix, "date", freq, how="mean"),
            resample_agg(contrib, "date", freq, how="sum"))

# ----------------------------- analytics -----------------------------

def rolling_stats(panel: pd.DataFrame, mapping: Dict[str,str], windows: List[int]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    y = idx.get("dlog_credit")
    if y is None: return pd.DataFrame()
    for key, col in mapping.items():
        if key=="credit" or not col or col not in idx.columns: continue
        x = idx[col]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({"driver": key, "column": col, "window": w, "tag": tag,
                         "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan})
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, mapping: Dict[str,str], lags: int) -> pd.DataFrame:
    rows = []
    y = panel.get("dlog_credit")
    if y is None or y.dropna().empty: return pd.DataFrame()
    for key, col in mapping.items():
        if key=="credit" or not col or col not in panel.columns: continue
        tab = leadlag_corr(panel[col], y, lags)
        tab["driver"] = key
        tab["column"] = col
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","driver","column"])

def dlag_regression(panel: pd.DataFrame, mapping: Dict[str,str], L: int, min_obs: int) -> pd.DataFrame:
    """
    Δlog(Credit)_t on lags (0..L) of selected drivers with HAC SEs.
    We include: ΔRepo, ΔDepositRate, Δlog(Deposits), Δ10Y, first macro Δlog if present.
    """
    if "dlog_credit" not in panel.columns: return pd.DataFrame()
    df = panel.copy()
    dep = "dlog_credit"
    Xparts = [pd.Series(1.0, index=df.index, name="const")]
    names = ["const"]
    use_keys = ["repo_change","dep_rate_change","deposit_growth","y10_change","macro"]
    for key in use_keys:
        col = mapping.get(key)
        if not col or col not in df.columns: continue
        for l in range(0, L+1):
            nm = f"{col}_l{l}"
            Xparts.append(df[col].shift(l).rename(nm))
            names.append(nm)
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
                    "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan), "lags": L})
    # cumulative effects 0..L for key drivers
    def cum_row(prefix: str, label: str):
        idxs = [i for i, nm in enumerate(names) if nm.startswith(f"{prefix}_l")]
        if idxs:
            bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
            out.append({"var": f"{label}_cum_0..L", "coef": bsum, "se": ses,
                        "t_stat": bsum/(ses if ses>0 else np.nan), "lags": L})
    for lab, pref in [("d_repo_rate","d_repo_rate"), ("d_deposit_rate","d_deposit_rate"),
                      ("dlog_deposits","dlog_deposits"), ("d_gsec_10y","d_gsec_10y")]:
        if f"{pref}_l0" in [r["var"] for r in out]:
            cum_row(pref, lab)
    return pd.DataFrame(out)

def scenarios(reg: pd.DataFrame) -> pd.DataFrame:
    """
    Map shocks to Δlog(Credit) impact (percentage points, pp):
      Repo +50bp / +100bp           → use cum β on d_repo_rate (per 1.00 = 100bp)
      Deposit rate +50bp            → use cum β on d_deposit_rate
      Deposit growth −5pp YoY       → approximate: −5pp/12 per month applied to cum β on dlog_deposits
    """
    if reg.empty: return pd.DataFrame()
    def get_cum(label: str) -> Optional[float]:
        r = reg[reg["var"] == f"{label}_cum_0..L"]
        return float(r["coef"].iloc[0]) if not r.empty else None

    b_repo = get_cum("d_repo_rate")
    b_depr = get_cum("d_deposit_rate")
    b_depG = get_cum("dlog_deposits")

    rows = []
    if b_repo is not None:
        for bp in [0.50, 1.00]:
            rows.append({"shock": f"Repo +{int(bp*100)}bp", "impact_pp": b_repo * bp * 100.0})
    if b_depr is not None:
        rows.append({"shock": "Deposit rate +50bp", "impact_pp": b_depr * 0.50 * 100.0})
    if b_depG is not None:
        # −5pp YoY ~ −0.05/12 in monthly log terms (rough), impact = β_depG * Δlog_dep
        dl = -0.05/12.0
        rows.append({"shock": "Deposit growth −5pp YoY", "impact_pp": b_depG * dl * 100.0})
    return pd.DataFrame(rows)

def event_study(panel: pd.DataFrame, events: pd.DataFrame, window: int=6) -> pd.DataFrame:
    if events.empty or "dlog_credit" not in panel.columns:
        return pd.DataFrame()
    rows = []
    dates = panel["date"].values
    for _, ev in events.iterrows():
        d0 = pd.to_datetime(ev["date"])
        lbl = str(ev["label"])
        if len(dates)==0: continue
        idx = np.argmin(np.abs(dates - np.datetime64(d0)))
        anchor = pd.to_datetime(dates[idx])
        sl = panel[(panel["date"] >= anchor - pd.offsets.DateOffset(months=window)) &
                   (panel["date"] <= anchor + pd.offsets.DateOffset(months=window))]
        for _, r in sl.iterrows():
            h = (r["date"].to_period("M") - anchor.to_period("M")).n if isinstance(r["date"], pd.Timestamp) else 0
            rows.append({"event": lbl, "event_date": anchor, "h": int(h),
                         "dlog_credit": float(r.get("dlog_credit", np.nan)) if pd.notna(r.get("dlog_credit", np.nan)) else np.nan,
                         "d_repo_rate": float(r.get("d_repo_rate", np.nan)) if pd.notna(r.get("d_repo_rate", np.nan)) else np.nan})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(["event","event_date","h"])
              .agg({"dlog_credit":"mean","d_repo_rate":"mean"})
              .reset_index()
              .sort_values(["event","h"]))

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    credit: str
    deposits: Optional[str]
    rates: Optional[str]
    npa: Optional[str]
    sector: Optional[str]
    macro: Optional[str]
    events: Optional[str]
    psu_list: Optional[str]
    freq: str
    windows: str
    lags: int
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PSU bank lending analytics (India)")
    ap.add_argument("--credit", required=True)
    ap.add_argument("--deposits", default="")
    ap.add_argument("--rates", default="")
    ap.add_argument("--npa", default="")
    ap.add_argument("--sector", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--psu_list", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","quarterly"])
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_psu")
    ap.add_argument("--min_obs", type=int, default=36)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq = "M" if args.freq.startswith("m") else "Q"

    psu_names = load_psu_list(args.psu_list)

    CREDIT, BANK_PANEL = load_credit(args.credit, freq=freq, psu_names=psu_names)
    DEPO, BANK_DEP = load_deposits(args.deposits, freq=freq, psu_names=psu_names) if args.deposits else (pd.DataFrame(), None)
    RATES = load_rates(args.rates, freq=freq) if args.rates else pd.DataFrame()
    NPA   = load_npa(args.npa, freq=freq, psu_names=psu_names) if args.npa else pd.DataFrame()
    MACRO = load_macro(args.macro, freq=freq) if args.macro else pd.DataFrame()
    SECTOR = load_sector(args.sector, freq=freq) if args.sector else pd.DataFrame()
    EVENTS = load_events(args.events, freq=freq) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        for df in [CREDIT, DEPO, RATES, NPA, MACRO, SECTOR]:
            if not df.empty:
                df.drop(df[df["date"] < pd.to_datetime(args.start)].index, inplace=True)
    if args.end:
        for df in [CREDIT, DEPO, RATES, NPA, MACRO, SECTOR]:
            if not df.empty:
                df.drop(df[df["date"] > pd.to_datetime(args.end)].index, inplace=True)

    PANEL, mapping = build_panel(freq=freq, CREDIT=CREDIT, DEPO=DEPO, RATES=RATES, NPA=NPA, MACRO=MACRO)
    if PANEL["dlog_credit"].dropna().shape[0] < args.min_obs:
        raise ValueError("Insufficient overlapping periods after alignment (need ≥ min_obs on Δlog credit).")

    # Persist panel
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Bank contributions (if bank-level)
    BANK_CONTRIB = bank_contributions(BANK_PANEL, freq=freq)
    if not BANK_CONTRIB.empty:
        BANK_CONTRIB.to_csv(outdir / "bank_contrib.csv", index=False)

    # Sector mix & contributions
    SECTOR_MIX = pd.DataFrame(); SECTOR_CONTRIB = pd.DataFrame()
    if not SECTOR.empty:
        SECTOR_MIX, SECTOR_CONTRIB = sector_mix_and_contrib(SECTOR, CREDIT, freq=freq)
        if not SECTOR_MIX.empty: SECTOR_MIX.to_csv(outdir / "sector_mix.csv", index=False)
        if not SECTOR_CONTRIB.empty: SECTOR_CONTRIB.to_csv(outdir / "sector_contrib.csv", index=False)

    # Rolling & lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, mapping, windows)
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)
    LL = leadlag_tables(PANEL, mapping, lags=int(args.lags))
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # D-lag regression
    REG = dlag_regression(PANEL, mapping, L=int(args.lags), min_obs=int(args.min_obs))
    if not REG.empty: REG.to_csv(outdir / "dlag_regression.csv", index=False)

    # Scenarios
    SCN = scenarios(REG) if not REG.empty else pd.DataFrame()
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Event study
    ES = event_study(PANEL, EVENTS, window=max(6, int(args.lags)//2)) if not EVENTS.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Best lead/lag summary
    best_ll = {}
    if not LL.empty:
        for drv, g in LL.dropna(subset=["corr"]).groupby("driver"):
            row = g.iloc[g["corr"].abs().argmax()]
            best_ll[drv] = {"lag": int(row["lag"]), "corr": float(row["corr"]), "column": row["column"]}

    # Summary
    last = PANEL.dropna(subset=["psu_credit"]).tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "freq": "monthly" if freq=="M" else "quarterly",
        "columns_present": {
            "psu_credit": "psu_credit" in PANEL.columns,
            "psu_deposits": "psu_deposits" in PANEL.columns,
            "repo_rate": "repo_rate" in PANEL.columns,
            "deposit_rate": "deposit_rate" in PANEL.columns,
            "gsec_10y": "gsec_10y" in PANEL.columns
        },
        "latest": {
            "date": str(last["date"].date()),
            "psu_credit": float(last["psu_credit"]),
            "psu_deposits": float(last["psu_deposits"]) if "psu_deposits" in PANEL.columns and pd.notna(last.get("psu_deposits", np.nan)) else None,
            "cd_ratio": float(last.get("cd_ratio", np.nan)) if pd.notna(last.get("cd_ratio", np.nan)) else None,
            "repo_rate": float(last.get("repo_rate", np.nan)) if pd.notna(last.get("repo_rate", np.nan)) else None,
            "deposit_rate": float(last.get("deposit_rate", np.nan)) if pd.notna(last.get("deposit_rate", np.nan)) else None
        },
        "rolling_windows": windows,
        "leadlag_best": best_ll,
        "reg_terms": REG["var"].tolist() if not REG.empty else [],
        "scenarios": SCN.to_dict(orient="records") if not SCN.empty else [],
        "bank_contrib_available": (not BANK_CONTRIB.empty),
        "sector_breakdown_available": (not SECTOR_MIX.empty)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        credit=args.credit, deposits=(args.deposits or None), rates=(args.rates or None),
        npa=(args.npa or None), sector=(args.sector or None), macro=(args.macro or None),
        events=(args.events or None), psu_list=(args.psu_list or None),
        freq=args.freq, windows=args.windows, lags=int(args.lags),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== PSU Bank Lending Analytics ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Freq: {summary['freq']}")
    if summary["latest"]["cd_ratio"] is not None:
        print(f"Latest CD ratio: {summary['latest']['cd_ratio']:.2f}")
    if best_ll:
        for k, st in best_ll.items():
            print(f"Lead–lag {k}: max |corr| at lag {st['lag']:+d} → {st['corr']:+.2f}")
    if not SCN.empty:
        for _, r in SCN.iterrows():
            print(f"{r['shock']}: impact ≈ {r['impact_pp']:+.2f} pp on Δlog(credit)")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
