#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rbi_digital_rupee.py — e₹ (RBI CBDC) adoption, substitution, liquidity & scenarios
----------------------------------------------------------------------------------

What this does
==============
Given timeseries for India's **Digital Rupee (e₹/CBDC)** pilots alongside UPI/cash usage,
bank deposit rates/liquidity, and optional bank balance-sheet snapshots, this script:

1) Cleans & aligns all inputs to **monthly** (or **daily**) frequency.
2) Builds key transforms and diagnostics:
   • Δlog growth of CBDC tx counts/values, UPI, ATM withdrawals, Cash-in-Circulation (CiC)
   • Share-of-wallet & substitution metrics (CBDC vs UPI vs Cash proxies)
   • Per-wallet & per-merchant activity, avg ticket size
   • Rolling correlations & lead–lag tables vs UPI/cash/liquidity
3) Cannibalization regressions (Newey–West HAC SE):
   Δlog(UPI)_t ~ Σ β·Δlog(CBDC)_t−i + controls (rates/liquidity)
4) S-curve (logistic) fit for **active retail wallets** and/or **merchant count** with grid-searched carrying capacity.
5) Event study (policy launches/incentives): ARound averages for activity deltas.
6) Scenario engine:
   • Adoption push (merchant incentive / fee change): target CBDC share by horizon → implied path
   • Interest-bearing CBDC: **deposit migration** given elasticities → bank NIM impact
   • Payments cost view: system cost delta if CBDC replaces share of UPI/cash

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--cbdc cbdc.csv              REQUIRED
  date,
  tx_count[, tx_value_inr], active_wallets_retail[, active_wallets_wholesale],
  merchants_active[, merchants_total]

--upi upi.csv                OPTIONAL
  date, upi_tx_count[, upi_tx_value_inr][, merchants_upi]

--cash cash.csv              OPTIONAL (proxies)
  date, atm_withdrawal_value_inr[, cic_inr]

--rates rates.csv            OPTIONAL (monthly averages)
  date, deposit_rate[, repo_rate, gsec_10y]

--liquidity liq.csv          OPTIONAL (bank/system liquidity)
  date, laf_absorption_inr[, laf_injection_inr, net_liquidity_inr]

--bank bank.csv              OPTIONAL (balance-sheet snapshot for scenarios)
  date, assets_inr, loans_inr, deposits_inr, asset_yield_pct, deposit_rate_pct[, alt_funding_cost_pct]

--events events.csv          OPTIONAL
  date, label

--costs costs.csv            OPTIONAL (payments unit costs; INR per txn)
  channel, unit_cost_inr
  (channels expected: CBDC, UPI, CASH; missing entries default to nan)

Key CLI
-------
--freq monthly|daily         Output frequency (default: monthly)
--lags 6                     Max lag for lead–lag & d-lag regressions (periods)
--windows 3,6,12             Rolling windows
--s_grid 1.5,3.0,5.0         Carrying-capacity multipliers for S-curve gridsearch vs max observed
--target_share 0.10          Scenario: target CBDC share of digital value (CBDC/(CBDC+UPI)) in 12m
--elas_dep 0.25              Deposit migration elasticity w.r.t. (r_cbdc − r_deposit) in abs terms
--share_movable 0.30         Share of deposits at risk to move to CBDC (transactional/low-friction)
--cbdc_rate_pct 2.0          Hypothetical CBDC interest (annualized, %)
--horizon_m 12               Scenario horizon in months
--start / --end              Date filters (YYYY-MM-DD)
--outdir out_einr            Output directory
--min_obs 24                 Minimum obs for regressions

Outputs
-------
- panel.csv                  Master aligned panel (levels & transforms)
- rolling_stats.csv          Rolling corr of Δlog series (CBDC vs UPI/cash/rates/liquidity)
- leadlag_corr.csv           Lead–lag Corr(X_{t−k}, Δlog(CBDC/UPI))
- cannibalization_reg.csv    D-lag regression results (coef, HAC se, t) & cumulative effects
- s_curve_fit.csv            Logistic fit for wallets/merchants (capacity, k, t0) & fitted path
- event_study.csv            Mean deltas around events (±h)
- scenarios.csv              Adoption/Deposit migration/Payments cost scenarios
- summary.json               Headline diagnostics
- config.json                Run configuration

DISCLAIMER
----------
Research tooling with simplifying assumptions. Validate against RBI data, pilot design,
and internal cost/accounting before policy or investment decisions.
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
    if freq.startswith("D"):
        return dt.dt.normalize()
    return dt.dt.to_period("M").dt.to_timestamp("M")

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def yoy_log(s: pd.Series, periods: int) -> pd.Series:
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
        c = xx.corr(yy)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

def winsorize_g(s: pd.Series, p: float=0.01) -> pd.Series:
    x = s.copy()
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

# HAC (Newey–West) utilities
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

def load_cbdc(path: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    cnt = ncol(df, "tx_count","transactions","txn_count")
    val = ncol(df, "tx_value_inr","value_inr","txn_value_inr","turnover_inr")
    wal_r = ncol(df, "active_wallets_retail","wallets_retail","active_retail_wallets")
    wal_w = ncol(df, "active_wallets_wholesale","wallets_wholesale","active_wholesale_wallets")
    mch_a = ncol(df, "merchants_active","active_merchants","merchants")
    mch_t = ncol(df, "merchants_total","total_merchants")
    if not dt or not cnt:
        raise ValueError("cbdc.csv must include date and tx_count columns.")
    df = df.rename(columns={dt:"date", cnt:"cbdc_tx_count"})
    if val: df = df.rename(columns={val:"cbdc_tx_value_inr"})
    if wal_r: df = df.rename(columns={wal_r:"wallets_retail"})
    if wal_w: df = df.rename(columns={wal_w:"wallets_wholesale"})
    if mch_a: df = df.rename(columns={mch_a:"merchants_active"})
    if mch_t: df = df.rename(columns={mch_t:"merchants_total"})
    for c in df.columns:
        if c != "date":
            df[c] = safe_num(df[c])
    df["date"] = to_period_end(df["date"], "D" if freq.startswith("d") else "M")
    # derived
    if "cbdc_tx_value_inr" in df.columns and "cbdc_tx_count" in df.columns:
        df["cbdc_avg_ticket_inr"] = df["cbdc_tx_value_inr"] / df["cbdc_tx_count"].replace(0, np.nan)
    if "wallets_retail" in df.columns:
        df["cbdc_tx_per_wallet"] = df["cbdc_tx_count"] / df["wallets_retail"].replace(0, np.nan)
        if "cbdc_tx_value_inr" in df.columns:
            df["cbdc_value_per_wallet_inr"] = df["cbdc_tx_value_inr"] / df["wallets_retail"].replace(0, np.nan)
    return df.sort_values("date")

def load_upi(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    cnt = ncol(df, "upi_tx_count","tx_count","transactions")
    val = ncol(df, "upi_tx_value_inr","value_inr","txn_value_inr","turnover_inr")
    mch = ncol(df, "merchants_upi","merchants","qrr_merchants")
    if not dt or not (cnt or val):
        raise ValueError("upi.csv needs date and tx_count and/or tx_value_inr.")
    df = df.rename(columns={dt:"date"})
    if cnt: df = df.rename(columns={cnt:"upi_tx_count"})
    if val: df = df.rename(columns={val:"upi_tx_value_inr"})
    if mch: df = df.rename(columns={mch:"merchants_upi"})
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    df["date"] = to_period_end(df["date"], "D" if freq.startswith("d") else "M")
    if "upi_tx_value_inr" in df.columns and "upi_tx_count" in df.columns:
        df["upi_avg_ticket_inr"] = df["upi_tx_value_inr"] / df["upi_tx_count"].replace(0, np.nan)
    return df.sort_values("date")

def load_cash(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    atm = ncol(df, "atm_withdrawal_value_inr","atm_value_inr")
    cic = ncol(df, "cic_inr","cash_in_circulation_inr")
    if not dt: raise ValueError("cash.csv needs a date column.")
    df = df.rename(columns={dt:"date"})
    if atm: df = df.rename(columns={atm:"atm_withdrawal_value_inr"})
    if cic: df = df.rename(columns={cic:"cic_inr"})
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    df["date"] = to_period_end(df["date"], "D" if freq.startswith("d") else "M")
    return df.sort_values("date")

def load_rates(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    dep = ncol(df, "deposit_rate","term_deposit_rate")
    rep = ncol(df, "repo_rate","policy_rate")
    y10 = ncol(df, "gsec_10y","y10","yield10")
    if not dt: raise ValueError("rates.csv needs date.")
    df = df.rename(columns={dt:"date"})
    if dep: df = df.rename(columns={dep:"deposit_rate"})
    if rep: df = df.rename(columns={rep:"repo_rate"})
    if y10: df = df.rename(columns={y10:"gsec_10y"})
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    df["date"] = to_period_end(df["date"], "D" if freq.startswith("d") else "M")
    return df.sort_values("date")

def load_liq(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    absn= ncol(df, "laf_absorption_inr","absorption_inr")
    inj = ncol(df, "laf_injection_inr","injection_inr")
    net = ncol(df, "net_liquidity_inr","net_liq_inr","system_liquidity_inr")
    if not dt: raise ValueError("liq.csv needs date.")
    df = df.rename(columns={dt:"date"})
    for c in [absn, inj, net]:
        if c and c in df.columns: df[c] = safe_num(df[c])
    if absn: df = df.rename(columns={absn:"laf_absorption_inr"})
    if inj:  df = df.rename(columns={inj:"laf_injection_inr"})
    if net:  df = df.rename(columns={net:"net_liquidity_inr"})
    df["date"] = to_period_end(df["date"], "D" if freq.startswith("d") else "M")
    return df.sort_values("date")

def load_bank(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    cols = {
        "assets_inr":["assets_inr","total_assets_inr"],
        "loans_inr":["loans_inr","credit_inr"],
        "deposits_inr":["deposits_inr"],
        "asset_yield_pct":["asset_yield_pct","asset_yield"],
        "deposit_rate_pct":["deposit_rate_pct","deposit_rate"],
        "alt_funding_cost_pct":["alt_funding_cost_pct","wholesale_cost_pct"]
    }
    if not dt: raise ValueError("bank.csv needs date.")
    df = df.rename(columns={dt:"date"})
    for std, cands in cols.items():
        cc = None
        for c in cands:
            cc = ncol(df, c) or cc
        if cc: df = df.rename(columns={cc: std})
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    df["date"] = to_period_end(df["date"], "D" if freq.startswith("d") else "M")
    return df.sort_values("date")

def load_events(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    lab= ncol(df, "label","event","name") or "label"
    df = df.rename(columns={dt:"date", lab:"label"})
    df["date"] = to_period_end(df["date"], "D" if freq.startswith("d") else "M")
    df["label"] = df["label"].astype(str)
    return df[["date","label"]].dropna().sort_values("date")

def load_costs(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ch = ncol(df, "channel") or "channel"
    uc = ncol(df, "unit_cost_inr","cost_inr","unit_cost")
    if not uc: raise ValueError("costs.csv must include unit_cost_inr.")
    df = df.rename(columns={ch:"channel", uc:"unit_cost_inr"})
    df["channel"] = df["channel"].astype(str).str.upper()
    df["unit_cost_inr"] = safe_num(df["unit_cost_inr"])
    return df[["channel","unit_cost_inr"]]

# ----------------------------- core construction -----------------------------

def resample_mean_sum(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty: return df
    idx = df.set_index("date").sort_index()
    rule = "D" if freq.startswith("d") else "M"
    agg = {}
    for c in idx.columns:
        if "count" in c or "tx_" in c or "withdrawal" in c or "value" in c:
            agg[c] = "sum"
        else:
            agg[c] = "mean"
    out = idx.resample(rule).agg(agg)
    out = out.reset_index()
    return out

def build_panel(freq: str,
                CBDC: pd.DataFrame, UPI: pd.DataFrame,
                CASH: pd.DataFrame, RATES: pd.DataFrame,
                LIQ: pd.DataFrame, BANK: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str,str]]:
    df = CBDC.copy()
    for d in [UPI, CASH, RATES, LIQ, BANK]:
        if not d.empty:
            df = df.merge(d, on="date", how="left")
    # Resample to target freq with sensible aggregations
    df = resample_mean_sum(df, freq)
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    # Derived shares (digital = CBDC + UPI; cash proxy via ATM value)
    if "cbdc_tx_value_inr" in df.columns:
        df["digital_value_cbdc"] = df["cbdc_tx_value_inr"]
    if "upi_tx_value_inr" in df.columns:
        df["digital_value_upi"] = df["upi_tx_value_inr"]
    if set(["digital_value_cbdc","digital_value_upi"]).issubset(df.columns):
        total_digital = (df["digital_value_cbdc"].fillna(0) + df["digital_value_upi"].fillna(0))
        df["share_cbdc_value"] = df["digital_value_cbdc"] / total_digital.replace(0, np.nan)
    if "atm_withdrawal_value_inr" in df.columns:
        denom = (df.get("digital_value_cbdc",0).fillna(0) + df.get("digital_value_upi",0).fillna(0) + df["atm_withdrawal_value_inr"].fillna(0))
        df["share_cash_proxy"] = df["atm_withdrawal_value_inr"] / denom.replace(0, np.nan)

    # Transforms
    # CBDC
    if "cbdc_tx_count" in df.columns: df["dlog_cbdc_cnt"] = dlog(df["cbdc_tx_count"])
    if "cbdc_tx_value_inr" in df.columns: df["dlog_cbdc_val"] = dlog(df["cbdc_tx_value_inr"])
    # UPI
    if "upi_tx_count" in df.columns: df["dlog_upi_cnt"] = dlog(df["upi_tx_count"])
    if "upi_tx_value_inr" in df.columns: df["dlog_upi_val"] = dlog(df["upi_tx_value_inr"])
    # Cash & CiC
    if "atm_withdrawal_value_inr" in df.columns: df["dlog_atm_val"] = dlog(df["atm_withdrawal_value_inr"])
    if "cic_inr" in df.columns:
        df["dlog_cic"] = dlog(df["cic_inr"])
        df["yoy_cic"] = yoy_log(df["cic_inr"], 12 if freq.startswith("m") else 365)

    # Rates/Liquidity deltas
    for c in ["deposit_rate","repo_rate","gsec_10y","laf_absorption_inr","laf_injection_inr","net_liquidity_inr"]:
        if c in df.columns:
            df[f"d_{c}"] = df[c].astype(float).diff()

    # Mapping for analytics
    mapping = {
        "cbdc_val": "dlog_cbdc_val" if "dlog_cbdc_val" in df.columns else None,
        "cbdc_cnt": "dlog_cbdc_cnt" if "dlog_cbdc_cnt" in df.columns else None,
        "upi_val":  "dlog_upi_val"  if "dlog_upi_val"  in df.columns else None,
        "upi_cnt":  "dlog_upi_cnt"  if "dlog_upi_cnt"  in df.columns else None,
        "cash_val": "dlog_atm_val"  if "dlog_atm_val"  in df.columns else None,
        "cic":      "dlog_cic"      if "dlog_cic"      in df.columns else None,
        "deposit_rate": "d_deposit_rate" if "d_deposit_rate" in df.columns else None,
        "net_liq": "d_net_liquidity_inr" if "d_net_liquidity_inr" in df.columns else None
    }
    return df, mapping

# ----------------------------- analytics -----------------------------

def rolling_stats(panel: pd.DataFrame, mapping: Dict[str,str], windows: List[int]) -> pd.DataFrame:
    rows = []
    idx = panel.set_index("date")
    y = idx.get(mapping.get("cbdc_val") or mapping.get("cbdc_cnt"))
    if y is None: return pd.DataFrame()
    for key in ["upi_val","upi_cnt","cash_val","cic","deposit_rate","net_liq"]:
        col = mapping.get(key)
        if not col or col not in idx.columns: continue
        x = idx[col]
        for w, tag in zip(windows, ["short","med","long"]):
            rows.append({"driver": key, "column": col, "window": w, "tag": tag,
                         "corr": float(roll_corr(x, y, w).iloc[-1]) if len(idx)>=w else np.nan})
    return pd.DataFrame(rows)

def leadlag_tables(panel: pd.DataFrame, mapping: Dict[str,str], lags: int) -> pd.DataFrame:
    rows = []
    y = panel.get(mapping.get("cbdc_val") or mapping.get("cbdc_cnt"))
    if y is None or y.dropna().empty: return pd.DataFrame()
    for key in ["upi_val","upi_cnt","cash_val","cic","deposit_rate","net_liq"]:
        col = mapping.get(key)
        if not col or col not in panel.columns: continue
        tab = leadlag_corr(panel[col], y, lags)
        tab["driver"] = key
        tab["column"] = col
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lag","corr","driver","column"])

def cannibalization_reg(panel: pd.DataFrame, mapping: Dict[str,str], L: int, min_obs: int) -> pd.DataFrame:
    """
    Δlog(UPI)_t on lags (0..L) of Δlog(CBDC) + controls (deposit rate Δ, net liquidity Δ).
    """
    y_col = mapping.get("upi_val") or mapping.get("upi_cnt")
    x_col = mapping.get("cbdc_val") or mapping.get("cbdc_cnt")
    if not y_col or not x_col: return pd.DataFrame()
    df = panel.copy()
    dep = y_col
    Xparts = [pd.Series(1.0, index=df.index, name="const")]
    names = ["const"]
    # main driver (CBDC)
    for l in range(0, L+1):
        nm = f"{x_col}_l{l}"
        Xparts.append(df[x_col].shift(l).rename(nm))
        names.append(nm)
    # controls
    for ctrl_key in ["deposit_rate","net_liq"]:
        c = mapping.get(ctrl_key)
        if c and c in df.columns:
            nm = f"{c}_l0"
            Xparts.append(df[c].shift(0).rename(nm)); names.append(nm)
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
    # cumulative effect of CBDC lags on UPI (0..L)
    idxs = [i for i, nm in enumerate(names) if nm.startswith(f"{x_col}_l")]
    if idxs:
        bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
        out.append({"var": f"{x_col}_cum_0..L", "coef": bsum, "se": ses,
                    "t_stat": bsum/(ses if ses>0 else np.nan), "lags": L})
    return pd.DataFrame(out)

# ----------------------------- S-curve estimation -----------------------------

def logistic_fit(series: pd.Series, grid_mult: List[float]) -> Tuple[Optional[Dict], pd.DataFrame]:
    """
    Fit simple logistic: y(t) = K / (1 + exp(-k*(t - t0)))
    • K (capacity) is grid-searched as mult × max(y).
    • k and t0 via OLS on logit(y/K) ≈ k*t - k*t0.
    Returns params and fitted path.
    """
    y = series.astype(float).dropna()
    if y.empty or len(y) < 8:
        return None, pd.DataFrame()
    t = np.arange(len(y), dtype=float)
    Kcands = [m * np.nanmax(y.values) for m in grid_mult if m>1.0]
    best = None; best_err = np.inf; best_fit = None
    for K in Kcands:
        z = y / K
        # clamp for logit
        z = np.clip(z, 1e-6, 1-1e-6)
        logit = np.log(z/(1-z))
        # OLS: logit = a + b*t  → k=b, t0 = -a/b
        X = np.column_stack([np.ones_like(t), t])
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ logit.values.reshape(-1,1))
        a, b = float(beta[0,0]), float(beta[1,0])
        fitted = K / (1.0 + np.exp(-(b*(t) + a)))
        err = float(np.nanmean((fitted - y.values)**2))
        if err < best_err and b>0:
            best_err = err
            best = {"K": K, "k": b, "t0": -a/b, "mse": err}
            best_fit = fitted
    if best is None:
        return None, pd.DataFrame()
    # Build full path on original index
    fit_df = pd.DataFrame({"date": y.index, "actual": y.values, "fitted": best_fit})
    return best, fit_df

# ----------------------------- Event study -----------------------------

def event_study(panel: pd.DataFrame, events: pd.DataFrame, window: int=6) -> pd.DataFrame:
    if events.empty: return pd.DataFrame()
    key_cols = [c for c in ["cbdc_tx_count","cbdc_tx_value_inr","upi_tx_count","upi_tx_value_inr","atm_withdrawal_value_inr"] if c in panel.columns]
    if not key_cols: return pd.DataFrame()
    rows = []
    idx = panel.set_index("date")
    for _, ev in events.iterrows():
        d0 = pd.to_datetime(ev["date"])
        if d0 not in idx.index:  # pick nearest prev date
            prev = idx.index[idx.index <= d0]
            if len(prev)==0: continue
            d0 = prev.max()
        sl = idx.loc[(idx.index >= d0 - pd.offsets.DateOffset(months=window)) &
                     (idx.index <= d0 + pd.offsets.DateOffset(months=window))]
        for h, (dt, r) in enumerate(sl.iterrows(), start=-window):
            out = {"event": ev["label"], "event_date": d0, "h": h}
            for c in key_cols:
                out[c] = float(r[c]) if pd.notna(r[c]) else np.nan
            rows.append(out)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # convert to deltas relative to h = -1 avg (pre-window baseline)
    out_rows = []
    for (lab, d0), g in df.groupby(["event","event_date"]):
        base = g[g["h"]<0][key_cols].mean(numeric_only=True)
        for _, r in g.iterrows():
            rr = {"event": lab, "event_date": d0, "h": int(r["h"])}
            for c in key_cols:
                rr[f"d_{c}"] = float(r[c] - base.get(c, np.nan)) if pd.notna(r.get(c, np.nan)) else np.nan
            out_rows.append(rr)
    return pd.DataFrame(out_rows).sort_values(["event","event_date","h"])

# ----------------------------- Scenarios -----------------------------

def scenario_adoption(target_share: float, horizon_m: int, panel: pd.DataFrame) -> Dict:
    """
    Push CBDC share of (CBDC+UPI) value to 'target_share' in 'horizon_m' months by linear ramp from last obs.
    Returns projected CBDC and UPI values (keeping total digital value growth as recent trend).
    """
    if not {"digital_value_cbdc","digital_value_upi"}.issubset(panel.columns):
        return {"note":"insufficient value columns for adoption scenario"}
    tail = panel.tail(6).copy()
    share0 = float(tail["digital_value_cbdc"].sum() / (tail["digital_value_cbdc"].sum() + tail["digital_value_upi"].sum() + 1e-9))
    total_digital_last = float(panel["digital_value_cbdc"].iloc[-1] + panel["digital_value_upi"].iloc[-1])
    # recent growth rate of total digital value
    g = panel["digital_value_cbdc"].add(panel["digital_value_upi"], fill_value=0.0)
    g_rate = float(dlog(g).tail(3).mean()) if g.dropna().shape[0] >= 4 else 0.0
    shares = np.linspace(share0, target_share, max(2, horizon_m))
    proj = []
    total = total_digital_last
    for i, s in enumerate(shares, start=1):
        total *= np.exp(g_rate)  # compound each month
        cbdc = total * s
        upi  = total * (1.0 - s)
        proj.append({"t": i, "share_cbdc_value": float(s), "digital_value_total": float(total),
                     "digital_value_cbdc": float(cbdc), "digital_value_upi": float(upi)})
    return {"share0": share0, "g_rate": g_rate, "projection": proj}

def scenario_deposit_migration(panel: pd.DataFrame, bank: pd.DataFrame,
                               cbdc_rate_pct: float, elas_dep: float,
                               share_movable: float) -> Dict:
    """
    Simple deposit migration model:
      ΔDeposits = share_movable * deposits * elas_dep * max(0, r_cbdc − r_deposit)
    Bank NIM impact (approx):
      Replace migrated deposits with alt funding @ r_alt → ΔNII ≈ −ΔD * (r_alt − r_deposit)
      ΔNIM ≈ ΔNII / assets
    """
    if bank.empty:
        return {"note":"bank balance sheet not provided"}
    last = bank.dropna().tail(1).iloc[0]
    deposits = float(last.get("deposits_inr", np.nan))
    assets   = float(last.get("assets_inr", np.nan))
    r_dep    = float(last.get("deposit_rate_pct", np.nan))/100.0
    r_alt    = float(last.get("alt_funding_cost_pct", r_dep*1.15))/100.0 if pd.notna(last.get("alt_funding_cost_pct", np.nan)) else r_dep*1.15
    r_cbdc   = cbdc_rate_pct/100.0
    if not np.isfinite(deposits) or not np.isfinite(assets) or not np.isfinite(r_dep):
        return {"note":"missing deposits/assets/rates"}
    spread = max(0.0, r_cbdc - r_dep)
    migr = share_movable * deposits * elas_dep * spread
    dNII = - migr * (r_alt - r_dep)
    dNIM_pp = (dNII / assets) * 100.0  # percentage points
    return {"deposits_inr": deposits, "assets_inr": assets, "r_dep": r_dep, "r_alt": r_alt,
            "r_cbdc": r_cbdc, "share_movable": share_movable, "elas_dep": elas_dep,
            "migrated_inr": migr, "delta_nii_inr": dNII, "delta_nim_pp": dNIM_pp}

def scenario_payment_costs(panel: pd.DataFrame, costs: pd.DataFrame, horizon_m: int, target_share: float) -> Dict:
    """
    System cost delta over horizon if CBDC share of digital rises to target_share.
    Unit costs: from costs.csv (channels CBDC/UPI/CASH). If missing, returns note.
    """
    if costs.empty:
        return {"note":"costs.csv not provided"}
    # map unit costs
    def uc(ch): 
        r = costs[costs["channel"]==ch]
        return float(r["unit_cost_inr"].iloc[0]) if not r.empty else np.nan
    uc_cbdc, uc_upi, uc_cash = uc("CBDC"), uc("UPI"), uc("CASH")
    if not np.isfinite(uc_cbdc) or not np.isfinite(uc_upi):
        return {"note":"unit costs for CBDC/UPI required"}
    # recent total digital txn counts
    if not {"cbdc_tx_count","upi_tx_count"}.issubset(panel.columns):
        return {"note":"tx_count columns missing"}
    last_cbdc = float(panel["cbdc_tx_count"].iloc[-1])
    last_upi  = float(panel["upi_tx_count"].iloc[-1])
    total = last_cbdc + last_upi
    # linear ramp of share to target over horizon; hold total tx count flat (conservative)
    shares = np.linspace(last_cbdc/total if total>0 else 0.0, target_share, max(2,horizon_m))
    savings = []
    for i, s in enumerate(shares, start=1):
        cbdc = total * s
        upi  = total * (1.0 - s)
        base_cost = upi * uc_upi + cbdc * uc_cbdc  # assuming migration only within digital
        # counterfactual if no shift (stay at initial share):
        s0 = shares[0]; cbdc0 = total * s0; upi0 = total * (1.0 - s0)
        cf_cost = upi0 * uc_upi + cbdc0 * uc_cbdc
        savings.append({"t": i, "net_savings_inr": float(cf_cost - base_cost)})
    return {"unit_costs":{"CBDC": uc_cbdc, "UPI": uc_upi, "CASH": uc_cash}, "savings_path": savings}

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    cbdc: str
    upi: Optional[str]
    cash: Optional[str]
    rates: Optional[str]
    liquidity: Optional[str]
    bank: Optional[str]
    events: Optional[str]
    costs: Optional[str]
    freq: str
    lags: int
    windows: str
    s_grid: str
    target_share: float
    elas_dep: float
    share_movable: float
    cbdc_rate_pct: float
    horizon_m: int
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RBI e₹ (Digital Rupee) adoption & scenarios")
    ap.add_argument("--cbdc", required=True)
    ap.add_argument("--upi", default="")
    ap.add_argument("--cash", default="")
    ap.add_argument("--rates", default="")
    ap.add_argument("--liquidity", default="")
    ap.add_argument("--bank", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--costs", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","daily"])
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--s_grid", default="1.5,3.0,5.0")
    ap.add_argument("--target_share", type=float, default=0.10)
    ap.add_argument("--elas_dep", type=float, default=0.25)
    ap.add_argument("--share_movable", type=float, default=0.30)
    ap.add_argument("--cbdc_rate_pct", type=float, default=2.0)
    ap.add_argument("--horizon_m", type=int, default=12)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_einr")
    ap.add_argument("--min_obs", type=int, default=24)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq = "M" if args.freq.startswith("m") else "D"

    CBDC = load_cbdc(args.cbdc, freq=freq)
    UPI  = load_upi(args.upi, freq=freq) if args.upi else pd.DataFrame()
    CASH = load_cash(args.cash, freq=freq) if args.cash else pd.DataFrame()
    RATES= load_rates(args.rates, freq=freq) if args.rates else pd.DataFrame()
    LIQ  = load_liq(args.liquidity, freq=freq) if args.liquidity else pd.DataFrame()
    BANK = load_bank(args.bank, freq=freq) if args.bank else pd.DataFrame()
    EVTS = load_events(args.events, freq=freq) if args.events else pd.DataFrame()
    COST = load_costs(args.costs) if args.costs else pd.DataFrame()

    # Date filters
    if args.start:
        start = pd.to_datetime(args.start)
        for df in [CBDC, UPI, CASH, RATES, LIQ, BANK]:
            if not df.empty:
                df.drop(df[df["date"] < start].index, inplace=True)
    if args.end:
        end = pd.to_datetime(args.end)
        for df in [CBDC, UPI, CASH, RATES, LIQ, BANK]:
            if not df.empty:
                df.drop(df[df["date"] > end].index, inplace=True)

    PANEL, mapping = build_panel(freq=freq, CBDC=CBDC, UPI=UPI, CASH=CASH, RATES=RATES, LIQ=LIQ, BANK=BANK)
    PANEL.to_csv(outdir / "panel.csv", index=False)

    # Rolling & lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL = rolling_stats(PANEL, mapping, windows)
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)
    LL = leadlag_tables(PANEL, mapping, lags=int(args.lags))
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Cannibalization regression
    REG = cannibalization_reg(PANEL, mapping, L=int(args.lags), min_obs=int(args.min_obs))
    if not REG.empty: REG.to_csv(outdir / "cannibalization_reg.csv", index=False)

    # S-curve(s): wallets_retail and merchants_active if available (monthly frequency works best)
    S_CURVE = pd.DataFrame()
    sgrid = [float(x.strip()) for x in args.s_grid.split(",") if x.strip()]
    out_params = []
    for col in ["wallets_retail","merchants_active"]:
        if col in PANEL.columns and PANEL[col].dropna().shape[0] >= 8:
            params, fit = logistic_fit(PANEL.set_index("date")[col], sgrid)
            if params:
                fit["series"] = col
                S_CURVE = pd.concat([S_CURVE, fit], ignore_index=True)
                params["series"] = col
                out_params.append(params | {"series": col})
    if not S_CURVE.empty: S_CURVE.to_csv(outdir / "s_curve_fit.csv", index=False)

    # Event study
    ES = event_study(PANEL, EVTS, window=max(6, int(args.lags)//2)) if not EVTS.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Scenarios
    SCN_ROWS = []

    # Adoption push
    scn_adopt = scenario_adoption(target_share=float(args.target_share), horizon_m=int(args.horizon_m), panel=PANEL)
    if "projection" in scn_adopt:
        for r in scn_adopt["projection"]:
            SCN_ROWS.append({"scenario":"adoption_push", **r})
    # Deposit migration
    scn_dep = scenario_deposit_migration(PANEL, BANK, cbdc_rate_pct=float(args.cbdc_rate_pct),
                                         elas_dep=float(args.elas_dep), share_movable=float(args.share_movable))
    if "migrated_inr" in scn_dep:
        SCN_ROWS.append({"scenario":"deposit_migration", **{k:v for k,v in scn_dep.items() if k not in {"note"}}})
    # Payments costs
    scn_cost = scenario_payment_costs(PANEL, COST, horizon_m=int(args.horizon_m), target_share=float(args.target_share)) if not COST.empty else {"note":"no_costs"}
    if "savings_path" in scn_cost:
        for r in scn_cost["savings_path"]:
            SCN_ROWS.append({"scenario":"payments_costs", **r, "unit_costs_cbdc": scn_cost["unit_costs"]["CBDC"], "unit_costs_upi": scn_cost["unit_costs"]["UPI"]})

    SCN = pd.DataFrame(SCN_ROWS)
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    latest = PANEL.tail(1).iloc[0]
    summary = {
        "date_range": {"start": str(PANEL["date"].min().date()), "end": str(PANEL["date"].max().date())},
        "freq": args.freq,
        "cols": list(PANEL.columns),
        "latest": {
            "date": str(latest["date"].date()),
            "cbdc_tx_count": float(latest.get("cbdc_tx_count", np.nan)) if "cbdc_tx_count" in PANEL.columns else None,
            "cbdc_tx_value_inr": float(latest.get("cbdc_tx_value_inr", np.nan)) if "cbdc_tx_value_inr" in PANEL.columns else None,
            "upi_tx_count": float(latest.get("upi_tx_count", np.nan)) if "upi_tx_count" in PANEL.columns else None,
            "upi_tx_value_inr": float(latest.get("upi_tx_value_inr", np.nan)) if "upi_tx_value_inr" in PANEL.columns else None,
            "share_cbdc_value": float(latest.get("share_cbdc_value", np.nan)) if "share_cbdc_value" in PANEL.columns else None
        },
        "rolling_windows": windows,
        "leadlag_drivers": LL["driver"].unique().tolist() if not LL.empty else [],
        "reg_terms": REG["var"].tolist() if not REG.empty else [],
        "s_curve_series": [p["series"] for p in out_params] if out_params else [],
        "scenarios_included": SCN["scenario"].unique().tolist() if not SCN.empty else [],
        "notes": {
            "adoption_share0": scn_adopt.get("share0", None) if isinstance(scn_adopt, dict) else None,
            "adoption_g_rate": scn_adopt.get("g_rate", None) if isinstance(scn_adopt, dict) else None,
            "deposit_migration_note": scn_dep.get("note", None) if "note" in scn_dep else None,
            "costs_note": scn_cost.get("note", None) if isinstance(scn_cost, dict) and "note" in scn_cost else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        cbdc=args.cbdc, upi=(args.upi or None), cash=(args.cash or None), rates=(args.rates or None),
        liquidity=(args.liquidity or None), bank=(args.bank or None), events=(args.events or None),
        costs=(args.costs or None), freq=args.freq, lags=int(args.lags), windows=args.windows,
        s_grid=args.s_grid, target_share=float(args.target_share), elas_dep=float(args.elas_dep),
        share_movable=float(args.share_movable), cbdc_rate_pct=float(args.cbdc_rate_pct),
        horizon_m=int(args.horizon_m), start=(args.start or None), end=(args.end or None),
        outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== e₹ (Digital Rupee) Analytics ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Freq: {summary['freq']}")
    if summary["latest"]["share_cbdc_value"] is not None:
        print(f"Latest CBDC share of digital value: {summary['latest']['share_cbdc_value']:.2%}")
    if not REG.empty:
        cum = REG[REG["var"].str.contains("_cum_0..L")]
        if not cum.empty:
            c = cum.iloc[0]
            print(f"Cannibalization cum β (CBDC → UPI): {c['coef']:+.3f} (t={c['t_stat']:+.2f})")
    if out_params:
        for p in out_params:
            print(f"S-curve {p['series']}: K≈{p['K']:.0f}, k={p['k']:.3f}, t0={p['t0']:.1f}")
    if "migrated_inr" in scn_dep:
        print(f"Deposit migration (₹): {scn_dep['migrated_inr']:,.0f} | ΔNIM (pp): {scn_dep['delta_nim_pp']:+.3f}")
    if "projection" in scn_adopt:
        end_share = scn_adopt['projection'][-1]['share_cbdc_value']
        print(f"Adoption scenario: target share in {args.horizon_m}m → {end_share:.1%}")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
