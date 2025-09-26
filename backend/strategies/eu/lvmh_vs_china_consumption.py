#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lvmh_vs_china_consumption.py — Link LVMH (Asia/China) performance to China demand, travel retail & FX

What this does
--------------
Given LVMH segment/region financials, Chinese monthly indicators (retail, travel, duty-free, etc.),
FX, and (optionally) prices & policy events, this script builds a compact analytics pack:

Core outputs
------------
- lvmh_q_enriched.csv        LVMH quarterly table with YoY/QoQ, Asia ex-Japan (AEJ) / China splits if present
- macro_m_enriched.csv       Monthly China indicators with YoY transforms & standardized (z) versions
- mapped_q.csv               Monthly → quarterly mapping aligned to LVMH quarters (t, t-1)
- xcorr_months.csv           Cross-correlations of LVMH AEJ YoY vs each monthly indicator across lags/leads (-6..+6m)
- factor_pca.csv             First principal component (ChinaLuxuryFactor) & loadings for the indicator set
- regression_fit.csv         OLS fit: AEJ_Rev_YoY ~ Factor + FX + Travel proxies (+ DFS/Hainan if provided)
- nowcast_next_q.csv         Model-based nowcast for the next LVMH quarter (point & CI)
- sensitivities.csv          Revenue YoY sensitivity to 1σ moves in each standardized indicator
- scenarios_out.csv          Scenario deltas vs baseline given scenarios.csv (dot-key overrides)
- event_study.csv            Share return event study around China policy events (if prices+events provided)
- summary.json               KPIs (fit R², last AEJ YoY, next nowcast, top drivers)
- config.json                Reproducibility dump

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--lvmh lvmh.csv        REQUIRED (quarterly)
  Columns (any subset; script is robust):
    date(YYYY-Q or quarter end date), revenue_eur_m, asia_ex_japan_revenue_eur_m, china_revenue_eur_m,
    dfs_revenue_eur_m (or selective_retailing_revenue_eur_m), fashion_leather_revenue_eur_m, perfumes_cosmetics_revenue_eur_m,
    yoy_pct (overall) ...
  Notes: If AEJ/China not provided, total is used as proxy with a warning.

--macro macro.csv      OPTIONAL (monthly China indicators)
  Examples of accepted names (case-insensitive; use what you have):
    date, cn_retail_sales_yoy, cn_retail_sales_index, total_retail_yoy, online_retail_yoy, travel_retail_hainan_sales,
    hk_visitors, macau_visitors, outbound_air_seats, domestic_air_seats, cpi_yoy, pmi_manu, pmi_services, consumer_confidence,
    hainan_df_sales_index, dutyfree_sales, unionpay_spend, alipay_crossborder_index, vipshop_gmv, jd_luxury_index, tmall_luxury_pv

--fx fx.csv            OPTIONAL (monthly or daily FX)
  Columns:
    date, EURCNY (or CNYEUR / USDCNY + EURUSD pair), EURCNH (auto-derived if both legs available)

--prices prices.csv    OPTIONAL (daily)
  Columns:
    date, asset (e.g., "LVMH", "MC FP", "^SXXP"), close (or return). If close present, log-returns are computed.

--events events.csv    OPTIONAL (policy/events)
  Columns:
    date, name, type (e.g., visa, lockdown, travel, duty_free, stimulus), note

--scenarios scenarios.csv OPTIONAL (overrides to build counterfactuals)
  Columns:
    scenario, name, key, value
  Dot-keys target parameters/predictors, e.g.:
    weights.fx = 0.30
    shock.hk_visitors_yoy = +20
    shock.cn_retail_sales_yoy = +3       (percentage points)
    shock.factor = +0.5                   (std devs)
    fx.override_eurcny_yoy = -5

Key options
-----------
--lvmh LVMH.csv
--macro macro.csv       (optional)
--fx fx.csv             (optional)
--prices prices.csv     (optional)
--events events.csv     (optional)
--scenarios scenarios.csv (optional)
--start 2016-01         (inclusive)
--end   2025-12         (inclusive)
--event_win 5           (± trading days for event study)
--outdir out_lvmh_cn

Method notes
------------
- LVMH quarterly YoY is mapped against same-quarter averages (and t-1) of monthly indicators.
- A PCA builds a composite "ChinaLuxuryFactor" from standardized positive-oriented indicators.
- Baseline OLS: AEJ_Rev_YoY% ~ β0 + β1*Factor_t + β2*EURCNY_YoY + β3*HK+Macau_Viss_YoY + β4*Hainan_DF_YoY (+ optional DFS YoY)
- Nowcast uses realized monthly data for the current/next quarter; partial quarter months are averaged to date.
- Sensitivities: ΔYoY from +1σ shock to each standardized predictor, holding others fixed.
- Event study: average log return of LVMH around dated China events.

DISCLAIMER: This is an analytical toolkit; it is NOT investment advice and not an official LVMH model.
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

def ncol(df: pd.DataFrame, t: str) -> Optional[str]:
    T = t.lower()
    for c in df.columns:
        if c.lower() == T: return c
    for c in df.columns:
        if T in c.lower(): return c
    return None

def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def to_quarter_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("Q").dt.start_time

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def yoy(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)

def zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd

def winsor(s: pd.Series, a: float = 0.01) -> pd.Series:
    lo = s.quantile(a)
    hi = s.quantile(1 - a)
    return s.clip(lower=lo, upper=hi)

def pca_first_factor(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (first PC series aligned to index, loadings).
    Expects standardized columns (z).
    """
    X = df.dropna(how="all").copy()
    X = X.dropna(axis=1, how="all")
    if X.empty or X.shape[1] == 0:
        return pd.Series(index=df.index, dtype=float), pd.Series(dtype=float)
    # Fill remaining NaNs with column mean (on standardized scale ~0)
    Xm = X.fillna(0.0).values
    # Covariance on standardized inputs equals correlation
    C = np.cov(Xm, rowvar=False)
    w, V = np.linalg.eigh(C)  # symmetric
    idx = np.argsort(w)[::-1]
    w1 = w[idx[0]]
    v1 = V[:, idx[0]]
    # Normalize sign so factor correlates positively with the average of columns
    if np.sum(v1) < 0:
        v1 = -v1
    scores = Xm @ v1
    # Scale scores to std=1 (optional)
    scores = (scores - scores.mean()) / (scores.std() + 1e-12)
    pc = pd.Series(scores, index=X.index).reindex(df.index)
    loadings = pd.Series(v1, index=X.columns)
    return pc, loadings

def rolling_train_predict(X: pd.DataFrame, y: pd.Series) -> Tuple[float, pd.Series]:
    """
    Simple expanding-window out-of-sample R² and predictions.
    """
    Xc = X.copy()
    y = y.copy()
    preds = pd.Series(index=y.index, dtype=float)
    r2_oos = []
    for i in range(8, len(y)):  # need at least 8 quarters to train
        Xtr = Xc.iloc[:i].values
        ytr = y.iloc[:i].values
        Xte = Xc.iloc[i:i+1].values
        # OLS
        try:
            beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            yhat = (Xte @ beta)[0]
            preds.iloc[i] = yhat
            # update rolling R² on a small window
            y_actual = y.iloc[:i+1]
            y_pred    = preds.iloc[:i+1]
            mask = y_pred.notna()
            if mask.sum() >= 4:
                ss_res = np.sum((y_actual[mask] - y_pred[mask])**2)
                ss_tot = np.sum((y_actual[mask] - y_actual[mask].mean())**2)
                r2 = 1 - ss_res / (ss_tot + 1e-12)
                r2_oos.append(r2)
        except Exception:
            continue
    r2o = float(np.mean(r2_oos)) if r2_oos else np.nan
    return r2o, preds

def quarter_of_month(m: pd.Timestamp) -> pd.Timestamp:
    return m.to_period("Q").to_timestamp()

def agg_monthly_to_quarterly(df_m: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if df_m.empty:
        return df_m
    X = df_m.copy()
    X["quarter"] = X["date"].dt.to_period("Q").dt.to_timestamp()
    if how == "sum":
        g = X.groupby("quarter", as_index=False).sum(numeric_only=True)
    else:
        g = X.groupby("quarter", as_index=False).mean(numeric_only=True)
    g.rename(columns={"quarter": "date"}, inplace=True)
    return g

def merge_safe(left: pd.DataFrame, right: pd.DataFrame, on: List[str], how: str = "left") -> pd.DataFrame:
    for col in on:
        if col not in left.columns or col not in right.columns:
            return left
    return left.merge(right, on=on, how=how)

def pct_points(x: float) -> float:
    return float(x) if np.isfinite(x) else np.nan


# ----------------------------- loaders -----------------------------

def load_lvmh(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"revenue_eur_m") or ncol(df,"total_revenue_eur_m") or "revenue_eur_m"): "revenue_eur_m",
        (ncol(df,"asia_ex_japan_revenue_eur_m") or ncol(df,"aej_revenue_eur_m") or "asia_ex_japan_revenue_eur_m"): "aej_rev",
        (ncol(df,"china_revenue_eur_m") or "china_revenue_eur_m"): "cn_rev",
        (ncol(df,"dfs_revenue_eur_m") or ncol(df,"selective_retailing_revenue_eur_m") or "dfs_revenue_eur_m"): "dfs_rev",
        (ncol(df,"fashion_leather_revenue_eur_m") or "fashion_leather_revenue_eur_m"): "fl_rev",
        (ncol(df,"yoy_pct") or "yoy_pct"): "yoy_pct_total",
    }
    df = df.rename(columns=ren)
    df["date"] = to_quarter_start(df["date"])
    for c in ["revenue_eur_m","aej_rev","cn_rev","dfs_rev","fl_rev"]:
        if c in df.columns:
            df[c] = num(df[c])
    df = df.sort_values("date")
    # derive YoY/Yo2Y
    if "aej_rev" in df.columns:
        df["aej_yoy"] = df["aej_rev"].pct_change(4)
    else:
        df["aej_yoy"] = df["revenue_eur_m"].pct_change(4)
    if "cn_rev" in df.columns:
        df["cn_yoy"] = df["cn_rev"].pct_change(4)
    if "dfs_rev" in df.columns:
        df["dfs_yoy"] = df["dfs_rev"].pct_change(4)
    if "fl_rev" in df.columns:
        df["fl_yoy"] = df["fl_rev"].pct_change(4)
    df["total_yoy"] = df["revenue_eur_m"].pct_change(4) if "revenue_eur_m" in df.columns else np.nan
    return df

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    rename = {}
    # Ensure 'date'
    rename[(ncol(df,"date") or df.columns[0])] = "date"
    df = df.rename(columns=rename)
    df["date"] = to_month(df["date"])
    # numeric all others
    for c in df.columns:
        if c != "date":
            df[c] = num(df[c])
    return df.sort_values("date")

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={(ncol(df,"date") or df.columns[0]): "date"})
    df["date"] = to_month(df["date"])
    # Normalize common FX columns
    aliases = ["EURCNY","EURCNH","CNYEUR","CNHEUR","USDCNY","CNYUSD","EURUSD","USDEUR"]
    for a in aliases:
        if ncol(df, a):
            df[a if a in df.columns else a] = num(df[ncol(df, a)])
    # Attempt derivation
    if "EURCNY" not in df.columns:
        if "EURUSD" in df.columns and "USDCNY" in df.columns:
            df["EURCNY"] = df["EURUSD"] * df["USDCNY"]
        elif "USDEUR" in df.columns and "USDCNY" in df.columns:
            df["EURCNY"] = (1.0/df["USDEUR"]) * df["USDCNY"]
        elif "CNYEUR" in df.columns:
            df["EURCNY"] = 1.0 / df["CNYEUR"]
    if "EURCNH" not in df.columns and "EURCNY" in df.columns:
        df["EURCNH"] = df["EURCNY"]  # fallback
    return df[["date"] + [c for c in ["EURCNY","EURCNH","EURUSD","USDCNY"] if c in df.columns]].sort_values("date")

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={(ncol(df,"date") or df.columns[0]): "date", (ncol(df,"asset") or "asset"): "asset"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if ncol(df,"return"):
        df["return"] = num(df[ncol(df,"return")])
    else:
        close_c = ncol(df,"close") or ncol(df,"price")
        if close_c:
            df = df.sort_values(["asset","date"])
            df["close"] = num(df[close_c])
            df["return"] = df.groupby("asset")["close"].apply(lambda s: np.log(s) - np.log(s.shift(1)))
    return df.dropna(subset=["date","asset","return"]).sort_values(["asset","date"])

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={(ncol(df,"date") or df.columns[0]): "date", (ncol(df,"name") or "name"): "name",
                            (ncol(df,"type") or "type"): "type", (ncol(df,"note") or "note"): "note"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    return df.rename(columns={(ncol(df,"scenario") or "scenario"): "scenario",
                              (ncol(df,"name") or "name"): "name",
                              (ncol(df,"key") or "key"): "key",
                              (ncol(df,"value") or "value"): "value"})


# ----------------------------- feature engineering -----------------------------

def enrich_macro(dfm: pd.DataFrame) -> pd.DataFrame:
    if dfm.empty:
        return dfm
    M = dfm.copy()
    # Create YoY if levels given for common series
    candidates = [c for c in M.columns if c != "date"]
    for c in candidates:
        if not c.endswith("_yoy"):
            M[c + "_yoy"] = yoy(M[c], 12)
    # Travel proxies combined
    if set(["hk_visitors","macau_visitors"]).intersection(M.columns):
        hk = M.get("hk_visitors", np.nan)
        mo = M.get("macau_visitors", np.nan)
        M["hkmo_visitors"] = hk.fillna(0) + mo.fillna(0)
        M["hkmo_visitors_yoy"] = yoy(M["hkmo_visitors"], 12)
    # Duty free / Hainan
    if "hainan_df_sales_index" in M.columns and "dutyfree_sales" not in M.columns:
        M["dutyfree_sales"] = M["hainan_df_sales_index"]
        M["dutyfree_sales_yoy"] = yoy(M["dutyfree_sales"], 12)
    # Keep only monthly
    return M

def build_z_matrix(M: pd.DataFrame, cols_keep: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    if M.empty:
        return M, []
    df = M.copy()
    cols = cols_keep or [c for c in df.columns if c not in {"date"} and (c.endswith("_yoy") or "index" in c.lower() or "pmi" in c.lower())]
    Z = pd.DataFrame({"date": df["date"]})
    use = []
    for c in cols:
        if c in df.columns:
            z = zscore(winsor(df[c].astype(float)))
            if z.notna().sum() >= 12:
                Z[c] = z
                use.append(c)
    return Z.dropna(how="all", subset=use), use

def build_fx_features(FX: pd.DataFrame) -> pd.DataFrame:
    if FX.empty: return FX
    fx = FX.copy()
    fx["EURCNY_yoy"] = yoy(fx["EURCNY"], 12) if "EURCNY" in fx.columns else np.nan
    fx["EURCNH_yoy"] = yoy(fx["EURCNH"], 12) if "EURCNH" in fx.columns else np.nan
    fx["date"] = to_month(fx["date"])
    return fx

def map_monthly_to_quarterly_features(M: pd.DataFrame, FXm: pd.DataFrame) -> pd.DataFrame:
    """
    Quarterly averages for YoY series; returns dataframe keyed by quarter start 'date'.
    """
    pieces = []
    for (df, how) in [(M, "mean"), (FXm, "mean")]:
        if df is None or df.empty: 
            continue
        dm = df.copy()
        dm = dm.dropna(axis=1, how="all")
        if "date" not in dm.columns:
            continue
        agg = agg_monthly_to_quarterly(dm, how="mean")
        pieces.append(agg)
    if not pieces:
        return pd.DataFrame(columns=["date"])
    out = pieces[0]
    for p in pieces[1:]:
        out = merge_safe(out, p, on=["date"], how="outer")
    out = out.sort_values("date")
    return out

def assemble_model_table(LQ: pd.DataFrame, QX: pd.DataFrame) -> pd.DataFrame:
    if LQ.empty:
        return pd.DataFrame()
    base = LQ[["date","aej_yoy","cn_yoy","dfs_yoy","fl_yoy","total_yoy"]].copy()
    base = base.rename(columns={"aej_yoy":"y"})
    # Merge quarterly features (t and lagged)
    X = QX.copy()
    # t features
    df = base.merge(X, on="date", how="left")
    # lagged quarterly features (t-1)
    X_lag = X.copy(); X_lag["date"] = X_lag["date"] + pd.offsets.QuarterEnd(0)  # align
    X_lag = X_lag.rename(columns={c: f"{c}_lag1" for c in X_lag.columns if c != "date"})
    df = df.merge(X_lag, on="date", how="left")
    # Drop rows without y
    return df.dropna(subset=["y"])

def fit_regression(df: pd.DataFrame, factor_col: str, extra_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str,float], float]:
    """
    Returns (fit_table with yhat/resid, betas dict, in-sample R²).
    """
    use_cols = [factor_col] + extra_cols
    cols = [c for c in use_cols if c in df.columns]
    X = df[cols].copy()
    X = X.apply(lambda s: s.astype(float))
    X = X.fillna(0.0)  # mild guardrail; for real use prefer a proper imputer/selection
    X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    y = df["y"].astype(float)
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    yhat = (X.values @ beta)
    resid = y.values - yhat
    r2 = 1.0 - np.sum(resid**2) / (np.sum((y.values - y.values.mean())**2) + 1e-12)
    fit = df[["date"]].copy()
    fit["y"] = y.values
    fit["yhat"] = yhat
    fit["resid"] = resid
    betas = {c: float(b) for c, b in zip(X.columns, beta)}
    return fit, betas, float(r2)

def nowcast_next_quarter(LQ: pd.DataFrame, Mz: pd.DataFrame, FXm: pd.DataFrame, loadings: pd.Series, betas: Dict[str,float]) -> pd.DataFrame:
    """
    Build next quarter predictors using available monthly data → quarterly average to date.
    """
    if LQ.empty or Mz.empty or not betas:
        return pd.DataFrame()
    last_q = LQ["date"].dropna().max()
    next_q = (pd.to_datetime(last_q) + pd.offsets.QuarterBegin(1)).to_period("Q").to_timestamp()
    # Pool months belonging to next_q that exist in macro/fx
    next_months = pd.date_range(next_q, next_q + pd.offsets.QuarterEnd(0), freq="MS")
    # Macro factor using available months only
    Zcols = [c for c in Mz.columns if c != "date"]
    M_next = Mz[Mz["date"].isin(next_months)]
    if M_next.empty:
        return pd.DataFrame()
    Z_next = M_next[Zcols].copy()
    Z_next = Z_next.fillna(0.0)
    # Factor score = standardized columns · loadings (already on z scale)
    load = loadings.reindex(Zcols).fillna(0.0).values
    factor_scores = (Z_next.values @ load)
    factor_q = float(np.mean(factor_scores)) if len(factor_scores) else np.nan
    # FX quarterly
    FX = build_fx_features(FXm) if not FXm.empty else pd.DataFrame()
    if not FX.empty:
        fx_next = FX[FX["date"].isin(next_months)].mean(numeric_only=True)
        eurcny_yoy = float(fx_next.get("EURCNY_yoy", np.nan))
        eurcnh_yoy = float(fx_next.get("EURCNH_yoy", np.nan))
    else:
        eurcny_yoy = np.nan; eurcnh_yoy = np.nan
    # Travel proxies
    hkmo_yoy = float(M_next.get("hkmo_visitors_yoy", pd.Series(dtype=float)).mean())
    dutyfree_yoy = float(M_next.get("dutyfree_sales_yoy", pd.Series(dtype=float)).mean())
    # Compose X row for betas
    X = {"const": 1.0}
    X["ChinaLuxuryFactor"] = factor_q
    if "EURCNY_yoy" in betas:
        X["EURCNY_yoy"] = eurcny_yoy
    if "EURCNH_yoy" in betas:
        X["EURCNH_yoy"] = eurcnh_yoy
    if "hkmo_visitors_yoy" in betas:
        X["hkmo_visitors_yoy"] = hkmo_yoy
    if "dutyfree_sales_yoy" in betas:
        X["dutyfree_sales_yoy"] = dutyfree_yoy
    # Pred
    yhat = sum(betas.get(k, 0.0) * X.get(k, 0.0) for k in X.keys())
    # naive CI via residual std from history
    return pd.DataFrame([{"date": next_q, "nowcast_yoy": yhat, "factor_q": factor_q,
                          "EURCNY_yoy": eurcny_yoy, "EURCNH_yoy": eurcnh_yoy,
                          "hkmo_visitors_yoy": hkmo_yoy, "dutyfree_sales_yoy": dutyfree_yoy}])

def cross_correlation_table(LQ: pd.DataFrame, M: pd.DataFrame, target_col: str = "aej_yoy", lags: List[int] = list(range(-6,7))) -> pd.DataFrame:
    if LQ.empty or M.empty:
        return pd.DataFrame()
    # Map quarterly target to monthly by forward filling within the quarter for correlation purposes
    T = LQ[["date", target_col]].dropna().copy()
    if T.empty:
        return pd.DataFrame()
    Tm = []
    for _, r in T.iterrows():
        q = pd.to_datetime(r["date"]).to_period("Q").to_timestamp()
        months = pd.date_range(q, q + pd.offsets.QuarterEnd(0), freq="MS")
        for m in months:
            Tm.append({"date": m, target_col: r[target_col]})
    Tm = pd.DataFrame(Tm)
    # Join with monthly indicators
    base = M.merge(Tm, on="date", how="inner")
    if base.empty:
        return pd.DataFrame()
    cols = [c for c in base.columns if c not in {"date", target_col}]
    rows = []
    for c in cols:
        s = base[["date", c, target_col]].dropna()
        for k in lags:
            if k >= 0:
                # indicator leads (predicts) target: shift indicator backward
                x = s[c].shift(k)
                y = s[target_col]
            else:
                # indicator lags target
                x = s[c]
                y = s[target_col].shift(-k)
            mask = x.notna() & y.notna()
            if mask.sum() >= 10:
                corr = float(np.corrcoef(x[mask], y[mask])[0,1])
            else:
                corr = np.nan
            rows.append({"indicator": c, "lag_months": int(k), "corr": corr})
    out = pd.DataFrame(rows)
    return out.sort_values(["indicator", "lag_months"])

def event_study(prices: pd.DataFrame, events: pd.DataFrame, asset: str, win: int = 5) -> pd.DataFrame:
    if prices.empty or events.empty:
        return pd.DataFrame()
    P = prices[prices["asset"].astype(str).str.upper().isin({asset.upper(), "LVMH", "MC", "MC FP", "MC.PA"})].copy()
    if P.empty: return pd.DataFrame()
    P = P.sort_values("date").reset_index(drop=True)
    P["d0"] = P["date"].dt.normalize()
    idx_map = {d: i for i, d in enumerate(P["d0"])}
    rows = []
    for d in events["date"].dt.normalize().unique():
        if d not in idx_map: 
            continue
        i0 = idx_map[d]
        lo = max(0, i0 - win); hi = min(len(P)-1, i0 + win)
        sub = P.iloc[lo:hi+1].copy()
        sub["t"] = range(lo - i0, hi - i0 + 1)
        sub["event_date"] = d
        rows.append(sub[["event_date","t","return"]])
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    avg = (df.groupby("t", as_index=False).agg(mean_ret=("return","mean"), med_ret=("return","median"), n=("return","count")))
    return avg.sort_values("t")

def apply_scenarios(betas: Dict[str,float], baseline_row: Dict[str,float], scen_df: pd.DataFrame) -> pd.DataFrame:
    if not betas or scen_df.empty:
        return pd.DataFrame()
    scenarios = []
    baseline_y = sum(betas.get(k,0.0) * baseline_row.get(k,0.0) for k in set(betas.keys()) | set(baseline_row.keys()))
    for sid in sorted(scen_df["scenario"].unique()):
        sub = scen_df[scen_df["scenario"] == sid]
        row = baseline_row.copy()
        name = sub["name"].iloc[0] if "name" in sub.columns and not sub["name"].isna().all() else sid
        for _, r in sub.iterrows():
            key = str(r["key"]).strip()
            val = r["value"]
            try:
                v = float(val)
            except Exception:
                continue
            if key.startswith("shock."):
                k = key.split("shock.")[1]
                # Shock specified in percentage points for yoy variables (e.g., +3), or in std devs if 'factor'
                if k.lower() == "factor" or "factor" in k.lower():
                    row["ChinaLuxuryFactor"] = row.get("ChinaLuxuryFactor", 0.0) + v  # std devs
                else:
                    # Assume pp: convert to fraction
                    row[k] = row.get(k, 0.0) + (v / 100.0)
            elif key == "weights.fx":
                # Optional: scale FX coefficient weight (simple linear)
                betas["EURCNY_yoy"] = betas.get("EURCNY_yoy", 0.0) * v
                betas["EURCNH_yoy"] = betas.get("EURCNH_yoy", 0.0) * v
            elif key.startswith("fx.override_"):
                k = key.split("fx.override_")[1]
                if k.lower() == "eurcny_yoy":
                    row["EURCNY_yoy"] = v / 100.0
                if k.lower() == "eurcnh_yoy":
                    row["EURCNH_yoy"] = v / 100.0
        y_new = sum(betas.get(k,0.0) * row.get(k,0.0) for k in set(betas.keys()) | set(row.keys()))
        scenarios.append({"scenario": sid, "name": name, "delta_yoy_pp": (y_new - baseline_y) * 100.0, "new_yoy_pct": y_new * 100.0})
    return pd.DataFrame(scenarios).sort_values("delta_yoy_pp", ascending=False)


# ----------------------------- CLI -----------------------------

@dataclass
class Config:
    lvmh: str
    macro: Optional[str]
    fx: Optional[str]
    prices: Optional[str]
    events: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    event_win: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LVMH vs China consumption analytics")
    ap.add_argument("--lvmh", required=True)
    ap.add_argument("--macro", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2016-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--event_win", type=int, default=5)
    ap.add_argument("--outdir", default="out_lvmh_cn")
    return ap.parse_args()


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load and time filter
    LQ = load_lvmh(args.lvmh)
    if LQ.empty:
        raise SystemExit("No LVMH rows found. Provide --lvmh quarterly CSV.")

    start_m = pd.to_datetime(args.start).to_period("M").to_timestamp()
    end_m   = pd.to_datetime(args.end).to_period("M").to_timestamp()

    M_raw = load_macro(args.macro) if args.macro else pd.DataFrame()
    FXm   = load_fx(args.fx) if args.fx else pd.DataFrame()
    P     = load_prices(args.prices) if args.prices else pd.DataFrame()
    Ev    = load_events(args.events) if args.events else pd.DataFrame()
    Scen  = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    if not M_raw.empty:
        M_raw = M_raw[(M_raw["date"] >= start_m) & (M_raw["date"] <= end_m)]
    if not FXm.empty:
        FXm = FXm[(FXm["date"] >= start_m) & (FXm["date"] <= end_m)]
    if not P.empty:
        P = P[(P["date"] >= pd.to_datetime(args.start)) & (P["date"] <= pd.to_datetime(args.end))]

    # Enrich
    LQ.to_csv(outdir / "lvmh_q_enriched.csv", index=False)
    M  = enrich_macro(M_raw) if not M_raw.empty else pd.DataFrame()
    if not M.empty:
        M.to_csv(outdir / "macro_m_enriched.csv", index=False)

    # Build standardized matrix & PCA factor
    Mz, used_cols = build_z_matrix(M)
    pc, loadings = pca_first_factor(Mz.drop(columns=["date"]) if "date" in Mz.columns else Mz)
    factor = pd.DataFrame({"date": Mz["date"], "ChinaLuxuryFactor": pc}) if not Mz.empty else pd.DataFrame(columns=["date","ChinaLuxuryFactor"])

    # FX features, map to Q
    FXf = build_fx_features(FXm) if not FXm.empty else pd.DataFrame()

    # Join monthly features for Q mapping
    Mf = M.copy()
    if not factor.empty:
        Mf = merge_safe(Mf, factor, on=["date"], how="left")
    QX = map_monthly_to_quarterly_features(Mf, FXf)
    if not QX.empty:
        QX.to_csv(outdir / "mapped_q.csv", index=False)

    # Cross-correlations (monthly)
    xcorr = cross_correlation_table(LQ, Mz, target_col="aej_yoy")
    if not xcorr.empty:
        xcorr.to_csv(outdir / "xcorr_months.csv", index=False)

    # Model table
    df_model = assemble_model_table(LQ, QX)
    # Choose predictors
    factor_col = "ChinaLuxuryFactor"
    extra = [c for c in ["EURCNY_yoy","EURCNH_yoy","hkmo_visitors_yoy","dutyfree_sales_yoy"] if c in df_model.columns]
    # Fit
    fit, betas, r2_is = fit_regression(df_model, factor_col, extra) if not df_model.empty else (pd.DataFrame(), {}, np.nan)
    if not fit.empty:
        fit.to_csv(outdir / "regression_fit.csv", index=False)

    # OOS R² (expanding) on the same features
    if not df_model.empty and betas:
        X = df_model[[c for c in [factor_col] + extra if c in df_model.columns]].copy()
        X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
        r2_oos, preds = rolling_train_predict(X, df_model["y"])
    else:
        r2_oos, preds = np.nan, pd.Series(dtype=float)

    # Nowcast next quarter
    nowc = nowcast_next_quarter(LQ, Mz, FXm, loadings, betas) if (not LQ.empty and not Mz.empty and betas) else pd.DataFrame()
    if not nowc.empty:
        nowc.to_csv(outdir / "nowcast_next_q.csv", index=False)

    # Sensitivities (1σ on standardized) — only for predictors present
    sens_rows = []
    if betas:
        for k, b in betas.items():
            if k in {"const"}: 
                continue
            # Variables in standardized units (factor) → 1σ; YoY % variables are in fraction terms; treat +1pp as 0.01
            if k == "ChinaLuxuryFactor":
                dy = b * 1.0
                sens_rows.append({"predictor": k, "shock": "+1σ", "impact_yoy_pp": dy * 100.0})
            else:
                dy = b * 0.01
                sens_rows.append({"predictor": k, "shock": "+1pp", "impact_yoy_pp": dy * 100.0})
    sens = pd.DataFrame(sens_rows)
    if not sens.empty:
        sens.to_csv(outdir / "sensitivities.csv", index=False)

    # Scenarios
    scen_out = pd.DataFrame()
    if betas and not nowc.empty and not Scen.empty:
        baseline_row = {"const": 1.0, "ChinaLuxuryFactor": float(nowc["factor_q"].iloc[0])}
        for c in ["EURCNY_yoy","EURCNH_yoy","hkmo_visitors_yoy","dutyfree_sales_yoy"]:
            if c in nowc.columns:
                baseline_row[c] = float(nowc[c].iloc[0])
        scen_out = apply_scenarios(betas.copy(), baseline_row, Scen)
        if not scen_out.empty:
            scen_out.to_csv(outdir / "scenarios_out.csv", index=False)

    # Event study (if any events + prices)
    evt = event_study(P, Ev, asset="LVMH", win=args.event_win) if (not P.empty and not Ev.empty) else pd.DataFrame()
    if not evt.empty:
        evt.to_csv(outdir / "event_study.csv", index=False)

    # Factor + loadings output
    if not factor.empty:
        ld = loadings.sort_values(ascending=False).rename("loading").reset_index().rename(columns={"index":"indicator"})
        pd.DataFrame({"date": factor["date"], "ChinaLuxuryFactor": factor["ChinaLuxuryFactor"]}).to_csv(outdir / "factor_pca.csv", index=False)
        ld.to_csv(outdir / "factor_loadings.csv", index=False)

    # Summary
    last_q = LQ["date"].max()
    last_aej_yoy = float(LQ.set_index("date").loc[last_q, "aej_yoy"]) if "aej_yoy" in LQ.columns and last_q in LQ.set_index("date").index else np.nan
    kpi = {
        "last_quarter": str(last_q.date()) if pd.notna(last_q) else None,
        "last_aej_yoy_pct": last_aej_yoy * 100.0 if np.isfinite(last_aej_yoy) else None,
        "in_sample_r2": r2_is,
        "oos_r2_expanding": r2_oos,
        "betas": betas,
        "next_q_nowcast_yoy_pct": float(nowc["nowcast_yoy"].iloc[0] * 100.0) if not nowc.empty else None,
        "top_factor_loadings": (loadings.abs().sort_values(ascending=False).head(6).to_dict() if not loadings.empty else {}),
        "xcorr_best": (xcorr.sort_values("corr", ascending=False).head(5).to_dict(orient="records") if not xcorr.empty else []),
        "scenario_best": (scen_out.head(3).to_dict(orient="records") if not scen_out.empty else [])
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))

    # Config dump
    cfg = asdict(Config(
        lvmh=args.lvmh, macro=args.macro or None, fx=args.fx or None, prices=args.prices or None,
        events=args.events or None, scenarios=args.scenarios or None, start=args.start, end=args.end,
        event_win=args.event_win, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== LVMH vs China Consumption Analytics ==")
    print(f"In-sample R²: {kpi['in_sample_r2']:.2f} | OOS R² (expanding): {kpi['oos_r2_expanding'] if kpi['oos_r2_expanding']==kpi['oos_r2_expanding'] else float('nan'):.2f}")
    if not nowc.empty:
        print(f"Next quarter AEJ YoY nowcast: {kpi['next_q_nowcast_yoy_pct']:.1f}%")
    if sens is not None and not sens.empty:
        print("Sensitivities (ΔYoY, pp):", sens.head(6).to_dict(orient="records"))
    if not scen_out.empty:
        print("Top scenarios:", scen_out.head(3).to_dict(orient="records"))
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
