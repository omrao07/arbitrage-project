#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telecom_tariff_cycles.py — Price hikes, churn, elasticity, and revenue scenarios
-------------------------------------------------------------------------------

What this does
==============
A compact toolkit to study **telecom tariff cycles** (price hikes/cuts) and their effects on:
  • Subscribers/net adds, churn, ARPU, revenue
  • Cross-operator competitive responses
  • Elasticities via distributed-lag regressions (with Newey–West SEs)
  • Event studies around tariff changes
  • Forward scenarios (what-if price hike %) with revenue/EBITDA impact

It works with operator-level monthly (preferred) or quarterly data. All joins are by date (& operator
where appropriate); headers are flexible and case-insensitive.

Inputs (CSV; any subset is fine)
--------------------------------
--subs subs.csv            [REQUIRED]
  Columns (any subset):
    date, operator,
    subs_total[, prepaid_subs, postpaid_subs],
    churn_rate_pct[, churn_rate], net_adds

--prices prices.csv        [RECOMMENDED]
  Columns (any subset):
    date, operator,
    price_index[, avg_tariff_inr, headline_plan_inr],
    promo_discount_pct

--arpu arpu.csv            [OPTIONAL]
  Columns:
    date, operator, arpu_inr[, prepaid_arpu, postpaid_arpu]

--mnp mnp.csv              [OPTIONAL]
  Columns:
    date, operator, mnp_in, mnp_out

--traffic traffic.csv      [OPTIONAL]
  Columns:
    date, operator, data_usage_gb[, voice_min], cost_per_gb_inr

--network network.csv      [OPTIONAL]
  Columns:
    date, operator, coverage_4g_pct[, coverage_5g_pct], capex_inr

--macro macro.csv          [OPTIONAL]
  Columns:
    date, cpi_all[, cpi_telecom], income_proxy_idx

--events events.csv        [OPTIONAL]  (policy or operator price actions)
  Columns:
    date[, operator], label[, type]      # type e.g., HIKE / CUT / PROMO

Key CLI
-------
--lags 6                        Max lag for elasticity regressions
--windows 3,6,12                Rolling corr windows (periods)
--freq monthly|quarterly        Output frequency (default monthly)
--horizon_m 12                  Scenario projection horizon (months)
--hike_pct 15                   Scenario: immediate price hike (%) for targeted operators
--apply_to "all"                Comma-separated operator list or "all"
--comp_response_pct 50          % of own hike competitors mirror (cross pass-through)
--churn_bump_pp 0.5             One-off churn bump (pp) in month 0 after hike
--ebitda_margin0 35             Baseline EBITDA margin % (for simple impact calc)
--start / --end                 Date filters (YYYY-MM-DD)
--outdir out_telco              Output directory
--min_obs 18                    Min obs for per-operator regressions

Outputs
-------
- operator_panel.csv            Long panel (date, operator, derived metrics)
- market_panel.csv              Aggregated (market) panel
- rolling_stats.csv             Rolling corr: price vs dlog_subs / churn / dlog_arpu
- leadlag_corr.csv              Lead–lag tables for key pairs
- elasticity_subs.csv          Per-operator distributed-lag regression (subs growth)
- elasticity_arpu.csv          Per-operator distributed-lag regression (arpu growth)
- elasticity_churn.csv         Per-operator regression for churn
- event_study.csv               Event deltas around labeled tariff events
- scenarios.csv                 Forward projections for subs, ARPU, revenue & EBITDA
- summary.json                  Headline diagnostics
- config.json                   Run configuration

Notes/assumptions
-----------------
• Δlog variables are first-diff logs; churn_rate_pct is % of base per month (0–100).
• Price proxy precedence: price_index → avg_tariff_inr → headline_plan_inr (converted to index).
• Competitor price = average of other operators' price index at date t.
• Elasticities: dlog(subs)_t on Σ Δlog(price)_{t−l} plus Δlog(price_comp) & controls.
• Scenario is a simple accounting model using estimated cumulative elasticities (if available).

DISCLAIMER
----------
This is research tooling with simplifying assumptions. Validate mappings, sampling, and robustness
before investment decisions.
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

def to_period_end(s: pd.Series, freq_rule: str) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    if freq_rule.startswith("Q"):
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

def leadlag_table(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            xx = x.shift(k); yy = y
        else:
            xx = x; yy = y.shift(-k)
        c = xx.corr(yy)
        rows.append({"lag": k, "corr": float(c) if c==c else np.nan})
    return pd.DataFrame(rows)

# OLS + HAC (Newey–West)
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

def load_subs(path: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    op  = ncol(df, "operator","op","brand")
    subs= ncol(df, "subs_total","subscribers","subs","users")
    churn=ncol(df, "churn_rate_pct","churn_rate","churn")
    neta= ncol(df, "net_adds","netadds","net_additions")
    if not (dt and op and subs):
        raise ValueError("subs.csv must include date, operator, and subs_total.")
    df = df.rename(columns={dt:"date", op:"operator", subs:"subs_total"})
    df["date"] = to_period_end(df["date"], freq)
    df["subs_total"] = safe_num(df["subs_total"])
    if churn:
        df = df.rename(columns={churn:"churn_rate_pct"})
        df["churn_rate_pct"] = safe_num(df["churn_rate_pct"])
    if neta:
        df = df.rename(columns={neta:"net_adds"})
        df["net_adds"] = safe_num(df["net_adds"])
    return df.sort_values(["operator","date"])

def load_prices(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    op  = ncol(df, "operator","op","brand")
    pidx= ncol(df, "price_index","tariff_index","p_index")
    avg = ncol(df, "avg_tariff_inr","avg_recharge_inr","average_tariff")
    head= ncol(df, "headline_plan_inr","headline_recharge_inr")
    promo=ncol(df, "promo_discount_pct","discount_pct")
    if not (dt and op): raise ValueError("prices.csv needs date and operator.")
    df = df.rename(columns={dt:"date", op:"operator"})
    df["date"] = to_period_end(df["date"], freq)
    if pidx:
        df = df.rename(columns={pidx:"price_index"}); df["price_index"] = safe_num(df["price_index"])
    elif avg or head:
        # build index from nominal plan price
        base = safe_num(df[avg] if avg else df[head])
        df["price_index"] = base / base.dropna().iloc[0] * 100.0
    if promo:
        df = df.rename(columns={promo:"promo_discount_pct"})
        df["promo_discount_pct"] = safe_num(df["promo_discount_pct"])
    return df[["date","operator","price_index"] + (["promo_discount_pct"] if "promo_discount_pct" in df.columns else [])].sort_values(["operator","date"])

def load_arpu(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    op = ncol(df, "operator","op","brand")
    ar = ncol(df, "arpu_inr","arpu","avg_rev_per_user")
    if not (dt and op and ar): raise ValueError("arpu.csv needs date, operator, arpu.")
    df = df.rename(columns={dt:"date", op:"operator", ar:"arpu_inr"})
    df["date"] = to_period_end(df["date"], freq)
    df["arpu_inr"] = safe_num(df["arpu_inr"])
    return df.sort_values(["operator","date"])

def load_mnp(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    op = ncol(df, "operator","op","brand")
    min_ = ncol(df, "mnp_in","mnpin","port_in")
    mout = ncol(df, "mnp_out","mnpout","port_out")
    if not (dt and op): raise ValueError("mnp.csv needs date, operator.")
    df = df.rename(columns={dt:"date", op:"operator"})
    df["date"] = to_period_end(df["date"], freq)
    if min_: df = df.rename(columns={min_:"mnp_in"}); df["mnp_in"] = safe_num(df["mnp_in"])
    if mout: df = df.rename(columns={mout:"mnp_out"}); df["mnp_out"] = safe_num(df["mnp_out"])
    if "mnp_in" in df.columns and "mnp_out" in df.columns:
        df["mnp_net"] = df["mnp_in"] - df["mnp_out"]
    return df.sort_values(["operator","date"])

def load_traffic(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    op = ncol(df, "operator","op","brand")
    gb = ncol(df, "data_usage_gb","data_gb","gb")
    vm = ncol(df, "voice_min","voice_minutes")
    cg = ncol(df, "cost_per_gb_inr","cost_gb_inr")
    if not (dt and op): raise ValueError("traffic.csv needs date, operator.")
    df = df.rename(columns={dt:"date", op:"operator"})
    df["date"] = to_period_end(df["date"], freq)
    for c in [gb, vm, cg]:
        if c: df[c] = safe_num(df[c])
    ren = {}
    if gb: ren[gb] = "data_usage_gb"
    if vm: ren[vm] = "voice_min"
    if cg: ren[cg] = "cost_per_gb_inr"
    df = df.rename(columns=ren)
    return df.sort_values(["operator","date"])

def load_network(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    op = ncol(df, "operator","op","brand")
    c4 = ncol(df, "coverage_4g_pct","cov_4g_pct")
    c5 = ncol(df, "coverage_5g_pct","cov_5g_pct")
    cap= ncol(df, "capex_inr","capex")
    if not (dt and op): raise ValueError("network.csv needs date, operator.")
    df = df.rename(columns={dt:"date", op:"operator"})
    df["date"] = to_period_end(df["date"], freq)
    for c in [c4,c5,cap]:
        if c: df[c] = safe_num(df[c])
    ren = {}
    if c4: ren[c4] = "coverage_4g_pct"
    if c5: ren[c5] = "coverage_5g_pct"
    if cap: ren[cap] = "capex_inr"
    df = df.rename(columns=ren)
    return df.sort_values(["operator","date"])

def load_macro(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_period_end(df["date"], freq)
    ren = {}
    for c,inm in [("cpi_all","cpi_all"),("cpi_telecom","cpi_telecom"),("income_proxy_idx","income_proxy_idx")]:
        cc = ncol(df, c)
        if cc: ren[cc]=inm
    df = df.rename(columns=ren)
    for c in df.columns:
        if c!="date": df[c] = safe_num(df[c])
    return df.sort_values("date")

def load_events(path: Optional[str], freq: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt  = ncol(df, "date") or df.columns[0]
    lab = ncol(df, "label","event","name") or "label"
    typ = ncol(df, "type","kind") or None
    op  = ncol(df, "operator","op","brand") or None
    df = df.rename(columns={dt:"date", lab:"label"})
    if typ: df = df.rename(columns={typ:"type"})
    if op:  df = df.rename(columns={op:"operator"})
    df["date"] = to_period_end(df["date"], freq)
    if "type" not in df.columns: df["type"] = ""
    return df[["date"] + (["operator"] if "operator" in df.columns else []) + ["label","type"]].dropna(subset=["date"]).sort_values("date")


# ----------------------------- build panels -----------------------------

def build_operator_panel(freq_rule: str,
                         SUBS: pd.DataFrame, PRICES: pd.DataFrame, ARPU: pd.DataFrame,
                         MNP: pd.DataFrame, TRAF: pd.DataFrame, NET: pd.DataFrame, MACRO: pd.DataFrame) -> pd.DataFrame:
    # Merge operator-level dfs
    ops = SUBS[["date","operator","subs_total"]].copy()
    for d in [PRICES, ARPU, MNP, TRAF, NET]:
        if not d.empty:
            ops = ops.merge(d, on=["date","operator"], how="left")
    # Add macro (date-only)
    if not MACRO.empty:
        ops = ops.merge(MACRO, on="date", how="left")

    ops = ops.sort_values(["operator","date"])
    yo = 12 if freq_rule.startswith("M") else 4

    # Price transforms
    if "price_index" in ops.columns:
        ops["dlog_price"] = ops.groupby("operator")["price_index"].apply(dlog).reset_index(level=0, drop=True)

    # Competitor price index (peer average excluding self)
    if "price_index" in ops.columns:
        tmp = ops[["date","operator","price_index"]].copy()
        g = tmp.groupby("date")["price_index"]
        tot = g.transform("sum"); cnt = g.transform("count")
        ops["price_index_comp"] = (tot - ops["price_index"]) / (cnt - 1).replace(0,np.nan)
        ops["dlog_price_comp"] = ops.groupby("operator")["price_index_comp"].apply(dlog).reset_index(level=0, drop=True)

    # ARPU transforms
    if "arpu_inr" in ops.columns:
        ops["dlog_arpu"] = ops.groupby("operator")["arpu_inr"].apply(dlog).reset_index(level=0, drop=True)
        ops["yoy_arpu"] = ops.groupby("operator")["arpu_inr"].apply(lambda s: yoy(s, yo)).reset_index(level=0, drop=True)

    # Subscriber transforms
    ops["dlog_subs"] = ops.groupby("operator")["subs_total"].apply(dlog).reset_index(level=0, drop=True)
    ops["yoy_subs"]  = ops.groupby("operator")["subs_total"].apply(lambda s: yoy(s, yo)).reset_index(level=0, drop=True)
    if "net_adds" not in ops.columns:
        ops["net_adds"] = ops.groupby("operator")["subs_total"].diff()
    if "churn_rate_pct" in ops.columns:
        ops["d_churn_pp"] = ops.groupby("operator")["churn_rate_pct"].diff()

    # Revenue proxy (monthly): subs × arpu
    if "arpu_inr" in ops.columns:
        ops["revenue_inr"] = ops["subs_total"] * ops["arpu_inr"]

    # Network & traffic transforms
    if "coverage_4g_pct" in ops.columns: ops["d_coverage_4g_pp"] = ops.groupby("operator")["coverage_4g_pct"].diff()
    if "coverage_5g_pct" in ops.columns: ops["d_coverage_5g_pp"] = ops.groupby("operator")["coverage_5g_pct"].diff()
    if "promo_discount_pct" in ops.columns: ops["d_promo_pp"] = ops.groupby("operator")["promo_discount_pct"].diff()

    # Market aggregate (for later)
    return ops

def market_panel(OP: pd.DataFrame, freq_rule: str) -> pd.DataFrame:
    if OP.empty: return pd.DataFrame()
    df = OP.groupby("date", as_index=False).agg({
        "subs_total":"sum",
        "revenue_inr":"sum" if "revenue_inr" in OP.columns else "first",
        "data_usage_gb":"sum" if "data_usage_gb" in OP.columns else "first",
        "capex_inr":"sum" if "capex_inr" in OP.columns else "first"
    })
    yo = 12 if freq_rule.startswith("M") else 4
    df["dlog_market_subs"] = dlog(df["subs_total"])
    if "revenue_inr" in df.columns: df["dlog_market_rev"] = dlog(df["revenue_inr"])
    if "data_usage_gb" in df.columns: df["dlog_market_data"] = dlog(df["data_usage_gb"])
    df["hh"] = 1  # placeholder
    return df


# ----------------------------- diagnostics -----------------------------

def rolling_and_leadlag(OP: pd.DataFrame, windows: List[int], lags: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if OP.empty: return pd.DataFrame(), pd.DataFrame()
    rows_roll = []
    rows_ll = []
    for op_name, g in OP.groupby("operator"):
        idx = g.set_index("date")
        # Rolling
        pairs = []
        if "dlog_price" in idx.columns: 
            pairs += [("dlog_price","dlog_subs"), ("dlog_price","dlog_arpu"), ("dlog_price","d_churn_pp")]
        if "dlog_price_comp" in idx.columns:
            pairs += [("dlog_price_comp","dlog_subs")]
        for x,y in pairs:
            if x in idx.columns and y in idx.columns:
                for w,tag in zip(windows, ["short","med","long"]):
                    rows_roll.append({"operator": op_name, "x": x, "y": y, "window": w,
                                      "tag": tag, "corr": float(roll_corr(idx[x], idx[y], w).iloc[-1]) if len(idx)>=w else np.nan})
        # Lead–lag (price → subs/ARPU/churn)
        if "dlog_price" in idx.columns:
            for dep in ["dlog_subs","dlog_arpu","d_churn_pp"]:
                if dep in idx.columns:
                    tab = leadlag_table(idx["dlog_price"], idx[dep], lags)
                    tab["operator"] = op_name; tab["driver"] = "dlog_price"; tab["dep"] = dep
                    rows_ll.append(tab)
        if "dlog_price_comp" in idx.columns and "dlog_subs" in idx.columns:
            tab = leadlag_table(idx["dlog_price_comp"], idx["dlog_subs"], lags)
            tab["operator"] = op_name; tab["driver"] = "dlog_price_comp"; tab["dep"] = "dlog_subs"
            rows_ll.append(tab)
    ROLL = pd.DataFrame(rows_roll)
    LL = pd.concat(rows_ll, ignore_index=True) if rows_ll else pd.DataFrame(columns=["lag","corr","operator","driver","dep"])
    return ROLL, LL


# ----------------------------- D-lag regressions (elasticities) -----------------------------

def dlag_regression(OP: pd.DataFrame, dep_col: str, L: int, min_obs: int) -> pd.DataFrame:
    """
    Per-operator regression with HAC SEs:
      dep_t = α + Σ_{l=0..L} β_l * dlog_price_{t−l} + Σ_{l=0..L} γ_l * dlog_price_comp_{t−l}
              + δ * d_promo_pp + φ * d_coverage_4g_pp + θ * d_coverage_5g_pp + ε_t
    dep_col in {"dlog_subs","dlog_arpu","d_churn_pp"}.
    """
    out_rows = []
    ops = OP["operator"].dropna().unique().tolist()
    for op in ops:
        g = OP[OP["operator"]==op].copy()
        if dep_col not in g.columns: continue
        dep = g[dep_col]
        Xparts = [pd.Series(1.0, index=g.index, name="const")]
        names = ["const"]
        avail_main = []
        for key in ["dlog_price","dlog_price_comp"]:
            if key in g.columns:
                avail_main.append(key)
                for l in range(0, L+1):
                    nm = f"{key}_l{l}"
                    Xparts.append(g[key].shift(l).rename(nm)); names.append(nm)
        # controls
        controls = []
        for c in ["d_promo_pp","d_coverage_4g_pp","d_coverage_5g_pp"]:
            if c in g.columns:
                controls.append(c)
                Xparts.append(g[c].rename(c)); names.append(c)
        if not avail_main:
            continue
        X = pd.concat(Xparts, axis=1)
        XY = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        if XY.shape[0] < max(min_obs, 5*X.shape[1]):
            continue
        yv = XY["dep"].values.reshape(-1,1)
        Xv = XY.drop(columns=["dep"]).values
        beta, resid, XtX_inv = ols_beta_se(Xv, yv)
        se = hac_se(Xv, resid, XtX_inv, L=max(6, L))
        for i, nm in enumerate(names):
            out_rows.append({"operator": op, "dep": dep_col, "var": nm, "coef": float(beta[i,0]),
                             "se": float(se[i]), "t_stat": float(beta[i,0]/se[i] if se[i]>0 else np.nan),
                             "lags": L, "n": int(XY.shape[0])})
        # cumulative effects
        for key in avail_main:
            idxs = [i for i,nm in enumerate(names) if nm.startswith(f"{key}_l")]
            if idxs:
                bsum = float(beta[idxs,0].sum()); ses = float(np.sqrt(np.sum(se[idxs]**2)))
                out_rows.append({"operator": op, "dep": dep_col, "var": f"{key}_cum_0..L",
                                 "coef": bsum, "se": ses,
                                 "t_stat": bsum/(ses if ses>0 else np.nan), "lags": L, "n": int(XY.shape[0])})
    return pd.DataFrame(out_rows)


# ----------------------------- event study -----------------------------

def event_study(OP: pd.DataFrame, EVENTS: pd.DataFrame, window: int=3) -> pd.DataFrame:
    """
    For each event (optionally operator-specific), compute deltas vs pre-event mean (h=-1..-window).
    Tracks: dlog_subs, d_churn_pp, dlog_arpu, dlog_price.
    """
    if EVENTS.empty: return pd.DataFrame()
    rows = []
    idx = OP.set_index(["date","operator"]).sort_index()
    for _, ev in EVENTS.iterrows():
        d0 = ev["date"]
        op = ev["operator"] if "operator" in ev and isinstance(ev.get("operator",None), str) else None
        ops = [op] if op else OP["operator"].unique().tolist()
        for oo in ops:
            # slice windows
            dates = sorted(OP["date"].unique())
            if d0 not in dates:
                # choose previous available date
                prevs = [x for x in dates if x <= d0]
                if not prevs: continue
                d0_use = prevs[-1]
            else:
                d0_use = d0
            # build a list of h from -window..+window
            for h in range(-window, window+1):
                # date offset by h months/quarters: use positional approach
                all_dates = dates
                i = all_dates.index(d0_use)
                j = i + h
                if j < 0 or j >= len(all_dates): continue
                dt = all_dates[j]
                if (dt, oo) not in idx.index: continue
                r = idx.loc[(dt, oo)]
                rec = {"event_date": d0_use, "h": h, "operator": oo,
                       "label": ev.get("label",""), "type": ev.get("type","")}
                for c in ["dlog_subs","d_churn_pp","dlog_arpu","dlog_price"]:
                    rec[c] = float(r.get(c, np.nan)) if c in r.index else np.nan
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    # convert to deltas against pre-window average (h<0)
    out = []
    for (op, d0), g in df.groupby(["operator","event_date"]):
        base = g[g["h"]<0][["dlog_subs","d_churn_pp","dlog_arpu","dlog_price"]].mean(numeric_only=True)
        for _, r in g.iterrows():
            rr = {"operator": op, "event_date": d0, "h": int(r["h"]), "label": r["label"], "type": r["type"]}
            for c in ["dlog_subs","d_churn_pp","dlog_arpu","dlog_price"]:
                val = r.get(c, np.nan); rr[f"delta_{c}"] = float(val - base.get(c, np.nan)) if pd.notna(val) else np.nan
            out.append(rr)
    return pd.DataFrame(out).sort_values(["operator","event_date","h"])


# ----------------------------- scenarios -----------------------------

def pick_cum_coef(df: pd.DataFrame, op: str, dep: str, key: str) -> Optional[float]:
    if df.empty: return None
    r = df[(df["operator"]==op) & (df["dep"]==dep) & (df["var"]==f"{key}_cum_0..L")]
    return float(r["coef"].iloc[0]) if not r.empty else None

def scenario_price_hike(OP: pd.DataFrame,
                        EL_SUBS: pd.DataFrame, EL_ARPU: pd.DataFrame, EL_CHURN: pd.DataFrame,
                        hike_pct: float, horizon_m: int, apply_ops: List[str],
                        comp_pass_pct: float, churn_bump_pp: float, ebitda_margin0: float) -> pd.DataFrame:
    """
    Apply an immediate hike at t0 (last date in panel). Use cumulative elasticities:
      Δlog(subs) ≈ β_price * Δlog(price) + β_comp * Δlog(price_comp)
      Δlog(arpu)  ≈ κ_price * Δlog(price)
      Δchurn_pp   ≈ ω_price * Δlog(price)  (monthly; add one-off bump)
    Revenue impact ≈ (subs * arpu) changes; EBITDA ≈ margin0 * revenue (approx).
    """
    if OP.empty: return pd.DataFrame()
    last_date = OP["date"].max()
    ops = sorted(OP["operator"].unique().tolist())
    apply_set = set(ops if apply_ops==["all"] else [o.strip() for o in apply_ops if o.strip() in ops])
    # Δlog(price) for own and competitor (competitor reacts partially)
    dlog_p = np.log(1.0 + hike_pct/100.0)
    dlog_p_comp = np.log(1.0 + (hike_pct*comp_pass_pct/100.0)/100.0)  # mirror % of hike, then convert to Δlog
    rows = []
    margin0 = ebitda_margin0/100.0

    # current levels
    base = OP[OP["date"]==last_date].set_index("operator")
    for op in ops:
        subs0 = float(base.loc[op]["subs_total"]) if (op in base.index and pd.notna(base.loc[op]["subs_total"])) else np.nan
        arpu0 = float(base.loc[op]["arpu_inr"]) if ("arpu_inr" in base.columns and op in base.index and pd.notna(base.loc[op]["arpu_inr"])) else np.nan
        rev0  = subs0*arpu0 if (subs0==subs0 and arpu0==arpu0) else np.nan
        # elasticities
        b_self = pick_cum_coef(EL_SUBS, op, "dlog_subs", "dlog_price")
        b_comp = pick_cum_coef(EL_SUBS, op, "dlog_subs", "dlog_price_comp")
        k_self = pick_cum_coef(EL_ARPU, op, "dlog_arpu", "dlog_price")
        w_self = pick_cum_coef(EL_CHURN, op, "d_churn_pp", "dlog_price")

        # fallbacks (if not estimated)
        if b_self is None: b_self = -0.25   # own-price elasticity (cum) default
        if b_comp is None: b_comp = +0.10   # cross elasticity
        if k_self is None: k_self = +0.70   # ARPU pass-through (cum) default
        if w_self is None: w_self = +0.10   # churn pp per log price (cum) default

        # only apply own hike to selected ops; others get competitor pass-through only
        own = dlog_p if op in apply_set else 0.0
        comp= dlog_p_comp

        # one-step impact; project flat over horizon for simplicity (level shift)
        dlog_subs = b_self*own + b_comp*comp
        dlog_arpu = k_self*own
        d_churn   = w_self*own + (churn_bump_pp if own>0 else 0.0)

        # build path
        subs = subs0; arpu = arpu0; rev = rev0
        for t in range(1, horizon_m+1):
            subs *= np.exp(dlog_subs) if subs==subs else np.nan
            arpu *= np.exp(dlog_arpu) if arpu==arpu else np.nan
            rev = subs*arpu if subs==subs and arpu==arpu else np.nan
            ebitda = rev*margin0 if rev==rev else np.nan
            rows.append({
                "operator": op, "t": t, "date0": str(last_date.date()),
                "own_hike": op in apply_set, "dlog_price": own, "dlog_price_comp": comp,
                "subs": subs, "arpu_inr": arpu, "revenue_inr": rev, "ebitda_inr": ebitda,
                "d_churn_pp_month0": d_churn if t==1 else np.nan
            })
    return pd.DataFrame(rows)


# ----------------------------- CLI / main -----------------------------

@dataclass
class Config:
    subs: str
    prices: Optional[str]
    arpu: Optional[str]
    mnp: Optional[str]
    traffic: Optional[str]
    network: Optional[str]
    macro: Optional[str]
    events: Optional[str]
    freq: str
    lags: int
    windows: str
    horizon_m: int
    hike_pct: float
    apply_to: str
    comp_response_pct: float
    churn_bump_pp: float
    ebitda_margin0: float
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Telecom tariff cycles: elasticity, churn, event study & scenarios")
    ap.add_argument("--subs", required=True)
    ap.add_argument("--prices", default="")
    ap.add_argument("--arpu", default="")
    ap.add_argument("--mnp", default="")
    ap.add_argument("--traffic", default="")
    ap.add_argument("--network", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","quarterly"])
    ap.add_argument("--lags", type=int, default=6)
    ap.add_argument("--windows", default="3,6,12")
    ap.add_argument("--horizon_m", type=int, default=12)
    ap.add_argument("--hike_pct", type=float, default=15.0)
    ap.add_argument("--apply_to", default="all")  # comma-separated ops or "all"
    ap.add_argument("--comp_response_pct", type=float, default=50.0)
    ap.add_argument("--churn_bump_pp", type=float, default=0.5)
    ap.add_argument("--ebitda_margin0", type=float, default=35.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_telco")
    ap.add_argument("--min_obs", type=int, default=18)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    freq_rule = "M" if args.freq.startswith("m") else "Q"

    SUBS = load_subs(args.subs, freq_rule)
    PRIC = load_prices(args.prices, freq_rule) if args.prices else pd.DataFrame()
    ARPU = load_arpu(args.arpu, freq_rule) if args.arpu else pd.DataFrame()
    MNP  = load_mnp(args.mnp, freq_rule) if args.mnp else pd.DataFrame()
    TRAF = load_traffic(args.traffic, freq_rule) if args.traffic else pd.DataFrame()
    NET  = load_network(args.network, freq_rule) if args.network else pd.DataFrame()
    MAC  = load_macro(args.macro, freq_rule) if args.macro else pd.DataFrame()
    EVTS = load_events(args.events, freq_rule) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        t0 = pd.to_datetime(args.start)
        for df in [SUBS, PRIC, ARPU, MNP, TRAF, NET, MAC]:
            if not df.empty:
                df.drop(df[df["date"] < t0].index, inplace=True)
    if args.end:
        t1 = pd.to_datetime(args.end)
        for df in [SUBS, PRIC, ARPU, MNP, TRAF, NET, MAC]:
            if not df.empty:
                df.drop(df[df["date"] > t1].index, inplace=True)

    OP = build_operator_panel(freq_rule, SUBS, PRIC, ARPU, MNP, TRAF, NET, MAC)
    if OP.empty: raise ValueError("After merging, operator panel is empty. Check inputs.")
    OP.to_csv(outdir / "operator_panel.csv", index=False)

    MRKT = market_panel(OP, freq_rule)
    if not MRKT.empty: MRKT.to_csv(outdir / "market_panel.csv", index=False)

    # Rolling & lead–lag
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    ROLL, LL = rolling_and_leadlag(OP, windows=windows, lags=int(args.lags))
    if not ROLL.empty: ROLL.to_csv(outdir / "rolling_stats.csv", index=False)
    if not LL.empty: LL.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Regressions (elasticities)
    EL_SUBS  = dlag_regression(OP, dep_col="dlog_subs",  L=int(args.lags), min_obs=int(args.min_obs))
    if not EL_SUBS.empty: EL_SUBS.to_csv(outdir / "elasticity_subs.csv", index=False)
    EL_ARPU  = dlag_regression(OP, dep_col="dlog_arpu",  L=int(args.lags), min_obs=int(args.min_obs)) if "dlog_arpu" in OP.columns else pd.DataFrame()
    if not EL_ARPU.empty: EL_ARPU.to_csv(outdir / "elasticity_arpu.csv", index=False)
    EL_CHURN = dlag_regression(OP, dep_col="d_churn_pp", L=int(args.lags), min_obs=int(args.min_obs)) if "d_churn_pp" in OP.columns else pd.DataFrame()
    if not EL_CHURN.empty: EL_CHURN.to_csv(outdir / "elasticity_churn.csv", index=False)

    # Event study
    ES = event_study(OP, EVTS, window=max(3, int(args.lags)//2)) if not EVTS.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Scenario
    apply_ops = [x.strip() for x in args.apply_to.split(",")] if args.apply_to else ["all"]
    SCN = scenario_price_hike(OP,
                              EL_SUBS if not EL_SUBS.empty else pd.DataFrame(),
                              EL_ARPU if not EL_ARPU.empty else pd.DataFrame(),
                              EL_CHURN if not EL_CHURN.empty else pd.DataFrame(),
                              hike_pct=float(args.hike_pct), horizon_m=int(args.horizon_m),
                              apply_ops=apply_ops, comp_pass_pct=float(args.comp_response_pct),
                              churn_bump_pp=float(args.churn_bump_pp), ebitda_margin0=float(args.ebitda_margin0))
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Summary
    latest = OP.groupby("operator").tail(1)
    # pick best lead–lag per operator for price->subs
    best_ll = {}
    if not LL.empty:
        for op in LL["operator"].unique():
            g = LL[(LL["operator"]==op) & (LL["driver"]=="dlog_price") & (LL["dep"]=="dlog_subs")].dropna(subset=["corr"])
            if not g.empty:
                row = g.iloc[g["corr"].abs().argmax()]
                best_ll[op] = {"lag": int(row["lag"]), "corr": float(row["corr"])}

    summary = {
        "date_range": {"start": str(OP["date"].min().date()), "end": str(OP["date"].max().date())},
        "freq": args.freq,
        "n_operators": int(OP["operator"].nunique()),
        "latest_snapshot": [
            {
                "operator": r["operator"],
                "subs_total": float(r.get("subs_total", np.nan)),
                "arpu_inr": float(r.get("arpu_inr", np.nan)) if "arpu_inr" in OP.columns else None,
                "churn_rate_pct": float(r.get("churn_rate_pct", np.nan)) if "churn_rate_pct" in OP.columns else None,
            } for _, r in latest.sort_values("operator").iterrows()
        ],
        "rolling_windows": windows,
        "leadlag_price_to_subs": best_ll,
        "reg_files": {
            "subs": ("elasticity_subs.csv" if not EL_SUBS.empty else None),
            "arpu": ("elasticity_arpu.csv" if not EL_ARPU.empty else None),
            "churn": ("elasticity_churn.csv" if not EL_CHURN.empty else None)
        },
        "has_event_study": (not ES.empty),
        "has_scenario": (not SCN.empty)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        subs=args.subs, prices=(args.prices or None), arpu=(args.arpu or None), mnp=(args.mnp or None),
        traffic=(args.traffic or None), network=(args.network or None), macro=(args.macro or None),
        events=(args.events or None), freq=args.freq, lags=int(args.lags), windows=args.windows,
        horizon_m=int(args.horizon_m), hike_pct=float(args.hike_pct), apply_to=args.apply_to,
        comp_response_pct=float(args.comp_response_pct), churn_bump_pp=float(args.churn_bump_pp),
        ebitda_margin0=float(args.ebitda_margin0), start=(args.start or None), end=(args.end or None),
        outdir=args.outdir, min_obs=int(args.min_obs)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Telecom Tariff Cycles ==")
    print(f"Sample: {summary['date_range']['start']} → {summary['date_range']['end']} | Freq: {summary['freq']} | Operators: {summary['n_operators']}")
    if best_ll:
        for k,v in best_ll.items():
            print(f"{k}: price→subs max |corr| at lag {v['lag']:+d} → {v['corr']:+.2f}")
    if summary["has_scenario"]:
        print(f"Scenario: {args.hike_pct:.1f}% hike | horizon {args.horizon_m}m | competitor mirror {args.comp_response_pct:.0f}%")
        print("See scenarios.csv for projected subs/ARPU/revenue/EBITDA paths.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
