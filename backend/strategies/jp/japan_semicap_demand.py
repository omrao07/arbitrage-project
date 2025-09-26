#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
japan_semicap_demand.py — Japan semi-cap demand: sales drivers, elasticities, events, scenarios & stress
-------------------------------------------------------------------------------------------------------

What this does
==============
Builds a **monthly panel** for Japanese semiconductor capital equipment (semi-cap) demand and estimates:
- Elasticities of vendor **sales/bookings** to global WFE, memory prices, AI/DC capex, book-to-bill, exports (HS 8486), and FX (USDJPY)
- Event studies around policy/controls & large vendor capex announcements
- Forward **scenario engine** (H months) with user shocks (WFE, memory, AI capex, FX, China share)
- **Stress (VaR/ES)** using Monte Carlo with joint driver covariances
- Optional **segment/company mix** snapshots

Inputs (CSV; flexible headers; monthly or higher freq, auto EOM align)
----------------------------------------------------------------------
--sales sales.csv                REQUIRED
  Columns (any subset):
    date, sales_jpy[, revenue_jpy, billings_jpy], bookings_jpy[, orders_jpy], backlog_jpy,
    vendor[, company], segment[, product]
  Notes: If multiple vendors/segments, they will be aggregated unless --by_vendor or --by_segment used.

--wfe global_wfe.csv            OPTIONAL (SEMI/WFE or vendor capex proxies)
  Columns: date, wfe_usd[, wfe_total_usd], foundry_wfe_usd, logic_wfe_usd, memory_wfe_usd, drame_wfe_usd, nand_wfe_usd

--memory memory.csv             OPTIONAL (price/indices)
  Columns: date, dram_idx[, dram_price_idx], nand_idx[, nand_price_idx]

--ai ai_capex.csv               OPTIONAL (hyperscaler/DC/AI capex)
  Columns: date, ai_capex_usd[, dc_capex_usd, accelerator_units_idx]

--devices devices.csv           OPTIONAL (downstream unit proxies)
  Columns: date, smartphone_units_m, pc_units_m, auto_units_m, electronics_pmi

--btb btb.csv                   OPTIONAL (book-to-bill)
  Columns: date, b2b[, book_to_bill]

--exports exports_8486.csv      OPTIONAL (Japan exports of HS 8486)
  Columns: date, exports_8486_usd[, jp_semicap_exports_usd], exports_to_china_usd[, china_8486_usd]

--fx fx.csv                     OPTIONAL
  Columns: date, USDJPY[, JPYUSD], JPY_REER optional

--events events.csv             OPTIONAL (policy/vendor events)
  Columns: date, type[, label, value]
    Examples: US_EXPORT_CONTROLS, CHINA_SCOPE_EXPAND, TSMC_CAPEX_UPDATE, SAMSUNG_CAPEX_UPDATE

CLI (key)
---------
--lags 3                        HAC/Newey–West lags (months)
--min_obs 36                    Minimum obs for regressions
--horizon 12                    Scenario horizon (months)
--wfe_pct 0                     % shock to global WFE
--memory_pct 0                  % shock to memory price index
--ai_capex_pct 0                % shock to AI/DC capex
--fx_pct 0                      % change in USDJPY (+ = USD↑/JPY↓)
--china_share_pp 0              ±pp change in China share of JP 8486 exports
--outdir out_semicap            Output directory
--by_vendor                     Keep vendor-level outputs (else aggregated)
--by_segment                    Keep segment-level outputs (else aggregated)
--start / --end                 Optional inclusive filters (YYYY-MM-DD)

Outputs
-------
panel.csv                       Feature matrix
elasticity_sales.csv            HAC/NW elasticities for Δlog(sales)
event_study.csv                 CARs in Δlog(sales) around events
scenarios.csv                   H-month projections under user shocks
stress_vares.csv                One-step VaR/ES of sales change (JPY)
mix_latest.csv                  Latest vendor/segment shares (if available)
summary.json, config.json       Metadata and config echo

DISCLAIMER: Research tooling with simplifying assumptions. Validate with local data before use.
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
    if df is None or df.empty: return None
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
    if x.isna().all(): return x
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

def load_sales(path: str, by_vendor: bool, by_segment: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df,"date")
    if not d: raise ValueError("sales.csv needs a date column.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    # normalize value columns
    ren = {
        ncol(df,"sales_jpy","revenue_jpy","billings_jpy"): "sales_jpy",
        ncol(df,"bookings_jpy","orders_jpy"): "bookings_jpy",
        ncol(df,"backlog_jpy"): "backlog_jpy",
        ncol(df,"vendor","company"): "vendor",
        ncol(df,"segment","product"): "segment"
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for k in ["sales_jpy","bookings_jpy","backlog_jpy"]:
        if k in df.columns: df[k] = safe_num(df[k])
    if not by_vendor and "vendor" in df.columns: df.drop(columns=["vendor"], inplace=True)
    if not by_segment and "segment" in df.columns: df.drop(columns=["segment"], inplace=True)
    # aggregate to chosen grain
    keys = ["date"] + [c for c in ["vendor","segment"] if c in df.columns]
    valcols = [c for c in ["sales_jpy","bookings_jpy","backlog_jpy"] if c in df.columns]
    df = df.groupby(keys, as_index=False)[valcols].sum()
    return df

def load_wfe(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); w = ncol(df,"wfe_usd","wfe_total_usd")
    if not d or not w: raise ValueError("global_wfe.csv needs date and wfe_usd.")
    df = df.rename(columns={d:"date", w:"wfe_usd"})
    df["date"] = eom(df["date"])
    for c in ["foundry_wfe_usd","logic_wfe_usd","memory_wfe_usd","drame_wfe_usd","nand_wfe_usd","wfe_usd"]:
        c0 = ncol(df,c)
        if c0 and c0 != c: df = df.rename(columns={c0:c})
        if c in df.columns: df[c] = safe_num(df[c])
    return df

def load_memory(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); dr = ncol(df,"dram_idx","dram_price_idx","dram"); nd = ncol(df,"nand_idx","nand_price_idx","nand")
    if not d: raise ValueError("memory.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    if dr: df = df.rename(columns={dr:"dram_idx"})
    if nd: df = df.rename(columns={nd:"nand_idx"})
    for k in ["dram_idx","nand_idx"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_ai(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); a = ncol(df,"ai_capex_usd","dc_capex_usd","datacenter_capex_usd")
    if not d or not a: raise ValueError("ai_capex.csv needs date and ai_capex_usd.")
    df = df.rename(columns={d:"date", a:"ai_capex_usd"})
    df["date"] = eom(df["date"])
    df["ai_capex_usd"] = safe_num(df["ai_capex_usd"])
    acc = ncol(df,"accelerator_units_idx","ai_units_idx")
    if acc: df = df.rename(columns={acc:"accelerator_units_idx"}); df["accelerator_units_idx"] = safe_num(df["accelerator_units_idx"])
    return df

def load_devices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date")
    if not d: raise ValueError("devices.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    ren = {
        ncol(df,"smartphone_units_m","smartphones_m"): "smartphone_units_m",
        ncol(df,"pc_units_m","pcs_m"): "pc_units_m",
        ncol(df,"auto_units_m","autos_m"): "auto_units_m",
        ncol(df,"electronics_pmi","elec_pmi","pmi_electronics"): "electronics_pmi"
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for k in ["smartphone_units_m","pc_units_m","auto_units_m","electronics_pmi"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_btb(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); b = ncol(df,"b2b","book_to_bill")
    if not d or not b: raise ValueError("btb.csv needs date and b2b.")
    df = df.rename(columns={d:"date", b:"b2b"})
    df["date"] = eom(df["date"])
    df["b2b"] = safe_num(df["b2b"])
    return df

def load_exports(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"exports_8486_usd","jp_semicap_exports_usd","exports_total_usd")
    if not d or not t: raise ValueError("exports_8486.csv needs date and exports_8486_usd.")
    df = df.rename(columns={d:"date", t:"exports_8486_usd"})
    df["date"] = eom(df["date"])
    df["exports_8486_usd"] = safe_num(df["exports_8486_usd"])
    c = ncol(df,"exports_to_china_usd","china_8486_usd","exports_china_usd")
    if c:
        df = df.rename(columns={c:"exports_to_china_usd"})
        df["exports_to_china_usd"] = safe_num(df["exports_to_china_usd"])
        df["china_share"] = df["exports_to_china_usd"] / df["exports_8486_usd"].replace(0,np.nan)
    return df

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); u = ncol(df,"USDJPY","usdjpy"); j = ncol(df,"JPYUSD","jpyusd"); r = ncol(df,"JPY_REER","reer")
    if not d: raise ValueError("fx.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    if u:
        df["USDJPY"] = safe_num(df[u])
    elif j:
        df["USDJPY"] = 1.0/safe_num(df[j])
    else:
        num = [c for c in df.columns if c!="date"]
        if not num: raise ValueError("fx.csv must include USDJPY/JPYUSD.")
        df["USDJPY"] = safe_num(df[num[0]])
    if r: df = df.rename(columns={r:"JPY_REER"})
    df["r_fx_usdjpy"] = dlog(df["USDJPY"])
    if "JPY_REER" in df.columns: df["r_jpy_reer"] = dlog(df["JPY_REER"])
    return df

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"type"); v = ncol(df,"value"); l = ncol(df,"label","event")
    if not d: raise ValueError("events.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    if t: df = df.rename(columns={t:"type"})
    if v: df = df.rename(columns={v:"value"})
    if l: df = df.rename(columns={l:"label"})
    if "type" not in df.columns: df["type"] = "EVENT"
    if "label" not in df.columns: df["label"] = df["type"]
    if "value" not in df.columns:
        df["value"] = 0.0
        df.loc[df["type"].str.contains("EASE|EXPAND|CAPEX_UP|LAUNCH", case=False, na=False), "value"] = +1.0
        df.loc[df["type"].str.contains("TIGHT|RESTRICT|CAPEX_DOWN|BAN", case=False, na=False), "value"] = -1.0
    df["value"] = safe_num(df["value"])
    return df.sort_values("date")


# ----------------------------- panel construction -----------------------------

def aggregate_sales(S: pd.DataFrame) -> pd.DataFrame:
    """Sum across vendor/segment if present, otherwise pass-through."""
    keys = ["date"]
    for c in ["vendor","segment"]:
        if c in S.columns: 
            # keep for later mixes but return both aggregated and disagg
            pass
    agg = S.groupby("date", as_index=False)[[c for c in ["sales_jpy","bookings_jpy","backlog_jpy"] if c in S.columns]].sum()
    return agg

def build_panel(S: pd.DataFrame, W: pd.DataFrame, M: pd.DataFrame, A: pd.DataFrame,
                D: pd.DataFrame, B: pd.DataFrame, X: pd.DataFrame, FX: pd.DataFrame,
                EV: pd.DataFrame) -> pd.DataFrame:
    P = aggregate_sales(S)
    for df in [W,M,A,D,B,X,FX]:
        if not df.empty:
            P = P.merge(df, on="date", how="left")
    # event policy index (sum of values per month)
    if not EV.empty:
        pol = EV.groupby("date", as_index=False)["value"].sum().rename(columns={"value":"policy_index"})
        P = P.merge(pol, on="date", how="left")
    # derived growth rates
    P = P.sort_values("date")
    if "sales_jpy" in P.columns: P["dlog_sales"] = winsor(dlog(P["sales_jpy"]))
    if "bookings_jpy" in P.columns: P["dlog_bookings"] = winsor(dlog(P["bookings_jpy"]))
    if "backlog_jpy" in P.columns: P["dlog_backlog"] = winsor(dlog(P["backlog_jpy"]))
    for k in ["wfe_usd","foundry_wfe_usd","logic_wfe_usd","memory_wfe_usd","drame_wfe_usd","nand_wfe_usd",
              "dram_idx","nand_idx","ai_capex_usd","accelerator_units_idx",
              "smartphone_units_m","pc_units_m","auto_units_m","electronics_pmi",
              "b2b","exports_8486_usd","china_share"]:
        if k in P.columns: P["dlog_"+k] = winsor(dlog(P[k]))
    # month for seasonality dummies (optional)
    P["month"] = P["date"].dt.month
    return P

# ----------------------------- elasticities -----------------------------

def regress_elasticities(P: pd.DataFrame, target: str, L: int, min_obs: int) -> pd.DataFrame:
    """
    Δlog(sales) or Δlog(bookings) on drivers with HAC/NW SEs.
    """
    yname = target
    if P.empty or yname not in P.columns: return pd.DataFrame()
    df = P.dropna(subset=[yname]).copy()
    # build X
    Xcols = [c for c in [
        "dlog_wfe_usd","dlog_foundry_wfe_usd","dlog_logic_wfe_usd","dlog_memory_wfe_usd",
        "dlog_dram_idx","dlog_nand_idx",
        "dlog_ai_capex_usd","dlog_accelerator_units_idx",
        "dlog_smartphone_units_m","dlog_pc_units_m","dlog_auto_units_m","dlog_electronics_pmi",
        "dlog_b2b",
        "dlog_exports_8486_usd","dlog_china_share",
        "r_fx_usdjpy","r_jpy_reer",
        "policy_index"
    ] if c in P.columns]
    if not Xcols: return pd.DataFrame()
    X = df[Xcols].values
    # add const
    X = np.hstack([np.ones((len(df),1)), X])
    names = ["const"] + Xcols
    y = df[[yname]].values
    if len(df) < max(min_obs, 10 + X.shape[1]): 
        return pd.DataFrame()
    beta, resid, XTX_inv = ols_beta_resid(X, y)
    se = hac_se(X, resid, XTX_inv, L=max(4, L))
    rows = []
    for i, nm in enumerate(names):
        b = float(beta[i,0]); s = float(se[i]); t = b/s if s>0 else np.nan
        rows.append({"target": yname, "var": nm, "coef": b, "se": s, "t_stat": t, "n": int(len(df)), "lags": int(L)})
    return pd.DataFrame(rows).sort_values("var")

def pick_coef(EL: pd.DataFrame, var: str, default: float) -> float:
    r = EL[EL["var"]==var]
    return float(r["coef"].iloc[0]) if not r.empty else float(default)


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

def event_study(P: pd.DataFrame, EV: pd.DataFrame, window: int=2) -> pd.DataFrame:
    if P.empty or EV.empty or "dlog_sales" not in P.columns: return pd.DataFrame()
    ser = P.set_index("date")["dlog_sales"].dropna().sort_index()
    rows = []
    for _, e in EV.iterrows():
        dt = pd.Timestamp(e["date"])
        rows.append({"event_date": str(dt.date()), "type": e["type"],
                     "CAR_dlog_sales": car(ser, dt, pre=window, post=window)})
    return pd.DataFrame(rows).sort_values(["event_date","type"])


# ----------------------------- scenarios -----------------------------

def run_scenarios(P: pd.DataFrame, EL: pd.DataFrame,
                  wfe_pct: float, memory_pct: float, ai_capex_pct: float,
                  fx_pct: float, china_share_pp: float, horizon: int) -> pd.DataFrame:
    """
    Log-linear projection from latest month:
      Δlog(sales) ≈ sum_i beta_i * Δx_i
      Apply same Δ each month (can be changed to decay if needed).
    """
    if P.empty or "sales_jpy" not in P.columns: return pd.DataFrame()
    last = P.tail(1).iloc[0]
    base = float(last["sales_jpy"])
    if not np.isfinite(base) or base <= 0: return pd.DataFrame()

    # Fallback priors if regressions not available
    b_wfe  = pick_coef(EL, "dlog_wfe_usd", default=+0.8)
    b_mem  = pick_coef(EL, "dlog_dram_idx", default=+0.2) + pick_coef(EL, "dlog_nand_idx", default=+0.1)
    b_ai   = pick_coef(EL, "dlog_ai_capex_usd", default=+0.15)
    b_fx   = pick_coef(EL, "r_fx_usdjpy", default=+0.20)  # USD↑/JPY↓ helps JPY sales for exporters
    b_ch   = pick_coef(EL, "dlog_china_share", default=+0.10)

    dv = b_wfe * np.log1p(wfe_pct/100.0) \
       + b_mem * np.log1p(memory_pct/100.0) \
       + b_ai  * np.log1p(ai_capex_pct/100.0) \
       + b_fx  * np.log1p(fx_pct/100.0) \
       + b_ch  * (china_share_pp/100.0)  # approx

    rows = []
    lvl = base
    for h in range(1, int(horizon)+1):
        lvl = float(lvl * np.exp(dv))
        rows.append({"h_month": h, "sales_jpy": lvl,
                     "wfe_pct": wfe_pct, "memory_pct": memory_pct,
                     "ai_capex_pct": ai_capex_pct, "fx_pct": fx_pct,
                     "china_share_pp": china_share_pp})
    return pd.DataFrame(rows)

# ----------------------------- stress (VaR/ES) -----------------------------

def stress_var_es(P: pd.DataFrame, EL: pd.DataFrame, n_sims: int=20000) -> pd.DataFrame:
    """
    One-month sales distribution using historical driver covariances:
      drivers = [dlog_wfe_usd, dlog_dram_idx, dlog_ai_capex_usd, r_fx_usdjpy]
    """
    need = [c for c in ["dlog_wfe_usd","dlog_dram_idx","dlog_ai_capex_usd","r_fx_usdjpy"] if c in P.columns]
    if not need or "sales_jpy" not in P.columns: return pd.DataFrame()
    R = P[need].dropna()
    if R.shape[0] < 24: return pd.DataFrame()
    mu = R.mean().values
    cov = R.cov().values
    try:
        L = np.linalg.cholesky(cov + 1e-12*np.eye(len(need)))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        L = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12)))
    rng = np.random.default_rng(42)
    Z = rng.standard_normal(size=(n_sims, len(need)))
    shocks = Z @ L.T + mu

    # elasticities/prior
    b = {
        "dlog_wfe_usd": pick_coef(EL, "dlog_wfe_usd", default=0.8),
        "dlog_dram_idx": pick_coef(EL, "dlog_dram_idx", default=0.2),
        "dlog_ai_capex_usd": pick_coef(EL, "dlog_ai_capex_usd", default=0.15),
        "r_fx_usdjpy": pick_coef(EL, "r_fx_usdjpy", default=0.20)
    }
    idx = {nm:i for i,nm in enumerate(need)}
    dlog_sales = np.zeros(n_sims)
    for k, bk in b.items():
        if k in idx: dlog_sales += bk * shocks[:, idx[k]]
    base = float(P["sales_jpy"].dropna().iloc[-1])
    sales = base * np.exp(dlog_sales)
    s_sorted = np.sort(sales)
    var5 = float(np.percentile(sales, 5))
    es5 = float(s_sorted[:max(1,int(0.05*len(s_sorted)))].mean())
    return pd.DataFrame([{
        "VaR_5pct_sales_jpy": var5,
        "ES_5pct_sales_jpy": es5,
        "mean_sales_jpy": float(sales.mean()),
        "sd_sales_jpy": float(sales.std(ddof=0)),
        "n_sims": int(n_sims)
    }])

# ----------------------------- mixes/snapshots -----------------------------

def latest_mix(S: pd.DataFrame) -> pd.DataFrame:
    if S.empty: return pd.DataFrame()
    last_date = S["date"].max()
    G = S[S["date"]==last_date].copy()
    out = []
    tot = float(G.get("sales_jpy", pd.Series(dtype=float)).sum())
    if "vendor" in G.columns:
        gv = (G.groupby("vendor", as_index=False)["sales_jpy"].sum() if "sales_jpy" in G.columns else pd.DataFrame())
        if not gv.empty:
            gv["share_%"] = 100.0 * gv["sales_jpy"] / (tot + 1e-12)
            gv["type"] = "vendor"
            out.append(gv.rename(columns={"vendor":"name"}))
    if "segment" in G.columns:
        gs = (G.groupby("segment", as_index=False)["sales_jpy"].sum() if "sales_jpy" in G.columns else pd.DataFrame())
        if not gs.empty:
            gs["share_%"] = 100.0 * gs["sales_jpy"] / (tot + 1e-12)
            gs["type"] = "segment"
            out.append(gs.rename(columns={"segment":"name"}))
    if out:
        M = pd.concat(out, ignore_index=True)
        M["date"] = last_date
        return M[["date","type","name","sales_jpy","share_%"]].sort_values(["type","share_%"], ascending=[True, False])
    return pd.DataFrame()


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    sales: str
    wfe: Optional[str]
    memory: Optional[str]
    ai: Optional[str]
    devices: Optional[str]
    btb: Optional[str]
    exports: Optional[str]
    fx: Optional[str]
    events: Optional[str]
    lags: int
    min_obs: int
    horizon: int
    wfe_pct: float
    memory_pct: float
    ai_capex_pct: float
    fx_pct: float
    china_share_pp: float
    outdir: str
    by_vendor: bool
    by_segment: bool
    start: Optional[str]
    end: Optional[str]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Japan semi-cap demand: elasticities, events, scenarios & stress")
    ap.add_argument("--sales", required=True)
    ap.add_argument("--wfe", default="")
    ap.add_argument("--memory", default="")
    ap.add_argument("--ai", default="")
    ap.add_argument("--devices", default="")
    ap.add_argument("--btb", default="")
    ap.add_argument("--exports", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--min_obs", type=int, default=36)
    ap.add_argument("--horizon", type=int, default=12)
    # Scenario shocks
    ap.add_argument("--wfe_pct", type=float, default=0.0)
    ap.add_argument("--memory_pct", type=float, default=0.0)
    ap.add_argument("--ai_capex_pct", type=float, default=0.0)
    ap.add_argument("--fx_pct", type=float, default=0.0)
    ap.add_argument("--china_share_pp", type=float, default=0.0)
    ap.add_argument("--outdir", default="out_semicap")
    ap.add_argument("--by_vendor", action="store_true")
    ap.add_argument("--by_segment", action="store_true")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    S  = load_sales(args.sales, by_vendor=args.by_vendor, by_segment=args.by_segment)
    W  = load_wfe(args.wfe) if args.wfe else pd.DataFrame()
    M  = load_memory(args.memory) if args.memory else pd.DataFrame()
    A  = load_ai(args.ai) if args.ai else pd.DataFrame()
    Dv = load_devices(args.devices) if args.devices else pd.DataFrame()
    B  = load_btb(args.btb) if args.btb else pd.DataFrame()
    X  = load_exports(args.exports) if args.exports else pd.DataFrame()
    FX = load_fx(args.fx) if args.fx else pd.DataFrame()
    EV = load_events(args.events) if args.events else pd.DataFrame()

    # Time filters
    if args.start:
        s = eom(pd.Series([args.start])).iloc[0]
        for df in [S,W,M,A,Dv,B,X,FX,EV]:
            if not df.empty: df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = eom(pd.Series([args.end])).iloc[0]
        for df in [S,W,M,A,Dv,B,X,FX,EV]:
            if not df.empty: df.drop(df[df["date"] > e].index, inplace=True)

    # Panel
    P = build_panel(S, W, M, A, Dv, B, X, FX, EV)
    if P.empty: raise ValueError("Panel is empty after merges/filters. Check inputs.")
    P.to_csv(outdir / "panel.csv", index=False)

    # Elasticities for sales and (if available) bookings
    EL1 = regress_elasticities(P, target="dlog_sales", L=int(args.lags), min_obs=int(args.min_obs))
    EL2 = regress_elasticities(P, target="dlog_bookings", L=int(args.lags), min_obs=int(args.min_obs)) if "dlog_bookings" in P.columns else pd.DataFrame()
    EL = pd.concat([EL1, EL2], ignore_index=True) if not EL1.empty or not EL2.empty else pd.DataFrame()
    if not EL.empty: EL.to_csv(outdir / "elasticity_sales.csv", index=False)

    # Event study
    ES = event_study(P, EV, window=2) if not EV.empty else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Scenarios (use sales elasticities)
    EL_use = EL[EL["target"]=="dlog_sales"] if not EL.empty else pd.DataFrame()
    SC = run_scenarios(P, EL_use,
                       wfe_pct=float(args.wfe_pct),
                       memory_pct=float(args.memory_pct),
                       ai_capex_pct=float(args.ai_capex_pct),
                       fx_pct=float(args.fx_pct),
                       china_share_pp=float(args.china_share_pp),
                       horizon=int(args.horizon))
    if not SC.empty: SC.to_csv(outdir / "scenarios.csv", index=False)

    # Stress VaR/ES
    ST = stress_var_es(P, EL_use, n_sims=20000)
    if not ST.empty: ST.to_csv(outdir / "stress_vares.csv", index=False)

    # Mix snapshot
    MIX = latest_mix(S)
    if not MIX.empty: MIX.to_csv(outdir / "mix_latest.csv", index=False)

    # Summary
    summary = {
        "sample": {
            "start": str(P["date"].min().date()),
            "end": str(P["date"].max().date()),
            "months": int(P["date"].nunique())
        },
        "has": {
            "wfe": bool(not W.empty), "memory": bool(not M.empty), "ai_capex": bool(not A.empty),
            "devices": bool(not Dv.empty), "btb": bool(not B.empty), "exports": bool(not X.empty),
            "fx": bool(not FX.empty), "events": bool(not EV.empty)
        },
        "latest": {
            "sales_jpy": float(P["sales_jpy"].dropna().iloc[-1]) if "sales_jpy" in P.columns and P["sales_jpy"].notna().any() else None,
            "bookings_jpy": float(P["bookings_jpy"].dropna().iloc[-1]) if "bookings_jpy" in P.columns and P["bookings_jpy"].notna().any() else None,
            "backlog_cover_m": float((P["backlog_jpy"]/P["sales_jpy"].rolling(3).mean()).iloc[-1]) if {"backlog_jpy","sales_jpy"}.issubset(P.columns) else None
        },
        "outputs": {
            "panel": "panel.csv",
            "elasticity_sales": "elasticity_sales.csv" if not EL.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "scenarios": "scenarios.csv" if not SC.empty else None,
            "stress_vares": "stress_vares.csv" if not ST.empty else None,
            "mix_latest": "mix_latest.csv" if not MIX.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        sales=args.sales, wfe=(args.wfe or None), memory=(args.memory or None), ai=(args.ai or None),
        devices=(args.devices or None), btb=(args.btb or None), exports=(args.exports or None),
        fx=(args.fx or None), events=(args.events or None),
        lags=int(args.lags), min_obs=int(args.min_obs), horizon=int(args.horizon),
        wfe_pct=float(args.wfe_pct), memory_pct=float(args.memory_pct), ai_capex_pct=float(args.ai_capex_pct),
        fx_pct=float(args.fx_pct), china_share_pp=float(args.china_share_pp),
        outdir=args.outdir, by_vendor=bool(args.by_vendor), by_segment=bool(args.by_segment),
        start=(args.start or None), end=(args.end or None)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Japan Semi-cap Demand Toolkit ==")
    print(f"Sample: {summary['sample']['start']} → {summary['sample']['end']}  ({summary['sample']['months']} months)")
    if summary["outputs"]["elasticity_sales"]: print("Elasticities estimated (HAC/NW).")
    if summary["outputs"]["event_study"]: print("Event study completed (±2 months).")
    if summary["outputs"]["scenarios"]: print(f"Scenario run: WFE {args.wfe_pct}%, Memory {args.memory_pct}%, AI capex {args.ai_capex_pct}%, USDJPY {args.fx_pct}%, China share {args.china_share_pp}pp.")
    if summary["outputs"]["stress_vares"]: print("Stress VaR/ES computed.")
    if summary["outputs"]["mix_latest"]: print("Latest vendor/segment mix snapshot written.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
