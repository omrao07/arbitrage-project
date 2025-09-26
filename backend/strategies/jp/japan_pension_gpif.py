#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
japan_pension_gpif.py — GPIF-style portfolio analytics: attribution, flows, rebalancing, FX & risk
--------------------------------------------------------------------------------------------------

What this does
==============
Loads GPIF-like portfolio data (by asset class) and produces:
1) Clean monthly panel (weights, returns, FX, hedge ratios)
2) **Brinson–Fachler attribution** vs policy portfolio (allocation, selection, interaction)
3) **Flow estimates** from market value changes (contribution/redemption proxy)
4) **Band rebalancing** signals & estimated trades/turnover
5) **FX attribution** (unhedged vs hedged legs) and hedge-ratio sensitivity
6) **Risk metrics**: rolling vol, tracking error, information ratio, var/es (mc/historical)
7) **Scenario P&L**: equity shocks, rate shocks via duration, USDJPY shock through unhedged legs
8) **Event study** around BoJ/policy dates if daily series provided

Inputs (CSV; headers are flexible/case-insensitive; dates monthly unless noted)
------------------------------------------------------------------------------
--positions positions.csv   REQUIRED
  Columns (any subset):
    date, asset[, class, bucket], market_value_jpy[, mv_jpy], weight[, w],
    hedge_ratio[, hedged], duration[, d], region[, dom/for], category
  Notes:
    • If weight missing, it is inferred from market value per date.
    • duration (modified) used for rate-shock P&L on bond buckets.
    • hedge_ratio in [0..1]; if missing for domestic assets, assumed 1.0; for foreign, 0.0.

--returns returns.csv       OPTIONAL (monthly total returns)
  Columns (any subset):
    date, asset, ret_jpy[, total_return_jpy], ret_local[, r_local], fx_return[, r_fx]
  Notes:
    • If only local & fx returns exist, total_jpy = (1+r_local)*(1+r_fx)-1
    • If only ret_jpy exists, that is used as asset return in JPY.

--bench bench.csv           OPTIONAL (policy/benchmark)
  Columns (any subset):
    date, asset, policy_weight[, bench_weight, w_policy, w_bench]
    bench_ret[, r_bench, ret_bench_jpy]
  Notes:
    • If bench weights omitted, policy_weight is used as benchmark weight.
    • If bench_ret omitted, will use asset ret_jpy as proxy (no selection effect).

--fx fx.csv                 OPTIONAL (monthly)
  Columns:
    date, USDJPY[, JPYUSD], JPY_REER (optional)
  Notes:
    • Used for FX attribution and scenarios (USDJPY shock).

--daily daily.csv           OPTIONAL (daily)
  Columns:
    date, port_ret_excess[, port_excess], bank_index, jgb_10y, usdjpy_r
  Notes:
    • Used for event study if --events is given.

--events events.csv         OPTIONAL (event list)
  Columns:
    date, type[, label]
  Notes:
    • BoJ meetings, YCC regime changes, reserve policy tweaks, etc.

--map map.csv               OPTIONAL (asset mapping)
  Columns:
    asset, group
  Notes:
    • Map raw asset names into analysis buckets, e.g.:
      JP_EQUITY -> "Domestic Equity"; FX_EQUITY -> "Foreign Equity";
      JP_BOND   -> "Domestic Bond";  FX_BOND   -> "Foreign Bond"; ALTS -> "Alternatives"

Key outputs
-----------
out/
  panel_asset.csv               # tidy panel (date, asset_group, mv, w, ret_jpy, r_local, r_fx, hedge_ratio, duration)
  attribution_brinson.csv       # allocation/selection/interaction, by period & cumulative
  flow_estimates.csv            # estimated contributions/redemptions by group & total
  rebalancing.csv               # band breaches, trades to mid/band, turnover
  currency_attribution.csv      # contribution of FX & hedge overlays by group
  risk_metrics.csv              # rolling vol/TE/IR + var/es
  scenarios.csv                 # one-step P&L under shocks
  event_study.csv              # CARs around events (daily)
  summary.json, config.json

DISCLAIMER: Research toolkit. Validate with your internal data/definitions before decisions.
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
    # fuzzy contains
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

def pct(x) -> float:
    return float(x) if x is not None else 0.0


# ----------------------------- loaders -----------------------------

def load_map(path: Optional[str]) -> Dict[str, str]:
    if not path: return {}
    df = pd.read_csv(path)
    a = ncol(df,"asset"); g = ncol(df,"group")
    if not (a and g): return {}
    df = df.rename(columns={a:"asset", g:"group"})
    return {str(r["asset"]): str(r["group"]) for _, r in df.iterrows()}

def map_group(s: pd.Series, mapping: Dict[str,str]) -> pd.Series:
    if not mapping: return s
    return s.apply(lambda x: mapping.get(str(x), str(x)))

def load_positions(path: str, mapping: Dict[str,str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df,"date"); a = ncol(df,"asset","class","bucket"); mv = ncol(df,"market_value_jpy","mv_jpy","market_value")
    w = ncol(df,"weight","w"); hr = ncol(df,"hedge_ratio","hedged"); dur = ncol(df,"duration","d")
    reg = ncol(df,"region","dom_for","domestic_foreign","domfor"); cat = ncol(df,"category")
    if not (d and a): raise ValueError("positions.csv needs date and asset columns.")
    df = df.rename(columns={d:"date", a:"asset"})
    if mv: df = df.rename(columns={mv:"mv_jpy"})
    if w:  df = df.rename(columns={w:"weight"})
    if hr: df = df.rename(columns={hr:"hedge_ratio"})
    if dur: df = df.rename(columns={dur:"duration"})
    if reg: df = df.rename(columns={reg:"region"})
    if cat: df = df.rename(columns={cat:"category"})
    df["date"] = eom(df["date"])
    for k in ["mv_jpy","weight","hedge_ratio","duration"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # infer weights if missing
    if "weight" not in df.columns and "mv_jpy" in df.columns:
        tmp = df.groupby("date")["mv_jpy"].transform("sum")
        df["weight"] = df["mv_jpy"] / tmp.replace(0,np.nan)
    # group mapping
    df["asset_group"] = map_group(df["asset"].astype(str), mapping)
    # defaults for hedge ratio
    if "hedge_ratio" not in df.columns:
        # assume domestic assets fully JPY (hedge=1), foreign unhedged (0) unless region helps
        df["hedge_ratio"] = np.where(df.get("region","").astype(str).str.lower().str.contains("dom"), 1.0,
                              np.where(df.get("region","").astype(str).str.lower().str.contains("for"), 0.0, np.nan))
    # fill hedge defaults by group heuristics
    df["asset_group_low"] = df["asset_group"].str.lower()
    df["hedge_ratio"] = df["hedge_ratio"].fillna(
        np.where(df["asset_group_low"].str.contains("domestic|japan"), 1.0,
        np.where(df["asset_group_low"].str.contains("foreign|global|overseas"), 0.0, 1.0))
    )
    df.drop(columns=["asset_group_low"], inplace=True)
    return df

def load_returns(path: Optional[str], mapping: Dict[str,str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); a = ncol(df,"asset","class"); rj = ncol(df,"ret_jpy","total_return_jpy","r_jpy")
    rl = ncol(df,"ret_local","r_local"); rf = ncol(df,"fx_return","r_fx")
    if not (d and a): raise ValueError("returns.csv needs date and asset.")
    df = df.rename(columns={d:"date", a:"asset"})
    if rj: df = df.rename(columns={rj:"ret_jpy"})
    if rl: df = df.rename(columns={rl:"ret_local"})
    if rf: df = df.rename(columns={rf:"r_fx"})
    df["date"] = eom(df["date"])
    for k in ["ret_jpy","ret_local","r_fx"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # derive JPY total if needed
    if "ret_jpy" not in df.columns and {"ret_local","r_fx"}.issubset(df.columns):
        df["ret_jpy"] = (1.0 + df["ret_local"]) * (1.0 + df["r_fx"]) - 1.0
    df["asset_group"] = map_group(df["asset"].astype(str), mapping)
    return df

def load_bench(path: Optional[str], mapping: Dict[str,str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); a = ncol(df,"asset","class"); wp = ncol(df,"policy_weight","w_policy"); wb = ncol(df,"bench_weight","w_bench")
    rb = ncol(df,"bench_ret","r_bench","ret_bench_jpy")
    if not (d and a): raise ValueError("bench.csv needs date and asset.")
    df = df.rename(columns={d:"date", a:"asset"})
    if wp: df = df.rename(columns={wp:"policy_weight"})
    if wb: df = df.rename(columns={wb:"bench_weight"})
    if rb: df = df.rename(columns={rb:"bench_ret"})
    df["date"] = eom(df["date"])
    for k in ["policy_weight","bench_weight","bench_ret"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # fallback: if bench_weight missing, use policy_weight
    if "bench_weight" not in df.columns and "policy_weight" in df.columns:
        df["bench_weight"] = df["policy_weight"]
    df["asset_group"] = map_group(df["asset"].astype(str), mapping)
    return df

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); u = ncol(df,"USDJPY","usdjpy"); j = ncol(df,"JPYUSD","jpyusd"); reer = ncol(df,"JPY_REER","reer")
    if not d: raise ValueError("fx.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    if u:
        df["USDJPY"] = safe_num(df[u])
    elif j:
        df["USDJPY"] = 1.0 / safe_num(df[j])
    else:
        num = [c for c in df.columns if c != "date"]
        if not num: raise ValueError("Provide USDJPY or JPYUSD in fx.csv.")
        df["USDJPY"] = safe_num(df[num[0]])
    if reer: df = df.rename(columns={reer:"JPY_REER"})
    df["r_usdjpy"] = dlog(df["USDJPY"])
    return df

def load_daily(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); pe = ncol(df,"port_ret_excess","port_excess"); bi = ncol(df,"bank_index"); jy = ncol(df,"usdjpy_r","r_usdjpy")
    y10 = ncol(df,"jgb_10y","jgb10y")
    if not d: raise ValueError("daily.csv needs date.")
    df = df.rename(columns={d:"date"})
    if pe: df = df.rename(columns={pe:"port_excess"})
    if bi: df = df.rename(columns={bi:"bank_index"})
    if jy: df = df.rename(columns={jy:"r_usdjpy"})
    if y10: df = df.rename(columns={y10:"jgb_10y"})
    df["date"] = to_dt(df["date"])
    for k in ["port_excess","bank_index","r_usdjpy","jgb_10y"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df.sort_values("date")

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"type"); l = ncol(df,"label","event")
    if not d: raise ValueError("events.csv needs date.")
    df = df.rename(columns={d:"date"})
    if t: df = df.rename(columns={t:"type"})
    if l: df = df.rename(columns={l:"label"})
    df["date"] = to_dt(df["date"])
    if "type" not in df.columns: df["type"] = "EVENT"
    if "label" not in df.columns: df["label"] = df["type"]
    return df.sort_values("date")


# ----------------------------- panel construction -----------------------------

def build_panel(POS: pd.DataFrame, RET: pd.DataFrame, BEN: pd.DataFrame, FX: pd.DataFrame) -> pd.DataFrame:
    # aggregate to asset_group
    p = POS.copy()
    # prefer explicit group columns
    p["asset_group"] = p["asset_group"].astype(str)
    # aggregate mv/weight/hedge/duration
    agg = (p.groupby(["date","asset_group"], as_index=False)
             .agg(mv_jpy=("mv_jpy","sum"),
                  weight=("weight","sum"),
                  hedge_ratio=("hedge_ratio","mean"),
                  duration=("duration","mean")))
    # returns
    if not RET.empty:
        r = RET.groupby(["date","asset_group"], as_index=False).agg(
            ret_jpy=("ret_jpy","mean"),
            ret_local=("ret_local","mean"),
            r_fx=("r_fx","mean")
        )
        agg = agg.merge(r, on=["date","asset_group"], how="left")
    # bench/policy
    if not BEN.empty:
        b = (BEN.groupby(["date","asset_group"], as_index=False)
               .agg(policy_weight=("policy_weight","mean"),
                    bench_weight=("bench_weight","mean"),
                    bench_ret=("bench_ret","mean")))
        agg = agg.merge(b, on=["date","asset_group"], how="left")
    # FX join (for r_usdjpy only)
    if not FX.empty:
        agg = agg.merge(FX[["date","USDJPY","r_usdjpy"]], on="date", how="left")

    # normalize weights if necessary
    Wsum = agg.groupby("date")["weight"].transform(lambda s: s.fillna(0).sum())
    agg["weight"] = np.where(Wsum>0, agg["weight"]/Wsum, agg["weight"])
    # derive weight from mv if still missing
    if "mv_jpy" in agg.columns:
        tmp = agg.groupby("date")["mv_jpy"].transform("sum")
        agg["weight"] = agg["weight"].fillna(agg["mv_jpy"]/tmp.replace(0,np.nan))
    # fallback bench_ret
    if "bench_ret" not in agg.columns:
        agg["bench_ret"] = np.nan
    # sanity
    return agg.sort_values(["date","asset_group"]).reset_index(drop=True)


# ----------------------------- Brinson–Fachler attribution -----------------------------

def brinson_fachler(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Arithmetic Brinson–Fachler per period:
      Allocation_i = (wP_i - wB_i) * (rB_i - rB)
      Selection_i  = wB_i * (rP_i - rB_i)
      Interaction_i= (wP_i - wB_i) * (rP_i - rB_i)
    where rB = Σ wB_i rB_i ; rP = Σ wP_i rP_i
    """
    df = panel.copy()
    # weights
    df["wP"] = df["weight"].astype(float)
    df["wB"] = df.get("bench_weight", df.get("policy_weight", np.nan)).astype(float)
    # returns
    df["rP"] = df.get("ret_jpy", np.nan).astype(float)
    df["rB_i"] = df.get("bench_ret", np.nan).astype(float)
    # if rB_i missing, proxy with rP (no selection)
    df["rB_i"] = df["rB_i"].fillna(df["rP"])
    out_rows = []
    for dt, g in df.groupby("date"):
        if g["wB"].isna().all():
            # if no bench weights, use wP as wB (zero allocation)
            g["wB"] = g["wP"]
        rB = float(np.nansum(g["wB"] * g["rB_i"]))
        rP = float(np.nansum(g["wP"] * g["rP"]))
        for _, r in g.iterrows():
            alloc = (pct(r["wP"]) - pct(r["wB"])) * (pct(r["rB_i"]) - rB)
            selec = pct(r["wB"]) * (pct(r["rP"]) - pct(r["rB_i"]))
            inter = (pct(r["wP"]) - pct(r["wB"])) * (pct(r["rP"]) - pct(r["rB_i"]))
            out_rows.append({
                "date": dt, "asset_group": r["asset_group"],
                "r_port": rP, "r_bench": rB,
                "allocation": alloc, "selection": selec, "interaction": inter,
                "excess": rP - rB
            })
    A = pd.DataFrame(out_rows).sort_values(["date","asset_group"])
    # cumulative sums
    Cum = (A.groupby("asset_group")[["allocation","selection","interaction","excess"]]
             .cumsum().rename(columns=lambda c: "cum_"+c))
    return pd.concat([A, Cum], axis=1)


# ----------------------------- Flow estimation -----------------------------

def estimate_flows(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate net contributions/redemptions by asset group:
      Flow_t ≈ MV_t - MV_{t-1} * (1 + r_t)
    where r_t is the asset group's JPY total return.
    """
    if "mv_jpy" not in panel.columns:
        return pd.DataFrame()
    df = panel.sort_values(["asset_group","date"]).copy()
    df["ret_jpy"] = df.get("ret_jpy", 0.0)
    df["mv_prev"] = df.groupby("asset_group")["mv_jpy"].shift(1)
    df["flow"] = df["mv_jpy"] - df["mv_prev"] * (1.0 + df["ret_jpy"])
    tot = (df.groupby("date", as_index=False)
             .agg(total_flow=("flow","sum"),
                  total_mv=("mv_jpy","sum")))
    out = df[["date","asset_group","mv_jpy","ret_jpy","flow"]].merge(tot, on="date", how="left")
    return out.sort_values(["date","asset_group"])


# ----------------------------- Band rebalancing -----------------------------

def rebalancing_signals(panel: pd.DataFrame, bench: pd.DataFrame,
                        band_low: float=0.03, band_high: float=0.03,
                        to_mid: bool=True) -> pd.DataFrame:
    """
    If weights deviate from policy by more than ±band, compute trade to bring to mid or band edge.
    """
    if bench.empty and "policy_weight" not in panel.columns:
        return pd.DataFrame()
    df = panel.copy()
    if "policy_weight" not in df.columns and not bench.empty:
        B = bench[["date","asset_group","policy_weight"]]
        df = df.merge(B, on=["date","asset_group"], how="left")
    # normalize policy per date
    Wsum = df.groupby("date")["policy_weight"].transform("sum")
    df["policy_weight"] = df["policy_weight"] / Wsum.replace(0,np.nan)
    df["dev"] = df["weight"] - df["policy_weight"]
    # thresholds
    hi = df["policy_weight"] + band_high
    lo = df["policy_weight"] - band_low
    breach_hi = df["weight"] > hi
    breach_lo = df["weight"] < lo
    target = np.where(to_mid, df["policy_weight"], np.where(breach_hi, hi, lo))
    trade_w = np.where(breach_hi | breach_lo, target - df["weight"], 0.0)
    # turnover in JPY if mv provided
    if "mv_jpy" in df.columns:
        tot_mv = df.groupby("date")["mv_jpy"].transform("sum")
        trade_jpy = trade_w * tot_mv
    else:
        trade_jpy = np.nan
    out = df[["date","asset_group","weight","policy_weight"]].copy()
    out["dev"] = df["dev"]
    out["breach"] = (breach_hi | breach_lo).astype(int)
    out["trade_to_target_w"] = trade_w
    out["trade_to_target_jpy"] = trade_jpy
    # portfolio turnover (%)
    tun = (out.groupby("date")["trade_to_target_w"].apply(lambda s: np.nansum(np.abs(s)))).reset_index(name="turnover_pct")
    out = out.merge(tun, on="date", how="left")
    return out.sort_values(["date","asset_group"])


# ----------------------------- Currency attribution -----------------------------

def currency_attrib(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For foreign groups, decompose ret_jpy ≈ ret_local + (1-hedge_ratio)*r_fx + cross term.
    """
    df = panel.copy()
    if not {"ret_jpy","ret_local","r_fx","hedge_ratio"}.issubset(df.columns):
        return pd.DataFrame()
    df["unhedged"] = 1.0 - df["hedge_ratio"].clip(0,1)
    df["fx_contrib"] = df["unhedged"] * df["r_fx"]
    df["local_contrib"] = df["ret_local"]
    df["cross_term"] = (1.0 + df["ret_local"]) * (1.0 + df["fx_contrib"]) - 1.0 - df["ret_local"] - df["fx_contrib"]
    df["residual"] = df["ret_jpy"] - (df["local_contrib"] + df["fx_contrib"] + df["cross_term"])
    return df[["date","asset_group","hedge_ratio","ret_jpy","ret_local","r_fx","fx_contrib","local_contrib","cross_term","residual"]]


# ----------------------------- Risk metrics -----------------------------

def rolling_metrics(panel: pd.DataFrame, window: int=36) -> pd.DataFrame:
    """
    Compute rolling vol of portfolio & benchmark, tracking error and IR (excess / TE).
    """
    df = panel.copy()
    # build portfolio and benchmark series
    gP = (df.groupby(["date"]).apply(lambda g: float(np.nansum(g["weight"] * g.get("ret_jpy",0.0)))))
    gB = (df.groupby(["date"]).apply(lambda g: float(np.nansum(g.get("bench_weight", g.get("policy_weight", g["weight"])) * g.get("bench_ret", g.get("ret_jpy",0.0))))))
    ret = pd.DataFrame({"date": gP.index, "rP": gP.values, "rB": gB.values}).sort_values("date")
    ret["excess"] = ret["rP"] - ret["rB"]
    ret["volP"] = ret["rP"].rolling(window, min_periods=max(6, window//3)).std(ddof=0) * np.sqrt(12)
    ret["volB"] = ret["rB"].rolling(window, min_periods=max(6, window//3)).std(ddof=0) * np.sqrt(12)
    ret["TE"] = ret["excess"].rolling(window, min_periods=max(6, window//3)).std(ddof=0) * np.sqrt(12)
    ann_excess = ret["excess"].rolling(window, min_periods=max(6, window//3)).mean() * 12
    ret["IR"] = ann_excess / (ret["TE"] + 1e-12)
    return ret


# ----------------------------- VaR/ES (historical) -----------------------------

def var_es_hist(excess: pd.Series, alpha: float=0.05) -> Tuple[float,float]:
    x = excess.dropna().values
    if len(x) < 24: return (np.nan, np.nan)
    q = np.quantile(x, alpha)
    es = x[x<=q].mean() if np.any(x<=q) else np.nan
    return float(q), float(es)


# ----------------------------- Scenarios -----------------------------

def scenarios(panel: pd.DataFrame,
              eq_shock_jp: float=-0.05, eq_shock_fx: float=-0.07,
              jgb_bp: float=+25.0, global_bp: float=+25.0,
              usdjpy_pct: float=+5.0, hedge_shift: float=0.0) -> pd.DataFrame:
    """
    One-step P&L approximation using last month exposures:
      • Equities: ΔP ≈ shock * MV
      • Bonds: ΔP ≈ -Duration * Δy * MV  (Δy in decimal)
      • FX: unhedged fraction * ΔFX * MV (applied to foreign assets)
    hedge_shift: Δ in hedge ratio (e.g., +0.1 = increase hedge by 10pp across foreign groups)
    """
    if panel.empty: return pd.DataFrame()
    last = (panel.sort_values("date").groupby("asset_group").tail(1))
    mv_tot = float(last["mv_jpy"].sum()) if "mv_jpy" in last.columns else np.nan
    rows = []
    for _, r in last.iterrows():
        g = str(r["asset_group"]).lower()
        mv = float(r.get("mv_jpy", np.nan))
        dur = float(r.get("duration", 0.0) or 0.0)
        hr = float(r.get("hedge_ratio", 1.0) or 1.0)
        # equity shock map
        if "equity" in g:
            shock = eq_shock_jp if ("dom" in g or "japan" in g) else eq_shock_fx
            pnl_eq = mv * shock
        else:
            pnl_eq = 0.0
        # rates shock map
        is_bond = ("bond" in g or "fixed" in g)
        d_bp = jgb_bp if ("dom" in g or "japan" in g) else global_bp
        pnl_rate = - dur * (d_bp/10000.0) * mv if is_bond else 0.0
        # FX effect on foreign (non-domestic) buckets: apply to unhedged share
        unhedged = max(0.0, min(1.0, 1.0 - (hr + hedge_shift)))
        is_foreign = ("foreign" in g or "global" in g or "overseas" in g or ("equity" in g and "dom" not in g) or ("bond" in g and "dom" not in g))
        pnl_fx = mv * unhedged * (usdjpy_pct/100.0) if is_foreign else 0.0
        rows.append({"asset_group": r["asset_group"], "mv_jpy": mv,
                     "pnl_equity": pnl_eq, "pnl_rate": pnl_rate, "pnl_fx": pnl_fx})
    S = pd.DataFrame(rows)
    if S.empty: return S
    S["pnl_total"] = S[["pnl_equity","pnl_rate","pnl_fx"]].sum(axis=1)
    S.loc["TOTAL"] = {"asset_group":"TOTAL","mv_jpy": mv_tot,
                      "pnl_equity": S["pnl_equity"].sum(),
                      "pnl_rate": S["pnl_rate"].sum(),
                      "pnl_fx": S["pnl_fx"].sum(),
                      "pnl_total": S["pnl_total"].sum()}
    return S.reset_index(drop=True)


# ----------------------------- Event study (daily) -----------------------------

def event_study_daily(DAY: pd.DataFrame, EVT: pd.DataFrame, window: int=5) -> pd.DataFrame:
    if DAY.empty or EVT.empty or "port_excess" not in DAY.columns: return pd.DataFrame()
    s = DAY.set_index("date")["port_excess"].dropna()
    rows = []
    for _, e in EVT.iterrows():
        dt = pd.Timestamp(e["date"])
        if dt not in s.index:
            idx = s.index
            pos = idx.searchsorted(dt)
            if pos >= len(idx): continue
            dt = idx[pos]
        i = s.index.get_loc(dt)
        L = max(0, i - window)
        R = min(len(s)-1, i + window)
        car = float(s.iloc[L:R+1].sum())
        rows.append({"event_date": str(DAY["date"].iloc[i].date()),
                     "type": e.get("type","EVENT"), "label": e.get("label", e.get("type","EVENT")),
                     "CAR_port_excess": car})
    return pd.DataFrame(rows).sort_values(["event_date","type"])


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    positions: str
    returns: Optional[str]
    bench: Optional[str]
    fx: Optional[str]
    daily: Optional[str]
    events: Optional[str]
    amap: Optional[str]
    outdir: str
    band_low: float
    band_high: float
    window: int
    eq_shock_jp: float
    eq_shock_fx: float
    jgb_bp: float
    global_bp: float
    usdjpy_pct: float
    hedge_shift: float

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GPIF-style portfolio analytics")
    ap.add_argument("--positions", required=True)
    ap.add_argument("--returns", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--daily", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--amap", default="", help="asset mapping CSV (asset→group)")
    ap.add_argument("--outdir", default="out_gpif")
    # rebal bands
    ap.add_argument("--band_low", type=float, default=0.03)
    ap.add_argument("--band_high", type=float, default=0.03)
    ap.add_argument("--window", type=int, default=36, help="rolling window (months)")
    # scenarios
    ap.add_argument("--eq_shock_jp", type=float, default=-5.0, help="% shock domestic equity")
    ap.add_argument("--eq_shock_fx", type=float, default=-7.0, help="% shock foreign equity")
    ap.add_argument("--jgb_bp", type=float, default=+25.0, help="JGB parallel shift (bp)")
    ap.add_argument("--global_bp", type=float, default=+25.0, help="Global rates shift (bp)")
    ap.add_argument("--usdjpy_pct", type=float, default=+5.0, help="% change in USDJPY (+ = USD up/JPY down)")
    ap.add_argument("--hedge_shift", type=float, default=0.0, help="Δ hedge ratio on foreign groups (e.g., 0.1)")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    amap = load_map(args.amap)
    POS = load_positions(args.positions, amap)
    RET = load_returns(args.returns, amap) if args.returns else pd.DataFrame()
    BEN = load_bench(args.bench, amap) if args.bench else pd.DataFrame()
    FX  = load_fx(args.fx) if args.fx else pd.DataFrame()
    DAY = load_daily(args.daily) if args.daily else pd.DataFrame()
    EVT = load_events(args.events) if args.events else pd.DataFrame()

    # Panel
    P = build_panel(POS, RET, BEN, FX)
    if P.empty:
        raise ValueError("Panel is empty — check inputs/mapping.")
    P.to_csv(outdir / "panel_asset.csv", index=False)

    # Attribution
    ATTR = brinson_fachler(P)
    if not ATTR.empty: ATTR.to_csv(outdir / "attribution_brinson.csv", index=False)

    # Flows
    FLO = estimate_flows(P)
    if not FLO.empty: FLO.to_csv(outdir / "flow_estimates.csv", index=False)

    # Rebalancing
    REB = rebalancing_signals(P, BEN, band_low=float(args.band_low), band_high=float(args.band_high), to_mid=True)
    if not REB.empty: REB.to_csv(outdir / "rebalancing.csv", index=False)

    # Currency attribution
    CUR = currency_attrib(P)
    if not CUR.empty: CUR.to_csv(outdir / "currency_attribution.csv", index=False)

    # Risk metrics
    RM = rolling_metrics(P, window=int(args.window))
    if not RM.empty:
        # add hist VaR/ES (excess)
        q5, es5 = var_es_hist(RM["excess"], alpha=0.05)
        RM.to_csv(outdir / "risk_metrics.csv", index=False)
    else:
        q5, es5 = (np.nan, np.nan)

    # Scenarios
    SCN = scenarios(P,
                    eq_shock_jp=float(args.eq_shock_jp)/100.0,
                    eq_shock_fx=float(args.eq_shock_fx)/100.0,
                    jgb_bp=float(args.jgb_bp),
                    global_bp=float(args.global_bp),
                    usdjpy_pct=float(args.usdjpy_pct),
                    hedge_shift=float(args.hedge_shift))
    if not SCN.empty: SCN.to_csv(outdir / "scenarios.csv", index=False)

    # Event study (daily)
    ES = event_study_daily(DAY, EVT, window=5) if (not DAY.empty and not EVT.empty) else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Summary
    # Portfolio/benchmark aggregates (last period)
    last = P.sort_values("date").groupby("asset_group").tail(1)
    tot_mv = float(last.get("mv_jpy", pd.Series(dtype=float)).sum()) if "mv_jpy" in last.columns else None
    port_w = last[["asset_group","weight"]].set_index("asset_group")["weight"].to_dict()
    pol_w = {}
    if "policy_weight" in last.columns:
        pol_w = last[["asset_group","policy_weight"]].set_index("asset_group")["policy_weight"].to_dict()

    summary = {
        "sample": {
            "start": str(P["date"].min().date()),
            "end": str(P["date"].max().date()),
            "months": int(P["date"].nunique())
        },
        "tot_mv_jpy_latest": tot_mv,
        "latest_weights_port": port_w,
        "latest_weights_policy": pol_w,
        "risk": {
            "rolling_window_m": int(args.window),
            "hist_VaR5_excess": q5,
            "hist_ES5_excess": es5
        },
        "scenarios_inputs": {
            "eq_shock_jp_%": float(args.eq_shock_jp),
            "eq_shock_fx_%": float(args.eq_shock_fx),
            "jgb_bp": float(args.jgb_bp),
            "global_bp": float(args.global_bp),
            "usdjpy_%": float(args.usdjpy_pct),
            "hedge_shift": float(args.hedge_shift)
        },
        "outputs": {
            "panel_asset": "panel_asset.csv",
            "attribution_brinson": "attribution_brinson.csv" if not ATTR.empty else None,
            "flow_estimates": "flow_estimates.csv" if not FLO.empty else None,
            "rebalancing": "rebalancing.csv" if not REB.empty else None,
            "currency_attribution": "currency_attribution.csv" if not CUR.empty else None,
            "risk_metrics": "risk_metrics.csv" if not RM.empty else None,
            "scenarios": "scenarios.csv" if not SCN.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        positions=args.positions, returns=(args.returns or None), bench=(args.bench or None), fx=(args.fx or None),
        daily=(args.daily or None), events=(args.events or None), amap=(args.amap or None),
        outdir=args.outdir, band_low=float(args.band_low), band_high=float(args.band_high),
        window=int(args.window), eq_shock_jp=float(args.eq_shock_jp), eq_shock_fx=float(args.eq_shock_fx),
        jgb_bp=float(args.jgb_bp), global_bp=float(args.global_bp), usdjpy_pct=float(args.usdjpy_pct),
        hedge_shift=float(args.hedge_shift)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== GPIF Toolkit ==")
    print(f"Sample: {summary['sample']['start']} → {summary['sample']['end']}  ({summary['sample']['months']} months)")
    if summary["outputs"]["attribution_brinson"]: print("Brinson–Fachler attribution computed.")
    if summary["outputs"]["flow_estimates"]: print("Flow estimates written.")
    if summary["outputs"]["rebalancing"]: print("Rebalancing signals (band) computed.")
    if summary["outputs"]["currency_attribution"]: print("FX attribution computed.")
    if summary["outputs"]["risk_metrics"]: print("Risk metrics (vol/TE/IR) & VaR/ES computed.")
    if summary["outputs"]["scenarios"]: print("Scenario P&L generated.")
    if summary["outputs"]["event_study"]: print("Event study (daily) around policy events written.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
