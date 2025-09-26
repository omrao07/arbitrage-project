#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scandi_soverign_fund.py — Analytics kit for Scandinavian sovereign & public funds
(think: Norway GPFG, Sweden AP funds, Denmark ATP, etc.)

What this does
--------------
Takes flexible inputs (NAV/flows, allocations, returns, FX, benchmarks, ESG) and produces:

Core outputs (CSV)
------------------
- fund_timeseries.csv       Per-fund monthly TWR, asset return, currency/hedge effect, flows, NAV, weights
- attribution.csv           Brinson-Like attribution (Allocation, Selection, Currency) by period & YTD
- tracking_error.csv        TE & IR vs composite benchmark (annualized)
- risk_drawdowns.csv        Max DD, recovery stats, vol (ann), Sharpe (if rf provided)
- liquidity_stress.csv      30/60/90-day liquidity coverage vs assumed outflow
- policy_bands.csv          Compliance vs policy ranges (weights or asset limits)
- decarb_path.csv           Carbon intensity vs target path (if ESG file provided)
- scenarios_out.csv         Scenario P/L for shocks (equity, rates, spreads, FX, hedge ratio override)
- peer_comparison.csv       Simple peer KPIs table (if provided)
- summary.json              KPIs per fund (latest NAV, YTD, 5y ann, DD, TE, IR)
- config.json               Reproducibility dump

Input files (CSV; headers are case-insensitive & flexible)
----------------------------------------------------------
--funds funds.csv           REQUIRED (fund metadata)
  Columns:
    fund, base_ccy, hedge_ratio (0..1), rf_eur_m (optional monthly RF if specific), policy_json (optional JSON)
    Example policy_json fields: {"bands":{"equity":[0.6,0.8],"credit":[0.1,0.3],"real_estate":[0.0,0.1]}}

--nav nav.csv               REQUIRED (monthly)
  Columns: date, fund, nav (EUR or base_ccy; needs to be consistent per fund)

--flows flows.csv           OPTIONAL (monthly)
  Columns: date, fund, flow (contrib + / withdrawal -) in same units as NAV

--alloc alloc.csv           OPTIONAL (monthly allocations per fund)
  Columns (any of):
    date, fund, asset_class, weight (0..1) or mv (same units as NAV)
    If mv provided and weight missing, weights are inferred per fund-month

--returns returns.csv       OPTIONAL (monthly asset-class total returns)
  Columns: date, asset_class, return (decimal) [, fund(optional)]
  If fund missing, same returns applied to all funds

--bench bench.csv           OPTIONAL (monthly benchmark returns)
  Columns: date, asset_class, bench_return (decimal) [, fund(optional)]
  Bench weights can be supplied in alloc_bench.csv (below). If missing, we use the same weights as portfolio.

--alloc_bench alloc_bench.csv  OPTIONAL (monthly benchmark weights by fund)
  Columns: date, fund, asset_class, bench_weight (0..1)

--fx fx.csv                 OPTIONAL (monthly FX levels; for currency effects)
  Columns: date, ccy, rate_to_base
  Notes: rate_to_base must be price of 1 unit of ccy in fund base currency (e.g. USDNOK if base=NOK & ccy=USD).
         If you only have cross rates vs EUR or USD, provide both and the script will back out pairs if base_ccy given.

--ccy_exposure ccy_exposure.csv OPTIONAL (monthly currency exposure weights by fund)
  Columns: date, fund, ccy, weight (0..1). If missing, currency effect not computed (assumed hedged per hedge_ratio).

--esg esg.csv               OPTIONAL
  Columns: date, fund, carbon_intensity (tCO2e per $m or tCO2e per EVIC), target_annual_reduction_pct

--liquidity liquidity.csv   OPTIONAL (asset liquidity buckets)
  Columns: asset_class, bucket_days (e.g., 1, 7, 30, 90), haircut_pct (0..100)

--peers peers.csv           OPTIONAL (summary peer metrics)
  Columns: peer, region, aum_eur_bn, ytd_pct, one_year_pct, five_year_ann_pct, max_dd_pct

--rf rf.csv                 OPTIONAL (monthly risk-free per fund or base currency)
  Columns: date, fund(or base_ccy), rf_monthly (decimal)

--scenarios scenarios.csv   OPTIONAL
  Columns: scenario, name, key, value
  Supported keys (examples):
    equity.shock_pct = -15
    rates.shock_bp = +100
    credit.spread_bp = +150
    fx.NOK_pct = -5          (base currency appreciation (–) vs basket; or supply specific leg fx.USD_pct, fx.EUR_pct etc.)
    hedge.override = 0.50
    outflow.pct_nav_30d = 5  (liquidity outflow assumption)

CLI
---
--start 2012-01 --end 2025-12
--scenario baseline
--outdir out_scandi_funds

Notes
-----
- Portfolio return (TWR) per month: (NAV_t - NAV_{t-1} - Flow)/NAV_{t-1}
- Currency effect is approximate: sum_ccy( exposure_ccy * (1-hedge_ratio) * FX_return_ccy_to_base )
- Brinson-like attribution requires both asset returns and benchmark returns; otherwise only total vs simple bench.
- Policy compliance reads "bands" from policy_json; if alloc missing for an asset, it's treated as 0.

DISCLAIMER: For research only. No investment advice.
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

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def ann_factor(freq: str = "M") -> float:
    return 12.0 if freq.upper().startswith("M") else 252.0

def safe_div(a, b):
    try:
        a = float(a); b = float(b)
        return a / b if b != 0 else np.nan
    except Exception:
        return np.nan

def dd_stats(returns: pd.Series) -> Dict[str, float]:
    """Max drawdown & recovery length from monthly total returns (decimal)."""
    if returns.dropna().empty:
        return {"max_dd": np.nan, "dd_start": None, "dd_end": None, "rec_months": np.nan}
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    min_dd = dd.min()
    end_idx = dd.idxmin()
    # start at prior peak
    start_idx = (peak.loc[:end_idx]).idxmax()
    # recovery
    rec = cum.loc[end_idx:]
    rec_idx = rec[rec >= peak.loc[start_idx]].index
    rec_months = (rec_idx[0].to_period("M") - end_idx.to_period("M")).n if len(rec_idx) > 0 else np.nan
    return {"max_dd": float(min_dd), "dd_start": str(start_idx.date()) if start_idx is not None else None,
            "dd_end": str(end_idx.date()) if end_idx is not None else None, "rec_months": float(rec_months) if rec_months==rec_months else np.nan}

def rolling_annualized_vol(r: pd.Series) -> float:
    if r.dropna().empty: return np.nan
    return float(r.std(ddof=0) * np.sqrt(ann_factor()))

def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty: return np.nan
    years = (r.index[-1].to_period("M").ordinal - r.index[0].to_period("M").ordinal) / 12.0 + 1e-9
    cum = (1 + r).prod()
    return float(cum ** (1/years) - 1.0)

def ytd_return(r: pd.Series) -> float:
    if r.dropna().empty: return np.nan
    y = r.index[-1].year
    s = r[r.index.year == y]
    if s.empty: return np.nan
    return float((1 + s).prod() - 1.0)

def infer_weights_from_mv(panel: pd.DataFrame, value_col: str, w_col: str = "weight") -> pd.DataFrame:
    df = panel.copy()
    sumv = df.groupby(["date","fund"], as_index=False)[value_col].sum().rename(columns={value_col:"_sum_mv"})
    df = df.merge(sumv, on=["date","fund"], how="left")
    df[w_col] = df[value_col] / df["_sum_mv"]
    df.drop(columns=["_sum_mv"], inplace=True)
    return df

def pivot_sum(df: pd.DataFrame, index_cols: List[str], col_col: str, val_col: str) -> pd.DataFrame:
    return df.pivot_table(index=index_cols, columns=col_col, values=val_col, aggfunc="sum").reset_index().rename_axis(None, axis=1)

def as_json(x) -> dict:
    try:
        if isinstance(x, dict): return x
        return json.loads(str(x))
    except Exception:
        return {}

# ----------------------------- loaders -----------------------------

def load_funds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"fund") or df.columns[0]): "fund",
        (ncol(df,"base_ccy") or "base_ccy"): "base_ccy",
        (ncol(df,"hedge_ratio") or "hedge_ratio"): "hedge_ratio",
        (ncol(df,"policy_json") or "policy_json"): "policy_json",
        (ncol(df,"rf_eur_m") or "rf_eur_m"): "rf_eur_m",
    }
    df = df.rename(columns=ren)
    for c in ["hedge_ratio","rf_eur_m"]:
        if c in df.columns: df[c] = num(df[c])
    df["policy_json"] = df.get("policy_json", "").apply(as_json)
    return df

def load_nav(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"fund") or "fund"): "fund",
           (ncol(df,"nav") or ncol(df,"aum") or "nav"): "nav"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["nav"] = num(df["nav"])
    return df.sort_values(["fund","date"])

def load_flows(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"fund") or "fund"): "fund",
           (ncol(df,"flow") or "flow"): "flow"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["flow"] = num(df["flow"])
    return df.sort_values(["fund","date"])

def load_alloc(path: Optional[str], bench: bool=False) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"fund") or "fund"): "fund",
           (ncol(df,"asset_class") or ncol(df,"asset") or "asset_class"): "asset_class"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    if bench:
        bw = ncol(df,"bench_weight") or "bench_weight"
        df[bw] = num(df[bw])
        df = df.rename(columns={bw: "bench_weight"})
    else:
        if ncol(df,"weight"):
            df["weight"] = num(df[ncol(df,"weight")])
        if ncol(df,"mv"):
            df["mv"] = num(df[ncol(df,"mv")])
            if "weight" not in df.columns or df["weight"].isna().all():
                df = infer_weights_from_mv(df, "mv", "weight")
    return df.sort_values(["fund","date","asset_class"])

def load_returns(path: Optional[str], colname: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"asset_class") or "asset_class"): "asset_class"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df[colname] = num(df[ncol(df, colname)] if ncol(df, colname) else df[df.columns[-1]])
    if ncol(df,"fund"): df["fund"] = df[ncol(df,"fund")]
    return df.sort_values(["date","asset_class"])

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"ccy") or "ccy"): "ccy",
           (ncol(df,"rate_to_base") or "rate_to_base"): "rate_to_base"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["ccy"] = df["ccy"].astype(str).str.upper()
    df["rate_to_base"] = num(df["rate_to_base"])
    return df.sort_values(["ccy","date"])

def load_ccy_exp(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"fund") or "fund"): "fund",
           (ncol(df,"ccy") or "ccy"): "ccy",
           (ncol(df,"weight") or "weight"): "weight"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["ccy"] = df["ccy"].astype(str).str.upper()
    df["weight"] = num(df["weight"])
    return df.sort_values(["fund","date","ccy"])

def load_esg(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"fund") or "fund"): "fund",
           (ncol(df,"carbon_intensity") or "carbon_intensity"): "carbon_intensity",
           (ncol(df,"target_annual_reduction_pct") or "target_annual_reduction_pct"): "target_annual_reduction_pct"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    for c in ["carbon_intensity","target_annual_reduction_pct"]:
        if c in df.columns: df[c] = num(df[c])
    return df.sort_values(["fund","date"])

def load_liquidity(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"asset_class") or "asset_class"): "asset_class",
           (ncol(df,"bucket_days") or "bucket_days"): "bucket_days",
           (ncol(df,"haircut_pct") or "haircut_pct"): "haircut_pct"}
    df = df.rename(columns=ren)
    df["bucket_days"] = num(df["bucket_days"])
    df["haircut_pct"] = num(df["haircut_pct"])
    return df

def load_rf(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    date_c = (ncol(df,"date") or df.columns[0])
    df = df.rename(columns={date_c:"date"})
    df["date"] = to_month(df["date"])
    # Allow either 'fund' or 'base_ccy'
    if ncol(df,"fund"):
        df["fund"] = df[ncol(df,"fund")]
    elif ncol(df,"base_ccy"):
        df["fund"] = df[ncol(df,"base_ccy")]
    df["rf_monthly"] = num(df[ncol(df,"rf_monthly")] if ncol(df,"rf_monthly") else df[df.columns[-1]])
    return df[["date","fund","rf_monthly"]]

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"): "scenario",
           (ncol(df,"name") or "name"): "name",
           (ncol(df,"key") or "key"): "key",
           (ncol(df,"value") or "value"): "value"}
    return df.rename(columns=ren)


# ----------------------------- core calculations -----------------------------

def compute_twr(nav: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    d = nav.copy()
    d = d.merge(flows, on=["date","fund"], how="left") if not flows.empty else d.assign(flow=0.0)
    d = d.sort_values(["fund","date"])
    d["nav_lag"] = d.groupby("fund")["nav"].shift(1)
    d["flow"] = d["flow"].fillna(0.0)
    d["twr"] = (d["nav"] - d["nav_lag"] - d["flow"]) / d["nav_lag"]
    return d.drop(columns=["nav_lag"])

def portfolio_asset_return(alloc: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    if alloc.empty or rets.empty:
        return pd.DataFrame()
    A = alloc.copy()
    R = rets.copy()
    if "fund" not in R.columns:
        # replicate for all funds in alloc
        funds = A["fund"].unique()
        R = R.assign(key=1).merge(pd.DataFrame({"fund": funds, "key":[1]*len(funds)}), on="key", how="outer").drop(columns=["key"])
    M = A.merge(R, on=["date","fund","asset_class"], how="left")
    M["contrib"] = M["weight"].fillna(0.0) * M["return"].fillna(0.0)
    out = (M.groupby(["date","fund"], as_index=False)
             .agg(asset_return=("contrib","sum"))
           )
    # currency exposure via ccy_exposure file is handled elsewhere
    return out

def benchmark_return(alloc_b: pd.DataFrame, bench_ret: pd.DataFrame, fallback_alloc: pd.DataFrame) -> pd.DataFrame:
    # if no explicit bench weights, fall back to portfolio weights
    if alloc_b.empty and fallback_alloc.empty:
        return pd.DataFrame()
    if bench_ret.empty:
        return pd.DataFrame()
    if alloc_b.empty:
        alloc_b = fallback_alloc.rename(columns={"weight":"bench_weight"})
    B = alloc_b.copy()
    Rb = bench_ret.copy()
    if "fund" not in Rb.columns:
        funds = B["fund"].unique()
        Rb = Rb.assign(key=1).merge(pd.DataFrame({"fund": funds, "key":[1]*len(funds)}), on="key", how="outer").drop(columns=["key"])
    M = B.merge(Rb, on=["date","fund","asset_class"], how="left")
    M["contrib_b"] = M["bench_weight"].fillna(0.0) * M["bench_return"].fillna(0.0)
    out = (M.groupby(["date","fund"], as_index=False)
             .agg(bench_total_return=("contrib_b","sum")))
    return out

def brinson_attribution(alloc: pd.DataFrame, port_ret: pd.DataFrame, alloc_b: pd.DataFrame, bench_ret: pd.DataFrame) -> pd.DataFrame:
    """Allocation & Selection vs benchmark by month: A = Σ (w_p - w_b)*r_b ; S = Σ w_b*(r_p - r_b) + Σ(w_p - w_b)*(r_p - r_b) if using interaction; keep simple A+S."""
    if alloc.empty or bench_ret.empty:
        return pd.DataFrame()
    # Harmonize inputs
    A = alloc.copy()
    B = alloc_b.copy()
    if B.empty:
        B = A.rename(columns={"weight":"bench_weight"})[["date","fund","asset_class","bench_weight"]]
    R_p = port_ret.copy().rename(columns={"return":"p_ret"})
    R_b = bench_ret.copy().rename(columns={"bench_return":"b_ret"})
    # replicate if returns missing fund dimension
    if "fund" not in R_p.columns:
        funds = A["fund"].unique()
        R_p = R_p.assign(key=1).merge(pd.DataFrame({"fund":funds,"key":[1]*len(funds)}), on="key", how="outer").drop(columns=["key"])
    if "fund" not in R_b.columns:
        funds = A["fund"].unique()
        R_b = R_b.assign(key=1).merge(pd.DataFrame({"fund":funds,"key":[1]*len(funds)}), on="key", how="outer").drop(columns=["key"])
    M = A.merge(B, on=["date","fund","asset_class"], how="outer").fillna({"weight":0.0,"bench_weight":0.0})
    M = M.merge(R_p, on=["date","fund","asset_class"], how="left").merge(R_b, on=["date","fund","asset_class"], how="left")
    M["p_ret"] = M["p_ret"].fillna(M["b_ret"]).fillna(0.0)
    M["b_ret"] = M["b_ret"].fillna(0.0)
    M["alloc_eff"] = (M["weight"] - M["bench_weight"]) * M["b_ret"]
    M["sel_eff"]   = M["weight"] * (M["p_ret"] - M["b_ret"])
    out = (M.groupby(["date","fund"], as_index=False)
             .agg(allocation=("alloc_eff","sum"), selection=("sel_eff","sum"))
          )
    out["currency"] = 0.0  # filled later if FX provided
    out["total_attr"] = out["allocation"] + out["selection"] + out["currency"]
    return out

def currency_effect(ccy_exp: pd.DataFrame, fx: pd.DataFrame, funds_meta: pd.DataFrame) -> pd.DataFrame:
    if ccy_exp.empty or fx.empty or "hedge_ratio" not in funds_meta.columns:
        return pd.DataFrame()
    # compute FX returns per ccy: r_fx = rate_t / rate_{t-1} - 1  (rate_to_base)
    FX = fx.copy().sort_values(["ccy","date"])
    FX["fx_ret"] = FX.groupby("ccy")["rate_to_base"].pct_change()
    E = ccy_exp.copy()
    out = E.merge(FX[["date","ccy","fx_ret"]], on=["date","ccy"], how="left")
    # attach hedge ratio per fund
    hr = funds_meta[["fund","hedge_ratio"]]
    out = out.merge(hr, on="fund", how="left")
    out["ccy_contrib"] = out["weight"].fillna(0.0) * (1.0 - out["hedge_ratio"].fillna(0.0)) * out["fx_ret"].fillna(0.0)
    cur = (out.groupby(["date","fund"], as_index=False).agg(currency=("ccy_contrib","sum")))
    return cur

def tracking_error(active_ret: pd.Series) -> Tuple[float, float]:
    te = float(active_ret.std(ddof=0) * np.sqrt(ann_factor()))
    ar = float((1 + active_ret).prod() ** (ann_factor()/len(active_ret.dropna())) - 1.0) if active_ret.dropna().size>0 else np.nan
    ir = float(ar / te) if te and te==te and te>0 else np.nan
    return te, ir

def liquidity_coverage(alloc: pd.DataFrame, nav: pd.DataFrame, liq: pd.DataFrame, outflow_pct: float, horizons=[30,60,90]) -> pd.DataFrame:
    if alloc.empty or liq.empty or nav.empty:
        return pd.DataFrame()
    # latest date per fund
    latest = nav.groupby("fund")["date"].max().reset_index()
    NAV = nav.merge(latest, on=["fund","date"], how="inner")[["fund","date","nav"]]
    W = alloc[alloc["date"].isin(NAV["date"].unique())][["date","fund","asset_class","weight"]]
    L = liq.copy()
    # Haircut and availability by horizon
    rows = []
    for _, r in NAV.iterrows():
        sub = W[(W["fund"]==r["fund"]) & (W["date"]==r["date"])]
        if sub.empty: 
            continue
        for H in horizons:
            avail = 0.0
            base = float(r["nav"])
            for _, a in sub.iterrows():
                aset = str(a["asset_class"])
                w = float(a["weight"])
                li = L[L["asset_class"]==aset]
                # Find best (<= H days) bucket; if none, skip
                liH = li[li["bucket_days"]<=H]
                if liH.empty: 
                    continue
                hc = float(liH.sort_values("bucket_days").iloc[-1]["haircut_pct"]) if not liH.empty else 100.0
                net = w * base * (1.0 - hc/100.0)
                avail += max(net, 0.0)
            need = outflow_pct/100.0 * base
            cov = safe_div(avail, need) if need>0 else np.nan
            rows.append({"fund": r["fund"], "date": r["date"], "horizon_days": H, "available_eur": avail, "need_eur": need, "coverage_x": cov})
    return pd.DataFrame(rows)

def policy_compliance(alloc: pd.DataFrame, funds_meta: pd.DataFrame) -> pd.DataFrame:
    if alloc.empty: return pd.DataFrame()
    latest = alloc.groupby(["fund"])["date"].max().reset_index().rename(columns={"date":"latest"})
    A = alloc.merge(latest, on="fund", how="left")
    A = A[A["date"]==A["latest"]]
    rows = []
    for f, g in A.groupby("fund"):
        pol = as_json(funds_meta.set_index("fund").loc[f, "policy_json"]) if "policy_json" in funds_meta.columns and f in funds_meta["fund"].values else {}
        bands = pol.get("bands", {})
        for k, band in bands.items():
            lo, hi = float(band[0]), float(band[1])
            w = float(g[g["asset_class"].str.lower()==k.lower()]["weight"].sum())
            ok = (w >= lo) and (w <= hi)
            rows.append({"fund": f, "asset_class": k, "weight": w, "band_lo": lo, "band_hi": hi, "in_band": bool(ok)})
    return pd.DataFrame(rows)

def decarb_path(esg: pd.DataFrame) -> pd.DataFrame:
    if esg.empty: return pd.DataFrame()
    out = []
    for (f), g in esg.groupby(["fund"]):
        g = g.sort_values("date").copy()
        if "target_annual_reduction_pct" not in g.columns or g["target_annual_reduction_pct"].isna().all():
            g["target_annual_reduction_pct"] = 0.0
        base = float(g["carbon_intensity"].dropna().iloc[0]) if g["carbon_intensity"].notna().any() else np.nan
        for i, r in g.iterrows():
            years = max(0, r["date"].year - g["date"].min().year)
            target = base * ((1 - r["target_annual_reduction_pct"]/100.0) ** years) if base==base else np.nan
            out.append({"date": r["date"], "fund": f, "carbon_intensity": r["carbon_intensity"], "target_path": target, "gap": (r["carbon_intensity"] - target) if (target==target and r["carbon_intensity"]==r["carbon_intensity"]) else np.nan})
    return pd.DataFrame(out).sort_values(["fund","date"])

def scenario_pl(alloc: pd.DataFrame, funds_meta: pd.DataFrame, scen_df: pd.DataFrame, scenario: str, ccy_exp: pd.DataFrame) -> pd.DataFrame:
    if alloc.empty or scen_df.empty:
        return pd.DataFrame()
    sub = scen_df[scen_df["scenario"]==scenario]
    if sub.empty: return pd.DataFrame()
    keys = {k.lower(): v for k, v in zip(sub["key"].str.lower(), sub["value"])}
    # Pick latest weights per fund
    latest = alloc.groupby("fund")["date"].max().reset_index().rename(columns={"date":"latest"})
    W = alloc.merge(latest, on="fund", how="left")
    W = W[W["date"]==W["latest"]].copy()
    # Default betas (per 1% equity shock, 100bp rate rise, 100bp spread widen)
    beta = {"equity": 1.0, "rates": -6.0, "spread": -3.0, "real_estate": -0.6, "infra": -0.4, "private_equity": 0.8, "cash": 0.0, "commodity": 0.3}
    def b_for_asset(a: str) -> Tuple[float,float,float]:
        x = a.lower()
        if "equity" in x or "stock" in x: return (beta["equity"], 0.0, 0.0)
        if "gov" in x or "sovereign" in x or "rates" in x or "duration" in x or "bond" in x and "credit" not in x: return (0.0, beta["rates"], 0.0)
        if "credit" in x or "corp" in x or "ig" in x or "hy" in x or "em credit" in x: return (0.2, beta["rates"], beta["spread"])
        if "real" in x and "estate" in x: return (beta["real_estate"], -1.0, -0.5)
        if "infra" in x: return (beta["infra"], -1.5, -0.5)
        if "private" in x and "equity" in x: return (beta["private_equity"], 0.0, 0.0)
        if "cash" in x: return (0.0, 0.0, 0.0)
        if "commodity" in x: return (beta["commodity"], 0.0, 0.0)
        return (0.2, -1.0, -0.5)
    eq_shock = float(keys.get("equity.shock_pct", 0.0))
    rt_bp    = float(keys.get("rates.shock_bp", 0.0))
    sp_bp    = float(keys.get("credit.spread_bp", 0.0))
    hedge_override = keys.get("hedge.override", None)
    # Currency shock as % base move; if detailed legs provided (fx.USD_pct etc.), we would need mapping from ccy_exp
    fx_base_pct = float(keys.get("fx.base_pct", keys.get("fx.nok_pct", keys.get("fx.sek_pct", keys.get("fx.dkk_pct", 0.0)))))
    rows = []
    for f, g in W.groupby("fund"):
        w = g[["asset_class","weight"]]
        pl = 0.0
        for _, r in w.iterrows():
            be, br, bs = b_for_asset(str(r["asset_class"]))
            dr = (be * (eq_shock/100.0)) + (br * (rt_bp/100.0)) + (bs * (sp_bp/100.0))
            pl += float(r["weight"]) * dr
        # FX effect using currency exposures if provided
        fx_contrib = 0.0
        if not ccy_exp.empty and (f in ccy_exp["fund"].unique()):
            exp = ccy_exp[ccy_exp["fund"]==f].copy()
            hr = float(funds_meta.set_index("fund").loc[f, "hedge_ratio"]) if "hedge_ratio" in funds_meta.columns else 0.0
            if hedge_override is not None:
                try: hr = float(hedge_override)
                except Exception: pass
            fx_contrib = float(((exp["weight"] * (1-hr)).sum()) * (fx_base_pct/100.0))
        rows.append({"fund": f, "scenario": scenario, "equity_shock_pct": eq_shock, "rates_shock_bp": rt_bp, "spread_shock_bp": sp_bp,
                     "fx_base_pct": fx_base_pct, "hedge_ratio_used": float(hedge_override) if hedge_override is not None else (float(hr) if 'hr' in locals() else np.nan),
                     "portfolio_pl_pct": pl + fx_contrib})
    return pd.DataFrame(rows)

# ----------------------------- CLI -----------------------------

@dataclass
class Config:
    funds: str
    nav: str
    flows: Optional[str]
    alloc: Optional[str]
    returns: Optional[str]
    bench: Optional[str]
    alloc_bench: Optional[str]
    fx: Optional[str]
    ccy_exposure: Optional[str]
    esg: Optional[str]
    liquidity: Optional[str]
    peers: Optional[str]
    rf: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    scenario: str
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scandinavian sovereign/public funds analytics")
    ap.add_argument("--funds", required=True)
    ap.add_argument("--nav", required=True)
    ap.add_argument("--flows", default="")
    ap.add_argument("--alloc", default="")
    ap.add_argument("--returns", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--alloc_bench", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--ccy_exposure", default="")
    ap.add_argument("--esg", default="")
    ap.add_argument("--liquidity", default="")
    ap.add_argument("--peers", default="")
    ap.add_argument("--rf", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2012-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--outdir", default="out_scandi_funds")
    return ap.parse_args()


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    funds = load_funds(args.funds)
    nav = load_nav(args.nav)
    flows = load_flows(args.flows) if args.flows else pd.DataFrame()
    alloc = load_alloc(args.alloc, bench=False) if args.alloc else pd.DataFrame()
    rets = load_returns(args.returns, "return") if args.returns else pd.DataFrame()
    bench_ret = load_returns(args.bench, "bench_return") if args.bench else pd.DataFrame()
    alloc_b = load_alloc(args.alloc_bench, bench=True) if args.alloc_bench else pd.DataFrame()
    fx = load_fx(args.fx) if args.fx else pd.DataFrame()
    cexp = load_ccy_exp(args.ccy_exposure) if args.ccy_exposure else pd.DataFrame()
    esg = load_esg(args.esg) if args.esg else pd.DataFrame()
    liq = load_liquidity(args.liquidity) if args.liquidity else pd.DataFrame()
    peers = pd.read_csv(args.peers) if args.peers else pd.DataFrame()
    rf = load_rf(args.rf) if args.rf else pd.DataFrame()
    scenarios_df = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Filter by window
    start = pd.to_datetime(args.start).to_period("M").to_timestamp()
    end   = pd.to_datetime(args.end).to_period("M").to_timestamp()
    for df in [nav, flows, alloc, rets, bench_ret, alloc_b, fx, cexp, esg, rf]:
        if not df.empty:
            df = df[(df["date"]>=start) & (df["date"]<=end)]
            if df is nav: nav = df
            elif df is flows: flows = df
            elif df is alloc: alloc = df
            elif df is rets: rets = df
            elif df is bench_ret: bench_ret = df
            elif df is alloc_b: alloc_b = df
            elif df is fx: fx = df
            elif df is cexp: cexp = df
            elif df is esg: esg = df
            elif df is rf: rf = df

    # Compute portfolio TWR
    twr = compute_twr(nav, flows)
    # Portfolio asset return (if alloc + returns exist)
    pa = portfolio_asset_return(alloc, rets) if (not alloc.empty and not rets.empty) else pd.DataFrame()
    # Benchmark total return
    btot = benchmark_return(alloc_b, bench_ret, alloc) if (not bench_ret.empty) else pd.DataFrame()

    # Currency effect (optional)
    cur = currency_effect(cexp, fx, funds) if (not cexp.empty and not fx.empty) else pd.DataFrame()

    # Join into fund timeseries
    ft = twr.merge(pa, on=["date","fund"], how="left").merge(btot, on=["date","fund"], how="left")
    ft = ft.merge(cur, on=["date","fund"], how="left") if not cur.empty else ft.assign(currency=np.nan)
    # Hedge effect is proxied by 'currency' here; device to show columns clearly
    ft["net_return"] = ft["twr"]
    ft["asset_return"] = ft.get("asset_return", np.nan)
    ft["bench_total_return"] = ft.get("bench_total_return", np.nan)
    ft["currency_effect"] = ft.get("currency", np.nan)
    # Add RF if available
    if not rf.empty:
        ft = ft.merge(rf, on=["date","fund"], how="left")
    ft.to_csv(outdir / "fund_timeseries.csv", index=False)

    # Attribution
    attr = pd.DataFrame()
    if not alloc.empty and not rets.empty and not bench_ret.empty:
        attr = brinson_attribution(alloc, rets, alloc_b, bench_ret)
        # If we computed currency effect, add it in
        if not cur.empty:
            attr = attr.merge(cur, on=["date","fund"], how="left")
            attr["total_attr"] = attr["allocation"] + attr["selection"] + attr["currency"]
        attr.to_csv(outdir / "attribution.csv", index=False)

    # Tracking Error & Information Ratio
    te_rows = []
    if not ft.empty and "bench_total_return" in ft.columns:
        for f, g in ft.groupby("fund"):
            s = (g["net_return"] - g["bench_total_return"]).dropna()
            if len(s) >= 12:
                te, ir = tracking_error(s)
                te_rows.append({"fund": f, "tracking_error_ann": te, "information_ratio": ir, "active_return_ann": ann_return(g["net_return"] - g["bench_total_return"])})
            else:
                te_rows.append({"fund": f, "tracking_error_ann": np.nan, "information_ratio": np.nan, "active_return_ann": np.nan})
    te_df = pd.DataFrame(te_rows)
    te_df.to_csv(outdir / "tracking_error.csv", index=False)

    # Risk & drawdowns
    rd_rows = []
    for f, g in ft.groupby("fund"):
        r = g["net_return"]
        dd = dd_stats(r)
        rd_rows.append({"fund": f, "vol_ann": rolling_annualized_vol(r), "ytd": ytd_return(r), "ann_5y": ann_return(r.tail(min(60, len(r)))), "max_dd": dd["max_dd"], "dd_start": dd["dd_start"], "dd_end": dd["dd_end"], "dd_recovery_months": dd["rec_months"]})
    rd = pd.DataFrame(rd_rows)
    rd.to_csv(outdir / "risk_drawdowns.csv", index=False)

    # Liquidity stress
    scen_outflow = None
    if not scenarios_df.empty and args.scenario in scenarios_df["scenario"].unique():
        sub = scenarios_df[scenarios_df["scenario"]==args.scenario]
        if not sub[sub["key"].str.lower()=="outflow.pct_nav_30d"].empty:
            try: scen_outflow = float(sub[sub["key"].str.lower()=="outflow.pct_nav_30d"]["value"].iloc[0])
            except Exception: scen_outflow = None
    if scen_outflow is None: scen_outflow = 5.0  # default 5% of NAV in 30d
    liq_cov = liquidity_coverage(alloc, nav, liq, scen_outflow) if (not alloc.empty and not liq.empty) else pd.DataFrame()
    if not liq_cov.empty:
        liq_cov.to_csv(outdir / "liquidity_stress.csv", index=False)

    # Policy bands
    pol = policy_compliance(alloc, funds) if not alloc.empty else pd.DataFrame()
    if not pol.empty:
        pol.to_csv(outdir / "policy_bands.csv", index=False)

    # Decarbonization path
    dec = decarb_path(esg) if not esg.empty else pd.DataFrame()
    if not dec.empty:
        dec.to_csv(outdir / "decarb_path.csv", index=False)

    # Scenario P/L
    scen = scenario_pl(alloc, funds, scenarios_df, args.scenario, cexp) if not alloc.empty and not scenarios_df.empty else pd.DataFrame()
    if not scen.empty:
        scen.to_csv(outdir / "scenarios_out.csv", index=False)

    # Peer table passthrough (optional)
    if not peers.empty:
        peers.to_csv(outdir / "peer_comparison.csv", index=False)

    # Summary per fund
    kpi_rows = []
    for f, g in ft.groupby("fund"):
        last_nav = float(g["nav"].iloc[-1]) if "nav" in g.columns and not g.empty else np.nan
        ytd = ytd_return(g["net_return"])
        ann5 = ann_return(g["net_return"].tail(min(60, len(g))))
        dd = dd_stats(g["net_return"])
        te_val = float(te_df[te_df["fund"]==f]["tracking_error_ann"].iloc[0]) if not te_df.empty and f in te_df["fund"].values else np.nan
        ir_val = float(te_df[te_df["fund"]==f]["information_ratio"].iloc[0]) if not te_df.empty and f in te_df["fund"].values else np.nan
        kpi_rows.append({
            "fund": f, "latest_date": str(g["date"].max().date()), "latest_nav": last_nav, "ytd": ytd, "ann_5y": ann5,
            "max_dd": dd["max_dd"], "dd_start": dd["dd_start"], "dd_end": dd["dd_end"],
            "tracking_error_ann": te_val, "information_ratio": ir_val
        })
    summary = {"funds": kpi_rows}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config
    cfg = asdict(Config(
        funds=args.funds, nav=args.nav, flows=(args.flows or None), alloc=(args.alloc or None), returns=(args.returns or None),
        bench=(args.bench or None), alloc_bench=(args.alloc_bench or None), fx=(args.fx or None), ccy_exposure=(args.ccy_exposure or None),
        esg=(args.esg or None), liquidity=(args.liquidity or None), peers=(args.peers or None), rf=(args.rf or None),
        scenarios=(args.scenarios or None), start=args.start, end=args.end, scenario=args.scenario, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Scandi Sovereign/Public Funds Analytics ==")
    for r in kpi_rows:
        print(f"{r['fund']}: NAV {r['latest_nav']:.2f} | YTD {r['ytd']*100: .2f}% | 5y ann {r['ann_5y']*100: .2f}% | TE {r['tracking_error_ann'] if r['tracking_error_ann']==r['tracking_error_ann'] else float('nan'): .2f} | IR {r['information_ratio'] if r['information_ratio']==r['information_ratio'] else float('nan'): .2f}")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
