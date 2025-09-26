#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swiss_wealth_flows.py — Swiss private banking & wealth management flow decomposition

What this does
--------------
Given bank-level AUM (CHF), asset weights, returns, FX and (optionally) currency/client-region/channel
breakdowns & fund vehicle flows, this script decomposes month-to-month AUM changes into:

  ΔAUM  =  Performance effect  +  FX effect  +  Net New Assets (NNA)

It then aggregates NNA by bank, currency, client domicile/region, and channel; builds a simple hedge ratio
inference; produces sensitivities (AUM beta to equity/rates/FX) and scenario impacts.

Core outputs (CSV)
------------------
- bank_timeseries.csv      Per bank per month: AUM, ΔAUM, perf_effect, fx_effect, NNA, organic_growth%
- perf_attribution.csv     Performance effect by asset class (per bank per month)
- fx_attribution.csv       FX effect by currency (per bank per month)
- flows_by_region.csv      NNA by client region/country (latest month & trailing sums)
- flows_by_currency.csv    NNA by exposure currency (if ccy_exposure provided)
- channel_mix.csv          Channel-level AUM & NNA (funds/mandates/advisory/custody) if provided
- inferred_hedge.csv       Inferred effective hedge ratio (per bank per month)
- sensitivities.csv        AUM sensitivities (ΔAUM per 1% equity, 100bp rates, 1% CHF TWI move)
- scenarios_out.csv        Scenario AUM impact given scenarios.csv (equity, rates, CHF, hedge override)
- summary.json             KPIs per bank: latest AUM, TTM NNA, organic growth, perf/FX split, hedge ratio
- config.json              Run configuration dump

Inputs (CSV; headers are case-insensitive & flexible)
-----------------------------------------------------
--aum aum.csv       REQUIRED (monthly)
  Columns: date, bank, aum_chf_m   [units: million CHF]
          Optional: segment (managed/mandate/custody/etc.) for channel views

--weights weights.csv   OPTIONAL (monthly portfolio weights)
  Columns: date, bank, asset_class, weight (0..1)  OR  mv_chf_m (we infer weights if mv provided)
  Typical asset_class: equity, fixed_income, cash, alternatives, real_estate, multi_asset, other

--returns returns.csv   OPTIONAL (monthly total returns by asset class, decimal)
  Columns: date, asset_class, return
  If missing, performance effect is not computed, and NNA defaults to ΔAUM.

--fx fx.csv             OPTIONAL (monthly FX levels to CHF; CHF per 1 unit of ccy)
  Columns: date, ccy, rate_to_chf
  We compute FX returns as pct_change of rate_to_chf (positive if CHF weakens vs ccy).

--ccy_exposure ccy.csv  OPTIONAL (monthly currency exposure weights by bank)
  Columns: date, bank, ccy, weight (0..1)
  Used for FX effect. If missing, we assume FX effect = 0 unless 'weights' includes 'cash_chf' proxy.

--domicile domicile.csv OPTIONAL (client region/country mix by bank)
  Columns: date, bank, region (or country), weight (0..1)  OR aum_chf_m

--channels channels.csv OPTIONAL (channel mix by bank)
  Columns: date, bank, channel (funds, mandates, advisory, custody, etc.), weight or aum_chf_m

--fund_flows fund_flows.csv OPTIONAL (vehicle-level measured flows)
  Columns: date, bank(or group), vehicle (fund/ETF/AM), flow_chf_m  [, ccy + flow_in_ccy]
  Used as an auxiliary signal; we reconcile to NNA if 'reconcile_fund_flows' flag is set.

--scenarios scenarios.csv OPTIONAL
  Columns: scenario, name, key, value
  Supported keys:
    equity.shock_pct   = -10        (broad equity total return shock)
    rates.shock_bp     = +100       (parallel CHF rates +100bp; negative for rally)
    chf.shock_pct      = +5         (broad CHF appreciation vs basket; + = CHF up)
    hedge.override     = 0.50       (override effective hedge ratio)
    nna.behavior_pp    = -50        (behavioural flow response in bp of AUM to equity shock; e.g., -50 = -0.5% of AUM)

Key options
-----------
--start 2018-01
--end   2025-12
--reconcile_fund_flows  (bool; default False) — blend measured vehicle flows with residual NNA
--outdir out_swiss_flows

Method notes (high level)
-------------------------
- Performance effect_t ≈ AUM_{t-1} × Σ_i (w_{i,t-1} × r_{i,t})
- FX effect_t ≈ AUM_{t-1} × Σ_c (exp_{c,t-1}^{unhedged} × fx_ret_{c→CHF,t}), where unhedged = exposure × (1-hedge_ratio).
  We infer an effective hedge ratio per bank by comparing observed FX effect vs unhedged exposure FX effect.
- NNA_t (residual) = ΔAUM_t - perf_t - fx_t
- Sensitivities: dAUM/d1% equity ≈ AUM × equity_weight; dAUM/d100bp rates ≈ -D_port × AUM (if fixed_income present; we use -6% per 100bp as a blunt proxy × FI weight)
- Scenario impacts = sensitivities × shocks + behavioural NNA (if specified).

DISCLAIMER: This is a research tool. It simplifies many realities of Swiss PB reporting. Not investment advice.
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
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None

def to_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def pivot_sum(df: pd.DataFrame, index_cols: List[str], col_col: str, val_col: str) -> pd.DataFrame:
    return df.pivot_table(index=index_cols, columns=col_col, values=val_col, aggfunc="sum").reset_index().rename_axis(None, axis=1)

def infer_weights_from_mv(panel: pd.DataFrame, value_col: str, w_col: str = "weight") -> pd.DataFrame:
    df = panel.copy()
    sumv = df.groupby(["date","bank"], as_index=False)[value_col].sum().rename(columns={value_col:"_sum_mv"})
    df = df.merge(sumv, on=["date","bank"], how="left")
    df[w_col] = df[value_col] / df["_sum_mv"]
    df.drop(columns=["_sum_mv"], inplace=True)
    return df

def ann_fac() -> float:
    return 12.0


# ----------------------------- loaders -----------------------------

def load_aum(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"bank") or "bank"): "bank",
           (ncol(df,"aum_chf_m") or ncol(df,"aum") or "aum_chf_m"): "aum_chf_m",
           (ncol(df,"segment") or "segment"): "segment"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["bank"] = df["bank"].astype(str)
    df["aum_chf_m"] = num(df["aum_chf_m"])
    return df.sort_values(["bank","date"])

def load_weights(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"bank") or "bank"): "bank",
           (ncol(df,"asset_class") or "asset_class"): "asset_class"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["bank"] = df["bank"].astype(str)
    if ncol(df,"weight"):
        df["weight"] = num(df[ncol(df,"weight")])
    if ncol(df,"mv_chf_m"):
        df["mv_chf_m"] = num(df[ncol(df,"mv_chf_m")])
        if "weight" not in df.columns or df["weight"].isna().all():
            df = infer_weights_from_mv(df, "mv_chf_m", "weight")
    return df.sort_values(["bank","date","asset_class"])

def load_returns(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"asset_class") or "asset_class"): "asset_class",
           (ncol(df,"return") or "return"): "return"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["return"] = num(df["return"])
    return df.sort_values(["date","asset_class"])

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"ccy") or "ccy"): "ccy",
           (ncol(df,"rate_to_chf") or "rate_to_chf"): "rate_to_chf"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["ccy"] = df["ccy"].astype(str).str.upper()
    df["rate_to_chf"] = num(df["rate_to_chf"])
    df = df.sort_values(["ccy","date"])
    df["fx_ret"] = df.groupby("ccy")["rate_to_chf"].pct_change()
    return df

def load_ccy_exp(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"bank") or "bank"): "bank",
           (ncol(df,"ccy") or "ccy"): "ccy",
           (ncol(df,"weight") or "weight"): "weight"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["bank"] = df["bank"].astype(str)
    df["ccy"] = df["ccy"].astype(str).str.upper()
    df["weight"] = num(df["weight"])
    return df.sort_values(["bank","date","ccy"])

def load_domicile(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"bank") or "bank"): "bank",
           (ncol(df,"region") or ncol(df,"country") or "region"): "region"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["bank"] = df["bank"].astype(str)
    if ncol(df,"weight"):
        df["weight"] = num(df[ncol(df,"weight")])
    if ncol(df,"aum_chf_m"):
        df["aum_chf_m"] = num(df[ncol(df,"aum_chf_m")])
        if "weight" not in df.columns or df["weight"].isna().all():
            df = infer_weights_from_mv(df.rename(columns={"aum_chf_m":"mv_chf_m"}), "mv_chf_m", "weight")
    return df.sort_values(["bank","date","region"])

def load_channels(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"bank") or "bank"): "bank",
           (ncol(df,"channel") or "channel"): "channel"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["bank"] = df["bank"].astype(str)
    if ncol(df,"weight"):
        df["weight"] = num(df[ncol(df,"weight")])
    if ncol(df,"aum_chf_m"):
        df["aum_chf_m"] = num(df[ncol(df,"aum_chf_m")])
        if "weight" not in df.columns or df["weight"].isna().all():
            df = infer_weights_from_mv(df.rename(columns={"aum_chf_m":"mv_chf_m"}), "mv_chf_m", "weight")
    return df.sort_values(["bank","date","channel"])

def load_fund_flows(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]): "date",
           (ncol(df,"bank") or ncol(df,"group") or "bank"): "bank",
           (ncol(df,"vehicle") or "vehicle"): "vehicle",
           (ncol(df,"flow_chf_m") or "flow_chf_m"): "flow_chf_m"}
    df = df.rename(columns=ren)
    df["date"] = to_month(df["date"])
    df["bank"] = df["bank"].astype(str)
    df["flow_chf_m"] = num(df["flow_chf_m"])
    # If flow given in foreign ccy with 'ccy' + 'flow_in_ccy', user should pre-convert; keeping simple here.
    return df.sort_values(["bank","date","vehicle"])

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","name","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"): "scenario",
           (ncol(df,"name") or "name"): "name",
           (ncol(df,"key") or "key"): "key",
           (ncol(df,"value") or "value"): "value"}
    return df.rename(columns=ren)


# ----------------------------- core decomposition -----------------------------

def last_known_weights(W: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill weights per bank/asset to cover months without explicit weights."""
    if W.empty: return W
    W = W.sort_values(["bank","asset_class","date"]).copy()
    W["weight"] = W.groupby(["bank","asset_class"])["weight"].ffill()
    # Normalize to 1 per bank-month if possible
    sums = W.groupby(["bank","date"], as_index=False)["weight"].sum().rename(columns={"weight":"w_sum"})
    W = W.merge(sums, on=["bank","date"], how="left")
    W.loc[W["w_sum"].notna() & (W["w_sum"]>0), "weight"] = W["weight"] / W["w_sum"]
    W.drop(columns=["w_sum"], inplace=True)
    return W

def perf_effect(aum: pd.DataFrame, weights: pd.DataFrame, rets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      perf_tbl: per bank-month performance effect (CHF m)
      perf_attr: per bank-month by asset class (CHF m)
    """
    if aum.empty or rets.empty or weights.empty:
        return pd.DataFrame(), pd.DataFrame()
    A = aum.copy().sort_values(["bank","date"])
    A["aum_lag"] = A.groupby("bank")["aum_chf_m"].shift(1)
    W = last_known_weights(weights)
    R = rets.copy()
    M = W.merge(R, on=["date","asset_class"], how="left")
    # Align to AUM lag month weights (t-1 weights with t returns)
    M["date_lag"] = M["date"]
    M["date"] = (M["date"] + pd.offsets.MonthBegin(1))  # weights at t-1 roll to t
    M = M.merge(A[["bank","date","aum_lag"]], on=["bank","date"], how="right")
    M["contrib_chf_m"] = (M["aum_lag"].fillna(0.0) * M["weight"].fillna(0.0) * M["return"].fillna(0.0))
    perf_attr = (M.groupby(["bank","date","asset_class"], as_index=False)
                   .agg(perf_chf_m=("contrib_chf_m","sum")))
    perf_tbl = (perf_attr.groupby(["bank","date"], as_index=False)
                  .agg(perf_chf_m=("perf_chf_m","sum")))
    return perf_tbl, perf_attr

def fx_effect(aum: pd.DataFrame, ccy_exp: pd.DataFrame, fx: pd.DataFrame, hedge_ratio: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    FX effect using currency exposures at t-1 and FX returns at t.
    If hedge_ratio is None, we infer an 'effective' hedge ratio later; here we assume fully unhedged.
    """
    if aum.empty or ccy_exp.empty or fx.empty:
        return pd.DataFrame(), pd.DataFrame()
    A = aum.copy().sort_values(["bank","date"])
    A["aum_lag"] = A.groupby("bank")["aum_chf_m"].shift(1)
    E = ccy_exp.copy().sort_values(["bank","ccy","date"])
    E["weight"] = E.groupby(["bank","ccy"])["weight"].ffill()
    F = fx.copy()[["date","ccy","fx_ret"]]
    # t-1 exposures apply to t FX returns
    E["date_lag"] = E["date"]
    E["date"] = (E["date"] + pd.offsets.MonthBegin(1))
    M = E.merge(F, on=["date","ccy"], how="left").merge(A[["bank","date","aum_lag"]], on=["bank","date"], how="left")
    hr = 0.0 if hedge_ratio is None else float(hedge_ratio)
    M["contrib_chf_m"] = M["aum_lag"].fillna(0.0) * M["weight"].fillna(0.0) * (1.0 - hr) * M["fx_ret"].fillna(0.0)
    fx_attr = (M.groupby(["bank","date","ccy"], as_index=False)
                 .agg(fx_chf_m=("contrib_chf_m","sum")))
    fx_tbl = (fx_attr.groupby(["bank","date"], as_index=False)
                .agg(fx_chf_m=("fx_chf_m","sum")))
    return fx_tbl, fx_attr

def reconcile_nna(aum: pd.DataFrame, perf_tbl: pd.DataFrame, fx_tbl: pd.DataFrame) -> pd.DataFrame:
    A = aum.copy().sort_values(["bank","date"])
    A["aum_lag"] = A.groupby("bank")["aum_chf_m"].shift(1)
    A["delta_chf_m"] = A["aum_chf_m"] - A["aum_lag"]
    out = A.merge(perf_tbl, on=["bank","date"], how="left").merge(fx_tbl, on=["bank","date"], how="left")
    out["perf_chf_m"] = out["perf_chf_m"].fillna(0.0)
    out["fx_chf_m"]   = out["fx_chf_m"].fillna(0.0)
    out["nna_chf_m"]  = out["delta_chf_m"] - out["perf_chf_m"] - out["fx_chf_m"]
    out["organic_growth_pct"] = out["nna_chf_m"] / out["aum_lag"]
    return out

def infer_effective_hedge(bank_ts: pd.DataFrame, ccy_exp: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate hedge ratio h solving: observed_fx ≈ (1-h) * unhedged_fx_effect  →  h = 1 - obs/unhedged
    """
    if bank_ts.empty or ccy_exp.empty or fx.empty:
        return pd.DataFrame()
    # Build "unhedged" fx effect using exposures and FX returns (same as fx_effect with hr=0)
    tmp_fx_tbl, _ = fx_effect(bank_ts[["bank","date","aum_chf_m"]], ccy_exp, fx, hedge_ratio=0.0)
    if tmp_fx_tbl.empty:
        return pd.DataFrame()
    M = bank_ts.merge(tmp_fx_tbl.rename(columns={"fx_chf_m":"fx_unhedged_chf_m"}), on=["bank","date"], how="left")
    # Observed fx_chf_m may not yet exist; if missing we treat 0
    M["fx_unhedged_chf_m"] = M["fx_unhedged_chf_m"].fillna(0.0)
    M["fx_chf_m"] = M.get("fx_chf_m", pd.Series(0.0, index=M.index)).fillna(0.0)
    # h = 1 - obs/unhedged (clip to [0,1])
    M["hedge_ratio_eff"] = 1.0 - np.where(M["fx_unhedged_chf_m"].abs()>1e-9, M["fx_chf_m"]/M["fx_unhedged_chf_m"], np.nan)
    M["hedge_ratio_eff"] = M["hedge_ratio_eff"].clip(lower=0.0, upper=1.0)
    return M[["bank","date","hedge_ratio_eff"]]

def region_flows(nna_ts: pd.DataFrame, domicile: pd.DataFrame) -> pd.DataFrame:
    if nna_ts.empty or domicile.empty:
        return pd.DataFrame()
    # Use latest available region weights to split current-month NNA
    D = domicile.copy().sort_values(["bank","region","date"])
    D["weight"] = D.groupby(["bank","region"])["weight"].ffill()
    latest = D.groupby(["bank"])["date"].max().reset_index().rename(columns={"date":"latest"})
    Dlt = D.merge(latest, on=["bank"], how="inner")
    Dlt = Dlt[Dlt["date"]==Dlt["latest"]][["bank","region","weight"]]
    out = nna_ts.merge(Dlt, on="bank", how="left")
    out["nna_region_chf_m"] = out["nna_chf_m"] * out["weight"].fillna(0.0)
    return (out.groupby(["date","bank","region"], as_index=False)
              .agg(nna_chf_m=("nna_region_chf_m","sum")))

def channel_table(nna_ts: pd.DataFrame, channels: pd.DataFrame, aum: pd.DataFrame) -> pd.DataFrame:
    if channels.empty:
        return pd.DataFrame()
    C = channels.copy().sort_values(["bank","channel","date"])
    C["weight"] = C.groupby(["bank","channel"])["weight"].ffill()
    latest = C.groupby(["bank"])["date"].max().reset_index().rename(columns={"date":"latest"})
    Clt = C.merge(latest, on="bank", how="inner")
    Clt = Clt[Clt["date"]==Clt["latest"]][["bank","channel","weight"]]
    N = nna_ts.merge(Clt, on="bank", how="left")
    N["nna_channel_chf_m"] = N["nna_chf_m"] * N["weight"].fillna(0.0)
    # Also compute latest AUM by channel
    A = aum.sort_values(["bank","date"])
    Alast = A.groupby("bank").tail(1)[["bank","aum_chf_m"]]
    Alast = Alast.merge(Clt, on="bank", how="left")
    Alast["aum_channel_chf_m"] = Alast["aum_chf_m"] * Alast["weight"].fillna(0.0)
    aum_ch = Alast[["bank","channel","aum_channel_chf_m"]]
    nna_ch = (N.groupby(["date","bank","channel"], as_index=False)
                .agg(nna_chf_m=("nna_channel_chf_m","sum")))
    return nna_ch, aum_ch

def sensitivities(aum: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if aum.empty:
        return pd.DataFrame()
    W = last_known_weights(weights) if not weights.empty else pd.DataFrame(columns=["bank","date","asset_class","weight"])
    # roll to match AUM current month (we want sensitivity at t based on t-1 weights)
    if not W.empty:
        W["date"] = (W["date"] + pd.offsets.MonthBegin(1))
    A = aum.copy()
    # Equity beta: 1% equity move → AUM × equity_weight%
    # Rates beta: 100bp rise → AUM × FI_weight × (-6%) (stylized)
    rows = []
    for (b, d), g in A.groupby(["bank","date"]):
        a = float(g["aum_chf_m"].iloc[0])
        ww = W[(W["bank"]==b) & (W["date"]==d)]
        w_eq = float(ww[ww["asset_class"].str.lower().str.contains("equity")]["weight"].sum()) if not ww.empty else 0.0
        w_fi = float(ww[ww["asset_class"].str.lower().str.contains("fixed")]["weight"].sum()) if not ww.empty else 0.0
        beta_eq = a * w_eq * 0.01
        beta_rt = a * w_fi * (-0.06)  # per 100bp
        rows.append({"bank": b, "date": d, "aum_chf_m": a, "beta_equity_per_1pct": beta_eq, "beta_rates_per_100bp": beta_rt})
    return pd.DataFrame(rows).sort_values(["bank","date"])

def scenarios_impact(sens: pd.DataFrame, ccy_exp: pd.DataFrame, fx: pd.DataFrame, scen_df: pd.DataFrame, scenario: str, hedge_ratio_eff: Optional[pd.DataFrame]) -> pd.DataFrame:
    if sens.empty or scen_df.empty:
        return pd.DataFrame()
    sub = scen_df[scen_df["scenario"]==scenario]
    if sub.empty: return pd.DataFrame()
    keys = {str(k).lower(): float(v) if pd.notna(v) else np.nan for k, v in zip(sub["key"], sub["value"])}
    eq = keys.get("equity.shock_pct", 0.0)
    rt = keys.get("rates.shock_bp", 0.0)
    chf = keys.get("chf.shock_pct", 0.0)
    hedge_override = keys.get("hedge.override", np.nan)
    nna_beh_pp = keys.get("nna.behavior_pp", 0.0)  # basis points of AUM

    # FX piece: use exposure & inferred hedge ratio to turn CHF move into AUM effect
    fx_piece = []
    if not ccy_exp.empty and chf != 0:
        # Aggregate non-CHF exposure per bank at latest date in sens table
        latest_dates = sens.groupby("bank")["date"].max().reset_index().rename(columns={"date":"latest"})
        E = ccy_exp.copy()
        E = E.merge(latest_dates, on="bank", how="inner")
        E = E[E["date"]==E["latest"]]
        E["is_chf"] = (E["ccy"].str.upper()=="CHF")
        E_non = E[~E["is_chf"]].groupby("bank", as_index=False)["weight"].sum().rename(columns={"weight":"non_chf_w"})
        # Effective hedge ratio
        if hedge_ratio_eff is not None and not hedge_ratio_eff.empty:
            HR = hedge_ratio_eff.sort_values(["bank","date"]).groupby("bank").tail(1)[["bank","hedge_ratio_eff"]]
        else:
            HR = pd.DataFrame({"bank": E_non["bank"], "hedge_ratio_eff": np.nan})
        FX = E_non.merge(HR, on="bank", how="left")
        if not np.isnan(hedge_override):
            FX["hedge_ratio_eff"] = hedge_override
        FX["unhedged_w"] = FX["non_chf_w"].fillna(0.0) * (1.0 - FX["hedge_ratio_eff"].fillna(0.0))
        # chf.shock_pct is CHF appreciation vs basket; AUM effect ≈ -unhedged_w × chf%
        FX["fx_effect_perc"] = - FX["unhedged_w"] * (chf/100.0)
        fx_piece = FX[["bank","fx_effect_perc"]]

    rows = []
    for _, r in sens.iterrows():
        b, d, aum = r["bank"], r["date"], r["aum_chf_m"]
        dA_eq = r["beta_equity_per_1pct"] * eq
        dA_rt = r["beta_rates_per_100bp"] * (rt/100.0)
        # FX effect from scenario
        fx_perc = 0.0
        if fx_piece:
            m = pd.DataFrame(fx_piece)
            v = m[m["bank"]==b]["fx_effect_perc"]
            fx_perc = float(v.iloc[0]) if not v.empty else 0.0
        dA_fx = aum * fx_perc
        # Behavioural NNA (bps of AUM)
        dA_nna = aum * (nna_beh_pp / 10000.0)
        rows.append({"bank": b, "date": d, "scenario": scenario,
                     "dA_equity_chf_m": dA_eq, "dA_rates_chf_m": dA_rt, "dA_fx_chf_m": dA_fx, "dA_behavioral_chf_m": dA_nna,
                     "total_dA_chf_m": dA_eq + dA_rt + dA_fx + dA_nna})
    return pd.DataFrame(rows)

def blend_measured_vehicle_flows(nna_ts: pd.DataFrame, fund_flows: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Optional: convex blend of residual NNA and measured vehicle flows when both exist."""
    if nna_ts.empty or fund_flows.empty:
        return nna_ts
    F = (fund_flows.groupby(["bank","date"], as_index=False)
           .agg(fund_flow_chf_m=("flow_chf_m","sum")))
    M = nna_ts.merge(F, on=["bank","date"], how="left")
    mask = M["fund_flow_chf_m"].notna()
    M.loc[mask, "nna_chf_m"] = (1-alpha) * M.loc[mask, "nna_chf_m"] + alpha * M.loc[mask, "fund_flow_chf_m"]
    return M


# ----------------------------- CLI -----------------------------

@dataclass
class Config:
    aum: str
    weights: Optional[str]
    returns: Optional[str]
    fx: Optional[str]
    ccy_exposure: Optional[str]
    domicile: Optional[str]
    channels: Optional[str]
    fund_flows: Optional[str]
    scenarios: Optional[str]
    start: str
    end: str
    reconcile_fund_flows: bool
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Swiss wealth flows decomposition")
    ap.add_argument("--aum", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--returns", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--ccy_exposure", default="")
    ap.add_argument("--domicile", default="")
    ap.add_argument("--channels", default="")
    ap.add_argument("--fund_flows", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--start", default="2018-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--reconcile_fund_flows", action="store_true")
    ap.add_argument("--outdir", default="out_swiss_flows")
    return ap.parse_args()


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    AUM = load_aum(args.aum)
    W   = load_weights(args.weights) if args.weights else pd.DataFrame()
    R   = load_returns(args.returns) if args.returns else pd.DataFrame()
    FX  = load_fx(args.fx) if args.fx else pd.DataFrame()
    CEX = load_ccy_exp(args.ccy_exposure) if args.ccy_exposure else pd.DataFrame()
    DOM = load_domicile(args.domicile) if args.domicile else pd.DataFrame()
    CHN = load_channels(args.channels) if args.channels else pd.DataFrame()
    FFL = load_fund_flows(args.fund_flows) if args.fund_flows else pd.DataFrame()
    SCN = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Filter window
    start = pd.to_datetime(args.start).to_period("M").to_timestamp()
    end   = pd.to_datetime(args.end).to_period("M").to_timestamp()
    for df_name in ["AUM","W","R","FX","CEX","DOM","CHN","FFL"]:
        df = locals()[df_name]
        if not df.empty:
            df = df[(df["date"]>=start) & (df["date"]<=end)]
            locals()[df_name] = df

    # Performance effect
    perf_tbl, perf_attr = perf_effect(AUM, W, R) if (not AUM.empty and not W.empty and not R.empty) else (pd.DataFrame(), pd.DataFrame())
    # FX effect (initially unhedged if HR unknown)
    fx_tbl, fx_attr = fx_effect(AUM, CEX, FX) if (not AUM.empty and not CEX.empty and not FX.empty) else (pd.DataFrame(), pd.DataFrame())

    # Base reconciliation
    bank_ts = reconcile_nna(AUM, perf_tbl, fx_tbl)

    # Infer hedge ratio (optional)
    H = infer_effective_hedge(bank_ts, CEX, FX) if (not bank_ts.empty and not CEX.empty and not FX.empty) else pd.DataFrame()
    if not H.empty:
        bank_ts = bank_ts.merge(H, on=["bank","date"], how="left")

    # Optional: blend with measured fund flows
    if args.reconcile_fund_flows and not FFL.empty:
        bank_ts = blend_measured_vehicle_flows(bank_ts, FFL, alpha=0.5)

    # Region & channel splits
    reg_flows = region_flows(bank_ts, DOM) if not DOM.empty else pd.DataFrame()
    ch_flows, ch_aum = channel_table(bank_ts, CHN, AUM) if not CHN.empty else (pd.DataFrame(), pd.DataFrame())

    # Currency-level flows (if exposure provided)
    flows_ccy = pd.DataFrame()
    if not CEX.empty:
        # distribute NNA by currency exposure at latest weights
        latest = CEX.groupby("bank")["date"].max().reset_index().rename(columns={"date":"latest"})
        E = CEX.merge(latest, on="bank", how="inner"); E = E[E["date"]==E["latest"]][["bank","ccy","weight"]]
        flows_ccy = bank_ts.merge(E, on="bank", how="left")
        flows_ccy["nna_ccy_chf_m"] = flows_ccy["nna_chf_m"] * flows_ccy["weight"].fillna(0.0)
        flows_ccy = (flows_ccy.groupby(["date","bank","ccy"], as_index=False)
                       .agg(nna_chf_m=("nna_ccy_chf_m","sum")))

    # Sensitivities & scenarios
    sens = sensitivities(AUM, W)
    scen_out = pd.DataFrame()
    if not sens.empty and not SCN.empty:
        # Use most recent month sens per bank
        sens_latest = sens.sort_values(["bank","date"]).groupby("bank").tail(1)
        scen_name = SCN["scenario"].unique().tolist()
        scen_rows = []
        for s in scen_name:
            out = scenarios_impact(sens_latest, CEX, FX, SCN, s, H if not H.empty else None)
            scen_rows.append(out)
        scen_out = pd.concat(scen_rows, ignore_index=True) if scen_rows else pd.DataFrame()

    # Write outputs
    bank_ts_out = bank_ts[["bank","date","aum_chf_m","delta_chf_m","perf_chf_m","fx_chf_m","nna_chf_m","organic_growth_pct"] + (["hedge_ratio_eff"] if "hedge_ratio_eff" in bank_ts.columns else [])].copy()
    bank_ts_out.to_csv(outdir / "bank_timeseries.csv", index=False)
    if not perf_attr.empty: perf_attr.to_csv(outdir / "perf_attribution.csv", index=False)
    if not fx_attr.empty:   fx_attr.to_csv(outdir / "fx_attribution.csv", index=False)
    if not reg_flows.empty: reg_flows.to_csv(outdir / "flows_by_region.csv", index=False)
    if not flows_ccy.empty: flows_ccy.to_csv(outdir / "flows_by_currency.csv", index=False)
    if not ch_flows.empty:  ch_flows.to_csv(outdir / "channel_mix.csv", index=False)
    if not H.empty:         H.to_csv(outdir / "inferred_hedge.csv", index=False)
    if not sens.empty:      sens.to_csv(outdir / "sensitivities.csv", index=False)
    if not scen_out.empty:  scen_out.to_csv(outdir / "scenarios_out.csv", index=False)

    # Summary (TTM)
    ttm_cut = (bank_ts_out["date"].max() - pd.offsets.DateOffset(years=1)) if not bank_ts_out.empty else None
    summaries = []
    for b, g in bank_ts_out.groupby("bank"):
        last = g.sort_values("date").tail(1)
        ttm = g[g["date"] > (ttm_cut if ttm_cut is not None else g["date"].min())]
        k = {
            "bank": b,
            "latest_date": str(last["date"].iloc[0].date()),
            "latest_aum_chf_m": float(last["aum_chf_m"].iloc[0]),
            "ttm_nna_chf_m": float(ttm["nna_chf_m"].sum()) if not ttm.empty else float("nan"),
            "ttm_organic_growth_pct": float(ttm["nna_chf_m"].sum() / max(g["aum_chf_m"].shift(1).iloc[-12:].mean(), 1e-9)) if len(g) >= 13 else float("nan"),
            "ttm_perf_chf_m": float(ttm["perf_chf_m"].sum()) if "perf_chf_m" in g.columns else float("nan"),
            "ttm_fx_chf_m": float(ttm["fx_chf_m"].sum()) if "fx_chf_m" in g.columns else float("nan"),
            "hedge_ratio_eff_latest": (float(last["hedge_ratio_eff"].iloc[0]) if "hedge_ratio_eff" in last.columns and pd.notna(last["hedge_ratio_eff"].iloc[0]) else float("nan"))
        }
        summaries.append(k)
    (outdir / "summary.json").write_text(json.dumps({"banks": summaries}, indent=2))

    # Config
    cfg = asdict(Config(
        aum=args.aum, weights=(args.weights or None), returns=(args.returns or None), fx=(args.fx or None),
        ccy_exposure=(args.ccy_exposure or None), domicile=(args.domicile or None), channels=(args.channels or None),
        fund_flows=(args.fund_flows or None), scenarios=(args.scenarios or None),
        start=args.start, end=args.end, reconcile_fund_flows=bool(args.reconcile_fund_flows), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Swiss Wealth Flows Decomposition ==")
    if not bank_ts_out.empty:
        for b, g in bank_ts_out.groupby("bank"):
            last = g.sort_values("date").tail(1)
            org = last["organic_growth_pct"].iloc[0]*100 if pd.notna(last["organic_growth_pct"].iloc[0]) else float("nan")
            print(f"{b}: AUM {last['aum_chf_m'].iloc[0]:,.0f} m | Δ {last['delta_chf_m'].iloc[0]:+,.0f} | Perf {last['perf_chf_m'].iloc[0] if 'perf_chf_m' in last.columns else 0:+,.0f} | FX {last['fx_chf_m'].iloc[0] if 'fx_chf_m' in last.columns else 0:+,.0f} | NNA {last['nna_chf_m'].iloc[0]:+,.0f} ({org:.2f}% org)")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
