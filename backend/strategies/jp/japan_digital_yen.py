#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
japan_digital_yen.py — Digital Yen (CBDC) adoption, banking impact, events & scenarios
--------------------------------------------------------------------------------------

What this does
==============
A research toolkit to analyze potential **Digital Yen (retail CBDC)** effects on:
- Adoption path (Bass/logistic) & usage
- Payment mix substitution (cash/card/QR/bank transfer ↔ CBDC)
- Deposit displacement & **bank funding/NIM impact** (with tiered remuneration)
- Liquidity/run stress (CBDC-induced deposit flight; simple LCR survival)
- Market reaction **event studies** (JPY rates, USDJPY, bank equity index)
- Scenario projections and Monte Carlo stress on deposits/NI

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--policy policy.csv                  REQUIRED (timeline of CBDC/policy events)
  Columns: date, type[, label], value
    # examples of type: PILOT_START, PILOT_EXPANSION, WALLET_CAP_CHANGE, TIER_RATE_CHANGE, INTEROP_QR, CROSS_BORDER

--payments payments.csv              OPTIONAL (monthly)
  Columns (any subset):
    date, cash_txn_value, card_txn_value, bank_txn_value, qr_txn_value, cbdc_txn_value
    active_wallets[, users], tx_per_wallet[, avg_tx_per_wallet], merchants_accepting

--banks banks.csv                    OPTIONAL (monthly or quarterly)
  Columns:
    date, deposits_household, deposits_corp, deposit_rate, loans, assets_earning, hqla, lcr_outflow_rate
    wholesale_rate[, bond_rate], bank_equity_index[, bank_index]

--yields yields.csv                  OPTIONAL (daily or monthly; daily preferred)
  Columns:
    date, jgb_2y_yield, jgb_10y_yield[, ois_rate]

--fx fx.csv                          OPTIONAL (daily or monthly)
  Columns:
    date, USDJPY[, JPYUSD]  # if JPYUSD is given, it will be inverted

--macro macro.csv                    OPTIONAL (monthly)
  Columns:
    date, cpi[, headline_cpi], pop[, population]

CLI (key)
---------
--adopt_model bass|logistic          Adoption curve type (fit if data, else priors)
--horizon 36                         Projection horizon in months
--cap_per_user 100000                CBDC holding cap per user (JPY)
--tier1_limit 30000                  Tier-1 remuneration cap (JPY)
--tier1_rate 0.00                    Tier-1 annual rate (e.g., 0.00)
--tier2_rate -0.005                  Tier-2 annual rate (e.g., -0.5% to discourage hoarding)
--hold_sensitivity 8.0               Logistic sensitivity of holdings to (r_cbdc - r_deposit)
--target_users 70_000_000            Market size M for adoption if not inferable
--wallet_util 0.60                   Share of users active (if not given); also usage intensity proxy
--subs_elastic -0.25                 Payment substitution elasticity of non-CBDC share wrt CBDC share
--run_shock_users 0.05               One-month run shock as % of target users shifting deposits to CBDC
--wholesale_spread_bps 80            Funding spread (wholesale - deposits) in bps when replacing deposits
--nim_assets_share 0.85              Share of assets earning interest for NIM approximation
--events_window 5                    Event study ±days window for daily series
--n_sims 10000                       Monte Carlo paths for deposit/NII stress
--outdir out_digital_yen             Output directory
--start / --end                      Optional sample filters (YYYY-MM-DD)

Outputs
-------
panel_monthly.csv            Cleaned, aligned monthly panel
adoption_fit.json            Estimated adoption parameters
adoption_scenarios.csv       Projected users/usage & CBDC holdings (base/fast/slow)
payments_mix.csv             Historical & projected payment shares with CBDC substitution
bank_impact.csv              Deposit displacement, funding gap, NIM delta by month
lcr_stress.csv               Simple survival days under run shock & outflow rates
event_study.csv              CARs around policy events (JPY rates, USDJPY, bank index)
mc_stress.csv                Monte Carlo distribution for deposit loss & NII change
summary.json                 Headline metrics
config.json                  Echo of run configuration

Notes
-----
• This is **research tooling**, not advice. Parameters are deliberately explicit so you can
  override priors that won’t match your local assumptions.
• Annual rates are converted to monthly where appropriate (rate/12 approx).
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

def winsor(x: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

def ann_to_month(r: float) -> float:
    return r / 12.0

def pct_to_bps(x: float) -> float:
    return 10000.0 * x

def bps_to_pct(bps: float) -> float:
    return bps / 10000.0


# ----------------------------- loaders -----------------------------

def load_policy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"type"); v = ncol(df,"value")
    lab = ncol(df,"label","event","note")
    if not (d and t): raise ValueError("policy.csv needs at least date and type.")
    df = df.rename(columns={d:"date", t:"type"})
    if v: df = df.rename(columns={v:"value"})
    if lab: df = df.rename(columns={lab:"label"})
    df["date"] = to_dt(df["date"])
    if "label" not in df.columns: df["label"] = df["type"]
    df = df.sort_values("date")
    return df

def load_payments(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); 
    if not d: raise ValueError("payments.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    ren = {
        ncol(df,"cash_txn_value","cash_value"): "cash",
        ncol(df,"card_txn_value","card_value"): "card",
        ncol(df,"bank_txn_value","bank_transfer_value","bank_value"): "bank",
        ncol(df,"qr_txn_value","qr_value"): "qr",
        ncol(df,"cbdc_txn_value","cbdc_value"): "cbdc",
        ncol(df,"active_wallets","users","wallets"): "active_wallets",
        ncol(df,"tx_per_wallet","avg_tx_per_wallet"): "tx_per_wallet",
        ncol(df,"merchants_accepting","merchant_acceptance"): "merchants_accepting",
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for k in ["cash","card","bank","qr","cbdc","active_wallets","tx_per_wallet","merchants_accepting"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_banks(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); 
    if not d: raise ValueError("banks.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    ren = {
        ncol(df,"deposits_household","hh_deposits"): "deposits_household",
        ncol(df,"deposits_corp","corp_deposits"): "deposits_corp",
        ncol(df,"deposit_rate","dep_rate"): "deposit_rate",
        ncol(df,"loans"): "loans",
        ncol(df,"assets_earning","iea","interest_earning_assets"): "assets_earning",
        ncol(df,"hqla"): "hqla",
        ncol(df,"lcr_outflow_rate","net_outflow_rate"): "lcr_outflow_rate",
        ncol(df,"wholesale_rate","bond_rate"): "wholesale_rate",
        ncol(df,"bank_equity_index","bank_index"): "bank_index",
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for k in ["deposits_household","deposits_corp","deposit_rate","loans","assets_earning",
              "hqla","lcr_outflow_rate","wholesale_rate","bank_index"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df

def load_yields(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); y2 = ncol(df,"jgb_2y_yield","jgb2y"); y10 = ncol(df,"jgb_10y_yield","jgb10y")
    ois = ncol(df,"ois_rate","ycc_target")
    if not (d and (y2 or y10)): raise ValueError("yields.csv needs date and at least one of jgb_2y_yield/jgb_10y_yield.")
    df = df.rename(columns={d:"date"})
    if y2: df = df.rename(columns={y2:"jgb_2y"})
    if y10: df = df.rename(columns={y10:"jgb_10y"})
    if ois: df = df.rename(columns={ois:"ois"})
    df["date"] = to_dt(df["date"])
    for k in ["jgb_2y","jgb_10y","ois"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df.sort_values("date")

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); u = ncol(df,"USDJPY","usdjpy"); j = ncol(df,"JPYUSD","jpyusd")
    if not d: raise ValueError("fx.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = to_dt(df["date"])
    if u:
        df["USDJPY"] = safe_num(df[u])
    elif j:
        df["USDJPY"] = 1.0 / safe_num(df[j])
    else:
        num = [c for c in df.columns if c != "date"]
        if not num: raise ValueError("fx.csv must include USDJPY or JPYUSD.")
        df["USDJPY"] = safe_num(df[num[0]])
    df["r_fx"] = dlog(df["USDJPY"])
    return df.sort_values("date")

def load_macro(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); cpi = ncol(df,"cpi","headline_cpi"); pop = ncol(df,"pop","population")
    if not d: raise ValueError("macro.csv needs date.")
    df = df.rename(columns={d:"date"})
    df["date"] = eom(df["date"])
    if cpi: df = df.rename(columns={cpi:"cpi"})
    if pop: df = df.rename(columns={pop:"population"})
    for k in ["cpi","population"]:
        if k in df.columns: df[k] = safe_num(df[k])
    return df


# ----------------------------- adoption models -----------------------------

def fit_bass(active_users: pd.Series, market_size: int) -> Dict[str, float]:
    """
    Fit Bass model by coarse grid search (no scipy dependency).
    active_users: cumulative or active users proxy (will cumulate if not monotonic)
    Returns dict with p, q, M (market size).
    """
    s = active_users.dropna().astype(float).copy()
    # Force cumulative (monotone)
    s_cum = s.cummax()
    T = len(s_cum)
    M = float(max(market_size, s_cum.max()))
    # grid
    p_grid = np.linspace(0.001, 0.05, 25)
    q_grid = np.linspace(0.05, 0.9, 30)
    best = (1e99, 0.01, 0.3)
    for p in p_grid:
        for q in q_grid:
            # simulate
            N = np.zeros(T)
            for t in range(T):
                prev = N[t-1] if t>0 else 0.0
                ad = (p + q * (prev/M)) * (M - prev)  # monthly flow
                N[t] = prev + ad
            sse = float(np.mean((N - s_cum.values)**2))
            if sse < best[0]:
                best = (sse, p, q)
    return {"p": float(best[1]), "q": float(best[2]), "M": float(M)}

def project_bass(params: Dict[str,float], T: int, start_level: float=0.0, shock_mult: float=1.0) -> pd.DataFrame:
    p = params.get("p", 0.01) * shock_mult
    q = params.get("q", 0.3) * shock_mult
    M = params.get("M", 50_000_000.0)
    N = np.zeros(T)
    N0 = min(start_level, M)
    for t in range(T):
        prev = N[t-1] if t>0 else N0
        ad = (p + q * (prev/M)) * (M - prev)
        N[t] = prev + ad
    flows = np.diff(np.r_[N0, N])
    return pd.DataFrame({"h_month": np.arange(1, T+1), "users_cum": N, "new_users": flows})

def fit_logistic(active_users: pd.Series, market_size: int) -> Dict[str, float]:
    """
    Simple logistic fit on cumulative users: U_t = K / (1 + a * exp(-b t))
    Fit by grid-search over b and a; set K=market_size (or max observed).
    """
    s = active_users.dropna().astype(float).cummax().reset_index(drop=True)
    K = float(max(market_size, s.max()))
    T = len(s)
    b_grid = np.linspace(0.02, 0.6, 40)
    a_grid = np.linspace(0.5, 50.0, 40)
    best = (1e99, 0.2, 10.0)
    for b in b_grid:
        for a in a_grid:
            t = np.arange(1, T+1)
            U = K / (1.0 + a * np.exp(-b * t))
            sse = float(np.mean((U - s.values)**2))
            if sse < best[0]:
                best = (sse, b, a)
    return {"K": K, "b": float(best[1]), "a": float(best[2])}

def project_logistic(params: Dict[str,float], T: int, start_level: float=0.0, shock_mult: float=1.0) -> pd.DataFrame:
    K = params.get("K", 50_000_000.0)
    b = params.get("b", 0.2) * shock_mult
    a = params.get("a", 10.0)
    t = np.arange(1, T+1)
    U = K / (1.0 + a * np.exp(-b * t))
    # start at start_level by shifting if needed
    if start_level > 0:
        # crude alignment: force first point to start_level by scaling 'a'
        a_adj = (K - start_level) / start_level * np.exp(-b*1)
        U = K / (1.0 + a_adj * np.exp(-b * t))
    new = np.diff(np.r_[start_level, U])
    return pd.DataFrame({"h_month": t, "users_cum": U, "new_users": new})


# ----------------------------- payments & substitution -----------------------------

def build_payments_panel(PAY: pd.DataFrame) -> pd.DataFrame:
    if PAY.empty: return pd.DataFrame()
    df = PAY.copy().sort_values("date")
    cols = [c for c in ["cash","card","bank","qr","cbdc"] if c in df.columns]
    if not cols: return df
    df["total"] = df[cols].sum(axis=1, min_count=1)
    for c in cols:
        df[c+"_share"] = df[c] / df["total"].replace(0, np.nan)
    return df

def apply_cbdc_substitution(base_share_cbdc: float, subs_elastic: float) -> Dict[str,float]:
    """
    Given CBDC share s, distribute the reduction across {cash, card, bank, qr}
    proportionally to their current shares times elasticity magnitude.
    Here subs_elastic < 0: each non-CBDC share is scaled by (1 + subs_elastic * s)
    and renormalized.
    """
    s = max(0.0, min(0.95, base_share_cbdc))
    elastic = subs_elastic
    others = {"cash":0.35, "card":0.40, "bank":0.20, "qr":0.05}  # fallback weights
    scaled = {k: max(0.0, v * (1.0 + elastic * s)) for k,v in others.items()}
    total_others = sum(scaled.values())
    if total_others <= 1e-12:
        scaled = others; total_others = sum(scaled.values())
    # Renormalize remaining (1-s)
    rem = 1.0 - s
    out = {k: rem * (v/total_others) for k,v in scaled.items()}
    out["cbdc"] = s
    return out


# ----------------------------- banking impact -----------------------------

def cbdc_holdings_per_user(cap_per_user: float, r_cbdc_tier1: float, r_cbdc_tier2: float,
                           deposit_rate: float, tier1_limit: float, hold_sensitivity: float) -> float:
    """
    Simple behavioral rule:
      • Tier-1 bucket (up to tier1_limit) holding probability sigmoid on (r_cbdc1 - r_dep)
      • Tier-2 bucket (cap - tier1_limit) holds with lower probability sigmoid on (r_cbdc2 - r_dep)
    Returns expected per-user holdings (JPY).
    """
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-hold_sensitivity * x))
    cap1 = min(cap_per_user, tier1_limit)
    cap2 = max(0.0, cap_per_user - tier1_limit)
    p1 = sigmoid(r_cbdc_tier1 - deposit_rate)
    p2 = sigmoid(r_cbdc_tier2 - deposit_rate)
    return float(cap1 * p1 + cap2 * p2)

def deposit_displacement(users: float, per_user_hold: float, src_share_from_deposits: float=0.85) -> float:
    """
    Not all CBDC holdings displace bank deposits: some come from cash or card prefunding.
    src_share_from_deposits ~ share sourced from deposits (rest from cash/other).
    """
    return float(users * per_user_hold * src_share_from_deposits)

def funding_gap(displaced: float, dep_rate: float, wholesale_rate: float, assets_earning: float,
                nim_assets_share: float) -> Tuple[float, float]:
    """
    Funding gap is amount to replace with wholesale funding at wholesale_rate instead of dep_rate.
    Returns (Δfunding_cost, ΔNIM_pct_points) relative to earning assets.
    """
    delta_cost = displaced * max(0.0, wholesale_rate - dep_rate)
    base_assets = assets_earning * nim_assets_share if (assets_earning is not None and np.isfinite(assets_earning)) else displaced
    nim_delta = delta_cost / (base_assets + 1e-12)
    return float(delta_cost), float(nim_delta)

def lcr_survival_days(hqla: float, daily_outflow: float) -> float:
    if daily_outflow <= 0: return np.inf
    return float(hqla / daily_outflow)


# ----------------------------- event study -----------------------------

def daily_to_event_df(YLD: pd.DataFrame, FX: pd.DataFrame, BANK: pd.DataFrame, POL: pd.DataFrame,
                      window_days: int=5) -> pd.DataFrame:
    if POL.empty: return pd.DataFrame()
    rows = []
    # Prep series (daily)
    s10 = YLD.set_index("date")["jgb_10y"] if ("jgb_10y" in YLD.columns) else pd.Series(dtype=float)
    s2  = YLD.set_index("date")["jgb_2y"]  if ("jgb_2y"  in YLD.columns) else pd.Series(dtype=float)
    sj  = FX.set_index("date")["r_fx"]     if (not FX.empty and "r_fx" in FX.columns) else pd.Series(dtype=float)
    sb  = BANK.set_index("date")["bank_index"] if ("bank_index" in BANK.columns and "date" in BANK.columns) else pd.Series(dtype=float)

    for _, ev in POL.iterrows():
        dt = pd.Timestamp(ev["date"])
        # windows
        for name, ser in [("JGB10Y", s10), ("JGB2Y", s2), ("USDJPY_r", sj), ("BANK_INDEX", sb)]:
            if ser.empty or dt not in ser.index: 
                # find nearest trading day
                idx = ser.index
                if len(idx)==0: continue
                pos = idx.searchsorted(dt)
                if pos>=len(idx): pos = len(idx)-1
                dt0 = idx[pos]
            else:
                dt0 = dt
            # CAR: sum over ±window for returns; for yields use Δbp sum
            if ser.empty: continue
            i = ser.index.get_loc(dt0)
            L = max(0, i - window_days)
            R = min(len(ser)-1, i + window_days)
            car = float(ser.iloc[L:R+1].sum())
            rows.append({"event_date": str(pd.Timestamp(dt0).date()), "type": ev["type"], "series": name, "CAR": car})
    return pd.DataFrame(rows)


# ----------------------------- scenarios & Monte Carlo -----------------------------

def make_adoption_paths(PAY: pd.DataFrame, model: str, horizon: int, target_users: int) -> Tuple[Dict[str,float], pd.DataFrame]:
    # Use active_wallets if available; else synthetic seed (very low base)
    seed = PAY["active_wallets"].dropna() if (not PAY.empty and "active_wallets" in PAY.columns) else pd.Series(dtype=float)
    start_level = float(seed.iloc[-1]) if not seed.empty else 500_000.0
    if model.lower().startswith("bass"):
        params = fit_bass(seed if not seed.empty else pd.Series([start_level]), market_size=target_users)
        base = project_bass(params, horizon, start_level=start_level, shock_mult=1.0)
        fast = project_bass(params, horizon, start_level=start_level, shock_mult=1.25)
        slow = project_bass(params, horizon, start_level=start_level, shock_mult=0.75)
        fit = {"model":"bass", **params}
    else:
        params = fit_logistic(seed if not seed.empty else pd.Series([start_level]), market_size=target_users)
        base = project_logistic(params, horizon, start_level=start_level, shock_mult=1.0)
        fast = project_logistic(params, horizon, start_level=start_level, shock_mult=1.25)
        slow = project_logistic(params, horizon, start_level=start_level, shock_mult=0.75)
        fit = {"model":"logistic", **params}
    A = base.assign(path="base").append(fast.assign(path="fast"), ignore_index=True).append(slow.assign(path="slow"), ignore_index=True)
    return fit, A

def build_bank_impact(A: pd.DataFrame, BANK_M: pd.DataFrame, params: dict,
                      dep_src_share: float=0.85, nim_assets_share: float=0.85) -> pd.DataFrame:
    """
    Combine adoption with banking monthly panel to compute holdings, displacement, NIM deltas.
    """
    if A.empty: return pd.DataFrame()
    out = []
    dep_rate_m = ann_to_month(params["deposit_rate"])
    wh_rate_m  = ann_to_month(params["wholesale_rate"])
    for _, r in A.iterrows():
        users = float(r["users_cum"])
        per_user = cbdc_holdings_per_user(
            cap_per_user=params["cap_per_user"],
            r_cbdc_tier1=ann_to_month(params["tier1_rate"]),
            r_cbdc_tier2=ann_to_month(params["tier2_rate"]),
            deposit_rate=dep_rate_m,
            tier1_limit=params["tier1_limit"],
            hold_sensitivity=float(params["hold_sensitivity"])
        )
        cbdc_holdings = users * per_user
        displaced = deposit_displacement(users, per_user, src_share_from_deposits=dep_src_share)
        assets_earning = float(BANK_M.get("assets_earning", np.nan))
        delta_cost, nim_delta = funding_gap(displaced, dep_rate_m, wh_rate_m, assets_earning, nim_assets_share)
        out.append({
            "h_month": int(r["h_month"]), "path": r["path"], "users_cum": users,
            "per_user_hold_jpy": per_user, "cbdc_holdings_jpy": cbdc_holdings,
            "displaced_deposits_jpy": displaced,
            "delta_funding_cost_monthly_jpy": delta_cost,
            "nim_delta_monthly": nim_delta
        })
    return pd.DataFrame(out)

def run_mc_stress(A_base: pd.DataFrame, params: dict, n_sims: int=10000) -> pd.DataFrame:
    """
    Monte Carlo on behavioral parameters: hold_sensitivity, src_share_from_deposits, wholesale spread.
    Outputs distribution of displaced deposits and funding cost (month h=1).
    """
    if A_base.empty: return pd.DataFrame()
    row = A_base[A_base["h_month"]==1].iloc[0]
    users = float(row["users_cum"])
    rng = np.random.default_rng(42)
    # Distributions (simple):
    sens = rng.normal(loc=params["hold_sensitivity"], scale=1.5, size=n_sims).clip(1.0, 15.0)
    src_share = rng.normal(loc=0.85, scale=0.08, size=n_sims).clip(0.5, 0.98)
    spread_bps = rng.normal(loc=params["wholesale_spread_bps"], scale=25.0, size=n_sims).clip(20, 200)
    dep_rate_m = ann_to_month(params["deposit_rate"])
    base_wh = dep_rate_m + bps_to_pct(params["wholesale_spread_bps"])/12.0
    # simulate
    displaced = np.zeros(n_sims); cost = np.zeros(n_sims)
    for i in range(n_sims):
        per_user = cbdc_holdings_per_user(params["cap_per_user"],
                                          ann_to_month(params["tier1_rate"]),
                                          ann_to_month(params["tier2_rate"]),
                                          dep_rate_m,
                                          params["tier1_limit"],
                                          sens[i])
        d = deposit_displacement(users, per_user, src_share[i])
        displaced[i] = d
        wh = dep_rate_m + bps_to_pct(spread_bps[i])/12.0
        cost[i] = d * max(0.0, wh - dep_rate_m)
    return pd.DataFrame({
        "displaced_deposits_jpy": displaced,
        "delta_funding_cost_monthly_jpy": cost
    })


# ----------------------------- CLI / orchestration -----------------------------

@dataclass
class Config:
    policy: str
    payments: Optional[str]
    banks: Optional[str]
    yields: Optional[str]
    fx: Optional[str]
    macro: Optional[str]
    adopt_model: str
    horizon: int
    cap_per_user: float
    tier1_limit: float
    tier1_rate: float
    tier2_rate: float
    hold_sensitivity: float
    target_users: int
    wallet_util: float
    subs_elastic: float
    run_shock_users: float
    wholesale_spread_bps: float
    nim_assets_share: float
    events_window: int
    n_sims: int
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Digital Yen (CBDC) — adoption, banking impact, events & scenarios")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--payments", default="")
    ap.add_argument("--banks", default="")
    ap.add_argument("--yields", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--adopt_model", default="bass", choices=["bass","logistic"])
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--cap_per_user", type=float, default=100_000.0)
    ap.add_argument("--tier1_limit", type=float, default=30_000.0)
    ap.add_argument("--tier1_rate", type=float, default=0.00)
    ap.add_argument("--tier2_rate", type=float, default=-0.005)
    ap.add_argument("--hold_sensitivity", type=float, default=8.0)
    ap.add_argument("--target_users", type=int, default=70_000_000)
    ap.add_argument("--wallet_util", type=float, default=0.60)
    ap.add_argument("--subs_elastic", type=float, default=-0.25)
    ap.add_argument("--run_shock_users", type=float, default=0.05)
    ap.add_argument("--wholesale_spread_bps", type=float, default=80.0)
    ap.add_argument("--nim_assets_share", type=float, default=0.85)
    ap.add_argument("--events_window", type=int, default=5)
    ap.add_argument("--n_sims", type=int, default=10000)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_digital_yen")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    POL = load_policy(args.policy)
    PAY = load_payments(args.payments) if args.payments else pd.DataFrame()
    BNK = load_banks(args.banks) if args.banks else pd.DataFrame()
    YLD = load_yields(args.yields) if args.yields else pd.DataFrame()
    FX  = load_fx(args.fx) if args.fx else pd.DataFrame()
    MAC = load_macro(args.macro) if args.macro else pd.DataFrame()

    # Time filters
    if args.start:
        s = to_dt(pd.Series([args.start])).iloc[0]
        for df in [POL, PAY, BNK, YLD, FX, MAC]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] < (eom(pd.Series([s])).iloc[0] if df is not YLD and df is not FX else s)].index, inplace=True)
    if args.end:
        e = to_dt(pd.Series([args.end])).iloc[0]
        for df in [POL, PAY, BNK, YLD, FX, MAC]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] > (eom(pd.Series([e])).iloc[0] if df is not YLD and df is not FX else e)].index, inplace=True)

    # Build monthly panel
    PM = build_payments_panel(PAY)
    if not PM.empty:
        PM.to_csv(outdir / "panel_monthly.csv", index=False)

    # Adoption fit & scenarios
    fit_params, A = make_adoption_paths(PAY, args.adopt_model, horizon=int(args.horizon), target_users=int(args.target_users))
    # Active usage: if tx_per_wallet provided, project constant; else use wallet_util to get activity proxy
    if "tx_per_wallet" in PAY.columns and PAY["tx_per_wallet"].notna().any():
        base_txpw = float(PAY["tx_per_wallet"].dropna().iloc[-1])
    else:
        base_txpw = 8.0 * float(args.wallet_util)  # simple proxy
    A["active_wallets"] = A["users_cum"] * float(args.wallet_util)
    A["tx_per_wallet"] = base_txpw
    (outdir / "adoption_fit.json").write_text(json.dumps(fit_params, indent=2))
    A.to_csv(outdir / "adoption_scenarios.csv", index=False)

    # Payment mix (historical + illustrative projection using substitution rule)
    MIX = pd.DataFrame()
    if not PM.empty:
        last_shares = {c.replace("_share",""): PM[c].iloc[-1] for c in PM.columns if c.endswith("_share")}
    else:
        # fallback shares
        last_shares = {"cash":0.25,"card":0.40,"bank":0.25,"qr":0.10,"cbdc":0.00}
    proj = []
    for _, r in A[A["path"]=="base"].iterrows():
        # CBDC share proxy: bound by active wallets intensity vs total payments
        s_cbdc = min(0.90, (r["active_wallets"] / args.target_users) * 0.5)  # heuristic cap
        shares = apply_cbdc_substitution(s_cbdc, subs_elastic=float(args.subs_elastic))
        proj.append({"h_month": int(r["h_month"]), **shares})
    MIX = pd.DataFrame(proj)
    MIX.to_csv(outdir / "payments_mix.csv", index=False)

    # Banking impact on NIM (use last known bank panel as base)
    if not BNK.empty:
        B_last = BNK.sort_values("date").tail(1).iloc[0]
        params_bank = {
            "cap_per_user": float(args.cap_per_user),
            "tier1_limit": float(args.tier1_limit),
            "tier1_rate": float(args.tier1_rate),
            "tier2_rate": float(args.tier2_rate),
            "hold_sensitivity": float(args.hold_sensitivity),
            "deposit_rate": float(B_last.get("deposit_rate", 0.001)),
            "wholesale_rate": float(B_last.get("deposit_rate", 0.001) + bps_to_pct(args.wholesale_spread_bps)),
            "wholesale_spread_bps": float(args.wholesale_spread_bps)
        }
        BI = build_bank_impact(A[A["path"]=="base"], B_last, params_bank, dep_src_share=0.85,
                               nim_assets_share=float(args.nim_assets_share))
        BI.to_csv(outdir / "bank_impact.csv", index=False)

        # LCR run stress (single shock at h=1)
        run_users = float(args.run_shock_users) * float(args.target_users)
        per_user = cbdc_holdings_per_user(
            cap_per_user=float(args.cap_per_user),
            r_cbdc_tier1=ann_to_month(float(args.tier1_rate)),
            r_cbdc_tier2=ann_to_month(float(args.tier2_rate)),
            deposit_rate=ann_to_month(float(B_last.get("deposit_rate", 0.001))),
            tier1_limit=float(args.tier1_limit),
            hold_sensitivity=float(args.hold_sensitivity)
        )
        run_outflow = deposit_displacement(run_users, per_user, src_share_from_deposits=0.95)  # runs mostly from deposits
        daily_out = run_outflow / 30.0
        surv = lcr_survival_days(float(B_last.get("hqla", np.nan)), daily_out)
        pd.DataFrame([{
            "run_users": run_users,
            "per_user_hold_jpy": per_user,
            "outflow_1m_jpy": run_outflow,
            "daily_outflow_jpy": daily_out,
            "hqla_jpy": float(B_last.get("hqla", np.nan)),
            "survival_days": surv
        }]).to_csv(outdir / "lcr_stress.csv", index=False)

        # Monte Carlo stress (month 1)
        MC = run_mc_stress(A[A["path"]=="base"], params_bank, n_sims=int(args.n_sims))
        if not MC.empty:
            MC.to_csv(outdir / "mc_stress.csv", index=False)

    # Event study on daily series around policy events
    ES = pd.DataFrame()
    if not POL.empty and (not YLD.empty or not FX.empty or ("bank_index" in BNK.columns if not BNK.empty else False)):
        # Need daily bank index for event windows; if not daily, this part may be sparse
        BANK_D = BNK.copy()
        if not BANK_D.empty and "date" in BANK_D.columns:
            BANK_D["date"] = to_dt(BANK_D["date"])
        ES = daily_to_event_df(YLD if not YLD.empty else pd.DataFrame(),
                               FX if not FX.empty else pd.DataFrame(),
                               BANK_D if not BANK_D.empty else pd.DataFrame(),
                               POL, window_days=int(args.events_window))
        if not ES.empty:
            ES.to_csv(outdir / "event_study.csv", index=False)

    # Summary
    sample_start = None
    sample_end = None
    for df in [PAY, BNK, YLD, FX, MAC]:
        if not df.empty and "date" in df.columns:
            s = df["date"].min(); e = df["date"].max()
            sample_start = s if sample_start is None else min(sample_start, s)
            sample_end = e if sample_end is None else max(sample_end, e)

    summary = {
        "sample": {
            "start": str(sample_start.date()) if sample_start is not None else None,
            "end": str(sample_end.date()) if sample_end is not None else None
        },
        "adoption_fit": fit_params,
        "adoption_latest": {
            "users_cum": float(A[A["path"]=="base"]["users_cum"].iloc[-1]),
            "active_wallets": float(A[A["path"]=="base"]["active_wallets"].iloc[-1])
        } if not A.empty else None,
        "banking": {
            "has_banks": bool(not BNK.empty),
            "outputs": ["bank_impact.csv","lcr_stress.csv","mc_stress.csv"] if not BNK.empty else []
        },
        "events": {
            "n_events": int(ES.shape[0]) if not ES.empty else 0
        },
        "files": {
            "panel_monthly": "panel_monthly.csv" if not PM.empty else None,
            "adoption_fit": "adoption_fit.json",
            "adoption_scenarios": "adoption_scenarios.csv",
            "payments_mix": "payments_mix.csv",
            "bank_impact": "bank_impact.csv" if (not BNK.empty and 'BI' in locals()) else None,
            "lcr_stress": "lcr_stress.csv" if (not BNK.empty) else None,
            "mc_stress": "mc_stress.csv" if (not BNK.empty and not locals().get('MC', pd.DataFrame()).empty) else None,
            "event_study": "event_study.csv" if not ES.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        policy=args.policy, payments=(args.payments or None), banks=(args.banks or None),
        yields=(args.yields or None), fx=(args.fx or None), macro=(args.macro or None),
        adopt_model=args.adopt_model, horizon=int(args.horizon), cap_per_user=float(args.cap_per_user),
        tier1_limit=float(args.tier1_limit), tier1_rate=float(args.tier1_rate), tier2_rate=float(args.tier2_rate),
        hold_sensitivity=float(args.hold_sensitivity), target_users=int(args.target_users),
        wallet_util=float(args.wallet_util), subs_elastic=float(args.subs_elastic),
        run_shock_users=float(args.run_shock_users), wholesale_spread_bps=float(args.wholesale_spread_bps),
        nim_assets_share=float(args.nim_assets_share), events_window=int(args.events_window),
        n_sims=int(args.n_sims), start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Digital Yen (CBDC) Toolkit ==")
    print(f"Sample: {summary['sample']['start']} → {summary['sample']['end']}")
    print(f"Adoption model: {fit_params.get('model')} | Params: { {k:v for k,v in fit_params.items() if k!='model'} }")
    if not BNK.empty:
        print("Banking impact, LCR stress, and Monte Carlo outputs written.")
    if not ES.empty:
        print(f"Event study completed on {ES['event_date'].nunique()} event days.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
