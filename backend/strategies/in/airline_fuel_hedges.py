#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
airline_fuel_hedges.py — Jet fuel hedge ratio engine, contract sizing & backtest
---------------------------------------------------------------------------------

What this does
==============
For an airline with jet-fuel exposure, this script:
1) Cleans & aligns price data for Jet Fuel and candidate hedge instruments (e.g., Brent, WTI, ULSD/HO, Gasoil).
2) Estimates single- or multi-instrument *minimum-variance hedge ratios* via rolling OLS:
      R_JF ≈ α + β' R_H    →  β_t computed on a rolling lookback.
3) Converts hedge notional to *integer contracts* using instrument multipliers (bbl/contract), with costs.
4) Backtests a simple program:
      • Rebalance monthly (or as set) using prior-window β_t.
      • Hedge P&L = Σ (contracts * contract_multiplier * ΔPrice_H).
      • Exposure P&L = (Fuel volume) * ΔPrice_JF.
      • Net P&L = Exposure P&L − Hedge P&L − costs.
5) Outputs tidy CSVs (betas, trades, pnl) + JSON summary (hedge effectiveness, TE, VaR/ES).

Optional:
- Collar overlay (buy call, sell put on a chosen hedge underlyer) priced with Black–76 per rebalance bucket.

Inputs (CSV; headers are case-insensitive & flexible)
-----------------------------------------------------
--jetfuel jetfuel.csv       REQUIRED
  Long:  date, price  (USD/bbl)     OR Wide: date, JET_FOB_SING (etc.)
--hedges hedges.csv         REQUIRED
  Long:  date, ticker, price         OR Wide: date, BRENT, WTI, HO, GASOIL, ...
--exposure exposure.csv     OPTIONAL (fuel volumes)
  Columns: date, fuel_bbl            (if missing, assumes 1 bbl per step)
--costs costs.csv           OPTIONAL (trading frictions & multipliers)
  Columns: ticker, fee_bps, slippage_bps, half_spread_bps, mult_bbl, min_ticket_usd
--fx fx.csv                 OPTIONAL (if any non-USD contracts provided)
  Columns: currency, fx_to_usd       (USD per 1 unit of currency)
--vols vols.csv             OPTIONAL (for collar overlay)
  Columns: date, ticker, imp_vol, r (risk-free, decimal), tenor_days

Key CLI knobs
-------------
--jet_col JET               Column in jetfuel file to use (auto-guess if blank)
--tickers BRENT,HO          Hedge universe tickers from hedges.csv (comma list)
--rebalance M               Rebalance freq: D/W/M
--lookback 252              Rolling window for β
--start 2015-01-01          Backtest start
--end   2025-09-01          Backtest end
--turnover_pen_bps 0        Extra turnover penalty bps on hedge notional (round-trip)
--base_ccy USD              Currency (assumes USD everywhere unless fx provided)
--collar ""                 Optional: "<put_pct>,<call_pct>,<ticker>" e.g. "0.90,1.10,BRENT"
--collar_days 30            Tenor in days for collar buckets
--outdir out_fuel_hedge     Output folder

Assumptions & notes
-------------------
• Prices should be in USD/bbl (or will be converted via fx if provided).
• Default contract multipliers (bbl/contract): BRENT=1000, WTI=1000, HO/ULSD=1000, GASOIL=745 (≈100mt diesel), JET=1000.
• Hedge sizes round to nearest whole contract; fractional residual left unhedged.
• Hedge effectiveness = 1 − Var(Net)/Var(Unhedged) on same date set.
• VaR/ES use historical daily net P&L (USD) at 95% (modifiable in code).
• Collar is priced/settled per rebalance bucket using Black–76 on the named hedge ticker’s futures proxy
  (we use its spot/price series as the futures level proxy).

DISCLAIMER: Research tool with simplifying assumptions. Not investment advice.
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

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in str(c).lower():
            return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def dlog(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan).astype(float)).diff()

def pct_to_unit(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s/100.0 if s.max() and s.max()>1 else s

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def roll_ols_beta(Y: pd.Series, X: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling OLS betas for Y on X with intercept.
    Returns DataFrame aligned to index with columns of X for β.
    """
    # Using rolling moments: β = Σxx^{-1} Σxy
    X_ = X.astype(float)
    Y_ = Y.astype(float)
    idx = X_.index
    betas = pd.DataFrame(index=idx, columns=X_.columns, dtype=float)
    # Precompute rolling means
    mX = X_.rolling(window, min_periods=max(30, window//4)).mean()
    mY = Y_.rolling(window, min_periods=max(30, window//4)).mean()
    Xm = X_ - mX
    Ym = (Y_ - mY)
    # Covariances & variances
    # Σxx
    for t in X_.columns:
        for s in X_.columns:
            if t == s or True:
                if (t, s) not in globals().get("_cache_var", {}):
                    pass
    # Compute Σxx and Σxy in a loop (small dimensions)
    for i, dt in enumerate(idx):
        # window slice
        start = max(0, i - window + 1)
        Xw = X_.iloc[start:i+1, :]
        Yw = Y_.iloc[start:i+1]
        if Xw.shape[0] < max(30, window//4):
            continue
        Xc = Xw - Xw.mean()
        Yc = Yw - Yw.mean()
        Sxx = (Xc.T @ Xc).values
        Sxy = (Xc.T @ Yc.values.reshape(-1,1))
        try:
            beta_i = np.linalg.pinv(Sxx) @ Sxy
            betas.iloc[i, :] = beta_i.ravel()
        except Exception:
            continue
    return betas

# ----------------------------- data loaders -----------------------------

def load_price_series(path: str, pick_col: Optional[str]=None, label: str="JET") -> pd.Series:
    df = pd.read_csv(path)
    dt = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    df = df.sort_values("date")
    if pick_col and pick_col in df.columns:
        s = df.set_index("date")[pick_col].astype(float)
        s.name = label
        return s
    # Long: date, price
    if ncol(df,"price"):
        s = df.set_index("date")[ncol(df,"price")].astype(float)
        s.name = label
        return s
    # Wide: pick first non-date column if none specified
    cols = [c for c in df.columns if c != "date"]
    if not cols:
        raise ValueError(f"No price column found in {path}.")
    if pick_col is None and len(cols)==1:
        s = df.set_index("date")[cols[0]].astype(float)
        s.name = label
        return s
    # try to auto-guess jet column name
    candidates = [c for c in cols if "jet" in str(c).lower() or "kerosene" in str(c).lower()]
    use = pick_col or (candidates[0] if candidates else cols[0])
    s = df.set_index("date")[use].astype(float)
    s.name = label
    return s

def load_hedges(path: str, tickers: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    df = df.sort_values("date")
    if ncol(df,"ticker") and ncol(df,"price"):
        df = df.rename(columns={ncol(df,"ticker"):"ticker", ncol(df,"price"):"price"})
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        if tickers:
            df = df[df["ticker"].isin(tickers)]
        piv = df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
        return piv
    # wide
    piv = df.set_index("date")
    if tickers:
        keep = [t for t in tickers if t in piv.columns]
        if not keep:
            raise ValueError("Requested tickers not found in hedges file.")
        piv = piv[keep]
    return piv

def load_exposure(path: Optional[str]) -> pd.Series:
    if not path:
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    dt = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={dt:"date"})
    df["date"] = to_date(df["date"])
    vol_c = ncol(df,"fuel_bbl") or ncol(df,"volume_bbl") or ncol(df,"bbl")
    if not vol_c:
        raise ValueError("exposure.csv must have a fuel_bbl (or volume_bbl/bbl) column.")
    s = df.set_index("date")[vol_c].astype(float)
    s.name = "fuel_bbl"
    return s

def load_costs(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["ticker","fee_bps","slippage_bps","half_spread_bps","mult_bbl","min_ticket_usd"])
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"ticker") or "ticker"):"ticker",
        (ncol(df,"fee_bps") or "fee_bps"):"fee_bps",
        (ncol(df,"slippage_bps") or "slippage_bps"):"slippage_bps",
        (ncol(df,"half_spread_bps") or "half_spread_bps"):"half_spread_bps",
        (ncol(df,"mult_bbl") or "mult_bbl"):"mult_bbl",
        (ncol(df,"min_ticket_usd") or "min_ticket_usd"):"min_ticket_usd",
    }
    df = df.rename(columns=ren)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for c in ["fee_bps","slippage_bps","half_spread_bps","mult_bbl","min_ticket_usd"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    return df

def load_vols(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"ticker") or "ticker"): "ticker",
        (ncol(df,"imp_vol") or "imp_vol"): "imp_vol",
        (ncol(df,"r") or "r"): "r",
        (ncol(df,"tenor_days") or "tenor_days"): "tenor_days",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["imp_vol"] = safe_num(df["imp_vol"])
    df["r"] = safe_num(df["r"]).fillna(0.0)
    df["tenor_days"] = safe_num(df["tenor_days"]).fillna(30.0)
    return df

# ----------------------------- contracts & costs -----------------------------

DEFAULT_MULT = {
    "BRENT": 1000.0,  # bbl/contract
    "WTI": 1000.0,
    "HO": 1000.0,     # NY Harbor ULSD/Heating Oil ≈ 42k gal = 1000 bbl
    "ULSD": 1000.0,
    "GASOIL": 745.0,  # ICE Gasoil 100 mt ≈ 745 bbl
    "JET": 1000.0,
}

def contract_mult(ticker: str, COSTS: pd.DataFrame) -> float:
    if not COSTS.empty and "mult_bbl" in COSTS.columns:
        row = COSTS[COSTS["ticker"]==ticker]
        if not row.empty and pd.notna(row["mult_bbl"].iloc[0]):
            return float(row["mult_bbl"].iloc[0])
    return DEFAULT_MULT.get(ticker, 1000.0)

def ticket_cost_bps(ticker: str, COSTS: pd.DataFrame, default_bps: Tuple[float,float,float]=(1.0, 5.0, 5.0)) -> Tuple[float,float,float]:
    """
    Returns (fee_bps, half_spread_bps, slippage_bps)
    """
    if COSTS.empty:
        return default_bps
    row = COSTS[COSTS["ticker"]==ticker]
    if row.empty:
        return default_bps
    fee = float(row["fee_bps"].iloc[0]) if pd.notna(row["fee_bps"].iloc[0]) else default_bps[0]
    hs  = float(row["half_spread_bps"].iloc[0]) if pd.notna(row["half_spread_bps"].iloc[0]) else default_bps[1]
    sl  = float(row["slippage_bps"].iloc[0]) if pd.notna(row["slippage_bps"].iloc[0]) else default_bps[2]
    return fee, hs, sl

def min_ticket_usd(ticker: str, COSTS: pd.DataFrame, default_min: float=0.0) -> float:
    if COSTS.empty or "min_ticket_usd" not in COSTS.columns:
        return default_min
    row = COSTS[COSTS["ticker"]==ticker]
    if row.empty or pd.isna(row["min_ticket_usd"].iloc[0]):
        return default_min
    return float(row["min_ticket_usd"].iloc[0])

# ----------------------------- collar pricing (Black–76) -----------------------------

from math import log, sqrt, exp
from scipy.stats import norm  # acceptable dependency; if unavailable, can approximate with error function

def black76_price(F: float, K: float, vol: float, tau: float, r: float, is_call: bool=True) -> float:
    """
    Black–76 option on futures: discounted BS with F as underlying.
    """
    if F<=0 or K<=0 or vol<=0 or tau<=0:
        # intrinsic only
        payoff = max(0.0, (F-K)) if is_call else max(0.0, (K-F))
        return exp(-r*tau) * payoff
    d1 = (log(F/K) + 0.5*vol*vol*tau) / (vol*sqrt(tau))
    d2 = d1 - vol*sqrt(tau)
    df = exp(-r*tau)
    if is_call:
        return df*(F*norm.cdf(d1) - K*norm.cdf(d2))
    else:
        return df*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

# ----------------------------- core engine -----------------------------

def compute_betas(J: pd.Series, H: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Returns rolling betas (columns=hedge tickers).
    """
    Rj = dlog(J).rename("JET_R")
    Rh = H.apply(dlog)
    Rh = Rh.reindex(Rj.index).dropna(how="all")
    Rj = Rj.reindex(Rh.index)
    betas = roll_ols_beta(Rj, Rh, window=lookback)
    return betas

def rebalance_dates(idx: pd.DatetimeIndex, freq: str="M") -> List[pd.Timestamp]:
    if freq.upper().startswith("D"):
        return list(idx)
    if freq.upper().startswith("W"):
        return list(pd.Series(idx).resample("W-FRI").last().dropna())
    # monthly by default
    return list(pd.Series(idx).resample("M").last().dropna())

def backtest(
    J_price: pd.Series,
    H_price: pd.DataFrame,
    betas: pd.DataFrame,
    exposure_bbl: pd.Series,
    rebalance: str,
    COSTS: pd.DataFrame,
    turnover_pen_bps: float = 0.0,
    collar_cfg: Optional[Tuple[float,float,str]] = None,
    vols: Optional[pd.DataFrame] = None,
    collar_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Simulates the hedging program; returns (trades, pnl_daily, betas_used, summary).
    """
    # Align everything
    J = J_price.sort_index()
    H = H_price.sort_index().reindex(J.index).dropna(how="all")
    J = J.reindex(H.index)
    betas = betas.reindex(H.index).fillna(method="ffill")

    # Exposure bbl per day
    if exposure_bbl.empty:
        exposure = pd.Series(1.0, index=J.index, name="fuel_bbl")
    else:
        exposure = exposure_bbl.reindex(J.index).fillna(method="ffill")
        if exposure.isna().all():
            exposure = pd.Series(1.0, index=J.index, name="fuel_bbl")

    # Returns (for P&L)
    RJ = dlog(J)
    RH = H.apply(dlog)

    # Rebalance dates (use previous day betas)
    rdates = [d for d in rebalance_dates(J.index, rebalance) if d in J.index]

    # State
    current_contracts = {t: 0 for t in H.columns}
    last_prices = {t: np.nan for t in H.columns}
    trades_rows = []
    pnl_rows = []

    # Collars state per bucket
    collar_active = collar_cfg is not None and vols is not None and not vols.empty
    collar_positions = []  # list of dicts with open_date, close_date, ticker, Kp, Kc, premium, units (bbl)

    # Iterate days; execute trade on rebalance day using β(t-1)
    prev_day = None
    for day in J.index:
        # mark-to-market daily
        exp_bbl = float(exposure.loc[day])
        # Hedge P&L from futures changes
        hedge_pnl = 0.0
        for tkr in H.columns:
            px = float(H.loc[day, tkr]) if pd.notna(H.loc[day, tkr]) else np.nan
            if pd.notna(px) and pd.notna(last_prices.get(tkr, np.nan)):
                dP = px - last_prices[tkr]
                hedge_pnl += current_contracts[tkr] * contract_mult(tkr, COSTS) * dP
            last_prices[tkr] = px

        # Collar MTM/settlement if bucket end
        collar_pnl_today = 0.0
        if collar_active:
            # any positions that mature today?
            matured = [p for p in collar_positions if p["close_date"] == day]
            for pos in matured:
                F_T = float(H.loc[day, pos["ticker"]]) if pos["ticker"] in H.columns else np.nan
                if pd.notna(F_T):
                    # Payoff: short put + long call (net payoff), then subtract premium (paid at open)
                    put_pay = max(0.0, pos["Kp"] - F_T)
                    call_pay = max(0.0, F_T - pos["Kc"])
                    payoff = (-put_pay + call_pay) * pos["units_bbl"]
                    collar_pnl_today += payoff - pos["premium_usd"]
            # remove matured
            collar_positions = [p for p in collar_positions if p["close_date"] != day]

        # Exposure P&L (daily)
        # Using ΔPrice_JF * bbl
        if prev_day is not None and pd.notna(J.loc[day]) and pd.notna(J.loc[prev_day]):
            dP_j = float(J.loc[day] - J.loc[prev_day])
            exp_pnl = exp_bbl * dP_j
        else:
            exp_pnl = 0.0

        net_pnl = exp_pnl - hedge_pnl + collar_pnl_today
        pnl_rows.append({"date": day, "exp_pnl": exp_pnl, "hedge_pnl": -hedge_pnl, "collar_pnl": collar_pnl_today, "net_pnl": net_pnl})

        # Rebalance at day end (set positions for next day)
        if day in rdates:
            # Use yesterday's betas
            be = betas.loc[:day].iloc[-1] if not betas.loc[:day].empty else pd.Series(0.0, index=H.columns)
            # Target hedge notional in USD := β' × (exp_bbl × J_price(day))
            exp_usd = exp_bbl * float(J.loc[day])
            desired_contracts = {}
            for tkr in H.columns:
                β = float(be.get(tkr, 0.0)) if pd.notna(be.get(tkr, np.nan)) else 0.0
                px = float(H.loc[day, tkr]) if pd.notna(H.loc[day, tkr]) else np.nan
                mult = contract_mult(tkr, COSTS)
                # Position sign: LONG hedge if β>0 (instrument rises with jet fuel), needed notional = β * exp_usd
                if pd.isna(px) or px<=0 or mult<=0:
                    desired_contracts[tkr] = current_contracts[tkr]
                    continue
                cts = (β * exp_usd) / (mult * px)
                desired_contracts[tkr] = int(np.round(cts))

            # Tickets & costs
            for tkr in H.columns:
                delta_cts = desired_contracts[tkr] - current_contracts[tkr]
                if delta_cts == 0:
                    continue
                px = float(H.loc[day, tkr]) if pd.notna(H.loc[day, tkr]) else np.nan
                mult = contract_mult(tkr, COSTS)
                notion = abs(delta_cts) * mult * (px if pd.notna(px) else 0.0)
                # Drop micro tickets below min_ticket_usd if specified
                if notion < min_ticket_usd(tkr, COSTS, default_min=0.0):
                    continue
                fee_bps, hs_bps, sl_bps = ticket_cost_bps(tkr, COSTS)
                trade_cost = notion * (fee_bps + hs_bps + sl_bps)/1e4
                # Optional turnover penalty on notional
                trade_cost += notion * (turnover_pen_bps/1e4) if turnover_pen_bps else 0.0
                trades_rows.append({
                    "date": day, "ticker": tkr, "contracts_delta": int(delta_cts),
                    "contracts_after": int(current_contracts[tkr] + delta_cts),
                    "price": px, "mult_bbl": mult, "trade_notional_usd": notion, "est_costs_usd": trade_cost
                })
                # apply
                current_contracts[tkr] += int(delta_cts)

            # Open a new collar for next bucket
            if collar_active:
                put_pct, call_pct, under = collar_cfg
                if under in H.columns:
                    F0 = float(H.loc[day, under])
                    # Get vol & r for this date (or nearest prior)
                    vol_row = vols[vols["ticker"]==under]
                    vol_row = vol_row[vol_row["date"]<=day]
                    if not vol_row.empty:
                        row = vol_row.iloc[-1]
                        vol = float(row["imp_vol"])
                        r = float(row["r"])
                        tau = max(1.0, float(collar_days))/365.0
                        Kp = put_pct * F0
                        Kc = call_pct * F0
                        # Size in bbl equals exposure for the bucket (approx. next bucket exposure)
                        units_bbl = exp_bbl
                        prem = black76_price(F0, Kc, vol, tau, r, is_call=True) - black76_price(F0, Kp, vol, tau, r, is_call=False)
                        collar_positions.append({
                            "open_date": day,
                            "close_date": next_bucket_date(day, J.index, collar_days),
                            "ticker": under, "Kp": Kp, "Kc": Kc,
                            "premium_usd": prem * units_bbl,
                            "units_bbl": units_bbl
                        })

        prev_day = day

    trades = pd.DataFrame(trades_rows)
    pnl = pd.DataFrame(pnl_rows).set_index("date").sort_index()
    betas_used = betas.copy()

    # Effectiveness
    unhedged = pnl["exp_pnl"].fillna(0)
    net = pnl["net_pnl"].fillna(0)
    eff = 1.0 - (net.var(ddof=0) / (unhedged.var(ddof=0) if unhedged.var(ddof=0)>0 else np.nan))

    # VaR/ES (hist 95%)
    q = 0.05
    var95 = float(np.nanquantile(net.values, q)) if len(net) else np.nan
    es95  = float(net[net<=var95].mean()) if len(net[net<=var95]) else np.nan

    summary = {
        "dates": {"start": str(J.index.min().date()) if len(J)>0 else None,
                  "end": str(J.index.max().date()) if len(J)>0 else None},
        "hedge_effectiveness_pct": float(eff*100) if eff==eff else np.nan,
        "pnl_mean_usd": float(net.mean()) if len(net) else np.nan,
        "pnl_vol_usd": float(net.std(ddof=0)) if len(net) else np.nan,
        "VaR95_usd": var95,
        "ES95_usd": es95,
        "avg_daily_exposure_bbl": float(exposure.mean()) if len(exposure) else np.nan
    }

    return trades, pnl.reset_index(), betas_used, summary

def next_bucket_date(d: pd.Timestamp, idx: pd.DatetimeIndex, days: int) -> pd.Timestamp:
    target = d + pd.Timedelta(days=days)
    # choose nearest calendar date present in index at/after target
    later = idx[idx>=target]
    return later[0] if len(later)>0 else idx[-1]

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    jetfuel: str
    hedges: str
    exposure: Optional[str]
    costs: Optional[str]
    fx: Optional[str]
    vols: Optional[str]
    jet_col: Optional[str]
    tickers: List[str]
    rebalance: str
    lookback: int
    start: Optional[str]
    end: Optional[str]
    turnover_pen_bps: float
    base_ccy: str
    collar: Optional[str]
    collar_days: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Airline Jet Fuel Hedge backtester")
    ap.add_argument("--jetfuel", required=True)
    ap.add_argument("--hedges", required=True)
    ap.add_argument("--exposure", default="")
    ap.add_argument("--costs", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--vols", default="")
    ap.add_argument("--jet_col", default="")
    ap.add_argument("--tickers", default="BRENT,HO", help="Comma list of hedge tickers present in hedges.csv")
    ap.add_argument("--rebalance", default="M", help="Rebalance frequency: D/W/M")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--turnover_pen_bps", type=float, default=0.0)
    ap.add_argument("--base_ccy", default="USD")
    ap.add_argument("--collar", default="", help='Optional: "<put_pct>,<call_pct>,<ticker>" e.g. "0.9,1.1,BRENT"')
    ap.add_argument("--collar_days", type=int, default=30)
    ap.add_argument("--outdir", default="out_fuel_hedge")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    J = load_price_series(args.jetfuel, pick_col=(args.jet_col or None), label="JET")
    H = load_hedges(args.hedges, tickers=tickers)
    EXP = load_exposure(args.exposure) if args.exposure else pd.Series(dtype=float)
    COSTS = load_costs(args.costs) if args.costs else pd.DataFrame()
    VOLS  = load_vols(args.vols) if args.vols else pd.DataFrame()

    # Filter dates
    if args.start:
        J = J[J.index >= pd.to_datetime(args.start)]
        H = H[H.index >= pd.to_datetime(args.start)]
        if not EXP.empty: EXP = EXP[EXP.index >= pd.to_datetime(args.start)]
    if args.end:
        J = J[J.index <= pd.to_datetime(args.end)]
        H = H[H.index <= pd.to_datetime(args.end)]
        if not EXP.empty: EXP = EXP[EXP.index <= pd.to_datetime(args.end)]

    # Align to common dates
    idx = J.index.intersection(H.index)
    if len(idx) < max(60, int(args.lookback*0.5)):
        raise ValueError("Insufficient overlapping dates between jet fuel and hedges.")
    J = J.reindex(idx)
    H = H.reindex(idx).dropna(how="all")
    if not EXP.empty:
        EXP = EXP.reindex(idx).fillna(method="ffill")

    # Betas
    betas = compute_betas(J, H, lookback=int(args.lookback))
    betas.to_csv(outdir / "betas_rolling.csv", index=True)

    # Collar config
    collar_cfg = None
    if args.collar:
        try:
            put_pct, call_pct, under = args.collar.split(",")
            collar_cfg = (float(put_pct), float(call_pct), under.strip().upper())
        except Exception:
            print("Warning: could not parse --collar; expected '0.9,1.1,BRENT'. Ignoring collar.")
            collar_cfg = None

    # Backtest
    trades, pnl, betas_used, summary = backtest(
        J_price=J, H_price=H, betas=betas, exposure_bbl=EXP, rebalance=args.rebalance,
        COSTS=COSTS, turnover_pen_bps=float(args.turnover_pen_bps),
        collar_cfg=collar_cfg, vols=VOL(S=VOLs) if False else VOLS, collar_days=int(args.collar_days)
    )

    # Outputs
    trades.to_csv(outdir / "trades.csv", index=False)
    pnl.to_csv(outdir / "pnl_daily.csv", index=False)
    betas_used.to_csv(outdir / "betas_used.csv", index=True)

    # Diagnostics
    pnl_df = pnl.copy()
    pnl_df["cum_net"] = pnl_df["net_pnl"].fillna(0).cumsum()
    pnl_df.to_csv(outdir / "pnl_with_cum.csv", index=False)

    # Snapshot & summary
    latest_betas = betas_used.dropna().iloc[-1].sort_values(ascending=False).to_dict() if not betas_used.dropna().empty else {}
    summary.update({
        "latest_betas": latest_betas,
        "tickers": tickers,
        "rebalance": args.rebalance,
        "lookback": int(args.lookback),
        "collar": args.collar or None
    })
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        jetfuel=args.jetfuel, hedges=args.hedges, exposure=(args.exposure or None),
        costs=(args.costs or None), fx=(args.fx or None), vols=(args.vols or None),
        jet_col=(args.jet_col or None), tickers=tickers, rebalance=args.rebalance,
        lookback=int(args.lookback), start=(args.start or None), end=(args.end or None),
        turnover_pen_bps=float(args.turnover_pen_bps), base_ccy=args.base_ccy,
        collar=(args.collar or None), collar_days=int(args.collar_days), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Airline Fuel Hedges ==")
    print(f"Dates: {summary['dates']['start']} → {summary['dates']['end']} | Tickers: {', '.join(tickers)}")
    print(f"Hedge effectiveness: {summary['hedge_effectiveness_pct']:.2f}% | VaR95: {summary['VaR95_usd']:.0f} | ES95: {summary['ES95_usd']:.0f}")
    if latest_betas:
        top = sorted(latest_betas.items(), key=lambda x: -abs(x[1]))[:5]
        print("Latest betas:", ", ".join([f"{k}={v:+.2f}" for k,v in top]))
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
