#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_etf_rebalance.py — Global ETF rebalance planner (targets → integer trades)

What this does
--------------
Given current holdings, targets (by ticker OR by macro buckets like region/sector),
ETF exposures (ETF → bucket weights), prices, FX, costs, and optional tax lots,
this script computes an executable rebalance:

1) Solves for *target ETF weights*
   - If targets specify tickers: use them directly (normalize)
   - If targets specify buckets: solve a constrained NNLS to match bucket targets
     using the ETF exposures matrix (sum(weights)=1, weights>=0)

2) Converts target weights → target shares (integer)
   - Accounts for cash min/max, transaction costs, turnover cap, lot size constraints
   - Rounds with cash & turnover aware greedy adjustment

3) Produces *buy/sell* tickets
   - Estimates costs (slippage + fees + half-spread)
   - Optional tax-lot selection for sells (HIFO) with realized gain estimate
   - Handles multi-currency tickers with FX conversion to a base currency

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--holdings holdings.csv      REQUIRED
  Columns: ticker, shares [, price] [, currency] [, lotsize]
           If price/currency omitted, supply via --prices / --fx

--targets targets.csv        REQUIRED
  EITHER per-ticker targets:
    Columns: ticker, target_weight_pct
  OR bucket targets:
    Columns: bucket, target_weight_pct
    (then provide --exposures to map ETF→bucket)

--exposures exposures.csv    OPTIONAL (needed for bucket targets)
  Columns: ticker, bucket, weight_pct    (ETF's share of bucket; rows sum ≤/≈100%)

--prices prices.csv          OPTIONAL
  Columns (long or wide):
    long: date, ticker, price [, currency]
    wide: ticker columns with latest prices in base currency
  If currency given and != base_ccy, FX will be used to convert to base.

--fx fx.csv                  OPTIONAL
  Columns: currency, fx_to_base   (how many base_ccy per 1 unit of 'currency')

--costs costs.csv            OPTIONAL
  Columns: ticker, fee_bps, half_spread_bps, slippage_bps [, min_trade_usd] [, lotsize]
  (Any missing cost fields default to CLI defaults.)

--lots lots.csv              OPTIONAL (for sell tax estimation)
  Columns: ticker, lot_id, shares, cost_basis, acquisition_date

--constraints constraints.csv OPTIONAL (free-form key,value pairs)
  Keys (examples):
    turnover_cap_pct = 30
    cash_min_pct     = 0
    cash_max_pct     = 5
    buy_blacklist    = "RSX,EWW"
    sell_blacklist   = "…"
    allowlist        = "VEU,VT,AGG"   (if present, only these can be traded)

Key CLI knobs
-------------
--base_ccy USD                Base/portfolio currency
--portfolio_cash 0            Starting cash in base currency
--slippage_bps 5              Default slippage if not in costs
--fee_bps 1                   Default fees/commissions
--half_spread_bps 5           Default quoted spread half
--turnover_cap_pct 35         Max gross turnover % of portfolio MV
--cash_min_pct 0              Min cash after trades
--cash_max_pct 5              Max cash after trades
--min_trade_usd 50            Drop tiny tickets below this notional
--outdir out_rebalance        Output folder

Outputs
-------
- trades.csv             Ticker-level buy/sell with shares, notional (base), est_costs, realized_gain (if lots)
- before_after.csv       Holdings & weights before/after, plus deltas
- exposures_check.csv    Achieved bucket weights vs target (if exposures provided/used)
- cash_fx.csv            Cash before/after, FX breakdown, cost totals
- summary.json           Headline stats (turnover, drift fix, costs, tracking error proxy)
- config.json            Run configuration

DISCLAIMER: Research/ops helper; sanity-check before executing live orders.
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

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pct_to_unit(s: pd.Series) -> pd.Series:
    s = safe_num(s)
    return s/100.0 if s.max() and s.max()>1 else s

def parse_csv_any(path: Optional[str]) -> pd.DataFrame:
    return pd.read_csv(path) if path else pd.DataFrame()

def to_float(x, default=np.nan) -> float:
    try: return float(x)
    except Exception: return default

def split_csv_list(s: str) -> List[str]:
    if not isinstance(s, str): return []
    return [x.strip().upper() for x in s.split(",") if x.strip()]

# ----------------------------- loaders -----------------------------

def load_holdings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"ticker") or "ticker"): "ticker",
        (ncol(df,"shares") or "shares"): "shares",
        (ncol(df,"price") or "price"): "price",
        (ncol(df,"currency") or "currency"): "currency",
        (ncol(df,"lotsize") or "lot_size" or "lotsize"): "lotsize",
    }
    df = df.rename(columns=ren)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = safe_num(df["shares"]).fillna(0.0)
    if "price" in df.columns: df["price"] = safe_num(df["price"])
    if "lotsize" in df.columns: df["lotsize"] = safe_num(df["lotsize"]).fillna(1).astype(int)
    else: df["lotsize"] = 1
    if "currency" in df.columns: df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    else: df["currency"] = np.nan
    return df

def load_targets(path: str) -> Tuple[pd.DataFrame, str]:
    """
    Returns (targets_df, mode) where mode ∈ {"ticker","bucket"}
    """
    df = pd.read_csv(path)
    if ncol(df,"ticker"):
        ren = {(ncol(df,"ticker") or "ticker"):"ticker",
               (ncol(df,"target_weight_pct") or ncol(df,"target") or "target"):"target"}
        df = df.rename(columns=ren)
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["target"] = pct_to_unit(df["target"]).fillna(0.0)
        return df[["ticker","target"]], "ticker"
    else:
        ren = {(ncol(df,"bucket") or "bucket"):"bucket",
               (ncol(df,"target_weight_pct") or ncol(df,"target") or "target"):"target"}
        df = df.rename(columns=ren)
        df["bucket"] = df["bucket"].astype(str).str.upper().str.strip()
        df["target"] = pct_to_unit(df["target"]).fillna(0.0)
        return df[["bucket","target"]], "bucket"

def load_exposures(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"ticker") or "ticker"):"ticker",
           (ncol(df,"bucket") or "bucket"):"bucket",
           (ncol(df,"weight_pct") or ncol(df,"weight") or "weight"):"weight"}
    df = df.rename(columns=ren)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["bucket"] = df["bucket"].astype(str).str.upper().str.strip()
    df["weight"] = pct_to_unit(df["weight"]).fillna(0.0)
    return df

def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    if ncol(df,"date") and ncol(df,"ticker"):
        ren = {(ncol(df,"date") or "date"):"date",
               (ncol(df,"ticker") or "ticker"):"ticker",
               (ncol(df,"price") or ncol(df,"close") or "price"):"price",
               (ncol(df,"currency") or "currency"):"currency"}
        df = df.rename(columns=ren)
        df.sort_values("date", inplace=True)
        last = df.groupby("ticker").tail(1)
        last["currency"] = last.get("currency", pd.Series(np.nan, index=last.index))
        last = last[["ticker","price","currency"]]
        last["ticker"] = last["ticker"].astype(str).str.upper().str.strip()
        if "currency" in last.columns:
            last["currency"] = last["currency"].astype(str).str.upper().str.strip()
        return last
    else:
        # maybe wide with tickers as columns
        tickers = [c for c in df.columns if c.lower() != (ncol(df,"date") or "date")]
        if (ncol(df,"date") or None) in df.columns:
            df = df.sort_values(ncol(df,"date"))
            last_row = df.tail(1)
            recs = []
            for t in tickers:
                recs.append({"ticker": str(t).upper(), "price": to_float(last_row.iloc[0][t])})
            return pd.DataFrame(recs)
        else:
            # single row
            recs = []
            row = df.iloc[-1]
            for t in df.columns:
                recs.append({"ticker": str(t).upper(), "price": to_float(row[t])})
            return pd.DataFrame(recs)

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"currency") or "currency"):"currency",
           (ncol(df,"fx_to_base") or "fx_to_base"):"fx_to_base"}
    df = df.rename(columns=ren)
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    df["fx_to_base"] = safe_num(df["fx_to_base"])
    return df

def load_costs(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"ticker") or "ticker"):"ticker",
           (ncol(df,"fee_bps") or "fee_bps"):"fee_bps",
           (ncol(df,"half_spread_bps") or "half_spread_bps"):"half_spread_bps",
           (ncol(df,"slippage_bps") or "slippage_bps"):"slippage_bps",
           (ncol(df,"min_trade_usd") or "min_trade_usd"):"min_trade_usd",
           (ncol(df,"lotsize") or ncol(df,"lot_size") or "lotsize"):"lotsize"}
    df = df.rename(columns=ren)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for c in ["fee_bps","half_spread_bps","slippage_bps","min_trade_usd","lotsize"]:
        if c in df.columns: df[c] = safe_num(df[c])
    if "lotsize" in df.columns: df["lotsize"] = df["lotsize"].fillna(1).astype(int)
    return df

def load_lots(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"ticker") or "ticker"):"ticker",
           (ncol(df,"lot_id") or "lot_id"):"lot_id",
           (ncol(df,"shares") or "shares"):"shares",
           (ncol(df,"cost_basis") or ncol(df,"cost_basis_per_share") or "cost_basis"):"cost_basis",
           (ncol(df,"acquisition_date") or "acquisition_date"):"acquisition_date"}
    df = df.rename(columns=ren)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = safe_num(df["shares"]).fillna(0.0)
    df["cost_basis"] = safe_num(df["cost_basis"]).fillna(np.nan)
    return df

def load_constraints(path: Optional[str]) -> Dict[str, str]:
    if not path: return {}
    df = pd.read_csv(path)
    k = ncol(df,"key") or "key"; v = ncol(df,"value") or "value"
    out = {}
    for _, r in df.iterrows():
        out[str(r[k]).strip().lower()] = str(r[v]).strip()
    return out


# ----------------------------- core math -----------------------------

def nnls_exposures_solve(etfs: List[str], buckets: List[str], E: pd.DataFrame, target_b: pd.Series,
                         max_iter: int=2000, lr: float=0.1, tol: float=1e-8) -> pd.Series:
    """
    Projected gradient descent on min ||E w - t||^2, s.t. w>=0, 1'w=1.
    E: DataFrame indexed by bucket, columns by etf (weights per bucket).
    target_b: Series indexed by bucket (sum to 1).
    Returns Series of ETF weights.
    """
    C = E.reindex(index=buckets, columns=etfs).fillna(0.0).values
    t = target_b.reindex(buckets).fillna(0.0).values
    n = len(etfs)
    if n == 0: return pd.Series(dtype=float)
    # init proportional to cols' total exposure to covered buckets
    colsum = C.sum(axis=0)
    w = colsum / (colsum.sum() + 1e-12)
    w = np.maximum(0.0, w)
    w = w / (w.sum() + 1e-12)
    for it in range(max_iter):
        r = C @ w - t                      # residual
        grad = 2 * (C.T @ r)               # gradient
        w_new = w - lr * grad
        w_new = np.maximum(0.0, w_new)
        s = w_new.sum()
        if s <= 0:
            w_new = np.ones_like(w_new)/len(w_new)
        else:
            w_new = w_new / s
        if np.linalg.norm(w_new - w, 1) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=etfs)

def apply_fx(price: float, ccy: Optional[str], base_ccy: str, fx: pd.DataFrame) -> float:
    if not isinstance(ccy, str) or ccy.upper()==base_ccy.upper() or fx.empty:
        return float(price)
    row = fx[fx["currency"]==ccy.upper()]
    if row.empty: return float(price)
    return float(price) * float(row["fx_to_base"].iloc[0])

def cost_per_share_base(price_base: float, fee_bps: float, half_spread_bps: float, slippage_bps: float) -> float:
    bps = (fee_bps or 0) + (half_spread_bps or 0) + (slippage_bps or 0)
    return price_base * (bps / 1e4)

def greedy_round_shares(
    current_shares: pd.Series,
    target_weights: pd.Series,
    prices_base: pd.Series,
    lotsize: pd.Series,
    cash_base: float,
    mv_base: float,
    costs_df: pd.DataFrame,
    min_trade_usd: float,
    turnover_cap: float,
    cash_bounds: Tuple[float,float]
) -> Tuple[pd.Series, float, float, float]:
    """
    Greedy allocate/round to meet cash & turnover constraints.
    Returns (new_shares, cash_after, est_costs_total, est_turnover).
    """
    tickers = prices_base.index.tolist()
    w_tgt = target_weights.reindex(tickers).fillna(0.0).clip(lower=0.0)
    w_tgt = w_tgt / (w_tgt.sum() + 1e-12)
    px = prices_base.reindex(tickers).astype(float)
    lot = lotsize.reindex(tickers).fillna(1).astype(int)

    # Desired dollars (before costs)
    desired_usd = w_tgt * (mv_base + cash_base)
    cur_usd = current_shares.reindex(tickers).fillna(0.0) * px
    delta_usd = desired_usd - cur_usd

    # Initial integer suggestion by rounding shares
    tgt_shares_float = desired_usd / px
    tgt_shares_rounded = ((tgt_shares_float / lot).round() * lot).fillna(0.0)

    # Compute initial cash & turnover
    trade_shares = tgt_shares_rounded - current_shares.reindex(tickers).fillna(0.0)
    def est_ticket_cost(tkr, sh):
        c = costs_df.get(tkr, {"fee_bps":0,"half_spread_bps":0,"slippage_bps":0,"min_trade_usd":min_trade_usd})
        cps = cost_per_share_base(px[tkr], c["fee_bps"], c["half_spread_bps"], c["slippage_bps"])
        return abs(sh) * cps
    est_costs = sum(est_ticket_cost(t, trade_shares[t]) for t in tickers)
    cash_after = cash_base - (trade_shares * px).sum() - est_costs

    # Turnover
    gross_turnover = (abs(trade_shares) * px).sum() / (mv_base + 1e-12)

    # Drop micro tickets
    for t in tickers:
        notional = abs(trade_shares[t]) * px[t]
        c = costs_df.get(t, {"min_trade_usd":min_trade_usd})
        if notional < max(min_trade_usd, float(c.get("min_trade_usd") or 0)):
            trade_shares[t] = 0.0

    # Iterative adjust to fit cash & turnover and cash bounds
    max_iters = 5000
    it = 0
    def recalc():
        costs = sum(est_ticket_cost(t, trade_shares[t]) for t in tickers)
        cash = cash_base - (trade_shares * px).sum() - costs
        turnover = (abs(trade_shares) * px).sum() / (mv_base + 1e-12)
        return cash, costs, turnover
    cash_after, est_costs, gross_turnover = recalc()

    # If turnover too high, scale trades towards zero
    if gross_turnover > turnover_cap + 1e-9:
        scale = turnover_cap / gross_turnover
        trade_shares = (trade_shares * scale).round()  # keep ints
        cash_after, est_costs, gross_turnover = recalc()

    cash_min, cash_max = cash_bounds

    # Fix cash outside bounds by greedily selling/buying nearest-to-neutral tickers
    # Priority: move those with largest $drift (desired - current)
    order = pd.Series(delta_usd.abs(), index=tickers).sort_values(ascending=False).index.tolist()

    while (cash_after < cash_min - 1e-6 or cash_after > cash_max + 1e-6) and it < max_iters:
        it += 1
        # Need to raise cash (cash_after < min) → net sell; else net buy
        need_sell = cash_after < cash_min
        moved = False
        for t in order:
            step = lot[t]
            if need_sell:
                # If we can sell (current + planned > 0)
                if (current_shares.get(t,0.0) + trade_shares[t]) >= step:
                    trade_shares[t] -= step
                    moved = True
            else:
                # net buy if cash too high (push towards target)
                trade_shares[t] += step
                moved = True
            cash_after, est_costs, gross_turnover = recalc()
            # enforce turnover
            if gross_turnover > turnover_cap + 1e-9:
                # revert this step
                trade_shares[t] += step if need_sell else -step
                cash_after, est_costs, gross_turnover = recalc()
                continue
            if cash_min - 1e-6 <= cash_after <= cash_max + 1e-6:
                break
        if not moved:
            break  # cannot adjust further

    new_shares = current_shares.reindex(tickers).fillna(0.0) + trade_shares
    new_shares = new_shares.clip(lower=0.0)  # no shorting

    return new_shares, float(cash_after), float(est_costs), float(gross_turnover)

def choose_sell_lots_hifo(lots: pd.DataFrame, ticker: str, sell_shares: float) -> Tuple[List[Dict], float]:
    """
    Pick lots to sell by HIFO (highest cost first) to minimize realized gains.
    Returns (list of {'lot_id','sell_shares','realized_gain_usd'}, total_realized_gain).
    """
    if lots.empty or sell_shares <= 0:
        return [], 0.0
    L = lots[lots["ticker"]==ticker].copy()
    if L.empty:
        return [], 0.0
    L = L[L["shares"]>0].copy()
    if L.empty:
        return [], 0.0
    # price must be supplied later to compute gain; here store basis only; we'll compute gain with current price
    L = L.sort_values("cost_basis", ascending=False)  # HIFO
    remaining = sell_shares
    picks = []
    for _, r in L.iterrows():
        if remaining <= 0: break
        take = float(min(remaining, r["shares"]))
        picks.append({"lot_id": r.get("lot_id", ""), "sell_shares": take, "cost_basis": float(r["cost_basis"])})
        remaining -= take
    return picks, 0.0

# ----------------------------- orchestration -----------------------------

@dataclass
class Config:
    holdings: str
    targets: str
    exposures: Optional[str]
    prices: Optional[str]
    fx: Optional[str]
    costs: Optional[str]
    lots: Optional[str]
    constraints: Optional[str]
    base_ccy: str
    portfolio_cash: float
    slippage_bps: float
    fee_bps: float
    half_spread_bps: float
    turnover_cap_pct: float
    cash_min_pct: float
    cash_max_pct: float
    min_trade_usd: float
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Global ETF rebalance planner")
    ap.add_argument("--holdings", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--exposures", default="")
    ap.add_argument("--prices", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--costs", default="")
    ap.add_argument("--lots", default="")
    ap.add_argument("--constraints", default="")
    ap.add_argument("--base_ccy", default="USD")
    ap.add_argument("--portfolio_cash", type=float, default=0.0)
    ap.add_argument("--slippage_bps", type=float, default=5.0)
    ap.add_argument("--fee_bps", type=float, default=1.0)
    ap.add_argument("--half_spread_bps", type=float, default=5.0)
    ap.add_argument("--turnover_cap_pct", type=float, default=35.0)
    ap.add_argument("--cash_min_pct", type=float, default=0.0)
    ap.add_argument("--cash_max_pct", type=float, default=5.0)
    ap.add_argument("--min_trade_usd", type=float, default=50.0)
    ap.add_argument("--outdir", default="out_rebalance")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load data
    HOLD = load_holdings(args.holdings)
    TGT, mode = load_targets(args.targets)
    EXPO = load_exposures(args.exposures) if args.exposures else pd.DataFrame()
    PR   = load_prices(args.prices) if args.prices else pd.DataFrame()
    FX   = load_fx(args.fx) if args.fx else pd.DataFrame()
    COST = load_costs(args.costs) if args.costs else pd.DataFrame()
    LOTS = load_lots(args.lots) if args.lots else pd.DataFrame()
    CNST = load_constraints(args.constraints) if args.constraints else {}

    base_ccy = args.base_ccy.upper()

    # Prices & currencies
    # Merge prices into holdings; if currency mismatches, convert to base via FX
    H = HOLD.copy()
    prices = {}
    currencies = {}
    lotsizes = {}
    for _, r in H.iterrows():
        t = r["ticker"]
        p = r["price"]
        ccy = r.get("currency", np.nan)
        if pd.isna(p) or not np.isfinite(p):
            row = PR[PR["ticker"]==t]
            if not row.empty:
                p = float(row["price"].iloc[0])
                if "currency" in row.columns and pd.notna(row["currency"].iloc[0]):
                    ccy = str(row["currency"].iloc[0]).upper()
        prices[t] = p
        currencies[t] = ccy if isinstance(ccy, str) else base_ccy
        lotsizes[t] = int(r.get("lotsize", 1)) if pd.notna(r.get("lotsize", 1)) else 1

    # Costs map with defaults
    costs_map: Dict[str, Dict[str,float]] = {}
    for t in H["ticker"].unique():
        row = COST[COST["ticker"]==t]
        costs_map[t] = {
            "fee_bps": float(row["fee_bps"].iloc[0]) if not row.empty and pd.notna(row["fee_bps"].iloc[0]) else args.fee_bps,
            "half_spread_bps": float(row["half_spread_bps"].iloc[0]) if not row.empty and pd.notna(row["half_spread_bps"].iloc[0]) else args.half_spread_bps,
            "slippage_bps": float(row["slippage_bps"].iloc[0]) if not row.empty and pd.notna(row["slippage_bps"].iloc[0]) else args.slippage_bps,
            "min_trade_usd": float(row["min_trade_usd"].iloc[0]) if not row.empty and pd.notna(row["min_trade_usd"].iloc[0]) else args.min_trade_usd,
        }
        # Allow overriding lotsize from costs file
        if not row.empty and "lotsize" in row.columns and pd.notna(row["lotsize"].iloc[0]):
            lotsizes[t] = int(row["lotsize"].iloc[0])

    # Convert prices to base currency
    px_base = {}
    for t in H["ticker"].unique():
        pb = apply_fx(prices[t], currencies.get(t, base_ccy), base_ccy, FX)
        if not np.isfinite(pb) or pb<=0:
            raise ValueError(f"Missing/invalid price for {t}. Provide --prices and/or price in holdings.")
        px_base[t] = pb
    px_base = pd.Series(px_base).astype(float)

    # Current shares & MV
    cur_shares = H.set_index("ticker")["shares"].astype(float).reindex(px_base.index).fillna(0.0)
    cur_mv = (cur_shares * px_base).sum()
    cash0 = float(args.portfolio_cash)
    port_mv_total = cur_mv + cash0

    # Constraints & lists
    turnover_cap = float((to_float(CNST.get("turnover_cap_pct", args.turnover_cap_pct)) or args.turnover_cap_pct) / 100.0)
    cash_min = float((to_float(CNST.get("cash_min_pct", args.cash_min_pct)) or args.cash_min_pct) / 100.0) * port_mv_total
    cash_max = float((to_float(CNST.get("cash_max_pct", args.cash_max_pct)) or args.cash_max_pct) / 100.0) * port_mv_total
    allowlist = split_csv_list(CNST.get("allowlist",""))
    buy_black = set(split_csv_list(CNST.get("buy_blacklist","")))
    sell_black = set(split_csv_list(CNST.get("sell_blacklist","")))

    # Determine target ETF weights
    tickers_all = sorted(H["ticker"].unique().tolist())
    if mode == "ticker":
        tw = TGT.set_index("ticker")["target"].reindex(tickers_all).fillna(0.0)
        if tw.sum() <= 0: raise ValueError("Ticker targets sum to zero. Check targets.csv")
        tw = tw / tw.sum()
    else:
        if EXPO.empty:
            raise ValueError("Bucket-level targets provided but --exposures is missing.")
        buckets = sorted(TGT["bucket"].unique().tolist())
        etfs = tickers_all
        # Build exposures matrix (bucket x etf)
        E = EXPO.copy()
        E = E.groupby(["ticker","bucket"], as_index=False)["weight"].sum()
        mat = E.pivot_table(index="bucket", columns="ticker", values="weight", aggfunc="sum").fillna(0.0)
        # Normalize exposures by bucket sums (not strictly required)
        # Solve NNLS-like
        target_b = TGT.set_index("bucket")["target"]
        w = nnls_exposures_solve(etfs, buckets, mat, target_b)
        tw = w.reindex(tickers_all).fillna(0.0)

    # Apply tradeability lists
    if allowlist:
        tw = tw.where(tw.index.isin(allowlist), 0.0)
        if tw.sum() > 0:
            tw = tw / tw.sum()
        else:
            raise ValueError("Allowlist removed all target weights.")

    # Build costs dict for greedy function
    costs_for_fn: Dict[str, Dict[str,float]] = {t: costs_map[t] for t in tickers_all}

    # Greedy round with constraints
    new_shares, cash_after, est_costs, gross_turnover = greedy_round_shares(
        current_shares=cur_shares,
        target_weights=tw,
        prices_base=px_base,
        lotsize=pd.Series(lotsizes),
        cash_base=cash0,
        mv_base=cur_mv,
        costs_df=costs_for_fn,
        min_trade_usd=args.min_trade_usd,
        turnover_cap=turnover_cap,
        cash_bounds=(cash_min, cash_max)
    )

    # Build trades
    trades = (new_shares - cur_shares).rename("delta_shares").to_frame()
    trades["action"] = np.where(trades["delta_shares"]>0, "BUY", np.where(trades["delta_shares"]<0, "SELL", "HOLD"))
    trades["shares"] = trades["delta_shares"].abs().round().astype(int)
    trades["price_base"] = trades.index.map(px_base.to_dict())
    trades["notional_base"] = trades["shares"] * trades["price_base"]
    # Drop zero and micro tickets
    trades = trades[trades["shares"]>0].copy()
    trades = trades[trades["notional_base"]>=args.min_trade_usd].copy()

    # Remove tickets blocked by blacklists
    if buy_black:
        trades = trades[~((trades["action"]=="BUY") & (trades.index.isin(buy_black)))]
    if sell_black:
        trades = trades[~((trades["action"]=="SELL") & (trades.index.isin(sell_black)))]

    # Costs estimate per ticket
    rows = []
    for t, r in trades.iterrows():
        c = costs_map.get(t, {})
        cps = cost_per_share_base(r["price_base"], c.get("fee_bps", args.fee_bps), c.get("half_spread_bps", args.half_spread_bps), c.get("slippage_bps", args.slippage_bps))
        est_cost = cps * r["shares"]
        rows.append(est_cost)
    trades["est_costs"] = rows

    # Realized gains via HIFO for sells (if lots & cost_basis exist)
    realized_gain = []
    for t, r in trades.iterrows():
        if r["action"] != "SELL":
            realized_gain.append(0.0); continue
        picks, _ = choose_sell_lots_hifo(LOTS, t, float(r["shares"]))
        rg = 0.0
        for p in picks:
            rg += (r["price_base"] - float(p.get("cost_basis") or 0.0)) * float(p["sell_shares"])
        realized_gain.append(rg)
    trades["realized_gain_usd"] = realized_gain
    trades = trades.reset_index().rename(columns={"index":"ticker"})

    # Before/After & weights
    after_shares = new_shares
    after_mv = (after_shares * px_base).sum()
    mv_after_total = after_mv + cash_after
    before = pd.DataFrame({
        "ticker": tickers_all,
        "shares_before": cur_shares.reindex(tickers_all).values,
        "px_base": px_base.reindex(tickers_all).values,
    })
    before["mv_before"] = before["shares_before"] * before["px_base"]
    before["w_before"] = before["mv_before"] / (cur_mv + 1e-12)

    after = pd.DataFrame({
        "ticker": tickers_all,
        "shares_after": after_shares.reindex(tickers_all).values,
        "px_base": px_base.reindex(tickers_all).values,
    })
    after["mv_after"] = after["shares_after"] * after["px_base"]
    after["w_after"] = after["mv_after"] / (after_mv + 1e-12)

    ba = before.merge(after[["ticker","shares_after","mv_after","w_after"]], on="ticker", how="outer").fillna(0.0)
    ba["delta_shares"] = ba["shares_after"] - ba["shares_before"]
    ba["delta_mv"] = ba["mv_after"] - ba["mv_before"]
    ba["delta_w"] = ba["w_after"] - ba["w_before"]
    # Attach targets used
    ba["target_w"] = tw.reindex(ba["ticker"]).fillna(0.0).values

    # Exposures achieved vs target (if exposure model used)
    exposures_check = pd.DataFrame()
    if mode == "bucket" and not EXPO.empty:
        expo = EXPO.copy()
        expo["weight"] = expo["weight"].fillna(0.0)
        # Before weights by bucket
        wb = (ba.set_index("ticker")["w_before"].rename("w")
                .reset_index().merge(expo, on="ticker", how="left"))
        wb["bucket_weight"] = wb["w"] * wb["weight"]
        before_bucket = wb.groupby("bucket", as_index=False)["bucket_weight"].sum().rename(columns={"bucket_weight":"w_before"})
        # After
        wa = (ba.set_index("ticker")["w_after"].rename("w")
                .reset_index().merge(expo, on="ticker", how="left"))
        wa["bucket_weight"] = wa["w"] * wa["weight"]
        after_bucket = wa.groupby("bucket", as_index=False)["bucket_weight"].sum().rename(columns={"bucket_weight":"w_after"})
        exposures_check = before_bucket.merge(after_bucket, on="bucket", how="outer").fillna(0.0)
        tgt_b = TGT.set_index("bucket")["target"]
        exposures_check["w_target"] = exposures_check["bucket"].map(tgt_b).fillna(0.0)
        exposures_check["delta_w"] = exposures_check["w_after"] - exposures_check["w_target"]

    # Summaries
    drift_before = float((ba["w_before"] - ba["target_w"]).abs().sum()/2) if tw.sum()>0 else np.nan
    drift_after  = float((ba["w_after"] - ba["target_w"]).abs().sum()/2)  if tw.sum()>0 else np.nan

    summary = {
        "base_ccy": base_ccy,
        "mode": mode,
        "portfolio_mv_before": float(port_mv_total),
        "mv_equity_before": float(cur_mv),
        "cash_before": float(cash0),
        "cash_after": float(cash_after),
        "mv_equity_after": float(after_mv),
        "portfolio_mv_after": float(mv_after_total),
        "gross_turnover_pct": float(gross_turnover*100.0),
        "est_total_costs_usd": float(trades["est_costs"].sum()) if not trades.empty else 0.0,
        "drift_before_L1": drift_before,
        "drift_after_L1": drift_after,
        "num_buys": int((trades["action"]=="BUY").sum()),
        "num_sells": int((trades["action"]=="SELL").sum()),
    }

    # Outputs
    trades.to_csv(outdir / "trades.csv", index=False)
    ba.to_csv(outdir / "before_after.csv", index=False)
    if not exposures_check.empty:
        exposures_check.to_csv(outdir / "exposures_check.csv", index=False)
    pd.DataFrame([{
        "cash_before": cash0,
        "cash_after": cash_after,
        "est_costs_total": trades["est_costs"].sum() if not trades.empty else 0.0,
        "gross_turnover_pct": gross_turnover*100.0
    }]).to_csv(outdir / "cash_fx.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        holdings=args.holdings, targets=args.targets, exposures=(args.exposures or None),
        prices=(args.prices or None), fx=(args.fx or None), costs=(args.costs or None),
        lots=(args.lots or None), constraints=(args.constraints or None),
        base_ccy=args.base_ccy, portfolio_cash=args.portfolio_cash, slippage_bps=args.slippage_bps,
        fee_bps=args.fee_bps, half_spread_bps=args.half_spread_bps, turnover_cap_pct=args.turnover_cap_pct,
        cash_min_pct=args.cash_min_pct, cash_max_pct=args.cash_max_pct, min_trade_usd=args.min_trade_usd,
        outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Global ETF Rebalance ==")
    print(f"Mode: {mode} | MV before: {port_mv_total:,.0f} {base_ccy} | Cash → {cash_after:,.0f}")
    print(f"Turnover: {gross_turnover*100:.2f}% | Est. costs: {summary['est_total_costs_usd']:,.0f} {base_ccy}")
    if not trades.empty:
        print("Sample tickets:")
        for _, r in trades.head(10).iterrows():
            print(f"  {r['action']:>4} {r['ticker']:<8} {int(r['shares']):>8} @ {r['price_base']:>8.2f}  (notional {r['notional_base']:>10.0f})")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
