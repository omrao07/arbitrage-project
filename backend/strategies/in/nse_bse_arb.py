#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nse_bse_arb.py — Cross-venue cash equity arbitrage (NSE ↔ BSE) backtester
--------------------------------------------------------------------------

What this does
==============
Given top-of-book **quotes** (or trades) for Indian equities from **NSE** and **BSE**,
this script detects executable cross-venue price dislocations (after **latency, slippage,
and costs**) and simulates simple pair-wise roundtrips: buy on the cheaper venue,
sell on the richer one, size-limited by available top-of-book liquidity.

It outputs a trade-by-trade P&L, opportunity diagnostics, and summary stats.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--nse nse.csv            REQUIRED  (quotes or trades)
--bse bse.csv            REQUIRED
--symmap map.csv         OPTIONAL  (symbol harmonization, tick sizes)

Expected columns in quotes files (any reasonable variants are okay):
  timestamp, symbol, bid, ask, bid_size, ask_size
The loader will try to find fuzzy matches: time/date/time_ms; scrip/name; b/a; bq/aq; ltp/last for sanity.
If only trades are available (price, side, size), the script can synthesize a mid using last trade; but
you'll get weaker results. Quotes are strongly recommended.

Optional symbol map (map.csv):
  symbol_nse, symbol_bse, symbol, tick_size, lot_size
Only symbol mapping and tick_size (INR) are used if present.

Key CLI params
--------------
--latency_ms 5                    Round-trip latency from signal to both legs (ms)
--slip_ticks 1                    Extra ticks paid beyond top-of-book (both legs)
--min_edge_bps 1.0                Minimum net edge in bps (after fees) to trigger
--qty_mode fixed|topclip          Position sizing mode
--qty 100                         If qty_mode=fixed: shares per leg (capped by size on both sides)
--topclip_frac 0.5                If qty_mode=topclip: use frac × min(bid_size_rich, ask_size_cheap)
--fees_json fees.json             Optional JSON with per-venue fee/tax rates in bps (see template below)
--session 09:15-15:30             Trading session to consider (IST, HH:MM-HH:MM)
--symbols RELIANCE,TCS            Optional allowlist (comma-separated). Default: all overlapping.
--denylist PENNY1,PENNY2          Optional denylist (comma-separated)
--resample_ms 0                   Optional event-time thinning (0=use all events)
--mode book                       Pricing mode: "book" uses bid/ask; "mid" uses mid±half-spread (for diagnostics)
--outdir out_arb                  Output directory

Fees JSON template (bps; applied to notional on each leg)
---------------------------------------------------------
{
  "NSE": {
    "brokerage_bps": 0.5,
    "exchange_bps": 0.0032,
    "clearing_bps": 0.003,
    "stt_bps_buy": 0.0,
    "stt_bps_sell": 10.0,
    "stamp_bps_buy": 1.5,
    "stamp_bps_sell": 0.0,
    "gst_bps": 0.9
  },
  "BSE": { ... same keys ... }
}
(All fields optional; defaults are conservative placeholders. Use your broker’s real schedule.)

Outputs
-------
- opportunities.csv     All times with computed cross-venue edges & flags
- fills.csv             Executed roundtrips with qty, prices, fees, P&L
- pnl_timeseries.csv    Cumulative P&L by time
- summary.json          Headline metrics & configuration echo
- config.json           Run configuration for reproducibility

Notes & caveats
---------------
• Backtest assumes you can **short-sell intraday** to complete the pair (you open buy/sell near-simultaneously).
  If your venue/broker disallows one side, set min_edge_bps high and/or filter symbols accordingly.
• Execution uses **future** book snapshots at t+latency_ms to avoid look-ahead bias.
• Slippage modeled as extra ticks beyond the touch on **both legs**.
• Taxes/fees are parameterized; defaults are NOT advice—customize.
• This is research tooling; validate with your own data and controls before live use.

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- utilities -----------------------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    """Return first matching column via exact/lower/contains."""
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

def to_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize("Asia/Kolkata", nonexistent="NaT", ambiguous="NaT").tz_convert("Asia/Kolkata")

def parse_session(sess: str) -> Tuple[pd.Timedelta, pd.Timedelta]:
    # "09:15-15:30"
    a, b = sess.split("-")
    hh, mm = map(int, a.split(":"))
    s_hh, s_mm = map(int, b.split(":"))
    return pd.Timedelta(hours=hh, minutes=mm), pd.Timedelta(hours=s_hh, minutes=s_mm)

def clip_session(ts: pd.Series, session: str) -> pd.Series:
    start, end = parse_session(session)
    local = ts.dt.tz_convert("Asia/Kolkata")
    d0 = local.dt.normalize()
    return ts[(local - d0 >= start) & (local - d0 <= end)]

def guess_tick(price: float) -> float:
    # Simple SEBI tick bands are mostly 0.05 for liquid scrips, but can vary.
    # Use 0.05 unless price < 10 (use 0.01) or > 1000 (still 0.05).
    if price < 10: return 0.01
    return 0.05

def bps(x: float) -> float:
    return x / 10000.0

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ----------------------------- loaders -----------------------------

@dataclass
class VenueQuotes:
    venue: str
    df: pd.DataFrame  # columns: ts (tz-aware), symbol, bid, ask, bsz, asz

def load_quotes(path: str, venue: str, session: str) -> VenueQuotes:
    df = pd.read_csv(path)
    tcol = ncol(df, "timestamp","time","date_time","datetime","ts","t")
    scol = ncol(df, "symbol","scrip","name","ticker","isin")
    bidc = ncol(df, "bid","best_bid","b1","bb")
    askc = ncol(df, "ask","best_ask","a1","ba","offer")
    bqc  = ncol(df, "bid_size","bsize","bidqty","bq","bb_qty")
    aqc  = ncol(df, "ask_size","asize","askqty","aq","ba_qty")
    ltp  = ncol(df, "last","ltp","trade_price","price")

    if not tcol or not scol:
        raise ValueError(f"{venue}: need at least timestamp and symbol columns.")
    df = df.rename(columns={tcol:"timestamp", scol:"symbol"})
    df["timestamp"] = to_time(df["timestamp"])

    # basic cleaning
    if bidc and askc:
        df = df.rename(columns={bidc:"bid", askc:"ask"})
    elif ltp:
        # synthesize a spread around last trade (fall back)
        df = df.rename(columns={ltp:"last"})
        df["bid"] = safe_num(df["last"]) * (1 - 0.0005)  # 5 bps synthetic half-spread
        df["ask"] = safe_num(df["last"]) * (1 + 0.0005)
    else:
        raise ValueError(f"{venue}: cannot find bid/ask nor last trade to synthesize.")

    if bqc: df = df.rename(columns={bqc:"bid_size"})
    if aqc: df = df.rename(columns={aqc:"ask_size"})
    if "bid_size" not in df.columns: df["bid_size"] = np.nan
    if "ask_size" not in df.columns: df["ask_size"] = np.nan

    # sort & session filter
    df = df[["timestamp","symbol","bid","ask","bid_size","ask_size"]].copy()
    df = df.dropna(subset=["timestamp","symbol"]).sort_values(["symbol","timestamp"])
    # retain session
    mask = df["timestamp"].pipe(clip_session, session)
    df = df.loc[mask.index]  # clip_session returns filtered index already tz-aware range
    return VenueQuotes(venue=venue.upper(), df=df.reset_index(drop=True))


def load_symmap(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    m = pd.read_csv(path)
    nse = ncol(m, "symbol_nse","nse","nse_symbol")
    bse = ncol(m, "symbol_bse","bse","bse_symbol","bse_code")
    uni = ncol(m, "symbol","unified","ticker")
    tick = ncol(m, "tick_size","tick","tick_inr")
    lot  = ncol(m, "lot_size","lot","lotshares")
    cols = {}
    if nse: cols[nse] = "symbol_nse"
    if bse: cols[bse] = "symbol_bse"
    if uni: cols[uni] = "symbol"
    if tick:cols[tick] = "tick_size"
    if lot: cols[lot]  = "lot_size"
    m = m.rename(columns=cols)
    keep = ["symbol","symbol_nse","symbol_bse","tick_size","lot_size"]
    for c in keep:
        if c not in m.columns: m[c] = np.nan
    return m[keep]


# ----------------------------- costs & execution -----------------------------

@dataclass
class FeeSchedule:
    brokerage_bps: float = 0.5
    exchange_bps: float = 0.0032
    clearing_bps: float = 0.003
    stt_bps_buy: float = 0.0
    stt_bps_sell: float = 10.0
    stamp_bps_buy: float = 1.5
    stamp_bps_sell: float = 0.0
    gst_bps: float = 0.9

    def total_bps(self, side: str) -> float:
        # side: "buy" or "sell"
        x = self.brokerage_bps + self.exchange_bps + self.clearing_bps + self.gst_bps
        if side == "buy":
            x += self.stamp_bps_buy + self.stt_bps_buy
        else:
            x += self.stamp_bps_sell + self.stt_bps_sell
        return x

def load_fees(path: Optional[str]) -> Dict[str, FeeSchedule]:
    # default same for both if no file
    default = {"NSE": FeeSchedule(), "BSE": FeeSchedule()}
    if not path: return default
    raw = json.loads(Path(path).read_text())
    out = {}
    for v in ["NSE","BSE"]:
        cfg = raw.get(v, {})
        out[v] = FeeSchedule(**{k: float(cfg.get(k, getattr(FeeSchedule, k, 0.0))) for k in FeeSchedule().__dict__.keys()})
    return out


# ----------------------------- core construction -----------------------------

def harmonize_symbols(nse: pd.DataFrame, bse: pd.DataFrame, symmap: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str,float]]:
    """
    Return (nse_df, bse_df, tick_map) with unified 'symbol' names and per-symbol tick_size if known.
    """
    if not symmap.empty and "symbol" in symmap.columns:
        # build mapping dicts
        map_n = {r["symbol_nse"]: r["symbol"] for _, r in symmap.dropna(subset=["symbol_nse","symbol"]).iterrows()}
        map_b = {r["symbol_bse"]: r["symbol"] for _, r in symmap.dropna(subset=["symbol_bse","symbol"]).iterrows()}
        nse["symbol"] = nse["symbol"].map(lambda s: map_n.get(s, s))
        bse["symbol"] = bse["symbol"].map(lambda s: map_b.get(s, s))
        tick_map = {r["symbol"]: float(r["tick_size"]) for _, r in symmap.dropna(subset=["symbol","tick_size"]).iterrows()}
    else:
        tick_map = {}
    return nse, bse, tick_map

def build_event_grid(nse: pd.DataFrame, bse: pd.DataFrame, symbols: Optional[List[str]], resample_ms: int) -> pd.DataFrame:
    """
    Create a unified timeline ('timestamp','symbol') for overlapping symbols.
    Optionally thin to a regular grid (resample_ms > 0).
    """
    s_n = set(nse["symbol"].unique())
    s_b = set(bse["symbol"].unique())
    syms = sorted(list((s_n & s_b) if not symbols else (set(symbols) & s_n & s_b)))
    if not syms:
        raise ValueError("No overlapping symbols between NSE and BSE after filters.")
    # subset early for speed
    nse = nse[nse["symbol"].isin(syms)]
    bse = bse[bse["symbol"].isin(syms)]

    if resample_ms and resample_ms > 0:
        # downsample each venue per symbol to regular ms grid, carrying forward quotes
        def down(df):
            out = []
            for sym, g in df.groupby("symbol"):
                g = g.set_index("timestamp").sort_index()
                rng = pd.date_range(g.index.min(), g.index.max(), freq=f"{resample_ms}ms", tz="Asia/Kolkata")
                r = g[["bid","ask","bid_size","ask_size"]].resample(f"{resample_ms}ms").last().reindex(rng).ffill()
                r["symbol"] = sym
                r = r.reset_index().rename(columns={"index":"timestamp"})
                out.append(r)
            return pd.concat(out, ignore_index=True)
        nse = down(nse); bse = down(bse)

    # union of times per symbol
    grid = (pd.concat([nse[["timestamp","symbol"]], bse[["timestamp","symbol"]]], ignore_index=True)
              .drop_duplicates()
              .sort_values(["symbol","timestamp"])
              .reset_index(drop=True))
    return grid, nse, bse, syms

def asof_join(left: pd.DataFrame, right: pd.DataFrame, on: str="timestamp", by: str="symbol") -> pd.DataFrame:
    return pd.merge_asof(left.sort_values([by,on]),
                         right.sort_values([by,on]),
                         on=on, by=by, direction="backward")


# ----------------------------- opportunity detection -----------------------------

def effective_exec(px: float, side: str, ticks: int, tick_size: float) -> float:
    if not np.isfinite(tick_size) or tick_size <= 0:
        tick_size = guess_tick(px)
    slip = ticks * tick_size
    return px + slip if side == "buy" else px - slip

def compute_edges(grid: pd.DataFrame,
                  nse: pd.DataFrame, bse: pd.DataFrame,
                  latency_ms: int, slip_ticks: int,
                  tick_map: Dict[str,float],
                  fees: Dict[str, FeeSchedule],
                  mode: str="book") -> pd.DataFrame:
    """
    Build a panel with asof-merged **signal** quotes at t, and **exec** quotes at t+latency.
    Compute two directed edges: buy NSE→sell BSE and buy BSE→sell NSE.
    """
    # Signal quotes at t
    g = asof_join(grid, nse.rename(columns=lambda c: f"nse_{c}" if c not in ["timestamp","symbol"] else c))
    g = asof_join(g, bse.rename(columns=lambda c: f"bse_{c}" if c not in ["timestamp","symbol"] else c))

    # Exec quotes at t + latency
    lat = pd.Timedelta(milliseconds=max(0, latency_ms))
    g["t_exec"] = g["timestamp"] + lat
    nse_e = nse.rename(columns={"timestamp":"t_exec",
                                "bid":"nse_exec_bid","ask":"nse_exec_ask",
                                "bid_size":"nse_exec_bid_size","ask_size":"nse_exec_ask_size"})
    bse_e = bse.rename(columns={"timestamp":"t_exec",
                                "bid":"bse_exec_bid","ask":"bse_exec_ask",
                                "bid_size":"bse_exec_bid_size","ask_size":"bse_exec_ask_size"})
    g = asof_join(g, nse_e, on="t_exec")
    g = asof_join(g, bse_e, on="t_exec")

    # Mid prices (diagnostics)
    for pre in ["nse","bse"]:
        g[f"{pre}_mid"] = (g[f"{pre}_bid"] + g[f"{pre}_ask"]) / 2.0
        g[f"{pre}_exec_mid"] = (g[f"{pre}_exec_bid"] + g[f"{pre}_exec_ask"]) / 2.0

    # Directed opportunities
    out_rows = []
    # vectorized computations
    for idx, r in g.iterrows():
        sym = r["symbol"]
        tsize = tick_map.get(sym, np.nan)

        # Skip if either venue missing at signal OR exec time
        valid_a = np.isfinite(r.get("nse_exec_ask", np.nan)) and np.isfinite(r.get("bse_exec_bid", np.nan))
        valid_b = np.isfinite(r.get("bse_exec_ask", np.nan)) and np.isfinite(r.get("nse_exec_bid", np.nan))
        if not (valid_a or valid_b):
            continue

        # Mode pricing basis
        if mode == "mid":
            nse_buy_px  = r["nse_exec_mid"]
            nse_sell_px = r["nse_exec_mid"]
            bse_buy_px  = r["bse_exec_mid"]
            bse_sell_px = r["bse_exec_mid"]
            nse_buy_sz = r.get("nse_exec_ask_size", np.nan)
            bse_buy_sz = r.get("bse_exec_ask_size", np.nan)
            nse_sell_sz = r.get("nse_exec_bid_size", np.nan)
            bse_sell_sz = r.get("bse_exec_bid_size", np.nan)
        else:
            nse_buy_px  = r["nse_exec_ask"]
            nse_sell_px = r["nse_exec_bid"]
            bse_buy_px  = r["bse_exec_ask"]
            bse_sell_px = r["bse_exec_bid"]
            nse_buy_sz = r.get("nse_exec_ask_size", np.nan)
            bse_buy_sz = r.get("bse_exec_ask_size", np.nan)
            nse_sell_sz = r.get("nse_exec_bid_size", np.nan)
            bse_sell_sz = r.get("bse_exec_bid_size", np.nan)

        # Apply slippage ticks
        nse_buy_eff  = effective_exec(nse_buy_px,  "buy",  slip_ticks, tsize) if np.isfinite(nse_buy_px) else np.nan
        nse_sell_eff = effective_exec(nse_sell_px, "sell", slip_ticks, tsize) if np.isfinite(nse_sell_px) else np.nan
        bse_buy_eff  = effective_exec(bse_buy_px,  "buy",  slip_ticks, tsize) if np.isfinite(bse_buy_px) else np.nan
        bse_sell_eff = effective_exec(bse_sell_px, "sell", slip_ticks, tsize) if np.isfinite(bse_sell_px) else np.nan

        # Costs (bps of notional)
        fN = fees["NSE"]; fB = fees["BSE"]

        # Direction A: Buy NSE, Sell BSE
        if np.isfinite(nse_buy_eff) and np.isfinite(bse_sell_eff):
            gross_edge = (bse_sell_eff - nse_buy_eff)
            mid_basis  = np.nanmean([r.get("nse_exec_mid", np.nan), r.get("bse_exec_mid", np.nan)])
            denom = nse_buy_eff if (nse_buy_eff and nse_buy_eff>0) else mid_basis
            edge_bps = (gross_edge / denom) * 1e4 if denom and np.isfinite(denom) else np.nan
            # fees in price terms
            fee_buy  = nse_buy_eff  * bps(fN.total_bps("buy"))
            fee_sell = bse_sell_eff * bps(fB.total_bps("sell"))
            net_edge = gross_edge - (fee_buy + fee_sell)
            net_bps  = (net_edge / denom) * 1e4 if denom and np.isfinite(denom) else np.nan
            out_rows.append({
                "timestamp": r["timestamp"], "symbol": sym, "dir": "BUY_NSE_SELL_BSE",
                "buy_px": nse_buy_eff, "sell_px": bse_sell_eff,
                "buy_sz": nse_buy_sz, "sell_sz": bse_sell_sz,
                "gross_edge_inr": gross_edge, "net_edge_inr": net_edge,
                "gross_edge_bps": edge_bps, "net_edge_bps": net_bps
            })

        # Direction B: Buy BSE, Sell NSE
        if np.isfinite(bse_buy_eff) and np.isfinite(nse_sell_eff):
            gross_edge = (nse_sell_eff - bse_buy_eff)
            mid_basis  = np.nanmean([r.get("nse_exec_mid", np.nan), r.get("bse_exec_mid", np.nan)])
            denom = bse_buy_eff if (bse_buy_eff and bse_buy_eff>0) else mid_basis
            edge_bps = (gross_edge / denom) * 1e4 if denom and np.isfinite(denom) else np.nan
            fee_buy  = bse_buy_eff  * bps(fB.total_bps("buy"))
            fee_sell = nse_sell_eff * bps(fN.total_bps("sell"))
            net_edge = gross_edge - (fee_buy + fee_sell)
            net_bps  = (net_edge / denom) * 1e4 if denom and np.isfinite(denom) else np.nan
            out_rows.append({
                "timestamp": r["timestamp"], "symbol": sym, "dir": "BUY_BSE_SELL_NSE",
                "buy_px": bse_buy_eff, "sell_px": nse_sell_eff,
                "buy_sz": bse_buy_sz, "sell_sz": nse_sell_sz,
                "gross_edge_inr": gross_edge, "net_edge_inr": net_edge,
                "gross_edge_bps": edge_bps, "net_edge_bps": net_bps
            })

    if not out_rows:
        return pd.DataFrame(columns=["timestamp","symbol","dir","buy_px","sell_px","buy_sz","sell_sz",
                                     "gross_edge_inr","net_edge_inr","gross_edge_bps","net_edge_bps"])
    opp = pd.DataFrame(out_rows).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    return opp


# ----------------------------- execution simulator -----------------------------

def choose_qty(row, qty_mode: str, qty: int, topclip_frac: float) -> int:
    buy_sz  = row.get("buy_sz", np.nan)
    sell_sz = row.get("sell_sz", np.nan)
    if qty_mode == "fixed":
        cap = np.nanmin([buy_sz, sell_sz])
        if not np.isfinite(cap): cap = qty
        return int(max(0, min(qty, cap)))
    else:
        # topclip: fraction of available contra sizes
        base = np.nanmin([buy_sz, sell_sz])
        if not np.isfinite(base): base = qty
        return int(max(0, np.floor(base * max(0.0, min(1.0, topclip_frac)))))

def simulate(opp: pd.DataFrame, min_edge_bps: float,
             qty_mode: str, qty: int, topclip_frac: float,
             fees: Dict[str, FeeSchedule]) -> pd.DataFrame:
    """
    Take opportunities with net_edge_bps >= threshold and size-cap by top-of-book.
    Record per-trade notional, fees (by venue/side), and P&L.
    """
    if opp.empty:
        return pd.DataFrame(columns=["timestamp","symbol","dir","qty","buy_px","sell_px",
                                     "notional_buy","notional_sell","fees_buy","fees_sell","pnl_inr","net_edge_bps"])
    trades = []
    for _, r in opp.iterrows():
        if not np.isfinite(r["net_edge_bps"]) or r["net_edge_bps"] < min_edge_bps:
            continue
        q = choose_qty(r, qty_mode, qty, topclip_frac)
        if q <= 0:
            continue
        notional_buy  = r["buy_px"]  * q
        notional_sell = r["sell_px"] * q
        # venue fees
        if r["dir"].startswith("BUY_NSE"):
            f_buy  = notional_buy  * bps(fees["NSE"].total_bps("buy"))
            f_sell = notional_sell * bps(fees["BSE"].total_bps("sell"))
        else:
            f_buy  = notional_buy  * bps(fees["BSE"].total_bps("buy"))
            f_sell = notional_sell * bps(fees["NSE"].total_bps("sell"))
        pnl = (notional_sell - notional_buy) - (f_buy + f_sell)
        trades.append({
            "timestamp": r["timestamp"], "symbol": r["symbol"], "dir": r["dir"],
            "qty": int(q),
            "buy_px": float(r["buy_px"]), "sell_px": float(r["sell_px"]),
            "notional_buy": float(notional_buy), "notional_sell": float(notional_sell),
            "fees_buy": float(f_buy), "fees_sell": float(f_sell),
            "pnl_inr": float(pnl),
            "net_edge_bps": float(r["net_edge_bps"])
        })
    return pd.DataFrame(trades).sort_values(["symbol","timestamp"]).reset_index(drop=True)


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    nse: str
    bse: str
    symmap: Optional[str]
    latency_ms: int
    slip_ticks: int
    min_edge_bps: float
    qty_mode: str
    qty: int
    topclip_frac: float
    fees_json: Optional[str]
    session: str
    symbols: Optional[List[str]]
    denylist: Optional[List[str]]
    resample_ms: int
    mode: str
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="NSE↔BSE cash equity arbitrage backtester")
    ap.add_argument("--nse", required=True, help="NSE quotes CSV")
    ap.add_argument("--bse", required=True, help="BSE quotes CSV")
    ap.add_argument("--symmap", default="", help="Symbol map CSV (optional)")
    ap.add_argument("--latency_ms", type=int, default=5)
    ap.add_argument("--slip_ticks", type=int, default=1)
    ap.add_argument("--min_edge_bps", type=float, default=1.0)
    ap.add_argument("--qty_mode", choices=["fixed","topclip"], default="fixed")
    ap.add_argument("--qty", type=int, default=100)
    ap.add_argument("--topclip_frac", type=float, default=0.5)
    ap.add_argument("--fees_json", default="", help="Fees JSON path (optional)")
    ap.add_argument("--session", default="09:15-15:30")
    ap.add_argument("--symbols", default="", help="Comma-separated allowlist (optional)")
    ap.add_argument("--denylist", default="", help="Comma-separated denylist (optional)")
    ap.add_argument("--resample_ms", type=int, default=0, help="Event thinning to N ms (0=off)")
    ap.add_argument("--mode", choices=["book","mid"], default="book")
    ap.add_argument("--outdir", default="out_arb")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load data
    NSE = load_quotes(args.nse, "NSE", args.session)
    BSE = load_quotes(args.bse, "BSE", args.session)
    SYM = load_symmap(args.symmap) if args.symmap else pd.DataFrame()

    # Harmonize
    nse = NSE.df.copy(); bse = BSE.df.copy()
    nse, bse, tick_map = harmonize_symbols(nse, bse, SYM)

    # Symbol filters
    allow = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    deny  = set([s.strip() for s in args.denylist.split(",") if s.strip()]) if args.denylist else set()

    grid, nse, bse, syms = build_event_grid(nse, bse, allow, args.resample_ms)
    syms = [s for s in syms if s not in deny]
    if not syms:
        raise ValueError("All symbols filtered out by denylist.")
    grid = grid[grid["symbol"].isin(syms)]

    # Fees
    fees = load_fees(args.fees_json)

    # Compute edges
    OPP = compute_edges(grid, nse, bse,
                        latency_ms=int(args.latency_ms),
                        slip_ticks=int(args.slip_ticks),
                        tick_map=tick_map,
                        fees=fees,
                        mode=args.mode)

    if OPP.empty:
        # Persist empty artifacts for reproducibility
        OPP.to_csv(outdir/"opportunities.csv", index=False)
        pd.DataFrame(columns=["timestamp","pnl_inr","cum_pnl"]).to_csv(outdir/"pnl_timeseries.csv", index=False)
        summary = {
            "rows": 0, "symbols": syms, "latency_ms": args.latency_ms, "slip_ticks": args.slip_ticks,
            "min_edge_bps": args.min_edge_bps, "note": "No overlapping quotes/opportunities after alignment."
        }
        (outdir/"summary.json").write_text(json.dumps(summary, indent=2))
        (outdir/"config.json").write_text(json.dumps(asdict(Config(
            nse=args.nse, bse=args.bse, symmap=(args.symmap or None),
            latency_ms=int(args.latency_ms), slip_ticks=int(args.slip_ticks),
            min_edge_bps=float(args.min_edge_bps), qty_mode=args.qty_mode, qty=int(args.qty),
            topclip_frac=float(args.topclip_frac), fees_json=(args.fees_json or None),
            session=args.session, symbols=(allow or None), denylist=(list(deny) or None),
            resample_ms=int(args.resample_ms), mode=args.mode, outdir=args.outdir
        )), indent=2))
        print("No opportunities found after alignment; artifacts written.")
        return

    OPP.to_csv(outdir/"opportunities.csv", index=False)

    # Simulate fills
    FILLS = simulate(OPP, min_edge_bps=float(args.min_edge_bps),
                     qty_mode=args.qty_mode, qty=int(args.qty),
                     topclip_frac=float(args.topclip_frac),
                     fees=fees)
    if not FILLS.empty:
        FILLS.to_csv(outdir/"fills.csv", index=False)
        pnl = (FILLS[["timestamp","pnl_inr"]]
               .sort_values("timestamp")
               .assign(cum_pnl=lambda x: x["pnl_inr"].cumsum()))
        pnl.to_csv(outdir/"pnl_timeseries.csv", index=False)
    else:
        pd.DataFrame(columns=["timestamp","pnl_inr","cum_pnl"]).to_csv(outdir/"pnl_timeseries.csv", index=False)

    # Summary
    sym_stats = (FILLS.groupby("symbol")["pnl_inr"].sum().sort_values(ascending=False).to_dict()
                 if not FILLS.empty else {})
    dir_stats = (FILLS.groupby("dir")["pnl_inr"].sum().to_dict()
                 if not FILLS.empty else {})
    summary = {
        "rows": int(len(OPP)),
        "symbols": syms,
        "opportunities_ge_min_edge": int((OPP["net_edge_bps"] >= float(args.min_edge_bps)).sum()),
        "trades": int(len(FILLS)),
        "gross_pnl_inr": float(FILLS["pnl_inr"].sum()) if not FILLS.empty else 0.0,
        "avg_pnl_per_trade_inr": float(FILLS["pnl_inr"].mean()) if not FILLS.empty else 0.0,
        "median_net_edge_bps_traded": float(FILLS["net_edge_bps"].median()) if not FILLS.empty else None,
        "pnl_by_symbol": sym_stats,
        "pnl_by_direction": dir_stats,
        "latency_ms": int(args.latency_ms),
        "slip_ticks": int(args.slip_ticks),
        "min_edge_bps": float(args.min_edge_bps),
        "qty_mode": args.qty_mode,
        "qty": int(args.qty),
        "topclip_frac": float(args.topclip_frac),
        "mode": args.mode
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        nse=args.nse, bse=args.bse, symmap=(args.symmap or None),
        latency_ms=int(args.latency_ms), slip_ticks=int(args.slip_ticks),
        min_edge_bps=float(args.min_edge_bps), qty_mode=args.qty_mode, qty=int(args.qty),
        topclip_frac=float(args.topclip_frac), fees_json=(args.fees_json or None),
        session=args.session, symbols=(allow or None), denylist=(list(deny) or None),
        resample_ms=int(args.resample_ms), mode=args.mode, outdir=args.outdir
    ))
    (outdir/"config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== NSE ↔ BSE Arbitrage Backtest ==")
    print(f"Events: {len(OPP):,}  | Trades: {len(FILLS):,}  | Gross P&L: ₹{summary['gross_pnl_inr']:.2f}")
    if sym_stats:
        top = sorted(sym_stats.items(), key=lambda kv: -kv[1])[:5]
        print("Top symbols by P&L:", ", ".join([f"{s}:₹{p:.0f}" for s,p in top]))
    if dir_stats:
        print("P&L by direction:", ", ".join([f"{k}:{v:+.0f}" for k,v in dir_stats.items()]))
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
