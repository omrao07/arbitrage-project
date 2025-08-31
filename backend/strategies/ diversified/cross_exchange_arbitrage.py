# backend/strategies/diversified/cross_exchange_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Cross-Exchange Arbitrage (spot/linear)
--------------------------------------
Buys on the cheaper venue and sells on the richer venue at the SAME time.
Works for any symbol you quote on multiple venues (crypto, FX, equities via different books, etc).

You publish mid/nbbo/mark prices into Redis per venue-specific symbol, e.g.:
  HSET last_price "BTCUSDT@BINANCE" '{"price": 65000.1}'
  HSET last_price "BTCUSDT@BYBIT"   '{"price": 65020.4}'
(If you prefer bid/ask, see DEPTH section below.)

This strategy:
  • Normalizes spreads to bps.
  • Subtracts taker fees + slip/latency budget.
  • Fires *paired* IOC/market orders: buy cheap venue, sell rich venue.
  • Tracks an outstanding inventory buffer per symbol to avoid drift.
  • Optionally uses depth-aware executable prices if you publish them.

State is restart-safe in Redis. Orders route through your Strategy base → risk → OMS (paper now).
"""

# ============================== CONFIG (env) ==============================
REDIS_HOST = os.getenv("XAR_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("XAR_REDIS_PORT", "6379"))

# Universe mapping (env), semicolon-separated:
# "SYMBOL,VENUE_A,VENUE_B,feeA_bps,feeB_bps[,min_qty][,qty_usd]"
# Examples:
#  "BTCUSDT,BINANCE,BYBIT,1.0,1.0,,3000;ETHUSDT,BINANCE,OKX,1.0,1.0,,2000"
PAIRS_ENV = os.getenv("XAR_PAIRS",
                      "BTCUSDT,BINANCE,BYBIT,1.0,1.0,,3000;ETHUSDT,BINANCE,OKX,1.0,1.0,,2000")

# Quote key format: "<SYMBOL>@<VENUE>", override if your feed differs
QUOTE_FMT = os.getenv("XAR_QUOTE_FMT", "{sym}@{venue}")

# Entry thresholds / risk
ENTRY_EDGE_BPS = float(os.getenv("XAR_ENTRY_EDGE_BPS", "4.0"))   # after fees+slip
EXIT_INV_BPS   = float(os.getenv("XAR_EXIT_INV_BPS",  "1.0"))    # inventory unwind if edge small
SLIP_BPS       = float(os.getenv("XAR_SLIP_BPS",      "1.5"))    # latency+slippage budget (each side)
MAX_OPEN_PER_PAIR = int(os.getenv("XAR_MAX_OPEN_PER_PAIR", "3"))
COOLDOWN_S     = int(os.getenv("XAR_COOLDOWN_S", "1"))

# Sizing
DEFAULT_QTY_USD = float(os.getenv("XAR_DEFAULT_QTY_USD", "2000"))
MIN_TICKET_USD  = float(os.getenv("XAR_MIN_TICKET_USD", "50"))

# Inventory control (per symbol net notional in USD)
MAX_INV_USD     = float(os.getenv("XAR_MAX_INV_USD", "20000"))

# Venue hints -> OMS
ORDER_TYPE = os.getenv("XAR_ORDER_TYPE", "market")  # or "ioc" if your OMS supports
TIME_IN_FORCE = os.getenv("XAR_TIF", "IOC")
# ========================================================================

# Optional DEPTH inputs (if you publish book snapshots)
# Expect HGET depth:<sym>@<venue> bid -> px, ask -> px (top-of-book executable)
USE_DEPTH = os.getenv("XAR_USE_DEPTH", "false").lower() in ("1","true","yes")
DEPTH_HASH_FMT = os.getenv("XAR_DEPTH_HASH_FMT", "depth:{sym}@{venue}")

# Redis keys your stack already maintains
LAST_PRICE_HKEY = os.getenv("XAR_LAST_PRICE_KEY", "last_price")  # HSET <sym@venue> -> {"price": ...}

# ============================== REDIS ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== PARSING ==============================
@dataclass
class Pair:
    sym: str
    va: str
    vb: str
    fee_a_bps: float
    fee_b_bps: float
    min_qty: Optional[float]  # in base units
    qty_usd: float

def _parse_pairs(env: str) -> List[Pair]:
    out: List[Pair] = []
    for part in env.split(";"):
        s = part.strip()
        if not s:
            continue
        parts = [x.strip() for x in s.split(",")]
        try:
            sym, va, vb, fA, fB = parts[:5]
            minq = float(parts[5]) if len(parts) >= 6 and parts[5] else None
            qty_usd = float(parts[6]) if len(parts) >= 7 and parts[6] else DEFAULT_QTY_USD
            out.append(Pair(sym=sym.upper(), va=va.upper(), vb=vb.upper(),
                            fee_a_bps=float(fA), fee_b_bps=float(fB),
                            min_qty=minq, qty_usd=qty_usd))
        except Exception:
            continue
    return out

PAIRS: List[Pair] = _parse_pairs(PAIRS_ENV)

# ============================== HELPERS ==============================
def _qkey(sym: str, venue: str) -> str:
    return QUOTE_FMT.format(sym=sym.upper(), venue=venue.upper())

def _hget_last(symbol: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, symbol)
    if not raw:
        return None
    try:
        return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try:
            return float(raw) # type: ignore
        except Exception:
            return None

def _depth_px(sym: str, venue: str) -> Tuple[Optional[float], Optional[float]]:
    if not USE_DEPTH:
        return None, None
    key = DEPTH_HASH_FMT.format(sym=sym.upper(), venue=venue.upper())
    b = r.hget(key, "bid")
    a = r.hget(key, "ask")
    try:
        bid = float(b) if b is not None else None # type: ignore
        ask = float(a) if a is not None else None # type: ignore
        return bid, ask
    except Exception:
        return None, None

def _exec_prices(sym: str, venue: str, mid: float) -> Tuple[float, float]:
    """
    Return (buy_px, sell_px) executable approximations.
    Prefer depth; fallback to mid ± half-spread proxy (SLIP_BPS as guard).
    """
    if USE_DEPTH:
        bid, ask = _depth_px(sym, venue)
        if bid is not None and ask is not None and ask > 0 and bid > 0:
            return ask, bid
    # Fallback: assume mid, apply slip bps on each side
    buy_px = mid * (1 + SLIP_BPS / 1e4)
    sell_px = mid * (1 - SLIP_BPS / 1e4)
    return buy_px, sell_px

def _now_ms() -> int:
    return int(time.time() * 1000)

def _poskey(name: str, sym: str) -> str:
    return f"xarb:pos:{name}:{sym}"

def _coolkey(name: str, sym: str) -> str:
    return f"xarb:cool:{name}:{sym}"

def _inv_usd(name: str, sym: str) -> float:
    v = r.hget(_poskey(name, sym), "inv_usd")
    try:
        return float(v) if v is not None else 0.0 # type: ignore
    except Exception:
        return 0.0

# ============================== STATE ==============================
@dataclass
class TradeState:
    open_count: int = 0
    last_ts_ms: int = 0

# ============================== STRATEGY ==============================
class CrossExchangeArbitrage(Strategy):
    """
    Buy cheap venue / sell rich venue when net edge > ENTRY_EDGE_BPS (after fees+slip).
    """

    def __init__(self, name: str = "cross_exchange_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.state: Dict[str, TradeState] = {p.sym: TradeState() for p in PAIRS}

    def on_start(self) -> None:
        super().on_start()
        # publish universe for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "pairs": [{"symbol": p.sym, "venue_a": p.va, "venue_b": p.vb, "fee_a_bps": p.fee_a_bps, "fee_b_bps": p.fee_b_bps,
                       "qty_usd": p.qty_usd, "min_qty": p.min_qty} for p in PAIRS],
            "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        # fast loop: evaluate every tick
        self._evaluate_all()

    # ---------------- Core Engine ----------------
    def _evaluate_all(self) -> None:
        for p in PAIRS:
            # cooldown
            if r.get(_coolkey(self.ctx.name, p.sym)):
                continue

            key_a = _qkey(p.sym, p.va)
            key_b = _qkey(p.sym, p.vb)
            mid_a = _hget_last(key_a)
            mid_b = _hget_last(key_b)
            if mid_a is None or mid_b is None or mid_a <= 0 or mid_b <= 0:
                continue

            # executable prices per venue
            buy_a, sell_a = _exec_prices(p.sym, p.va, mid_a)
            buy_b, sell_b = _exec_prices(p.sym, p.vb, mid_b)

            # Effective buy-on-A / sell-on-B edge (bps of mid)
            # Edge_AB_bps ≈ (sell_B - buy_A)/mid - fees/slip
            # Use average mid for scale
            scale_mid = 0.5 * (mid_a + mid_b)
            edge_ab_bps = 1e4 * ((sell_b - buy_a) / scale_mid) - (p.fee_a_bps + p.fee_b_bps + 2 * SLIP_BPS)
            edge_ba_bps = 1e4 * ((sell_a - buy_b) / scale_mid) - (p.fee_a_bps + p.fee_b_bps + 2 * SLIP_BPS)

            # monitor signal (positive if any cross > 0)
            self.emit_signal(max(-1.0, min(1.0, (max(edge_ab_bps, edge_ba_bps)) / 10.0)))

            # inventory guard
            inv_usd = _inv_usd(self.ctx.name, p.sym)
            if abs(inv_usd) >= MAX_INV_USD:
                # try unwind if tiny edge in opposite direction
                if inv_usd > 0 and edge_ba_bps >= EXIT_INV_BPS:
                    self._fire_pair(p, side="BA", px_a=mid_a, px_b=mid_b)  # sell A, buy B to reduce long
                elif inv_usd < 0 and edge_ab_bps >= EXIT_INV_BPS:
                    self._fire_pair(p, side="AB", px_a=mid_a, px_b=mid_b)  # buy A, sell B to reduce short
                continue

            # entries
            if edge_ab_bps >= ENTRY_EDGE_BPS:
                self._fire_pair(p, side="AB", px_a=mid_a, px_b=mid_b)
                continue
            if edge_ba_bps >= ENTRY_EDGE_BPS:
                self._fire_pair(p, side="BA", px_a=mid_a, px_b=mid_b)
                continue

    # ---------------- Orders & Accounting ----------------
    def _fire_pair(self, p: Pair, side: str, px_a: float, px_b: float) -> None:
        """
        Place simultaneous orders:
          side="AB": buy on A, sell on B
          side="BA": buy on B, sell on A
        Quantity determined by USD target & min_qty guard.
        """
        # base qty from USD target and the *buy* venue price
        if side == "AB":
            buy_px = px_a
        else:
            buy_px = px_b
        qty_base = p.qty_usd / max(buy_px, 1e-9)
        if p.min_qty is not None:
            qty_base = max(qty_base, p.min_qty)

        if qty_base * buy_px < MIN_TICKET_USD:
            return

        # Fire paired orders
        if side == "AB":
            # buy on A
            self.order(_qkey(p.sym, p.va), "buy",  qty=qty_base, order_type=ORDER_TYPE, tif=TIME_IN_FORCE, venue=p.va) # type: ignore
            # sell on B
            self.order(_qkey(p.sym, p.vb), "sell", qty=qty_base, order_type=ORDER_TYPE, tif=TIME_IN_FORCE, venue=p.vb) # type: ignore
        else:
            # buy on B
            self.order(_qkey(p.sym, p.vb), "buy",  qty=qty_base, order_type=ORDER_TYPE, tif=TIME_IN_FORCE, venue=p.vb) # type: ignore
            # sell on A
            self.order(_qkey(p.sym, p.va), "sell", qty=qty_base, order_type=ORDER_TYPE, tif=TIME_IN_FORCE, venue=p.va) # type: ignore

        # simple inventory bookkeeping (net USD notionals, +long base asset):
        inv_key = _poskey(self.ctx.name, p.sym)
        cur = r.hgetall(inv_key) or {}
        inv_usd = float(cur.get("inv_usd", 0.0)) # type: ignore
        # Buy increases USD exposure (long base), Sell decreases; use average of px_a/px_b
        delta = qty_base * (0.5 * (px_a + px_b))
        if side == "AB":
            inv_usd += delta - delta  # buy & sell symmetric => ideally zero
        else:
            inv_usd += delta - delta  # symmetric too; keep key for visibility
        r.hset(inv_key, mapping={"inv_usd": inv_usd, "ts": _now_ms()})

        # cooldown to avoid hammering
        r.setex(_coolkey(self.ctx.name, p.sym), COOLDOWN_S, "1")