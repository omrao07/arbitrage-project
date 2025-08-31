# backend/strategies/diversified/low_latency_market_making.py
from __future__ import annotations

import json, math, os, time, random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Low-Latency Market Making (paper)
---------------------------------
Quotes a two-sided market around fair value with:
  • Fair = blend(mid, microprice) +/- inventory skew
  • Spread = base + k * vol_ewma + queue/imbalance adders
  • Adverse-selection guard using last trade direction & top-of-book changes
  • Inventory bands with skew and side-throttling
  • Replace/refresh cadence with maker-only semantics (paper)

Redis feeds you already maintain:
  HSET orderbook:best <SYM>  '{"bid":px,"ask":px,"bid_sz":q,"ask_sz":q}'
  HSET last_trade    <SYM>   '{"price":px,"side":"buy|sell","qty":q,"ts":ms}'
  HSET last_price    <SYM>   '{"price":px}'
Optional controls (set from ops UI / CLI):
  SET mm:halt 0|1
  HSET mm:risk <SYM> '{"pos_limit":N,"notional_limit":USD}'
"""

# ============ CONFIG (env) ============
REDIS_HOST = os.getenv("MM_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("MM_REDIS_PORT", "6379"))

SYM   = os.getenv("MM_SYMBOL", "BTCUSDT").upper()
VENUE = os.getenv("MM_VENUE", "BINANCE").upper()

# quoting engine
BASE_SPREAD_BPS  = float(os.getenv("MM_BASE_SPREAD_BPS", "4.0"))     # min half-spread each side (bps on mid)
VOL_K            = float(os.getenv("MM_VOL_K", "2.0"))               # multiplier on vol_ewma (bps)
MICROPRICE_W     = float(os.getenv("MM_MICRO_W", "0.60"))            # weight on microprice vs mid
INVENTORY_SKEW_BPS = float(os.getenv("MM_SKEW_BPS", "1.5"))          # per 10% of inv usage
QUEUE_ADDER_BPS  = float(os.getenv("MM_QUEUE_ADDER_BPS", "1.0"))     # add if top depth thin or adverse flow

# sizing
USD_QUOTE_SIZE   = float(os.getenv("MM_USD_QUOTE_SIZE", "1000"))     # per order
MAX_ACTIVE_ORDERS= int(os.getenv("MM_MAX_ACTIVE", "2"))              # per side
MIN_TICKET_USD   = float(os.getenv("MM_MIN_TICKET_USD", "50"))

# risk / inventory
POS_LIMIT_UNITS  = float(os.getenv("MM_POS_LIMIT_UNITS", "0.5"))     # max absolute position in units (e.g., BTC)
NOTIONAL_LIMIT   = float(os.getenv("MM_NOTIONAL_LIMIT", "5000"))     # cap on absolute notional
REBID_CROSS_BPS  = float(os.getenv("MM_REBID_CROSS_BPS", "1.5"))     # distance to reprice if mid moves (bps)
HARD_SPREAD_CAP_BPS = float(os.getenv("MM_SPREAD_CAP_BPS", "60.0"))  # cap total quoted half-spread

# cadence
TICK_SECS        = float(os.getenv("MM_TICK_SECS", "0.2"))
REPRICE_COOLDOWN = float(os.getenv("MM_REPRICE_COOLDOWN", "0.15"))   # min secs between replaces per side

# EWMA
VOL_ALPHA        = float(os.getenv("MM_VOL_ALPHA", "0.12"))          # for mid returns EWMA (per tick)
VOL_FLOOR_BPS    = float(os.getenv("MM_VOL_FLOOR_BPS", "2.0"))

# maker-only flags (paper hints for adapter)
MAKER_ONLY       = os.getenv("MM_MAKER_ONLY", "true").lower() in ("1","true","yes")

# redis keys
OB_HKEY    = os.getenv("MM_OB_KEY", "orderbook:best")
LT_HKEY    = os.getenv("MM_LAST_TRADE_KEY", "last_trade")
LAST_HKEY  = os.getenv("MM_LAST_PRICE_KEY", "last_price")
HALT_KEY   = os.getenv("MM_HALT_KEY", "mm:halt")
RISK_HKEY  = os.getenv("MM_RISK_HKEY", "mm:risk")

# ============ Redis ============
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============ helpers ============
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try: return json.loads(raw) # type: ignore
    except Exception:
        return None

def _bestbook(sym: str) -> Optional[Tuple[float,float,float,float]]:
    o = _hget_json(OB_HKEY, sym)
    if not o: return None
    b, a = float(o.get("bid",0)), float(o.get("ask",0))
    bs, asz = float(o.get("bid_sz",0)), float(o.get("ask_sz",0))
    if b<=0 or a<=0: return None
    return b, a, bs, asz

def _mid(b: float, a: float) -> float:
    return 0.5*(a+b)

def _microprice(b: float, a: float, bs: float, asz: float) -> float:
    # microprice weighted by queue depth
    if bs+asz <= 0: return _mid(b,a)
    w = asz / (bs + asz)
    return w*b + (1-w)*a

def _bps(x: float, y: float) -> float:
    return 1e4*(x - y)/y

def _now() -> float:
    return time.time()

def _side_from_flow() -> int:
    """+1 if last trade was a buy (lifted ask); -1 if sell; 0 if unknown."""
    lt = _hget_json(LT_HKEY, SYM) or {}
    side = (lt.get("side") or "").lower()
    if side == "buy": return +1
    if side == "sell": return -1
    return 0

def _risk_caps() -> Tuple[float,float]:
    j = _hget_json(RISK_HKEY, SYM) or {}
    pos_lim = float(j.get("pos_limit", POS_LIMIT_UNITS))
    not_lim = float(j.get("notional_limit", NOTIONAL_LIMIT))
    return pos_lim, not_lim

# ============ simple position cache (paper) ============
def _pos_key(sym: str) -> str:
    return f"pos:{sym}"

def _get_pos(sym: str) -> float:
    v = r.get(_pos_key(sym))
    try: return float(v) if v is not None else 0.0 # type: ignore
    except Exception: return 0.0

def _bump_pos(sym: str, side: str, qty: float):
    # naive fill model: assume resting quotes that get hit adjust position here when OMS confirms
    # In paper, we update on our own orders immediately; in adapters, you should update on real fills.
    p = _get_pos(sym)
    p += (qty if side == "buy" else -qty)
    r.set(_pos_key(sym), p)

# ============ EWMA vol ============
@dataclass
class VolEWMA:
    mean: float
    var: float
    alpha: float
    last_mid: float

def _vol_key(sym: str) -> str:
    return f"mm:vol:{sym}"

def _load_vol(sym: str) -> VolEWMA:
    raw = r.get(_vol_key(sym))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return VolEWMA(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", VOL_ALPHA)), last_mid=float(o.get("lm", 0)))
        except Exception:
            pass
    return VolEWMA(mean=0.0, var=1e-10, alpha=VOL_ALPHA, last_mid=0.0)

def _save_vol(sym: str, v: VolEWMA):
    r.set(_vol_key(sym), json.dumps({"m": v.mean, "v": v.var, "a": v.alpha, "lm": v.last_mid}))

def _update_vol(sym: str, mid_px: float) -> float:
    v = _load_vol(sym)
    if v.last_mid > 0 and mid_px > 0:
        ret = (mid_px - v.last_mid) / v.last_mid
        m0 = v.mean
        v.mean = (1 - v.alpha) * v.mean + v.alpha * ret
        v.var  = max(1e-16, (1 - v.alpha) * (v.var + (ret - m0) * (ret - v.mean)))
    v.last_mid = mid_px
    _save_vol(sym, v)
    # convert std to bps estimate per tick
    std = math.sqrt(max(v.var, 1e-16))
    return max(VOL_FLOOR_BPS, 1e4 * std)

# ============ quoting state ============
@dataclass
class SideState:
    last_replace_ts: float
    active: int

def _side_key(sym: str, side: str) -> str:
    return f"mm:state:{sym}:{side}"

def _load_side(sym: str, side: str) -> SideState:
    raw = r.get(_side_key(sym, side))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return SideState(last_replace_ts=float(o.get("ts", 0)), active=int(o.get("n", 0)))
        except Exception: pass
    return SideState(last_replace_ts=0.0, active=0)

def _save_side(sym: str, side: str, st: SideState):
    r.set(_side_key(sym, side), json.dumps({"ts": st.last_replace_ts, "n": st.active}))

# ============ strategy ============
class LowLatencyMarketMaking(Strategy):
    """
    Two-sided market making with microprice fair, EWMA vol spread, inventory skew, and flow guards.
    """
    def __init__(self, name: str = "low_latency_market_making", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._tlast = 0.0

    def on_tick(self, tick: Dict) -> None:
        now = _now()
        if now - self._tlast < TICK_SECS:
            return
        self._tlast = now

        # kill switch
        if (r.get(HALT_KEY) or "0") == "1":
            return

        book = _bestbook(SYM)
        if not book:
            return
        b, a, bs, asz = book
        if b <= 0 or a <= 0 or a <= b:
            return

        mid = _mid(b, a)
        micro = _microprice(b, a, bs, asz)
        vol_bps = _update_vol(SYM, mid)

        # inventory & risk
        pos = _get_pos(SYM)
        pos_cap, not_cap = _risk_caps()
        # notional check (use mid)
        if abs(pos) * mid >= not_cap:
            return

        inv_usage = 0.0 if pos_cap <= 0 else (pos / pos_cap)  # -1..1
        inv_skew_bps = (INVENTORY_SKEW_BPS * 0.1) * (inv_usage * 10.0)  # per 10% usage

        # flow/adverse guard
        flow = _side_from_flow()  # +1 buy, -1 sell
        queue_thin = (bs < asz and bs * b < USD_QUOTE_SIZE) or (asz < bs and asz * a < USD_QUOTE_SIZE)
        adverse = (flow > 0 and micro > mid) or (flow < 0 and micro < mid) or queue_thin

        # compute fair & spreads
        fair = MICROPRICE_W * micro + (1 - MICROPRICE_W) * mid
        half_spread_bps = BASE_SPREAD_BPS + VOL_K * vol_bps + (QUEUE_ADDER_BPS if adverse else 0.0)
        half_spread_bps = min(half_spread_bps, HARD_SPREAD_CAP_BPS)

        # apply inventory skew: push fair away from current inventory
        fair_adj = fair * (1 + inv_skew_bps * 1e-4)

        # target quotes
        bid_px = fair_adj * (1 - half_spread_bps * 1e-4)
        ask_px = fair_adj * (1 + half_spread_bps * 1e-4)

        # reprice gates: avoid spam if close to current book
        if abs(_bps(bid_px, b)) < REBID_CROSS_BPS:
            bid_px = b  # keep near best bid
        if abs(_bps(ask_px, a)) < REBID_CROSS_BPS:
            ask_px = a  # keep near best ask

        # sizes (equal notional on each side; throttle if inventory near cap)
        usd_bid = min(USD_QUOTE_SIZE, max(0.0, (not_cap - abs(pos)*mid)))
        usd_ask = usd_bid
        qty_bid = max(0.0, usd_bid / max(1e-9, bid_px))
        qty_ask = max(0.0, usd_ask / max(1e-9, ask_px))

        if bid_px * qty_bid < MIN_TICKET_USD and ask_px * qty_ask < MIN_TICKET_USD:
            return

        # side throttling if inventory biased
        if inv_usage > 0.5:   # long inventory → prioritize asks
            qty_bid *= 0.3
        elif inv_usage < -0.5:  # short → prioritize bids
            qty_ask *= 0.3

        # refresh/replace respecting cooldown & MAX_ACTIVE per side
        self._refresh_side("buy",  bid_px, qty_bid, now)
        self._refresh_side("sell", ask_px, qty_ask, now)

        # monitoring signal: positive when ask>bid spread wide / adverse
        sig = 0.25 * (1 if adverse else -1) + 0.75 * (half_spread_bps / max(1.0, BASE_SPREAD_BPS + VOL_FLOOR_BPS))
        self.emit_signal(max(-1.0, min(1.0, sig)))

    # ----- helpers -----
    def _refresh_side(self, side: str, px: float, qty: float, now: float) -> None:
        st = _load_side(SYM, side)
        if (now - st.last_replace_ts) < REPRICE_COOLDOWN:
            return
        if qty * px < MIN_TICKET_USD:
            return

        # Replace strategy: cancel excess active & place one fresh maker order
        if st.active > 0:
            # In paper, we don’t track order IDs; just indicate a cancel by placing with replace flag
            st.active = max(0, st.active - 1)

        flags = {"post_only": MAKER_ONLY, "replace": True}
        self.order(SYM, side, qty=qty, price=px, order_type="limit", venue=VENUE, flags=flags) # type: ignore
        st.last_replace_ts = now
        st.active = min(MAX_ACTIVE_ORDERS, st.active + 1)
        _save_side(SYM, side, st)

        # naive position bump on our own passive fill probability (very small; adapter should overwrite on real fills)
        if random.random() < 0.02:
            _bump_pos(SYM, side, qty * 0.1)  # tiny fraction simulating partial fills