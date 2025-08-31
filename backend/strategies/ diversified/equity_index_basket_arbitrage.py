# backend/strategies/diversified/equity_index_basket_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Equity Index ↔ Basket Arbitrage
--------------------------------
Compares an index proxy (ETF or future) to a weighted cash equity basket of its constituents.

Spread (percent of index):
    basket_px = Σ w_i * px_i
    basis_pct = (index_px - basket_px) / index_px

If basis_pct > +ENTRY_BPS (index rich)  -> SHORT index, LONG basket
If basis_pct < -ENTRY_BPS (index cheap) -> LONG index,  SHORT basket

• Weights come from Redis (HSET index:weight:<INDEX> <TICKER> <weight>), auto-normalized.
• Prices from HSET last_price <SYMBOL> '{"price": <px>}' (for index and all components).
• Optional trading cost guards (bps) deducted from edge.
• Sizing by USD notionals per *side*; per‑name basket notional = USD_BASKET * weight.

Paper symbols (route later in adapters):
  - Index: use your ETF "SPY" or a synthetic future "IFUT:SPX:NEAR"
  - Components: cash tickers ("AAPL","MSFT",...)
"""

# ============================== CONFIG ==============================
REDIS_HOST = os.getenv("IDXARB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("IDXARB_REDIS_PORT", "6379"))

INDEX = os.getenv("IDXARB_INDEX", "SPY").upper()                 # index proxy symbol
WEIGHT_HASH = os.getenv("IDXARB_WEIGHT_HASH", f"index:weight:{INDEX}")  # HSET index:weight:SPY AAPL 0.069 ...

# Liquidity / component subset (0 = all)
TOP_N = int(os.getenv("IDXARB_TOP_N", "50"))   # take top N by absolute weight; 0 means use all

# Thresholds (bps and z-score on basis_pct*1e4)
ENTRY_BPS = float(os.getenv("IDXARB_ENTRY_BPS", "12.0"))
EXIT_BPS  = float(os.getenv("IDXARB_EXIT_BPS",  "4.0"))
ENTRY_Z   = float(os.getenv("IDXARB_ENTRY_Z",   "1.5"))
EXIT_Z    = float(os.getenv("IDXARB_EXIT_Z",    "0.6"))

# Trading cost/slippage guards (bps, applied twice for basket and once for index)
COST_INDEX_BPS  = float(os.getenv("IDXARB_COST_INDEX_BPS",  "1.0"))
COST_BASKET_BPS = float(os.getenv("IDXARB_COST_BASKET_BPS", "2.0"))

# Sizing
USD_INDEX_SIDE  = float(os.getenv("IDXARB_USD_INDEX_SIDE",  "50000"))  # notional on index leg
USD_BASKET_SIDE = float(os.getenv("IDXARB_USD_BASKET_SIDE", "50000"))  # total basket notional
MIN_TICKET_USD  = float(os.getenv("IDXARB_MIN_TICKET_USD",  "200"))

# Concurrency
MAX_CONCURRENT  = int(os.getenv("IDXARB_MAX_CONCURRENT", "1"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("IDXARB_RECHECK_SECS", "3"))
EWMA_ALPHA   = float(os.getenv("IDXARB_EWMA_ALPHA", "0.05"))

# Venues (advisory)
VENUE_EQ   = os.getenv("IDXARB_VENUE_EQ", "ARCA").upper()
VENUE_FUT  = os.getenv("IDXARB_VENUE_FUT", "CME").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("IDXARB_LAST_PRICE_KEY", "last_price")

# ============================== REDIS ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== helpers ==============================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])#type:ignore
    except Exception:
        try: return float(raw)#type:ignore
        except Exception: return None

def _weights() -> List[Tuple[str, float]]:
    wmap = r.hgetall(WEIGHT_HASH) or {}
    items: List[Tuple[str, float]] = []
    for k, v in wmap.items():#type:ignore
        try:
            items.append((k.upper(), float(v)))
        except Exception:
            pass
    # keep top-N by |w|
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    if TOP_N and TOP_N > 0:
        items = items[:TOP_N]
    tot = sum(abs(w) for _, w in items) or 1.0
    # normalize by sum of absolute weights to keep per‑name exposure sane
    items = [(s, w / tot) for s, w in items]
    return items

def _basket_mark(components: List[Tuple[str, float]]) -> Optional[float]:
    acc = 0.0
    for sym, w in components:
        px = _hget_price(sym)
        if px is None or px <= 0:
            return None
        acc += w * px
    return acc

# ============================== EWMA ==============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float, float]:
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        return self.mean, self.var

def _ewma_key(index: str) -> str:
    return f"idxarb:ewma:{index}"

def _load_ewma(index: str) -> EwmaMV:
    raw = r.get(_ewma_key(index))
    if raw:
        try:
            o = json.loads(raw)#type:ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(index: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(index), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================== state ==============================
@dataclass
class OpenState:
    side: str  # "short_index_long_basket" or "long_index_short_basket"
    index_qty: float
    basket_qty: Dict[str, float] = field(default_factory=dict)  # per‑symbol share qty (+long / -short)
    entry_bps: float = 0.0
    entry_z: float = 0.0
    ts_ms: int = 0

def _poskey(name: str, index: str) -> str:
    return f"idxarb:open:{name}:{index}"

# ============================== strategy ==============================
class EquityIndexBasketArbitrage(Strategy):
    """
    Arb index proxy vs weighted component basket using mean‑reverting basis.
    """
    def __init__(self, name: str = "equity_index_basket_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        comps = _weights()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "index": INDEX, "components": [{"symbol": s, "weight": w} for s, w in comps],
            "top_n": TOP_N, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- engine ---------------
    def _evaluate(self) -> None:
        index_px = _hget_price(INDEX)
        comps = _weights()
        if index_px is None or index_px <= 0 or not comps:
            return
        basket_px = _basket_mark(comps)
        if basket_px is None or basket_px <= 0:
            return

        raw_basis_pct = (index_px - basket_px) / index_px
        # trading cost guard (bps → decimal)
        cost_guard = (COST_INDEX_BPS + COST_BASKET_BPS) / 1e4
        adj_basis_pct = raw_basis_pct - math.copysign(cost_guard, raw_basis_pct)  # shrink towards 0

        # EWMA on bps
        ew = _load_ewma(INDEX)
        bps = 1e4 * adj_basis_pct
        m, v = ew.update(bps)
        _save_ewma(INDEX, ew)
        z = (bps - m) / math.sqrt(max(v, 1e-12))

        # emit monitor signal (positive if index rich)
        self.emit_signal(max(-1.0, min(1.0, (bps - m) / 10.0)))

        st = self._load_state()
        # ---- exits first ----
        if st:
            if (abs(bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ---- entries ----
        if r.get(_poskey(self.ctx.name, INDEX)) is not None:
            return
        if not (abs(bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        # Sizing
        idx_qty = USD_INDEX_SIDE / index_px
        if idx_qty * index_px < MIN_TICKET_USD or USD_BASKET_SIDE < MIN_TICKET_USD:
            return
        basket_qty: Dict[str, float] = {}
        for sym, w in comps:
            px = _hget_price(sym)
            if px is None or px <= 0:
                return
            notional = USD_BASKET_SIDE * w
            qty = notional / px
            basket_qty[sym] = qty  # sign applied later by side

        if bps > 0:
            # index rich → SHORT index, LONG basket
            self.order(INDEX, "sell", qty=idx_qty, order_type="market", venue=VENUE_EQ if ":" not in INDEX else VENUE_FUT)
            for sym, qty in basket_qty.items():
                self.order(sym, "buy", qty=abs(qty), order_type="market", venue=VENUE_EQ)
            self._save_state(OpenState(
                side="short_index_long_basket",
                index_qty=idx_qty,
                basket_qty=basket_qty,
                entry_bps=bps,
                entry_z=z,
                ts_ms=int(time.time()*1000)
            ))
        else:
            # index cheap → LONG index, SHORT basket
            self.order(INDEX, "buy", qty=idx_qty, order_type="market", venue=VENUE_EQ if ":" not in INDEX else VENUE_FUT)
            for sym, qty in basket_qty.items():
                self.order(sym, "sell", qty=abs(qty), order_type="market", venue=VENUE_EQ)
            self._save_state(OpenState(
                side="long_index_short_basket",
                index_qty=idx_qty,
                basket_qty={k: -v for k, v in basket_qty.items()},  # store signed
                entry_bps=bps,
                entry_z=z,
                ts_ms=int(time.time()*1000)
            ))

    # --------------- state io ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, INDEX))
        if not raw: return None
        try:
            o = json.loads(raw)#type:ignore
            # backward compatibility for unsigned storage
            if isinstance(o.get("basket_qty"), list):
                o["basket_qty"] = dict(o["basket_qty"])
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, INDEX), json.dumps({
            "side": st.side,
            "index_qty": st.index_qty,
            "basket_qty": st.basket_qty,
            "entry_bps": st.entry_bps,
            "entry_z": st.entry_z,
            "ts_ms": st.ts_ms
        }))

    # --------------- unwind ---------------
    def _close(self, st: OpenState) -> None:
        # close index
        if st.side == "short_index_long_basket":
            self.order(INDEX, "buy", qty=st.index_qty, order_type="market", venue=VENUE_EQ if ":" not in INDEX else VENUE_FUT)
            # sell back basket longs
            for sym, qty in st.basket_qty.items():
                self.order(sym, "sell", qty=abs(qty), order_type="market", venue=VENUE_EQ)
        else:
            self.order(INDEX, "sell", qty=st.index_qty, order_type="market", venue=VENUE_EQ if ":" not in INDEX else VENUE_FUT)
            # buy back basket shorts
            for sym, qty in st.basket_qty.items():
                self.order(sym, "buy", qty=abs(qty), order_type="market", venue=VENUE_EQ)
        r.delete(_poskey(self.ctx.name, INDEX))