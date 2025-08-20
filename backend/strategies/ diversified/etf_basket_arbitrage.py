# backend/strategies/diversified/etf_basket_arbitrage.py
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
ETF ↔ Basket Arbitrage
----------------------
Live NAV (from components) vs ETF market price.

Definitions:
  NAV = sum_i ( w_i * px_i_in_base_ccy ) / PX_SCALE
  premium_pct = (ETF_px - NAV) / ETF_px

Trading logic (mean‑reverting):
  premium_pct > +ENTRY_BPS → SHORT ETF, LONG basket
  premium_pct < -ENTRY_BPS → LONG  ETF, SHORT basket

You publish:
  • HSET last_price <SYM> '{"price": <px>}'   # ETF & all components
  • HSET etf:weight:<ETF> <TICKER> <weight>   # raw basket weights (auto‑normalized)
  • (optional) HSET etf:fx <PAIR> <rate>      # FX if components in other ccy (e.g., JPYEUR)
  • (optional) HSET etf:inav <ETF> <inav>     # live iNAV (overrides computed NAV if present)
  • (optional) HSET etf:fee:<ETF> {create,redeem}  # bps costs; or set ENV guards

Sizing:
  • USD_ETF_SIDE: $ notional for the ETF leg
  • USD_BASKET_SIDE: $ total notional across components (proportional by weight)

Paper routing:
  • ETF trades with symbol <ETF> (e.g., "SPY")
  • Components are cash tickers ("AAPL", "MSFT", ...)
  • Later, map FX and real brokers via your adapters.
"""

# ============================== CONFIG ==============================
REDIS_HOST = os.getenv("ETFB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("ETFB_REDIS_PORT", "6379"))

ETF_SYMBOL = os.getenv("ETFB_ETF", "SPY").upper()
WEIGHT_HASH = os.getenv("ETFB_WEIGHT_HASH", f"etf:weight:{ETF_SYMBOL}")

# If some components need FX → base ccy, publish per‑pair FX in HSET etf:fx
BASE_CCY = os.getenv("ETFB_BASE_CCY", "USD").upper()
FX_HASH  = os.getenv("ETFB_FX_HASH", "etf:fx")  # HSET etf:fx JPYEUR 0.0062, etc.
# Optional per‑component FX mapping env: "SONY:JPYEUR,TSM:USDTWD" (overrides any defaults)
FX_MAP_ENV = os.getenv("ETFB_FX_MAP", "")

# Use iNAV directly if present?
PREFER_INAV = os.getenv("ETFB_PREFER_INAV", "true").lower() in ("1","true","yes")
INAV_HASH   = os.getenv("ETFB_INAV_HASH", "etf:inav")  # HSET etf:inav <ETF> <value>

# Thresholds (bps on premium & z-score)
ENTRY_BPS = float(os.getenv("ETFB_ENTRY_BPS", "15.0"))
EXIT_BPS  = float(os.getenv("ETFB_EXIT_BPS",  "5.0"))
ENTRY_Z   = float(os.getenv("ETFB_ENTRY_Z",   "1.5"))
EXIT_Z    = float(os.getenv("ETFB_EXIT_Z",    "0.6"))

# Creation/redemption/friction guards (bps shrink premium toward 0)
CREATE_BPS = float(os.getenv("ETFB_CREATE_BPS", "5.0"))   # for discount leg (buy ETF / short basket)
REDEEM_BPS = float(os.getenv("ETFB_REDEEM_BPS", "5.0"))   # for premium leg  (short ETF / long basket)
EXTRA_SLIP_BPS = float(os.getenv("ETFB_SLIP_BPS", "2.0")) # generic trading frictions

# Sizing
USD_ETF_SIDE    = float(os.getenv("ETFB_USD_ETF_SIDE",    "50000"))
USD_BASKET_SIDE = float(os.getenv("ETFB_USD_BASKET_SIDE", "50000"))
MIN_TICKET_USD  = float(os.getenv("ETFB_MIN_TICKET_USD",  "200"))

# Universe truncation
TOP_N = int(os.getenv("ETFB_TOP_N", "50"))  # top-N by |weight| (0 = all)

# Concurrency / cadence
MAX_CONCURRENT  = int(os.getenv("ETFB_MAX_CONCURRENT", "1"))
RECHECK_SECS    = int(os.getenv("ETFB_RECHECK_SECS", "2"))
EWMA_ALPHA      = float(os.getenv("ETFB_EWMA_ALPHA", "0.05"))

# Venues (advisory)
VENUE_EQ = os.getenv("ETFB_VENUE_EQ", "ARCA").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("ETFB_LAST_PRICE_KEY", "last_price")

# ============================== REDIS ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== helpers ==============================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception: return None

def _fx_map_from_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    s = FX_MAP_ENV.strip()
    if not s: return out
    for part in s.split(","):
        if ":" not in part: continue
        k, v = [x.strip().upper() for x in part.split(":", 1)]
        out[k] = v
    return out

FX_MAP = _fx_map_from_env()  # per-symbol -> fx pair (e.g., "SONY" -> "JPYEUR")

def _fx(pair: str) -> Optional[float]:
    v = r.hget(FX_HASH, pair.upper())
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _weights() -> List[Tuple[str, float]]:
    wmap = r.hgetall(WEIGHT_HASH) or {}
    items: List[Tuple[str, float]] = []
    for k, v in wmap.items():
        try:
            items.append((k.upper(), float(v)))
        except Exception:
            pass
    # keep top-N by |w|
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    if TOP_N and TOP_N > 0:
        items = items[:TOP_N]
    # normalize by sum of |w| to keep stable sizing
    tot = sum(abs(w) for _, w in items) or 1.0
    return [(s, w / tot) for s, w in items]

def _component_nav_value(sym: str, w: float) -> Optional[float]:
    px = _hget_price(sym)
    if px is None or px <= 0:
        return None
    pair = FX_MAP.get(sym)
    if pair:
        rate = _fx(pair)
        if rate is None or rate <= 0:
            return None
        px_base = px * rate
    else:
        # assume already in BASE currency if not mapped
        px_base = px
    return w * px_base

def _compute_nav(components: List[Tuple[str, float]]) -> Optional[float]:
    acc = 0.0
    for sym, w in components:
        v = _component_nav_value(sym, w)
        if v is None:
            return None
        acc += v
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

def _ewma_key(etf: str) -> str:
    return f"etfb:ewma:{etf}"

def _load_ewma(etf: str) -> EwmaMV:
    raw = r.get(_ewma_key(etf))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(etf: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(etf), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================== state ==============================
@dataclass
class OpenState:
    side: str  # "short_etf_long_basket" or "long_etf_short_basket"
    etf_qty: float
    basket_qty: Dict[str, float] = field(default_factory=dict)  # signed per‑name qty (+long / -short)
    entry_bps: float = 0.0
    entry_z: float = 0.0
    ts_ms: int = 0

def _poskey(name: str, etf: str) -> str:
    return f"etfb:open:{name}:{etf}"

# ============================== strategy ==============================
class EtfBasketArbitrage(Strategy):
    """
    Arb ETF vs component basket using live NAV, with friction guards and z‑score gating.
    """
    def __init__(self, name: str = "etf_basket_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        comps = _weights()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "etf": ETF_SYMBOL,
            "base_ccy": BASE_CCY,
            "components": [{"symbol": s, "weight": w, "fx_pair": FX_MAP.get(s)} for s, w in comps],
            "prefer_inav": PREFER_INAV,
            "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- engine ---------------
    def _evaluate(self) -> None:
        etf_px = _hget_price(ETF_SYMBOL)
        if etf_px is None or etf_px <= 0:
            return

        # pick NAV: iNAV first (if configured), else compute from components
        nav_inav = r.hget(INAV_HASH, ETF_SYMBOL) if PREFER_INAV else None
        nav = None
        if nav_inav is not None:
            try:
                nav = float(nav_inav)
            except Exception:
                nav = None
        if nav is None:
            comps = _weights()
            if not comps:
                return
            nav = _compute_nav(comps)
            if nav is None:
                return

        # premium as % of ETF price
        premium_pct = (etf_px - nav) / etf_px
        premium_bps = 1e4 * premium_pct

        # friction guard: shrink premium by creation/redemption + slip towards 0
        fric_bps = EXTRA_SLIP_BPS + (REDEEM_BPS if premium_bps > 0 else CREATE_BPS)
        adj_bps = premium_bps - math.copysign(fric_bps, premium_bps)

        # EWMA stats
        ew = _load_ewma(ETF_SYMBOL)
        m, v = ew.update(adj_bps)
        _save_ewma(ETF_SYMBOL, ew)
        z = (adj_bps - m) / math.sqrt(max(v, 1e-12))

        # monitoring signal (positive when ETF rich)
        self.emit_signal(max(-1.0, min(1.0, (adj_bps - m) / 10.0)))

        st = self._load_state()

        # ---- exits first ----
        if st:
            if (abs(adj_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ---- entries ----
        if r.get(_poskey(self.ctx.name, ETF_SYMBOL)) is not None:
            return
        if not (abs(adj_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        # Sizing
        etf_qty = USD_ETF_SIDE / etf_px
        if etf_qty * etf_px < MIN_TICKET_USD or USD_BASKET_SIDE < MIN_TICKET_USD:
            return

        # Build basket size map (signed later by side)
        basket_qty: Dict[str, float] = {}
        comps = _weights()
        for sym, w in comps:
            px = _hget_price(sym)
            if px is None or px <= 0:
                return
            # Convert component price to base if FX mapped
            pair = FX_MAP.get(sym)
            px_base = px * (_fx(pair) if pair else 1.0)
            if px_base is None or px_base <= 0:
                return
            notional = USD_BASKET_SIDE * abs(w)  # proportional by |w|
            qty = notional / px_base
            basket_qty[sym] = qty

        if adj_bps > 0:
            # ETF rich → SHORT ETF, LONG basket
            self.order(ETF_SYMBOL, "sell", qty=etf_qty, order_type="market", venue=VENUE_EQ)
            for sym, qty in basket_qty.items():
                self.order(sym, "buy", qty=abs(qty), order_type="market", venue=VENUE_EQ)
            self._save_state(OpenState(
                side="short_etf_long_basket",
                etf_qty=etf_qty,
                basket_qty=basket_qty,
                entry_bps=adj_bps,
                entry_z=z,
                ts_ms=int(time.time()*1000)
            ))
        else:
            # ETF cheap → LONG ETF, SHORT basket
            self.order(ETF_SYMBOL, "buy", qty=etf_qty, order_type="market", venue=VENUE_EQ)
            for sym, qty in basket_qty.items():
                self.order(sym, "sell", qty=abs(qty), order_type="market", venue=VENUE_EQ)
            # store signed for clarity (neg for shorts)
            self._save_state(OpenState(
                side="long_etf_short_basket",
                etf_qty=etf_qty,
                basket_qty={k: -v for k, v in basket_qty.items()},
                entry_bps=adj_bps,
                entry_z=z,
                ts_ms=int(time.time()*1000)
            ))

    # --------------- state io ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, ETF_SYMBOL))
        if not raw: return None
        try:
            o = json.loads(raw)
            if isinstance(o.get("basket_qty"), list):
                o["basket_qty"] = dict(o["basket_qty"])
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, ETF_SYMBOL), json.dumps({
            "side": st.side,
            "etf_qty": st.etf_qty,
            "basket_qty": st.basket_qty,
            "entry_bps": st.entry_bps,
            "entry_z": st.entry_z,
            "ts_ms": st.ts_ms
        }))

    # --------------- unwind ---------------
    def _close(self, st: OpenState) -> None:
        if st.side == "short_etf_long_basket":
            self.order(ETF_SYMBOL, "buy", qty=st.etf_qty, order_type="market", venue=VENUE_EQ)
            for sym, qty in st.basket_qty.items():
                self.order(sym, "sell", qty=abs(qty), order_type="market", venue=VENUE_EQ)
        else:
            self.order(ETF_SYMBOL, "sell", qty=st.etf_qty, order_type="market", venue=VENUE_EQ)
            for sym, qty in st.basket_qty.items():
                self.order(sym, "buy", qty=abs(qty), order_type="market", venue=VENUE_EQ)
        r.delete(_poskey(self.ctx.name, ETF_SYMBOL))