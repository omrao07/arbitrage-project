# backend/strategies/diversified/commodity_supply_demand.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import redis

from backend.engine.strategy_base import Strategy

"""
Commodity Supply–Demand Rotation
--------------------------------
Goes long commodities with tightening fundamentals (bullish supply–demand)
and shorts those with loosening conditions, with a bit of price/curve confirmation.

Inputs (published by your ETL into a Redis Stream, default: metrics.commodity_fundamentals)
Each message should be a JSON payload (stored under stream field "json") with fields like:
{
  "ts_ms": 1723550000000,
  "symbol": "CL.F1",                // your front-month symbol or roll-adjusted
  "price": 79.25,                   // spot/mark in USD per unit
  "inv_draw_bps":  +35,             // + = draw (bullish),  - = build (bearish)
  "prod_change_bps": -12,           // + supply up (bearish), - down (bullish)
  "net_imports_bps": -20,           // + imports up (bearish), - down (bullish)
  "demand_surprise_bps": +18,       // + demand beat (bullish), - miss (bearish)
  "refinery_util_bps": +10,         // crude: + bullish (more runs); products: optional
  "weather_shock_bps": +5,          // + outages/disruptions (bullish)
  "freight_cost_bps": +8,           // + logistics tightness (bullish)
  "basis_prompt_far": +0.45,        // prompt - far in USD/unit (backwardation > 0 = bullish)
  "term_spread_far_near": -0.40,    // far - near in USD/unit (contango > 0 = bearish)
  "price_mom": +0.012               // recent return (e.g., 5–20d), + confirms
}

You can send only a subset; missing factors are ignored. We normalize per‑symbol via
EWMA mean/variance → z‑scores, then combine with weights.
"""

# ------------------------------- Config (env) -------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# The stream your router feeds into this strategy (events carry "json")
FUND_STREAM = os.getenv("CSD_STREAM", "metrics.commodity_fundamentals")

# Trade universe (symbols must match your last_price & OMS)
UNIVERSE = [s.strip().upper() for s in os.getenv(
    "CSD_UNIVERSE",
    "CL.F1,RB.F1,HO.F1,NG.F1,GC.F1,SI.F1,HG.F1,C.F1,S.F1,W.F1"
).split(",") if s.strip()]

# Rebalance cadence (seconds)
REBALANCE_SECS = int(os.getenv("CSD_REBALANCE_SECS", "900"))  # 15m

# Portfolio sizing / construction
GROSS_USD    = float(os.getenv("CSD_GROSS_USD", "150000"))  # total gross notional target
LONG_BUCKET  = int(os.getenv("CSD_LONG_BUCKET", "4"))       # top N to long
SHORT_BUCKET = int(os.getenv("CSD_SHORT_BUCKET", "4"))      # bottom N to short; set 0 for long-only
NEUTRALIZE   = os.getenv("CSD_NEUTRALIZE", "true").lower() in ("1","true","yes")

# Minimums / guards
MIN_PRICE  = float(os.getenv("CSD_MIN_PRICE", "0.01"))
MIN_SCORE  = float(os.getenv("CSD_MIN_SCORE", "0.05"))   # ignore near-zero signals to reduce churn

# Factor weights (defaults reflect intuition of signs listed in header)
# You can override via env like: CSD_WEIGHTS="inv_draw_bps:1.0,prod_change_bps:0.8,term_spread_far_near:0.7"
_DEFAULT_WEIGHTS = {
    "inv_draw_bps":          +1.00,  # draws bullish
    "prod_change_bps":       +0.80,  # production down (negative) is bullish -> we z-score the raw number, weight stays +
    "net_imports_bps":       +0.40,  # imports down bullish
    "demand_surprise_bps":   +0.90,  # demand beats bullish
    "refinery_util_bps":     +0.30,  # (mostly for crude/products)
    "weather_shock_bps":     +0.40,  # disruptions bullish
    "freight_cost_bps":      +0.25,  # tight logistics bullish
    "basis_prompt_far":      +0.75,  # backwardation bullish
    "term_spread_far_near":  -0.75,  # contango bearish (negative weight)
    "price_mom":             +0.35,  # price confirmation
}
WEIGHTS_ENV = os.getenv("CSD_WEIGHTS", "").strip()

# EWMA for mean/variance (event-based)
EWMA_ALPHA = float(os.getenv("CSD_EWMA_ALPHA", "0.05"))  # 5% per event

# Venue hint (optional, for OMS display)
VENUE_HINTS = {
    "CL": "NYMEX", "RB": "NYMEX", "HO": "NYMEX", "NG": "NYMEX",
    "GC": "COMEX", "SI": "COMEX", "HG": "COMEX",
    "C": "CBOT", "S": "CBOT", "W": "CBOT"
}

LAST_PRICE_HKEY = os.getenv("CSD_LAST_PRICE_KEY", "last_price")  # HSET symbol -> {"price": ...}

# ------------------------------- Redis -------------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _venue_for(sym: str) -> Optional[str]:
    px = sym.split(".", 1)[0].upper()
    return VENUE_HINTS.get(px)

def _last_price(symbol: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, symbol.upper())
    if not raw:
        return None
    try:
        return float(json.loads(raw)["price"])
    except Exception:
        try:
            return float(raw)
        except Exception:
            return None

# --------------------------- EWMA z-score tracker ---------------------------
@dataclass
class EwmaMV:
    mean: float = 0.0
    var: float = 1.0
    alpha: float = EWMA_ALPHA

    def update(self, x: float) -> Tuple[float, float, float]:
        # returns (mean, var, z)
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        z = (x - self.mean) / math.sqrt(self.var)
        return self.mean, self.var, z

@dataclass
class SymState:
    factors: Dict[str, EwmaMV] = field(default_factory=dict)
    last_ts: int = 0
    last_price: float = float("nan")
    score: float = 0.0

# ------------------------------ Utilities ------------------------------
def _parse_weights(env_val: str) -> Dict[str, float]:
    if not env_val:
        return dict(_DEFAULT_WEIGHTS)
    out = dict(_DEFAULT_WEIGHTS)
    for part in env_val.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out

WEIGHTS = _parse_weights(WEIGHTS_ENV)

def _factor_fields() -> List[str]:
    return list(WEIGHTS.keys())

def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def _squash(x: float) -> float:
    # keep inside [-1, 1] with smooth saturation
    return math.tanh(x)

# ------------------------------ Strategy ------------------------------
class CommoditySupplyDemand(Strategy):
    """
    Combine normalized fundamentals into a composite 'tightness' score per symbol,
    rank the universe, and allocate long/short buckets on a schedule.
    """

    def __init__(self, name: str = "commodity_supply_demand", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.sym: Dict[str, SymState] = {u: SymState() for u in UNIVERSE}
        self.last_rebalance = 0.0

    # ---- lifecycle ----
    def on_start(self) -> None:
        super().on_start()
        # Advertise universe for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({"symbols": UNIVERSE, "ts": int(time.time()*1000)}))

    # ---- tick handler ----
    def on_tick(self, tick: Dict) -> None:
        """
        Attach the router so this handler receives events from FUND_STREAM where
        each record has a 'json' field with the payload described in the header.
        """
        payload = tick
        # If coming from Redis Streams via router, tick may be dict with already-parsed fields
        # e.g., {"symbol": "...", "price": ..., "inv_draw_bps": ...}
        # If you pipe raw Streams entries, adapt your router to hand off parsed JSON.

        sym = str(payload.get("symbol") or "").upper()
        if not sym or sym not in self.sym:
            return

        px = _safe_float(payload.get("price")) or _last_price(sym)
        if not px or px < MIN_PRICE:
            return

        st = self.sym[sym]
        st.last_ts = int(payload.get("ts_ms") or int(time.time() * 1000))
        st.last_price = px

        # update factor z-scores
        total = 0.0
        wsum = 0.0
        for f in _factor_fields():
            if f not in payload:
                continue
            val = _safe_float(payload.get(f))
            if val is None:
                continue
            if f not in st.factors:
                st.factors[f] = EwmaMV()
            _, _, z = st.factors[f].update(val)
            w = WEIGHTS.get(f, 0.0)
            total += w * z
            wsum += abs(w)

        if wsum > 0:
            # scale & squash
            raw = total / wsum
            st.score = _squash(raw)
            self.emit_signal(st.score)  # strategy-level aggregate will show activity
        else:
            # no change if no factors present
            return

        # time-based rebalance
        now = time.time()
        if now - self.last_rebalance >= REBALANCE_SECS:
            self._rebalance()
            self.last_rebalance = now

    # ---- ranking & targets ----
    def _rank(self) -> List[Tuple[str, float]]:
        ranked = [(s, st.score) for s, st in self.sym.items() if not math.isnan(st.last_price)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _target_weights(self) -> Dict[str, float]:
        ranked = self._rank()
        if not ranked:
            return {s: 0.0 for s in UNIVERSE}

        longs = [s for s, sc in ranked if sc >= MIN_SCORE][:LONG_BUCKET]
        shorts = [s for s, sc in reversed(ranked) if sc <= -MIN_SCORE][:SHORT_BUCKET] if SHORT_BUCKET > 0 else []

        w: Dict[str, float] = {s: 0.0 for s in UNIVERSE}
        if NEUTRALIZE and SHORT_BUCKET > 0:
            if longs:
                wl = 0.5 / max(1, len(longs))
                for s in longs: w[s] = +wl
            if shorts:
                ws = 0.5 / max(1, len(shorts))
                for s in shorts: w[s] = -ws
        else:
            if longs:
                wl = 1.0 / max(1, len(longs))
                for s in longs: w[s] = +wl
        return w

    # ---- orders ----
    def _rebalance(self) -> None:
        tgt_w = self._target_weights()
        if not tgt_w:
            return

        for s, w in tgt_w.items():
            if abs(w) < 1e-9:
                continue
            px = _last_price(s)
            if px is None or px < MIN_PRICE:
                continue

            target_notional = w * GROSS_USD
            # If you store per‑strategy positions in Redis, you can fetch and delta‑trade.
            # For simplicity (and because your OMS risk caps anyway), we just submit target qty.
            qty = target_notional / px
            if abs(qty) * px < max(10.0, 0.0005 * GROSS_USD):  # skip dust
                continue

            side = "buy" if qty > 0 else "sell"
            self.order(s, side, qty=abs(qty), order_type="market", venue=_venue_for(s))