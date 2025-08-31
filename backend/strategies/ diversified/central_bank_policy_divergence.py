# backend/strategies/diversified/central_bank_policy_divergence.py
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
Data it expects in Redis (all optional; falls back gracefully):

# Your house view: expected policy rate path (annualized decimals)
HSET policy:exp_path:USD  "3M" 0.055  "6M" 0.053  "12M" 0.048
HSET policy:exp_path:EUR  "3M" 0.040  "6M" 0.037  "12M" 0.033

# Market-implied path (e.g., OIS forwards, annualized decimals)
HSET ois:fwd:USD  "3M" 0.053  "6M" 0.050  "12M" 0.045
HSET ois:fwd:EUR  "3M" 0.039  "6M" 0.035  "12M" 0.031

# Optional: last policy surprise in bps (actual - expected)
# Positive surprise (hawkish) -> +bias
HSET policy:surprise "USD" 25 "EUR" -10

# Optional macro guard (0..1 risk aversion; >0.7 reduce risk)
SET macro:risk_aversion 0.3
"""

# ---------------- Env / Config ----------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Pairs you want to trade (BASEQUOTE). Symbols must match your data feed/OMS.
PAIRS = [s.strip().upper() for s in os.getenv(
    "CBD_PAIRS",
    "EURUSD,GBPUSD,AUDUSD,USDJPY,USDCAD,USDCHF"
).split(",") if s.strip()]

# Stream to attach this strategy to (router will feed ticks here)
EVENT_STREAM = os.getenv("CBD_EVENT_STREAM", "trades.fx")

# Rebalance cadence (seconds)
REBALANCE_SECS = int(os.getenv("CBD_REBALANCE_SECS", "900"))  # 15m

# Capital & sizing
GROSS_USD       = float(os.getenv("CBD_GROSS_USD", "100000"))
MAX_LEVER       = float(os.getenv("CBD_MAX_LEVER", "1.0"))    # cap total abs weights
PER_PAIR_CAP    = float(os.getenv("CBD_PER_PAIR_CAP", "0.3")) # max |weight| per pair

# Signal weights
W_PATH   = float(os.getenv("CBD_W_PATH", "0.7"))   # policy path divergence weight
W_SURP   = float(os.getenv("CBD_W_SURP", "0.3"))   # surprise weight
HORIZONS = [h.strip().upper() for h in os.getenv("CBD_HORIZONS", "3M,6M,12M").split(",") if h.strip()]

# Surprise decay (per day events; EWMA on surprises)
SURPRISE_HALFLIFE = int(os.getenv("CBD_SURPRISE_HALFLIFE", "10"))

# Macro risk guard
RISK_OFF_THRESHOLD = float(os.getenv("CBD_RISK_OFF_THRESHOLD", "0.7"))
RISK_SCALING_MIN   = float(os.getenv("CBD_RISK_SCALING_MIN", "0.25"))  # min risk when risk_aversion=1

# Safety
MIN_PX = float(os.getenv("CBD_MIN_FX_PRICE", "0.0001"))

# Venue hint for OMS
VENUE = os.getenv("CBD_VENUE", "OANDA").upper()

# ---------------- Redis ----------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception: return None

def _getf(key: str) -> Optional[float]:
    v = r.get(key)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception: return None

def _last_price(sym: str) -> Optional[float]:
    raw = r.hget("last_price", sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"]) # type: ignore
    except Exception: return None

def _pos_by_strategy(strategy: str, symbol: str) -> Dict:
    raw = r.hget(f"positions:by_strategy:{strategy}", symbol)
    if not raw: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0}
    try: return json.loads(raw) # type: ignore
    except Exception: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0}

# --------------- Helpers ----------------
def _currency(pair: str) -> Tuple[str, str]:
    # EURUSD -> (EUR, USD); USDJPY -> (USD, JPY)
    if len(pair) >= 6:
        base, quote = pair[:3], pair[3:6]
        return base.upper(), quote.upper()
    # fallback (rare)
    parts = (pair[:3].upper(), pair[3:].upper())
    return parts

def _symbol(pair: str) -> str:
    # Use your convention for spot symbol names
    return f"{pair}.SPOT"

def _smooth(prev: Optional[float], x: float, halflife: int = SURPRISE_HALFLIFE) -> float:
    if prev is None: return x
    alpha = 1.0 - math.exp(math.log(0.5) / max(1, halflife))
    return (1 - alpha) * prev + alpha * x

def _risk_scale() -> float:
    ra = _getf("macro:risk_aversion")
    if ra is None: return 1.0
    ra = max(0.0, min(1.0, ra))
    # linear scale down to RISK_SCALING_MIN as risk aversion approaches 1
    return max(RISK_SCALING_MIN, 1.0 - ra * (1.0 - RISK_SCALING_MIN))

# --------------- Core signal math ----------------
def _path_divergence(ccy: str) -> Optional[float]:
    """
    Return weighted average (our expected - market implied) over horizons, in decimals (e.g., 0.002 = 20 bps).
    """
    diffs = []
    weights = []
    for h in HORIZONS:
        expv = _hgetf(f"policy:exp_path:{ccy}", h)
        oisv = _hgetf(f"ois:fwd:{ccy}", h)
        if expv is None or oisv is None:
            continue
        w = 1.0
        if h.endswith("M"):
            try:
                w = float(h[:-1])  # e.g., 12M -> 12 weight
            except Exception:
                w = 1.0
        elif h.endswith("Y"):
            try:
                w = 12.0 * float(h[:-1])  # 1Y -> 12
            except Exception:
                w = 12.0
        diffs.append(expv - oisv)
        weights.append(w)
    if not diffs:
        return None
    num = sum(d * w for d, w in zip(diffs, weights))
    den = sum(weights) or 1.0
    return num / den

def _surprise_bias(ccy: str) -> float:
    """
    Surprise in decimals (25 bps -> 0.0025). EWMA smooth via Redis key cbd:surp:<CCY>.
    """
    raw = r.hget("policy:surprise", ccy)
    if raw is None:
        # try cached
        prev = _getf(f"cbd:surp:{ccy}")
        return float(prev or 0.0)
    try:
        bps = float(raw) # type: ignore
    except Exception:
        bps = 0.0
    surprise_dec = bps / 1e4  # 25 -> 0.0025
    prev = _getf(f"cbd:surp:{ccy}")
    sm = _smooth(prev, surprise_dec)
    r.set(f"cbd:surp:{ccy}", sm)
    return sm

def _pair_signal(pair: str) -> Optional[float]:
    """
    Positive signal -> long BASE / short QUOTE (i.e., buy BASEQUOTE)
    Negative -> short BASE / long QUOTE.
    """
    base, quote = _currency(pair)
    d_base  = _path_divergence(base)
    d_quote = _path_divergence(quote)
    if d_base is None or d_quote is None:
        return None

    s_base  = _surprise_bias(base)
    s_quote = _surprise_bias(quote)

    score = W_PATH * (d_base - d_quote) + W_SURP * (s_base - s_quote)

    # squash to [-1, 1] so allocator-sized buckets remain sensible
    return math.tanh(score * 10.0)  # scale factor to make ~25–50bps diffs meaningful

# --------------- Strategy ----------------
@dataclass
class PairState:
    last_signal: float = 0.0

class CentralBankPolicyDivergence(Strategy):
    """
    Long the currency whose expected policy path (house view) is above market-implied relative to its pair mate.
    Rebalances every REBALANCE_SECS, sizes by GROSS_USD and risk guard.
    """

    def __init__(self, name: str = "central_bank_policy_divergence", region: Optional[str] = "FX", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.states: Dict[str, PairState] = {p: PairState() for p in PAIRS}
        self.last_rebalance = 0.0

    def on_start(self) -> None:
        super().on_start()
        # publish the trading universe for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({"pairs": PAIRS, "ts": int(time.time()*1000)}))

    def on_tick(self, tick: Dict) -> None:
        """
        Attach to FX spot stream or a light heartbeat stream; we recompute on cadence.
        """
        now = time.time()
        # emit a global signal proxy (average absolute signal) for allocator visibility
        signals = []
        for pair in PAIRS:
            sig = _pair_signal(pair)
            if sig is None:
                continue
            self.states[pair].last_signal = sig
            signals.append(sig)
        if signals:
            # strategy-level signal: average strength (signed)
            self.emit_signal(sum(signals) / max(1, len(signals)))

        if now - self.last_rebalance >= REBALANCE_SECS:
            self._rebalance()
            self.last_rebalance = now

    # --------- Rebalance ---------
    def _rebalance(self) -> None:
        # build target weights per pair
        raw_w: Dict[str, float] = {}
        for pair in PAIRS:
            sig = self.states.get(pair, PairState()).last_signal
            if sig == 0.0:
                continue
            # cap per pair
            raw_w[pair] = max(-PER_PAIR_CAP, min(PER_PAIR_CAP, sig))

        if not raw_w:
            return

        # scale to leverage cap and macro guard
        abs_sum = sum(abs(w) for w in raw_w.values()) or 1.0
        scale_lever = min(1.0, MAX_LEVER / abs_sum)
        scale_macro = _risk_scale()
        scale = scale_lever * scale_macro

        tgt_w = {p: w * scale for p, w in raw_w.items()}

        # translate weights to orders
        for pair, w in tgt_w.items():
            sym = _symbol(pair)
            px = _last_price(sym)
            if px is None or px < MIN_PX:
                continue

            target_notional = w * GROSS_USD  # + => long base/short quote
            pos = _pos_by_strategy(self.ctx.name, sym)
            cur_qty = float(pos.get("qty", 0.0))
            cur_notional = cur_qty * px

            delta = target_notional - cur_notional
            if abs(delta) < max(10.0, 0.0005 * GROSS_USD):
                continue

            qty = delta / px
            side = "buy" if qty > 0 else "sell"
            self.order(sym, side, qty=abs(qty), order_type="market", venue=VENUE)