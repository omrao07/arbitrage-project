# backend/strategies/diversified/currency_carry.py
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
Currency Carry (cross‑sectional, with momentum & risk guards)
------------------------------------------------------------
Core idea:
  • For FX pair BASE/QUOTE quoted as QUOTE per 1 BASE (e.g., EURUSD),
    long the pair (buy BASE, sell QUOTE) earns approx carry = r_BASE - r_QUOTE.
  • Shorting earns −carry.

This strategy:
  1) Reads short‑tenor interest rates for currencies from Redis (e.g., policy/OIS or money-market).
  2) Computes per‑pair carry = r_base - r_quote (annualized, decimals).
  3) Optional forward‑points guard (uncovered‑vs‑covered carry): if forward premium
     contradicts spot‑based carry by more than a band, de‑emphasize the signal.
  4) Blends carry with simple price momentum into a score; ranks the universe.
  5) Goes long top N, short bottom N, neutralized by USD notional; rebalances periodically.
  6) Uses your OMS paper fills via Strategy.order().

Expected Redis inputs you already maintain:
  • HSET last_price           <PAIR>   '{"price": 1.08542}'   # spot quote (QUOTE per BASE)
  • HSET fx:rate              <CCY>    0.040                  # r_ccy (decimal, e.g., 0.04 = 4%)
  • (Optional) HSET fx:fwdpt:<TENOR>  <PAIR>   0.0025         # forward points (Fwd - Spot), same units as price
  • (Optional) HSET mom:<WINDOW>      <PAIR>   0.012          # momentum (e.g., 60d return)

Symbols:
  • Pairs like "EURUSD","USDJPY","GBPUSD","AUDUSD","USDCAD","USDCHF","NZDUSD","EURJPY", etc.
  • Order("EURUSD","buy",qty) = long EUR / short USD; qty measured in BASE notional units.
"""

# ============================== CONFIG (env) ==============================
REDIS_HOST = os.getenv("FXC_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("FXC_REDIS_PORT", "6379"))

UNIVERSE = [s.strip().upper() for s in os.getenv(
    "FXC_UNIVERSE",
    "EURUSD,GBPUSD,AUDUSD,NZDUSD,USDJPY,USDCAD,USDCHF,EURJPY,EURGBP,EURCHF"
).split(",") if s.strip()]

# Currencies we expect rates for; derive from pairs if not set
CCYS_ENV = os.getenv("FXC_CCY_LIST", "")

# Factor weights
W_CARRY  = float(os.getenv("FXC_W_CARRY",  "0.70"))
W_MOM    = float(os.getenv("FXC_W_MOM",    "0.30"))
W_FWD_GUARD = float(os.getenv("FXC_W_FWD_GUARD", "0.50"))  # multiplies penalty when fwd disagrees

# Momentum config
MOM_KEY    = os.getenv("FXC_MOM_KEY", "mom:60d")   # HSET mom:60d <PAIR> <ret>
MOM_MINABS = float(os.getenv("FXC_MOM_MINABS", "0.0"))  # ignore tiny mom

# Forward points guard
USE_FWD_GUARD = os.getenv("FXC_USE_FWD_GUARD", "true").lower() in ("1","true","yes")
FWD_TENOR     = os.getenv("FXC_FWD_TENOR", "3M").upper()   # fx:fwdpt:<TENOR>
FWD_BAND_BPS  = float(os.getenv("FXC_FWD_BAND_BPS", "30")) # if |fwd_annualized - carry| > band ⇒ penalize

# Rebalance cadence (seconds)
REBALANCE_SECS = int(os.getenv("FXC_REBALANCE_SECS", "900"))  # 15m

# Portfolio construction
GROSS_USD    = float(os.getenv("FXC_GROSS_USD", "200000"))   # total gross (long + short)
LONG_BUCKET  = int(os.getenv("FXC_LONG_BUCKET", "4"))
SHORT_BUCKET = int(os.getenv("FXC_SHORT_BUCKET", "4"))
NEUTRALIZE   = os.getenv("FXC_NEUTRALIZE", "true").lower() in ("1","true","yes")

# Risk guards
MAX_PCT_PER_PAIR = float(os.getenv("FXC_MAX_PCT_PER_PAIR", "0.30"))  # cap of gross per leg
MIN_PRICE        = float(os.getenv("FXC_MIN_PRICE", "0.0001"))

# Redis keys
LAST_PRICE_HKEY = os.getenv("FXC_LAST_PRICE_KEY", "last_price")   # HSET <PAIR> -> {"price": ...}
RATE_HKEY       = os.getenv("FXC_RATE_KEY",       "fx:rate")      # HSET fx:rate <CCY> -> 0.04
FWDPT_HKEY      = f"fx:fwdpt:{FWD_TENOR}"                         # HSET fx:fwdpt:3M <PAIR> -> +0.0025

# Venue hint
VENUE_FX = os.getenv("FXC_VENUE", "FX").upper()

# ============================== Redis ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== Helpers ==============================
def _split_pair(pair: str) -> Tuple[str, str]:
    pair = pair.upper()
    if len(pair) in (6,7) and "/" not in pair:
        # handle 6-char majors or 7 with 'X' like 'USDCNH'? We'll split first 3 and next 3+.
        base = pair[:3]; quote = pair[3:]
        return base, quote
    if "/" in pair:
        base, quote = pair.split("/", 1)
        return base.upper(), quote.upper()
    # fallback: try common suffixes
    return pair[:3], pair[3:]

def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception: return None

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try:
        return float(v)
    except Exception:
        try:
            return float(json.loads(v))
        except Exception:
            return None

def _ccy_rates(ccys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in ccys:
        v = _hgetf(RATE_HKEY, c)
        if v is not None:
            out[c] = float(v)
    return out

def _annualize_forward(points: float, spot: float, tenor_months: float) -> float:
    """
    Approx covered interest parity implied rate diff (annualized):
      (F - S)/S * (12/tenor_months)
    """
    if spot <= 0:
        return 0.0
    return float(points) / float(spot) * (12.0 / max(0.0001, tenor_months))

def _tenor_months(tag: str) -> float:
    tag = tag.upper().strip()
    if tag.endswith("M"):
        return float(tag[:-1] or 1.0)
    if tag.endswith("Y"):
        return 12.0 * float(tag[:-1] or 1.0)
    return 3.0

def _squash(x: float) -> float:
    return math.tanh(x)

# ============================== State structs ==============================
@dataclass
class SymState:
    px: float = float("nan")
    carry: float = 0.0         # r_base - r_quote (annualized)
    mom: float = 0.0           # momentum (e.g., 60d return)
    fwd_guard: float = 1.0     # penalty ∈ [0,1]
    score: float = 0.0

# ============================== Strategy ==============================
class CurrencyCarry(Strategy):
    """
    Cross‑sectional carry + momentum with optional forward‑points consistency guard.
    """
    def __init__(self, name: str = "currency_carry", region: Optional[str] = "FX", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        # infer currency list if not provided
        if CCYS_ENV.strip():
            self.ccys = [c.strip().upper() for c in CCYS_ENV.split(",") if c.strip()]
        else:
            s = set()
            for p in UNIVERSE:
                b, q = _split_pair(p)
                s.add(b); s.add(q)
            self.ccys = sorted(s)
        self.sym: Dict[str, SymState] = {p: SymState() for p in UNIVERSE}
        self.last_reb = 0.0
        self.tenor_months = _tenor_months(FWD_TENOR)

    # ---- lifecycle ----
    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "pairs": UNIVERSE, "ccys": self.ccys, "tenor": FWD_TENOR,
            "use_fwd_guard": USE_FWD_GUARD, "ts": int(time.time()*1000)
        }))

    # ---- tick ----
    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_reb < REBALANCE_SECS:
            return
        self.last_reb = now
        self._rebalance()

    # ---- core ----
    def _rebalance(self) -> None:
        rates = _ccy_rates(self.ccys)
        ranking: List[Tuple[str, float]] = []

        for pair in UNIVERSE:
            st = self.sym[pair]
            px = _hget_price(pair)
            if px is None or px <= MIN_PRICE:
                st.score = 0.0
                continue
            st.px = px

            base, quote = _split_pair(pair)
            r_base = rates.get(base)
            r_quote = rates.get(quote)
            if r_base is None or r_quote is None:
                st.score = 0.0
                continue

            carry = float(r_base - r_quote)     # annualized diff (decimal)
            st.carry = carry

            # Momentum (optional)
            mom = _hgetf(MOM_KEY, pair) or 0.0
            st.mom = mom if abs(mom) >= MOM_MINABS else 0.0

            # Forward guard (optional)
            guard = 1.0
            if USE_FWD_GUARD:
                fwdpt = _hgetf(FWDPT_HKEY, pair)
                if fwdpt is not None:
                    cip = _annualize_forward(fwdpt, px, self.tenor_months)  # ≈ r_base - r_quote from covered parity
                    # if CIP and spot-based carry disagree a lot, dampen:
                    diff_bps = abs((carry - cip) * 1e4)
                    if diff_bps > FWD_BAND_BPS:
                        # penalty shrinks score towards 0, proportional to disagreement
                        guard = max(0.0, 1.0 - W_FWD_GUARD * (diff_bps - FWD_BAND_BPS) / max(1.0, FWD_BAND_BPS))
            st.fwd_guard = guard

            raw = W_CARRY * carry + W_MOM * mom
            st.score = guard * _squash(raw * 5.0)  # scale then tanh to [-1,1]
            ranking.append((pair, st.score))

        # sort by score desc (best carry/mom longs first)
        ranking.sort(key=lambda x: x[1], reverse=True)

        # Build target weights
        tgt_w = self._target_weights(ranking)
        if not tgt_w:
            return

        # Trade to target notionals (simple: submit target qty; OMS/risk can delta trade)
        per_pair_cap = MAX_PCT_PER_PAIR * GROSS_USD
        for pair, w in tgt_w.items():
            if abs(w) < 1e-9:
                continue
            px = self.sym[pair].px
            if px <= MIN_PRICE:
                continue
            notional = max(-per_pair_cap, min(per_pair_cap, w * GROSS_USD))
            qty_base = notional / px
            side = "buy" if qty_base > 0 else "sell"
            self.order(pair, side, qty=abs(qty_base), order_type="market", venue=VENUE_FX)

        # Emit an aggregate signal for monitoring (average of absolute scores)
        if ranking:
            avg_sig = sum(abs(s) for _, s in ranking) / len(ranking)
            self.emit_signal(max(-1.0, min(1.0, avg_sig)))

    def _target_weights(self, ranking: List[Tuple[str, float]]) -> Dict[str, float]:
        if not ranking:
            return {}
        longs = [p for p, s in ranking if s > 0][:LONG_BUCKET]
        shorts = [p for p, s in reversed(ranking) if s < 0][:SHORT_BUCKET]

        w: Dict[str, float] = {p: 0.0 for p, _ in ranking}
        if NEUTRALIZE and SHORT_BUCKET > 0:
            if longs:
                wl = 0.5 / max(1, len(longs))
                for p in longs: w[p] = +wl
            if shorts:
                ws = 0.5 / max(1, len(shorts))
                for p in shorts: w[p] = -ws
        else:
            if longs:
                wl = 1.0 / max(1, len(longs))
                for p in longs: w[p] = +wl
        return w