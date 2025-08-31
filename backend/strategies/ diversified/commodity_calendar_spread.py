# backend/strategies/diversified/commodity_calendar_spread.py
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
Commodity Calendar Spread (relative value)
-----------------------------------------
Trades the shape of the futures curve (near vs far) for one or more commodities.
Works in paper mode with your OMS and Redis wiring.

Mechanics
---------
1) Normalize per-contract prices (USD per unit). (If your feed is already USD, FX=1.)
2) For each pair (near, far), compute spread = far - near (USD/unit).
3) Maintain EWMA mean/variance → z-score.
4) Enter when BOTH absolute spread and z-score exceed thresholds:
      - If spread >> mean (contango rich): SHORT far, LONG near
      - If spread << mean (deep backwardation): LONG far, SHORT near
5) Exit when |spread| <= EXIT_USD OR |z| <= EXIT_Z.

Supports multiple commodities at once (e.g., CL, RB, HO, NG, C, S, W, HG, SI).
"""

# ============================== CONFIG ==============================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Pairs to trade (near, far). Symbols must match your market data/OMS.
# Examples:
#   "CL.F24" = WTI Jan 2024, "CL.G24" = Feb 2024 (ICE/CME your convention)
#   "NG.H25" = NatGas Mar 2025, etc.
PAIR_LIST = [s.strip().upper() for s in os.getenv(
    "CCAL_PAIRS",
    "CL.F25,CL.G25;RB.F25,RB.G25;NG.F25,NG.G25"
).split(";") if s.strip()]

# Contract units (how many units per 1 futures contract) for sizing
# E.g., CL=1000 bbl, RB=42000 gal, NG=10000 mmBtu, HG=25000 lbs, SI=5000 oz
CONTRACT_UNITS = {
    "CL": float(os.getenv("CCAL_CL_UNITS", "1000")),
    "RB": float(os.getenv("CCAL_RB_UNITS", "42000")),
    "HO": float(os.getenv("CCAL_HO_UNITS", "42000")),
    "NG": float(os.getenv("CCAL_NG_UNITS", "10000")),
    "GC": float(os.getenv("CCAL_GC_UNITS", "100")),
    "SI": float(os.getenv("CCAL_SI_UNITS", "5000")),
    "HG": float(os.getenv("CCAL_HG_UNITS", "25000")),
    "C":  float(os.getenv("CCAL_C_UNITS",  "5000")),
    "S":  float(os.getenv("CCAL_S_UNITS",  "5000")),
    "W":  float(os.getenv("CCAL_W_UNITS",  "5000")),
}

# FX normalization (prefix -> USD FX pair); set to USDUSD if already USD
FX_FOR = {
    "CL": os.getenv("CCAL_CL_FX", "USDUSD").upper(),
    "RB": os.getenv("CCAL_RB_FX", "USDUSD").upper(),
    "HO": os.getenv("CCAL_HO_FX", "USDUSD").upper(),
    "NG": os.getenv("CCAL_NG_FX", "USDUSD").upper(),
    "GC": os.getenv("CCAL_GC_FX", "USDUSD").upper(),
    "SI": os.getenv("CCAL_SI_FX", "USDUSD").upper(),
    "HG": os.getenv("CCAL_HG_FX", "USDUSD").upper(),
    "C":  os.getenv("CCAL_C_FX",  "USDUSD").upper(),
    "S":  os.getenv("CCAL_S_FX",  "USDUSD").upper(),
    "W":  os.getenv("CCAL_W_FX",  "USDUSD").upper(),
}

# Entry/exit thresholds
ENTRY_USD = float(os.getenv("CCAL_ENTRY_USD", "0.20"))  # abs USD/unit spread
EXIT_USD  = float(os.getenv("CCAL_EXIT_USD",  "0.05"))
ENTRY_Z   = float(os.getenv("CCAL_ENTRY_Z",   "2.0"))
EXIT_Z    = float(os.getenv("CCAL_EXIT_Z",    "0.7"))

# Sizing
USD_PER_LEG   = float(os.getenv("CCAL_USD_PER_LEG", "30000"))  # notional per leg
MAX_CONCURRENT = int(os.getenv("CCAL_MAX_CONCURRENT", "4"))

# EWMA (events-based)
EWMA_ALPHA = float(os.getenv("CCAL_EWMA_ALPHA", "0.02"))  # 2% per tick

# Seasonality/carry guard (block entries near seasonal roll/carry zones)
USE_CARRY_GUARD = os.getenv("CCAL_USE_CARRY_GUARD", "true").lower() in ("1","true","yes")
CARRY_CAP_USD   = float(os.getenv("CCAL_CARRY_CAP_USD", "0.30"))  # if |spread| < carry cap, skip entries

# Rebalance cadence (seconds)
RECHECK_SECS = int(os.getenv("CCAL_RECHECK_SECS", "10"))

# Venue hint (advisory)
VENUE_HINTS = {
    "CL": "NYMEX", "RB": "NYMEX", "HO": "NYMEX", "NG": "NYMEX",
    "GC": "COMEX", "SI": "COMEX", "HG": "COMEX",
    "C": "CBOT", "S": "CBOT", "W": "CBOT"
}

# Redis keys your stack already uses
LAST_PRICE_HKEY = os.getenv("CCAL_LAST_PRICE_KEY", "last_price")   # HSET symbol -> {"price": ...}
FX_SPOT_HKEY    = os.getenv("CCAL_FX_SPOT_KEY",    "fx:spot")      # HSET "EURUSD" -> 1.095
# ===================================================================

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ============================== HELPERS ==============================
def _prefix(sym: str) -> str:
    return sym.split(".", 1)[0].upper()

def _venue(sym: str) -> Optional[str]:
    return VENUE_HINTS.get(_prefix(sym))

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

def _fx(pair: str) -> float:
    if pair == "USDUSD":
        return 1.0
    v = r.hget(FX_SPOT_HKEY, pair)
    if v:
        try:
            return float(v) # type: ignore
        except Exception:
            pass
    # allow last_price for FX as fallback
    px = _hget_last(pair)
    return float(px or 1.0)

def _units(sym: string) -> float:  # type: ignore[name-defined]
    return CONTRACT_UNITS.get(_prefix(sym), 1.0)

def _usd_price(sym: str) -> Optional[float]:
    px = _hget_last(sym)
    if px is None: return None
    fx = _fx(FX_FOR.get(_prefix(sym), "USDUSD"))
    return float(px) * float(fx)

def _carry_guard_usd(days: int = 90, r_quote: float = 0.05, storage: float = 0.0) -> float:
    return (r_quote + storage) * (days / 365.0)

def _ewma_key(near: str, far: str) -> str:
    return f"ccal:ewma:{near}-{far}"

def _poskey(name: str, near: str, far: str) -> str:
    return f"ccal:open:{name}:{near}-{far}"

# ============================== EWMA MV ==============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float

    def update(self, x: float) -> Tuple[float, float]:
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        return self.mean, self.var

def _load_ewma(pair: Tuple[str, str], alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key(*pair))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(pair: Tuple[str, str], ew: EwmaMV) -> None:
    r.set(_ewma_key(*pair), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================== STRATEGY ==============================
@dataclass
class PairState:
    has_pos: bool = False
    side: str = ""            # "long_far_short_near" or "short_far_long_near"
    qty_far: float = 0.0
    qty_near: float = 0.0
    entry_spread: float = 0.0
    entry_z: float = 0.0
    ts_ms: int = 0

class CommodityCalendarSpread(Strategy):
    """
    Calendar spread: trade far minus near on threshold & z-score.
    """

    def __init__(self, name: str = "commodity_calendar_spread", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        # parse env PAIR_LIST -> [(near, far), ...]
        self.pairs: List[Tuple[str, str]] = []
        for p in PAIR_LIST:
            try:
                a, b = [x.strip().upper() for x in p.split(",")]
                self.pairs.append((a, b))
            except Exception:
                continue
        self.last_check = 0.0

    # ------------ lifecycle ------------
    def on_start(self) -> None:
        super().on_start()
        # advertise universe for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({"calendar_pairs": self.pairs, "ts": int(time.time()*1000)}))

    # ------------ core tick ------------
    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate_all()

    # ------------ engine ------------
    def _evaluate_all(self) -> None:
        open_count = sum(1 for pr in self.pairs if r.get(_poskey(self.ctx.name, *pr)))
        for near, far in self.pairs:
            pn = _usd_price(near)
            pf = _usd_price(far)
            if pn is None or pf is None:
                continue

            spread = pf - pn  # USD per unit (far - near)
            ew = _load_ewma((near, far), EWMA_ALPHA)
            m, v = ew.update(spread)
            _save_ewma((near, far), ew)
            z = (spread - m) / math.sqrt(max(v, 1e-12))

            st = self._load_state(near, far)

            # Exit first
            if st:
                if (abs(spread) <= EXIT_USD) or (abs(z) <= EXIT_Z):
                    self._close_pair(near, far, st)
                continue

            # Entry checks
            if open_count >= MAX_CONCURRENT:
                continue
            if not (abs(spread) >= ENTRY_USD and abs(z) >= ENTRY_Z):
                continue

            if USE_CARRY_GUARD:
                if abs(spread) < min(CARRY_CAP_USD, _carry_guard_usd(days=60)):
                    continue

            # Sizing: convert USD notionals to contracts per leg
            units_n = _units(near)
            units_f = _units(far)
            if units_n <= 0 or units_f <= 0:
                continue

            # contracts = (USD notional per leg) / (price (USD/unit) * units per contract)
            qn = USD_PER_LEG / max(pn * units_n, 1e-9)
            qf = USD_PER_LEG / max(pf * units_f, 1e-9)

            if spread > 0:
                # contango rich -> SHORT far, LONG near
                self.order(far, "sell", qty=qf, order_type="market", venue=_venue(far))
                self.order(near, "buy",  qty=qn, order_type="market", venue=_venue(near))
                self._save_state(near, far, PairState(
                    has_pos=True, side="short_far_long_near", qty_far=qf, qty_near=qn,
                    entry_spread=spread, entry_z=z, ts_ms=int(time.time()*1000)
                ))
            else:
                # backwardation extreme -> LONG far, SHORT near
                self.order(far, "buy",  qty=qf, order_type="market", venue=_venue(far))
                self.order(near, "sell", qty=qn, order_type="market", venue=_venue(near))
                self._save_state(near, far, PairState(
                    has_pos=True, side="long_far_short_near", qty_far=qf, qty_near=qn,
                    entry_spread=spread, entry_z=z, ts_ms=int(time.time()*1000)
                ))
            open_count += 1

    # ------------ state/persistence ------------
    def _load_state(self, near: str, far: str) -> Optional[PairState]:
        raw = r.get(_poskey(self.ctx.name, near, far))
        if not raw:
            return None
        try:
            o = json.loads(raw) # type: ignore
            return PairState(**o)
        except Exception:
            return None

    def _save_state(self, near: str, far: str, st: Optional[PairState]) -> None:
        k = _poskey(self.ctx.name, near, far)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # ------------ closing ------------
    def _close_pair(self, near: str, far: str, st: PairState) -> None:
        if not st or not st.has_pos:
            self._save_state(near, far, None)
            return
        if st.side == "short_far_long_near":
            self.order(far, "buy",  qty=st.qty_far, order_type="market", venue=_venue(far))
            self.order(near, "sell", qty=st.qty_near, order_type="market", venue=_venue(near))
        else:
            self.order(far, "sell", qty=st.qty_far, order_type="market", venue=_venue(far))
            self.order(near, "buy",  qty=st.qty_near, order_type="market", venue=_venue(near))
        self._save_state(near, far, None)