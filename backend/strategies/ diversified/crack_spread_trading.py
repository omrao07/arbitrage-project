# backend/strategies/diversified/crack_spread_trading.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Crack Spread Trading (3-2-1 by default)
---------------------------------------
Trades refining margin between crude oil and refined products (gasoline & distillate).
Default: 3-2-1 spread using:
  - CL.*  (WTI Crude, $/bbl)
  - RB.*  (Gasoline RBOB, $/gal)  -> convert to $/bbl via 42 gal/bbl
  - HO.*  (ULSD/Heating Oil, $/gal) -> convert to $/bbl via 42 gal/bbl

Spread (USD per barrel of crude):
    crack = 2 * RB_bbl + 1 * HO_bbl - 3 * CL_bbl

We monitor the spread in USD/bbl, maintain EWMA mean/variance -> z-score,
and enter when BOTH absolute level and z exceed thresholds:
  - If crack >> mean (rich margin): SHORT products basket, LONG crude (short crack)
  - If crack << mean (poor margin): LONG products basket, SHORT crude (long crack)

Sizing converts a USD target per leg to contracts using current prices & contract sizes.

Wiring: Strategy emits orders via Strategy.order() -> risk_manager -> OMS (paper fills).
Inputs required: last prices cached in Redis (HSET last_price <SYM> '{"price": ...}')
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("CRACK_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CRACK_REDIS_PORT", "6379"))

# Symbols (must match your feed/OMS). Use active/front-months or roll-adjusted.
CL_SYMBOL = os.getenv("CRACK_CL", "CL.F1").upper()  # WTI Crude
RB_SYMBOL = os.getenv("CRACK_RB", "RB.F1").upper()  # RBOB Gasoline
HO_SYMBOL = os.getenv("CRACK_HO", "HO.F1").upper()  # ULSD/Heating Oil

# Conversion: gallons per barrel (for RB/HO which quote $/gal)
GAL_PER_BBL = float(os.getenv("CRACK_GAL_PER_BBL", "42"))

# Contract units (per 1 futures contract)
# CME standard: CL=1000 bbl; RB=42,000 gal; HO=42,000 gal
CL_UNITS_BBL = float(os.getenv("CRACK_CL_UNITS_BBL", "1000"))
RB_UNITS_GAL = float(os.getenv("CRACK_RB_UNITS_GAL", "42000"))
HO_UNITS_GAL = float(os.getenv("CRACK_HO_UNITS_GAL", "42000"))

# 3-2-1 coefficients (editable; e.g., 5-3-2 or 2-1-1 for Brent cracks)
COEF_CL = float(os.getenv("CRACK_COEF_CL", "3"))   # barrels of crude
COEF_RB = float(os.getenv("CRACK_COEF_RB", "2"))   # barrels of gasoline
COEF_HO = float(os.getenv("CRACK_COEF_HO", "1"))   # barrels of distillate

# Thresholds (USD per bbl & z-score)
ENTRY_USD = float(os.getenv("CRACK_ENTRY_USD", "2.00"))
EXIT_USD  = float(os.getenv("CRACK_EXIT_USD",  "0.75"))
ENTRY_Z   = float(os.getenv("CRACK_ENTRY_Z",   "2.0"))
EXIT_Z    = float(os.getenv("CRACK_EXIT_Z",    "0.7"))

# Sizing
USD_PER_CRUDE_LEG = float(os.getenv("CRACK_USD_PER_CRUDE_LEG", "40000"))  # target notional on crude leg
MAX_CONCURRENT    = int(os.getenv("CRACK_MAX_CONCURRENT", "1"))

# Cadence & EWMA
RECHECK_SECS = int(os.getenv("CRACK_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("CRACK_EWMA_ALPHA", "0.03"))

# Venue hints (advisory)
VENUE_CL = os.getenv("CRACK_VENUE_CL", "NYMEX").upper()
VENUE_RB = os.getenv("CRACK_VENUE_RB", "NYMEX").upper()
VENUE_HO = os.getenv("CRACK_VENUE_HO", "NYMEX").upper()

# Redis keys your stack already maintains
LAST_PRICE_HKEY = os.getenv("CRACK_LAST_PRICE_KEY", "last_price")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _hget_price(symbol: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, symbol)
    if not raw: return None
    try: return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

# ============================ EWMA tracker ============================
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

def _ewma_key() -> str:
    return f"crack:ewma:{CL_SYMBOL}:{RB_SYMBOL}:{HO_SYMBOL}:{int(COEF_CL)}-{int(COEF_RB)}-{int(COEF_HO)}"

def _load_ewma(alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ State ============================
def _poskey(name: str) -> str:
    return f"crack:open:{name}:{CL_SYMBOL}:{RB_SYMBOL}:{HO_SYMBOL}"

@dataclass
class OpenState:
    side: str              # "long_crack" or "short_crack"
    q_cl: float            # contracts crude
    q_rb: float            # contracts gasoline
    q_ho: float            # contracts distillate
    entry_spread: float    # USD/bbl
    entry_z: float
    ts_ms: int

# ============================ Core helpers ============================
def _rb_bbl(px_rb_per_gal: float) -> float:
    return px_rb_per_gal * GAL_PER_BBL

def _ho_bbl(px_ho_per_gal: float) -> float:
    return px_ho_per_gal * GAL_PER_BBL

def _crack_usd_per_bbl(px_cl_bbl: float, px_rb_gal: float, px_ho_gal: float) -> float:
    # crack = RB*coef + HO*coef - CL*coef, all in $/bbl terms
    rb_bbl = _rb_bbl(px_rb_gal)
    ho_bbl = _ho_bbl(px_ho_gal)
    return COEF_RB * rb_bbl + COEF_HO * ho_bbl - COEF_CL * px_cl_bbl

def _contracts_for_usd(usdn: float, px: float, units: float) -> float:
    """
    Convert target USD notional to #contracts given price (USD per unit) and contract units.
    """
    denom = max(px * units, 1e-9)
    return usdn / denom

# ============================ Strategy ============================
class CrackSpreadTrading(Strategy):
    """
    Trades the refining margin (crack spread) using crude vs refined product futures.
    Long-crack = LONG products (RB, HO), SHORT crude.
    Short-crack = SHORT products, LONG crude.
    """

    def __init__(self, name: str = "crack_spread_trading", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        # publish for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "cl": CL_SYMBOL, "rb": RB_SYMBOL, "ho": HO_SYMBOL,
            "coef": {"cl": COEF_CL, "rb": COEF_RB, "ho": COEF_HO},
            "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate()

    # ---------- engine ----------
    def _evaluate(self) -> None:
        cl = _hget_price(CL_SYMBOL)  # $/bbl
        rb = _hget_price(RB_SYMBOL)  # $/gal
        ho = _hget_price(HO_SYMBOL)  # $/gal
        if cl is None or rb is None or ho is None:
            return

        spread = _crack_usd_per_bbl(px_cl_bbl=float(cl), px_rb_gal=float(rb), px_ho_gal=float(ho))

        ew = _load_ewma(EWMA_ALPHA)
        m, v = ew.update(spread)
        _save_ewma(ew)
        z = (spread - m) / math.sqrt(max(v, 1e-12))

        # expose normalized signal (for dashboards/allocator)
        self.emit_signal(max(-1.0, min(1.0, math.tanh((spread - m) / 5.0))))

        st = self._load_state()

        # -------- exits first --------
        if st:
            if (abs(spread - m) <= EXIT_USD) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # -------- entries --------
        if r.get(_poskey(self.ctx.name)) is not None:  # concurrency guard
            return

        if abs(spread - m) >= ENTRY_USD and abs(z) >= ENTRY_Z:
            # size legs proportionally to classic 3-2-1 notionals
            # crude leg sized by USD_PER_CRUDE_LEG, products sized to match coefficients ratio
            q_cl = _contracts_for_usd(USD_PER_CRUDE_LEG, px=float(cl), units=CL_UNITS_BBL)
            # For notional matching: scale products so that crude notional corresponds to COEF_CL barrels.
            # Compute equivalent USD per bbl products legs to match coefficients ratio:
            rb_bbl = _rb_bbl(float(rb))
            ho_bbl = _ho_bbl(float(ho))
            # Derive product notional targets relative to crude notional using coefficients:
            # Let K be per-bbl crude notional = q_cl * CL_UNITS_BBL * cl / COEF_CL barrels “bundle”.
            crude_bundle_bbl = COEF_CL  # definition of a bundle
            crude_bundle_usd = cl * crude_bundle_bbl
            # scale factor so that product contracts approximate bundle components:
            # Target USD per bundle for RB part:
            rb_usd_target = COEF_RB * rb_bbl
            ho_usd_target = COEF_HO * ho_bbl
            # Convert to contracts using contract units:
            q_rb = _contracts_for_usd((rb_usd_target / crude_bundle_usd) * (q_cl * CL_UNITS_BBL * cl), px=float(rb), units=RB_UNITS_GAL)
            q_ho = _contracts_for_usd((ho_usd_target / crude_bundle_usd) * (q_cl * CL_UNITS_BBL * cl), px=float(ho), units=HO_UNITS_GAL)

            # Direction:
            if spread < m:  # crack cheap -> LONG crack: LONG RB & HO, SHORT CL
                self.order(RB_SYMBOL, "buy",  qty=abs(q_rb), order_type="market", venue=VENUE_RB)
                self.order(HO_SYMBOL, "buy",  qty=abs(q_ho), order_type="market", venue=VENUE_HO)
                self.order(CL_SYMBOL, "sell", qty=abs(q_cl), order_type="market", venue=VENUE_CL)
                self._save_state(OpenState(
                    side="long_crack",
                    q_cl=abs(q_cl), q_rb=abs(q_rb), q_ho=abs(q_ho),
                    entry_spread=spread, entry_z=z, ts_ms=int(time.time()*1000)
                ))
            else:           # crack rich -> SHORT crack: SHORT RB & HO, LONG CL
                self.order(RB_SYMBOL, "sell", qty=abs(q_rb), order_type="market", venue=VENUE_RB)
                self.order(HO_SYMBOL, "sell", qty=abs(q_ho), order_type="market", venue=VENUE_HO)
                self.order(CL_SYMBOL, "buy",  qty=abs(q_cl), order_type="market", venue=VENUE_CL)
                self._save_state(OpenState(
                    side="short_crack",
                    q_cl=abs(q_cl), q_rb=abs(q_rb), q_ho=abs(q_ho),
                    entry_spread=spread, entry_z=z, ts_ms=int(time.time()*1000)
                ))

    # ---------- state helpers ----------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw:
            return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # ---------- closing ----------
    def _close(self, st: OpenState) -> None:
        if st.side == "long_crack":
            # close: sell products, buy crude
            self.order(RB_SYMBOL, "sell", qty=st.q_rb, order_type="market", venue=VENUE_RB)
            self.order(HO_SYMBOL, "sell", qty=st.q_ho, order_type="market", venue=VENUE_HO)
            self.order(CL_SYMBOL, "buy",  qty=st.q_cl, order_type="market", venue=VENUE_CL)
        else:
            # close: buy products, sell crude
            self.order(RB_SYMBOL, "buy",  qty=st.q_rb, order_type="market", venue=VENUE_RB)
            self.order(HO_SYMBOL, "buy",  qty=st.q_ho, order_type="market", venue=VENUE_HO)
            self.order(CL_SYMBOL, "sell", qty=st.q_cl, order_type="market", venue=VENUE_CL)
        self._save_state(None)