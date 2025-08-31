# backend/strategies/diversified/funding_arbitrage.py
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
Funding Arbitrage (Perpetual ↔ Spot)
------------------------------------
Delta‑neutral carry between a perpetual swap (PERP) and spot (SPOT):

If expected net funding (what PERP pays/charges) minus borrow/carry/frictions is
SUFFICIENTLY POSITIVE → LONG SPOT / SHORT PERP  (collect funding)
If SUFFICIENTLY NEGATIVE → SHORT SPOT / LONG PERP  (pay borrow, collect negative funding)

We use a simple expectation:
  exp_funding_apr = blend( last_funding, twap_funding )  (annualized)
  net_edge_apr    = exp_funding_apr  -  borrow_apr  -  fee_apr_guard  -  basis_guard_apr

Enter when net_edge_apr > +ENTRY_APR (or < -ENTRY_APR), with EWMA+z gating on the edge.
Exit when |edge| < EXIT_APR or z small, or after MAX_HOLD_HRS.

Paper routing (map later via adapters):
  • Spot  : "SPOT:<SYM>"
  • Perp  : "PERP:<SYM>@<VENUE>"

Redis you already publish elsewhere in this repo:
  HSET last_price <SYM>                 '{"price": <spot>}'
  HSET perp:mark  <SYM>                 <perp_mark_price>                (optional, for monitoring)
  HSET perp:funding:last_apr <SYM>      <decimal_apr>                    (e.g., 0.36 = 36% APR)
  HSET perp:funding:twap_apr <SYM>      <decimal_apr>
  HSET borrow:apr <SYM>                 <decimal_apr>                    (spot borrow or stablecoin lend)
  (optional) HSET fee:apr  <VENUE>      <decimal_apr>                    (approx taker fees * turnover)
  (optional) HSET basis:apr <SYM>       <decimal_apr>                    (guard for perp-spot basis drift)

This module keeps a single delta‑neutral package per instance and re‑balances if drift > threshold.
"""

# =============================== CONFIG (env) ===============================
REDIS_HOST = os.getenv("FUND_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("FUND_REDIS_PORT", "6379"))

SYM   = os.getenv("FUND_SYMBOL", "BTCUSDT").upper()
VENUE = os.getenv("FUND_VENUE", "BINANCE").upper()   # for paper symbol PERP:<SYM>@<VENUE>

# Thresholds (APR as decimals) and z-gates
ENTRY_APR = float(os.getenv("FUND_ENTRY_APR", "0.10"))  # e.g., 0.10 = 10% annualized
EXIT_APR  = float(os.getenv("FUND_EXIT_APR",  "0.03"))
ENTRY_Z   = float(os.getenv("FUND_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("FUND_EXIT_Z",    "0.4"))

# Sizing / risk
USD_NOTIONAL_PER_SIDE = float(os.getenv("FUND_USD_PER_SIDE", "20000"))
MIN_TICKET_USD        = float(os.getenv("FUND_MIN_TICKET_USD", "200"))
MAX_CONCURRENT        = int(os.getenv("FUND_MAX_CONCURRENT", "1"))

# Hedge upkeep
DELTA_REHEDGE_BPS = float(os.getenv("FUND_DELTA_REHEDGE_BPS", "10"))   # rebalance if |basis| > 10 bps of spot
MAX_HOLD_HRS      = float(os.getenv("FUND_MAX_HOLD_HRS", "72"))        # safety exit after this many hours

# Cadence & stats
RECHECK_SECS = int(os.getenv("FUND_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("FUND_EWMA_ALPHA", "0.06"))

# Redis keys
LAST_PRICE_HKEY      = os.getenv("FUND_LAST_PRICE_KEY", "last_price")
PERP_MARK_HKEY       = os.getenv("FUND_PERP_MARK_KEY", "perp:mark")
FUND_LAST_APR_HKEY   = os.getenv("FUND_LAST_APR_KEY", "perp:funding:last_apr")
FUND_TWAP_APR_HKEY   = os.getenv("FUND_TWAP_APR_KEY", "perp:funding:twap_apr")
BORROW_APR_HKEY      = os.getenv("FUND_BORROW_APR_KEY", "borrow:apr")
FEE_APR_HKEY         = os.getenv("FUND_FEE_APR_KEY", "fee:apr")          # field = VENUE
BASIS_APR_HKEY       = os.getenv("FUND_BASIS_APR_KEY", "basis:apr")

# Venues (advisory)
VENUE_SPOT = os.getenv("FUND_VENUE_SPOT", VENUE)
VENUE_PERP = os.getenv("FUND_VENUE_PERP", VENUE)

# =============================== Redis =====================================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# =============================== Helpers ===================================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _hgetf(hkey: str, field: str) -> Optional[float]:
    v = r.hget(hkey, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _get_fee_apr(venue: str) -> float:
    v = _hgetf(FEE_APR_HKEY, venue)
    return float(v) if v is not None else 0.0

def _basis_apr(sym: str) -> float:
    v = _hgetf(BASIS_APR_HKEY, sym)
    return float(v) if v is not None else 0.0

def _now_ms() -> int:
    return int(time.time() * 1000)

def _blend(a: Optional[float], b: Optional[float], w: float = 0.6) -> Optional[float]:
    if a is None and b is None: return None
    if a is None: return b
    if b is None: return a
    return w * a + (1 - w) * b

# =============================== EWMA =======================================
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

def _ewma_key(sym: str) -> str:
    return f"fund:ewma:{sym}"

def _load_ewma(sym: str) -> EwmaMV:
    raw = r.get(_ewma_key(sym))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(sym: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(sym), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# =============================== State ======================================
@dataclass
class OpenState:
    side: str          # "long_spot_short_perp" or "short_spot_long_perp"
    qty_spot: float
    qty_perp: float
    entry_edge_apr: float
    entry_z: float
    entry_px: float
    ts_ms: int

def _poskey(name: str, sym: str) -> str:
    return f"fund:open:{name}:{sym}"

# =============================== Strategy ===================================
class FundingArbitrage(Strategy):
    """
    Delta‑neutral PERP funding carry vs SPOT with fee/basis guards and z‑score gating.
    """
    def __init__(self, name: str = "funding_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "symbol": SYM, "venue": VENUE, "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # ---------------- core ----------------
    def _inputs(self) -> Optional[Tuple[float, float]]:
        spot = _hget_price(SYM)
        if spot is None or spot <= 0:
            return None
        # funding apr sources
        last_apr = _hgetf(FUND_LAST_APR_HKEY, SYM)
        twap_apr = _hgetf(FUND_TWAP_APR_HKEY, SYM)
        exp_funding_apr = _blend(last_apr, twap_apr, w=0.6)
        if exp_funding_apr is None:
            return None

        borrow_apr = _hgetf(BORROW_APR_HKEY, SYM) or 0.0
        fee_apr = _get_fee_apr(VENUE_PERP) + _get_fee_apr(VENUE_SPOT)
        basis_guard = _basis_apr(SYM)

        net_edge_apr = exp_funding_apr - borrow_apr - fee_apr - basis_guard
        return spot, net_edge_apr

    def _evaluate(self) -> None:
        vals = self._inputs()
        if not vals: return
        spot, edge_apr = vals

        # EWMA + z on *edge_apr*
        ew = _load_ewma(SYM)
        m, v = ew.update(edge_apr)
        _save_ewma(SYM, ew)
        z = (edge_apr - m) / math.sqrt(max(v, 1e-12))

        # monitoring signal: positive when funding edge is favorable for long spot / short perp
        self.emit_signal(max(-1.0, min(1.0, edge_apr / max(1e-6, ENTRY_APR))))

        st = self._load_state()

        # -------- upkeep / exits --------
        if st:
            # time stop
            hrs_open = (time.time()*1000 - st.ts_ms) / (1000*3600)
            if hrs_open >= MAX_HOLD_HRS:
                self._close(st)
                return

            # edge mean‑reversion exit
            if (abs(edge_apr) <= EXIT_APR) or (abs(z) <= EXIT_Z):
                self._close(st)
                return

            # delta re‑hedge if perp mark drifted vs spot (basic basis proxy)
            perp_mark = _hgetf(PERP_MARK_HKEY, SYM)
            if perp_mark is not None and perp_mark > 0:
                basis_pct = (perp_mark - spot) / perp_mark
                if abs(basis_pct) * 1e4 >= DELTA_REHEDGE_BPS:
                    # target delta ~ 0: set |qty_spot| ≈ |qty_perp|
                    # if basis positive, increase the smaller leg
                    target_qty = max(abs(st.qty_spot), abs(st.qty_perp))
                    # rebalance in units (paper: 1 contract ~ 1 coin for simplicity)
                    if abs(abs(st.qty_spot) - target_qty) * spot >= MIN_TICKET_USD:
                        if st.side == "long_spot_short_perp" and abs(st.qty_spot) < target_qty:
                            self.order(f"SPOT:{SYM}", "buy", qty=(target_qty - abs(st.qty_spot)), order_type="market", venue=VENUE_SPOT)
                            st.qty_spot = math.copysign(target_qty, st.qty_spot or 1.0)
                        elif st.side == "short_spot_long_perp" and abs(st.qty_spot) < target_qty:
                            self.order(f"SPOT:{SYM}", "sell", qty=(target_qty - abs(st.qty_spot)), order_type="market", venue=VENUE_SPOT)
                            st.qty_spot = -abs(target_qty)
                        self._save_state(st)
            return

        # -------- entries --------
        if r.get(_poskey(self.ctx.name, SYM)) is not None:
            return
        if not (abs(edge_apr) >= ENTRY_APR and abs(z) >= ENTRY_Z):
            return

        # size:  USD_NOTIONAL_PER_SIDE on each leg
        qty = USD_NOTIONAL_PER_SIDE / max(1e-9, spot)
        if qty * spot < MIN_TICKET_USD:
            return

        if edge_apr > 0:
            # funding positive ⇒ collect: LONG SPOT, SHORT PERP
            self.order(f"SPOT:{SYM}", "buy",  qty=qty, order_type="market", venue=VENUE_SPOT)
            self.order(f"PERP:{SYM}@{VENUE}", "sell", qty=qty, order_type="market", venue=VENUE_PERP)
            side = "long_spot_short_perp"
        else:
            # funding negative ⇒ pay borrow, collect negative funding: SHORT SPOT, LONG PERP
            self.order(f"SPOT:{SYM}", "sell", qty=qty, order_type="market", venue=VENUE_SPOT)
            self.order(f"PERP:{SYM}@{VENUE}", "buy",  qty=qty, order_type="market", venue=VENUE_PERP)
            side = "short_spot_long_perp"

        self._save_state(OpenState(
            side=side, qty_spot=(qty if side.startswith("long_spot") else -qty),
            qty_perp=( -qty if side.startswith("long_spot") else qty ),
            entry_edge_apr=edge_apr, entry_z=z, entry_px=spot, ts_ms=_now_ms()
        ))

    # ---------------- state io ----------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, SYM))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, SYM), json.dumps(st.__dict__))

    # ---------------- close ----------------
    def _close(self, st: OpenState) -> None:
        if st.side == "long_spot_short_perp":
            self.order(f"SPOT:{SYM}", "sell", qty=abs(st.qty_spot), order_type="market", venue=VENUE_SPOT)
            self.order(f"PERP:{SYM}@{VENUE}", "buy",  qty=abs(st.qty_perp), order_type="market", venue=VENUE_PERP)
        else:
            self.order(f"SPOT:{SYM}", "buy",  qty=abs(st.qty_spot), order_type="market", venue=VENUE_SPOT)
            self.order(f"PERP:{SYM}@{VENUE}", "sell", qty=abs(st.qty_perp), order_type="market", venue=VENUE_PERP)
        r.delete(_poskey(self.ctx.name, SYM))