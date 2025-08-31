# backend/strategies/diversified/insurance_linked_securities_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Insurance-Linked Securities (ILS) Arbitrage (paper)
---------------------------------------------------
Idea:
  Compare the cat bond’s *running spread carry* to its *expected loss (EL) + risk load*.
  If carry >> EL+load → bond is "rich" to risk (overpaying) ⇒ SHORT bond / LONG index hedge.
  If carry << EL+load → bond "cheap" ⇒ LONG bond / SHORT index hedge.

We work in annualized decimals and bps on the edge:

  carry = coupon - risk_free         # both annualized decimals
  fair  = EL + LOAD + FEES_GUARD
  edge  = carry - fair               # >0 rich; <0 cheap
  edge_bps = 1e4 * edge

Hedges:
  • Use an ILS index future or total-return proxy (paper symbol).
  • Hedge ratio (beta) from Redis or ENV.

Redis feeds you publish elsewhere:
  HSET ils:coupon <BOND> <decimal>        # running coupon (annualized); e.g., 0.115 = 11.5%
  HSET ils:el     <BOND> <decimal>        # expected loss per annum (EL)
  HSET ils:load   <BOND> <decimal>        # risk load / liquidity premium (optional, else 0)
  HSET rate:risk_free:<CCY> <CCY> <r>     # risk-free; e.g., USD 0.045
  (optional) HSET ils:beta <BOND> <beta>  # hedge beta vs index (default 0.8)
  (optional) HSET ils:fees <BOND> <decimal>  # platform/expense guard added to fair

Marks for paper routing (not used in edge but for OMS visibility):
  HSET last_price <SYM> '{"price": <px>}'   # e.g., CATBOND:<ID> and ILSX:<INDEX>
Symbols (paper → map in adapters later):
  • Bond leg : "CATBOND:<ID>"
  • Index    : "ILSX:<INDEX>"   (SwissRe-style cat bond index proxy or custom ILS basket)

Entries are gated by absolute edge (bps) and z-score on EWMA(edge).
Exits when |edge| shrinks or z reverts. Restart-safe via Redis.
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("ILS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("ILS_REDIS_PORT", "6379"))

BOND_ID   = os.getenv("ILS_BOND_ID", "CAT2025A").upper()           # your cat bond identifier
BOND_SYM  = os.getenv("ILS_BOND_SYM", f"CATBOND:{BOND_ID}").upper()
INDEX_SYM = os.getenv("ILS_INDEX_SYM", "ILSX:GLOBAL").upper()
CCY       = os.getenv("ILS_CCY", "USD").upper()

# Thresholds
ENTRY_BPS = float(os.getenv("ILS_ENTRY_BPS", "40.0"))   # need >= 40 bps edge to act
EXIT_BPS  = float(os.getenv("ILS_EXIT_BPS",  "15.0"))
ENTRY_Z   = float(os.getenv("ILS_ENTRY_Z",   "1.3"))
EXIT_Z    = float(os.getenv("ILS_EXIT_Z",    "0.5"))

# Sizing / hedge
USD_NOTIONAL_BOND = float(os.getenv("ILS_USD_BOND", "100000"))   # face to trade per entry
USD_NOTIONAL_HEDGE = float(os.getenv("ILS_USD_HEDGE", "80000"))  # base hedge notional (scaled by beta)
BETA_DEFAULT = float(os.getenv("ILS_BETA", "0.80"))
MIN_TICKET_USD = float(os.getenv("ILS_MIN_TICKET_USD", "500"))
MAX_CONCURRENT = int(os.getenv("ILS_MAX_CONCURRENT", "1"))

# Cadence / stats
RECHECK_SECS = int(os.getenv("ILS_RECHECK_SECS", "10"))
EWMA_ALPHA   = float(os.getenv("ILS_EWMA_ALPHA", "0.06"))

# Venues (advisory)
VENUE_BOND  = os.getenv("ILS_VENUE_BOND", "OTC").upper()
VENUE_INDEX = os.getenv("ILS_VENUE_INDEX", "OTC").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("ILS_LAST_PRICE_KEY", "last_price")

COUPON_HKEY = os.getenv("ILS_COUPON_KEY", "ils:coupon")     # HSET ils:coupon <BOND_ID> <decimal>
EL_HKEY     = os.getenv("ILS_EL_KEY",     "ils:el")         # HSET ils:el     <BOND_ID> <decimal>
LOAD_HKEY   = os.getenv("ILS_LOAD_KEY",   "ils:load")       # HSET ils:load   <BOND_ID> <decimal>
FEES_HKEY   = os.getenv("ILS_FEES_KEY",   "ils:fees")       # HSET ils:fees   <BOND_ID> <decimal>
BETA_HKEY   = os.getenv("ILS_BETA_KEY",   "ils:beta")       # HSET ils:beta   <BOND_ID> <beta>
RATE_FMT    = os.getenv("ILS_RATE_FMT",   "rate:risk_free:{ccy}")  # HSET rate:risk_free:USD USD 0.045

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hgetf(hkey: str, field: str) -> Optional[float]:
    v = r.hget(hkey, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _rate(ccy: str) -> float:
    v = _hgetf(RATE_FMT.format(ccy=ccy), ccy)
    return float(v) if v is not None else 0.02

def _beta(bond_id: str) -> float:
    v = _hgetf(BETA_HKEY, bond_id)
    try: return float(v) if v is not None else BETA_DEFAULT
    except Exception: return BETA_DEFAULT

def _get_edge_bps(bond_id: str) -> Optional[float]:
    coupon = _hgetf(COUPON_HKEY, bond_id)   # annualized decimal
    el     = _hgetf(EL_HKEY, bond_id)       # annualized decimal
    rf     = _rate(CCY)
    carry  = None if coupon is None else (coupon - rf)
    if carry is None or el is None:
        return None
    load   = _hgetf(LOAD_HKEY, bond_id) or 0.0
    fees   = _hgetf(FEES_HKEY, bond_id) or 0.0
    fair   = el + load + fees
    edge   = carry - fair
    return 1e4 * edge

# ============================ EWMA ============================
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

def _ewma_key(bond_id: str) -> str:
    return f"ils:ewma:{bond_id}"

def _load_ewma(bond_id: str) -> EwmaMV:
    raw = r.get(_ewma_key(bond_id))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(bond_id: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(bond_id), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    side: str  # "short_bond_long_idx" or "long_bond_short_idx"
    qty_bond: float
    qty_index: float
    entry_edge_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, bond_id: str) -> str:
    return f"ils:open:{name}:{bond_id}"

# ============================ strategy ============================
class InsuranceLinkedSecuritiesArbitrage(Strategy):
    """
    Carry vs EL+load mispricing trade: CATBOND:<ID> vs ILSX:<INDEX> with beta hedge.
    """
    def __init__(self, name: str = "insurance_linked_securities_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "bond": BOND_SYM, "index": INDEX_SYM, "ccy": CCY, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    def _evaluate(self) -> None:
        edge_bps = _get_edge_bps(BOND_ID)
        if edge_bps is None:
            return

        ew = _load_ewma(BOND_ID)
        m, v = ew.update(edge_bps)
        _save_ewma(BOND_ID, ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))

        # monitoring signal: positive when bond is "rich" (carry > fair)
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state()

        # ----- exits -----
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ----- entries -----
        if r.get(_poskey(self.ctx.name, BOND_ID)) is not None:
            return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        # sizing: use notionals + beta for index hedge
        beta = _beta(BOND_ID)
        qty_bond  = USD_NOTIONAL_BOND
        qty_index = USD_NOTIONAL_HEDGE * beta

        if qty_bond < MIN_TICKET_USD or qty_index < MIN_TICKET_USD:
            return

        if edge_bps > 0:
            # Bond rich ⇒ SHORT bond / LONG index
            self.order(BOND_SYM,  "sell", qty=qty_bond,  order_type="market", venue=VENUE_BOND)
            self.order(INDEX_SYM, "buy",  qty=qty_index, order_type="market", venue=VENUE_INDEX)
            side = "short_bond_long_idx"
        else:
            # Bond cheap ⇒ LONG bond / SHORT index
            self.order(BOND_SYM,  "buy",  qty=qty_bond,  order_type="market", venue=VENUE_BOND)
            self.order(INDEX_SYM, "sell", qty=qty_index, order_type="market", venue=VENUE_INDEX)
            side = "long_bond_short_idx"

        self._save_state(OpenState(
            side=side, qty_bond=qty_bond, qty_index=qty_index,
            entry_edge_bps=edge_bps, entry_z=z, ts_ms=int(time.time()*1000)
        ))

    # --------------- state io ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, BOND_ID))
        if not raw: return None
        try:
            return OpenState(**json.loads(raw)) # type: ignore
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, BOND_ID), json.dumps(st.__dict__))

    # --------------- close ---------------
    def _close(self, st: OpenState) -> None:
        if st.side == "short_bond_long_idx":
            self.order(BOND_SYM,  "buy",  qty=st.qty_bond,  order_type="market", venue=VENUE_BOND)
            self.order(INDEX_SYM, "sell", qty=st.qty_index, order_type="market", venue=VENUE_INDEX)
        else:
            self.order(BOND_SYM,  "sell", qty=st.qty_bond,  order_type="market", venue=VENUE_BOND)
            self.order(INDEX_SYM, "buy",  qty=st.qty_index, order_type="market", venue=VENUE_INDEX)
        r.delete(_poskey(self.ctx.name, BOND_ID))