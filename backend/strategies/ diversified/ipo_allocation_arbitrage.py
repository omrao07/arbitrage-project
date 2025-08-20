# backend/strategies/diversified/ipo_allocation_arbitrage.py
from __future__ import annotations

import json, math, os, time, datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
IPO Allocation Arbitrage (paper)
--------------------------------
Idea:
  If the grey/when‑issued (WI) price trades at a rich premium to the IPO offer price,
  and you expect to receive shares on allocation, you can pre‑hedge by SHORTING WI
  ≈ expected allocation. After allocation, reconcile hedge to actual shares and exit
  near listing/open.

Edge (bps over offer):
   edge_bps = 1e4 * ( WI_px / Offer_px - 1 )  -  FEES_BPS  -  RISK_GUARD_BPS

Enter if edge_bps ≥ ENTRY_BPS (premium rich) and we’re prior to allocation time.
Exit (pre‑allocation) if edge mean‑reverts (≤ EXIT_BPS or z small).
On allocation result:
   • If allocated ≥ hedged: keep/trim and exit around listing.
   • If allocated < hedged: buy back the surplus WI immediately (reduce risk).
Hard stop: exit all after LIST_TS + EXIT_SECS.

Symbols (paper → map later in adapters):
  • WI leg   : "WI:<SYM>"      (when‑issued/grey market)
  • Spot leg : "<SYM>"         (post‑listing spot; optional for exit visualization)

Redis feeds you publish elsewhere:
  HSET ipo:offer     <SYM> <offer_price>
  HSET ipo:wi        <SYM> <wi_mid_price>
  HSET ipo:alloc_prob <SYM> <0..1>        # expected probability (retail) or fraction (inst)
  HSET ipo:applied_qty <SYM> <shares>     # how many shares you applied/requested
  HSET ipo:fees_bps   <SYM> <bps>         # combined fees/slippage guard (optional)
  # Event times (ms since epoch). If absent, module falls back to ENV values.
  HSET ipo:schedule:<SYM> alloc_ts_ms <ms>
  HSET ipo:schedule:<SYM> list_ts_ms  <ms>

Allocation result (set when known):
  HSET ipo:allocated_qty <SYM> <shares>

Optional marks for UI (not required for logic):
  HSET last_price "WI:<SYM>" '{"price": <px>}'
  HSET last_price "<SYM>"    '{"price": <px>}'      # post‑listing

ENV config below controls thresholds, times, and safety exits.
"""

# ========================== CONFIG (env) ==========================
REDIS_HOST = os.getenv("IPO_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("IPO_REDIS_PORT", "6379"))

SYM = os.getenv("IPO_SYMBOL", "ACME").upper()

# Times (ms since epoch). Prefer Redis schedule; these are fallbacks.
ALLOC_TS_ENV = os.getenv("IPO_ALLOC_TS_MS", "")      # allocation announcement time
LIST_TS_ENV  = os.getenv("IPO_LIST_TS_MS", "")       # listing/open time
EXIT_SECS_AFTER_LIST = int(os.getenv("IPO_EXIT_SECS", "1800"))  # hard exit X seconds after list

# Thresholds
ENTRY_BPS = float(os.getenv("IPO_ENTRY_BPS", "300.0"))   # need 300 bps premium net of fees
EXIT_BPS  = float(os.getenv("IPO_EXIT_BPS",  "120.0"))
ENTRY_Z   = float(os.getenv("IPO_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("IPO_EXIT_Z",    "0.5"))

# Guards
DEFAULT_FEES_BPS   = float(os.getenv("IPO_FEES_BPS", "40.0"))   # taker + borrow + friction
RISK_GUARD_BPS     = float(os.getenv("IPO_RISK_GUARD_BPS", "60.0"))
MAX_HEDGE_FRACTION = float(os.getenv("IPO_MAX_HEDGE_FRAC", "0.9"))  # <=90% of expected allocation

# Sizing
MIN_TICKET_USD = float(os.getenv("IPO_MIN_TICKET_USD", "200"))
USD_NOTIONAL_CAP = float(os.getenv("IPO_USD_NOTIONAL_CAP", "100000"))  # cap pre‑hedge notional

# Cadence / stats
RECHECK_SECS = int(os.getenv("IPO_RECHECK_SECS", "2"))
EWMA_ALPHA   = float(os.getenv("IPO_EWMA_ALPHA", "0.06"))

# Venues (advisory)
VENUE_WI   = os.getenv("IPO_VENUE_WI", "OTC").upper()
VENUE_SPOT = os.getenv("IPO_VENUE_SPOT", "EXCH").upper()

# Redis keys
OFFER_HKEY      = os.getenv("IPO_OFFER_HKEY", "ipo:offer")          # HSET ipo:offer <SYM> <offer_px>
WI_HKEY         = os.getenv("IPO_WI_HKEY",    "ipo:wi")             # HSET ipo:wi    <SYM> <wi_px>
ALLOC_PROB_HKEY = os.getenv("IPO_ALLOC_PROB", "ipo:alloc_prob")     # HSET ipo:alloc_prob <SYM> <0..1>
APPLIED_QTY_HK  = os.getenv("IPO_APPLIED_QTY", "ipo:applied_qty")   # HSET ipo:applied_qty <SYM> <shares>
FEES_BPS_HKEY   = os.getenv("IPO_FEES_BPS_HKEY", "ipo:fees_bps")    # HSET ipo:fees_bps <SYM> <bps>
SCHED_HASH      = os.getenv("IPO_SCHED_HASH", "ipo:schedule:{sym}") # HSET ipo:schedule:<SYM> alloc_ts_ms <ms> | list_ts_ms <ms>
ALLOC_QTY_HKEY  = os.getenv("IPO_ALLOC_QTY_HKEY", "ipo:allocated_qty")  # HSET ipo:allocated_qty <SYM> <shares>

LAST_PRICE_HKEY = os.getenv("IPO_LAST_PRICE_KEY", "last_price")

# ========================== Redis ==========================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ========================== helpers ==========================
def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _sched_ts(field: str) -> Optional[int]:
    key = SCHED_HASH.format(sym=SYM)
    v = r.hget(key, field)
    if v is None or v == "":  # fallbacks
        env = ALLOC_TS_ENV if field == "alloc_ts_ms" else LIST_TS_ENV
        if not env: return None
        try: return int(env)
        except Exception: return None
    try: return int(v)
    except Exception: return None

def _now_ms() -> int:
    return int(time.time() * 1000)

def _exp_alloc_qty() -> Optional[float]:
    p = _hgetf(ALLOC_PROB_HKEY, SYM)
    q = _hgetf(APPLIED_QTY_HK, SYM)
    if p is None or q is None: return None
    return max(0.0, p * q)

def _fees_bps() -> float:
    v = _hgetf(FEES_BPS_HKEY, SYM)
    return v if v is not None else DEFAULT_FEES_BPS

def _marks() -> Optional[Tuple[float, float]]:
    offer = _hgetf(OFFER_HKEY, SYM)
    wi    = _hgetf(WI_HKEY, SYM)
    if offer is None or wi is None or offer <= 0 or wi <= 0:
        return None
    return offer, wi

# ========================== EWMA ==========================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float):
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        return self.mean, self.var

def _ewma_key() -> str:
    return f"ipoarb:ewma:{SYM}"

def _load_ewma() -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ========================== state ==========================
@dataclass
class OpenState:
    phase: str         # "PRE_ALLOCATION" | "AFTER_ALLOCATION"
    qty_hedge: float   # current WI short quantity (positive absolute)
    exp_alloc_at_entry: float
    entry_edge_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"ipoarb:open:{name}:{SYM}"

# ========================== strategy ==========================
class IPOAllocationArbitrage(Strategy):
    """
    Pre‑hedge WI vs Offer, reconcile to actual allocation, and exit near listing.
    """
    def __init__(self, name: str = "ipo_allocation_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "sym": SYM,
            "alloc_ts_ms": _sched_ts("alloc_ts_ms"),
            "list_ts_ms": _sched_ts("list_ts_ms"),
            "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now
        self._evaluate()

    # --------------- core ---------------
    def _evaluate(self) -> None:
        mx = _marks()
        if not mx: return
        offer, wi = mx
        alloc_ts = _sched_ts("alloc_ts_ms")
        list_ts  = _sched_ts("list_ts_ms")

        # Compute edge
        raw_prem_bps = 1e4 * (wi / offer - 1.0)
        edge_bps = raw_prem_bps - _fees_bps() - RISK_GUARD_BPS

        # Stats on edge
        ew = _load_ewma(); m, v = ew.update(edge_bps); _save_ewma(ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))

        # Emit monitor signal (scaled)
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state()
        now_ms = _now_ms()

        # Hard exit after listing+grace
        if list_ts and now_ms >= (list_ts + 1000 * EXIT_SECS_AFTER_LIST):
            if st: self._close_all(st)
            return

        # If we already have an open package, manage it
        if st:
            if st.phase == "PRE_ALLOCATION":
                # Pre‑allocation risk controls: if edge collapses, flatten
                if (edge_bps <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                    self._close_all(st)
                    return
                # If allocation result arrived early, reconcile
                alloc_qty = _hgetf(ALLOC_QTY_HKEY, SYM)
                if alloc_qty is not None:
                    self._reconcile_after_allocation(st, alloc_qty)
                    return
                # If past allocation time, poll for result; if not available, trim to safer size
                if alloc_ts and now_ms > alloc_ts:
                    alloc_qty = _hgetf(ALLOC_QTY_HKEY, SYM)
                    if alloc_qty is not None:
                        self._reconcile_after_allocation(st, alloc_qty)
                    else:
                        # Safety: reduce hedge to 50% of expected entry size if nothing arrives
                        target = max(0.0, 0.5 * st.exp_alloc_at_entry)
                        if st.qty_hedge > target:
                            self.order(f"WI:{SYM}", "buy", qty=(st.qty_hedge - target), order_type="market", venue=VENUE_WI)
                            st.qty_hedge = target
                            self._save_state(st)
                return

            else:  # AFTER_ALLOCATION phase
                # Wait for listing; optionally tighten hedge as edge mean‑reverts
                if (edge_bps <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                    self._close_all(st)
                    return
                # If listing passed, close
                if list_ts and now_ms >= list_ts:
                    self._close_all(st)
                return

        # -------- entries (no open state) --------
        # Only pre‑allocation entries allowed
        if alloc_ts and now_ms >= alloc_ts:
            return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        exp_alloc = _exp_alloc_qty()
        if exp_alloc is None or exp_alloc <= 0:
            return

        # Hedge size: cap by fraction and notional
        qty = MAX_HEDGE_FRACTION * exp_alloc
        # Notional cap
        if wi * qty > USD_NOTIONAL_CAP:
            qty = USD_NOTIONAL_CAP / wi

        if wi * qty < MIN_TICKET_USD:
            return

        # Enter: SHORT WI
        self.order(f"WI:{SYM}", "sell", qty=qty, order_type="market", venue=VENUE_WI)

        st = OpenState(
            phase="PRE_ALLOCATION",
            qty_hedge=qty,
            exp_alloc_at_entry=exp_alloc,
            entry_edge_bps=edge_bps,
            entry_z=z,
            ts_ms=now_ms
        )
        self._save_state(st)

    # --------------- allocation reconcile ---------------
    def _reconcile_after_allocation(self, st: OpenState, alloc_qty: float) -> None:
        # If allocated less than hedge, buy back surplus now.
        surplus = st.qty_hedge - max(0.0, alloc_qty)
        if surplus > 0:
            self.order(f"WI:{SYM}", "buy", qty=surplus, order_type="market", venue=VENUE_WI)
            st.qty_hedge -= surplus

        # Move to AFTER_ALLOCATION phase (keep remaining hedge until listing/exit)
        st.phase = "AFTER_ALLOCATION"
        self._save_state(st)

    # --------------- state io ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw: return None
        try:
            return OpenState(**json.loads(raw))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name), json.dumps(st.__dict__))

    # --------------- exit / close ---------------
    def _close_all(self, st: OpenState) -> None:
        # Buy back WI short
        if st.qty_hedge > 0:
            self.order(f"WI:{SYM}", "buy", qty=st.qty_hedge, order_type="market", venue=VENUE_WI)
        r.delete(_poskey(self.ctx.name))