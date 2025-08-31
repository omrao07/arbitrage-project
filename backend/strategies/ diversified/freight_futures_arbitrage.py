# backend/strategies/diversified/freight_futures_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis

from backend.engine.strategy_base import Strategy

"""
Freight Futures Arbitrage (paper)
---------------------------------
Two arbitrages commonly used in FFAs (dry bulk / tankers):

A) ROUTE SPREAD
   Compare two routes normalized by a hedge beta:
      S = PX_A  -  beta * PX_B  (+ optional basis guard)
   If S is rich/high -> SHORT A / LONG B
   If S is cheap/low -> LONG A  / SHORT B

   Examples:
     • Dry bulk Capesize: C5 (W Australia->Qingdao) vs C3 (Tubarao->Qingdao)
     • Panamax P1A (TA round) vs P2A (Pacific round)

B) CALENDAR SPREAD
   Same route, two expiries:
      S = PX_NEAR  -  PX_FAR  (+ optional carry guard)
   Mean‑reverting gate on |S - EWMA_mean| and z‑score.

All prices are read from Redis as *index points* or *USD/ton/day*. You also publish
a **point value** to convert points → USD per 1 contract, so we can size lots correctly.

Paper symbols you route later in your adapter (examples):
  • "FFA:C5:2025M09@SGX"
  • "FFA:C3:2025M09@SGX"
  • "FFA:P1A:2025M10@SGX"

Redis you publish elsewhere:
  HSET last_price <SYMBOL> '{"price": <px>}'         # for each leg symbol
  HSET ffa:ptval  <SYMBOL> <usd_per_point_per_lot>   # point value (USD per 1.00 price point per lot)
  (optional) HSET ffa:beta   <PAIRKEY> <beta>        # e.g., "C5_C3" -> 0.85
  (optional) HSET ffa:basis  <PAIRKEY> <guard_pts>   # conservative add to spread (route) or carry (calendar)

ENV selects MODE & legs. See config block below.
"""

# ============================== CONFIG (env) ==============================
REDIS_HOST = os.getenv("FFA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("FFA_REDIS_PORT", "6379"))

MODE = os.getenv("FFA_MODE", "ROUTE").upper()  # "ROUTE" or "CALENDAR"

# ROUTE mode legs (full symbols, not just codes)
LEG_A = os.getenv("FFA_LEG_A", "FFA:C5:2025M09@SGX").upper()
LEG_B = os.getenv("FFA_LEG_B", "FFA:C3:2025M09@SGX").upper()
PAIR_KEY = os.getenv("FFA_PAIR_KEY", "C5_C3").upper()  # used to lookup beta/basis guards in Redis

# CALENDAR mode legs (same route, two expiries)
LEG_NEAR = os.getenv("FFA_LEG_NEAR", "FFA:C5:2025M09@SGX").upper()
LEG_FAR  = os.getenv("FFA_LEG_FAR",  "FFA:C5:2025M12@SGX").upper()
CAL_KEY  = os.getenv("FFA_CAL_KEY",  "C5_CAL").upper()

# Thresholds (points and z-score on spread deviation)
ENTRY_PTS = float(os.getenv("FFA_ENTRY_PTS", "0.8"))
EXIT_PTS  = float(os.getenv("FFA_EXIT_PTS",  "0.3"))
ENTRY_Z   = float(os.getenv("FFA_ENTRY_Z",   "1.4"))
EXIT_Z    = float(os.getenv("FFA_EXIT_Z",    "0.5"))

# Sizing
USD_NOTIONAL_PER_LEG = float(os.getenv("FFA_USD_PER_LEG", "30000"))
MIN_TICKET_USD       = float(os.getenv("FFA_MIN_TICKET_USD", "200"))
MAX_CONCURRENT       = int(os.getenv("FFA_MAX_CONCURRENT", "1"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("FFA_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("FFA_EWMA_ALPHA", "0.06"))

# Venues (advisory)
VENUE = os.getenv("FFA_VENUE", "SGX").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("FFA_LAST_PRICE_KEY", "last_price")       # HSET last_price <SYM> -> {"price": ...}
PTVAL_HKEY      = os.getenv("FFA_PTVAL_KEY", "ffa:ptval")             # HSET ffa:ptval <SYM> <usd_per_point_per_lot>
BETA_HKEY       = os.getenv("FFA_BETA_KEY", "ffa:beta")               # HSET ffa:beta <PAIRKEY> <beta>
BASIS_HKEY      = os.getenv("FFA_BASIS_KEY", "ffa:basis")             # HSET ffa:basis <PAIRKEY> <pts_guard>

# ============================== Redis ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== helpers ==============================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try:
        return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _ptval(sym: str) -> Optional[float]:
    v = r.hget(PTVAL_HKEY, sym)
    if v is None: return None
    try:
        return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _beta(pair_key: str, default: float) -> float:
    v = r.hget(BETA_HKEY, pair_key)
    try:
        return float(v) if v is not None else default # type: ignore
    except Exception:
        return default

def _basis_guard(pair_key: str) -> float:
    v = r.hget(BASIS_HKEY, pair_key)
    try:
        return float(v) if v is not None else 0.0 # type: ignore
    except Exception:
        return 0.0

def _now_ms() -> int:
    return int(time.time() * 1000)

# ============================== EWMA ==============================
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

def _ewma_key(name: str) -> str:
    return f"ffa:ewma:{name}"

def _load_ewma(name: str) -> EwmaMV:
    raw = r.get(_ewma_key(name))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(name: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(name), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================== state ==============================
@dataclass
class OpenState:
    side: str  # "short_spread" or "long_spread"
    qa: float
    qb: float
    entry_pts: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"ffa:open:{name}:{MODE}"

# ============================== strategy ==============================
class FreightFuturesArbitrage(Strategy):
    """
    Route or Calendar spread arb with point‑value aware sizing and EWMA+z gating.
    """
    def __init__(self, name: str = "freight_futures_arbitrage", region: Optional[str] = "EU", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        if MODE == "ROUTE":
            universe = {"mode": MODE, "A": LEG_A, "B": LEG_B, "pair": PAIR_KEY}
        else:
            universe = {"mode": MODE, "NEAR": LEG_NEAR, "FAR": LEG_FAR, "pair": CAL_KEY}
        r.hset("strategy:universe", self.ctx.name, json.dumps({**universe, "ts": _now_ms()}))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        if MODE == "ROUTE":
            self._route_eval()
        else:
            self._calendar_eval()

    # ---------------- ROUTE spread ----------------
    def _route_eval(self) -> None:
        pa = _hget_price(LEG_A)
        pb = _hget_price(LEG_B)
        if pa is None or pb is None or pa <= 0 or pb <= 0:
            return
        beta = _beta(PAIR_KEY, 1.0)
        guard = _basis_guard(PAIR_KEY)
        spread = (pa - beta * pb) + guard

        ew = _load_ewma(f"ROUTE:{PAIR_KEY}")
        m, v = ew.update(spread)
        _save_ewma(f"ROUTE:{PAIR_KEY}", ew)
        z = (spread - m) / math.sqrt(max(v, 1e-12))

        # signal (positive when A rich vs B)
        self.emit_signal(max(-1.0, min(1.0, (spread - m) / max(0.1, ENTRY_PTS)) ))

        st = self._load_state()
        if st:
            if (abs(spread - m) <= EXIT_PTS) or (abs(z) <= EXIT_Z):
                self._close(st, mode="ROUTE")
            return

        if r.get(_poskey(self.ctx.name)) is not None:
            return
        if not (abs(spread - m) >= ENTRY_PTS and abs(z) >= ENTRY_Z):
            return

        qa, qb = self._size_two_legs(LEG_A, LEG_B, beta, pa, pb)
        if qa <= 0 or qb <= 0:
            return

        if spread > m:
            # A rich ⇒ SHORT A / LONG B
            self.order(LEG_A, "sell", qty=qa, order_type="market", venue=VENUE)
            self.order(LEG_B, "buy",  qty=qb, order_type="market", venue=VENUE)
            side = "short_spread"
        else:
            # A cheap ⇒ LONG A / SHORT B
            self.order(LEG_A, "buy",  qty=qa, order_type="market", venue=VENUE)
            self.order(LEG_B, "sell", qty=qb, order_type="market", venue=VENUE)
            side = "long_spread"

        self._save_state(OpenState(side=side, qa=qa, qb=qb, entry_pts=spread, entry_z=z, ts_ms=_now_ms()))

    # ---------------- CALENDAR spread ----------------
    def _calendar_eval(self) -> None:
        pn = _hget_price(LEG_NEAR)
        pf = _hget_price(LEG_FAR)
        if pn is None or pf is None or pn <= 0 or pf <= 0:
            return
        guard = _basis_guard(CAL_KEY)  # e.g., carry/seasonal guard in points
        spread = (pn - pf) + guard

        ew = _load_ewma(f"CAL:{CAL_KEY}")
        m, v = ew.update(spread)
        _save_ewma(f"CAL:{CAL_KEY}", ew)
        z = (spread - m) / math.sqrt(max(v, 1e-12))

        # signal (positive when NEAR rich vs FAR)
        self.emit_signal(max(-1.0, min(1.0, (spread - m) / max(0.1, ENTRY_PTS)) ))

        st = self._load_state()
        if st:
            if (abs(spread - m) <= EXIT_PTS) or (abs(z) <= EXIT_Z):
                self._close(st, mode="CAL")
            return

        if r.get(_poskey(self.ctx.name)) is not None:
            return
        if not (abs(spread - m) >= ENTRY_PTS and abs(z) >= ENTRY_Z):
            return

        # beta = 1 for calendar; size near & far via their own point values
        qn, qf = self._size_calendar_legs(LEG_NEAR, LEG_FAR, pn, pf)
        if qn <= 0 or qf <= 0:
            return

        if spread > m:
            # NEAR rich ⇒ SHORT NEAR / LONG FAR
            self.order(LEG_NEAR, "sell", qty=qn, order_type="market", venue=VENUE)
            self.order(LEG_FAR,  "buy",  qty=qf, order_type="market", venue=VENUE)
            side = "short_spread"
        else:
            # NEAR cheap ⇒ LONG NEAR / SHORT FAR
            self.order(LEG_NEAR, "buy",  qty=qn, order_type="market", venue=VENUE)
            self.order(LEG_FAR,  "sell", qty=qf, order_type="market", venue=VENUE)
            side = "long_spread"

        self._save_state(OpenState(side=side, qa=qn, qb=qf, entry_pts=spread, entry_z=z, ts_ms=_now_ms()))

    # ---------------- sizing helpers ----------------
    def _size_two_legs(self, A: str, B: str, beta: float, pa: float, pb: float) -> Tuple[float, float]:
        """
        USD_NOTIONAL_PER_LEG on each side. Convert using point values:
        qty = USD_NOTIONAL / (price * point_value)
        """
        pv_a = _ptval(A)
        pv_b = _ptval(B)
        if pv_a is None or pv_b is None or pv_a <= 0 or pv_b <= 0:
            return 0.0, 0.0

        qa = USD_NOTIONAL_PER_LEG / max(1e-9, pa * pv_a)
        qb = (USD_NOTIONAL_PER_LEG / max(1e-9, pb * pv_b)) * beta  # scale B by beta

        if qa * pa * pv_a < MIN_TICKET_USD or qb * pb * pv_b < MIN_TICKET_USD:
            return 0.0, 0.0
        return qa, qb

    def _size_calendar_legs(self, NEAR: str, FAR: str, pn: float, pf: float) -> Tuple[float, float]:
        pv_n = _ptval(NEAR)
        pv_f = _ptval(FAR)
        if pv_n is None or pv_f is None or pv_n <= 0 or pv_f <= 0:
            return 0.0, 0.0

        qn = USD_NOTIONAL_PER_LEG / max(1e-9, pn * pv_n)
        qf = USD_NOTIONAL_PER_LEG / max(1e-9, pf * pv_f)

        if qn * pn * pv_n < MIN_TICKET_USD or qf * pf * pv_f < MIN_TICKET_USD:
            return 0.0, 0.0
        return qn, qf

    # ---------------- state io ----------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw: return None
        try:
            return OpenState(**json.loads(raw)) # type: ignore
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # ---------------- close ----------------
    def _close(self, st: OpenState, mode: str) -> None:
        if mode == "ROUTE":
            if st.side == "short_spread":
                # unwind: BUY A / SELL B
                self.order(LEG_A, "buy",  qty=st.qa, order_type="market", venue=VENUE)
                self.order(LEG_B, "sell", qty=st.qb, order_type="market", venue=VENUE)
            else:
                self.order(LEG_A, "sell", qty=st.qa, order_type="market", venue=VENUE)
                self.order(LEG_B, "buy",  qty=st.qb, order_type="market", venue=VENUE)
        else:  # CALENDAR
            if st.side == "short_spread":
                self.order(LEG_NEAR, "buy",  qty=st.qa, order_type="market", venue=VENUE)
                self.order(LEG_FAR,  "sell", qty=st.qb, order_type="market", venue=VENUE)
            else:
                self.order(LEG_NEAR, "sell", qty=st.qa, order_type="market", venue=VENUE)
                self.order(LEG_FAR,  "buy",  qty=st.qb, order_type="market", venue=VENUE)
        self._save_state(None)