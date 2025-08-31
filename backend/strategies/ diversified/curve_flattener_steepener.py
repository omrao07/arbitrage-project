# backend/strategies/diversified/curve_flatener_steepener.py
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
Curve Flattener / Steepener
---------------------------
Trades slope changes between two curve points using interest‑rate futures (or ETFs).
Example (US):
  • 2s10s: TU(2Y) vs TY(10Y)
  • 5s30s: FV(5Y) vs US(30Y)

Spread definition (bps):
  slope = y_long - y_short     # e.g., 10Y minus 2Y

Signals (mean‑reverting by default):
  • If slope >> mean (steep): enter **Flattener**  → LONG long‑tenor future, SHORT short‑tenor future
  • If slope << mean (flat/inverted): enter **Steepener** → SHORT long‑tenor future, LONG short‑tenor future
(Prices move opposite to yields; DV01‑ratioed for level‑DV01 neutrality.)

Redis inputs you already publish:
  - HSET yield <TENOR> <rate_decimal>        # e.g., HSET yield 2Y 0.046 ; HSET yield 10Y 0.042
  - HSET dv01  <SYMBOL> <usd_per_contract>   # e.g., HSET dv01 TU  85 ; HSET dv01 TY  120
  - (optional) HSET last_price <SYMBOL> '{"price": <px>}'  # for UI/health; not required for sizing

Paper OMS legs are just the futures/ETFs symbols you route elsewhere (e.g., TU, TY, FV, US, IEF, TLT).
"""

# ============================== CONFIG ==============================
REDIS_HOST = os.getenv("CURVE_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CURVE_REDIS_PORT", "6379"))

# Curve pairs (semicolon‑separated):
# "name,SHORT_TENOR,LONG_TENOR,SHORT_SYMBOL,LONG_SYMBOL"
# Tenors must match fields in your HSET 'yield'. Symbols must match your OMS.
PAIRS_ENV = os.getenv(
    "CURVE_PAIRS",
    "2s10s,2Y,10Y,TU,TY;5s30s,5Y,30Y,FV,US"
)

# Trading style: mean‑revert by default; set TREND=true to chase steepen/flatten trends.
MEAN_REVERT = os.getenv("CURVE_MEAN_REVERT", "true").lower() in ("1","true","yes")

# Thresholds
ENTRY_BPS = float(os.getenv("CURVE_ENTRY_BPS", "12"))   # |slope - mean| in bps
EXIT_BPS  = float(os.getenv("CURVE_EXIT_BPS",  "4"))
ENTRY_Z   = float(os.getenv("CURVE_ENTRY_Z",   "1.5"))
EXIT_Z    = float(os.getenv("CURVE_EXIT_Z",    "0.5"))

# Sizing
USD_DV01_TARGET = float(os.getenv("CURVE_USD_DV01_TARGET", "200"))  # target *per leg* DV01 in USD
MAX_CONCURRENT  = int(os.getenv("CURVE_MAX_CONCURRENT", "3"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("CURVE_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("CURVE_EWMA_ALPHA", "0.06"))   # event‑based EWMA on slope

# Venues (advisory)
VENUE_IRF = os.getenv("CURVE_VENUE", "CME").upper()

# Redis keys
YIELD_HKEY      = os.getenv("CURVE_YIELD_KEY", "yield")  # HSET yield 2Y 0.046
DV01_HKEY       = os.getenv("CURVE_DV01_KEY",  "dv01")   # HSET dv01 TU 85
LAST_PRICE_HKEY = os.getenv("CURVE_LAST_PRICE_KEY", "last_price")

# ============================== REDIS ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== HELPERS ==============================
@dataclass
class CurvePair:
    name: str
    t_short: str
    t_long: str
    sym_short: str
    sym_long: str

def _parse_pairs(env: str) -> List[CurvePair]:
    out: List[CurvePair] = []
    for part in env.split(";"):
        s = part.strip()
        if not s:
            continue
        try:
            name, ts, tl, ss, sl = [x.strip().upper() for x in s.split(",")]
            out.append(CurvePair(name=name, t_short=ts, t_long=tl, sym_short=ss, sym_long=sl))
        except Exception:
            continue
    return out

PAIRS = _parse_pairs(PAIRS_ENV)

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _now_ms() -> int:
    return int(time.time() * 1000)

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

def _ewma_key(cp: CurvePair) -> str:
    return f"curve:ewma:{cp.name}:{cp.t_short}-{cp.t_long}"

def _load_ewma(cp: CurvePair, alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key(cp))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(cp: CurvePair, ew: EwmaMV) -> None:
    r.set(_ewma_key(cp), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================== STATE ==============================
@dataclass
class OpenState:
    side: str           # "flattener" or "steepener"
    q_short: float      # contracts of short‑tenor future
    q_long: float       # contracts of long‑tenor future
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, cp: CurvePair) -> str:
    return f"curve:open:{name}:{cp.name}"

# ============================== STRATEGY ==============================
class CurveFlattenerSteepener(Strategy):
    """
    DV01‑ratioed flattener/steepener on a configurable curve pair.
    """
    def __init__(self, name: str = "curve_flattener_steepener", region: Optional[str] = "RATES", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "pairs": [cp.__dict__ for cp in PAIRS],
            "style": "mean_reversion" if MEAN_REVERT else "trend",
            "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate_all()

    # ---------------- core ----------------
    def _evaluate_all(self) -> None:
        open_count = sum(1 for cp in PAIRS if r.get(_poskey(self.ctx.name, cp)))

        for cp in PAIRS:
            y_s = _hgetf(YIELD_HKEY, cp.t_short)
            y_l = _hgetf(YIELD_HKEY, cp.t_long)
            if y_s is None or y_l is None:
                continue

            slope_bps = (y_l - y_s) * 1e4
            ew = _load_ewma(cp, EWMA_ALPHA)
            m, v = ew.update(slope_bps)
            _save_ewma(cp, ew)
            z = (slope_bps - m) / math.sqrt(max(v, 1e-12))

            # emit monitor signal (positive when steepening)
            self.emit_signal(max(-1.0, min(1.0, (slope_bps - m) / 50.0)))

            st = self._load_state(cp)

            # -------- exits first --------
            if st:
                if (abs(slope_bps - m) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                    self._close(cp, st)
                continue

            # -------- entries --------
            if open_count >= MAX_CONCURRENT:
                continue
            dev = slope_bps - m
            if not (abs(dev) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
                continue

            # decide direction
            # mean‑revert: rich → flattener ; cheap → steepener
            want_flattener = (dev > 0) if MEAN_REVERT else (dev < 0)
            if want_flattener:
                self._enter_flattener(cp, slope_bps, z)
            else:
                self._enter_steepener(cp, slope_bps, z)
            open_count += 1

    # ---------------- sizing (DV01 ratio) ----------------
    def _dv01(self, sym: str) -> Optional[float]:
        return _hgetf(DV01_HKEY, sym)

    def _ratio_qty(self, cp: CurvePair) -> Tuple[float, float]:
        """
        Return (q_short, q_long) contracts sized so each leg carries ~USD_DV01_TARGET.
        Also ratio long/short so *total level DV01 is near zero*:
            q_short * DV01_short ≈ q_long * DV01_long
        We simply target the same USD DV01 per leg; your global risk will cap notionals.
        """
        d_short = self._dv01(cp.sym_short)
        d_long  = self._dv01(cp.sym_long)
        if not d_short or not d_long or d_short <= 0 or d_long <= 0:
            return 0.0, 0.0
        q_s = USD_DV01_TARGET / d_short
        q_l = USD_DV01_TARGET / d_long
        return q_s, q_l

    # ---------------- enter/close ----------------
    def _enter_flattener(self, cp: CurvePair, slope_bps: float, z: float) -> None:
        """
        Flattener = LONG long‑tenor duration, SHORT short‑tenor duration
                  = BUY long‑tenor future, SELL short‑tenor future
        """
        q_s, q_l = self._ratio_qty(cp)
        if q_s <= 0 or q_l <= 0:
            return
        self.order(cp.sym_long,  "buy",  qty=q_l, order_type="market", venue=VENUE_IRF)
        self.order(cp.sym_short, "sell", qty=q_s, order_type="market", venue=VENUE_IRF)
        self._save_state(cp, OpenState(
            side="flattener", q_short=q_s, q_long=q_l,
            entry_bps=slope_bps, entry_z=z, ts_ms=_now_ms()
        ))

    def _enter_steepener(self, cp: CurvePair, slope_bps: float, z: float) -> None:
        """
        Steepener = SHORT long‑tenor duration, LONG short‑tenor duration
                  = SELL long‑tenor future, BUY short‑tenor future
        """
        q_s, q_l = self._ratio_qty(cp)
        if q_s <= 0 or q_l <= 0:
            return
        self.order(cp.sym_long,  "sell", qty=q_l, order_type="market", venue=VENUE_IRF)
        self.order(cp.sym_short, "buy",  qty=q_s, order_type="market", venue=VENUE_IRF)
        self._save_state(cp, OpenState(
            side="steepener", q_short=q_s, q_long=q_l,
            entry_bps=slope_bps, entry_z=z, ts_ms=_now_ms()
        ))

    def _close(self, cp: CurvePair, st: OpenState) -> None:
        if st.side == "flattener":
            self.order(cp.sym_long,  "sell", qty=st.q_long,  order_type="market", venue=VENUE_IRF)
            self.order(cp.sym_short, "buy",  qty=st.q_short, order_type="market", venue=VENUE_IRF)
        else:
            self.order(cp.sym_long,  "buy",  qty=st.q_long,  order_type="market", venue=VENUE_IRF)
            self.order(cp.sym_short, "sell", qty=st.q_short, order_type="market", venue=VENUE_IRF)
        self._save_state(cp, None)

    # ---------------- state helpers ----------------
    def _load_state(self, cp: CurvePair) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, cp))
        if not raw:
            return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, cp: CurvePair, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name, cp)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))