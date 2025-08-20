# backend/strategies/diversified/index_future_vs_spot.py
from __future__ import annotations

import json, math, os, time, datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Index Future ↔ Spot Basis Arbitrage (paper)
-------------------------------------------
Fair forward (continuous carry):
    F_fair = S * exp((r - q) * T)
Where:
  S = spot (ETF or index proxy)
  r = risk-free rate (decimal)
  q = dividend yield (decimal)  OR use dividend PV via redis (optional, see below)
  T = time to expiry (years)

Signals:
  basis_bps = 1e4 * (F_mkt - F_fair) / S
    > +ENTRY_BPS ⇒ SHORT future / LONG spot
    < -ENTRY_BPS ⇒ LONG future / SHORT spot
Exit on |basis_bps| ≤ EXIT_BPS or |z| ≤ EXIT_Z.

Redis feeds (you already use these patterns elsewhere):
  HSET last_price <SPOT_SYM>  '{"price": <S>}'
  HSET fut:price  <FUT_SYM>   <F_mkt>
  HSET rate:risk_free:<CCY> <CCY> <r>         (optional; env fallback exists)
  HSET div:yield  <SPOT_SYM>  <q>             (optional; env fallback exists)

Optional alternative (div PV vs yield):
  HSET div:pv:<FUT_SYM> <SPOT_SYM> <PVdiv>    # present value of cash dividends till expiry
  If present AND USE_DIV_PV=true → fair = (S - PVdiv) * exp(r*T)

Paper routing (map later in your adapters):
  • Spot leg: <SPOT_SYM>  (e.g., SPY)
  • Future  : <FUT_SYM>   (e.g., IFUT:SPX:SEP25@CME or NIFTY:I:2025AUG@NSE)
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("IFVS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("IFVS_REDIS_PORT", "6379"))

SPOT_SYM   = os.getenv("IFVS_SPOT", "SPY").upper()
FUT_SYM    = os.getenv("IFVS_FUT",  "IFUT:SPX:SEP25@CME").upper()
CCY        = os.getenv("IFVS_CCY",  "USD").upper()

# Expiry: set either IFVS_EXPIRY=YYYY-MM-DD or IFVS_T_DAYS (fallback)
EXPIRY_STR = os.getenv("IFVS_EXPIRY", "")  # e.g., 2025-09-20
T_DAYS_ENV = os.getenv("IFVS_T_DAYS", "")

# Carry model
USE_DIV_PV   = os.getenv("IFVS_USE_DIV_PV", "false").lower() in ("1","true","yes")
RF_FALLBACK  = float(os.getenv("IFVS_RF", "0.03"))
Q_FALLBACK   = float(os.getenv("IFVS_Q",  "0.015"))  # dividend yield fallback (1.5%)

# Thresholds (bps & z)
ENTRY_BPS = float(os.getenv("IFVS_ENTRY_BPS", "12.0"))
EXIT_BPS  = float(os.getenv("IFVS_EXIT_BPS",  "4.0"))
ENTRY_Z   = float(os.getenv("IFVS_ENTRY_Z",   "1.4"))
EXIT_Z    = float(os.getenv("IFVS_EXIT_Z",    "0.5"))

# Sizing
USD_NOTIONAL_PER_LEG = float(os.getenv("IFVS_USD_PER_LEG", "50000"))
MIN_TICKET_USD       = float(os.getenv("IFVS_MIN_TICKET_USD", "200"))
MAX_CONCURRENT       = int(os.getenv("IFVS_MAX_CONCURRENT", "1"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("IFVS_RECHECK_SECS", "2"))
EWMA_ALPHA   = float(os.getenv("IFVS_EWMA_ALPHA", "0.06"))

# Venues (advisory only)
VENUE_SPOT = os.getenv("IFVS_VENUE_SPOT", "ARCA").upper()
VENUE_FUT  = os.getenv("IFVS_VENUE_FUT",  "CME").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("IFVS_LAST_PRICE_KEY", "last_price")      # spot
FUT_PRICE_HKEY  = os.getenv("IFVS_FUT_PRICE_KEY",  "fut:price")       # HSET fut:price <FUT_SYM> <px>
RATE_HKEY_FMT   = os.getenv("IFVS_RATE_KEY_FMT",   "rate:risk_free:{ccy}")  # HSET rate:risk_free:USD USD 0.035
DIV_YIELD_HKEY  = os.getenv("IFVS_DIV_YIELD_KEY",  "div:yield")       # HSET div:yield <SPOT_SYM> <q>
DIV_PV_HKEY     = os.getenv("IFVS_DIV_PV_KEY",     "div:pv:{fut}")    # HSET div:pv:<FUT_SYM> <SPOT_SYM> <PVdiv>

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _now_date() -> dt.date:
    # use system date; in your deployment runner this is wall-clock UTC/local
    return dt.datetime.utcnow().date()

def _years_to_expiry() -> float:
    # prefer explicit expiry date
    if EXPIRY_STR:
        try:
            exp = dt.date.fromisoformat(EXPIRY_STR)
            days = max(0, (exp - _now_date()).days)
            return max(1e-6, days / 365.0)
        except Exception:
            pass
    # fallback: env days
    if T_DAYS_ENV:
        try:
            return max(1e-6, float(T_DAYS_ENV) / 365.0)
        except Exception:
            pass
    # default: one month
    return 30.0 / 365.0

def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception: return None

def _hgetf(hashkey: str, field: str) -> Optional[float]:
    v = r.hget(hashkey, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _rate(ccy: str) -> float:
    v = _hgetf(RATE_HKEY_FMT.format(ccy=ccy), ccy)
    return float(v) if v is not None else RF_FALLBACK

def _div_yield(sym: str) -> float:
    v = _hgetf(DIV_YIELD_HKEY, sym)
    return float(v) if v is not None else Q_FALLBACK

def _div_pv(fut: str, sym: str) -> Optional[float]:
    key = DIV_PV_HKEY.format(fut=fut)
    return _hgetf(key, sym)

def _fut_px(sym: str) -> Optional[float]:
    return _hgetf(FUT_PRICE_HKEY, sym)

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

def _ewma_key() -> str:
    return f"ifvs:ewma:{SPOT_SYM}:{FUT_SYM}"

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

# ============================ state ============================
@dataclass
class OpenState:
    side: str  # "short_fut_long_spot" or "long_fut_short_spot"
    qty_spot: float
    qty_fut: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"ifvs:open:{name}:{SPOT_SYM}:{FUT_SYM}"

# ============================ strategy ============================
class IndexFutureVsSpot(Strategy):
    """
    Trades the index future vs spot when basis deviates from fair carry.
    """
    def __init__(self, name: str = "index_future_vs_spot", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "spot": SPOT_SYM, "future": FUT_SYM, "ccy": CCY,
            "expiry": EXPIRY_STR or f"{T_DAYS_ENV}d_fallback",
            "use_div_pv": USE_DIV_PV, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- engine ---------------
    def _evaluate(self) -> None:
        S = _hget_price(SPOT_SYM)
        Fm = _fut_px(FUT_SYM)
        if S is None or Fm is None or S <= 0 or Fm <= 0:
            return

        r_rf = _rate(CCY)
        T = _years_to_expiry()

        if USE_DIV_PV:
            pvdiv = _div_pv(FUT_SYM, SPOT_SYM)
            if pvdiv is None:
                return
            Ffair = (S - pvdiv) * math.exp(r_rf * T)
        else:
            q = _div_yield(SPOT_SYM)
            Ffair = S * math.exp((r_rf - q) * T)

        basis = (Fm - Ffair) / max(1e-9, S)
        bps = 1e4 * basis

        ew = _load_ewma()
        m, v = ew.update(bps)
        _save_ewma(ew)
        z = (bps - m) / math.sqrt(max(v, 1e-12))

        # emit monitor signal: positive when future rich vs fair
        self.emit_signal(max(-1.0, min(1.0, (bps - m) / max(1.0, ENTRY_BPS))))

        st = self._load_state()

        # ---- exits ----
        if st:
            if (abs(bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ---- entries ----
        if r.get(_poskey(self.ctx.name)) is not None:
            return
        if not (abs(bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        qty_spot = USD_NOTIONAL_PER_LEG / S
        qty_fut  = USD_NOTIONAL_PER_LEG / Fm
        if qty_spot * S < MIN_TICKET_USD or qty_fut * Fm < MIN_TICKET_USD:
            return

        if bps > 0:
            # Future rich ⇒ SHORT future / LONG spot
            self.order(FUT_SYM, "sell", qty=qty_fut, order_type="market", venue=VENUE_FUT)
            self.order(SPOT_SYM, "buy",  qty=qty_spot, order_type="market", venue=VENUE_SPOT)
            side = "short_fut_long_spot"
        else:
            # Future cheap ⇒ LONG future / SHORT spot
            self.order(FUT_SYM, "buy",  qty=qty_fut, order_type="market", venue=VENUE_FUT)
            self.order(SPOT_SYM, "sell", qty=qty_spot, order_type="market", venue=VENUE_SPOT)
            side = "long_fut_short_spot"

        self._save_state(OpenState(
            side=side, qty_spot=qty_spot, qty_fut=qty_fut,
            entry_bps=bps, entry_z=z, ts_ms=int(time.time()*1000)
        ))

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

    # --------------- close ---------------
    def _close(self, st: OpenState) -> None:
        if st.side == "short_fut_long_spot":
            self.order(FUT_SYM, "buy",  qty=st.qty_fut, order_type="market", venue=VENUE_FUT)
            self.order(SPOT_SYM, "sell", qty=st.qty_spot, order_type="market", venue=VENUE_SPOT)
        else:
            self.order(FUT_SYM, "sell", qty=st.qty_fut, order_type="market", venue=VENUE_FUT)
            self.order(SPOT_SYM, "buy",  qty=st.qty_spot, order_type="market", venue=VENUE_SPOT)
        r.delete(_poskey(self.ctx.name))