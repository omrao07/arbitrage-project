# backend/strategies/diversified/dividend_arbitrage.py
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
Dividend Arbitrage (cash-and-carry)
-----------------------------------
Fair forward with discrete dividends (PVdiv):
    F_fair = (S - PVdiv_forecast) * exp(r * T)

We read:
  • Spot S:                    HSET last_price <SYM> '{"price": <spot>}'
  • Forward/Futures price F:   HSET fwd:<TENOR>  <SYM>  <price>
      - if you don't have an outright forward, you may publish a near futures price here.
  • Sum of forecast cash divs PV (present value in USD): 
                               HSET div:pv:<TENOR> <SYM> <usd>
      - or provide undiscounted sum and we discount with r; see ENV switch below.
  • Risk-free rate r:          HSET rate:risk_free:<CCY> <decimal>  (fallback RATE_RF)
  • (optional) FX for non-USD stocks if publishing in local ccy:
                               HSET fx:spot <PAIR> <spot>  (we’ll assume USD universe by default)

Signal:
  basis = (F_mkt - F_fair) / S
    > +ENTRY_BPS → F rich → SHORT forward, LONG spot  (collect implied dividend/carry)
    < -ENTRY_BPS → F cheap → LONG forward, SHORT spot

Exits when |basis| reverts (level + z-score gates).

Paper symbols:
  • Underlier spot: <SYM>
  • Forward/Future: FWD:<SYM>:<TENOR>  (your OMS can map this to the real future later)
"""

# ================== CONFIG (env) ==================
REDIS_HOST = os.getenv("DIVARB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("DIVARB_REDIS_PORT", "6379"))

UNIVERSE = [s.strip().upper() for s in os.getenv(
    "DIVARB_UNIVERSE",
    "SPY,AAPL,MSFT"
).split(",") if s.strip()]

TENOR = os.getenv("DIVARB_TENOR", "3M").upper()  # keys fwd:<TENOR>, div:pv:<TENOR>

# If your dividend store is *undiscounted* (sum of announced cash amounts),
# we’ll discount by r; otherwise set DIV_PV_IS_DISCOUNTED=true.
DIV_PV_IS_DISCOUNTED = os.getenv("DIV_PV_IS_DISCOUNTED", "true").lower() in ("1","true","yes")

# Thresholds (bps & z-score)
ENTRY_BPS = float(os.getenv("DIVARB_ENTRY_BPS", "20"))   # enter when |basis| >= 20 bps of S
EXIT_BPS  = float(os.getenv("DIVARB_EXIT_BPS",  "7"))
ENTRY_Z   = float(os.getenv("DIVARB_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("DIVARB_EXIT_Z",    "0.4"))

# Sizing
USD_NOTIONAL_PER_LEG = float(os.getenv("DIVARB_USD_PER_LEG", "50000"))
MIN_TICKET_USD       = float(os.getenv("DIVARB_MIN_TICKET_USD", "200"))
MAX_CONCURRENT       = int(os.getenv("DIVARB_MAX_CONCURRENT", "5"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("DIVARB_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("DIVARB_EWMA_ALPHA", "0.05"))

# Rates / time
RATE_RF      = float(os.getenv("DIVARB_RATE_RF", "0.03"))    # fallback risk-free (decimal)
TENOR_MONTHS = float(os.getenv("DIVARB_TENOR_MONTHS", "3"))  # 3M default
DAYS_IN_YEAR = float(os.getenv("DIVARB_DAYCOUNT", "365"))

# Venues (advisory)
VENUE_EQ   = os.getenv("DIVARB_VENUE_EQ", "ARCA").upper()
VENUE_FWD  = os.getenv("DIVARB_VENUE_FWD", "CFE").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("DIVARB_LAST_PRICE_KEY", "last_price")          # HSET <SYM> -> {"price": ...}
FWD_HKEY_FMT    = os.getenv("DIVARB_FWD_HKEY_FMT", "fwd:{tenor}")           # HSET fwd:<TENOR> <SYM> -> F
DIVPV_HKEY_FMT  = os.getenv("DIVARB_DIVPV_HKEY_FMT", "div:pv:{tenor}")      # HSET div:pv:<TENOR> <SYM> -> PVdiv
RATE_HKEY_FMT   = os.getenv("DIVARB_RATE_HKEY_FMT", "rate:risk_free:{ccy}") # HSET rate:risk_free:USD 0.03

# ================== Redis ==================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ================== Helpers ==================
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
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _t_years(months: float) -> float:
    return max(1e-6, (30.0 * months) / DAYS_IN_YEAR)

def _rate_for(sym: str) -> float:
    # assume USD; extend later with per‑ccy map
    v = _hgetf(RATE_HKEY_FMT.format(ccy="USD"), "USD")
    return float(v) if v is not None else RATE_RF

def _fwd_key() -> str:
    return FWD_HKEY_FMT.format(tenor=TENOR)

def _divpv_key() -> str:
    return DIVPV_HKEY_FMT.format(tenor=TENOR)

def _fwd_sym(sym: str) -> str:
    return f"FWD:{sym}:{TENOR}"

# ================== EWMA ==================
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
    return f"divarb:ewma:{sym}:{TENOR}"

def _load_ewma(sym: str) -> EwmaMV:
    raw = r.get(_ewma_key(sym))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(sym: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(sym), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ================== State ==================
@dataclass
class OpenState:
    side: str           # "short_fwd_long_spot" or "long_fwd_short_spot"
    sym: str
    qty_spot: float
    qty_fwd: float
    entry_basis_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, sym: str) -> str:
    return f"divarb:open:{name}:{sym}:{TENOR}"

# ================== Strategy ==================
class DividendArbitrage(Strategy):
    """
    Cash-and-carry dividend arbitrage on single names or indices.
    """
    def __init__(self, name: str = "dividend_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "tenor": TENOR, "symbols": UNIVERSE, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate_all()

    # -------------- engine --------------
    def _evaluate_all(self) -> None:
        open_count = sum(1 for sym in UNIVERSE if r.get(_poskey(self.ctx.name, sym)))
        for sym in UNIVERSE:
            spot = _hget_price(sym)
            fwd  = _hgetf(_fwd_key(), sym)
            pvdiv = _hgetf(_divpv_key(), sym)
            if spot is None or spot <= 0 or fwd is None or pvdiv is None:
                continue

            r_rf = _rate_for(sym)
            T = _t_years(TENOR_MONTHS)

            # If div amounts are undiscounted, discount to PV
            pvdiv_eff = pvdiv if DIV_PV_IS_DISCOUNTED else pvdiv / math.exp(r_rf * T)

            f_fair = (spot - pvdiv_eff) * math.exp(r_rf * T)
            basis = (fwd - f_fair) / max(1e-9, spot)   # as fraction of spot
            basis_bps = 1e4 * basis

            ew = _load_ewma(sym)
            m, v = ew.update(basis_bps)
            _save_ewma(sym, ew)
            z = (basis_bps - m) / math.sqrt(max(v, 1e-12))

            # dashboard signal: positive when F rich vs fair
            self.emit_signal(max(-1.0, min(1.0, (basis_bps - m) / 10.0)))

            st = self._load_state(sym)

            # ----- exits -----
            if st:
                if (abs(basis_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                    self._close(st)
                continue

            # ----- entries -----
            if open_count >= MAX_CONCURRENT:
                continue
            if not (abs(basis_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
                continue

            # size by USD notionals
            qty_spot = USD_NOTIONAL_PER_LEG / spot
            qty_fwd  = USD_NOTIONAL_PER_LEG / max(1e-9, fwd)
            if USD_NOTIONAL_PER_LEG < MIN_TICKET_USD:
                continue

            if basis_bps > 0:
                # Forward rich: SHORT fwd, LONG spot
                self.order(_fwd_sym(sym), "sell", qty=qty_fwd, order_type="market", venue=VENUE_FWD)
                self.order(sym,              "buy",  qty=qty_spot, order_type="market", venue=VENUE_EQ)
                self._save_state(OpenState(
                    side="short_fwd_long_spot", sym=sym, qty_spot=qty_spot, qty_fwd=qty_fwd,
                    entry_basis_bps=basis_bps, entry_z=z, ts_ms=int(time.time()*1000)
                ))
            else:
                # Forward cheap: LONG fwd, SHORT spot
                self.order(_fwd_sym(sym), "buy",  qty=qty_fwd, order_type="market", venue=VENUE_FWD)
                self.order(sym,              "sell", qty=qty_spot, order_type="market", venue=VENUE_EQ)
                self._save_state(OpenState(
                    side="long_fwd_short_spot", sym=sym, qty_spot=qty_spot, qty_fwd=qty_fwd,
                    entry_basis_bps=basis_bps, entry_z=z, ts_ms=int(time.time()*1000)
                ))
            open_count += 1

    # -------------- state io --------------
    def _load_state(self, sym: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, sym))
        if not raw: return None
        try:
            return OpenState(**json.loads(raw))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, st.sym), json.dumps(st.__dict__))

    # -------------- close --------------
    def _close(self, st: OpenState) -> None:
        if st.side == "short_fwd_long_spot":
            # buy back fwd short, sell spot long
            self.order(_fwd_sym(st.sym), "buy",  qty=st.qty_fwd, order_type="market", venue=VENUE_FWD)
            self.order(st.sym,           "sell", qty=st.qty_spot, order_type="market", venue=VENUE_EQ)
        else:
            # sell back fwd long, buy back spot short
            self.order(_fwd_sym(st.sym), "sell", qty=st.qty_fwd, order_type="market", venue=VENUE_FWD)
            self.order(st.sym,           "buy",  qty=st.qty_spot, order_type="market", venue=VENUE_EQ)
        r.delete(_poskey(self.ctx.name, st.sym))