# backend/strategies/diversified/delta_neutral_vol_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Delta‑Neutral Volatility Arbitrage (ATM straddle, paper)
-------------------------------------------------------
Core idea:
  • Compare implied vol (IV) at tenor T with a realized/forecast anchor (RV).
  • If IV - RV >> threshold ⇒ SHORT volatility: sell ATM straddle, delta‑hedge with underlier.
  • If IV - RV << -threshold ⇒ LONG volatility: buy ATM straddle, delta‑hedge.

Synthetics for paper OMS:
  • STRAD:<SYM>:<TENOR>  → one "unit" = 1 ATM CALL + 1 ATM PUT (same T, strike ≈ spot)
  • Underlier hedging uses <SYM> (spot/ETF).

Redis inputs you already maintain elsewhere:
  HSET last_price          <SYM>            '{"price": <spot>}'
  HSET iv:imp:<TENOR>      <SYM>            <iv_decimal>
  HSET iv:real:<TENOR>     <SYM>            <rv_decimal>                    (optional)
  SET  vol:forecast:<TENOR>:<SYM>           <rv_decimal>                    (optional; overrides iv:real)
  SET  rate:risk_free                       <r_decimal>                     (optional)
  HSET div:yield           <SYM>            <q_decimal>                     (optional)

This module:
  • Computes ATM straddle greeks (Black–Scholes, K≈S).
  • Sizes trades by *vega* (USD per vol‑pt) to a target notional.
  • Re‑hedges delta when drift exceeds a threshold.
  • Exits on vol spread mean‑reversion (level + z‑score gates).
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("DNV_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("DNV_REDIS_PORT", "6379"))

SYM    = os.getenv("DNV_SYMBOL", "ACME").upper()
TENOR  = os.getenv("DNV_TENOR", "30D").upper()

# Thresholds (vol points and z-score)
ENTRY_VOL = float(os.getenv("DNV_ENTRY_VOL", "0.06"))   # e.g., 0.06 = 6 vol pts IV-RV
EXIT_VOL  = float(os.getenv("DNV_EXIT_VOL",  "0.02"))
ENTRY_Z   = float(os.getenv("DNV_ENTRY_Z",   "1.5"))
EXIT_Z    = float(os.getenv("DNV_EXIT_Z",    "0.5"))

# Sizing (target vega in USD per package and risk guards)
USD_VEGA_TARGET = float(os.getenv("DNV_USD_VEGA_TARGET", "2000"))   # per package |vega| in $ / vol‑pt
MAX_CONCURRENT  = int(os.getenv("DNV_MAX_CONCURRENT", "1"))
MIN_TICKET_USD  = float(os.getenv("DNV_MIN_TICKET_USD", "200"))

# Hedge controls
DELTA_REHEDGE_ABS = float(os.getenv("DNV_DELTA_REHEDGE_ABS", "0.05"))  # re-hedge when |delta| > 5% of share
HEDGE_VENUE       = os.getenv("DNV_HEDGE_VENUE", "ARCA").upper()

# Cadence & stats
RECHECK_SECS = int(os.getenv("DNV_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("DNV_EWMA_ALPHA", "0.05"))

# Venues (synthetic straddle venue advisory)
VENUE_OPT = os.getenv("DNV_VENUE_OPT", "CBOE").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("DNV_LAST_PRICE_KEY", "last_price")  # HSET <SYM> -> {"price": ...}
IV_IMP_KEY      = f"iv:imp:{TENOR}"                              # HSET iv:imp:<TENOR> <SYM> -> iv
IV_REAL_KEY     = f"iv:real:{TENOR}"                             # HSET iv:real:<TENOR> <SYM> -> rv
RV_FCAST_KEY    = f"vol:forecast:{TENOR}:{SYM}"                  # SET  vol:forecast:<TENOR>:<SYM> -> rv
RATE_KEY        = "rate:risk_free"                               # SET  rate:risk_free -> r
DIV_YIELD_HKEY  = "div:yield"                                    # HSET div:yield <SYM> -> q

# Synthetic symbol helpers
def _strad_sym(sym: str, tenor: str) -> str:
    return f"STRAD:{sym}:{tenor}"

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ Math (BS greeks) ============================
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _time_to_maturity_yr(tenor: str) -> float:
    t = tenor.upper().strip()
    if t.endswith("D"):
        d = float(t[:-1] or 30.0)
        return d / 365.0
    if t.endswith("M"):
        m = float(t[:-1] or 1.0)
        return (30.0 * m) / 365.0
    if t.endswith("Y"):
        y = float(t[:-1] or 1.0)
        return y
    # default 30D
    return 30.0 / 365.0

def _bs_greeks_atm(S: float, sigma: float, r: float, q: float, T: float) -> Tuple[float, float, float]:
    """
    ATM CALL greeks; ATM PUT greeks via parity. For ATM straddle:
      delta_strad ≈ 0
      vega_strad  ≈ 2 * (S * e^{-qT} * sqrt(T) * φ(d1))
    Returns (vega_strad_USD_per_vol, gamma_strad, d1) – we only use vega for sizing, delta for hedging.
    """
    if S <= 0 or sigma <= 0 or T <= 0:
        return 0.0, 0.0, 0.0
    K = S  # ATM approximation
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / max(1e-12, srt)
    disc_q = math.exp(-q * T)
    vega_call = S * disc_q * math.sqrt(T) * _norm_pdf(d1)  # per 1.0 change in sigma (i.e., per +1.00 = +100 vol pts)
    vega_strad = 2.0 * vega_call                           # call + put
    # Convert to per 1 vol‑pt (0.01) in *USD*: multiply by 0.01
    vega_per_volpt_usd = vega_strad * 0.01
    # Gamma (per $1) for the *pair* (approx 2× call gamma)
    gamma_call = disc_q * _norm_pdf(d1) / (S * srt)
    gamma_strad = 2.0 * gamma_call
    return vega_per_volpt_usd, gamma_strad, d1

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
    return f"dnv:ewma:{SYM}:{TENOR}"

def _load_ewma(alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o['m']), var=float(o['v']), alpha=float(o.get('a', alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ Helpers ============================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _getf(key: str) -> Optional[float]:
    v = r.get(key)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

# ============================ State ============================
@dataclass
class OpenState:
    side: str            # "long_vol" or "short_vol"
    n_units: float       # number of straddle "units" (paper)
    hedge_shares: float  # current delta hedge in shares
    last_delta: float    # most recent computed straddle delta (should be near 0 but moves with spot)
    entry_spread: float  # IV - RV at entry (vol points)
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"dnv:open:{name}:{SYM}:{TENOR}"

# ============================ Strategy ============================
class DeltaNeutralVolArbitrage(Strategy):
    """
    Long/short ATM straddle vs realized/forecast vol with delta hedging.
    """
    def __init__(self, name: str = "delta_neutral_vol_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "underlier": SYM, "tenor": TENOR, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last_check < RECHECK_SECS:
            return
        self._last_check = now
        self._evaluate()

    # ---------- core ----------
    def _inputs(self) -> Optional[Tuple[float, float, float, float, float]]:
        S = _hget_price(SYM)
        if S is None or S <= 0:
            return None
        iv = _hgetf(IV_IMP_KEY, SYM)
        if iv is None or iv <= 0:
            return None
        # anchor: forecast vol overrides realized if present
        rv = _getf(RV_FCAST_KEY)
        if rv is None:
            rv = _hgetf(IV_REAL_KEY, SYM)
        if rv is None or rv <= 0:
            # conservative default: use 80% of IV as anchor if not provided
            rv = 0.8 * iv

        r_rf = _getf(RATE_KEY) or 0.02
        q_div = _hgetf(DIV_YIELD_HKEY, SYM) or 0.0
        T = _time_to_maturity_yr(TENOR)
        return S, iv, rv, r_rf, q_div

    def _evaluate(self) -> None:
        vals = self._inputs()
        if not vals:
            return
        S, iv, rv, r_rf, q_div = vals
        T = _time_to_maturity_yr(TENOR)

        vega_per_volpt_usd, gamma_strad, d1 = _bs_greeks_atm(S, iv, r_rf, q_div, T)
        if vega_per_volpt_usd <= 0:
            return

        spread = iv - rv   # positive => IV rich vs anchor (sell vol)
        ew = _load_ewma(EWMA_ALPHA)
        m, v = ew.update(spread)
        _save_ewma(ew)
        z = (spread - m) / math.sqrt(max(v, 1e-12))

        # monitor: squash to [-1,1] (positive when IV>RV)
        self.emit_signal(max(-1.0, min(1.0, math.tanh(spread / 0.05))))

        st = self._load_state()

        # ----- exits & hedge upkeep -----
        if st:
            # Recompute instantaneous straddle delta (approx: call_delta - put_delta ≈ 2N(d1)-1; at ATM ≈ 0)
            # For robustness, clamp by gamma * dS notionally; but we’ll just re‑hedge if drift > threshold.
            delta_est = (2.0 * _norm_cdf(d1) - 1.0) * st.n_units  # units ~ contracts; works as relative measure
            drift = delta_est - st.last_delta

            # Re‑hedge if absolute delta exceeds threshold of underlying "share" notionally
            if abs(delta_est) >= DELTA_REHEDGE_ABS:
                qty = abs(delta_est - st.hedge_shares)
                if qty * S >= MIN_TICKET_USD and qty > 0:
                    side = "sell" if (delta_est - st.hedge_shares) > 0 else "buy"
                    self.order(SYM, side, qty=qty, order_type="market", venue=HEDGE_VENUE)
                    st.hedge_shares += (delta_est - st.hedge_shares)
                    st.last_delta = delta_est
                    self._save_state(st)

            # Exit if convergence
            if (abs(spread) <= EXIT_VOL) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ----- entries -----
        if r.get(_poskey(self.ctx.name)) is not None:
            return

        # Gate on both absolute vol gap and z‑score
        if not (abs(spread) >= ENTRY_VOL and abs(z) >= ENTRY_Z):
            return

        # Size by USD vega target: n_units = USD_VEGA_TARGET / vega_per_volpt_usd
        n_units = USD_VEGA_TARGET / max(1e-9, vega_per_volpt_usd)
        if n_units * S * 0.01 < MIN_TICKET_USD:
            return

        # Place synthetic straddle leg
        strad = _strad_sym(SYM, TENOR)
        if spread > 0:
            # IV rich => SHORT vol
            self.order(strad, "sell", qty=n_units, order_type="market", venue=VENUE_OPT)
            side = "short_vol"
        else:
            # IV cheap => LONG vol
            self.order(strad, "buy", qty=n_units, order_type="market", venue=VENUE_OPT)
            side = "long_vol"

        # Initial delta is ~0 at ATM; seed hedge values
        st_new = OpenState(
            side=side, n_units=n_units, hedge_shares=0.0, last_delta=0.0,
            entry_spread=spread, entry_z=z, ts_ms=int(time.time()*1000)
        )
        self._save_state(st_new)

    # ---------- state ----------
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

    # ---------- close ----------
    def _close(self, st: OpenState) -> None:
        strad = _strad_sym(SYM, TENOR)
        if st.side == "short_vol":
            self.order(strad, "buy", qty=st.n_units, order_type="market", venue=VENUE_OPT)
        else:
            self.order(strad, "sell", qty=st.n_units, order_type="market", venue=VENUE_OPT)

        # unwind hedge
        if st.hedge_shares != 0.0:
            side = "buy" if st.hedge_shares < 0 else "sell"
            self.order(SYM, side, qty=abs(st.hedge_shares), order_type="market", venue=HEDGE_VENUE)

        self._save_state(None)