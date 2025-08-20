# backend/strategies/diversified/forward_start_option_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Forward‑Start Option Arbitrage (paper)
-------------------------------------
Idea:
  • For underlier S with term‑structure IVs σ(T), the *fair* forward vol for the slice T1→T2 is:
        σ_fwd^2 = (σ(T2)^2 * T2  −  σ(T1)^2 * T1) / (T2 − T1)
    (all times in YEARS)
  • A forward‑start ATM straddle that 'starts' at T1 and expires at T2 should be priced off σ_fwd
    with effective maturity (T2 − T1).
  • If market forward‑start IV (σ_mkt) is rich vs σ_fwd ⇒ **sell** forward‑start vol (short FWD‑straddle).
    If cheap ⇒ **buy** forward‑start vol (long FWD‑straddle).

This module:
  • Reads spot S, term IVs at T1 & T2, and (optionally) a direct market IV for the forward‑start.
  • Computes σ_fwd via variance‑additivity.
  • Gates entries by absolute vol gap + z‑score (EWMA).
  • Sizes by USD vega of an ATM forward‑start straddle (simple BS ATM vega at T=T2−T1).
  • Routes **paper** orders using synthetic symbol:
        FWDSTRAD:<SYM>:<T1><T2>
    which later you can map in your options adapter to the actual strip.

Redis you already publish elsewhere:
  HSET last_price         <SYM>            '{"price": <spot>}'             # underlier
  HSET iv:imp:<T>         <SYM>            <iv_decimal>                    # e.g., iv:imp:30D, iv:imp:180D
  (optional) HSET iv:fwd:<T1>-<T2> <SYM>   <iv_decimal>                    # direct market FWD IV, if you have it

If iv:fwd is absent, we’ll **proxy σ_mkt** as σ_fwd + small bias guard (i.e., no trade unless iv:fwd provided).
"""

# ======================= CONFIG (env) =======================
REDIS_HOST = os.getenv("FSO_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("FSO_REDIS_PORT", "6379"))

SYM      = os.getenv("FSO_SYMBOL", "ACME").upper()
T1_TENOR = os.getenv("FSO_T1", "30D").upper()
T2_TENOR = os.getenv("FSO_T2", "180D").upper()

# Thresholds (vol pts & z)
ENTRY_VOL = float(os.getenv("FSO_ENTRY_VOL", "0.05"))   # 5 vol pts
EXIT_VOL  = float(os.getenv("FSO_EXIT_VOL",  "0.02"))
ENTRY_Z   = float(os.getenv("FSO_ENTRY_Z",   "1.3"))
EXIT_Z    = float(os.getenv("FSO_EXIT_Z",    "0.5"))

# Sizing
USD_VEGA_TARGET = float(os.getenv("FSO_USD_VEGA_TARGET", "2000"))  # per package
MAX_CONCURRENT  = int(os.getenv("FSO_MAX_CONCURRENT", "1"))
MIN_TICKET_USD  = float(os.getenv("FSO_MIN_TICKET_USD", "200"))

# Cadence / stats
RECHECK_SECS = int(os.getenv("FSO_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("FSO_EWMA_ALPHA", "0.06"))

VENUE_OPT = os.getenv("FSO_VENUE_OPT", "CBOE").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("FSO_LAST_PRICE_KEY", "last_price")
IV_T1_KEY       = f"iv:imp:{T1_TENOR}"
IV_T2_KEY       = f"iv:imp:{T2_TENOR}"
IV_FWD_KEY      = f"iv:fwd:{T1_TENOR}-{T2_TENOR}"     # optional direct market quote

# ======================= Redis =======================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ======================= utils =======================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception: return None

def _hgetf(hkey: str, field: str) -> Optional[float]:
    v = r.hget(hkey, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _tenor_to_years(t: str) -> float:
    t = t.upper().strip()
    if t.endswith("D"):
        return max(1e-9, float(t[:-1] or 1.0) / 365.0)
    if t.endswith("M"):
        return max(1e-9, (30.0 * float(t[:-1] or 1.0)) / 365.0)
    if t.endswith("Y"):
        return max(1e-9, float(t[:-1] or 1.0))
    return max(1e-9, 30.0 / 365.0)

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _atm_straddle_vega_usd(S: float, sigma: float, T: float, q: float = 0.0) -> float:
    """ATM straddle vega per 1 vol‑pt (0.01), in USD."""
    if S <= 0 or sigma <= 0 or T <= 0:
        return 0.0
    # Black‑Scholes ATM d1 with K=S
    srt = sigma * math.sqrt(T)
    d1 = (0.5 * sigma * sigma * T - q * T) / max(1e-12, srt)  # simplified ATM d1 (r ignored)
    vega_call = S * math.exp(-q * T) * math.sqrt(T) * _norm_pdf(d1)   # per 1.00 change in vol
    vega_strad = 2.0 * vega_call
    return vega_strad * 0.01  # per vol‑pt

def _fwdstrad_sym(sym: str, t1: str, t2: str) -> str:
    return f"FWDSTRAD:{sym}:{t1}-{t2}"

# ======================= EWMA =======================
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

def _ewma_key(sym: str, t1: str, t2: str) -> str:
    return f"fso:ewma:{sym}:{t1}-{t2}"

def _load_ewma(sym: str, t1: str, t2: str) -> EwmaMV:
    raw = r.get(_ewma_key(sym, t1, t2))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(sym: str, t1: str, t2: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(sym, t1, t2), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ======================= State =======================
@dataclass
class OpenState:
    side: str          # "long_vol" or "short_vol"
    n_units: float
    entry_gap: float   # σ_mkt - σ_fair  (vol pts)
    entry_z: float
    ts_ms: int

def _poskey(name: str, sym: str, t1: str, t2: str) -> str:
    return f"fso:open:{name}:{sym}:{t1}-{t2}"

# ======================= Strategy =======================
class ForwardStartOptionArbitrage(Strategy):
    """
    Trade market forward‑start IV vs fair forward vol from the term‑structure.
    """
    def __init__(self, name: str = "forward_start_option_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "symbol": SYM, "t1": T1_TENOR, "t2": T2_TENOR, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- Core ---------------
    def _inputs(self) -> Optional[Tuple[float, float, float, float, float]]:
        S = _hget_price(SYM)
        if S is None or S <= 0:
            return None
        sig1 = _hgetf(IV_T1_KEY, SYM)
        sig2 = _hgetf(IV_T2_KEY, SYM)
        if sig1 is None or sig2 is None or sig1 <= 0 or sig2 <= 0:
            return None
        T1 = _tenor_to_years(T1_TENOR)
        T2 = _tenor_to_years(T2_TENOR)
        if T2 <= T1:
            return None
        sig_fwd = math.sqrt(max(1e-12, (sig2*sig2*T2 - sig1*sig1*T1) / (T2 - T1)))
        # Market forward IV if available
        sig_mkt = _hgetf(IV_FWD_KEY, SYM)
        return S, T1, T2, sig_fwd, (sig_mkt if sig_mkt and sig_mkt > 0 else float("nan"))

    def _evaluate(self) -> None:
        vals = self._inputs()
        if not vals:
            return
        S, T1, T2, sig_fair, sig_mkt = vals
        Tf = T2 - T1

        # If no market forward IV is provided, bail (prevents false positives).
        if not (sig_mkt == sig_mkt):  # NaN check
            return

        # Gap (vol pts)
        gap = sig_mkt - sig_fair

        # EWMA stats on the gap (in vol pts)
        ew = _load_ewma(SYM, T1_TENOR, T2_TENOR)
        m, v = ew.update(gap)
        _save_ewma(SYM, T1_TENOR, T2_TENOR, ew)
        z = (gap - m) / math.sqrt(max(v, 1e-12))

        # monitoring signal: positive when σ_mkt > σ_fair (rich)
        self.emit_signal(max(-1.0, min(1.0, gap / 0.05)))

        st = self._load_state()
        # ----- exits -----
        if st:
            if (abs(gap) <= EXIT_VOL) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ----- entries -----
        if r.get(_poskey(self.ctx.name, SYM, T1_TENOR, T2_TENOR)) is not None:
            return
        if not (abs(gap) >= ENTRY_VOL and abs(z) >= ENTRY_Z):
            return

        # Size by vega of a forward‑start ATM straddle using σ_mkt, T=Tf
        vega_per_volpt = _atm_straddle_vega_usd(S, max(1e-6, sig_mkt), Tf)
        if vega_per_volpt <= 0:
            return
        n_units = USD_VEGA_TARGET / vega_per_volpt
        if n_units * vega_per_volpt < MIN_TICKET_USD:
            return

        sym = _fwdstrad_sym(SYM, T1_TENOR, T2_TENOR)
        if gap > 0:
            # Market forward vol rich → SHORT forward‑start straddle
            self.order(sym, "sell", qty=n_units, order_type="market", venue=VENUE_OPT)
            side = "short_vol"
        else:
            # Market forward vol cheap → LONG forward‑start straddle
            self.order(sym, "buy", qty=n_units, order_type="market", venue=VENUE_OPT)
            side = "long_vol"

        self._save_state(OpenState(
            side=side, n_units=n_units, entry_gap=gap, entry_z=z, ts_ms=int(time.time()*1000)
        ))

    # --------------- state helpers ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, SYM, T1_TENOR, T2_TENOR))
        if not raw: return None
        try:
            return OpenState(**json.loads(raw))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name, SYM, T1_TENOR, T2_TENOR)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # --------------- close ---------------
    def _close(self, st: OpenState) -> None:
        sym = _fwdstrad_sym(SYM, T1_TENOR, T2_TENOR)
        if st.side == "short_vol":
            self.order(sym, "buy", qty=st.n_units, order_type="market", venue=VENUE_OPT)
        else:
            self.order(sym, "sell", qty=st.n_units, order_type="market", venue=VENUE_OPT)
        self._save_state(None)