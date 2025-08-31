# backend/strategies/diversified/dispersion_trade.py
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
Dispersion Trade (variance: index vs components)
------------------------------------------------
Classic setup: SHORT index variance and LONG single‑name variances (vega‑neutral-ish),
harvesting the correlation risk premium. Optionally flip to "reverse dispersion"
when the edge is meaningfully negative.

Paper execution uses **synthetic variance symbols** your OMS accepts:
  • "<SYM>.VAR<TENOR>"  e.g., SPX.VAR30, AAPL.VAR30
You can later map these to option strips / variance futures.

Inputs you already publish to Redis:
  HSET iv:imp:<TENOR> <SYM> <iv_decimal>      # annualized implied vols, index & each component
  (optional) HSET iv:real:<TENOR> <SYM> <rv>  # realized vols for anchor
  (optional) SET corr:expected:<INDEX> <rho>  # expected avg correlation anchor in [-1,1]
  HSET last_price <SYM> '{"price": ...}'      # for monitoring signal only (not required)

Edge metrics:
  • Basket variance with **zero correlation**: V0 = Σ (w_i^2 σ_i^2)
  • Index variance: VI = σ_I^2
  • Dispersion edge: D = V0 − VI  (≈ amount explainable by correlation term)
      If D is HIGH → index variance rich vs sum of singles → do classic dispersion:
         SHORT index VAR, LONG singles VAR
      If D is LOW/negative (rare) → reverse dispersion:
         LONG index VAR, SHORT singles VAR

We also compute **implied average correlation ρ_imp** and compare to an anchor
to guard/trend. Both D and z(D) must exceed thresholds to enter.
"""

# ============== CONFIG (env) ==============
REDIS_HOST = os.getenv("DSP_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("DSP_REDIS_PORT", "6379"))

INDEX  = os.getenv("DSP_INDEX", "SPX").upper()
TENOR  = os.getenv("DSP_TENOR", "30D").upper()

# Components with weights (sum ~ 1). "AAPL:0.07,MSFT:0.06,..."
COMPONENTS_ENV = os.getenv(
    "DSP_COMPONENTS",
    "AAPL:0.07,MSFT:0.06,AMZN:0.04,GOOGL:0.04,META:0.03,NVDA:0.05,JPM:0.02,JNJ:0.02,XOM:0.02,BRK.B:0.02"
)

# Entry/exit thresholds
ENTRY_EDGE = float(os.getenv("DSP_ENTRY_EDGE", "0.004"))   # absolute D edge in variance units (σ^2)
EXIT_EDGE  = float(os.getenv("DSP_EXIT_EDGE",  "0.0015"))
ENTRY_Z    = float(os.getenv("DSP_ENTRY_Z",    "1.5"))
EXIT_Z     = float(os.getenv("DSP_EXIT_Z",     "0.5"))

# Sizing (USD variance notional)
USD_INDEX_VAR   = float(os.getenv("DSP_USD_INDEX_VAR",   "60000"))
USD_SINGLE_VAR  = float(os.getenv("DSP_USD_SINGLE_VAR",  "8000"))  # per component leg
MAX_CONCURRENT  = int(os.getenv("DSP_MAX_CONCURRENT", "1"))

# Cadence
RECHECK_SECS = int(os.getenv("DSP_RECHECK_SECS", "10"))

# Venues (advisory)
VENUE_VAR = os.getenv("DSP_VENUE_VAR", "CBOE").upper()

# Redis keys
IV_IMP_KEY   = f"iv:imp:{TENOR}"        # HSET iv:imp:30D <SYM> <iv>
IV_REAL_KEY  = f"iv:real:{TENOR}"       # optional realized vols
EXP_CORR_KEY = f"corr:expected:{INDEX}" # SET corr:expected:SPX <rho>
LAST_PRICE_HKEY = os.getenv("DSP_LAST_PRICE_KEY", "last_price")

# ============== Redis ==============
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============== Helpers ==============
def _parse_components(env: str) -> List[Tuple[str, float]]:
    lst: List[Tuple[str, float]] = []
    for part in env.split(","):
        if ":" not in part: continue
        s, w = part.split(":", 1)
        try:
            lst.append((s.strip().upper(), float(w)))
        except Exception:
            pass
    tot = sum(w for _, w in lst) or 1.0
    return [(s, w / tot) for s, w in lst]

COMPS: List[Tuple[str, float]] = _parse_components(COMPONENTS_ENV)

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _getf(key: str) -> Optional[float]:
    v = r.get(key)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _last(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _implied_corr(sigma_I: float, comps: List[Tuple[float, float]]) -> Optional[float]:
    """ comps: [(w_i, sigma_i)] """
    if sigma_I <= 0 or len(comps) < 2: return None
    sI2 = sigma_I * sigma_I
    sum_w2s2 = sum(w*w*s*s for w, s in comps)
    sum_pairs = 0.0
    for i in range(len(comps)):
        wi, si = comps[i]
        for j in range(i+1, len(comps)):
            wj, sj = comps[j]
            sum_pairs += wi * wj * si * sj
    denom = 2.0 * sum_pairs
    if denom <= 1e-12: return None
    rho = (sI2 - sum_w2s2) / denom
    return max(-1.0, min(1.0, rho))

def _var_symbol(sym: str) -> str:
    return f"{sym}.VAR{TENOR}"

# ============== EWMA on D ==============
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

EWMA_ALPHA = float(os.getenv("DSP_EWMA_ALPHA", "0.05"))

def _ewma_key() -> str:
    return f"dsp:ewma:{INDEX}:{TENOR}"

def _load_ewma() -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============== State ==============
@dataclass
class OpenState:
    side: str   # "classic" (short index / long singles) or "reverse"
    entry_D: float
    entry_z: float
    entry_rho: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"dsp:open:{name}:{INDEX}:{TENOR}"

# ============== Strategy ==============
class DispersionTrade(Strategy):
    """
    Short index variance / long components when dispersion edge is high; reverse otherwise.
    """
    def __init__(self, name: str = "dispersion_trade", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "index": INDEX, "tenor": TENOR,
            "components": [{"symbol": s, "weight": w} for s, w in COMPS],
            "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate()

    # -------- core --------
    def _read_vols(self) -> Tuple[Optional[float], List[Tuple[str, float, Optional[float]]]]:
        idx = _hgetf(IV_IMP_KEY, INDEX)
        comps: List[Tuple[str, float, Optional[float]]] = []
        for s, w in COMPS:
            comps.append((s, w, _hgetf(IV_IMP_KEY, s)))
        return (idx, comps)

    def _evaluate(self) -> None:
        idx_iv, comps = self._read_vols()
        if idx_iv is None or any(sigma is None or sigma <= 0 for _, _, sigma in comps):
            return

        sigma_I = float(idx_iv)
        comp_pairs = [(float(w), float(s)) for (_, w, s) in comps] # type: ignore

        # V0 (zero‑corr basket variance) and VI
        V0 = sum(w*w*s*s for w, s in comp_pairs)
        VI = sigma_I * sigma_I
        D  = V0 - VI

        # implied correlation & anchor guard
        rho_imp = _implied_corr(sigma_I, comp_pairs) or 0.0
        rho_anchor = self._anchor_corr(idx_iv, comps, rho_imp)

        # EWMA stats on D
        ew = _load_ewma()
        m, v = ew.update(D)
        _save_ewma(ew)
        z = (D - m) / math.sqrt(max(v, 1e-12))

        # monitoring signal (bigger D → more positive)
        self.emit_signal(max(-1.0, min(1.0, (D - m) / 0.005)))

        st = self._load_state()

        # ---- exits ----
        if st:
            if (abs(D - m) <= EXIT_EDGE) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ---- entries ----
        if r.get(_poskey(self.ctx.name)) is not None:
            return
        if not (abs(D - m) >= ENTRY_EDGE and abs(z) >= ENTRY_Z):
            return

        if D > m:
            # Classic dispersion: SHORT index VAR / LONG singles VAR
            self._short_index_long_singles()
            self._save_state(OpenState(
                side="classic", entry_D=D, entry_z=z, entry_rho=rho_imp, ts_ms=int(time.time()*1000)
            ))
        else:
            # Reverse dispersion (rare)
            self._long_index_short_singles()
            self._save_state(OpenState(
                side="reverse", entry_D=D, entry_z=z, entry_rho=rho_imp, ts_ms=int(time.time()*1000)
            ))

    # -------- anchor/guards --------
    def _anchor_corr(self, idx_iv_imp: float, comps: List[Tuple[str, float, Optional[float]]], rho_now: float) -> float:
        exp = _getf(EXP_CORR_KEY)
        if exp is not None:
            return max(-1.0, min(1.0, exp))
        # fallback: realized vols implied correlation
        idx_rv = _hgetf(IV_REAL_KEY, INDEX)
        rv_comps = [(w, _hgetf(IV_REAL_KEY, s)) for (s, w, _) in comps]
        if idx_rv is not None and all((rv is not None and rv > 0) for w, rv in rv_comps):
            rho_real = _implied_corr(idx_rv, [(w, rv) for w, rv in rv_comps])  # type: ignore
            if rho_real is not None:
                return rho_real
        # last resort: use current implied
        return rho_now

    # -------- orders --------
    def _var(self, sym: str) -> str:
        return _var_symbol(sym)

    def _short_index_long_singles(self) -> None:
        self.order(self._var(INDEX), "sell", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
        for s, _ in COMPS:
            self.order(self._var(s), "buy", qty=USD_SINGLE_VAR, order_type="market", venue=VENUE_VAR)

    def _long_index_short_singles(self) -> None:
        self.order(self._var(INDEX), "buy", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
        for s, _ in COMPS:
            self.order(self._var(s), "sell", qty=USD_SINGLE_VAR, order_type="market", venue=VENUE_VAR)

    # -------- state io --------
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

    # -------- close --------
    def _close(self, st: OpenState) -> None:
        if st.side == "classic":
            self.order(self._var(INDEX), "buy", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
            for s, _ in COMPS:
                self.order(self._var(s), "sell", qty=USD_SINGLE_VAR, order_type="market", venue=VENUE_VAR)
        else:
            self.order(self._var(INDEX), "sell", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
            for s, _ in COMPS:
                self.order(self._var(s), "buy", qty=USD_SINGLE_VAR, order_type="market", venue=VENUE_VAR)
        self._save_state(None)