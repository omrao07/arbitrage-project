# backend/strategies/diversified/correlation_swap_arbitrage.py
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
Correlation Swap / Dispersion Arbitrage
---------------------------------------
We approximate the classic variance-dispersion trade.

Inputs (from Redis):
  - Implied vols (annualized decimals) for index & components at a chosen tenor:
      HSET iv:imp:<TENOR> <SYM> <vol>      e.g., HSET iv:imp:30D SPX 0.19 ; HSET iv:imp:30D AAPL 0.30
  - Optional realized vols (annualized) for a correlation anchor:
      HSET iv:real:<TENOR> <SYM> <vol>
  - Optional direct expected correlation anchor:
      SET corr:expected:<INDEX> <rho>      e.g., SET corr:expected:SPX 0.25

Execution (paper):
  - Synthetic variance exposures via symbols like "<SYM>.VAR<TENOR>"
    Example: "SPX.VAR30" (index variance), "AAPL.VAR30" (single-name variance).
    Your OMS will just accept these as abstract instruments for paper trading.

Positioning idea:
  - Let σ_I be index IV, σ_i component IVs, w_i weights (sum to 1).
  - Implied average correlation ρ_imp from:
        σ_I^2 = Σ w_i^2 σ_i^2 + 2 ρ_imp Σ_{i<j} w_i w_j σ_i σ_j
    Solve for ρ_imp.
  - Compare ρ_imp to an anchor ρ_anchor (from realized vols or SET).
  - If Δρ = ρ_imp - ρ_anchor > +ENTRY_RHO and z >= ENTRY_Z → SHORT index VAR, LONG components VAR (dispersion).
  - If Δρ < -ENTRY_RHO and z <= -ENTRY_Z → LONG index VAR, SHORT components VAR.

This module is restart-safe and uses Redis to persist EWMA stats and open state.
"""

# ===================== CONFIG (env) =====================
REDIS_HOST = os.getenv("CORR_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CORR_REDIS_PORT", "6379"))

INDEX = os.getenv("CORR_INDEX", "SPX").upper()
# Components as "SYMA:0.06,SYMB:0.05,..." (weights sum ≈ 1). Defaults are examples.
COMPONENTS_ENV = os.getenv(
    "CORR_COMPONENTS",
    "AAPL:0.07,MSFT:0.06,AMZN:0.04,GOOGL:0.04,META:0.02,NVDA:0.05,JPM:0.02,JNJ:0.02,XOM:0.02,BRK.B:0.02"
)

# Tenor string for Redis keys and synthetic symbols, e.g., "30D" or "3M"
TENOR = os.getenv("CORR_TENOR", "30D").upper()

# Thresholds
ENTRY_RHO = float(os.getenv("CORR_ENTRY_RHO", "0.08"))   # absolute correlation gap (e.g., 0.08 = 8 pts)
EXIT_RHO  = float(os.getenv("CORR_EXIT_RHO",  "0.03"))
ENTRY_Z   = float(os.getenv("CORR_ENTRY_Z",   "1.5"))    # z-score on Δρ
EXIT_Z    = float(os.getenv("CORR_EXIT_Z",    "0.5"))

# EWMA for Δρ mean/var (event-based)
EWMA_ALPHA = float(os.getenv("CORR_EWMA_ALPHA", "0.05"))

# Sizing (variance notional per leg)
USD_INDEX_VAR   = float(os.getenv("CORR_USD_INDEX_VAR", "50000"))
USD_SN_VAR_EACH = float(os.getenv("CORR_USD_SN_VAR_EACH", "7000"))  # per component leg
MAX_CONCURRENT  = int(os.getenv("CORR_MAX_CONCURRENT", "1"))

# Recompute cadence (seconds)
RECHECK_SECS = int(os.getenv("CORR_RECHECK_SECS", "10"))

# Venue hint (advisory; OMS can ignore)
VENUE_VAR = os.getenv("CORR_VENUE_VAR", "CBOE").upper()

# Redis key formats
IV_IMP_KEY   = f"iv:imp:{TENOR}"     # HSET iv:imp:30D <SYM> <vol>
IV_REAL_KEY  = f"iv:real:{TENOR}"    # HSET iv:real:30D <SYM> <vol>
EXP_CORR_KEY = f"corr:expected:{INDEX}"  # SET corr:expected:SPX <rho>

# Last price store (used only to emit a signal proxy; variance exposures ignore spot)
LAST_PRICE_HKEY = os.getenv("CORR_LAST_PRICE_KEY", "last_price")

# ===================== Redis =====================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ===================== Helpers =====================
def _parse_components(env: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    s = env.strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        sym, w = part.split(":", 1)
        try:
            out.append((sym.strip().upper(), float(w)))
        except Exception:
            continue
    # renormalize weights to sum to 1
    total = sum(w for _, w in out) or 1.0
    out = [(sym, w / total) for sym, w in out]
    return out

COMPONENTS: List[Tuple[str, float]] = _parse_components(COMPONENTS_ENV)

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        try:
            return float(json.loads(v))
        except Exception:
            return None

def _getf(key: str) -> Optional[float]:
    v = r.get(key)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _last_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception: return None

def _implied_corr(index_iv: float, comp_ivs: List[Tuple[float, float]]) -> Optional[float]:
    """
    comp_ivs: list of (weight, sigma_i)
    Solve for average ρ in: σ_I^2 = Σ w_i^2 σ_i^2 + 2ρ Σ_{i<j} w_i w_j σ_i σ_j
    """
    if index_iv is None or index_iv <= 0:
        return None
    # components must all be valid
    vals = [(w, s) for (w, s) in comp_ivs if (w > 0 and s is not None and s > 0)]
    if len(vals) < 2:
        return None
    sigma_I2 = index_iv * index_iv
    sum_w2_sigma2 = sum(w * w * s * s for w, s in vals)
    # compute Σ_{i<j} w_i w_j σ_i σ_j
    sum_pair = 0.0
    for i in range(len(vals)):
        wi, si = vals[i]
        for j in range(i + 1, len(vals)):
            wj, sj = vals[j]
            sum_pair += wi * wj * si * sj
    denom = 2.0 * sum_pair
    if denom <= 1e-12:
        return None
    rho = (sigma_I2 - sum_w2_sigma2) / denom
    # clamp to sensible range
    return max(-1.0, min(1.0, rho))

# ============ EWMA mean/var tracker ============
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
    return f"corrswap:ewma:{INDEX}:{TENOR}"

def _load_ewma(alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============ State =============
def _poskey(name: str) -> str:
    return f"corrswap:open:{name}:{INDEX}:{TENOR}"

@dataclass
class OpenState:
    side: str           # "short_index_long_sn" OR "long_index_short_sn"
    entry_rho_imp: float
    entry_rho_anchor: float
    entry_delta: float   # rho_imp - rho_anchor at entry
    ts_ms: int
    # (we don’t track greeks here since variance legs are synthetic in paper)

# ============ Strategy ============
class CorrelationSwapArbitrage(Strategy):
    """
    Dispersion/correlation arbitrage with synthetic variance legs.
    """
    def __init__(self, name: str = "correlation_swap_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        # Announce index + components for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "index": INDEX, "tenor": TENOR,
            "components": [{"symbol": s, "weight": w} for s, w in COMPONENTS],
            "ts": int(time.time() * 1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate()

    # ---------- Core evaluation ----------
    def _read_vols(self) -> Tuple[Optional[float], List[Tuple[str, float, Optional[float]]]]:
        idx_iv = _hgetf(IV_IMP_KEY, INDEX)
        comps: List[Tuple[str, float, Optional[float]]] = []
        for sym, w in COMPONENTS:
            comps.append((sym, w, _hgetf(IV_IMP_KEY, sym)))
        return idx_iv, comps

    def _anchor_corr(self, idx_iv_imp: Optional[float], comps_imp: List[Tuple[str, float, Optional[float]]]) -> Optional[float]:
        # Priority 1: explicit expected correlation
        exp = _getf(EXP_CORR_KEY)
        if exp is not None:
            return max(-1.0, min(1.0, exp))
        # Priority 2: realized vols proxy using iv:real:<TENOR>
        idx_rv = _hgetf(IV_REAL_KEY, INDEX)
        rv_comps = [(w, _hgetf(IV_REAL_KEY, s)) for (s, w, _) in comps_imp]
        if idx_rv is not None and all((s is not None and s > 0) for _, s in rv_comps if _ is not None):
            rho_real = _implied_corr(idx_rv, rv_comps)  # use same formula with realized vols
            if rho_real is not None:
                return rho_real
        # Fallback: light historical anchor via EWMA mean of Δρ + current implied
        # (If no ewma history, just return a conservative 0.2).
        ew = _load_ewma(EWMA_ALPHA)
        if ew.mean != 0.0 and idx_iv_imp is not None:
            # if ew.mean stores recent average delta (ρ_imp - ρ_anchor), rearrange:
            # ρ_anchor ≈ ρ_imp - ew.mean
            comp_list = [(w, (s or 0.0)) for (_, w, s) in comps_imp]
            rho_imp_now = _implied_corr(idx_iv_imp, comp_list)
            if rho_imp_now is not None:
                return max(-1.0, min(1.0, rho_imp_now - ew.mean))
        return 0.20  # very coarse default

    def _evaluate(self) -> None:
        idx_iv, comps = self._read_vols()
        if idx_iv is None or any(sigma is None or sigma <= 0 for _, _, sigma in comps):
            # insufficient data
            return

        comp_list = [(w, float(sigma)) for (_, w, sigma) in comps]
        rho_imp = _implied_corr(float(idx_iv), comp_list)
        if rho_imp is None:
            return

        rho_anchor = self._anchor_corr(idx_iv, comps)
        if rho_anchor is None:
            return

        delta = rho_imp - rho_anchor

        # Maintain EWMA stats for z-score on Δρ
        ew = _load_ewma(EWMA_ALPHA)
        m, v = ew.update(delta)
        _save_ewma(ew)
        z = (delta - m) / math.sqrt(max(v, 1e-12))

        # Emit strategy-level signal for monitoring (squashed Δρ)
        self.emit_signal(max(-1.0, min(1.0, math.tanh(delta / 0.1))))  # scale 0.1 ~ 10 corr pts

        # Manage position
        st = self._load_state()

        # Exits
        if st:
            if (abs(delta) <= EXIT_RHO) or (abs(z) <= EXIT_Z):
                self._close(st)
                return
            # otherwise, hold
            return

        # Entries (single concurrent position per instance)
        if r.get(_poskey(self.ctx.name)) is not None:
            return

        if delta >= ENTRY_RHO and z >= ENTRY_Z:
            # ρ_imp too HIGH → dispersion:
            # SHORT index variance, LONG single-name variance
            self._short_index_long_singles()
            self._save_state(OpenState(
                side="short_index_long_sn",
                entry_rho_imp=rho_imp, entry_rho_anchor=rho_anchor,
                entry_delta=delta, ts_ms=int(time.time()*1000)
            ))
        elif delta <= -ENTRY_RHO and z <= -ENTRY_Z:
            # ρ_imp too LOW → reverse dispersion:
            # LONG index variance, SHORT single-name variance
            self._long_index_short_singles()
            self._save_state(OpenState(
                side="long_index_short_sn",
                entry_rho_imp=rho_imp, entry_rho_anchor=rho_anchor,
                entry_delta=delta, ts_ms=int(time.time()*1000)
            ))

    # ---------- Orders (synthetic variance exposures) ----------
    def _var_symbol(self, sym: str) -> str:
        return f"{sym}.VAR{TENOR}"

    def _short_index_long_singles(self) -> None:
        # index leg
        self.order(self._var_symbol(INDEX), "sell", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
        # components (proportional to index notional by weights)
        for sym, w in COMPONENTS:
            usd = USD_SN_VAR_EACH  # flat per-leg; you can set usd = w * USD_INDEX_VAR for proportional sizing
            self.order(self._var_symbol(sym), "buy", qty=usd, order_type="market", venue=VENUE_VAR)

    def _long_index_short_singles(self) -> None:
        self.order(self._var_symbol(INDEX), "buy", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
        for sym, _ in COMPONENTS:
            usd = USD_SN_VAR_EACH
            self.order(self._var_symbol(sym), "sell", qty=usd, order_type="market", venue=VENUE_VAR)

    # ---------- State ----------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw:
            return None
        try:
            o = json.loads(raw)
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # ---------- Close ----------
    def _close(self, st: OpenState) -> None:
        if st.side == "short_index_long_sn":
            self.order(self._var_symbol(INDEX), "buy", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
            for sym, _ in COMPONENTS:
                self.order(self._var_symbol(sym), "sell", qty=USD_SN_VAR_EACH, order_type="market", venue=VENUE_VAR)
        else:
            self.order(self._var_symbol(INDEX), "sell", qty=USD_INDEX_VAR, order_type="market", venue=VENUE_VAR)
            for sym, _ in COMPONENTS:
                self.order(self._var_symbol(sym), "buy", qty=USD_SN_VAR_EACH, order_type="market", venue=VENUE_VAR)
        self._save_state(None)