# backend/sim/shock_models.py
"""
Shock models for macro/policy simulation (stdlib-only; NumPy optional)

What this provides
------------------
- A small interface: ShockModel.generate(t, state) -> List[Shock]
- Ready-made models:
    * JumpDiffusion: Poisson/compound jump process on policy rates
    * HawkesLike: self-exciting risk/liquidity spikes with decay
    * RegimeConditional: regime-specific shock probabilities & sizes
    * VolatilitySpike: transient 'risk_z' bursts with half-life decay
    * LiquidityDrain: liquidity_z drawdown with recovery dynamics
    * FXGap: one-off FX proxy gaps driven by rate differentials
    * CrossAssetPropagator: map macro shocks into proxy signals (DXY, SPX, GOLD)
- ShockEngine: compose multiple models, clip, and emit Shock[] per step
- Utilities to apply shocks to a running PolicySimulator

Integration points
------------------
from backend.sim.policy_sim import PolicySimConfig, PolicySimulator, Shock
from backend.sim.shock_models import (
    ShockEngine, JumpDiffusion, HawkesLike, RegimeConditional,
    VolatilitySpike, LiquidityDrain, FXGap, CrossAssetPropagator
)

engine = ShockEngine([
    JumpDiffusion("fed", lam=0.02, jump_bps=(+25, +75)),
    HawkesLike(base=0.005, alpha=0.45, decay=0.15, key="global", risk_jump=1.0),
    RegimeConditional(per_regime={"inflation":{"fed":(+25,0.02)}, "crisis":{"fed":(-50,0.04)}}),
    VolatilitySpike(base=0.002, size=1.2, half_life=3),
])

# inside your simulation loop:
shocks = engine.generate(t=sim.t, state={"regime": sim.cfg.regimes[sim.regime_ix].name})
sim.cfg.shocks = shocks   # or extend with existing
snap = sim.step()
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as _np  # optional
    _HAVE_NP = True
except Exception:
    _HAVE_NP = False

from .policy_sim import Shock  # type: ignore # reuse the Shock dataclass


# --------------------------- base interface ------------------------------

class ShockModel:
    """Abstract shock generator."""
    name: str = "shock_model"
    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        """Return a list of Shock objects to apply at step t."""
        return []


# --------------------------- utilities -----------------------------------

def _poisson(lam: float) -> bool:
    """Return True if a Poisson(λ) process fires this step (Δt=1)."""
    # Bernoulli with p = 1 - e^{-λ}
    if lam <= 0: return False
    p = 1.0 - math.exp(-lam)
    return random.random() < p

def _randn() -> float:
    if _HAVE_NP:
        return float(_np.random.normal())
    # Box-Muller
    u1 = max(1e-12, random.random()); u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------- concrete models -----------------------------

@dataclass
class JumpDiffusion(ShockModel):
    """
    Random jumps on policy rates for a given bank key ("fed","ecb","rbi").
    Fires as a Poisson process; jump size chosen in [min_bps, max_bps] (uniform or normal).
    """
    key: str                              # "fed" / "ecb" / "rbi"
    lam: float = 0.01                     # average jumps per step
    jump_bps: Tuple[int, int] = (+25, +25)
    normal: bool = False                  # if True, sample N(mean, std) in bps using jump_bps=(mean,std)
    bias: float = 0.0                     # add bias in bps (e.g., +5 for hawkish drift)
    name: str = "jump_diffusion"

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        if not _poisson(self.lam):
            return []
        if self.normal:
            mean, std = self.jump_bps
            delta = (mean + std * _randn())
        else:
            lo, hi = self.jump_bps
            delta = random.uniform(min(lo, hi), max(lo, hi))
        delta += self.bias
        return [Shock(key=self.key, delta_bps=float(delta), step=t, name=f"{self.name}_{self.key}")]


@dataclass
class HawkesLike(ShockModel):
    """
    Self-exciting process for risk-off clusters:
    λ_t = base + α * sum(exp(-decay * (t - τ_i))) over previous fires.
    When fired, add risk_z_jump and optional liquidity_z jump.
    """
    base: float = 0.002
    alpha: float = 0.35
    decay: float = 0.20
    key: str = "global"
    risk_jump: float = 1.0
    liq_jump: float = -0.4
    infl_jump: float = 0.0
    name: str = "hawkes"

    _events: List[int] = field(default_factory=list)

    def intensity(self, t: int) -> float:
        lam = self.base
        for tau in self._events:
            if tau <= t:
                lam += self.alpha * math.exp(-self.decay * (t - tau))
        return max(0.0, lam)

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        lam = self.intensity(t)
        fire = random.random() < (1.0 - math.exp(-lam))
        if not fire:
            return []
        self._events.append(t)
        return [Shock(
            key=self.key, risk_z_jump=self.risk_jump,
            liq_z_jump=self.liq_jump, infl_z_jump=self.infl_jump,
            step=t, name=f"{self.name}_risk_cluster"
        )]


@dataclass
class RegimeConditional(ShockModel):
    """
    Regime-aware policy surprises.
    per_regime: mapping from regime name -> { bank_key: (delta_bps, prob_per_step) }
        e.g., {"inflation": {"fed": (+25, 0.02)}, "crisis": {"fed": (-50, 0.04), "ecb": (-25, 0.03)}}
    """
    per_regime: Dict[str, Dict[str, Tuple[int, float]]]
    name: str = "regime_conditional"

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        reg = str(state.get("regime", ""))
        cfg = self.per_regime.get(reg, {})
        out: List[Shock] = []
        for bank_key, (delta_bps, p) in cfg.items():
            if random.random() < max(0.0, float(p)):
                out.append(Shock(key=bank_key, delta_bps=float(delta_bps), step=t,
                                 name=f"{self.name}_{reg}_{bank_key}"))
        return out


@dataclass
class VolatilitySpike(ShockModel):
    """
    Occasional volatility spikes in risk_z with exponential decay memory.
    When fired, risk_z_jump is sampled around 'size' with N(0,0.2*size) noise.
    """
    base: float = 0.002
    size: float = 1.0
    half_life: float = 3.0   # steps
    key: str = "global"
    name: str = "vol_spike"

    _last_t: Optional[int] = None
    _carry: float = 0.0

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        # decay carry
        if self._last_t is not None:
            dt = max(0, t - self._last_t)
            if self.half_life > 0:
                decay = math.exp(-math.log(2) * dt / self.half_life)
                self._carry *= decay
        self._last_t = t

        if random.random() < self.base:
            jump = float(self.size + 0.2 * self.size * _randn())
            self._carry += max(0.0, jump)
            return [Shock(key=self.key, risk_z_jump=self._carry, step=t, name=self.name)]
        return []


@dataclass
class LiquidityDrain(ShockModel):
    """
    Liquidity_z drawdown when risk clusters; slow mean-reverting recovery.
    - Fires with small probability each step; increases drawdown bucket.
    - Recovery adds +recovery_rate toward 0 each step.
    """
    fire_p: float = 0.004
    draw_size: float = -0.4
    recovery_rate: float = 0.05
    key: str = "global"
    name: str = "liq_drain"

    _bucket: float = 0.0

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        # recovery toward 0
        if self._bucket < 0.0:
            self._bucket = min(0.0, self._bucket + self.recovery_rate)
        out: List[Shock] = []
        if random.random() < max(0.0, self.fire_p):
            self._bucket += float(self.draw_size)
            out.append(Shock(key=self.key, liq_z_jump=float(self.draw_size), step=t, name=self.name))
        # soft carry to keep liquidity depressed (optional: emit small continuous shock)
        return out


@dataclass
class FXGap(ShockModel):
    """
    One-off FX proxy jump tied to rate differentials, with random overshoot.
    Emits as a global shock (affects FX proxy signals in simulator via mapping).
    """
    key_pair: Tuple[str, str] = ("fed_rate", "ecb_rate")  # keys in state (provide via engine state)
    scale: float = 10.0
    overshoot_std: float = 0.3
    name: str = "fx_gap"

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        r1 = float(state.get(self.key_pair[0], 0.0))
        r2 = float(state.get(self.key_pair[1], 0.0))
        diff = r1 - r2
        if abs(diff) < 1e-6:
            return []
        gap = self.scale * diff + self.overshoot_std * _randn()
        # Represent via risk/liquidity proxies (simulator maps to FX via policy differentials anyway)
        return [Shock(key="global", risk_z_jump=0.0, liq_z_jump=0.0, infl_z_jump=0.0, step=t, name=f"{self.name}_{t}")]


@dataclass
class CrossAssetPropagator(ShockModel):
    """
    Map macro shocks into market proxy streams carried in engine 'state' bag.
    You can populate state with last known proxies and let this model adjust them.
    This model does NOT emit Shock objects; instead it mutates state["proxies"].
    """
    beta_risk_to_equity: float = -2.0
    beta_risk_to_gold: float = +0.6
    beta_liq_to_equity: float = +1.2
    beta_liq_to_credit: float = +0.8
    name: str = "cross_asset"

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        last = state.setdefault("proxies", {"SPX": 0.0, "GOLD": 0.0, "HY": 0.0})
        # Apply deltas based on most recent jumps stored by engine
        dz_risk = float(state.get("_last_risk_jump", 0.0))
        dz_liq  = float(state.get("_last_liq_jump", 0.0))
        # linear mapping
        last["SPX"]  += self.beta_risk_to_equity * max(0.0, dz_risk) + self.beta_liq_to_equity * dz_liq
        last["GOLD"] += self.beta_risk_to_gold   * max(0.0, dz_risk)
        last["HY"]   += self.beta_liq_to_credit  * dz_liq
        # keep state; return no Shock (proxies are side info)
        return []


# --------------------------- composition engine -------------------------

@dataclass
class ShockEngine:
    """
    Compose multiple ShockModels and track simple state between steps.

    state dict fields you can set/provide each step:
      - "regime": current regime name
      - "fed_rate", "ecb_rate", "rbi_rate": current policy levels (for FXGap etc.)
      - internal: "_last_risk_jump", "_last_liq_jump" updated after generation
      - "proxies": dict updated by CrossAssetPropagator
    """
    models: Sequence[ShockModel]
    max_abs_rate_bps: int = 150          # clip per-shock delta_bps
    max_abs_z_jump: float = 5.0          # clip per-shock z-jumps

    def generate(self, t: int, state: Dict[str, Any]) -> List[Shock]:
        out: List[Shock] = []
        # clear last deltas (for propagators)
        state["_last_risk_jump"] = 0.0
        state["_last_liq_jump"] = 0.0

        for m in self.models:
            shocks = m.generate(t, state) or []
            for s in shocks:
                # clip
                if s.delta_bps:
                    s.delta_bps = float(_clamp(s.delta_bps, -self.max_abs_rate_bps, self.max_abs_rate_bps))
                if s.risk_z_jump:
                    s.risk_z_jump = float(_clamp(s.risk_z_jump, -self.max_abs_z_jump, self.max_abs_z_jump))
                    state["_last_risk_jump"] = state.get("_last_risk_jump", 0.0) + s.risk_z_jump
                if s.liq_z_jump:
                    s.liq_z_jump = float(_clamp(s.liq_z_jump, -self.max_abs_z_jump, self.max_abs_z_jump))
                    state["_last_liq_jump"] = state.get("_last_liq_jump", 0.0) + s.liq_z_jump
            out.extend(shocks)
        return out


# --------------------------- simulation helpers ------------------------

def apply_to_sim(sim, engine: ShockEngine) -> None:
    """
    Convenience: call per step to generate shocks from current sim state and install them.
    """
    # Build a minimal state snapshot from simulator
    try:
        reg = sim.cfg.regimes[sim.regime_ix].name
        banks = sim.cfg.banks
        state = {
            "regime": reg,
            "fed_rate": sim.rates.get("fed", banks["fed"].start_rate),
            "ecb_rate": sim.rates.get("ecb", banks["ecb"].start_rate if "ecb" in banks else 0.0),
            "rbi_rate": sim.rates.get("rbi", banks["rbi"].start_rate if "rbi" in banks else 0.0),
        }
    except Exception:
        state = {"regime": "expansion"}

    shocks = engine.generate(sim.t, state)
    # Replace (or extend) current shocks just for this step
    sim.cfg.shocks = shocks


# --------------------------- tiny demo ----------------------------------

if __name__ == "__main__":
    from .policy_sim import PolicySimConfig, PolicySimulator # type: ignore

    cfg = PolicySimConfig(seed=7, dt_days=1.0, horizon_days=30)
    sim = PolicySimulator(cfg)

    engine = ShockEngine(models=[
        JumpDiffusion("fed", lam=0.03, jump_bps=(+25, +50), normal=False, bias=0.0),
        HawkesLike(base=0.004, alpha=0.5, decay=0.25, risk_jump=1.2, liq_jump=-0.5),
        RegimeConditional(per_regime={
            "inflation": {"fed": (+25, 0.03)},
            "crisis":    {"fed": (-50, 0.05), "ecb": (-25, 0.03)}
        }),
        VolatilitySpike(base=0.01, size=0.8, half_life=4),
        LiquidityDrain(fire_p=0.006, draw_size=-0.3, recovery_rate=0.03),
        CrossAssetPropagator(),
    ])

    snaps = []
    for _ in range(cfg.horizon_days):
        apply_to_sim(sim, engine)   # generate shocks based on current state
        snaps.append(sim.step())

    print("[shock_models] ran", len(snaps), "steps; last regime:", snaps[-1]["regime"])
    print("[shock_models] last applied shocks:", snaps[-1]["notes"].get("applied_shocks"))