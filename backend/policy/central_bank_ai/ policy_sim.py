# backend/sim/policy_sim.py
"""
Policy & Macro Simulator (stdlib-only; NumPy optional)

What this does
--------------
- Simulate short-rate/policy paths for multiple central banks (FED/ECB/RBI by default)
- Regime switching (expansion, slowdown, inflation, crisis) with transition matrix
- Shocks (scheduled or random) for surprise hikes/cuts and liquidity events
- Derive simple term structure (2y/5y/10y) from policy via affine toy model
- Emit a normalized signals dict compatible with your signal bus:
    fed.ff_upper, ecb.main_refi, rbi.repo, USD yield curve, risk_z, liquidity_z, etc.

Design
------
- No external deps; uses random + math. If NumPy is available, MC runs get faster.
- Deterministic seeding supported.
- Time unit = days; dt can be <1.0 for sub-daily steps.

Typical usage
-------------
from backend.sim.policy_sim import (
    PolicySimConfig, PolicySimulator, Shock, RegimeSpec
)

cfg = PolicySimConfig(seed=42, horizon_days=120)
sim = PolicySimulator(cfg)
for snap in sim.run():
    do_something_with(snap["signals"])

Or one-shot Monte Carlo:
paths = sim.mc_paths(n_paths=100, horizon_days=180)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Optional NumPy (for fast MC)
try:
    import numpy as _np  # type: ignore
    _HAVE_NP = True
except Exception:
    _HAVE_NP = False


# --------------------------- Config structures ---------------------------

@dataclass
class RegimeSpec:
    """
    One macro regime with policy drift/vol and macro factor drifts.
    """
    name: str
    policy_drift_bps: float       # daily drift of policy rate (bps/day)
    policy_vol_bps: float         # daily vol of policy rate innovations (bps/sqrt(day))
    infl_drift: float             # drift of inflation nowcast z
    liq_drift: float              # drift of liquidity z
    risk_drift: float             # drift of risk_z (VIX-style; positive = risk-off)
    infl_vol: float = 0.10
    liq_vol: float = 0.10
    risk_vol: float = 0.15


@dataclass
class Shock:
    """
    Discrete event applied at an absolute step index (t)
    or with probability p each step if step is None.
    """
    key: str                     # "fed", "ecb", "rbi", "global"
    delta_bps: float = 0.0       # add to policy rate (bps)
    risk_z_jump: float = 0.0     # jump in risk_z
    liq_z_jump: float = 0.0      # jump in liquidity_z
    infl_z_jump: float = 0.0     # jump in inflation nowcast z
    step: Optional[int] = None   # apply exactly at this step
    prob: float = 0.0            # else Bernoulli(prob) each step
    name: str = "shock"


@dataclass
class CBSpec:
    """
    Central bank specific parameters.
    """
    name: str
    key_rate_name: str              # output alias (e.g., "FED.ff_upper")
    start_rate: float               # fraction (e.g., 0.0525 = 5.25%)
    floor_rate: float = -0.005
    ceiling_rate: float = 0.15
    mean_revert: float = 0.03       # θ in simple OU pull (per day)
    neutral_rate: float = 0.02      # long-run neutral
    term_beta: Tuple[float, float, float] = (0.9, 0.6, 0.4)  # mapping to 2y/5y/10y


@dataclass
class PolicySimConfig:
    """
    Global configuration for the simulator.
    """
    seed: Optional[int] = None
    dt_days: float = 1.0
    horizon_days: int = 180
    start_ts: float = 0.0  # arbitrary epoch carried through snapshots

    # Defaults for 3 banks
    banks: Dict[str, CBSpec] = field(default_factory=lambda: {
        "fed": CBSpec("FED", "FED.ff_upper", start_rate=0.0525, neutral_rate=0.025),
        "ecb": CBSpec("ECB", "ECB.main_refi", start_rate=0.0400, neutral_rate=0.020),
        "rbi": CBSpec("RBI", "RBI.repo",     start_rate=0.0650, neutral_rate=0.055),
    })

    # Regimes & transitions (row-stochastic)
    regimes: List[RegimeSpec] = field(default_factory=lambda: [
        RegimeSpec("expansion", policy_drift_bps= 0.2, policy_vol_bps=1.2, infl_drift= 0.00, liq_drift= 0.02, risk_drift=-0.02),
        RegimeSpec("slowdown",  policy_drift_bps=-0.1, policy_vol_bps=1.5, infl_drift=-0.02, liq_drift=-0.01, risk_drift= 0.02),
        RegimeSpec("inflation", policy_drift_bps= 0.6, policy_vol_bps=2.5, infl_drift= 0.05, liq_drift=-0.03, risk_drift= 0.02),
        RegimeSpec("crisis",    policy_drift_bps=-1.0, policy_vol_bps=4.0, infl_drift=-0.05, liq_drift= 0.10, risk_drift= 0.15),
    ])
    trans_matrix: List[List[float]] = field(default_factory=lambda: [
        # to:    expan  slow   infl   crisis
        [       0.80,  0.10,  0.07,   0.03],   # from expansion
        [       0.25,  0.55,  0.10,   0.10],   # from slowdown
        [       0.30,  0.20,  0.40,   0.10],   # from inflation
        [       0.35,  0.35,  0.10,   0.20],   # from crisis
    ])
    init_regime: int = 0  # start in "expansion"

    # Global macro initial levels (z-scores)
    init_infl_z: float = 0.0
    init_liq_z: float = 0.0
    init_risk_z: float = 0.0

    # Shocks
    shocks: List[Shock] = field(default_factory=lambda: [
        Shock("global", risk_z_jump=+1.0, liq_z_jump=-0.5, name="risk_off_spike", prob=0.01),
        Shock("fed", delta_bps=+25, name="hawkish_surprise", prob=0.005),
        Shock("ecb", delta_bps=-25, name="dovish_surprise", prob=0.004),
    ])

    # Output toggles
    emit_yield_curve: bool = True
    emit_fx_proxies: bool = True
    emit_liquidity_proxy: bool = True


# --------------------------- Simulator core -----------------------------

class PolicySimulator:
    def __init__(self, cfg: PolicySimConfig):
        self.cfg = cfg
        if cfg.seed is not None:
            random.seed(cfg.seed)
            if _HAVE_NP:
                _np.random.seed(cfg.seed)

        # state
        self.t = 0
        self.ts = cfg.start_ts
        self.regime_ix = cfg.init_regime
        self.infl_z = cfg.init_infl_z
        self.liq_z = cfg.init_liq_z
        self.risk_z = cfg.init_risk_z
        self.rates: Dict[str, float] = {k: spec.start_rate for k, spec in cfg.banks.items()}

        # precompute cumulative regime transition rows
        self._cumT = [_cumsum_row(row) for row in cfg.trans_matrix]

    # ---- public API ----

    def reset(self) -> None:
        self.__init__(self.cfg)

    def step(self) -> Dict[str, any]: # type: ignore
        """
        Advance by dt and return a snapshot: {"t":..., "ts":..., "regime":..., "signals": {...}}
        """
        dt = self.cfg.dt_days
        reg = self.cfg.regimes[self.regime_ix]

        # --- macro latent factors (z) evolve
        self.infl_z = _ou_step(self.infl_z, reg.infl_drift * dt, reg.infl_vol * math.sqrt(dt))
        self.liq_z  = _ou_step(self.liq_z,  reg.liq_drift  * dt, reg.liq_vol  * math.sqrt(dt))
        self.risk_z = _ou_step(self.risk_z, reg.risk_drift * dt, reg.risk_vol * math.sqrt(dt))

        # --- policy rates per bank
        for k, spec in self.cfg.banks.items():
            # OU pull to neutral + regime drift + random shock
            drft = (reg.policy_drift_bps / 10_000.0) * dt
            pull = spec.mean_revert * dt * (spec.neutral_rate - self.rates[k])
            eps = (reg.policy_vol_bps / 10_000.0) * math.sqrt(dt) * _randn()
            self.rates[k] = _clamp(self.rates[k] + drft + pull + eps, spec.floor_rate, spec.ceiling_rate)

        # --- apply shocks
        applied: List[str] = []
        for s in self.cfg.shocks:
            fire = (s.step is not None and self.t == s.step) or (s.step is None and s.prob > 0 and random.random() < s.prob)
            if not fire:
                continue
            applied.append(s.name)
            if s.key == "global":
                self.risk_z += s.risk_z_jump
                self.liq_z  += s.liq_z_jump
                self.infl_z += s.infl_z_jump
            else:
                if s.delta_bps != 0.0 and s.key in self.cfg.banks:
                    self.rates[s.key] = _clamp(self.rates[s.key] + s.delta_bps / 10_000.0,
                                               self.cfg.banks[s.key].floor_rate, self.cfg.banks[s.key].ceiling_rate)

        # --- regime transition (Markov)
        self.regime_ix = _markov_next(self.regime_ix, self._cumT)

        # --- derive signals
        signals = self._emit_signals()

        # advance time
        self.t += 1
        self.ts += dt * 86400.0

        return {
            "t": self.t,
            "ts": self.ts,
            "regime": self.cfg.regimes[self.regime_ix].name,
            "signals": signals,
            "notes": {"applied_shocks": applied}
        }

    def run(self, *, horizon_days: Optional[int] = None) -> List[Dict[str, any]]: # type: ignore
        steps = int((horizon_days or self.cfg.horizon_days) / max(1e-9, self.cfg.dt_days))
        out = []
        for _ in range(steps):
            out.append(self.step())
        return out

    # ---- Monte Carlo (policy only, faster path; macro factors evolve too) ----

    def mc_paths(self, n_paths: int = 100, horizon_days: Optional[int] = None) -> Dict[str, List[List[float]]]:
        """
        Return dict of rate paths per bank: {"fed": [[..],..], "ecb":[[..],..], ...}
        """
        steps = int((horizon_days or self.cfg.horizon_days) / max(1e-9, self.cfg.dt_days))
        banks = list(self.cfg.banks.keys())
        # initialize paths
        paths = {k: [[self.cfg.banks[k].start_rate] for _ in range(n_paths)] for k in banks}
        regime = self.regime_ix

        if _HAVE_NP:
            # vectorized MC
            for t in range(steps):
                reg = self.cfg.regimes[regime]
                for k in banks:
                    spec = self.cfg.banks[k]
                    drft = (reg.policy_drift_bps / 10_000.0) * self.cfg.dt_days
                    pull = spec.mean_revert * self.cfg.dt_days * (spec.neutral_rate - _np.array([p[-1] for p in paths[k]]))
                    eps = (reg.policy_vol_bps / 10_000.0) * math.sqrt(self.cfg.dt_days) * _np.random.normal(size=n_paths)
                    nxt = _np.clip(_np.array([p[-1] for p in paths[k]]) + drft + pull + eps, spec.floor_rate, spec.ceiling_rate)
                    for i in range(n_paths):
                        paths[k][i].append(float(nxt[i]))
                regime = _markov_next(regime, self._cumT)
        else:
            # looped MC
            for t in range(steps):
                reg = self.cfg.regimes[regime]
                for k in banks:
                    spec = self.cfg.banks[k]
                    for i in range(n_paths):
                        cur = paths[k][i][-1]
                        drft = (reg.policy_drift_bps / 10_000.0) * self.cfg.dt_days
                        pull = spec.mean_revert * self.cfg.dt_days * (spec.neutral_rate - cur)
                        eps = (reg.policy_vol_bps / 10_000.0) * math.sqrt(self.cfg.dt_days) * _randn()
                        nxt = _clamp(cur + drft + pull + eps, spec.floor_rate, spec.ceiling_rate)
                        paths[k][i].append(nxt)
                regime = _markov_next(regime, self._cumT)

        return paths

    # ---- internals ----

    def _emit_signals(self) -> Dict[str, float]:
        sig: Dict[str, float] = {}

        # Policy key rates
        for k, spec in self.cfg.banks.items():
            sig[spec.key_rate_name] = self.rates[k]

        # Yield curves (affine toy: y_tau = a + b * policy + c1*infl - c2*liq + c3*risk)
        if self.cfg.emit_yield_curve:
            for k, spec in self.cfg.banks.items():
                b2, b5, b10 = spec.term_beta
                for tenor, b in (("2y", b2), ("5y", b5), ("10y", b10)):
                    y = (0.002
                         + b * self.rates[k]
                         + 0.0015 * self.infl_z
                         - 0.0010 * self.liq_z
                         + 0.0012 * max(0.0, self.risk_z))
                    sig[f"{spec.name}.y_{tenor}"] = max(-0.01, y)

        # Macro/regime factors
        sig["risk_z"] = self.risk_z
        sig["liquidity_z"] = self.liq_z
        sig["infl_nowcast"] = self.infl_z

        # Simple FX proxies (policy differentials + liquidity/risk)
        if self.cfg.emit_fx_proxies:
            # USD vs EUR / INR
            usd = self.rates["fed"]; eur = self.rates.get("ecb", usd); inr = self.rates.get("rbi", usd)
            sig["FX_USD_EUR_proxy"] = 10.0 * (usd - eur) + 0.5 * (-self.liq_z) + 0.7 * max(0.0, self.risk_z)
            sig["FX_USD_INR_proxy"] =  8.0 * (usd - inr) + 0.4 * (-self.liq_z) + 0.6 * max(0.0, self.risk_z)

        # Liquidity proxy stream
        if self.cfg.emit_liquidity_proxy:
            sig["liquidity_proxy"] = 0.5 * (-self.rates["fed"] + -0.5 * self.rates.get("ecb", 0.0)) + 0.3 * self.liq_z

        return sig


# --------------------------- helpers ------------------------------------

def _ou_step(x: float, drift: float, vol: float) -> float:
    return x + drift + vol * _randn()

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _randn() -> float:
    # Box–Muller if numpy not present
    if _HAVE_NP:
        return float(_np.random.normal())
    import random as _r
    u1 = max(1e-12, _r.random()); u2 = _r.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

def _cumsum_row(row: List[float]) -> List[float]:
    s = 0.0; out = []
    for v in row:
        s += float(v)
        out.append(s)
    # normalize (safety)
    if out[-1] <= 0:
        return [1.0] * len(row)
    return [v / out[-1] for v in out]

def _markov_next(ix: int, cumT: List[List[float]]) -> int:
    r = random.random()
    row = cumT[ix]
    for j, p in enumerate(row):
        if r <= p:
            return j
    return len(row) - 1


# --------------------------- tiny CLI demo ------------------------------

if __name__ == "__main__":
    cfg = PolicySimConfig(seed=7, dt_days=1/2, horizon_days=10)  # 12-hour steps, 10 days
    sim = PolicySimulator(cfg)
    snaps = sim.run()
    print("[policy_sim] steps:", len(snaps))
    print("[policy_sim] last regime:", snaps[-1]["regime"])
    s = snaps[-1]["signals"]
    sample = {k: s[k] for k in list(s.keys())[:10]}
    print("[policy_sim] sample signals:", sample)