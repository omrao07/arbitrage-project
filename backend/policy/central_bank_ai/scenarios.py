# backend/sim/scenarios.py
"""
Scenario engine for policy + macro simulation.

Works with backend/sim/policy_sim.py:
- Define scenarios as named bundles of config overrides and shock sequences.
- Run deterministic trajectories (e.g., taper tantrum, stagflation).
- Replay templates of historical episodes (2020 crisis, 2013 taper).
- Export signals dicts for your signal bus.

Usage
-----
from backend.sim.policy_sim import PolicySimulator, PolicySimConfig
from backend.sim.scenarios import ScenarioLibrary

lib = ScenarioLibrary()
sim = lib.build("stagflation", horizon_days=180)
for snap in sim.run():
    print(snap["t"], snap["signals"]["FED.ff_upper"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.sim.policy_sim import ( # type: ignore
    PolicySimConfig,
    PolicySimulator,
    Shock,
    RegimeSpec,
)

# ----------------------- Scenario Definitions ---------------------------

@dataclass
class Scenario:
    name: str
    description: str
    cfg_overrides: Dict[str, any] = field(default_factory=dict) # type: ignore
    shocks: List[Shock] = field(default_factory=list)
    regime_ix: Optional[int] = None   # force initial regime index


class ScenarioLibrary:
    """
    A registry of named scenarios with helpers to build simulators.
    """

    def __init__(self):
        self.scenarios: Dict[str, Scenario] = {}
        self._register_defaults()

    def _register_defaults(self):
        # --- 2013 Taper Tantrum ---
        self.register(Scenario(
            name="taper_tantrum",
            description="UST yields spike, risk-off, USD strengthens.",
            shocks=[
                Shock("fed", delta_bps=+75, step=10, name="rapid_hike"),
                Shock("global", risk_z_jump=+1.5, liq_z_jump=-1.0, step=12, name="risk_off"),
            ],
            cfg_overrides={"horizon_days": 90}
        ))

        # --- 2020 COVID Crisis ---
        self.register(Scenario(
            name="covid_crash",
            description="Massive cuts, liquidity injections, risk spike.",
            shocks=[
                Shock("fed", delta_bps=-150, step=5, name="emergency_cut"),
                Shock("ecb", delta_bps=-50, step=6, name="ecb_liquidity"),
                Shock("global", risk_z_jump=+3.0, liq_z_jump=+2.0, step=7, name="panic_liquidity"),
            ],
            cfg_overrides={"horizon_days": 60, "init_regime": 3}  # crisis
        ))

        # --- Stagflation scenario ---
        self.register(Scenario(
            name="stagflation",
            description="Inflation persists, growth slows, rates grind higher.",
            shocks=[
                Shock("fed", delta_bps=+25, prob=0.05, name="hawkish_bias"),
                Shock("ecb", delta_bps=+25, prob=0.05, name="hawkish_bias"),
            ],
            cfg_overrides={"horizon_days": 180, "init_regime": 2}  # inflation regime
        ))

        # --- Soft Landing ---
        self.register(Scenario(
            name="soft_landing",
            description="Inflation moderates, growth steady, gradual cuts.",
            shocks=[
                Shock("fed", delta_bps=-25, prob=0.02, name="dovish_glide"),
                Shock("ecb", delta_bps=-25, prob=0.02, name="dovish_glide"),
            ],
            cfg_overrides={"horizon_days": 180, "init_regime": 0}  # expansion
        ))

        # --- RBI Stress ---
        self.register(Scenario(
            name="rbi_stress",
            description="INR pressure, RBI forced hikes.",
            shocks=[
                Shock("rbi", delta_bps=+50, prob=0.03, name="rupee_defense"),
                Shock("global", risk_z_jump=+1.0, step=20, name="em_weakness"),
            ],
            cfg_overrides={"horizon_days": 120}
        ))

    # ---------------------------------------------------------------------

    def register(self, scenario: Scenario) -> None:
        self.scenarios[scenario.name] = scenario

    def list(self) -> List[str]:
        return list(self.scenarios.keys())

    def build(self, name: str, *, horizon_days: Optional[int] = None, seed: Optional[int] = None) -> PolicySimulator:
        """
        Return a PolicySimulator instance wired for this scenario.
        """
        sc = self.scenarios[name]
        cfg = PolicySimConfig()
        # apply overrides
        for k, v in sc.cfg_overrides.items():
            setattr(cfg, k, v)
        if horizon_days:
            cfg.horizon_days = horizon_days
        if seed is not None:
            cfg.seed = seed
        if sc.regime_ix is not None:
            cfg.init_regime = sc.regime_ix
        # add scenario-specific shocks
        cfg.shocks.extend(sc.shocks)
        return PolicySimulator(cfg)


# ----------------------- CLI Demo ---------------------------------------

if __name__ == "__main__":
    lib = ScenarioLibrary()
    print("Available scenarios:", lib.list())
    sim = lib.build("stagflation", seed=42)
    snaps = sim.run()
    print("[scenario] ran", len(snaps), "steps. Last regime:", snaps[-1]["regime"])
    s = snaps[-1]["signals"]
    sample = {k: s[k] for k in list(s.keys())[:8]}
    print("[scenario] sample signals:", sample)