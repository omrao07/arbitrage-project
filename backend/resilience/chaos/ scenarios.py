# backend/sim/scenarios.py
"""
Scenario Runner for PolicySimulator (stdlib-only; YAML optional)

What this gives you
-------------------
- ScenarioSpec: declarative scenario with timeline blocks and shock overlays
- ScenarioRunner: executes a scenario on a PolicySimulator (from policy_sim.py)
- Presets: soft_landing, hard_landing, stagflation, crisis_liquidity
- YAML loader: define scenarios in YAML and run them without code changes
- Outputs: list of step snapshots [{"t":..,"ts":..,"signals":{...},"notes":{...}}, ...]

Usage
-----
from backend.sim.policy_sim import PolicySimConfig
from backend.sim.scenarios import ScenarioRunner, presets, ScenarioSpec

cfg = PolicySimConfig(seed=7, dt_days=1.0, horizon_days=180)
runner = ScenarioRunner(cfg)
snaps = runner.run(presets.soft_landing())
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # optional
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

from .policy_sim import PolicySimConfig, PolicySimulator, Shock # type: ignore


# --------------------------- data models --------------------------------

@dataclass
class ScenarioBlock:
    """One timeline block: run N steps in a desired regime, optionally adding shocks."""
    steps: int
    regime: Optional[str] = None        # "expansion" | "slowdown" | "inflation" | "crisis"
    add_shocks: Optional[List[Dict[str, Any]]] = None  # list of Shock-like dicts


@dataclass
class ScenarioSpec:
    """
    Scenario definition.

    - overrides: dotted-key overrides applied to PolicySimConfig before run
      e.g., {"banks.fed.start_rate": 0.055, "horizon_days": 240}
    - timeline: list[ScenarioBlock] executed in order
    - repeat: optional repetition count (>=1)
    """
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    repeat: int = 1
    notes: Dict[str, Any] = field(default_factory=dict)


# --------------------------- Runner -------------------------------------

class ScenarioRunner:
    """
    Runs a ScenarioSpec using a PolicySimulator.

    Usage:
        runner = ScenarioRunner(PolicySimConfig(...))
        snaps = runner.run(ScenarioSpec(...))
    """

    def __init__(self, base_cfg: PolicySimConfig):
        self.base_cfg = base_cfg

    def run(self, scenario: ScenarioSpec) -> List[Dict[str, Any]]:
        cfg = _apply_overrides(self.base_cfg, scenario.overrides)
        sim = PolicySimulator(cfg)

        # Map regime name -> index from cfg.regimes
        reg_map = {r.name: i for i, r in enumerate(cfg.regimes)}
        snaps: List[Dict[str, Any]] = []

        for _ in range(max(1, int(scenario.repeat))):
            for blk in scenario.timeline:
                block = _coerce_block(blk)

                # Force regime for this block if provided
                if block.regime:
                    if block.regime not in reg_map:
                        raise ValueError(f"Unknown regime '{block.regime}'. Available: {list(reg_map.keys())}")
                    sim.regime_ix = reg_map[block.regime]

                # Temporarily extend shocks for this block
                original = list(sim.cfg.shocks)
                if block.add_shocks:
                    sim.cfg.shocks = original + [_coerce_shock(d) for d in block.add_shocks]

                # Step through this block
                for _ in range(int(block.steps)):
                    snaps.append(sim.step())

                # Restore shocks
                sim.cfg.shocks = original

        if snaps:
            snaps[-1].set("scenario", {"name": scenario.name, "notes": scenario.notes}) # type: ignore
        return snaps

    # Convenience helpers
    def run_to_signals(self, scenario: ScenarioSpec) -> List[Dict[str, float]]:
        return [s["signals"] for s in self.run(scenario)]

    def run_to_json(self, scenario: ScenarioSpec, path: str) -> str:
        snaps = self.run(scenario)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snaps, f, indent=2)
        return path


# --------------------------- Presets ------------------------------------

class presets:
    """Handy pre-built scenarios."""

    @staticmethod
    def soft_landing(horizon_days: int = 180) -> ScenarioSpec:
        return ScenarioSpec(
            name="soft_landing",
            overrides={"horizon_days": horizon_days},
            timeline=[
                {"steps": horizon_days // 2, "regime": "expansion"},
                {"steps": horizon_days - (horizon_days // 2), "regime": "slowdown",
                 "add_shocks": [{"key": "ecb", "delta_bps": -25, "step": 15}]},
            ],
            notes={"story": "Growth cools, inflation fades, gradual easing."}
        )

    @staticmethod
    def hard_landing(horizon_days: int = 180) -> ScenarioSpec:
        return ScenarioSpec(
            name="hard_landing",
            overrides={"horizon_days": horizon_days},
            timeline=[
                {"steps": 30, "regime": "inflation"},
                {"steps": 30, "regime": "slowdown"},
                {"steps": horizon_days - 60, "regime": "crisis",
                 "add_shocks": [
                     {"key": "global", "risk_z_jump": +2.0, "liq_z_jump": -1.0, "step": 5},
                     {"key": "fed", "delta_bps": -75, "step": 10},
                 ]},
            ],
            notes={"story": "Inflation bite → growth cracks → crisis and rapid cuts."}
        )

    @staticmethod
    def stagflation(horizon_days: int = 240) -> ScenarioSpec:
        return ScenarioSpec(
            name="stagflation",
            overrides={"horizon_days": horizon_days},
            timeline=[
                {"steps": horizon_days // 2, "regime": "inflation",
                 "add_shocks": [{"key": "global", "infl_z_jump": +0.8, "prob": 0.02}]},
                {"steps": horizon_days // 2, "regime": "slowdown",
                 "add_shocks": [{"key": "ecb", "delta_bps": +25, "prob": 0.01}]},
            ],
            notes={"story": "Persistent inflation while growth slows; stop‑go policy."}
        )

    @staticmethod
    def crisis_liquidity(horizon_days: int = 120) -> ScenarioSpec:
        return ScenarioSpec(
            name="crisis_liquidity",
            overrides={"horizon_days": horizon_days},
            timeline=[
                {"steps": horizon_days // 3, "regime": "expansion"},
                {"steps": horizon_days // 3, "regime": "crisis",
                 "add_shocks": [{"key":"global","risk_z_jump":+2.5,"liq_z_jump":-1.2,"step":3}]},
                {"steps": horizon_days - 2*(horizon_days // 3), "regime": "slowdown",
                 "add_shocks": [{"key":"fed","delta_bps":-100,"step":1}]},
            ],
            notes={"story": "Sudden stress, liquidity dries up, then policy rescue."}
        )


# --------------------------- YAML I/O -----------------------------------

def load_yaml(path: str) -> List[ScenarioSpec]:
    """Load a list of ScenarioSpec from a YAML file."""
    if not _HAVE_YAML:
        raise RuntimeError("pyyaml not installed; cannot load YAML scenarios")
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if doc is None:
        return []
    if isinstance(doc, dict):
        doc = [doc]
    out: List[ScenarioSpec] = []
    for item in doc:
        if not isinstance(item, dict):
            continue
        out.append(ScenarioSpec(
            name=item.get("name", "scenario"),
            overrides=item.get("overrides", {}) or {},
            timeline=item.get("timeline", []) or [],
            repeat=int(item.get("repeat", 1) or 1),
            notes=item.get("notes", {}) or {},
        ))
    return out


# --------------------------- helpers ------------------------------------

def _coerce_block(d: Dict[str, Any]) -> ScenarioBlock:
    return ScenarioBlock(
        steps=int(d.get("steps", 0)),
        regime=d.get("regime"),
        add_shocks=d.get("add_shocks")
    )

def _coerce_shock(d: Dict[str, Any]) -> Shock:
    return Shock(
        key=str(d.get("key", "global")),
        delta_bps=float(d.get("delta_bps", 0.0)),
        risk_z_jump=float(d.get("risk_z_jump", 0.0)),
        liq_z_jump=float(d.get("liq_z_jump", 0.0)),
        infl_z_jump=float(d.get("infl_z_jump", 0.0)),
        step=(int(d["step"]) if d.get("step") is not None else None),
        prob=float(d.get("prob", 0.0)),
        name=str(d.get("name", "shock"))
    )

def _apply_overrides(cfg: PolicySimConfig, overrides: Dict[str, Any]) -> PolicySimConfig:
    """
    Apply dotted-key overrides to a deep copy of PolicySimConfig.
    Supported paths include nested fields and dict entries, e.g.:
        "horizon_days": 240
        "banks.fed.start_rate": 0.055
        "regimes[2].policy_drift_bps": 0.8
    """
    new_cfg = copy.deepcopy(cfg)

    def set_path(root: Any, path: str, value: Any) -> None:
        cur = root
        parts = path.split(".")
        for i, raw in enumerate(parts):
            key, idx = _split_index(raw)
            last = (i == len(parts) - 1)
            if isinstance(cur, dict):
                if last:
                    cur[key] = value; return
                cur = cur[key]
            else:
                if last:
                    setattr(cur, key, value); return
                cur = getattr(cur, key)
            if idx is not None:
                cur = cur[idx]

    for k, v in (overrides or {}).items():
        set_path(new_cfg, k, v)
    return new_cfg

def _split_index(token: str) -> Tuple[str, Optional[int]]:
    """Parse 'name[2]' -> ('name', 2)."""
    if "[" in token and token.endswith("]"):
        name, ix = token.split("[", 1)
        ix = ix[:-1]
        try:
            return name, int(ix)
        except Exception:
            return name, None
    return token, None


# --------------------------- tiny demo ----------------------------------

if __name__ == "__main__":
    from .policy_sim import PolicySimConfig # type: ignore
    runner = ScenarioRunner(PolicySimConfig(seed=11, dt_days=1.0, horizon_days=120))
    sc = presets.stagflation(horizon_days=90)
    snaps = runner.run(sc)
    print("[scenarios] ran", len(snaps), "steps in", sc.name)
    print("[scenarios] last regime:", snaps[-1]["regime"])
    sample = snaps[-1]["signals"]
    print("[scenarios] sample signals:", {k: sample[k] for k in list(sample.keys())[:8]})