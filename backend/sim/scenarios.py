# backend/risk/scenarios.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from backend.risk.policy_sim import PolicyShock, RateShock, EquityShock, FXShock, VolShock # type: ignore


# ----------------------------- Registry -----------------------------
@dataclass
class ScenarioRegistry:
    """
    Simple in-memory registry of scenarios.
    Provides:
      - presets (hardcoded common shocks)
      - add/get/list for custom scenarios
    """
    scenarios: Dict[str, PolicyShock]

    def add(self, key: str, shock: PolicyShock) -> None:
        self.scenarios[key] = shock

    def get(self, key: str) -> PolicyShock:
        if key not in self.scenarios:
            raise KeyError(f"Scenario '{key}' not found")
        return self.scenarios[key]

    def list(self) -> List[str]:
        return list(self.scenarios.keys())


# ----------------------------- Built-in Presets -----------------------------
def built_in_scenarios() -> Dict[str, PolicyShock]:
    """
    A handful of preset scenarios for quick use.
    """
    return {
        "fed_hike_75bp": PolicyShock(
            name="Fed +75bp shock",
            rates=RateShock(parallel_bp=75.0, steepen_bp=15.0, twist_pivot_yrs=5.0),
            equities=EquityShock(default_pct=-5.0),
            notes=["Parallel +75bp", "bear steepener", "equities -5%"],
        ),
        "rbi_cut_25bp": PolicyShock(
            name="RBI -25bp dovish cut",
            rates=RateShock(parallel_bp=-25.0, steepen_bp=-5.0, twist_pivot_yrs=4.0),
            equities=EquityShock(default_pct=+1.5),
            fx=FXShock(pct_by_pair={"USDINR": -0.7}),  # INR strengthens
            notes=["Parallel -25bp", "mild bull steepener", "equities +1.5%", "INR +0.7% vs USD"],
        ),
        "oil_spike": PolicyShock(
            name="Oil spike risk-off",
            rates=RateShock(steepen_bp=25.0, butterfly_bp=10.0, twist_pivot_yrs=7.0),
            equities=EquityShock(default_pct=-3.0),
            fx=FXShock(pct_by_pair={"USDINR": 1.2}),  # USDINR up = INR weakens
            notes=["Steepener with wing pressure", "equities -3%", "USD +1.2% vs INR"],
        ),
        "china_slowdown": PolicyShock(
            name="China slowdown shock",
            equities=EquityShock(pct_by_symbol={"HSI": -4.0, "AAPL": -2.5}),
            fx=FXShock(pct_by_pair={"USDCNY": 2.0}),  # CNY weakens
            notes=["HSI -4%", "AAPL -2.5%", "CNY -2% vs USD"],
        ),
        "vol_spike": PolicyShock(
            name="VIX spike",
            equities=EquityShock(default_pct=-4.0),
            vol=VolShock(vol_pts_by_symbol={"SPX": +10.0}),
            notes=["Equities -4%", "Vol +10 points"],
        ),
    }


# ----------------------------- Global Registry -----------------------------
_registry = ScenarioRegistry(scenarios=built_in_scenarios())


def get_registry() -> ScenarioRegistry:
    return _registry


# ----------------------------- Example CLI -----------------------------
if __name__ == "__main__":
    reg = get_registry()
    print("Available scenarios:")
    for k in reg.list():
        s = reg.get(k)
        print(f"- {k}: {s.name}")