# research/risk/scenarios.py
"""
Scenario engine for risk management.

Supports:
  - Percentage shocks (e.g., -10% market crash)
  - Absolute shocks (e.g., +50 USD move in crude)
  - Multi-asset baskets
  - Yield curve parallel shifts and twists
  - FX devaluation
  - Custom user-defined shocks

Input:
    positions: dict[str, float]  # symbol -> position (shares, contracts)
    ref_prices: dict[str, float] # symbol -> reference price
    scenarios: list[ScenarioSpec]

Output:
    dict of scenario_name -> result {
        "pnl": float,
        "contrib": {sym: pnl_sym, ...},
        "shocked_prices": {sym: new_px, ...}
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional


# ------------------------------------------------------------------------
# Scenario specifications
# ------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    shock: Callable[[Dict[str, float]], Dict[str, float]]
    note: Optional[str] = None


# ------------------------------------------------------------------------
# Shock constructors
# ------------------------------------------------------------------------

def pct_shock(symbols: List[str], pct: float) -> ScenarioSpec:
    """Shock given symbols by percentage (e.g., -0.1 for -10%)."""
    def shock(prices: Dict[str, float]) -> Dict[str, float]:
        return {s: prices[s] * (1.0 + pct) for s in symbols if s in prices}
    return ScenarioSpec(name=f"pct_{pct:+.0%}", shock=shock)


def abs_shock(symbols: List[str], amt: float) -> ScenarioSpec:
    """Shock given symbols by absolute amount."""
    def shock(prices: Dict[str, float]) -> Dict[str, float]:
        return {s: prices[s] + amt for s in symbols if s in prices}
    return ScenarioSpec(name=f"abs_{amt:+}", shock=shock)


def fx_deval(ccy_pair: str, pct: float) -> ScenarioSpec:
    """Devalue FX pair (e.g., 'USD/INR', pct=+0.1 → INR weakens 10%)."""
    def shock(prices: Dict[str, float]) -> Dict[str, float]:
        if ccy_pair not in prices:
            return {}
        return {ccy_pair: prices[ccy_pair] * (1.0 + pct)}
    return ScenarioSpec(name=f"fx_{ccy_pair}_{pct:+.0%}", shock=shock)


def yield_curve_parallel(symbols: List[str], bp: float) -> ScenarioSpec:
    """
    Parallel shift for yields (in basis points).
    Assumes input prices are yields (not bond prices).
    """
    def shock(prices: Dict[str, float]) -> Dict[str, float]:
        return {s: prices[s] + bp * 1e-4 for s in symbols if s in prices}
    return ScenarioSpec(name=f"yc_parallel_{bp:+.0f}bp", shock=shock)


# ------------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------------

def apply_scenario(
    spec: ScenarioSpec,
    positions: Dict[str, float],
    ref_prices: Dict[str, float],
) -> Dict[str, object]:
    """
    Apply scenario shock to portfolio.
    Returns dict with pnl, contrib, shocked_prices.
    """
    shocked = dict(ref_prices)  # copy
    shocked.update(spec.shock(ref_prices))

    contrib: Dict[str, float] = {}
    pnl = 0.0
    for sym, qty in positions.items():
        if sym not in ref_prices:
            continue
        px0 = ref_prices[sym]
        px1 = shocked.get(sym, px0)
        pnl_sym = qty * (px1 - px0)
        contrib[sym] = pnl_sym
        pnl += pnl_sym

    return {
        "name": spec.name,
        "note": spec.note,
        "pnl": pnl,
        "contrib": contrib,
        "shocked_prices": {s: shocked[s] for s in ref_prices},
    }


def run_scenarios(
    specs: List[ScenarioSpec],
    positions: Dict[str, float],
    ref_prices: Dict[str, float],
) -> Dict[str, Dict[str, object]]:
    """
    Run multiple scenarios and return dict of results keyed by scenario name.
    """
    results: Dict[str, Dict[str, object]] = {}
    for s in specs:
        results[s.name] = apply_scenario(s, positions, ref_prices)
    return results


# ------------------------------------------------------------------------
# Example
# ------------------------------------------------------------------------

if __name__ == "__main__":
    positions = {"AAPL": 100, "MSFT": -50, "USD/INR": 1_000_000}
    ref_prices = {"AAPL": 200.0, "MSFT": 120.0, "USD/INR": 82.0}

    specs = [
        pct_shock(["AAPL", "MSFT"], -0.1),
        abs_shock(["AAPL"], +5.0),
        fx_deval("USD/INR", +0.05),
        yield_curve_parallel(["10Y", "30Y"], +25),
    ]

    results = run_scenarios(specs, positions, ref_prices) # type: ignore
    for name, r in results.items():
        print(name, "PNL:", r["pnl"], "Top contrib:", max(r["contrib"].items(), key=lambda x: abs(x[1]))) # type: ignore