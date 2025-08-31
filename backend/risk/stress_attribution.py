# backend/risk/stress_attribution.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Iterable

# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass
class Asset:
    """
    Asset meta used for stress P&L decomposition.
    - price: current mid (in local currency)
    - fx: FX rate to base currency (e.g., USD per local; 1.0 if base)
    - spread_bps: effective half-spread for taking (per side) used as cost proxy
    - adv_qty: average daily volume (for impact capacity)
    - impact_k_sqrt: square-root-law impact coefficient (bps * sqrt(q/ADV))
    - vega: P&L per +1.00 absolute vol (100 vol points). If your vega is per +0.01, scale accordingly.
    - gamma: P&L per (ΔS)^2 with S in price units (i.e., 0.5*gamma*(ΔS)^2 used)
    """
    symbol: str
    price: float
    fx: float = 1.0
    spread_bps: float = 6.0
    adv_qty: float = 1_000_000.0
    impact_k_sqrt: float = 20.0
    vega: float = 0.0
    gamma: float = 0.0

@dataclass
class Position:
    symbol: str
    qty: float  # signed (+ long, - short)

@dataclass
class Scenario:
    """
    Scenario inputs (all optional; missing keys default to 0).
    - price_rets: symbol -> return (fraction), applied on local price.
    - fx_rets:    ccy symbol or "*" -> return (fraction) for FX to base.
    - vol_shift:  absolute change in vol (e.g., +0.10 = +10 vol pts) per symbol or "*".
    - spread_widen_mult: multiply spread costs.
    - adv_drop_mult: haircut ADV (e.g., 0.5 = 50% liquidity).
    """
    name: str
    price_rets: Dict[str, float] = field(default_factory=dict)
    fx_rets: Dict[str, float] = field(default_factory=dict)
    vol_shift: Dict[str, float] = field(default_factory=dict)
    spread_widen_mult: float = 1.0
    adv_drop_mult: float = 1.0

@dataclass
class FactorModel:
    """
    Linear factor model for Euler/Shapley attribution.
    - factors: list of factor names
    - exposures: symbol -> {factor: loading}
    - factor_shocks: factor -> return shock (fraction)
    """
    factors: List[str]
    exposures: Dict[str, Dict[str, float]] = field(default_factory=dict)
    factor_shocks: Dict[str, float] = field(default_factory=dict)

@dataclass
class SymbolBreakdown:
    symbol: str
    base_notional: float
    d_price_cash: float
    d_fx_cash: float
    gamma_cash: float
    vega_cash: float
    liquidity_cost_cash: float
    total_cash: float

@dataclass
class FactorBreakdown:
    by_factor_cash: Dict[str, float]  # Euler factor contributions (cash)
    total_cash: float

@dataclass
class StressReport:
    scenario: str
    base_value: float
    total_pnl_cash: float
    by_symbol: Dict[str, SymbolBreakdown]
    by_factor: Optional[FactorBreakdown]
    waterfall: List[Tuple[str, float]]             # [(label, cash_delta), ...]
    flags: List[str]                                # e.g., losses above thresholds


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class StressAttributor:
    """
    P&L decomposition under a stress:
      - Symbol-level: Price, FX, Gamma, Vega, Liquidity/Costs → Total
      - Factor-level (optional): Euler contributions from factor shocks
      - Waterfall suitable for UI

    Design notes:
      * Price P&L holds FX constant; FX P&L uses shocked price for cross-term clarity.
      * Gamma uses 0.5 * gamma * (ΔS)^2; Vega uses absolute vol shift.
      * Liquidity cost combines spread widening and a coarse sqrt-law impact on the *minimum* size
        required to reduce risk (here we proxy with |qty|; swap with your TCA if available).
    """

    def __init__(self, base_ccy: str = "USD"):
        self.base_ccy = base_ccy
        self.assets: Dict[str, Asset] = {}
        self.pos: Dict[str, Position] = {}

    # ------------ registry ------------
    def upsert_asset(self, a: Asset) -> None:
        self.assets[a.symbol] = a

    def upsert_position(self, p: Position) -> None:
        if p.symbol not in self.assets:
            raise ValueError(f"Unknown asset for position: {p.symbol}")
        self.pos[p.symbol] = p

    def set_positions(self, positions: Iterable[Position]) -> None:
        self.pos.clear()
        for p in positions:
            self.upsert_position(p)

    # ------------ core calc ------------
    def run(self,
            scenario: Scenario,
            *,
            factor_model: Optional[FactorModel] = None,
            loss_warn_bps: float = 150.0,
            loss_crit_bps: float = 300.0) -> StressReport:

        # Base portfolio value (cash)
        base_val = 0.0
        for s, p in self.pos.items():
            a = self.assets[s]
            base_val += p.qty * a.price * a.fx

        by_symbol: Dict[str, SymbolBreakdown] = {}

        # Symbol effects
        total_price = total_fx = total_gamma = total_vega = total_liq = 0.0

        for s, p in self.pos.items():
            a = self.assets[s]
            r_p = scenario.price_rets.get(s, scenario.price_rets.get("*", 0.0))
            r_fx = scenario.fx_rets.get(s, scenario.fx_rets.get("*", 0.0))
            dv = scenario.vol_shift.get(s, scenario.vol_shift.get("*", 0.0))

            S0 = a.price
            FX0 = a.fx
            dS = S0 * r_p
            dFX = FX0 * r_fx

            # Decompose:
            d_price_cash = p.qty * dS * FX0                      # price move at base FX
            d_fx_cash    = p.qty * (S0 + dS) * dFX               # FX move on shocked price
            gamma_cash   = 0.5 * a.gamma * (dS ** 2) * (FX0)     # 2nd order price
            vega_cash    = a.vega * (dv or 0.0) * (FX0)          # vol shift to cash
            # Liquidity/impact (very coarse): take |qty| * price with widened spread + sqrt impact
            spread_bps = a.spread_bps * max(1.0, scenario.spread_widen_mult)
            adv = max(1e-9, a.adv_qty * max(1e-6, scenario.adv_drop_mult or 1.0))
            impact_bps = a.impact_k_sqrt * math.sqrt(min(abs(p.qty), adv) / adv)
            liq_cost = (spread_bps + impact_bps) / 1e4 * abs(p.qty) * S0 * FX0

            total = d_price_cash + d_fx_cash + gamma_cash + vega_cash - liq_cost

            by_symbol[s] = SymbolBreakdown(
                symbol=s,
                base_notional=p.qty * S0 * FX0,
                d_price_cash=d_price_cash,
                d_fx_cash=d_fx_cash,
                gamma_cash=gamma_cash,
                vega_cash=vega_cash,
                liquidity_cost_cash=liq_cost,
                total_cash=total
            )

            total_price += d_price_cash
            total_fx    += d_fx_cash
            total_gamma += gamma_cash
            total_vega  += vega_cash
            total_liq   += liq_cost

        total_pnl = total_price + total_fx + total_gamma + total_vega - total_liq

        # Factor (Euler) attribution if provided
        by_factor: Optional[FactorBreakdown] = None
        if factor_model and factor_model.factors:
            f_contrib: Dict[str, float] = {f: 0.0 for f in factor_model.factors}
            for s, p in self.pos.items():
                a = self.assets[s]
                fx = a.fx
                S0 = a.price
                expo = factor_model.exposures.get(s, {})
                for f in factor_model.factors:
                    beta = float(expo.get(f, 0.0))
                    shock = float(factor_model.factor_shocks.get(f, 0.0))
                    # Euler contribution: cash ≈ qty * price * beta * shock * fx
                    f_contrib[f] += p.qty * S0 * beta * shock * fx
            by_factor = FactorBreakdown(by_factor_cash=f_contrib, total_cash=sum(f_contrib.values()))

        # Waterfall (portfolio)
        waterfall: List[Tuple[str, float]] = [
            ("Price", total_price),
            ("FX", total_fx),
            ("Gamma", total_gamma),
            ("Vega", total_vega),
            ("Liquidity/Costs", -total_liq),
            ("Total", total_pnl),
        ]

        # Flags vs base value (in bps)
        flags: List[str] = []
        loss_bps = (-total_pnl / max(1e-9, base_val)) * 1e4 if total_pnl < 0 else 0.0
        if loss_bps >= loss_warn_bps:
            flags.append(f"Stress loss warning: {loss_bps:.0f} bps")
        if loss_bps >= loss_crit_bps:
            flags.append(f"Stress loss CRITICAL: {loss_bps:.0f} bps")

        return StressReport(
            scenario=scenario.name,
            base_value=base_val,
            total_pnl_cash=total_pnl,
            by_symbol=by_symbol,
            by_factor=by_factor,
            waterfall=waterfall,
            flags=flags
        )

    # ------------ optional Shapley-lite for factors ------------
    @staticmethod
    def shapley_factors(factors: List[str],
                        factor_shocks: Dict[str, float],
                        factor_to_cash: Dict[str, float],
                        *,
                        n_perms: int = 64,
                        seed: int = 7) -> Dict[str, float]:
        """
        Shapley-lite: average marginal cash contribution over random orderings.
        Input factor_to_cash should be the *Euler* cash mapping or any linearized map
        (we treat interactions as negligible in this approximation).
        """
        if not factors:
            return {}
        rng = random.Random(seed)
        attributions = {f: 0.0 for f in factors}
        # With linear map the Shapley equals Euler; we still compute perms to mimic generality.
        for _ in range(max(1, n_perms)):
            order = factors[:]
            rng.shuffle(order)
            accum = 0.0
            for f in order:
                contrib = factor_to_cash.get(f, 0.0)
                attributions[f] += contrib
                accum += contrib
        # Normalize by number of permutations
        m = float(max(1, n_perms))
        return {f: v / m for f, v in attributions.items()}


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Universe
    AAPL = Asset(symbol="AAPL", price=190.0, fx=1.0, spread_bps=4.0, adv_qty=80_000_000, impact_k_sqrt=12.0, gamma=0.0, vega=0.0)
    MSFT = Asset(symbol="MSFT", price=420.0, fx=1.0, spread_bps=3.0, adv_qty=30_000_000, impact_k_sqrt=12.0, gamma=0.0, vega=0.0)
    SPY  = Asset(symbol="SPY",  price=520.0, fx=1.0, spread_bps=1.0, adv_qty=80_000_000, impact_k_sqrt=8.0,  gamma=0.0, vega=0.0)

    eng = StressAttributor()
    for a in (AAPL, MSFT, SPY):
        eng.upsert_asset(a)
    eng.set_positions([
        Position("AAPL", 1_000),
        Position("MSFT", 800),
        Position("SPY", -500),
    ])

    # Scenario: equities -4%, USD flat; liquidity worse
    scen = Scenario(
        name="Equity -4% / Liquidity Tight",
        price_rets={"*": -0.04},
        fx_rets={"*": 0.0},
        vol_shift={},                # no option greeks in this demo
        spread_widen_mult=1.5,
        adv_drop_mult=0.6
    )

    # Simple 2-factor model: {MKT, TECH} shocks for Euler attribution
    fmodel = FactorModel(
        factors=["MKT", "TECH"],
        exposures={
            "AAPL": {"MKT": 1.2, "TECH": 0.8},
            "MSFT": {"MKT": 1.1, "TECH": 0.7},
            "SPY":  {"MKT": 1.0, "TECH": 0.0},
        },
        factor_shocks={"MKT": -0.04, "TECH": -0.02}
    )

    rpt = eng.run(scenario=scen, factor_model=fmodel)
    print("--- Stress Report ---")
    print("Scenario:", rpt.scenario)
    print("Base value:", round(rpt.base_value, 2))
    print("Total PnL (cash):", round(rpt.total_pnl_cash, 2))
    print("Waterfall:", [(k, round(v, 2)) for k, v in rpt.waterfall])
    if rpt.by_factor:
        print("Factors:", {k: round(v, 2) for k, v in rpt.by_factor.by_factor_cash.items()})
    print("Flags:", rpt.flags)