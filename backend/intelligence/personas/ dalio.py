# agents/dalio.py
"""
DalioAgent
----------
Macro "Pure Alpha"-style agent.
Blends high-level macro signals (growth, inflation, liquidity, risk regime)
to tilt across broad asset buckets:
  - Equities (SPY)
  - Bonds (TLT as proxy for US Treasuries)
  - Commodities (DBC or GSG)
  - Gold (GLD)
  - FX (USD index proxy: DXY)

Signals expected in MarketContext.signals:
- growth_nowcast     : GDP/Growth nowcast (z-score vs trend)
- infl_nowcast       : Inflation nowcast (z-score vs trend)
- policy_rate        : Policy rate level (e.g., Fed Funds, in %)
- liquidity_z        : Liquidity index (z-score, higher = looser)
- risk_z             : Risk regime (e.g., VIX z-score)

Rules (simple heuristics):
- Strong growth + loose liquidity → overweight equities
- High inflation → overweight commodities & gold, underweight bonds
- Risk-off (risk_z > +1.0) → tilt to bonds/gold, underweight equities/commodities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agents.base import ( # type: ignore
    AgentBase, MarketContext, Proposal, OrderPlan, RiskReport,
    soft_score_from_edge, clamp
)


@dataclass
class DalioConfig:
    symbols: Dict[str, str] = field(default_factory=lambda: {
        "equities": "SPY",
        "bonds": "TLT",
        "commods": "DBC",
        "gold": "GLD",
        "usd": "DXY",
    })
    base_qty: Dict[str, float] = field(default_factory=lambda: {
        "equities": 100,
        "bonds": 100,
        "commods": 50,
        "gold": 50,
        "usd": 1000,   # DXY synthetic exposure
    })
    min_abs_score: float = 0.15
    horizon_sec: float = 10 * 24 * 3600   # ~10 days horizon
    max_score: float = 1.0


class DalioAgent(AgentBase):
    name = "dalio"

    def __init__(self, cfg: Optional[DalioConfig] = None):
        self.cfg = cfg or DalioConfig()

    def propose(self, context: MarketContext) -> Proposal:
        s = context.signals or {}
        sym = self.cfg.symbols

        growth = float(s.get("growth_nowcast", 0.0))
        infl = float(s.get("infl_nowcast", 0.0))
        policy = float(s.get("policy_rate", 0.0))
        liq = float(s.get("liquidity_z", 0.0))
        riskz = float(s.get("risk_z", 0.0))

        parts: List[str] = []
        legs: List[OrderPlan] = []

        # Equities: growth + liquidity, penalized in risk-off
        eq_score = 0.0
        eq_score += 0.6 * growth + 0.3 * liq
        eq_score -= 0.5 * max(0.0, riskz)
        eq_score = clamp(eq_score, -self.cfg.max_score, self.cfg.max_score)
        if abs(eq_score) >= self.cfg.min_abs_score:
            side = "BUY" if eq_score > 0 else "SELL"
            qty = abs(eq_score) * self.cfg.base_qty["equities"]
            legs.append(OrderPlan(symbol=sym["equities"], side=side, qty=qty, meta={"score": eq_score}))
            parts.append(f"Equities {side} (growth={growth:.2f}, liq={liq:.2f}, risk={riskz:.2f})")

        # Bonds: inverse inflation, benefit in risk-off
        bd_score = 0.0
        bd_score -= 0.7 * infl
        bd_score += 0.5 * riskz
        bd_score = clamp(bd_score, -self.cfg.max_score, self.cfg.max_score)
        if abs(bd_score) >= self.cfg.min_abs_score:
            side = "BUY" if bd_score > 0 else "SELL"
            qty = abs(bd_score) * self.cfg.base_qty["bonds"]
            legs.append(OrderPlan(symbol=sym["bonds"], side=side, qty=qty, meta={"score": bd_score}))
            parts.append(f"Bonds {side} (infl={infl:.2f}, risk={riskz:.2f})")

        # Commodities: inflation hedge, but risk-off penalized
        cm_score = 0.0
        cm_score += 0.8 * infl
        cm_score -= 0.4 * max(0.0, riskz)
        cm_score = clamp(cm_score, -self.cfg.max_score, self.cfg.max_score)
        if abs(cm_score) >= self.cfg.min_abs_score:
            side = "BUY" if cm_score > 0 else "SELL"
            qty = abs(cm_score) * self.cfg.base_qty["commods"]
            legs.append(OrderPlan(symbol=sym["commods"], side=side, qty=qty, meta={"score": cm_score}))
            parts.append(f"Commodities {side} (infl={infl:.2f}, risk={riskz:.2f})")

        # Gold: inflation + risk hedge
        gd_score = 0.0
        gd_score += 0.5 * infl + 0.4 * riskz
        gd_score = clamp(gd_score, -self.cfg.max_score, self.cfg.max_score)
        if abs(gd_score) >= self.cfg.min_abs_score:
            side = "BUY" if gd_score > 0 else "SELL"
            qty = abs(gd_score) * self.cfg.base_qty["gold"]
            legs.append(OrderPlan(symbol=sym["gold"], side=side, qty=qty, meta={"score": gd_score}))
            parts.append(f"Gold {side} (infl={infl:.2f}, risk={riskz:.2f})")

        # USD (DXY): stronger policy rate + risk-off → long USD
        usd_score = 0.0
        usd_score += 0.5 * policy + 0.5 * riskz
        usd_score = clamp(usd_score, -self.cfg.max_score, self.cfg.max_score)
        if abs(usd_score) >= self.cfg.min_abs_score:
            side = "BUY" if usd_score > 0 else "SELL"
            qty = abs(usd_score) * self.cfg.base_qty["usd"]
            legs.append(OrderPlan(symbol=sym["usd"], side=side, qty=qty, meta={"score": usd_score}))
            parts.append(f"USD {side} (policy={policy:.2f}, risk={riskz:.2f})")

        if not legs:
            return Proposal(orders=[], thesis="No macro tilts triggered", score=0.0,
                            horizon_sec=self.cfg.horizon_sec, confidence=0.4, tags=["macro"])

        avg_score = sum(o.meta.get("score", 0.0) for o in legs) / len(legs)
        thesis = " | ".join(parts)

        return Proposal(
            orders=legs,
            thesis=thesis,
            score=avg_score,
            horizon_sec=self.cfg.horizon_sec,
            confidence=0.6,
            tags=["macro", "dalio"],
            diagnostics={"growth": growth, "infl": infl, "policy": policy, "liq": liq, "riskz": riskz}
        )

    def risk(self, proposal: Proposal, context: MarketContext) -> RiskReport:
        return self.base_risk(proposal, context)

    def explain(self, proposal: Proposal, risk: RiskReport | None = None) -> str:
        if not proposal.orders:
            return f"[{self.name}] {proposal.thesis}"
        legs_txt = ", ".join([f"{o.side} {o.qty:g} {o.symbol}" for o in proposal.orders])
        risk_txt = ""
        if risk:
            risk_txt = f" | gross=${risk.gross_notional_usd:,.0f} net=${risk.exposure_usd:,.0f} ({'OK' if risk.ok else 'FAIL'})"
        return f"[{self.name}] {legs_txt} | score={proposal.score:.2f}, conf={proposal.confidence:.2f} | {proposal.thesis}{risk_txt}"