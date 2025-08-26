# agents/commodities.py
"""
CommoditiesAgent
----------------
Rule-driven commodities agent that converts climate/seasonal signals into trades.

Inputs (from MarketContext.signals)
-----------------------------------
Expected (but optional) numeric keys; if missing, that rule is skipped:
- "hurricane_prob_gom"     : [0..1]   — hurricane probability (Gulf of Mexico); bullish WTI when high
- "hdd_z_us"               : z-score  — Heating Degree Days anomaly; bullish NatGas when > +th
- "precip_anom_midwest"    : % or std — Midwest precipitation anomaly; bullish Corn when deeply negative (drought)
- "precip_anom_brazil_cof" : % or std — Brazil coffee-belt precipitation anomaly; bullish Coffee when negative (dry)

Symbols (defaults; override via config):
- WTI Crude futures (CME): "CLZ5"
- Henry Hub NatGas (CME):  "NGZ5"
- Corn (CBOT):             "ZCZ5"
- Coffee Arabica (ICE):    "KCZ5"

Scoring
-------
Each rule computes a rough "edge" (bps) proportional to the signal strength,
then converts to a normalized score via base.soft_score_from_edge() with a short horizon.

Usage
-----
from agents.commodities import CommoditiesAgent, CommoditiesConfig
from agents.base import MarketContext

ctx = MarketContext.now(
    prices={"CLZ5": 80.0, "NGZ5": 3.2, "ZCZ5": 480.0, "KCZ5": 210.0},
    signals={"hurricane_prob_gom": 0.35, "hdd_z_us": 1.2, "precip_anom_midwest": -0.6}
)
agent = CommoditiesAgent()
prop = agent.propose(ctx)
risk = agent.risk(prop, ctx)
print(agent.explain(prop, risk))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import ( # type: ignore
    AgentBase, MarketContext, Proposal, OrderPlan, Constraints,
    soft_score_from_edge, clamp
)


# ------------------- configuration -------------------

@dataclass
class CommoditiesConfig:
    # Symbol map (override if your codes differ)
    sym_wti: str = "CLZ5"
    sym_ng: str = "NGZ5"
    sym_corn: str = "ZCZ5"
    sym_coffee: str = "KCZ5"

    # Base order sizes (contract units)
    qty_wti: float = 1.0
    qty_ng: float = 1.0
    qty_corn: float = 2.0
    qty_coffee: float = 1.0

    # Thresholds / scaling
    hurricane_buy_threshold: float = 0.25      # P(hurricane) to start buying crude
    hdd_z_buy_threshold: float = 0.8          # HDD z-score for NatGas long
    precip_midwest_sell_threshold: float = 0.4  # if > +th, good rains -> bearish corn
    precip_midwest_buy_threshold: float = -0.4  # if < -th, drought -> bullish corn
    precip_brazil_cof_buy_threshold: float = -0.3  # dry Brazil -> bullish coffee

    # Edge scaling (bps per unit of signal)
    edge_bps_per_hurricane: float = 120.0      # crude inventory/supply risk
    edge_bps_per_hdd_z: float = 80.0           # natgas weather sensitivity
    edge_bps_per_precip_pct: float = 90.0      # grains/coffee precip sensitivity

    # Horizon (seconds) per motif
    horizon_crude_sec: float = 3 * 24 * 3600   # multi‑day
    horizon_ng_sec: float = 2 * 24 * 3600
    horizon_grains_sec: float = 7 * 24 * 3600
    horizon_coffee_sec: float = 10 * 24 * 3600

    # Confidence caps
    conf_crude: float = 0.7
    conf_ng: float = 0.65
    conf_grains: float = 0.6
    conf_coffee: float = 0.6


# ------------------- agent -------------------

class CommoditiesAgent(AgentBase):
    name = "commodities"

    def __init__(self, cfg: Optional[CommoditiesConfig] = None):
        self.cfg = cfg or CommoditiesConfig()

    # -------- propose --------

    def propose(self, context: MarketContext) -> Proposal:
        s = context.signals or {}
        prices = context.prices or {}

        legs: List[OrderPlan] = []
        tags: List[str] = []
        analyses: List[str] = []
        score_accum = 0.0
        score_w = 0.0
        conf_accum = 0.0

        # --- 1) Crude oil: hurricane risk in Gulf of Mexico ---
        p_hurr = _f(s.get("hurricane_prob_gom"), 0.0)
        if p_hurr >= self.cfg.hurricane_buy_threshold and self.cfg.sym_wti in prices:
            edge_bps = (p_hurr - self.cfg.hurricane_buy_threshold) * self.cfg.edge_bps_per_hurricane
            score = soft_score_from_edge(edge_bps, self.cfg.horizon_crude_sec, cap=1.0)
            qty = self.cfg.qty_wti
            legs.append(OrderPlan(symbol=self.cfg.sym_wti, side="BUY", qty=qty, type="MARKET", meta={"motif": "hurricane"}))
            analyses.append(f"WTI: P(hurricane)={p_hurr:.2f} → edge≈{edge_bps:.0f}bps, score={score:.2f}")
            tags += ["energy", "hurricane", "crude"]
            score_accum += score * 1.0
            score_w += 1.0
            conf_accum += self.cfg.conf_crude

        # --- 2) NatGas: HDD anomaly (cold surprise) ---
        hdd_z = _f(s.get("hdd_z_us"), 0.0)
        if hdd_z >= self.cfg.hdd_z_buy_threshold and self.cfg.sym_ng in prices:
            edge_bps = (hdd_z - self.cfg.hdd_z_buy_threshold) * self.cfg.edge_bps_per_hdd_z
            score = soft_score_from_edge(edge_bps, self.cfg.horizon_ng_sec, cap=1.0)
            legs.append(OrderPlan(symbol=self.cfg.sym_ng, side="BUY", qty=self.cfg.qty_ng, type="MARKET", meta={"motif": "HDD"}))
            analyses.append(f"NG: HDD z={hdd_z:.2f} → edge≈{edge_bps:.0f}bps, score={score:.2f}")
            tags += ["energy", "natgas", "weather"]
            score_accum += score * 1.0
            score_w += 1.0
            conf_accum += self.cfg.conf_ng

        # --- 3) Corn: Midwest precipitation anomaly ---
        precip_mw = _f(s.get("precip_anom_midwest"), 0.0)
        if self.cfg.sym_corn in prices:
            if precip_mw <= self.cfg.precip_midwest_buy_threshold:
                # drought -> bullish corn
                edge_bps = (abs(precip_mw - self.cfg.precip_midwest_buy_threshold)) * self.cfg.edge_bps_per_precip_pct
                score = soft_score_from_edge(edge_bps, self.cfg.horizon_grains_sec, cap=1.0)
                legs.append(OrderPlan(symbol=self.cfg.sym_corn, side="BUY", qty=self.cfg.qty_corn, type="MARKET", meta={"motif": "drought"}))
                analyses.append(f"Corn: precip_mw={precip_mw:.2f} (dry) → edge≈{edge_bps:.0f}bps, score={score:.2f}")
                tags += ["grains", "corn", "weather"]
                score_accum += score; score_w += 1.0; conf_accum += self.cfg.conf_grains
            elif precip_mw >= self.cfg.precip_midwest_sell_threshold:
                # abundant rain -> bearish corn
                edge_bps = (precip_mw - self.cfg.precip_midwest_sell_threshold) * self.cfg.edge_bps_per_precip_pct
                score = soft_score_from_edge(edge_bps, self.cfg.horizon_grains_sec, cap=1.0)
                legs.append(OrderPlan(symbol=self.cfg.sym_corn, side="SELL", qty=self.cfg.qty_corn, type="MARKET", meta={"motif": "good_rains"}))
                analyses.append(f"Corn: precip_mw={precip_mw:.2f} (wet) → edge≈{edge_bps:.0f}bps, score={score:.2f}")
                tags += ["grains", "corn", "weather"]
                score_accum += score; score_w += 1.0; conf_accum += self.cfg.conf_grains

        # --- 4) Coffee: Brazil dryness anomaly ---
        precip_brz = _f(s.get("precip_anom_brazil_cof"), 0.0)
        if precip_brz <= self.cfg.precip_brazil_cof_buy_threshold and self.cfg.sym_coffee in prices:
            edge_bps = (abs(precip_brz - self.cfg.precip_brazil_cof_buy_threshold)) * self.cfg.edge_bps_per_precip_pct
            score = soft_score_from_edge(edge_bps, self.cfg.horizon_coffee_sec, cap=1.0)
            legs.append(OrderPlan(symbol=self.cfg.sym_coffee, side="BUY", qty=self.cfg.qty_coffee, type="MARKET", meta={"motif": "brazil_dry"}))
            analyses.append(f"Coffee: precip_brz={precip_brz:.2f} (dry) → edge≈{edge_bps:.0f}bps, score={score:.2f}")
            tags += ["softs", "coffee", "weather"]
            score_accum += score; score_w += 1.0; conf_accum += self.cfg.conf_coffee

        # If nothing triggered, return a neutral, empty proposal
        if not legs:
            return Proposal(orders=[], thesis="No commodity climate edges triggered.", score=0.0, horizon_sec=300.0, confidence=0.3, tags=["idle"])

        avg_score = clamp(score_accum / max(1.0, score_w), -1.0, 1.0)
        avg_conf = clamp(conf_accum / max(1.0, score_w), 0.0, 1.0)
        thesis = " | ".join(analyses)

        return Proposal(
            orders=legs,
            thesis=thesis,
            score=avg_score,
            horizon_sec= max(self.cfg.horizon_ng_sec, self.cfg.horizon_crude_sec, self.cfg.horizon_grains_sec, self.cfg.horizon_coffee_sec),
            confidence=avg_conf,
            tags=list(dict.fromkeys(tags)),  # de-dup while preserving order
            diagnostics={
                "signals_used": {
                    "hurricane_prob_gom": p_hurr,
                    "hdd_z_us": hdd_z,
                    "precip_anom_midwest": precip_mw,
                    "precip_anom_brazil_cof": precip_brz,
                }
            },
        )

    # -------- risk --------

    def risk(self, proposal: Proposal, context: MarketContext):
        # Reuse base constraint checks + notional summary
        return self.base_risk(proposal, context)

    # -------- explain --------

    def explain(self, proposal: Proposal, risk=None) -> str:
        if not proposal.orders:
            return f"[{self.name}] {proposal.thesis}"
        legs_txt = ", ".join([f"{o.side} {o.qty} {o.symbol}" for o in proposal.orders])
        risk_txt = ""
        if risk:
            ok = "PASS" if risk.ok else "FAIL"
            risk_txt = f" | risk={ok} gross=${risk.gross_notional_usd:,.0f} net=${risk.exposure_usd:,.0f}"
        return f"[{self.name}] {legs_txt} | score={proposal.score:.2f}, conf={proposal.confidence:.2f} | {proposal.thesis}{risk_txt}"


# ------------------- helpers -------------------

def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)