# backend/ai/agents/concrete/explainer_agent.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# Framework shims (safe fallbacks)
# ------------------------------------------------------------
try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name: str = "explainer_agent"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# Optionally use risk metrics, TCA, signals if available
try:
    from ..skills.trading.tca import estimate_cost_bps  # type: ignore
except Exception:
    def estimate_cost_bps(symbol: str, side: str, qty: float, px: float, venue: Optional[str] = None) -> float:
        return 2.0 + min(20.0, qty / 1_000_000.0 * 10.0)

try:
    from ..skills.risk.var_engine import VaREngine  # type: ignore
except Exception:
    class VaREngine:
        @staticmethod
        def gaussian(returns: List[float], alpha: float = 0.99, horizon_days: int = 1):
            return type("VarEstimate", (), {
                "var": -0.02, "mean": 0.0005, "stdev": 0.01,
                "alpha": alpha, "horizon_days": horizon_days
            })

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class TradeEvent:
    symbol: str
    side: str           # "buy"/"sell"
    qty: float
    px: float
    strategy: str
    ts_ms: int = int(time.time() * 1000)
    venue: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class Explanation:
    headline: str
    reasoning: str
    risks: List[str]
    alternatives: List[str]
    metrics: Dict[str, Any]

@dataclass
class ExplainRequest:
    trade: TradeEvent
    context: Dict[str, Any]

@dataclass
class ExplainResponse:
    trade: TradeEvent
    explanation: Explanation
    generated_at: int

# ------------------------------------------------------------
# Explainer Agent
# ------------------------------------------------------------
class ExplainerAgent(BaseAgent): # type: ignore
    """
    Given a TradeEvent + context (signals, risk metrics),
    produce a structured natural-language rationale:
      - headline summary
      - reasoning
      - risks considered
      - alternatives
      - quantitative metrics
    """

    name = "explainer_agent"

    def plan(self, req: ExplainRequest | Dict[str, Any]) -> ExplainRequest:
        if isinstance(req, ExplainRequest):
            return req
        t = req.get("trade", {})
        ctx = req.get("context", {})
        return ExplainRequest(
            trade=TradeEvent(
                symbol=t.get("symbol", "UNK"),
                side=t.get("side", "buy"),
                qty=float(t.get("qty", 0)),
                px=float(t.get("px", 0)),
                strategy=t.get("strategy", "unknown"),
                ts_ms=int(t.get("ts_ms", time.time() * 1000)),
                venue=t.get("venue"),
                notes=t.get("notes"),
            ),
            context=ctx
        )

    def act(self, request: ExplainRequest | Dict[str, Any]) -> ExplainResponse:
        req = self.plan(request)
        t = req.trade

        # ---- Quant metrics ----
        cost_bps = estimate_cost_bps(t.symbol, t.side, t.qty, t.px, t.venue)
        var_est = VaREngine.gaussian([0.001, -0.002, 0.0005])  # toy sample

        metrics = {
            "cost_bps": cost_bps,
            "VaR_99": var_est.var,#type:ignore
            "expected_return": var_est.mean, # type: ignore
            "stdev": var_est.stdev # type: ignore
        }

        # ---- Reasoning template ----
        headline = f"{t.strategy.upper()} executed {t.side.upper()} {t.qty} {t.symbol} @ {t.px:.2f}"
        reasoning = (
            f"Trade initiated under strategy '{t.strategy}' at {t.px:.2f}. "
            f"Estimated execution cost {cost_bps:.2f} bps. "
            f"Context suggests {'bullish' if t.side=='buy' else 'bearish'} tilt."
        )

        risks = [
            f"1-day 99% VaR ~ {var_est.var:.2%}", # type: ignore
            f"Volatility ~ {var_est.stdev:.2%}", # type: ignore
            "Execution slippage risk if liquidity dries up",
        ]

        alternatives = [
            "Stagger order via TWAP to reduce market impact",
            "Use options overlay for convex exposure",
            "Route via dark pool to minimize footprint",
        ]

        explanation = Explanation(
            headline=headline,
            reasoning=reasoning,
            risks=risks,
            alternatives=alternatives,
            metrics=metrics
        )

        return ExplainResponse(
            trade=t,
            explanation=explanation,
            generated_at=int(time.time() * 1000)
        )

    def explain(self) -> str:
        return (
            "ExplainerAgent converts a TradeEvent into a human-friendly rationale: "
            "what the trade did, why it happened, what risks it carries, and what "
            "alternatives were available. Designed for audit trails, compliance, and UI."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "ts": int(time.time())}


# ------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = ExplainerAgent()
    req = ExplainRequest(
        trade=TradeEvent(symbol="AAPL", side="buy", qty=10_000, px=187.32, strategy="mean_reversion"),
        context={"book": "demo"}
    )
    resp = agent.act(req)
    print(resp.explanation.headline)
    print(resp.explanation.reasoning)
    print("Risks:", resp.explanation.risks)
    print("Alternatives:", resp.explanation.alternatives)
    print("Metrics:", resp.explanation.metrics)