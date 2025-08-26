# agents/base.py
"""
Agent base interfaces and light models for the arbitrage "swarm".

Each concrete agent (fx.py, equities.py, crypto.py, commodities.py) should:
  - implement propose(context) -> Proposal
  - implement risk(proposal, context) -> RiskReport
  - implement explain(proposal, risk) -> str

Design notes
------------
- No external deps. Pure dataclasses + typing.
- Keep outputs small & serializable: primitives and dicts only.
- Risk is split from propose so a separate risk service could reuse it.

Typical flow
------------
ctx = MarketContext.now(prices={"BTCUSDT": 65000}, balances={"CASH": 1_000_000})
agent = CryptoAgent()  # subclass of AgentBase
proposal = agent.propose(ctx)
risk = agent.risk(proposal, ctx)
print(agent.explain(proposal, risk))
"""

from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal


# ---------------------- shared primitives ----------------------

Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]


@dataclass
class OrderPlan:
    """What the agent wants to trade."""
    symbol: str
    side: Side
    qty: float
    type: OrderType = "MARKET"
    limit_price: Optional[float] = None
    venue: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """
    Agent's action proposal.
    - score: relative desirability [-1, 1] (agent-defined)
    - horizon_sec: expected alpha horizon (s)
    - confidence: model confidence [0,1]
    """
    orders: List[OrderPlan]
    thesis: str
    score: float = 0.0
    horizon_sec: float = 300.0
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orders": [asdict(o) for o in self.orders],
            "thesis": self.thesis,
            "score": float(self.score),
            "horizon_sec": float(self.horizon_sec),
            "confidence": float(self.confidence),
            "tags": list(self.tags or []),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass
class RiskCheck:
    """A single pre‑trade check outcome."""
    name: str
    ok: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskReport:
    """
    Summarized risk assessment for a proposal.
    - ok: overall pass/fail
    - exposure_usd: signed net notional in USD terms (if available)
    - max_drawdown_est: rough point estimate (optional)
    """
    ok: bool
    exposure_usd: float = 0.0
    gross_notional_usd: float = 0.0
    max_drawdown_est: Optional[float] = None
    checks: List[RiskCheck] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "exposure_usd": float(self.exposure_usd),
            "gross_notional_usd": float(self.gross_notional_usd),
            "max_drawdown_est": self.max_drawdown_est if self.max_drawdown_est is not None else None,
            "checks": [asdict(c) for c in self.checks],
            "notes": self.notes,
        }


@dataclass
class Constraints:
    """
    Soft/hard limits an agent should obey.
    Any None/0 values are treated as 'no limit'.
    """
    max_notional_usd: Optional[float] = 0.0
    max_leg_notional_usd: Optional[float] = 0.0
    max_positions: Optional[int] = 0
    forbid_symbols: Sequence[str] = field(default_factory=tuple)
    max_leverage: Optional[float] = 0.0
    require_venues: Sequence[str] = field(default_factory=tuple)
    min_confidence: float = 0.0
    max_horizon_sec: Optional[float] = None


@dataclass
class MarketContext:
    """
    Minimal context snapshot fed to agents.
    - prices: symbol -> mid price in venue/base currency (or USD if unified)
    - fx_usd_per_base: base ccy -> USD rate (default 1.0)
    - balances: e.g., {"CASH": 1_000_000, "BTC": 2.0}
    - signals: arbitrary numeric signals (altdata/sentiment etc.)
    """
    ts: float
    prices: Dict[str, float] = field(default_factory=dict)
    fx_usd_per_base: Dict[str, float] = field(default_factory=dict)
    balances: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, float] = field(default_factory=dict)
    constraints: Constraints = field(default_factory=Constraints)

    @classmethod
    def now(cls, **kw) -> "MarketContext":
        return cls(ts=time.time(), **kw)


# ---------------------- convenience helpers ----------------------

def notional_usd(symbol: str, qty: float, prices: Dict[str, float], fx_usd_per_base: Dict[str, float] | None = None) -> float:
    """
    Compute |qty| * price * fx; assumes price quoted in base that maps in fx_usd_per_base.
    If no FX known, assume 1.0 (USD).
    """
    px = float(prices.get(symbol, 0.0))
    fx = 1.0
    if fx_usd_per_base:
        # Try to infer base from suffix like "XXXUSDT" or ".NS" etc.; fallback to USD=1
        if symbol.endswith("USDT") or symbol.endswith("USD"):
            fx = 1.0
        else:
            fx = float(next(iter(fx_usd_per_base.values()), 1.0))
    return abs(float(qty)) * px * fx


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def soft_score_from_edge(edge_bps: float, horizon_sec: float, cap: float = 1.0) -> float:
    """
    Turn an expected edge (bps) and horizon into a normalized score [-1,1].
    Positive edge -> positive score; longer horizons slightly discounted.
    """
    if horizon_sec <= 0:
        horizon_sec = 1.0
    raw = edge_bps / 50.0  # 50 bps ~ score 1 before horizon discount
    decay = 1.0 / (1.0 + math.log1p(max(0.0, horizon_sec)) / 3.0)
    return clamp(raw * decay, -cap, cap)


def check_constraints(proposal: Proposal, ctx: MarketContext) -> List[RiskCheck]:
    c = ctx.constraints or Constraints()
    checks: List[RiskCheck] = []

    # Forbidden symbols
    bad = [o.symbol for o in proposal.orders if (o.symbol in (c.forbid_symbols or []))]
    checks.append(RiskCheck("forbid_symbols", ok=(len(bad) == 0), details={"blocked": bad} if bad else {}))

    # Required venues
    if c.require_venues:
        missing = [o.venue for o in proposal.orders if (o.venue and o.venue not in c.require_venues)]
        checks.append(RiskCheck("require_venues", ok=(len(missing) == 0), details={"missing": missing} if missing else {}))

    # Confidence / horizon
    checks.append(RiskCheck("min_confidence", ok=(proposal.confidence >= (c.min_confidence or 0.0)),
                            details={"conf": proposal.confidence, "min": c.min_confidence or 0.0}))
    if c.max_horizon_sec:
        checks.append(RiskCheck("max_horizon_sec", ok=(proposal.horizon_sec <= c.max_horizon_sec),
                                details={"h": proposal.horizon_sec, "max": c.max_horizon_sec}))

    # Position count
    if c.max_positions and c.max_positions > 0:
        checks.append(RiskCheck("max_positions", ok=(len(proposal.orders) <= c.max_positions),
                                details={"n": len(proposal.orders), "max": c.max_positions}))

    # Notionals
    gross = 0.0
    leg_violations: List[Tuple[str, float]] = []
    for o in proposal.orders:
        n = notional_usd(o.symbol, o.qty, ctx.prices, ctx.fx_usd_per_base)
        gross += n
        if c.max_leg_notional_usd and c.max_leg_notional_usd > 0 and n > c.max_leg_notional_usd:
            leg_violations.append((o.symbol, n))
    if c.max_notional_usd and c.max_notional_usd > 0:
        checks.append(RiskCheck("max_notional_usd", ok=(gross <= c.max_notional_usd),
                                details={"gross": gross, "max": c.max_notional_usd}))
    if c.max_leg_notional_usd and c.max_leg_notional_usd > 0:
        checks.append(RiskCheck("max_leg_notional_usd", ok=(len(leg_violations) == 0),
                                details={"violations": leg_violations} if leg_violations else {}))

    # Leverage (very rough: gross / cash)
    if c.max_leverage and c.max_leverage > 0:
        cash = float(ctx.balances.get("CASH", 0.0))
        lev = (gross / max(1.0, cash)) if cash > 0 else float("inf")
        checks.append(RiskCheck("max_leverage", ok=(lev <= c.max_leverage),
                                details={"gross": gross, "cash": cash, "lev": lev, "max": c.max_leverage}))

    return checks


# ---------------------- core abstract base ----------------------

class AgentBase(abc.ABC):
    """
    All agents must implement:
      - propose(context) -> Proposal
      - risk(proposal, context) -> RiskReport
      - explain(proposal, risk) -> str
    """

    name: str = "agent"

    @abc.abstractmethod
    def propose(self, context: MarketContext) -> Proposal:
        """Generate a Proposal given current market context."""
        raise NotImplementedError

    @abc.abstractmethod
    def risk(self, proposal: Proposal, context: MarketContext) -> RiskReport:
        """Run pre‑trade risk checks; return a RiskReport."""
        raise NotImplementedError

    @abc.abstractmethod
    def explain(self, proposal: Proposal, risk: RiskReport | None = None) -> str:
        """Human‑readable explanation of the idea."""
        raise NotImplementedError

    # ---- optional utilities shared by subclasses ----

    def base_risk(self, proposal: Proposal, context: MarketContext) -> RiskReport:
        """
        A sensible default: apply constraint checks and compute (gross/net) exposure.
        Subclasses can extend or override.
        """
        checks = check_constraints(proposal, context)

        gross = 0.0
        net = 0.0
        for o in proposal.orders:
            n = notional_usd(o.symbol, o.qty, context.prices, context.fx_usd_per_base)
            gross += n
            sign = +1.0 if o.side == "BUY" else -1.0
            net += sign * n

        ok = all(ch.ok for ch in checks)
        return RiskReport(
            ok=ok,
            exposure_usd=net,
            gross_notional_usd=gross,
            checks=checks,
            notes="; ".join([f"{c.name}:{'ok' if c.ok else 'fail'}" for c in checks]),
        )

    # quick helper to build a single‑leg proposal
    def single_leg(self, *, symbol: str, side: Side, qty: float, thesis: str,
                   score: float = 0.0, horizon_sec: float = 300.0,
                   confidence: float = 0.5, venue: Optional[str] = None,
                   type: OrderType = "MARKET", limit_price: Optional[float] = None,
                   tags: Optional[Sequence[str]] = None,
                   diagnostics: Optional[Dict[str, Any]] = None) -> Proposal:
        return Proposal(
            orders=[OrderPlan(symbol=symbol, side=side, qty=qty, type=type, limit_price=limit_price, venue=venue)],
            thesis=thesis,
            score=score,
            horizon_sec=horizon_sec,
            confidence=confidence,
            tags=list(tags or []),
            diagnostics=dict(diagnostics or {}),
        )