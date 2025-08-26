# coordinator.py
"""
Multi-agent Coordinator for the "swarm"

Responsibilities
----------------
1) Collect proposals from all enabled agents (fx/equities/crypto/commodities, etc.)
2) Run per-agent risk, then a simple *negotiation/consensus* to resolve conflicts.
3) Produce a consolidated trade slate (ExecutionDecision).
4) Optionally execute via the Global Arbitrage Router (plan + execute).

Design choices
--------------
- Negotiation uses a transparent scoring rule:
    contribution = score * confidence
  Pro/anti signals net out by symbol/side.
- Quantity aggregation:
    qty = median(agent_qtys) * clip(|net_score|, min_scale, max_scale)
- Guardrails:
    - per-name min absolute score
    - max number of legs
    - portfolio gross notional cap (optional via context.constraints)
- Router integration is optional (enable with do_execute=True).

You can call `Coordinator.step(context)` from a scheduler.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

# Agents & shared primitives
from agents.base import AgentBase, MarketContext, Proposal, RiskReport, OrderPlan, Constraints, clamp # type: ignore
from agents.crypto import CryptoAgent # type: ignore
from agents.equities import EquitiesAgent # type: ignore
from agents.fx import FXAgent # type: ignore
from agents.commodities import CommoditiesAgent # type: ignore

# Optional router (only if you plan/execute)
try:
    from backend.execution_plus.factory import build_router # type: ignore
    from backend.execution_plus.adapters import Order, OrderType, Side # type: ignore
    HAVE_ROUTER = True
except Exception:
    HAVE_ROUTER = False


# ------------------------------ models --------------------------------

@dataclass
class AgentOutcome:
    name: str
    proposal: Proposal
    risk: RiskReport


@dataclass
class LegDecision:
    symbol: str
    side: str            # "BUY" | "SELL"
    qty: float
    venue: Optional[str] = None
    rationale: str = ""
    contributors: List[str] = field(default_factory=list)   # agent names
    meta: Dict[str, float] = field(default_factory=dict)    # scores, conf, etc.


@dataclass
class ExecutionDecision:
    ok: bool
    legs: List[LegDecision]
    notes: str = ""
    diagnostics: Dict[str, any] = field(default_factory=dict) # type: ignore


# ------------------------------ coordinator ---------------------------

class Coordinator:
    """
    Orchestrates agent proposals, risk checks, negotiation, and (optional) routing.
    """

    def __init__(
        self,
        agents: Optional[List[AgentBase]] = None,
        *,
        min_abs_score: float = 0.20,          # discard tiny edges
        max_legs: int = 12,                   # cap final legs
        qty_scale_min: float = 0.75,          # scale factor at |net_score| ~= min_abs_score
        qty_scale_max: float = 1.75,          # scale factor at |net_score| ~= 1.0
        prefer_venue_votes: bool = True,      # if multiple venues suggested, pick by majority
        enable_router: bool = True            # allow route/execute if router available
    ) -> None:
        self.agents = agents or [CryptoAgent(), EquitiesAgent(), FXAgent(), CommoditiesAgent()]
        self.min_abs_score = float(min_abs_score)
        self.max_legs = int(max_legs)
        self.qty_scale_min = float(qty_scale_min)
        self.qty_scale_max = float(qty_scale_max)
        self.prefer_venue_votes = bool(prefer_venue_votes)
        self.enable_router = bool(enable_router and HAVE_ROUTER)
        self._router = build_router() if self.enable_router else None

    # ---- public API ----

    def step(self, ctx: MarketContext, *, do_execute: bool = False) -> ExecutionDecision:
        """
        Run one full coordination cycle:
          - query agents
          - risk gate
          - negotiate
          - (optional) execute
        """
        outcomes = self._gather(ctx)
        slate = self._negotiate(outcomes, ctx)

        if do_execute and self.enable_router and slate.ok and slate.legs:
            exec_report = self._execute_via_router(slate)
            slate.diagnostics["router_report"] = {
                "ok": exec_report.ok,
                "filled_qty": exec_report.filled_qty,
                "vw_price": exec_report.vw_price,
                "legs": [vars(l) for l in exec_report.legs],
            }
            slate.ok = slate.ok and exec_report.ok
            if not exec_report.ok:
                slate.notes += " | router execution had errors"

        return slate

    # ---- internal: gather ----

    def _gather(self, ctx: MarketContext) -> List[AgentOutcome]:
        outcomes: List[AgentOutcome] = []
        for agent in self.agents:
            try:
                prop = agent.propose(ctx)
                risk = agent.risk(prop, ctx)
                outcomes.append(AgentOutcome(agent.name, prop, risk))
            except Exception as e:
                outcomes.append(AgentOutcome(agent.name, Proposal(orders=[], thesis=f"error: {e}", score=0.0), RiskReport(ok=False)))
        return outcomes

    # ---- internal: negotiation ----

    def _negotiate(self, outcomes: List[AgentOutcome], ctx: MarketContext) -> ExecutionDecision:
        """
        Build a consolidated slate from agent outcomes.
        Aggregates by (symbol), nets BUY/SELL signals via signed contributions.
        """
        # symbol -> aggregation buckets
        agg: Dict[str, Dict[str, any]] = {} # type: ignore

        for out in outcomes:
            if not out.risk.ok or not out.proposal.orders:
                continue

            for o in out.proposal.orders:
                # Weighted contribution (score * conf), signed by side
                sc = float(out.proposal.score if out.proposal.score else 0.0)
                # prefer per-leg score if available in meta
                leg_sc = float(o.meta.get("score", sc)) if isinstance(o.meta, dict) else sc
                conf = float(out.proposal.confidence or 0.5)
                contrib = leg_sc * conf
                side_sign = +1.0 if o.side.upper() == "BUY" else -1.0
                key = o.symbol

                bucket = agg.setdefault(key, {
                    "sum": 0.0,
                    "sum_abs": 0.0,
                    "qtys": [],
                    "venues": [],
                    "contributors": [],
                    "notes": [],
                })

                bucket["sum"] += side_sign * contrib
                bucket["sum_abs"] += abs(contrib)
                bucket["qtys"].append(float(o.qty or 0.0))
                if o.venue:
                    bucket["venues"].append(str(o.venue))
                bucket["contributors"].append(out.name)
                bucket["notes"].append(getattr(out.proposal, "thesis", "")[:160])

        # Turn into LegDecisions
        legs: List[LegDecision] = []
        for sym, b in agg.items():
            net = float(b["sum"])
            if abs(net) < self.min_abs_score:
                continue

            # Side + size scaling
            side = "BUY" if net > 0 else "SELL"
            base_qty = _median(b["qtys"]) if b["qtys"] else 0.0
            scale = self._scale_from_net(abs(net))
            qty = max(0.0, base_qty * scale)

            # Venue vote (simple plurality)
            venue = None
            if self.prefer_venue_votes and b["venues"]:
                venue = _plurality(b["venues"])

            legs.append(LegDecision(
                symbol=sym, side=side, qty=qty, venue=venue,
                rationale=f"net={net:.2f} scale={scale:.2f} base_qty≈{base_qty:g}",
                contributors=sorted(set(b["contributors"])),
                meta={"net_score": net, "sum_abs": b["sum_abs"], "n_votes": len(b["contributors"])},
            ))

        # Sort by |net_score| desc and cap max_legs
        legs.sort(key=lambda L: abs(L.meta.get("net_score", 0.0)), reverse=True)
        legs = legs[: self.max_legs]

        if not legs:
            return ExecutionDecision(ok=False, legs=[], notes="No consensus legs above threshold.",
                                     diagnostics={"outcomes": _outcomes_debug(outcomes)})

        # Portfolio guardrail: respect gross notional cap (if any)
        max_notional = float(ctx.constraints.max_notional_usd or 0.0)
        if max_notional > 0.0:
            legs = self._apply_gross_cap(legs, ctx, max_notional)

        return ExecutionDecision(
            ok=len(legs) > 0,
            legs=legs,
            notes=f"selected {len(legs)} legs",
            diagnostics={"outcomes": _outcomes_debug(outcomes)}
        )

    def _scale_from_net(self, net_abs: float) -> float:
        """
        Map |net_score| in [min_abs_score .. 1] to [qty_scale_min .. qty_scale_max].
        """
        x0, x1 = self.min_abs_score, 1.0
        y0, y1 = self.qty_scale_min, self.qty_scale_max
        x = clamp((net_abs - x0) / max(1e-9, (x1 - x0)), 0.0, 1.0)
        return y0 + x * (y1 - y0)

    def _apply_gross_cap(self, legs: List[LegDecision], ctx: MarketContext, cap_usd: float) -> List[LegDecision]:
        """
        Scale down legs proportionally if gross notional exceeds cap.
        """
        # rough notional using ctx.prices; assumes USD or 1.0 FX
        gross = 0.0
        for L in legs:
            px = float(ctx.prices.get(L.symbol, 0.0))
            gross += abs(px * L.qty)

        if gross <= cap_usd or gross <= 0.0:
            return legs

        k = cap_usd / gross
        out: List[LegDecision] = []
        for L in legs:
            out.append(LegDecision(
                symbol=L.symbol, side=L.side, qty=L.qty * k, venue=L.venue,
                rationale=L.rationale + f" | scaled by {k:.3f} for gross cap",
                contributors=L.contributors, meta=L.meta
            ))
        return out

    # ---- internal: execution via router ----

    def _execute_via_router(self, slate: ExecutionDecision):
        """
        Convert LegDecision -> router orders and route each.
        """
        router = self._router
        reports = []
        total_qty = 0.0
        vw_cash = 0.0
        any_ok = False

        for L in slate.legs:
            try:
                order = Order(
                    symbol=L.symbol,
                    side=Side.BUY if L.side == "BUY" else Side.SELL,
                    qty=L.qty,
                    type=OrderType.MARKET,
                    venue_id=L.venue  # hint; router may ignore if None
                )
                rep = router.route(order, dry_run=False) # type: ignore
                reports.append(rep)
                if rep.ok and rep.filled_qty > 0 and rep.vw_price is not None:
                    any_ok = True
                    total_qty += rep.filled_qty
                    # signed cash: SELL positive, BUY negative
                    sgn = +1.0 if L.side == "SELL" else -1.0
                    vw_cash += sgn * rep.vw_price * rep.filled_qty
            except Exception as e:
                # synthesize a failed report-like dict
                reports.append(type("X", (), {"ok": False, "filled_qty": 0.0, "vw_price": None, "legs": [], "diagnostics": {"error": str(e)}}))

        # Build a minimal aggregate-like object to stash in diagnostics
        class _Agg:
            ok = any_ok
            filled_qty = total_qty
            vw_price = (abs(vw_cash) / max(total_qty, 1e-12)) if total_qty > 0 else None
            legs = [l for r in reports for l in getattr(r, "legs", [])]

        return _Agg()


# ------------------------------ helpers --------------------------------

def _median(xs: Iterable[float]) -> float:
    arr = sorted(float(x) for x in xs if x is not None)
    n = len(arr)
    if n == 0:
        return 0.0
    m = n // 2
    if n % 2 == 1:
        return arr[m]
    return 0.5 * (arr[m - 1] + arr[m])


def _plurality(items: Iterable[str]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _outcomes_debug(outcomes: List[AgentOutcome]) -> Dict[str, any]: # type: ignore
    dbg: Dict[str, any] = {} # type: ignore
    for out in outcomes:
        dbg[out.name] = {
            "ok": out.risk.ok,
            "orders": [vars(o) for o in out.proposal.orders],
            "score": out.proposal.score,
            "confidence": out.proposal.confidence,
            "thesis": out.proposal.thesis[:180] if out.proposal.thesis else "",
        }
    return dbg


# ------------------------------ tiny demo --------------------------------

if __name__ == "__main__":
    # Minimal smoke test with synthetic context
    ctx = MarketContext.now(
        prices={"BTCUSDT": 65000, "ETHUSDT": 3200, "AAPL": 210.0, "EURUSD": 1.09},
        signals={
            "btc_basis_annual": 0.08, "btc_funding_8h": 0.0001, "social_sent_btc": 0.35,
            "mom_z_AAPL": 1.0, "earn_surprise_AAPL": 0.06, "sent_AAPL": 0.3,
            "carry_EURUSD": 0.012, "mom_z_EURUSD": 0.8, "ppp_gap_EURUSD": -0.10,
        },
        balances={"CASH": 1_000_000},
        constraints=Constraints(max_notional_usd=250_000)
    )

    coord = Coordinator(enable_router=False)  # set True if you wired the router
    decision = coord.step(ctx, do_execute=False)
    print("OK:", decision.ok, "legs:", len(decision.legs), decision.notes)
    for L in decision.legs:
        print(" -", L.side, f"{L.qty:g}", L.symbol, "@", (L.venue or "ANY"), "|", L.rationale)