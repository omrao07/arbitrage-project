# backend/execution_plus/arb_router/router.py
"""
Global Arbitrage Router

Plans and executes orders across multiple venues:
- Pull adapters via discovery.load_adapters_from_yaml (or HUB fallback)
- Get fresh quotes for the target symbol
- Estimate all‑in costs via execution_plus.cost_model.DefaultCostModel
- Rank venues by *effective unit cost* (price +/- costs)
- Optionally split across top‑K venues (size‑weighted by depth proxy)
- Place child orders and aggregate fills into a single report

Env knobs (defaults shown):
  ROUTER_TOPK=2                   # split across best K venues
  ROUTER_MIN_CHILD_NOTIONAL=5000  # don't send < this notional to any venue
  ROUTER_DRY_RUN=false            # plan without placing orders
  ROUTER_MAX_SLIPPAGE_BPS=50      # guardrail vs mid for marketables

Dependencies already in your tree:
  - backend/execution_plus/adapters.py   (AdapterBase, Order, OrderType, Side, Quote)
  - backend/execution_plus/cost_model.py (DefaultCostModel)
  - backend/execution_plus/arb_router/discovery.py (Discovery)
  - backend/execution_plus/registry.py   (HUB as fallback)

Usage:
  from backend.execution_plus.arb_router.router import ArbRouter
  r = ArbRouter("backend/config/venues.yaml")
  plan = r.plan(Order(symbol="BTCUSDT", side=Side.BUY, qty=0.5))
  report = r.execute(plan)  # or r.route(order) to plan+execute
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.execution_plus.adapters import AdapterBase, Order, OrderType, Side, Quote # type: ignore
from backend.execution_plus.cost_model import DefaultCostModel, CostBreakdown, get_default_model # type: ignore
from backend.execution_plus.arb_router.discovery import Discovery, DiscoveryResult
from backend.execution_plus.registry import HUB # type: ignore


# ----------------------------- models ---------------------------------

@dataclass
class VenueScore:
    venue_id: str
    adapter: AdapterBase
    quote: Quote
    cost: CostBreakdown
    eff_unit_price: float   # all‑in effective unit price
    spread_bps: Optional[float] = None


@dataclass
class RouteLeg:
    venue_id: str
    qty: float
    order: Order
    score: VenueScore


@dataclass
class RoutePlan:
    symbol: str
    side: Side
    total_qty: float
    legs: List[RouteLeg] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fill:
    venue_id: str
    ok: bool
    order_id: Optional[str]
    filled_qty: float
    avg_price: Optional[float]
    fees: float
    status: str
    raw: Dict[str, Any]


@dataclass
class RouteReport:
    ok: bool
    symbol: str
    side: Side
    requested_qty: float
    filled_qty: float
    vw_price: Optional[float]
    legs: List[Fill]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ----------------------------- helpers --------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name, "") or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _spread_bps(q: Quote) -> Optional[float]:
    if q.bid is None or q.ask is None or q.bid <= 0 or q.ask <= q.bid:
        return None
    mid = q.mid or (q.bid + q.ask) / 2.0
    if mid <= 0:
        return None
    return 10_000.0 * (q.ask - q.bid) / mid


def _eff_unit_price(side: Side, mid: float, cost_total: float, qty: float) -> float:
    """
    Convert total USD costs to a per‑unit add‑on to price.
    Assumes FX already applied inside CostBreakdown.
    """
    addon = (cost_total / max(qty, 1e-12))
    # For BUY, effective price = mid + addons; for SELL, subtract addons
    return (mid + addon) if side == Side.BUY else (mid - addon)


# ----------------------------- router ---------------------------------

class ArbRouter:
    def __init__(
        self,
        venues_yaml: Optional[str] = None,
        *,
        cost_model: Optional[DefaultCostModel] = None,
        topk: Optional[int] = None,
        min_child_notional: Optional[float] = None,
        max_slippage_bps: Optional[float] = None,
    ) -> None:
        self.venues_yaml = venues_yaml or os.getenv("VENUES_YAML", "backend/config/venues.yaml")
        self.discovery = Discovery(self.venues_yaml)
        self.cost_model = cost_model or get_default_model()
        self.topk = int(topk or _env_int("ROUTER_TOPK", 2))
        self.min_child_notional = float(min_child_notional or _env_float("ROUTER_MIN_CHILD_NOTIONAL", 5_000.0))
        self.max_slippage_bps = float(max_slippage_bps or _env_float("ROUTER_MAX_SLIPPAGE_BPS", 50.0))

        # cache discovery snapshot
        self._snap: Optional[DiscoveryResult] = None

    # -------- core API --------

    def route(self, order: Order, *, dry_run: Optional[bool] = None) -> RouteReport:
        """Plan + (optionally) execute in one call."""
        plan = self.plan(order)
        return self.execute(plan, dry_run=dry_run)

    def plan(self, order: Order) -> RoutePlan:
        """
        Build a route plan by scoring venues and optionally splitting across best‑K.
        """
        snap = self._ensure_snapshot()
        symbol = order.symbol
        side = order.side
        qty = float(order.qty)

        # 1) Pull fresh quotes + score each venue for this symbol
        scores = self._score_all(symbol, side, qty)

        if not scores:
            return RoutePlan(symbol=symbol, side=side, total_qty=qty, legs=[], meta={"reason": "no_quotes"})

        # sort by effective unit price (lowest for BUY, highest for SELL)
        rev = (side == Side.SELL)
        scores = sorted(scores, key=lambda s: s.eff_unit_price, reverse=rev)

        # 2) Allocate qty across top‑K by a simple capacity proxy (inverse spread)
        remaining = qty
        legs: List[RouteLeg] = []
        selected = scores[: max(1, self.topk)]

        # capacity weights ~ 1/spread (tighter spreads → more capacity).
        weights: List[float] = []
        for s in selected:
            sp = max(1.0, (s.spread_bps or 8.0))
            weights.append(1.0 / sp)
        wsum = sum(weights) or 1.0
        allocs = [remaining * (w / wsum) for w in weights]

        # enforce min child notional (drop tiny legs)
        pruned: List[Tuple[VenueScore, float]] = []
        for sc, qchild in zip(selected, allocs):
            notional = (sc.quote.mid or 0.0) * qchild
            if notional >= self.min_child_notional:
                pruned.append((sc, qchild))

        # if everything pruned, send all to best venue
        if not pruned:
            best = selected[0]
            pruned = [(best, remaining)]

        # 3) Build legs (Market or marketable Limit)
        legs_out: List[RouteLeg] = []
        for sc, qchild in pruned:
            # guardrail: limit extreme slippage vs mid
            mid = sc.quote.mid or 0.0
            if order.type == OrderType.MARKET:
                limit_px = None
            else:
                limit_px = order.limit_price

            # optional soft price‑band for marketables
            band = (self.max_slippage_bps / 10_000.0) * mid
            if order.type == OrderType.MARKET:
                # convert to "protection limit" if you want; keep MARKET for mocks
                child = Order(symbol=symbol, side=side, qty=qchild, type=OrderType.MARKET, venue_id=sc.venue_id, meta=dict(order.meta))
            else:
                # keep user's limit but clamp to protection band relative to mid
                px = float(limit_px if limit_px is not None else mid + (band if side == Side.BUY else -band))
                child = Order(symbol=symbol, side=side, qty=qchild, type=OrderType.LIMIT, limit_price=px, venue_id=sc.venue_id, meta=dict(order.meta))

            legs_out.append(RouteLeg(venue_id=sc.venue_id, qty=qchild, order=child, score=sc))

        return RoutePlan(symbol=symbol, side=side, total_qty=qty, legs=legs_out, meta={
            "ranked": [self._score_summary(s) for s in scores],
            "topk": self.topk,
            "min_child_notional": self.min_child_notional,
        })

    def execute(self, plan: RoutePlan, *, dry_run: Optional[bool] = None) -> RouteReport:
        """
        Place child orders as per plan and aggregate the fills.
        """
        if dry_run is None:
            dry_run = _env_bool("ROUTER_DRY_RUN", False)

        fills: List[Fill] = []
        total_qty = 0.0
        total_cash = 0.0  # signed cash flow (+ for SELL, - for BUY)
        any_ok = False

        for leg in plan.legs:
            adapter = leg.score.adapter
            if dry_run:
                # fabricate a dry‑run fill at the quote mid
                px = leg.score.quote.mid or 0.0
                fqty = leg.qty
                fees = leg.score.cost.fees
                fills.append(Fill(
                    venue_id=leg.venue_id, ok=True, order_id=None,
                    filled_qty=fqty, avg_price=px, fees=fees,
                    status="dry_run", raw={"note": "dry_run"}
                ))
                any_ok = True
                total_qty += fqty
                total_cash += (px * fqty) * (+1 if plan.side == Side.SELL else -1) - fees
                continue

            try:
                res = adapter.place_order(leg.order)
                fills.append(Fill(
                    venue_id=leg.venue_id, ok=res.ok, order_id=res.order_id,
                    filled_qty=res.filled_qty, avg_price=res.avg_price,
                    fees=res.fees, status=res.status, raw=res.raw
                ))
                if res.ok and res.filled_qty > 0 and res.avg_price is not None:
                    any_ok = True
                    total_qty += float(res.filled_qty)
                    total_cash += (float(res.avg_price) * float(res.filled_qty)) * (+1 if plan.side == Side.SELL else -1) - float(res.fees)
            except Exception as e:
                fills.append(Fill(
                    venue_id=leg.venue_id, ok=False, order_id=None,
                    filled_qty=0.0, avg_price=None, fees=0.0, status="error",
                    raw={"exception": str(e)}
                ))

        vw_price = (abs(total_cash) / max(total_qty, 1e-12)) if total_qty > 0 else None
        return RouteReport(
            ok=any_ok,
            symbol=plan.symbol,
            side=plan.side,
            requested_qty=plan.total_qty,
            filled_qty=total_qty,
            vw_price=vw_price,
            legs=fills,
            diagnostics={"plan_meta": plan.meta},
        )

    # -------- scoring --------

    def _score_all(self, symbol: str, side: Side, qty: float) -> List[VenueScore]:
        snap = self._ensure_snapshot()
        scores: List[VenueScore] = []

        adapters = snap.adapters or HUB.adapters.all()
        for vid, adapter in adapters.items():
            # Only venues that list the symbol
            try:
                if symbol not in (adapter.get_symbols() or []):
                    continue
            except Exception:
                continue

            try:
                q = adapter.get_quote(symbol)
                mid = q.mid or (q.bid or 0.0 + q.ask or 0.0) / 2.0
                # Estimate costs (let caller pass better vol/fx later if needed)
                cb = self.cost_model.estimate(adapter, Order(symbol=symbol, side=side, qty=qty, type=OrderType.MARKET), q)
                eff = _eff_unit_price(side, mid, cb.total, qty)
                scores.append(VenueScore(
                    venue_id=vid, adapter=adapter, quote=q, cost=cb,
                    eff_unit_price=eff, spread_bps=_spread_bps(q)
                ))
            except Exception:
                continue

        return scores

    # -------- discovery cache --------

    def _ensure_snapshot(self) -> DiscoveryResult:
        if self._snap is None:
            self._snap = self.discovery.run()
        return self._snap

    # -------- misc --------

    @staticmethod
    def _score_summary(s: VenueScore) -> Dict[str, Any]:
        return {
            "venue": s.venue_id,
            "mid": s.quote.mid,
            "spread_bps": s.spread_bps,
            "fees": s.cost.fees,
            "impact": s.cost.impact,
            "latency": s.cost.latency_adverse,
            "total_cost": s.cost.total,
            "eff_unit_price": s.eff_unit_price,
        }


# ----------------------------- CLI ------------------------------------

if __name__ == "__main__":
    # Tiny self‑test with built‑in mock adapters
    from backend.execution_plus.adapters import OrderType, Side # type: ignore

    router = ArbRouter()  # will fall back to HUB mock adapters if venues.yaml missing
    order = Order(symbol="BTCUSDT", side=Side.BUY, qty=0.25, type=OrderType.MARKET)
    plan = router.plan(order)
    print("Plan:")
    for leg in plan.legs:
        print(" ", leg.venue_id, "qty=", round(leg.qty, 6), "mid=", round(leg.score.quote.mid or 0.0, 4),
              "eff_unit=", round(leg.score.eff_unit_price, 6))

    report = router.execute(plan, dry_run=True)
    print("\nReport (dry‑run):", report.ok, "filled", report.filled_qty, "@", report.vw_price)
    for f in report.legs:
        print(" ", f.venue_id, f.status, "qty=", f.filled_qty, "px=", f.avg_price, "fees=", f.fees)