# engines/credit_cds/order_router.py
from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Iterable, Callable
import pandas as pd

# =============================================================================
# Data models
# =============================================================================

Side = str  # "BUY_PROTECTION" or "SELL_PROTECTION"

@dataclass(frozen=True)
class VenueConfig:
    name: str
    max_child_notional: float = 2_000_000.0     # slice size (USD)
    min_child_notional: float = 100_000.0
    throttle_per_sec: int = 5                   # max messages per second
    currency: str = "USD"
    supports_partial: bool = True               # allow partial fills
    slippage_bps_hint: float = 0.0              # optional hint for expected slip

@dataclass(frozen=True)
class OrderTicket:
    """High-level parent order from allocator/simulator."""
    ticker: str                   # CDS identifier (e.g., "IG_A_5Y", "CDX.NA.IG.40.5Y")
    side: Side                    # BUY_PROTECTION / SELL_PROTECTION
    notional_usd: float           # signed notional is discouraged; use side
    tenor_years: float = 5.0
    currency: str = "USD"
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChildOrder:
    venue: str
    parent_id: str
    ticker: str
    side: Side
    child_notional_usd: float
    tenor_years: float
    currency: str
    clordid: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "NEW"        # NEW / ACK / PARTIAL / FILLED / REJECTED / CANCELED
    filled_usd: float = 0.0
    avg_price_bps: Optional[float] = None
    reason: Optional[str] = None
    ts_epoch_ms: int = field(default_factory=lambda: int(time.time() * 1000))

@dataclass
class RiskLimits:
    max_gross_notional_usd: float = 50_000_000.0
    max_notional_per_name_usd: float = 10_000_000.0
    max_daily_trade_usd: float = 25_000_000.0
    allow_short_protection: bool = True

@dataclass
class RouteResult:
    parent_id: str
    status: str
    filled_usd: float
    avg_price_bps: Optional[float]
    children: List[ChildOrder]
    errors: List[str] = field(default_factory=list)

# =============================================================================
# Broker adapters (stubs)
# =============================================================================

class BaseBrokerAdapter:
    def __init__(self, venue_cfg: VenueConfig):
        self.cfg = venue_cfg
        self._last_second = 0
        self._msgs_this_second = 0

    def _rate_limit(self):
        now = int(time.time())
        if now != self._last_second:
            self._last_second = now
            self._msgs_this_second = 0
        if self._msgs_this_second >= self.cfg.throttle_per_sec:
            time.sleep(1.0)
            self._last_second = int(time.time())
            self._msgs_this_second = 0
        self._msgs_this_second += 1

    def send_child(self, child: ChildOrder) -> ChildOrder:
        raise NotImplementedError

class MockBrokerAdapter(BaseBrokerAdapter):
    """Simulated broker with instant acks and deterministic fills."""
    def __init__(self, venue_cfg: VenueConfig, fill_ratio: float = 1.0, slip_bps: float = 0.0):
        super().__init__(venue_cfg)
        self.fill_ratio = max(0.0, min(1.0, fill_ratio))
        self.slip_bps = slip_bps

    def send_child(self, child: ChildOrder) -> ChildOrder:
        self._rate_limit()
        child.status = "ACK"
        # Simulate fill
        fill = child.child_notional_usd * self.fill_ratio
        child.filled_usd = fill
        if fill <= 0:
            child.status = "REJECTED"
            child.reason = "Zero fill"
        elif fill < child.child_notional_usd:
            child.status = "PARTIAL" if self.cfg.supports_partial else "REJECTED"
        else:
            child.status = "FILLED"
        # Price: we represent execution as bps (spread), with optional slippage
        child.avg_price_bps = float(child.meta["px_hint_bps"]) + self.slip_bps if "px_hint_bps" in child.meta else self.slip_bps # type: ignore
        return child

class FixAdapter(BaseBrokerAdapter):
    """Sketch for a FIX adapter; wire to your FIX engine here."""
    def send_child(self, child: ChildOrder) -> ChildOrder:
        self._rate_limit()
        # TODO: translate to FIX (35=D), manage sessions, acks, fills, rejects.
        # For now behave like full fill with no slip.
        child.status = "FILLED"
        child.filled_usd = child.child_notional_usd
        child.avg_price_bps = float(child.meta["px_hint_bps"]) if "px_hint_bps" in child.meta else None # type: ignore
        return child

# =============================================================================
# Router
# =============================================================================

class OrderRouter:
    def __init__(
        self,
        venues: Dict[str, VenueConfig],
        adapters: Dict[str, BaseBrokerAdapter],
        limits: RiskLimits = RiskLimits(),
        position_provider: Optional[Callable[[str], float]] = None,  # current notional per name
        day_trade_provider: Optional[Callable[[], float]] = None,    # traded notional so far today
        pnl_hook: Optional[Callable[[ChildOrder], None]] = None,     # optional callback on fills
    ):
        self.venues = venues
        self.adapters = adapters
        self.limits = limits
        self.position_provider = position_provider or (lambda name: 0.0)
        self.day_trade_provider = day_trade_provider or (lambda: 0.0)
        self.pnl_hook = pnl_hook

    # --------------------------- Public API ---------------------------

    def route_from_df(
        self,
        trades_df: pd.DataFrame,
        venue_name: str,
        px_hint_bps: Optional[float] = None,
    ) -> List[RouteResult]:
        """
        trades_df columns: ['ticker','trade_notional','side'] (+ optional: tenor_years, currency, client_order_id)
        """
        results: List[RouteResult] = []
        for _, row in trades_df.iterrows():
            ticket = OrderTicket(
                ticker=str(row["ticker"]),
                side=str(row["side"]),
                notional_usd=float(row["trade_notional"]),
                tenor_years=float(row["tenor_years"]) if "tenor_years" in trades_df.columns else 5.0,
                currency=str(row["currency"]) if "currency" in trades_df.columns else self.venues[venue_name].currency,
                client_order_id=str(row["client_order_id"]) if "client_order_id" in trades_df.columns else str(uuid.uuid4()),
                meta={"px_hint_bps": None if px_hint_bps is None else float(px_hint_bps)},
            )
            res = self.route_order(ticket, venue_name)
            results.append(res)
        return results

    def route_order(self, ticket: OrderTicket, venue_name: str) -> RouteResult:
        """
        Pre-trade checks → slice → send children → aggregate fills.
        """
        errors = self._pretrade_checks(ticket)
        if errors:
            return RouteResult(parent_id=ticket.client_order_id, status="REJECTED", filled_usd=0.0, avg_price_bps=None, children=[], errors=errors)

        children = self._slice(ticket, self.venues[venue_name])
        fills: List[ChildOrder] = []
        adapter = self.adapters[venue_name]

        for child in children:
            # Add px hint to child metadata for adapters that want it
            child.meta = dict(ticket.meta) # type: ignore
            filled = adapter.send_child(child)
            fills.append(filled)
            if self.pnl_hook:
                try:
                    self.pnl_hook(filled)
                except Exception:
                    pass

        status, filled_usd, avg_px = self._aggregate(fills)
        return RouteResult(parent_id=ticket.client_order_id, status=status, filled_usd=filled_usd, avg_price_bps=avg_px, children=fills, errors=[])

    # --------------------------- Internals ---------------------------

    def _pretrade_checks(self, t: OrderTicket) -> List[str]:
        errs: List[str] = []
        if t.notional_usd == 0:
            errs.append("Zero notional")
        if t.side not in ("BUY_PROTECTION", "SELL_PROTECTION"):
            errs.append(f"Invalid side {t.side}")

        name = t.ticker.split("_")[0] if "_" in t.ticker else t.ticker
        cur_pos = float(self.position_provider(name))
        signed = t.notional_usd if t.side == "BUY_PROTECTION" else -t.notional_usd

        if not self.limits.allow_short_protection and signed < 0:
            errs.append("Short protection not allowed by limits")

        # Per-name and gross checks
        if abs(cur_pos + signed) > self.limits.max_notional_per_name_usd + 1e-6:
            errs.append(f"Per-name limit exceeded for {name}: {abs(cur_pos + signed):,.0f} > {self.limits.max_notional_per_name_usd:,.0f}")

        day_traded = float(self.day_trade_provider())
        if (day_traded + abs(t.notional_usd)) > self.limits.max_daily_trade_usd + 1e-6:
            errs.append(f"Daily trade limit exceeded: {day_traded + abs(t.notional_usd):,.0f} > {self.limits.max_daily_trade_usd:,.0f}")

        return errs

    def _slice(self, t: OrderTicket, venue: VenueConfig) -> List[ChildOrder]:
        """Create child orders respecting venue slice constraints."""
        size = abs(t.notional_usd)
        n_children = max(1, int(math.ceil(size / venue.max_child_notional)))
        child_size = min(venue.max_child_notional, max(venue.min_child_notional, size / n_children))

        children: List[ChildOrder] = []
        remaining = size
        for i in range(n_children):
            qty = child_size if remaining > child_size else remaining
            remaining -= qty
            children.append(
                ChildOrder(
                    venue=venue.name,
                    parent_id=t.client_order_id,
                    ticker=t.ticker,
                    side=t.side,
                    child_notional_usd=qty,
                    tenor_years=t.tenor_years,
                    currency=t.currency,
                )
            )
        return children

    def _aggregate(self, fills: List[ChildOrder]) -> Tuple[str, float, Optional[float]]:
        if not fills:
            return "REJECTED", 0.0, None
        total = sum(c.child_notional_usd for c in fills)
        filled = sum(c.filled_usd for c in fills)
        if filled == 0:
            status = "REJECTED" if any(c.status == "REJECTED" for c in fills) else "NOFILL"
            return status, 0.0, None
        wpx = [c.avg_price_bps * c.filled_usd for c in fills if c.avg_price_bps is not None]
        avg_px = (sum(wpx) / filled) if wpx else None
        if math.isclose(filled, total, rel_tol=1e-9, abs_tol=1.0):
            status = "FILLED"
        elif filled > 0:
            status = "PARTIAL"
        else:
            status = "NOFILL"
        return status, filled, avg_px

# =============================================================================
# Convenience: build router with defaults
# =============================================================================

def build_default_router() -> OrderRouter:
    venues = {
        "MOCK": VenueConfig(name="MOCK", max_child_notional=2_000_000, min_child_notional=100_000, throttle_per_sec=10, currency="USD", supports_partial=True),
    }
    adapters = {
        "MOCK": MockBrokerAdapter(venues["MOCK"], fill_ratio=1.0, slip_bps=0.0),
    }
    limits = RiskLimits(
        max_gross_notional_usd=50_000_000,
        max_notional_per_name_usd=10_000_000,
        max_daily_trade_usd=25_000_000,
        allow_short_protection=True,
    )
    return OrderRouter(venues=venues, adapters=adapters, limits=limits) # type: ignore

# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example: route trades from allocator output
    router = build_default_router()

    # Mock providers
    current_positions = {"IG_A": 0.0, "HY_B": 1_000_000.0}
    day_traded = 0.0
    router.position_provider = lambda name: current_positions.get(name, 0.0)
    router.day_trade_provider = lambda: day_traded

    # Batch of trades
    trades = pd.DataFrame([
        {"ticker": "IG_A_5Y", "trade_notional": 3_500_000.0, "side": "BUY_PROTECTION"},
        {"ticker": "HY_B_5Y", "trade_notional": 1_200_000.0, "side": "SELL_PROTECTION"},
    ])

    results = router.route_from_df(trades, venue_name="MOCK", px_hint_bps=125.0)
    for r in results:
        print(f"Parent {r.parent_id} -> {r.status} | filled ${r.filled_usd:,.0f} @ {r.avg_price_bps} bps")
        for ch in r.children:
            print("  ", asdict(ch))