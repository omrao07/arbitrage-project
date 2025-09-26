# engines/equity_ls/execution/order_router.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Iterable, Tuple, Protocol, Literal
import time
import uuid
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Side = Literal["BUY", "SELL"]
TIF = Literal["DAY", "IOC", "FOK", "GTC"]
OrderType = Literal["MKT", "LMT", "POV"]  # POV = % of volume algo

# ============================ Data models ============================

@dataclass
class Order:
    """Parent order from allocator (shares)."""
    ticker: str
    side: Side
    qty: float                                   # signed handled via side; qty >= 0
    price: Optional[float] = None                # for LMT
    order_type: OrderType = "MKT"
    tif: TIF = "DAY"
    account: str = "DEFAULT"
    broker: str = "paper"
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict = field(default_factory=dict)     # freeform (strategy_id, engine, etc.)

@dataclass
class ChildOrder:
    """Child order sent to broker (slices)."""
    parent_id: str
    ticker: str
    side: Side
    qty: float
    price: Optional[float]
    order_type: OrderType
    tif: TIF
    account: str
    broker: str
    child_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ExecutionReport:
    """Normalized fill/cancel/replace events."""
    child_id: str
    parent_id: str
    ticker: str
    side: Side
    filled_qty: float = 0.0
    avg_price: float = 0.0
    status: Literal["NEW", "PARTIAL", "FILLED", "CANCELED", "REPLACED", "REJECTED"] = "NEW"
    reason: Optional[str] = None
    ts: float = field(default_factory=time.time)

# ============================ Broker Adapter interface ============================

class BrokerAdapter(Protocol):
    name: str
    def send(self, order: ChildOrder) -> ExecutionReport: ...
    def cancel(self, child_id: str) -> ExecutionReport: ...
    def replace(self, child_id: str, new_qty: Optional[float] = None,
                new_price: Optional[float] = None) -> ExecutionReport: ...
    def get(self, child_id: str) -> ExecutionReport: ...

# ============================ Paper adapter ============================

class PaperBroker(BrokerAdapter):
    """Deterministic, in-memory paper broker with immediate-or-close fills."""
    def __init__(self, name: str = "paper", slip_bps: float = 1.5):
        self.name = name
        self._orders: Dict[str, ExecutionReport] = {}
        self.slip_bps = slip_bps

    def _slip(self, price: Optional[float]) -> float:
        if price is None:
            price = 100.0
        return price * (1.0 + (self.slip_bps / 1e4))

    def send(self, order: ChildOrder) -> ExecutionReport:
        # Market/POV fill instantly at slipped price; Limit fills at limit if price is present
        px = self._slip(order.price)
        rep = ExecutionReport(
            child_id=order.child_id,
            parent_id=order.parent_id,
            ticker=order.ticker,
            side=order.side,
            filled_qty=order.qty,
            avg_price=float(px if order.order_type in ("MKT", "POV") else (order.price or px)),
            status="FILLED",
        )
        self._orders[order.child_id] = rep
        return rep

    def cancel(self, child_id: str) -> ExecutionReport:
        rep = self._orders.get(child_id)
        if not rep:
            rep = ExecutionReport(child_id=child_id, parent_id="", ticker="", side="BUY", status="REJECTED",
                                  reason="Unknown child_id")
        else:
            rep.status = "CANCELED"
        return rep

    def replace(self, child_id: str, new_qty: Optional[float] = None,
                new_price: Optional[float] = None) -> ExecutionReport:
        rep = self._orders.get(child_id)
        if not rep:
            return ExecutionReport(child_id=child_id, parent_id="", ticker="", side="BUY",
                                   status="REJECTED", reason="Unknown child_id")
        # simulate “replace” as cancel+new fill if not filled
        if rep.status == "FILLED":
            return ExecutionReport(child_id=child_id, parent_id=rep.parent_id, ticker=rep.ticker,
                                   side=rep.side, status="REJECTED", reason="Already filled")
        rep.status = "REPLACED"
        if new_qty:
            rep.filled_qty = new_qty
        if new_price:
            rep.avg_price = new_price
        return rep

    def get(self, child_id: str) -> ExecutionReport:
        return self._orders.get(child_id) or ExecutionReport(
            child_id=child_id, parent_id="", ticker="", side="BUY", status="REJECTED", reason="Unknown child_id"
        )

# ============================ Risk guards / Throttles ============================

@dataclass
class RiskLimits:
    max_order_notional: float = 5_000_000.0
    max_symbol_participation: float = 0.10     # <=10% of ADV
    banned_symbols: List[str] = field(default_factory=list)
    allow_shorts: bool = True

def check_order_risk(
    order: Order,
    last_price: float,
    adv_usd: Optional[float],
    limits: RiskLimits,
) -> Tuple[bool, Optional[str]]:
    if order.ticker in set(limits.banned_symbols):
        return False, f"Symbol {order.ticker} banned"
    notional = abs(order.qty) * float(last_price)
    if notional > limits.max_order_notional:
        return False, f"Order notional {notional:,.0f} exceeds max {limits.max_order_notional:,.0f}"
    if (order.side == "SELL") and (not limits.allow_shorts):
        return False, "Shorts disabled"
    if adv_usd is not None and adv_usd > 0:
        if (notional / adv_usd) > limits.max_symbol_participation + 1e-12:
            return False, f"Participation {(notional/adv_usd):.2%} exceeds cap {limits.max_symbol_participation:.0%}"
    return True, None

# ============================ Router ============================

@dataclass
class RouteConfig:
    default_broker: str = "paper"
    child_clip_shares: float = 5_000.0    # max shares per child slice
    pov_participation: float = 0.05       # 5% POV
    throttle_ms: int = 25                 # sleep between children to avoid bursts
    default_tif: TIF = "DAY"

class OrderRouter:
    """
    Order router with:
      - idempotent parent->children mapping
      - MKT/LMT/POV support
      - Risk checks (notional, %ADV, banned symbols, shorting)
      - Simple POV slicer across time (based on ADV proxy)
    """
    def __init__(self, adapters: Dict[str, BrokerAdapter], cfg: Optional[RouteConfig] = None,
                 limits: Optional[RiskLimits] = None):
        self.adapters = adapters
        self.cfg = cfg or RouteConfig()
        self.limits = limits or RiskLimits()
        self._parent_to_children: Dict[str, List[str]] = {}
        self._sent_child: Dict[str, ChildOrder] = {}

    # ---------- Utilities ----------

    def _adapter(self, name: Optional[str]) -> BrokerAdapter:
        key = name or self.cfg.default_broker
        if key not in self.adapters:
            raise ValueError(f"Broker adapter '{key}' not registered")
        return self.adapters[key]

    # ---------- Public API ----------

    def route_batch(
        self,
        orders: Iterable[Order],
        last_prices: Dict[str, float],
        adv_map: Optional[Dict[str, float]] = None,
    ) -> List[ExecutionReport]:
        """
        Routes a batch of parent orders. Returns a flat list of execution reports (children).
        """
        adv_map = adv_map or {}
        reports: List[ExecutionReport] = []

        for parent in orders:
            price = float(last_prices.get(parent.ticker, 0.0))
            adv_usd = adv_map.get(parent.ticker)

            ok, reason = check_order_risk(parent, price, adv_usd, self.limits)
            if not ok:
                logger.warning(f"Risk reject: {parent.ticker} — {reason}")
                reports.append(ExecutionReport(
                    child_id="",
                    parent_id=parent.client_order_id,
                    ticker=parent.ticker,
                    side=parent.side,
                    status="REJECTED",
                    reason=reason,
                ))
                continue

            children = self._slice(parent, price, adv_usd)
            self._parent_to_children[parent.client_order_id] = [c.child_id for c in children]

            for ch in children:
                rep = self._adapter(ch.broker).send(ch)
                self._sent_child[ch.child_id] = ch
                reports.append(rep)
                time.sleep(self.cfg.throttle_ms / 1000.0)

        return reports

    def cancel_parent(self, parent_id: str) -> List[ExecutionReport]:
        out: List[ExecutionReport] = []
        for child_id in self._parent_to_children.get(parent_id, []):
            ch = self._sent_child.get(child_id)
            if not ch:  # already gone
                continue
            rep = self._adapter(ch.broker).cancel(child_id)
            out.append(rep)
        return out

    def replace_parent(
        self,
        parent_id: str,
        new_qty: Optional[float] = None,
        new_price: Optional[float] = None,
    ) -> List[ExecutionReport]:
        out: List[ExecutionReport] = []
        for child_id in self._parent_to_children.get(parent_id, []):
            ch = self._sent_child.get(child_id)
            if not ch:
                continue
            rep = self._adapter(ch.broker).replace(child_id, new_qty=new_qty, new_price=new_price)
            out.append(rep)
        return out

    # ---------- Slicing ----------

    def _slice(self, parent: Order, last_price: float, adv_usd: Optional[float]) -> List[ChildOrder]:
        """
        Turn parent into one or more child orders based on order_type and router config.
        """
        if parent.order_type in ("MKT", "LMT"):
            qtys = self._chunk(parent.qty, self.cfg.child_clip_shares)
            return [self._child(parent, q, parent.price) for q in qtys]

        # POV: slice such that each child participates at most cfg.pov_participation of ADV
        if adv_usd and adv_usd > 0 and last_price > 0:
            max_shares = (adv_usd * self.cfg.pov_participation) / last_price
            clip = max(1.0, min(self.cfg.child_clip_shares, max_shares))
        else:
            clip = self.cfg.child_clip_shares
        qtys = self._chunk(parent.qty, clip)
        return [self._child(parent, q, parent.price) for q in qtys]

    def _chunk(self, qty: float, clip: float) -> List[float]:
        n_children = max(1, int(math.ceil(abs(qty) / max(1.0, clip))))
        # spread remainder equally
        base = qty / n_children
        return [base] * n_children

    def _child(self, parent: Order, qty: float, price: Optional[float]) -> ChildOrder:
        return ChildOrder(
            parent_id=parent.client_order_id,
            ticker=parent.ticker,
            side=parent.side,
            qty=float(qty),
            price=price,
            order_type=parent.order_type,
            tif=parent.tif or self.cfg.default_tif,
            account=parent.account,
            broker=parent.broker or self.cfg.default_broker,
        )

# ============================ Convenience wiring ============================

def default_router() -> OrderRouter:
    """Paper-only default router (safe to import in notebooks/tests)."""
    adapters = {"paper": PaperBroker()}
    return OrderRouter(adapters=adapters, cfg=RouteConfig(), limits=RiskLimits()) # type: ignore