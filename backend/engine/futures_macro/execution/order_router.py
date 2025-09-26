# engines/futures_macro/execution/order_router.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Protocol, Literal
import math
import time
import uuid
import logging

import pandas as pd

from engines.futures_macro.backtest.pnl import ContractSpec # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Side = Literal["BUY", "SELL"]
OrderType = Literal["MKT", "LMT", "POV", "ICEBERG"]
TIF = Literal["DAY", "IOC", "FOK", "GTC"]

# ============================ Data models ============================

@dataclass
class Order:
    """
    Parent order for a single futures symbol (contracts, not shares).
    `meta` can include anything useful: {"strategy":"CTM-01","account":"FUT-USD"}
    """
    symbol: str
    side: Side
    contracts: float                      # unsigned contracts (router will slice/round)
    price: Optional[float] = None         # for LMT
    order_type: OrderType = "MKT"
    tif: TIF = "DAY"
    broker: str = "paper"
    account: str = "DEFAULT"
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict = field(default_factory=dict)

@dataclass
class ChildOrder:
    parent_id: str
    symbol: str
    side: Side
    contracts: float
    price: Optional[float]
    order_type: OrderType
    tif: TIF
    broker: str
    account: str
    child_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ExecutionReport:
    child_id: str
    parent_id: str
    symbol: str
    side: Side
    filled_contracts: float = 0.0
    avg_price: float = 0.0
    status: Literal["NEW","PARTIAL","FILLED","CANCELED","REPLACED","REJECTED"] = "NEW"
    reason: Optional[str] = None
    ts: float = field(default_factory=time.time)

# ============================ Broker interface ============================

class BrokerAdapter(Protocol):
    name: str
    def send(self, order: ChildOrder, *, last_price: float, spec: ContractSpec) -> ExecutionReport: ...
    def cancel(self, child_id: str) -> ExecutionReport: ...
    def replace(self, child_id: str, new_contracts: Optional[float] = None,
                new_price: Optional[float] = None) -> ExecutionReport: ...
    def get(self, child_id: str) -> ExecutionReport: ...

# ============================ Paper broker ============================

class PaperFuturesBroker(BrokerAdapter):
    """
    Deterministic fills:
      - MKT/POV/ICEBERG fill immediately at (last_price ± slip_ticks * tick_size)
      - LMT fills at limit if market is crossable (we simulate by comparing to last_price)
    """
    def __init__(self, name: str = "paper", slip_ticks: float = 0.25):
        self.name = name
        self.slip_ticks = float(slip_ticks)
        self._orders: Dict[str, ExecutionReport] = {}

    def _slipped_px(self, side: Side, last_price: float, spec: ContractSpec) -> float:
        delta = self.slip_ticks * spec.tick_size
        return last_price + (delta if side == "BUY" else -delta)

    def send(self, order: ChildOrder, *, last_price: float, spec: ContractSpec) -> ExecutionReport:
        if order.order_type in ("MKT","POV","ICEBERG"):
            px = self._slipped_px(order.side, last_price, spec)
            rep = ExecutionReport(
                child_id=order.child_id, parent_id=order.parent_id,
                symbol=order.symbol, side=order.side,
                filled_contracts=float(order.contracts), avg_price=float(px),
                status="FILLED",
            )
            self._orders[order.child_id] = rep
            return rep

        # LMT logic
        lim = order.price if order.price is not None else last_price
        crossable = (order.side == "BUY" and lim >= last_price) or (order.side == "SELL" and lim <= last_price)
        if crossable:
            rep = ExecutionReport(
                child_id=order.child_id, parent_id=order.parent_id,
                symbol=order.symbol, side=order.side,
                filled_contracts=float(order.contracts), avg_price=float(lim),
                status="FILLED",
            )
        else:
            rep = ExecutionReport(
                child_id=order.child_id, parent_id=order.parent_id,
                symbol=order.symbol, side=order.side,
                filled_contracts=0.0, avg_price=float(lim),
                status="REJECTED", reason="LIMIT_NOT_CROSSABLE",
            )
        self._orders[order.child_id] = rep
        return rep

    def cancel(self, child_id: str) -> ExecutionReport:
        rep = self._orders.get(child_id)
        if not rep:
            return ExecutionReport(child_id=child_id, parent_id="", symbol="", side="BUY",
                                   status="REJECTED", reason="UNKNOWN_CHILD_ID")
        rep.status = "CANCELED"
        return rep

    def replace(self, child_id: str, new_contracts: Optional[float] = None,
                new_price: Optional[float] = None) -> ExecutionReport:
        rep = self._orders.get(child_id)
        if not rep:
            return ExecutionReport(child_id=child_id, parent_id="", symbol="", side="BUY",
                                   status="REJECTED", reason="UNKNOWN_CHILD_ID")
        if rep.status == "FILLED":
            return ExecutionReport(child_id=child_id, parent_id=rep.parent_id, symbol=rep.symbol, side=rep.side,
                                   status="REJECTED", reason="ALREADY_FILLED")
        rep.status = "REPLACED"
        if new_contracts is not None:
            rep.filled_contracts = float(new_contracts)
        if new_price is not None:
            rep.avg_price = float(new_price)
        return rep

    def get(self, child_id: str) -> ExecutionReport:
        return self._orders.get(child_id) or ExecutionReport(
            child_id=child_id, parent_id="", symbol="", side="BUY",
            status="REJECTED", reason="UNKNOWN_CHILD_ID"
        )

# ============================ Risk limits ============================

@dataclass
class RiskLimits:
    max_child_contracts: float = 2_000.0          # per child slice
    max_child_notional_usd: float = 50_000_000.0  # per child notional cap
    max_symbol_participation: float = 0.15        # vs ADV (contracts)
    banned_symbols: List[str] = field(default_factory=list)
    allow_shorts: bool = True

# ============================ Router config ============================

@dataclass
class RouteConfig:
    default_broker: str = "paper"
    child_clip_contracts: float = 200.0   # max contracts per child slice (pre-risk)
    pov_participation: float = 0.05       # 5% ADV per slice for POV
    iceberg_peak_contracts: float = 50.0  # visible size
    throttle_ms: int = 10                 # tiny sleep between child sends
    default_tif: TIF = "DAY"

# ============================ Router ============================

class OrderRouter:
    """
    Macro futures order router with:
      - MKT/LMT/POV/ICEBERG slicing in *contracts*
      - Notional & participation risk checks
      - Pluggable broker adapters (paper by default)
    """
    def __init__(self, specs: Dict[str, ContractSpec],
                 adapters: Optional[Dict[str, BrokerAdapter]] = None,
                 cfg: Optional[RouteConfig] = None,
                 limits: Optional[RiskLimits] = None):
        self.specs = specs
        self.adapters = adapters or {"paper": PaperFuturesBroker()}
        self.cfg = cfg or RouteConfig()
        self.limits = limits or RiskLimits()
        self._parent_to_children: Dict[str, List[str]] = {}
        self._children: Dict[str, ChildOrder] = {}

    # -- Utilities --

    def _adapter(self, name: Optional[str]) -> BrokerAdapter:
        key = name or self.cfg.default_broker
        if key not in self.adapters:
            raise ValueError(f"Broker adapter '{key}' not registered")
        return self.adapters[key]

    def _risk_check(self, parent: Order, last_price: float,
                    adv_contracts: Optional[float]) -> Optional[str]:
        if parent.symbol in set(self.limits.banned_symbols):
            return f"Symbol {parent.symbol} banned"
        if (parent.side == "SELL") and (not self.limits.allow_shorts):
            return "Shorts disabled"
        # child limits checked post-slice; quick pre-check on parent notional:
        spec = self.specs[parent.symbol]
        notional = abs(parent.contracts) * last_price * spec.multiplier
        if notional > self.limits.max_child_notional_usd:
            return f"Order notional {notional:,.0f} exceeds child cap {self.limits.max_child_notional_usd:,.0f}"
        if adv_contracts and adv_contracts > 0:
            part = abs(parent.contracts) / adv_contracts
            if part > self.limits.max_symbol_participation + 1e-12:
                return f"Participation {part:.2%} exceeds cap {self.limits.max_symbol_participation:.0%}"
        return None

    # -- Public API --

    def route_batch(
        self,
        orders: Iterable[Order],
        last_prices: Dict[str, float],                     # symbol → last/settlement
        adv_contracts_map: Optional[Dict[str, float]] = None,  # symbol → ADV in *contracts*
    ) -> List[ExecutionReport]:
        reports: List[ExecutionReport] = []
        adv_contracts_map = adv_contracts_map or {}

        for parent in orders:
            lp = float(last_prices.get(parent.symbol, 0.0))
            if lp <= 0:
                reports.append(ExecutionReport(
                    child_id="", parent_id=parent.client_order_id, symbol=parent.symbol, side=parent.side,
                    status="REJECTED", reason="NO_MARKET_PRICE"
                ))
                continue

            reason = self._risk_check(parent, lp, adv_contracts_map.get(parent.symbol))
            if reason:
                logger.warning(f"Risk reject {parent.symbol}: {reason}")
                reports.append(ExecutionReport(
                    child_id="", parent_id=parent.client_order_id, symbol=parent.symbol, side=parent.side,
                    status="REJECTED", reason=reason
                ))
                continue

            children = self._slice(parent, lp, adv_contracts_map.get(parent.symbol))
            self._parent_to_children[parent.client_order_id] = [c.child_id for c in children]

            adapter = self._adapter(parent.broker)
            spec = self.specs[parent.symbol]
            for ch in children:
                rep = adapter.send(ch, last_price=lp, spec=spec)
                self._children[ch.child_id] = ch
                reports.append(rep)
                time.sleep(self.cfg.throttle_ms / 1000.0)

        return reports

    def cancel_parent(self, parent_id: str) -> List[ExecutionReport]:
        out: List[ExecutionReport] = []
        for child_id in self._parent_to_children.get(parent_id, []):
            ch = self._children.get(child_id)
            if not ch:
                continue
            rep = self._adapter(ch.broker).cancel(child_id)
            out.append(rep)
        return out

    def replace_parent(self, parent_id: str, *,
                       new_contracts: Optional[float] = None,
                       new_price: Optional[float] = None) -> List[ExecutionReport]:
        out: List[ExecutionReport] = []
        for child_id in self._parent_to_children.get(parent_id, []):
            ch = self._children.get(child_id)
            if not ch:
                continue
            rep = self._adapter(ch.broker).replace(child_id, new_contracts, new_price)
            out.append(rep)
        return out

    # -- Slicing --

    def _slice(self, parent: Order, last_price: float,
               adv_contracts: Optional[float]) -> List[ChildOrder]:
        # base clip
        clip = max(1.0, float(self.cfg.child_clip_contracts))

        if parent.order_type == "POV" and adv_contracts and adv_contracts > 0:
            clip = min(clip, float(self.cfg.pov_participation) * float(adv_contracts))

        if parent.order_type == "ICEBERG":
            clip = min(clip, float(self.cfg.iceberg_peak_contracts))

        # enforce child contracts cap from limits
        clip = min(clip, float(self.limits.max_child_contracts))

        n_children = max(1, int(math.ceil(abs(parent.contracts) / clip)))
        per_child = parent.contracts / n_children

        children: List[ChildOrder] = []
        for _ in range(n_children):
            children.append(ChildOrder(
                parent_id=parent.client_order_id,
                symbol=parent.symbol,
                side=parent.side,
                contracts=float(per_child),
                price=parent.price,
                order_type=parent.order_type,
                tif=parent.tif,
                broker=parent.broker,
                account=parent.account,
            ))
        return children

# ============================ Convenience helpers ============================

def default_router(specs: Dict[str, ContractSpec]) -> OrderRouter:
    return OrderRouter(specs=specs, adapters={"paper": PaperFuturesBroker()}, cfg=RouteConfig(), limits=RiskLimits())

def orders_df_to_parents(orders_df) -> List[Order]:
    """
    Convert allocator output (with 'order_contracts','price','side','symbol') into parent Orders.
    """
    parents: List[Order] = []
    for sym, row in orders_df.iterrows():
        parents.append(Order(
            symbol=str(sym),
            side=("BUY" if row["order_contracts"] > 0 else "SELL"),
            contracts=abs(float(row["order_contracts"])),
            price=(float(row["price"]) if "price" in row and not pd.isna(row["price"]) else None),
            order_type=(row.get("order_type","MKT") if hasattr(row, "get") else "MKT"),
            meta={"source":"allocator"}
        ))
    return parents