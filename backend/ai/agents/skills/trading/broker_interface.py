# backend/brokers/broker_interface.py
from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Any

Side = Literal["buy", "sell"]

# -------------------------------------------------------------------
# Common order & fill models
# -------------------------------------------------------------------

@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    qty: float
    order_type: str = "market"        # "market" | "limit"
    limit_price: Optional[float] = None
    tif: str = "DAY"                  # time-in-force
    extra: Dict[str, Any] = None # type: ignore

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    px: float
    ts: float

@dataclass
class Position:
    symbol: str
    qty: float
    avg_px: float

@dataclass
class Account:
    cash: float
    equity: float
    margin_used: float

# -------------------------------------------------------------------
# Abstract Broker
# -------------------------------------------------------------------

class BrokerInterface(abc.ABC):
    """
    Abstract base class for all brokers (real or paper).
    Your OMS / allocator / execution engine should only talk to this.
    """

    @abc.abstractmethod
    def place_order(self, order: Order) -> str:
        """Submit order → return broker-assigned id (or reuse order.id)."""
        ...

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order; return True if acknowledged."""
        ...

    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Return order status (e.g., open/filled/cancelled, fill qty)."""
        ...

    @abc.abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Return dict of open positions by symbol."""
        ...

    @abc.abstractmethod
    def get_account(self) -> Account:
        """Return current account summary."""
        ...

# -------------------------------------------------------------------
# Example PaperBroker (in-memory simulator)
# -------------------------------------------------------------------

class PaperBroker(BrokerInterface):
    """
    Very simple in-memory paper broker:
      • market orders fill immediately at last_price (if provided)
      • limit orders fill if mark crosses limit
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, Fill] = {}
        self.positions: Dict[str, Position] = {}
        self.cash: float = 1_000_000.0
        self.equity: float = 1_000_000.0
        self.last_price: Dict[str, float] = {}  # symbol -> px
        self._oid = 0

    def _gen_id(self) -> str:
        self._oid += 1
        return f"PB-{self._oid}"

    def place_order(self, order: Order) -> str:
        oid = order.id or self._gen_id()
        order.id = oid
        self.orders[oid] = order
        px = self.last_price.get(order.symbol, order.limit_price or 0.0)
        if order.order_type == "market":
            self._fill(order, px)
        elif order.order_type == "limit" and order.limit_price is not None:
            if (order.side == "buy" and px <= order.limit_price) or (order.side == "sell" and px >= order.limit_price):
                self._fill(order, px)
        return oid

    def _fill(self, order: Order, px: float):
        fill = Fill(order_id=order.id, symbol=order.symbol, side=order.side,
                    qty=order.qty, px=px, ts=time.time())
        self.fills[order.id] = fill
        pos = self.positions.get(order.symbol, Position(symbol=order.symbol, qty=0.0, avg_px=px))
        # update avg_px
        new_qty = pos.qty + (order.qty if order.side == "buy" else -order.qty)
        if new_qty != 0:
            pos.avg_px = ((pos.qty * pos.avg_px) + (order.qty if order.side == "buy" else -order.qty) * px) / new_qty
        pos.qty = new_qty
        self.positions[order.symbol] = pos
        # update cash/equity
        notional = order.qty * px
        if order.side == "buy":
            self.cash -= notional
        else:
            self.cash += notional
        self.equity = self.cash + sum(p.qty * self.last_price.get(p.symbol, p.avg_px) for p in self.positions.values())

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders and order_id not in self.fills:
            del self.orders[order_id]
            return True
        return False

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        if order_id in self.fills:
            return {"status": "filled", "fill": self.fills[order_id]}
        elif order_id in self.orders:
            return {"status": "open", "order": self.orders[order_id]}
        return {"status": "unknown"}

    def get_positions(self) -> Dict[str, Position]:
        return self.positions

    def get_account(self) -> Account:
        return Account(cash=self.cash, equity=self.equity, margin_used=0.0)