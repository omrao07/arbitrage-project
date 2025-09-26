# agents/execution_agent.py
"""
ExecutionAgent
--------------
A pluggable execution layer for backtests/sims or paper trading.

Features
- Order routing to an abstract Broker (swap in your live/paper broker).
- Basic risk checks (per-order notional, per-symbol exposure, gross/net caps, leverage).
- Positions, cash, realized/unrealized PnL, fees, and slippage.
- Partial fills support.
- Mark-to-market updates from a price feed.
- Deterministic, test-friendly design (no external deps).

You can plug this into simulators/backtests by:
1) Feeding quotes via `update_price(symbol, price)`.
2) Submitting orders via `submit_order(...)`.
3) Calling `mark_to_market()` after price updates (or let submit/fill do it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Iterable, List, Tuple
import time


# ----------------------------- Domain Models ---------------------------------


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()


@dataclass(frozen=True)
class OrderId:
    val: str


@dataclass
class Order:
    order_id: OrderId
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    ts: float = field(default_factory=lambda: time.time())

    def notional_at(self, px: float) -> float:
        return abs(self.qty) * px


@dataclass
class Fill:
    order_id: OrderId
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    ts: float = field(default_factory=lambda: time.time())


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_px: float = 0.0  # average entry price of the open position
    realized_pnl: float = 0.0

    def apply_fill(self, fill: Fill) -> None:
        signed_qty = fill.qty if fill.side is Side.BUY else -fill.qty
        # If adding to same-side position or going from flat
        if self.qty == 0 or (self.qty > 0 and signed_qty > 0) or (self.qty < 0 and signed_qty < 0):
            new_qty = self.qty + signed_qty
            if new_qty == 0:
                # Round-trip to flat; realized PnL handled below
                pass
            else:
                # Weighted average price update
                if self.qty == 0:
                    self.avg_px = fill.price
                else:
                    total_cost = self.avg_px * abs(self.qty) + fill.price * abs(signed_qty)
                    self.avg_px = total_cost / (abs(self.qty) + abs(signed_qty))
            self.qty = new_qty
        else:
            # Closing or flipping
            close_qty = min(abs(self.qty), abs(signed_qty)) * (1 if signed_qty > 0 else 1)
            # Realized PnL is (exit - entry) * closed_quantity with sign of original position
            if self.qty > 0:  # closing long => sell fill
                self.realized_pnl += (fill.price - self.avg_px) * min(abs(self.qty), abs(signed_qty))
            else:  # closing short => buy fill
                self.realized_pnl += (self.avg_px - fill.price) * min(abs(self.qty), abs(signed_qty))
            self.qty += signed_qty
            if self.qty == 0:
                self.avg_px = 0.0
            elif (self.qty > 0 and signed_qty > 0) or (self.qty < 0 and signed_qty < 0):
                # Flipped and added beyond flat; reset avg_px to fill price for the new side
                self.avg_px = fill.price

    def unrealized_pnl(self, last_price: Optional[float]) -> float:
        if last_price is None or self.qty == 0:
            return 0.0
        if self.qty > 0:
            return (last_price - self.avg_px) * abs(self.qty)
        else:
            return (self.avg_px - last_price) * abs(self.qty)


# ----------------------------- Risk & Models ---------------------------------


@dataclass
class RiskLimits:
    max_per_order_notional: float = 1_000_000.0
    max_symbol_exposure_usd: float = 2_000_000.0
    max_gross_exposure_usd: float = 5_000_000.0
    max_leverage: float = 5.0  # gross_exposure / equity


class RiskReject(Exception):
    pass


class FeeModel:
    """Override for custom fees (per exchange/broker)."""

    def fee(self, symbol: str, qty: float, price: float) -> float:
        # Default: 1 bps of notional + flat 0.50
        return 0.0001 * abs(qty) * price + 0.50


class SlippageModel:
    """Override to simulate execution slippage."""

    def slip(self, side: Side, price: float, qty: float, symbol: str) -> float:
        # Default: 2 bps impact with sign
        bps = 0.0002
        return price * (1 + bps) if side is Side.BUY else price * (1 - bps)


# ------------------------------- Broker API ----------------------------------


class Broker:
    """
    Abstract broker. Implementations must provide:
    - place(order, last_price) -> Iterable[Fill]
    - cancel(order_id) -> None
    - open_orders() -> List[Order]
    """

    def place(self, order: Order, last_price: Optional[float]) -> Iterable[Fill]:
        raise NotImplementedError

    def cancel(self, order_id: OrderId) -> None:
        raise NotImplementedError

    def open_orders(self) -> List[Order]:
        return []


class InMemoryImmediateBroker(Broker):
    """
    A simple broker that:
    - Fills MARKET at last_price (after slippage).
    - Fills LIMIT if price is marketable.
    - Fills STOP when triggered.
    Supports partial fills only via chunk_size.
    """

    def __init__(self, fee_model: Optional[FeeModel] = None, slippage_model: Optional[SlippageModel] = None, chunk_size: float = 0.0):
        self._fee = fee_model or FeeModel()
        self._slip = slippage_model or SlippageModel()
        self._chunk_size = max(0.0, chunk_size)  # 0.0 => fill all
        self._working: Dict[str, Order] = {}

    def place(self, order: Order, last_price: Optional[float]) -> Iterable[Fill]:
        if order.type == OrderType.MARKET:
            yield from self._fill_now(order, last_price)
            return

        # For STOP/LIMIT we can store and reevaluate when price updates
        self._working[order.order_id.val] = order
        # Also try to fill immediately if marketable
        yield from self._try_fill_working(order.order_id, last_price)

    def cancel(self, order_id: OrderId) -> None:
        self._working.pop(order_id.val, None)

    def open_orders(self) -> List[Order]:
        return list(self._working.values())

    # --- Helpers ---

    def _fill_now(self, order: Order, last_price: Optional[float]) -> Iterable[Fill]:
        if last_price is None:
            return  # Cannot fill without a price
        px = self._slip.slip(order.side, last_price, order.qty, order.symbol)
        qtys = self._split(order.qty)
        for q in qtys:
            fee = self._fee.fee(order.symbol, q, px)
            yield Fill(order.order_id, order.symbol, order.side, q, px, fee)

    def _try_fill_working(self, order_id: OrderId, last_price: Optional[float]) -> Iterable[Fill]:
        if last_price is None:
            return
        order = self._working.get(order_id.val)
        if not order:
            return

        should_fill = False
        trigger_ok = True

        if order.type == OrderType.LIMIT and order.limit_price is not None:
            if order.side is Side.BUY:
                should_fill = last_price <= order.limit_price
            else:
                should_fill = last_price >= order.limit_price
        elif order.type == OrderType.STOP and order.stop_price is not None:
            if order.side is Side.BUY:
                trigger_ok = last_price >= order.stop_price
            else:
                trigger_ok = last_price <= order.stop_price
            # After trigger, treat as market
            if trigger_ok:
                should_fill = True

        if should_fill:
            # Remove working order and fill now (marketable)
            self._working.pop(order_id.val, None)
            yield from self._fill_now(order, last_price)

    def on_price(self, symbol: str, last_price: float) -> Iterable[Fill]:
        # Re-evaluate all working orders for this symbol
        for order in list(self._working.values()):
            if order.symbol != symbol:
                continue
            yield from self._try_fill_working(order.order_id, last_price)

    def _split(self, qty: float) -> Iterable[float]:
        if self._chunk_size <= 0 or abs(qty) <= self._chunk_size:
            yield abs(qty)
            return
        remaining = abs(qty)
        while remaining > 1e-12:
            q = min(self._chunk_size, remaining)
            remaining -= q
            yield q


# ----------------------------- Execution Agent --------------------------------


class ExecutionAgent:
    """
    Stateful execution layer that:
    - Tracks equity, cash, positions, orders.
    - Enforces RiskLimits before routing orders.
    - Applies fills to positions, cash, fees.
    - Computes PnL and exposures.
    """

    def __init__(
        self,
        starting_cash: float = 1_000_000.0,
        risk: Optional[RiskLimits] = None,
        broker: Optional[Broker] = None,
    ):
        self.cash: float = starting_cash
        self._risk = risk or RiskLimits()
        self._broker = broker or InMemoryImmediateBroker()
        self._positions: Dict[str, Position] = {}
        self._last: Dict[str, float] = {}  # last price per symbol
        self._order_seq = 0
        self._fills_log: List[Fill] = []

    # -------- Public API --------

    def submit_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderId:
        if qty <= 0:
            raise ValueError("qty must be > 0")

        last_px = self._last.get(symbol)
        order = Order(
            order_id=self._next_order_id(symbol),
            symbol=symbol,
            side=side,
            qty=qty,
            type=type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        # Risk checks pre-trade
        self._check_order_risk(order, last_px)

        # Route to broker
        for fill in self._broker.place(order, last_px):
            self._apply_fill(fill)

        return order.order_id

    def cancel(self, order_id: OrderId) -> None:
        self._broker.cancel(order_id)

    def update_price(self, symbol: str, last_price: float) -> None:
        self._last[symbol] = float(last_price)
        # Let broker re-evaluate stop/limit orders
        for fill in self._broker.on_price(symbol, last_price): # type: ignore
            self._apply_fill(fill)

    def mark_to_market(self) -> None:
        # No-op here, but kept for API symmetry; PnL is derived on demand.
        pass

    # -------- State & Metrics --------

    def position(self, symbol: str) -> Position:
        return self._positions.setdefault(symbol, Position(symbol))

    def portfolio_value(self) -> float:
        return self.cash + self.gross_unrealized_pnl()

    def equity(self) -> float:
        # Equity = cash + sum of unrealized + realized across positions
        return self.cash + self.gross_unrealized_pnl() + self.total_realized_pnl()

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())

    def gross_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl(self._last.get(sym)) for sym, p in self._positions.items())

    def exposure_by_symbol(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sym, pos in self._positions.items():
            px = self._last.get(sym)
            if px is None:
                continue
            out[sym] = abs(pos.qty) * px
        return out

    def gross_exposure(self) -> float:
        return sum(self.exposure_by_symbol().values())

    def net_exposure(self) -> float:
        total = 0.0
        for sym, pos in self._positions.items():
            px = self._last.get(sym)
            if px is None:
                continue
            total += pos.qty * px
        return total

    def leverage(self) -> float:
        eq = max(1e-9, self.equity())
        return self.gross_exposure() / eq

    def last_price(self, symbol: str) -> Optional[float]:
        return self._last.get(symbol)

    def fills(self) -> List[Fill]:
        return list(self._fills_log)

    def open_orders(self) -> List[Order]:
        return self._broker.open_orders()

    # -------- Internals --------

    def _next_order_id(self, symbol: str) -> OrderId:
        self._order_seq += 1
        return OrderId(f"{symbol}-{self._order_seq:08d}")

    def _check_order_risk(self, order: Order, last_price: Optional[float]) -> None:
        # Determine reference price for notional checks
        ref_px = last_price
        if order.type is OrderType.LIMIT and order.limit_price is not None:
            ref_px = order.limit_price
        if order.type is OrderType.STOP and last_price is not None:
            ref_px = last_price  # Pre-check against current market

        if ref_px is None:
            # Allow creating non-marketable orders without price; defer symbol exposure check.
            return

        notional = order.notional_at(ref_px)
        if notional > self._risk.max_per_order_notional:
            raise RiskReject(f"Order notional {notional:,.2f} exceeds per-order cap {self._risk.max_per_order_notional:,.2f}")

        # Projected symbol exposure
        pos = self.position(order.symbol)
        proj_qty = pos.qty + (order.qty if order.side is Side.BUY else -order.qty)
        proj_expo = abs(proj_qty) * ref_px
        if proj_expo > self._risk.max_symbol_exposure_usd:
            raise RiskReject(
                f"Projected {order.symbol} exposure {proj_expo:,.2f} exceeds per-symbol cap {self._risk.max_symbol_exposure_usd:,.2f}"
            )

        # Projected gross exposure
        current_expo = self.gross_exposure()
        # Adjust current exposure for this symbol difference
        cur_sym_expo = abs(pos.qty) * ref_px
        proj_gross = current_expo - cur_sym_expo + proj_expo
        if proj_gross > self._risk.max_gross_exposure_usd:
            raise RiskReject(
                f"Projected gross exposure {proj_gross:,.2f} exceeds cap {self._risk.max_gross_exposure_usd:,.2f}"
            )

        # Leverage check (approximate, using projected gross / current equity)
        eq = self.equity()
        if eq <= 0:
            raise RiskReject("Equity is non-positive; cannot take risk.")
        proj_lev = proj_gross / eq
        if proj_lev > self._risk.max_leverage:
            raise RiskReject(f"Projected leverage {proj_lev:.2f} exceeds max {self._risk.max_leverage:.2f}")

    def _apply_fill(self, fill: Fill) -> None:
        pos = self.position(fill.symbol)

        # Cash change: buys decrease cash, sells increase cash
        signed_qty = fill.qty if fill.side is Side.BUY else -fill.qty
        cash_delta = -signed_qty * fill.price - fill.fee  # fee always reduces cash
        self.cash += cash_delta

        # Update position & realized PnL
        pos.apply_fill(fill)

        # Log fill
        self._fills_log.append(fill)


# ----------------------------- Quick Example ----------------------------------
if __name__ == "__main__":
    # Minimal smoke test
    agent = ExecutionAgent(starting_cash=100_000.0)
    agent.update_price("AAPL", 200.0)

    oid = agent.submit_order("AAPL", Side.BUY, qty=100)  # market buy 100
    print("After buy -> cash:", round(agent.cash, 2), "pos:", agent.position("AAPL"))

    agent.update_price("AAPL", 210.0)
    agent.submit_order("AAPL", Side.SELL, qty=50, type=OrderType.LIMIT, limit_price=209.0)  # marketable now
    print("After partial sell -> cash:", round(agent.cash, 2), "pos:", agent.position("AAPL"))

    print("Equity:", round(agent.equity(), 2), "GrossEx:", round(agent.gross_exposure(), 2), "Leverage:", round(agent.leverage(), 3))