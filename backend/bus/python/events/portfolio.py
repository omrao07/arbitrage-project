# bus/python/events/portfolio.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Literal


# =========================
# Base event
# =========================
@dataclass
class PortfolioEvent:
    event_type: str                 # "order", "order_ack", "fill", "position", "pnl", etc.
    ts_event: int                   # business event time (ms since epoch, UTC)
    ts_ingest: int                  # ingestion time (ms since epoch, UTC)
    source: str                     # e.g., "ems", "oms", "broker:ib", "sim"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioEvent":
        return cls(**d)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @classmethod
    def _base(cls, event_type: str, source: str, ts_event: Optional[int] = None) -> Dict[str, Any]:
        now = cls._now_ms()
        return {
            "event_type": event_type,
            "ts_event": ts_event if ts_event is not None else now,
            "ts_ingest": now,
            "source": source,
        }


Side = Literal["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"]
OrderType = Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT", "PEG", "MOC", "LOC"]
TimeInForce = Literal["DAY", "GTC", "IOC", "FOK", "OPG", "GTD"]
OrderStatus = Literal["NEW", "ACK", "PARTIALLY_FILLED", "FILLED", "CANCELED", "REJECTED", "REPLACED"]


# =========================
# Order (submission)
# =========================
@dataclass
class Order(PortfolioEvent):
    order_id: str                   # system order id (UUID)
    client_order_id: Optional[str]  # external id / EMS clOrdId
    portfolio_id: str               # account/portfolio
    strategy_tag: Optional[str]     # logical strategy/book name
    symbol: str
    side: Side
    quantity: float                 # shares/contracts
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = "DAY"
    venue: Optional[str] = None     # exchange/venue code
    currency: str = "USD"
    notional: Optional[float] = None
    algo: Optional[str] = None      # e.g., "VWAP", "POV", "TWAP"
    algo_params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        order_id: str,
        portfolio_id: str,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: OrderType,
        source: str = "ems",
        ts_event: Optional[int] = None,
        client_order_id: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tif: TimeInForce = "DAY",
        venue: Optional[str] = None,
        currency: str = "USD",
        notional: Optional[float] = None,
        strategy_tag: Optional[str] = None,
        algo: Optional[str] = None,
        algo_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> "Order":
        base = cls._base("order", source, ts_event)
        return cls(
            order_id=order_id,
            client_order_id=client_order_id,
            portfolio_id=portfolio_id,
            strategy_tag=strategy_tag,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            order_type=order_type,
            limit_price=None if limit_price is None else float(limit_price),
            stop_price=None if stop_price is None else float(stop_price),
            tif=tif,
            venue=venue,
            currency=currency,
            notional=None if notional is None else float(notional),
            algo=algo,
            algo_params=algo_params or {},
            tags=tags or [],
            **base,
        )


# =========================
# Order lifecycle events
# =========================
@dataclass
class OrderAck(PortfolioEvent):
    order_id: str
    status: OrderStatus = "ACK"
    venue_order_id: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def create(cls, order_id: str, source: str = "ems", ts_event: Optional[int] = None,
               venue_order_id: Optional[str] = None, message: Optional[str] = None) -> "OrderAck":
        base = cls._base("order_ack", source, ts_event)
        return cls(order_id=order_id, venue_order_id=venue_order_id, message=message, **base)


@dataclass
class OrderCancel(PortfolioEvent):
    order_id: str
    reason: Optional[str] = None

    @classmethod
    def create(cls, order_id: str, source: str = "ems", ts_event: Optional[int] = None,
               reason: Optional[str] = None) -> "OrderCancel":
        base = cls._base("order_cancel", source, ts_event)
        return cls(order_id=order_id, reason=reason, **base)


@dataclass
class OrderReject(PortfolioEvent):
    order_id: str
    code: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def create(cls, order_id: str, source: str = "ems", ts_event: Optional[int] = None,
               code: Optional[str] = None, message: Optional[str] = None) -> "OrderReject":
        base = cls._base("order_reject", source, ts_event)
        return cls(order_id=order_id, code=code, message=message, **base)


@dataclass
class OrderReplace(PortfolioEvent):
    order_id: str
    new_quantity: Optional[float] = None
    new_limit_price: Optional[float] = None
    new_stop_price: Optional[float] = None
    reason: Optional[str] = None

    @classmethod
    def create(cls, order_id: str, source: str = "ems", ts_event: Optional[int] = None,
               new_quantity: Optional[float] = None, new_limit_price: Optional[float] = None,
               new_stop_price: Optional[float] = None, reason: Optional[str] = None) -> "OrderReplace":
        base = cls._base("order_replace", source, ts_event)
        return cls(order_id=order_id, new_quantity=new_quantity, new_limit_price=new_limit_price,
                   new_stop_price=new_stop_price, reason=reason, **base)


# =========================
# Execution / Fill
# =========================
@dataclass
class Fill(PortfolioEvent):
    order_id: str
    fill_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    venue: Optional[str] = None
    commission: float = 0.0
    fees: float = 0.0
    slippage_bps: Optional[float] = None
    currency: str = "USD"
    liquidity: Optional[Literal["MAKER", "TAKER"]] = None

    @property
    def notional(self) -> float:
        return float(self.quantity) * float(self.price)

    @classmethod
    def create(
        cls,
        order_id: str,
        fill_id: str,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        source: str = "oms",
        ts_event: Optional[int] = None,
        venue: Optional[str] = None,
        commission: float = 0.0,
        fees: float = 0.0,
        slippage_bps: Optional[float] = None,
        currency: str = "USD",
        liquidity: Optional[Literal["MAKER", "TAKER"]] = None,
    ) -> "Fill":
        base = cls._base("fill", source, ts_event)
        return cls(
            order_id=order_id,
            fill_id=fill_id,
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            price=float(price),
            venue=venue,
            commission=float(commission),
            fees=float(fees),
            slippage_bps=None if slippage_bps is None else float(slippage_bps),
            currency=currency,
            liquidity=liquidity,
            **base,
        )


# =========================
# Positions
# =========================
@dataclass
class PositionSnapshot(PortfolioEvent):
    portfolio_id: str
    symbol: str
    quantity: float
    avg_price: float                 # average cost
    market_price: float
    currency: str = "USD"
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    beta: Optional[float] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def market_value(self) -> float:
        return float(self.quantity) * float(self.market_price)

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        symbol: str,
        quantity: float,
        avg_price: float,
        market_price: float,
        source: str = "risk",
        ts_event: Optional[int] = None,
        currency: str = "USD",
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        beta: Optional[float] = None,
        sector: Optional[str] = None,
        country: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "PositionSnapshot":
        base = cls._base("position", source, ts_event)
        return cls(
            portfolio_id=portfolio_id,
            symbol=symbol,
            quantity=float(quantity),
            avg_price=float(avg_price),
            market_price=float(market_price),
            currency=currency,
            realized_pnl=float(realized_pnl),
            unrealized_pnl=float(unrealized_pnl),
            beta=beta,
            sector=sector,
            country=country,
            tags=tags or [],
            **base,
        )


@dataclass
class PositionDelta(PortfolioEvent):
    portfolio_id: str
    symbol: str
    delta_qty: float
    reason: Literal["FILL", "CORPORATE_ACTION", "MANUAL_ADJUST", "DIVIDEND", "SPLIT"] = "FILL"
    reference_id: Optional[str] = None  # e.g., fill_id or corporate action id

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        symbol: str,
        delta_qty: float,
        source: str = "risk",
        ts_event: Optional[int] = None,
        reason: Literal["FILL", "CORPORATE_ACTION", "MANUAL_ADJUST", "DIVIDEND", "SPLIT"] = "FILL",
        reference_id: Optional[str] = None,
    ) -> "PositionDelta":
        base = cls._base("position_delta", source, ts_event)
        return cls(
            portfolio_id=portfolio_id,
            symbol=symbol,
            delta_qty=float(delta_qty),
            reason=reason,
            reference_id=reference_id,
            **base,
        )


# =========================
# PnL
# =========================
@dataclass
class PnLSnapshot(PortfolioEvent):
    portfolio_id: str
    currency: str = "USD"
    # point-in-time aggregates
    nav: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    beta_exposure: Optional[float] = None
    # daily components
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    fees: float = 0.0
    commissions: float = 0.0
    borrow_costs: float = 0.0
    slippage_costs: float = 0.0
    # attribution buckets (sector/factor/strategy)
    attribution: Dict[str, float] = field(default_factory=dict)
    # risk metrics snapshot (optional, keep light)
    var_95: Optional[float] = None
    var_99: Optional[float] = None

    @property
    def pnl_total(self) -> float:
        return float(self.pnl_realized) + float(self.pnl_unrealized) - float(self.fees) - float(self.commissions) - float(self.borrow_costs) - float(self.slippage_costs)

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        nav: float,
        gross_exposure: float,
        net_exposure: float,
        source: str = "risk",
        ts_event: Optional[int] = None,
        currency: str = "USD",
        pnl_realized: float = 0.0,
        pnl_unrealized: float = 0.0,
        fees: float = 0.0,
        commissions: float = 0.0,
        borrow_costs: float = 0.0,
        slippage_costs: float = 0.0,
        attribution: Optional[Dict[str, float]] = None,
        beta_exposure: Optional[float] = None,
        var_95: Optional[float] = None,
        var_99: Optional[float] = None,
    ) -> "PnLSnapshot":
        base = cls._base("pnl", source, ts_event)
        return cls(
            portfolio_id=portfolio_id,
            currency=currency,
            nav=float(nav),
            gross_exposure=float(gross_exposure),
            net_exposure=float(net_exposure),
            beta_exposure=beta_exposure,
            pnl_realized=float(pnl_realized),
            pnl_unrealized=float(pnl_unrealized),
            fees=float(fees),
            commissions=float(commissions),
            borrow_costs=float(borrow_costs),
            slippage_costs=float(slippage_costs),
            attribution=attribution or {},
            var_95=var_95,
            var_99=var_99,
            **base,
        )


# =========================
# Helpers
# =========================
def to_json(obj: Any) -> str:
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if hasattr(obj, "to_dict"):
        return json.dumps(obj.to_dict(), ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False)


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # 1) Order submission
    o = Order.create(
        order_id="o-123",
        client_order_id="cl-abc",
        portfolio_id="PORT-1",
        strategy_tag="EquityLS_Momo",
        symbol="AAPL",
        side="BUY",
        quantity=100,
        order_type="LIMIT",
        limit_price=190.25,
        venue="NASDAQ",
        tags=["live"],
    )
    print(o.to_json())

    # 2) Ack + partial fill
    ack = OrderAck.create(order_id=o.order_id, venue_order_id="ex-789")
    print(ack.to_json())

    f1 = Fill.create(
        order_id=o.order_id,
        fill_id="f-1",
        symbol=o.symbol,
        side=o.side,
        quantity=40,
        price=190.20,
        venue="NASDAQ",
        liquidity="TAKER",
        commission=0.2,
        fees=0.03,
    )
    print(f1.to_json())

    # 3) Position snapshot
    pos = PositionSnapshot.create(
        portfolio_id=o.portfolio_id,
        symbol=o.symbol,
        quantity=40,
        avg_price=190.20,
        market_price=190.40,
        beta=1.15,
        sector="Technology",
    )
    print(pos.to_json())

    # 4) Daily PnL snapshot
    pnl = PnLSnapshot.create(
        portfolio_id=o.portfolio_id,
        nav=10_000_000,
        gross_exposure=12_500_000,
        net_exposure=1_200_000,
        pnl_realized=1_250.0,
        pnl_unrealized=800.0,
        commissions=12.50,
        fees=3.20,
        borrow_costs=0.0,
        slippage_costs=5.0,
        attribution={"Sector:Tech": 900.0, "Factor:Momentum": 600.0},
        var_95=0.035,  # 3.5% NAV
    )
    print(pnl.to_json())