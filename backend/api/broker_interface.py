# backend/execution/broker_interface.py
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol


# =======================
# Data Models
# =======================

@dataclass
class Order:
    id: str
    symbol: str
    side: str          # "buy" | "sell"
    qty: float         # absolute units (shares/contracts)
    price: float       # decision/limit reference (for market we still keep last seen)
    type: str = "market"     # "market" | "limit"
    tif: str = "day"         # "day" | "ioc" | "gtc"
    strategy: str = "unknown"
    ts: float = field(default_factory=lambda: time.time())
    attrs: Dict[str, Any] = field(default_factory=dict)  # venue, algo, etc.


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    qty: float         # signed (+buy, -sell)
    price: float
    fee: float
    ts: float


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

    def apply_fill(self, f: Fill):
        # update average price logically for long/short
        if self.qty == 0 or (self.qty > 0 and f.qty > 0) or (self.qty < 0 and f.qty < 0):
            new_qty = self.qty + f.qty
            if abs(new_qty) > 1e-12:
                self.avg_price = ((abs(self.qty) * self.avg_price) + (abs(f.qty) * f.price)) / abs(new_qty)
            else:
                self.avg_price = 0.0
            self.qty = new_qty
            return

        # offsetting or flipping
        offset = -min(abs(self.qty), abs(f.qty)) * (1 if f.qty < 0 else -1)
        self.qty += offset
        remainder = f.qty - offset
        if remainder != 0:
            # flipped
            self.avg_price = f.price
            self.qty += remainder
        elif abs(self.qty) < 1e-12:
            self.avg_price = 0.0


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"


# =======================
# Broker Interface
# =======================

class BaseBroker(Protocol):
    name: str

    # lifecycle
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    # state
    def get_account(self) -> Account: ...
    def get_positions(self) -> Dict[str, Position]: ...
    def get_open_orders(self) -> Dict[str, Order]: ...

    # trading
    def place_order(self, order: Order) -> Fill: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def replace_order(self, order_id: str, new_qty: Optional[float] = None, new_price: Optional[float] = None) -> bool: ...

    # (optional) market context (used by PaperBroker)
    def set_prices(self, prices: Dict[str, float]) -> None: ...


# =======================
# Paper Broker (local sim)
# =======================

class PaperBroker:
    """
    Simple synchronous simulator.
    - Fills at current price (last mark) with optional slippage/fees.
    - Tracks cash, positions, equity.
    """
    name = "paper"

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        fees_bps: float = 2.0,
        slippage_bps: float = 1.0,
        base_ccy: str = "USD",
    ):
        self._cash = float(starting_cash)
        self._fees_bps = float(fees_bps)
        self._slip_bps = float(slippage_bps)
        self._ccy = base_ccy

        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._prices: Dict[str, float] = {}

    # ----- lifecycle -----
    def connect(self) -> None:  # noqa
        pass

    def disconnect(self) -> None:  # noqa
        pass

    # ----- state -----
    def get_account(self) -> Account:
        equity = self._cash + sum(pos.qty * self._prices.get(sym, pos.avg_price) for sym, pos in self._positions.items())
        # naive x2 buying power
        bp = self._cash * 2.0
        return Account(equity=equity, cash=self._cash, buying_power=bp, currency=self._ccy)

    def get_positions(self) -> Dict[str, Position]:
        return {k: Position(symbol=v.symbol, qty=v.qty, avg_price=v.avg_price) for k, v in self._positions.items()}

    def get_open_orders(self) -> Dict[str, Order]:
        return dict(self._orders)

    # ----- market context -----
    def set_prices(self, prices: Dict[str, float]) -> None:
        self._prices.update({k: float(v) for k, v in prices.items()})

    # ----- trading -----
    def _slip_price(self, side: str, px: float) -> float:
        sgn = +1 if side.lower().startswith("b") else -1
        return px * (1.0 + sgn * self._slip_bps / 1e4)

    def _fee_for_notional(self, qty: float, px: float) -> float:
        return abs(qty * px) * (self._fees_bps / 1e4)

    def place_order(self, order: Order) -> Fill:
        sym = order.symbol
        px = order.price
        last = self._prices.get(sym, px)
        fill_px = self._slip_price(order.side, last)
        fee = self._fee_for_notional(order.qty, fill_px)

        # cash move: buys spend, sells receive
        signed_qty = order.qty if order.side.lower().startswith("b") else -order.qty
        cash_delta = -(signed_qty * fill_px) - fee  # negative when buy
        self._cash += cash_delta

        # positions
        pos = self._positions.get(sym, Position(symbol=sym))
        f = Fill(
            order_id=order.id,
            symbol=sym,
            side=order.side,
            qty=signed_qty,
            price=fill_px,
            fee=fee,
            ts=time.time(),
        )
        pos.apply_fill(f)
        self._positions[sym] = pos

        # record & close immediately (market IOC fill model)
        self._orders[order.id] = order

        return f

    def cancel_order(self, order_id: str) -> bool:
        return self._orders.pop(order_id, None) is not None

    def replace_order(self, order_id: str, new_qty: Optional[float] = None, new_price: Optional[float] = None) -> bool:
        o = self._orders.get(order_id)
        if not o:
            return False
        if new_qty is not None:
            o.qty = float(new_qty)
        if new_price is not None:
            o.price = float(new_price)
        return True


# =======================
# IBKR Adapter (skeleton)
# =======================

class IBKRBroker:
    """
    Thin adapter over IB API / ib_insync (recommended).
    Fill in TODOs with real calls once you add dependency.
    """
    name = "ibkr"

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1, currency: str = "USD"):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.currency = currency
        self._connected = False
        # self.ib = IB()  # from ib_insync

    def connect(self) -> None:
        # self.ib.connect(self.host, self.port, clientId=self.client_id)
        self._connected = True

    def disconnect(self) -> None:
        # self.ib.disconnect()
        self._connected = False

    def get_account(self) -> Account:
        # acct_values = self.ib.accountSummary()
        # equity = float(...)
        # cash = float(...)
        # bp = float(...)
        # return Account(equity=equity, cash=cash, buying_power=bp, currency=self.currency)
        return Account(equity=0.0, cash=0.0, buying_power=0.0, currency=self.currency)  # TODO

    def get_positions(self) -> Dict[str, Position]:
        # pos = {}
        # for p in self.ib.positions():
        #     sym = p.contract.symbol
        #     pos[sym] = Position(symbol=sym, qty=p.position, avg_price=p.avgCost)
        # return pos
        return {}

    def get_open_orders(self) -> Dict[str, Order]:
        # return {str(o.order.orderId): ...}
        return {}

    def place_order(self, order: Order) -> Fill:
        # TODO: translate Order -> IB Order, Contract; submit; await fill
        # For now, raise so we don't silently "fake" fills
        raise NotImplementedError("Wire IBKR API/ib_insync here")

    def cancel_order(self, order_id: str) -> bool:
        # self.ib.cancelOrder( ... )
        return False

    def replace_order(self, order_id: str, new_qty: Optional[float] = None, new_price: Optional[float] = None) -> bool:
        # IB uses cancel/replace; implement accordingly
        return False

    def set_prices(self, prices: Dict[str, float]) -> None:  # not used for live broker
        pass


# =======================
# Zerodha (Kite) Adapter (skeleton)
# =======================

class ZerodhaBroker:
    """
    Adapter over Kite Connect (zerodha). Replace TODOs with real SDK calls.
    """
    name = "zerodha"

    def __init__(self, api_key: str, access_token: str, user_id: Optional[str] = None, currency: str = "INR"):
        self.api_key = api_key
        self.access_token = access_token
        self.user_id = user_id
        self.currency = currency
        self._connected = False
        # self.kite = KiteConnect(api_key=api_key); self.kite.set_access_token(access_token)

    def connect(self) -> None:
        # test auth / profile
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_account(self) -> Account:
        # profile = self.kite.margins()
        # equity = ...
        # cash = ...
        # bp = ...
        return Account(equity=0.0, cash=0.0, buying_power=0.0, currency=self.currency)  # TODO

    def get_positions(self) -> Dict[str, Position]:
        # positions = {}
        # for p in self.kite.positions()["net"]:
        #     sym = p["tradingsymbol"]
        #     positions[sym] = Position(symbol=sym, qty=float(p["quantity"]), avg_price=float(p["average_price"]))
        return {}

    def get_open_orders(self) -> Dict[str, Order]:
        return {}

    def place_order(self, order: Order) -> Fill:
        # Map to kite.order_place(...)
        raise NotImplementedError("Wire Zerodha Kite Connect SDK here")

    def cancel_order(self, order_id: str) -> bool:
        return False

    def replace_order(self, order_id: str, new_qty: Optional[float] = None, new_price: Optional[float] = None) -> bool:
        return False

    def set_prices(self, prices: Dict[str, float]) -> None:
        pass


# =======================
# Factory
# =======================

def make_broker(cfg: Dict[str, Any]) -> BaseBroker:
    """
    Factory: chooses broker based on cfg.
    Example cfg:
        {
          "broker": {"name": "paper", "fees_bps": 2.0, "slippage_bps": 1.5, "starting_cash": 200000},
          # or
          "broker": {"name": "ibkr", "host":"127.0.0.1","port":7497,"client_id":1},
          # or
          "broker": {"name": "zerodha", "api_key":"...", "access_token":"..."}
        }
    """
    bcfg = cfg.get("broker", {}) or {}
    name = str(bcfg.get("name", "paper")).lower()

    if name == "paper":
        return PaperBroker(
            starting_cash=float(bcfg.get("starting_cash", 100_000)),
            fees_bps=float(bcfg.get("fees_bps", 2.0)),
            slippage_bps=float(bcfg.get("slippage_bps", 1.0)),
            base_ccy=str(bcfg.get("currency", "USD")),
        )

    if name == "ibkr":
        return IBKRBroker(
            host=str(bcfg.get("host", "127.0.0.1")),
            port=int(bcfg.get("port", 7497)),
            client_id=int(bcfg.get("client_id", 1)),
            currency=str(bcfg.get("currency", "USD")),
        )

    if name == "zerodha":
        return ZerodhaBroker(
            api_key=str(bcfg["api_key"]),
            access_token=str(bcfg["access_token"]),
            user_id=bcfg.get("user_id"),
            currency=str(bcfg.get("currency", "INR")),
        )

    raise ValueError(f"Unknown broker: {name}")


# =======================
# Convenience helpers
# =======================

def new_order(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    *,
    type: str = "market",
    tif: str = "day",
    strategy: str = "unknown",
    attrs: Optional[Dict[str, Any]] = None,
) -> Order:
    return Order(
        id=str(uuid.uuid4()),
        symbol=symbol,
        side=side.lower(),
        qty=float(abs(qty)),
        price=float(price),
        type=type,
        tif=tif,
        strategy=strategy,
        attrs=dict(attrs or {}),
    )