# backend/execution/broker_base.py
from __future__ import annotations

import enum
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("broker")
if not log.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ------------------------------- Types ---------------------------------

class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"

class TIF(str, enum.Enum):
    DAY = "day"
    IOC = "ioc"
    GTC = "gtc"

@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    tif: TIF = TIF.DAY
    client_order_id: Optional[str] = None
    strategy: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # venue, region, notes, etc.

@dataclass
class OrderAck:
    ok: bool
    order_id: Optional[str]
    reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    ts_ms: int
    strategy: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float

@dataclass
class Account:
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    currency: str = "USD"

# --------------------------- Broker Base --------------------------------

class BrokerBase:
    """
    Abstract broker adapter.
    Concrete adapters must implement: connect(), _place(), cancel(), positions(), account().

    Safety features provided here:
      • idempotency via client_order_id cache
      • lot-size & tick-size normalization
      • simple rate limiting
      • retry wrapper
    """

    # Per-symbol meta you can hydrate at connect(): lot, tick, venue symbol, etc.
    _instrument_meta: Dict[str, Dict[str, Any]]

    def __init__(self, *, rate_limit_per_sec: float = 8.0):
        self.connected: bool = False
        self._idempotency: Dict[str, str] = {}  # client_order_id -> broker order_id
        self._instrument_meta = {}
        self._rate_limit_per_sec = float(rate_limit_per_sec)
        self._last_call_ts = 0.0
        self._rate_lock = threading.Lock()

    # -------- lifecycle --------
    def connect(self) -> None:
        """Open sessions / authenticate. Concrete classes should set self.connected=True."""
        raise NotImplementedError

    def ensure_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("Broker not connected. Call connect() first.")

    def close(self) -> None:
        """Optional: close sessions/sockets."""
        self.connected = False

    # -------- symbol helpers --------
    def normalize_symbol(self, sym: str) -> str:
        """Override if the venue uses a different symbol (e.g., RELIANCE.NS -> RELIANCE)."""
        meta = self._instrument_meta.get(sym.upper()) or {}
        return (meta.get("venue_symbol") or sym).upper()

    def lot_size(self, sym: str) -> float:
        return float((self._instrument_meta.get(sym.upper()) or {}).get("lot", 1.0))

    def tick_size(self, sym: str) -> float:
        return float((self._instrument_meta.get(sym.upper()) or {}).get("tick", 0.01))

    # -------- rate limiting / retry --------
    def _rate_gate(self) -> None:
        if self._rate_limit_per_sec <= 0:
            return
        with self._rate_lock:
            now = time.time()
            min_gap = 1.0 / self._rate_limit_per_sec
            wait = max(0.0, self._last_call_ts + min_gap - now)
            if wait > 0:
                time.sleep(wait)
            self._last_call_ts = time.time()

    def _retry(self, fn: Callable, *, tries: int = 3, backoff: float = 0.25):
        last = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(backoff * (2 ** i))
        raise last  # type: ignore # re-raise last error

    # -------- lot/tick normalization --------
    def _normalize_qty(self, sym: str, qty: float) -> float:
        lot = self.lot_size(sym)
        if lot <= 0:
            return max(qty, 0.0)
        return math.floor(max(qty, 0.0) / lot) * lot

    def _normalize_px(self, sym: str, px: Optional[float]) -> Optional[float]:
        if px is None:
            return None
        tick = self.tick_size(sym)
        if tick <= 0:
            return float(px)
        return round(round(px / tick) * tick, 10)

    # -------- public API (stable) --------
    def place_order(self, order: Order) -> OrderAck:
        """
        Public entry: idempotent, normalized, rate-limited, retried -> _place().
        """
        self.ensure_connected()
        self._rate_gate()

        sym = order.symbol.upper()
        norm_qty = self._normalize_qty(sym, order.qty)
        if norm_qty <= 0:
            return OrderAck(ok=False, order_id=None, reason="QTY_ZERO_AFTER_NORMALIZE")

        order = Order(
            symbol=self.normalize_symbol(sym),
            side=order.side,
            qty=norm_qty,
            order_type=order.order_type,
            limit_price=self._normalize_px(sym, order.limit_price),
            tif=order.tif,
            client_order_id=order.client_order_id,
            strategy=order.strategy,
            meta=dict(order.meta),
        )

        # idempotency
        if order.client_order_id and order.client_order_id in self._idempotency:
            oid = self._idempotency[order.client_order_id]
            return OrderAck(ok=True, order_id=oid, reason="IDEMPOTENT_HIT")

        def _do():
            return self._place(order)

        ack: OrderAck = self._retry(_do)
        if ack.ok and order.client_order_id and ack.order_id:
            self._idempotency[order.client_order_id] = ack.order_id
        return ack

    def cancel_order(self, order_id: str) -> bool:
        self.ensure_connected()
        self._rate_gate()
        return self._retry(lambda: self._cancel(order_id))

    def replace_order(
        self,
        order_id: str,
        *,
        new_qty: Optional[float] = None,
        new_limit: Optional[float] = None,
        new_tif: Optional[TIF] = None,
    ) -> OrderAck:
        """
        Optional. Default implementation: cancel+place.
        Override if venue supports native replace.
        """
        self.ensure_connected()
        ok = self.cancel_order(order_id)
        if not ok:
            return OrderAck(ok=False, order_id=None, reason="REPLACE_CANCEL_FAILED")
        # Caller should supply the original order to rebuild — this is a minimal stub.

        return OrderAck(ok=False, order_id=None, reason="REPLACE_NOT_IMPLEMENTED")

    # -------- read-only --------
    def get_positions(self) -> List[Position]:
        self.ensure_connected()
        return self._positions()

    def get_account(self) -> Account:
        self.ensure_connected()
        return self._account()

    # -------- optional streaming --------
    def stream_prices(self, symbols: Iterable[str], on_tick: Callable[[Dict[str, Any]], None]) -> None:
        """
        Optional: stream venue prices and call on_tick({"symbol":..., "price":..., "ts_ms": ...})
        Default: not implemented.
        """
        raise NotImplementedError

    # -------- methods to implement in concrete subclasses --------
    def _place(self, order: Order) -> OrderAck:
        raise NotImplementedError

    def _cancel(self, order_id: str) -> bool:
        raise NotImplementedError

    def _positions(self) -> List[Position]:
        raise NotImplementedError

    def _account(self) -> Account:
        raise NotImplementedError


# ------------------------ Minimal Paper Broker --------------------------
class PaperBroker(BrokerBase):
    """
    Super-light paper broker:
      • Fills MARKET instantly at last price
      • Fills LIMIT if price is marketable (buy: last<=limit, sell: last>=limit)
      • Tracks positions & simple account
      • Not thread-safe; intended for local runs and tests
    """
    def __init__(self, *, currency: str = "USD", start_cash: float = 100_000.0):
        super().__init__(rate_limit_per_sec=0.0)
        self.acct = Account(equity=start_cash, cash=start_cash, buying_power=start_cash*2, currency=currency)
        self.last_px: Dict[str, float] = {}     # symbol -> last price
        self.pos: Dict[str, Position] = {}      # symbol -> Position
        self._oid = 0

    # ------- helpers -------
    def set_price(self, symbol: str, price: float) -> None:
        """Feed last price from your data loop."""
        if price and price > 0:
            self.last_px[symbol.upper()] = float(price)

    # ------- overrides -------
    def connect(self) -> None:
        self.connected = True
        log.info("PaperBroker connected.")

    def _next_oid(self) -> str:
        self._oid += 1
        return f"PB-{self._oid}"

    def _fillable(self, o: Order, last: Optional[float]) -> Tuple[bool, Optional[float]]:
        if o.order_type == OrderType.MARKET:
            return True, last
        if last is None:
            return False, None
        if o.side == Side.BUY and o.limit_price is not None and last <= o.limit_price:
            return True, min(o.limit_price, last)
        if o.side == Side.SELL and o.limit_price is not None and last >= o.limit_price:
            return True, max(o.limit_price, last)
        return False, None

    def _place(self, order: Order) -> OrderAck:
        oid = self._next_oid()
        sym = order.symbol.upper()
        last = self.last_px.get(sym)

        fillable, px = self._fillable(order, last)
        if not fillable or px is None:
            # emulate IOC reject for unmarketable IOC
            if order.tif == TIF.IOC:
                return OrderAck(ok=False, order_id=None, reason="IOC_UNFILLABLE", raw=asdict(order))
            # otherwise accept but not fill (not modeled here)
            return OrderAck(ok=True, order_id=oid, reason="ACCEPTED_NOFILL", raw=asdict(order))

        # fill fully
        qty = float(order.qty)
        cash_delta = -qty * px if order.side == Side.BUY else qty * px
        self.acct.cash += cash_delta
        self.acct.equity += 0.0  # MTM handled externally if you want
        self.acct.buying_power = max(self.acct.cash * 2, 0.0)

        pos = self.pos.get(sym) or Position(symbol=sym, qty=0.0, avg_price=px)
        if order.side == Side.BUY:
            new_qty = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + qty * px) / max(new_qty, 1e-9)
            pos.qty = new_qty
        else:
            pos.qty -= qty
            if pos.qty <= 0:
                pos.avg_price = px
        self.pos[sym] = pos

        ts_ms = int(time.time() * 1000)
        fill = Fill(order_id=oid, symbol=sym, side=order.side, qty=qty, price=px, ts_ms=ts_ms, strategy=order.strategy)
        log.debug(f"Paper fill: {fill}")
        return OrderAck(ok=True, order_id=oid, reason="FILLED", raw={"fill": fill.__dict__})

    def _cancel(self, order_id: str) -> bool:
        # Nothing resting in this simple model
        return True

    def _positions(self) -> List[Position]:
        return list(self.pos.values())

    def _account(self) -> Account:
        return self.acct