# backend/ai/agents/connectors/brokers/paperbroker.py
from __future__ import annotations

import os
import time
import uuid
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

# ============================================================
# Config (ENV)
# ============================================================
PB_LATENCY_MS = int(os.getenv("PB_LATENCY_MS", "20"))          # artificial latency per action
PB_SLIPPAGE_BPS = float(os.getenv("PB_SLIPPAGE_BPS", "0.0"))   # market order slippage vs mid
PB_DEFAULT_CASH = float(os.getenv("PB_DEFAULT_CASH", "1_000_000"))
PB_IMPACT_COEFF = float(os.getenv("PB_IMPACT_COEFF", "0.0"))   # simple impact (bps per 1% ADV proxy)
PB_FILL_ON_TOUCH = os.getenv("PB_FILL_ON_TOUCH", "true").lower() in ("1","true","yes")
PB_ACCOUNT = os.getenv("PB_ACCOUNT", "PAPER-001")

# ============================================================
# Types
# ============================================================
@dataclass
class Order:
    order_id: str
    symbol: str
    side: str                 # 'buy' | 'sell'
    qty: float
    remaining: float
    order_type: str           # 'market' | 'limit'
    limit_price: Optional[float]
    ts_ms: int
    status: str = "working"   # 'working' | 'filled' | 'cancelled' | 'replaced' | 'rejected'
    tag: Optional[str] = None
    venue: Optional[str] = None

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_cost: float = 0.0

@dataclass
class Trade:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    ts_ms: int

@dataclass
class Book:
    last: float = 100.0
    # resting limits
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(px, qty), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)

# ============================================================
# Core simulator
# ============================================================
class _PaperBroker:
    def __init__(self):
        self._lock = threading.RLock()
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._cash: float = PB_DEFAULT_CASH
        self._equity: float = PB_DEFAULT_CASH
        self._books: Dict[str, Book] = {}
        self._trades: List[Trade] = []

    # ------------- Market data priming -------------
    def _book(self, sym: str) -> Book:
        bk = self._books.get(sym)
        if not bk:
            bk = Book(last=100.0)
            self._books[sym] = bk
        return bk

    def set_price(self, symbol: str, last: float) -> None:
        """External feed can push last price; triggers 'fill-on-touch' matching."""
        with self._lock:
            bk = self._book(symbol)
            bk.last = float(last)
            # touch fill for resting limits
            if PB_FILL_ON_TOUCH:
                self._match_resting_on_touch(symbol, bk)

    def last_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._book(symbol).last

    # ------------- Trading API -------------
    def submit_order(self, symbol: str, side: str, qty: float, order_type: str = "market",
                     limit_price: Optional[float] = None, tag: Optional[str] = None,
                     venue: Optional[str] = None) -> str:
        ts = self._now_ms()
        if qty <= 0:
            raise ValueError("qty must be > 0")
        side = side.lower()
        order_type = order_type.lower()
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price required for limit order")

        oid = f"PB-{uuid.uuid4().hex[:10]}"
        ord = Order(order_id=oid, symbol=symbol, side=side, qty=float(qty), remaining=float(qty),
                    order_type=order_type, limit_price=float(limit_price) if limit_price else None,
                    ts_ms=ts, tag=tag, venue=venue)
        with self._lock:
            self._orders[oid] = ord

        self._sleep_latency()

        if order_type == "market":
            self._fill_market(ord)
        else:
            self._accept_limit(ord)

        return oid

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o.status not in ("working", "replaced"):
                return False
            o.status = "cancelled"
            o.remaining = 0.0
            # remove from resting book
            self._remove_from_book(o)
            return True

    def replace_order(self, order_id: str, *, new_qty: Optional[float] = None,
                      new_limit: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o.status not in ("working", "replaced"):
                return (False, None)

            if new_qty is not None:
                if new_qty < (o.qty - o.remaining):
                    # can't go below already filled qty
                    return (False, None)
                delta = new_qty - o.qty
                o.qty = float(new_qty)
                o.remaining = max(0.0, o.remaining + delta)

            if o.order_type == "limit" and new_limit is not None:
                o.limit_price = float(new_limit)
                # re-sort resting book
                self._remove_from_book(o)
                self._rest_limit(o)

            o.status = "replaced"
            return (True, order_id)

    def account_summary(self) -> Dict[str, Any]:
        with self._lock:
            net_liq = self._cash + sum(self._mtm(p) for p in self._positions.values()) # type: ignore
            return {"account": PB_ACCOUNT, "cash": self._cash, "net_liq": net_liq, "paper": True}

    def positions(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for p in self._positions.values():
                out.append({"account": PB_ACCOUNT, "symbol": p.symbol, "qty": p.qty, "avg_cost": p.avg_cost})
            return out

    # ------------- Fill models -------------
    def _fill_market(self, o: Order) -> None:
        with self._lock:
            last = self._book(o.symbol).last
            slip = (PB_SLIPPAGE_BPS / 1e4) * last
            px = last + (slip if o.side == "buy" else -slip)
            self._execute(o, o.remaining, px)
            o.status = "filled"
            o.remaining = 0.0

    def _accept_limit(self, o: Order) -> None:
        with self._lock:
            bk = self._book(o.symbol)
            # If price crosses now, fill immediately; else rest in book
            if self._is_crossed(o, bk.last):
                px = o.limit_price or bk.last
                self._execute(o, o.remaining, px)
                o.status = "filled"
                o.remaining = 0.0
            else:
                self._rest_limit(o)

    def _rest_limit(self, o: Order) -> None:
        bk = self._book(o.symbol)
        if o.side == "buy":
            bk.bids.append((o.limit_price, o.remaining)) # type: ignore
            bk.bids.sort(key=lambda x: (-x[0],))  # best price first
        else:
            bk.asks.append((o.limit_price, o.remaining)) # type: ignore
            bk.asks.sort(key=lambda x: (x[0],))   # best price first

    def _remove_from_book(self, o: Order) -> None:
        bk = self._book(o.symbol)
        lst = bk.bids if o.side == "buy" else bk.asks
        # remove matching (px,qty) entries until remaining cleared
        new = []
        rem = o.remaining
        for px, q in lst:
            if abs(px - (o.limit_price or px)) < 1e-9 and abs(q - rem) < 1e-9:
                rem = 0.0
                continue
            new.append((px, q))
        if o.side == "buy":
            bk.bids = new
        else:
            bk.asks = new

    def _is_crossed(self, o: Order, last: float) -> bool:
        if o.limit_price is None:
            return True
        if o.side == "buy":
            return last <= o.limit_price + 1e-12
        return last >= o.limit_price - 1e-12

    def _match_resting_on_touch(self, symbol: str, bk: Book) -> None:
        # Simple: if last trades through a level, fill all working orders at that price
        # (no partial queueing or time priority simulation here)
        to_fill: List[Tuple[Order, float]] = []
        for o in list(self._orders.values()):
            if o.symbol != symbol or o.status not in ("working", "replaced") or o.order_type != "limit":
                continue
            if self._is_crossed(o, bk.last) and o.remaining > 0:
                to_fill.append((o, o.remaining))
        for o, q in to_fill:
            self._execute(o, q, o.limit_price or bk.last)
            o.remaining -= q
            o.status = "filled"
            # prune from book
            self._remove_from_book(o)

    # ------------- Execution -------------
    def _execute(self, o: Order, qty: float, price: float) -> None:
        qty = float(qty)
        price = float(price)
        trade = Trade(order_id=o.order_id, symbol=o.symbol, side=o.side, qty=qty, price=price, ts_ms=self._now_ms())
        self._trades.append(trade)

        # update position & cash
        pos = self._positions.get(o.symbol)
        if not pos:
            pos = Position(symbol=o.symbol, qty=0.0, avg_cost=0.0)
            self._positions[o.symbol] = pos

        if o.side == "buy":
            new_qty = pos.qty + qty
            if new_qty != 0:
                pos.avg_cost = (pos.avg_cost * pos.qty + price * qty) / new_qty
            pos.qty = new_qty
            self._cash -= price * qty
        else:  # sell
            new_qty = pos.qty - qty
            # If crossing through zero, assume FIFO at avg_cost; P&L realized in cash
            pnl = (price - pos.avg_cost) * min(pos.qty, qty)
            self._cash += price * qty + pnl  # add notional + realized P&L component
            pos.qty = new_qty
            if pos.qty <= 0:
                pos.avg_cost = 0.0

        # optional market impact (very coarse)
        if PB_IMPACT_COEFF > 0:
            impact_bps = PB_IMPACT_COEFF * (qty / max(1.0, 100_000.0)) * 1e4
            bump = (impact_bps / 1e4) * price
            bk = self._book(o.symbol)
            bk.last = price + (bump if o.side == "buy" else -bump)

    # ------------- Utils -------------
    def _sleep_latency(self):
        if PB_LATENCY_MS > 0:
            time.sleep(PB_LATENCY_MS / 1000.0)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

# ============================================================
# Module-level singleton & helpers
# ============================================================
_client = _PaperBroker()

def connect() -> bool:
    """For parity with IBKR adapter; always 'connected'."""
    return True

def is_connected() -> bool:
    return True

def disconnect() -> None:
    pass

def submit_order(symbol: str, side: str, qty: float, *,
                 order_type: str = "market",
                 limit_price: Optional[float] = None,
                 tag: Optional[str] = None,
                 venue: Optional[str] = None) -> str:
    return _client.submit_order(symbol, side, qty, order_type, limit_price, tag, venue)

def cancel_order(order_id: str) -> bool:
    return _client.cancel_order(order_id)

def replace_order(order_id: str, *, new_qty: Optional[float] = None,
                  new_limit: Optional[float] = None) -> Tuple[bool, Optional[str]]:
    return _client.replace_order(order_id, new_qty=new_qty, new_limit=new_limit)

def account_summary() -> Dict[str, Any]:
    return _client.account_summary()

def positions() -> List[Dict[str, Any]]:
    return _client.positions()

def last_price(symbol: str, venue: Optional[str] = None) -> Optional[float]:
    return _client.last_price(symbol)

# --- extras useful for tests / wiring ---
def push_price(symbol: str, last: float) -> None:
    """Manually push a last price (e.g., from your quotes skill or a test)."""
    _client.set_price(symbol, last)

def _unsafe__dump_state() -> Dict[str, Any]:
    """Debug helper (not for production)."""
    return {
        "orders": {k: vars(v) for k, v in _client._orders.items()},
        "positions": {k: vars(v) for k, v in _client._positions.items()},
        "cash": _client._cash,
        "books": {k: {"last": v.last, "bids": v.bids, "asks": v.asks} for k, v in _client._books.items()},
    }

# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":  # pragma: no cover
    print("paper connect:", connect())
    print("account:", account_summary())
    push_price("AAPL", 100.00)
    oid1 = submit_order("AAPL", "buy", 100, order_type="market")
    print("buy mkt id:", oid1, "pos:", positions(), "cash:", account_summary()["cash"])
    # rest a limit and fill by touching price
    oid2 = submit_order("AAPL", "sell", 50, order_type="limit", limit_price=101.00)
    print("rested limit sell id:", oid2, "pos:", positions())
    print("touch @101 → should fill sell")
    push_price("AAPL", 101.00)
    print("pos:", positions(), "cash:", account_summary()["cash"])