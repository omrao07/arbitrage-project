# backend/market/limit_order_book.py
"""
Limit Order Book (price-time priority)
--------------------------------------
Features
- Sides: bid/ask, price-time priority queues
- Order types: LIMIT, MARKET
- TIF: GTC (default), IOC, FOK, POST_ONLY (maker-only)
- Actions: place, cancel, amend (qty/price), replace (atomic cancel+place)
- Matching: partial/complete fills, multi-counterparty sweeps
- Iceberg: display_qty < qty (simple, fixed peak; refresh = single use)
- Stats: best bid/ask, spread, mid, depth ladders, trades tape
- Streams (optional): publishes trades/updates if backend.bus.streams available

Example
-------
from backend.market.limit_order_book import LimitOrderBook, Order

lob = LimitOrderBook("NIFTY")
o1 = lob.place(Order(side="buy",  typ="limit", price=22000, qty=10))
o2 = lob.place(Order(side="sell", typ="limit", price=22010, qty=8))
o3 = lob.place(Order(side="buy",  typ="market", qty=5))  # matches @ 22010
print(lob.best_bid(), lob.best_ask(), lob.last_trades[-1])

CLI
----
python -m backend.market.limit_order_book --probe
"""

from __future__ import annotations

import bisect
import heapq
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

NOW_MS = lambda: int(time.time() * 1000)

# -------------------- Data types --------------------

@dataclass
class Order:
    side: str                 # "buy" or "sell"
    typ: str                  # "limit" or "market"
    qty: float
    price: Optional[float] = None  # required for limit
    tif: str = "GTC"          # "GTC" | "IOC" | "FOK" | "POST_ONLY"
    client_id: Optional[str] = None
    symbol: Optional[str] = None
    user_data: Dict[str, Any] = field(default_factory=dict)
    display_qty: Optional[float] = None   # for iceberg; if set < qty
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts_ms: int = field(default_factory=NOW_MS)

@dataclass
class Fill:
    taker_id: str
    maker_id: str
    price: float
    qty: float
    ts_ms: int

@dataclass
class BookLevel:
    price: float
    total_qty: float
    orders: List[str]  # order ids at this price (time priority)

# -------------------- Internal side --------------------

class _Side:
    """
    One side of the book. Maintains a price ladder and FIFO queues at each price.
    For bids, best is max price; for asks, best is min price.
    """
    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.levels: Dict[float, Deque[str]] = {}   # price -> order ids (FIFO)
        self.prices: List[float] = []               # sorted ascending
        self.total_qty_at_price: Dict[float, float] = {}
        self.order_lookup: Dict[str, Tuple[float, float]] = {}  # id -> (px, remaining_qty)

    def _cmp_ok(self, incoming_px: float, maker_px: float) -> bool:
        return (incoming_px >= maker_px) if self.is_bid else (incoming_px <= maker_px)

    def best_price(self) -> Optional[float]:
        if not self.prices:
            return None
        return self.prices[-1] if self.is_bid else self.prices[0]

    def add(self, oid: str, px: float, qty: float):
        if px not in self.levels:
            self.levels[px] = deque()
            bisect.insort(self.prices, px)
            self.total_qty_at_price[px] = 0.0
        self.levels[px].append(oid)
        self.total_qty_at_price[px] += qty
        self.order_lookup[oid] = (px, qty)

    def remove_order(self, oid: str) -> Optional[Tuple[float, float]]:
        rec = self.order_lookup.pop(oid, None)
        if not rec:
            return None
        px, qty = rec
        dq = self.levels.get(px)
        if dq:
            try:
                dq.remove(oid)
            except ValueError:
                pass
            if not dq:
                # remove empty level
                del self.levels[px]
                i = bisect.bisect_left(self.prices, px)
                if i < len(self.prices) and self.prices[i] == px:
                    self.prices.pop(i)
            self.total_qty_at_price[px] = max(0.0, self.total_qty_at_price.get(px, 0.0) - qty)
            if self.total_qty_at_price[px] <= 0:
                self.total_qty_at_price.pop(px, None)
        return (px, qty)

    def reduce(self, oid: str, new_qty: float) -> Optional[Tuple[float, float]]:
        rec = self.order_lookup.get(oid)
        if not rec:
            return None
        px, old_qty = rec
        delta = new_qty - old_qty
        self.order_lookup[oid] = (px, new_qty)
        self.total_qty_at_price[px] = max(0.0, self.total_qty_at_price.get(px, 0.0) + delta)
        return (px, new_qty)

    def pop_best_maker(self) -> Optional[str]:
        """Peek best price level and return first order id (maker) without removing it."""
        bp = self.best_price()
        if bp is None:
            return None
        dq = self.levels[bp]
        return dq[0] if dq else None

    def rotate_best(self):
        """Move the head order id to the end (used only if necessary)."""
        bp = self.best_price()
        if bp is None:
            return
        dq = self.levels[bp]
        if dq:
            dq.rotate(-1)

# -------------------- Order Book --------------------

class LimitOrderBook:
    def __init__(self, symbol: str, *, tape_size: int = 2000):
        self.symbol = symbol
        self.bids = _Side(is_bid=True)
        self.asks = _Side(is_bid=False)
        self.orders: Dict[str, Order] = {}
        self.last_trades: List[Fill] = []
        self.max_tape = tape_size

    # ---- Public API ----
    def place(self, o: Order) -> Dict[str, Any]:
        if o.typ not in ("limit", "market"):
            raise ValueError("typ must be 'limit' or 'market'")
        if o.typ == "limit" and (o.price is None):
            raise ValueError("limit order requires price")
        if o.qty <= 0:
            raise ValueError("qty must be > 0")
        o.symbol = o.symbol or self.symbol

        # POST_ONLY: reject if it would cross and take liquidity
        if o.typ == "limit" and o.tif.upper() == "POST_ONLY":
            crossing = self._would_cross(o)
            if crossing:
                return {"status": "rejected", "reason": "post_only_would_cross", "order_id": o.id}

        # FOK: if not fully fillable immediately, reject
        if o.tif.upper() == "FOK":
            if not self._check_fillable(o):
                return {"status": "rejected", "reason": "fok_not_fillable", "order_id": o.id}

        # IOC/MKT/LMT -> try to match
        fills, remaining = self._match(o)

        status = "filled" if remaining <= 1e-12 else ("accepted" if o.tif.upper()=="GTC" and o.typ=="limit" else "done")
        # if residual and eligible to rest, add to book
        if remaining > 1e-12 and o.typ == "limit" and o.tif.upper() == "GTC":
            self._rest(o, remaining)

        out = {
            "status": status,
            "order_id": o.id,
            "fills": [asdict(f) for f in fills],
            "remaining_qty": max(0.0, remaining)
        }
        return out

    def cancel(self, order_id: str) -> bool:
        o = self.orders.pop(order_id, None)
        if not o:
            # order may be GTC not found
            # try both sides
            side = self.bids if (order_id in self.bids.order_lookup) else (self.asks if (order_id in self.asks.order_lookup) else None)
            if side:
                side.remove_order(order_id)
            return False
        side = self.bids if o.side == "buy" else self.asks
        side.remove_order(order_id)
        return True

    def amend(self, order_id: str, *, new_qty: Optional[float] = None, new_price: Optional[float] = None) -> bool:
        """
        Amend qty and/or price. If price changes, we reinsert with a new timestamp (loses time priority).
        """
        o = self.orders.get(order_id)
        if not o:
            return False
        side = self.bids if o.side == "buy" else self.asks

        if new_price is not None and o.price != new_price:
            # remove and re-add
            side.remove_order(order_id)
            o.price = float(new_price)
            o.ts_ms = NOW_MS()
            side.add(o.id, o.price, new_qty if new_qty is not None else self._remaining_qty(order_id, side))
            return True

        if new_qty is not None:
            side.reduce(order_id, new_qty)
            return True
        return False

    def replace(self, order_id: str, new_order: Order) -> Dict[str, Any]:
        self.cancel(order_id)
        return self.place(new_order)

    # ---- Snapshots / queries ----
    def best_bid(self) -> Optional[Tuple[float, float]]:
        px = self.bids.best_price()
        return (px, self.bids.total_qty_at_price.get(px, 0.0)) if px is not None else None

    def best_ask(self) -> Optional[Tuple[float, float]]:
        px = self.asks.best_price()
        return (px, self.asks.total_qty_at_price.get(px, 0.0)) if px is not None else None

    def spread(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if not bb or not ba:
            return None
        return max(0.0, ba[0] - bb[0])

    def mid(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if not bb or not ba:
            return None
        return 0.5 * (bb[0] + ba[0])

    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        return {
            "bids": self._ladder(self.bids, levels, reverse=True),
            "asks": self._ladder(self.asks, levels, reverse=False),
        }

    # ---------------- internal mechanics ----------------

    def _ladder(self, side: _Side, levels: int, reverse: bool) -> List[Tuple[float, float]]:
        pxs = reversed(side.prices) if reverse else iter(side.prices)
        out: List[Tuple[float, float]] = []
        for px in pxs:
            out.append((px, side.total_qty_at_price.get(px, 0.0)))
            if len(out) >= levels:
                break
        return out

    def _would_cross(self, o: Order) -> bool:
        if o.typ != "limit":
            return False
        if o.side == "buy":
            ask = self.asks.best_price()
            return ask is not None and o.price >= ask # type: ignore
        else:
            bid = self.bids.best_price()
            return bid is not None and o.price <= bid # type: ignore

    def _check_fillable(self, o: Order) -> bool:
        """For FOK: is full qty fillable *immediately*?"""
        need = o.qty
        if o.side == "buy":
            # consume asks up to price (limit) or infinity (market)
            for px in list(self.asks.prices):
                if o.typ == "limit" and px > o.price:  # type: ignore # worse than limit
                    break
                need -= self.asks.total_qty_at_price.get(px, 0.0)
                if need <= 1e-12:
                    return True
            return False
        else:
            for px in reversed(self.bids.prices):
                if o.typ == "limit" and px < o.price: # type: ignore
                    break
                need -= self.bids.total_qty_at_price.get(px, 0.0)
                if need <= 1e-12:
                    return True
            return False

    def _match(self, o: Order) -> Tuple[List[Fill], float]:
        fills: List[Fill] = []
        remaining = o.qty

        def take(level_side: _Side, contra_side: _Side, px_ok) -> None:
            nonlocal remaining, fills
            while remaining > 1e-12:
                best_px = contra_side.best_price()
                if best_px is None or not px_ok(best_px):
                    break

                dq = contra_side.levels.get(best_px)
                if not dq:
                    # cleanup empty level
                    contra_side.levels.pop(best_px, None)
                    i = bisect.bisect_left(contra_side.prices, best_px)
                    if i < len(contra_side.prices) and contra_side.prices[i] == best_px:
                        contra_side.prices.pop(i)
                    continue

                maker_id = dq[0]  # head of FIFO
                mp, mqty = contra_side.order_lookup.get(maker_id, (best_px, 0.0))

                trade_qty = min(remaining, mqty)
                trade_px = best_px  # price at maker (conventional)
                fills.append(Fill(
                    taker_id=o.id, maker_id=maker_id, price=trade_px, qty=trade_qty, ts_ms=NOW_MS()
                ))

                # update maker
                new_mqty = mqty - trade_qty
                contra_side.order_lookup[maker_id] = (mp, new_mqty)
                contra_side.total_qty_at_price[best_px] = max(0.0, contra_side.total_qty_at_price.get(best_px, 0.0) - trade_qty)
                if new_mqty <= 1e-12:
                    # remove from level
                    dq.popleft()
                    contra_side.order_lookup.pop(maker_id, None)
                    if not dq:
                        contra_side.levels.pop(best_px, None)
                        i = bisect.bisect_left(contra_side.prices, best_px)
                        if i < len(contra_side.prices) and contra_side.prices[i] == best_px:
                            contra_side.prices.pop(i)

                remaining -= trade_qty

        if o.side == "buy":
            # match vs asks at best ask and up to limit
            px_ok = (lambda px: True) if o.typ == "market" else (lambda px: px <= o.price)
            take(self.bids, self.asks, px_ok)
        else:
            px_ok = (lambda px: True) if o.typ == "market" else (lambda px: px >= o.price)
            take(self.asks, self.bids, px_ok)

        # Publish fills to tape
        for f in fills:
            self._record_trade(f)

        # IOC: discard remainder
        if o.tif.upper() == "IOC":
            remaining = 0.0

        return (fills, remaining)

    def _rest(self, o: Order, qty: float) -> None:
        """Rest (add) residual limit order to the book."""
        side = self.bids if o.side == "buy" else self.asks
        # iceberg handling (simple one-shot: display rests, hidden remainder not refreshed)
        if o.display_qty is not None and o.display_qty > 0 and o.display_qty < qty:
            disp = o.display_qty
            hidden = qty - disp
            # Rest only the display part; track hidden in user_data if you want to refresh externally
            side.add(o.id, o.price, disp) # type: ignore
            self.orders[o.id] = Order(**{**asdict(o), "qty": disp})
            # Hidden remainder can be returned to caller or tracked elsewhere; here we stash it:
            self.orders[o.id].user_data["hidden_remainder"] = hidden
        else:
            side.add(o.id, o.price, qty) # type: ignore
            self.orders[o.id] = Order(**{**asdict(o), "qty": qty})

    def _record_trade(self, f: Fill) -> None:
        self.last_trades.append(f)
        if len(self.last_trades) > self.max_tape:
            self.last_trades.pop(0)
        if publish_stream:
            try:
                publish_stream("tape.trades", {
                    "ts_ms": f.ts_ms, "symbol": self.symbol, "price": f.price, "qty": f.qty,
                    "taker_id": f.taker_id, "maker_id": f.maker_id
                })
            except Exception:
                pass

    def _remaining_qty(self, oid: str, side: _Side) -> float:
        rec = side.order_lookup.get(oid)
        return rec[1] if rec else 0.0


# -------------------- CLI probe --------------------

def _probe():
    book = LimitOrderBook("TEST")

    # seed
    book.place(Order(side="buy",  typ="limit", price=99.0, qty=10))
    book.place(Order(side="buy",  typ="limit", price=100.0, qty=5))
    book.place(Order(side="sell", typ="limit", price=101.0, qty=7))
    book.place(Order(side="sell", typ="limit", price=102.0, qty=9))

    print("BBO before:", book.best_bid(), book.best_ask(), "mid:", book.mid(), "spread:", book.spread())
    print("Depth:", book.depth(3))

    # taker hits
    res = book.place(Order(side="buy", typ="market", qty=8))
    print("MKT buy -> fills:", res["fills"])
    print("BBO after:", book.best_bid(), book.best_ask(), "depth:", book.depth(3))

    # post-only reject
    rej = book.place(Order(side="buy", typ="limit", price=103.0, qty=1, tif="POST_ONLY"))
    print("POST_ONLY:", rej)

    # FOK
    fok_ok = book.place(Order(side="sell", typ="limit", price=99.0, qty=3, tif="FOK"))
    fok_no = book.place(Order(side="sell", typ="limit", price=104.0, qty=999, tif="FOK"))
    print("FOK ok:", fok_ok["status"], "FOK no:", fok_no["status"])

    # Iceberg
    ice = book.place(Order(side="sell", typ="limit", price=103.0, qty=20, display_qty=5))
    print("Iceberg rest:", ice["status"], "hidden:", book.orders[ice["order_id"]].user_data.get("hidden_remainder"))

def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Limit Order Book")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        _probe()
    else:
        _probe()

if __name__ == "__main__":
    main()