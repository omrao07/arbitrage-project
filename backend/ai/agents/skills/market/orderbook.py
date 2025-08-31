# backend/market/orderbook.py
from __future__ import annotations

import bisect
import itertools
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Deque, Dict, List, Optional, Tuple, Literal, Any

Side = Literal["buy", "sell"]

def now_ms() -> int: return int(time.time() * 1000)

# ---------- Data models ----------

@dataclass
class Order:
    order_id: str
    side: Side
    price: Optional[float]  # None for market orders (ephemeral)
    qty: float
    ts_ms: int = field(default_factory=now_ms)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Fill:
    ts_ms: int
    taker_order_id: str
    maker_order_id: str
    price: float
    qty: float
    side: Side              # taker side
    venue: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BBO:
    ts_ms: int
    bid_px: Optional[float]
    bid_sz: float
    ask_px: Optional[float]
    ask_sz: float

# ---------- Price ladder helpers ----------

class _PriceLevels:
    """
    One side of the book (either bids or asks).
    Maintains:
      - price index (sorted list)
      - per-price FIFO queue of L3 orders
    """
    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.prices: List[float] = []                    # sorted ascending
        self.levels: Dict[float, Deque[Order]] = {}      # price -> FIFO queue

    def _keypos(self, px: float) -> int:
        # ascending sorted list; for bids we still keep ascending, but "best" is the rightmost
        return bisect.bisect_left(self.prices, px)

    def top_price(self) -> Optional[float]:
        if not self.prices:
            return None
        return self.prices[-1] if self.is_bid else self.prices[0]

    def top_size(self) -> float:
        px = self.top_price()
        if px is None: return 0.0
        q = self.levels.get(px)
        return sum(o.qty for o in q) if q else 0.0

    def add(self, order: Order) -> None:
        assert order.price is not None
        px = float(order.price)
        if px not in self.levels:
            pos = self._keypos(px)
            self.prices.insert(pos, px)
            self.levels[px] = deque()
        self.levels[px].append(order)

    def pop_best(self) -> Optional[Order]:
        px = self.top_price()
        if px is None:
            return None
        q = self.levels[px]
        o = q.popleft()
        if not q:
            del self.levels[px]
            # remove px from price index
            pos = self._keypos(px)
            # guard in case of float compare mismatch
            if pos < len(self.prices) and self.prices[pos] == px:
                self.prices.pop(pos)
            else:
                # fallback linear remove
                try: self.prices.remove(px)
                except ValueError: pass
        return o

    def peek_best(self) -> Optional[Order]:
        px = self.top_price()
        if px is None: return None
        q = self.levels[px]
        return q[0] if q else None

    def remove(self, order_id: str) -> Optional[Order]:
        # O(level depth) scan; acceptable for small per-price queues
        for px in list(self.prices):
            q = self.levels.get(px)
            if not q: continue
            for i, o in enumerate(q):
                if o.order_id == order_id:
                    out = q[i]
                    del q[i]
                    if not q:
                        del self.levels[px]
                        try: self.prices.remove(px)
                        except ValueError: pass
                    return out
        return None

    def total_depth(self) -> int:
        return sum(len(q) for q in self.levels.values())

    def l2(self, depth: int) -> List[Tuple[float, float]]:
        """
        Returns top 'depth' levels as (price, size) sorted best->worse for this side.
        """
        out: List[Tuple[float, float]] = []
        if self.is_bid:
            it = reversed(self.prices)
        else:
            it = iter(self.prices)
        for px in it:
            q = self.levels[px]
            sz = sum(o.qty for o in q)
            out.append((px, sz))
            if len(out) >= depth: break
        return out

# ---------- Main order book ----------

class OrderBook:
    """
    In-memory L3 order book with matching.
    - Submit/cancel/replace user orders (for simulation or internal venue).
    - Or mirror external venues with set_level / apply_delta.
    - Emits trades and BBO updates via callbacks.
    """
    def __init__(
        self,
        symbol: str,
        *,
        on_trade: Optional[Callable[[Fill], None]] = None,
        on_bbo: Optional[Callable[[BBO], None]] = None,
        on_book_change: Optional[Callable[[], None]] = None,
        venue: Optional[str] = None,
    ):
        self.symbol = symbol
        self.venue = venue
        self.bids = _PriceLevels(is_bid=True)
        self.asks = _PriceLevels(is_bid=False)
        self._orders: Dict[str, Tuple[Side, float]] = {}  # id -> (side, qty_remaining)
        self._on_trade = on_trade
        self._on_bbo = on_bbo
        self._on_book_change = on_book_change
        self._last_bbo: Optional[BBO] = None
        self._seq = itertools.count(1)

    # --------- Public: matching engine ---------

    def submit_limit(self, side: Side, price: float, qty: float, order_id: Optional[str] = None, meta: Optional[Dict[str,Any]]=None) -> List[Fill]:
        """
        Add a limit order. Returns list of Fills against opposite side (if crossed).
        """
        oid = order_id or f"{self.symbol}-{next(self._seq)}"
        order = Order(order_id=oid, side=side, price=float(price), qty=float(qty), meta=meta or {})
        fills = self._match(order)
        if order.qty > 1e-12:  # residual rests
            self._add(order)
        self._emit_bbo_if_changed()
        self._fire_book_change()
        return fills

    def submit_market(self, side: Side, qty: float, order_id: Optional[str] = None, meta: Optional[Dict[str,Any]]=None) -> List[Fill]:
        """
        Market order consumes opposite book until qty filled or book empty.
        """
        oid = order_id or f"{self.symbol}-M{next(self._seq)}"
        order = Order(order_id=oid, side=side, price=None, qty=float(qty), meta=meta or {})
        fills = self._match(order, market=True)
        self._emit_bbo_if_changed()
        self._fire_book_change()
        return fills

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self.bids.remove(order_id) or self.asks.remove(order_id)
        if o:
            self._orders.pop(order_id, None)
            self._emit_bbo_if_changed()
            self._fire_book_change()
        return o

    def replace(self, order_id: str, new_price: Optional[float] = None, new_qty: Optional[float] = None) -> Optional[str]:
        """
        Cancel/replace: remove and re-add with new params (keeps new time priority).
        Returns new order_id if replaced.
        """
        existing = self.cancel(order_id)
        if not existing:
            return None
        px = existing.price if new_price is None else float(new_price)
        q = existing.qty if new_qty is None else float(new_qty)
        new_id = f"{self.symbol}-R{next(self._seq)}"
        self.submit_limit(existing.side, px, q, order_id=new_id, meta=existing.meta) # type: ignore
        return new_id

    # --------- Public: snapshots / metrics ---------

    def best_bid(self) -> Tuple[Optional[float], float]:
        return self.bids.top_price(), self.bids.top_size()

    def best_ask(self) -> Tuple[Optional[float], float]:
        return self.asks.top_price(), self.asks.top_size()

    def spread(self) -> Optional[float]:
        bp, _ = self.best_bid()
        ap, _ = self.best_ask()
        if bp is None or ap is None: return None
        return max(0.0, ap - bp)

    def mid(self) -> Optional[float]:
        bp, _ = self.best_bid()
        ap, _ = self.best_ask()
        if bp is None or ap is None: return None
        return 0.5 * (bp + ap)

    def l2(self, depth: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        return {"bids": self.bids.l2(depth), "asks": self.asks.l2(depth)}

    def size(self) -> int:
        return self.bids.total_depth() + self.asks.total_depth()

    # --------- Public: mirror external books (optional) ---------

    def set_level(self, side: Side, price: float, size: float) -> None:
        """
        L2 utility: force a specific (price -> aggregated size) on one side.
        Use for exchange snapshots/deltas where per-order info is unavailable.
        Implementation: clear that price then insert a synthetic L3 order with given size.
        """
        sidebook = self.bids if side == "buy" else self.asks
        # remove all orders at price
        px = float(price)
        q = sidebook.levels.get(px)
        if q:
            # remove every order id from registry
            for o in list(q):
                self._orders.pop(o.order_id, None)
            sidebook.levels.pop(px, None)
            try: sidebook.prices.remove(px)
            except ValueError: pass
        if size > 0:
            oid = f"{self.symbol}-SYN{next(self._seq)}"
            sidebook.add(Order(order_id=oid, side=side, price=px, qty=float(size)))
            self._orders[oid] = (side, float(size))
        self._emit_bbo_if_changed()
        self._fire_book_change()

    # --------- Internals ---------

    def _add(self, order: Order) -> None:
        sidebook = self.bids if order.side == "buy" else self.asks
        sidebook.add(order)
        self._orders[order.order_id] = (order.side, order.qty)

    def _crossed(self, taker: Order) -> bool:
        if taker.side == "buy":
            ap = self.asks.top_price()
            return ap is not None and (taker.price is None or taker.price >= ap)
        else:
            bp = self.bids.top_price()
            return bp is not None and (taker.price is None or taker.price <= bp)

    def _match(self, taker: Order, market: bool = False) -> List[Fill]:
        fills: List[Fill] = []
        while taker.qty > 1e-12 and self._crossed(taker):
            maker = self.asks.peek_best() if taker.side == "buy" else self.bids.peek_best()
            if maker is None:
                break
            trade_px = maker.price if maker.price is not None else (self.asks.top_price() if taker.side=="buy" else self.bids.top_price())
            if trade_px is None:
                break
            # respect taker limit if provided
            if taker.price is not None:
                if taker.side == "buy" and trade_px > taker.price: break
                if taker.side == "sell" and trade_px < taker.price: break

            maker = (self.asks.pop_best() if taker.side == "buy" else self.bids.pop_best())  # take from top
            traded = min(taker.qty, maker.qty) # type: ignore
            taker.qty -= traded
            maker.qty -= traded # type: ignore

            # If maker not fully consumed, put remainder back at top (keeps FIFO)
            if maker.qty > 1e-12: # type: ignore
                (self.asks if taker.side == "buy" else self.bids).levels[maker.price].appendleft(maker) # type: ignore

            f = Fill(
                ts_ms=now_ms(),
                taker_order_id=taker.order_id,
                maker_order_id=maker.order_id, # type: ignore
                price=float(trade_px),
                qty=float(traded),
                side=taker.side,
                venue=self.venue or taker.meta.get("venue"),
                meta={"symbol": self.symbol}
            )
            fills.append(f)
            if self._on_trade:
                try: self._on_trade(f)
                except Exception: pass

        return fills

    def _emit_bbo_if_changed(self) -> None:
        if not self._on_bbo:
            return
        bp, bs = self.best_bid()
        ap, asz = self.best_ask()
        bbo = BBO(ts_ms=now_ms(), bid_px=bp, bid_sz=bs or 0.0, ask_px=ap, ask_sz=asz or 0.0)
        if self._last_bbo is None or (
            bbo.bid_px != self._last_bbo.bid_px or bbo.ask_px != self._last_bbo.ask_px or
            bbo.bid_sz != self._last_bbo.bid_sz or bbo.ask_sz != self._last_bbo.ask_sz
        ):
            self._last_bbo = bbo
            try: self._on_bbo(bbo)
            except Exception: pass

    def _fire_book_change(self) -> None:
        if self._on_book_change:
            try: self._on_book_change()
            except Exception: pass


# ---------- Tiny smoke test ----------
if __name__ == "__main__":  # pragma: no cover
    trades: List[Fill] = []
    def on_trade(f: Fill):
        trades.append(f)
        print("TRADE", asdict(f))
    def on_bbo(b: BBO):
        print("BBO", asdict(b))

    ob = OrderBook("DEMO", on_trade=on_trade, on_bbo=on_bbo)

    # Add a passive ask then cross it with a market buy
    ob.submit_limit("sell", price=100.5, qty=50, order_id="ask1")
    ob.submit_limit("sell", price=100.6, qty=30, order_id="ask2")
    ob.submit_limit("buy", price=100.3, qty=25, order_id="bid1")

    print("L2", ob.l2(5))
    print("mid", ob.mid(), "spread", ob.spread())

    ob.submit_market("buy", qty=60, order_id="taker1")
    print("trades:", len(trades))
    print("L2", ob.l2(5))