# backend/microstructure/queue_position.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple, Literal
from collections import deque
import time
import math

Side = Literal["buy", "sell"]


@dataclass
class LevelState:
    price: float = 0.0
    # Total visible resting size at this price (ex-self). Updated from L2.
    visible_qty: float = 0.0
    # Our own resting quantity at this price (live working qty that is in queue).
    self_qty: float = 0.0
    # Estimated queue ahead of us (ex-self) in the FIFO queue at this price.
    queue_ahead: float = 0.0
    # Rolling trade-through rate (qty/sec) that removes queue_ahead at this price (and our qty).
    drain_rate_qps: float = 0.0
    # Last update timestamp (ms)
    ts_ms: int = 0


@dataclass
class OrderRef:
    order_id: str
    side: Side
    price: float
    orig_qty: float
    live_qty: float
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class QueueTracker:
    """
    Queue position tracker under price-time priority (FIFO).
    Subscribe it to:
      • L2 book updates: on_l2(symbol, bids[(p, q), ...], asks[(p, q), ...])
      • Prints/trades: on_trade(symbol, price, size, aggressor_side)
      • Your OMS acks/fills: on_order_ack / on_replace_ack / on_cancel_ack / on_fill
    You then query:
      • queue_ahead(symbol, side) → float
      • eta_to_first_fill(symbol, side) / fill_prob_horizon(symbol, side, horizon_s)
    Assumptions:
      • Price-time priority, visible queues only (icebergs make it conservative).
      • We’re always posted at **best price** for the given side (top-of-book). If you work deeper,
        pass the correct price and we’ll anchor to that level.
    """

    def __init__(self):
        # state per (symbol, side, price)
        self.levels: Dict[Tuple[str, Side, float], LevelState] = {}
        # our active order per (symbol, side) → OrderRef
        self.working: Dict[Tuple[str, Side], OrderRef] = {}
        # recent prints ring per symbol for short-horizon drain rate
        self._prints: Dict[str, Deque[Tuple[int, float, Side]]] = {}

        # config
        self._drain_window_ms = 3000  # compute trade-through drain over last 3s
        self._min_qps_for_eta = 1e-6

    # -------------- helpers --------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _lvl(self, sym: str, side: Side, price: float) -> LevelState:
        key = (sym, side, price)
        if key not in self.levels:
            self.levels[key] = LevelState(price=price)
        return self.levels[key]

    def _prints_buf(self, sym: str) -> Deque[Tuple[int, float, Side]]:
        if sym not in self._prints:
            self._prints[sym] = deque(maxlen=512)
        return self._prints[sym]

    # -------------- market data ingest --------------
    def on_l2(self, sym: str,
              bids: Optional[Tuple[Tuple[float, float], ...]] = None,
              asks: Optional[Tuple[Tuple[float, float], ...]] = None) -> None:
        """
        L2 snapshot or incremental (pass the current top level tuples you have).
        Each tuple: (price, size). Provide at least the best level for each side you care about.
        """
        ts = self._now_ms()
        if bids:
            bp, bq = bids[0]
            # If we have a working buy at bp, update visible and adjust queue_ahead conservatively.
            ref = self.working.get((sym, "buy"))
            lvl = self._lvl(sym, "buy", bp)
            lvl.ts_ms = ts
            lvl.price = bp
            # Visible resting EX-self (best guess)
            ex_self = max(0.0, bq - (ref.live_qty if ref and abs(ref.price - bp) < 1e-12 else 0.0))
            # If price moved, reset queue_ahead to ex_self
            if abs(lvl.visible_qty - ex_self) > 1e-9 and lvl.price == bp:
                # If size decreased, reduce queue_ahead, but never below 0
                if ex_self < lvl.visible_qty:
                    lvl.queue_ahead = max(0.0, min(lvl.queue_ahead, ex_self))
                else:
                    # size increased ahead of us → conservatively add to queue ahead
                    growth = ex_self - lvl.visible_qty
                    lvl.queue_ahead += max(0.0, growth)
            lvl.visible_qty = ex_self

        if asks:
            ap, aq = asks[0]
            ref = self.working.get((sym, "sell"))
            lvl = self._lvl(sym, "sell", ap)
            lvl.ts_ms = ts
            lvl.price = ap
            ex_self = max(0.0, aq - (ref.live_qty if ref and abs(ref.price - ap) < 1e-12 else 0.0))
            if abs(lvl.visible_qty - ex_self) > 1e-9 and lvl.price == ap:
                if ex_self < lvl.visible_qty:
                    lvl.queue_ahead = max(0.0, min(lvl.queue_ahead, ex_self))
                else:
                    growth = ex_self - lvl.visible_qty
                    lvl.queue_ahead += max(0.0, growth)
            lvl.visible_qty = ex_self

    def on_trade(self, sym: str, price: float, size: float, aggressor_side: Optional[Side]) -> None:
        """
        Prints with aggressor side if available:
          aggressor_side == "buy"  → trade likely consumed ASK
          aggressor_side == "sell" → trade likely consumed BID
        We drain queue_ahead at the touched side if price == best price.
        """
        ts = self._now_ms()
        buf = self._prints_buf(sym)
        if size <= 0:
            return
        side = aggressor_side or "buy"  # if unknown, assume buy→consume ask (conservative for buys)
        buf.append((ts, float(size), side))

        # prune
        cutoff = ts - self._drain_window_ms
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        # compute qps per side at current best
        buy_q = sum(sz for t, sz, s in buf if s == "buy" and t >= cutoff)
        sell_q = sum(sz for t, sz, s in buf if s == "sell" and t >= cutoff)
        buy_qps = buy_q / max(1.0, self._drain_window_ms / 1000.0)
        sell_qps = sell_q / max(1.0, self._drain_window_ms / 1000.0)

        # update drain rate + queue_ahead for our working orders if resting at touch
        for (k_sym, k_side), ref in list(self.working.items()):
            if k_sym != sym or ref.live_qty <= 0:
                continue
            lvl = self._lvl(sym, k_side, ref.price)
            if k_side == "buy":
                # sells consume bids → aggressor_side == "sell"
                lvl.drain_rate_qps = sell_qps
                # when touch traded and price == our level, reduce queue ahead
                if side == "sell":
                    drained = float(size)
                    # allocate only to queue_ahead first; if queue ahead becomes 0, fills will arrive via on_fill()
                    lvl.queue_ahead = max(0.0, lvl.queue_ahead - drained)
            else:
                lvl.drain_rate_qps = buy_qps
                if side == "buy":
                    drained = float(size)
                    lvl.queue_ahead = max(0.0, lvl.queue_ahead - drained)

    # -------------- OMS events --------------
    def on_order_ack(self, sym: str, order_id: str, side: Side, price: float, qty: float) -> None:
        """
        Called once an order is accepted & live on venue at given price.
        We assume we joined the END of the FIFO queue at that level → queue_ahead = current visible at level.
        """
        ref = OrderRef(order_id=order_id, side=side, price=float(price), orig_qty=float(qty), live_qty=float(qty))
        self.working[(sym, side)] = ref
        lvl = self._lvl(sym, side, ref.price)
        # Initialize queue_ahead as best guess = full visible ex-self
        ex_self = max(0.0, lvl.visible_qty)
        lvl.self_qty = ref.live_qty
        lvl.queue_ahead = ex_self

    def on_replace_ack(self, sym: str, order_id: str, new_price: float, new_qty: Optional[float] = None) -> None:
        """
        We lost queue priority when we price-improved or moved; reset queue_ahead at new price.
        """
        # find working by (sym, side)
        side = None
        for (k_sym, k_side), ref in self.working.items():
            if k_sym == sym and ref.order_id == order_id:
                side = k_side
                break
        if side is None:
            return
        ref = self.working[(sym, side)]
        ref.price = float(new_price)
        if new_qty is not None:
            ref.live_qty = float(new_qty)
        lvl = self._lvl(sym, side, ref.price)
        lvl.self_qty = ref.live_qty
        lvl.queue_ahead = max(0.0, lvl.visible_qty)  # rejoin at tail

    def on_cancel_ack(self, sym: str, order_id: str) -> None:
        # zero out our self qty at that level
        for (k_sym, k_side), ref in list(self.working.items()):
            if k_sym == sym and ref.order_id == order_id:
                lvl = self._lvl(sym, k_side, ref.price)
                lvl.self_qty = 0.0
                ref.live_qty = 0.0
                del self.working[(k_sym, k_side)]
                break

    def on_fill(self, sym: str, order_id: str, fill_qty: float, fill_price: float) -> None:
        """
        Venue reported fills. Reduce our live_qty; queue_ahead should already be near 0 when first fill happens.
        """
        for (k_sym, k_side), ref in list(self.working.items()):
            if k_sym == sym and ref.order_id == order_id:
                ref.live_qty = max(0.0, ref.live_qty - float(fill_qty))
                lvl = self._lvl(sym, k_side, ref.price)
                lvl.self_qty = ref.live_qty
                # If fully filled, remove
                if ref.live_qty <= 1e-9:
                    del self.working[(k_sym, k_side)]
                break

    # -------------- Queries --------------
    def queue_ahead(self, sym: str, side: Side) -> Optional[float]:
        ref = self.working.get((sym, side))
        if not ref:
            return None
        lvl = self._lvl(sym, side, ref.price)
        return max(0.0, lvl.queue_ahead)

    def eta_to_first_fill(self, sym: str, side: Side) -> Optional[float]:
        """
        Estimated seconds until our FIRST fill (i.e., queue ahead drains to ~0).
        """
        ref = self.working.get((sym, side))
        if not ref:
            return None
        lvl = self._lvl(sym, side, ref.price)
        qps = max(self._min_qps_for_eta, lvl.drain_rate_qps)
        return float(lvl.queue_ahead) / qps

    def fill_prob_horizon(self, sym: str, side: Side, horizon_s: float) -> Optional[float]:
        """
        Probability that we START getting filled within horizon (Poisson-like with rate = qps/visible?).
        We use a simple exponential survival proxy based on queue-ahead and drain rate.
        """
        ref = self.working.get((sym, side))
        if not ref:
            return None
        lvl = self._lvl(sym, side, ref.price)
        qps = max(self._min_qps_for_eta, lvl.drain_rate_qps)
        # Expected drained in horizon
        drained = qps * max(0.0, horizon_s)
        # If drained exceeds queue_ahead, probability ~ 1 - exp(-(drained-qa)/scale)
        qa = max(0.0, lvl.queue_ahead)
        if drained <= 0:
            return 0.0
        # heuristic scale = max(1, visible_qty) to soften
        scale = max(1.0, lvl.visible_qty)
        surplus = max(0.0, drained - qa)
        p = 1.0 - math.exp(-surplus / scale)
        return max(0.0, min(1.0, p))

    # -------------- Convenience for strategies --------------
    def post_or_improve_hint(self, sym: str, side: Side) -> str:
        """
        Simple hint:
          - If queue_ahead too large → consider price-improve (lose priority but faster fill).
          - If drain_rate strong and qa small → keep.
        """
        ref = self.working.get((sym, side))
        if not ref:
            return "no_order"
        lvl = self._lvl(sym, side, ref.price)
        qa = lvl.queue_ahead
        qps = lvl.drain_rate_qps
        if qa <= 0:
            return "expect_fill"
        # thresholds can be tuned per symbol
        if qa > 5 * ref.live_qty and qps < 0.5 * ref.live_qty:
            return "consider_improve"
        if qps >= ref.live_qty:
            return "keep"
        return "keep"


# --------------------- Example usage ---------------------
if __name__ == "__main__":
    qt = QueueTracker()
    sym = "AAPL"

    # seed book
    qt.on_l2(sym, bids=((100.00, 5000.0),), asks=((100.02, 6000.0),))
    # we post a buy at bid
    qt.on_order_ack(sym, order_id="ord1", side="buy", price=100.00, qty=1000.0)
    print("Queue ahead (init):", qt.queue_ahead(sym, "buy"))

    # prints hitting the bid
    qt.on_trade(sym, price=100.00, size=1200.0, aggressor_side="sell")
    print("QPS drain:", qt._lvl(sym, "buy", 100.00).drain_rate_qps)
    print("ETA first fill (s):", round(qt.eta_to_first_fill(sym, "buy") or 0.0, 3))

    # simulate fills from OMS
    qt.on_fill(sym, order_id="ord1", fill_qty=400.0, fill_price=100.00)
    print("Queue ahead (post fill):", qt.queue_ahead(sym, "buy"))
    print("Hint:", qt.post_or_improve_hint(sym, "buy"))