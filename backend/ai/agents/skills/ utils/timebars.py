# backend/market/timebars.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

# ---------------- Common types ----------------

@dataclass
class Trade:
    symbol: str
    ts_ms: int
    price: float
    size: float

@dataclass
class Bar:
    symbol: str
    start_ms: int
    end_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

# --------------- Utilities --------------------

def floor_to_bucket(ts_ms: int, bucket_ms: int) -> int:
    if bucket_ms <= 0:
        raise ValueError("bucket_ms must be > 0")
    return (ts_ms // bucket_ms) * bucket_ms

def now_ms() -> int:
    return int(time.time() * 1000)

# --------------- Base Aggregator --------------

class _BaseAggregator:
    """
    Minimal incremental aggregator base: handles lifecycle & callbacks.
    Subclasses set rollover condition and update logic.
    """
    def __init__(
        self,
        symbol: str,
        on_bar: Optional[Callable[[Bar], None]] = None,
        *,
        carry_partial_on_flush: bool = True
    ):
        self.symbol = symbol
        self.on_bar = on_bar
        self.carry_partial_on_flush = carry_partial_on_flush
        self._cur: Optional[Bar] = None
        self._finished: List[Bar] = []

    # ----- public -----
    def add_trade(self, t: Trade) -> None:
        if t.symbol != self.symbol:
            return
        if self._cur is None:
            self._cur = self._new_bar_from_trade(t)
            self._after_first_trade(t)
            return
        # rollover?
        if self._should_rollover(self._cur, t):
            self._finish_current_bar(before_trade=t)
        # update
        self._update_with_trade(self._cur, t)

    def flush(self, *, force_close: bool = True, end_ts_ms: Optional[int] = None) -> None:
        """
        Close the current bar (if any). If force_close=False, keep it open.
        """
        if not self._cur:
            return
        if force_close:
            if end_ts_ms is None:
                end_ts_ms = now_ms()
            self._cur.end_ms = end_ts_ms
            self._emit(self._cur)
            self._cur = None
        else:
            # emit a copy as partial
            b = Bar(**self._cur.to_dict())
            self._emit(b)

    def get_finished(self, clear: bool = True) -> List[Bar]:
        out, self._finished = self._finished, ([] if clear else self._finished)
        return out

    # ----- protected hooks (override in subclasses) -----
    def _new_bar_from_trade(self, t: Trade) -> Bar:
        return Bar(
            symbol=t.symbol,
            start_ms=self._bar_start_ms(t.ts_ms),
            end_ms=self._bar_end_ms(t.ts_ms),
            open=t.price, high=t.price, low=t.price, close=t.price,
            volume=t.size, trades=1
        )

    def _after_first_trade(self, t: Trade) -> None:
        pass

    def _should_rollover(self, cur: Bar, t: Trade) -> bool:
        raise NotImplementedError

    def _update_with_trade(self, cur: Bar, t: Trade) -> None:
        cur.high = max(cur.high, t.price)
        cur.low = min(cur.low, t.price)
        cur.close = t.price
        cur.volume += t.size
        cur.trades += 1

    def _bar_start_ms(self, ts_ms: int) -> int:
        return ts_ms

    def _bar_end_ms(self, ts_ms: int) -> int:
        return ts_ms

    def _emit(self, bar: Bar) -> None:
        self._finished.append(bar)
        if self.on_bar:
            try:
                self.on_bar(bar)
            except Exception:
                # don't crash the feed if callback fails
                pass

    def _finish_current_bar(self, before_trade: Optional[Trade] = None) -> None:
        if not self._cur:
            return
        # close at the last known end
        self._emit(self._cur)
        self._cur = None
        if before_trade is not None:
            # start a fresh bar using the incoming trade
            self._cur = self._new_bar_from_trade(before_trade)
            self._after_first_trade(before_trade)

# --------------- Time Bars --------------------

class TimeBarAggregator(_BaseAggregator):
    """
    Fixed-duration bars (e.g., 1s/1m/5m).
    Rounds bar start to bucket boundary so bars are stable across restarts.
    """
    def __init__(
        self,
        symbol: str,
        bucket_ms: int,
        on_bar: Optional[Callable[[Bar], None]] = None
    ):
        super().__init__(symbol, on_bar)
        if bucket_ms <= 0:
            raise ValueError("bucket_ms must be > 0")
        self.bucket_ms = int(bucket_ms)

    def _bar_start_ms(self, ts_ms: int) -> int:
        return floor_to_bucket(ts_ms, self.bucket_ms)

    def _bar_end_ms(self, ts_ms: int) -> int:
        start = self._bar_start_ms(ts_ms)
        return start + self.bucket_ms

    def _should_rollover(self, cur: Bar, t: Trade) -> bool:
        # rollover when the incoming trade sits in the next bucket
        return t.ts_ms >= cur.start_ms + self.bucket_ms

    def _update_with_trade(self, cur: Bar, t: Trade) -> None:
        super()._update_with_trade(cur, t)
        # keep end bound aligned to bucket
        cur.end_ms = cur.start_ms + self.bucket_ms

# --------------- Tick Bars --------------------

class TickBarAggregator(_BaseAggregator):
    """Roll after N trades."""
    def __init__(self, symbol: str, trades_per_bar: int, on_bar: Optional[Callable[[Bar], None]] = None):
        super().__init__(symbol, on_bar)
        if trades_per_bar <= 0:
            raise ValueError("trades_per_bar must be > 0")
        self.n = int(trades_per_bar)

    def _should_rollover(self, cur: Bar, t: Trade) -> bool:
        return cur.trades >= self.n

# --------------- Volume Bars ------------------

class VolumeBarAggregator(_BaseAggregator):
    """Roll after cum volume >= threshold (allows overshoot)."""
    def __init__(self, symbol: str, vol_per_bar: float, on_bar: Optional[Callable[[Bar], None]] = None):
        super().__init__(symbol, on_bar)
        if vol_per_bar <= 0:
            raise ValueError("vol_per_bar must be > 0")
        self.v = float(vol_per_bar)

    def _should_rollover(self, cur: Bar, t: Trade) -> bool:
        # rollover BEFORE applying this trade if current volume already reached
        return cur.volume >= self.v

# --------------- Dollar Bars ------------------

class DollarBarAggregator(_BaseAggregator):
    """Roll after price*size sum >= threshold (allows overshoot)."""
    def __init__(self, symbol: str, notional_per_bar: float, on_bar: Optional[Callable[[Bar], None]] = None):
        super().__init__(symbol, on_bar)
        if notional_per_bar <= 0:
            raise ValueError("notional_per_bar must be > 0")
        self.notional = float(notional_per_bar)
        self._notional_acc = 0.0

    def _after_first_trade(self, t: Trade) -> None:
        self._notional_acc = t.price * t.size

    def _update_with_trade(self, cur: Bar, t: Trade) -> None:
        super()._update_with_trade(cur, t)
        self._notional_acc += t.price * t.size

    def _should_rollover(self, cur: Bar, t: Trade) -> bool:
        return self._notional_acc >= self.notional

    def _finish_current_bar(self, before_trade: Optional[Trade] = None) -> None:
        self._notional_acc = 0.0
        super()._finish_current_bar(before_trade)

# --------------- Multi-symbol wrapper ---------

class MultiSymbolTimeBars:
    """
    Convenience: manage many symbols with the same bar policy.
    Example:
        agg = MultiSymbolTimeBars(TimeBarAggregator, bucket_ms=60_000)
        agg.add_trade(Trade("AAPL", ts, 190.1, 50))
        finished = agg.drain_finished()
    """
    def __init__(self, factory, **factory_kwargs):
        self.factory = factory
        self.kw = factory_kwargs
        self._by_sym: Dict[str, _BaseAggregator] = {}
        self._finished: List[Bar] = []

    def add_trade(self, t: Trade) -> None:
        agg = self._by_sym.get(t.symbol)
        if agg is None:
            agg = self.factory(symbol=t.symbol, **self.kw)
            # capture per-symbol finished bars
            def _cb(bar: Bar, sym=t.symbol, sink=self._finished):
                sink.append(bar)
            agg.on_bar = _cb if agg.on_bar is None else agg.on_bar
            self._by_sym[t.symbol] = agg
        agg.add_trade(t)

    def flush_all(self, force_close: bool = True) -> None:
        for agg in self._by_sym.values():
            agg.flush(force_close=force_close)

    def drain_finished(self) -> List[Bar]:
        out, self._finished = self._finished, []
        return out

# --------------- Simple backfill helper -------

def build_time_bars(
    trades: List[Trade],
    bucket_ms: int
) -> List[Bar]:
    """
    Stateless helper to convert a list of trades into fixed time bars.
    Trades may be unsorted; we sort by ts_ms.
    """
    trades = sorted(trades, key=lambda x: x.ts_ms)
    out: List[Bar] = []
    agg = TimeBarAggregator(symbol=trades[0].symbol if trades else "", bucket_ms=bucket_ms)
    for t in trades:
        if agg.symbol and t.symbol != agg.symbol:
            # finalize current symbol and restart for a different one (rare in batch mode)
            agg.flush(force_close=True, end_ts_ms=t.ts_ms)
            out.extend(agg.get_finished())
            agg = TimeBarAggregator(symbol=t.symbol, bucket_ms=bucket_ms)
        agg.add_trade(t)
    agg.flush(force_close=True)
    out.extend(agg.get_finished())
    return out

# --------------- Smoke test -------------------

if __name__ == "__main__":  # pragma: no cover
    import random

    sym = "DEMO"
    start = now_ms()
    ticks = []
    px = 100.0
    for i in range(300):
        px = max(0.01, px + random.uniform(-0.05, 0.05))
        ticks.append(Trade(sym, start + i * 250, round(px, 4), size=random.randint(1, 5)))

    # 1m bars from 250ms ticks (~4 trades/sec)
    tb = TimeBarAggregator(symbol=sym, bucket_ms=60_000)
    for t in ticks:
        tb.add_trade(t)
    tb.flush()
    bars = tb.get_finished()
    print(f"time bars: {len(bars)}; first={bars[0].to_dict()}")

    # Volume bars
    vb = VolumeBarAggregator(symbol=sym, vol_per_bar=50)
    for t in ticks:
        vb.add_trade(t)
    vb.flush()
    print("volume bars:", len(vb.get_finished()))

    # Dollar bars
    db = DollarBarAggregator(symbol=sym, notional_per_bar=2_000)
    for t in ticks:
        db.add_trade(t)
    db.flush()
    print("dollar bars:", len(db.get_finished()))