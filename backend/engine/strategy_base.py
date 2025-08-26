# backend/engine/strategy_base.py
from __future__ import annotations

import abc
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import redis

from backend.bus.streams import (
    consume_stream,
    publish_stream,
    hset,
)

# ---------- Environment / Defaults ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
INCOMING_ORDERS = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


@dataclass
class Context:
    """
    Lightweight context passed to strategies.
    Extend as needed (e.g., cache last prices, calendar, etc.)
    """
    name: str
    region: Optional[str] = None
    capital_base: float = 100_000.0
    default_qty: float = 1.0


class Strategy(abc.ABC):
    """
    Base class for all strategies (alpha & diversified).
    Implement `on_tick`. Optionally override `on_start` and `on_stop`.
    Use `order()` to submit orders (goes through risk manager before OMS).
    Use `emit_signal/emit_vol/emit_drawdown` to power allocator.
    """

    def __init__(self, name: str, region: Optional[str] = None, default_qty: float = 1.0):
        self.ctx = Context(name=name, region=region, default_qty=default_qty)
        self._running = False

    # ---- Lifecycle ---------------------------------------------------------
    def on_start(self) -> None:
        """Called once at start. Override as needed."""
        # Mark as enabled by default (UI can toggle later)
        hset("strategy:enabled", self.ctx.name, "true")

    @abc.abstractmethod
    def on_tick(self, tick: Dict[str, Any]) -> None:
        """
        Called for each incoming tick (already normalized enough to contain symbol/price/ts if available).
        You should call self.order(...) when you want to trade.
        """
        raise NotImplementedError

    def on_stop(self) -> None:
        """Called once when stopping. Override as needed."""
        pass

    # ---- Order API ---------------------------------------------------------
    def order(
        self,
        symbol: str,
        side: str,
        qty: float | None = None,
        *,
        order_type: str = "market",
        limit_price: float | None = None,
        venue: Optional[str] = None,
        mark_price: float | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Submit an order to the pre-risk stream. Risk manager will validate, then OMS fills.
        """
        q = self.ctx.default_qty if (qty is None or qty <= 0) else float(qty)
        payload = {
            "ts_ms": int(time.time() * 1000),
            "strategy": self.ctx.name,
            "symbol": str(symbol).upper(),
            "side": side.lower(),  # 'buy' | 'sell'
            "qty": q,
            "typ": order_type,
            "limit_price": limit_price,
        }
        if venue:      payload["venue"] = venue
        if mark_price: payload["mark_price"] = float(mark_price)
        if self.ctx.region: payload["region"] = self.ctx.region
        if extra: payload.update(extra)

        publish_stream(INCOMING_ORDERS, payload)

    # ---- Metrics API (allocator inputs) -----------------------------------
    def emit_signal(self, score: float) -> None:
        """
        score in [-1, +1]. Stored under HSET strategy:signal <name> {"score": x}
        """
        hset("strategy:signal", self.ctx.name, {"score": float(max(-1.0, min(1.0, score)))})

    def emit_vol(self, vol: float) -> None:
        """
        Strategy risk metric (e.g., stdev of returns). Scale consistently across strategies.
        """
        hset("strategy:vol", self.ctx.name, {"vol": float(max(0.0, vol))})

    def emit_drawdown(self, dd: float) -> None:
        """
        Rolling drawdown fraction (e.g., 8% -> 0.08).
        """
        hset("strategy:drawdown", self.ctx.name, {"dd": float(max(0.0, min(1.0, dd)))})

    # ---- Runner ------------------------------------------------------------
    def run(self, stream: str) -> None:
        """
        Attach this strategy to a Redis Stream (e.g., 'trades.crypto').
        Processes new messages only (start_id='$').
        """
        self._running = True

        def _graceful_stop(signum, frame):
            try:
                self.on_stop()
            finally:
                self._running = False
                sys.exit(0)

        signal.signal(signal.SIGINT, _graceful_stop)
        signal.signal(signal.SIGTERM, _graceful_stop)

        self.on_start()

        for _, tick in consume_stream(stream, start_id="$", block_ms=1000, count=200):
            if not self._running:
                break
            try:
                if isinstance(tick, str):
                    tick = json.loads(tick)
                self.on_tick(tick)
            except Exception as e:
                # minimal safety log to Redis (optional)
                r.lpush(f"strategy:errors:{self.ctx.name}", json.dumps({"ts": int(time.time()*1000), "err": str(e)}))


# ---------------- Convenience: a toy example ----------------
class ExampleBuyTheDip(Strategy):
    """
    Minimal example strategy:
    - Keeps a simple running mid price per symbol
    - Buys small qty when price dips X bps below running avg; sells when above
    This is just to prove wiring; replace with your real 72 strategies.
    """
    def __init__(self, name="example_buy_dip", region=None, default_qty=0.001, bps=10.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.bps = float(bps)
        self._avg: Dict[str, float] = {}

    def on_tick(self, tick: Dict[str, Any]) -> None:
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        px  = float(tick.get("price") or tick.get("p") or 0.0)
        if not sym or px <= 0:
            return

        # update running average (EWMA)
        a = self._avg.get(sym, px)
        a = 0.98 * a + 0.02 * px
        self._avg[sym] = a

        # simple signal
        diff_bps = (px - a) / a * 1e4
        score = max(-1.0, min(1.0, -diff_bps / (10 * self.bps)))  # rough scale
        self.emit_signal(score)

        # trade rules
        if diff_bps <= -self.bps:            # dipped below avg by X bps -> buy
            self.order(sym, "buy")
        elif diff_bps >= self.bps:           # above avg by X bps -> sell
            self.order(sym, "sell")


# ---------------- News handler base ----------------
class BaseStrategy:
    def on_news(self, event: Any) -> None:
        """
        Handle incoming news event.
        Override this in custom strategy classes.
        """
        pass