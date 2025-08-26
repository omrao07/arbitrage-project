# backend/engine/strategies/market_maker.py
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from backend.engine.strategy_base import Strategy


class MarketMakerStrategy(Strategy):
    """
    Minimal market-making strategy:
      - Maintains an EWMA/vol estimate per symbol
      - Computes a target spread (bps) that scales with realized vol
      - Skews quotes based on inventory (inventory aversion)
      - Posts IOC limit orders on both sides (so stale quotes don't rest)
      - Enforces position caps & notional sanity

    Notes:
      • This uses `order(..., order_type="limit", extra={"tif": "ioc", "role": "maker"})`.
        Your OMS/broker adapter should respect TIF=IOC (or ignore gracefully).
      • To keep inventory accurate, call `on_fill(...)` from your fill bus, or
        call `set_position(symbol, qty)` on startup/rebuild.
    """

    def __init__(
        self,
        *,
        symbols: List[str],
        name: str = "mm_core",
        region: Optional[str] = None,
        # sizing
        default_qty: float = 1.0,
        lot_size: float = 1.0,
        max_position: float = 10.0,           # absolute cap in units
        max_notional_per_order: float = 1e6,  # safety (ignored if mark_price missing)
        # quoting
        refresh_secs: float = 1.0,            # min time between quote cycles per symbol
        target_spread_bps: float = 8.0,       # baseline half-spread ~4 bps each side
        min_spread_bps: float = 4.0,          # floor to avoid crossing / micro flicker
        edge_bps: float = 1.5,                # only quote if half-spread >= edge
        inventory_gamma: float = 0.5,         # skew strength (0..1)
        # stats
        vol_window: int = 180,                # price buffer length (ticks)
        vol_lookback: int = 60,               # how many rets for stdev
        ewma_alpha: float = 0.02,             # for running mid
        # microstructure
        min_tick: Optional[float] = None,
    ):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.symbols = [s.upper() for s in symbols]
        self.lot = float(lot_size)
        self.pos_cap = float(max_position)
        self.max_notional = float(max_notional_per_order)

        self.refresh_secs = float(refresh_secs)
        self.target_spread_bps = float(target_spread_bps)
        self.min_spread_bps = float(min_spread_bps)
        self.edge_bps = float(edge_bps)
        self.gamma = float(inventory_gamma)

        self.win = int(vol_window)
        self.lb = int(vol_lookback)
        self.alpha = float(ewma_alpha)
        self.min_tick = float(min_tick) if min_tick else None

        # state per symbol
        self._buf: Dict[str, Deque[float]] = {s: deque(maxlen=self.win) for s in self.symbols}
        self._last_quote_ts: Dict[str, float] = defaultdict(lambda: 0.0)
        self._mid: Dict[str, float] = {}
        self._pos: Dict[str, float] = defaultdict(float)   # internal position estimate
        self._last_px: Dict[str, float] = {}

    # ----------- optional external hooks -----------
    def on_fill(self, fill: Dict[str, Any]) -> None:
        """
        Optional: wire your OMS fills into this to keep inventory exact.
        Expected keys: symbol, side ('buy'|'sell'), qty (float)
        """
        sym = str(fill.get("symbol", "")).upper()
        if not sym:
            return
        q = float(fill.get("qty", 0.0) or 0.0)
        if q <= 0:
            return
        if str(fill.get("side", "")).lower() == "buy":
            self._pos[sym] += q
        else:
            self._pos[sym] -= q

    def set_position(self, symbol: str, qty: float) -> None:
        """Call this at startup with true positions rebuilt from storage."""
        self._pos[symbol.upper()] = float(qty)

    # ----------- main handler -----------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        px = float(tick.get("price") or tick.get("p") or 0.0)
        if sym not in self._buf or px <= 0:
            return

        # update buffers
        self._buf[sym].append(px)
        self._last_px[sym] = px

        # running mid (EWMA)
        mid = self._mid.get(sym, px)
        mid = (1.0 - self.alpha) * mid + self.alpha * px
        self._mid[sym] = mid

        # throttle quoting
        now = time.time()
        if (now - self._last_quote_ts[sym]) < self.refresh_secs:
            return

        # compute realized vol (fraction) over lookback
        buf = self._buf[sym]
        if len(buf) < max(self.lb + 2, 32):  # need a bit of history
            return
        arr = np.asarray(buf, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        lb = min(self.lb, len(rets))
        vol = float(np.std(rets[-lb:]))  # fraction

        # dynamic spread in bps
        dyn_spread_bps = max(
            self.min_spread_bps,
            self.target_spread_bps * (1.0 + 5.0 * vol)  # widen with vol (tunable)
        )

        # only quote if we expect at least minimal edge
        if (dyn_spread_bps / 2.0) < self.edge_bps:
            return

        # inventory-aware skew (move both quotes toward flattening side)
        pos = float(self._pos.get(sym, 0.0))
        inv_frac = np.clip(pos / max(self.pos_cap, 1e-9), -1.0, 1.0)
        skew_bps = self.gamma * inv_frac * (dyn_spread_bps / 2.0)

        half = dyn_spread_bps / 2.0
        bid_bps = half - skew_bps
        ask_bps = half + skew_bps

        bid = mid * (1.0 - bid_bps * 1e-4)
        ask = mid * (1.0 + ask_bps * 1e-4)

        if self.min_tick:
            bid = self._round_down(bid, self.min_tick)
            ask = self._round_up(ask, self.min_tick)
            if ask <= bid:  # respect tick
                ask = bid + self.min_tick

        # size (don’t cross caps)
        free_cap = max(self.pos_cap - abs(pos), 0.0)
        if free_cap <= 0.0:
            # at cap: quote only the flattening side
            if pos > 0:
                self._quote_one(sym, side="sell", px=ask)
            elif pos < 0:
                self._quote_one(sym, side="buy", px=bid)
            self._last_quote_ts[sym] = now
            return

        qty = max(self.lot, min(self.ctx.default_qty, free_cap))
        # notional guard if we have last price
        if px > 0 and (qty * px) > self.max_notional:
            qty = max(self.lot, self.max_notional / px)

        # Two-sided IOC quotes
        self._quote_one(sym, side="buy", px=bid, qty=qty)
        self._quote_one(sym, side="sell", px=ask, qty=qty)
        self._last_quote_ts[sym] = now

        # emit a tiny “confidence” signal based on tightness (optional, for allocator)
        tightness = 1.0 / (1.0 + dyn_spread_bps)       # smaller spread => larger tightness
        signal = float(np.clip((tightness - 0.5) * 2.0, -1.0, 1.0))
        self.emit_signal(signal)

    # ----------- helpers -----------
    def _quote_one(self, symbol: str, side: str, px: float, qty: Optional[float] = None) -> None:
        q = qty if (qty is not None) else self.ctx.default_qty
        if q <= 0 or px <= 0:
            return
        self.order(
            symbol=symbol,
            side=side,
            qty=q,
            order_type="limit",
            limit_price=float(px),
            mark_price=self._last_px.get(symbol),
            extra={"tif": "ioc", "role": "maker"}  # adapter may ignore/override
        )

    @staticmethod
    def _round_down(x: float, tick: float) -> float:
        return math.floor(x / tick) * tick

    @staticmethod
    def _round_up(x: float, tick: float) -> float:
        return math.ceil(x / tick) * tick