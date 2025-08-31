# backend/engine/strategies/batch_auction.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


# -------------------- Config --------------------

@dataclass
class BatchAuctionConfig:
    symbol: str = "SPY"
    side: str = "buy"                 # "buy" | "sell"
    parent_qty: float = 50_000.0      # total quantity to execute

    # Auction schedule (choose one or mix):
    # 1) Explicit wall-clock fire times (epoch-ms)
    fire_times_ms: List[int] = None # type: ignore
    # 2) Periodic micro-batch every N seconds (e.g., 300s = 5m)
    period_sec: Optional[int] = None
    # 3) Exchange auction toggles (if your broker honors flags)
    use_open_auction: bool = False
    use_close_auction: bool = False
    # (You can still route using extra={"auction":"open"/"close"} even without broker-native flags.)

    # Sizing
    min_child_qty: float = 500.0
    max_child_qty: float = 15_000.0
    batch_ratio: float = 0.25         # fraction of remaining to send per batch (capped by min/max)

    # Pricing / microstructure
    post_improve_bps: float = 0.5     # improve the touch for passive limit
    vwap_band_bps: float = 15.0       # optional: limit price band around rolling VWAP
    allow_market_when_close: bool = True  # if within last bucket to close, use market

    # Risk / pacing
    cooldown_ms: int = 1500
    notional_cap: float = 2_500_000.0
    kill_pct_complete: float = 1.02   # stop after slight overshoot
    hard_kill: bool = False


# -------------------- Strategy --------------------

class BatchAuction(Strategy):
    """
    Batch Auction executor:
      - Accumulates parent order and fires child orders at scheduled "auction" times.
      - Supports explicit timestamps, periodic buckets, and open/close hints.
      - Prices child orders at mid ± improvement and (optionally) within a VWAP band.
      - OMS/broker decides whether an order is routed as MOO/MOC or as regular limit/market.

    Tick tolerance:
      - market data: {symbol|s, bid, ask, last|price|mid, size?}
      - fills: call on_fill(qty, px) from your OMS adapter (optional).
    """

    def __init__(self, name="exec_batch_auction", region=None, cfg: Optional[BatchAuctionConfig] = None):
        cfg = cfg or BatchAuctionConfig()
        super().__init__(name=name, region=region, default_qty=cfg.min_child_qty)
        self.cfg = cfg
        self.sym = cfg.symbol.upper()
        self.buy = (cfg.side.lower() == "buy")

        # market state
        self.last_mid = 0.0
        self.bid = 0.0
        self.ask = 0.0

        # rolling vwap
        self.vwap_num = 0.0
        self.vwap_den = 0.0

        # progress / schedule state
        self.filled_qty = 0.0
        self._last_fire_ms = 0
        self._period_anchor_ms = self._now_ms()
        self._dead = False

        # explicit times
        self.fire_times_ms = list(cfg.fire_times_ms) if cfg.fire_times_ms else []

    # ---------- lifecycle ----------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["execution", "auction", "batch"],
            "underlying": self.sym,
            "notes": "Releases child orders in scheduled batches (open/close/periodic) with VWAP band guard."
        })

    # ---------- helpers ----------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _safe_float(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d

    def _rolling_vwap(self) -> float:
        return self.vwap_num / self.vwap_den if self.vwap_den > 0 else (self.last_mid or 0.0)

    def _update_md(self, tick: Dict[str, Any]) -> None:
        b = tick.get("bid"); a = tick.get("ask")
        p = tick.get("last") or tick.get("price") or tick.get("p") or tick.get("mid")
        q = tick.get("size") or tick.get("q") or tick.get("qty")

        if b is not None and a is not None:
            try:
                self.bid = float(b); self.ask = float(a)
                if self.bid > 0 and self.ask > 0:
                    self.last_mid = 0.5 * (self.bid + self.ask)
            except Exception:
                pass
        elif p is not None:
            try:
                px = float(p)
                if px > 0:
                    # fallback mid from last trade
                    if self.last_mid == 0.0:
                        self.last_mid = px
            except Exception:
                pass

        if p is not None and q is not None:
            try:
                px = float(p); sz = float(q)
                if px > 0 and sz > 0:
                    self.vwap_num += px * sz
                    self.vwap_den += sz
            except Exception:
                pass

    def _remaining(self) -> float:
        return max(0.0, self.cfg.parent_qty - self.filled_qty)

    def _should_fire(self, now: int) -> Optional[str]:
        """
        Decide whether to fire this tick.
        Returns a reason string: "explicit" | "periodic" | "open" | "close" or None.
        """
        if now - self._last_fire_ms < self.cfg.cooldown_ms:
            return None

        # explicit times (epoch-ms)
        if self.fire_times_ms:
            # fire if we are within the current cooldown window
            for t in list(self.fire_times_ms):
                if now >= t:
                    self.fire_times_ms.remove(t)
                    return "explicit"

        # periodic buckets
        if self.cfg.period_sec:
            period_ms = int(self.cfg.period_sec * 1000)
            if period_ms > 0 and now - self._period_anchor_ms >= period_ms:
                # align to grid
                self._period_anchor_ms = now
                return "periodic"

        # exchange open/close toggles can be handled by your orchestrator
        # Here we expose hooks in case you call them manually via extra streams.
        # (Return "open"/"close" if you trigger them externally.)
        return None

    def _child_size(self) -> float:
        rem = self._remaining()
        if rem <= 0:
            return 0.0
        qty = max(self.cfg.min_child_qty, min(self.cfg.max_child_qty, rem * self.cfg.batch_ratio))
        # notional cap
        px = self.last_mid or 0.0
        if px > 0 and qty * px > self.cfg.notional_cap:
            qty = max(self.cfg.min_child_qty, self.cfg.notional_cap / px)
        return qty

    def _limit_px(self, side: str) -> Optional[float]:
        mid = self.last_mid
        if mid <= 0:
            return None
        improve = mid * (self.cfg.post_improve_bps / 1e4)
        if side == "buy":
            return mid - improve
        return mid + improve

    def _band_ok(self, side: str, limit_px: float) -> bool:
        """
        Optional guard: keep price within band around rolling VWAP.
        For buys → do not exceed vwap + band; for sells → do not go below vwap - band.
        """
        vwap = self._rolling_vwap()
        if vwap <= 0 or limit_px <= 0:
            return True
        band = vwap * (self.cfg.vwap_band_bps / 1e4)
        if side == "buy":
            return limit_px <= vwap + band
        return limit_px >= vwap - band

    def _place_child(self, reason: str) -> None:
        side = "buy" if self.buy else "sell"
        qty = self._child_size()
        if qty <= 0 or self.last_mid <= 0:
            return

        limit_px = self._limit_px(side)
        if limit_px is None:
            return

        # VWAP band guard
        if not self._band_ok(side, limit_px):
            # too far from band → skip this batch
            return

        self.order(
            self.sym,
            side=side,
            qty=qty,
            order_type="limit",
            limit_price=limit_px,
            extra={
                "reason": f"batch_auction_{reason}",
                "auction": reason,
                "vwap_band_bps": self.cfg.vwap_band_bps,
            },
        )

        self._last_fire_ms = self._now_ms()

    # ---------- main ----------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self._dead or self.cfg.hard_kill:
            return
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym != self.sym:
            return

        # update market data & VWAP
        self._update_md(tick)

        # stop if done
        if self.filled_qty >= self.cfg.parent_qty * self.cfg.kill_pct_complete:
            self._dead = True
            self.emit_signal(0.0)
            return

        # scheduling
        now = self._now_ms()
        reason = self._should_fire(now)
        if reason:
            # If at very end of program and allowed, take market to finish
            if self.cfg.allow_market_when_close and self._remaining() > 0 and not self.fire_times_ms and not self.cfg.period_sec:
                # if this was the last explicit time, be aggressive
                pass
            self._place_child(reason)

        # simple health signal: remaining fraction (negative when sell)
        rem_frac = self._remaining() / max(1.0, self.cfg.parent_qty)
        self.emit_signal(rem_frac if self.buy else -rem_frac)

    # optional: OMS calls this when fills happen
    def on_fill(self, qty: float, price: float) -> None:
        self.filled_qty += max(0.0, qty)
        if price > 0 and qty > 0:
            self.vwap_num += price * qty
            self.vwap_den += qty


# ---------------- runner (optional) ----------------

if __name__ == "__main__":
    """
    Example:
        from time import time
        now = int(time()*1000)
        cfg = BatchAuctionConfig(
            symbol="AAPL",
            side="buy",
            parent_qty=80_000,
            fire_times_ms=[now + 60_000, now + 180_000, now + 300_000],  # 1m, 3m, 5m
            vwap_band_bps=10.0,
            post_improve_bps=0.6,
        )
        strat = BatchAuction(cfg=cfg)
        # strat.run(stream="ticks.equities.us")
    """
    strat = BatchAuction()
    # strat.run(stream="ticks.equities.us")