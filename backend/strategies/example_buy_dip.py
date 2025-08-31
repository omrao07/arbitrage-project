# backend/engine/strategies/example_buy_dip.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


@dataclass
class BuyDipConfig:
    symbols: tuple[str, ...] = ("BTCUSDT",)     # or any symbols you feed into the stream
    ewma_alpha: float = 0.02                    # smoothing for running average (0.01–0.1 typical)
    dip_bps: float = 10.0                       # buy if price <= avg - dip_bps
    rip_bps: float = 10.0                       # sell if price >= avg + rip_bps
    default_qty: float = 0.001                  # fallback size
    rebalance_cooldown_ms: int = 1_000          # min time between trades per symbol
    hard_kill: bool = False                     # quick off switch


class ExampleBuyTheDip(Strategy):
    """
    Minimal example strategy:
      - Maintains an EWMA 'fair value' per symbol
      - Buys when price dips X bps below EWMA; sells when rises X bps above
      - Emits a signal in [-1, +1] based on distance to EWMA
    This is a toy to prove the plumbing (signal → order → risk → OMS).
    """

    def __init__(self, name: str = "example_buy_dip", region: Optional[str] = None, cfg: Optional[BuyDipConfig] = None):
        cfg = cfg or BuyDipConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg

        # state
        self._avg: Dict[str, float] = {}                             # EWMA per symbol
        self._last_ts_ms: DefaultDict[str, int] = defaultdict(lambda: 0)

    # -------- lifecycle --------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["example", "mean_reversion"],
            "region": self.ctx.region or "GLOBAL",
            "notes": f"Buy dip/sell rip around EWMA; dip={self.cfg.dip_bps}bps, rip={self.cfg.rip_bps}bps"
        })

    # -------- helpers --------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _update_ewma(self, sym: str, px: float) -> float:
        a = max(1e-4, min(0.99, self.cfg.ewma_alpha))
        prev = self._avg.get(sym, px)
        ewma = (1 - a) * prev + a * px
        self._avg[sym] = ewma
        return ewma

    # -------- main --------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        # tolerate multiple tick shapes
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if not sym or sym not in self.cfg.symbols:
            return

        px = tick.get("price") or tick.get("p") or tick.get("mid")
        if px is None:
            bid, ask = tick.get("bid"), tick.get("ask")
            if bid and ask and bid > 0 and ask > 0:
                px = 0.5 * (float(bid) + float(ask))
        if px is None:
            return

        try:
            px = float(px)
        except Exception:
            return
        if px <= 0:
            return

        # update model
        avg = self._update_ewma(sym, px)

        # distance in bps and normalized signal
        diff_bps = (px - avg) / avg * 1e4
        sig = max(-1.0, min(1.0, -diff_bps / (10.0 * max(self.cfg.dip_bps, self.cfg.rip_bps, 1.0))))
        self.emit_signal(sig)

        # throttle
        now = self._now_ms()
        if now - self._last_ts_ms[sym] < self.cfg.rebalance_cooldown_ms:
            return

        # trade rules
        if diff_bps <= -self.cfg.dip_bps:
            # buy the dip
            self.order(
                symbol=sym, side="buy", qty=self.ctx.default_qty,
                order_type="market", mark_price=px,
                extra={"reason": "buy_the_dip", "avg": avg, "diff_bps": diff_bps}
            )
            self._last_ts_ms[sym] = now

        elif diff_bps >= self.cfg.rip_bps:
            # sell the rip
            self.order(
                symbol=sym, side="sell", qty=self.ctx.default_qty,
                order_type="market", mark_price=px,
                extra={"reason": "sell_the_rip", "avg": avg, "diff_bps": diff_bps}
            )
            self._last_ts_ms[sym] = now


# -------- optional runner --------
if __name__ == "__main__":
    """
    Example:
      python -m backend.engine.strategies.example_buy_dip
    Typically you attach via Strategy.run(stream="ticks.crypto") elsewhere.
    """
    strat = ExampleBuyTheDip()
    # strat.run(stream="ticks.crypto")