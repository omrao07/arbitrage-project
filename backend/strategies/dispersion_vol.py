# backend/engine/strategies/dispersion_vol.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, DefaultDict
from collections import defaultdict

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


@dataclass
class DispersionConfig:
    index: str = "SPY"                           # index or ETF proxy
    constituents: tuple[str, ...] = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA")
    # EWMA params
    ewma_alpha: float = 0.03                     # ~ fast-ish; tune 0.01–0.10
    var_floor: float = 1e-10                     # avoid div-by-zero
    # Trading thresholds (on normalized dispersion)
    enter_thresh: float = 0.25                   # go long/short dispersion if |z| >= 0.25
    exit_thresh: float = 0.10                    # flatten if |z| < 0.10
    # Sizing / limits
    basket_gross_notional: float = 10_000.0      # target basket notional when active
    per_leg_qty: float = 1.0                     # fallback qty if prices unknown
    max_gross_notional: float = 100_000.0
    # Rebalance & cooldowns
    rebalance_cooldown_ms: int = 2_000
    # Behavior
    hard_kill: bool = False
    # Optional manual weights for basket (same order as constituents); default equal-weight
    weights: Optional[tuple[float, ...]] = None


class DispersionVol(Strategy):
    """
    Cash dispersion proxy using EWMA variances:
      - Maintain EWMA mean & variance per symbol (index + constituents)
      - Compute normalized dispersion = (avg var_const - var_index) / max(var_index, floor)
      - Enter long-dispersion (buy basket, short index) if above +enter_thresh
      - Enter short-dispersion (sell basket, buy index) if below -enter_thresh
      - Exit/flatten when |dispersion| < exit_thresh
      - Emits signal in [-1, +1] scaled from dispersion

    Notes:
      - This approximates options dispersion using cash legs. It’s a decent proxy
        for demo/routing; your true options version can plug the same signal.
    """

    def __init__(self, name="alpha_dispersion_vol", region=None, cfg: Optional[DispersionConfig] = None):
        cfg = cfg or DispersionConfig()
        super().__init__(name=name, region=region, default_qty=cfg.per_leg_qty)
        self.cfg = cfg

        self.index = cfg.index.upper()
        self.constituents = tuple(s.upper() for s in cfg.constituents)
        self.weights = self._normalize_weights(cfg.weights, len(self.constituents))

        # EWMA state: {sym: (mean, var, last_px)}
        self._ewma: Dict[str, Tuple[float, float, float]] = {}
        # Last seen prices for notional sizing
        self._last_px: Dict[str, float] = {}
        # Position state: -1 short-dispersion, 0 flat, +1 long-dispersion
        self._state: int = 0
        # Cooldown
        self._last_rebalance_ms: int = 0

    # ------------- lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["dispersion", "volatility", "relative_value"],
            "region": self.ctx.region or "US",
            "index": self.index,
            "basket": list(self.constituents),
            "notes": "Cash dispersion proxy using EWMA variances; buys basket, shorts index when single-name var outruns index var."
        })

    # ------------- helpers ------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _normalize_weights(w: Optional[tuple[float, ...]], n: int) -> tuple[float, ...]:
        if not w or len(w) != n:
            return tuple([1.0 / n] * n)
        s = sum(abs(x) for x in w) or 1.0
        return tuple(float(x) / s for x in w)

    def _update_ewma(self, sym: str, px: float) -> None:
        a = max(1e-6, min(0.99, self.cfg.ewma_alpha))
        if sym not in self._ewma:
            # initialize with zero var
            self._ewma[sym] = (px, 0.0, px)
            self._last_px[sym] = px
            return
        mean, var, last = self._ewma[sym]
        # use returns on price to stabilize (pct move)
        if last > 0:
            r = (px / last) - 1.0
        else:
            r = 0.0
        # EWMA mean of returns (not used in logic, but kept for completeness)
        mean_r = (1 - a) * mean + a * px  # keep mean price for sizing; not strictly needed
        # EWMA variance of returns (classic: v_t = (1-a)*v_{t-1} + a*r_t^2)
        var_r = (1 - a) * var + a * (r * r)
        self._ewma[sym] = (mean_r, var_r, px)
        self._last_px[sym] = px

    def _var(self, sym: str) -> float:
        if sym in self._ewma:
            return max(self.cfg.var_floor, self._ewma[sym][1])
        return self.cfg.var_floor

    def _dispersion_normalized(self) -> Optional[float]:
        # Need index + all constituents tracked
        if self.index not in self._ewma:
            return None
        if any(c not in self._ewma for c in self.constituents):
            return None
        v_index = self._var(self.index)
        v_names = [self._var(c) for c in self.constituents]
        v_avg = sum(v_names) / max(1, len(v_names))
        disp = (v_avg - v_index) / max(v_index, self.cfg.var_floor)
        return disp

    def _basket_notional_side(self, side: str) -> None:
        """
        Execute basket trades to reach ~basket_gross_notional on the given side.
        side: "buy" to go long basket, "sell" to short basket
        """
        target = float(self.cfg.basket_gross_notional)
        # spread across constituents by weights and prices
        for w, sym in zip(self.weights, self.constituents):
            px = float(self._last_px.get(sym, 0.0))
            if px <= 0:
                # fallback to qty if no price yet
                self.order(sym, side, qty=self.ctx.default_qty, order_type="market",
                           extra={"reason": "dispersion_basket_fallback"})
                continue
            notional = max(0.0, target * abs(w))
            qty = max(1.0, notional / px)
            self.order(sym, side, qty=qty, order_type="market",
                       extra={"reason": "dispersion_basket", "w": w, "px": px})

    def _index_notional_side(self, side: str) -> None:
        px = float(self._last_px.get(self.index, 0.0))
        target = float(self.cfg.basket_gross_notional)
        if px <= 0:
            self.order(self.index, side, qty=self.ctx.default_qty, order_type="market",
                       extra={"reason": "dispersion_index_fallback"})
            return
        qty = max(1.0, target / px)
        self.order(self.index, side, qty=qty, order_type="market",
                   extra={"reason": "dispersion_index", "px": px})

    def _flatten(self) -> None:
        """
        Simple flatten: reverse last state legs with same notional target.
        (In real system, query OMS positions; here we just send opposite legs.)
        """
        if self._state == 0:
            return
        if self._state > 0:
            # was long dispersion: long basket / short index -> unwind
            self._basket_notional_side("sell")
            self._index_notional_side("buy")
        else:
            # was short dispersion: short basket / long index -> unwind
            self._basket_notional_side("buy")
            self._index_notional_side("sell")
        self._state = 0

    # ------------- main ---------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym not in (self.index,) + self.constituents:
            return

        # Accept common shapes: {price|p} or trade/quote
        px = tick.get("price") or tick.get("p") or tick.get("mid")
        if px is None:
            bid = tick.get("bid")
            ask = tick.get("ask")
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

        # Update EWMA stats
        self._update_ewma(sym, px)

        # Compute normalized dispersion
        disp = self._dispersion_normalized()
        if disp is None:
            return

        # Emit signal in [-1, 1] (clamp)
        # Scale: +/- 1.0 at |disp| = 1.0 (tune as desired)
        sig = max(-1.0, min(1.0, disp))
        self.emit_signal(sig)

        now = self._now_ms()
        if now - self._last_rebalance_ms < self.cfg.rebalance_cooldown_ms:
            return

        # Risk: rough notional guard
        # (Use index price as proxy since basket target equals index target)
        idx_px = float(self._last_px.get(self.index, 0.0))
        if idx_px > 0 and self.cfg.basket_gross_notional > self.cfg.max_gross_notional:
            return

        # Trading logic with hysteresis
        if abs(disp) < self.cfg.exit_thresh:
            if self._state != 0:
                self._flatten()
                self._last_rebalance_ms = now
            return

        if disp >= self.cfg.enter_thresh:
            # LONG DISPERSION: buy basket, short index
            if self._state <= 0:  # enter or flip
                self._flatten()
                self._basket_notional_side("buy")
                self._index_notional_side("sell")
                self._state = +1
                self._last_rebalance_ms = now

        elif disp <= -self.cfg.enter_thresh:
            # SHORT DISPERSION: short basket, buy index
            if self._state >= 0:  # enter or flip
                self._flatten()
                self._basket_notional_side("sell")
                self._index_notional_side("buy")
                self._state = -1
                self._last_rebalance_ms = now


# ---------------------- optional: quick runner ----------------------
if __name__ == "__main__":
    """
    Example quick attach:
      python -m backend.engine.strategies.dispersion_vol
    Typically run via Strategy.run(stream="ticks.equities.us") elsewhere.
    """
    strat = DispersionVol()
    # strat.run(stream="ticks.equities.us")