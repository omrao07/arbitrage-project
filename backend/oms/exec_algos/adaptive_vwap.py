# backend/engine/strategies/adaptive_vwap.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


@dataclass
class AdaptiveVWAPConfig:
    # Order/target
    symbol: str = "SPY"
    side: str = "buy"              # "buy" | "sell"
    target_qty: float = 10_000.0   # total shares/contracts to execute
    start_ms: Optional[int] = None # if None = now
    end_ms: Optional[int] = None   # if None = +6.5h (US cash)

    # Participation & adaptivity
    base_participation: float = 0.10     # base POV (10%)
    min_participation: float = 0.02
    max_participation: float = 0.35
    lag_tolerance_bps: float = 10.0      # how far behind schedule before we step up (bps of price)
    vol_boost_k: float = 2.0             # multiply participation when intraday vol spikes
    spread_penalty_bps: float = 0.5      # reduce participation when spreads widen (per bps over median)

    # Child order sizing / pacing
    min_child_qty: float = 100.0
    max_child_qty: float = 5_000.0
    child_interval_ms: int = 1500        # space child orders
    post_improve_bps: float = 0.4        # passive: improve the touch by this bps
    take_when_behind: bool = True        # allow aggression when behind schedule

    # Guards
    notional_cap: float = 2_000_000.0    # live notional cap
    pause_on_vol_z: float = 3.0          # pause if short-horizon vol Z > this
    kill_pct_complete: float = 1.02      # stop if we slightly exceed target
    hard_kill: bool = False


class AdaptiveVWAP(Strategy):
    """
    Adaptive VWAP / POV:
      - Tracks rolling session VWAP and session progress.
      - Schedules cumulative fills ~ linear (or better: realized volume curve via POV).
      - Adjusts participation up/down using: spread, short-horizon volatility, and slippage to VWAP.
      - Uses passive pegs; goes aggressive when behind schedule (optional).
    Feed tolerance:
      Tick needs {symbol|s, price|p|mid} or {bid,ask, last, size} for trade prints (p*size).
    """

    def __init__(self, name="exec_adaptive_vwap", region=None, cfg: Optional[AdaptiveVWAPConfig] = None):
        cfg = cfg or AdaptiveVWAPConfig()
        super().__init__(name=name, region=region, default_qty=cfg.min_child_qty)
        self.cfg = cfg
        self.sym = cfg.symbol.upper()
        self.buy = (cfg.side.lower() == "buy")

        # session clock
        now = self._now_ms()
        self.t0 = cfg.start_ms or now
        self.t1 = cfg.end_ms or (self.t0 + int(6.5 * 3600 * 1000))  # default 6.5h

        # rolling vwap + volume
        self.vwap_num = 0.0
        self.vwap_den = 0.0
        self.session_vol = 0.0

        # microstructure snapshots
        self.last_px = 0.0
        self.median_spread_bps = 2.0    # will learn
        self.spread_ewma = 2.0
        self.ret2_ewma = 0.0            # short-horizon variance proxy

        # progress
        self.filled_qty = 0.0
        self.last_child_ms = 0
        self.dead = False

    # ---------- lifecycle ----------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["execution", "VWAP", "POV", "adaptive"],
            "underlying": self.sym,
            "notes": "Adaptive VWAP/POV with spread/vol controls and lateness catch-up."
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

    def _progress_time(self, now: int) -> float:
        if now <= self.t0: return 0.0
        if now >= self.t1: return 1.0
        return (now - self.t0) / max(1.0, (self.t1 - self.t0))

    def _update_vwap(self, tick: Dict[str, Any]) -> None:
        # Prefer trade prints when available
        px = tick.get("last") or tick.get("price") or tick.get("p") or tick.get("mid")
        size = tick.get("size") or tick.get("q") or tick.get("qty")
        bid = tick.get("bid"); ask = tick.get("ask")
        # maintain last price and spread ewma
        if bid and ask:
            try:
                b = float(bid); a = float(ask)
                if b > 0 and a > 0:
                    mid = 0.5 * (b + a)
                    spread_bps = (a - b) / max(1e-9, mid) * 1e4
                    self.spread_ewma = 0.95 * self.spread_ewma + 0.05 * spread_bps
                    self.median_spread_bps = 0.99 * self.median_spread_bps + 0.01 * spread_bps
                    if self.last_px > 0:
                        r = (mid / self.last_px) - 1.0
                        self.ret2_ewma = 0.97 * self.ret2_ewma + 0.03 * (r * r)
                    self.last_px = mid
            except Exception:
                pass

        if px is None or size is None:
            return
        try:
            px = float(px); q = float(size)
        except Exception:
            return
        if px <= 0 or q <= 0:
            return

        self.vwap_num += px * q
        self.vwap_den += q
        self.session_vol += q

    def _rolling_vwap(self) -> float:
        return self.vwap_num / self.vwap_den if self.vwap_den > 0 else (self.last_px or 0.0)

    # ---------- adaptivity ----------
    def _planned_cum_qty(self, now: int) -> float:
        """
        Linear schedule blended with POV: base_participation * realized session volume.
        """
        time_prog = self._progress_time(now)
        linear = self.cfg.target_qty * time_prog
        pov = self.cfg.base_participation * self.session_vol
        # Blend (50/50). Feel free to weight differently.
        return 0.5 * linear + 0.5 * pov

    def _current_participation(self) -> float:
        part = self.cfg.base_participation

        # Volatility boost: if short-horizon vol z is high, we execute faster to avoid risk
        vol_z = math.sqrt(self.ret2_ewma) * 100.0  # ~% move proxy
        if vol_z > 0.5:  # rough normalization
            part *= (1.0 + self.cfg.vol_boost_k * min(2.0, (vol_z - 0.5)))

        # Spread penalty: if spread widens above median, reduce passive posting
        excess = max(0.0, self.spread_ewma - self.median_spread_bps)
        part *= max(0.5, 1.0 - self.cfg.spread_penalty_bps * excess / 100.0)

        return max(self.cfg.min_participation, min(self.cfg.max_participation, part))

    def _child_qty(self, now: int) -> float:
        # Catch-up if behind schedule
        planned = self._planned_cum_qty(now)
        behind = planned - self.filled_qty
        part = self._current_participation()

        # compute participation child versus current tape volume proxy:
        # use a small fraction of remaining vs. POV of recent flow
        pov_child = max(self.cfg.min_child_qty, part * max(1.0, self.session_vol) * 0.002)
        # add catch-up component
        catch = max(0.0, behind) * 0.15
        qty = max(self.cfg.min_child_qty, min(self.cfg.max_child_qty, max(pov_child, catch)))

        # stop once target reached
        rem = self.cfg.target_qty - self.filled_qty
        qty = min(qty, max(0.0, rem))
        return qty

    # ---------- execution ----------
    def _place_child(self, side: str, qty: float) -> None:
        px = self.last_px
        if px <= 0:
            return
        improve = px * (self.cfg.post_improve_bps / 1e4)
        if side == "buy":
            limit_px = px - improve
        else:
            limit_px = px + improve
        self.order(self.sym, side, qty=qty, order_type="limit", limit_price=limit_px,
                   extra={"reason": "adaptive_vwap_passive", "pov": True})

    def _go_aggressive(self, side: str, qty: float) -> None:
        px = self.last_px
        if px <= 0:
            return
        self.order(self.sym, side, qty=qty, order_type="market", mark_price=px,
                   extra={"reason": "adaptive_vwap_catchup", "take": True})

    # ---------- main ----------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill or self.dead:
            return
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym != self.sym:
            return

        self._update_vwap(tick)
        now = self._now_ms()
        if now < self.t0:
            return

        # completion/kill
        if self.filled_qty >= self.cfg.target_qty * self.cfg.kill_pct_complete:
            self.dead = True
            self.emit_signal(0.0)
            return

        # safety: pause on extreme volatility
        vol_z = math.sqrt(self.ret2_ewma) * 100.0
        if vol_z >= self.cfg.pause_on_vol_z:
            self.emit_signal(-0.1 if self.buy else 0.1)
            return

        # pacing
        if now - self.last_child_ms < self.cfg.child_interval_ms:
            return

        # planned vs actual; decide aggression
        planned = self._planned_cum_qty(now)
        lag_qty = planned - self.filled_qty
        behind = lag_qty > 0

        # choose side & qty
        side = "buy" if self.buy else "sell"
        qty = self._child_qty(now)
        if qty <= 0:
            return

        # choose passive vs aggressive
        vwap_px = self._rolling_vwap()
        slippage_bps = 0.0
        if vwap_px > 0 and self.last_px > 0:
            # how far are we from VWAP given our side
            if self.buy:
                slippage_bps = (self.last_px - vwap_px) / vwap_px * 1e4
            else:
                slippage_bps = (vwap_px - self.last_px) / vwap_px * 1e4

        aggressive = False
        if self.cfg.take_when_behind and behind and slippage_bps <= self.cfg.lag_tolerance_bps:
            aggressive = True

        # risk: notional cap
        if self.last_px * qty > self.cfg.notional_cap:
            qty = max(self.cfg.min_child_qty, self.cfg.notional_cap / max(1e-9, self.last_px))

        if aggressive:
            self._go_aggressive(side, qty)
        else:
            self._place_child(side, qty)

        self.last_child_ms = now

    # ---------- optional fills handler ----------
    # If your OMS publishes fills to a stream you consume elsewhere, wire an updater.
    # For a simple placeholder, you can call this externally from your broker adapter.
    def on_fill(self, qty: float, price: float) -> None:
        self.filled_qty += max(0.0, qty)
        # Keep vwap aggregates consistent (optional)
        if price > 0 and qty > 0:
            self.vwap_num += price * qty
            self.vwap_den += qty


# --------------- optional runner ---------------
if __name__ == "__main__":
    """
    Example:
        strat = AdaptiveVWAP(cfg=AdaptiveVWAPConfig(symbol="AAPL", side="buy", target_qty=50_000))
        # strat.run(stream="ticks.equities.us")
    """
    strat = AdaptiveVWAP()
    # strat.run(stream="ticks.equities.us")