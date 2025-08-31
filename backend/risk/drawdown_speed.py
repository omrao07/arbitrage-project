# backend/risk/drawdown_speed.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
import math


_now_ms = lambda: int(time.time() * 1000)


@dataclass
class DDSConfig:
    # Rolling windows (milliseconds)
    slope_window_ms: int = 15 * 60 * 1000      # 15 minutes for fast slope
    lookback_ms: int = 6 * 60 * 60 * 1000      # 6 hours context for peak/trough
    uw_window_ms: int = 5 * 24 * 60 * 60 * 1000  # 5 days for Under-Water stats

    # Smoothing
    ewma_alpha_fast: float = 0.25
    ewma_alpha_slow: float = 0.08

    # Capitalization
    base_nav: float = 100_000.0                # only used if you feed PnL deltas instead of NAV

    # Thresholds (all in drawdown-speed bps per *day* unless specified)
    warn_speed_bps_per_day: float = 120.0      # warn if decay faster than this
    block_speed_bps_per_day: float = 250.0     # trip kill policies above this
    max_dd_warn: float = 0.06                  # 6% DD warn
    max_dd_block: float = 0.10                 # 10% DD block

    # Hard safety
    min_points_for_stats: int = 6
    max_ring: int = 20_000


@dataclass
class Snapshot:
    ts_ms: int
    nav: float
    ret: float                  # last return over last tick (fraction)
    dd: float                   # current drawdown fraction (0.08 = -8%)
    max_dd: float               # rolling max drawdown over lookback
    speed_bps_min: float        # drawdown speed, bps per minute (>=0 is bad)
    speed_bps_day: float        # speed extrapolated per day
    peak_ts_ms: int
    trough_ts_ms: int
    time_under_water_min: float
    pain_index: float           # average depth while underwater over uw_window
    half_life_min: Optional[float]  # est minutes to halve the DD if current slope persists
    regime: str                 # calm | caution | critical
    alerts: List[str]


class DrawdownSpeed:
    """
    Streaming drawdown-speed monitor.

    Feed with either:
      - on_nav(nav, ts_ms=...)               # absolute NAV/equity level
      - on_pnl(delta_pnl, ts_ms=...)         # PnL delta; uses cfg.base_nav for init

    Query:
      - snapshot()                            # latest metrics for dashboards/alerts
      - should_block()                        # True when kill-switch conditions met
      - regime()                              # 'calm' | 'caution' | 'critical'
    """

    def __init__(self, cfg: Optional[DDSConfig] = None):
        self.cfg = cfg or DDSConfig()
        self._ring: Deque[Tuple[int, float]] = deque(maxlen=self.cfg.max_ring)  # (ts_ms, nav)
        self._peak_nav: float = self.cfg.base_nav
        self._peak_ts: int = _now_ms()
        self._trough_nav: float = self.cfg.base_nav
        self._trough_ts: int = self._peak_ts
        self._dd_fast_ewma: float = 0.0      # drawdown fraction ewma
        self._speed_ewma_min: float = 0.0    # bps/min
        self._speed_ewma_day: float = 0.0    # bps/day
        self._uw_area: float = 0.0           # integral of drawdown over time (for pain index)
        self._uw_last_ts: Optional[int] = None
        self._last_snap: Optional[Snapshot] = None

    # ------------- ingestion -------------
    def on_nav(self, nav: float, ts_ms: Optional[int] = None) -> None:
        ts = ts_ms or _now_ms()
        if nav <= 0:
            return
        # append and prune windows
        self._ring.append((ts, float(nav)))
        self._prune(self.cfg.lookback_ms)

        # update peak/trough within lookback
        self._recalc_peak_trough()

        # compute drawdown vs rolling peak
        dd = self._drawdown_frac()

        # EWMA of drawdown
        a = self.cfg.ewma_alpha_fast
        self._dd_fast_ewma = (1 - a) * self._dd_fast_ewma + a * dd

        # update time-under-water area (for pain index)
        self._accumulate_underwater(dd, ts)

        # compute slope-based speed (bps/min & bps/day)
        speed_min, speed_day = self._drawdown_speed()

        # regime + alerts
        alerts, regime = self._alerts(dd, speed_day)

        # half-life estimate (minutes) assuming *reversal* with same magnitude slope
        hl = self._half_life_minutes(dd, speed_min)

        self._last_snap = Snapshot(
            ts_ms=ts,
            nav=nav,
            ret=self._last_ret(),
            dd=dd,
            max_dd=self._max_drawdown_over_lookback(),
            speed_bps_min=speed_min,
            speed_bps_day=speed_day,
            peak_ts_ms=self._peak_ts,
            trough_ts_ms=self._trough_ts,
            time_under_water_min=self._time_under_water_minutes(),
            pain_index=self._pain_index(),
            half_life_min=hl,
            regime=regime,
            alerts=alerts,
        )

    def on_pnl(self, delta_pnl: float, ts_ms: Optional[int] = None) -> None:
        """If you don't have NAV, feed PnL deltas; we integrate off last NAV or base_nav."""
        if not self._ring:
            self._ring.append((_now_ms(), float(self.cfg.base_nav)))
        _, last_nav = self._ring[-1]
        self.on_nav(max(1e-6, last_nav + float(delta_pnl)), ts_ms=ts_ms)

    # ------------- internals -------------
    def _prune(self, window_ms: int) -> None:
        cutoff = (self._ring[-1][0] if self._ring else _now_ms()) - window_ms
        while self._ring and self._ring[0][0] < cutoff:
            self._ring.popleft()

    def _recalc_peak_trough(self) -> None:
        """Recompute rolling peak/trough within lookback window efficiently."""
        if not self._ring:
            return
        # peak is max NAV in ring; trough is min NAV since the peak
        peak_nav, peak_ts = -1.0, self._ring[0][0]
        trough_nav, trough_ts = float("inf"), self._ring[0][0]
        seen_peak = False
        for ts, nav in self._ring:
            if nav > peak_nav:
                peak_nav, peak_ts = nav, ts
                seen_peak = True
                trough_nav, trough_ts = nav, ts
            elif seen_peak and nav < trough_nav:
                trough_nav, trough_ts = nav, ts
        self._peak_nav, self._peak_ts = peak_nav, peak_ts
        self._trough_nav, self._trough_ts = trough_nav, trough_ts

    def _drawdown_frac(self) -> float:
        if not self._ring:
            return 0.0
        _, nav = self._ring[-1]
        if self._peak_nav <= 0:
            return 0.0
        dd = max(0.0, (self._peak_nav - nav) / self._peak_nav)
        return dd

    def _last_ret(self) -> float:
        if len(self._ring) < 2:
            return 0.0
        (_, n0), (_, n1) = self._ring[-2], self._ring[-1]
        return (n1 / n0) - 1.0

    def _max_drawdown_over_lookback(self) -> float:
        """Peak-to-trough percentage drop within current ring."""
        if not self._ring:
            return 0.0
        # We already computed rolling peak/trough within lookback
        if self._peak_nav <= 0:
            return 0.0
        return max(0.0, (self._peak_nav - self._trough_nav) / self._peak_nav)

    def _drawdown_speed(self) -> Tuple[float, float]:
        """
        Returns (bps_per_min, bps_per_day) of drawdown speed.
        Computed from linear slope of NAV (in log space) over slope_window,
        then translated into DD speed relative to peak. Non-negative.
        """
        if not self._ring:
            return (0.0, 0.0)

        ts_now = self._ring[-1][0]
        cutoff = ts_now - self.cfg.slope_window_ms
        xs: List[float] = []
        ys: List[float] = []  # log NAV
        for ts, nav in self._ring:
            if ts >= cutoff and nav > 0:
                xs.append((ts - ts_now) / 60000.0)     # minutes from now (<=0)
                ys.append(math.log(nav))

        if len(xs) < self.cfg.min_points_for_stats:
            return (0.0, 0.0)

        # simple OLS slope on (x, y): y = a + b x ; b ≈ d log(NAV)/d minutes
        n = float(len(xs))
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            b_per_min = 0.0
        else:
            b_per_min = (n * sxy - sx * sy) / denom  # per minute in log space

        # translate to expected fractional change per minute ~ b
        # Turn this into DD speed relative to current peak (non-negative)
        dd_speed_min = max(0.0, -b_per_min) * 1e4  # bps/min (negative slope -> losing)
        # EWMA smooth
        a_fast = self.cfg.ewma_alpha_fast
        self._speed_ewma_min = (1 - a_fast) * self._speed_ewma_min + a_fast * dd_speed_min

        dd_speed_day = self._speed_ewma_min * 60 * 24
        # slow smoothing for per-day
        a_slow = self.cfg.ewma_alpha_slow
        self._speed_ewma_day = (1 - a_slow) * self._speed_ewma_day + a_slow * dd_speed_day

        return (self._speed_ewma_min, self._speed_ewma_day)

    def _accumulate_underwater(self, dd: float, ts_ms: int) -> None:
        """Integrate drawdown depth over time (for pain index)."""
        if dd <= 1e-12:
            # reset UW timer when back to peak/high
            self._uw_last_ts = None
            return
        if self._uw_last_ts is None:
            self._uw_last_ts = ts_ms
            return
        dt_min = max(0.0, (ts_ms - self._uw_last_ts) / 60000.0)
        self._uw_last_ts = ts_ms
        # accumulate area
        self._uw_area += dd * dt_min
        # decay outside window
        self._decay_uw(ts_ms)

    def _decay_uw(self, ts_ms: int) -> None:
        # drop old area proportionally (coarse) so pain index stays within uw_window
        if not self._ring:
            return
        first_ts = self._ring[0][0]
        window_min = self.cfg.uw_window_ms / 60000.0
        span_min = max(1.0, (ts_ms - first_ts) / 60000.0)
        if span_min > window_min:
            # scale down area to keep effective horizon ~ window_min
            self._uw_area *= window_min / span_min

    def _time_under_water_minutes(self) -> float:
        if self._uw_last_ts is None:
            return 0.0
        # find last time we were at peak (dd≈0): approximate using peak_ts
        if not self._ring:
            return 0.0
        ts_last = self._ring[-1][0]
        return max(0.0, (ts_last - self._peak_ts) / 60000.0) if self._drawdown_frac() > 0 else 0.0

    def _pain_index(self) -> float:
        """Average DD while underwater over uw_window (fraction)."""
        window_min = self.cfg.uw_window_ms / 60000.0
        return min(1.0, self._uw_area / max(1e-6, window_min))

    def _half_life_minutes(self, dd: float, speed_bps_min: float) -> Optional[float]:
        """
        If we immediately reversed slope to same magnitude *up*, estimate time
        (in minutes) to halve the current DD. Purely diagnostic; None if not in DD.
        """
        if dd <= 1e-12 or speed_bps_min <= 1e-6:
            return None
        # dd_halve_target = dd/2. If slope magnitude is s bps/min, time ≈ (dd/2)*1e4 / s
        return (dd * 0.5 * 1e4) / speed_bps_min

    def _alerts(self, dd: float, speed_bps_day: float) -> Tuple[List[str], str]:
        alerts: List[str] = []
        cfg = self.cfg
        regime = "calm"
        if speed_bps_day >= cfg.warn_speed_bps_per_day or dd >= cfg.max_dd_warn:
            regime = "caution"
            if speed_bps_day >= cfg.warn_speed_bps_per_day:
                alerts.append(f"Drawdown speed warning: {speed_bps_day:.1f} bps/day")
            if dd >= cfg.max_dd_warn:
                alerts.append(f"Drawdown warning: {dd*100:.2f}%")
        if speed_bps_day >= cfg.block_speed_bps_per_day or dd >= cfg.max_dd_block:
            regime = "critical"
            if speed_bps_day >= cfg.block_speed_bps_per_day:
                alerts.append(f"Drawdown speed CRITICAL: {speed_bps_day:.1f} bps/day")
            if dd >= cfg.max_dd_block:
                alerts.append(f"Drawdown CRITICAL: {dd*100:.2f}%")
        return alerts, regime

    # ------------- public API -------------
    def snapshot(self) -> Snapshot:
        return self._last_snap or Snapshot(
            ts_ms=_now_ms(), nav=self.cfg.base_nav, ret=0.0, dd=0.0, max_dd=0.0,
            speed_bps_min=0.0, speed_bps_day=0.0, peak_ts_ms=_now_ms(), trough_ts_ms=_now_ms(),
            time_under_water_min=0.0, pain_index=0.0, half_life_min=None, regime="calm", alerts=[]
        )

    def regime(self) -> str:
        return (self._last_snap.regime if self._last_snap else "calm")

    def should_block(self) -> bool:
        """Gate executions when in 'critical' regime."""
        s = self.snapshot()
        return s.regime == "critical"

    # ------------- utilities -------------
    def reset(self, nav: Optional[float] = None) -> None:
        self._ring.clear()
        base = float(nav if nav is not None else self.cfg.base_nav)
        now = _now_ms()
        self._ring.append((now, base))
        self._peak_nav, self._trough_nav = base, base
        self._peak_ts = self._trough_ts = now
        self._dd_fast_ewma = self._speed_ewma_min = self._speed_ewma_day = 0.0
        self._uw_area = 0.0
        self._uw_last_ts = None
        self._last_snap = None


# ------------------------------ demo ------------------------------
if __name__ == "__main__":
    dds = DrawdownSpeed(DDSConfig(base_nav=1_000_000.0))
    now = _now_ms()

    # Simulate a slow bleed, then an accelerated hit, then stabilization
    nav = 1_000_000.0
    for i in range(120):  # 2 hours, 1-min steps
        nav *= (1.0 - 0.0002)  # ~2 bps/min bleed
        dds.on_nav(nav, ts_ms=now + i * 60_000)

    for i in range(30):   # 30 mins fast hit
        nav *= (1.0 - 0.0015)  # 15 bps/min hit
        dds.on_nav(nav, ts_ms=now + (120 + i) * 60_000)

    for i in range(60):   # flat
        dds.on_nav(nav, ts_ms=now + (150 + i) * 60_000)

    snap = dds.snapshot()
    from pprint import pprint
    pprint(snap)