# backend/risk/drawdown_speed.py
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------------------------------------------
# Online rolling regression (least-squares) over a sliding window
#   slope is ∂y/∂x (we'll feed x in minutes for numerical stability)
# ------------------------------------------------------------

class RollingOLS:
    """
    Sliding-window simple linear regression with O(1) updates.
    Add (x,y); pop from the left when over capacity.
    slope() and intercept() computed from cached sums.
    """
    def __init__(self, maxlen: int):
        if maxlen < 2:
            raise ValueError("RollingOLS maxlen must be >= 2")
        self.maxlen = int(maxlen)
        self.q: Deque[Tuple[float, float]] = deque()
        self.sx = self.sy = self.sxx = self.sxy = 0.0

    def __len__(self) -> int:
        return len(self.q)

    def add(self, x: float, y: float) -> None:
        self.q.append((x, y))
        self.sx  += x
        self.sy  += y
        self.sxx += x * x
        self.sxy += x * y
        if len(self.q) > self.maxlen:
            ox, oy = self.q.popleft()
            self.sx  -= ox
            self.sy  -= oy
            self.sxx -= ox * ox
            self.sxy -= ox * oy

    def slope(self) -> Optional[float]:
        n = len(self.q)
        if n < 2:
            return None
        den = n * self.sxx - self.sx * self.sx
        if abs(den) < 1e-12:
            return 0.0
        return (n * self.sxy - self.sx * self.sy) / den

    def intercept(self) -> Optional[float]:
        n = len(self.q)
        if n < 2:
            return None
        m = self.slope() or 0.0
        return (self.sy - m * self.sx) / n

# ------------------------------------------------------------
# Rolling volatility of log returns (for normalization)
# ------------------------------------------------------------

class RollingVol:
    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self.q: Deque[float] = deque()
        self.sum = 0.0
        self.sumsq = 0.0

    def add(self, r: float) -> None:
        self.q.append(r)
        self.sum += r
        self.sumsq += r * r
        if len(self.q) > self.maxlen:
            o = self.q.popleft()
            self.sum -= o
            self.sumsq -= o * o

    def stdev(self) -> Optional[float]:
        n = len(self.q)
        if n < 2:
            return None
        mean = self.sum / n
        var = max(0.0, self.sumsq / n - mean * mean)
        return math.sqrt(var)

# ------------------------------------------------------------
# Public data structures
# ------------------------------------------------------------

@dataclass
class DrawdownPoint:
    ts_ms: int
    nav: float
    peak_nav: float
    dd_frac: float             # current drawdown, fraction of peak (e.g., 0.06 = 6%)
    dd_speed_per_min: float    # slope of dd vs time (fraction per minute, + means worsening)
    dd_accel_per_min2: float   # acceleration of dd (per minute^2)
    z_speed: Optional[float] = None  # speed normalized by return vol (unitless)

@dataclass
class DrawdownAlert:
    level: str                 # "ok" | "info" | "warn" | "block" | "halt"
    reason: str
    ts_ms: int
    dd_frac: float
    dd_speed_per_min: float
    dd_accel_per_min2: float
    z_speed: Optional[float]
    suggested_risk_mult: float # multiply gross exposure by this (0..1)
    meta: Dict[str, float]

# ------------------------------------------------------------
# DrawdownSpeed tracker
# ------------------------------------------------------------

class DrawdownSpeed:
    """
    Tracks running peak, drawdown, *speed* (slope of drawdown over time), and *acceleration*.
    Emits graded alerts and a suggested risk multiplier.

    Units:
      • dd_frac                 = fraction (0.08 = 8% below peak)
      • dd_speed_per_min        = fraction per minute (0.01/min = 1% extra DD per minute)
      • dd_accel_per_min2       = fraction per minute^2
      • z_speed                 = speed normalized by rolling return stdev

    Typical wiring:
      tracker = DrawdownSpeed()
      for (ts, nav) in equity_curve:
          pt, alert = tracker.update(ts, nav)
          if alert and alert.level in ("block","halt"): trigger_kill_switch()
    """
    def __init__(
        self,
        *,
        speed_window: int = 15,       # samples for slope regression
        accel_window: int = 6,        # samples for speed acceleration (slope-of-slope)
        vol_window: int = 60,         # samples for return vol
        min_dt_ms: int = 30_000,      # ignore updates faster than this to reduce noise
        base_ts_ms: Optional[int] = None,
        risk_cap: float = 0.9         # max de-risking (up to 90% cut if worst)
    ):
        if speed_window < 3:
            raise ValueError("speed_window must be >= 3")
        self.min_dt_ms = int(min_dt_ms)
        self.risk_cap = float(risk_cap)
        self.base_ts_ms = int(base_ts_ms or now_ms())

        self.peak_nav: Optional[float] = None
        self.last_ts: Optional[int] = None
        self.last_nav: Optional[float] = None

        # regression of drawdown vs time (x = minutes since base)
        self.reg = RollingOLS(speed_window)
        # regression of speed vs time (acceleration)
        self.reg_accel = RollingOLS(accel_window)
        # rolling vol of log returns
        self.vol = RollingVol(vol_window)

        self._last_speed: Optional[float] = None  # per minute

    # ---------------- core update ----------------

    def update(self, ts_ms: int, nav: float) -> Tuple[DrawdownPoint, Optional[DrawdownAlert]]:
        if nav <= 0:
            raise ValueError("NAV must be positive")

        # throttle updates
        if self.last_ts is not None and (ts_ms - self.last_ts) < self.min_dt_ms:
            # still update peak and vol but don't add regression point
            self._update_state_only(ts_ms, nav)
            pt = self._snapshot(ts_ms, nav)
            return pt, None

        # initialize state
        if self.peak_nav is None:
            self.peak_nav = nav
        if self.last_ts is None:
            self.last_ts = ts_ms
            self.last_nav = nav

        # update peak
        if nav > (self.peak_nav or 0.0):
            self.peak_nav = nav

        # compute drawdown
        peak = float(self.peak_nav or nav)
        dd = clamp((peak - nav) / max(1e-12, peak), 0.0, 1.0)

        # x in minutes since base for numerical stability
        x_min = (ts_ms - self.base_ts_ms) / 60_000.0
        self.reg.add(x_min, dd)

        # instantaneous speed estimate (finite diff, per minute)
        dt_min = max(1e-9, (ts_ms - (self.last_ts or ts_ms)) / 60_000.0)
        prev_dd = clamp((peak - (self.last_nav or nav)) / max(1e-12, peak), 0.0, 1.0)
        inst_speed = (dd - prev_dd) / dt_min

        # regression speed (smoother)
        slope = self.reg.slope()
        dd_speed = float(slope if slope is not None else inst_speed)

        # acceleration via second regression on speeds
        self.reg_accel.add(x_min, dd_speed)
        acc_slope = self.reg_accel.slope()
        dd_accel = float(acc_slope if acc_slope is not None else 0.0)

        # rolling return vol (log returns)
        if self.last_nav:
            lr = math.log(nav / self.last_nav)
            self.vol.add(lr)
        vol = self.vol.stdev()
        z_speed = None
        if vol is not None and vol > 1e-9:
            # normalize speed by per-minute expected move ≈ vol * sqrt(freq)
            # If samples are ~1/min, this is a rough z-like score for speed.
            z_speed = dd_speed / (vol + 1e-9)

        # persist for next step
        self.last_ts = ts_ms
        self.last_nav = nav
        self._last_speed = dd_speed

        # snapshot + alert
        pt = DrawdownPoint(
            ts_ms=ts_ms, nav=nav, peak_nav=peak,
            dd_frac=dd, dd_speed_per_min=dd_speed, dd_accel_per_min2=dd_accel,
            z_speed=z_speed
        )
        alert = self._classify(pt)
        return pt, alert

    # ---------------- helpers ----------------

    def _update_state_only(self, ts_ms: int, nav: float) -> None:
        if self.peak_nav is None or nav > self.peak_nav:
            self.peak_nav = nav
        if self.last_nav:
            lr = math.log(nav / self.last_nav)
            self.vol.add(lr)
        self.last_ts = ts_ms
        self.last_nav = nav

    def _snapshot(self, ts_ms: int, nav: float) -> DrawdownPoint:
        peak = float(self.peak_nav or nav)
        dd = clamp((peak - nav) / max(1e-12, peak), 0.0, 1.0)
        spd = self._last_speed or 0.0
        # acceleration estimate from reg_accel even if we didn't add a point now
        acc = self.reg_accel.slope() or 0.0
        vol = self.vol.stdev()
        z = (spd / (vol + 1e-9)) if (vol and vol > 1e-9) else None
        return DrawdownPoint(ts_ms=ts_ms, nav=nav, peak_nav=peak, dd_frac=dd,
                             dd_speed_per_min=spd, dd_accel_per_min2=acc, z_speed=z)

    # Rule set for alerts (tunable)
    def _classify(self, pt: DrawdownPoint) -> Optional[DrawdownAlert]:
        dd = pt.dd_frac
        spd = pt.dd_speed_per_min
        acc = pt.dd_accel_per_min2
        z   = pt.z_speed if pt.z_speed is not None else 0.0

        level = "ok"
        reason = "normal"

        # Progressive thresholds (conservative defaults).
        # You can tighten/loosen to match your risk appetite.
        if dd >= 0.12 or spd >= 0.020 or z >= 6.0:
            level, reason = "halt", "extreme drawdown or speed"
        elif dd >= 0.08 or spd >= 0.012 or z >= 4.0:
            level, reason = "block", "fast drawdown regime"
        elif dd >= 0.05 or spd >= 0.008 or z >= 3.0:
            level, reason = "warn", "elevated drawdown speed"
        elif dd >= 0.03 or spd >= 0.004 or z >= 2.0:
            level, reason = "info", "early drawdown deterioration"

        # Acceleration kicker: if drawdown is speeding up rapidly, escalate one level
        if acc >= 0.004 and level in ("ok", "info"):
            level = "warn"
            reason = "acceleration in drawdown"

        suggested_mult = self._suggest_risk_multiplier(dd, spd, acc, z)

        if level == "ok":
            return None
        return DrawdownAlert(
            level=level,
            reason=reason,
            ts_ms=pt.ts_ms,
            dd_frac=dd,
            dd_speed_per_min=spd,
            dd_accel_per_min2=acc,
            z_speed=pt.z_speed,
            suggested_risk_mult=suggested_mult,
            meta={
                "peak_nav": pt.peak_nav,
                "nav": pt.nav
            }
        )

    def _suggest_risk_multiplier(self, dd: float, spd: float, acc: float, z: float) -> float:
        """
        Heuristic de-risking curve:
          - base cut grows with dd (up to risk_cap)
          - extra cut when speed & accel are high
          - normalize by z (vol-adjusted speed) to avoid overreacting in calm periods
        """
        base_cut = clamp(1.5 * dd, 0.0, self.risk_cap)              # e.g., 10% DD -> up to 15% cut
        speed_cut = clamp(10.0 * spd, 0.0, self.risk_cap)           # 1%/min -> +10% cut
        accel_cut = clamp(50.0 * max(0.0, acc), 0.0, self.risk_cap) # accel kicker
        z_cut     = clamp(0.05 * max(0.0, z - 2.0), 0.0, self.risk_cap)

        cut = clamp(base_cut + speed_cut + accel_cut + z_cut, 0.0, self.risk_cap)
        return clamp(1.0 - cut, 0.0, 1.0)

    # convenient dump
    def state(self) -> Dict[str, float]:
        return {
            "peak_nav": float(self.peak_nav or 0.0),
            "last_nav": float(self.last_nav or 0.0),
            "last_ts": float(self.last_ts or 0.0),
        }

# ------------------------------------------------------------
# Batch helpers
# ------------------------------------------------------------

def compute_series(points: List[Tuple[int, float]], *, speed_window: int = 15,
                   accel_window: int = 6, vol_window: int = 60) -> Tuple[List[DrawdownPoint], List[DrawdownAlert]]:
    """
    Convenience to run the tracker over a full NAV time series.
    Returns (points, alerts).
    """
    if not points:
        return [], []
    base = points[0][0]
    tracker = DrawdownSpeed(speed_window=speed_window, accel_window=accel_window,
                            vol_window=vol_window, base_ts_ms=base)
    out_pts: List[DrawdownPoint] = []
    out_alerts: List[DrawdownAlert] = []
    for ts, nav in points:
        pt, alert = tracker.update(ts, nav)
        out_pts.append(pt)
        if alert:
            out_alerts.append(alert)
    return out_pts, out_alerts

# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import random

    # Synthetic equity curve: steady, then crash, then recovery
    start = now_ms()
    nav = 1_000_000.0
    series: List[Tuple[int, float]] = []
    # flat-ish 30 mins
    for i in range(30):
        nav *= (1.0 + random.uniform(-0.0005, 0.0005))
        series.append((start + i * 60_000, nav))
    # sharp 10% drawdown over ~8 mins
    for i in range(8):
        nav *= (1.0 - 0.0125 + random.uniform(-0.0005, 0.0005))
        series.append((start + (30 + i) * 60_000, nav))
    # chop + slow recovery
    for i in range(40):
        nav *= (1.0 + random.uniform(-0.0008, 0.0012))
        series.append((start + (38 + i) * 60_000, nav))

    pts, alerts = compute_series(series)
    print(f"points={len(pts)} alerts={len(alerts)}")
    if alerts:
        print("first alert:", alerts[0].level, alerts[0].reason,
              "dd=", round(alerts[0].dd_frac*100,2), "%",
              "speed=", round(alerts[0].dd_speed_per_min*100,3), "%/min",
              "mult=", alerts[0].suggested_risk_mult)