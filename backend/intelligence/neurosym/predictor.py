# backend/common/predictor.py
"""
Tiny forecasting toolkit (stdlib-only; NumPy optional).

What you get
------------
- OnlineStats: running mean/var/std
- RollingWindow: fixed-size deque with helpers
- EMA / EWMAVol: exponential moving average & vol
- ZScore: rolling z-score
- AR1Predictor: AR(1) (with/without NumPy) + one-step forecast
- Kalman1D: simple 1D Kalman filter (level/trend-lite)
- RegimeSwitch: two-regime detector by volatility thresholds
- FeatureScaler: mean/std scaler (fit/transform/invert)
- EnsemblePredictor: weighted blend of child predictors
- SignalPredictor: convenience wrapper to turn prices -> signals
  (returns dict of {feature_name: value} and an optional forecast)

Design
------
- No heavy ML deps; pure math. If NumPy is present, it's used for
  least-squares; otherwise fallback to hand-rolled formulas.
- Stateless predictors (predict), or stateful (update -> predict).
- Safe defaults; all functions validate inputs.

Use quickly
-----------
from backend.common.predictor import (
    EMA, EWMAVol, ZScore, AR1Predictor, SignalPredictor
)

ema = EMA(alpha=0.2)
for px in [100, 101, 100.5, 102]:
    ema.update(px)
print("EMA:", ema.value)

ar1 = AR1Predictor()
for x in [0.0, 0.2, 0.1, 0.3, 0.25]:
    ar1.update(x)
print("AR1 φ:", ar1.phi, "μ:", ar1.mu, "→ next:", ar1.predict())
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Optional NumPy (used if available)
try:
    import numpy as _np  # type: ignore
    _HAVE_NP = True
except Exception:
    _HAVE_NP = False


# ---------------------------------------------------------------------
# Online stats & rolling utilities
# ---------------------------------------------------------------------

@dataclass
class OnlineStats:
    """Welford running mean/variance."""
    n: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        v = self.var
        return math.sqrt(v) if v > 0 else 0.0

    def reset(self) -> None:
        self.n = 0; self.mean = 0.0; self._m2 = 0.0


@dataclass
class RollingWindow:
    """Fixed-length window with handy reducers."""
    size: int
    _dq: Deque[float] = field(default_factory=deque, init=False)

    def push(self, x: float) -> None:
        self._dq.append(float(x))
        while len(self._dq) > self.size:
            self._dq.popleft()

    def values(self) -> List[float]:
        return list(self._dq)

    def ready(self) -> bool:
        return len(self._dq) >= self.size

    def mean(self) -> float:
        v = self.values()
        return sum(v) / len(v) if v else 0.0

    def std(self, ddof: int = 1) -> float:
        v = self.values()
        n = len(v)
        if n <= ddof:
            return 0.0
        mu = sum(v) / n
        s2 = sum((x - mu) ** 2 for x in v) / (n - ddof)
        return math.sqrt(max(0.0, s2))

    def last(self) -> Optional[float]:
        return self._dq[-1] if self._dq else None


# ---------------------------------------------------------------------
# Basic indicators
# ---------------------------------------------------------------------

@dataclass
class EMA:
    """Exponential Moving Average; alpha in (0,1]."""
    alpha: float = 0.2
    value: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            a = max(1e-9, min(1.0, self.alpha))
            self.value = a * x + (1.0 - a) * self.value
        return self.value

@dataclass
class EWMAVol:
    """Exponentially weighted volatility (std of returns)."""
    alpha: float = 0.2
    last: Optional[float] = None
    ema_ret2: Optional[float] = None

    def update(self, price: float) -> float:
        p = float(price)
        if self.last is None:
            self.last = p
            return 0.0
        r = (p - self.last) / max(1e-12, self.last)
        self.last = p
        a = max(1e-9, min(1.0, self.alpha))
        r2 = r * r
        self.ema_ret2 = r2 if self.ema_ret2 is None else a * r2 + (1.0 - a) * self.ema_ret2
        return math.sqrt(max(0.0, self.ema_ret2 or 0.0))

@dataclass
class ZScore:
    """Rolling z-score of value vs rolling mean/std."""
    lookback: int = 20
    _win: RollingWindow = field(init=False)

    def __post_init__(self) -> None:
        self._win = RollingWindow(self.lookback)

    def update(self, x: float) -> float:
        self._win.push(float(x))
        if not self._win.ready():
            return 0.0
        mu = self._win.mean()
        sd = self._win.std()
        if sd <= 0:
            return 0.0
        return (self._win.last() - mu) / sd # type: ignore


# ---------------------------------------------------------------------
# Simple models
# ---------------------------------------------------------------------

@dataclass
class AR1Predictor:
    """
    AR(1): x_t = μ + φ (x_{t-1} - μ) + ε_t
    Estimates μ, φ online by least squares on a rolling window.
    """
    lookback: int = 100
    mu: float = 0.0
    phi: float = 0.0
    _win: RollingWindow = field(init=False)

    def __post_init__(self) -> None:
        self._win = RollingWindow(self.lookback)

    def update(self, x: float) -> Tuple[float, float]:
        self._win.push(float(x))
        xs = self._win.values()
        if len(xs) < 3:
            self.mu, self.phi = (xs[-1] if xs else 0.0), 0.0
            return self.mu, self.phi
        # Estimate μ as sample mean over window
        mu = sum(xs) / len(xs)
        # Estimate φ via OLS on deviations
        x0 = [xi - mu for xi in xs[:-1]]
        x1 = [xi - mu for xi in xs[1:]]
        if _HAVE_NP:
            X = _np.array(x0)
            Y = _np.array(x1)
            denom = float((X * X).sum())
            phi = float((X * Y).sum() / denom) if denom > 0 else 0.0
        else:
            denom = sum(z * z for z in x0)
            num = sum(a * b for a, b in zip(x0, x1))
            phi = (num / denom) if denom > 1e-12 else 0.0
        # Clamp φ to stationary region
        self.mu = mu
        self.phi = max(-0.999, min(0.999, phi))
        return self.mu, self.phi

    def predict(self, last: Optional[float] = None, steps: int = 1) -> float:
        """
        One-step (or multi-step) ahead forecast. If last None, uses window last.
        """
        if last is None:
            last = self._win.last()
        x = float(last if last is not None else self.mu)
        phi = self.phi
        mu = self.mu
        # multi-step closed form: μ + φ^k (x-μ)
        k = max(1, int(steps))
        return mu + (phi ** k) * (x - mu)


@dataclass
class Kalman1D:
    """
    Minimal 1D Kalman filter (constant level with white noise).
    state: level L
    """
    process_var: float = 1e-4    # Q
    meas_var: float = 1e-2       # R
    L: Optional[float] = None
    P: float = 1.0               # covariance

    def update(self, z: float) -> float:
        # Predict
        if self.L is None:
            self.L = float(z)
        # P_pred = P + Q
        Pp = self.P + self.process_var
        # Update
        K = Pp / (Pp + self.meas_var)
        self.L = self.L + K * (float(z) - self.L)
        self.P = (1 - K) * Pp
        return float(self.L)

    def value(self) -> Optional[float]:
        return self.L


# ---------------------------------------------------------------------
# Regime & scaling
# ---------------------------------------------------------------------

@dataclass
class RegimeSwitch:
    """
    Two-regime classifier by volatility:
    - low_vol if vol <= th_lo
    - high_vol if vol >= th_hi
    - sticky (hysteresis) in between using last regime
    """
    th_lo: float
    th_hi: float
    current: str = "low"

    def update(self, vol: float) -> str:
        v = float(vol)
        if v >= self.th_hi:
            self.current = "high"
        elif v <= self.th_lo:
            self.current = "low"
        return self.current


@dataclass
class FeatureScaler:
    """Mean/std standardizer (fit on-the-fly)."""
    fitted: bool = False
    mean: float = 0.0
    std: float = 1.0
    _stats: OnlineStats = field(default_factory=OnlineStats)

    def partial_fit(self, x: float) -> None:
        self._stats.update(x)
        if self._stats.n >= 5:
            self.mean = self._stats.mean
            self.std = self._stats.std or 1.0
            self.fitted = True

    def transform(self, x: float) -> float:
        if not self.fitted:
            return float(x)
        return (float(x) - self.mean) / (self.std or 1.0)

    def inverse(self, z: float) -> float:
        if not self.fitted:
            return float(z)
        return self.mean + float(z) * (self.std or 1.0)


# ---------------------------------------------------------------------
# Ensembles & convenience
# ---------------------------------------------------------------------

class PredictorBase:
    """Interface: update(x) -> None, predict() -> float"""
    def update(self, x: float) -> None: raise NotImplementedError
    def predict(self) -> float: raise NotImplementedError

@dataclass
class EnsemblePredictor(PredictorBase):
    """
    Weighted blend of child predictors' predictions.
    Children must implement predict(); weights auto-normalized.
    """
    children: List[PredictorBase]
    weights: Optional[List[float]] = None

    def update(self, x: float) -> None:
        for c in self.children:
            c.update(x)

    def predict(self) -> float:
        preds = [c.predict() for c in self.children]
        if not preds:
            return 0.0
        w = self.weights or [1.0] * len(preds)
        if len(w) != len(preds):
            raise ValueError("weights length mismatch")
        s = sum(max(0.0, float(a)) for a in w)
        w = [max(0.0, float(a)) / (s or 1.0) for a in w]
        return sum(pi * wi for pi, wi in zip(preds, w))


# ---------------------------------------------------------------------
# High-level convenience: prices -> signals -> forecast
# ---------------------------------------------------------------------

@dataclass
class SignalPredictor:
    """
    Convenience wrapper that turns a price stream into:
      - technical features: returns, ema, vol, zscore
      - forecast: AR(1) of returns (optionally Kalman-smoothed level)
    Use it to feed agents or as a sanity predictor baseline.

    set use_kalman=True to smooth price before features.
    """
    ema_alpha: float = 0.2
    vol_alpha: float = 0.2
    z_lookback: int = 20
    ar_lookback: int = 100
    use_kalman: bool = False

    _ema: EMA = field(init=False)
    _vol: EWMAVol = field(init=False)
    _z: ZScore = field(init=False)
    _ar: AR1Predictor = field(init=False)
    _k: Optional[Kalman1D] = field(init=False, default=None)
    _last_price: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._ema = EMA(self.ema_alpha)
        self._vol = EWMAVol(self.vol_alpha)
        self._z = ZScore(self.z_lookback)
        self._ar = AR1Predictor(self.ar_lookback)
        self._k = Kalman1D() if self.use_kalman else None

    def update(self, price: float) -> Dict[str, float]:
        p = float(price)
        if self._k is not None:
            p_eff = self._k.update(p)
        else:
            p_eff = p

        ema = self._ema.update(p_eff)
        vol = self._vol.update(p_eff)
        z = self._z.update(p_eff)

        # simple return (vs prev effective price)
        ret = 0.0
        if self._last_price is not None:
            ret = (p_eff - self._last_price) / max(1e-12, self._last_price)
        self._last_price = p_eff

        # AR(1) on returns (centered), not prices
        self._ar.update(ret)
        fwd_ret = self._ar.predict(last=ret, steps=1)  # next-step return forecast

        return {
            "price": p_eff,
            "ema": float(ema),
            "vol": float(vol),
            "zscore": float(z),
            "ret": float(ret),
            "ret_ar1_forecast": float(fwd_ret),
        }


# ---------------------------------------------------------------------
# Tiny CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic stream demo
    import random

    sp = SignalPredictor(use_kalman=True)
    xs = []
    for t in range(80):
        # mean-reverting price around 100 with small noise
        base = 100.0 + 2.0 * math.sin(t / 8.0)
        price = base + random.gauss(0, 0.6)
        feats = sp.update(price)
        xs.append((price, feats["ret_ar1_forecast"]))

    print("[predictor] last features:", feats)
    print("[predictor] sample of last 5 (price, fwd_ret_pred):", xs[-5:])