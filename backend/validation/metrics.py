"""
Performance metrics for the validation gauntlet.

Every metric here refuses to compute on a sample too small to support it. A
Sharpe ratio from 8 observations is not a weak signal, it is noise with a
decimal point, and the whole purpose of this pipeline is to stop treating noise
as evidence.

For short-vol books, CVaR — not Sharpe — is the number that decides size.
Sharpe rewards steady penny-collection and hides the steamroller: a strategy
can post a great Sharpe and still hold a tail that ends the account on one
expiry gap.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

MIN_OBS = int(os.getenv("VALIDATION_MIN_OBS", "20"))
TRADING_DAYS = 252


class InsufficientSampleError(ValueError):
    """Raised when a statistic is requested on a sample that cannot support it."""


def _clean(returns, minimum: int = MIN_OBS, what: str = "metric") -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < minimum:
        raise InsufficientSampleError(
            f"{what}: need >= {minimum} finite observations, got {arr.size} "
            f"— do not trust this result"
        )
    return arr


def _require_dispersion(r: np.ndarray, sd: float, what: str) -> None:
    """
    Reject an effectively-constant series.

    Testing `sd <= 0` is not enough: a constant float array has a standard
    deviation of ~1e-19 rather than exactly 0, so the naive guard passes and the
    ratio explodes to ~1e16. That number is not a strong signal, it is a
    division artefact — and it would top any tournament ranking sorted by Sharpe.
    """
    scale = max(abs(float(r.mean())), float(np.max(np.abs(r))), 1e-300)
    if sd <= scale * 1e-12:
        raise InsufficientSampleError(
            f"{what}: return series is effectively constant "
            f"(sd={sd:.3g} vs scale={scale:.3g}) — undefined, not infinite"
        )


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    volatility: float
    total_return: float
    skew: float
    kurtosis: float
    n_obs: int

    def as_dict(self) -> dict:
        return {
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "calmar": round(self.calmar, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "var_95": round(self.var_95, 6),
            "cvar_95": round(self.cvar_95, 6),
            "volatility": round(self.volatility, 4),
            "total_return": round(self.total_return, 4),
            "skew": round(self.skew, 4),
            "kurtosis": round(self.kurtosis, 4),
            "n_obs": self.n_obs,
        }


def sharpe_ratio(returns, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    r = _clean(returns, what="sharpe")
    excess = r - rf / periods_per_year
    sd = float(excess.std(ddof=1))
    _require_dispersion(r, sd, "sharpe")
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    """Downside deviation only — the right measure for negatively-skewed P&L."""
    r = _clean(returns, what="sortino")
    excess = r - rf / periods_per_year
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float((downside**2).mean()))
    _require_dispersion(r, dd, "sortino: downside deviation")
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    """Peak-to-trough of the compounded equity curve, as a positive fraction."""
    r = _clean(returns, what="max_drawdown")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def value_at_risk(returns, alpha: float = 0.05) -> float:
    """Historical VaR as a positive loss fraction."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    need = max(MIN_OBS, int(math.ceil(1.0 / alpha)))
    r = _clean(returns, minimum=need, what=f"var_{1 - alpha:.0%}")
    return float(-np.quantile(r, alpha))


def conditional_var(returns, alpha: float = 0.05) -> float:
    """
    CVaR / Expected Shortfall: mean loss in the tail beyond VaR.
    This is the number you size on, not Sharpe.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    need = max(MIN_OBS, int(math.ceil(1.0 / alpha)))
    r = _clean(returns, minimum=need, what=f"cvar_{1 - alpha:.0%}")
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return float(-tail.mean()) if tail.size else float(-q)


def calmar_ratio(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns, what="calmar")
    mdd = max_drawdown(r)
    if mdd <= 0:
        raise InsufficientSampleError("calmar: no drawdown in sample — undefined, not infinite")
    return float(r.mean() * periods_per_year / mdd)


def compute_all(
    returns,
    periods_per_year: int = TRADING_DAYS,
    rf: float = 0.0,
    alpha: float = 0.05,
) -> PerformanceMetrics:
    """Full metric set. Raises rather than emitting a partially-fabricated row."""
    from scipy import stats as _st

    r = _clean(returns, what="performance_metrics")
    return PerformanceMetrics(
        sharpe=sharpe_ratio(r, periods_per_year, rf),
        sortino=sortino_ratio(r, periods_per_year, rf),
        calmar=calmar_ratio(r, periods_per_year),
        max_drawdown=max_drawdown(r),
        var_95=value_at_risk(r, alpha),
        cvar_95=conditional_var(r, alpha),
        volatility=float(r.std(ddof=1) * math.sqrt(periods_per_year)),
        total_return=float(np.prod(1.0 + r) - 1.0),
        skew=float(_st.skew(r)),
        kurtosis=float(_st.kurtosis(r, fisher=False)),
        n_obs=int(r.size),
    )
