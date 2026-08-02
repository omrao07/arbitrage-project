"""
The validation gauntlet — the machine that separates edge from noise.

    BACKTEST ──► WALK-FORWARD ──► SHADOW ──► LIVE (tiny)
    (real costs)  (DSR >= 0.95,   (>= 4wk,   (top-3 by
                   PBO < 0.5)      5 gates)   tournament)

No strategy touches real capital until it clears every gate in order. Expect
this to kill the overwhelming majority of what you feed it — that is the
pipeline working correctly, not failing.

Use `backend.costs.nse_fo` for the cost side; a gauntlet run against optimistic
costs just launders an overfit into a confident one.
"""

from backend.validation.dsr import (
    DSR_GATE,
    DSRError,
    DSRResult,
    deflated_sharpe_ratio,
    evaluate_trial_family,
    sr_benchmark_deflated,
)
from backend.validation.metrics import (
    InsufficientSampleError,
    PerformanceMetrics,
    calmar_ratio,
    compute_all,
    conditional_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from backend.validation.pbo import (
    PBO_GATE,
    PBOError,
    PBOResult,
    pbo_from_trials,
    probability_of_backtest_overfitting,
)

__all__ = [
    # metrics
    "PerformanceMetrics",
    "InsufficientSampleError",
    "compute_all",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "value_at_risk",
    "conditional_var",
    # deflated sharpe
    "DSRResult",
    "DSRError",
    "DSR_GATE",
    "deflated_sharpe_ratio",
    "sr_benchmark_deflated",
    "evaluate_trial_family",
    # overfitting probability
    "PBOResult",
    "PBOError",
    "PBO_GATE",
    "probability_of_backtest_overfitting",
    "pbo_from_trials",
]
