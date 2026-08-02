"""
Deflated Sharpe Ratio (Bailey & Lopez de Prado).

THE PROBLEM IT SOLVES
---------------------
Run 300 strategies and some will post a great Sharpe by luck alone. The DSR
corrects an observed Sharpe for three things a raw Sharpe ignores:
  (a) how many things you tried (multiple testing),
  (b) non-normal returns (skew and fat tails — short-vol P&L is both),
  (c) sample length.

It answers the only question that matters after a 300-candidate sweep:
*given that I tried this many things, is this Sharpe real, or is it the
luckiest of many coin flips?*

    DSR = Φ( (SR_hat - SR_0) * sqrt(T-1) / sqrt(1 - γ3*SR_hat + (γ4-1)/4 * SR_hat²) )

where SR_0 is the multiple-testing-adjusted benchmark:

    SR_0 = sqrt(V[{SR_m}]) * ( (1-γ)·Φ⁻¹(1 - 1/M) + γ·Φ⁻¹(1 - 1/(M·e)) )

with γ the Euler-Mascheroni constant and V[{SR_m}] the variance of Sharpes
across your M trials.

TWO RULES THAT ARE EASY TO GET WRONG
------------------------------------
1. `sr_hat` must be the PER-PERIOD Sharpe, not the annualised one. Feeding an
   annualised Sharpe here silently inflates DSR toward 1.0 and defeats the gate.
   `deflated_sharpe_ratio` takes raw returns and handles this for you.
2. Compute DSR on the CONCATENATED OUT-OF-SAMPLE walk-forward track, never on
   the in-sample fit. The in-sample Sharpe is marketing; the OOS DSR is the
   truth.

Gate: DSR >= 0.95.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_MASCHERONI = 0.5772156649015329

# DSR leans on third and fourth moments; those are unstable on short samples.
MIN_OBS_FOR_DSR = 30

DSR_GATE = 0.95


class DSRError(ValueError):
    """Raised when a DSR cannot be computed honestly."""


@dataclass(frozen=True, slots=True)
class DSRResult:
    dsr: float
    sr_hat_per_period: float
    sr_hat_annualized: float
    sr_benchmark: float
    skew: float
    kurtosis: float
    n_obs: int
    n_trials: int
    passed: bool

    def as_dict(self) -> dict:
        return {
            "dsr": round(self.dsr, 6),
            "sr_hat_per_period": round(self.sr_hat_per_period, 6),
            "sr_hat_annualized": round(self.sr_hat_annualized, 4),
            "sr_benchmark": round(self.sr_benchmark, 6),
            "skew": round(self.skew, 4),
            "kurtosis": round(self.kurtosis, 4),
            "n_obs": self.n_obs,
            "n_trials": self.n_trials,
            "passed": self.passed,
        }


def sr_benchmark_deflated(sr_variance_across_trials: float, n_trials: int) -> float:
    """
    The multiple-testing-adjusted benchmark SR_0.

    `sr_variance_across_trials` is the variance of the per-period Sharpes over
    all M candidates you tested — including the ones you threw away. Counting
    only the survivors is the most common way to cheat this gate.
    """
    if n_trials < 2:
        # With a single trial there is no selection bias to deflate away.
        return 0.0
    if sr_variance_across_trials < 0:
        raise DSRError(f"sr variance must be non-negative, got {sr_variance_across_trials}")
    if sr_variance_across_trials == 0:
        return 0.0

    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(
        math.sqrt(sr_variance_across_trials)
        * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    )


def deflated_sharpe_ratio(
    returns: Sequence[float] | np.ndarray,
    n_trials: int = 1,
    sr_variance_across_trials: Optional[float] = None,
    sr_benchmark: Optional[float] = None,
    periods_per_year: int = 252,
) -> DSRResult:
    """
    Compute the DSR for one candidate's OUT-OF-SAMPLE return track.

    Provide either `sr_variance_across_trials` (preferred — the variance of
    per-period Sharpes across all M candidates) or an explicit `sr_benchmark`.
    With neither, the benchmark is 0.0, which tests only "is the Sharpe
    positive" and applies NO multiple-testing correction.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    T = r.size
    if T < MIN_OBS_FOR_DSR:
        raise DSRError(
            f"DSR unstable on {T} observations (need >= {MIN_OBS_FOR_DSR}); "
            "skew/kurtosis estimates are meaningless on short samples"
        )

    # `sd <= 0` is not a sufficient guard: a constant float series has sd ~1e-19,
    # which sails through and yields a Sharpe of ~1e16. Compare against the
    # series' own scale instead.
    sd = float(r.std(ddof=1))
    scale = max(abs(float(r.mean())), float(np.max(np.abs(r))), 1e-300)
    if sd <= scale * 1e-12:
        raise DSRError(
            f"return series is effectively constant (sd={sd:.3g} vs scale={scale:.3g}) "
            "— Sharpe undefined, not infinite"
        )

    sr_hat = float(r.mean() / sd)  # PER-PERIOD, not annualised
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess

    if sr_benchmark is None:
        sr_benchmark = (
            sr_benchmark_deflated(sr_variance_across_trials, n_trials)
            if sr_variance_across_trials is not None
            else 0.0
        )

    # Variance of the Sharpe estimator under non-normality.
    denom_sq = 1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat**2
    if denom_sq <= 0:
        raise DSRError(
            f"non-normality correction is non-positive ({denom_sq:.4g}); "
            "the return distribution is too extreme for the DSR approximation"
        )

    z = (sr_hat - sr_benchmark) * math.sqrt(T - 1) / math.sqrt(denom_sq)
    dsr = float(norm.cdf(z))

    return DSRResult(
        dsr=dsr,
        sr_hat_per_period=sr_hat,
        sr_hat_annualized=sr_hat * math.sqrt(periods_per_year),
        sr_benchmark=float(sr_benchmark),
        skew=g3,
        kurtosis=g4,
        n_obs=T,
        n_trials=n_trials,
        passed=dsr >= DSR_GATE,
    )


def evaluate_trial_family(
    trial_returns: dict[str, Sequence[float] | np.ndarray],
    periods_per_year: int = 252,
) -> dict[str, DSRResult]:
    """
    Score a whole family of candidates against each other.

    This is the intended entry point after a sweep: it derives the Sharpe
    variance from ALL M trials — including the losers — so the benchmark
    reflects how wide a net you actually cast. Scoring survivors alone
    understates SR_0 and lets overfits through.
    """
    if not trial_returns:
        raise DSRError("no trials supplied")

    per_period: dict[str, float] = {}
    for name, rets in trial_returns.items():
        r = np.asarray(rets, dtype=float).ravel()
        r = r[np.isfinite(r)]
        if r.size < MIN_OBS_FOR_DSR:
            continue
        sd = r.std(ddof=1)
        if sd > 0:
            per_period[name] = float(r.mean() / sd)

    if not per_period:
        raise DSRError(
            f"no trial had >= {MIN_OBS_FOR_DSR} observations with non-zero variance"
        )

    m = len(trial_returns)
    sr_var = float(np.var(list(per_period.values()), ddof=1)) if len(per_period) > 1 else 0.0

    out: dict[str, DSRResult] = {}
    for name, rets in trial_returns.items():
        try:
            out[name] = deflated_sharpe_ratio(
                rets,
                n_trials=m,
                sr_variance_across_trials=sr_var,
                periods_per_year=periods_per_year,
            )
        except DSRError:
            # A candidate that cannot be scored has not passed; it is simply absent.
            continue
    return out
