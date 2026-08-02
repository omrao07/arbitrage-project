"""
Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (CSCV) — Bailey, Borwein, Lopez de Prado, Zhu.

WHAT IT MEASURES
----------------
DSR asks "is THIS strategy's Sharpe real?". PBO asks a harder and more
uncomfortable question: **is my whole selection procedure any better than
picking at random?**

Method: split the return matrix (T observations x N candidates) into S even
blocks. For every way of choosing S/2 blocks as in-sample, the complement is
out-of-sample. Pick the best candidate in-sample, then look up where that
candidate ranks out-of-sample. If the in-sample winner routinely lands in the
bottom half out-of-sample, your ranking is noise.

    PBO = P(logit(relative OOS rank of the IS-best) <= 0)

PBO > 0.5 means the selection process is worse than random — retire the whole
family, not just the losers. Gate: PBO < 0.5.

WHY BOTH GATES
--------------
A family can contain one genuinely good strategy (high DSR) while the process
that found it is still garbage (high PBO). Passing DSR alone tells you a number
survived; passing both tells you the machine that produced it can be trusted to
produce more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Optional, Sequence

import numpy as np

PBO_GATE = 0.5

# S must be even (blocks split half in / half out) and large enough that
# C(S, S/2) gives a usable number of splits. S=16 -> 12,870 splits.
DEFAULT_N_SPLITS = 16
MAX_COMBINATIONS = 20_000


class PBOError(ValueError):
    """Raised when PBO cannot be computed honestly."""


@dataclass(frozen=True, slots=True)
class PBOResult:
    pbo: float
    n_splits: int
    n_candidates: int
    n_obs: int
    median_oos_rank: float
    logits: np.ndarray
    passed: bool

    def as_dict(self) -> dict:
        return {
            "pbo": round(self.pbo, 6),
            "n_splits": self.n_splits,
            "n_candidates": self.n_candidates,
            "n_obs": self.n_obs,
            "median_oos_rank": round(self.median_oos_rank, 4),
            "passed": self.passed,
        }


def _sharpe(x: np.ndarray) -> float:
    """Per-period Sharpe of one column; 0.0 for a degenerate (flat) series."""
    sd = x.std(ddof=1)
    return 0.0 if sd <= 0 else float(x.mean() / sd)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    performance_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> PBOResult:
    """
    CSCV PBO.

    `returns_matrix` is (T observations, N candidates) — one column per
    strategy variant, all evaluated over the SAME period. Columns must be
    genuine alternatives you would actually have chosen between; padding it
    with obviously-broken variants deflates PBO and flatters the process.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise PBOError(f"returns_matrix must be 2D (T, N), got shape {M.shape}")
    T, N = M.shape
    if N < 2:
        raise PBOError(f"PBO compares candidates against each other; need N >= 2, got {N}")
    if n_splits % 2 != 0:
        raise PBOError(f"n_splits must be even, got {n_splits}")
    if n_splits < 4:
        raise PBOError(f"n_splits must be >= 4 for a meaningful split count, got {n_splits}")
    if T < n_splits * 2:
        raise PBOError(
            f"need at least {n_splits * 2} observations for {n_splits} blocks, got {T}"
        )
    if not np.all(np.isfinite(M)):
        raise PBOError("returns_matrix contains non-finite values — clean or halt, do not impute")

    perf = performance_fn or _sharpe

    # Even blocks; drop the ragged tail so every block has equal weight.
    block_len = T // n_splits
    usable = block_len * n_splits
    blocks = [M[i * block_len : (i + 1) * block_len, :] for i in range(n_splits)]

    all_idx = set(range(n_splits))
    combos = list(combinations(range(n_splits), n_splits // 2))
    if len(combos) > MAX_COMBINATIONS:
        rng = np.random.default_rng(42)
        picks = rng.choice(len(combos), size=MAX_COMBINATIONS, replace=False)
        combos = [combos[i] for i in picks]

    logits: List[float] = []
    oos_ranks: List[float] = []

    for is_idx in combos:
        oos_idx = sorted(all_idx - set(is_idx))
        is_data = np.vstack([blocks[i] for i in is_idx])
        oos_data = np.vstack([blocks[i] for i in oos_idx])

        is_perf = np.array([perf(is_data[:, j]) for j in range(N)])
        oos_perf = np.array([perf(oos_data[:, j]) for j in range(N)])

        best_is = int(np.argmax(is_perf))

        # Relative rank of the IS winner within the OOS ranking, in (0, 1).
        # rank 1 = worst OOS, rank N = best OOS.
        order = np.argsort(oos_perf)
        rank_of = np.empty(N, dtype=float)
        rank_of[order] = np.arange(1, N + 1, dtype=float)
        omega = rank_of[best_is] / (N + 1.0)
        oos_ranks.append(omega)

        # logit: <= 0 means the IS winner was at or below the OOS median
        logits.append(math.log(omega / (1.0 - omega)))

    arr = np.asarray(logits, dtype=float)
    pbo = float(np.mean(arr <= 0.0))

    return PBOResult(
        pbo=pbo,
        n_splits=len(combos),
        n_candidates=N,
        n_obs=usable,
        median_oos_rank=float(np.median(oos_ranks)),
        logits=arr,
        passed=pbo < PBO_GATE,
    )


def pbo_from_trials(
    trial_returns: dict[str, Sequence[float] | np.ndarray],
    n_splits: int = DEFAULT_N_SPLITS,
) -> PBOResult:
    """
    Convenience wrapper: build the (T, N) matrix from a name->returns mapping.

    All series are truncated to the shortest common length, because CSCV
    requires every candidate to be scored over identical periods — comparing
    strategies across different windows is not a comparison.
    """
    if len(trial_returns) < 2:
        raise PBOError(f"need >= 2 candidates, got {len(trial_returns)}")

    series = {}
    for k, v in trial_returns.items():
        a = np.asarray(v, dtype=float).ravel()
        if not np.all(np.isfinite(a)):
            a = a[np.isfinite(a)]
        series[k] = a

    n = min(a.size for a in series.values())
    if n < n_splits * 2:
        raise PBOError(
            f"shortest track has {n} observations; need >= {n_splits * 2} for {n_splits} blocks"
        )

    matrix = np.column_stack([series[k][-n:] for k in sorted(series)])
    return probability_of_backtest_overfitting(matrix, n_splits=n_splits)
