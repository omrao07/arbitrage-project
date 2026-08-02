"""
Validation gauntlet tests: DSR and PBO must actually discriminate signal
from luck, and must refuse to score samples that cannot support the statistic.

The property that matters most is in
`test_luckiest_of_many_noise_trials_is_rejected`: run enough coin flips and one
will post a great-looking Sharpe. If that one passes the gate, the gate is
decoration and every downstream capital decision is being made on noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.validation import (
    DSR_GATE,
    PBO_GATE,
    DSRError,
    InsufficientSampleError,
    PBOError,
    compute_all,
    conditional_var,
    deflated_sharpe_ratio,
    evaluate_trial_family,
    max_drawdown,
    pbo_from_trials,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    sortino_ratio,
    sr_benchmark_deflated,
    value_at_risk,
)

T = 750


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# ── metrics: refuse small samples, never fabricate ──────────────────────────

@pytest.mark.parametrize("fn", [sharpe_ratio, sortino_ratio, max_drawdown])
def test_metrics_reject_tiny_samples(fn):
    with pytest.raises(InsufficientSampleError):
        fn(np.array([0.01, -0.01, 0.02]))


def test_var_requires_enough_points_for_the_quantile():
    """A 95% VaR needs >= 20 points before the quantile means anything."""
    with pytest.raises(InsufficientSampleError):
        value_at_risk(np.full(10, 0.001), alpha=0.05)


def test_constant_series_sharpe_is_undefined_not_astronomical():
    """
    Regression: a constant float array has sd ~1e-19, not 0.0. The old
    `sd <= 0` guard missed it and returned a Sharpe of ~7e16, which would top
    any tournament sorted by Sharpe.
    """
    with pytest.raises(InsufficientSampleError, match="effectively constant"):
        sharpe_ratio(np.full(100, 0.001))


def test_constant_series_sortino_is_undefined():
    with pytest.raises(InsufficientSampleError, match="effectively constant"):
        sortino_ratio(np.full(100, 0.001))


def test_constant_series_dsr_is_undefined():
    with pytest.raises(DSRError, match="effectively constant"):
        deflated_sharpe_ratio(np.full(100, 0.001))


def test_genuinely_low_vol_series_still_computes(rng):
    """The dispersion guard is relative, so a real low-vol series must survive."""
    r = rng.normal(1e-6, 1e-7, 500)
    assert np.isfinite(sharpe_ratio(r))


def test_cvar_exceeds_var(rng):
    r = rng.normal(0, 0.01, 2000)
    assert conditional_var(r, 0.05) > value_at_risk(r, 0.05)


def test_compute_all_returns_full_metric_set(rng):
    m = compute_all(rng.normal(0.0005, 0.01, T))
    d = m.as_dict()
    for key in ("sharpe", "sortino", "calmar", "max_drawdown", "var_95",
                "cvar_95", "volatility", "skew", "kurtosis", "n_obs"):
        assert key in d
    assert m.n_obs == T
    assert 0.0 <= m.max_drawdown <= 1.0


def test_sortino_exceeds_sharpe_for_profitable_skewed_returns(rng):
    """
    Downside deviation < total deviation, so for a PROFITABLE series Sortino
    reads higher than Sharpe. (The inequality flips when the mean is negative —
    a smaller denominator makes a negative ratio more negative — which is why
    this fixes the sign of the mean explicitly.)
    """
    raw = rng.lognormal(0, 0.01, T) - 1.0
    r = raw - raw.mean() + 0.0005     # de-mean, then impose a known positive drift
    assert r.mean() > 0
    assert sortino_ratio(r) > sharpe_ratio(r)


# ── DSR: the multiple-testing correction ────────────────────────────────────

def test_benchmark_rises_with_number_of_trials():
    prev = -1.0
    for m in (2, 10, 100, 300):
        sr0 = sr_benchmark_deflated(0.01, m)
        assert sr0 > prev
        prev = sr0


def test_single_trial_has_no_deflation():
    assert sr_benchmark_deflated(0.01, 1) == 0.0


def test_zero_dispersion_across_trials_gives_zero_benchmark():
    assert sr_benchmark_deflated(0.0, 300) == 0.0


def test_dsr_rejects_short_samples(rng):
    with pytest.raises(DSRError, match="unstable"):
        deflated_sharpe_ratio(rng.normal(0, 0.01, 25))


def test_dsr_uses_per_period_not_annualized_sharpe(rng):
    """Guards the classic error that silently inflates DSR toward 1.0."""
    r = rng.normal(0.0005, 0.01, T)
    res = deflated_sharpe_ratio(r, n_trials=1)
    assert res.sr_hat_annualized == pytest.approx(res.sr_hat_per_period * np.sqrt(252))
    assert abs(res.sr_hat_per_period) < abs(res.sr_hat_annualized)


def test_luckiest_of_many_noise_trials_is_rejected(rng):
    """
    THE test. 300 pure-noise strategies; the best-looking one posts a
    respectable annualized Sharpe purely by chance. None may pass.
    """
    trials = {f"s{i}": rng.normal(0, 0.01, T) for i in range(300)}
    results = evaluate_trial_family(trials)

    best = max(results.values(), key=lambda r: r.sr_hat_per_period)
    assert best.sr_hat_annualized > 1.0, "expected some lucky noise to look good"
    assert not best.passed, "luck must not clear the DSR gate"
    assert sum(r.passed for r in results.values()) == 0


def test_strong_genuine_signal_can_pass_the_gate(rng):
    """The gate must be hard, not impossible — otherwise it is useless."""
    trials = {f"s{i}": rng.normal(0, 0.01, T) for i in range(300)}
    trials["real"] = rng.normal(3.5 / np.sqrt(252) * 0.01, 0.01, T)
    res = evaluate_trial_family(trials)["real"]
    assert res.dsr >= DSR_GATE and res.passed


def test_more_trials_makes_the_same_track_harder_to_pass(rng):
    r = rng.normal(0.0008, 0.01, T)
    few = deflated_sharpe_ratio(r, n_trials=5, sr_variance_across_trials=0.001)
    many = deflated_sharpe_ratio(r, n_trials=500, sr_variance_across_trials=0.001)
    assert many.dsr < few.dsr


def test_evaluate_trial_family_needs_trials():
    with pytest.raises(DSRError):
        evaluate_trial_family({})


# ── PBO: is the selection process better than random? ───────────────────────

def test_pbo_flags_noise_family_as_overfit(rng):
    noise = {f"s{i}": rng.normal(0, 0.01, 600) for i in range(10)}
    res = pbo_from_trials(noise, n_splits=10)
    assert res.pbo > PBO_GATE
    assert not res.passed


def test_pbo_accepts_a_genuinely_dominant_strategy(rng):
    trials = {f"s{i}": rng.normal(0, 0.01, 600) for i in range(9)}
    trials["winner"] = rng.normal(0.0015, 0.01, 600)
    res = pbo_from_trials(trials, n_splits=10)
    assert res.pbo < PBO_GATE
    assert res.passed
    assert res.median_oos_rank > 0.5


def test_pbo_is_a_probability(rng):
    trials = {f"s{i}": rng.normal(0, 0.01, 400) for i in range(6)}
    res = pbo_from_trials(trials, n_splits=8)
    assert 0.0 <= res.pbo <= 1.0
    assert len(res.logits) == res.n_splits


def test_pbo_requires_at_least_two_candidates(rng):
    with pytest.raises(PBOError, match="N >= 2|>= 2 candidates"):
        pbo_from_trials({"only": rng.normal(0, 0.01, 400)})


def test_pbo_rejects_odd_split_count(rng):
    m = rng.normal(0, 0.01, (400, 4))
    with pytest.raises(PBOError, match="even"):
        probability_of_backtest_overfitting(m, n_splits=7)


def test_pbo_rejects_too_few_observations(rng):
    m = rng.normal(0, 0.01, (10, 4))
    with pytest.raises(PBOError, match="at least"):
        probability_of_backtest_overfitting(m, n_splits=16)


def test_pbo_rejects_non_finite_input():
    m = np.full((400, 3), 0.01)
    m[5, 1] = np.nan
    with pytest.raises(PBOError, match="non-finite"):
        probability_of_backtest_overfitting(m, n_splits=8)


def test_pbo_rejects_1d_input(rng):
    with pytest.raises(PBOError, match="2D"):
        probability_of_backtest_overfitting(rng.normal(0, 0.01, 400), n_splits=8)


# ── Both gates together ─────────────────────────────────────────────────────

def test_a_noise_family_fails_both_gates(rng):
    """
    The realistic outcome of a 300-candidate sweep with no real edge: nothing
    passes DSR, and PBO says the ranking itself is worthless.
    """
    trials = {f"s{i}": rng.normal(0, 0.01, 600) for i in range(12)}
    dsr_results = evaluate_trial_family(trials)
    assert sum(r.passed for r in dsr_results.values()) == 0
    assert not pbo_from_trials(trials, n_splits=10).passed
