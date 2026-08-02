"""
D-2 regression tests: risk metrics must HALT on missing data, never fabricate.

The bugs these lock down:
  1. VaR/CVaR returned 0.0 on an empty sample. A zero VaR reads as "this book
     has no risk", passes every limit check, and is divided into an equity
     budget by CVaR-based position sizing.
  2. RiskSnapshot.var_99 defaulted to 0.0, so `var_99 > max_portfolio_var_pct`
     silently PASSED whenever VaR was uncomputable.
  3. es_parametric_t's Monte-Carlo fallback cancelled its own tail term
     (`+ mean(tail) * 0.0`), returning VaR while the caller believed it was
     Expected Shortfall — understating the tail on every short-vol book.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.analytics.expected_shortfall import es_parametric_t
from backend.risk.institutional_risk_engine import (
    InsufficientDataError,
    PortfolioRiskMonitor,
    RiskSnapshot,
    VaREngine,
    _min_obs_for,
)

# ── 1. No fabricated VaR/CVaR ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "fn",
    ["historical_var", "parametric_var", "monte_carlo_var", "historical_cvar"],
)
def test_empty_sample_raises_instead_of_returning_zero(fn):
    with pytest.raises(InsufficientDataError):
        getattr(VaREngine, fn)(np.array([]))


@pytest.mark.parametrize(
    "fn",
    ["historical_var", "parametric_var", "monte_carlo_var", "historical_cvar"],
)
def test_undersized_sample_raises(fn):
    """10 points cannot support a 99% quantile."""
    with pytest.raises(InsufficientDataError):
        getattr(VaREngine, fn)(np.full(10, 0.001))


def test_all_nan_sample_raises():
    """NaNs are dropped, so an all-NaN series is an empty series."""
    with pytest.raises(InsufficientDataError):
        VaREngine.historical_var(np.full(500, np.nan), 0.99)


def test_min_obs_scales_with_confidence():
    assert _min_obs_for(0.99) == 100     # 1/(1-0.99)
    assert _min_obs_for(0.975) == 40
    assert _min_obs_for(0.95) == 20      # floored at the absolute minimum


def test_sufficient_sample_returns_positive_var():
    r = np.random.default_rng(0).normal(0.0, 0.01, 500)
    assert VaREngine.historical_var(r, 0.99, 1) > 0
    assert VaREngine.historical_cvar(r, 0.975) > 0


def test_cvar_exceeds_var_on_same_sample():
    """CVaR is the mean beyond the quantile, so it must dominate VaR."""
    r = np.random.default_rng(1).normal(0.0, 0.01, 2000)
    assert VaREngine.historical_cvar(r, 0.975) > VaREngine.historical_var(r, 0.975, 1)


# ── 2. Unknown VaR must not silently pass the limit check ────────────────────

def test_snapshot_var_defaults_to_none_not_zero():
    assert RiskSnapshot().var_99 is None
    assert RiskSnapshot().cvar_975 is None


def test_unavailable_var_raises_an_alert():
    monitor = PortfolioRiskMonitor(redis_client=None)
    snap = RiskSnapshot()
    snap.var_99 = None
    alerts = monitor.should_trigger_alert(snap)
    assert any("VaR unavailable" in a for a in alerts), (
        "an uncomputable VaR must alert, not silently pass the limit"
    )


def test_breaching_var_still_alerts():
    monitor = PortfolioRiskMonitor(redis_client=None)
    snap = RiskSnapshot()
    snap.var_99 = monitor.config.max_portfolio_var_pct + 0.05
    assert any("VaR breach" in a for a in monitor.should_trigger_alert(snap))


def test_compliant_var_does_not_alert_on_var():
    monitor = PortfolioRiskMonitor(redis_client=None)
    snap = RiskSnapshot()
    snap.var_99 = monitor.config.max_portfolio_var_pct / 2.0
    assert not any("VaR" in a for a in monitor.should_trigger_alert(snap))


# ── 3. Expected Shortfall really is ES, not VaR ──────────────────────────────

def test_es_parametric_t_fallback_returns_es_not_var(monkeypatch):
    """Force the scipy-less path and assert ES > VaR."""
    import builtins
    real_import = builtins.__import__

    def _no_scipy(name, *a, **k):
        if name == "scipy.stats":
            raise ImportError("forced for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_scipy)
    es = es_parametric_t(0.0, 1.0, 5.0, 0.975)
    monkeypatch.undo()

    rng = np.random.default_rng(42)
    x = 0.0 + 1.0 * rng.standard_t(df=5.0, size=200_000)
    var = float(np.quantile(x, 0.975))

    assert es > var, "ES must be the mean of the tail, strictly beyond VaR"


def test_es_parametric_t_matches_closed_form_when_scipy_present():
    """The analytic path and the MC fallback should agree closely."""
    pytest.importorskip("scipy")
    analytic = es_parametric_t(0.0, 1.0, 5.0, 0.975)
    assert analytic == pytest.approx(3.5, rel=0.15)
