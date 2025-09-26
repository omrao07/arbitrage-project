# engines/stat_arb/tests/test_pairs.py
import numpy as np
import pandas as pd

from engines.stat_arb.signals.pairs import ( # type: ignore
    rolling_corr,
    engle_granger,
    select_pairs,
    generate_pair_signal,
    compute_pair_diagnostics,
)

# ----------------------- Helpers -----------------------

def _mk_series(seed=0, n=400, mu=0.0002, sigma=0.015, start=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    x = np.zeros(n)
    x[0] = start
    for t in range(1, n):
        x[t] = x[t-1] * (1.0 + mu + sigma * rng.standard_normal())
    return pd.Series(x, index=idx)

def _cointegrated_pair(seed=1, n=400):
    """
    Build a cointegrated pair:
      y_t = a + b * x_t + eps_t  with small stationary noise.
    """
    x = _mk_series(seed=seed, n=n)
    rng = np.random.default_rng(seed + 123)
    a, b = 5.0, 1.25
    eps = rng.normal(0, 0.5, size=n)  # small noise
    y = a + b * x.values + eps # type: ignore
    y = pd.Series(y, index=x.index)
    return y, x

def _independent_walks(seed=2, n=400):
    """Two independent random walks (typically NOT cointegrated)."""
    x = _mk_series(seed=seed, n=n)
    y = _mk_series(seed=seed + 7, n=n)
    return y, x

# ----------------------- Tests -----------------------

def test_rolling_corr_and_engle_granger_on_coint_pair():
    y, x = _cointegrated_pair()
    corr = rolling_corr(y, x, lookback=60).iloc[-1]
    pval, is_coint = engle_granger(y, x, lookback=252, signif=0.05)

    assert corr > 0.8  # highly correlated
    assert is_coint
    assert pval < 0.05

def test_rolling_corr_and_engle_granger_on_non_coint_pair():
    y, x = _independent_walks()
    corr = rolling_corr(y, x, lookback=60).iloc[-1]
    pval, is_coint = engle_granger(y, x, lookback=252, signif=0.05)

    # Corr could be anything; we only assert lack of cointegration
    assert not is_coint or pval >= 0.01  # usually not cointegrated

def test_select_pairs_filters_correctly():
    # Build a small price panel with one cointegrated pair and one not
    y1, x1 = _cointegrated_pair(seed=3)
    y2, x2 = _independent_walks(seed=4)
    prices = pd.DataFrame({
        "Y1": y1, "X1": x1,  # cointegrated
        "Y2": y2, "X2": x2,  # not cointegrated
    })
    candidates = [("Y1", "X1"), ("Y2", "X2")]
    selected = select_pairs(prices, candidates, corr_lb=60, coint_lb=252, min_corr=0.5, signif=0.05)

    assert ("Y1", "X1") in selected
    # very likely excluded:
    assert ("Y2", "X2") not in selected

def test_generate_pair_signal_threshold_logic_and_caps():
    y, x = _cointegrated_pair()
    # Force a large positive spread z at the last bar to trigger -1 units (short Y / long X)
    # Do this by bumping y up at the last timestamp.
    bump = y.iloc[-1] * 0.05
    y.iloc[-1] = y.iloc[-1] + bump

    units, beta, z = generate_pair_signal(
        y, x,
        entry_z=1.0, exit_z=0.25,
        beta_method="rolling_ols",
        beta_lookback=120, z_lookback=60, max_units=0.7
    )
    # Large positive z => -1 units (capped at max_units)
    assert units <= 0.0
    assert abs(units) <= 0.7 + 1e-9
    assert np.isfinite(beta)
    assert np.isfinite(z)

    # Flip to negative z: drop y to create long +1 units
    y.iloc[-1] = y.iloc[-1] - 2.0 * bump
    units2, beta2, z2 = generate_pair_signal(
        y, x, entry_z=1.0, exit_z=0.25, beta_method="rolling_ols",
        beta_lookback=120, z_lookback=60, max_units=1.0
    )
    assert units2 >= 0.0
    assert abs(units2) <= 1.0 + 1e-9
    assert np.isfinite(beta2)
    assert np.isfinite(z2)

def test_compute_pair_diagnostics_outputs_are_finite():
    y, x = _cointegrated_pair()
    diag = compute_pair_diagnostics(
        y, x,
        beta_method="rolling_ols",
        beta_lookback=120,
        z_lookback=60,
    )
    assert np.isfinite(diag.beta)
    assert np.isfinite(diag.spread)
    assert np.isfinite(diag.z) or diag.z == 0.0
    assert np.isfinite(diag.spread_vol)
    assert np.isfinite(diag.half_life) or diag.half_life == np.inf