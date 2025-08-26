# test_corr_matrix.py
import numpy as np
import pandas as pd
import pytest # type: ignore

from backend.analytics.corr_matrix import compute_corr_matrix  # type: ignore # adjust path

def make_data(seed=42, n_assets=5, n_days=200):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.01, size=(n_days, n_assets))
    cols = [f"Asset{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)

# --- Core invariants ---

def test_shape_and_symmetry():
    df = make_data()
    corr = compute_corr_matrix(df)
    assert corr.shape[0] == corr.shape[1]
    assert np.allclose(corr, corr.T)

def test_diag_is_one():
    df = make_data()
    corr = compute_corr_matrix(df)
    assert np.allclose(np.diag(corr), 1.0)

def test_values_within_bounds():
    df = make_data()
    corr = compute_corr_matrix(df)
    assert (corr.values <= 1).all() and (corr.values >= -1).all()

# --- Edge cases ---

def test_perfect_positive_corr():
    x = np.arange(100)
    df = pd.DataFrame({"A": x, "B": x})
    corr = compute_corr_matrix(df)
    assert corr.loc["A","B"] == pytest.approx(1.0)

def test_perfect_negative_corr():
    x = np.arange(100)
    df = pd.DataFrame({"A": x, "B": -x})
    corr = compute_corr_matrix(df)
    assert corr.loc["A","B"] == pytest.approx(-1.0)

def test_constant_series_yields_nan_or_zero():
    df = pd.DataFrame({"A":[5]*100, "B":np.arange(100)})
    corr = compute_corr_matrix(df)
    val = corr.loc["A","B"]
    assert np.isnan(val) or abs(val) < 1e-12

def test_missing_values_are_handled():
    df = make_data()
    df.iloc[0:10,0] = np.nan
    corr = compute_corr_matrix(df)
    # Pandas should skip NaN and still compute
    assert np.isfinite(corr.values).all()

# --- Randomized consistency ---

@pytest.mark.parametrize("seed", [0,1,2,3,4])
def test_multiple_seeds_consistent(seed):
    df = make_data(seed=seed)
    corr = compute_corr_matrix(df)
    # Each matrix should still respect invariants
    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0)
    assert (corr.values <= 1).all() and (corr.values >= -1).all()