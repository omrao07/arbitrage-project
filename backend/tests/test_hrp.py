# tests/test_hrp.py
"""
HRP tests (duck-typed)
- Verifies weights sum to 1 and are non-negative
- Penalizes highly correlated / high-vol assets
- Invariance to rescaling inputs
- Handles near-collinear & zero-variance assets
- Deterministic under fixed seed (if implementation is deterministic)
"""

import numpy as np
import pandas as pd
import pytest # type: ignore

hrp_mod = pytest.importorskip("backend.analytics.hrp", reason="backend.analytics.hrp not found")

# ---------- Helpers -----------------------------------------------------------

def _gen_returns(n_assets=6, n_obs=1000, seed=42, vol_skew=True, block_corr=True):
    rng = np.random.default_rng(seed)
    # Base factors to induce correlation blocks
    f1 = rng.normal(0, 0.01, n_obs)
    f2 = rng.normal(0, 0.01, n_obs)
    e = rng.normal(0, 0.005, (n_obs, n_assets))
    R = np.zeros((n_obs, n_assets))
    # two clusters: {0,1,2} load on f1; {3,4,5} load on f2
    for j in range(n_assets):
        base = (f1 if j < n_assets // 2 else f2) if block_corr else 0.0
        R[:, j] = base + e[:, j]
        if vol_skew:
            R[:, j] *= (1.0 + 0.5 * (j / max(1, n_assets - 1)))  # higher vol for later assets
    cols = [f"A{j}" for j in range(n_assets)]
    return pd.DataFrame(R, columns=cols)

def _get_weights(ret_df: pd.DataFrame, **kwargs) -> pd.Series: # type: ignore
    # Try function api
    if hasattr(hrp_mod, "hrp_weights"):
        w = hrp_mod.hrp_weights(ret_df, **kwargs)
        assert isinstance(w, (pd.Series, dict))
        return pd.Series(w).sort_index()
    # Try class api
    if hasattr(hrp_mod, "HRP"):
        model = hrp_mod.HRP(**kwargs)
        model.fit(ret_df)
        w = getattr(model, "weights_", None)
        if isinstance(w, dict):
            w = pd.Series(w)
        assert isinstance(w, pd.Series)
        return w.sort_index()
    pytest.skip("No compatible HRP API found (hrp_weights() or HRP class).")

# ---------- Tests -------------------------------------------------------------

def test_weights_sum_to_one_and_non_negative():
    df = _gen_returns()
    w = _get_weights(df)
    assert pytest.approx(w.sum(), rel=1e-8, abs=1e-10) == 1.0
    assert (w >= -1e-12).all(), "HRP should not produce negative weights"

def test_more_weight_to_lower_vol_assets():
    df = _gen_returns(vol_skew=True)
    w = _get_weights(df)
    # Lower-index assets have lower vol; expect higher weights on A0/A1 than A4/A5
    assert w["A0"] > w["A4"]
    assert w["A1"] > w["A5"]

def test_block_correlation_effect():
    df = _gen_returns(block_corr=True)
    w = _get_weights(df)
    # Sibling assets in the same cluster should not both get dominant weights.
    # Check concentration by cluster roughly balanced.
    left = w[[c for c in w.index if c in ["A0","A1","A2"]]].sum()
    right = w[[c for c in w.index if c in ["A3","A4","A5"]]].sum()
    assert 0.3 <= left <= 0.7 and 0.3 <= right <= 0.7

def test_scale_invariance():
    df = _gen_returns()
    w1 = _get_weights(df)
    # Rescale asset A3 by a constant factor; HRP should be invariant to return units scaling
    df2 = df.copy()
    df2["A3"] = df2["A3"] * 100.0
    w2 = _get_weights(df2)
    # Allow tiny numerical differences
    diff = (w1.sort_index() - w2.sort_index()).abs().max()
    assert diff < 1e-6

def test_handles_zero_variance_asset():
    df = _gen_returns()
    df["ZERO"] = 0.0
    w = _get_weights(df)
    # ZERO-variance asset should get (almost) zero weight
    assert w.get("ZERO", 0.0) <= 1e-6
    # Weights should still sum to ~1
    assert pytest.approx(w.sum(), rel=1e-8, abs=1e-10) == 1.0

def test_handles_near_duplicate_assets():
    df = _gen_returns()
    df["CLONE"] = df["A0"] * 1.00001  # nearly identical
    w = _get_weights(df)
    # Combined weight on the pair should not explode (should be comparable to A0 original share)
    pair = w["A0"] + w["CLONE"]
    assert pair <= 2 * max(w.drop(["A0","CLONE"]).max(), 1e-9)

def test_deterministic_with_fixed_seed_if_supported():
    # If implementation takes a random_state/seed kwarg, we test determinism.
    df = _gen_returns(seed=123)
    kwargs = {}
    for key in ("random_state", "seed", "rng", "np_random_state"):
        try:
            _ = _get_weights(df, **{key: 7})
            kwargs[key] = 7
            break
        except TypeError:
            continue
    if not kwargs:
        pytest.skip("HRP impl does not expose a seed parameter; skipping determinism test.")
    w1 = _get_weights(df, **kwargs)
    w2 = _get_weights(df, **kwargs)
    assert w1.equals(w2)