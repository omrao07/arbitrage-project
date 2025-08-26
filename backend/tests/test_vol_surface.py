# tests/test_vol_surface.py
# Comprehensive test suite for vol surface utilities.
# Adjust the import path to your module if needed.
import importlib
import math
import os
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import pytest # type: ignore

vs = importlib.import_module("vol_surface")  # rename if needed: backend.analytics.vol_surface


# ---------------------------- Helpers ----------------------------

def mk_options(
    n_strikes=9,
    maturities=(0.0833, 0.25, 0.5, 1.0, 2.0),
    fwd=100.0,
    skew=-0.25,
    smile=0.15,
    base_vol=0.22,
    seed=1234,
    noise=0.004,
):
    """
    Build a toy option quote surface with skew + smile + small noise.
    Columns: maturity [y], strike, market_vol
    """
    rng = np.random.default_rng(seed)
    ks = np.linspace(0.7 * fwd, 1.3 * fwd, n_strikes)  # deep OTM put -> deep OTM call
    rows = []
    for T in maturities:
        for K in ks:
            moneyness = (K - fwd) / fwd
            vol = base_vol + skew * moneyness + smile * (moneyness**2)
            vol = max(0.05, vol) + rng.normal(0.0, noise)
            rows.append({"maturity": float(T), "strike": float(K), "market_vol": float(vol)})
    return pd.DataFrame(rows)


def approx_convex(y):
    """
    Return True if y is "roughly" convex: second finite differences >= -tol on average.
    We allow small negative due to noise/interp.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return True
    d2 = y[:-2] - 2 * y[1:-1] + y[2:]
    return (d2.mean() >= -0.01) and (np.quantile(d2, 0.1) >= -0.05)


def has(func_name):
    return hasattr(vs, func_name)


# ---------------------------- Fixtures ----------------------------

@pytest.fixture(scope="module")
def quotes():
    return mk_options()

@pytest.fixture(scope="module")
def quotes_noisy():
    return mk_options(noise=0.02)

@pytest.fixture(scope="module")
def quotes_sparse():
    # Sparse grid: single maturity with many strikes + another with few
    df = mk_options(n_strikes=7, maturities=(0.25, 1.0, 3.0))
    # Drop some rows to create holes
    return df.sample(frac=0.7, random_state=7).reset_index(drop=True)

@pytest.fixture(scope="module")
def surface(quotes):
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    return vs.fit_vol_surface(quotes)


# ---------------------------- Core Invariants (1–8) ----------------------------

def test_fit_returns_dataframe(quotes):
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    surf = vs.fit_vol_surface(quotes)
    assert isinstance(surf, pd.DataFrame)
    assert set(["maturity", "strike"]).issubset(surf.columns)

def test_interpolate_scalar_type(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    v = vs.interpolate_vol(0.5, 100)
    assert isinstance(v, (float, np.floating))

def test_interpolate_non_negative(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    v = vs.interpolate_vol(1.0, 80.0)
    assert v >= 0.0

def test_interpolate_reasonable_bounds(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    # Should be between 0% and, say, 300% (very generous bound)
    v = vs.interpolate_vol(2.0, 120.0)
    assert 0.0 <= v <= 3.0

def test_smile_slice_shape(surface):
    if not has("smile_slice"):
        pytest.skip("smile_slice not implemented")
    s = vs.smile_slice(0.5)
    assert isinstance(s, (pd.Series, dict))
    assert len(s) >= 5

def test_smile_slice_keys_sorted(surface):
    if not has("smile_slice"):
        pytest.skip("smile_slice not implemented")
    s = vs.smile_slice(1.0)
    xs = list(s.keys()) if isinstance(s, dict) else list(s.index)
    assert xs == sorted(xs)

def test_surface_handles_empty_input():
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    empty = pd.DataFrame(columns=["maturity", "strike", "market_vol"])
    surf = vs.fit_vol_surface(empty)
    assert isinstance(surf, pd.DataFrame)

def test_surface_ignores_nans(quotes):
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    df = quotes.copy()
    df.loc[df.sample(frac=0.05, random_state=1).index, "market_vol"] = np.nan
    surf = vs.fit_vol_surface(df)
    assert surf["market_vol"].notna().all() if "market_vol" in surf.columns else True


# ---------------------------- Smile / Term Structure (9–16) ----------------------------

@pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 2.0])
def test_smile_convex_heuristic(surface, T):
    if not has("smile_slice"):
        pytest.skip("smile_slice not implemented")
    s = vs.smile_slice(T)
    y = list(s.values()) if isinstance(s, dict) else list(s.to_numpy())
    assert approx_convex(y)

def test_term_structure_not_pathological(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    # Total variance should not decline too fast with maturity.
    K = 100.0
    v_short = vs.interpolate_vol(0.25, K)
    v_long = vs.interpolate_vol(2.0, K)
    w_short = (v_short ** 2) * 0.25
    w_long = (v_long ** 2) * 2.0
    assert w_long >= 0.5 * w_short  # allow modest anomalies but avoid inverted term variance

@pytest.mark.parametrize("K", [80.0, 90.0, 100.0, 110.0, 120.0])
def test_calendar_arbitrage_guard(surface, K):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    vols = [(T, vs.interpolate_vol(T, K)) for T in [0.25, 0.5, 1.0, 2.0]]
    # total variance non-decreasing in T (allow tiny numeric slack)
    prev = 0.0
    for T, sig in vols:
        tv = max(0.0, (sig ** 2) * T)
        assert tv + 1e-6 >= prev - 1e-6
        prev = tv

def test_deep_otm_vol_not_negative(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    assert vs.interpolate_vol(1.0, 40.0) >= 0.0
    assert vs.interpolate_vol(1.0, 200.0) >= 0.0

def test_atm_vs_otm_reasonable(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    atm = vs.interpolate_vol(1.0, 100.0)
    otm = vs.interpolate_vol(1.0, 70.0)
    assert 0.5 * atm <= otm <= 2.5 * atm  # broad sanity envelope

@pytest.mark.parametrize("T", [0.25, 1.0, 2.0])
def test_smile_sorted_strikes_increasing(surface, T):
    if not has("smile_slice"):
        pytest.skip("smile_slice not implemented")
    s = vs.smile_slice(T)
    xs = np.array(list(s.keys()) if isinstance(s, dict) else list(s.index))
    assert np.all(np.diff(xs) > 0)


# ---------------------------- Robustness (17–22) ----------------------------

def test_noisy_quotes_do_not_blow_up(quotes_noisy):
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    surf = vs.fit_vol_surface(quotes_noisy)
    # interpolate at a few points
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    val = vs.interpolate_vol(0.5, 100.0)
    assert np.isfinite(val)

@pytest.mark.parametrize("T,K", [(0.25, 95.0), (0.5, 105.0), (1.0, 120.0)])
def test_repeatability_same_inputs_same_output(surface, T, K):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    v1 = vs.interpolate_vol(T, K)
    v2 = vs.interpolate_vol(T, K)
    assert v1 == pytest.approx(v2)

def test_sparse_grid_interpolates(quotes_sparse):
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    vs.fit_vol_surface(quotes_sparse)
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    v = vs.interpolate_vol(0.75, 103.0)
    assert np.isfinite(v) and v >= 0.0

def test_extrapolation_is_clamped(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    # Far outside grid should not produce insane or negative values
    v_far = vs.interpolate_vol(5.0, 1000.0)
    assert 0.0 <= v_far <= 5.0

def test_handles_all_nan_column():
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    df = mk_options()
    df["market_vol"] = np.nan
    surf = vs.fit_vol_surface(df)
    # If you propagate NaNs, that's fine; just don't crash
    assert isinstance(surf, pd.DataFrame)

def test_handles_single_maturity_many_strikes():
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    single = mk_options(maturities=(0.5,), n_strikes=11)
    vs.fit_vol_surface(single)
    if not has("smile_slice"):
        pytest.skip("smile_slice not implemented")
    s = vs.smile_slice(0.5)
    assert len(s) >= 5


# ---------------------------- Performance (23–25) ----------------------------

def test_fit_performance_large_grid():
    if not has("fit_vol_surface"):
        pytest.skip("fit_vol_surface not implemented")
    df = mk_options(n_strikes=25, maturities=np.linspace(0.0833, 2.0, 16), noise=0.0)
    t0 = perf_counter()
    _ = vs.fit_vol_surface(df)
    dt = perf_counter() - t0
    assert dt < 2.0  # 2 seconds budget for a fairly big toy grid

def test_interpolate_bulk_speed(surface):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    pts = [(float(T), float(K)) for T in np.linspace(0.1, 2.0, 40) for K in np.linspace(70, 130, 60)]
    t0 = perf_counter()
    arr = [vs.interpolate_vol(T, K) for (T, K) in pts]
    dt = perf_counter() - t0
    assert np.isfinite(arr).all()
    assert dt < 1.5  # 2400 points under 1.5s

def test_surface_grid_if_available(surface):
    if not has("surface_grid"):
        pytest.skip("surface_grid not implemented")
    mats = [0.25, 0.5, 1.0]
    ks = [80, 90, 100, 110, 120]
    grid = vs.surface_grid(mats, ks)
    assert isinstance(grid, pd.DataFrame)
    assert set(["maturity", "strike", "vol"]).issubset(grid.columns)
    assert len(grid) == len(mats) * len(ks)


# ---------------------------- Integration (26–30) ----------------------------

def test_interpolate_matches_quotes_at_nodes(quotes):
    if not (has("fit_vol_surface") and has("interpolate_vol")):
        pytest.skip("fit/interpolate not implemented")
    vs.fit_vol_surface(quotes)
    # Pick 10 random nodes and ensure interpolator reproduces close to market vol
    sub = quotes.sample(n=min(10, len(quotes)), random_state=99)
    diffs = []
    for _, row in sub.iterrows():
        v = vs.interpolate_vol(float(row["maturity"]), float(row["strike"]))
        diffs.append(abs(v - float(row["market_vol"])))
    assert np.median(diffs) < 0.03  # interpolator close to quotes at nodes

def test_smile_slice_consistent_with_interpolate(surface):
    if not (has("smile_slice") and has("interpolate_vol")):
        pytest.skip("smile/interpolate not implemented")
    T = 1.0
    s = vs.smile_slice(T)
    xs = (list(s.keys()) if isinstance(s, dict) else list(s.index))[:5]
    # values from the slice should be close to direct interpolation
    for K in xs:
        v_dir = vs.interpolate_vol(T, float(K))
        v_sli = s[K] if isinstance(s, dict) else float(s.loc[K])
        assert v_sli == pytest.approx(v_dir, abs=0.05)

def test_serialize_roundtrip_if_available(surface):
    # Support either save/load on module, or serialize()/deserialize() on returned object
    # We try several conventions and skip if none exist.
    if has("save") and has("load"):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "surf.bin")
            vs.save(path)
            # mutate: (not strictly needed, just ensure call works)
            vs.load(path)
    elif has("serialize") and has("deserialize"):
        blob = vs.serialize()
        vs.deserialize(blob)
    else:
        pytest.skip("No save/load or serialize/deserialize API; skipping.")

def test_label_integrity(surface, quotes):
    # If your surface stores labels/axes, ensure they include the original bounds
    # Skip if you don't expose this metadata.
    if not hasattr(surface, "columns"):
        pytest.skip("Surface not a DataFrame-like with columns.")
    has_m = "maturity" in surface.columns
    has_k = "strike" in surface.columns
    assert has_m or has_k

@pytest.mark.parametrize("T,K", [(0.25, 100.0), (1.0, 100.0), (2.0, 100.0)])
def test_small_lipschitz_nearby_points(surface, T, K):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    v0 = vs.interpolate_vol(T, K)
    v1 = vs.interpolate_vol(T + 0.01, K + 0.5)
    assert abs(v1 - v0) < 0.15  # prevent wild oscillations


# ---------------------------- Extra Arbitrage Heuristics (31–32) ----------------------------

@pytest.mark.parametrize("Kpair", [(90.0, 100.0), (100.0, 110.0)])
def test_vertical_spread_monotone_variance(surface, Kpair):
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    K1, K2 = Kpair
    T = 1.0
    # As strike increases on calls, implied vol shouldn't spike erratically between adjacent strikes
    v1 = vs.interpolate_vol(T, K1)
    v2 = vs.interpolate_vol(T, K2)
    assert abs(v2 - v1) < 0.25

def test_calendar_spread_price_consistency(surface):
    # Heuristic: ATM total variance increases with T -> ATM price proxy also increases.
    if not has("interpolate_vol"):
        pytest.skip("interpolate_vol not implemented")
    atm_short = vs.interpolate_vol(0.25, 100.0)
    atm_long  = vs.interpolate_vol(2.0, 100.0)
    tv_short = atm_short**2 * 0.25
    tv_long = atm_long**2 * 2.0
    assert tv_long >= tv_short * 0.8  # generous, but catches inverted/buggy fits