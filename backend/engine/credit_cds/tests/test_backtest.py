# tests/test_hy_vs_ig_backtest.py
import numpy as np
import pandas as pd

import engines.credit.hy_vs_ig as hyig # type: ignore


def make_data(n=300, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    ig = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, n)), index=idx, name="IG")
    hy = pd.Series(400 + np.cumsum(rng.normal(0, 0.8, n)), index=idx, name="HY")
    return hy, ig


def test_signal_shapes_and_features():
    hy, ig = make_data()
    sig = hyig.build_signal(hy, ig, hyig.StrategyConfig())
    feats = sig["features"]

    # Ensure features present
    assert {"hy_bps", "ig_bps", "ratio", "zscore", "score"} <= set(feats.columns)
    # ratio = hy / ig
    ratio_calc = (feats["hy_bps"] / feats["ig_bps"]).iloc[-1]
    assert np.isclose(ratio_calc, feats["ratio"].iloc[-1])


def test_backtest_runs_and_nav_changes():
    hy, ig = make_data()
    out = hyig.backtest(hy, ig, hyig.StrategyConfig(), hyig.BacktestConfig())
    summ = out["summary"]

    # NAV path should not be constant
    assert summ["nav$"].var() > 0
    # Costs nonnegative
    assert (summ["costs$"] >= 0).all()
    # Score bounded by clip
    cfg = hyig.StrategyConfig()
    assert abs(summ["score"]).max() <= cfg.clip_z + 1e-8


def test_backtest_position_effect():
    # If ratio is constant, pnl should be ~0
    idx = pd.date_range("2023-01-02", periods=50, freq="B")
    ig = pd.Series(100.0, index=idx)
    hy = pd.Series(400.0, index=idx)
    out = hyig.backtest(hy, ig)
    summ = out["summary"]

    assert abs(summ["pnl$"].sum()) < 1e-6
    assert np.isclose(summ["nav$"].iloc[-1], hyig.BacktestConfig().nav0)