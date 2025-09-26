# tests/test_var_risk_premium.py
import numpy as np
import pandas as pd

from engines.options.signals.var_risk_premium import ( # type: ignore
    VRPConfig,
    BacktestConfig,
    realized_variance,
    implied_variance_from_vix,
    build_vrp_series,
    build_vrp_signal,
    backtest_vrp,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_index(n=520, start="2022-01-03"):
    return pd.date_range(start, periods=n, freq="B")

def make_equity_path(idx, seed=0):
    rng = np.random.default_rng(seed)
    vol_state = 0.012 + 0.008*(rng.standard_normal(len(idx)) > 1.2)
    r = 0.0002 + vol_state * rng.standard_normal(len(idx))
    px = pd.Series(4000*np.exp(np.cumsum(r)), index=idx, name="SPX")
    return px

def make_vix_from_rv(rv: pd.Series, seed=1):
    rng = np.random.default_rng(seed)
    vix = np.sqrt(rv.clip(1e-6)) * 100.0 + 3.0 + 2.0*np.sin(np.linspace(0, 6, len(rv)))
    vix = pd.Series(vix, index=rv.index, name="VIX").clip(8, 80)
    return vix

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_realized_variance_and_implied_variance_shapes():
    idx = make_index(260)
    px = make_equity_path(idx)
    rv = realized_variance(px, window=21)
    assert not rv.empty
    assert rv.index.equals(idx)
    # synthetic VIX should give IV variance
    vix = pd.Series(20.0, index=idx)
    iv_var = implied_variance_from_vix(vix)
    assert (iv_var > 0).all()

def test_build_vrp_series_and_zscores():
    idx = make_index(300)
    px = make_equity_path(idx)
    rv = realized_variance(px, window=21)
    vix = make_vix_from_rv(rv)
    cfg = VRPConfig(mode="vix_only", z_lookback=126, clip_z=3.0)
    parts = build_vrp_series(underlying=px, vix_30d=vix, cfg=cfg)
    assert {"iv_var", "rv", "vrp", "vrp_z"} <= set(parts.keys())
    # VRP = iv - rv
    diff = (parts["iv_var"] - parts["rv"]).dropna()
    assert np.isclose(diff.iloc[-1], parts["vrp"].iloc[-1])
    # z-scores bounded by clip
    assert abs(parts["vrp_z"]).max() <= cfg.clip_z + 1e-8

def test_build_vrp_signal_positions_are_bounded():
    idx = make_index(400)
    px = make_equity_path(idx)
    rv = realized_variance(px, 21)
    vix = make_vix_from_rv(rv)
    cfg = VRPConfig(mode="vix_only", side="carry_short", cap=1.0)
    sig = build_vrp_signal(underlying=px, vix_30d=vix, cfg=cfg)
    pos = sig["position"]
    assert (pos.abs() <= cfg.cap + 1e-9).all()
    # diag has the latest snapshot
    assert "position" in sig["diag"].columns

def test_backtest_vrp_produces_nav_and_pnl():
    idx = make_index(500)
    px = make_equity_path(idx, seed=42)
    rv = realized_variance(px, 21)
    vix = make_vix_from_rv(rv)
    cfg = VRPConfig(mode="vix_only", side="carry_short", rebal_freq="W-FRI",
                    unit_gross=1.0, cap=1.0, tc_bps=5.0, var_notional_per_unit=1_000_000)
    bt = BacktestConfig(nav0=1_000_000, hold_days=21, slippage_bps=1.0, use_log_returns=True)

    out = backtest_vrp(underlying=px, vix_30d=vix, cfg=cfg, bt=bt)
    summ = out["summary"]
    assert not summ.empty
    assert "nav$" in summ.columns
    assert "pnl$" in summ.columns
    # NAV path should change
    assert summ["nav$"].var() > 0
    # Costs nonnegative
    assert (summ["costs$"] >= 0).all()