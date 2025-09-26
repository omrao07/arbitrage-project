# tests/test_cds_basis.py
import numpy as np
import pandas as pd

import engines.cap_struct.cds_basis as basis # type: ignore


def make_idx(n=300, start="2023-01-02"):
    return pd.date_range(start, periods=n, freq="B")


def test_signal_shapes_and_consistency():
    idx = make_idx()
    rng = np.random.default_rng(0)

    # Synthetic series
    gov = pd.Series(0.02 + 0.00001*np.arange(len(idx)), index=idx, name="gov")           # 2% drifting up
    bond = gov + 0.015 + 0.001*rng.standard_normal(len(idx))                              # ~150 bps over gov
    cds = pd.Series(120 + np.cumsum(rng.normal(0, 0.5, len(idx))), index=idx, name="cds") # ~120 bps with noise

    out = basis.build_basis_signal(cds_spread_bps=cds, bond_yield=bond, gov_yield=gov)
    feats = out["features"]

    # Required columns exist
    assert {"bond_spread_bps", "cds_bps", "basis_bps", "z_basis"} <= set(feats.columns)

    # bond_spread_bps must equal (bond - gov) * 10000
    calc_spread = ((bond - gov) * 10000).reindex(feats.index)
    assert np.isclose(calc_spread.iloc[-1], feats["bond_spread_bps"].iloc[-1])

    # basis = bond_spread - cds
    exp_basis_last = feats["bond_spread_bps"].iloc[-1] - feats["cds_bps"].iloc[-1]
    assert np.isclose(exp_basis_last, feats["basis_bps"].iloc[-1])


def test_zscore_clip_respected():
    idx = make_idx(260)
    # Force wild swings to hit the clip
    gov = pd.Series(0.02, index=idx)
    bond = pd.Series(0.10, index=idx)   # huge spread → big basis
    cds = pd.Series(50.0, index=idx)    # 50 bps

    cfg = basis.BasisConfig(basis_z_lookback=63, clip_z=2.5)
    out = basis.build_basis_signal(cds_spread_bps=cds, bond_yield=bond, gov_yield=gov, cfg=cfg)
    z = out["features"]["z_basis"]
    assert abs(z).max() <= cfg.clip_z + 1e-8


def test_backtest_runs_costs_nonnegative_and_nav_moves():
    idx = make_idx()
    rng = np.random.default_rng(1)

    gy = pd.Series(0.025 + 0.00001*np.arange(len(idx)), index=idx)
    by = gy + 0.012 + 0.001*rng.standard_normal(len(idx))
    cds = pd.Series(100 + np.cumsum(rng.normal(0, 0.4, len(idx))), index=idx)

    out = basis.backtest(cds_spread_bps=cds, bond_yield=by, gov_yield=gy)
    summ = out["summary"]

    assert "nav$" in summ.columns and "pnl$" in summ.columns and "costs$" in summ.columns
    assert summ["nav$"].var() > 0
    assert (summ["costs$"] >= 0).all()


def test_backtest_zero_basis_yields_near_zero_pnl():
    # Construct bond spread exactly equal to CDS → basis = 0 → d_basis = 0 → pnl ≈ 0
    idx = make_idx(80)
    cds = pd.Series(120.0, index=idx)
    gy = pd.Series(0.02, index=idx)
    # Set bond yield so that (bond - gov)*10000 == cds (bps)
    by = gy + (cds / 10000.0)

    out = basis.backtest(cds_spread_bps=cds, bond_yield=by, gov_yield=gy)
    summ = out["summary"]

    # PnL should be (almost) zero; NAV ~ initial
    assert abs(float(summ["pnl$"].sum())) < 1e-6
    assert np.isclose(float(summ["nav$"].iloc[-1]), basis.BacktestConfig().nav0)


def test_sign_convention_when_basis_trends_up():
    # If basis steadily increases, strategy takes +notional (basis>0) and
    # pnl_t = -pos * d_basis/10000 → negative drift (ignoring costs).
    idx = make_idx(120)
    gy = pd.Series(0.02, index=idx)
    # Make bond spread grow from 100 → 200 bps, CDS flat 120 bps ⇒ basis from -20 → +80 bps
    bond_spread_bps = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    by = gy + (bond_spread_bps / 10000.0)
    cds = pd.Series(120.0, index=idx)

    cfg = basis.BasisConfig(notional=1_000_000.0, tc_bps=0.0)  # remove costs to test sign
    out = basis.backtest(cds_spread_bps=cds, bond_yield=by, gov_yield=gy, cfg=cfg)
    total_pnl = float(out["summary"]["pnl$"].sum())

    assert total_pnl < 0.0  # negative as basis grinds higher with long-basis stance