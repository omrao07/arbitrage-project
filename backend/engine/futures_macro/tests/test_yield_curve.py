# engines/rates/tests/test_yield_curve.py
import numpy as np
import pandas as pd

from engines.rates.signals.yield_curve import ( # type: ignore
    CurveConfig, MapConfig, build_signal, interpolate_panel,
    slope_series, curvature_series, pca_factors,
    dv01_neutral_steepener, dv01_neutral_butterfly,
    dv01_to_contracts, build_yield_curve_trades
)

# ----------------------------- helpers -----------------------------

def _synthetic_curve(idx, seed=0):
    """
    Build a realistic-looking UST curve panel with cols=[0.5,2,5,10,30] in DECIMAL.
    We embed a mild steepening trend to make slope z-score positive near the end.
    """
    rng = np.random.default_rng(seed)
    n = len(idx)
    base = {
        0.5: 0.035 + 0.0003*np.linspace(0, 1, n) + 0.004*rng.standard_normal(n),
        2.0: 0.037 + 0.0004*np.linspace(0, 1, n) + 0.004*rng.standard_normal(n),
        5.0: 0.038 + 0.0006*np.linspace(0, 1, n) + 0.004*rng.standard_normal(n),
        10.0:0.040 + 0.0008*np.linspace(0, 1, n) + 0.004*rng.standard_normal(n),
        30.0:0.041 + 0.0010*np.linspace(0, 1, n) + 0.004*rng.standard_normal(n),
    }
    df = pd.DataFrame(base, index=idx).clip(lower=0.0005)  # keep positive
    return df

# ----------------------------- tests -----------------------------

def test_interpolation_linear_and_feature_shapes():
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    yc_full = _synthetic_curve(idx, seed=1)
    # Remove the 5y column to force interpolation
    yc_sparse = yc_full.drop(columns=[5.0])

    cfg = CurveConfig(tenors=[0.5, 2.0, 5.0, 10.0, 30.0], z_lookback=120)
    yc_interp = interpolate_panel(yc_sparse, cfg.tenors)

    # Interpolated panel should have all requested columns and finite values
    assert set(yc_interp.columns) == set(cfg.tenors)
    assert yc_interp.notna().sum().sum() > 0

    # Basic features exist and align
    sl = slope_series(yc_interp, 2.0, 10.0)
    cu = curvature_series(yc_interp, 2.0, 5.0, 10.0)
    pca_scores, pca_load = pca_factors(yc_interp, n=3)
    assert len(sl) == len(idx)
    assert len(cu) == len(idx)
    assert {"PC1", "PC2", "PC3"}.issubset(pca_scores.columns)


def test_build_signal_combo_outputs_zscores_and_snapshot():
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    yc = _synthetic_curve(idx, seed=2)
    cfg = CurveConfig(z_lookback=120)

    out = build_signal(yields=yc, mode="combo", cfg=cfg)
    # Expect non-empty panels
    assert not out["yc"].empty
    assert not out["features"].empty
    assert not out["signal"].empty
    # Diagnostic contains the three z-scores
    for k in ["slope_z", "curv_z", "pc2_z"]:
        assert k in out["diag"].columns
        assert np.isfinite(out["diag"][k].iloc[-1])


def test_dv01_neutral_steepener_and_butterfly_contract_conversion():
    # DV01 targets → contracts with sane rounding/signs
    map_cfg = MapConfig()  # defaults: ZT/ZF/TY/US
    # Steepener with $10k gross DV01 (split across legs)
    st = dv01_neutral_steepener(
        yc_snap=pd.Series({2.0: 0.04, 10.0: 0.045}),
        target_gross_dv01_usd=10_000.0,
        map_cfg=map_cfg,
        cap_leg=0.6,
    )
    # Should produce two legs with opposite signs and equal magnitude
    assert set(st.index) == set([min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-2.0)),
                                 min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-10.0))])
    assert np.isclose(abs(st.iloc[0]), abs(st.iloc[1]), rtol=1e-6)
    assert np.sign(st.iloc[0]) != np.sign(st.iloc[1])

    # Convert to contracts (integers)
    c_st = dv01_to_contracts(dv01_targets=st, dv01_per_contract=map_cfg.dv01_usd)
    assert (c_st.round() == c_st).all()

    # Butterfly allocates 1:-2:1 style DV01 across wings/body
    bf = dv01_neutral_butterfly(target_gross_dv01_usd=12_000.0, map_cfg=map_cfg, body_tenor=5.0, wings=(2.0, 10.0))
    assert len(bf.index) == 3
    # Body magnitude roughly double a wing (per simple ratio)
    assert abs(bf.abs().max() / bf.abs().min()) >= 2.0 - 1e-6

    c_bf = dv01_to_contracts(dv01_targets=bf, dv01_per_contract=map_cfg.dv01_usd)
    assert (c_bf.round() == c_bf).all()


def test_end_to_end_build_yield_curve_trades_sign_consistency():
    """
    With a constructed steepening trend, the snapshot slope_z should be positive,
    so the steepener should be long the long-end contract (US/TY) and short the short-end (ZT).
    """
    idx = pd.date_range("2023-01-02", periods=320, freq="B")
    yc = _synthetic_curve(idx, seed=3)
    cfg = CurveConfig(z_lookback=120)
    map_cfg = MapConfig()  # ZT~2y, TY~10y, US~30y

    out = build_yield_curve_trades(
        yields=yc, mode="combo", cfg=cfg, map_cfg=map_cfg, target_gross_dv01_usd=10_000.0
    )
    diag = out["diag"].iloc[-1]
    st = out["steepener_dv01"]

    # Sanity: output shapes
    assert not out["yc"].empty
    assert "contracts_steepener" in out and "contracts_butterfly" in out

    if diag["slope_z"] > 0:
        # Find symbols mapped closest to 2y and 10y
        sym_short = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-2.0))
        sym_long  = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-10.0))
        # Expect long long-end DV01, short short-end DV01
        assert st[sym_long] > 0
        assert st[sym_short] < 0
    else:
        # If rare negative due to randomness, assert opposite
        sym_short = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-2.0))
        sym_long  = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-10.0))
        assert st[sym_long] < 0
        assert st[sym_short] > 0

    # Contracts are finite numbers (integers after rounding)
    cst = out["contracts_steepener"]
    cbf = out["contracts_butterfly"]
    assert np.isfinite(cst.fillna(0)).all()
    assert np.isfinite(cbf.fillna(0)).all()
    assert (cst.round() == cst).all()
    assert (cbf.round() == cbf).all()