# engines/fx/tests/test_fx_carry.py
import numpy as np
import pandas as pd

from engines.fx.signals.fx_carry import ( # type: ignore
    SignalConfig, BacktestConfig,
    build_fx_carry_weights, backtest_fx_carry,
    weights_to_notional_orders,
)

# ------------------------- helpers -------------------------

def _mk_spot(idx, seed=0):
    rng = np.random.default_rng(seed)
    spot = pd.DataFrame({
        "EUR": 1.10 + 0.03*rng.standard_normal(len(idx)),
        "JPY": 0.009 + 0.0004*rng.standard_normal(len(idx)),
        "GBP": 1.28 + 0.03*rng.standard_normal(len(idx)),
        "AUD": 0.70 + 0.03*rng.standard_normal(len(idx)),
        "CAD": 0.75 + 0.02*rng.standard_normal(len(idx)),
        "CHF": 1.12 + 0.02*rng.standard_normal(len(idx)),
    }, index=idx).abs()
    return spot

def _mk_rates(idx):
    # USD higher than EUR/JPY; NZD/AUD/GBP higher than USD, etc.
    r_usd = pd.Series(0.05 + 0.001*np.sin(np.linspace(0, 4, len(idx))), index=idx, name="USD")
    r_ccy = pd.DataFrame({
        "EUR": 0.03 + 0.001*np.cos(np.linspace(0, 5, len(idx))),
        "JPY": -0.001 + 0.001*np.sin(np.linspace(0, 3, len(idx))),
        "GBP": 0.045 + 0.001*np.cos(np.linspace(0, 6, len(idx))),
        "AUD": 0.042 + 0.001*np.sin(np.linspace(0, 2, len(idx))),
        "CAD": 0.041 + 0.001*np.sin(np.linspace(0, 3, len(idx))),
        "CHF": 0.02 + 0.001*np.cos(np.linspace(0, 4, len(idx))),
    }, index=idx)
    return r_usd, r_ccy

# ------------------------- tests -------------------------

def test_weights_unit_gross_and_caps_rates_source():
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    spot = _mk_spot(idx)
    r_usd, r_ccy = _mk_rates(idx)

    cfg = SignalConfig(carry_source="rates", use_momentum_filter=False, unit_gross=1.0, cap_per_ccy=0.25)
    w, diag = build_fx_carry_weights(
        spot_usd=spot,
        carry_source="rates",
        usd_short_rate=r_usd,
        ccy_short_rates=r_ccy,
        cfg=cfg,
    )
    assert not w.empty
    # Unit gross (after caps)
    assert abs(w.abs().sum() - cfg.unit_gross) < 1e-8
    # Cap respected
    assert (w.abs() <= cfg.cap_per_ccy + 1e-12).all()
    # Diagnostics has required columns
    for col in ["carry", "inv_vol", "mom_filter", "raw_score", "weight"]:
        assert col in diag.columns


def test_momentum_filter_sign_effect():
    idx = pd.date_range("2023-01-02", periods=380, freq="B")
    spot = _mk_spot(idx)
    r_usd, r_ccy = _mk_rates(idx)

    # With momentum filter ON
    cfg_on = SignalConfig(carry_source="rates", use_momentum_filter=True, unit_gross=1.0, cap_per_ccy=0.4)
    w_on, _ = build_fx_carry_weights(
        spot_usd=spot, carry_source="rates",
        usd_short_rate=r_usd, ccy_short_rates=r_ccy, cfg=cfg_on
    )
    # With momentum filter OFF
    cfg_off = SignalConfig(carry_source="rates", use_momentum_filter=False, unit_gross=1.0, cap_per_ccy=0.4)
    w_off, _ = build_fx_carry_weights(
        spot_usd=spot, carry_source="rates",
        usd_short_rate=r_usd, ccy_short_rates=r_ccy, cfg=cfg_off
    )
    # They can differ; ensure at least one currency sign flips in general case
    assert (np.sign(w_on).ne(np.sign(w_off))).any()


def test_backtest_shapes_and_cost_behavior():
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    spot = _mk_spot(idx)
    r_usd, r_ccy = _mk_rates(idx)

    cfg = SignalConfig(carry_source="rates", use_momentum_filter=True, unit_gross=1.0, cap_per_ccy=0.25)
    bt = BacktestConfig(rebalance="W-FRI", tc_bps=4.0, nav0=1_000_000.0)

    out = backtest_fx_carry(
        spot_usd=spot,
        cfg=cfg,
        bt=bt,
        usd_short_rate=r_usd,
        ccy_short_rates=r_ccy,
        carry_source="rates",
    )
    summary, weights = out["summary"], out["weights"]
    assert not summary.empty and not weights.empty
    for col in ["price_pnl$", "costs$", "pnl$", "nav", "ret_net"]:
        assert col in summary.columns
        assert summary[col].notna().all()
    # Costs only accrue on rebalance days (or immediately after) — sanity check
    turn = weights.diff().abs().sum(axis=1)
    cost_days = summary["costs$"] > 0
    assert cost_days.sum() <= (turn > 0).sum() + 3  # allow a couple of edges for start/end


def test_orders_conversion_min_notional_filter():
    idx = pd.date_range("2023-01-02", periods=260, freq="B")
    spot = _mk_spot(idx)
    # Small weights should be filtered out by min_notional
    w_small = pd.Series({"EUR": 0.001, "JPY": -0.001, "GBP": 0.0, "AUD": 0.0, "CAD": 0.0, "CHF": 0.0})
    last = spot.iloc[-1]
    orders = weights_to_notional_orders(weights=w_small, nav_usd=100_000, last_spot=last, min_notional_usd=10_000)
    assert orders.empty

    # Larger weights should pass
    w = pd.Series({"EUR": 0.20, "JPY": -0.10, "GBP": 0.0, "AUD": 0.0, "CAD": 0.0, "CHF": 0.0})
    orders2 = weights_to_notional_orders(weights=w, nav_usd=1_000_000, last_spot=last, min_notional_usd=25_000)
    assert not orders2.empty
    assert set(orders2.columns) >= {"side", "usd_notional", "units_ccy", "px"}


def test_forwards_vs_rates_sign_consistency():
    """
    If forward points imply positive carry for a ccy, its weight should be positive
    (assuming momentum filter permits). We construct a simple forward-points panel
    consistent with rates differentials and check signs.
    """
    idx = pd.date_range("2023-01-02", periods=280, freq="B")
    spot = _mk_spot(idx)
    r_usd, r_ccy = _mk_rates(idx)

    # Annualized "F/S - 1" proxy consistent with r_ccy - r_usd
    carry_rates = r_ccy.subtract(r_usd, axis=0)
    fwd_points = carry_rates.copy()  # treat as annualized carry for test

    cfg = SignalConfig(carry_source="forwards", forwards_are_annualized=True, use_momentum_filter=False, unit_gross=1.0, cap_per_ccy=0.3)
    w_fwd, _ = build_fx_carry_weights(
        spot_usd=spot, carry_source="forwards",
        forward_points=fwd_points, cfg=cfg
    )
    # With rates path
    cfg2 = SignalConfig(carry_source="rates", use_momentum_filter=False, unit_gross=1.0, cap_per_ccy=0.3)
    w_rates, _ = build_fx_carry_weights(
        spot_usd=spot, carry_source="rates",
        usd_short_rate=r_usd, ccy_short_rates=r_ccy, cfg=cfg2
    )

    # Signs should broadly agree on currencies present
    common = w_fwd.index.intersection(w_rates.index)
    assert (np.sign(w_fwd.reindex(common)) == np.sign(w_rates.reindex(common))).mean() > 0.8