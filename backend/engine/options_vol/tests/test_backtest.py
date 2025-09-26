# tests/test_backtester.py
import numpy as np
import pandas as pd

# ---- Imports from your project ----
from engines.core.simulator import SimConfig, simulate_backtest # type: ignore
from engines.stat_arb.signals.dispersion import ( # type: ignore
    SignalConfig as DispCfg,
    BacktestConfig as DispBT,
    backtest_dispersion,
    build_weights as build_dispersion_weights,
)
from engines.options.signals.skew import ( # type: ignore
    SkewConfig,
    backtest_skew_rr,
    build_rr_bf_from_iv,
    build_realized_skew,
)
from engines.options.hedging.tail_hedges import ( # type: ignore
    SignalConfig as THCfg,
    BacktestConfig as THBT,
    backtest_tail_hedge,
)
from engines.options.signals.var_risk_premium import ( # type: ignore
    VRPConfig,
    BacktestConfig as VRPBT,
    backtest_vrp,
    realized_variance,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def make_index(n=400, start="2023-01-02"):
    return pd.date_range(start, periods=n, freq="B")

def make_equity_panel(idx, names=tuple(f"N{i:02d}" for i in range(12)), seed=1):
    rng = np.random.default_rng(seed)
    n = len(idx); k = len(names)
    common = 0.7 * rng.standard_normal((n, 1))
    idio   = 0.6 * rng.standard_normal((n, k))
    r = 0.00025 + 0.012 * (common @ np.ones((1, k)) + idio)
    px = 100 * (1 + pd.DataFrame(r, index=idx, columns=names)).cumprod()
    idx_px = (px.mean(axis=1) * (1 + 0.001 * rng.standard_normal(n))).rename("INDEX")
    return idx_px, px

def make_iv_series(idx, base_atm=0.22, seed=7):
    rng = np.random.default_rng(seed)
    atm = base_atm + 0.02*np.sin(np.linspace(0, 2, len(idx))) + 0.01*rng.standard_normal(len(idx))
    rr  = 0.00 + 0.01*np.sin(np.linspace(0, 6, len(idx))) + 0.004*rng.standard_normal(len(idx))  # C-P
    bf  = -0.004 + 0.004*np.cos(np.linspace(0, 5, len(idx))) + 0.003*rng.standard_normal(len(idx))
    ivc25 = atm + bf + 0.5*rr
    ivp25 = atm + bf - 0.5*rr
    return pd.Series(ivp25, index=idx, name="P25"), pd.Series(atm, index=idx, name="ATM"), pd.Series(ivc25, index=idx, name="C25")

def pos_is_bounded(series, cap=1.0):
    s = series.dropna()
    return (s.abs() <= cap + 1e-9).all()

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_core_simulator_runs_and_respects_caps():
    idx = make_index(300)
    rng = np.random.default_rng(0)
    px = pd.DataFrame({
        "AAPL": 150*np.exp(np.cumsum(0.0003+0.02*rng.standard_normal(len(idx)))),
        "MSFT": 200*np.exp(np.cumsum(0.0002+0.018*rng.standard_normal(len(idx)))),
        "SPY":  300*np.exp(np.cumsum(0.00025+0.015*rng.standard_normal(len(idx)))),
    }, index=idx)

    def sig(hist: pd.DataFrame) -> pd.Series:
        m = hist.pct_change(60).iloc[-1]
        w = pd.Series(0.0, index=hist.columns); w[m.idxmax()] = 1.0
        return w

    out = simulate_backtest(prices=px, signal_func=sig, cfg=SimConfig(nav0=1_000_000, tc_bps=3.0, leverage_cap=1.5))
    summ = out["summary"]
    assert not summ.empty
    assert (summ["costs$"] >= 0).all()
    # weights gross <= cap
    gross = out["weights"].abs().sum(axis=1)
    assert (gross <= 1.5 + 1e-9).all()

def test_dispersion_backtest_and_weights_shape():
    idx = make_index(360)
    idx_px, panel = make_equity_panel(idx, names=tuple(f"N{i:02d}" for i in range(16)))
    # weights snapshot
    w_out = build_dispersion_weights(idx_px=idx_px, members_px=panel, cfg=DispCfg(lookback_days=60))
    assert not w_out["w_members"].empty
    assert abs(float(w_out["w_index"].iloc[0]) + w_out["w_members"].sum()) < 1e-8
    # backtest
    bt = backtest_dispersion(idx_px=idx_px, members_px=panel, cfg=DispCfg(lookback_days=60), bt=DispBT(rebalance="W-FRI"))
    assert "nav" in bt["summary"].columns
    assert bt["summary"]["nav"].notna().any()

def test_skew_signal_and_proxy_pnl():
    idx = make_index(320)
    ivp25, ivatm, ivc25 = make_iv_series(idx)
    rr_bf = build_rr_bf_from_iv(iv_put_25d=ivp25, iv_atm=ivatm, iv_call_25d=ivc25)

    # Underlying for realized skew
    rng = np.random.default_rng(3)
    px = pd.Series(4000*np.exp(np.cumsum(0.0002 + 0.012*rng.standard_normal(len(idx)))), index=idx)
    rskew = build_realized_skew(px, lookback=63)

    cfg = SkewConfig(mode="combo", side="mean_revert", z_lookback=126, cap=1.0, rebal_freq="W-FRI")
    out = backtest_skew_rr(rr_bf=rr_bf, realized_skew=rskew, cfg=cfg)
    summ = out["summary"]
    assert not summ.empty
    assert pos_is_bounded(summ["position"], cap=1.0)
    # PnL should move (non-zero variance)
    assert summ["pnl$"].var() > 0

def test_tail_hedge_triggers_and_budget_progress():
    idx = make_index(320)
    rng = np.random.default_rng(7)
    r = 0.0002 + 0.012*rng.standard_normal(len(idx))
    r[180:190] -= 0.03  # crash burst
    px = pd.Series(4000*np.exp(np.cumsum(r)), index=idx, name="SPX")
    vix = pd.Series(16 + 4*np.sin(np.linspace(0,5,len(idx))) + 6*(r< -0.02), index=idx, name="VIX").clip(10, 80)

    cfg = THCfg(trigger_mode="regime", dd_lookback=100, dd_trigger=-0.07, rv_lb=21, rv_trigger=0.28,
                budget_bps_per_year=150, ladder_tranches=3, put_tenor_days=45, put_moneyness=(0.95,0.90,0.85))
    bt = THBT(nav0=1_000_000, tc_bps=5.0, decay_daily=0.0015, shock_beta_put=7.0, roll_days=7)

    out = backtest_tail_hedge(index_px=px, vix_level=vix, cfg=cfg, bt=bt, hedge="combo")
    summ = out["summary"]
    assert not summ.empty
    # Budget should be spent at least once if triggers fire
    assert summ["budget_spent$"].iloc[-1] >= 0
    # NAV path should exist and vary
    assert summ["nav$"].var() > 0

def test_vrp_backtest_behaves_and_uses_vix():
    idx = make_index(520)
    rng = np.random.default_rng(11)
    vol_state = 0.012 + 0.008*(rng.standard_normal(len(idx)) > 1.1)
    r = 0.0002 + vol_state * rng.standard_normal(len(idx))
    px = pd.Series(4000*np.exp(np.cumsum(r)), index=idx, name="SPX")

    rv = realized_variance(px, 21)
    vix = (np.sqrt(rv.clip(0.0001)) * 100.0 + 4.0 + 2.0*np.sin(np.linspace(0,6,len(idx))))
    vix = pd.Series(vix, index=idx, name="VIX").clip(8, 80)

    cfg = VRPConfig(mode="vix_only", side="carry_short", rebal_freq="W-FRI",
                    unit_gross=1.0, cap=1.0, tc_bps=5.0, var_notional_per_unit=1_000_000)
    bt = VRPBT(nav0=1_000_000, hold_days=21, slippage_bps=1.0, use_log_returns=True)

    out = backtest_vrp(underlying=px, vix_30d=vix, cfg=cfg, bt=bt)
    summ = out["summary"]
    assert not summ.empty
    # Position bounded and nontrivial
    assert pos_is_bounded(summ["position"], cap=1.0)
    assert summ["pnl$"].abs().sum() != 0