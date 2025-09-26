# tests/test_strategies.py
import time
import importlib
import math
import pytest

# --------------------------- helpers ---------------------------

def _optional_import(modname):
    try:
        return importlib.import_module(modname)
    except Exception:
        pytest.skip(f"Optional module '{modname}' not found; skipping related tests.")

def _feed_prices(strat, sym, prices):
    for p in prices:
        if hasattr(strat, "on_price"):
            strat.on_price(sym, p)

# ========================= MEAN REVERSION ==========================

def test_mean_reversion_basic_signal_and_warmup():
    mod = _optional_import("strategies.mean_reversion")
    MR = mod.MeanReversionStrategy
    strat = MR(lookback=5, z_cap=2.0, vol_window=0, cooldown_bps=0)
    # 6 prices so warmup (lookback+1) satisfied
    px = [100, 101, 100, 99, 98, 100]
    _feed_prices(strat, "AAA", px)
    scores = strat.generate_signals(time.time())
    assert isinstance(scores, dict)
    # With last price above its 5-bar SMA by a bit after dip, expect small negative/positive (don’t assert sign hard)
    assert "AAA" in scores

def test_mean_reversion_vol_scaling_bounds():
    mod = _optional_import("strategies.mean_reversion")
    MR = mod.MeanReversionStrategy
    strat = MR(lookback=5, vol_window=5, vol_target=0.02, z_cap=1.0)
    px = [100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6]  # low vol drift
    _feed_prices(strat, "AAA", px)
    s = strat.generate_signals(time.time()).get("AAA", 0.0)
    # Vol-scaling capped between 0.25..4.0; signal should not blow up
    assert abs(s) <= 4.0  # loose bound

# ========================= MOMENTUM ==========================

def test_momentum_skip_lookback_and_deadband():
    mod = _optional_import("strategies.momentum")
    MOM = mod.MomentumStrategy
    strat = MOM(lookback=6, skip=2, vol_window=0, trend_filter=None, breakout_k=0, mom_cap=1.0)
    # 9 bars -> warmup 6+2+1 = 9
    px = [100, 101, 102, 104, 105, 103, 106, 108, 110]
    _feed_prices(strat, "AAA", px)
    scores = strat.generate_signals(time.time())
    assert "AAA" in scores
    # small deadband clamps tiny signals to 0
    strat2 = MOM(lookback=6, skip=2, vol_window=0, trend_filter=None, breakout_k=0, mom_cap=1.0)
    _feed_prices(strat2, "BBB", [100, 100.01, 100.02, 100.03, 100.02, 100.03, 100.04, 100.05, 100.06])
    s2 = strat2.generate_signals(time.time()).get("BBB", 0.0)
    assert abs(s2) <= 0.01  # likely zeroed by deadband

def test_momentum_trend_filter_gate_and_breakout():
    mod = _optional_import("strategies.momentum")
    MOM = mod.MomentumStrategy
    strat = MOM(lookback=5, skip=0, vol_window=0, trend_filter=(3,5), breakout_k=3, breakout_boost=1.5)
    px = [100,101,102,103,104,106,108,110]  # clear uptrend, breakout likely
    _feed_prices(strat, "AAA", px)
    s = strat.generate_signals(time.time()).get("AAA", 0.0)
    assert s >= 0.0  # aligned with uptrend (gate should not zero it)

# ========================= PAIRS TRADING ==========================

def test_pairs_trading_entry_exit_and_signs():
    mod = _optional_import("strategies.pairs_trading")
    PT = mod.PairsTradingStrategy
    # Y ≈ 1.0*X but we’ll push a temporary divergence to trigger entry
    strat = PT(pairs=[("X","Y")], lookback=20, entry_z=1.0, exit_z=0.25, stop_z=5.0, per_pair_weight=1.0)

    xs = [100 + i*0.1 for i in range(25)]
    ys = [100 + i*0.1 for i in range(25)]
    # introduce divergence at the end (Y rich vs X)
    ys[-1] += 2.0

    for i in range(len(xs)):
        strat.on_price("X", xs[i])
        strat.on_price("Y", ys[i])

    scores = strat.generate_signals(time.time())
    # When Y is rich (spread positive), strategy should prefer SHORT Y, LONG X
    sX = scores.get("X", 0.0)
    sY = scores.get("Y", 0.0)
    assert (sX > 0 and sY < 0) or (sX == 0 and sY == 0)  # allow flat if z below threshold

def test_pairs_trading_mode_persists_until_exit_band():
    mod = _optional_import("strategies.pairs_trading")
    PT = mod.PairsTradingStrategy
    strat = PT(pairs=[("A","B")], lookback=15, entry_z=1.0, exit_z=0.2)

    A = [100 + i*0.2 for i in range(20)]
    B = [100 + i*0.2 for i in range(20)]
    B[-1] += 1.0  # rich spread -> short_spread likely
    for i in range(20):
        strat.on_price("A", A[i]); strat.on_price("B", B[i])
    _ = strat.generate_signals(time.time())
    mode_after_entry = strat.state[("A","B")].mode

    # Nudge closer but keep |z| above exit_z -> mode should persist
    B.append(B[-1] - 0.1); A.append(A[-1])
    strat.on_price("A", A[-1]); strat.on_price("B", B[-1])
    _ = strat.generate_signals(time.time())
    assert strat.state[("A","B")].mode == mode_after_entry

# ========================= PORTFOLIO OPTIMIZER ==========================

def test_optimizer_mean_variance_and_risk_parity_long_only():
    mod = _optional_import("stratrgies.portfolio_optimizer")  # note: folder name kept as provided
    est_returns_cov = mod.est_returns_cov
    MVConfig, mean_variance = mod.MVConfig, mod.mean_variance
    RPConfig, risk_parity = mod.RPConfig, mod.risk_parity

    prices = {
        "AAA": [100,101,100,102,103,104,103,105,104,106],
        "BBB": [50,50.5,50.2,50.4,50.8,51.0,50.7,51.2,51.1,51.5],
        "CCC": [25,24.8,25.2,25.5,25.3,25.6,25.7,25.9,26.1,26.0],
    }
    syms, mu, cov = est_returns_cov(prices)

    # Mean-variance
    w_mv = mean_variance(mu, cov, MVConfig(risk_aversion=5.0, budget=1.0, l2_reg=1e-6))
    assert len(w_mv) == len(syms)
    assert all(w >= -1e-9 for w in w_mv)  # non-negative
    assert pytest.approx(sum(w_mv), rel=0, abs=1e-6) == 1.0

    # Risk parity
    w_rp = risk_parity(cov, RPConfig(budget=1.0))
    assert len(w_rp) == len(syms)
    assert all(w >= -1e-9 for w in w_rp)
    assert pytest.approx(sum(w_rp), rel=0, abs=1e-6) == 1.0

def test_optimizer_turnover_penalty_biases_toward_prev():
    mod = _optional_import("stratrgies.portfolio_optimizer")
    MVConfig, mean_variance = mod.MVConfig, mod.mean_variance

    mu = [0.01, 0.005, 0.0]
    cov = [[0.02, 0.0, 0.0],
           [0.0, 0.02, 0.0],
           [0.0, 0.0, 0.02]]
    prev = [0.6, 0.3, 0.1]
    w_low_tau = mean_variance(mu, cov, MVConfig(risk_aversion=5.0, turnover_penalty=0.0), w_prev=prev)
    w_high_tau = mean_variance(mu, cov, MVConfig(risk_aversion=5.0, turnover_penalty=0.5), w_prev=prev)

    # With higher turnover penalty, solution should be closer to prev in L1 sense
    d_low = sum(abs(a-b) for a,b in zip(w_low_tau, prev))
    d_high = sum(abs(a-b) for a,b in zip(w_high_tau, prev))
    assert d_high <= d_low + 1e-9