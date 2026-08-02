# backend/tests/test_live.py
"""Tests for the generic/paper path of backend/live_engine/."""
import time

from backend.live_engine import LiveEngine, RiskGates, SignalAggregator

# ---- RiskGates tests -------------------------------------------------------

def test_gate1_daily_loss_ok():
    rg = RiskGates(capital=100_000, daily_loss_limit_pct=2.0)
    rg.update_pnl(-1_000)  # -1% loss, within limit
    ok, reason = rg.gate1_daily_loss()
    assert ok


def test_gate1_daily_loss_breach():
    rg = RiskGates(capital=100_000, daily_loss_limit_pct=2.0)
    rg.update_pnl(-2_500)  # -2.5% loss, exceeds limit
    ok, reason = rg.gate1_daily_loss()
    assert not ok
    assert "daily_loss" in reason


def test_gate2_drawdown_ok():
    rg = RiskGates(capital=100_000, drawdown_limit_pct=10.0)
    rg.update_pnl(5_000)   # profit first
    rg.update_pnl(-5_000)  # drawdown from HWM = -5000 (5%), within limit
    ok, _ = rg.gate2_drawdown()
    assert ok


def test_gate2_drawdown_breach():
    rg = RiskGates(capital=100_000, drawdown_limit_pct=10.0)
    rg.update_pnl(20_000)    # HWM = 20000
    rg.update_pnl(-15_000)   # cumulative = 5000, drawdown = -15000 (15%)
    ok, reason = rg.gate2_drawdown()
    assert not ok
    assert "drawdown" in reason


def test_gate3_beta():
    rg = RiskGates(beta_limit=0.8)
    ok, _ = rg.gate3_beta(0.7)
    assert ok
    ok, _ = rg.gate3_beta(0.9)
    assert not ok


def test_gate4_position_size():
    rg = RiskGates(capital=1_000_000, position_pct_limit=5.0)
    ok, _ = rg.gate4_position_size("AAPL", 40_000)   # 4%
    assert ok
    ok, _ = rg.gate4_position_size("AAPL", 60_000)   # 6%
    assert not ok


def test_gate5_vix():
    rg = RiskGates(vix_halt_threshold=30.0)
    ok, _ = rg.gate5_vix(28.0)
    assert ok
    ok, reason = rg.gate5_vix(32.0)
    assert not ok


def test_gate7_order_rate():
    rg = RiskGates(order_rate_limit_per_min=5)
    for _ in range(5):
        ok, _ = rg.gate7_order_rate()
    ok, reason = rg.gate7_order_rate()  # 6th order → breach
    assert not ok


def test_gate9_circuit():
    rg = RiskGates()
    rg.update_circuit_halted(frozenset(["RELIANCE"]))
    ok, reason = rg.gate9_circuit("RELIANCE")
    assert not ok
    ok, _ = rg.gate9_circuit("TCS")
    assert ok


def test_gate_fo_ban():
    rg = RiskGates()
    rg.update_fo_ban(frozenset(["VEDL"]))
    ok, reason = rg.gate_fo_ban("VEDL")
    assert not ok
    ok, _ = rg.gate_fo_ban("INFY")
    assert ok


def test_kelly_size():
    rg = RiskGates(capital=1_000_000, kelly_fraction=0.25)
    size = rg.gate_kelly_size(win_rate=0.6, win_loss_ratio=1.5)
    assert size > 0
    assert size <= 1_000_000


# ---- SignalAggregator tests ------------------------------------------------

def test_aggregator_empty():
    sa = SignalAggregator()
    assert sa.aggregate() == {}
    assert sa.combined_score() == 0.0


def test_aggregator_single_signal():
    sa = SignalAggregator(mode="equal")
    sa.update("strat_a", score=0.8)
    agg = sa.aggregate()
    assert "strat_a" in agg
    assert abs(agg["strat_a"] - 0.8) < 1e-6


def test_aggregator_two_signals_net():
    sa = SignalAggregator(mode="equal")
    sa.update("long_strat", score=1.0)
    sa.update("short_strat", score=-1.0)
    score = sa.combined_score()
    assert abs(score) < 0.01  # should roughly cancel out


def test_aggregator_vol_mode():
    sa = SignalAggregator(mode="vol")
    sa.update("high_vol", score=0.5, vol=0.5)
    sa.update("low_vol", score=0.5, vol=0.1)
    agg = sa.aggregate()
    # low_vol strategy should have higher weight
    assert agg["low_vol"] > agg["high_vol"]


def test_aggregator_stale_signals():
    sa = SignalAggregator(max_signal_age_ms=100)
    sa.update("old_strat", score=0.9)
    time.sleep(0.15)
    active = sa._active_signals()
    assert "old_strat" not in active


# ---- LiveEngine + PaperBroker tests ----------------------------------------

def test_engine_paper_mode_default():
    engine = LiveEngine(capital=500_000, paper_mode=True)
    assert engine.paper_mode is True
    assert engine.broker is not None


def test_engine_live_mode_no_broker():
    engine = LiveEngine(capital=500_000, paper_mode=False)
    assert engine.paper_mode is False
    assert engine.broker is None


def test_engine_status_has_broker_key():
    engine = LiveEngine(capital=100_000, paper_mode=True)
    s = engine.status()
    assert "broker" in s
    assert "cash" in s["broker"]
    assert "equity" in s["broker"]


def test_engine_status_no_broker_key_in_live_mode():
    engine = LiveEngine(capital=100_000, paper_mode=False)
    s = engine.status()
    assert "broker" not in s


def test_engine_submit_order_paper():
    from backend.execution.brokers.paper import OrderRequest
    engine = LiveEngine(capital=100_000, paper_mode=True)
    req = OrderRequest(symbol="RELIANCE", side="buy", qty=10, price_hint=2500.0)
    result = engine.submit_order(req)
    assert result is not None
    assert result.status == "filled"
    assert result.symbol == "RELIANCE"


def test_engine_submit_order_live_mode_returns_none():
    from backend.execution.brokers.paper import OrderRequest
    engine = LiveEngine(capital=100_000, paper_mode=False)
    req = OrderRequest(symbol="TCS", side="buy", qty=5, price_hint=3500.0)
    result = engine.submit_order(req)
    assert result is None
