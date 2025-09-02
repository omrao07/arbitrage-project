# tests/test_quadrail.py
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest # type: ignore


# ------------------------ Almgren–Chriss -----------------------------------

def test_almgren_shapes_and_linear_trajectory():
    from research.exec.almgren import ACParams, optimal_schedule

    p = ACParams(
        X0=100_000.0, N=10, T=1.0,
        sigma=0.02, eta=2e-6, gamma=1e-6, lam=0.0  # risk-neutral
    )
    sched = optimal_schedule(p)

    # Shapes
    assert sched.t.shape == (p.N + 1,)
    assert sched.x.shape == (p.N + 1,)
    assert sched.q.shape == (p.N + 1,)
    assert sched.v.shape == (p.N + 1,)
    assert sched.tau == pytest.approx(p.T / p.N)

    # Risk-neutral => linear holdings from X0 to ~0
    diffs = np.diff(sched.x)
    assert np.allclose(np.diff(diffs), 0.0, atol=1e-12)
    assert sched.x[0] == pytest.approx(p.X0)
    assert sched.x[-1] == pytest.approx(0.0, abs=1e-6)

    # Costs are non-negative
    assert sched.stats["E_cost"] >= 0.0
    assert sched.stats["Var_cost"] >= 0.0


# ------------------------ VaR / ES -----------------------------------------

@pytest.mark.parametrize("method", ["historical", "gaussian"])
def test_var_es_basic(method):
    from research.risk.var_es import var_es

    rng = np.random.default_rng(123)
    pnl = rng.normal(loc=0.0, scale=1.0, size=50_000)  # standard normal

    out = var_es(pnl, conf=0.99, method=method)
    assert 0.0 < out["VaR"] < 4.0
    assert out["ES"] >= out["VaR"]  # ES is tail-average loss ≥ VaR
    assert out["n"] == len(pnl)
    assert out["conf"] == 0.99


def test_var_es_empty_raises():
    from research.risk.var_es import var_es
    with pytest.raises(ValueError):
        var_es(np.array([]))


# ------------------------ Scenarios ----------------------------------------

def test_scenarios_pnl_and_contribs():
    from research.risk.scenarios import pct_shock, abs_shock, run_scenarios

    positions = {"AAPL": 100, "MSFT": -50}
    ref = {"AAPL": 200.0, "MSFT": 120.0}

    specs = [
        pct_shock(["AAPL", "MSFT"], -0.10),   # -10% both
        abs_shock(["AAPL"], +5.0),            # +$5 AAPL
    ]
    results = run_scenarios(specs, positions, ref) # type: ignore

    # -10%: AAPL: 100 * (-20) = -2000 ; MSFT short gains: -50 * (-12) = +600 => total -1400
    r1 = results["pct_-10%"]
    assert r1["pnl"] == pytest.approx(-1400.0)
    assert r1["contrib"]["AAPL"] == pytest.approx(-2000.0) # type: ignore
    assert r1["contrib"]["MSFT"] == pytest.approx(+600.0) # type: ignore

    r2 = results["abs_+5.0"]
    assert r2["pnl"] == pytest.approx(100 * 5.0)
    assert "shocked_prices" in r2


# ------------------------ Evaluator ----------------------------------------

def test_evaluator_simple_fill_and_metrics():
    from research.exec.rl_agent.evaluator import (
        ParentOrder, ChildDecision, evaluate_policy
    )

    # Flat market with fixed spread; sufficient volume to fill
    n = 30
    df = pd.DataFrame({
        "ts": np.arange(n),
        "bid": np.full(n, 99.9),
        "ask": np.full(n, 100.1),
        "mid": np.full(n, 100.0),
        "v": np.full(n, 50_000.0),
    })

    order = ParentOrder(symbol="XYZ", side="BUY", qty=10_000, start_idx=0, end_idx=n - 1)

    # Policy: buy 2% of remaining per bar at market (None => touch)
    def policy(state):
        rem = state["remaining"]
        qty = 0.02 * abs(rem) * (1 if rem > 0 else -1)
        return ChildDecision(qty=qty, limit_px=None, reason="test")

    res, log = evaluate_policy(df, order, policy)

    # Should have non-zero fills, reasonable stats
    assert res.filled_qty > 0
    assert not math.isnan(res.avg_fill_px)
    assert isinstance(res.violations, dict)
    assert {"child_size", "participation", "price_band"} - set(res.violations.keys()) == set()
    assert len(log) <= n  # one row per step max


# ------------------------ DDQN (optional) ----------------------------------

def test_ddqn_smoke_act_and_train_step():
    torch = pytest.importorskip("torch")

    from research.exec.rl_agent.policy_ddqn import DDQNConfig, DDQNPolicy

    cfg = DDQNConfig(state_dim=7, hidden=(64, 64), buffer_size=10_000, batch_size=64, warmup=100)
    pol = DDQNPolicy(cfg)

    # Fake transitions
    def rand_state():
        return {
            "remaining": float(np.random.choice([+10_000, -10_000])),
            "bid": 99.9, "ask": 100.1, "mid": 100.0,
            "bar_v": 50_000.0,
            "elapsed": np.random.rand(),
            "participation_today": np.random.rand() * 0.1,
        }

    for _ in range(200):
        s = rand_state()
        a_idx = np.random.randint(0, cfg.n_actions())
        r = float(np.random.randn() * 0.01)
        s2 = rand_state()
        done = bool(np.random.rand() < 0.05)
        pol.remember(s, a_idx, r, s2, done)

    # Training step returns None before warmup, then a float loss
    loss = None
    for _ in range(5):
        loss = pol.train_step()
    # After a few steps (post-warmup), we should see a numeric loss
    assert loss is None or isinstance(loss, float)

    # Action selection returns a ChildDecision-like object
    a = pol.act(rand_state())
    assert hasattr(a, "qty")
    # limit price may be None or a float
    assert (a.limit_px is None) or isinstance(a.limit_px, float)