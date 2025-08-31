# backend/policies/tests/test_policies.py
from __future__ import annotations

"""
Behavior tests for policy layer:
- RiskPolicies (VaR/ES/LVaR caps, drawdown kill)
- ExecutionPolicy (algo & slicing)
- RoutingRules (venue selection)
- Throttle (rate limiting)
- PolicyEngine end-to-end (plan() -> actions)

Run:
  pytest -q backend/policies/tests/test_policies.py
"""

import math
import time
import types
import pytest # type: ignore

# ---------------------------------------------------------------------
# Imports (skip tests cleanly if a module is missing in your tree)
# ---------------------------------------------------------------------

PolicyEngine = pytest.importorskip("backend.policies.policy_engine", reason="policy_engine.py not found").PolicyEngine
risk_pols_mod = pytest.importorskip("backend.policies.risk_policies", reason="risk_policies.py not found")
exec_policy_mod = pytest.importorskip("backend.policies.execution_policy", reason="execution_policy.py not found")
routing_mod = pytest.importorskip("backend.policies.routing_rules", reason="routing_rules.py not found")
throttle_mod = pytest.importorskip("backend.policies.throttle", reason="throttle.py not found")

RiskPolicies = risk_pols_mod.RiskPolicies
ExecutionPolicy = exec_policy_mod.ExecutionPolicy
RoutingRules = routing_mod.RoutingRules
Throttle = throttle_mod.Throttle

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def dummy_market():
    """Tiny market snapshot fixture."""
    return {
        "AAPL": {"price": 190.0, "spread_bps": 2.0, "adv": 6_000_000, "tox": 0.15, "venues":[
            {"id":"XNAS","lat_ms":1.2,"liq":1.0,"fee_bps":0.1,"tox":0.10},
            {"id":"EDGX","lat_ms":1.0,"liq":0.8,"fee_bps":0.0,"tox":0.20}
        ]},
        "TSLA": {"price": 250.0, "spread_bps": 3.0, "adv": 7_000_000, "tox": 0.25, "venues":[
            {"id":"XNAS","lat_ms":1.2,"liq":0.9,"fee_bps":0.1,"tox":0.22},
            {"id":"BATS","lat_ms":0.9,"liq":0.7,"fee_bps":0.0,"tox":0.28}
        ]},
    }

@pytest.fixture
def dummy_state():
    return {
        "nav": 2_000_000.0,
        "positions": {"AAPL": 500.0, "TSLA": 0.0},
        "pnl_1d": 0.0,
    }

@pytest.fixture
def dummy_signals():
    # scores in [-1,1]
    return {"AAPL": +0.7, "TSLA": +0.3}

@pytest.fixture
def risk_policies():
    # Sensible caps for tests
    return RiskPolicies(
        var_cap_frac=0.03,           # 3% 1d VaR cap
        es_cap_frac=0.05,            # 5% ES cap
        lvar_cap_bps=80.0,           # per-symbol LVaR cap
        max_dd_frac=0.10,            # 10% drawdown hard stop
        dd_speed_block=0.012,        # fast DD speed block
        max_gross=0.50,              # 50% gross exposure cap
        max_symbol_w=0.08,           # 8% per symbol
        turnover_cap=0.30,           # 30% NAV per rebalance
    )

@pytest.fixture
def execution_policy():
    # Thresholds tune algo selection (TWAP/POV/VWAP/IOC)
    return ExecutionPolicy(
        urgencies={"low": 0.25, "med": 0.50, "high": 0.80},
        size_vs_adv_breaks={"twap": 0.01, "vwap": 0.03, "pov": 0.10},
        default_child_ms=2_000,
        min_child_qty=10
    )

@pytest.fixture
def routing_rules():
    return RoutingRules(
        prefer_low_tox=True,
        latency_weight=0.4,
        fee_weight=0.1,
        liquidity_weight=0.5
    )

@pytest.fixture
def throttle():
    return Throttle(max_orders_per_sec=10, burst=20)

@pytest.fixture
def engine(risk_policies, execution_policy, routing_rules, throttle):
    return PolicyEngine(
        risk=risk_policies,
        execution=execution_policy,
        routing=routing_rules,
        throttle=throttle
    )

# ---------------------------------------------------------------------
# Mocks for risk hooks (VaR/ES/LVaR)
# ---------------------------------------------------------------------

@pytest.fixture
def risk_hooks():
    class Hooks:
        def portfolio_var(self, proposal_notional: dict[str, float]) -> float:
            # rough: 2.8% when gross ~ 40% NAV, scale linearly with gross
            gross = sum(abs(v) for v in proposal_notional.values())
            nav = 2_000_000.0
            g = gross / nav
            return 0.028 * (g / 0.40)
        def portfolio_es(self, proposal_notional: dict[str, float]) -> float:
            # ES a bit larger
            gross = sum(abs(v) for v in proposal_notional.values())
            nav = 2_000_000.0
            g = gross / nav
            return 0.045 * (g / 0.40)
        def symbol_lvar_bps(self, sym: str, notional: float) -> float:
            # more notional → higher LVaR bps (very rough)
            base = 40.0 if sym == "AAPL" else 55.0
            bump = 20.0 * (notional / 200_000.0)
            return base + bump
    return Hooks()

# ---------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------

def test_risk_caps_reduce_gross(engine, dummy_market, dummy_state, dummy_signals, risk_hooks):
    # Ask for high risk_mult to see the cap kick in
    plan = engine.plan(
        signals=dummy_signals,
        market=dummy_market,
        state=dummy_state,
        risk_mult=1.0,
        risk_hooks={
            "portfolio_var_fn": risk_hooks.portfolio_var,
            "portfolio_es_fn": risk_hooks.portfolio_es,
            "symbol_lvar_bps_fn": risk_hooks.symbol_lvar_bps
        }
    )
    assert "target_weights" in plan and plan["target_weights"]
    gross = sum(abs(w) for w in plan["target_weights"].values())
    assert gross <= engine.risk.max_gross + 1e-6
    # per-symbol cap respected
    assert all(abs(w) <= engine.risk.max_symbol_w + 1e-9 for w in plan["target_weights"].values())

def test_drawdown_block_triggers_kill(engine, dummy_market, dummy_state, dummy_signals, risk_hooks, monkeypatch):
    # simulate a risk feed that signals a fast drawdown regime
    dd_alert = {"level": "block", "suggested_risk_mult": 0.2}
    plan = engine.plan(
        signals=dummy_signals,
        market=dummy_market,
        state=dummy_state,
        risk_mult=1.0,
        dd_alert=dd_alert,
        risk_hooks={
            "portfolio_var_fn": risk_hooks.portfolio_var,
            "portfolio_es_fn": risk_hooks.portfolio_es,
            "symbol_lvar_bps_fn": risk_hooks.symbol_lvar_bps
        }
    )
    # applied multiplier should reflect dd cut
    assert plan["applied_multiplier"] <= 0.25
    # optional: engine may emit a kill flag or zero out trades on "halt"
    dd_alert_halt = {"level": "halt", "suggested_risk_mult": 0.0}
    plan2 = engine.plan(
        signals=dummy_signals,
        market=dummy_market,
        state=dummy_state,
        risk_mult=1.0,
        dd_alert=dd_alert_halt
    )
    assert sum(abs(w) for w in plan2["target_weights"].values()) < 1e-9

def test_execution_policy_algo_choice(execution_policy, dummy_market):
    # Large order vs ADV → expect POV
    sym = "AAPL"
    px = dummy_market[sym]["price"]
    adv = dummy_market[sym]["adv"]
    size = 0.12 * adv
    rec = execution_policy.recommend(symbol=sym, side="buy", qty=size, px=px, adv=adv, urgency=0.7)
    assert rec["algo"] in ("POV", "POV-ADAPT")
    # Tiny order, low urgency → TWAP/VWAP
    rec2 = execution_policy.recommend(symbol=sym, side="sell", qty=0.004*adv, px=px, adv=adv, urgency=0.2)
    assert rec2["algo"] in ("TWAP","VWAP")

def test_routing_rule_prefers_liquidity_low_tox(routing_rules, dummy_market):
    venues = dummy_market["TSLA"]["venues"]
    choice = routing_rules.pick(symbol="TSLA", side="buy", venues=venues)
    assert choice and "id" in choice
    # Should not pick the obviously worst (highest tox & low liq) if a better exists
    worst = max(venues, key=lambda v: (v["tox"], -v["liq"]))
    assert choice["id"] != worst["id"]

def test_throttle_allows_burst(throttle):
    # Allow burst then enforce rate
    ok = [throttle.allow("orders") for _ in range(15)]
    assert sum(ok) >= 10     # burst capacity
    # After many calls, rate should clamp
    hits_after = sum(throttle.allow("orders") for _ in range(100))
    assert hits_after < 100

def test_engine_emits_trades(engine, dummy_market, dummy_state, dummy_signals):
    plan = engine.plan(signals=dummy_signals, market=dummy_market, state=dummy_state, risk_mult=0.8)
    assert "trades" in plan
    # Trades should be directional with target_qty present
    for t in plan["trades"]:
        assert t["qty"] > 0
        assert t["side"] in ("buy","sell")
        assert "target_qty" in t

# ---------------------------------------------------------------------
# Marks for slower / optional checks
# ---------------------------------------------------------------------

@pytest.mark.slow
def test_turnover_cap(engine, dummy_market, dummy_state):
    # Ask engine to go from flat to sizable positions to test turnover clamp
    signals = {"AAPL": 1.0, "TSLA": 1.0}
    plan = engine.plan(signals=signals, market=dummy_market, state={"nav": 2_000_000.0, "positions": {}}, risk_mult=1.0)
    assert plan["diagnostics"]["gross_target"] <= engine.risk.max_gross + 1e-6
    # turnover should be reported if available (engine optional)
    if "turnover_est" in plan["diagnostics"]:
        assert plan["diagnostics"]["turnover_est"] <= engine.risk.turnover_cap + 1e-6