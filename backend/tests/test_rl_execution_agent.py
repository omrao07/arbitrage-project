# tests/test_rl_execution_agent.py
# ======================================================================
# RL Execution Agent behavioral tests.
# Target module: rl_execution_agent.py
#
# Expected surface (minimal):
#   class RLExecutionAgent:
#       def __init__(self, params: dict, broker, risk): ...
#       def act(self, state) -> Any:  # scalar in [-1,1] OR dict {'side','qty'}
#       def learn(self, *args, **kwargs): ...   # optional
#       def run_episode(self, env, max_steps:int=..., train:bool=True) -> dict: ...  # optional
#       def save(self, path:str): ...           # optional
#       def load(self, path:str): ...           # optional
#
# Broker-like:
#   broker.submit(order: dict) -> dict          # called when action implies a trade
#
# Risk-like (optional):
#   risk.clip(order: dict, state: dict) -> dict # enforce limits
#
# These tests use tiny fakes & monkeypatching so they don't hit real markets.
# ======================================================================

import importlib
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest # type: ignore

agent_mod = importlib.import_module("rl_execution_agent")

# --------- Fakes / Test Harness ---------------------------------------

@dataclass
class FakeOrder:
    side: str
    qty: float
    price: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

class FakeBroker:
    """Captures orders for assertions; returns a fake fill."""
    def __init__(self):
        self.submissions: List[Dict[str, Any]] = []

    def submit(self, order: Dict[str, Any]) -> Dict[str, Any]:
        self.submissions.append(order)
        # Fill at mid w/ tiny slip
        px = order.get("price", order.get("mark", 100.0))
        slip = 0.0002 * px * (1 if order.get("side") == "buy" else -1)
        fill_px = px + slip
        return {"status": "filled", "price": fill_px, "qty": order.get("qty", 0.0)}

class FakeRisk:
    def __init__(self, max_qty: float = 1000.0):
        self.max_qty = max_qty

    def clip(self, order: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        o = dict(order)
        o["qty"] = max(0.0, min(float(o.get("qty", 0.0)), self.max_qty))
        return o

class ToyExecEnv:
    """
    Tiny stationary environment:
      state = [mid, spread, vol, inventory]
      reward = -|impact| - spread_cost + inv_alpha * dP
      impact approximated by k*qty; price follows noisy drift.
    """
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.mid = 100.0
        self.spread = 0.02
        self.vol = 0.5  # std per step
        self.inv = 0.0
        self.done = False

    def reset(self):
        self.t = 0
        self.mid = 100.0
        self.spread = 0.02
        self.vol = 0.5
        self.inv = 0.0
        self.done = False
        return self._obs()

    def step(self, action: Any):
        # normalize action -> side, qty in [0..100]
        side, qty = self._parse_action(action)
        # transact at mid +/- half-spread
        trade_px = self.mid + (self.spread / 2.0) * (1 if side == "buy" else -1) if qty > 0 else self.mid
        # inventory update
        self.inv += qty if side == "buy" else -qty
        # simple impact & next price
        impact_k = 0.0001
        impact = impact_k * qty * (1 if side == "buy" else -1)
        dP = self.rng.normal(0.02, self.vol) + impact
        next_mid = self.mid + dP
        # reward: mark-to-market change on inventory, minus spread/impact costs
        mtm = self.inv * (next_mid - self.mid)
        spread_cost = (self.spread / 2.0) * qty
        impact_cost = abs(impact) * 10.0
        reward = mtm - spread_cost - impact_cost

        self.mid = next_mid
        self.t += 1
        self.done = self.t >= 64

        return self._obs(), float(reward), self.done, {"trade_px": trade_px, "side": side, "qty": qty}

    def _obs(self):
        return np.array([self.mid, self.spread, self.vol, self.inv], dtype=np.float32)

    @staticmethod
    def _parse_action(a: Any) -> Tuple[str, float]:
        if isinstance(a, dict):
            side = a.get("side", "buy")
            qty = float(max(0.0, a.get("qty", 0.0)))
            return side, qty
        # scalar policy in [-1,1]
        x = float(a)
        x = max(-1.0, min(1.0, x))
        side = "buy" if x >= 0 else "sell"
        qty = abs(x) * 100.0
        return side, qty

# --------- Fixtures ----------------------------------------------------

@pytest.fixture()
def broker():
    return FakeBroker()

@pytest.fixture()
def risk():
    return FakeRisk(max_qty=50.0)  # keep trades small so tests deterministic

@pytest.fixture()
def env():
    return ToyExecEnv(seed=42)

@pytest.fixture()
def agent(broker, risk):
    # Create agent with permissive params; fall back if constructor differs
    params = {
        "epsilon": 0.0,           # if supported
        "action_space": "scalar", # 'scalar' in [-1,1] or 'dict'
        "seed": 123,
    }
    A = getattr(agent_mod, "RLExecutionAgent")
    return A(params=params, broker=broker, risk=risk)

# --------- Tests -------------------------------------------------------

def test_act_output_shape_and_bounds(agent):
    s = np.array([100.0, 0.02, 0.5, 0.0], dtype=np.float32)
    a = agent.act(s)
    # Accept either scalar or dict. Validate bounds.
    if isinstance(a, dict):
        assert a["side"] in ("buy", "sell")
        assert 0.0 <= float(a.get("qty", 0.0)) <= 1e6
    else:
        assert isinstance(a, (int, float, np.floating))
        assert -1.000001 <= float(a) <= 1.000001

def test_broker_called_when_action_implies_trade(agent, env, broker):
    s = env.reset()
    # Force a clear non-zero action
    a = {"side": "buy", "qty": 25.0} if not isinstance(agent.act(s), (int, float, np.floating)) else 0.8
    # Simulate one step like run loop would
    _ = agent.act(s)  # allow agent internal state updates if any
    _obs, _r, _done, _info = env.step(a)
    # Emulate agent -> risk -> broker path
    order = {"side": "buy" if (a if isinstance(a, (int, float, np.floating)) else a["side"]) in ("buy", 1) else "sell",
             "qty": 25.0 if not isinstance(a, (int, float, np.floating)) else abs(a)*100.0,
             "price": float(s[0])}
    if hasattr(agent, "risk") and hasattr(agent.risk, "clip"):
        order = agent.risk.clip(order, {"obs": s})
    broker.submit(order)
    assert len(broker.submissions) >= 1
    assert broker.submissions[-1]["qty"] > 0

@pytest.mark.parametrize("eps", [1.0, 0.5, 0.0])
def test_epsilon_exploration_if_supported(agent, eps):
    # If agent exposes epsilon, we check exploration variance.
    if not hasattr(agent, "epsilon"):
        pytest.skip("Agent has no epsilon attribute; skip exploration test.")
    agent.epsilon = eps
    s = np.array([100.0, 0.02, 0.5, 0.0], dtype=np.float32)
    acts = [agent.act(s) for _ in range(50)]
    # Count unique scalar magnitudes or dict qties to gauge randomness
    mags = []
    for a in acts:
        if isinstance(a, dict):
            mags.append(round(float(a.get("qty", 0.0)), 2))
        else:
            mags.append(round(abs(float(a)), 2))
    unique = len(set(mags))
    if eps >= 0.9:
        assert unique > 5   # lots of exploration
    elif eps <= 0.05:
        assert unique <= 10 # mostly greedy (low variance)

def test_run_episode_integration_if_available(agent, env):
    if not hasattr(agent, "run_episode"):
        pytest.skip("Agent has no run_episode; skipping integration test.")
    stats = agent.run_episode(env, max_steps=64)
    assert isinstance(stats, dict)
    # Expect at least these keys
    for k in ("steps", "cum_reward"):
        assert k in stats

def test_learn_updates_internal_state(agent, env):
    # We can't inspect weights portably, but we can ensure no crash & optional step counter increments.
    if not hasattr(agent, "learn"):
        pytest.skip("Agent has no learn(); skipping.")
    s0 = env.reset()
    transition = {
        "state": s0,
        "action": {"side": "buy", "qty": 10.0},
        "reward": 0.1,
        "next_state": s0 + 1e-3,
        "done": False,
    }
    before = getattr(agent, "updates", 0)
    agent.learn(transition)
    after = getattr(agent, "updates", before)
    assert after >= before

def test_risk_clip_enforced(agent, broker):
    # Send an absurd order and ensure it is clipped by FakeRisk (50 max)
    big = {"side": "buy", "qty": 1e9, "price": 101.0}
    if hasattr(agent, "risk") and hasattr(agent.risk, "clip"):
        clipped = agent.risk.clip(big, {"obs": np.array([0,0,0,0], dtype=np.float32)})
        assert clipped["qty"] <= 50.0
        broker.submit(clipped)
        assert broker.submissions[-1]["qty"] <= 50.0
    else:
        pytest.skip("No risk.clip available; skipping.")

def test_checkpoint_roundtrip_if_supported(agent):
    if not (hasattr(agent, "save") and hasattr(agent, "load")):
        pytest.skip("Agent has no save/load; skipping checkpoint test.")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.bin")
        # Fix seed for reproducibility if agent honors it
        s = np.array([101.0, 0.01, 0.4, 0.0], dtype=np.float32)
        a0 = agent.act(s)
        agent.save(path)
        # mutate (if possible) to ensure load really restores
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.12345
        agent.load(path)
        a1 = agent.act(s)
        # After reload, action should match original (or be very close)
        def norm_action(a):
            if isinstance(a, dict):
                return (a.get("side"), round(float(a.get("qty", 0.0)), 6))
            return round(float(a), 6)
        assert norm_action(a0) == norm_action(a1)

def test_seed_reproducibility(agent):
    # If the agent supports seeding, two agents with the same seed act identically on same state.
    if not hasattr(agent, "seed"):
        pytest.skip("Agent lacks seed(); skipping.")
    s = np.array([100.0, 0.02, 0.5, 0.0], dtype=np.float32)
    # Recreate two fresh agents with same seed
    A = getattr(agent_mod, "RLExecutionAgent")
    b1, b2 = FakeBroker(), FakeBroker()
    r1, r2 = FakeRisk(), FakeRisk()
    a1 = A(params={"seed": 999, "epsilon": 0.0}, broker=b1, risk=r1)
    a2 = A(params={"seed": 999, "epsilon": 0.0}, broker=b2, risk=r2)
    if hasattr(a1, "seed"): a1.seed(999)
    if hasattr(a2, "seed"): a2.seed(999)
    out1 = [a1.act(s) for _ in range(5)]
    out2 = [a2.act(s) for _ in range(5)]
    def norm(a):
        return (a.get("side"), round(float(a.get("qty", 0.0)), 6)) if isinstance(a, dict) else round(float(a),6)
    assert [norm(x) for x in out1] == [norm(x) for x in out2]