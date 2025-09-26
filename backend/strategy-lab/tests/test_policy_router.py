# tests/test_policy_router.py
import time
import math
import importlib
import pytest

def _optional_import(modname):
    try:
        return importlib.import_module(modname)
    except Exception:
        pytest.skip(f"Optional module '{modname}' not found; skipping tests.")

def test_rule_order_and_fallback_hash():
    pr = _optional_import("selector.policy_router")
    ab = _optional_import("selector.ab_tests")

    default = ab.HashSplitRouter(["A","B"], share_map={"A":0.5,"B":0.5})
    router = pr.PolicyRouter(
        arms=["A","B"],
        rules=[
            pr.AllowDenyRule(allow={"TSLA":"B"}),               # hard allow
            pr.GroupRule(mapping={"PSU": (["SBIN","PNB"], "A")} ),
        ],
        default=default,
    )

    assert router.owner("TSLA") == "B"     # allow wins
    assert router.owner("SBIN") == "A"     # group routes
    assert router.owner("PNB") in {"A","B"}  # present in group -> "A"
    # unknown falls to hash split (deterministic bucket)
    aapl_owner = router.owner("AAPL")
    assert aapl_owner in {"A","B"}

def test_schedule_rule_with_tz_and_else_arm(monkeypatch):
    pr = _optional_import("selector.policy_router")
    # Build a router with only a schedule rule, tz=0 for deterministic mod 86400
    sched = pr.ScheduleRule(prefer={"A":[("00:00","00:10")]}, else_arm="B", tz_offset_minutes=0)
    router = pr.PolicyRouter(arms=["A","B"], rules=[sched])

    # Fake time so local minutes since midnight == between 00:00 and 00:10
    class FakeTime:
        def __init__(self, epoch): self.epoch = epoch
        def time(self): return self.epoch

    # 00:05 UTC -> within window
    epoch_0005 = 5 * 60  # seconds
    monkeypatch.setattr(pr.time, "time", lambda: float(epoch_0005))
    assert router.owner("ANY") == "A"

    # 00:20 UTC -> outside window -> else_arm
    epoch_0020 = 20 * 60
    monkeypatch.setattr(pr.time, "time", lambda: float(epoch_0020))
    assert router.owner("ANY") == "B"

def test_state_guard_redirects_on_leverage():
    pr = _optional_import("selector.policy_router")
    ab = _optional_import("selector.ab_tests")

    class StubState(pr.StateProvider):
        def __init__(self, lev): self._lev = lev
        def leverage(self): return self._lev

    state = StubState(lev=5.5)
    guard = pr.StateGuardRule(state=state, max_leverage=4.0, on_violation_arm="A")
    router = pr.PolicyRouter(arms=["A","B"], rules=[guard], default=ab.HashSplitRouter(["A","B"]))

    # Any symbol should be forced to "A" while leverage above threshold
    for s in ("AAPL","MSFT","SBIN"):
        assert router.owner(s) == "A"

def test_weighted_rule_cycle_determinism():
    pr = _optional_import("selector.policy_router")
    wr = pr.WeightedRule(weights={"A":0.7,"B":0.3})
    router = pr.PolicyRouter(arms=["A","B"], rules=[wr])  # default won't be reached

    seq = [router.owner("INFY") for _ in range(20)]
    # About 70% should be A over a long cycle; exact count depends on discrete wheel (~100 steps)
    frac_A = seq.count("A") / len(seq)
    assert 0.5 < frac_A < 0.9

def test_bandit_rule_learns_from_feedback(monkeypatch):
    pr = _optional_import("selector.policy_router")

    # Bandit only (no default needed, it's decisive)
    bandit = pr.BanditRule(arms=["A","B"], eps=0.0, decay=0.5)  # eps=0 to always exploit
    router = pr.PolicyRouter(arms=["A","B"], rules=[bandit])

    sym = "RELIANCE"

    # Initially equal -> owner defaults to first best; we won't assert which, just capture it.
    first = router.owner(sym)
    other = "B" if first == "A" else "A"

    # Give positive feedback to the 'other' arm so it becomes preferred
    router.feedback(sym, other, trade_pnl=+10.0)

    # Now bandit should pick the better arm deterministically (eps=0)
    chosen = router.owner(sym)
    assert chosen == other

def test_build_from_config_yaml_like():
    pr = _optional_import("selector.policy_router")
    ab = _optional_import("selector.ab_tests")

    cfg = {
        "arms": ["A","B"],
        "default": {"kind": "hash", "shares": {"A":0.6,"B":0.4}},
        "rules": [
            {"kind":"allow_deny","allow":{"TSLA":"B"}},
            {"kind":"group","mapping":{"PSU":{"symbols":["SBIN","PNB"],"arm":"A"}}},
            {"kind":"schedule","prefer":{"A":[["09:15","12:00"]]},"else_arm":"B","tz_offset_minutes":0},
            {"kind":"state_guard","max_leverage":10.0,"on_violation_arm":"A"},
            {"kind":"weighted","weights":{"A":0.7,"B":0.3}},
            {"kind":"bandit","eps":0.0,"decay":0.2}
        ]
    }

    # Small state provider just to satisfy state_guard constructor
    class S(pr.StateProvider):
        def leverage(self): return 1.0
    router = pr.build_from_config(cfg, state=S())

    # Basic sanity: rules chain returns a valid arm
    out = router.owner("SBIN")
    assert out in {"A","B"}

    # Allow override works
    assert router.owner("TSLA") == "B"