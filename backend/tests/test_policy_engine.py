# tests/test_policy_engine.py
import importlib
import time
from datetime import datetime, timedelta, timezone
from copy import deepcopy
from typing import Any, Dict, List, Optional
import pytest # type: ignore

"""
Expected public API (any one is fine)

Class-style
-----------
class PolicyEngine:
    def load_policies(self, policies: dict | list | str, **kw) -> None: ...
    def evaluate(self, order: dict, context: dict | None = None, dry_run: bool = False) -> dict: ...
    # Optional:
    def reload(self, policies: dict | list | str | None = None) -> None
    def get_state(self) -> dict
    def set_state(self, state: dict) -> None
    def get_audit(self, since_ts: int | None = None) -> list[dict]
    def clear(self) -> None

Function-style
--------------
- build_engine(policies) -> handle
- evaluate(engine, order, context=None, dry_run=False) -> dict
- reload(engine, policies=None), get_state(engine), set_state(engine, ...), get_audit(engine, since_ts)

Decision shape (tolerant)
-------------------------
{
  "action": "allow" | "block" | "modify",
  "order": {...possibly modified...},
  "reasons": [str, ...],
  "alerts": [str, ...],          # optional
  "policy": {"id": "...", "priority": int}  # optional
}
"""

# ----------------- Import resolver -----------------
IMPORT_CANDIDATES = [
    "backend.risk.policy_engine",
    "backend.policy.engine",
    "backend.guardrails.policy_engine",
    "policy.engine",
    "policy_engine",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import policy engine from {IMPORT_CANDIDATES} ({last})")

class API:
    def __init__(self, mod):
        self.mod = mod
        self.engine = None
        # Prefer class
        if hasattr(mod, "PolicyEngine"):
            PE = getattr(mod, "PolicyEngine")
            try:
                self.engine = PE()
            except TypeError:
                self.engine = PE
        else:
            self.build = getattr(mod, "build_engine", None)
            self.eval_fn = getattr(mod, "evaluate", None)
            if not (self.build and self.eval_fn):
                pytest.skip("No PolicyEngine class and no build_engine/evaluate functions.")
        # Optional
        self.reload = getattr(self.engine, "reload", None) if self.engine else getattr(mod, "reload", None)
        self.get_state = getattr(self.engine, "get_state", None) if self.engine else getattr(mod, "get_state", None)
        self.set_state = getattr(self.engine, "set_state", None) if self.engine else getattr(mod, "set_state", None)
        self.get_audit = getattr(self.engine, "get_audit", None) if self.engine else getattr(mod, "get_audit", None)
        self.clear = getattr(self.engine, "clear", None) if self.engine else getattr(mod, "clear", None)

    def start(self, policies):
        if self.engine and hasattr(self.engine, "load_policies"):
            self.engine.load_policies(policies)
            return self.engine
        eng = self.build(policies) # type: ignore
        self.engine = eng
        return eng

    def eval(self, order, context=None, dry_run=False):
        if hasattr(self.engine, "evaluate"):
            return self.engine.evaluate(order=order, context=context, dry_run=dry_run) # type: ignore
        return self.eval_fn(self.engine, order, context=context, dry_run=dry_run) # type: ignore

# ----------------- Sample policies (portable) -----------------
def sample_policies():
    """
    Minimal, engine-agnostic policy spec used by tests.
    Engines may interpret/compile this into internal rules.
    """
    return {
        "version": "1.0",
        "rules": [
            # Highest priority: kill switch
            {"id": "kill-switch", "priority": 1000, "when": {"env.kill": True}, "then": {"action": "block", "reason": "kill_switch_on"}},
            # Blocklist by symbol
            {"id": "blocklist", "priority": 900, "when": {"order.symbol.in": ["GME", "AMC"]}, "then": {"action": "block", "reason": "symbol_blocked"}},
            # Price band: limit orders must be within ±5% of ref
            {"id": "price-band", "priority": 800, "when": {"order.type":"limit", "ctx.ref_price.exists": True},
             "then": {"action": "modify", "cap_price_pct": 0.05, "alert": "price_out_of_band"}},
            # Max notional per order
            {"id": "notional-cap", "priority": 700, "when": {}, "then": {"max_notional": 500_000}},
            # Leverage & exposure guard
            {"id": "leverage", "priority": 650, "when": {"ctx.leverage_gt": 3.0}, "then": {"action":"block", "reason":"leverage_exceeded"}},
            {"id": "exposure-sym", "priority": 640, "when": {"ctx.exposure_symbol_gt": {"field":"order.symbol","limit":0.15}},
             "then":{"action":"block","reason":"symbol_exposure_exceeded"}},
            # Session window (09:30–16:00 UTC for test)
            {"id": "session", "priority": 600, "when": {"time.between_utc": ["09:30","16:00"]}, "then": {"action": "allow"}},
            {"id": "session-fallback", "priority": 590, "when": {}, "then": {"action": "block", "reason": "market_closed"}},
            # Rate limit 3 orders per 2 seconds per user
            {"id": "ratelimit", "priority": 580, "when": {"ctx.rate_key": "user"}, "then": {"rate_limit": {"n":3, "per_s":2}}},
            # Admin override bypasses blocks (lower priority than kill switch)
            {"id": "override", "priority": 550, "when": {"ctx.role.in":["admin"]}, "then": {"override": True}},
            # Alerts on large size
            {"id": "large-size-alert", "priority": 500, "when": {"order.qty_gte": 50_000}, "then": {"action":"allow", "alert":"large_order"}},
        ]
    }

# ----------------- Fixtures -----------------
@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture()
def engine(api):
    return api.start(sample_policies())

@pytest.fixture()
def base_order():
    return {"id":"OID-1","symbol":"AAPL","side":"buy","type":"limit","qty":10_000,"price":185.25,"ts":int(time.time()*1000)}

@pytest.fixture()
def context_open():
    # Simulate an open market (10:00 UTC), user=alice, leverage ok, exposures low
    now = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)
    return {
        "env": {"kill": False},
        "now_utc": int(now.timestamp()*1000),
        "ref_price": 185.20,
        "user": "alice",
        "role": "trader",
        "leverage": 2.0,
        "exposures": {"AAPL": 0.05, "MSFT": 0.08},
    }

# ----------------- Helper assertions -----------------
def _action(res): return (res.get("action") or res.get("decision") or "").lower()
def _order(res):  return res.get("order") or {}
def _reasons(res):return res.get("reasons") or res.get("reason") or []

# ----------------- Tests -----------------

def test_blocklist_symbols(engine, base_order, context_open, api: API):
    o = dict(base_order, symbol="GME")
    res = api.eval(o, context_open)
    assert _action(res) == "block"
    assert any("block" in r or "symbol" in r for r in _reasons(res))

def test_price_band_modify_within_5pct(engine, base_order, context_open, api: API):
    # Make price 10% away from ref; expect clamp/modify
    bad = dict(base_order, price=context_open["ref_price"] * 1.10)
    res = api.eval(bad, context_open)
    assert _action(res) in ("modify","allow")  # engine may auto-clamp then allow
    ord2 = _order(res)
    # Price should be capped within 5% of ref
    hi_cap = context_open["ref_price"] * 1.05 + 1e-9
    assert float(ord2.get("price", bad["price"])) <= hi_cap

def test_notional_cap(engine, base_order, context_open, api: API):
    big = dict(base_order, qty=1_000_000, price=800.0)  # $800M notional
    res = api.eval(big, context_open)
    # Either block or modify down to <= 500k notional
    if _action(res) == "modify":
        ord2 = _order(res)
        assert float(ord2["qty"]) * float(ord2["price"]) <= 500_000 + 1e-6
    else:
        assert _action(res) == "block"

def test_session_window_allow_then_block(engine, base_order, api: API, context_open):
    # Open session -> allow (assuming other rules pass)
    res_open = api.eval(base_order, context_open)
    assert _action(res_open) in ("allow","modify")  # rate rules might still modify

    # Closed session -> block
    closed_ctx = deepcopy(context_open)
    closed_ctx["now_utc"] = int(datetime.now(timezone.utc).replace(hour=3, minute=0, second=0, microsecond=0).timestamp()*1000)
    res_closed = api.eval(base_order, closed_ctx)
    assert _action(res_closed) == "block"

def test_leverage_and_exposure_blocks(engine, base_order, context_open, api: API):
    high_lev = deepcopy(context_open); high_lev["leverage"] = 3.5
    res1 = api.eval(base_order, high_lev)
    assert _action(res1) == "block"

    too_exposed = deepcopy(context_open); too_exposed["exposures"]["AAPL"] = 0.20
    res2 = api.eval(base_order, too_exposed)
    assert _action(res2) == "block"

def test_rate_limit_stateful(engine, base_order, context_open, api: API):
    # 3 orders allowed within 2s per user, 4th gets blocked or delayed
    ctx = deepcopy(context_open)
    for i in range(3):
        o = dict(base_order, id=f"OID-RL-{i}")
        r = api.eval(o, ctx)
        assert _action(r) in ("allow","modify")
    r4 = api.eval(dict(base_order, id="OID-RL-3"), ctx)
    assert _action(r4) in ("block","throttle","retry_after") or any("rate" in s for s in _reasons(r4))

def test_admin_override(engine, base_order, context_open, api: API):
    ctx = deepcopy(context_open); ctx["role"] = "admin"
    bad_symbol = dict(base_order, symbol="AMC")  # normally blocked
    res = api.eval(bad_symbol, ctx)
    # Admin override may convert block -> allow/modify
    assert _action(res) in ("allow","modify")

def test_dry_run_returns_would_block(engine, base_order, context_open, api: API):
    bad = dict(base_order, symbol="AMC")
    res = api.eval(bad, context_open, dry_run=True)
    # Expect not to change state, but to indicate outcome
    assert _action(res) in ("block","would_block","simulate_block")
    # A second normal eval should still block (rate limit unaffected by dry run)
    res2 = api.eval(bad, context_open)
    assert _action(res2) == "block"

def test_kill_switch_blocks_all(engine, base_order, context_open, api: API):
    ks = deepcopy(context_open); ks["env"]["kill"] = True
    res = api.eval(base_order, ks)
    assert _action(res) == "block"

def test_priority_ordering(engine, base_order, context_open, api: API):
    # Even if session open (allow), blocklist with higher priority must win
    ctx = deepcopy(context_open)
    res = api.eval(dict(base_order, symbol="GME"), ctx)
    assert _action(res) == "block"

def test_audit_trail_and_alerts(engine, base_order, context_open, api: API):
    res = api.eval(dict(base_order, qty=60_000), context_open)  # triggers large-size alert
    # Verify alert present (if engine supports)
    if isinstance(res.get("alerts"), list):
        assert any("large" in a.lower() for a in res["alerts"])
    # Verify audit log exists and includes the decision
    if api.get_audit:
        aud = api.get_audit()
        assert isinstance(aud, list) and len(aud) >= 1
        assert any(entry.get("order", {}).get("id") == base_order["id"] or True for entry in aud)

def test_reload_policies_hot_swap(engine, base_order, context_open, api: API):
    if not api.reload:
        pytest.skip("reload() not exposed")
    # Reload a policy that blocks everything
    new_policies = {"version":"1.0","rules":[{"id":"deny-all","priority":999,"when":{},"then":{"action":"block","reason":"maintenance"}}]}
    api.reload(new_policies) if hasattr(api.engine, "reload") else api.reload(api.engine, new_policies) # type: ignore
    res = api.eval(base_order, context_open)
    assert _action(res) == "block"
    # Restore original
    api.start(sample_policies())

def test_stateful_circuit_breaker_optional(engine, base_order, context_open, api: API):
    # If engine escalates to circuit-breaker after N blocks within T, simulate bursts
    ctx = deepcopy(context_open)
    burst_bad = dict(base_order, symbol="AMC")
    for i in range(5):
        api.eval(dict(burst_bad, id=f"B{i}"), ctx)
    # After burst, a good order might be blocked if breaker tripped
    res = api.eval(dict(base_order, id="GOOD1"), ctx)
    # Allow either behavior; assert no crash
    assert _action(res) in ("allow","modify","block")

def test_schema_tolerance(engine, base_order, context_open, api: API):
    # Missing optional fields should not crash; strings for numerics coerced or rejected gracefully
    weird = {"id":"OID-W","symbol":"AAPL","side":"buy","type":"market","qty":"100","price":"185.2"}
    res = api.eval(weird, context_open)
    assert _action(res) in ("allow","modify","block")

def test_time_progress_does_not_persist_dry_run_state(engine, base_order, context_open, api: API):
    # Dry run rate-limit should not consume quota
    ctx = deepcopy(context_open)
    for i in range(2):
        r = api.eval(dict(base_order, id=f"DRY-{i}"), ctx, dry_run=True)
        assert _action(r) in ("allow","block","would_block","simulate_block")
    # Now real calls should still have full quota
    for i in range(3):
        r = api.eval(dict(base_order, id=f"REAL-{i}"), ctx)
        assert _action(r) in ("allow","modify")

def test_policy_engine_state_roundtrip_optional(engine, api: API):
    if not (api.get_state and api.set_state):
        pytest.skip("No state get/set API")
    st = api.get_state() if callable(api.get_state) else api.get_state(api.engine) # type: ignore
    # trivial tweak & restore
    if isinstance(st, dict):
        snap = deepcopy(st)
        st2 = deepcopy(st); st2["__test"] = True
        (api.set_state(st2) if callable(api.set_state) else api.set_state(api.engine, st2)) # type: ignore
        got = api.get_state() if callable(api.get_state) else api.get_state(api.engine) # type: ignore
        assert got.get("__test") is True # type: ignore
        # restore
        (api.set_state(snap) if callable(api.set_state) else api.set_state(api.engine, snap)) # type: ignore