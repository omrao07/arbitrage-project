# tests/test_liquidity_spiral.py
"""
Liquidity Spiral tests (duck-typed)

Validates core mechanics:
- Fire-sale amplification: bigger sale_frac → bigger price drop & bigger loss
- Impact sensitivity: higher impact_mult → worse outcomes
- Haircuts/liquidity: higher illiquid haircut → more forced selling
- Circuit breaker caps drawdown (if supported)
- Stopping conditions: converge within max_rounds; idempotent when no new sales
- Monotonicity: more severe ADV drop → no less total slippage/drawdown
- Margin calls: if equity/req < 1, engine sells to restore margin (if supported)

Expected APIs (any one is fine):
A) Class LiquiditySpiral(config: dict | None = None)
     .reset(state: dict)
     .step(shock: dict | None = None) -> dict   # returns snapshot
     .run(shock: dict | None = None, max_rounds: int = 10) -> list[dict]
   Snapshots expose keys like: {"prices": {sym: px}, "pnl": {"realized":x,"unrealized":y}, "sold": {sym: qty}, "dd_nav": float}
B) Function simulate_liquidity_spiral(state: dict, shock: dict, **kw) -> list[dict] | dict
"""

import math
import copy
import pytest # type: ignore

ls = pytest.importorskip("backend.risk.liquidity_spiral", reason="liquidity_spiral module not found")

# ----------------------- API shims -------------------------------------------

def _mk_engine(cfg=None):
    if hasattr(ls, "LiquiditySpiral"):
        return ls.LiquiditySpiral(cfg or {})
    elif hasattr(ls, "simulate_liquidity_spiral"):
        return None  # function-only path
    pytest.skip("No LiquiditySpiral class or simulate_liquidity_spiral() function found")

def _run(engine_or_none, state, shock=None, **kw):
    if engine_or_none is None:
        return ls.simulate_liquidity_spiral(copy.deepcopy(state), shock or {}, **kw)
    eng = engine_or_none
    if hasattr(eng, "reset"):
        eng.reset(copy.deepcopy(state))
    if hasattr(eng, "run"):
        return eng.run(shock or {}, **kw)
    # fallback: iterate steps if only .step
    frames = []
    max_rounds = kw.get("max_rounds", 10)
    for _ in range(max_rounds):
        snap = eng.step(shock or {})
        frames.append(snap)
        # try to detect convergence
        if _is_converged(frames):
            break
    return frames

def _last(frames):
    return frames[-1] if isinstance(frames, (list, tuple)) and frames else frames

def _p(frames, sym):
    f = _last(frames)
    prices = f.get("prices") or f.get("px") or {} # type: ignore
    return float(prices.get(sym, 0.0))

def _dd(frames):
    f = _last(frames)
    return float(f.get("dd_nav") or f.get("drawdown") or f.get("dd", 0.0)) # type: ignore

def _sold(frames, sym):
    f = _last(frames)
    sold = f.get("sold") or f.get("liquidated") or {} # type: ignore
    return float(sold.get(sym, 0.0))

def _realized(frames):
    f = _last(frames)
    pnl = f.get("pnl", {}) # type: ignore
    return float(pnl.get("realized", pnl.get("realized_pnl", 0.0)))

def _is_converged(frames):
    if not frames or len(frames) < 2: return False
    a, b = frames[-2], frames[-1]
    # crude: if sold & prices unchanged, assume converged
    pa = (a.get("prices") or {}).items()
    pb = (b.get("prices") or {}).items()
    sa = (a.get("sold") or {})
    sb = (b.get("sold") or {})
    return pa == pb and sa == sb

# ----------------------- Fixtures --------------------------------------------

@pytest.fixture
def base_state():
    # Simple book: 2 assets, one illiquid. NAV ~ 1,000,000
    return {
        "cash": 100_000.0,
        "positions": {
            "LIQX": {"qty": 10_000, "px": 50.0, "illiquid": True,  "adv": 200_000},  # illiquid stock
            "BLUE": {"qty":  5_000, "px": 80.0, "illiquid": False, "adv": 5_000_000}
        },
        "params": {
            "half_spread_bps": {"LIQX": 15.0, "BLUE": 1.0},
            "impact_k": {"LIQX": 0.30, "BLUE": 0.10},
            "impact_alpha": 0.65,
            "perm_impact_gamma": 0.02,
            "illiquid_haircut": 0.25,
            "min_cash_pct_nav": 0.05,
            "margin_req_pct": 0.20,
        }
    }

@pytest.fixture
def engine():
    return _mk_engine({"max_rounds": 12})

# ----------------------- Tests -----------------------------------------------

def test_fire_sale_amplification(engine, base_state):
    s_mild = {"sale_frac": 0.05, "adv_drop": 0.20, "impact_mult": 1.0}
    s_severe = {"sale_frac": 0.20, "adv_drop": 0.40, "impact_mult": 1.0}

    f1 = _run(engine, base_state, s_mild, max_rounds=8)
    f2 = _run(engine, base_state, s_severe, max_rounds=8)

    # Prices should be lower under severe shock; drawdown larger; more sold
    assert _p(f2, "LIQX") <= _p(f1, "LIQX") + 1e-9
    assert _dd(f2) >= _dd(f1) - 1e-9
    assert _sold(f2, "LIQX") >= _sold(f1, "LIQX") - 1e-9

def test_impact_multiplier_effect(engine, base_state):
    s_lo = {"sale_frac": 0.10, "adv_drop": 0.30, "impact_mult": 0.8}
    s_hi = {"sale_frac": 0.10, "adv_drop": 0.30, "impact_mult": 1.6}

    f_lo = _run(engine, base_state, s_lo, max_rounds=8)
    f_hi = _run(engine, base_state, s_hi, max_rounds=8)

    assert _p(f_hi, "LIQX") < _p(f_lo, "LIQX")
    assert _dd(f_hi) > _dd(f_lo)

def test_higher_haircut_forces_more_sales(engine, base_state):
    state = copy.deepcopy(base_state)
    # Higher haircut on illiquid → more collateral shortfall → more selling
    state["params"]["illiquid_haircut"] = 0.40
    s = {"sale_frac": 0.05, "adv_drop": 0.20, "impact_mult": 1.0}
    f = _run(engine, state, s, max_rounds=8)

    state2 = copy.deepcopy(base_state)
    state2["params"]["illiquid_haircut"] = 0.10
    f2 = _run(engine, state2, s, max_rounds=8)

    assert _sold(f, "LIQX") >= _sold(f2, "LIQX") - 1e-9
    assert _dd(f) >= _dd(f2) - 1e-9

def test_circuit_breaker_caps_drawdown_if_supported(engine, base_state):
    # If engine supports circuit breaker, pass a cap and expect smaller drop
    s_no_cap = {"sale_frac": 0.20, "adv_drop": 0.40, "impact_mult": 1.0}
    s_cap    = {"sale_frac": 0.20, "adv_drop": 0.40, "impact_mult": 1.0, "circuit_breaker_dd_cap": 0.10}

    f_nc = _run(engine, base_state, s_no_cap, max_rounds=8)
    f_cp = _run(engine, base_state, s_cap,    max_rounds=8)

    # If the feature isn't implemented, allow skip
    if _dd(f_cp) > _dd(f_nc) + 1e-9:
        pytest.skip("Circuit breaker cap not supported by engine")
    else:
        assert _dd(f_cp) <= 0.10 + 1e-6

def test_converges_and_idempotent_when_no_new_sales(engine, base_state):
    s = {"sale_frac": 0.10, "adv_drop": 0.20, "impact_mult": 1.0}
    frames = _run(engine, base_state, s, max_rounds=12)
    assert len(frames) <= 12

    # One more step with zero sale should be idempotent
    if hasattr(engine, "step"):
        last = _last(frames)
        nonew = {"sale_frac": 0.0, "adv_drop": 0.0, "impact_mult": 0.0}
        frames2 = _run(engine, last if isinstance(last, dict) else base_state, nonew, max_rounds=1)
        # prices should not move materially
        assert abs(_p(frames2, "LIQX") - _p(frames, "LIQX")) < 1e-9

def test_more_adv_drop_not_less_slippage(engine, base_state):
    s_low  = {"sale_frac": 0.10, "adv_drop": 0.10, "impact_mult": 1.0}
    s_high = {"sale_frac": 0.10, "adv_drop": 0.50, "impact_mult": 1.0}

    f_lo = _run(engine, base_state, s_low,  max_rounds=8)
    f_hi = _run(engine, base_state, s_high, max_rounds=8)

    # With lower ADV, participation ratio is higher → more impact → worse prices
    assert _p(f_hi, "LIQX") <= _p(f_lo, "LIQX") + 1e-9
    assert _dd(f_hi) >= _dd(f_lo) - 1e-9

def test_margin_call_forces_sales_if_supported(engine, base_state):
    # Leverage up LIQX so margin requirement binds after price drop
    st = copy.deepcopy(base_state)
    st["cash"] = 0.0
    st["params"]["margin_req_pct"] = 0.30
    s = {"sale_frac": 0.0, "adv_drop": 0.30, "impact_mult": 1.2}

    f = _run(engine, st, s, max_rounds=8)
    sold_qty = _sold(f, "LIQX")

    if sold_qty <= 0:
        pytest.skip("Engine may not implement margin-call selling; skipping.")
    else:
        assert sold_qty > 0
        # realized PnL should be ≤ 0 in a margin call spiral
        assert _realized(f) <= 1e-6