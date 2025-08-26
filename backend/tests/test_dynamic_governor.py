# test_dynamic_governor.py
# ------------------------------------------------------------
# Behavioral tests for the dynamic risk governor.
# Assumed API in dynamic_governor.py:
#   decide(state: dict, params: dict) -> dict
#   DEFAULT_PARAMS: dict (optional)
#
# The decision dict is expected to contain at least:
#   {
#     "target_gross": float,     # desired gross exposure (e.g., 1.20 = 120%)
#     "target_net": float,       # desired net exposure (e.g., 0.30 = 30%)
#     "halt_trading": bool,      # whether to halt
#     "cooldown_sec": int,       # post-shock cooldown
#     "reasons": list[str],      # optional: why decisions were taken
#   }
# ------------------------------------------------------------
import math
import importlib
import pytest # type: ignore

dg = importlib.import_module("dynamic_governor")

# Helpers to get a params base we can safely mutate in tests
def base_params():
    if hasattr(dg, "DEFAULT_PARAMS"):
        # copy to avoid cross-test mutation
        import copy
        return copy.deepcopy(dg.DEFAULT_PARAMS)
    # Sensible defaults if your module doesn’t export DEFAULT_PARAMS
    return {
        "base_gross": 1.00,         # 100%
        "max_gross": 2.00,
        "min_gross": 0.10,
        "base_net": 0.00,
        "vol_floor": 10.0,          # realized vol (%)
        "vol_ceiling": 40.0,
        "vol_clip_min": 0.25,       # minimum multiplier under very high vol
        "dd_curve": [               # (drawdown_threshold, gross_multiplier)
            [0.05, 0.8],            # >5% dd -> 80% of current gross target
            [0.10, 0.6],            # >10% dd -> 60%
            [0.15, 0.4],            # >15% dd -> 40%
        ],
        "liq_floor": 0.2,           # 0..1 (lower == worse liquidity)
        "liq_min_gross": 0.25,      # cap gross when below liq_floor
        "halt_on_news_level": 3,    # 0..5
        "circuit_breaker_halt": True,
        "cooldown_sec": 900,        # 15 minutes
    }

def decide(**state_overrides):
    """Call the governor with a baseline state + overrides."""
    params = base_params()
    state = {
        "vol_pct": 20.0,                # realized/forecast vol %
        "drawdown": 0.00,               # rolling dd fraction [0..1]
        "liquidity": 0.7,               # 0..1 (higher is better)
        "news_level": 0,                # 0..5 severity
        "circuit_breaker": False,       # limit down / exchange halt flag
        "time_since_shock_sec": 3600,   # 1 hour since last big shock
        "target_hint": None,            # optional external hint
    }
    state.update(state_overrides)
    return dg.decide(state, params), params


# ------------------------ Tests ------------------------

def test_bounds_respected():
    dec, params = decide()
    assert params["min_gross"] <= dec["target_gross"] <= params["max_gross"]

def test_high_vol_reduces_gross_monotonic():
    dec_low, _ = decide(vol_pct=12.0)   # below ceiling
    dec_hi, _  = decide(vol_pct=60.0)   # well above ceiling
    assert dec_hi["target_gross"] <= dec_low["target_gross"]

def test_drawdown_curve_derisks():
    dec_small, _ = decide(drawdown=0.04)
    dec_med, _   = decide(drawdown=0.10)
    dec_large, _ = decide(drawdown=0.16)
    # higher dd ⇒ lower gross
    assert dec_med["target_gross"] <= dec_small["target_gross"]
    assert dec_large["target_gross"] <= dec_med["target_gross"]

def test_liquidity_floor_caps_gross():
    dec_ok, params = decide(liquidity=0.7)      # normal
    dec_bad, _     = decide(liquidity=0.05)     # very poor liq
    assert dec_bad["target_gross"] <= max(params["liq_min_gross"], params["min_gross"])
    assert dec_bad["target_gross"] <= dec_ok["target_gross"]

def test_news_halt_triggers():
    dec, params = decide(news_level=params()["halt_on_news_level"]) # type: ignore
    assert dec["halt_trading"] is True
    assert dec["cooldown_sec"] >= params()["cooldown_sec"] # type: ignore

def test_circuit_breaker_halt():
    dec, _ = decide(circuit_breaker=True)
    assert dec["halt_trading"] is True

def test_cooldown_applies_after_shock():
    # Immediately after shock -> cooldown expected
    dec0, params = decide(time_since_shock_sec=0)
    assert dec0["cooldown_sec"] >= params["cooldown_sec"]
    # After cooldown elapsed -> either zero or much smaller
    dec1, _ = decide(time_since_shock_sec=params["cooldown_sec"] + 1)
    assert dec1["cooldown_sec"] <= 1_000  # effectively no enforced cooldown

def test_target_hint_is_respected_if_safe():
    # If your governor supports a hint but clamps to bounds/risk
    dec, params = decide(target_hint={"gross": 1.8, "net": 0.3})
    assert params["min_gross"] <= dec["target_gross"] <= params["max_gross"]
    # Net should be near the hint but not exceed gross magnitude
    assert abs(dec["target_net"]) <= dec["target_gross"] + 1e-6

def test_reasons_present_and_informative():
    dec, _ = decide(vol_pct=55.0, drawdown=0.12, liquidity=0.05, news_level=4)
    # Optional but recommended: provide human-readable reasons
    if "reasons" in dec:
        assert isinstance(dec["reasons"], list)
        assert any("vol" in r.lower() for r in dec["reasons"])
        assert any("drawdown" in r.lower() or "dd" in r.lower() for r in dec["reasons"])
        assert any("liquid" in r.lower() for r in dec["reasons"])
        assert any("news" in r.lower() or "headline" in r.lower() for r in dec["reasons"])

@pytest.mark.parametrize("vol", [8.0, 20.0, 35.0, 70.0])
def test_vol_response_is_nonincreasing(vol):
    # Check monotonicity across a sweep by comparing to a baseline
    dec_ref, _ = decide(vol_pct=20.0)
    dec_var, _ = decide(vol_pct=vol)
    # If vol increases above baseline, gross should not increase
    if vol > 20.0:
        assert dec_var["target_gross"] <= dec_ref["target_gross"]

def test_min_max_clamping():
    # Extreme bearish case -> clamp at min_gross
    dec_min, params = decide(vol_pct=120.0, drawdown=0.5, liquidity=0.0, news_level=5)
    assert math.isclose(dec_min["target_gross"], params["min_gross"], rel_tol=0, abs_tol=1e-9)
    # Extreme bullish hint -> clamp at max_gross
    dec_max, params = decide(target_hint={"gross": 999, "net": 0.9})
    assert math.isclose(dec_max["target_gross"], params["max_gross"], rel_tol=0, abs_tol=1e-9)