# tests/test_capacity.py
import math
import importlib
from copy import deepcopy
from typing import Any, Dict
import pytest # type: ignore

"""
What this validates
-------------------
- Capacity respects:
  * Liquidity (ADV / top-of-book / depth)
  * Participation cap (e.g., <= X% of ADV or volume)
  * Venue cap (per-exchange max)
  * Leverage / margin requirements
  * Borrow availability for shorts
  * Risk budgets: volatility cap, VaR, Expected Shortfall
- Monotonicity (more liquidity -> capacity not smaller)
- Edge cases (zero/NaN inputs, negative prices) handled gracefully

Supported public API shapes (pick one in your module):
  1) Class CapacityModel with .compute(symbol, side, strategy, market_ctx, risk_ctx, **cfg) -> dict
  2) Function compute_capacity(symbol, side, strategy, market_ctx, risk_ctx=None, **cfg) -> dict
  3) Function capacity(market_ctx, **cfg) -> dict with per-symbol results (we use symbol key)

Return dict is expected to contain:
  { "qty": float, "notional": float, "constraints": { "<name>": {"limit":..., "used":..., "binding": bool}, ... } }
Keys can differ—assertions are tolerant; see helpers below.
"""

# ---------- Adjust import candidates if your path differs ----------
IMPORT_PATH_CANDIDATES = [
    "backend.risk.capacity",
    "backend.engine.capacity",
    "backend.analytics.capacity",
    "risk.capacity",
    "capacity",
]


# ------------------------- Loader / API resolver -------------------------

def load_module():
    last_err = None
    for path in IMPORT_PATH_CANDIDATES:
        try:
            return importlib.import_module(path)
        except ModuleNotFoundError as e:
            last_err = e
    pytest.skip(f"Could not import capacity module from {IMPORT_PATH_CANDIDATES}: {last_err}")


def resolve_api(mod):
    """
    Returns (mode, callable)
      mode: "class" | "compute" | "bulk"
      callable:
        - class type (instantiate then .compute)
        - function compute_capacity(...)
        - function bulk capacity(market_ctx, ...) returning map
    """
    if hasattr(mod, "CapacityModel"):
        return ("class", getattr(mod, "CapacityModel"))
    if hasattr(mod, "compute_capacity"):
        return ("compute", getattr(mod, "compute_capacity"))
    if hasattr(mod, "capacity"):
        return ("bulk", getattr(mod, "capacity"))
    pytest.skip("No CapacityModel class, compute_capacity(), or capacity() function found in module.")


# ----------------------------- Test fixtures -----------------------------

@pytest.fixture()
def market_ctx() -> Dict[str, Any]:
    """
    Minimal but rich market snapshot for a single symbol across venues.
    Prices USD, volumes in shares/contracts, timestamps ms.
    """
    return {
        "symbol": "AAPL",
        "price": 185.25,
        "adv": 8_000_000,                 # average daily volume (shares)
        "today_volume": 4_200_000,        # running
        "vwap": 185.10,
        "vol_20d": 0.28,                  # 20d annualized vol
        "atr": 3.2,                       # optional
        "lot_size": 1,
        "venues": {
            "NASDAQ": {"bid": 185.24, "ask": 185.26, "depth_bid": 120_000, "depth_ask": 140_000, "fee_bps": 0.3},
            "ARCA":   {"bid": 185.23, "ask": 185.27, "depth_bid": 80_000,  "depth_ask": 60_000,  "fee_bps": 0.25},
            "IEX":    {"bid": 185.22, "ask": 185.28, "depth_bid": 40_000,  "depth_ask": 50_000,  "fee_bps": 0.0},
        },
        "ts": 1_700_000_000_000
    }


@pytest.fixture()
def risk_ctx() -> Dict[str, Any]:
    """
    Risk budgets / account constraints
    """
    return {
        "equity": 5_000_000.0,        # account equity USD
        "leverage_max": 3.0,          # notional <= equity * leverage_max
        "margin_long": 0.5,           # 50% initial margin (equities example)
        "margin_short": 0.7,          # 70% initial margin for shorts
        "borrow_avail": 500_000,      # shares available to borrow for shorting
        "var_limit": 60_000.0,        # 1d VaR budget (USD)
        "es_limit": 90_000.0,         # 1d ES budget (USD)
        "var_horizon_d": 1.0,
        "var_conf": 0.99,
        "multiplier": 1.0,            # contract multiplier (1 for shares)
    }


@pytest.fixture()
def caps_cfg() -> Dict[str, Any]:
    """
    Capacity policy knobs (participation caps, venue caps, slippage).
    """
    return {
        "participation_cap": 0.1,      # <= 10% of ADV
        "intraday_participation_cap": 0.2,  # <= 20% of today's volume
        "per_venue_cap": 100_000,      # shares per venue
        "slippage_bps": 3.0,           # estimate
        "min_clip": 100,               # min order size
        "max_clip": 50_000,            # max per-child
    }


# ------------------------------- Utilities -------------------------------

def get_result_fields(res: Dict[str, Any]):
    """
    Normalize result fields for assertions.
    We look for common keys with fallbacks.
    """
    qty = res.get("qty", res.get("size", res.get("quantity")))
    notional = res.get("notional", res.get("usd", res.get("value")))
    constraints = res.get("constraints", res.get("limits", {}))
    return qty, notional, constraints


def compute_capacity(api_mode, api_obj, symbol, side, strategy, market_ctx, risk_ctx, **cfg):
    """
    Call the module under different API shapes and return normalized (qty, notional, constraints, raw_result).
    """
    if api_mode == "class":
        model = api_obj(**cfg) if _accepts_kwargs(api_obj) else api_obj()
        if hasattr(model, "compute"):
            res = model.compute(symbol=symbol, side=side, strategy=strategy,
                                market_ctx=market_ctx, risk_ctx=risk_ctx, **cfg)
        else:
            pytest.skip("CapacityModel missing .compute(...)")
    elif api_mode == "compute":
        res = api_obj(symbol=symbol, side=side, strategy=strategy,
                      market_ctx=market_ctx, risk_ctx=risk_ctx, **cfg)
    else:  # "bulk"
        out_map = api_obj(market_ctx=market_ctx, risk_ctx=risk_ctx, **cfg)
        res = out_map.get(symbol) if isinstance(out_map, dict) else out_map

    qty, notional, constraints = get_result_fields(res) # type: ignore
    assert isinstance(res, dict)
    assert qty is not None and math.isfinite(float(qty))
    assert notional is not None and math.isfinite(float(notional))
    assert isinstance(constraints, dict)
    return float(qty), float(notional), constraints, res


def _accepts_kwargs(cls):
    try:
        import inspect
        sig = inspect.signature(cls)
        return any(p.kind in (p.VAR_KEYWORD, p.KEYWORD_ONLY) for p in sig.parameters.values())
    except Exception:
        return True


# --------------------------------- Tests ---------------------------------

def test_respects_participation_caps(market_ctx, risk_ctx, caps_cfg):
    mod = load_module()
    mode, api = resolve_api(mod) # type: ignore

    symbol, side, strategy = market_ctx["symbol"], "buy", "mean_rev"

    qty, _, constraints, res = compute_capacity(mode, api, symbol, side, strategy, market_ctx, risk_ctx, **caps_cfg)

    # Expected caps
    max_adv = caps_cfg["participation_cap"] * market_ctx["adv"]
    max_intraday = caps_cfg["intraday_participation_cap"] * market_ctx["today_volume"]

    # At least one of these should be binding if small caps
    cap_vals = [max_adv, max_intraday]
    assert qty <= max(cap_vals) + 1e-6
    # If constraints expose labels, check binding flags
    part_keys = [k for k in constraints.keys() if "participation" in k or "adv" in k or "intraday" in k]
    for k in part_keys:
        c = constraints[k]
        if isinstance(c, dict) and "binding" in c:
            assert isinstance(c["binding"], bool)


def test_respects_per_venue_caps(market_ctx, risk_ctx, caps_cfg):
    mod = load_module()
    mode, api = resolve_api(mod) # type: ignore

    symbol, side, strategy = market_ctx["symbol"], "buy", "mm"

    qty, _, constraints, _ = compute_capacity(mode, api, symbol, side, strategy, market_ctx, risk_ctx, **caps_cfg)

    # total available top-of-book ask across venues (simplistic upper bound)
    venue_asks = [v["depth_ask"] for v in market_ctx["venues"].values()]
    tob_cap = sum(min(caps_cfg["per_venue_cap"], d) for d in venue_asks)
    assert qty <= tob_cap + 1e-6


def test_respects_leverage_and_margin(market_ctx, risk_ctx, caps_cfg):
    mod = load_module()
    mode, api = resolve_api(mod) # type: ignore

    symbol, side, strategy = market_ctx["symbol"], "buy", "carry"

    qty, notional, constraints, _ = compute_capacity(mode, api, symbol, side, strategy, market_ctx, risk_ctx, **caps_cfg)

    # leverage limit: notional <= equity * leverage_max
    lev_lim = risk_ctx["equity"] * risk_ctx["leverage_max"]
    assert notional <= lev_lim + 1e-6

    # margin limit: equity must cover margin requirement
    margin_need = notional * risk_ctx["margin_long"]
    assert margin_need <= risk_ctx["equity"] * risk_ctx["leverage_max"] + 1e-6


def test_short_capacity_respects_borrow_and_margin(market_ctx, risk_ctx, caps_cfg):
    mod = load_module()
    mode, api = resolve_api(mod) # type: ignore

    symbol, side, strategy = market_ctx["symbol"], "sell", "stat_arb"

    qty, notional, constraints, _ = compute_capacity(mode, api, symbol, side, strategy, market_ctx, risk_ctx, **caps_cfg)

    # Borrow constraint (shares)
    assert qty <= risk_ctx["borrow_avail"] + 1e-6

    # Margin for shorts
    assert notional * risk_ctx["margin_short"] <= risk_ctx["equity"] * risk_ctx["leverage_max"] + 1e-6


def test_var_and_es_budgets(market_ctx, risk_ctx, caps_cfg):
    mod = load_module()
    mode, api = resolve_api(mod) # type: ignore

    # Make risk limits tight to force them to bind
    r = deepcopy(risk_ctx)
    r["var_limit"] = 15_000.0
    r["es_limit"] = 20_000.0

    symbol, side, strategy = market_ctx["symbol"], "buy", "trend"

    qty, notional, constraints, _ = compute_capacity(mode, api, symbol, side, strategy, market_ctx, r, **caps_cfg)

    # A loose check: capacity shouldn't exceed a naive VaR-based cap:
    # VaR ≈ price * qty * vol_1d * z (z~2.33 for 99%); solve for qty.
    z = 2.33
    var_cap_qty = r["var_limit"] / (market_ctx["price"] * market_ctx["vol_20d"] * z)
    assert qty <= var_cap_qty * 1.05  # allow 5% modeling differences

    # If constraints expose ES explicitly, ensure it's <= es_limit
    es_keys = [k for k in constraints if "es" in k.lower() or "expected_shortfall" in k.lower()]
    for k in es_keys:
        es_used = constraints[k].get("used")
        if es_used is not None:
            assert float(es_used) <= r["es_limit"] + 1e-6


def test_monotonicity_more_liquidity_never_reduces_capacity(market_ctx, risk_ctx, caps_cfg):
    mod = load_module()
    mode, api = resolve_api(mod) # type: ignore

    symbol, side, strategy = market_ctx["symbol"], "buy", "mm"

    qty0, _, _, _ = compute_capacity(mode, api, symbol, side, strategy, market_ctx, risk_ctx, **caps_cfg)

    richer = deepcopy(market_ctx)
    # Increase ADV and venue depth
    richer["adv"] *= 2.0
    for v in richer["venues"].values():
        v["depth_ask"] *= 2.0

    qty1, _, _, _ = compute_capacity(mode, api, symbol, side, strategy, richer, risk_ctx, **caps_cfg)

    assert qty1 + 1e-6 >= qty0, f"Monotonicity violated: qty1={qty1} < qty0={qty0}"


def test_edge_cases_zero_liquidity_and_bad_inputs(market_ctx, risk_ctx, caps_cfg):
    mod = load_module()
    mode, api = resolve_api(mod) # type: ignore

    broken = deepcopy(market_ctx)
    broken["adv"] = 0
    for v in broken["venues"].values():
        v["depth_ask"] = 0

    symbol, side, strategy = broken["symbol"], "buy", "any"

    qty, notional, constraints, _ = compute_capacity(mode, api, symbol, side, strategy, broken, risk_ctx, **caps_cfg)
    assert qty == pytest.approx(0.0)
    assert notional == pytest.approx(0.0)

    # Negative price should not blow up; expect zero capacity
    broken["price"] = -1.0
    qty2, notional2, _, _ = compute_capacity(mode, api, symbol, side, strategy, broken, risk_ctx, **caps_cfg)
    assert qty2 == pytest.approx(0.0)
    assert notional2 == pytest.approx(0.0)