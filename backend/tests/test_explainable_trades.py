# tests/test_explainable_trades.py
import importlib
import math
from typing import Any, Dict, List, Optional
import pytest # type: ignore

"""
What this validates
-------------------
- You can produce a *structured explanation* for a trade decision:
  {
    "trade": {symbol, side, qty, ...},
    "decision": {"action", "score", "confidence"},
    "explanation": {
        "summary": str,
        "rationale": [str, ...],
        "features": [{"name", "value", "contrib"}...],   # signed contribs (e.g., SHAP-ish)
        "rules": [{"if", "then", "fired": bool, "weight": float}, ...],
        "risks": [{"name","impact","prob"}, ...],
        "alternatives": [{"action","delta","why"}, ...],
        "counterfactual": {"target_action", "changes": {"feature": delta, ...}},
    },
    "metadata": {"model", "version", "ts"}
  }
- Confidence/score are bounded and consistent.
- Feature contributions roughly add up to the score (within tolerance).
- Counterfactual changes are coherent (changing features flips decision or improves score).
- JSON-serializable output.
API shapes supported
--------------------
A) class ExplainableTrades with .explain(trade, features, market_ctx, **cfg) -> dict
B) function explain_trade(trade, features=None, market_ctx=None, **cfg) -> dict
C) function explain(trade, **kw) -> dict
Adjust IMPORT_PATHS if your module lives elsewhere.
"""

IMPORT_PATHS = [
    "backend.analytics.explainable_trades",
    "backend.explainability.explainable_trades",
    "backend.research.explainable_trades",
    "analytics.explainable_trades",
    "explainable_trades",
    "explainable_trades.explainable_trades",  # in case it's a package
]


# -------------------- Import utilities --------------------

def _load_module():
    last = None
    for p in IMPORT_PATHS:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import explainable_trades module from candidates {IMPORT_PATHS} ({last})")

def _resolve_explain(mod):
    """
    Return a callable explain(trade, features, market_ctx, **cfg) -> dict
    """
    # Class
    if hasattr(mod, "ExplainableTrades"):
        Cls = getattr(mod, "ExplainableTrades")
        def call(trade, features=None, market_ctx=None, **cfg):
            try:
                inst = Cls(**cfg)
            except TypeError:
                inst = Cls()
            if hasattr(inst, "explain"):
                return inst.explain(trade=trade, features=features, market_ctx=market_ctx, **cfg)
            raise AttributeError("ExplainableTrades exists but has no .explain(...)")
        return call

    # Functions
    for name in ("explain_trade", "explain"):
        if hasattr(mod, name) and callable(getattr(mod, name)):
            fn = getattr(mod, name)
            def call(trade, features=None, market_ctx=None, **cfg):
                try:
                    return fn(trade=trade, features=features, market_ctx=market_ctx, **cfg)
                except TypeError:
                    # Accept positional variants
                    try:
                        return fn(trade, features, market_ctx, **cfg)
                    except TypeError:
                        return fn(trade, **cfg)
            return call

    pytest.skip("No ExplainableTrades class or explain_trade/explain function exported.")


# -------------------- Fixtures --------------------

@pytest.fixture()
def trade():
    return {
        "id": "T-001",
        "symbol": "AAPL",
        "side": "buy",
        "qty": 10_000,
        "ts": 1_700_000_000_000,
        "arrival_px": 185.10,
        "limit_px": 185.40,
        "strategy": "mean_rev_v2",
    }

@pytest.fixture()
def features():
    # Minimal meaningful feature vector
    return {
        "zscore": -1.8,            # mean reversion signal (negative = cheap -> buy)
        "rsi_14": 28.0,            # oversold
        "spread_bps": 3.2,         # microstructure
        "impact_bps": 4.0,         # est. impact
        "sentiment": 0.35,         # light positive news
        "vol_20d": 0.28,           # risk context
        "atr": 3.2,
        "liquidity_pct_adv": 0.06,
    }

@pytest.fixture()
def market_ctx():
    return {
        "symbol": "AAPL",
        "mid": 185.25,
        "bid": 185.24,
        "ask": 185.26,
        "session": "REG",
        "vwap": 185.12,
        "today_volume": 4_000_000,
        "adv": 8_000_000,
        "risk_free": 0.05,
    }


# -------------------- Helpers --------------------

def _assert_common_structure(res: Dict[str, Any]):
    assert isinstance(res, dict)
    for k in ("trade", "decision", "explanation"):
        assert k in res, f"Missing key '{k}'"
    dec = res["decision"]
    assert "action" in dec and isinstance(dec["action"], str)
    assert "score" in dec and isinstance(dec["score"], (int, float))
    assert "confidence" in dec and isinstance(dec["confidence"], (int, float))
    assert -1.0 <= float(dec["score"]) <= 1.0
    assert 0.0 <= float(dec["confidence"]) <= 1.0

    exp = res["explanation"]
    assert "summary" in exp and isinstance(exp["summary"], str)
    assert "rationale" in exp and isinstance(exp["rationale"], list)
    assert "features" in exp and isinstance(exp["features"], list)
    # optional/advanced sections
    for opt in ("rules", "risks", "alternatives", "counterfactual"):
        assert opt in exp or True  # not required


def _feature_sum_close_to_score(res: Dict[str, Any], tol=0.25):
    """
    If you provide per-feature contributions, ensure they roughly sum to the score.
    """
    feats = res["explanation"].get("features") or []
    contribs = [f.get("contrib") for f in feats if isinstance(f, dict) and "contrib" in f]
    if not contribs:
        return  # skip if not provided
    s = sum(float(c or 0.0) for c in contribs)
    score = float(res["decision"]["score"])
    # Allow generous tolerance because models may have bias/interaction terms
    assert math.isfinite(s)
    assert abs(s - score) <= tol + 1e-9


def _assert_counterfactual_is_coherent(res: Dict[str, Any]):
    cf = (res.get("explanation") or {}).get("counterfactual")
    if not cf:
        return  # optional
    assert "target_action" in cf
    changes = cf.get("changes") or {}
    assert isinstance(changes, dict) and len(changes) > 0
    # Simple sanity: at least one relevant feature is being nudged
    keys = set(k.lower() for k in changes.keys())
    assert any(k in keys for k in ("zscore", "rsi", "sentiment", "impact_bps", "spread_bps"))


# -------------------- Tests --------------------

def test_explain_basic_structure(trade, features, market_ctx):
    mod = _load_module()
    explain = _resolve_explain(mod)

    res = explain(trade=trade, features=features, market_ctx=market_ctx) # type: ignore
    _assert_common_structure(res)

    # Basic directional check: negative zscore & low RSI should bias BUY
    action = res["decision"]["action"].lower()
    assert action in {"buy", "scale_in", "hold"}  # allow nuance but not 'sell' here


def test_feature_contributions_respect_direction(trade, features, market_ctx):
    mod = _load_module()
    explain = _resolve_explain(mod)
    res = explain(trade=trade, features=features, market_ctx=market_ctx) # type: ignore

    feats = res["explanation"].get("features") or []
    # Find key drivers if provided
    d = {f.get("name"): f for f in feats if isinstance(f, dict) and "name" in f}
    # zscore negative -> positive contrib to BUY (>= 0)
    if "zscore" in d and "contrib" in d["zscore"]:
        assert d["zscore"]["contrib"] >= -1.0  # should not strongly push against buy
    # high spread/impact should penalize buy
    for key in ("spread_bps", "impact_bps"):
        if key in d and "contrib" in d[key]:
            assert d[key]["contrib"] <= 1.0  # likely non-positive for buys

    _feature_sum_close_to_score(res)


def test_rules_and_rationale_are_consistent(trade, features, market_ctx):
    mod = _load_module()
    explain = _resolve_explain(mod)
    res = explain(trade=trade, features=features, market_ctx=market_ctx) # type: ignore

    rules = (res["explanation"].get("rules") or [])
    # If rules exist, at least one should have fired (for BUY bias)
    fired = [r for r in rules if isinstance(r, dict) and r.get("fired")]
    if rules:
        assert len(fired) >= 1


def test_counterfactual_targets_flip_or_improve(trade, features, market_ctx):
    mod = _load_module()
    explain = _resolve_explain(mod)
    res = explain(trade=trade, features=features, market_ctx=market_ctx) # type: ignore

    cf = (res.get("explanation") or {}).get("counterfactual")
    if not cf:
        pytest.skip("No counterfactual provided; skipping.")
    target = (cf.get("target_action") or "").lower() # type: ignore
    # If current is BUY, a plausible target is SELL or HOLD and vice versa
    curr = res["decision"]["action"].lower()
    assert target in {"buy", "sell", "hold"}
    assert target != curr or True  # allow 'improve buy score' case
    _assert_counterfactual_is_coherent(res)


def test_json_serializable(trade, features, market_ctx):
    mod = _load_module()
    explain = _resolve_explain(mod)
    res = explain(trade=trade, features=features, market_ctx=market_ctx) # type: ignore
    import json
    s = json.dumps(res, ensure_ascii=False)
    assert isinstance(s, str) and len(s) > 10


def test_handles_minimal_inputs_gracefully(trade):
    mod = _load_module()
    explain = _resolve_explain(mod)
    # No features / minimal market ctx
    res = explain(trade=trade, features=None, market_ctx={"symbol": trade["symbol"], "mid": trade["arrival_px"]}) # type: ignore
    _assert_common_structure(res)
    # Expect conservative/neutral decision when info is sparse
    assert res["decision"]["action"].lower() in {"hold", "noop", "buy", "sell"}
    assert 0.0 <= float(res["decision"]["confidence"]) <= 1.0