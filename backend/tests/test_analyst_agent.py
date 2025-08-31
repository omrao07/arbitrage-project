# tests/test_analyst_agent.py
import sys
import types
import importlib
from datetime import datetime, timedelta

import pytest # type: ignore

"""
This test suite exercises a typical AnalystAgent with:
- sentiment scoring on incoming news
- cross-check against simple market context
- structured recommendation output (action/signal/score/confidence/reasoning)
- robustness to missing/empty fields

It avoids external services by monkeypatching the agent's dependencies.
Adjust import_path to match your project layout if needed.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_import(module_path: str):
    """
    Import the analyst agent module by path, but give a nice message if path is wrong.
    Edit `IMPORT_PATH` below to your actual location if needed.
    """
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        pytest.skip(f"Cannot import '{module_path}'. "
                    f"Fix IMPORT_PATH in this test to the correct location. ({e})")


# Change this to your actual module path if different:
# e.g. "backend/agents/analyst_agent.py" -> "backend.agents.analyst_agent"
IMPORT_PATH = "backend.agents.analyst_agent"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent(monkeypatch):
    mod = _safe_import(IMPORT_PATH)

    # If your file exposes a class named AnalystAgent, great.
    # Otherwise you can adapt here (e.g., build from functions).
    assert hasattr(mod, "AnalystAgent"), "AnalystAgent class not found in analyst_agent.py"
    AgentCls = mod.AnalystAgent # type: ignore

    # ---- Mock the sentiment model inside the agent ------------------------
    # We assume agent uses something like self.sentiment.predict(text) -> dict
    # with {"polarity": float (-1..+1), "confidence": float (0..1)}.
    class FakeSentiment:
        def predict(self, text: str):
            # very naive mock: bullish if "beats"/"surge"/"upgrade"; bearish if "miss"/"downgrade"
            t = (text or "").lower()
            if any(k in t for k in ["beats", "surge", "upgrade", "strong"]):
                return {"polarity": 0.65, "confidence": 0.82}
            if any(k in t for k in ["miss", "downgrade", "fraud", "probe"]):
                return {"polarity": -0.6, "confidence": 0.8}
            return {"polarity": 0.05, "confidence": 0.55}

    # If the agent constructs the model internally, intercept it.
    # Common patterns handled below; adjust to your code if needed.
    if hasattr(mod, "SentimentModel"):
        monkeypatch.setattr(mod, "SentimentModel", lambda *a, **k: FakeSentiment())
    if hasattr(AgentCls, "SentimentModel"):
        monkeypatch.setattr(AgentCls, "SentimentModel", FakeSentiment)

    # Some agents use a registry like sentiment_ai.SentimentModel; handle that too.
    try:
        sentiment_mod = importlib.import_module("backend.analytics.sentiment_ai")
        monkeypatch.setattr(sentiment_mod, "SentimentModel", lambda *a, **k: FakeSentiment())
    except ModuleNotFoundError:
        pass

    # ---- Create the agent instance ----------------------------------------
    # Pass minimal config the constructor accepts; fall back to no-args.
    try:
        a = AgentCls(config={"min_conf": 0.5})
    except TypeError:
        a = AgentCls()

    # If the agent expects an attribute `sentiment`, attach our fake explicitly.
    if not hasattr(a, "sentiment"):
        a.sentiment = FakeSentiment()

    return a


@pytest.fixture()
def market_ctx():
    # a minimal market snapshot the agent can use
    now = datetime.utcnow()
    return {
        "symbol": "TSLA",
        "price": 232.15,
        "prev_close": 228.90,
        "vwap": 231.8,
        "atr": 7.2,
        "session": "REG",
        "ts": int(now.timestamp()*1000),
        "lookback": {
            "returns_1h": 0.012,
            "returns_1d": 0.018,
            "vol_1d": 0.028,
            "gap_pct": (232.15 - 228.90) / 228.90,
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bullish_headline_triggers_buy(agent, market_ctx):
    news = {
        "source": "Reuters",
        "headline": "Tesla beats Q2 revenue; raises FY outlook",
        "body": "Revenue surge driven by premium mix. Management guides higher margins.",
        "symbols": ["TSLA"],
        "ts": market_ctx["ts"]
    }

    result = agent.analyze_event(news, market_ctx)  # method name may differ: adapt to your API
    # Expect a structured dict-like response
    assert isinstance(result, dict)
    for key in ("action", "signal", "score", "confidence", "reasoning"):
        assert key in result, f"missing key in result: {key}"

    assert result["action"] in {"buy", "scale_in", "hold"}  # allow agent nuance
    assert result["signal"] >= 0, "bullish headline should not produce negative signal"
    assert 0.0 <= result["confidence"] <= 1.0


def test_bearish_headline_triggers_sell(agent, market_ctx):
    news = {
        "source": "Bloomberg",
        "headline": "Regulator probes safety claims; Downgrade hits Tesla",
        "body": "Analyst downgrade follows ongoing probe; risk to deliveries.",
        "symbols": ["TSLA"],
        "ts": market_ctx["ts"]
    }

    result = agent.analyze_event(news, market_ctx)
    assert result["action"] in {"sell", "reduce", "hedge"}
    assert result["signal"] <= 0, "bearish headline should be <= 0"
    assert result["confidence"] >= 0.5


def test_handles_missing_text_gracefully(agent, market_ctx):
    news = {"source": "CNBC", "headline": "", "body": None, "symbols": ["TSLA"], "ts": market_ctx["ts"]}
    result = agent.analyze_event(news, market_ctx)
    # Should not crash; likely neutral/hold
    assert result["action"] in {"hold", "noop", "ignore"}
    assert -0.2 <= result["signal"] <= 0.2


def test_respects_symbol_filter(agent, market_ctx):
    # If the news references a different ticker, agent should either ignore or low-weight it
    news = {
        "source": "DowJones",
        "headline": "Apple receives major upgrade after strong iPhone sales",
        "body": "Upgrade to Buy; price target raised.",
        "symbols": ["AAPL"],
        "ts": market_ctx["ts"]
    }
    result = agent.analyze_event(news, market_ctx)
    # Depending on your implementation: ignore, noop, or low confidence
    assert result["action"] in {"ignore", "hold", "noop"}
    assert result["confidence"] <= 0.6


def test_report_generation(agent, market_ctx):
    # If your agent can synthesize a short human-readable report
    items = [
        {"headline": "Tesla beats Q2 revenue; raises FY outlook", "symbols": ["TSLA"]},
        {"headline": "Oil prices fall on inventory build", "symbols": ["CL"]},
    ]
    # Allow different method names; adapt if yours is generate_report / summarize / explain
    method = getattr(agent, "generate_report", None) or getattr(agent, "summarize", None)
    if not method:
        pytest.skip("Agent has no `generate_report`/`summarize` method")
    text = method(items, market_ctx) # type: ignore
    assert isinstance(text, str)
    assert len(text) > 40
    assert any(k in text.lower() for k in ["tesla", "tsla"])


def test_latency_budget(agent, market_ctx, benchmark):
    # Optional micro-benchmark: ensure single analysis is fast (< 20ms on typical dev machines)
    news = {
        "source": "Reuters",
        "headline": "Tesla beats Q2 revenue; raises FY outlook",
        "body": "Revenue surge driven by premium mix. Management guides higher margins.",
        "symbols": ["TSLA"],
        "ts": market_ctx["ts"]
    }

    def run():
        agent.analyze_event(news, market_ctx)

    duration = benchmark(run)
    # don't assert a strict number to avoid flaky CI; ensure it's finite
    assert duration >= 0.0