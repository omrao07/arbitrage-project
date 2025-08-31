# backend/ai/agents/concrete/query_copilot.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# BaseAgent fallback
# ------------------------------------------------------------
try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name = "query_copilot"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# Reuse skills from QueryAgent (with stubs)
try:
    from ..skills.market.quotes import get_candles  # type: ignore
except Exception:
    def get_candles(symbol: str, interval: str = "1d", lookback: int = 30) -> List[Dict[str, Any]]:
        import random
        now = int(time.time() * 1000)
        px = 100.0
        out = []
        for i in range(lookback):
            px *= 1 + random.gauss(0.0003, 0.01)
            out.append({"ts": now - (lookback - i) * 86400000, "o": px, "h": px*1.01, "l": px*0.99, "c": px, "v": 1e6})
        return out

try:
    from ..skills.market.orderbook import get_orderbook  # type: ignore
except Exception:
    def get_orderbook(symbol: str, depth: int = 5) -> Dict[str, Any]:
        px = 100.0
        return {"bids": [{"px": px - i*0.1, "qty": 1000} for i in range(depth)],
                "asks": [{"px": px + i*0.1, "qty": 1000} for i in range(depth)],
                "ts": int(time.time()*1000)}

try:
    from ..skills.news.news_yahoo import fetch_news  # type: ignore
except Exception:
    def fetch_news(symbol: str, limit: int = 3) -> List[Dict[str, Any]]:
        return [{"headline": f"{symbol} stub news", "summary": "No live feed."}]

try:
    from ..skills.risk.var_engine import VaREngine  # type: ignore
except Exception:
    class VaREngine:
        @staticmethod
        def gaussian(returns: List[float], alpha: float = 0.99, horizon_days: int = 1):
            return type("VarEstimate", (), {"var": -0.02, "mean": 0.001, "stdev": 0.01,
                                            "alpha": alpha, "horizon_days": horizon_days})

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class CopilotRequest:
    query: str
    notes: Optional[str] = None

@dataclass
class CopilotResponse:
    query: str
    intents: List[str]
    symbol: Optional[str]
    results: Dict[str, Any]
    generated_at: int

# ------------------------------------------------------------
# Query Copilot Agent
# ------------------------------------------------------------
class QueryCopilotAgent(BaseAgent): # type: ignore
    """
    Natural-language copilot that parses queries like:
      - "show me AAPL candles 1d 60d"
      - "get news on TSLA"
      - "what's the VaR of MSFT last 30d"
      - "orderbook for ETH depth=10"
    and routes to appropriate fetchers.
    """

    name = "query_copilot"

    def plan(self, req: CopilotRequest | Dict[str, Any]) -> CopilotRequest:
        if isinstance(req, CopilotRequest):
            return req
        return CopilotRequest(query=req.get("query", ""), notes=req.get("notes"))

    def act(self, request: CopilotRequest | Dict[str, Any]) -> CopilotResponse:
        req = self.plan(request)
        q = req.query.lower()
        intents: List[str] = []
        symbol: Optional[str] = None
        results: Dict[str, Any] = {}

        # crude symbol extraction (all caps ticker guess)
        match = re.findall(r"\b[A-Z]{2,5}\b", req.query)
        if match:
            symbol = match[0]

        # detect intent
        if "candle" in q or "price" in q or "chart" in q:
            intents.append("price")
            candles = get_candles(symbol or "AAPL", interval="1d", lookback=30)
            results["price"] = candles
        if "orderbook" in q or "depth" in q or "bids" in q:
            intents.append("orderbook")
            ob = get_orderbook(symbol or "AAPL", depth=5)
            results["orderbook"] = ob
        if "news" in q or "headline" in q:
            intents.append("news")
            news = fetch_news(symbol or "AAPL", limit=3)
            results["news"] = news
        if "var" in q or "risk" in q:
            intents.append("risk")
            candles = get_candles(symbol or "AAPL", interval="1d", lookback=30)
            closes = [c["c"] for c in candles]
            rets = [(closes[i]/closes[i-1]-1) for i in range(1, len(closes))] if len(closes) > 1 else []
            est = VaREngine.gaussian(rets)
            results["risk"] = {"VaR": est.var, "mean": est.mean, "stdev": est.stdev} # type: ignore

        if not intents:
            intents.append("unknown")
            results["error"] = "could not parse query"

        return CopilotResponse(
            query=req.query,
            intents=intents,
            symbol=symbol,
            results=results,
            generated_at=int(time.time()*1000)
        )

    def explain(self) -> str:
        return (
            "QueryCopilotAgent accepts natural-language queries and parses them "
            "into structured intents: price, orderbook, news, or risk. "
            "It then routes to the correct data skills and returns results. "
            "Think of it as a Bloomberg-style command line (GP/OV/N/OMON)."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "ts": int(time.time())}


# ------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = QueryCopilotAgent()
    queries = [
        "show me AAPL candles for 60 days",
        "get news on TSLA",
        "orderbook for ETH depth 10",
        "what's the VaR of MSFT last 30d",
    ]
    for q in queries:
        resp = agent.act({"query": q})
        print("Q:", q)
        print("→ intents:", resp.intents, "symbol:", resp.symbol)
        print("→ keys:", list(resp.results.keys()))