# backend/ai/agents/concrete/query_agent.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# Framework fallback
# ------------------------------------------------------------
try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name: str = "query_agent"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# Skills (optional, with safe stubs)
try:
    from ..skills.market.quotes import get_candles  # type: ignore
except Exception:
    def get_candles(symbol: str, interval: str = "1d", lookback: int = 30) -> List[Dict[str, Any]]:
        import math, random
        px = 100.0
        out = []
        now = int(time.time() * 1000)
        for i in range(lookback):
            px *= 1 + random.gauss(0.0002, 0.01)
            ts = now - (lookback - i) * 86_400_000
            out.append({"ts": ts, "o": px, "h": px * 1.01, "l": px * 0.99, "c": px, "v": random.randint(1e5, 5e5)}) # type: ignore
        return out

try:
    from ..skills.market.orderbook import get_orderbook  # type: ignore
except Exception:
    def get_orderbook(symbol: str, depth: int = 5) -> Dict[str, Any]:
        px = 100.0
        bids = [{"px": px - i * 0.1, "qty": 1000 - 50*i} for i in range(depth)]
        asks = [{"px": px + i * 0.1, "qty": 1000 - 50*i} for i in range(depth)]
        return {"bids": bids, "asks": asks, "ts": int(time.time() * 1000)}

try:
    from ..skills.news.news_yahoo import fetch_news  # type: ignore
except Exception:
    def fetch_news(symbol: str, limit: int = 3) -> List[Dict[str, Any]]:
        return [{"headline": f"{symbol}: fallback headline", "summary": "No live news connected."}]

try:
    from ..skills.risk.var_engine import VaREngine  # type: ignore
except Exception:
    class VaREngine:
        @staticmethod
        def gaussian(returns: List[float], alpha: float = 0.99, horizon_days: int = 1):
            return type("VarEstimate", (), {
                "var": -0.02, "mean": 0.001, "stdev": 0.01,
                "alpha": alpha, "horizon_days": horizon_days
            })

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class QueryRequest:
    symbol: str
    query_type: str             # "price" | "orderbook" | "news" | "risk"
    interval: str = "1d"
    lookback: int = 30
    limit: int = 3
    notes: Optional[str] = None

@dataclass
class QueryResponse:
    symbol: str
    query_type: str
    data: Any
    generated_at: int

# ------------------------------------------------------------
# Query Agent
# ------------------------------------------------------------
class QueryAgent(BaseAgent): # type: ignore
    """
    Multi-purpose query agent:
      • fetches candles (price history)
      • fetches orderbook snapshots
      • fetches news headlines
      • fetches risk metrics (VaR)
    """

    name = "query_agent"

    def plan(self, req: QueryRequest | Dict[str, Any]) -> QueryRequest:
        if isinstance(req, QueryRequest):
            return req
        return QueryRequest(
            symbol=req.get("symbol", "UNK"),
            query_type=req.get("query_type", "price"),
            interval=req.get("interval", "1d"),
            lookback=int(req.get("lookback", 30)),
            limit=int(req.get("limit", 3)),
            notes=req.get("notes"),
        )

    def act(self, request: QueryRequest | Dict[str, Any]) -> QueryResponse:
        req = self.plan(request)
        sym = req.symbol
        t = req.query_type.lower()
        data: Any = None

        if t == "price":
            candles = get_candles(sym, interval=req.interval, lookback=req.lookback)
            data = {"candles": candles}
        elif t == "orderbook":
            ob = get_orderbook(sym, depth=req.limit)
            data = ob
        elif t == "news":
            headlines = fetch_news(sym, limit=req.limit)
            data = {"headlines": headlines}
        elif t == "risk":
            candles = get_candles(sym, interval=req.interval, lookback=req.lookback)
            closes = [float(c["c"]) for c in candles]
            rets = [ (closes[i]/closes[i-1]-1) for i in range(1, len(closes)) ] if len(closes) > 1 else []
            est = VaREngine.gaussian(rets, alpha=0.99, horizon_days=1)
            data = {"VaR": est.var, "mean": est.mean, "stdev": est.stdev} # type: ignore
        else:
            data = {"error": f"unknown query_type={t}"}

        return QueryResponse(
            symbol=sym,
            query_type=req.query_type,
            data=data,
            generated_at=int(time.time() * 1000)
        )

    def explain(self) -> str:
        return (
            "QueryAgent routes symbol+query_type requests to skills: "
            "price (candles), orderbook, news, or risk metrics. "
            "It returns structured results for use in dashboards or APIs."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "ts": int(time.time())}


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = QueryAgent()
    for t in ["price", "orderbook", "news", "risk"]:
        resp = agent.act({"symbol": "AAPL", "query_type": t})
        print(">>>", t, resp.data)