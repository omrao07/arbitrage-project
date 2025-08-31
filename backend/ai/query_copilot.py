# backend/ai/query_copilot.py
from __future__ import annotations

"""
Query Copilot
-------------
Natural language → structured queries over your hedge fund system.

- Input:
  Free text query ("pnl today by strategy") OR JSON query spec
  { "intent": "pnl_by_strategy", "params": {"date": "2025-08-28"} }

- Sources:
  * PnL snapshots stream/hash (pnl.snapshots)
  * Risk state (risk.state)
  * Trades stream (orders.updates)
  * Analyst signals (signals.analyst)
  * Explainable trades (explain.trades)

- Output:
  Structured JSON response { "summary": str, "data": ... }
"""

import os
import re
import json
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- Optional Redis ----------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    AsyncRedis = None  # type: ignore
    USE_REDIS = False

# ---------- Env ----------
REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PNL_STREAM  = os.getenv("PNL_STREAM", "pnl.snapshots")
RISK_HASH   = os.getenv("RISK_HASH", "risk.state")
TRADES_STR  = os.getenv("TRADES_STREAM", "orders.updates")
EXPLAIN_STR = os.getenv("EXPLAIN_STREAM", "explain.trades")
SIGNALS_STR = os.getenv("SIGNALS_STREAM", "signals.analyst")

# ---------- Small LLM hook (optional) ----------
class LLMProvider:
    def __init__(self):
        pass
    def parse(self, q: str) -> Dict[str, Any]:
        """
        Very basic parser. Replace with OpenAI/LLM later.
        Maps text → structured intent.
        """
        t = q.lower()
        if "pnl" in t and "strategy" in t:
            return {"intent": "pnl_by_strategy", "params": {}}
        if "exposure" in t:
            return {"intent": "exposure_by_region", "params": {}}
        if "confidence" in t and "trades" in t:
            return {"intent": "trades_by_confidence", "params": {"thresh": 0.4}}
        if "var" in t or "drawdown" in t:
            return {"intent": "risk_summary", "params": {}}
        return {"intent": "unknown", "raw": q}

# ---------- Core handlers ----------
async def handle_pnl_by_strategy(r: AsyncRedis, params: Dict[str, Any]) -> Dict[str, Any]: # type: ignore
    # expect Redis Hash pnl:strategy:<name> {date:..., value:...}
    out = {}
    keys = await r.keys("pnl:strategy:*")
    for k in keys:
        h = await r.hgetall(k)
        if not h: continue
        strat = k.split(":")[-1]
        out[strat] = float(h.get("value", 0.0))
    return {"summary": "PnL by strategy", "data": out}

async def handle_exposure_by_region(r: AsyncRedis, params: Dict[str, Any]) -> Dict[str, Any]: # type: ignore
    out = {}
    h = await r.hgetall(RISK_HASH)
    for k,v in h.items():
        if k.startswith("exposure:"):
            region = k.split(":")[1]
            out[region] = float(v)
    return {"summary": "Exposure by region", "data": out}

async def handle_trades_by_confidence(r: AsyncRedis, params: Dict[str, Any]) -> Dict[str, Any]: # type: ignore
    thresh = float(params.get("thresh", 0.4))
    # look at explain.trades stream for drivers.features.analyst_confidence
    resp = await r.xrevrange(EXPLAIN_STR, "+", "-", count=100)
    bad = []
    for _, fields in resp:
        try:
            j = json.loads(fields.get("json","{}"))
            ac = j.get("drivers",{}).get("features",{}).get("analyst_confidence")
            if ac is not None and float(ac) < thresh:
                bad.append({"id": j.get("order_id"), "symbol": j.get("symbol"), "conf": ac, "summary": j.get("summary")})
        except Exception:
            continue
    return {"summary": f"Trades executed with analyst_confidence < {thresh}", "data": bad}

async def handle_risk_summary(r: AsyncRedis, params: Dict[str, Any]) -> Dict[str, Any]: # type: ignore
    h = await r.hgetall(RISK_HASH)
    return {
        "summary": "Risk summary",
        "data": {
            "dd": float(h.get("dd", 0.0)),
            "var_1d": float(h.get("var_1d", 0.0)),
            "gross_exposure": float(h.get("gross_exposure", 0.0)),
        }
    }

# ---------- Dispatcher ----------
class QueryCopilot:
    def __init__(self, use_llm: bool = True):
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.llm = LLMProvider() if use_llm else None

    async def connect(self):
        if USE_REDIS:
            try:
                self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
                await self.r.ping() # type: ignore
            except Exception:
                self.r = None

    async def query(self, q: Union[str, Dict[str,Any]]) -> Dict[str, Any]:
        if isinstance(q, str):
            spec = self.llm.parse(q) if self.llm else {"intent": "unknown","raw":q}
        else:
            spec = q
        if self.r is None:
            await self.connect()
        if self.r is None:
            return {"error": "no redis"}
        intent = spec.get("intent")
        params = spec.get("params",{})
        if intent == "pnl_by_strategy":
            return await handle_pnl_by_strategy(self.r, params) # type: ignore
        if intent == "exposure_by_region":
            return await handle_exposure_by_region(self.r, params) # type: ignore
        if intent == "trades_by_confidence":
            return await handle_trades_by_confidence(self.r, params) # type: ignore
        if intent == "risk_summary":
            return await handle_risk_summary(self.r, params) # type: ignore
        return {"summary": "unknown", "data": {}, "spec": spec}

# ---------- CLI entry ----------
async def _amain():
    qc = QueryCopilot()
    await qc.connect()
    print("[query_copilot] ready")
    # demo queries
    for q in ["pnl today by strategy", "show exposure by region", "trades with low confidence", "risk summary"]:
        res = await qc.query(q)
        print(q, "->", json.dumps(res, indent=2))

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass