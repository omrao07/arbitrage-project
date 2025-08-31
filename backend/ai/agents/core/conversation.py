# backend/ai/agents/core/conversation.py
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Callable, Tuple

# ------------------------------------------------------------
# Optional: BaseAgent (safe shim)
# ------------------------------------------------------------
try:
    from .base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name = "conversation"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self): return "shim"
        def heartbeat(self): return {"ok": True}

# ------------------------------------------------------------
# Optional concrete agents (safe fallbacks)
# ------------------------------------------------------------
def _safe_ctor(ctor: Callable[[], Any], fallback: Callable[[], Any]) -> Any:
    try:
        return ctor()
    except Exception:
        return fallback()

# QueryCopilot → NL to intents
try:
    from ..concrete.query_copilot import QueryCopilotAgent  # type: ignore
except Exception:
    class QueryCopilotAgent(BaseAgent):
        name = "query_copilot"
        def act(self, req):  # req: {"query": "..."}
            q = req.get("query","")
            return {"intents": ["unknown"], "symbol": None, "results": {"echo": q}, "generated_at": int(time.time()*1000)}

# Raw QueryAgent → direct data pulls
try:
    from ..concrete.query_agent import QueryAgent  # type: ignore
except Exception:
    class QueryAgent(BaseAgent):
        name = "query_agent"
        def act(self, req): return {"error": "query(stub)"}

# Execution (RL)
try:
    from ..concrete.rl_execution_agent import RLExecutionAgent  # type: ignore
except Exception:
    class RLExecutionAgent(BaseAgent):
        name = "rl_execution_agent"
        def act(self, req): return {"ok": True, "messages": ["rl-exec(stub)"]}

# Explainer
try:
    from ..concrete.explainer_agent import ExplainerAgent  # type: ignore
except Exception:
    class ExplainerAgent(BaseAgent):
        name = "explainer_agent"
        def act(self, req): return {"explanation": {"headline": "explainer(stub)"}}

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str
    ts_ms: int = field(default_factory=lambda: int(time.time()*1000))
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Memory:
    # ultra-light “profile + preferences + pinned symbols”
    user_name: Optional[str] = None
    timezone: Optional[str] = None
    default_symbol: Optional[str] = None
    watchlist: List[str] = field(default_factory=list)
    risk_prefs: Dict[str, Any] = field(default_factory=dict)  # e.g., {"max_order_qty": 5000}

@dataclass
class TurnResult:
    ok: bool
    reply_text: str
    intents: List[str]
    symbol: Optional[str]
    artifacts: Dict[str, Any] = field(default_factory=dict)

# ------------------------------------------------------------
# Conversation Manager
# ------------------------------------------------------------
class Conversation(BaseAgent):
    """
    Minimal conversation orchestrator:
      • stores rolling chat history + memory
      • parses intents (lightweight rules or via QueryCopilot)
      • routes to data/exec/explainer agents
      • optional persistence to disk JSON
    """

    name = "conversation"
    max_history: int = 200

    def __init__(self, persist_path: Optional[str] = None):
        super().__init__()
        self.persist_path = persist_path
        self.history: List[Message] = []
        self.memory = Memory()
        # Agents
        self.copilot = _safe_ctor(QueryCopilotAgent, QueryCopilotAgent)
        self.query = _safe_ctor(QueryAgent, QueryAgent)
        self.exec_rl = _safe_ctor(RLExecutionAgent, RLExecutionAgent)
        self.explainer = _safe_ctor(ExplainerAgent, ExplainerAgent)
        # Load persisted state if available
        self._load()

    # ---------------- Public API ----------------
    def add_user(self, text: str, **meta) -> None:
        self._append(Message(role="user", content=text, meta=meta))

    def add_assistant(self, text: str, **meta) -> None:
        self._append(Message(role="assistant", content=text, meta=meta))

    def set_memory(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.memory, k):
                setattr(self.memory, k, v)
        self._save()

    def turn(self, text: str) -> TurnResult:
        """
        Main entry: consume user text, produce assistant text + artifacts.
        """
        self.add_user(text)

        # 1) First pass: quick rules for meta-intents
        intents, symbol, params = self._rule_intents(text)

        # 2) If unknown or multi-domain, ask Copilot to parse
        if not intents or intents == ["unknown"]:
            cp = self.copilot.act({"query": text})
            intents = list(getattr(cp, "intents", None) or cp.get("intents", []) or ["unknown"])
            symbol = symbol or getattr(cp, "symbol", None) or cp.get("symbol")
            params.setdefault("copilot_results", getattr(cp, "results", None) or cp.get("results", {}))

        # 3) Route
        reply, artifacts = self._route(intents, symbol, text, params)

        # 4) Save and return
        self.add_assistant(reply, intents=intents, symbol=symbol)
        self._save()
        return TurnResult(ok=True, reply_text=reply, intents=intents, symbol=symbol, artifacts=artifacts)

    # ---------------- Internals ----------------
    def _append(self, msg: Message) -> None:
        self.history.append(msg)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _rule_intents(self, text: str) -> Tuple[List[str], Optional[str], Dict[str, Any]]:
        q = text.strip()
        lower = q.lower()
        intents: List[str] = []
        params: Dict[str, Any] = {}

        # extract ticker guess (ALLCAPS 2-5)
        m = re.findall(r"\b[A-Z]{2,5}\b", q)
        symbol = m[0] if m else (self.memory.default_symbol or None)

        # quick mappings
        if any(k in lower for k in ["price", "chart", "candle", "ohlc"]):
            intents.append("price")
        if any(k in lower for k in ["orderbook", "depth", "bids", "asks", "lob"]):
            intents.append("orderbook")
        if "news" in lower or "headline" in lower:
            intents.append("news")
        if "risk" in lower or "var" in lower or "greeks" in lower:
            intents.append("risk")
        if any(k in lower for k in ["execute", "buy", "sell", "send order", "place order"]):
            intents.append("execute")
            # rough extraction
            side = "buy" if "buy" in lower else ("sell" if "sell" in lower else "buy")
            qty = _extract_number(lower) or 100
            params["exec"] = {"side": side, "qty": qty}
        if any(k in lower for k in ["why", "because", "explain", "rationale"]):
            intents.append("explain")

        if not intents:
            intents = ["unknown"]
        return intents, symbol, params

    def _route(self, intents: List[str], symbol: Optional[str], text: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        artifacts: Dict[str, Any] = {}
        primary = intents[0]

        if primary == "price":
            resp = self.query.act({"symbol": symbol or "AAPL", "query_type": "price", "interval": "1d", "lookback": 30})
            candles = resp.get("data", {}).get("candles") if isinstance(resp, dict) else getattr(resp, "data", {}).get("candles")
            artifacts["price"] = candles or resp
            last = (candles[-1]["c"] if candles else None)
            return (f"{symbol or 'Symbol'} last ≈ {round(float(last),2) if last else 'n/a'}", artifacts)

        if primary == "orderbook":
            resp = self.query.act({"symbol": symbol or "AAPL", "query_type": "orderbook", "limit": 5})
            ob = resp if isinstance(resp, dict) else getattr(resp, "data", resp)
            artifacts["orderbook"] = ob
            try:
                b0 = ob["bids"][0]["px"]; a0 = ob["asks"][0]["px"]
                return (f"Top-of-book {symbol or ''}: bid {b0} / ask {a0}", artifacts)
            except Exception:
                return ("Orderbook unavailable.", artifacts)

        if primary == "news":
            resp = self.query.act({"symbol": symbol or "AAPL", "query_type": "news", "limit": 5})
            headlines = (resp.get("data", {}) if isinstance(resp, dict) else getattr(resp, "data", {})).get("headlines", [])
            artifacts["news"] = headlines
            titles = "; ".join(h.get("headline","") for h in headlines[:3]) if headlines else "no recent headlines"
            return (f"Latest news for {symbol or ''}: {titles}", artifacts)

        if primary == "risk":
            resp = self.query.act({"symbol": symbol or "AAPL", "query_type": "risk", "lookback": 60})
            r = resp.get("data", {}) if isinstance(resp, dict) else getattr(resp, "data", {})
            artifacts["risk"] = r
            return (f"VaR ≈ {round(float(r.get('VaR') or 0)*100,2)}% | μ={round(float(r.get('mean') or 0)*100,2)}% | σ={round(float(r.get('stdev') or 0)*100,2)}%", artifacts)

        if primary == "execute":
            e = params.get("exec", {})
            side = e.get("side","buy"); qty = float(e.get("qty",100))
            req = {
                "target": {"symbol": symbol or "AAPL", "side": side, "qty": qty, "tag": "CONV"},
                "schedule": {"horizon_min": 2, "step_ms": 15_000, "max_participation": 0.12},
                "dry_run": False,  # flip to True for demo
                "context": {"source": "conversation"}
            }
            rep = self.exec_rl.act(req)
            artifacts["execution"] = rep
            ok = (rep.get("ok") if isinstance(rep, dict) else getattr(rep, "ok", False))
            filled = rep.get("filled_qty") if isinstance(rep, dict) else getattr(rep, "filled_qty", 0.0)
            return (f"Execution {'OK' if ok else 'FAILED'} — filled {int(filled)} {symbol or ''}.", artifacts) # type: ignore

        if primary == "explain":
            # Use last execution or copilot results as context if present
            trade = params.get("copilot_results", {})
            ex = self.explainer.act({"trade": {"symbol": symbol, "context": trade}})
            artifacts["explanation"] = ex
            headline = (ex.get("explanation", {}) or {}).get("headline", "Explanation ready.")
            return (headline, artifacts)

        # unknown
        return ("I couldn't parse that. Try: 'AAPL price', 'TSLA orderbook', 'news MSFT', 'execute buy 500 AAPL'.", artifacts)

    # ---------------- Persistence ----------------
    def _load(self) -> None:
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = [Message(**m) for m in data.get("history", [])]
            self.memory = Memory(**data.get("memory", {}))
        except Exception:
            # start fresh if corrupted
            self.history, self.memory = [], Memory()

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({
                    "history": [asdict(m) for m in self.history],
                    "memory": asdict(self.memory)
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _extract_number(s: str) -> Optional[float]:
    # finds the first integer/float
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

# ------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    conv = Conversation(persist_path=os.path.join(os.path.dirname(__file__), ".conv_state.json"))
    print("—", conv.turn("Show AAPL price last month").reply_text)
    print("—", conv.turn("orderbook for TSLA depth").reply_text)
    print("—", conv.turn("news MSFT").reply_text)
    print("—", conv.turn("risk on NVDA").reply_text)
    print("—", conv.turn("execute buy 200 AAPL").reply_text)
    print("—", conv.turn("explain that trade").reply_text)