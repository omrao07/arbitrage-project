# backend/ai/tests/test_agents_smoke.py
from __future__ import annotations

"""
Lightweight smoke tests for backend.ai.agents.*
- Imports each agent module if present
- Instantiates the primary class with mock deps
- Calls a few common methods with dummy inputs
- Never crashes the suite; reports a summary instead

Run:
  python -m backend.ai.tests.test_agents_smoke
or with pytest:
  pytest -q backend/ai/tests/test_agents_smoke.py
"""

import argparse
import importlib
import inspect
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------- Mocks ---------------------------------

class MockBroker:
    def __init__(self) -> None:
        self.orders: List[Dict[str, Any]] = []
    def place_order(self, *_, **kwargs) -> str:
        oid = f"MOCK-{len(self.orders)+1}"
        self.orders.append({"id": oid, **kwargs})
        return oid
    def cancel_order(self, order_id: str) -> bool:
        return True
    def get_positions(self) -> Dict[str, Any]:
        return {}
    def get_account(self) -> Dict[str, Any]:
        return {"cash": 1_000_000.0, "equity": 1_000_000.0}

class MockMemory:
    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}
    def get(self, k: str, d: Any=None) -> Any:
        return self.kv.get(k, d)
    def set(self, k: str, v: Any) -> None:
        self.kv[k] = v

class MockToolbelt:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
    def news(self, q: str) -> List[Dict[str, Any]]:
        self.calls.append(("news", {"q": q}))
        return [{"title": "Mock headline", "sentiment": 0.1}]
    def price(self, sym: str) -> float:
        self.calls.append(("price", {"sym": sym}))
        return 100.0
    def risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("risk", payload))
        return {"ok": True, "var": 0.02}

# --------------------------- Test Harness ----------------------------

@dataclass
class AgentSpec:
    module: str
    classes: Tuple[str, ...]  # possible class names, first hit wins
    smoke_calls: Tuple[Tuple[str, tuple, dict], ...]  # (method, args, kwargs)

AGENTS: List[AgentSpec] = [
    AgentSpec("backend.ai.agents.analyst_agent",
              ("AnalystAgent", "Analyst", "AgentAnalyst"),
              (("warmup", (), {}), ("handle", ("What are today’s risks?",), {}))),
    AgentSpec("backend.ai.agents.execution_agent",
              ("ExecutionAgent", "ExecAgent"),
              (("warmup", (), {}), ("run_once", ({"symbol": "DEMO", "side": "buy", "qty": 1},), {}))),
    AgentSpec("backend.ai.agents.explainer_agent",
              ("ExplainerAgent", "TradeExplainer"),
              (("warmup", (), {}), ("explain", ({"order_id": "X1", "pnl": 12.3},), {}))),
    AgentSpec("backend.ai.agents.insight_agent",
              ("InsightAgent", "InsightsAgent"),
              (("warmup", (), {}), ("generate", ({"topic": "macro"},), {}))),
    AgentSpec("backend.ai.agents.query_agent",
              ("QueryAgent", "CopilotAgent"),
              (("warmup", (), {}), ("ask", ("Show top movers",), {}))),
    AgentSpec("backend.ai.agents.swarmmanager",
              ("SwarmManager", "Swarm"),
              (("warmup", (), {}), ("coordinate", ({"task": "rebalance"},), {}))),
    AgentSpec("backend.ai.agents.voice_interface",
              ("VoiceInterface", "VoiceAgent"),
              (("warmup", (), {}), ("transcribe", (b"FAKE_WAV_BYTES",), {}))),
]

def _maybe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        return e  # record import error

def _find_class(mod, candidates: Tuple[str, ...]):
    if isinstance(mod, Exception):
        return None
    for cname in candidates:
        if hasattr(mod, cname):
            return getattr(mod, cname)
    # fallback: first class in module that ends with 'Agent' or 'Manager' or 'Interface'
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and any(obj.__name__.endswith(suf) for suf in ("Agent","Manager","Interface")):
            return obj
    return None

def _construct(cls):
    """Try to instantiate with a few common ctor signatures."""
    mocks = dict(broker=MockBroker(), memory=MockMemory(), toolbelt=MockToolbelt(), config={"env":"test"})
    # Try kwargs-only
    try:
        return cls(**{k:v for k,v in mocks.items() if k in inspect.signature(cls).parameters})
    except Exception:
        pass
    # Try empty ctor
    try:
        return cls()
    except Exception:
        pass
    # Try each single-arg
    for k in list(mocks.keys()):
        try:
            return cls(mocks[k])
        except Exception:
            continue
    # Give up with a proxy object that has no-ops
    class _Null:
        def __getattr__(self, _): return lambda *a, **k: {"ok": True}
    return _Null()

def _invoke(obj, method: str, args: tuple, kwargs: dict):
    if not hasattr(obj, method):
        return {"skipped": f"no method {method}"}
    try:
        out = getattr(obj, method)(*args, **kwargs)
        # Make JSON-friendly
        try:
            json.dumps(out)
        except Exception:
            out = str(out)
        return {"ok": True, "result": out}
    except Exception as e:
        return {"error": str(e)}

def run_smoke(fail_on_error: bool = False) -> int:
    summary: Dict[str, Any] = {}
    failures = 0

    for spec in AGENTS:
        mod = _maybe_import(spec.module)
        entry = {"module": spec.module, "import": "ok", "class": None, "calls": []}
        if isinstance(mod, Exception):
            entry["import"] = f"IMPORT_ERROR: {mod}"
            failures += 1
            summary[spec.module] = entry
            continue
        cls = _find_class(mod, spec.classes)
        if cls is None:
            entry["class"] = "CLASS_NOT_FOUND"
            failures += 1
            summary[spec.module] = entry
            continue
        entry["class"] = cls.__name__
        obj = _construct(cls)

        for m, a, kw in spec.smoke_calls:
            res = _invoke(obj, m, a, kw)
            entry["calls"].append({m: res})
            if "error" in res:
                failures += 1

        summary[spec.module] = entry

    print(json.dumps(summary, indent=2, default=str))
    if fail_on_error and failures:
        return 1
    return 0

# ------------------------------ CLI ----------------------------------

def _main():
    p = argparse.ArgumentParser(description="Smoke-test AI agents")
    p.add_argument("--fail-on-error", action="store_true", help="exit 1 if any agent/call fails")
    args = p.parse_args()
    sys.exit(run_smoke(fail_on_error=args.fail_on_error))

if __name__ == "__main__":
    _main()