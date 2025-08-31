# backend/ai/agents/concrete/swarm_manager.py
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

# ============================================================
# Base framework shim
# ============================================================
try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name: str = "base_agent"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# ============================================================
# Optional concrete agents (with safe fallbacks)
# ============================================================
def _safe(agent_ctor: Callable[[], Any], fallback_ctor: Callable[[], Any]) -> Any:
    try:
        return agent_ctor()
    except Exception:
        return fallback_ctor()

# Analyst
try:
    from .analyst_agent import AnalystAgent  # type: ignore
except Exception:
    class AnalystAgent(BaseAgent): # type: ignore
        name = "analyst_agent"
        def act(self, req): return {"summary": "analyst(stub): no data", "insights": []}

# Insight
try:
    from .insight_agent import InsightAgent  # type: ignore
except Exception:
    class InsightAgent(BaseAgent): # type: ignore
        name = "insight_agent"
        def act(self, req): return {"summary": "insight(stub): no anomalies", "items": []}

# Query
try:
    from .query_agent import QueryAgent  # type: ignore
except Exception:
    class QueryAgent(BaseAgent): # type: ignore
        name = "query_agent"
        def act(self, req): return {"error": "query(stub)"}

# Deterministic Execution
try:
    from .execution_agent import ExecutionAgent  # type: ignore
except Exception:
    class ExecutionAgent(BaseAgent): # type: ignore
        name = "execution_agent"
        def act(self, req): return {"ok": True, "messages": ["exec(stub)"]}

# RL Execution
try:
    from .rl_execution_agent import RLExecutionAgent  # type: ignore
except Exception:
    class RLExecutionAgent(BaseAgent): # type: ignore
        name = "rl_execution_agent"
        def act(self, req): return {"ok": True, "messages": ["rl-exec(stub)"]}

# Explainer
try:
    from .explainer_agent import ExplainerAgent  # type: ignore
except Exception:
    class ExplainerAgent(BaseAgent): # type: ignore
        name = "explainer_agent"
        def act(self, req): return {"explanation": {"headline": "explainer(stub)"}}

# ============================================================
# Swarm data models
# ============================================================
@dataclass
class Task:
    """A single unit of work routed to an agent."""
    task_id: str
    agent: str                       # 'analyst' | 'insight' | 'query' | 'execution' | 'rl_execution' | 'explainer'
    payload: Dict[str, Any]          # agent-specific request
    depends_on: List[str] = field(default_factory=list)  # other task_ids this one waits for
    timeout_ms: int = 30_000
    notes: Optional[str] = None

@dataclass
class TaskResult:
    task_id: str
    ok: bool
    agent: str
    started_at: int
    finished_at: int
    result: Any
    error: Optional[str] = None

@dataclass
class Playbook:
    """A directed acyclic graph of tasks (DAG)."""
    name: str
    tasks: List[Task]

@dataclass
class SwarmReport:
    playbook: str
    started_at: int
    finished_at: int
    results: Dict[str, TaskResult]
    ok: bool
    summary: str

# ============================================================
# Swarm Manager
# ============================================================
class SwarmManager(BaseAgent): # type: ignore
    """
    Orchestrates a DAG of agent tasks:
      • resolves dependencies
      • executes tasks in waves (no external concurrency deps)
      • passes selected outputs downstream (map step)
      • emits a SwarmReport with per-task traces
    """

    name = "swarm_manager"

    def __init__(self):
        super().__init__()
        # Instantiate agents (upgrade to real if available)
        self.agents: Dict[str, BaseAgent] = {
            "analyst": _safe(AnalystAgent, AnalystAgent),
            "insight": _safe(InsightAgent, InsightAgent),
            "query": _safe(QueryAgent, QueryAgent),
            "execution": _safe(ExecutionAgent, ExecutionAgent),
            "rl_execution": _safe(RLExecutionAgent, RLExecutionAgent),
            "explainer": _safe(ExplainerAgent, ExplainerAgent),
        }

        # simple mappers: how to feed outputs forward (task_id -> transform(result)->dict to merge in child payload)
        self.mappers: Dict[str, Callable[[Any], Dict[str, Any]]] = {
            "analyst": self._map_from_analyst,
            "insight": self._map_from_insight,
            "query": self._map_from_query,
            "execution": self._map_from_execution,
            "rl_execution": self._map_from_execution,
            "explainer": self._map_from_explainer,
        }

    # ----------------- Public API -----------------

    def act(self, playbook_or_dict: Playbook | Dict[str, Any]) -> SwarmReport:
        pb = self._plan(playbook_or_dict)
        start = int(time.time() * 1000)
        results: Dict[str, TaskResult] = {}
        unresolved = {t.task_id: t for t in pb.tasks}

        while unresolved:
            # tasks whose dependencies are satisfied
            ready = [t for t in unresolved.values() if all(d in results and results[d].ok for d in t.depends_on)]
            if not ready:
                # Some dependency failed or a cycle exists; abort remaining
                for t in list(unresolved.values()):
                    results[t.task_id] = TaskResult(
                        task_id=t.task_id, ok=False, agent=t.agent,
                        started_at=int(time.time()*1000), finished_at=int(time.time()*1000),
                        result=None, error="blocked: unmet dependencies or prior failure"
                    )
                    del unresolved[t.task_id]
                break

            # Execute ready tasks sequentially (no external threads to stay dependency-free)
            for task in ready:
                res = self._run_task(task, results)
                results[task.task_id] = res
                del unresolved[task.task_id]

        end = int(time.time() * 1000)
        ok_all = all(r.ok for r in results.values()) if results else False
        summary = self._summarize(pb, results)
        return SwarmReport(playbook=pb.name, started_at=start, finished_at=end, results=results, ok=ok_all, summary=summary)

    def explain(self) -> str:
        return (
            "SwarmManager executes a DAG of tasks across multiple agents "
            "(analyst, insight, query, execution, RL, explainer). It resolves "
            "dependencies, forwards key outputs downstream, and returns a full "
            "per-task trace in a SwarmReport."
        )

    # ----------------- Internals -----------------

    def _plan(self, obj: Playbook | Dict[str, Any]) -> Playbook:
        if isinstance(obj, Playbook):
            return obj
        tasks = []
        for i, t in enumerate(obj.get("tasks", [])):
            tasks.append(Task(
                task_id=t.get("task_id", f"t{i+1}"),
                agent=t.get("agent"),
                payload=dict(t.get("payload", {})),
                depends_on=list(t.get("depends_on", [])),
                timeout_ms=int(t.get("timeout_ms", 30_000)),
                notes=t.get("notes"),
            ))
        return Playbook(name=obj.get("name", "swarm"), tasks=tasks)

    def _run_task(self, task: Task, prior: Dict[str, TaskResult]) -> TaskResult:
        started = int(time.time() * 1000)
        try:
            agent = self.agents.get(task.agent)
            if not agent:
                raise ValueError(f"unknown agent '{task.agent}'")

            # Merge mapped outputs from dependencies into payload
            merged_payload = dict(task.payload)
            for dep_id in task.depends_on:
                dep_res = prior.get(dep_id)
                if dep_res and dep_res.ok:
                    mapper = self.mappers.get(dep_res.agent, lambda _: {})
                    merged_payload.update(mapper(dep_res.result))

            out = agent.act(merged_payload)  # each agent accepts dicts fine (they plan() internally)
            finished = int(time.time() * 1000)
            return TaskResult(task_id=task.task_id, ok=True, agent=task.agent,
                              started_at=started, finished_at=finished, result=out, error=None)
        except Exception as e:
            finished = int(time.time() * 1000)
            return TaskResult(task_id=task.task_id, ok=False, agent=task.agent,
                              started_at=started, finished_at=finished, result=None,
                              error=f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}")

    # ----------------- Mappers (output -> next input) -----------------

    def _map_from_analyst(self, res: Any) -> Dict[str, Any]:
        """
        Pull first symbol + simple risk summary to seed execution/explainer.
        Expects AnalystResponse-like dict:
          { insights: [{symbol, last_price, risk:{var_cash_per_100}}], summary: "..." }
        """
        try:
            ins = (res.get("insights") or res.insights)  # support dict or dataclass
        except Exception:
            ins = []
        sym = None; last_px = None
        if ins:
            item = ins[0]
            sym = item.get("symbol") if isinstance(item, dict) else getattr(item, "symbol", None)
            last_px = item.get("last_price") if isinstance(item, dict) else getattr(item, "last_price", None)
        out: Dict[str, Any] = {}
        if sym:
            out["symbol"] = sym
        if last_px:
            out["px_hint"] = last_px
        return out

    def _map_from_insight(self, res: Any) -> Dict[str, Any]:
        # surface anomaly flag into notes
        summary = res.get("summary") if isinstance(res, dict) else getattr(res, "summary", None)
        return {"notes": f"insight: {summary}"} if summary else {}

    def _map_from_query(self, res: Any) -> Dict[str, Any]:
        # pass through for convenience
        return {"query_result": res}

    def _map_from_execution(self, res: Any) -> Dict[str, Any]:
        # feed fills/plan to explainer
        return {"exec_report": res}

    def _map_from_explainer(self, res: Any) -> Dict[str, Any]:
        # terminal step usually; no downstream fields required
        return {}

    # ----------------- Reporting -----------------

    def _summarize(self, pb: Playbook, results: Dict[str, TaskResult]) -> str:
        ok = sum(1 for r in results.values() if r.ok)
        fail = sum(1 for r in results.values() if not r.ok)
        took = (max((r.finished_at for r in results.values()), default=0) -
                min((r.started_at for r in results.values()), default=0))
        return f"playbook '{pb.name}': {ok} ok / {fail} failed in {took} ms"

# ============================================================
# Convenience: sample playbooks
# ============================================================

def playbook_research_and_execute(symbols: List[str], buy_qty: float, dry_run: bool = True) -> Dict[str, Any]:
    """
    1) Analyst overview → 2) Insight scan → 3) Deterministic execution → 4) Explain trade
    """
    return {
        "name": "research→exec→explain",
        "tasks": [
            {"task_id": "t1", "agent": "analyst",
             "payload": {"symbols": symbols, "interval": "1d", "lookback": 120}},
            {"task_id": "t2", "agent": "insight", "depends_on": ["t1"],
             "payload": {"symbols": symbols, "interval": "1m", "lookback": 120}},
            {"task_id": "t3", "agent": "execution", "depends_on": ["t1", "t2"],
             "payload": {
                 "target": {"symbol": None, "side": "buy", "qty": buy_qty, "order_type": "limit"},
                 "schedule": {"algo": "VWAP", "duration_min": 15, "slice_minutes": 1, "sliding_limit": True},
                 "dry_run": dry_run
             }},
            {"task_id": "t4", "agent": "explainer", "depends_on": ["t3"],
             "payload": {"trade": {"symbol": None, "side": "buy", "qty": buy_qty, "px": 0.0, "strategy": "VWAP"}}}
        ]
    }

def playbook_rl_execute(symbol: str, side: str, qty: float, dry_run: bool = True) -> Dict[str, Any]:
    return {
        "name": "rl-exec",
        "tasks": [
            {"task_id": "t1", "agent": "rl_execution",
             "payload": {"target": {"symbol": symbol, "side": side, "qty": qty, "tag": "RL"},
                         "schedule": {"horizon_min": 2, "step_ms": 15000, "max_participation": 0.12},
                         "dry_run": dry_run}},
            {"task_id": "t2", "agent": "explainer", "depends_on": ["t1"],
             "payload": {"trade": {"symbol": symbol, "side": side, "qty": qty, "px": 0.0, "strategy": "RL"}}}
        ]
    }

# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":  # pragma: no cover
    swarm = SwarmManager()
    pb = playbook_research_and_execute(["AAPL", "MSFT"], buy_qty=25_000, dry_run=True)
    rep = swarm.act(pb)
    print(rep.summary)
    for tid, tr in rep.results.items():
        print(f"- {tid} [{tr.agent}] ok={tr.ok} err={tr.error}")