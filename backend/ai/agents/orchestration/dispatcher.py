# backend/ai/agents/core/dispatcher.py
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from queue import PriorityQueue, Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# -------- Optional local helpers (safe fallbacks) --------
try:
    from .base_agent import BaseAgent, AgentResult # type: ignore
except Exception:
    class BaseAgent:  # tiny shim
        name = "base"
        def plan(self, req): return req
        def act(self, req): return {"ok": True}
        def heartbeat(self): return {"ok": True}
    @dataclass
    class AgentResult:
        ok: bool; agent: str; started_at: int; finished_at: int; took_ms: int
        payload: Any = None; error: Optional[str] = None; trace: Optional[str] = None; meta: Dict[str, Any] = field(default_factory=dict)

try:
    from .toolbelt import now_ms, with_timeout, RateLimiter, TTLCache, to_json # type: ignore
except Exception:
    def now_ms(): return int(time.time()*1000)
    def with_timeout(fn, timeout_ms, *a, **k):
        # minimal soft timeout
        res = {"done": False, "val": None, "err": None}
        def r(): 
            try: res["val"] = fn(*a, **k)
            except Exception as e: res["err"] = e
            finally: res["done"] = True
        th = threading.Thread(target=r, daemon=True); th.start(); th.join(timeout_ms/1000.0)
        if not res["done"]: raise TimeoutError("operation timed out")
        if res["err"]: raise res["err"]
        return res["val"]
    class RateLimiter:
        def __init__(self, rate_per_sec: float, burst: float):
            self.rate=rate_per_sec; self.burst=burst; self.tokens=burst; self.updated=time.time(); self._lock=threading.RLock()
        def allow(self, cost: float=1.0)->bool:
            with self._lock:
                now=time.time(); dt=max(0.0, now-self.updated); self.updated=now
                self.tokens=min(self.burst, self.tokens+dt*self.rate)
                if self.tokens>=cost: self.tokens-=cost; return True
                return False
    class TTLCache(dict):
        def __init__(self, maxsize=1024, ttl_ms=60000): super().__init__(); self._ttl=ttl_ms; self._t={}
        def __setitem__(self,k,v): super().__setitem__(k,v); self._t[k]=now_ms()
        def get(self,k,d=None):
            if k in self and now_ms()-self._t.get(k,0)<=self._ttl: return super().get(k)
            return d
    def to_json(x, indent=None): 
        try: return json.dumps(x, default=lambda o: getattr(o,"__dict__", str(o)), ensure_ascii=False, indent=indent)
        except Exception: return str(x)

# Optional Redis for event emission
try:
    import redis  # type: ignore
    _R = redis.Redis(host=os.getenv("REDIS_HOST","localhost"), port=int(os.getenv("REDIS_PORT","6379")), decode_responses=True)
except Exception:
    _R = None

# ================= Models =================
@dataclass(order=True)
class _QItem:
    priority: int
    enq_ms: int
    task_id: str = field(compare=False)
    task: Dict[str, Any] = field(compare=False)

@dataclass
class DispatchResult:
    ok: bool
    task_id: str
    agent: str
    started_at: int
    finished_at: int
    took_ms: int
    result: Any = None
    error: Optional[str] = None
    trace: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ================= Dispatcher =================
class Dispatcher:
    """
    Multi-agent task dispatcher with:
      • registration (agent factory or singleton)
      • priority queue & worker threads
      • retries, timeouts, rate limits, cancellation
      • pre/post middleware hooks & safety check
      • optional Redis event emission
    """

    def __init__(self, *, workers: int = 4, default_timeout_ms: int = 10_000, max_queue: int = 5000):
        self.workers = max(1, int(workers))
        self.default_timeout_ms = int(default_timeout_ms)
        self._q: PriorityQueue[_QItem] = PriorityQueue(maxsize=max_queue)
        self._res_q: Queue[DispatchResult] = Queue()
        self._stop = threading.Event()
        self._agents: Dict[str, Union[BaseAgent, Callable[[], BaseAgent]]] = {}
        self._intents: Dict[str, str] = {}   # intent -> agent_name
        self._rate: Dict[str, RateLimiter] = {}
        self._inflight: Dict[str, Dict[str, Any]] = {}  # task_id -> state
        self._lock = threading.RLock()
        self._middlewares_pre: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []
        self._middlewares_post: List[Callable[[DispatchResult], DispatchResult]] = []
        self._canceled: Dict[str, bool] = {}
        self._cache = TTLCache(maxsize=2048, ttl_ms=30_000)

        # spawn workers
        self._threads = [threading.Thread(target=self._worker, name=f"dispatcher-{i}", daemon=True) for i in range(self.workers)]
        for t in self._threads: t.start()

    # ---------- Registration ----------
    def register(self, name: str, agent: Union[BaseAgent, Callable[[], BaseAgent]]) -> None:
        with self._lock:
            self._agents[name] = agent

    def bind_intent(self, intent: str, agent_name: str) -> None:
        with self._lock:
            self._intents[intent] = agent_name

    def add_rate_limit(self, key: str, *, rate_per_sec: float, burst: float) -> None:
        self._rate[key] = RateLimiter(rate_per_sec=rate_per_sec, burst=burst)

    def use_pre(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._middlewares_pre.append(fn)

    def use_post(self, fn: Callable[[DispatchResult], DispatchResult]) -> None:
        self._middlewares_post.append(fn)

    # ---------- Submit / Cancel ----------
    def submit(self, task: Dict[str, Any], *, priority: int = 10) -> str:
        """
        Task schema (suggested):
          {
            "task_id": "optional",  # auto if missing
            "agent": "analyst_agent" | None,
            "intent": "risk" | "price" | ...,
            "payload": {...},
            "timeout_ms": 8000,
            "retries": 1,
            "rate_key": "exec"  # optional limiter bucket
          }
        """
        t = dict(task)  # shallow copy
        tid = t.get("task_id") or f"T{now_ms()}{len(self._inflight)%997}"
        t["task_id"] = tid

        # middlewares (pre)
        for mw in self._middlewares_pre:
            try:
                t = mw(t)
            except Exception as e:
                # If pre-mw fails, enqueue a failed result immediately
                res = DispatchResult(ok=False, task_id=tid, agent=t.get("agent") or "?", started_at=now_ms(),
                                     finished_at=now_ms(), took_ms=0, error=f"pre-middleware: {e}")
                self._res_q.put(res)
                return tid

        # rate limit
        rk = t.get("rate_key")
        if rk and rk in self._rate and not self._rate[rk].allow(1.0):
            # soft-drop: queue a failure immediately
            res = DispatchResult(ok=False, task_id=tid, agent=t.get("agent") or "?", started_at=now_ms(),
                                 finished_at=now_ms(), took_ms=0, error="rate_limited")
            self._res_q.put(res)
            return tid

        self._q.put(_QItem(priority=int(priority), enq_ms=now_ms(), task_id=tid, task=t))
        with self._lock:
            self._inflight[tid] = {"enq_ms": now_ms(), "task": t, "status": "queued"}
        return tid

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            self._canceled[task_id] = True
            st = self._inflight.get(task_id)
            if st: st["status"] = "canceled"
        return True

    # ---------- Results ----------
    def get_result(self, block: bool = True, timeout: Optional[float] = None) -> Optional[DispatchResult]:
        try:
            return self._res_q.get(block=block, timeout=timeout)
        except Empty:
            return None

    # ---------- Introspection ----------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queued": self._q.qsize(),
                "inflight": len(self._inflight),
                "workers": self.workers
            }

    # ---------- Shutdown ----------
    def shutdown(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            for _ in self._threads:
                self._q.put(_QItem(priority=-999999, enq_ms=now_ms(), task_id="_poison_", task={"_poison": True}))
            for t in self._threads: t.join(timeout=2.0)

    # ================= Workers =================
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except Empty:
                continue
            if item.task.get("_poison"):  # shutdown signal
                break

            t = item.task
            tid = item.task_id
            try:
                # canceled before start?
                if self._canceled.get(tid):
                    self._finish_cancel(t)
                    continue

                agent_name = self._resolve_agent(t)
                agent = self._get_agent(agent_name)

                timeout_ms = int(t.get("timeout_ms", self.default_timeout_ms))
                retries = int(t.get("retries", 0))
                payload = t.get("payload", {})

                with self._mark_inflight(tid, agent_name, "running"):
                    res = self._run_with_retries(agent, payload, timeout_ms, retries)

                self._emit_result(res)
            except Exception as e:
                res = DispatchResult(
                    ok=False, task_id=tid, agent=t.get("agent") or t.get("intent") or "?",
                    started_at=now_ms(), finished_at=now_ms(), took_ms=0,
                    error=f"dispatch_error: {e}"
                )
                self._emit_result(res)

    # ---------- Runner helpers ----------
    def _run_with_retries(self, agent: BaseAgent, payload: Dict[str, Any], timeout_ms: int, retries: int) -> DispatchResult:
        tid = payload.get("_task_id") or f"T{now_ms()}"
        start = now_ms()
        err_text = None; trace = None; result: Optional[AgentResult] = None
        attempt = -1

        def call():
            # Allow agent.run() if present; otherwise plan+act
            if hasattr(agent, "run"):
                return agent.run(payload) # type: ignore
            # Minimal emulation
            t0 = now_ms()
            try:
                planned = agent.plan(payload)
                out = agent.act(planned)
                return AgentResult(True, getattr(agent,"name","agent"), t0, now_ms(), now_ms()-t0, out)
            except Exception as e:
                import traceback as tb
                return AgentResult(False, getattr(agent,"name","agent"), t0, now_ms(), now_ms()-t0, None, str(e), tb.format_exc(), {})

        for attempt in range(retries + 1):
            try:
                ar = with_timeout(call, timeout_ms)
                result = ar if isinstance(ar, AgentResult) else None
                if result and result.ok:
                    break
                # if AgentResult but failed, bubble error message for retry
                err_text = (result.error if result else "unknown")
                trace = (result.trace if result else None)
            except Exception as e:
                err_text = f"{e.__class__.__name__}: {e}"
                trace = None
            time.sleep(min(2.0, 0.2 * (attempt + 1)))

        finish = now_ms()
        if result and result.ok:
            dr = DispatchResult(ok=True, task_id=tid, agent=result.agent, started_at=start,
                                finished_at=finish, took_ms=finish-start, result=result.payload,
                                meta={"attempts": attempt+1})
        else:
            dr = DispatchResult(ok=False, task_id=tid, agent=getattr(agent,"name","agent"), started_at=start,
                                finished_at=finish, took_ms=finish-start, error=err_text, trace=trace,
                                meta={"attempts": attempt+1})
        # post middlewares
        for mw in self._middlewares_post:
            try:
                dr = mw(dr)
            except Exception:
                pass
        return dr

    def _resolve_agent(self, task: Dict[str, Any]) -> str:
        # priority: explicit agent name > intent mapping > default "base"
        a = task.get("agent")
        if a: return a
        intent = task.get("intent")
        if intent and intent in self._intents:
            return self._intents[intent]
        return "base"

    def _get_agent(self, name: str) -> BaseAgent:
        with self._lock:
            entry = self._agents.get(name)
        if entry is None:
            # lazy default
            return BaseAgent()
        if isinstance(entry, BaseAgent):
            return entry
        # factory
        return entry()

    def _mark_inflight(self, task_id: str, agent: str, status: str):
        class _Ctx:
            def __init__(self, outer, tid, agent, status):
                self.o=outer; self.tid=tid; self.agent=agent; self.status=status
            def __enter__(self):
                with self.o._lock:
                    self.o._inflight[self.tid] = {"agent": self.agent, "status": self.status, "start": now_ms()}
                return self
            def __exit__(self, exc_type, exc, tb):
                with self.o._lock:
                    self.o._inflight.pop(self.tid, None)
        return _Ctx(self, task_id, agent, status)

    def _finish_cancel(self, task: Dict[str, Any]) -> None:
        tid = task.get("task_id","?")
        res = DispatchResult(ok=False, task_id=tid, agent=task.get("agent") or task.get("intent") or "?",
                             started_at=now_ms(), finished_at=now_ms(), took_ms=0, error="canceled")
        self._emit_result(res)

    def _emit_result(self, res: DispatchResult) -> None:
        # to results queue
        self._res_q.put(res)
        # optional Redis event
        if _R is not None:
            try:
                _R.xadd("agents.dispatch", {"result": json.dumps(res.to_dict())})
            except Exception:
                pass

# ================= Example wiring =================
"""
from backend.ai.agents.core.dispatcher import Dispatcher
from backend.ai.agents.concrete.analyst_agent import AnalystAgent
from backend.ai.agents.concrete.rl_execution_agent import RLExecutionAgent

d = Dispatcher(workers=6, default_timeout_ms=8000)
d.register("analyst", AnalystAgent)      # factory
d.register("exec_rl", RLExecutionAgent())# singleton
d.bind_intent("price", "analyst")
d.bind_intent("execute", "exec_rl")

# Optional middlewares
def safety_mw(task):
    # e.g., drop live orders if dry_run not set
    if task.get("intent") == "execute":
        p = task.setdefault("payload", {})
        p.setdefault("dry_run", True)
    return task
d.use_pre(safety_mw)

def post_log(res):
    print("[DISPATCH]", res.agent, res.ok, res.took_ms, "ms")
    return res
d.use_post(post_log)

tid = d.submit({"intent":"price", "payload":{"symbol":"AAPL","query_type":"price"}}, priority=5)
print("task id", tid)
res = d.get_result(True, 5.0)
print("RES:", res.to_dict())
d.shutdown()
"""