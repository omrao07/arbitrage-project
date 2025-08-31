# backend/ai/agents/core/base_agent.py
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Callable, Tuple

# ------------------------------------------------------------
# Optional Redis hook (safe fallback if not present)
# ------------------------------------------------------------
try:
    import redis  # type: ignore
    _REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    _REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    _r: Optional["redis.Redis"] = redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT, decode_responses=True)
except Exception:
    _r = None  # noop publisher

# ------------------------------------------------------------
# Result envelope (consistent across agents)
# ------------------------------------------------------------
@dataclass
class AgentResult:
    ok: bool
    agent: str
    started_at: int
    finished_at: int
    took_ms: int
    payload: Any = None
    error: Optional[str] = None
    trace: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ------------------------------------------------------------
# BaseAgent
# ------------------------------------------------------------
class BaseAgent:
    """
    Minimal, batteries-included agent base:
      • plan(req)  -> normalized request (dict or dataclass)
      • act(req)   -> do the work and return domain object (dict/dataclass)
      • explain()  -> one-liner about what this agent does
      • heartbeat()-> liveness/health info

    Extras:
      • run() wrapper adds timing, retries, and a uniform AgentResult
      • emit() helper publishes events to Redis stream (optional) or stdout
      • guard() helper wraps arbitrary callables with error capture
    """

    # Override in subclasses
    name: str = "base_agent"
    version: str = "0.1.0"
    default_retries: int = 0
    default_backoff_ms: int = 250

    # ---------- public API expected by the swarm ----------
    def plan(self, request: Any) -> Any:
        """Normalize/validate request. Subclasses commonly override."""
        return request

    def act(self, request: Any) -> Any:
        """Perform the task. Subclasses **should** override."""
        return {"info": "noop"}

    def explain(self) -> str:
        return "Base agent (noop). Subclasses implement specific behaviors."

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "version": self.version, "ts": int(time.time())}

    # ---------- convenience: one-shot runner with retries ----------
    def run(self, request: Any, *, retries: Optional[int] = None, backoff_ms: Optional[int] = None,
            tag: Optional[str] = None, emit_stream: Optional[str] = None) -> AgentResult:
        """
        Wraps plan/act with timing, retry, and error handling.
        If emit_stream is given, emits a compact event for observability.
        """
        started = _now_ms()
        tries = int(self.default_retries if retries is None else retries)
        backoff = int(self.default_backoff_ms if backoff_ms is None else backoff_ms)
        attempt = 0
        error_txt: Optional[str] = None
        trace_txt: Optional[str] = None

        planned: Any = None
        out: Any = None
        ok = False

        while True:
            attempt += 1
            try:
                planned = self.plan(request)
                out = self.act(planned)
                ok = True
                break
            except Exception as e:
                error_txt = f"{e.__class__.__name__}: {e}"
                trace_txt = traceback.format_exc()
                if attempt > tries:
                    break
                time.sleep(backoff / 1000.0)
                backoff = int(backoff * 1.7)  # exponential backoff

        finished = _now_ms()
        took = finished - started
        res = AgentResult(
            ok=ok,
            agent=self.name,
            started_at=started,
            finished_at=finished,
            took_ms=took,
            payload=out if ok else None,
            error=None if ok else error_txt,
            trace=None if ok else trace_txt,
            meta={
                "attempts": attempt,
                "tag": tag,
                "planned": _safe_preview(planned),
            },
        )

        if emit_stream:
            self.emit(emit_stream, {
                "ok": res.ok, "agent": res.agent, "took_ms": res.took_ms,
                "tag": tag, "attempts": attempt, "ts_ms": finished
            })

        return res

    # ---------- emit helpers ----------
    def emit(self, stream: str, event: Dict[str, Any]) -> None:
        """
        Publish an event to a Redis stream (if configured) or print JSON to stdout.
        """
        try:
            payload = json.dumps(event, default=_json_default)
        except Exception:
            payload = json.dumps({"_bad_event_repr": str(event)})

        if _r is not None:
            try:
                _r.xadd(stream, {"event": payload})
                return
            except Exception:
                pass
        # Fallback: stdout
        print(f"[{self.name}::{stream}] {payload}")

    # ---------- guard helper ----------
    def guard(self, fn: Callable[[], Any]) -> Tuple[bool, Any, Optional[str]]:
        """
        Run fn() and capture exceptions. Returns (ok, value, error_text|None).
        """
        try:
            v = fn()
            return (True, v, None)
        except Exception as e:
            return (False, None, f"{e.__class__.__name__}: {e}")

# ------------------------------------------------------------
# Registry (optional convenience for DI / lookups)
# ------------------------------------------------------------
class AgentRegistry:
    """
    Simple name→agent factory registry so other modules can request agents by string id.
    """
    def __init__(self):
        self._factories: Dict[str, Callable[[], BaseAgent]] = {}

    def register(self, name: str, factory: Callable[[], BaseAgent]) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> BaseAgent:
        if name not in self._factories:
            raise KeyError(f"Agent '{name}' is not registered")
        return self._factories[name]()

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _json_default(o: Any) -> Any:
    try:
        from dataclasses import asdict as _asdict
        return _asdict(o)  # type: ignore
    except Exception:
        try:
            return o.__dict__  # type: ignore
        except Exception:
            return str(o)

def _safe_preview(x: Any, limit: int = 512) -> Any:
    """
    Keep meta small: convert to JSON-ish string and truncate.
    """
    try:
        s = json.dumps(x, default=_json_default) if not isinstance(x, str) else x
    except Exception:
        s = str(x)
    if len(s) > limit:
        s = s[:limit] + "...(+trunc)"
    return s