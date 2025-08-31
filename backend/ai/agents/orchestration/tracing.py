# backend/ai/agents/core/tracing.py
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import threading
import contextvars
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Callable, Iterable, Tuple

# ---------------- Optional Redis (safe fallback) ----------------
try:
    import redis  # type: ignore
    _R = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
    )
except Exception:
    _R = None

# ---------------- Env / defaults ----------------
TRACE_ENABLED        = os.getenv("TRACE_ENABLED", "1").strip().lower() in ("1","true","yes")
TRACE_STDOUT         = os.getenv("TRACE_STDOUT", "1").strip().lower() in ("1","true","yes")
TRACE_FILE           = os.getenv("TRACE_FILE", "")  # e.g., ./traces.jsonl
TRACE_REDIS_STREAM   = os.getenv("TRACE_REDIS_STREAM", "traces")
TRACE_SAMPLE_PCT     = float(os.getenv("TRACE_SAMPLE_PCT", "1.0"))  # 1.0 = 100%

# ---------------- Utilities ----------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _rand_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]

def _to_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)), ensure_ascii=False)
    except Exception:
        return json.dumps({"_repr": str(obj)})

# ---------------- Context propagation ----------------
_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
_current_span_id:  contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("span_id",  default=None)

# ---------------- Data models ----------------
@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    start_ms: int
    end_ms: int
    duration_ms: int
    attrs: Dict[str, Any] = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"        # "ok" | "error"
    error: Optional[str] = None

# ---------------- Exporters ----------------
class Exporter:
    def export(self, rec: SpanRecord) -> None:
        raise NotImplementedError

class StdoutExporter(Exporter):
    def export(self, rec: SpanRecord) -> None:
        if not TRACE_STDOUT:
            return
        sys.stdout.write(_to_json(asdict_safe(rec)) + "\n")
        try: sys.stdout.flush()
        except Exception: pass

class FileExporter(Exporter):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.RLock()
    def export(self, rec: SpanRecord) -> None:
        if not self.path:
            return
        line = _to_json(asdict_safe(rec)) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

class RedisExporter(Exporter):
    def __init__(self, stream: str = TRACE_REDIS_STREAM):
        self.stream = stream
    def export(self, rec: SpanRecord) -> None:
        if _R is None:
            return
        try:
            _R.xadd(self.stream, {"span": _to_json(asdict_safe(rec))})
        except Exception:
            pass

# ---------------- Tracer / Span ----------------
class Tracer:
    def __init__(self, *, service: str = "bolt", exporters: Optional[Iterable[Exporter]] = None, sample_pct: float = TRACE_SAMPLE_PCT):
        self.service = service
        self.exporters = list(exporters or [])
        self.sample_pct = max(0.0, min(1.0, float(sample_pct)))
        if not self.exporters:
            # default exporters
            self.exporters.append(StdoutExporter())
            if TRACE_FILE:
                self.exporters.append(FileExporter(TRACE_FILE))
            if _R is not None:
                self.exporters.append(RedisExporter())

    def start_span(self, name: str, *, attrs: Optional[Dict[str, Any]] = None) -> "Span":
        if not TRACE_ENABLED:
            return _NoopSpan()
        # sampling
        if self.sample_pct < 1.0:
            import random
            if random.random() > self.sample_pct:
                return _NoopSpan()
        trace_id = _current_trace_id.get() or _rand_id(32)
        parent_id = _current_span_id.get()
        span_id = _rand_id(16)
        return Span(self, trace_id, span_id, parent_id, name, attrs or {})

    # decorators
    def trace(self, name: Optional[str] = None, *, attrs_fn: Optional[Callable[..., Dict[str, Any]]] = None):
        """
        @tracer.trace("fetch_price", attrs_fn=lambda sym, **k: {"symbol": sym})
        """
        def deco(fn: Callable):
            nm = name or fn.__name__
            def wrapped(*args, **kwargs):
                attrs = {}
                if attrs_fn:
                    try: attrs = attrs_fn(*args, **kwargs) or {}
                    except Exception: attrs = {}
                with self.start_span(nm, attrs=attrs) as sp:
                    try:
                        out = fn(*args, **kwargs)
                        return out
                    except Exception as e:
                        sp.record_error(repr(e))
                        raise
            return wrapped
        return deco

    def trace_async(self, name: Optional[str] = None, *, attrs_fn: Optional[Callable[..., Dict[str, Any]]] = None):
        """
        Async variant.
        """
        def deco(fn: Callable):
            nm = name or fn.__name__
            async def wrapped(*args, **kwargs):
                attrs = {}
                if attrs_fn:
                    try: attrs = attrs_fn(*args, **kwargs) or {}
                    except Exception: attrs = {}
                with self.start_span(nm, attrs=attrs) as sp:
                    try:
                        out = await fn(*args, **kwargs)  # type: ignore
                        return out
                    except Exception as e:
                        sp.record_error(repr(e))
                        raise
            return wrapped
        return deco

    # internal exporter fanout
    def _export(self, rec: SpanRecord) -> None:
        for ex in self.exporters:
            try:
                ex.export(rec)
            except Exception:
                # don’t let tracing kill your process
                pass

class Span:
    def __init__(self, tracer: Tracer, trace_id: str, span_id: str, parent_id: Optional[str], name: str, attrs: Dict[str, Any]):
        self.tracer = tracer
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.attrs = dict(attrs or {})
        self.start_ms = _now_ms()
        self.end_ms = self.start_ms
        self.events: list = []
        self.status = "ok"
        self.error: Optional[str] = None
        # push into context
        self._token_trace = _current_trace_id.set(self.trace_id)
        self._token_span = _current_span_id.set(self.span_id)

    # context manager
    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.record_error(f"{exc_type.__name__}: {exc}")
        self.end()

    # controls
    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[str(key)] = value

    def add_event(self, name: str, **fields) -> None:
        self.events.append({"t": _now_ms(), "name": name, "fields": fields})

    def record_error(self, message: str, **fields) -> None:
        self.status = "error"
        self.error = message
        self.add_event("error", message=message, **fields)

    def end(self) -> None:
        self.end_ms = _now_ms()
        rec = SpanRecord(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_id=self.parent_id,
            name=self.name,
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            duration_ms=self.end_ms - self.start_ms,
            attrs={**self.attrs},
            events=list(self.events),
            status=self.status,
            error=self.error,
        )
        # pop context
        try:
            _current_span_id.reset(self._token_span)
            _current_trace_id.reset(self._token_trace)
        except Exception:
            pass
        # export
        self.tracer._export(rec)

class _NoopSpan(Span):
    def __init__(self):  # type: ignore
        self.tracer = None  # type: ignore
        self.trace_id = ""
        self.span_id = ""
        self.parent_id = ""
        self.name = "noop"
        self.attrs = {}
        self.start_ms = _now_ms()
        self.end_ms = self.start_ms
        self.events = []
        self.status = "ok"
        self.error = None
        self._token_trace = None
        self._token_span = None
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_attr(self, *a, **k): pass
    def add_event(self, *a, **k): pass
    def record_error(self, *a, **k): pass
    def end(self): pass

# ---------------- Helpers / bridge ----------------
def asdict_safe(rec: SpanRecord) -> Dict[str, Any]:
    d = asdict(rec)
    d["service"] = os.getenv("TRACE_SERVICE", "bolt")
    return d

# ---------------- Singleton tracer (optional) ----------------
_tracer_singleton: Optional[Tracer] = None

def get_tracer() -> Tracer:
    global _tracer_singleton
    if _tracer_singleton is None:
        _tracer_singleton = Tracer()
    return _tracer_singleton

# ---------------- Convenience wrappers ----------------
def start_span(name: str, *, attrs: Optional[Dict[str, Any]] = None) -> Span:
    return get_tracer().start_span(name, attrs=attrs or {})

def trace(name: Optional[str] = None, *, attrs_fn: Optional[Callable[..., Dict[str, Any]]] = None):
    return get_tracer().trace(name, attrs_fn=attrs_fn)

def trace_async(name: Optional[str] = None, *, attrs_fn: Optional[Callable[..., Dict[str, Any]]] = None):
    return get_tracer().trace_async(name, attrs_fn=attrs_fn)

# ---------------- Quick smoke test ----------------
if __name__ == "__main__":  # pragma: no cover
    tr = get_tracer()

    @trace("compute_pi", attrs_fn=lambda n=1_000_000: {"samples": n})
    def approx_pi(n=200_000):
        import random
        inside = 0
        for _ in range(n):
            x, y = random.random(), random.random()
            if x*x + y*y <= 1.0:
                inside += 1
        return 4.0 * inside / n

    with start_span("demo", attrs={"user":"om"}) as sp:
        sp.add_event("phase", step="start")
        val = approx_pi(50_000)
        sp.set_attr("pi", round(val, 5))
        sp.add_event("phase", step="end")