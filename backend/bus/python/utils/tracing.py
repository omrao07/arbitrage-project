# bus/python/utils/tracing.py
from __future__ import annotations

import time
import functools
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional, Callable, Awaitable, Tuple

# ---------------- Optional OpenTelemetry ----------------
try:
    from opentelemetry import trace, context, propagation # type: ignore
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.propagate import inject, extract
    _OTEL = True
except Exception:  # OpenTelemetry not available -> no-op fallbacks
    trace = None
    context = None
    propagation = None
    SpanKind = None
    Status = None
    StatusCode = None
    inject = None
    extract = None
    _OTEL = False


# ============ Public helpers (safe to call even without OTel) ============

def get_tracer(service_name: str = "hyper-os", version: Optional[str] = None):
    """
    Return an OpenTelemetry tracer if available, else a no-op stub.
    """
    if _OTEL:
        return trace.get_tracer(service_name, version or "0.0.0") # type: ignore
    return _NoopTracer()


def inject_headers(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Inject the current trace context into headers (W3C traceparent).
    Always returns a dict (copies the input).
    """
    h = dict(headers or {})
    if _OTEL:
        inject(lambda k, v: h.__setitem__(k, v)) # type: ignore
    return h


def extract_context(headers: Optional[Dict[str, str]] = None):
    """
    Extract a context from headers (W3C traceparent). If OTel missing, returns None.
    """
    if _OTEL:
        return extract(_DictGetter(headers or {})) # type: ignore
    return None


def get_current_trace_ids() -> Tuple[Optional[str], Optional[str]]:
    """
    Return (trace_id_hex, span_id_hex) if a span is current, else (None, None).
    """
    if not _OTEL:
        return None, None
    span = trace.get_current_span() # type: ignore
    ctx = span.get_span_context() if span else None
    if not ctx or not ctx.is_valid:
        return None, None
    return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"


def format_traceparent() -> Optional[str]:
    """
    Return a W3C traceparent string for the current span, or None.
    """
    if not _OTEL:
        return None
    span = trace.get_current_span() # type: ignore
    ctx = span.get_span_context() if span else None
    if not ctx or not ctx.is_valid:
        return None
    # version 00, sampled flag according to trace_flags
    sampled = "01" if int(ctx.trace_flags) & 0x01 else "00"
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{sampled}"


# ================== Span helpers / contexts ==================

@contextmanager
def start_span(
    name: str,
    *,
    kind: str = "INTERNAL",
    attributes: Optional[Dict[str, Any]] = None,
    parent_headers: Optional[Dict[str, str]] = None,
):
    """
    Generic span context manager. Safe no-op if OTel is absent.
    """
    if not _OTEL:
        yield _NoopSpan(name, attributes or {})
        return

    tracer = get_tracer()
    parent_ctx = extract_context(parent_headers) if parent_headers else None
    span_kind = {
        "INTERNAL": SpanKind.INTERNAL, # type: ignore
        "SERVER": SpanKind.SERVER, # type: ignore
        "CLIENT": SpanKind.CLIENT, # type: ignore
        "PRODUCER": SpanKind.PRODUCER, # type: ignore
        "CONSUMER": SpanKind.CONSUMER, # type: ignore
    }.get(kind.upper(), SpanKind.INTERNAL) # type: ignore

    with tracer.start_as_current_span(name, context=parent_ctx, kind=span_kind) as span:
        if attributes:
            for k, v in attributes.items():
                try:
                    span.set_attribute(k, v)
                except Exception:
                    pass
        yield span


@contextmanager
def start_consume_span(
    topic: str,
    key: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None,
    messaging_system: str = "bus",
):
    """
    Span specialized for message CONSUME. Extracts parent from headers if present.
    """
    attrs = {
        "messaging.system": messaging_system,
        "messaging.operation": "receive",
        "messaging.destination": topic,
    }
    if key is not None:
        attrs["messaging.message.key"] = str(key)
    with start_span(f"{messaging_system}.consume", kind="CONSUMER", attributes=attrs, parent_headers=headers) as sp:
        yield sp


@contextmanager
def start_produce_span(
    topic: str,
    key: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    messaging_system: str = "bus",
):
    """
    Span specialized for message PRODUCE. Injects trace headers into provided headers.
    """
    attrs = {
        "messaging.system": messaging_system,
        "messaging.operation": "send",
        "messaging.destination": topic,
    }
    if key is not None:
        attrs["messaging.message.key"] = str(key)

    # Ensure we can mutate headers for injection
    hdrs = headers if headers is not None else {}

    with start_span(f"{messaging_system}.produce", kind="PRODUCER", attributes=attrs) as sp:
        # inject current context into headers for downstream propagation
        inj = inject_headers(hdrs)
        # copy back into caller's dict reference
        headers.update(inj) if headers is not None else None
        yield sp


# ================== Decorators (sync/async) ==================

def traced(name: Optional[str] = None, kind: str = "INTERNAL"):
    """
    Decorate a sync function, creating a span around the call.
    """
    def deco(fn: Callable):
        span_name = name or f"{fn.__module__}.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with start_span(span_name, kind=kind, attributes={"code.function": fn.__name__}):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    _record_exception(e)
                    raise
        return wrapper
    return deco


def atraced(name: Optional[str] = None, kind: str = "INTERNAL"):
    """
    Decorate an async function, creating a span around the coroutine.
    """
    def deco(fn: Callable[..., Awaitable[Any]]):
        span_name = name or f"{fn.__module__}.{fn.__name__}"

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # Use manual context manager for async
            cm = start_span(span_name, kind=kind, attributes={"code.function": fn.__name__})
            sp = cm.__enter__()
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                _record_exception(e, sp)
                raise
            finally:
                cm.__exit__(None, None, None)
        return wrapper
    return deco


# ================== Internal: no-op tracer/span ==================

class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **_kw):
        yield _NoopSpan(name, {})


class _NoopSpan:
    def __init__(self, name: str, attributes: Dict[str, Any]):
        self.name = name
        self.attributes = attributes
        self.start_ns = time.time_ns()

    # No-op API compatibility
    def set_attribute(self, *_a, **_k):  # noqa: D401
        return

    def set_status(self, *_a, **_k):
        return

    def record_exception(self, *_a, **_k):
        return


def _record_exception(exc: BaseException, span: Optional[Any] = None) -> None:
    if _OTEL and span is not None:
        try:
            span.record_exception(exc)
            if Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
        except Exception:
            pass


# ================== Header carrier for OTel extract ==================

class _DictGetter:
    """OTel extraction carrier: maps get_all + keys over a dict-like."""
    def __init__(self, data: Dict[str, str]):
        self._d = {str(k): str(v) for k, v in (data or {}).items()}

    def get(self, key: str) -> Optional[str]:  # not used by OTel, but handy
        return self._d.get(key)

    def get_all(self, key: str):
        v = self._d.get(key)
        return [v] if v is not None else []

    def keys(self):
        return list(self._d.keys())