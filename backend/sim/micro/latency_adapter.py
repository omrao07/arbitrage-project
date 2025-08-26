# backend/utils/latency_adapter.py
from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("latency")
if not log.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

# ===================== Data Model =====================

@dataclass
class Sample:
    name: str
    start_ms: int
    end_ms: int
    duration_ms: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stats:
    count: int
    mean: float
    p50: float
    p90: float
    p99: float
    max: float


# ===================== Core Adapter =====================

class LatencyAdapter:
    """
    Wrap arbitrary callables to measure latency.
    Stores rolling samples and exposes stats.

    Example:
        la = LatencyAdapter("broker.place_order")
        wrapped = la.wrap(broker.place_order)
        ack = wrapped(order)
        print(la.stats())
    """

    def __init__(self, name: str, *, window: int = 500):
        self.name = name
        self.window = window
        self._samples: List[Sample] = []
        self._lock = threading.Lock()

    # ---- context manager ----
    @contextmanager
    def measure(self, **meta):
        t0 = time.time()
        try:
            yield
        finally:
            t1 = time.time()
            self._record(t0, t1, meta)

    # ---- function wrapper ----
    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrap(*a, **k):
            t0 = time.time()
            try:
                return fn(*a, **k)
            finally:
                t1 = time.time()
                self._record(t0, t1, {"fn": fn.__name__})
        return _wrap

    # ---- record & stats ----
    def _record(self, t0: float, t1: float, meta: Optional[Dict[str, Any]] = None):
        s = Sample(
            name=self.name,
            start_ms=int(t0 * 1000),
            end_ms=int(t1 * 1000),
            duration_ms=(t1 - t0) * 1000,
            meta=meta or {},
        )
        with self._lock:
            self._samples.append(s)
            if len(self._samples) > self.window:
                self._samples.pop(0)
        log.debug("latency %s: %.2fms", self.name, s.duration_ms)

    def samples(self) -> List[Sample]:
        with self._lock:
            return list(self._samples)

    def stats(self) -> Optional[Stats]:
        with self._lock:
            if not self._samples:
                return None
            durs = [s.duration_ms for s in self._samples]
        return Stats(
            count=len(durs),
            mean=statistics.mean(durs),
            p50=statistics.median(durs),
            p90=statistics.quantiles(durs, n=10)[8],
            p99=statistics.quantiles(durs, n=100)[98],
            max=max(durs),
        )

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


# ===================== Global Registry =====================

_registry: Dict[str, LatencyAdapter] = {}
_reg_lock = threading.Lock()

def get_adapter(name: str) -> LatencyAdapter:
    with _reg_lock:
        if name not in _registry:
            _registry[name] = LatencyAdapter(name)
        return _registry[name]

def all_stats() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with _reg_lock:
        for k, la in _registry.items():
            s = la.stats()
            if s:
                out[k] = {
                    "count": s.count,
                    "mean_ms": round(s.mean, 3),
                    "p50_ms": round(s.p50, 3),
                    "p90_ms": round(s.p90, 3),
                    "p99_ms": round(s.p99, 3),
                    "max_ms": round(s.max, 3),
                }
    return out