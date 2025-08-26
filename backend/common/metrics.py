# backend/common/metrics.py
"""
Minimal, production-friendly metrics (stdlib only).

What you get
------------
- Counter:       incr("orders_sent")
- Gauge:         gauge_set("open_orders", 7)
- Histogram:     hist_obs("latency_ms", 12.3)
- Timer (ctx):   with timer("router_ms"): router.route(...)
- Timer (decor): @timed("manager_ms")
- Export:        export_json(), export_prometheus()
- Sink:          flush_jsonl("runs/metrics.jsonl", reset=False)

Design notes
-----------
- Thread-safe (RLock).
- Aggregated histograms (no raw samples kept).
- Percentiles estimated using fixed buckets + HDR-ish interpolation.
- No dependencies; integrate with any process.
"""

from __future__ import annotations
import time, math, json, os, threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple, Callable, Any
from contextlib import contextmanager

__all__ = [
    "incr", "add", "gauge_set", "gauge_add",
    "hist_obs", "timer", "timed",
    "export_json", "export_prometheus", "flush_jsonl",
    "set_default_buckets", "reset_all"
]

# ---------------- Registry ----------------

@dataclass
class _HistAgg:
    count: int = 0
    sum: float = 0.0
    min: float = math.inf
    max: float = -math.inf
    buckets: Dict[float, int] = None  # type: ignore # upper_bound -> count

    def to_dict(self):
        return {
            "count": self.count, "sum": self.sum,
            "min": (None if self.min is math.inf else self.min),
            "max": (None if self.max == -math.inf else self.max),
            "buckets": dict(self.buckets or {})
        }

class _Registry:
    def __init__(self):
        self._lock = threading.RLock()
        self._ctr: Dict[str, int] = {}
        self._gauge: Dict[str, float] = {}
        self._hist: Dict[str, _HistAgg] = {}
        # default latency buckets (ms): 0.25..10000
        self._buckets: List[float] = [
            0.25, 0.5, 1, 2, 5, 10, 25, 50, 75, 100,
            250, 500, 750, 1000, 2000, 5000, 10000
        ]

    # ------- counters -------
    def incr(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._ctr[key] = self._ctr.get(key, 0) + int(n)

    def add(self, key: str, n: float) -> None:
        # for counters that may add floats (fees, notional, etc.)
        with self._lock:
            self._ctr[key] = self._ctr.get(key, 0) + n # type: ignore

    # ------- gauges -------
    def gauge_set(self, key: str, val: float) -> None:
        with self._lock:
            self._gauge[key] = float(val)

    def gauge_add(self, key: str, delta: float) -> None:
        with self._lock:
            self._gauge[key] = float(self._gauge.get(key, 0.0) + delta)

    # ------- histograms -------
    def hist_obs(self, key: str, v: float) -> None:
        if v is None or math.isnan(v): return
        with self._lock:
            h = self._hist.get(key)
            if h is None:
                h = _HistAgg(buckets={b:0 for b in self._buckets})
                self._hist[key] = h
            h.count += 1
            h.sum += v
            if v < h.min: h.min = v
            if v > h.max: h.max = v
            # bucket: first upper bound >= v (last bucket is +Inf style)
            placed = False
            for ub in self._buckets:
                if v <= ub:
                    h.buckets[ub] += 1
                    placed = True
                    break
            if not placed:
                # overflow -> extend +Inf bucket
                ub = float("inf")
                h.buckets[ub] = h.buckets.get(ub, 0) + 1

    # ------- export -------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            hist = {k: v.to_dict() for k, v in self._hist.items()}
            return {"ts": time.time(), "counters": dict(self._ctr),
                    "gauges": dict(self._gauge), "histograms": hist}

    def reset(self) -> None:
        with self._lock:
            self._ctr.clear(); self._gauge.clear(); self._hist.clear()

    def set_buckets(self, buckets: Iterable[float]) -> None:
        with self._lock:
            bs = sorted(float(x) for x in buckets)
            if not bs: raise ValueError("buckets must be non-empty")
            self._buckets = bs

_REG = _Registry()

# ---------------- Public API ----------------

def incr(key: str, n: int = 1) -> None: _REG.incr(key, n)
def add(key: str, n: float) -> None: _REG.add(key, n)
def gauge_set(key: str, v: float) -> None: _REG.gauge_set(key, v)
def gauge_add(key: str, dv: float) -> None: _REG.gauge_add(key, dv)
def hist_obs(key: str, v: float) -> None: _REG.hist_obs(key, v)

def set_default_buckets(buckets: Iterable[float]) -> None:
    """Override global histogram buckets (e.g., in seconds vs ms)."""
    _REG.set_buckets(buckets)

def reset_all() -> None: _REG.reset()

# ---- timers ----

@contextmanager
def timer(key: str, *, scale: float = 1000.0):
    """
    Measure block latency and record as histogram.
    scale=1000.0 -> milliseconds; set 1.0 for seconds, 1e6 for microseconds.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * scale
        hist_obs(key, dt)

def timed(key: str, *, scale: float = 1000.0) -> Callable:
    """Decorator form of `timer`."""
    def deco(fn: Callable):
        def wrap(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = (time.perf_counter() - t0) * scale
                hist_obs(key, dt)
        return wrap
    return deco

# ---- export helpers ----

def _percentiles_from_buckets(bmap: Dict[float, int], qs: Iterable[float]) -> Dict[str, float]:
    # Convert cumulative buckets to quantiles via interpolation.
    items = sorted(bmap.items(), key=lambda x: (math.inf if math.isinf(x[0]) else x[0]))
    counts = [c for _, c in items]
    total = sum(counts)
    if total == 0: return {f"p{int(q*100)}": None for q in qs} # type: ignore

    cumsum = []
    s = 0
    for c in counts:
        s += c; cumsum.append(s)

    # Build edges for interpolation
    edges: List[float] = []
    last = 0.0
    for ub, _ in items:
        if math.isinf(ub):
            ub = last * 2 if last > 0 else 1.0
        edges.append(ub); last = ub

    res = {}
    for q in qs:
        target = q * total
        # find first bucket with cum >= target
        i = 0
        while i < len(cumsum) and cumsum[i] < target: i += 1
        if i == 0:
            res[f"p{int(q*100)}"] = edges[0]; continue
        lo_edge = edges[i-1]; hi_edge = edges[i]
        lo_cum = cumsum[i-1]; hi_cum = cumsum[i]
        frac = 0.0 if hi_cum == lo_cum else (target - lo_cum) / (hi_cum - lo_cum)
        res[f"p{int(q*100)}"] = lo_edge + (hi_edge - lo_edge) * frac
    return res

def export_json() -> Dict[str, Any]:
    """Returns a point-in-time metrics snapshot with p50/p90/p99 for histograms."""
    snap = _REG.snapshot()
    out = {"ts": snap["ts"], "counters": snap["counters"], "gauges": snap["gauges"], "histograms": {}}
    for k, h in snap["histograms"].items():
        pct = _percentiles_from_buckets(h["buckets"], (0.5, 0.9, 0.99))
        mean = (h["sum"] / h["count"]) if h["count"] else None
        out["histograms"][k] = {
            "count": h["count"], "sum": h["sum"], "mean": mean,
            "min": h["min"], "max": h["max"], **pct
        }
    return out

def export_prometheus(namespace: str = "arb") -> str:
    """
    Prometheus exposition format (text). Example:
      arb_counter_orders_sent 123
      arb_gauge_open_orders 7
      arb_hist_latency_ms_count 42
      arb_hist_latency_ms_bucket{le="10"} 12
      ...
    """
    snap = _REG.snapshot()
    lines: List[str] = []
    pref = namespace

    for k, v in snap["counters"].items():
        lines.append(f"{pref}_counter_{k} {v}")

    for k, v in snap["gauges"].items():
        lines.append(f"{pref}_gauge_{k} {v}")

    for k, h in snap["histograms"].items():
        # buckets (cumulative)
        total = 0
        items = sorted(h["buckets"].items(), key=lambda x: (math.inf if math.isinf(x[0]) else x[0]))
        for ub, c in items:
            total += c
            le = "+Inf" if math.isinf(ub) else str(ub)
            lines.append(f'{pref}_hist_{k}_bucket{{le="{le}"}} {total}')
        lines.append(f"{pref}_hist_{k}_count {h['count']}")
        lines.append(f"{pref}_hist_{k}_sum {h['sum']}")
    return "\n".join(lines) + "\n"

def flush_jsonl(path: str, *, reset: bool = False) -> str:
    """
    Append a metrics JSON snapshot to a JSONL file. Creates dirs as needed.
    reset=True clears in-memory metrics after flush (useful for intervals).
    """
    snap = export_json()
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snap) + "\n")
    if reset:
        _REG.reset()
    return path