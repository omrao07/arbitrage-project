# backend/utils/latency.py
from __future__ import annotations

import os
import time
import math
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List

# ----------------------- Optional Redis backend -----------------------
def _get_redis():
    try:
        import redis  # type: ignore
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        r = redis.Redis(host=host, port=port, decode_responses=True)
        # ping lazily on first write/read
        return r
    except Exception:
        return None

_REDIS = _get_redis()
_KEY_PREFIX = os.getenv("LAT_KEY_PREFIX", "latency")
_LOCK = threading.RLock()

# ----------------------- Histogram helper -----------------------------
@dataclass
class Histo:
    """
    Fixed-bucket histogram in milliseconds.
    Buckets are upper bounds; the last bucket "+inf" collects the rest.
    """
    bounds_ms: List[float] = field(default_factory=lambda: [
        0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000
    ])
    counts: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.counts:
            self.counts = [0 for _ in range(len(self.bounds_ms) + 1)]  # +inf

    def add(self, ms: float) -> None:
        x = max(0.0, float(ms))
        for i, b in enumerate(self.bounds_ms):
            if x <= b:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def merge(self, other: "Histo") -> None:
        if self.bounds_ms != other.bounds_ms:
            raise ValueError("histogram bounds mismatch")
        if len(self.counts) != len(other.counts):
            raise ValueError("histogram counts shape mismatch")
        for i in range(len(self.counts)):
            self.counts[i] += other.counts[i]

    def quantile(self, q: float) -> float:
        """Approximate percentile from the histogram (linear within bucket)."""
        q = min(0.999999, max(0.0, q))
        total = sum(self.counts)
        if total <= 0:
            return 0.0
        target = q * total
        cum = 0
        prev_edge = 0.0
        for i, c in enumerate(self.counts):
            edge = self.bounds_ms[i] if i < len(self.bounds_ms) else float("inf")
            prev_cum = cum
            cum += c
            if cum >= target:
                # interpolate inside this bucket
                inside = max(0, int(target - prev_cum))
                width = (edge - prev_edge) if math.isfinite(edge) else max(prev_edge, 1.0)
                frac = 0.0 if c == 0 else inside / c
                return prev_edge + frac * width
            prev_edge = edge
        return self.bounds_ms[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {"bounds_ms": self.bounds_ms, "counts": self.counts}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Histo":
        h = Histo(bounds_ms=list(map(float, d.get("bounds_ms", []))) or None)  # type: ignore[arg-type]
        h.counts = list(map(int, d.get("counts", []))) or [0]*(len(h.bounds_ms)+1)
        return h


# ----------------------- Metric record --------------------------------
@dataclass
class Metric:
    name: str
    count: int = 0
    mean_ms: float = 0.0
    m2: float = 0.0          # for variance (Welford)
    max_ms: float = 0.0
    last_ms: float = 0.0
    ema_ms: float = 0.0
    ema_alpha: float = 0.2   # configurable via env LAT_EMA_ALPHA
    last_ts: float = 0.0
    histo: Histo = field(default_factory=Histo)

    def add(self, ms: float, ts: Optional[float] = None) -> None:
        x = max(0.0, float(ms))
        self.count += 1
        # Welford
        delta = x - self.mean_ms
        self.mean_ms += delta / self.count
        self.m2 += delta * (x - self.mean_ms)
        # EMA
        a = self.ema_alpha
        self.ema_ms = (a * x + (1 - a) * self.ema_ms) if self.count > 1 else x
        # extrema, last
        self.max_ms = max(self.max_ms, x)
        self.last_ms = x
        self.last_ts = ts if ts is not None else time.time()
        # histo
        self.histo.add(x)

    def var(self) -> float:
        return 0.0 if self.count < 2 else self.m2 / (self.count - 1)

    def std(self) -> float:
        v = self.var()
        return math.sqrt(v) if v > 0 else 0.0

    def percentiles(self) -> Dict[str, float]:
        return {
            "p50_ms": self.histo.quantile(0.50),
            "p90_ms": self.histo.quantile(0.90),
            "p95_ms": self.histo.quantile(0.95),
            "p99_ms": self.histo.quantile(0.99),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["histo"] = self.histo.to_dict()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Metric":
        m = Metric(
            name=d["name"],
            count=int(d.get("count", 0)),
            mean_ms=float(d.get("mean_ms", 0.0)),
            m2=float(d.get("m2", 0.0)),
            max_ms=float(d.get("max_ms", 0.0)),
            last_ms=float(d.get("last_ms", 0.0)),
            ema_ms=float(d.get("ema_ms", 0.0)),
            ema_alpha=float(d.get("ema_alpha", float(os.getenv("LAT_EMA_ALPHA", "0.2")))),
            last_ts=float(d.get("last_ts", 0.0)),
        )
        m.histo = Histo.from_dict(d.get("histo", {}))
        return m


# ----------------------- Global registry ------------------------------
class LatencyRegistry:
    """
    Thread-safe metric registry; flushes to Redis (optional).
    """
    def __init__(self):
        self._m: Dict[str, Metric] = {}
        try:
            self.default_alpha = float(os.getenv("LAT_EMA_ALPHA", "0.2"))
        except Exception:
            self.default_alpha = 0.2

    def _get(self, name: str) -> Metric:
        with _LOCK:
            m = self._m.get(name)
            if m is None:
                m = Metric(name=name, ema_alpha=self.default_alpha)
                self._m[name] = m
        return m

    # ---- record datapoint ----
    def record(self, name: str, ms: float) -> None:
        m = self._get(name)
        m.add(ms)

        # write-through to Redis (best-effort, non-blocking-ish)
        if _REDIS:
            try:
                key = f"{_KEY_PREFIX}:{name}"
                # store as JSON blob; HSET could work but blob is simpler
                _REDIS.set(key, json.dumps(m.to_dict()))
            except Exception:
                pass

    # ---- fetch single/all ----
    def get(self, name: str) -> Dict[str, Any]:
        # prefer in-memory; fall back to Redis if missing
        m = self._m.get(name)
        if not m and _REDIS:
            try:
                s = _REDIS.get(f"{_KEY_PREFIX}:{name}")
                if s:
                    m = Metric.from_dict(json.loads(s)) # type: ignore
                    with _LOCK:
                        self._m[name] = m
            except Exception:
                pass
        if not m:
            return {"name": name, "count": 0}
        out = m.to_dict()
        out.update(m.percentiles())
        out["std_ms"] = m.std()
        return out

    def all(self) -> Dict[str, Any]:
        # Combine in-memory and Redis keys
        out: Dict[str, Any] = {}
        names = set(self._m.keys())
        if _REDIS:
            try:
                # naive scan
                for key in _REDIS.scan_iter(f"{_KEY_PREFIX}:*"):
                    names.add(key.split(":", 1)[1])
            except Exception:
                pass
        for n in sorted(names):
            out[n] = self.get(n)
        return out

# Singleton
_REG = LatencyRegistry()

# ----------------------- Public API -----------------------------------
def record(name: str, ms: float) -> None:
    """Record a latency sample (milliseconds)."""
    _REG.record(name, ms)

@contextmanager
def timer(name: str):
    """
    Context manager:
        with timer("api.order_submit"):
            <code>
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        _REG.record(name, dt)

def track(name: Optional[str] = None):
    """
    Decorator:
        @track("alpha.signal")
        def compute(...): ...
    """
    def _wrap(fn):
        label = name or f"{fn.__module__}.{fn.__name__}"
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = (time.perf_counter() - t0) * 1000.0
                _REG.record(label, dt)
        return inner
    return _wrap

def get_stats(metric: str) -> Dict[str, Any]:
    """Get one metric with percentiles, std, etc."""
    return _REG.get(metric)

def all_stats() -> Dict[str, Any]:
    """Get all metrics. Compatible with runner’s `latency` command."""
    return _REG.all()

def to_prometheus() -> str:
    """
    Render a Prometheus-compatible text exposition (summary-like).
    Exported metrics:
      latency_count{name}
      latency_mean_ms{name}
      latency_ema_ms{name}
      latency_max_ms{name}
      latency_p50_ms{name}, p90, p95, p99
    """
    allm = _REG.all()
    lines = []
    for name, m in allm.items():
        safe = name.replace('"', '\\"')
        def line(k, v):
            lines.append(f'latency_{k}{{name="{safe}"}} {float(v)}')
        line("count", m.get("count", 0))
        line("mean_ms", m.get("mean_ms", 0.0))
        line("ema_ms", m.get("ema_ms", 0.0))
        line("max_ms", m.get("max_ms", 0.0))
        line("p50_ms", m.get("p50_ms", 0.0))
        line("p90_ms", m.get("p90_ms", 0.0))
        line("p95_ms", m.get("p95_ms", 0.0))
        line("p99_ms", m.get("p99_ms", 0.0))
    return "\n".join(lines) + ("\n" if lines else "")

# ----------------------- Tiny smoke test ------------------------------
if __name__ == "__main__":
    # Example usage
    with timer("demo.block"):
        time.sleep(0.012)
    @track("demo.fn")
    def foo():
        time.sleep(0.003)
    for _ in range(5):
        foo()
    record("demo.manual", 7.5)
    print(json.dumps(all_stats(), indent=2))
    print("\n--- Prometheus ---\n")
    print(to_prometheus())