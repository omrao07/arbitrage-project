# backend/ops/metrics.py
from __future__ import annotations

import os, time, math, json, threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable

# ---------- optional Redis (graceful fallback) ----------
HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_METRIC = os.getenv("METRICS_STREAM", "ops.metrics")
HSET_GAUGES   = os.getenv("METRICS_GAUGE_KEY", "ops:gauges")
EMIT_EVERY_MS = int(os.getenv("METRICS_EMIT_MS", "2000"))

# ---------- time helpers ----------
def now_ms() -> int: return int(time.time() * 1000)
def ns_to_ms(ns: int) -> float: return ns / 1_000_000.0

# ---------- tiny quantile estimator (P² algorithm) ----------
class P2Quantiles:
    """
    Memory-light quantile estimator (P²). Tracks q in [0,1] list, default [0.5,0.9,0.95,0.99].
    """
    def __init__(self, probs: List[float] = [0.5, 0.9, 0.95, 0.99]):
        self.probs = sorted(probs)
        self._n = 0
        self._q = []   # marker heights
        self._npos = []  # desired marker positions
        self._npos_inc = [p*2 for p in self.probs]  # used after init

    def add(self, x: float):
        if self._n < 5:
            # bootstrap with exact values
            self._q.append(float(x))
            self._q.sort()
            self._n += 1
            if self._n == 5:
                self._init_markers()
            return
        self._n += 1
        # locate cell k
        k = 0
        if x < self._q[0]:
            self._q[0] = x
            k = 0
        elif x >= self._q[-1]:
            self._q[-1] = x
            k = len(self._q) - 2
        else:
            for i in range(1, len(self._q)):
                if x < self._q[i]:
                    k = i - 1
                    break
        # update desired positions
        for i in range(len(self._npos)):
            self._npos[i] += self._npos_inc[i] # type: ignore
        # adjust heights
        self._adjust(k)

    def _init_markers(self):
        # five markers for min, q1-ish, median-ish, q3-ish, max; we repurpose for our probs by interpolation
        # we still keep 5 markers to preserve invariants; map self.probs to interpolation on get()
        self._q = [self._q[0], self._q[1], self._q[2], self._q[3], self._q[4]]
        self._npos = [1, 2, 3, 4, 5]
        self._npos_inc = [0.0, 0.0, 0.0, 0.0, 0.0]  # managed differently during adjust

    def _adjust(self, k: int):
        # desired positions for classic P² quantiles (0, 0.25, 0.5, 0.75, 1)
        # we approximate by nudging internal markers toward their target based on n
        # This is a minimalist implementation adequate for live dashboards.
        # Move inner markers by at most 1 position
        for i in range(1, 4):
            d = (self._npos[i] - (i+1))  # desired - current
            s = 1 if d >= 1 else (-1 if d <= -1 else 0)
            if s != 0:
                # parabolic prediction
                q_im1, q_i, q_ip1 = self._q[i-1], self._q[i], self._q[i+1]
                q_hat = q_i + s * (( (i+s - (i-1)) * (q_ip1 - q_i) / ( (i+1) - i ) ) + ( (i - (i+s)) * (q_i - q_im1) / ( i - (i-1) ) ))
                # guard monotonicity
                if q_im1 < q_hat < q_ip1:
                    self._q[i] = q_hat
                else:
                    self._q[i] = q_i + s * (q_ip1 - q_im1) / 2.0
                # update current position
                self._npos[i] += s

    def get(self, p: float) -> float:
        if self._n == 0:
            return float("nan")
        if self._n < 5:
            # approximate from raw bootstrap
            arr = sorted(self._q)
            idx = max(0, min(len(arr)-1, int(round((len(arr)-1)*p))))
            return arr[idx]
        # interpolate p over the 5 base markers (0, 0.25, 0.5, 0.75, 1)
        base_p = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i in range(1, 5):
            if p <= base_p[i]:
                t = (p - base_p[i-1]) / (base_p[i] - base_p[i-1] + 1e-12)
                return self._q[i-1] * (1 - t) + self._q[i] * t
        return self._q[-1]

# ---------- rolling EWMA rate ----------
class EWMA:
    def __init__(self, half_life_sec: float = 10.0):
        self.alpha = math.exp(math.log(0.5) / max(half_life_sec, 1e-6))
        self.last_ts = None
        self.rate = 0.0
        self.count = 0

    def incr(self, n: int = 1):
        now = time.time()
        if self.last_ts is None:
            self.rate = 0.0
        else:
            dt = max(1e-6, now - self.last_ts)
            inst_rate = n / dt
            self.rate = self.alpha * self.rate + (1 - self.alpha) * inst_rate
        self.last_ts = now
        self.count += n

# ---------- metric primitives ----------
@dataclass
class Counter:
    name: str
    help: str = ""
    labels: Tuple[str, ...] = field(default_factory=tuple)
    value: float = 0.0
    rate: EWMA = field(default_factory=lambda: EWMA(half_life_sec=10))

    def inc(self, n: float = 1.0):
        self.value += n
        self.rate.incr(int(n))

    def to_prom(self, label_vals: Tuple[str, ...] = ()) -> str:
        lab = self._lab(label_vals)
        return f"# HELP {self.name} {self.help}\n# TYPE {self.name} counter\n{self.name}{lab} {self.value}\n{self.name}_rate_per_sec{lab} {self.rate.rate}\n"

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "rate_per_sec": self.rate.rate}

    def _lab(self, vals: Tuple[str, ...]) -> str:
        if not self.labels: return ""
        pairs = ",".join(f'{k}="{v}"' for k, v in zip(self.labels, vals))
        return f"{{{pairs}}}"

@dataclass
class Gauge:
    name: str
    help: str = ""
    labels: Tuple[str, ...] = field(default_factory=tuple)
    value: float = 0.0

    def set(self, v: float): self.value = float(v)
    def add(self, dv: float): self.value += float(dv)

    def to_prom(self, label_vals: Tuple[str, ...] = ()) -> str:
        lab = self._lab(label_vals)
        return f"# HELP {self.name} {self.help}\n# TYPE {self.name} gauge\n{self.name}{lab} {self.value}\n"

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value}

    def _lab(self, vals: Tuple[str, ...]) -> str:
        if not self.labels: return ""
        pairs = ",".join(f'{k}="{v}"' for k, v in zip(self.labels, vals))
        return f"{{{pairs}}}"

@dataclass
class Histogram:
    name: str
    help: str = ""
    buckets: List[float] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    counts: List[int] = field(init=False)
    sum: float = 0.0
    total: int = 0
    q: P2Quantiles = field(default_factory=P2Quantiles)

    def __post_init__(self): self.counts = [0 for _ in self.buckets]

    def observe(self, x_ms: float):
        self.sum += x_ms
        self.total += 1
        for i, b in enumerate(self.buckets):
            if x_ms <= b:
                self.counts[i] += 1
                break
        self.q.add(x_ms)

    def to_prom(self) -> str:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        acc = 0
        for b, c in zip(self.buckets, self.counts):
            acc += c
            out.append(f'{self.name}_bucket{{le="{b}"}} {acc}')
        out.append(f'{self.name}_bucket{{le="+Inf"}} {self.total}')
        out.append(f"{self.name}_sum {self.sum}")
        out.append(f"{self.name}_count {self.total}")
        # export P-quantiles as gauges for convenience
        for p in (0.5, 0.9, 0.95, 0.99):
            out.append(f'{self.name}_p{int(p*100)} {self.q.get(p)}')
        return "\n".join(out) + "\n"

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name, "sum_ms": self.sum, "count": self.total,
            "p50_ms": self.q.get(0.5), "p90_ms": self.q.get(0.9), "p95_ms": self.q.get(0.95), "p99_ms": self.q.get(0.99)
        }

# ---------- registry ----------
class Metrics:
    """
    Global-ish registry. Use one per process or subservice.
    """
    def __init__(self, *, redis_url: Optional[str] = None, auto_emit: bool = True):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._hists: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._r = None
        if HAVE_REDIS:
            try:
                self._r = Redis.from_url(redis_url or REDIS_URL, decode_responses=True)  # type: ignore
                self._r.ping()
            except Exception:
                self._r = None
        self._stop = threading.Event()
        if auto_emit:
            t = threading.Thread(target=self._emitter_loop, daemon=True)
            t.start()

    # ---- builders
    def counter(self, name: str, help: str = "") -> Counter:
        with self._lock:
            return self._counters.setdefault(name, Counter(name=name, help=help))
    def gauge(self, name: str, help: str = "") -> Gauge:
        with self._lock:
            return self._gauges.setdefault(name, Gauge(name=name, help=help))
    def histogram(self, name: str, help: str = "", buckets: Optional[List[float]] = None) -> Histogram:
        with self._lock:
            if name not in self._hists:
                self._hists[name] = Histogram(name=name, help=help, buckets=buckets or [1,2,5,10,20,50,100,200,500,1000])
            return self._hists[name]

    # ---- timer convenience ---------------------------------------------------
    def timed(self, name: str, *, buckets: Optional[List[float]] = None):
        """
        Decorator/context manager that observes function wall time into a histogram in ms.
        """
        hist = self.histogram(name, help="Function wall time (ms)", buckets=buckets)
        def deco(fn: Callable):
            def wrap(*a, **k):
                t0 = time.perf_counter_ns()
                try:
                    return fn(*a, **k)
                finally:
                    t1 = time.perf_counter_ns()
                    hist.observe(ns_to_ms(t1 - t0))
            return wrap
        return deco

    def timer(self, name: str, *, buckets: Optional[List[float]] = None):
        hist = self.histogram(name, help="Timer (ms)", buckets=buckets)
        class _T:
            def __enter__(self):
                self._t0 = time.perf_counter_ns(); return self
            def __exit__(self, et, ev, tb):
                hist.observe(ns_to_ms(time.perf_counter_ns() - self._t0))
        return _T()

    # ---- exporters -----------------------------------------------------------
    def to_prometheus_text(self) -> str:
        with self._lock:
            out = []
            for g in self._gauges.values():
                out.append(g.to_prom())
            for c in self._counters.values():
                out.append(c.to_prom())
            for h in self._hists.values():
                out.append(h.to_prom())
            return "\n".join(out)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ts_ms": now_ms(),
                "gauges": {k: v.snapshot() for k,v in self._gauges.items()},
                "counters": {k: v.snapshot() for k,v in self._counters.items()},
                "histograms": {k: v.snapshot() for k,v in self._hists.items()},
            }

    # ---- background emitter --------------------------------------------------
    def _emitter_loop(self):
        while not self._stop.is_set():
            try:
                self._emit_once()
            except Exception:
                pass
            self._stop.wait(EMIT_EVERY_MS/1000.0)

    def _emit_once(self):
        snap = self.snapshot()
        if self._r:
            try:
                self._r.xadd(STREAM_METRIC, {"json": json.dumps(snap)}, maxlen=200_000, approximate=True)  # type: ignore
                # also HSET current gauges for fast UI reads
                if self._gauges:
                    g = {k: json.dumps(v.snapshot()) for k,v in self._gauges.items()}
                    self._r.hset(HSET_GAUGES, mapping=g)  # type: ignore
            except Exception:
                pass

    def stop(self): self._stop.set()

# ---------- convenience singleton ----------
_global_metrics: Optional[Metrics] = None
def get_metrics() -> Metrics:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = Metrics()
    return _global_metrics

# ---------- CLI (quick demo) -------------------------------------------------
def _cli():
    import argparse, random
    ap = argparse.ArgumentParser("metrics")
    sub = ap.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Emit some sample metrics")
    demo.add_argument("--iters", type=int, default=200)

    expo = sub.add_parser("prom", help="Print Prometheus text once")

    args = ap.parse_args()
    m = get_metrics()

    if args.cmd == "demo":
        hits = m.counter("orders_total", "Number of orders seen")
        rqsz = m.histogram("order_route_ms", "Route path time (ms)", buckets=[1,2,5,10,20,50,100,200,500])
        inq = m.gauge("risk_queue_depth", "In-flight orders in risk queue")
        for i in range(args.iters):
            hits.inc()
            inq.set(max(0, math.sin(i/10.0)*50 + 50))
            rqsz.observe(max(0.1, random.gauss(8, 4)))
            time.sleep(0.02)
        print(json.dumps(m.snapshot(), indent=2))
        m.stop()

    elif args.cmd == "prom":
        print(m.to_prometheus_text())
        m.stop()

if __name__ == "__main__":
    _cli()