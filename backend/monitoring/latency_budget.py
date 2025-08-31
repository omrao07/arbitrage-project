# backend/ops/latency_budget.py
from __future__ import annotations

import os, time, json, math, threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Callable, Tuple

# ---------- optional Redis (graceful fallback) ----------
HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_LAT  = os.getenv("LAT_STREAM", "ops.latency")
STREAM_VIOL = os.getenv("LAT_VIOL_STREAM", "ops.latency.viol")
STREAM_SUM  = os.getenv("LAT_SUM_STREAM", "ops.latency.summary")

# ---------- small math helpers ----------
def now_ms() -> int: return int(time.time() * 1000)
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))

# ---------- models ----------
@dataclass
class Budget:
    name: str
    slo_ms: int                                   # target P99 (or max) budget
    stages: Dict[str, int] = field(default_factory=dict)  # {stage: alloc_ms}
    mode: str = "p99"                             # "p99" | "max"
    soft_margin_ms: int = 0                       # slack before flagging
    enabled: bool = True

@dataclass
class Sample:
    name: str
    stage: str
    dur_ms: float
    ts_ms: int

# ---------- backend sink ----------
class _Sink:
    def __init__(self, url: Optional[str] = None):
        self.r = None
        if HAVE_REDIS:
            try:
                self.r = Redis.from_url(url or REDIS_URL, decode_responses=True)  # type: ignore
                self.r.ping()
            except Exception:
                self.r = None

    def emit(self, stream: str, obj: Dict[str, Any], maxlen: int = 100_000):
        payload = {"json": json.dumps(obj)}
        if self.r:
            try:
                self.r.xadd(stream, payload, maxlen=maxlen, approximate=True)  # type: ignore
                return
            except Exception:
                pass
        # fallback: print
        print(f"[latency:{stream}] {obj}")

# ---------- rolling stats (EWMA + compact hist) ----------
class RollingStats:
    """
    Lightweight rolling stats:
      - EWMA mean & variance
      - fixed-size log-bucket histogram for quantiles
    """
    def __init__(self, half_life: float = 30.0, buckets: int = 20, max_ms: float = 10_000.0):
        self.alpha = math.exp(math.log(0.5) / max(1e-6, half_life))  # decay per sample
        self.mean = 0.0
        self.var = 0.0
        self.n = 0
        self.max_ms = float(max_ms)
        self.buckets = buckets
        self.hist = [0] * buckets  # log scale
        self.total = 0

    def _bucket_of(self, ms: float) -> int:
        x = clamp(ms, 0.0, self.max_ms)
        if x <= 0: return 0
        # log scale: bucket 0..b-1
        r = math.log10(1 + 9 * x / self.max_ms)  # 0..1
        idx = int(r * self.buckets)
        return min(self.buckets - 1, max(0, idx))

    def add(self, ms: float):
        # EWMA (West's algorithm for var with forgetting)
        self.n += 1
        if self.n == 1:
            self.mean = ms; self.var = 0.0
        else:
            diff = ms - self.mean
            self.mean += (1 - self.alpha) * diff
            self.var = self.alpha * (self.var + (1 - self.alpha) * diff * diff)
        # hist
        self.hist[self._bucket_of(ms)] += 1
        self.total += 1

    def quantile(self, q: float) -> float:
        q = clamp(q, 0.0, 1.0)
        if self.total <= 0: return float("nan")
        target = int(math.ceil(self.total * q))
        acc = 0
        for i, c in enumerate(self.hist):
            acc += c
            if acc >= target:
                # invert bucket to ms midpoint
                lo = 0 if i == 0 else (10**(i / self.buckets) - 1) / 9 * self.max_ms
                hi = (10**((i + 1) / self.buckets) - 1) / 9 * self.max_ms
                return 0.5 * (lo + hi)
        # fallback high end
        return self.max_ms

    @property
    def p50(self): return self.quantile(0.50)
    @property
    def p90(self): return self.quantile(0.90)
    @property
    def p95(self): return self.quantile(0.95)
    @property
    def p99(self): return self.quantile(0.99)

# ---------- core manager ----------
class LatencyBudget:
    """
    Define budgets, time stages, compute headroom/violations, emit events.
    Usage:
        lb = LatencyBudget()
        lb.register(Budget(name="order_path", slo_ms=50, stages={"ingest":10,"risk":20,"route":20}))
        with lb.timer("order_path","risk"):
            risky_fn()
    """
    def __init__(self):
        self.sink = _Sink()
        self.budgets: Dict[str, Budget] = {}
        self.stats_total: Dict[str, RollingStats] = {}
        self.stats_stage: Dict[Tuple[str,str], RollingStats] = {}
        self.lock = threading.Lock()

    # ---- registration
    def register(self, budget: Budget) -> None:
        with self.lock:
            self.budgets[budget.name] = budget
            self.stats_total.setdefault(budget.name, RollingStats())
            for st in budget.stages.keys():
                self.stats_stage.setdefault((budget.name, st), RollingStats())

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self.budgets:
            self.budgets[name].enabled = enabled

    # ---- timing API
    class _Timer:
        def __init__(self, parent: 'LatencyBudget', name: str, stage: str):
            self.p = parent; self.name = name; self.stage = stage; self.t0 = None; self.dur = None
        def __enter__(self):
            self.t0 = time.perf_counter_ns(); return self
        def __exit__(self, exc_type, exc, tb):
            t1 = time.perf_counter_ns()
            self.dur = (t1 - (self.t0 or t1)) / 1_000_000.0
            self.p._record(self.name, self.stage, self.dur)

    def timer(self, name: str, stage: str) -> '_Timer':
        # ensure registration for ad-hoc stages
        if name not in self.budgets:
            self.register(Budget(name=name, slo_ms=1000, stages={stage:1000}))
        elif stage not in self.budgets[name].stages:
            self.budgets[name].stages[stage] = max(1, int(self.budgets[name].slo_ms // max(1,len(self.budgets[name].stages)+1)))
            self.stats_stage.setdefault((name, stage), RollingStats())
        return LatencyBudget._Timer(self, name, stage)

    # decorator shortcut
    def timed(self, name: str, stage: str):
        def deco(fn: Callable):
            def wrap(*a, **k):
                with self.timer(name, stage):
                    return fn(*a, **k)
            return wrap
        return deco

    # ---- record + evaluate
    def _record(self, name: str, stage: str, dur_ms: float):
        ts = now_ms()
        with self.lock:
            self.stats_total.setdefault(name, RollingStats()).add(dur_ms)
            self.stats_stage.setdefault((name, stage), RollingStats()).add(dur_ms)

        # emit sample (lightweight)
        self.sink.emit(STREAM_LAT, {
            "ts_ms": ts, "name": name, "stage": stage, "dur_ms": round(dur_ms, 3)
        })

        # check against budget
        bud = self.budgets.get(name)
        if not bud or not bud.enabled: return

        # stage allocation
        alloc = bud.stages.get(stage)
        violated = False
        reason = ""
        if alloc is not None:
            cap = alloc + bud.soft_margin_ms
            if dur_ms > cap:
                violated = True
                reason = f"stage>{cap}ms"
        # best-effort end-to-end mode (p99)
        # when stage == '_end' you can record overall latency explicitly
        if bud.mode.lower() == "max" and stage == "_end":
            cap = bud.slo_ms + bud.soft_margin_ms
            if dur_ms > cap:
                violated = True
                reason = f"e2e>{cap}ms"

        if violated:
            headroom = (alloc if alloc is not None else bud.slo_ms) - dur_ms
            self.sink.emit(STREAM_VIOL, {
                "ts_ms": ts, "name": name, "stage": stage, "dur_ms": round(dur_ms,3),
                "alloc_ms": alloc, "slo_ms": bud.slo_ms, "headroom_ms": round(headroom,3),
                "reason": reason
            })

    # ---- snapshot / summary (for dashboards) --------------------------------
    def snapshot(self, name: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            names = [name] if name else list(self.budgets.keys())
            out = {}
            for n in names:
                bud = self.budgets.get(n)
                if not bud: continue
                tot = self.stats_total.get(n, RollingStats())
                stages = {}
                for st in bud.stages.keys():
                    rs = self.stats_stage.get((n,st), RollingStats())
                    stages[st] = {
                        "p50": rs.p50, "p90": rs.p90, "p95": rs.p95, "p99": rs.p99,
                        "mean": rs.mean, "var": rs.var
                    }
                out[n] = {
                    "budget": asdict(bud),
                    "total": {"p50": tot.p50, "p90": tot.p90, "p95": tot.p95, "p99": tot.p99, "mean": tot.mean, "var": tot.var},
                    "stages": stages
                }
        # also emit to summary stream (best-effort)
        try:
            self.sink.emit(STREAM_SUM, {"ts_ms": now_ms(), "snapshot": out})
        except Exception:
            pass
        return out

# ---------- simple CLI -------------------------------------------------------
def _cli():
    import argparse, random
    ap = argparse.ArgumentParser("latency_budget")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate a pipeline and print snapshots")
    r.add_argument("--name", default="order_path")
    r.add_argument("--slo", type=int, default=50)
    r.add_argument("--stages", default="ingest:10,risk:20,route:20")
    r.add_argument("--iters", type=int, default=200)
    r.add_argument("--print-every", type=int, default=50)

    s = sub.add_parser("snapshot", help="emit one snapshot to stream")
    s.add_argument("--name", default=None)

    args = ap.parse_args()
    lb = LatencyBudget()

    if args.cmd == "run":
        stages = {}
        for pair in args.stages.split(","):
            if not pair.strip(): continue
            k,v = pair.split(":")
            stages[k.strip()] = int(v)
        lb.register(Budget(name=args.name, slo_ms=args.slo, stages=stages))

        for i in range(args.iters):
            for st, alloc in stages.items():
                # simulate with random around allocation
                mu = alloc * 0.8; sd = max(1.0, alloc * 0.2)
                sim = max(0.1, random.gauss(mu, sd))
                with lb.timer(args.name, st):
                    time.sleep(sim/1000.0)
            # optional end-to-end measurement
            total = sum(stages.values())
            with lb.timer(args.name, "_end"):
                time.sleep(max(0.1, random.gauss(total*0.8, total*0.2))/1000.0)

            if (i+1) % args.print_every == 0:
                snap = lb.snapshot(args.name)
                print(json.dumps(snap, indent=2, default=lambda o: round(o,3) if isinstance(o,float) else o))

    elif args.cmd == "snapshot":
        print(json.dumps(lb.snapshot(args.name), indent=2))

if __name__ == "__main__":
    _cli()