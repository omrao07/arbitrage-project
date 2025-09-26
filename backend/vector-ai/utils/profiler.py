 # utils/profiler.py
"""
Lightweight profiling utilities for timing, memory, and hot-spot discovery.

Features
--------
- @profile_it decorator (sync & async) with wall/CPU time + peak memory
- Timer/AsyncTimer context managers for ad-hoc scoped timing
- Profiler session to aggregate many spans and export CSV/JSON
- Optional cProfile runs (per-call or whole block)
- Optional tracemalloc snapshots & top allocations
- Simple sampling profiler (sys.setprofile) for hot functions
- Low overhead (pure stdlib). Optional extras: yappi, line_profiler (if installed)

Usage
-----
from utils.profiler import profile_it, Timer, Profiler

@profile_it(name="load_data")
def load_data(...): ...

async @profile_it(name="handler")
async def handler(...): ...

with Timer("etl.step1"): ...
with Profiler("backtest") as p:
    with p.span("signals.compute"): ...
    with p.span("risk.update"): ...
p.report(top=10)
p.to_json("prof.json"); p.to_csv("prof.csv")

Notes
-----
- Times are wall (perf_counter) and CPU (process_time).
- Memory uses tracemalloc if available; otherwise RSS deltas best-effort.
"""

from __future__ import annotations

import atexit
import contextlib
import csv
import functools
import io
import json
import os
import sys
import time
import typing as _t
from dataclasses import dataclass, asdict, field

# ---------- Optional imports (soft dependencies) ----------
try:
    import tracemalloc  # stdlib but may be disabled in some envs
    _TRACEMALLOC_OK = True
except Exception:
    tracemalloc = None
    _TRACEMALLOC_OK = False

try:
    import resource  # Unix only
    _HAS_RESOURCE = True
except Exception:
    resource = None
    _HAS_RESOURCE = False

# yappi (thread/greenlet profiler) optional
try:
    import yappi  # type: ignore
    _HAS_YAPPI = True
except Exception:
    _HAS_YAPPI = False

# ---------- Helpers ----------

def _now_wall() -> float:
    return time.perf_counter()

def _now_cpu() -> float:
    return time.process_time()

def _rss_bytes() -> int:
    # Best-effort RSS in bytes (Unix)
    if _HAS_RESOURCE:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != "darwin" else 1_000) # type: ignore
    return 0

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024 # type: ignore
    return f"{n:.0f}PB"

# ---------- Data models ----------

@dataclass
class SpanRecord:
    name: str
    wall_ms: float
    cpu_ms: float
    start_ts: float
    end_ts: float
    peak_mem_bytes: int = 0
    delta_mem_bytes: int = 0
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d["wall_ms"] = round(self.wall_ms, 3)
        d["cpu_ms"] = round(self.cpu_ms, 3)
        return d


# ---------- Timer Contexts ----------

class Timer(contextlib.AbstractContextManager):
    """Measure a scoped block. Records wall/cpu and peak mem (tracemalloc)."""

    def __init__(self, name: str, records: list[SpanRecord] | None = None, extra: dict | None = None):
        self.name = name
        self._records = records
        self.extra = extra or {}
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._start_ts = 0.0
        self._rss0 = 0
        self._tm_started = False

    def __enter__(self):
        self._start_ts = time.time()
        self._start_wall = _now_wall()
        self._start_cpu = _now_cpu()
        self._rss0 = _rss_bytes()
        if _TRACEMALLOC_OK and not tracemalloc.is_tracing(): # type: ignore
            # local trace if not already tracing globally
            tracemalloc.start(25) # type: ignore
            self._tm_started = True
        self._snap0 = tracemalloc.take_snapshot() if _TRACEMALLOC_OK else None # type: ignore
        return self

    def __exit__(self, exc_type, exc, tb):
        wall_ms = (_now_wall() - self._start_wall) * 1e3
        cpu_ms = (_now_cpu() - self._start_cpu) * 1e3
        end_ts = time.time()
        peak = 0
        delta = max(_rss_bytes() - self._rss0, 0)

        if _TRACEMALLOC_OK:
            try:
                snap1 = tracemalloc.take_snapshot() # type: ignore
                stats = snap1.compare_to(self._snap0, "lineno") if self._snap0 else []
                peak = max((s.size_diff for s in stats), default=0)
            except Exception:
                peak = 0
            finally:
                if self._tm_started:
                    with contextlib.suppress(Exception):
                        tracemalloc.stop() # type: ignore

        rec = SpanRecord(
            name=self.name,
            wall_ms=wall_ms,
            cpu_ms=cpu_ms,
            start_ts=self._start_ts,
            end_ts=end_ts,
            peak_mem_bytes=peak,
            delta_mem_bytes=delta,
            extra=self.extra,
        )
        if self._records is not None:
            self._records.append(rec)
        return False  # don't suppress exceptions


class AsyncTimer(Timer):
    """Same as Timer, but supports 'async with'."""

    async def __aenter__(self):
        return super().__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return super().__exit__(exc_type, exc, tb)


# ---------- Decorators ----------

def profile_it(name: str | None = None, capture_return: bool = False, with_cprofile: bool = False):
    """
    Decorator for functions / coroutines. Adds .__profile_last__ on the wrapped func.
    """
    def _wrap(fn):
        label = name or fn.__name__

        @functools.wraps(fn)
        def sync_wrapped(*args, **kwargs):
            profiler_io = None
            if with_cprofile:
                import cProfile, pstats
                profiler = cProfile.Profile()
                profiler.enable()

            recs: list[SpanRecord] = []
            with Timer(label, records=recs):
                out = fn(*args, **kwargs)

            if with_cprofile:
                profiler.disable()
                profiler_io = io.StringIO()
                pstats.Stats(profiler, stream=profiler_io).sort_stats("tottime").print_stats(30)

            fn.__profile_last__ = {"span": recs[-1].as_row(), "cprofile": profiler_io.getvalue() if profiler_io else None}
            return (out, fn.__profile_last__) if capture_return else out

        @functools.wraps(fn)
        async def async_wrapped(*args, **kwargs):
            profiler_io = None
            if with_cprofile:
                import cProfile, pstats
                profiler = cProfile.Profile()
                profiler.enable()

            recs: list[SpanRecord] = []
            async with AsyncTimer(label, records=recs):
                out = await fn(*args, **kwargs)

            if with_cprofile:
                profiler.disable()
                profiler_io = io.StringIO()
                pstats.Stats(profiler, stream=profiler_io).sort_stats("tottime").print_stats(30)

            fn.__profile_last__ = {"span": recs[-1].as_row(), "cprofile": profiler_io.getvalue() if profiler_io else None}
            return (out, fn.__profile_last__) if capture_return else out

        if _is_coro(fn):
            return async_wrapped
        return sync_wrapped
    return _wrap

def _is_coro(fn) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


# ---------- Aggregating Profiler ----------

class Profiler(contextlib.AbstractContextManager):
    """
    Aggregates spans into a session. Also provides a simple sampling profiler.

    with Profiler("backtest") as p:
        with p.span("load"): ...
        with p.span("signals"): ...
    p.report()
    """

    def __init__(self, session_name: str = "session", autoprint: bool = False, at_exit: bool = False):
        self.session_name = session_name
        self.records: list[SpanRecord] = []
        self.autoprint = autoprint
        if at_exit:
            atexit.register(self.report)

    # span context
    def span(self, name: str, **extra):
        return Timer(name, records=self.records, extra=extra)

    # sampling profiler (very simple)
    @contextlib.contextmanager
    def sampling(self, interval_sec: float = 0.001):
        """
        Sampling profiler using sys.setprofile (per-call hooks).
        Captures function hit counts during the context.
        """
        import threading
        import collections
        counts = collections.Counter()

        def tracer(frame, event, arg):
            if event != "call":
                return
            code = frame.f_code
            key = (code.co_filename, code.co_name, code.co_firstlineno)
            counts[key] += 1

        old = sys.getprofile()
        sys.setprofile(tracer)
        try:
            yield counts
        finally:
            sys.setprofile(old)
            # attach last sampling result
            self._last_sampling = counts

    # yappi run
    @contextlib.contextmanager
    def yappi_run(self):
        if not _HAS_YAPPI:
            yield None
            return
        yappi.clear_stats()
        yappi.start()
        try:
            yield yappi
        finally:
            yappi.stop()

    # reporting / export
    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.as_row() for r in self.records], f, indent=2)

    def to_csv(self, path: str):
        rows = [r.as_row() for r in self.records]
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def summary(self) -> list[dict]:
        """Aggregate by span name (sum wall/cpu, count, avg)."""
        import collections
        agg = collections.defaultdict(lambda: {"count": 0, "wall_ms_sum": 0.0, "cpu_ms_sum": 0.0,
                                               "peak_mem_bytes_max": 0, "delta_mem_bytes_sum": 0})
        for r in self.records:
            a = agg[r.name]
            a["count"] += 1
            a["wall_ms_sum"] += r.wall_ms
            a["cpu_ms_sum"] += r.cpu_ms
            a["peak_mem_bytes_max"] = max(a["peak_mem_bytes_max"], r.peak_mem_bytes)
            a["delta_mem_bytes_sum"] += r.delta_mem_bytes
        out = []
        for name, a in agg.items():
            avg_wall = a["wall_ms_sum"] / max(a["count"], 1)
            avg_cpu = a["cpu_ms_sum"] / max(a["count"], 1)
            out.append({
                "name": name,
                "count": a["count"],
                "wall_ms_sum": round(a["wall_ms_sum"], 3),
                "cpu_ms_sum": round(a["cpu_ms_sum"], 3),
                "wall_ms_avg": round(avg_wall, 3),
                "cpu_ms_avg": round(avg_cpu, 3),
                "peak_mem_bytes_max": a["peak_mem_bytes_max"],
                "delta_mem_bytes_sum": a["delta_mem_bytes_sum"],
            })
        out.sort(key=lambda x: x["wall_ms_sum"], reverse=True)
        return out

    def report(self, top: int | None = None, file: io.TextIOBase | None = None):
        rows = self.summary()
        if top:
            rows = rows[:top]
        buf = io.StringIO()
        print(f"== Profiler session: {self.session_name} ==", file=buf)
        if not rows:
            print("(no spans recorded)", file=buf)
        else:
            w_name = max((len(r['name']) for r in rows), default=4)
            print(f"{'name'.ljust(w_name)}  count  wall_ms(sum/avg)  cpu_ms(sum/avg)  peak_mem  Δmem(sum)", file=buf)
            for r in rows:
                print(
                    f"{r['name'].ljust(w_name)}  {r['count']:>5}  "
                    f"{r['wall_ms_sum']:>10.3f}/{r['wall_ms_avg']:>7.3f}  "
                    f"{r['cpu_ms_sum']:>10.3f}/{r['cpu_ms_avg']:>7.3f}  "
                    f"{_fmt_bytes(r['peak_mem_bytes_max']).rjust(8)}  "
                    f"{_fmt_bytes(r['delta_mem_bytes_sum']).rjust(9)}",
                    file=buf
                )
        out = buf.getvalue()
        if file is None:
            print(out)
        else:
            file.write(out)
        if self.autoprint:
            print(out)

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: contextlib.TracebackType | None) -> _ExitT_co: # type: ignore
        raise NotImplementedError


# ---------- Convenience one-liners ----------

def profile_block(name: str):
    """Shorthand context for ad-hoc profiling: with profile_block('step'): ..."""
    return Timer(name)

def cprofile_block(sort_by: str = "tottime", lines: int = 30):
    """Context that prints cProfile stats on exit."""
    import cProfile, pstats
    class _C:
        def __enter__(self):
            self.pr = cProfile.Profile(); self.pr.enable(); return self
        def __exit__(self, exc_type, exc, tb):
            self.pr.disable()
            s = io.StringIO()
            pstats.Stats(self.pr, stream=s).sort_stats(sort_by).print_stats(lines)
            print("== cProfile ==")
            print(s.getvalue())
            return False
    return _C()

# ---------- Example (manual run) ----------

if __name__ == "__main__":
    import random

    @profile_it(with_cprofile=False)
    def work(n=20000):
        s = 0
        xs = [random.random() for _ in range(n)]
        for x in xs:
            s += x * x
        return s

    res, prof = work(10000), None
    try:
        _, prof = work(capture_return=True) # type: ignore
    except TypeError:
        pass  # older Python may not support positional after kwonly in decorator call

    p = Profiler("demo")
    with p.span("warmup"):
        time.sleep(0.05)
    with p.span("compute", phase="A"):
        work(5000)
    with p.span("compute", phase="B"):
        work(8000)
    p.report(top=10)