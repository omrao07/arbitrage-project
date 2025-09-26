# utils/profiler.py
"""
Lightweight profiling helpers (stdlib-only)
-------------------------------------------
Tools:
- @timeit / timeblock(): wall-clock timers with optional JSONL logging
- SectionProfiler: aggregate timing across named sections in a run
- MemTracker: quick memory snapshots via tracemalloc
- profile_cpu(): run a callable under cProfile; dump human-readable stats

All utilities are dependency-free and safe to use anywhere in the codebase.
"""

from __future__ import annotations

import atexit
import cProfile
import io
import json
import os
import pstats
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional


# ----------------------------- tiny JSONL logger ------------------------------

class _JSONL:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            atexit.register(self.close)

    def write(self, obj: Dict[str, Any]) -> None:
        if not self._fh:
            return
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            if self._fh:
                self._fh.close()
                self._fh = None
        except Exception:
            pass


# ----------------------------- timer decorator/context ------------------------

def timeit(name: Optional[str] = None, logger_path: Optional[str] = None):
    """
    Decorator: measure wall time of a function and optionally append a JSONL entry.
    Usage:
        @timeit("load_data", logger_path="./logs/profile.jsonl")
        def load_data(...): ...
    """
    log = _JSONL(logger_path) if logger_path else None

    def deco(fn: Callable):
        label = name or fn.__name__

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                if log:
                    log.write({"t": time.time(), "type": "func", "name": label, "ms": round(dt * 1000.0, 3)})
        return wrapper
    return deco


class timeblock:
    """
    Context manager for ad-hoc timing.
    Usage:
        with timeblock("rebalance", logger_path="./logs/profile.jsonl"):
            do_work()
    """
    def __init__(self, name: str, logger_path: Optional[str] = None):
        self.name = name
        self._t0 = 0.0
        self._log = _JSONL(logger_path) if logger_path else None
        self.ms: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        if self._log:
            self._log.write({"t": time.time(), "type": "block", "name": self.name, "ms": round(self.ms, 3)})
        # don't suppress exceptions
        return False


# ----------------------------- aggregated section profiler --------------------

@dataclass
class SectionStats:
    calls: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, ms: float) -> None:
        self.calls += 1
        self.total_ms += ms
        if ms > self.max_ms:
            self.max_ms = ms

    def avg_ms(self) -> float:
        return (self.total_ms / self.calls) if self.calls else 0.0


class SectionProfiler:
    """
    Aggregate timing for named sections; thread-safe.
    Typical use:
        profiler = SectionProfiler("./logs/sections.jsonl")
        with profiler.section("price_update"):
            update_prices()
        print(profiler.summary())
        profiler.dump_summary("./logs/sections_summary.json")
    """
    def __init__(self, logger_path: Optional[str] = None):
        self._stats: Dict[str, SectionStats] = {}
        self._lock = threading.RLock()
        self._log = _JSONL(logger_path) if logger_path else None

    def section(self, name: str) -> "SectionProfiler._Section":
        return SectionProfiler._Section(self, name)

    def _record(self, name: str, ms: float) -> None:
        with self._lock:
            st = self._stats.get(name)
            if not st:
                st = SectionStats(); self._stats[name] = st
            st.add(ms)
        if self._log:
            self._log.write({"t": time.time(), "type": "section", "name": name, "ms": round(ms, 3)})

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                name: {"calls": s.calls, "total_ms": round(s.total_ms, 3),
                       "avg_ms": round(s.avg_ms(), 3), "max_ms": round(s.max_ms, 3)}
                for name, s in sorted(self._stats.items(), key=lambda kv: kv[1].total_ms, reverse=True)
            }

    def dump_summary(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, separators=(",", ":"), indent=2)

    class _Section:
        def __init__(self, parent: "SectionProfiler", name: str):
            self.p = parent; self.name = name; self._t0 = 0.0
        def __enter__(self):
            self._t0 = time.perf_counter(); return self
        def __exit__(self, exc_type, exc, tb):
            ms = (time.perf_counter() - self._t0) * 1000.0
            self.p._record(self.name, ms)
            return False


# ----------------------------- memory tracker ---------------------------------

class MemTracker:
    """
    Simple memory usage snapshots using tracemalloc.
    Usage:
        m = MemTracker(); m.start()
        ... code ...
        snap = m.snapshot("after_work")
        print(snap["cur_MB"], snap["peak_MB"])
        m.stop()
    """
    def __init__(self, top: int = 10):
        self.top = top
        self._running = False

    def start(self) -> None:
        if not self._running:
            tracemalloc.start()
            self._running = True

    def stop(self) -> None:
        if self._running:
            tracemalloc.stop()
            self._running = False

    def snapshot(self, label: str = "") -> Dict[str, Any]:
        if not self._running:
            tracemalloc.start()
            self._running = True
        cur, peak = tracemalloc.get_traced_memory()
        stats = {
            "t": time.time(),
            "label": label,
            "cur_MB": round(cur / (1024 * 1024), 3),
            "peak_MB": round(peak / (1024 * 1024), 3),
        }
        return stats

    def top_stats(self, limit: Optional[int] = None) -> str:
        """
        Returns a human-readable table of top allocating traces.
        """
        limit = limit or self.top
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("lineno")[:limit]
        lines = ["# Top allocations:"]
        for s in stats:
            lines.append(f"{s.count:6d} blocks | {s.size/1024:10.1f} KB | {s.traceback.format()[-1].strip()}")
        return "\n".join(lines)


# ----------------------------- CPU profiler (cProfile) -------------------------

def profile_cpu(fn: Callable, *args, sort_by: str = "cumulative", lines: int = 50,
                out_path: Optional[str] = None, **kwargs) -> str:
    """
    Profile a callable via cProfile and return the stats text.
    If out_path is provided, write the report there (and return the same text).

    Example:
        txt = profile_cpu(run_backtest, dataset, x, sa, sort_by="tottime",
                          lines=30, out_path="./logs/cprofile.txt")
        print(txt)
    """
    pr = cProfile.Profile()
    pr.enable()
    try:
        fn(*args, **kwargs)
    finally:
        pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_by)
    ps.print_stats(lines)
    report = s.getvalue()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


# ----------------------------- quick self-test ---------------------------------

if __name__ == "__main__":
    @timeit("sleepy", logger_path="./logs/profile.jsonl")
    def sleepy(n=3):
        for _ in range(n):
            with timeblock("inner_sleep", logger_path="./logs/profile.jsonl"):
                time.sleep(0.01)

    sp = SectionProfiler("./logs/sections.jsonl")
    with sp.section("setup"):
        time.sleep(0.005)
    sleepy(2)
    with sp.section("compute"):
        sum(i*i for i in range(10000))
    sp.dump_summary("./logs/sections_summary.json")

    m = MemTracker(); m.start()
    arr = [b"x"*1024 for _ in range(2000)]  # ~2MB
    print("Mem snapshot:", m.snapshot("after_alloc"))
    print(m.top_stats(5))
    m.stop()

    print(profile_cpu(time.sleep, 0.02, out_path="./logs/cprofile.txt"))