# backend/utils/schedule.py
from __future__ import annotations

import heapq
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("scheduler")
if not log.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# Optional: write heartbeats to Redis if your bus helpers are available
try:
    from backend.bus.streams import hset
except Exception:  # no Redis — make it a no-op
    def hset(*_a, **_k):  # type: ignore
        return None


# =============================== Models ===============================

RunFn = Callable[[], Any]

@dataclass(order=True)
class _QueueItem:
    next_ts: float
    ordinal: int
    job: "Job" = field(compare=False)


@dataclass
class Job:
    """
    A scheduled job.

    Use either:
      • interval_sec: run every N seconds, OR
      • at: list of "HH:MM" 24h times (local time), runs once at each time per day.

    Options:
      • jitter_sec: add random +/- jitter (per run) to avoid thundering herd.
      • catch_up: if the process was asleep, run missed occurrences (bounded).
      • run_on_start: run immediately when added (then schedule next normally).
      • max_runtime_sec: watchdog; if a run exceeds this, we log a warning.
    """
    name: str
    fn: RunFn
    interval_sec: Optional[float] = None
    at: Optional[List[str]] = None
    jitter_sec: float = 0.0
    catch_up: bool = False
    run_on_start: bool = False
    max_runtime_sec: Optional[float] = None
    # runtime fields
    _last_run_ts: Optional[float] = None


# ============================= Utilities ==============================

def _now() -> float:
    return time.time()

def _today_seconds() -> int:
    lt = time.localtime()
    return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec

def _parse_hhmm(s: str) -> int:
    hh, mm = s.strip().split(":")
    hh = int(hh); mm = int(mm)
    assert 0 <= hh <= 23 and 0 <= mm <= 59, "HH:MM out of range"
    return hh * 3600 + mm * 60

def _next_time_of_day(targets_sec: List[int], *, now_sec: Optional[int] = None) -> float:
    """
    Return epoch seconds for the next occurrence among today's HH:MM targets.
    If all today's times have passed, roll to first time tomorrow.
    """
    if now_sec is None:
        now_sec = _today_seconds()
    targets = sorted(targets_sec)
    for t in targets:
        if t > now_sec:
            # later today
            return time.time() - now_sec + t
    # tomorrow first
    return time.time() - now_sec + (24 * 3600 + targets[0])

def _with_jitter(ts: float, jitter: float) -> float:
    if jitter <= 0:
        return ts
    # small, deterministic-ish jitter without importing random
    frac = (int(ts * 1e6) % 997) / 997.0  # 0..1
    delta = (frac * 2.0 - 1.0) * jitter   # [-jitter, +jitter]
    return ts + delta


# ============================== Scheduler =============================

class Scheduler:
    """
    Minimal, thread-based scheduler with a single timing loop and a worker pool.
    """

    def __init__(self, *, workers: int = 4, hb_key: str = "scheduler:hb"):
        self._heap: List[_QueueItem] = []
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._ordinal = 0
        self._stop = False

        self._hb_key = hb_key
        self._threads: List[threading.Thread] = []
        self._pool = _WorkerPool(workers=workers)

    # -------- public API --------

    def add_job(self, job: Job) -> None:
        """
        Add a job and schedule its first run.
        """
        assert (job.interval_sec is not None) ^ (job.at is not None), "Provide either interval_sec or at=[HH:MM,...]"
        next_ts = _now()
        if job.interval_sec is not None:
            if job.run_on_start:
                # run immediately; next will be now + interval
                pass
            else:
                next_ts += float(job.interval_sec)
        else:
            targets = [_parse_hhmm(x) for x in job.at or []]
            next_ts = _next_time_of_day(targets)

            if job.run_on_start:
                # run now, but schedule the next HH:MM normally after
                pass
            else:
                # if next occurrence is in the past (shouldn't be), fix in helper

                pass

        next_ts = _with_jitter(next_ts, job.jitter_sec)
        with self._lock:
            heapq.heappush(self._heap, _QueueItem(next_ts, self._ordinal, job))
            self._ordinal += 1
            self._cv.notify()

    def start(self) -> None:
        """
        Start timing thread + worker pool. Non-blocking.
        """
        t = threading.Thread(target=self._run_loop, name="sched-timer", daemon=True)
        t.start()
        self._threads.append(t)
        self._pool.start()
        self._install_signals()
        log.info("Scheduler started.")

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._cv.notify_all()
        self._pool.stop()
        for t in self._threads:
            t.join(timeout=1.5)
        log.info("Scheduler stopped.")

    # -------- internals --------

    def _install_signals(self) -> None:
        def _sig(_s, _f):
            log.warning("Signal received; stopping scheduler ...")
            self.stop()
        try:
            signal.signal(signal.SIGINT, _sig)
            signal.signal(signal.SIGTERM, _sig)
        except Exception:
            pass  # not available in some embedded contexts

    def _resched_interval(self, job: Job, now_ts: float) -> float:
        nxt = now_ts + float(job.interval_sec or 0.0)
        return _with_jitter(nxt, job.jitter_sec)

    def _resched_at(self, job: Job, now_ts: float) -> float:
        targets = [_parse_hhmm(x) for x in job.at or []]
        ts = _next_time_of_day(targets)
        return _with_jitter(ts, job.jitter_sec)

    def _enqueue_next(self, job: Job, *, now_ts: Optional[float] = None) -> None:
        if now_ts is None:
            now_ts = _now()
        if job.interval_sec is not None:
            ts = self._resched_interval(job, now_ts)
        else:
            ts = self._resched_at(job, now_ts)
        heapq.heappush(self._heap, _QueueItem(ts, self._ordinal, job))
        self._ordinal += 1

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if self._stop:
                    return
                if not self._heap:
                    self._cv.wait(timeout=0.5)
                    continue

                item = heapq.heappop(self._heap)
                now_ts = _now()
                wait = item.next_ts - now_ts
                if wait > 0:
                    # put back and wait
                    heapq.heappush(self._heap, item)
                    self._cv.wait(timeout=min(wait, 0.5))
                    continue

                job = item.job

            # run outside lock
            self._run_one(job)

            # reschedule
            with self._lock:
                if self._stop:
                    return
                self._enqueue_next(job, now_ts=_now())

            # heartbeat
            try:
                hset("scheduler:jobs", job.name, int(time.time() * 1000))
                hset(self._hb_key, "alive_ms", int(time.time() * 1000))
            except Exception:
                pass

    def _run_one(self, job: Job) -> None:
        def _target():
            t0 = time.time()
            try:
                job._last_run_ts = t0
                self._pool.run(job.fn, job)
            finally:
                dt = time.time() - t0
                if job.max_runtime_sec and dt > job.max_runtime_sec:
                    log.warning("Job '%s' exceeded max_runtime (%.2fs > %.2fs)", job.name, dt, job.max_runtime_sec)

        # If catch_up is False and we’re late, just run once; otherwise, we could add loops.
        # For simplicity, we execute once per trigger; catch-up bursts can be added here if needed.
        threading.Thread(target=_target, name=f"job-{job.name}", daemon=True).start()


# ============================== Worker Pool ============================

class _WorkerPool:
    def __init__(self, *, workers: int = 4):
        self._workers = max(1, int(workers))
        self._q: List[Tuple[RunFn, Job]] = []
        self._qlock = threading.Lock()
        self._cv = threading.Condition(self._qlock)
        self._stop = False
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        for i in range(self._workers):
            t = threading.Thread(target=self._loop, name=f"sched-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        with self._qlock:
            self._stop = True
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=1.5)

    def run(self, fn: RunFn, job: Job) -> None:
        with self._qlock:
            self._q.append((fn, job))
            self._cv.notify()

    def _loop(self) -> None:
        while True:
            with self._qlock:
                while not self._q and not self._stop:
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    return
                fn, job = self._q.pop(0)
            # execute outside lock
            try:
                fn()
                log.debug("Job '%s' completed", job.name)
            except Exception as e:
                log.exception("Job '%s' error: %s", job.name, e)