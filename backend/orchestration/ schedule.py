# backend/runtime/schedule.py
"""
Lightweight Job Scheduler (stdlib-only)

Features
--------
- Fixed-rate (every N seconds) or cron‑ish expressions ("*/5 * * * *", "0 9 * * 1-5")
- Timezone aware windows via zoneinfo
- Per-job jitter, max_concurrency, and exponential backoff on failure
- Optional trading window gate (open/close hhmm, by tz)
- Simple persistence of next/last run state to a JSON file
- Imperative API or @scheduled decorator

Design notes
------------
- No threads per job: a single background thread (or use run_loop in current thread)
- Job function signature: fn(context) or fn() — context is passed through from run_loop
- Exceptions don't kill the scheduler; they mark the job failed and backoff
"""

from __future__ import annotations

import time
import json
import math
import queue
import random
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import zoneinfo  # Python 3.9+
except Exception:  # pragma: no cover
    zoneinfo = None  # type: ignore

CronField = Union[int, str]  # supports "*/5", "1-5", "1,15,30", "*"

# ----------------------------- Cron parsing -----------------------------

def _expand_field(expr: str, min_v: int, max_v: int) -> List[int]:
    """Expand a cron field into a sorted list of ints."""
    vals: set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if part == "*":
            vals.update(range(min_v, max_v + 1))
            continue
        if part.startswith("*/"):
            step = int(part[2:])
            vals.update(range(min_v, max_v + 1, max(1, step)))
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b: a, b = b, a
            vals.update(range(max(min_v, a), min(max_v, b) + 1))
            continue
        vals.add(int(part))
    out = sorted([v for v in vals if min_v <= v <= max_v])
    return out

@dataclass
class CronSpec:
    """Minimal 5-field cron: minute hour day month dow (0=Sun..6=Sat)."""
    minute: str = "*"
    hour: str = "*"
    day: str = "*"
    month: str = "*"
    dow: str = "*"

    @classmethod
    def parse(cls, expr: str) -> "CronSpec":
        parts = expr.split()
        if len(parts) != 5:
            raise ValueError("cron must have 5 fields: 'm h D M dow'")
        return cls(*parts)  # type: ignore[arg-type]

    def next_after(self, ts: float, *, tz: Optional[str] = None) -> float:
        """
        Compute next fire time (epoch seconds) strictly after ts.
        """
        from datetime import datetime, timedelta

        if zoneinfo and tz:
            z = zoneinfo.ZoneInfo(tz)
        else:
            z = None

        def dt_from_epoch(x: float) -> datetime:
            return datetime.fromtimestamp(x, tz=z)

        def epoch(dt) -> float:
            return dt.timestamp()

        m = _expand_field(self.minute, 0, 59)
        h = _expand_field(self.hour, 0, 23)
        D = _expand_field(self.day, 1, 31)
        M = _expand_field(self.month, 1, 12)
        W = _expand_field(self.dow, 0, 6)

        dt = dt_from_epoch(ts + 60)  # at least next minute
        # normalize to start of minute
        dt = dt.replace(second=0, microsecond=0)
        # brute-force scan with sensible upper bound (2 years)
        lim = dt + timedelta(days=730)
        while dt <= lim:
            if dt.month in M and dt.day in D and dt.weekday() in W and dt.hour in h and dt.minute in m:
                return epoch(dt)
            # advance 1 minute
            dt += timedelta(minutes=1)
        raise RuntimeError("cron.next_after: no match found within 2 years")

# ----------------------------- Job model --------------------------------

@dataclass
class Job:
    name: str
    fn: Callable[..., Any]
    # One of:
    every_sec: Optional[float] = None          # fixed rate
    cron: Optional[CronSpec] = None            # cron schedule
    tz: Optional[str] = None                   # timezone used for cron + windows
    # Optional constraints:
    open_hhmm: Optional[int] = None            # e.g. 930 (09:30) local in tz
    close_hhmm: Optional[int] = None           # e.g. 1600 (16:00) local in tz
    # Behavior:
    jitter_sec: float = 0.0
    backoff_base: float = 2.0                  # exp backoff base
    backoff_max_sec: float = 900.0             # cap
    max_concurrency: int = 1                   # >1 allows overlapping runs
    # State:
    next_ts: float = 0.0
    last_ts: float = 0.0
    last_ok: Optional[bool] = None
    _running: int = 0
    _backoff_sec: float = 0.0

    def due(self, now: float) -> bool:
        return now >= self.next_ts and (self._running < self.max_concurrency)

    def schedule_next(self, now: float, *, success: bool) -> None:
        self.last_ts = now
        self.last_ok = success
        if success:
            self._backoff_sec = 0.0
            self._advance_regular(now)
        else:
            # exponential backoff (adds on top of regular delay/cron)
            self._backoff_sec = min(self.backoff_max_sec, max(1.0, (self._backoff_sec or 1.0) * self.backoff_base))
            base_next = self._regular_next(now)
            self.next_ts = base_next + self._backoff_sec + _jitter(self.jitter_sec)

    def _regular_next(self, now: float) -> float:
        if self.every_sec is not None:
            return now + self.every_sec
        if self.cron is not None:
            return self.cron.next_after(now, tz=self.tz)
        # default: fire in 1 min
        return now + 60.0

    def _advance_regular(self, now: float) -> None:
        base_next = self._regular_next(now)
        j = _jitter(self.jitter_sec)
        self.next_ts = base_next + j

# ----------------------------- Scheduler --------------------------------

@dataclass
class Scheduler:
    persist_path: Optional[str] = None  # where to store last/next runs (JSON)
    tick_sec: float = 0.5
    _jobs: Dict[str, Job] = field(default_factory=dict, init=False)
    _stop: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _ctx: Dict[str, Any] = field(default_factory=dict, init=False)

    # ---- lifecycle ----

    def set_context(self, **ctx) -> None:
        """Context dict passed to job functions as first positional arg if they accept it."""
        self._ctx.update(ctx)

    def add(self, job: Job) -> None:
        if job.name in self._jobs:
            raise ValueError(f"job '{job.name}' already exists")
        if job.next_ts <= 0:
            now = time.time()
            # initial schedule: if cron, compute next; if rate, fire immediately
            if job.cron:
                job.next_ts = job.cron.next_after(now - 60, tz=job.tz)
            else:
                job.next_ts = now + _jitter(job.jitter_sec)
        self._jobs[job.name] = job

    def remove(self, name: str) -> None:
        self._jobs.pop(name, None)

    def jobs(self) -> List[Job]:
        return list(self._jobs.values())

    # ---- persistence ----

    def save(self) -> None:
        if not self.persist_path:
            return
        blob = {name: {"next_ts": j.next_ts, "last_ts": j.last_ts, "last_ok": j.last_ok} for name, j in self._jobs.items()}
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(blob, f, indent=2)
        except Exception:
            pass

    def load(self) -> None:
        if not self.persist_path:
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                blob = json.load(f) or {}
            now = time.time()
            for name, meta in blob.items():
                if name in self._jobs:
                    j = self._jobs[name]
                    j.next_ts = float(meta.get("next_ts", now))
                    j.last_ts = float(meta.get("last_ts", 0.0))
                    j.last_ok = meta.get("last_ok")
        except Exception:
            pass

    # ---- run loop ----

    def start(self, *, daemon: bool = True) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self.run_loop, name="scheduler", daemon=daemon)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        t = self._thread
        if t:
            t.join(timeout=2.0)

    def run_loop(self) -> None:
        while not self._stop:
            now = time.time()
            # scan jobs due
            for j in sorted(self._jobs.values(), key=lambda x: x.next_ts):
                if not j.due(now):
                    continue
                if not _within_window(now, j.tz, j.open_hhmm, j.close_hhmm):
                    # push to next regular slot if outside trading window
                    j._advance_regular(now)
                    continue
                self._launch(j, now)
            time.sleep(self.tick_sec)

    # ---- executor ----

    def _launch(self, job: Job, now: float) -> None:
        def runner():
            job._running += 1
            ok = True
            try:
                _call(job.fn, self._ctx)
            except Exception:
                ok = False
            finally:
                job._running -= 1
                job.schedule_next(time.time(), success=ok)
                self.save()

        # schedule next immediately to prevent stampede for max_concurrency=1
        job.next_ts = now + 10_000  # temporary sentinel
        threading.Thread(target=runner, name=f"job:{job.name}", daemon=True).start()

# ----------------------------- Decorator --------------------------------

def scheduled(
    *,
    name: Optional[str] = None,
    every_sec: Optional[float] = None,
    cron: Optional[str] = None,
    tz: Optional[str] = None,
    jitter_sec: float = 0.0,
    open_hhmm: Optional[int] = None,
    close_hhmm: Optional[int] = None,
    backoff_base: float = 2.0,
    backoff_max_sec: float = 900.0,
    max_concurrency: int = 1,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate a function and attach a Job object as .__job__ for easy registration.
    """
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        j = Job(
            name=name or fn.__name__,
            fn=fn,
            every_sec=every_sec,
            cron=(CronSpec.parse(cron) if cron else None),
            tz=tz,
            jitter_sec=jitter_sec,
            open_hhmm=open_hhmm,
            close_hhmm=close_hhmm,
            backoff_base=backoff_base,
            backoff_max_sec=backoff_max_sec,
            max_concurrency=max_concurrency,
        )
        setattr(fn, "__job__", j)
        return fn
    return wrap

# ----------------------------- Helpers ---------------------------------

def _within_window(now_ts: float, tz: Optional[str], open_hhmm: Optional[int], close_hhmm: Optional[int]) -> bool:
    if open_hhmm is None or close_hhmm is None:
        return True
    if zoneinfo is None:
        # If tz is missing, allow; or interpret in UTC.
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    else:
        from datetime import datetime
        z = zoneinfo.ZoneInfo(tz) if tz else None
        dt = datetime.fromtimestamp(now_ts, tz=z)
    hhmm = dt.hour * 100 + dt.minute
    return open_hhmm <= hhmm <= close_hhmm

def _jitter(j: float) -> float:
    return random.uniform(-j, j) if j and j > 0 else 0.0

def _call(fn: Callable[..., Any], ctx: Dict[str, Any]) -> None:
    # try fn(context) first; else fn()
    try:
        fn(ctx)
    except TypeError:
        fn()

# ----------------------------- Tiny demo --------------------------------

if __name__ == "__main__":
    # Example: print a heartbeat every 2s and a cron at 9:30 NY time on weekdays
    sch = Scheduler(persist_path="scheduler_state.json")

    @scheduled(name="heartbeat", every_sec=2.0, jitter_sec=0.2)
    def heartbeat(ctx):
        print("[hb]", int(time.time()) % 1000, "ctx_keys=", list(ctx.keys()))

    @scheduled(name="ny_open", cron="0 9 * * 1-5", tz="America/New_York", open_hhmm=930, close_hhmm=1600)
    def on_open():
        print("[open] market open tasks")

    # Register decorated jobs
    for fn in (heartbeat, on_open):
        sch.add(getattr(fn, "__job__"))

    # Provide context (e.g., manager, adapters, router)
    sch.set_context(env="demo", version="0.1.0")

    # Start (CTRL+C to stop)
    sch.start(daemon=False)