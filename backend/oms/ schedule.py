# backend/utils/schedule.py
"""
Lightweight Scheduler
---------------------
Features
- Interval jobs (`every(seconds=...)`), one-shot `at(ts_ms=...)`, and simple cron (`cron("*/5 * * * *")`)
- Optional market-hour windows (US/NYSE or IN/NSE) & weekday filters
- Jitter to de-correlate tasks
- Redis-based best-effort distributed lock (optional)
- Threaded runner with graceful start/stop

Usage
-----
from backend.utils.schedule import Scheduler, JobSpec, market_window

sched = Scheduler()  # or Scheduler(redis_lock_key="sched:lock")
def pull_news(): ...
def rebalance(): ...

# every 60s between 09:15–15:30 IST on NSE days
sched.every(60, fn=pull_news,
            only_if=market_window("IN", (9,15), (15,30)))

# cron: every 5 minutes on weekdays 9:30–16:00 ET
sched.cron("*/5 * * * 1-5", fn=rebalance,
           only_if=market_window("US", (9,30), (16,0)))

# one-shot
sched.at(ts_ms=int(time.time()*1000)+10_000, fn=lambda: print("hi in 10s"))

sched.start()
...
sched.stop()
"""

from __future__ import annotations

import dataclasses
import os
import random
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- optional Redis for distributed lock ----
try:
    import redis  # pip install redis (optional)
except Exception:  # pragma: no cover
    redis = None  # type: ignore

# ---- Small utils ----
def _utc_ms() -> int:
    return int(time.time() * 1000)

def _now_tuple(tz_offset_minutes: int = 0) -> Tuple[int, int, int, int, int, int, int]:
    """
    Return (year, mon, day, hour, minute, second, weekday) for tz with fixed offset.
    weekday: 0=Mon ... 6=Sun
    """
    ts = time.time() + tz_offset_minutes * 60
    lt = time.gmtime(ts)  # do math in UTC then shift by offset
    # Recompute weekday with shift semantics already baked by shifted ts
    return lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec, (lt.tm_wday)

def _minutes(hhmm: Tuple[int, int]) -> int:
    h, m = hhmm
    return h * 60 + m

def _is_weekday(weekday: int) -> bool:
    return 0 <= weekday <= 4  # Mon–Fri

# ---- Market calendars (coarse; add holiday lists if you want) ----
# Fixed offsets; if you need DST-perfect US hours, pass a custom predicate.
TZ_OFFSETS = {  # minutes offset from UTC
    "US": -300,  # ET standard (approx; DST not handled)
    "IN": 330,   # IST
}
MARKET_HOURS = {
    "US": ((9, 30), (16, 0)),  # NYSE
    "IN": ((9, 15), (15, 30)), # NSE/BSE
}

def market_window(
    region: str,
    open_hhmm: Tuple[int, int] | None = None,
    close_hhmm: Tuple[int, int] | None = None,
    *,
    weekdays_only: bool = True,
    extra_predicate: Optional[Callable[[], bool]] = None,
):
    """
    Returns a predicate callable → bool that’s True when inside the market window.
    Note: coarse; ignores holidays and US DST by default.
    """
    region = (region or "US").upper()
    tz = TZ_OFFSETS.get(region, 0)
    open_hhmm = open_hhmm or MARKET_HOURS.get(region, ((0,0),(23,59)))[0]
    close_hhmm = close_hhmm or MARKET_HOURS.get(region, ((0,0),(23,59)))[1]

    def _pred() -> bool:
        y, mo, d, H, M, S, W = _now_tuple(tz)
        if weekdays_only and not _is_weekday(W):
            return False
        minutes = H * 60 + M
        inside = _minutes(open_hhmm) <= minutes <= _minutes(close_hhmm)
        if not inside:
            return False
        if extra_predicate:
            try:
                return bool(extra_predicate())
            except Exception:
                return False
        return True
    return _pred

# ---- Cron parsing (subset) ----
def _parse_cron(expr: str):
    """
    Parse 'm h dom mon dow' with ranges and steps (*, */n, A-B, A-B/n, lists).
    Returns dict with callables test_minute / test_hour / test_dom / test_mon / test_dow.
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        raise ValueError("cron format must be 'm h dom mon dow'")

    def parse_part(part: str, lo: int, hi: int):
        vals: set[int] = set()
        for token in part.split(","):
            token = token.strip()
            if token == "*":
                vals.update(range(lo, hi + 1))
                continue
            # step
            if token.startswith("*/"):
                step = int(token[2:])
                vals.update(range(lo, hi + 1, step))
                continue
            # range or single or range/step
            if "/" in token:
                base, step_s = token.split("/", 1)
                step = int(step_s)
            else:
                base, step = token, 1
            if "-" in base:
                a, b = base.split("-", 1)
                a, b = int(a), int(b)
                vals.update(range(a, b + 1, step))
            else:
                vals.add(int(base))
        return vals

    minute = parse_part(fields[0], 0, 59)
    hour   = parse_part(fields[1], 0, 23)
    dom    = parse_part(fields[2], 1, 31)
    mon    = parse_part(fields[3], 1, 12)
    dow    = parse_part(fields[4], 0, 6)  # 0=Mon..6=Sun

    return minute, hour, dom, mon, dow

# ---- Data model ----
Callback = Callable[[], Any]
Predicate = Callable[[], bool]

@dataclass
class JobSpec:
    id: str
    kind: str                     # 'interval' | 'cron' | 'at'
    fn: Callback
    every_s: Optional[float] = None
    cron_expr: Optional[str] = None
    at_ts_ms: Optional[int] = None
    jitter_s: float = 0.0
    only_if: Optional[Predicate] = None
    last_run_ms: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# ---- Scheduler ----
class Scheduler:
    def __init__(
        self,
        *,
        poll_ms: int = 500,
        redis_lock_key: Optional[str] = None,
        redis_url: Optional[str] = None,
        lock_ttl_s: int = 5,
    ):
        """
        If redis_lock_key is provided and redis is available, the scheduler
        will acquire a simple NX lock per tick (best-effort) to avoid
        multi-process duplication.
        """
        self.poll_ms = int(poll_ms)
        self.jobs: Dict[str, JobSpec] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.redis_lock_key = redis_lock_key
        self.lock_ttl_s = lock_ttl_s
        self._r = None
        if redis_lock_key and redis is not None:
            try:
                url = redis_url or os.getenv("REDIS_URL")
                if url:
                    self._r = redis.from_url(url, decode_responses=True)
                else:
                    host = os.getenv("REDIS_HOST", "localhost")
                    port = int(os.getenv("REDIS_PORT", "6379"))
                    self._r = redis.Redis(host=host, port=port, decode_responses=True)
            except Exception:
                self._r = None

    # ---- Job APIs ----
    def every(
        self, seconds: float, *, fn: Callback, jitter_s: float = 0.0,
        only_if: Optional[Predicate] = None, job_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
    ) -> str:
        jid = job_id or f"every:{int(seconds)}:{len(self.jobs)+1}"
        with self._lock:
            self.jobs[jid] = JobSpec(
                id=jid, kind="interval", fn=fn, every_s=float(seconds),
                jitter_s=float(jitter_s), only_if=only_if, meta=meta or {}
            )
        return jid

    def cron(
        self, expr: str, *, fn: Callback, jitter_s: float = 0.0,
        only_if: Optional[Predicate] = None, job_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None
    ) -> str:
        # Validate at registration
        _parse_cron(expr)
        jid = job_id or f"cron:{expr}:{len(self.jobs)+1}"
        with self._lock:
            self.jobs[jid] = JobSpec(
                id=jid, kind="cron", fn=fn, cron_expr=expr,
                jitter_s=float(jitter_s), only_if=only_if, meta=meta or {}
            )
        return jid

    def at(self, ts_ms: int, *, fn: Callback, job_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        jid = job_id or f"at:{ts_ms}"
        with self._lock:
            self.jobs[jid] = JobSpec(id=jid, kind="at", fn=fn, at_ts_ms=int(ts_ms), meta=meta or {})
        return jid

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            return self.jobs.pop(job_id, None) is not None

    def list_jobs(self) -> List[JobSpec]:
        with self._lock:
            return list(self.jobs.values())

    # ---- Runner ----
    def _tick(self):
        now = _utc_ms()
        # Redis NX lock (optional, best-effort)
        if self._r and self.redis_lock_key:
            try:
                ok = self._r.set(self.redis_lock_key, str(now), nx=True, ex=self.lock_ttl_s)
                if not ok:
                    return  # another instance holds the lock
            except Exception:
                pass

        to_run: List[Tuple[str, Callback, float]] = []  # (job_id, fn, delay_s_for_jitter)
        with self._lock:
            for jid, job in self.jobs.items():
                try:
                    if job.kind == "interval":
                        if job.last_run_ms is None or (now - job.last_run_ms) >= int(job.every_s * 1000): # type: ignore
                            if not job.only_if or job.only_if():
                                delay = random.uniform(0, job.jitter_s) if job.jitter_s > 0 else 0.0
                                to_run.append((jid, job.fn, delay))
                                job.last_run_ms = now
                    elif job.kind == "cron":
                        minute, hour, dom, mon, dow = _parse_cron(job.cron_expr or "* * * * *")
                        # Evaluate in local (system) time
                        lt = time.localtime(now / 1000)
                        # Our DOW is 0=Mon..6=Sun; time.localtime: 0=Mon..6=Sun => aligned
                        cond = (lt.tm_min in minute and lt.tm_hour in hour and
                                lt.tm_mday in dom and (lt.tm_mon) in mon and lt.tm_wday in dow)
                        # fire once per matching minute
                        if cond:
                            last = job.last_run_ms or 0
                            if (now - last) >= 60_000:
                                if not job.only_if or job.only_if():
                                    delay = random.uniform(0, job.jitter_s) if job.jitter_s > 0 else 0.0
                                    to_run.append((jid, job.fn, delay))
                                    job.last_run_ms = now
                    elif job.kind == "at":
                        if now >= (job.at_ts_ms or now):
                            to_run.append((jid, job.fn, 0.0))
                except Exception:
                    # keep scheduler robust
                    pass

        # Execute outside the lock
        for jid, fn, delay in to_run:
            if delay > 0:
                time.sleep(delay)
            try:
                fn()
            except Exception:
                # swallow exceptions to keep scheduler running
                pass
            # remove one-shot
            if jid.startswith("at:"):
                with self._lock:
                    self.jobs.pop(jid, None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                self._tick()
                time.sleep(self.poll_ms / 1000.0)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

# ---- CLI quick test ----
if __name__ == "__main__":
    s = Scheduler(redis_lock_key=os.getenv("SCHED_LOCK"))
    def hello():
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} hello")
    s.every(2, fn=hello, jitter_s=0.5)
    s.cron("*/1 * * * 1-5", fn=lambda: print("cron minute weekday"),
           only_if=market_window("IN", (9,15), (15,30)))
    s.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.stop()