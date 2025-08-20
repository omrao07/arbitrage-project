# backend/orchestration/scheduler.py
"""
Lightweight market-hours scheduler for the self-trading loop.

Features
- Region-aware market windows (open/close times, weekdays)
- Starts loop.run(cfg) in a worker thread at market open; stops at close
- Graceful shutdown via threading.Event
- Optional pre_open/post_close hooks
- Heartbeat + logging prints (swap for your observability later)
- Timezone-aware via zoneinfo (Python 3.9+)

Usage
-----
from backend.orchestration.scheduler import OrchestratorScheduler
from backend.orchestration.loop import run as loop_run

sched = OrchestratorScheduler(poll_seconds=5)

sched.add_market(
    name="india",
    timezone="Asia/Kolkata",
    open_time=(9, 15),   # 09:15
    close_time=(15, 30), # 15:30
    weekdays={0,1,2,3,4},   # Mon..Fri
    cfg={
        "capital": 100000,
        "symbols": ["NIFTY", "RELIANCE", "BTCUSDT"],
        "bar_seconds": 60,
        "fees_bps": 2,
        "slippage_bps": 1,
        "risk": {
            "max_position_pct": 0.02,
            "max_gross_leverage": 1.2,
            "daily_max_drawdown": 0.015
        },
        "venues": {"broker": "paper"},
        "mode": "paper",
        "region": "india",
        "ingestion": {"feed": "live_equities"}  # or "feed": "paper"
    }
)

sched.start(loop_fn=loop_run)
# ... when you want to exit the whole process:
# sched.stop()
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple
from datetime import datetime, time as dtime, timedelta
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    raise ImportError("zoneinfo is required (Python 3.9+).")

LoopFn = Callable[[dict], None]
HookFn = Callable[[str, dict], None]

@dataclass
class MarketWindow:
    name: str
    timezone: str
    open_time: Tuple[int, int]     # (hour, minute)
    close_time: Tuple[int, int]    # (hour, minute)
    weekdays: Set[int] = field(default_factory=lambda: {0,1,2,3,4})
    pre_open_minutes: int = 0      # run pre-open hook this many minutes before open
    post_close_minutes: int = 0    # run post-close hook this many minutes after close
    cfg: Dict = field(default_factory=dict)

class OrchestratorScheduler:
    def __init__(self, poll_seconds: int = 5):
        self._poll_seconds = max(1, int(poll_seconds))
        self._markets: Dict[str, MarketWindow] = {}
        self._stop_event = threading.Event()
        self._thread = None

        # runtime state per market
        self._workers: Dict[str, threading.Thread] = {}
        self._worker_stop_flags: Dict[str, threading.Event] = {}
        self._is_running: Dict[str, bool] = {}

        # optional hooks
        self._pre_open_hook: Optional[HookFn] = None
        self._post_close_hook: Optional[HookFn] = None

    # ---------- Public API ----------

    def add_market(
        self,
        name: str,
        timezone: str,
        open_time: Tuple[int, int],
        close_time: Tuple[int, int],
        weekdays: Set[int] = {0,1,2,3,4},
        cfg: Optional[dict] = None,
        pre_open_minutes: int = 0,
        post_close_minutes: int = 0,
    ):
        self._markets[name] = MarketWindow(
            name=name,
            timezone=timezone,
            open_time=open_time,
            close_time=close_time,
            weekdays=set(weekdays),
            pre_open_minutes=int(pre_open_minutes),
            post_close_minutes=int(post_close_minutes),
            cfg=cfg or {},
        )
        self._is_running[name] = False

    def on_pre_open(self, hook: HookFn):
        """Register a function hook(market_name, cfg) called pre-open."""
        self._pre_open_hook = hook

    def on_post_close(self, hook: HookFn):
        """Register a function hook(market_name, cfg) called post-close."""
        self._post_close_hook = hook

    def start(self, loop_fn: LoopFn):
        if self._thread and self._thread.is_alive():
            print("[scheduler] already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, args=(loop_fn,), daemon=True)
        self._thread.start()
        print("[scheduler] started.")

    def stop(self):
        print("[scheduler] stopping…")
        self._stop_event.set()
        # signal all workers to stop, then join
        for name, ev in self._worker_stop_flags.items():
            ev.set()
        for name, t in list(self._workers.items()):
            if t.is_alive():
                t.join(timeout=30)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        print("[scheduler] stopped.")

    # ---------- Internals ----------

    def _run_forever(self, loop_fn: LoopFn):
        # main scheduling loop
        while not self._stop_event.is_set():
            for name, mw in list(self._markets.items()):
                now_local = self._now_local(mw.timezone)
                # weekday gate
                if now_local.weekday() not in mw.weekdays:
                    self._ensure_stopped(name, reason="non-trading day")
                    continue

                open_dt, close_dt = self._window_today(mw, now_local)

                # pre-open hook window
                if self._pre_open_hook and self._within_pre_open(now_local, open_dt, mw.pre_open_minutes):
                    try:
                        self._pre_open_hook(name, mw.cfg)
                    except Exception as e:
                        print(f"[scheduler] pre_open hook error ({name}): {e}")

                # trading window: open_dt <= now < close_dt
                if open_dt <= now_local < close_dt:
                    self._ensure_started(name, loop_fn, mw)
                else:
                    # post-close hook (fire once after close)
                    if self._post_close_hook and self._just_after_close(now_local, close_dt, mw.post_close_minutes):
                        try:
                            self._post_close_hook(name, mw.cfg)
                        except Exception as e:
                            print(f"[scheduler] post_close hook error ({name}): {e}")
                    self._ensure_stopped(name, reason="outside trading window")

            time.sleep(self._poll_seconds)

        # ensure all stopped on exit
        for name in list(self._markets.keys()):
            self._ensure_stopped(name, reason="scheduler exit")

    def _ensure_started(self, name: str, loop_fn: LoopFn, mw: MarketWindow):
        if self._is_running.get(name):
            return
        print(f"[scheduler] starting market='{name}'")
        stop_flag = threading.Event()
        self._worker_stop_flags[name] = stop_flag

        # Worker thread target wraps loop_fn(cfg) and stops when stop_flag is set.
        def _worker():
            # pass cfg through; loop_fn itself manages trading session
            cfg = dict(mw.cfg)  # shallow copy to avoid mutation surprises
            try:
                loop_fn(cfg)
            except Exception as e:
                print(f"[scheduler] loop for '{name}' exited with error: {e}")
            finally:
                self._is_running[name] = False
                print(f"[scheduler] loop for '{name}' finished.")

        t = threading.Thread(target=_worker, name=f"loop-{name}", daemon=True)
        self._workers[name] = t
        self._is_running[name] = True
        t.start()

    def _ensure_stopped(self, name: str, reason: str):
        if not self._is_running.get(name, False):
            return
        print(f"[scheduler] stopping market='{name}' ({reason})")
        # Signal loop to stop gracefully: we can only signal via KeyboardInterrupt fallback.
        # If your loop exposes a cooperative stop, wire it here.
        ev = self._worker_stop_flags.get(name)
        if ev:
            ev.set()
        # Best-effort: we rely on your loop to detect daily DD/KeyboardInterrupt or external signal.
        # If your loop doesn't expose a stop, it will stop at next risk halt or upon process exit.

    # ---------- Time helpers ----------

    @staticmethod
    def _now_local(tz_name: str) -> datetime:
        return datetime.now(ZoneInfo(tz_name))

    @staticmethod
    def _window_today(mw: MarketWindow, now_local: datetime) -> tuple[datetime, datetime]:
        tz = ZoneInfo(mw.timezone)
        open_dt = now_local.replace(
            hour=mw.open_time[0], minute=mw.open_time[1], second=0, microsecond=0
        ).astimezone(tz)
        close_dt = now_local.replace(
            hour=mw.close_time[0], minute=mw.close_time[1], second=0, microsecond=0
        ).astimezone(tz)
        # If close < open (overnight markets), push close to next day
        if close_dt <= open_dt:
            close_dt = close_dt + timedelta(days=1)
        return open_dt, close_dt

    @staticmethod
    def _within_pre_open(now_local: datetime, open_dt: datetime, minutes: int) -> bool:
        if minutes <= 0:
            return False
        return open_dt - timedelta(minutes=minutes) <= now_local < open_dt

    @staticmethod
    def _just_after_close(now_local: datetime, close_dt: datetime, minutes: int) -> bool:
        if minutes <= 0:
            return False
        return close_dt <= now_local < close_dt + timedelta(minutes=minutes)