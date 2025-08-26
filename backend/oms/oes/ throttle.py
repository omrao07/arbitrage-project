# backend/utils/throttle.py
"""
Throttle / Rate-Limiter
-----------------------
Provides lightweight in-process throttling to avoid
overloading external APIs (e.g., Yahoo Finance, Moneycontrol, broker adapters).

Features
--------
- Token-based throttling: identify by "key" (e.g., "yahoo", "moneycontrol")
- Limits calls per time window (sliding window)
- Thread-safe, works in async or sync
- Decorator + context manager
- Optional sleep/wait, or raise if exceeded

Examples
--------
from backend.utils.throttle import throttle

# Decorator form
@throttle.limit("yahoo", calls=5, per=60)
def fetch_yahoo(...):
    ...

# Manual
if throttle.allow("moneycontrol"):
    call_api()
else:
    print("Throttled!")

# Async
@throttle.limit("ibkr", calls=1, per=2, mode="async")
async def place_order(...):
    ...
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

class _ThrottleBucket:
    def __init__(self, calls: int, per: float):
        self.calls = calls
        self.per = per
        self.lock = threading.Lock()
        self.timestamps: Deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        with self.lock:
            # purge old
            while self.timestamps and self.timestamps[0] <= now - self.per:
                self.timestamps.popleft()
            if len(self.timestamps) < self.calls:
                self.timestamps.append(now)
                return True
            return False

    def wait(self) -> None:
        while not self.allow():
            with self.lock:
                if self.timestamps:
                    sleep_for = (self.timestamps[0] + self.per) - time.monotonic()
                else:
                    sleep_for = self.per
            if sleep_for > 0:
                time.sleep(sleep_for)

    async def wait_async(self) -> None:
        while not self.allow():
            with self.lock:
                if self.timestamps:
                    sleep_for = (self.timestamps[0] + self.per) - time.monotonic()
                else:
                    sleep_for = self.per
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

class ThrottleManager:
    def __init__(self):
        self.buckets: Dict[str, _ThrottleBucket] = {}
        self.lock = threading.Lock()

    def get_bucket(self, key: str, calls: int, per: float) -> _ThrottleBucket:
        with self.lock:
            if key not in self.buckets:
                self.buckets[key] = _ThrottleBucket(calls, per)
            return self.buckets[key]

    def allow(self, key: str, calls: int = 1, per: float = 1.0) -> bool:
        return self.get_bucket(key, calls, per).allow()

    def wait(self, key: str, calls: int = 1, per: float = 1.0) -> None:
        return self.get_bucket(key, calls, per).wait()

    async def wait_async(self, key: str, calls: int = 1, per: float = 1.0) -> None:
        return await self.get_bucket(key, calls, per).wait_async()

    def limit(self, key: str, calls: int = 1, per: float = 1.0, mode: str = "sync"):
        """
        Decorator to throttle function.
        mode = "sync" | "async"
        """
        def decorator(fn: Callable):
            if mode == "sync":
                @functools.wraps(fn)
                def wrapper(*args, **kwargs): # type: ignore
                    self.wait(key, calls, per)
                    return fn(*args, **kwargs)
                return wrapper
            else:
                @functools.wraps(fn)
                async def wrapper(*args, **kwargs):
                    await self.wait_async(key, calls, per)
                    return await fn(*args, **kwargs)
                return wrapper
        return decorator

# Global singleton
throttle = ThrottleManager()

# CLI demo
if __name__ == "__main__":
    import random
    @throttle.limit("demo", calls=3, per=5)
    def work(i): 
        print(f"{time.strftime('%X')} call {i}")
    for i in range(10):
        work(i)