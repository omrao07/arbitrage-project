# backend/common/throttle.py
from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ThrottleConfig:
    rate: float       # tokens per second
    burst: int        # max tokens (bucket capacity)


class TokenBucket:
    """
    Simple thread-safe token bucket for rate limiting.
    - refill: tokens added per second (rate)
    - burst: maximum number of tokens
    """

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.capacity = burst
        self.tokens = float(burst)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed <= 0:
            return
        added = elapsed * self.rate
        if added > 0:
            self.tokens = min(self.capacity, self.tokens + added)
            self.last_refill = now

    def allow(self, n: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def wait(self, n: float = 1.0) -> None:
        """Block until at least n tokens are available."""
        while True:
            if self.allow(n):
                return
            time.sleep(1.0 / max(self.rate, 1e-6))


class ThrottleManager:
    """
    Registry of buckets keyed by name (e.g., "yahoo_api", "news_ingest").
    """

    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.configs: Dict[str, ThrottleConfig] = {}

    def register(self, key: str, rate: float, burst: int) -> None:
        self.configs[key] = ThrottleConfig(rate=rate, burst=burst)
        self.buckets[key] = TokenBucket(rate, burst)

    def allow(self, key: str, n: float = 1.0) -> bool:
        if key not in self.buckets:
            raise KeyError(f"No throttle registered for {key}")
        return self.buckets[key].allow(n)

    def wait(self, key: str, n: float = 1.0) -> None:
        if key not in self.buckets:
            raise KeyError(f"No throttle registered for {key}")
        return self.buckets[key].wait(n)


# -------- Decorators --------
def throttle(key: str, n: float = 1.0):
    """
    Decorator for synchronous functions.
    Example:
        @throttle("yahoo_api")
        def fetch():
            ...
    """
    mgr = _GLOBAL_THROTTLE

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            mgr.wait(key, n)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def athrottle(key: str, n: float = 1.0):
    """
    Decorator for async functions.
    Example:
        @athrottle("news_feed", n=2)
        async def get_news():
            ...
    """
    mgr = _GLOBAL_THROTTLE

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            while not mgr.allow(key, n):
                await asyncio.sleep(1.0 / mgr.configs[key].rate)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# -------- Global manager --------
_GLOBAL_THROTTLE = ThrottleManager()


def register_default_throttles():
    """
    Example defaults — tune as needed.
    """
    _GLOBAL_THROTTLE.register("yahoo_api", rate=5, burst=10)         # 5 req/sec, burst 10
    _GLOBAL_THROTTLE.register("moneycontrol_api", rate=2, burst=4)   # 2 req/sec, burst 4
    _GLOBAL_THROTTLE.register("redis_streams", rate=100, burst=200)  # fast path


if __name__ == "__main__":  # pragma: no cover
    register_default_throttles()
    t = _GLOBAL_THROTTLE

    # simple demo
    for i in range(15):
        if t.allow("yahoo_api"):
            print(f"{i}: allowed @ {now_ms()}")
        else:
            print(f"{i}: throttled @ {now_ms()}")
        time.sleep(0.1)