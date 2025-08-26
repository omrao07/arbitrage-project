# backend/utils/rate_linits.py
from __future__ import annotations

import time
import threading
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

# ============================================================
# Simple fixed-gap "rate gate" (like sleep-until-next-slot)
# ============================================================

class RateGate:
    """
    Ensure at most N calls per second (global).
    Example:
        gate = RateGate(rps=5)  # 5 calls/sec max
        gate.wait()             # call before each API hit
    """
    def __init__(self, rps: float):
        self.rps = float(max(0.0, rps))
        self._min_gap = 1.0 / self.rps if self.rps > 0 else 0.0
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._min_gap <= 0:
            return
        with self._lock:
            now = time.time()
            wait = max(0.0, (self._last + self._min_gap) - now)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

# ============================================================
# Token bucket (bursty up to 'burst', average 'rate_per_sec')
# ============================================================

class TokenBucket:
    """
    Classic token bucket limiter.

    rate_per_sec: average token refill (tokens/sec)
    burst: max tokens (bucket capacity)
    """
    def __init__(self, rate_per_sec: float, burst: Optional[int] = None):
        self.rate = float(max(0.0, rate_per_sec))
        self.capacity = float(burst if burst is not None else max(1.0, self.rate))
        self.tokens = self.capacity
        self.timestamp = time.time()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        dt = max(0.0, now - self.timestamp)
        self.tokens = min(self.capacity, self.tokens + dt * self.rate)
        self.timestamp = now

    def consume(self, n: float = 1.0) -> bool:
        """
        Try to consume n tokens; return True if allowed, False otherwise.
        """
        if self.rate <= 0:
            return False
        with self._lock:
            now = time.time()
            self._refill(now)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def wait(self, n: float = 1.0) -> None:
        """
        Block until n tokens are available, then consume them.
        """
        if self.rate <= 0:
            raise RuntimeError("rate_per_sec is 0; cannot proceed")
        while True:
            with self._lock:
                now = time.time()
                self._refill(now)
                if self.tokens >= n:
                    self.tokens -= n
                    return
                # time to next token(s)
                deficit = n - self.tokens
                sleep = max(0.0, deficit / self.rate)
            time.sleep(min(sleep, 0.25))

# ============================================================
# Sliding-window limiter (requests per window per key)
# ============================================================

@dataclass
class WindowRule:
    max_calls: int
    window_sec: float

class SlidingWindowLimiter:
    """
    Enforce 'max_calls' per 'window_sec' for each key (or globally if key=None).

    Example:
        lim = SlidingWindowLimiter(WindowRule(100, 60))  # 100/min
        if lim.allow("zerodha.place_order"):
            ...
    """
    def __init__(self, rule: WindowRule):
        self.rule = rule
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, q: Deque[float], now: float) -> None:
        cutoff = now - self.rule.window_sec
        while q and q[0] < cutoff:
            q.popleft()

    def allow(self, key: str = "global") -> bool:
        now = time.time()
        with self._lock:
            q = self._events[key]
            self._prune(q, now)
            if len(q) < self.rule.max_calls:
                q.append(now)
                return True
            return False

    def wait(self, key: str = "global") -> None:
        """
        Block until an event is allowed, then record it.
        """
        while True:
            now = time.time()
            with self._lock:
                q = self._events[key]
                self._prune(q, now)
                if len(q) < self.rule.max_calls:
                    q.append(now)
                    return
                # time until earliest event exits window
                wait = max(0.0, (q[0] + self.rule.window_sec) - now)
            time.sleep(min(wait, 0.25))

# ============================================================
# Concurrency limiter (semaphore) + decorator/context manager
# ============================================================

class ConcurrencyLimiter:
    """
    Limit concurrent calls (e.g., only 2 overlapping order submissions).
    """
    def __init__(self, max_concurrent: int):
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))

    @contextmanager
    def slot(self):
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()

# ============================================================
# Decorators for easy wrapping of call sites
# ============================================================

def rate_limited_call(fn: Callable, *, rps: float = 0.0) -> Callable:
    """
    Wrap a function with a simple RateGate.
    """
    gate = RateGate(rps=rps)
    def _wrap(*a, **k):
        gate.wait()
        return fn(*a, **k)
    return _wrap

def token_bucket_call(fn: Callable, *, rate_per_sec: float, burst: Optional[int] = None, tokens_per_call: float = 1.0) -> Callable:
    bucket = TokenBucket(rate_per_sec=rate_per_sec, burst=burst)
    def _wrap(*a, **k):
        bucket.wait(tokens_per_call)
        return fn(*a, **k)
    return _wrap

def sliding_window_call(fn: Callable, *, max_calls: int, window_sec: float, key_func: Optional[Callable[..., str]] = None) -> Callable:
    lim = SlidingWindowLimiter(WindowRule(max_calls=max_calls, window_sec=window_sec))
    def _wrap(*a, **k):
        key = key_func(*a, **k) if key_func else "global"
        lim.wait(key)
        return fn(*a, **k)
    return _wrap

# ============================================================
# Example usage notes
# ============================================================

if __name__ == "__main__":
    # 1) Simple fixed RPS
    gate = RateGate(5)  # 5 calls/sec
    for i in range(10):
        gate.wait()
        print("fixed", i, time.time())

    # 2) Token bucket: burst 10, average 2/sec
    bucket = TokenBucket(rate_per_sec=2.0, burst=10)
    for i in range(20):
        bucket.wait()
        print("bucket", i, time.time())

    # 3) Sliding window: 100 calls / 60s per key
    sw = SlidingWindowLimiter(WindowRule(100, 60))
    for i in range(3):
        if sw.allow("api/place"):
            print("allowed", i)