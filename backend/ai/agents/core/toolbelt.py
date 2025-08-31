# backend/ai/agents/core/toolbelt.py
from __future__ import annotations

import base64
import functools
import hashlib
import json
import logging
import os
import random
import string
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar, Union

# ============================================================
# Logging (simple, safe default)
# ============================================================
_LOG_LEVEL = os.getenv("TOOLBELT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s [toolbelt] %(message)s")
log = logging.getLogger("toolbelt")

T = TypeVar("T")
R = TypeVar("R")

# ============================================================
# JSON / dataclass helpers
# ============================================================
def to_json(obj: Any, *, indent: Optional[int] = None) -> str:
    def _default(o):
        if is_dataclass(o):
            return asdict(o) # type: ignore
        try:
            return o.__dict__  # type: ignore
        except Exception:
            return str(o)
    return json.dumps(obj, default=_default, ensure_ascii=False, indent=indent)

def from_json(s: Union[str, bytes]) -> Any:
    return json.loads(s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else s)

# ============================================================
# Env helpers
# ============================================================
def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None: return default
    return v.strip().lower() in ("1","true","yes","y","on")

def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except Exception:
        return default

def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except Exception:
        return default

# ============================================================
# Strings / IDs
# ============================================================
def short_id(prefix: str = "", n: int = 10) -> str:
    s = "".join(random.choices(string.ascii_uppercase + string.digits, k=n))
    return f"{prefix}{s}"

def sha1_hex(data: Union[str, bytes]) -> str:
    if isinstance(data, str): data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

# ============================================================
# Time utilities
# ============================================================
def now_ms() -> int:
    return int(time.time() * 1000)

def sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)

# ============================================================
# Retry / Backoff / Timeout (portable)
# ============================================================
def retry(*, tries: int = 3, backoff_ms: int = 200, jitter_ms: int = 100,
          exceptions: Tuple[type, ...] = (Exception,)) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator: retry a function with exponential backoff + jitter.
    """
    def deco(fn: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs) -> R:
            attempt = 0
            delay = backoff_ms
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e: # type: ignore
                    attempt += 1
                    if attempt > tries:
                        raise
                    j = random.randint(0, jitter_ms)
                    sleep_ms(delay + j)
                    delay = int(delay * 1.7)
        return wrapped
    return deco

class TimeoutError_(TimeoutError):
    pass

def with_timeout(fn: Callable[..., R], timeout_ms: int, *args, **kwargs) -> R:
    """
    Run fn in a thread with a soft timeout (cross-platform). If it doesn't finish, raise TimeoutError_.
    """
    res: Dict[str, Any] = {"done": False, "val": None, "err": None}
    def runner():
        try:
            res["val"] = fn(*args, **kwargs)
        except Exception as e:
            res["err"] = e
        finally:
            res["done"] = True
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    th.join(timeout_ms / 1000.0)
    if not res["done"]:
        raise TimeoutError_(f"operation timed out after {timeout_ms} ms")
    if res["err"] is not None:
        raise res["err"]
    return res["val"]  # type: ignore

# ============================================================
# Circuit breaker
# ============================================================
class CircuitBreaker:
    def __init__(self, *, failure_threshold: int = 5, reset_after_ms: int = 10_000):
        self.failure_threshold = int(failure_threshold)
        self.reset_after_ms = int(reset_after_ms)
        self.failures = 0
        self.opened_at_ms: Optional[int] = None
        self._lock = threading.RLock()

    def call(self, fn: Callable[[], R]) -> R:
        with self._lock:
            if self.is_open():
                raise RuntimeError("circuit open")
        try:
            v = fn()
            self._on_success()
            return v
        except Exception:
            self._on_failure()
            raise

    def is_open(self) -> bool:
        if self.opened_at_ms is None:
            return False
        if now_ms() - self.opened_at_ms >= self.reset_after_ms:
            # half-open
            return False
        return True

    def _on_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at_ms = None

    def _on_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at_ms = now_ms()

# ============================================================
# Rate limiting (token bucket)
# ============================================================
class RateLimiter:
    def __init__(self, *, rate_per_sec: float, burst: float):
        self.rate = float(rate_per_sec)
        self.burst = float(burst)
        self.tokens = burst
        self.updated = time.time()
        self._lock = threading.RLock()

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            dt = max(0.0, now - self.updated)
            self.updated = now
            self.tokens = min(self.burst, self.tokens + dt * self.rate)
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

# ============================================================
# TTL cache / LRU cache (thread-safe)
# ============================================================
class TTLCache(MutableMapping[str, Any]):
    def __init__(self, maxsize: int = 2048, ttl_ms: int = 60_000):
        self.maxsize = maxsize
        self.ttl_ms = ttl_ms
        self._data: OrderedDict[str, Tuple[int, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            ts, v = self._data[key]
            if now_ms() - ts > self.ttl_ms:
                del self._data[key]
                raise KeyError(key)
            self._data.move_to_end(key)
            return v

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                del self._data[key]
            self._data[key] = (now_ms(), value)
            if len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._data.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

# ============================================================
# Simple metrics (in-process counters & timers)
# ============================================================
class Counter:
    def __init__(self): self._v = 0; self._lock = threading.RLock()
    def inc(self, n: int = 1) -> None:
        with self._lock: self._v += n
    def get(self) -> int:
        with self._lock: return self._v

class Timer:
    def __init__(self): self._sum = 0.0; self._n = 0; self._lock = threading.RLock()
    @contextmanager
    def timeit(self):
        t0 = time.time()
        try: yield
        finally:
            dt = time.time() - t0
            with self._lock:
                self._sum += dt; self._n += 1
    def avg(self) -> float:
        with self._lock:
            return (self._sum / self._n) if self._n else 0.0

# ============================================================
# Parallel map (threads, dependency-free)
# ============================================================
def pmap(fn: Callable[[T], R], items: Sequence[T], *, workers: int = 4) -> List[R]:
    items = list(items)
    n = max(1, int(workers))
    out: List[Optional[R]] = [None] * len(items)
    q: Deque[Tuple[int, T]] = deque(enumerate(items))
    lock = threading.RLock()

    def worker():
        while True:
            with lock:
                if not q: return
                i, x = q.popleft()
            try:
                out[i] = fn(x)
            except Exception as e:
                log.warning("pmap worker error: %s", e, exc_info=False)
                out[i] = None  # type: ignore

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    return [o for o in out if o is not None]  # drop failed slots

# ============================================================
# Rolling math
# ============================================================
def ewma(prev: float, x: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    return (1 - alpha) * float(prev) + alpha * float(x)

def pct_change(a: float, b: float) -> float:
    """Return (b/a - 1)."""
    a = float(a)
    return (float(b) / (a if a != 0 else 1e-12)) - 1.0

def to_bps(x: float) -> float:
    return float(x) * 1e4

def from_bps(bps: float) -> float:
    return float(bps) / 1e4

def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    q = max(0.0, min(100.0, q))
    k = (len(v) - 1) * (q / 100.0)
    f = int(k); c = min(f + 1, len(v) - 1)
    if f == c: return v[int(k)]
    d = k - f
    return v[f] + (v[c] - v[f]) * d

# ============================================================
# Throttle / Debounce
# ============================================================
def throttle(min_interval_ms: int) -> Callable[[Callable[..., R]], Callable[..., Optional[R]]]:
    """
    Ensure calls are at least min_interval_ms apart; drop calls that arrive too soon.
    """
    def deco(fn: Callable[..., R]) -> Callable[..., Optional[R]]:
        last = {"t": 0}
        lock = threading.RLock()
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with lock:
                t = now_ms()
                if t - last["t"] < min_interval_ms:
                    return None
                last["t"] = t
            return fn(*args, **kwargs)
        return wrapped
    return deco

def debounce(wait_ms: int) -> Callable[[Callable[..., R]], Callable[..., None]]: # type: ignore
    """
    Delay execution until wait_ms after the last call (trailing edge).
    """
    def deco(fn: Callable[..., R]) -> Callable[..., None]:
        timer = {"t": None}
        lock = threading.RLock()

        def _call(*args, **kwargs):
            fn(*args, **kwargs)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with lock:
                if timer["t"] is not None:
                    timer["t"].cancel()  # type: ignore
                t = threading.Timer(wait_ms / 1000.0, _call, args=args, kwargs=kwargs)
                timer["t"] = t # type: ignore
                t.daemon = True
                t.start()
        return wrapped
    return deco

# ============================================================
# Tiny YAML loader (optional PyYAML; otherwise tolerant INI-ish)
# ============================================================
def load_yaml(path: str) -> Dict[str, Any]:
    """
    Try PyYAML; if missing, fall back to a minimal key: value parser
    (handles flat dicts and simple nested maps by indentation).
    """
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        pass

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, root)]

    for raw in lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        while stack and indent < stack[-1][0]:
            stack.pop()
        cur = stack[-1][1]
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v == "":
                # start of nested map
                d: Dict[str, Any] = {}
                cur[k] = d
                stack.append((indent + 2, d))
            else:
                # try to coerce types
                if v.lower() in ("true","false"):
                    cur[k] = (v.lower() == "true")
                else:
                    try:
                        cur[k] = int(v)
                    except Exception:
                        try:
                            cur[k] = float(v)
                        except Exception:
                            cur[k] = v
    return root

# ============================================================
# Safe import helper (import by dotted path)
# ============================================================
def import_by_path(path: str) -> Any:
    """
    Import a symbol by dotted path, e.g. 'pkg.module:ClassName' or 'pkg.module.func'.
    """
    mod_path, _, attr = path.replace(":", ".").rpartition(".")
    if not mod_path:
        raise ImportError(f"bad path '{path}'")
    mod = __import__(mod_path, fromlist=[attr] if attr else [])
    return getattr(mod, attr) if attr else mod

# ============================================================
# Sliding window stats (for latencies, spreads, etc.)
# ============================================================
class Window:
    def __init__(self, size: int = 512):
        self.size = int(size)
        self.buf: Deque[float] = deque(maxlen=self.size)

    def add(self, x: float) -> None:
        self.buf.append(float(x))

    def mean(self) -> float:
        if not self.buf: return 0.0
        return sum(self.buf) / len(self.buf)

    def std(self) -> float:
        n = len(self.buf)
        if n < 2: return 0.0
        m = self.mean()
        return (sum((x - m) ** 2 for x in self.buf) / (n - 1)) ** 0.5

    def min(self) -> float:
        return min(self.buf) if self.buf else 0.0

    def max(self) -> float:
        return max(self.buf) if self.buf else 0.0

# ============================================================
# Simple span tracer (context manager)
# ============================================================
@contextmanager
def span(name: str, *, attrs: Optional[Mapping[str, Any]] = None):
    t0 = time.time()
    try:
        yield
    finally:
        dt = (time.time() - t0) * 1000.0
        if env_bool("TOOLBELT_TRACE", False):
            log.info("span '%s' %s took=%.1fms", name, to_json(dict(attrs or {})), dt)