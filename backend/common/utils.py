# backend/common/utils.py
from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib
import inspect
import io # type: ignore
import json
import os
import random
import re
import string
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

T = TypeVar("T")
JSON = Dict[str, Any]

# ----------------------------- Time / IDs -----------------------------

def utc_now_ts() -> float:
    """Unix timestamp (float, UTC)."""
    return time.time()

def utc_now_iso() -> str:
    """ISO8601 UTC string with 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def new_coid(prefix: str = "arb") -> str:
    """
    Globally unique client-order-id: <prefix>-<ms>-<rand8>.
    Idempotent-friendly: you can reuse the same COID for retries.
    """
    ms = int(time.time() * 1000)
    rand = "".join(random.choice(string.hexdigits.lower()) for _ in range(8))
    return f"{prefix}-{ms}-{rand}"

# ----------------------------- ENV / Config ---------------------------

_BOOL_TRUE = {"1", "true", "yes", "on", "y", "t"}
_BOOL_FALSE = {"0", "false", "no", "off", "n", "f"}

def env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    s = v.strip().lower()
    if s in _BOOL_TRUE: return True
    if s in _BOOL_FALSE: return False
    return default

def env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: Any, path: str | Path, *, pretty: bool = True) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
    return str(path)

def load_yaml(path: str | Path) -> Any:
    """
    Lazy YAML loader: imports pyyaml if available.
    Caller can catch ImportError if YAML isn't installed.
    """
    import importlib
    yaml = importlib.import_module("yaml")  # raises if not installed
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ----------------------------- Hashing / Fingerprints -----------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def hash_dict(d: Dict[str, Any], *, sort_keys: bool = True) -> str:
    """Stable hash for dicts (for config fingerprints)."""
    return sha256_text(stable_json_dumps(d, sort_keys=sort_keys))

def stable_json_dumps(obj: Any, *, sort_keys: bool = True) -> str:
    """Canonical JSON string (no spaces, sorted keys)."""
    return json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False, default=str)

# ----------------------------- Retry / Backoff ------------------------

class RetryError(RuntimeError): ...

def retry(
    tries: int = 3,
    *,
    delay: float = 0.1,
    max_delay: float = 2.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function with exponential backoff and jitter.
    Example:
        @retry(tries=5, delay=0.2, exceptions=(IOError,))
        def send(): ...
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrap(*a: Any, **kw: Any) -> T:
            _tries, _delay = tries, delay
            while True:
                try:
                    return fn(*a, **kw)
                except exceptions as e:
                    _tries -= 1
                    if _tries <= 0:
                        raise RetryError(f"{fn.__name__} failed after retries") from e
                    sleep = _delay + random.uniform(-jitter, jitter) * _delay
                    sleep = max(0.0, min(sleep, max_delay))
                    time.sleep(sleep)
                    _delay = min(_delay * backoff, max_delay)
        return wrap
    return deco

# ----------------------------- Rate limiting --------------------------

class TokenBucket:
    """
    Simple token bucket: capacity C, refill R tokens/sec.
    call .allow(n) to consume; returns True if allowed, else False.
    Thread-safe.
    """
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = float(capacity)
        self._ts = time.perf_counter()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._ts
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._ts = now

    def allow(self, n: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

# ----------------------------- Logging / Redaction --------------------

_SECRET_PAT = re.compile(r"(api[_-]?key|secret|token|password|pass|bearer)", re.I)

def redact(value: Any) -> Any:
    """
    Redact likely secrets in strings or dicts (recursive).
    """
    if isinstance(value, dict):
        return {k: ("***" if _SECRET_PAT.search(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        if len(value) > 6 and any(w in value.lower() for w in ("sk_", "pk_", "eyJ")):
            return "***"
    return value

def json_log(**fields: Any) -> str:
    """
    Return a JSON line (string) with automatic timestamp and redaction.
    You write it to a file/console yourself for max portability.
    """
    base = {"ts": utc_now_iso()}
    base.update(redact(fields))
    return stable_json_dumps(base)

# ----------------------------- Math helpers ---------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def moving_avg(xs: Iterable[float], window: int) -> List[float]:
    xs = list(xs); n = len(xs)
    if window <= 0 or window > n: return []
    out: List[float] = []
    s = sum(xs[:window])
    out.append(s / window)
    for i in range(window, n):
        s += xs[i] - xs[i - window]
        out.append(s / window)
    return out

def percentile(xs: Iterable[float], p: float) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    p = clamp(p, 0.0, 1.0)
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    if lo == hi: return xs[lo]
    frac = i - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac

# ----------------------------- Safe import ----------------------------

def safe_import(path: str) -> Any:
    """
    Import module by dotted path; returns module or raises ImportError with context.
    """
    try:
        return importlib.import_module(path)
    except Exception as e:
        raise ImportError(f"Failed to import '{path}': {e}") from e

# ----------------------------- TTL memoization ------------------------

def memoize_ttl(ttl_seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Cache function result for ttl_seconds (per-args). Thread-safe.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        cache: Dict[Tuple[Any, ...], Tuple[float, T]] = {}
        lock = threading.Lock()

        @functools.wraps(fn)
        def wrap(*a: Any, **kw: Any) -> T:
            key = (a, tuple(sorted(kw.items())))
            now = time.time()
            with lock:
                hit = cache.get(key)
                if hit and now - hit[0] <= ttl_seconds:
                    return hit[1]
            res = fn(*a, **kw)
            with lock:
                cache[key] = (now, res)
            return res
        return wrap
    return deco

# ----------------------------- Files / Runs ---------------------------

def timestamped_run_dir(root: str = "runs", *, prefix: str = "run") -> Path:
    """
    Create runs/<prefix>-YYYYmmdd-HHMMSS/, return Path.
    """
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    p = Path(root) / f"{prefix}-{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def atomic_write(path: str | Path, data: bytes) -> str:
    """
    Atomic file write: write to tmp then rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return str(path)

# ----------------------------- Dataclass helpers ----------------------

def dataclass_to_dict(obj: Any, *, drop_none: bool = True) -> Dict[str, Any]:
    if not dataclasses.is_dataclass(obj):
        raise TypeError("obj is not a dataclass")
    d = dataclasses.asdict(obj) # type: ignore
    if drop_none:
        d = {k: v for k, v in d.items() if v is not None}
    return d

# ----------------------------- Timing / Context -----------------------

@contextlib.contextmanager
def timer(scale: float = 1000.0) -> Iterator[float]:
    """
    Context manager that yields elapsed on exit (in ms by default).
    Usage:
        with timer() as t:
            ...
        print(t)  # elapsed ms
    """
    t0 = time.perf_counter()
    elapsed = 0.0
    try:
        yield elapsed
    finally:
        elapsed = (time.perf_counter() - t0) * scale

# ----------------------------- Small net/clock sanity -----------------

def approx_clock_skew(peer_ts: float, *, now_ts: Optional[float] = None) -> float:
    """
    Estimate clock skew vs a peer-provided timestamp (seconds).
    Positive → peer ahead; negative → peer behind.
    """
    now = now_ts if now_ts is not None else time.time()
    return peer_ts - now