# backend/orchestration/utils.py
"""
Utilities for orchestration:
- Config loader with region/broker overlays and ${ENV_VAR} expansion
- Structured logger (console + rotating file)
- Graceful shutdown signals (SIGINT/SIGTERM)
- Cooperative StopFlag you can share across modules
- Heartbeat for liveness checks
- RateLimiter for API call throttling
- retry() helper with exponential backoff
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
import math
import signal
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Callable, Type, Optional

# -----------------------------
# Time helpers
# -----------------------------

def utc_now() -> float:
    """Seconds since epoch (UTC)."""
    return time.time()

def now_ms() -> int:
    return int(time.time() * 1000)

# -----------------------------
# Logger setup
# -----------------------------

def setup_logger(
    name: str = "arb-fund",
    logfile: str = "logs/app.log",
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Create a console + rotating-file logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already set up

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug("Logger initialized")
    return logger

# -----------------------------
# Config loading
# -----------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")

def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR[:default]} in strings within dict/list trees."""
    if isinstance(value, str):
        def repl(m):
            var, default = m.group(1), m.group(2)
            return os.getenv(var, default if default is not None else "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value

def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge overlay into base (overlay wins)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _read_yaml(path: str) -> Dict[str, Any]:
    import yaml  # std in project deps; if missing, add pyyaml to requirements
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data

def load_config(
    base_path: str = "backend/config/base.yaml",
    region: Optional[str] = None,
    broker: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load base config and optionally overlay:
      - backend/config/regions/{region}.yaml
      - backend/config/brokers/{broker}.yaml
    Expands ${ENV} variables in the final dict.
    """
    cfg: Dict[str, Any] = {}
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base config not found: {base_path}")

    cfg = _read_yaml(base_path)

    # Region overlay
    if region:
        rp = os.path.join("backend", "config", "regions", f"{region}.yaml")
        if os.path.exists(rp):
            region_cfg = _read_yaml(rp)
            _deep_update(cfg, region_cfg)

    # Broker overlay
    if broker:
        bp = os.path.join("backend", "config", "brokers", f"{broker}.yaml")
        if os.path.exists(bp):
            broker_cfg = _read_yaml(bp)
            _deep_update(cfg, broker_cfg)

    # Fallbacks
    cfg.setdefault("mode", "paper")
    if broker and "venues" not in cfg:
        cfg["venues"] = {"broker": broker}
    elif "venues" in cfg and broker:
        cfg["venues"]["broker"] = broker

    # Environment expansion
    cfg = _expand_env_vars(cfg)
    return cfg

# -----------------------------
# Graceful shutdown handling
# -----------------------------

class StopFlag:
    """Thread-safe, process-signal-aware stop flag."""
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self) -> bool:
        return self._set

class GracefulShutdown:
    """
    Context manager to wire SIGINT/SIGTERM to a StopFlag.
    Usage:
        stop = StopFlag()
        with GracefulShutdown(stop):
            loop.run(cfg, stop)
    """
    def __init__(self, stop_flag: StopFlag, logger: Optional[logging.Logger] = None):
        self.stop_flag = stop_flag
        self.logger = logger or logging.getLogger("arb-fund")
        self._prev_handlers = {}

    def __enter__(self):
        def handler(signum, frame):
            self.logger.warning(f"Received signal {signum}. Stopping gracefully…")
            self.stop_flag.set()
        # Save old handlers and set new ones
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore previous handlers
        for sig, prev in self._prev_handlers.items():
            signal.signal(sig, prev)
        return False  # don't suppress exceptions

# -----------------------------
# Heartbeat & health checks
# -----------------------------

@dataclass
class Heartbeat:
    name: str
    ttl_seconds: int = 30
    last: float = 0.0

    def beat(self):
        self.last = utc_now()

    def is_stale(self) -> bool:
        if self.last == 0:
            return True
        return (utc_now() - self.last) > self.ttl_seconds

# -----------------------------
# Rate limiter
# -----------------------------

class RateLimiter:
    """
    Token-bucket-ish limiter: allow 'rate' events per 'per_seconds'.
    Call .allow() before an action; returns True if permitted.
    """
    def __init__(self, rate: int, per_seconds: int):
        self.rate = max(1, int(rate))
        self.per = max(1, int(per_seconds))
        self.tokens = self.rate
        self.updated = utc_now()

    def allow(self) -> bool:
        now = utc_now()
        elapsed = now - self.updated
        # Refill
        refill = (elapsed / self.per) * self.rate
        if refill > 0:
            self.tokens = min(self.rate, self.tokens + refill)
            self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

# -----------------------------
# Retry helper (exponential backoff)
# -----------------------------

def retry(
    exceptions: Iterable[Type[BaseException]] = (Exception,),
    tries: int = 5,
    base_delay: float = 0.25,
    max_delay: float = 5.0,
    jitter: float = 0.1,
):
    """
    Decorator: retry a function on exceptions with exponential backoff.
    Example:
        @retry((IOError,), tries=3)
        def fetch():
            ...
    """
    exc_tuple = tuple(exceptions)

    def deco(fn: Callable):
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            while True:
                try:
                    return fn(*args, **kwargs)
                except exc_tuple as e:
                    attempt += 1
                    if attempt >= tries:
                        raise
                    sleep_for = min(max_delay, delay)
                    # add jitter
                    sleep_for += (os.urandom(1)[0] / 255.0 - 0.5) * 2 * jitter
                    sleep_for = max(0.0, sleep_for)
                    time.sleep(sleep_for)
                    delay *= 2
        return wrapper
    return deco

# -----------------------------
# Pretty printers (optional)
# -----------------------------

def pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True, default=str)
    except Exception:
        return str(obj)