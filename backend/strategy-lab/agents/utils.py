# agents/utils.py
"""
agents/utils.py
---------------
General-purpose helpers shared across agents.

Features
--------
- EnumJSONEncoder: safe JSON dumps for dataclasses/enums.
- Logger: simple leveled logger (stdout print, JSON optional).
- math/stat helpers: safe_div, pct_change, drawdown, sharpe, ir.
- time helpers: utcnow_ts, sleep_secs.
- ID helpers: short_uid.
- Simple moving average & rolling stats (std, mean).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
import uuid
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------- JSON encoder -----------------------------------

class EnumJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.name
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)


def json_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, cls=EnumJSONEncoder, separators=(",", ":"), **kwargs)


# ----------------------------- Logger ----------------------------------------

class Logger:
    def __init__(self, name: str = "agent", json_mode: bool = False):
        self.name = name
        self.json_mode = json_mode

    def _emit(self, level: str, msg: str, payload: Optional[Dict] = None) -> None:
        ts = utcnow_ts()
        if self.json_mode:
            rec = {"t": ts, "name": self.name, "level": level, "msg": msg}
            if payload:
                rec.update(payload)
            sys.stdout.write(json_dumps(rec) + "\n")
        else:
            prefix = f"[{ts:.3f}] {self.name} {level}: {msg}"
            if payload:
                prefix += " " + str(payload)
            sys.stdout.write(prefix + "\n")
        sys.stdout.flush()

    def info(self, msg: str, payload: Optional[Dict] = None) -> None:
        self._emit("INFO", msg, payload)

    def warn(self, msg: str, payload: Optional[Dict] = None) -> None:
        self._emit("WARN", msg, payload)

    def error(self, msg: str, payload: Optional[Dict] = None) -> None:
        self._emit("ERROR", msg, payload)


# ----------------------------- Math / stats ----------------------------------

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def sharpe(returns: List[float], risk_free: float = 0.0, ann: int = 252) -> float:
    if not returns:
        return 0.0
    mean = statistics.mean(returns) - (risk_free / ann)
    std = statistics.pstdev(returns)
    daily_sr = safe_div(mean, std)
    return daily_sr * math.sqrt(ann)

def information_ratio(series: List[float], ann: int = 252) -> float:
    if not series:
        return 0.0
    mu = statistics.mean(series)
    sigma = statistics.pstdev(series)
    return safe_div(mu, sigma) * math.sqrt(ann)

def max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    peak = -float("inf")
    mdd = 0.0
    start = end = 0
    for i, v in enumerate(equity_curve):
        if v > peak:
            peak = v
            start = i
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
            end = i
    return (mdd, start, end)


# ----------------------------- Time helpers ----------------------------------

def utcnow_ts() -> float:
    """UTC timestamp in seconds."""
    return time.time()

def sleep_secs(sec: float) -> None:
    time.sleep(sec)


# ----------------------------- ID helpers ------------------------------------

def short_uid(k: int = 8) -> str:
    return uuid.uuid4().hex[:k]


# ----------------------------- Rolling stats ---------------------------------

class SMA:
    """Simple moving average."""
    def __init__(self, period: int):
        self.period = max(1, period)
        self.buf: List[float] = []

    def update(self, x: float) -> float:
        self.buf.append(x)
        if len(self.buf) > self.period:
            self.buf.pop(0)
        return self.value()

    def value(self) -> float:
        if not self.buf:
            return 0.0
        return sum(self.buf) / len(self.buf)


class RollingStats:
    """Rolling mean/std (Welford)."""
    def __init__(self, window: int):
        self.window = max(1, window)
        self.buf: List[float] = []

    def update(self, x: float) -> Tuple[float, float]:
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        mu = statistics.mean(self.buf)
        sigma = statistics.pstdev(self.buf) if len(self.buf) > 1 else 0.0
        return mu, sigma