# utils/helpers.py
"""
General-purpose helper utilities
--------------------------------
Common functions used across agents, selectors, simulators, and strategies.

Features:
- Safe math ops: pct_change, zscore, clip, rolling_mean/std
- Date/time helpers: utc_now, parse_time, format_ts
- Stats: sharpe, sortino, max_drawdown
- Data transforms: project_simplex, normalize_dict
- Misc: chunks, flatten

Keep light; stdlib-only.
"""

from __future__ import annotations

import math
import statistics
import time
import datetime
from typing import Dict, Iterable, List, Tuple, Any, Generator


# --------------------------- math helpers ---------------------------

def pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def zscore(x: List[float]) -> float:
    if not x:
        return 0.0
    mu = statistics.mean(x)
    sd = statistics.pstdev(x) if len(x) > 1 else 0.0
    return safe_div(x[-1] - mu, sd, 0.0)

def rolling_mean(x: List[float], n: int) -> float:
    if len(x) < n:
        return 0.0
    return sum(x[-n:]) / n

def rolling_std(x: List[float], n: int) -> float:
    if len(x) < n:
        return 0.0
    mu = rolling_mean(x, n)
    var = sum((xi - mu)**2 for xi in x[-n:]) / max(1, (n-1))
    return math.sqrt(var)

# --------------------------- stats ---------------------------

def sharpe(returns: List[float], rf: float = 0.0, ann: int = 252) -> float:
    if not returns:
        return 0.0
    mu = statistics.mean(returns) - rf / ann
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return (mu / sd) * math.sqrt(ann) if sd else 0.0

def sortino(returns: List[float], rf: float = 0.0, ann: int = 252) -> float:
    if not returns:
        return 0.0
    excess = [r - rf / ann for r in returns]
    downside = [min(0.0, r) for r in excess]
    dd = math.sqrt(sum(d*d for d in downside) / max(1, len(downside)))
    mu = statistics.mean(excess)
    return (mu / dd) * math.sqrt(ann) if dd else 0.0

def max_drawdown(series: List[float]) -> Tuple[float,int,int]:
    peak = -float("inf")
    mdd = 0.0
    start = end = 0
    peak_idx = 0
    for i, v in enumerate(series):
        if v > peak:
            peak = v; peak_idx = i
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd; start = peak_idx; end = i
    return mdd, start, end

# --------------------------- time utils ---------------------------

def utc_now() -> float:
    return time.time()

def format_ts(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.datetime.utcfromtimestamp(ts).strftime(fmt)

def parse_time(s: str, fmt: str = "%H:%M") -> Tuple[int,int]:
    """
    Parse "HH:MM" -> (hour, minute).
    """
    h, m = s.split(":")
    return int(h), int(m)

# --------------------------- transforms ---------------------------

def project_simplex(v: List[float], budget: float = 1.0) -> List[float]:
    """
    Project onto { w >= 0, sum w = budget }.
    """
    n = len(v)
    if n == 0:
        return []
    if budget <= 0:
        return [0.0]*n
    u = sorted(v, reverse=True)
    css = 0.0; rho = -1
    for i, ui in enumerate(u, start=1):
        css += ui
        t = (css - budget) / i
        if ui - t > 0:
            rho = i
    theta = (sum(u[:rho]) - budget) / max(1, rho)
    return [max(0.0, x - theta) for x in v]

def normalize_dict(d: Dict[Any,float], scale: float = 1.0) -> Dict[Any,float]:
    s = sum(abs(v) for v in d.values())
    if s <= 0: 
        return {k:0.0 for k in d}
    return {k:(v/s)*scale for k,v in d.items()}

# --------------------------- misc ---------------------------

def chunks(lst: List[Any], n: int) -> Generator[List[Any],None,None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def flatten(lst: Iterable[Iterable[Any]]) -> List[Any]:
    return [x for sub in lst for x in sub]