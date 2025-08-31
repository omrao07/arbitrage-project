# backend/analytics/correlations.py
from __future__ import annotations

"""
Correlations Toolkit
--------------------
- Compute correlation matrices from returns
- Rolling/pairwise correlations
- Distance matrices and (optional) hierarchical clustering order
- Stability diagnostics to detect correlation regime shifts
- Optional Redis worker to stream matrices to dashboards

Streams (env):
  returns.bars  : {"ts_ms","symbol","ret"}  or
  prices.bars   : {"ts_ms","symbol","close"} (we convert to log-returns)
  corr.matrices : {"ts_ms","window","symbols":[...],"rho":[[...]],"method":"pearson","diag":{...}}

This file is dependency-light: requires numpy; pandas/scipy optional.
"""

import os, json, time, math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional

# ---- deps (graceful) --------------------------------------------------------
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("correlations requires numpy") from e

try:
    import pandas as pd  # optional (for spearman/kendall and convenience)
except Exception:
    pd = None  # type: ignore

HAVE_SCIPY = True
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
except Exception:
    HAVE_SCIPY = False

# ---- optional redis ---------------------------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

# ---- env / streams ----------------------------------------------------------
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RETURNS_STREAM   = os.getenv("RETURNS_STREAM", "returns.bars")
PRICES_BARS      = os.getenv("PRICES_BARS_STREAM", "prices.bars")
CORR_OUT_STREAM  = os.getenv("CORR_OUT_STREAM", "corr.matrices")
MAXLEN           = int(os.getenv("CORR_MAXLEN", "500"))

# -----------------------------------------------------------------------------
# Core math
# -----------------------------------------------------------------------------

def corr_matrix_pearson(returns_by_symbol: Dict[str, Iterable[float]]) -> Tuple[np.ndarray, List[str]]:
    """
    returns_by_symbol: symbol -> iterable of returns aligned in time (same length).
    Returns (rho, symbols) where rho is (N,N).
    """
    keys = [k for k, v in returns_by_symbol.items() if v is not None]
    if not keys:
        return np.zeros((0, 0)), []
    arrays = []
    mlen = min(len(list(returns_by_symbol[k])) for k in keys)
    if mlen == 0:
        return np.zeros((0, 0)), []
    for k in keys:
        a = np.asarray(list(returns_by_symbol[k])[-mlen:], dtype=float)
        arrays.append(a)
    X = np.vstack(arrays)  # (N, T)
    # subtract mean row-wise
    X = X - X.mean(axis=1, keepdims=True)
    denom = np.sqrt((X**2).sum(axis=1, keepdims=True))
    denom = np.where(denom == 0, 1e-18, denom)
    Xn = X / denom
    rho = Xn @ Xn.T
    rho = np.clip(rho, -1.0, 1.0)
    return rho, keys

def corr_matrix_rank(returns_by_symbol: Dict[str, Iterable[float]], method: str = "spearman") -> Tuple[np.ndarray, List[str]]:
    """
    Spearman or Kendall (requires pandas). Falls back to Pearson if pandas missing.
    """
    if pd is None:
        return corr_matrix_pearson(returns_by_symbol)
    keys = [k for k in returns_by_symbol if returns_by_symbol[k] is not None]
    if not keys:
        return np.zeros((0, 0)), []
    mlen = min(len(list(returns_by_symbol[k])) for k in keys)
    if mlen == 0:
        return np.zeros((0, 0)), []
    df = pd.DataFrame({k: list(returns_by_symbol[k])[-mlen:] for k in keys})
    rho = df.corr(method=method).values # type: ignore
    rho = np.clip(rho, -1.0, 1.0)
    return rho, keys

def distance_from_corr(rho: np.ndarray) -> np.ndarray:
    """Mantegna distance: d_ij = sqrt(2*(1 - rho_ij))."""
    rho = np.clip(rho, -1.0, 1.0)
    return np.sqrt(2.0 * (1.0 - rho))

def cluster_order(rho: np.ndarray) -> List[int]:
    """
    Return an ordering of assets to quasi-diagonalize the correlation (clustering).
    Uses SciPy if available; else falls back to sorting by average correlation.
    """
    N = rho.shape[0]
    if N == 0:
        return []
    if HAVE_SCIPY:
        dist = np.sqrt(0.5 * (1 - np.clip(rho, -1.0, 1.0)))
        Z = linkage(squareform(dist, checks=False), method="single")
        dn = dendrogram(Z, no_plot=True)
        return list(map(int, dn["leaves"]))
    # Fallback: order by average correlation ascending (low to high)
    avg = np.mean(rho - np.eye(N), axis=1)
    return list(np.argsort(avg))

def pairwise_rolling_corr(a: Iterable[float], b: Iterable[float], window: int) -> np.ndarray:
    """
    Rolling Pearson correlation of two series.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    n = len(x)
    if n == 0 or len(y) != n or window <= 1:
        return np.array([])
    out = np.full(n, np.nan)
    # cumulative sums for O(1) window stats
    c1 = np.cumsum(x); c2 = np.cumsum(y)
    c1s = np.cumsum(x * x); c2s = np.cumsum(y * y)
    c12 = np.cumsum(x * y)
    for i in range(window - 1, n):
        j = i - window
        sx = c1[i] - (c1[j] if j >= 0 else 0.0)
        sy = c2[i] - (c2[j] if j >= 0 else 0.0)
        sxx = c1s[i] - (c1s[j] if j >= 0 else 0.0)
        syy = c2s[i] - (c2s[j] if j >= 0 else 0.0)
        sxy = c12[i] - (c12[j] if j >= 0 else 0.0)
        vx = sxx - sx * sx / window
        vy = syy - sy * sy / window
        cov = sxy - sx * sy / window
        denom = math.sqrt(max(vx, 0.0) * max(vy, 0.0)) + 1e-18
        out[i] = cov / denom
    return out

def rolling_corr_matrix(returns: Dict[str, Iterable[float]], window: int) -> Tuple[List[np.ndarray], List[str]]:
    """
    Produces a list of correlation matrices aligned with time (NaN until enough data).
    For efficiency, this uses pandas if available; otherwise builds per-pair rolling.
    """
    keys = [k for k in returns]
    if pd is not None:
        # Align into DataFrame
        mlen = min(len(list(returns[k])) for k in keys) if keys else 0
        if mlen == 0:
            return [], keys
        df = pd.DataFrame({k: list(returns[k])[-mlen:] for k in keys})
        # rolling corr returns a Panel-like structure in older pandas; do manual loop
        mats: List[np.ndarray] = []
        for i in range(mlen):
            if i + 1 < window:
                mats.append(np.full((len(keys), len(keys)), np.nan))
                continue
            seg = df.iloc[i+1-window:i+1]
            rho = seg.corr().values
            mats.append(np.clip(rho, -1.0, 1.0))
        return mats, keys

    # Fallback without pandas: compute pairwise for each step (O(N^2 T))
    series = [np.asarray(list(returns[k]), dtype=float) for k in keys]
    T = min(len(s) for s in series) if series else 0
    mats = []
    for t in range(T):
        if t + 1 < window:
            mats.append(np.full((len(keys), len(keys)), np.nan))
            continue
        rho = np.ones((len(keys), len(keys)))
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                r = pairwise_rolling_corr(series[i][:t+1], series[j][:t+1], window)
                val = r[-1]
                rho[i, j] = rho[j, i] = np.clip(val, -1.0, 1.0)
        mats.append(rho)
    return mats, keys

# -----------------------------------------------------------------------------
# Diagnostics / stability
# -----------------------------------------------------------------------------

@dataclass
class CorrDiagnostics:
    ts_ms: int
    symbols: List[str]
    avg_abs_corr: float
    eig_max: float
    eig_min: float
    eig_cond: float
    avg_abs_change: Optional[float] = None  # vs previous rho
    method: str = "pearson"

def corr_diagnostics(rho: np.ndarray, symbols: List[str], prev_rho: Optional[np.ndarray] = None, method: str = "pearson") -> CorrDiagnostics:
    N = rho.shape[0]
    if N == 0:
        return CorrDiagnostics(int(time.time()*1000), [], 0.0, 0.0, 0.0, 0.0, None, method)
    mask = ~np.eye(N, dtype=bool)
    avg_abs = float(np.mean(np.abs(rho[mask])))
    # eigen spread (PSD guard)
    try:
        w = np.linalg.eigvalsh((rho + rho.T) * 0.5)
    except Exception:
        w = np.linalg.eigvals(rho)
    w = np.real(w)
    eig_max = float(np.max(w))
    eig_min = float(np.min(w))
    cond = float(eig_max / (eig_min + 1e-9))
    avg_delta = None
    if prev_rho is not None and prev_rho.shape == rho.shape:
        avg_delta = float(np.mean(np.abs(rho[mask] - prev_rho[mask])))
    return CorrDiagnostics(
        ts_ms=int(time.time()*1000),
        symbols=symbols,
        avg_abs_corr=avg_abs,
        eig_max=eig_max,
        eig_min=eig_min,
        eig_cond=cond,
        avg_abs_change=avg_delta,
        method=method
    )

# -----------------------------------------------------------------------------
# Streaming worker (Redis)
# -----------------------------------------------------------------------------

class CorrWorker:
    """
    Tails returns/prices streams, keeps rolling returns per symbol,
    emits correlation matrices periodically.
    Input messages:
      returns.bars : {"ts_ms","symbol","ret"}
      prices.bars  : {"ts_ms","symbol","close"}
    Output:
      corr.matrices: {"ts_ms","window","symbols":[...],"method":"pearson","rho":[[...]],"diag":{...}}
    """
    def __init__(self, window: int = 120, method: str = "pearson", emit_every: int = 30):
        self.window = int(window)
        self.method = method  # "pearson"|"spearman"|"kendall"
        self.emit_every = int(emit_every)
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.last_returns_id = "$"
        self.last_prices_id = "$"
        self.buff_ret: Dict[str, List[float]] = {}
        self.last_px: Dict[str, float] = {}
        self.ticks = 0
        self.prev_rho: Optional[np.ndarray] = None
        self.prev_symbols: List[str] = []

    async def connect(self):
        if not USE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    def _push_ret(self, sym: str, r: float):
        buf = self.buff_ret.setdefault(sym, [])
        buf.append(float(r))
        if len(buf) > max(self.window * 4, self.window + 10):  # keep some extra
            self.buff_ret[sym] = buf[-max(self.window * 4, self.window + 10):]

    async def run(self):
        await self.connect()
        if not self.r:
            print("[correlations] no redis; worker idle"); return

        while True:
            try:
                resp = await self.r.xread({RETURNS_STREAM: self.last_returns_id, PRICES_BARS: self.last_prices_id}, count=500, block=2000)  # type: ignore
                if not resp:
                    continue
                update = False
                for stream, entries in resp:
                    for _id, fields in entries:
                        j = {}
                        try:
                            j = json.loads(fields.get("json", "{}"))
                        except Exception:
                            continue
                        sym = str(j.get("symbol") or "").upper()
                        ts  = int(j.get("ts_ms") or 0)
                        if not sym: 
                            continue

                        if stream == RETURNS_STREAM:
                            self.last_returns_id = _id
                            r = j.get("ret")
                            if r is not None:
                                self._push_ret(sym, float(r))
                                update = True
                        elif stream == PRICES_BARS:
                            self.last_prices_id = _id
                            close = j.get("close")
                            if close is not None:
                                px = float(close)
                                if px > 0:
                                    if sym in self.last_px and self.last_px[sym] > 0:
                                        r = math.log(px) - math.log(self.last_px[sym])
                                        self._push_ret(sym, r)
                                        update = True
                                    self.last_px[sym] = px

                if update:
                    self.ticks += 1
                    if self.ticks % self.emit_every == 0:
                        await self._emit_snapshot()

            except Exception as e:
                await self._publish({"ts_ms": int(time.time()*1000), "error": str(e)})

    async def _emit_snapshot(self):
        # filter symbols that have enough observations
        ready = {k: v for k, v in self.buff_ret.items() if len(v) >= self.window}
        if len(ready) < 2:
            return
        if self.method == "pearson":
            rho, symbols = corr_matrix_pearson(ready) # type: ignore
        else:
            rho, symbols = corr_matrix_rank(ready, method=("spearman" if self.method != "kendall" else "kendall")) # type: ignore

        order = cluster_order(rho) if rho.size > 0 else []
        if order:
            rho = rho[np.ix_(order, order)]
            symbols = [symbols[i] for i in order]

        diag = asdict(corr_diagnostics(rho, symbols, prev_rho=self.prev_rho if symbols == self.prev_symbols else None, method=self.method))
        self.prev_rho = rho.copy()
        self.prev_symbols = list(symbols)

        await self._publish({
            "ts_ms": int(time.time()*1000),
            "window": self.window,
            "method": self.method,
            "symbols": symbols,
            "rho": rho.round(6).tolist(),
            "diag": diag
        })

    async def _publish(self, obj: Dict):
        if self.r:
            try:
                await self.r.xadd(CORR_OUT_STREAM, {"json": json.dumps(obj)}, maxlen=MAXLEN, approximate=True)  # type: ignore
            except Exception:
                pass
        else:
            print("[correlations]", obj)

# -----------------------------------------------------------------------------
# Quick helpers & demo
# -----------------------------------------------------------------------------

def build_returns_from_prices(prices: Iterable[float]) -> List[float]:
    r = []
    last = None
    for px in prices:
        px = float(px)
        if px <= 0: 
            last = px
            continue
        if last is not None and last > 0:
            r.append(math.log(px) - math.log(last))
        last = px
    return r

def demo():
    rng = np.random.default_rng(7)
    T = 1000
    # Make three correlated series
    base = rng.standard_normal(T) * 0.01
    a = base + 0.005 * rng.standard_normal(T)
    b = 0.7 * base + 0.009 * rng.standard_normal(T)
    c = -0.3 * base + 0.011 * rng.standard_normal(T)
    d = rng.standard_normal(T) * 0.012

    rho, names = corr_matrix_pearson({"AAPL": a, "MSFT": b, "TSLA": c, "META": d})
    print("symbols:", names)
    print("rho[0]:", rho[0].round(3))

    mats, keys = rolling_corr_matrix({"AAPL": a, "MSFT": b, "TSLA": c}, window=60)
    print("rolling mats:", len(mats), "last non-nan mean:", np.nanmean(mats[-1]))

if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser("correlations")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--method", type=str, default="pearson")
    args = ap.parse_args()
    if args.demo:
        demo()
    elif args.worker:
        worker = CorrWorker(window=args.window, method=args.method)
        asyncio.run(worker.run())
    else:
        demo()