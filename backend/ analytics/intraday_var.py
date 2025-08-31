# backend/risk/intraday_var.py
from __future__ import annotations

import os, json, time, math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable

# -------- optional deps (all graceful) ---------------------------------------
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd  # only used if present; otherwise we stay in numpy
except Exception:
    pd = None  # type: ignore

# -------- optional redis (graceful) ------------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

# -------- env / streams ------------------------------------------------------
REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PRICES_BARS       = os.getenv("PRICES_BARS_STREAM", "prices.bars")     # {symbol, ts_ms, close, ret?}
PRICES_MARKS      = os.getenv("PRICES_MARKS_STREAM", "prices.marks")   # {symbol, ts_ms, price}
INTRADAY_VAR_OUT  = os.getenv("INTRADAY_VAR_STREAM", "risk.intraday_var")
MAXLEN            = int(os.getenv("INTRADAY_VAR_MAXLEN", "20000"))
BAR_MS_DEFAULT    = int(os.getenv("VAR_BAR_MS", "60000"))  # 1 minute

# -------- small utils --------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def _safe_np_array(x: Iterable[float]) -> "np.ndarray": # type: ignore
    if np is None:
        raise RuntimeError("numpy is required for VaR math")
    return np.asarray(list(x), dtype=float)

def winsorize(a: "np.ndarray", p: float = 0.01) -> "np.ndarray": # type: ignore
    if a.size == 0: return a
    lo, hi = (np.quantile(a, p), np.quantile(a, 1 - p)) # type: ignore
    a = a.copy()
    a[a < lo] = lo
    a[a > hi] = hi
    return a

# -------- models -------------------------------------------------------------
@dataclass
class VarConfig:
    alpha: float = 0.99           # VaR confidence (right-tail quantile of loss)
    lookback: int = 390           # intraday bars to keep (e.g., 1 trading day of 1-min bars ~ 390)
    ewma_lambda: float = 0.94     # RiskMetrics lambda
    use_cornish_fisher: bool = False
    winsor_p: float = 0.0         # e.g., 0.01 to clip extremes
    bar_ms: int = BAR_MS_DEFAULT  # bar size in ms
    scale_to_daily: bool = False  # if True, scale intraday VaR to daily via sqrt(T)
    bars_per_day: int = 390       # used when scale_to_daily=True

@dataclass
class VarSnapshot:
    ts_ms: int
    symbol: Optional[str]           # None for portfolio
    alpha: float
    var_hist: Optional[float]       # positive = loss magnitude
    var_ewma: Optional[float]
    vol_ewma: Optional[float]
    n: int                          # samples used
    method: str = "logret"
    meta: Dict[str, float] = None # type: ignore

# -------- core math ----------------------------------------------------------
def log_returns_from_prices(prices: Iterable[float]) -> "np.ndarray": # type: ignore
    a = _safe_np_array(prices)
    if a.size < 2: return np.array([]) # type: ignore
    r = np.diff(np.log(a)) # type: ignore
    return r

def var_historical_intraday(returns: Iterable[float], alpha: float = 0.99) -> float:
    a = _safe_np_array(returns)
    if a.size == 0: return float("nan")
    L = -a  # losses as positive
    q = float(np.quantile(L, alpha, method="higher" if hasattr(np, "quantile") else None)) # type: ignore
    return q

def _phi(z: float) -> float:
    return 1.0 / math.sqrt(2 * math.pi) * math.exp(-0.5 * z * z)

def _cornish_fisher_z(z: float, skew: float, kurt: float) -> float:
    # excess kurtosis = kurt - 3
    ex = kurt - 3.0
    return (z
            + (1.0/6.0)*(z*z - 1.0)*skew
            + (1.0/24.0)*(z*z*z - 3.0*z)*ex
            - (1.0/36.0)*(2.0*z*z*z - 5.0*z)*(skew*skew))

def var_ewma_parametric(returns: Iterable[float], alpha: float = 0.99, lam: float = 0.94, cornish: bool = False) -> Tuple[float, float]:
    a = _safe_np_array(returns)
    if a.size == 0: return float("nan"), float("nan")
    mu = float(np.mean(a)) # type: ignore
    # EWMA variance
    v = 0.0
    for x in a:
        v = lam * v + (1 - lam) * (x - mu) ** 2
    sigma = math.sqrt(max(v, 0.0) + 1e-18)
    # z score
    try:
        from scipy.stats import norm, skew, kurtosis
        z = float(norm.ppf(alpha))
        if cornish:
            z = _cornish_fisher_z(z, float(skew(a)), float(kurtosis(a, fisher=False)))
    except Exception:
        # simple erfcinv fallback for z; skip CF if scipy not present
        z = math.sqrt(2) * math.erfcinv(2 * (1 - alpha)) if hasattr(math, "erfcinv") else 2.33 # type: ignore
    VaR = max(0.0, -(mu) + z * sigma)  # loss magnitude
    return VaR, sigma

def scale_to_daily(var_intraday: float, bars_per_day: int, used_bars: int) -> float:
    """
    Root-time scaling from used_bars to daily horizon assuming iid.
    """
    if var_intraday != var_intraday:  # NaN check
        return var_intraday
    if used_bars <= 0 or bars_per_day <= 0:
        return var_intraday
    scale = math.sqrt(bars_per_day / max(1, used_bars))
    return float(var_intraday * scale)

# -------- rolling state container -------------------------------------------
class RollingSeries:
    """
    Fixed-length rolling buffer for prices and returns per symbol.
    """
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.prices: List[float] = []
        self.returns: List[float] = []
        self.last_price: Optional[float] = None

    def push_price(self, px: float):
        if px is None or px <= 0: return
        if self.last_price is not None:
            r = math.log(px) - math.log(self.last_price)
            self.returns.append(r)
            if len(self.returns) > self.maxlen:
                self.returns = self.returns[-self.maxlen:]
        self.last_price = px
        self.prices.append(px)
        if len(self.prices) > self.maxlen + 1:
            self.prices = self.prices[-(self.maxlen + 1):]

    def push_return(self, r: float):
        self.returns.append(float(r))
        if len(self.returns) > self.maxlen:
            self.returns = self.returns[-self.maxlen:]

    def window(self) -> "np.ndarray": # type: ignore
        return _safe_np_array(self.returns)

# -------- per-symbol VaR compute --------------------------------------------
def compute_symbol_var(rs: RollingSeries, cfg: VarConfig) -> VarSnapshot:
    x = rs.window()
    n = int(x.size)
    if n == 0:
        return VarSnapshot(ts_ms=now_ms(), symbol=None, alpha=cfg.alpha, var_hist=None, var_ewma=None, vol_ewma=None, n=0, meta={})
    if cfg.winsor_p and cfg.winsor_p > 0:
        x = winsorize(x, p=float(cfg.winsor_p))
    var_hist = var_historical_intraday(x, alpha=cfg.alpha)
    var_ewma, vol_ewma = var_ewma_parametric(x, alpha=cfg.alpha, lam=cfg.ewma_lambda, cornish=cfg.use_cornish_fisher)
    if cfg.scale_to_daily:
        var_hist = scale_to_daily(var_hist, cfg.bars_per_day, n)
        var_ewma = scale_to_daily(var_ewma, cfg.bars_per_day, n)
    return VarSnapshot(
        ts_ms=now_ms(), symbol=None, alpha=cfg.alpha,
        var_hist=float(var_hist) if var_hist == var_hist else None,
        var_ewma=float(var_ewma) if var_ewma == var_ewma else None,
        vol_ewma=float(vol_ewma) if vol_ewma == vol_ewma else None,
        n=n,
        meta={}
    )

# -------- portfolio VaR ------------------------------------------------------
def portfolio_var(
    symbol_returns: Dict[str, Iterable[float]],
    weights: Dict[str, float],
    alpha: float = 0.99,
    method: str = "ewma",         # 'ewma' | 'hist'
    lam: float = 0.94,
    winsor_p: float = 0.0,
    cornish: bool = False,
) -> Tuple[float, float]:
    """
    Returns (VaR, sigma_port) on the intraday horizon of the input returns.
    """
    if np is None: raise RuntimeError("numpy is required")
    keys = [k for k in weights if k in symbol_returns]
    if not keys:
        return float("nan"), float("nan")
    # align by truncating to min length
    arrs = [ _safe_np_array(symbol_returns[k]) for k in keys ]
    m = min(a.size for a in arrs)
    if m == 0: return float("nan"), float("nan")
    R = np.column_stack([ (winsorize(a[-m:], p=winsor_p) if winsor_p>0 else a[-m:]) for a in arrs ])  # (T,N)
    w = np.array([weights[k] for k in keys], dtype=float).reshape(-1, 1)  # (N,1)
    port = R @ w  # (T,1)
    port = port.reshape(-1)
    if method == "hist":
        return var_historical_intraday(port, alpha=alpha), float(np.std(port, ddof=0))
    else:
        v, s = var_ewma_parametric(port, alpha=alpha, lam=lam, cornish=cornish)
        return v, s

# -------- stream worker ------------------------------------------------------
class IntradayVarWorker:
    """
    Tails price/ret streams, keeps rolling windows per symbol, publishes VaR.
    Input message shapes:
      prices.bars : {"ts_ms", "symbol", "close", "ret"?}
      prices.marks: {"ts_ms", "symbol", "price"}
    Output:
      risk.intraday_var : {"ts_ms","symbol","alpha","var_hist","var_ewma","vol_ewma","n"}
    """
    def __init__(self, cfg: Optional[VarConfig] = None):
        self.cfg = cfg or VarConfig()
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.last_id_bars = "$"
        self.last_id_marks = "$"
        self.series: Dict[str, RollingSeries] = {}

    async def connect(self):
        if not USE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    def _series(self, sym: str) -> RollingSeries:
        rs = self.series.get(sym)
        if not rs:
            rs = RollingSeries(self.cfg.lookback)
            self.series[sym] = rs
        return rs

    async def run(self):
        await self.connect()
        if not self.r:
            print("[intraday_var] no redis; worker idle")
            return
        while True:
            try:
                resp = await self.r.xread(
                    {PRICES_BARS: self.last_id_bars, PRICES_MARKS: self.last_id_marks},
                    count=500, block=2000
                )  # type: ignore
                if not resp:
                    continue
                tick_syms = set()
                for stream, entries in resp:
                    for _id, fields in entries:
                        j = {}
                        try:
                            j = json.loads(fields.get("json","{}"))
                        except Exception:
                            continue
                        sym = str(j.get("symbol") or "").upper()
                        if not sym: continue
                        if stream == PRICES_BARS:
                            self.last_id_bars = _id
                            r = j.get("ret")
                            if r is not None:
                                self._series(sym).push_return(float(r))
                            else:
                                close = j.get("close")
                                if close: self._series(sym).push_price(float(close))
                            tick_syms.add(sym)
                        elif stream == PRICES_MARKS:
                            self.last_id_marks = _id
                            px = j.get("price")
                            if px: self._series(sym).push_price(float(px))
                            tick_syms.add(sym)

                for sym in sorted(tick_syms):
                    snap = compute_symbol_var(self._series(sym), self.cfg)
                    snap.symbol = sym
                    await self._publish(snap)

            except Exception as e:
                await self._publish_err({"error": str(e)}) # type: ignore
                await _sleep(0.5) # type: ignore

    async def _publish(self, snap: VarSnapshot):
        if not self.r:
            print("[intraday_var]", asdict(snap))
            return
        try:
            msg = asdict(snap)
            await self.r.xadd(INTRADAY_VAR_OUT, {"json": json.dumps(msg)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

    async def _publish_err(self, obj: Dict[str, float]):
        if not self.r:
            print("[intraday_var][err]", obj)
            return
        try:
            await self.r.xadd(INTRADAY_VAR_OUT, {"json": json.dumps({"ts_ms": now_ms(), **obj})}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

# -------- backtest (coverage of VaR exceptions) ------------------------------
@dataclass
class VarBacktestResult:
    alpha: float
    exceptions: int
    expected: float
    n: int
    p_exceed: float

def backtest_intraday_var(returns: Iterable[float], alpha: float = 0.99) -> VarBacktestResult:
    if np is None: raise RuntimeError("numpy is required")
    r = _safe_np_array(returns)
    if r.size == 0:
        return VarBacktestResult(alpha, 0, 0.0, 0, float("nan"))
    q = var_historical_intraday(r, alpha=alpha)
    L = -r
    exc = int(np.sum(L > q))
    n = int(L.size)
    expected = (1 - alpha) * n
    p = exc / max(1, n)
    return VarBacktestResult(alpha, exc, float(expected), n, float(p))

# -------- CLI / demo ---------------------------------------------------------
def _demo():
    if np is None:
        print("numpy required for demo"); return
    rng = np.random.default_rng(7)
    # simulate 390 1-min returns with a couple of shocks
    r = 0.0001 + 0.0008 * rng.standard_t(df=6, size=390)
    r[100] -= 0.02
    r[250] += 0.015
    cfg = VarConfig(alpha=0.99, lookback=390, ewma_lambda=0.94, use_cornish_fisher=True, winsor_p=0.0)
    rs = RollingSeries(cfg.lookback)
    for x in (100*np.exp(np.cumsum(r))):  # synth price path
        rs.push_price(float(x))
    snap = compute_symbol_var(rs, cfg)
    snap.symbol = "DEMO"
    print("SNAP:", asdict(snap))
    bt = backtest_intraday_var(rs.window(), alpha=cfg.alpha)
    print("BACKTEST:", asdict(bt))

if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser("intraday_var")
    ap.add_argument("--demo", action="store_true", help="run a synthetic path demo")
    ap.add_argument("--worker", action="store_true", help="run the Redis stream worker")
    args = ap.parse_args()
    if args.demo:
        _demo()
    elif args.worker:
        worker = IntradayVarWorker()
        asyncio.run(worker.run())
    else:
        _demo()