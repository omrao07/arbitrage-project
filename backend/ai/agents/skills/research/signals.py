# backend/alpha/signals.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Protocol

# ---------------------------------------------------------------------
# Core interfaces
# ---------------------------------------------------------------------

class Signal(Protocol):
    """
    Minimal signal interface: stateless or stateful.
    Implement `update(symbol, price, ts_ms, **kwargs)` and return a float in [-1, 1] or None.
    """
    def update(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Optional[float]: ...

@dataclass
class SignalResult:
    symbol: str
    name: str
    ts_ms: int
    value: Optional[float]
    extras: Dict[str, Any] = field(default_factory=dict)

    def clipped(self, lo: float=-1.0, hi: float=1.0) -> "SignalResult":
        if self.value is None:
            return self
        v = max(lo, min(hi, float(self.value)))
        return SignalResult(self.symbol, self.name, self.ts_ms, v, dict(self.extras))

# ---------------------------------------------------------------------
# Utilities (rolling ops with no third-party deps)
# ---------------------------------------------------------------------

class Ring:
    """Fixed-size ring buffer for O(1) push / rolling sums."""
    __slots__ = ("buf","cap","i","n","sum","sumsq")
    def __init__(self, cap: int):
        self.buf: List[float] = [0.0]*max(1,cap)
        self.cap = max(1,cap)
        self.i = 0
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
    def push(self, x: float) -> None:
        old = self.buf[self.i]
        self.buf[self.i] = x
        self.i = (self.i + 1) % self.cap
        if self.n < self.cap:
            self.n += 1
        else:
            self.sum  -= old
            self.sumsq-= old*old
        self.sum  += x
        self.sumsq+= x*x
    def mean(self) -> Optional[float]:
        return None if self.n == 0 else self.sum / self.n
    def var(self) -> Optional[float]:
        if self.n == 0: return None
        m = self.mean() or 0.0
        return max(0.0, self.sumsq/self.n - m*m)
    def stdev(self) -> Optional[float]:
        v = self.var()
        return None if v is None else math.sqrt(v)
    def full(self) -> bool:
        return self.n >= self.cap
    def values(self) -> List[float]:
        # return logical order (oldest->newest)
        if self.n == 0: return []
        start = (self.i - self.n) % self.cap
        out = []
        for k in range(self.n):
            out.append(self.buf[(start + k) % self.cap])
        return out

def zscore(x: float, mean: float, stdev: float, clamp: float = 3.5) -> float:
    if stdev <= 1e-12:
        return 0.0
    z = (x - mean) / stdev
    # squash into [-1,1] smoothly
    z = max(-clamp, min(clamp, z))
    return float(z / clamp)

def tanh_scale(x: float, scale: float = 1.0) -> float:
    return float(math.tanh((x or 0.0) / max(1e-12, scale)))

# ---------------------------------------------------------------------
# Common Signals
# ---------------------------------------------------------------------

@dataclass
class SMA:
    """Simple Moving Average. Output is normalized deviation (z-score) clipped to [-1,1]."""
    window: int
    _r: Ring = field(init=False)
    def __post_init__(self): self._r = Ring(self.window)
    def update(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Optional[float]:
        self._r.push(price)
        if not self._r.full():
            return None
        m = self._r.mean() or price
        s = self._r.stdev() or 1.0
        return zscore(price, m, s)

@dataclass
class EMA:
    """Exponential Moving Average. Signal = -(close - ema)/vol_scale → mean-reversion if positive."""
    span: int
    vol_span: Optional[int] = None
    _ema: Optional[float] = None
    _vol: Optional[float] = None
    _alpha: float = field(init=False)
    _vr: Optional[Ring] = field(default=None, init=False)
    def __post_init__(self):
        self._alpha = 2.0 / (self.span + 1.0)
        if self.vol_span: self._vr = Ring(self.vol_span)
    def update(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Optional[float]:
        self._ema = price if self._ema is None else (1-self._alpha)*self._ema + self._alpha*price
        resid = price - self._ema
        if self._vr is not None:
            self._vr.push(abs(resid))
            if not self._vr.full(): return None
            vol_scale = max(1e-6, (self._vr.mean() or 1.0))
        else:
            vol_scale = max(1e-6, abs(self._ema) * 1e-3)
        return tanh_scale(-resid, scale=vol_scale)

@dataclass
class MACross:
    """Moving Average Crossover → momentum signal in [-1,1]."""
    fast: int
    slow: int
    _ema_f: Optional[float] = None
    _ema_s: Optional[float] = None
    def _upd(self, ema: Optional[float], price: float, span: int) -> float:
        a = 2.0/(span+1.0)
        return price if ema is None else (1-a)*ema + a*price
    def update(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Optional[float]:
        self._ema_f = self._upd(self._ema_f, price, self.fast)
        self._ema_s = self._upd(self._ema_s, price, self.slow)
        if self._ema_f is None or self._ema_s is None:
            return None
        diff = self._ema_f - self._ema_s
        base = abs(self._ema_s) if self._ema_s != 0 else 1.0
        return tanh_scale(diff/base, scale=0.01)

@dataclass
class RSI:
    """Relative Strength Index (Wilder's). Output centered to [-1,1]: (RSI-50)/50."""
    period: int = 14
    _avg_gain: Optional[float] = None
    _avg_loss: Optional[float] = None
    _last: Optional[float] = None
    def update(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Optional[float]:
        if self._last is None:
            self._last = price
            return None
        ch = price - self._last
        gain = max(0.0, ch)
        loss = max(0.0, -ch)
        alpha = 1.0 / self.period
        self._avg_gain = gain if self._avg_gain is None else (1-alpha)*self._avg_gain + alpha*gain
        self._avg_loss = loss if self._avg_loss is None else (1-alpha)*self._avg_loss + alpha*loss
        self._last = price
        if (self._avg_loss or 0.0) <= 1e-12:
            rsi = 100.0
        else:
            rs = (self._avg_gain or 0.0) / max(1e-12, self._avg_loss or 0.0)
            rsi = 100.0 - 100.0/(1.0+rs)
        return max(-1.0, min(1.0, (rsi - 50.0) / 50.0))

@dataclass
class VolatilityBreakout:
    """Signal = sign(price - rolling_mean) * min(1, |z|/z_cap)."""
    window: int = 20
    z_cap: float = 2.5
    _r: Ring = field(init=False)
    def __post_init__(self): self._r = Ring(self.window)
    def update(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Optional[float]:
        self._r.push(price)
        if not self._r.full(): return None
        m = self._r.mean() or price
        s = max(1e-9, self._r.stdev() or 1.0)
        zz = (price - m) / s
        return max(-1.0, min(1.0, zz / self.z_cap))

@dataclass
class Momentum:
    """N-period momentum normalized with stdev."""
    window: int = 20
    _r: Ring = field(init=False)
    def __post_init__(self): self._r = Ring(self.window)
    def update(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Optional[float]:
        vals = self._r.values()
        self._r.push(price)
        if len(vals) < self.window-1: return None
        mom = price - vals[0]
        st = max(1e-6, self._r.stdev() or 1.0)
        return tanh_scale(mom, scale=st)

# ---------------------------------------------------------------------
# News / Sentiment fusion (optional inputs from your news pipeline)
# ---------------------------------------------------------------------

@dataclass
class SentimentOverlay:
    """
    Smoothly maps sentiment score s in [-1,1] into a bias multiplier in [1-bias, 1+bias].
    """
    bias: float = 0.3
    decay_ms: int = 15 * 60_000
    _last_s: Optional[float] = None
    _last_ts: Optional[int] = None
    def update(self, symbol: str, price: float, ts_ms: int, *, sent: Optional[float] = None, **_) -> Optional[float]:
        if sent is None and self._last_s is None: return None
        if sent is not None:
            self._last_s, self._last_ts = float(sent), int(ts_ms)
        elif self._last_ts is not None and ts_ms - self._last_ts > self.decay_ms:
            self._last_s = None
            return None
        s = self._last_s or 0.0
        # map to multiplier deviation around 0: +s -> +bias, -s -> -bias
        return max(-1.0, min(1.0, s * self.bias))

# ---------------------------------------------------------------------
# Ensemble / Registry
# ---------------------------------------------------------------------

class SignalRegistry:
    """
    Registry that owns multiple signals per symbol, evaluates them,
    and returns a normalized blended score with optional weights.
    """
    def __init__(self):
        self._signals: Dict[str, Dict[str, Signal]] = {}
        self._weights: Dict[str, Dict[str, float]] = {}
        self._last: Dict[Tuple[str,str], SignalResult] = {}

    def add(self, symbol: str, name: str, signal: Signal, weight: float = 1.0) -> None:
        self._signals.setdefault(symbol, {})[name] = signal
        self._weights.setdefault(symbol, {})[name] = float(weight)

    def remove(self, symbol: str, name: str) -> None:
        if symbol in self._signals:
            self._signals[symbol].pop(name, None)
        if symbol in self._weights:
            self._weights[symbol].pop(name, None)

    def names(self, symbol: str) -> List[str]:
        return list(self._signals.get(symbol, {}).keys())

    def evaluate(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Tuple[Optional[float], List[SignalResult]]:
        sigs = self._signals.get(symbol, {})
        if not sigs: return None, []
        parts: List[Tuple[float,float]] = []  # (value, weight)
        results: List[SignalResult] = []
        for name, s in sigs.items():
            v = s.update(symbol, price, ts_ms, **kwargs)
            res = SignalResult(symbol, name, ts_ms, v)
            self._last[(symbol,name)] = res
            results.append(res)
            w = self._weights.get(symbol, {}).get(name, 1.0)
            if v is not None and w != 0.0:
                parts.append((max(-1.0, min(1.0, float(v))), float(w)))
        if not parts:
            return None, results
        wsum = sum(w for _, w in parts) or 1.0
        blended = sum(v*w for v, w in parts) / wsum
        # extra small shrink toward 0 to reduce whipsaw
        blended = 0.95 * max(-1.0, min(1.0, blended))
        return blended, results

    def last(self, symbol: str, name: str) -> Optional[SignalResult]:
        return self._last.get((symbol,name))

# ---------------------------------------------------------------------
# Policy helper: convert score → order intent
# ---------------------------------------------------------------------

@dataclass
class SignalPolicy:
    """
    Converts a blended signal into a target position (or trade) with caps.
    - `max_gross_w`: cap absolute weight per symbol (e.g., 0.05 = 5% NAV)
    - `target_scale`: maps score [-1,1] → target weight [-scale, +scale]
    - `rebalance_eps`: deadband in weight to avoid tiny trades
    """
    max_gross_w: float = 0.05
    target_scale: float = 0.05
    rebalance_eps: float = 0.002

    def target_weight(self, score: float) -> float:
        s = max(-1.0, min(1.0, score))
        tw = s * self.target_scale
        return max(-self.max_gross_w, min(self.max_gross_w, tw))

# ---------------------------------------------------------------------
# Example: wiring to your strategy loop
# ---------------------------------------------------------------------

class SignalDrivenStrategy:
    """
    Minimal strategy using SignalRegistry:
    - call `on_price(symbol, price, ts_ms)` per tick/bar
    - consult `blended` score and produce action externally
    """
    def __init__(self, name: str, registry: SignalRegistry, policy: Optional[SignalPolicy] = None):
        self.name = name
        self.registry = registry
        self.policy = policy or SignalPolicy()
        self.last_weight: Dict[str, float] = {}

    def on_price(self, symbol: str, price: float, ts_ms: int, **kwargs: Any) -> Dict[str, Any]:
        blended, parts = self.registry.evaluate(symbol, price, ts_ms, **kwargs)
        if blended is None:
            return {"score": None, "parts": parts}
        # translate to target weight
        tw = self.policy.target_weight(blended)
        lw = self.last_weight.get(symbol, 0.0)
        delta_w = tw - lw
        action = "hold"
        if abs(delta_w) > self.policy.rebalance_eps:
            action = "rebalance"
            self.last_weight[symbol] = tw
        return {
            "score": blended,
            "target_weight": tw,
            "delta_weight": delta_w,
            "action": action,
            "parts": parts,
        }

# ---------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import random, time as _t
    sym = "DEMO"
    reg = SignalRegistry()
    reg.add(sym, "ema_mean_rev", EMA(span=20, vol_span=20), weight=1.0)
    reg.add(sym, "macross", MACross(fast=8, slow=21), weight=0.7)
    reg.add(sym, "rsi", RSI(period=14), weight=0.3)
    reg.add(sym, "vol_break", VolatilityBreakout(window=30, z_cap=3.0), weight=0.5)

    strat = SignalDrivenStrategy("demo", reg)

    px = 100.0
    now = int(time.time()*1000)
    for i in range(300):
        px *= (1 + random.uniform(-0.002, 0.002))
        r = strat.on_price(sym, px, now + i*60_000)
        if i % 50 == 0:
            print(i, "score=", round((r["score"] or 0.0), 4), "tw=", round(r.get("target_weight", 0.0),4), r["action"])