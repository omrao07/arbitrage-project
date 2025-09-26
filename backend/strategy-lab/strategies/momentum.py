# strategies/momentum.py
"""
Cross-Sectional Momentum Strategy
---------------------------------
Generates StrategyAgent-friendly scores (higher => more long).

Signal (default)
    mom = close[t - skip] / close[t - lookback - skip] - 1
    score = clip(mom, ±mom_cap) * vol_scale * trend_gate * breakout_boost

Key knobs
- lookback: momentum lookback length (bars)
- skip: bars to skip between lookback end and today (avoid reversal)
- mom_cap: clips raw momentum to tame outliers
- vol_window: realized vol window for inverse-vol scaling (risk parity-ish)
- vol_target: target daily vol for vol scaling
- breakout_k: boost if close[t] > max(close[t-k..t-1]) (or < min for shorts)
- trend_filter: gate with SMA fast>slow (or disable)
- eligible_universe / blacklist: hard include/exclude
- min_price / max_price: price guards
- warmup_bars is set to satisfy the strictest requirement

Pairs well with:
- agents/strategy_agent.StrategyAgent (weights, caps, rebal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set
import math
import statistics

# ----------------------------- helpers -----------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ----------------------------- Strategy ----------------------------

class MomentumStrategy:
    """
    StrategyBase-compatible (see StrategyAgent).
    Exposes: name, warmup_bars, max_positions, gross_target, long_only,
             universe(), on_price(), generate_signals().
    """
    name = "momentum"

    # StrategyBase knobs consumed by StrategyAgent
    warmup_bars: int = 0
    max_positions: Optional[int] = None
    gross_target: float = 0.5
    long_only: bool = False

    def __init__(
        self,
        lookback: int = 126,              # ~6 months on daily data
        skip: int = 5,                    # 1 trading week skip by default
        mom_cap: float = 1.50,            # cap raw momentum at ±150%
        vol_window: int = 20,             # inverse-vol scaling window (0 disables)
        vol_target: float = 0.01,         # ~1% daily target
        breakout_k: int = 20,             # breakout lookback (0 disables)
        breakout_boost: float = 1.25,     # multiplicative boost on breakouts
        trend_filter: Optional[tuple] = (20, 60),  # (fast, slow) SMA gate; None to disable
        eligible_universe: Optional[Iterable[str]] = None,
        blacklist: Optional[Iterable[str]] = None,
        min_price: float = 1.0,
        max_price: Optional[float] = None,
        min_history: Optional[int] = None,
    ):
        self.lookback = max(2, int(lookback))
        self.skip = max(0, int(skip))
        self.mom_cap = max(0.1, float(mom_cap))
        self.vol_window = max(0, int(vol_window))
        self.vol_target = max(0.0, float(vol_target))
        self.breakout_k = max(0, int(breakout_k))
        self.breakout_boost = max(1.0, float(breakout_boost))
        self.trend_filter = tuple(trend_filter) if trend_filter else None

        self._eligible: Optional[Set[str]] = set(eligible_universe) if eligible_universe else None
        self._black: Set[str] = set(blacklist) if blacklist else set()
        self.min_price = float(min_price)
        self.max_price = float(max_price) if max_price is not None else None

        # buffers
        self.prices: Dict[str, List[float]] = {}
        self.returns: Dict[str, List[float]] = {}

        # warmup to support: lookback+skip, vol_window, breakout_k, trend_filter slow leg
        req = [self.lookback + self.skip + 1]
        if self.vol_window > 0: req.append(self.vol_window + 1)
        if self.breakout_k > 0: req.append(self.breakout_k + 1)
        if self.trend_filter:   req.append(max(self.trend_filter) + 1)
        self.warmup_bars = max(req) if req else (self.lookback + self.skip + 1)
        if min_history: self.warmup_bars = max(self.warmup_bars, int(min_history))

    # --- optional universe provider ---
    def universe(self) -> Iterable[str]:
        if self._eligible is not None:
            return list(self._eligible)
        return list(self.prices.keys())

    # --- event hooks ---
    def on_price(self, symbol: str, price: float) -> None:
        if symbol in self._black:
            return
        if self._eligible is not None and symbol not in self._eligible:
            return
        if not price or price <= 0:
            return

        buf = self.prices.setdefault(symbol, [])
        buf.append(float(price))
        if len(buf) >= 2:
            self.returns.setdefault(symbol, []).append(_pct_change(buf[-2], buf[-1]))

    def on_fill(self, order_id: str, symbol: str, qty: float, price: float) -> None:
        # not used; kept for API symmetry
        pass

    # --- core: signal gen ---
    def generate_signals(self, now_ts: float) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        symbols = self._eligible if self._eligible is not None else set(self.prices.keys())

        for s in symbols:
            if s in self._black:
                continue
            hist = self.prices.get(s, [])
            if len(hist) < self.warmup_bars:
                continue

            p = hist[-1]
            if p < self.min_price or (self.max_price is not None and p > self.max_price):
                continue

            # --- momentum with skip ---
            a = len(hist) - 1 - self.skip
            b = a - self.lookback
            if b < 0:
                continue
            p_end = hist[a]
            p_start = hist[b]
            raw_mom = _safe_div(p_end - p_start, p_start, 0.0)  # close[t-skip]/close[t-lookback-skip]-1
            raw_mom = _clip(raw_mom, -self.mom_cap, self.mom_cap)

            # --- inverse-vol scaling (optional) ---
            vol_scale = 1.0
            if self.vol_window > 0:
                rbuf = self.returns.get(s, [])
                if len(rbuf) >= self.vol_window:
                    realized = statistics.pstdev(rbuf[-self.vol_window:]) if self.vol_window > 1 else 0.0
                    vol_scale = _safe_div(self.vol_target, realized, 1.0)
                    vol_scale = _clip(vol_scale, 0.25, 4.0)

            # --- breakout boost (optional) ---
            boost = 1.0
            if self.breakout_k > 0:
                past = hist[-self.breakout_k - 1 : -1]
                if past:
                    hi = max(past); lo = min(past)
                    if p > hi and raw_mom > 0:
                        boost = self.breakout_boost
                    elif p < lo and raw_mom < 0:
                        boost = self.breakout_boost

            # --- trend filter gate (optional) ---
            gate = 1.0
            if self.trend_filter:
                f, sslow = self.trend_filter
                if len(hist) >= sslow:
                    fast = sum(hist[-f:]) / f
                    slow = sum(hist[-sslow:]) / sslow
                    # require momentum sign to match fast-vs-slow trend; else gate down
                    if (raw_mom > 0 and fast <= slow) or (raw_mom < 0 and fast >= slow):
                        gate = 0.25  # damp instead of zero to avoid hard flips

            score = raw_mom * vol_scale * boost * gate

            # small deadband to reduce churn
            if abs(score) < 0.01:
                score = 0.0

            scores[s] = float(score)

        # StrategyAgent will z-score cross-sectionally and apply caps/limits.
        return scores