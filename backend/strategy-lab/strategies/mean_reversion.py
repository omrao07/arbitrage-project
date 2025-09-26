# strategies/mean_reversion.py
"""
Cross-Sectional Mean Reversion Strategy
--------------------------------------
Generates scores for StrategyAgent from simple z-score deviations vs SMA.
Higher score => more long; negative => short (so we return -z to mean revert).

Plays nicely with:
- agents/strategy_agent.StrategyAgent (scores -> weights conversion)
- agents/execution_agent.ExecutionAgent (via StrategyAgent)

Key knobs
- lookback: SMA/STD window for z-score
- z_cap: clip extreme z (robust to outliers)
- vol_window: recent window to compute volatility scaler (optional)
- vol_target: scale scores by target / realized vol (if vol_window>0)
- cooldown_bps: after a big move (|ret| >= cooldown_bps), reduce the score
- universe: fixed list or dynamic (learned from prices seen)
- max_positions: limit names per rebalance (StrategyAgent also has one)
- gross_target / long_only: StrategyBase knobs (used by StrategyAgent)

Signal math
    z = (price - SMA_lookback) / (STD_lookback + 1e-9)
    score = -clip(z, ±z_cap) * vol_scale * cooldown_scale

All stdlib, no pandas/numpy.

Example
-------
from agents.strategy_agent import StrategyAgent
from strategies.mean_reversion import MeanReversionStrategy

mr = MeanReversionStrategy(lookback=10, vol_window=20, z_cap=3.0)
sa = StrategyAgent(exec_agent)
sa.register(mr, weight=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import math
import statistics
import time


# ----------------------------- small helpers ----------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------- Strategy ---------------------------------------

class MeanReversionStrategy:
    """
    StrategyBase-compatible class.
    Implements: name, warmup_bars, max_positions, gross_target, long_only,
                optional universe(), on_price(), generate_signals().
    """
    name = "mean_reversion"

    # StrategyBase knobs (read by StrategyAgent)
    warmup_bars: int = 0
    max_positions: Optional[int] = None
    gross_target: float = 0.5
    long_only: bool = False

    def __init__(
        self,
        lookback: int = 10,
        z_cap: float = 3.0,
        vol_window: int = 0,
        vol_target: float = 0.01,              # ~1% daily target scale
        cooldown_bps: float = 150.0,           # damp signals after |ret| >= 150 bps in last bar
        eligible_universe: Optional[Iterable[str]] = None,
        blacklist: Optional[Iterable[str]] = None,
        min_price: float = 1.0,
        max_price: Optional[float] = None,
        min_history: Optional[int] = None,     # override warmup if you want more history
    ):
        self.lookback = max(2, int(lookback))
        self.z_cap = max(0.5, float(z_cap))
        self.vol_window = max(0, int(vol_window))
        self.vol_target = max(0.0, float(vol_target))
        self.cooldown_bps = max(0.0, float(cooldown_bps))
        self._eligible: Optional[Set[str]] = set(eligible_universe) if eligible_universe else None
        self._black: Set[str] = set(blacklist) if blacklist else set()
        self.min_price = float(min_price)
        self.max_price = float(max_price) if max_price is not None else None

        # History buffers
        self.prices: Dict[str, List[float]] = {}
        self.returns: Dict[str, List[float]] = {}

        # warmup bars required to compute z + (optional) vol
        self.warmup_bars = max(self.lookback + 1, (self.vol_window + 1) if self.vol_window > 0 else self.lookback + 1)
        if min_history:
            self.warmup_bars = max(self.warmup_bars, int(min_history))

    # --- optional universe provider (StrategyAgent may ignore) ---
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
        if price is None or price <= 0:
            return
        if price < self.min_price or (self.max_price is not None and price > self.max_price):
            # keep buffer but skip generating signals later if outside price bounds
            pass

        buf = self.prices.setdefault(symbol, [])
        buf.append(float(price))
        # returns
        if len(buf) >= 2:
            r = _pct_change(buf[-2], buf[-1])
            self.returns.setdefault(symbol, []).append(r)

    def on_fill(self, order_id: str, symbol: str, qty: float, price: float) -> None:
        # Not needed for stateless signal gen; hook exists for extensions (cooldowns per fill, etc.)
        pass

    # --- core: generate signals ---

    def generate_signals(self, now_ts: float) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        # Pick symbols from eligible set (if provided) or what we've seen
        symbols = self._eligible if self._eligible is not None else set(self.prices.keys())
        for s in symbols:
            if s in self._black:
                continue
            hist = self.prices.get(s, [])
            if len(hist) < self.warmup_bars:
                continue

            p = hist[-1]
            if p <= 0 or p < self.min_price or (self.max_price is not None and p > self.max_price):
                continue

            # z-score vs SMA lookback
            window = hist[-self.lookback :]
            mu = sum(window) / len(window)
            # unbiased-ish std (population std is fine for stability on small N)
            var = sum((x - mu) ** 2 for x in window) / max(1, (len(window) - 1))
            sd = math.sqrt(var) if var > 0 else 0.0
            if sd <= 1e-12:
                # flat price -> no signal
                continue

            z = (p - mu) / sd
            z = _clip(z, -self.z_cap, self.z_cap)

            # volatility scaling (optional)
            vol_scale = 1.0
            if self.vol_window > 0:
                rbuf = self.returns.get(s, [])
                if len(rbuf) >= self.vol_window:
                    recent = rbuf[-self.vol_window :]
                    realized = statistics.pstdev(recent) if len(recent) > 1 else 0.0
                    vol_scale = _safe_div(self.vol_target, realized, 1.0)
                    # cap to avoid huge leverage in calm periods
                    vol_scale = _clip(vol_scale, 0.25, 4.0)

            # cooldown after big bar move
            cooldown_scale = 1.0
            if self.cooldown_bps > 0 and self.returns.get(s):
                last_ret = self.returns[s][-1]
                if abs(last_ret) >= (self.cooldown_bps / 1e4):
                    # damp by 50% after a large bar; linear ramp could be used too
                    cooldown_scale = 0.5

            # mean reversion => opposite the z (above mean -> short; below -> long)
            score = -z * vol_scale * cooldown_scale

            # final tiny deadband to avoid churn on micro-z
            if abs(score) < 0.05:
                score = 0.0

            scores[s] = float(score)

        # StrategyAgent will handle:
        # - z-scoring of scores again (cross-sectionally) -> weights
        # - per-symbol cap, gross/long/short caps, max_positions, etc.
        return scores