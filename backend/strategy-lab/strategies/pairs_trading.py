# strategies/pairs_trading.py
"""
Pairs Trading (stat-arb) Strategy
---------------------------------
Signal per pair:
  1) Estimate hedge ratio β_t over a rolling lookback using OLS: Y ≈ α + β X
  2) Spread_t = Y_t - β_t * X_t
  3) z_t = (Spread_t - mean(Spread_{t-L..t-1})) / std(Spread_{t-L..t-1})
  4) Trading logic:
       if z_t > entry_z  -> SHORT spread (short Y, long X*β)
       if z_t < -entry_z -> LONG  spread (long  Y, short X*β)
       exit when |z_t| < exit_z
       optional hard stop if |z_t| > stop_z (flatten signal toward 0)

Outputs
- StrategyAgent-compatible *scores* dict {symbol: score}, aggregated across pairs.
  Higher score => tilt long; negative => tilt short.
- Sizing is left to StrategyAgent (z-scoring + caps). This strategy scales
  scores by |z| (clipped) and by a per-pair weight.

Notes
- No external deps. Rolling OLS and stats implemented with plain Python.
- Works with your StrategyAgent: register this class and call on_price(...) in your loop.
- You can track per-pair "position state" to avoid churn between entry/exit bands;
  here we keep a *soft* state that damps signals when inside the no-trade band.

Example
-------
from agents.strategy_agent import StrategyAgent
from strategies.pairs_trading import PairsTradingStrategy

pairs = [("XOM","CVX"), ("MSFT","AAPL")]
strat = PairsTradingStrategy(pairs, lookback=60, entry_z=2.0, exit_z=0.5)
sa.register(strat, weight=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math
import statistics
import time


Pair = Tuple[str, str]

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class PairState:
    beta: float = 1.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    z: float = 0.0
    mode: str = "flat"   # "flat" | "long_spread" | "short_spread"


class PairsTradingStrategy:
    """
    StrategyBase-compatible (see StrategyAgent).
    """

    name = "pairs_trading"

    # StrategyBase knobs consumed by StrategyAgent
    warmup_bars: int = 0
    max_positions: Optional[int] = None
    gross_target: float = 0.3          # cross-pair, keep modest by default
    long_only: bool = False            # ignored in practice; pairs are long/short by design

    def __init__(
        self,
        pairs: Iterable[Pair],
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: Optional[float] = 4.0,
        per_pair_weight: float = 1.0,       # score scale multiplier per pair
        beta_mode: str = "rolling",         # "rolling" | "fixed"
        fixed_betas: Optional[Dict[Pair, float]] = None,
        min_price: float = 1.0,
        max_price: Optional[float] = None,
        damp_inside_band: float = 0.25,     # when in flat mode & |z|<entry, damp residual scores
    ):
        """
        pairs: list of (X, Y) where spread = Y - β*X (β estimated rolling or fixed)
        """
        self.pairs: List[Pair] = [tuple(p) for p in pairs] # type: ignore
        self.lookback = max(10, int(lookback))
        self.entry_z = max(0.5, float(entry_z))
        self.exit_z = max(0.0, float(exit_z))
        self.stop_z = float(stop_z) if stop_z is not None else None
        self.per_pair_weight = float(per_pair_weight)
        self.beta_mode = beta_mode.lower()
        self.fixed_betas = {(tuple(k) if not isinstance(k, tuple) else k): float(v) for k, v in (fixed_betas or {}).items()}
        self.min_price = float(min_price)
        self.max_price = float(max_price) if max_price is not None else None
        self.damp_inside_band = float(damp_inside_band)

        # Price buffers
        self.prices: Dict[str, List[float]] = {}

        # Per-pair runtime state
        self.state: Dict[Pair, PairState] = {p: PairState() for p in self.pairs}

        # Warmup requirement: enough bars to estimate beta + z on spread
        self.warmup_bars = max(self.lookback + 1, 2 * self.lookback)

    # ---- optional universe used by StrategyAgent (not strictly required) ----
    def universe(self) -> Iterable[str]:
        u = set()
        for x, y in self.pairs:
            u.add(x); u.add(y)
        return sorted(u)

    # ---- event hooks ----
    def on_price(self, symbol: str, price: float) -> None:
        if not price or price <= 0:
            return
        if price < self.min_price or (self.max_price is not None and price > self.max_price):
            # still record but signals may skip later
            pass
        self.prices.setdefault(symbol, []).append(float(price))

    def on_fill(self, order_id: str, symbol: str, qty: float, price: float) -> None:
        # Not used; kept for StrategyBase symmetry
        pass

    # ---- core signal generation ----
    def generate_signals(self, now_ts: float) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        for pair in self.pairs:
            X, Y = pair
            hx = self.prices.get(X, [])
            hy = self.prices.get(Y, [])
            n = min(len(hx), len(hy))
            if n < self.lookback + 1:
                # not enough data yet
                continue

            # Recent windows
            xx = hx[-self.lookback:]
            yy = hy[-self.lookback:]

            # Guard prices
            if xx[-1] < self.min_price or yy[-1] < self.min_price:
                continue
            if (self.max_price is not None) and (xx[-1] > self.max_price or yy[-1] > self.max_price):
                continue

            # Hedge ratio β
            if self.beta_mode == "fixed" and pair in self.fixed_betas:
                beta = self.fixed_betas[pair]
            else:
                beta = _rolling_beta(xx, yy)  # robust to const series
            beta = 1.0 if not math.isfinite(beta) else beta

            # Spread & z-score
            spread_series = [yy[i] - beta * xx[i] for i in range(len(xx))]
            mu = statistics.mean(spread_series[:-1])  # exclude the last point when computing z
            sd = statistics.pstdev(spread_series[:-1]) if len(spread_series) > 2 else 0.0
            sd = sd if sd > 1e-12 else 0.0

            cur_spread = spread_series[-1]
            z = (cur_spread - mu) / sd if sd > 0 else 0.0

            st = self.state[pair]
            st.beta, st.spread_mean, st.spread_std, st.z = beta, mu, sd, z

            # Position logic -> target "direction" in spread space
            # +1  => SHORT spread (short Y, long X*β) when z > entry_z
            # -1  => LONG  spread (long  Y, short X*β) when z < -entry_z
            direction = 0
            if st.mode == "flat":
                if z > self.entry_z:
                    st.mode = "short_spread"; direction = +1
                elif z < -self.entry_z:
                    st.mode = "long_spread"; direction = -1
                else:
                    direction = 0  # flat band
            elif st.mode == "short_spread":
                if abs(z) < self.exit_z:
                    st.mode = "flat"; direction = 0
                else:
                    direction = +1
            elif st.mode == "long_spread":
                if abs(z) < self.exit_z:
                    st.mode = "flat"; direction = 0
                else:
                    direction = -1

            # Hard stop: if |z| explodes, fade toward flat (risk-off)
            if self.stop_z is not None and abs(z) > self.stop_z:
                direction = 0

            # Score magnitude scales with |z| (clipped), then damp inside band if flat.
            mag = _clip(abs(z), 0.0, 3.0)  # cap contribution
            if st.mode == "flat" and abs(z) < self.entry_z:
                mag *= self.damp_inside_band

            pair_scale = self.per_pair_weight * mag

            # Map spread direction to leg scores:
            # SHORT spread (+1): short Y, long X*β
            # LONG  spread (-1): long  Y, short X*β
            if direction != 0 and pair_scale > 0:
                # Allocate symmetric scores; β influences relative strength
                # Normalize (1 + |β|) to avoid oversizing the β leg in score space
                denom = 1.0 + abs(beta)
                sX = ( direction * (+abs(beta)) / denom )   # + for long X when short spread
                sY = ( direction * (-1.0)        / denom )   # - for short Y when short spread
                # For long spread (direction=-1), signs flip naturally.
                scores[X] = scores.get(X, 0.0) + pair_scale * sX
                scores[Y] = scores.get(Y, 0.0) + pair_scale * sY
            else:
                # In flat/no-trade: gently pull legs toward neutral (small mean-reverting nudge).
                # This helps StrategyAgent de-allocate gradually when z nears exit band.
                if self.damp_inside_band > 0:
                    # tiny opposite of current z direction (nudges toward 0)
                    tiny = self.damp_inside_band * 0.1
                    if z > 0:
                        # lean to LONG X, SHORT Y just a bit (anticipate reversion)
                        scores[X] = scores.get(X, 0.0) + tiny
                        scores[Y] = scores.get(Y, 0.0) - tiny
                    elif z < 0:
                        scores[X] = scores.get(X, 0.0) - tiny
                        scores[Y] = scores.get(Y, 0.0) + tiny
                    # if z==0: no nudge

        return scores


# --------------------------- math helpers (rolling OLS) ---------------------------

def _rolling_beta(x: List[float], y: List[float]) -> float:
    """
    OLS slope with intercept on last len(x) points:
        minimize sum (y - a - b x)^2  -> b = Cov(x,y) / Var(x)
    Numerically stable enough for small windows, stdlib only.
    """
    n = len(x)
    if n == 0:
        return 1.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = 0.0
    varx = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        cov += dx * dy
        varx += dx * dx
    if varx <= 1e-12:
        return 1.0
    return cov / varx


# --------------------------- tiny self-test -------------------------------------

if __name__ == "__main__":
    # Minimal smoke: synth two correlated series where Y ≈ 1.5 X + noise
    import random
    random.seed(7)
    xs = [100.0]
    ys = [150.0]
    for _ in range(200):
        xs.append(xs[-1] * (1 + random.uniform(-0.01, 0.01)))
        ys.append(1.5 * xs[-1] * (1 + random.uniform(-0.005, 0.005)))

    strat = PairsTradingStrategy([("X","Y")], lookback=60, entry_z=2.0, exit_z=0.5)
    for i in range(len(xs)):
        strat.on_price("X", xs[i])
        strat.on_price("Y", ys[i])
        _ = strat.generate_signals(time.time())
    print("Last z:", round(strat.state[('X','Y')].z, 3), "mode:", strat.state[('X','Y')].mode)