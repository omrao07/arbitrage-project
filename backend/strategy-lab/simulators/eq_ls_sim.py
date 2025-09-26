# simulators/eq_ls_sim.py
"""
Equal-Weight Long–Short Simulator (stdlib-only)
----------------------------------------------
Quickly backtest a cross-sectional factor on an aligned DataSet.

What it does
- Computes factor scores each bar via your function: f(symbol, i, history, ctx) -> float
- Ranks scores cross-sectionally per day.
- Forms top/bottom quantile portfolios (equal-weight), dollar-neutral by default.
- Applies one-bar lag: signals at t produce positions for return t->t+1 (close-to-close).
- Tracks returns, equity curve, Sharpe, max drawdown, turnover, and simple costs.

No external dependencies.

Usage (sketch)
-------------
from simulators.envs.data_loader import MarketDataLoader
from simulators.eq_ls_sim import EqLSSimConfig, EqLSSimulator

loader = MarketDataLoader()
ds = loader.load_dir("./data/*.csv", fmt="csv", align=True)

def my_factor(symbol, i, hist, ctx):
    # Simple momentum: 10-bar change
    if i < 10: return 0.0
    return hist[i]["close"] - hist[i-10]["close"]

cfg = EqLSSimConfig(quantile=0.2, cost_bps=5.0)
sim = EqLSSimulator(ds, cfg)
report = sim.run(my_factor)

print("Sharpe:", round(report["metrics"]["sharpe"], 3),
      "MDD:", round(report["metrics"]["max_drawdown"], 3),
      "Turnover:", round(report["metrics"]["turnover"], 3))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import math
import statistics
import json

Row = Dict[str, float]  # expects keys: ts, open, high, low, close, volume
FactorFn = Callable[[str, int, List[Row], Dict], float]


# ----------------------------- small helpers ----------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def _max_drawdown(curve: List[float]) -> Tuple[float, int, int]:
    peak = -float("inf")
    mdd = 0.0
    s = e = 0
    for i, v in enumerate(curve):
        if v > peak:
            peak = v; s = i
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd; e = i
    return mdd, s, e

def _sharpe(returns: List[float], rf: float = 0.0, ann: int = 252) -> float:
    if not returns:
        return 0.0
    mu = statistics.mean(returns) - (rf / ann)
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return _safe_div(mu, sd) * math.sqrt(ann)


# ----------------------------- config -----------------------------------------

@dataclass
class EqLSSimConfig:
    quantile: float = 0.2            # top/bottom fraction (0<q<=0.5). If long_only=True, uses only top q.
    long_only: bool = False          # if True, only long the top quantile
    cost_bps: float = 0.0            # per notional turnover cost (one-way)
    rebalance_every_bars: int = 1    # 1=daily (each bar); set >1 for slower cadence
    max_positions: Optional[int] = None  # cap names per leg after quantile
    neutralize_gross: float = 1.0    # target gross (1.0 => 100% gross; scaled for long_only as well)
    rf_rate: float = 0.0             # annualized risk-free for Sharpe
    annualization: int = 252         # bars per year (daily data default)
    report_panel: bool = False       # include factor panel & weights time series (heavy)


# ----------------------------- simulator --------------------------------------

class EqLSSimulator:
    def __init__(self, dataset, cfg: Optional[EqLSSimConfig] = None):
        """
        dataset: simulators.envs.data_loader.DataSet (aligned calendar across symbols)
        """
        self.ds = dataset
        self.cfg = cfg or EqLSSimConfig()
        self.symbols = sorted(self.ds.data.keys())
        self.n = len(self.ds.calendar)

    def run(self, factor_fn: FactorFn, ctx: Optional[Dict] = None) -> Dict:
        """
        Execute the sim with the given factor function.
        Returns a report dict with metrics and curves.
        """
        if self.n < 3 or not self.symbols:
            return {"metrics": {}, "equity_curve": [], "daily_returns": []}

        ctx = ctx or {}
        q = max(1e-9, min(0.5, float(self.cfg.quantile)))
        cost = self.cfg.cost_bps / 1e4
        cadence = max(1, int(self.cfg.rebalance_every_bars))
        gross_target = max(0.0, float(self.cfg.neutralize_gross))

        # Precompute forward returns close[t] -> close[t+1] for all symbols
        fwd: List[Dict[str, float]] = []
        for t in range(self.n - 1):
            day = {}
            for s in self.symbols:
                cur = float(self.ds.data[s][t]["close"])
                nxt = float(self.ds.data[s][t + 1]["close"])
                day[s] = _pct_change(cur, nxt)
            fwd.append(day)

        # Compute factor panel per bar
        panel: List[Dict[str, float]] = []
        for t in range(self.n):
            scores = {}
            for s in self.symbols:
                hist = self.ds.data[s]
                try:
                    scores[s] = float(factor_fn(s, t, hist, ctx))
                except Exception:
                    scores[s] = 0.0
            panel.append(scores)

        # Sim loop (positions from t apply to fwd[t])
        equity = 1.0
        curve = [equity]
        rets: List[float] = []
        turnover_sum = 0.0

        prev_w: Dict[str, float] = {}
        weights_series: List[Dict[str, float]] = [] if self.cfg.report_panel else None  # type: ignore

        for t in range(min(len(panel) - 1, len(fwd))):
            # Rebalance on cadence
            do_rebalance = (t % cadence == 0)

            if do_rebalance:
                scores = panel[t]
                # rank cross-section
                names = [s for s in self.symbols if s in scores and s in fwd[t]]
                if len(names) < 5:
                    w = prev_w  # keep previous
                else:
                    ranks = {s: r for r, s in enumerate(sorted(names, key=lambda s: (scores[s], s)))}
                    n = len(names)
                    qn = max(1, int(n * q))
                    lows = sorted(names, key=lambda s: ranks[s])[:qn]
                    highs = sorted(names, key=lambda s: ranks[s])[-qn:]

                    if self.cfg.max_positions:
                        highs = highs[-self.cfg.max_positions:]
                        lows = lows[:self.cfg.max_positions]

                    w: Dict[str, float] = {}
                    if self.cfg.long_only:
                        # allocate gross_target on the top bucket
                        wl = _safe_div(gross_target, len(highs))
                        for s in highs:
                            w[s] = wl
                    else:
                        # dollar-neutral: +gross/2 to longs, -gross/2 to shorts
                        wl = _safe_div(gross_target * 0.5, len(highs)) if highs else 0.0
                        ws = -_safe_div(gross_target * 0.5, len(lows)) if lows else 0.0
                        for s in highs:
                            w[s] = wl
                        for s in lows:
                            w[s] = ws
            else:
                w = prev_w  # hold

            # Compute turnover (|w - prev_w| sum)
            names_union = set(w) | set(prev_w)
            day_turn = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in names_union)
            turnover_sum += day_turn

            # Portfolio return (before cost)
            r = sum(w.get(s, 0.0) * fwd[t].get(s, 0.0) for s in w)

            # Apply simple linear cost on turnover
            r_net = r - day_turn * cost
            rets.append(r_net)
            equity *= (1 + r_net)
            curve.append(equity)

            prev_w = w
            if weights_series is not None:
                weights_series.append(dict(w))

        # Metrics
        mdd, mdd_s, mdd_e = _max_drawdown(curve)
        sharpe = _sharpe(rets, self.cfg.rf_rate, self.cfg.annualization)
        hit = _safe_div(sum(1 for x in rets if x > 0), len(rets))
        avg = statistics.mean(rets) if rets else 0.0
        vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
        tavg = _safe_div(turnover_sum, max(1, len(rets)))

        report = {
            "metrics": {
                "sharpe": sharpe,
                "avg_return": avg,
                "vol": vol,
                "hit_rate": hit,
                "max_drawdown": mdd,
                "turnover": tavg,
                "bars": len(rets),
            },
            "equity_curve": curve,           # starts at 1.0
            "daily_returns": rets,           # length == bars
            "config": self.cfg.__dict__,
        }
        if self.cfg.report_panel:
            report["factor_panel"] = panel
            report["weights_series"] = weights_series
        return report


# ----------------------------- CLI smoke test ---------------------------------

if __name__ == "__main__":
    # Tiny synthetic demo without external files
    import random
    random.seed(7)

    def synth(n: int, start: float):
        x = start
        rows = []
        t0 = 1_700_000_000
        for i in range(n):
            x *= (1.0 + random.uniform(-0.01, 0.01))
            rows.append({"ts": t0 + i*86400, "open": x, "high": x, "low": x, "close": x, "volume": 1_000_000})
        return rows

    # Build an aligned DataSet-like object
    class DS:
        pass

    data = {"AAA": synth(300, 100.0), "BBB": synth(300, 50.0), "CCC": synth(300, 25.0), "DDD": synth(300, 10.0)}
    cal = sorted(set(r["ts"] for r in data["AAA"]))
    ds = DS()
    ds.data = data # type: ignore
    ds.calendar = cal # type: ignore

    # Simple factor: 5–20 SMA cross (positive if fast>slow)
    def sma_factor(symbol: str, i: int, hist: List[Row], ctx: Dict) -> float:
        pf, ps = 5, 20
        if i < ps: return 0.0
        fast = sum(hist[i-k]["close"] for k in range(pf)) / pf
        slow = sum(hist[i-k]["close"] for k in range(ps)) / ps
        return fast - slow

    cfg = EqLSSimConfig(quantile=0.25, cost_bps=5.0, long_only=False, report_panel=False)
    sim = EqLSSimulator(ds, cfg)
    rep = sim.run(sma_factor)

    print(json.dumps(rep["metrics"], separators=(",", ":"), default=float))