# agents/research_agent.py
"""
ResearchAgent
-------------
A lightweight, dependency-free research workbench for factor/alpha studies.

What you get
- Pluggable factor functions: f(symbol, row, history)->float score
- Daily long-short (quantile) backtest with equal weights
- Rank-IC (Kendall/Spearman-style by explicit ranking), IR, hit rate
- Factor decay (autocorrelation of ranks), cross-factor correlation
- Turnover & simple cost model (bps), slippage hook
- Grid search over factor params
- JSON-serializable report dicts

Data model (minimal)
- market_data: Dict[str, List[Row]]
  where Row = {"ts": int|float, "open": float, "high": float, "low": float, "close": float, "volume": float}
  You provide one sorted (oldest->newest) list per symbol.

Factor fn signature
    def my_factor(symbol: str, i: int, hist: List[dict], ctx: dict) -> float:
        # i is the index into hist for current day
        # hist[:i] = past, hist[i] = today
        return score

Backtest conventions (simple, transparent)
- Signals computed at close[t] (ranked cross-sectionally per day)
- Positions entered for t+1 open and held until t+1 close (approx via close-to-close).
  Since we don't have open here, we approximate using close-to-close.
- Rebalance daily.
- Long leg: top Q, Short leg: bottom Q (equal-weight, dollar-neutral).
- Return on day t+1 uses positions from t (one-day lag).

All math done with stdlib for portability & testability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import math
import statistics
import json


Row = Dict[str, float]  # one day of OHLCV (expects at least "ts" and "close")


# ------------------------- Utilities (no numpy) -------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def _rank(values: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Dense ranks (average ties). Input: [(symbol, value), ...]
    Returns: symbol -> rank in [1..N] (1=lowest). Use higher-is-better later if needed.
    """
    sorted_vals = sorted(values, key=lambda kv: (kv[1], kv[0]))
    ranks: Dict[str, float] = {}
    i = 0
    n = len(sorted_vals)
    while i < n:
        j = i
        v = sorted_vals[i][1]
        while j < n and sorted_vals[j][1] == v:
            j += 1
        # average rank for ties, ranks are 1-indexed
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j
    return ranks

def _corr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    try:
        mx, my = statistics.mean(x), statistics.mean(y)
        num = sum((a - mx) * (b - my) for a, b in zip(x, y))
        den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
        return num / den if den else 0.0
    except Exception:
        return 0.0

def _max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    peak = -float("inf")
    mdd = 0.0
    mdd_start = 0
    mdd_end = 0
    for i, v in enumerate(equity_curve):
        if v > peak:
            peak = v
            start_i = i
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
            mdd_start = start_i
            mdd_end = i
    return (mdd, mdd_start, mdd_end)

def _sharpe(returns: List[float], risk_free: float = 0.0, annualization: int = 252) -> float:
    if not returns:
        return 0.0
    mean = statistics.mean(returns) - (risk_free / annualization)
    std = statistics.pstdev(returns)  # population std (stable for small samples)
    daily_sr = _safe_div(mean, std, 0.0)
    return daily_sr * math.sqrt(annualization)

def _ir(ic_series: List[float], annualization: int = 252) -> float:
    if not ic_series:
        return 0.0
    mean = statistics.mean(ic_series)
    std = statistics.pstdev(ic_series)
    daily_ir = _safe_div(mean, std, 0.0)
    return daily_ir * math.sqrt(annualization)


# --------------------------- Config & Models ----------------------------------

@dataclass
class LSConfig:
    quantile: float = 0.2            # top/bottom fraction (e.g., 0.2 => quintiles)
    cost_bps: float = 0.0            # per-leg per-rebalance cost
    max_positions: Optional[int] = None  # cap per leg (after quantile)
    long_only: bool = False          # if True, only long the top quantile

@dataclass
class BacktestResult:
    daily_returns: List[float]
    equity_curve: List[float]
    sharpe: float
    mdd: float
    turnover: float
    hit_rate: float                  # % positive days
    avg_return: float
    cost_bps: float

@dataclass
class FactorStats:
    daily_ic: List[float]
    ir: float
    mean_ic: float
    ic_positive_rate: float
    decay: Dict[int, float]          # lag -> autocorr of factor ranks
    cross_corr: Dict[str, float]     # factor -> corr (if supplied)

@dataclass
class ResearchReport:
    factor_name: str
    backtest: BacktestResult
    stats: FactorStats
    config: LSConfig
    meta: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        def _default(o):
            if isinstance(o, (LSConfig, BacktestResult, FactorStats)):
                return o.__dict__
            return str(o)
        return json.dumps(self.__dict__, default=_default, separators=(",", ":"))


# ------------------------------- ResearchAgent --------------------------------

FactorFn = Callable[[str, int, List[Row], Dict], float]

class ResearchAgent:
    def __init__(self, market_data: Dict[str, List[Row]]):
        """
        market_data: symbol -> chronological list[Row]
        Each list must share the same calendar length/indices across symbols (typical for end-of-day data).
        """
        self.data = market_data
        self.symbols = sorted(market_data.keys())
        self.n = min(len(v) for v in market_data.values()) if market_data else 0

    # -------- Factor computation --------

    def compute_factor_panel(self, name: str, fn: FactorFn, ctx: Optional[Dict] = None) -> List[Dict[str, float]]:
        """
        Returns a list (time) of dicts: [{sym->score} at t0, {sym->score} at t1, ...]
        """
        ctx = ctx or {}
        panel: List[Dict[str, float]] = []
        for i in range(self.n):
            day_scores: Dict[str, float] = {}
            for sym in self.symbols:
                hist = self.data[sym]
                try:
                    score = fn(sym, i, hist, ctx)
                except Exception:
                    score = 0.0
                day_scores[sym] = float(score)
            panel.append(day_scores)
        return panel

    # -------- Rank-IC / Decay / Cross-corr --------

    def rank_ic_series(self, factor_panel: List[Dict[str, float]], fwd_returns: List[Dict[str, float]]) -> List[float]:
        """
        Rank-IC between factor[t] and forward return[t+1] (align so both vectors same date).
        """
        out: List[float] = []
        T = min(len(factor_panel), len(fwd_returns))
        for t in range(T):
            fv = factor_panel[t]
            rv = fwd_returns[t]
            inter = sorted(set(fv.keys()) & set(rv.keys()))
            if len(inter) < 5:
                out.append(0.0); continue
            ranks = _rank([(s, fv[s]) for s in inter])
            # Forward returns already aligned to next period relative to factor signal at t.
            # Since both are same t-index here, fwd_returns[t] should be "return from t to t+1".
            rvec = [rv[s] for s in inter]
            rrank = _rank([(s, rv[s]) for s in inter])
            x = [ranks[s] for s in inter]
            y = [rrank[s] for s in inter]
            out.append(_corr(x, y))
        return out

    def factor_decay(self, factor_panel: List[Dict[str, float]], lags: List[int]) -> Dict[int, float]:
        """
        Autocorrelation of factor ranks across lags (cross-sectional).
        """
        res: Dict[int, float] = {}
        T = len(factor_panel)
        for L in lags:
            vals: List[float] = []
            for t in range(T - L):
                A = factor_panel[t]
                B = factor_panel[t + L]
                inter = sorted(set(A.keys()) & set(B.keys()))
                if len(inter) < 5:
                    continue
                ra = _rank([(s, A[s]) for s in inter])
                rb = _rank([(s, B[s]) for s in inter])
                x = [ra[s] for s in inter]
                y = [rb[s] for s in inter]
                vals.append(_corr(x, y))
            res[L] = statistics.mean(vals) if vals else 0.0
        return res

    def cross_factor_corr(self, panel_a: List[Dict[str, float]], panel_b: List[Dict[str, float]]) -> float:
        """
        Average same-day cross-sectional correlation of ranks between two factor panels.
        """
        T = min(len(panel_a), len(panel_b))
        vals: List[float] = []
        for t in range(T):
            A = panel_a[t]; B = panel_b[t]
            inter = sorted(set(A.keys()) & set(B.keys()))
            if len(inter) < 5:
                continue
            ra = _rank([(s, A[s]) for s in inter])
            rb = _rank([(s, B[s]) for s in inter])
            x = [ra[s] for s in inter]
            y = [rb[s] for s in inter]
            vals.append(_corr(x, y))
        return statistics.mean(vals) if vals else 0.0

    # -------- Backtest (quantile long-short) --------

    def _forward_returns(self) -> List[Dict[str, float]]:
        """
        Compute close-to-close forward returns for each symbol at time t as (close[t+1]/close[t] - 1).
        Last day has no forward return and is dropped to keep alignment with positions from previous day.
        """
        out: List[Dict[str, float]] = []
        for t in range(self.n - 1):
            day: Dict[str, float] = {}
            for sym in self.symbols:
                cur = self.data[sym][t]["close"]
                nxt = self.data[sym][t + 1]["close"]
                day[sym] = _pct_change(cur, nxt)
            out.append(day)
        return out

    def backtest_ls(self, factor_panel: List[Dict[str, float]], cfg: LSConfig) -> BacktestResult:
        """
        Daily rebalanced LS using factor ranks from day t, applied to return t->t+1.
        """
        fwd = self._forward_returns()
        T = min(len(factor_panel), len(fwd))
        daily_rets: List[float] = []
        equity: List[float] = []
        turnover_sum = 0.0
        prev_weights: Dict[str, float] = {}

        cost = cfg.cost_bps / 1e4  # per leg
        for t in range(T - 1):  # stop at T-1 so factor[t] maps to fwd[t]
            scores = factor_panel[t]
            rnext = fwd[t]
            # Rank and pick quantiles
            inter = sorted(set(scores.keys()) & set(rnext.keys()))
            if len(inter) < 5:
                daily_rets.append(0.0)
                equity.append((equity[-1] if equity else 1.0) * (1 + 0.0))
                continue

            ranks = _rank([(s, scores[s]) for s in inter])
            # Higher score -> higher rank; top quantile = highest ranks
            n = len(inter)
            qn = max(1, int(n * cfg.quantile))
            # Build candidate lists (ties handled by sorted order)
            sorted_syms = sorted(inter, key=lambda s: ranks[s])
            lows = sorted_syms[:qn]
            highs = sorted_syms[-qn:]

            if cfg.max_positions:
                highs = highs[-cfg.max_positions:]
                lows = lows[:cfg.max_positions]

            weights: Dict[str, float] = {}
            if cfg.long_only:
                w = 1.0 / len(highs)
                for s in highs:
                    weights[s] = w
            else:
                wl = 0.5 / len(highs) if highs else 0.0
                ws = -0.5 / len(lows) if lows else 0.0
                for s in highs:
                    weights[s] = wl
                for s in lows:
                    weights[s] = ws

            # Cost from turnover (sum |w_t - w_{t-1}| over all names)
            names = set(weights) | set(prev_weights)
            day_turnover = sum(abs(weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in names)
            turnover_sum += day_turnover

            # Portfolio return before cost
            port_ret = sum(weights[s] * rnext.get(s, 0.0) for s in weights)
            # Apply simple costs: cost per leg per notional change (approx via turnover)
            # Since portfolio is 100% gross (0.5 long + 0.5 short) in LS, cost ~ turnover * cost
            net_ret = port_ret - day_turnover * cost

            daily_rets.append(net_ret)
            equity.append((equity[-1] if equity else 1.0) * (1 + net_ret))
            prev_weights = weights

        sharpe = _sharpe(daily_rets)
        mdd, _, _ = _max_drawdown(equity) if equity else (0.0, 0, 0)
        hit = _safe_div(sum(1 for r in daily_rets if r > 0), len(daily_rets))
        avg = statistics.mean(daily_rets) if daily_rets else 0.0
        turnover = _safe_div(turnover_sum, max(1, len(daily_rets)))

        return BacktestResult(
            daily_returns=daily_rets,
            equity_curve=equity,
            sharpe=sharpe,
            mdd=mdd,
            turnover=turnover,
            hit_rate=hit,
            avg_return=avg,
            cost_bps=cfg.cost_bps,
        )

    # -------- Full study (stats + backtest) --------

    def study(
        self,
        factor_name: str,
        factor_fn: FactorFn,
        cfg: Optional[LSConfig] = None,
        ctx: Optional[Dict] = None,
        compare_to: Optional[Dict[str, List[Dict[str, float]]]] = None,  # other factor panels for cross-corr
        decay_lags: List[int] = [1, 5, 10, 20],
    ) -> ResearchReport:
        cfg = cfg or LSConfig()
        panel = self.compute_factor_panel(factor_name, factor_fn, ctx=ctx)
        fwd = self._forward_returns()
        ic_series = self.rank_ic_series(panel[:-1], fwd[:-1])  # align lengths
        stats = FactorStats(
            daily_ic=ic_series,
            ir=_ir(ic_series),
            mean_ic=statistics.mean(ic_series) if ic_series else 0.0,
            ic_positive_rate=_safe_div(sum(1 for x in ic_series if x > 0), len(ic_series)),
            decay=self.factor_decay(panel, decay_lags),
            cross_corr={}
        )
        if compare_to:
            for name, pan in compare_to.items():
                stats.cross_corr[name] = self.cross_factor_corr(panel, pan)

        bt = self.backtest_ls(panel, cfg)
        return ResearchReport(
            factor_name=factor_name,
            backtest=bt,
            stats=stats,
            config=cfg,
            meta={"symbols": ",".join(self.symbols), "bars": str(self.n)},
        )

    # -------- Grid search over factor params --------

    def grid_search(
        self,
        factor_name: str,
        mk_factor: Callable[[Dict], FactorFn],
        param_grid: List[Dict],
        cfg: Optional[LSConfig] = None,
        score_key: str = "sharpe",
    ) -> Tuple[Dict, ResearchReport]:
        """
        mk_factor(params) -> factor_fn
        Returns (best_params, best_report) by score_key in {'sharpe','mean_ic','ir'}.
        """
        best_score = -1e9
        best_params: Dict = {}
        best_report: Optional[ResearchReport] = None
        for p in param_grid:
            fn = mk_factor(p)
            rep = self.study(factor_name, fn, cfg=cfg)
            if score_key == "sharpe":
                sc = rep.backtest.sharpe
            elif score_key == "mean_ic":
                sc = rep.stats.mean_ic
            elif score_key == "ir":
                sc = rep.stats.ir
            else:
                sc = rep.backtest.sharpe
            if sc > best_score:
                best_score, best_params, best_report = sc, p, rep
        return best_params, best_report  # type: ignore


# --------------------------- Example factor(s) --------------------------------
# You can delete below or keep as templates.

def factor_sma_cross(period_fast: int = 5, period_slow: int = 20) -> FactorFn:
    """Positive when fast > slow (trend)."""
    def fn(symbol: str, i: int, hist: List[Row], ctx: Dict) -> float:
        if i < period_slow:
            return 0.0
        fast = sum(hist[i - k]["close"] for k in range(period_fast)) / period_fast
        slow = sum(hist[i - k]["close"] for k in range(period_slow)) / period_slow
        return fast - slow
    return fn

def factor_volume_rev(lookback: int = 10) -> FactorFn:
    """Mean reversion using price change normalized by recent volume."""
    def fn(symbol: str, i: int, hist: List[Row], ctx: Dict) -> float:
        if i < lookback:
            return 0.0
        p0 = hist[i - lookback]["close"]
        p1 = hist[i]["close"]
        dv = sum(hist[i - k]["volume"] for k in range(lookback)) / lookback or 1.0
        return -_pct_change(p0, p1) / dv
    return fn


# --------------------------- Quick smoke test ---------------------------------
if __name__ == "__main__":
    # Tiny synthetic dataset for two symbols
    import random
    random.seed(7)
    def synth(n: int, start: float) -> List[Row]:
        x = start
        out = []
        for t in range(n):
            x *= (1.0 + random.uniform(-0.01, 0.01))
            out.append({"ts": t, "open": x, "high": x, "low": x, "close": x, "volume": 1_000_000})
        return out

    data = {"AAA": synth(300, 100.0), "BBB": synth(300, 50.0), "CCC": synth(300, 25.0), "DDD": synth(300, 10.0), "EEE": synth(300, 5.0)}
    ra = ResearchAgent(data)

    # Study a factor
    factor = factor_sma_cross(5, 20)
    report = ra.study("sma_cross_5_20", factor, cfg=LSConfig(quantile=0.2, cost_bps=5.0))
    print("Sharpe:", round(report.backtest.sharpe, 3), "MDD:", round(report.backtest.mdd, 3), "Mean IC:", round(report.stats.mean_ic, 3))
    # Grid example
    grid = [{"fast": f, "slow": s} for f in (5, 10) for s in (20, 40)]
    def mk(p): return factor_sma_cross(p["fast"], p["slow"])
    best_p, best_rep = ra.grid_search("sma_cross", mk, grid, cfg=LSConfig(quantile=0.2))
    print("Best params:", best_p, "Best Sharpe:", round(best_rep.backtest.sharpe, 3))