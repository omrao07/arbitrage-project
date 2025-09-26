# simulators/backtester.py
"""
Backtester (stdlib-only)
------------------------
Runs a historical simulation using:
- DataSet from simulators/envs/data_loader.py
- ExecutionAgent from agents/execution_agent.py
- StrategyAgent from agents/strategy_agent.py (hosting your strategies)

What it does
- Iterates bars from a calendar-aligned dataset.
- Feeds last prices into ExecutionAgent, allowing LIMIT/STOP to fill.
- Rebalances on 'daily'/'weekly'/'monthly' or fixed cadence in bars.
- Records equity curve, returns, Sharpe, max drawdown.
- (Optional) Writes a JSONL fills log.

No external dependencies.

Quick start
-----------
from simulators.envs.data_loader import MarketDataLoader
from agents.execution_agent import ExecutionAgent
from agents.strategy_agent import StrategyAgent, RebalanceConfig
# (register your StrategyBase subclass on StrategyAgent)

loader = MarketDataLoader()
ds = loader.load_dir("./data/*.csv", fmt="csv", align=True)

x = ExecutionAgent(starting_cash=1_000_000.0)
sa = StrategyAgent(x, RebalanceConfig(cadence_sec=0))  # cadence_sec ignored in backtests

from simulators.backtester import BacktestConfig, Backtester
bt = Backtester(ds, x, sa, BacktestConfig(rebalance="daily", warmup_bars=20))
result = bt.run()

print("Sharpe:", round(result["metrics"]["sharpe"], 3),
      "MDD:", round(result["metrics"]["max_drawdown"], 3))
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

try:
    # Local imports in repo structure
    from simulators.envs.data_loader import DataSet
    from agents.execution_agent import ExecutionAgent, Fill # type: ignore
except Exception:  # pragma: no cover
    DataSet = object  # type: ignore
    ExecutionAgent = object  # type: ignore
    class Fill: ...  # type: ignore


# ------------------------------- Config ---------------------------------------

@dataclass
class BacktestConfig:
    rebalance: str = "daily"            # 'daily' | 'weekly' | 'monthly' | 'none'
    cadence_bars: Optional[int] = None  # if set, overrides 'rebalance' schedule (e.g., 5 => every 5 bars)
    warmup_bars: int = 0                # bars to pass through before first rebalance (for indicator warmup)
    start_bar: Optional[int] = None     # inclusive index into ds.calendar
    end_bar: Optional[int] = None       # inclusive index into ds.calendar
    mark_to_market_each_bar: bool = True
    fills_log_path: Optional[str] = None  # JSONL file to append fill events (optional)
    print_progress: bool = False


# ------------------------------- Utilities ------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def _max_drawdown(curve: List[float]) -> Tuple[float, int, int]:
    peak = -float("inf")
    mdd = 0.0
    start = end = 0
    for i, v in enumerate(curve):
        if v > peak:
            peak = v
            start = i
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
            end = i
    return mdd, start, end

def _sharpe(returns: List[float], rf: float = 0.0, ann: int = 252) -> float:
    if not returns:
        return 0.0
    mu = statistics.mean(returns) - (rf / ann)
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return _safe_div(mu, sd) * math.sqrt(ann)

def _ts_to_ymd(ts: float) -> Tuple[int, int, int, int]:
    """Return (year, month, iso_week, yday) from epoch seconds (UTC)."""
    t = time.gmtime(ts)
    iso_week = int(time.strftime("%G%V", t))  # ISO year+week as int (e.g., 202401)
    return t.tm_year, t.tm_mon, iso_week, t.tm_yday


# ------------------------------- Backtester -----------------------------------

class Backtester:
    """
    Replays DataSet bars into ExecutionAgent and StrategyAgent.
    """

    def __init__(self, dataset: DataSet, execution: ExecutionAgent, strategy_agent, cfg: Optional[BacktestConfig] = None): # type: ignore
        self.ds = dataset
        self.x = execution
        self.sa = strategy_agent
        self.cfg = cfg or BacktestConfig()
        self._last_fills_len = 0

        # prepare fills log
        self._fills_file = None
        if self.cfg.fills_log_path:
            d = os.path.dirname(self.cfg.fills_log_path)
            if d:
                os.makedirs(d, exist_ok=True)
            self._fills_file = open(self.cfg.fills_log_path, "a", encoding="utf-8")

    def __del__(self):
        try:
            if self._fills_file:
                self._fills_file.close()
        except Exception:
            pass

    # --------------- main loop ----------------

    def run(self) -> Dict:
        n = len(self.ds.calendar)
        if n == 0:
            return {"equity_curve": [], "returns": [], "metrics": {}, "bars": 0}

        start = self.cfg.start_bar if self.cfg.start_bar is not None else 0
        end = self.cfg.end_bar if self.cfg.end_bar is not None else (n - 1)
        start = max(0, int(start))
        end = min(n - 1, int(end))
        if start > end:
            return {"equity_curve": [], "returns": [], "metrics": {}, "bars": 0}

        equity_curve: List[Tuple[float, float]] = []  # (ts, equity)
        returns: List[float] = []

        # Track schedule state for weekly/monthly
        last_week = last_month = None

        # Warmup counter
        bars_seen = 0

        # Prime last prices for all symbols at the first bar (so limits can be marketable)
        # We'll set prices iteratively below; this is just a safety no-op.

        for t in range(start, end + 1):
            ts = float(self.ds.calendar[t])

            # 1) Feed this bar's prices into execution & strategies
            cross = {s: self.ds.data[s][t] for s in self.ds.data.keys()}
            for sym, row in cross.items():
                # Use 'close' as the bar price proxy
                px = float(row.get("close", 0.0))
                if px <= 0:
                    continue
                self.sa.on_price(sym, px)  # forwards into ExecutionAgent.update_price(...)

            # 2) Rebalance if schedule allows and warmup satisfied
            bars_seen += 1
            if bars_seen > self.cfg.warmup_bars and self._should_rebalance(t, ts, last_week, last_month):
                self.sa.maybe_rebalance(force=True)

            # Update schedule trackers
            _, month, iso_week, _ = _ts_to_ymd(ts)
            last_month = month
            last_week = iso_week

            # 3) Mark-to-market (ExecutionAgent computes PnL off last prices)
            if self.cfg.mark_to_market_each_bar and hasattr(self.x, "mark_to_market"):
                try:
                    self.x.mark_to_market()
                except Exception:
                    pass

            # 4) Capture fills since last bar (optional log)
            self._log_new_fills(ts)

            # 5) Record equity & returns
            eq = float(self.x.equity() if hasattr(self.x, "equity") else 0.0)
            equity_curve.append((ts, eq))
            if len(equity_curve) >= 2:
                r = _pct_change(equity_curve[-2][1], equity_curve[-1][1])
                returns.append(r)

            # Progress
            if self.cfg.print_progress and (t - start) % max(1, (end - start) // 10 or 1) == 0:
                print(f"[BT] {100*(t-start)/(end-start+1):5.1f}%  t={t}/{end}  eq={eq:,.2f}")

        # Metrics
        curve_vals = [v for _, v in equity_curve]
        mdd, mdd_s, mdd_e = _max_drawdown(curve_vals) if curve_vals else (0.0, 0, 0)
        sharpe = _sharpe(returns)
        hit = _safe_div(sum(1 for r in returns if r > 0), len(returns))
        avg = statistics.mean(returns) if returns else 0.0
        vol = statistics.pstdev(returns) if len(returns) > 1 else 0.0

        out = {
            "equity_curve": equity_curve,
            "returns": returns,
            "metrics": {
                "sharpe": sharpe,
                "avg_return": avg,
                "vol": vol,
                "hit_rate": hit,
                "max_drawdown": mdd,
                "mdd_start_idx": mdd_s,
                "mdd_end_idx": mdd_e,
                "bars": len(equity_curve),
            },
        }
        return out

    # --------------- helpers ----------------

    def _should_rebalance(self, t: int, ts: float, last_week: Optional[int], last_month: Optional[int]) -> bool:
        # If cadence_bars provided, use that
        if self.cfg.cadence_bars and self.cfg.cadence_bars > 0:
            return (t % int(self.cfg.cadence_bars)) == 0

        mode = (self.cfg.rebalance or "daily").lower()
        if mode == "none":
            return False
        if mode == "daily":
            return True
        if mode == "weekly":
            _, _, iso_week, _ = _ts_to_ymd(ts)
            return (last_week is None) or (iso_week != last_week)
        if mode == "monthly":
            _, month, _, _ = _ts_to_ymd(ts)
            return (last_month is None) or (month != last_month)
        # fallback
        return True

    def _log_new_fills(self, ts: float) -> None:
        if not self._fills_file:
            # still advance pointer to avoid re-processing on later call
            try:
                fills = self.x.fills() if hasattr(self.x, "fills") else []
                self._last_fills_len = len(fills)
            except Exception:
                pass
            return

        try:
            fills: List[Fill] = self.x.fills()  # type: ignore
        except Exception:
            return

        for i in range(self._last_fills_len, len(fills)):
            f = fills[i]
            rec = {
                "t": ts,
                "order_id": getattr(f.order_id, "val", str(getattr(f, "order_id", ""))), # type: ignore
                "symbol": getattr(f, "symbol", ""),
                "side": getattr(getattr(f, "side", ""), "name", str(getattr(f, "side", ""))),
                "qty": float(getattr(f, "qty", 0.0)),
                "price": float(getattr(f, "price", 0.0)),
                "fee": float(getattr(f, "fee", 0.0)),
            }
            self._fills_file.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._fills_file.flush()
        self._last_fills_len = len(fills)


# ------------------------------- CLI smoke test --------------------------------

if __name__ == "__main__":
    # Minimal end-to-end smoke test with synthetic data
    from simulators.envs.data_loader import MarketDataLoader, LoadSpec
    from agents.execution_agent import ExecutionAgent, Side, OrderType
    from agents.strategy_agent import StrategyAgent, StrategyBase, RebalanceConfig

    # Build a tiny synthetic dataset (two symbols, ~200 bars)
    import random
    random.seed(7)

    def synth(n: int, start: float):
        x = start
        rows = []
        t0 = 1_700_000_000  # arbitrary epoch
        for i in range(n):
            x *= (1.0 + random.uniform(-0.01, 0.01))
            rows.append({"ts": t0 + i*86400, "open": x, "high": x, "low": x, "close": x, "volume": 1_000_000})
        return rows

    ds = type("DS", (), {})()  # quick stub instead of reading files
    ds.data = {"AAA": synth(220, 100.0), "BBB": synth(220, 50.0)} # type: ignore
    # align calendar by intersection of ts
    cal = sorted(set(r["ts"] for r in ds.data["AAA"]) & set(r["ts"] for r in ds.data["BBB"])) # type: ignore
    ds.data = {s: [r for r in rows if r["ts"] in set(cal)] for s, rows in ds.data.items()} # type: ignore
    ds.calendar = cal # type: ignore

    # Simple demo strategy: buy higher momentum, short lower momentum (z-scored diff)
    class MomStrat(StrategyBase):
        name = "mom"
        max_positions = 2
        gross_target = 0.5
        def __init__(self, lookback: int = 5):
            self.lookback = lookback
            self.history: Dict[str, List[float]] = {}
        def on_price(self, symbol: str, price: float) -> None:
            self.history.setdefault(symbol, []).append(price)
        def generate_signals(self, now_ts: float):
            scores = {}
            for s, hist in self.history.items():
                if len(hist) < self.lookback + 1:
                    continue
                scores[s] = (hist[-1] / (hist[-self.lookback-1] or 1.0)) - 1.0
            return scores

    x = ExecutionAgent(starting_cash=1_000_000.0)
    sa = StrategyAgent(x, RebalanceConfig(cadence_sec=0))
    sa.register(MomStrat(lookback=5), weight=1.0)

    bt = Backtester(ds, x, sa, BacktestConfig(rebalance="daily", warmup_bars=10, print_progress=False))
    res = bt.run()
    print(json.dumps(res["metrics"], separators=(",", ":"), default=float))