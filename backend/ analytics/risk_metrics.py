# backend/analytics/risk_metrics.py
"""
Risk Metrics Engine:
- Computes volatility, Sharpe, Sortino, drawdowns, VaR, exposure
- Works on rolling PnL or returns series
- Splits by strategy, region if tagged

Usage:
    rm = RiskMetrics(window=252)  # daily, 1 year window
    rm.update(pnl=cur_pnl, equity=cur_equity, strategy="mean_rev")

    snap = rm.snapshot()
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple


class RiskMetrics:
    def __init__(self, window: int = 252, risk_free_rate: float = 0.02):
        """
        Args:
            window: lookback window in observations (252 = 1 year of daily)
            risk_free_rate: annual risk-free rate for Sharpe calc
        """
        self.window = window
        self.rf = risk_free_rate

        # rolling storage per strategy
        self.pnl_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.equity_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, pnl: float, equity: float, strategy: str = "total"):
        """Add latest pnl + equity point for a given strategy."""
        self.pnl_history[strategy].append(pnl)
        self.equity_history[strategy].append(equity)

    def _calc_metrics(self, pnl_series, equity_series) -> Dict[str, float]:
        arr = np.array(pnl_series, dtype=float)
        eq  = np.array(equity_series, dtype=float)

        if len(arr) < 2:
            return {"sharpe": 0.0, "sortino": 0.0, "vol": 0.0,
                    "max_drawdown": 0.0, "cvar": 0.0, "var": 0.0}

        # returns from pnl/equity
        rets = arr / np.clip(eq[:-1], 1e-9, None)
        mean_ret = np.mean(rets)
        std_ret  = np.std(rets)

        downside = rets[rets < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-9

        sharpe  = (mean_ret - self.rf/252) / (std_ret + 1e-9) * np.sqrt(252)
        sortino = (mean_ret - self.rf/252) / (downside_std + 1e-9) * np.sqrt(252)

        # max drawdown
        cum_eq = np.cumsum(rets)
        highwater = np.maximum.accumulate(cum_eq)
        dd = np.min(cum_eq - highwater)

        # parametric VaR 95%
        var = np.percentile(rets, 5)
        cvar = np.mean(rets[rets <= var]) if np.any(rets <= var) else var

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "vol": float(std_ret * np.sqrt(252)),
            "max_drawdown": float(dd),
            "var_95": float(var),
            "cvar_95": float(cvar),
        }

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return risk metrics per strategy + total."""
        out = {}
        for strat, pnl_hist in self.pnl_history.items():
            out[strat] = self._calc_metrics(pnl_hist, self.equity_history[strat])
        return out