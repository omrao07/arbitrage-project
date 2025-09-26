# simulation-farm/utils/metrics.py
"""
metrics.py
==========

Reusable performance + risk metrics for Simulation Farm jobs.

Dependencies:
    pip install numpy pandas
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


# ---------------- Basic KPIs ----------------

def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    if equity.empty: return 0.0
    start, end = equity.iloc[0], equity.iloc[-1]
    years = len(equity) / periods_per_year
    return (end / start) ** (1 / years) - 1 if start > 0 else 0.0


def sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if returns.empty: return 0.0
    excess = returns - (risk_free / periods_per_year)
    mu, sd = excess.mean(), excess.std(ddof=1)
    return (mu * periods_per_year) / (sd + 1e-12)


def sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio (uses downside deviation)."""
    if returns.empty: return 0.0
    excess = returns - (risk_free / periods_per_year)
    downside = returns[returns < 0]
    dwn = downside.std(ddof=1) if len(downside) > 1 else 0.0
    mu = excess.mean()
    return (mu * periods_per_year) / (dwn + 1e-12)


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized volatility."""
    if returns.empty: return 0.0
    return returns.std(ddof=1) * math.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (fraction)."""
    if equity.empty: return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min()


def calmar(equity: pd.Series, returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio = CAGR / |MaxDD|."""
    dd = abs(max_drawdown(equity))
    return cagr(equity, periods_per_year) / (dd + 1e-12)


# ---------------- Tail risk ----------------

def var_cvar(returns: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    """
    Value-at-Risk and Conditional VaR (Expected Shortfall).
    Returns as negative fractions (loss).
    """
    if returns.empty: return (0.0, 0.0)
    q = returns.quantile(1 - level)
    tail = returns[returns <= q]
    cvar = tail.mean() if not tail.empty else q
    return float(q), float(cvar)


# ---------------- Turnover / exposure ----------------

def turnover(weights: pd.DataFrame) -> float:
    """
    Average turnover per rebalance period given weights DataFrame (time x assets).
    Assumes rows sum to 1.0 (portfolio fully invested).
    """
    if weights.shape[0] < 2: return 0.0
    diff = weights.diff().abs().sum(axis=1).dropna()
    return diff.mean()


def exposure(weights: pd.DataFrame) -> Dict[str, float]:
    """
    Simple exposures summary: long%, short%, net%.
    """
    long = weights[weights > 0].sum(axis=1).mean()
    short = -weights[weights < 0].sum(axis=1).mean()
    return {"long": float(long), "short": float(short), "net": float(long - short)}


# ---------------- Aggregator ----------------

def compute_all_metrics(
    equity: pd.Series,
    returns: pd.Series,
    weights: Optional[pd.DataFrame] = None,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
    var_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compute a standard set of KPIs for reporting.
    """
    return {
        "cagr": cagr(equity, periods_per_year),
        "sharpe": sharpe(returns, risk_free, periods_per_year),
        "sortino": sortino(returns, risk_free, periods_per_year),
        "vol_annual": volatility(returns, periods_per_year),
        "max_dd": max_drawdown(equity),
        "calmar": calmar(equity, returns, periods_per_year),
        "var": var_cvar(returns, var_level)[0],
        "cvar": var_cvar(returns, var_level)[1],
        "turnover": turnover(weights) if weights is not None else None, # type: ignore
    }