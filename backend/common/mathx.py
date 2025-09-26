"""
mathx.py
---------
Extended math & finance utilities for quant strategies.
Provides safe wrappers around common stats, finance, and matrix operations.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional


# ---------- Basic Safe Math ----------

def safe_div(a: float, b: float, default: float = np.nan) -> float:
    """Safe divide (returns default if denominator is 0 or NaN)."""
    try:
        if b == 0 or pd.isna(b):
            return default
        return a / b
    except Exception:
        return default


def pct_change(new: float, old: float, default: float = np.nan) -> float:
    """Percent change (new/old - 1)."""
    return safe_div(new - old, old, default)


def zscore(series: Union[pd.Series, np.ndarray], window: Optional[int] = None) -> pd.Series:
    """
    Rolling or full-sample z-score.
    Args:
        series: pandas Series or numpy array
        window: optional rolling window
    """
    s = pd.Series(series)
    if window:
        mean = s.rolling(window).mean()
        std = s.rolling(window).std(ddof=0)
    else:
        mean = s.mean()
        std = s.std(ddof=0)
    return (s - mean) / std


# ---------- Portfolio / Risk ----------

def sharpe_ratio(returns: Union[pd.Series, np.ndarray], rf: float = 0.0, freq: int = 252) -> float:
    """
    Annualized Sharpe ratio.
    Args:
        returns: return series
        rf: risk-free rate (per period)
        freq: periods per year (252 = daily, 12 = monthly)
    """
    r = pd.Series(returns).dropna()
    excess = r - rf
    mu = r.mean() * freq
    sigma = r.std(ddof=0) * np.sqrt(freq)
    return safe_div(mu, sigma)


def sortino_ratio(returns: Union[pd.Series, np.ndarray], rf: float = 0.0, freq: int = 252) -> float:
    """Annualized Sortino ratio (uses downside deviation)."""
    r = pd.Series(returns).dropna()
    excess = r - rf
    downside = r[r < 0].std(ddof=0) * np.sqrt(freq)
    mu = excess.mean() * freq
    return safe_div(mu, downside)


def max_drawdown(series: Union[pd.Series, np.ndarray]) -> float:
    """Compute max drawdown from cumulative returns or PnL series."""
    s = pd.Series(series).dropna()
    peak = s.cummax()
    dd = (s - peak) / peak
    return dd.min()


def volatility(returns: Union[pd.Series, np.ndarray], freq: int = 252) -> float:
    """Annualized volatility."""
    r = pd.Series(returns).dropna()
    return r.std(ddof=0) * np.sqrt(freq)


# ---------- Finance-Specific ----------

def wacc(cost_of_equity: float, cost_of_debt: float, equity_value: float, debt_value: float, tax_rate: float) -> float:
    """Weighted Average Cost of Capital (WACC)."""
    v = equity_value + debt_value
    if v == 0: 
        return np.nan
    we, wd = equity_value / v, debt_value / v
    return we * cost_of_equity + wd * cost_of_debt * (1 - tax_rate)


def npv(cashflows: List[float], discount_rate: float) -> float:
    """Net Present Value of a list of cashflows given a discount rate."""
    return sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cashflows, start=1))


def irr(cashflows: List[float], guess: float = 0.1) -> float:
    """Internal Rate of Return (IRR)."""
    return np.irr(cashflows) # type: ignore


def monte_carlo_sim(mu: float, sigma: float, T: int, n: int = 1000, s0: float = 1.0) -> np.ndarray:
    """
    Monte Carlo simulation for geometric Brownian motion.
    Args:
        mu: drift
        sigma: volatility
        T: time horizon (steps)
        n: number of paths
        s0: initial price
    Returns:
        array of shape (n, T+1)
    """
    dt = 1.0 / T
    paths = np.zeros((n, T+1))
    paths[:, 0] = s0
    for t in range(1, T+1):
        z = np.random.standard_normal(n)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths


# ---------- Correlation / Factor Analysis ----------

def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix with NaN safety."""
    return df.corr().fillna(0)


def factor_ic(factor: pd.Series, forward_returns: pd.Series) -> float:
    """
    Information Coefficient (IC): correlation between factor scores and next-period returns.
    """
    aligned = pd.concat([factor, forward_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return aligned.corr().iloc[0, 1] # type: ignore


# ---------- Example Usage ----------
if __name__ == "__main__":
    rets = np.random.normal(0.001, 0.02, 252)
    print("Sharpe:", sharpe_ratio(rets))
    print("Sortino:", sortino_ratio(rets))
    print("Max DD:", max_drawdown(np.cumprod(1+rets)))
    print("Vol:", volatility(rets))
    print("NPV:", npv([100, 100, 100], 0.1))
    print("IC demo:", factor_ic(pd.Series(np.random.randn(100)), pd.Series(np.random.randn(100))))