# utils/metrics.py
"""
Performance & Risk Metrics
--------------------------
Reusable functions to evaluate returns, equity curves, and strategies.

Features:
- Return metrics: CAGR, cumulative return, volatility
- Risk metrics: Sharpe, Sortino, Calmar, max drawdown
- Distribution stats: skew, kurtosis
- Attribution: rolling beta/alpha vs benchmark (CAPM-style)
- Convenience: summarize() to collect key stats in a dict

Stdlib-only (uses math, statistics).
"""

from __future__ import annotations
import math
import statistics
from typing import List, Tuple, Dict, Optional


# ---------------------- core return calcs ----------------------

def cumulative_return(returns: List[float]) -> float:
    r = 1.0
    for x in returns:
        r *= (1.0 + x)
    return r - 1.0

def cagr(returns: List[float], periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    tot = cumulative_return(returns)
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return (1.0 + tot) ** (1.0/years) - 1.0

def volatility(returns: List[float], periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return sd * math.sqrt(periods_per_year)

# ---------------------- risk metrics ----------------------

def sharpe(returns: List[float], rf: float = 0.0, periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    mu = statistics.mean(returns) - rf/periods_per_year
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return (mu/sd) * math.sqrt(periods_per_year) if sd else 0.0

def sortino(returns: List[float], rf: float = 0.0, periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    excess = [r - rf/periods_per_year for r in returns]
    downside = [min(0.0, x) for x in excess]
    dd = math.sqrt(sum(d*d for d in downside) / max(1, len(downside)))
    mu = statistics.mean(excess)
    return (mu/dd) * math.sqrt(periods_per_year) if dd else 0.0

def max_drawdown(series: List[float]) -> Tuple[float,int,int]:
    """
    series: equity curve (levels, not returns).
    Returns (mdd, start_idx, end_idx) with mdd as negative fraction.
    """
    peak = -float("inf")
    mdd = 0.0
    s = e = 0
    peak_idx = 0
    for i,v in enumerate(series):
        if v > peak:
            peak = v; peak_idx = i
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd; s = peak_idx; e = i
    return mdd, s, e

def calmar(returns: List[float], equity: Optional[List[float]]=None, periods_per_year: int=252) -> float:
    """
    Calmar ratio = CAGR / |MaxDD|.
    equity optional: equity curve aligned with returns.
    """
    cagr_val = cagr(returns, periods_per_year)
    if equity:
        mdd,_,_ = max_drawdown(equity)
    else:
        # rebuild equity curve from returns
        eq = [1.0]
        for r in returns: eq.append(eq[-1]*(1+r))
        mdd,_,_ = max_drawdown(eq)
    return (cagr_val / abs(mdd)) if mdd < 0 else 0.0

# ---------------------- distribution stats ----------------------

def skewness(returns: List[float]) -> float:
    n = len(returns)
    if n < 3:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    if sd == 0: return 0.0
    return sum((x-mu)**3 for x in returns)/(n*sd**3)

def kurtosis(returns: List[float]) -> float:
    n = len(returns)
    if n < 4:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    if sd == 0: return 0.0
    return sum((x-mu)**4 for x in returns)/(n*sd**4) - 3.0  # excess kurtosis

# ---------------------- attribution (CAPM-style) ----------------------

def rolling_beta_alpha(
    returns: List[float], bench: List[float], rf: float=0.0
) -> Tuple[float,float]:
    """
    Compute CAPM beta/alpha via simple OLS:
        r_p - r_f = alpha + beta*(r_b - r_f) + eps
    """
    n = min(len(returns), len(bench))
    if n < 2:
        return 0.0, 0.0
    rp = returns[:n]; rb = bench[:n]
    ex_rp = [r - rf for r in rp]; ex_rb = [r - rf for r in rb]
    mu_p = statistics.mean(ex_rp); mu_b = statistics.mean(ex_rb)
    cov = sum((ex_rp[i]-mu_p)*(ex_rb[i]-mu_b) for i in range(n))/n
    varb = statistics.pvariance(ex_rb)
    beta = cov/varb if varb>0 else 0.0
    alpha = mu_p - beta*mu_b
    return beta, alpha

# ---------------------- summary ----------------------

def summarize(returns: List[float], equity: Optional[List[float]]=None, rf: float=0.0) -> Dict[str,float]:
    out = {}
    out["cagr"] = cagr(returns)
    out["cum_return"] = cumulative_return(returns)
    out["volatility"] = volatility(returns)
    out["sharpe"] = sharpe(returns, rf)
    out["sortino"] = sortino(returns, rf)
    if equity:
        mdd,_,_ = max_drawdown(equity)
    else:
        eq = [1.0]
        for r in returns: eq.append(eq[-1]*(1+r))
        mdd,_,_ = max_drawdown(eq)
    out["max_drawdown"] = mdd
    out["calmar"] = calmar(returns, equity)
    out["skew"] = skewness(returns)
    out["kurtosis"] = kurtosis(returns)
    return out

# ---------------------- demo ----------------------

if __name__ == "__main__":
    # Quick self-test with synthetic returns
    import random
    random.seed(42)
    rets = [random.gauss(0.0005,0.01) for _ in range(252*3)]
    eq = [1.0]
    for r in rets: eq.append(eq[-1]*(1+r))
    print("Summary:", summarize(rets, eq))