# utils/metrics.py
"""
Utility functions for computing evaluation metrics.

Includes:
- Trading metrics (PnL, Sharpe ratio, Max Drawdown, Sortino ratio, CAGR)
- Information retrieval metrics (precision, recall, F1, MRR, nDCG)
- General-purpose statistical utilities

Author: Your Project
"""

from __future__ import annotations
import numpy as np
from typing import List, Sequence


# ------------------------ Trading Metrics ------------------------

def cumulative_pnl(returns: Sequence[float]) -> float:
    """Compute cumulative PnL (sum of returns)."""
    return float(np.nansum(returns))


def sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio."""
    r = np.asarray(returns, dtype=float)
    excess = r - risk_free_rate
    mean = np.nanmean(excess)
    std = np.nanstd(excess)
    if std == 0:
        return 0.0
    return float(mean / std * np.sqrt(252))  # daily → annualized


def sortino_ratio(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio (downside risk only)."""
    r = np.asarray(returns, dtype=float)
    excess = r - risk_free_rate
    downside = r[r < 0]
    dd = np.nanstd(downside) if downside.size > 0 else np.nan
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(np.nanmean(excess) / dd * np.sqrt(252))


def max_drawdown(returns: Sequence[float]) -> float:
    """Compute maximum drawdown."""
    r = np.asarray(returns, dtype=float)
    cum = np.nancumsum(r)
    highwater = np.maximum.accumulate(cum)
    dd = cum - highwater
    return float(np.min(dd))


def cagr(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate (CAGR)."""
    r = np.asarray(returns, dtype=float)
    cum = np.nancumsum(r)
    total = cum[-1] if cum.size > 0 else 0.0
    years = len(r) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total) ** (1 / years) - 1)


# ------------------------ Retrieval Metrics ------------------------

def precision_at_k(relevant: set, retrieved: List[str], k: int) -> float:
    """Precision@k: proportion of top-k retrieved that are relevant."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    rel = sum(1 for x in retrieved_k if x in relevant)
    return rel / k


def recall_at_k(relevant: set, retrieved: List[str], k: int) -> float:
    """Recall@k: proportion of relevant items retrieved in top-k."""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    rel = sum(1 for x in retrieved_k if x in relevant)
    return rel / len(relevant)


def f1_at_k(relevant: set, retrieved: List[str], k: int) -> float:
    """F1@k: harmonic mean of precision and recall at k."""
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mean_reciprocal_rank(relevant: set, retrieved: List[str]) -> float:
    """MRR: mean reciprocal rank of the first relevant item."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set, retrieved: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (nDCG@k).
    Binary relevance: 1 if item in relevant set, else 0.
    """
    def dcg(rel_list: List[int]) -> float:
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rel_list))

    retrieved_k = retrieved[:k]
    rel_list = [1 if x in relevant else 0 for x in retrieved_k]
    dcg_val = dcg(rel_list)
    ideal_rel = sorted(rel_list, reverse=True)
    idcg_val = dcg(ideal_rel)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# ------------------------ Convenience ------------------------

def summarize_trading_metrics(returns: Sequence[float]) -> dict:
    """Return a dictionary with common trading metrics."""
    return {
        "cumulative_pnl": cumulative_pnl(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "cagr": cagr(returns),
    }