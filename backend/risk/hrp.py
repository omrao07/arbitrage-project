# backend/analytics/hrp.py
"""
Hierarchical Risk Parity (HRP) portfolio allocation.

Implements the algorithm described by Marcos López de Prado (2016):
- Uses hierarchical clustering of assets by correlation distance
- Recursively allocates risk across clusters
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


# =============================================================================
# Core helpers
# =============================================================================

def correl_distance(corr: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to a distance matrix (0 = perfect corr, 1 = none).
    dist_ij = sqrt(0.5 * (1 - corr_ij))
    """
    dist = np.sqrt(0.5 * (1 - corr))
    return dist


def quasi_diag(link: np.ndarray) -> List[int]:
    """
    Quasi-diagonalize linkage output from hierarchical clustering.
    Returns the order of indices.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # number of original items in final cluster
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # type: ignore # make space
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = sort_ix.append(df1)
        sort_ix = sort_ix.sort_index()
        sort_ix = sort_ix.reset_index(drop=True)
    return sort_ix.tolist()


def get_cluster_var(cov: np.ndarray, items: List[int]) -> float:
    """
    Compute variance of a cluster (based on weights ∝ 1/variance).
    """
    cov_slice = cov[np.ix_(items, items)]
    inv_diag = 1.0 / np.diag(cov_slice)
    weights = inv_diag / inv_diag.sum()
    cluster_var = np.dot(weights, np.dot(cov_slice, weights))
    return cluster_var


def recursive_bisection(cov: np.ndarray, sort_ix: List[int]) -> np.ndarray:
    """
    Allocate risk recursively down the hierarchy.
    """
    w = pd.Series(1, index=sort_ix)
    clusters = [sort_ix]
    while len(clusters) > 0:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for c in range(0, len(clusters), 2):
            c0 = clusters[c]
            c1 = clusters[c + 1]
            var0 = get_cluster_var(cov, c0)
            var1 = get_cluster_var(cov, c1)
            alpha = 1 - var0 / (var0 + var1)
            w[c0] *= alpha # pyright: ignore[reportArgumentType, reportCallIssue]
            w[c1] *= 1 - alpha # type: ignore
    return w / w.sum() # type: ignore


# =============================================================================
# HRP Allocator
# =============================================================================

@dataclass
class HRPResult:
    weights: pd.Series
    order: List[int]
    linkage: Optional[np.ndarray] = None


class HRPAllocator:
    def __init__(self, method: str = "single"):
        """
        method: linkage method passed to scipy.cluster.hierarchy.linkage
        Options: 'single', 'average', 'complete', 'ward'
        """
        self.method = method

    def allocate(self, returns: pd.DataFrame) -> HRPResult:
        """
        Compute HRP allocation given a return matrix (T x N).
        """
        import scipy.cluster.hierarchy as sch

        cov = returns.cov().values
        corr = returns.corr().values

        dist = correl_distance(corr)
        dist_condensed = sch.distance.squareform(dist, checks=False)
        link = sch.linkage(dist_condensed, method=self.method)
        sort_ix = quasi_diag(link)
        w = recursive_bisection(cov, sort_ix)

        weights = pd.Series(w, index=returns.columns[sort_ix])
        return HRPResult(weights=weights, order=sort_ix, linkage=link)


# =============================================================================
# Tiny demo
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Fake returns
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM", "GS", "XOM"]
    rets = pd.DataFrame(np.random.randn(250, len(tickers)) * 0.01, columns=tickers)

    hrp = HRPAllocator(method="ward")
    res = hrp.allocate(rets)

    print("HRP Weights:")
    print(res.weights.round(4))

    # Optional: visualize dendrogram
    try:
        import scipy.cluster.hierarchy as sch
        sch.dendrogram(res.linkage, labels=tickers)
        plt.show()
    except Exception as e:
        print("Plotting dendrogram failed:", e)