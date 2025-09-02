# research/risk/var_es.py
"""
Value-at-Risk (VaR) and Expected Shortfall (ES / CVaR).

Supports:
  - Historical simulation (non-parametric)
  - Parametric Gaussian approximation
  - Flexible confidence levels
  - Portfolio P&L series or returns

All returns/PNL assumed in same currency.

VaR is reported as a positive number = loss threshold (e.g., 0.05 means 5% loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Literal


Method = Literal["historical", "gaussian"]


def _check_conf(conf: float) -> float:
    if conf <= 0.0 or conf >= 1.0:
        raise ValueError("confidence must be in (0,1)")
    return conf


def var_es(
    pnl: np.ndarray | pd.Series,
    conf: float = 0.99,
    method: Method = "historical",
) -> Dict[str, float]:
    """
    Compute one-sided left-tail Value-at-Risk (VaR) and Expected Shortfall (ES).

    Parameters
    ----------
    pnl : array-like
        Profit & loss series (can be returns). Positive = gain, negative = loss.
    conf : float, default 0.99
        Confidence level (e.g., 0.99 = 99%).
    method : "historical" or "gaussian"
        - historical: empirical quantiles
        - gaussian: mean + sigma * quantile

    Returns
    -------
    dict with:
      - 'VaR' : Value-at-Risk (positive loss threshold)
      - 'ES'  : Expected Shortfall (average loss beyond VaR)
      - 'conf': confidence level
      - 'method': method used
      - 'mean': mean PnL
      - 'std': standard deviation
      - 'n': sample size
    """
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    n = len(pnl)
    if n == 0:
        raise ValueError("empty pnl series")

    conf = _check_conf(conf)
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))

    alpha = 1.0 - conf

    if method == "historical":
        # empirical quantile (left-tail)
        var_q = np.quantile(pnl, alpha)
        var_val = -var_q
        es_val = -pnl[pnl <= var_q].mean() if np.any(pnl <= var_q) else var_val

    elif method == "gaussian":
        from scipy.stats import norm

        z = norm.ppf(alpha)
        var_q = mean + std * z
        var_val = -var_q
        es_val = -(mean + std * (norm.pdf(z) / alpha))

    else:
        raise ValueError(f"unsupported method {method}")

    return {
        "VaR": float(var_val),
        "ES": float(es_val),
        "conf": conf,
        "method": method, # type: ignore
        "mean": mean,
        "std": std,
        "n": n,
    }


# ---------------- Example usage ----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulate PnL series
    rng = np.random.default_rng(42)
    pnl = rng.normal(loc=0.001, scale=0.02, size=10_000)

    out_h = var_es(pnl, conf=0.99, method="historical")
    out_g = var_es(pnl, conf=0.99, method="gaussian")

    print("Historical:", out_h)
    print("Gaussian:", out_g)

    # Visualize tail
    plt.hist(pnl, bins=100, density=True, alpha=0.5)
    plt.axvline(-out_h["VaR"], color="r", linestyle="--", label=f"VaR@{out_h['conf']:.2f}")
    plt.legend()
    plt.show()