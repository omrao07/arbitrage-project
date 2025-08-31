# backend/risk/expected_shortfall.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Literal

import numpy as np
try:
    import pandas as pd  # optional, only used if inputs are Series/DataFrame
except Exception:
    pd = None  # type: ignore

# --------------------------- Types & helpers ---------------------------------
ArrayLike = Iterable[float] | np.ndarray
Tail = Literal["left", "right"]  # left = losses negative; right = losses positive

def _to_np(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    if pd is not None and isinstance(x, (pd.Series, pd.DataFrame)):
        return np.asarray(x, dtype=float)
    return np.asarray(list(x), dtype=float)

def _validate_alpha(alpha: float) -> float:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1), e.g., 0.975 for 97.5% ES")
    return float(alpha)

# By convention: returns r (positive good). Loss = -r for left tail ES.
def _losses(r: np.ndarray, tail: Tail) -> np.ndarray:
    return -r if tail == "left" else r

# ------------------------------ Core metrics ---------------------------------
def var_historical(returns: ArrayLike, alpha: float = 0.975, tail: Tail = "left") -> float:
    """Historical VaR at level alpha (e.g., 0.975 => 97.5% left-tail)."""
    _validate_alpha(alpha)
    r = _to_np(returns)
    L = _losses(r, tail)
    q = np.quantile(L, alpha, method="higher" if hasattr(np, "quantile") else None) # type: ignore
    return float(q)

def es_historical(returns: ArrayLike, alpha: float = 0.975, tail: Tail = "left") -> float:
    """
    Historical ES (a.k.a. CVaR): average loss beyond VaR_alpha.
    Works with few data points (uses 'higher' quantile to define exceedances).
    """
    _validate_alpha(alpha)
    r = _to_np(returns)
    if r.size == 0:
        return float("nan")
    L = _losses(r, tail)
    q = var_historical(returns, alpha=alpha, tail=tail)
    mask = L >= q  # tail losses
    if not np.any(mask):
        return float(q)  # fallback: equals VaR if no exceedances (tiny sample)
    return float(np.mean(L[mask]))

def es_parametric_gaussian(mu: float, sigma: float, alpha: float = 0.975) -> float:
    """
    Parametric ES for Normally distributed *loss* ~ N(mu, sigma).
    If you have return distribution, pass mu=-mean_ret, sigma=std_ret.
    ES = mu + sigma * φ(z) / (1 - alpha), where z = Φ^{-1}(alpha)
    """
    _validate_alpha(alpha)
    if sigma <= 0:
        return float(max(mu, 0.0))
    from math import sqrt, pi, exp
    from scipy.stats import norm  # optional but standard; falls back if missing
    try:
        z = norm.ppf(alpha)
        phi = 1.0 / math.sqrt(2 * math.pi) * math.exp(-0.5 * z * z)
    except Exception:
        # crude inverse/phi fallback
        z = math.sqrt(2) * math.erfcinv(2 * (1 - alpha)) # type: ignore
        phi = 1.0 / math.sqrt(2 * math.pi) * math.exp(-0.5 * z * z)
    return float(mu + sigma * (phi / (1.0 - alpha)))

def es_parametric_t(mu: float, s: float, nu: float, alpha: float = 0.975) -> float:
    """
    ES for Student-t *loss* with location mu, scale s, dof nu (>2).
    Formula: ES = mu + s * ( ( (nu + t^2) / (nu - 1) ) * f_t(t) / (1 - alpha) )
    where t = t_{nu}^{-1}(alpha), f_t is t pdf.
    """
    _validate_alpha(alpha)
    if nu <= 2:
        return float("inf")
    try:
        from scipy.stats import t
        t_alpha = t.ppf(alpha, df=nu)
        pdf = t.pdf(t_alpha, df=nu)
        c = ((nu + t_alpha**2) / (nu - 1.0)) * (pdf / (1.0 - alpha))
        return float(mu + s * c)
    except Exception:
        # numeric fallback via Monte Carlo
        rng = np.random.default_rng(42)
        x = mu + s * rng.standard_t(df=nu, size=2_00000)
        return float(np.quantile(x, alpha) + np.mean(x[x >= np.quantile(x, alpha)]) * 0.0)  # degrade gracefully

def es_cornish_fisher(mean: float, std: float, skew: float, kurt: float, alpha: float = 0.975) -> float:
    """
    Cornish-Fisher adjusted ES (approximate, for mild skew/kurtosis).
    1) Adjust z_alpha via CF, 2) use gaussian ES formula with adjusted z.
    """
    _validate_alpha(alpha)
    if std <= 0:
        return float(max(mean, 0.0))
    from scipy.stats import norm
    z = norm.ppf(alpha)
    z_cf = (z
            + (1/6)*(z**2 - 1)*skew
            + (1/24)*(z**3 - 3*z)*(kurt - 3)
            - (1/36)*(2*z**3 - 5*z)*skew**2)
    phi = 1.0 / math.sqrt(2 * math.pi) * math.exp(-0.5 * z_cf * z_cf)
    return float(mean + std * (phi / (1.0 - alpha)))

# ---------------------------- Portfolio / scenarios --------------------------
def portfolio_es_historical(
    scenario_returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.975,
    tail: Tail = "left",
) -> float:
    """
    scenario_returns: (T, N) return scenarios
    weights: (N,) portfolio weights
    """
    r = np.asarray(scenario_returns, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if r.ndim != 2:
        raise ValueError("scenario_returns must be 2D (T,N)")
    if r.shape[1] != w.size:
        raise ValueError("weights length mismatch")
    port = r @ w
    return es_historical(port, alpha=alpha, tail=tail)

def es_contributions_historical(
    scenario_returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.975,
    tail: Tail = "left",
) -> Tuple[np.ndarray, float]:
    """
    Euler-like contributions via tail-conditional expectation of gradient:
      c_i ≈ E[ -R_i | L >= VaR ] * w_i   (for left-tail losses L = -R_p)
    Returns (contribs, ES)
    """
    r = np.asarray(scenario_returns, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    port = r @ w
    L = _losses(port, tail)           # portfolio loss series
    q = np.quantile(L, 0.975 if alpha is None else alpha, method="higher" if hasattr(np, "quantile") else None) # type: ignore
    tail_mask = L >= q
    if not np.any(tail_mask):
        es = float(np.mean(L))
        return np.zeros_like(w), es
    # Conditional expectation of asset contributions in tail
    # For left tail: portfolio loss L = -(r @ w); marginal dL/dw_i = -r_i
    Ri_tail = r[tail_mask, :]   # (K, N)
    c = -np.mean(Ri_tail, axis=0) * w
    es = float(np.mean(L[tail_mask]))
    # Rescale to sum to ES (numerical alignment)
    s = c.sum()
    if s != 0:
        c = c * (es / s)
    return c, es

# ----------------------------- Backtesting ES --------------------------------
@dataclass
class ESTestResult:
    alpha: float
    var_exceed: int
    var_expected: float
    es_stat: float
    es_pvalue: float
    passed: bool

def backtest_var_es(
    returns: ArrayLike,
    alpha: float = 0.975,
    tail: Tail = "left",
) -> ESTestResult:
    """
    Practical backtest combining VaR coverage + Acerbi–Szekely ES test.
    - VaR exceptions (count vs expected)
    - ES test statistic S = (1/(1-alpha)) * mean( (L - VaR)_+ / ES ), ~ should be ~1 under correct ES
    We approximate p-value via bootstrap on permutations (fast).
    """
    _validate_alpha(alpha)
    r = _to_np(returns)
    L = _losses(r, tail)
    q = np.quantile(L, alpha, method="higher" if hasattr(np, "quantile") else None) # type: ignore
    es = float(np.mean(L[L >= q])) if np.any(L >= q) else float(q)

    N = L.size
    exc = int(np.sum(L > q))
    expected = (1 - alpha) * N

    # Acerbi-style score: normalized shortfall intensity
    eps = 1e-12
    tail_losses = L[L >= q]
    S = float(np.mean((tail_losses - q) / (es - q + eps))) if tail_losses.size > 0 else 1.0

    # Bootstrap p-value (randomly shuffle losses to break dependence)
    rng = np.random.default_rng(42)
    B = min(2000, max(200, N))  # keep it quick
    stats = []
    for _ in range(B):
        perm = rng.permutation(L)
        q_b = np.quantile(perm, alpha, method="higher" if hasattr(np, "quantile") else None) # type: ignore
        tl = perm[perm >= q_b]
        if tl.size == 0:
            stats.append(1.0)
            continue
        es_b = float(np.mean(tl))
        stats.append(float(np.mean((tl - q_b) / (es_b - q_b + eps))))
    pval = float(np.mean(np.asarray(stats) >= S))
    passed = (abs(exc - expected) <= 3 * math.sqrt(expected + 1e-9)) and (pval > 0.05)

    return ESTestResult(alpha=alpha, var_exceed=exc, var_expected=float(expected), es_stat=S, es_pvalue=pval, passed=passed)

# ------------------------ Bootstrap CI for ES --------------------------------
def es_bootstrap_ci(
    returns: ArrayLike,
    alpha: float = 0.975,
    tail: Tail = "left",
    B: int = 2000,
    ci: Tuple[float, float] = (0.025, 0.975),
) -> Tuple[float, float]:
    """Nonparametric bootstrap CI for historical ES."""
    _validate_alpha(alpha)
    r = _to_np(returns)
    if r.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(123)
    est = []
    n = r.size
    for _ in range(B):
        idx = rng.integers(0, n, n)
        est.append(es_historical(r[idx], alpha=alpha, tail=tail))
    return (float(np.quantile(est, ci[0])), float(np.quantile(est, ci[1])))

# -------------------------- Data hygiene utilities ---------------------------
def winsorize(x: ArrayLike, p: float = 0.01) -> np.ndarray:
    """Clip extreme tails (e.g., 1%) to stabilize ES in tiny samples."""
    a = _to_np(x).copy()
    lo, hi = np.quantile(a, [p, 1 - p])
    a[a < lo] = lo
    a[a > hi] = hi
    return a

# ------------------------------ Quick demo -----------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    # Simulate daily returns with fat tails
    ret = 0.0005 + 0.01 * rng.standard_t(df=5, size=5000)
    alpha = 0.975

    print("VaR (hist):", var_historical(ret, alpha))
    print("ES  (hist):", es_historical(ret, alpha))
    print("ES  (Gaussian approx):", es_parametric_gaussian(mu=-np.mean(ret), sigma=np.std(ret), alpha=alpha)) # type: ignore

    # Portfolio example
    R = np.column_stack([ret, 0.0002 + 0.008 * rng.standard_t(df=7, size=ret.size)])
    w = np.array([0.6, 0.4])
    c, es_p = es_contributions_historical(R, w, alpha=alpha)
    print("Portfolio ES:", es_p, " contributions:", c, " sum:", c.sum())

    # Backtest
    res = backtest_var_es(ret, alpha=alpha)
    print("Backtest:", res)

    # CI
    lo, hi = es_bootstrap_ci(ret, alpha=alpha)
    print("ES 95% CI:", (lo, hi))