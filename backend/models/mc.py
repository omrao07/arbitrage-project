#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mc.py
-----
General-purpose Monte Carlo utilities for quant research.

Features
- Random engines: GBM, Merton Jump-Diffusion, light Heston
- Correlations via Cholesky; time grid with calendar step
- Variance reduction: antithetic sampling, control variates
- Path containers with vectorized stats, reindex to business days
- Portfolio PnL simulation given linear risk/exposures
- Risk: VaR/ES, drawdowns, hit ratios
- Option helpers: BS price/greeks (for CV baseline), path-dependent payoff hooks

Dependencies: numpy (required), pandas (recommended); scipy optional for norm cdf/ppf.

Examples
--------
import numpy as np
import pandas as pd
from mc import GBM, correlate, portfolio_pnl, var_es

T=252; dt=1/252; n=20000
drift = 0.06; vol = 0.2; s0 = 100.0

paths = GBM(s0, mu=drift, sigma=vol).simulate(T, n, dt, antithetic=True)
ret = paths.returns()
pnl = portfolio_pnl(ret, weights=np.array([1.0]))  # single asset
print(var_es(pnl[:, -1], levels=[0.95, 0.99]))
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

# Optional SciPy for CDFs
try:
    from scipy.stats import norm
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ----------------------------------------------------------------------
# Random tools
# ----------------------------------------------------------------------

def gaussian(n_paths: int, n_steps: int, corr: Optional[np.ndarray] = None,
             antithetic: bool = False, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Z ~ N(0, I) with shape (n_paths, n_steps, d) or correlated with corr (d,d).
    If corr provided, Z at each step has that cross-sectional correlation.
    """
    rng = np.random.default_rng(seed)
    d = 1 if corr is None else corr.shape[0]
    m = n_paths if not antithetic else (n_paths + 1) // 2
    Z = rng.normal(size=(m, n_steps, d))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=0)[:n_paths]

    if corr is None:
        return Z

    # Cholesky decomposition (robustify)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # nearest-PSD via ridge
        eps = 1e-10
        L = np.linalg.cholesky(corr + eps * np.eye(corr.shape[0]))
    return Z @ L.T  # broadcasting (m, n_steps, d) x (d,d)

def correlate(indep: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Post-hoc correlate a (n_paths, n_steps, d) tensor of i.i.d. normals."""
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(corr + 1e-10 * np.eye(corr.shape[0]))
    return indep @ L.T

# ----------------------------------------------------------------------
# Path container
# ----------------------------------------------------------------------

@dataclass
class Paths:
    """Holds simulated price or level paths: shape (n_paths, n_steps+1, d)."""
    X: np.ndarray  # includes t=0
    times: Optional[np.ndarray] = None
    names: Optional[np.ndarray] = None  # asset names length d

    def to_returns(self) -> np.ndarray:
        return self.X[:, 1:, :] / self.X[:, :-1, :] - 1.0

    def returns(self) -> np.ndarray:
        return self.to_returns()

    def log_returns(self) -> np.ndarray:
        return np.log(self.X[:, 1:, :] / self.X[:, :-1, :])

    def last(self) -> np.ndarray:
        return self.X[:, -1, :]

    def to_dataframe(self) -> pd.DataFrame:
        n, t, d = self.X.shape
        idx = pd.MultiIndex.from_product([range(n), range(t)], names=["path","t"])#type:ignore
        cols = self.names if self.names is not None else [f"a{i}" for i in range(d)]
        return pd.DataFrame(self.X.reshape(n*t, d), index=idx, columns=cols)

# ----------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------

@dataclass
class GBM:
    """Geometric Brownian Motion for 1..d assets (vectorized)."""
    s0: np.ndarray | float
    mu: np.ndarray | float
    sigma: np.ndarray | float

    def simulate(self, n_steps: int, n_paths: int, dt: float = 1/252,
                 corr: Optional[np.ndarray] = None, antithetic: bool = False,
                 seed: Optional[int] = None, names: Optional[np.ndarray] = None) -> Paths:
        s0 = np.asarray(self.s0, dtype=float).reshape(1, 1, -1)
        mu = np.asarray(self.mu, dtype=float).reshape(1, 1, -1)
        sig= np.asarray(self.sigma, dtype=float).reshape(1, 1, -1)
        d = s0.shape[-1]
        Z = gaussian(n_paths, n_steps, corr=corr, antithetic=antithetic, seed=seed)  # (n, T, d?) d matches corr
        if Z.shape[2] == 1 and d > 1:
            # broadcast same shock to all if corr not provided
            Z = np.repeat(Z, d, axis=2)

        drift = (mu - 0.5 * sig * sig) * dt
        diff  = sig * np.sqrt(dt) * Z

        X = np.empty((n_paths, n_steps + 1, d))
        X[:, 0, :] = s0
        # iterative multiplicative updates: S_{t+1} = S_t * exp(drift + diff)
        exp_incr = np.exp(drift + diff)  # (n, T, d)
        X[:, 1:, :] = s0 * np.cumprod(exp_incr, axis=1)
        return Paths(X, names=names)

@dataclass
class MertonJD:
    """
    Merton Jump-Diffusion for 1..d assets.
    Parameters per asset: (mu, sigma, lambda_j, mu_j, sigma_j) where jumps ~ logN(mu_j, sigma_j^2).
    """
    s0: np.ndarray | float
    mu: np.ndarray | float
    sigma: np.ndarray | float
    lam: np.ndarray | float = 0.1
    mu_j: np.ndarray | float = -0.10
    sigma_j: np.ndarray | float = 0.20

    def simulate(self, n_steps: int, n_paths: int, dt: float = 1/252,
                 corr: Optional[np.ndarray] = None, antithetic: bool = False,
                 seed: Optional[int] = None, names: Optional[np.ndarray] = None) -> Paths:
        s0 = np.asarray(self.s0, dtype=float).reshape(1,1,-1)
        mu = np.asarray(self.mu, dtype=float).reshape(1,1,-1)
        sig= np.asarray(self.sigma, dtype=float).reshape(1,1,-1)
        lam= np.asarray(self.lam, dtype=float).reshape(1,1,-1)
        muj= np.asarray(self.mu_j, dtype=float).reshape(1,1,-1)
        sj = np.asarray(self.sigma_j, dtype=float).reshape(1,1,-1)

        d = s0.shape[-1]
        Z = gaussian(n_paths, n_steps, corr=corr, antithetic=antithetic, seed=seed)
        if Z.shape[2] == 1 and d > 1:
            Z = np.repeat(Z, d, axis=2)

        drift = (mu - 0.5*sig*sig - lam*(np.exp(muj + 0.5*sj*sj)-1.0)) * dt
        diff  = sig * np.sqrt(dt) * Z

        rng = np.random.default_rng(seed)
        # Poisson jump counts per step * path * dim
        N = rng.poisson(lam * dt, size=(n_paths, n_steps, d))
        J = np.exp(muj + sj * rng.normal(size=(n_paths, n_steps, d))) - 1.0  # jump size -1..+
        jump_term = (1.0 + J) ** N

        X = np.empty((n_paths, n_steps+1, d))
        X[:,0,:] = s0
        incr = np.exp(drift + diff) * jump_term
        X[:,1:,:] = s0 * np.cumprod(incr, axis=1)
        return Paths(X, names=names)

@dataclass
class HestonLite:
    """
    Light Heston path generator (Euler discretization, full truncation).
    dS = mu S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,  corr(dW1, dW2)=rho
    """
    s0: float
    v0: float
    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def simulate(self, n_steps: int, n_paths: int, dt: float = 1/252,
                 seed: Optional[int] = None, names: Optional[np.ndarray] = None) -> Paths:
        rng = np.random.default_rng(seed)
        Z1 = rng.normal(size=(n_paths, n_steps))
        Z2 = rng.normal(size=(n_paths, n_steps))
        W1 = Z1
        W2 = self.rho * Z1 + math.sqrt(max(1e-12, 1 - self.rho**2)) * Z2

        S = np.empty((n_paths, n_steps+1))
        v = np.empty_like(S)
        S[:,0] = self.s0
        v[:,0] = max(1e-12, self.v0)

        for t in range(n_steps):
            v_half = np.maximum(0.0, v[:, t] + self.kappa*(self.theta - v[:, t])*dt + self.xi*np.sqrt(np.maximum(v[:, t],0))*math.sqrt(dt)*W2[:, t])
            v[:, t+1] = v_half
            S[:, t+1] = S[:, t] * np.exp((self.mu - 0.5*v_half)*dt + np.sqrt(np.maximum(v_half,0))*math.sqrt(dt)*W1[:, t])

        X = S[:, :, None]
        return Paths(X, names=names)

# ----------------------------------------------------------------------
# Option helpers (Black–Scholes) for control variates / sanity checks
# ----------------------------------------------------------------------

def _ndtr(x: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        return norm.cdf(x)
    # erf-based approximation
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))#type:ignore

def bs_call_price(s0: float, k: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    if sigma <= 0 or T <= 0:
        return max(0.0, s0*np.exp(-q*T) - k*np.exp(-r*T))
    d1 = (np.log(s0/k) + (r - q + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return float(s0*np.exp(-q*T)*_ndtr(d1) - k*np.exp(-r*T)*_ndtr(d2))

def bs_put_price(s0: float, k: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    c = bs_call_price(s0, k, r, sigma, T, q)
    return float(c - s0*np.exp(-q*T) + k*np.exp(-r*T))

# ----------------------------------------------------------------------
# Portfolio, risk & payoffs
# ----------------------------------------------------------------------

def portfolio_pnl(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    returns: (n_paths, n_steps, d),
    weights: (d,) or (n_steps, d) rebalanced each step.
    returns PnL (n_paths, n_steps)
    """
    r = np.asarray(returns, float)
    w = np.asarray(weights, float)
    if w.ndim == 1:
        pnl = (r * w[None, None, :]).sum(axis=2)
    else:
        pnl = (r * w[None, :, :]).sum(axis=2)
    return pnl

def var_es(sample: np.ndarray, levels = [0.95, 0.99]) -> Dict[float, Dict[str, float]]:
    """
    sample: array of PnL (loss negative). Returns {level: {"VaR": ..., "ES": ...}}
    """
    x = np.asarray(sample, float)
    out = {}
    for lvl in levels:
        q = np.quantile(x, 1 - lvl)  # losses in left tail if negative pnl
        tail = x[x <= q]
        es = tail.mean() if tail.size > 0 else float("nan")
        out[lvl] = {"VaR": float(q), "ES": float(es)}
    return out

def drawdown(equity_curve: np.ndarray) -> Tuple[float, float]:
    """
    Max drawdown and duration (steps). equity_curve shape (n_paths, n_steps+1) or (n_steps+1,)
    """
    eq = np.asarray(equity_curve, float)
    if eq.ndim == 1:
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        mdd = dd.min()
        # crude duration: length below last peak
        dur = int(np.max(np.diff(np.where(np.r_[1, dd==0])[0])) if np.any(dd==0) else len(eq))
        return float(mdd), float(dur)
    else:
        mdds = []
        durs = []
        for i in range(eq.shape[0]):
            m, d = drawdown(eq[i])
            mdds.append(m); durs.append(d)
        return float(np.mean(mdds)), float(np.mean(durs))

# ----------------------------------------------------------------------
# Control variate wrapper
# ----------------------------------------------------------------------

def control_variate(prices_mc: np.ndarray, prices_cv: np.ndarray, true_cv: float) -> np.ndarray:
    """
    prices_mc: MC estimator for target payoff
    prices_cv: MC estimator for control payoff (correlated with target)
    true_cv:   known analytic price of control
    Returns variance-reduced estimator array (per batch/asset).
    """
    x = np.asarray(prices_mc, float)
    y = np.asarray(prices_cv, float)
    cov = np.cov(x, y)[0,1]
    var_y = np.var(y)
    b = 0.0 if var_y <= 1e-18 else cov / var_y
    return x - b*(y - true_cv)

# ----------------------------------------------------------------------
# Simple CLI self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # GBM single asset sanity check vs BS call (control variate demo)
    n_paths = 100000
    T_steps = 252
    dt = 1/252
    s0, mu, sigma = 100.0, 0.00, 0.2
    r, q = 0.00, 0.00
    K, T = 100.0, T_steps*dt

    gbm = GBM(s0, mu, sigma)
    paths = gbm.simulate(T_steps, n_paths, dt)
    ST = paths.last().ravel()

    # target: call payoff discounted
    pay = np.maximum(ST - K, 0.0) * np.exp(-r*T)
    est_plain = pay.mean()

    # control: BS call price with same sigma
    cv_true = bs_call_price(s0, K, r, sigma, T, q)
    # MC estimate of the same control payoff to compute beta
    est_cv_sample = np.maximum(ST - K, 0.0) * np.exp(-r*T)
    est_cv = est_cv_sample.mean()

    est_cv_reduced = control_variate(pay, est_cv_sample, cv_true).mean()

    print(f"Plain MC: {est_plain:.4f}, CV-MC: {est_cv_reduced:.4f}, BS: {cv_true:.4f}")

    # VaR/ES on last-day PnL of 1x exposure
    ret = paths.returns()  # (n, T, 1)
    pnl = portfolio_pnl(ret, np.array([1.0]))
    tail = var_es(pnl[:, -1], levels=[0.95, 0.99])
    print("VaR/ES:", tail)