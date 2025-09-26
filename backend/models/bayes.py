#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayes.py
--------
Lightweight Bayesian utilities for quant research and product A/Bs.

Includes:
- BetaBinomial: conjugate updates for rates (clicks, wins, fills)
- NormalMeanKnownVar: conjugate mean updates with known observation variance
- ABTest: Bayesian A/B with Beta-Binomial, posteriors, P(A>B), expected lift, Bayes factor
- ThompsonSampler: multi-arm bandit using Beta posteriors
- BayesLinearRegression: conjugate Gaussian linear model with ridge prior
- Helpers: credible_interval, hpd_interval, posterior predictive, sampling utilities

Dependencies: numpy (required), scipy (optional for special funcs & t-quantiles)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import math
import numpy as np

try:
    from scipy import stats, special
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =============================================================================
# Helpers
# =============================================================================

def credible_interval(samples: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """Equal-tail credible interval from samples."""
    a = (1 - level) / 2.0
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))

def hpd_interval(samples: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """Highest posterior density interval from samples."""
    s = np.sort(samples)
    n = len(s)
    if n == 0:
        return (float("nan"), float("nan"))
    m = int(np.floor(level * n))
    if m < 1:
        return (float("nan"), float("nan"))
    widths = s[m:] - s[: n - m]
    j = int(np.argmin(widths))
    return float(s[j]), float(s[j + m])

def _beta_mean_var(a: float, b: float) -> Tuple[float, float]:
    mu = a / (a + b)
    var = (a * b) / (((a + b) ** 2) * (a + b + 1))
    return mu, var

def _safe_betacdf(x: float, a: float, b: float) -> float:
    if not _HAS_SCIPY:
        # crude Monte Carlo fallback
        u = np.random.beta(a, b, size=20000)
        return float((u <= x).mean())
    return float(stats.beta.cdf(x, a, b))

def _safe_betainc(a: float, b: float, x: float) -> float:
    if _HAS_SCIPY:
        return float(special.betainc(a, b, x))
    # very rough fallback via MC
    return _safe_betacdf(x, a, b)

def _student_t_ppf(q: float, df: float) -> float:
    if _HAS_SCIPY:
        return float(stats.t.ppf(q, df))
    # normal approx for large df
    from math import sqrt
    z = math.sqrt(2) * _erfinv(2*q - 1)
    return z * sqrt(df / (df - 2)) if df > 2 else z

def _erfinv(x: float) -> float:
    # Winitzki approximation for inverse erf
    a = 0.147
    sgn = 1 if x >= 0 else -1
    ln = math.log(1 - x * x)
    t = (2 / (math.pi * a) + ln / 2)
    return sgn * math.sqrt(math.sqrt(t * t - ln / a) - t)

# =============================================================================
# 1) Beta–Binomial model (conjugate for Bernoulli rate)
# =============================================================================

@dataclass
class BetaBinomial:
    """Conjugate updates for a Bernoulli rate θ with Beta(α, β) prior."""
    alpha: float = 1.0
    beta: float = 1.0

    def update(self, success: int, trials: int) -> "BetaBinomial":
        if trials < success or success < 0 or trials < 0:
            raise ValueError("Invalid success/trials")
        self.alpha += success
        self.beta += (trials - success)
        return self

    def posterior(self) -> Tuple[float, float]:
        return self.alpha, self.beta

    def mean(self) -> float:
        a, b = self.alpha, self.beta
        return a / (a + b)

    def var(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / (((a + b) ** 2) * (a + b + 1))

    def sample(self, n: int = 10000, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        g = rng or np.random.default_rng()
        return g.beta(self.alpha, self.beta, size=n)

    def prob_greater(self, other: "BetaBinomial", n: int = 200000) -> float:
        """P(θ_self > θ_other) via MC (fast, vectorized)."""
        g = np.random.default_rng()
        x = g.beta(self.alpha, self.beta, n)
        y = g.beta(other.alpha, other.beta, n)
        return float(np.mean(x > y))

    def bayes_factor_against(self, other: "BetaBinomial") -> float:
        """
        Savage–Dickey style BF ~ ratio of prior-to-posterior mass near equality.
        Here a simple symmetric approximation via overlap integral (requires scipy).
        """
        if not _HAS_SCIPY:
            # Monte Carlo overlap
            s = self.sample(200000); t = other.sample(200000)
            kde_bins = np.linspace(0, 1, 501)
            p, _ = np.histogram(s, bins=kde_bins, density=True)
            q, _ = np.histogram(t, bins=kde_bins, density=True)
            dx = np.diff(kde_bins)[0]
            overlap = float(np.minimum(p, q).sum() * dx)
            return 1.0 / max(overlap, 1e-9)
        # Continuous overlap of two Beta densities
        # ∫ min(f, g) dx ≈ similarity; BF ~ 1/overlap
        xs = np.linspace(0, 1, 4001)
        f = stats.beta.pdf(xs, self.alpha, self.beta)
        g = stats.beta.pdf(xs, other.alpha, other.beta)
        overlap = float(np.trapz(np.minimum(f, g), xs))
        return 1.0 / max(overlap, 1e-12)

    def posterior_predictive(self, m: int = 1) -> float:
        """P(k successes in m trials) has Beta-Binomial; return mean success prob for one trial."""
        # mean of posterior θ:
        return self.mean()

# =============================================================================
# 2) Normal mean with known observation variance (conjugate)
# =============================================================================

@dataclass
class NormalMeanKnownVar:
    """
    Observations x_i ~ N(μ, σ2), known σ2.
    Prior μ ~ N(μ0, τ0^2).
    """
    mu0: float = 0.0
    tau0_sq: float = 1.0
    sigma_sq: float = 1.0
    n: int = 0
    xbar: float = 0.0

    def update_batch(self, x: np.ndarray) -> "NormalMeanKnownVar":
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return self
        n0 = self.n
        sum0 = self.xbar * self.n
        self.n = int(n0 + x.size)
        self.xbar = float((sum0 + x.sum()) / max(1, self.n))
        return self

    def posterior(self) -> Tuple[float, float]:
        """Return (mu_n, tau_n_sq) of posterior for μ."""
        prec0 = 1.0 / self.tau0_sq
        prec = 1.0 / self.sigma_sq
        mu_n = (prec0 * self.mu0 + self.n * prec * self.xbar) / (prec0 + self.n * prec)
        tau_n_sq = 1.0 / (prec0 + self.n * prec)
        return float(mu_n), float(tau_n_sq)

    def credible(self, level: float = 0.95) -> Tuple[float, float]:
        mu_n, tau_n_sq = self.posterior()
        if _HAS_SCIPY:
            q = stats.norm.ppf([(1-level)/2, 1-(1-level)/2], loc=mu_n, scale=math.sqrt(tau_n_sq))
            return float(q[0]), float(q[1])
        z = 1.959963984540054  # 95% approx
        s = math.sqrt(tau_n_sq)
        return mu_n - z*s, mu_n + z*s

# =============================================================================
# 3) Bayesian A/B testing (conversion rates via Beta–Binomial)
# =============================================================================

@dataclass
class ABTest:
    """A/B test using Beta-Binomial posteriors for conversion rates."""
    a_prior: Tuple[float, float] = (1.0, 1.0)
    b_prior: Tuple[float, float] = (1.0, 1.0)

    def fit(self, a_success: int, a_trials: int, b_success: int, b_trials: int) -> Dict[str, Any]:
        A = BetaBinomial(*self.a_prior).update(a_success, a_trials)
        B = BetaBinomial(*self.b_prior).update(b_success, b_trials)
        p_better = A.prob_greater(B)
        # Expected lift E[θ_A - θ_B]
        g = np.random.default_rng()
        sA = g.beta(A.alpha, A.beta, 150000)
        sB = g.beta(B.alpha, B.beta, 150000)
        lift = float(np.mean(sA - sB))
        ci_l, ci_u = credible_interval(sA - sB, 0.95)
        bf = A.bayes_factor_against(B)
        return {
            "A": {"alpha": A.alpha, "beta": A.beta, "mean": A.mean()},
            "B": {"alpha": B.alpha, "beta": B.beta, "mean": B.mean()},
            "p_A_gt_B": float(p_better),
            "expected_lift": lift,
            "lift_ci95": (ci_l, ci_u),
            "bayes_factor": float(bf),
        }

# =============================================================================
# 4) Thompson Sampling (multi-arm, Bernoulli rewards)
# =============================================================================

class ThompsonSampler:
    """
    Multi-armed bandit for Bernoulli rewards.
    Use .update(arm, success, trials) after observing outcomes.
    Use .choose() to pick the next arm.
    """
    def __init__(self, n_arms: int, alpha: float = 1.0, beta: float = 1.0, seed: Optional[int] = None):
        self.n = int(n_arms)
        self.alpha = np.full(self.n, float(alpha), dtype=float)
        self.beta = np.full(self.n, float(beta), dtype=float)
        self._rng = np.random.default_rng(seed)

    def update(self, arm: int, success: int, trials: int = 1) -> None:
        if arm < 0 or arm >= self.n: raise IndexError("arm out of range")
        if success < 0 or trials < success: raise ValueError("invalid success/trials")
        self.alpha[arm] += success
        self.beta[arm]  += (trials - success)

    def choose(self) -> int:
        samples = self._rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

# =============================================================================
# 5) Bayesian Linear Regression (Gaussian prior, conjugate posterior)
# =============================================================================

@dataclass
class BayesLinearRegression:
    """
    y = X w + ε,   ε ~ N(0, σ2 I)
    Prior: w ~ N(0, σ2 * (λ I)^-1)  -> ridge prior with precision λ
    If σ2 unknown, you can pass an empirical estimate.

    Fits via closed form; provides posterior mean/cov for w and predictive.
    """
    lam: float = 1.0           # ridge strength λ
    sigma2: float = 1.0        # observation noise variance σ^2
    w_mean_: Optional[np.ndarray] = None
    w_cov_: Optional[np.ndarray] = None
    fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesLinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        lamI = self.lam * np.eye(d)
        A = lamI + (X.T @ X) / self.sigma2
        b = (X.T @ y) / self.sigma2
        # Solve A w = b for posterior mean
        self.w_mean_ = np.linalg.solve(A, b)
        self.w_cov_  = np.linalg.inv(A)  # posterior covariance of w
        self.fitted_ = True
        return self

    def coef_(self) -> np.ndarray:
        if not self.fitted_: raise RuntimeError("Model not fitted")
        return self.w_mean_ # type: ignore

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_: raise RuntimeError("Model not fitted")
        return np.asarray(X, float) @ self.w_mean_

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predictive: y* | X* ~ N(X* w_mean, σ2 + X* Σ_w X*^T)
        """
        if not self.fitted_: raise RuntimeError("Model not fitted")
        X = np.asarray(X, float)
        mean = X @ self.w_mean_
        if not return_std:
            return mean, None
        var = self.sigma2 + np.sum(X @ self.w_cov_ * X, axis=1)
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

# =============================================================================
# Quick self-test (demo)
# =============================================================================

if __name__ == "__main__":
    # 1) Beta–Binomial update
    A = BetaBinomial(1, 1).update(50, 100)
    B = BetaBinomial(1, 1).update(45, 100)
    print("[Beta] A mean=", round(A.mean(), 4), "B mean=", round(B.mean(), 4), "P(A>B)≈", round(A.prob_greater(B), 4))

    # 2) Normal mean known variance
    nm = NormalMeanKnownVar(mu0=0, tau0_sq=10.0, sigma_sq=1.0)
    nm.update_batch(np.random.normal(0.2, 1.0, size=200))
    print("[Normal] posterior:", nm.posterior(), "CI95:", nm.credible())

    # 3) A/B test
    ab = ABTest().fit(a_success=520, a_trials=10000, b_success=480, b_trials=10000)
    print("[AB] p(A>B)=", round(ab["p_A_gt_B"], 4), "lift_ci95=", tuple(round(x, 5) for x in ab["lift_ci95"]))

    # 4) Thompson Sampling
    ts = ThompsonSampler(n_arms=3, alpha=1, beta=1, seed=42)
    for _ in range(1000):
        arm = ts.choose()
        # simulate true rates
        true = [0.03, 0.04, 0.025][arm]
        reward = 1 if np.random.random() < true else 0
        ts.update(arm, reward, 1)
    print("[TS] means≈", np.round(ts.means(), 4))

    # 5) Bayesian Linear Regression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    w_true = np.array([0.5, -1.2, 2.0])
    y = X @ w_true + rng.normal(scale=0.5, size=500)
    blr = BayesLinearRegression(lam=1.0, sigma2=0.25).fit(X, y)
    mu, std = blr.predict(X[:5], return_std=True)
    print("[BLR] first preds:", np.round(mu, 3), "±", np.round(std, 3)) # type: ignore