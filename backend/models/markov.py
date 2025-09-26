#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
markov.py
---------
Hidden Markov Model (Gaussian emissions) + plain Markov Chain utilities.

Use-cases:
- Regime detection (bull/bear/vol spikes) on returns or feature vectors
- Regime-conditioned strategies: size risk by P(regime==risk_on)
- Transition analysis of discrete states (e.g., signal buckets)

Dependencies: numpy (required). If scipy present, logsumexp is used automatically.

Examples
--------
# HMM on daily returns (1D)
import numpy as np
from markov import HiddenMarkovModel

r = np.loadtxt("returns.csv")  # shape (T,)
hmm = HiddenMarkovModel(n_states=3, random_state=42).fit(r.reshape(-1,1), max_iter=200)
states = hmm.viterbi(r.reshape(-1,1))
probs  = hmm.predict_proba(r.reshape(-1,1))  # smoothed posterior p(z_t=k|X)

# Use regime probabilities to build a position
risk_on = probs[:, hmm.state_by_mean_ascending(-1)]  # highest-mean state
position = np.clip(risk_on * 2.0 - 0.5, -1, 1)

# Plain Markov Chain from discrete states (e.g., signal quantiles)
from markov import MarkovChain
seq = np.loadtxt("signal_states.csv", dtype=int)
mc = MarkovChain.from_sequence(seq, n_states=5, smoothing=1.0)
pi_inf = mc.stationary()
sim = mc.simulate(n=1000)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# Optional SciPy for logsumexp
try:
    from scipy.special import logsumexp as _logsumexp  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _safe_logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    if _HAS_SCIPY:
        return _logsumexp(a, axis=axis, keepdims=keepdims)#type:ignore
    # Stable log-sum-exp
    m = np.max(a, axis=axis, keepdims=True)
    s = np.sum(np.exp(a - m), axis=axis, keepdims=True)
    out = m + np.log(s + 1e-300)
    return out if keepdims else np.squeeze(out, axis=axis)


# =============================================================================
# Hidden Markov Model with Gaussian emissions
# =============================================================================

@dataclass
class HiddenMarkovModel:
    n_states: int
    cov_type: str = "full"           # 'full' or 'diag'
    random_state: Optional[int] = None
    tol: float = 1e-4
    reg_covar: float = 1e-6          # covariance ridge for stability
    max_iter_default: int = 100

    # Learned parameters after fit:
    pi_: Optional[np.ndarray] = None         # (K,)
    A_: Optional[np.ndarray] = None          # (K,K)
    means_: Optional[np.ndarray] = None      # (K,D)
    covs_: Optional[np.ndarray] = None       # (K,D,D) or (K,D)
    loglike_: Optional[float] = None         # last log-likelihood
    converged_: bool = False
    state_order_: Optional[np.ndarray] = None  # indices sorted by mean ascending (for convenience)

    # ---------------------- Core API ----------------------

    def fit(self, X: np.ndarray, max_iter: Optional[int] = None, init: str = "kmeans++") -> "HiddenMarkovModel":
        """
        EM (Baum–Welch) to fit Gaussian HMM.
        X: shape (T, D) or (T,) -> reshaped to (T,1)
        """
        X = self._as_2d(X)
        T, D = X.shape
        K = int(self.n_states)
        rng = np.random.default_rng(self.random_state)

        # --- initialize params ---
        self._init_params(X, K, D, rng, init=init)

        ll_prev = -np.inf
        max_iter = max_iter or self.max_iter_default

        for it in range(max_iter):
            # E-step: forward-backward to get gamma (posterior state probs) and xi (pairwise)
            logB = self._log_emission(X)         # (T,K)
            log_alpha, log_c = self._forward(logB)         # scaling via log domain
            log_beta = self._backward(logB, log_c)
            log_gamma = log_alpha + log_beta
            log_gamma -= _safe_logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)            # (T,K)

            # xi_t(i,j) ∝ α_t(i) A(i,j) b_{t+1}(j) β_{t+1}(j)
            xi = self._expected_transitions(log_alpha, log_beta, logB)  # (K,K)

            # M-step: update pi, A, means, covariances
            self.pi_ = gamma[0] / np.maximum(gamma[0].sum(), 1e-12)

            A_new = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-12)
            # Avoid zero rows (stickiness)
            A_new = np.clip(A_new, 1e-12, None)
            A_new /= A_new.sum(axis=1, keepdims=True)
            self.A_ = A_new

            # Means
            Nk = gamma.sum(axis=0) + 1e-12                     # (K,)
            means = (gamma.T @ X) / Nk[:, None]                 # (K,D)

            # Covariances
            if self.cov_type == "diag":
                covs = np.empty((K, D))
                for k in range(K):
                    diff = X - means[k]
                    covs[k] = (gamma[:, k][:, None] * (diff * diff)).sum(axis=0) / Nk[k]
                    covs[k] = np.maximum(covs[k], self.reg_covar)
            else:  # full
                covs = np.empty((K, D, D))
                for k in range(K):
                    diff = X - means[k]
                    cov = (gamma[:, k][:, None, None] * np.einsum("ti,tj->tij", diff, diff)).sum(axis=0) / Nk[k]
                    cov.flat[:: D + 1] += self.reg_covar
                    covs[k] = cov

            self.means_, self.covs_ = means, covs

            # Log-likelihood
            ll = float(_safe_logsumexp(log_alpha[-1], axis=0))
            self.loglike_ = ll

            if abs(ll - ll_prev) < self.tol * (1 + abs(ll_prev)):
                self.converged_ = True
                break
            ll_prev = ll

        # cache ordered states (ascending mean of first dimension)
        self.state_order_ = np.argsort(self.means_.mean(axis=1))  # average across D for robust ordering#type:ignore
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias to viterbi() hard decode."""
        return self.viterbi(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed posterior p(z_t=k | X_{1:T}) via forward-backward."""
        X = self._as_2d(X)
        logB = self._log_emission(X)
        log_alpha, log_c = self._forward(logB)
        log_beta = self._backward(logB, log_c)
        log_gamma = log_alpha + log_beta
        log_gamma -= _safe_logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most likely state path (int array of shape (T,))."""
        X = self._as_2d(X)
        T, K = X.shape[0], self.n_states
        logB = self._log_emission(X)  # (T,K)
        logA = np.log(self.A_ + 1e-300)#type:ignore
        logpi = np.log(self.pi_ + 1e-300)#type:ignore

        delta = np.empty((T, K))
        psi = np.empty((T, K), dtype=int)

        delta[0] = logpi + logB[0]
        psi[0] = -1
        for t in range(1, T):
            tmp = delta[t-1][:, None] + logA  # (K,K)
            psi[t] = np.argmax(tmp, axis=0)
            delta[t] = np.max(tmp, axis=0) + logB[t]

        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def simulate(self, n: int, start_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a state path and observations.
        Returns (X, states). X shape (n,D).
        """
        rng = np.random.default_rng(self.random_state)
        D = self.means_.shape[1]#type:ignore
        K = self.n_states

        states = np.empty(n, dtype=int)
        X = np.empty((n, D))
        # initial state
        if start_state is None:
            states[0] = rng.choice(K, p=self.pi_)
        else:
            states[0] = int(start_state)

        X[0] = self._sample_emission(states[0], rng)
        for t in range(1, n):
            states[t] = rng.choice(K, p=self.A_[states[t-1]])#type:ignore
            X[t] = self._sample_emission(states[t], rng)
        return X, states

    def stationary(self) -> np.ndarray:
        """Stationary distribution of A (left eigenvector for eigenvalue 1)."""
        A = self.A_
        w, v = np.linalg.eig(A.T)#type:ignore
        i = np.argmin(np.abs(w - 1))
        pi = np.real(v[:, i])
        pi = np.maximum(pi, 0)
        pi = pi / np.sum(pi)
        return pi

    # ---------------------- Convenience ----------------------

    def state_by_mean_ascending(self, rank: int) -> int:
        """
        Return state index by rank after sorting states by average mean (ascending).
        rank=-1 gives the highest-mean state (e.g., 'risk-on').
        """
        order = self.state_order_ if self.state_order_ is not None else np.argsort(self.means_.mean(axis=1))#type:ignore
        return int(order[rank])

    # ---------------------- Internals ----------------------

    def _as_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _init_params(self, X: np.ndarray, K: int, D: int, rng: np.random.Generator, init: str = "kmeans++") -> None:
        # pi uniform, A near-diagonal for regime persistence
        self.pi_ = np.ones(K) / K
        A = np.full((K, K), 1.0 / (K - 1))
        np.fill_diagonal(A, 0.0)
        # add persistence
        np.fill_diagonal(A, 0.85)
        A = A / A.sum(axis=1, keepdims=True)
        self.A_ = A

        # means & covariances
        if init.startswith("kmeans"):
            means, labels = self._kmeans_init(X, K, rng, plusplus=("++" in init))
        else:
            idx = rng.choice(len(X), size=K, replace=False)
            means = X[idx]

        self.means_ = means
        if self.cov_type == "diag":
            var = np.var(X, axis=0) + self.reg_covar
            self.covs_ = np.tile(var, (K, 1))
        else:
            cov = np.cov(X.T) + self.reg_covar * np.eye(D)
            self.covs_ = np.tile(cov[None, :, :], (K, 1, 1))

    def _kmeans_init(self, X: np.ndarray, K: int, rng: np.random.Generator, plusplus: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        T, D = X.shape
        if plusplus:
            centers = np.empty((K, D))
            centers[0] = X[rng.integers(T)]
            d2 = np.full(T, np.inf)
            for k in range(1, K):
                d2 = np.minimum(d2, np.sum((X - centers[k-1])**2, axis=1))
                probs = d2 / np.sum(d2)
                centers[k] = X[rng.choice(T, p=probs)]
        else:
            centers = X[rng.choice(T, size=K, replace=False)]
        # one quick Lloyd step
        dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)  # (T,K)
        labels = np.argmin(dists, axis=1)
        for k in range(K):
            if np.any(labels == k):
                centers[k] = X[labels == k].mean(axis=0)
        return centers, labels

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log p(x_t | z_t=k) for Gaussian emissions; returns (T,K)."""
        T, D = X.shape
        K = self.n_states
        logB = np.empty((T, K))
        for k in range(K):
            mu = self.means_[k]#type:ignore
            if self.cov_type == "diag":
                var = self.covs_[k]#type:ignore
                inv = 1.0 / var
                diff = X - mu
                maha = np.sum(diff * diff * inv, axis=1)
                logdet = np.sum(np.log(var + 1e-300))
            else:
                cov = self.covs_[k]#type:ignore
                try:
                    L = np.linalg.cholesky(cov)
                    # solve L y = diff^T
                    diff = X - mu
                    y = np.linalg.solve(L, diff.T)
                    maha = np.sum(y * y, axis=0)
                    logdet = 2.0 * np.sum(np.log(np.diag(L)))
                except np.linalg.LinAlgError:
                    # add ridge if numerically unstable
                    cov = cov + self.reg_covar * np.eye(D)
                    L = np.linalg.cholesky(cov)
                    diff = X - mu
                    y = np.linalg.solve(L, diff.T)
                    maha = np.sum(y * y, axis=0)
                    logdet = 2.0 * np.sum(np.log(np.diag(L)))
            logB[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + maha)
        return logB

    def _forward(self, logB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass in log domain. Returns (log_alpha, log_c) where log_c are scaling logs per t."""
        T, K = logB.shape
        logA = np.log(self.A_ + 1e-300)#type:ignore
        logpi = np.log(self.pi_ + 1e-300)#type:ignore
        log_alpha = np.empty((T, K))
        # t=0
        log_alpha[0] = logpi + logB[0]
        log_c0 = _safe_logsumexp(log_alpha[0], axis=0)
        log_alpha[0] -= log_c0
        log_c = np.empty(T)
        log_c[0] = log_c0
        # t>0
        for t in range(1, T):
            tmp = log_alpha[t-1][:, None] + logA   # (K,K)
            log_alpha[t] = _safe_logsumexp(tmp, axis=0) + logB[t]
            ct = _safe_logsumexp(log_alpha[t], axis=0)
            log_alpha[t] -= ct
            log_c[t] = ct
        return log_alpha, log_c

    def _backward(self, logB: np.ndarray, log_c: np.ndarray) -> np.ndarray:
        T, K = logB.shape
        logA = np.log(self.A_ + 1e-300)#type:ignore
        log_beta = np.empty((T, K))
        log_beta[-1] = -log_c[-1]  # because alpha scaled by c_t
        for t in range(T - 2, -1, -1):
            tmp = logA + logB[t+1] + log_beta[t+1]  # (K,K) broadcasting
            lb = _safe_logsumexp(tmp, axis=1)
            log_beta[t] = lb - log_c[t]
        return log_beta

    def _expected_transitions(self, log_alpha: np.ndarray, log_beta: np.ndarray, logB: np.ndarray) -> np.ndarray:
        """Sum over time of expected transitions xi_t(i,j). Returns (K,K)."""
        T, K = logB.shape
        logA = np.log(self.A_ + 1e-300)#type:ignore
        xi_sum = np.full((K, K), -np.inf)
        for t in range(T - 1):
            # log xi_t(i,j) ∝ log_alpha[t,i] + logA[i,j] + logB[t+1,j] + log_beta[t+1,j]
            M = log_alpha[t][:, None] + logA + (logB[t+1] + log_beta[t+1])[None, :]
            # log-sum-exp accumulation over time
            if t == 0:
                xi_sum = M
            else:
                # log(a) + log(b) -> logsumexp on stacked
                xi_sum = np.logaddexp(xi_sum, M)
        # exponentiate back and renormalize per row
        xi = np.exp(xi_sum - _safe_logsumexp(xi_sum, axis=1, keepdims=True))
        # rescale by expected counts so rows sum to expected transitions
        gamma_t = np.exp(log_alpha[:-1] + log_beta[:-1])  # (T-1,K)
        row_scale = np.maximum(gamma_t.sum(axis=0), 1e-12)[:, None]
        xi = xi * row_scale
        return xi

    def _sample_emission(self, k: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.means_[k]#type:ignore
        if self.cov_type == "diag":
            return rng.normal(mu, np.sqrt(self.covs_[k]))#type:ignore
        else:
            return rng.multivariate_normal(mu, self.covs_[k], check_valid='ignore')#type:ignore


# =============================================================================
# Plain Markov Chain (discrete observed states)
# =============================================================================

@dataclass
class MarkovChain:
    A: np.ndarray  # (K,K) transition matrix

    @staticmethod
    def from_sequence(states: np.ndarray, n_states: Optional[int] = None, smoothing: float = 0.0) -> "MarkovChain":
        """
        Estimate transitions from integer state sequence with optional Laplace smoothing.
        """
        s = np.asarray(states, dtype=int)
        K = int(n_states or (np.max(s) + 1))
        C = np.zeros((K, K), dtype=float)
        for i, j in zip(s[:-1], s[1:]):
            if 0 <= i < K and 0 <= j < K:
                C[i, j] += 1.0
        C += smoothing
        A = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-12)
        return MarkovChain(A=A)

    def stationary(self) -> np.ndarray:
        """Steady-state distribution π such that π A = π."""
        w, v = np.linalg.eig(self.A.T)
        i = np.argmin(np.abs(w - 1))
        pi = np.real(v[:, i])
        pi = np.maximum(pi, 0)
        pi = pi / np.sum(pi)
        return pi

    def simulate(self, n: int, start_state: Optional[int] = None, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        K = self.A.shape[0]
        s = np.empty(n, dtype=int)
        if start_state is None:
            s[0] = rng.choice(K, p=self.stationary())
        else:
            s[0] = int(start_state)
        for t in range(1, n):
            s[t] = rng.choice(K, p=self.A[s[t-1]])
        return s


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Simulate a 2-state HMM (1D)
    K, D, T = 2, 1, 2000
    pi_true = np.array([0.6, 0.4])
    A_true = np.array([[0.95, 0.05],
                       [0.08, 0.92]])
    means_true = np.array([[0.0], [0.5]])
    covs_true = np.array([[[0.04]], [[0.09]]])

    # manual simulation
    states = np.empty(T, dtype=int)
    x = np.empty((T, D))
    states[0] = rng.choice(K, p=pi_true)
    x[0] = rng.normal(means_true[states[0], 0], np.sqrt(covs_true[states[0], 0, 0]))
    for t in range(1, T):
        states[t] = rng.choice(K, p=A_true[states[t-1]])
        k = states[t]
        x[t] = rng.normal(means_true[k, 0], np.sqrt(covs_true[k, 0, 0]))

    hmm = HiddenMarkovModel(n_states=2, cov_type="full", random_state=42).fit(x, max_iter=200)
    v = hmm.viterbi(x)
    acc = (v == states).mean()
    print(f"[HMM] loglike={hmm.loglike_:.2f} converged={hmm.converged_} viterbi_acc≈{acc:.3f}")

    # MarkovChain from discrete states
    mc = MarkovChain.from_sequence(states, smoothing=1.0)
    print("[MC] A row sums:", mc.A.sum(axis=1))
    print("[MC] stationary:", np.round(mc.stationary(), 3))