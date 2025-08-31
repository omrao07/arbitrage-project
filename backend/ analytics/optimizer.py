# backend/alloc/optimizer.py
from __future__ import annotations

"""
Portfolio Optimizer
-------------------
- Mean-Variance (max Sharpe or min variance) with soft/hard constraints
- Risk Parity & HRP (hierarchical risk parity) implementations
- Black–Litterman blending (optional)
- Turnover & transaction costs (linear + quadratic) and position limits
- Long-only or long/short with gross/net exposure constraints and leverage caps
- Target-vol overlay (scale weights to hit vol target)
- Uses CVXPY if available; otherwise a robust projected-gradient fallback
- No hard deps except numpy/scipy (scipy optional). Pandas optional for labels.

Plugs into:
  - risk.* for cov/vol
  - pricer.* / data_api.* for expected returns (mu) or alpha signals
  - strategy allocator / hedger as a clean function call

Typical use:
  opt = Optimizer(Config(objective="max_sharpe", long_only=True, vol_target=0.15))
  w, diag = opt.optimize(mu, Sigma, w_prev=w_prev, ids=symbols)

All inputs:
  mu    : (N,) expected returns (daily or annualized; be consistent with Sigma)
  Sigma : (N,N) covariance matrix
  w_prev: (N,) current weights for turnover/costs (optional)
  ids   : list[str] tickers (optional, for nicer diagnostics)
"""

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("optimizer requires numpy") from e

try:
    import pandas as pd  # optional, only for pretty outputs
except Exception:
    pd = None  # type: ignore

# Optional solvers
HAVE_CVXPY = True
try:
    import cvxpy as cp  # type: ignore
except Exception:
    HAVE_CVXPY = False
    cp = None  # type: ignore

# Optional SciPy for clustering (HRP)
HAVE_SCIPY = True
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
except Exception:
    HAVE_SCIPY = False


# ----------------------- Config & constraints -------------------------------

@dataclass
class Config:
    objective: str = "max_sharpe"   # "max_sharpe" | "min_var" | "risk_parity" | "hrp"
    risk_aversion: float = 1.0      # λ (only used for utility objective if needed)
    vol_target: Optional[float] = None  # If set, scale weights to hit this σ (same units as Sigma)
    long_only: bool = True
    leverage_cap: Optional[float] = 1.0  # sum |w| <= cap (for long/short)
    net_exposure: Optional[float] = 1.0  # sum w == net_exposure (1.0 means fully invested)
    lower_bound: float = 0.0        # per-asset floor (long-only default 0.0; long/short e.g. -0.05)
    upper_bound: float = 0.10       # per-asset cap (e.g., 10%)
    sector_caps: Optional[Dict[str, Tuple[float, float]]] = None  # {"TECH": (0, 0.35), ...}
    sector_map: Optional[Dict[str, str]] = None  # {"AAPL": "TECH", ...}
    turnover_penalty: float = 0.0   # λ_turnover * ||w - w_prev||_1 (approx in fallback)
    tc_linear: float = 0.0          # linear cost per 1 weight unit changed
    tc_quad: float = 0.0            # quadratic cost per (Δw)^2
    black_litterman: Optional[Dict[str, Any]] = None  # { "tau":..., "P":..., "Q":..., "Omega":... }
    clip_sigma_eps: float = 1e-10   # numerical guard
    max_iters: int = 10_000         # fallback optimizer iterations
    stepsize: float = 0.05          # fallback gradient step
    tol: float = 1e-7               # fallback stopping criterion
    seed: Optional[int] = None


# ----------------------- Public API -----------------------------------------

class Optimizer:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()

    # ---- main entry ---------------------------------------------------------
    def optimize(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        *,
        w_prev: Optional[np.ndarray] = None,
        ids: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Returns (weights, diagnostics)
        """
        mu = _as_1d(mu)
        Sigma = _as_2d(Sigma)
        N = mu.shape[0]
        assert Sigma.shape == (N, N), "Sigma shape mismatch"

        if self.cfg.seed is not None:
            np.random.seed(int(self.cfg.seed))

        # Optional: Black–Litterman posterior
        if self.cfg.black_litterman:
            mu = self._black_litterman(mu, Sigma, self.cfg.black_litterman)

        if self.cfg.objective in ("risk_parity", "hrp"):
            w = self._risk_parity_or_hrp(mu, Sigma, ids=ids)
        else:
            if HAVE_CVXPY:
                w = self._solve_cvx(mu, Sigma, w_prev=w_prev, ids=ids)
            else:
                w = self._solve_projected(mu, Sigma, w_prev=w_prev, ids=ids)

        # Target-vol scaling
        if self.cfg.vol_target is not None and self.cfg.vol_target > 0:
            vol = _portfolio_vol(w, Sigma)
            if vol > self.cfg.clip_sigma_eps:
                w = (self.cfg.vol_target / vol) * w

        diag = _diagnostics(w, mu, Sigma, ids=ids)
        return w, diag

    # ---- Black–Litterman ----------------------------------------------------
    def _black_litterman(self, mu: np.ndarray, Sigma: np.ndarray, bl: Dict[str, Any]) -> np.ndarray:
        """
        Basic BL posterior:
           μ_BL = μ_prior + τΣ Pᵀ (P τΣ Pᵀ + Ω)^{-1} (Q - P μ_prior)
        """
        tau = float(bl.get("tau", 0.05))
        P = np.asarray(bl["P"], dtype=float)     # (K,N)
        Q = np.asarray(bl["Q"], dtype=float)     # (K,)
        Omega = np.asarray(bl.get("Omega", None), dtype=float) if bl.get("Omega") is not None else None
        if Omega is None:
            # confidence proportional to variance of views
            Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))
        A = P @ (tau * Sigma) @ P.T + Omega
        adj = (tau * Sigma) @ P.T @ np.linalg.solve(A, (Q - P @ mu))
        return mu + adj

    # ---- CVXPY solver -------------------------------------------------------
    def _solve_cvx(self, mu: np.ndarray, Sigma: np.ndarray, *, w_prev: Optional[np.ndarray], ids: Optional[List[str]]) -> np.ndarray:
        N = mu.shape[0]
        w = cp.Variable(N) # type: ignore

        # Objective: max Sharpe -> min (γ * wᵀΣw - μᵀw)
        if self.cfg.objective == "max_sharpe":
            gamma = 1.0  # scale of variance term; relative to mu term
            obj = cp.Minimize(gamma * cp.quad_form(w, Sigma) - mu @ w) # type: ignore
        elif self.cfg.objective == "min_var":
            obj = cp.Minimize(cp.quad_form(w, Sigma)) # type: ignore
        else:
            # default: mean-variance utility
            lam = float(self.cfg.risk_aversion)
            obj = cp.Minimize(lam * cp.quad_form(w, Sigma) - mu @ w) # type: ignore

        cons = []

        # Exposure
        if self.cfg.net_exposure is not None:
            cons.append(cp.sum(w) == float(self.cfg.net_exposure)) # type: ignore

        # Leverage
        if (not self.cfg.long_only) and (self.cfg.leverage_cap is not None):
            cons.append(cp.norm1(w) <= float(self.cfg.leverage_cap)) # type: ignore

        # Bounds
        lb = float(self.cfg.lower_bound)
        ub = float(self.cfg.upper_bound)
        if self.cfg.long_only:
            lb = max(0.0, lb)
        cons += [w >= lb, w <= ub]

        # Sector caps (sum w in sector within [lo,hi])
        if self.cfg.sector_caps and self.cfg.sector_map and ids:
            for sec, (lo, hi) in self.cfg.sector_caps.items():
                mask = np.array([1.0 if self.cfg.sector_map.get(a, None) == sec else 0.0 for a in ids], dtype=float)
                if mask.sum() > 0:
                    cons.append(mask @ w >= float(lo))
                    cons.append(mask @ w <= float(hi))

        # Turnover / transaction costs
        if w_prev is not None and (self.cfg.turnover_penalty > 0 or self.cfg.tc_linear > 0 or self.cfg.tc_quad > 0):
            dw = w - w_prev
            pen = 0
            if self.cfg.turnover_penalty > 0:
                pen += float(self.cfg.turnover_penalty) * cp.norm1(dw) # type: ignore
            if self.cfg.tc_linear > 0:
                pen += float(self.cfg.tc_linear) * cp.norm1(dw) # type: ignore
            if self.cfg.tc_quad > 0:
                pen += float(self.cfg.tc_quad) * cp.sum_squares(dw) # type: ignore
            obj = cp.Minimize(obj.args[0] + pen)  # type: ignore 

        prob = cp.Problem(obj, cons) # type: ignore
        prob.solve(solver=cp.OSQP, verbose=False)  # type: ignore # OSQP handles PSD QPs well

        if w.value is None:
            # try SCS as a fallback
            prob.solve(solver=cp.SCS, verbose=False, max_iters=8_000) # type: ignore
        if w.value is None:
            raise RuntimeError("CVXPY failed to find a solution")

        return np.asarray(w.value, dtype=float).reshape(-1)

    # ---- Projected gradient fallback ---------------------------------------
    def _solve_projected(self, mu: np.ndarray, Sigma: np.ndarray, *, w_prev: Optional[np.ndarray], ids: Optional[List[str]]) -> np.ndarray:
        """
        Projected gradient descent on:
            min f(w) = λ wᵀΣw - μᵀw + costs
        with box constraints and exposure constraints. Simple & robust.
        """
        N = mu.shape[0]
        rng = np.random.default_rng(self.cfg.seed or 7)

        # start from previous or equal-weight feasible
        if w_prev is not None and w_prev.shape[0] == N:
            w = w_prev.copy()
        else:
            w = np.ones(N) / max(1, N)
        w = _project_box_and_exposure(
            w,
            lb=(max(0.0, self.cfg.lower_bound) if self.cfg.long_only else self.cfg.lower_bound),
            ub=self.cfg.upper_bound,
            net=self.cfg.net_exposure,
        )

        lam = float(self.cfg.risk_aversion if self.cfg.objective != "max_sharpe" else 1.0)

        def obj_grad(wv: np.ndarray) -> Tuple[float, np.ndarray]:
            # f = lam w' Σ w - mu' w + costs(dw)
            quad = lam * float(wv @ Sigma @ wv)
            lin = - float(mu @ wv)
            val = quad + lin
            grad = 2 * lam * (Sigma @ wv) - mu

            if w_prev is not None and (self.cfg.turnover_penalty > 0 or self.cfg.tc_linear > 0 or self.cfg.tc_quad > 0):
                dw = wv - w_prev
                # L1 part: subgradient ~ sign(dw), smooth with epsilon
                if self.cfg.turnover_penalty > 0 or self.cfg.tc_linear > 0:
                    lam1 = float(self.cfg.turnover_penalty + self.cfg.tc_linear)
                    val += lam1 * np.sum(np.abs(dw))
                    grad += lam1 * np.tanh(100.0 * dw)  # smooth sign
                if self.cfg.tc_quad > 0:
                    lam2 = float(self.cfg.tc_quad)
                    val += lam2 * float(dw @ dw)
                    grad += 2 * lam2 * dw

            return val, grad

        step = float(self.cfg.stepsize)
        best = (1e99, w.copy())
        for it in range(int(self.cfg.max_iters)):
            val, g = obj_grad(w)
            # backtracking
            t = step
            for _ in range(8):
                w_try = w - t * g
                w_try = _project_box_and_exposure(
                    w_try,
                    lb=(max(0.0, self.cfg.lower_bound) if self.cfg.long_only else self.cfg.lower_bound),
                    ub=self.cfg.upper_bound,
                    net=self.cfg.net_exposure,
                )
                v2, _ = obj_grad(w_try)
                if v2 <= val - 1e-8:
                    w = w_try
                    val = v2
                    break
                t *= 0.5

            if val < best[0]:
                best = (val, w.copy())

            # simple convergence check
            if np.linalg.norm(g, ord=2) * t < self.cfg.tol:
                break

            # small random jiggle to escape plateaus
            if it % 250 == 249:
                w = _project_box_and_exposure(w + 1e-4 * rng.normal(size=N),
                                              lb=(max(0.0, self.cfg.lower_bound) if self.cfg.long_only else self.cfg.lower_bound),
                                              ub=self.cfg.upper_bound, net=self.cfg.net_exposure)

        return best[1]

    # ---- Risk Parity / HRP --------------------------------------------------
    def _risk_parity_or_hrp(self, mu: np.ndarray, Sigma: np.ndarray, ids: Optional[List[str]]) -> np.ndarray:
        if self.cfg.objective == "risk_parity":
            return _risk_parity(Sigma, long_only=self.cfg.long_only,
                                lb=(max(0.0, self.cfg.lower_bound) if self.cfg.long_only else self.cfg.lower_bound),
                                ub=self.cfg.upper_bound,
                                net=self.cfg.net_exposure)
        # HRP
        return _hrp(Sigma, ids=ids, long_only=self.cfg.long_only,
                    lb=(max(0.0, self.cfg.lower_bound) if self.cfg.long_only else self.cfg.lower_bound),
                    ub=self.cfg.upper_bound,
                    net=self.cfg.net_exposure)


# ----------------------- helpers / math -------------------------------------

def _as_1d(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    return a

def _as_2d(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = np.diag(a)
    return a

def _project_box_and_exposure(w: np.ndarray, *, lb: float, ub: float, net: Optional[float]) -> np.ndarray:
    # clip to [lb, ub], then project to sum constraint if provided
    w = np.clip(w, lb, ub)
    if net is not None:
        s = w.sum()
        if abs(s) < 1e-12:
            # distribute evenly
            w = w + (float(net) / max(1, w.size))
        else:
            w = w * (float(net) / s)
        # After scaling, re-clip (iterate once)
        w = np.clip(w, lb, ub)
        # small renorm
        s2 = w.sum()
        if s2 != 0 and net is not None:
            w *= float(net) / s2
    return w

def _portfolio_vol(w: np.ndarray, Sigma: np.ndarray) -> float:
    v = float(w @ Sigma @ w)
    return math.sqrt(max(v, 0.0))

def _diagnostics(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, ids: Optional[List[str]]) -> Dict[str, Any]:
    ret = float(mu @ w)
    vol = _portfolio_vol(w, Sigma)
    sharpe = ret / vol if vol > 1e-12 else float("nan")
    diag = {"ret": ret, "vol": vol, "sharpe": sharpe}
    if ids and pd is not None:
        diag["weights"] = pd.Series(w, index=ids).sort_values(ascending=False) # type: ignore
    else:
        diag["weights"] = w # type: ignore
    return diag

# ---- Risk parity (simple iterative method) ----
def _risk_parity(Sigma: np.ndarray, long_only: bool, lb: float, ub: float, net: Optional[float]) -> np.ndarray:
    N = Sigma.shape[0]
    w = np.ones(N) / N
    for _ in range(5000):
        m = Sigma @ w
        # target: w_i * (Σw)_i equal for all i
        w_new = np.where(m > 0, 1.0 / m, 0.0)
        w_new = _project_box_and_exposure(w_new, lb=lb, ub=ub, net=net)
        if np.linalg.norm(w_new - w) < 1e-9:
            w = w_new
            break
        w = 0.5 * w + 0.5 * w_new
    return _project_box_and_exposure(w, lb=lb, ub=ub, net=net)

# ---- HRP (López de Prado) minimal implementation ----
def _hrp(Sigma: np.ndarray, ids: Optional[List[str]], long_only: bool, lb: float, ub: float, net: Optional[float]) -> np.ndarray:
    N = Sigma.shape[0]
    if N == 1:
        return np.array([float(net or 1.0)])
    # correlation & distance
    D = np.diag(1.0 / (np.sqrt(np.diag(Sigma)) + 1e-12))
    Corr = D @ Sigma @ D
    Corr = np.clip(Corr, -1.0, 1.0)
    if HAVE_SCIPY:
        dist = np.sqrt(0.5 * (1 - Corr))
        Z = linkage(squareform(dist, checks=False), method="single")
        sort_ix = _quasi_diag(Z, N)
    else:
        # fallback: order by average correlation (roughly)
        avg_c = np.argsort(Corr.mean(axis=0))
        sort_ix = list(map(int, avg_c))
    # recursive bisection
    cov_sorted = Sigma[np.ix_(sort_ix, sort_ix)]
    w_sorted = _hrp_bisect(cov_sorted)
    w = np.zeros(N)
    w[np.array(sort_ix, dtype=int)] = w_sorted
    return _project_box_and_exposure(w, lb=lb, ub=ub, net=net)

def _quasi_diag(Z: np.ndarray, N: int) -> List[int]:
    # from scipy dendrogram leaves order (quasi-diagonalization)
    dn = dendrogram(Z, no_plot=True)
    return list(map(int, dn["leaves"]))

def _hrp_bisect(cov: np.ndarray) -> np.ndarray:
    # allocate by inverse variance within clusters
    def _inv_var_weights(c):
        iv = 1.0 / (np.diag(c) + 1e-12)
        w = iv / iv.sum()
        return w
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    split = n // 2
    c1, c2 = cov[:split, :split], cov[split:, split:]
    w1 = _hrp_bisect(c1)
    w2 = _hrp_bisect(c2)
    var1 = float(w1 @ c1 @ w1)
    var2 = float(w2 @ c2 @ w2)
    a = 1.0 - var1 / (var1 + var2 + 1e-12)
    return np.concatenate([a * w1, (1 - a) * w2])