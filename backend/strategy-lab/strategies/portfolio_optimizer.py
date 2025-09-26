# stratrgies/portfolio_optimizer.py
"""
Portfolio Optimizer (stdlib-only)
---------------------------------
Solvers:
- mean_variance(): projected-gradient on the long-only simplex (sum(w)=budget, w>=0)
                   with optional box-bounds, L2 regularization, and turnover penalty.
- min_variance():  mean-variance with lambda=0 (just variance) on the simplex.
- risk_parity():   iterative multiplicative updates to equalize risk contributions.

Helpers:
- est_returns_cov(): estimate (mu, cov) from close-price histories (dict[symbol]->list[close])
- project_simplex(): projection onto { w>=0, sum(w)=budget }
- small linear algebra utilities (dot, mv, add, sub, scal, cholesky)

Intended use:
- Feed StrategyAgent cross-sectional expected returns (scores) into mean_variance()
  to convert scores -> target weights under a covariance + constraints.
- Use risk_parity() when you want balanced marginal risk without explicit μ.

All functions are pure-Python and work fine up to a few hundred names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import math
import statistics
import time
import random


Vector = List[float]
Matrix = List[List[float]]


# ----------------------------- small linalg ------------------------------------

def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))

def mv(M: Matrix, v: Vector) -> Vector:
    return [dot(row, v) for row in M]

def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]

def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]

def scal(a: Vector, k: float) -> Vector:
    return [k * x for x in a]

def norm2(a: Vector) -> float:
    return math.sqrt(dot(a, a))

def eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def cholesky_pd(A: Matrix) -> Matrix:
    """Naive Cholesky for small positive-definite matrices; raises if not PD."""
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 1e-12:
                    raise ValueError("Matrix not PD")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def solve_spd(A: Matrix, b: Vector, reg_lambda: float = 0.0) -> Vector:
    """Solve (A + reg_lambda*I) x = b using Cholesky (A should be SPD)."""
    n = len(A)
    # form A' = A + λI (copy)
    Ap = [row[:] for row in A]
    if reg_lambda > 0:
        for i in range(n):
            Ap[i][i] += reg_lambda
    L = cholesky_pd(Ap)
    # forward solve Ly=b
    y = [0.0]*n
    for i in range(n):
        s = b[i] - sum(L[i][k]*y[k] for k in range(i))
        y[i] = s / L[i][i]
    # back solve L^T x = y
    x = [0.0]*n
    for i in reversed(range(n)):
        s = y[i] - sum(L[k][i]*x[k] for k in range(i+1, n))
        x[i] = s / L[i][i]
    return x


# ----------------------------- projections / constraints -----------------------

def project_simplex(v: Vector, budget: float = 1.0) -> Vector:
    """
    Project onto the probability simplex { w >= 0, sum w = budget }.
    Michelot-style O(n log n) projection.
    """
    n = len(v)
    if n == 0:
        return []
    if budget <= 0:
        return [0.0]*n
    u = sorted(v, reverse=True)
    css = 0.0
    rho = -1
    for i, ui in enumerate(u, start=1):
        css += ui
        t = (css - budget) / i
        if ui - t > 0:
            rho = i
    theta = (sum(u[:rho]) - budget) / max(1, rho)
    return [max(0.0, x - theta) for x in v]

def clip_box(v: Vector, lo: float, hi: float) -> Vector:
    hi = max(hi, lo)
    return [min(hi, max(lo, x)) for x in v]


# ----------------------------- estimators --------------------------------------

def _pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def est_returns_cov(price_hist: Dict[str, List[float]]) -> Tuple[List[str], Vector, Matrix]:
    """
    Estimate μ (mean daily return) and Σ (cov matrix) from close-price histories.

    price_hist: symbol -> list[close]  (aligned in time; last element = most recent)
    Returns: (symbols, mu_vector, cov_matrix)
    """
    symbols = sorted(price_hist.keys())
    rets: Dict[str, List[float]] = {}
    for s in symbols:
        px = price_hist[s]
        r = []
        for i in range(1, len(px)):
            r.append(_pct_change(px[i-1], px[i]))
        rets[s] = r

    # equal-length guard: pad/truncate to min length
    L = min(len(rets[s]) for s in symbols) if symbols else 0
    if L <= 1:
        return symbols, [0.0]*len(symbols), [[0.0]*len(symbols) for _ in symbols]
    for s in symbols:
        rets[s] = rets[s][-L:]

    # μ
    mu = [statistics.mean(rets[s]) for s in symbols]
    # Σ
    cov = [[0.0]*len(symbols) for _ in symbols]
    for i, si in enumerate(symbols):
        ri = rets[si]
        mi = mu[i]
        for j, sj in enumerate(symbols):
            rj = rets[sj]
            mj = mu[j]
            # population covariance
            cov_ij = sum((ri[k]-mi)*(rj[k]-mj) for k in range(L)) / max(1, L)
            cov[i][j] = cov_ij
    return symbols, mu, cov


# ----------------------------- objective pieces --------------------------------

def quad_form(S: Matrix, w: Vector) -> float:
    """w^T S w"""
    return dot(w, mv(S, w))

@dataclass
class MVConfig:
    risk_aversion: float = 5.0          # lambda; higher -> cares more about return (since we *subtract* μ term)
    budget: float = 1.0                 # sum(w) after projection (gross for long-only)
    max_iter: int = 500
    step: float = 0.1                   # gradient step (will be auto-tuned)
    l2_reg: float = 0.0                 # ridge on weights (stabilizes near-singular Σ)
    turnover_penalty: float = 0.0       # per-unit L1 penalty vs previous weights
    box_lo: float = 0.0                 # lower bound per weight (works with simplex projection if >=0)
    box_hi: float = 1.0                 # upper bound per weight
    tol: float = 1e-6                   # stop when ||w_{t}-w_{t-1}|| < tol
    seed: Optional[int] = None          # for randomized starts (optional)

def mean_variance(mu: Vector, cov: Matrix, cfg: MVConfig, w_prev: Optional[Vector] = None) -> Vector:
    """
    Solve:  min_w  0.5 w^T Σ w - λ μ^T w + (η/2)||w||^2 + τ * ||w - w_prev||_1
            s.t.   w >= 0, sum(w) = budget, and box bounds if provided (box & simplex both applied)

    Approach: projected (proximal) gradient with soft-threshold for L1 turnover, then simplex projection.
    """
    n = len(mu)
    if n == 0:
        return []
    # Lipschitz estimate for step control: use trace(Σ) as crude proxy
    tr = sum(cov[i][i] for i in range(n)) / max(1, n)
    step = cfg.step if cfg.step > 0 else (1.0 / max(1e-6, tr))
    rnd = random.Random(cfg.seed)

    # init: warm start from w_prev if provided else uniform
    if w_prev and len(w_prev) == n:
        w = project_simplex(clip_box(w_prev, cfg.box_lo, cfg.box_hi), budget=cfg.budget)
    else:
        w = project_simplex([cfg.budget / n + 1e-9 * rnd.random() for _ in range(n)], budget=cfg.budget)

    lam = float(cfg.risk_aversion)
    l2 = float(cfg.l2_reg)
    tau = float(cfg.turnover_penalty)

    def grad(wv: Vector) -> Vector:
        # ∇(0.5 w^T Σ w) = Σ w ;  ∇(-λ μ^T w) = -λ μ ;  ∇(η/2 ||w||^2) = η w
        g = add(mv(cov, wv), add(scal(wv, l2), scal(mu, -lam)))
        return g

    last = w[:]
    for it in range(cfg.max_iter):
        g = grad(w)
        # gradient step
        z = sub(w, scal(g, step))
        # proximal L1 for turnover vs w_prev: soft-threshold z toward w_prev by tau*step
        if w_prev is not None and tau > 0:
            thr = tau * step
            z = [ _soft_threshold(z_i - w_prev[i], thr) + w_prev[i] for i, z_i in enumerate(z) ]

        # box clamp then simplex projection (nonnegativity is implicitly enforced by simplex)
        z = clip_box(z, cfg.box_lo, cfg.box_hi)
        w = project_simplex(z, budget=cfg.budget)

        # convergence
        if norm2(sub(w, last)) < cfg.tol:
            break
        last = w[:]
    return w

def _soft_threshold(x: float, a: float) -> float:
    if x > a: return x - a
    if x < -a: return x + a
    return 0.0

def min_variance(cov: Matrix, budget: float = 1.0, box_lo: float = 0.0, box_hi: float = 1.0, max_iter: int = 500) -> Vector:
    """
    Long-only minimum-variance on the simplex using projected gradient (μ term=0).
    """
    n = len(cov)
    mu = [0.0] * n
    cfg = MVConfig(risk_aversion=0.0, budget=budget, box_lo=box_lo, box_hi=box_hi, max_iter=max_iter)
    return mean_variance(mu, cov, cfg)

# ----------------------------- risk parity -------------------------------------

@dataclass
class RPConfig:
    budget: float = 1.0
    box_lo: float = 0.0
    box_hi: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-6
    step: float = 0.1          # multiplicative step dampener

def risk_parity(cov: Matrix, cfg: RPConfig) -> Vector:
    """
    Equal risk contribution (ERC) solution, long-only simplex.
    Targets: w_i * (Σ w)_i == c for all i (same contribution), with sum(w)=budget, w>=0.
    Algorithm: iterative multiplicative updates with simplex projection.
    """
    n = len(cov)
    if n == 0:
        return []

    # init uniform
    w = [cfg.budget / n] * n
    last = w[:]

    for _ in range(cfg.max_iter):
        Sw = mv(cov, w)
        # avoid zero divisions
        g = []
        total = sum(w[i] * Sw[i] for i in range(n)) or 1e-12
        target = total / n
        for i in range(n):
            # scaling factor: want w_i * Sw_i -> target
            if Sw[i] <= 1e-12:
                g.append(1.0)
            else:
                g.append((target / max(1e-12, w[i] * Sw[i])) ** cfg.step)
        # multiplicative update, then box & simplex
        w = [max(cfg.box_lo, min(cfg.box_hi, w[i]*g[i])) for i in range(n)]
        w = project_simplex(w, budget=cfg.budget)

        if norm2(sub(w, last)) < cfg.tol:
            break
        last = w[:]
    return w


# ----------------------------- convenience wrapper -----------------------------

@dataclass
class OptimizeRequest:
    symbols: List[str]
    mu: Vector
    cov: Matrix
    kind: str = "mean_variance"   # "mean_variance" | "min_variance" | "risk_parity"
    mv_cfg: Optional[MVConfig] = None
    rp_cfg: Optional[RPConfig] = None
    w_prev: Optional[Vector] = None

def optimize(req: OptimizeRequest) -> Dict[str, float]:
    """
    One-stop API: returns {symbol: weight}.
    """
    if req.kind == "risk_parity":
        cfg = req.rp_cfg or RPConfig()
        w = risk_parity(req.cov, cfg)
    elif req.kind == "min_variance":
        w = min_variance(req.cov, budget=(req.mv_cfg.budget if req.mv_cfg else 1.0))
    else:
        cfg = req.mv_cfg or MVConfig()
        w = mean_variance(req.mu, req.cov, cfg, w_prev=req.w_prev)

    return {s: float(w[i]) for i, s in enumerate(req.symbols)}


# ----------------------------- example usage -----------------------------------

if __name__ == "__main__":
    # Tiny smoke test with 4 assets
    prices = {
        "AAA": [100,101,100,102,103,104,103,105,104,106],
        "BBB": [50,50.5,50.2,50.4,50.8,51.0,50.7,51.2,51.1,51.5],
        "CCC": [25,24.8,25.2,25.5,25.3,25.6,25.7,25.9,26.1,26.0],
        "DDD": [10,10.5,10.2,10.6,10.4,10.8,10.9,11.1,11.0,11.3],
    }
    syms, mu, cov = est_returns_cov(prices)

    # Mean-variance (long-only, turnover penalty vs previous weights)
    mv_req = OptimizeRequest(
        symbols=syms,
        mu=mu,
        cov=cov,
        kind="mean_variance",
        mv_cfg=MVConfig(risk_aversion=5.0, budget=1.0, turnover_penalty=0.01, l2_reg=1e-6),
        w_prev=[0.25,0.25,0.25,0.25],
    )
    w_mv = optimize(mv_req)
    print("MV weights:", {k: round(v,4) for k,v in w_mv.items()})

    # Risk parity
    rp_req = OptimizeRequest(symbols=syms, mu=mu, cov=cov, kind="risk_parity", rp_cfg=RPConfig(budget=1.0))
    w_rp = optimize(rp_req)
    print("RP weights:", {k: round(v,4) for k,v in w_rp.items()})