# research/exec/almgren.py
"""
Almgren–Chriss (2000/2003) discrete-time optimal execution.

We minimize:
    J = E[Cost] + λ * Var[Cost]

Price dynamics (discrete):
    S_{k+1} = S_k + σ * ε_{k+1} + γ * v_{k+1} * τ
Temporary impact (execution price vs mid):
    P_{k+1} = S_k + η * v_{k+1}

Notation:
    X0      : initial shares to sell (>0 for sell, <0 for buy)
    N       : number of slices
    T       : horizon (time units)
    τ       : T / N  (slice length)
    σ       : volatility (per √time unit of τ)
    γ       : permanent impact coefficient
    η       : temporary impact coefficient
    λ       : risk aversion (>= 0)

Optimal trajectory (discrete AC):
    Define κ̄ by:
        cosh(κ̄) = 1 + (λ * σ^2 * τ) / (2η)
    Then holdings x_j (j=0..N) are:
        x_j = X0 * sinh(κ̄ * (N - j)) / sinh(κ̄ * N)
    Special case λ = 0  ⇒ linear trajectory:
        x_j = X0 * (1 - j/N)

Expected cost, variance, utility (implementation shortfall formulation):
    E[Cost] = 0.5 * γ * X0^2 + η * τ * Σ v_j^2
    Var[Cost] = σ^2 * τ * Σ x_{j-1}^2
    J = E[Cost] + λ * Var[Cost]

All formulas assume **sell positive** X0. For buys, set X0 < 0; signs flow through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ACParams:
    X0: float           # shares (sell>0, buy<0)
    N: int              # number of slices (>=1)
    T: float            # horizon (time units)
    sigma: float        # volatility per sqrt(time unit)
    eta: float          # temporary impact coefficient
    gamma: float        # permanent impact coefficient
    lam: float = 0.0    # risk aversion λ >= 0


@dataclass(frozen=True)
class ACSchedule:
    t: np.ndarray       # shape (N+1,), time grid [0..T]
    x: np.ndarray       # holdings after j-th trade; x[0]=X0, x[N]≈0
    q: np.ndarray       # shares executed each slice j=1..N (q[j] >= 0 if X0>0 sell)
    v: np.ndarray       # participation rate per slice (shares / τ)
    tau: float
    kappa_bar: float    # discrete κ̄
    stats: Dict[str, float]  # E_cost, Var_cost, Utility


def _kappa_bar(lam: float, sigma: float, eta: float, tau: float) -> float:
    """Discrete-time κ̄ from cosh(κ̄) = 1 + (λ σ^2 τ)/(2η)."""
    if lam <= 0:
        return 0.0
    c = 1.0 + (lam * sigma * sigma * tau) / (2.0 * eta)
    # numerical safety
    c = max(1.0, float(c))
    return float(np.arccosh(c))


def optimal_schedule(params: ACParams) -> ACSchedule:
    """
    Compute optimal AC schedule and cost/variance.

    Returns ACSchedule with time grid, holdings x_j, trades q_j, rates v_j,
    and stats {E_cost, Var_cost, Utility}.
    """
    X0, N, T, sigma, eta, gamma, lam = (
        float(params.X0),
        int(params.N),
        float(params.T),
        float(params.sigma),
        float(params.eta),
        float(params.gamma),
        float(params.lam),
    )
    if N < 1:
        raise ValueError("N must be >= 1")
    if T <= 0 or eta <= 0 or sigma < 0:
        raise ValueError("T>0, eta>0, sigma>=0 required")

    tau = T / N
    kb = _kappa_bar(lam, sigma, eta, tau)

    # Holdings x_j
    t = np.linspace(0.0, T, N + 1, dtype=float)
    x = np.empty(N + 1, dtype=float)

    if lam <= 0 or kb == 0.0:
        # Risk-neutral ⇒ linear trajectory
        x[:] = X0 * (1.0 - np.arange(N + 1) / N)
    else:
        denom = math.sinh(kb * N)
        # numerical guard
        if abs(denom) < 1e-18:
            x[:] = X0 * (1.0 - np.arange(N + 1) / N)
        else:
            for j in range(N + 1):
                x[j] = X0 * math.sinh(kb * (N - j)) / denom

    # Shares traded each slice (q_j ≥0 for sells when X0>0)
    # We define q_j as the amount executed in (j-1 -> j]
    q = -np.diff(x, prepend=x[0])  # q[0]=0 by construction; use indices 1..N
    q[0] = 0.0
    q = q[1:]  # size N
    # Participation rate v_j = q_j / τ (signed shares per time)
    v = q / tau

    # Expected cost & variance (discrete AC)
    # E[Cost] = 0.5*γ*X0^2 + η * τ * Σ v_j^2
    E_temp = eta * tau * float(np.sum(v * v))
    E_perm = 0.5 * gamma * X0 * X0
    E_cost = E_temp + E_perm

    # Var[Cost] = σ^2 * τ * Σ x_{j-1}^2  (use holdings BEFORE trading in slice j)
    x_before = x[:-1]  # length N
    Var_cost = (sigma * sigma) * tau * float(np.sum(x_before * x_before))

    Utility = E_cost + lam * Var_cost

    stats = {
        "E_cost": E_cost,
        "Var_cost": Var_cost,
        "Utility": Utility,
        "E_temp": E_temp,
        "E_perm": E_perm,
    }

    # build arrays back to length N+1 for convenience
    q_full = np.concatenate([[0.0], q])         # length N+1, q_full[0]=0
    v_full = np.concatenate([[0.0], v])         # length N+1, v_full[0]=0

    return ACSchedule(t=t, x=x, q=q_full, v=v_full, tau=tau, kappa_bar=kb, stats=stats)


# ------------------------------ Helpers -----------------------------------

def expected_slippage_per_share(stats: Dict[str, float], X0: float) -> float:
    """
    Return expected implementation shortfall per share (currency units),
    i.e., E[Cost] / |X0|.
    """
    X = abs(float(X0))
    return (stats["E_cost"] / X) if X > 0 else 0.0


def summarize(params: ACParams) -> Dict[str, float]:
    """Convenience: compute schedule and return scalar summary."""
    sched = optimal_schedule(params)
    e_cost = sched.stats["E_cost"]
    var_cost = sched.stats["Var_cost"]
    util = sched.stats["Utility"]
    e_cost_ps = expected_slippage_per_share(sched.stats, params.X0)
    return {
        "tau": sched.tau,
        "kappa_bar": sched.kappa_bar,
        "E_cost": e_cost,
        "Var_cost": var_cost,
        "Utility": util,
        "E_cost_per_share": e_cost_ps,
    }


# ------------------------------ Example -----------------------------------

if __name__ == "__main__":
    # Example: sell 100k shares over 1 hour, 60 slices (1min bars)
    p = ACParams(
        X0=100_000.0,
        N=60,
        T=1.0,
        sigma=0.02,     # 2% per sqrt(hour)
        eta=2e-6,       # temporary impact coeff (currency per (shares/time))
        gamma=1e-6,     # permanent impact coeff (currency per (shares/time))
        lam=1e-6,       # risk aversion
    )
    sched = optimal_schedule(p)
    print("kappa_bar:", sched.kappa_bar, "tau:", sched.tau)
    print("E[Cost]:", sched.stats["E_cost"], "Var[Cost]:", sched.stats["Var_cost"])
    print("First 5 holdings:", np.round(sched.x[:5], 2))
    print("First 5 child trades:", np.round(sched.q[1:6], 2))