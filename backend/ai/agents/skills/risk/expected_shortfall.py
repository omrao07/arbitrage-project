# backend/risk/expected_shortfall.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

# -------------------------- Utilities --------------------------

def _sorted(xs: Iterable[float]) -> List[float]:
    arr = list(xs)
    arr.sort()
    return arr

def _mean(xs: Iterable[float]) -> float:
    arr = list(xs)
    return sum(arr) / max(1, len(arr))

def _stdev(xs: Iterable[float]) -> float:
    arr = list(xs)
    if len(arr) < 2: return 0.0
    m = _mean(arr)
    v = sum((x - m) * (x - m) for x in arr) / (len(arr) - 1)
    return math.sqrt(max(0.0, v))

def _quantile(xs_sorted: List[float], q: float) -> float:
    """Linear interpolation quantile, xs_sorted ascending."""
    n = len(xs_sorted)
    if n == 0: return 0.0
    if q <= 0: return xs_sorted[0]
    if q >= 1: return xs_sorted[-1]
    pos = (n - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return xs_sorted[lo]
    w = pos - lo
    return (1 - w) * xs_sorted[lo] + w * xs_sorted[hi]

# -------------------------- Historical ES --------------------------

def historical_var(returns: Iterable[float], alpha: float = 0.975) -> float:
    """
    One-sided left-tail VaR at confidence alpha (e.g., 0.975).
    We assume loss = -return. VaR is a positive number representing loss threshold.
    """
    xs = _sorted(returns)
    p = 1 - alpha
    q = _quantile(xs, p)  # low quantile of returns
    return max(0.0, -q)

def historical_es(returns: Iterable[float], alpha: float = 0.975) -> float:
    """
    Historical ES/CVaR: average loss beyond VaR.
    Returns a positive number (expected loss conditional on exceeding VaR).
    """
    xs = _sorted(returns)
    n = len(xs)
    if n == 0: return 0.0
    p = 1 - alpha
    k = max(1, int(math.floor(p * n)))  # number of worst observations to average
    # If p*n < 1, include at least the single worst observation for stability.
    tail = xs[:k]
    if not tail:
        tail = [xs[0]]
    return -_mean(tail)

# -------------------------- Parametric ES --------------------------

def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def _ncdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def parametric_var_normal(mu: float, sigma: float, alpha: float = 0.975) -> float:
    """
    Normal VaR for returns ~ N(mu, sigma^2). Positive loss number.
    VaR = -(mu + sigma * z_p) for p = 1 - alpha (left tail).
    """
    p = 1 - alpha
    # inverse CDF via approximation (Moro-ish). Keep it simple with binary search.
    z = _inv_norm_cdf(p)
    return max(0.0, -(mu + sigma * z))

def parametric_es_normal(mu: float, sigma: float, alpha: float = 0.975) -> float:
    """
    Normal ES: ES = -(mu) + sigma * φ(z_p) / p   where p = 1 - alpha, z_p = Φ^{-1}(p)
    Positive loss number.
    """
    p = 1 - alpha
    if p <= 0 or sigma <= 0: 
        return max(0.0, -mu)
    z = _inv_norm_cdf(p)
    return max(0.0, -mu + sigma * (_phi(z) / p))

def parametric_es_cornish_fisher(mu: float, sigma: float, skew: float, kurt: float, alpha: float = 0.975) -> float:
    """
    Cornish–Fisher adjusted ES:
      1) Adjust quantile via CF: z* = z + (1/6)(z^2-1)S + (1/24)(z^3-3z)K - (1/36)(2z^3-5z)S^2
      2) Approx ES ≈ -(mu) + sigma * φ(z*) / p   (using adjusted quantile)
    This is an approximation; suitable when tails deviate from normal.
    """
    p = 1 - alpha
    if p <= 0 or sigma <= 0:
        return max(0.0, -mu)
    z = _inv_norm_cdf(p)
    S, K = float(skew), float(kurt)
    z2, z3 = z*z, z*z*z
    z_star = z + (1/6)*(z2-1)*S + (1/24)*(z3-3*z)*K - (1/36)*(2*z3-5*z)*S*S
    return max(0.0, -mu + sigma * (_phi(z_star) / p))

# Simple inverse normal CDF via binary search (no deps, ~1e-6 OK)
def _inv_norm_cdf(p: float, lo: float = -8.0, hi: float = 8.0, tol: float = 1e-7) -> float:
    if p <= 0.0: return -8.0
    if p >= 1.0: return 8.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _ncdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# -------------------------- Stressed ES --------------------------

def stressed_es(returns: Iterable[float], alpha: float = 0.975, tail_weight: float = 2.0) -> float:
    """
    Weighted-tail ES: emphasize worst observations with weight ~ exp(rank * c).
    tail_weight=1 equals historical ES; >1 overweights extremes.
    """
    xs = _sorted(returns)
    n = len(xs)
    if n == 0: return 0.0
    p = 1 - alpha
    k = max(1, int(math.floor(p * n)))
    tail = xs[:k]
    if not tail: tail = [xs[0]]
    # weights increasing toward the worst obs
    m = len(tail)
    w = [tail_weight ** (i+1) for i in range(m)]
    s = sum(w)
    es = -sum(t * wi for t, wi in zip(tail, w)) / s
    return es

# -------------------------- Bootstrap CI --------------------------

def bootstrap_es_ci(
    returns: Iterable[float], alpha: float = 0.975, n_boot: int = 2000, 
    ci: float = 0.90, seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Returns (ES_hat, lo, hi) with percentile bootstrap CI.
    """
    xs = list(returns)
    n = len(xs)
    if n == 0: return 0.0, 0.0, 0.0
    if seed is not None: random.seed(seed)
    es_hat = historical_es(xs, alpha=alpha)
    sims: List[float] = []
    for _ in range(n_boot):
        sample = [xs[random.randrange(n)] for _ in range(n)]
        sims.append(historical_es(sample, alpha=alpha))
    sims.sort()
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    lo = _quantile(sims, lo_q)
    hi = _quantile(sims, hi_q)
    return es_hat, lo, hi

# -------------------------- ES Contributions --------------------------

@dataclass
class ESContribResult:
    total_es: float
    contribs: Dict[str, float]  # per-name contribution (sums to total_es approximately)

def es_contributions_normal(
    weights: Dict[str, float],
    mu: Dict[str, float],
    cov: Dict[Tuple[str, str], float],
    alpha: float = 0.975
) -> ESContribResult:
    """
    Euler allocation under (approximately) normal P&L:
    Portfolio ES ≈ -μ_p + σ_p * φ(z_p)/p, with p=1-α, z_p=Φ^{-1}(p)
    Marginal ES_i ≈ ∂ES/∂w_i = -μ_i + (φ(z_p)/p) * ( (Σ w)_i / σ_p )
    Contribution_i = w_i * Marginal ES_i
    """
    names = list(weights.keys())
    w = [weights[k] for k in names]
    mu_vec = [mu.get(k, 0.0) for k in names]

    # portfolio mean & variance
    mu_p = sum(wi * mi for wi, mi in zip(w, mu_vec))
    # σ_p^2 = w' Σ w
    var_p = 0.0
    m = len(names)
    for i in range(m):
        for j in range(m):
            var_p += w[i] * w[j] * cov.get((names[i], names[j]), 0.0)
    sigma_p = math.sqrt(max(1e-18, var_p))

    p = 1 - alpha
    z = _inv_norm_cdf(p)
    k = _phi(z) / max(1e-12, p)
    total_es = max(0.0, -mu_p + sigma_p * k)

    # Σ w vector
    sig_grad = [0.0] * m
    for i in range(m):
        # (Σ w)_i = sum_j Σ_{i,j} w_j
        s = 0.0
        for j in range(m):
            s += cov.get((names[i], names[j]), 0.0) * w[j]
        sig_grad[i] = s

    contribs: Dict[str, float] = {}
    for i, name in enumerate(names):
        # marginal ES_i
        mes_i = -mu_vec[i] + (k / max(1e-12, sigma_p)) * sig_grad[i]
        contribs[name] = weights[name] * mes_i

    # small normalization so sum(contribs) ≈ total_es
    s = sum(contribs.values())
    if s != 0:
        scale = total_es / s
        for kname in contribs:
            contribs[kname] *= scale

    return ESContribResult(total_es=total_es, contribs=contribs)

# -------------------------- Incremental ES --------------------------

def incremental_es(
    returns_port: Iterable[float],
    returns_asset: Iterable[float],
    alpha: float = 0.975,
    bump_weight: float = 0.01
) -> float:
    """
    Historical incremental ES by bumping a small weight of asset into the portfolio returns
    (assumes returns are additive approximation).
    Returns ΔES ≈ ES(w + ε e_i) - ES(w)
    """
    rp = list(returns_port)
    ra = list(returns_asset)
    n = min(len(rp), len(ra))
    if n == 0: return 0.0
    rp = rp[-n:]; ra = ra[-n:]
    base = historical_es(rp, alpha=alpha)
    bumped = historical_es([rp[i] + bump_weight * ra[i] for i in range(n)], alpha=alpha)
    return bumped - base

# -------------------------- Horizon scaling --------------------------

def horizon_scale_es(es_1d: float, horizon_days: float, iid: bool = True) -> float:
    """
    Scale 1-day ES to H-day ES.
    - If iid=True: ES_H ~ ES_1 * sqrt(H) (Gaussian-ish scaling)
    - Else: linear fallback for conservative stress.
    """
    if horizon_days <= 0: return es_1d
    return es_1d * (math.sqrt(horizon_days) if iid else horizon_days)

# -------------------------- Example / Smoke --------------------------

if __name__ == "__main__":  # pragma: no cover
    import random, time
    random.seed(7)

    # Synthetic returns with slight negative skew
    rets = []
    for _ in range(2000):
        x = random.gauss(0.0003, 0.01)
        # add fat-tail downside
        if random.random() < 0.01:
            x -= random.uniform(0.03, 0.07)
        rets.append(x)

    alpha = 0.975
    hv = historical_var(rets, alpha)
    hes = historical_es(rets, alpha)
    mu, sig = _mean(rets), _stdev(rets)
    pes = parametric_es_normal(mu, sig, alpha)
    ces = parametric_es_cornish_fisher(mu, sig, skew=-0.4, kurt=1.2, alpha=alpha)
    ses = stressed_es(rets, alpha, tail_weight=2.5)
    es_hat, lo, hi = bootstrap_es_ci(rets, alpha, n_boot=500, ci=0.90, seed=1)

    print(f"VaR {int(alpha*100)}%:", round(hv*100, 3), "%")
    print(f"Hist ES:", round(hes*100, 3), "%  Param ES:", round(pes*100, 3), "%  CF ES:", round(ces*100, 3), "%")
    print(f"Stressed ES:", round(ses*100, 3), "%   CI90%: [{round(lo*100,3)}, {round(hi*100,3)}]%")

    # Contributions demo (2-asset)
    w = {"A": 0.6, "B": 0.4}
    mu = {"A": 0.0004, "B": 0.0002}
    cov = {("A","A"): 0.0001, ("B","B"): 0.0002, ("A","B"): 0.00005, ("B","A"): 0.00005}
    c = es_contributions_normal(w, mu, cov, alpha=alpha)
    print("Total ES:", round(c.total_es*100,3), "%; contribs:", {k: round(v*100,3) for k,v in c.contribs.items()})

    # Incremental ES of asset B
    rp = rets
    ra = [r*1.2 for r in rets]
    dES = incremental_es(rp, ra, alpha=alpha, bump_weight=0.02)
    print("Incremental ES (2% bump of B):", round(dES*100, 3), "%")