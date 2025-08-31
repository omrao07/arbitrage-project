# backend/risk/var_engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# =============================================================================
# Utilities (no third-party deps)
# =============================================================================

def _sorted(xs: Iterable[float]) -> List[float]:
    a = list(xs); a.sort(); return a

def _mean(xs: Iterable[float]) -> float:
    xs = list(xs); 
    return sum(xs) / max(1, len(xs))

def _stdev(xs: Iterable[float]) -> float:
    xs = list(xs); n = len(xs)
    if n < 2: return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(0.0, v))

def _quantile(xs_sorted: Sequence[float], q: float) -> float:
    n = len(xs_sorted)
    if n == 0: return 0.0
    if q <= 0: return xs_sorted[0]
    if q >= 1: return xs_sorted[-1]
    pos = (n - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return xs_sorted[lo]
    w = pos - lo
    return (1 - w) * xs_sorted[lo] + w * xs_sorted[hi]

def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def _ncdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def _inv_norm_cdf(p: float, lo: float = -8.0, hi: float = 8.0, tol: float = 1e-7) -> float:
    if p <= 0.0: return -8.0
    if p >= 1.0: return 8.0
    a, b = lo, hi
    while b - a > tol:
        m = 0.5 * (a + b)
        if _ncdf(m) < p: a = m
        else: b = m
    return 0.5 * (a + b)

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# =============================================================================
# Data classes
# =============================================================================

@dataclass
class VaRResult:
    alpha: float                # confidence, e.g. 0.99
    var: float                  # positive loss number (fraction, e.g., 0.025 = 2.5%)
    method: str                 # "historical" | "normal" | "cornish-fisher" | "fhs" | "mc"
    horizon_days: float = 1.0
    details: Dict[str, float] = None  # type: ignore # optional fields (mu, sigma, z, etc.)

@dataclass
class VaRBacktest:
    alpha: float
    n: int
    exceptions: int
    hit_ratio: float
    lr_uc: float                # Kupiec unconditional coverage statistic
    lr_ind: Optional[float]     # Christoffersen independence statistic (None if not computed)
    lr_cc: Optional[float]      # combined conditional coverage
    band: str                   # "green" | "yellow" | "red" (Basel traffic light)

# =============================================================================
# 1) Single-series VaR
# =============================================================================

def var_historical(returns: Iterable[float], *, alpha: float = 0.99) -> VaRResult:
    """
    Historical (non-parametric) 1-day VaR.
    Input: returns (P&L / notional) where losses are negative numbers.
    Output: positive loss number (fraction).
    """
    xs = _sorted(list(returns))
    p = 1.0 - float(alpha)
    q = _quantile(xs, p)
    return VaRResult(alpha=alpha, var=max(0.0, -q), method="historical", details={"q_ret": q})

def var_normal(mu: float, sigma: float, *, alpha: float = 0.99) -> VaRResult:
    """
    Parametric Gaussian VaR. Loss VaR = -(mu + z_p * sigma), with p = 1 - alpha (left tail).
    """
    p = 1.0 - float(alpha)
    z = _inv_norm_cdf(p)
    v = max(0.0, -(mu + z * sigma))
    return VaRResult(alpha=alpha, var=v, method="normal", details={"mu": mu, "sigma": sigma, "z": z})

def var_cornish_fisher(mu: float, sigma: float, skew: float, kurt: float, *, alpha: float = 0.99) -> VaRResult:
    """
    Cornish–Fisher adjusted VaR to account for skew/kurtosis.
    """
    p = 1.0 - float(alpha)
    z = _inv_norm_cdf(p)
    S, K = float(skew), float(kurt)
    z2, z3 = z*z, z*z*z
    z_star = z + (1/6)*(z2-1)*S + (1/24)*(z3-3*z)*K - (1/36)*(2*z3-5*z)*S*S
    v = max(0.0, -(mu + z_star * sigma))
    return VaRResult(alpha=alpha, var=v, method="cornish-fisher",
                     details={"mu": mu, "sigma": sigma, "z*": z_star, "skew": S, "kurt": K})

# =============================================================================
# 2) Portfolio VaR (delta-normal, EWMA/FHS, Monte Carlo)
# =============================================================================

def ewma_vol(returns: Iterable[float], lam: float = 0.94) -> float:
    """
    EWMA volatility of a single series (RiskMetrics). Returns sigma.
    """
    lam = _clip(lam, 0.80, 0.9999)
    s2 = 0.0
    for r in returns:
        s2 = lam * s2 + (1 - lam) * (r * r)
    return math.sqrt(s2)

def ewma_cov_matrix(ret_series: Dict[str, List[float]], lam: float = 0.94) -> Dict[Tuple[str, str], float]:
    """
    EWMA covariance matrix for a dict of {name: returns[]}, aligned by index.
    """
    names = list(ret_series.keys())
    n = min(len(ret_series[n]) for n in names) if names else 0
    cov: Dict[Tuple[str, str], float] = {}
    lam = _clip(lam, 0.80, 0.9999)

    # initialize with zeros
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            cov[(a, b)] = 0.0

    for t in range(n):
        for i, a in enumerate(names):
            ra = ret_series[a][t]
            for j, b in enumerate(names):
                rb = ret_series[b][t]
                cov[(a, b)] = lam * cov[(a, b)] + (1 - lam) * (ra * rb)
    return cov

def portfolio_var_delta_normal(
    weights: Dict[str, float],
    cov: Dict[Tuple[str, str], float],
    mu: Optional[Dict[str, float]] = None,
    *,
    alpha: float = 0.99
) -> VaRResult:
    """
    Delta-normal VaR: portfolio r ~ N(mu_p, sigma_p^2).
    Weights are portfolio weights (sum to 1) or P&L sensitivities.
    """
    names = list(weights.keys())
    w = [weights[k] for k in names]
    mu_vec = [ (mu or {}).get(k, 0.0) for k in names ]
    mu_p = sum(wi * mi for wi, mi in zip(w, mu_vec))

    var_p = 0.0
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            var_p += w[i] * w[j] * cov.get((a, b), 0.0)
    sigma_p = math.sqrt(max(1e-18, var_p))

    p = 1.0 - float(alpha)
    z = _inv_norm_cdf(p)
    v = max(0.0, -(mu_p + z * sigma_p))
    return VaRResult(alpha=alpha, var=v, method="delta-normal",
                     details={"mu_p": mu_p, "sigma_p": sigma_p, "z": z})

def portfolio_var_fhs(
    weights: Dict[str, float],
    returns: Dict[str, List[float]],
    lam: float = 0.94,
    *,
    alpha: float = 0.99
) -> VaRResult:
    """
    Filtered Historical Simulation (FHS):
      1) Standardize each series by EWMA sigma_t
      2) Sample historical standardized shocks ε_t
      3) Re-scale by current σ_T to form portfolio return scenarios
      4) Take empirical quantile
    """
    names = list(weights.keys())
    n = min(len(returns[k]) for k in names) if names else 0
    if n == 0:
        return VaRResult(alpha=alpha, var=0.0, method="fhs", details={})

    # 1) rolling EWMA sigma_t per asset
    sig_t = {k: [] for k in names}
    s2 = {k: 0.0 for k in names}
    for t in range(n):
        for k in names:
            r = returns[k][t]
            s2[k] = lam * s2[k] + (1 - lam) * (r * r)
            sig_t[k].append(math.sqrt(max(1e-18, s2[k])))

    # current sigma_T for each
    sig_T = {k: sig_t[k][-1] for k in names}

    # 2) standardized shocks
    eps = []
    for t in range(n):
        row = []
        for k in names:
            s = sig_t[k][t]
            r = returns[k][t]
            row.append(r / max(1e-12, s))
        eps.append(row)

    # 3) re-scale by current σ_T, sum by weights -> portfolio scenarios
    scen: List[float] = []
    for t in range(n):
        pt = 0.0
        for i, k in enumerate(names):
            pt += weights[k] * eps[t][i] * sig_T[k]
        scen.append(pt)

    # 4) empirical quantile of portfolio returns
    xs = _sorted(scen)
    p = 1.0 - float(alpha)
    q = _quantile(xs, p)
    return VaRResult(alpha=alpha, var=max(0.0, -q), method="fhs", details={"samples": len(xs)})

def portfolio_var_mc(
    weights: Dict[str, float],
    mu: Dict[str, float],
    cov: Dict[Tuple[str, str], float],
    *,
    alpha: float = 0.99,
    n_sim: int = 50_000,
    seed: Optional[int] = None
) -> VaRResult:
    """
    Monte Carlo VaR under joint Normal using covariance matrix (no external RNG libs).
    Simple Cholesky (robustified) + Box-Muller for normals.
    """
    import random
    if seed is not None: random.seed(seed)

    names = list(weights.keys())
    m = len(names)
    # Build dense matrix
    Sigma = [[cov.get((names[i], names[j]), 0.0) for j in range(m)] for i in range(m)]

    # Robust Cholesky (add small jitter if needed)
    L = _cholesky_psd(Sigma)

    # weights, mean vector
    w = [weights[k] for k in names]
    mu_vec = [mu.get(k, 0.0) for k in names]

    def _box_muller() -> Tuple[float, float]:
        u1 = max(1e-12, random.random())
        u2 = random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2 * math.pi * u2)
        return z0, z1

    rets: List[float] = []
    sims = 0
    while sims < n_sim:
        z = []
        # generate m standard normals
        while len(z) < m:
            a, b = _box_muller()
            z.append(a)
            if len(z) < m: z.append(b)
        # correlate: x = L * z
        x = [0.0] * m
        for i in range(m):
            s = 0.0
            for j in range(i + 1):  # L lower-tri
                s += L[i][j] * z[j]
            x[i] = s + mu_vec[i]
        # portfolio return
        r = sum(w[i] * x[i] for i in range(m))
        rets.append(r)
        sims += 1

    xs = _sorted(rets)
    p = 1.0 - float(alpha)
    q = _quantile(xs, p)
    return VaRResult(alpha=alpha, var=max(0.0, -q), method="mc", details={"samples": len(xs)})

def _cholesky_psd(A: List[List[float]], jitter: float = 1e-12, max_tries: int = 5) -> List[List[float]]:
    """Basic Cholesky with diagonal jitter for near-PSD matrices."""
    n = len(A)
    for t in range(max_tries):
        try:
            L = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    s = sum(L[i][k] * L[j][k] for k in range(j))
                    if i == j:
                        val = A[i][i] - s
                        if val <= 0:
                            raise ValueError("not PD")
                        L[i][i] = math.sqrt(val)
                    else:
                        L[i][j] = (A[i][j] - s) / L[j][j]
            return L
        except Exception:
            # add jitter to diagonal
            for i in range(n):
                A[i][i] += jitter * (10 ** t)
    # final fallback: diagonalize with stdevs only
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = math.sqrt(max(1e-18, A[i][i]))
    return L

# =============================================================================
# 3) Component / Incremental VaR (Euler under Normal)
# =============================================================================

@dataclass
class VaRContrib:
    total_var: float
    contribs: Dict[str, float]     # w_i * mVaR_i; approximately sums to total_var

def var_contributions_normal(
    weights: Dict[str, float],
    mu: Dict[str, float],
    cov: Dict[Tuple[str, str], float],
    *,
    alpha: float = 0.99
) -> VaRContrib:
    """
    Euler allocation under Normal:
      VaR_p = -(μ_p) + z * σ_p,  z = Φ^{-1}(1-α)
      mVaR_i = ∂VaR/∂w_i = -μ_i + z * (Σ w)_i / σ_p
      c_i = w_i * mVaR_i
    """
    names = list(weights.keys())
    w = [weights[k] for k in names]
    mu_vec = [mu.get(k, 0.0) for k in names]

    mu_p = sum(wi * mi for wi, mi in zip(w, mu_vec))
    var_p = 0.0
    m = len(names)
    for i in range(m):
        for j in range(m):
            var_p += w[i] * w[j] * cov.get((names[i], names[j]), 0.0)
    sigma_p = math.sqrt(max(1e-18, var_p))

    z = _inv_norm_cdf(1.0 - float(alpha))
    total = max(0.0, -(mu_p) + z * sigma_p)

    # Sigma * w
    sigw = [0.0] * m
    for i in range(m):
        s = 0.0
        for j in range(m):
            s += cov.get((names[i], names[j]), 0.0) * w[j]
        sigw[i] = s

    contribs: Dict[str, float] = {}
    for i, name in enumerate(names):
        mvar_i = -mu_vec[i] + (z / max(1e-12, sigma_p)) * sigw[i]
        contribs[name] = weights[name] * mvar_i

    # Normalize tiny mismatch
    s = sum(contribs.values())
    if s != 0:
        scale = total / s
        for k in contribs:
            contribs[k] *= scale
    return VaRContrib(total_var=total, contribs=contribs)

def incremental_var_historical(
    port_returns: Iterable[float],
    asset_returns: Iterable[float],
    *,
    alpha: float = 0.99,
    bump_weight: float = 0.01
) -> float:
    """
    Historical incremental VaR by bumping a small weight of an asset.
    """
    rp = list(port_returns)
    ra = list(asset_returns)
    n = min(len(rp), len(ra))
    if n == 0: return 0.0
    rp = rp[-n:]; ra = ra[-n:]
    base = var_historical(rp, alpha=alpha).var
    bumped = var_historical([rp[i] + bump_weight * ra[i] for i in range(n)], alpha=alpha).var
    return bumped - base

# =============================================================================
# 4) Rolling VaR & Backtesting
# =============================================================================

def rolling_var(
    returns: List[float],
    window: int = 250,
    *,
    alpha: float = 0.99,
    method: str = "historical"  # "historical" | "normal"
) -> List[VaRResult]:
    out: List[VaRResult] = []
    if window <= 1 or not returns: return out
    for i in range(window, len(returns) + 1):
        sub = returns[i - window:i]
        if method == "normal":
            mu = _mean(sub); sig = _stdev(sub)
            out.append(var_normal(mu, sig, alpha=alpha))
        else:
            out.append(var_historical(sub, alpha=alpha))
    return out

def basel_traffic_light(exceptions: int, n: int, alpha: float = 0.99) -> str:
    """
    Basel traffic light (for α=0.99, n≈250). Uses standard buckets.
    """
    if alpha >= 0.99 and 200 <= n <= 300:
        # 0-4 green, 5-9 yellow, >=10 red
        if exceptions <= 4: return "green"
        if exceptions <= 9: return "yellow"
        return "red"
    # generic: compare to expected n*(1-α)
    exp = n * (1.0 - alpha)
    # allow 2*sqrt(exp) band for green
    if exceptions <= exp + 2.0 * math.sqrt(max(1.0, exp)):
        return "green"
    if exceptions <= exp + 4.0 * math.sqrt(max(1.0, exp)):
        return "yellow"
    return "red"

def kupiec_pof(exceptions: int, n: int, alpha: float) -> float:
    """
    Kupiec LR_uc statistic (compare to χ² with 1 dof; 3.841 ~ 95%).
    """
    if n <= 0: return 0.0
    p = 1.0 - alpha
    x = exceptions
    pi_hat = x / n if n > 0 else 0.0
    # avoid log(0)
    def _ll(k: int, n_: int, p_: float) -> float:
        if p_ <= 0 or p_ >= 1:
            return -1e9
        return (n_-k)*math.log(1-p_) + k*math.log(p_)
    ll1 = _ll(x, n, p)
    ll0 = _ll(x, n, pi_hat) if 0 < pi_hat < 1 else -1e9
    lr = -2.0 * (ll1 - ll0)
    return max(0.0, lr)

def christoffersen_ind(exceed: List[bool]) -> float:
    """
    Christoffersen independence LR_ind (χ² with 1 dof). Needs exceedance sequence.
    """
    if not exceed: return 0.0
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(exceed)):
        a, b = exceed[i-1], exceed[i]
        if not a and not b: n00 += 1
        if not a and b:     n01 += 1
        if a and not b:     n10 += 1
        if a and b:         n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0
    pi  = (n01 + n11) / max(1, (n0 + n1))
    def _L(n_, k_, p_):
        if p_ <= 0 or p_ >= 1: return -1e9
        return (n_-k_) * math.log(1-p_) + k_ * math.log(p_)
    ll_ind = _L(n0, n01, pi0) + _L(n1, n11, pi1)
    ll_p  = _L(n0, n01, pi) + _L(n1, n11, pi)
    lr = -2.0 * (ll_p - ll_ind)
    return max(0.0, lr)

def backtest_var(
    returns: List[float],
    vars_series: List[VaRResult]
) -> VaRBacktest:
    """
    Compare realized returns to VaR series (aligned to the same dates: len(vars)=len(returns)-window+1).
    Exceedance when return < -VaR (i.e., loss worse than VaR).
    """
    n = min(len(returns), len(vars_series))
    if n == 0:
        return VaRBacktest(alpha=0.99, n=0, exceptions=0, hit_ratio=0.0, lr_uc=0.0, lr_ind=None, lr_cc=None, band="green")
    # Align the tail: we compare the last n observations
    rets = returns[-n:]
    vas = vars_series[-n:]
    alpha = vas[-1].alpha
    exceed = [ (rets[i] < -vas[i].var) for i in range(n) ]
    x = sum(1 for e in exceed if e)
    lr_uc = kupiec_pof(x, n, alpha)
    lr_ind = christoffersen_ind(exceed) if n >= 2 else None
    lr_cc = (lr_uc + (lr_ind or 0.0)) if lr_ind is not None else None
    band = basel_traffic_light(x, n, alpha)
    hit_ratio = 1.0 - (x / max(1, n))
    return VaRBacktest(alpha=alpha, n=n, exceptions=x, hit_ratio=hit_ratio, lr_uc=lr_uc, lr_ind=lr_ind, lr_cc=lr_cc, band=band)

# =============================================================================
# 5) Scaling helpers & currency conversion
# =============================================================================

def scale_horizon(var_1d: float, horizon_days: float, iid: bool = True) -> float:
    """
    Scale one-day VaR to H-day horizon. If iid=True → sqrt(H); else linear fallback.
    """
    if horizon_days <= 0: return var_1d
    return var_1d * (math.sqrt(horizon_days) if iid else horizon_days)

def to_currency(var_frac: float, notional: float) -> float:
    """Convert fractional VaR to currency VaR."""
    return abs(var_frac) * abs(notional)

# =============================================================================
# Demo / Smoke
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    import random, time
    random.seed(42)

    # Synthetic daily returns with a few fat-tail hits
    rets = []
    for _ in range(1000):
        r = random.gauss(0.0003, 0.012)
        if random.random() < 0.01:
            r -= random.uniform(0.03, 0.07)
        rets.append(r)

    # Single-series VaR
    hv = var_historical(rets[-250:], alpha=0.99)
    nv = var_normal(_mean(rets[-250:]), _stdev(rets[-250:]), alpha=0.99)
    cv = var_cornish_fisher(_mean(rets[-250:]), _stdev(rets[-250:]), skew=-0.4, kurt=1.0, alpha=0.99)
    print("Hist:", round(hv.var*100,3), "%  Norm:", round(nv.var*100,3), "%  CF:", round(cv.var*100,3), "%")

    # Rolling VaR & backtest
    rvars = rolling_var(rets, window=250, alpha=0.99, method="historical")
    bt = backtest_var(rets[-len(rvars):], rvars)
    print("Backtest:", bt.band, "exceptions:", bt.exceptions, "LR_uc≈", round(bt.lr_uc,3))

    # Portfolio delta-normal VaR
    weights = {"A": 0.6, "B": 0.4}
    ra = [r + 0.0002 for r in rets[-500:]]
    rb = [0.8*r + 0.0001 for r in rets[-500:]]
    cov = ewma_cov_matrix({"A": ra, "B": rb}, lam=0.94)
    mu = {"A": _mean(ra), "B": _mean(rb)}
    pvar = portfolio_var_delta_normal(weights, cov, mu, alpha=0.99)
    print("Portfolio VaR (delta-normal):", round(pvar.var*100,3), "%")

    # FHS VaR
    fhs = portfolio_var_fhs(weights, {"A": ra, "B": rb}, lam=0.94, alpha=0.99)
    print("FHS VaR:", round(fhs.var*100,3), "%")

    # MC VaR
    mc = portfolio_var_mc(weights, mu, cov, alpha=0.99, n_sim=20000, seed=7)
    print("MC VaR:", round(mc.var*100,3), "%")

    # Contributions
    vc = var_contributions_normal(weights, mu, cov, alpha=0.99)
    print("Contrib sum≈", round(sum(vc.contribs.values())*100,3), "% of total", round(vc.total_var*100,3), "%")