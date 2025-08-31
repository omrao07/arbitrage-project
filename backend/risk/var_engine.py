# backend/risk/var_engine.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

# ---------------------------------------------------------------------
# small helpers (no external deps)
# ---------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def _stdev(xs: List[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - ddof)
    return math.sqrt(var)

def _skew(xs: List[float]) -> float:
    n = len(xs)
    if n < 3: return 0.0
    m = _mean(xs); s = _stdev(xs) or 1e-12
    return (sum(((x - m) / s) ** 3 for x in xs) * n) / ((n - 1) * (n - 2))

def _excess_kurt(xs: List[float]) -> float:
    n = len(xs)
    if n < 4: return 0.0
    m = _mean(xs); s2 = (_stdev(xs) or 1e-12) ** 2
    num = sum(((x - m) ** 4) for x in xs) * n * (n + 1)
    den = (n - 1) * (n - 2) * (n - 3) * (s2 ** 2)
    g2 = num / max(1e-12, den) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return g2

def _quantile(xs: List[float], q: float) -> float:
    if not xs: return 0.0
    ys = sorted(xs)
    if q <= 0: return ys[0]
    if q >= 1: return ys[-1]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w

def _norm_ppf(p: float) -> float:
    # Acklam/Beasley–Springer inverse normal CDF
    if p <= 0.0: return -1e9
    if p >= 1.0: return +1e9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425; phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def _sqrt_time_scale(x: float, days: int) -> float:
    return x * math.sqrt(max(1, days))

# ---------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------

@dataclass
class VarEstimate:
    alpha: float        # e.g., 0.99 (99% VaR)
    var: float          # VaR in RETURN space (negative number = loss)
    method: str         # 'historical' | 'gaussian' | 'cornish-fisher' | 'ewma'
    mean: float
    stdev: float
    n: int
    horizon_days: int = 1

# ---------------------------------------------------------------------
# core engine
# ---------------------------------------------------------------------

class VaREngine:
    """
    All returns are **fractions** (e.g., -0.012 = -1.2%).
    VaR returned is in **return space** (typically negative). For cash VaR:
        cash_var = -estimate.var * notional
    """

    # ---------- point VaR ----------

    @staticmethod
    def historical(returns: Iterable[float],
                   alpha: float = 0.99,
                   horizon_days: int = 1) -> VarEstimate:
        r = [float(x) for x in returns]
        n = len(r)
        if n == 0:
            return VarEstimate(alpha, 0.0, "historical", 0.0, 0.0, 0, horizon_days)
        mu = _mean(r); sd = _stdev(r) if n > 1 else 0.0
        var_1d = _quantile(sorted(r), 1 - alpha)  # left-tail
        var_h = var_1d if horizon_days <= 1 else (mu * (horizon_days - 1) + _sqrt_time_scale(var_1d - mu, horizon_days) + mu)
        # The above keeps mean separate and scales the centered quantile by sqrt-time (pragmatic)
        return VarEstimate(alpha, var_h, "historical", mu, sd, n, horizon_days)

    @staticmethod
    def gaussian(returns: Iterable[float],
                 alpha: float = 0.99,
                 horizon_days: int = 1) -> VarEstimate:
        r = [float(x) for x in returns]
        n = len(r)
        mu = _mean(r) if n else 0.0
        sd = _stdev(r) if n > 1 else 0.0
        if sd <= 0:
            return VarEstimate(alpha, mu, "gaussian", mu, sd, n, horizon_days)
        z = _norm_ppf(1 - alpha)  # negative
        mu_h = mu * horizon_days
        sd_h = sd * math.sqrt(max(1, horizon_days))
        var_h = mu_h + sd_h * z
        return VarEstimate(alpha, var_h, "gaussian", mu, sd, n, horizon_days)

    @staticmethod
    def cornish_fisher(returns: Iterable[float],
                       alpha: float = 0.99,
                       horizon_days: int = 1) -> VarEstimate:
        r = [float(x) for x in returns]
        n = len(r)
        mu = _mean(r) if n else 0.0
        sd = _stdev(r) if n > 1 else 0.0
        if sd <= 0:
            return VarEstimate(alpha, mu, "cornish-fisher", mu, sd, n, horizon_days)
        g1 = _skew(r); g2 = _excess_kurt(r)
        z = _norm_ppf(1 - alpha)
        z_cf = (z
                + (g1/6.0)*(z*z - 1)
                + (g2/24.0)*(z**3 - 3*z)
                - (g1*g1/36.0)*(2*z**3 - 5*z))
        mu_h = mu * horizon_days
        sd_h = sd * math.sqrt(max(1, horizon_days))
        var_h = mu_h + sd_h * z_cf
        return VarEstimate(alpha, var_h, "cornish-fisher", mu, sd, n, horizon_days)

    @staticmethod
    def ewma(returns: Iterable[float],
             alpha: float = 0.99,
             lam: float = 0.94,         # RiskMetrics default for daily
             horizon_days: int = 1) -> VarEstimate:
        r = [float(x) for x in returns]
        n = len(r)
        if n == 0:
            return VarEstimate(alpha, 0.0, "ewma", 0.0, 0.0, 0, horizon_days)
        mu = _mean(r)
        # EWMA variance
        var = 0.0
        for x in reversed(r):
            var = lam * var + (1 - lam) * (x - mu) * (x - mu)
        sd = math.sqrt(max(0.0, var))
        z = _norm_ppf(1 - alpha)
        mu_h = mu * horizon_days
        sd_h = sd * math.sqrt(max(1, horizon_days))
        var_h = mu_h + sd_h * z
        return VarEstimate(alpha, var_h, "ewma", mu, sd, n, horizon_days)

    # ---------- rolling windows ----------

    @staticmethod
    def rolling_historical(returns: Iterable[float], window: int, alpha: float = 0.99) -> List[VarEstimate]:
        r = [float(x) for x in returns]
        out: List[VarEstimate] = []
        for i in range(window, len(r) + 1):
            out.append(VaREngine.historical(r[i-window:i], alpha))
        return out

    # ---------- portfolio VaR ----------

    @staticmethod
    def portfolio_historical(returns_by_asset: Dict[str, List[float]],
                             weights: Dict[str, float],
                             alpha: float = 0.99) -> VarEstimate:
        """
        Combine assets by weights (sum ~ 1). Align by index; compute historical VaR on portfolio return series.
        """
        keys = [k for k in weights.keys() if k in returns_by_asset]
        if not keys:
            return VarEstimate(alpha, 0.0, "historical", 0.0, 0.0, 0)
        n = min(len(returns_by_asset[k]) for k in keys)
        port: List[float] = []
        for t in range(n):
            rt = 0.0
            for k in keys:
                rt += weights.get(k, 0.0) * float(returns_by_asset[k][t])
            port.append(rt)
        return VaREngine.historical(port, alpha)

    @staticmethod
    def portfolio_parametric_gaussian(returns_by_asset: Dict[str, List[float]],
                                      weights: Dict[str, float],
                                      alpha: float = 0.99,
                                      horizon_days: int = 1,
                                      use_mean: bool = False) -> Tuple[VarEstimate, List[List[float]]]:
        """
        Parametric Normal portfolio VaR using covariance:
          σ_p = sqrt(w' Σ w), VaR = μ_p + z * σ_p
        Returns (VarEstimate, covariance_matrix).
        """
        keys = [k for k in weights.keys() if k in returns_by_asset]
        if not keys:
            return VarEstimate(alpha, 0.0, "gaussian", 0.0, 0.0, 0, horizon_days), []
        # build matrix (T x N)
        n = min(len(returns_by_asset[k]) for k in keys)
        T = n; N = len(keys)
        # means
        mu_vec = [_mean(returns_by_asset[k][:n]) for k in keys]
        # covariance
        C = [[0.0]*N for _ in range(N)]
        for i in range(N):
            xi = [returns_by_asset[keys[i]][t] for t in range(n)]
            mi = mu_vec[i]
            for j in range(i, N):
                xj = [returns_by_asset[keys[j]][t] for t in range(n)]
                mj = mu_vec[j]
                cov = sum((xi[t]-mi)*(xj[t]-mj) for t in range(n)) / max(1, (n-1))
                C[i][j] = C[j][i] = cov
        # portfolio stats
        w = [weights[k] for k in keys]
        # σ_p^2 = w' Σ w
        var_p = 0.0
        for i in range(N):
            for j in range(N):
                var_p += w[i] * C[i][j] * w[j]
        sd_p = math.sqrt(max(0.0, var_p))
        mu_p = sum(w[i] * mu_vec[i] for i in range(N)) if use_mean else 0.0

        z = _norm_ppf(1 - alpha)
        mu_h = mu_p * horizon_days
        sd_h = sd_p * math.sqrt(max(1, horizon_days))
        var_ret = mu_h + sd_h * z
        est = VarEstimate(alpha, var_ret, "gaussian", mu_p, sd_p, T, horizon_days)
        return est, C

    # ---------- component VaR (Euler) for parametric normal ----------

    @staticmethod
    def component_var_parametric(weights: Dict[str, float],
                                 cov: List[List[float]],
                                 alpha: float = 0.99,
                                 horizon_days: int = 1) -> Dict[str, float]:
        """
        Euler contributions RC_i = w_i * ∂VaR/∂w_i
        For Normal VaR (zero-mean): VaR = z * σ_p
            σ_p = sqrt(w' Σ w),  ∂σ_p/∂w_i = (Σ w)_i / σ_p
            => RC_i = z * w_i * (Σ w)_i / σ_p
        Returns contributions in **return space** (sum ≈ VaR).
        """
        keys = list(weights.keys())
        N = len(keys)
        w = [weights[k] for k in keys]
        # portfolio sigma
        var_p = 0.0
        for i in range(N):
            for j in range(N):
                var_p += w[i] * cov[i][j] * w[j]
        sd_p = math.sqrt(max(1e-12, var_p))
        # Sigma * w
        Sigw = [0.0]*N
        for i in range(N):
            Sigw[i] = sum(cov[i][j] * w[j] for j in range(N))
        z = _norm_ppf(1 - alpha)
        scale = z * math.sqrt(max(1, horizon_days)) / sd_p
        contrib = {keys[i]: w[i] * Sigw[i] * scale for i in range(N)}
        # Numerical guard to keep sum close to VaR
        sump = sum(contrib.values()) or 1.0
        target = z * sd_p * math.sqrt(max(1, horizon_days))
        if sump != 0:
            k = target / sump
            contrib = {k_: v * k for k_, v in contrib.items()}
        return contrib

    # ---------- backtest (Kupiec POF) ----------

    @staticmethod
    def kupiec_pof(breach_count: int, n_obs: int, alpha: float) -> Dict[str, float]:
        """
        Kupiec Proportion of Failures test:
          H0: breach rate = q,  q = 1 - alpha
          LR = -2 ln [ (1-q)^(n-x) q^x / ( (1 - x/n)^(n-x) (x/n)^x ) ] ~ χ^2(1)
        Returns dict with 'breach_rate', 'expected', 'LR', 'p_value'
        """
        q = 1 - alpha
        x = breach_count
        n = max(1, n_obs)
        pi = x / n
        # guard edges
        pi = min(max(pi, 1e-9), 1 - 1e-9)
        q = min(max(q, 1e-9), 1 - 1e-9)
        LR = -2.0 * ( (n - x) * math.log((1 - q)/(1 - pi)) + x * math.log(q/pi) )
        # p-value from chi^2(1) ~ 1 - CDF; use survival ≈ exp(-LR/2) for 1 dof tail
        p = math.exp(-LR / 2.0)
        return {"breach_rate": pi, "expected": (1 - alpha), "LR": LR, "p_value": p, "n": n, "breaches": x}

    @staticmethod
    def backtest_series(returns: Iterable[float],
                        alpha: float = 0.99,
                        method: str = "historical",
                        window: int = 250) -> Dict[str, float]:
        """
        Rolling one-step VaR backtest. Counts breaches where r_t <= VaR_{t-1}.
        """
        r = [float(x) for x in returns]
        if len(r) <= window:
            return {"n": 0, "breaches": 0, "breach_rate": 0.0, "expected": (1 - alpha), "LR": 0.0, "p_value": 1.0}
        breaches = 0
        for t in range(window, len(r)):
            hist = r[t - window:t]
            if method == "gaussian":
                est = VaREngine.gaussian(hist, alpha)
            elif method == "cornish-fisher":
                est = VaREngine.cornish_fisher(hist, alpha)
            elif method == "ewma":
                est = VaREngine.ewma(hist, alpha)
            else:
                est = VaREngine.historical(hist, alpha)
            if r[t] <= est.var:
                breaches += 1
        res = VaREngine.kupiec_pof(breaches, len(r) - window, alpha)
        return res

# ---------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(7)
    rets = [random.gauss(0.0003, 0.012) for _ in range(1000)]

    print("=== Point VaR ===")
    print("Hist 99%:", VaREngine.historical(rets, 0.99).var)
    print("Gauss 99%:", VaREngine.gaussian(rets, 0.99).var)
    print("CF   99%:", VaREngine.cornish_fisher(rets, 0.99).var)
    print("EWMA 99%:", VaREngine.ewma(rets, 0.99).var)

    print("\n=== Backtest (Kupiec) ===")
    print(VaREngine.backtest_series(rets, 0.99, method="historical", window=250))

    print("\n=== Portfolio (parametric) ===")
    A = [random.gauss(0.0004, 0.015) for _ in range(800)]
    B = [0.5*a + random.gauss(0.0002, 0.01) for a in A]  # correlated-ish
    C = [random.gauss(0.0001, 0.02) for _ in range(800)]
    w = {"A": 0.4, "B": 0.4, "C": 0.2}
    est, cov = VaREngine.portfolio_parametric_gaussian({"A": A, "B": B, "C": C}, w, 0.99)
    print("Param VaR (ret):", est.var)
    comp = VaREngine.component_var_parametric(w, cov, 0.99)
    print("Component VaR:", comp)