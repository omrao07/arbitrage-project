# backend/risk/expected_shortfall.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

# ---------------------------------------------------------------------
# Helpers (no external deps)
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
    if n < 3:
        return 0.0
    m = _mean(xs)
    s = _stdev(xs, ddof=1) or 1e-12
    return (sum(((x - m) / s) ** 3 for x in xs) * n) / ((n - 1) * (n - 2))

def _excess_kurt(xs: List[float]) -> float:
    n = len(xs)
    if n < 4:
        return 0.0
    m = _mean(xs)
    s2 = (_stdev(xs, ddof=1) or 1e-12) ** 2
    num = sum(((x - m) ** 4) for x in xs) * n * (n + 1)
    den = (n - 1) * (n - 2) * (n - 3) * (s2 ** 2)
    g2 = num / max(1e-12, den) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return g2  # excess kurtosis

def _quantile(xs: List[float], q: float) -> float:
    """
    Simple quantile for sorted xs, with linear interpolation.
    For left tail (e.g., q=0.05) we expect xs to be returns (can be negative).
    """
    if not xs:
        return 0.0
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w

def _norm_ppf(p: float) -> float:
    # Acklam/Beasley–Springer inverse normal CDF approximation
    if p <= 0.0: return -1e9
    if p >= 1.0: return +1e9
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------

@dataclass
class ESEstimate:
    alpha: float           # tail level, e.g., 0.975 for 97.5% ES
    var: float             # VaR at alpha (return threshold; typically negative for losses)
    es: float              # Expected Shortfall (average loss beyond VaR; negative number in return space)
    method: str            # 'historical' | 'gaussian' | 'cornish-fisher'
    mean: float
    stdev: float
    n: int

class ExpectedShortfall:
    """
    Compute portfolio/strategy Expected Shortfall (a.k.a. CVaR).
    Returns are in **fractional** terms (e.g., -0.0125 = -1.25%).
    """

    # ------------- Point estimates -------------

    @staticmethod
    def historical(returns: Iterable[float], alpha: float = 0.975) -> ESEstimate:
        """
        Historical ES: VaR is alpha-quantile of empirical distribution of returns.
        ES is the mean of returns <= VaR.
        """
        r = sorted(float(x) for x in returns)
        n = len(r)
        if n == 0:
            return ESEstimate(alpha, 0.0, 0.0, "historical", 0.0, 0.0, 0)
        mu = _mean(r)
        sd = _stdev(r) if n > 1 else 0.0
        var = _quantile(r, 1 - alpha)  # left tail quantile (e.g., 2.5% if alpha=97.5%)
        tail = [x for x in r if x <= var]
        es = (sum(tail) / max(1, len(tail))) if tail else var
        return ESEstimate(alpha, var, es, "historical", mu, sd, n)

    @staticmethod
    def gaussian(returns: Iterable[float], alpha: float = 0.975) -> ESEstimate:
        """
        Parametric Normal ES: VaR = μ + σ z_q ; ES = μ - σ φ(z_q)/(1-α)
        using left-tail convention (losses are negative).
        """
        r = [float(x) for x in returns]
        n = len(r)
        mu = _mean(r) if n else 0.0
        sd = _stdev(r) if n > 1 else 0.0
        if sd <= 0:
            return ESEstimate(alpha, mu, mu, "gaussian", mu, sd, n)
        q = 1 - alpha
        z = _norm_ppf(q)  # negative for q<0.5
        phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        var = mu + sd * z
        es = mu - sd * (phi / q)
        return ESEstimate(alpha, var, es, "gaussian", mu, sd, n)

    @staticmethod
    def cornish_fisher(returns: Iterable[float], alpha: float = 0.975) -> ESEstimate:
        """
        Cornish–Fisher expansion to adjust Normal quantile for skew/kurtosis.
        ES here uses a trapezoid integration of adjusted tail (practical & robust).
        """
        r = [float(x) for x in returns]
        n = len(r)
        if n == 0:
            return ESEstimate(alpha, 0.0, 0.0, "cornish-fisher", 0.0, 0.0, 0)
        mu = _mean(r)
        sd = _stdev(r) if n > 1 else 0.0
        if sd <= 0:
            return ESEstimate(alpha, mu, mu, "cornish-fisher", mu, sd, n)

        g1 = _skew(r)
        g2 = _excess_kurt(r)
        q = 1 - alpha
        z = _norm_ppf(q)

        # Cornish-Fisher adjusted quantile (up to kurtosis)
        z_cf = (z
                + (g1 / 6.0) * (z**2 - 1)
                + (g2 / 24.0) * (z**3 - 3*z)
                - (g1**2 / 36.0) * (2*z**3 - 5*z))

        var = mu + sd * z_cf

        # Approx ES by integrating adjusted tail via numeric slices between q and 0
        # (coarse but avoids full CF-PDF). Use 50 slices.
        slices = 50
        acc = 0.0
        last_q = 0.0
        last_zcf = _norm_ppf(last_q + 1e-9)  # near -inf; clamp
        last_adj = (last_zcf
                    + (g1 / 6.0) * (last_zcf**2 - 1)
                    + (g2 / 24.0) * (last_zcf**3 - 3*last_zcf)
                    - (g1**2 / 36.0) * (2*last_zcf**3 - 5*last_zcf))
        for i in range(1, slices + 1):
            qi = q * i / slices
            zi = _norm_ppf(qi)
            adj = (zi
                   + (g1 / 6.0) * (zi**2 - 1)
                   + (g2 / 24.0) * (zi**3 - 3*zi)
                   - (g1**2 / 36.0) * (2*zi**3 - 5*zi))
            # trapezoid on quantile space produces average return level
            ri = mu + sd * adj
            rlast = mu + sd * last_adj
            acc += 0.5 * (ri + rlast) * (qi - last_q)
            last_q, last_adj = qi, adj

        es = acc / max(1e-12, q)
        return ESEstimate(alpha, var, es, "cornish-fisher", mu, sd, n)

    # ------------- Rolling windows -------------

    @staticmethod
    def rolling_historical(returns: Iterable[float], window: int, alpha: float = 0.975) -> List[ESEstimate]:
        r = [float(x) for x in returns]
        out: List[ESEstimate] = []
        for i in range(window, len(r) + 1):
            out.append(ExpectedShortfall.historical(r[i - window:i], alpha))
        return out

    # ------------- Portfolio ES (simple) -------------

    @staticmethod
    def portfolio_historical(returns_by_asset: Dict[str, List[float]],
                             weights: Dict[str, float],
                             alpha: float = 0.975) -> ESEstimate:
        """
        Combine assets by weights (sum to ~1). Lengths should match; we align by index.
        ES is computed on the combined return series historically.
        """
        keys = [k for k in weights.keys() if k in returns_by_asset]
        if not keys:
            return ESEstimate(alpha, 0.0, 0.0, "historical", 0.0, 0.0, 0)
        n = min(len(returns_by_asset[k]) for k in keys)
        port: List[float] = []
        for t in range(n):
            ret_t = 0.0
            for k in keys:
                ret_t += (weights.get(k, 0.0) * float(returns_by_asset[k][t]))
            port.append(ret_t)
        return ExpectedShortfall.historical(port, alpha)

    @staticmethod
    def component_es_historical(returns_by_asset: Dict[str, List[float]],
                                weights: Dict[str, float],
                                alpha: float = 0.975,
                                eps: float = 1e-4) -> Tuple[ESEstimate, Dict[str, float]]:
        """
        Component ES via finite-difference marginal contributions:
          CES_i ≈ w_i * ∂ES/∂w_i  ≈ w_i * (ES(w+e_i*eps) - ES(w-e_i*eps)) / (2*eps)
        Returns (portfolio ES estimate, {asset: contribution}).
        """
        base = ExpectedShortfall.portfolio_historical(returns_by_asset, weights, alpha)
        contrib: Dict[str, float] = {}
        keys = [k for k in weights.keys() if k in returns_by_asset]
        wsum = sum(weights.get(k, 0.0) for k in keys) or 1.0

        for k in keys:
            w_up = dict(weights);  w_down = dict(weights)
            w_up[k] = weights[k] + eps
            w_down[k] = max(0.0, weights[k] - eps)

            # re-normalize to keep scale stable
            s_up = sum(w_up.values()) or 1.0
            s_dn = sum(w_down.values()) or 1.0
            w_up = {a: w_up[a]/s_up for a in w_up}
            w_down = {a: w_down[a]/s_dn for a in w_down}

            es_up = ExpectedShortfall.portfolio_historical(returns_by_asset, w_up, alpha).es
            es_dn = ExpectedShortfall.portfolio_historical(returns_by_asset, w_down, alpha).es
            d_es = (es_up - es_dn) / (2 * eps)
            contrib[k] = weights[k] * d_es

        # Scale contributions to sum approximately to ES (numerical drift guard)
        total_c = sum(contrib.values()) or 1.0
        if total_c != 0:
            scale = base.es / total_c
            contrib = {k: v * scale for k, v in contrib.items()}
        return base, contrib

    # ------------- Backtest (basic) -------------

    @staticmethod
    def backtest_var_es(returns: Iterable[float],
                        alpha: float = 0.975,
                        method: str = "historical",
                        window: Optional[int] = None) -> Dict[str, float]:
        """
        Very light backtest:
          - If window is provided: rolling VaR/ES; count VaR breaches and average tail loss.
          - Else: single-shot on all returns.
        Metrics: breach_rate, expected_breach_rate, avg_tail_loss_vs_es, n
        """
        r = [float(x) for x in returns]
        n = len(r)
        if n == 0:
            return {"n": 0, "breach_rate": 0.0, "expected_breach_rate": (1 - alpha), "avg_tail_loss_vs_es": 0.0}

        def est(seq: List[float]) -> ESEstimate:
            if method == "gaussian":
                return ExpectedShortfall.gaussian(seq, alpha)
            if method == "cornish-fisher":
                return ExpectedShortfall.cornish_fisher(seq, alpha)
            return ExpectedShortfall.historical(seq, alpha)

        if window and window >= 30 and window < n:
            breaches = 0
            tail_losses: List[float] = []
            for i in range(window, n):
                e = est(r[i - window:i])
                x = r[i]
                if x <= e.var:    # breach (return worse than VaR)
                    breaches += 1
                    tail_losses.append(x)
            breach_rate = breaches / (n - window)
            avg_tail = (sum(tail_losses) / max(1, len(tail_losses))) if tail_losses else 0.0
            return {
                "n": n - window,
                "breach_rate": breach_rate,
                "expected_breach_rate": (1 - alpha),
                "avg_tail_loss_vs_es": (avg_tail - e.es) if (tail_losses and 'e' in locals()) else 0.0,
            }
        else:
            e = est(r)
            # single-shot: compare realized tail mean to ES
            r_sorted = sorted(r)
            tail = [x for x in r_sorted if x <= e.var]
            avg_tail = (sum(tail) / max(1, len(tail))) if tail else 0.0
            return {
                "n": n,
                "breach_rate": len(tail) / n,
                "expected_breach_rate": (1 - alpha),
                "avg_tail_loss_vs_es": (avg_tail - e.es),
            }