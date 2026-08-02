"""
institutional_risk_engine.py
============================
Institutional-grade risk engine for an Indian equity algo trading platform.

Replaces the basic backend/engine/risk_manager.py with a full suite of:
  - Configurable RiskConfig dataclass (all thresholds runtime-adjustable)
  - VaR / CVaR / Stressed-VaR / Component-VaR / Liquidity-adjusted VaR
  - Portfolio risk metrics (beta, TE, IR, Sharpe, Sortino, Calmar, Ulcer …)
  - Position sizing: Kelly, vol-targeting, risk-parity, HRP, optimal-f, MVO
  - Stress testing with 10 named Indian-market historical scenarios
  - F&O Greeks aggregation
  - All 11 institutional risk gates
  - Master pre_trade_check() with priority ordering and multiplicative scaling
  - PortfolioRiskMonitor with Redis caching
  - FastAPI-ready Redis config helpers
  - NIFTY_SECTOR_MAP for ~100 NSE large-caps

Author  : D-Strategies Risk Team
Platform: NSE / BSE via Zerodha
Python  : 3.10+
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------- optional scipy -----------------------------------------------
try:
    from scipy import stats as _scipy_stats
    from scipy.cluster import hierarchy as _scipy_hierarchy
    from scipy.spatial.distance import squareform as _squareform

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- optional pandas -----------------------------------------------
try:
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

logger = logging.getLogger(__name__)

# =========================================================================
# Part 0 — Pure-NumPy helpers (used when scipy is absent)
# =========================================================================

def _norm_ppf(p: float) -> float:
    """Rational approximation to the normal quantile (Abramowitz & Stegun)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if _HAS_SCIPY:
        return float(_scipy_stats.norm.ppf(p))
    # Beasley-Springer-Moro approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        x = y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0)
        return x
    r = p if y < 0.0 else 1.0 - p
    r = math.log(-math.log(r))
    x = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] + r*(c[5] + r*(c[6] + r*(c[7] + r*c[8])))))))
    return -x if y < 0.0 else x


def _cholesky_safe(cov: np.ndarray) -> np.ndarray:
    """Cholesky decomposition with jitter for near-singular matrices."""
    jitter = 1e-8
    for _ in range(10):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(len(cov)))
        except np.linalg.LinAlgError:
            jitter *= 10
    raise np.linalg.LinAlgError("Covariance matrix is not positive-definite after jitter.")


# =========================================================================
# Part 1 — RiskConfig
# =========================================================================

@dataclass
class RiskConfig:
    """
    All risk thresholds for the InstitutionalRiskEngine.
    Every field is runtime-adjustable via save_risk_config_to_redis().
    """

    # ---- Loss / drawdown limits ----
    daily_loss_limit_pct: float = 0.02        # 2% NAV daily loss halt
    drawdown_kill_switch_pct: float = 0.10    # 10% peak-to-trough kill switch

    # ---- Beta / correlation limits ----
    max_portfolio_beta: float = 0.80          # vs Nifty50
    max_correlation_to_add: float = 0.70      # reject if incremental corr exceeds

    # ---- Position / concentration limits ----
    max_position_size_pct: float = 0.05       # 5% of NAV per position
    max_sector_concentration_pct: float = 0.30  # 30% in any single sector
    max_gross_exposure_pct: float = 1.50      # 150% of NAV gross leverage

    # ---- VIX / volatility filters ----
    vix_halt_threshold: float = 30.0          # CBOE VIX absolute halt
    india_vix_halt_threshold: float = 25.0    # India VIX absolute halt
    vix_size_reduction_pct: float = 0.50      # reduce position sizes by 50%

    # ---- Order rate limiting ----
    max_orders_per_minute: int = 60

    # ---- NSE circuit breaker levels ----
    circuit_breaker_levels: List[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.10, 0.20]
    )

    # ---- F&O ban handling ----
    fo_ban_action: str = "block"              # "block" or "warn"

    # ---- VaR / CVaR parameters ----
    var_confidence: float = 0.99
    var_lookback_days: int = 252
    cvar_confidence: float = 0.975
    max_portfolio_var_pct: float = 0.02       # 2% 1-day VaR limit

    # ---- Position sizing parameters ----
    kelly_fraction: float = 0.25             # fractional Kelly (25% of full Kelly)
    target_vol: float = 0.12                 # 12% annualised

    # ---- Liquidity limits ----
    min_liquidity_ratio: float = 0.05        # max 5% of 30-day ADV per order
    margin_buffer_pct: float = 0.10          # keep 10% margin buffer

    # ---- Stress testing ----
    stress_loss_limit_pct: float = 0.15      # block if stress scenario > 15% loss


# =========================================================================
# Part 2a — VaR Engine
# =========================================================================

class InsufficientDataError(ValueError):
    """
    Raised when a risk statistic is requested but the sample cannot support it.

    This exists because the alternative — returning 0.0 — is a capital-loss bug,
    not a convenience. A VaR of 0.0 reads as "this book has no risk", is published
    to the risk dashboard as green, and is divided into an equity budget by
    CVaR-based position sizing. Missing data must HALT, never fabricate.
    """


# A quantile at confidence c is not supported by data until the sample has at
# least 1/(1-c) points; below that np.percentile is pure interpolation off the
# extreme observation. Floor of 20 for any statistic at all.
_ABSOLUTE_MIN_OBS = int(os.getenv("RISK_MIN_OBSERVATIONS", "20"))


def _min_obs_for(confidence: float) -> int:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    return max(_ABSOLUTE_MIN_OBS, int(math.ceil(1.0 / (1.0 - confidence))))


def _require_sample(returns: np.ndarray, confidence: float, what: str) -> np.ndarray:
    """Validate a return sample or raise. Never returns a fabricated value."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    need = _min_obs_for(confidence)
    if arr.size < need:
        raise InsufficientDataError(
            f"{what}: need >= {need} finite observations at confidence "
            f"{confidence:.4g}, got {arr.size} — HALT, do not fabricate"
        )
    return arr


class VaREngine:
    """Pure NumPy implementation of multiple VaR / CVaR methods."""

    @staticmethod
    def historical_var(
        returns: np.ndarray,
        confidence: float = 0.99,
        horizon_days: int = 1,
    ) -> float:
        """Historical VaR: percentile(returns, 1-confidence) × √horizon."""
        r = _require_sample(returns, confidence, "historical_var")
        q = np.percentile(r, (1.0 - confidence) * 100)
        return float(-q * math.sqrt(horizon_days))

    @staticmethod
    def parametric_var(
        returns: np.ndarray,
        confidence: float = 0.99,
        horizon_days: int = 1,
    ) -> float:
        """Parametric (Gaussian) VaR: -(μ - z_α × σ) × √horizon."""
        r = _require_sample(returns, confidence, "parametric_var")
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        z = _norm_ppf(confidence)
        return float((z * sigma - mu) * math.sqrt(horizon_days))

    @staticmethod
    def monte_carlo_var(
        returns: np.ndarray,
        n_sims: int = 10_000,
        confidence: float = 0.99,
        horizon_days: int = 10,
    ) -> float:
        """
        MC VaR over multi-asset paths using Cholesky decomposition.
        For single-asset: simulate horizon_days paths, compound, take percentile.
        """
        r = _require_sample(returns, confidence, "monte_carlo_var")
        rng = np.random.default_rng(seed=42)
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        # simulate portfolio returns over horizon
        sim_daily = rng.normal(mu, sigma, size=(n_sims, horizon_days))
        sim_horizon = sim_daily.sum(axis=1)
        q = np.percentile(sim_horizon, (1.0 - confidence) * 100)
        return float(-q)

    @staticmethod
    def historical_cvar(
        returns: np.ndarray,
        confidence: float = 0.975,
    ) -> float:
        """Expected Shortfall (CVaR): mean of returns below VaR threshold."""
        r = _require_sample(returns, confidence, "historical_cvar")
        threshold = np.percentile(r, (1.0 - confidence) * 100)
        tail = r[r <= threshold]
        if len(tail) == 0:
            return float(-threshold)
        return float(-np.mean(tail))

    @staticmethod
    def liquidity_adjusted_var(
        returns: np.ndarray,
        adv_ratio: float,
        confidence: float = 0.99,
    ) -> float:
        """
        LVaR (Almgren-Chriss): VaR × (1 + 0.5×spread + liquidation_cost).
        liquidation_cost = adv_ratio^0.5 × σ.
        """
        base_var = VaREngine.historical_var(returns, confidence)
        sigma = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        spread = 0.001  # assume 10 bps bid-ask for Indian large-caps
        liquidation_cost = math.sqrt(max(adv_ratio, 0.0)) * sigma
        return float(base_var * (1.0 + 0.5 * spread + liquidation_cost))

    @staticmethod
    def component_var(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        confidence: float = 0.99,
    ) -> np.ndarray:
        """
        Component VaR: CVaR_i = w_i × (Σw)_i / σ_p × z_α.
        Returns array of per-position VaR contributions.
        """
        w = np.asarray(weights, dtype=float)
        sigma_p = math.sqrt(float(w @ cov_matrix @ w))
        if sigma_p == 0:
            return np.zeros_like(w)
        z = _norm_ppf(confidence)
        marginal = (cov_matrix @ w) / sigma_p
        return w * marginal * z

    @staticmethod
    def stressed_var(
        returns: np.ndarray,
        stress_period_mask: np.ndarray,
        confidence: float = 0.99,
    ) -> float:
        """Stressed VaR: historical VaR computed only on the stress-period subset."""
        stressed = returns[stress_period_mask.astype(bool)]
        if len(stressed) < 5:
            logger.warning("Stress period too short (%d obs); using full history.", len(stressed))
            return VaREngine.historical_var(returns, confidence)
        return VaREngine.historical_var(stressed, confidence)


# =========================================================================
# Part 2b — Portfolio Risk Engine
# =========================================================================

class PortfolioRiskEngine:
    """Classical portfolio performance and risk metrics — pure NumPy."""

    @staticmethod
    def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Portfolio vol: √(wᵀ Σ w)."""
        w = np.asarray(weights, dtype=float)
        return float(math.sqrt(max(float(w @ cov_matrix @ w), 0.0)))

    @staticmethod
    def portfolio_beta(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Beta: Cov(rp, rb) / Var(rb)."""
        if len(portfolio_returns) < 2:
            return 1.0
        cov = float(np.cov(portfolio_returns, benchmark_returns)[0, 1])
        var_b = float(np.var(benchmark_returns, ddof=1))
        return cov / var_b if var_b != 0 else 1.0

    @staticmethod
    def tracking_error(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Tracking error: std(rp - rb) × √252 (annualised)."""
        active = portfolio_returns - benchmark_returns
        return float(np.std(active, ddof=1) * math.sqrt(252))

    @staticmethod
    def information_ratio(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """IR: mean(rp - rb) × 252 / tracking_error."""
        active = portfolio_returns - benchmark_returns
        te = PortfolioRiskEngine.tracking_error(portfolio_returns, benchmark_returns)
        if te == 0:
            return 0.0
        return float(np.mean(active) * 252 / te)

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.065,
    ) -> float:
        """Sharpe: (mean(r)×252 - rf) / (std(r)×√252)."""
        if len(returns) < 2:
            return 0.0
        annualised_ret = float(np.mean(returns)) * 252
        annualised_vol = float(np.std(returns, ddof=1)) * math.sqrt(252)
        if annualised_vol == 0:
            return 0.0
        return (annualised_ret - risk_free_rate) / annualised_vol

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.065,
        mar: float = 0.0,
    ) -> float:
        """Sortino: (mean(r)×252 - rf) / (downside_std × √252)."""
        if len(returns) < 2:
            return 0.0
        annualised_ret = float(np.mean(returns)) * 252
        downside = returns[returns < mar]
        if len(downside) < 2:
            return float("inf") if annualised_ret > risk_free_rate else 0.0
        downside_std = float(np.std(downside, ddof=1)) * math.sqrt(252)
        if downside_std == 0:
            return 0.0
        return (annualised_ret - risk_free_rate) / downside_std

    @staticmethod
    def calmar_ratio(returns: np.ndarray) -> float:
        """Calmar: CAGR / |max_drawdown|."""
        if len(returns) < 2:
            return 0.0
        equity = np.cumprod(1.0 + returns)
        cagr = float(equity[-1] ** (252 / len(returns)) - 1)
        mdd = PortfolioRiskEngine.max_drawdown(equity)
        if mdd == 0:
            return float("inf") if cagr > 0 else 0.0
        return cagr / abs(mdd)

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """Max drawdown: max of (peak − trough) / peak over rolling windows."""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / np.where(peak == 0, 1, peak)
        return float(np.min(drawdowns))

    @staticmethod
    def ulcer_index(equity_curve: np.ndarray) -> float:
        """Ulcer Index: √(mean of squared percentage drawdowns from peaks)."""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        dd_pct = (equity_curve - peak) / np.where(peak == 0, 1, peak) * 100.0
        return float(math.sqrt(np.mean(dd_pct ** 2)))

    @staticmethod
    def pain_index(equity_curve: np.ndarray) -> float:
        """Pain Index: mean of all absolute drawdown depths (not squared)."""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        dd_pct = (peak - equity_curve) / np.where(peak == 0, 1, peak) * 100.0
        return float(np.mean(dd_pct))

    @staticmethod
    def tail_ratio(returns: np.ndarray) -> float:
        """Tail ratio: |P95| / |P5| — measure of asymmetry in tails."""
        if len(returns) < 20:
            return 1.0
        p95 = float(np.percentile(returns, 95))
        p5 = float(np.percentile(returns, 5))
        if p5 == 0:
            return float("inf")
        return abs(p95) / abs(p5)

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Omega: Σmax(r−t,0) / Σmax(t−r,0)."""
        if len(returns) == 0:
            return 1.0
        gains = np.sum(np.maximum(returns - threshold, 0.0))
        losses = np.sum(np.maximum(threshold - returns, 0.0))
        if losses == 0:
            return float("inf")
        return float(gains / losses)


# =========================================================================
# Part 2c — Position Sizer
# =========================================================================

class PositionSizer:
    """Institutional position sizing: Kelly, vol-target, risk-parity, HRP, MVO."""

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Full Kelly: f* = (p×b − q) / b where b = avg_win/avg_loss."""
        if avg_loss == 0:
            return 0.0
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1.0 - p
        f = (p * b - q) / b
        return max(0.0, float(f))

    @staticmethod
    def fractional_kelly(kelly_f: float, fraction: float = 0.25) -> float:
        """Fractional Kelly: kelly_f × fraction."""
        return max(0.0, float(kelly_f * fraction))

    @staticmethod
    def volatility_targeting(
        target_vol: float,
        realized_vol: float,
        capital: float,
        price: float,
    ) -> int:
        """Vol-targeting: qty = floor((target_vol / realized_vol) × capital / price)."""
        if realized_vol <= 0 or price <= 0:
            return 0
        qty = math.floor((target_vol / realized_vol) * capital / price)
        return max(0, qty)

    @staticmethod
    def equal_weight(capital: float, n_positions: int, price: float) -> int:
        """Equal weight: floor(capital / n_positions / price)."""
        if n_positions <= 0 or price <= 0:
            return 0
        return max(0, math.floor(capital / n_positions / price))

    @staticmethod
    def risk_parity(
        vols: np.ndarray,
        capital: float,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Risk parity: w_i = (1/σ_i) / Σ(1/σ_j).
        Returns integer quantities for each asset.
        """
        vols = np.asarray(vols, dtype=float)
        prices = np.asarray(prices, dtype=float)
        inv_vols = np.where(vols > 0, 1.0 / vols, 0.0)
        total = inv_vols.sum()
        if total == 0:
            return np.zeros(len(vols), dtype=int)
        weights = inv_vols / total
        raw_qty = weights * capital / np.where(prices > 0, prices, 1.0)
        return np.floor(raw_qty).astype(int)

    @staticmethod
    def hrp_weights(cov_matrix: np.ndarray) -> np.ndarray:
        """
        Hierarchical Risk Parity (Lopez de Prado 2016).
        Steps: correlation → single-linkage clustering → quasi-diag → recursive bisection.
        """
        n = len(cov_matrix)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # Compute correlation matrix
        std = np.sqrt(np.diag(cov_matrix))
        std = np.where(std == 0, 1.0, std)
        corr = cov_matrix / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        # Distance matrix
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

        # Hierarchical clustering (single linkage — manual if scipy absent)
        if _HAS_SCIPY:
            condensed = _squareform(dist)
            linkage = _scipy_hierarchy.linkage(condensed, method="single")
            order = _scipy_hierarchy.leaves_list(linkage).tolist()
        else:
            order = PositionSizer._manual_single_linkage(dist)

        # Recursive bisection
        weights = np.ones(n)
        clusters = [order]
        while clusters:
            clusters_new = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                def _cluster_var(idxs: list) -> float:
                    sub_cov = cov_matrix[np.ix_(idxs, idxs)]
                    sub_w = np.ones(len(idxs)) / len(idxs)
                    return float(sub_w @ sub_cov @ sub_w)

                var_l = _cluster_var(left)
                var_r = _cluster_var(right)
                total_var = var_l + var_r
                alpha = 1.0 - (var_l / total_var) if total_var > 0 else 0.5
                weights[left] *= alpha
                weights[right] *= (1.0 - alpha)
                clusters_new += [left, right]
            clusters = clusters_new

        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        return weights / total if total > 0 else np.ones(n) / n

    @staticmethod
    def _manual_single_linkage(dist: np.ndarray) -> list:
        """Greedy single-linkage ordering when scipy is unavailable."""
        n = len(dist)
        visited = [False] * n
        order = [0]
        visited[0] = True
        for _ in range(n - 1):
            last = order[-1]
            best_j, best_d = -1, float("inf")
            for j in range(n):
                if not visited[j] and dist[last, j] < best_d:
                    best_j, best_d = j, dist[last, j]
            order.append(best_j)
            visited[best_j] = True
        return order

    @staticmethod
    def optimal_f(equity_curve: np.ndarray) -> float:
        """Optimal-f: argmax E[log(1 + f × r)] over discrete grid search."""
        if len(equity_curve) < 2:
            return 0.25
        returns = np.diff(equity_curve) / np.where(equity_curve[:-1] == 0, 1, equity_curve[:-1])
        best_f, best_g = 0.0, -float("inf")
        for f_candidate in np.linspace(0.01, 1.0, 100):
            log_returns = np.log1p(f_candidate * returns)
            if np.any(np.isnan(log_returns)) or np.any(np.isinf(log_returns)):
                continue
            g = float(np.mean(log_returns))
            if g > best_g:
                best_g = g
                best_f = f_candidate
        return float(best_f)

    @staticmethod
    def mean_variance_weights(
        mu: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: float,
    ) -> np.ndarray:
        """
        Markowitz MVO closed-form: w* = Σ⁻¹(μ − λ1) / 1ᵀΣ⁻¹(μ − λ1).
        Finds lambda via bisection to hit target_return; falls back to tangency if fails.
        """
        n = len(mu)
        try:
            cov_inv = np.linalg.inv(cov_matrix + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            return np.ones(n) / n

        ones = np.ones(n)

        # Tangency portfolio first
        tangency_num = cov_inv @ mu
        tangency_den = float(ones @ cov_inv @ mu)
        if tangency_den == 0:
            return np.ones(n) / n
        w_tan = tangency_num / tangency_den

        # Bisect lambda to match target_return
        def _port_ret(lam: float) -> float:
            w = cov_inv @ (mu - lam * ones)
            denom = float(ones @ w)
            if denom == 0:
                return 0.0
            return float(mu @ (w / denom))

        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if _port_ret(mid) > target_return:
                lo = mid
            else:
                hi = mid

        lam_star = (lo + hi) / 2.0
        w_raw = cov_inv @ (mu - lam_star * ones)
        denom = float(ones @ w_raw)
        if abs(denom) < 1e-12:
            return w_tan / (float(ones @ w_tan) or 1.0)
        w = w_raw / denom
        # Long-only constraint: clip and renormalise
        w = np.maximum(w, 0.0)
        total = w.sum()
        return w / total if total > 0 else np.ones(n) / n


# =========================================================================
# Part 2d — Stress Test Engine
# =========================================================================

class StressTestEngine:
    """Historical and hypothetical stress scenarios for Indian equity portfolios."""

    # Historical date windows (approximate trading-day masks applied externally)
    SCENARIO_WINDOWS: Dict[str, Tuple[str, str]] = {
        "covid_crash_2020": ("2020-02-15", "2020-03-31"),
        "demonetization_2016": ("2016-11-01", "2016-12-15"),
        "covid_2nd_wave_2021": ("2021-04-01", "2021-05-31"),
        "russia_ukraine_2022": ("2022-02-21", "2022-03-31"),
        "india_election_2024": ("2024-05-04", "2024-06-04"),
    }

    @staticmethod
    def scenario_pnl(
        weights: np.ndarray,
        scenario_shocks: Dict[str, float],
        symbol_index: Dict[str, int],
    ) -> float:
        """
        Apply symbol-specific return shocks to portfolio weights.
        scenario_shocks: {symbol: shock_return}, symbol_index: {symbol: weight_index}.
        """
        w = np.asarray(weights, dtype=float)
        shocks = np.zeros(len(w))
        for sym, shock in scenario_shocks.items():
            idx = symbol_index.get(sym)
            if idx is not None and idx < len(shocks):
                shocks[idx] = shock
        return float(w @ shocks)

    @staticmethod
    def historical_scenarios(
        returns_df: Any,   # pd.DataFrame indexed by date, columns = symbols
        weights: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute portfolio P&L for each named historical stress window.
        Requires pandas. Returns {scenario_name: pnl_fraction}.
        """
        results: Dict[str, float] = {}
        if not _HAS_PANDAS:
            logger.warning("pandas not available; skipping historical_scenarios.")
            return results

        w = np.asarray(weights, dtype=float)
        if len(w) != returns_df.shape[1]:
            logger.error("Weight vector length %d != columns %d", len(w), returns_df.shape[1])
            return results

        for name, (start, end) in StressTestEngine.SCENARIO_WINDOWS.items():
            try:
                window = returns_df.loc[start:end]
                if len(window) == 0:
                    continue
                port_returns = (window.values * w).sum(axis=1)
                cum_pnl = float(np.prod(1.0 + port_returns) - 1.0)
                results[name] = cum_pnl
            except Exception as exc:
                logger.warning("Scenario %s failed: %s", name, exc)

        # Hypothetical scenarios using the most recent volatility
        try:
            recent_vol = float(np.std(
                (returns_df.values[-60:] * w).sum(axis=1), ddof=1
            )) if len(returns_df) >= 60 else 0.01

            results["nifty_crash_5pct"] = float(w.sum() * -0.05)
            results["nifty_crash_10pct"] = float(w.sum() * -0.10)
            results["vol_spike_3x"] = float(-3.0 * recent_vol * math.sqrt(5))
            results["liquidity_halt"] = float(w.sum() * -0.02)
            results["flash_crash"] = float(w.sum() * (-0.20 + 0.15))
        except Exception as exc:
            logger.warning("Hypothetical scenarios failed: %s", exc)

        return results

    @staticmethod
    def tail_risk_analysis(returns: np.ndarray) -> dict:
        """
        Comprehensive tail-risk statistics.
        Returns: skewness, excess_kurtosis, p99_loss, ES at multiple levels,
                 max_consecutive_losses, drawdown_distribution.
        """
        if len(returns) < 10:
            return {}

        n = len(returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))

        # Skewness and excess kurtosis
        if sigma > 0:
            z = (returns - mu) / sigma
            skewness = float(np.mean(z ** 3))
            excess_kurtosis = float(np.mean(z ** 4)) - 3.0
        else:
            skewness, excess_kurtosis = 0.0, 0.0

        # P99 loss
        p99_loss = float(-np.percentile(returns, 1))

        # Expected Shortfall at multiple confidence levels
        es_dict = {}
        for conf in [0.95, 0.975, 0.99]:
            threshold = np.percentile(returns, (1 - conf) * 100)
            tail = returns[returns <= threshold]
            es_dict[f"ES_{int(conf*1000)}"] = float(-np.mean(tail)) if len(tail) > 0 else p99_loss

        # Max consecutive losses
        max_consec = 0
        current = 0
        for r in returns:
            if r < 0:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0

        # Drawdown distribution
        equity = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.where(peak == 0, 1, peak)
        dd_negative = dd[dd < 0]
        dd_dist: dict = {}
        if len(dd_negative) > 0:
            for pct in [25, 50, 75, 90, 95, 99]:
                dd_dist[f"p{pct}"] = float(np.percentile(dd_negative, pct))

        return {
            "skewness": skewness,
            "excess_kurtosis": excess_kurtosis,
            "p99_loss": p99_loss,
            "expected_shortfall": es_dict,
            "max_consecutive_losses": max_consec,
            "drawdown_distribution": dd_dist,
            "n_observations": n,
        }


# =========================================================================
# Part 2e — Greeks Engine (F&O)
# =========================================================================

class GreeksEngine:
    """Aggregate F&O Greeks across a portfolio of option/futures positions."""

    @staticmethod
    def portfolio_delta(positions: List[dict]) -> float:
        """Portfolio delta: Σ delta_i × qty_i."""
        return sum(float(p.get("delta", 0.0)) * float(p.get("qty", 0)) for p in positions)

    @staticmethod
    def portfolio_gamma(positions: List[dict]) -> float:
        """Portfolio gamma: Σ gamma_i × qty_i."""
        return sum(float(p.get("gamma", 0.0)) * float(p.get("qty", 0)) for p in positions)

    @staticmethod
    def portfolio_vega(positions: List[dict]) -> float:
        """Portfolio vega: Σ vega_i × qty_i."""
        return sum(float(p.get("vega", 0.0)) * float(p.get("qty", 0)) for p in positions)

    @staticmethod
    def portfolio_theta(positions: List[dict]) -> float:
        """Portfolio theta: Σ theta_i × qty_i."""
        return sum(float(p.get("theta", 0.0)) * float(p.get("qty", 0)) for p in positions)

    @staticmethod
    def net_delta_exposure(
        positions: List[dict],
        spot_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Net delta exposure per underlying in INR.
        Each position needs: {underlying, delta, qty, lot_size}.
        """
        exposure: Dict[str, float] = {}
        for p in positions:
            und = p.get("underlying", p.get("symbol", "UNKNOWN"))
            delta = float(p.get("delta", 0.0))
            qty = float(p.get("qty", 0))
            lot_size = float(p.get("lot_size", 1))
            spot = float(spot_prices.get(und, 0.0))
            contrib = delta * qty * lot_size * spot
            exposure[und] = exposure.get(und, 0.0) + contrib
        return exposure

    @staticmethod
    def delta_hedge_qty(
        net_delta: float,
        spot_price: float,
        lot_size: int = 50,
    ) -> int:
        """
        Nifty delta hedge: -round(net_delta / (spot_price × lot_size)).
        Returns number of futures lots needed to flatten delta.
        """
        if spot_price <= 0 or lot_size <= 0:
            return 0
        lots = net_delta / (spot_price * lot_size)
        return -int(round(lots))


# =========================================================================
# Part 3 — The 11 Risk Gates
# =========================================================================

@dataclass
class GateResult:
    """Result of a single risk gate evaluation."""
    gate: str
    passed: bool
    value: float
    threshold: float
    action: str          # "allow" | "block" | "scale" | "warn"
    message: str
    scale_factor: float = 1.0   # only meaningful when action == "scale"


# ---------- NSE Sector Map -------------------------------------------------
NIFTY_SECTOR_MAP: Dict[str, str] = {
    # Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "HINDPETRO": "Energy", "GAIL": "Energy",
    "PETRONET": "Energy", "POWERGRID": "Energy", "NTPC": "Energy",
    "ADANIGREEN": "Energy", "ADANIPORTS": "Industrials",
    # Financials
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
    "KOTAKBANK": "Financials", "AXISBANK": "Financials", "INDUSINDBK": "Financials",
    "BANDHANBNK": "Financials", "FEDERALBNK": "Financials", "IDFCFIRSTB": "Financials",
    "PNB": "Financials", "CANBK": "Financials", "UNIONBANK": "Financials",
    "HDFCLIFE": "Financials", "SBILIFE": "Financials", "ICICIPRULI": "Financials",
    "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "MUTHOOTFIN": "Financials",
    "CHOLAFIN": "Financials", "MANAPPURAM": "Financials",
    # Information Technology
    "TCS": "Information Technology", "INFY": "Information Technology",
    "WIPRO": "Information Technology", "HCLTECH": "Information Technology",
    "TECHM": "Information Technology", "LTIM": "Information Technology",
    "MPHASIS": "Information Technology", "COFORGE": "Information Technology",
    "PERSISTENT": "Information Technology", "OFSS": "Information Technology",
    # Consumer Discretionary
    "MARUTI": "Consumer Discretionary", "M&M": "Consumer Discretionary",
    "TATAMOTORS": "Consumer Discretionary", "BAJAJ-AUTO": "Consumer Discretionary",
    "HEROMOTOCO": "Consumer Discretionary", "EICHERMOT": "Consumer Discretionary",
    "TVSMOTOR": "Consumer Discretionary", "TITAN": "Consumer Discretionary",
    "CROMPTON": "Consumer Discretionary", "VOLTAS": "Consumer Discretionary",
    # Consumer Staples
    "HINDUNILVR": "Consumer Staples", "ITC": "Consumer Staples",
    "NESTLEIND": "Consumer Staples", "BRITANNIA": "Consumer Staples",
    "DABUR": "Consumer Staples", "GODREJCP": "Consumer Staples",
    "MARICO": "Consumer Staples", "COLPAL": "Consumer Staples",
    "TATACONSUM": "Consumer Staples", "VARUNBEV": "Consumer Staples",
    # Healthcare
    "SUNPHARMA": "Healthcare", "DRREDDY": "Healthcare", "CIPLA": "Healthcare",
    "DIVISLAB": "Healthcare", "BIOCON": "Healthcare", "LUPIN": "Healthcare",
    "AUROPHARMA": "Healthcare", "TORNTPHARM": "Healthcare", "ALKEM": "Healthcare",
    "ABBOTINDIA": "Healthcare",
    # Industrials
    "LT": "Industrials", "SIEMENS": "Industrials", "ABB": "Industrials",
    "BHEL": "Industrials", "HAL": "Industrials", "BEL": "Industrials",
    "CUMMINSIND": "Industrials", "THERMAX": "Industrials", "TIINDIA": "Industrials",
    # Materials
    "TATASTEEL": "Materials", "JSWSTEEL": "Materials", "HINDALCO": "Materials",
    "VEDL": "Materials", "COALINDIA": "Materials", "NMDC": "Materials",
    "SAIL": "Materials", "NATIONALUM": "Materials", "ULTRACEMCO": "Materials",
    "SHREECEM": "Materials", "AMBUJACEM": "Materials", "ACC": "Materials",
    "GRASIM": "Materials", "PIDILITIND": "Materials", "ASIANPAINT": "Materials",
    # Telecom
    "BHARTIARTL": "Communication Services", "IDEA": "Communication Services",
    "TATACOMM": "Communication Services",
    # Real Estate
    "DLF": "Real Estate", "GODREJPROP": "Real Estate", "OBEROIRLTY": "Real Estate",
    "PHOENIXLTD": "Real Estate", "PRESTIGE": "Real Estate",
    # Utilities
    "TATAPOWER": "Utilities", "TORNTPOWER": "Utilities", "CESC": "Utilities",
    "JSW Energy": "Utilities",
    # Index / ETF / Misc
    "NIFTYBEES": "Index", "JUNIORBEES": "Index",
}


class InstitutionalRiskEngine:
    """
    Central pre-trade risk engine implementing all 11 institutional risk gates.

    Usage:
        engine = InstitutionalRiskEngine(config=RiskConfig(), redis_client=r)
        approved, gates, modified_order = engine.pre_trade_check(order, portfolio_state, market_state)
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        redis_client: Any = None,
    ):
        self.config = config or RiskConfig()
        self.redis = redis_client
        logger.info("InstitutionalRiskEngine initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Gate 1: Daily Loss Limit
    # ------------------------------------------------------------------
    def check_daily_loss_limit(
        self,
        daily_pnl: float,
        portfolio_nav: float,
    ) -> GateResult:
        """Gate 1 — Block all trading if daily P&L loss exceeds 2% of NAV."""
        if portfolio_nav <= 0:
            return GateResult("daily_loss_limit", True, 0.0, 0.0, "allow", "NAV unavailable", 1.0)
        loss_pct = daily_pnl / portfolio_nav   # negative when losing
        threshold = -self.config.daily_loss_limit_pct
        breached = loss_pct < threshold
        if breached:
            self._set_redis_flag("risk:daily_trading_halted", "1")
            return GateResult(
                gate="daily_loss_limit",
                passed=False,
                value=loss_pct,
                threshold=threshold,
                action="block",
                message=f"Daily loss {loss_pct:.2%} exceeds limit {threshold:.2%}. Trading halted for the day.",
                scale_factor=0.0,
            )
        return GateResult(
            gate="daily_loss_limit",
            passed=True,
            value=loss_pct,
            threshold=threshold,
            action="allow",
            message=f"Daily P&L {loss_pct:.2%} within limit.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 2: Drawdown Kill Switch
    # ------------------------------------------------------------------
    def check_drawdown_kill_switch(
        self,
        current_equity: float,
        peak_equity: float,
    ) -> GateResult:
        """Gate 2 — Halt + cancel orders if drawdown from peak exceeds 10%."""
        if peak_equity <= 0:
            return GateResult("drawdown_kill_switch", True, 0.0, 0.0, "allow", "Peak equity unavailable", 1.0)
        dd = (peak_equity - current_equity) / peak_equity
        threshold = self.config.drawdown_kill_switch_pct
        breached = dd > threshold
        if breached:
            self._set_redis_flag("risk:kill_switch_active", "1")
            return GateResult(
                gate="drawdown_kill_switch",
                passed=False,
                value=dd,
                threshold=threshold,
                action="block",
                message=f"Kill switch: drawdown {dd:.2%} exceeds {threshold:.2%}. All orders blocked.",
                scale_factor=0.0,
            )
        return GateResult(
            gate="drawdown_kill_switch",
            passed=True,
            value=dd,
            threshold=threshold,
            action="allow",
            message=f"Drawdown {dd:.2%} within kill-switch threshold.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 3: Portfolio Beta / Correlation
    # ------------------------------------------------------------------
    def check_correlation_limit(
        self,
        order_symbol: str,
        portfolio_returns: np.ndarray,
        symbol_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        order_weight: float = 0.01,
    ) -> GateResult:
        """Gate 3 — Block if adding the order would push portfolio beta above 0.80."""
        if len(portfolio_returns) < 20:
            return GateResult("correlation_limit", True, 0.0, self.config.max_portfolio_beta, "allow",
                              "Insufficient history for beta check.", 1.0)
        # Incremental beta: blend portfolio and new symbol
        blend = (1.0 - order_weight) * portfolio_returns + order_weight * symbol_returns
        new_beta = PortfolioRiskEngine.portfolio_beta(blend, benchmark_returns)
        threshold = self.config.max_portfolio_beta
        if new_beta > threshold:
            return GateResult(
                gate="correlation_limit",
                passed=False,
                value=new_beta,
                threshold=threshold,
                action="block",
                message=f"Adding {order_symbol} would push portfolio beta to {new_beta:.3f} > {threshold:.2f}.",
                scale_factor=0.0,
            )
        return GateResult(
            gate="correlation_limit",
            passed=True,
            value=new_beta,
            threshold=threshold,
            action="allow",
            message=f"Post-addition beta {new_beta:.3f} within limit.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 4: Position Size
    # ------------------------------------------------------------------
    def check_position_size(
        self,
        order_qty: float,
        order_price: float,
        portfolio_nav: float,
        existing_qty: float = 0.0,
    ) -> GateResult:
        """Gate 4 — Scale order so (existing + new) × price / NAV ≤ 5%."""
        if portfolio_nav <= 0 or order_price <= 0:
            return GateResult("position_size", True, 0.0, 0.0, "allow", "NAV or price unavailable.", 1.0)
        total_qty = existing_qty + order_qty
        total_pct = total_qty * order_price / portfolio_nav
        threshold = self.config.max_position_size_pct
        if total_pct > threshold:
            # Scale: how much of the new order can we add within the limit?
            allowed_total_qty = threshold * portfolio_nav / order_price
            addable_qty = max(0.0, allowed_total_qty - existing_qty)
            scale = addable_qty / order_qty if order_qty > 0 else 0.0
            return GateResult(
                gate="position_size",
                passed=False,
                value=total_pct,
                threshold=threshold,
                action="scale",
                message=(
                    f"Position {total_pct:.2%} exceeds limit {threshold:.2%}. "
                    f"Scaling order to {scale:.2%} of requested qty."
                ),
                scale_factor=min(1.0, max(0.0, scale)),
            )
        return GateResult(
            gate="position_size",
            passed=True,
            value=total_pct,
            threshold=threshold,
            action="allow",
            message=f"Position size {total_pct:.2%} within {threshold:.2%} limit.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 5: VIX Filter
    # ------------------------------------------------------------------
    def check_vix_filter(
        self,
        india_vix: float,
        cboe_vix: float = 20.0,
    ) -> GateResult:
        """Gate 5 — Reduce position sizes by 50% when India VIX > 25 or CBOE VIX > 30."""
        india_breach = india_vix > self.config.india_vix_halt_threshold
        cboe_breach = cboe_vix > self.config.vix_halt_threshold
        breached = india_breach or cboe_breach
        max(india_vix / self.config.india_vix_halt_threshold,
                        cboe_vix / self.config.vix_halt_threshold)
        scale = 1.0 - self.config.vix_size_reduction_pct if breached else 1.0
        threshold = max(self.config.india_vix_halt_threshold, self.config.vix_halt_threshold)
        if breached:
            return GateResult(
                gate="vix_filter",
                passed=False,
                value=max(india_vix, cboe_vix),
                threshold=threshold,
                action="scale",
                message=(
                    f"High volatility: India VIX={india_vix:.1f}, CBOE VIX={cboe_vix:.1f}. "
                    f"Reducing position sizes by {self.config.vix_size_reduction_pct:.0%}."
                ),
                scale_factor=scale,
            )
        return GateResult(
            gate="vix_filter",
            passed=True,
            value=max(india_vix, cboe_vix),
            threshold=threshold,
            action="allow",
            message=f"VIX within normal range (India={india_vix:.1f}, CBOE={cboe_vix:.1f}).",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 6: Sector Concentration
    # ------------------------------------------------------------------
    def check_sector_concentration(
        self,
        order_symbol: str,
        sector: str,
        portfolio_sector_weights: Dict[str, float],
        order_weight: float = 0.01,
    ) -> GateResult:
        """Gate 6 — Block if adding the order would push any sector > 30% of NAV."""
        # Use sector map if sector not explicitly provided
        if not sector:
            sector = NIFTY_SECTOR_MAP.get(order_symbol.upper(), "Unknown")
        current_weight = portfolio_sector_weights.get(sector, 0.0)
        new_weight = current_weight + order_weight
        threshold = self.config.max_sector_concentration_pct
        if new_weight > threshold:
            return GateResult(
                gate="sector_concentration",
                passed=False,
                value=new_weight,
                threshold=threshold,
                action="block",
                message=(
                    f"Sector '{sector}' concentration {new_weight:.2%} would exceed "
                    f"{threshold:.2%} limit after adding {order_symbol}."
                ),
                scale_factor=0.0,
            )
        return GateResult(
            gate="sector_concentration",
            passed=True,
            value=new_weight,
            threshold=threshold,
            action="allow",
            message=f"Sector '{sector}' concentration {new_weight:.2%} within {threshold:.2%} limit.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 7: Order Rate Limit
    # ------------------------------------------------------------------
    def check_order_rate(self, strategy_name: str) -> GateResult:
        """Gate 7 — Block if strategy submits > 60 orders/min (Redis sliding window)."""
        threshold = self.config.max_orders_per_minute
        now = time.time()
        key = f"risk:order_rate:{strategy_name}"
        try:
            if self.redis is not None:
                pipe = self.redis.pipeline()
                pipe.zremrangebyscore(key, 0, now - 60)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, 120)
                results = pipe.execute()
                count = int(results[2])
            else:
                count = 0   # no Redis — pass gate
        except Exception as exc:
            logger.warning("Gate 7 Redis error: %s", exc)
            count = 0

        if count > threshold:
            return GateResult(
                gate="order_rate",
                passed=False,
                value=float(count),
                threshold=float(threshold),
                action="block",
                message=(
                    f"Strategy '{strategy_name}' submitted {count} orders in the last 60s "
                    f"(limit {threshold}). Order queued."
                ),
                scale_factor=0.0,
            )
        return GateResult(
            gate="order_rate",
            passed=True,
            value=float(count),
            threshold=float(threshold),
            action="allow",
            message=f"Order rate {count}/min within {threshold}/min limit.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 8: Margin Safety / Gross Exposure
    # ------------------------------------------------------------------
    def check_margin_safety(
        self,
        gross_exposure: float,
        portfolio_nav: float,
    ) -> GateResult:
        """Gate 8 — Block if gross exposure / NAV > 150% (leveraged)."""
        if portfolio_nav <= 0:
            return GateResult("margin_safety", True, 0.0, 0.0, "allow", "NAV unavailable.", 1.0)
        ratio = gross_exposure / portfolio_nav
        threshold = self.config.max_gross_exposure_pct
        if ratio > threshold:
            return GateResult(
                gate="margin_safety",
                passed=False,
                value=ratio,
                threshold=threshold,
                action="block",
                message=(
                    f"Gross exposure {ratio:.2%} exceeds max leverage {threshold:.2%}. "
                    "Leveraged order blocked."
                ),
                scale_factor=0.0,
            )
        return GateResult(
            gate="margin_safety",
            passed=True,
            value=ratio,
            threshold=threshold,
            action="allow",
            message=f"Gross exposure {ratio:.2%} within {threshold:.2%} limit.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 9: NSE Circuit Breaker
    # ------------------------------------------------------------------
    def check_circuit_breaker(
        self,
        symbol: str,
        current_move_pct: float,
    ) -> GateResult:
        """Gate 9 — Block all orders for symbol if price move hits NSE circuit levels."""
        levels = sorted(self.config.circuit_breaker_levels)
        abs_move = abs(current_move_pct)
        triggered_level: Optional[float] = None
        for lvl in levels:
            if abs_move >= lvl:
                triggered_level = lvl

        if triggered_level is not None:
            self._send_circuit_breaker_alert(symbol, current_move_pct, triggered_level)
            return GateResult(
                gate="circuit_breaker",
                passed=False,
                value=abs_move,
                threshold=triggered_level,
                action="block",
                message=(
                    f"NSE circuit breaker triggered for {symbol}: "
                    f"move={current_move_pct:.2%}, level={triggered_level:.0%}."
                ),
                scale_factor=0.0,
            )
        return GateResult(
            gate="circuit_breaker",
            passed=True,
            value=abs_move,
            threshold=levels[0] if levels else 0.02,
            action="allow",
            message=f"{symbol} move {current_move_pct:.2%} below circuit breaker levels.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 10: F&O Ban List
    # ------------------------------------------------------------------
    def check_fo_ban(
        self,
        symbol: str,
        fo_ban_list: List[str],
    ) -> GateResult:
        """Gate 10 — Block new F&O positions in SEBI-banned symbols."""
        banned_upper = [s.upper() for s in fo_ban_list]
        is_banned = symbol.upper() in banned_upper
        if is_banned:
            action = self.config.fo_ban_action  # "block" or "warn"
            return GateResult(
                gate="fo_ban",
                passed=False,
                value=1.0,
                threshold=0.0,
                action=action,
                message=f"{symbol} is on SEBI F&O ban list. Action: {action}.",
                scale_factor=0.0 if action == "block" else 1.0,
            )
        return GateResult(
            gate="fo_ban",
            passed=True,
            value=0.0,
            threshold=0.0,
            action="allow",
            message=f"{symbol} not on F&O ban list.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Gate 11: Liquidity
    # ------------------------------------------------------------------
    def check_liquidity(
        self,
        order_qty: float,
        symbol: str,
        adv_30d: float,
    ) -> GateResult:
        """Gate 11 — Scale order to max 5% of 30-day ADV to avoid market impact."""
        if adv_30d <= 0:
            return GateResult("liquidity", True, 0.0, 0.0, "allow", "ADV not available.", 1.0)
        adv_ratio = order_qty / adv_30d
        threshold = self.config.min_liquidity_ratio
        if adv_ratio > threshold:
            scale = threshold / adv_ratio
            return GateResult(
                gate="liquidity",
                passed=False,
                value=adv_ratio,
                threshold=threshold,
                action="scale",
                message=(
                    f"{symbol}: order qty {order_qty:.0f} = {adv_ratio:.2%} of 30d ADV {adv_30d:.0f}. "
                    f"Scaling to {threshold:.0%} of ADV."
                ),
                scale_factor=min(1.0, scale),
            )
        return GateResult(
            gate="liquidity",
            passed=True,
            value=adv_ratio,
            threshold=threshold,
            action="allow",
            message=f"{symbol} order is {adv_ratio:.2%} of ADV — within {threshold:.0%} limit.",
            scale_factor=1.0,
        )

    # ------------------------------------------------------------------
    # Master Pre-Trade Check
    # ------------------------------------------------------------------
    def pre_trade_check(
        self,
        order: dict,
        portfolio_state: dict,
        market_state: dict,
    ) -> Tuple[bool, List[GateResult], dict]:
        """
        Run all 11 risk gates in priority order.

        Parameters
        ----------
        order : {symbol, side, qty, price, order_type, strategy, sector, fo_type, adv_30d}
        portfolio_state : {nav, peak_equity, daily_pnl, positions, sector_weights,
                           gross_exposure, existing_qty}
        market_state : {india_vix, cboe_vix, fo_ban_list, circuit_breakers,
                        benchmark_returns, portfolio_returns, symbol_returns}

        Returns
        -------
        (approved: bool, gate_results: List[GateResult], modified_order: dict)
        """
        results: List[GateResult] = []
        modified_order = dict(order)
        qty = float(order.get("qty", 0))
        cumulative_scale = 1.0

        nav = float(portfolio_state.get("nav", 1.0))
        peak_equity = float(portfolio_state.get("peak_equity", nav))
        daily_pnl = float(portfolio_state.get("daily_pnl", 0.0))
        gross_exposure = float(portfolio_state.get("gross_exposure", 0.0))
        sector_weights: Dict[str, float] = portfolio_state.get("sector_weights", {})
        existing_qty = float(portfolio_state.get("existing_qty", 0.0))

        symbol: str = str(order.get("symbol", ""))
        price = float(order.get("price", 0.0))
        strategy: str = str(order.get("strategy", "default"))
        sector: str = str(order.get("sector", NIFTY_SECTOR_MAP.get(symbol.upper(), "")))
        fo_ban_list: List[str] = market_state.get("fo_ban_list", [])
        india_vix = float(market_state.get("india_vix", 15.0))
        cboe_vix = float(market_state.get("cboe_vix", 20.0))
        current_move_pct = float(market_state.get("circuit_breakers", {}).get(symbol, 0.0))
        adv_30d = float(order.get("adv_30d", 0.0))

        portfolio_returns: np.ndarray = np.asarray(
            market_state.get("portfolio_returns", []), dtype=float
        )
        symbol_returns: np.ndarray = np.asarray(
            market_state.get("symbol_returns", []), dtype=float
        )
        benchmark_returns: np.ndarray = np.asarray(
            market_state.get("benchmark_returns", []), dtype=float
        )

        # --- Check Redis kill-switch / daily halt flags first (fast path) ---
        if self._get_redis_flag("risk:kill_switch_active") == "1":
            gr = GateResult(
                gate="kill_switch_redis",
                passed=False,
                value=1.0,
                threshold=0.0,
                action="block",
                message="Kill switch active (set in Redis). All orders blocked.",
                scale_factor=0.0,
            )
            return False, [gr], modified_order

        if self._get_redis_flag("risk:daily_trading_halted") == "1":
            gr = GateResult(
                gate="daily_halt_redis",
                passed=False,
                value=1.0,
                threshold=0.0,
                action="block",
                message="Daily trading halted (set in Redis). All orders blocked.",
                scale_factor=0.0,
            )
            return False, [gr], modified_order

        # ---- GATE 2: Kill Switch (live check) ----
        g2 = self.check_drawdown_kill_switch(nav, peak_equity)
        results.append(g2)
        if not g2.passed:
            return False, results, modified_order

        # ---- GATE 1: Daily Loss ----
        g1 = self.check_daily_loss_limit(daily_pnl, nav)
        results.append(g1)
        if not g1.passed:
            return False, results, modified_order

        # ---- GATE 9: Circuit Breaker ----
        g9 = self.check_circuit_breaker(symbol, current_move_pct)
        results.append(g9)
        if not g9.passed:
            return False, results, modified_order

        # ---- GATE 10: F&O Ban ----
        g10 = self.check_fo_ban(symbol, fo_ban_list)
        results.append(g10)
        if not g10.passed and g10.action == "block":
            return False, results, modified_order

        # ---- GATE 8: Margin Safety ----
        order_notional = qty * price
        g8 = self.check_margin_safety(gross_exposure + order_notional, nav)
        results.append(g8)
        if not g8.passed:
            return False, results, modified_order

        # ---- GATE 3: Correlation / Beta ----
        g3 = self.check_correlation_limit(
            symbol, portfolio_returns, symbol_returns, benchmark_returns,
            order_weight=min(order_notional / max(nav, 1), 0.10),
        )
        results.append(g3)
        if not g3.passed:
            return False, results, modified_order

        # ---- GATE 6: Sector Concentration ----
        g6 = self.check_sector_concentration(
            symbol, sector, sector_weights,
            order_weight=order_notional / max(nav, 1),
        )
        results.append(g6)
        if not g6.passed:
            return False, results, modified_order

        # ---- GATE 7: Order Rate ----
        g7 = self.check_order_rate(strategy)
        results.append(g7)
        if not g7.passed:
            return False, results, modified_order

        # ---- GATE 5: VIX Filter (scale) ----
        g5 = self.check_vix_filter(india_vix, cboe_vix)
        results.append(g5)
        cumulative_scale *= g5.scale_factor

        # ---- GATE 4: Position Size (scale) ----
        scaled_qty = qty * cumulative_scale
        g4 = self.check_position_size(scaled_qty, price, nav, existing_qty)
        results.append(g4)
        cumulative_scale *= g4.scale_factor

        # ---- GATE 11: Liquidity (scale) ----
        scaled_qty = qty * cumulative_scale
        g11 = self.check_liquidity(scaled_qty, symbol, adv_30d)
        results.append(g11)
        cumulative_scale *= g11.scale_factor

        # ---- Apply final scale to order ----
        final_qty = math.floor(qty * cumulative_scale)
        modified_order["qty"] = final_qty
        modified_order["_scale_factor"] = cumulative_scale
        modified_order["_original_qty"] = qty

        approved = final_qty > 0
        if not approved:
            logger.warning(
                "Order for %s scaled to zero (original_qty=%s, scale=%.4f).",
                symbol, qty, cumulative_scale,
            )

        return approved, results, modified_order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_redis_flag(self, key: str, value: str, ttl_seconds: int = 86400) -> None:
        """Set a Redis flag with TTL; silently swallow errors."""
        try:
            if self.redis is not None:
                self.redis.setex(key, ttl_seconds, value)
        except Exception as exc:
            logger.warning("Redis set failed for key %s: %s", key, exc)

    def _get_redis_flag(self, key: str) -> Optional[str]:
        """Get a Redis string value; returns None on error or absence."""
        try:
            if self.redis is not None:
                val = self.redis.get(key)
                if val is None:
                    return None
                return val.decode() if isinstance(val, bytes) else str(val)
        except Exception as exc:
            logger.warning("Redis get failed for key %s: %s", key, exc)
        return None

    def _send_circuit_breaker_alert(
        self,
        symbol: str,
        move_pct: float,
        level: float,
    ) -> None:
        """Publish circuit-breaker alert to Redis pub/sub channel."""
        try:
            if self.redis is not None:
                payload = json.dumps({
                    "event": "circuit_breaker",
                    "symbol": symbol,
                    "move_pct": move_pct,
                    "level": level,
                    "timestamp": time.time(),
                })
                self.redis.publish("risk:alerts", payload)
                logger.warning("CIRCUIT BREAKER: %s move=%.2f%% level=%.0f%%", symbol, move_pct*100, level*100)
        except Exception as exc:
            logger.warning("Failed to send circuit breaker alert: %s", exc)


# =========================================================================
# Part 4 — Real-time Portfolio Risk Monitor
# =========================================================================

@dataclass
class RiskSnapshot:
    """Point-in-time portfolio risk metrics cached in Redis."""
    # None == "not computable from available data". Never 0.0: a zero VaR
    # passes every limit check and renders green on the dashboard.
    var_99: Optional[float] = None
    cvar_975: Optional[float] = None
    portfolio_vol: float = 0.0
    portfolio_beta: float = 1.0
    max_drawdown: float = 0.0
    sector_concentrations: Dict[str, float] = field(default_factory=dict)
    largest_position_pct: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    stress_results: Dict[str, float] = field(default_factory=dict)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    ulcer_index: float = 0.0
    timestamp: float = field(default_factory=time.time)


class PortfolioRiskMonitor:
    """Continuously updated risk metrics with Redis caching."""

    REDIS_KEY = "risk:snapshot"

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        redis_client: Any = None,
    ):
        self.config = config or RiskConfig()
        self.redis = redis_client
        self._var_engine = VaREngine()
        self._port_engine = PortfolioRiskEngine()
        self._stress_engine = StressTestEngine()

    def compute_snapshot(
        self,
        positions_df: Any,          # pd.DataFrame: columns=[symbol, qty, price, sector, adv]
        returns_df: Any,             # pd.DataFrame: rows=dates, cols=symbols
        benchmark_returns: np.ndarray,
    ) -> RiskSnapshot:
        """Compute a full RiskSnapshot from current positions and historical returns."""
        snap = RiskSnapshot()

        if not _HAS_PANDAS:
            logger.warning("pandas unavailable — returning empty snapshot.")
            return snap

        try:
            # Portfolio weights
            positions_df = positions_df.copy()
            positions_df["notional"] = positions_df["qty"] * positions_df["price"]
            total_long = float(positions_df.loc[positions_df["notional"] > 0, "notional"].sum())
            total_short = float(positions_df.loc[positions_df["notional"] < 0, "notional"].sum())
            snap.gross_exposure = total_long + abs(total_short)
            snap.net_exposure = total_long + total_short
            nav = max(snap.gross_exposure, 1.0)  # approximate
            snap.leverage = snap.gross_exposure / nav

            if total_long > 0:
                snap.largest_position_pct = float(
                    positions_df["notional"].abs().max() / nav
                )

            # Sector concentrations
            if "sector" in positions_df.columns:
                sector_notional = (
                    positions_df.groupby("sector")["notional"].sum().abs()
                )
                snap.sector_concentrations = (sector_notional / nav).to_dict()

            # Returns-based metrics
            if len(returns_df) >= 20 and returns_df.shape[1] > 0:
                symbols = returns_df.columns.tolist()
                pos_dict = positions_df.set_index("symbol")["notional"].to_dict()
                weights = np.array([pos_dict.get(s, 0.0) for s in symbols])
                w_sum = float(np.sum(np.abs(weights)))
                weights_norm = weights / w_sum if w_sum > 0 else np.ones(len(symbols)) / len(symbols)

                port_returns = (returns_df.values * weights_norm).sum(axis=1)

                try:
                    snap.var_99 = VaREngine.historical_var(port_returns, confidence=0.99)
                except InsufficientDataError as exc:
                    snap.var_99 = None
                    logger.warning("var_99 unavailable: %s", exc)
                try:
                    snap.cvar_975 = VaREngine.historical_cvar(port_returns, confidence=0.975)
                except InsufficientDataError as exc:
                    snap.cvar_975 = None
                    logger.warning("cvar_975 unavailable: %s", exc)

                if len(port_returns) >= 2:
                    snap.portfolio_vol = float(
                        np.std(port_returns, ddof=1) * math.sqrt(252)
                    )

                if len(benchmark_returns) >= len(port_returns):
                    bench_aligned = benchmark_returns[-len(port_returns):]
                    snap.portfolio_beta = PortfolioRiskEngine.portfolio_beta(port_returns, bench_aligned)

                equity_curve = np.cumprod(1.0 + port_returns)
                snap.max_drawdown = PortfolioRiskEngine.max_drawdown(equity_curve)
                snap.sharpe_ratio = PortfolioRiskEngine.sharpe_ratio(port_returns)
                snap.sortino_ratio = PortfolioRiskEngine.sortino_ratio(port_returns)
                snap.ulcer_index = PortfolioRiskEngine.ulcer_index(equity_curve)

                # Stress results
                snap.stress_results = self._stress_engine.historical_scenarios(
                    returns_df, weights_norm
                )

        except Exception as exc:
            logger.exception("Error computing risk snapshot: %s", exc)

        snap.timestamp = time.time()
        return snap

    def update_redis(self, snapshot: RiskSnapshot) -> None:
        """Serialise RiskSnapshot to Redis hash `risk:snapshot`."""
        try:
            if self.redis is None:
                return
            flat: Dict[str, str] = {
                # "unavailable" (not "0.0") so a consumer cannot read a missing
                # metric as a benign one.
                "var_99": "unavailable" if snapshot.var_99 is None else str(snapshot.var_99),
                "cvar_975": "unavailable" if snapshot.cvar_975 is None else str(snapshot.cvar_975),
                "portfolio_vol": str(snapshot.portfolio_vol),
                "portfolio_beta": str(snapshot.portfolio_beta),
                "max_drawdown": str(snapshot.max_drawdown),
                "largest_position_pct": str(snapshot.largest_position_pct),
                "gross_exposure": str(snapshot.gross_exposure),
                "net_exposure": str(snapshot.net_exposure),
                "leverage": str(snapshot.leverage),
                "sharpe_ratio": str(snapshot.sharpe_ratio),
                "sortino_ratio": str(snapshot.sortino_ratio),
                "ulcer_index": str(snapshot.ulcer_index),
                "timestamp": str(snapshot.timestamp),
                "sector_concentrations": json.dumps(snapshot.sector_concentrations),
                "stress_results": json.dumps(snapshot.stress_results),
            }
            self.redis.hset(self.REDIS_KEY, mapping=flat)
            self.redis.expire(self.REDIS_KEY, 300)  # 5-minute TTL
        except Exception as exc:
            logger.warning("Failed to update Redis risk snapshot: %s", exc)

    def get_snapshot_from_redis(self) -> Optional[RiskSnapshot]:
        """Deserialise RiskSnapshot from Redis hash `risk:snapshot`."""
        try:
            if self.redis is None:
                return None
            raw = self.redis.hgetall(self.REDIS_KEY)
            if not raw:
                return None

            def _f(key: bytes | str) -> str:
                k = key.decode() if isinstance(key, bytes) else key
                v = raw.get(key) or raw.get(k.encode()) or raw.get(k) or b"0"
                return v.decode() if isinstance(v, bytes) else str(v)

            snap = RiskSnapshot(
                var_99=float(_f("var_99")),
                cvar_975=float(_f("cvar_975")),
                portfolio_vol=float(_f("portfolio_vol")),
                portfolio_beta=float(_f("portfolio_beta")),
                max_drawdown=float(_f("max_drawdown")),
                largest_position_pct=float(_f("largest_position_pct")),
                gross_exposure=float(_f("gross_exposure")),
                net_exposure=float(_f("net_exposure")),
                leverage=float(_f("leverage")),
                sharpe_ratio=float(_f("sharpe_ratio")),
                sortino_ratio=float(_f("sortino_ratio")),
                ulcer_index=float(_f("ulcer_index")),
                timestamp=float(_f("timestamp")),
                sector_concentrations=json.loads(_f("sector_concentrations") or "{}"),
                stress_results=json.loads(_f("stress_results") or "{}"),
            )
            return snap
        except Exception as exc:
            logger.warning("Failed to read Redis risk snapshot: %s", exc)
            return None

    def should_trigger_alert(self, snapshot: RiskSnapshot) -> List[str]:
        """Return list of breach descriptions for any threshold violation."""
        alerts: List[str] = []
        cfg = self.config

        # Fail-closed: an uncomputable VaR is an alert, not a silent pass.
        if snapshot.var_99 is None:
            alerts.append(
                "VaR unavailable: cannot verify the 1-day 99% VaR limit "
                f"({cfg.max_portfolio_var_pct:.2%}) — treat as breached until data is restored"
            )
        elif snapshot.var_99 > cfg.max_portfolio_var_pct:
            alerts.append(
                f"VaR breach: 1-day 99% VaR = {snapshot.var_99:.2%} > limit {cfg.max_portfolio_var_pct:.2%}"
            )
        if abs(snapshot.max_drawdown) > cfg.drawdown_kill_switch_pct:
            alerts.append(
                f"Drawdown breach: {snapshot.max_drawdown:.2%} > kill-switch level {cfg.drawdown_kill_switch_pct:.2%}"
            )
        if abs(snapshot.portfolio_beta) > cfg.max_portfolio_beta:
            alerts.append(
                f"Beta breach: portfolio beta = {snapshot.portfolio_beta:.3f} > {cfg.max_portfolio_beta:.2f}"
            )
        if snapshot.largest_position_pct > cfg.max_position_size_pct:
            alerts.append(
                f"Position concentration: largest position = {snapshot.largest_position_pct:.2%} > {cfg.max_position_size_pct:.2%}"
            )
        if snapshot.leverage > cfg.max_gross_exposure_pct:
            alerts.append(
                f"Leverage breach: {snapshot.leverage:.2%} > {cfg.max_gross_exposure_pct:.2%}"
            )
        for sector, wgt in snapshot.sector_concentrations.items():
            if wgt > cfg.max_sector_concentration_pct:
                alerts.append(
                    f"Sector breach: {sector} = {wgt:.2%} > {cfg.max_sector_concentration_pct:.2%}"
                )
        worst_stress = min(snapshot.stress_results.values(), default=0.0)
        if worst_stress < -cfg.stress_loss_limit_pct:
            alerts.append(
                f"Stress scenario breach: worst loss = {worst_stress:.2%} < -{cfg.stress_loss_limit_pct:.2%}"
            )

        return alerts


# =========================================================================
# Part 5 — Risk API helpers (for FastAPI router)
# =========================================================================

_RISK_CONFIG_REDIS_KEY = "risk:config"


def get_risk_config_from_redis(r: Any) -> RiskConfig:
    """Load RiskConfig from Redis hash; fall back to defaults if absent or error."""
    try:
        raw = r.hgetall(_RISK_CONFIG_REDIS_KEY)
        if not raw:
            return RiskConfig()
        cfg_dict: dict = {}
        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            val_str = v.decode() if isinstance(v, bytes) else str(v)
            # Parse lists vs scalars
            if key == "circuit_breaker_levels":
                try:
                    cfg_dict[key] = json.loads(val_str)
                except Exception:
                    cfg_dict[key] = [0.02, 0.05, 0.10, 0.20]
            elif key in ("fo_ban_action",):
                cfg_dict[key] = val_str
            elif key == "max_orders_per_minute":
                cfg_dict[key] = int(float(val_str))
            elif key == "var_lookback_days":
                cfg_dict[key] = int(float(val_str))
            else:
                try:
                    cfg_dict[key] = float(val_str)
                except ValueError:
                    cfg_dict[key] = val_str
        return RiskConfig(**{k: v for k, v in cfg_dict.items() if hasattr(RiskConfig, k) or k in RiskConfig.__dataclass_fields__})
    except Exception as exc:
        logger.warning("Failed to load RiskConfig from Redis: %s. Using defaults.", exc)
        return RiskConfig()


def save_risk_config_to_redis(config: RiskConfig, r: Any) -> None:
    """Persist RiskConfig to Redis hash."""
    try:
        flat: Dict[str, str] = {}
        for k, v in asdict(config).items():
            if isinstance(v, list):
                flat[k] = json.dumps(v)
            else:
                flat[k] = str(v)
        r.hset(_RISK_CONFIG_REDIS_KEY, mapping=flat)
        logger.info("RiskConfig saved to Redis.")
    except Exception as exc:
        logger.warning("Failed to save RiskConfig to Redis: %s", exc)


def update_risk_param(param: str, value: Any, r: Any) -> bool:
    """
    Update a single RiskConfig parameter in Redis at runtime.
    Returns True on success.
    """
    try:
        # Validate the param exists in RiskConfig
        fields = RiskConfig.__dataclass_fields__
        if param not in fields:
            logger.error("Unknown RiskConfig param: %s", param)
            return False
        val_str = json.dumps(value) if isinstance(value, list) else str(value)
        r.hset(_RISK_CONFIG_REDIS_KEY, param, val_str)
        logger.info("Updated risk param %s = %s", param, val_str)
        return True
    except Exception as exc:
        logger.warning("Failed to update risk param %s: %s", param, exc)
        return False
