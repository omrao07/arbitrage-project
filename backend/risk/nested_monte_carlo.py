# backend/risk/nested_monte_carlo.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------
# Small utils (no external deps)
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

def _quantile(xs: List[float], q: float) -> float:
    """Linear interpolation quantile; xs needn't be sorted (we sort locally)."""
    n = len(xs)
    if n == 0:
        return 0.0
    ys = sorted(xs)
    if q <= 0: return ys[0]
    if q >= 1: return ys[-1]
    pos = q * (n - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return ys[lo]
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _cholesky_spd(cov: List[List[float]]) -> List[List[float]]:
    """Basic Cholesky for symmetric positive-definite matrices."""
    n = len(cov)
    L: List[List[float]] = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = cov[i][i] - s
                if v <= 1e-12:
                    v = 1e-12   # tiny jitter
                L[i][j] = math.sqrt(v)
            else:
                denom = L[j][j] if abs(L[j][j]) > 1e-12 else 1e-12
                L[i][j] = (cov[i][j] - s) / denom
    return L

def _mvn_correlated_normals(rng: random.Random, L: List[List[float]]) -> List[float]:
    """Generate correlated standard normals using lower-triangular L (Cholesky of Corr)."""
    n = len(L)
    z = [rng.gauss(0.0, 1.0) for _ in range(n)]
    y = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(i+1):
            s += L[i][j] * z[j]
        y[i] = s
    return y


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

@dataclass
class Portfolio:
    """
    Linear portfolio in **notional** per asset: value = Σ (qty[i] * price[i]).
    """
    symbols: List[str]
    qty: List[float]                 # same length as symbols

    def value(self, prices: Dict[str, float]) -> float:
        v = 0.0
        for s, q in zip(self.symbols, self.qty):
            v += q * float(prices.get(s, 0.0))
        return v


@dataclass
class Regime:
    """
    Outer scenario (uncertain parameters).
    - prob: selection probability in outer loop.
    - mu_mult: multiply base drift vector.
    - vol_mult: multiply base vol vector.
    - corr_blend: 0..1 to blend base Corr with stress Corr*: Corr_regime = (1-w)*Corr + w*Corr_stress
    - jump_lambda_yr: Poisson jump intensity (per year).
    - jump_mean: mean jump log-return (e.g., -0.04); jump_std: std of jump log-return.
    """
    name: str
    prob: float
    mu_mult: float = 1.0
    vol_mult: float = 1.0
    corr_blend: float = 0.0
    jump_lambda_yr: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0


@dataclass
class MarketSpec:
    """
    Base market specification for inner paths.
    - mu: annualized drift per symbol
    - vol: annualized vol per symbol
    - corr: correlation matrix (len N)
    - corr_stress: alternate correlation used when a regime blends it in
    - s0: starting prices
    """
    symbols: List[str]
    mu: List[float]
    vol: List[float]
    corr: List[List[float]]
    s0: Dict[str, float]
    corr_stress: Optional[List[List[float]]] = None

    def N(self) -> int:
        return len(self.symbols)


@dataclass
class GridRisk:
    """VaR/ES curves across the time grid."""
    times_days: List[float]
    var_alpha: float
    es_alpha: float
    var: List[float]          # cash VaR (positive numbers)
    es: List[float]           # cash ES  (positive numbers)


@dataclass
class NestedResult:
    """Full results for dashboards/loggers."""
    seed: int
    outer_paths: int
    inner_paths: int
    times_days: List[float]
    by_regime: Dict[str, Dict[str, float]]                  # regime-level totals (expected PnL, tail loss, weight)
    by_regime_grid: Dict[str, GridRisk]                     # regime-level VaR/ES curves
    aggregate_grid: GridRisk                                # probability-weighted grid across regimes
    samples: Dict[str, float] = field(default_factory=dict) # light stats (mean, stdev, worst, best)


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class NestedMonteCarlo:
    """
    Nested Monte Carlo:
      OUTER: sample regime/parameters
      INNER: simulate correlated GBM + optional Poisson jumps across a time grid
      OUTPUT: per-regime and aggregated VaR/ES curves for portfolio cash PnL, plus stats.

    All time units are **years** internally (grid is provided in days).
    """

    def __init__(self,
                 market: MarketSpec,
                 portfolio: Portfolio,
                 *,
                 regimes: List[Regime],
                 seed: int = 77):
        assert abs(sum(r.prob for r in regimes) - 1.0) < 1e-6, "Regime probabilities must sum to 1"
        self.market = market
        self.portfolio = portfolio
        self.regimes = regimes
        self.seed = int(seed)

    # ---------------- public API ----------------

    def run(self,
            n_outer: int,
            n_inner: int,
            times_days: List[float],
            *,
            var_alpha: float = 0.99,
            es_alpha: float = 0.975,
            custom_payoff: Optional[Callable[[Dict[str, float]], float]] = None,
            ) -> NestedResult:
        """
        Run nested MC.
        - times_days: evaluation grid in days (e.g., [1, 5, 10, 20])
        - var/es computed on **cash PnL** vs t=0 portfolio value
        - custom_payoff: if provided, overrides portfolio.value(prices) to compute cash at each time
        """
        rng_outer = random.Random(self.seed)
        base_prices = dict(self.market.s0)
        base_val0 = self._evaluate_portfolio(base_prices, custom_payoff) # type: ignore

        # Prepare grids and accumulators
        times_years = [max(1e-9, t/252.0) for t in times_days]   # trading days ≈ 252/yr
        agg_var_grid = [0.0] * len(times_days)
        agg_es_grid  = [0.0] * len(times_days)
        weight_sum   = 0.0

        by_regime: Dict[str, Dict[str, float]] = {}
        by_regime_grid: Dict[str, GridRisk] = {}

        # Precompute base and stress corr Cholesky
        Corr_base = self.market.corr
        Corr_stress = self.market.corr_stress or self.market.corr
        # Safety: ensure diagonals = 1
        Corr_base = _unit_diag(Corr_base)
        Corr_stress = _unit_diag(Corr_stress)
        L_base = _cholesky_spd(Corr_base)
        L_stress = _cholesky_spd(Corr_stress)

        # OUTER loop
        for o in range(n_outer):
            reg = self._sample_regime(rng_outer)
            # Build per-regime parameters
            mu, vol = self._scaled_mu_vol(reg)
            L_regime = _blend_cholesky(L_base, L_stress, reg.corr_blend)

            # INNER loop (simulate prices across grid)
            pnl_grid_samples: List[List[float]] = [[] for _ in times_days]
            payoff_fn = (lambda p: float(custom_payoff(p))) if custom_payoff else (lambda p: self.portfolio.value(p))

            # reuse one inner RNG seeded from outer
            rng_inner = random.Random(self.seed + 17 * (o + 1))

            for _ in range(n_inner):
                grid_vals = self._simulate_path_prices(
                    rng_inner, mu, vol, L_regime, times_years,
                    jumps=(reg.jump_lambda_yr, reg.jump_mean, reg.jump_std)
                )
                # convert prices→cash PnL vs base
                for ti, price_vec in enumerate(grid_vals):
                    prices = {sym: price_vec[i] for i, sym in enumerate(self.market.symbols)}
                    v = payoff_fn(prices)
                    pnl = v - base_val0
                    pnl_grid_samples[ti].append(pnl)

            # Regime-level risk curves
            var_curve = []
            es_curve  = []
            for ti in range(len(times_days)):
                xs = pnl_grid_samples[ti]
                # cash VaR/ES are positive losses -> take left tail of PnL
                var_cash = -_quantile(xs, 1 - var_alpha)
                es_cash  = -_tail_mean(xs, 1 - es_alpha)
                var_curve.append(max(0.0, var_cash))
                es_curve.append(max(0.0, es_cash))

            # Simple regime summaries
            flat = [x for row in pnl_grid_samples for x in row]  # flatten across grid (rough overview)
            by_regime[reg.name] = {
                "weight": reg.prob,
                "mean_pnl": _mean(flat) if flat else 0.0,
                "stdev_pnl": _stdev(flat) if len(flat) > 1 else 0.0,
                "q01_pnl": _quantile(flat, 0.01) if flat else 0.0,
                "q99_pnl": _quantile(flat, 0.99) if flat else 0.0,
            }
            by_regime_grid[reg.name] = GridRisk(
                times_days=times_days[:],
                var_alpha=var_alpha, es_alpha=es_alpha,
                var=var_curve, es=es_curve
            )

            # Aggregate probability-weighted curves
            w = reg.prob
            weight_sum += w
            for i in range(len(times_days)):
                agg_var_grid[i] += w * var_curve[i]
                agg_es_grid[i]  += w * es_curve[i]

        # Normalize (probs should sum to 1, but guard anyway)
        if weight_sum > 0:
            agg_var_grid = [x / weight_sum for x in agg_var_grid]
            agg_es_grid  = [x / weight_sum for x in agg_es_grid]

        # Light top-level stats under the mixture (approx: average of regime means / stdevs not exact)
        samples = {
            "baseline_value": base_val0,
            "outer_paths": float(n_outer),
            "inner_paths_per_outer": float(n_inner),
        }

        return NestedResult(
            seed=self.seed, outer_paths=n_outer, inner_paths=n_inner,
            times_days=times_days,
            by_regime=by_regime,
            by_regime_grid=by_regime_grid,
            aggregate_grid=GridRisk(times_days=times_days, var_alpha=var_alpha, es_alpha=es_alpha,
                                    var=agg_var_grid, es=agg_es_grid),
            samples=samples
        )

    # ---------------- internals ----------------

    def _sample_regime(self, rng: random.Random) -> Regime:
        u = rng.random()
        acc = 0.0
        for r in self.regimes:
            acc += r.prob
            if u <= acc:
                return r
        return self.regimes[-1]

    def _scaled_mu_vol(self, reg: Regime) -> Tuple[List[float], List[float]]:
        mu = [m * reg.mu_mult for m in self.market.mu]
        vol = [max(1e-9, v * reg.vol_mult) for v in self.market.vol]
        return (mu, vol)

    def _simulate_path_prices(
        self,
        rng: random.Random,
        mu_ann: List[float],
        vol_ann: List[float],
        L_corr: List[List[float]],
        times_years: List[float],
        *,
        jumps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[List[float]]:
        """
        One correlated GBM+J path, sampled at grid times.
        Returns: list of price vectors per grid time [[S1_t1..SN_t1], ..., [..tK]]
        """
        N = self.market.N()
        S = [float(self.market.s0[self.market.symbols[i]]) for i in range(N)]
        out: List[List[float]] = []

        last_t = 0.0
        lam, j_mu, j_sd = jumps

        for T in times_years:
            dt = max(1e-9, T - last_t)
            # step in chunks of, say, daily granularity proportional to dt to model jumps reasonably
            steps = max(1, int(math.ceil(dt * 252)))
            h = dt / steps
            for _ in range(steps):
                # correlated normals
                z = _mvn_from_chol(rng, L_corr)
                for i in range(N):
                    mu = mu_ann[i]
                    sig = vol_ann[i]
                    # GBM: d ln S = (mu - 0.5 sigma^2) dt + sigma sqrt(dt) z
                    dlogS = (mu - 0.5 * sig * sig) * h + sig * math.sqrt(h) * z[i]
                    # Jumps (compound Poisson with normal log jump size)
                    if lam > 0.0:
                        # expected number in h years:
                        nJ = rng.random()
                        if nJ < lam * h:  # at most one jump per tiny step (Poisson thinning)
                            dlogS += rng.gauss(j_mu, j_sd)
                    S[i] = max(1e-9, S[i] * math.exp(dlogS))
            out.append(S[:])
            last_t = T
        return out


# ---------------------------------------------------------------------
# Helpers specific to the engine
# ---------------------------------------------------------------------

def _unit_diag(C: List[List[float]]) -> List[List[float]]:
    """Coerce diagonal to 1, symmetrize minimal (defensive coding for user inputs)."""
    n = len(C)
    out = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i][j] = 1.0
            else:
                # keep as-is but clamp to [-1,1]
                out[i][j] = _clip(C[i][j], -0.9999, 0.9999)
    return out

def _blend_cholesky(L_base: List[List[float]], L_stress: List[List[float]], w: float) -> List[List[float]]:
    """
    Blend base vs stress correlations by w, by reconstructing Corr and re-Cholesky.
    """
    w = _clip(w, 0.0, 1.0)
    # Rebuild Corr from L = chol(Corr)
    def to_corr(L: List[List[float]]) -> List[List[float]]:
        n = len(L); C = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                s = sum(L[i][k]*L[j][k] for k in range(j+1))
                C[i][j] = C[j][i] = s
        # normalize tiny drifts (should already be unit diag)
        for i in range(n): C[i][i] = 1.0
        return C
    Cb = to_corr(L_base)
    Cs = to_corr(L_stress)
    n = len(Cb)
    Cmix = [[(1-w)*Cb[i][j] + w*Cs[i][j] for j in range(n)] for i in range(n)]
    Cmix = _unit_diag(Cmix)
    return _cholesky_spd(Cmix)

def _mvn_from_chol(rng: random.Random, L: List[List[float]]) -> List[float]:
    # standard normals -> correlated via L
    n = len(L)
    z = [rng.gauss(0.0, 1.0) for _ in range(n)]
    y = [0.0]*n
    for i in range(n):
        s = 0.0
        for j in range(i+1):
            s += L[i][j] * z[j]
        y[i] = s
    return y

def _tail_mean(xs: List[float], q_left: float) -> float:
    """
    Mean of the left q_left tail. If q_left=0.025, average the worst 2.5% outcomes.
    """
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = max(1, int(math.floor(q_left * len(ys))))
    return _mean(ys[:k])


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Toy 3-asset portfolio
    symbols = ["AAPL", "MSFT", "SPY"]
    s0 = {"AAPL": 190.0, "MSFT": 420.0, "SPY": 520.0}
    mu  = [0.08, 0.07, 0.06]         # 8%, 7%, 6% annual drifts
    vol = [0.35, 0.30, 0.18]         # annualized vols
    Corr = [
        [1.0, 0.65, 0.55],
        [0.65, 1.0, 0.60],
        [0.55, 0.60, 1.0],
    ]
    Corr_stress = [
        [1.0, 0.85, 0.80],
        [0.85, 1.0, 0.85],
        [0.80, 0.85, 1.0],
    ]
    market = MarketSpec(symbols=symbols, mu=mu, vol=vol, corr=Corr, corr_stress=Corr_stress, s0=s0)
    port = Portfolio(symbols=symbols, qty=[1000, 800, -500])  # net long tech, short index

    regimes = [
        Regime(name="Calm",   prob=0.50, mu_mult=1.0, vol_mult=0.8,  corr_blend=0.0,  jump_lambda_yr=0.1, jump_mean=-0.02, jump_std=0.05),
        Regime(name="Base",   prob=0.35, mu_mult=1.0, vol_mult=1.0,  corr_blend=0.3,  jump_lambda_yr=0.2, jump_mean=-0.03, jump_std=0.07),
        Regime(name="Stress", prob=0.15, mu_mult=0.7, vol_mult=1.6,  corr_blend=0.9,  jump_lambda_yr=1.0, jump_mean=-0.06, jump_std=0.10),
    ]

    engine = NestedMonteCarlo(market, port, regimes=regimes, seed=42)
    res = engine.run(
        n_outer=10,            # number of outer regime draws (can be >>; keep small for demo)
        n_inner=200,           # inner paths per outer
        times_days=[1, 5, 20], # 1d, 1w, 1m grid
        var_alpha=0.99,
        es_alpha=0.975
    )

    # Pretty print a short summary
    print("=== Aggregate VaR/ES (probability-weighted) ===")
    for t, v, e in zip(res.aggregate_grid.times_days, res.aggregate_grid.var, res.aggregate_grid.es):
        print(f"T+{t:>3.0f}d  VaR={v:,.0f}  ES={e:,.0f}  (cash)")

    print("\n=== By Regime (mean±stdev, tails) ===")
    for name, row in res.by_regime.items():
        print(f"{name:8s}  w={row['weight']:.2f}  mean={row['mean_pnl']:,.0f}  stdev={row['stdev_pnl']:,.0f}  "
              f"q01={row['q01_pnl']:,.0f}  q99={row['q99_pnl']:,.0f}")