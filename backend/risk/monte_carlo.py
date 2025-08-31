# backend/risk/monte_carlo.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple, Literal

ModelKind = Literal["gbm", "t"]  # gbm = geometric Brownian (lognormal), t = studentized returns

@dataclass
class MCConfig:
    horizon_days: int = 10          # risk horizon (calendar days)
    steps_per_day: int = 1          # time-steps per day (>=1)
    n_paths: int = 50_000           # simulation paths
    model: ModelKind = "gbm"        # "gbm" | "t"
    df_t: int = 6                   # degrees of freedom (for model="t")
    use_antithetic: bool = True     # variance reduction
    jump_intensity: float = 0.0     # lambda per year (Poisson); 0 disables jumps
    jump_mu: float = 0.0            # average jump size (log-return)
    jump_sigma: float = 0.1         # jump size stdev (log-return)
    seed: Optional[int] = 42        # RNG seed for reproducibility
    annualization: float = 252.0    # trading days/year

@dataclass
class MCMarket:
    """Per-asset market params. Arrays aligned by the same index order as 'assets'."""
    assets: Iterable[str]
    spot: np.ndarray         # current prices
    mu_annual: np.ndarray    # annualized drift (per asset)
    vol_annual: np.ndarray   # annualized vol (per asset)
    corr: np.ndarray         # correlation matrix (nxn)

@dataclass
class Portfolio:
    """Holdings to value PnL in currency units."""
    qty: Dict[str, float]         # {asset -> units} (positive long, negative short)

class MonteCarloEngine:
    """
    Multi-asset Monte Carlo with correlation, optional jumps, and VaR/ES analytics.
    """

    def __init__(self, cfg: MCConfig, mkt: MCMarket, portfolio: Optional[Portfolio] = None):
        self.cfg = cfg
        self.mkt = mkt
        self.assets = list(mkt.assets)
        self.n = len(self.assets)
        self.portfolio = portfolio or Portfolio(qty={a: 0.0 for a in self.assets})
        self._validate()

        self._rng = np.random.default_rng(cfg.seed)
        self._dt = 1.0 / (cfg.annualization * max(1, cfg.steps_per_day))
        self._steps = max(1, cfg.horizon_days * cfg.steps_per_day)

        # Precompute Cholesky for correlation
        self._chol = np.linalg.cholesky(np.clip(mkt.corr, -0.999, 0.999))

        # Per-step drift/vol for GBM in log space
        self._mu_step = (mkt.mu_annual - 0.5 * mkt.vol_annual**2) * self._dt
        self._sigma_step = mkt.vol_annual * np.sqrt(self._dt)

        # For t model, we scale standard normals by sqrt(df / chi2) to get t
        self._t_scale_df = cfg.df_t

        # Portfolio vector aligned to assets
        self._qty_vec = np.array([self.portfolio.qty.get(a, 0.0) for a in self.assets], dtype=float)

    def _validate(self):
        n = len(self.mkt.spot)
        if any(len(x) != n for x in [self.mkt.mu_annual, self.mkt.vol_annual]):
            raise ValueError("spot, mu_annual, vol_annual must be same length")
        if self.mkt.corr.shape != (n, n):
            raise ValueError("corr must be (n x n)")
        if not np.allclose(self.mkt.corr, self.mkt.corr.T, atol=1e-8):
            raise ValueError("corr must be symmetric")
        if np.any(np.linalg.eigvalsh(self.mkt.corr) <= 0):
            raise ValueError("corr must be positive definite (or near PD)")

    # --------------------------------------------------------------------- #
    # Core sampling
    # --------------------------------------------------------------------- #
    def _sample_normals(self, n_paths: int, steps: int) -> np.ndarray:
        """
        Return Z of shape (steps, n_paths, n_assets) with cross-asset correlation applied.
        If use_antithetic=True, paths are paired to reduce variance.
        """
        p = n_paths
        if self.cfg.use_antithetic and (p % 2 == 1):
            p += 1  # make even
        # base iid normals
        Z = self._rng.standard_normal(size=(steps, p, self.n))
        # correlate per step: Z[t] @ chol.T
        Z = Z @ self._chol.T  # broadcast over steps
        if self.cfg.use_antithetic:
            half = p // 2
            Z[: , half: , :] = -Z[: , :half, :]
        if p != n_paths:
            Z = Z[:, :n_paths, :]
        return Z

    def _sample_t(self, n_paths: int, steps: int) -> np.ndarray:
        """
        Student-t innovations with correlation. We draw normals then scale by sqrt(df/chi2).
        """
        df = max(3, int(self.cfg.df_t))
        Z = self._sample_normals(n_paths, steps)  # correlated normals
        # chi-square scaling per (step, path)
        chi2 = self._rng.chisquare(df, size=(steps, n_paths, 1))
        scale = np.sqrt(df / np.maximum(1e-12, chi2))
        return Z * scale  # heavy-tailed

    def _sample_jumps(self, n_paths: int, steps: int) -> np.ndarray:
        """Compound Poisson log-jumps per asset (independent across assets)."""
        lam = float(self.cfg.jump_intensity)
        if lam <= 0:
            return np.zeros((steps, n_paths, self.n), dtype=float)
        # Expected number per step: lam per year → lam*dt
        p = lam * self._dt
        # N~Poisson, jump sizes ~ N(mu, sigma)
        N = self._rng.poisson(p, size=(steps, n_paths, self.n))
        J = self._rng.normal(self.cfg.jump_mu, self.cfg.jump_sigma, size=(steps, n_paths, self.n))
        return N * J

    # --------------------------------------------------------------------- #
    # Paths & valuation
    # --------------------------------------------------------------------- #
    def simulate_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            prices: shape (steps+1, n_paths, n_assets)
            log_returns: shape (steps, n_paths, n_assets)
        """
        steps, p, n = self._steps, self.cfg.n_paths, self.n

        if self.cfg.model == "gbm":
            shocks = self._sample_normals(p, steps)
        elif self.cfg.model == "t":
            shocks = self._sample_t(p, steps)
        else:
            raise ValueError("Unknown model kind")

        jumps = self._sample_jumps(p, steps)  # log-jumps
        # per-step log-return: mu*dt + sigma*Z + jumps
        lr = (self._mu_step.reshape(1, 1, n) +
              self._sigma_step.reshape(1, 1, n) * shocks +
              jumps)
        # accumulate
        prices = np.empty((steps + 1, p, n), dtype=float)
        prices[0] = self.mkt.spot.reshape(1, n)
        prices[1:] = prices[0:1] * np.exp(np.cumsum(lr, axis=0))
        return prices, lr

    def portfolio_terminal_pnl(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute portfolio terminal PnL per path (currency units).
        prices: (steps+1, n_paths, n_assets)
        """
        p0 = prices[0]   # (n_paths, n_assets)
        pT = prices[-1]
        # portfolio value change per path
        dv = (pT - p0) @ self._qty_vec
        return dv  # shape (n_paths,)

    # --------------------------------------------------------------------- #
    # Risk measures
    # --------------------------------------------------------------------- #
    @staticmethod
    def _var_es(sample: np.ndarray, alpha: float = 0.99) -> Tuple[float, float]:
        """
        For a loss distribution X (positive = loss), report VaR_alpha, ES_alpha.
        """
        x = np.sort(sample)
        idx = int(np.floor(alpha * len(x))) - 1
        idx = max(0, min(len(x) - 1, idx))
        var = x[idx]
        es = x[: idx + 1].mean() if idx >= 0 else x.mean()
        return float(var), float(es)

    def distribution(self, prices: Optional[np.ndarray] = None, alpha: float = 0.99) -> Dict[str, float]:
        """
        Returns dict with VaR/ES for portfolio and per-asset contributions (linear approx).
        """
        if prices is None:
            prices, _ = self.simulate_paths()
        pnl_paths = self.portfolio_terminal_pnl(prices)  # currency PnL per path
        loss = -pnl_paths  # loss = -PnL

        var, es = self._var_es(loss, alpha=alpha)
        out = {
            "alpha": alpha,
            "VaR": var,
            "ES": es,
            "mean_pnl": float(pnl_paths.mean()),
            "std_pnl": float(pnl_paths.std(ddof=1)),
            "max_drawdown_est": float(self._max_drawdown_from_paths(prices)),
        }
        return out

    def _max_drawdown_from_paths(self, prices: np.ndarray) -> float:
        """
        Estimate portfolio max drawdown using pathwise equity curves (mean across paths).
        Lightweight summary for dashboards; for exact tail, inspect full distribution.
        """
        # portfolio equity per path over time
        eq = (prices * self._qty_vec.reshape(1, 1, self.n)).sum(axis=2)  # (steps+1, n_paths)
        eq = eq - eq[0:1]  # start at 0
        # per-path drawdown
        peak = np.maximum.accumulate(eq, axis=0)
        dd = peak - eq
        return np.mean(dd.max(axis=0))

    # --------------------------------------------------------------------- #
    # Sensitivities (bump-and-revalue)
    # --------------------------------------------------------------------- #
    def bump_greeks(self, bump_pct: float = 0.01) -> Dict[str, float]:
        """
        One-step Delta-like sensitivities via bump-and-revalue on spot.
        Returns dict {asset: dPnL/dS * S (approx currency delta contribution)}.
        """
        base_prices, _ = self.simulate_paths()
        base_pnl = self.portfolio_terminal_pnl(base_prices).mean()

        out: Dict[str, float] = {}
        for i, a in enumerate(self.assets):
            bumped = self.mkt.spot.copy()
            bumped[i] *= (1.0 + bump_pct)
            alt_mkt = MCMarket(
                assets=self.assets,
                spot=bumped,
                mu_annual=self.mkt.mu_annual,
                vol_annual=self.mkt.vol_annual,
                corr=self.mkt.corr,
            )
            eng = MonteCarloEngine(self.cfg, alt_mkt, self.portfolio)
            p, _ = eng.simulate_paths()
            pnl = eng.portfolio_terminal_pnl(p).mean()
            out[a] = float((pnl - base_pnl) / (bump_pct + 1e-12))
        return out

# ------------------------------------------------------------------------- #
# Utilities
# ------------------------------------------------------------------------- #
def estimate_market_from_history(prices: pd.DataFrame, annualization: float = 252.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a price history (rows=time, cols=assets), estimate (mu_annual, vol_annual, corr).
    """
    rets = np.log(prices / prices.shift(1)).dropna() # type: ignore
    mu_daily = rets.mean().values
    vol_daily = rets.std(ddof=1).values
    mu_annual = mu_daily * annualization
    vol_annual = vol_daily * np.sqrt(annualization)
    corr = np.corrcoef(rets.values.T)
    return mu_annual, vol_annual, corr

def make_mc_from_frames(
    spot: pd.Series,
    mu_annual: pd.Series | np.ndarray,
    vol_annual: pd.Series | np.ndarray,
    corr: np.ndarray,
    cfg: Optional[MCConfig] = None,
    qty: Optional[pd.Series] = None,
) -> MonteCarloEngine:
    assets = list(spot.index)
    mkt = MCMarket(
        assets=assets,
        spot=spot.values.astype(float),
        mu_annual=np.asarray(mu_annual, dtype=float),
        vol_annual=np.asarray(vol_annual, dtype=float),
        corr=corr.astype(float),
    )
    port = Portfolio(qty={a: float((qty or pd.Series(0.0, index=assets)).get(a, 0.0)) for a in assets})
    return MonteCarloEngine(cfg or MCConfig(), mkt, port)

# ------------------------------------------------------------------------- #
# CLI quick runner
# ------------------------------------------------------------------------- #
def _cli():
    import argparse, json, sys
    ap = argparse.ArgumentParser("monte_carlo")
    ap.add_argument("--csv", help="CSV with columns=assets, rows=prices. If absent, uses toy data.")
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--steps-per-day", type=int, default=1)
    ap.add_argument("--paths", type=int, default=50000)
    ap.add_argument("--model", choices=["gbm","t"], default="gbm")
    ap.add_argument("--df", type=int, default=6)
    ap.add_argument("--jumps", type=float, default=0.0, help="jump intensity λ/year")
    ap.add_argument("--jump-mu", type=float, default=0.0)
    ap.add_argument("--jump-sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        spot = df.tail(1).T.squeeze()
        mu, vol, corr = estimate_market_from_history(df)
        qty = pd.Series(1.0, index=spot.index)  # type: ignore # equal 1 unit per asset unless replaced later
    else:
        # toy 3-asset setup
        idx = ["AAPL","MSFT","SPY"]
        spot = pd.Series([190.0, 410.0, 520.0], index=idx, dtype=float)
        mu = np.array([0.08, 0.07, 0.06])
        vol = np.array([0.35, 0.30, 0.20])
        corr = np.array([[1.0,0.7,0.6],[0.7,1.0,0.65],[0.6,0.65,1.0]])
        qty = pd.Series([100, 80, -50], index=idx, dtype=float)

    cfg = MCConfig(
        horizon_days=args.horizon,
        steps_per_day=args.steps_per_day,
        n_paths=args.paths,
        model=args.model,
        df_t=args.df,
        jump_intensity=args.jumps,
        jump_mu=args.jump_mu,
        jump_sigma=args.jump_sigma,
        seed=args.seed,
    )
    eng = make_mc_from_frames(spot, mu, vol, corr, cfg=cfg, qty=qty) # type: ignore

    prices, _ = eng.simulate_paths()
    summ = eng.distribution(prices, alpha=args.alpha)
    delta = eng.bump_greeks(bump_pct=0.01)

    out = {
        "config": asdict(cfg),
        "assets": eng.assets,
        "summary": summ,
        "delta_approx": delta,
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    _cli()