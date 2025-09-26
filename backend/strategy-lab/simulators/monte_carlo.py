# simulators/monte_carlo.py
"""
MonteCarlo Simulator (stdlib-only)
---------------------------------
Features
- Multi-asset daily return or price-path simulation
- Correlated Gaussian shocks via Cholesky (no numpy)
- Jump diffusion (compound Poisson with lognormal-ish jump size)
- Optional mean-reversion (on log-price drift)
- Portfolio aggregation with weights
- Risk metrics: VaR / ES (Expected Shortfall), drawdowns, Sharpe/Sortino
- Export paths to CSV/JSONL

Integrates well with:
- ResearchAgent (for stress testing factor returns externally)
- Selector/Evaluator (for scenario analysis)
- Backtester (for bootstrapped stress paths)

Usage (quick)
-------------
from simulators.monte_carlo import (
    AssetSpec, MCConfig, MonteCarlo, PortfolioSpec
)

assets = [
    AssetSpec("AAPL", s0=200.0, mu=0.10, sigma=0.30),
    AssetSpec("MSFT", s0=350.0, mu=0.09, sigma=0.25),
]
cfg = MCConfig(n_paths=1_000, horizon_days=252, seed=42, jump_lambda=0.02, jump_mu=-0.02, jump_sigma=0.05)
mc = MonteCarlo(assets, cfg, corr=[ [1.0, 0.6], [0.6, 1.0] ])

report = mc.run(portfolio=PortfolioSpec(weights={"AAPL":0.5,"MSFT":0.5}))
print(report["portfolio"]["risk"])   # VaR/ES etc.

# Export one asset's paths
mc.export_paths_csv("AAPL", "./logs/aapl_mc.csv")
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --------------------------- Config & Models ----------------------------------

@dataclass
class AssetSpec:
    symbol: str
    s0: float                    # starting price (or index level)
    mu: float = 0.08             # annualized drift
    sigma: float = 0.20          # annualized volatility
    mr_kappa: float = 0.0        # mean-reversion strength on log-price (0 disables)
    mr_level_mult: float = 1.0   # long-run level as multiple of s0 (for MR)

@dataclass
class PortfolioSpec:
    weights: Dict[str, float]    # symbol -> weight (sum may be != 1; we normalize)

@dataclass
class MCConfig:
    n_paths: int = 1000
    horizon_days: int = 252
    seed: Optional[int] = 7
    rf_rate: float = 0.0         # annual risk-free (for Sharpe/Sortino)
    # Jumps (compound Poisson on daily step; jump applied in return space)
    jump_lambda: float = 0.0     # expected jumps per day (e.g., 0.02 -> ~1 every 50d)
    jump_mu: float = 0.0         # mean jump (additive in log-return), negative for crash bias
    jump_sigma: float = 0.0      # std of jump log-return
    # Export controls (optional)
    export_dir: Optional[str] = None
    export_precision: int = 6

# --------------------------- Monte Carlo Engine --------------------------------

class MonteCarlo:
    def __init__(self, assets: List[AssetSpec], cfg: MCConfig, corr: Optional[List[List[float]]] = None):
        if not assets:
            raise ValueError("Need at least one AssetSpec")
        self.assets = assets
        self.cfg = cfg
        self.symbols = [a.symbol for a in assets]
        self.rng = random.Random(cfg.seed)
        self.n = len(assets)
        self.dt_years = 1.0 / 252.0

        # correlation matrix (n x n)
        self.corr = corr or self._eye(self.n)
        self._chol = self._cholesky(self.corr)  # raises if not PD

        # storage: symbol -> List[path], each path is List[prices]
        self._paths: Dict[str, List[List[float]]] = {}

    # ---------------------- Public API ----------------------

    def run(self, portfolio: Optional[PortfolioSpec] = None) -> Dict:
        """
        Simulate all asset paths; compute per-asset stats and (optional) portfolio stats.
        Returns a JSON-serializable report with metrics and (optionally) paths summary.
        """
        # Simulate
        for i, asset in enumerate(self.assets):
            self._paths[asset.symbol] = [[] for _ in range(self.cfg.n_paths)]

        for p in range(self.cfg.n_paths):
            # generate correlated normals for horizon_days
            shocks = self._correlated_shocks(self.cfg.horizon_days)
            # evolve prices path-wise
            for i, asset in enumerate(self.assets):
                series = self._simulate_one(asset, [row[i] for row in shocks])
                self._paths[asset.symbol][p] = series

        # Per-asset metrics
        asset_reports = {}
        for a in self.assets:
            prices = self._paths[a.symbol]
            asset_reports[a.symbol] = self._report_asset(a, prices)

        # Portfolio metrics (if requested)
        port_report = None
        if portfolio:
            port_report = self._report_portfolio(portfolio)

        return {
            "config": self.cfg.__dict__,
            "assets": asset_reports,
            "portfolio": port_report,
            "symbols": self.symbols,
        }

    def export_paths_csv(self, symbol: str, path: str) -> None:
        """Dump all simulated price paths for a symbol to CSV: columns t, path_0, path_1, ..."""
        prices = self._paths.get(symbol)
        if not prices:
            raise ValueError("No paths found; run() first or invalid symbol.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # transpose: time-major
        T = len(prices[0])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["t"] + [f"path_{i}" for i in range(len(prices))]
            w.writerow(header)
            for t in range(T):
                row = [t] + [round(prices[p][t], self.cfg.export_precision) for p in range(len(prices))]
                w.writerow(row)

    def export_paths_jsonl(self, symbol: str, path: str) -> None:
        """Dump paths as JSONL with {"t": i, "values": [...]} per line."""
        prices = self._paths.get(symbol)
        if not prices:
            raise ValueError("No paths found; run() first or invalid symbol.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        T = len(prices[0])
        with open(path, "a", encoding="utf-8") as f:
            for t in range(T):
                rec = {"t": t, "values": [round(prices[p][t], self.cfg.export_precision) for p in range(len(prices))]}
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ---------------------- Internals: simulation ----------------------

    def _simulate_one(self, a: AssetSpec, epsilons: List[float]) -> List[float]:
        """
        Simulate one price path for a single asset given daily shocks (standard normals).
        Returns price series of length horizon_days+1 (includes s0).
        Model:
          log S_{t+1} = log S_t + (mu - 0.5*sigma^2)*dt + kappa*(log(theta)-log S_t)*dt + sigma*sqrt(dt)*Z + J
          where J ~ jumps (compound Poisson; if jump occurs, add N(jump_mu, jump_sigma^2))
          theta = mr_level_mult * s0
        """
        s = a.s0
        out = [s]
        theta = max(1e-9, a.mr_level_mult * a.s0)
        mu = a.mu
        sigma = a.sigma
        kappa = max(0.0, a.mr_kappa)

        for t in range(self.cfg.horizon_days):
            z = epsilons[t]
            # base log-return
            drift = (mu - 0.5 * sigma * sigma) * self.dt_years
            mr = kappa * (math.log(theta) - math.log(max(1e-9, s))) * self.dt_years if kappa > 0 else 0.0
            shock = sigma * math.sqrt(self.dt_years) * z
            j = self._jump_draw()  # additive to log-return
            lr = drift + mr + shock + j
            s = max(1e-9, s * math.exp(lr))
            out.append(s)
        return out

    def _jump_draw(self) -> float:
        """Poisson jump with prob=1-exp(-lambda); jump size ~ Normal(jump_mu, jump_sigma)."""
        lam = max(0.0, self.cfg.jump_lambda)
        if lam <= 0.0:
            return 0.0
        # daily probability p ~ lam (for small lam); use exact: 1 - e^{-lam}
        p = 1.0 - math.exp(-lam)
        if self.rng.random() < p:
            # jump size in log-return space
            return self.rng.gauss(self.cfg.jump_mu, self.cfg.jump_sigma)
        return 0.0

    def _correlated_shocks(self, T: int) -> List[List[float]]:
        """
        Generate T vectors of N correlated standard normals with Cholesky.
        Returns list of length T; each item is list length N.
        """
        out = []
        for _ in range(T):
            z = [self._std_normal() for _ in range(self.n)]  # iid N(0,1)
            # y = L * z
            y = [0.0] * self.n
            for i in range(self.n):
                acc = 0.0
                Li = self._chol[i]
                for k in range(i + 1):  # lower-tri
                    acc += Li[k] * z[k]
                y[i] = acc
            out.append(y)
        return out

    def _std_normal(self) -> float:
        # Box-Muller
        u1 = max(1e-12, self.rng.random())
        u2 = self.rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        return r * math.cos(theta)

    # ---------------------- Internals: reports & risk ----------------------

    def _report_asset(self, a: AssetSpec, paths: List[List[float]]) -> Dict:
        rets = []      # list of path total returns (S_T / S_0 - 1)
        dd_list = []   # max drawdown per path
        sr_list = []   # Sharpe per path using daily returns
        so_list = []   # Sortino per path

        for series in paths:
            if len(series) < 2:
                continue
            # daily returns
            dr = [(series[t] / series[t-1] - 1.0) for t in range(1, len(series))]
            rets.append(series[-1] / series[0] - 1.0)
            dd_list.append(self._max_drawdown(series)[0])
            sr_list.append(self._sharpe(dr, self.cfg.rf_rate, 252))
            so_list.append(self._sortino(dr, self.cfg.rf_rate, 252))

        var99, es99 = self._var_es(rets, 0.99)
        var95, es95 = self._var_es(rets, 0.95)

        return {
            "spec": a.__dict__,
            "paths": len(paths),
            "horizon_days": self.cfg.horizon_days,
            "return": {
                "mean": statistics.mean(rets) if rets else 0.0,
                "stdev": statistics.pstdev(rets) if len(rets) > 1 else 0.0,
                "p5": self._percentile(rets, 5) if rets else 0.0,
                "p50": self._percentile(rets, 50) if rets else 0.0,
                "p95": self._percentile(rets, 95) if rets else 0.0,
                "var95": var95, "es95": es95, "var99": var99, "es99": es99,
            },
            "risk": {
                "avg_drawdown": statistics.mean(dd_list) if dd_list else 0.0,
                "avg_sharpe": statistics.mean(sr_list) if sr_list else 0.0,
                "avg_sortino": statistics.mean(so_list) if so_list else 0.0,
            }
        }

    def _report_portfolio(self, port: PortfolioSpec) -> Dict:
        # normalize weights
        wsum = sum(abs(v) for v in port.weights.values()) or 1.0
        w = {k: v / wsum for k, v in port.weights.items()}

        # build portfolio price paths as weighted sum of indexed series (rebased to 1.0 at t=0)
        # Note: This is a *notional index* (not rebalanced per day) because each asset path is absolute price;
        # here we rebalance **daily** to maintain weights on index levels.
        # Compute daily portfolio return as sum_i w_i * r_i,t.
        n_paths = self.cfg.n_paths
        T = self.cfg.horizon_days
        port_series = [[1.0] * (T + 1) for _ in range(n_paths)]

        # Build symbol -> path daily returns for efficiency
        sym_dr: Dict[str, List[List[float]]] = {}
        for sym in w.keys():
            aset_paths = self._paths.get(sym) or []
            dr_paths = []
            for series in aset_paths:
                dr_paths.append([(series[t] / series[t-1] - 1.0) for t in range(1, len(series))])
            sym_dr[sym] = dr_paths

        for p in range(n_paths):
            for t in range(T):
                r = 0.0
                for sym, wi in w.items():
                    r += wi * sym_dr[sym][p][t]
                port_series[p][t+1] = port_series[p][t] * (1 + r)

        # metrics
        total_returns = [series[-1] - 1.0 for series in port_series]  # since series[0]=1
        var99, es99 = self._var_es(total_returns, 0.99)
        var95, es95 = self._var_es(total_returns, 0.95)

        # per-path risk
        dd = [self._max_drawdown(series)[0] for series in port_series]
        sharpe = []
        sortino = []
        for series in port_series:
            dr = [(series[t] / series[t-1] - 1.0) for t in range(1, len(series))]
            sharpe.append(self._sharpe(dr, self.cfg.rf_rate, 252))
            sortino.append(self._sortino(dr, self.cfg.rf_rate, 252))

        return {
            "weights": w,
            "risk": {
                "var95": var95, "es95": es95, "var99": var99, "es99": es99,
                "avg_drawdown": statistics.mean(dd) if dd else 0.0,
                "avg_sharpe": statistics.mean(sharpe) if sharpe else 0.0,
                "avg_sortino": statistics.mean(sortino) if sortino else 0.0,
            },
            "distribution": {
                "mean_return": statistics.mean(total_returns) if total_returns else 0.0,
                "stdev_return": statistics.pstdev(total_returns) if len(total_returns) > 1 else 0.0,
                "p5": self._percentile(total_returns, 5) if total_returns else 0.0,
                "p50": self._percentile(total_returns, 50) if total_returns else 0.0,
                "p95": self._percentile(total_returns, 95) if total_returns else 0.0,
            }
        }

    # ---------------------- Math helpers (no numpy) ----------------------

    def _eye(self, n: int) -> List[List[float]]:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def _cholesky(self, A: List[List[float]]) -> List[List[float]]:
        """Return lower-triangular L such that A ≈ L L^T. Raises on non-PD."""
        n = len(A)
        # shallow validation
        for row in A:
            if len(row) != n:
                raise ValueError("Correlation matrix must be square")
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    if val <= 0:
                        raise ValueError("Matrix not positive definite (Cholesky failed)")
                    L[i][j] = math.sqrt(val)
                else:
                    denom = L[j][j]
                    if abs(denom) < 1e-12:
                        raise ValueError("Matrix not positive definite (zero on diagonal)")
                    L[i][j] = (A[i][j] - s) / denom
        return L

    def _max_drawdown(self, series: List[float]) -> Tuple[float, int, int]:
        peak = -float("inf")
        mdd = 0.0
        s = e = 0
        start_i = 0
        for i, v in enumerate(series):
            if v > peak:
                peak = v
                start_i = i
            dd = (v - peak) / peak if peak > 0 else 0.0
            if dd < mdd:
                mdd = dd; s = start_i; e = i
        return mdd, s, e

    def _percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        p = max(0.0, min(100.0, p))
        xs = sorted(data)
        k = (len(xs) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return xs[int(k)]
        d0 = xs[f] * (c - k)
        d1 = xs[c] * (k - f)
        return d0 + d1

    def _sharpe(self, returns: List[float], rf: float, ann: int) -> float:
        if not returns:
            return 0.0
        mu = statistics.mean(returns) - (rf / ann)
        sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        return (mu / sd) * math.sqrt(ann) if sd else 0.0

    def _sortino(self, returns: List[float], rf: float, ann: int) -> float:
        if not returns:
            return 0.0
        excess = [r - rf / ann for r in returns]
        downside = [min(0.0, r) for r in excess]
        dd = math.sqrt(sum(d * d for d in downside) / max(1, len(downside)))
        mu = statistics.mean(excess)
        return (mu / dd) * math.sqrt(ann) if dd else 0.0

    def _var_es(self, total_returns: List[float], conf: float) -> Tuple[float, float]:
        """
        VaR/ES on TOTAL RETURN distribution at confidence 'conf' (e.g., 0.95, 0.99).
        Returns (VaR, ES) as positive loss numbers (e.g., 0.12 => 12% loss).
        """
        if not total_returns:
            return (0.0, 0.0)
        losses = sorted([-r for r in total_returns])  # loss = -return
        # VaR at conf -> quantile of losses
        pct = conf * 100.0
        var = self._percentile(losses, pct)
        # ES = mean loss beyond VaR
        tail = [x for x in losses if x >= var - 1e-12]
        es = statistics.mean(tail) if tail else var
        return var, es


# --------------------------- CLI Smoke Test ------------------------------------

if __name__ == "__main__":
    assets = [
        AssetSpec("AAA", s0=100.0, mu=0.08, sigma=0.25, mr_kappa=0.05, mr_level_mult=1.0),
        AssetSpec("BBB", s0=50.0, mu=0.06, sigma=0.20),
        AssetSpec("CCC", s0=25.0, mu=0.10, sigma=0.35),
    ]
    corr = [
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]
    cfg = MCConfig(n_paths=500, horizon_days=252, seed=42, jump_lambda=0.01, jump_mu=-0.02, jump_sigma=0.05)
    mc = MonteCarlo(assets, cfg, corr=corr)
    rep = mc.run(portfolio=PortfolioSpec(weights={"AAA":0.5,"BBB":0.3,"CCC":0.2}))
    print(json.dumps(rep["portfolio"], separators=(",", ":"), default=float))