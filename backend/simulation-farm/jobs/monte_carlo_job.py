# simulation-farm/jobs/monte_carlo_job.py
"""
MonteCarloJob: multivariate MC of portfolio returns + report generation.

Model
-----
- Calibrate daily returns from historical data (CSV/Parquet; wide or long).
- Estimate mean vector (mu) & covariance (Sigma).
- Simulate i.i.d. daily multivariate normal returns for 'horizon_days'.
- Portfolio weights: equal-weight (EW) or inverse-vol (IVOL).
- Outputs median path (for equity charts) and risk distribution stats (VaR/CVaR).

Dependencies:
    pip install pandas numpy jinja2

CLI Example:
python -m simulation_farm.jobs.monte_carlo_job \
  --run-id run_2025_09_16_2210 \
  --data data/sp100_daily.parquet \
  --horizon-days 252 --n-paths 2000 \
  --weights ivol --recalc-vol-days 252 \
  --out-prefix runs/mc_sp100/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from simulation_farm.artifacts.reports.report_generator import ReportGenerator, ReportInputs  # type: ignore


# ----------------------------- Spec -----------------------------

@dataclass
class MonteCarloSpec:
    run_id: str
    data_path: str
    start: Optional[str] = None      # calibration window
    end: Optional[str] = None
    universe: Optional[List[str]] = None

    # MC controls
    n_paths: int = 2000
    horizon_days: int = 252
    weights: str = "ew"              # "ew" | "ivol"
    recalc_vol_days: int = 252       # lookback for vol used in ivol weights

    # reporting/export
    cash: float = 1_000_000.0
    output_prefix: str = "runs/mc_demo/"
    exporter: Tuple[str, Dict] = ("local", {"root": "artifacts/reports/out"})
    title: str = "Monte Carlo Simulation"
    tz: str = "UTC"

    # risk options
    var_level: float = 0.95          # 95% VaR/CVaR on daily returns


# ----------------------------- Job -----------------------------

class MonteCarloJob:
    def __init__(self, spec: MonteCarloSpec):
        self.spec = spec

    def run(self) -> Dict[str, str]:
        # 1) Load price data and compute daily returns (wide)
        px = _load_prices(self.spec.data_path, self.spec.start, self.spec.end, self.spec.universe)
        if px.shape[1] < 2:
            raise ValueError("Need at least 2 symbols for MC calibration.")
        rets_hist = px.pct_change().dropna(how="any")
        if rets_hist.empty:
            raise ValueError("No historical returns after filtering.")

        # 2) Estimate mu, Sigma
        mu = rets_hist.mean().values  # shape (N,)
        Sigma = rets_hist.cov().values  # shape (N,N)

        # 3) Choose portfolio weights
        W = _make_weights(self.spec.weights, rets_hist, self.spec.recalc_vol_days)  # pd.Series (N,)
        w = W.values  # ndarray
        # normalize strictly
        w = np.where(w < 0, 0.0, w) # type: ignore
        if w.sum() <= 0:
            raise ValueError("All weights are zero after preprocessing.")
        w = w / w.sum()

        # 4) Simulate multivariate normal daily returns
        n, T = len(w), int(self.spec.horizon_days)
        P = int(self.spec.n_paths)
        # Cholesky (regularize if needed)
        Sigma_reg = _nearest_psd(Sigma)
        L = np.linalg.cholesky(Sigma_reg)
        # simulate Z ~ N(0, I) -> R = mu + L Z
        rng = np.random.default_rng()
        Z = rng.standard_normal(size=(T, P, n))  # time x path x assets
        # matmul: (n,n) @ (T,P,n)^T per (T,P) — do batched multiply
        # Efficient: (Z @ L.T) gives (T,P,n) * (n,n) -> (T,P,n)
        R_assets = Z @ L.T
        R_assets += mu  # type: ignore # broadcast

        # 5) Portfolio aggregation
        # daily portfolio return per (T,P): dot along assets
        R_port = (R_assets * w).sum(axis=2)  # (T, P)
        # equity paths (start at 1.0; scale to cash later)
        Eq_paths = np.cumprod(1.0 + R_port, axis=0)  # (T,P)

        # representative series for charts: median path
        eq_median = np.median(Eq_paths, axis=1) * self.spec.cash
        pnl_median = np.median(R_port, axis=1)  # daily fractional return (median across paths)

        # 6) Risk stats (daily VaR/CVaR from simulated returns across all T*P)
        flat_returns = R_port.ravel()
        var, cvar = _var_cvar(flat_returns, self.spec.var_level)

        # 7) Build report inputs
        dates = _future_dates(rets_hist.index[-1], T)
        sectors = {"Portfolio": 100.0}  # placeholder
        var_hist = _histogram_pct(flat_returns, bins=41)  # for risk report

        # drawdowns of representative path (median)
        eq_series = eq_median.tolist()
        dd_series = _compute_drawdown(eq_series)

        # KPIs for summary
        kpis = _basic_kpis(dates, eq_series, pnl_median.tolist())
        kpis.update({
            "vol_annual": float(np.std(flat_returns) * math.sqrt(252)),
            "var": float(var),
            "cvar": float(cvar),
            "trades": None,   # MC doesn't produce trades
        })

        rg = ReportGenerator(
            run_id=self.spec.run_id,
            output_prefix=self.spec.output_prefix.rstrip("/") + f"/{self.spec.run_id}/",
            exporter=self.spec.exporter,
        )

        inputs = ReportInputs(
            dates=dates,
            equity=eq_series,
            benchmark=None,
            drawdown=dd_series,
            pnl=pnl_median.tolist(),
            title=self.spec.title,
            strategy=f"mc_{self.spec.weights}",
            start_date=dates[0] if dates else None,
            end_date=dates[-1] if dates else None,
            tz=self.spec.tz,
            params={
                "n_paths": self.spec.n_paths,
                "horizon_days": self.spec.horizon_days,
                "weights": self.spec.weights,
                "var_level": self.spec.var_level,
            },
            config={
                "engine": "mc-normal-v1",
                "data": Path(self.spec.data_path).name,
                "universe": f"{rets_hist.shape[1]} symbols",
                "cash": self.spec.cash,
            },
            # Risk blocks
            var_hist=var_hist,
            factors={},                 # not computed here
            sectors=sectors,
            drawdowns_table=[],         # could be computed from eq_median if desired
            stress=[],
            diag={"warnings": []},
        )

        urls = rg.generate_all(inputs)
        return urls


# ----------------------------- Helpers -----------------------------

def _load_prices(path: str, start: Optional[str], end: Optional[str], universe: Optional[List[str]]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    cols = {c.lower(): c for c in df.columns}
    def has(*names): return all(n in cols or n in df.columns for n in names)

    if has("date") and not has("symbol", "close"):
        # wide format
        df = df.rename(columns={cols.get("date", "date"): "date"})
        df["date"] = pd.to_datetime(df["date"])
        wide = df.set_index("date").sort_index()
    elif has("date", "symbol", "close"):
        # long → pivot
        df = df.rename(columns={
            cols.get("date", "date"): "date",
            cols.get("symbol", "symbol"): "symbol",
            cols.get("close", "close"): "close",
        })
        df["date"] = pd.to_datetime(df["date"])
        wide = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    else:
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            wide = df.copy()
        else:
            raise ValueError("Unrecognized data shape. Expect wide (date + symbols) or long (date,symbol,close).")

    if start:
        wide = wide[wide.index >= pd.to_datetime(start)]
    if end:
        wide = wide[wide.index <= pd.to_datetime(end)]

    if universe:
        keep = [c for c in universe if c in wide.columns]
        if not keep:
            raise ValueError("Universe symbols not found in data.")
        wide = wide[keep]

    wide = wide.dropna(axis=1, how="all").ffill().dropna(how="any")
    return wide


def _make_weights(kind: str, rets_hist: pd.DataFrame, vol_lookback: int) -> pd.Series:
    kind = (kind or "ew").lower()
    cols = rets_hist.columns
    if kind in ("ew", "equal_weight"):
        return pd.Series(1.0, index=cols)
    # inverse vol: 1/σ over last vol_lookback days
    tail = rets_hist.tail(int(vol_lookback))
    vol = tail.std().replace(0.0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv.sum() <= 0:
        return pd.Series(1.0, index=cols)
    return inv / inv.sum()


def _nearest_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    # Make covariance positive semidefinite (Higham 1988 projection)
    B = (A + A.T) / 2.0
    eigval, eigvec = np.linalg.eigh(B)
    eigval[eigval < eps] = eps
    return (eigvec @ np.diag(eigval) @ eigvec.T)


def _var_cvar(returns: np.ndarray, level: float) -> Tuple[float, float]:
    """
    Returns daily VaR and CVaR (negative quantiles as positive numbers if losses),
    but here we keep sign (e.g., VaR = -0.025 means -2.5%).
    """
    if returns.size == 0:
        return (0.0, 0.0)
    q = np.quantile(returns, 1.0 - level)
    tail = returns[returns <= q]
    cvar = tail.mean() if tail.size else q
    return (float(q), float(cvar))


def _future_dates(last_dt: pd.Timestamp, horizon: int) -> List[str]:
    # Generate trading-day-like sequence (just daily increments, skipping weekends)
    dates = []
    d = pd.Timestamp(last_dt)
    while len(dates) < horizon:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
    return dates


def _compute_drawdown(equity: List[float]) -> List[float]:
    peak = -float("inf")
    out = []
    for v in equity:
        peak = max(peak, v)
        out.append((v / peak - 1.0) if peak > 0 else 0.0)
    return out


def _basic_kpis(dates: List[str], equity: List[float], pnl: Optional[List[float]]) -> Dict:
    if not equity:
        return {}
    first, last = equity[0], equity[-1]
    years = max(1e-9, len(dates) / 252.0)
    cagr = (last / first) ** (1 / years) - 1 if first > 0 else 0.0
    rets = pnl or []
    if not rets:
        sharpe = sortino = vol_annual = 0.0
    else:
        mu = float(np.mean(rets))
        sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        dwn = float(np.std([min(0.0, r) for r in rets], ddof=1)) if len(rets) > 1 else 0.0
        vol_annual = sd * math.sqrt(252)
        sharpe = (mu * 252) / (sd + 1e-12)
        sortino = (mu * 252) / (dwn + 1e-12)
    max_dd = min(_compute_drawdown(equity)) if equity else 0.0
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd, "vol_annual": vol_annual}


def _histogram_pct(x: np.ndarray, bins: int = 41) -> Dict[str, List[float]]:
    x = np.asarray(x)
    if x.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(x * 100.0, bins=bins)  # percent scale
    # represent by bin centers for display
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"bins": centers.tolist(), "counts": counts.tolist()}


# ----------------------------- CLI -----------------------------

def _parse_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Run a multivariate Monte Carlo simulation and emit reports.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", required=True, help="CSV/Parquet (wide or long)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--horizon-days", type=int, default=252)
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--weights", default="ew", choices=["ew", "ivol"])
    ap.add_argument("--recalc-vol-days", type=int, default=252)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--out-prefix", default="runs/mc_demo/")
    ap.add_argument("--exporter", default="local", choices=["local", "s3", "gcs"])
    ap.add_argument("--export-arg", action="append", default=[], help="key=value (e.g., root=artifacts/reports/out)")
    args = ap.parse_args()

    kwargs = {}
    for kv in args.export_arg:
        if "=" in kv:
            k, v = kv.split("=", 1)
            kwargs[k] = v

    return MonteCarloSpec(
        run_id=args.run_id,
        data_path=args.data,
        start=args.start,
        end=args.end,
        n_paths=args.n_paths,
        horizon_days=args.horizon_days,
        weights=args.weights,
        recalc_vol_days=args.recalc_vol_days,
        cash=args.cash,
        output_prefix=args.out_prefix,
        exporter=(args.exporter, kwargs or ({"root": "artifacts/reports/out"} if args.exporter == "local" else {})),
    )


if __name__ == "__main__":
    spec = _parse_cli()
    job = MonteCarloJob(spec)
    urls = job.run()
    import json
    print(json.dumps(urls, indent=2))