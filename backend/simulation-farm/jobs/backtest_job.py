# simulation-farm/jobs/backtest_job.py
"""
BacktestJob: daily vector backtester + report generation.

Dependencies (recommended):
    pip install pandas numpy jinja2

Usage (as a library)
--------------------
from simulation_farm.jobs.backtest_job import BacktestJob, BacktestSpec

spec = BacktestSpec(
    run_id="run_2025_09_16_2130",
    data_path="data/sp500_daily.parquet",    # or .csv; columns: date, symbol, close (wide also OK)
    strategy="momentum",                      # momentum | mean_reversion | equal_weight
    params={"lookback_days": 90, "top_k": 50},
    cash=1_000_000,
    start="2020-01-01",
    end="2023-12-31",
    costs_bps=5.0,        # round-trip cost in basis points (approx)
    slippage_bps=2.0,     # applied to trades like extra cost
    max_weight=0.03,      # per-name cap (3%)
    rebalance_days=5,     # rebalance every N trading days
    output_prefix="runs/momentum_us/",
    exporter=("local", {"root": "artifacts/reports/out"}),  # or ("s3", {...}) / ("gcs", {...})
)

job = BacktestJob(spec)
result_urls = job.run()
print(result_urls)

CLI
---
python -m simulation_farm.jobs.backtest_job \
  --run-id run_2025_09_16_2130 \
  --data data/sp500_daily.parquet \
  --strategy momentum \
  --start 2020-01-01 --end 2023-12-31 \
  --cash 1000000 --lookback-days 90 --top-k 50
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# soft deps
try:
    import pandas as pd  # type: ignore
    import numpy as np   # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("backtest_job.py requires pandas and numpy. Install with: pip install pandas numpy") from e

from pathlib import Path

# reports
from simulation_farm.artifacts.reports.report_generator import ReportGenerator, ReportInputs  # type: ignore


# --------------------------- Spec ---------------------------

@dataclass
class BacktestSpec:
    run_id: str
    data_path: str
    strategy: str = "momentum"   # momentum | mean_reversion | equal_weight
    params: Dict = field(default_factory=dict)
    cash: float = 1_000_000.0
    start: Optional[str] = None
    end: Optional[str] = None
    universe: Optional[List[str]] = None

    # trading controls
    costs_bps: float = 5.0          # round-trip fee/slippage approximation (bps)
    slippage_bps: float = 2.0       # extra, applied to turnover (bps)
    max_weight: float = 0.05        # 5% cap per name
    rebalance_days: int = 5         # rebalance every N trading days
    max_names: Optional[int] = None # cap on number of names held (post ranking)

    # reporting/export
    output_prefix: str = "runs/demo/"
    exporter: Tuple[str, Dict] = ("local", {"root": "artifacts/reports/out"})
    title: str = "Backtest Report"
    tz: str = "UTC"


# --------------------------- Job ---------------------------

class BacktestJob:
    def __init__(self, spec: BacktestSpec):
        self.spec = spec

    def run(self) -> Dict[str, str]:
        # 1) Load price data → wide close price matrix
        df_px = _load_prices(self.spec.data_path, self.spec.start, self.spec.end, self.spec.universe)
        if df_px.empty:
            raise ValueError("No price data loaded after filtering.")
        dates = df_px.index
        symbols = list(df_px.columns)

        # 2) Compute daily returns
        rets = df_px.pct_change().fillna(0.0)

        # 3) Strategy weights (vectorized per rebalance)
        weights = _compute_weights(
            strategy=self.spec.strategy,
            prices=df_px,
            returns=rets,
            params=self.spec.params,
            rebalance_days=self.spec.rebalance_days,
            max_weight=self.spec.max_weight,
            max_names=self.spec.max_names,
        )  # DataFrame index=dates, cols=symbols, rows sum to ~1.0

        # 4) Portfolio path
        port = _portfolio_path(
            returns=rets,
            weights=weights,
            costs_bps=self.spec.costs_bps,
            slippage_bps=self.spec.slippage_bps,
        )
        equity = list(port["equity"].values)
        pnl = list(port["pnl"].values)  # daily return as fraction
        drawdown = list(port["drawdown"].values)
        turnover = list(port["turnover"].values)

        # 5) Trades detail snapshot (basic)
        trades = _extract_trades(weights, df_px)

        # 6) Benchmark (equal-weight universe, rebalanced daily)
        bench = rets.mean(axis=1).cumprod()
        bench = list((1.0 + rets.mean(axis=1)).cumprod() * (self.spec.cash))

        # 7) Report
        rg = ReportGenerator(
            run_id=self.spec.run_id,
            output_prefix=self.spec.output_prefix.rstrip("/") + f"/{self.spec.run_id}/",
            exporter=self.spec.exporter,
        )

        inputs = ReportInputs(
            dates=[d.strftime("%Y-%m-%d") for d in dates],
            equity=equity,
            benchmark=bench,
            drawdown=drawdown,
            pnl=pnl,
            title=self.spec.title,
            strategy=self.spec.strategy,
            start_date=dates[0].strftime("%Y-%m-%d"),
            end_date=dates[-1].strftime("%Y-%m-%d"),
            tz=self.spec.tz,
            params=self.spec.params,
            config={
                "engine": "vector-daily-v1",
                "data": Path(self.spec.data_path).name,
                "universe": f"{len(symbols)} symbols",
                "cash": self.spec.cash,
                "controls": {
                    "costs_bps": self.spec.costs_bps,
                    "slippage_bps": self.spec.slippage_bps,
                    "max_weight": self.spec.max_weight,
                    "rebalance_days": self.spec.rebalance_days,
                    "max_names": self.spec.max_names,
                },
            },
            trades=trades[:2000],  # cap to avoid huge HTML
            diag={
                "turnover": {"dates": [d.strftime("%Y-%m-%d") for d in dates], "values": [t * 100 for t in turnover]},
                # (fill runtime/CPU/peak RSS from runner if available)
                "runtime_s": None,
                "cpu_s": None,
                "peak_rss_mb": None,
                "speed_x": None,
                "warnings": [],
            },
        )
        urls = rg.generate_all(inputs)
        return urls


# --------------------------- Data loading ---------------------------

def _load_prices(path: str, start: Optional[str], end: Optional[str], universe: Optional[List[str]]) -> pd.DataFrame:
    """
    Accepts:
      - wide: CSV/Parquet with first column date index, others symbols
      - long: CSV/Parquet with columns: date, symbol, close
    Returns a wide DataFrame of close prices (index=date, columns=symbol)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Normalize columns to lower
    cols = {c.lower(): c for c in df.columns}
    def has(*names): return all(n in cols or n in df.columns for n in names)

    # Heuristics: wide vs long
    if has("date") and not has("symbol", "close"):
        # Likely wide: date + many symbols
        df = df.rename(columns={cols.get("date", "date"): "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        wide = df
    elif has("date", "symbol", "close"):
        # Long → pivot
        df = df.rename(columns={
            cols.get("date", "date"): "date",
            cols.get("symbol", "symbol"): "symbol",
            cols.get("close", "close"): "close",
        })
        df["date"] = pd.to_datetime(df["date"])
        wide = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    else:
        # Already indexed?
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            wide = df.copy()
        else:
            raise ValueError("Unrecognized data shape. Expect wide (date + symbols) or long (date,symbol,close).")

    if start:
        wide = wide[wide.index >= pd.to_datetime(start)]
    if end:
        wide = wide[wide.index <= pd.to_datetime(end)]

    if universe:
        available = [c for c in universe if c in wide.columns]
        if not available:
            raise ValueError("Universe symbols not found in data.")
        wide = wide[available]

    # Drop all-NaN columns, forward-fill missing, drop initial NaNs
    wide = wide.dropna(axis=1, how="all").ffill().dropna(how="any")
    return wide


# --------------------------- Strategies (vectorized) ---------------------------

def _compute_weights(
    strategy: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    params: Dict,
    rebalance_days: int,
    max_weight: float,
    max_names: Optional[int],
) -> pd.DataFrame:
    """
    Produce target weights per rebalance day; forward-fill between rebalances.
    Sum of positive weights normalized to 1.0 (long-only).
    """
    strategy = strategy.lower()
    if strategy == "momentum":
        lookback = int(params.get("lookback_days", 90))
        rank = prices.pct_change(lookback)
        score = rank
    elif strategy == "mean_reversion":
        lookback = int(params.get("lookback_days", 10))
        score = -prices.pct_change(lookback)
    elif strategy in ("ew", "equal_weight"):
        score = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    else:
        # fallback: equal weight
        score = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

    # Rebalance schedule (every Nth day)
    idx = prices.index
    mask = np.zeros(len(idx), dtype=bool)
    mask[::max(1, int(rebalance_days))] = True
    reb_idx = idx[mask]

    weights_list = []
    top_k = params.get("top_k")
    cap = float(max_weight)

    for t in reb_idx:
        s = score.loc[t].copy()
        if top_k:
            # keep top K positive scores
            s = s.sort_values(ascending=False).head(int(top_k))
            s[s < 0] = 0.0
        else:
            s[s < 0] = 0.0

        if s.sum() <= 0:
            w = pd.Series(0.0, index=prices.columns)
        else:
            w = s / s.sum()
            if cap > 0:
                w = w.clip(upper=cap)
                # renormalize after cap
                total = w.sum()
                w = (w / total) if total > 0 else w

        # expand to full universe
        w = w.reindex(prices.columns).fillna(0.0)
        w.name = t
        weights_list.append(w)

    W = pd.DataFrame(weights_list)
    W = W.reindex(prices.index).ffill().fillna(0.0)
    return W


# --------------------------- Portfolio path ---------------------------

def _portfolio_path(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    costs_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    """
    Compute daily portfolio equity given target weights and returns.
    Costs modeled via turnover * (costs_bps + slippage_bps).
    """
    # daily portfolio return before costs
    rp = (weights.shift().fillna(0.0) * returns).sum(axis=1)

    # turnover = 0.5 * L1 change in weights (buy+sell not double-counted)
    dw = (weights - weights.shift()).abs().sum(axis=1) * 0.5
    # cost per day (as return hit)
    c_bps = float(costs_bps) + float(slippage_bps)
    cost_ret = dw * (c_bps / 10_000.0)

    net_ret = rp - cost_ret
    equity = (1.0 + net_ret).cumprod()
    # scale to cash notional for report pages
    equity_cash = equity * 1_000_000.0  # normalized; report can show relative path
    # drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0

    out = pd.DataFrame({
        "pnl": net_ret.values,            # fractional daily return
        "equity": equity_cash.values,     # scaled equity
        "turnover": dw.values,
        "drawdown": dd.values,
    }, index=returns.index)
    return out


# --------------------------- Trades extraction ---------------------------

def _extract_trades(weights: pd.DataFrame, prices: pd.DataFrame) -> List[Dict]:
    """
    Convert weight changes into approximate trades snapshot for report tables.
    Assumes unit notional == 1.0 at previous close; this is illustrative.
    """
    trades: List[Dict] = []
    W0 = weights.shift().fillna(0.0)
    dW = (weights - W0)
    for ts, row in dW.iterrows():
        changes = row[row.abs() > 1e-9]
        if changes.empty:
            continue
        px = prices.loc[ts] # type: ignore
        for sym, dw in changes.items():
            if dw == 0 or sym not in px or pd.isna(px[sym]):
                continue
            side = "buy" if dw > 0 else "sell"
            # Synthetic qty: proportional to weight change; notional 1 → qty = dw / px
            qty = abs(dw) / max(px[sym], 1e-9)
            trades.append({
                "ts": ts.strftime("%Y-%m-%d"), # type: ignore
                "symbol": sym,
                "side": side,
                "qty": float(qty),
                "px": float(px[sym]),
                "fee": 0.0,
            })
    return trades


# --------------------------- CLI ---------------------------

def _parse_cli():
    import argparse, json

    ap = argparse.ArgumentParser(description="Run a vectorized backtest and emit reports.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", required=True, help="CSV/Parquet (wide or long)")
    ap.add_argument("--strategy", default="momentum", choices=["momentum", "mean_reversion", "equal_weight"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--lookback-days", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--rebalance-days", type=int, default=5)
    ap.add_argument("--max-weight", type=float, default=0.05)
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--costs-bps", type=float, default=5.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--out-prefix", default="runs/demo/")
    ap.add_argument("--exporter", default="local", choices=["local", "s3", "gcs"])
    ap.add_argument("--export-arg", action="append", default=[], help="key=value (e.g., root=artifacts/reports/out)")
    args = ap.parse_args()

    kwargs = {}
    for kv in args.export_arg:
        if "=" in kv:
            k, v = kv.split("=", 1)
            kwargs[k] = v

    spec = BacktestSpec(
        run_id=args.run_id,
        data_path=args.data,
        strategy=args.strategy,
        params={k: v for k, v in {
            "lookback_days": args.lookback_days,
            "top_k": args.top_k,
        }.items() if v is not None},
        cash=args.cash,
        start=args.start,
        end=args.end,
        costs_bps=args.costs_bps,
        slippage_bps=args.slippage_bps,
        max_weight=args.max_weight,
        rebalance_days=args.rebalance_days,
        max_names=args.max_names,
        output_prefix=args.out_prefix,
        exporter=(args.exporter, kwargs or ({"root": "artifacts/reports/out"} if args.exporter == "local" else {})),
    )
    return spec


if __name__ == "__main__":
    spec = _parse_cli()
    job = BacktestJob(spec)
    urls = job.run()
    import json
    print(json.dumps(urls, indent=2))