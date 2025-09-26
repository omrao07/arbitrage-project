# simulation-farm/jobs/replay_job.py
"""
ReplayJob: deterministic replay of historical data streams.

- Reads CSV/Parquet with timestamped prices.
- Steps through events at configurable speed (e.g. 1x, 4x).
- At each step, strategy logic can fire signals/trades.
- Produces equity path, trades log, reports.

Intended to test execution logic under "real" timestamped flow.

Dependencies:
    pip install pandas numpy jinja2

CLI example:
python -m simulation_farm.jobs.replay_job \
  --run-id run_2025_09_16_2230 \
  --data data/nasdaq_intraday_2022.parquet \
  --strategy momentum \
  --speed 4.0 --cash 250000 \
  --start 2022-01-01 --end 2022-12-31
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation_farm.artifacts.reports.report_generator import ReportGenerator, ReportInputs # type: ignore


# ------------------------- Spec -------------------------

@dataclass
class ReplaySpec:
    run_id: str
    data_path: str
    strategy: str = "momentum"
    start: Optional[str] = None
    end: Optional[str] = None
    universe: Optional[List[str]] = None

    cash: float = 250_000.0
    speed: float = 1.0              # replay speed (1.0 = realtime, 4.0 = 4x faster)
    window_minutes: int = 30        # for intraday indicators

    output_prefix: str = "runs/replay_demo/"
    exporter: Tuple[str, Dict] = ("local", {"root": "artifacts/reports/out"})
    title: str = "Replay Report"
    tz: str = "UTC"


# ------------------------- Job -------------------------

class ReplayJob:
    def __init__(self, spec: ReplaySpec):
        self.spec = spec

    def run(self) -> Dict[str, str]:
        # 1) Load tick/intraday or daily data
        df = _load_prices(self.spec.data_path, self.spec.start, self.spec.end, self.spec.universe)
        if df.empty:
            raise ValueError("No data loaded for replay window.")
        df = df.sort_index()

        # 2) Apply simple strategy logic
        equity, trades = _simulate_replay(df, self.spec)

        # 3) Build KPIs & drawdown
        dates = [d.strftime("%Y-%m-%d %H:%M") for d in df.index]
        dd = _compute_drawdown(equity)
        pnl = [0.0] + [equity[i]/equity[i-1]-1 for i in range(1,len(equity))]

        kpis = _basic_kpis(dates, equity, pnl)

        # 4) Emit report
        rg = ReportGenerator(
            run_id=self.spec.run_id,
            output_prefix=self.spec.output_prefix.rstrip("/") + f"/{self.spec.run_id}/",
            exporter=self.spec.exporter,
        )

        inputs = ReportInputs(
            dates=dates,
            equity=equity,
            benchmark=None,
            drawdown=dd,
            pnl=pnl,
            title=self.spec.title,
            strategy=self.spec.strategy,
            start_date=dates[0],
            end_date=dates[-1],
            tz=self.spec.tz,
            params={"speed": self.spec.speed, "window_minutes": self.spec.window_minutes},
            config={"engine": "replay-v1", "data": Path(self.spec.data_path).name, "cash": self.spec.cash},
            trades=trades[:1000],
            diag={"warnings": []},
        )
        return rg.generate_all(inputs)


# ------------------------- Helpers -------------------------

def _load_prices(path: str, start: Optional[str], end: Optional[str], universe: Optional[List[str]]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    cols = {c.lower(): c for c in df.columns}
    if {"date","symbol","close"} <= set(cols.keys()):
        df = df.rename(columns={cols["date"]:"date", cols["symbol"]:"symbol", cols["close"]:"close"})
        df["date"] = pd.to_datetime(df["date"])
        wide = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    elif "date" in cols:
        df = df.rename(columns={cols["date"]:"date"})
        df["date"] = pd.to_datetime(df["date"])
        wide = df.set_index("date")
    else:
        raise ValueError("Data must have date/symbol/close or wide format.")

    if start: wide = wide[wide.index >= pd.to_datetime(start)]
    if end: wide = wide[wide.index <= pd.to_datetime(end)]
    if universe: wide = wide[[c for c in universe if c in wide.columns]]
    return wide.dropna(how="any")


def _simulate_replay(df: pd.DataFrame, spec: ReplaySpec) -> Tuple[List[float], List[Dict]]:
    """
    Very simple: momentum rule using rolling window. 
    On signal, allocate fully to best symbol, else in cash.
    """
    cash = spec.cash
    equity = []
    trades: List[Dict] = []
    current_sym = None
    pos_value = 0.0

    for i, (ts, row) in enumerate(df.iterrows()):
        if i < spec.window_minutes:
            equity.append(cash)
            continue
        # compute momentum score = return over window
        window = df.iloc[i-spec.window_minutes:i]
        rets = window.iloc[-1] / window.iloc[0] - 1
        best_sym = rets.idxmax()
        best_ret = rets.max()

        # rotate if signal changes
        if best_sym != current_sym and best_ret > 0:
            # close old pos
            if current_sym:
                trades.append({"ts":ts, "symbol":current_sym, "side":"sell", "qty":1, "px":row[current_sym], "fee":0})
            current_sym = best_sym
            trades.append({"ts":ts, "symbol":current_sym, "side":"buy", "qty":1, "px":row[current_sym], "fee":0})

        # update equity
        if current_sym:
            pos_value = row[current_sym]
            total = pos_value  # 1 share synthetic
        else:
            total = cash
        equity.append(total)
    return equity, trades


def _compute_drawdown(equity: List[float]) -> List[float]:
    peak = -float("inf")
    dd = []
    for v in equity:
        peak = max(peak, v)
        dd.append((v/peak - 1.0) if peak>0 else 0.0)
    return dd


def _basic_kpis(dates: List[str], equity: List[float], pnl: List[float]) -> Dict:
    if not equity: return {}
    first, last = equity[0], equity[-1]
    years = max(1e-9, len(dates)/252)
    cagr = (last/first)**(1/years) - 1 if first>0 else 0.0
    sharpe = (np.mean(pnl)*252) / (np.std(pnl)+1e-12)
    sortino = (np.mean(pnl)*252) / (np.std([min(0,r) for r in pnl])+1e-12)
    max_dd = min(_compute_drawdown(equity)) if equity else 0
    vol = np.std(pnl)*math.sqrt(252)
    return {"cagr":cagr,"sharpe":sharpe,"sortino":sortino,"max_dd":max_dd,"vol_annual":vol,"trades":len(pnl)}


# ------------------------- CLI -------------------------

def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Replay historical data and emit reports.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--strategy", default="momentum")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cash", type=float, default=250000)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--window-minutes", type=int, default=30)
    ap.add_argument("--out-prefix", default="runs/replay_demo/")
    ap.add_argument("--exporter", default="local", choices=["local","s3","gcs"])
    ap.add_argument("--export-arg", action="append", default=[], help="key=value")
    args = ap.parse_args()

    kwargs = {}
    for kv in args.export_arg:
        if "=" in kv: k,v = kv.split("=",1); kwargs[k]=v
    return ReplaySpec(
        run_id=args.run_id,
        data_path=args.data,
        strategy=args.strategy,
        start=args.start,
        end=args.end,
        cash=args.cash,
        speed=args.speed,
        window_minutes=args.window_minutes,
        output_prefix=args.out_prefix,
        exporter=(args.exporter, kwargs or ({"root":"artifacts/reports/out"} if args.exporter=="local" else {})),
    )

if __name__ == "__main__":
    spec = _parse_cli()
    job = ReplayJob(spec)
    urls = job.run()
    import json; print(json.dumps(urls, indent=2))