# simulation-farm/jobs/stress_test_job.py
"""
StressTestJob: portfolio stress testing + report generation.

Scenarios supported
-------------------
1) Historical window ("historical"):
   - Replay a past period (e.g., 2020-02-15..2020-04-15) using equal-weight or inverse-vol
     weights over the provided universe. Produces full equity path, drawdowns, VaR hist.

2) Single-day shock ("single_day"):
   - Apply a uniform equity return shock (e.g., -0.25) to the whole universe weights.
   - Optional "vol_spike" factor scales the estimated daily sigma and adds a variance hit proxy.
   - Produces a 2-point equity path (pre/post).

3) Parallel rate shift ("rates"):
   - Given portfolio duration (years) and a shift in basis points (e.g., +50),
     estimate P&L = -Duration * Δy (as fraction). Adds a 2-point equity path.

4) Factor shocks ("factor"):
   - Given factor exposures (beta dict) and shock magnitudes (fractional),
     P&L ≈ sum(beta_i * shock_i). Adds a 2-point equity path.

Weights
-------
- "ew"  : equal-weight across available symbols
- "ivol": inverse volatility (lookback days configurable)

Inputs
------
- Daily (or higher) price panel in CSV/Parquet, wide or long:
  - Wide: columns = date, <symbols...>
  - Long: columns = date, symbol, close

Dependencies:
    pip install pandas numpy jinja2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from simulation_farm.artifacts.reports.report_generator import ReportGenerator, ReportInputs # type: ignore


# ----------------------------- Spec -----------------------------

@dataclass
class StressTestSpec:
    run_id: str
    data_path: str

    # Universe/time filters for historical stress or for estimating vols
    start: Optional[str] = None
    end: Optional[str] = None
    universe: Optional[List[str]] = None

    weights: str = "ew"               # "ew" | "ivol"
    vol_lookback_days: int = 252

    # Cash notional for report scaling
    cash: float = 1_000_000.0

    # Scenario knobs (use any subset)
    # 1) Historical
    historical_window: Optional[str] = None   # "YYYY-MM-DD:YYYY-MM-DD"
    # 2) Single day
    equity_drop: Optional[float] = None       # e.g. -0.25 for -25%
    vol_spike: Optional[float] = None         # e.g. 2.0 doubles sigma proxy
    # 3) Rates
    rate_shift_bps: Optional[float] = None    # e.g. +50 (bps)
    duration_years: float = 0.0               # effective portfolio duration
    # 4) Factor shocks
    factor_betas: Dict[str, float] = field(default_factory=dict)     # {"MKT": 0.6, "MOM": 0.5}
    factor_shocks: Dict[str, float] = field(default_factory=dict)    # {"MKT": -0.05, "MOM": +0.02}

    # reporting/export
    output_prefix: str = "runs/stress_demo/"
    exporter: Tuple[str, Dict] = ("local", {"root": "artifacts/reports/out"})
    title: str = "Stress Test Report"
    tz: str = "UTC"


# ----------------------------- Job -----------------------------

class StressTestJob:
    def __init__(self, spec: StressTestSpec):
        self.spec = spec

    def run(self) -> Dict[str, str]:
        # Load prices (for weights/vols/historical scenario)
        px = _load_prices(self.spec.data_path, self.spec.start, self.spec.end, self.spec.universe)
        if px.empty:
            raise ValueError("No price data after filtering.")
        rets = px.pct_change().dropna(how="any")
        if rets.empty:
            raise ValueError("Not enough data to compute returns.")

        w_series = _make_weights(self.spec.weights, rets, self.spec.vol_lookback_days)  # pd.Series

        # Aggregate scenarios
        run_equity_dates: List[str] = []
        run_equity_values: List[float] = []
        benchmark: Optional[List[float]] = None
        drawdown: Optional[List[float]] = None
        pnl_series: Optional[List[float]] = None

        stress_rows: List[Dict] = []
        var_hist = {"bins": [], "counts": []}

        # 1) Historical window stress
        if self.spec.historical_window:
            eq_dates, eq_vals, dd, pnl, var_hist = _historical_stress(px, rets, w_series, self.spec.historical_window, self.spec.cash)
            run_equity_dates = eq_dates
            run_equity_values = eq_vals
            drawdown = dd
            pnl_series = pnl
            stress_rows.append({"name": f"Historical {self.spec.historical_window}", "pnl_pct": (eq_vals[-1]/eq_vals[0]-1)*100, "note":"replay"})
        else:
            # If no historical path, create a neutral 2-point path to host single-day outputs
            run_equity_dates = [px.index[-2].strftime("%Y-%m-%d"), px.index[-1].strftime("%Y-%m-%d")]
            run_equity_values = [self.spec.cash, self.spec.cash]
            drawdown = _compute_drawdown(run_equity_values)
            pnl_series = [0.0, 0.0]

        # 2) Single-day uniform equity shock
        if self.spec.equity_drop is not None:
            shock = float(self.spec.equity_drop)
            # portfolio one-day return under uniform shock
            port_ret = shock  # weights sum to 1
            # vol spike proxy: if provided, subtract an extra % of daily sigma estimate
            if self.spec.vol_spike and self.spec.vol_spike > 1.0:
                sigma = float(rets.dot(w_series).std())
                extra = (self.spec.vol_spike - 1.0) * sigma
                port_ret += (-abs(extra))
            last = run_equity_values[-1]
            run_equity_values.append(last * (1.0 + port_ret))
            run_equity_dates.append(_next_trading_date(run_equity_dates[-1]))
            drawdown = _compute_drawdown(run_equity_values)
            pnl_series.append(port_ret)
            stress_rows.append({"name": f"Single-day equity {shock:+.1%}", "pnl_pct": port_ret*100, "note": f"vol×{self.spec.vol_spike or 1.0:.2f}"})

        # 3) Rates parallel shift
        if self.spec.rate_shift_bps is not None and self.spec.duration_years:
            dy = float(self.spec.rate_shift_bps) / 10_000.0
            pnl_pct = -self.spec.duration_years * dy
            last = run_equity_values[-1]
            run_equity_values.append(last * (1.0 + pnl_pct))
            run_equity_dates.append(_next_trading_date(run_equity_dates[-1]))
            drawdown = _compute_drawdown(run_equity_values)
            pnl_series.append(pnl_pct)
            stress_rows.append({"name": f"Rates {self.spec.rate_shift_bps:+.0f} bps", "pnl_pct": pnl_pct*100, "note": f"Duration={self.spec.duration_years:.2f}y"})

        # 4) Factor shocks
        if self.spec.factor_betas and self.spec.factor_shocks:
            pnl_pct = 0.0
            notes = []
            for f, beta in self.spec.factor_betas.items():
                s = float(self.spec.factor_shocks.get(f, 0.0))
                pnl_pct += beta * s
                notes.append(f"{f}:{beta:+.2f}×{s:+.2%}")
            last = run_equity_values[-1]
            run_equity_values.append(last * (1.0 + pnl_pct))
            run_equity_dates.append(_next_trading_date(run_equity_dates[-1]))
            drawdown = _compute_drawdown(run_equity_values)
            pnl_series.append(pnl_pct)
            stress_rows.append({"name": "Factor shocks", "pnl_pct": pnl_pct*100, "note": "; ".join(notes)})

        # KPIs
        kpis = _basic_kpis(run_equity_dates, run_equity_values, pnl_series)

        # Build reports
        rg = ReportGenerator(
            run_id=self.spec.run_id,
            output_prefix=self.spec.output_prefix.rstrip("/") + f"/{self.spec.run_id}/",
            exporter=self.spec.exporter,
        )

        inputs = ReportInputs(
            dates=run_equity_dates,
            equity=run_equity_values,
            benchmark=None,
            drawdown=drawdown,
            pnl=pnl_series,
            title=self.spec.title,
            strategy="stress_test",
            start_date=run_equity_dates[0],
            end_date=run_equity_dates[-1],
            tz=self.spec.tz,
            params={
                "weights": self.spec.weights,
                "vol_lookback_days": self.spec.vol_lookback_days,
                "historical_window": self.spec.historical_window,
                "equity_drop": self.spec.equity_drop,
                "vol_spike": self.spec.vol_spike,
                "rate_shift_bps": self.spec.rate_shift_bps,
                "duration_years": self.spec.duration_years,
                "factor_betas": self.spec.factor_betas,
                "factor_shocks": self.spec.factor_shocks,
            },
            config={
                "engine": "stress-v1",
                "data": Path(self.spec.data_path).name,
                "universe": f"{len(px.columns)} symbols",
                "cash": self.spec.cash,
            },
            # Risk page blocks
            var_hist=var_hist,
            factors=self.spec.factor_betas or {},
            sectors={},  # optional
            drawdowns_table=_top_drawdowns(run_equity_dates, run_equity_values, k=10), # type: ignore
            stress=[{"name": r["name"], "pnl_pct": r["pnl_pct"], "note": r.get("note", "")} for r in stress_rows],
            diag={"warnings": []},
        )

        return rg.generate_all(inputs)


# ----------------------------- Scenario helpers -----------------------------

def _historical_stress(px: pd.DataFrame,
                       rets: pd.DataFrame,
                       w: pd.Series,
                       window: str,
                       cash: float) -> Tuple[List[str], List[float], List[float], List[float], Dict[str, List[float]]]:
    """
    Replay a historical subperiod using fixed weights vector w (normalized long-only).
    Returns (dates, equity(₹), drawdown, pnl, var_hist)
    """
    try:
        s_str, e_str = window.split(":")
    except ValueError:
        raise ValueError("historical_window must be 'YYYY-MM-DD:YYYY-MM-DD'")

    sub = rets[(rets.index >= pd.to_datetime(s_str)) & (rets.index <= pd.to_datetime(e_str))].copy()
    if sub.empty:
        raise ValueError("No returns in the specified historical window.")

    # normalize weights
    ww = w.clip(lower=0.0)
    ww = ww / max(1e-12, ww.sum())
    port_ret = sub.dot(ww)  # daily portfolio returns
    eq = (1.0 + port_ret).cumprod() * cash

    dates = [d.strftime("%Y-%m-%d") for d in sub.index]
    equity = eq.tolist()
    dd = _compute_drawdown(equity)
    pnl = port_ret.tolist()
    var_hist = _histogram_pct(port_ret.values, bins=41) # type: ignore
    return dates, equity, dd, pnl, var_hist


# ----------------------------- Loading / weights -----------------------------

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
        df = df.rename(columns={cols.get("date", "date"): "date"})
        df["date"] = pd.to_datetime(df["date"])
        wide = df.set_index("date").sort_index()
    elif has("date", "symbol", "close"):
        df = df.rename(columns={
            cols.get("date", "date"): "date",
            cols.get("symbol", "symbol"): "symbol",
            cols.get("close", "close"): "close",
        })
        df["date"] = pd.to_datetime(df["date"])
        wide = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    else:
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            wide = df.copy().sort_index()
        else:
            raise ValueError("Unrecognized data shape. Expect wide (date + symbols) or long (date,symbol,close).")

    if start: wide = wide[wide.index >= pd.to_datetime(start)]
    if end:   wide = wide[wide.index <= pd.to_datetime(end)]
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
        w = pd.Series(1.0, index=cols)
    else:
        tail = rets_hist.tail(int(vol_lookback))
        vol = tail.std().replace(0.0, np.nan)
        inv = 1.0 / vol
        inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if inv.sum() <= 0:
            w = pd.Series(1.0, index=cols)
        else:
            w = inv
    w = w.clip(lower=0.0)
    s = w.sum()
    return (w / s) if s > 0 else pd.Series(1.0/len(cols), index=cols)


# ----------------------------- Metrics / utils -----------------------------

def _compute_drawdown(equity: List[float]) -> List[float]:
    peak = -float("inf")
    dd = []
    for v in equity:
        peak = max(peak, v)
        dd.append((v / peak - 1.0) if peak > 0 else 0.0)
    return dd

def _basic_kpis(dates: List[str], equity: List[float], pnl: Optional[List[float]]) -> Dict:
    if not equity:
        return {}
    first, last = equity[0], equity[-1]
    years = max(1e-9, len(dates) / 252.0)
    cagr = (last / first) ** (1 / years) - 1 if first > 0 else 0.0
    if pnl:
        mu = float(np.mean(pnl))
        sd = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
        dwn = float(np.std([min(0.0, r) for r in pnl], ddof=1)) if len(pnl) > 1 else 0.0
        vol_annual = sd * math.sqrt(252)
        sharpe = (mu * 252) / (sd + 1e-12)
        sortino = (mu * 252) / (dwn + 1e-12)
    else:
        sharpe = sortino = vol_annual = 0.0
    max_dd = min(_compute_drawdown(equity)) if equity else 0.0
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd, "vol_annual": vol_annual if pnl else 0.0}

def _histogram_pct(x: np.ndarray, bins: int = 41) -> Dict[str, List[float]]:
    x = np.asarray(x)
    if x.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(x * 100.0, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"bins": centers.tolist(), "counts": counts.tolist()}

def _next_trading_date(iso_date: str) -> str:
    d = pd.Timestamp(iso_date)
    while True:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:
            return d.strftime("%Y-%m-%d")


# ----------------------------- CLI -----------------------------

def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Run portfolio stress tests and emit reports.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", required=True, help="CSV/Parquet (wide or long)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--weights", default="ew", choices=["ew","ivol"])
    ap.add_argument("--vol-lookback-days", type=int, default=252)
    ap.add_argument("--cash", type=float, default=1_000_000.0)

    ap.add_argument("--historical-window", default=None, help="YYYY-MM-DD:YYYY-MM-DD")
    ap.add_argument("--equity-drop", type=float, default=None, help="e.g. -0.25")
    ap.add_argument("--vol-spike", type=float, default=None, help="e.g. 2.0 (×sigma)")
    ap.add_argument("--rate-shift-bps", type=float, default=None, help="+50 means +0.50% parallel")
    ap.add_argument("--duration-years", type=float, default=0.0)
    ap.add_argument("--factor-beta", action="append", default=[], help="name=beta (repeatable)")
    ap.add_argument("--factor-shock", action="append", default=[], help="name=shock (repeatable; shock as fraction)")

    ap.add_argument("--out-prefix", default="runs/stress_demo/")
    ap.add_argument("--exporter", default="local", choices=["local","s3","gcs"])
    ap.add_argument("--export-arg", action="append", default=[], help="key=value (e.g., root=artifacts/reports/out)")
    args = ap.parse_args()

    def kv_list_to_dict(items):
        out = {}
        for kv in items:
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    out[k] = float(v)
                except Exception:
                    pass
        return out

    kwargs = {}
    for kv in args.export_arg:
        if "=" in kv:
            k, v = kv.split("=", 1)
            kwargs[k] = v

    spec = StressTestSpec(
        run_id=args.run_id,
        data_path=args.data,
        start=args.start,
        end=args.end,
        weights=args.weights,
        vol_lookback_days=args.vol_lookback_days,
        cash=args.cash,
        historical_window=args.historical_window,
        equity_drop=args.equity_drop,
        vol_spike=args.vol_spike,
        rate_shift_bps=args.rate_shift_bps,
        duration_years=args.duration_years,
        factor_betas=kv_list_to_dict(args.factor_beta),
        factor_shocks=kv_list_to_dict(args.factor_shock),
        output_prefix=args.out_prefix,
        exporter=(args.exporter, kwargs or ({"root":"artifacts/reports/out"} if args.exporter=="local" else {})),
    )
    return spec


if __name__ == "__main__":
    spec = _parse_cli()
    job = StressTestJob(spec)
    urls = job.run()
    import json; print(json.dumps(urls, indent=2))