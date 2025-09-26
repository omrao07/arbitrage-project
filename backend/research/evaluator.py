#!/usr/bin/env python3
"""
evaluator.py — compare and grade backtest runs.

Works with ledgers produced by backtester.py / auto_backtester.py
(expects a CSV with columns at least: Date/DatetimeIndex, Close/ret/net_pnl/equity)

Features
- Load one or many ledgers (files or folders)
- Compute robust metrics (CAGR, Sharpe, Sortino, Calmar, MDD, MAR, Vol, Skew, Kurt, VaR/ES)
- Rolling stats (rolling Sharpe, rolling max drawdown)
- Bootstrap confidence intervals for Sharpe/CAGR (stationary bootstrap optional)
- Turnover & average trade stats if `trade` column is present
- Rank and export summaries to CSV/JSON
- Optional lightweight HTML tear sheet

Usage
  python evaluator.py --inputs results_sma/ledger.csv results_rsi/ledger.csv --outdir eval_out
  python evaluator.py --inputs results_sma results_rsi --outdir eval_out     # will auto-pick ledger.csv
  python evaluator.py --inputs 'runs/*/ledger.csv' --outdir eval_out         # shell glob (quote on Windows)

Outputs in --outdir
  metrics_summary.csv         # one row per run
  metrics_summary.json        # dict keyed by run name
  rolling_stats.csv           # stacked long-form rolling metrics
  ranking.csv                 # sorted by primary metric
  report.html                 # (optional) tear sheet if --html is set
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------
# Utils
# --------------------------
def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date").sort_index()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
    raise ValueError("Ledger must have a DatetimeIndex or a Date/date column.")


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def discover_ledger(path: Path) -> Path:
    """
    Accept either a CSV file or a directory containing a ledger.csv.
    """
    if path.is_file():
        return path
    if path.is_dir():
        cand = _first_existing([path / "ledger.csv", path / "Ledger.csv"])
        if cand:
            return cand
    raise FileNotFoundError(f"Could not find ledger at {path}")


# --------------------------
# Metrics
# --------------------------
@dataclass
class MetricConfig:
    periods_per_year: int = 252
    rf_rate: float = 0.0
    var_alpha: float = 0.05
    es_alpha: float = 0.05
    sortino_mar: float = 0.0  # minimum acceptable return per period


def drawdown_stats(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Return (max_drawdown, peak_date, trough_date)."""
    eq = equity.astype(float).dropna()
    cummax = eq.cummax()
    dd = eq / cummax - 1.0
    mdd = dd.min()
    trough = dd.idxmin()
    peak = (eq.loc[:trough]).idxmax() if pd.notna(trough) else dd.index[0]
    return float(mdd), peak, trough # type: ignore


def basic_moments(x: pd.Series) -> Tuple[float, float]:
    """Return (skew, kurtosis_fisher)."""
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        return (float("nan"), float("nan"))
    mu = x.mean()
    sig = x.std(ddof=0)
    if sig == 0 or np.isnan(sig):
        return (0.0, -3.0)  # zero-variance: skew 0, kurtosis_fisher -3
    z = (x - mu) / sig
    skew = (z**3).mean()
    kurt_fisher = (z**4).mean() - 3.0
    return float(skew), float(kurt_fisher)


def value_at_risk(x: pd.Series, alpha: float = 0.05) -> float:
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        return float("nan")
    return float(np.quantile(x, alpha))


def expected_shortfall(x: pd.Series, alpha: float = 0.05) -> float:
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        return float("nan")
    var = np.quantile(x, alpha)
    tail = x[x <= var]
    if tail.empty:
        return float("nan")
    return float(tail.mean())


def turnover_stats(trade: pd.Series, price: pd.Series) -> Dict[str, float]:
    if trade is None or trade.isna().all():
        return {"TURNOVER": float("nan"), "AVG_ABS_TRADE": float("nan")}
    notional = (trade.abs() * price).fillna(0.0)
    return {
        "TURNOVER": float(notional.sum() / max(price.abs().sum(), 1e-12)),
        "AVG_ABS_TRADE": float(trade.abs().mean()),
    }


def compute_metrics(
    ledger: pd.DataFrame,
    cfg: MetricConfig,
) -> Dict[str, float]:
    """
    Compute performance/risk metrics from ledger with columns:
    - equity (preferred) OR net_pnl to integrate
    - ret OR net_pnl (per-period return as equity fraction)
    - price, pos, trade (optional)
    """
    df = ledger.copy()
    df = _as_dt_index(df)

    # Returns
    if "net_pnl" in df.columns:
        r = df["net_pnl"].astype(float).fillna(0.0)
    elif "ret" in df.columns:
        r = df["ret"].astype(float).fillna(0.0)
    else:
        raise ValueError("Ledger needs 'net_pnl' or 'ret' column.")

    # Equity
    if "equity" in df.columns:
        eq = df["equity"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if eq.empty:
            eq = (1.0 + r).cumprod()
    else:
        eq = (1.0 + r).cumprod()

    # Core metrics
    total_return = float(eq.iloc[-1] - 1.0)
    span_days = max((eq.index[-1] - eq.index[0]).days, 1) # type: ignore
    years = span_days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0

    vol = float(r.std(ddof=0) * math.sqrt(cfg.periods_per_year))
    sharpe = float((r.mean() * cfg.periods_per_year - cfg.rf_rate) / (vol + 1e-12))

    downside = r.copy()
    downside[downside > cfg.sortino_mar] = cfg.sortino_mar
    downside_dev = float(((cfg.sortino_mar - downside).std(ddof=0)) * math.sqrt(cfg.periods_per_year))
    sortino = float((r.mean() * cfg.periods_per_year - cfg.sortino_mar) / (downside_dev + 1e-12))

    mdd, dd_peak, dd_trough = drawdown_stats(eq)
    calmar = float(cagr / abs(mdd)) if mdd != 0 else float("nan")
    mar = float((r.mean() * cfg.periods_per_year) / abs(mdd)) if mdd != 0 else float("nan")

    skew, kurt = basic_moments(r)
    var = value_at_risk(r, alpha=cfg.var_alpha)
    es = expected_shortfall(r, alpha=cfg.es_alpha)

    to_stats = turnover_stats(df.get("trade", pd.Series(index=df.index, dtype=float)), df.get("price", pd.Series(index=df.index, dtype=float)))

    # Hit ratio metrics
    hit_rate = float((r > 0).mean())
    avg_win = float(r[r > 0].mean()) if (r > 0).any() else float("nan")
    avg_loss = float(r[r < 0].mean()) if (r < 0).any() else float("nan")

    return {
        "TOTAL_RETURN": total_return,
        "CAGR": cagr,
        "VOL": vol,
        "SHARPE": sharpe,
        "SORTINO": sortino,
        "MAX_DRAWDOWN": float(mdd),
        "CALMAR": calmar,
        "MAR": mar,
        "SKEW": skew,
        "KURTOSIS_FISHER": kurt,
        f"VaR_{int(cfg.var_alpha*100)}%": var,
        f"ES_{int(cfg.es_alpha*100)}%": es,
        "HIT_RATE": hit_rate,
        "AVG_WIN": avg_win,
        "AVG_LOSS": avg_loss,
        **to_stats,
        "START": str(eq.index[0].date()), # type: ignore
        "END": str(eq.index[-1].date()), # type: ignore
        "N_PERIODS": int(len(eq)),
    }


# --------------------------
# Bootstrap CIs
# --------------------------
def stationary_bootstrap_indices(n: int, p: float) -> np.ndarray:
    """
    Politis & Romano stationary bootstrap: block length geometric with parameter p.
    Returns a 1D array of indices length n.
    """
    idx = np.zeros(n, dtype=int)
    t = np.random.randint(0, n)
    for i in range(n):
        idx[i] = t
        if np.random.rand() < p:
            t = np.random.randint(0, n)
        else:
            t = (t + 1) % n
    return idx


def bootstrap_ci(
    series: pd.Series,
    stat_fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    stationary_p: Optional[float] = 0.1,  # None -> iid bootstrap
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    x = series.dropna().astype(float).values
    if x.size < 5: # type: ignore
        return (float("nan"), float("nan"))
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        if stationary_p is None:
            sample = rng.choice(x, size=x.size, replace=True) # type: ignore
        else:
            idx = stationary_bootstrap_indices(x.size, stationary_p) # type: ignore
            sample = x[idx]
        stats[b] = stat_fn(sample)
    low = (1 - ci) / 2
    high = 1 - low
    return (float(np.quantile(stats, low)), float(np.quantile(stats, high)))


def sharpe_ci(returns: pd.Series, ppy: int, n_boot=2000, ci=0.95, stationary_p=0.1) -> Tuple[float, float]:
    def _stat(s):
        s = np.asarray(s)
        mu = s.mean() * ppy
        vol = s.std(ddof=0) * math.sqrt(ppy)
        return 0.0 if vol == 0 else mu / vol
    return bootstrap_ci(returns, _stat, n_boot=n_boot, ci=ci, stationary_p=stationary_p)


def cagr_ci(returns: pd.Series, ppy: int, n_boot=2000, ci=0.95, stationary_p=0.1) -> Tuple[float, float]:
    def _stat(s):
        eq = np.cumprod(1 + np.asarray(s))
        years = max(len(eq) / ppy, 1e-9)
        last = eq[-1] if len(eq) else 1.0
        return (last ** (1 / years) - 1.0) if last > 0 else -1.0
    return bootstrap_ci(returns, _stat, n_boot=n_boot, ci=ci, stationary_p=stationary_p)


# --------------------------
# Rolling stats
# --------------------------
def rolling_stats(ledger: pd.DataFrame, window: int, ppy: int) -> pd.DataFrame:
    df = _as_dt_index(ledger.copy())
    r = df["net_pnl"] if "net_pnl" in df.columns else df["ret"]
    r = r.astype(float).fillna(0.0)

    roll_vol = r.rolling(window).std(ddof=0) * math.sqrt(ppy)
    roll_mean = r.rolling(window).mean() * ppy
    roll_sharpe = roll_mean / (roll_vol + 1e-12)

    eq = df["equity"] if "equity" in df.columns else (1 + r).cumprod()
    peak = eq.cummax()
    roll_dd = eq / peak - 1.0

    out = pd.DataFrame(
        {
            "rolling_vol": roll_vol,
            "rolling_mean": roll_mean,
            "rolling_sharpe": roll_sharpe,
            "rolling_drawdown": roll_dd,
        }
    )
    return out


# --------------------------
# I/O + Orchestration
# --------------------------
def load_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _as_dt_index(df)


def name_from_path(path: Path) -> str:
    if path.name.lower().endswith(".csv"):
        return path.parent.name or path.stem
    return path.name


def evaluate_runs(
    inputs: List[str],
    outdir: Path,
    cfg: MetricConfig,
    primary_metric: str = "SHARPE",
    ci_bootstrap: bool = True,
    ci_level: float = 0.95,
    stationary_p: Optional[float] = 0.1,
    rolling_window: int = 126,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, float]] = []
    roll_frames: List[pd.DataFrame] = []
    json_blob: Dict[str, Dict[str, float]] = {}

    # Expand globs and discover ledgers
    expanded: List[Path] = []
    for i in inputs:
        matches = list(Path().glob(i))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(Path(i))

    for raw in expanded:
        try:
            ledger_path = discover_ledger(raw)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        run_name = name_from_path(ledger_path)
        ledger = load_ledger(ledger_path)

        # Metrics
        m = compute_metrics(ledger, cfg)

        # Bootstrap CIs
        r = ledger["net_pnl"] if "net_pnl" in ledger.columns else ledger["ret"]
        if ci_bootstrap:
            lo_s, hi_s = sharpe_ci(r, cfg.periods_per_year, ci=ci_level, stationary_p=stationary_p) # type: ignore
            lo_c, hi_c = cagr_ci(r, cfg.periods_per_year, ci=ci_level, stationary_p=stationary_p) # type: ignore
            m.update({
                f"SHARPE_CI_{int(ci_level*100)}%_LO": lo_s,
                f"SHARPE_CI_{int(ci_level*100)}%_HI": hi_s,
                f"CAGR_CI_{int(ci_level*100)}%_LO": lo_c,
                f"CAGR_CI_{int(ci_level*100)}%_HI": hi_c,
            })

        record = {"RUN": run_name, **m}
        records.append(record)
        json_blob[run_name] = m

        # Rolling
        rs = rolling_stats(ledger, window=rolling_window, ppy=cfg.periods_per_year)
        rs["RUN"] = run_name
        roll_frames.append(rs)

    if not records:
        raise SystemExit("No valid ledgers found to evaluate.")

    # Summaries
    summary_df = pd.DataFrame(records).set_index("RUN").sort_index()
    ranking = summary_df.sort_values(primary_metric, ascending=False)

    # Save
    summary_csv = outdir / "metrics_summary.csv"
    summary_json = outdir / "metrics_summary.json"
    ranking_csv = outdir / "ranking.csv"
    rolling_csv = outdir / "rolling_stats.csv"

    summary_df.to_csv(summary_csv)
    ranking.to_csv(ranking_csv)
    with open(summary_json, "w") as f:
        json.dump(json_blob, f, indent=2)

    if roll_frames:
        rolled = pd.concat(roll_frames, axis=0)
        rolled.to_csv(rolling_csv)

    print(f"[OK] Saved:\n- {summary_csv}\n- {ranking_csv}\n- {summary_json}\n- {rolling_csv if roll_frames else '(no rolling file)'}")


def maybe_make_html(outdir: Path, title: str = "Backtest Evaluation Report") -> None:
    """
    Very light HTML to quickly inspect CSVs (no external deps).
    """
    summary_csv = outdir / "metrics_summary.csv"
    ranking_csv = outdir / "ranking.csv"
    rolling_csv = outdir / "rolling_stats.csv"

    def table_from_csv(path: Path) -> str:
        if not path.exists():
            return "<p>(missing)</p>"
        df = pd.read_csv(path)
        return df.to_html(index=False, border=0, classes="table", float_format=lambda x: f"{x:,.6f}")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; padding:20px; line-height:1.4}}
h1,h2{{margin:0.4rem 0}}
.table{{border-collapse:collapse; width:100%}}
.table th,.table td{{border-bottom:1px solid #ddd; padding:6px 8px; text-align:right}}
.table th:first-child,.table td:first-child{{text-align:left}}
small{{color:#666}}
</style>
</head>
<body>
<h1>{title}</h1>
<p><small>Generated by evaluator.py</small></p>

<h2>Ranking</h2>
{table_from_csv(ranking_csv)}

<h2>Metrics Summary</h2>
{table_from_csv(summary_csv)}

<h2>Rolling Stats (sample)</h2>
{table_from_csv(rolling_csv)}

</body>
</html>
"""
    (outdir / "report.html").write_text(html, encoding="utf-8")
    print(f"[OK] Saved: {outdir / 'report.html'}")


# --------------------------
# CLI
# --------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and compare backtest ledgers.")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of ledger.csv files, directories (containing ledger.csv), or glob patterns.",
    )
    p.add_argument("--outdir", default="eval_out", help="Output directory.")
    p.add_argument("--ppy", type=int, default=252, help="Periods per year for annualization (default 252).")
    p.add_argument("--rf", type=float, default=0.0, help="Risk-free rate (annualized).")
    p.add_argument("--var", type=float, default=0.05, help="Value-at-Risk alpha (default 0.05).")
    p.add_argument("--es", type=float, default=0.05, help="Expected Shortfall alpha (default 0.05).")
    p.add_argument("--mar", type=float, default=0.0, help="Minimum acceptable return per period for Sortino.")
    p.add_argument("--metric", default="SHARPE", help="Primary metric for ranking (e.g., SHARPE, CAGR, CALMAR).")
    p.add_argument("--no-ci", action="store_true", help="Disable bootstrap confidence intervals.")
    p.add_argument("--ci", type=float, default=0.95, help="CI level (default 0.95).")
    p.add_argument(
        "--iid-bootstrap",
        action="store_true",
        help="Use IID bootstrap instead of stationary bootstrap (default is stationary p=0.1).",
    )
    p.add_argument("--stationary-p", type=float, default=0.1, help="Stationary bootstrap p (ignored if --iid-bootstrap).")
    p.add_argument("--roll-window", type=int, default=126, help="Rolling window for sharpe/DD (default 126).")
    p.add_argument("--html", action="store_true", help="Emit a simple HTML tear sheet.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    cfg = MetricConfig(
        periods_per_year=args.ppy,
        rf_rate=args.rf,
        var_alpha=args.var,
        es_alpha=args.es,
        sortino_mar=args.mar,
    )

    evaluate_runs(
        inputs=args.inputs,
        outdir=outdir,
        cfg=cfg,
        primary_metric=args.metric.upper(),
        ci_bootstrap=not args.no_ci,
        ci_level=args.ci,
        stationary_p=None if args.iid_bootstrap else args.stationary_p,
        rolling_window=args.roll_window,
    )

    if args.html:
        maybe_make_html(outdir)


if __name__ == "__main__":
    main()
