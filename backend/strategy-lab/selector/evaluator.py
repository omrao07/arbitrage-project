# selector/evaluator.py
"""
Evaluator for A/B Strategy Tests
--------------------------------
Purpose
- Sample ABTestRunner + (optional) MonitoringAgent, aggregate per-arm cashflows,
  estimate contribution returns, and compute metrics: Sharpe, MDD, CAGR, hit rate, etc.
- Stream results to JSONL/CSV for dashboards.

Inputs
- ABTestRunner (from selector/ab_tests.py) -> provides per-arm realized cashflow (approx),
  trades, win_rate, plus portfolio equity/exposure.
- MonitoringAgent (optional) -> provides portfolio-level realized/unrealized PnL for sanity.

Outputs
- Rolling snapshots (JSONL/CSV) with:
    t, equity, gross_exposure, leverage,
    per-arm: realized_cashflow_delta, contrib_return, trades, win_rate
- Final metrics via `summary()`.

Notes
- Per-arm PnL in AB runner is an *approximation* using cash flows. If you need
  exact realized PnL per arm, extend AttributionRecorder to track inventory by arm.

No external dependencies.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------- small utils ------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _pct_change(prev: float, cur: float) -> float:
    if prev is None or prev == 0:
        return 0.0
    return (cur - prev) / prev

def _sharpe(returns: List[float], rf: float = 0.0, ann: int = 252) -> float:
    if not returns:
        return 0.0
    mu = statistics.mean(returns) - (rf / ann)
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return _safe_div(mu, sd) * math.sqrt(ann)

def _sortino(returns: List[float], rf: float = 0.0, ann: int = 252) -> float:
    if not returns:
        return 0.0
    excess = [r - (rf / ann) for r in returns]
    downside = [min(0.0, r) for r in excess]
    dd = math.sqrt(sum(d * d for d in downside) / max(1, len(downside)))
    return _safe_div(statistics.mean(excess), dd) * math.sqrt(ann) if dd else 0.0

def _max_drawdown(curve: List[float]) -> Tuple[float, int, int]:
    peak = -float("inf")
    mdd = 0.0
    start = end = 0
    for i, v in enumerate(curve):
        if v > peak:
            peak = v
            start = i
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
            end = i
    return mdd, start, end

def _cagr_from_curve(curve: List[float], bars_per_year: int = 252) -> float:
    if not curve or curve[0] <= 0 or curve[-1] <= 0:
        return 0.0
    years = max(1e-9, (len(curve) - 1) / bars_per_year)
    return (curve[-1] / curve[0]) ** (1 / years) - 1.0

def _now() -> float:
    return time.time()


# ----------------------------- config models ----------------------------------

@dataclass
class EvalConfig:
    sample_every_sec: float = 60.0           # cadence to sample runner.snapshot()
    output_path: str = "./logs/ab_eval.jsonl"
    output_format: str = "jsonl"             # jsonl | csv
    bars_per_year: int = 252                 # used for Sharpe/CAGR on contribution series
    rf_rate: float = 0.0                     # annualized risk-free for metrics
    write_headers_csv: bool = True           # for CSV output
    keep_in_memory: bool = True              # also store samples in RAM for summary()
    # When computing per-arm "contribution returns", scale cashflow delta by current equity:
    contrib_scale: str = "equity"            # "equity" or "gross" (use equity unless you have a reason)


# ----------------------------- evaluator --------------------------------------

class Evaluator:
    """
    Live/offline evaluator for A/B tests.
    - Call tick() periodically (or run start_loop()) to record samples.
    - Call summary() to compute metrics so far.
    - Use static load_and_summarize() to analyze a past JSONL/CSV.
    """
    def __init__(self, runner: object, monitoring: Optional[object] = None, cfg: Optional[EvalConfig] = None):
        """
        runner: ABTestRunner (must provide .snapshot())
        monitoring: MonitoringAgent (optional, must provide .snapshot())
        """
        self.runner = runner
        self.mon = monitoring
        self.cfg = cfg or EvalConfig()
        self._last_arm_cash: Dict[str, float] = {}
        self._equity_curve: List[float] = []
        self._arm_ret_series: Dict[str, List[float]] = {}  # per-arm contribution returns
        self._rows_mem: List[Dict] = []
        self._csv_header_written = False

        # ensure path exists
        d = os.path.dirname(self.cfg.output_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # ------------------- live sampling -------------------

    def tick(self) -> Optional[Dict]:
        """
        Take a sample from runner (+monitor), compute per-arm deltas/returns,
        append to file/memory, and return the row.
        """
        snap = {}
        try:
            snap = self.runner.snapshot() # type: ignore
        except Exception:
            return None

        ts = float(snap.get("t", _now()))
        equity = float(snap.get("equity") or 0.0)
        gross = float(snap.get("gross_exposure") or 0.0)
        lev = float(snap.get("leverage") or 0.0)
        arms = snap.get("arms", {}) or {}

        # optional monitoring info
        mon_snap = {}
        if self.mon:
            try:
                mon_snap = self.mon.snapshot() # type: ignore
            except Exception:
                mon_snap = {}

        row: Dict[str, float] = {
            "t": ts,
            "equity": equity,
            "gross_exposure": gross,
            "leverage": lev,
            "pnl_realized": float(mon_snap.get("pnl_realized")) if "pnl_realized" in mon_snap else None,  # may be None # type: ignore
            "pnl_unrealized": float(mon_snap.get("pnl_unrealized")) if "pnl_unrealized" in mon_snap else None, # type: ignore
        }

        # compute equity curve (portfolio level)
        if equity > 0:
            self._equity_curve.append(equity)

        scale_base = equity if self.cfg.contrib_scale == "equity" else (gross or equity or 1.0)

        # Per-arm deltas and contribution returns
        for arm, info in arms.items():
            cash_now = float(info.get("realized_cashflow") or 0.0)
            cash_prev = self._last_arm_cash.get(arm, cash_now)
            delta = cash_now - cash_prev
            self._last_arm_cash[arm] = cash_now

            # contribution return ~ cash_delta / scale_base (small-signal approx)
            contrib_ret = _safe_div(delta, scale_base, 0.0)
            self._arm_ret_series.setdefault(arm, []).append(contrib_ret)

            row[f"{arm}_cashflow_delta"] = delta
            row[f"{arm}_contrib_ret"] = contrib_ret
            row[f"{arm}_trades"] = int(info.get("trades") or 0)
            wr = info.get("win_rate")
            row[f"{arm}_win_rate"] = float(wr) if wr is not None else None # type: ignore

        # Write out
        self._write_row(row)
        if self.cfg.keep_in_memory:
            self._rows_mem.append(row)
        return row

    def start_loop(self, runtime_sec: Optional[float] = None) -> None:
        """
        Blocking loop: sample every cfg.sample_every_sec until runtime_sec (if provided).
        Use in a sidecar process/thread.
        """
        t0 = _now()
        interval = max(0.2, float(self.cfg.sample_every_sec))
        while True:
            self.tick()
            if runtime_sec is not None and _now() - t0 >= runtime_sec:
                break
            time.sleep(interval)

    # ------------------- metrics & summary -------------------

    def summary(self) -> Dict:
        """
        Compute metrics from in-memory rows (or return empty if memory disabled).
        """
        if not self.cfg.keep_in_memory or not self._rows_mem:
            return {"note": "no in-memory rows. Enable keep_in_memory or use load_and_summarize()."}

        out = {
            "t": _now(),
            "portfolio": self._portfolio_metrics(),
            "arms": {},
        }
        for arm, series in self._arm_ret_series.items():
            out["arms"][arm] = self._series_metrics(series)
        return out

    # ------------------- offline analysis -------------------

    @staticmethod
    def load_and_summarize(path: str, bars_per_year: int = 252, rf: float = 0.0) -> Dict:
        """
        Load JSONL/CSV produced by this evaluator and compute the same summary.
        """
        rows = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    # cast floats where possible
                    rows.append({k: (float(v) if isinstance(v, str) and v not in ("", "None") else v) for k, v in row.items()})

        # rebuild series
        eq_curve: List[float] = [float(r["equity"]) for r in rows if r.get("equity") not in (None, "None", "")]
        arms = set()
        for k in rows[0].keys():
            if k.endswith("_contrib_ret"):
                arms.add(k.replace("_contrib_ret", ""))

        arm_series: Dict[str, List[float]] = {a: [] for a in arms}
        for r in rows:
            for a in arms:
                v = r.get(f"{a}_contrib_ret")
                if v not in (None, "None", ""):
                    arm_series[a].append(float(v))

        def _series_metrics_local(series: List[float]) -> Dict:
            sr = _sharpe(series, rf, bars_per_year)
            so = _sortino(series, rf, bars_per_year)
            avg = statistics.mean(series) if series else 0.0
            vol = statistics.pstdev(series) if len(series) > 1 else 0.0
            return {"sharpe": sr, "sortino": so, "avg": avg, "vol": vol}

        mdd, s, e = _max_drawdown(eq_curve) if eq_curve else (0.0, 0, 0)
        cagr = _cagr_from_curve(eq_curve, bars_per_year) if eq_curve else 0.0

        summary = {"portfolio": {"mdd": mdd, "mdd_start": s, "mdd_end": e, "cagr": cagr}, "arms": {}}
        for a, ser in arm_series.items():
            summary["arms"][a] = _series_metrics_local(ser)
        return summary

    # ------------------- internals -------------------

    def _portfolio_metrics(self) -> Dict:
        curve = self._equity_curve
        mdd, s, e = _max_drawdown(curve) if curve else (0.0, 0, 0)
        cagr = _cagr_from_curve(curve, self.cfg.bars_per_year) if curve else 0.0
        ret_series = []
        for i in range(1, len(curve)):
            ret_series.append(_pct_change(curve[i - 1], curve[i]))
        sr = _sharpe(ret_series, self.cfg.rf_rate, self.cfg.bars_per_year) if ret_series else 0.0
        so = _sortino(ret_series, self.cfg.rf_rate, self.cfg.bars_per_year) if ret_series else 0.0
        vol = statistics.pstdev(ret_series) if len(ret_series) > 1 else 0.0
        return {
            "sharpe": sr,
            "sortino": so,
            "vol": vol,
            "mdd": mdd,
            "mdd_start": s,
            "mdd_end": e,
            "cagr": cagr,
        }

    def _series_metrics(self, series: List[float]) -> Dict:
        sr = _sharpe(series, self.cfg.rf_rate, self.cfg.bars_per_year)
        so = _sortino(series, self.cfg.rf_rate, self.cfg.bars_per_year)
        avg = statistics.mean(series) if series else 0.0
        vol = statistics.pstdev(series) if len(series) > 1 else 0.0
        return {"sharpe": sr, "sortino": so, "avg": avg, "vol": vol, "n": len(series)}

    def _write_row(self, row: Dict) -> None:
        path = self.cfg.output_path
        if self.cfg.output_format.lower() == "jsonl":
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        else:
            # CSV
            fieldnames = list(row.keys())
            write_header = (not self._csv_header_written) and (not os.path.exists(path) or os.path.getsize(path) == 0)
            with open(path, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header and self.cfg.write_headers_csv:
                    w.writeheader()
                    self._csv_header_written = True
                w.writerow(row)


# ----------------------------- CLI smoke test ---------------------------------

if __name__ == "__main__":
    # Minimal stub runner to prove it runs
    class StubRunner:
        def __init__(self):
            self.t = _now()
            self.eq = 1_000_000.0
            self.g = 0.0
            self.l = 0.0
            self.cfA = 0.0
            self.cfB = 0.0
            self.trA = self.trB = 0

        def snapshot(self):
            # Simulate random walk equity and trickle cashflow per arm
            import random
            self.t += 60
            r = random.uniform(-0.001, 0.001)
            self.eq *= (1 + r)
            self.cfA += random.uniform(-50, 80)
            self.cfB += random.uniform(-60, 70)
            self.trA += random.randint(0, 3)
            self.trB += random.randint(0, 3)
            return {
                "t": self.t,
                "equity": self.eq,
                "gross_exposure": self.g,
                "leverage": self.l,
                "arms": {
                    "A": {"realized_cashflow": self.cfA, "trades": self.trA, "win_rate": 0.55},
                    "B": {"realized_cashflow": self.cfB, "trades": self.trB, "win_rate": 0.52},
                },
            }

    runner = StubRunner()
    ev = Evaluator(runner, monitoring=None, cfg=EvalConfig(sample_every_sec=0.1, output_path="./logs/ab_eval.jsonl"))
    # Collect a few samples
    for _ in range(50):
        ev.tick()
        time.sleep(0.05)

    print(json.dumps(ev.summary(), separators=(",", ":"), default=float))
    # Offline re-load
    summ = Evaluator.load_and_summarize("./logs/ab_eval.jsonl")
    print(json.dumps(summ, separators=(",", ":"), default=float))