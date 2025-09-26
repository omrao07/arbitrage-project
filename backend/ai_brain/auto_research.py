# ai_brain/auto_research.py
"""
AutoResearch: autonomous research orchestrator
- Plans experiments (optionally via ai_brain.strategy_generator)
- Runs vectorized backtests in parallel (you provide a backtest callable)
- Scores results (Sharpe, drawdown, hit rate, turnover)
- Allocates compute to the highest expected alpha-per-compute
- Writes tamper-evident audit entries (backend/audit/merkle_ledger.py)
- Emits promotion events via a user-supplied callback

Dependencies (runtime):
  - numpy, pandas
  - (optional) your backend.audit.merkle_ledger.MerkleLedger

Design notes:
  • Backtests are user-supplied callables so this file doesn’t hard-couple to your framework.
  • Compute budgeting enforces a max concurrent workers + total core-seconds.
  • Results are persisted to a compact JSONL “run log” for reproducibility.

Example:
    from ai_brain.auto_research import AutoResearch, BacktestSpec

    def my_backtest(spec: BacktestSpec) -> dict:
        # run your framework here, but return a dict with equity curve & trades
        import numpy as np, pandas as pd
        idx = pd.date_range("2022-01-01", periods=250, freq="B")
        equity = pd.Series(1_000_000 * (1 + 0.0005*np.arange(250)), index=idx)
        trades = pd.DataFrame({"ts": idx, "notional": 1e5, "side": 1})
        return {"equity": equity, "trades": trades, "meta": {"spec": spec.as_dict()}}

    ar = AutoResearch(
        backtest_fn=my_backtest,
        run_dir=".runs/auto_research",
        max_workers=4,
        compute_budget_core_seconds=3600,
    )
    plan = [
        BacktestSpec(region="US", name="labor_union_power", params={"lb": 120}),
        BacktestSpec(region="EU", name="north_sea_gas_infra", params={"lb": 180}),
    ]
    results = ar.run(plan)
    top = ar.top_k(results, k=2)
    for r in top:
        print(r.id, r.metrics.sharpe, r.meta["alpha_per_compute"])
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------- Types & Specs ----------

@dataclass(frozen=True)
class BacktestSpec:
    region: str
    name: str
    params: Dict[str, Any]
    id_suffix: Optional[str] = None  # to differentiate hyperparams

    @property
    def id(self) -> str:
        suf = f":{self.id_suffix}" if self.id_suffix else ""
        return f"{self.region}.{self.name}{suf}"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        return d


@dataclass
class Metrics:
    sharpe: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    hit_rate: float
    turnover_pa: float
    calmar: float
    sortino: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    id: str
    spec: BacktestSpec
    metrics: Metrics
    equity: pd.Series
    trades: pd.DataFrame
    wall_time_sec: float
    cpu_time_sec: float
    meta: Dict[str, Any]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metrics": self.metrics.as_dict(),
            "wall_time_sec": round(self.wall_time_sec, 3),
            "cpu_time_sec": round(self.cpu_time_sec, 3),
            "alpha_per_compute": self.meta.get("alpha_per_compute"),
            "region": self.spec.region,
            "name": self.spec.name,
            "params": self.spec.params,
        }


# ---------- Metric utilities ----------

def _safe_ann_factor(index: pd.DatetimeIndex) -> float:
    # business days per year if possible; fallback to 252
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        days = (index[-1] - index[0]).days
        if days > 0:
            return 365.0 / (days / max(1, len(index)))
    return 252.0

def equity_metrics(equity: pd.Series, trades: Optional[pd.DataFrame] = None) -> Metrics:
    equity = equity.sort_index()
    ret = equity.pct_change().dropna()
    if ret.empty:
        raise ValueError("Equity series too short for metrics.")
    ann = _safe_ann_factor(equity.index) # type: ignore
    mu = ret.mean() * ann
    vol = ret.std(ddof=1) * math.sqrt(ann)
    sharpe = (mu / vol) if vol > 0 else 0.0

    # drawdown
    roll_max = equity.cummax()
    dd = (equity / roll_max - 1.0).min()
    max_dd = abs(float(dd))

    # sortino (downside deviation)
    neg = ret.clip(upper=0.0)
    down = neg.std(ddof=1) * math.sqrt(ann)
    sortino = (mu / down) if down > 0 else 0.0

    # calmar
    calmar = (mu / max_dd) if max_dd > 1e-9 else float("inf")

    # hit rate from trades PnL if given; else from returns
    if trades is not None and {"pnl"}.issubset(set(trades.columns)):
        wins = (trades["pnl"] > 0).sum()
        tot = len(trades)
    else:
        wins = (ret > 0).sum()
        tot = len(ret)
    hit = float(wins) / max(1, tot)

    # turnover per annum (approx): sum(|notional|)/avg_equity / years
    turnover_pa = 0.0
    if trades is not None and {"notional"}.issubset(set(trades.columns)):
        notional = trades["notional"].abs().sum()
        avg_eq = float(equity.mean())
        years = max(1e-9, (equity.index[-1] - equity.index[0]).days / 365.0) if isinstance(equity.index, pd.DatetimeIndex) else 1.0
        turnover_pa = (notional / max(1e-9, avg_eq)) / years

    return Metrics(
        sharpe=float(sharpe),
        ann_return=float(mu),
        ann_vol=float(vol),
        max_drawdown=float(max_dd),
        hit_rate=float(hit),
        turnover_pa=float(turnover_pa),
        calmar=float(calmar),
        sortino=float(sortino),
    )


# ---------- AutoResearch Engine ----------

class AutoResearch:
    """
    Orchestrates strategy research with compute budgeting and immutable audit.

    backtest_fn: Callable[[BacktestSpec], dict]
        Must return: {"equity": pd.Series, "trades": pd.DataFrame (optional), "meta": dict}
    """

    def __init__(
        self,
        backtest_fn: Callable[[BacktestSpec], Dict[str, Any]],
        run_dir: str,
        max_workers: int = 4,
        compute_budget_core_seconds: Optional[int] = None,
        ledger_path: Optional[str] = None,
        on_promote: Optional[Callable[[BacktestResult], None]] = None,
    ) -> None:
        self.backtest_fn = backtest_fn
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(1, int(max_workers))
        self.compute_budget_core_seconds = compute_budget_core_seconds  # None = unlimited
        self.on_promote = on_promote

        self.ledger = None
        if ledger_path:
            self.ledger = self._init_ledger(ledger_path)

        # persistent JSONL logs
        self.results_log = self.run_dir / "results.jsonl"
        self.audit_log = self.run_dir / "audit.jsonl"

    # ---- Ledger helpers ----
    def _init_ledger(self, ledger_path: str):
        try:
            from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        except Exception as e:
            raise RuntimeError("MerkleLedger not found. Ensure backend/audit/merkle_ledger.py exists.") from e
        return MerkleLedger(ledger_path)

    def _audit(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["ts"] = int(time.time() * 1000)
        if self.ledger:
            self.ledger.append(payload)
        # also append to human-readable JSONL
        with open(self.audit_log, "a") as f:
            f.write(json.dumps(payload, separators=(",", ":"), default=self._json_default) + "\n")

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, (pd.Series,)):
            return obj.to_dict()
        if isinstance(obj, (pd.DataFrame,)):
            return obj.to_dict(orient="records")
        return str(obj)

    # ---- Planning ----
    def plan_from_generator(self, max_candidates: int = 50) -> List[BacktestSpec]:
        """
        Optional: use ai_brain.strategy_generator to propose specs.
        Falls back to empty list if generator is unavailable.
        """
        try:
            from ai_brain.strategy_generator import propose  # type: ignore
        except Exception:
            return []
        specs = propose(limit=max_candidates)
        # coerce into BacktestSpec
        out: List[BacktestSpec] = []
        for s in specs:
            out.append(BacktestSpec(region=s["region"], name=s["name"], params=s.get("params", {}), id_suffix=s.get("id_suffix")))
        return out

    # ---- Execution ----
    def run(self, plan: Sequence[BacktestSpec]) -> List[BacktestResult]:
        """
        Execute a batch of backtests subject to compute budget.
        """
        if not plan:
            return []

        self._audit({"type": "run_start", "count": len(plan)})

        # compute budget tracking
        budget = self.compute_budget_core_seconds
        used_core_sec = 0.0
        results: List[BacktestResult] = []

        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_to_spec = {ex.submit(self._run_one, spec): spec for spec in plan}
            for fut in cf.as_completed(fut_to_spec):
                spec = fut_to_spec[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    self._audit({"type": "run_error", "id": spec.id, "err": str(e)})
                    continue
                results.append(res)

                # persist compact result summary
                with open(self.results_log, "a") as f:
                    f.write(json.dumps(res.to_summary(), separators=(",", ":"), default=self._json_default) + "\n")

                used_core_sec += max(1e-6, res.cpu_time_sec)  # approximate
                if budget is not None and used_core_sec >= budget:
                    self._audit({"type": "budget_exhausted", "used_core_sec": used_core_sec})
                    break

        self._audit({"type": "run_end", "produced": len(results), "used_core_sec": used_core_sec})
        return results

    def _run_one(self, spec: BacktestSpec) -> BacktestResult:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        raw = self.backtest_fn(spec)
        eq = raw.get("equity")
        if not isinstance(eq, pd.Series):
            raise TypeError(f"{spec.id}: backtest_fn must return 'equity' as pd.Series.")
        trades = raw.get("trades")
        if trades is not None and not isinstance(trades, pd.DataFrame):
            raise TypeError(f"{spec.id}: 'trades' must be a pandas DataFrame or None.")

        metrics = equity_metrics(eq, trades)
        wall = time.perf_counter() - start_wall
        cpu = time.process_time() - start_cpu
        apc = self._alpha_per_compute(metrics, cpu)

        meta = dict(raw.get("meta", {}))
        meta.update({"alpha_per_compute": apc})

        res = BacktestResult(
            id=spec.id,
            spec=spec,
            metrics=metrics,
            equity=eq,
            trades=(trades if trades is not None else pd.DataFrame()),
            wall_time_sec=float(wall),
            cpu_time_sec=float(cpu),
            meta=meta,
        )

        self._audit({
            "type": "result",
            "id": res.id,
            "metrics": res.metrics.as_dict(),
            "cpu": res.cpu_time_sec,
            "wall": res.wall_time_sec,
            "alpha_per_compute": apc,
        })
        return res

    @staticmethod
    def _alpha_per_compute(m: Metrics, cpu_sec: float) -> float:
        """
        Alpha-per-compute: risk-adjusted return per CPU second (higher is better).
        """
        risk_adj = max(0.0, m.ann_return) * (1 + 0.5 * max(0.0, m.sharpe))  # reward higher Sharpe
        denom = max(1e-6, cpu_sec)
        return float(risk_adj / denom)

    # ---- Selection & Promotion ----
    @staticmethod
    def top_k(results: Sequence[BacktestResult], k: int = 10, min_sharpe: float = 0.0) -> List[BacktestResult]:
        eligible = [r for r in results if r.metrics.sharpe >= min_sharpe]
        return sorted(eligible, key=lambda r: r.meta.get("alpha_per_compute", 0.0), reverse=True)[:k]

    def promote(self, results: Sequence[BacktestResult], k: int = 5, min_sharpe: float = 0.5) -> List[BacktestResult]:
        """
        Mark top-k candidates for paper/live; call user callback and audit.
        """
        chosen = self.top_k(results, k=k, min_sharpe=min_sharpe)
        for r in chosen:
            self._audit({"type": "promote", "id": r.id, "metrics": r.metrics.as_dict(), "alpha_per_compute": r.meta.get("alpha_per_compute")})
            if self.on_promote:
                try:
                    self.on_promote(r)
                except Exception as e:
                    self._audit({"type": "promote_error", "id": r.id, "err": str(e)})
        return chosen

    # ---- Repro & Utilities ----
    def load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def verify_ledger(self) -> bool:
        if not self.ledger:
            return False
        return bool(self.ledger.verify())