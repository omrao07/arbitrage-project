"""
The gauntlet pipeline — one command that takes strategies from the registry,
runs them, scores them, and records the verdict.

    python -m backend.validation.pipeline --self-test
    python -m backend.validation.pipeline --prices data/nifty_daily.csv --commit

DESIGN RULE THAT MATTERS MOST
-----------------------------
`DataSource.is_real` gates whether results may touch the registry. A run on
synthetic data is a PLUMBING TEST: it proves the machine executes, and it is
structurally forbidden from promoting anything or writing a DSR/PBO verdict.
Without that interlock, the very first convenience shortcut anyone takes is to
"just run it on generated data to see it work", and three weeks later a capital
decision is resting on noise nobody remembers generating.

This is the same rule as the FeedGuard and the risk engine, applied to the
validation layer: real data, or an honest refusal.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.validation.adapter import StrategyAdapter, StrategyShape, detect_shape
from backend.validation.dsr import DSR_GATE, DSRError, evaluate_trial_family
from backend.validation.metrics import InsufficientSampleError, compute_all
from backend.validation.pbo import PBO_GATE, PBOError, pbo_from_trials
from backend.validation.registry import RegistryError, StrategyRegistry

log = logging.getLogger(__name__)


# ── Data sources ────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    """
    A price series plus its provenance.

    `is_real` is not decoration. Results derived from a source where it is
    False may not be committed to the registry.
    """

    name: str
    prices: np.ndarray
    is_real: bool
    description: str = ""

    def __post_init__(self) -> None:
        self.prices = np.asarray(self.prices, dtype=float).ravel()
        if self.prices.size < 2:
            raise ValueError(f"{self.name}: need >= 2 prices, got {self.prices.size}")
        if not np.all(np.isfinite(self.prices)) or np.any(self.prices <= 0):
            raise ValueError(f"{self.name}: prices must be finite and positive")


def synthetic_source(n: int = 800, seed: int = 42) -> DataSource:
    """
    A geometric random walk for testing the PLUMBING ONLY.

    Any strategy scored against this is being scored against noise. That is the
    point — it verifies the machine runs — and it is exactly why `is_real` is
    False and the registry stays untouched.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0002, 0.012, n)
    return DataSource(
        name=f"synthetic-gbm-seed{seed}",
        prices=100.0 * np.cumprod(1.0 + steps),
        is_real=False,
        description="geometric random walk — plumbing test only, NOT a verdict",
    )


def csv_source(path: Path | str, column: str = "close") -> DataSource:
    """Load a real price series from CSV. Fails loudly on anything unusable."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"price file not found: {p}")

    df = pd.read_csv(p)
    col = column if column in df.columns else None
    if col is None:
        for cand in ("close", "Close", "c", "adj_close", "Adj Close", "price"):
            if cand in df.columns:
                col = cand
                break
    if col is None:
        raise ValueError(f"{p}: no price column found; have {list(df.columns)[:10]}")

    prices = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if prices.size < 100:
        raise ValueError(
            f"{p}: only {prices.size} usable rows in column {col!r}. A gauntlet run "
            "on this would produce a verdict no one should act on — get real history first."
        )
    return DataSource(name=str(p), prices=prices, is_real=True, description=f"column={col}")


# ── Results ─────────────────────────────────────────────────────────────────

@dataclass
class StrategyRun:
    strategy_id: str
    ran: bool
    reason: str = ""
    returns: Optional[np.ndarray] = field(default=None, repr=False)
    metrics: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "ran": self.ran,
            "reason": self.reason,
            "metrics": self.metrics,
        }


@dataclass
class GauntletReport:
    data_source: str
    is_real_data: bool
    n_registered: int
    n_attempted: int
    n_ran: int
    n_failed: int
    dsr_passed: List[str] = field(default_factory=list)
    pbo: Optional[float] = None
    pbo_passed: Optional[bool] = None
    committed: bool = False
    runs: List[StrategyRun] = field(default_factory=list)
    failures: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "GAUNTLET REPORT",
            "=" * 68,
            f"  data source     : {self.data_source}",
            f"  real data       : {self.is_real_data}",
            f"  registered      : {self.n_registered}",
            f"  attempted       : {self.n_attempted}",
            f"  ran             : {self.n_ran}",
            f"  failed to run   : {self.n_failed}",
        ]
        if self.pbo is not None:
            lines.append(f"  PBO             : {self.pbo:.4f} (gate < {PBO_GATE}) -> "
                         f"{'PASS' if self.pbo_passed else 'FAIL'}")
        lines.append(f"  DSR >= {DSR_GATE} : {len(self.dsr_passed)} of {self.n_ran}")
        if self.dsr_passed:
            for s in self.dsr_passed[:10]:
                lines.append(f"      + {s}")
        lines.append(f"  committed       : {self.committed}")
        if not self.is_real_data:
            lines += [
                "",
                "  !! SYNTHETIC DATA — plumbing test only.",
                "  !! No verdict here means anything about any strategy, and",
                "  !! nothing was written to the registry.",
            ]
        lines.append("=" * 68)
        return "\n".join(lines)


# ── The pipeline ────────────────────────────────────────────────────────────

def run_gauntlet(
    source: DataSource,
    registry: Optional[StrategyRegistry] = None,
    limit: Optional[int] = None,
    commit: bool = False,
) -> GauntletReport:
    """
    Execute every runnable registered strategy against `source`, then score the
    whole family together with DSR and PBO.

    Scoring the family together is deliberate: the multiple-testing correction
    is only honest if the Sharpe dispersion comes from every candidate tried,
    losers included.
    """
    reg = registry or StrategyRegistry.load()
    entries = reg.all()

    if commit and not source.is_real:
        raise ValueError(
            "refusing to commit results derived from synthetic data — "
            "this interlock is the whole point of DataSource.is_real"
        )

    candidates = []
    for e in entries:
        parts = e.module.split(".")
        path = Path(*parts).with_suffix(".py")
        if not path.exists():
            continue
        shape, _ = detect_shape(path)
        if shape in (StrategyShape.ON_BAR_CLASS, StrategyShape.RUN_FN):
            candidates.append((e, path))
    if limit:
        candidates = candidates[:limit]

    runs: List[StrategyRun] = []
    failures: Dict[str, str] = {}
    usable: Dict[str, np.ndarray] = {}

    for entry, path in candidates:
        adapter = StrategyAdapter(entry.id, entry.module, path)
        loaded = adapter.load()
        if not loaded.ok:
            runs.append(StrategyRun(entry.id, False, loaded.reason))
            failures[entry.id] = loaded.reason
            continue
        try:
            rets = adapter.generate_returns(source.prices)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - one file must not kill the batch
            reason = f"{type(exc).__name__}: {exc}"
            runs.append(StrategyRun(entry.id, False, reason))
            failures[entry.id] = reason
            continue

        rets = np.asarray(rets, dtype=float)
        rets = rets[np.isfinite(rets)]
        try:
            metrics = compute_all(rets).as_dict()
        except InsufficientSampleError as exc:
            runs.append(StrategyRun(entry.id, False, f"metrics: {exc}"))
            failures[entry.id] = f"metrics: {exc}"
            continue

        runs.append(StrategyRun(entry.id, True, "", returns=rets, metrics=metrics))
        usable[entry.id] = rets

    report = GauntletReport(
        data_source=f"{source.name} ({source.description})",
        is_real_data=source.is_real,
        n_registered=len(entries),
        n_attempted=len(candidates),
        n_ran=len(usable),
        n_failed=len(failures),
        runs=runs,
        failures=failures,
    )

    if len(usable) >= 2:
        try:
            dsr_results = evaluate_trial_family(usable)
            report.dsr_passed = sorted(k for k, v in dsr_results.items() if v.passed)
        except DSRError as exc:
            log.warning("DSR scoring skipped: %s", exc)
        try:
            pbo_res = pbo_from_trials(usable, n_splits=10)
            report.pbo = pbo_res.pbo
            report.pbo_passed = pbo_res.passed
        except PBOError as exc:
            log.warning("PBO scoring skipped: %s", exc)

    if commit and source.is_real:
        _commit(reg, report, usable)
        report.committed = True

    return report


def _commit(reg: StrategyRegistry, report: GauntletReport, usable: Dict[str, np.ndarray]) -> None:
    """Record DSR/PBO on entries and advance those that earned it."""
    dsr_results = evaluate_trial_family(usable)
    for sid, res in dsr_results.items():
        entry = reg.get(sid)
        if entry is None:
            continue
        entry.dsr = res.dsr
        entry.pbo = report.pbo
        if entry.status == "backtest":
            try:
                reg.promote(sid)  # -> walkforward; no gate at this rung
            except RegistryError as exc:
                log.debug("no promotion for %s: %s", sid, exc)
    reg.save()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the strategy validation gauntlet.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--self-test", action="store_true",
                     help="run on a synthetic walk to verify the machine works (no verdicts)")
    src.add_argument("--prices", type=Path, help="CSV of real prices")
    ap.add_argument("--column", default="close", help="price column name (default: close)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of strategies")
    ap.add_argument("--commit", action="store_true",
                    help="write DSR/PBO back to the registry (real data only)")
    ap.add_argument("--json", type=Path, help="write the full report as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    source = synthetic_source() if args.self_test else csv_source(args.prices, args.column)

    try:
        report = run_gauntlet(source, limit=args.limit, commit=args.commit)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(report.summary())

    if report.failures:
        print("\nfailures (first 10):")
        for sid, why in list(report.failures.items())[:10]:
            print(f"  - {sid}: {why[:100]}")

    if args.json:
        args.json.write_text(json.dumps(
            {
                "data_source": report.data_source,
                "is_real_data": report.is_real_data,
                "n_registered": report.n_registered,
                "n_attempted": report.n_attempted,
                "n_ran": report.n_ran,
                "n_failed": report.n_failed,
                "pbo": report.pbo,
                "dsr_passed": report.dsr_passed,
                "committed": report.committed,
                "runs": [r.as_dict() for r in report.runs],
                "failures": report.failures,
            },
            indent=2,
            default=str,
        ))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
