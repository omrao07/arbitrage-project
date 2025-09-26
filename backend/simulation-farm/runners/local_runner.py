# simulation-farm/runners/local_runner.py
"""
LocalRunner: run Simulation Farm jobs locally (single or concurrent).

Supports job types:
- backtest      -> simulation_farm.jobs.backtest_job.BacktestJob
- monte_carlo   -> simulation_farm.jobs.monte_carlo_job.MonteCarloJob
- replay        -> simulation_farm.jobs.replay_job.ReplayJob
- stress_test   -> simulation_farm.jobs.stress_test_job.StressTestJob

Example
-------
from simulation_farm.runners.local_runner import LocalRunner
from simulation_farm.jobs.backtest_job import BacktestSpec

runner = LocalRunner(max_workers=4)
spec = BacktestSpec(
    run_id="bt_local_demo",
    data_path="data/sp500_daily.parquet",
    strategy="momentum",
    params={"lookback_days": 90, "top_k": 50},
    start="2020-01-01", end="2023-12-31",
    output_prefix="runs/momentum_us/",
)
res = runner.run("backtest", spec)
print(res["status"], res["urls"])

# Multiple jobs concurrently
jobs = [("backtest", spec1), ("monte_carlo", spec2)]
results = runner.run_many(jobs)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Optional peak-RSS on Unix
try:
    import resource  # type: ignore
    _HAVE_RESOURCE = True
except Exception:
    _HAVE_RESOURCE = False

from concurrent.futures import ThreadPoolExecutor, as_completed


# -------- Job registry (import lazily to keep import cost small) --------

def _job_class_for(kind: str):
    kind = kind.lower()
    if kind == "backtest":
        from simulation_farm.jobs.backtest_job import BacktestJob # type: ignore
        return BacktestJob
    if kind in ("mc", "monte_carlo", "montecarlo"):
        from simulation_farm.jobs.monte_carlo_job import MonteCarloJob # type: ignore
        return MonteCarloJob
    if kind == "replay":
        from simulation_farm.jobs.replay_job import ReplayJob # type: ignore
        return ReplayJob
    if kind in ("stress", "stress_test", "stresstest"):
        from simulation_farm.jobs.stress_test_job import StressTestJob # type: ignore
        return StressTestJob
    raise ValueError(f"Unknown job kind: {kind}")


# -------- Runner --------

class LocalRunner:
    def __init__(self, max_workers: int = 1, use_threads: bool = True, verbose: bool = True):
        """
        Args:
          max_workers: degree of concurrency (>=1). Each job runs in its own thread by default.
          use_threads: True for threads (default), False to run sequentially.
          verbose: print small status messages to stdout.
        """
        self.max_workers = max(1, int(max_workers))
        self.use_threads = bool(use_threads)
        self.verbose = bool(verbose)

    # ---- public API ----

    def run(self, job_type: str, spec: Any) -> Dict[str, Any]:
        """Run a single job and return a result dict with diagnostics."""
        return _run_once(job_type, spec, verbose=self.verbose)

    def run_many(self, jobs: Sequence[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run multiple jobs concurrently.
        Args:
          jobs: sequence of (job_type, spec)
        Returns:
          list of result dicts in completion order
        """
        if not self.use_threads or self.max_workers == 1 or len(jobs) <= 1:
            return [self.run(jt, sp) for jt, sp in jobs]

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(_run_once, jt, sp, self.verbose): (jt, sp) for jt, sp in jobs}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:  # should be caught inside _run_once already
                    jt, sp = futs[fut]
                    results.append({
                        "status": "error",
                        "job_type": jt,
                        "run_id": getattr(sp, "run_id", None),
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(limit=5),
                    })
        return results


# -------- internal: one-shot execution with diagnostics --------

def _run_once(job_type: str, spec: Any, verbose: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    peak_before = _peak_rss_mb()

    meta = {
        "job_type": job_type,
        "run_id": getattr(spec, "run_id", None),
        "spec": _safe_dataclass_to_dict(spec),
        "pid": os.getpid(),
    }
    if verbose:
        print(f"[LocalRunner] start {job_type} run_id={meta['run_id']}", flush=True)

    try:
        JobCls = _job_class_for(job_type)
        job = JobCls(spec)
        urls = job.run()   # expected to return dict of URLs from ReportGenerator
        status = "succeeded"
        err = None
        tb = None
    except Exception as e:
        urls = {}
        status = "failed"
        err = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()
        if verbose:
            print(f"[LocalRunner] FAILED {job_type} run_id={meta['run_id']}: {err}", file=sys.stderr)

    t1 = time.perf_counter()
    cpu1 = time.process_time()
    peak_after = _peak_rss_mb()

    diag = {
        "runtime_s": round(t1 - t0, 4),
        "cpu_time_s": round(cpu1 - cpu0, 4),
        "peak_rss_mb": peak_after or peak_before,
    }

    result = {
        "status": status,
        "urls": urls,
        "error": err,
        "traceback": tb,
        "diag": diag,
        **meta,
    }

    if verbose:
        print(f"[LocalRunner] done {job_type} run_id={meta['run_id']} status={status} "
              f"(wall={diag['runtime_s']}s cpu={diag['cpu_time_s']}s rss~{diag['peak_rss_mb']}MB)", flush=True)
    return result


def _peak_rss_mb() -> float | None:
    if not _HAVE_RESOURCE:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is kilobytes on Linux/macOS (but bytes on some BSDs); Linux is KiB.
        val_kb = getattr(usage, "ru_maxrss", 0) or 0
        return round(val_kb / 1024.0, 2)  # -> MB
    except Exception:
        return None


def _safe_dataclass_to_dict(obj: Any) -> Dict[str, Any] | Any:
    if is_dataclass(obj):
        try:
            d = asdict(obj) # type: ignore
            # avoid dumping huge params
            if "params" in d and isinstance(d["params"], dict) and len(d["params"]) > 100:
                d["params"] = {"__len__": len(d["params"])}
            return d
        except Exception:
            return {"__repr__": repr(obj)}
    return obj


# ---------------- CLI ----------------

def _parse_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Run Simulation Farm jobs locally.")
    ap.add_argument("--job", required=True, choices=["backtest", "monte_carlo", "replay", "stress_test"])
    ap.add_argument("--spec-json", help="Path to a JSON file containing the job spec fields.")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    args, unknown = ap.parse_known_args()

    spec_kwargs: Dict[str, Any] = {}
    if args.spec_json:
        import json
        with open(args.spec_json, "r", encoding="utf-8") as f:
            spec_kwargs.update(json.load(f))

    # Minimal inline flags for convenience (override JSON)
    def pop_flag(name: str, conv=str):
        if f"--{name}" in unknown:
            i = unknown.index(f"--{name}")
            if i + 1 < len(unknown):
                val = unknown[i + 1]
                spec_kwargs[name] = conv(val)

    # Common fields
    pop_flag("run-id", str); pop_flag("data", str); pop_flag("start", str); pop_flag("end", str)
    pop_flag("cash", float); pop_flag("strategy", str); pop_flag("out-prefix", str) # type: ignore

    # Build spec object for the selected job
    job_type = args.job
    if job_type == "backtest":
        from simulation_farm.jobs.backtest_job import BacktestSpec # type: ignore
        # map CLI-like keys to dataclass field names
        field_map = {
            "run-id": "run_id", "data": "data_path", "out-prefix": "output_prefix",
        }
        spec = BacktestSpec(**_remap_keys(spec_kwargs, field_map))
    elif job_type == "monte_carlo":
        from simulation_farm.jobs.monte_carlo_job import MonteCarloSpec # type: ignore
        field_map = {
            "run-id": "run_id", "data": "data_path", "out-prefix": "output_prefix",
        }
        spec = MonteCarloSpec(**_remap_keys(spec_kwargs, field_map))
    elif job_type == "replay":
        from simulation_farm.jobs.replay_job import ReplaySpec # type: ignore
        field_map = {
            "run-id": "run_id", "data": "data_path", "out-prefix": "output_prefix",
        }
        spec = ReplaySpec(**_remap_keys(spec_kwargs, field_map))
    else:
        from simulation_farm.jobs.stress_test_job import StressTestSpec # type: ignore
        field_map = {
            "run-id": "run_id", "data": "data_path", "out-prefix": "output_prefix",
        }
        spec = StressTestSpec(**_remap_keys(spec_kwargs, field_map))

    runner = LocalRunner(max_workers=args.max_workers, use_threads=True, verbose=not args.quiet)
    return job_type, spec, runner


def _remap_keys(d: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        out[mapping.get(k, k).replace("-", "_")] = v
    return out


if __name__ == "__main__":
    job_type, spec, runner = _parse_cli()
    result = runner.run(job_type, spec)
    print(json.dumps(result, indent=2))