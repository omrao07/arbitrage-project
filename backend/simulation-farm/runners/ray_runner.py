# simulation-farm/runners/ray_runner.py
"""
RayRunner: execute Simulation Farm jobs on a Ray cluster.

Features
--------
- Works with local Ray (ray.init()) or a remote cluster (address="auto")
- Per-job resource hints (num_cpus / num_gpus / custom resources)
- Run single jobs or many concurrently
- Optional retries on failure
- Returns each job's report URLs (from ReportGenerator) + diagnostics

Install:
    pip install ray

Quick example:
--------------
from simulation_farm.runners.ray_runner import RayRunner
from simulation_farm.jobs.backtest_job import BacktestSpec

runner = RayRunner(address="auto", num_cpus=2)
spec = BacktestSpec(
    run_id="bt_ray_demo",
    data_path="data/sp500_daily.parquet",
    strategy="momentum",
    params={"lookback_days": 90, "top_k": 50},
    start="2020-01-01", end="2023-12-31",
    output_prefix="runs/momentum_us/",
)

res = runner.run("backtest", spec)
print(res["status"], res["urls"])
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ray is an optional dependency for the project
try:
    import ray  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("ray_runner.py requires Ray. Install with: pip install ray") from e


# ---------------- Job registry (same mapping used elsewhere) ----------------

def _job_class_for(kind: str):
    kind = kind.lower()
    if kind == "backtest":
        from simulation_farm.jobs.backtest_job import BacktestJob # type: ignore
        return BacktestJob
    if kind in ("mc", "monte_carlo", "montecarlo"):
        from simulation_farm.jobs.monte_carlo_job import MonteCarloJob# type: ignore
        return MonteCarloJob
    if kind == "replay":
        from simulation_farm.jobs.replay_job import ReplayJob# type: ignore
        return ReplayJob
    if kind in ("stress", "stress_test", "stresstest"):
        from simulation_farm.jobs.stress_test_job import StressTestJob# type: ignore
        return StressTestJob
    raise ValueError(f"Unknown job kind: {kind}")


# ---------------- Ray remote wrapper ----------------

@ray.remote
def _run_job_remote(job_type: str, spec: Any) -> Dict[str, Any]:
    """
    Remote function that constructs + runs a job, catching exceptions.
    Returns a result dict similar to LocalRunner.
    """
    t0 = time.perf_counter()
    cpu0 = time.process_time()

    meta = {
        "job_type": job_type,
        "run_id": getattr(spec, "run_id", None),
        "spec": _safe_dataclass_to_dict(spec),
        "pid": os.getpid(),
        "node": ray.util.get_node_ip_address() if hasattr(ray.util, "get_node_ip_address") else None,
    }

    try:
        JobCls = _job_class_for(job_type)
        job = JobCls(spec)
        urls = job.run()  # must return dict (from ReportGenerator)
        status = "succeeded"
        err = None
        tb = None
    except Exception as e:
        urls = {}
        status = "failed"
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()

    t1 = time.perf_counter()
    cpu1 = time.process_time()

    return {
        "status": status,
        "urls": urls,
        "error": err,
        "traceback": tb,
        "diag": {
            "runtime_s": round(t1 - t0, 4),
            "cpu_time_s": round(cpu1 - cpu0, 4),
        },
        **meta,
    }


# ---------------- Runner ----------------

class RayRunner:
    def __init__(
        self,
        address: Optional[str] = "auto",
        namespace: str = "simfarm",
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        resources: Optional[Dict[str, float]] = None,
        retry: int = 0,
        verbose: bool = True,
    ):
        """
        Args:
            address: Ray cluster address ("auto" for discovered cluster, None for local).
            namespace: Ray namespace to isolate jobs.
            num_cpus: default CPU reservation per job (can be overridden per-call).
            num_gpus: default GPU reservation per job.
            resources: extra Ray resources dict (e.g., {"accelerator_type:T4": 1})
            retry: how many times to retry a failed job (simple re-submit).
            verbose: print small status logs.
        """
        self._ensure_ray(address, namespace)
        self.default_opts = {"num_cpus": num_cpus, "num_gpus": num_gpus, "resources": resources or {}}
        self.retry = max(0, int(retry))
        self.verbose = bool(verbose)

    def _ensure_ray(self, address: Optional[str], namespace: str):
        # Initialize ray once. If already initialized, noop.
        if ray.is_initialized():
            return
        if address in (None, "", "local"):
            ray.init(namespace=namespace, ignore_reinit_error=True)
        else:
            ray.init(address=address, namespace=namespace, ignore_reinit_error=True)

    # ---- public API ----

    def run(
        self,
        job_type: str,
        spec: Any,
        *,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        resources: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run a single job; returns its result dict."""
        opts = _merge_opts(self.default_opts, num_cpus=num_cpus, num_gpus=num_gpus, resources=resources)
        return self._submit_and_wait(job_type, spec, opts)

    def run_many(
        self,
        jobs: Sequence[Tuple[str, Any]],
        *,
        num_cpus: Optional[float] = None,
        num_gpus: Optional[float] = None,
        resources: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Submit many jobs concurrently; returns list of results in completion order.
        """
        opts = _merge_opts(self.default_opts, num_cpus=num_cpus, num_gpus=num_gpus, resources=resources)
        # Submit all
        submitted = []
        for (jt, sp) in jobs:
            handle = self._submit_remote(jt, sp, opts)
            submitted.append((jt, sp, handle, 0))  # (type, spec, obj_ref, attempts)

        results: List[Dict[str, Any]] = []
        pending = [h for (_, _, h, _) in submitted]
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            obj_ref = done[0]
            # Find the tuple
            idx = next(i for i, tup in enumerate(submitted) if tup[2] == obj_ref)
            jt, sp, _, attempts = submitted[idx]
            try:
                res = ray.get(obj_ref)
            except Exception as e:
                if attempts < self.retry:
                    if self.verbose:
                        print(f"[RayRunner] retry {attempts+1}/{self.retry} for job {getattr(sp, 'run_id', None)}", flush=True)
                    new_ref = self._submit_remote(jt, sp, opts)
                    submitted[idx] = (jt, sp, new_ref, attempts + 1)
                    pending.append(new_ref)
                    continue
                # Could not recover
                res = {
                    "status": "failed",
                    "urls": {},
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": "".join(traceback.format_exception_only(type(e), e)),
                    "job_type": jt,
                    "run_id": getattr(sp, "run_id", None),
                    "diag": {},
                }
            results.append(res)
        return results

    # ---- internals ----

    def _submit_remote(self, job_type: str, spec: Any, opts: Dict[str, Any]):
        # Apply Ray options (resources)
        fn = _run_job_remote.options(
            num_cpus=opts.get("num_cpus"),
            num_gpus=opts.get("num_gpus"),
            resources=opts.get("resources") or None,
        )
        if self.verbose:
            rid = getattr(spec, "run_id", None)
            print(f"[RayRunner] submit {job_type} run_id={rid} "
                  f"(cpus={opts.get('num_cpus')} gpus={opts.get('num_gpus')} res={opts.get('resources')})", flush=True)
        return fn.remote(job_type, spec)

    def _submit_and_wait(self, job_type: str, spec: Any, opts: Dict[str, Any]) -> Dict[str, Any]:
        attempts = 0
        last_err: Optional[Dict[str, Any]] = None
        while attempts <= self.retry:
            ref = self._submit_remote(job_type, spec, opts)
            try:
                return ray.get(ref)
            except Exception as e:
                last_err = {
                    "status": "failed",
                    "urls": {},
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": "".join(traceback.format_exception_only(type(e), e)),
                    "job_type": job_type,
                    "run_id": getattr(spec, "run_id", None),
                    "diag": {},
                }
                attempts += 1
                if attempts <= self.retry and self.verbose:
                    print(f"[RayRunner] retry {attempts}/{self.retry} for {job_type} run_id={getattr(spec, 'run_id', None)}", flush=True)
        return last_err or {"status": "failed", "urls": {}, "error": "Unknown error"}

# ---------------- Utilities ----------------

def _merge_opts(base: Dict[str, Any], **over: Any) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if v is not None:
            if k == "resources":
                r = dict(out.get("resources") or {})
                r.update(v or {})
                out["resources"] = r
            else:
                out[k] = v
    return out

def _safe_dataclass_to_dict(obj: Any) -> Dict[str, Any] | Any:
    if is_dataclass(obj):
        try:
            d = asdict(obj)# type: ignore
            if "params" in d and isinstance(d["params"], dict) and len(d["params"]) > 100:
                d["params"] = {"__len__": len(d["params"])}
            return d
        except Exception:
            return {"__repr__": repr(obj)}
    return obj


# ---------------- CLI ----------------

def _parse_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Run Simulation Farm jobs on Ray.")
    ap.add_argument("--job", required=True, choices=["backtest", "monte_carlo", "replay", "stress_test"])
    ap.add_argument("--spec-json", required=True, help="Path to a JSON file containing the job spec fields.")
    ap.add_argument("--address", default="auto")
    ap.add_argument("--namespace", default="simfarm")
    ap.add_argument("--num-cpus", type=float, default=None)
    ap.add_argument("--num-gpus", type=float, default=None)
    ap.add_argument("--retry", type=int, default=0)
    args = ap.parse_args()

    with open(args.spec_json, "r", encoding="utf-8") as f:
        spec_kwargs = json.load(f)

    # Build the right spec
    if args.job == "backtest":# type: ignore
        from simulation_farm.jobs.backtest_job import BacktestSpec# type: ignore
    elif args.job == "monte_carlo":# type: ignore
        from simulation_farm.jobs.monte_carlo_job import MonteCarloSpec# type: ignore
        Spec = MonteCarloSpec
    elif args.job == "replay":
        from simulation_farm.jobs.replay_job import ReplaySpec# type: ignore
        Spec = ReplaySpec
    else:
        from simulation_farm.jobs.stress_test_job import StressTestSpec# type: ignore
        Spec = StressTestSpec

    spec = Spec(**spec_kwargs)
    runner = RayRunner(
        address=args.address,
        namespace=args.namespace,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        retry=args.retry,
    )
    return args.job, spec, runner


if __name__ == "__main__":
    job_type, spec, runner = _parse_cli()
    result = runner.run(job_type, spec)
    print(json.dumps(result, indent=2))