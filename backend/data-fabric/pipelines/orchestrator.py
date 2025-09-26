# data-fabric/pipelines/orchestrator.py
"""
Hyper-OS Data Fabric — Orchestrator

Runs and schedules pipelines:
  - equities_pipeline.py
  - fx_pipeline.py
  - macro_pipeline.py

Features
- Config-driven jobs (YAML or JSON)
- Schedules: interval / hourly / daily
- Per-job retries with exponential backoff
- Concurrent execution (ThreadPoolExecutor)
- Backfill helpers for daily windows
- Run state persisted to .state/orchestrator.json
- Structured JSON logging

Usage
  # Run once (all jobs) using config
  python orchestrator.py --config pipelines.yaml --run-once

  # Daemon mode: loop forever and trigger on schedule
  python orchestrator.py --config pipelines.yaml --watch

  # Run a single job by name now
  python orchestrator.py --config pipelines.yaml --job equities_eod

  # Backfill a job across dates (for pipelines that accept start/end)
  python orchestrator.py --config pipelines.yaml --job equities_eod \
    --backfill 2024-01-01 2024-01-31
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import traceback
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional YAML config
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# Import pipeline entrypoints (their main(argv) returns int)
# These imports match the files we created earlier.
try:
    from data_fabric.pipelines.equities_pipeline import main as equities_main  # type: ignore
except Exception:
    from equities_pipeline import main as equities_main  # type: ignore

try:
    from data_fabric.pipelines.fx_pipeline import main as fx_main  # type: ignore
except Exception:
    from fx_pipeline import main as fx_main  # type: ignore

try:
    from data_fabric.pipelines.macro_pipeline import main as macro_main  # type: ignore
except Exception:
    from macro_pipeline import main as macro_main  # type: ignore


# --------------------------- Logging ---------------------------

def log(msg: str, **kv):
    now = datetime.now(timezone.utc).isoformat()
    rec = {"ts": now, "msg": msg}
    rec.update(kv)
    print(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))


# -------------------------- Dataclasses ------------------------

@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_base_sec: float = 1.0
    backoff_max_sec: float = 60.0

    def delay(self, attempt: int) -> float:
        # attempt is 1-based
        return min(self.backoff_max_sec, self.backoff_base_sec * (2 ** (attempt - 1)))


@dataclass
class Schedule:
    kind: str = "interval"          # "interval" | "hourly" | "daily" | "none"
    every_seconds: Optional[int] = None  # for interval
    at_minute: Optional[int] = None      # for hourly: minute(0-59)
    at_time: Optional[str] = None        # for daily: "HH:MM"
    timezone: str = "UTC"                # (reserved, not used beyond UTC in this simple impl)

    def next_run_after(self, ref: datetime) -> datetime:
        if self.kind == "none":
            return ref  # trigger immediately (one-shot)
        if self.kind == "interval":
            sec = int(self.every_seconds or 3600)
            return ref + timedelta(seconds=sec)
        if self.kind == "hourly":
            minute = int(self.at_minute or 0)
            next_dt = ref.replace(second=0, microsecond=0)
            # move to next minute trigger
            if next_dt.minute >= minute:
                next_dt = next_dt + timedelta(hours=1)
            return next_dt.replace(minute=minute)
        if self.kind == "daily":
            hh, mm = (self.at_time or "00:00").split(":")
            target = ref.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            if ref >= target:
                target = target + timedelta(days=1)
            return target
        # default safety
        return ref + timedelta(hours=1)


@dataclass
class Job:
    name: str
    pipeline: str                    # "equities" | "fx" | "macro"
    args: Dict[str, Any] = field(default_factory=dict)  # CLI-style args
    schedule: Schedule = field(default_factory=Schedule)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    enabled: bool = True

    # runtime-managed
    _next_run: Optional[datetime] = None


@dataclass
class OrchestratorConfig:
    concurrency: int = 4
    state_path: str = "./.state/orchestrator.json"
    jobs: List[Job] = field(default_factory=list)


# -------------------------- State Store ------------------------

class RunState:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get_last_run(self, job_name: str) -> Optional[str]:
        return (self.data.get("jobs", {}).get(job_name, {}) or {}).get("last_run")

    def set_last_run(self, job_name: str, ts_iso: str):
        self.data.setdefault("jobs", {}).setdefault(job_name, {})["last_run"] = ts_iso
        self.save()

    def record_result(self, job_name: str, status: str, detail: Dict[str, Any]):
        d = self.data.setdefault("jobs", {}).setdefault(job_name, {})
        d["last_status"] = status
        d["last_detail"] = detail
        d["last_update"] = datetime.now(timezone.utc).isoformat()
        self.save()


# -------------------------- Helpers ----------------------------

def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def to_utc_now() -> datetime:
    return datetime.now(timezone.utc)

def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def build_argv(pipeline: str, args: Dict[str, Any]) -> List[str]:
    """
    Convert job.args dict to argv list understood by each pipeline's CLI.
    Only includes keys with non-None values. Boolean True flags become '--flag'.
    """
    argv: List[str] = []
    # allow a 'mode' for macro (fred/worldbank)
    if pipeline == "macro":
        mode = args.get("mode", "fred")
        argv.append(mode)

    for k, v in args.items():
        if pipeline == "macro" and k == "mode":
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend([str(x) for x in v])
        else:
            argv.append(flag)
            argv.append(str(v))
    return argv

PIPELINE_DISPATCH: Dict[str, Callable[[List[str]], int]] = {
    "equities": equities_main,
    "fx": fx_main,
    "macro": macro_main,
}


# ------------------------- Orchestrator ------------------------

class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        self.state = RunState(cfg.state_path)

    def _init_next_runs(self):
        now = to_utc_now()
        for job in self.cfg.jobs:
            if not job.enabled:
                continue
            last = self.state.get_last_run(job.name)
            if last:
                # schedule from now
                job._next_run = job.schedule.next_run_after(now)
            else:
                # first time: run immediately for interval/none, or at next boundary for hourly/daily
                if job.schedule.kind in ("none", "interval"):
                    job._next_run = now
                else:
                    job._next_run = job.schedule.next_run_after(now)

    # ---------- Core execution ----------

    def run_job_once(self, job: Job, extra_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not job.enabled:
            log("job_skipped_disabled", job=job.name)
            return {"job": job.name, "status": "disabled"}

        args = dict(job.args)
        if extra_args:
            args.update(extra_args)

        pipeline = job.pipeline
        if pipeline not in PIPELINE_DISPATCH:
            raise ValueError(f"Unknown pipeline '{pipeline}' for job '{job.name}'")

        argv = build_argv(pipeline, args)
        entry = PIPELINE_DISPATCH[pipeline]

        attempts = 0
        while True:
            attempts += 1
            t0 = time.time()
            log("job_start", job=job.name, pipeline=pipeline, argv=argv, attempt=attempts)
            try:
                rc = entry(argv)
                dur = round(time.time() - t0, 2)
                if rc == 0:
                    self.state.set_last_run(job.name, to_utc_now().isoformat())
                    self.state.record_result(job.name, "ok", {"rc": rc, "secs": dur})
                    log("job_done", job=job.name, rc=rc, secs=dur)
                    return {"job": job.name, "status": "ok", "rc": rc, "secs": dur}
                else:
                    # Non-zero but not exception
                    self.state.record_result(job.name, "error", {"rc": rc})
                    log("job_error_rc", job=job.name, rc=rc)
                    if attempts >= job.retry.max_attempts:
                        return {"job": job.name, "status": "error", "rc": rc}
            except SystemExit as e:
                # Pipelines use SystemExit(main_rc)
                rc = int(getattr(e, "code", 1) or 1)
                dur = round(time.time() - t0, 2)
                if rc == 0:
                    self.state.set_last_run(job.name, to_utc_now().isoformat())
                    self.state.record_result(job.name, "ok", {"rc": rc, "secs": dur})
                    log("job_done", job=job.name, rc=rc, secs=dur)
                    return {"job": job.name, "status": "ok", "rc": rc, "secs": dur}
                else:
                    self.state.record_result(job.name, "error", {"rc": rc})
                    log("job_error_rc", job=job.name, rc=rc)
                    if attempts >= job.retry.max_attempts:
                        return {"job": job.name, "status": "error", "rc": rc}
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                self.state.record_result(job.name, "exception", {"error": str(e)})
                log("job_exception", job=job.name, error=str(e), tb=tb)
                if attempts >= job.retry.max_attempts:
                    return {"job": job.name, "status": "exception", "error": str(e)}

            # Backoff then retry
            delay = job.retry.delay(attempts)
            log("job_retry_backoff", job=job.name, seconds=round(delay, 2))
            time.sleep(delay)

    def backfill_job_daily(self, job: Job, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        For pipelines that accept date windows (start/end), run once per day sequentially.
        It injects the per-day window into args keys named 'start' and 'end'.
        """
        ok = 0
        fail = 0
        for d in daterange(start_date, end_date):
            s = d.isoformat()
            e = d.isoformat()
            res = self.run_job_once(job, extra_args={"start": s, "end": e})
            if res.get("status") == "ok":
                ok += 1
            else:
                fail += 1
        return {"job": job.name, "ok_days": ok, "failed_days": fail}

    # ---------- Schedulers ----------

    def run_once_all(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            futs = []
            for job in self.cfg.jobs:
                if not job.enabled:
                    continue
                futs.append(pool.submit(self.run_job_once, job))
            for f in as_completed(futs):
                results.append(f.result())
        return results

    def watch(self):
        self._init_next_runs()
        log("watch_start", jobs=len(self.cfg.jobs), concurrency=self.cfg.concurrency)

        # Simple loop scheduler
        while True:
            now = to_utc_now()
            due: List[Job] = []
            for job in self.cfg.jobs:
                if not job.enabled:
                    continue
                if job._next_run and now >= job._next_run:
                    due.append(job)

            if due:
                log("due_jobs", count=len(due), names=[j.name for j in due])
                # Fire concurrently, then reschedule
                with ThreadPoolExecutor(max_workers=min(self.cfg.concurrency, len(due))) as pool:
                    futs = {pool.submit(self.run_job_once, j): j for j in due}
                    for fut in as_completed(futs):
                        _ = fut.result()

                # compute next runs
                now2 = to_utc_now()
                for j in due:
                    j._next_run = j.schedule.next_run_after(now2)

            # sleep a bit
            time.sleep(2.0)


# --------------------------- Config IO -------------------------

def _coerce_schedule(d: Dict[str, Any]) -> Schedule:
    return Schedule(
        kind=str(d.get("kind", "interval")),
        every_seconds=int(d["every_seconds"]) if d.get("every_seconds") is not None else None,
        at_minute=int(d["at_minute"]) if d.get("at_minute") is not None else None,
        at_time=str(d["at_time"]) if d.get("at_time") is not None else None,
        timezone=str(d.get("timezone", "UTC")),
    )

def _coerce_retry(d: Dict[str, Any]) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=int(d.get("max_attempts", 3)),
        backoff_base_sec=float(d.get("backoff_base_sec", 1.0)),
        backoff_max_sec=float(d.get("backoff_max_sec", 60.0)),
    )

def load_config(path: str) -> OrchestratorConfig:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        if not HAVE_YAML:
            raise RuntimeError("PyYAML not installed. `pip install pyyaml` or use JSON config.")
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    jobs: List[Job] = []
    for jd in raw.get("jobs", []):
        job = Job(
            name=str(jd["name"]),
            pipeline=str(jd["pipeline"]),
            args=dict(jd.get("args", {})),
            schedule=_coerce_schedule(jd.get("schedule", {})),
            retry=_coerce_retry(jd.get("retry", {})),
            enabled=bool(jd.get("enabled", True)),
        )
        jobs.append(job)

    return OrchestratorConfig(
        concurrency=int(raw.get("concurrency", 4)),
        state_path=str(raw.get("state_path", "./.state/orchestrator.json")),
        jobs=jobs,
    )


# ----------------------------- CLI ----------------------------

def parse_args(argv: List[str] | None = None):
    import argparse
    p = argparse.ArgumentParser("orchestrator")
    p.add_argument("--config", required=True, help="Path to YAML/JSON job config")
    p.add_argument("--run-once", action="store_true", help="Run all enabled jobs once now")
    p.add_argument("--watch", action="store_true", help="Loop forever and run jobs on schedule")
    p.add_argument("--job", help="Run a single job by name (immediately)")
    p.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                   help="Backfill dates (YYYY-MM-DD YYYY-MM-DD) for --job")
    return p.parse_args(argv or sys.argv[1:])


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    orch = Orchestrator(cfg)

    if args.job:
        # Single job actions
        job = next((j for j in cfg.jobs if j.name == args.job), None)
        if not job:
            log("job_not_found", job=args.job)
            return 2

        if args.backfill:
            s, e = args.backfill
            start_d = date.fromisoformat(s)
            end_d = date.fromisoformat(e)
            res = orch.backfill_job_daily(job, start_d, end_d)
            log("backfill_done", **res)
            return 0
        else:
            res = orch.run_job_once(job)
            log("job_result", **res)
            return 0 if res.get("status") == "ok" else 2

    if args.run_once:
        results = orch.run_once_all()
        ok = sum(1 for r in results if r.get("status") == "ok")
        err = sum(1 for r in results if r.get("status") not in ("ok", "disabled"))
        log("run_once_done", ok=ok, err=err, results=results)
        return 0 if err == 0 else 2

    if args.watch:
        orch.watch()
        return 0

    # Default: show help
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())# backend/bus/python/events/risk.py