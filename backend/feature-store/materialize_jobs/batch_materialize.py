# materialize_jobs/batch_materialize.py
"""
Batch materialize multiple Feast feature views, optionally in parallel.

Highlights
----------
- Drive runs from a YAML plan (views, windows, chunk size, online/offline).
- Or auto-discover views by prefix (eq_, fx_, macro_, credit_, vol_).
- Chunked windows (e.g., 30d) with retries.
- Offline-only or also push to online store.
- Incremental mode (start -> NOW).
- Optional parallelization by view group.

Requirements
------------
pip install feast pandas pyyaml

Example plan.yaml
-----------------
repo: ./feature-store
defaults:
  chunk_days: 30
  to_online: true
  incremental: false
  retries: 2
  sleep_sec: 10
  parallel_workers: 3
batches:
  - name: equities
    view_prefixes: ["eq_"]         # auto-pick views by name prefix
    start: 2023-01-01
    end:   2025-09-01
    chunk_days: 30
    to_online: true
  - name: fx
    views: ["fx_carry_signals","fx_carry_derived"]   # explicit list
    start: 2022-01-01
    end:   2025-09-01
    chunk_days: 30
    to_online: true
  - name: macro (inc)
    view_prefixes: ["macro_"]
    start: 2024-01-01
    incremental: true             # end is ignored, uses NOW
    to_online: false

CLI
---
python materialize_jobs/batch_materialize.py --plan plan.yaml
# or run without plan, auto-pick by prefixes:
python materialize_jobs/batch_materialize.py --repo ./feature-store --prefixes eq_ fx_ --start 2023-01-01 --end 2025-09-01 --to-online
"""

from __future__ import annotations

import os
import sys
import time
import yaml
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Iterable, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from feast import FeatureStore # type: ignore
from feast.feature_view import FeatureView # type: ignore

ISO_FMT = "%Y-%m-%d"


# ----------------------------- utils ------------------------------

def parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)

def daterange_chunks(start: datetime, end: datetime, chunk_days: int) -> Iterable[Tuple[datetime, datetime]]:
    """Yield [chunk_start, chunk_end) half-open ranges."""
    cur = start
    step = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(end, cur + step)
        yield cur, nxt
        cur = nxt

def human_range(a: datetime, b: datetime) -> str:
    return f"{a.strftime(ISO_FMT)} → {b.strftime(ISO_FMT)}"

def resolve_views(store: FeatureStore, explicit: Optional[List[str]], prefixes: Optional[List[str]]) -> List[FeatureView]:
    fv_map = store.list_feature_views()
    if explicit:
        missing = [v for v in explicit if v not in fv_map]
        if missing:
            raise SystemExit(f"[ERROR] Feature views not found: {missing}. Available: {sorted(fv_map.keys())}")
        return [fv_map[v] for v in explicit]
    # prefix discovery
    if not prefixes:
        raise SystemExit("[ERROR] No views or prefixes provided.")
    picks: List[FeatureView] = []
    for name, fv in fv_map.items():
        if any(name.startswith(p) for p in prefixes):
            picks.append(fv)
    if not picks:
        raise SystemExit(f"[ERROR] No feature views match prefixes: {prefixes}")
    return picks


# ----------------------------- configs ----------------------------

@dataclass
class BatchSpec:
    name: str
    views: Optional[List[str]] = None
    view_prefixes: Optional[List[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    incremental: bool = False
    chunk_days: Optional[int] = None
    to_online: Optional[bool] = None
    retries: Optional[int] = None
    sleep_sec: Optional[int] = None

@dataclass
class Plan:
    repo: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    batches: List[BatchSpec] = field(default_factory=list)

def load_plan(path: str) -> Plan:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    batches = [BatchSpec(**b) for b in raw.get("batches", [])]
    return Plan(repo=raw["repo"], defaults=raw.get("defaults", {}), batches=batches)


# ---------------------------- core runs ---------------------------

def _materialize_with_retry(
    store: FeatureStore,
    views: List[FeatureView],
    start: datetime,
    end: datetime,
    to_online: bool,
    retries: int,
    sleep_sec: int,
    incremental: bool,
    prefix: str,
):
    attempt = 0
    while True:
        try:
            if incremental:
                if to_online:
                    print(f"[{prefix}] incremental → online: end={end.strftime(ISO_FMT)} views={[v.name for v in views]}")
                    store.materialize_incremental(end_date=end, feature_views=views)
                else:
                    print(f"[{prefix}] incremental → offline: {human_range(start, end)}")
                    store.materialize(start_date=start, end_date=end, feature_views=views)
            else:
                print(f"[{prefix}] offline: {human_range(start, end)}")
                store.materialize(start_date=start, end_date=end, feature_views=views)
                if to_online:
                    print(f"[{prefix}] push online (incremental): end={end.strftime(ISO_FMT)}")
                    store.materialize_incremental(end_date=end, feature_views=views)
            return
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"[{prefix}] Failed after {retries} retries: {e}") from e
            print(f"[{prefix}] WARN attempt {attempt}/{retries} failed: {e}. Retry in {sleep_sec}s...")
            time.sleep(sleep_sec)

def run_batch(
    store: FeatureStore,
    spec: BatchSpec,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    name = spec.name
    chunk_days = int(spec.chunk_days or defaults.get("chunk_days", 30))
    to_online = bool(spec.to_online if spec.to_online is not None else defaults.get("to_online", False))
    retries = int(spec.retries or defaults.get("retries", 2))
    sleep_sec = int(spec.sleep_sec or defaults.get("sleep_sec", 10))
    incremental = bool(spec.incremental if spec.incremental is not None else defaults.get("incremental", False))

    views = resolve_views(store, spec.views, spec.view_prefixes)

    start = parse_date(spec.start) if spec.start else None
    end = parse_date(spec.end) if spec.end else None
    now_utc = datetime.now(timezone.utc)

    if incremental:
        if not start:
            raise SystemExit(f"[{name}] --incremental requires a start date.")
        end = now_utc
        # one call covers the whole range; no chunk loop needed
        _materialize_with_retry(store, views, start, end, to_online, retries, sleep_sec, incremental=True, prefix=name)
        return {"batch": name, "status": "ok", "mode": "incremental", "views": [v.name for v in views]}

    if not (start and end):
        raise SystemExit(f"[{name}] start and end are required for non-incremental runs.")

    # chunked runs
    for a, b in daterange_chunks(start, end, chunk_days):
        _materialize_with_retry(store, views, a, b, to_online, retries, sleep_sec, incremental=False, prefix=name)

    return {"batch": name, "status": "ok", "mode": "window", "views": [v.name for v in views]}


# ------------------------------- CLI -------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser("Batch materialize multiple Feast feature view groups")
    ap.add_argument("--plan", help="YAML plan (recommended).")
    ap.add_argument("--repo", help="Feast repo path (folder with repo.yaml)")
    ap.add_argument("--prefixes", nargs="*", help="View name prefixes (e.g., eq_ fx_ macro_)")
    ap.add_argument("--views", nargs="*", help="Explicit view names")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--to-online", action="store_true")
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--parallel-workers", type=int, default=1, help="Parallelize batches (plan mode) or groups (ad-hoc)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.plan:
        plan = load_plan(args.plan)
        store = FeatureStore(repo_path=plan.repo)

        workers = int(plan.defaults.get("parallel_workers", 1))
        if workers < 1:
            workers = 1

        results = []
        if workers == 1 or len(plan.batches) <= 1:
            for spec in plan.batches:
                print(f"\n[RUN] batch: {spec.name}")
                res = run_batch(store, spec, plan.defaults)
                results.append(res)
        else:
            # Parallelize batches
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = []
                for spec in plan.batches:
                    futs.append(ex.submit(run_batch, store, spec, plan.defaults))
                for fut in as_completed(futs):
                    results.append(fut.result())

        print("\n[SUMMARY]")
        for r in results:
            print(f"  - {r['batch']}: {r['status']} ({r['mode']}), views={r['views']}")
        return

    # Ad-hoc mode (no plan): single batch from CLI flags
    if not args.repo:
        raise SystemExit("--repo is required when not using --plan")

    store = FeatureStore(repo_path=args.repo)
    spec = BatchSpec(
        name="adhoc",
        views=args.views,
        view_prefixes=args.prefixes,
        start=args.start,
        end=args.end,
        incremental=bool(args.incremental),
        chunk_days=int(args.chunk_days),
        to_online=bool(args.to_online),
    )
    res = run_batch(store, spec, defaults={})
    print(f"\n[SUMMARY] {res['batch']}: {res['status']} ({res['mode']}), views={res['views']}")


if __name__ == "__main__":
    main()