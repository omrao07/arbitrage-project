# materialize_jobs/backfill_equities.py
"""
Backfill job for equities feature views using Feast.

What it does
------------
1) Optionally sanity-checks the underlying Parquet source(s)
2) Calls Feast 'materialize' / 'materialize_incremental' in daily (or custom) chunks
3) Can target offline-only or also push to online store
4) Supports dry-run and simple retry

Requirements
------------
pip install feast pandas pyarrow

Usage
-----
# Backfill last 3 years, daily chunks, to offline+online:
python materialize_jobs/backfill_equities.py \
  --repo ./feature-store \
  --views eq_returns_1d eq_returns_derived \
  --start 2022-01-01 \
  --end 2025-09-01 \
  --chunk-days 30 \
  --to-online

# Dry-run (no writes), just validates sources + prints a plan:
python materialize_jobs/backfill_equities.py --repo ./feature-store --dry-run

# Incremental (to "now"):
python materialize_jobs/backfill_equities.py --repo ./feature-store --views eq_returns_1d --start 2024-01-01 --incremental --to-online
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# Feast
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
    if end <= start:
        return
    cur = start
    delta = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(end, cur + delta)
        yield cur, nxt
        cur = nxt

def human_range(a: datetime, b: datetime) -> str:
    return f"{a.strftime(ISO_FMT)} → {b.strftime(ISO_FMT)}"

def guess_equity_paths(store: FeatureStore) -> List[str]:
    """
    Best-effort scan of FileSources used by eq* feature views to show users where data is expected.
    """
    paths = []
    for name, obj in store.list_feature_views().items():
        if not isinstance(obj, FeatureView):
            continue
        if not name.startswith("eq_") and "equity" not in name:
            continue
        try:
            src = obj.source
            path = getattr(src, "path", None)
            if path and path not in paths:
                paths.append(path)
        except Exception:
            pass
    return paths

def sanity_check_sources(paths: List[str], sample: int = 1_000) -> None:
    """
    Lightweight check: ensure required columns exist in at least a sample of the files.
    Only for FileSource Parquet paths with wildcards; skip if not found.
    """
    import glob
    req_cols = {"ts", "ticker"}
    for p in paths:
        if not isinstance(p, str):
            continue
        if not (p.endswith(".parquet") or p.endswith("/*.parquet")):
            continue
        files = glob.glob(p) if not p.startswith(("s3://", "gs://", "gcs://")) else []
        if not files:
            # For cloud paths or no matches, skip silently (Feast will error if truly missing)
            continue
        f0 = files[0]
        try:
            df = pd.read_parquet(f0, columns=list(req_cols))
            missing = req_cols - set(df.columns)
            if missing:
                print(f"[WARN] {f0}: missing {sorted(missing)} (Feast may still work if FV schema doesn’t require them).")
        except Exception as e:
            print(f"[WARN] Could not read sample from {f0}: {e}")

# ----------------------------- job -------------------------------

def do_materialize(
    repo: str,
    view_names: Optional[List[str]],
    start: Optional[datetime],
    end: Optional[datetime],
    chunk_days: int,
    to_online: bool,
    incremental: bool,
    dry_run: bool,
    retries: int,
    sleep_sec: int,
):
    store = FeatureStore(repo_path=repo)

    # Resolve feature views
    fv_map = store.list_feature_views()
    if view_names:
        missing = [v for v in view_names if v not in fv_map]
        if missing:
            raise SystemExit(f"Feature views not found in repo: {missing}. Available: {sorted(fv_map.keys())}")
        fvs = [fv_map[v] for v in view_names]
    else:
        # default: all equity-relevant views
        fvs = [fv for name, fv in fv_map.items() if name.startswith("eq_") or "equity" in name.lower()]
        if not fvs:
            raise SystemExit("No equity feature views found. Pass --views or ensure your repo has eq_* views.")

    # Show sources + basic sanity checks
    eq_paths = guess_equity_paths(store)
    if eq_paths:
        print("[INFO] Equity FileSource paths:")
        for p in eq_paths:
            print(f"  - {p}")
        sanity_check_sources(eq_paths)

    # Time window logic
    now_utc = datetime.now(timezone.utc)
    if incremental:
        if not start:
            raise SystemExit("--incremental requires --start")
        end = now_utc
        print(f"[PLAN] materialize_incremental from {start.strftime(ISO_FMT)} to NOW ({end.strftime(ISO_FMT)})")
    else:
        if not (start and end):
            raise SystemExit("--start and --end required when not using --incremental")
        print(f"[PLAN] materialize in chunks of {chunk_days}d from {human_range(start, end)}")

    if dry_run:
        print("[DRY-RUN] No writes will be performed.")
        return

    # Execute
    if incremental:
        _materialize_with_retry(store, fvs, start, end, to_online, retries, sleep_sec, incremental=True)
        print("[DONE] incremental materialize complete.")
    else:
        for a, b in daterange_chunks(start, end, chunk_days):
            print(f"[RUN] chunk: {human_range(a, b)}")
            _materialize_with_retry(store, fvs, a, b, to_online, retries, sleep_sec, incremental=False)
        print("[DONE] windowed materialize complete.")

def _materialize_with_retry(
    store: FeatureStore,
    fvs: List[FeatureView],
    start: datetime,
    end: datetime,
    to_online: bool,
    retries: int,
    sleep_sec: int,
    incremental: bool,
):
    attempt = 0
    while True:
        try:
            if incremental:
                # Feast incremental pushes from start to now; end is ignored by the API but we log it
                if to_online:
                    store.materialize_incremental(end_date=end, feature_views=fvs)  # pushes offline->online for the range up to 'end'
                else:
                    # Offline-only: use materialize on a fixed [start, end)
                    store.materialize(start_date=start, end_date=end, feature_views=fvs)
            else:
                if to_online:
                    # Standard materialize loads offline, then you can separately run a materialize_incremental to push online.
                    store.materialize(start_date=start, end_date=end, feature_views=fvs)
                    store.materialize_incremental(end_date=end, feature_views=fvs)
                else:
                    store.materialize(start_date=start, end_date=end, feature_views=fvs)
            return
        except Exception as e:
            attempt += 1
            if attempt > retries:
                print(f"[ERROR] Failed after {retries} retries for {human_range(start, end)}: {e}")
                raise
            print(f"[WARN] Materialize error (attempt {attempt}/{retries}): {e}. Retrying in {sleep_sec}s…")
            time.sleep(sleep_sec)


# ----------------------------- CLI --------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser("Backfill equities feature views (Feast)")

    ap.add_argument("--repo", required=True, help="Path to Feast repo (folder with repo.yaml)")
    ap.add_argument("--views", nargs="*", default=None, help="Feature view names (default: all eq_* views)")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--chunk-days", type=int, default=30, help="Chunk window in days for non-incremental backfill")
    ap.add_argument("--to-online", action="store_true", help="Push to online store as well")
    ap.add_argument("--incremental", action="store_true", help="Use materialize_incremental to 'now' (requires --start)")

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep-sec", type=int, default=10)

    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    start = parse_date(args.start) if args.start else None
    end = parse_date(args.end) if args.end else None

    do_materialize(
        repo=args.repo,
        view_names=args.views,
        start=start,
        end=end,
        chunk_days=args.chunk_days,
        to_online=bool(args.to_online),
        incremental=bool(args.incremental),
        dry_run=bool(args.dry_run),
        retries=int(args.retries),
        sleep_sec=int(args.sleep_sec),
    )


if __name__ == "__main__":
    main()