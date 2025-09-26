# materialize_jobs/backfill_macro.py
"""
Backfill job for macro feature views using Feast.

What it does
------------
1) (Optional) prints/sanity-checks FileSource parquet paths for macro views
2) Runs Feast materialize / materialize_incremental in chunks
3) Can write offline-only, or also push to the online store
4) Supports dry-run, retries, and date chunking

Requirements
------------
pip install feast pandas pyarrow

Examples
--------
# Backfill last 5 years in 45d chunks for macro views (offline + online)
python materialize_jobs/backfill_macro.py \
  --repo ./feature-store \
  --views macro_rates macro_rates_derived \
  --start 2020-01-01 --end 2025-09-01 \
  --chunk-days 45 --to-online

# Dry-run (prints plan, validates sources, no writes)
python materialize_jobs/backfill_macro.py --repo ./feature-store --dry-run

# Incremental to "now" starting 2023-01-01
python materialize_jobs/backfill_macro.py \
  --repo ./feature-store \
  --views macro_rates \
  --start 2023-01-01 --incremental --to-online
"""

from __future__ import annotations

import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

import pandas as pd
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
    step = timedelta(days=chunk_days)
    while cur < end:
        nxt = min(end, cur + step)
        yield cur, nxt
        cur = nxt

def human_range(a: datetime, b: datetime) -> str:
    return f"{a.strftime(ISO_FMT)} → {b.strftime(ISO_FMT)}"

def guess_macro_paths(store: FeatureStore) -> List[str]:
    """
    Scan FileSources on macro_* views to hint where data is expected.
    """
    paths: List[str] = []
    for name, obj in store.list_feature_views().items():
        if not isinstance(obj, FeatureView):
            continue
        if not (name.startswith("macro_") or "macro" in name.lower()):
            continue
        try:
            src = obj.source
            p = getattr(src, "path", None)
            if isinstance(p, str) and p not in paths:
                paths.append(p)
        except Exception:
            pass
    return paths

def sanity_check_sources(paths: List[str], sample_cols=("ts", "country")) -> None:
    """
    Light check: read a local parquet sample for each path and verify columns.
    s3:// / gs:// globs are skipped (Feast will surface errors if missing).
    """
    import glob
    for p in paths:
        if not isinstance(p, str):
            continue
        if not (p.endswith(".parquet") or p.endswith("/*.parquet")):
            continue
        if p.startswith(("s3://", "gs://", "gcs://")):
            continue
        files = glob.glob(p)
        if not files:
            continue
        try:
            pd.read_parquet(files[0], columns=list(sample_cols))
        except Exception as e:
            print(f"[WARN] sample read failed for {files[0]}: {e}")

# ----------------------------- job core ---------------------------

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

    # Resolve macro feature views
    fv_map = store.list_feature_views()
    if view_names:
        missing = [v for v in view_names if v not in fv_map]
        if missing:
            raise SystemExit(f"Feature views not found: {missing}. Available: {sorted(fv_map.keys())}")
        fvs = [fv_map[v] for v in view_names]
    else:
        fvs = [fv for name, fv in fv_map.items() if name.startswith("macro_") or "macro" in name.lower()]
        if not fvs:
            raise SystemExit("No macro feature views found. Pass --views or define macro_* views in the repo.")

    # Show FileSource paths + quick sniff
    macro_paths = guess_macro_paths(store)
    if macro_paths:
        print("[INFO] Macro FileSource paths:")
        for p in macro_paths:
            print(f"  - {p}")
        sanity_check_sources(macro_paths, sample_cols=("ts", "country"))

    # Time plan
    now_utc = datetime.now(timezone.utc)
    if incremental:
        if not start:
            raise SystemExit("--incremental requires --start")
        end = now_utc
        print(f"[PLAN] materialize_incremental from {start.strftime(ISO_FMT)} to NOW ({end.strftime(ISO_FMT)})")
    else:
        if not (start and end):
            raise SystemExit("--start and --end required when not using --incremental")
        print(f"[PLAN] materialize in {chunk_days}d chunks over {human_range(start, end)}")

    if dry_run:
        print("[DRY-RUN] Skipping writes.")
        return

    # Execute
    if incremental:
        _materialize_with_retry(store, fvs, start, end, to_online, retries, sleep_sec, incremental=True)
        print("[DONE] incremental materialize.")
    else:
        for a, b in daterange_chunks(start, end, chunk_days):
            print(f"[RUN] chunk: {human_range(a, b)}")
            _materialize_with_retry(store, fvs, a, b, to_online, retries, sleep_sec, incremental=False)
        print("[DONE] windowed materialize.")

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
                if to_online:
                    store.materialize_incremental(end_date=end, feature_views=fvs)
                else:
                    store.materialize(start_date=start, end_date=end, feature_views=fvs)
            else:
                if to_online:
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
            print(f"[WARN] materialize attempt {attempt}/{retries} failed: {e}. Retrying in {sleep_sec}s…")
            time.sleep(sleep_sec)


# ----------------------------- CLI --------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser("Backfill macro feature views (Feast)")
    ap.add_argument("--repo", required=True, help="Path to Feast repo (folder containing repo.yaml)")
    ap.add_argument("--views", nargs="*", default=None, help="Feature view names (default: all macro_* views)")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--chunk-days", type=int, default=60, help="Macro can be chunked larger (default 60d)")
    ap.add_argument("--to-online", action="store_true")
    ap.add_argument("--incremental", action="store_true")
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
        chunk_days=int(args.chunk_days),
        to_online=bool(args.to_online),
        incremental=bool(args.incremental),
        dry_run=bool(args.dry_run),
        retries=int(args.retries),
        sleep_sec=int(args.sleep_sec),
    )

if __name__ == "__main__":
    main()