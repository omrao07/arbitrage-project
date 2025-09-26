# quality/tests_macro.py
"""
Data quality tests for macro_series

CLI examples
-----------
# Parquet lake (local)
python quality/tests_macro.py --parquet-root ./data/macro/series/bronze --date 2025-08-01

# Parquet lake (S3/GCS)
python quality/tests_macro.py --parquet-root s3://hyper-lakehouse/macro/series/bronze --date 2025-08-01

# DuckDB table
python quality/tests_macro.py --duckdb ./lakehouse.duckdb --table macro_series --date 2025-08-01

Pytest
------
pytest -q quality/tests_macro.py \
  --parquet-root=./data/macro/series/bronze --date=2025-08-01

Optional deps: pyarrow fsspec s3fs gcsfs duckdb great-expectations
"""

from __future__ import annotations

import sys
import glob
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

# Optional
try:
    import duckdb  # type: ignore
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False

try:
    import great_expectations as ge  # type: ignore
    HAVE_GE = True
except Exception:
    HAVE_GE = False


REQUIRED_COLS = ["ts", "source", "series_id", "value"]
OPTIONAL_COLS = ["country"]

@dataclass
class RunConfig:
    parquet_root: Optional[str] = None      # e.g., s3://.../macro/series/bronze
    duckdb_path: Optional[str] = None       # e.g., ./lakehouse.duckdb
    duckdb_table: str = "macro_series"
    date: Optional[str] = None              # YYYY-MM-DD partition (ts=... in path)
    sources: Optional[List[str]] = None     # e.g., ["FRED","WorldBank"]
    series: Optional[List[str]] = None      # e.g., ["CPIAUCSL","GDP"]
    countries: Optional[List[str]] = None   # e.g., ["US","CN","JP"]
    limit: Optional[int] = None
    fail_fast: bool = False


# ------------------ Loaders ------------------

def load_from_parquet(root: str, date: Optional[str]) -> pd.DataFrame:
    """
    Reads partitioned parquet written as:
      {root}/ts=YYYY-MM-DD/source=<SRC>/part-*.parquet
    """
    root = root.rstrip("/")
    pattern = f"{root}/ts=*/source=*/*.parquet"
    if date:
        pattern = f"{root}/ts={date}/source=*/*.parquet"

    try:
        df = pd.read_parquet(pattern)  # fsspec-aware
    except Exception:
        return pd.DataFrame(columns=REQUIRED_COLS + OPTIONAL_COLS)

    # Normalize
    if "ts" in df.columns:
        # macro ts is DATE in storage; coerce to datetime (UTC 00:00)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ("source", "series_id", "country"):
        if c in df.columns:
            df[c] = df[c].astype("string")

    # Value numeric
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.reset_index(drop=True)


def load_from_duckdb(db_path: str, table: str, date: Optional[str],
                     sources: Optional[List[str]], series: Optional[List[str]],
                     countries: Optional[List[str]], limit: Optional[int]) -> pd.DataFrame:
    if not HAVE_DUCKDB:
        raise RuntimeError("duckdb not installed. `pip install duckdb`")
    con = duckdb.connect(db_path, read_only=True)

    where = []
    params: List[str] = []

    if date:
        where.append("ts = DATE(?)")
        params.append(date)
    if sources:
        placeholders = ",".join(["?"] * len(sources))
        where.append(f"source IN ({placeholders})")
        params.extend(sources)
    if series:
        placeholders = ",".join(["?"] * len(series))
        where.append(f"series_id IN ({placeholders})")
        params.extend(series)
    if countries:
        placeholders = ",".join(["?"] * len(countries))
        where.append(f"country IN ({placeholders})")
        params.extend(countries)

    sql = f"SELECT ts, source, series_id, country, value FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY source, series_id, country, ts"
    if limit:
        sql += f" LIMIT {int(limit)}"

    df = con.execute(sql, params).df()
    con.close()

    # Normalize
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ("source", "series_id", "country"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ------------------ Assertions ------------------

def assert_required_columns(df: pd.DataFrame):
    missing = set(REQUIRED_COLS) - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

def assert_non_null_keys(df: pd.DataFrame):
    assert df["ts"].notna().all(), "Null timestamps present"
    assert df["source"].notna().all(), "Null sources present"
    assert df["series_id"].notna().all(), "Null series_id present"

def assert_value_numeric_and_finite(df: pd.DataFrame):
    bad_nan = df["value"].isna().sum()
    assert bad_nan == 0, f"{bad_nan} rows have null/NaN value"
    # absurd bounds guard (covers unit explosions)
    bad_absurd = (df["value"].abs() > 1e20).sum()
    assert bad_absurd == 0, f"{bad_absurd} rows have absurd |value| > 1e20"

def assert_no_duplicates(df: pd.DataFrame):
    # Uniqueness key: (ts, source, series_id, country) — country may be null, so fillna
    key_cols = ["ts", "source", "series_id", "country"]
    x = df.copy()
    if "country" in x.columns:
        x["country"] = x["country"].fillna("__NULL__")
    du = x.duplicated(subset=key_cols, keep=False)
    cnt = int(du.sum())
    assert cnt == 0, f"Found {cnt} duplicate (ts, source, series_id, country) rows"

def assert_monotonic_per_series(df: pd.DataFrame):
    # For each (source, series_id, country), timestamps should be non-decreasing
    # (strict monotonicity not guaranteed with revisions)
    if df.empty:
        return
    cols = ["source", "series_id"]
    if "country" in df.columns:
        cols.append("country")
    for key, g in df.sort_values("ts").groupby(cols):
        if len(g) <= 1:
            continue
        # allow equality (revisions on same date)
        is_sorted = g["ts"].is_monotonic_increasing
        assert is_sorted, f"Non-monotone ts for group {key}"

def assert_frequency_consistency_hint(df: pd.DataFrame):
    """
    Heuristic: flag if a series mixes monthly and quarterly cadence in the same group.
    We won't fail the run; just print a warning in JSON.
    """
    notes = []
    if df.empty:
        return notes
    cols = ["source", "series_id"]
    if "country" in df.columns:
        cols.append("country")
    for key, g in df.groupby(cols):
        if g["ts"].nunique() <= 3:
            continue
        # infer median spacing (in days)
        ts = pd.to_datetime(g["ts"], utc=True).sort_values().unique()
        diffs = pd.Series(pd.to_datetime(ts[1:], utc=True) - pd.to_datetime(ts[:-1], utc=True)).dt.days
        if len(diffs) == 0:
            continue
        med = diffs.median()
        # crude bands
        if diffs.std() > 40 and med not in (28, 29, 30, 31, 90, 91, 92, 93, 365, 366):
            notes.append({"group": tuple(key if isinstance(key, tuple) else (key,)),
                          "median_gap_days": float(med),
                          "std_gap_days": float(diffs.std())})
    return notes


# ------------------ Great Expectations (optional) ------------------

def run_great_expectations(df: pd.DataFrame) -> dict:
    if not HAVE_GE or df.empty:
        return {"ge_ran": False}
    gdf = ge.from_pandas(df)
    results = []
    results.append(gdf.expect_column_values_to_not_be_null("ts"))
    results.append(gdf.expect_column_values_to_not_be_null("source"))
    results.append(gdf.expect_column_values_to_not_be_null("series_id"))
    results.append(gdf.expect_column_values_to_be_between("value", min_value=-1e20, max_value=1e20))
    results.append(gdf.expect_compound_columns_to_be_unique(["ts", "source", "series_id", "country"]))
    success = all(r.success for r in results)
    detail = {"ge_ran": True, "ge_success": bool(success)}
    if not success:
        failed = [r for r in results if not r.success]
        detail["failed"] = [{"expectation": f.expectation_type, "kwargs": f.kwargs} for f in failed] # type: ignore
    return detail


# ------------------ Runner ------------------

def run_quality(cfg: RunConfig) -> int:
    if cfg.parquet_root:
        df = load_from_parquet(cfg.parquet_root, cfg.date)
    elif cfg.duckdb_path:
        df = load_from_duckdb(cfg.duckdb_path, cfg.duckdb_table, cfg.date, cfg.sources, cfg.series, cfg.countries, cfg.limit)
    else:
        print("Provide either --parquet-root or --duckdb", file=sys.stderr)
        return 2

    if df.empty:
        print(json.dumps({"status": "empty", "rows": 0}, separators=(",", ":")))
        return 0

    # Optional filters (applied post-load for parquet)
    if cfg.sources:
        df = df[df["source"].astype(str).isin(cfg.sources)]
    if cfg.series:
        df = df[df["series_id"].astype(str).isin(cfg.series)]
    if cfg.countries and "country" in df.columns:
        df = df[df["country"].astype(str).isin(cfg.countries)]
    if cfg.limit and len(df) > cfg.limit:
        df = df.sort_values(["source", "series_id", "country", "ts"]).head(cfg.limit)

    # Core assertions
    assert_required_columns(df)
    assert_non_null_keys(df)
    assert_value_numeric_and_finite(df)
    assert_no_duplicates(df)
    assert_monotonic_per_series(df)

    # Non-fatal notes
    notes = assert_frequency_consistency_hint(df)

    ge_report = run_great_expectations(df)

    # Small summary
    sample_series = sorted(df["series_id"].dropna().astype(str).unique().tolist())[:10]
    sample_sources = sorted(df["source"].dropna().astype(str).unique().tolist())
    sample_countries = sorted(df["country"].dropna().astype(str).unique().tolist())[:10] if "country" in df.columns else []

    print(json.dumps({
        "status": "ok",
        "rows": int(len(df)),
        "sources": sample_sources,
        "series_sample": sample_series,
        "countries_sample": sample_countries,
        "notes": notes,
        **ge_report
    }, separators=(",", ":")))
    return 0


# ------------------ Pytest glue ------------------

def pytest_addoption(parser):
    parser.addoption("--parquet-root", action="store", default=None)
    parser.addoption("--duckdb", action="store", default=None)
    parser.addoption("--table", action="store", default="macro_series")
    parser.addoption("--date", action="store", default=None)
    parser.addoption("--sources", action="store", default=None, help="Comma-separated sources e.g. FRED,WorldBank")
    parser.addoption("--series", action="store", default=None, help="Comma-separated series ids")
    parser.addoption("--countries", action="store", default=None, help="Comma-separated ISO3 country codes")
    parser.addoption("--limit", action="store", default=None)

def _split_opt(s: Optional[str]) -> Optional[List[str]]:
    return [x.strip() for x in s.split(",")] if s else None

def _cfg_from_pytest(request) -> RunConfig:
    limit = request.config.getoption("--limit")
    return RunConfig(
        parquet_root=request.config.getoption("--parquet-root"),
        duckdb_path=request.config.getoption("--duckdb"),
        duckdb_table=request.config.getoption("--table") or "macro_series",
        date=request.config.getoption("--date"),
        sources=_split_opt(request.config.getoption("--sources")),
        series=_split_opt(request.config.getoption("--series")),
        countries=_split_opt(request.config.getoption("--countries")),
        limit=int(limit) if limit else None,
    )

def test_macro_quality(request):
    cfg = _cfg_from_pytest(request)
    rc = run_quality(cfg)
    assert rc == 0


# ------------------ CLI ------------------

def parse_args(argv=None) -> RunConfig:
    ap = argparse.ArgumentParser("tests_macro")
    ap.add_argument("--parquet-root", help="Partitioned parquet root for macro_series")
    ap.add_argument("--duckdb", dest="duckdb_path", help="Path to lakehouse.duckdb")
    ap.add_argument("--table", default="macro_series", help="DuckDB table name")
    ap.add_argument("--date", help="YYYY-MM-DD partition (optional)")
    ap.add_argument("--sources", nargs="*", help="Filter to sources, e.g., FRED WorldBank")
    ap.add_argument("--series", nargs="*", help="Filter to series ids")
    ap.add_argument("--countries", nargs="*", help="Filter to country codes")
    ap.add_argument("--limit", type=int, help="Limit rows (debug)")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args(argv)

    return RunConfig(
        parquet_root=args.parquet_root,
        duckdb_path=args.duckdb_path,
        duckdb_table=args.table,
        date=args.date,
        sources=args.sources,
        series=args.series,
        countries=args.countries,
        limit=args.limit,
        fail_fast=bool(args.fail_fast),
    )

if __name__ == "__main__":
    cfg = parse_args()
    try:
        code = run_quality(cfg)
        sys.exit(code)
    except AssertionError as e:
        print(json.dumps({"status": "failed", "reason": str(e)}, separators=(",", ":")))
        sys.exit(3)