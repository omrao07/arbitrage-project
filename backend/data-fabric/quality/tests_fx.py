# quality/tests_fx.py
"""
Data quality tests for fx_rates

CLI examples
-----------
# Parquet lake (local)
python quality/tests_fx.py --parquet-root ./data/fx/rates/bronze --date 2025-09-01

# Parquet lake (S3)
python quality/tests_fx.py --parquet-root s3://hyper-lakehouse/fx/rates/bronze --date 2025-09-01

# DuckDB table
python quality/tests_fx.py --duckdb ./lakehouse.duckdb --table fx_rates --date 2025-09-01

Pytest
------
pytest -q quality/tests_fx.py --parquet-root=./data/fx/rates/bronze --date=2025-09-01

Optional deps: pyarrow fsspec s3fs gcsfs duckdb great-expectations
"""

from __future__ import annotations

import os
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


REQUIRED_COLS = ["ts", "base", "quote", "rate"]
OPTIONAL_COLS = ["source"]

@dataclass
class RunConfig:
    parquet_root: Optional[str] = None
    duckdb_path: Optional[str] = None
    duckdb_table: str = "fx_rates"
    date: Optional[str] = None              # YYYY-MM-DD (partition)
    pairs: Optional[List[str]] = None       # e.g. ["EURUSD","USDJPY"]
    limit: Optional[int] = None
    fail_fast: bool = False


# ------------- Loaders -------------

def _pair_from_row(base: str, quote: str) -> str:
    return f"{str(base).upper()}{str(quote).upper()}"

def load_from_parquet(root: str, date: Optional[str], pairs: Optional[List[str]], limit: Optional[int]) -> pd.DataFrame:
    """
    Reads partitions written as:
      {root}/date=YYYY-MM-DD/base=USD/quote=JPY/part-*.parquet
    """
    root = root.rstrip("/")
    pattern = f"{root}/date=*/base=*/quote=*/*.parquet"
    if date:
        pattern = f"{root}/date={date}/base=*/quote=*/*.parquet"

    # For local paths, resolve to a concrete file list to avoid memory surprises
    if pattern.startswith(("s3://", "gcs://")):
        # Let pandas + fsspec expand the glob internally
        df = pd.read_parquet(pattern)
    else:
        files = glob.glob(pattern)
        if not files:
            return pd.DataFrame(columns=REQUIRED_COLS + OPTIONAL_COLS)
        df = pd.read_parquet(files) # type: ignore

    # Normalize types
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "base" in df.columns:
        df["base"] = df["base"].astype(str).str.upper()
    if "quote" in df.columns:
        df["quote"] = df["quote"].astype(str).str.upper()

    # Filter by requested pairs if provided
    if pairs:
        want = {p.upper().replace("=X", "").replace("_", "").replace("-", "") for p in pairs}
        have = (_pair_from_row(df["base"], df["quote"]) if isinstance(df["base"], str) # type: ignore
                else (df["base"].astype(str) + df["quote"].astype(str)))
        df = df[have.isin(want)] # type: ignore

    if limit and len(df) > limit:
        df = df.sort_values(["base", "quote", "ts"]).head(limit)

    return df.reset_index(drop=True)

def load_from_duckdb(db_path: str, table: str, date: Optional[str], pairs: Optional[List[str]], limit: Optional[int]) -> pd.DataFrame:
    if not HAVE_DUCKDB:
        raise RuntimeError("duckdb not installed. `pip install duckdb`")
    con = duckdb.connect(db_path, read_only=True)

    where = []
    params: List[str] = []
    if date:
        where.append("DATE(ts) = DATE(?)")
        params.append(date)
    if pairs:
        # Convert pairs into (base, quote) tuples; accept formats like EURUSD, EUR/USD, EURUSD=X
        bqs = []
        for p in pairs:
            s = p.upper().replace("=X", "").replace("/", "").replace("_", "").replace("-", "")
            bqs.append((s[:3], s[3:]))
        # Build (base,quote) IN ((?,?),(?,?),...)
        tuples = ",".join(["(?,?)"] * len(bqs))
        where.append(f"(upper(base), upper(quote)) IN ({tuples})")
        for b, q in bqs:
            params.extend([b, q])

    sql = f"SELECT ts, base, quote, rate, source FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY base, quote, ts"
    if limit:
        sql += f" LIMIT {int(limit)}"

    df = con.execute(sql, params).df()
    con.close()

    # Normalize
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["base"] = df["base"].astype(str).str.upper()
    df["quote"] = df["quote"].astype(str).str.upper()
    return df


# ------------- Assertions -------------

def assert_required_columns(df: pd.DataFrame):
    missing = set(REQUIRED_COLS) - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

def assert_non_null_keys(df: pd.DataFrame):
    assert df["ts"].notna().all(), "Null timestamps present"
    assert df["base"].notna().all() and df["quote"].notna().all(), "Null base/quote present"

def assert_positive_rates(df: pd.DataFrame):
    bad = (pd.to_numeric(df["rate"], errors="coerce") <= 0).sum()
    assert bad == 0, f"{bad} rows have non-positive FX rate"

def assert_monotonic_per_pair(df: pd.DataFrame):
    for (b, q), g in df.groupby(["base", "quote"]):
        if len(g) <= 1:
            continue
        assert g["ts"].is_monotonic_increasing, f"Timestamps not increasing for {b}{q}"

def assert_no_duplicates(df: pd.DataFrame):
    du = df.duplicated(subset=["ts", "base", "quote"], keep=False)
    cnt = int(du.sum())
    assert cnt == 0, f"Found {cnt} duplicate (ts, base, quote) rows"

def assert_sanity_inversion(df: pd.DataFrame):
    """
    Lightweight arbitrage sanity: if same timestamp has both A/B and B/A,
    their product should be ~1 within a generous tolerance (0.5%).
    """
    if df.empty:
        return
    # build pivot of pairs at same ts
    ab = df.copy()
    ab["pair"] = ab["base"] + ab["quote"]
    # join with inverted pair
    inv = ab.rename(columns={"base": "quote", "quote": "base", "rate": "inv_rate"})
    merged = pd.merge(
        ab[["ts", "base", "quote", "rate"]],
        inv[["ts", "base", "quote", "inv_rate"]],
        on=["ts", "base", "quote"],
        how="inner",
        suffixes=("", "_dup"),
    )
    if merged.empty:
        return
    prod = (pd.to_numeric(merged["rate"], errors="coerce") *
            pd.to_numeric(merged["inv_rate"], errors="coerce"))
    bad = ((prod < 0.995) | (prod > 1.005)).sum()
    assert bad == 0, f"{bad} rows violate inversion sanity (rate * inv_rate ≈ 1)"

# ------------- Great Expectations (optional) -------------

def run_great_expectations(df: pd.DataFrame) -> dict:
    if not HAVE_GE or df.empty:
        return {"ge_ran": False}
    gdf = ge.from_pandas(df)
    results = []
    results.append(gdf.expect_column_values_to_not_be_null("ts"))
    results.append(gdf.expect_column_values_to_not_be_null("base"))
    results.append(gdf.expect_column_values_to_not_be_null("quote"))
    results.append(gdf.expect_column_values_to_be_between("rate", min_value=1e-6, mostly=1.0))
    results.append(gdf.expect_compound_columns_to_be_unique(["ts", "base", "quote"]))
    success = all(r.success for r in results)
    detail = {"ge_ran": True, "ge_success": bool(success)}
    if not success:
        failed = [r for r in results if not r.success]
        detail["failed"] = [{"expectation": f.expectation_type, "kwargs": f.kwargs} for f in failed] # type: ignore
    return detail


# ------------- Runner -------------

def run_quality(cfg: RunConfig) -> int:
    if cfg.parquet_root:
        df = load_from_parquet(cfg.parquet_root, cfg.date, cfg.pairs, cfg.limit)
    elif cfg.duckdb_path:
        df = load_from_duckdb(cfg.duckdb_path, cfg.duckdb_table, cfg.date, cfg.pairs, cfg.limit)
    else:
        print("Provide either --parquet-root or --duckdb", file=sys.stderr)
        return 2

    if df.empty:
        print(json.dumps({"status": "empty", "rows": 0}, separators=(",", ":")))
        return 0

    assert_required_columns(df)
    assert_non_null_keys(df)
    assert_positive_rates(df)
    assert_monotonic_per_pair(df)
    assert_no_duplicates(df)
    # inversion sanity is optional (depends on both directions present)
    try:
        assert_sanity_inversion(df)
    except AssertionError as e:
        # don't fail the whole run for missing opposite quotes; only fail if explicit violations
        if "violate inversion sanity" in str(e):
            raise

    ge_report = run_great_expectations(df)

    print(json.dumps({
        "status": "ok",
        "rows": int(len(df)),
        "pairs": sorted({f"{b}{q}" for b, q in df[["base","quote"]].drop_duplicates().itertuples(index=False)} )[:10],
        **ge_report
    }, separators=(",", ":")))
    return 0


# ------------- Pytest glue -------------

def pytest_addoption(parser):
    parser.addoption("--parquet-root", action="store", default=None)
    parser.addoption("--duckdb", action="store", default=None)
    parser.addoption("--table", action="store", default="fx_rates")
    parser.addoption("--date", action="store", default=None)
    parser.addoption("--pairs", action="store", default=None, help="Comma-separated pairs like EURUSD,USDJPY")
    parser.addoption("--limit", action="store", default=None)

def _cfg_from_pytest(request) -> RunConfig:
    pairs = request.config.getoption("--pairs")
    pairs_list = [p.strip() for p in pairs.split(",")] if pairs else None
    limit = request.config.getoption("--limit")
    return RunConfig(
        parquet_root=request.config.getoption("--parquet-root"),
        duckdb_path=request.config.getoption("--duckdb"),
        duckdb_table=request.config.getoption("--table") or "fx_rates",
        date=request.config.getoption("--date"),
        pairs=pairs_list,
        limit=int(limit) if limit else None,
    )

def test_fx_quality(request):
    cfg = _cfg_from_pytest(request)
    rc = run_quality(cfg)
    assert rc == 0


# ------------- CLI -------------

def parse_args(argv=None) -> RunConfig:
    ap = argparse.ArgumentParser("tests_fx")
    ap.add_argument("--parquet-root", help="Partitioned parquet root for fx_rates")
    ap.add_argument("--duckdb", dest="duckdb_path", help="Path to lakehouse.duckdb")
    ap.add_argument("--table", default="fx_rates", help="DuckDB table name")
    ap.add_argument("--date", help="YYYY-MM-DD partition (optional)")
    ap.add_argument("--pairs", nargs="*", help="Filter to pairs (EURUSD, USDJPY, ...)")
    ap.add_argument("--limit", type=int, help="Limit rows (debug)")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args(argv)

    return RunConfig(
        parquet_root=args.parquet_root,
        duckdb_path=args.duckdb_path,
        duckdb_table=args.table,
        date=args.date,
        pairs=args.pairs,
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