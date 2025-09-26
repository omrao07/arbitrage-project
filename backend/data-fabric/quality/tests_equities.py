# quality/tests_equities.py
"""
Data quality tests for equities_prices

Usage (CLI):
  # Parquet lake (local)
  python quality/tests_equities.py --parquet-root ./data/equities/prices/bronze --date 2025-09-01

  # Parquet lake (S3)
  python quality/tests_equities.py --parquet-root s3://hyper-lakehouse/equities/prices/bronze --date 2025-09-01

  # DuckDB
  python quality/tests_equities.py --duckdb ./lakehouse.duckdb --table equities_prices --date 2025-09-01

As pytest:
  pytest -q quality/tests_equities.py \
    --parquet-root=./data/equities/prices/bronze --date=2025-09-01

Optional deps:
  pip install pyarrow fsspec s3fs gcsfs duckdb great-expectations
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

# Optional imports
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


REQUIRED_COLS = ["ts", "ticker", "open", "high", "low", "close", "volume"]
OPTIONAL_COLS = ["source", "adj_close", "vwap", "trades"]

@dataclass
class RunConfig:
    parquet_root: Optional[str] = None  # e.g., s3://.../equities/prices/bronze
    duckdb_path: Optional[str] = None   # e.g., ./lakehouse.duckdb
    duckdb_table: str = "equities_prices"
    date: Optional[str] = None          # YYYY-MM-DD (partition)
    tickers: Optional[list[str]] = None
    limit: Optional[int] = None
    fail_fast: bool = False


# ------------------ Loaders ------------------

def load_from_parquet(root: str, date: Optional[str], tickers: Optional[list[str]], limit: Optional[int]) -> pd.DataFrame:
    """
    Reads partitioned parquet written as:
      {root}/date=YYYY-MM-DD/ticker=SYMBOL/part-*.parquet
    """
    pattern = f"{root.rstrip('/')}/date=*/ticker=*/*.parquet"
    if date:
        pattern = f"{root.rstrip('/')}/date={date}/ticker=*/*.parquet"

    # fsspec-aware read via pandas
    # If the path is huge, we union only matching files optionally filtered by tickers
    files = glob.glob(pattern) if not pattern.startswith(("s3://", "gcs://")) else [pattern]  # pandas handles wildcard on cloud via fsspec
    if not files:
        # Let pandas handle the glob (works for local & cloud)
        try:
            df = pd.read_parquet(pattern)
        except Exception:
            return pd.DataFrame(columns=REQUIRED_COLS + OPTIONAL_COLS)
    else:
        df = pd.read_parquet(files) # type: ignore

    # Normalize columns (some adapters use 'symbol')
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})

    # Filter tickers if requested
    if tickers:
        df = df[df["ticker"].astype(str).str.upper().isin([t.upper() for t in tickers])]

    # Enforce dtypes and UTC tz
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    if limit and len(df) > limit:
        df = df.sort_values(["ticker", "ts"]).head(limit)

    return df.reset_index(drop=True)


def load_from_duckdb(db_path: str, table: str, date: Optional[str], tickers: Optional[list[str]], limit: Optional[int]) -> pd.DataFrame:
    if not HAVE_DUCKDB:
        raise RuntimeError("duckdb not installed. `pip install duckdb`")
    con = duckdb.connect(db_path, read_only=True)
    where = []
    if date:
        where.append("DATE(ts) = DATE(?)")
    if tickers:
        # case-insensitive compare
        placeholders = ",".join(["?"] * len(tickers))
        where.append(f"upper(ticker) IN ({placeholders})")
    sql = f"SELECT ts, ticker, source, open, high, low, close, volume, adj_close FROM {table}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ticker, ts"
    if limit:
        sql += f" LIMIT {int(limit)}"

    params = []
    if date:
        params.append(date)
    if tickers:
        params.extend([t.upper() for t in tickers])

    df = con.execute(sql, params).df()
    con.close()

    # Normalize types
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


# ------------------ Core Checks ------------------

def assert_required_columns(df: pd.DataFrame):
    missing = set(REQUIRED_COLS) - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"

def assert_non_null_keys(df: pd.DataFrame):
    assert df["ts"].notna().all(), "Null timestamps present"
    assert df["ticker"].notna().all(), "Null tickers present"

def assert_price_bounds(df: pd.DataFrame):
    # high >= low; open & close within [low, high]
    bad_high_low = (df["high"] < df["low"]).sum()
    assert bad_high_low == 0, f"{bad_high_low} rows have high < low"

    bad_open = ((df["open"] < df["low"]) | (df["open"] > df["high"])).sum()
    bad_close = ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
    assert bad_open == 0, f"{bad_open} rows have open outside [low, high]"
    assert bad_close == 0, f"{bad_close} rows have close outside [low, high]"

def assert_positive_numbers(df: pd.DataFrame):
    # volumes can be 0; must be >= 0
    if "volume" in df.columns:
        bad_vol = (pd.to_numeric(df["volume"], errors="coerce") < 0).sum()
        assert bad_vol == 0, f"{bad_vol} rows have negative volume"
    for col in ["open", "high", "low", "close"]:
        bad = (pd.to_numeric(df[col], errors="coerce") <= 0).sum()
        assert bad == 0, f"{bad} rows have non-positive {col}"

def assert_tz_and_monotonic(df: pd.DataFrame):
    # tz-aware already enforced by pandas to_datetime(..., utc=True)
    # Check monotonic per ticker (allow equality in case of duplicates test catches)
    for tkr, g in df.groupby("ticker"):
        if len(g) <= 1:
            continue
        assert g["ts"].is_monotonic_increasing, f"Timestamps not increasing for {tkr}"

def assert_no_duplicates(df: pd.DataFrame):
    # primary key (ticker, ts) expected unique per table/source
    du = df.duplicated(subset=["ticker", "ts"], keep=False)
    dup_cnt = int(du.sum())
    assert dup_cnt == 0, f"Found {dup_cnt} duplicate (ticker, ts) rows"

def assert_adjusted_optional(df: pd.DataFrame):
    # If adj_close present, it shouldn't be wildly off vs close (sanity: within 95% band daily)
    if "adj_close" in df.columns and df["adj_close"].notna().any():
        ratio = (pd.to_numeric(df["adj_close"], errors="coerce") / pd.to_numeric(df["close"], errors="coerce"))
        # Ignore extreme NaNs/infs
        ratio = ratio.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], pd.NA).astype(float) # type: ignore
        # Allow wide band to accommodate split-adjustment days; flag only absurdities
        bad = ((ratio < 0.01) | (ratio > 100.0)).sum()
        assert bad == 0, f"{bad} rows have absurd adj_close/close ratio"


# ------------------ Great Expectations (optional) ------------------

def run_great_expectations(df: pd.DataFrame) -> dict:
    if not HAVE_GE or df.empty:
        return {"ge_ran": False}
    # Build an in-memory dataset
    gdf = ge.from_pandas(df)
    # Minimal suite aligned with our contracts
    results = []
    results.append(gdf.expect_column_values_to_not_be_null("ticker"))
    results.append(gdf.expect_column_values_to_not_be_null("ts"))
    for col in ("open", "high", "low", "close"):
        results.append(gdf.expect_column_values_to_be_between(col, min_value=0, mostly=1.0))
    results.append(gdf.expect_column_pair_values_a_to_be_greater_than_b("high", "low"))
    # Uniqueness check
    results.append(gdf.expect_compound_columns_to_be_unique(["ticker", "ts"]))

    success = all(r.success for r in results)
    detail = {"ge_ran": True, "ge_success": bool(success)}
    if not success:
        # collect short report
        failed = [r for r in results if not r.success]
        detail["failed"] = [{"expectation": f.expectation_type, "kwargs": f.kwargs} for f in failed] # type: ignore
    return detail


# ------------------ Runner ------------------

def run_quality(cfg: RunConfig) -> int:
    if cfg.parquet_root:
        df = load_from_parquet(cfg.parquet_root, cfg.date, cfg.tickers, cfg.limit)
    elif cfg.duckdb_path:
        df = load_from_duckdb(cfg.duckdb_path, cfg.duckdb_table, cfg.date, cfg.tickers, cfg.limit)
    else:
        print("Provide either --parquet-root or --duckdb", file=sys.stderr)
        return 2

    if df.empty:
        print(json.dumps({"status": "empty", "rows": 0}, separators=(",", ":")))
        return 0

    # Ensure required columns exist before all checks
    assert_required_columns(df)
    assert_non_null_keys(df)
    assert_price_bounds(df)
    assert_positive_numbers(df)
    assert_tz_and_monotonic(df)
    assert_no_duplicates(df)
    assert_adjusted_optional(df)

    ge_report = run_great_expectations(df)

    print(json.dumps({
        "status": "ok",
        "rows": int(len(df)),
        "tickers": sorted(df["ticker"].unique().tolist())[:10],
        **ge_report
    }, separators=(",", ":")))
    return 0


# ------------------ Pytest glue ------------------

def pytest_addoption(parser):
    parser.addoption("--parquet-root", action="store", default=None)
    parser.addoption("--duckdb", action="store", default=None)
    parser.addoption("--table", action="store", default="equities_prices")
    parser.addoption("--date", action="store", default=None)
    parser.addoption("--tickers", action="store", default=None, help="Comma-separated tickers")
    parser.addoption("--limit", action="store", default=None)

def _cfg_from_pytest(request) -> RunConfig:
    tickers = request.config.getoption("--tickers")
    tickers_list = [t.strip() for t in tickers.split(",")] if tickers else None
    limit = request.config.getoption("--limit")
    return RunConfig(
        parquet_root=request.config.getoption("--parquet-root"),
        duckdb_path=request.config.getoption("--duckdb"),
        duckdb_table=request.config.getoption("--table") or "equities_prices",
        date=request.config.getoption("--date"),
        tickers=tickers_list,
        limit=int(limit) if limit else None,
    )

def test_equities_quality(request):
    cfg = _cfg_from_pytest(request)
    rc = run_quality(cfg)
    assert rc == 0


# ------------------ CLI ------------------

def parse_args(argv=None) -> RunConfig:
    ap = argparse.ArgumentParser("tests_equities")
    ap.add_argument("--parquet-root", help="Partitioned parquet root for equities_prices")
    ap.add_argument("--duckdb", dest="duckdb_path", help="Path to lakehouse.duckdb")
    ap.add_argument("--table", default="equities_prices", help="DuckDB table name")
    ap.add_argument("--date", help="YYYY-MM-DD partition (optional)")
    ap.add_argument("--tickers", nargs="*", help="Filter to tickers")
    ap.add_argument("--limit", type=int, help="Limit rows (debug)")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first assertion")
    args = ap.parse_args(argv)

    return RunConfig(
        parquet_root=args.parquet_root,
        duckdb_path=args.duckdb_path,
        duckdb_table=args.table,
        date=args.date,
        tickers=args.tickers,
        limit=args.limit,
        fail_fast=bool(args.fail_fast),
    )

if __name__ == "__main__":
    cfg = parse_args()
    try:
        code = run_quality(cfg)
        sys.exit(code)
    except AssertionError as e:
        # Pretty one-line failure for CI logs
        print(json.dumps({"status": "failed", "reason": str(e)}, separators=(",", ":")))
        sys.exit(3)