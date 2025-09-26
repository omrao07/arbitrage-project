# data-fabric/pipelines/equities_pipeline.py
"""
Equities pipeline (bars → lakehouse)

Features
- Pulls OHLCV bars from Polygon (via alt-adapters/equities_polygon.py)
- Symbols from CLI or file (CSV/JSON/TXT) or env
- Date window & interval controls
- Great-Expectations-lite validations
- Partitioned Parquet writer (local or S3/GCS via fsspec)
- Optional DuckDB load into equities_prices table (lakehouse.sql)
- Idempotent checkpoints (.checkpoint/) to skip done work
- Structured logging

Env (examples)
  POLYGON_API_KEY=xxxx
  OUTPUT_ROOT=s3://hyper-lakehouse/equities/prices/bronze
  DUCKDB_PATH=/data/lakehouse.duckdb
  MAX_WORKERS=6

Usage
  python equities_pipeline.py \
    --symbols AAPL MSFT NVDA \
    --start 2024-01-01 --end 2024-03-01 \
    --timespan day --multiplier 1 \
    --write parquet --load-duckdb

  python equities_pipeline.py \
    --symbols-file symbols.txt \
    --start 2023-01-01 --end 2023-12-31 \
    --output-root s3://bucket/equities/prices/bronze
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import glob
import hashlib
import typing as T
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Optional I/O engines
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False

# fsspec lets us write to s3://, gcs://, etc. seamlessly
try:
    import fsspec  # noqa: F401
    HAVE_FSSPEC = True
except Exception:
    HAVE_FSSPEC = False

# Our Polygon adapter
try:
    from data_fabric.alt_adapters.equities_polygon import BarParams, fetch_bars  # type: ignore
except Exception:
    # fallback relative import for typical project layout
    from alt_adapters.equities_polygon import BarParams, fetch_bars  # type: ignore


# --------------------------- Logging ---------------------------

def log(msg: str, **kv):
    now = datetime.now(timezone.utc).isoformat()
    if kv:
        print(json.dumps({"ts": now, "msg": msg, **kv}, separators=(",", ":")))
    else:
        print(f"{now} | {msg}")


# --------------------------- Config ----------------------------

@dataclass
class PipelineConfig:
    symbols: list[str]
    start: str
    end: str
    timespan: str = "day"       # minute|hour|day|week|month|quarter|year
    multiplier: int = 1
    adjusted: bool = True

    output_root: str = os.getenv("OUTPUT_ROOT", "./data/equities/prices/bronze")
    tmp_dir: str = "./.tmp/equities"
    checkpoint_dir: str = "./.checkpoint/equities"

    write_format: str = "parquet"  # parquet|csv
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    batch_bytes_target: int = 128 * 1024 * 1024  # rotate parquet every ~128MB

    # Optional DuckDB load
    load_duckdb: bool = False
    duckdb_path: str = os.getenv("DUCKDB_PATH", "./lakehouse.duckdb")
    duckdb_table: str = "equities_prices"  # matches catalog/lakehouse.sql

    # Data quality toggles
    validate: bool = True


# ------------------------- Utilities ---------------------------

def ensure_dir(path: str):
    if path.startswith("s3://") or path.startswith("gcs://"):
        # fsspec handles directories implicitly; no mkdir
        return
    os.makedirs(path, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def to_date(dt: pd.Timestamp | str) -> str:
    return pd.to_datetime(dt, utc=True).date().isoformat()

def checkpoint_path(cfg: PipelineConfig, symbol: str) -> str:
    ensure_dir(cfg.checkpoint_dir)
    key = sha1("|".join([symbol.upper(), cfg.start, cfg.end, cfg.timespan, str(cfg.multiplier), str(cfg.adjusted)]))
    return os.path.join(cfg.checkpoint_dir, f"{symbol.upper()}_{key}.done")

def is_done(cfg: PipelineConfig, symbol: str) -> bool:
    return os.path.exists(checkpoint_path(cfg, symbol))

def mark_done(cfg: PipelineConfig, symbol: str):
    p = checkpoint_path(cfg, symbol)
    ensure_dir(os.path.dirname(p))
    with open(p, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())

def write_df(df: pd.DataFrame, root: str, fmt: str) -> int:
    """
    Partition by date(ts) and ticker. Returns rows written.
    """
    if df.empty:
        return 0

    # Add partitions
    df = df.copy()
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.date.astype(str)

    # Columns normalized to lakehouse schema
    # symbol->ticker, add source if missing
    if "ticker" not in df.columns:
        df["ticker"] = df["symbol"].str.upper()
    if "source" not in df.columns:
        df["source"] = "polygon"

    cols = ["ts", "ticker", "source", "open", "high", "low", "close", "volume"]
    # Optional columns present in our adapter
    if "vwap" in df.columns: cols.append("vwap")
    if "trades" in df.columns: cols.append("trades")
    if "adj_close" in df.columns: cols.append("adj_close")

    # write per partition to avoid huge single files
    total = 0
    for (d, tkr), part in df.groupby(["date", "ticker"], sort=True):
        out_dir = f"{root}/date={d}/ticker={tkr}"
        if fmt == "parquet":
            if not HAVE_PARQUET:
                raise RuntimeError("pyarrow not installed. `pip install pyarrow`")
            # fsspec path is okay; pq writes local paths natively; use pandas to_parquet with fsspec
            path = f"{out_dir}/part-{sha1(d+tkr)}.parquet"
            ensure_dir(os.path.dirname(path)) if not (path.startswith("s3://") or path.startswith("gcs://")) else None
            part[cols].to_parquet(path, index=False)  # pandas uses pyarrow engine
        elif fmt == "csv":
            path = f"{out_dir}/part-{sha1(d+tkr)}.csv"
            ensure_dir(os.path.dirname(path)) if not (path.startswith("s3://") or path.startswith("gcs://")) else None
            part[cols].to_csv(path, index=False)
        else:
            raise ValueError("fmt must be parquet or csv")
        total += len(part)
    return total

def validate_bars(df: pd.DataFrame, symbol: str) -> None:
    """
    Basic sanity checks (raise on failure).
    """
    if df.empty:
        return
    # Required columns
    required = {"ts", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    # Types
    if not pd.api.types.is_datetime64tz_dtype(df["ts"]): # type: ignore
        raise TypeError(f"{symbol}: ts must be tz-aware datetime")

    # Price sanity
    bad_high = (df["high"] < df["low"]).sum()
    bad_open = ((df["open"] < df["low"]) | (df["open"] > df["high"])).sum()
    bad_close = ((df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
    if bad_high or bad_open or bad_close:
        raise ValueError(f"{symbol}: price bounds violated (high>=low, open/close within range)")

    # Monotonic timestamps
    if not df["ts"].is_monotonic_increasing:
        # We’ll sort later, but flag for visibility
        log("warning: non-monotonic timestamps", symbol=symbol)

def load_duckdb_append(cfg: PipelineConfig, since_date: str | None = None):
    """
    Load newly written Parquet files under OUTPUT_ROOT into DuckDB table.
    """
    if not cfg.load_duckdb:
        return
    if not HAVE_DUCKDB:
        raise RuntimeError("duckdb not installed. `pip install duckdb`")
    db = duckdb.connect(cfg.duckdb_path)
    db.execute("INSTALL httpfs; LOAD httpfs;")  # enable s3/gcs paths

    # Create table if not exists per lakehouse.sql
    db.execute(f"""
        CREATE TABLE IF NOT EXISTS {cfg.duckdb_table} (
            ts TIMESTAMP,
            ticker VARCHAR,
            source VARCHAR,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            volume BIGINT,
            adj_close DOUBLE,
            ingest_ts TIMESTAMP DEFAULT now()
        );
    """)

    root = cfg.output_root.rstrip("/")
    # Use DuckDB globbing; include only todays’ or since_date partitions if provided
    glob_expr = f"{root}/**/*.parquet"
    if since_date:
        glob_expr = f"{root}/date={since_date}/**/*.parquet"

    log("duckdb: loading", table=cfg.duckdb_table, path=glob_expr)
    db.execute(f"""
        INSERT INTO {cfg.duckdb_table}
        SELECT
            ts,
            ticker,
            COALESCE(source, 'polygon') AS source,
            open, high, low, close,
            CAST(volume AS BIGINT) AS volume,
            adj_close,
            now() AS ingest_ts
        FROM read_parquet('{glob_expr}')
    """)
    db.close()


# ------------------------- Worker -----------------------------

def process_symbol(cfg: PipelineConfig, symbol: str) -> dict:
    sym = symbol.upper().strip()
    if not sym:
        return {"symbol": symbol, "rows": 0, "status": "skipped"}

    if is_done(cfg, sym):
        return {"symbol": sym, "rows": 0, "status": "cached"}

    try:
        params = BarParams(
            symbol=sym,
            multiplier=cfg.multiplier,
            timespan=cfg.timespan,
            start=cfg.start,
            end=cfg.end,
            adjusted=cfg.adjusted,
            limit=50_000,
        )
        df = fetch_bars(params)

        if cfg.validate:
            validate_bars(df, sym)

        # Normalize + sort
        if not df.empty:
            df = df.sort_values("ts").reset_index(drop=True)

        rows = write_df(df, cfg.output_root, cfg.write_format)
        mark_done(cfg, sym)
        return {"symbol": sym, "rows": int(rows), "status": "ok"}

    except Exception as e:
        return {"symbol": sym, "rows": 0, "status": "error", "error": str(e)}


# --------------------------- CLI ------------------------------

def parse_symbols_from_file(path: str) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".json", ".jsonl"):
            data = json.load(f)
            if isinstance(data, list):
                return [str(x).strip() for x in data]
            raise ValueError("JSON file must be a list of symbols")
        # fallback: newline or comma separated
        text = f.read()
        if "," in text:
            return [s.strip() for s in text.split(",") if s.strip()]
        return [s.strip() for s in text.splitlines() if s.strip()]

def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    import argparse
    p = argparse.ArgumentParser("equities_pipeline")
    p.add_argument("--symbols", nargs="*", help="Symbols list, e.g., AAPL MSFT NVDA")
    p.add_argument("--symbols-file", help="Path to file with symbols (txt/csv/json)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--timespan", default="day", choices=["minute","hour","day","week","month","quarter","year"])
    p.add_argument("--multiplier", type=int, default=1)
    p.add_argument("--no-adjusted", action="store_true", help="Do not request adjusted prices")
    p.add_argument("--output-root", default=os.getenv("OUTPUT_ROOT", "./data/equities/prices/bronze"))
    p.add_argument("--write", dest="write_format", default="parquet", choices=["parquet","csv"])
    p.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "4")))
    p.add_argument("--load-duckdb", action="store_true")
    p.add_argument("--duckdb-path", default=os.getenv("DUCKDB_PATH", "./lakehouse.duckdb"))
    p.add_argument("--duckdb-table", default="equities_prices")
    p.add_argument("--no-validate", action="store_true")

    args = p.parse_args(argv or sys.argv[1:])

    symbols: list[str] = []
    if args.symbols:
        symbols.extend(args.symbols)
    if args.symbols_file:
        symbols.extend(parse_symbols_from_file(args.symbols_file))
    if not symbols:
        # allow env var fallback
        env_syms = os.getenv("SYMBOLS", "")
        if env_syms:
            symbols = [s.strip() for s in env_syms.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided. Use --symbols or --symbols-file or SYMBOLS env var.")

    return PipelineConfig(
        symbols=symbols,
        start=args.start,
        end=args.end,
        timespan=args.timespan,
        multiplier=args.multiplier,
        adjusted=not args.no_adjusted,
        output_root=args.output_root,
        write_format=args.write_format,
        max_workers=args.max_workers,
        load_duckdb=bool(args.load_duckdb),
        duckdb_path=args.duckdb_path,
        duckdb_table=args.duckdb_table,
        validate=not args.no_validate,
    )


# --------------------------- Main -----------------------------

def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    t0 = time.time()

    # sanity for remote FS
    if (cfg.output_root.startswith("s3://") or cfg.output_root.startswith("gcs://")) and not HAVE_FSSPEC:
        raise RuntimeError("Writing to cloud paths requires fsspec + s3fs/gcsfs (pip install fsspec s3fs gcsfs)")

    ensure_dir(cfg.tmp_dir)
    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.output_root) if not (cfg.output_root.startswith("s3://") or cfg.output_root.startswith("gcs://")) else None

    log("pipeline_start", symbols=len(cfg.symbols), start=cfg.start, end=cfg.end, timespan=cfg.timespan, mult=cfg.multiplier)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futs = {pool.submit(process_symbol, cfg, s): s for s in cfg.symbols}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            log("symbol_done", **res)

    # Optional warehouse load for today's partitions only (speed)
    # You can also pass --load-duckdb and a --start equal to today to keep it light.
    if cfg.load_duckdb:
        # If the window spans multiple days, you can call load per day in a loop.
        try:
            uniq_dates = set()
            for r in results:
                if r.get("rows", 0) > 0:
                    # derive current date(s) from start..end
                    pass
            # For simplicity, load everything we just wrote:
            load_duckdb_append(cfg)
        except Exception as e:
            log("duckdb_load_error", error=str(e))

    ok = sum(1 for r in results if r.get("status") == "ok")
    errs = [r for r in results if r.get("status") == "error"]
    cached = sum(1 for r in results if r.get("status") == "cached")
    dur = round(time.time() - t0, 2)
    log("pipeline_done", ok=ok, cached=cached, errors=len(errs), secs=dur)

    if errs:
        # Print a concise error report then exit non-zero
        for r in errs:
            log("error_report", symbol=r["symbol"], error=r.get("error", ""))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())