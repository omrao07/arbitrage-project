# data-fabric/pipelines/macro_pipeline.py
"""
Macro pipeline (FRED + WorldBank → lakehouse)

Features
- FRED: pull multiple series IDs (e.g., CPIAUCSL, GDP)
- World Bank: pull one indicator across many countries and years
- Validations (types, non-null keys, numeric value coercion)
- Partitioned Parquet/CSV writer (local or s3://, gcs:// via fsspec)
- Optional DuckDB load into macro_series table (lakehouse.sql)
- Idempotent checkpoints (.checkpoint/macro) per task signature
- Structured JSON logging compatible with log shippers

Env (examples)
  FRED_API_KEY=xxxx
  OUTPUT_ROOT=s3://hyper-lakehouse/macro/series/bronze
  DUCKDB_PATH=/data/lakehouse.duckdb
  MAX_WORKERS=6
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import typing as T
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

# Optional I/O engines
try:
    import pyarrow  # noqa: F401
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False

try:
    import fsspec  # noqa: F401
    HAVE_FSSPEC = True
except Exception:
    HAVE_FSSPEC = False

# Our Macro adapter
try:
    from data_fabric.alt_adapters.macro_feed import ( # type: ignore
        FredParams, WorldBankParams, fetch_fred, fetch_worldbank
    )  # type: ignore
except Exception:
    from alt_adapters.macro_feed import (  # type: ignore
        FredParams, WorldBankParams, fetch_fred, fetch_worldbank
    )


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
    # Mode: "fred" or "worldbank"
    mode: str

    # FRED inputs
    fred_series: list[str] | None = None
    start: str | None = None            # YYYY-MM-DD
    end: str | None = None              # YYYY-MM-DD
    fred_units: str | None = None       # pc1|pch|lin|...
    fred_frequency: str | None = None   # m|q|a|...

    # WorldBank inputs
    wb_indicator: str | None = None     # e.g., NY.GDP.MKTP.CD
    wb_countries: list[str] | None = None  # e.g., ["US","CN","JP"] ; None => all
    wb_date: str | None = None          # e.g., 1990:2024

    # Output
    output_root: str = os.getenv("OUTPUT_ROOT", "./data/macro/series/bronze")
    write_format: str = "parquet"       # parquet|csv
    checkpoint_dir: str = "./.checkpoint/macro"

    # Optional DuckDB load
    load_duckdb: bool = False
    duckdb_path: str = os.getenv("DUCKDB_PATH", "./lakehouse.duckdb")
    duckdb_table: str = "macro_series"  # matches catalog/lakehouse.sql

    # Validations
    validate: bool = True


# ------------------------- Utilities ---------------------------

def ensure_dir(path: str):
    if path.startswith(("s3://", "gcs://")):
        return
    os.makedirs(path, exist_ok=True)

def sha1(s: str) -> str:
    import hashlib as _h
    return _h.sha1(s.encode("utf-8")).hexdigest()

def checkpoint_path(cfg: PipelineConfig) -> str:
    ensure_dir(cfg.checkpoint_dir)
    sig_parts = [
        cfg.mode,
        ",".join(cfg.fred_series or []),
        cfg.start or "",
        cfg.end or "",
        cfg.fred_units or "",
        cfg.fred_frequency or "",
        cfg.wb_indicator or "",
        ",".join(cfg.wb_countries or []),
        cfg.wb_date or "",
        cfg.write_format,
        cfg.output_root,
    ]
    key = sha1("|".join(sig_parts))
    return os.path.join(cfg.checkpoint_dir, f"{cfg.mode}_{key}.done")

def is_done(cfg: PipelineConfig) -> bool:
    return os.path.exists(checkpoint_path(cfg))

def mark_done(cfg: PipelineConfig):
    p = checkpoint_path(cfg)
    ensure_dir(os.path.dirname(p))
    with open(p, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())

def normalize_fred(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapter returns columns:
      ["source","series_id","ts","value","realtime_start","realtime_end"]
    We keep: ts (DATE), source, series_id, value, country (null)
    """
    if df.empty:
        return pd.DataFrame(columns=["ts","source","series_id","country","value"])
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["ts"], utc=True).dt.date.astype("string"),
        "source": df.get("source", "FRED"),
        "series_id": df["series_id"],
        "country": pd.Series([None] * len(df), dtype="object"),
        "value": pd.to_numeric(df["value"], errors="coerce"),
    })
    return out

def normalize_wb(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """
    Adapter returns columns:
      ["source","indicator","country","countryiso3code","ts","value"]
    We map indicator -> series_id; keep country name.
    """
    if df.empty:
        return pd.DataFrame(columns=["ts","source","series_id","country","value"])
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["ts"], utc=True).dt.date.astype("string"),
        "source": df.get("source", "WorldBank"),
        "series_id": df.get("indicator", indicator),
        "country": df.get("country"),
        "value": pd.to_numeric(df["value"], errors="coerce"),
    })
    return out

def validate_macro(df: pd.DataFrame, mode: str):
    if df.empty:
        return
    req = {"ts", "source", "series_id", "value"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{mode}: missing columns {miss}")
    # ts should be DATE-like strings (YYYY-MM-DD) at this stage
    try:
        _ = pd.to_datetime(df["ts"], errors="raise")
    except Exception:
        raise TypeError("ts must be parseable dates")
    # value numeric (allow NaN)
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise TypeError("value must be numeric")
    # series_id non-null
    if df["series_id"].isna().any():
        raise ValueError("series_id contains nulls")

def write_df(df: pd.DataFrame, root: str, fmt: str) -> int:
    """
    Partition by ts (date) and source.
    Target lakehouse schema: macro_series(ts DATE, source STRING, series_id STRING, country STRING, value DOUBLE)
    """
    if df.empty:
        return 0
    dfx = df.copy()
    # Ensure partition columns exist as strings
    dfx["date"] = pd.to_datetime(dfx["ts"]).dt.date.astype(str)
    dfx["source"] = dfx["source"].astype(str)

    total = 0
    for (d, src), part in dfx.groupby(["date", "source"], sort=True):
        out_dir = f"{root}/ts={d}/source={src}"
        if fmt == "parquet":
            if not HAVE_PARQUET:
                raise RuntimeError("pyarrow not installed. `pip install pyarrow`")
            path = f"{out_dir}/part-{sha1(d+src)}.parquet"
            if not path.startswith(("s3://", "gcs://")):
                ensure_dir(os.path.dirname(path))
            part.drop(columns=["date"]).to_parquet(path, index=False)
        elif fmt == "csv":
            path = f"{out_dir}/part-{sha1(d+src)}.csv"
            if not path.startswith(("s3://", "gcs://")):
                ensure_dir(os.path.dirname(path))
            part.drop(columns=["date"]).to_csv(path, index=False)
        else:
            raise ValueError("fmt must be parquet or csv")
        total += len(part)
    return total

def load_duckdb_append(cfg: PipelineConfig, since_date: str | None = None):
    if not cfg.load_duckdb:
        return
    if not HAVE_DUCKDB:
        raise RuntimeError("duckdb not installed. `pip install duckdb`")
    db = duckdb.connect(cfg.duckdb_path)
    db.execute("INSTALL httpfs; LOAD httpfs;")
    db.execute("""
        CREATE TABLE IF NOT EXISTS macro_series (
            ts DATE,
            source VARCHAR,
            series_id VARCHAR,
            country VARCHAR,
            value DOUBLE,
            ingest_ts TIMESTAMP DEFAULT now()
        );
    """)
    root = cfg.output_root.rstrip("/")
    glob_expr = f"{root}/**/*.parquet" if since_date is None else f"{root}/ts={since_date}/**/*.parquet"
    log("duckdb: loading", table=cfg.duckdb_table, path=glob_expr)
    db.execute(f"""
        INSERT INTO {cfg.duckdb_table}
        SELECT
            CAST(ts AS DATE)      AS ts,
            source                AS source,
            series_id             AS series_id,
            CAST(country AS VARCHAR) AS country,
            CAST(value AS DOUBLE) AS value,
            now()                 AS ingest_ts
        FROM read_parquet('{glob_expr}')
    """)
    db.close()


# --------------------------- Runner ----------------------------

def run_fred(cfg: PipelineConfig) -> dict:
    if not cfg.fred_series:
        raise SystemExit("FRED mode requires --series (one or more FRED series IDs).")
    df = fetch_fred(FredParams(
        series_ids=cfg.fred_series,
        start=cfg.start,
        end=cfg.end,
        units=cfg.fred_units,
        frequency=cfg.fred_frequency,
    ))
    out = normalize_fred(df)
    if cfg.validate:
        validate_macro(out, "FRED")
    rows = write_df(out, cfg.output_root, cfg.write_format)
    return {"mode": "fred", "rows": int(rows), "status": "ok"}

def run_worldbank(cfg: PipelineConfig) -> dict:
    if not cfg.wb_indicator:
        raise SystemExit("WorldBank mode requires --indicator (e.g., NY.GDP.MKTP.CD).")
    df = fetch_worldbank(WorldBankParams(
        indicator=cfg.wb_indicator,
        countries=cfg.wb_countries,
        date=cfg.wb_date,
    ))
    out = normalize_wb(df, cfg.wb_indicator)
    if cfg.validate:
        validate_macro(out, "WorldBank")
    rows = write_df(out, cfg.output_root, cfg.write_format)
    return {"mode": "worldbank", "rows": int(rows), "status": "ok"}


# --------------------------- CLI ------------------------------

def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    import argparse
    p = argparse.ArgumentParser("macro_pipeline")

    sub = p.add_subparsers(dest="mode", required=True)

    sp_f = sub.add_parser("fred", help="Fetch macro series from FRED")
    sp_f.add_argument("--series", nargs="+", required=True, help="e.g., CPIAUCSL GDP FEDFUNDS")
    sp_f.add_argument("--start", help="YYYY-MM-DD")
    sp_f.add_argument("--end", help="YYYY-MM-DD")
    sp_f.add_argument("--units", help="pc1|pch|lin|... (FRED units)")
    sp_f.add_argument("--frequency", help="m|q|a|... (FRED frequency)")

    sp_w = sub.add_parser("worldbank", help="Fetch indicator(s) from World Bank")
    sp_w.add_argument("--indicator", required=True, help="e.g., NY.GDP.MKTP.CD")
    sp_w.add_argument("--countries", nargs="*", help="ISO codes, e.g., US CN JP (default: all)")
    sp_w.add_argument("--date", help="YYYY:YYYY range, e.g., 1990:2024")

    # Common
    for sp in (sp_f, sp_w):
        sp.add_argument("--output-root", default=os.getenv("OUTPUT_ROOT", "./data/macro/series/bronze"))
        sp.add_argument("--write", dest="write_format", default="parquet", choices=["parquet","csv"])
        sp.add_argument("--load-duckdb", action="store_true")
        sp.add_argument("--duckdb-path", default=os.getenv("DUCKDB_PATH", "./lakehouse.duckdb"))
        sp.add_argument("--duckdb-table", default="macro_series")
        sp.add_argument("--no-validate", action="store_true")

    args = p.parse_args(argv or sys.argv[1:])

    if args.mode == "fred":
        return PipelineConfig(
            mode="fred",
            fred_series=args.series,
            start=args.start,
            end=args.end,
            fred_units=args.units,
            fred_frequency=args.frequency,
            output_root=args.output_root,
            write_format=args.write_format,
            load_duckdb=bool(args.load_duckdb),
            duckdb_path=args.duckdb_path,
            duckdb_table=args.duckdb_table,
            validate=not args.no_validate,
        )
    else:
        return PipelineConfig(
            mode="worldbank",
            wb_indicator=args.indicator,
            wb_countries=args.countries,
            wb_date=args.date,
            output_root=args.output_root,
            write_format=args.write_format,
            load_duckdb=bool(args.load_duckdb),
            duckdb_path=args.duckdb_path,
            duckdb_table=args.duckdb_table,
            validate=not args.no_validate,
        )


# --------------------------- Main -----------------------------

def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    if (cfg.output_root.startswith(("s3://", "gcs://"))) and not HAVE_FSSPEC:
        raise RuntimeError("Writing to cloud paths requires fsspec + s3fs/gcsfs (pip install fsspec s3fs gcsfs)")

    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.output_root) if not cfg.output_root.startswith(("s3://", "gcs://")) else None

    if is_done(cfg):
        log("pipeline_cached", mode=cfg.mode, output_root=cfg.output_root)
        return 0

    t0 = time.time()
    log("pipeline_start", mode=cfg.mode, output_root=cfg.output_root)

    try:
        if cfg.mode == "fred":
            res = run_fred(cfg)
        else:
            res = run_worldbank(cfg)
    except SystemExit as e:
        raise
    except Exception as e:
        log("pipeline_error", mode=cfg.mode, error=str(e))
        return 2

    # Optional warehouse load
    if cfg.load_duckdb:
        try:
            # If you want to load only today's partitions, pass since_date=str(date.today()).
            load_duckdb_append(cfg)
        except Exception as e:
            log("duckdb_load_error", error=str(e))

    mark_done(cfg)
    dur = round(time.time() - t0, 2)
    log("pipeline_done", **res, secs=dur)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())