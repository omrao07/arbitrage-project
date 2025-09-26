# data-fabric/pipelines/fx_pipeline.py
"""
FX pipeline (bars → lakehouse)

Features
- Pulls FX bars via alt-adapters/fx_yahoo.py (yfinance first, HTTP fallback)
- Pairs from CLI or file (CSV/JSON/TXT) or env
- Date window & interval controls (1m..1h..1d..1wk..1mo)
- Validations (rate>0, monotonic ts)
- Partitioned Parquet/CSV writer (local or s3://, gcs:// via fsspec)
- Optional DuckDB load into fx_rates table (lakehouse.sql)
- Idempotent checkpoints (.checkpoint/fx)
- Structured JSON logging

Env (examples)
  OUTPUT_ROOT=s3://hyper-lakehouse/fx/rates/bronze
  DUCKDB_PATH=/data/lakehouse.duckdb
  MAX_WORKERS=6

Usage
  python fx_pipeline.py \
    --pairs EURUSD USDJPY \
    --start 2024-01-01 --end 2024-03-01 \
    --interval 1d --write parquet --load-duckdb

  python fx_pipeline.py \
    --pairs-file pairs.txt \
    --start 2024-06-01 --end 2024-06-07 \
    --interval 1h --output-root ./data/fx/rates/bronze
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Our FX adapter
try:
    from data_fabric.alt_adapters.fx_yahoo import FxBarParams, fetch_bars  # type: ignore
except Exception:
    from alt_adapters.fx_yahoo import FxBarParams, fetch_bars  # type: ignore


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
    pairs: list[str]
    start: str
    end: str
    interval: str = "1d"       # 1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo

    output_root: str = os.getenv("OUTPUT_ROOT", "./data/fx/rates/bronze")
    checkpoint_dir: str = "./.checkpoint/fx"

    write_format: str = "parquet"      # parquet|csv
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))

    # Optional DuckDB load
    load_duckdb: bool = False
    duckdb_path: str = os.getenv("DUCKDB_PATH", "./lakehouse.duckdb")
    duckdb_table: str = "fx_rates"     # matches catalog/lakehouse.sql

    # Data quality toggles
    validate: bool = True


# ------------------------- Utilities ---------------------------

def ensure_dir(path: str):
    if path.startswith(("s3://", "gcs://")):
        return
    os.makedirs(path, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def checkpoint_path(cfg: PipelineConfig, pair: str) -> str:
    ensure_dir(cfg.checkpoint_dir)
    key = sha1("|".join([pair.upper(), cfg.start, cfg.end, cfg.interval]))
    return os.path.join(cfg.checkpoint_dir, f"{pair.upper()}_{key}.done")

def is_done(cfg: PipelineConfig, pair: str) -> bool:
    return os.path.exists(checkpoint_path(cfg, pair))

def mark_done(cfg: PipelineConfig, pair: str):
    p = checkpoint_path(cfg, pair)
    ensure_dir(os.path.dirname(p))
    with open(p, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())

def write_df(df: pd.DataFrame, root: str, fmt: str) -> int:
    """
    Partition by date(ts) and pair. Returns rows written.
    Input schema from adapter: [pair, ts (UTC), open, high, low, close, volume]
    Output schema (lakehouse fx_rates): [ts, base, quote, rate, source, ingest_ts]
      - We store 'rate' = close
      - 'source' = yahoo
    """
    if df.empty:
        return 0

    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["ts"], utc=True).dt.date.astype(str)
    dfx["pair"] = dfx["pair"].astype(str).str.upper()

    # Split base/quote from pair like EURUSD or EURUSD=X
    def split_pair(p: str) -> tuple[str, str]:
        s = p.replace("=X", "")
        if len(s) == 6:
            return s[:3], s[3:]
        # Fallback: try underscore (EUR_USD) or dash (EUR-USD)
        s = s.replace("-", "").replace("_", "")
        return s[:3], s[3:]

    base, quote = zip(*dfx["pair"].map(split_pair))
    dfx["base"] = list(base)
    dfx["quote"] = list(quote)

    # Lakehouse columns
    out = pd.DataFrame({
        "ts": pd.to_datetime(dfx["ts"], utc=True),
        "base": dfx["base"],
        "quote": dfx["quote"],
        "rate": pd.to_numeric(dfx["close"], errors="coerce"),
        "source": "yahoo",
    })

    # Write per partition
    total = 0
    out["date"] = out["ts"].dt.date.astype(str)
    for (d, b, q), part in out.groupby(["date", "base", "quote"], sort=True):
        out_dir = f"{root}/date={d}/base={b}/quote={q}"
        if fmt == "parquet":
            if not HAVE_PARQUET:
                raise RuntimeError("pyarrow not installed. `pip install pyarrow`")
            path = f"{out_dir}/part-{sha1(d+b+q)}.parquet"
            ensure_dir(os.path.dirname(path)) if not path.startswith(("s3://", "gcs://")) else None
            part.drop(columns=["date"]).to_parquet(path, index=False)
        elif fmt == "csv":
            path = f"{out_dir}/part-{sha1(d+b+q)}.csv"
            ensure_dir(os.path.dirname(path)) if not path.startswith(("s3://", "gcs://")) else None
            part.drop(columns=["date"]).to_csv(path, index=False)
        else:
            raise ValueError("fmt must be parquet or csv")
        total += len(part)
    return total

def validate_fx(df: pd.DataFrame, pair: str) -> None:
    if df.empty:
        return
    req = {"pair", "ts", "open", "high", "low", "close"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{pair}: missing columns {miss}")
    if not pd.api.types.is_datetime64tz_dtype(df["ts"]): # type: ignore
        raise TypeError(f"{pair}: ts must be tz-aware datetime")
    # sanity: positive rates; OHLC bounds
    neg = (pd.to_numeric(df["close"], errors="coerce") <= 0).sum()
    if neg:
        raise ValueError(f"{pair}: non-positive FX rates in close")
    if ((df["high"] < df["low"]) | (df["open"] < df["low"]) | (df["open"] > df["high"]) |
        (df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise ValueError(f"{pair}: OHLC bounds violated")
    if not df["ts"].is_monotonic_increasing:
        # We'll sort, but warn
        log("warning: non-monotonic timestamps", pair=pair)

def load_duckdb_append(cfg: PipelineConfig, since_date: str | None = None):
    if not cfg.load_duckdb:
        return
    if not HAVE_DUCKDB:
        raise RuntimeError("duckdb not installed. `pip install duckdb`")
    db = duckdb.connect(cfg.duckdb_path)
    db.execute("INSTALL httpfs; LOAD httpfs;")
    db.execute("""
        CREATE TABLE IF NOT EXISTS fx_rates (
            ts TIMESTAMP,
            base VARCHAR,
            quote VARCHAR,
            rate DOUBLE,
            source VARCHAR,
            ingest_ts TIMESTAMP DEFAULT now()
        );
    """)
    root = cfg.output_root.rstrip("/")
    glob_expr = f"{root}/**/*.parquet" if since_date is None else f"{root}/date={since_date}/**/*.parquet"
    log("duckdb: loading", table=cfg.duckdb_table, path=glob_expr)
    db.execute(f"""
        INSERT INTO {cfg.duckdb_table}
        SELECT ts, base, quote, rate, COALESCE(source, 'yahoo') AS source, now() AS ingest_ts
        FROM read_parquet('{glob_expr}')
    """)
    db.close()


# ------------------------- Worker -----------------------------

def process_pair(cfg: PipelineConfig, pair: str) -> dict:
    p = pair.strip().upper()
    if not p:
        return {"pair": pair, "rows": 0, "status": "skipped"}

    if is_done(cfg, p):
        return {"pair": p, "rows": 0, "status": "cached"}

    try:
        params = FxBarParams(pair=p, start=cfg.start, end=cfg.end, interval=cfg.interval, tz_utc=True)
        df = fetch_bars(params)
        if cfg.validate:
            validate_fx(df, p)
        if not df.empty:
            df = df.sort_values("ts").reset_index(drop=True)
        rows = write_df(df, cfg.output_root, cfg.write_format)
        mark_done(cfg, p)
        return {"pair": p, "rows": int(rows), "status": "ok"}
    except Exception as e:
        return {"pair": p, "rows": 0, "status": "error", "error": str(e)}


# --------------------------- CLI ------------------------------

def parse_pairs_from_file(path: str) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".json", ".jsonl"):
            data = json.load(f)
            if isinstance(data, list):
                return [str(x).strip() for x in data]
            raise ValueError("JSON file must be a list of pairs")
        text = f.read()
        if "," in text:
            return [s.strip() for s in text.split(",") if s.strip()]
        return [s.strip() for s in text.splitlines() if s.strip()]

def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    import argparse
    p = argparse.ArgumentParser("fx_pipeline")
    p.add_argument("--pairs", nargs="*", help="Pairs, e.g., EURUSD USDJPY or EURUSD=X")
    p.add_argument("--pairs-file", help="Path to file with pairs (txt/csv/json)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1d",
                   choices=["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"])
    p.add_argument("--output-root", default=os.getenv("OUTPUT_ROOT", "./data/fx/rates/bronze"))
    p.add_argument("--write", dest="write_format", default="parquet", choices=["parquet","csv"])
    p.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "4")))
    p.add_argument("--load-duckdb", action="store_true")
    p.add_argument("--duckdb-path", default=os.getenv("DUCKDB_PATH", "./lakehouse.duckdb"))
    p.add_argument("--duckdb-table", default="fx_rates")
    p.add_argument("--no-validate", action="store_true")

    args = p.parse_args(argv or sys.argv[1:])

    pairs: list[str] = []
    if args.pairs:
        pairs.extend(args.pairs)
    if args.pairs_file:
        pairs.extend(parse_pairs_from_file(args.pairs_file))
    if not pairs:
        env_pairs = os.getenv("PAIRS", "")
        if env_pairs:
            pairs = [s.strip() for s in env_pairs.split(",") if s.strip()]
    if not pairs:
        raise SystemExit("No pairs provided. Use --pairs or --pairs-file or PAIRS env var.")

    return PipelineConfig(
        pairs=pairs,
        start=args.start,
        end=args.end,
        interval=args.interval,
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

    if (cfg.output_root.startswith(("s3://", "gcs://"))) and not HAVE_FSSPEC:
        raise RuntimeError("Writing to cloud paths requires fsspec + s3fs/gcsfs (pip install fsspec s3fs gcsfs)")

    ensure_dir(cfg.checkpoint_dir)
    ensure_dir(cfg.output_root) if not cfg.output_root.startswith(("s3://", "gcs://")) else None

    log("pipeline_start", pairs=len(cfg.pairs), start=cfg.start, end=cfg.end, interval=cfg.interval)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futs = {pool.submit(process_pair, cfg, pr): pr for pr in cfg.pairs}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            log("pair_done", **res)

    if cfg.load_duckdb:
        try:
            load_duckdb_append(cfg)
        except Exception as e:
            log("duckdb_load_error", error=str(e))

    ok = sum(1 for r in results if r.get("status") == "ok")
    errs = [r for r in results if r.get("status") == "error"]
    cached = sum(1 for r in results if r.get("status") == "cached")
    dur = round(time.time() - t0, 2)
    log("pipeline_done", ok=ok, cached=cached, errors=len(errs), secs=dur)

    if errs:
        for r in errs:
            log("error_report", pair=r["pair"], error=r.get("error", ""))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())