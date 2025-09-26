# data-fabric/alt-adapters/equities_polygon.py
"""
Equities adapter for Polygon.io
- Aggregated (bars)
- Quotes NBBO
- Trades (tick-level)
Output: pandas.DataFrame -> Parquet/CSV

Env:
  POLYGON_API_KEY=xxxx
  POLYGON_BASE_URL=https://api.polygon.io
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import typing as T
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
import pandas as pd

# Optional but recommended for Parquet
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False


# ----------------------- Config -----------------------

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io")

if not POLYGON_API_KEY:
    raise EnvironmentError("POLYGON_API_KEY not set")

DEFAULT_TIMEOUT = 30
USER_AGENT = "hyper-os-data-fabric/1.0 (+https://github.com/your-org)"

# Respectful defaults; tune in orchestrator if needed
MAX_RETRIES = 4
BACKOFF_BASE = 0.6  # seconds
RATE_SLEEP = 0.25   # seconds between calls (cooperate with free/standard limits)


# ----------------------- Utilities -----------------------

def _ts_iso(ts: T.Union[str, int, float, datetime]) -> str:
    """Coerce to RFC3339/ISO string in UTC."""
    if isinstance(ts, datetime):
        dt = ts.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    if isinstance(ts, (int, float)):
        # interpret as seconds since epoch
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    if isinstance(ts, str):
        # assume ISO already
        return ts
    raise TypeError("Unsupported timestamp type")


def _sleep_backoff(attempt: int):
    delay = min(5.0, BACKOFF_BASE * (2 ** (attempt - 1)))
    time.sleep(delay)


def _http_get(url: str, params: dict) -> dict: # type: ignore
    """GET with retries + rate-limit awareness."""
    headers = {
        "Authorization": f"Bearer {POLYGON_API_KEY}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429:
                # rate limited
                retry_after = float(resp.headers.get("Retry-After", "1.0"))
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            # retry on 5xx; raise on 4xx != 429
            code = getattr(e.response, "status_code", None)
            if code and 400 <= code < 500 and code != 429:
                raise
            if attempt == MAX_RETRIES:
                raise
            _sleep_backoff(attempt)
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            _sleep_backoff(attempt)
        finally:
            time.sleep(RATE_SLEEP)


# ----------------------- Data classes -----------------------

@dataclass
class BarParams:
    symbol: str
    multiplier: int = 1
    timespan: str = "day"   # "minute", "hour", "day"
    start: T.Union[str, int, float, datetime] = "2020-01-01"
    end: T.Union[str, int, float, datetime] = datetime.now(timezone.utc)
    limit: int = 50000  # polygon max per page
    adjusted: bool = True


# ----------------------- Aggregates (Bars) -----------------------

def fetch_bars(params: BarParams) -> pd.DataFrame:
    """
    /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}
    Paginates via 'next_url'.
    """
    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{params.symbol.upper()}"
        f"/range/{params.multiplier}/{params.timespan}/"
        f"{_ts_iso(params.start)}/{_ts_iso(params.end)}"
    )

    q = {
        "adjusted": "true" if params.adjusted else "false",
        "limit": params.limit,
        "sort": "asc",
    }

    frames: list[pd.DataFrame] = []
    next_url: T.Optional[str] = url
    next_params = q

    while next_url:
        data = _http_get(next_url, next_params)
        results = data.get("results", [])
        if not results:
            break

        df = pd.DataFrame.from_records(results)
        # Normalize epoch millis to UTC datetime
        if "t" in df.columns:
            df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        # Polygon naming: o/h/l/c/v/vw/n/t
        rename_map = {
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trades",
        }
        df = df.rename(columns=rename_map)
        df["symbol"] = params.symbol.upper()
        frames.append(df)

        # pagination
        next_url = data.get("next_url")
        next_params = {"adjusted": q["adjusted"], "limit": params.limit, "sort": "asc"}
        if next_url and not next_url.startswith("http"):
            next_url = f"{POLYGON_BASE_URL}{next_url}"

    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trades"])

    out = pd.concat(frames, ignore_index=True)
    out = out[["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trades"]]
    out = out.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return out


# ----------------------- Quotes (NBBO) -----------------------

def fetch_quotes(symbol: str, start: T.Union[str, int, float, datetime], end: T.Union[str, int, float, datetime],
                 limit: int = 50000) -> pd.DataFrame:
    """
    /v3/quotes/{ticker} — paginated with next_url
    """
    url = f"{POLYGON_BASE_URL}/v3/quotes/{symbol.upper()}"
    params = {"limit": limit, "sort": "asc", "timestamp.gte": _ts_iso(start), "timestamp.lte": _ts_iso(end)}

    frames: list[pd.DataFrame] = []
    next_url = url
    next_params = dict(params)

    while next_url:
        data = _http_get(next_url, next_params)
        results = data.get("results", [])
        if not results:
            break
        df = pd.DataFrame.from_records(results)
        # Normalize
        if "sip_timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True, errors="coerce")
        df["symbol"] = symbol.upper()
        # keep common columns
        keep = ["symbol", "ts", "bid_price", "bid_size", "ask_price", "ask_size", "conditions"]
        for k in keep:
            if k not in df.columns:
                df[k] = pd.NA
        frames.append(df[keep])

        next_url = data.get("next_url")
        next_params = {}
        if next_url and not next_url.startswith("http"):
            next_url = f"{POLYGON_BASE_URL}{next_url}"

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "ts", "bid_price", "bid_size", "ask_price", "ask_size", "conditions"]
    )


# ----------------------- Trades -----------------------

def fetch_trades(symbol: str, start: T.Union[str, int, float, datetime], end: T.Union[str, int, float, datetime],
                 limit: int = 50000) -> pd.DataFrame:
    """
    /v3/trades/{ticker}
    """
    url = f"{POLYGON_BASE_URL}/v3/trades/{symbol.upper()}"
    params = {"limit": limit, "sort": "asc", "timestamp.gte": _ts_iso(start), "timestamp.lte": _ts_iso(end)}

    frames: list[pd.DataFrame] = []
    next_url = url
    next_params = dict(params)

    while next_url:
        data = _http_get(next_url, next_params)
        results = data.get("results", [])
        if not results:
            break
        df = pd.DataFrame.from_records(results)
        if "sip_timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True, errors="coerce")
        df["symbol"] = symbol.upper()
        keep = ["symbol", "ts", "price", "size", "conditions", "exchange"]
        for k in keep:
            if k not in df.columns:
                df[k] = pd.NA
        frames.append(df[keep])

        next_url = data.get("next_url")
        next_params = {}
        if next_url and not next_url.startswith("http"):
            next_url = f"{POLYGON_BASE_URL}{next_url}"

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "ts", "price", "size", "conditions", "exchange"]
    )


# ----------------------- Writers -----------------------

def to_parquet(df: pd.DataFrame, path: str, partition_cols: T.Sequence[str] | None = None) -> str:
    if not HAVE_PARQUET:
        raise RuntimeError("pyarrow not installed; run `pip install pyarrow`")
    table = pa.Table.from_pandas(df, preserve_index=False)
    if partition_cols:
        pq.write_to_dataset(table, root_path=path, partition_cols=list(partition_cols))
        return path
    pq.write_table(table, path)
    return path


def to_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    df.to_csv(path, index=index)
    return path


# ----------------------- CLI -----------------------

def _print(df: pd.DataFrame, n: int = 5):
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head(n))


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser("equities_polygon")
    sub = p.add_subparsers(dest="cmd", required=True)

    # bars
    sp_bars = sub.add_parser("bars", help="Fetch aggregate bars")
    sp_bars.add_argument("--symbol", required=True)
    sp_bars.add_argument("--start", required=True)
    sp_bars.add_argument("--end", required=True)
    sp_bars.add_argument("--multiplier", type=int, default=1)
    sp_bars.add_argument("--timespan", default="day", choices=["minute", "hour", "day", "week", "month", "quarter", "year"])
    sp_bars.add_argument("--out", help="Parquet/CSV path (by extension)")

    # quotes
    sp_q = sub.add_parser("quotes", help="Fetch NBBO quotes")
    sp_q.add_argument("--symbol", required=True)
    sp_q.add_argument("--start", required=True)
    sp_q.add_argument("--end", required=True)
    sp_q.add_argument("--out", help="Parquet/CSV path (by extension)")

    # trades
    sp_t = sub.add_parser("trades", help="Fetch trades")
    sp_t.add_argument("--symbol", required=True)
    sp_t.add_argument("--start", required=True)
    sp_t.add_argument("--end", required=True)
    sp_t.add_argument("--out", help="Parquet/CSV path (by extension)")

    args = p.parse_args(argv or sys.argv[1:])

    if args.cmd == "bars":
        df = fetch_bars(
            BarParams(
                symbol=args.symbol,
                start=args.start,
                end=args.end,
                multiplier=args.multiplier,
                timespan=args.timespan,
            )
        )
    elif args.cmd == "quotes":
        df = fetch_quotes(args.symbol, args.start, args.end)
    else:
        df = fetch_trades(args.symbol, args.start, args.end)

    if args.out:
        ext = os.path.splitext(args.out)[1].lower()
        if ext == ".parquet":
            to_parquet(df, args.out)
        elif ext == ".csv":
            to_csv(df, args.out)
        else:
            raise ValueError("Unsupported extension; use .parquet or .csv")
        print(f"Wrote {len(df):,} rows -> {args.out}")
    else:
        _print(df)
        print(f"Rows: {len(df):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())