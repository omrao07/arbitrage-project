# data-fabric/alt-adapters/fx_yahoo.py
"""
FX adapter for Yahoo Finance

- Supports intraday and daily bars
- Works with yfinance if installed; otherwise falls back to Yahoo chart API
- Normalizes to: ["pair","ts","open","high","low","close","volume"]

Examples (CLI):
  python fx_yahoo.py bars --pair EURUSD --start 2024-01-01 --end 2024-03-01 --interval 1d --out eurusd.parquet
  python fx_yahoo.py bars --pair USDJPY=X --start 2024-06-01 --end 2024-06-07 --interval 1h --out usdjpy.csv
"""

from __future__ import annotations

import os
import sys
import time
import math
import typing as T
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

# Optional dependencies
try:
    import yfinance as yf           # preferred
    HAVE_YF = True
except Exception:
    HAVE_YF = False

try:
    import pyarrow as pa            # for parquet output
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False


# ---------------- Config & helpers ----------------

USER_AGENT = "hyper-os-data-fabric/1.0 (+https://github.com/your-org)"
MAX_RETRIES = 4
BACKOFF_BASE = 0.6  # seconds

YF_ALLOWED = {
    # Yahoo supported intervals (FX)
    "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"
}

def sleep_backoff(attempt: int) -> None:
    time.sleep(min(5.0, BACKOFF_BASE * (2 ** (attempt - 1))))

def to_iso_utc(x: T.Union[str, int, float, datetime]) -> str:
    if isinstance(x, datetime):
        return x.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(x, str):
        # assume acceptable as-is (pandas/yf will parse)
        return x
    raise TypeError("Unsupported timestamp type")

def normalize_pair(pair: str) -> str:
    """Yahoo uses 'EURUSD=X'. Accept 'EURUSD' and add '=X' automatically."""
    s = pair.strip().upper()
    if s.endswith("=X"):
        return s
    if len(s) == 6 and s.isalpha():
        return s + "=X"
    return s  # allow raw tickers too


# ---------------- Public params ----------------

@dataclass
class FxBarParams:
    pair: str
    start: T.Union[str, int, float, datetime]
    end: T.Union[str, int, float, datetime]
    interval: str = "1d"            # 1m..1h..1d..1wk..1mo
    # Output controls
    tz_utc: bool = True


# ---------------- yfinance path ----------------

def _fetch_with_yfinance(p: FxBarParams) -> pd.DataFrame:
    if p.interval not in YF_ALLOWED:
        raise ValueError(f"Unsupported interval '{p.interval}'. Allowed: {sorted(YF_ALLOWED)}")

    ticker = normalize_pair(p.pair)
    # yfinance expects naive strings for start/end; it handles TZ
    start_s = to_iso_utc(p.start)
    end_s   = to_iso_utc(p.end)

    # yfinance.download returns OHLCV with DatetimeIndex
    df = yf.download(
        tickers=ticker,
        start=start_s,
        end=end_s,
        interval=p.interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return empty_frame()

    # df columns: ['Open','High','Low','Close','Adj Close','Volume']
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["pair"] = ticker
    df["ts"] = pd.to_datetime(df.index, utc=True)
    out = df[["pair","ts","open","high","low","close","volume"]].reset_index(drop=True)
    if p.tz_utc:
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values(["pair","ts"]).reset_index(drop=True)


# ---------------- HTTP chart API fallback ----------------

# https://query1.finance.yahoo.com/v8/finance/chart/EURUSD=X?interval=1d&range=1mo
Y_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

def _http_get(url: str, params: dict) -> dict: # type: ignore
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 429:
                # rate-limited; Yahoo rarely sets Retry-After, so backoff
                sleep_backoff(attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            if attempt == MAX_RETRIES:
                raise
            sleep_backoff(attempt)
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            sleep_backoff(attempt)

def _fetch_with_http(p: FxBarParams) -> pd.DataFrame:
    symbol = normalize_pair(p.pair)

    # Yahoo chart endpoint wants either range or period1/period2 (seconds)
    # We’ll use period timestamps for precise windows.
    start_dt = pd.to_datetime(to_iso_utc(p.start), utc=True)
    end_dt   = pd.to_datetime(to_iso_utc(p.end),   utc=True)
    period1  = int(start_dt.timestamp())
    period2  = int(end_dt.timestamp())

    params = {"interval": p.interval, "period1": period1, "period2": period2}
    data = _http_get(Y_CHART.format(symbol=symbol), params)

    r = data.get("chart", {}).get("result", [])
    if not r:
        return empty_frame()

    node = r[0]
    ts = node.get("timestamp", [])
    ind = node.get("indicators", {})
    quote = (ind.get("quote") or [{}])[0]

    # Build dataframe
    df = pd.DataFrame({
        "ts": pd.to_datetime(pd.Series(ts, dtype="float"), unit="s", utc=True),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })
    df["pair"] = symbol
    # Drop rows with missing timestamps (Yahoo sometimes returns trailing None)
    df = df.dropna(subset=["ts"]).reset_index(drop=True)
    cols = ["pair","ts","open","high","low","close","volume"]
    if p.tz_utc:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[cols].sort_values(["pair","ts"]).reset_index(drop=True)


# ---------------- Common helpers ----------------

def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["pair","ts","open","high","low","close","volume"])

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


# ---------------- Public API ----------------

def fetch_bars(params: FxBarParams) -> pd.DataFrame:
    """
    Fetch FX bars for a pair between [start, end] at interval.
    Tries yfinance first, then HTTP fallback.
    """
    if HAVE_YF:
        try:
            return _fetch_with_yfinance(params)
        except Exception as e:
            # fall back silently to HTTP if yfinance fails (interval limits, etc.)
            sys.stderr.write(f"[fx_yahoo] yfinance failed: {e}; falling back to HTTP\n")
    return _fetch_with_http(params)


# ---------------- CLI ----------------

def _print(df: pd.DataFrame, n: int = 5):
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head(n))

def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser("fx_yahoo")
    p.add_argument("--pair", required=True, help="FX pair, e.g. EURUSD or USDJPY=X")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1d", choices=sorted(YF_ALLOWED))
    p.add_argument("--out", help="Output path (.parquet or .csv). If omitted, prints head()")
    args = p.parse_args(argv or sys.argv[1:])

    df = fetch_bars(FxBarParams(
        pair=args.pair,
        start=args.start,
        end=args.end,
        interval=args.interval,
        tz_utc=True,
    ))

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