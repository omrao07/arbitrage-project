# data-fabric/alt-adapters/macro_feed.py
"""
Macro feed adapter:
  - FRED:    https://api.stlouisfed.org/
  - WorldBank: https://api.worldbank.org/

Env:
  FRED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import os
import sys
import time
import typing as T
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
import pandas as pd

# Optional Parquet
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False


# --------------------------- Config ---------------------------

USER_AGENT = "hyper-os-data-fabric/1.0 (+https://github.com/your-org)"
TIMEOUT = 30
MAX_RETRIES = 4
BACKOFF_BASE = 0.6  # seconds
RATE_SLEEP = 0.15   # be gentle

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE = "https://api.stlouisfed.org"
WB_BASE = "https://api.worldbank.org/v2"

# --------------------------- Utils ---------------------------

def _sleep_backoff(attempt: int):
    time.sleep(min(5.0, BACKOFF_BASE * (2 ** (attempt - 1))))

def _iso(ts: T.Union[str, int, float, datetime] | None) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).date().isoformat()
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()
    if isinstance(ts, str):
        return ts
    raise TypeError("Unsupported timestamp type")

def _http_get(url: str, params: dict | None = None, headers: dict | None = None) -> dict: # type: ignore
    hdr = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        hdr.update(headers)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=hdr, timeout=TIMEOUT)
            if r.status_code == 429:
                # backoff on rate limit
                _sleep_backoff(attempt)
                continue
            r.raise_for_status()
            time.sleep(RATE_SLEEP)
            return r.json()
        except requests.HTTPError:
            if attempt == MAX_RETRIES:
                raise
            _sleep_backoff(attempt)
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            _sleep_backoff(attempt)

def _to_parquet(df: pd.DataFrame, path: str, partition_cols: T.Sequence[str] | None = None) -> str:
    if not HAVE_PARQUET:
        raise RuntimeError("pyarrow not installed; run `pip install pyarrow`")
    table = pa.Table.from_pandas(df, preserve_index=False)
    if partition_cols:
        pq.write_to_dataset(table, root_path=path, partition_cols=list(partition_cols))
        return path
    pq.write_table(table, path)
    return path

def _to_csv(df: pd.DataFrame, path: str, index: bool = False) -> str:
    df.to_csv(path, index=index)
    return path

def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=cols)


# =========================== FRED ============================

@dataclass
class FredParams:
    series_ids: list[str]                 # e.g. ["CPIAUCSL", "GDP"]
    start: T.Union[str, int, float, datetime] | None = None  # "YYYY-MM-DD"
    end:   T.Union[str, int, float, datetime] | None = None
    realtime_start: str | None = None     # optional
    realtime_end: str | None = None
    units: str | None = None              # e.g. "pc1", "pch", "lin"
    frequency: str | None = None          # e.g. "m", "q", "a"


def fetch_fred(params: FredParams) -> pd.DataFrame:
    """
    Fetch observations for multiple FRED series.
    Output columns:
      ["source","series_id","ts","value","realtime_start","realtime_end"]
    """
    if not FRED_API_KEY:
        raise EnvironmentError("FRED_API_KEY not set")

    frames: list[pd.DataFrame] = []
    for sid in params.series_ids:
        url = f"{FRED_BASE}/fred/series/observations"
        q = {
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "series_id": sid,
        }
        if params.start: q["observation_start"] = _iso(params.start) # type: ignore
        if params.end:   q["observation_end"]   = _iso(params.end) # type: ignore
        if params.realtime_start: q["realtime_start"] = params.realtime_start
        if params.realtime_end:   q["realtime_end"]   = params.realtime_end
        if params.units:      q["units"] = params.units
        if params.frequency:  q["frequency"] = params.frequency

        data = _http_get(url, params=q)
        obs = data.get("observations", [])
        if not obs:
            continue

        df = pd.DataFrame.from_records(obs)
        # Normalize
        df["ts"] = pd.to_datetime(df["date"], utc=True)
        # FRED values are strings; convert coercing errors to NaN
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        keep = ["realtime_start", "realtime_end", "ts", "value"]
        for k in keep:
            if k not in df.columns:
                df[k] = pd.NA
        df["source"] = "FRED"
        df["series_id"] = sid
        frames.append(df[["source","series_id","ts","value","realtime_start","realtime_end"]])

    if not frames:
        return _empty(["source","series_id","ts","value","realtime_start","realtime_end"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["series_id","ts"]).reset_index(drop=True)


# ========================= World Bank ========================

@dataclass
class WorldBankParams:
    indicator: str                 # e.g. "NY.GDP.MKTP.CD"
    countries: list[str] | None = None  # e.g. ["US","CN","JP"] ; None => "all"
    date: str | None = None        # "YYYY:YYYY" (inclusive range)
    per_page: int = 20000          # large pages to minimize requests


def fetch_worldbank(params: WorldBankParams) -> pd.DataFrame:
    """
    World Bank indicator fetcher with pagination.
    Output columns:
      ["source","indicator","country","countryiso3code","ts","value"]
    """
    # Build path like /country/US;CN/indicator/NY.GDP.MKTP.CD
    if params.countries and len(params.countries) > 0:
        country_path = ";".join(params.countries)
        url = f"{WB_BASE}/country/{country_path}/indicator/{params.indicator}"
    else:
        url = f"{WB_BASE}/country/all/indicator/{params.indicator}"

    page = 1
    frames: list[pd.DataFrame] = []

    while True:
        q = {
            "format": "json",
            "per_page": params.per_page,
            "page": page,
        }
        if params.date:
            q["date"] = params.date

        data = _http_get(url, params=q)
        if not isinstance(data, list) or len(data) < 2:
            break

        meta, rows = data[0], data[1]
        if not rows:
            break

        df = pd.DataFrame.from_records(rows)
        # Normalize
        df["ts"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["country"] = df["country"].apply(lambda x: (x or {}).get("value") if isinstance(x, dict) else x)
        df["countryiso3code"] = df["countryiso3code"]
        df["indicator"] = params.indicator
        df["source"] = "WorldBank"

        keep = ["source","indicator","country","countryiso3code","ts","value"]
        for k in keep:
            if k not in df.columns:
                df[k] = pd.NA

        frames.append(df[keep])

        # pagination
        pages = (meta or {}).get("pages") or 1
        if page >= int(pages):
            break
        page += 1

    if not frames:
        return _empty(["source","indicator","country","countryiso3code","ts","value"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["indicator","countryiso3code","ts"]).reset_index(drop=True)


# ============================ CLI ============================

def _print(df: pd.DataFrame, n: int = 6):
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head(n))

def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser("macro_feed")
    sub = p.add_subparsers(dest="cmd", required=True)

    # FRED
    pf = sub.add_parser("fred", help="Fetch series from FRED")
    pf.add_argument("--series", nargs="+", required=True, help="Series IDs, e.g. CPIAUCSL GDP")
    pf.add_argument("--start", help="YYYY-MM-DD")
    pf.add_argument("--end", help="YYYY-MM-DD")
    pf.add_argument("--units", help="pc1|pch|lin|... (FRED units)")
    pf.add_argument("--frequency", help="m|q|a|... (FRED frequency)")
    pf.add_argument("--out", help="Output path (.parquet or .csv)")

    # World Bank
    pw = sub.add_parser("worldbank", help="Fetch indicator from World Bank")
    pw.add_argument("--indicator", required=True, help="e.g. NY.GDP.MKTP.CD")
    pw.add_argument("--countries", nargs="*", help="ISO codes, e.g. US CN JP (default: all)")
    pw.add_argument("--date", help="YYYY:YYYY (range)")
    pw.add_argument("--out", help="Output path (.parquet or .csv)")

    args = p.parse_args(argv or sys.argv[1:])

    if args.cmd == "fred":
        df = fetch_fred(FredParams(
            series_ids=args.series,
            start=args.start,
            end=args.end,
            units=args.units,
            frequency=args.frequency,
        ))
    else:
        df = fetch_worldbank(WorldBankParams(
            indicator=args.indicator,
            countries=args.countries,
            date=args.date,
        ))

    if args.out:
        ext = os.path.splitext(args.out)[1].lower()
        if ext == ".parquet":
            _to_parquet(df, args.out)
        elif ext == ".csv":
            _to_csv(df, args.out)
        else:
            raise ValueError("Unsupported extension; use .parquet or .csv")
        print(f"Wrote {len(df):,} rows -> {args.out}")
    else:
        _print(df)
        print(f"Rows: {len(df):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())