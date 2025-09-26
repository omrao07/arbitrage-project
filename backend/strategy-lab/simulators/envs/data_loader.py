# simulators/envs/data_loader.py
"""
Data Loader (stdlib-only)
-------------------------
Purpose
- Load per-symbol OHLCV time series from local CSV/JSONL.
- Normalize columns, enforce ascending time, and align calendars.
- Provide fast, memory-friendly iterators for sims/backtests.
- Optional date slicing and simple caching.

Conventions
- Row dict: {"ts": int|float, "open": float, "high": float, "low": float, "close": float, "volume": float}
- Timestamps: epoch seconds (float) preferred; ISO-8601 accepted in CSV and will be parsed.
- CSV required headers (case-insensitive, underscores/dashes ignored):
    ts/time/date, open, high, low, close, volume
- JSONL rows may contain extra fields; only the above are used.

No external deps.
"""

from __future__ import annotations

import csv
import glob
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ----------------------------- small utils ------------------------------------

def _to_epoch(value: str) -> float:
    """
    Parse timestamp from common formats to epoch seconds.
    Accepts: integer/float strings, ISO-8601 like '2024-01-02 15:30:00' or '2024-01-02T15:30:00Z'
    """
    s = str(value).strip()
    # numeric?
    try:
        return float(s)
    except Exception:
        pass
    # basic ISO handling (no tz math beyond 'Z' -> UTC)
    s = s.replace("T", " ").replace("Z", "")
    # try common formats
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"]
    for f in fmts:
        try:
            return float(int(time.mktime(time.strptime(s, f))))
        except Exception:
            continue
    # give up, return 0 (caller should sanitize)
    return 0.0

def _norm_header(h: str) -> str:
    return h.lower().replace("-", "").replace("_", "")

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _uniq_sorted(seq: Iterable[float]) -> List[float]:
    return sorted(set(seq))

# ----------------------------- config model -----------------------------------

@dataclass
class LoadSpec:
    path: str                        # file path or glob (CSV or JSONL)
    symbol: Optional[str] = None     # if None, infer from filename stem
    fmt: Optional[str] = None        # "csv" | "jsonl" | None(auto)
    tz_shift_sec: int = 0            # shift timestamps (e.g., local->UTC)
    min_rows: int = 1                # drop series shorter than this

@dataclass
class SliceSpec:
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    inclusive: bool = True

@dataclass
class DataSet:
    """
    Normalized, aligned market data set.
    - data: symbol -> list[Row]
    - calendar: ascending list of timestamps present across ALL symbols (intersection)
    """
    data: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    calendar: List[float] = field(default_factory=list)

    def symbols(self) -> List[str]:
        return sorted(self.data.keys())

    def __len__(self) -> int:
        # shared calendar length
        return len(self.calendar)

# ----------------------------- loader API -------------------------------------

class MarketDataLoader:
    def __init__(self):
        self._cache_files: Dict[str, List[Dict[str, float]]] = {}

    # ---- public entrypoints ----

    def load_many(self, specs: List[LoadSpec], slice_spec: Optional[SliceSpec] = None, align: bool = True) -> DataSet:
        """
        Load multiple symbols and optionally align calendars (intersection).
        """
        raw: Dict[str, List[Dict[str, float]]] = {}
        for spec in specs:
            for path, sym in self._expand_spec(spec):
                rows = self._load_file(path, fmt=spec.fmt, tz_shift_sec=spec.tz_shift_sec)
                if len(rows) < spec.min_rows:
                    continue
                raw[sym] = rows

        if not raw:
            return DataSet()

        # Optional slicing BEFORE alignment (faster)
        if slice_spec:
            for s in list(raw.keys()):
                raw[s] = _slice_rows(raw[s], slice_spec)

        if align:
            calendar = _aligned_calendar(raw)
            # keep only timestamps common to all (intersection)
            aligned = {s: _align_series(raw[s], calendar) for s in raw}
            return DataSet(data=aligned, calendar=calendar)
        else:
            # build union calendar (in case caller wants it)
            union = _uniq_sorted(ts for series in raw.values() for ts in (r["ts"] for r in series))
            return DataSet(data=raw, calendar=union)

    def load_dir(self, directory_glob: str, fmt: Optional[str] = None, tz_shift_sec: int = 0,
                 min_rows: int = 1, slice_spec: Optional[SliceSpec] = None, align: bool = True) -> DataSet:
        """
        Convenience: load *.csv or *.jsonl from a directory glob into a dataset.
        Symbol = filename stem by default.
        """
        paths = sorted(glob.glob(directory_glob))
        specs = [LoadSpec(path=p, symbol=os.path.splitext(os.path.basename(p))[0], fmt=fmt, tz_shift_sec=tz_shift_sec, min_rows=min_rows) for p in paths]
        return self.load_many(specs, slice_spec=slice_spec, align=align)

    # ---- iterators for sims ----

    def rolling(self, ds: DataSet, window: int) -> Iterator[Tuple[int, Dict[str, List[Dict[str, float]]]]]:
        """
        Yield (t_index, {symbol: rows_window}) with a fixed-length rolling window
        across the ALIGNED calendar. Use this inside your simulator.
        """
        n = len(ds.calendar)
        if n == 0:
            return
        w = max(1, int(window))
        for t in range(w - 1, n):
            # slice by timestamps (faster to map index->ts once)
            out: Dict[str, List[Dict[str, float]]] = {}
            for s in ds.symbols():
                out[s] = ds.data[s][t - w + 1 : t + 1]
            yield t, out

    def iter_bars(self, ds: DataSet) -> Iterator[Tuple[int, Dict[str, Dict[str, float]]]]:
        """
        Yield per aligned bar: (t_index, {symbol: row_at_t})
        """
        n = len(ds.calendar)
        if n == 0:
            return
        for t in range(n):
            out = {s: ds.data[s][t] for s in ds.symbols()}
            yield t, out

    # ------------------------- internal loaders -------------------------

    def _expand_spec(self, spec: LoadSpec) -> List[Tuple[str, str]]:
        """
        Expand a LoadSpec into (path, symbol) pairs. Supports glob patterns.
        """
        matched = glob.glob(spec.path) if any(ch in spec.path for ch in "*?[]") else [spec.path]
        out: List[Tuple[str, str]] = []
        for p in matched:
            sym = spec.symbol or os.path.splitext(os.path.basename(p))[0]
            out.append((p, sym))
        return out

    def _load_file(self, path: str, fmt: Optional[str], tz_shift_sec: int) -> List[Dict[str, float]]:
        if path in self._cache_files:
            return self._cache_files[path]
        ext = (fmt or os.path.splitext(path)[1].lstrip(".").lower())
        if ext in ("csv",):
            rows = _load_csv(path, tz_shift_sec=tz_shift_sec)
        elif ext in ("jsonl", "json"):
            rows = _load_jsonl(path, tz_shift_sec=tz_shift_sec)
        else:
            # try autodetect: peek first non-empty line
            with open(path, "r", encoding="utf-8") as f:
                head = f.read(2048)
            if "," in head.splitlines()[0]:
                rows = _load_csv(path, tz_shift_sec=tz_shift_sec)
            else:
                rows = _load_jsonl(path, tz_shift_sec=tz_shift_sec)

        # enforce ascending, drop dup ts
        rows = _dedupe_and_sort(rows)
        self._cache_files[path] = rows
        return rows

# ----------------------------- CSV / JSONL readers ----------------------------

def _load_csv(path: str, tz_shift_sec: int = 0) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # map headers
        cols = {k: _norm_header(k) for k in reader.fieldnames or []}
        # minimal header resolution
        keymap = {
            "ts": _pick(cols, ("ts", "time", "timestamp", "date", "datetime")),
            "open": _pick(cols, ("open",)),
            "high": _pick(cols, ("high",)),
            "low": _pick(cols, ("low",)),
            "close": _pick(cols, ("close", "adjclose", "adj_close")),
            "volume": _pick(cols, ("volume", "vol")),
        }
        out: List[Dict[str, float]] = []
        for raw in reader:
            try:
                ts = _to_epoch(raw[keymap["ts"]]) + tz_shift_sec
                row = {
                    "ts": float(ts),
                    "open": _safe_float(raw[keymap["open"]]),
                    "high": _safe_float(raw[keymap["high"]]),
                    "low": _safe_float(raw[keymap["low"]]),
                    "close": _safe_float(raw[keymap["close"]]),
                    "volume": _safe_float(raw[keymap["volume"]]),
                }
                if row["ts"] <= 0:
                    continue
                out.append(row)
            except Exception:
                # bad row -> skip
                continue
        return out

def _load_jsonl(path: str, tz_shift_sec: int = 0) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            ts_raw = obj.get("ts") or obj.get("time") or obj.get("timestamp") or obj.get("date") or obj.get("datetime")
            ts = _to_epoch(str(ts_raw)) + tz_shift_sec
            row = {
                "ts": float(ts),
                "open": _safe_float(obj.get("open", 0.0)),
                "high": _safe_float(obj.get("high", 0.0)),
                "low": _safe_float(obj.get("low", 0.0)),
                "close": _safe_float(obj.get("close", obj.get("adj_close", obj.get("adjclose", 0.0)))),
                "volume": _safe_float(obj.get("volume", obj.get("vol", 0.0))),
            }
            if row["ts"] <= 0:
                continue
            out.append(row)
    return out

# ----------------------------- alignment & slicing ----------------------------

def _dedupe_and_sort(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    rows = sorted(rows, key=lambda r: r["ts"])
    deduped: List[Dict[str, float]] = []
    last_ts: Optional[float] = None
    for r in rows:
        if r["ts"] == last_ts:
            # keep the latter row (assume more recent correction)
            deduped[-1] = r
        else:
            deduped.append(r)
        last_ts = r["ts"]
    return deduped

def _slice_rows(rows: List[Dict[str, float]], ss: SliceSpec) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for r in rows:
        if ss.start_ts is not None:
            if ss.inclusive and r["ts"] < ss.start_ts:
                continue
            if not ss.inclusive and r["ts"] <= ss.start_ts:
                continue
        if ss.end_ts is not None:
            if ss.inclusive and r["ts"] > ss.end_ts:
                continue
            if not ss.inclusive and r["ts"] >= ss.end_ts:
                continue
        out.append(r)
    return out

def _aligned_calendar(raw: Dict[str, List[Dict[str, float]]]) -> List[float]:
    """
    Intersection of timestamps across all symbols.
    """
    it = iter(raw.values())
    base = set(r["ts"] for r in next(it))
    for series in it:
        base &= set(r["ts"] for r in series)
        if not base:
            break
    return sorted(base)

def _align_series(series: List[Dict[str, float]], calendar: List[float]) -> List[Dict[str, float]]:
    """
    Filter series to the shared calendar. Assumes series is sorted by ts.
    """
    idx = {r["ts"]: r for r in series}
    out = []
    for ts in calendar:
        r = idx.get(ts)
        if r:
            out.append(r)
    return out

def _pick(cols_map: Dict[str, str], candidates: Tuple[str, ...]) -> str:
    """
    Find the original header for any of the normalized candidate names.
    """
    inv = {v: k for k, v in cols_map.items()}
    for c in candidates:
        if c in inv:
            return inv[c]
    # If not found, raise to trigger skip
    raise KeyError(f"Missing required column among {candidates}")

# ----------------------------- simple demo ------------------------------------

if __name__ == "__main__":
    # Example usage:
    # Assume directory structure:
    #   data/AAPL.csv, data/MSFT.csv (with headers: date, open, high, low, close, volume)
    loader = MarketDataLoader()

    # Build specs explicitly
    specs = [
        LoadSpec(path="./data/AAPL.csv", symbol="AAPL"),
        LoadSpec(path="./data/MSFT.csv", symbol="MSFT"),
    ]

    # Optional date slice: keep 2022-01-01 .. 2024-12-31 (inclusive)
    ss = SliceSpec(start_ts=_to_epoch("2022-01-01"), end_ts=_to_epoch("2024-12-31"))

    ds = loader.load_many(specs, slice_spec=ss, align=True)
    print("Loaded symbols:", ds.symbols())
    print("Aligned bars:", len(ds))

    # Rolling 20-bar windows
    for t, window in loader.rolling(ds, window=20):
        # window["AAPL"] is a list of last 20 rows ending at t
        last_bar = window["AAPL"][-1]
        if t % 50 == 0:
            print(f"t={t} ts={int(last_bar['ts'])} AAPL close={last_bar['close']:.2f}")
        # break early in demo
        if t > 120:
            break

    # Per-bar iteration (aligned)
    for t, bar in loader.iter_bars(ds):
        # bar is {symbol: row_at_t}
        if t < 3:
            print("bar", t, {k: round(v['close'], 2) for k, v in bar.items()})
        else:
            break