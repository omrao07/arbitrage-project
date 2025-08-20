
# backend/ingestion/recorder.py
"""
Stream Recorder -> Parquet (optional: ClickHouse)
- Consumes one or more Redis Streams and writes rows to Parquet with time-based rotation.
- Safe with mixed/dynamic payloads; flattens common fields and preserves raw JSON.

Usage examples:
    # record crypto trades (default)
    python backend/ingestion/recorder.py

    # record multiple streams
    python backend/ingestion/recorder.py --streams trades.crypto orders fills

    # change output dir and rotation
    python backend/ingestion/recorder.py --out data/recordings --rotate-sec 300 --batch 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import redis
import pyarrow as pa
import pyarrow.parquet as pq

# ---- Env / defaults ----
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

DEFAULT_STREAMS = ["trades.crypto"]  # you can pass others via CLI
DEFAULT_OUTDIR = Path("data/recordings")

# Optional ClickHouse (off by default)
CLICKHOUSE_ENABLED = os.getenv("REC_CH_ENABLE", "false").lower() in ("1", "true", "yes")
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "9000"))
CLICKHOUSE_DB   = os.getenv("CLICKHOUSE_DB", "market_data")
CLICKHOUSE_TABLE_PREFIX = os.getenv("REC_CH_TABLE_PREFIX", "rec")

# ---- Redis client ----
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ---------- Helpers ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _utc_iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None

def _flatten(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map heterogeneous messages to a stable row schema.
    Keeps raw JSON in 'raw' for full fidelity.
    """
    ts_ms = msg.get("ts_ms") or msg.get("timestamp") or msg.get("T")
    try:
        ts_ms = int(ts_ms) if ts_ms is not None else None
    except Exception:
        ts_ms = None

    row = {
        "ts_ms": ts_ms,
        "ts_iso": _utc_iso(ts_ms) if ts_ms else None,
        "symbol": str(msg.get("symbol") or msg.get("s") or "") or None,
        "venue": msg.get("venue"),
        "side": msg.get("side"),
        "price": _safe_float(msg.get("price") or msg.get("p")),
        "size": _safe_float(msg.get("size") or msg.get("q")),
        "strategy": msg.get("strategy"),
        "typ": msg.get("typ") or msg.get("type"),
        "region": msg.get("region"),
        "order_id": msg.get("order_id"),
        "limit_price": _safe_float(msg.get("limit_price")),
        "raw": json.dumps(msg, separators=(",", ":"), ensure_ascii=False),
    }
    return row

def _schema() -> pa.schema:
    return pa.schema([
        pa.field("ts_ms", pa.int64()),
        pa.field("ts_iso", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("venue", pa.string()),
        pa.field("side", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("strategy", pa.string()),
        pa.field("typ", pa.string()),
        pa.field("region", pa.string()),
        pa.field("order_id", pa.string()),
        pa.field("limit_price", pa.float64()),
        pa.field("raw", pa.string()),
    ])

def _writer_for(path: Path) -> pq.ParquetWriter:
    return pq.ParquetWriter(path.as_posix(), _schema(), compression="zstd", use_dictionary=True)


# ---------- Recorder ----------
class StreamRecorder:
    def __init__(self, streams: List[str], outdir: Path, rotate_sec: int, batch: int):
        self.streams = streams
        self.outdir = outdir
        self.rotate_sec = rotate_sec
        self.batch = batch

        self.buffers: Dict[str, List[Dict[str, Any]]] = {s: [] for s in streams}
        self.writers: Dict[str, pq.ParquetWriter | None] = {s: None for s in streams}
        self.open_times: Dict[str, float] = {s: 0.0 for s in streams}

        _ensure_dir(outdir)

    def _current_path(self, stream: str) -> Path:
        now = datetime.utcnow()
        base = self.outdir / stream / f"{now:%Y%m%d}"
        _ensure_dir(base)
        fname = f"part-{now:%H%M%S}.parquet"
        return base / fname

    def _rotate_if_needed(self, stream: str) -> None:
        now = time.time()
        if self.writers[stream] is None:
            path = self._current_path(stream)
            self.writers[stream] = _writer_for(path)
            self.open_times[stream] = now
            return
        if now - self.open_times[stream] >= self.rotate_sec:
            # close current, open new
            self.writers[stream].close()  # type: ignore
            self.writers[stream] = _writer_for(self._current_path(stream))
            self.open_times[stream] = now

    def _flush(self, stream: str) -> None:
        buf = self.buffers[stream]
        if not buf:
            return
        self._rotate_if_needed(stream)
        table = pa.Table.from_pylist(buf, schema=_schema())
        self.writers[stream].write_table(table)  # type: ignore
        self.buffers[stream].clear()

    def _flush_all(self) -> None:
        for s in self.streams:
            self._flush(s)

    def _consume(self, stream: str) -> None:
        from backend.bus.streams import consume_stream  # local import to avoid circulars
        for _, msg in consume_stream(stream, start_id="$", block_ms=1000, count=500):
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except Exception:
                    msg = {"raw_str": msg, "ts_ms": _now_ms()}

            self.buffers[stream].append(_flatten(msg))
            if len(self.buffers[stream]) >= self.batch:
                self._flush(stream)

    def run(self) -> None:
        print(f"[recorder] streams={self.streams} outdir={self.outdir} rotate={self.rotate_sec}s batch={self.batch}")
        try:
            # simple round‑robin poll of streams (one process, low overhead)
            while True:
                for s in self.streams:
                    self._consume_one(s)
        except KeyboardInterrupt:
            print("\n[recorder] shutting down...")
        finally:
            self._flush_all()
            for w in self.writers.values():
                try:
                    if w: w.close()
                except Exception:
                    pass

    # light polling wrapper to allow periodic flush even if quiet
    def _consume_one(self, stream: str) -> None:
        from backend.bus.streams import consume_stream
        # one blocking read, then drain quickly
        res_iter = consume_stream(stream, start_id="$", block_ms=1000, count=500)
        got = False
        for i, (_, msg) in enumerate(res_iter):
            got = True
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except Exception:
                    msg = {"raw_str": msg, "ts_ms": _now_ms()}
            self.buffers[stream].append(_flatten(msg))
            if len(self.buffers[stream]) >= self.batch:
                self._flush(stream)
            # limit burst per tick to keep it fair across streams
            if i >= 2000:
                break
        # periodic flush even if quiet (rotate timer handles file roll)
        if not got:
            # touch rotate timers
            self._rotate_if_needed(stream)
            
# (continued) backend/ingestion/recorder.py

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record Redis Streams to Parquet")
    p.add_argument(
        "--streams",
        nargs="+",
        default=DEFAULT_STREAMS,
        help="Redis Streams to record (e.g., trades.crypto orders fills trades.us)",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUTDIR),
        help="Output base directory (default: data/recordings)",
    )
    p.add_argument(
        "--rotate-sec",
        type=int,
        default=300,
        help="Parquet file rotation interval in seconds (default: 300)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=2000,
        help="Rows per buffered batch write (default: 2000)",
    )
    return p.parse_args(argv)

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    rec = StreamRecorder(
        streams=list(dict.fromkeys(args.streams)),  # dedupe, keep order
        outdir=Path(args.out),
        rotate_sec=max(30, int(args.rotate_sec)),
        batch=max(100, int(args.batch)),
    )
    rec.run()

if __name__ == "__main__":
    main()