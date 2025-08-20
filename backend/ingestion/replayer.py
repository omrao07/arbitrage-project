# backend/ingestion/replayer.py
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import redis

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay Parquet recordings to a Redis Stream")
    p.add_argument("--path", required=True, help="Parquet file or directory (recorder output)")
    p.add_argument("--stream", default="trades.crypto", help="Redis Stream to publish to")
    p.add_argument("--speed", type=float, default=1.0, help="1.0=real-time, 10.0=10x faster")
    p.add_argument("--from-ms", type=int, default=None, help="Only rows with ts_ms >= this epoch ms")
    p.add_argument("--to-ms", type=int, default=None, help="Only rows with ts_ms <= this epoch ms")
    p.add_argument("--loop", action="store_true", help="Loop forever")
    p.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    p.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    p.add_argument("--max-rows", type=int, default=None, help="Stop after N rows (debug)")
    return p.parse_args()

def _open_dataset(path: Path) -> ds.Dataset:
    if path.is_file():
        return ds.dataset([path.as_posix()], format="parquet")
    return ds.dataset(path.as_posix(), format="parquet", partitioning="hive", ignore_prefixes=[".", "_"])

def _scan_rows(dataset: ds.Dataset, start_ms: Optional[int], end_ms: Optional[int]) -> Iterator[Tuple[int, str]]:
    filt = None
    if start_ms is not None:
        f = ds.field("ts_ms") >= pa.scalar(start_ms, pa.int64())
        filt = f if filt is None else filt & f
    if end_ms is not None:
        f = ds.field("ts_ms") <= pa.scalar(end_ms, pa.int64())
        filt = f if filt is None else filt & f
    table = dataset.scanner(columns=["ts_ms", "raw"], filter=filt).to_table()
    if table.num_rows == 0:
        return iter(())
    table = table.sort_by([("ts_ms", "ascending")])
    ts_col = table.column("ts_ms").to_pylist()
    raw_col = table.column("raw").to_pylist()
    def _it():
        for ts, raw in zip(ts_col, raw_col):
            if ts is None or raw is None: continue
            yield int(ts), str(raw)
    return _it()

def _sleep(delta_ms: int, speed: float) -> None:
    if delta_ms > 0:
        time.sleep(delta_ms / 1000.0 / max(speed, 1e-9))

def replay_once(r: redis.Redis, dataset: ds.Dataset, stream: str, speed: float,
                start_ms: Optional[int], end_ms: Optional[int], max_rows: Optional[int]) -> int:
    sent = 0
    rows = _scan_rows(dataset, start_ms, end_ms)
    prev_ts: Optional[int] = None
    for ts_ms, raw in rows:
        if prev_ts is None:
            prev_ts = ts_ms
        else:
            _sleep(ts_ms - prev_ts, speed)
            prev_ts = ts_ms
        try:
            json.loads(raw)  # validate
            r.xadd(stream, {"json": raw})
        except Exception:
            r.xadd(stream, {"json": json.dumps({"ts_ms": ts_ms, "raw": raw})})
        sent += 1
        if max_rows and sent >= max_rows:
            break
    return sent

def main():
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"[replayer] path not found: {path}")
    dataset = _open_dataset(path)
    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    print(f"[replayer] stream={args.stream} path={path} speed={args.speed} loop={args.loop} range=({args.from_ms},{args.to_ms})")
    total = 0
    try:
        while True:
            sent = replay_once(r, dataset, args.stream, args.speed, args.from_ms, args.to_ms, args.max_rows)
            total += sent
            print(f"[replayer] pass sent={sent} total={total}")
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    print("[replayer] done.")

if __name__ == "__main__":
    main()