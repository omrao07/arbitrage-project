# backend/persistence/clickhouse_loader.py
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import redis

# pip install clickhouse-connect
import clickhouse_connect

# ---------- ENV ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CH_USER = os.getenv("CLICKHOUSE_USER", "")
CH_PASS = os.getenv("CLICKHOUSE_PASSWORD", "")
CH_DB   = os.getenv("CLICKHOUSE_DB", "hedge")

# Comma-separated Redis Streams to consume; defaults cover your stack
# You can add region trades like "trades.crypto,trades.us,trades.eu"
STREAMS = os.getenv("CH_STREAMS", "trades.crypto,orders,fills").split(",")

# batching / flush cadence
BATCH_MAX_ROWS = int(os.getenv("CH_BATCH_MAX_ROWS", "2000"))
BATCH_MAX_SEC  = float(os.getenv("CH_BATCH_MAX_SEC", "2.0"))

# snapshot cadence (seconds)
SNAPSHOT_EVERY_SEC = int(os.getenv("CH_SNAPSHOT_EVERY_SEC", "15"))

# ---------- Clients ----------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
ch = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, username=CH_USER or None, password=CH_PASS or None, database=CH_DB)

_stop = False
def _graceful(*_a):
    global _stop
    _stop = True

signal.signal(signal.SIGINT, _graceful)
signal.signal(signal.SIGTERM, _graceful)

# ---------- Helpers ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def _side_enum(side: str | None) -> int:
    if not side:
        return 0  # 'na'
    s = side.lower()
    if s == "buy": return 1
    if s == "sell": return -1
    return 0

def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None

def _parse_payload(fields: Dict[str, str]) -> Dict[str, Any]:
    # Our stream messages are stored as {"json": "<compact-json>"}
    js = fields.get("json")
    if js:
        try:
            return json.loads(js)
        except Exception:
            return {"raw": js, "ts_ms": _now_ms()}
    # Fallback: treat as flat map
    return fields

# ---------- Row builders ----------
def build_ticks_row(msg: Dict[str, Any]) -> Tuple:
    ts_ms = msg.get("ts_ms") or msg.get("timestamp") or msg.get("T")
    try: ts_ms = int(ts_ms) if ts_ms is not None else None
    except Exception: ts_ms = None

    symbol = (msg.get("symbol") or msg.get("s") or "") or None
    venue  = msg.get("venue")
    region = msg.get("region")
    side   = msg.get("side") or ("buy" if msg.get("m") is False else "sell" if msg.get("m") is True else "na")
    price  = _as_float(msg.get("price") or msg.get("p"))
    size   = _as_float(msg.get("size") or msg.get("q"))
    raw    = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
    return (
        _iso(ts_ms),         # ts DateTime64(3)
        symbol,              # symbol
        venue,               # venue
        region,              # region
        _side_enum(side),    # side enum
        float(price or 0.0), # price
        float(size or 0.0),  # size
        raw                  # raw JSON
    )

def build_orders_row(msg: Dict[str, Any]) -> Tuple:
    ts_ms = msg.get("ts_ms") or _now_ms()
    order_id = msg.get("order_id") or f"{msg.get('strategy','?')}:{msg.get('symbol','?')}:{ts_ms}"
    return (
        _iso(int(ts_ms)),                          # ts
        str(order_id),                             # order_id
        (msg.get("strategy") or ""),               # strategy
        (msg.get("symbol") or ""),                 # symbol
        (msg.get("region") or ""),                 # region
        (msg.get("venue") or ""),                  # venue
        _side_enum(msg.get("side")),               # side
        float(_as_float(msg.get("qty")) or 0.0),   # qty
        (msg.get("typ") or msg.get("type") or ""), # typ
        _as_float(msg.get("limit_price")),         # limit_price Nullable(Float64)
        (msg.get("status") or "accepted"),         # status
        (msg.get("reason") or ""),                 # reason
    )

def build_fills_row(msg: Dict[str, Any]) -> Tuple:
    ts_ms = msg.get("ts_ms") or _now_ms()
    return (
        _iso(int(ts_ms)),                          # ts
        str(msg.get("fill_id") or f"f:{ts_ms}"),   # fill_id
        str(msg.get("order_id") or ""),            # order_id
        (msg.get("strategy") or ""),               # strategy
        (msg.get("symbol") or ""),                 # symbol
        (msg.get("region") or ""),                 # region
        (msg.get("venue") or ""),                  # venue
        _side_enum(msg.get("side")),               # side
        float(_as_float(msg.get("qty")) or 0.0),   # qty
        float(_as_float(msg.get("price")) or 0.0), # price
        float(_as_float(msg.get("realized_delta")) or 0.0),  # realized_delta
        json.dumps(msg.get("meta", {}), separators=(",", ":"), ensure_ascii=False),  # meta
    )

# ---------- Batch buffers ----------
class BatchBuffer:
    def __init__(self, table: str, columns: List[str]):
        self.table = table
        self.columns = columns
        self.rows: List[Tuple] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()

    def add(self, row: Tuple):
        with self.lock:
            self.rows.append(row)

    def maybe_flush(self, force: bool = False):
        with self.lock:
            if not self.rows:
                self.last_flush = time.time()
                return
            too_many = len(self.rows) >= BATCH_MAX_ROWS
            too_old  = (time.time() - self.last_flush) >= BATCH_MAX_SEC
            if force or too_many or too_old:
                ch.insert(self.table, self.rows, column_names=self.columns)
                self.rows.clear()
                self.last_flush = time.time()

# Buffers for each target table
buf_ticks  = BatchBuffer("ticks", ["ts","symbol","venue","region","side","price","size","raw"])
buf_orders = BatchBuffer("orders", ["ts","order_id","strategy","symbol","region","venue","side","qty","typ","limit_price","status","reason"])
buf_fills  = BatchBuffer("fills", ["ts","fill_id","order_id","strategy","symbol","region","venue","side","qty","price","realized_delta","meta"])

# ---------- Consumers ----------
def _consume_stream(stream: str):
    """
    Basic XREAD loop per stream. Start from newest ($) to avoid historical flood.
    """
    last_id = "$"
    mapping = {stream: last_id}
    while not _stop:
        res = r.xread(mapping, block=1000, count=500)
        if not res:
            # periodic flush to keep latency low
            buf_ticks.maybe_flush()
            buf_orders.maybe_flush()
            buf_fills.maybe_flush()
            continue
        st, entries = res[0]
        for mid, fields in entries:
            try:
                msg = _parse_payload(fields)
                if stream.startswith("trades."):
                    buf_ticks.add(build_ticks_row(msg))
                elif stream == "orders":
                    buf_orders.add(build_orders_row(msg))
                elif stream == "fills":
                    buf_fills.add(build_fills_row(msg))
            except Exception as e:
                # lightweight error channel in Redis (optional)
                r.lpush("clickhouse_loader:errors", json.dumps({"stream": stream, "err": str(e)}))
            last_id = mid
            mapping[stream] = last_id

        # flush if batch full/old
        buf_ticks.maybe_flush()
        buf_orders.maybe_flush()
        buf_fills.maybe_flush()

    # final flush on stop
    buf_ticks.maybe_flush(force=True)
    buf_orders.maybe_flush(force=True)
    buf_fills.maybe_flush(force=True)

# ---------- Snapshots ----------
def _snapshot_positions_and_pnl():
    """
    Periodically snapshot:
      - positions (aggregate + per-strategy if available) -> positions_snap
      - pnl (aggregate) -> pnl_snap
    """
    while not _stop:
        try:
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # positions (aggregate by symbol)
            pos = r.hgetall("positions") or {}
            rows_pos = []
            for sym, raw in pos.items():
                try:
                    p = json.loads(raw)
                    rows_pos.append((
                        ts,
                        sym,
                        "",  # strategy empty for aggregate snapshot
                        float(p.get("qty", 0.0)),
                        float(p.get("avg_price", 0.0)),
                        float(p.get("realized_pnl", 0.0)),
                    ))
                except Exception:
                    continue

            # positions by strategy (if we can discover strategy names)
            enabled = r.hgetall("strategy:enabled") or {}
            for strat in enabled.keys():
                bys = r.hgetall(f"positions:by_strategy:{strat}") or {}
                for sym, raw in bys.items():
                    try:
                        p = json.loads(raw)
                        rows_pos.append((
                            ts,
                            sym,
                            strat,
                            float(p.get("qty", 0.0)),
                            float(p.get("avg_price", 0.0)),
                            float(p.get("realized_pnl", 0.0)),
                        ))
                    except Exception:
                        continue

            if rows_pos:
                ch.insert(
                    "positions_snap",
                    rows_pos,
                    column_names=["ts","symbol","strategy","qty","avg_price","realized_pnl"],
                )

            # pnl
            pnl_raw = r.get("pnl")
            if pnl_raw:
                try:
                    p = json.loads(pnl_raw)
                    ch.insert(
                        "pnl_snap",
                        [(ts, float(p.get("realized", 0.0)), float(p.get("unrealized", 0.0)), float(p.get("total", 0.0)))],
                        column_names=["ts","realized","unrealized","total"],
                    )
                except Exception:
                    pass

        except Exception as e:
            r.lpush("clickhouse_loader:errors", json.dumps({"component": "snapshot", "err": str(e)}))

        for _ in range(SNAPSHOT_EVERY_SEC):
            if _stop: break
            time.sleep(1)

# ---------- Main ----------
def main():
    # sanity check DB (optional)
    try:
        ch.query("SELECT 1")
    except Exception as e:
        print(f"[clickhouse_loader] ClickHouse connection failed: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[clickhouse_loader] Streams={STREAMS}  Batch={BATCH_MAX_ROWS}/{BATCH_MAX_SEC}s  Snapshots every {SNAPSHOT_EVERY_SEC}s")
    threads: List[threading.Thread] = []

    # start consumers
    for s in STREAMS:
        s = s.strip()
        if not s: continue
        t = threading.Thread(target=_consume_stream, args=(s,), name=f"ch-consumer-{s}", daemon=True)
        t.start()
        threads.append(t)

    # start snapshotter
    t_snap = threading.Thread(target=_snapshot_positions_and_pnl, name="ch-snapshot", daemon=True)
    t_snap.start()
    threads.append(t_snap)

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    print("[clickhouse_loader] Stopping...")

if __name__ == "__main__":
    main()