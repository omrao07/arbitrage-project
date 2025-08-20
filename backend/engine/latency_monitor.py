# backend/engine/latency_monitor.py
"""
Latency monitor:
- Subscribes to all trades streams (per region) and computes ingestion latency: now - tick.ts_ms
- Tracks OMS latency: order->fill timing
- Keeps rolling stats (avg, p50, p95, p99, EWMA) and exposes to Redis
- Sends alerts to Pub/Sub when thresholds are breached

Run alongside aggregator/execution_engine. Safe to start anytime.
"""

from __future__ import annotations

import json
import os
import statistics as stats
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import redis
import yaml

from backend.bus.streams import (
    consume_stream,
    publish_pubsub,
    STREAM_ORDERS,
    STREAM_FILLS,
)

# ---------------- Config ----------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
ALERT_CHAN = os.getenv("ALERT_CHAN", "alerts.latency")

# thresholds (ms)
INGEST_WARN_MS = int(os.getenv("LATENCY_INGEST_WARN_MS", "250"))
INGEST_CRIT_MS = int(os.getenv("LATENCY_INGEST_CRIT_MS", "750"))
OMS_WARN_MS    = int(os.getenv("LATENCY_OMS_WARN_MS", "150"))
OMS_CRIT_MS    = int(os.getenv("LATENCY_OMS_CRIT_MS", "500"))

# windows
WINDOW_SIZE = int(os.getenv("LATENCY_WINDOW", "2000"))  # rolling samples
EWMA_ALPHA  = float(os.getenv("LATENCY_EWMA_ALPHA", "0.2"))

# paths
BASE_DIR = Path(__file__).resolve().parents[2]  # repo root (../../)
CFG_DIR  = BASE_DIR / "backend" / "config"
REG_FILE = CFG_DIR / "register.yaml"
FEEDS_DIR = CFG_DIR / "feeds"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ---------------- Utils ----------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _load_regions() -> List[Dict[str, Any]]:
    with open(REG_FILE, "r") as f:
        reg = yaml.safe_load(f) or {}
    out = []
    for item in reg.get("feeds", []):
        if not item.get("enabled", True):
            continue
        cfgp = FEEDS_DIR / Path(item["config_file"]).name
        if not cfgp.exists():
            continue
        with open(cfgp, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        region = cfg.get("region", item.get("region", "XX"))
        trades_stream = (cfg.get("streams", {}) or {}).get("trades", f"trades.{region.lower()}")
        out.append({"region": region, "trades_stream": trades_stream})
    return out

def _percentiles(values: List[float], ps=(50, 95, 99)) -> Dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    out = {}
    for p in ps:
        k = max(0, min(n - 1, int(round((p / 100) * (n - 1)))))
        out[f"p{p}"] = float(sorted_vals[k])
    return out

# ---------------- RollingStats ----------------
class RollingStats:
    def __init__(self, window: int = WINDOW_SIZE, alpha: float = EWMA_ALPHA):
        self.buf: deque[float] = deque(maxlen=window)
        self.ewma: float | None = None
        self.alpha = alpha

    def add(self, x: float) -> None:
        self.buf.append(x)
        if self.ewma is None:
            self.ewma = x
        else:
            self.ewma = self.alpha * x + (1 - self.alpha) * self.ewma

    def snapshot(self) -> Dict[str, float]:
        vals = list(self.buf)
        if not vals:
            return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "ewma": float(self.ewma or 0.0)}
        avg = stats.fmean(vals)
        pct = _percentiles(vals, ps=(50, 95, 99))
        return {
            "count": len(vals),
            "avg": float(avg),
            "p50": pct["p50"],
            "p95": pct["p95"],
            "p99": pct["p99"],
            "ewma": float(self.ewma or avg),
        }

# ---------------- Regional ingestion latency ----------------
def _ingest_loop(region: str, stream: str) -> None:
    key_stats = f"latency:ingest:{region}"
    rs = RollingStats()
    while True:
        try:
            for _, tick in consume_stream(stream, start_id="$", block_ms=1000, count=200):
                # tick schema flexibility: support {'ts_ms':..} or {'timestamp':..} or nested
                ts_ms = None
                if isinstance(tick, dict):
                    ts_ms = tick.get("ts_ms") or tick.get("timestamp") or tick.get("T")
                    if isinstance(ts_ms, dict):  # weird case
                        ts_ms = ts_ms.get("$date")
                try:
                    ts_ms = int(ts_ms) if ts_ms is not None else None
                except Exception:
                    ts_ms = None

                if ts_ms is None:
                    continue

                lat = max(0, _now_ms() - ts_ms)
                rs.add(float(lat))

                snap = rs.snapshot()
                r.hset(key_stats, mapping={k: json.dumps({k: v}) for k, v in snap.items()})
                # lightweight heartbeat
                r.hset("latency:heartbeat", region, _now_ms()) # type: ignore

                # alerts
                if lat >= INGEST_CRIT_MS:
                    publish_pubsub(ALERT_CHAN, {
                        "severity": "critical",
                        "component": "ingestion",
                        "region": region,
                        "latency_ms": lat,
                        "threshold_ms": INGEST_CRIT_MS,
                        "ts_ms": _now_ms(),
                    })
                elif lat >= INGEST_WARN_MS:
                    publish_pubsub(ALERT_CHAN, {
                        "severity": "warning",
                        "component": "ingestion",
                        "region": region,
                        "latency_ms": lat,
                        "threshold_ms": INGEST_WARN_MS,
                        "ts_ms": _now_ms(),
                    })
        except Exception as e:
            publish_pubsub(ALERT_CHAN, {
                "severity": "error",
                "component": "ingestion",
                "region": region,
                "error": str(e),
                "ts_ms": _now_ms(),
            })
            time.sleep(1)

# ---------------- OMS latency (order -> fill) ----------------
def _oms_loop() -> None:
    key_stats = "latency:oms"
    rs = RollingStats()
    # correlate order id/timestamps
    orders_ts: Dict[str, int] = {}

    # Spawn consumer threads? Keep it simple: sequentially read both streams.
    def orders_reader():
        for _, order in consume_stream(STREAM_ORDERS, start_id="$", block_ms=1000, count=200):
            try:
                # We accept either {'order_id':..} or construct a key from (strategy,symbol,ts)
                if isinstance(order, str):
                    order = json.loads(order)
                oid = order.get("order_id") or f"{order.get('strategy','?')}:{order.get('symbol','?')}:{order.get('ts_ms', _now_ms())}"
                ts = int(order.get("ts_ms") or _now_ms())
                orders_ts[oid] = ts
                # heartbeat
                r.hset("latency:heartbeat", "oms_orders", _now_ms()) # type: ignore
            except Exception:
                continue

    def fills_reader():
        for _, fill in consume_stream(STREAM_FILLS, start_id="$", block_ms=1000, count=200):
            try:
                if isinstance(fill, str):
                    fill = json.loads(fill)
                # correlate by best-effort: prefer 'order_id', fallback to composite key heuristic
                oid = fill.get("order_id") or f"{fill.get('strategy','?')}:{fill.get('symbol','?')}:{fill.get('ts_ms', _now_ms())}"
                fts = int(fill.get("ts_ms") or _now_ms())
                ots = orders_ts.pop(oid, None)
                if ots is None:
                    # unknown order id; skip latency calc but still heartbeat
                    r.hset("latency:heartbeat", "oms_fills", _now_ms()) # type: ignore
                    continue
                lat = max(0, fts - ots)
                rs.add(float(lat))

                snap = rs.snapshot()
                r.hset(key_stats, mapping={k: json.dumps({k: v}) for k, v in snap.items()})
                r.hset("latency:heartbeat", "oms_fills", _now_ms()) # type: ignore

                # alerts
                if lat >= OMS_CRIT_MS:
                    publish_pubsub(ALERT_CHAN, {
                        "severity": "critical",
                        "component": "oms",
                        "latency_ms": lat,
                        "threshold_ms": OMS_CRIT_MS,
                        "ts_ms": _now_ms(),
                    })
                elif lat >= OMS_WARN_MS:
                    publish_pubsub(ALERT_CHAN, {
                        "severity": "warning",
                        "component": "oms",
                        "latency_ms": lat,
                        "threshold_ms": OMS_WARN_MS,
                        "ts_ms": _now_ms(),
                    })
            except Exception:
                continue

    threading.Thread(target=orders_reader, name="lat-oms-orders", daemon=True).start()
    threading.Thread(target=fills_reader,  name="lat-oms-fills",  daemon=True).start()

    # Keep a tiny heartbeat for monitor liveness
    while True:
        r.set("latency_monitor:alive", json.dumps({"ts": _now_ms()}))
        time.sleep(5)

# ---------------- Main ----------------
def main():
    regions = _load_regions()
    if not regions:
        publish_pubsub(ALERT_CHAN, {
            "severity": "error",
            "component": "latency_monitor",
            "error": "No enabled regions found in register.yaml",
            "ts_ms": _now_ms(),
        })
        return

    # Start per-region ingestion latency threads
    for reg in regions:
        region = reg["region"]
        stream = reg["trades_stream"]
        threading.Thread(target=_ingest_loop, args=(region, stream), name=f"lat-ingest-{region}", daemon=True).start()

    # Start OMS latency correlator
    threading.Thread(target=_oms_loop, name="lat-oms", daemon=True).start()

    # Block forever
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()