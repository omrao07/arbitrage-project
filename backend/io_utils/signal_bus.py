# backend/io/signal_bus.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import pydantic as _pd  # optional validation
    _HAVE_PD = True
except Exception:
    _HAVE_PD = False

from . import streams
from backend.common.metrics import incr, hist_obs

# ---------------------------------------------------------------------
# Keys & Streams
# ---------------------------------------------------------------------
# Stream for all signal events (append-only)
STREAM = streams.STREAM_ALT_SIGNALS  # "arb.alt_signals"

# Redis keys for fast "latest" lookups
KEY_LATEST_HASH = "arb:signals:latest"       # HSET key -> json(value,ts,src,ns,ver)
KEY_SOURCE_SET  = "arb:signals:sources"      # SET of registered sources
KEY_NAMESPACE_SET = "arb:signals:namespaces" # SET of namespaces (optional if you add set ops)

# ---------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------

@dataclass
class SignalEvent:
    """
    Canonical event we put on the stream and (compacted) into the latest-hash.
    """
    key: str
    value: Any
    ts: float                   # unix epoch seconds
    src: str                    # e.g. "fitbit", "reddit", "economy", "governor"
    ns: str = "default"         # logical namespace/bucket
    ver: int = 1                # schema/version you control
    meta: Optional[Dict[str, Any]] = None

    def to_fields(self) -> Dict[str, str]:
        """Redis Streams fields are str→str; store JSON for complex values."""
        return {
            "key": self.key,
            "ts": f"{self.ts:.6f}",
            "src": self.src,
            "ns": self.ns,
            "ver": str(self.ver),
            "value": json.dumps(self.value, ensure_ascii=False),
            "meta": json.dumps(self.meta or {}, ensure_ascii=False),
        }

# Optional: Pydantic schema for validation (auto-noop if pydantic not installed)
if _HAVE_PD:
    class SignalSchema(_pd.BaseModel):
        key: str
        value: Any
        ts: float
        src: str
        ns: Optional[str] = "default"
        ver: Optional[int] = 1
        meta: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def register_source(name: str) -> None:
    """Register a producer source (bookkeeping)."""
    # You can store sources in a Redis SET; with our simple streams wrapper we’ll just HSET a marker.
    streams.hset(KEY_SOURCE_SET, name, "1")  # idempotent
    incr("signal_sources_registered", 1)

def publish(
    signals: Dict[str, Any],
    *,
    src: str,
    ns: str = "default",
    ts: Optional[float] = None,
    ver: int = 1,
    maxlen: Optional[int] = 10_000,
    meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Publish a batch of key->value signals:
      - Appends each entry to Redis Stream (durable)
      - Upserts 'latest' hash (fast lookup)
    Returns stream IDs for the appended events (in order).
    """
    t0 = time.perf_counter()
    ts = ts if ts is not None else time.time()
    ids: List[str] = []

    for k, v in signals.items():
        ev = SignalEvent(key=k, value=v, ts=ts, src=src, ns=ns, ver=ver, meta=meta)
        # Validate if pydantic available
        if _HAVE_PD:
            SignalSchema(**asdict(ev))  # raises if invalid
        sid = streams.publish_stream(STREAM, ev.to_fields(), maxlen=maxlen)
        ids.append(sid)
        # Update latest-hash with compact JSON blob
        _latest_set(ev)

    incr("signal_pub_count", len(signals))
    hist_obs("signal_publish_ms", (time.perf_counter() - t0) * 1000.0)
    return ids

def _latest_set(ev: SignalEvent) -> None:
    blob = {
        "value": ev.value,
        "ts": ev.ts,
        "src": ev.src,
        "ns": ev.ns,
        "ver": ev.ver,
        "meta": ev.meta or {},
    }
    # Store JSON in a hash field keyed by signal name
    streams.hset(KEY_LATEST_HASH, ev.key, json.dumps(blob, ensure_ascii=False))
    # Track namespace membership (optional bookkeeping)
    streams.hset(KEY_NAMESPACE_SET, ev.ns, "1")

def latest(keys: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Fetch last known values quickly from the latest-hash.
    Returns mapping key -> {value, ts, src, ns, ver, meta}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if keys:
        for k in keys:
            raw = streams.hgetall(KEY_LATEST_HASH).get(k)  # simple path using hgetall cache
            if raw:
                try:
                    out[k] = json.loads(raw)
                except Exception:
                    pass
    else:
        all_fields = streams.hgetall(KEY_LATEST_HASH)
        for k, raw in all_fields.items():
            try:
                out[k] = json.loads(raw)
            except Exception:
                continue
    return out

def snapshot(namespace: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Return a full snapshot (optionally filtered by namespace).
    """
    snap = latest()
    if namespace is None:
        return snap
    return {k: v for k, v in snap.items() if v.get("ns") == namespace}

def consume(
    group: str,
    consumer: str,
    *,
    last_ids: str = ">",
    block_ms: int = 1000,
    count: int = 10,
    ack: bool = True,
) -> Optional[List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]]:
    """
    Read from the signal stream using a consumer group.
    Returns the raw XREADGROUP response structure or None if no messages.
    """
    t0 = time.perf_counter()
    resp = streams.consume_stream(STREAM, group=group, consumer=consumer,
                                  last_ids=last_ids, block_ms=block_ms, count=count, ack=ack)
    incr("signal_consume_batches", 1 if resp else 0)
    if resp:
        incr("signal_consume_msgs", sum(len(msgs) for _, msgs in resp))
    hist_obs("signal_consume_ms", (time.perf_counter() - t0) * 1000.0)
    return resp

# ---------------------------------------------------------------------
# Helpers for common use cases
# ---------------------------------------------------------------------

def publish_normalized(
    values: Dict[str, float],
    *,
    src: str,
    ns: str = "signals",
    ts: Optional[float] = None,
    ver: int = 1,
) -> List[str]:
    """
    Convenience: publish numeric signals with standard {z,raw} structure.
    Input: values = {"risk_z": 1.2, "infl_z": -0.3, ...}
    Stored as {"value": {"z": x, "raw": x}, ...} so downstream code can
    accept both human and standardized fields.
    """
    payload = {k: {"z": float(v), "raw": float(v)} for k, v in values.items()}
    return publish(payload, src=src, ns=ns, ts=ts, ver=ver)

def upsert_single(key: str, value: Any, *, src: str, ns: str = "default", ver: int = 1) -> str:
    """Publish a single signal (thin wrapper)."""
    ids = publish({key: value}, src=src, ns=ns, ver=ver)
    return ids[0]

def read_values(keys: List[str]) -> Dict[str, Any]:
    """Return just the bare values for selected keys."""
    blob = latest(keys)
    return {k: v.get("value") for k, v in blob.items()}

# ---------------------------------------------------------------------
# Example CLI usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Producer example
    register_source("reddit")
    publish_normalized({"risk_z": 1.1, "liq_z": -0.4}, src="reddit", ns="sentiment")

    # Consumer example (group "sigproc", consumer "worker-1")
    msgs = consume(group="sigproc", consumer="worker-1", count=5, block_ms=500)
    if msgs:
        for stream, batch in msgs:
            for mid, fields in batch:
                key = fields["key"]; val = json.loads(fields["value"])
                print("got", mid, key, val)