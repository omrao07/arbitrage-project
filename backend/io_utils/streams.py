"""
Redis Streams + PubSub wrapper
==============================
Safe, typed helpers for producing/consuming from Redis Streams.
- Use XADD for append-only event logs.
- Use XREADGROUP for consumer-group processing (durable).
- Pub/Sub for broadcast UI updates.
- Also exposes basic KV + Hash ops.

ENV:
  REDIS_HOST (default=localhost)
  REDIS_PORT (default=6379)
  REDIS_DB   (default=0)
"""

from __future__ import annotations
import os, time
from typing import Any, Dict, List, Optional, Tuple
import redis

# --- Redis client ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# --- Constants (define your project streams here) ---
STREAM_ORDERS     = "arb.orders"
STREAM_FILLS      = "arb.fills"
STREAM_ALT_SIGNALS= "arb.alt_signals"

CHAN_ORDERS       = "arb.chan.orders"
CHAN_FILLS        = "arb.chan.fills"

# ---------------- Producers ----------------

def publish_stream(stream: str, data: Dict[str, Any], *, maxlen: Optional[int] = None) -> str:
    """
    Append an event to a Redis Stream.
    Returns entry ID.
    """
    return _r.xadd(stream, data, maxlen=maxlen, approximate=True) # type: ignore

def publish_pubsub(chan: str, data: Dict[str, Any]) -> int:
    """
    Publish event to a Pub/Sub channel.
    Returns number of clients that received.
    """
    return _r.publish(chan, str(data)) # type: ignore

# ---------------- Consumers ----------------

def consume_stream(
    streams: str | List[str],
    group: str,
    consumer: str,
    last_ids: str = ">",
    block_ms: int = 1000,
    count: int = 1,
    ack: bool = True,
) -> Optional[List[Tuple[str, List[Tuple[str, Dict[str,str]]]]]]:
    """
    Consume messages from a stream (XREADGROUP).
    Args:
        streams: stream name(s) to read
        group: consumer group name (must be created with XGROUP CREATE)
        consumer: consumer name (unique per worker)
        last_ids: ">" for new messages, or ID for replay
        block_ms: block time in ms
        count: max number of messages
        ack: whether to auto-ack after read
    Returns:
        List of (stream, [(id, fields_dict)...])
    """
    if isinstance(streams, str): streams = [streams]
    ids = [last_ids] * len(streams)

    try:
        resp = _r.xreadgroup(group, consumer, dict(zip(streams, ids)), count=count, block=block_ms)
        if not resp: return None
        if ack:
            for stream, msgs in resp: # type: ignore
                for mid, fields in msgs:
                    _r.xack(stream, group, mid)
        return resp # type: ignore
    except redis.exceptions.ResponseError as e: # type: ignore
        if "NOGROUP" in str(e):
            # auto-create group at beginning of stream
            for s in streams:
                try: _r.xgroup_create(s, group, id="0", mkstream=True)
                except Exception: pass
        raise

# ---------------- KV helpers ----------------

def kv_get(key: str) -> Optional[str]:
    v = _r.get(key)
    return v if v is not None else None # type: ignore

def kv_set(key: str, val: str, *, ex: Optional[int] = None) -> bool:
    return bool(_r.set(key, val, ex=ex))

def hgetall(key: str) -> Dict[str,str]:
    return _r.hgetall(key) # type: ignore

def hset(key: str, field: str, value: str) -> int:
    return _r.hset(key, field, value) # pyright: ignore[reportReturnType]

# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Publish example
    oid = publish_stream(STREAM_ORDERS, {"symbol":"AAPL","side":"BUY","qty":"10"})
    print("XADD ->", oid)

    # Consume example
    resp = consume_stream(STREAM_ORDERS, group="test", consumer="c1", block_ms=500, count=10)
    print("XREADGROUP ->", resp)