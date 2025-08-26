# backend/bus/streams.py
"""
Central registry of Redis streams/channels and thin helpers for publishing/consuming.
Designed to be lightweight and dependency-free (besides redis + orjson/json).

Usage:
  from backend.bus.streams import (
      publish_stream, consume_stream, publish_pubsub,
      hset, hgetall, get as kv_get, set as kv_set,
      STREAM_ORDERS, STREAM_FILLS, CHAN_ORDERS, ...
  )
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

try:
    import orjson as _json # type: ignore
    def _dumps(obj: Any) -> bytes:  # fast bytes
        return _json.dumps(obj)
    def _loads(b: Union[bytes, bytearray, memoryview]) -> Any:
        return _json.loads(b)
except Exception:
    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")
    def _loads(b: Union[bytes, bytearray, memoryview]) -> Any:
        return json.loads(b.decode("utf-8")) # type: ignore

import redis

# --------------------------------------------------------------------------------------
# Redis connection
# --------------------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))
REDIS_PASS = os.getenv("REDIS_PASSWORD") or None
REDIS_SSL  = os.getenv("REDIS_SSL", "false").lower() in ("1", "true", "yes")

_pool: Optional[redis.ConnectionPool] = None

def _client() -> redis.Redis:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASS,
            ssl=REDIS_SSL,
            decode_responses=False,  # we handle bytes ourselves
        )
    return redis.Redis(connection_pool=_pool)

# --------------------------------------------------------------------------------------
# Stream & channel names (core + new features)
# --------------------------------------------------------------------------------------

# Core trading
STREAM_ORDERS             = "stream.orders"               # incoming trade orders
STREAM_FILLS              = "stream.fills"                # confirmed fills
CHAN_ORDERS               = "chan.orders"                 # pubsub for UI/notifications

# Intelligence / Data
STREAM_ALT_SIGNALS        = "stream.alt.signals"          # alt-data: satellites/sentiment/bio/climate
STREAM_POLICY_SIGNALS     = "stream.policy.signals"       # sandbox + central bank AI outputs

# Execution enhancements
STREAM_ROUTING_DECISIONS  = "stream.exec.routing"         # route plans from global router
STREAM_SYNTH_QUOTES       = "stream.market.synth.quotes"  # synthetic MM quotes/inventory

# Resilience / Health
STREAM_CHAOS_EVENTS       = "stream.risk.chaos"           # injected shocks (2008, flash crash, covid)
STREAM_HEALTH             = "stream.sys.health"           # watchdog pings, failover alerts

# --------------------------------------------------------------------------------------
# Thin helpers
# --------------------------------------------------------------------------------------

def publish_stream(
    stream: str,
    payload: Mapping[str, Any],
    *,
    maxlen: Optional[int] = 10000,
    approximate: bool = True,
    id: str = "*",
) -> bytes:
    """
    XADD <stream> with JSON-encoded 'data' field. Returns entry id (bytes).

    maxlen=None to disable trimming. approximate=True uses '~' trimming.
    """
    r = _client()
    fields = {b"data": _dumps(dict(payload))}
    if maxlen is not None:
        return r.xadd(stream, fields, id=id, maxlen=maxlen, approximate=approximate) # type: ignore
    return r.xadd(stream, fields, id=id) # type: ignore


def consume_stream(
    streams: Union[str, Iterable[str]],
    *,
    group: Optional[str] = None,
    consumer: Optional[str] = None,
    last_ids: Optional[Union[str, Mapping[str, str]]] = None,
    block_ms: int = 1000,
    count: int = 100,
    ack: bool = False,
) -> List[Tuple[str, List[Tuple[bytes, Dict[bytes, bytes]]]]]:
    """
    Consume entries from one or more streams.

    - If group & consumer are provided, uses XREADGROUP (consumer group).
      'last_ids' should be '>' for new messages or a dict of {stream: id}.
    - Otherwise uses XREAD with 'last_ids' (default '$' reads new entries).

    Returns: list of (stream, [(entry_id, {b"data": b"..."}), ...])

    NOTE: Caller is responsible for XACK when ack=True (use xack()).
    """
    r = _client()
    if isinstance(streams, str):
        stream_list = [streams]
    else:
        stream_list = list(streams)

    if last_ids is None:
        last_ids = {s: ">" if group else "$" for s in stream_list}
    elif isinstance(last_ids, str):
        last_ids = {s: last_ids for s in stream_list}
    else:
        last_ids = dict(last_ids)

    if group and consumer:
        # Ensure group exists
        for s in stream_list:
            try:
                r.xgroup_create(name=s, groupname=group, id="0", mkstream=True)
            except redis.ResponseError as e:
                # Ignore "BUSYGROUP Consumer Group name already exists"
                if "BUSYGROUP" not in str(e):
                    raise
        # Read
        resp = r.xreadgroup(group, consumer, streams=last_ids, count=count, block=block_ms) # type: ignore
    else:
        resp = r.xread(streams=last_ids, count=count, block=block_ms) # type: ignore

    # resp format already suitable; return as-is
    return resp or [] # type: ignore


def xack(stream: str, group: str, *entry_ids: Union[str, bytes]) -> int:
    """Acknowledge processed entries when using consumer groups."""
    r = _client()
    return r.xack(stream, group, *entry_ids) # type: ignore


def publish_pubsub(channel: str, message: Mapping[str, Any]) -> int:
    """PUBLISH JSON message on a pubsub channel. Returns subscriber count."""
    r = _client()
    return r.publish(channel, _dumps(dict(message))) # type: ignore


# Simple KV & hash helpers (used across modules)

def set(key: str, value: Any, *, ex: Optional[int] = None) -> bool:
    """SET key with JSON-encoded value. ex=expiry seconds."""
    r = _client()
    return bool(r.set(key, _dumps(value), ex=ex))

def get(key: str) -> Optional[Any]:
    """GET key and decode JSON."""
    r = _client()
    b = r.get(key)
    return _loads(b) if b is not None else None # type: ignore

def hset(name: str, key: str, value: Any) -> int:
    """HSET name key JSON(value)."""
    r = _client()
    return r.hset(name, key, _dumps(value)) # type: ignore

def hgetall(name: str) -> Dict[str, Any]:
    """HGETALL name, values decoded from JSON where possible."""
    r = _client()
    raw = r.hgetall(name)  # Dict[bytes, bytes]
    out: Dict[str, Any] = {}
    for k_b, v_b in raw.items(): # type: ignore
        k = k_b.decode("utf-8")
        try:
            out[k] = _loads(v_b)
        except Exception:
            out[k] = v_b.decode("utf-8", errors="replace")
    return out