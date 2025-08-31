# backend/bus/streams.py
import os
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Generator, Iterable, Optional, Union

import redis

try:
    # optional pydantic support
    from pydantic import BaseModel  # type: ignore
    def _is_pydantic(obj: Any) -> bool:
        return isinstance(obj, BaseModel)
    def _to_dict(obj: Any) -> dict:
        if _is_pydantic(obj):
            return obj.model_dump()
        if is_dataclass(obj):
            return asdict(obj) # type: ignore
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Cannot serialize type: {type(obj)}")
except Exception:
    BaseModel = object  # fallback
    def _is_pydantic(obj: Any) -> bool: return False
    def _to_dict(obj: Any) -> dict:
        if is_dataclass(obj): return asdict(obj) # type: ignore
        if isinstance(obj, dict): return obj
        raise TypeError(f"Cannot serialize type: {type(obj)}")

# ---- Redis client ----
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

_r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,   # JSON strings in/out
)

# ---- Helpers ----
def _dumps(obj: Union[dict, Any]) -> str:
    if isinstance(obj, str):
        return obj
    return json.dumps(_to_dict(obj), separators=(",", ":"))

def _loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return s

# =========================
# Streams (durable pipeline)
# =========================
def publish_stream(stream: str, item: Union[dict, Any]) -> str:
    """
    Add an entry to a Redis Stream.
    Returns the Redis stream entry ID.
    """
    return _r.xadd(stream, {"json": _dumps(item)}) # type: ignore

def consume_stream(
    stream: str,
    start_id: str = "$",
    block_ms: int = 1000,
    count: int = 100,
) -> Generator[tuple[str, Any], None, None]:
    """
    Continuously read from a Redis Stream.
    - start_id="$" -> only new messages
    - start_id="0-0" -> from the beginning
    Yields (entry_id, payload_dict)
    """
    last_id = start_id
    while True:
        res = _r.xread({stream: last_id}, block=block_ms, count=count)
        if not res:
            continue
        _, entries = res[0] # type: ignore
        for entry_id, fields in entries:
            payload = _loads(fields.get("json", "{}"))
            yield entry_id, payload
        last_id = entries[-1][0]

# =========================
# Pub/Sub (fan-out to UI)
# =========================
def publish_pubsub(channel: str, item: Union[dict, Any]) -> None:
    """
    Publish a message to a Pub/Sub channel (for websockets/clients).
    """
    _r.publish(channel, _dumps(item))

def subscribe_pubsub(channel: str) -> Generator[Any, None, None]:
    """
    Subscribe to a Pub/Sub channel. Yields deserialized messages.
    """
    ps = _r.pubsub()
    ps.subscribe(channel)
    try:
        for msg in ps.listen():
            if msg.get("type") != "message":
                continue
            yield _loads(msg["data"])
    finally:
        ps.close()

# =========================
# State KV (positions, pnl)
# =========================
def hgetall(name: str) -> dict:
    return _r.hgetall(name) or {} # type: ignore

def hset(name: str, key: str, value: Union[str, dict, Any]) -> None:
    _r.hset(name, key, _dumps(value) if not isinstance(value, str) else value)

def get(name: str, default: Optional[str] = None) -> Optional[str]:
    val = _r.get(name)
    return val if val is not None else default # type: ignore

def set(name: str, value: Union[str, dict, Any]) -> None:
    _r.set(name, _dumps(value) if not isinstance(value, str) else value)

# =========================
# Common topics/channels
# =========================
STREAM_TRADES_CRYPTO = "trades.crypto"
STREAM_ORDERS = "orders"
STREAM_FILLS = "fills"

CHAN_TICKS = "ticks"           # UI live tape (Pub/Sub)
CHAN_ORDERS = "orders_stream"  # UI order events (Pub/Sub)