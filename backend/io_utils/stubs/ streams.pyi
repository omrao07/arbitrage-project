from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- Constants ----------
STREAM_ORDERS: str
STREAM_FILLS: str
STREAM_ALT_SIGNALS: str

CHAN_ORDERS: str
CHAN_FILLS: str

# ---------- Type aliases ----------
StreamName = str
StreamID = str
FieldMap = Dict[str, str]
# XREADGROUP return shape:
# [ (stream_name, [ (message_id, {field: value}), ... ]), ... ]
XReadGroupResp = List[Tuple[StreamName, List[Tuple[StreamID, FieldMap]]]]

# ---------- Producers ----------
def publish_stream(
    stream: str,
    data: Dict[str, Any],
    *,
    maxlen: Optional[int] = ...
) -> str: ...

def publish_pubsub(
    chan: str,
    data: Dict[str, Any]
) -> int: ...

# ---------- Consumers ----------
def consume_stream(
    streams: Union[str, List[str]],
    group: str,
    consumer: str,
    last_ids: str = ...,
    block_ms: int = ...,
    count: int = ...,
    ack: bool = ...
) -> Optional[XReadGroupResp]: ...

# ---------- KV helpers ----------
def kv_get(key: str) -> Optional[str]: ...
def kv_set(key: str, val: str, *, ex: Optional[int] = ...) -> bool: ...
def hgetall(key: str) -> Dict[str, str]: ...
def hset(key: str, field: str, value: str) -> int: ...