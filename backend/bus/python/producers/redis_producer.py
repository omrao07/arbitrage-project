# bus/python/producers/redis_producer.py
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import redis.asyncio as redis  # redis-py >= 4.2
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install 'redis>=4.5' (pip install redis)") from e

# Optional local serializer (Avro/Protobuf/JSON). Defaults to JSON.
try:
    from ..utils.serializer import encode_message  # type: ignore # (obj, headers) -> bytes
except Exception:
    def encode_message(obj: Any, headers: Optional[Dict[str, str]] = None) -> bytes:
        if is_dataclass(obj):
            obj = asdict(obj) # type: ignore
        elif hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        elif hasattr(obj, "__dict__") and not isinstance(obj, dict):
            obj = obj.__dict__
        if isinstance(obj, dict):
            obj.setdefault("env", os.getenv("APP_ENV", "prod"))
            obj.setdefault("emitted_at", int(time.time() * 1000))
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class RedisStreamProducer:
    """
    Async Redis Streams producer.
    - Writes to a single stream key (configurable)
    - Optional approximate trimming (MAXLEN ~)
    - Optional idempotency via SET NX (dedupe key)
    - Encodes messages via utils.serializer.encode_message (fallback JSON)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        stream: Optional[str] = None,
        maxlen: Optional[int] = None,          # approximate trim length (e.g., 1_000_000)
        idempotency_ttl_sec: int = 120,        # TTL for dedupe keys
        headers_field: str = "__headers__",    # JSON-encoded headers field
        payload_field: str = "payload",        # main payload field
        create_stream_if_missing: bool = True,
    ):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.stream = stream or os.getenv("REDIS_STREAM", "bus.market.ticks.v1")
        self.maxlen = maxlen
        self.idempotency_ttl_sec = idempotency_ttl_sec
        self.headers_field = headers_field
        self.payload_field = payload_field
        self.create_stream_if_missing = create_stream_if_missing

        self._r: Optional[redis.Redis] = None
        self._host = socket.gethostname()

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._r = redis.from_url(self.url, decode_responses=False)
        if self.create_stream_if_missing:
            await self._ensure_stream()
        # ping as a sanity check
        await self._r.ping()

    async def close(self) -> None:
        if self._r:
            await self._r.close()
        self._r = None

    async def _ensure_stream(self) -> None:
        assert self._r is not None
        try:
            exists = await self._r.exists(self.stream)
            if not exists:
                # Create with tiny seed entry so XGROUP/XREAD works elsewhere
                await self._r.xadd(self.stream, {b"__init__": b"1"})
        except Exception:
            # If race/permissions, ignore; XADD later will create implicitly
            pass

    # ---------- idempotency helper ----------

    def _dedupe_key(self, msg_id: str) -> str:
        return f"dedupe:{self.stream}:{msg_id}"

    async def _check_and_mark_idempotent(self, msg_id: str) -> bool:
        """
        Returns True if this msg_id is new and marks it; False if seen already.
        """
        if not msg_id:
            return True
        assert self._r is not None
        ok = await self._r.set(self._dedupe_key(msg_id), b"1", ex=self.idempotency_ttl_sec, nx=True)
        return bool(ok)

    # ---------- publish ----------

    async def publish(
        self,
        event: Any,
        headers: Optional[Dict[str, str]] = None,
        msg_id: Optional[str] = None,          # for idempotency (producer-supplied)
        trim: Optional[bool] = None,           # override trimming behavior per call
    ) -> str:
        """
        Publish a single event to the configured stream.
        Returns the Redis stream entry id (e.g., '1726300000000-0').
        """
        assert self._r is not None, "Call start() first."

        # Idempotency check (optional)
        if msg_id:
            fresh = await self._check_and_mark_idempotent(msg_id)
            if not fresh:
                # Already published recently
                return "DUPLICATE"

        # Encode payload and headers
        payload_bytes = encode_message(event, headers=headers)
        hdrs = headers.copy() if headers else {}
        hdrs.setdefault("x-env", os.getenv("APP_ENV", "prod"))
        hdrs.setdefault("x-host", self._host)
        hdrs.setdefault("x-ts", str(int(time.time() * 1000)))

        fields = {
            self.payload_field.encode(): payload_bytes,
            self.headers_field.encode(): json.dumps(hdrs, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        }

        approx_trim = self.maxlen if (trim is None) else (self.maxlen if trim else None)

        entry_id: bytes
        if approx_trim and approx_trim > 0:
            entry_id = await self._r.xadd(self.stream, fields, maxlen=approx_trim, approximate=True) # type: ignore
        else:
            entry_id = await self._r.xadd(self.stream, fields) # type: ignore

        return entry_id.decode("utf-8") if isinstance(entry_id, (bytes, bytearray)) else str(entry_id)

    async def publish_batch(
        self,
        events: Iterable[Any],
        headers: Optional[Dict[str, str]] = None,
        idempotent: bool = True,
        trim: Optional[bool] = None,
    ) -> List[str]:
        """
        Publish multiple events (sequentially). Returns list of entry ids.
        """
        ids: List[str] = []
        for ev in events:
            mid = str(uuid.uuid4()) if idempotent else None
            eid = await self.publish(ev, headers=headers, msg_id=mid, trim=trim)
            ids.append(eid)
        return ids


# -------------------- Example usage --------------------
async def _demo():
    from ..events.market import Tick

    prod = RedisStreamProducer(
        url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        stream=os.getenv("REDIS_STREAM", "bus.market.ticks.v1"),
        maxlen=1_000_000,  # keep about 1M entries
    )
    await prod.start()

    tick = Tick.create(symbol="AAPL", price=190.25, size=100, bid=190.20, ask=190.30, source="nasdaq")
    eid = await prod.publish(tick, msg_id=f"tick-{tick.symbol}-{tick.ts_event}")
    print("XADD ->", eid)

    # Batch publish
    ticks = [
        Tick.create(symbol="MSFT", price=420.0, size=50, bid=419.9, ask=420.1, source="nasdaq"),
        Tick.create(symbol="GOOGL", price=130.0, size=75, bid=129.9, ask=130.1, source="nasdaq"),
    ]
    ids = await prod.publish_batch(ticks)
    print(f"Batch XADD -> {len(ids)} entries")

    await prod.close()


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass