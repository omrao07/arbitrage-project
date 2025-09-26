# bus/python/consumers/redis_consumer.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

try:
    import redis.asyncio as redis  # redis-py >= 4.2 has asyncio API
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install 'redis>=4.5' (pip install redis)") from e

# ---------------------------------------------------------------------
# Optional local utils (serializer/tracing). Fallbacks are no-ops.
# ---------------------------------------------------------------------
try:
    from ..utils.serializer import decode_message  # type: ignore # (bytes|str, headers_dict) -> Any
except Exception:
    def decode_message(payload: Any, headers: Optional[Dict[str, str]] = None) -> Any:
        if isinstance(payload, (bytes, bytearray)):
            try:
                return json.loads(payload.decode("utf-8"))
            except Exception:
                return payload
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except Exception:
                return payload
        return payload

try:
    from ..utils.tracing import start_consume_span  # type: ignore # context manager
except Exception:
    from contextlib import nullcontext
    def start_consume_span(*_a, **_k):
        return nullcontext()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class RedisConsumerConfig:
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Streams and consumer-group settings
    stream: str = os.getenv("REDIS_STREAM", "bus.market.ticks.v1")
    group: str = os.getenv("REDIS_GROUP", "hyper-os")
    consumer_name: str = os.getenv("REDIS_CONSUMER", "consumer-1")

    # Fetch / flow control
    batch_size: int = int(os.getenv("REDIS_BATCH_SIZE", "100"))
    block_ms: int = int(os.getenv("REDIS_BLOCK_MS", "1000"))  # XREADGROUP block timeout
    idle_ms_for_claim: int = int(os.getenv("REDIS_IDLE_MS_FOR_CLAIM", "60000"))  # 60s

    # Retry / backoff
    max_retries: int = int(os.getenv("REDIS_MAX_RETRIES", "5"))
    base_backoff_sec: float = float(os.getenv("REDIS_BASE_BACKOFF_SEC", "0.5"))
    max_backoff_sec: float = float(os.getenv("REDIS_MAX_BACKOFF_SEC", "8.0"))

    # Acknowledge behavior
    ack_on_success: bool = True
    auto_create_group: bool = True

    # Optional: headers field name convention (if present in entry fields)
    headers_field: str = os.getenv("REDIS_HEADERS_FIELD", "__headers__")  # JSON object in field


# ---------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------
class RedisStreamConsumer:
    """
    Async Redis Streams consumer using consumer groups.

    - Uses XREADGROUP to pull messages
    - Manual XACK on success; NACK simulated via not acking (will appear in PEL)
    - Periodically claims 'stuck' pending messages (XPENDING/XCLAIM) and reprocesses
    - Pluggable decode via utils.serializer.decode_message
    - OpenTelemetry span hook (optional)
    """

    def __init__(
        self,
        config: Optional[RedisConsumerConfig] = None,
        on_message: Optional[Callable[[Dict[str, Any]], Awaitable[None] | None]] = None,
        health_probe: Optional[Callable[[], None]] = None,
    ):
        self.cfg = config or RedisConsumerConfig()
        self.on_message = on_message
        self.health_probe = health_probe

        self._r: Optional[redis.Redis] = None
        self._stop = asyncio.Event()
        self._claim_task: Optional[asyncio.Task] = None

    # ------------- lifecycle -------------

    async def start(self) -> None:
        self._r = redis.from_url(self.cfg.url, decode_responses=False)  # keep bytes
        if self.cfg.auto_create_group:
            await self._ensure_group()
        self._install_signal_handlers()
        # Start background task to claim stale messages
        self._claim_task = asyncio.create_task(self._claim_loop())
        log.info(
            "Redis consumer started (stream=%s group=%s consumer=%s url=%s)",
            self.cfg.stream, self.cfg.group, self.cfg.consumer_name, self.cfg.url
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._claim_task:
            self._claim_task.cancel()
            try:
                await self._claim_task
            except asyncio.CancelledError:
                pass
        if self._r:
            await self._r.close()
        log.info("Redis consumer stopped.")

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass

    # ------------- group helpers -------------

    async def _ensure_group(self) -> None:
        assert self._r is not None
        # Create stream with an empty record if it doesn't exist to allow XGROUP CREATE
        try:
            exists = await self._r.exists(self.cfg.stream)
            if not exists:
                await self._r.xadd(self.cfg.stream, {"__init__": b"1"})
        except Exception:
            pass
        try:
            await self._r.xgroup_create(self.cfg.stream, self.cfg.group, id="$", mkstream=True)
            log.info("Created Redis consumer group '%s' on stream '%s'", self.cfg.group, self.cfg.stream)
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # group already exists
                pass
            else:
                raise

    # ------------- main loop -------------

    async def run_forever(self) -> None:
        if self._r is None:
            await self.start()

        retries = 0
        read_from = ">"  # read new messages (not already owned)

        while not self._stop.is_set():
            try:
                msgs = await self._xreadgroup(read_from)
                if msgs is None:
                    if self.health_probe:
                        self.health_probe()
                    continue

                retries = 0  # successful fetch

                for stream_name, entries in msgs:
                    for msg_id, fields in entries:
                        try:
                            headers, value = self._extract(headers_field=self.cfg.headers_field, fields=fields)
                            with start_consume_span(topic=self.cfg.stream, key=msg_id, headers=headers):
                                event = {
                                    "stream": stream_name.decode("utf-8") if isinstance(stream_name, (bytes, bytearray)) else stream_name,
                                    "id": msg_id.decode("utf-8") if isinstance(msg_id, (bytes, bytearray)) else msg_id,
                                    "timestamp": int(time.time() * 1000),
                                    "headers": headers,
                                    "value": value,
                                }
                                if self.on_message:
                                    maybe_awaitable = self.on_message(event)
                                    if asyncio.iscoroutine(maybe_awaitable):
                                        await maybe_awaitable
                            if self.cfg.ack_on_success:
                                await self._xack(msg_id)
                        except Exception as e:
                            # Do not ack -> remains pending; claim loop will retry later
                            log.exception("Handler error for %s: %s", msg_id, e)

            except Exception as e:
                retries += 1
                backoff = min(self.cfg.base_backoff_sec * (2 ** (retries - 1)), self.cfg.max_backoff_sec)
                log.exception("XREADGROUP error (attempt %d): %s. Backoff %.2fs", retries, e, backoff)
                await asyncio.sleep(backoff)
                if retries >= self.cfg.max_retries:
                    log.error("Max retries reached. Stopping consumer.")
                    await self.stop()

        # graceful shutdown
        await self.stop()

    # ------------- redis ops -------------

    async def _xreadgroup(self, read_id: str) -> Optional[List[Tuple[bytes, List[Tuple[bytes, Dict[bytes, bytes]]]]]]:
        assert self._r is not None
        # Format for redis-py: {stream: id}
        streams = {self.cfg.stream: read_id}
        # returns: List[(stream_name, [(id, {field: value}), ...])]
        return await self._r.xreadgroup(
            groupname=self.cfg.group,
            consumername=self.cfg.consumer_name,
            streams=streams, # type: ignore
            count=self.cfg.batch_size,
            block=self.cfg.block_ms,
        )

    async def _xack(self, msg_id: Any) -> None:
        assert self._r is not None
        try:
            await self._r.xack(self.cfg.stream, self.cfg.group, msg_id)
        except Exception as e:
            log.warning("XACK failed for %s: %s", msg_id, e)

    # ------------- claim loop for stuck messages -------------

    async def _claim_loop(self) -> None:
        """Periodically claim messages that have been pending (unacked) for too long."""
        assert self._r is not None
        while not self._stop.is_set():
            try:
                # XPENDING summary: count, min, max, consumers
                summary = await self._r.xpending_range(
                    name=self.cfg.stream,
                    groupname=self.cfg.group,
                    min="-",
                    max="+",
                    count=self.cfg.batch_size,
                    consumername=None,
                    idle=self.cfg.idle_ms_for_claim,
                )
                if summary:
                    ids_to_claim = [item["message_id"] for item in summary]
                    if ids_to_claim:
                        claimed = await self._r.xclaim(
                            name=self.cfg.stream,
                            groupname=self.cfg.group,
                            consumername=self.cfg.consumer_name,
                            min_idle_time=self.cfg.idle_ms_for_claim,
                            message_ids=ids_to_claim,
                            justid=False,
                        )
                        if claimed:
                            log.info("Claimed %d stale messages for reprocessing", len(claimed))
                            # push them back to this consumer by processing immediately:
                            for msg_id, fields in claimed:
                                try:
                                    headers, value = self._extract(self.cfg.headers_field, fields)
                                    with start_consume_span(topic=self.cfg.stream, key=msg_id, headers=headers):
                                        event = {
                                            "stream": self.cfg.stream,
                                            "id": msg_id.decode() if isinstance(msg_id, (bytes, bytearray)) else msg_id,
                                            "timestamp": int(time.time() * 1000),
                                            "headers": headers,
                                            "value": value,
                                        }
                                        if self.on_message:
                                            maybe = self.on_message(event)
                                            if asyncio.iscoroutine(maybe):
                                                await maybe
                                    if self.cfg.ack_on_success:
                                        await self._xack(msg_id)
                                except Exception as e:
                                    log.exception("Error handling claimed %s: %s", msg_id, e)
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Claim loop error: %s", e)
                await asyncio.sleep(2.0)

    # ------------- helpers -------------

    @staticmethod
    def _extract(headers_field: str, fields: Dict[bytes, bytes]) -> Tuple[Dict[str, Any], Any]:
        """
        Extracts headers (if present) and decodes the primary message value.

        Conventions supported:
          - If a field named headers_field exists and is JSON, use as headers
          - If a field named 'payload' or 'value' exists, decode as message
          - Else: if only one field, treat that as payload
        """
        headers: Dict[str, Any] = {}
        value: Any = None

        # Try headers
        if headers_field.encode() in fields:
            try:
                headers = json.loads(fields[headers_field.encode()].decode("utf-8"))
            except Exception:
                headers = {}

        # Identify payload
        if b"payload" in fields:
            raw = fields[b"payload"]
        elif b"value" in fields:
            raw = fields[b"value"]
        else:
            # if only one meaningful field, pick it
            raw = None
            for k, v in fields.items():
                if k not in (headers_field.encode(), b"__init__"):
                    raw = v
                    break

        value = decode_message(raw, headers=headers) if raw is not None else None
        return headers, value


# -------------------- CLI for local testing --------------------
async def _demo():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = RedisConsumerConfig()

    async def printer(event: Dict[str, Any]) -> None:
        v = event["value"]
        if isinstance(v, (dict, list)):
            s = json.dumps(v)[:500]
        else:
            s = str(v)[:500]
        log.info("[Redis %s] id=%s value=%s", event["stream"], event["id"], s)

    consumer = RedisStreamConsumer(config=cfg, on_message=printer)
    await consumer.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass