# bus/python/consumers/nats_consumer.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

try:
    import nats # type: ignore
    from nats.aio.client import Client as NATS # type: ignore
    from nats.js import JetStreamContext # type: ignore
    from nats.js.client import ConsumerConfig # type: ignore
    from nats.js.errors import NotFoundError # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install 'nats-py' (pip install nats-py)") from e

# ---- Local utils (optional) -------------------------------------------------
try:
    from ..utils.serializer import decode_message  # type: ignore # (bytes, headers_dict) -> Any
except Exception:  # fallback: JSON or raw bytes
    def decode_message(payload: bytes, headers: Optional[Dict[str, str]] = None) -> Any:
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return payload

try:
    from ..utils.tracing import start_consume_span  # type: ignore # context manager
except Exception:
    from contextlib import nullcontext
    def start_consume_span(*_a, **_k):
        return nullcontext()

# ----------------------------------------------------------------------------
log = logging.getLogger(__name__)


@dataclass
class NATSConsumerConfig:
    # Connection
    servers: str = os.getenv("NATS_SERVERS", "nats://127.0.0.1:4222")
    name: str = os.getenv("NATS_CLIENT_NAME", "hyper-os")
    tls: bool = bool(os.getenv("NATS_TLS", "false").lower() == "true")
    user: Optional[str] = os.getenv("NATS_USER") or None
    password: Optional[str] = os.getenv("NATS_PASSWORD") or None
    token: Optional[str] = os.getenv("NATS_TOKEN") or None
    # JetStream
    stream: str = os.getenv("NATS_STREAM", "BUS")
    subject: str = os.getenv("NATS_SUBJECT", "market.ticks.v1")
    durable: str = os.getenv("NATS_DURABLE", "hyper-os-consumer")
    deliver_group: Optional[str] = os.getenv("NATS_QUEUE_GROUP") or None  # for push/queue subs (unused in pull)
    ack_wait_ms: int = int(os.getenv("NATS_ACK_WAIT_MS", "30000"))        # 30s
    max_deliver: int = int(os.getenv("NATS_MAX_DELIVER", "10"))
    # Pull settings
    batch_size: int = int(os.getenv("NATS_PULL_BATCH", "100"))
    pull_timeout_sec: float = float(os.getenv("NATS_PULL_TIMEOUT_SEC", "1.0"))
    # Backoff
    max_retries: int = int(os.getenv("NATS_MAX_RETRIES", "5"))
    base_backoff_sec: float = float(os.getenv("NATS_BASE_BACKOFF_SEC", "0.5"))
    max_backoff_sec: float = float(os.getenv("NATS_MAX_BACKOFF_SEC", "8.0"))
    # Commit/ACK cadence
    ack_on_success: bool = True


class NATSConsumer:
    """
    Async NATS JetStream durable *pull* consumer.

    - Validates/decodes payload (Avro/JSON via utils.serializer)
    - Manual ack/nak with exponential backoff on handler errors
    - OpenTelemetry span hooks (optional)
    - Graceful shutdown on SIGINT/SIGTERM
    """

    def __init__(
        self,
        config: Optional[NATSConsumerConfig] = None,
        on_message: Optional[Callable[[Dict[str, Any]], Awaitable[None] | None]] = None,
        health_probe: Optional[Callable[[], None]] = None,
    ):
        self.cfg = config or NATSConsumerConfig()
        self.on_message = on_message
        self.health_probe = health_probe

        self._nc: Optional[NATS] = None
        self._js: Optional[JetStreamContext] = None
        self._sub = None  # pull subscription
        self._stop = asyncio.Event()

    # ---------- lifecycle ----------

    async def start(self) -> None:
        opts: Dict[str, Any] = {"servers": self.cfg.servers.split(","), "name": self.cfg.name}
        if self.cfg.user and self.cfg.password:
            opts.update({"user": self.cfg.user, "password": self.cfg.password})
        if self.cfg.token:
            opts.update({"token": self.cfg.token})
        if self.cfg.tls:
            opts.update({"tls": True})

        self._nc = await nats.connect(**opts)
        self._js = self._nc.jetstream() # type: ignore

        # Ensure stream exists
        await self._ensure_stream(self.cfg.stream, self.cfg.subject)

        # Ensure durable *pull* consumer exists
        await self._ensure_durable(self.cfg.stream, self.cfg.subject, self.cfg.durable)

        # Bind the pull subscription to the durable
        self._sub = await self._js.pull_subscribe( # type: ignore
            subject=self.cfg.subject,
            durable=self.cfg.durable,
            stream=self.cfg.stream,
        )
        log.info(
            "NATS consumer started (stream=%s subject=%s durable=%s servers=%s)",
            self.cfg.stream, self.cfg.subject, self.cfg.durable, self.cfg.servers
        )
        self._install_signal_handlers()

    async def stop(self) -> None:
        self._stop.set()
        try:
            if self._sub:
                await self._sub.unsubscribe()
        except Exception:
            pass
        try:
            if self._nc and self._nc.is_connected:
                await self._nc.drain()
        except Exception:
            pass
        try:
            if self._nc:
                await self._nc.close()
        except Exception:
            pass
        log.info("NATS consumer stopped.")

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop()))
            except NotImplementedError:
                # e.g., on Windows or in certain containers
                pass

    # ---------- ensure JS resources ----------

    async def _ensure_stream(self, stream: str, subject: str) -> None:
        assert self._js is not None
        try:
            await self._js.stream_info(stream)
        except NotFoundError:
            await self._js.add_stream(name=stream, subjects=[subject])
            log.info("Created JetStream stream %s for subject %s", stream, subject)

    async def _ensure_durable(self, stream: str, subject: str, durable: str) -> None:
        assert self._js is not None
        try:
            await self._js.consumer_info(stream, durable)
        except NotFoundError:
            cfg = ConsumerConfig(
                durable_name=durable,
                ack_policy="explicit",
                ack_wait=self.cfg.ack_wait_ms * 1_000_000,  # ns
                max_deliver=self.cfg.max_deliver,
                filter_subject=subject,
                deliver_policy="all",  # start at earliest; tune as needed
            )
            await self._js.add_consumer(stream=stream, config=cfg)
            log.info("Created JetStream durable consumer %s (stream=%s)", durable, stream)

    # ---------- main loop ----------

    async def run_forever(self) -> None:
        if self._nc is None:
            await self.start()

        assert self._sub is not None
        retries = 0

        while not self._stop.is_set():
            try:
                msgs = await self._sub.fetch(self.cfg.batch_size, timeout=self.cfg.pull_timeout_sec)
            except asyncio.TimeoutError:
                if self.health_probe:
                    self.health_probe()
                continue
            except Exception as e:
                retries += 1
                backoff = min(self.cfg.base_backoff_sec * (2 ** (retries - 1)), self.cfg.max_backoff_sec)
                log.exception("Fetch error (attempt %d): %s. Backoff %.2fs", retries, e, backoff)
                await asyncio.sleep(backoff)
                if retries >= self.cfg.max_retries:
                    log.error("Max retries reached, stopping consumer.")
                    await self.stop()
                continue

            retries = 0  # reset after successful fetch

            for msg in msgs:
                try:
                    headers = {k: v for k, v in (msg.headers or {}).items()}
                    with start_consume_span(topic=self.cfg.subject, key=None, headers=headers):
                        payload = decode_message(msg.data, headers=headers)
                        event = {
                            "subject": msg.subject,
                            "sequence": msg.metadata.sequence.stream if msg.metadata else None,
                            "timestamp": int(time.time() * 1000),
                            "headers": headers,
                            "value": payload,
                        }
                        if self.on_message:
                            maybe_awaitable = self.on_message(event)
                            if asyncio.iscoroutine(maybe_awaitable):
                                await maybe_awaitable

                    if self.cfg.ack_on_success:
                        await msg.ack()
                except Exception as e:
                    # Negative-ack with delay (server-side redelivery)
                    log.exception("Handler error: %s (nack)", e)
                    try:
                        await msg.nak(delay=int(min(self.cfg.max_backoff_sec, 2.0) * 1_000_000_000))  # ns
                    except Exception:
                        # As a last resort, don't ack; it will timeout and redeliver
                        pass

        await self.stop()

# ---------- CLI for local testing ----------
async def _demo():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = NATSConsumerConfig()

    async def printer(ev: Dict[str, Any]) -> None:
        v = ev["value"]
        if isinstance(v, (dict, list)):
            s = json.dumps(v)[:500]
        else:
            s = str(v)[:500]
        log.info("[NATS %s] seq=%s value=%s", ev["subject"], ev.get("sequence"), s)

    consumer = NATSConsumer(config=cfg, on_message=printer)
    await consumer.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass