# bus/python/producers/nats_producer.py
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import nats
    from nats.aio.client import Client as NATS
    from nats.js import JetStreamContext
    from nats.errors import TimeoutError
    from nats.js.errors import NotFoundError
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install 'nats-py' (pip install nats-py)") from e

# Optional local serializer (Avro/Protobuf/JSON). We default to JSON.
try:
    from ..utils.serializer import encode_message  # type: ignore # (obj, headers) -> bytes
except Exception:
    def encode_message(obj: Any, headers: Optional[Dict[str, str]] = None) -> bytes:
        if is_dataclass(obj):
            obj = asdict(obj)  # type: ignore
        elif hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        elif hasattr(obj, "__dict__") and not isinstance(obj, dict):
            obj = obj.__dict__
        if not isinstance(obj, dict):
            return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        # add common envelope fields if not present
        obj.setdefault("env", os.getenv("APP_ENV", "prod"))
        obj.setdefault("emitted_at", int(time.time() * 1000))
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class NATSProducer:
    """
    Async NATS JetStream producer.
    - JSON by default (pluggable encode_message)
    - Idempotency via 'Nats-Msg-Id' (JetStream de-dup)
    - Auto ensure stream/subject (optional)
    - Batch publish helper
    """

    def __init__(
        self,
        servers: Optional[str] = None,
        name: Optional[str] = None,
        stream: Optional[str] = None,
        subject: Optional[str] = None,
        ensure_stream: bool = True,
        retention: str = "limits",               # limits|workqueue|interest (JetStream)
        max_age_seconds: Optional[int] = 7 * 24 * 3600,  # message TTL
        replicas: int = 1,
        tls: bool = bool(os.getenv("NATS_TLS", "false").lower() == "true"),
        user: Optional[str] = os.getenv("NATS_USER") or None,
        password: Optional[str] = os.getenv("NATS_PASSWORD") or None,
        token: Optional[str] = os.getenv("NATS_TOKEN") or None,
    ):
        self.servers = (servers or os.getenv("NATS_SERVERS", "nats://127.0.0.1:4222")).split(",")
        self.name = name or f"hyper-os-producer@{socket.gethostname()}"
        self.stream = stream or os.getenv("NATS_STREAM", "BUS")
        self.subject = subject or os.getenv("NATS_SUBJECT", "market.ticks.v1")
        self.ensure_stream = ensure_stream
        self.retention = retention
        self.max_age_seconds = max_age_seconds
        self.replicas = replicas

        self._nc: Optional[NATS] = None
        self._js: Optional[JetStreamContext] = None

        self._connect_opts: Dict[str, Any] = {"servers": self.servers, "name": self.name}
        if tls:
            self._connect_opts["tls"] = True
        if user and password:
            self._connect_opts.update({"user": user, "password": password})
        if token:
            self._connect_opts.update({"token": token})

    # ---------- lifecycle ----------

    async def start(self) -> None:
        self._nc = await nats.connect(**self._connect_opts)
        self._js = self._nc.jetstream()
        if self.ensure_stream:
            await self._ensure_stream()

    async def close(self) -> None:
        if self._nc:
            try:
                await self._nc.drain()
            except Exception:
                pass
            try:
                await self._nc.close()
            except Exception:
                pass
        self._nc = None
        self._js = None

    # ---------- stream ensure ----------

    async def _ensure_stream(self) -> None:
        assert self._js is not None
        try:
            await self._js.stream_info(self.stream)
        except NotFoundError:
            cfg = {
                "name": self.stream,
                "subjects": [self.subject],
                "retention": self.retention,
                "allow_rollup_hdrs": True,
                "duplicate_window": 120_000_000_000,  # 120s in ns for idempotency
                "replicas": self.replicas,
            }
            if self.max_age_seconds is not None:
                cfg["max_age"] = int(self.max_age_seconds * 1_000_000_000)  # ns
            await self._js.add_stream(**cfg)

    # ---------- publish ----------

    async def publish(
        self,
        event: Any,
        subject: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        msg_id: Optional[str] = None,
        timeout: float = 2.0,
    ) -> Tuple[int, int]:
        """
        Publish a single event. Returns (stream_seq, domain_seq).
        Uses JetStream idempotency via Nats-Msg-Id when 'msg_id' is provided.
        """
        assert self._js is not None, "Call start() first."
        subj = subject or self.subject
        payload = encode_message(event, headers=headers)

        # Prepare headers
        h = headers.copy() if headers else {}
        if msg_id:
            h["Nats-Msg-Id"] = msg_id  # enables JetStream de-duplication
        # Common metadata
        h.setdefault("x-env", os.getenv("APP_ENV", "prod"))
        h.setdefault("x-host", socket.gethostname())

        try:
            ack = await self._js.publish(subject=subj, payload=payload, headers=h, timeout=timeout)
            # ack has .seq and possibly .domain, depending on server version
            stream_seq = getattr(ack, "seq", 0)
            domain_seq = getattr(ack, "domain", 0) or 0
            return stream_seq, domain_seq
        except TimeoutError as e:
            raise TimeoutError(f"Publish timeout to subject '{subj}': {e}") from e

    async def publish_batch(
        self,
        events: Iterable[Any],
        subject: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        idempotent: bool = True,
        timeout: float = 2.0,
    ) -> List[Tuple[int, int]]:
        """
        Publish a batch of events with optional idempotency (auto msg ids).
        Returns list of (stream_seq, domain_seq).
        """
        results: List[Tuple[int, int]] = []
        subj = subject or self.subject
        for ev in events:
            msg_id = str(uuid.uuid4()) if idempotent else None
            res = await self.publish(ev, subject=subj, headers=headers, msg_id=msg_id, timeout=timeout)
            results.append(res)
        return results

    # ---------- request/reply (optional utility) ----------

    async def request(
        self,
        subject: str,
        payload: Union[bytes, Dict[str, Any], Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 2.0,
    ) -> bytes:
        """
        Fire a classic request (non-JetStream). Useful for synchronous RPC-style calls.
        """
        assert self._nc is not None, "Call start() first."
        if not isinstance(payload, (bytes, bytearray)):
            payload = encode_message(payload, headers=headers)
        msg = await self._nc.request(subject=subject, payload=payload, headers=headers, timeout=timeout)
        return msg.data


# -------------------- Example usage --------------------
async def _demo():
    from ..events.market import Tick

    producer = NATSProducer(
        servers=os.getenv("NATS_SERVERS", "nats://127.0.0.1:4222"),
        stream=os.getenv("NATS_STREAM", "BUS"),
        subject=os.getenv("NATS_SUBJECT", "market.ticks.v1"),
        ensure_stream=True,
    )
    await producer.start()

    tick = Tick.create(symbol="AAPL", price=190.25, size=100, bid=190.20, ask=190.30, source="nasdaq")
    stream_seq, _ = await producer.publish(event=tick, msg_id=f"tick-{tick.symbol}-{tick.ts_event}")
    print(f"Published Tick -> seq={stream_seq}")

    # Batch publish
    ticks = [
        Tick.create(symbol="MSFT", price=420.0, size=50, bid=419.9, ask=420.1, source="nasdaq"),
        Tick.create(symbol="GOOGL", price=130.0, size=75, bid=129.9, ask=130.1, source="nasdaq"),
    ]
    res = await producer.publish_batch(ticks)
    print(f"Published {len(res)} ticks")

    await producer.close()


if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass