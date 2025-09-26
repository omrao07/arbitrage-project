# bus/python/consumers/kafka_consumer.py
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Dict, Any

from confluent_kafka import Consumer, KafkaError, KafkaException # type: ignore
from confluent_kafka.serialization import StringDeserializer # type: ignore

# Local utilities (you already have these in bus/python/utils/)
try:
    from ..utils.serializer import decode_message  # type: ignore # (payload: bytes, headers) -> Any
except Exception:  # fallback if utils not wired yet
    def decode_message(payload: bytes, headers: Optional[Iterable] = None) -> Any:
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception:
            return payload  # raw bytes if not JSON

try:
    from ..utils.tracing import start_consume_span  # type: ignore # (topic, key, headers) -> context manager
except Exception:
    from contextlib import nullcontext
    def start_consume_span(*_args, **_kwargs):
        return nullcontext()


log = logging.getLogger(__name__)


@dataclass
class KafkaConsumerConfig:
    brokers: str = os.getenv("KAFKA_BROKERS", "localhost:9092")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "hyper-os-consumer")
    client_id: str = os.getenv("KAFKA_CLIENT_ID", "hyper-os")
    enable_auto_commit: bool = False
    auto_offset_reset: str = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")  # earliest|latest
    security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")  # SSL|SASL_SSL|PLAINTEXT
    sasl_mechanism: Optional[str] = os.getenv("KAFKA_SASL_MECHANISM")  # e.g., PLAIN, SCRAM-SHA-512
    sasl_username: Optional[str] = os.getenv("KAFKA_SASL_USERNAME")
    sasl_password: Optional[str] = os.getenv("KAFKA_SASL_PASSWORD")
    session_timeout_ms: int = 45_000
    max_poll_interval_ms: int = 300_000
    fetch_min_bytes: int = 1
    fetch_max_bytes: int = 5_000_000
    max_in_flight: int = 8
    # commit settings
    commit_sync: bool = True
    commit_interval_sec: float = 5.0
    # backoff
    max_retries: int = 5
    base_backoff_sec: float = 0.5
    max_backoff_sec: float = 8.0
    # deserialization
    key_is_string: bool = True

    def to_kafka_conf(self) -> Dict[str, Any]:
        conf = {
            "bootstrap.servers": self.brokers,
            "group.id": self.group_id,
            "client.id": self.client_id,
            "enable.auto.commit": self.enable_auto_commit,
            "auto.offset.reset": self.auto_offset_reset,
            "session.timeout.ms": self.session_timeout_ms,
            "max.poll.interval.ms": self.max_poll_interval_ms,
            "fetch.min.bytes": self.fetch_min_bytes,
            "fetch.max.bytes": self.fetch_max_bytes,
            "max.in.flight.requests.per.connection": self.max_in_flight,
            "security.protocol": self.security_protocol,
        }
        if self.sasl_mechanism:
            conf.update({
                "sasl.mechanisms": self.sasl_mechanism,
                "sasl.username": self.sasl_username or "",
                "sasl.password": self.sasl_password or "",
            })
        return conf


class KafkaConsumerClient:
    """
    Thin wrapper around confluent-kafka Consumer with:
    - safe shutdown
    - manual commit
    - exponential backoff on handler errors
    - OpenTelemetry span hooks (optional)
    - pluggable decoder (Avro/JSON), via utils.serializer.decode_message
    """

    def __init__(
        self,
        topics: Iterable[str],
        config: Optional[KafkaConsumerConfig] = None,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        health_probe: Optional[Callable[[], None]] = None,
    ):
        self.cfg = config or KafkaConsumerConfig()
        self.topics = list(topics)
        self.on_message = on_message
        self.health_probe = health_probe
        self._stop = threading.Event()
        self._consumer: Optional[Consumer] = None
        self._last_commit = time.time()
        self._key_deser = StringDeserializer("utf_8") if self.cfg.key_is_string else None

    # ------------- lifecycle -------------

    def start(self) -> None:
        self._consumer = Consumer(self.cfg.to_kafka_conf())
        self._consumer.subscribe(self.topics, on_assign=self._on_assign, on_revoke=self._on_revoke) # type: ignore
        log.info("KafkaConsumer started: topics=%s group_id=%s brokers=%s",
                 self.topics, self.cfg.group_id, self.cfg.brokers)
        self._install_signal_handlers()

    def stop(self) -> None:
        self._stop.set()
        if self._consumer:
            try:
                self._consumer.close()
            except Exception as e:
                log.warning("Error closing consumer: %s", e)

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.info("Signal %s received -> stopping consumer", signum)
            self.stop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, handler)
            except Exception:
                # If running in a non-main thread or certain runtimes, ignore
                pass

    # ------------- callbacks -------------

    def _on_assign(self, _consumer, partitions):
        parts = ", ".join([f"{p.topic}:{p.partition}@{p.offset}" for p in partitions])
        log.info("Partitions assigned: %s", parts)

    def _on_revoke(self, _consumer, partitions):
        parts = ", ".join([f"{p.topic}:{p.partition}" for p in partitions])
        log.info("Partitions revoked: %s", parts)

    # ------------- main loop -------------

    def run_forever(self, poll_timeout_sec: float = 1.0) -> None:
        if not self._consumer:
            self.start()

        retries = 0
        while not self._stop.is_set():
            try:
                msg = self._consumer.poll(poll_timeout_sec) # type: ignore
                if msg is None:
                    self._maybe_commit()
                    if self.health_probe:
                        self.health_probe()
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue  # benign
                    raise KafkaException(msg.error())

                headers = {k: v for (k, v) in (msg.headers() or [])}
                key = self._key_deser(msg.key(), None) if (self._key_deser and msg.key() is not None) else msg.key()

                with start_consume_span(topic=msg.topic(), key=key, headers=headers):
                    payload = decode_message(msg.value(), headers=headers)
                    event = {
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                        "timestamp": msg.timestamp()[1],  # (type, ts)
                        "key": key,
                        "headers": headers,
                        "value": payload,
                    }

                    if self.on_message:
                        self.on_message(event)

                self._maybe_commit(force=False)
                retries = 0  # reset after successful handle

            except Exception as e:
                retries += 1
                backoff = min(self.cfg.base_backoff_sec * (2 ** (retries - 1)), self.cfg.max_backoff_sec)
                log.exception("Consumer error (attempt %d): %s. Backing off %.2fs", retries, e, backoff)
                time.sleep(backoff)
                if retries >= self.cfg.max_retries:
                    log.error("Max retries reached. Stopping consumer.")
                    self.stop()

        # final commit on graceful exit
        self._maybe_commit(force=True)
        log.info("KafkaConsumer stopped.")

    # ------------- commits -------------

    def _maybe_commit(self, force: bool = False) -> None:
        if self.cfg.enable_auto_commit:
            return
        if not self._consumer:
            return
        now = time.time()
        if force or (now - self._last_commit) >= self.cfg.commit_interval_sec:
            try:
                if self.cfg.commit_sync:
                    self._consumer.commit(asynchronous=False)
                else:
                    self._consumer.commit(asynchronous=True)
                self._last_commit = now
            except Exception as e:
                log.warning("Commit failed: %s", e)


# ---------- quick CLI for local testing ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    topics_env = os.getenv("KAFKA_TOPICS", "market.ticks.v1").split(",")

    def pretty_print(event: Dict[str, Any]) -> None:
        value = event["value"]
        if isinstance(value, (dict, list)):
            v = json.dumps(value)[:500]
        else:
            v = str(value)[:500]
        log.info(
            "[%s p%d@%d] key=%s value=%s",
            event["topic"], event["partition"], event["offset"], event["key"], v
        )

    consumer = KafkaConsumerClient(
        topics=topics_env,
        config=KafkaConsumerConfig(),
        on_message=pretty_print,
    )
    consumer.run_forever()