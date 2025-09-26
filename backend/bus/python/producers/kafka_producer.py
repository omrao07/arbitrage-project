# bus/python/producers/kafka_producer.py
from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Dict, Optional

from confluent_kafka import Producer # type: ignore


class KafkaProducerWrapper:
    """
    Simple Kafka producer wrapper.
    - Uses Avro/JSON (JSON here, but can be extended).
    - Adds metadata (env, host).
    - Auto-retry and flush support.
    """

    def __init__(
        self,
        brokers: Optional[str] = None,
        client_id: Optional[str] = None,
        acks: str = "all",
        linger_ms: int = 5,
    ):
        brokers = brokers or os.getenv("KAFKA_BROKERS", "localhost:9092")
        client_id = client_id or f"{socket.gethostname()}-{int(time.time())}"

        self._config: Dict[str, Any] = {
            "bootstrap.servers": brokers,
            "client.id": client_id,
            "acks": acks,
            "linger.ms": linger_ms,
        }
        self._producer = Producer(self._config)

    # ---------------------------
    # internal delivery report
    # ---------------------------
    def _delivery_report(self, err, msg):
        if err is not None:
            print(f"[KafkaProducer] Delivery failed for {msg.topic()} [{msg.partition()}]: {err}")
        else:
            print(
                f"[KafkaProducer] Delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}"
            )

    # ---------------------------
    # publish JSON-serializable event
    # ---------------------------
    def publish(
        self,
        topic: str,
        event: Any,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        sync: bool = False,
    ) -> None:
        """
        Publish an event (dataclass or dict) to a Kafka topic.
        """
        if hasattr(event, "to_dict"):
            payload = event.to_dict()
        elif hasattr(event, "__dict__"):
            payload = event.__dict__
        elif isinstance(event, dict):
            payload = event
        else:
            raise TypeError("Event must be dataclass, dict, or have to_dict()")

        payload["env"] = os.getenv("APP_ENV", "prod")
        payload["emitted_at"] = int(time.time() * 1000)

        self._producer.produce(
            topic=topic,
            key=key,
            value=json.dumps(payload, separators=(",", ":")),
            headers=headers,
            callback=self._delivery_report,
        )

        if sync:
            self._producer.flush()

    def flush(self):
        """Flush outstanding messages."""
        self._producer.flush()


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    from events.market import Tick

    producer = KafkaProducerWrapper()

    # Create a sample tick
    tick = Tick.create(symbol="AAPL", price=190.25, size=100, bid=190.20, ask=190.30, source="nasdaq")

    producer.publish(topic="market.ticks.v1", event=tick, key=tick.symbol, sync=True)