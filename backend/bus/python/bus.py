# bus/python/bus.py
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional

from utils.retry import aretry
from utils.serializer import encode_message, decode_message, default_headers
from utils import tracing

# Import your producers/consumers
from producers.kafka_producer import KafkaProducerWrapper
from producers.nats_producer import NATSProducer
from producers.redis_producer import RedisStreamProducer

from consumers.kafka_consumer import KafkaConsumerWrapper  # type: ignore # assuming you implemented
from consumers.nats_consumer import NATSConsumer           # "
from consumers.redis_consumer import RedisStreamConsumer  # "

logger = logging.getLogger("bus")


class EventBus:
    """
    Unified façade for Kafka / NATS / Redis event bus.
    - Backend selected by BUS_BACKEND env ("kafka"|"nats"|"redis").
    - Consistent publish/consume API.
    - Automatic serialization, retry, and tracing.
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or os.getenv("BUS_BACKEND", "kafka")).lower()

        # Producers
        self.kafka: Optional[KafkaProducerWrapper] = None
        self.nats: Optional[NATSProducer] = None
        self.redis: Optional[RedisStreamProducer] = None

        # Consumers
        self.kafka_cons: Optional[KafkaConsumerWrapper] = None
        self.nats_cons: Optional[NATSConsumer] = None
        self.redis_cons: Optional[RedisStreamConsumer] = None

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------
    async def start(self):
        logger.info(f"[bus] starting backend={self.backend}")
        if self.backend == "kafka":
            self.kafka = KafkaProducerWrapper()
            # consumer init left to user

        elif self.backend == "nats":
            self.nats = NATSProducer()
            await self.nats.start()

        elif self.backend == "redis":
            self.redis = RedisStreamProducer()
            await self.redis.start()

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def close(self):
        logger.info("[bus] closing...")
        try:
            if self.nats:
                await self.nats.close()
            if self.redis:
                await self.redis.close()
            # Kafka producer flush/close
            if self.kafka:
                self.kafka.flush()
        except Exception as e:
            logger.warning(f"[bus] error during close: {e}")

    # -------------------------------------------------------------------
    # Publish
    # -------------------------------------------------------------------
    @aretry(tries=3, base_delay=0.2, max_delay=2.0)
    async def publish(
        self,
        topic: str,
        event: Any,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        sync: bool = False,
    ):
        """
        Publish an event to the selected backend.
        - Adds tracing headers.
        - Auto encodes payload.
        - Retries on transient failures.
        """
        headers = headers or default_headers()
        with tracing.start_produce_span(topic, key=key, headers=headers, messaging_system=self.backend):
            if self.backend == "kafka" and self.kafka:
                self.kafka.publish(topic, event, key=key, headers=headers, sync=sync)

            elif self.backend == "nats" and self.nats:
                await self.nats.publish(event, subject=topic, headers=headers, msg_id=key)

            elif self.backend == "redis" and self.redis:
                await self.redis.publish(event, headers=headers, msg_id=key)

            else:
                raise RuntimeError(f"Producer not initialized for backend={self.backend}")

    # -------------------------------------------------------------------
    # Consume
    # -------------------------------------------------------------------
    async def consume(self, topic: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator: yields decoded events from the configured backend.
        - Wraps each message in a CONSUMER span.
        """
        if self.backend == "kafka" and self.kafka_cons:
            async for msg in self.kafka_cons.consume(topic):
                with tracing.start_consume_span(topic, key=msg.key, headers=msg.headers, messaging_system="kafka"):
                    yield decode_message(msg.value, msg.headers)

        elif self.backend == "nats" and self.nats_cons:
            async for msg in self.nats_cons.consume(topic): # type: ignore
                with tracing.start_consume_span(topic, headers=msg.headers, messaging_system="nats"):
                    yield decode_message(msg.data, msg.headers)

        elif self.backend == "redis" and self.redis_cons:
            async for msg in self.redis_cons.consume(self.redis.stream): # type: ignore
                with tracing.start_consume_span(topic, headers=msg.headers, messaging_system="redis"):
                    yield decode_message(msg.payload, msg.headers)

        else:
            raise RuntimeError(f"Consumer not initialized for backend={self.backend}")


# -------------------------------------------------------------------
# Example
# -------------------------------------------------------------------
async def _demo():
    from events.market import Tick

    bus = EventBus(backend="nats")  # or "kafka" / "redis"
    await bus.start()

    tick = Tick.create(symbol="AAPL", price=191.2, size=100, bid=191.1, ask=191.3, source="nasdaq")
    await bus.publish("market.ticks.v1", tick, key=tick.symbol)

    await bus.close()


if __name__ == "__main__":
    asyncio.run(_demo())