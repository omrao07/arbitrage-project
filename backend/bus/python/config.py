# bus/python/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ===============================
# Kafka configuration
# ===============================
@dataclass
class KafkaConfig:
    brokers: str = os.getenv("KAFKA_BROKERS", "localhost:9092")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "hyper-os-consumer")
    acks: str = os.getenv("KAFKA_ACKS", "all")         # "0" | "1" | "all"
    security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    sasl_mechanism: Optional[str] = os.getenv("KAFKA_SASL_MECHANISM")  # "PLAIN"|"SCRAM-SHA-256"|"SCRAM-SHA-512"
    sasl_username: Optional[str] = os.getenv("KAFKA_SASL_USERNAME")
    sasl_password: Optional[str] = os.getenv("KAFKA_SASL_PASSWORD")


# ===============================
# NATS configuration
# ===============================
@dataclass
class NATSConfig:
    servers: str = os.getenv("NATS_SERVERS", "nats://127.0.0.1:4222")
    stream: str = os.getenv("NATS_STREAM", "BUS")
    subject: str = os.getenv("NATS_SUBJECT", "market.ticks.v1")
    tls: bool = os.getenv("NATS_TLS", "false").lower() == "true"
    user: Optional[str] = os.getenv("NATS_USER")
    password: Optional[str] = os.getenv("NATS_PASSWORD")
    token: Optional[str] = os.getenv("NATS_TOKEN")


# ===============================
# Redis configuration
# ===============================
@dataclass
class RedisConfig:
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    stream: str = os.getenv("REDIS_STREAM", "bus.market.ticks.v1")
    maxlen: Optional[int] = (
        int(os.getenv("REDIS_STREAM_MAXLEN")) if os.getenv("REDIS_STREAM_MAXLEN") else None # type: ignore
    )
    idempotency_ttl_sec: int = int(os.getenv("REDIS_IDEMP_TTL", "120"))


# ===============================
# Global bus configuration
# ===============================
@dataclass
class BusConfig:
    backend: str = os.getenv("BUS_BACKEND", "kafka").lower()  # "kafka"|"nats"|"redis"
    app_env: str = os.getenv("APP_ENV", "prod")
    region: Optional[str] = os.getenv("REGION")

    kafka: KafkaConfig = KafkaConfig()
    nats: NATSConfig = NATSConfig()
    redis: RedisConfig = RedisConfig()


# Singleton-ish access
bus_config = BusConfig()


if __name__ == "__main__":
    # Quick dump for sanity check
    from pprint import pprint
    pprint(bus_config)