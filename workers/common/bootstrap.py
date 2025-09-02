# platform/bootstrap.py
"""
Bootstrap module for all workers / services.

Centralizes:
  - Logging config
  - Env loading
  - Redis connection
  - OpenTelemetry tracer + Prometheus metrics
  - Entitlements policy
  - Market calendars
  - Audit logger
  - DLQ handler
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Tuple

import redis

# Local platform modules
from platform import otel, entitlements, calendars, audit, dlq # type: ignore

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def _init_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# Redis
# --------------------------------------------------------------------------

def _init_redis() -> redis.Redis:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def init(service_name: str) -> Dict[str, Any]:
    """
    Initialize all platform services for a given worker/service.

    Returns a dict containing:
      - tracer: OpenTelemetry tracer
      - metrics: WorkerMetrics helper
      - redis: Redis client
      - ent: Entitlements checker
      - cal: CalendarRegistry
      - audit: AuditLogger
      - dlq: DeadLetterQueue
    """
    # Logging
    _init_logging(os.getenv("LOG_LEVEL", "INFO"))
    log = logging.getLogger(service_name)
    log.info("Bootstrapping service=%s env=%s", service_name, os.getenv("DEPLOY_ENV", "prod"))

    # Redis
    r = _init_redis()
    try:
        r.ping()
        log.info("Connected to Redis %s:%s", r.connection_pool.connection_kwargs["host"], r.connection_pool.connection_kwargs["port"])
    except Exception as e:
        log.error("Redis not reachable: %s", e)
        # Still return client; workers may fail gracefully

    # OTEL: tracing + metrics
    tracer, metrics = otel.init(service_name)

    # Entitlements
    policy_path = os.getenv("ENTITLEMENTS_PATH", "configs/policies/entitlements.yml")
    ent = entitlements.Entitlements(policy_path)

    # Calendars
    cal_dir = os.getenv("CALENDARS_DIR", "configs/calendars")
    cal = calendars._registry(cal_dir)

    # Audit
    aud = audit.AuditLogger(service_name, redis_client=r)

    # DLQ
    dlq_handler = dlq.DeadLetterQueue(worker_name=service_name, redis_client=r)

    log.info("Bootstrap complete for service=%s", service_name)

    return {
        "tracer": tracer,
        "metrics": metrics,
        "redis": r,
        "ent": ent,
        "cal": cal,
        "audit": aud,
        "dlq": dlq_handler,
    }