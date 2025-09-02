# platform/env.py
"""
Centralized environment variable loader.

- Provides typed getters: get_str, get_int, get_float, get_bool
- Supports defaults and required vars
- All env config for workers and services should be declared here
"""

from __future__ import annotations

import os
from typing import Optional


class EnvError(RuntimeError):
    pass


def get_str(key: str, default: Optional[str] = None, *, required: bool = False) -> str:
    val = os.getenv(key, default)
    if val is None and required:
        raise EnvError(f"Missing required env var: {key}")
    return val # type: ignore


def get_int(key: str, default: Optional[int] = None, *, required: bool = False) -> int:
    val = os.getenv(key)
    if val is None:
        if required:
            raise EnvError(f"Missing required env var: {key}")
        return int(default) if default is not None else 0
    try:
        return int(val)
    except ValueError:
        raise EnvError(f"Invalid int for {key}: {val}")


def get_float(key: str, default: Optional[float] = None, *, required: bool = False) -> float:
    val = os.getenv(key)
    if val is None:
        if required:
            raise EnvError(f"Missing required env var: {key}")
        return float(default) if default is not None else 0.0
    try:
        return float(val)
    except ValueError:
        raise EnvError(f"Invalid float for {key}: {val}")


def get_bool(key: str, default: Optional[bool] = None) -> bool:
    val = os.getenv(key)
    if val is None:
        return bool(default)
    return str(val).lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------
# Common configuration keys
# ---------------------------------------------------------------------

# Redis
REDIS_HOST = get_str("REDIS_HOST", "localhost")
REDIS_PORT = get_int("REDIS_PORT", 6379)
REDIS_DB = get_int("REDIS_DB", 0)

# Streams
STREAM_ORDERS = get_str("STREAM_ORDERS", "STREAM_ORDERS")
STREAM_FILLS = get_str("STREAM_FILLS", "STREAM_FILLS")
STREAM_AUDIT = get_str("STREAM_AUDIT", "STREAM_AUDIT")
STREAM_DLQ = get_str("STREAM_DLQ", "STREAM_DLQ")

# OTEL / metrics
OTEL_ENDPOINT = get_str("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
METRICS_PORT = get_int("METRICS_PORT", 9090)

# Deployment info
DEPLOY_ENV = get_str("DEPLOY_ENV", "dev")
SERVICE_VERSION = get_str("SERVICE_VERSION", os.getenv("GIT_SHA", "0.0.0"))
REGION = get_str("REGION", "US")

# Policy/config paths
ENTITLEMENTS_PATH = get_str("ENTITLEMENTS_PATH", "configs/policies/entitlements.yml")
CALENDARS_DIR = get_str("CALENDARS_DIR", "configs/calendars")
RISK_GUARD_PATH = get_str("RISK_GUARD_PATH", "configs/policies/risk_guard.yml")