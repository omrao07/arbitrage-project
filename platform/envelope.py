# platform/envelope.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Optional, Tuple

ISO_MS = "%Y-%m-%dT%H:%M:%S"

CURRENT_SCHEMA_VERSION = 1
SUPPORTED_VERSIONS = {1}


def _now_iso_ms() -> str:
    # e.g. 2025-09-02T12:34:56.123Z
    t = time.gmtime()
    base = time.strftime(ISO_MS, t)
    ms = int((time.time() % 1) * 1000)
    return f"{base}.{ms:03d}Z"


@dataclass(frozen=True)
class Envelope:
    """
    Canonical event/message envelope used across all services.

    Fields:
      schema.name     : logical topic name (e.g., "risk.var.request")
      schema.version  : integer schema version
      id              : UUID4 idempotency key
      ts              : ISO timestamp with milliseconds (UTC)
      corr_id         : correlation id for tracing (UUID4 if not provided)
      producer        : dict(meta), e.g. {"svc":"api","region":"US","node":"..."}
      entitlements    : list[str] of claims needed to consume/process this message
      payload         : domain-specific JSON object
    """
    schema: Dict[str, Any]
    id: str
    ts: str
    corr_id: str
    producer: Dict[str, Any] = field(default_factory=dict)
    entitlements: Iterable[str] = field(default_factory=tuple)
    payload: Dict[str, Any] = field(default_factory=dict)

    # ----------------------- Serialization helpers -----------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=False)

    def flatten_for_stream(self) -> Dict[str, str]:
        """
        Flattened map ideal for Redis Streams / Kafka headers-less bodies.
        Keeps topic in a 'topic' field and JSON in 'payload'.
        """
        return {"topic": self.schema["name"], "payload": self.to_json()}

    # -------------------------- Validation -------------------------------

    def require(self, keys: Iterable[str]) -> None:
        missing = [k for k in keys if k not in self.payload]
        if missing:
            raise ValueError(f"schema_mismatch missing_fields={missing}")

    def assert_version_supported(self) -> None:
        v = int(self.schema.get("version", -1))
        if v not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported_schema_version={v} supported={sorted(SUPPORTED_VERSIONS)}")

    # -------------------------- Convenience ------------------------------

    @property
    def topic(self) -> str:
        return str(self.schema.get("name", ""))

    @property
    def version(self) -> int:
        return int(self.schema.get("version", -1))


# ---------------------------- Constructors -------------------------------

def new(
    *,
    schema_name: str,
    payload: Dict[str, Any],
    corr_id: Optional[str] = None,
    producer: Optional[Dict[str, Any]] = None,
    entitlements: Optional[Iterable[str]] = None,
    schema_version: int = CURRENT_SCHEMA_VERSION,
) -> Envelope:
    """Create a fresh Envelope with sane defaults."""
    env = Envelope(
        schema={"name": schema_name, "version": int(schema_version)},
        id=str(uuid.uuid4()),
        ts=_now_iso_ms(),
        corr_id=corr_id or str(uuid.uuid4()),
        producer=producer or {},
        entitlements=tuple(entitlements or ()),
        payload=payload or {},
    )
    env.assert_version_supported()
    return env


def parse(raw: str | Dict[str, Any]) -> Envelope:
    """
    Parse an incoming message (JSON string or dict) and validate minimal shape.
    Raises ValueError on structural issues.
    """
    obj = json.loads(raw) if isinstance(raw, str) else dict(raw)

    # Minimal structural checks
    if not isinstance(obj.get("schema"), dict) or "name" not in obj["schema"]:
        raise ValueError("invalid_envelope: missing schema.name")
    if "version" not in obj["schema"]:
        raise ValueError("invalid_envelope: missing schema.version")
    if "payload" not in obj:
        raise ValueError("invalid_envelope: missing payload")
    for k in ("id", "ts", "corr_id"):
        if k not in obj:
            raise ValueError(f"invalid_envelope: missing {k}")

    env = Envelope(
        schema={"name": obj["schema"]["name"], "version": int(obj["schema"]["version"])},
        id=str(obj["id"]),
        ts=str(obj["ts"]),
        corr_id=str(obj["corr_id"]),
        producer=dict(obj.get("producer") or {}),
        entitlements=tuple(obj.get("entitlements") or ()),
        payload=dict(obj.get("payload") or {}),
    )
    env.assert_version_supported()
    return env


# ------------------------- Utility functions -----------------------------

def require_fields(env_or_payload: Envelope | Dict[str, Any], keys: Iterable[str]) -> None:
    """
    Require keys in payload regardless of whether you pass an Envelope or a payload dict.
    """
    payload = env_or_payload.payload if isinstance(env_or_payload, Envelope) else env_or_payload
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"schema_mismatch missing_fields={missing}")


def correlation_tuple(env: Envelope) -> Tuple[str, str]:
    """Return (topic, corr_id) for quick logging/metrics."""
    return env.topic, env.corr_id


def is_duplicate(seen_set, msg_id: str, ttl_seconds: Optional[int] = None) -> bool:
    """
    Lightweight idempotency helper for Redis-like SETs:
      if not seen → add and return False
      if seen     → return True

    seen_set is expected to implement:
      - setnx(key, value) -> bool
      - expire(key, ttl)  -> None
    """
    key = f"seen:{msg_id}"
    try:
        # setnx returns True if the key was set (i.e., not seen before)
        if seen_set.setnx(key, 1):
            if ttl_seconds:
                seen_set.expire(key, int(ttl_seconds))
            return False
        return True
    except Exception:
        # On failure, fail open (treat as not duplicate)
        return False