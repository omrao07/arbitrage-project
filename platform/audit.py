# platform/audit.py
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    # Optional at runtime—only needed if you pass a Redis client
    from redis import Redis  # type: ignore
except Exception:  # pragma: no cover
    Redis = None  # type: ignore


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _now_iso() -> str:
    return time.strftime(ISO_FORMAT, time.gmtime())


def _sha256(obj: Any) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Basic PII scrubber (configurable via AUDIT_SCRUB_KEYS env, comma-separated)
_SCRUB_DEFAULTS = {"ssn", "pan", "aadhaar", "password", "token", "api_key", "secret", "email", "phone"}


def _scrub(obj: Any, extra_keys: Optional[set[str]] = None) -> Any:
    keys_to_scrub = _SCRUB_DEFAULTS | (extra_keys or set())
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in keys_to_scrub or any(tag in lk for tag in ("pwd", "pass", "secret", "key")):
                clean[k] = "***"
            else:
                clean[k] = _scrub(v, extra_keys)
        return clean
    if isinstance(obj, list):
        return [_scrub(v, extra_keys) for v in obj]
    return obj


@dataclass(frozen=True)
class AuditRecord:
    ts: str
    action: str
    resource: str
    user: Optional[str]
    corr_id: Optional[str]
    region: Optional[str]
    policy_hash: Optional[str]
    input_sha: str
    details: Dict[str, Any]
    host: str
    service: str
    version: int = 1


class AuditLogger:
    """
    Append-only audit logger.

    - Writes to Redis Stream (XADD) if a Redis client is provided.
    - Always writes to a local JSONL file as a fallback/secondary sink.
    - Scrubs common PII keys from 'details' before persisting.

    Usage:
        r = redis.Redis(host=os.getenv("REDIS_HOST","localhost"), port=int(os.getenv("REDIS_PORT","6379")), decode_responses=True)
        audit = AuditLogger(service_name="analyst-worker", redis_client=r)
        audit.record(action="run_scenario", resource="scenarios/vol_up",
                     user="alice@fund", corr_id="trace-123",
                     region="IN", policy_hash="ddqn_v3@e1d2",
                     details={"symbols":["AAPL","RELIANCE.NS"], "params":{"shock":2.0}},
                     input_for_hash={"symbols":["AAPL","RELIANCE.NS"], "params":{"shock":2.0}})
    """

    def __init__(
        self,
        service_name: str,
        redis_client: Optional["Redis"] = None, # type: ignore
        stream_name: str = os.getenv("AUDIT_STREAM", "STREAM_AUDIT"),
        jsonl_path: str = os.getenv("AUDIT_JSONL_PATH", "./audit.log.jsonl"),
        scrub_extra_keys: Optional[list[str]] = None,
    ) -> None:
        self.service = service_name
        self.r = redis_client
        self.stream = stream_name
        self.jsonl_path = jsonl_path
        self.host = socket.gethostname()
        # Parse extra scrub keys from env + param
        extra_env = {k.strip().lower() for k in os.getenv("AUDIT_SCRUB_KEYS", "").split(",") if k.strip()}
        self.scrub_keys = set(scrub_extra_keys or []) | extra_env

    # ---- Public API -----------------------------------------------------

    def record(
        self,
        *,
        action: str,
        resource: str,
        user: Optional[str],
        corr_id: Optional[str],
        region: Optional[str],
        policy_hash: Optional[str],
        details: Dict[str, Any],
        input_for_hash: Any,
    ) -> None:
        """
        Create and persist an audit record.

        - `details` is scrubbed for PII.
        - `input_for_hash` forms the immutable fingerprint for reproducibility.
        """
        ts = _now_iso()
        record = AuditRecord(
            ts=ts,
            action=action,
            resource=resource,
            user=user,
            corr_id=corr_id,
            region=region,
            policy_hash=policy_hash,
            input_sha=_sha256(input_for_hash),
            details=_scrub(details, self.scrub_keys),
            host=self.host,
            service=self.service,
        )
        payload = asdict(record)

        # Best-effort: write to Redis stream (if available), then to JSONL
        self._write_redis(payload)
        self._write_jsonl(payload)

    # ---- Internal sinks -------------------------------------------------

    def _write_redis(self, payload: Dict[str, Any]) -> None:
        if not self.r:
            return
        try:
            # Flatten for XADD: keep one 'payload' field to avoid huge entries
            self.r.xadd(self.stream, {"topic": "audit", "payload": json.dumps(payload, separators=(",", ":"))})
        except Exception:
            # Never raise—auditing must not crash the caller
            pass

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            # Last resort: drop on the floor rather than breaking the app
            pass