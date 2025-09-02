# platform/dlq.py
"""
Dead Letter Queue (DLQ) utility for workers.

- Push failed messages to Redis Stream `STREAM_DLQ` (default).
- Includes metadata: reason, worker, correlation id, retries, ts.
- Messages can later be inspected or replayed.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

try:
    from redis import Redis  # type: ignore
except Exception:  # pragma: no cover
    Redis = None  # type: ignore


class DeadLetterQueue:
    def __init__(
        self,
        worker_name: str,
        redis_client: Optional["Redis"] = None, # type: ignore
        stream_name: str = os.getenv("DLQ_STREAM", "STREAM_DLQ"),
    ) -> None:
        self.worker = worker_name
        self.r = redis_client
        self.stream = stream_name

    def push(
        self,
        payload: Dict[str, Any],
        reason: str,
        corr_id: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> None:
        """Send message to DLQ stream with metadata."""
        entry = {
            "worker": self.worker,
            "reason": reason,
            "corr_id": corr_id or "",
            "retries": str(retries or 0),
            "ts": str(int(time.time())),
            "payload": json.dumps(payload, separators=(",", ":")),
        }
        if self.r:
            try:
                self.r.xadd(self.stream, entry)
            except Exception:
                # never raise from DLQ
                pass
        else:
            # fallback to local file if no Redis client
            with open(f"{self.worker}_dlq.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def replay(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Read and return last N DLQ messages (JSON decoded)."""
        if not self.r:
            return []
        msgs = self.r.xrevrange(self.stream, count=limit)
        out = []
        for mid, fields in msgs:
            try:
                payload = json.loads(fields.get("payload", "{}"))
            except Exception:
                payload = {}
            out.append({
                "id": mid,
                "worker": fields.get("worker"),
                "reason": fields.get("reason"),
                "corr_id": fields.get("corr_id"),
                "retries": fields.get("retries"),
                "ts": fields.get("ts"),
                "payload": payload,
            })
        return out