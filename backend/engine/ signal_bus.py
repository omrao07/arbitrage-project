# backend/engine/signal_bus.py
"""
Signal Bus: central dispatcher for market-relevant signals.

Consumes:
- streams.STREAM_ALT_SIGNALS      (alt-data, sentiment, biodata, climate composites)
- streams.STREAM_POLICY_SIGNALS   (policy sandbox / central-bank AI, optional)

Maintains in Redis:
- H_SIG_LATEST  : series_id -> {"ts": str, "value": float, "metric": str, "region": str}
- H_SIG_EMA     : series_id -> {"v": float, "ts": str, "alpha": float}
- H_SIG_STATS   : series_id -> {"count": int}  (tiny usage stats)

Publishes (Redis PubSub):
- chan.signals.updates : light messages for dashboards/other services

Env/config:
- FEATURE_* flags from backend.config.feature_flags gate which sources we consume
- SIGBUS_EMA_ALPHA (default 0.2)
- SIGBUS_THROTTLE_MS (default 1000)
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.config.feature_flags import is_enabled
from backend.bus import streams

log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Redis keys / channels
# ---------------------------------------------------------------------

H_SIG_LATEST = "sigbus:latest"   # hash: series_id -> JSON {"ts","value","metric","region"}
H_SIG_EMA    = "sigbus:ema"      # hash: series_id -> JSON {"v","ts","alpha"}
H_SIG_STATS  = "sigbus:stats"    # hash: series_id -> JSON {"count": int}

CHAN_SIG_UPDATES = "chan.signals.updates"  # pubsub: {"series_id","ts","value","ema","metric","region"}

# Tunables
EMA_ALPHA = float(os.getenv("SIGBUS_EMA_ALPHA", "0.2"))
THROTTLE_MS = int(os.getenv("SIGBUS_THROTTLE_MS", "1000"))

# We’ll read from these streams conditionally via feature flags
INPUT_STREAMS = tuple(
    s for s in [
        streams.STREAM_ALT_SIGNALS,       # alt-data + derived composites (sentiment, bio, climate) # type: ignore
        streams.STREAM_POLICY_SIGNALS,    # policy sandbox / central-bank AI # type: ignore
    ] if s
)


def _ema(prev: Optional[float], x: float, alpha: float = EMA_ALPHA) -> float:
    if prev is None:
        return x
    return alpha * x + (1.0 - alpha) * prev


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class SignalBus:
    """
    - Consumes incoming signal records
    - Validates and normalizes minimal fields
    - Stores latest & EMA per series in Redis
    - Publishes light updates for UI/consumers
    """

    def __init__(self, group: str = "signal_bus", consumer: Optional[str] = None):
        self.group = group
        self.consumer = consumer or f"sigbus_{int(time.time())}"
        # simple throttle map: series_id -> last_publish_ms
        self._last_pub_ms: Dict[str, int] = {}

    # ------------------------ core processing ------------------------

    def _allowed(self) -> bool:
        """Gate by feature flags: if *all* relevant inputs disabled, no-op."""
        any_enabled = any([
            is_enabled("ALTDATA"),
            is_enabled("SENTIMENT"),
            is_enabled("BIODATA"),
            is_enabled("CLIMATE"),
            is_enabled("SANDBOX"),
            is_enabled("CBANK_AI"),
        ])
        return any_enabled

    def _validate_min(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Expect at least:
          series_id: str
          timestamp: str (ISO)
          value: float
        Optional:
          metric: str, region: str, meta: dict
        """
        sid = str(obj.get("series_id", "")).strip()
        ts = str(obj.get("timestamp", "")).strip()
        try:
            val = float(obj.get("value")) # type: ignore
        except Exception:
            return None
        if not sid or not ts:
            return None

        metric = str(obj.get("metric", "")).strip()
        region = str(obj.get("region", "GLOBAL")).strip() or "GLOBAL"
        return {"series_id": sid, "timestamp": ts, "value": val, "metric": metric, "region": region}

    def _throttled(self, series_id: str) -> bool:
        if THROTTLE_MS <= 0:
            return False
        now_ms = int(time.time() * 1000)
        last = self._last_pub_ms.get(series_id, 0)
        if now_ms - last < THROTTLE_MS:
            return True
        self._last_pub_ms[series_id] = now_ms
        return False

    def _update_latest(self, sid: str, ts: str, val: float, metric: str, region: str) -> None:
        streams.hset(H_SIG_LATEST, sid, {"ts": ts, "value": float(val), "metric": metric, "region": region})
        st = streams.hgetall(H_SIG_STATS).get(sid) or {"count": 0}
        try:
            cnt = int(st.get("count", 0)) + 1
        except Exception:
            cnt = 1
        streams.hset(H_SIG_STATS, sid, {"count": cnt})

    def _update_ema(self, sid: str, ts: str, val: float) -> float:
        st = streams.hgetall(H_SIG_EMA).get(sid)
        prev = None
        if isinstance(st, dict):
            prev = _float(st.get("v"), None)  # type: ignore[arg-type]
        ema_v = _ema(prev, val, EMA_ALPHA)
        streams.hset(H_SIG_EMA, sid, {"v": float(ema_v), "ts": ts, "alpha": EMA_ALPHA})
        return ema_v

    def _publish_update(self, sid: str, ts: str, val: float, ema: float, metric: str, region: str) -> None:
        # Lightweight pubsub message for dashboards / listeners
        streams.publish_pubsub(
            CHAN_SIG_UPDATES,
            {
                "series_id": sid,
                "timestamp": ts,
                "value": float(val),
                "ema": float(ema),
                "metric": metric,
                "region": region,
            },
        )

    def _process_payload(self, payload: Dict[str, Any]) -> None:
        m = self._validate_min(payload)
        if not m:
            return

        sid, ts, val, metric, region = m["series_id"], m["timestamp"], m["value"], m["metric"], m["region"]

        # Persist latest & EMA
        self._update_latest(sid, ts, val, metric, region)
        ema_v = self._update_ema(sid, ts, val)

        # Throttle chatter before pubsub fanout
        if not self._throttled(sid):
            self._publish_update(sid, ts, val, ema_v, metric, region)

    # ------------------------ main loop -----------------------------

    def run_forever(self, block_ms: int = 1000, count: int = 256) -> None:
        """
        Consumer-group loop across all input streams. Safe to run multiple replicas.
        """
        if not self._allowed():
            log.info("All relevant features disabled; SignalBus not running.")
            return

        log.info("SignalBus starting (group=%s consumer=%s)", self.group, self.consumer)

        streams_to_read = []
        # Respect feature flags to reduce load
        if is_enabled("ALTDATA") or is_enabled("SENTIMENT") or is_enabled("BIODATA") or is_enabled("CLIMATE"):
            streams_to_read.append(streams.STREAM_ALT_SIGNALS) # type: ignore
        if is_enabled("SANDBOX") or is_enabled("CBANK_AI"):
            streams_to_read.append(streams.STREAM_POLICY_SIGNALS) # type: ignore

        if not streams_to_read:
            log.info("No streams to read based on feature flags.")
            return

        while True:
            try:
                resp = streams.consume_stream(
                    streams=streams_to_read if len(streams_to_read) > 1 else streams_to_read[0], # type: ignore
                    group=self.group, # type: ignore
                    consumer=self.consumer, # type: ignore
                    last_ids=">", # type: ignore
                    block_ms=block_ms,
                    count=count,
                    ack=True, # type: ignore
                )
                if not resp:
                    continue

                for _sname, entries in resp:
                    for entry_id, fields in entries:
                        try:
                            raw = fields.get(b"data")
                            if not raw:
                                continue
                            payload: Dict[str, Any] = streams._loads(raw)
                            self._process_payload(payload)
                        except Exception as e:
                            log.exception("SignalBus failed on entry %s: %s", entry_id, e)
                        finally:
                            # Ack regardless to avoid stalling; bad payloads won't block the group
                            try:
                                # Using ack=True in consume_stream, but extra safety is okay
                                streams.xack(_sname, self.group, entry_id) # type: ignore
                            except Exception:
                                pass
            except Exception as loop_err:
                log.exception("SignalBus loop error: %s", loop_err)
                time.sleep(1.0)


# ------------------------ CLI ------------------------

if __name__ == "__main__":
    SignalBus().run_forever()