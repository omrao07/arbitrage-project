# backend/data_ext/biodata/health_signals.py
"""
Health -> Macro Signals transformer.

Listens to normalized biodata metrics (from Apple/Fitbit ingestors) and produces
macro-facing signals:

- BIO-CONSUMPTION  ~ f(steps + sleep - stress)
- BIO-PRODUCTIVITY ~ f(steps + sleep - stress)
- BIO-STRESS       ~ f(resting_heart_rate - sleep)

Input schema (already normalized by Apple/Fitbit ingestors):
    {
      "series_id": "APPLE-resting_heart_rate-US" | "FITBIT-step_count-IN" | ...,
      "timestamp": "2025-08-21T23:00:00Z",
      "region": "US",
      "metric": "resting_heart_rate" | "step_count" | "sleep_hours",
      "value": <float>
    }

Output schema (published to STREAM_ALT_SIGNALS):
    {
      "series_id": "BIO-CONSUMPTION-US",
      "timestamp": "<same as input>",
      "region": "US",
      "metric": "consumption_proxy",
      "value": <float in [-1, 1] approx>,
      "meta": {"ema": true, "components": {"steps": ..., "sleep": ..., "rhr": ...}}
    }

Feature flag: FEATURE_BIODATA must be true.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

from backend.config.feature_flags import is_enabled
from backend.bus import streams

log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Config (env overrides)
# ---------------------------------------------------------------------

# Baselines used to compute % deviations before EMA smoothing
BASE_STEPS = float(os.getenv("BIO_BASELINE_STEPS", "8000"))          # steps/day
BASE_SLEEP = float(os.getenv("BIO_BASELINE_SLEEP_HOURS", "7.0"))     # hours/night
BASE_RHR   = float(os.getenv("BIO_BASELINE_RHR_BPM", "60"))          # bpm resting HR

# EMA smoothing
EMA_ALPHA_STEPS = float(os.getenv("BIO_EMA_ALPHA_STEPS", "0.2"))
EMA_ALPHA_SLEEP = float(os.getenv("BIO_EMA_ALPHA_SLEEP", "0.2"))
EMA_ALPHA_RHR   = float(os.getenv("BIO_EMA_ALPHA_RHR", "0.2"))

# Weights for composite signals
W_STEPS = float(os.getenv("BIO_W_STEPS", "0.5"))
W_SLEEP = float(os.getenv("BIO_W_SLEEP", "0.3"))
W_RHR   = float(os.getenv("BIO_W_RHR",   "0.2"))  # subtractive (stress)

# Redis keys
H_EMA_STEPS = "bio:ema:steps"   # hash field = region, value = JSON {"v":float,"ts":str}
H_EMA_SLEEP = "bio:ema:sleep"
H_EMA_RHR   = "bio:ema:rhr"


def _ema(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None:
        return x
    return alpha * x + (1.0 - alpha) * prev


def _norm_pct(x: float, base: float) -> float:
    """Return (x/base - 1) clipped to reasonable range."""
    if base <= 0:
        return 0.0
    val = (x / base) - 1.0
    # Soft clip to [-1.5, 1.5] then compress into [-1,1]-ish
    if val > 1.5: val = 1.5
    if val < -1.5: val = -1.5
    return val


class HealthSignalTransformer:
    """
    Consumes individual biodata metrics from STREAM_ALT_SIGNALS, maintains EMAs
    per region, and emits composite macro proxies back to STREAM_ALT_SIGNALS.
    """

    def __init__(self, group: str = "bio_signals", consumer: Optional[str] = None):
        self.group = group
        self.consumer = consumer or f"bio_signals_{int(time.time())}"

    # -----------------------------
    # State helpers (Redis hashes)
    # -----------------------------
    def _load_ema(self, hname: str, region: str) -> Optional[float]:
        state = streams.hgetall(hname)
        item = state.get(region)
        if isinstance(item, dict):
            return float(item.get("v", 0.0))
        # backward compat if plain value present
        try:
            return float(item) if item is not None else None
        except Exception:
            return None

    def _store_ema(self, hname: str, region: str, value: float, ts: str) -> None:
        streams.hset(hname, region, {"v": float(value), "ts": ts})

    # -----------------------------
    # Composite calculators
    # -----------------------------
    def _compute_composites(self, region: str, ts: str) -> None:
        """
        Read EMAs for region, compute 3 composites, publish each.
        """
        ema_steps = self._load_ema(H_EMA_STEPS, region)
        ema_sleep = self._load_ema(H_EMA_SLEEP, region)
        ema_rhr   = self._load_ema(H_EMA_RHR,   region)

        if ema_steps is None or ema_sleep is None or ema_rhr is None:
            # Not enough info yet; wait for all three metrics
            return

        # Normalize vs baselines
        steps_z = _norm_pct(ema_steps, BASE_STEPS)
        sleep_z = _norm_pct(ema_sleep, BASE_SLEEP)
        rhr_z   = _norm_pct(ema_rhr,   BASE_RHR)

        # Composite proxies in approx [-1, 1]
        consumption = (W_STEPS * steps_z) + (W_SLEEP * sleep_z) - (W_RHR * rhr_z)
        productivity = (0.6 * steps_z) + (0.4 * sleep_z) - (0.3 * rhr_z)
        stress = (0.7 * rhr_z) - (0.3 * sleep_z)  # higher rhr + lower sleep => stress

        meta = {
            "ema": True,
            "components": {"steps": ema_steps, "sleep": ema_sleep, "rhr": ema_rhr},
            "z": {"steps": steps_z, "sleep": sleep_z, "rhr": rhr_z},
        }

        # Publish three signals
        streams.publish_stream(
            streams.STREAM_ALT_SIGNALS, # type: ignore
            {
                "series_id": f"BIO-CONSUMPTION-{region}",
                "timestamp": ts,
                "region": region,
                "metric": "consumption_proxy",
                "value": float(consumption),
                "meta": meta,
            },
        )
        streams.publish_stream(
            streams.STREAM_ALT_SIGNALS, # type: ignore
            {
                "series_id": f"BIO-PRODUCTIVITY-{region}",
                "timestamp": ts,
                "region": region,
                "metric": "productivity_proxy",
                "value": float(productivity),
                "meta": meta,
            },
        )
        streams.publish_stream(
            streams.STREAM_ALT_SIGNALS, # type: ignore
            {
                "series_id": f"BIO-STRESS-{region}",
                "timestamp": ts,
                "region": region,
                "metric": "stress_proxy",
                "value": float(stress),
                "meta": meta,
            },
        )

    # -----------------------------
    # Process a single biodata record
    # -----------------------------
    def _process_record(self, rec: Dict[str, Any]) -> None:
        """
        Update EMAs per metric, then compute and publish composites.
        """
        metric = str(rec.get("metric", "")).lower()
        region = str(rec.get("region", "US"))
        ts = str(rec.get("timestamp"))

        if metric not in ("step_count", "sleep_hours", "resting_heart_rate"):
            return

        val = float(rec.get("value", 0.0))

        if metric == "step_count":
            prev = self._load_ema(H_EMA_STEPS, region)
            ema = _ema(prev, val, EMA_ALPHA_STEPS)
            self._store_ema(H_EMA_STEPS, region, ema, ts)

        elif metric == "sleep_hours":
            prev = self._load_ema(H_EMA_SLEEP, region)
            ema = _ema(prev, val, EMA_ALPHA_SLEEP)
            self._store_ema(H_EMA_SLEEP, region, ema, ts)

        elif metric == "resting_heart_rate":
            prev = self._load_ema(H_EMA_RHR, region)
            ema = _ema(prev, val, EMA_ALPHA_RHR)
            self._store_ema(H_EMA_RHR, region, ema, ts)

        # Try to compute composites whenever any metric updates
        self._compute_composites(region, ts)

    # -----------------------------
    # Stream loop (consumer group)
    # -----------------------------
    def run_forever(self, block_ms: int = 1000, count: int = 100) -> None:
        """
        Consume from STREAM_ALT_SIGNALS as a group, looking for biodata metrics.
        Emits composite proxies back to STREAM_ALT_SIGNALS.
        """
        if not is_enabled("BIODATA"):
            log.info("FEATURE_BIODATA disabled; HealthSignalTransformer not running.")
            return

        log.info("HealthSignalTransformer starting (group=%s, consumer=%s)", self.group, self.consumer)

        while True:
            try:
                resp = streams.consume_stream(
                    streams=streams.STREAM_ALT_SIGNALS, # type: ignore
                    group=self.group, # type: ignore
                    consumer=self.consumer, # type: ignore
                    last_ids=">", # type: ignore
                    block_ms=block_ms,
                    count=count,
                    ack=True, # type: ignore
                )
                if not resp:
                    continue

                for _stream, entries in resp:
                    for entry_id, fields in entries:
                        try:
                            payload_raw = fields.get(b"data")
                            if not payload_raw:
                                continue
                            payload = streams._loads(payload_raw)  # same encoder as streams.py
                            # Only process biodata metrics
                            m = str(payload.get("metric", "")).lower()
                            if m in ("step_count", "sleep_hours", "resting_heart_rate"):
                                self._process_record(payload)
                        except Exception as e:
                            log.exception("Failed processing entry %s: %s", entry_id, e)
                        finally:
                            # Ack regardless to avoid blocking the group; failed ones can be re-derived next tick
                            try:
                                streams.xack(streams.STREAM_ALT_SIGNALS, self.group, entry_id) # type: ignore
                            except Exception:
                                pass
            except Exception as loop_err:
                log.exception("Consume loop error: %s", loop_err)
                time.sleep(1.0)


if __name__ == "__main__":
    # Simple runner
    if not is_enabled("BIODATA"):
        print("FEATURE_BIODATA disabled.")
    else:
        HealthSignalTransformer().run_forever()