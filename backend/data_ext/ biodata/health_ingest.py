# backend/data_ext/biodata/health_ingest.py
"""
Bio/health ingestion orchestrator.

- Pulls from one or more biodata sources (Apple, Fitbit; others plug-in later)
- Normalizes to unified signal schema:
    series_id, timestamp, region, metric, value
- Publishes to STREAM_ALT_SIGNALS for downstream strategies
- Respects FEATURE_BIODATA from feature_flags.py

Config:
- Environment variables for quick start:
    BIODATA_REGION=US
    APPLE_API_KEY=...
    FITBIT_CLIENT_ID=...
    FITBIT_CLIENT_SECRET=...
    FITBIT_ACCESS_TOKEN=...

- Optional: pass a config dict to HealthIngest.run_once(config=...) with:
    {
      "region": "US",
      "apple": {"enabled": true, "api_key": "..."},
      "fitbit": {"enabled": true, "client_id": "...", "client_secret": "...", "access_token": "..."}
    }

Streams:
- Publishes to backend.bus.streams.STREAM_ALT_SIGNALS
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from backend.config.feature_flags import is_enabled
from backend.bus import streams

# Source loaders
from backend.sources.apple import AppleHealthIngestor # type: ignore
from backend.sources.fitbit import FitbitIngestor # type: ignore

log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)


Signal = Dict[str, Any]


class HealthIngest:
    """
    Orchestrates bio/health sources and publishes normalized signals.
    """

    def __init__(self, region: Optional[str] = None):
        self.region = region or os.getenv("BIODATA_REGION", "US")
        # Simple in-memory dedupe cache: {(series_id, timestamp)}
        self._dedupe: Set[Tuple[str, str]] = set()

    # -----------------------------
    # Config handling
    # -----------------------------
    def _default_config(self) -> Dict[str, Any]:
        return {
            "region": self.region,
            "apple": {
                "enabled": True,
                "api_key": os.getenv("APPLE_API_KEY"),
            },
            "fitbit": {
                "enabled": True,
                "client_id": os.getenv("FITBIT_CLIENT_ID"),
                "client_secret": os.getenv("FITBIT_CLIENT_SECRET"),
                "access_token": os.getenv("FITBIT_ACCESS_TOKEN"),
            },
        }

    # -----------------------------
    # Fetch from sources
    # -----------------------------
    def _fetch_from_sources(self, cfg: Dict[str, Any]) -> List[Signal]:
        out: List[Signal] = []
        region = cfg.get("region", self.region)

        # Apple
        apple_cfg = cfg.get("apple", {})
        if apple_cfg.get("enabled"):
            try:
                ing = AppleHealthIngestor(api_key=apple_cfg.get("api_key"))
                raw = ing.fetch_data()
                apple_norm = ing.normalize(raw)
                out.extend(apple_norm)
                log.info("AppleHealthIngestor: %d signals", len(apple_norm))
            except Exception as e:
                log.exception("AppleHealthIngestor failed: %s", e)

        # Fitbit
        fitbit_cfg = cfg.get("fitbit", {})
        if fitbit_cfg.get("enabled"):
            try:
                ing = FitbitIngestor(
                    client_id=fitbit_cfg.get("client_id"),
                    client_secret=fitbit_cfg.get("client_secret"),
                    access_token=fitbit_cfg.get("access_token"),
                    region=region,
                )
                raw = ing.fetch_data()
                fitbit_norm = ing.normalize(raw)
                out.extend(fitbit_norm)
                log.info("FitbitIngestor: %d signals", len(fitbit_norm))
            except Exception as e:
                log.exception("FitbitIngestor failed: %s", e)

        # TODO: Add WHO / other vendors here as needed

        return out

    # -----------------------------
    # Dedupe
    # -----------------------------
    def _dedupe_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Drop duplicates by (series_id, timestamp). Keep first occurrence.
        """
        unique: List[Signal] = []
        for s in signals:
            key = (str(s.get("series_id")), str(s.get("timestamp")))
            if key in self._dedupe:
                continue
            self._dedupe.add(key)
            unique.append(s)
        return unique

    # -----------------------------
    # Publish
    # -----------------------------
    def _publish_signals(self, signals: Iterable[Signal]) -> int:
        count = 0
        for sig in signals:
            try:
                streams.publish_stream(streams.STREAM_ALT_SIGNALS, sig) # pyright: ignore[reportAttributeAccessIssue]
                count += 1
            except Exception as e:
                log.exception("Failed to publish signal %s: %s", sig, e)
        return count

    # -----------------------------
    # Orchestration
    # -----------------------------
    def run_once(self, config: Optional[Dict[str, Any]] = None) -> int:
        """
        Single pass: fetch -> dedupe -> publish.
        Returns number of signals published.
        """
        if not is_enabled("BIODATA"):
            log.info("FEATURE_BIODATA disabled; skipping health ingest.")
            return 0

        cfg = {**self._default_config(), **(config or {})}
        signals = self._fetch_from_sources(cfg)
        signals = self._dedupe_signals(signals)
        published = self._publish_signals(signals)
        log.info("HealthIngest: published %d signals", published)
        return published

    def run_forever(self, interval_sec: int = 900, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Loop with sleep; suitable for orchestration/supervisor.
        """
        log.info("HealthIngest loop starting (interval=%ss)", interval_sec)
        while True:
            try:
                self.run_once(config=config)
            except Exception as e:
                log.exception("HealthIngest tick failed: %s", e)
            time.sleep(interval_sec)


# CLI entry
if __name__ == "__main__":
    ing = HealthIngest()
    ing.run_once()