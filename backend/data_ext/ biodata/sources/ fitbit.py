# backend/data_ext/biodata/sources/fitbit.py
"""
Fitbit aggregate ingestion.

Goal:
- Pull anonymized/aggregate Fitbit metrics (resting HR, steps, sleep, etc.)
- Normalize to the unified signal schema
- Publish to STREAM_ALT_SIGNALS for downstream strategies

Notes:
- Real Fitbit access uses OAuth2 (client_id/secret) and per-user access tokens.
- For research/aggregate data, you’d typically ingest from a data vendor or internal exports.
- This stub provides a drop-in structure with fake data for wiring & tests.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from backend.bus import streams


class FitbitIngestor:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        region: str = "US",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.region = region

    # -----------------------------
    # Fetch
    # -----------------------------
    def fetch_data(self) -> List[Dict[str, Any]]:
        """
        Fetch Fitbit aggregates. Replace with real API/vendor calls.
        Returns list of raw records:
        {
          "metric": "resting_heart_rate" | "step_count" | "sleep_hours",
          "value": float,
          "timestamp": ISO-8601 UTC string,
          "region": "US" | "IN" | ...
        }
        """
        now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        # Demo values
        sample = [
            {"metric": "resting_heart_rate", "value": 60, "timestamp": now, "region": self.region},
            {"metric": "step_count", "value": 10234, "timestamp": now, "region": self.region},
            {"metric": "sleep_hours", "value": 6.9, "timestamp": now, "region": self.region},
        ]
        return sample

    # -----------------------------
    # Normalize
    # -----------------------------
    def normalize(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map raw Fitbit records into the unified signal schema:
        series_id, timestamp, region, metric, value
        """
        out: List[Dict[str, Any]] = []
        for rec in records:
            series = f"FITBIT-{rec['metric']}-{rec.get('region', self.region)}"
            out.append(
                {
                    "series_id": series,
                    "timestamp": rec["timestamp"],
                    "region": rec.get("region", self.region),
                    "metric": rec["metric"],
                    "value": rec["value"],
                }
            )
        return out

    # -----------------------------
    # Publish
    # -----------------------------
    def publish(self, signals: List[Dict[str, Any]]) -> None:
        """
        Write normalized signals to STREAM_ALT_SIGNALS.
        """
        for sig in signals:
            streams.publish_stream(streams.STREAM_ALT_SIGNALS, sig) # type: ignore

    # -----------------------------
    # Orchestration
    # -----------------------------
    def run_once(self) -> None:
        raw = self.fetch_data()
        signals = self.normalize(raw)
        self.publish(signals)


if __name__ == "__main__":
    ing = FitbitIngestor(region="US")
    ing.run_once()
    print("Fitbit signals published.")