# backend/data_ext/biodata/sources/apple.py
"""
Apple Health / Apple Watch ingestion.

Goal:
- Pull aggregate bio/health data (e.g., activity, heart rate, sleep trends)
- Normalize into Hedge Fund X's signal schema
- Publish into STREAM_ALT_SIGNALS for downstream strategies

NOTE:
- Apple HealthKit API itself is not directly available server-side.
- In practice you'd need:
  1. Aggregated CSV/JSON exports (Apple Health app can export data)
  2. Third-party providers that resell anonymized Apple Watch trends
  3. Or a research dataset (for prototyping)
Here we provide a stub interface for plugging any of those sources in.
"""

import datetime as dt
from typing import Dict, List, Any

from backend.bus import streams


class AppleHealthIngestor:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def fetch_data(self) -> List[Dict[str, Any]]:
        """
        Fetch Apple Watch / Health data. In production, connect to a data vendor.
        Here we stub with fake values for demo.
        Returns list of records like:
        {
          "metric": "resting_heart_rate",
          "value": 62,
          "timestamp": "2025-08-21T23:00:00Z",
          "region": "US"
        }
        """
        now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        sample = [
            {
                "metric": "resting_heart_rate",
                "value": 62,
                "timestamp": now,
                "region": "US"
            },
            {
                "metric": "step_count",
                "value": 8900,
                "timestamp": now,
                "region": "US"
            },
            {
                "metric": "sleep_hours",
                "value": 7.2,
                "timestamp": now,
                "region": "US"
            },
        ]
        return sample

    def normalize(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize Apple health data into the signal schema.
        Schema fields: series_id, timestamp, region, metric, value
        """
        out: List[Dict[str, Any]] = []
        for rec in records:
            out.append(
                {
                    "series_id": f"APPLE-{rec['metric']}-{rec['region']}",
                    "timestamp": rec["timestamp"],
                    "region": rec["region"],
                    "metric": rec["metric"],
                    "value": rec["value"],
                }
            )
        return out

    def publish(self, signals: List[Dict[str, Any]]) -> None:
        """
        Publish normalized signals into STREAM_ALT_SIGNALS.
        """
        for sig in signals:
            streams.publish_stream(streams.STREAM_ALT_SIGNALS, sig) # type: ignore

    def run_once(self) -> None:
        """
        Fetch, normalize, publish (single run).
        """
        raw = self.fetch_data()
        signals = self.normalize(raw)
        self.publish(signals)


if __name__ == "__main__":
    ing = AppleHealthIngestor()
    ing.run_once()
    print("Apple health signals published.")