# backend/data_ext/altdata/satellites.py
"""
Satellite alt-data ingestion.

- Pulls NDVI, nightlights, soil moisture, etc. from configured regions
- Maps to Hedge Fund X signal schema
- Returns list[dict] records for AltDataNormalizer to process

Config (altdata.yaml):
----------------------
sources:
  satellites:
    enabled: true
    provider: "nasa" | "esa" | "sentinelhub"
    api_key: "${SATELLITE_API_KEY}"
    regions:
      - name: "us_corn_belt"
        bbox: [-98.0, 37.0, -89.0, 43.0]
        signals: ["ndvi", "soil_moisture"]
      - name: "china_industrial"
        bbox: [110.0, 30.0, 120.0, 40.0]
        signals: ["nightlights"]

Contract:
---------
fetch(cfg: dict) -> List[dict]
Each record:
  {
    "metric": "ndvi",
    "value": float,
    "timestamp": ISO8601 str,
    "region": "us_corn_belt",
    "meta": { "provider": "nasa", "bbox": [...] }
  }
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List


def _fake_value(metric: str) -> float:
    if metric == "ndvi":
        return round(random.uniform(0.2, 0.9), 3)   # vegetation index 0-1
    if metric == "soil_moisture":
        return round(random.uniform(0.1, 0.5), 3)   # volumetric water content
    if metric == "nightlights":
        return round(random.uniform(10, 300), 1)    # radiance index
    return round(random.random(), 3)


def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ingest configured satellite regions/metrics.
    Returns list of dict records.
    """
    if not cfg.get("enabled", False):
        return []

    provider = cfg.get("provider", "demo")
    api_key = cfg.get("api_key")  # not used in stub
    regions = cfg.get("regions", [])
    ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    out: List[Dict[str, Any]] = []
    for region in regions:
        rname = region.get("name", "UNKNOWN")
        bbox = region.get("bbox")
        metrics = region.get("signals", [])
        for m in metrics:
            val = _fake_value(m)
            rec = {
                "metric": m,
                "value": val,
                "timestamp": ts,
                "region": rname,
                "meta": {
                    "provider": provider,
                    "bbox": bbox,
                },
            }
            out.append(rec)
    return out


if __name__ == "__main__":
    # Simple demo run
    demo_cfg = {
        "enabled": True,
        "provider": "demo",
        "regions": [
            {"name": "us_corn_belt", "bbox": [-98, 37, -89, 43], "signals": ["ndvi", "soil_moisture"]},
            {"name": "china_industrial", "bbox": [110, 30, 120, 40], "signals": ["nightlights"]},
        ],
    }
    records = fetch(demo_cfg)
    for r in records:
        print(r)