# backend/data_ext/altdata/shipping_ais.py
"""
AIS (shipping) alt-data ingestion.

Purpose
-------
Convert maritime traffic (AIS) into commodity-relevant signals such as:
- oil_tankers: count of crude/product tankers in region
- lng_carriers: count of LNG carriers in region
- avg_speed: average knots (proxy for congestion / weather)
- port_dwell_hours: average dwell time in hours (proxy for port congestion)

Config (altdata.yaml)
---------------------
sources:
  shipping:
    enabled: true
    provider: "ais"                 # ais | marine_traffic | other vendor
    api_key: "${SHIPPING_API_KEY}"
    signals:
      - "oil_tankers"
      - "lng_carriers"
    regions:
      - name: "strait_of_hormuz"
        bbox: [55.0, 24.0, 57.0, 27.0]
      - name: "singapore_strait"
        bbox: [103.5, 1.0, 104.5, 1.5]

Contract
--------
fetch(cfg: dict) -> List[dict]
Each record (raw; normalizer will standardize):
{
  "metric": "oil_tankers" | "lng_carriers" | "avg_speed" | "port_dwell_hours",
  "value": <float>,
  "timestamp": ISO8601 str,
  "region": "<region name>",
  "meta": { "provider": "...", "bbox": [...], "sample_size": <int> }
}
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List, Sequence


# --- Fake generators (replace with real vendor calls later) --------------------

def _fake_tanker_count(region_name: str) -> int:
    # Hormuz / Singapore are busy; add bias
    base = 120 if "hormuz" in region_name.lower() else 150 if "singapore" in region_name.lower() else 60
    jitter = random.randint(-20, 25)
    return max(0, base + jitter)

def _fake_lng_count(region_name: str) -> int:
    base = 18 if "hormuz" in region_name.lower() else 22 if "singapore" in region_name.lower() else 8
    jitter = random.randint(-5, 6)
    return max(0, base + jitter)

def _fake_avg_speed_knots() -> float:
    # Typical ocean transits 10–20 knots; congestion/weather drops it
    return round(random.uniform(9.0, 18.5), 2)

def _fake_port_dwell_hours(region_name: str) -> float:
    # Congested hubs show higher dwell
    base = 24.0 if "singapore" in region_name.lower() else 18.0
    jitter = random.uniform(-5.0, 8.0)
    return round(max(2.0, base + jitter), 1)


# --- Public API ----------------------------------------------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ingest configured AIS regions/metrics and return raw records.

    Replace fake generators with real vendor API calls:
      - Query vessel positions filtered by ship_type in bbox (time window)
      - Aggregate counts, average speed, dwell time per region
    """
    if not cfg.get("enabled", False):
        return []

    provider = str(cfg.get("provider", "demo"))
    regions: Sequence[Dict[str, Any]] = cfg.get("regions", []) or []
    requested_signals = set(map(str.lower, cfg.get("signals", []) or []))
    ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Provide some defaults if user didn't list signals
    if not requested_signals:
        requested_signals = {"oil_tankers", "lng_carriers", "avg_speed", "port_dwell_hours"}

    out: List[Dict[str, Any]] = []

    for region in regions:
        name = str(region.get("name", "UNKNOWN"))
        bbox = region.get("bbox")

        # Simulate a sample size (vessels observed)
        sample_size = random.randint(300, 1200)

        if "oil_tankers" in requested_signals:
            out.append({
                "metric": "oil_tankers",
                "value": float(_fake_tanker_count(name)),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "bbox": bbox, "sample_size": sample_size}
            })

        if "lng_carriers" in requested_signals:
            out.append({
                "metric": "lng_carriers",
                "value": float(_fake_lng_count(name)),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "bbox": bbox, "sample_size": sample_size}
            })

        if "avg_speed" in requested_signals:
            out.append({
                "metric": "avg_speed",
                "value": _fake_avg_speed_knots(),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "bbox": bbox, "sample_size": sample_size, "units": "knots"}
            })

        if "port_dwell_hours" in requested_signals:
            out.append({
                "metric": "port_dwell_hours",
                "value": _fake_port_dwell_hours(name),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "bbox": bbox, "sample_size": sample_size, "units": "hours"}
            })

    return out


# --- Demo CLI ------------------------------------------------------------------

if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "provider": "demo",
        "signals": ["oil_tankers", "lng_carriers", "avg_speed", "port_dwell_hours"],
        "regions": [
            {"name": "strait_of_hormuz", "bbox": [55.0, 24.0, 57.0, 27.0]},
            {"name": "singapore_strait", "bbox": [103.5, 1.0, 104.5, 1.5]},
        ],
    }
    for rec in fetch(demo_cfg):
        print(rec)