# backend/data_ext/climate/sources/ecmwf.py
"""
ECMWF climate/weather model ingestion (stub with optional real client).

Purpose
-------
Produce hazard-relevant metrics (e.g., precip anomaly, temp anomaly, wind gusts,
cyclone probability) per configured region. Results feed Climate -> Strategy signals.

Config (example in altdata.yaml or climate config):
---------------------------------------------------
sources:
  climate:
    enabled: true
    provider: "ecmwf"
    api_key: "${ECMWF_API_KEY}"     # optional; real access uses ~/.cdsapirc
    variables:
      - "precip_anomaly"            # precipitation anomaly (% vs norm)
      - "temp_anomaly"              # temperature anomaly (°C vs norm)
      - "wind_gust"                 # 10m wind gust (m/s)
      - "cyclone_prob"              # 0..1
    lead_time_hours: 24
    regions:
      - name: "gulf_of_mexico"
        bbox: [-93.0, 21.0, -81.0, 30.0]  # lon_min, lat_min, lon_max, lat_max
      - name: "india_west_coast"
        bbox: [70.0, 8.0, 77.0, 21.0]

Contract
--------
fetch(cfg: dict) -> List[dict]
Record schema (raw; normalizer will standardize):
{
  "metric": "precip_anomaly" | "temp_anomaly" | "wind_gust" | "cyclone_prob",
  "value": <float>,
  "timestamp": ISO8601 str (forecast valid time),
  "region": "<region name>",
  "meta": {
    "provider": "ecmwf",
    "bbox": [...],
    "lead_time_hours": 24,
    "ensemble": "mean",
    "units": "%|degC|m/s|prob"
  }
}
"""

from __future__ import annotations

import datetime as dt
import math
import random
from typing import Any, Dict, List, Sequence

# Optional real client (if present and configured)
try:
    import cdsapi  # type: ignore
    _HAVE_CDS = True
except Exception:
    _HAVE_CDS = False


def _iso_at_lead(hours: int) -> str:
    valid = dt.datetime.utcnow().replace(microsecond=0) + dt.timedelta(hours=hours)
    return valid.isoformat() + "Z"


# ---------------------------
# Fake generators (demo mode)
# ---------------------------

def _fake_precip_anomaly(region: str) -> float:
    # ±80% anomaly, biased wetter in monsoon/coastal regions
    bias = 0.25 if any(k in region.lower() for k in ("india", "gulf", "coast")) else 0.0
    val = random.uniform(-0.6, 0.6) + bias
    return round(100.0 * max(-0.8, min(0.8, val)), 1)  # percent

def _fake_temp_anomaly(region: str) -> float:
    # ±5°C anomaly, slight warming bias over industrial belts
    bias = 0.6 if any(k in region.lower() for k in ("china", "industrial")) else 0.0
    val = random.uniform(-2.5, 2.5) + bias
    return round(max(-5.0, min(5.0, val)), 2)  # °C

def _fake_wind_gust(region: str) -> float:
    # Typical gusts 5–20 m/s; storms 20–40 m/s
    stormy = any(k in region.lower() for k in ("gulf", "typhoon", "hurricane", "cyclone"))
    base = random.uniform(6.0, 18.0)
    if stormy and random.random() < 0.3:
        base += random.uniform(8.0, 20.0)
    return round(base, 1)  # m/s

def _fake_cyclone_prob(region: str) -> float:
    # Probability 0..1; higher in cyclone basins
    basin = any(k in region.lower() for k in ("gulf", "bay_of_bengal", "arabian_sea", "west_pacific"))
    base = random.uniform(0.0, 0.15)
    if basin:
        base += random.uniform(0.05, 0.35)
    return round(max(0.0, min(1.0, base)), 2)


# ---------------------------
# Real fetch (placeholder)
# ---------------------------

def _fetch_ecmwf_real(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Placeholder for real ECMWF CDS retrieval.
    In practice:
      - Use cdsapi.Client() with a configured ~/.cdsapirc
      - Request ERA5 or SEAS5/IFS forecasts for variables and area=bbox
      - Aggregate to regional means or extremes
    For now, we return [] to let the caller fall back to demo generators.
    """
    if not _HAVE_CDS:
        return []
    # Implement real calls here if you have credentials and product ids.
    # Example pseudocode:
    # c = cdsapi.Client()
    # c.retrieve("<dataset>", {<request>}, "<download>.grib")
    # parse GRIB/NetCDF -> aggregate -> records
    return []


# ---------------------------
# Public API
# ---------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ingest ECMWF climate signals for configured regions/variables.
    """
    if not cfg.get("enabled", False):
        return []

    provider = str(cfg.get("provider", "ecmwf")).lower()
    if provider != "ecmwf":
        return []

    regions: Sequence[Dict[str, Any]] = cfg.get("regions", []) or []
    variables: Sequence[str] = [str(v).lower() for v in (cfg.get("variables") or [])]
    lead_h = int(cfg.get("lead_time_hours", 24))

    # If you wire real ECMWF access, try it first:
    real_records = _fetch_ecmwf_real(cfg)
    if real_records:
        return real_records

    # Demo fallback: synthesize plausible values
    out: List[Dict[str, Any]] = []
    ts = _iso_at_lead(lead_h)

    # Default variables if none supplied
    if not variables:
        variables = ["precip_anomaly", "temp_anomaly", "wind_gust", "cyclone_prob"]

    for region in regions:
        name = str(region.get("name", "UNKNOWN"))
        bbox = region.get("bbox")

        for var in variables:
            if var == "precip_anomaly":
                val = _fake_precip_anomaly(name)
                units = "%"
            elif var == "temp_anomaly":
                val = _fake_temp_anomaly(name)
                units = "degC"
            elif var == "wind_gust":
                val = _fake_wind_gust(name)
                units = "m/s"
            elif var == "cyclone_prob":
                val = _fake_cyclone_prob(name)
                units = "prob"
            else:
                # Unknown variable, generate a bounded dummy value
                val = round(random.uniform(-1.0, 1.0), 3)
                units = "arb"

            out.append(
                {
                    "metric": var,
                    "value": float(val),
                    "timestamp": ts,
                    "region": name,
                    "meta": {
                        "provider": "ecmwf",
                        "bbox": bbox,
                        "lead_time_hours": lead_h,
                        "ensemble": "mean",
                        "units": units,
                    },
                }
            )

    return out


# ---------------------------
# Demo CLI
# ---------------------------

if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "provider": "ecmwf",
        "variables": ["precip_anomaly", "temp_anomaly", "wind_gust", "cyclone_prob"],
        "lead_time_hours": 24,
        "regions": [
            {"name": "gulf_of_mexico", "bbox": [-93.0, 21.0, -81.0, 30.0]},
            {"name": "india_west_coast", "bbox": [70.0, 8.0, 77.0, 21.0]},
        ],
    }
    for rec in fetch(demo_cfg):
        print(rec)