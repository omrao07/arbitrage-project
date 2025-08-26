# backend/data_ext/climate/sources/noaa.py
"""
NOAA climate/weather ingestion (stub with optional real API hooks).

Purpose
-------
Emit climate-relevant metrics per region, such as:
- precip_24h_mm     : 24-hour accumulated precipitation (mm)
- temp_mean_c       : mean 2m air temperature (°C)
- wind_gust_ms      : 10m wind gust (m/s)
- drought_spi       : Standardized Precipitation Index (SPI, -3..+3)
- storm_alerts      : count of active tropical alerts (integer)

These are useful precursors for energy, agri, insurance, and transport signals.

Config (example)
----------------
sources:
  climate:
    enabled: true
    provider: "noaa"
    api_token: "${NOAA_API_TOKEN}"           # optional; for CDO API
    variables:
      - precip_24h_mm
      - temp_mean_c
      - wind_gust_ms
      - drought_spi
      - storm_alerts
    lookback_hours: 6                        # how far back to sample/latest
    regions:
      - name: "texas_gulf_coast"
        bbox: [-97.5, 26.0, -93.0, 30.5]    # lon_min, lat_min, lon_max, lat_max
      - name: "central_india"
        bbox: [75.0, 18.0, 82.0, 24.0]

Contract
--------
fetch(cfg: dict) -> List[dict]
Record schema (raw; your normalizer/transformer will standardize):
{
  "metric": "precip_24h_mm" | "temp_mean_c" | "wind_gust_ms" | "drought_spi" | "storm_alerts",
  "value": <float|int>,
  "timestamp": ISO8601 str (UTC),
  "region": "<region name>",
  "meta": {
    "provider": "noaa",
    "bbox": [...],
    "lookback_hours": <int>,
    "dataset": "CDO|NHC|GFS|CPC",            # indicative source
    "units": "mm|degC|m/s|index|count"
  }
}
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List, Sequence, Union

# If you later wire real calls, you'll likely use:
# import requests
# NOAA_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
# NOMADS/GFS, NHC RSS/KML feeds, CPC drought indices, etc.


def _iso_now_minus(hours: int) -> str:
    t = dt.datetime.utcnow().replace(microsecond=0) - dt.timedelta(hours=hours)
    return t.isoformat() + "Z"


# ---------------------------
# Demo generators (plausible)
# ---------------------------

def _fake_precip_mm(region: str) -> float:
    # Coastal/monsoon bias
    monsoonish = any(k in region.lower() for k in ("india", "coast", "gulf"))
    base = random.uniform(0.0, 15.0)
    if monsoonish:
        base += random.uniform(0.0, 30.0)
    return round(base, 1)

def _fake_temp_c(region: str) -> float:
    # Warmer for tropics, cooler elsewhere
    tropic = any(k in region.lower() for k in ("india", "gulf"))
    base = random.uniform(18.0, 35.0) if tropic else random.uniform(5.0, 28.0)
    return round(base, 1)

def _fake_gust_ms(region: str) -> float:
    # Occasional storms spike gusts
    stormy = any(k in region.lower() for k in ("gulf", "coast", "cyclone", "hurricane"))
    base = random.uniform(4.0, 14.0)
    if stormy and random.random() < 0.25:
        base += random.uniform(6.0, 20.0)
    return round(base, 1)

def _fake_spi(region: str) -> float:
    # SPI index in [-3, +3], centered around 0
    val = random.uniform(-2.0, 2.0)
    # Slight drought bias for "central" inland regions
    if "central" in region.lower():
        val -= random.uniform(0.1, 0.6)
    return round(max(-3.0, min(3.0, val)), 2)

def _fake_alerts(region: str) -> int:
    # Storm alerts rare but clustered in basins
    basin = any(k in region.lower() for k in ("gulf", "atlantic", "bay", "coast"))
    if basin and random.random() < 0.15:
        return random.randint(1, 4)
    return 0


# ---------------------------
# (Placeholder) real fetchers
# ---------------------------

def _fetch_noaa_cdo_real(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Placeholder for NOAA CDO (Climate Data Online) calls with api_token.
    Implement when you have an API token:
      - requests.get(..., headers={'token': token})
      - Aggregate station data over bbox/time window
    Returning [] delegates to demo generators.
    """
    token = (cfg.get("api_token") or "").strip()
    if not token:
        return []
    # TODO: implement real calls; return [] for now.
    return []


# ---------------------------
# Public API
# ---------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ingest NOAA signals for configured regions & variables.
    """
    if not cfg.get("enabled", False):
        return []

    provider = str(cfg.get("provider", "noaa")).lower()
    if provider != "noaa":
        return []

    regions: Sequence[Dict[str, Any]] = cfg.get("regions", []) or []
    variables: Sequence[str] = [str(v).lower() for v in (cfg.get("variables") or [])]
    lookback_h: int = int(cfg.get("lookback_hours", 6))

    # If you wire real NOAA CDO queries, try those first:
    real_records = _fetch_noaa_cdo_real(cfg)
    if real_records:
        return real_records

    # Demo fallback: synthesize plausible values at "now - lookback"
    ts = _iso_now_minus(lookback_h)
    if not variables:
        variables = ["precip_24h_mm", "temp_mean_c", "wind_gust_ms", "drought_spi", "storm_alerts"]

    out: List[Dict[str, Any]] = []

    for region in regions:
        name = str(region.get("name", "UNKNOWN"))
        bbox = region.get("bbox")

        for var in variables:
            if var == "precip_24h_mm":
                val, units, dataset = _fake_precip_mm(name), "mm", "CDO"
            elif var == "temp_mean_c":
                val, units, dataset = _fake_temp_c(name), "degC", "CDO"
            elif var == "wind_gust_ms":
                val, units, dataset = _fake_gust_ms(name), "m/s", "NOMADS"
            elif var == "drought_spi":
                val, units, dataset = _fake_spi(name), "index", "CPC"
            elif var == "storm_alerts":
                val, units, dataset = _fake_alerts(name), "count", "NHC"
            else:
                val, units, dataset = round(random.uniform(-1.0, 1.0), 3), "arb", "NOAA"

            out.append(
                {
                    "metric": var,
                    "value": float(val) if isinstance(val, (int, float)) else val,
                    "timestamp": ts,
                    "region": name,
                    "meta": {
                        "provider": "noaa",
                        "bbox": bbox,
                        "lookback_hours": lookback_h,
                        "dataset": dataset,
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
        "provider": "noaa",
        "variables": ["precip_24h_mm", "temp_mean_c", "wind_gust_ms", "drought_spi", "storm_alerts"],
        "lookback_hours": 6,
        "regions": [
            {"name": "texas_gulf_coast", "bbox": [-97.5, 26.0, -93.0, 30.5]},
            {"name": "central_india", "bbox": [75.0, 18.0, 82.0, 24.0]},
        ],
    }
    for rec in fetch(demo_cfg):
        print(rec)