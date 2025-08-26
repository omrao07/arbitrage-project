# backend/data_ext/altdata/starlink.py
"""
Starlink / LEO connectivity alt‑data ingestion (stub).

Purpose
-------
Produce network quality & adoption proxies by region:
- latency_ms          : median round‑trip latency (ms)
- packet_loss_pct     : % packet loss over short windows
- throughput_mbps     : median downlink throughput (Mb/s)
- active_terminals    : estimated # of active user terminals

These can act as leading indicators for consumer demand, cloud edge usage, rural
commerce enablement, disaster recovery capability, etc.

Config (altdata.yaml)
---------------------
sources:
  starlink:
    enabled: true
    provider: "starlink_demo"     # no official public API; swap when you have one
    regions:
      - name: "US"
      - name: "EU"
      - name: "IN"
    metrics: ["latency_ms","packet_loss_pct","throughput_mbps","active_terminals"]

Returned record schema (raw; your normalizer will standardize):
{
  "metric": "latency_ms" | "packet_loss_pct" | "throughput_mbps" | "active_terminals",
  "value": <float>,
  "timestamp": "<ISO8601Z>",
  "region": "<region name>",
  "meta": { "provider": "starlink_demo", "method": "synthetic", "sample_size": <int> }
}
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List, Sequence


def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------------------------
# Demo generators (plausible)
# ---------------------------

def _bias_for_region(region: str) -> Dict[str, float]:
    r = region.lower()
    # Very rough priors by geography / coverage maturity
    if r in ("us", "canada", "eu", "uk"):
        return {"lat_ms": 35.0, "thr_mbps": 120.0, "loss_pct": 0.3, "terms": 200_000}
    if r in ("au", "nz"):
        return {"lat_ms": 45.0, "thr_mbps": 110.0, "loss_pct": 0.4, "terms": 40_000}
    if r in ("in", "india"):
        return {"lat_ms": 45.0, "thr_mbps": 100.0, "loss_pct": 0.5, "terms": 60_000}
    if r in ("sa", "za", "south_africa"):
        return {"lat_ms": 55.0, "thr_mbps": 90.0, "loss_pct": 0.7, "terms": 25_000}
    return {"lat_ms": 50.0, "thr_mbps": 95.0, "loss_pct": 0.6, "terms": 30_000}


def _fake_latency_ms(bias: float) -> float:
    # Typical user‑reported latencies ~25–60 ms; jitter ±8 ms
    return round(max(15.0, random.gauss(bias, 6.0)), 1)


def _fake_loss_pct(bias: float) -> float:
    # 0–1.5% in good conditions; spikes in congestion/weather
    base = max(0.0, random.gauss(bias, 0.25))
    spike = random.random() < 0.1
    return round(min(5.0, base + (0.5 if spike else 0.0)), 2)


def _fake_throughput_mbps(bias: float) -> float:
    # 50–200 Mb/s typical; weather/load variance
    val = max(20.0, random.gauss(bias, 20.0))
    return round(val, 1)


def _fake_active_terminals(bias: float) -> float:
    # Slow drift with mild noise
    jitter = random.randint(-2000, 4000)
    return max(1_000.0, float(bias + jitter))


# ---------------------------
# Public API
# ---------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ingest Starlink‑style connectivity signals for configured regions.

    Note: There is no official public Starlink API. Replace generators with:
      - Approved data partners, or
      - Your own fleet of opt‑in probes reporting ping/iperf/packet loss.
    """
    if not cfg.get("enabled", False):
        return []

    regions: Sequence[Dict[str, Any]] = cfg.get("regions", []) or [{"name": "GLOBAL"}]
    metrics = set((cfg.get("metrics") or [
        "latency_ms", "packet_loss_pct", "throughput_mbps", "active_terminals"
    ]))
    provider = str(cfg.get("provider", "starlink_demo"))
    ts = _iso_now()

    out: List[Dict[str, Any]] = []

    for r in regions:
        name = str(r.get("name", "GLOBAL"))
        priors = _bias_for_region(name)

        if "latency_ms" in metrics:
            out.append({
                "metric": "latency_ms",
                "value": _fake_latency_ms(priors["lat_ms"]),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "method": "synthetic", "sample_size": random.randint(200, 2000), "units": "ms"},
            })

        if "packet_loss_pct" in metrics:
            out.append({
                "metric": "packet_loss_pct",
                "value": _fake_loss_pct(priors["loss_pct"]),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "method": "synthetic", "sample_size": random.randint(200, 2000), "units": "%"},
            })

        if "throughput_mbps" in metrics:
            out.append({
                "metric": "throughput_mbps",
                "value": _fake_throughput_mbps(priors["thr_mbps"]),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "method": "synthetic", "sample_size": random.randint(100, 1000), "units": "Mb/s"},
            })

        if "active_terminals" in metrics:
            out.append({
                "metric": "active_terminals",
                "value": _fake_active_terminals(priors["terms"]),
                "timestamp": ts,
                "region": name,
                "meta": {"provider": provider, "method": "synthetic", "units": "count"},
            })

    return out


# ---------------------------
# Demo CLI
# ---------------------------

if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "provider": "starlink_demo",
        "regions": [{"name": "US"}, {"name": "EU"}, {"name": "IN"}],
        "metrics": ["latency_ms", "packet_loss_pct", "throughput_mbps", "active_terminals"],
    }
    for rec in fetch(demo_cfg):
        print(rec)