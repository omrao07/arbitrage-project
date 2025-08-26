# backend/data_ext/altdata/fiber.py
"""
Terrestrial fiber / ISP network alt‑data (stub).

Purpose
-------
Produce network quality & reliability proxies by geography/ASN/IXP:
- latency_ms            : median RTT (ms)
- packet_loss_pct       : % packet loss
- throughput_mbps       : median downlink throughput (Mb/s)
- outages                : active outage incidents (count)
- peering_congestion     : fraction of saturated peering links (0..1)
- route_changes          : BGP update intensity (per hour)

You can wire real data later from approved probes (RIPE Atlas, ThousandEyes,
Kentik, public status pages, IXP sFlow/NetFlow partners, etc.).

Config (altdata.yaml)
---------------------
sources:
  fiber:
    enabled: true
    provider: "fiber_demo"
    regions:
      - name: "US"
        asns: ["AS15169","AS7922"]      # optional
        ixps: ["DE-CIX","LINX"]         # optional
      - name: "IN"
        asns: ["AS55836"]
    metrics: ["latency_ms","packet_loss_pct","throughput_mbps","outages","peering_congestion","route_changes"]

Returned record schema (raw; your normalizer will standardize):
{
  "metric": "<one of above>",
  "value": <float>,
  "timestamp": "<ISO8601Z>",
  "region": "<region>",
  "meta": { "provider": "fiber_demo", "asn": "AS15169", "ixp": "DE-CIX", "sample_size": <int>, "units": "..." }
}
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List, Sequence, Optional


def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------------------------
# Demo distributions (plausible)
# ---------------------------

def _priors(region: str) -> Dict[str, float]:
    r = region.lower()
    if r in ("us", "eu", "uk"):
        return {"lat": 20.0, "thr": 180.0, "loss": 0.15, "out": 1.0, "peer": 0.15, "bgp": 40.0}
    if r in ("in", "india"):
        return {"lat": 35.0, "thr": 120.0, "loss": 0.35, "out": 2.0, "peer": 0.25, "bgp": 55.0}
    if r in ("sa", "za", "south_africa"):
        return {"lat": 40.0, "thr": 100.0, "loss": 0.4, "out": 1.5, "peer": 0.3, "bgp": 50.0}
    if r in ("apac", "sg", "au", "nz"):
        return {"lat": 25.0, "thr": 150.0, "loss": 0.2, "out": 0.8, "peer": 0.18, "bgp": 45.0}
    return {"lat": 30.0, "thr": 130.0, "loss": 0.25, "out": 1.0, "peer": 0.2, "bgp": 45.0}


def _latency_ms(mu: float) -> float:
    return round(max(5.0, random.gauss(mu, mu * 0.2)), 1)

def _loss_pct(mu: float) -> float:
    base = max(0.0, random.gauss(mu, mu * 0.25))
    if random.random() < 0.08:  # occasional spike
        base += random.uniform(0.3, 1.2)
    return round(min(5.0, base), 2)

def _throughput_mbps(mu: float) -> float:
    return round(max(25.0, random.gauss(mu, mu * 0.25)), 1)

def _outages(mu: float) -> float:
    # Poisson-ish with regional mean
    lam = max(0.2, mu)
    # convert mean like 1.0 -> small integer counts with noise
    val = max(0, int(random.gauss(lam, lam * 0.4)))
    return float(val)

def _peering_sat(mu: float) -> float:
    # fraction 0..1, clamp
    v = max(0.0, min(1.0, random.gauss(mu, 0.08)))
    return round(v, 2)

def _bgp_updates(mu: float) -> float:
    # Route change intensity per hour
    return float(max(5, int(random.gauss(mu, mu * 0.3))))


# ---------------------------
# Public API
# ---------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate fiber/ISP network health records for configured regions (demo).
    Swap generators with real partner feeds when available.
    """
    if not cfg.get("enabled", False):
        return []

    regions: Sequence[Dict[str, Any]] = cfg.get("regions", []) or [{"name": "GLOBAL"}]
    metrics = set(cfg.get("metrics") or [
        "latency_ms", "packet_loss_pct", "throughput_mbps", "outages", "peering_congestion", "route_changes"
    ])
    provider = str(cfg.get("provider", "fiber_demo"))
    ts = _iso_now()

    out: List[Dict[str, Any]] = []

    for r in regions:
        name = str(r.get("name", "GLOBAL"))
        asns: Sequence[str] = r.get("asns", []) or [None]  # type: ignore[list-item]
        ixps: Sequence[str] = r.get("ixps", []) or [None]  # type: ignore[list-item]
        pri = _priors(name)

        for asn in asns:
            for ixp in ixps:
                meta_base = {
                    "provider": provider,
                    "asn": asn,
                    "ixp": ixp,
                }

                if "latency_ms" in metrics:
                    out.append({
                        "metric": "latency_ms",
                        "value": _latency_ms(pri["lat"]),
                        "timestamp": ts,
                        "region": name,
                        "meta": {**meta_base, "sample_size": random.randint(200, 2000), "units": "ms"},
                    })

                if "packet_loss_pct" in metrics:
                    out.append({
                        "metric": "packet_loss_pct",
                        "value": _loss_pct(pri["loss"]),
                        "timestamp": ts,
                        "region": name,
                        "meta": {**meta_base, "sample_size": random.randint(200, 2000), "units": "%"},
                    })

                if "throughput_mbps" in metrics:
                    out.append({
                        "metric": "throughput_mbps",
                        "value": _throughput_mbps(pri["thr"]),
                        "timestamp": ts,
                        "region": name,
                        "meta": {**meta_base, "sample_size": random.randint(150, 1500), "units": "Mb/s"},
                    })

                if "outages" in metrics:
                    out.append({
                        "metric": "outages",
                        "value": _outages(pri["out"]),
                        "timestamp": ts,
                        "region": name,
                        "meta": {**meta_base, "units": "count"},
                    })

                if "peering_congestion" in metrics:
                    out.append({
                        "metric": "peering_congestion",
                        "value": _peering_sat(pri["peer"]),
                        "timestamp": ts,
                        "region": name,
                        "meta": {**meta_base, "units": "fraction"},
                    })

                if "route_changes" in metrics:
                    out.append({
                        "metric": "route_changes",
                        "value": _bgp_updates(pri["bgp"]),
                        "timestamp": ts,
                        "region": name,
                        "meta": {**meta_base, "units": "updates_per_hour"},
                    })

    return out


# ---------------------------
# Demo CLI
# ---------------------------

if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "provider": "fiber_demo",
        "regions": [
            {"name": "US", "asns": ["AS15169", "AS7922"], "ixps": ["DE-CIX"]},
            {"name": "IN", "asns": ["AS55836"], "ixps": ["NIXI"]},
        ],
        "metrics": ["latency_ms","packet_loss_pct","throughput_mbps","outages","peering_congestion","route_changes"],
    }
    for rec in fetch(demo_cfg):
        print(rec)