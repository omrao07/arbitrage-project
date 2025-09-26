# backend/data/altdata_weather.py
"""
Alt-Data Weather Adapter
------------------------
Purpose
  • Ingest daily/hourly weather observations and emit trading-ready signals:
    - HDD/CDD deviations vs seasonal normals
    - Precip anomalies (flood/drought flags)
    - Wind capacity factor proxy, solar irradiance proxy
    - Hydro inflow proxy for Nordic/Alpine regions
  • Map signals to regions and tickers (power, nat-gas, ags, airlines, retail).
  • Publish on the internal data bus with tamper-evident hashes.

Dependencies
  • pandas, numpy
  • Optional: requests (for pull mode)
  • Bus hook: backend.bus.streams.publish_stream (stubbed if missing)

Usage
-----
from backend.data.altdata_weather import (
    WeatherAdapter, WeatherConfig, WeatherStationObs, RegionMap
)

cfg = WeatherConfig(publish_stream="STREAM_WEATHER_SIGNALS")
wa = WeatherAdapter(cfg)

# Ingest observations you got from any weather feed:
obs = WeatherStationObs(
    station_id="US-NYC",
    ts_ms=1735603200000,
    temp_c=2.1,
    wind_mps=8.5,
    solar_wm2=120.0,
    precip_mm=6.0,
    normal_temp_c=3.5
)
env = wa.process_obs(obs, region="US_NE", tickers=["NG1", "PJM"])
wa.publish(env)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import requests  # optional for pull mode
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

# Data bus hook
try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        print(f"[stub publish_stream] {stream} <- {json.dumps(payload)[:200]}...")


# ----------------------------- Data Models -----------------------------

@dataclass
class WeatherConfig:
    publish_stream: str = "STREAM_WEATHER_SIGNALS"
    base_temp_c: float = 18.0              # CDD/HDD base temperature
    flood_precip_mm_day: float = 50.0      # daily precip threshold for flood flag
    drought_rolling_days: int = 21         # window for drought check
    drought_mm_sum_thresh: float = 5.0     # total precip over window below ⇒ drought flag
    wind_cf_scale_mps: float = 12.0        # ~ rated wind speed for CF proxy
    solar_clear_sky_wm2: float = 1000.0    # nominal clear-sky irradiance
    hydro_saturation_mm_day: float = 25.0  # runoff saturation proxy
    use_cache: bool = True                 # leave for future caching hooks

@dataclass
class WeatherStationObs:
    station_id: str
    ts_ms: int
    temp_c: Optional[float] = None
    wind_mps: Optional[float] = None
    solar_wm2: Optional[float] = None
    precip_mm: Optional[float] = None
    normal_temp_c: Optional[float] = None  # climatology mean for same DOY (if available)

@dataclass
class RegionMap:
    region: str
    tickers: List[str]          # instruments affected (e.g., ["NG1","PJM","NORDIC_PWR"])
    weights: Optional[List[float]] = None  # optional mapping weights (same len as tickers)


# ----------------------------- Core Adapter -----------------------------

class WeatherAdapter:
    def __init__(self, cfg: WeatherConfig) -> None:
        self.cfg = cfg
        # Rolling stores (for drought/aggregation)
        self._precip_roll: Dict[str, List[Tuple[int, float]]] = {}  # station_id -> [(ts_ms, precip_mm), ...]

    # ---------- Public API ----------

    def process_obs(
        self,
        obs: WeatherStationObs,
        *,
        region: Optional[str] = None,
        tickers: Optional[Sequence[str]] = None,
        mapping: Optional[RegionMap] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert a single observation into an envelope of signals.
        You can pass either (region,tickers) or a RegionMap.
        """
        station = obs.station_id
        ts = int(obs.ts_ms)

        # Degree days
        hdd, cdd, dd_surprise = self._degree_day_signals(obs)

        # Wind/Solar/Hydro/Precip signals
        wind_cf = self._wind_capacity_factor(obs.wind_mps)
        solar_cf = self._solar_capacity_factor(obs.solar_wm2)
        flood = self._flood_flag(obs.precip_mm)
        drought = self._drought_flag(station, ts, obs.precip_mm)

        # Region/ticker mapping
        reg, tix, wts = self._resolve_mapping(region, tickers, mapping)

        signals = {
            "hdd": hdd,                          # heating degree days (C)
            "cdd": cdd,                          # cooling degree days (C)
            "dd_surprise": dd_surprise,          # vs normal temp if provided
            "wind_capacity_factor": wind_cf,     # 0..1 proxy
            "solar_capacity_factor": solar_cf,   # 0..1 proxy
            "precip_mm": obs.precip_mm,
            "flood_flag": int(flood),
            "drought_flag": int(drought),
            "hydro_inflow_proxy": self._hydro_inflow_proxy(obs.precip_mm),  # 0..1 proxy
        }

        env = self._to_envelope(
            station_id=station,
            ts_ms=ts,
            region=reg,
            tickers=tix,
            weights=wts,
            signals=signals,
            meta=meta or {},
        )
        return env

    def process_batch(
        self,
        observations: Sequence[WeatherStationObs],
        *,
        region: Optional[str] = None,
        tickers: Optional[Sequence[str]] = None,
        mapping: Optional[RegionMap] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [self.process_obs(o, region=region, tickers=tickers, mapping=mapping, meta=meta) for o in observations]

    def publish(self, env: Dict[str, Any]) -> None:
        publish_stream(self.cfg.publish_stream, env)

    # ---------- Optional Pull Mode ----------

    def pull_and_process_json(
        self,
        *,
        url: str,
        serializer: callable, # type: ignore
        region: Optional[str] = None,
        tickers: Optional[Sequence[str]] = None,
        mapping: Optional[RegionMap] = None,
        meta: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Pull JSON from an endpoint (if `requests` is installed) and process.
        `serializer` must convert the remote JSON into a list[WeatherStationObs].
        """
        if not _HAS_REQUESTS:
            raise RuntimeError("requests not installed. Run `pip install requests` or use process_obs/process_batch with your own data.")
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        obs_list: List[WeatherStationObs] = serializer(data)
        return self.process_batch(obs_list, region=region, tickers=tickers, mapping=mapping, meta=meta)

    # ---------- Signal Computations ----------

    def _degree_day_signals(self, obs: WeatherStationObs) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        base = float(self.cfg.base_temp_c)
        t = obs.temp_c
        if t is None:
            return None, None, None
        hdd = max(0.0, base - t)
        cdd = max(0.0, t - base)
        dd_surp = None
        if obs.normal_temp_c is not None:
            # Surprise = (actual - normal): positive → warmer than normal (CDD-ish)
            dd_surp = float(t - float(obs.normal_temp_c))
        return hdd, cdd, dd_surp

    def _wind_capacity_factor(self, wind_mps: Optional[float]) -> Optional[float]:
        if wind_mps is None:
            return None
        # Simple logistic CF proxy: ~0 below cut-in (~3 m/s), ~1 near rated (~12 m/s), declines after 25 m/s (cut-out)
        v = float(wind_mps)
        rated = self.cfg.wind_cf_scale_mps
        cf = 1.0 / (1.0 + math.exp(-(v - 0.5 * rated) / (0.15 * rated)))
        # cut-out above ~25 m/s ⇒ reduce
        if v >= 25.0:
            cf *= 0.3
        return float(np.clip(cf, 0.0, 1.0))

    def _solar_capacity_factor(self, solar_wm2: Optional[float]) -> Optional[float]:
        if solar_wm2 is None:
            return None
        cf = float(solar_wm2) / max(1.0, self.cfg.solar_clear_sky_wm2)
        return float(np.clip(cf, 0.0, 1.2))  # allow slight >1 on anomalies

    def _flood_flag(self, precip_mm: Optional[float]) -> bool:
        try:
            return float(precip_mm or 0.0) >= self.cfg.flood_precip_mm_day
        except Exception:
            return False

    def _drought_flag(self, station_id: str, ts_ms: int, precip_mm: Optional[float]) -> bool:
        """Rolling drought detector: total precip over N days below threshold."""
        if precip_mm is None:
            precip = 0.0
        else:
            precip = float(precip_mm)
        w = self.cfg.drought_rolling_days
        buf = self._precip_roll.setdefault(station_id, [])
        buf.append((ts_ms, precip))
        # keep only last w days (assume ~daily cadence; tolerate irregular spacing)
        cutoff = ts_ms - w * 24 * 3600 * 1000
        self._precip_roll[station_id] = [(t, p) for (t, p) in buf if t >= cutoff]
        tot = sum(p for _, p in self._precip_roll[station_id])
        return tot <= self.cfg.drought_mm_sum_thresh

    def _hydro_inflow_proxy(self, precip_mm: Optional[float]) -> Optional[float]:
        if precip_mm is None:
            return None
        # Saturating function for runoff/inflow proxy, normalized 0..1
        x = float(precip_mm)
        k = self.cfg.hydro_saturation_mm_day
        return float(np.clip(x / (k + x), 0.0, 1.0))

    def _resolve_mapping(
        self,
        region: Optional[str],
        tickers: Optional[Sequence[str]],
        mapping: Optional[RegionMap],
    ) -> Tuple[Optional[str], List[str], Optional[List[float]]]:
        if mapping is not None:
            tix = list(mapping.tickers)
            wts = list(mapping.weights) if mapping.weights else None
            return mapping.region, tix, wts
        return (region, list(tickers) if tickers else [], None)

    # ---------- Envelope ----------

    def _to_envelope(
        self,
        *,
        station_id: str,
        ts_ms: int,
        region: Optional[str],
        tickers: List[str],
        weights: Optional[List[float]],
        signals: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        env = {
            "ts": ts_ms,
            "station_id": station_id,
            "region": region,
            "tickers": tickers,
            "weights": weights,
            "signals": signals,
            "meta": meta,
            "adapter": "altdata_weather",
            "version": 1,
        }
        env["hash"] = hashlib.sha256(json.dumps(env, sort_keys=True, default=str).encode()).hexdigest()
        return env


# ----------------------------- Helpers & Example -----------------------------

def example_serializer_open_meteo(json_obj: Dict[str, Any]) -> List[WeatherStationObs]:
    """
    Example serializer for Open-Meteo style JSON (hourly). Converts to daily obs using simple averages/sums.
    Expects structure:
      {"latitude":..,"longitude":..,"hourly":{"time":[...], "temperature_2m":[...], "windspeed_10m":[...],
                                              "shortwave_radiation":[...], "precipitation":[...]}}
    Returns one WeatherStationObs per day.
    """
    try:
        hourly = json_obj["hourly"]
    except Exception as e:
        raise ValueError("Invalid JSON shape for example serializer") from e

    times = pd.to_datetime(hourly["time"])
    df = pd.DataFrame({
        "temp_c": hourly.get("temperature_2m"),
        "wind_mps": hourly.get("windspeed_10m"),
        "solar_wm2": hourly.get("shortwave_radiation"),
        "precip_mm": hourly.get("precipitation"),
    }, index=times)
    daily = pd.DataFrame({
        "temp_c": df["temp_c"].mean(level=0, axis=0) if hasattr(df.index, "levels") else df["temp_c"].resample("D").mean(), # type: ignore
        "wind_mps": df["wind_mps"].resample("D").mean(),
        "solar_wm2": df["solar_wm2"].resample("D").mean(),
        "precip_mm": df["precip_mm"].resample("D").sum(),
    })
    out: List[WeatherStationObs] = []
    station = json_obj.get("timezone", "STATION")
    for ts, row in daily.iterrows():
        out.append(WeatherStationObs(
            station_id=str(station),
            ts_ms=int(ts.value // 10**6), # type: ignore
            temp_c=None if pd.isna(row["temp_c"]) else float(row["temp_c"]),
            wind_mps=None if pd.isna(row["wind_mps"]) else float(row["wind_mps"]),
            solar_wm2=None if pd.isna(row["solar_wm2"]) else float(row["solar_wm2"]),
            precip_mm=None if pd.isna(row["precip_mm"]) else float(row["precip_mm"]),
        ))
    return out


if __name__ == "__main__":
    # Minimal example run (no network):
    cfg = WeatherConfig()
    wa = WeatherAdapter(cfg)

    # Create two synthetic daily observations for a station
    now = int(time.time() * 1000)
    obs1 = WeatherStationObs(station_id="EU-OSL", ts_ms=now-24*3600*1000, temp_c=-2.0, wind_mps=7.0, solar_wm2=80.0, precip_mm=1.0, normal_temp_c=-1.0)
    obs2 = WeatherStationObs(station_id="EU-OSL", ts_ms=now, temp_c=-6.0, wind_mps=11.0, solar_wm2=60.0, precip_mm=0.0, normal_temp_c=-2.5)

    envs = wa.process_batch([obs1, obs2], region="NO", tickers=["NORDIC_PWR","EU_GAS"])
    for e in envs:
        print(json.dumps(e, indent=2))
        wa.publish(e)