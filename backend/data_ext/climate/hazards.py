# backend/data_ext/climate/hazards.py
"""
Climate Hazards Orchestrator

- Loads backend/config/climate.yaml (override with env CLIMATE_YAML)
- Calls provider fetchers (NOAA, ECMWF) to get raw climate metrics per region
- Normalizes to unified signal schema:
    series_id, timestamp, region, metric, value, (optional) meta
- Publishes to STREAM_ALT_SIGNALS for downstream strategies
- Respects FEATURE_CLIMATE flag

Expected provider fetchers (already stubbed):
- sources.noaa.fetch(cfg)  -> List[Dict[str, Any]]
- sources.ecmwf.fetch(cfg) -> List[Dict[str, Any]]

Example climate.yaml (minimal):
--------------------------------
sources:
  noaa:
    enabled: true
    provider: "noaa"
    variables: ["precip_24h_mm", "temp_mean_c", "wind_gust_ms", "drought_spi", "storm_alerts"]
    lookback_hours: 6
    regions:
      - name: "texas_gulf_coast"
        bbox: [-97.5, 26.0, -93.0, 30.5]
  ecmwf:
    enabled: true
    provider: "ecmwf"
    variables: ["precip_anomaly", "temp_anomaly", "wind_gust", "cyclone_prob"]
    lead_time_hours: 24
    regions:
      - name: "gulf_of_mexico"
        bbox: [-93.0, 21.0, -81.0, 30.0]

normalization:
  update_interval_sec: 1800
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.config.feature_flags import is_enabled
from backend.bus import streams

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Providers
from .sources.noaa import fetch as fetch_noaa
from .sources.ecmwf import fetch as fetch_ecmwf

log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

CLIMATE_YAML = os.getenv(
    "CLIMATE_YAML",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "climate.yaml"),
)

Signal = Dict[str, Any]


class ClimateHazards:
    """
    Orchestrates climate providers and publishes normalized hazard signals.
    """

    def __init__(self, yaml_path: Optional[str] = None):
        self.yaml_path = yaml_path or CLIMATE_YAML
        self._dedupe: set[Tuple[str, str]] = set()  # (series_id, timestamp)

    # ----------------------------- Config -----------------------------

    def _load_yaml(self) -> Dict[str, Any]:
        if not yaml:
            raise RuntimeError("pyyaml is not installed; cannot load climate.yaml")
        if not os.path.isfile(self.yaml_path):
            raise FileNotFoundError(f"climate config not found: {self.yaml_path}")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg

    # ----------------------------- Fetch ------------------------------

    def _fetch(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        srcs = (cfg.get("sources") or {})

        # NOAA
        noaa_cfg = srcs.get("noaa") or {}
        if noaa_cfg.get("enabled"):
            try:
                recs = fetch_noaa(noaa_cfg)
                if recs:
                    out.extend(recs)
                    log.info("NOAA: %d records", len(recs))
            except Exception as e:  # pragma: no cover
                log.exception("NOAA fetch failed: %s", e)

        # ECMWF
        ecmwf_cfg = srcs.get("ecmwf") or {}
        if ecmwf_cfg.get("enabled"):
            try:
                recs = fetch_ecmwf(ecmwf_cfg)
                if recs:
                    out.extend(recs)
                    log.info("ECMWF: %d records", len(recs))
            except Exception as e:  # pragma: no cover
                log.exception("ECMWF fetch failed: %s", e)

        # Add other providers here if needed
        return out

    # --------------------------- Normalize ----------------------------

    @staticmethod
    def _coerce_str(x: Any, default: str = "") -> str:
        return str(x) if x is not None else default

    @staticmethod
    def _coerce_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _series_id(self, source: str, metric: str, region: str) -> str:
        return f"{source.upper()}-{metric.upper()}-{region.upper()}"

    def _normalize(self, records: Iterable[Dict[str, Any]], *, source: str) -> List[Signal]:
        """
        Convert provider-specific records to unified schema.
        """
        out: List[Signal] = []
        for r in records:
            metric = self._coerce_str(r.get("metric"))
            value = self._coerce_float(r.get("value"))
            ts = self._coerce_str(r.get("timestamp"))
            region = self._coerce_str(r.get("region", "GLOBAL"))

            if not metric or not ts:
                continue

            sig: Signal = {
                "series_id": self._series_id(source, metric, region),
                "timestamp": ts,
                "region": region,
                "metric": metric,
                "value": value,
            }

            meta = r.get("meta")
            if meta is not None:
                sig["meta"] = meta

            out.append(sig)
        return out

    # --------------------------- Publish/Dedupe -----------------------

    def _dedupe(self, signals: Iterable[Signal]) -> List[Signal]: # type: ignore
        uniq: List[Signal] = []
        for s in signals:
            key = (str(s.get("series_id")), str(s.get("timestamp")))
            if key in self._dedupe:
                continue
            self._dedupe.add(key)
            uniq.append(s)
        return uniq

    def _publish(self, signals: Iterable[Signal]) -> int:
        n = 0
        for s in signals:
            try:
                streams.publish_stream(streams.STREAM_ALT_SIGNALS, s) # type: ignore
                n += 1
            except Exception as e:  # pragma: no cover
                log.exception("publish_stream failed for %s: %s", s, e)
        return n

    # ----------------------------- Orchestration ----------------------

    def run_once(self, cfg_override: Optional[Dict[str, Any]] = None) -> int:
        """
        Single cycle: load cfg -> fetch -> normalize -> dedupe -> publish.
        Returns number of signals published.
        """
        if not is_enabled("CLIMATE"):
            log.info("FEATURE_CLIMATE disabled; skipping ClimateHazards.")
            return 0

        cfg = self._load_yaml()
        if cfg_override:
            cfg = {**cfg, **cfg_override}

        raw = self._fetch(cfg)

        # Tag per-provider source names for series_id
        noaa_raw = [r for r in raw if (r.get("meta") or {}).get("provider", "").lower() == "noaa"]
        ecmwf_raw = [r for r in raw if (r.get("meta") or {}).get("provider", "").lower() == "ecmwf"]
        other_raw = [r for r in raw if r not in noaa_raw and r not in ecmwf_raw]

        signals: List[Signal] = []
        if noaa_raw:
            signals.extend(self._normalize(noaa_raw, source="noaa"))
        if ecmwf_raw:
            signals.extend(self._normalize(ecmwf_raw, source="ecmwf"))
        if other_raw:
            signals.extend(self._normalize(other_raw, source="climate"))

        uniq = self._dedupe(signals) # pyright: ignore[reportCallIssue]
        published = self._publish(uniq)
        log.info("ClimateHazards: published %d signals", published)
        return published

    def run_forever(self, interval_sec: Optional[int] = None) -> None:
        """
        Loop forever using normalization.update_interval_sec from YAML
        (override with env CLIMATE_INTERVAL_SEC).
        """
        if not is_enabled("CLIMATE"):
            log.info("FEATURE_CLIMATE disabled; ClimateHazards not running.")
            return

        # Determine interval
        try:
            cfg = self._load_yaml()
            default_interval = int(
                os.getenv(
                    "CLIMATE_INTERVAL_SEC",
                    str(cfg.get("normalization", {}).get("update_interval_sec", 1800)),
                )
            )
        except Exception:
            default_interval = int(os.getenv("CLIMATE_INTERVAL_SEC", "1800"))

        interval = interval_sec or default_interval
        log.info("ClimateHazards loop starting (interval=%ss)", interval)

        while True:
            try:
                self.run_once()
            except Exception as e:  # pragma: no cover
                log.exception("ClimateHazards tick failed: %s", e)
            time.sleep(interval)


# CLI
if __name__ == "__main__":
    ClimateHazards().run_once()