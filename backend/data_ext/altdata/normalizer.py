# backend/data_ext/altdata/normalizer.py
"""
Alt-data normalizer & publisher.

- Loads backend/config/altdata.yaml
- Calls data source fetchers (satellites, shipping_ais, web_trends)
- Normalizes to the unified signal schema:
    series_id, timestamp, region, metric, value, (optional) meta
- Publishes to STREAM_ALT_SIGNALS
- Respects FEATURE_ALTDATA flag

Expected source fetcher contracts (lightweight):
- satellites.fetch(config: dict) -> List[Dict[str, Any]]
- shipping_ais.fetch(config: dict) -> List[Dict[str, Any]]
- web_trends.fetch(config: dict) -> List[Dict[str, Any]]

Each fetch returns raw-ish records with at least:
  { "metric": str, "value": float, "timestamp": ISO8601, "region": str, ... }

You can extend/adjust inside each source module to map your providers cleanly.
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

# Source modules (stubs you already planned)
from .satellites import fetch as fetch_satellites
from .shipping_ais import fetch as fetch_shipping
from .web_trends import fetch as fetch_trends


log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)


ALT_YAML_PATH = os.getenv(
    "ALTDATA_YAML",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "altdata.yaml"),
)

Signal = Dict[str, Any]


class AltDataNormalizer:
    """
    Orchestrates alt-data sources and publishes normalized signals to STREAM_ALT_SIGNALS.
    """

    def __init__(self, yaml_path: Optional[str] = None):
        self.yaml_path = yaml_path or ALT_YAML_PATH
        # very simple dedupe on (series_id, timestamp)
        self._dedupe: set[Tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _load_yaml(self) -> Dict[str, Any]:
        if not yaml:
            raise RuntimeError("pyyaml is not installed; cannot load altdata.yaml")
        if not os.path.isfile(self.yaml_path):
            raise FileNotFoundError(f"altdata config not found: {self.yaml_path}")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg

    # ------------------------------------------------------------------
    # Fetchers
    # ------------------------------------------------------------------
    def _fetch_satellites(self, cfg: Dict[str, Any]) -> List[Signal]:
        src = cfg.get("sources", {}).get("satellites", {})
        if not src or not src.get("enabled", False):
            return []
        try:
            return fetch_satellites(src)
        except Exception as e:  # pragma: no cover
            log.exception("satellites.fetch failed: %s", e)
            return []

    def _fetch_shipping(self, cfg: Dict[str, Any]) -> List[Signal]:
        src = cfg.get("sources", {}).get("shipping", {})
        if not src or not src.get("enabled", False):
            return []
        try:
            return fetch_shipping(src)
        except Exception as e:  # pragma: no cover
            log.exception("shipping_ais.fetch failed: %s", e)
            return []

    def _fetch_trends(self, cfg: Dict[str, Any]) -> List[Signal]:
        src = cfg.get("sources", {}).get("web_trends", {})
        if not src or not src.get("enabled", False):
            return []
        try:
            return fetch_trends(src)
        except Exception as e:  # pragma: no cover
            log.exception("web_trends.fetch failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_str(x: Any, default: str = "") -> str:
        return str(x) if x is not None else default

    @staticmethod
    def _coerce_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _make_series_id(self, source: str, metric: str, region: str, symbol: Optional[str] = None) -> str:
        base = f"{source.upper()}-{metric.upper()}-{region.upper()}"
        if symbol:
            return f"{base}-{symbol.upper()}"
        return base

    def _normalize_records(self, records: Iterable[Dict[str, Any]], *, source: str) -> List[Signal]:
        """
        Convert provider-specific records to unified schema.
        Accepts extra keys like symbol, meta; pass-through via 'meta'.
        """
        out: List[Signal] = []
        for r in records:
            metric = self._coerce_str(r.get("metric"))
            value = self._coerce_float(r.get("value"))
            ts = self._coerce_str(r.get("timestamp"))
            region = self._coerce_str(r.get("region", "GLOBAL"))
            symbol = r.get("symbol")  # optional symbol mapping for equities/commodities

            if not metric or not ts:
                continue  # skip incomplete

            sig: Signal = {
                "series_id": self._make_series_id(source, metric, region, symbol),
                "timestamp": ts,
                "region": region,
                "metric": metric,
                "value": value,
            }

            # Optional metadata passthrough
            meta = r.get("meta")
            if meta is not None:
                sig["meta"] = meta

            # Optional explicit symbol passthrough
            if symbol is not None:
                sig["symbol"] = symbol

            out.append(sig)
        return out

    # ------------------------------------------------------------------
    # Dedupe & Publish
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_once(self, cfg_override: Optional[Dict[str, Any]] = None) -> int:
        """
        Single cycle: load cfg -> fetch -> normalize -> dedupe -> publish.
        Returns number of signals published.
        """
        if not is_enabled("ALTDATA"):
            log.info("FEATURE_ALTDATA disabled; skipping AltDataNormalizer.")
            return 0

        cfg = self._load_yaml()
        if cfg_override:
            # shallow merge at top-level keys
            cfg = {**cfg, **cfg_override}

        all_signals: List[Signal] = []

        # Satellites
        sat_raw = self._fetch_satellites(cfg)
        sat_norm = self._normalize_records(sat_raw, source="sat")
        all_signals.extend(sat_norm)
        if sat_norm:
            log.info("satellites: normalized %d signals", len(sat_norm))

        # Shipping AIS
        ship_raw = self._fetch_shipping(cfg)
        ship_norm = self._normalize_records(ship_raw, source="ais")
        all_signals.extend(ship_norm)
        if ship_norm:
            log.info("shipping_ais: normalized %d signals", len(ship_norm))

        # Web trends
        tr_raw = self._fetch_trends(cfg)
        tr_norm = self._normalize_records(tr_raw, source="trends")
        all_signals.extend(tr_norm)
        if tr_norm:
            log.info("web_trends: normalized %d signals", len(tr_norm))

        # Dedupe & publish
        uniq = self._dedupe(all_signals) # type: ignore
        published = self._publish(uniq)
        log.info("AltDataNormalizer: published %d signals", published)
        return published

    def run_forever(self, interval_sec: Optional[int] = None) -> None:
        """
        Loop forever using `normalization.update_interval_sec` in YAML (or override via env ALTDATA_INTERVAL_SEC).
        """
        if not is_enabled("ALTDATA"):
            log.info("FEATURE_ALTDATA disabled; AltDataNormalizer not running.")
            return

        # Determine interval
        try:
            cfg = self._load_yaml()
            default_interval = int(
                os.getenv(
                    "ALTDATA_INTERVAL_SEC",
                    str(cfg.get("normalization", {}).get("update_interval_sec", 3600)),
                )
            )
        except Exception:
            default_interval = int(os.getenv("ALTDATA_INTERVAL_SEC", "3600"))

        interval = interval_sec or default_interval
        log.info("AltDataNormalizer loop starting (interval=%ss)", interval)

        while True:
            try:
                self.run_once()
            except Exception as e:  # pragma: no cover
                log.exception("AltDataNormalizer tick failed: %s", e)
            time.sleep(interval)


# Simple CLI
if __name__ == "__main__":
    AltDataNormalizer().run_once()