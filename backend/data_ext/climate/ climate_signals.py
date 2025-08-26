# backend/data_ext/climate/climate_signals.py
"""
Climate -> Market risk proxies.

Consumes raw climate observations/forecasts already published into STREAM_ALT_SIGNALS
(e.g., from data_ext/climate/sources/noaa.py and ecmwf.py) and produces composite
risk signals per region:

- storm_risk   (0..1) ~ f(cyclone_prob, wind_gust, storm_alerts)
- flood_risk   (0..1) ~ f(precip_anomaly+, precip_24h_mm)
- drought_risk (0..1) ~ f(precip_anomaly-, spi_drought)
- heat_risk    (0..1) ~ f(temp_anomaly+)

Input metrics it looks for (case-insensitive):
    "cyclone_prob"        # 0..1
    "wind_gust"|"wind_gust_ms"   # m/s
    "storm_alerts"        # integer count
    "precip_anomaly"      # percent, +/- (e.g., +40%)
    "precip_24h_mm"       # mm
    "drought_spi"|"spi"   # standardized precipitation index (-3..+3)
    "temp_anomaly"        # degC, +/- (e.g., +1.5°C)

Output schema (published to STREAM_ALT_SIGNALS):
    {
      "series_id": "CLIMATE-STORM-RISK-<REGION>",
      "timestamp": "<ts>",
      "region": "<REGION>",
      "metric": "storm_risk",
      "value": <float in 0..1>,
      "meta": {"ema": true, "components": {...}, "weights": {...}}
    }

Feature flag required: FEATURE_CLIMATE=true
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

from backend.config.feature_flags import is_enabled
from backend.bus import streams

log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

# -----------------------------
# Tunables (ENV overrides)
# -----------------------------

# EMAs
EMA_ALPHA = float(os.getenv("CLIMATE_EMA_ALPHA", "0.25"))

# Normalization caps (to keep 0..1 mapping sane)
CAP_WIND_MS       = float(os.getenv("CLIMATE_CAP_WIND_MS", "35.0"))     # ~storm gusts
CAP_PRECIP_MM_24H = float(os.getenv("CLIMATE_CAP_PRECIP_MM_24H", "120"))
CAP_PRECIP_ANOM_P = float(os.getenv("CLIMATE_CAP_PRECIP_ANOM_P", "100"))   # +/- %
CAP_TEMP_ANOM_C   = float(os.getenv("CLIMATE_CAP_TEMP_ANOM_C", "6.0"))  # +/- degC
CAP_STORM_ALERTS  = float(os.getenv("CLIMATE_CAP_STORM_ALERTS", "5"))

# Weights for composites
W_STORM_CYCLONE = float(os.getenv("CLIMATE_W_STORM_CYCLONE", "0.5"))
W_STORM_WIND    = float(os.getenv("CLIMATE_W_STORM_WIND", "0.3"))
W_STORM_ALERTS  = float(os.getenv("CLIMATE_W_STORM_ALERTS", "0.2"))

W_FLOOD_ANOM    = float(os.getenv("CLIMATE_W_FLOOD_ANOM", "0.6"))
W_FLOOD_24H     = float(os.getenv("CLIMATE_W_FLOOD_24H", "0.4"))

W_DROUGHT_ANOM  = float(os.getenv("CLIMATE_W_DROUGHT_ANOM", "0.5"))
W_DROUGHT_SPI   = float(os.getenv("CLIMATE_W_DROUGHT_SPI", "0.5"))

W_HEAT_ANOM     = float(os.getenv("CLIMATE_W_HEAT_ANOM", "1.0"))

# Redis state keys (per-region EMA storage)
H_WIND   = "clim:ema:wind_ms"
H_ALERTS = "clim:ema:alerts"
H_CYCL   = "clim:ema:cycl_prob"
H_PANOM  = "clim:ema:precip_anom_pct"
H_P24    = "clim:ema:precip_24h_mm"
H_SPI    = "clim:ema:drought_spi"
H_TANOM  = "clim:ema:temp_anom_c"


# -----------------------------
# Helpers
# -----------------------------

def _ema(prev: Optional[float], x: float, alpha: float = EMA_ALPHA) -> float:
    if prev is None:
        return x
    return alpha * x + (1 - alpha) * prev

def _clip01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def _norm_cap(x: float, cap: float, minv: float = 0.0) -> float:
    """Linear map [minv..cap] -> [0..1], clipped."""
    if cap <= minv:
        return 0.0
    v = (x - minv) / (cap - minv)
    return _clip01(v)

def _norm_bilateral(x: float, cap: float) -> float:
    """For anomalies: map |x| <= cap -> |x|/cap in [0..1]. Positive direction only."""
    return _clip01(abs(x) / cap)

def _load(hname: str, region: str) -> Optional[float]:
    item = streams.hgetall(hname).get(region)
    if isinstance(item, dict):
        try:
            return float(item.get("v")) # type: ignore
        except Exception:
            return None
    try:
        return float(item) if item is not None else None
    except Exception:
        return None

def _store(hname: str, region: str, value: float, ts: str) -> None:
    streams.hset(hname, region, {"v": float(value), "ts": ts})


# -----------------------------
# Transformer
# -----------------------------

class ClimateSignalTransformer:
    """
    Consumes raw climate metrics from STREAM_ALT_SIGNALS and emits composite risks.
    """

    def __init__(self, group: str = "climate_signals", consumer: Optional[str] = None):
        self.group = group
        self.consumer = consumer or f"clim_{int(time.time())}"

    # Update EMAs for each metric
    def _update_metric(self, region: str, metric: str, value: float, ts: str) -> None:
        m = metric.lower()
        if m in ("wind_gust", "wind_gust_ms"):
            prev = _load(H_WIND, region)
            _store(H_WIND, region, _ema(prev, value), ts)
        elif m == "storm_alerts":
            prev = _load(H_ALERTS, region)
            _store(H_ALERTS, region, _ema(prev, value), ts)
        elif m == "cyclone_prob":
            prev = _load(H_CYCL, region)
            _store(H_CYCL, region, _ema(prev, value), ts)
        elif m == "precip_anomaly":
            prev = _load(H_PANOM, region)
            _store(H_PANOM, region, _ema(prev, value), ts)
        elif m == "precip_24h_mm":
            prev = _load(H_P24, region)
            _store(H_P24, region, _ema(prev, value), ts)
        elif m in ("drought_spi", "spi"):
            prev = _load(H_SPI, region)
            _store(H_SPI, region, _ema(prev, value), ts)
        elif m == "temp_anomaly":
            prev = _load(H_TANOM, region)
            _store(H_TANOM, region, _ema(prev, value), ts)

    # Compute and publish composites for a region
    def _compute_and_publish(self, region: str, ts: str) -> None:
        wind   = _load(H_WIND, region)
        alerts = _load(H_ALERTS, region)
        cycl   = _load(H_CYCL, region)
        panom  = _load(H_PANOM, region)
        p24    = _load(H_P24, region)
        spi    = _load(H_SPI, region)
        tanom  = _load(H_TANOM, region)

        # Need at least some metrics to compute each risk
        # --- Storm risk ---
        if cycl is not None or wind is not None or alerts is not None:
            storm = 0.0
            comps = {}
            if cycl is not None:
                c = _clip01(cycl)  # already 0..1
                storm += W_STORM_CYCLONE * c
                comps["cyclone_prob"] = c
            if wind is not None:
                w = _norm_cap(wind, CAP_WIND_MS)
                storm += W_STORM_WIND * w
                comps["wind_gust_norm"] = w
            if alerts is not None:
                a = _norm_cap(alerts, CAP_STORM_ALERTS)
                storm += W_STORM_ALERTS * a
                comps["storm_alerts_norm"] = a

            streams.publish_stream(
                streams.STREAM_ALT_SIGNALS, # type: ignore
                {
                    "series_id": f"CLIMATE-STORM-RISK-{region}",
                    "timestamp": ts,
                    "region": region,
                    "metric": "storm_risk",
                    "value": float(_clip01(storm)),
                    "meta": {"ema": True, "components": comps,
                             "weights": {"cyclone": W_STORM_CYCLONE, "wind": W_STORM_WIND, "alerts": W_STORM_ALERTS}},
                },
            )

        # --- Flood risk ---
        if (panom is not None and panom > 0) or (p24 is not None):
            comps = {}
            flood = 0.0
            if panom is not None and panom > 0:
                pa = _norm_bilateral(panom, CAP_PRECIP_ANOM_P)  # positive anomaly only
                flood += W_FLOOD_ANOM * pa
                comps["precip_anom_pos_norm"] = pa
            if p24 is not None:
                p = _norm_cap(p24, CAP_PRECIP_MM_24H)
                flood += W_FLOOD_24H * p
                comps["precip_24h_norm"] = p

            streams.publish_stream(
                streams.STREAM_ALT_SIGNALS, # type: ignore
                {
                    "series_id": f"CLIMATE-FLOOD-RISK-{region}",
                    "timestamp": ts,
                    "region": region,
                    "metric": "flood_risk",
                    "value": float(_clip01(flood)),
                    "meta": {"ema": True, "components": comps,
                             "weights": {"anom_pos": W_FLOOD_ANOM, "p24h": W_FLOOD_24H}},
                },
            )

        # --- Drought risk ---
        if (panom is not None and panom < 0) or (spi is not None):
            comps = {}
            drought = 0.0
            if panom is not None and panom < 0:
                pdry = _norm_bilateral(panom, CAP_PRECIP_ANOM_P)  # use |anom|; negative indicates dry
                drought += W_DROUGHT_ANOM * pdry
                comps["precip_anom_neg_norm"] = pdry
            if spi is not None:
                # SPI: -3 (extreme drought) .. +3 (extreme wet); map negative side to 0..1
                spi_dry = _clip01(max(0.0, -spi / 3.0))
                drought += W_DROUGHT_SPI * spi_dry
                comps["spi_dry_norm"] = spi_dry

            streams.publish_stream(
                streams.STREAM_ALT_SIGNALS, # type: ignore
                {
                    "series_id": f"CLIMATE-DROUGHT-RISK-{region}",
                    "timestamp": ts,
                    "region": region,
                    "metric": "drought_risk",
                    "value": float(_clip01(drought)),
                    "meta": {"ema": True, "components": comps,
                             "weights": {"anom_neg": W_DROUGHT_ANOM, "spi": W_DROUGHT_SPI}},
                },
            )

        # --- Heat risk ---
        if tanom is not None and tanom > 0:
            ha = _norm_bilateral(tanom, CAP_TEMP_ANOM_C)  # positive anomaly only
            streams.publish_stream(
                streams.STREAM_ALT_SIGNALS, # type: ignore
                {
                    "series_id": f"CLIMATE-HEAT-RISK-{region}",
                    "timestamp": ts,
                    "region": region,
                    "metric": "heat_risk",
                    "value": float(_clip01(W_HEAT_ANOM * ha)),
                    "meta": {"ema": True, "components": {"temp_anom_pos_norm": ha},
                             "weights": {"temp_anom": W_HEAT_ANOM}},
                },
            )

    # Main consume loop
    def run_forever(self, block_ms: int = 1000, count: int = 100) -> None:
        if not is_enabled("CLIMATE"):
            log.info("FEATURE_CLIMATE disabled; ClimateSignalTransformer not running.")
            return

        log.info("ClimateSignalTransformer starting (group=%s, consumer=%s)", self.group, self.consumer)

        while True:
            try:
                resp = streams.consume_stream(
                    streams=streams.STREAM_ALT_SIGNALS, # type: ignore
                    group=self.group, # type: ignore
                    consumer=self.consumer, # type: ignore
                    last_ids=">", # type: ignore
                    block_ms=block_ms,
                    count=count,
                    ack=True, # type: ignore
                )
                if not resp:
                    continue

                for _stream, entries in resp:
                    for entry_id, fields in entries:
                        try:
                            raw = fields.get(b"data")
                            if not raw:
                                continue
                            msg: Dict[str, Any] = streams._loads(raw)

                            # Only look at climate metrics (others flow through this stream too)
                            metric = str(msg.get("metric", "")).lower()
                            if metric not in (
                                "cyclone_prob",
                                "wind_gust", "wind_gust_ms",
                                "storm_alerts",
                                "precip_anomaly", "precip_24h_mm",
                                "drought_spi", "spi",
                                "temp_anomaly",
                            ):
                                # ignore non-climate messages
                                continue

                            region = str(msg.get("region", "GLOBAL"))
                            ts = str(msg.get("timestamp"))
                            val = float(msg.get("value", 0.0))

                            # Update state
                            self._update_metric(region, metric, val, ts)
                            # Try to compute and publish composites
                            self._compute_and_publish(region, ts)
                        except Exception as e:
                            log.exception("Failed processing climate entry: %s", e)
                        finally:
                            try:
                                streams.xack(streams.STREAM_ALT_SIGNALS, self.group, entry_id) # type: ignore
                            except Exception:
                                pass

            except Exception as loop_err:
                log.exception("Consume loop error: %s", loop_err)
                time.sleep(1.0)


if __name__ == "__main__":
    if not is_enabled("CLIMATE"):
        print("FEATURE_CLIMATE disabled.")
    else:
        ClimateSignalTransformer().run_forever()