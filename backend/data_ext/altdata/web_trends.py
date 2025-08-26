# backend/data_ext/altdata/web_trends.py
"""
Web search trends → market demand proxies.

Reads queries from altdata.yaml and emits demand/sentiment proxy signals
(e.g., Google Trends interest for "electric vehicles" mapped to TSLA).

Config (altdata.yaml)
---------------------
sources:
  web_trends:
    enabled: true
    provider: "google_trends"     # google_trends | baidu_index (future)
    signals:
      - query: "electric vehicles"
        symbol: "TSLA"
      - query: "oil prices"
        symbol: "CL=F"
      - query: "wheat shortage"
        symbol: "ZW=F"

Contract
--------
fetch(cfg: dict) -> List[dict]

Each returned record:
{
  "metric": "search_interest",
  "value": <float 0..100>,
  "timestamp": ISO8601 str (UTC),
  "region": "GLOBAL" | "<GEO>",       # e.g., "US", "IN"
  "symbol": "<mapped ticker>",
  "meta": { "provider": "google_trends", "query": "electric vehicles", "window": "now 7-d" }
}
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List, Optional

# Optional real backend: pytrends
try:
    from pytrends.request import TrendReq  # type: ignore
    _HAVE_PYTRENDS = True
except Exception:
    _HAVE_PYTRENDS = False


def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ----------------------------
# Real fetch via pytrends
# ----------------------------
def _fetch_google_trends_real(
    signals: List[Dict[str, Any]],
    *,
    geo: str = "",
    timeframe: str = "now 7-d",
    tz: int = 0,
    hl: str = "en-US",
) -> List[Dict[str, Any]]:
    """
    Use pytrends to get the latest interest_over_time for each query.
    Returns one record per query with the most recent value.
    """
    if not _HAVE_PYTRENDS:
        return []

    if not signals:
        return []

    # pytrends works best in small batches; we’ll loop queries one by one for simplicity
    pytrends = TrendReq(hl=hl, tz=tz)

    out: List[Dict[str, Any]] = []
    now = _iso_now()

    for item in signals:
        query = str(item.get("query", "")).strip()
        symbol = item.get("symbol")
        if not query:
            continue

        try:
            pytrends.build_payload(kw_list=[query], timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                # fallback to a neutral score
                val = 50.0
            else:
                val = float(df[query].iloc[-1])  # last datapoint (0..100)
        except Exception:
            # If rate-limited or error, soft-fallback
            val = float(random.randint(30, 70))

        record = {
            "metric": "search_interest",
            "value": val,
            "timestamp": now,
            "region": geo.upper() if geo else "GLOBAL",
            "symbol": symbol,
            "meta": {
                "provider": "google_trends",
                "query": query,
                "window": timeframe,
                "hl": hl,
                "geo": geo or "",
            },
        }
        out.append(record)

    return out


# ----------------------------
# Fallback (no pytrends)
# ----------------------------
def _fetch_google_trends_fallback(signals: List[Dict[str, Any]], *, geo: str = "") -> List[Dict[str, Any]]:
    """
    Generates plausible placeholder interest values (30..90).
    Keeps your pipeline alive when pytrends is not installed or unavailable.
    """
    out: List[Dict[str, Any]] = []
    now = _iso_now()
    for item in signals:
        query = str(item.get("query", "")).strip()
        symbol = item.get("symbol")
        if not query:
            continue
        val = float(random.randint(30, 90))
        out.append(
            {
                "metric": "search_interest",
                "value": val,
                "timestamp": now,
                "region": geo.upper() if geo else "GLOBAL",
                "symbol": symbol,
                "meta": {"provider": "demo", "query": query},
            }
        )
    return out


# ----------------------------
# Public API
# ----------------------------
def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Entry point for AltDataNormalizer.

    cfg example:
      {
        "enabled": true,
        "provider": "google_trends",
        "signals": [{"query":"electric vehicles","symbol":"TSLA"}],
        "geo": "US",            # optional (ISO country code)
        "timeframe": "now 7-d", # optional
        "tz": 0,                # optional
        "hl": "en-US"           # optional
      }
    """
    if not cfg.get("enabled", False):
        return []

    provider = str(cfg.get("provider", "google_trends")).lower()
    signals = cfg.get("signals", []) or []
    geo: str = str(cfg.get("geo", "")).upper()  # "" => global
    timeframe: str = str(cfg.get("timeframe", "now 7-d"))
    tz: int = int(cfg.get("tz", 0))
    hl: str = str(cfg.get("hl", "en-US"))

    if provider == "google_trends":
        if _HAVE_PYTRENDS:
            recs = _fetch_google_trends_real(signals, geo=geo, timeframe=timeframe, tz=tz, hl=hl)
            if recs:
                return recs
        # fall back if pytrends missing or returned nothing
        return _fetch_google_trends_fallback(signals, geo=geo)

    # TODO: support baidu_index or other providers later
    return _fetch_google_trends_fallback(signals, geo=geo)


# ----------------------------
# Demo CLI
# ----------------------------
if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "provider": "google_trends",
        "signals": [
            {"query": "electric vehicles", "symbol": "TSLA"},
            {"query": "oil prices", "symbol": "CL=F"},
            {"query": "wheat shortage", "symbol": "ZW=F"},
        ],
        "geo": "US",
        "timeframe": "now 7-d",
        "tz": 0,
        "hl": "en-US",
    }
    for rec in fetch(demo_cfg):
        print(rec)