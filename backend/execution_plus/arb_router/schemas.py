# backend/common/schemas.py
"""
Project-wide lightweight schemas (dataclasses + helpers).

Why this exists
---------------
Keep a single source of truth for the shapes that flow through streams,
buses, and routers—without pulling heavy deps.

Covers:
- SignalRecord       -> what you publish to STREAM_ALT_SIGNALS / POLICY_SIGNALS
- SeriesUpdate       -> light pub/sub message for the UI (chan.signals.updates)
- Quote / Order / OrderResult (aligned with adapters.py)
- SocialRawPost      -> raw social messages (reddit/x/tiktok/discord)
- AltRawRecord       -> generic alt/climate/bio/raw records before normalization

Each model has:
- to_dict() for JSON serialization
- classmethod from_dict() for safe hydration (best-effort)

These are intentionally liberal in casting/optional fields so they won't
explode your workers if a field is missing. Validate stricter at the edges.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List


# ----------------------------- utils ----------------------------------

def _now_ts() -> float:
    return time.time()

def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _maybe_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def _str(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default


# ----------------------------- signals --------------------------------

@dataclass
class SignalRecord:
    """
    Unified signal payload published to Redis streams.

    Minimal required:
      series_id: unique key (e.g., "SOC-REDDIT-TSLA" or "ECMWF-PRECIP_ANOMALY-GULF")
      timestamp: ISO8601Z or epoch seconds as str
      value: float

    Optional:
      metric: semantic label ("social_sentiment", "precip_24h_mm", etc.)
      region: region tag ("GLOBAL" if N/A)
      symbol: asset tag (e.g., "TSLA"); optional
      meta: small dict (keep compact)
    """
    series_id: str
    timestamp: str
    value: float
    metric: str = ""
    region: str = "GLOBAL"
    symbol: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": _str(self.series_id),
            "timestamp": _str(self.timestamp),
            "value": _float(self.value),
            "metric": _str(self.metric),
            "region": _str(self.region or "GLOBAL"),
            "symbol": self.symbol,
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalRecord":
        return cls(
            series_id=_str(d.get("series_id")),
            timestamp=_str(d.get("timestamp")),
            value=_float(d.get("value")),
            metric=_str(d.get("metric")),
            region=_str(d.get("region") or "GLOBAL"),
            symbol=_str(d.get("symbol")) or None,
            meta=dict(d.get("meta") or {}),
        )


# ------------------------- pub/sub update ------------------------------

@dataclass
class SeriesUpdate:
    """
    Lightweight message for UI subscriptions on chan.signals.updates.
    """
    series_id: str
    timestamp: str
    value: float
    ema: Optional[float] = None
    metric: str = ""
    region: str = "GLOBAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": _str(self.series_id),
            "timestamp": _str(self.timestamp),
            "value": _float(self.value),
            "ema": _maybe_float(self.ema),
            "metric": _str(self.metric),
            "region": _str(self.region or "GLOBAL"),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeriesUpdate":
        return cls(
            series_id=_str(d.get("series_id")),
            timestamp=_str(d.get("timestamp")),
            value=_float(d.get("value")),
            ema=_maybe_float(d.get("ema")),
            metric=_str(d.get("metric")),
            region=_str(d.get("region") or "GLOBAL"),
        )


# --------------------------- market I/O --------------------------------
# Kept consistent with backend/execution_plus/adapters.py, but light-weight so
# other services can import without circular deps.

@dataclass
class Quote:
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    ts: float
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid": _maybe_float(self.bid),
            "ask": _maybe_float(self.ask),
            "mid": _maybe_float(self.mid),
            "ts": _float(self.ts, _now_ts()),
            "raw": dict(self.raw or {}),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Quote":
        return cls(
            bid=_maybe_float(d.get("bid")),
            ask=_maybe_float(d.get("ask")),
            mid=_maybe_float(d.get("mid")),
            ts=_float(d.get("ts"), _now_ts()),
            raw=dict(d.get("raw") or {}),
        )


@dataclass
class Order:
    symbol: str
    side: str           # "BUY" | "SELL"
    qty: float
    type: str = "MARKET"       # "MARKET" | "LIMIT"
    limit_price: Optional[float] = None
    client_id: Optional[str] = None
    venue_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": _str(self.symbol),
            "side": _str(self.side).upper(),
            "qty": _float(self.qty),
            "type": _str(self.type).upper(),
            "limit_price": _maybe_float(self.limit_price),
            "client_id": self.client_id,
            "venue_id": self.venue_id,
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Order":
        return cls(
            symbol=_str(d.get("symbol")),
            side=_str(d.get("side") or "BUY").upper(),
            qty=_float(d.get("qty")),
            type=_str(d.get("type") or "MARKET").upper(),
            limit_price=_maybe_float(d.get("limit_price")),
            client_id=_str(d.get("client_id")) or None,
            venue_id=_str(d.get("venue_id")) or None,
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str]
    symbol: str
    side: str
    filled_qty: float
    avg_price: Optional[float]
    fees: float
    status: str           # "filled" | "partial" | "rejected" | "cancelled" | "accepted" | "error"
    ts: float
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "order_id": self.order_id,
            "symbol": _str(self.symbol),
            "side": _str(self.side).upper(),
            "filled_qty": _float(self.filled_qty),
            "avg_price": _maybe_float(self.avg_price),
            "fees": _float(self.fees),
            "status": _str(self.status),
            "ts": _float(self.ts, _now_ts()),
            "raw": dict(self.raw or {}),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrderResult":
        return cls(
            ok=bool(d.get("ok")),
            order_id=_str(d.get("order_id")) or None,
            symbol=_str(d.get("symbol")),
            side=_str(d.get("side") or "BUY").upper(),
            filled_qty=_float(d.get("filled_qty")),
            avg_price=_maybe_float(d.get("avg_price")),
            fees=_float(d.get("fees")),
            status=_str(d.get("status")),
            ts=_float(d.get("ts"), _now_ts()),
            raw=dict(d.get("raw") or {}),
        )


# ----------------------- social/raw records ----------------------------

@dataclass
class SocialRawPost:
    """
    Raw social payload from sources (reddit/x/tiktok/discord) before scoring.
    """
    source: str                  # "reddit" | "x" | "tiktok" | "discord"
    text: str
    timestamp: str
    symbol: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": _str(self.source),
            "text": _str(self.text),
            "timestamp": _str(self.timestamp),
            "symbol": _str(self.symbol) or None,
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SocialRawPost":
        return cls(
            source=_str(d.get("source")),
            text=_str(d.get("text")),
            timestamp=_str(d.get("timestamp")),
            symbol=_str(d.get("symbol")) or None,
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class AltRawRecord:
    """
    Generic raw alt-/climate-/bio-data record BEFORE normalization.
    Mirrors what your altdata/climate fetchers emit.
    """
    metric: str
    value: float
    timestamp: str
    region: str = "GLOBAL"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": _str(self.metric),
            "value": _float(self.value),
            "timestamp": _str(self.timestamp),
            "region": _str(self.region or "GLOBAL"),
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AltRawRecord":
        return cls(
            metric=_str(d.get("metric")),
            value=_float(d.get("value")),
            timestamp=_str(d.get("timestamp")),
            region=_str(d.get("region") or "GLOBAL"),
            meta=dict(d.get("meta") or {}),
        )


# --------------------------- tiny JSON schema --------------------------
# Optional: export JSON "schemas" (very shallow) for quick sanity checks.

def json_schema_for(cls) -> Dict[str, Any]:
    """
    Return a minimal JSON schema-like dict (best-effort).
    """
    if cls is SignalRecord:
        return {
            "type": "object",
            "required": ["series_id", "timestamp", "value"],
            "properties": {
                "series_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "value": {"type": "number"},
                "metric": {"type": "string"},
                "region": {"type": "string"},
                "symbol": {"type": ["string", "null"]},
                "meta": {"type": "object"},
            },
        }
    if cls is SeriesUpdate:
        return {
            "type": "object",
            "required": ["series_id", "timestamp", "value"],
            "properties": {
                "series_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "value": {"type": "number"},
                "ema": {"type": ["number", "null"]},
                "metric": {"type": "string"},
                "region": {"type": "string"},
            },
        }
    if cls is SocialRawPost:
        return {
            "type": "object",
            "required": ["source", "text", "timestamp"],
            "properties": {
                "source": {"type": "string"},
                "text": {"type": "string"},
                "timestamp": {"type": "string"},
                "symbol": {"type": ["string", "null"]},
                "meta": {"type": "object"},
            },
        }
    if cls is AltRawRecord:
        return {
            "type": "object",
            "required": ["metric", "value", "timestamp"],
            "properties": {
                "metric": {"type": "string"},
                "value": {"type": "number"},
                "timestamp": {"type": "string"},
                "region": {"type": "string"},
                "meta": {"type": "object"},
            },
        }
    if cls is Quote:
        return {
            "type": "object",
            "required": ["ts"],
            "properties": {
                "bid": {"type": ["number", "null"]},
                "ask": {"type": ["number", "null"]},
                "mid": {"type": ["number", "null"]},
                "ts": {"type": "number"},
                "raw": {"type": "object"},
            },
        }
    if cls is Order:
        return {
            "type": "object",
            "required": ["symbol", "side", "qty"],
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["BUY", "SELL"]},
                "qty": {"type": "number"},
                "type": {"type": "string", "enum": ["MARKET", "LIMIT"]},
                "limit_price": {"type": ["number", "null"]},
                "client_id": {"type": ["string", "null"]},
                "venue_id": {"type": ["string", "null"]},
                "meta": {"type": "object"},
            },
        }
    if cls is OrderResult:
        return {
            "type": "object",
            "required": ["ok", "symbol", "side", "filled_qty", "fees", "status", "ts"],
            "properties": {
                "ok": {"type": "boolean"},
                "order_id": {"type": ["string", "null"]},
                "symbol": {"type": "string"},
                "side": {"type": "string"},
                "filled_qty": {"type": "number"},
                "avg_price": {"type": ["number", "null"]},
                "fees": {"type": "number"},
                "status": {"type": "string"},
                "ts": {"type": "number"},
                "raw": {"type": "object"},
            },
        }
    return {"type": "object"}

# --------------------------- tiny self-test ----------------------------

if __name__ == "__main__":
    s = SignalRecord(series_id="SOC-REDDIT-TSLA", timestamp="2025-08-22T01:23:45Z", value=0.87, metric="social_sentiment", region="GLOBAL")
    j = json.dumps(s.to_dict())
    print("SignalRecord JSON:", j)
    s2 = SignalRecord.from_dict(json.loads(j))
    print("Roundtrip ok:", s2 == s)