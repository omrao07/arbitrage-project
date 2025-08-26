# backend/utils/schemas.py
"""
Unified Data Schemas (validation + serialization)
-------------------------------------------------
- Prefer Pydantic v2 (or v1) if installed for strict runtime validation + schema export
- Fallback to stdlib dataclasses with light field coercion
- All payloads carry: schema (stable name) and v (semver-ish minor bump)

Usage
-----
from backend.utils.schemas import (
    Tick, Order, Fill, Position, StrategySignal, NewsEvent,
    OptionQuote, ChainSnapshot, VolSurfacePayload,
    CardSpendIndex, LightsIndex, ShippingIndex,
    EnsembleOutput, RiskMetric,
    ensure_tick, ensure_order, json_schema,
)

tick = ensure_tick({"symbol":"AAPL","price":212.34,"ts_ms":...})
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Mapping, Union

# ---------- Pydantic or dataclasses fallback ----------

try:
    # Try Pydantic v2, then v1
    import pydantic as _pd
    try:
        BaseModel = _pd.BaseModel  # v2
        _PD_V2 = True
    except Exception:  # pragma: no cover
        from pydantic import BaseModel as _BM  # type: ignore
        BaseModel = _BM
        _PD_V2 = False
    _HAS_PD = True
except Exception:  # pragma: no cover
    _HAS_PD = False
    from dataclasses import dataclass, field as dc_field

# ---------- helpers ----------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _sf(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)

def _si(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _ss(x: Any, default: str = "") -> str:
    return str(x) if x is not None else default

def _lb(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true","1","yes","y","on"): return True
    if s in ("false","0","no","n","off"): return False
    return default

# ---------- Base mixin ----------

if _HAS_PD:
    class _Model(BaseModel): # type: ignore
        model_config = _pd.ConfigDict(extra="ignore", validate_assignment=True, frozen=False)
        schema: str = "unknown"
        v: str = "1.0.0"

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump()

        @classmethod
        def json_schema(cls) -> Dict[str, Any]:
            return cls.model_json_schema()
else:
    @dataclass
    class _Model:
        schema: str = "unknown"
        v: str = "1.0.0"
        def to_dict(self) -> Dict[str, Any]:
            return self.__dict__
        @classmethod
        def json_schema(cls) -> Dict[str, Any]:  # minimal placeholder
            return {"title": cls.__name__, "type": "object"}

# ---------- Market data ----------

class Tick(_Model): # type: ignore
    """Normalized tick/quote/trade."""
    if _HAS_PD:
        symbol: str
        price: float
        ts_ms: int
        size: Optional[float] = None
        venue: Optional[str] = None
        typ: str = "trade"       # trade|quote|bar
        currency: Optional[str] = None
        schema: str = "md.tick"; v: str = "1.1.0"
    else:
        def __init__(self, **kw):
            self.schema="md.tick"; self.v="1.1.0"
            self.symbol=_ss(kw.get("symbol")).upper()
            self.price=_sf(kw.get("price"))
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.size=_sf(kw.get("size"), None) if kw.get("size") is not None else None # type: ignore
            self.venue=_ss(kw.get("venue"), None) # type: ignore
            self.typ=_ss(kw.get("typ") or "trade")
            self.currency=_ss(kw.get("currency"), None) # type: ignore

# ---------- Orders / Fills / Positions ----------

SIDE_L = ("buy","sell")
TYPE_L = ("market","limit","stop","stop_limit")

class Order(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "oms.order"; v: str = "1.2.0"
        ts_ms: int
        strategy: str
        symbol: str
        side: str
        qty: float
        typ: str = "market"
        limit_price: Optional[float] = None
        stop_price: Optional[float] = None
        venue: Optional[str] = None
        region: Optional[str] = None
        mark_price: Optional[float] = None
        extra: Optional[Dict[str, Any]] = None

        @(_pd.field_validator("side"))
        @classmethod
        def _side_valid(cls, v):
            v = str(v).lower()
            if v not in SIDE_L: raise ValueError("side must be buy|sell")
            return v
        @(_pd.field_validator("typ"))
        @classmethod
        def _type_valid(cls, v):
            v = str(v).lower()
            if v not in TYPE_L: raise ValueError(f"type must be one of {TYPE_L}")
            return v
    else:
        def __init__(self, **kw):
            self.schema="oms.order"; self.v="1.2.0"
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.strategy=_ss(kw.get("strategy"))
            self.symbol=_ss(kw.get("symbol")).upper()
            side=_ss(kw.get("side")).lower(); self.side= side if side in SIDE_L else "buy"
            typ=_ss(kw.get("typ") or "market").lower(); self.typ= typ if typ in TYPE_L else "market"
            self.qty=_sf(kw.get("qty"), 0.0)
            self.limit_price=_sf(kw.get("limit_price"), None) if kw.get("limit_price") is not None else None # type: ignore
            self.stop_price=_sf(kw.get("stop_price"), None) if kw.get("stop_price") is not None else None # type: ignore
            self.venue=_ss(kw.get("venue"), None) # type: ignore
            self.region=_ss(kw.get("region"), None) # type: ignore
            self.mark_price=_sf(kw.get("mark_price"), None) if kw.get("mark_price") is not None else None # type: ignore
            self.extra=kw.get("extra") if isinstance(kw.get("extra"), dict) else None

class Fill(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "oms.fill"; v: str = "1.0.0"
        ts_ms: int
        order_id: str
        symbol: str
        side: str
        qty: float
        price: float
        venue: Optional[str] = None
        fee: Optional[float] = None
    else:
        def __init__(self, **kw):
            self.schema="oms.fill"; self.v="1.0.0"
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.order_id=_ss(kw.get("order_id"))
            self.symbol=_ss(kw.get("symbol")).upper()
            self.side=_ss(kw.get("side")).lower()
            self.qty=_sf(kw.get("qty"))
            self.price=_sf(kw.get("price"))
            self.venue=_ss(kw.get("venue"), None) # type: ignore
            self.fee=_sf(kw.get("fee"), None) if kw.get("fee") is not None else None # type: ignore

class Position(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "oms.position"; v: str = "1.0.0"
        symbol: str
        qty: float
        avg_price: float
        pnl: Optional[float] = None
        ts_ms: int = _now_ms()
    else:
        def __init__(self, **kw):
            self.schema="oms.position"; self.v="1.0.0"
            self.symbol=_ss(kw.get("symbol")).upper()
            self.qty=_sf(kw.get("qty"))
            self.avg_price=_sf(kw.get("avg_price"))
            self.pnl=_sf(kw.get("pnl"), None) if kw.get("pnl") is not None else None # type: ignore
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())

# ---------- Strategy / Ensemble ----------

class StrategySignal(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "alpha.signal"; v: str = "1.0.0"
        ts_ms: int
        strategy: str
        score: float       # [-1, +1]
        meta: Optional[Dict[str, Any]] = None
    else:
        def __init__(self, **kw):
            self.schema="alpha.signal"; self.v="1.0.0"
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.strategy=_ss(kw.get("strategy"))
            self.score=max(-1.0, min(1.0, _sf(kw.get("score"))))
            self.meta=kw.get("meta") if isinstance(kw.get("meta"), dict) else None

class EnsembleOutput(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "alpha.ensemble"; v: str = "1.0.0"
        ts_ms: int
        strategy: str
        method: str
        ensemble: Dict[str, float]  # SYM -> score
        targets: Optional[Dict[str, float]] = None
        meta: Optional[Dict[str, Any]] = None
    else:
        def __init__(self, **kw):
            self.schema="alpha.ensemble"; self.v="1.0.0"
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.strategy=_ss(kw.get("strategy") or "ensemble")
            self.method=_ss(kw.get("method") or "mean")
            self.ensemble={str(k).upper(): float(v) for k,v in (kw.get("ensemble") or {}).items()}
            self.targets={str(k).upper(): float(v) for k,v in (kw.get("targets") or {}).items()} or None
            self.meta=kw.get("meta") if isinstance(kw.get("meta"), dict) else None

# ---------- News ----------

class NewsEvent(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "news.event"; v: str = "1.1.0"
        ts_ms: int
        source: str
        title: str
        url: str
        summary: Optional[str] = None
        tickers: List[str] = []
        symbols: List[str] = []
        category: Optional[str] = None
        lang: Optional[str] = "en"
    else:
        def __init__(self, **kw):
            self.schema="news.event"; self.v="1.1.0"
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.source=_ss(kw.get("source"))
            self.title=_ss(kw.get("title"))
            self.url=_ss(kw.get("url"))
            self.summary=_ss(kw.get("summary"), None) # type: ignore
            self.tickers=[str(x).upper() for x in (kw.get("tickers") or [])]
            self.symbols=[str(x).upper() for x in (kw.get("symbols") or [])]
            self.category=_ss(kw.get("category"), None) # type: ignore
            self.lang=_ss(kw.get("lang") or "en")

# ---------- Options (chain & surface) ----------

class OptionQuote(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "derivs.opt.quote"; v: str = "1.0.0"
        underlying: str
        expiry: str           # YYYY-MM-DD
        strike: float
        right: str            # C|P
        iv: Optional[float] = None
        bid: Optional[float] = None
        ask: Optional[float] = None
        mid: Optional[float] = None
        ts_ms: int = _now_ms()
    else:
        def __init__(self, **kw):
            self.schema="derivs.opt.quote"; self.v="1.0.0"
            self.underlying=_ss(kw.get("underlying")).upper()
            self.expiry=_ss(kw.get("expiry"))
            self.strike=_sf(kw.get("strike"))
            self.right=_ss(kw.get("right") or "C").upper()
            self.iv=_sf(kw.get("iv"), None) if kw.get("iv") is not None else None # type: ignore
            self.bid=_sf(kw.get("bid"), None) if kw.get("bid") is not None else None # type: ignore
            self.ask=_sf(kw.get("ask"), None) if kw.get("ask") is not None else None # type: ignore
            self.mid=_sf(kw.get("mid"), None) if kw.get("mid") is not None else None # type: ignore
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())

class ChainSnapshot(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "derivs.opt.chain"; v: str = "1.0.0"
        underlying: str
        provider: str
        spot: float
        rows: List[Dict[str, Any]]  # OptionQuote-like dicts
        r: float = 0.0
        q: float = 0.0
        ts_ms: int = _now_ms()
    else:
        def __init__(self, **kw):
            self.schema="derivs.opt.chain"; self.v="1.0.0"
            self.underlying=_ss(kw.get("underlying")).upper()
            self.provider=_ss(kw.get("provider") or "custom")
            self.spot=_sf(kw.get("spot"))
            self.rows=list(kw.get("rows") or [])
            self.r=_sf(kw.get("r"), 0.0)
            self.q=_sf(kw.get("q"), 0.0)
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())

class VolSurfacePayload(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "derivs.vol.surface"; v: str = "1.0.0"
        ts_ms: int
        underlying: str
        provider: str
        F: float
        r: float = 0.0
        q: float = 0.0
        expiries: List[str]
        knot_T: List[float]
        smiles: Dict[str, List[Tuple[float, float]]]  # exp -> [(m, iv)]
        meta: Optional[Dict[str, Any]] = None
    else:
        def __init__(self, **kw):
            self.schema="derivs.vol.surface"; self.v="1.0.0"
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.underlying=_ss(kw.get("underlying")).upper()
            self.provider=_ss(kw.get("provider") or "custom")
            self.F=_sf(kw.get("F"))
            self.r=_sf(kw.get("r"), 0.0); self.q=_sf(kw.get("q"), 0.0)
            self.expiries=list(kw.get("expiries") or [])
            self.knot_T=[_sf(x) for x in (kw.get("knot_T") or [])]
            self.smiles={str(k): [(float(m), float(iv)) for (m,iv) in v] for k,v in (kw.get("smiles") or {}).items()}
            self.meta=kw.get("meta") if isinstance(kw.get("meta"), dict) else None

# ---------- Alt-data indices ----------

class CardSpendIndex(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "alt.card_spend"; v: str = "1.0.0"
        level: str           # merchant|category|country|ticker
        key: str
        day: str             # YYYY-MM-DD
        gross: float
        users: Optional[int] = None
        avg_ticket: Optional[float] = None
        dod: Optional[float] = None
        wow: Optional[float] = None
        mom: Optional[float] = None
        yoy: Optional[float] = None
        trend: Optional[float] = None
        anomaly: Optional[float] = None
        meta: Optional[Dict[str, Any]] = None
    else:
        def __init__(self, **kw):
            self.schema="alt.card_spend"; self.v="1.0.0"
            self.level=_ss(kw.get("level")); self.key=_ss(kw.get("key"))
            self.day=_ss(kw.get("day"))
            self.gross=_sf(kw.get("gross"))
            self.users=_si(kw.get("users"), None) if kw.get("users") is not None else None # type: ignore
            self.avg_ticket=_sf(kw.get("avg_ticket"), None) if kw.get("avg_ticket") is not None else None # type: ignore
            for k in ("dod","wow","mom","yoy","trend","anomaly"):
                setattr(self, k, _sf(kw.get(k), None) if kw.get(k) is not None else None) # type: ignore
            self.meta=kw.get("meta") if isinstance(kw.get("meta"), dict) else None

class LightsIndex(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "alt.lights"; v: str = "1.0.0"
        level: str        # region|facility
        key: str
        day: str
        mean_dn: float
        sum_dn: Optional[float] = None
        pct_area_lit: Optional[float] = None
        dod: Optional[float] = None
        wow: Optional[float] = None
        mom: Optional[float] = None
        yoy: Optional[float] = None
        trend: Optional[float] = None
        anomaly: Optional[float] = None
        meta: Optional[Dict[str, Any]] = None
    else:
        def __init__(self, **kw):
            self.schema="alt.lights"; self.v="1.0.0"
            self.level=_ss(kw.get("level")); self.key=_ss(kw.get("key")); self.day=_ss(kw.get("day"))
            self.mean_dn=_sf(kw.get("mean_dn")); self.sum_dn=_sf(kw.get("sum_dn"), None) if kw.get("sum_dn") is not None else None # type: ignore
            self.pct_area_lit=_sf(kw.get("pct_area_lit"), None) if kw.get("pct_area_lit") is not None else None # type: ignore
            for k in ("dod","wow","mom","yoy","trend","anomaly"):
                setattr(self, k, _sf(kw.get(k), None) if kw.get(k) is not None else None) # type: ignore
            self.meta=kw.get("meta") if isinstance(kw.get("meta"), dict) else None

class ShippingIndex(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "alt.shipping"; v: str = "1.0.0"
        level: str            # port|choke
        key: str
        day: str
        arrivals: Optional[int] = None
        departures: Optional[int] = None
        unique_vessels: Optional[int] = None
        avg_speed_kn: Optional[float] = None
        dwell_hours: Optional[float] = None
        queue_len: Optional[float] = None
        congestion: Optional[float] = None
        meta: Optional[Dict[str, Any]] = None
    else:
        def __init__(self, **kw):
            self.schema="alt.shipping"; self.v="1.0.0"
            self.level=_ss(kw.get("level")); self.key=_ss(kw.get("key")); self.day=_ss(kw.get("day"))
            for k in ("arrivals","departures","unique_vessels"):
                setattr(self, k, _si(kw.get(k), None) if kw.get(k) is not None else None) # type: ignore
            for k in ("avg_speed_kn","dwell_hours","queue_len","congestion"):
                setattr(self, k, _sf(kw.get(k), None) if kw.get(k) is not None else None) # type: ignore
            self.meta=kw.get("meta") if isinstance(kw.get("meta"), dict) else None

# ---------- Risk / PnL / Metrics ----------

class RiskMetric(_Model): # type: ignore
    if _HAS_PD:
        schema: str = "risk.metric"; v: str = "1.0.0"
        ts_ms: int
        kind: str                   # 'dd'|'vol'|'var'|'es'|'pnl'|'exposure'
        strategy: Optional[str] = None
        value: float
        window: Optional[str] = None
        meta: Optional[Dict[str, Any]] = None
    else:
        def __init__(self, **kw):
            self.schema="risk.metric"; self.v="1.0.0"
            self.ts_ms=_si(kw.get("ts_ms") or _now_ms())
            self.kind=_ss(kw.get("kind"))
            self.strategy=_ss(kw.get("strategy"), None) # type: ignore
            self.value=_sf(kw.get("value"))
            self.window=_ss(kw.get("window"), None) # type: ignore
            self.meta=kw.get("meta") if isinstance(kw.get("meta"), dict) else None

# ---------- ensure_* coercers (safe wrappers) ----------

def ensure_tick(d: Mapping[str, Any]) -> Tick:            return Tick(**dict(d))
def ensure_order(d: Mapping[str, Any]) -> Order:          return Order(**dict(d))
def ensure_fill(d: Mapping[str, Any]) -> Fill:            return Fill(**dict(d))
def ensure_position(d: Mapping[str, Any]) -> Position:    return Position(**dict(d))
def ensure_signal(d: Mapping[str, Any]) -> StrategySignal:return StrategySignal(**dict(d))
def ensure_news(d: Mapping[str, Any]) -> NewsEvent:       return NewsEvent(**dict(d))
def ensure_option(d: Mapping[str, Any]) -> OptionQuote:   return OptionQuote(**dict(d))
def ensure_chain(d: Mapping[str, Any]) -> ChainSnapshot:  return ChainSnapshot(**dict(d))
def ensure_surface(d: Mapping[str, Any]) -> VolSurfacePayload: return VolSurfacePayload(**dict(d))
def ensure_cardidx(d: Mapping[str, Any]) -> CardSpendIndex: return CardSpendIndex(**dict(d))
def ensure_lights(d: Mapping[str, Any]) -> LightsIndex:   return LightsIndex(**dict(d))
def ensure_ship(d: Mapping[str, Any]) -> ShippingIndex:   return ShippingIndex(**dict(d))
def ensure_risk(d: Mapping[str, Any]) -> RiskMetric:      return RiskMetric(**dict(d))
def ensure_ensemble(d: Mapping[str, Any]) -> EnsembleOutput: return EnsembleOutput(**dict(d))

# ---------- JSON Schema exporter ----------

def json_schema() -> Dict[str, Any]:
    """
    Return a dict of {name: JSON Schema}. If Pydantic unavailable, emits placeholders.
    """
    models = [
        Tick, Order, Fill, Position, StrategySignal, EnsembleOutput,
        NewsEvent, OptionQuote, ChainSnapshot, VolSurfacePayload,
        CardSpendIndex, LightsIndex, ShippingIndex, RiskMetric,
    ]
    out: Dict[str, Any] = {}
    for m in models:
        try:
            out[m.__name__] = m.json_schema()
        except Exception:
            out[m.__name__] = {"title": m.__name__, "type": "object"}
    return out