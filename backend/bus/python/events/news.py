# bus/python/events/news.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------
# Base event
# ---------------------------
@dataclass
class NewsEvent:
    event_type: str                  # "headline" | "signal" | "source"
    ts_event: int                    # event/publish time (ms since epoch, UTC)
    ts_ingest: int                   # ingest time (ms since epoch, UTC)
    source: str                      # e.g., "reuters", "bloomberg", "rss"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NewsEvent":
        return cls(**d)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @classmethod
    def _base(cls, event_type: str, source: str, ts_event: Optional[int] = None) -> Dict[str, Any]:
        now = cls._now_ms()
        return {
            "event_type": event_type,
            "ts_event": ts_event if ts_event is not None else now,
            "ts_ingest": now,
            "source": source,
        }


# ---------------------------
# Source metadata (optional)
# ---------------------------
@dataclass
class Source(NewsEvent):
    publisher: str                   # e.g., "Reuters"
    feed: Optional[str] = None       # e.g., "TopNews", "CompanyNews"
    language: Optional[str] = None   # ISO 639-1, e.g., "en"
    country: Optional[str] = None    # ISO 3166-1 alpha-2, e.g., "US"
    provider_id: Optional[str] = None  # provider-specific ID

    @classmethod
    def create(
        cls,
        publisher: str,
        source: str = "news",
        ts_event: Optional[int] = None,
        feed: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> "Source":
        base = cls._base("source", source, ts_event)
        return cls(publisher=publisher, feed=feed, language=language, country=country, provider_id=provider_id, **base)


# ---------------------------
# Raw headline/article
# ---------------------------
@dataclass
class Headline(NewsEvent):
    headline_id: str                 # unique id (provider GUID or hash)
    title: str
    body: Optional[str] = None
    url: Optional[str] = None
    symbols: List[str] = field(default_factory=list)      # e.g., ["AAPL", "MSFT"]
    tickers: List[str] = field(default_factory=list)      # alias for symbols (if needed)
    entities: List[str] = field(default_factory=list)     # extracted entities
    topics: List[str] = field(default_factory=list)       # e.g., ["earnings","m&a"]
    categories: List[str] = field(default_factory=list)   # custom taxonomy tags
    importance: Optional[int] = None                      # 1..5 or provider score
    # Provenance
    publisher: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    country: Optional[str] = None

    @classmethod
    def create(
        cls,
        headline_id: str,
        title: str,
        source: str = "news",
        ts_event: Optional[int] = None,
        body: Optional[str] = None,
        url: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        importance: Optional[int] = None,
        publisher: Optional[str] = None,
        author: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> "Headline":
        base = cls._base("headline", source, ts_event)
        return cls(
            headline_id=headline_id,
            title=title,
            body=body,
            url=url,
            symbols=symbols or [],
            tickers=tickers or (symbols or []),
            entities=entities or [],
            topics=topics or [],
            categories=categories or [],
            importance=importance,
            publisher=publisher,
            author=author,
            language=language,
            country=country,
            **base,
        )


# ---------------------------
# Model output / signal
# ---------------------------
@dataclass
class Signal(NewsEvent):
    headline_id: str
    symbols: List[str] = field(default_factory=list)      # which instruments the score applies to
    # Core scores
    score: float = 0.0            # primary alpha score in [-1, 1]
    sentiment: float = 0.0        # sentiment polarity [-1, 1]
    relevance: float = 0.0        # relevance to symbol [0, 1]
    novelty: float = 0.0          # penalize repeats [0, 1]
    vol_impact: float = 0.0       # expected volatility impact [0, 1]
    # Optional classification
    topics: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    # Explainability / features
    features: Dict[str, float] = field(default_factory=dict)  # arbitrary feature name -> value
    model_name: str = "news-bert"
    model_version: str = "v1"
    # Ops
    confidence: Optional[float] = None   # optional confidence measure [0,1]
    ttl_sec: Optional[int] = None        # optional time-to-live hint for online store

    @classmethod
    def create(
        cls,
        headline_id: str,
        symbols: List[str],
        score: float,
        source: str = "nlp",
        ts_event: Optional[int] = None,
        sentiment: float = 0.0,
        relevance: float = 0.0,
        novelty: float = 0.0,
        vol_impact: float = 0.0,
        topics: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        features: Optional[Dict[str, float]] = None,
        model_name: str = "news-bert",
        model_version: str = "v1",
        confidence: Optional[float] = None,
        ttl_sec: Optional[int] = None,
    ) -> "Signal":
        base = cls._base("signal", source, ts_event)
        return cls(
            headline_id=headline_id,
            symbols=symbols or [],
            score=float(score),
            sentiment=float(sentiment),
            relevance=float(relevance),
            novelty=float(novelty),
            vol_impact=float(vol_impact),
            topics=topics or [],
            categories=categories or [],
            entities=entities or [],
            features=features or {},
            model_name=model_name,
            model_version=model_version,
            confidence=confidence,
            ttl_sec=ttl_sec,
            **base,
        )


# ---------------------------
# Example (manual run)
# ---------------------------
if __name__ == "__main__":
    # 1) Raw headline
    h = Headline.create(
        headline_id="reuters:12345",
        title="Apple announces record iPhone sales",
        url="https://example.com/aapl-record-sales",
        symbols=["AAPL"],
        entities=["Apple Inc"],
        topics=["earnings"],
        categories=["equities"],
        importance=5,
        publisher="Reuters",
        source="reuters",
    )
    print(h.to_json())

    # 2) Model signal from the headline
    s = Signal.create(
        headline_id=h.headline_id,
        symbols=h.symbols,
        score=0.72,
        sentiment=0.85,
        relevance=0.92,
        novelty=0.40,
        vol_impact=0.30,
        features={"sentiment": 0.85, "novelty": 0.40, "relevance": 0.92},
        model_name="news-bert-v3",
        model_version="3.1.0",
    )
    print(s.to_json())

    # 3) Optional source metadata
    src = Source.create(publisher="Reuters", source="reuters", language="en", country="US")
    print(src.to_json())