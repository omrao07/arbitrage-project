# news_bridge.py
# -*- coding: utf-8 -*-
"""
NewsBridge: ingest → normalize → enrich → route

- Sources:
    * RSS/Atom via feedparser (optional)
    * JSON HTTP APIs (e.g., NewsAPI-style), or your internal endpoints
- Enrichment:
    * Dedup by (source,url,title hash)
    * Ticker detection (cashtags + fuzzy name map)
    * Lightweight sentiment (VADER if available → fallback lexical)
    * Topic tags via simple heuristics (macro, credit, equity, fx, rates, vol)
- Routing:
    * In-process pub/sub callback (plug your own bus later)

Minimal deps. All 3rd-party libraries are optional.
"""

from __future__ import annotations
import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# -------- Optional imports (graceful fallback) --------
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import feedparser  # type: ignore
except Exception:  # pragma: no cover
    feedparser = None  # type: ignore

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _vader = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover
    _vader = None


# ========= Models =========

@dataclass
class RawItem:
    source: str
    url: str
    title: str
    summary: str = ""
    published: Optional[datetime] = None
    tickers_hint: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsItem:
    id: str
    source: str
    url: str
    domain: str
    title: str
    summary: str
    published: datetime
    tickers: List[str]
    sentiment: float  # [-1..+1]
    topics: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsEvent:
    """What strategies subscribe to."""
    kind: str = "news.signal"
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)


# ========= Utils =========

_CASHTAG = re.compile(r"(?<![A-Z0-9])\$([A-Z]{1,6})(?![A-Z])")
_WS = re.compile(r"\s+")

def _hash_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""

def _to_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val.astimezone(timezone.utc)
    if isinstance(val, str):
        try:
            # ISO / RFC2822 best-effort
            return datetime.fromisoformat(val.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass
    return _now_utc()

def _clean_text(x: str) -> str:
    return _WS.sub(" ", (x or "").strip())


# ========= Sentiment =========

_NEG = set("downgrade lawsuit fraud probe default slump plunge halt deny cut miss warn crisis shock ban outage hack".split())
_POS = set("upgrade beat boost rally soar expand raise approve win record surge profit growth".split())

def score_sentiment(text: str) -> float:
    text = text.lower()
    if _vader:
        s = _vader.polarity_scores(text)
        return float(s.get("compound", 0.0))
    # fallback: very light lexical scoring
    pos = sum(text.count(w) for w in _POS)
    neg = sum(text.count(w) for w in _NEG)
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / max(1.0, pos + neg)


# ========= Topic tagging =========

_TAGS = [
    ("macro", r"\b(fed|ecb|boj|cpi|inflation|gdp|jobs|payrolls|nfp|recession|yield curve|breakeven)\b"),
    ("credit", r"\b(cds|spread|hy|ig|default|downgrade|upgrade|bond|coupon|issuance|leverage loan|clo)\b"),
    ("equity", r"\b(earnings|buyback|ipo|dividend|shareholder|eps|guidance|valuation|split)\b"),
    ("fx", r"\b(fx|currency|yen|euro|dollar|carry trade|usd|devaluation|peg)\b"),
    ("rates", r"\b(treasury|bund|jgb|rates|auction|duration|term premium)\b"),
    ("vol", r"\b(volatility|vix|skew|gamma|dispersion|tail hedge)\b"),
    ("commodities", r"\b(oil|brent|wti|metals|gold|gas|grain|corn|soy|copper)\b"),
    ("crypto", r"\b(bitcoin|eth|crypto|etf|spot etf)\b"),
]

_TAG_RX = [(name, re.compile(rx, re.I)) for name, rx in _TAGS]

def tag_topics(title: str, summary: str) -> List[str]:
    text = f"{title} {summary}"
    tags = [name for name, rx in _TAG_RX if rx.search(text)]
    return tags or ["general"]


# ========= Ticker mapping =========

def extract_cashtags(text: str) -> List[str]:
    return list({m.group(1) for m in _CASHTAG.finditer(text)})

def map_tickers(title: str, summary: str, name_map: Optional[Dict[str, str]] = None, hints: Sequence[str] = ()) -> List[str]:
    text = f"{title} {summary}"
    tickers = set(extract_cashtags(text))
    for h in hints:
        if h and h.isupper() and 1 <= len(h) <= 6:
            tickers.add(h)
    # fuzzy exact-name hits (simple contains)
    if name_map:
        lower = text.lower()
        for name, ticker in name_map.items():
            if name and ticker and name.lower() in lower:
                tickers.add(ticker.upper())
    return sorted(tickers)


# ========= De-dup cache =========

class SeenCache:
    def __init__(self, ttl_sec: int = 6 * 3600):
        self.ttl = ttl_sec
        self._store: Dict[str, float] = {}

    def _sweep(self) -> None:
        now = time.time()
        drop = [k for k, t in self._store.items() if now - t > self.ttl]
        for k in drop:
            self._store.pop(k, None)

    def seen(self, key: str) -> bool:
        self._sweep()
        if key in self._store:
            return True
        self._store[key] = time.time()
        return False


# ========= Bridge =========

class NewsBridge:
    def __init__(
        self,
        routes: Optional[Dict[str, Callable[[NewsEvent], None]]] = None,
        name_map: Optional[Dict[str, str]] = None,
        dedup_ttl_sec: int = 6 * 3600,
    ):
        """
        routes: mapping of route_name -> handler(event)
        name_map: {'Apple Inc': 'AAPL', 'Microsoft': 'MSFT', ...}
        """
        self.routes = routes or {"default": lambda e: None}
        self.name_map = name_map or {}
        self.seen = SeenCache(ttl_sec=dedup_ttl_sec)

    # ---- Ingestors ----

    def ingest_rss(self, feed_url: str, source: Optional[str] = None) -> List[RawItem]:
        if feedparser is None:
            raise RuntimeError("feedparser not installed. `pip install feedparser`")
        fp = feedparser.parse(feed_url)
        out: List[RawItem] = []
        src = source or _domain(feed_url) or "rss"
        for e in fp.entries:
            url = getattr(e, "link", "") or ""
            title = _clean_text(getattr(e, "title", "") or "")
            summary = _clean_text(getattr(e, "summary", "") or getattr(e, "description", "") or "")
            published = _to_dt(getattr(e, "published", None) or getattr(e, "updated", None) or _now_utc())
            if not url or not title:
                continue
            out.append(RawItem(source=src, url=url, title=title, summary=summary, published=published))
        return out

    def ingest_json_api(self, url: str, headers: Optional[Dict[str, str]] = None, json_path: Optional[str] = None, source: Optional[str] = None) -> List[RawItem]:
        """Generic JSON list endpoint. `json_path` like 'articles' to pluck array."""
        if requests is None:
            raise RuntimeError("requests not installed. `pip install requests`")
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        arr = data.get(json_path, data) if isinstance(data, dict) and json_path else data
        if not isinstance(arr, list):
            raise ValueError("API did not return a list; set a correct json_path")
        src = source or _domain(url) or "api"
        out: List[RawItem] = []
        for it in arr:
            title = _clean_text(str(it.get("title", "")))
            url_i = str(it.get("url") or it.get("link") or "")
            summary = _clean_text(str(it.get("description") or it.get("summary") or ""))
            published = _to_dt(it.get("publishedAt") or it.get("pubDate") or it.get("published") or _now_utc())
            hints = []
            if "tickers" in it and isinstance(it["tickers"], (list, tuple)):
                hints = [str(x).upper() for x in it["tickers"]]
            if title and url_i:
                out.append(RawItem(source=src, url=url_i, title=title, summary=summary, published=published, tickers_hint=hints, extra=it))
        return out

    # ---- Normalize / Enrich ----

    def normalize(self, raw: RawItem) -> Optional[NewsItem]:
        key = _hash_id(raw.source, raw.url, raw.title)
        if self.seen.seen(key):
            return None  # duplicate
        ticks = map_tickers(raw.title, raw.summary, self.name_map, raw.tickers_hint)
        sent = score_sentiment(f"{raw.title}. {raw.summary}")
        topics = tag_topics(raw.title, raw.summary)
        return NewsItem(
            id=key,
            source=raw.source,
            url=raw.url,
            domain=_domain(raw.url),
            title=_clean_text(raw.title),
            summary=_clean_text(raw.summary),
            published=raw.published or _now_utc(),
            tickers=ticks,
            sentiment=sent,
            topics=topics,
            raw={"extra": raw.extra} if raw.extra else {},
        )

    # ---- Routing ----

    def to_event(self, item: NewsItem) -> NewsEvent:
        pri = 2
        if any(t in {"macro", "rates", "credit"} for t in item.topics):
            pri = 3
        if abs(item.sentiment) > 0.5 or item.tickers:
            pri = max(pri, 4)
        return NewsEvent(
            kind="news.signal",
            at=item.published,
            priority=pri,
            payload={
                "id": item.id,
                "source": item.source,
                "domain": item.domain,
                "url": item.url,
                "title": item.title,
                "summary": item.summary,
                "published": item.published.isoformat(),
                "tickers": item.tickers,
                "sentiment": item.sentiment,
                "topics": item.topics,
            },
        )

    def route(self, item: NewsItem) -> None:
        evt = self.to_event(item)
        # basic routing rules (extend as needed)
        if "macro" in item.topics or "rates" in item.topics:
            handler = self.routes.get("macro", self.routes.get("default"))
        elif "credit" in item.topics:
            handler = self.routes.get("credit", self.routes.get("default"))
        elif "equity" in item.topics and item.tickers:
            handler = self.routes.get("equity", self.routes.get("default"))
        elif "vol" in item.topics:
            handler = self.routes.get("vol", self.routes.get("default"))
        elif "fx" in item.topics:
            handler = self.routes.get("fx", self.routes.get("default"))
        else:
            handler = self.routes.get("default")
        if handler:
            handler(evt)

    # ---- End-to-end helpers ----

    def run_rss(self, feeds: Sequence[str]) -> List[NewsItem]:
        items: List[NewsItem] = []
        for f in feeds:
            try:
                for raw in self.ingest_rss(f):
                    item = self.normalize(raw)
                    if item:
                        items.append(item)
                        self.route(item)
            except Exception as e:
                # log & continue
                print(f"[NewsBridge] RSS error for {f}: {e}")
        return items

    def run_api(self, endpoints: Sequence[Tuple[str, Optional[Dict[str, str]], Optional[str], Optional[str]]]) -> List[NewsItem]:
        """
        endpoints: list of (url, headers, json_path, source)
        """
        items: List[NewsItem] = []
        for url, headers, path, src in endpoints:
            try:
                raws = self.ingest_json_api(url, headers=headers, json_path=path, source=src)
                for raw in raws:
                    item = self.normalize(raw)
                    if item:
                        items.append(item)
                        self.route(item)
            except Exception as e:
                print(f"[NewsBridge] API error for {url}: {e}")
        return items


# ========= Example wiring =========

def _print_handler(evt: NewsEvent) -> None:
    p = evt.payload
    tick = f" [{','.join(p.get('tickers') or [])}]" if p.get("tickers") else ""
    print(f"[{evt.priority}] {p.get('published')} {p.get('title')}{tick}  <{p.get('domain')}>")

def load_name_map(path: Optional[str]) -> Dict[str, str]:
    """Load 'company_name,ticker' CSV (no heavy deps)."""
    if not path:
        return {}
    out: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    out[parts[0]] = parts[1].upper()
    except Exception as e:
        print(f"[NewsBridge] name_map load failed: {e}")
    return out


# ========= CLI (optional) =========

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NewsBridge CLI")
    ap.add_argument("--rss", nargs="*", help="RSS/Atom feed URLs", default=[])
    ap.add_argument("--api", nargs="*", help="JSON endpoints (NewsAPI-style)", default=[])
    ap.add_argument("--api-path", default=None, help="JSON path (e.g., 'articles')")
    ap.add_argument("--name-map", default=None, help="CSV path 'Company,Ticker'")
    ap.add_argument("--dump-json", action="store_true", help="Print normalized JSON to stdout")
    args = ap.parse_args()

    routes = {
        "default": _print_handler,
        "equity": _print_handler,
        "macro": _print_handler,
        "credit": _print_handler,
        "fx": _print_handler,
        "vol": _print_handler,
    }

    bridge = NewsBridge(routes=routes, name_map=load_name_map(args.name_map))

    items: List[NewsItem] = []
    if args.rss:
        items += bridge.run_rss(args.rss)
    if args.api:
        # naive shared headers; customize as needed or duplicate endpoints with their headers
        hdrs = {"User-Agent": "NewsBridge/1.0"}
        endpoints = [(u, hdrs, args.api_path, None) for u in args.api]
        items += bridge.run_api(endpoints)

    if args.dump_json:
        for it in items:
            print(json.dumps(asdict(it), default=str))