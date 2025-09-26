# news-intel/ingest/rss_feed.py
"""
RSS/Atom fetcher with caching + normalization.

Features
- Uses `feedparser` if available (recommended). Falls back to stdlib XML parse.
- Honors HTTP caching: ETag / Last-Modified to minimize bandwidth.
- Simple retries with exponential backoff + jitter.
- Normalizes items to the common Article schema via ingest.parser.normalize_article.
- Dedup across polls using a stable item key (id/link/title+date).

API
---
client = RSSClient(user_agent="news-intel/0.1", rps=1.5)
client.add_feed("https://example.com/rss", source="ExampleWire")

# one-shot
articles = client.poll_once()

# continuous (generator)
for batch in client.poll_loop(interval_s=60):
    ...

You can also call `fetch_and_parse(url)` to get a feed dict on demand.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import time
import typing as _t
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

try:
    import feedparser  # type: ignore
except Exception:  # noqa: BLE001
    feedparser = None  # type: ignore[assignment]

from .parser import normalize_article, clean_text

# ---------------------------- utils ---------------------------------


def _http_get(url: str, headers: dict, timeout: float) -> urllib.response.addinfourl: # type: ignore
    req = urllib.request.Request(url, headers=headers, method="GET")
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310 (caller-controlled URLs)


def _backoff_sleep(attempt: int, base: float = 0.4) -> None:
    delay = base * (2 ** attempt)
    time.sleep(delay + random.uniform(0, delay * 0.25))


def _to_iso(tm: _t.Optional[dt.datetime]) -> str:
    if not tm:
        return ""
    if tm.tzinfo is None:
        tm = tm.replace(tzinfo=dt.timezone.utc)
    return tm.astimezone(dt.timezone.utc).isoformat()


def _choose(*vals):
    for v in vals:
        if v:
            return v
    return ""


# ---------------------------- RSS client -----------------------------


class RSSClient:
    def __init__(
        self,
        user_agent: str = "news-intel/0.1",
        rps: float = 1.0,
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self._min_interval = 1.0 / max(0.001, rps)
        self._last_req_ts = 0.0

        # feed registry: url -> meta
        self._feeds: dict[str, dict] = {}
        # seen IDs to dedup across polls
        self._seen: set[str] = set()

    # ------------- public API -------------

    def add_feed(self, url: str, *, source: str = "rss", tags: _t.Optional[list[str]] = None) -> None:
        self._feeds[url] = {
            "source": source,
            "tags": tags or [],
            "etag": None,
            "last_modified": None,
        }

    def fetch_and_parse(self, url: str) -> dict:
        """Return a parsed feed dict (title, entries[]). Prefers feedparser; fallback to stdlib."""
        # rate limit
        now = time.monotonic()
        if now - self._last_req_ts < self._min_interval:
            time.sleep(self._min_interval - (now - self._last_req_ts))
        self._last_req_ts = time.monotonic()

        meta = self._feeds.get(url, {})
        headers = {
            "User-Agent": self.user_agent,
        }
        if meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]
        if meta.get("last_modified"):
            headers["If-Modified-Since"] = meta["last_modified"]

        # retry loop
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with _http_get(url, headers, self.timeout) as resp:
                    status = getattr(resp, "status", 200)
                    if status == 304:
                        return {"status": 304, "title": meta.get("title", ""), "entries": []}

                    body = resp.read()
                    etag = resp.headers.get("ETag")
                    lm = resp.headers.get("Last-Modified")

                    if feedparser is not None:
                        fp = feedparser.parse(body)
                        entries = []
                        for e in fp.entries:
                            entries.append(
                                {
                                    "id": _choose(getattr(e, "id", None), getattr(e, "guid", None), getattr(e, "link", None)),
                                    "title": getattr(e, "title", ""),
                                    "summary": getattr(e, "summary", ""),
                                    "link": getattr(e, "link", ""),
                                    "published": _to_iso(
                                        dt.datetime(*e.published_parsed[:6]) if getattr(e, "published_parsed", None) else None # type: ignore
                                    ),
                                }
                            )
                        out = {
                            "status": status,
                            "title": getattr(fp.feed, "title", "") if getattr(fp, "feed", None) else "",
                            "entries": entries,
                        }
                    else:
                        # minimal XML fallback: try to handle RSS 2.0 and Atom 1.0
                        entries, feed_title = _parse_xml_feed(body)
                        out = {"status": status, "title": feed_title, "entries": entries}

                    # update cache keys
                    meta["etag"] = etag or meta.get("etag")
                    meta["last_modified"] = lm or meta.get("last_modified")
                    meta["title"] = out["title"]
                    self._feeds[url] = meta
                    return out

            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    _backoff_sleep(attempt)
                    continue
                raise
            except Exception as e:  # network errors, parse issues
                last_exc = e
                if attempt < self.max_retries:
                    _backoff_sleep(attempt)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def poll_once(self) -> list[dict]:
        """Fetch all registered feeds once; return a list of normalized Article dicts (deduped)."""
        articles: list[dict] = []
        for url, meta in list(self._feeds.items()):
            try:
                feed = self.fetch_and_parse(url)
                if feed.get("status") == 304:
                    continue
                for it in feed.get("entries", []):
                    raw = {
                        "title": clean_text(it.get("title", "")),
                        "description": it.get("summary", ""),
                        "link": it.get("link", ""),
                        "pubDate": it.get("published", ""),
                        "id": it.get("id", ""),
                        "source": meta.get("source", "rss"),
                    }
                    art = normalize_article(raw, source=meta.get("source", "rss"))
                    # stable dedup key
                    key = art.get("id") or f"{art.get('url')}|{art.get('published_at')}|{art.get('title')[:64]}" # type: ignore
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                    # attach tags if any
                    if meta.get("tags"):
                        art.setdefault("tags", list(meta["tags"]))
                    articles.append(art)
            except Exception as e:
                # non-fatal; continue with other feeds
                err = {"feed": url, "error": str(e)}
                print("[rss] error:", json.dumps(err))
                continue
        return articles

    def poll_loop(self, interval_s: float = 60.0):
        """Generator that continuously polls, yielding non-empty article batches."""
        while True:
            batch = self.poll_once()
            if batch:
                yield batch
            time.sleep(max(1.0, interval_s))


# ---------------------------- XML fallback --------------------------


def _parse_xml_feed(xml_bytes: bytes) -> tuple[list[dict], str]:
    """
    Very small RSS/Atom parser for when 'feedparser' isn't available.
    Returns (entries, feed_title).
    """
    root = ET.fromstring(xml_bytes)

    # Namespaces we might encounter
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "rss": "http://purl.org/rss/1.0/",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    entries: list[dict] = []
    feed_title = ""

    # Atom 1.0
    if root.tag.endswith("feed"):
        t = root.find("./{http://www.w3.org/2005/Atom}title")
        feed_title = (t.text or "").strip() if t is not None else ""
        for entry in root.findall("./{http://www.w3.org/2005/Atom}entry"):
            eid = _first_text(entry, ["{http://www.w3.org/2005/Atom}id", "{http://purl.org/dc/elements/1.1/}identifier"])
            title = _first_text(entry, ["{http://www.w3.org/2005/Atom}title"])
            summary = _first_text(entry, ["{http://www.w3.org/2005/Atom}summary", "{http://purl.org/rss/1.0/modules/content/}encoded"])
            link = ""
            l = entry.find("{http://www.w3.org/2005/Atom}link")
            if l is not None:
                link = l.attrib.get("href", "")
            pub = _first_text(entry, ["{http://www.w3.org/2005/Atom}updated", "{http://purl.org/dc/elements/1.1/}date"])
            entries.append({"id": eid, "title": title, "summary": summary, "link": link, "published": pub})

    else:
        # RSS 2.0
        channel = root.find("channel")
        if channel is not None:
            t = channel.find("title")
            feed_title = (t.text or "").strip() if t is not None else ""
            for item in channel.findall("item"):
                eid = _first_text(item, ["guid", "{http://purl.org/dc/elements/1.1/}identifier"])
                title = _first_text(item, ["title"])
                summary = _first_text(item, ["description", "{http://purl.org/rss/1.0/modules/content/}encoded"])
                link = _first_text(item, ["link"])
                pub = _first_text(item, ["pubDate", "{http://purl.org/dc/elements/1.1/}date"])
                entries.append({"id": eid, "title": title, "summary": summary, "link": link, "published": pub})

    return entries, feed_title


def _first_text(node: ET.Element, paths: list[str]) -> str:
    for p in paths:
        el = node.find(p)
        if el is not None and el.text:
            return el.text.strip()
    return ""


# ---------------------------- self-test -----------------------------

if __name__ == "__main__":
    # quick demo (replace with feeds you use)
    feeds = [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.reuters.com/finance/markets/rss",
    ]
    cli = RSSClient(user_agent="news-intel/0.1", rps=0.5)
    for u in feeds:
        cli.add_feed(u, source="demo")

    # one-shot
    batch = cli.poll_once()
    print(f"fetched {len(batch)} articles")
    if batch:
        print(json.dumps(batch[0], indent=2)[:800])

    # or continuous
    # for arts in cli.poll_loop(120):
    #     print("batch:", len(arts))