# services/news_runner.py
from __future__ import annotations

import hashlib
import html
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import feedparser  # pip install feedparser
from bs4 import BeautifulSoup  # pip install beautifulsoup4

from platform import bootstrap # type: ignore
from platform import envelope as env # type: ignore

SERVICE = "news-runner"

# Streams (override via env)
OUT_STREAM = os.getenv("OUT_STREAM", "STREAM_SENTIMENT_REQUESTS")
GROUP = os.getenv("GROUP", "news_runner_v1")

# Config
DEFAULT_FEEDS = [
    # Add/remove as needed; you can also load from configs/news_feeds.yml later
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.ft.com/?format=rss",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "https://www.bloomberg.com/feed/podcast/etfs-quant?srnd=technology-vp",  # example
]
POLL_SECONDS = int(os.getenv("NEWS_POLL_SECONDS", "60"))
BATCH_MAX = int(os.getenv("NEWS_BATCH_MAX", "25"))
LANG = os.getenv("SENTIMENT_LANG", "en")


# ------------------------------- Helpers -----------------------------------

def _norm_text(html_text: str) -> str:
    """Strip HTML → clean text."""
    if not html_text:
        return ""
    txt = html.unescape(html_text)
    soup = BeautifulSoup(txt, "html.parser")
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return " ".join(soup.get_text(separator=" ").split())


def _article_fingerprint(url: str, title: str, summary: str) -> str:
    s = f"{url}||{title}||{summary}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Article:
    url: str
    title: str
    summary: str
    published_ts: Optional[str]
    source: Optional[str]


def _parse_feed(url: str) -> Tuple[str, List[Article]]:
    d = feedparser.parse(url)
    src = d.get("feed", {}).get("title") or url # type: ignore
    out: List[Article] = []
    for e in d.get("entries", []): # type: ignore
        link = (e.get("link") or "").strip() # type: ignore
        title = (e.get("title") or "").strip() # type: ignore
        # prefer "summary" then "description" then nothing
        summary_html = e.get("summary") or e.get("description") or ""
        summary = _norm_text(summary_html) # type: ignore
        published = None
        for key in ("published", "updated", "created"):
            if e.get(key):
                published = str(e.get(key))
                break
        if link or title or summary:
            out.append(Article(url=link, title=title, summary=summary, published_ts=published, source=src))
    return src, out


# ------------------------------- Main --------------------------------------

class NewsRunner:
    def __init__(self, feeds: List[str]):
        self.ctx = bootstrap.init(SERVICE)
        self.tracer = self.ctx["tracer"]
        self.METRICS = self.ctx["metrics"]
        self.r = self.ctx["redis"]
        self.audit = self.ctx["audit"]
        self.ent = self.ctx["ent"]

        self.feeds = feeds
        self._stop = False

        # Simple de-dupe cache in Redis (TTL)
        self.hash_ttl = int(os.getenv("NEWS_HASH_TTL", "86400"))  # 1 day
        self.hash_prefix = os.getenv("NEWS_HASH_PREFIX", "news:seen:")

        # Metrics
        self._m_ingested = self.METRICS.tasks_total.labels(SERVICE, "ingest")
        self._m_published = self.METRICS.tasks_total.labels(SERVICE, "publish")
        self._m_errors = self.METRICS.task_errors.labels(SERVICE, "ingest")

    def _seen(self, fp: str) -> bool:
        key = self.hash_prefix + fp
        try:
            # setnx returns True if key set (i.e., not seen before)
            if self.r.setnx(key, 1):
                self.r.expire(key, self.hash_ttl)
                return False
            return True
        except Exception:
            # Fail-open: treat as unseen
            return False

    def _publish_sentiment(self, arts: List[Article]) -> int:
        if not arts:
            return 0

        texts = [f"{a.title}. {a.summary}" if a.summary else a.title for a in arts]
        meta = [{"url": a.url, "source": a.source, "published_ts": a.published_ts} for a in arts]

        e = env.new(
            schema_name="sentiment.request",
            payload={
                "texts": texts[:BATCH_MAX],
                "language": LANG,
                "meta": {"batch": meta[:BATCH_MAX], "source": "news_runner"},
            },
            producer={"svc": SERVICE, "roles": ["research"]},
        )
        self.r.xadd(OUT_STREAM, e.flatten_for_stream())
        self._m_published.inc()
        return len(texts[:BATCH_MAX])

    def run_once(self) -> int:
        total_new = 0
        for url in self.feeds:
            with self.tracer.start_as_current_span("news.fetch", attributes={"feed": url}):
                try:
                    src, arts = _parse_feed(url)
                except Exception as e:
                    self._m_errors.inc()
                    continue

            new_batch: List[Article] = []
            for a in arts:
                fp = _article_fingerprint(a.url, a.title, a.summary)
                if not self._seen(fp):
                    new_batch.append(a)

            if new_batch:
                with self.METRICS.latency_timer("publish"):
                    sent = self._publish_sentiment(new_batch)
                    total_new += sent

                # Audit summary (no raw text)
                self.audit.record(
                    action="news_batch_publish",
                    resource="sentiment/analyze",
                    user=None,
                    corr_id=None,
                    region=os.getenv("REGION", "US"),
                    policy_hash=os.getenv("POLICY_HASH"),
                    details={"feed": url, "source": src, "n_articles": len(new_batch), "sent_to_sentiment": sent},
                    input_for_hash={"feed": url, "n": len(new_batch)},
                )

            # Ingest metric per feed
            self._m_ingested.inc()

        return total_new

    def run_forever(self):
        log = __import__("logging").getLogger(SERVICE)
        log.info("Starting news runner with %d feeds, poll=%ss", len(self.feeds), POLL_SECONDS)

        while not self._stop:
            try:
                n = self.run_once()
                log.info("Cycle complete: published=%d", n)
            except Exception as e:
                self._m_errors.inc()
                log.exception("Error in run_once: %s", e)
            # Sleep with small increments so we can stop promptly
            for _ in range(POLL_SECONDS):
                if self._stop:
                    break
                time.sleep(1)

    def stop(self, *_):
        self._stop = True


def main():
    feeds_env = os.getenv("NEWS_FEEDS", "")
    feeds = [f.strip() for f in feeds_env.split(",") if f.strip()] or DEFAULT_FEEDS

    nr = NewsRunner(feeds)
    signal.signal(signal.SIGINT, nr.stop)
    signal.signal(signal.SIGTERM, nr.stop)
    nr.run_forever()


if __name__ == "__main__":
    main()