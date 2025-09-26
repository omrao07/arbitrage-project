# news-intel/pipeline/pipeline_runner.py
"""
End-to-end enrichment pipeline runner.

Data flow
    ingest (RSS/API) -> normalize -> enrich:
        - NER -> actor_linker (KB)
        - topics (rule-based tagger, optional ML)
        - sentiment
    -> storage (adapter/jsonl/stdout)

Usage
-----
python -m news_intel.pipeline.pipeline_runner \
  --rss https://feeds.bbci.co.uk/news/world/rss.xml \
  --rss https://www.reuters.com/finance/markets/rss \
  --jsonl-out out/news_enriched.jsonl \
  --loop 120

or call from code:
    runner = PipelineRunner(kb=my_kb)
    runner.add_rss("https://...")
    runner.run_once()
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# ---- ingest & normalize ----
from news_intel.ingest.rss_feed import RSSClient # type: ignore
from news_intel.ingest.api_fetcher import ApiFetcher # type: ignore
from news_intel.ingest.parser import normalize_article # type: ignore

# ---- enrich ----
from news_intel.models.ner_model import NERModel # type: ignore
from news_intel.enrich.actor_linker import KnowledgeBase, KBEntity, extract_mentions, link_actors # type: ignore
from news_intel.enrich.topic_tagging import TopicTagger # type: ignore
from news_intel.models.sentiment_model import SentimentModel # type: ignore
from news_intel.models.topic_model import TopicModel  # type: ignore # optional ML multi-label

# ---- optional storage adapter ----
try:
    # expected to expose StorageAdapter with .write_many(list[dict])
    from news_intel.pipeline.storage_adapter import StorageAdapter  # type: ignore
except Exception:  # noqa: BLE001
    StorageAdapter = None  # type: ignore[assignment]


# ---------------------------- helpers ----------------------------

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


@dataclass
class PipelineConfig:
    # ingest
    rss_feeds: List[str] = field(default_factory=list)
    api_base: Optional[str] = None
    api_params: Dict[str, Any] = field(default_factory=dict)
    api_headers: Dict[str, str] = field(default_factory=dict)
    rps: float = 1.0
    timeout: float = 10.0

    # enrich
    topic_top_k: int = 4
    min_topic_conf: float = 0.15
    title_sent_weight: float = 0.35

    # storage
    jsonl_out: Optional[str] = None
    pretty_stdout: bool = False


# ---------------------------- pipeline runner ----------------------------

class PipelineRunner:
    def __init__(
        self,
        cfg: Optional[PipelineConfig] = None,
        *,
        kb: Optional[KnowledgeBase] = None,
        storage: Optional[Any] = None,  # StorageAdapter-like
        use_ml_topics: bool = False,
    ):
        self.cfg = cfg or PipelineConfig()
        # ingestors
        self.rss = RSSClient(rps=self.cfg.rps, timeout=self.cfg.timeout)
        self.api: Optional[ApiFetcher] = None
        if self.cfg.api_base:
            self.api = ApiFetcher(
                base_url=self.cfg.api_base,
                default_headers=self.cfg.api_headers,
                rps=max(0.1, self.cfg.rps),
                timeout=self.cfg.timeout,
            )

        # enrich components
        self.ner = NERModel.auto()                # spaCy -> HF -> regex
        self.tagger = TopicTagger()               # rule-based tagger
        self.topic_ml: Optional[TopicModel] = TopicModel() if use_ml_topics else None
        self.sent = SentimentModel.auto()

        # knowledge base for actor linking
        self.kb = kb or self._default_kb()

        # storage
        self.storage = storage or self._default_storage()

        # dedup across runs
        self._seen_ids: set[str] = set()

    # --- public API ---

    def add_rss(self, url: str, *, source: str = "rss") -> None:
        self.rss.add_feed(url, source=source)

    def run_once(self) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []

        # 1) Ingest RSS
        rss_articles = self.rss.poll_once() if self.cfg.rss_feeds else []
        batch.extend(rss_articles)

        # 2) Ingest API (one-shot simple example)
        if self.api and self.cfg.api_base:
            try:
                page = self.api.get_json("", params=self.cfg.api_params)  # base_url already set
                items = page.get("articles") or page.get("results") or page.get("data") or []
                arts = self.api.map_to_articles(
                    items,
                    mapping={
                        "id": "id",
                        "title": "title",
                        "body": "description",
                        "url": "url",
                        "published_at": "published_at",
                        "source": lambda it: (it.get("source") or {}).get("name", "api"),
                    },
                    default_source="api",
                )
                batch.extend(arts)
            except Exception as e:
                print(f"[api] error: {e}", file=sys.stderr)

        if not batch:
            return []

        # 3) Enrich
        enriched = self._enrich_many(batch)

        # 4) Store
        self._store(enriched)

        return enriched

    def run_loop(self, interval_s: float = 60.0) -> None:
        stop = False

        def _sig(_sig, _frm):
            nonlocal stop
            stop = True
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, _sig)

        while not stop:
            try:
                out = self.run_once()
                if out:
                    print(f"[{_now_iso()}] processed {len(out)} articles")
            except Exception as e:
                print(f"[pipeline] run_once error: {e}", file=sys.stderr)
            time.sleep(max(1.0, interval_s))

    # --- internals ---

    def _enrich_many(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        texts: List[str] = []

        # prepare batch sentiment/topic text
        for a in articles:
            text = f"{a.get('title','')}\n\n{a.get('body','')}".strip()
            texts.append(text)

        # batch topic ML (if available/trained)
        ml_topics: Optional[List[List[Dict[str, Any]]]] = None
        if self.topic_ml:
            ml_topics = []
            for probs in self.topic_ml.predict_batch(texts, top_k=self.cfg.topic_top_k, min_conf=self.cfg.min_topic_conf):
                ml_topics.append([{"topic": t.topic, "confidence": t.confidence} for t in probs])

        # process each article
        for i, a in enumerate(articles):
            # dedup
            key = a.get("id") or f"{a.get('url')}|{a.get('published_at')}|{a.get('title','')[:64]}"
            if key in self._seen_ids:
                continue
            self._seen_ids.add(key)

            text = texts[i]
            # 1) NER → actor linking
            ents = self.ner.extract(text)
            mentions = self.ner.to_mentions(ents)
            links = link_actors(text, self.kb, mentions=mentions, min_confidence=0.80)
            actors = [
                {
                    "text": l.mention.text,
                    "span": l.mention.span,
                    "label": l.mention.label,
                    "entity_id": l.entity_id,
                    "confidence": l.confidence,
                    "method": l.method,
                    "evidence": l.evidence,
                }
                for l in links if l.entity_id
            ]

            # 2) topics (rule tagger + optional ML)
            rb_topics = self.tagger.tag(text, top_k=self.cfg.topic_top_k, min_conf=self.cfg.min_topic_conf)
            topics = [{"topic": t.topic, "confidence": t.confidence, "evidence": t.evidence} for t in rb_topics]
            # optional late-fuse ML topics if present
            if ml_topics:
                # merge by topic with max confidence
                by_topic: Dict[str, Dict[str, Any]] = {t["topic"]: {"topic": t["topic"], "confidence": float(t["confidence"])} for t in ml_topics[i]}
                for t in topics:
                    cur = by_topic.get(t["topic"], {"topic": t["topic"], "confidence": 0.0})
                    cur["confidence"] = max(cur["confidence"], float(t["confidence"]))
                    by_topic[t["topic"]] = cur
                # keep top_k
                topics = sorted(by_topic.values(), key=lambda x: x["confidence"], reverse=True)[: self.cfg.topic_top_k]

            # 3) sentiment
            sent = self.sent.score_article(a, title_weight=self.cfg.title_sent_weight)

            # assemble enriched article
            enr = dict(a)
            enr["enriched_at"] = _now_iso()
            enr["topics"] = topics
            enr["actors"] = actors
            enr["sentiment"] = sent
            out.append(enr)

        return out

    def _store(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        # try storage adapter
        if StorageAdapter and isinstance(self.storage, StorageAdapter):
            try:
                self.storage.write_many(rows)  # type: ignore[attr-defined]
                return
            except Exception as e:
                print(f"[storage] adapter error, falling back to file/stdout: {e}", file=sys.stderr)

        # file output
        if self.cfg.jsonl_out:
            os.makedirs(os.path.dirname(self.cfg.jsonl_out) or ".", exist_ok=True)
            with open(self.cfg.jsonl_out, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            for r in rows:
                if self.cfg.pretty_stdout:
                    print(json.dumps(r, indent=2, ensure_ascii=False))
                else:
                    print(json.dumps(r, ensure_ascii=False))

    @staticmethod
    def _default_kb() -> KnowledgeBase:
        # Minimal seed KB; replace/extend with your own entities
        ents = [
            KBEntity(id="org:msft", name="Microsoft Corporation", type="org", aliases=["Microsoft", "MSFT"], tickers=["MSFT"]),
            KBEntity(id="org:aapl", name="Apple Inc.", type="org", aliases=["Apple", "AAPL"], tickers=["AAPL"]),
            KBEntity(id="org:ibm",  name="International Business Machines", type="org", aliases=["IBM"], tickers=["IBM"]),
            KBEntity(id="person:nadella", name="Satya Nadella", type="person", aliases=["Mr. Nadella", "Satya N."]),
        ]
        return KnowledgeBase(ents)

    def _default_storage(self):
        # If a StorageAdapter is implemented in your repo, prefer that; otherwise sentinel object
        return object()


# ---------------------------- CLI ----------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="News Intel Enrichment Pipeline")
    p.add_argument("--rss", action="append", default=[], help="RSS/Atom feed URL (repeatable)")
    p.add_argument("--api-base", default=None, help="Base URL for a JSON API (optional)")
    p.add_argument("--api-param", action="append", default=[], help="key=value API param (repeatable)")
    p.add_argument("--api-header", action="append", default=[], help="key=value API header (repeatable)")
    p.add_argument("--rps", type=float, default=1.0, help="Requests per second for fetchers")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    p.add_argument("--jsonl-out", default=None, help="Append-enriched rows to this JSONL file")
    p.add_argument("--pretty", action="store_true", help="Pretty-print to stdout instead of JSONL")
    p.add_argument("--loop", type=float, default=0.0, help="Poll loop interval seconds (0 = run once)")
    p.add_argument("--use-ml-topics", action="store_true", help="Enable optional ML TopicModel fusion if available")
    return p.parse_args(argv)


def _kv_pairs(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in pairs:
        if "=" in s:
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = PipelineConfig(
        rss_feeds=list(args.rss),
        api_base=args.api_base,
        api_params=_kv_pairs(args.api_param),
        api_headers=_kv_pairs(args.api_header),
        rps=args.rps,
        timeout=args.timeout,
        jsonl_out=args.jsonl_out,
        pretty_stdout=args.pretty,
    )

    runner = PipelineRunner(cfg, use_ml_topics=bool(args.use_ml_topics))

    # register RSS feeds
    for u in cfg.rss_feeds:
        runner.add_rss(u, source="rss")

    if args.loop and args.loop > 0:
        runner.run_loop(interval_s=float(args.loop))
    else:
        out = runner.run_once()
        print(f"processed {len(out)} articles", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())