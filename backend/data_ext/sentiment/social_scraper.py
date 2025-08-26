# backend/data_ext/sentiment/social_scraper.py
"""
Social Sentiment Orchestrator

Pipeline:
  sources (reddit/x/tiktok/discord) -> raw posts
  -> sentiment_model.SentimentModel(score)
  -> normalize to unified schema
  -> publish to STREAM_ALT_SIGNALS

Feature flag: FEATURE_SENTIMENT must be true.

Config: backend/config/sentiment.yaml
-------------------------------------
sources:
  reddit:   {...}
  x:        {...}
  tiktok:   {...}
  discord:  {...}
scoring:
  model: "finbert"      # finbert | transformers | textblob
  use_transformers: true
  cache_results: true
  output_scale: [-1, 1]

Normalized output (published):
{
  "series_id": "SOC-<SRC>-<SYMBOL>",   # e.g., SOC-REDDIT-TSLA
  "timestamp": "<ISO8601Z>",
  "region": "GLOBAL",
  "metric": "social_sentiment",
  "value": <float in [-1,1]>,
  "symbol": "<SYMBOL or UNKNOWN>",
  "meta": {
    "source": "<reddit|x|tiktok|discord>",
    "confidence": <0..1>,
    "raw_ts": "<timestamp from source>",
    "extra": {...}  # select source metadata
  }
}
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
except Exception:
    yaml = None

# Source loaders
from .sources import reddit as src_reddit # type: ignore
from .sources import x as src_x
from .sources import tiktok as src_tiktok # type: ignore
from .sources import discord as src_discord

# Scoring
from .sentiment_model import SentimentModel

log = logging.getLogger(__name__)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)

SENTIMENT_YAML = os.getenv(
    "SENTIMENT_YAML",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "sentiment.yaml"),
)

Signal = Dict[str, Any]
RawPost = Dict[str, Any]


class SocialScraper:
    """
    Orchestrates multiple social sources -> sentiment scoring -> normalized publishing.
    """

    def __init__(self, yaml_path: Optional[str] = None):
        self.yaml_path = yaml_path or SENTIMENT_YAML
        self.model: Optional[SentimentModel] = None
        # simple dedupe on (series_id, timestamp); prevents double publishes in tight loops
        self._dedupe: set[Tuple[str, str]] = set()

    # --------------------------- Config ---------------------------

    def _load_yaml(self) -> Dict[str, Any]:
        if not yaml:
            raise RuntimeError("pyyaml not installed; cannot load sentiment.yaml")
        if not os.path.isfile(self.yaml_path):
            raise FileNotFoundError(f"sentiment config not found: {self.yaml_path}")
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # --------------------------- Sources --------------------------

    def _fetch_all(self, cfg: Dict[str, Any]) -> List[RawPost]:
        out: List[RawPost] = []
        scfg = (cfg.get("sources") or {})

        def run(src_name: str, fn, sc: Dict[str, Any]):
            if not sc or not sc.get("enabled"):
                return
            try:
                recs = fn(sc) or []
                if recs:
                    out.extend(recs)
                    log.info("%s: %d records", src_name, len(recs))
            except Exception as e:
                log.exception("%s fetch failed: %s", src_name, e)

        run("reddit", src_reddit.fetch, scfg.get("reddit", {}))
        run("x", src_x.fetch, scfg.get("x", {}))
        run("tiktok", src_tiktok.fetch, scfg.get("tiktok", {}))
        run("discord", src_discord.fetch, scfg.get("discord", {}))
        return out

    # --------------------------- Scoring --------------------------

    def _ensure_model(self, cfg: Dict[str, Any]) -> None:
        if self.model is None:
            self.model = SentimentModel.from_config(cfg)

    def _score_post(self, text: str) -> Dict[str, Any]:
        if not self.model:
            return {"sentiment": 0.0, "confidence": 0.0, "model": "none"}
        try:
            return self.model.score(text or "")
        except Exception:
            return {"sentiment": 0.0, "confidence": 0.0, "model": "error"}

    # --------------------------- Normalize ------------------------

    @staticmethod
    def _coerce_str(x: Any, default: str = "") -> str:
        return str(x) if x is not None else default

    def _normalize(self, raw: RawPost, score: Dict[str, Any]) -> Optional[Signal]:
        """
        Map a raw social post + score into the unified schema.
        """
        source = self._coerce_str(raw.get("source", "social")).lower()
        ts = self._coerce_str(raw.get("timestamp"))
        text = self._coerce_str(raw.get("text"))
        symbol = self._coerce_str(raw.get("symbol", "")).upper() or "UNKNOWN"

        if not ts or not text:
            return None

        series_id = f"SOC-{source.upper()}-{symbol}"

        # Choose a compact subset of metadata to avoid huge payloads
        meta_extra: Dict[str, Any] = {}
        rmeta = raw.get("meta") or {}
        for k in ("id", "score", "num_comments", "retweets", "likes", "views", "shares", "permalink", "link", "author", "server", "channel"):
            if k in rmeta:
                meta_extra[k] = rmeta[k]

        sig: Signal = {
            "series_id": series_id,
            "timestamp": ts,
            "region": "GLOBAL",
            "metric": "social_sentiment",
            "value": float(score.get("sentiment", 0.0)),
            "symbol": symbol,
            "meta": {
                "source": source,
                "confidence": float(score.get("confidence", 0.0)),
                "model": str(score.get("model", "")),
                "raw_ts": ts,
                "extra": meta_extra,
            },
        }
        return sig

    # --------------------------- Publish & Dedupe ------------------

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
            except Exception as e:
                log.exception("publish_stream failed: %s", e)
        return n

    # --------------------------- Orchestration ---------------------

    def run_once(self, cfg_override: Optional[Dict[str, Any]] = None) -> int:
        """
        Single cycle: load cfg -> fetch -> score -> normalize -> dedupe -> publish.
        Returns #published.
        """
        if not is_enabled("SENTIMENT"):
            log.info("FEATURE_SENTIMENT disabled; skipping SocialScraper.")
            return 0

        cfg = self._load_yaml()
        if cfg_override:
            cfg = {**cfg, **cfg_override}

        self._ensure_model(cfg)

        raw_posts = self._fetch_all(cfg)
        signals: List[Signal] = []

        for post in raw_posts:
            sc = self._score_post(self._coerce_str(post.get("text")))
            sig = self._normalize(post, sc)
            if sig:
                signals.append(sig)

        uniq = self._dedupe(signals) # type: ignore
        published = self._publish(uniq)
        log.info("SocialScraper: published %d sentiment signals", published)
        return published

    def run_forever(self, interval_sec: Optional[int] = None) -> None:
        """
        Loop forever using scoring/update cadence from YAML (default 5 min).
        Env override: SENTIMENT_INTERVAL_SEC.
        """
        if not is_enabled("SENTIMENT"):
            log.info("FEATURE_SENTIMENT disabled; SocialScraper not running.")
            return

        # Pick an interval
        default_interval = 300
        try:
            cfg = self._load_yaml()
            default_interval = int(cfg.get("normalization", {}).get("update_interval_sec", default_interval))
        except Exception:
            pass
        interval = int(os.getenv("SENTIMENT_INTERVAL_SEC", str(interval_sec or default_interval)))

        log.info("SocialScraper loop starting (interval=%ss)", interval)
        while True:
            try:
                self.run_once()
            except Exception as e:
                log.exception("SocialScraper tick failed: %s", e)
            time.sleep(interval)


# Simple CLI
if __name__ == "__main__":
    SocialScraper().run_once()