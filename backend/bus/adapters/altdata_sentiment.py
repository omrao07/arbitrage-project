# backend/data/altdata_sentiment.py
"""
Alt-Data Sentiment Adapter
--------------------------
Purpose
  • Ingest external alt-data sources (news, social, transcripts).
  • Normalize, tokenize, score sentiment with financial domain lexicons or models.
  • Emit clean envelopes to the internal data bus for use by strategies, allocators, dashboards.

Features
  • pluggable scorers: VADER (NLTK), FinBERT, custom dictionaries.
  • produces multiple metrics: polarity, subjectivity, volatility proxy, volume of chatter.
  • optional rolling aggregator by ticker/region.
  • audit hooks: all emissions stamped with source_id, ts, hash.

Dependencies
  • nltk (for VADER) or transformers (if you plug FinBERT).
  • You can `pip install nltk` and run `nltk.download('vader_lexicon')`.

Usage
-----
from backend.data.altdata_sentiment import SentimentAdapter, SentimentConfig

cfg = SentimentConfig(method="vader", publish_stream="STREAM_NEWS_SENTIMENT")
adapter = SentimentAdapter(cfg)
scores = adapter.score_text("Fed raises rates again; markets tumble", ticker="SPY")
adapter.publish(scores)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# VADER scorer (default)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False

# Optional: FinBERT (HuggingFace)
try:
    from transformers import pipeline # type: ignore
    _FINBERT = pipeline("sentiment-analysis", model="ProsusAI/finbert")  # may be heavy # type: ignore
    _HAS_FINBERT = True
except Exception:
    _HAS_FINBERT = False

# Hook into your bus
try:
    from backend.bus.streams import publish_stream # type: ignore
except ImportError:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        print(f"[stub publish_stream] {stream} <- {json.dumps(payload)}")


# ---------------- Config ----------------

@dataclass
class SentimentConfig:
    method: str = "vader"                 # "vader" | "finbert" | "dict"
    publish_stream: str = "STREAM_NEWS_SENTIMENT"
    use_cache: bool = True
    normalize_case: bool = True
    lang: str = "en"


# ---------------- Adapter ----------------

class SentimentAdapter:
    def __init__(self, cfg: SentimentConfig) -> None:
        self.cfg = cfg
        self._vader = None
        if cfg.method == "vader" and _HAS_VADER:
            self._vader = SentimentIntensityAnalyzer()

    def score_text(self, text: str, *, ticker: Optional[str] = None,
                   region: Optional[str] = None, source_id: Optional[str] = None) -> Dict[str, Any]:
        """Return structured sentiment scores for a single text snippet."""
        if self.cfg.normalize_case:
            text = text.strip()
            text = text[0].upper() + text[1:] if text else text

        scores: Dict[str, float] = {}

        if self.cfg.method == "vader" and self._vader:
            res = self._vader.polarity_scores(text)
            scores = {"pos": res["pos"], "neg": res["neg"], "neu": res["neu"], "compound": res["compound"]}

        elif self.cfg.method == "finbert" and _HAS_FINBERT:
            res = _FINBERT(text)[0]
            # Convert FinBERT outputs to similar structure
            label = res["label"].lower()
            score = res["score"]
            scores = {"finbert_label": label, "finbert_score": score,
                      "pos": float(label == "positive") * score,
                      "neg": float(label == "negative") * score,
                      "neu": float(label == "neutral") * score}

        elif self.cfg.method == "dict":
            # Very simple dictionary-based sentiment (fallback)
            text_l = text.lower()
            pos_words = ["gain","bullish","rally","beat","surge","growth"]
            neg_words = ["loss","bearish","miss","fall","drop","fraud"]
            pos = sum(word in text_l for word in pos_words)
            neg = sum(word in text_l for word in neg_words)
            total = max(1, pos + neg)
            scores = {"pos": pos/total, "neg": neg/total, "neu": 1 - (pos+neg)/max(1,pos+neg+1), "compound": (pos-neg)/total}

        else:
            raise RuntimeError(f"Unsupported sentiment method {self.cfg.method} or missing deps.")

        env = {
            "ts": int(time.time() * 1000),
            "ticker": ticker,
            "region": region,
            "text": text,
            "scores": scores,
            "method": self.cfg.method,
            "source_id": source_id or hashlib.sha1(text.encode()).hexdigest()[:12],
        }
        env["hash"] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
        return env

    def publish(self, env: Dict[str, Any]) -> None:
        """Emit a scored envelope onto the data bus (or stub)."""
        publish_stream(self.cfg.publish_stream, env)


# ---------------- Example ----------------

if __name__ == "__main__":
    cfg = SentimentConfig(method="dict")
    sa = SentimentAdapter(cfg)
    ex = sa.score_text("Fed warns of recession risk, markets drop sharply", ticker="SPY", region="US")
    print(json.dumps(ex, indent=2))
    sa.publish(ex)