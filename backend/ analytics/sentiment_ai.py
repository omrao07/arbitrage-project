# backend/analytics/sentiment_ai.py
from __future__ import annotations

import re
import functools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# optional type import (doesn't hard-depend)
try:
    from backend.ingestion.news.news_base import NewsEvent # type: ignore
except Exception:
    class NewsEvent:  # minimal stub
        def __init__(self, headline: str, summary: str = "", **kw):
            self.headline = headline
            self.summary = summary


@dataclass(slots=True)
class SentimentResult:
    label: str        # "POS" | "NEG" | "NEU"
    score: float      # normalized to [-1, 1]
    confidence: float # [0, 1]
    model: str        # "finbert" | "vader"
    text: str
    meta: Dict[str, Any]


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:512]  # keep inputs short for speed


class _FinBert:
    """Finance-tuned BERT sentiment via transformers pipeline."""
    def __init__(self):
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            raise ImportError("transformers required for FinBERT") from e

        # Try common finance-tuned checkpoints
        model_candidates = [
            "ProsusAI/finbert",            # widely used finance sentiment
            "yiyanghkust/finbert-tone",    # alternative
        ]

        last_err = None
        for m in model_candidates:
            try:
                self.pipe = pipeline("text-classification", model=m, tokenizer=m, truncation=True)
                self.model_id = m
                break
            except Exception as e:
                last_err = e
                self.pipe = None
        if self.pipe is None:
            raise RuntimeError(f"Failed to load FinBERT models: {last_err}")

    def __call__(self, texts: List[str]) -> List[SentimentResult]:
        outs = self.pipe(texts, batch_size=min(8, len(texts))) # type: ignore
        results: List[SentimentResult] = []
        for text, o in zip(texts, outs):
            raw_label = o.get("label", "").upper()
            score = float(o.get("score", 0.0))
            # Map raw_label -> (label, signed score in [-1,1])
            if "NEG" in raw_label:
                label = "NEG"; sent = -score
            elif "POS" in raw_label:
                label = "POS"; sent = +score
            else:
                label = "NEU"; sent = 0.0

            results.append(SentimentResult(
                label=label,
                score=sent,                 # already signed in [-1,1]
                confidence=abs(sent),       # simple proxy
                model="finbert",
                text=text,
                meta={"raw": o, "model_id": self.model_id},
            ))
        return results


class _Vader:
    """VADER fallback (no GPU, fast)."""
    def __init__(self):
        try:
            import nltk  # type: ignore
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
        except Exception as e:
            raise ImportError("nltk required for VADER") from e

        # ensure lexicon
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")  # type: ignore
        except LookupError:
            nltk.download("vader_lexicon")  # type: ignore
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
        self.an = SentimentIntensityAnalyzer()

    def __call__(self, texts: List[str]) -> List[SentimentResult]:
        results: List[SentimentResult] = []
        for t in texts:
            s = self.an.polarity_scores(t)
            comp = float(s.get("compound", 0.0))       # already in [-1, 1]
            if comp > 0.05:
                label = "POS"
            elif comp < -0.05:
                label = "NEG"
            else:
                label = "NEU"
            results.append(SentimentResult(
                label=label,
                score=comp,
                confidence=abs(comp),
                model="vader",
                text=t,
                meta={"raw": s},
            ))
        return results


class SentimentModel:
    """
    Unified interface with graceful fallback:
      - Tries FinBERT (transformers); if unavailable, falls back to VADER (nltk).
    Methods:
      - analyze(text) -> SentimentResult
      - analyze_many(texts) -> List[SentimentResult]
      - analyze_event(NewsEvent) -> SentimentResult
    """
    def __init__(self, prefer_finbert: bool = True):
        self.backend = None
        self.name = ""
        errors: List[str] = []

        if prefer_finbert:
            try:
                self.backend = _FinBert()
                self.name = "finbert"
            except Exception as e:
                errors.append(f"FinBERT unavailable: {e}")

        if self.backend is None:
            try:
                self.backend = _Vader()
                self.name = "vader"
            except Exception as e:
                errors.append(f"VADER unavailable: {e}")

        if self.backend is None:
            raise RuntimeError("No sentiment backend available. " + " | ".join(errors))

    @functools.lru_cache(maxsize=4096)
    def _analyze_cached(self, text: str) -> SentimentResult:
        # dispatch to backend; the backend expects list[str]
        return self.backend([text])[0] # type: ignore

    def analyze(self, text: str) -> SentimentResult:
        text = _clean_text(text)
        if not text:
            return SentimentResult(label="NEU", score=0.0, confidence=0.0, model=self.name, text="", meta={})
        return self._analyze_cached(text)

    def analyze_many(self, texts: Iterable[str]) -> List[SentimentResult]:
        cleaned = [_clean_text(t) for t in texts if t and _clean_text(t)]
        if not cleaned:
            return []
        # Use backend batch for speed; caching helps repeated headlines
        return self.backend.process(cleaned) # type: ignore

    def analyze_event(self, ev: NewsEvent) -> SentimentResult:
        # Prefer headline; append a short summary if present for context
        text = ev.headline
        if ev.summary:
            text = f"{ev.headline}. {ev.summary}"
        return self.analyze(text)


# ---------- Convenience: label -> numeric weight ----------

LABEL_WEIGHTS = {"NEG": -1.0, "NEU": 0.0, "POS": +1.0}

def sentiment_weight(sr: SentimentResult, min_conf: float = 0.25) -> float:
    """
    Convert a SentimentResult to a portfolio weight contribution in [-1, 1].
    Applies a confidence floor to ignore weak/noisy signals.
    """
    if sr.confidence < min_conf:
        return 0.0
    return float(max(-1.0, min(1.0, sr.score)))