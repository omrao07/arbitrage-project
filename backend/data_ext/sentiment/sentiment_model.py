# backend/data_ext/sentiment/sentiment_model.py
"""
Unified sentiment model wrapper for social signals.

Priority of backends:
  1) FinBERT (ProsusAI/finbert) via transformers (finance-tuned)
  2) Generic Transformers sentiment pipeline (e.g., distilbert-base-uncased-finetuned-sst-2-english)
  3) TextBlob fallback (rule-based polarity)

Config (sentiment.yaml -> scoring):
-----------------------------------
scoring:
  model: "finbert"          # finbert | transformers | textblob
  use_transformers: true
  cache_results: true
  output_scale: [-1, 1]

Public API:
-----------
SentimentModel.from_config(cfg_dict)
SentimentModel(...).score(text) -> {"sentiment": float, "confidence": float, "model": str}
SentimentModel(...).score_batch(texts) -> List[...]
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
_HAVE_TRANSFORMERS = False
_HAVE_TEXTBLOB = False
try:
    from transformers import pipeline  # type: ignore
    _ HAVE_TRANSFORMERS = True  # type: ignore
except Exception:
    _HAVE_TRANSFORMERS = False

try:
    from textblob import TextBlob  # type: ignore
    _HAVE_TEXTBLOB = True
except Exception:
    _HAVE_TEXTBLOB = False


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class _Result:
    sentiment: float  # [-1, 1]
    confidence: float  # [0, 1]
    model: str


class SentimentModel:
    """
    Thin orchestrator over multiple possible backends with a uniform interface.
    """

    def __init__(
        self,
        backend: str = "finbert",
        use_transformers: bool = True,
        cache_results: bool = True,
        output_scale: Tuple[float, float] = (-1.0, 1.0),
        transformers_model: Optional[str] = None,
    ) -> None:
        self.backend = backend.lower().strip()
        self.use_transformers = bool(use_transformers)
        self.cache_results = bool(cache_results)
        self.out_min, self.out_max = output_scale
        self.transformers_model = transformers_model  # optional override

        self._cache: Dict[str, _Result] = {}

        # Initialize pipelines lazily
        self._pipe = None
        self._pipe_name = None

        if self.backend in ("finbert", "transformers"):
            if not (self.use_transformers and _HAVE_TRANSFORMERS):
                # Fall back cleanly
                self.backend = "textblob" if _HAVE_TEXTBLOB else "rule"
        elif self.backend == "textblob":
            if not _HAVE_TEXTBLOB:
                # If textblob not installed, try transformers or rule
                if self.use_transformers and _HAVE_TRANSFORMERS:
                    self.backend = "transformers"
                else:
                    self.backend = "rule"
        else:
            # Unknown backend -> try transformers -> textblob -> rule
            if self.use_transformers and _HAVE_TRANSFORMERS:
                self.backend = "transformers"
            elif _HAVE_TEXTBLOB:
                self.backend = "textblob"
            else:
                self.backend = "rule"

    # ------------------------------------------------------------------ #
    # Construction from config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SentimentModel":
        scoring = (cfg or {}).get("scoring", {})
        backend = str(scoring.get("model", "finbert")).lower()
        use_tf = bool(scoring.get("use_transformers", True))
        cache = bool(scoring.get("cache_results", True))
        out_scale = scoring.get("output_scale") or [-1, 1]
        if isinstance(out_scale, (list, tuple)) and len(out_scale) == 2:
            scale = (float(out_scale[0]), float(out_scale[1]))
        else:
            scale = (-1.0, 1.0)
        return cls(
            backend=backend,
            use_transformers=use_tf,
            cache_results=cache,
            output_scale=scale,
        )

    # ------------------------------------------------------------------ #
    # Lazy loader for transformers pipe
    # ------------------------------------------------------------------ #
    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        if not (self.use_transformers and _HAVE_TRANSFORMERS):
            return

        model_name = self.transformers_model
        if self.backend == "finbert":
            # finance-tuned
            model_name = model_name or os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
        else:
            # generic binary sentiment
            model_name = model_name or os.getenv(
                "SENTIMENT_MODEL",
                "distilbert-base-uncased-finetuned-sst-2-english",
            )
        try:
            self._pipe = pipeline("sentiment-analysis", model=model_name, truncation=True)
            self._pipe_name = model_name
        except Exception:
            self._pipe = None
            self._pipe_name = None
            # fallback chain is handled in score()

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"sentiment": 0.0, "confidence": 0.0, "model": "none"}

        if self.cache_results:
            key = _hash(text)
            hit = self._cache.get(key)
            if hit:
                return {"sentiment": hit.sentiment, "confidence": hit.confidence, "model": hit.model}

        # Try backend path
        res: _Result
        if self.backend in ("finbert", "transformers"):
            res = self._score_transformers(text) # type: ignore
            if res is None:
                res = self._score_textblob(text) or self._score_rule(text)
        elif self.backend == "textblob":
            res = self._score_textblob(text) or self._score_transformers(text) or self._score_rule(text)
        else:
            res = self._score_rule(text)

        # Clip/scale (already in [-1,1], but keep invariant)
        s = float(max(self.out_min, min(self.out_max, res.sentiment)))
        out = {"sentiment": s, "confidence": _clip01(float(res.confidence)), "model": res.model}

        if self.cache_results:
            self._cache[_hash(text)] = _Result(**out)  # type: ignore[arg-type]

        return out

    def score_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.score(t) for t in texts]

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #
    def _score_transformers(self, text: str) -> Optional[_Result]:
        if not (self.use_transformers and _HAVE_TRANSFORMERS):
            return None
        self._ensure_pipe()
        if self._pipe is None:
            return None

        try:
            out = self._pipe(text, truncation=True)
            # out example (sst2): [{'label': 'POSITIVE', 'score': 0.998}]
            # finbert: labels: Positive/Negative/Neutral
            obj = out[0] if isinstance(out, list) else out
            label = str(obj.get("label", "")).lower()
            score = float(obj.get("score", 0.5))

            if "neutral" in label:
                # map neutral to 0 with moderate confidence
                return _Result(sentiment=0.0, confidence=score, model=f"transformers:{self._pipe_name or 'auto'}")
            elif "neg" in label:
                return _Result(sentiment=-(score), confidence=score, model=f"transformers:{self._pipe_name or 'auto'}")
            elif "pos" in label:
                return _Result(sentiment=+(score), confidence=score, model=f"transformers:{self._pipe_name or 'auto'}")
            else:
                # Unknown label, fallback to centered mapping
                sent = (score * 2.0) - 1.0  # [0,1] -> [-1,1]
                return _Result(sentiment=sent, confidence=score, model=f"transformers:{self._pipe_name or 'auto'}")
        except Exception:
            return None

    def _score_textblob(self, text: str) -> Optional[_Result]:
        if not _HAVE_TEXTBLOB:
            return None
        try:
            blob = TextBlob(text)
            # polarity in [-1, 1], subjectivity [0,1]
            pol = float(blob.sentiment.polarity)
            sub = float(blob.sentiment.subjectivity)
            conf = 1.0 - (0.5 * sub)  # heuristic: more objective -> higher confidence
            return _Result(sentiment=pol, confidence=_clip01(conf), model="textblob")
        except Exception:
            return None

    def _score_rule(self, text: str) -> _Result:
        """
        Minimal rule-based fallback so pipeline never breaks.
        Very light sentiment lexicon.
        """
        t = text.lower()
        pos_words = ("beat", "bull", "moon", "rip", "rally", "strong", "surprise", "up", "growth", "profit")
        neg_words = ("miss", "bear", "dump", "fall", "weak", "warning", "down", "loss", "bankrupt", "fraud")

        score = 0
        for w in pos_words:
            if w in t:
                score += 1
        for w in neg_words:
            if w in t:
                score -= 1

        # Map integer score to [-0.8, 0.8] with soft clipping
        if score == 0:
            s = 0.0
        else:
            s = max(-0.8, min(0.8, score / 5.0))

        conf = 0.5 + 0.1 * abs(score)
        return _Result(sentiment=float(s), confidence=_clip01(float(conf)), model="rule")


# ---------------------------------------------------------------------- #
# Simple CLI for quick tests
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    example_cfg = {
        "scoring": {
            "model": os.getenv("SENTIMENT_BACKEND", "finbert"),
            "use_transformers": True,
            "cache_results": True,
            "output_scale": [-1, 1],
        }
    }
    model = SentimentModel.from_config(example_cfg)
    tests = [
        "TSLA to the moon, insane earnings beat!",
        "AAPL guidance weak, services decelerating.",
        "Market neutral into FOMC, no strong bias.",
    ]
    for t in tests:
        print(t, "->", model.score(t))