# news-intel/models/sentiment_model.py
"""
Sentiment Model (finance/news friendly)

API
----
sm = SentimentModel.auto()            # choose best available backend
sm.score("Stocks plunge on recession fears")  -> {"label": "neg", "score": -0.72, "confidence": 0.90}
sm.score_article({"title": "...", "body": "..."})
sm.batch_score(list_of_texts)

Notes
- Backends (in priority order):
    1) HF 'sentiment-analysis' pipeline (e.g., distilbert/roberta variants)
    2) NLTK VADER (vader_lexicon)
    3) Lightweight rule-based lexicon (this file)
- Output score is in [-1, 1]; label in {"neg","neu","pos"}; confidence in [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import importlib
import math
import re
import unicodedata

# --------------------- utils ---------------------

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\.\,\-\+\!\?\$%]")

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u2019", "'").strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def _to_label(score: float) -> str:
    if score > 0.15: return "pos"
    if score < -0.15: return "neg"
    return "neu"

def _to_confidence(score: float) -> float:
    # map |score| in [0,1] to confidence; gentle curve
    return max(0.0, min(1.0, 0.55 + 0.45 * abs(score)))


# --------------------- backends ---------------------

class _HFBackend:
    """HuggingFace transformers pipeline('sentiment-analysis')."""
    def __init__(self, model: Optional[str] = None):
        transformers = importlib.import_module("transformers")
        kwargs = {"top_k": None}
        if model:
            kwargs["model"] = model # type: ignore
        self.pipe = transformers.pipeline("sentiment-analysis", **kwargs)

    def score(self, text: str) -> Tuple[float, float]:
        res = self.pipe(text)[0]  # {'label': 'POSITIVE', 'score': 0.998}
        lbl = (res["label"] or "").upper()
        prob = float(res.get("score", 0.5))
        score = prob if "POS" in lbl else -prob if "NEG" in lbl else 0.0
        return (max(-1.0, min(1.0, score)), prob)


class _VaderBackend:
    """NLTK VADER; install via: pip install nltk && python -m nltk.downloader vader_lexicon"""
    def __init__(self):
        nltk = importlib.import_module("nltk")
        sa = importlib.import_module("nltk.sentiment")
        self.sid = sa.vader.SentimentIntensityAnalyzer()

    def score(self, text: str) -> Tuple[float, float]:
        vs = self.sid.polarity_scores(text)  # {'neg':0.12,'neu':0.70,'pos':0.18,'compound':0.06}
        comp = float(vs.get("compound", 0.0))
        return (max(-1.0, min(1.0, comp)), 0.5 + 0.5 * abs(comp))


class _RuleBackend:
    """
    Very small finance-tilted lexicon with negation/intensity handling.
    Not SOTA, but deterministic and fast.
    """
    POS = {
        "beat", "beats", "surge", "surged", "rally", "rallies", "rallied", "strong",
        "gain", "gains", "gained", "bullish", "optimism", "upgrade", "upgraded",
        "profit", "profits", "growth", "record", "outperform", "tops", "improve",
        "improves", "improved", "rebound", "rebounds", "rebounded", "exceed", "exceeds",
    }
    NEG = {
        "miss", "misses", "plunge", "plunges", "plunged", "slump", "slumps", "slumped",
        "weak", "downgrade", "downgraded", "loss", "losses", "fall", "falls", "fell",
        "bearish", "fear", "fears", "concern", "concerns", "warning", "warns", "warned",
        "recession", "lawsuit", "probe", "ban", "penalty", "decline", "declines", "declined",
    }
    NEGATORS = {"no", "not", "n't", "never", "hardly", "rarely"}
    BOOST_POS = {"very", "strongly", "significantly", "sharply"}
    BOOST_NEG = {"very", "severely", "sharply", "significantly"}

    def score(self, text: str) -> Tuple[float, float]:
        text = _normalize(text.lower())
        if not text:
            return (0.0, 0.4)
        toks = text.split()

        pos = neg = 0.0
        negated = False
        for i, t in enumerate(toks):
            w = t.strip(".,!?")
            if w in self.NEGATORS:
                negated = True
                continue
            boost = 1.0
            # simple preceding booster
            if i > 0:
                prev = toks[i-1].strip(".,!?")
                if prev in self.BOOST_POS or prev in self.BOOST_NEG:
                    boost = 1.3
            if w in self.POS:
                pos += ( -1.0 if negated else 1.0 ) * boost
                negated = False
            elif w in self.NEG:
                neg += ( -1.0 if negated else 1.0 ) * boost
                negated = False
            # reset negation after punctuation
            if t.endswith((".", "!", "?")):
                negated = False

        raw = 0.0
        if pos or neg:
            raw = (pos - neg) / max(1.0, (abs(pos) + abs(neg)))
        # squash to [-1,1]
        raw = max(-1.0, min(1.0, raw))
        return (raw, _to_confidence(raw))


# --------------------- public facade ---------------------

@dataclass
class SentimentResult:
    label: str
    score: float        # [-1,1]; positive -> bullish
    confidence: float   # [0,1]
    backend: str

class SentimentModel:
    """
    Facade over available backends.
    Prefer HF pipeline; else VADER; else rule-based.
    """

    def __init__(self, backend: Any, name: str):
        self.backend = backend
        self.name = name

    @classmethod
    def auto(cls, hf_model: Optional[str] = None) -> "SentimentModel":
        # Try HF
        try:
            return cls(_HFBackend(hf_model), "hf")
        except Exception:
            # Try VADER
            try:
                return cls(_VaderBackend(), "vader")
            except Exception:
                return cls(_RuleBackend(), "rule")

    @classmethod
    def prefer_hf(cls, model: Optional[str] = None) -> "SentimentModel":
        return cls(_HFBackend(model), "hf")

    @classmethod
    def prefer_vader(cls) -> "SentimentModel":
        return cls(_VaderBackend(), "vader")

    @classmethod
    def rule_only(cls) -> "SentimentModel":
        return cls(_RuleBackend(), "rule")

    def score(self, text: str) -> Dict[str, Any]:
        text = _normalize(text)
        s, conf = self.backend.score(text)
        return {
            "label": _to_label(s),
            "score": float(s),
            "confidence": float(conf),
            "backend": self.name,
        }

    def batch_score(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.score(t) for t in texts]

    def score_article(self, article: Dict[str, Any], title_weight: float = 0.35) -> Dict[str, Any]:
        """
        Blend title + body sentiment (title_weight in [0,1]).
        """
        title = article.get("title", "") or ""
        body = article.get("body", "") or ""
        rt = self.score(title)
        rb = self.score(body)
        s = title_weight * rt["score"] + (1.0 - title_weight) * rb["score"]
        conf = min(1.0, 0.5 * (rt["confidence"] + rb["confidence"]) + 0.25 * abs(s))
        return {
            "label": _to_label(s),
            "score": float(max(-1.0, min(1.0, s))),
            "confidence": float(conf),
            "backend": self.name,
            "parts": {"title": rt, "body": rb},
        }


# --------------------- self-test ---------------------

if __name__ == "__main__":
    sm = SentimentModel.auto()
    print(sm.score("Shares surged after the company beat expectations."))
    print(sm.score("Stock plunges on weak outlook and lawsuit warnings."))
    art = {"title": "Big Tech rally extends", "body": "Investors show optimism as profits improve significantly."}
    print(sm.score_article(art))