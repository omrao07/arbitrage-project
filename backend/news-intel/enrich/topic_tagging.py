# news-intel/enrich/topic_tagging.py
"""
Topic Tagging
-------------
Assign finance/news topics to an article using a simple hybrid approach:
- rule-based keyword scoring (fast, no deps)
- optional ML (TF-IDF + linear model) if scikit-learn is available

Public API:
    tag_text(text, taxonomy=None, top_k=3) -> List[TopicScore]
    tag_article(article: dict with 'title'/'body') -> List[TopicScore]

A TopicScore is:
    {
        "topic": str,
        "confidence": float (0..1),
        "evidence": {"hits": [matched terms], "counts": {term: n}, "ml": score?}
    }

You can pass a custom taxonomy like:
    custom = {
        "Earnings": ["earnings", "EPS", "guidance", "beat", "miss"],
        "M&A": ["acquisition", "merger", "takeover", "LBO", "deal"],
        ...
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import math
import re
import unicodedata

# ------------------------- default taxonomy -------------------------

DEFAULT_TAXONOMY: Dict[str, Iterable[str]] = {
    "Markets": [
        "stocks", "equities", "bonds", "treasur[yi]", "credit", "volatility",
        "selloff", "rally", "index", "S&P", "Nasdaq", "Dow", "ETF",
    ],
    "Macro": [
        "inflation", "cpi", "ppi", "jobs report", "nonfarm", "unemployment",
        "gdp", "growth", "recession", "rate hike", "rate cut", r"fed(er(al)?|eral) reserve",
        "ecb", "boe", "boj", "central bank", "yield curve", "core inflation",
    ],
    "Earnings": [
        "earnings", "eps", "revenue", "guidance", "outlook", "beat", "miss",
        "quarter", "fiscal", "results", "buyback",
    ],
    "M&A": [
        "merger", "acquisition", "takeover", "buyout", "lbo", "all-cash deal",
        "offer", "bid", "antitrust", "shareholder vote",
    ],
    "Regulation": [
        "regulator", "lawsuit", "antitrust", "doj", "ftc", "sec", "fine", "penalty",
        "ban", "probe", "investigation", "settlement", "compliance",
    ],
    "Tech/AI": [
        "ai", "artificial intelligence", "model", "llm", "chip", "semiconductor",
        "cloud", "data center", "gpu", "training", "inference", "compute",
    ],
    "Geopolitics": [
        "sanction", "tariff", "embargo", "trade war", "conflict", "border",
        "election", "parliament", "coalition", "summit",
    ],
    "Energy": [
        "oil", "crude", "opec", "opec\\+", "gas", "lng", "refinery", "brent", "wti",
        "renewable", "solar", "wind", "nuclear",
    ],
    "Crypto": [
        "bitcoin", "btc", "ethereum", "eth", "stablecoin", "defi", "etf filing",
        "spot etf", "mining", "halving", "wallet", "exchange",
    ],
}

# precompile regexes for speed
def _compile_taxonomy(tax: Dict[str, Iterable[str]]) -> Dict[str, List[re.Pattern]]:
    compiled: Dict[str, List[re.Pattern]] = {}
    for topic, terms in tax.items():
        pats: List[re.Pattern] = []
        for t in terms:
            # word-ish boundaries; allow plurals and punctuation variants via regex terms above
            pat = re.compile(rf"(?i)\b{t}\b")
            pats.append(pat)
        compiled[topic] = pats
    return compiled


# ------------------------- utilities -------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'")
    return s

def _count_hits(text: str, patterns: List[re.Pattern]) -> Tuple[int, List[str], Dict[str, int]]:
    hits: List[str] = []
    counts: Dict[str, int] = {}
    total = 0
    for p in patterns:
        found = p.findall(text)
        if found:
            term = p.pattern  # include pattern string for introspection
            total += len(found)
            counts[term] = len(found)
            # keep a few exemplars
            hits.extend(found[:3])
    return total, hits[:6], counts

def _squash(n: int, scale: float = 3.0) -> float:
    """Map count n to [0,1) with diminishing returns."""
    return 1.0 - math.exp(-n / scale)


# ------------------------- optional ML (sklearn) -------------------------

class _SklearnWrapper:
    """Tiny lazy TF-IDF + logistic regression model (if scikit-learn is present)."""
    def __init__(self, topics: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
        from sklearn.linear_model import LogisticRegression  # noqa
        self.TfidfVectorizer = TfidfVectorizer
        self.LogisticRegression = LogisticRegression
        self.topics = topics
        self.vec = self.TfidfVectorizer(
            lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.9
        )
        # one-vs-rest classifiers per topic
        self.clfs = {t: self.LogisticRegression(max_iter=100) for t in topics}
        self._fitted = False

    def fit(self, docs: List[str], labels: List[List[str]]):
        X = self.vec.fit_transform(docs)
        for t in self.topics:
            y = [1 if t in ls else 0 for ls in labels]
            self.clfs[t].fit(X, y)
        self._fitted = True

    def predict_proba(self, docs: List[str]) -> Dict[str, List[float]]:
        if not self._fitted:
            # not trained → return zeros
            return {t: [0.0]*len(docs) for t in self.topics}
        X = self.vec.transform(docs)
        out = {}
        for t, clf in self.clfs.items():
            prob = clf.predict_proba(X)[:, 1].tolist()
            out[t] = prob
        return out


def _maybe_sklearn(topics: List[str]):
    try:
        import sklearn  # noqa: F401
        return _SklearnWrapper(topics)
    except Exception:
        return None


# ------------------------- public API -------------------------

@dataclass
class TopicScore:
    topic: str
    confidence: float
    evidence: Dict[str, object]


class TopicTagger:
    def __init__(self, taxonomy: Optional[Dict[str, Iterable[str]]] = None):
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY
        self.compiled = _compile_taxonomy(self.taxonomy)
        self.topics = list(self.taxonomy.keys())
        self.ml = _maybe_sklearn(self.topics)  # optional, may be None

    def fit_ml(self, docs: List[str], labels: List[List[str]]):
        """Optional: train the TF-IDF model with your labeled data."""
        if self.ml is None:
            raise RuntimeError("scikit-learn not available")
        self.ml.fit(docs, labels)

    def score(self, text: str) -> List[TopicScore]:
        text_n = _norm(text)
        # 1) rule-based counts
        rb_scores: Dict[str, float] = {}
        ev_hits: Dict[str, List[str]] = {}
        ev_counts: Dict[str, Dict[str,int]] = {}
        for topic, pats in self.compiled.items():
            total, hits, counts = _count_hits(text_n, pats)
            rb_scores[topic] = _squash(total, scale=3.0)  # map counts → 0..1
            ev_hits[topic] = hits
            ev_counts[topic] = counts

        # 2) optional ML probabilities
        ml_scores = {t: 0.0 for t in self.topics}
        if self.ml is not None:
            probs = self.ml.predict_proba([text_n])
            for t in self.topics:
                ml_scores[t] = probs.get(t, [0.0])[0]

        # 3) late fusion
        results: List[TopicScore] = []
        for t in self.topics:
            # weighted blend; rule-based is strong prior (0.65), ML adds nuance (0.35)
            conf = 0.65 * rb_scores[t] + 0.35 * ml_scores[t]
            results.append(TopicScore(
                topic=t,
                confidence=float(min(1.0, max(0.0, conf))),
                evidence={"hits": ev_hits[t], "counts": ev_counts[t], "ml": ml_scores[t]}
            ))
        # sort by confidence desc
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def tag(self, text: str, top_k: int = 3, min_conf: float = 0.15) -> List[TopicScore]:
        scores = self.score(text)
        return [s for s in scores[:top_k] if s.confidence >= min_conf]


# Convenience functions

_singleton: Optional[TopicTagger] = None

def tag_text(text: str, taxonomy: Optional[Dict[str, Iterable[str]]] = None, top_k: int = 3) -> List[Dict[str, object]]:
    """Return top_k topics with confidences and evidence."""
    global _singleton
    if taxonomy or _singleton is None:
        _singleton = TopicTagger(taxonomy)
    out = []
    for s in _singleton.tag(text, top_k=top_k):
        out.append({"topic": s.topic, "confidence": s.confidence, "evidence": s.evidence})
    return out

def tag_article(article: Dict[str, str], top_k: int = 3) -> List[Dict[str, object]]:
    """Article with 'title' and/or 'body' fields."""
    title = article.get("title", "")
    body = article.get("body", "")
    text = f"{title}\n\n{body}".strip()
    return tag_text(text, top_k=top_k)


# ------------------------- self-test -------------------------

if __name__ == "__main__":
    sample = {
        "title": "Fed signals rate cut as inflation cools; stocks rally",
        "body": (
            "U.S. equities gained while Treasury yields fell after the Federal Reserve "
            "signaled a possible rate cut. Energy shares moved with Brent crude. "
            "Bitcoin rose as ETF inflows continued. Microsoft earnings beat expectations."
        ),
    }
    for r in tag_article(sample, top_k=5):
        print(f"{r['topic']:<12}  conf={r['confidence']:.2f}  hits={r['evidence']['hits']}") # type: ignore