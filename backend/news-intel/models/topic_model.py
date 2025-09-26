# news-intel/models/topic_model.py
"""
TopicModel
----------
Multi-label topic classifier for news text.

Two operating modes:
1) ML mode (preferred, if scikit-learn is installed):
   - TF-IDF (1-2 grams) → One-vs-Rest Logistic Regression
   - .fit(docs, labels), .predict(text), .predict_proba(texts)
   - .save(path) / .load(path)

2) Rule-based fallback (no external deps):
   - Keyword taxonomy compiled to regexes
   - Deterministic confidences based on keyword hits

Labels are arbitrary strings, e.g.: ["Macro", "Earnings", "M&A", "Tech/AI", ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any
import re
import json
import math
import os
import pickle
import unicodedata

# ----------------------- optional deps -----------------------
try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # type: ignore[assignment]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.multiclass import OneVsRestClassifier  # type: ignore
    from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# ----------------------- utilities -----------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u2019", "'")
    return s

def _squash_count(n: int, scale: float = 3.0) -> float:
    """Map integer count → [0,1) with diminishing returns."""
    return 1.0 - math.exp(-n / max(1e-6, scale))


# ----------------------- rule-based backend -----------------------

class _RuleBackend:
    def __init__(self, labels: List[str], taxonomy: Optional[Dict[str, Iterable[str]]] = None):
        self.labels = labels
        if taxonomy is None:
            # minimal defaults; consider importing from your enrich/topic_tagging.DEFAULT_TAXONOMY
            taxonomy = {
                "Markets": ["stocks", "equities", "bonds", "treasury", "nasdaq", "s&p", "dow", "etf", "volatility", "rally", "selloff"],
                "Macro": ["inflation", "cpi", "ppi", "gdp", "recession", "rate hike", "rate cut", "federal reserve", "ecb", "boj", "yield curve"],
                "Earnings": ["earnings", "eps", "revenue", "guidance", "outlook", "beat", "miss", "quarter", "buyback"],
                "M&A": ["merger", "acquisition", "takeover", "buyout", "lbo", "deal", "antitrust", "shareholder vote"],
                "Regulation": ["regulator", "lawsuit", "antitrust", "doj", "ftc", "sec", "fine", "penalty", "probe", "investigation"],
                "Tech/AI": ["ai", "artificial intelligence", "llm", "chip", "semiconductor", "cloud", "gpu", "data center", "compute"],
                "Geopolitics": ["sanction", "tariff", "embargo", "conflict", "border", "election", "parliament", "coalition", "summit"],
                "Energy": ["oil", "crude", "opec", "gas", "brent", "wti", "lng", "refinery", "solar", "wind", "nuclear"],
                "Crypto": ["bitcoin", "btc", "ethereum", "eth", "stablecoin", "defi", "wallet", "exchange", "mining", "halving"],
            }
        # compile
        self.compiled: Dict[str, List[re.Pattern]] = {
            topic: [re.compile(rf"(?i)\b{t}\b") for t in terms]
            for topic, terms in taxonomy.items()
        }

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            text = _norm(text)
            scores: Dict[str, float] = {t: 0.0 for t in self.labels}
            for topic, pats in self.compiled.items():
                # unseen labels get zero score
                if topic not in scores:
                    continue
                count = 0
                for p in pats:
                    count += len(p.findall(text))
                scores[topic] = _squash_count(count, scale=3.0)
            # order by self.labels
            out.append([scores.get(t, 0.0) for t in self.labels])
        return out


# ----------------------- sklearn backend -----------------------

class _SkBackend:
    def __init__(self, labels: Optional[List[str]] = None):
        if not _HAVE_SK:
            raise RuntimeError("scikit-learn not available")
        self.vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95)
        self.clf = OneVsRestClassifier(LogisticRegression(max_iter=200, n_jobs=None))
        self.mlb = MultiLabelBinarizer(classes=labels if labels else None)
        self._fitted = False

    def fit(self, docs: List[str], labels: List[List[str]]) -> None:
        y = self.mlb.fit_transform(labels)
        X = self.vec.fit_transform(docs)
        self.clf.fit(X, y)
        self._fitted = True

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            return [[0.0]*len(self.mlb.classes_)]*len(texts)
        X = self.vec.transform(texts)
        # LogisticRegression has predict_proba; OneVsRest wraps it
        probas = self.clf.predict_proba(X)  # shape [n_samples, n_classes]
        return probas.tolist()

    def labels(self) -> List[str]:
        return list(self.mlb.classes_)

    # persistence
    def save(self, path: str) -> None:
        obj = {"vec": self.vec, "clf": self.clf, "mlb": self.mlb}
        if joblib is not None:
            joblib.dump(obj, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(obj, f)

    @staticmethod
    def load(path: str) -> "_SkBackend":
        if joblib is not None:
            obj = joblib.load(path)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        backend = _SkBackend()
        backend.vec = obj["vec"]
        backend.clf = obj["clf"]
        backend.mlb = obj["mlb"]
        backend._fitted = True
        return backend


# ----------------------- public facade -----------------------

@dataclass
class TopicScore:
    topic: str
    confidence: float

class TopicModel:
    """
    Multi-label topic classifier.

    If scikit-learn is available:
      - call .fit(docs, labels) to train
      - call .predict(text, top_k) / .predict_batch(texts)
      - .save(path) / .load(path)

    If not, rule-based fallback uses a taxonomy (keywords).
    """

    def __init__(self, labels: Optional[List[str]] = None, taxonomy: Optional[Dict[str, Iterable[str]]] = None):
        self._use_ml = _HAVE_SK
        self._labels: List[str] = labels or []
        self._taxonomy = taxonomy
        self._sk: Optional[_SkBackend] = None
        self._rb: Optional[_RuleBackend] = None

        if self._use_ml:
            self._sk = _SkBackend(labels=self._labels if self._labels else None)
        # rule backend is always available as a fallback scorer
        self._rb = _RuleBackend(labels=self._labels or self.default_labels(), taxonomy=taxonomy)

    # --------- defaults ---------
    @staticmethod
    def default_labels() -> List[str]:
        return ["Markets", "Macro", "Earnings", "M&A", "Regulation", "Tech/AI", "Geopolitics", "Energy", "Crypto"]

    # --------- training (ML only) ---------
    def fit(self, docs: List[str], labels: List[List[str]]) -> None:
        """
        Train the ML backend. If sklearn is unavailable, noop (rule-based has nothing to fit).
        """
        if not self._use_ml:
            # accept call for API compatibility
            # ensure labels set drives the rule-backend label order
            lbls = sorted({t for row in labels for t in row}) or self.default_labels()
            self._labels = lbls
            self._rb = _RuleBackend(labels=lbls, taxonomy=self._taxonomy)
            return
        assert self._sk is not None
        docs = [_norm(d) for d in docs]
        self._sk.fit(docs, labels)
        self._labels = self._sk.labels()
        # keep rule-backend label order consistent
        self._rb = _RuleBackend(labels=self._labels, taxonomy=self._taxonomy)

    # --------- inference ---------
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        texts_n = [_norm(t) for t in texts]
        if self._use_ml and self._sk is not None:
            probs = self._sk.predict_proba(texts_n)
        else:
            probs = self._rb.predict_proba(texts_n)  # type: ignore
        return probs

    def predict(self, text: str, top_k: int = 3, min_conf: float = 0.15) -> List[TopicScore]:
        probs = self.predict_proba([text])[0]
        labels = self.labels()
        pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        out = [TopicScore(topic=l, confidence=float(p)) for l, p in pairs[:top_k] if p >= min_conf]
        return out

    def predict_batch(self, texts: List[str], top_k: int = 3, min_conf: float = 0.15) -> List[List[TopicScore]]:
        prob_list = self.predict_proba(texts)
        labels = self.labels()
        all_out: List[List[TopicScore]] = []
        for probs in prob_list:
            pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
            out = [TopicScore(topic=l, confidence=float(p)) for l, p in pairs[:top_k] if p >= min_conf]
            all_out.append(out)
        return all_out

    def labels(self) -> List[str]:
        if self._use_ml and self._sk is not None and getattr(self._sk, "_fitted", False):
            return self._sk.labels()
        if self._labels:
            return self._labels
        return self.default_labels()

    # --------- persistence (ML only) ---------
    def save(self, path: str) -> None:
        """
        Save ML model to `path`. If running rule-based only, we save a tiny json stub.
        """
        if self._use_ml and self._sk is not None and getattr(self._sk, "_fitted", False):
            self._sk.save(path)
        else:
            stub = {"mode": "rule", "labels": self.labels()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(stub, f)

    @classmethod
    def load(cls, path: str) -> "TopicModel":
        """
        Load model. If file is a pickle/joblib (ML), we restore ML backend.
        If it's a JSON stub (rule), we restore labels only.
        """
        # try json stub first
        try:
            with open(path, "r", encoding="utf-8") as f:
                stub = json.load(f)
            tm = cls(labels=stub.get("labels") or cls.default_labels())
            # force rule mode even if sklearn is available
            tm._use_ml = False
            return tm
        except Exception:
            pass

        # try ML artifact
        if not _HAVE_SK:
            raise RuntimeError("Cannot load ML TopicModel: scikit-learn not available")
        sk = _SkBackend.load(path)
        tm = cls(labels=sk.labels())
        tm._sk = sk
        tm._use_ml = True
        # keep rule-backend aligned for fallback / hybrid use
        tm._rb = _RuleBackend(labels=tm.labels())
        return tm


# ----------------------- self-test -----------------------

if __name__ == "__main__":
    # Rule-only quick test
    tm = TopicModel()
    print([t.__dict__ for t in tm.predict("Fed signals rate cut; stocks rally as yields fall", top_k=5)])

    if _HAVE_SK:
        # Tiny ML demo
        docs = [
            "Fed raises rates, inflation cools in CPI report",
            "Company beats earnings, raises guidance for next quarter",
            "Oil prices surge as OPEC cuts supply",
            "Bitcoin ETF inflows boost crypto markets",
            "Regulator files antitrust lawsuit against tech giant",
            "Mega merger announced in all-cash acquisition deal",
        ]
        labels = [
            ["Macro"], ["Earnings"], ["Energy"], ["Crypto"], ["Regulation"], ["M&A"]
        ]
        tm.fit(docs, labels)
        print(tm.labels())
        print([t.__dict__ for t in tm.predict("Tech giant faces antitrust probe; shares slump", top_k=4)])