# search/reranker.py
"""
Rerankers for search pipelines.

Features
--------
- Candidate dataclass with text + metadata + base_score
- BaseReranker interface + three implementations:
    1) CosineReranker (embedding-based, fast; needs an embedder)
    2) BM25LiteReranker (pure-Python token TF/IDF; no external deps)
    3) CrossEncoderReranker (SOTA quality; uses sentence-transformers CrossEncoder)
- MMR diversify() to reduce redundancy
- fuse_scores() to blend base ANN/BM25 scores with reranker scores
- Safe fallbacks: if CrossEncoder is unavailable, it politely explains via warnings

Install (optional)
------------------
pip install sentence-transformers  # for CrossEncoder
pip install numpy
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ============================== Model Types ==============================

@dataclass
class Candidate:
    """
    One search hit prior to reranking.
    - text: the full snippet/passage to score
    - meta: any metadata (ids, source, tags, etc.)
    - base_score: original retriever score (e.g., ANN similarity, BM25 score)
    """
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    base_score: float = 0.0
    # containers for downstream scores
    rr_score: Optional[float] = None
    fused: Optional[float] = None


class BaseReranker:
    def score(self, query: str, candidates: Sequence[Candidate]) -> List[float]:
        """Return rerank scores (higher is better) aligned to candidates."""
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: Optional[int] = None,
        normalize: bool = True,
    ) -> List[Candidate]:
        scores = self.score(query, candidates)
        out = []
        for c, s in zip(candidates, scores):
            cc = Candidate(text=c.text, meta=dict(c.meta), base_score=c.base_score)
            cc.rr_score = float(s)
            out.append(cc)
        # normalize reranker scores to [0,1] for easier fusion
        if normalize and out:
            vals = np.asarray([x.rr_score for x in out], dtype=float)
            vals = _minmax(vals)
            for i, v in enumerate(vals):
                out[i].rr_score = float(v)
        out.sort(key=lambda x: x.rr_score if x.rr_score is not None else -1e9, reverse=True)
        return out[:top_k] if (top_k is not None) else out


# ============================== Utilities ===============================

def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / n

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


# ============================== Cosine Reranker =========================

class CosineReranker(BaseReranker):
    """
    Fast reranker using the same (or a compact) embedder as your dense retriever.
    Provide an embedder with:
        .encode(list[str]) -> List[List[float]] (or np.ndarray)
        .dim  (optional)
    """
    def __init__(self, embedder, normalize_embeddings: bool = True):
        self.embedder = embedder
        self.normalize_embeddings = normalize_embeddings

    def score(self, query: str, candidates: Sequence[Candidate]) -> List[float]:
        texts = [c.text for c in candidates]
        qv = np.asarray(self.embedder.encode([query])[0], dtype="float32")
        D = qv.shape[0]
        cv = np.asarray(self.embedder.encode(texts), dtype="float32")
        if self.normalize_embeddings:
            qv = _l2norm(qv[None, :])[0]
            cv = _l2norm(cv)
        # cosine = dot when normalized
        scores = (cv @ qv).astype("float32")
        return scores.tolist()


# ============================== BM25 Lite Reranker ======================

class BM25LiteReranker(BaseReranker):
    """
    Tiny BM25-ish reranker (no external deps).
    Good as a lexical tie-breaker on top of ANN results.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)

    def score(self, query: str, candidates: Sequence[Candidate]) -> List[float]:
        q_terms = _tokenize(query)
        if not candidates or not q_terms:
            return [0.0] * len(candidates)

        # Build simple DF + length stats for current batch
        docs = [ _tokenize(c.text) for c in candidates ]
        N = len(docs)
        lens = np.asarray([len(d) for d in docs], dtype=float)
        avgdl = float(lens.mean() if N else 0.0)

        # document frequencies
        df: Dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1

        # idf
        idf: Dict[str, float] = {}
        for t in set(q_terms):
            # +0.5 smoothing to avoid negatives for very frequent terms
            idf[t] = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)

        scores: List[float] = []
        for d, L in zip(docs, lens):
            tf: Dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                denom = tf[t] + self.k1 * (1 - self.b + self.b * (L / (avgdl + 1e-9)))
                s += idf.get(t, 0.0) * (tf[t] * (self.k1 + 1)) / (denom + 1e-9)
            scores.append(s)
        return scores


# ============================== CrossEncoder Reranker ===================

class CrossEncoderReranker(BaseReranker):
    """
    High-accuracy reranker using a classification/regression CrossEncoder.
    Defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2' if available.

    Note: Requires sentence-transformers. If unavailable, we warn and fall back
    to a cosine reranker (if a fallback embedder is provided), else zeros.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        normalize_to_0_1: bool = True,
        fallback_embedder: Any = None,
    ):
        self.model_name = model_name
        self.normalize_to_0_1 = normalize_to_0_1
        self.fallback_embedder = fallback_embedder
        self._model = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(model_name, max_length=512)
        except Exception as e:
            warnings.warn(
                f"[CrossEncoderReranker] Could not load '{model_name}': {e}. "
                "Falling back (cosine or zeros). Install: pip install sentence-transformers"
            )

    def score(self, query: str, candidates: Sequence[Candidate]) -> List[float]:
        if not candidates:
            return []

        if self._model is not None:
            pairs = [(query, c.text) for c in candidates]
            scores = self._model.predict(pairs, convert_to_numpy=True).tolist()
            # Many CE models output in arbitrary ranges; normalize if requested
            if self.normalize_to_0_1:
                arr = _minmax(np.asarray(scores, dtype=float))
                return arr.tolist()
            return [float(s) for s in scores]

        # Fallbacks
        if self.fallback_embedder is not None:
            return CosineReranker(self.fallback_embedder).score(query, candidates)
        # No model & no embedder: zeros (keeps original order)
        return [0.0] * len(candidates)


# ============================== Diversification ========================

def diversify_mmr(
    candidates: Sequence[Candidate],
    lambda_mult: float = 0.7,
    top_k: Optional[int] = None,
    sim_fn: Optional[Any] = None,
) -> List[Candidate]:
    """
    Maximal Marginal Relevance selection to reduce redundancy.
    Expects candidates to have `rr_score` (or `base_score` if rr_score missing).
    sim_fn(a: Candidate, b: Candidate) -> similarity in [0,1].
    If sim_fn is None, we approximate similarity by token Jaccard.
    """
    if not candidates:
        return []

    def _sim(a: Candidate, b: Candidate) -> float:
        if sim_fn:
            return float(sim_fn(a, b))
        ta, tb = set(_tokenize(a.text)), set(_tokenize(b.text))
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb); union = len(ta | tb)
        return inter / union

    scores = [ (c.rr_score if c.rr_score is not None else c.base_score) or 0.0 for c in candidates ]
    picked: List[int] = []
    cand_idx = list(range(len(candidates)))
    # seed with best score
    cur = int(np.argmax(scores))
    picked.append(cur)
    remaining = [i for i in cand_idx if i != cur]

    while remaining and (top_k is None or len(picked) < top_k):
        best_i = None
        best_val = -1e9
        for i in remaining:
            rel = scores[i]
            div = max(_sim(candidates[i], candidates[j]) for j in picked) if picked else 0.0
            val = lambda_mult * rel - (1 - lambda_mult) * div
            if val > best_val:
                best_val = val
                best_i = i
        picked.append(best_i) # type: ignore
        remaining.remove(best_i) # type: ignore

    return [candidates[i] for i in picked[: (top_k or len(picked))]]


# ============================== Fusion =================================

def fuse_scores(
    candidates: Sequence[Candidate],
    alpha: float = 0.5,
    normalize_base: bool = True,
    normalize_rerank: bool = False,  # usually already normalized in rerankers
) -> List[Candidate]:
    """
    Compute fused score = alpha*rr_score + (1-alpha)*base_score (with optional normalization).
    Returns a new sorted list (desc).
    """
    if not candidates:
        return []
    base = np.asarray([c.base_score for c in candidates], dtype=float)
    rr   = np.asarray([c.rr_score if c.rr_score is not None else 0.0 for c in candidates], dtype=float)

    if normalize_base:
        base = _minmax(base)
    if normalize_rerank:
        rr = _minmax(rr)

    fused = alpha * rr + (1.0 - alpha) * base
    out: List[Candidate] = []
    for c, f in zip(candidates, fused):
        cc = Candidate(text=c.text, meta=dict(c.meta), base_score=float(c.base_score))
        cc.rr_score = c.rr_score
        cc.fused = float(f)
        out.append(cc)
    out.sort(key=lambda x: x.fused if x.fused is not None else -1e9, reverse=True)
    return out


# ============================== Example CLI ============================

if __name__ == "__main__":
    # Toy example showing how to use the rerankers + MMR
    docs = [
        "The yen carry trade is unwinding as BOJ hints at normalization.",
        "BOJ policy shift impacts JGB yields; yen strengthens versus USD.",
        "Apple releases new iPhone models; supply chain upbeat.",
        "Carry trades across EM FX see volatility amid US rates move.",
        "Japanese equities fall as yield curve steepens; exporters hit."
    ]
    cands = [Candidate(text=t, meta={"id": i}, base_score=np.random.rand()) for i, t in enumerate(docs)]
    query = "yen carry trade unwind and BOJ yields"

    # Cosine reranker with dummy embedder
    class _DummyEmbedder:
        dim = 384
        def encode(self, texts):
            # For demo: deterministic bag-of-words random-ish
            rng = np.random.default_rng(7)
            arr = rng.normal(size=(len(texts), self.dim)).astype("float32")
            return (arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)).tolist()

    cosine_rr = CosineReranker(_DummyEmbedder())
    reranked = cosine_rr.rerank(query, cands, top_k=None, normalize=True)

    diversified = diversify_mmr(reranked, lambda_mult=0.7, top_k=3)
    fused = fuse_scores(diversified, alpha=0.6)

    print("Top (MMR+fused):")
    for r in fused[:3]:
        print(f"- id={r.meta.get('id')} rr={r.rr_score:.3f} base={r.base_score:.3f} fused={r.fused:.3f} :: {r.text[:70]}")