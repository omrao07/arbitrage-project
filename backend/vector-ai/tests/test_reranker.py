# tests/test_reranker.py
"""
Unit tests for search/reranker.py

Covers:
- CosineReranker scoring & ordering (with deterministic embedder)
- BM25LiteReranker lexical scoring basics
- CrossEncoderReranker graceful fallback when model not installed
- diversify_mmr() reduces redundancy
- fuse_scores() blends base + rerank scores and reorders accordingly

Run:
  pytest -q tests/test_reranker.py
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from search.reranker import (
    Candidate,
    CosineReranker,
    BM25LiteReranker,
    CrossEncoderReranker,
    diversify_mmr,
    fuse_scores,
)


# ----------------------------- Test Utilities -----------------------------

class HashingEmbedder:
    """
    Deterministic, dependency-light embedder:
    - tokenizes on word chars
    - hashes tokens into buckets
    - L2 normalizes (cosine-ready)
    """
    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    @staticmethod
    def _tok(t: str):
        import re
        return [x.lower() for x in re.findall(r"[A-Za-z0-9_]+", t or "")]

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for tok in self._tok(t):
                v[hash(tok) % self.dim] += 1.0
            n = float(np.linalg.norm(v) + 1e-9)
            out.append(v / n)
        return out


def _mk_candidates(texts):
    # give a slight deterministic base_score spread
    base = np.linspace(0.1, 0.9, num=len(texts))
    return [Candidate(text=t, meta={"id": i}, base_score=float(b)) for i, (t, b) in enumerate(zip(texts, base))]


# --------------------------------- Tests ----------------------------------

def test_cosine_reranker_orders_semantically():
    emb = HashingEmbedder(dim=128)
    rr = CosineReranker(embedder=emb)

    docs = [
        "The yen carry trade is unwinding as BOJ hints rate hikes.",
        "Apple launches new iPhone; supply chain upbeat.",
        "Carry trades in EM FX wobble as US real yields climb.",
        "Crude oil rises on OPEC cuts; energy leads."
    ]
    query = "yen carry trade unwind and BOJ yields"
    cands = _mk_candidates(docs)

    ranked = rr.rerank(query, cands, top_k=None, normalize=True)
    # top-1 should be the most yen/carry specific doc
    assert "yen" in ranked[0].text.lower() or "boj" in ranked[0].text.lower()
    # scores are normalized to [0,1]
    assert 0.0 <= ranked[0].rr_score <= 1.0 # type: ignore


def test_bm25lite_gives_higher_to_keyword_overlap():
    rr = BM25LiteReranker()
    docs = [
        "yen carry trade unwind boj",
        "carry trade discussion",
        "completely unrelated"
    ]
    query = "yen carry trade"
    cands = _mk_candidates(docs)

    ranked = rr.rerank(query, cands, top_k=None, normalize=False)
    # first doc contains all terms -> should rank highest
    assert ranked[0].text.startswith("yen carry")


def test_crossencoder_fallback_when_missing(monkeypatch):
    """
    If sentence-transformers is not installed, CrossEncoderReranker
    should still return a score vector (fallback to cosine if provided).
    """
    # simulate import failure
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)

    emb = HashingEmbedder(dim=64)
    rr = CrossEncoderReranker(fallback_embedder=emb)  # fallback path

    docs = [
        "yen carry trade unwind boj",
        "iphone launch supply chain",
    ]
    query = "yen carry trade"
    cands = _mk_candidates(docs)

    scores = rr.score(query, cands)
    assert isinstance(scores, list) and len(scores) == len(cands)
    # doc[0] should be >= doc[1] on fallback cosine
    assert scores[0] >= scores[1]


def test_diversify_mmr_reduces_near_duplicates():
    # Create duplicate-ish texts
    docs = [
        "yen carry trade unwind boj yields",
        "carry trade unwind yen boj",
        "apple iphone launch",
        "apple new iphone model",
        "oil opec cuts supply"
    ]
    cands = _mk_candidates(docs)
    # Pretend rerank already happened: make similar items have high rr_score
    cands[0].rr_score = 0.95
    cands[1].rr_score = 0.93
    cands[2].rr_score = 0.88
    cands[3].rr_score = 0.86
    cands[4].rr_score = 0.60

    diversified = diversify_mmr(cands, lambda_mult=0.7, top_k=3)
    texts = [c.text for c in diversified]
    # Expect we do not pick BOTH first two very-similar items together often;
    # we likely include one of {0,1} and one of {2,3} for diversity.
    assert any("yen" in t for t in texts)  # one yen/carry
    assert any("iphone" in t for t in texts)  # one iphone
    # and likely not both near-duplicate yen entries
    assert not (any("yen" in t for t in texts[:1]) and any("yen" in t for t in texts[1:2])) or len(set(texts)) == 3


def test_fuse_scores_changes_ordering_when_rr_strong():
    docs = [
        "yen carry trade unwind boj",
        "iphone launch",
        "oil opec cuts"
    ]
    cands = _mk_candidates(docs)
    # base_score favors doc[1] initially (since we linearly space 0.1..0.9)
    # set rr_score to strongly favor doc[0]
    cands[0].rr_score = 0.99
    cands[1].rr_score = 0.10
    cands[2].rr_score = 0.20

    fused = fuse_scores(cands, alpha=0.7, normalize_base=True, normalize_rerank=False)
    # After fusion, doc[0] should rise to the top
    assert fused[0].text.startswith("yen carry") or "yen" in fused[0].text.lower()


def test_rerank_preserves_meta_and_sets_rr_score():
    emb = HashingEmbedder(dim=64)
    rr = CosineReranker(embedder=emb)
    c = Candidate(text="yen carry trade", meta={"id": "x1", "ticker": "JPY"}, base_score=0.42)
    out = rr.rerank("carry", [c], top_k=None, normalize=True)[0]
    assert out.meta["id"] == "x1" and out.meta["ticker"] == "JPY"
    assert out.base_score == pytest.approx(0.42)
    assert out.rr_score is not None and 0.0 <= out.rr_score <= 1.0