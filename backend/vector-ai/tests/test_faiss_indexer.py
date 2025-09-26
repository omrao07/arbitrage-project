# tests/test_faiss_indexer.py
"""
Integration test for FAISS-backed HybridIndexer.

What it does
------------
- Creates a tiny corpus with market/macro texts
- Embeds with a deterministic hashing embedder (no external deps)
- Builds a FAISS IndexFlatIP (cosine via L2-normalized vectors)
- Writes aligned Parquet metadata
- Builds a Whoosh sparse index from the parquet
- Runs search_with_query_vec() and asserts sensible top hits
- Checks filter behavior and RRF vs fusion modes

Requires
--------
pytest
numpy
pandas
pyarrow
faiss-cpu
whoosh

Run
---
pytest -q tests/test_faiss_indexer.py
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None

# Try to import your HybridIndexer
try:
    from analytics_engine.vector_ai.index_builder.hybrid_indexer import HybridIndexer # type: ignore
except Exception:
    # fallback relative path if repo layout differs in local dev
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    try:
        from analytics_engine.vector_ai.index_builder.hybrid_indexer import HybridIndexer # type: ignore
    except Exception as e:
        HybridIndexer = None


pytestmark = [
    pytest.mark.skipif(faiss is None, reason="faiss-cpu not installed"),
    pytest.mark.skipif(HybridIndexer is None, reason="HybridIndexer not importable"),
]


class HashingEmbedder:
    """
    Deterministic, dependency-light text embedder.
    - Tokenizes on simple word chars
    - Hashes tokens into a fixed dim, accumulates counts
    - L2 normalizes the vector (cosine-ready)
    """
    def __init__(self, dim: int = 384):
        self.dim = int(dim)

    def _tokenize(self, text: str):
        import re
        return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]

    def encode(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for tok in self._tokenize(t):
                # stable bucket using Python's hash -> convert to unsigned
                h = hash(tok) & 0xFFFFFFFF
                idx = int(h % self.dim)
                v[idx] += 1.0
            # L2 normalize
            n = float(np.linalg.norm(v) + 1e-9)
            vecs.append((v / n).astype("float32"))
        return vecs


def _build_faiss_index(vectors: np.ndarray, index_path: Path):
    """
    Build an IndexFlatIP (cosine when inputs are normalized) and save to disk.
    """
    d = vectors.shape[1]
    idx = faiss.IndexFlatIP(d) # type: ignore
    idx.add(vectors)
    faiss.write_index(idx, str(index_path)) # type: ignore
    return index_path


def _make_corpus():
    """
    Return a small corpus of (text, meta) rows.
    """
    texts = [
        "The yen carry trade is unwinding as the BOJ signals policy normalization.",
        "Bank of Japan hints at further rate hikes; JGB yields rise and yen strengthens.",
        "Apple launches new products; supply chain sentiment improves.",
        "Carry trades across EM FX face volatility amid rising US real yields.",
        "Nikkei falls as yield curve steepens; exporters under pressure from stronger yen.",
        "Crude oil rallies on OPEC cuts; energy equities outperform.",
        "US CPI surprises to the upside; breakevens widen and USD ticks higher."
    ]
    metas = [
        {"chunk_id": "c1", "doc_id": "d1", "source": "unit", "region": "Asia", "assetClass": "FX"},
        {"chunk_id": "c2", "doc_id": "d1", "source": "unit", "region": "Asia", "assetClass": "Rates"},
        {"chunk_id": "c3", "doc_id": "d2", "source": "unit", "region": "US",   "assetClass": "Equity"},
        {"chunk_id": "c4", "doc_id": "d3", "source": "unit", "region": "EM",   "assetClass": "FX"},
        {"chunk_id": "c5", "doc_id": "d4", "source": "unit", "region": "Asia", "assetClass": "Equity"},
        {"chunk_id": "c6", "doc_id": "d5", "source": "unit", "region": "US",   "assetClass": "Commodities"},
        {"chunk_id": "c7", "doc_id": "d6", "source": "unit", "region": "US",   "assetClass": "Macro"},
    ]
    return texts, metas


def test_hybrid_faiss_end_to_end(tmp_path: Path):
    # Paths
    whoosh_dir = tmp_path / "whoosh"
    parquet_path = tmp_path / "live.parquet"
    faiss_path = tmp_path / "live.faiss"

    # Data
    texts, metas = _make_corpus()
    emb = HashingEmbedder(dim=384)

    # Build dense vectors for corpus
    vecs = np.asarray(emb.encode(texts), dtype="float32")
    assert vecs.shape == (len(texts), 384)

    # Save FAISS
    _build_faiss_index(vecs, faiss_path)

    # Build Parquet metadata aligned to FAISS ids (row index = FAISS id)
    rows = []
    for i, (t, m) in enumerate(zip(texts, metas)):
        r = {"text": t, **m}
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    # Init HybridIndexer
    hy = HybridIndexer(
        whoosh_dir=str(whoosh_dir),
        dense_meta=str(parquet_path),
        dense_faiss=str(faiss_path),
        alpha=0.4,
    ) # type: ignore
    hy.ensure_ready()
    hy.rebuild_sparse_from_parquet()  # BM25 index from parquet

    # Query
    query = "yen carry trade unwind and BOJ yields"
    qv = np.asarray(emb.encode([query])[0], dtype="float32")[None, :]
    # FAISS path uses search_with_query_vec
    hits = hy.search_with_query_vec(qv, query_text=query, top_k=5, mode="fusion")

    # Assertions
    assert len(hits) > 0
    # top 1-2 should be yen/BOJ-related texts
    top_texts = " || ".join([(h.get("text") or "") for h in hits[:2]])
    assert ("yen" in top_texts.lower()) or ("boj" in top_texts.lower())

    # Check that meta fields are present and fused_score exists
    top = hits[0]
    assert "chunk_id" in top and "doc_id" in top and "meta" in top
    assert "fused_score" in top and isinstance(top["fused_score"], float)

    # Filters (client-side) should work
    asia_hits = hy.search_with_query_vec(
        qv, query_text=query, top_k=7, mode="fusion", filters={"region": "Asia"}
    )
    assert all(h.get("meta", {}).get("region") == "Asia" for h in asia_hits)

    # Filters excluding Asia should remove Asia rows
    non_asia_hits = hy.search_with_query_vec(
        qv, query_text=query, top_k=7, mode="fusion", filters={"region": ["US", "EM"]}
    )
    assert all(h.get("meta", {}).get("region") in {"US", "EM"} for h in non_asia_hits)

    # RRF mode smoke test
    hits_rrf = hy.search_with_query_vec(qv, query_text=query, top_k=5, mode="rrf")
    assert len(hits_rrf) > 0
    assert "fused_score" in hits_rrf[0]


def test_returns_empty_on_missing_faiss(tmp_path: Path):
    """
    If FAISS index file is missing/corrupt, HybridIndexer.ensure_ready() will succeed,
    but search_with_query_vec should raise a clean error earlier (ensure_loaded does).
    """
    whoosh_dir = tmp_path / "whoosh"
    parquet_path = tmp_path / "live.parquet"
    faiss_path = tmp_path / "missing.faiss"

    # minimal parquet
    pd.DataFrame([{"chunk_id": "c1", "doc_id": "d1", "source": "unit", "text": "hello"}]).to_parquet(parquet_path, index=False)

    hy = HybridIndexer(
        whoosh_dir=str(whoosh_dir),
        dense_meta=str(parquet_path),
        dense_faiss=str(faiss_path),
        alpha=0.4,
    ) # type: ignore

    # ensure_ready will try to load dense; missing FAISS should error only when dense.ensure_loaded runs.
    with pytest.raises(RuntimeError):
        hy.ensure_ready()