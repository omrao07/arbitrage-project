# tests/test_hybrid_indexer.py
"""
Integration tests for HybridIndexer (dense + sparse hybrid search).

Covers:
- Building FAISS index + Parquet metadata
- Rebuilding Whoosh index from Parquet
- Running queries in "fusion" and "rrf" modes
- Verifying filters (region, assetClass, etc.)
- Ensuring metadata round-trips properly

Requires:
    pytest
    numpy
    pandas
    pyarrow
    faiss-cpu
    whoosh
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import faiss # type: ignore
except Exception:
    faiss = None

try:
    from analytics_engine.vector_ai.index_builder.hybrid_indexer import HybridIndexer # type: ignore
except Exception:
    HybridIndexer = None


pytestmark = [
    pytest.mark.skipif(faiss is None, reason="faiss-cpu not installed"),
    pytest.mark.skipif(HybridIndexer is None, reason="HybridIndexer not importable"),
]


class DummyEmbedder:
    """
    Deterministic embedder for test: simple hash-based vectorizer.
    """
    def __init__(self, dim=128):
        self.dim = dim

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for tok in t.lower().split():
                idx = hash(tok) % self.dim
                v[idx] += 1.0
            n = np.linalg.norm(v) + 1e-9
            out.append(v / n)
        return out


def _make_corpus():
    texts = [
        "Yen carry trade unwinds as BOJ signals tightening.",
        "Japan equities fall as yields rise; exporters pressured.",
        "US inflation surprises to upside, yields spike.",
        "Apple launches new iPhone; supply chain optimistic.",
        "Crude oil rallies as OPEC cuts supply.",
    ]
    metas = [
        {"chunk_id": "c1", "doc_id": "d1", "region": "Asia", "assetClass": "FX"},
        {"chunk_id": "c2", "doc_id": "d2", "region": "Asia", "assetClass": "Equity"},
        {"chunk_id": "c3", "doc_id": "d3", "region": "US",   "assetClass": "Rates"},
        {"chunk_id": "c4", "doc_id": "d4", "region": "US",   "assetClass": "Equity"},
        {"chunk_id": "c5", "doc_id": "d5", "region": "EM",   "assetClass": "Commodities"},
    ]
    return texts, metas


def _build_faiss_index(vecs: np.ndarray, path: Path):
    idx = faiss.IndexFlatIP(vecs.shape[1]) # type: ignore
    idx.add(vecs)
    faiss.write_index(idx, str(path)) # type: ignore
    return path


def test_hybrid_indexer_end_to_end(tmp_path: Path):
    whoosh_dir = tmp_path / "whoosh"
    parquet_path = tmp_path / "meta.parquet"
    faiss_path = tmp_path / "vecs.faiss"

    texts, metas = _make_corpus()
    emb = DummyEmbedder(dim=128)

    # build dense vectors
    vecs = np.asarray(emb.encode(texts), dtype="float32")
    _build_faiss_index(vecs, faiss_path)

    # parquet metadata
    rows = []
    for t, m in zip(texts, metas):
        rows.append({"text": t, **m})
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    # init
    hy = HybridIndexer(
        whoosh_dir=str(whoosh_dir),
        dense_meta=str(parquet_path),
        dense_faiss=str(faiss_path),
        alpha=0.5,
    ) # type: ignore
    hy.ensure_ready()
    hy.rebuild_sparse_from_parquet()

    # query
    query = "yen carry trade and BOJ"
    qv = np.asarray(emb.encode([query])[0], dtype="float32")[None, :]
    hits = hy.search_with_query_vec(qv, query_text=query, top_k=3, mode="fusion")

    assert len(hits) > 0
    top_texts = " ".join(h["text"] for h in hits[:2])
    assert "yen" in top_texts.lower() or "boj" in top_texts.lower()

    # check meta roundtrip
    top = hits[0]
    assert "chunk_id" in top and "doc_id" in top
    assert "fused_score" in top

    # filters: region=Asia
    asia_hits = hy.search_with_query_vec(qv, query_text=query, top_k=5, mode="fusion", filters={"region": "Asia"})
    assert all(h["meta"]["region"] == "Asia" for h in asia_hits)

    # rrf mode smoke test
    hits_rrf = hy.search_with_query_vec(qv, query_text=query, top_k=3, mode="rrf")
    assert len(hits_rrf) > 0
    assert "fused_score" in hits_rrf[0]


def test_missing_faiss_file(tmp_path: Path):
    whoosh_dir = tmp_path / "whoosh"
    parquet_path = tmp_path / "meta.parquet"
    faiss_path = tmp_path / "doesnotexist.faiss"

    # minimal parquet
    pd.DataFrame([{"chunk_id": "c1", "doc_id": "d1", "text": "hello"}]).to_parquet(parquet_path, index=False)

    hy = HybridIndexer(
        whoosh_dir=str(whoosh_dir),
        dense_meta=str(parquet_path),
        dense_faiss=str(faiss_path),
    ) # type: ignore

    with pytest.raises(RuntimeError):
        hy.ensure_ready()