# tests/test_retriever.py
"""
Unit tests for search/retriever.py

Covers:
- VectorRetriever (matrix backend) end-to-end with a deterministic embedder
- Optional FAISS path smoke test (skipped if faiss not installed)
- LexicalRetriever using a Whoosh index (skipped if whoosh not installed)
- HybridRetriever blending vector + lexical and handling filters
- Edge cases (missing embedder / missing indexes)

Run:
  pytest -q tests/test_retriever.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import numpy as np
import pytest

from search.retriever import VectorRetriever, LexicalRetriever, HybridRetriever
from search.filters import Filter

# Optional deps for specific tests
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from whoosh import index as windex  # type: ignore
    from whoosh.fields import Schema, TEXT, ID, STORED  # type: ignore
    from whoosh.analysis import StemmingAnalyzer  # type: ignore
except Exception:
    windex = None
    Schema = None
    TEXT = None
    ID = None
    STORED = None
    StemmingAnalyzer = None


# ----------------------------- Test Utilities -----------------------------

class HashingEmbedder:
    """
    Deterministic, dependency-light text embedder.
    - tokenizes on word chars
    - hashes tokens into fixed buckets
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


def make_corpus():
    texts = [
        "Yen carry trade unwinds as BOJ signals tightening.",
        "Japan equities fall as yields rise; exporters pressured.",
        "US inflation surprises to upside, yields spike.",
        "Apple launches new iPhone; supply chain optimistic.",
        "Crude oil rallies as OPEC cuts supply."
    ]
    metas = [
        {"id": "c1", "region": "Asia", "assetClass": "FX"},
        {"id": "c2", "region": "Asia", "assetClass": "Equity"},
        {"id": "c3", "region": "US",   "assetClass": "Rates"},
        {"id": "c4", "region": "US",   "assetClass": "Equity"},
        {"id": "c5", "region": "EM",   "assetClass": "Commodities"},
    ]
    return texts, metas


def build_whoosh_index(tmp_dir: Path, docs):
    analyzer = StemmingAnalyzer() if StemmingAnalyzer else None
    schema = Schema(#type:ignore
        id=ID(stored=True, unique=True), # type: ignore
        text=TEXT(stored=True, analyzer=analyzer), # type: ignore
        region=ID(stored=True), # type: ignore
        assetClass=ID(stored=True) # type: ignore
    )
    ixdir = tmp_dir / "whoosh"
    ixdir.mkdir(parents=True, exist_ok=True)
    ix = windex.create_in(ixdir, schema) # type: ignore
    writer = ix.writer()
    for doc in docs:
        writer.add_document(
            id=doc["id"],
            text=doc["text"],
            region=doc["region"],
            assetClass=doc["assetClass"],
        )
    writer.commit()
    return ix


# --------------------------------- Tests ----------------------------------

def test_vector_matrix_backend_orders_relevant_first():
    emb = HashingEmbedder(dim=128)
    texts, metas = make_corpus()
    vecs = np.asarray(emb.encode(texts), dtype="float32")  # normalized

    vr = VectorRetriever(backend="matrix", embedder=emb)
    vr.add_index(vecs)  # rows align to texts indices

    q = "yen carry trade and BOJ"
    hits = vr.search(q, top_k=3, filters=None)
    assert hits and len(hits) <= 3
    # matrix backend returns meta={"id": idx}; ensure top hit corresponds to the most relevant doc (index 0 likely)
    top_ids = [h.meta.get("id") for h in hits]
    assert isinstance(top_ids[0], int)
    # map back to text to sanity-check relevance
    top_text = texts[top_ids[0]]
    assert ("yen" in top_text.lower()) or ("boj" in top_text.lower())


@pytest.mark.skipif(faiss is None, reason="faiss-cpu not installed")
def test_vector_faiss_backend_smoke():
    emb = HashingEmbedder(dim=64)
    texts, metas = make_corpus()
    X = np.asarray(emb.encode(texts), dtype="float32")
    idx = faiss.IndexFlatIP(X.shape[1]) # type: ignore
    idx.add(X)

    vr = VectorRetriever(backend="faiss", embedder=emb)
    vr.add_index(idx)

    q = "iphone launch supply chain"
    hits = vr.search(q, top_k=2, filters=None)
    assert hits and len(hits) == 2
    # We don't have text in FAISS path; just ensure ids are valid range and scores present
    assert all(isinstance(h.meta.get("id"), int) for h in hits)
    assert all(isinstance(h.base_score, float) for h in hits)


@pytest.mark.skipif(windex is None, reason="whoosh not installed")
def test_lexical_whoosh_retrieves_keywords(tmp_path: Path):
    texts, metas = make_corpus()
    docs = [{"id": m["id"], "text": t, **m} for t, m in zip(texts, metas)]
    ix = build_whoosh_index(tmp_path, docs)

    lr = LexicalRetriever(ix=ix, field="text")
    hits = lr.search("yen carry trade", top_k=3)
    assert hits and len(hits) <= 3
    assert "yen" in hits[0].text.lower() or "carry" in hits[0].text.lower()


@pytest.mark.skipif(windex is None, reason="whoosh not installed")
def test_hybrid_blends_vector_and_lexical(tmp_path: Path):
    # Build vector (matrix) side
    emb = HashingEmbedder(dim=128)
    texts, metas = make_corpus()
    X = np.asarray(emb.encode(texts), dtype="float32")
    vr = VectorRetriever(backend="matrix", embedder=emb)
    vr.add_index(X)

    # Build lexical side
    docs = [{"id": m["id"], "text": t, **m} for t, m in zip(texts, metas)]
    ix = build_whoosh_index(tmp_path, docs)
    lr = LexicalRetriever(ix=ix, field="text")

    hy = HybridRetriever(vector=vr, lexical=lr)
    q = "yen carry trade and BOJ"
    hits = hy.search(q, top_k=5, alpha=0.6, filters=Filter.from_dict({"region": "Asia"}))
    assert hits and len(hits) <= 5
    # Top results should be relevant to the query
    joined = " ".join([h.text for h in hits if h.text])
    assert any(k in joined.lower() for k in ["yen", "carry", "boj"])


def test_vector_retriever_without_embedder_raises():
    vr = VectorRetriever(backend="matrix", embedder=None)
    with pytest.raises(ValueError):
        vr.search("test", top_k=5)


def test_lexical_without_index_returns_empty():
    lr = LexicalRetriever(ix=None, field="text")
    res = lr.search("anything", top_k=5)
    assert res == []


def test_hybrid_handles_empty_components_gracefully():
    # No lexical, only vector (matrix)
    emb = HashingEmbedder(dim=32)
    texts, metas = make_corpus()
    X = np.asarray(emb.encode(texts), dtype="float32")
    vr = VectorRetriever(backend="matrix", embedder=emb)
    vr.add_index(X)

    hy = HybridRetriever(vector=vr, lexical=None)
    hits = hy.search("iphone supply chain", top_k=3, alpha=0.5, filters=None)
    assert hits and len(hits) <= 3

    # No vector, only lexical (skip if whoosh missing)
    if windex is not None:
        tmp = Path(tempfile.mkdtemp())
        try:
            docs = [{"id": m["id"], "text": t, **m} for t, m in zip(texts, metas)]
            ix = build_whoosh_index(tmp, docs)
            hy2 = HybridRetriever(vector=None, lexical=LexicalRetriever(ix=ix, field="text"))
            hits2 = hy2.search("yen carry", top_k=3, alpha=0.5, filters=None)
            assert hits2 and len(hits2) <= 3
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_filters_object_passes_through_without_crashing_matrix_backend():
    # Matrix backend ignores server-side filters, but should not crash when provided
    emb = HashingEmbedder(dim=64)
    texts, metas = make_corpus()
    X = np.asarray(emb.encode(texts), dtype="float32")
    vr = VectorRetriever(backend="matrix", embedder=emb)
    vr.add_index(X)

    flt = Filter.from_dict({"region": {"in": ["US", "Asia"]}})
    res = vr.search("yields spike", top_k=4, filters=flt)  # should not error
    assert isinstance(res, list)