# tests/test_pinecone_indexer.py
"""
Offline tests for the Pinecone retrieval path using a local fake index.

Why a fake?
-----------
We don't want network calls in CI. This test provides a lightweight, fully
in-memory `FakePineconeIndex` that mimics the .upsert(...) and .query(...)
surface we use in VectorRetriever (search/retriever.py).

It validates:
- Vector search ordering (cosine on normalized vectors)
- Filter handling via Filter.to_pinecone()
- Metadata round-trip

Requires:
  pytest
  numpy

Run:
  pytest -q tests/test_pinecone_indexer.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from search.retriever import VectorRetriever
from search.filters import Filter


# ---------------------------- Test Utilities ----------------------------

class HashingEmbedder:
    """
    Deterministic, dependency-light text embedder.
    - Hash tokens into fixed buckets
    - L2 normalize -> cosine ready
    """
    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    @staticmethod
    def _tok(text: str) -> List[str]:
        import re
        return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]

    def encode(self, texts: List[str]):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype="float32")
            for tok in self._tok(t):
                v[hash(tok) % self.dim] += 1.0
            n = float(np.linalg.norm(v) + 1e-9)
            out.append((v / n).astype("float32"))
        return out


# ------------------------ Fake Pinecone Interface -----------------------

class FakePineconeIndex:
    """
    Minimal in-memory stand-in for pinecone.Index used by VectorRetriever.

    Supports:
      - upsert(vectors=[{"id": str, "values": [float], "metadata": {...}}])
      - query(vector=[float], top_k=int, filter=dict)
        -> returns {"matches":[{"id":..., "score":..., "text":..., ...}]}
    Assumptions:
      - cosine similarity on *already normalized* vectors
      - metadata includes "text" (the retriever expects to read it)
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._store: Dict[str, Dict[str, Any]] = {}  # id -> {"vec": np.array, "meta": {...}}

    # Match Pinecone's call shape used in our retriever
    def upsert(self, vectors: List[Dict[str, Any]]):
        for row in vectors:
            vid = str(row["id"])
            vals = np.asarray(row["values"], dtype="float32")
            if vals.shape[0] != self.dim:
                raise ValueError(f"Vector dim {vals.shape[0]} != {self.dim}")
            meta = dict(row.get("metadata") or {})
            self._store[vid] = {"vec": vals, "meta": meta}
        return {"upserted_count": len(vectors)}

    def query(self, vector: List[float], top_k: int = 10, filter: Optional[Dict[str, Any]] = None):
        q = np.asarray(vector, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        matches = []
        for vid, payload in self._store.items():
            if filter and not _filter_match(payload["meta"], filter):
                continue
            v = payload["vec"]
            v = v / (np.linalg.norm(v) + 1e-9)
            score = float(np.dot(v, q))  # cosine since normalized
            m = {"id": vid, "score": score, **payload["meta"]}
            # carry "text" up so VectorRetriever can read it
            if "text" in payload["meta"]:
                m["text"] = payload["meta"]["text"]
            matches.append(m)
        matches.sort(key=lambda m: m["score"], reverse=True)
        return {"matches": matches[:top_k]}


def _filter_match(meta: Dict[str, Any], f: Dict[str, Any]) -> bool:
    """
    Evaluate a subset of Pinecone's filter grammar on metadata.
    Supports {"$and":[...]} and field-level {"$eq","$ne","$in","$nin","$gte","$lte"} combos.
    """
    if not f:
        return True

    def field_ok(field: str, cond: Dict[str, Any]) -> bool:
        val = meta.get(field)
        for op, want in cond.items():
            if op == "$eq":
                if val != want:
                    return False
            elif op == "$ne":
                if val == want:
                    return False
            elif op == "$in":
                if val not in set(want):
                    return False
            elif op == "$nin":
                if val in set(want):
                    return False
            elif op == "$gte":
                try:
                    if not (float(val) >= float(want)): # type: ignore
                        return False
                except Exception:
                    return False
            elif op == "$lte":
                try:
                    if not (float(val) <= float(want)): # type: ignore
                        return False
                except Exception:
                    return False
            else:
                # ignore unsupported for this fake
                return False
        return True

    # AND over clauses; support OR groups
    if "$and" in f:
        return all(_filter_match(meta, sub) for sub in f["$and"])

    if "$or" in f:
        return any(_filter_match(meta, sub) for sub in f["$or"])

    # leaf map: {field: {op:...}} or {field: scalar} (eq)
    for field, cond in f.items():
        if isinstance(cond, dict):
            if not field_ok(field, cond):
                return False
        else:
            if meta.get(field) != cond:
                return False
    return True


# --------------------------------- Corpus --------------------------------

def _make_corpus():
    texts = [
        "Yen carry trade unwinds as BOJ signals tightening.",
        "Japan equities fall as yields rise; exporters pressured.",
        "US inflation surprises to upside, yields spike.",
        "Apple launches new iPhone; supply chain optimistic.",
        "Crude oil rallies as OPEC cuts supply.",
    ]
    metas = [
        {"id": "c1", "region": "Asia", "assetClass": "FX"},
        {"id": "c2", "region": "Asia", "assetClass": "Equity"},
        {"id": "c3", "region": "US",   "assetClass": "Rates"},
        {"id": "c4", "region": "US",   "assetClass": "Equity"},
        {"id": "c5", "region": "EM",   "assetClass": "Commodities"},
    ]
    return texts, metas


# ---------------------------------- Tests ---------------------------------

def test_vector_retriever_with_fake_pinecone_and_filters():
    emb = HashingEmbedder(dim=256)
    idx = FakePineconeIndex(dim=256)

    texts, metas = _make_corpus()
    # upsert into fake pinecone
    vecs = emb.encode(texts)
    payload = []
    for t, m, v in zip(texts, metas, vecs):
        meta = {"text": t, **m}
        payload.append({"id": m["id"], "values": v, "metadata": meta})
    out = idx.upsert(payload)
    assert out["upserted_count"] == len(texts)

    # Make retriever
    vr = VectorRetriever(backend="pinecone", embedder=emb)
    vr.add_index(idx)  # <-- pass the fake index

    # Basic query
    q = "yen carry trade and BOJ"
    hits = vr.search(q, top_k=3, filters=None)
    assert hits and len(hits) <= 3
    # top-1 likely yen/boj text
    top_texts = " ".join([(h.text or "") for h in hits[:2]])
    assert ("yen" in top_texts.lower()) or ("boj" in top_texts.lower())

    # With filters: only Asia
    flt = Filter.from_dict({"region": "Asia"})
    hits_asia = vr.search(q, top_k=5, filters=flt)
    assert hits_asia
    assert all(h.meta.get("region") == "Asia" for h in hits_asia)

    # With filters: IN on asset class
    flt2 = Filter.from_dict({"assetClass": {"in": ["FX", "Rates"]}})
    hits_ac = vr.search(q, top_k=5, filters=flt2)
    assert hits_ac
    assert all(h.meta.get("assetClass") in {"FX", "Rates"} for h in hits_ac)


def test_fake_pinecone_respects_scores_ordering():
    emb = HashingEmbedder(dim=128)
    idx = FakePineconeIndex(dim=128)

    texts = [
        "carry trade yen boj",
        "apple iphone launch",
        "oil opec supply cut"
    ]
    metas = [{"id": f"c{i+1}"} for i in range(len(texts))]
    vecs = emb.encode(texts)
    idx.upsert([{"id": m["id"], "values": v, "metadata": {"text": t, **m}} for t, m, v in zip(texts, metas, vecs)])

    vr = VectorRetriever(backend="pinecone", embedder=emb)
    vr.add_index(idx)

    q = "boj yen carry"
    hits = vr.search(q, top_k=3)
    assert len(hits) == 3
    # Expect first hit to be the carry/yen doc
    assert hits[0].text.startswith("carry") or "yen" in hits[0].text.lower()


def test_filter_in_and_nin_behaviour():
    emb = HashingEmbedder(dim=64)
    idx = FakePineconeIndex(dim=64)

    texts = ["doc a", "doc b", "doc c", "doc d"]
    metas = [
        {"id": "a", "region": "US"},
        {"id": "b", "region": "EU"},
        {"id": "c", "region": "Asia"},
        {"id": "d", "region": "US"},
    ]
    vecs = emb.encode(texts)
    idx.upsert([{"id": m["id"], "values": v, "metadata": {"text": t, **m}} for t, m, v in zip(texts, metas, vecs)])

    vr = VectorRetriever(backend="pinecone", embedder=emb)
    vr.add_index(idx)

    # IN
    flt_in = Filter.from_dict({"region": {"in": ["US", "EU"]}})
    hits_in = vr.search("doc", top_k=10, filters=flt_in)
    assert {h.meta["region"] for h in hits_in}.issubset({"US", "EU"}) and len(hits_in) >= 3

    # NIN
    flt_nin = Filter.from_dict({"region": {"nin": ["US", "EU"]}})
    hits_nin = vr.search("doc", top_k=10, filters=flt_nin)
    assert all(h.meta["region"] not in {"US", "EU"} for h in hits_nin)