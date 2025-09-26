# tests/test_weaviate_indexer.py
"""
Offline tests for the Weaviate retrieval path using a tiny in-memory fake client.

Why a fake?
-----------
CI should not hit a live Weaviate cluster. This test provides a minimal
`FakeWeaviateClient` that mimics the chained query interface used in
VectorRetriever (search/retriever.py):

  client.query
        .get(class_name, ["text", "_additional {certainty id}"])
        .with_near_vector({"vector": ...})
        .with_where({...})
        .with_limit(k)
        .do()

It validates:
- Vector search ranking via cosine on normalized vectors
- Translation of Filter -> Weaviate "where" JSON (basic smoke)
- Metadata round-trip (text + id + certainty)

Requires:
  pytest
  numpy

Run:
  pytest -q tests/test_weaviate_indexer.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from search.retriever import VectorRetriever
from search.filters import Filter


# ----------------------------- Fake Weaviate -----------------------------

class FakeWeaviateQueryBuilder:
    def __init__(self, store, class_name_default="Document"):
        self._store = store  # dict: id -> {"vec": np.ndarray, "obj": {...}}
        self._class = class_name_default
        self._props = []
        self._near = None
        self._where = None
        self._limit = 10

    # API surface mimicked from retriever
    def get(self, class_name: str, props: List[str]):
        self._class = class_name
        self._props = props
        return self

    def with_near_vector(self, near: Dict[str, Any]):
        self._near = near
        return self

    def with_where(self, where: Dict[str, Any]):
        self._where = where
        return self

    def with_limit(self, k: int):
        self._limit = int(k)
        return self

    def do(self) -> Dict[str, Any]:
        qvec = np.asarray(self._near.get("vector"), dtype="float32") # type: ignore
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)

        # Simple evaluator for a subset of Weaviate "where"
        def where_ok(meta: Dict[str, Any]) -> bool:
            w = self._where or {}
            if not w:
                return True

            def eval_node(node):
                if not isinstance(node, dict):
                    return True
                op = node.get("operator")
                if op in (None, "And"):
                    ops = node.get("operands", [])
                    return all(eval_node(x) for x in ops)
                if op == "Or":
                    ops = node.get("operands", [])
                    return any(eval_node(x) for x in ops)
                if op == "Not":
                    ops = node.get("operands", [])
                    return not any(eval_node(x) for x in ops)
                # leaf: path + operator over a single field
                path = node.get("path", [])
                field = path[0] if path else None
                if field is None:
                    return True
                vbool = node.get("valueBoolean")
                vtext = node.get("valueText")
                varr  = node.get("valueTextArray")
                val = meta.get(field)
                if op == "Equal":
                    return str(val) == str(vtext)
                if op == "NotEqual":
                    return str(val) != str(vtext)
                if op == "ContainsAny":
                    return str(val) in set(str(x) for x in (varr or []))
                if op == "IsNull":
                    return (val is None) == bool(vbool)
                if op == "Like":
                    # crude *pattern*: treat '*' as wildcard anywhere
                    pat = str(vtext or "")
                    s = str(val or "")
                    # convert to regex-ish: escape then replace '*' with '.*'
                    import re
                    rx = re.compile("^" + re.escape(pat).replace("\\*", ".*") + "$", re.IGNORECASE)
                    return bool(rx.match(s))
                # GreaterThan/Equal etc. lightly supported as text/float
                if op in ("GreaterThan", "GreaterThanEqual", "LessThan", "LessThanEqual"):
                    try:
                        a = float(val) # type: ignore
                        b = float(vtext) # type: ignore
                    except Exception:
                        return False
                    if op == "GreaterThan": return a > b
                    if op == "GreaterThanEqual": return a >= b
                    if op == "LessThan": return a < b
                    if op == "LessThanEqual": return a <= b
                return True

            return eval_node(w)

        rows = []
        for _id, payload in self._store.items():
            v = payload["vec"]
            meta = payload["obj"]
            if not where_ok(meta):
                continue
            v = v / (np.linalg.norm(v) + 1e-9)
            cert = float(np.dot(v, qvec))  # cosine similarity as "certainty"
            row = {
                "text": meta.get("text", ""),
                "_additional": {"certainty": cert, "id": _id},
            }
            rows.append(row)

        rows.sort(key=lambda r: r["_additional"]["certainty"], reverse=True)
        rows = rows[: self._limit]
        return {"data": {"Get": {self._class: rows}}}


class FakeWeaviateClient:
    def __init__(self, dim: int = 256, class_name: str = "Document"):
        self._dim = dim
        self._class = class_name
        self._store: Dict[str, Dict[str, Any]] = {}  # id -> {"vec": np.ndarray, "obj": dict}
        self.query = FakeWeaviateQueryBuilder(self._store, class_name_default=class_name)

    def upsert(self, _id: str, vector: List[float], obj: Dict[str, Any]):
        v = np.asarray(vector, dtype="float32")
        if v.shape[0] != self._dim:
            raise ValueError("dim mismatch")
        self._store[str(_id)] = {"vec": v, "obj": dict(obj)}


# ---------------------------- Test Utilities -----------------------------

class HashingEmbedder:
    """Deterministic, dependency-light embedder (hash buckets + L2)."""
    def __init__(self, dim: int = 256):
        self.dim = int(dim)

    @staticmethod
    def _tok(text: str):
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

def test_vector_retriever_weaviate_fake_with_filters(monkeypatch):
    # Monkeypatch VectorRetriever to use our fake client's query chain
    emb = HashingEmbedder(dim=256)
    fake = FakeWeaviateClient(dim=256, class_name="Document")

    texts, metas = _make_corpus()
    vecs = emb.encode(texts)

    # Upsert docs into fake store
    for t, m, v in zip(texts, metas, vecs):
        fake.upsert(m["id"], v, {"text": t, **m})

    # Patch the retriever to think the "index" is our fake client
    vr = VectorRetriever(backend="weaviate", embedder=emb, class_name="Document")
    vr.add_index(fake)  # NOTE: retriever expects .query chain present on this object

    # Basic query
    q = "yen carry trade and BOJ"
    hits = vr.search(q, top_k=3, filters=None)
    assert hits and len(hits) <= 3
    top_texts = " ".join([(h.text or "") for h in hits[:2]])
    assert ("yen" in top_texts.lower()) or ("boj" in top_texts.lower())

    # With filters: only Asia (Filter -> Weaviate where)
    flt = Filter.from_dict({"region": {"eq": "Asia"}})
    hits_asia = vr.search(q, top_k=10, filters=flt)
    assert hits_asia
    assert all(h.meta.get("region") == "Asia" for h in hits_asia)

    # Prefix filter on ticker-like field (use Like)
    flt2 = Filter.from_dict({"assetClass": {"in": ["FX", "Rates"]}})
    hits_fx_rates = vr.search(q, top_k=10, filters=flt2)
    assert hits_fx_rates
    assert all(h.meta.get("assetClass") in {"FX", "Rates"} for h in hits_fx_rates)


def test_weaviate_like_operator_and_isnull(monkeypatch):
    emb = HashingEmbedder(dim=128)
    fake = FakeWeaviateClient(dim=128, class_name="Document")

    texts = ["alpha beta", "beta gamma", "carry yen", "iphone"]
    metas  = [
        {"id": "a1", "region": "US", "tag": "alpha"},
        {"id": "a2", "region": None, "tag": "beta"},
        {"id": "a3", "region": "Asia", "tag": "carry"},
        {"id": "a4", "region": "EU", "tag": "phone"},
    ]
    vecs = emb.encode(texts)
    for t, m, v in zip(texts, metas, vecs):
        fake.upsert(m["id"], v, {"text": t, **m})

    vr = VectorRetriever(backend="weaviate", embedder=emb, class_name="Document")
    vr.add_index(fake)

    # contains/prefix -> Like (implemented as wildcard)
    wf = Filter.from_dict({"tag": {"contains": "car"}})  # "*car*"
    hits = vr.search("carry", top_k=5, filters=wf)
    assert hits and any("carry" in h.text for h in hits)

    # exists:false -> IsNull true
    wf2 = Filter.from_dict({"region": {"exists": False}})
    hits2 = vr.search("beta", top_k=5, filters=wf2)
    assert hits2 and all(h.meta.get("region") is None for h in hits2)