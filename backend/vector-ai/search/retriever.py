# search/retriever.py
"""
Retrievers for the search stack.

Supports:
- VectorRetriever: Pinecone / FAISS / Weaviate / Local embeddings
- LexicalRetriever: BM25/Whoosh-style inverted index
- HybridRetriever: blends vector + lexical
- Common Candidate dataclass (from reranker.py)

Dependencies:
- numpy
- (optional) pinecone-client, weaviate-client, faiss, whoosh

Install:
pip install numpy pinecone-client weaviate-client faiss-cpu whoosh
"""

from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from search.reranker import Candidate
from search.filters import Filter

# optional deps
try:
    import pinecone  # type: ignore
except Exception:
    pinecone = None
try:
    import weaviate  # type: ignore
except Exception:
    weaviate = None
try:
    import faiss  # type: ignore
except Exception:
    faiss = None
try:
    from whoosh import index, qparser  # type: ignore
except Exception:
    index = None
    qparser = None


# ============================ Vector Retriever =============================

class VectorRetriever:
    """
    Retrieve via vector similarity from Pinecone, Weaviate, FAISS, or a local matrix.
    """

    def __init__(
        self,
        backend: str = "faiss",  # "pinecone" | "weaviate" | "faiss" | "matrix"
        embedder: Any = None,
        **kwargs,
    ):
        self.backend = backend
        self.embedder = embedder
        self.kwargs = kwargs
        self.index = None

        if backend == "pinecone" and pinecone is None:
            warnings.warn("Pinecone not installed. pip install pinecone-client")
        if backend == "weaviate" and weaviate is None:
            warnings.warn("Weaviate client not installed. pip install weaviate-client")
        if backend == "faiss" and faiss is None:
            warnings.warn("faiss not installed. pip install faiss-cpu")

    def add_index(self, index):
        """Attach a ready index handle (pinecone.Index, weaviate.Client, faiss index, or np.ndarray)."""
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Filter] = None,
    ) -> List[Candidate]:
        if self.embedder is None:
            raise ValueError("Embedder required for vector search")

        qv = np.asarray(self.embedder.encode([query])[0], dtype="float32")

        if self.backend == "pinecone":
            if pinecone is None or self.index is None:
                return []
            res = self.index.query(vector=qv.tolist(), top_k=top_k, filter=filters.to_pinecone() if filters else None)
            return [
                Candidate(text=m.get("text", ""), meta=m, base_score=m.get("score", 0.0))
                for m in res.get("matches", [])
            ]

        if self.backend == "weaviate":
            if weaviate is None or self.index is None:
                return []
            where = filters.to_weaviate() if filters else {}
            res = (
                self.index.query
                .get(self.kwargs.get("class_name", "Document"), ["text", "_additional {certainty id}"])
                .with_near_vector({"vector": qv.tolist()})
                .with_where(where)
                .with_limit(top_k)
                .do()
            )
            out = []
            for obj in res.get("data", {}).get("Get", {}).get("Document", []):
                meta = {**obj}
                score = obj["_additional"]["certainty"]
                out.append(Candidate(text=obj.get("text", ""), meta=meta, base_score=score))
            return out

        if self.backend == "faiss":
            if faiss is None or self.index is None:
                return []
            qv_norm = qv / (np.linalg.norm(qv) + 1e-9)
            D, I = self.index.search(qv_norm[None, :], top_k)
            out = []
            for score, idx in zip(D[0], I[0]):
                out.append(Candidate(text="", meta={"id": int(idx)}, base_score=float(score)))
            return out

        if self.backend == "matrix":
            if not isinstance(self.index, np.ndarray):
                return []
            qv_norm = qv / (np.linalg.norm(qv) + 1e-9)
            M = self.index / (np.linalg.norm(self.index, axis=1, keepdims=True) + 1e-9)
            sims = M @ qv_norm
            top_idx = np.argsort(-sims)[:top_k]
            return [Candidate(text="", meta={"id": int(i)}, base_score=float(sims[i])) for i in top_idx]

        return []


# ============================ Lexical Retriever ============================

class LexicalRetriever:
    """
    Retrieve via Whoosh/BM25-style index.
    """

    def __init__(self, ix=None, field: str = "text"):
        self.ix = ix
        self.field = field

    def add_index(self, ix):
        self.ix = ix

    def search(self, query: str, top_k: int = 20) -> List[Candidate]:
        if self.ix is None or index is None or qparser is None:
            return []
        qp = qparser.QueryParser(self.field, schema=self.ix.schema)
        q = qp.parse(query)
        out = []
        with self.ix.searcher() as s:
            res = s.search(q, limit=top_k)
            for hit in res:
                out.append(Candidate(text=hit[self.field], meta=dict(hit), base_score=float(hit.score)))
        return out


# ============================ Hybrid Retriever ============================

class HybridRetriever:
    """
    Blend vector and lexical retrievers.
    """

    def __init__(self, vector: Optional[VectorRetriever] = None, lexical: Optional[LexicalRetriever] = None):
        self.vector = vector
        self.lexical = lexical

    def search(
        self,
        query: str,
        top_k: int = 20,
        alpha: float = 0.5,
        filters: Optional[Filter] = None,
    ) -> List[Candidate]:
        vec = self.vector.search(query, top_k=top_k, filters=filters) if self.vector else []
        lex = self.lexical.search(query, top_k=top_k) if self.lexical else []
        allc = vec + lex
        if not allc:
            return []

        # Normalize scores
        base = np.asarray([c.base_score for c in allc], dtype=float)
        if base.size > 0:
            lo, hi = float(base.min()), float(base.max())
            rng = hi - lo if hi > lo else 1.0
            base = (base - lo) / rng
            for c, s in zip(allc, base):
                c.base_score = float(s)

        # Blend: vector vs lexical
        for c in allc:
            if "source" in c.meta and c.meta["source"] == "vector":
                c.base_score = alpha * c.base_score
            else:
                c.base_score = (1 - alpha) * c.base_score

        allc.sort(key=lambda x: x.base_score, reverse=True)
        return allc[:top_k]