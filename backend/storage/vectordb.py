#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vectordb.py
-----------
Minimal, robust vector database with pluggable backends and metadata persistence.

Backends
- "numpy"   : in-memory flat index (cosine/dot/L2)
- "faiss"   : if faiss-cpu installed
- "hnsw"    : if hnswlib installed
- "annoy"   : if annoy installed

Features
- Namespaces & ids; upsert / delete / rebuild
- Persist to a directory (index + jsonl metadata)
- Query by text, vector, or hybrid (sparse+dense via BM25-lite)
- Similarity: cosine / dot / L2
- Simple metadata filtering (equals), pagination, deterministic toy embedder
"""

from __future__ import annotations
import os
import io
import re
import json
import math
import time
import shutil
import tempfile
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Protocol

import numpy as np

# Optional backends
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    import hnswlib  # type: ignore
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

try:
    from annoy import AnnoyIndex  # type: ignore
    _HAS_ANNOY = True
except Exception:
    _HAS_ANNOY = False


# =============================================================================
# Embedding interface
# =============================================================================

class EmbeddingFunction(Protocol):
    dim: int
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_one(self, text: str) -> np.ndarray: ...

class SimpleEmbedder:
    """
    Deterministic toy embedder (hash trigram). Replace in prod with your model.
    """
    def __init__(self, dim: int = 384):
        self.dim = int(dim)

    def _hash(self, s: str) -> np.ndarray:
        s = (s or "").lower()
        v = np.zeros(self.dim, dtype=np.float32)
        if len(s) < 3:
            return v
        for i in range(len(s) - 2):
            tri = s[i:i+3]
            h = int(hashlib.blake2b(tri.encode("utf-8"), digest_size=16).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._hash(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self._hash(text)


# =============================================================================
# Similarity & normalization
# =============================================================================

def normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def sim_cosine(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return Q @ X.T  # expects normalized

def sim_dot(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return Q @ X.T

def dist_l2(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    q2 = np.sum(Q * Q, axis=1, keepdims=True)
    x2 = np.sum(X * X, axis=1, keepdims=True).T
    return q2 + x2 - 2 * (Q @ X.T)


# =============================================================================
# Sparse (BM25-lite) for hybrid
# =============================================================================

class BM25Lite:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = float(k1), float(b)
        self.df: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.avg_len: float = 0.0
        self.N: int = 0
        self.idx: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]

    def fit(self, corpus: Sequence[str]) -> None:
        self.idx = []
        self.df.clear()
        self.doc_len = []
        for t in corpus:
            toks = self._tokenize(t)
            self.idx.append(toks)
            self.doc_len.append(len(toks))
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(self.idx)
        self.avg_len = float(np.mean(self.doc_len)) if self.doc_len else 0.0

    def query(self, text: str) -> np.ndarray:
        if self.N == 0:
            return np.zeros(0, dtype=np.float32)
        toks = self._tokenize(text)
        scores = np.zeros(self.N, dtype=np.float32)
        for i, doc in enumerate(self.idx):
            s = 0.0
            L = self.doc_len[i] or 1
            for w in toks:
                f = doc.count(w)
                if f == 0:
                    continue
                df = self.df.get(w, 0)
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0
                denom = f + self.k1 * (1 - self.b + self.b * L / (self.avg_len or 1))
                s += idf * (f * (self.k1 + 1) / denom)
            scores[i] = s
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-12)
        return scores


# =============================================================================
# Storage helpers
# =============================================================================

def _atomic_write(path: str, bytes_or_text: bytes | str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "wb") as f:
        if isinstance(bytes_or_text, str):
            bytes_or_text = bytes_or_text.encode("utf-8")
        f.write(bytes_or_text)
    shutil.move(tmp, path)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    buf = io.StringIO()
    for r in rows:
        buf.write(json.dumps(r, ensure_ascii=False) + "\n")
    _atomic_write(path, buf.getvalue())


# =============================================================================
# VectorDB
# =============================================================================

@dataclass
class VectorDB:
    dim: int
    backend: str = "numpy"         # "numpy" | "faiss" | "hnsw" | "annoy"
    metric: str = "cosine"         # "cosine" | "dot" | "l2"
    embedder: Optional[EmbeddingFunction] = None
    persist_dir: Optional[str] = None
    ef_search: int = 64            # hnsw
    M: int = 32                    # hnsw
    trees: int = 50                # annoy
    seed: int = 42

    # runtime
    _ids: List[str] = field(default_factory=list, init=False)
    _ns: List[str]  = field(default_factory=list, init=False)
    _meta: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _texts: List[str] = field(default_factory=list, init=False)
    _X: Optional[np.ndarray] = field(default=None, init=False)
    _bm25: Optional[BM25Lite] = field(default=None, init=False)

    # backend handles
    _faiss: Any = field(default=None, init=False)
    _hnsw: Any = field(default=None, init=False)
    _annoy: Any = field(default=None, init=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # ----------------------------- core --------------------------------

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = SimpleEmbedder(dim=self.dim)

    def _metric_annoy(self):
        if self.metric in ("cosine", "dot"):
            return "angular"
        elif self.metric == "l2":
            return "euclidean"
        raise ValueError("Unsupported metric for annoy")

    def _maybe_normalize(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return normalize_rows(X.astype(np.float32))
        return X.astype(np.float32)

    # -------------------------- CRUD ops --------------------------------

    def upsert(self, items: Sequence[Dict[str, Any]], namespace: str = "default") -> List[str]:
        """
        Items: dicts with keys {id?, text?, vector?, meta?}. Returns list of ids.
        """
        self._ensure_embedder()
        new_ids, new_ns, new_meta, new_texts = [], [], [], []
        vecs = []

        for it in items:
            _id = str(it.get("id") or hashlib.md5(json.dumps(it, sort_keys=True).encode()).hexdigest()[:12])
            text = it.get("text", "")
            vec = it.get("vector", None)
            meta = it.get("meta", {})

            if vec is None:
                vec = self.embedder.embed_one(text)#type:ignore
            else:
                vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            if vec.shape[-1] != self.dim:
                raise ValueError(f"Vector dim mismatch: got {vec.shape[-1]}, expected {self.dim}")
            vecs.append(vec)
            new_ids.append(_id)
            new_ns.append(namespace)
            new_meta.append(meta)
            new_texts.append(text)

        Xnew = self._maybe_normalize(np.vstack(vecs))

        with self._lock:
            # Remove existing ids (overwrite)
            ext = set(new_ids)
            keep = [i for i, _id in enumerate(self._ids) if _id not in ext]
            if self._X is not None and keep != list(range(len(self._ids))):
                self._X = self._X[keep] if keep else None
                self._ids = [self._ids[i] for i in keep]
                self._ns  = [self._ns[i] for i in keep]
                self._meta= [self._meta[i] for i in keep]
                self._texts=[self._texts[i] for i in keep]

            # Append
            self._X = Xnew if self._X is None else np.vstack([self._X, Xnew])
            self._ids.extend(new_ids)
            self._ns.extend(new_ns)
            self._meta.extend(new_meta)
            self._texts.extend(new_texts)

            self._rebuild_backend(incremental=True)
            self._fit_bm25()

            if self.persist_dir:
                self.save(self.persist_dir)

        return new_ids

    def delete(self, ids: Sequence[str] | None = None, namespace: Optional[str] = None) -> int:
        with self._lock:
            if self._X is None:
                return 0
            ids = set(ids or [])#type:ignore
            rm = []
            for idx, (_id, ns) in enumerate(zip(self._ids, self._ns)):
                cond = (not ids or _id in ids) and (namespace is None or ns == namespace)
                if cond:
                    rm.append(idx)
            if not rm:
                return 0
            keep = [i for i in range(len(self._ids)) if i not in rm]
            self._X = self._X[keep] if keep else None
            self._ids = [self._ids[i] for i in keep]
            self._ns  = [self._ns[i] for i in keep]
            self._meta= [self._meta[i] for i in keep]
            self._texts=[self._texts[i] for i in keep]

            self._rebuild_backend(incremental=False)
            self._fit_bm25()
            if self.persist_dir:
                self.save(self.persist_dir)
            return len(rm)

    def rebuild(self) -> None:
        with self._lock:
            self._rebuild_backend(incremental=False)

    # --------------------------- Querying --------------------------------

    def query(self, query: str | np.ndarray, top_k: int = 10,
              namespace: Optional[str] = None,
              hybrid: bool = False, alpha: float = 0.5,
              where: Optional[Dict[str, Any]] = None,
              offset: int = 0) -> List[Dict[str, Any]]:
        """
        Query by text or vector.
        - hybrid=True: combine dense sim and BM25-lite with weight alpha in [0,1]
        - filters: namespace, where={key:value}
        - pagination: offset + top_k
        """
        with self._lock:
            if self._X is None or len(self._ids) == 0:
                return []

            mask = np.ones(len(self._ids), dtype=bool)
            if namespace is not None:
                mask &= (np.array(self._ns) == namespace)
            if where:
                for k, v in where.items():
                    mask &= np.array([m.get(k) == v for m in self._meta], dtype=bool)
            if not mask.any():
                return []

            X = self._X[mask]
            ids = [i for i, m in zip(self._ids, mask) if m]
            metas = [m for m, mm in zip(self._meta, mask) if mm]
            texts = [t for t, mm in zip(self._texts, mask) if mm]

            # Dense query vector
            if isinstance(query, str):
                self._ensure_embedder()
                qv = self.embedder.embed_one(query).reshape(1, -1).astype(np.float32)#type:ignore
                qv = self._maybe_normalize(qv)
            else:
                qv = np.asarray(query, dtype=np.float32).reshape(1, -1)
                if qv.shape[-1] != self.dim:
                    raise ValueError("Query vector dim mismatch.")
                qv = self._maybe_normalize(qv)

            dense_scores = self._dense_search(qv, X)

            if hybrid:
                if self._bm25 is None or len(self._bm25.idx) != len(self._texts):
                    self._fit_bm25()
                sparse_scores_full = self._bm25.query(query if isinstance(query, str) else "")#type:ignore
                # select masked subset
                sparse_scores = sparse_scores_full[np.where(mask)[0]] if len(sparse_scores_full) == len(self._ids) else np.zeros_like(dense_scores.ravel())
                d = dense_scores.ravel(); s = sparse_scores.ravel()
                if d.max() > 0: d = d / (d.max() + 1e-12)
                if s.max() > 0: s = s / (s.max() + 1e-12)
                scores = (1 - alpha) * d + alpha * s
            else:
                scores = dense_scores.ravel()

            order = np.argsort(-scores)
            if offset > 0:
                order = order[offset:]
            order = order[:top_k]

            out = []
            for r in order:
                out.append({
                    "id": ids[r],
                    "score": float(scores[r]),
                    "meta": metas[r],
                    "text": texts[r],
                })
            return out

    def _dense_search(self, qv: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.backend == "faiss" and _HAS_FAISS and self._faiss is not None:
            D, _ = self._faiss.search(qv, min(1000, X.shape[0]))
            return D if self.metric in ("cosine", "dot") else -D
        elif self.backend == "hnsw" and _HAS_HNSW and self._hnsw is not None:
            labels, distances = self._hnsw.knn_query(qv, k=min(1000, X.shape[0]))
            return (1.0 - distances.astype(np.float32)) if self.metric in ("cosine", "dot") else -distances.astype(np.float32)
        elif self.backend == "annoy" and _HAS_ANNOY and self._annoy is not None:
            k = min(1000, X.shape[0])
            idxs, dists = self._annoy.get_nns_by_vector(qv.ravel().tolist(), k, include_distances=True)
            idxs = np.array(idxs, dtype=int)
            dists = np.array(dists, dtype=np.float32)
            sim = -dists if self.metric == "l2" else 1.0 - dists
            full = np.full((1, X.shape[0]), -np.inf, dtype=np.float32)
            full[0, idxs] = sim
            return full
        else:
            if self.metric == "cosine":
                return sim_cosine(qv, X)
            elif self.metric == "dot":
                return sim_dot(qv, X)
            elif self.metric == "l2":
                return -dist_l2(qv, X)
            else:
                raise ValueError("Unknown metric")

    # --------------------------- Backend setup --------------------------

    def _rebuild_backend(self, incremental: bool = True) -> None:
        X = self._X
        if X is None or X.shape[0] == 0:
            self._faiss = self._hnsw = self._annoy = None
            return

        if self.backend == "faiss" and _HAS_FAISS:
            index = faiss.IndexFlatIP(self.dim) if self.metric in ("cosine", "dot") else faiss.IndexFlatL2(self.dim)
            index.add(X)
            self._faiss = index

        elif self.backend == "hnsw" and _HAS_HNSW:
            space = "cosine" if self.metric in ("cosine", "dot") else "l2"
            p = hnswlib.Index(space=space, dim=self.dim)
            p.init_index(max_elements=X.shape[0], ef_construction=200, M=self.M, random_seed=self.seed)
            p.add_items(X, np.arange(X.shape[0]))
            p.set_ef(max(self.ef_search, 32))
            self._hnsw = p

        elif self.backend == "annoy" and _HAS_ANNOY:
            m = self._metric_annoy()
            a = AnnoyIndex(self.dim, m)
            for i in range(X.shape[0]):
                a.add_item(i, X[i].tolist())
            a.build(self.trees)
            self._annoy = a
        # numpy backend: nothing to build

    def _fit_bm25(self) -> None:
        self._bm25 = BM25Lite()
        self._bm25.fit(self._texts)

    # --------------------------- Persistence ---------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        rows = [{"id": i, "ns": ns, "meta": m, "text": t} for i, ns, m, t in zip(self._ids, self._ns, self._meta, self._texts)]
        _write_jsonl(os.path.join(directory, "meta.jsonl"), rows)
        if self._X is not None:
            np.save(os.path.join(directory, "vectors.npy"), self._X)
        info = {"dim": self.dim, "metric": self.metric, "backend": self.backend, "count": len(self._ids), "ts": time.time()}
        _atomic_write(os.path.join(directory, "_info.json"), json.dumps(info, indent=2))

        if self.backend == "annoy" and _HAS_ANNOY and self._annoy is not None:
            self._annoy.save(os.path.join(directory, "annoy.idx"))
        elif self.backend == "hnsw" and _HAS_HNSW and self._hnsw is not None:
            self._hnsw.save_index(os.path.join(directory, "hnsw.idx"))
        elif self.backend == "faiss" and _HAS_FAISS and self._faiss is not None:
            faiss.write_index(self._faiss, os.path.join(directory, "faiss.idx"))

    def load(self, directory: str) -> "VectorDB":
        meta_path = os.path.join(directory, "meta.jsonl")
        vec_path = os.path.join(directory, "vectors.npy")
        info_path = os.path.join(directory, "_info.json")
        if not (os.path.exists(meta_path) and os.path.exists(vec_path) and os.path.exists(info_path)):
            raise FileNotFoundError(f"Incomplete index at {directory}")

        rows = _read_jsonl(meta_path)
        self._ids = [r["id"] for r in rows]
        self._ns  = [r.get("ns", "default") for r in rows]
        self._meta= [r.get("meta", {}) for r in rows]
        self._texts=[r.get("text", "") for r in rows]

        self._X = np.load(vec_path).astype(np.float32)
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        self.dim = int(info.get("dim", self.dim))
        self.metric = info.get("metric", self.metric)
        self.backend = info.get("backend", self.backend)

        # Load backend index if available; else rebuild
        if self.backend == "annoy" and _HAS_ANNOY and os.path.exists(os.path.join(directory, "annoy.idx")):
            a = AnnoyIndex(self.dim, self._metric_annoy())
            a.load(os.path.join(directory, "annoy.idx"))
            self._annoy = a
        elif self.backend == "hnsw" and _HAS_HNSW and os.path.exists(os.path.join(directory, "hnsw.idx")):
            p = hnswlib.Index(space=("cosine" if self.metric in ("cosine", "dot") else "l2"), dim=self.dim)
            p.load_index(os.path.join(directory, "hnsw.idx"))
            p.set_ef(max(self.ef_search, 32))
            self._hnsw = p
        elif self.backend == "faiss" and _HAS_FAISS and os.path.exists(os.path.join(directory, "faiss.idx")):
            self._faiss = faiss.read_index(os.path.join(directory, "faiss.idx"))
        else:
            self._rebuild_backend(incremental=False)

        self._fit_bm25()
        self.persist_dir = directory
        return self

    # --------------------------- Utilities -----------------------------

    def count(self, namespace: Optional[str] = None) -> int:
        if namespace is None:
            return len(self._ids)
        return sum(1 for ns in self._ns if ns == namespace)

    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self._ids),
            "dim": self.dim,
            "backend": self.backend,
            "metric": self.metric,
            "namespaces": {ns: int(sum(1 for n in self._ns if n == ns)) for ns in set(self._ns)},
        }

    def iter_meta(self, namespace: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        for _id, ns, m, t in zip(self._ids, self._ns, self._meta, self._texts):
            if namespace is None or ns == namespace:
                yield {"id": _id, "ns": ns, "meta": m, "text": t}

    def compact(self) -> None:
        with self._lock:
            if self._X is None:
                return
            mask = np.isfinite(self._X).all(axis=1)
            if mask.all():
                return
            self._X = self._X[mask]
            self._ids = [i for i, ok in zip(self._ids, mask) if ok]#type:ignore
            self._ns  = [i for i, ok in zip(self._ns, mask) if ok]#type:ignore
            self._meta= [i for i, ok in zip(self._meta, mask) if ok]#type:ignore
            self._texts=[i for i, ok in zip(self._texts, mask) if ok]#type:ignore
            self._rebuild_backend(incremental=False)
            self._fit_bm25()
            if self.persist_dir:
                self.save(self.persist_dir)


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    emb = SimpleEmbedder(dim=128)
    db = VectorDB(dim=128, backend="numpy", metric="cosine", embedder=emb)

    docs = [
        {"id": "a1", "text": "Yen carry trade profits from higher foreign yields than JPY.", "meta": {"topic":"fx","country":"JP"}},
        {"id": "a2", "text": "Momentum in equities: buy winners and sell losers.", "meta": {"topic":"equity"}},
        {"id": "a3", "text": "Value investors like low EV/EBITDA and P/E multiples.", "meta": {"topic":"equity","style":"value"}},
        {"id": "a4", "text": "Oil supply shocks move inflation and breakevens.", "meta": {"topic":"macro"}},
    ]
    db.upsert(docs, namespace="notes")
    print("Stats:", db.stats())

    hits = db.query("rate differentials carry trade", top_k=3, namespace="notes", hybrid=True, alpha=0.4)
    print("Query:", [(h["id"], round(h["score"], 3)) for h in hits])

    tmp = tempfile.mkdtemp(prefix="vdb_")
    db.save(tmp)
    fresh = VectorDB(dim=128, backend="numpy").load(tmp)
    again = fresh.query("value multiples", top_k=2, namespace="notes")
    print("Reload:", [h["id"] for h in again])