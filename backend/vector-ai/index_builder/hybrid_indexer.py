# analytics-engine/vector-ai/index_builder/hybrid_indexer.py
"""
Hybrid Indexer = BM25 (sparse) + Vector ANN (dense) with score fusion.

What it does
------------
- Builds/loads a Whoosh BM25 index for lexical search over `text`
- Delegates dense ANN to FAISS (via local .faiss + row-aligned parquet) OR Pinecone
- Fuses scores with min-max normalization + weight alpha (or RRF)
- Returns deduped, fused-ranked results with aligned metadata

Install
-------
pip install whoosh pandas pyarrow numpy
# optional dense backends
pip install faiss-cpu pinecone-client sentence-transformers

Example
-------
hy = HybridIndexer(
    whoosh_dir="./indices/whoosh",
    dense_meta="./indices/live.parquet",
    dense_faiss="./indices/live.faiss",   # or set pinecone_index + namespace
    alpha=0.35,                           # weight on dense score
)
hy.ensure_ready()
hy.rebuild_sparse_from_parquet()         # build BM25 from parquet once (or on schedule)
hits = hy.search("yen carry trade unwind Japan yields", top_k=25, mode="fusion")
"""

from __future__ import annotations

import os
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------- Optional dense backends ----------
try:
    import faiss  # type: ignore # noqa: F401
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

try:
    from pinecone import Pinecone  # noqa: F401
    _PINE_OK = True
except Exception:
    _PINE_OK = False

# ---------- Sparse (Whoosh) ----------
try:
    from whoosh.index import create_in, open_dir # type: ignore
    from whoosh.fields import Schema, ID, TEXT, STORED # type: ignore
    from whoosh.qparser import MultifieldParser # type: ignore
except Exception as e:
    raise RuntimeError("Missing Whoosh. Install: pip install whoosh") from e


# ============================== Config ==============================

@dataclass
class HybridConfig:
    whoosh_dir: Path
    dense_meta_path: Optional[Path] = None
    dense_faiss_path: Optional[Path] = None
    pinecone_index: Optional[str] = None
    pinecone_namespace: Optional[str] = None
    id_field: str = "chunk_id"
    text_field: str = "text"
    source_field: str = "source"
    deleted_field: str = "deleted"
    alpha: float = 0.35                   # weight on dense score in fusion
    rrf_k: float = 60.0                   # RRF constant
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # for query encoding if Pinecone used


# ============================== Dense Adapter =======================

class _DenseAdapter:
    """
    Wraps FAISS (local + parquet) or Pinecone (cloud).
    For FAISS: FAISS ids == row index in parquet (append-only alignment).
    """
    def __init__(self, meta_path: Optional[Path], faiss_path: Optional[Path],
                 pine_index: Optional[str], pine_ns: Optional[str],
                 embedder_name: str):
        self.meta_path = meta_path
        self.faiss_path = faiss_path
        self.pine_index = pine_index
        self.pine_ns = pine_ns
        self.embedder_name = embedder_name

        self.meta_df: Optional[pd.DataFrame] = None
        self.faiss_index = None
        self.pine = None
        self._embedder = None  # only needed for Pinecone queries

    def ensure_loaded(self):
        # metadata
        if self.meta_path and self.meta_path.exists():
            self.meta_df = pd.read_parquet(self.meta_path).reset_index(drop=True)

        # FAISS local
        if self.faiss_path:
            if not _FAISS_OK:
                raise RuntimeError("FAISS selected but faiss-cpu not installed")
            if not self.faiss_path.exists():
                raise RuntimeError(f"FAISS index not found: {self.faiss_path}")
            import faiss# type: ignore
            self.faiss_index = faiss.read_index(str(self.faiss_path))

        # Pinecone cloud
        if self.pine_index and not self.faiss_index:
            if not _PINE_OK:
                raise RuntimeError("Pinecone selected but pinecone-client not installed")
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise RuntimeError("Set PINECONE_API_KEY for Pinecone")
            from pinecone import Pinecone
            self.pine = Pinecone(api_key=api_key).Index(self.pine_index)
            # load embedder for queries
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedder_name)
            self._embedder.max_seq_length = 512

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        if self.faiss_index is not None:
            # Query encoding handled by caller (API) usually; if not, you can inject here.
            return None
        if self.pine is not None and self._embedder is not None:
            v = self._embedder.encode([query], normalize_embeddings=True)
            return np.asarray(v, dtype="float32")
        return None

    def search(self, query_vec: Optional[np.ndarray], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Returns list of {chunk_id, vector_score, doc_id, text, source, meta}
        """
        out: List[Dict[str, Any]] = []
        # FAISS path
        if self.faiss_index is not None and query_vec is not None:
            scores, ids = self.faiss_index.search(query_vec, top_k)
            scores, ids = scores[0], ids[0]
            for idx, sc in zip(ids, scores):
                if idx < 0:
                    continue
                row = {"vector_score": float(sc)}
                if self.meta_df is not None and 0 <= int(idx) < len(self.meta_df):
                    meta = self.meta_df.iloc[int(idx)].to_dict()
                    row.update({
                        "chunk_id": meta.get("chunk_id") or meta.get("doc_id") or str(idx),
                        "doc_id": meta.get("doc_id") or meta.get("chunk_id") or str(idx),
                        "text": meta.get("text"),
                        "source": meta.get("source"),
                        "meta": {k: v for k, v in meta.items() if k != "text"},
                    }) # type: ignore
                else:
                    row.update({"chunk_id": str(idx), "doc_id": str(idx), "text": None, "source": None, "meta": {}})# type: ignore
                out.append(row)
            return out

        # Pinecone path (compute embedding inside)
        if self.pine is not None:
            if query_vec is None:
                # encode now
                raise RuntimeError("query_vec is None; provide encode_query() result or embed externally")
            qv = query_vec[0].tolist()
            res = self.pine.query(vector=qv, top_k=top_k, namespace=self.pine_ns,
                                  include_values=False, include_metadata=True)
            for m in res.matches:# type: ignore
                md = dict(m.metadata or {})
                out.append({
                    "chunk_id": md.get("chunk_id") or m.id,
                    "doc_id": md.get("doc_id") or md.get("chunk_id") or m.id,
                    "text": md.get("text"),
                    "source": md.get("source"),
                    "meta": {k: v for k, v in md.items() if k not in {"text"}},
                    "vector_score": float(m.score or 0.0),
                })
            return out

        return out


# ============================== Hybrid Indexer =======================

class HybridIndexer:
    """
    Maintains a Whoosh BM25 index + dense adapter, and provides hybrid search.
    """
    def __init__(self,
                 whoosh_dir: str | Path,
                 dense_meta: Optional[str | Path] = None,
                 dense_faiss: Optional[str | Path] = None,
                 pinecone_index: Optional[str] = None,
                 pinecone_namespace: Optional[str] = None,
                 id_field: str = "chunk_id",
                 text_field: str = "text",
                 source_field: str = "source",
                 deleted_field: str = "deleted",
                 alpha: float = 0.35,
                 rrf_k: float = 60.0,
                 embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.cfg = HybridConfig(
            whoosh_dir=Path(whoosh_dir),
            dense_meta_path=Path(dense_meta) if dense_meta else None,
            dense_faiss_path=Path(dense_faiss) if dense_faiss else None,
            pinecone_index=pinecone_index,
            pinecone_namespace=pinecone_namespace,
            id_field=id_field,
            text_field=text_field,
            source_field=source_field,
            deleted_field=deleted_field,
            alpha=alpha,
            rrf_k=rrf_k,
            embedder_name=embedder_name,
        )
        self._schema = Schema(
            chunk_id=ID(stored=True),
            doc_id=ID(stored=True),
            source=ID(stored=True),
            text=TEXT(stored=True),
            meta=STORED,
        )
        self._ix = None
        self._dense = _DenseAdapter(
            self.cfg.dense_meta_path, self.cfg.dense_faiss_path,
            self.cfg.pinecone_index, self.cfg.pinecone_namespace,
            self.cfg.embedder_name
        )

    # ---------------- Lifecycle ----------------

    def ensure_ready(self):
        self.cfg.whoosh_dir.mkdir(parents=True, exist_ok=True)
        # clean stale lock if present
        lock = self.cfg.whoosh_dir / "MAIN_WRITELOCK"
        if lock.exists():
            lock.unlink(missing_ok=True)
        # open/create Whoosh index
        try:
            self._ix = open_dir(str(self.cfg.whoosh_dir))
        except Exception:
            self._ix = create_in(str(self.cfg.whoosh_dir), self._schema)
        # dense side
        self._dense.ensure_loaded()

    def rebuild_sparse_from_parquet(self, where: Optional[str] = None):
        """
        Rebuild Whoosh index from the dense metadata parquet.
        Optional `where` is a pandas query string (e.g., "deleted != True").
        """
        if self.cfg.dense_meta_path is None or not self.cfg.dense_meta_path.exists():
            raise RuntimeError("dense_meta_path not set or missing; cannot rebuild sparse index")
        df = pd.read_parquet(self.cfg.dense_meta_path).reset_index(drop=True)

        # Filter out deleted by default
        if self.cfg.deleted_field in df.columns:
            df = df[df[self.cfg.deleted_field] != True]
        if where:
            df = df.query(where)

        # wipe directory and recreate
        try:
            self._ix.close()# type: ignore
        except Exception:
            pass
        for p in self.cfg.whoosh_dir.glob("*"):
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p, ignore_errors=True)
        self._ix = create_in(str(self.cfg.whoosh_dir), self._schema)

        writer = self._ix.writer(limitmb=512, procs=0, multisegment=True)
        for _, r in df.iterrows():
            cid = str(r.get(self.cfg.id_field) or r.get("doc_id") or "")
            did = str(r.get("doc_id") or r.get(self.cfg.id_field) or "")
            src = str(r.get(self.cfg.source_field) or "")
            txt = str(r.get(self.cfg.text_field) or "")
            meta = {k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                    for k, v in r.items() if k not in {self.cfg.text_field}}
            writer.add_document(chunk_id=cid, doc_id=did, source=src, text=txt, meta=meta)
        writer.commit(optimize=True)

    # ---------------- Search ----------------

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    def _search_sparse(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        with self._ix.searcher() as s:# type: ignore
            parser = MultifieldParser(["text"], schema=self._ix.schema)# type: ignore
            q = parser.parse(query)
            rs = s.search(q, limit=top_k)
            out = []
            for hit in rs:
                out.append({
                    "chunk_id": hit["chunk_id"],
                    "doc_id": hit["doc_id"],
                    "text": hit.get("text"),
                    "source": hit.get("source"),
                    "meta": dict(hit.get("meta") or {}),
                    "sparse_score": float(hit.score or 0.0),
                })
            return out

    def _search_dense(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        # If FAISS: you must supply a query vector externally (API) and pass it here after encoding.
        # For convenience, we assume FAISS path and external encode in API; for Pinecone we can encode here.
        if self._dense.faiss_index is not None:
            raise RuntimeError("For FAISS, call search_with_query_vec() and provide the encoded vector.")
        qv = self._dense.encode_query(query)
        return self._dense.search(qv, top_k=top_k)

    def search_with_query_vec(
        self,
        query_vec: np.ndarray,
        query_text: str,
        top_k: int = 25,
        mode: str = "fusion",          # "fusion" or "rrf"
        alpha: Optional[float] = None, # override cfg.alpha
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use this for FAISS-backed dense retrieval where you already embedded the query.
        """
        alpha = self.cfg.alpha if alpha is None else float(alpha)

        dense_hits = self._dense.search(query_vec, top_k=top_k * 2)
        sparse_hits = self._search_sparse(query_text, top_k=top_k * 2)

        def _key(r: Dict[str, Any]) -> str:
            return str(r.get("chunk_id") or r.get("doc_id") or "")

        dense_map = {_key(r): r for r in dense_hits}
        sparse_map = {_key(r): r for r in sparse_hits}
        keys = list({*_key(r) for r in dense_hits} | {*_key(r) for r in sparse_hits})# type: ignore

        def _passes(r: Dict[str, Any]) -> bool:
            if not filters:
                return True
            meta = r.get("meta", {})
            for k, v in (filters or {}).items():
                rv = meta.get(k) if k in meta else r.get(k)
                if isinstance(v, list):
                    if rv not in v:
                        return False
                else:
                    if rv != v:
                        return False
            return True

        out: List[Dict[str, Any]] = []
        if mode.lower() == "rrf":
            dense_order = { _key(r): i+1 for i, r in enumerate(dense_hits) }
            sparse_order = { _key(r): i+1 for i, r in enumerate(sparse_hits) }
            for k in keys:
                r_dense = dense_map.get(k)
                r_sparse = sparse_map.get(k)
                merged = (r_dense or {}).copy()
                for src_r in (r_sparse or {},):
                    for kk, vv in src_r.items():
                        merged.setdefault(kk, vv)
                if not _passes(merged):
                    continue
                score = 0.0
                if k in dense_order:
                    score += 1.0 / (self.cfg.rrf_k + dense_order[k])
                if k in sparse_order:
                    score += 1.0 / (self.cfg.rrf_k + sparse_order[k])
                merged["fused_score"] = float(score)
                out.append(merged)
        else:
            d_scores = np.asarray([dense_map.get(k, {}).get("vector_score", 0.0) for k in keys], dtype=np.float32)
            s_scores = np.asarray([sparse_map.get(k, {}).get("sparse_score", 0.0) for k in keys], dtype=np.float32)
            d_norm = self._minmax(d_scores)
            s_norm = self._minmax(s_scores)

            for i, k in enumerate(keys):
                r_dense = dense_map.get(k)
                r_sparse = sparse_map.get(k)
                merged = (r_dense or {}).copy()
                for src_r in (r_sparse or {},):
                    for kk, vv in src_r.items():
                        if kk not in merged or merged[kk] in (None, "", []):
                            merged[kk] = vv
                if not _passes(merged):
                    continue
                fused = alpha * float(d_norm[i]) + (1.0 - alpha) * float(s_norm[i])
                merged["fused_score"] = float(fused)
                out.append(merged)

        out.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
        return out[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 25,
        mode: str = "fusion",
        alpha: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method when using Pinecone (dense encoding done internally).
        If using FAISS, prefer search_with_query_vec() with an externally encoded vector.
        """
        if self._dense.faiss_index is not None:
            raise RuntimeError("Use search_with_query_vec(query_vec, query_text, ...) for FAISS-backed setup.")
        alpha = self.cfg.alpha if alpha is None else float(alpha)
        dense_hits = self._search_dense(query, top_k=top_k * 2)
        sparse_hits = self._search_sparse(query, top_k=top_k * 2)

        def _key(r: Dict[str, Any]) -> str:
            return str(r.get("chunk_id") or r.get("doc_id") or "")

        dense_map = {_key(r): r for r in dense_hits}
        sparse_map = {_key(r): r for r in sparse_hits}
        keys = list({*_key(r) for r in dense_hits} | {*_key(r) for r in sparse_hits})# type: ignore

        def _passes(r: Dict[str, Any]) -> bool:
            if not filters:
                return True
            meta = r.get("meta", {})
            for k, v in (filters or {}).items():
                rv = meta.get(k) if k in meta else r.get(k)
                if isinstance(v, list):
                    if rv not in v:
                        return False
                else:
                    if rv != v:
                        return False
            return True

        out: List[Dict[str, Any]] = []
        if mode.lower() == "rrf":
            dense_order = { _key(r): i+1 for i, r in enumerate(dense_hits) }
            sparse_order = { _key(r): i+1 for i, r in enumerate(sparse_hits) }
            for k in keys:
                r_dense = dense_map.get(k)
                r_sparse = sparse_map.get(k)
                merged = (r_dense or {}).copy()
                for src_r in (r_sparse or {},):
                    for kk, vv in src_r.items():
                        merged.setdefault(kk, vv)
                if not _passes(merged):
                    continue
                score = 0.0
                if k in dense_order:
                    score += 1.0 / (self.cfg.rrf_k + dense_order[k])
                if k in sparse_order:
                    score += 1.0 / (self.cfg.rrf_k + sparse_order[k])
                merged["fused_score"] = float(score)
                out.append(merged)
        else:
            d_scores = np.asarray([dense_map.get(k, {}).get("vector_score", 0.0) for k in keys], dtype=np.float32)
            s_scores = np.asarray([sparse_map.get(k, {}).get("sparse_score", 0.0) for k in keys], dtype=np.float32)
            d_norm = self._minmax(d_scores)
            s_norm = self._minmax(s_scores)
            for i, k in enumerate(keys):
                r_dense = dense_map.get(k)
                r_sparse = sparse_map.get(k)
                merged = (r_dense or {}).copy()
                for src_r in (r_sparse or {},):
                    for kk, vv in src_r.items():
                        if kk not in merged or merged[kk] in (None, "", []):
                            merged[kk] = vv
                if not _passes(merged):
                    continue
                fused = alpha * float(d_norm[i]) + (1.0 - alpha) * float(s_norm[i])
                merged["fused_score"] = float(fused)
                out.append(merged)

        out.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
        return out[:top_k]


# ============================== Quick test ==========================

if __name__ == "__main__":
    # Minimal run assuming you already have dense parquet + faiss
    hy = HybridIndexer(
        whoosh_dir="./indices/whoosh",
        dense_meta="./indices/live.parquet",
        dense_faiss="./indices/live.faiss",
        alpha=0.35
    )
    hy.ensure_ready()
    hy.rebuild_sparse_from_parquet()
    print(json.dumps(hy.search("BOJ easing, JGB yield curve and yen carry", top_k=10)[:2], indent=2, ensure_ascii=False))