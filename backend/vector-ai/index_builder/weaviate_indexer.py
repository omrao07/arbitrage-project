# analytics-engine/vector-ai/index_builder/weaviate_indexer.py
"""
WeaviateIndexer
---------------
Thin wrapper around Weaviate with a local Parquet metadata store.

Design
------
- External embeddings: you supply vectors from your embedder (.encode -> List[List[float]])
- One Weaviate class (collection) per index, vectorizer = "none"
- Stores flexible metadata as JSON string (meta_json) alongside typed fields
- Local Parquet is the append-only metadata log of what's in Weaviate

Features
--------
- ensure_schema(): create class w/ properties if missing
- load() / save_meta()
- add_texts(): append new objects (+ vectors)
- upsert(): replace or create object by chunk_id
- delete(): hard delete by ids
- fetch(): read by ids
- search(): near-vector ANN + optional filters (exact / IN)
- Optional multi-tenant: set `tenant` on calls if you use Weaviate tenants

Install
-------
pip install weaviate-client pandas pyarrow numpy

Env
---
WEAVIATE_URL=https://<cluster>.weaviate.network   # or http://localhost:8080
WEAVIATE_API_KEY=...                               # omit for local/no-auth
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import weaviate
    from weaviate.util import generate_uuid5 # type: ignore
except Exception as e:
    raise RuntimeError("Missing weaviate-client. Install: pip install weaviate-client") from e


# =============================== Config ===============================

@dataclass
class WeaviateConfig:
    class_name: str                          # e.g., "ResearchChunks"
    vector_dim: int                          # must match your embedder
    meta_path: Path = Path("./indices/weaviate_meta.parquet")
    url: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY") or None
    # Optional multi-tenancy (set at class creation time manually in cluster if you need it)
    tenant: Optional[str] = None


# =============================== Indexer ==============================

class WeaviateIndexer:
    """
    Weaviate + Parquet metadata wrapper (external embeddings).
    """

    # -------- schema --------
    BASE_PROPERTIES = [
        {"name": "chunk_id", "dataType": ["text"], "indexInverted": True},
        {"name": "doc_id",   "dataType": ["text"], "indexInverted": True},
        {"name": "source",   "dataType": ["text"], "indexInverted": True},
        {"name": "text",     "dataType": ["text"], "indexInverted": True},
        {"name": "tags",     "dataType": ["text[]"], "indexInverted": True},
        {"name": "meta_json","dataType": ["text"], "indexInverted": False},  # arbitrary JSON
    ]

    def __init__(self, cfg: WeaviateConfig):
        self.cfg = cfg
        auth = weaviate.AuthApiKey(api_key=cfg.api_key) if cfg.api_key else None # type: ignore
        self.client = weaviate.Client(url=cfg.url, auth_client_secret=auth, timeout_config=(10, 120))
        self.meta_df: Optional[pd.DataFrame] = None

    # -------- lifecycle --------

    def ensure_schema(self) -> None:
        """
        Create class if missing. Vectorizer = none; we provide vectors.
        """
        schema = self.client.schema.get()
        classes = {c["class"] for c in schema.get("classes", [])} # type: ignore
        if self.cfg.class_name in classes:
            return

        class_obj = {
            "class": self.cfg.class_name,
            "description": "External-embedded research chunks",
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": "cosine",   # cosine for normalized embeddings
                "efConstruction": 128,
                "maxConnections": 64,
            },
            "invertedIndexConfig": {"bm25": {"k1": 1.2, "b": 0.75}},
            "properties": self.BASE_PROPERTIES,
        }
        self.client.schema.create_class(class_obj)

    def load(self) -> None:
        """Load local parquet (append-only log)."""
        if self.cfg.meta_path.exists():
            self.meta_df = pd.read_parquet(self.cfg.meta_path).reset_index(drop=True)
        else:
            # minimal columns; meta_json keeps arbitrary stuff
            self.meta_df = pd.DataFrame(columns=["chunk_id", "doc_id", "source", "text", "tags", "meta_json"])

    def save_meta(self) -> None:
        if self.meta_df is None:
            return
        self.cfg.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_df.reset_index(drop=True).to_parquet(self.cfg.meta_path, index=False)

    # -------- helpers --------

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v) + 1e-9)
        return (v / n).astype("float32")

    @staticmethod
    def _mk_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Map simple filters into Weaviate "where" filter:
          {"ticker": "AAPL"} or {"ticker": ["AAPL","MSFT"], "lang":"en"}
        -> AND of equality / ContainsAny
        """
        if not filters:
            return None
        clauses = []
        for k, v in filters.items():
            if isinstance(v, list):
                clauses.append({
                    "path": [k],
                    "operator": "ContainsAny",
                    "valueTextArray": [str(s) for s in v],
                })
            else:
                clauses.append({
                    "path": [k],
                    "operator": "Equal",
                    "valueText": str(v),
                })
        if len(clauses) == 1:
            return clauses[0]
        return {"operator": "And", "operands": clauses}

    # -------- append / upsert / delete --------

    def add_texts(
        self,
        texts: List[str],
        embedder,
        base_meta: Optional[Dict[str, Any]] = None,
        metas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 256,
    ) -> List[str]:
        """
        Append new objects with vectors. Returns assigned chunk_ids.
        """
        if self.meta_df is None:
            self.load()
        self.ensure_schema()

        # Encode in batches
        ids: List[str] = []
        rows: List[Dict[str, Any]] = []

        # Weaviate batcher
        with self.client.batch as batch: # type: ignore
            batch.batch_size = min(max(16, batch_size // 4), 128)
            for i in range(0, len(texts), batch_size):
                subtexts = texts[i : i + batch_size]
                vecs = embedder.encode(subtexts)  # List[List[float]]
                arr = np.asarray(vecs, dtype="float32")

                for j, text in enumerate(subtexts):
                    row_meta = dict(base_meta or {})
                    if metas and (i + j) < len(metas):
                        row_meta.update(metas[i + j] or {})
                    chunk_id = str(row_meta.get("chunk_id") or uuid.uuid4())
                    doc_id = str(row_meta.get("doc_id") or chunk_id)
                    tags = row_meta.get("tags") or []

                    # pack arbitrary fields
                    extra = {k: v for k, v in row_meta.items() if k not in {"chunk_id", "doc_id", "source", "tags"}}
                    meta_json = json.dumps(extra, ensure_ascii=False)

                    props = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source": row_meta.get("source"),
                        "text": text,
                        "tags": tags,
                        "meta_json": meta_json,
                    }

                    vec = self._norm(arr[j])
                    batch.add_data_object(
                        data_object=props,
                        class_name=self.cfg.class_name,
                        uuid=chunk_id,            # stable: your chunk_id is the object id
                        vector=vec.tolist(),
                        tenant=self.cfg.tenant,
                    )

                    ids.append(chunk_id)
                    rows.append(props)

        # Append to local parquet
        add_df = pd.DataFrame(rows)
        self.meta_df = pd.concat([self.meta_df, add_df], ignore_index=True)
        self.save_meta()
        return ids

    def upsert(
        self,
        chunk_id: str,
        text: str,
        embedder,
        meta_updates: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upsert by `chunk_id` (id is stable / deterministic).
        """
        if self.meta_df is None:
            self.load()
        self.ensure_schema()

        vec = np.asarray(embedder.encode([text])[0], dtype="float32")
        vec = self._norm(vec)

        # Find current record in parquet (if any) to preserve doc_id/source/tags
        doc_id = chunk_id
        source = None
        tags: List[str] = []
        if self.meta_df is not None and not self.meta_df.empty:
            cur = self.meta_df[self.meta_df["chunk_id"] == chunk_id]
            if not cur.empty:
                row = cur.iloc[0]
                doc_id = str(row.get("doc_id") or doc_id)
                source = row.get("source")
                tags = row.get("tags") or tags

        updates = dict(meta_updates or {})
        extra = {k: v for k, v in updates.items() if k not in {"chunk_id", "doc_id", "source", "tags", "text"}}
        meta_json = json.dumps(extra, ensure_ascii=False)

        props = {
            "chunk_id": chunk_id,
            "doc_id": updates.get("doc_id", doc_id),
            "source": updates.get("source", source),
            "text": text,
            "tags": updates.get("tags", tags),
            "meta_json": meta_json,
        }

        # Weaviate upsert: delete+create (safe) or use merge (data_object.update -> replaces props)
        try:
            self.client.data_object.delete(uuid=chunk_id, class_name=self.cfg.class_name, tenant=self.cfg.tenant)
        except Exception:
            pass
        self.client.data_object.create(
            data_object=props,
            class_name=self.cfg.class_name,
            uuid=chunk_id,
            vector=vec.tolist(),
            tenant=self.cfg.tenant,
        )

        # Update parquet (replace or append)
        if self.meta_df is not None and not self.meta_df.empty and (self.meta_df["chunk_id"] == chunk_id).any():
            mask = self.meta_df["chunk_id"] == chunk_id
            self.meta_df.loc[mask, :] = pd.DataFrame([props])
        else:
            self.meta_df = pd.concat([self.meta_df, pd.DataFrame([props])], ignore_index=True)
        self.save_meta()
        return chunk_id

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = list(set(map(str, chunk_ids)))
        if not ids:
            return 0
        n = 0
        for cid in ids:
            try:
                self.client.data_object.delete(uuid=cid, class_name=self.cfg.class_name, tenant=self.cfg.tenant)
                n += 1
            except Exception:
                pass
        if self.meta_df is not None and not self.meta_df.empty:
            mask = self.meta_df["chunk_id"].isin(ids)
            if mask.any():
                self.meta_df = self.meta_df.loc[~mask].reset_index(drop=True)
                self.save_meta()
        return n

    # -------- fetch / search --------

    def fetch(self, chunk_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(set(map(str, chunk_ids)))
        out: List[Dict[str, Any]] = []
        for cid in ids:
            try:
                obj = self.client.data_object.get_by_id(
                    uuid=cid, class_name=self.cfg.class_name, tenant=self.cfg.tenant
                )
                if not obj:
                    continue
                props = obj.get("properties", {}) or {}
                out.append({"chunk_id": cid, "meta": props})
            except Exception:
                # fall back to parquet if present
                if self.meta_df is not None and not self.meta_df.empty:
                    cur = self.meta_df[self.meta_df["chunk_id"] == cid]
                    if not cur.empty:
                        out.append({"chunk_id": cid, "meta": cur.iloc[0].to_dict()})
        return out

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        include_text: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        ANN search using nearVector. Provide a (1, D) or (D,) normalized vector.
        """
        if query_vec.ndim == 2:
            qv = query_vec[0]
        else:
            qv = query_vec
        qv = self._norm(np.asarray(qv, dtype="float32"))

        props = ["chunk_id", "doc_id", "source", "tags", "meta_json"]
        if include_text:
            props.append("text")

        where = self._mk_where(filters)

        q = (
            self.client.query.get(self.cfg.class_name, props)
            .with_limit(int(top_k))
            .with_near_vector({"vector": qv.tolist()})
        )
        if where:
            q = q.with_where(where)
        if self.cfg.tenant:
            q = q.with_tenant(self.cfg.tenant)

        res = q.do()
        data = (((res or {}).get("data") or {}).get("Get") or {}).get(self.cfg.class_name, []) or []

        out: List[Dict[str, Any]] = []
        for hit in data: # type: ignore
            meta = dict(hit or {})
            # unpack meta_json
            mj = meta.pop("meta_json", None)
            extra = {}
            if mj:
                try:
                    extra = json.loads(mj)
                except Exception:
                    extra = {"_meta_json_parse_error": True, "raw": mj}

            row = {
                "chunk_id": meta.get("chunk_id"),
                "doc_id": meta.get("doc_id") or meta.get("chunk_id"),
                "source": meta.get("source"),
                "text": meta.get("text") if include_text else None,
                "meta": {**{k: v for k, v in meta.items() if k not in {"text"}}, **extra},
                # Weaviate doesn't return a numeric distance/score in v3 .get(); for that use .with_additional("distance")
            }
            out.append(row)

        return out


# ============================== Quick test ===========================

if __name__ == "__main__":
    # Smoke test (needs a running Weaviate; for local: docker run -p 8080:8080 semitechnologies/weaviate:latest)
    cfg = WeaviateConfig(class_name="ResearchChunks", vector_dim=384, url=os.getenv("WEAVIATE_URL", "http://localhost:8080"))
    ix = WeaviateIndexer(cfg)
    ix.ensure_schema()
    ix.load()

    class _Dummy:
        dim = 384
        def encode(self, texts):
            rng = np.random.default_rng(7)
            v = rng.normal(size=(len(texts), self.dim)).astype("float32")
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            return v.tolist()

    emb = _Dummy()
    ids = ix.add_texts(["hello markets", "carry trade unwind risk"], emb, base_meta={"source": "unit", "ticker": "JPY"})
    qv = np.asarray(emb.encode(["carry trade unwind"])[0], dtype="float32")[None, :]
    hits = ix.search(qv, top_k=5, filters={"ticker": "JPY"})
    print("hits:", hits[:2]) 