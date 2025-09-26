# analytics-engine/vector-ai/index_builder/pinecone_indexer.py
"""
Pinecone Indexer
----------------
Thin wrapper around Pinecone's vector DB with a local Parquet metadata log.
Features:
- ensure_index() (create if missing), load(), save_meta()
- add_texts()  -> append new vectors + metadata
- upsert()     -> upsert by custom id (e.g., chunk_id)
- delete()     -> hard delete by ids
- fetch()      -> retrieve vectors/metadata by ids (from Pinecone + local parquet)
- search()     -> ANN query by vector (or by text with an embedder)
- sync()       -> reconcile Pinecone state to metadata parquet (optional, best-effort)

Assumptions
-----------
- You store document-level info (doc_id, source, tags, etc.) in the Parquet.
- Each Pinecone vector `id` corresponds to your `chunk_id` (unique).
- You keep the metadata Parquet append-only (no rewrite), except via this wrapper.

Install
-------
pip install pinecone-client pandas pyarrow numpy

Env
---
PINECONE_API_KEY=...
# Optional serverless creation params (if creating an index):
PINECONE_CLOUD=aws|gcp
PINECONE_REGION=us-east-1|...
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pinecone import Pinecone
except Exception as e:
    raise RuntimeError("Missing pinecone-client. Install: pip install pinecone-client") from e


# ------------------------------ Config ------------------------------

@dataclass
class PineconeConfig:
    index_name: str
    namespace: Optional[str]
    dim: int
    metric: str = "cosine"     # "cosine" | "dotproduct" | "euclidean"
    meta_path: Path = Path("./indices/pinecone_meta.parquet")
    # If creating a serverless index:
    cloud: Optional[str] = os.getenv("PINECONE_CLOUD") or None   # e.g., "aws" or "gcp"
    region: Optional[str] = os.getenv("PINECONE_REGION") or None # e.g., "us-east-1"


# ------------------------------ Indexer -----------------------------

class PineconeIndexer:
    """
    Pinecone + Parquet metadata wrapper.
    """

    def __init__(self, cfg: PineconeConfig):
        self.cfg = cfg
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("Set PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        self.meta_df: Optional[pd.DataFrame] = None

    # ---------- lifecycle ----------

    def ensure_index(self, pods: Optional[int] = None, replicas: Optional[int] = None) -> None:
        """
        Ensure the index exists; auto-creates a *serverless* index if missing.
        For pod-based (legacy) setups, create/manage externally.
        """
        existing = {ix.name for ix in self.pc.list_indexes()}
        if self.cfg.index_name in existing:
            return
        # serverless create
        cloud = self.cfg.cloud or "aws"
        region = self.cfg.region or "us-east-1"
        self.pc.create_index(
            name=self.cfg.index_name,
            dimension=int(self.cfg.dim),
            metric=self.cfg.metric,
            spec={"serverless": {"cloud": cloud, "region": region}},
        )

    def load(self) -> None:
        """Open the Pinecone index and load metadata parquet (if any)."""
        self.index = self.pc.Index(self.cfg.index_name)
        if self.cfg.meta_path.exists():
            self.meta_df = pd.read_parquet(self.cfg.meta_path).reset_index(drop=True)
        else:
            self.meta_df = pd.DataFrame(
                columns=["chunk_id", "doc_id", "source", "text", "tags", "meta"]
            )

    def save_meta(self) -> None:
        if self.meta_df is None:
            return
        self.cfg.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_df.reset_index(drop=True).to_parquet(self.cfg.meta_path, index=False)

    # ---------- append / upsert / delete ----------

    def add_texts(
        self,
        texts: List[str],
        embedder,
        base_meta: Optional[Dict[str, Any]] = None,
        metas: Optional[List[Dict[str, Any]]] = None,
        batch: int = 256,
    ) -> List[str]:
        """
        Append new texts with metadata. Generates chunk_ids if absent and upserts to Pinecone.
        Returns the list of assigned chunk_ids.
        """
        if self.index is None or self.meta_df is None:
            self.load()

        # Embed in batches
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch):
            vecs.extend(embedder.encode(texts[i : i + batch]))
        arr = np.asarray(vecs, dtype="float32")

        # Prepare Pinecone vectors
        ids: List[str] = []
        to_upsert: List[Dict[str, Any]] = []
        meta_rows: List[Dict[str, Any]] = []

        base_meta = base_meta or {}
        for i, text in enumerate(texts):
            row_meta = dict(base_meta)
            if metas and i < len(metas):
                row_meta.update(metas[i] or {})
            chunk_id = str(row_meta.get("chunk_id") or uuid.uuid4())
            doc_id = str(row_meta.get("doc_id") or chunk_id)

            # keep small text in metadata only if you really need it downstream
            md = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": row_meta.get("source"),
                "text": text,  # store text to allow direct preview (optional; can omit if privacy/cost concerns)
                "tags": row_meta.get("tags") or [],
                **{k: v for k, v in row_meta.items() if k not in {"chunk_id", "doc_id", "source", "tags"}},
            }

            ids.append(chunk_id)
            to_upsert.append({"id": chunk_id, "values": arr[i].tolist(), "metadata": md})
            meta_rows.append(md)

        # Upsert to Pinecone
        # (Pinecone handles batching internally up to ~100 vectors; we chunk anyway)
        for i in range(0, len(to_upsert), 100):
            batch_items = to_upsert[i : i + 100]
            self.index.upsert(vectors=batch_items, namespace=self.cfg.namespace) # type: ignore

        # Append to local parquet
        add_df = pd.DataFrame(meta_rows)
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
        Upsert by `chunk_id`. If it doesn't exist, it will be created.
        """
        if self.index is None or self.meta_df is None:
            self.load()

        # embed
        vec = np.asarray(embedder.encode([text])[0], dtype="float32").tolist()

        # choose doc_id (stable or equals chunk_id)
        doc_id = chunk_id
        if not self.meta_df.empty:# type: ignore
            sel = self.meta_df[self.meta_df["chunk_id"] == chunk_id]# type: ignore
            if not sel.empty:
                doc_id = str(sel.iloc[0].get("doc_id") or chunk_id)

        # merge metadata
        md = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
        }
        if meta_updates:
            md.update(meta_updates)

        # upsert vector
        self.index.upsert(# type: ignore
            vectors=[{"id": chunk_id, "values": vec, "metadata": md}],
            namespace=self.cfg.namespace,
        )

        # update local parquet: replace/append
        if not self.meta_df.empty and (self.meta_df["chunk_id"] == chunk_id).any():# type: ignore
            mask = self.meta_df["chunk_id"] == chunk_id# type: ignore
            # fill + keep columns union
            cur = self.meta_df.loc[mask].iloc[0].to_dict()# type: ignore
            cur.update(md)
            self.meta_df.loc[mask, :] = pd.DataFrame([cur])# type: ignore
        else:
            self.meta_df = pd.concat([self.meta_df, pd.DataFrame([md])], ignore_index=True)

        self.save_meta()
        return chunk_id

    def delete(self, chunk_ids: Iterable[str]) -> int:
        """
        Hard delete (Pinecone delete) and remove from local parquet.
        """
        ids = list(set(map(str, chunk_ids)))
        if not ids:
            return 0
        self.index.delete(ids=ids, namespace=self.cfg.namespace)# type: ignore
        if self.meta_df is not None and not self.meta_df.empty:
            mask = self.meta_df["chunk_id"].isin(ids)
            n = int(mask.sum())
            if n:
                self.meta_df = self.meta_df.loc[~mask].reset_index(drop=True)
                self.save_meta()
            return n
        return 0

    # ---------- fetch / search ----------

    def fetch(self, chunk_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """
        Fetch vectors/metadata by ids. Returns list of {chunk_id, values?, meta?}.
        """
        ids = list(set(map(str, chunk_ids)))
        if not ids:
            return []
        out: List[Dict[str, Any]] = []
        res = self.index.fetch(ids=ids, namespace=self.cfg.namespace)# type: ignore
        vecs = res.vectors or {}
        for cid, entry in vecs.items():
            md = dict(entry.metadata or {})
            out.append({
                "chunk_id": cid,
                "values": entry.values,   # can be large; drop in API if not needed
                "meta": md,
            })
        # Fill any that are not in Pinecone (but maybe in parquet)
        missing = set(ids) - set(vecs.keys())
        if missing and self.meta_df is not None and not self.meta_df.empty:
            local = self.meta_df[self.meta_df["chunk_id"].isin(missing)]
            for _, r in local.iterrows():
                out.append({"chunk_id": str(r["chunk_id"]), "values": None, "meta": r.to_dict()})
        return out

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        include_text: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        ANN search by *vector*. For text queries, embed first with your embedder:
            qv = np.asarray(embedder.encode(["query"])[0], dtype="float32")[None, :]
            hits = pine.search(qv, top_k=10)
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        q = query_vec[0].tolist()

        # Basic metadata filtering is best done *post*-query; Pinecone also supports server-side filter dicts.
        res = self.index.query(# type: ignore
            vector=q,
            top_k=int(top_k),
            namespace=self.cfg.namespace,
            include_values=False,
            include_metadata=True,
            # filter=...  # TODO: map `filters` into Pinecone JSON filter if desired
        )

        out: List[Dict[str, Any]] = []
        for m in res.matches or []:# type: ignore
            md = dict(m.metadata or {})# type: ignore
            row = {
                "chunk_id": md.get("chunk_id") or m.id,
                "doc_id": md.get("doc_id") or md.get("chunk_id") or m.id,
                "source": md.get("source"),
                "text": (md.get("text") if include_text else None),
                "meta": {k: v for k, v in md.items() if (k != "text" or include_text)},
                "vector_score": float(m.score or 0.0),
            }
            # client-side filters (exact match or list membership)
            if filters:
                ok = True
                for k, v in filters.items():
                    rv = row["meta"].get(k) if k in row["meta"] else row.get(k)
                    if isinstance(v, list):
                        if rv not in v: ok = False; break
                    else:
                        if rv != v: ok = False; break
                if not ok:
                    continue
            out.append(row)

        return out

    # ---------- reconciliation ----------

    def sync(self, sample_k: int = 1000) -> Tuple[int, int]:
        """
        Best-effort sanity check: sample fetch from Pinecone and ensure entries
        exist in local parquet. Returns (#checked, #repaired). Repairs are limited
        to adding any missing metadata rows discovered via fetch().
        """
        if self.meta_df is None or self.meta_df.empty:
            return (0, 0)
        ids = self.meta_df["chunk_id"].astype(str).tolist()[:sample_k]
        res = self.fetch(ids)
        have = {r["chunk_id"] for r in res}
        missing_ids = [i for i in ids if i not in have]
        # attempt another fetch in case of pagination issues
        repaired = 0
        if missing_ids:
            res2 = self.fetch(missing_ids)
            repaired += len(res2)
        return (len(ids), repaired)


# ------------------------------ Quick test --------------------------

if __name__ == "__main__":
    # Minimal smoke test (requires a real PINECONE_API_KEY)
    cfg = PineconeConfig(
        index_name="demo-index",
        namespace="live",
        dim=384,
        metric="cosine",
        meta_path=Path("./indices/pinecone_meta.parquet"),
    )
    ix = PineconeIndexer(cfg)
    ix.ensure_index()
    ix.load()

    # Dummy embedder for demo
    class _Dummy:
        dim = 384
        def encode(self, texts):  # normalized random for demo
            rng = np.random.default_rng(42)
            vs = rng.normal(size=(len(texts), self.dim)).astype("float32")
            vs /= (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
            return vs.tolist()

    emb = _Dummy()
    ids = ix.add_texts(["hello markets", "carry trade unwind risk"], emb, base_meta={"source": "unit"})
    qv = np.asarray(emb.encode(["carry trade unwind"])[0], dtype="float32")[None, :]
    hits = ix.search(qv, top_k=5)
    print("hits:", hits[:2])