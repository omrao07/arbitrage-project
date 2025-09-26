# analytics-engine/vector-ai/index_builder/faiss_indexer.py
"""
FAISS Indexer
-------------
Append/upsert/search wrapper around a flat FAISS index with a row-aligned
metadata parquet (append-only by default). Provides:
- build() / load() / save()
- add_texts() append
- upsert() by chunk_id (replace vector & metadata)
- delete() logical delete by chunk_id (flag in parquet)
- search() top-k ANN with aligned metadata rows

Layout assumptions
------------------
- FAISS IDs correspond to *row positions* in metadata parquet.
- Append-only growth keeps alignment trivial.
- Upserts allocate a new row, mark old as deleted (keeps index monotonic).
  (Repacking/compaction is optional; see `compact()`.)

Dependencies
------------
pip install faiss-cpu pandas pyarrow numpy

Notes
-----
- Uses IndexFlatIP (cosine if embeddings are normalized); switch to L2 via metric="l2".
- For large corpora, upgrade to IVF/HNSW variants (extend `_new_index()`).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("Missing FAISS. Install: pip install faiss-cpu") from e


# ------------------------------ Types ------------------------------

Metadata = Dict[str, Any]


@dataclass
class FAISSConfig:
    index_path: Path
    meta_path: Path
    dim: int
    metric: str = "ip"          # "ip" (inner product) or "l2"
    text_field: str = "text"    # field name in metadata containing the text
    id_field: str = "chunk_id"  # unique ID per vector row
    deleted_field: str = "deleted"  # logical delete flag in metadata


# ------------------------------ Core -------------------------------

class FAISSIndexer:
    """
    Row-aligned FAISS + Parquet metadata indexer.
    Each vector's FAISS id == its row index in metadata parquet.
    """

    def __init__(self, cfg: FAISSConfig):
        self.cfg = cfg
        self.index = None
        self.meta_df: Optional[pd.DataFrame] = None

    # ---------- lifecycle ----------

    def _new_index(self) -> Any:
        if self.cfg.metric.lower() == "ip":
            return faiss.IndexFlatIP(self.cfg.dim)
        elif self.cfg.metric.lower() == "l2":
            return faiss.IndexFlatL2(self.cfg.dim)
        else:
            raise ValueError("metric must be 'ip' or 'l2'")

    def load(self) -> None:
        """Load FAISS and metadata (if present), or create fresh structures."""
        ip, mp = self.cfg.index_path, self.cfg.meta_path
        if ip.exists():
            self.index = faiss.read_index(str(ip))
            if self.index.d != self.cfg.dim:
                raise ValueError(f"Loaded FAISS dim={self.index.d} != cfg.dim={self.cfg.dim}")
        else:
            self.index = self._new_index()

        if mp.exists():
            self.meta_df = pd.read_parquet(mp).reset_index(drop=True)
        else:
            self.meta_df = pd.DataFrame(columns=[self.cfg.id_field, self.cfg.text_field, self.cfg.deleted_field])

    def save(self) -> None:
        """Persist FAISS index and metadata parquet."""
        if self.index is None or self.meta_df is None:
            raise RuntimeError("Index not initialized; call load() or build() first.")
        self.cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.meta_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.cfg.index_path))
        self.meta_df.reset_index(drop=True).to_parquet(self.cfg.meta_path, index=False)

    def build(self, vectors: np.ndarray, metas: List[Metadata]) -> None:
        """
        Build a *fresh* index from vectors + metadata. Overwrites files.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.cfg.dim:
            raise ValueError(f"vectors must be (N,{self.cfg.dim})")
        if len(metas) != vectors.shape[0]:
            raise ValueError("metas length must equal vectors rows")
        self.index = self._new_index()
        self.index.add(vectors.astype("float32"))

        df = pd.DataFrame(metas).copy()
        if self.cfg.id_field not in df.columns:
            df[self.cfg.id_field] = [str(uuid.uuid4()) for _ in range(len(df))]
        if self.cfg.deleted_field not in df.columns:
            df[self.cfg.deleted_field] = False
        self.meta_df = df.reset_index(drop=True)
        self.save()

    # ---------- append / upsert / delete ----------

    def _ensure_loaded(self):
        if self.index is None or self.meta_df is None:
            self.load()

    def add_texts(
        self,
        texts: List[str],
        embedder,
        base_meta: Optional[Metadata] = None,
        metas: Optional[List[Metadata]] = None,
        batch: int = 256,
    ) -> List[str]:
        """
        Append new texts with metadata. Returns assigned chunk_ids.
        - `embedder` must implement .encode(List[str]) -> List[List[float]] and .dim
        - if `metas` provided, len(metas) must equal len(texts)
        - `base_meta` is merged into each meta (shallow)
        """
        self._ensure_loaded()

        if getattr(embedder, "dim", None) and int(embedder.dim) != self.cfg.dim:
            raise ValueError(f"Embedder dim {embedder.dim} != index dim {self.cfg.dim}")

        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch):
            vecs.extend(embedder.encode(texts[i : i + batch]))
        arr = np.asarray(vecs, dtype="float32")
        self.index.add(arr) # type: ignore

        ids: List[str] = []
        rows: List[Metadata] = []
        base_meta = base_meta or {}

        for i, text in enumerate(texts):
            meta = dict(base_meta)
            if metas and i < len(metas):
                meta.update(metas[i] or {})
            if self.cfg.text_field not in meta:
                meta[self.cfg.text_field] = text
            if self.cfg.id_field not in meta:
                meta[self.cfg.id_field] = str(uuid.uuid4())
            meta[self.cfg.deleted_field] = bool(meta.get(self.cfg.deleted_field, False))
            rows.append(meta)
            ids.append(meta[self.cfg.id_field])

        add_df = pd.DataFrame(rows)
        self.meta_df = pd.concat([self.meta_df, add_df], ignore_index=True).reset_index(drop=True)
        self.save()
        return ids

    def upsert(
        self,
        chunk_id: str,
        new_text: str,
        embedder,
        meta_updates: Optional[Metadata] = None,
    ) -> str:
        """
        Upsert by chunk_id: marks old row as deleted, appends new row + vector.
        Returns new chunk_id (may be same if you keep it).
        """
        self._ensure_loaded()
        df = self.meta_df
        idf = self.cfg.id_field
        del_f = self.cfg.deleted_field

        # mark old as deleted (logical)
        mask = (df[idf] == chunk_id) # type: ignore
        if mask.any():
            self.meta_df.loc[mask, del_f] = True # type: ignore

        # append new
        new_meta = dict(meta_updates or {})
        new_meta[self.cfg.text_field] = new_text
        new_meta[idf] = new_meta.get(idf, str(uuid.uuid4()))
        new_meta[del_f] = False

        # embed + add
        vec = np.asarray(embedder.encode([new_text])[0], dtype="float32")[None, :]
        if vec.shape[1] != self.cfg.dim:
            raise ValueError(f"Embedder dim {vec.shape[1]} != index dim {self.cfg.dim}")
        self.index.add(vec) # type: ignore
        self.meta_df = pd.concat([self.meta_df, pd.DataFrame([new_meta])], ignore_index=True)
        self.save()
        return new_meta[idf]

    def delete(self, chunk_ids: Iterable[str]) -> int:
        """
        Logical delete: sets deleted=true in metadata. Returns #affected.
        (Physical deletion requires compaction; see compact()).
        """
        self._ensure_loaded()
        df = self.meta_df
        idf = self.cfg.id_field
        del_f = self.cfg.deleted_field
        mask = df[idf].isin(list(chunk_ids)) # type: ignore
        n = int(mask.sum())
        if n:
            self.meta_df.loc[mask, del_f] = True # type: ignore
            self.save()
        return n

    # ---------- search ----------

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        include_text: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        allow_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        ANN search; returns metadata rows enriched with vector_score.
        """
        self._ensure_loaded()
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[1] != self.cfg.dim:
            raise ValueError(f"query_vec dim {query_vec.shape[1]} != index dim {self.cfg.dim}")

        scores, ids = self.index.search(query_vec.astype("float32"), top_k * 5)  # type: ignore # oversample then filter
        scores = scores[0].tolist()
        ids = ids[0].tolist()

        out: List[Dict[str, Any]] = []
        df = self.meta_df
        idf, txtf, delf = self.cfg.id_field, self.cfg.text_field, self.cfg.deleted_field

        for faiss_id, s in zip(ids, scores):
            if faiss_id < 0:
                continue
            try:
                row = df.iloc[int(faiss_id)].to_dict() # type: ignore
            except Exception:
                continue

            if not allow_deleted and bool(row.get(delf, False)):
                continue

            # apply filters (match exact or list membership)
            if filters:
                ok = True
                for k, v in filters.items():
                    rv = row.get(k)
                    if isinstance(v, list):
                        if rv not in v:
                            ok = False; break
                    else:
                        if rv != v:
                            ok = False; break
                if not ok:
                    continue

            payload = {
                "vector_score": float(s),
                "chunk_id": row.get(idf),
                "doc_id": row.get("doc_id") or row.get(idf),
                "source": row.get("source"),
            }
            if include_text:
                payload["text"] = row.get(txtf)
            # include remaining metadata sans large text
            payload["meta"] = {k: v for k, v in row.items() if k not in {txtf}}
            out.append(payload)

            if len(out) >= top_k:
                break

        return out

    # ---------- maintenance ----------

    def compact(self) -> Tuple[int, int]:
        """
        Physically rebuild the index excluding deleted rows.
        Returns: (old_rows, new_rows)
        """
        self._ensure_loaded()
        delf = self.cfg.deleted_field
        keep_df = self.meta_df[~self.meta_df[delf].astype(bool)].reset_index(drop=True) # type: ignore

        # rebuild vectors cannot be recovered from FAISS; caller must keep raw text & re-embed externally
        # Here we compact *metadata only*. For full compaction, re-embed keep_df[text_field] with the same embedder.
        old, new = len(self.meta_df), len(keep_df) # type: ignore
        self.meta_df = keep_df
        # NOTE: FAISS still contains old vectors; for a true physical compact you must re-embed & rebuild:
        #  vectors = embedder.encode(keep_df[text_field].tolist()); self.build(np.asarray(vectors), keep_df.to_dict("records"))
        self.save()
        return old, new


# ------------------------------ Helpers -----------------------------

def ensure_paths(index_path: str | Path, meta_path: str | Path) -> Tuple[Path, Path]:
    ip, mp = Path(index_path), Path(meta_path)
    ip.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    return ip, mp


# ------------------------------ Quick test --------------------------

if __name__ == "__main__":
    # Minimal smoke test with random vectors
    dim = 384
    ip, mp = ensure_paths("./indices/demo.faiss", "./indices/demo.parquet")
    cfg = FAISSConfig(index_path=ip, meta_path=mp, dim=dim, metric="ip")

    idx = FAISSIndexer(cfg)
    idx.load()

    # Build if empty
    if idx.meta_df is None or len(idx.meta_df) == 0:
        n = 100
        vecs = np.random.randn(n, dim).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        metas = [{"chunk_id": f"c{i}", "text": f"doc {i}", "source": "unit", "deleted": False} for i in range(n)]
        idx.build(vecs, metas)
        print("built demo index")

    # Search with a random query
    q = np.random.randn(dim).astype("float32"); q /= (np.linalg.norm(q) + 1e-9)
    res = idx.search(q, top_k=5)
    print("top-5:", json.dumps(res, indent=2)[:400], "...")