#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed.py
--------
Universal text embedding utilities + lightweight vector store.

Backends (auto-detected, can be forced via --backend):
  - sentence-transformers (preferred): e.g. "sentence-transformers/all-MiniLM-L6-v2"
  - transformers (feature-extraction) as fallback
  - openai Embeddings (set OPENAI_API_KEY; model e.g. "text-embedding-3-small")

Works nicely with chunker.py:
  python embed.py path/to/doc.md --out out/embeddings.parquet
  python embed.py path/to/doc.csv --kind table --rows-per-chunk 500
  python embed.py path/to/code.py

Write formats: .parquet or .jsonl

Optional FAISS index:
  python embed.py data/ --glob "*.md" --faiss out/index.faiss --meta out/meta.parquet

Env:
  OPENAI_API_KEY=<key>           # if using --backend openai

"""

from __future__ import annotations
import os
import re
import io
import sys
import json
import glob
import time
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Optional deps ----------
_has_st = False
_has_tf = False
_has_openai = False
_has_faiss = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _has_st = True
except Exception:
    pass

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    import torch  # type: ignore
    _has_tf = True
except Exception:
    pass

try:
    import openai  # type: ignore
    _has_openai = True
except Exception:
    pass

try:
    import faiss  # type: ignore
    _has_faiss = True
except Exception:
    pass

# ---------- Local imports ----------
try:
    from chunker import chunk_any  # your chunker.py
except Exception as e:
    raise RuntimeError("embed.py requires chunker.py in PYTHONPATH") from e


# =========================================================
# Embedding backends
# =========================================================

@dataclass
class EmbeddingConfig:
    backend: str = "auto"   # auto | st | hf | openai
    model: str = ""         # model name; auto picks a good default
    device: str = "auto"    # auto|cpu|cuda
    batch_size: int = 64
    normalize: bool = True
    openai_model: str = "text-embedding-3-small"
    openai_timeout: float = 15.0
    openai_max_retries: int = 5


class BaseEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        # L2-normalize rows
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms


# ---- Sentence-Transformers ----
class STEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        if not _has_st:
            raise RuntimeError("sentence-transformers not installed")
        model_name = cfg.model or "sentence-transformers/all-MiniLM-L6-v2"
        device = self._pick_device(cfg.device)
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _pick_device(pref: str) -> str:
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            try:
                import torch  # type: ignore
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        # auto
        try:
            import torch  # type: ignore
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if self.cfg.normalize:
            vecs = self._normalize(vecs)
        return vecs


# ---- Transformers feature-extraction fallback ----
class HFEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        if not _has_tf:
            raise RuntimeError("transformers/torch not installed")
        self.model_name = cfg.model or "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.device = "cpu"
        if cfg.device in ("auto", "cuda"):
            if hasattr(self.model, "to"):
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        self.device = "cuda"
                        self.model.to(self.device)
                except Exception:
                    pass

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # mean pooling on last hidden state
        import torch  # type: ignore
        self.model.eval()
        all_vecs = []
        bs = max(1, self.cfg.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            with torch.no_grad():
                toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self.model(**toks)
                last = out.last_hidden_state  # (B, T, H)
                mask = toks["attention_mask"].unsqueeze(-1)  # (B, T, 1)
                masked = last * mask
                summed = masked.sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                mean = summed / counts
                all_vecs.append(mean.cpu().numpy())
        vecs = np.vstack(all_vecs)
        if self.cfg.normalize:
            vecs = self._normalize(vecs)
        return vecs


# ---- OpenAI ----
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig):
        super().__init__(cfg)
        if not _has_openai:
            raise RuntimeError("openai python library not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        # set client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = cfg.openai_model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # batch in small chunks (OpenAI has payload limits)
        vecs: List[np.ndarray] = []
        bs = min(self.cfg.batch_size, 256)
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            out = self._call_openai(chunk)
            arr = np.array([d["embedding"] for d in out["data"]], dtype=np.float32)
            vecs.append(arr)
        allv = np.vstack(vecs)
        if self.cfg.normalize:
            allv = self._normalize(allv)
        return allv

    def _call_openai(self, batch: List[str]) -> Dict[str, Any]:#type:ignore
        # simple retry loop
        for attempt in range(self.cfg.openai_max_retries):
            try:
                resp = openai.Embeddings.create(
                    model=self.model,
                    input=batch,
                    timeout=self.cfg.openai_timeout,
                )
                return resp
            except Exception as e:
                wait = min(2 ** attempt, 30)
                if attempt == self.cfg.openai_max_retries - 1:
                    raise
                time.sleep(wait)


# ---- factory ----
def create_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    backend = cfg.backend.lower()
    if backend == "auto":
        if _has_st:
            return STEmbedder(cfg)
        if _has_tf:
            return HFEmbedder(cfg)
        if _has_openai:
            return OpenAIEmbedder(cfg)
        raise RuntimeError("No embedding backend available. Install sentence-transformers OR transformers OR openai.")
    if backend in ("st", "sentence-transformers"):
        return STEmbedder(cfg)
    if backend in ("hf", "transformers"):
        return HFEmbedder(cfg)
    if backend == "openai":
        return OpenAIEmbedder(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")


# =========================================================
# Vector store (in-memory + optional FAISS)
# =========================================================

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.vecs: Optional[np.ndarray] = None

    def add(self, ids: List[str], vecs: np.ndarray, meta: List[Dict[str, Any]]):
        assert vecs.shape[1] == self.dim
        if self.vecs is None:
            self.vecs = vecs.copy()
        else:
            self.vecs = np.vstack([self.vecs, vecs])
        self.ids.extend(ids)
        self.meta.extend(meta)

    def search(self, q: np.ndarray, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        q: shape (dim,) cosine similarity (assumes embeddings normalized)
        """
        if self.vecs is None or len(self.ids) == 0:
            return []
        sims = (self.vecs @ q.reshape(-1))
        topk = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i]), self.meta[i]) for i in topk]

    # FAISS persistence (optional)
    def to_faiss(self, path: str):
        if not _has_faiss:
            raise RuntimeError("faiss not installed")
        import faiss  # type: ignore
        index = faiss.IndexFlatIP(self.dim)  # cosine (assumes normalized)
        index.add(self.vecs.astype(np.float32))#type:ignore
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)

    @staticmethod
    def faiss_search(path: str, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if not _has_faiss:
            raise RuntimeError("faiss not installed")
        import faiss  # type: ignore
        index = faiss.read_index(path)
        D, I = index.search(queries.astype(np.float32), k)
        return D, I


# =========================================================
# Chunk → embed pipeline
# =========================================================

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_file(path: str,
               kind_hint: Optional[str],
               max_tokens: int,
               overlap: int,
               rows_per_chunk: int) -> List[Dict[str, Any]]:
    text = read_text(path)
    kw = {}
    if kind_hint == "table":
        # chunk_any will route by ext, but for CSV we can pass rows_per_chunk via manual call
        # quick hack: embed rows_per_chunk into text split AFTER chunk_any returns too-large chunks
        pass
    chunks = chunk_any(text, source=path, kind_hint=kind_hint, max_tokens=max_tokens, overlap=overlap)
    # If table and user wants specific rows_per_chunk but chunk_any didn’t use it, we can re-split;
    # but our chunker already supports table chunking if ext is .csv/.tsv.
    return chunks

def gather_inputs(target: str, pattern: Optional[str]) -> List[str]:
    if os.path.isdir(target):
        pat = pattern or "**/*"
        paths = [p for p in glob.glob(os.path.join(target, pat), recursive=True) if os.path.isfile(p)]
        return paths
    if os.path.isfile(target):
        return [target]
    raise FileNotFoundError(target)

def embed_chunks(embedder: BaseEmbedder, chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c["meta"] for c in chunks]
    vecs = embedder.embed_texts(texts)
    return vecs, ids, metas

def write_embeddings(path: str, ids: List[str], vecs: np.ndarray, meta: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "id": ids,
        "embedding": [v.astype(np.float32).tolist() for v in vecs],
        "meta": [json.dumps(m, ensure_ascii=False) for m in meta],
    })
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for _, r in df.iterrows():
                f.write(json.dumps({"id": r["id"], "embedding": r["embedding"], "meta": json.loads(r["meta"])}) + "\n")
    else:
        # default parquet
        df.to_parquet(path + ".parquet", index=False)


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Chunk files and generate embeddings")
    ap.add_argument("target", help="File or directory")
    ap.add_argument("--glob", default=None, help='Glob pattern within directory (e.g., "*.md")')
    ap.add_argument("--backend", default="auto", choices=["auto","st","hf","openai"], help="Embedding backend")
    ap.add_argument("--model", default="", help="Model name (ST/HF). Leave blank for defaults")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Compute device (for ST/HF)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended)")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--rows-per-chunk", type=int, default=200, help="For CSV/TSV chunking (handled by chunker)")
    ap.add_argument("--kind", default=None, choices=[None, "text", "markdown", "code", "table"], help="Force chunk kind")
    ap.add_argument("--out", required=True, help="Output .parquet or .jsonl")
    ap.add_argument("--faiss", default=None, help="Optional FAISS index path to write")
    ap.add_argument("--meta-out", default=None, help="Optional parquet with (id, meta) for FAISS mapping")
    args = ap.parse_args()

    cfg = EmbeddingConfig(
        backend=args.backend,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True if args.normalize else False,
    )

    embedder = create_embedder(cfg)

    files = gather_inputs(args.target, args.glob)
    all_ids: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    all_vecs: List[np.ndarray] = []

    for p in files:
        try:
            chunks = chunk_file(p, args.kind, args.max_tokens, args.overlap, args.rows_per_chunk)
            if not chunks:
                continue
            vecs, ids, metas = embed_chunks(embedder, chunks)
            all_ids.extend(ids)
            all_meta.extend(metas)
            all_vecs.append(vecs)
            print(f"✓ {p}: {len(chunks)} chunks")
        except Exception as e:
            warnings.warn(f"Skipping {p}: {e}")

    if not all_vecs:
        print("No embeddings generated.", file=sys.stderr)
        sys.exit(2)

    V = np.vstack(all_vecs)
    if cfg.normalize:  # ensure normalized for cosine search
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V = V / norms

    write_embeddings(args.out, all_ids, V, all_meta)
    print(f"✅ wrote embeddings: {args.out} ({V.shape[0]} rows, dim={V.shape[1]})")

    # Optional FAISS
    if args.faiss:
        if not _has_faiss:
            print("faiss not installed; skipping index build", file=sys.stderr)
        else:
            os.makedirs(os.path.dirname(args.faiss), exist_ok=True)
            import faiss  # type: ignore
            index = faiss.IndexFlatIP(V.shape[1])
            index.add(V.astype(np.float32))
            faiss.write_index(index, args.faiss)
            print(f"✅ wrote FAISS index: {args.faiss}")
            if args.meta_out:
                pd.DataFrame({"id": all_ids, "meta": [json.dumps(m) for m in all_meta]}).to_parquet(args.meta_out, index=False)
                print(f"✅ wrote meta: {args.meta_out}")


if __name__ == "__main__":
    main()