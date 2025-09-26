#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_builder.py
----------------
Build a unified FAISS index + metadata store for all your docs/code/csvs.

Pipeline:
  1. Gather files (recursive glob)
  2. Chunk each file (chunker.py)
  3. Embed chunks (embed.py backends: ST/HF/OpenAI)
  4. Write:
     - FAISS index (.faiss)
     - Metadata parquet (.parquet with id, text, meta)
     - Optional embeddings parquet/jsonl

Usage:
  python index_builder.py data/ --glob "*.md" --faiss out/index.faiss --meta out/meta.parquet

  # Multi-type repo crawl
  python index_builder.py repo/ --glob "**/*.*" \
      --faiss out/index.faiss --meta out/meta.parquet --embeddings out/embed.parquet

"""

from __future__ import annotations
import os
import glob
import json
import argparse
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# local imports
from chunker import chunk_any
from embed import EmbeddingConfig, create_embedder, embed_chunks


# =========================================================
# Helpers
# =========================================================

def gather_files(target: str, pattern: Optional[str]) -> List[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        pat = pattern or "**/*"
        files = [f for f in glob.glob(os.path.join(target, pat), recursive=True) if os.path.isfile(f)]
        return files
    raise FileNotFoundError(target)

def write_meta(path: str, ids: List[str], texts: List[str], meta: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "id": ids,
        "text": texts,
        "meta": [json.dumps(m, ensure_ascii=False) for m in meta],
    })
    df.to_parquet(path, index=False)


def build_index(files: List[str], cfg: EmbeddingConfig,
                max_tokens: int, overlap: int, rows_per_chunk: int,
                faiss_path: str, meta_path: str, embeddings_out: Optional[str] = None):
    embedder = create_embedder(cfg)

    all_vecs: List[np.ndarray] = []
    all_ids: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    all_texts: List[str] = []

    for p in files:
        try:
            text = open(p, "r", encoding="utf-8", errors="ignore").read()
            chunks = chunk_any(text, source=p, max_tokens=max_tokens, overlap=overlap)
            if not chunks:
                continue
            vecs, ids, metas = embed_chunks(embedder, chunks)
            all_vecs.append(vecs)
            all_ids.extend(ids)
            all_meta.extend(metas)
            all_texts.extend([c["text"] for c in chunks])
            print(f"✓ {p}: {len(chunks)} chunks")
        except Exception as e:
            warnings.warn(f"Skipping {p}: {e}")

    if not all_vecs:
        raise RuntimeError("No chunks embedded. Nothing to index.")

    V = np.vstack(all_vecs).astype(np.float32)
    # L2 normalize if needed (cosine search)
    if cfg.normalize:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V = V / norms

    # Save FAISS
    try:
        import faiss  # type: ignore
    except ImportError:
        raise RuntimeError("faiss not installed. pip install faiss-cpu")

    index = faiss.IndexFlatIP(V.shape[1])  # cosine similarity
    index.add(V)
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    faiss.write_index(index, faiss_path)
    print(f"✅ wrote FAISS index: {faiss_path}")

    # Save metadata
    write_meta(meta_path, all_ids, all_texts, all_meta)
    print(f"✅ wrote metadata: {meta_path}")

    # Save embeddings (optional)
    if embeddings_out:
        df = pd.DataFrame({
            "id": all_ids,
            "embedding": [v.tolist() for v in V],
            "meta": [json.dumps(m) for m in all_meta],
        })
        if embeddings_out.endswith(".jsonl"):
            with open(embeddings_out, "w", encoding="utf-8") as f:
                for _, r in df.iterrows():
                    f.write(json.dumps({"id": r["id"], "embedding": r["embedding"], "meta": json.loads(r["meta"])}) + "\n")
        else:
            df.to_parquet(embeddings_out, index=False)
        print(f"✅ wrote embeddings: {embeddings_out}")


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Build FAISS index + metadata for docs")
    ap.add_argument("target", help="File or directory")
    ap.add_argument("--glob", default=None, help="Glob pattern (default=**/*)")
    ap.add_argument("--backend", default="auto", choices=["auto","st","hf","openai"])
    ap.add_argument("--model", default="", help="Model name (ST/HF/OpenAI)")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--rows-per-chunk", type=int, default=200)
    ap.add_argument("--faiss", required=True, help="Output FAISS index path")
    ap.add_argument("--meta", required=True, help="Output metadata parquet path")
    ap.add_argument("--embeddings", default=None, help="Optional embeddings parquet/jsonl")
    args = ap.parse_args()

    cfg = EmbeddingConfig(
        backend=args.backend,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        normalize=True if args.normalize else False,
    )

    files = gather_files(args.target, args.glob)
    build_index(files, cfg, args.max_tokens, args.overlap, args.rows_per_chunk,
                args.faiss, args.meta, args.embeddings)


if __name__ == "__main__":
    main()