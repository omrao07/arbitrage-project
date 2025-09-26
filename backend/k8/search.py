#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search.py
---------
Interactive / batch search over a FAISS index + metadata, using the same
embedding backends as embed.py.

Typical usage:
  python search.py --faiss out/index.faiss --meta out/meta.parquet "what is our risk framework"
  python search.py --faiss out/index.faiss --meta out/meta.parquet -k 10 --source "docs/*.md" --kind markdown "cds basis"

Fallback (no FAISS): pass an embeddings parquet/jsonl (from embed.py):
  python search.py --embeddings out/embed.parquet --meta out/meta.parquet "beta estimation"

Options:
  - Multiple queries allowed (positional args)
  - Filters: --source (glob/regex), --kind, --meta-key/--meta-value
  - Export: --json out.json  --csv out.csv
"""

from __future__ import annotations
import os
import re
import sys
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# local imports
from embed import EmbeddingConfig, create_embedder
from chunker import _count_tokens  # optional, only for display

# optional dependency
_HAS_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    _HAS_FAISS = False


# =========================================================
# I/O helpers
# =========================================================

def load_meta(meta_path: str) -> pd.DataFrame:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    df = pd.read_parquet(meta_path)
    # meta stored as JSON string
    if "meta" in df.columns and not isinstance(df["meta"].iloc[0], dict):
        df["meta"] = df["meta"].apply(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
    return df

def load_embeddings(emb_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings parquet/jsonl (as saved by embed.py).
    Returns (V, ids)
    """
    if emb_path.endswith(".jsonl"):
        vecs = []
        ids = []
        with open(emb_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ids.append(row["id"])
                vecs.append(np.array(row["embedding"], dtype=np.float32))
        V = np.vstack(vecs)
        return V, ids
    df = pd.read_parquet(emb_path)
    ids = df["id"].tolist()
    V = np.vstack(df["embedding"].apply(lambda x: np.array(x, dtype=np.float32)).values)#type:ignore
    return V, ids


# =========================================================
# FAISS utilities
# =========================================================

def faiss_search(faiss_path: str, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_FAISS:
        raise RuntimeError("faiss not installed. pip install faiss-cpu")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(faiss_path)
    index = faiss.read_index(faiss_path)
    # queries assumed normalized (cosine/IP)
    D, I = index.search(queries.astype(np.float32), k)
    return D, I


# =========================================================
# Embedding of queries
# =========================================================

def embed_queries(qs: List[str], backend: str, model: str, device: str, batch_size: int, normalize: bool) -> np.ndarray:
    cfg = EmbeddingConfig(
        backend=backend,
        model=model,
        device=device,
        batch_size=batch_size,
        normalize=normalize,
    )
    embedder = create_embedder(cfg)
    vecs = embedder.embed_texts(qs)
    # ensure normalized for cosine
    if not normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
    return vecs.astype(np.float32)


# =========================================================
# Filtering / re-ranking
# =========================================================

def meta_filter(df: pd.DataFrame,
                source_pat: Optional[str],
                kind: Optional[str],
                meta_key: Optional[str],
                meta_value: Optional[str]) -> pd.DataFrame:
    out = df
    if source_pat:
        # allow regex or glob-like
        pat = source_pat.replace(".", r"\.").replace("*", ".*")
        out = out[out["meta"].apply(lambda m: bool(re.search(pat, (m.get("source") or ""), re.IGNORECASE)))]
    if kind:
        out = out[out["meta"].apply(lambda m: (m.get("kind") or "") == kind)]
    if meta_key:
        out = out[out["meta"].apply(lambda m: str(m.get(meta_key, "")) == (meta_value or ""))]
    return out

def rerank(scores: np.ndarray,
           texts: List[str],
           metas: List[Dict[str, Any]],
           boosts: Optional[Dict[str, float]] = None,
           length_penalty: float = 0.0) -> np.ndarray:
    """
    Adjust scores with:
      - metadata boosts: e.g., {'kind=markdown': +0.05, 'source=.md': +0.03} # type: ignore # type: ignore
      - length penalty: subtract penalty * log(tokens)
    Returns adjusted scores array.
    """
    adj = scores.copy().astype(np.float32)
    # length penalty
    if length_penalty > 0:
        for i, t in enumerate(texts):
            toks = max(1, _count_tokens(t))
            adj[i] -= length_penalty * math.log(toks)
    # meta boosts
    if boosts:
        for i, m in enumerate(metas):
            combined = []
            if "kind" in m:
                combined.append(f"kind={m['kind']}")
            if "source" in m and isinstance(m["source"], str):
                combined.append(f"source={m['source']}")
            for rule, bump in boosts.items():
                if rule.startswith("kind="):
                    if f"kind={m.get('kind','')}" == rule:
                        adj[i] += bump
                elif rule.startswith("source=") and "source" in m:
                    pat = rule.split("=",1)[1]
                    try:
                        if re.search(pat, m["source"], re.IGNORECASE):
                            adj[i] += bump
                    except re.error:
                        pass
    return adj


# =========================================================
# Pretty printing
# =========================================================

def highlight(text: str, terms: List[str], width: int = 96) -> str:
    try:
        import shutil
        width = shutil.get_terminal_size((width, 20)).columns
    except Exception:
        pass
    esc_on, esc_off = "\033[93m", "\033[0m"  # yellow
    snippet = text.replace("\n", " ")
    # crude highlight
    for t in sorted(set([t for t in terms if t.strip()]), key=len, reverse=True):
        try:
            snippet = re.sub(f"({re.escape(t)})", rf"{esc_on}\1{esc_off}", snippet, flags=re.IGNORECASE)
        except re.error:
            continue
    return (snippet[:width-3] + "...") if len(snippet) > width else snippet


# =========================================================
# Runner
# =========================================================

def run(queries: List[str],
        faiss_path: Optional[str],
        meta_path: str,
        embeddings_path: Optional[str],
        backend: str, model: str, device: str, batch_size: int,
        k: int,
        source_pat: Optional[str],
        kind: Optional[str],
        meta_key: Optional[str],
        meta_value: Optional[str],
        length_penalty: float,
        boosts_path: Optional[str],
        out_json: Optional[str],
        out_csv: Optional[str]) -> int:

    meta_df = load_meta(meta_path)

    # Optional embeddings fallback if FAISS not present
    V, ids = (None, None)
    if (not faiss_path) and embeddings_path:
        V, ids = load_embeddings(embeddings_path)
        # Normalize for cosine
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V = (V / norms).astype(np.float32)
        # align meta to embeddings id order
        meta_df = meta_df.set_index("id").loc[ids].reset_index()

    # Apply coarse metadata filters ahead of search (only if we have embeddings and can subselect)
    if V is not None:
        before = len(meta_df)
        meta_df = meta_filter(meta_df, source_pat, kind, meta_key, meta_value)
        mask = meta_df["id"].isin(ids)#type:ignore
        sub_idx = np.where(mask.values)[0]#type:ignore
        if sub_idx.size and sub_idx.size < V.shape[0]:
            # restrict to filtered subset
            keep_ids = meta_df["id"].tolist()
            idx_map = {i: j for j, i in enumerate(ids)}#type:ignore
            take = np.array([idx_map[i] for i in keep_ids], dtype=np.int64)
            V = V[take]
            ids = [ids[t] for t in take]#type:ignore
        after = len(meta_df)
        if before != after:
            print(f"Applied metadata pre-filter: {before} → {after}")

    # Load boosts
    boosts = None
    if boosts_path and os.path.exists(boosts_path):
        with open(boosts_path, "r", encoding="utf-8") as f:
            boosts = json.load(f)  # dict[str, float]

    # Embed queries
    Q = embed_queries(queries, backend, model, device, batch_size, normalize=True)

    results_all: List[Dict[str, Any]] = []

    for qi, q in enumerate(queries):
        print(f"\n=== Query {qi+1}/{len(queries)}: {q!r} ===")

        # Search
        if faiss_path:
            D, I = faiss_search(faiss_path, Q[qi:qi+1], k=max(k, 50))  # fetch more; we’ll re-rank later
            scores = D[0]
            idxs = I[0]
            # map to meta rows
            cand = meta_df.iloc[idxs]
        else:
            # brute force cosine
            sims = (V @ Q[qi].reshape(-1))
            idxs = np.argsort(-sims)[:max(k, 50)]
            scores = sims[idxs]
            cand = meta_df.iloc[idxs]

        # Optional post-filter on metadata (if FAISS used we filter now; if V filtered earlier it’s already narrow)
        if faiss_path:
            cand = meta_filter(cand, source_pat, kind, meta_key, meta_value)
            # need to update scores/idcs to match filtered cand
            # easiest: rebuild from mapping
            new_scores, new_idxs = [], []
            for r in cand.itertuples():
                pos = np.where(cand.index.values == r.Index)[0]
                # but cand reindexed; fallback to original ids
                pass  # We will align by id below
            # align by id
            id2score = {meta_df.iloc[i]["id"]: float(scores[j]) for j, i in enumerate(idxs)}
            scores = np.array([id2score[i] for i in cand["id"].tolist()], dtype=np.float32)

        # Re-rank (length penalty + boosts)
        adj = rerank(scores, cand["text"].tolist(), cand["meta"].tolist(),
                     boosts=boosts, length_penalty=length_penalty)
        order = np.argsort(-adj)[:k]
        cand = cand.iloc[order]
        adj = adj[order]

        # Pretty print
        terms = re.findall(r"\w+", q)
        for rank, (row, sc) in enumerate(zip(cand.itertuples(index=False), adj), start=1):
            meta = row.meta or {}
            src = meta.get("source", "")#type:ignore
            kind_str = meta.get("kind", "")#type:ignore
            sect = meta.get("section", "") or meta.get("symbol", "") or ""#type:ignore
            snip = highlight(row.text, terms)#type:ignore
            print(f"{rank:>2}. score={sc:>6.3f}  kind={kind_str:<9}  src={src}")
            if sect:
                print(f"    section: {sect}")
            print(f"    {snip}\n")

            results_all.append({
                "query": q,
                "rank": rank,
                "score": float(sc),
                "id": row.id,
                "source": src,
                "kind": kind_str,
                "section": sect,
                "text": row.text,
                "meta": meta,
            })

    # Exports
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results_all, f, ensure_ascii=False, indent=2)
        print(f"\n✅ wrote JSON: {out_json}")
    if out_csv:
        pd.DataFrame(results_all).to_csv(out_csv, index=False)
        print(f"✅ wrote CSV: {out_csv}")

    return 0


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Search FAISS index + metadata")
    ap.add_argument("query", nargs="+", help="One or more queries")
    ap.add_argument("--faiss", default=None, help="Path to index.faiss")
    ap.add_argument("--meta", required=True, help="Path to meta.parquet (id,text,meta)")
    ap.add_argument("--embeddings", default=None, help="Embeddings parquet/jsonl (fallback if no FAISS)")
    ap.add_argument("-k", type=int, default=5, help="Top-k results")
    # backend config (for query embeddings)
    ap.add_argument("--backend", default="auto", choices=["auto","st","hf","openai"])
    ap.add_argument("--model", default="", help="Embedding model (ST/HF/OpenAI)")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--batch-size", type=int, default=32)
    # filters
    ap.add_argument("--source", default=None, help="Regex/glob on meta.source (e.g., 'docs/.*\\.md')")
    ap.add_argument("--kind", default=None, help="Filter by meta.kind (text|markdown|code|table)")
    ap.add_argument("--meta-key", default=None, help="Filter: meta key equals value")
    ap.add_argument("--meta-value", default=None, help="Filter: value for --meta-key")
    # ranking tweaks
    ap.add_argument("--length-penalty", type=float, default=0.0, help="Subtract c*log(tokens) from score")
    ap.add_argument("--boosts", default=None, help="JSON file: {'kind=markdown':0.05,'source=.*md':0.03}")
    # exports
    ap.add_argument("--json", dest="out_json", default=None)
    ap.add_argument("--csv", dest="out_csv", default=None)
    args = ap.parse_args()

    if not args.faiss and not args.embeddings:
        ap.error("Provide --faiss or --embeddings")

    sys.exit(run(
        queries=args.query,
        faiss_path=args.faiss,
        meta_path=args.meta,
        embeddings_path=args.embeddings,
        backend=args.backend,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        k=args.k,
        source_pat=args.source,
        kind=args.kind,
        meta_key=args.meta_key,
        meta_value=args.meta_value,
        length_penalty=args.length_penalty,
        boosts_path=args.boosts,
        out_json=args.out_json,
        out_csv=args.out_csv,
    ))


if __name__ == "__main__":
    main()