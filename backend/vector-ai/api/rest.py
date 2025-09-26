# analytics-engine/vector-ai/api/rest.py
"""
FastAPI REST API for vector search + (optional) rerank + knowledge graph.

Endpoints
---------
GET  /health                         -> service liveness
POST /search                         -> semantic search (FAISS) + optional rerank
GET  /doc/{doc_id}                   -> fetch document metadata/text from parquet
GET  /kg/neighbors                   -> KG neighborhood (optional Neo4j)
GET  /kg/correlations                -> KG correlations (optional Neo4j)

Env config
----------
FAISS_INDEX=./indices/live.faiss
META_PATH=./indices/live.parquet
# Optional KG:
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=pass

Install
-------
pip install fastapi uvicorn pandas pyarrow sentence-transformers faiss-cpu
# Optional:
pip install neo4j  # for KG
pip install pydantic-settings
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional deps
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover
    GraphDatabase = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("pip install sentence-transformers") from e


# ============================== Config ===============================

FAISS_PATH = Path(os.getenv("FAISS_INDEX", "./indices/live.faiss"))
META_PATH = Path(os.getenv("META_PATH", "./indices/live.parquet"))
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================== Models ===============================

class SearchRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: int = Field(10, ge=1, le=200, description="Number of results to return")
    rerank: bool = Field(False, description="Enable cross-encoder reranking")
    alpha: float = Field(0.2, ge=0.0, le=1.0, description="Fusion weight on vector score when rerank=true")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filters, e.g. {'ticker':'AAPL','lang':'en'}"
    )

class DocResponse(BaseModel):
    doc_id: str
    source: Optional[str] = None
    text: Optional[str] = None
    vector_score: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[DocResponse]

class Neighbor(BaseModel):
    id: str
    labels: List[str] = Field(default_factory=list)

class Correlation(BaseModel):
    id: str
    weight: Optional[float] = None


# ============================== App Init =============================

app = FastAPI(title="Vector-AI REST", version="1.0.0")

# CORS (open by default; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class AppState:
    def __init__(self):
        # Embedding model
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.embedder.max_seq_length = 512

        # FAISS index
        self.index = None
        if not faiss:
            raise RuntimeError("FAISS not available. pip install faiss-cpu")
        if not FAISS_PATH.exists():
            raise RuntimeError(f"FAISS index not found at {FAISS_PATH}")
        self.index = faiss.read_index(str(FAISS_PATH))

        # Metadata Parquet
        if META_PATH.exists():
            self.meta_df = pd.read_parquet(META_PATH)
            # Ensure stable row order linking FAISS ids to rows: append-only usage recommended.
            self.meta_df = self.meta_df.reset_index(drop=True)
        else:
            self.meta_df = None

        # Optional KG (Neo4j)
        self.kg_driver = None
        if NEO4J_URI and GraphDatabase:
            try:
                self.kg_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            except Exception as e:
                print(f"[WARN] Neo4j init failed: {e}")

        # Optional cross-encoder
        self.cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self.cross_encoder = CrossEncoder(RERANK_MODEL, max_length=512)
        except Exception:
            # Rerank endpoint can still run with rerank=False
            self.cross_encoder = None
            print("[INFO] Cross-encoder not available; rerank will be disabled unless installed.")

state = AppState()


# ============================== Utils ================================

def _apply_filters(rows: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not filters:
        return rows
    out = []
    for r in rows:
        meta = r.get("meta", {})
        ok = True
        for k, v in filters.items():
            rv = meta.get(k) if k in meta else r.get(k)
            if isinstance(v, list):
                if rv not in v:
                    ok = False; break
            else:
                if rv != v:
                    ok = False; break
        if ok:
            out.append(r)
    return out

def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ============================== Endpoints ============================

@app.get("/health")
def health():
    return {"status": "ok", "faiss": bool(state.index is not None), "meta_loaded": bool(state.meta_df is not None)}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if state.index is None:
        raise HTTPException(500, "FAISS index not loaded")

    # Encode query
    q = state.embedder.encode([req.query], normalize_embeddings=True)
    qv = np.asarray(q, dtype="float32")

    # Vector search
    top_k = int(req.top_k)
    scores, ids = state.index.search(qv, top_k)
    scores = scores[0]
    ids = ids[0]

    results: List[Dict[str, Any]] = []
    for idx, sc in zip(ids, scores):
        if idx < 0:
            continue
        row = {"vector_score": float(sc)}
        if state.meta_df is not None:
            try:
                meta = state.meta_df.iloc[int(idx)].to_dict()
            except Exception:
                meta = {}
            # Common fields
            doc_id = meta.get("doc_id") or meta.get("chunk_id") or str(idx)
            source = meta.get("source")
            text = meta.get("text") if "text" in meta else None
            # Keep entire row as meta minus large text if needed
            meta_payload = {k: v for k, v in meta.items() if k not in {"text"}}
            row.update({
                "doc_id": doc_id,
                "source": source,
                "text": text,
                "meta": meta_payload,
            }) # type: ignore
        else:
            row.update({"doc_id": str(idx), "source": None, "text": None, "meta": {}}) # type: ignore
        results.append(row)

    # Apply metadata filters if provided
    results = _apply_filters(results, req.filters)

    # Optional rerank
    if req.rerank:
        if state.cross_encoder is None:
            raise HTTPException(400, "Rerank requested but cross-encoder backend is not installed.")
        if not results:
            return SearchResponse(query=req.query, count=0, results=[])
        texts = [r.get("text") or "" for r in results]
        ce_scores = state.cross_encoder.predict([[req.query, t] for t in texts], show_progress_bar=False)
        ce_scores = np.asarray(ce_scores, dtype=np.float32)
        ce_norm = _minmax(ce_scores)
        vec_norm = _minmax(np.asarray([r["vector_score"] for r in results], dtype=np.float32))
        fused = req.alpha * vec_norm + (1.0 - req.alpha) * ce_norm
        for i, r in enumerate(results):
            r["rerank_score"] = float(ce_scores[i])
            r["fused_score"] = float(fused[i])
        results.sort(key=lambda x: x["fused_score"], reverse=True)

    # Format response
    out = [
        DocResponse(
            doc_id=r["doc_id"],
            source=r.get("source"),
            text=r.get("text"),
            vector_score=float(r.get("fused_score", r["vector_score"])),
            meta=r.get("meta", {}),
        )
        for r in results[:top_k]
    ]
    return SearchResponse(query=req.query, count=len(out), results=out)


@app.get("/doc/{doc_id}", response_model=DocResponse)
def get_doc(doc_id: str):
    if state.meta_df is None:
        raise HTTPException(404, "Metadata store not available")
    df = state.meta_df
    # Prefer exact doc_id match; else try chunk_id
    m = df[df["doc_id"] == doc_id]
    if m.empty and "chunk_id" in df.columns:
        m = df[df["chunk_id"] == doc_id]
    if m.empty:
        raise HTTPException(404, f"Document {doc_id} not found")

    row = m.iloc[0].to_dict()
    text = row.get("text")
    meta_payload = {k: v for k, v in row.items() if k not in {"text"}}
    return DocResponse(
        doc_id=row.get("doc_id") or row.get("chunk_id") or doc_id,
        source=row.get("source"),
        text=text,
        vector_score=None,
        meta=meta_payload,
    )


@app.get("/kg/neighbors", response_model=List[Neighbor])
def kg_neighbors(node_id: str = Query(...), depth: int = Query(1, ge=1, le=6)):
    if state.kg_driver is None:
        raise HTTPException(400, "Knowledge graph not configured")
    q = f"""
    MATCH (n {{id:$id}})-[*1..{depth}]-(m)
    RETURN DISTINCT m.id AS id, labels(m) AS labels
    """
    with state.kg_driver.session() as s:
        rows = s.run(q, id=node_id)
        out = []
        for r in rows:
            data = r.data()
            out.append(Neighbor(id=str(data.get("id")), labels=list(data.get("labels") or [])))
        return out


@app.get("/kg/correlations", response_model=List[Correlation])
def kg_correlations(node_id: str = Query(...)):
    if state.kg_driver is None:
        raise HTTPException(400, "Knowledge graph not configured")
    q = """
    MATCH (n {id:$id})-[:CORRELATED_WITH]->(m)
    RETURN m.id AS id, m.weight AS weight
    """
    with state.kg_driver.session() as s:
        rows = s.run(q, id=node_id)
        out = []
        for r in rows:
            data = r.data()
            out.append(Correlation(id=str(data.get("id")), weight=float(data.get("weight") or 0.0)))
        return out


# ============================== Run (dev) ============================

# For local dev:
# uvicorn analytics-engine.vector_ai.api.rest:app --reload --port 8001
# (note: module path may be analytics-engine.vector-ai.api.rest depending on your loader)