# analytics-engine/vector-ai/api/websocket.py
"""
FastAPI WebSocket for real-time vector search streaming.

Protocol (JSON messages)
------------------------
Client -> Server:
  { "type": "search", "id": "req-1", "query": "yen carry unwind",
    "top_k": 25, "rerank": true, "alpha": 0.3,
    "filters": {"ticker": "AAPL"} }

  { "type": "doc", "id": "req-2", "doc_id": "abc123" }

  { "type": "kg_neighbors", "id": "req-3", "node_id": "AAPL", "depth": 2 }

  { "type": "cancel", "id": "req-1" }  # cancel an in-flight request

  { "type": "ping" }  # heartbeat

Server -> Client:
  { "type": "ack", "id": "req-1" }
  { "type": "partial", "id": "req-1", "data": [<DocResponse...>], "done": false }
  { "type": "update", "id": "req-1", "data": [<DocResponse...>], "phase": "reranked" }
  { "type": "done", "id": "req-1", "count": 25 }
  { "type": "doc", "id": "req-2", "data": <DocResponse> }
  { "type": "neighbors", "id": "req-3", "data": [ {id, labels[]} ] }
  { "type": "pong" }
  { "type": "error", "id": "req-1", "message": "..." }

Auth (optional)
---------------
Pass a Bearer token as a query param or header:
  ws://host/ws?token=YOUR_TOKEN
Or Sec-WebSocket-Protocol: bearer, YOUR_TOKEN

Env
---
FAISS_INDEX=./indices/live.faiss
META_PATH=./indices/live.parquet
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
NEO4J_URI=bolt://localhost:7687 (optional)
NEO4J_USER=neo4j
NEO4J_PASS=pass

Run
---
uvicorn analytics-engine.vector_ai.api.websocket:app --reload --port 8002
"""

from __future__ import annotations

import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse

# Optional deps
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None

from sentence_transformers import SentenceTransformer

# ----------------------------- Config -----------------------------

FAISS_PATH = Path(os.getenv("FAISS_INDEX", "./indices/live.faiss"))
META_PATH = Path(os.getenv("META_PATH", "./indices/live.parquet"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
AUTH_TOKEN = os.getenv("WS_TOKEN")  # optional shared secret

# --------------------------- App State ----------------------------

app = FastAPI(title="Vector-AI WebSocket", version="1.0.0")

class State:
    def __init__(self):
        if not faiss:
            raise RuntimeError("FAISS not available. pip install faiss-cpu")
        if not FAISS_PATH.exists():
            raise RuntimeError(f"FAISS index not found at {FAISS_PATH}")

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.embedder.max_seq_length = 512

        self.index = faiss.read_index(str(FAISS_PATH))

        self.meta_df = pd.read_parquet(META_PATH).reset_index(drop=True) if META_PATH.exists() else None

        # Optional KG
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
            print("[INFO] Cross-encoder not available; rerank updates will be disabled.")

state = State()

# --------------------------- Utilities ----------------------------

def _ok_auth(token: Optional[str]) -> bool:
    if AUTH_TOKEN is None:
        return True  # auth not enforced
    return token == AUTH_TOKEN

def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)

def _apply_filters(rows: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not filters:
        return rows
    out: List[Dict[str, Any]] = []
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

async def _send(ws: WebSocket, msg: Dict[str, Any]):
    await ws.send_text(json.dumps(msg, ensure_ascii=False))

def _faiss_search(query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    scores, ids = state.index.search(query_vec, top_k)
    scores = scores[0]; ids = ids[0]
    out: List[Dict[str, Any]] = []
    for idx, sc in zip(ids, scores):
        if idx < 0: continue
        row = {"vector_score": float(sc)}
        if state.meta_df is not None:
            try:
                meta = state.meta_df.iloc[int(idx)].to_dict()
            except Exception:
                meta = {}
            doc_id = meta.get("doc_id") or meta.get("chunk_id") or str(idx)
            text = meta.get("text") if "text" in meta else None
            meta_payload = {k: v for k, v in meta.items() if k != "text"}
            row.update({"doc_id": doc_id, "text": text, "meta": meta_payload}) # type: ignore
        else:
            row.update({"doc_id": str(idx), "text": None, "meta": {}}) # type: ignore
        out.append(row)
    return out

def _format_docs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # minimize payload; clients can render meta if needed
    formatted = []
    for r in rows:
        formatted.append({
            "doc_id": r["doc_id"],
            "vector_score": r.get("vector_score"),
            "text": r.get("text"),
            "meta": r.get("meta", {}),
        })
    return formatted

# --------------------------- WebSocket ----------------------------

@app.get("/")
async def home():
    # simple test page
    return HTMLResponse("""
<!doctype html><meta charset="utf-8">
<h3>Vector-AI WS test</h3>
<script>
let ws = new WebSocket("ws://" + location.host + "/ws");
ws.onopen = () => {
  console.log("open");
  ws.send(JSON.stringify({type:"search", id:"req-1", query:"carry trade", top_k:10, rerank:false}));
};
ws.onmessage = (ev) => console.log("msg", ev.data);
</script>
""")

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, token: Optional[str] = Query(None)):
    # Try headers protocol too: "bearer, TOKEN"
    proto = websocket.headers.get("sec-websocket-protocol", "")
    parts = [p.strip() for p in proto.split(",") if p.strip()]
    header_token = parts[1] if len(parts) >= 2 and parts[0].lower() == "bearer" else None

    if not _ok_auth(token or header_token):
        await websocket.close(code=4401)
        return

    await websocket.accept(subprotocol="json")

    # Track cancellable tasks by id
    in_flight: Dict[str, asyncio.Task] = {}

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await _send(websocket, {"type": "error", "message": "invalid JSON"})
                continue

            mtype = msg.get("type")
            rid = msg.get("id")

            if mtype == "ping":
                await _send(websocket, {"type": "pong"})
                continue

            if mtype == "cancel" and rid:
                t = in_flight.pop(rid, None)
                if t and not t.done():
                    t.cancel()
                    await _send(websocket, {"type": "done", "id": rid, "count": 0})
                continue

            if mtype == "doc":
                doc_id = msg.get("doc_id")
                if not doc_id:
                    await _send(websocket, {"type": "error", "id": rid, "message": "doc_id required"})
                    continue
                if state.meta_df is None:
                    await _send(websocket, {"type": "error", "id": rid, "message": "metadata store unavailable"})
                    continue
                df = state.meta_df
                sel = df[df["doc_id"] == doc_id]
                if sel.empty and "chunk_id" in df.columns:
                    sel = df[df["chunk_id"] == doc_id]
                if sel.empty:
                    await _send(websocket, {"type": "error", "id": rid, "message": "not found"})
                    continue
                row = sel.iloc[0].to_dict()
                meta = {k: v for k, v in row.items() if k != "text"}
                data = {"doc_id": row.get("doc_id") or row.get("chunk_id") or doc_id,
                        "text": row.get("text"), "meta": meta}
                await _send(websocket, {"type": "doc", "id": rid, "data": data})
                continue

            if mtype == "kg_neighbors":
                if state.kg_driver is None:
                    await _send(websocket, {"type": "neighbors", "id": rid, "data": []})
                    continue
                node_id = msg.get("node_id")
                depth = int(msg.get("depth", 1))
                q = f"MATCH (n {{id:$id}})-[*1..{depth}]-(m) RETURN DISTINCT m.id AS id, labels(m) AS labels"
                with state.kg_driver.session() as s:
                    rows = s.run(q, id=node_id)
                    data = [{"id": r["id"], "labels": list(r["labels"] or [])} for r in rows]
                await _send(websocket, {"type": "neighbors", "id": rid, "data": data})
                continue

            if mtype == "search":
                # spin a task so client can cancel
                task = asyncio.create_task(_handle_search(websocket, msg, rid))
                in_flight[rid or f"anon-{id(task)}"] = task
                await _send(websocket, {"type": "ack", "id": rid})
                continue

            await _send(websocket, {"type": "error", "id": rid, "message": f"unknown type {mtype}"})

    except WebSocketDisconnect:
        # cancel all tasks
        for t in in_flight.values():
            if not t.done():
                t.cancel()

# ------------------------ Search Handling ------------------------

async def _handle_search(ws: WebSocket, msg: Dict[str, Any], rid: Optional[str]):
    try:
        query: str = msg.get("query", "")
        if not query:
            await _send(ws, {"type": "error", "id": rid, "message": "query required"})
            return

        top_k: int = int(msg.get("top_k", 10))
        rerank: bool = bool(msg.get("rerank", False))
        alpha: float = float(msg.get("alpha", 0.2))
        filters: Optional[Dict[str, Any]] = msg.get("filters")

        # Encode query
        qv = state.embedder.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")

        # First-stage ANN search
        results = _faiss_search(qv, top_k)
        results = _apply_filters(results, filters)

        # Stream partial (vector-ranked)
        await _send(ws, {
            "type": "partial", "id": rid, "done": False,
            "data": _format_docs(results)
        })

        # Optional rerank update
        if rerank:
            if state.cross_encoder is None:
                await _send(ws, {"type": "error", "id": rid, "message": "rerank backend unavailable"})
            else:
                texts = [r.get("text") or "" for r in results]
                ce_scores = state.cross_encoder.predict([[query, t] for t in texts], show_progress_bar=False)
                ce_scores = np.asarray(ce_scores, dtype=np.float32)
                ce_norm = _minmax(ce_scores)
                vec_norm = _minmax(np.asarray([r["vector_score"] for r in results], dtype=np.float32))
                fused = alpha * vec_norm + (1.0 - alpha) * ce_norm
                for i, r in enumerate(results):
                    r["fused_score"] = float(fused[i])
                results.sort(key=lambda x: x["fused_score"], reverse=True)
                await _send(ws, {
                    "type": "update", "id": rid, "phase": "reranked",
                    "data": _format_docs(results)
                })

        await _send(ws, {"type": "done", "id": rid, "count": len(results)})

    except asyncio.CancelledError:
        # search cancelled by client
        pass
    except Exception as e:
        await _send(ws, {"type": "error", "id": rid, "message": str(e)})