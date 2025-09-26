# analytics-engine/vector-ai/api/graphql.py
"""
GraphQL API for vector search + knowledge graph.

Features
--------
- search(query, top_k, alpha): semantic search with optional rerank fusion
- getDoc(id): fetch doc metadata from parquet
- kgNeighbors(nodeId, depth): neighborhood expansion (knowledge graph)
- kgCorrelations(nodeId): entity correlations (knowledge graph)

Backends
--------
- FAISS (local) / Pinecone (cloud)
- Parquet metadata store
- Neo4j (for KG queries, optional)

Usage
-----
pip install strawberry-graphql[fastapi] fastapi uvicorn pandas faiss-cpu
# (optional) pip install openai pinecone-client neo4j sentence-transformers

Run:
uvicorn analytics-engine.vector-ai.api.graphql:app --reload --port 8000

Query:
http://localhost:8000/graphql
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Optional

import strawberry # type: ignore
from strawberry.asgi import GraphQL # type: ignore
import pandas as pd

# optional: FAISS + Pinecone
try:
    import faiss # type: ignore
except ImportError:
    faiss = None

# ================ Vector Index Layer ====================

class VectorIndex:
    def __init__(self, faiss_path: Optional[str] = None, meta_path: Optional[str] = None):
        self.faiss_path = Path(faiss_path) if faiss_path else None
        self.meta_path = Path(meta_path) if meta_path else None
        self.index = None
        if self.faiss_path and self.faiss_path.exists() and faiss:
            self.index = faiss.read_index(str(self.faiss_path))
        self.meta_df = pd.read_parquet(self.meta_path) if self.meta_path and self.meta_path.exists() else None

    def search(self, query_vec, top_k: int = 10) -> List[dict]:
        if not self.index:
            raise RuntimeError("FAISS index not loaded")
        import numpy as np
        q = query_vec.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(q, top_k)
        out = []
        for i, s in zip(ids[0], scores[0]):
            if i < 0: continue
            row = {"chunk_id": str(i), "vector_score": float(s)}
            if self.meta_df is not None:
                meta = self.meta_df.iloc[int(i)].to_dict()
                row.update(meta)
            out.append(row)
        return out

    def get_doc(self, doc_id: str) -> Optional[dict]:
        if self.meta_df is None:
            return None
        matches = self.meta_df[self.meta_df["doc_id"] == doc_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()


# ================ Knowledge Graph Layer ====================

class KnowledgeGraph:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        if uri:
            try:
                from neo4j import GraphDatabase # type: ignore
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
            except Exception as e:
                print(f"[WARN] Neo4j unavailable: {e}")

    def neighbors(self, node_id: str, depth: int = 1) -> List[dict]:
        if not self.driver:
            return []
        q = f"""
        MATCH (n {{id:$id}})-[*1..{depth}]-(m)
        RETURN m.id as id, labels(m) as labels
        """
        with self.driver.session() as s:
            rows = s.run(q, id=node_id)
            return [r.data() for r in rows]

    def correlations(self, node_id: str) -> List[dict]:
        if not self.driver:
            return []
        q = """
        MATCH (n {id:$id})-[:CORRELATED_WITH]->(m)
        RETURN m.id as id, m.weight as weight
        """
        with self.driver.session() as s:
            rows = s.run(q, id=node_id)
            return [r.data() for r in rows]


# ================ GraphQL Schema ====================

@strawberry.type
class Document:
    doc_id: str
    source: Optional[str]
    text: Optional[str]
    vector_score: Optional[float]


@strawberry.type
class Query:
    @strawberry.field
    def search(self, query: str, top_k: int = 10, alpha: float = 0.2) -> List[Document]:
        """
        Perform semantic search. Currently uses vector-only; rerank fusion can be added here.
        """
        # embed query
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vec = model.encode([query], normalize_embeddings=True)
        results = index.search(vec[0], top_k=top_k)
        out = []
        for r in results:
            out.append(Document(
                doc_id=r.get("doc_id", r.get("chunk_id")),# type: ignore
                source=r.get("source"),# type: ignore
                text=r.get("text"),# type: ignore
                vector_score=r.get("vector_score"),# type: ignore
            ))
        return out

    @strawberry.field
    def getDoc(self, id: str) -> Optional[Document]:
        doc = index.get_doc(id)
        if not doc:
            return None
        return Document(
            doc_id=doc.get("doc_id"), # type: ignore
            source=doc.get("source"),# type: ignore
            text=doc.get("text"),# type: ignore
            vector_score=None,# type: ignore
        )

    @strawberry.field
    def kgNeighbors(self, node_id: str, depth: int = 1) -> List[str]:
        rows = kg.neighbors(node_id, depth)
        return [json.dumps(r) for r in rows]

    @strawberry.field
    def kgCorrelations(self, node_id: str) -> List[str]:
        rows = kg.correlations(node_id)
        return [json.dumps(r) for r in rows]


# ================ App Init ====================

# config
FAISS_PATH = os.getenv("FAISS_INDEX", "./indices/live.faiss")
META_PATH = os.getenv("META_PATH", "./indices/live.parquet")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

index = VectorIndex(FAISS_PATH, META_PATH)
kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASS)

schema = strawberry.Schema(query=Query)
app = GraphQL(schema)