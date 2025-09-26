# knowledge_graph/loaders.py
"""
Knowledge Graph Loaders
-----------------------
Load nodes and edges into a graph backend from tabular files.

Features
- Neo4j loader (MERGE upserts, uniqueness constraints, batched writes)
- In-memory NetworkX loader for testing
- Read CSV / Parquet / JSON / NDJSON with pandas
- Explicit schema via NodeSpec / EdgeSpec
- Minimal, dependency-light surface

Install
- pip install neo4j pandas pyarrow networkx

Env (Neo4j)
- NEO4J_URI=bolt://localhost:7687
- NEO4J_USER=neo4j
- NEO4J_PASS=pass

Example
-------
from knowledge_graph.loaders import (
    Neo4jLoader, NodeSpec, EdgeSpec, read_table
)

# Define schema
company = NodeSpec(label="Company", key="id", props=["name","sector","country"])
person  = NodeSpec(label="Person",  key="id", props=["name","role"])

works_at = EdgeSpec(
    rel_type="WORKS_AT",
    src_label="Person",  src_key="id",
    dst_label="Company", dst_key="id",
    props=["title","since"]
)

# Read data
df_comp = read_table("data/companies.parquet")
df_pers = read_table("data/people.csv")
df_rel  = read_table("data/works_at.ndjson")

# Load
kg = Neo4jLoader()  # uses env
kg.ensure_constraints([company, person])
kg.load_nodes(df_comp, company)
kg.load_nodes(df_pers, person)
kg.load_edges(df_rel, works_at)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Optional deps guarded
try:
    from neo4j import GraphDatabase, Driver, Transaction  # type: ignore
except Exception:
    GraphDatabase = None
    Driver = None
    Transaction = None

try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None


# ------------------------------- Schema -------------------------------

@dataclass(frozen=True)
class NodeSpec:
    """Describe how to map a table into nodes."""
    label: str                   # e.g., "Company"
    key: str                     # column name that uniquely identifies node
    props: Sequence[str] = field(default_factory=list)  # additional property columns
    # Optional: rename columns {column_in_df: property_name_in_graph}
    rename: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    """Describe how to map a table into relationships."""
    rel_type: str                # e.g., "OWNS"
    src_label: str
    src_key: str                 # column in df that maps to source node key
    dst_label: str
    dst_key: str                 # column in df that maps to target node key
    props: Sequence[str] = field(default_factory=list)
    rename: Dict[str, str] = field(default_factory=dict)


# ----------------------------- I/O Helpers ----------------------------

def read_table(path: str, orient: Optional[str] = None) -> pd.DataFrame:
    """
    Read CSV / Parquet / JSON / NDJSON into a DataFrame.
    - CSV: *.csv
    - Parquet: *.parquet, *.pq
    - JSON (records array): *.json
    - NDJSON (one json per line): *.ndjson, *.jsonl
    """
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(path)
    if p.endswith(".ndjson") or p.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if p.endswith(".json"):
        return pd.read_json(path, orient=orient) # type: ignore
    raise ValueError(f"Unsupported file format: {path}")


def _select_props(row: pd.Series, cols: Iterable[str], rename: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        if c in row:
            v = row[c]
            k = rename.get(c, c)
            # Convert NaN to None for DB storage
            if pd.isna(v):
                v = None
            out[k] = v
    return out


# --------------------------- Neo4j Loader ----------------------------

class Neo4jLoader:
    """
    Loads nodes/edges into Neo4j with MERGE semantics and batched writes.
    Adds uniqueness constraints for node keys.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        batch_size: int = 1_000,
    ):
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver not installed. pip install neo4j")

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASS", "pass")
        self.database = database or os.getenv("NEO4J_DATABASE") or None
        self.batch_size = max(100, batch_size)
        self._driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password)) # type: ignore

    # ---- lifecycle ----

    def close(self):
        self._driver.close()

    # ---- constraints ----

    def ensure_constraints(self, nodes: Iterable[NodeSpec]) -> None:
        """
        Create uniqueness constraints on (label.key) if not present.
        """
        cypher = (
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) "
            "REQUIRE n.`{key}` IS UNIQUE"
        )
        with self._driver.session(database=self.database) as s:
            for ns in nodes:
                s.run(cypher.format(label=ns.label, key=ns.key))

    # ---- nodes ----

    def load_nodes(self, df: pd.DataFrame, spec: NodeSpec) -> int:
        """
        MERGE nodes by (label, key) and set/update properties.
        Returns number of rows processed.
        """
        required = [spec.key]
        _assert_columns(df, required, f"Node:{spec.label}")

        cols = set([spec.key, *spec.props])
        subset = df[list(cols)].copy()

        # rename columns into property names (except key)
        rename = {k: v for k, v in spec.rename.items() if k != spec.key}
        if rename:
            subset = subset.rename(columns=rename)

        # batch upserts
        total = 0
        with self._driver.session(database=self.database) as s:
            for start in range(0, len(subset), self.batch_size):
                chunk = subset.iloc[start : start + self.batch_size]
                payload = []
                key_name = spec.rename.get(spec.key, spec.key)
                for _, r in chunk.iterrows():
                    key_val = r[key_name]
                    props = _select_props(r, [c for c in chunk.columns if c != key_name], rename={})
                    payload.append({"key": key_val, "props": props})

                s.execute_write(self._merge_nodes_tx, spec.label, spec.key, payload)
                total += len(chunk)
        return total

    @staticmethod
    def _merge_nodes_tx(tx: Transaction, label: str, key: str, batch: List[Dict[str, Any]]) -> None: # type: ignore
        """
        Unwinds a list of {key, props} and MERGEs by key.
        """
        query = f"""
        UNWIND $rows AS row
        MERGE (n:`{label}` {{ `{key}`: row.key }})
        SET n += row.props
        """
        tx.run(query, rows=batch)

    # ---- edges ----

    def load_edges(self, df: pd.DataFrame, spec: EdgeSpec) -> int:
        """
        MERGE relationships. Assumes nodes already exist (or will be created implicitly).
        src_key/dst_key columns must exist; additional props are set on the relationship.
        """
        required = [spec.src_key, spec.dst_key]
        _assert_columns(df, required, f"Rel:{spec.rel_type}")

        cols = set([spec.src_key, spec.dst_key, *spec.props])
        subset = df[list(cols)].copy()

        # rename columns into props (not for src/dst join keys)
        rename = {k: v for k, v in spec.rename.items() if k not in {spec.src_key, spec.dst_key}}
        if rename:
            subset = subset.rename(columns=rename)

        total = 0
        with self._driver.session(database=self.database) as s:
            for start in range(0, len(subset), self.batch_size):
                chunk = subset.iloc[start : start + self.batch_size]
                rows: List[Dict[str, Any]] = []
                for _, r in chunk.iterrows():
                    src_val = r[spec.src_key]
                    dst_val = r[spec.dst_key]
                    props = _select_props(r, [c for c in chunk.columns if c not in {spec.src_key, spec.dst_key}], rename={})
                    rows.append({"src": src_val, "dst": dst_val, "props": props})

                s.execute_write(
                    self._merge_edges_tx,
                    spec.src_label, spec.src_key,
                    spec.dst_label, spec.dst_key,
                    spec.rel_type,
                    rows
                )
                total += len(chunk)
        return total

    @staticmethod
    def _merge_edges_tx(
        tx: Transaction, # type: ignore
        src_label: str, src_key: str,
        dst_label: str, dst_key: str,
        rel_type: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        query = f"""
        UNWIND $rows AS row
        MATCH (s:`{src_label}` {{ `{src_key}`: row.src }})
        MATCH (t:`{dst_label}` {{ `{dst_key}`: row.dst }})
        MERGE (s)-[r:`{rel_type}`]->(t)
        SET r += row.props
        """
        tx.run(query, rows=rows)


# ------------------------ In-memory (NetworkX) ------------------------

class InMemoryLoader:
    """
    Lightweight, dependency-free-ish loader using NetworkX for tests/dev.
    Stores node/edge attributes similarly to Neo4jLoader.
    """

    def __init__(self):
        if nx is None:
            raise RuntimeError("networkx not installed. pip install networkx")
        self.G = nx.MultiDiGraph()

    def load_nodes(self, df: pd.DataFrame, spec: NodeSpec) -> int:
        _assert_columns(df, [spec.key], f"Node:{spec.label}")
        cols = set([spec.key, *spec.props])
        for _, r in df[list(cols)].iterrows():
            node_id = f"{spec.label}:{r[spec.key]}"
            props = _select_props(r, cols - {spec.key}, rename=spec.rename)
            props["label"] = spec.label
            props["key"] = r[spec.key]
            self.G.add_node(node_id, **props)
        return len(df)

    def load_edges(self, df: pd.DataFrame, spec: EdgeSpec) -> int:
        _assert_columns(df, [spec.src_key, spec.dst_key], f"Rel:{spec.rel_type}")
        cols = set([spec.src_key, spec.dst_key, *spec.props])
        for _, r in df[list(cols)].iterrows():
            s = f"{spec.src_label}:{r[spec.src_key]}"
            t = f"{spec.dst_label}:{r[spec.dst_key]}"
            props = _select_props(r, cols - {spec.src_key, spec.dst_key}, rename=spec.rename)
            props["type"] = spec.rel_type
            self.G.add_edge(s, t, **props)
        return len(df)


# ------------------------------- Utils -------------------------------

def _assert_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns: {missing}")


# ------------------------------- CLI ---------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load tabular data into a knowledge graph.")
    parser.add_argument("--backend", choices=["neo4j", "memory"], default="neo4j")
    parser.add_argument("--nodes", nargs="*", help="Node file paths (csv/parquet/json/ndjson)")
    parser.add_argument("--node-labels", nargs="*", help="Labels for node files (same length as --nodes)")
    parser.add_argument("--node-key", default="id", help="Key column for nodes (default: id)")
    parser.add_argument("--edges", nargs="*", help="Edge file paths")
    parser.add_argument("--edge-types", nargs="*", help="Rel types for edges (same length as --edges)")
    parser.add_argument("--src-label", default="Source", help="Source node label")
    parser.add_argument("--src-key", default="src_id", help="Source key column in edge file")
    parser.add_argument("--dst-label", default="Target", help="Target node label")
    parser.add_argument("--dst-key", default="dst_id", help="Target key column in edge file")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    if args.backend == "neo4j":
        loader = Neo4jLoader(batch_size=args.batch_size)
    else:
        loader = InMemoryLoader()

    # Simple CLI path: each nodes file → NodeSpec with inferred props
    if args.nodes and args.node_labels:
        if len(args.nodes) != len(args.node_labels):
            raise SystemExit("--nodes and --node-labels must match in length")
        specs: List[NodeSpec] = []
        for path, label in zip(args.nodes, args.node_labels):
            df = read_table(path)
            props = [c for c in df.columns if c != args.node_key]
            spec = NodeSpec(label=label, key=args.node_key, props=props)
            if isinstance(loader, Neo4jLoader):
                loader.ensure_constraints([spec])
            loader.load_nodes(df, spec)

    # Simple CLI path: each edges file → EdgeSpec with inferred props
    if args.edges and args.edge_types:
        if len(args.edges) != len(args.edge_types):
            raise SystemExit("--edges and --edge-types must match in length")
        for path, rtype in zip(args.edges, args.edge_types):
            df = read_table(path)
            props = [c for c in df.columns if c not in {args.src_key, args.dst_key}]
            espec = EdgeSpec(
                rel_type=rtype,
                src_label=args.src_label, src_key=args.src_key,
                dst_label=args.dst_label, dst_key=args.dst_key,
                props=props
            )
            loader.load_edges(df, espec)

    print("Done.")