# backend/graph/knowledge_graph.py
from __future__ import annotations

import os, io, json, time, math, threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable, Set

# ---------- Optional deps (graceful) -----------------------------------------
try:
    import ujson as _json  # faster if present
except Exception:
    _json = json  # type: ignore

HAVE_NX = True
try:
    import networkx as nx  # analytics (optional)
except Exception:
    HAVE_NX = False
    nx = None  # type: ignore

HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

# ---------- Env / paths ------------------------------------------------------
REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOG_DIR       = os.getenv("KG_LOG_DIR", "artifacts/kg")
EVENT_LOG     = os.path.join(LOG_DIR, "kg_events.jsonl")
SNAP_PATH     = os.path.join(LOG_DIR, "kg_snapshot.json")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- Core datatypes ---------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

@dataclass
class Node:
    id: str
    type: str
    props: Dict[str, Any] = field(default_factory=dict)
    ts_ms: int = field(default_factory=now_ms)
    ver: int = 1

@dataclass
class Edge:
    src: str
    dst: str
    rel: str
    weight: float = 1.0
    props: Dict[str, Any] = field(default_factory=dict)
    ts_ms: int = field(default_factory=now_ms)
    ver: int = 1

# ---------- KnowledgeGraph core ---------------------------------------------
class KnowledgeGraph:
    """
    In-memory graph with append-only event log, snapshotting, and optional analytics.
    Thread-safe for single-process (coarse lock).
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.nodes: Dict[str, Node] = {}
        self.adj_out: Dict[str, Dict[str, List[Edge]]] = {}  # src -> rel -> [edges]
        self.adj_in:  Dict[str, Dict[str, List[Edge]]] = {}  # dst -> rel -> [edges]

    # ---- persistence --------------------------------------------------------
    def load(self, snapshot: str = SNAP_PATH, events: str = EVENT_LOG) -> None:
        with self._lock:
            # reset
            self.nodes.clear(); self.adj_out.clear(); self.adj_in.clear()
            # load snapshot (optional)
            if os.path.exists(snapshot):
                with open(snapshot, "r", encoding="utf-8") as f:
                    snap = _json.load(f)
                for j in snap.get("nodes", []):
                    self.nodes[j["id"]] = Node(**j)
                for e in snap.get("edges", []):
                    self._attach_edge(Edge(**e))
            # replay events
            if os.path.exists(events):
                with open(events, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            ev = _json.loads(line)
                        except Exception:
                            continue
                        self._apply_event(ev)

    def snapshot(self, path: str = SNAP_PATH) -> None:
        with self._lock:
            data = {
                "ts_ms": now_ms(),
                "nodes": [asdict(n) for n in self.nodes.values()],
                "edges": [asdict(e) for _, rels in self.adj_out.items() for _, es in rels.items() for e in es],
            }
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(data, f)

    def _append_event(self, ev: Dict[str, Any], log_path: str = EVENT_LOG) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(ev, ensure_ascii=False) + "\n")

    def _apply_event(self, ev: Dict[str, Any]) -> None:
        typ = ev.get("type")
        if typ == "upsert_node":
            n = Node(**ev["node"])
            self.nodes[n.id] = n
        elif typ == "upsert_edge":
            e = Edge(**ev["edge"])
            self._attach_edge(e)

    # ---- node / edge API ----------------------------------------------------
    def upsert_node(self, nid: str, ntype: str, **props) -> Node:
        with self._lock:
            n = self.nodes.get(nid)
            if n:
                n.ver += 1
                n.props.update(props or {})
                n.ts_ms = now_ms()
            else:
                n = Node(id=nid, type=ntype, props=dict(props or {}))
                self.nodes[nid] = n
            self._append_event({"type":"upsert_node","node":asdict(n)})
            return n

    def upsert_edge(self, src: str, dst: str, rel: str, weight: float = 1.0, **props) -> Edge:
        with self._lock:
            e = Edge(src=src, dst=dst, rel=rel, weight=float(weight), props=dict(props or {}))
            self._attach_edge(e)
            self._append_event({"type":"upsert_edge","edge":asdict(e)})
            return e

    def _attach_edge(self, e: Edge) -> None:
        self.adj_out.setdefault(e.src, {}).setdefault(e.rel, []).append(e)
        self.adj_in.setdefault(e.dst, {}).setdefault(e.rel, []).append(e)

    # ---- lookup / query -----------------------------------------------------
    def get_node(self, nid: str) -> Optional[Node]:
        return self.nodes.get(nid)

    def neighbors(self, nid: str, *, direction: str = "out", rel: Optional[str] = None) -> List[str]:
        if direction == "in":
            rels = self.adj_in.get(nid, {})
        elif direction == "both":
            ids = set(self.neighbors(nid, direction="out", rel=rel)) | set(self.neighbors(nid, direction="in", rel=rel))
            return sorted(ids)
        else:
            rels = self.adj_out.get(nid, {})
        ids: Set[str] = set()
        for r, edges in rels.items():
            if rel and r != rel: continue
            ids |= { (e.dst if direction != "in" else e.src) for e in edges }
        return sorted(ids)

    def edges(self, src: Optional[str] = None, dst: Optional[str] = None, rel: Optional[str] = None) -> List[Edge]:
        out: List[Edge] = []
        if src:
            rels = self.adj_out.get(src, {})
            for r, es in rels.items():
                if rel and r != rel: continue
                out.extend([e for e in es if (dst is None or e.dst == dst)])
            return out
        # all edges
        for _, rels in self.adj_out.items():
            for r, es in rels.items():
                if rel and r != rel: continue
                out.extend(es)
        if dst:
            out = [e for e in out if e.dst == dst]
        return out

    def find(self, predicate: Callable[[Node], bool]) -> List[Node]:
        return [n for n in self.nodes.values() if predicate(n)]

    def path(self, src: str, dst: str, max_hops: int = 5) -> List[str]:
        """Unweighted BFS path; returns node IDs (empty if none)."""
        if src == dst: return [src]
        frontier = [(src, [src])]
        seen = {src}
        for _ in range(max_hops):
            new_frontier = []
            for nid, path in frontier:
                for nxt in self.neighbors(nid, direction="out"):
                    if nxt in seen: continue
                    if nxt == dst: return path + [nxt]
                    seen.add(nxt)
                    new_frontier.append((nxt, path + [nxt]))
            frontier = new_frontier
        return []

    # ---- analytics (optional via networkx) ----------------------------------
    def to_networkx(self) -> Optional["nx.MultiDiGraph"]: # type: ignore
        if not HAVE_NX:
            return None
        G = nx.MultiDiGraph() # type: ignore
        for n in self.nodes.values():
            G.add_node(n.id, **{"type": n.type, **n.props})
        for _, rels in self.adj_out.items():
            for r, es in rels.items():
                for e in es:
                    G.add_edge(e.src, e.dst, key=f"{e.rel}:{e.ts_ms}", rel=e.rel, weight=e.weight, **e.props)
        return G

    def centrality(self) -> Dict[str, float]:
        if not HAVE_NX:
            # degree centrality fallback
            deg: Dict[str, int] = {k: 0 for k in self.nodes}
            for src, rels in self.adj_out.items():
                for _, es in rels.items():
                    deg[src] += len(es)
                    for e in es:
                        deg[e.dst] = deg.get(e.dst, 0) + 1
            total = max(1, sum(deg.values()))
            return {k: v / total for k, v in deg.items()}
        G = self.to_networkx()
        try:
            c = nx.pagerank(G)  # type: ignore
        except Exception:
            c = nx.degree_centrality(G)  # type: ignore
        return {k: float(v) for k, v in c.items()}

    # ---- explainers ---------------------------------------------------------
    def explain_order(self, order_id: str) -> Dict[str, Any]:
        """
        Walks the graph around an Order node to retrieve: strategy → signals/news → venue/path → risk checks.
        """
        o = self.get_node(order_id)
        if not o or o.type != "Order":
            return {"error": "order not found"}
        out: Dict[str, Any] = {"order": asdict(o), "strategy": None, "signals": [], "news": [], "risk": [], "route": []}
        # Strategy that generated it
        for n in self.neighbors(order_id, direction="in", rel="GENERATED"):
            out["strategy"] = asdict(self.get_node(n)) # type: ignore
        # Signals/news used
        for s in self.neighbors(order_id, direction="in", rel="TRIGGERED_BY"):
            out["signals"].append(asdict(self.get_node(s))) # type: ignore
        for a in self.neighbors(order_id, direction="in", rel="JUSTIFIED_BY"):
            nd = self.get_node(a)
            if nd and nd.type in ("News","Metric"):
                out["news"].append(asdict(nd))
        # Risk checks
        for r in self.neighbors(order_id, direction="out", rel="CHECKED_BY"):
            out["risk"].append(asdict(self.get_node(r))) # type: ignore
        # Route/venue
        for v in self.neighbors(order_id, direction="out", rel="ROUTES_TO"):
            out["route"].append(asdict(self.get_node(v))) # type: ignore
        return out

    # ---- exports (for Neo4j / RedisGraph) -----------------------------------
    def export_neo4j(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns dict with 'nodes' and 'rels' suitable for bulk CSV/JSON import tools.
        """
        nodes = [{"id": n.id, "label": n.type, **n.props} for n in self.nodes.values()]
        rels = [{"src": e.src, "dst": e.dst, "rel": e.rel, "weight": e.weight, **e.props}
                for _, rels in self.adj_out.items() for _, es in rels.items() for e in es]
        return {"nodes": nodes, "rels": rels}

    def export_redisgraph(self) -> List[str]:
        """
        Returns a list of Cypher statements safe for RedisGraph.
        """
        stmts: List[str] = []
        for n in self.nodes.values():
            props = ", ".join([f'{k}: {json.dumps(v)}' for k, v in n.props.items()])
            stmts.append(f'CREATE (:${n.type} {{id: "{n.id}"{(", " + props) if props else ""}}});')
        for _, rels in self.adj_out.items():
            for r, es in rels.items():
                for e in es:
                    props = ", ".join([f'{k}: {json.dumps(v)}' for k, v in e.props.items()])
                    stmts.append(f'MATCH (a {{id:"{e.src}"}}),(b {{id:"{e.dst}"}}) CREATE (a)-[:{e.rel} {{weight:{e.weight}{(", " + props) if props else ""}}}]->(b);')
        return stmts

# ---------- Builders / canonical upserts -------------------------------------
def node_instrument(kg: KnowledgeGraph, symbol: str, **props) -> Node:
    return kg.upsert_node(f"inst:{symbol.upper()}", "Instrument", symbol=symbol.upper(), **props)

def node_strategy(kg: KnowledgeGraph, name: str, **props) -> Node:
    return kg.upsert_node(f"strat:{name}", "Strategy", name=name, **props)

def node_venue(kg: KnowledgeGraph, code: str, **props) -> Node:
    return kg.upsert_node(f"venue:{code}", "Venue", venue=code, **props)

def node_risk(kg: KnowledgeGraph, name: str, **props) -> Node:
    return kg.upsert_node(f"risk:{name}", "Risk", name=name, **props)

def node_news(kg: KnowledgeGraph, nid: str, **props) -> Node:
    return kg.upsert_node(f"news:{nid}", "News", news_id=nid, **props)

def node_order(kg: KnowledgeGraph, oid: str, **props) -> Node:
    return kg.upsert_node(f"order:{oid}", "Order", order_id=oid, **props)

def node_position(kg: KnowledgeGraph, pid: str, **props) -> Node:
    return kg.upsert_node(f"pos:{pid}", "Position", pos_id=pid, **props)

def node_metric(kg: KnowledgeGraph, key: str, **props) -> Node:
    return kg.upsert_node(f"metric:{key}", "Metric", key=key, **props)

def node_person(kg: KnowledgeGraph, name: str, **props) -> Node:
    return kg.upsert_node(f"person:{name}", "Person", name=name, **props)

def node_tag(kg: KnowledgeGraph, tag: str, **props) -> Node:
    return kg.upsert_node(f"tag:{tag}", "Tag", tag=tag, **props)

# ---------- Stream wiring (optional) -----------------------------------------
class StreamIngestor:
    """
    Optional: tail your Redis streams and stitch the graph in real time.
    Streams (config via env):
      - orders.incoming / orders.filled / orders.rejected
      - positions.snapshots
      - features.alt.news
      - regime.state
      - risk.metrics
    """
    def __init__(self, kg: KnowledgeGraph, redis_url: str = REDIS_URL):
        self.kg = kg
        self.url = redis_url
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.last_ids: Dict[str, str] = {
            "orders.incoming": "$",
            "orders.filled": "$",
            "orders.rejected": "$",
            "positions.snapshots": "$",
            "features.alt.news": "$",
            "regime.state": "$",
            "risk.metrics": "$",
        }

    async def connect(self):
        if not HAVE_REDIS: return
        try:
            self.r = AsyncRedis.from_url(self.url, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def run(self):
        if not self.r:
            return
        while True:
            try:
                resp = await self.r.xread(self.last_ids, count=400, block=1500)  # type: ignore
                if not resp: 
                    continue
                for stream, entries in resp:
                    self.last_ids[stream] = entries[-1][0]
                    for _id, fields in entries:
                        j = {}
                        try:
                            j = json.loads(fields.get("json", "{}"))
                        except Exception:
                            continue
                        self._ingest(stream, j)
                # snapshot occasionally
                if int(time.time()) % 30 == 0:
                    self.kg.snapshot()
            except Exception:
                pass

    def _ingest(self, stream: str, j: Dict[str, Any]) -> None:
        if stream.startswith("orders."):
            oid = str(j.get("id") or j.get("order_id") or j.get("ts_ms"))
            sym = str(j.get("symbol","")).upper()
            strat = str(j.get("strategy",""))
            venue = j.get("venue") or j.get("exchange")
            # nodes
            o = node_order(self.kg, oid, **j)
            if sym: 
                inst = node_instrument(self.kg, sym)
                self.kg.upsert_edge(o.id, inst.id, "ON")
            if strat:
                s = node_strategy(self.kg, strat)
                self.kg.upsert_edge(s.id, o.id, "GENERATED", confidence=1.0)
            if venue:
                v = node_venue(self.kg, str(venue))
                self.kg.upsert_edge(o.id, v.id, "ROUTES_TO")
            # risk linkage if any reason
            if j.get("risk_reason"):
                r = node_risk(self.kg, j["risk_reason"])
                self.kg.upsert_edge(o.id, r.id, "CHECKED_BY", ok=False)

        elif stream == "positions.snapshots":
            pid = f'{j.get("book","BOOK")}@{j.get("ts_ms", now_ms())}'
            p = node_position(self.kg, pid, **j)
            for pos in j.get("positions", []):
                sym = str(pos.get("symbol","")).upper()
                if not sym: continue
                inst = node_instrument(self.kg, sym)
                self.kg.upsert_edge(p.id, inst.id, "HOLDS", qty=pos.get("qty"))

        elif stream == "features.alt.news":
            nid = str(j.get("id") or j.get("news_id") or j.get("url") or now_ms())
            n = node_news(self.kg, nid, **j)
            for sym in (j.get("symbols") or []):
                inst = node_instrument(self.kg, str(sym))
                self.kg.upsert_edge(n.id, inst.id, "MENTIONS")
            # tag by topic if present
            for t in (j.get("tags") or []):
                tg = node_tag(self.kg, str(t))
                self.kg.upsert_edge(n.id, tg.id, "TAGGED")

        elif stream == "regime.state":
            key = f"regime@{j.get('ts_ms', now_ms())}"
            m = node_metric(self.kg, key, **j)
            self.kg.upsert_edge(m.id, "metric:regime", "BELONGS_TO")

        elif stream == "risk.metrics":
            key = f"risk@{j.get('ts_ms', now_ms())}"
            r = node_metric(self.kg, key, **j)
            self.kg.upsert_edge(r.id, "metric:risk", "BELONGS_TO")

# ---------- Convenience: tiny CLI -------------------------------------------
def _cli():
    import argparse, asyncio
    ap = argparse.ArgumentParser("knowledge_graph")
    ap.add_argument("--rebuild", action="store_true", help="Drop, reload snapshot + events.")
    ap.add_argument("--snapshot", action="store_true", help="Write snapshot now.")
    ap.add_argument("--centrality", action="store_true", help="Print node centrality.")
    ap.add_argument("--explain-order", type=str, default=None, help="Order ID to explain.")
    ap.add_argument("--stream", action="store_true", help="Start Redis stream ingestor.")
    args = ap.parse_args()

    kg = KnowledgeGraph()
    if args.rebuild or os.path.exists(SNAP_PATH) or os.path.exists(EVENT_LOG):
        kg.load()

    if args.centrality:
        print(json.dumps(kg.centrality(), indent=2))

    if args.explain_order:
        print(json.dumps(kg.explain_order(args.explain_order), indent=2))

    if args.snapshot:
        kg.snapshot()
        print("[kg] snapshot written:", SNAP_PATH)

    if args.stream:
        async def _run():
            if not HAVE_REDIS:
                print("[kg] redis not available"); return
            ing = StreamIngestor(kg)
            await ing.connect()
            await ing.run()
        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    _cli()