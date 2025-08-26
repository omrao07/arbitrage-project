# backend/common/graph.py
"""
Lightweight directed graph + runner utilities (stdlib only).

Use cases in your project
-------------------------
- Wire data pipelines: scrapers -> normalizer -> signal_bus publisher
- Orchestrate trading flow: agents -> coordinator -> router -> adapters
- Represent backtests: loaders -> feature_calc -> strategy -> evaluator

Features
--------
- DirectedGraph: add/remove nodes/edges, cycle detect, topo sort
- Metrics: indegree/outdegree, ancestors/descendants, shortest/longest path (DAG)
- DOT export (Graphviz) + tiny ASCII rendering for quick logs
- GraphRunner: execute nodes in topological layers with a thread pool,
  respecting dependencies; collect timing + results + exceptions.

No external dependencies. Nodes are referenced by string ids; attach any
callable to a node for execution (or leave data=None if you only need topology).
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# ------------------------------ Graph model ----------------------------

@dataclass
class Node:
    id: str
    label: Optional[str] = None
    data: Any = None               # arbitrary payload: callable, config, etc.
    weight_ms: float = 0.0         # optional duration estimate (for critical path / pretty)
    meta: Dict[str, Any] = field(default_factory=dict)


class DirectedGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._succ: Dict[str, Set[str]] = {}
        self._pred: Dict[str, Set[str]] = {}

    # ---- mutation ----
    def add_node(self, node_id: str, *, label: Optional[str] = None, data: Any = None, weight_ms: float = 0.0, **meta) -> Node:
        if node_id in self._nodes:
            n = self._nodes[node_id]
            if label is not None: n.label = label
            if data is not None: n.data = data
            if weight_ms: n.weight_ms = weight_ms
            n.meta.update(meta)
            return n
        n = Node(node_id, label=label, data=data, weight_ms=float(weight_ms or 0.0), meta=dict(meta or {}))
        self._nodes[node_id] = n
        self._succ.setdefault(node_id, set())
        self._pred.setdefault(node_id, set())
        return n

    def add_edge(self, src: str, dst: str) -> None:
        if src not in self._nodes or dst not in self._nodes:
            raise KeyError("add_edge: missing node(s)")
        self._succ[src].add(dst)
        self._pred[dst].add(src)

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        for p in list(self._pred[node_id]):
            self._succ[p].discard(node_id)
        for s in list(self._succ[node_id]):
            self._pred[s].discard(node_id)
        self._succ.pop(node_id, None)
        self._pred.pop(node_id, None)
        self._nodes.pop(node_id, None)

    def remove_edge(self, src: str, dst: str) -> None:
        self._succ.get(src, set()).discard(dst)
        self._pred.get(dst, set()).discard(src)

    # ---- introspection ----
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def edges(self) -> List[Tuple[str, str]]:
        return [(u, v) for u, vs in self._succ.items() for v in vs]

    def successors(self, node_id: str) -> Set[str]:
        return set(self._succ.get(node_id, set()))

    def predecessors(self, node_id: str) -> Set[str]:
        return set(self._pred.get(node_id, set()))

    def indegree(self, node_id: str) -> int:
        return len(self._pred.get(node_id, set()))

    def outdegree(self, node_id: str) -> int:
        return len(self._succ.get(node_id, set()))

    def ancestors(self, node_id: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [node_id]
        while stack:
            u = stack.pop()
            for p in self._pred.get(u, set()):
                if p not in seen:
                    seen.add(p); stack.append(p)
        return seen

    def descendants(self, node_id: str) -> Set[str]:
        seen: Set[str] = set()
        stack = [node_id]
        while stack:
            u = stack.pop()
            for s in self._succ.get(u, set()):
                if s not in seen:
                    seen.add(s); stack.append(s)
        return seen

    # ---- algorithms ----
    def topo_sort(self) -> List[str]:
        indeg: Dict[str, int] = {k: len(v) for k, v in self._pred.items()}
        q: List[str] = [n for n, d in indeg.items() if d == 0]
        out: List[str] = []
        idx = 0
        while idx < len(q):
            u = q[idx]; idx += 1
            out.append(u)
            for v in self._succ.get(u, set()):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(out) != len(self._nodes):
            raise ValueError("Graph has at least one cycle; topological sort impossible")
        return out

    def layers(self) -> List[List[str]]:
        """Return nodes grouped by topo layers (parallelizable sets)."""
        indeg: Dict[str, int] = {k: len(v) for k, v in self._pred.items()}
        layer: List[List[str]] = []
        frontier = [n for n, d in indeg.items() if d == 0]
        used: Set[str] = set()
        while frontier:
            layer.append(list(frontier))
            used.update(frontier)
            nxt: Set[str] = set()
            for u in frontier:
                for v in self._succ.get(u, set()):
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        nxt.add(v)
            frontier = list(nxt)
        if len(used) != len(self._nodes):
            raise ValueError("Graph has at least one cycle; cannot compute layers")
        return layer

    def find_cycle(self) -> Optional[List[str]]:
        """Return a node sequence in a cycle if any (simple DFS), else None."""
        color: Dict[str, int] = {k: 0 for k in self._nodes}  # 0=unseen,1=gray,2=black
        parent: Dict[str, Optional[str]] = {k: None for k in self._nodes}

        def dfs(u: str) -> Optional[List[str]]:
            color[u] = 1
            for v in self._succ.get(u, set()):
                if color[v] == 0:
                    parent[v] = u
                    res = dfs(v)
                    if res: return res
                elif color[v] == 1:
                    # back-edge -> cycle reconstruct
                    cyc = [v, u]
                    w = parent[u]
                    while w is not None and w != v:
                        cyc.append(w)
                        w = parent[w]
                    cyc.reverse()
                    return cyc
            color[u] = 2
            return None

        for n in self._nodes:
            if color[n] == 0:
                r = dfs(n)
                if r: return r
        return None

    # --- path scores (for DAGs) ---
    def longest_path_weight(self) -> Tuple[float, List[str]]:
        """Critical path by weight_ms (requires DAG)."""
        order = self.topo_sort()
        dist: Dict[str, float] = {u: float("-inf") for u in order}
        prev: Dict[str, Optional[str]] = {u: None for u in order}
        # roots: zero indegree
        roots = [u for u in order if self.indegree(u) == 0]
        for r in roots:
            dist[r] = self._nodes[r].weight_ms
        for u in order:
            for v in self._succ.get(u, set()):
                w = dist[u] + self._nodes[v].weight_ms
                if w > dist[v]:
                    dist[v] = w
                    prev[v] = u
        # best sink
        sink = max(order, key=lambda x: dist[x])
        # reconstruct
        path: List[str] = []
        cur: Optional[str] = sink
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return dist[sink], path

    def shortest_path_unweighted(self, src: str, dst: str) -> Optional[List[str]]:
        """BFS shortest path by hops."""
        from collections import deque
        q = deque([src]); parent: Dict[str, Optional[str]] = {src: None}; seen = {src}
        while q:
            u = q.popleft()
            if u == dst:
                path: List[str] = []
                while u is not None:
                    path.append(u); u = parent[u]
                return path[::-1]
            for v in self._succ.get(u, set()):
                if v not in seen:
                    seen.add(v); parent[v] = u; q.append(v)
        return None

    # ---- export ----
    def to_dot(self, *, title: Optional[str] = None) -> str:
        """Graphviz DOT string."""
        lines = ["digraph G {"]
        if title:
            lines.append(f'  label="{_esc(title)}"; labelloc=top; fontsize=20;')
        lines.append('  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef5ff"];')
        for n in self._nodes.values():
            lbl = n.label or n.id
            extra = []
            if n.weight_ms:
                extra.append(f"t≈{int(n.weight_ms)}ms")
            if extra:
                lbl = f"{lbl}\\n" + "\\n".join(extra)
            lines.append(f'  "{_esc(n.id)}" [label="{_esc(lbl)}"];')
        for u, v in self.edges():
            lines.append(f'  "{_esc(u)}" -> "{_esc(v)}";')
        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self) -> str:
        """Very small left‑to‑right ASCII layer dump."""
        try:
            layers = self.layers()
        except Exception as e:
            cyc = self.find_cycle()
            return f"[graph] cannot render layers: {e}. cycle={cyc}"
        parts = ["[graph] layers:"]
        for i, L in enumerate(layers, 1):
            parts.append(f"  L{i}: " + "  ->  ".join(L))
        return "\n".join(parts)


def _esc(s: str) -> str:
    return str(s).replace('"', '\\"')


# ------------------------------ Runner ---------------------------------

@dataclass
class RunResult:
    ok: bool
    results: Dict[str, Any]
    errors: Dict[str, str]
    timings_ms: Dict[str, float]
    critical_ms: float
    critical_path: List[str]


class GraphRunner:
    """
    Execute node callables respecting dependencies.

    Each node with a callable `data` is invoked as:
        result = fn(node, context)
    where `context` is a dict you pass to run(). Non-callable nodes are skipped.

    Concurrency:
    - Nodes in the same topo layer run concurrently (ThreadPoolExecutor).
    - Dependencies are honored across layers.

    Error policy:
    - By default, one node failing does NOT stop the run; dependents that
      require its outputs may still run (you can enforce checks in your fn).
    """

    def __init__(self, graph: DirectedGraph, *, max_workers: int = 8):
        self.g = graph
        self.max_workers = int(max(1, max_workers))
        self._lock = threading.RLock()

    def run(self, *, context: Optional[Dict[str, Any]] = None) -> RunResult:
        ctx = dict(context or {})
        layers = self.g.layers()
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for L in layers:
                futs = {}
                for node_id in L:
                    node = self.g._nodes[node_id]
                    fn = node.data if callable(node.data) else None
                    if fn is None:
                        timings[node_id] = 0.0
                        results.setdefault(node_id, None)
                        continue
                    futs[pool.submit(_safe_call, fn, node, ctx, results)] = node_id

                for fut in as_completed(futs):
                    nid = futs[fut]
                    t1 = time.perf_counter()
                    try:
                        res, elapsed = fut.result()
                        results[nid] = res
                        timings[nid] = elapsed * 1000.0
                    except Exception as e:
                        errors[nid] = str(e)
                        timings[nid] = (time.perf_counter() - t1) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        crit_ms, crit_path = self.g.longest_path_weight() if not errors else (total_ms, [])
        ok = len(errors) == 0
        return RunResult(ok=ok, results=results, errors=errors, timings_ms=timings,
                         critical_ms=crit_ms, critical_path=crit_path)


def _safe_call(fn: Callable[[Node, Dict[str, Any], Dict[str, Any]], Any],
               node: Node, ctx: Dict[str, Any], results: Dict[str, Any]) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(node, ctx) # type: ignore
    return out, (time.perf_counter() - t0)


# ------------------------------ Tiny demo ------------------------------

if __name__ == "__main__":
    # Build a small trading flow graph:
    g = DirectedGraph()
    g.add_node("load_prices", label="Load Prices", weight_ms=20,
               data=lambda n, ctx: {"BTCUSDT": 65000, "AAPL": 210})
    g.add_node("load_signals", label="Load Signals", weight_ms=35,
               data=lambda n, ctx: {"social_sent_btc": 0.35, "mom_z_AAPL": 1.0})
    g.add_node("agents", label="Agents (FX/Equities/Crypto/Comms)", weight_ms=40,
               data=lambda n, ctx: {"orders": [{"symbol": "BTCUSDT", "side": "BUY", "qty": 0.05}]})
    g.add_node("coordinator", label="Coordinator", weight_ms=25,
               data=lambda n, ctx: {"legs": [{"symbol": "BTCUSDT", "side": "BUY", "qty": 0.05}]})
    g.add_node("router", label="ArbRouter", weight_ms=30,
               data=lambda n, ctx: {"report": {"ok": True, "filled_qty": 0.05}})

    # Edges
    for nid in ("load_prices", "load_signals"):
        g.add_edge(nid, "agents")
    g.add_edge("agents", "coordinator")
    g.add_edge("coordinator", "router")

    # Print quick views
    print(g.to_ascii())
    print("\nDOT:\n", g.to_dot(title="Trading Flow"))

    # Run with a thread pool
    runner = GraphRunner(g, max_workers=4)
    res = runner.run(context={"env": "demo"})
    print("\nRun ok:", res.ok, "critical(ms):", int(res.critical_ms))
    print("Timings:", {k: int(v) for k, v in res.timings_ms.items()})
    if res.errors:
        print("Errors:", res.errors)