// query/lineage.ts
// Zero-dependency lineage graph utilities.
// Direction convention: an edge { from, to } means "to DERIVES FROM from"
// i.e., data/logic flows from `from` ➜ `to`.
//
// What you get:
//   • createGraph(nodes, edges) → LineageGraph (indexes + adjacency built)
//   • addNode, addEdge (dedup-safe)
//   • upstreamSet / downstreamSet (impact analysis)
//   • subgraphAround(id, hopsUp, hopsDown)
//   • topologicalOrder (Kahn). If cycles exist, returns partial order + leftover
//   • findCycles (Tarjan strongly-connected components; size>1 or self-loop)
//   • pathsBetween(src, dst, {maxDepth,maxPaths,allowedEdgeTypes})
//   • filter/prune helpers, merging, DOT/Mermaid exporters
//
// Types are flexible: add whatever metadata you like under `.meta`.

export type Dict<T = any> = { [k: string]: T };

export type NodeKind =
  | "dataset" | "field" | "table" | "view" | "column"
  | "provider" | "job" | "task" | "metric" | "model"
  | "source" | "sink" | "other";

export type EdgeKind =
  | "derives" | "reads" | "writes" | "joins" | "aggregates"
  | "transforms" | "maps" | "filters" | "loads" | "exports" | "depends";

export type LineageNode = {
  id: string;               // unique id (namespace it if you need)
  kind?: NodeKind;
  label?: string;           // human label; default = id
  tags?: string[];
  meta?: Dict;              // arbitrary metadata (owner, domain, SLA…)
};

export type LineageEdge = {
  from: string;
  to: string;
  type?: EdgeKind | string; // allow custom edge kinds
  label?: string;           // human label for the edge (optional)
  meta?: Dict;
};

export type LineageGraph = {
  nodes: Dict<LineageNode>;       // id -> node
  edges: LineageEdge[];           // deduped edges
  outAdj: Dict<string[]>;         // id -> neighbors (outgoing)
  inAdj: Dict<string[]>;          // id -> neighbors (incoming)
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Construction
 * ────────────────────────────────────────────────────────────────────────── */

export function createGraph(nodes: Array<LineageNode | string>, edges: LineageEdge[]): LineageGraph {
  const map: Dict<LineageNode> = Object.create(null);
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const id = typeof n === "string" ? n : n.id;
    if (!id) continue;
    if (!map[id]) {
      if (typeof n === "string") map[id] = { id, label: n, kind: "other" };
      else map[id] = { id, label: n.label || n.id, kind: n.kind || "other", tags: n.tags || [], meta: n.meta };
    } else if (typeof n !== "string") {
      // merge metadata shallowly
      const cur = map[id];
      cur.label = n.label || cur.label;
      cur.kind = n.kind || cur.kind;
      if (n.tags && n.tags.length) cur.tags = uniq([...(cur.tags || []), ...n.tags]);
      if (n.meta) cur.meta = { ...(cur.meta || {}), ...n.meta };
    }
  }

  // Deduplicate edges and ensure endpoints exist
  const seen = new Set<string>();
  const dedup: LineageEdge[] = [];
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    if (!e || !e.from || !e.to) continue;
    if (!map[e.from]) map[e.from] = { id: e.from, label: e.from, kind: "other" };
    if (!map[e.to]) map[e.to] = { id: e.to, label: e.to, kind: "other" };
    const k = edgeKey(e);
    if (seen.has(k)) continue;
    seen.add(k);
    dedup.push({ from: e.from, to: e.to, type: e.type || "derives", label: e.label, meta: e.meta });
  }

  const outAdj: Dict<string[]> = Object.create(null);
  const inAdj: Dict<string[]> = Object.create(null);
  for (const id in map) { outAdj[id] = []; inAdj[id] = []; }
  for (let i = 0; i < dedup.length; i++) {
    const e = dedup[i];
    outAdj[e.from].push(e.to);
    inAdj[e.to].push(e.from);
  }

  return { nodes: map, edges: dedup, outAdj, inAdj };
}

export function addNode(g: LineageGraph, node: LineageNode | string): void {
  const id = typeof node === "string" ? node : node.id;
  if (!id) return;
  if (!g.nodes[id]) {
    g.nodes[id] = typeof node === "string" ? { id, label: id, kind: "other" } : { id, label: node.label || id, kind: node.kind || "other", tags: node.tags || [], meta: node.meta };
    g.outAdj[id] = []; g.inAdj[id] = [];
  } else if (typeof node !== "string") {
    const cur = g.nodes[id];
    cur.label = node.label || cur.label;
    cur.kind = node.kind || cur.kind;
    if (node.tags && node.tags.length) cur.tags = uniq([...(cur.tags || []), ...node.tags]);
    if (node.meta) cur.meta = { ...(cur.meta || {}), ...node.meta };
  }
}

export function addEdge(g: LineageGraph, edge: LineageEdge): void {
  if (!edge || !edge.from || !edge.to) return;
  if (!g.nodes[edge.from]) addNode(g, edge.from);
  if (!g.nodes[edge.to]) addNode(g, edge.to);
  const k = edgeKey(edge);
  for (let i = 0; i < g.edges.length; i++) {
    const e = g.edges[i];
    if (edgeKey(e) === k) return; // already present
  }
  g.edges.push({ from: edge.from, to: edge.to, type: edge.type || "derives", label: edge.label, meta: edge.meta });
  g.outAdj[edge.from].push(edge.to);
  g.inAdj[edge.to].push(edge.from);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Queries
 * ────────────────────────────────────────────────────────────────────────── */

export function upstreamSet(g: LineageGraph, id: string, maxDepth = Infinity, allowedEdgeTypes?: string[]): string[] {
  return walkDirected(g, id, "up", maxDepth, allowedEdgeTypes);
}
export function downstreamSet(g: LineageGraph, id: string, maxDepth = Infinity, allowedEdgeTypes?: string[]): string[] {
  return walkDirected(g, id, "down", maxDepth, allowedEdgeTypes);
}

export function subgraphAround(g: LineageGraph, centerId: string, hopsUp = 1, hopsDown = 1, allowedEdgeTypes?: string[]): LineageGraph {
  const up = new Set(upstreamSet(g, centerId, hopsUp, allowedEdgeTypes));
  const down = new Set(downstreamSet(g, centerId, hopsDown, allowedEdgeTypes));
  

  const nodes: LineageNode[] = [];
  

  const edges: LineageEdge[] = [];
  for (let i = 0; i < g.edges.length; i++) {
    const e = g.edges[i];
    if (e.from === centerId || e.to === centerId || (up.has(e.from) && up.has(e.to)) || (down.has(e.from) && down.has(e.to)) || (up.has(e.from) && down.has(e.to))) {
      if (!nodes.find(n => n.id === e.from)) nodes.push(g.nodes[e.from]);
      if (!nodes.find(n => n.id === e.to)) nodes.push(g.nodes[e.to]);
      edges.push({ ...e });
    }
  }
  return createGraph(nodes, edges);
}

export function topologicalOrder(g: LineageGraph): { order: string[]; leftover: string[] } {
  // Kahn's algorithm on a copy of in-degree
  const indeg: Dict<number> = {};
  for (const id in g.nodes) indeg[id] = g.inAdj[id]?.length || 0;

  const q: string[] = [];
  for (const id in indeg) if (indeg[id] === 0) q.push(id);

  const order: string[] = [];
  let qi = 0;
  while (qi < q.length) {
    const n = q[qi++];
    order.push(n);
    const outs = g.outAdj[n] || [];
    for (let i = 0; i < outs.length; i++) {
      const m = outs[i];
      indeg[m] = (indeg[m] || 0) - 1;
      if (indeg[m] === 0) q.push(m);
    }
  }

  const leftover: string[] = [];
  for (const id in indeg) if (indeg[id] > 0) leftover.push(id);
  return { order, leftover };
}

/** Strongly connected components using Tarjan; components with size>1 or self-loops are cycles. */
export function findCycles(g: LineageGraph): string[][] {
  const ids = Object.keys(g.nodes);
  const index: Dict<number> = {};
  const low: Dict<number> = {};
  const onStack: Dict<boolean> = {};
  let idx = 0;
  const stack: string[] = [];
  const comps: string[][] = [];

  function strong(v: string) {
    index[v] = idx; low[v] = idx; idx++; stack.push(v); onStack[v] = true;
    const outs = g.outAdj[v] || [];
    for (let i = 0; i < outs.length; i++) {
      const w = outs[i];
      if (index[w] === undefined) { strong(w); low[v] = Math.min(low[v], low[w]); }
      else if (onStack[w]) { low[v] = Math.min(low[v], index[w]); }
    }
    if (low[v] === index[v]) {
      const comp: string[] = [];
      while (true) {
        const w = stack.pop()!; onStack[w] = false; comp.push(w);
        if (w === v) break;
      }
      // self-loop counts as cycle if edge exists v->v
      if (comp.length > 1 || hasSelfLoop(g, comp[0])) comps.push(comp);
    }
  }

  for (let i = 0; i < ids.length; i++) if (index[ids[i]] === undefined) strong(ids[i]);
  return comps;
}

export function pathsBetween(
  g: LineageGraph,
  src: string,
  dst: string,
  opts?: { maxDepth?: number; maxPaths?: number; allowedEdgeTypes?: string[]; direction?: "forward" | "backward" | "both" }
): string[][] {
  const maxDepth = opts?.maxDepth ?? 8;
  const maxPaths = opts?.maxPaths ?? 200;
  const allowed = opts?.allowedEdgeTypes;
  const dir = opts?.direction || "forward";

  const results: string[][] = [];
  const path: string[] = [];
  const seen = new Set<string>();

  function step(v: string, depth: number) {
    if (results.length >= maxPaths) return;
    path.push(v); seen.add(v);

    if (v === dst) {
      results.push(path.slice());
      path.pop(); seen.delete(v);
      return;
    }
    if (depth >= maxDepth) { path.pop(); seen.delete(v); return; }

    const nexts: string[] = [];
    if (dir === "forward" || dir === "both") nexts.push(...neighborsByType(g, v, "out", allowed));
    if (dir === "backward" || dir === "both") nexts.push(...neighborsByType(g, v, "in", allowed));

    for (let i = 0; i < nexts.length; i++) {
      const w = nexts[i];
      if (seen.has(w)) continue;
      step(w, depth + 1);
      if (results.length >= maxPaths) break;
    }

    path.pop(); seen.delete(v);
  }

  step(src, 0);
  return results;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Filtering, merging, exporting
 * ────────────────────────────────────────────────────────────────────────── */

export function filterEdges(g: LineageGraph, predicate: (e: LineageEdge) => boolean): LineageGraph {
  const keptEdges = g.edges.filter(predicate);
  // Keep only nodes that are endpoints of kept edges
  const keepIds = new Set<string>();
  for (let i = 0; i < keptEdges.length; i++) { keepIds.add(keptEdges[i].from); keepIds.add(keptEdges[i].to); }
  const keptNodes: LineageNode[] = [];
  for (const id in g.nodes) if (keepIds.has(id)) keptNodes.push(g.nodes[id]);
  return createGraph(keptNodes, keptEdges);
}

export function pruneNodes(g: LineageGraph, keepPredicate: (n: LineageNode) => boolean): LineageGraph {
  const kept: LineageNode[] = [];
  for (const id in g.nodes) if (keepPredicate(g.nodes[id])) kept.push(g.nodes[id]);
  const keepIds = new Set(kept.map(n => n.id));
  const edges = g.edges.filter(e => keepIds.has(e.from) && keepIds.has(e.to));
  return createGraph(kept, edges);
}

export function mergeGraphs(graphs: LineageGraph[]): LineageGraph {
  const allNodes: Dict<LineageNode> = Object.create(null);
  const allEdges: LineageEdge[] = [];
  for (let i = 0; i < graphs.length; i++) {
    const g = graphs[i];
    for (const id in g.nodes) {
      const n = g.nodes[id];
      if (!allNodes[id]) allNodes[id] = { ...n, tags: (n.tags || []).slice(), meta: n.meta ? { ...n.meta } : undefined };
      else {
        const cur = allNodes[id];
        cur.label = n.label || cur.label;
        cur.kind = n.kind || cur.kind;
        if (n.tags && n.tags.length) cur.tags = uniq([...(cur.tags || []), ...n.tags]);
        if (n.meta) cur.meta = { ...(cur.meta || {}), ...n.meta };
      }
    }
    for (let j = 0; j < g.edges.length; j++) allEdges.push(g.edges[j]);
  }
  return createGraph(Object.values(allNodes), allEdges);
}

export function toDOT(g: LineageGraph, opts?: { compact?: boolean; rankdir?: "LR" | "TB" }): string {
  const rankdir = opts?.rankdir || "LR";
  const compact = !!opts?.compact;
  const esc = (s: string) => String(s).replace(/"/g, '\\"');

  const lines: string[] = [];
  lines.push(`digraph G {`);
  lines.push(`  rankdir=${rankdir};`);
  if (compact) lines.push(`  node [shape=box, style="rounded,filled", fillcolor="#eef5ff", fontname="Inter"];`);
  else lines.push(`  node [shape=box, style="rounded,filled", fillcolor="#eef5ff"];`);

  for (const id in g.nodes) {
    const n = g.nodes[id];
    const label = n.label || n.id;
    const kind = n.kind || "other";
    lines.push(`  "${esc(id)}" [label="${esc(label)}\\n(${esc(kind)})"];`);
  }
  for (let i = 0; i < g.edges.length; i++) {
    const e = g.edges[i];
    const lbl = e.label || e.type || "";
    const attr = lbl ? ` [label="${esc(lbl)}", fontsize=10]` : "";
    lines.push(`  "${esc(e.from)}" -> "${esc(e.to)}"${attr};`);
  }
  lines.push(`}`);
  return lines.join("\n");
}

export function toMermaid(g: LineageGraph, opts?: { direction?: "LR" | "TB" }): string {
  const dir = opts?.direction || "LR";
  const esc = (s: string) => String(s).replace(/[[\](){}|]/g, "").replace(/"/g, '\\"');
  const lines: string[] = [];
  lines.push(`flowchart ${dir}`);
  for (const id in g.nodes) {
    const n = g.nodes[id];
    const label = esc(n.label || n.id);
    const kind = esc(n.kind || "other");
    lines.push(`  ${idSafe(id)}["${label}\\n(${kind})"]`);
  }
  for (let i = 0; i < g.edges.length; i++) {
    const e = g.edges[i];
    const lbl = esc(e.label || e.type || "");
    const mid = lbl ? `|${lbl}|` : "";
    lines.push(`  ${idSafe(e.from)} -->${mid} ${idSafe(e.to)}`);
  }
  return lines.join("\n");
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Search helpers
 * ────────────────────────────────────────────────────────────────────────── */

export function findNodes(g: LineageGraph, q: { text?: string; kinds?: string[]; tags?: string[] }): LineageNode[] {
  const text = (q.text || "").toLowerCase();
  const kinds = q.kinds && q.kinds.map(s => s.toLowerCase());
  const tags = q.tags && q.tags.map(s => s.toLowerCase());

  const out: LineageNode[] = [];
  for (const id in g.nodes) {
    const n = g.nodes[id];
    if (text) {
      const hay = `${n.id} ${n.label || ""}`.toLowerCase();
      if (!hay.includes(text)) continue;
    }
    if (kinds && kinds.length && (!n.kind || !kinds.includes(n.kind.toLowerCase()))) continue;
    if (tags && tags.length) {
      const nt = (n.tags || []).map(x => x.toLowerCase());
      let match = false;
      for (let i = 0; i < tags.length; i++) if (nt.includes(tags[i])) { match = true; break; }
      if (!match) continue;
    }
    out.push(n);
  }
  return out;
}

export function degree(g: LineageGraph, id: string): { in: number; out: number; total: number } {
  const inn = (g.inAdj[id] || []).length;
  const out = (g.outAdj[id] || []).length;
  return { in: inn, out, total: inn + out };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals
 * ────────────────────────────────────────────────────────────────────────── */

function walkDirected(g: LineageGraph, id: string, dir: "up" | "down", maxDepth: number, allowedEdgeTypes?: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>([id]);
  const q: Array<{ id: string; d: number }> = [{ id, d: 0 }];

  while (q.length) {
    const cur = q.shift()!;
    const nexts = neighborsByType(g, cur.id, dir === "down" ? "out" : "in", allowedEdgeTypes);
    for (let i = 0; i < nexts.length; i++) {
      const n = nexts[i];
      if (seen.has(n)) continue;
      seen.add(n); out.push(n);
      if (cur.d + 1 < maxDepth) q.push({ id: n, d: cur.d + 1 });
    }
  }
  return out;
}

function neighborsByType(g: LineageGraph, id: string, where: "in" | "out", allowedEdgeTypes?: string[]): string[] {
  if (!allowedEdgeTypes || allowedEdgeTypes.length === 0) {
    return where === "out" ? (g.outAdj[id] || []) : (g.inAdj[id] || []);
  }
  // Slow path: filter by edge types
  const adj = where === "out" ? (g.outAdj[id] || []) : (g.inAdj[id] || []);
  const keep = new Set<string>();
  for (let i = 0; i < g.edges.length; i++) {
    const e = g.edges[i];
    if (where === "out" && e.from === id && allowedEdgeTypes.includes(e.type || "derives")) keep.add(e.to);
    if (where === "in" && e.to === id && allowedEdgeTypes.includes(e.type || "derives")) keep.add(e.from);
  }
  return Array.from(keep);
}

function edgeKey(e: LineageEdge): string {
  return `${e.from}→${e.to}#${e.type || "derives"}`;
}

function uniq<T>(xs: T[]): T[] {
  const s = new Set<T>(xs);
  return Array.from(s);
}

function hasSelfLoop(g: LineageGraph, id: string): boolean {
  const outs = g.outAdj[id] || [];
  for (let i = 0; i < outs.length; i++) if (outs[i] === id) return true;
  return false;
}

function idSafe(id: string): string {
  // mermaid identifiers cannot contain some punctuation; replace with _
  return id.replace(/[^A-Za-z0-9_]/g, "_");
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Example (commented)
 *
 * // const g = createGraph(
 * //   [
 * //     { id: "px_daily", kind: "dataset" },
 * //     { id: "returns_daily", kind: "dataset" },
 * //     { id: "calc_returns", kind: "job" }
 * //   ],
 * //   [
 * //     { from: "px_daily", to: "calc_returns", type: "reads" },
 * //     { from: "calc_returns", to: "returns_daily", type: "writes" }
 * //   ]
 * // );
 * //
 * // console.log(upstreamSet(g, "returns_daily"));   // ['calc_returns','px_daily']
 * // console.log(topologicalOrder(g));               // order with 'px_daily' before 'returns_daily'
 * // console.log(findCycles(g));                     // []
 * // console.log(toMermaid(g));
 * // console.log(toDOT(g));
 * ────────────────────────────────────────────────────────────────────────── */