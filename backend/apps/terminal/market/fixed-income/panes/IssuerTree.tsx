"use client";

/**
 * issuertree.tsx (clean rewrite)
 * Zero-import, self-contained hierarchical browser for fixed-income issuers.
 *
 * Goals of this rewrite:
 * - Rock-solid recursion with no undefined access
 * - Deterministic roll-ups (count, MV, Avg YTM, Duration)
 * - Fast search filter that keeps ancestor chain visible
 * - Expand/Collapse per node + Expand All / Collapse All
 * - Tri-state selection with subtree propagation (no flicker)
 * - Minimal state surface: expanded, checked, query
 *
 * Tailwind only. No external imports, no links.
 */

/* =============================== Public Types =============================== */

export type IssuerNode = {
  id: string;
  name: string;
  level: "sector" | "industry" | "issuer" | "book";
  metrics?: {
    count?: number;          // number of bonds
    marketValue?: number;    // total MV (currency-agnostic)
    avgYtm?: number;         // decimal e.g. 0.045
    duration?: number;       // modified duration
  };
  children?: IssuerNode[];
};

/* =============================== Page Component ============================ */

export default function IssuerTree({
  data = [],
  title = "Issuer Tree",
  className = "",
  defaultExpanded = true,
  selectable = true,
  showBooks = true,
}: {
  data: IssuerNode[];
  title?: string;
  className?: string;
  defaultExpanded?: boolean;
  selectable?: boolean;
  showBooks?: boolean;
}) {
  /** Normalize once per `data` change */
  const root = useMemo(() => normalizeToTree(data, { showBooks }), [data, showBooks]);

  /** Expanded map: initialize all nodes to defaultExpanded */
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() => {
    const map: Record<string, boolean> = {};
    walk(root, (n) => (map[n.id] = defaultExpanded));
    // Root (synthetic) stays expanded to render top level
    map[root.id] = true;
    return map;
  });

  /** Checked map: true/false only for nodes explicitly set by the user */
  const [checked, setChecked] = useState<Record<string, boolean>>({});

  /** Search query */
  const [q, setQ] = useState("");

  /** Keyboard helpers */
  const searchRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (e as any).isComposing;
      if (!inField && e.key === "/") { e.preventDefault(); searchRef.current?.focus(); }
      if (!inField && (e.key === "e" || e.key === "E")) setAllExpanded(true);
      if (!inField && (e.key === "c" || e.key === "C")) setAllExpanded(false);
      if (inField && e.key === "Escape") { (e.target as HTMLInputElement).blur(); setQ(""); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  /** Expand/Collapse all */
  const setAllExpanded = (open: boolean) => {
    const next: Record<string, boolean> = {};
    walk(root, (n) => (next[n.id] = open));
    next[root.id] = true; // root always expanded
    setExpanded(next);
  };

  /** Visible flat rows (filter + flatten respecting expansion) */
  const rows = useMemo(() => {
    const out: FlatRow[] = [];
    const qlc = q.trim().toLowerCase();

    // A node passes if itself or any descendant matches the query
    const matches = (n: TreeNode): boolean => {
      if (!qlc) return true;
      if (n.name.toLowerCase().includes(qlc)) return true;
      return (n.children || []).some(matches);
    };

    const rec = (n: TreeNode, isRoot = false) => {
      if (!matches(n)) return;
      if (!isRoot) out.push(toRow(n)); // skip synthetic root
      if (expanded[n.id]) for (const c of n.children || []) rec(c);
    };

    rec(root, true);
    return out;
  }, [root, expanded, q]);

  /** Tri-state marks derived bottom-up from explicit `checked` */
  const tri = useMemo(() => deriveTriState(root, checked), [root, checked]);

  /** Selection handlers */
  const toggleExpand = (id: string) => setExpanded((m) => ({ ...m, [id]: !m[id] }));
  const onCheck = (id: string, val: boolean) => {
    const node = findNode(root, id); if (!node) return;
    const next = { ...checked };
    // Set subtree explicitly
    walk(node, (n) => (next[n.id] = val));
    // Clear explicit flags on ancestors to allow auto tri-state
    for (let p = node.parent; p; p = p.parent) delete next[p.id];
    setChecked(next);
  };

  /** Selected leaf count (issuers or books only) */
  const selectedLeaves = useMemo(() => {
    let n = 0;
    walk(root, (node) => {
      const isLeaf = !node.children?.length;
      if (isLeaf && tri[node.id] === true) n++;
    });
    return n;
  }, [root, tri]);

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">
            {rows.length} rows · {selectedLeaves} selected
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <label className="relative">
            <input
              ref={searchRef}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search…  (press /)"
              className="w-72 rounded-md border border-neutral-700 bg-neutral-950 pl-7 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
            />
            <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
              <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
              <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </label>
          <button onClick={() => setAllExpanded(true)}  className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Expand all</button>
          <button onClick={() => setAllExpanded(false)} className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800">Collapse all</button>
        </div>
      </div>

      {/* Table header */}
      <div className="grid grid-cols-12 gap-2 border-b border-neutral-800 px-3 py-2 text-[11px] uppercase text-neutral-500">
        <div className="col-span-6">Hierarchy</div>
        <div className="col-span-2 text-right">Bonds</div>
        <div className="col-span-2 text-right">Market Value</div>
        <div className="col-span-1 text-right">Avg YTM</div>
        <div className="col-span-1 text-right">Duration</div>
      </div>

      {/* Rows */}
      <div className="divide-y divide-neutral-800">
        {rows.length === 0 && <div className="px-3 py-8 text-center text-xs text-neutral-500">No matching nodes.</div>}
        {rows.map((r) => {
          const isOpen = !!expanded[r.id];
          const canToggle = r.hasChildren;
          const mark = tri[r.id];
          const pad = 8 + r.depth * 16;
          const chip =
            r.level === "sector" ? "bg-blue-600/20 text-blue-300" :
            r.level === "industry" ? "bg-purple-600/20 text-purple-300" :
            r.level === "issuer" ? "bg-emerald-600/20 text-emerald-300" :
            "bg-amber-600/20 text-amber-300";

          return (
            <div key={r.id} className="grid grid-cols-12 items-center gap-2 px-3 py-2">
              {/* Hierarchy */}
              <div className="col-span-6 min-w-0">
                <div className="flex items-center gap-2">
                  {/* Indent */}
                  <div style={{ width: pad }} className="shrink-0" />
                  {/* Toggle */}
                  {canToggle ? (
                    <button
                      onClick={() => toggleExpand(r.id)}
                      className="grid h-5 w-5 place-items-center rounded border border-neutral-700 bg-neutral-950 hover:bg-neutral-800"
                      title={isOpen ? "Collapse" : "Expand"}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24">
                        {isOpen
                          ? <path d="M7 15l5-5 5 5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />
                          : <path d="M7 10l5 5 5-5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />}
                      </svg>
                    </button>
                  ) : <div className="h-5 w-5" />}

                  {/* Checkbox */}
                  {selectable ? (
                    <label className="relative inline-flex h-5 w-5 items-center justify-center">
                      <input
                        type="checkbox"
                        checked={mark === true}
                        ref={(el) => { if (el) el.indeterminate = mark === "indeterminate"; }}
                        onChange={(e) => onCheck(r.id, e.target.checked)}
                        className="h-4 w-4 accent-emerald-500"
                        title="Select subtree"
                      />
                    </label>
                  ) : <div className="w-5" />}

                  {/* Name */}
                  <span className={`truncate text-sm ${r.level === "issuer" ? "text-neutral-100" : "text-neutral-200"}`}>
                    <span className={`mr-2 rounded px-1.5 py-0.5 text-[10px] ${chip}`}>{labelFor(r.level)}</span>
                    <span dangerouslySetInnerHTML={{ __html: highlight(r.name, q) }} />
                  </span>
                </div>
              </div>

              {/* Metrics */}
              <div className="col-span-2 text-right font-mono tabular-nums">{fmtInt(r.metrics.count)}</div>
              <div className="col-span-2 text-right font-mono tabular-nums">${fmtInt(r.metrics.marketValue)}</div>
              <div className="col-span-1 text-right font-mono tabular-nums">{fmtPct(r.metrics.avgYtm)}</div>
              <div className="col-span-1 text-right font-mono tabular-nums">{fmt(r.metrics.duration, 2)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* =============================== Tree Building ============================ */

type NodeMetrics = Required<NonNullable<IssuerNode["metrics"]>>;
type TreeNode = {
  id: string;
  name: string;
  level: IssuerNode["level"];
  metrics: NodeMetrics;
  parent: TreeNode | null;
  children: TreeNode[];
  depth: number;   // 0 for sector level
};

type FlatRow = {
  id: string;
  name: string;
  level: IssuerNode["level"];
  metrics: NodeMetrics;
  depth: number;
  hasChildren: boolean;
};

function normalizeToTree(data: IssuerNode[], opts: { showBooks: boolean }): TreeNode {
  // Synthetic root to simplify flatten logic
  const root: TreeNode = {
    id: "__root__",
    name: "root",
    level: "sector",
    metrics: { count: 0, marketValue: 0, avgYtm: 0, duration: 0 },
    parent: null,
    children: [],
    depth: -1,
  };

  const build = (inNode: IssuerNode, parent: TreeNode, depth: number): TreeNode => {
    const kids = (inNode.children || []).filter((c) => (opts.showBooks ? true : c.level !== "book"));
    const node: TreeNode = {
      id: inNode.id,
      name: inNode.name,
      level: inNode.level,
      metrics: {
        count: inNode.metrics?.count ?? 0,
        marketValue: inNode.metrics?.marketValue ?? 0,
        avgYtm: inNode.metrics?.avgYtm ?? 0,
        duration: inNode.metrics?.duration ?? 0,
      },
      parent,
      children: [],
      depth,
    };
    node.children = kids.map((c) => build(c, node, depth + 1));
    node.metrics = rollup(node); // ensure consistent rollups
    return node;
  };

  root.children = data.map((s) => build(s, root, 0));
  root.metrics = rollup(root);
  return root;
}

function rollup(node: TreeNode): NodeMetrics {
  if (!node.children.length) return node.metrics;
  // Sum counts and market value, then weighted averages for ytm/duration
  let count = 0, mv = 0;
  for (const c of node.children) { count += c.metrics.count; mv += c.metrics.marketValue; }
  // If leafs didn't carry counts, fallback to number of leaves
  if (count === 0) count = leafCount(node);
  let y = 0, d = 0;
  if (mv > 0) {
    for (const c of node.children) {
      const w = (c.metrics.marketValue || 0) / mv;
      y += c.metrics.avgYtm * w;
      d += c.metrics.duration * w;
    }
  } else {
    // Equal weight fallback
    const n = node.children.length || 1;
    for (const c of node.children) { y += c.metrics.avgYtm / n; d += c.metrics.duration / n; }
  }
  return {
    count: Math.max(node.metrics.count, count),
    marketValue: Math.max(node.metrics.marketValue, mv),
    avgYtm: y,
    duration: d,
  };
}

function leafCount(n: TreeNode): number {
  if (!n.children.length) return Math.max(1, n.metrics.count);
  let t = 0; for (const c of n.children) t += leafCount(c);
  return t;
}

function toRow(n: TreeNode): FlatRow {
  return {
    id: n.id,
    name: n.name,
    level: n.level,
    metrics: n.metrics,
    depth: n.depth,
    hasChildren: n.children.length > 0,
  };
}

function walk(n: TreeNode, fn: (n: TreeNode) => void) {
  fn(n);
  for (const c of n.children) walk(c, fn);
}

function findNode(n: TreeNode, id: string): TreeNode | null {
  if (n.id === id) return n;
  for (const c of n.children) {
    const hit = findNode(c, id);
    if (hit) return hit;
  }
  return null;
}

/* =============================== Tri-state logic =========================== */

function deriveTriState(root: TreeNode, checked: Record<string, boolean>) {
  const state: Record<string, boolean | "indeterminate"> = {};
  const rec = (n: TreeNode): boolean | "indeterminate" => {
    const self = checked[n.id];
    if (!n.children.length) {
      state[n.id] = !!self;
      return state[n.id];
    }
    const childMarks = n.children.map(rec);
    const allTrue = childMarks.every((v) => v === true);
    const allFalse = childMarks.every((v) => v === false);
    let mark: boolean | "indeterminate";
    if (self === true) mark = true;
    else if (self === false) mark = false;
    else if (allTrue) mark = true;
    else if (allFalse) mark = false;
    else mark = "indeterminate";
    state[n.id] = mark;
    return mark;
  };
  rec(root);
  return state;
}

/* ================================== UI utils =============================== */

function labelFor(l: IssuerNode["level"]) {
  return l === "sector" ? "Sector" : l === "industry" ? "Industry" : l === "issuer" ? "Issuer" : "Book";
}

function highlight(s: string, q: string) {
  if (!q) return escapeHtml(s);
  const re = new RegExp(`(${escapeReg(q)})`, "ig");
  return escapeHtml(s).replace(re, "<mark class='bg-amber-300/30 text-amber-200 rounded px-0.5'>$1</mark>");
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]!));
}
function escapeReg(s: string) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function fmtInt(n: number) { return Math.round(n).toLocaleString("en-US"); }
function fmtPct(x: number) { return (x * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%"; }

/* ------------------- Ambient React declarations (no imports) ------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useRef<T>(v: T | null): { current: T | null };

/* ----------------------------------- Notes -----------------------------------
- This file intentionally has ZERO imports to satisfy “pure code” constraint.
- Metrics roll-up rules:
  * count/mv sum children; fallback to leaf count if counts missing.
  * avgYtm/duration are MV-weighted if MV>0, else equally weighted.
- Keyboard: "/" focus search, "e" expand all, "c" collapse all, Esc clears focus.
----------------------------------------------------------------------------- */