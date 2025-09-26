// lib/triggerstress.action.ts
// No imports. Pure server action to run a sleeves stress scenario.
// - Accepts either FormData (Next.js Server Actions) or a typed object
// - Shocks are applied only to leaf sleeves; parents aggregate from children
// - Percent shocks can be provided as whole numbers (e.g., -3 for -3%)
// - Returns flattened results with P&L, new value, and contribution
//
// FormData fields (all optional except `sleeves`):
//   sleeves:        JSON Sleeve[] (see type below)
//   leafShocks:     JSON Record<string, number>  // id -> pct
//   globalShockPct: number                       // default 0
//   scenario:       string
//   wantCsv:        "1" | "true"                 // include CSV string in result
//
// Example (typed):
//   const res = await triggerStress({
//     sleeves,
//     leafShocks: { "IT": -3, "BANKS": -2 },
//     globalShockPct: 0,
//     scenario: "Macro -2%",
//   });

"use server";

/* ----------------------------- Types ----------------------------- */

export type Sleeve = {
  id: string;
  name: string;
  mv?: number;                 // base market value (absolute). If missing, summed from children
  children?: Sleeve[];
  note?: string;
  color?: string;
};

export type StressInput = {
  sleeves: Sleeve[];
  leafShocks?: Record<string, number>;   // percent for leaves (e.g., -3 means -3%)
  globalShockPct?: number;               // fallback for leaves without explicit shocks
  scenario?: string;
  wantCsv?: boolean;
};

export type StressRow = {
  id: string;
  name: string;
  depth: number;
  parentId?: string;
  baseMV: number;
  shockPct?: number;     // resulting percent (parents: weighted from kids)
  pnl: number;
  newMV: number;
  contrib: number;       // pnl / totalBaseMV (roots total)
  note?: string;
  color?: string;
};

export type StressResult = {
  scenario: string;
  updatedAt: string;   // ISO
  totals: { baseMV: number; pnl: number; newMV: number };
  rows: StressRow[];   // flattened, depth-first
  errors: string[];
  csv?: string;        // optional CSV if requested
};

/* ----------------------- Public Server Action ----------------------- */

export default async function triggerStress(
  input: StressInput | FormData,
): Promise<StressResult> {
  const errors: string[] = [];

  const parsed = isFormData(input) ? parseFormData(input) : sanitizeInput(input);
  errors.push(...parsed.errors);

  const sleeves = parsed.sleeves || [];
  const leafShocks = parsed.leafShocks || {};
  const globalShockPct = num(parsed.globalShockPct, 0);
  const scenario = parsed.scenario || "Stress";

  // Build computed forest (resolve base market values)
  const forest = sleeves.map((s) => computeNode(s));
  const totalBase = forest.reduce((s, n) => s + n.baseMV, 0);

  // Flatten + compute P&L via bottom-up aggregation
  const flat = flattenForest(forest);

  // Pass 1: apply shocks to leaves
  const map = new Map<string, { base: number; newV: number; pnl: number; shock: number }>();
  for (const n of flat) {
    if (n.isLeaf) {
      const pct = pctToFrac(leafShocks[n.id] ?? globalShockPct);
      const newV = n.baseMV * (1 + pct);
      map.set(n.id, { base: n.baseMV, newV, pnl: newV - n.baseMV, shock: pct });
    }
  }

  // Pass 2: aggregate parents from children (deepest -> shallowest)
  for (const n of [...flat].sort((a, b) => b.depth - a.depth)) {
    if (n.isLeaf) continue;
    const kids = flat.filter((k) => k.parentId === n.id);
    let newV = 0;
    let base = 0;
    for (const k of kids) {
      const m = map.get(k.id);
      if (m) { newV += m.newV; base += m.base; }
    }
    if (base === 0) {
      // no children with values — treat as neutral
      base = n.baseMV;
      newV = base;
    }
    const pnl = newV - base;
    const shock = base > 0 ? pnl / base : 0;
    map.set(n.id, { base, newV, pnl, shock });
  }

  // Assemble rows
  const rows: StressRow[] = flat.map((n) => {
    const m = map.get(n.id) || { base: n.baseMV, newV: n.baseMV, pnl: 0, shock: 0 };
    return {
      id: n.id,
      name: n.name,
      depth: n.depth,
      parentId: n.parentId,
      baseMV: fix(m.base),
      shockPct: m.shock,
      pnl: fix(m.pnl),
      newMV: fix(m.newV),
      contrib: totalBase > 0 ? fix(m.pnl / totalBase) : 0,
      note: n.note,
      color: n.color,
    };
  });

  // Totals (roots only)
  const totals = rows
    .filter((r) => r.depth === 0)
    .reduce(
      (acc, r) => {
        acc.baseMV += r.baseMV;
        acc.pnl += r.pnl;
        acc.newMV += r.newMV;
        return acc;
      },
      { baseMV: 0, pnl: 0, newMV: 0 },
    );

  const result: StressResult = {
    scenario,
    updatedAt: new Date().toISOString(),
    totals: {
      baseMV: fix(totals.baseMV),
      pnl: fix(totals.pnl),
      newMV: fix(totals.newMV),
    },
    rows,
    errors,
    csv: parsed.wantCsv ? buildCsv(rows) : undefined,
  };

  return result;
}

/* ----------------------------- Parsing ----------------------------- */

function isFormData(x: any): x is FormData {
  return typeof x === "object" && x?.constructor?.name === "FormData";
}

function parseFormData(fd: FormData): StressInput & { errors: string[] } {
  const errors: string[] = [];
  const sleeves = readJson<Sleeve[]>(fd.get("sleeves"));
  if (!Array.isArray(sleeves)) errors.push("`sleeves` is required (JSON array)");

  const leafShocks = readJson<Record<string, number>>(fd.get("leafShocks")) || {};
  const globalShockPct = num(fd.get("globalShockPct"), 0);
  const scenario = str(fd.get("scenario")) || "Stress";
  const wantCsv = truthy(fd.get("wantCsv"));

  return { sleeves: sleeves || [], leafShocks, globalShockPct, scenario, wantCsv, errors };
}

function sanitizeInput(x: StressInput): StressInput & { errors: string[] } {
  const errors: string[] = [];
  const sleeves = Array.isArray(x?.sleeves) ? x.sleeves : [];
  if (!sleeves.length) errors.push("`sleeves` is required (array).");

  const leafShocks: Record<string, number> = {};
  for (const k of Object.keys(x.leafShocks || {})) {
    const v = x.leafShocks![k];
    if (Number.isFinite(v as number)) leafShocks[k] = Number(v);
  }
  return {
    sleeves,
    leafShocks,
    globalShockPct: num(x.globalShockPct, 0),
    scenario: str(x.scenario) || "Stress",
    wantCsv: !!x.wantCsv,
    errors,
  };
}

/* -------------------------- Computation core -------------------------- */

type CNode = {
  id: string;
  name: string;
  baseMV: number;
  depth: number;
  parentId?: string;
  isLeaf: boolean;
  note?: string;
  color?: string;
};

function computeNode(n: Sleeve, depth = 0, parentId?: string): CNode {
  const kids = (n.children || []).map((c) => computeNode(c, depth + 1, n.id));
  const sumKids = kids.reduce((s, k) => s + k.baseMV, 0);
  const baseMV = Number.isFinite(n.mv as number) ? Number(n.mv) : sumKids;
  return {
    id: n.id,
    name: n.name,
    baseMV: Math.max(0, Number(baseMV) || 0),
    depth,
    parentId,
    isLeaf: (n.children || []).length === 0,
    note: n.note,
    color: n.color,
  };
}

function flattenForest(forest: CNode[]): CNode[] {
  const out: CNode[] = [];
  const walk = (x: CNode, kids?: CNode[]) => {
    out.push(x);
    (kids || []).forEach((k) => walk(k, []));
  };
  const rec = (n: Sleeve | CNode): CNode[] => {
    // helper not used here (we already have CNode forest), keeping for clarity
    return [];
  };
  const queue: CNode[] = [...forest];
  while (queue.length) {
    const n = queue.shift()!;
    out.push(n);
    // We don't have direct children list here; rebuild from forest when needed.
    // For performance, we rely on parentId scanning where necessary.
    for (const c of forest) void c; // noop to satisfy TS in this inline style
  }
  // The above produces only roots; replace with proper DFS:
  const res: CNode[] = [];
  const byParent = new Map<string | undefined, CNode[]>();
  for (const node of forest) {
    if (!byParent.has(node.parentId)) byParent.set(node.parentId, []);
    byParent.get(node.parentId)!.push(node);
  }
  const dfs = (node: CNode) => {
    res.push(node);
    const children = byParent.get(node.id) || [];
    children.forEach(dfs);
  };
  (byParent.get(undefined) || []).forEach(dfs);
  return res;
}

/* ------------------------------ Utils ------------------------------ */

function pctToFrac(pct: number): number {
  const n = Number(pct);
  if (!Number.isFinite(n)) return 0;
  return n / 100;
}

function readJson<T>(v: any): T | undefined {
  const s = str(v);
  if (!s) return undefined;
  try { return JSON.parse(s) as T; } catch { return undefined; }
}
function str(v: any): string | undefined {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}
function num(v: any, d = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}
function truthy(v: any): boolean {
  const s = String(v ?? "").toLowerCase().trim();
  return s === "1" || s === "true" || s === "yes";
}
function fix(n: number) {
  // keep a sensible number of decimals to avoid floating noise
  return Math.round(n * 1e10) / 1e10;
}

/* --------------------------- CSV (optional) --------------------------- */

function buildCsv(rows: StressRow[]): string {
  const head = ["Depth", "ID", "Name", "Base", "Shock%", "P&L", "New", "Contrib%", "ParentID"];
  const data = rows.map((r) => [
    String(r.depth),
    r.id,
    r.name,
    money(r.baseMV, ""),
    pct(r.shockPct ?? 0),
    money(r.pnl, ""),
    money(r.newMV, ""),
    pct(r.contrib),
    r.parentId || "",
  ]);
  return [head, ...data].map((r) => r.map(csvEsc).join(",")).join("\n");
}

function csvEsc(s: string) {
  const needs = /[",\n\r]/.test(s) || /^\s|\s$/.test(s);
  return needs ? `"${s.replace(/"/g, '""')}"` : s;
}
function money(n: number, unit: string) {
  const sign = n < 0 ? "-" : "";
  const v = Math.abs(n);
  if (v >= 1_000_000_000) return `${sign}${unit}${(v / 1_000_000_000).toFixed(2)}B`;
  if (v >= 1_000_000) return `${sign}${unit}${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `${sign}${unit}${(v / 1_000).toFixed(2)}k`;
  return `${sign}${unit}${v.toFixed(2)}`;
}
function pct(x: number) {
  const sign = x < 0 ? "-" : "";
  const v = Math.abs(x) * 100;
  return `${sign}${v.toFixed(2)}%`;
}

/* ---------------------- Convenience variant ---------------------- */

/** Quick helper to apply a single global shock to all leaves. */
export async function simpleTrigger(
  sleeves: Sleeve[],
  globalShockPct = 0,
  scenario = "Simple Stress",
): Promise<StressResult> {
  return triggerStress({ sleeves, globalShockPct, scenario });
}
