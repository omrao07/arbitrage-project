// query/select.ts
// Tiny, zero-dependency query helpers over arrays of plain objects.
// No imports. Works in Node and browsers.
//
// Core API:
//   - select({ from, where?, columns?, distinct?, orderBy?, limit?, offset? })
//   - groupSelect({ from, groupBy, aggregates, having?, orderBy?, limit?, offset? })
//
// Building blocks (also exported):
//   - where(rows, predicate)
//   - project(rows, columns)
//   - order(rows, keys)
//   - distinctRows(rows, keys?)
//   - groupBy(rows, keys)
//   - aggregate(groups, aggs)
//
// Conventions:
//   • A "row" is a plain object: Record<string, any>.
//   • Column selectors support strings (dot.path), functions, or objects { as, expr }.
//   • Order keys support strings or functions, with {desc?, nulls?}.
//   • Distinct can be boolean (entire row) or string[]/((row)=>any[]) to choose keys.
//   • Grouping and aggregates mirror simple SQL: groupBy + { alias: {fn, expr?} }.
//
// Example:
//   const res = select({
//     from: rows,
//     where: r => r.px > 0,
//     columns: ["ticker", { as: "ret", expr: (r)=> (r.px/r.px_prev - 1) }],
//     distinct: ["ticker"],
//     orderBy: [{ key: "ticker" }]
//   });
//
//   const agg = groupSelect({
//     from: rows,
//     groupBy: ["ticker"],
//     aggregates: {
//       n: { fn: "count" },
//       avg_px: { fn: "avg", expr: r => r.px },
//       last_dt: { fn: "max", expr: r => r.date } // works on ISO dates
//     },
//     having: g => g.n >= 5,
//     orderBy: [{ key: "avg_px", desc: true }],
//   });

export type Dict<T = any> = { [k: string]: T };
export type Row = Dict<any>;

export type Expr<T = any> = (row: Row, index: number, rows: Row[]) => T;
export type Pred = (row: Row, index: number, rows: Row[]) => boolean;

export type ColumnSpec =
  | string
  | Expr<any>
  | { as: string; expr: Expr<any> }
  | [string, Expr<any>]; // [alias, expr]

export type OrderKey =
  | string
  | Expr<any>
  | {
      key: string | Expr<any>;
      desc?: boolean;
      nulls?: "first" | "last";
    };

export type DistinctSpec =
  | boolean
  | string[]
  | ((row: Row) => any | any[]);

export type SelectOptions = {
  from: Row[];
  where?: Pred;
  columns?: ColumnSpec[];    // default: pass-through (shallow clone)
  distinct?: DistinctSpec;   // default: false
  orderBy?: OrderKey[];      // stable sort
  limit?: number;
  offset?: number;
};

export type GroupKeySpec = string | Expr<any>;
export type GroupOptions = {
  from: Row[];
  groupBy: GroupKeySpec[];   // one or more keys
  // aggregates: alias -> spec
  aggregates: Record<
    string,
    {
      fn:
        | "count"
        | "sum"
        | "avg"
        | "min"
        | "max"
        | "first"
        | "last"
        | "median"
        | "std"
        | "var";
      expr?: Expr<any>;       // default for count(): none
    }
  >;
  having?: (groupRow: Row, rows: Row[]) => boolean;
  orderBy?: OrderKey[];
  limit?: number;
  offset?: number;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * High-level
 * ────────────────────────────────────────────────────────────────────────── */

export function select(opts: SelectOptions): Row[] {
  const src = Array.isArray(opts.from) ? opts.from : [];
  const filtered = opts.where ? where(src, opts.where) : src.slice();

  const projected =
    opts.columns && opts.columns.length
      ? project(filtered, opts.columns)
      : cloneRows(filtered);

  const dedup =
    opts.distinct ? distinctRows(projected, opts.distinct) : projected;

  const ordered =
    opts.orderBy && opts.orderBy.length ? order(dedup, opts.orderBy) : dedup;

  const start = Math.max(0, opts.offset || 0);
  const end =
    typeof opts.limit === "number" ? start + Math.max(0, opts.limit) : ordered.length;

  return ordered.slice(start, end);
}

export function groupSelect(opts: GroupOptions): Row[] {
  const groups = groupBy(opts.from, opts.groupBy);
  const aggRows = aggregate(groups, opts.aggregates);

  const afterHaving = opts.having
    ? aggRows.filter((g) => opts.having!(g, groups[g.__key].rows))
    : aggRows;

  const ordered =
    opts.orderBy && opts.orderBy.length ? order(afterHaving, opts.orderBy) : afterHaving;

  const start = Math.max(0, opts.offset || 0);
  const end =
    typeof opts.limit === "number" ? start + Math.max(0, opts.limit) : ordered.length;

  // strip internal __key
  const out = ordered.slice(start, end).map((r) => {
    const o: Row = {};
    for (const k in r) if (k !== "__key") o[k] = r[k];
    return o;
  });
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Building blocks
 * ────────────────────────────────────────────────────────────────────────── */

export function where(rows: Row[], pred: Pred): Row[] {
  const out: Row[] = [];
  for (let i = 0; i < rows.length; i++) if (pred(rows[i], i, rows)) out.push(rows[i]);
  return out;
}

export function project(rows: Row[], columns: ColumnSpec[]): Row[] {
  const specs = normalizeColumns(columns);
  const out: Row[] = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const src = rows[i];
    const dst: Row = {};
    for (let c = 0; c < specs.length; c++) {
      const s = specs[c];
      dst[s.as] = s.kind === "path" ? getPath(src, s.path) : s.expr!(src, i, rows);
    }
    out[i] = dst;
  }
  return out;
}

export function order(rows: Row[], keys: OrderKey[]): Row[] {
  const ks = normalizeOrder(keys);
  if (ks.length === 0 || rows.length <= 1) return rows.slice();

  // Schwartzian transform with stability
  const decorated = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const keyvals = new Array(ks.length);
    for (let k = 0; k < ks.length; k++) {
      const kk = ks[k];
      keyvals[k] = kk.kind === "path" ? getPath(rows[i], kk.path) : kk.expr!(rows[i], i, rows);
    }
    decorated[i] = { i, r: rows[i], kv: keyvals };
  }

  decorated.sort((a, b) => {
    for (let k = 0; k < ks.length; k++) {
      const kk = ks[k];
      const va = a.kv[k];
      const vb = b.kv[k];
      const cmp = compareValues(va, vb, kk.desc, kk.nulls);
      if (cmp !== 0) return cmp;
    }
    return a.i - b.i; // stable
  });

  const out = new Array(rows.length);
  for (let i = 0; i < decorated.length; i++) out[i] = decorated[i].r;
  return out;
}

export function distinctRows(rows: Row[], spec: DistinctSpec): Row[] {
  if (!spec) return rows.slice();
  const out: Row[] = [];
  const seen = new Set<string>();

  const keyFn =
    typeof spec === "function"
      ? (r: Row) => toKey(spec(r))
      : Array.isArray(spec) && spec.length
      ? (r: Row) => toKey(spec.map((p) => getPath(r, p)))
      : (r: Row) => toKey(r);

  for (let i = 0; i < rows.length; i++) {
    const k = keyFn(rows[i]);
    if (!seen.has(k)) {
      seen.add(k);
      out.push(rows[i]);
    }
  }
  return out;
}

export type Group = { key: any[]; rows: Row[] };
export type GroupIndex = { [key: string]: Group };

export function groupBy(rows: Row[], keys: GroupKeySpec[]): GroupIndex {
  const ns = normalizeGroupKeys(keys);
  const idx: GroupIndex = Object.create(null);

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const keyVals = new Array(ns.length);
    for (let k = 0; k < ns.length; k++) {
      const s = ns[k];
      keyVals[k] = s.kind === "path" ? getPath(r, s.path) : s.expr!(r, i, rows);
    }
    const kstr = toKey(keyVals);
    let g = idx[kstr];
    if (!g) {
      g = idx[kstr] = { key: keyVals, rows: [] };
    }
    g.rows.push(r);
  }
  return idx;
}

export function aggregate(groups: GroupIndex, aggs: GroupOptions["aggregates"]): Row[] {
  const aliases = Object.keys(aggs);
  const out: Row[] = [];

  for (const k in groups) {
    const g = groups[k];
    const row: Row = { __key: k };

    for (let i = 0; i < aliases.length; i++) {
      const alias = aliases[i];
      const spec = aggs[alias];
      row[alias] = runAgg(spec.fn, g.rows, spec.expr);
    }

    // also expose group key parts as key_0, key_1,... for convenience
    for (let j = 0; j < g.key.length; j++) row[`key_${j}`] = g.key[j];

    out.push(row);
  }

  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Aggregations
 * ────────────────────────────────────────────────────────────────────────── */

function runAgg(
  fn:
    | "count"
    | "sum"
    | "avg"
    | "min"
    | "max"
    | "first"
    | "last"
    | "median"
    | "std"
    | "var",
  rows: Row[],
  expr?: Expr<any>
): any {
  const N = rows.length;

  if (fn === "count") {
    if (!expr) return N;
    let n = 0;
    for (let i = 0; i < N; i++) if (truthy(expr(rows[i], i, rows))) n++;
    return n;
  }

  const vals: any[] = new Array(N);
  for (let i = 0; i < N; i++) vals[i] = expr ? expr(rows[i], i, rows) : rows[i];

  if (fn === "first") return N ? vals[0] : undefined;
  if (fn === "last") return N ? vals[N - 1] : undefined;

  if (fn === "min" || fn === "max") {
    let best: any = undefined;
    for (let i = 0; i < N; i++) {
      const v = vals[i];
      if (v == null) continue;
      if (best === undefined) best = v;
      else {
        if (fn === "min") best = compareValues(v, best, false, "last") < 0 ? v : best;
        else best = compareValues(v, best, false, "last") > 0 ? v : best;
      }
    }
    return best;
  }

  // numeric-only
  const nums: number[] = [];
  for (let i = 0; i < N; i++) {
    const v = toNumberMaybe(vals[i]);
    if (v != null && isFinite(v)) nums.push(v);
  }

  if (fn === "sum") {
    let s = 0;
    for (let i = 0; i < nums.length; i++) s += nums[i];
    return s;
  }

  if (fn === "avg") {
    if (!nums.length) return undefined;
    let s = 0;
    for (let i = 0; i < nums.length; i++) s += nums[i];
    return s / nums.length;
  }

  if (fn === "median") {
    if (!nums.length) return undefined;
    nums.sort((a, b) => a - b);
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
    }

  if (fn === "std" || fn === "var") {
    if (nums.length < 2) return 0;
    let s = 0;
    for (let i = 0; i < nums.length; i++) s += nums[i];
    const m = s / nums.length;
    let v = 0;
    for (let i = 0; i < nums.length; i++) { const d = nums[i] - m; v += d * d; }
    const variance = v / (nums.length - 1);
    return fn === "var" ? variance : Math.sqrt(variance);
  }

  return undefined;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Normalizers
 * ────────────────────────────────────────────────────────────────────────── */

function normalizeColumns(cols: ColumnSpec[]): Array<{ as: string; kind: "path" | "expr"; path?: string; expr?: Expr<any> }> {
  const out: Array<{ as: string; kind: "path" | "expr"; path?: string; expr?: Expr<any> }> = [];
  for (let i = 0; i < cols.length; i++) {
    const c = cols[i] as any;
    if (typeof c === "string") {
      out.push({ as: lastSegment(c), kind: "path", path: c });
    } else if (typeof c === "function") {
      out.push({ as: `c${i}`, kind: "expr", expr: c });
    } else if (Array.isArray(c) && c.length >= 2 && typeof c[1] === "function") {
      out.push({ as: String(c[0]), kind: "expr", expr: c[1] });
    } else if (c && typeof c === "object" && typeof c.expr === "function") {
      out.push({ as: String(c.as), kind: "expr", expr: c.expr });
    }
  }
  return out;
}

function normalizeOrder(keys: OrderKey[]): Array<{ kind: "path" | "expr"; path?: string; expr?: Expr<any>; desc: boolean; nulls: "first" | "last" }> {
  const out: Array<{ kind: "path" | "expr"; path?: string; expr?: Expr<any>; desc: boolean; nulls: "first" | "last" }> = [];
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i] as any;
    if (typeof k === "string") {
      out.push({ kind: "path", path: k, desc: false, nulls: "last" });
    } else if (typeof k === "function") {
      out.push({ kind: "expr", expr: k, desc: false, nulls: "last" });
    } else if (k && typeof k === "object") {
      const key = k.key;
      const desc = !!k.desc;
      const nulls = k.nulls === "first" ? "first" : "last";
      if (typeof key === "string") out.push({ kind: "path", path: key, desc, nulls });
      else if (typeof key === "function") out.push({ kind: "expr", expr: key, desc, nulls });
    }
  }
  return out;
}

function normalizeGroupKeys(keys: GroupKeySpec[]): Array<{ kind: "path" | "expr"; path?: string; expr?: Expr<any> }> {
  const out: Array<{ kind: "path" | "expr"; path?: string; expr?: Expr<any> }> = [];
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i] as any;
    if (typeof k === "string") out.push({ kind: "path", path: k });
    else if (typeof k === "function") out.push({ kind: "expr", expr: k });
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Utilities
 * ────────────────────────────────────────────────────────────────────────── */

function cloneRows(rows: Row[]): Row[] {
  const out: Row[] = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const o: Row = {};
    for (const k in r) o[k] = r[k];
    out[i] = o;
  }
  return out;
}

function getPath(obj: any, path: string): any {
  if (!obj || !path) return undefined;
  // dot + bracket notation (very small)
  let cur: any = obj;
  const parts = path.replace(/\[(\d+)\]/g, ".$1").split(".");
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    if (!p) continue;
    if (cur == null) return undefined;
    cur = cur[p];
  }
  return cur;
}

function lastSegment(p: string): string {
  const s = p.replace(/\[(\d+)\]/g, ".$1");
  const segs = s.split(".").filter(Boolean);
  return segs.length ? segs[segs.length - 1] : p;
}

function toKey(v: any): string {
  try {
    if (Array.isArray(v)) return JSON.stringify(v);
    if (v && typeof v === "object") return JSON.stringify(v);
    return String(v);
  } catch {
    return String(v);
  }
}

function compareValues(a: any, b: any, desc?: boolean, nulls?: "first" | "last"): number {
  const na = a == null, nb = b == null;
  if (na || nb) {
    if (na && nb) return 0;
    const nf = nulls === "first";
    return (na ? (nf ? -1 : 1) : (nf ? 1 : -1)) * (desc ? -1 : 1);
  }

  // Numbers (including numeric strings)
  const an = toNumberMaybe(a);
  const bn = toNumberMaybe(b);
  if (an != null && bn != null && isFinite(an) && isFinite(bn)) {
    return (an < bn ? -1 : an > bn ? 1 : 0) * (desc ? -1 : 1);
  }

  // Dates (ISO strings)
  const ad = toEpochMaybe(a);
  const bd = toEpochMaybe(b);
  if (ad != null && bd != null) return (ad < bd ? -1 : ad > bd ? 1 : 0) * (desc ? -1 : 1);

  // Fallback string compare
  const as = String(a);
  const bs = String(b);
  return (as < bs ? -1 : as > bs ? 1 : 0) * (desc ? -1 : 1);
}

function toNumberMaybe(x: any): number | null {
  if (typeof x === "number") return x;
  if (typeof x === "boolean") return x ? 1 : 0;
  if (x instanceof Date) return x.getTime();
  if (x == null) return null;
  const s = String(x).trim().replace(/[_ ,]/g, "");
  if (!s) return null;
  const n = Number(s);
  return isFinite(n) ? n : null;
}

function toEpochMaybe(x: any): number | null {
  if (x instanceof Date) return x.getTime();
  if (typeof x === "number" && isFinite(x)) return x > 1e12 ? x : x * 1000; // allow seconds
  if (x == null) return null;
  const s = String(x).trim();
  // Accept YYYY-MM-DD or ISO-like
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) {
    const t = Date.parse(s + "T00:00:00Z");
    return isNaN(t) ? null : t;
  }
  if (/^\d{4}-\d{2}-\d{2}[tT ][\d:.]+([zZ]|[+-]\d{2}:?\d{2})?$/.test(s)) {
    const t = Date.parse(s);
    return isNaN(t) ? null : t;
  }
  return null;
}

function truthy(v: any): boolean {
  if (v == null) return false;
  if (typeof v === "number") return v !== 0 && !isNaN(v);
  if (typeof v === "string") return v.trim() !== "" && v !== "0" && v.toLowerCase() !== "false";
  return !!v;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */