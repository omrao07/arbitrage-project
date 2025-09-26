// analytics/src/join.ts
// Pure TypeScript helpers for joining arrays of objects and time series.
// No imports. Zero-dependency. Safe for Node or browser.

export type Dict<T = any> = { [k: string]: T };

export type JoinHow = "inner" | "left" | "right" | "full";
export type KeySpec<T> = string | string[] | ((row: T) => any);

export type MergeStrategy = "prefer_left" | "prefer_right" | "prefer_left_non_null" | "prefer_right_non_null" | "both";

export type JoinOptions<L extends Dict = Dict, R extends Dict = Dict> = {
  /** What rows to return when no match on the other side */
  how?: JoinHow;
  /** If true, include the join key as `_key` in the output */
  includeKey?: boolean;
  /** Resolve field name collisions (same property on left and right) */
  merge?: MergeStrategy;
  /** Suffixes when merge === "both" (default: ["_left","_right"]) */
  suffix?: [string, string];
  /** Emit only these fields from the left side (empty => all) */
  pickLeft?: string[];
  /** Emit only these fields from the right side (empty => all) */
  pickRight?: string[];
  /** Custom field merger; takes precedence over `merge` if provided */
  onFieldConflict?: (field: string, lVal: any, rVal: any) => { field?: string; value?: any; extra?: Dict };
  /** Name for the left/right wrappers when using object grouping helpers */
  leftLabel?: string;
  rightLabel?: string;
};

export type JoinOn<L, R> = {
  left: KeySpec<L>;
  right?: KeySpec<R>; // default -> same as left
};

export type JoinedRow<L extends Dict, R extends Dict> = Dict & { _left?: L; _right?: R; _key?: string };

/* ────────────────────────────────────────────────────────────────────────── *
 * Public API
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Hash join for two arrays of objects.
 * - Supports 1..N keys per side or custom key function(s).
 * - Many-to-many friendly (cross-product on match).
 * - Collisions handled by `merge` strategy or a custom `onFieldConflict`.
 */
export function hashJoin<L extends Dict, R extends Dict>(
  leftRows: L[],
  rightRows: R[],
  on: JoinOn<L, R>,
  options?: JoinOptions<L, R>
): JoinedRow<L, R>[] {
  const opts = withDefaults(options);
  const keyL = toKeyFn(on.left);
  const keyR = toKeyFn(on.right ?? on.left as any);

  const pickedLeft = opts.pickLeft && opts.pickLeft.length > 0;
  const pickedRight = opts.pickRight && opts.pickRight.length > 0;

  // Build index for right
  const idxRight = new Map<string, R[]>();
  for (let i = 0; i < rightRows.length; i++) {
    const r = rightRows[i];
    const k = normKeyValue(keyR(r));
    let bucket = idxRight.get(k);
    if (!bucket) { bucket = []; idxRight.set(k, bucket); }
    bucket.push(r);
  }

  // Track which right rows got matched
  const matchedR = new WeakSet<R>();

  const out: JoinedRow<L, R>[] = [];

  // Probe from left
  for (let i = 0; i < leftRows.length; i++) {
    const l = leftRows[i];
    const k = normKeyValue(keyL(l));
    const matches = idxRight.get(k);

    if (matches && matches.length > 0) {
      for (let j = 0; j < matches.length; j++) {
        const r = matches[j];
        matchedR.add(r);
        out.push(mergeRows(l, r, k, opts, pickedLeft, pickedRight));
      }
    } else {
      if (opts.how === "left" || opts.how === "full") {
        out.push(mergeRows(l, undefined, k, opts, pickedLeft, pickedRight));
      }
    }
  }

  if (opts.how === "right" || opts.how === "full") {
    for (let i = 0; i < rightRows.length; i++) {
      const r = rightRows[i];
      if (!matchedR.has(r)) {
        const k = normKeyValue(keyR(r));
        out.push(mergeRows(undefined, r, k, opts, pickedLeft, pickedRight));
      }
    }
  }

  return out;
}

/**
 * Grouped join: returns objects like
 *   { key, left: L[], right: R[] }
 * Useful for later custom merges or aggregate operations.
 */
export function groupJoin<L extends Dict, R extends Dict>(
  leftRows: L[],
  rightRows: R[],
  on: JoinOn<L, R>
): Array<{ key: string; left: L[]; right: R[] }> {
  const keyL = toKeyFn(on.left);
  const keyR = toKeyFn(on.right ?? on.left as any);
  const groups = new Map<string, { left: L[]; right: R[] }>();

  for (let i = 0; i < leftRows.length; i++) {
    const l = leftRows[i];
    const k = normKeyValue(keyL(l));
    let g = groups.get(k);
    if (!g) { g = { left: [], right: [] }; groups.set(k, g); }
    g.left.push(l);
  }
  for (let j = 0; j < rightRows.length; j++) {
    const r = rightRows[j];
    const k = normKeyValue(keyR(r));
    let g = groups.get(k);
    if (!g) { g = { left: [], right: [] }; groups.set(k, g); }
    g.right.push(r);
  }

  const out: Array<{ key: string; left: L[]; right: R[] }> = [];
  groups.forEach((v, k) => out.push({ key: k, left: v.left, right: v.right }));
  out.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  return out;
}

/**
 * Convenience: join two time series arrays by date.
 * - Each input is an array of `{ t: string, v: number }`.
 * - Returns rows `[t, vLeft, vRight]` aligned by `how`.
 */
export function joinSeriesByDate(
  left: Array<{ t: string; v: number | null | undefined }>,
  right: Array<{ t: string; v: number | null | undefined }>,
  how: JoinHow = "inner"
): Array<[string, number | null, number | null]> {
  const lr = left.map(x => ({ t: normDate(x.t), v: toNumOrNull(x.v) })).filter(x => x.t);
  const rr = right.map(x => ({ t: normDate(x.t), v: toNumOrNull(x.v) })).filter(x => x.t);

  // Build maps
  const mL = new Map<string, number | null>();
  const mR = new Map<string, number | null>();
  for (let i = 0; i < lr.length; i++) mL.set(lr[i].t!, lr[i].v);
  for (let j = 0; j < rr.length; j++) mR.set(rr[j].t!, rr[j].v);

  // Dates set
  const dates = new Set<string>();
  if (how === "inner") {
    mL.forEach((_v, k) => { if (mR.has(k)) dates.add(k); });
  } else if (how === "left") {
    mL.forEach((_v, k) => dates.add(k));
  } else if (how === "right") {
    mR.forEach((_v, k) => dates.add(k));
  } else { // full
    mL.forEach((_v, k) => dates.add(k));
    mR.forEach((_v, k) => dates.add(k));
  }

  const keys = Array.from(dates).sort();
  const out: Array<[string, number | null, number | null]> = [];
  for (let i = 0; i < keys.length; i++) {
    const t = keys[i];
    out.push([t, mL.get(t) ?? null, mR.get(t) ?? null]);
  }
  return out;
}

/**
 * Forward-fill a time series' missing values over a provided date domain.
 * - `domain` is an ordered list of dates (YYYY-MM-DD).
 * - If `maxGap` is set, do not fill across gaps longer than `maxGap` days.
 */
export function fillForwardSeries(
  series: Array<{ t: string; v: number | null | undefined }>,
  domain: string[],
  maxGap?: number
): Array<{ t: string; v: number | null }> {
  const s = new Map<string, number | null>();
  for (let i = 0; i < series.length; i++) {
    const t = normDate(series[i].t);
    if (!t) continue;
    const v = toNumOrNull(series[i].v);
    s.set(t, v);
  }
  const out: Array<{ t: string; v: number | null }> = [];
  let lastVal: number | null = null;
  let lastIdx: number | null = null;
  for (let di = 0; di < domain.length; di++) {
    const t = normDate(domain[di]);
    if (!t) continue;
    const cur = s.has(t) ? s.get(t)! : null;
    if (cur != null) {
      lastVal = cur;
      lastIdx = di;
      out.push({ t, v: cur });
    } else {
      if (lastVal == null) {
        out.push({ t, v: null });
      } else {
        if (typeof maxGap === "number" && lastIdx != null && di - lastIdx > maxGap) {
          out.push({ t, v: null }); // gap too large
        } else {
          out.push({ t, v: lastVal });
        }
      }
    }
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals
 * ────────────────────────────────────────────────────────────────────────── */

function withDefaults<L extends Dict, R extends Dict>(opts?: JoinOptions<L, R>): Required<JoinOptions<L, R>> {
  return {
    how: opts?.how ?? "inner",
    includeKey: !!opts?.includeKey,
    merge: opts?.merge ?? "prefer_left_non_null",
    suffix: opts?.suffix ?? ["_left", "_right"],
    pickLeft: opts?.pickLeft ?? [],
    pickRight: opts?.pickRight ?? [],
    onFieldConflict: opts?.onFieldConflict ?? undefined,
    leftLabel: opts?.leftLabel ?? "left",
    rightLabel: opts?.rightLabel ?? "right"
  };
}

function toKeyFn<T>(spec: KeySpec<T>): (row: T) => string {
  if (typeof spec === "function") {
    return (row: T) => normKeyValue((spec as any)(row));
  }
  if (typeof spec === "string") {
    return (row: any) => normKeyValue(row[spec]);
  }
  // array of fields
  const fields = (spec as string[]).slice();
  return (row: any) => {
    const parts: string[] = [];
    for (let i = 0; i < fields.length; i++) {
      parts.push(normKeyValue(row[fields[i]]));
    }
    return parts.join("|");
  };
}

function normKeyValue(v: any): string {
  if (v == null) return "";
  if (v instanceof Date) return toDateOnly(v);
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  const s = String(v);
  // If looks like ISO date/time, reduce to YYYY-MM-DD (for stable joins)
  const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : s;
}

function toDateOnly(d: Date): string {
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(d.getUTCDate()).padStart(2, "0");
  return `${d.getUTCFullYear()}-${mm}-${dd}`;
}

function normDate(v: any): string | null {
  if (v == null || v === "") return null;
  if (v instanceof Date) return toDateOnly(v);
  const s = String(v).trim();
  const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  const t = Date.parse(s);
  if (!isNaN(t)) return toDateOnly(new Date(t));
  return null;
}

function toNumOrNull(v: any): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number" && isFinite(v)) return v;
  const n = Number(String(v).replace(/[_ ,]/g, ""));
  return isFinite(n) ? n : null;
}

function mergeRows<L extends Dict, R extends Dict>(
  l: L | undefined,
  r: R | undefined,
  key: string,
  opts: Required<JoinOptions<L, R>>,
  pickedLeft: boolean,
  pickedRight: boolean
): JoinedRow<L, R> {
  const out: JoinedRow<L, R> = {};
  if (opts.includeKey) out._key = key;

  // Copy left fields
  if (l) {
    if (pickedLeft) {
      for (let i = 0; i < opts.pickLeft.length; i++) {
        const f = opts.pickLeft[i];
        if (f in l) (out as any)[f] = (l as any)[f];
      }
    } else {
      for (const k in l) (out as any)[k] = (l as any)[k];
    }
  }

  // Merge right fields
  if (r) {
    if (pickedRight) {
      for (let i = 0; i < opts.pickRight.length; i++) {
        const f = opts.pickRight[i];
        if (f in r) mergeField(out, f, (r as any)[f], opts);
      }
    } else {
      for (const k in r) mergeField(out, k, (r as any)[k], opts);
    }
  }

  // Attach raw pointers (useful for debugging or advanced consumers)
  if (l) out._left = l;
  if (r) out._right = r;

  return out;
}

function mergeField(target: Dict, field: string, rVal: any, opts: Required<JoinOptions>): void {
  if (!(field in target)) {
    target[field] = rVal;
    return;
  }
  const lVal = target[field];

  if (opts.onFieldConflict) {
    const res = opts.onFieldConflict(field, lVal, rVal);
    if (res) {
      if (res.field) {
        target[res.field] = res.value;
      } else if ("value" in res) {
        target[field] = res.value;
      }
      if (res.extra) {
        for (const ek in res.extra) target[ek] = (res.extra as any)[ek];
      }
      return;
    }
  }

  switch (opts.merge) {
    case "prefer_left":
      // keep existing
      return;
    case "prefer_right":
      target[field] = rVal;
      return;
    case "prefer_left_non_null":
      target[field] = coalesceLeftNonNull(lVal, rVal);
      return;
    case "prefer_right_non_null":
      target[field] = coalesceLeftNonNull(rVal, lVal);
      return;
    case "both":
      target[field + opts.suffix[0]] = lVal;
      target[field + opts.suffix[1]] = rVal;
      // keep original as left for stability
      target[field] = lVal;
      return;
  }
}

function coalesceLeftNonNull(a: any, b: any): any {
  return (a !== null && a !== undefined && a !== "") ? a : b;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Convenience shorthands for common joins
 * ────────────────────────────────────────────────────────────────────────── */

export function innerJoin<L extends Dict, R extends Dict>(
  leftRows: L[],
  rightRows: R[],
  on: JoinOn<L, R>,
  options?: Omit<JoinOptions<L, R>, "how">
): JoinedRow<L, R>[] {
  return hashJoin(leftRows, rightRows, on, { ...(options || {}), how: "inner" });
}

export function leftJoin<L extends Dict, R extends Dict>(
  leftRows: L[],
  rightRows: R[],
  on: JoinOn<L, R>,
  options?: Omit<JoinOptions<L, R>, "how">
): JoinedRow<L, R>[] {
  return hashJoin(leftRows, rightRows, on, { ...(options || {}), how: "left" });
}

export function rightJoin<L extends Dict, R extends Dict>(
  leftRows: L[],
  rightRows: R[],
  on: JoinOn<L, R>,
  options?: Omit<JoinOptions<L, R>, "how">
): JoinedRow<L, R>[] {
  return hashJoin(leftRows, rightRows, on, { ...(options || {}), how: "right" });
}

export function fullJoin<L extends Dict, R extends Dict>(
  leftRows: L[],
  rightRows: R[],
  on: JoinOn<L, R>,
  options?: Omit<JoinOptions<L, R>, "how">
): JoinedRow<L, R>[] {
  return hashJoin(leftRows, rightRows, on, { ...(options || {}), how: "full" });
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Example usage (commented)
 *
 * // const a = [{ id: 1, x: "A" }, { id: 2, x: "B" }];
 * // const b = [{ id: 2, y: 10 }, { id: 3, y: 20 }];
 * // const rows = fullJoin(a, b, { left: "id" }, { includeKey: true, merge: "both" });
 * // // -> rows with keys 1,2,3; for id=2 you get merged fields (x,y).
 *
 * // Time series:
 * // const L = [{ t: "2025-01-01", v: 10 }, { t: "2025-01-02", v: 11 }];
 * // const R = [{ t: "2025-01-02", v: 20 }, { t: "2025-01-03", v: 21 }];
 * // const aligned = joinSeriesByDate(L, R, "full");
 * // // -> [
 * // //   ["2025-01-01", 10, null],
 * // //   ["2025-01-02", 11, 20],
 * // //   ["2025-01-03", null, 21]
 * // // ]
 * ────────────────────────────────────────────────────────────────────────── */