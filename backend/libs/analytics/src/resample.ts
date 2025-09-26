// analytics/src/resample.ts
// Pure TypeScript resampling utilities (no imports).
// Works with simple { t: "YYYY-MM-DD", v: number } series and OHLC(V) rows.

export type Dict<T = any> = { [k: string]: T };

export type Point = { t: string; v: number | null | undefined };
export type Series = Point[];

export type OHLC = { t: string; o?: number | null; h?: number | null; l?: number | null; c?: number | null };
export type OHLCV = OHLC & { v?: number | null };

export type Freq = "D" | "W" | "M" | "Q" | "A"; // Daily, Weekly(ISO Mon-Sun), Monthly, Quarterly, Annual
export type FillMethod = "none" | "ffill" | "bfill" | "zero";

export type AggName =
  | "last" | "first"
  | "sum" | "mean" | "min" | "max" | "median";

/* ────────────────────────────────────────────────────────────────────────── *
 * Small shared helpers (duplicated locally to keep zero imports)
 * ────────────────────────────────────────────────────────────────────────── */

function isFiniteNum(x: any): x is number { return typeof x === "number" && isFinite(x); }

function pad2(n: number): string { return String(n).padStart(2, "0"); }

function toDateOnlyISO(v: any): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "string") {
    const s = v.trim();
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[1]}-${m[2]}-${m[3]}`;
    const t = Date.parse(s);
    if (!isNaN(t)) {
      const d = new Date(t);
      return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
    }
    return null;
  }
  if (v instanceof Date) {
    return `${v.getUTCFullYear()}-${pad2(v.getUTCMonth() + 1)}-${pad2(v.getUTCDate())}`;
  }
  return null;
}

function toNumberSafe(v: any): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number") return isFinite(v) ? v : null;
  const n = Number(String(v).replace(/[_ ,]/g, ""));
  return isFinite(n) ? n : null;
}

function cleanSeries(s: Series): Series {
  const out: Series = [];
  for (let i = 0; i < s.length; i++) {
    const t = toDateOnlyISO(s[i].t);
    const v = toNumberSafe(s[i].v);
    if (!t || v == null || !isFiniteNum(v)) continue;
    out.push({ t, v });
  }
  out.sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));
  // dedupe keep last
  const last: Dict<number> = {};
  for (let i = 0; i < out.length; i++) last[out[i].t] = i;
  const uniq: Series = [];
  const keys = Object.keys(last).sort();
  for (let i = 0; i < keys.length; i++) uniq.push(out[last[keys[i]]]);
  return uniq;
}

function dateFromISO(iso: string): Date {
  // Create Date in UTC (no TZ surprises)
  return new Date(iso + "T00:00:00Z");
}

function isoFromDateUTC(d: Date): string {
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
}

function addDaysUTC(d: Date, n: number): Date {
  const x = new Date(d.getTime());
  x.setUTCDate(x.getUTCDate() + n);
  return x;
}

function startOfWeekISO(iso: string): string {
  const d = dateFromISO(iso);
  const dow = d.getUTCDay();                 // 0=Sun..6=Sat
  const delta = dow === 0 ? -6 : 1 - dow;    // ISO week starts Monday
  return isoFromDateUTC(addDaysUTC(d, delta));
}
function startOfMonth(iso: string): string {
  const d = dateFromISO(iso);
  d.setUTCDate(1);
  return isoFromDateUTC(d);
}
function startOfQuarter(iso: string): string {
  const d = dateFromISO(iso);
  const m = d.getUTCMonth(); // 0..11
  const qStart = m - (m % 3);
  d.setUTCMonth(qStart, 1);
  return isoFromDateUTC(d);
}
function startOfYear(iso: string): string {
  const d = dateFromISO(iso);
  d.setUTCMonth(0, 1);
  return isoFromDateUTC(d);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Domain generators
 * ────────────────────────────────────────────────────────────────────────── */

export function makeDomain(startISO: string, endISO: string, freq: Freq, businessDaysOnly = false): string[] {
  const s = toDateOnlyISO(startISO); const e = toDateOnlyISO(endISO);
  if (!s || !e) return [];
  let cur = dateFromISO(bucketKeyForDate(s, freq));
  const end = dateFromISO(bucketKeyForDate(e, freq));
  const out: string[] = [];
  while (cur.getTime() <= end.getTime()) {
    const key = isoFromDateUTC(cur);
    if (freq === "D" && businessDaysOnly) {
      const dow = cur.getUTCDay();
      if (dow !== 0 && dow !== 6) out.push(key);
    } else {
      out.push(key);
    }
    cur = stepDate(cur, freq);
  }
  return out;
}

function stepDate(d: Date, freq: Freq): Date {
  const x = new Date(d.getTime());
  if (freq === "D") return addDaysUTC(x, 1);
  if (freq === "W") return addDaysUTC(x, 7);
  if (freq === "M") { x.setUTCMonth(x.getUTCMonth() + 1); return x; }
  if (freq === "Q") { x.setUTCMonth(x.getUTCMonth() + 3); return x; }
  // Annual
  x.setUTCFullYear(x.getUTCFullYear() + 1);
  return x;
}

function bucketKeyForDate(iso: string, freq: Freq): string {
  switch (freq) {
    case "D": return toDateOnlyISO(iso) as string;
    case "W": return startOfWeekISO(iso);
    case "M": return startOfMonth(iso);
    case "Q": return startOfQuarter(iso);
    case "A": return startOfYear(iso);
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Aggregation primitives
 * ────────────────────────────────────────────────────────────────────────── */

export function aggregate(values: number[], agg: AggName): number | null {
  const xs = values.filter(isFiniteNum);
  if (!xs.length) return null;
  switch (agg) {
    case "first": return xs[0];
    case "last":  return xs[xs.length - 1];
    case "sum":   return xs.reduce((s, v) => s + v, 0);
    case "mean":  return xs.reduce((s, v) => s + v, 0) / xs.length;
    case "min":   return Math.min.apply(null as any, xs);
    case "max":   return Math.max.apply(null as any, xs);
    case "median": {
      const a = xs.slice().sort((a, b) => a - b);
      const i = Math.floor(a.length / 2);
      return a.length % 2 ? a[i] : (a[i - 1] + a[i]) / 2;
    }
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Resample simple series
 * ────────────────────────────────────────────────────────────────────────── */

export function resampleSeries(src: Series, freq: Freq, agg: AggName = "last"): Series {
  const a = cleanSeries(src);
  if (a.length === 0) return [];
  // Group by bucket
  const groups: Dict<number[]> = Object.create(null);
  const order: Dict<string[]> = Object.create(null); // to preserve order for first/last
  for (let i = 0; i < a.length; i++) {
    const t = a[i].t;
    const key = bucketKeyForDate(t, freq);
    if (!groups[key]) { groups[key] = []; order[key] = []; }
    groups[key].push(a[i].v as number);
    order[key].push(t);
  }
  const keys = Object.keys(groups).sort();
  const out: Series = [];
  for (let k = 0; k < keys.length; k++) {
    const key = keys[k];
    let v: number | null;
    if (agg === "first" || agg === "last") {
      // Use original chronological order inside bucket
      const vals: number[] = [];
      const tList = order[key];
      // collect in order
      for (let i = 0; i < tList.length; i++) vals.push(a.find(p => p.t === tList[i])!.v as number);
      v = aggregate(vals, agg);
    } else {
      v = aggregate(groups[key], agg);
    }
    if (v != null) out.push({ t: key, v });
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Resample OHLC(V)
 * ────────────────────────────────────────────────────────────────────────── */

export type OHLCVOut = { t: string; o: number; h: number; l: number; c: number; v?: number };

export function resampleOHLC(rows: OHLCV[], freq: Freq): OHLCVOut[] {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  // Normalize + sort
  type RowN = { t: string; o?: number; h?: number; l?: number; c?: number; v?: number };
  const norm: RowN[] = [];
  for (let i = 0; i < rows.length; i++) {
    const t = toDateOnlyISO(rows[i].t);
    const o = toNumberSafe(rows[i].o);
    const h = toNumberSafe(rows[i].h);
    const l = toNumberSafe(rows[i].l);
    const c = toNumberSafe(rows[i].c);
    const v = toNumberSafe((rows[i] as any).v);
    if (!t) continue;
    norm.push({ t, o: o ?? undefined, h: h ?? undefined, l: l ?? undefined, c: c ?? undefined, v: v ?? undefined });
  }
  norm.sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));

  // Group by bucket
  type Acc = { t: string; o?: number; h?: number; l?: number; c?: number; v?: number; seenOpen: boolean };
  const map = new Map<string, Acc>();
  const keys: string[] = [];
  for (let i = 0; i < norm.length; i++) {
    const r = norm[i];
    const k = bucketKeyForDate(r.t, freq);
    let g = map.get(k);
    if (!g) { g = { t: k, seenOpen: false }; map.set(k, g); keys.push(k); }
    if (!g.seenOpen && isFiniteNum(r.o)) { g.o = r.o; g.seenOpen = true; }
    if (isFiniteNum(r.h)) g.h = g.h == null ? r.h! : Math.max(g.h!, r.h!);
    if (isFiniteNum(r.l)) g.l = g.l == null ? r.l! : Math.min(g.l!, r.l!);
    if (isFiniteNum(r.c)) g.c = r.c; // last close wins
    if (isFiniteNum(r.v)) g.v = (g.v || 0) + (r.v as number);
  }

  // Emit
  const out: OHLCVOut[] = [];
  keys.sort();
  for (let i = 0; i < keys.length; i++) {
    const g = map.get(keys[i])!;
    if (!isFiniteNum(g.o) || !isFiniteNum(g.h) || !isFiniteNum(g.l) || !isFiniteNum(g.c)) continue;
    const row: OHLCVOut = { t: g.t, o: g.o as number, h: g.h as number, l: g.l as number, c: g.c as number };
    if (isFiniteNum(g.v)) row.v = g.v as number;
    out.push(row);
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Gap filling
 * ────────────────────────────────────────────────────────────────────────── */

export function fillGaps(
  s: Series,
  freq: Freq,
  method: FillMethod = "none",
  opts?: { businessDaysOnly?: boolean; maxGap?: number } // maxGap only applies to ffill/bfill
): Series {
  const a = cleanSeries(s);
  if (!a.length) return [];

  const start = a[0].t;
  const end = a[a.length - 1].t;
  const domain = makeDomain(start, end, freq, !!opts?.businessDaysOnly);

  // Build map
  const m = new Map<string, number>();
  for (let i = 0; i < a.length; i++) m.set(a[i].t, a[i].v as number);

  const out: Series = [];
  let prevVal: number | null = null;
  let prevIdx: number | null = null;

  // If bfill needed, precompute next known value per date
  const nextMap = method === "bfill" ? computeNextMap(domain, m) : undefined;

  for (let di = 0; di < domain.length; di++) {
    const t = domain[di];
    if (m.has(t)) {
      const v = m.get(t)!;
      out.push({ t, v });
      prevVal = v; prevIdx = di;
    } else {
      if (method === "none") {
        out.push({ t, v: null });
      } else if (method === "zero") {
        out.push({ t, v: 0 });
      } else if (method === "ffill") {
        if (prevVal == null) out.push({ t, v: null });
        else if (typeof opts?.maxGap === "number" && prevIdx != null && di - prevIdx > opts.maxGap) {
          out.push({ t, v: null });
        } else {
          out.push({ t, v: prevVal });
        }
      } else if (method === "bfill") {
        const nxt = nextMap!.get(t) ?? null;
        if (nxt == null) out.push({ t, v: null });
        else {
          // optional maxGap for bfill: measure forward distance
          if (typeof opts?.maxGap === "number") {
            const gap = distanceInSteps(t, nxt.t, freq);
            out.push({ t, v: gap != null && gap > opts.maxGap ? null : nxt.v });
          } else {
            out.push({ t, v: nxt.v });
          }
        }
      }
    }
  }
  return out;
}

function computeNextMap(domain: string[], m: Map<string, number>): Map<string, { t: string; v: number }> {
  const res = new Map<string, { t: string; v: number }>();
  let nextT: string | null = null;
  let nextV: number | null = null;
  for (let i = domain.length - 1; i >= 0; i--) {
    const t = domain[i];
    if (m.has(t)) { nextT = t; nextV = m.get(t)!; }
    if (nextT != null && nextV != null) res.set(t, { t: nextT, v: nextV });
  }
  return res;
}

function distanceInSteps(aISO: string, bISO: string, freq: Freq): number | null {
  const a = dateFromISO(bucketKeyForDate(aISO, freq)).getTime();
  const b = dateFromISO(bucketKeyForDate(bISO, freq)).getTime();
  if (b < a) return null;
  if (freq === "D") return Math.round((b - a) / (24 * 3600 * 1000));
  if (freq === "W") return Math.round((b - a) / (7 * 24 * 3600 * 1000));
  if (freq === "M") return monthsBetween(aISO, bISO);
  if (freq === "Q") return Math.floor(monthsBetween(aISO, bISO) / 3);
  return dateFromISO(bISO).getUTCFullYear() - dateFromISO(aISO).getUTCFullYear();
}

function monthsBetween(aISO: string, bISO: string): number {
  const a = dateFromISO(aISO), b = dateFromISO(bISO);
  return (b.getUTCFullYear() - a.getUTCFullYear()) * 12 + (b.getUTCMonth() - a.getUTCMonth());
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Convenience helpers
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Downsample daily series to business days only (removing weekends).
 * If `fill` is provided, gaps are forward-filled over business days.
 */
export function toBusinessDays(s: Series, fill: FillMethod = "none", maxGap?: number): Series {
  const a = cleanSeries(s);
  if (!a.length) return [];
  const domain = makeDomain(a[0].t, a[a.length - 1].t, "D", true);
  const map = new Map<string, number>();
  for (let i = 0; i < a.length; i++) map.set(a[i].t, a[i].v as number);
  return fillGaps(Array.from(map.entries()).map(([t, v]) => ({ t, v })), "D", fill, { businessDaysOnly: true, maxGap });
}

/**
 * Uniformly sample series on a domain, using provided fill method.
 */
export function alignToDomain(s: Series, domain: string[], method: FillMethod = "none", maxGap?: number): Series {
  if (!domain || !domain.length) return [];
  const a = cleanSeries(s);
  const m = new Map<string, number>();
  for (let i = 0; i < a.length; i++) m.set(a[i].t, a[i].v as number);
  const out: Series = [];
  let prevVal: number | null = null;
  let prevIdx: number | null = null;
  // Precompute next for bfill
  const nextMap = method === "bfill" ? computeNextMap(domain, m) : undefined;
  for (let i = 0; i < domain.length; i++) {
    const t = domain[i];
    if (m.has(t)) {
      const v = m.get(t)!;
      out.push({ t, v });
      prevVal = v; prevIdx = i;
    } else {
      if (method === "none") out.push({ t, v: null });
      else if (method === "zero") out.push({ t, v: 0 });
      else if (method === "ffill") {
        if (prevVal == null) out.push({ t, v: null });
        else if (typeof maxGap === "number" && prevIdx != null && i - prevIdx > maxGap) out.push({ t, v: null });
        else out.push({ t, v: prevVal });
      } else if (method === "bfill") {
        const nxt = nextMap!.get(t);
        out.push({ t, v: nxt ? nxt.v : null });
      }
    }
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Examples (commented)
 *
 * // const px: Series = [{ t: "2025-01-02", v: 100 }, { t: "2025-01-03", v: 101 }, ...];
 * // const weeklyLast = resampleSeries(px, "W", "last");
 * // const monthlyMean = resampleSeries(px, "M", "mean");
 *
 * // OHLCV downsample:
 * // const bars: OHLCV[] = [{ t:"2025-01-02", o:100, h:105, l:99, c:102, v: 1_000_000 }, ...];
 * // const weeklyBars = resampleOHLC(bars, "W");
 *
 * // Business-day domain + forward fill:
 * // const bd = toBusinessDays(px, "ffill");
 *
 * // Align to monthly buckets with bfill:
 * // const domain = makeDomain("2024-01-01", "2024-12-31", "M");
 * // const aligned = alignToDomain(px, domain, "bfill");
 * ────────────────────────────────────────────────────────────────────────── */