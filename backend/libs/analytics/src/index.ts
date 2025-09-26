// analytics/src/index.ts
// Zero-dependency analytics helpers for time series, returns, risk, and portfolio math.
// Pure TypeScript. No imports. Works in Node or the browser.

export type Dict<T = any> = { [k: string]: T };

/* ────────────────────────────────────────────────────────────────────────── *
 * Types
 * ────────────────────────────────────────────────────────────────────────── */

export type Point = { t: string; v: number | null | undefined };
export type Series = Point[];

export type JoinMethod = "inner" | "left"; // base on first series for "left"
export type Annualization = 252 | 12 | 52 | 365; // trading days, months, weeks, days

export type StatsSummary = {
  n: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  sum: number;
  p50: number;
  p95: number;
  p99: number;
};

export type BetaAlpha = {
  beta: number;
  alpha_annual: number;
  r2: number;
  cov: number;
  var_bm: number;
  corr: number;
};

export type DrawdownPoint = { t: string; equity: number; peak: number; dd: number };

export type PortfolioWeights = { [symbol: string]: number };

/* ────────────────────────────────────────────────────────────────────────── *
 * Core utilities
 * ────────────────────────────────────────────────────────────────────────── */

export function isFiniteNum(x: any): x is number {
  return typeof x === "number" && isFinite(x);
}

export function clamp(n: number, lo: number, hi: number): number {
  return n < lo ? lo : n > hi ? hi : n;
}

export function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

export function toDateOnlyISO(v: any): string | null {
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

export function sortSeries(s: Series): Series {
  const a = s.slice(0);
  a.sort((x, y) => (x.t < y.t ? -1 : x.t > y.t ? 1 : 0));
  return a;
}

export function dedupeKeepLast(s: Series): Series {
  const map: Dict<number> = {};
  for (let i = 0; i < s.length; i++) map[s[i].t] = i;
  const out: Series = [];
  const keys = Object.keys(map).sort();
  for (let i = 0; i < keys.length; i++) out.push(s[map[keys[i]]]);
  return out;
}

export function cleanSeries(s: Series): Series {
  const out: Series = [];
  for (let i = 0; i < s.length; i++) {
    const v = s[i].v;
    if (v == null) continue;
    if (!isFiniteNum(v)) continue;
    const t = toDateOnlyISO(s[i].t);
    if (!t) continue;
    out.push({ t, v });
  }
  return sortSeries(dedupeKeepLast(out));
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Transformations & math
 * ────────────────────────────────────────────────────────────────────────── */

export function toSeries<T extends Dict>(
  rows: T[],
  timeField: keyof T,
  valueField: keyof T
): Series {
  const out: Series = [];
  for (let i = 0; i < rows.length; i++) {
    const r: any = rows[i];
    const t = toDateOnlyISO(r[timeField]);
    const v = toNumberSafe(r[valueField]);
    if (t && v != null) out.push({ t, v });
  }
  return cleanSeries(out);
}

export function toNumberSafe(v: any): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number") return isFinite(v) ? v : null;
  const n = Number(String(v).replace(/[_ ,]/g, ""));
  return isFinite(n) ? n : null;
}

export function pctChange(s: Series, periods = 1): Series {
  const a = cleanSeries(s);
  const out: Series = [];
  for (let i = 0; i < a.length; i++) {
    const j = i - periods;
    if (j < 0) continue;
    const prev = a[j].v as number;
    const cur = a[i].v as number;
    if (prev === 0) continue;
    out.push({ t: a[i].t, v: (cur - prev) / prev });
  }
  return out;
}

export function logReturnsFromPrice(s: Series, periods = 1): Series {
  const a = cleanSeries(s);
  const out: Series = [];
  for (let i = periods; i < a.length; i++) {
    const p0 = a[i - periods].v as number;
    const p1 = a[i].v as number;
    if (p0 <= 0 || p1 <= 0) continue;
    out.push({ t: a[i].t, v: Math.log(p1 / p0) });
  }
  return out;
}

export function simpleReturnsFromPrice(s: Series, periods = 1): Series {
  return pctChange(s, periods);
}

export function cumulativeFromReturns(returns: Series, kind: "simple" | "log" = "simple"): Series {
  const a = cleanSeries(returns);
  const out: Series = [];
  if (kind === "simple") {
    let growth = 1;
    for (let i = 0; i < a.length; i++) {
      const r = a[i].v as number;
      growth *= (1 + r);
      out.push({ t: a[i].t, v: growth });
    }
  } else {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i].v as number);
      out.push({ t: a[i].t, v: Math.exp(sum) });
    }
  }
  return out;
}

export function rolling(
  s: Series,
  window: number,
  reducer: (values: number[]) => number
): Series {
  const a = cleanSeries(s);
  const out: Series = [];
  const buf: number[] = [];
  for (let i = 0; i < a.length; i++) {
    buf.push(a[i].v as number);
    if (buf.length > window) buf.shift();
    if (buf.length === window) {
      out.push({ t: a[i].t, v: reducer(buf) });
    }
  }
  return out;
}

export function mean(xs: number[]): number {
  if (!xs.length) return NaN;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

export function std(xs: number[]): number {
  if (xs.length < 2) return NaN;
  const m = mean(xs);
  let v = 0;
  for (let i = 0; i < xs.length; i++) {
    const d = xs[i] - m;
    v += d * d;
  }
  return Math.sqrt(v / (xs.length - 1));
}

export function sum(xs: number[]): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s;
}

export function quantile(xs: number[], q: number): number {
  if (!xs.length) return NaN;
  const a = xs.slice().sort((x, y) => x - y);
  const p = clamp(q, 0, 1) * (a.length - 1);
  const i = Math.floor(p);
  const f = p - i;
  return i + 1 < a.length ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}

export function summarize(returns: Series): StatsSummary {
  const a = cleanSeries(returns).map(p => p.v as number);
  return {
    n: a.length,
    mean: mean(a),
    std: std(a),
    min: Math.min.apply(null as any, a),
    max: Math.max.apply(null as any, a),
    sum: sum(a),
    p50: quantile(a, 0.5),
    p95: quantile(a, 0.95),
    p99: quantile(a, 0.99)
  };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Risk metrics
 * ────────────────────────────────────────────────────────────────────────── */

export function annualizedVol(returns: Series, periodsPerYear: Annualization = 252): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  return std(a) * Math.sqrt(periodsPerYear);
}

export function annualizedReturn(returns: Series, periodsPerYear: Annualization = 252, kind: "simple" | "log" = "simple"): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  if (!a.length) return NaN;
  if (kind === "simple") {
    const g = a.reduce((acc, r) => acc * (1 + r), 1);
    const yrs = a.length / periodsPerYear;
    return Math.pow(g, 1 / yrs) - 1;
  } else {
    const mu = mean(a);
    return Math.expm1(mu * periodsPerYear);
  }
}

export function sharpe(
  returns: Series,
  rfPerPeriod = 0,
  periodsPerYear: Annualization = 252
): number {
  const a = cleanSeries(returns).map(p => (p.v as number) - rfPerPeriod);
  if (a.length < 2) return NaN;
  const mu = mean(a);
  const sd = std(a);
  if (sd === 0) return NaN;
  return (mu / sd) * Math.sqrt(periodsPerYear);
}

export function betaAlphaVsBenchmark(
  assetRets: Series,
  benchRets: Series,
  rfPerPeriod = 0,
  periodsPerYear: Annualization = 252
): BetaAlpha {
  const joined = alignSeries([assetRets, benchRets], "inner");
  const a: number[] = [];
  const b: number[] = [];
  for (let i = 0; i < joined.length; i++) {
    const row = joined[i];
    const ra = (row[1] as number) - rfPerPeriod;
    const rb = (row[2] as number) - rfPerPeriod;
    if (isFiniteNum(ra) && isFiniteNum(rb)) {
      a.push(ra); b.push(rb);
    }
  }
  const n = Math.min(a.length, b.length);
  if (n < 2) return { beta: NaN, alpha_annual: NaN, r2: NaN, cov: NaN, var_bm: NaN, corr: NaN };
  const ma = mean(a), mb = mean(b);
  let cov = 0, vab = 0, vbb = 0, r2num = 0, r2den = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    cov += da * db;
    vab += da * da;
    vbb += db * db;
    r2num += (mb * db) * (ma * da); // not standard; we'll compute r^2 via corr^2 below
  }
  cov /= (n - 1);
  const var_bm = vbb / (n - 1);
  const beta = var_bm === 0 ? NaN : (cov / var_bm);
  const alpha_per_period = ma - beta * mb;
  const alpha_annual = alpha_per_period * periodsPerYear;
  const corr = (Math.sqrt(var_bm) === 0 || Math.sqrt(vab / (n - 1)) === 0)
    ? NaN
    : (cov / (Math.sqrt(var_bm) * Math.sqrt(vab / (n - 1))));
  const r2 = isFiniteNum(corr) ? corr * corr : NaN;
  return { beta, alpha_annual, r2, cov, var_bm, corr };
}

export function maxDrawdown(equityCurve: Series): { mdd: number; troughDate?: string; peakDate?: string; path: DrawdownPoint[] } {
  const a = cleanSeries(equityCurve);
  let peak = -Infinity;
  let peakDate = "";
  let mdd = 0;
  let troughDate = "";
  const path: DrawdownPoint[] = [];
  for (let i = 0; i < a.length; i++) {
    const eq = a[i].v as number;
    if (eq > peak) { peak = eq; peakDate = a[i].t; }
    const dd = peak > 0 ? (eq / peak - 1) : 0;
    if (dd < mdd) { mdd = dd; troughDate = a[i].t; }
    path.push({ t: a[i].t, equity: eq, peak, dd });
  }
  return { mdd, troughDate, peakDate, path };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Joining & alignment
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Align multiple series by date.
 * Returns rows like: [t, v0, v1, v2, ...]
 */
export function alignSeries(seriesList: Series[], method: JoinMethod = "inner"): Array<any[]> {
  if (!seriesList.length) return [];
  const clean = seriesList.map(s => cleanSeries(s));
  if (method === "left") {
    const base = clean[0];
    const maps = clean.map(s => mapByDate(s));
    const out: any[] = [];
    for (let i = 0; i < base.length; i++) {
      const t = base[i].t;
      const row: any[] = [t];
      let ok = true;
      for (let j = 0; j < maps.length; j++) {
        const mv = maps[j][t];
        row.push(mv == null ? null : mv);
      }
      if (ok) out.push(row);
    }
    return out;
  } else {
    // inner: intersection of all dates
    let dates = setOfDates(clean[0]);
    for (let i = 1; i < clean.length; i++) {
      dates = intersectSets(dates, setOfDates(clean[i]));
    }
    const keys = Object.keys(dates).sort();
    const maps = clean.map(s => mapByDate(s));
    const out: any[] = [];
    for (let k = 0; k < keys.length; k++) {
      const t = keys[k];
      const row: any[] = [t];
      for (let j = 0; j < maps.length; j++) {
        row.push(maps[j][t]);
      }
      out.push(row);
    }
    return out;
  }
}

function mapByDate(s: Series): Dict<number> {
  const m: Dict<number> = {};
  for (let i = 0; i < s.length; i++) m[s[i].t] = s[i].v as number;
  return m;
}
function setOfDates(s: Series): Dict<boolean> {
  const d: Dict<boolean> = {};
  for (let i = 0; i < s.length; i++) d[s[i].t] = true;
  return d;
}
function intersectSets(a: Dict<boolean>, b: Dict<boolean>): Dict<boolean> {
  const out: Dict<boolean> = {};
  for (const k in a) if (b[k]) out[k] = true;
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Portfolio math (static weights)
 * ────────────────────────────────────────────────────────────────────────── */

export function normalizeWeights(w: PortfolioWeights): PortfolioWeights {
  let s = 0;
  for (const k in w) s += (w[k] || 0);
  if (s === 0) return {};
  const out: PortfolioWeights = {};
  for (const k in w) out[k] = (w[k] || 0) / s;
  return out;
}

/**
 * Combine individual asset return series into a portfolio return series
 * using static weights (sum to 1; will auto-normalize).
 */
export function portfolioReturns(weighted: { [symbol: string]: Series }, weights: PortfolioWeights): Series {
  const keys = Object.keys(weighted);
  if (!keys.length) return [];
  const W = normalizeWeights(weights);
  const seriesList = keys.map(k => weighted[k]);
  const aligned = alignSeries(seriesList, "inner"); // [t, v1, v2, ...]
  const out: Series = [];
  for (let i = 0; i < aligned.length; i++) {
    const row = aligned[i];
    let r = 0;
    for (let j = 0; j < keys.length; j++) {
      const sym = keys[j];
      const w = W[sym] || 0;
      const rv = row[j + 1] as number;
      r += w * rv;
    }
    out.push({ t: row[0] as string, v: r });
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Convenience OHLC helpers
 * ────────────────────────────────────────────────────────────────────────── */

export type OHLC = { t: string; o?: number; h?: number; l?: number; c?: number };

export function closeSeriesFromOHLC(rows: OHLC[]): Series {
  const out: Series = [];
  for (let i = 0; i < rows.length; i++) {
    const t = toDateOnlyISO(rows[i].t);
    const c = toNumberSafe(rows[i].c);
    if (t && c != null) out.push({ t, v: c });
  }
  return cleanSeries(out);
}

export function simpleReturnsFromOHLC(rows: OHLC[], periods = 1): Series {
  return simpleReturnsFromPrice(closeSeriesFromOHLC(rows), periods);
}

export function logReturnsFromOHLC(rows: OHLC[], periods = 1): Series {
  return logReturnsFromPrice(closeSeriesFromOHLC(rows), periods);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Misc
 * ────────────────────────────────────────────────────────────────────────── */

export function stableStringify(v: any): string {
  try { return JSON.stringify(sortKeysDeep(v)); } catch { return String(v); }
}
function sortKeysDeep(v: any): any {
  if (Array.isArray(v)) return v.map(sortKeysDeep);
  if (v && typeof v === "object") {
    const out: Dict = {};
    const ks = Object.keys(v).sort();
    for (let i = 0; i < ks.length; i++) out[ks[i]] = sortKeysDeep((v as any)[ks[i]]);
    return out;
  }
  return v;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Example usage (commented)
 *
 * // Build price series and compute returns:
 * // const px = toSeries(priceRows, "dt", "px_close");
 * // const rets = simpleReturnsFromPrice(px);
 * // const annRet = annualizedReturn(rets);
 * // const annVol = annualizedVol(rets);
 * // const sr = sharpe(rets, 0, 252);
 *
 * // Portfolio:
 * // const rAAPL = simpleReturnsFromPrice(toSeries(aapl, "dt","px_close"));
 * // const rMSFT = simpleReturnsFromPrice(toSeries(msft, "dt","px_close"));
 * // const port = portfolioReturns({ AAPL: rAAPL, MSFT: rMSFT }, { AAPL: 0.6, MSFT: 0.4 });
 * // const eq = cumulativeFromReturns(port, "simple"); // equity curve
 * // const { mdd } = maxDrawdown(eq);
 * ────────────────────────────────────────────────────────────────────────── */