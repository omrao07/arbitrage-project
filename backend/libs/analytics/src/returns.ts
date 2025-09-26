// analytics/src/returns.ts
// Zero-dependency utilities focused on returns math, risk metrics, and rolling stats.
// Pure TypeScript (no imports). Safe for Node or browser.

export type Dict<T = any> = { [k: string]: T };
export type Point = { t: string; v: number | null | undefined };
export type Series = Point[];
export type JoinMethod = "inner" | "left";
export type Annualization = 252 | 12 | 52 | 365;

/* ────────────────────────────────────────────────────────────────────────── *
 * Mini helpers
 * ────────────────────────────────────────────────────────────────────────── */

function isFiniteNum(x: any): x is number { return typeof x === "number" && isFinite(x); }
function clamp(n: number, lo: number, hi: number): number { return n < lo ? lo : n > hi ? hi : n; }
function pad2(n: number): string { return String(n).padStart(2, "0"); }

function toDateOnlyISO(v: any): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "string") {
    const s = v.trim();
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[1]}-${m[2]}-${m[3]}`;
    const ts = Date.parse(s);
    if (!isNaN(ts)) {
      const d = new Date(ts);
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
  if (typeof v === "number" && isFinite(v)) return v;
  const n = Number(String(v).replace(/[_ ,]/g, ""));
  return isFinite(n) ? n : null;
}

function sortSeries(s: Series): Series {
  const a = s.slice(0);
  a.sort((x, y) => (x.t < y.t ? -1 : x.t > y.t ? 1 : 0));
  return a;
}

function dedupeKeepLast(s: Series): Series {
  const idx: Dict<number> = {};
  for (let i = 0; i < s.length; i++) idx[s[i].t] = i;
  const out: Series = [];
  const keys = Object.keys(idx).sort();
  for (let i = 0; i < keys.length; i++) out.push(s[idx[keys[i]]]);
  return out;
}

export function cleanSeries(s: Series): Series {
  const out: Series = [];
  for (let i = 0; i < s.length; i++) {
    const t = toDateOnlyISO(s[i].t);
    const v = toNumberSafe(s[i].v);
    if (!t || v == null || !isFiniteNum(v)) continue;
    out.push({ t, v });
  }
  return sortSeries(dedupeKeepLast(out));
}

export function toSeries<T extends Dict>(rows: T[], timeField: keyof T, valueField: keyof T): Series {
  const out: Series = [];
  for (let i = 0; i < rows.length; i++) {
    const r: any = rows[i];
    const t = toDateOnlyISO(r[timeField]);
    const v = toNumberSafe(r[valueField]);
    if (t && v != null) out.push({ t, v });
  }
  return cleanSeries(out);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Align & join
 * ────────────────────────────────────────────────────────────────────────── */

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
      for (let j = 0; j < maps.length; j++) row.push(maps[j][t] ?? null);
      out.push(row);
    }
    return out;
  } else {
    let dates = setOfDates(clean[0]);
    for (let i = 1; i < clean.length; i++) dates = intersectSets(dates, setOfDates(clean[i]));
    const keys = Object.keys(dates).sort();
    const maps = clean.map(s => mapByDate(s));
    const out: any[] = [];
    for (let k = 0; k < keys.length; k++) {
      const t = keys[k];
      const row: any[] = [t];
      for (let j = 0; j < maps.length; j++) row.push(maps[j][t]);
      out.push(row);
    }
    return out;
  }
}
function mapByDate(s: Series): Dict<number> { const m: Dict<number> = {}; for (let i = 0; i < s.length; i++) m[s[i].t] = s[i].v as number; return m; }
function setOfDates(s: Series): Dict<boolean> { const d: Dict<boolean> = {}; for (let i = 0; i < s.length; i++) d[s[i].t] = true; return d; }
function intersectSets(a: Dict<boolean>, b: Dict<boolean>): Dict<boolean> { const out: Dict<boolean> = {}; for (const k in a) if (b[k]) out[k] = true; return out; }

/* ────────────────────────────────────────────────────────────────────────── *
 * Returns construction & compounding
 * ────────────────────────────────────────────────────────────────────────── */

export function simpleReturnsFromPrice(px: Series, periods = 1): Series {
  const a = cleanSeries(px);
  const out: Series = [];
  for (let i = periods; i < a.length; i++) {
    const p0 = a[i - periods].v as number;
    const p1 = a[i].v as number;
    if (p0 === 0) continue;
    out.push({ t: a[i].t, v: (p1 - p0) / p0 });
  }
  return out;
}

export function logReturnsFromPrice(px: Series, periods = 1): Series {
  const a = cleanSeries(px);
  const out: Series = [];
  for (let i = periods; i < a.length; i++) {
    const p0 = a[i - periods].v as number;
    const p1 = a[i].v as number;
    if (p0 <= 0 || p1 <= 0) continue;
    out.push({ t: a[i].t, v: Math.log(p1 / p0) });
  }
  return out;
}

/** Build equity curve from returns. `kind="simple"` compounds product(1+r); `"log"` sums logs. */
export function equityFromReturns(rets: Series, kind: "simple" | "log" = "simple", startEquity = 1): Series {
  const a = cleanSeries(rets);
  const out: Series = [];
  if (kind === "simple") {
    let eq = startEquity;
    for (let i = 0; i < a.length; i++) { eq *= (1 + (a[i].v as number)); out.push({ t: a[i].t, v: eq }); }
  } else {
    let s = Math.log(startEquity);
    for (let i = 0; i < a.length; i++) { s += (a[i].v as number); out.push({ t: a[i].t, v: Math.exp(s) }); }
  }
  return out;
}

/** Cumulative growth index from returns (alias for equityFromReturns with start=1). */
export function cumulativeFromReturns(rets: Series, kind: "simple" | "log" = "simple"): Series {
  return equityFromReturns(rets, kind, 1);
}

/** Convert returns to excess returns vs scalar rfPerPeriod or RF time series. */
export function excessReturns(returns: Series, rf: number | Series = 0): Series {
  const a = cleanSeries(returns);
  if (typeof rf === "number") {
    if (rf === 0) return a;
    const out: Series = [];
    for (let i = 0; i < a.length; i++) out.push({ t: a[i].t, v: (a[i].v as number) - rf });
    return out;
  } else {
    const rfS = cleanSeries(rf);
    const joined = alignSeries([a, rfS], "inner"); // [t, r, rf]
    const out: Series = [];
    for (let i = 0; i < joined.length; i++) out.push({ t: joined[i][0], v: (joined[i][1] as number) - (joined[i][2] as number) });
    return out;
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Moments & summaries
 * ────────────────────────────────────────────────────────────────────────── */

export function mean(xs: number[]): number { if (!xs.length) return NaN; let s = 0; for (let i = 0; i < xs.length; i++) s += xs[i]; return s / xs.length; }

export function std(xs: number[]): number {
  if (xs.length < 2) return NaN;
  const m = mean(xs);
  let v = 0;
  for (let i = 0; i < xs.length; i++) { const d = xs[i] - m; v += d * d; }
  return Math.sqrt(v / (xs.length - 1));
}

export function skewness(xs: number[]): number {
  if (xs.length < 3) return NaN;
  const m = mean(xs), s = std(xs);
  if (s === 0) return NaN;
  let num = 0;
  for (let i = 0; i < xs.length; i++) { const z = (xs[i] - m) / s; num += Math.pow(z, 3); }
  return num / xs.length;
}

export function kurtosis(xs: number[]): number {
  if (xs.length < 4) return NaN;
  const m = mean(xs), s = std(xs);
  if (s === 0) return NaN;
  let num = 0;
  for (let i = 0; i < xs.length; i++) { const z = (xs[i] - m) / s; num += Math.pow(z, 4); }
  return num / xs.length - 3; // excess kurtosis
}

export function quantile(xs: number[], q: number): number {
  if (!xs.length) return NaN;
  const a = xs.slice().sort((x, y) => x - y);
  const p = clamp(q, 0, 1) * (a.length - 1);
  const i = Math.floor(p), f = p - i;
  return i + 1 < a.length ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}

export function summarize(returns: Series): {
  n: number; mean: number; std: number; min: number; max: number; sum: number; p50: number; p95: number; p99: number; skew: number; kurt: number;
} {
  const a = cleanSeries(returns).map(p => p.v as number);
  return {
    n: a.length,
    mean: mean(a),
    std: std(a),
    min: Math.min.apply(null as any, a),
    max: Math.max.apply(null as any, a),
    sum: a.reduce((s, x) => s + x, 0),
    p50: quantile(a, 0.5),
    p95: quantile(a, 0.95),
    p99: quantile(a, 0.99),
    skew: skewness(a),
    kurt: kurtosis(a)
  };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Annualization & ratios
 * ────────────────────────────────────────────────────────────────────────── */

export function annualizedVol(returns: Series, periodsPerYear: Annualization = 252): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  return std(a) * Math.sqrt(periodsPerYear);
}

export function annualizedReturn(
  returns: Series,
  periodsPerYear: Annualization = 252,
  kind: "simple" | "log" = "simple"
): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  if (!a.length) return NaN;
  if (kind === "simple") {
    const growth = a.reduce((g, r) => g * (1 + r), 1);
    const yrs = a.length / periodsPerYear;
    return Math.pow(growth, 1 / yrs) - 1;
  } else {
    const mu = mean(a);
    return Math.expm1(mu * periodsPerYear);
  }
}

export function sharpe(returns: Series, rfPerPeriod = 0, periodsPerYear: Annualization = 252): number {
  const a = cleanSeries(returns).map(p => (p.v as number) - rfPerPeriod);
  if (a.length < 2) return NaN;
  const mu = mean(a), sd = std(a);
  if (sd === 0) return NaN;
  return (mu / sd) * Math.sqrt(periodsPerYear);
}

/** Sortino ratio using MAR (minimum acceptable return) per period (default 0). */
export function sortino(returns: Series, marPerPeriod = 0, periodsPerYear: Annualization = 252): number {
  const a = cleanSeries(returns).map(p => (p.v as number) - marPerPeriod);
  if (a.length < 2) return NaN;
  const downside = a.filter(x => x < 0);
  const dd = std(downside.length ? downside : [0]);
  if (dd === 0) return NaN;
  const mu = mean(a);
  return (mu / dd) * Math.sqrt(periodsPerYear);
}

/** Calmar ratio: annualized return / |max drawdown| (on equity from simple returns). */
export function calmar(returns: Series, periodsPerYear: Annualization = 252): number {
  const ann = annualizedReturn(returns, periodsPerYear, "simple");
  const eq = equityFromReturns(returns, "simple");
  const m = maxDrawdown(eq).mdd;
  return m === 0 ? NaN : ann / Math.abs(m);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Benchmark-relative metrics
 * ────────────────────────────────────────────────────────────────────────── */

export type BetaAlpha = { beta: number; alpha_annual: number; r2: number; cov: number; var_bm: number; corr: number };

export function trackingError(assetRets: Series, benchRets: Series, periodsPerYear: Annualization = 252): number {
  const j = alignSeries([assetRets, benchRets], "inner");
  const diff: number[] = [];
  for (let i = 0; i < j.length; i++) diff.push((j[i][1] as number) - (j[i][2] as number));
  return std(diff) * Math.sqrt(periodsPerYear);
}

export function informationRatio(assetRets: Series, benchRets: Series, periodsPerYear: Annualization = 252): number {
  const j = alignSeries([assetRets, benchRets], "inner");
  const diff: number[] = [];
  for (let i = 0; i < j.length; i++) diff.push((j[i][1] as number) - (j[i][2] as number));
  const mu = mean(diff);
  const te = std(diff);
  if (te === 0) return NaN;
  return (mu / te) * Math.sqrt(periodsPerYear);
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
    const ra = (joined[i][1] as number) - rfPerPeriod;
    const rb = (joined[i][2] as number) - rfPerPeriod;
    if (isFiniteNum(ra) && isFiniteNum(rb)) { a.push(ra); b.push(rb); }
  }
  const n = Math.min(a.length, b.length);
  if (n < 2) return { beta: NaN, alpha_annual: NaN, r2: NaN, cov: NaN, var_bm: NaN, corr: NaN };
  const ma = mean(a), mb = mean(b);
  let cov = 0, vab = 0, vbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    cov += da * db; vab += da * da; vbb += db * db;
  }
  cov /= (n - 1);
  const var_bm = vbb / (n - 1);
  const beta = var_bm === 0 ? NaN : (cov / var_bm);
  const alpha_per_period = ma - beta * mb;
  const alpha_annual = alpha_per_period * periodsPerYear;
  const corr = (Math.sqrt(var_bm) === 0 || Math.sqrt(vab / (n - 1)) === 0) ? NaN : (cov / (Math.sqrt(var_bm) * Math.sqrt(vab / (n - 1))));
  const r2 = isFiniteNum(corr) ? corr * corr : NaN;
  return { beta, alpha_annual, r2, cov, var_bm, corr };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Drawdowns
 * ────────────────────────────────────────────────────────────────────────── */

export type DrawdownPoint = { t: string; equity: number; peak: number; dd: number };

export function drawdownPath(equityCurve: Series): DrawdownPoint[] {
  const a = cleanSeries(equityCurve);
  const out: DrawdownPoint[] = [];
  let peak = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const eq = a[i].v as number;
    if (eq > peak) peak = eq;
    const dd = peak > 0 ? (eq / peak - 1) : 0;
    out.push({ t: a[i].t, equity: eq, peak, dd });
  }
  return out;
}

export function maxDrawdown(equityCurve: Series): { mdd: number; troughDate?: string; peakDate?: string; path: DrawdownPoint[] } {
  const path = drawdownPath(equityCurve);
  let mdd = 0, troughDate = "", peakDate = "";
  for (let i = 0; i < path.length; i++) {
    if (path[i].dd < mdd) { mdd = path[i].dd; troughDate = path[i].t; }
  }
  for (let i = 0; i < path.length; i++) {
    if (path[i].equity === path[i].peak && path[i].t <= troughDate) { peakDate = path[i].t; }
  }
  return { mdd, troughDate: troughDate || undefined, peakDate: peakDate || undefined, path };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * VaR / CVaR (Historical & Parametric)
 * ────────────────────────────────────────────────────────────────────────── */

export function varHistorical(returns: Series, p = 0.95): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  if (!a.length) return NaN;
  const q = quantile(a, 1 - p); // left tail (loss)
  return q;
}

export function cvarHistorical(returns: Series, p = 0.95): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  if (!a.length) return NaN;
  const v = varHistorical(returns, p);
  const tail = a.filter(x => x <= v);
  return tail.length ? mean(tail) : v;
}

/** Parametric (normal) VaR: μ + z*σ where z ≈ Φ^{-1}(1-p) */
export function varParametric(returns: Series, p = 0.95): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  if (!a.length) return NaN;
  const mu = mean(a), sd = std(a);
  const z = invNormCdf(1 - p);
  return mu + z * sd;
}

/** Parametric CVaR (normal): μ - σ * φ(z) / (1-p), where z = Φ^{-1}(p) on right tail; we use left-tail conv. */
export function cvarParametric(returns: Series, p = 0.95): number {
  const a = cleanSeries(returns).map(p => p.v as number);
  if (!a.length) return NaN;
  const mu = mean(a), sd = std(a);
  const z = invNormCdf(1 - p);
  const phi = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  return mu + (-sd * phi(z) / (1 - p)); // left-tail expected shortfall
}

// Beasley–Springer/Moro-ish approximation of inverse normal CDF (single precision OK)
function invNormCdf(u: number): number {
  // clamp to (0,1)
  const p = u <= 0 ? Number.EPSILON : u >= 1 ? 1 - Number.EPSILON : u;
  // Abramowitz & Stegun 26.2.23
  const a1 = -39.69683028665376, a2 = 220.9460984245205, a3 = -275.9285104469687;
  const a4 = 138.3577518672690, a5 = -30.66479806614716, a6 = 2.506628277459239;
  const b1 = -54.47609879822406, b2 = 161.5858368580409, b3 = -155.6989798598866;
  const b4 = 66.80131188771972, b5 = -13.28068155288572;
  const c1 = -0.007784894002430293, c2 = -0.3223964580411365, c3 = -2.400758277161838;
  const c4 = -2.549732539343734, c5 = 4.374664141464968, c6 = 2.938163982698783;
  const d1 = 0.007784695709041462, d2 = 0.3224671290700398, d3 = 2.445134137142996;
  const d4 = 3.754408661907416;
  const pl = 0.02425, ph = 1 - pl;
  let q: number, r: number;
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
  } else if (p <= ph) {
    q = p - 0.5; r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Downside risk
 * ────────────────────────────────────────────────────────────────────────── */

export function downsideDeviation(returns: Series, marPerPeriod = 0, periodsPerYear: Annualization = 252): number {
  const a = cleanSeries(returns).map(p => (p.v as number) - marPerPeriod);
  const neg = a.filter(x => x < 0);
  if (!neg.length) return 0;
  const sd = std(neg);
  return sd * Math.sqrt(periodsPerYear);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Rolling metrics
 * ────────────────────────────────────────────────────────────────────────── */

export function rollingApply(src: Series, window: number, reducer: (values: number[]) => number): Series {
  const a = cleanSeries(src);
  const vals = a.map(p => p.v as number);
  const out: Series = [];
  if (window <= 0 || a.length < window) return out;
  for (let i = window - 1; i < a.length; i++) {
    const wnd = vals.slice(i - window + 1, i + 1);
    out.push({ t: a[i].t, v: reducer(wnd) });
  }
  return out;
}

export function rollingVol(returns: Series, window: number, periodsPerYear: Annualization = 252): Series {
  return rollingApply(returns, window, (xs) => std(xs) * Math.sqrt(periodsPerYear));
}

export function rollingSharpe(returns: Series, window: number, rfPerPeriod = 0, periodsPerYear: Annualization = 252): Series {
  const adj = subtractScalar(returns, rfPerPeriod);
  return rollingApply(adj, window, (xs) => {
    const mu = mean(xs), sd = std(xs);
    return sd === 0 ? NaN : (mu / sd) * Math.sqrt(periodsPerYear);
  });
}

export function rollingReturn(returns: Series, window: number, kind: "simple" | "log" = "simple", periodsPerYear?: Annualization): Series {
  const a = cleanSeries(returns);
  const out: Series = [];
  if (window <= 0 || a.length < window) return out;
  if (kind === "simple") {
    for (let i = window - 1; i < a.length; i++) {
      let g = 1;
      for (let k = i - window + 1; k <= i; k++) g *= (1 + (a[k].v as number));
      const r = g - 1;
      out.push({ t: a[i].t, v: periodsPerYear ? Math.pow(1 + r, periodsPerYear / window) - 1 : r });
    }
  } else {
    for (let i = window - 1; i < a.length; i++) {
      let s = 0;
      for (let k = i - window + 1; k <= i; k++) s += (a[k].v as number);
      const r = Math.expm1(s);
      out.push({ t: a[i].t, v: periodsPerYear ? Math.expm1(s * (periodsPerYear / window)) : r });
    }
  }
  return out;
}

export function rollingMaxDrawdown(returns: Series, window: number): Series {
  const eq = equityFromReturns(returns, "simple", 1);
  // Compute rolling MDD on equity using a sliding window
  const a = cleanSeries(eq);
  const out: Series = [];
  if (window <= 1 || a.length < window) return out;
  for (let i = window - 1; i < a.length; i++) {
    let peak = -Infinity;
    let mdd = 0;
    for (let k = i - window + 1; k <= i; k++) {
      const v = a[k].v as number;
      if (v > peak) peak = v;
      const dd = peak > 0 ? (v / peak - 1) : 0;
      if (dd < mdd) mdd = dd;
    }
    out.push({ t: a[i].t, v: mdd });
  }
  return out;
}

function subtractScalar(s: Series, x: number): Series {
  if (x === 0) return cleanSeries(s);
  const a = cleanSeries(s);
  const out: Series = [];
  for (let i = 0; i < a.length; i++) out.push({ t: a[i].t, v: (a[i].v as number) - x });
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */