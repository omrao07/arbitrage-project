// utils/math.ts
// Pure, dependency-free math helpers for browser/Node (no BigInt, no imports).
// Designed to be safe with NaNs, work with TS targets < ES2020, and easy to tree-shake.

export const EPS = 1e-12;

/* =============================== Predicates =============================== */

export function isNum(x: any): x is number {
  return typeof x === "number" && isFinite(x);
}

export function almostEqual(a: number, b: number, eps: number = EPS): boolean {
  return Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b));
}

/* ============================== Basic helpers ============================= */

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
export function clamp01(x: number): number { return clamp(x, 0, 1); }

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
export function invLerp(a: number, b: number, x: number): number {
  return (x - a) / ((b - a) || EPS);
}
export function mapRange(
  x: number,
  inMin: number, inMax: number,
  outMin: number, outMax: number,
  clampOut?: boolean
): number {
  const t = invLerp(inMin, inMax, x);
  const u = clampOut ? clamp01(t) : t;
  return lerp(outMin, outMax, u);
}

export function roundTo(x: number, dp: number = 0): number {
  const m = Math.pow(10, dp | 0);
  return Math.round((x + 0) * m) / m;
}
export function floorTo(x: number, dp: number = 0): number {
  const m = Math.pow(10, dp | 0);
  return Math.floor((x + 0) * m) / m;
}
export function ceilTo(x: number, dp: number = 0): number {
  const m = Math.pow(10, dp | 0);
  return Math.ceil((x + 0) * m) / m;
}

/* ============================== Array helpers ============================= */

export function sum(a: number[], ignoreNaN: boolean = true): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (ignoreNaN ? isNum(v) : true) s += v || 0;
  }
  return s;
}
export function mean(a: number[], ignoreNaN: boolean = true): number {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (ignoreNaN ? isNum(v) : true) { s += v || 0; n++; }
  }
  return n ? s / n : NaN;
}
export function median(a: number[]): number {
  const v: number[] = [];
  for (let i = 0; i < a.length; i++) if (isNum(a[i])) v.push(a[i]);
  if (!v.length) return NaN;
  v.sort(function (x, y) { return x - y; });
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}
export function quantile(a: number[], q: number): number {
  const v: number[] = [];
  for (let i = 0; i < a.length; i++) if (isNum(a[i])) v.push(a[i]);
  if (!v.length) return NaN;
  v.sort(function (x, y) { return x - y; });
  const p = clamp01(q);
  const idx = (v.length - 1) * p;
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return v[lo];
  const t = idx - lo;
  return v[lo] * (1 - t) + v[hi] * t;
}
export function minmax(a: number[]): { min: number; max: number } {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!isNum(v)) continue;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return { min: mn, max: mx };
}
export function argmin(a: number[]): number {
  let idx = -1, best = Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!isNum(v)) continue;
    if (v < best) { best = v; idx = i; }
  }
  return idx;
}
export function argmax(a: number[]): number {
  let idx = -1, best = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!isNum(v)) continue;
    if (v > best) { best = v; idx = i; }
  }
  return idx;
}

export function variance(a: number[], sample: boolean = true): number {
  let n = 0, m = 0, M2 = 0; // Welford
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    if (!isNum(x)) continue;
    n++;
    const delta = x - m;
    m += delta / n;
    M2 += delta * (x - m);
  }
  if (n < 2) return NaN;
  return M2 / (sample ? (n - 1) : n);
}
export function stdev(a: number[], sample: boolean = true): number {
  const v = variance(a, sample);
  return isNum(v) ? Math.sqrt(v) : NaN;
}

export function zScores(a: number[]): number[] {
  const m = mean(a), s = stdev(a);
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    out[i] = isNum(v) && isNum(s) && s > 0 ? (v - m) / s : NaN;
  }
  return out;
}

export function normalizeMinMax(a: number[], lo: number = 0, hi: number = 1): number[] {
  const mm = minmax(a);
  const d = mm.max - mm.min || EPS;
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    out[i] = isNum(v) ? lerp(lo, hi, (v - mm.min) / d) : NaN;
  }
  return out;
}

/* ============================== Rolling utils ============================= */

export function cumsum(a: number[]): number[] {
  const out = new Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) { s += a[i] || 0; out[i] = s; }
  return out;
}

export function diff(a: number[], lag: number = 1): number[] {
  const n = a.length, L = Math.max(1, lag | 0);
  const out = new Array(n).fill(NaN);
  for (let i = L; i < n; i++) out[i] = (a[i] || 0) - (a[i - L] || 0);
  return out;
}

export function pctChange(a: number[], lag: number = 1): number[] {
  const n = a.length, L = Math.max(1, lag | 0);
  const out = new Array(n).fill(NaN);
  for (let i = L; i < n; i++) {
    const prev = a[i - L];
    out[i] = isNum(prev) && prev !== 0 ? (a[i] - prev) / prev : NaN;
  }
  return out;
}

export function sma(a: number[], window: number): number[] {
  const n = a.length, w = Math.max(1, window | 0);
  const out = new Array(n).fill(NaN);
  let s = 0, q: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = a[i] || 0;
    q.push(v); s += v;
    if (q.length > w) s -= q.shift() as number;
    if (q.length === w) out[i] = s / w;
  }
  return out;
}

export function ema(a: number[], window: number): number[] {
  const n = a.length, w = Math.max(1, window | 0);
  const out = new Array(n).fill(NaN);
  const alpha = 2 / (w + 1);
  let prev: number | null = null;
  for (let i = 0; i < n; i++) {
    const v = a[i];
    if (!isNum(v)) { out[i] = prev == null ? NaN : prev; continue; }
    prev = prev == null ? v : alpha * v + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

export function wma(a: number[], window: number): number[] {
  const n = a.length, w = Math.max(1, window | 0);
  const out = new Array(n).fill(NaN);
  const weights: number[] = [];
  for (let i = 1; i <= w; i++) weights.push(i);
  const W = (w * (w + 1)) / 2;
  const buf: number[] = [];
  for (let i = 0; i < n; i++) {
    buf.push(a[i] || 0);
    if (buf.length > w) buf.shift();
    if (buf.length === w) {
      let s = 0;
      for (let k = 0; k < w; k++) s += buf[k] * weights[k];
      out[i] = s / W;
    }
  }
  return out;
}

export function rollingApply<T = number>(
  a: T[],
  window: number,
  fn: (slice: T[], idxEnd: number) => number
): number[] {
  const n = a.length, w = Math.max(1, window | 0);
  const out = new Array(n).fill(NaN);
  const buf: T[] = [];
  for (let i = 0; i < n; i++) {
    buf.push(a[i]);
    if (buf.length > w) buf.shift();
    if (buf.length === w) out[i] = fn(buf.slice(), i);
  }
  return out;
}

/* ============================== Linear Algebra ============================== */

export function dot(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) s += (a[i] || 0) * (b[i] || 0);
  return s;
}
export function norm(a: number[], p: number = 2): number {
  if (p === Infinity) {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] || 0));
    return m;
  }
  if (p === 1) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] || 0);
    return s;
  }
  let s2 = 0;
  for (let i = 0; i < a.length; i++) { const v = a[i] || 0; s2 += v * v; }
  return Math.sqrt(s2);
}
export function distance(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let s2 = 0;
  for (let i = 0; i < n; i++) { const d = (a[i] || 0) - (b[i] || 0); s2 += d * d; }
  return Math.sqrt(s2);
}

/* ============================ Stats & Finance ============================ */

export function cov(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  let mx = 0, my = 0, k = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i], b = y[i];
    if (!isNum(a) || !isNum(b)) continue;
    k++; mx += a; my += b;
  }
  if (k < 2) return NaN;
  mx /= k; my /= k;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i], b = y[i];
    if (!isNum(a) || !isNum(b)) continue;
    s += (a - mx) * (b - my);
  }
  return s / (k - 1);
}
export function corr(x: number[], y: number[]): number {
  const c = cov(x, y);
  const sx = stdev(x), sy = stdev(y);
  return isNum(c) && isNum(sx) && isNum(sy) && sx > 0 && sy > 0 ? c / (sx * sy) : NaN;
}

export type LinReg = { alpha: number; beta: number; r2: number; n: number };
export function linreg(y: number[], x: number[]): LinReg {
  const n = Math.min(y.length, x.length);
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, k = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i], yi = y[i];
    if (!isNum(xi) || !isNum(yi)) continue;
    k++;
    sx += xi; sy += yi;
    sxx += xi * xi; syy += yi * yi; sxy += xi * yi;
  }
  if (k < 2) return { alpha: NaN, beta: NaN, r2: NaN, n: 0 };
  const beta = (sxy - (sx * sy) / k) / ((sxx - (sx * sx) / k) || EPS);
  const alpha = (sy - beta * sx) / k;
  // r^2
  const ybar = sy / k;
  let sse = 0, sst = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i], yi = y[i];
    if (!isNum(xi) || !isNum(yi)) continue;
    const e = yi - (alpha + beta * xi);
    sse += e * e;
    const d = yi - ybar;
    sst += d * d;
  }
  const r2 = sst > 0 ? Math.max(0, 1 - sse / sst) : 0;
  return { alpha, beta, r2, n: k };
}

export function softmax(a: number[], t: number = 1): number[] {
  const temp = t > 0 ? t : EPS;
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (isNum(a[i]) && a[i] > m) m = a[i];
  const exps: number[] = new Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const v = (a[i] - m) / temp;
    const ev = Math.exp(isNum(v) ? v : -Infinity);
    exps[i] = ev; s += ev;
  }
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = s > 0 ? exps[i] / s : NaN;
  return out;
}

/* --------- Returns / Risk metrics (daily by default; configurable) -------- */

export function equityFromReturns(returns: number[], start: number = 1): number[] {
  const out = new Array(returns.length);
  let eq = start;
  for (let i = 0; i < returns.length; i++) {
    eq = eq * (1 + (returns[i] || 0));
    out[i] = eq;
  }
  return out;
}

export function maxDrawdown(curve: number[]): number {
  let peak = curve[0] || 1, maxDD = 0;
  for (let i = 1; i < curve.length; i++) {
    const v = curve[i];
    if (!isNum(v)) continue;
    if (v > peak) peak = v;
    const dd = 1 - v / (peak || EPS);
    if (dd > maxDD) maxDD = dd;
  }
  return maxDD;
}

export function sharpe(returns: number[], periodsPerYear: number = 252, rf: number = 0): number {
  const adj = returns.slice(0);
  if (rf) for (let i = 0; i < adj.length; i++) adj[i] = (adj[i] || 0) - rf / periodsPerYear;
  const m = mean(adj), s = stdev(adj);
  return isNum(s) && s > 0 ? (m * periodsPerYear) / (s * Math.sqrt(periodsPerYear)) : 0;
}

export function cagrFromEquity(equity: number[], periodsPerYear: number = 252): number {
  if (!equity.length) return 0;
  const start = equity[0] || 1, end = equity[equity.length - 1] || start;
  const years = (equity.length - 1) / (periodsPerYear || 1);
  return years > 0 ? Math.pow((end || EPS) / (start || EPS), 1 / years) - 1 : 0;
}

/* ============================== Random utils ============================= */

export function randomInt(lo: number, hi: number): number {
  const a = Math.ceil(Math.min(lo, hi));
  const b = Math.floor(Math.max(lo, hi));
  return Math.floor(a + Math.random() * (b - a + 1));
}
export function randomChoice<T>(arr: T[]): T | undefined {
  if (!arr.length) return undefined;
  return arr[Math.floor(Math.random() * arr.length)];
}
export function randomNormal(mu: number = 0, sigma: number = 1): number {
  // Box–Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return mu + sigma * z;
}

/* ============================== Search / misc ============================= */

export function binarySearchAsc(arr: number[], x: number): number {
  // returns index of first element >= x (insertion point)
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

export function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }
export function tanh(x: number): number {
  const e1 = Math.exp(x), e2 = Math.exp(-x);
  return (e1 - e2) / (e1 + e2);
}
export function relu(x: number): number { return x > 0 ? x : 0; }

/* =============================== Default export =============================== */

const math = {
  EPS, isNum, almostEqual,
  clamp, clamp01, lerp, invLerp, mapRange, roundTo, floorTo, ceilTo,
  sum, mean, median, quantile, minmax, argmin, argmax, variance, stdev,
  zScores, normalizeMinMax, cumsum, diff, pctChange,
  sma, ema, wma, rollingApply,
  dot, norm, distance, cov, corr, linreg, softmax,
  equityFromReturns, maxDrawdown, sharpe, cagrFromEquity,
  randomInt, randomChoice, randomNormal,
  binarySearchAsc, sigmoid, tanh, relu,
};

export default math;