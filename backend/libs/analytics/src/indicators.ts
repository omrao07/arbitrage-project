// analytics/src/indicators.ts
// Pure TypeScript technical indicators (no imports).
// Works with simple { t: "YYYY-MM-DD", v: number } Series,
// plus a lightweight OHLC(V) shape where needed.

export type Dict<T = any> = { [k: string]: T };

export type Point = { t: string; v: number | null | undefined };
export type Series = Point[];

export type OHLC = { t: string; o?: number | null; h?: number | null; l?: number | null; c?: number | null };
export type OHLCV = OHLC & { v?: number | null };

// ────────────────────────────────────────────────────────────────────────────
// Small helpers (duplicated here to avoid imports)
// ────────────────────────────────────────────────────────────────────────────

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

function sortSeries(s: Series): Series {
  const a = s.slice(0);
  a.sort((x, y) => (x.t < y.t ? -1 : x.t > y.t ? 1 : 0));
  return a;
}

function dedupeKeepLast(s: Series): Series {
  const map: Dict<number> = {};
  for (let i = 0; i < s.length; i++) map[s[i].t] = i;
  const out: Series = [];
  const keys = Object.keys(map).sort();
  for (let i = 0; i < keys.length; i++) out.push(s[map[keys[i]]]);
  return out;
}

function cleanSeries(s: Series): Series {
  const out: Series = [];
  for (let i = 0; i < s.length; i++) {
    const v = s[i].v;
    if (!isFiniteNum(v)) continue;
    const t = toDateOnlyISO(s[i].t);
    if (!t) continue;
    out.push({ t, v });
  }
  return sortSeries(dedupeKeepLast(out));
}

function rollingWindow(values: number[], window: number): number[][] {
  const out: number[][] = [];
  if (window <= 0) return out;
  for (let i = 0; i + window <= values.length; i++) {
    out.push(values.slice(i, i + window));
  }
  return out;
}

function mean(xs: number[]): number {
  if (!xs.length) return NaN;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function std(xs: number[]): number {
  if (xs.length < 2) return NaN;
  const m = mean(xs);
  let v = 0;
  for (let i = 0; i < xs.length; i++) {
    const d = xs[i] - m;
    v += d * d;
  }
  return Math.sqrt(v / (xs.length - 1));
}

function mapByDate(s: Series): Dict<number> {
  const m: Dict<number> = {};
  for (let i = 0; i < s.length; i++) m[s[i].t] = s[i].v as number;
  return m;
}

function intersectDates(a: Series, b: Series): string[] {
  const A: Dict<boolean> = {};
  for (let i = 0; i < a.length; i++) A[a[i].t] = true;
  const out: string[] = [];
  for (let j = 0; j < b.length; j++) if (A[b[j].t]) out.push(b[j].t);
  out.sort();
  return out;
}

// Build series from arbitrary rows
export function toSeries<T extends Dict>(rows: T[], timeField: keyof T, valueField: keyof T): Series {
  const out: Series = [];
  for (let i = 0; i < rows.length; i++) {
    const r: any = rows[i];
    const t = toDateOnlyISO(r[timeField]);
    const v = toNumberSafe(r[valueField]);
    if (t && isFiniteNum(v)) out.push({ t, v });
  }
  return cleanSeries(out);
}

// ────────────────────────────────────────────────────────────────────────────
// Moving averages
// ────────────────────────────────────────────────────────────────────────────

export function sma(src: Series, window: number): Series {
  const a = cleanSeries(src);
  const vals = a.map(p => p.v as number);
  const times = a.map(p => p.t);
  const out: Series = [];
  if (window <= 0) return out;
  const rolls = rollingWindow(vals, window);
  for (let i = 0; i < rolls.length; i++) {
    out.push({ t: times[i + window - 1], v: mean(rolls[i]) });
  }
  return out;
}

/** Wilder's RMA (a.k.a. SMMA) */
export function rma(src: Series, window: number): Series {
  const a = cleanSeries(src);
  const out: Series = [];
  if (window <= 0 || a.length === 0) return out;
  // seed with SMA of first window
  if (a.length < window) return out;
  let seed = 0;
  for (let i = 0; i < window; i++) seed += (a[i].v as number);
  let prev = seed / window;
  out.push({ t: a[window - 1].t, v: prev });
  const alpha = 1 / window;
  for (let i = window; i < a.length; i++) {
    const v = a[i].v as number;
    prev = prev + alpha * (v - prev);
    out.push({ t: a[i].t, v: prev });
  }
  return out;
}

/** Exponential MA with alpha = 2/(n+1) (TradingView/TA-lib style) */
export function ema(src: Series, window: number): Series {
  const a = cleanSeries(src);
  const out: Series = [];
  if (window <= 0 || a.length === 0) return out;
  const k = 2 / (window + 1);
  // Seed with SMA of first window
  if (a.length < window) return out;
  let seed = 0;
  for (let i = 0; i < window; i++) seed += (a[i].v as number);
  let prev = seed / window;
  out.push({ t: a[window - 1].t, v: prev });
  for (let i = window; i < a.length; i++) {
    const v = a[i].v as number;
    prev = v * k + prev * (1 - k);
    out.push({ t: a[i].t, v: prev });
  }
  return out;
}

/** Weighted (linear) MA: weights 1..n */
export function wma(src: Series, window: number): Series {
  const a = cleanSeries(src);
  const out: Series = [];
  if (window <= 0 || a.length < window) return out;
  const weights = new Array(window).fill(0).map((_, i) => i + 1);
  const wsum = weights.reduce((s, x) => s + x, 0);
  for (let i = window - 1; i < a.length; i++) {
    let num = 0;
    for (let k = 0; k < window; k++) {
      num += (a[i - window + 1 + k].v as number) * weights[k];
    }
    out.push({ t: a[i].t, v: num / wsum });
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
/** Bollinger Bands: returns mid (SMA), upper, lower, bandwidth, %B */
export type Bollinger = {
  mid: Series;
  upper: Series;
  lower: Series;
  bandwidth: Series; // (upper-lower)/mid
  percentB: Series;  // (price - lower) / (upper - lower)
};

export function bollinger(src: Series, window = 20, mult = 2): Bollinger {
  const a = cleanSeries(src);
  const smaS = sma(a, window);
  const times = smaS.map(p => p.t);
  const outMid = smaS;
  const outUp: Series = [];
  const outLo: Series = [];
  const outBW: Series = [];
  const outPB: Series = [];
  const vals = a.map(p => p.v as number);
  for (let i = window - 1; i < a.length; i++) {
    const wnd = vals.slice(i - window + 1, i + 1);
    const m = mean(wnd);
    const s = std(wnd);
    const up = m + mult * s;
    const lo = m - mult * s;
    const mid = m;
    const t = times[i - window + 1];
    outUp.push({ t, v: up });
    outLo.push({ t, v: lo });
    outBW.push({ t, v: mid !== 0 ? (up - lo) / mid : 0 });
    const price = vals[i];
    outPB.push({ t, v: (up !== lo) ? (price - lo) / (up - lo) : 0 });
  }
  return { mid: outMid, upper: outUp, lower: outLo, bandwidth: outBW, percentB: outPB };
}

// ────────────────────────────────────────────────────────────────────────────
/** MACD (EMA fast/slow + signal EMA), also returns histogram */
export type MACD = { macd: Series; signal: Series; hist: Series };

export function macd(src: Series, fast = 12, slow = 26, signal = 9): MACD {
  const a = cleanSeries(src);
  const emaFast = ema(a, fast);
  const emaSlow = ema(a, slow);
  // align on intersection of dates
  const dates = intersectDates(emaFast, emaSlow);
  const mapF = mapByDate(emaFast);
  const mapS = mapByDate(emaSlow);
  const macdLine: Series = dates.map(t => ({ t, v: (mapF[t] as number) - (mapS[t] as number) }));
  const sig = ema(macdLine, signal);
  const mapSig = mapByDate(sig);
  const hist: Series = [];
  for (let i = 0; i < sig.length; i++) {
    const t = sig[i].t;
    const m = mapByDate(macdLine)[t];
    if (isFiniteNum(m)) hist.push({ t, v: m - (mapSig[t] as number) });
  }
  return { macd: macdLine, signal: sig, hist };
}

// ────────────────────────────────────────────────────────────────────────────
/** RSI (Wilder's method, period=14) */
export function rsi(src: Series, period = 14): Series {
  const a = cleanSeries(src);
  if (a.length < period + 1) return [];
  // Gains/losses
  const gains: number[] = [];
  const losses: number[] = [];
  for (let i = 1; i < a.length; i++) {
    const change = (a[i].v as number) - (a[i - 1].v as number);
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? -change : 0);
  }
  // Seed averages
  let avgGain = 0, avgLoss = 0;
  for (let i = 0; i < period; i++) { avgGain += gains[i]; avgLoss += losses[i]; }
  avgGain /= period; avgLoss /= period;

  const out: Series = [];
  // First RSI at index (period)
  const firstIdx = period;
  let rs = avgLoss === 0 ? Infinity : (avgGain / avgLoss);
  out.push({ t: a[firstIdx].t, v: 100 - 100 / (1 + rs) });

  // Wilder smoothing
  for (let i = period + 1; i < a.length; i++) {
    const g = gains[i - 1];
    const l = losses[i - 1];
    avgGain = (avgGain * (period - 1) + g) / period;
    avgLoss = (avgLoss * (period - 1) + l) / period;
    rs = avgLoss === 0 ? Infinity : (avgGain / avgLoss);
    out.push({ t: a[i].t, v: 100 - 100 / (1 + rs) });
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// True Range & ATR
// ────────────────────────────────────────────────────────────────────────────

export function trueRange(ohlc: OHLC[]): Series {
  const rows = ohlc.slice().sort((a, b) => (String(a.t) < String(b.t) ? -1 : 1));
  const out: Series = [];
  for (let i = 0; i < rows.length; i++) {
    const h = toNumberSafe(rows[i].h);
    const l = toNumberSafe(rows[i].l);
    const cPrev = i > 0 ? toNumberSafe(rows[i - 1].c) : null;
    if (!isFiniteNum(h) || !isFiniteNum(l)) continue;
    const range1 = h - (l as number);
    const range2 = isFiniteNum(cPrev) ? Math.abs((h as number) - (cPrev as number)) : range1;
    const range3 = isFiniteNum(cPrev) ? Math.abs((l as number) - (cPrev as number)) : range1;
    const tr = Math.max(range1, range2, range3);
    const t = toDateOnlyISO(rows[i].t);
    if (t) out.push({ t, v: tr });
  }
  return out;
}

export function atr(ohlc: OHLC[], period = 14): Series {
  const tr = trueRange(ohlc);
  return rma(tr, period); // Wilder ATR = RMA(TR, n)
}

// ────────────────────────────────────────────────────────────────────────────
// Stochastic Oscillator (%K, %D)
// ────────────────────────────────────────────────────────────────────────────

export type Stochastic = { k: Series; d: Series };

export function stochastic(ohlc: OHLC[], kLen = 14, dLen = 3, smoothK = 1): Stochastic {
  // %K = 100 * (close - lowestLow(kLen)) / (highestHigh(kLen) - lowestLow(kLen))
  const rows = ohlc.slice().sort((a, b) => (String(a.t) < String(b.t) ? -1 : 1));
  const highs: number[] = [];
  const lows: number[] = [];
  const closes: number[] = [];
  const times: string[] = [];
  for (let i = 0; i < rows.length; i++) {
    const t = toDateOnlyISO(rows[i].t);
    const h = toNumberSafe(rows[i].h);
    const l = toNumberSafe(rows[i].l);
    const c = toNumberSafe(rows[i].c);
    if (t && isFiniteNum(h) && isFiniteNum(l) && isFiniteNum(c)) {
      times.push(t); highs.push(h); lows.push(l); closes.push(c);
    }
  }
  const kRaw: Series = [];
  for (let i = kLen - 1; i < times.length; i++) {
    let hi = -Infinity, lo = Infinity;
    for (let j = i - kLen + 1; j <= i; j++) {
      if (highs[j] > hi) hi = highs[j];
      if (lows[j] < lo) lo = lows[j];
    }
    const denom = hi - lo;
    const val = denom === 0 ? 0 : (100 * (closes[i] - lo) / denom);
    kRaw.push({ t: times[i], v: val });
  }
  const k = smoothK > 1 ? sma(kRaw, smoothK) : kRaw;
  const d = dLen > 1 ? sma(k, dLen) : k;
  return { k, d };
}

// ────────────────────────────────────────────────────────────────────────────
// Donchian & Keltner Channels
// ────────────────────────────────────────────────────────────────────────────

export type Donchian = { upper: Series; lower: Series; mid: Series };

export function donchian(ohlc: OHLC[], length = 20): Donchian {
  const rows = ohlc.slice().sort((a, b) => (String(a.t) < String(b.t) ? -1 : 1));
  const highs: number[] = [];
  const lows: number[] = [];
  const times: string[] = [];
  for (let i = 0; i < rows.length; i++) {
    const t = toDateOnlyISO(rows[i].t);
    const h = toNumberSafe(rows[i].h);
    const l = toNumberSafe(rows[i].l);
    if (t && isFiniteNum(h) && isFiniteNum(l)) {
      times.push(t); highs.push(h); lows.push(l);
    }
  }
  const up: Series = [], lo: Series = [], md: Series = [];
  for (let i = length - 1; i < times.length; i++) {
    let hi = -Infinity, lw = Infinity;
    for (let j = i - length + 1; j <= i; j++) {
      if (highs[j] > hi) hi = highs[j];
      if (lows[j] < lw) lw = lows[j];
    }
    up.push({ t: times[i], v: hi });
    lo.push({ t: times[i], v: lw });
    md.push({ t: times[i], v: (hi + lw) / 2 });
  }
  return { upper: up, lower: lo, mid: md };
}

export type Keltner = { mid: Series; upper: Series; lower: Series };

export function keltner(ohlc: OHLC[], emaLen = 20, atrLen = 10, mult = 2): Keltner {
  // Typical price
  const tp: Series = [];
  const rows = ohlc.slice().sort((a, b) => (String(a.t) < String(b.t) ? -1 : 1));
  for (let i = 0; i < rows.length; i++) {
    const t = toDateOnlyISO(rows[i].t);
    const h = toNumberSafe(rows[i].h);
    const l = toNumberSafe(rows[i].l);
    const c = toNumberSafe(rows[i].c);
    if (t && isFiniteNum(h) && isFiniteNum(l) && isFiniteNum(c)) {
      tp.push({ t, v: (h + l + c) / 3 });
    }
  }
  const mid = ema(tp, emaLen);
  const atrS = atr(rows, atrLen);
  const mapATR = mapByDate(atrS);
  const outUp: Series = [], outLo: Series = [];
  for (let i = 0; i < mid.length; i++) {
    const t = mid[i].t;
    const m = mid[i].v as number;
    const a = mapATR[t];
    if (!isFiniteNum(a)) continue;
    outUp.push({ t, v: m + mult * a });
    outLo.push({ t, v: m - mult * a });
  }
  return { mid, upper: outUp, lower: outLo };
}

// ────────────────────────────────────────────────────────────────────────────
// VWAP (cumulative). For OHLCV rows; falls back to close if H/L missing.
// ────────────────────────────────────────────────────────────────────────────

export function vwap(rows: OHLCV[]): Series {
  const rs = rows.slice().sort((a, b) => (String(a.t) < String(b.t) ? -1 : 1));
  const out: Series = [];
  let pvSum = 0;
  let volSum = 0;
  for (let i = 0; i < rs.length; i++) {
    const t = toDateOnlyISO(rs[i].t);
    const h = toNumberSafe(rs[i].h);
    const l = toNumberSafe(rs[i].l);
    const c = toNumberSafe(rs[i].c);
    const v = toNumberSafe(rs[i].v);
    if (!t || !isFiniteNum(c) || !isFiniteNum(v)) continue;
    const typical = isFiniteNum(h) && isFiniteNum(l) ? ((h + l + c) / 3) : c;
    pvSum += typical * v;
    volSum += v;
    out.push({ t, v: volSum === 0 ? 0 : pvSum / volSum });
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Z-Score & generic rolling std/mean utilities
// ────────────────────────────────────────────────────────────────────────────

export function rollingMean(src: Series, window: number): Series { return sma(src, window); }

export function rollingStd(src: Series, window: number): Series {
  const a = cleanSeries(src);
  const vals = a.map(p => p.v as number);
  const times = a.map(p => p.t);
  const out: Series = [];
  if (window <= 1 || a.length < window) return out;
  for (let i = window - 1; i < vals.length; i++) {
    const wnd = vals.slice(i - window + 1, i + 1);
    out.push({ t: times[i], v: std(wnd) });
  }
  return out;
}

export function zScore(src: Series, window = 20): Series {
  const mu = rollingMean(src, window);
  const sd = rollingStd(src, window);
  const dates = intersectDates(mu, sd);
  const mapMu = mapByDate(mu);
  const mapSd = mapByDate(sd);
  const mapSrc = mapByDate(cleanSeries(src));
  const out: Series = [];
  for (let i = 0; i < dates.length; i++) {
    const t = dates[i];
    const s = mapSd[t];
    const m = mapMu[t];
    const x = mapSrc[t];
    if (isFiniteNum(s) && s !== 0 && isFiniteNum(m) && isFiniteNum(x)) {
      out.push({ t, v: (x - m) / s });
    }
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Crossovers & signals
// ────────────────────────────────────────────────────────────────────────────

export type Crossovers = { up: string[]; down: string[] };

/**
 * Given two series A and B (aligned by date), returns dates where A crosses above B (up)
 * and where A crosses below B (down).
 */
export function crossovers(a: Series, b: Series): Crossovers {
  const A = cleanSeries(a), B = cleanSeries(b);
  const dates = intersectDates(A, B);
  const mapA = mapByDate(A);
  const mapB = mapByDate(B);
  const up: string[] = [];
  const down: string[] = [];
  let prevDiff: number | null = null;
  for (let i = 0; i < dates.length; i++) {
    const t = dates[i];
    const d = (mapA[t] as number) - (mapB[t] as number);
    if (prevDiff != null) {
      if (prevDiff <= 0 && d > 0) up.push(t);
      if (prevDiff >= 0 && d < 0) down.push(t);
    }
    prevDiff = d;
  }
  return { up, down };
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience: price → returns
// ────────────────────────────────────────────────────────────────────────────

export function simpleReturnsFromPrice(src: Series, periods = 1): Series {
  const a = cleanSeries(src);
  const out: Series = [];
  for (let i = periods; i < a.length; i++) {
    const p0 = a[i - periods].v as number;
    const p1 = a[i].v as number;
    if (p0 === 0) continue;
    out.push({ t: a[i].t, v: (p1 - p0) / p0 });
  }
  return out;
}

export function logReturnsFromPrice(src: Series, periods = 1): Series {
  const a = cleanSeries(src);
  const out: Series = [];
  for (let i = periods; i < a.length; i++) {
    const p0 = a[i - periods].v as number;
    const p1 = a[i].v as number;
    if (p0 <= 0 || p1 <= 0) continue;
    out.push({ t: a[i].t, v: Math.log(p1 / p0) });
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// END
// ────────────────────────────────────────────────────────────────────────────