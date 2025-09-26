// alpha/momentum.ts
// Import-free momentum utilities for both time-series (trend) and cross-sectional momentum.
//
// What you get
// - mom(ret or px, lookback, method) → simple momentum score
// - tsMomentumSignal(px, {lookback, method, volAdj}) → +1/-1/0 trend signal
// - macSignal(px, {fast, slow}) → moving-average crossover (+1/-1/0)
// - xSectionRanks(latestRets, {topK, longShort, thr}) → ranked cross-sectional picks
// - volTargetWeights(pairs, volMap, {annFactor, maxWeight}) → 1/σ weights
// - rebalanceMomentum(port, targets, vol, opts) → trades vs current weights
// - backtestTSMom(px, {lookback, volTarget, annFactor, feeBps, slipBps}) → equity path
// - backtestCSMom(matrix, tickers, opts) → simple cross-sectional momentum backtest
//
// Conventions
// - Prices are chronological arrays (oldest → newest). Returns are simple unless specified.
// - Signals: +1 long trend, -1 short trend, 0 flat.
// - All weights are fractions of equity (dimensionless).
//
// NOTE: This is a pedagogical, dependency-free module. No external imports.

export type Num = number;

export type MomMethod = "price_return" | "log_return" | "zscore";
export type Side = -1 | 0 | 1;

export function mom(
  input: { prices?: Num[]; returns?: Num[]; lookback?: number; method?: MomMethod }
): number {
  const L = Math.max(1, (input.lookback ?? 126) | 0);
  const method = input.method ?? "price_return";
  if (Array.isArray(input.returns)) {
    const r = input.returns;
    const n = r.length;
    if (n < L) return 0;
    if (method === "zscore") {
      const seg = r.slice(n - L, n);
      const m = mean(seg), s = stdev(seg);
      return s > 0 ? (seg[seg.length - 1] - m) / s : 0;
    }
    // sum of returns over window (approx cumulative)
    let s = 0;
    for (let i = n - L; i < n; i++) s += r[i];
    return s;
  }
  const p = input.prices || [];
  const n = p.length;
  if (n < L + 1) return 0;
  const p0 = p[n - 1 - L], p1 = p[n - 1];
  if (p0 <= 0 || p1 <= 0) return 0;
  if (method === "log_return") return Math.log(p1 / p0);
  if (method === "zscore") {
    const ret = toRets(p);
    const seg = ret.slice(ret.length - L);
    const m = mean(seg), s = stdev(seg);
    return s > 0 ? (seg[seg.length - 1] - m) / s : 0;
  }
  return p1 / p0 - 1; // simple return over window
}

export function tsMomentumSignal(
  prices: Num[],
  opts: { lookback?: number; method?: MomMethod; volAdj?: boolean; annFactor?: number } = {}
): Side {
  const L = Math.max(10, (opts.lookback ?? 126) | 0);
  if (prices.length < L + 1) return 0;
  const score = mom({ prices, lookback: L, method: opts.method ?? "price_return" });
  if (opts.volAdj) {
    const ann = Math.max(1, (opts.annFactor ?? 252) | 0);
    const ret = toRets(prices).slice(-L);
    const v = stdev(ret) * Math.sqrt(ann);
    const adj = v > 0 ? score / Math.max(1e-9, v) : score;
    return adj > 0 ? 1 : adj < 0 ? -1 : 0;
  }
  return score > 0 ? 1 : score < 0 ? -1 : 0;
}

export function macSignal(
  prices: Num[],
  opts: { fast?: number; slow?: number; type?: "sma" | "ema" } = {}
): Side {
  const f = Math.max(2, (opts.fast ?? 50) | 0);
  const s = Math.max(f + 1, (opts.slow ?? 200) | 0);
  if (prices.length < s) return 0;
  const type = opts.type ?? "sma";
  const a = type === "ema" ? ema(prices, f) : sma(prices, f);
  const b = type === "ema" ? ema(prices, s) : sma(prices, s);
  const af = a[a.length - 1], bf = b[b.length - 1];
  if (!isFinite(af) || !isFinite(bf)) return 0;
  return af > bf ? 1 : af < bf ? -1 : 0;
}

/* ---------------------------- Cross-Section ----------------------------- */

export type RankOptions = {
  topK?: number;
  longShort?: boolean;     // if true, mirror bottom K shorts
  threshold?: number;      // min |momentum| to include
  blacklist?: string[];
};

export type RankRow = { ticker: string; score: number; rank: number };

export function xSectionRanks(
  lastWindowReturns: Record<string, number>, // e.g., { AAPL: 0.12, MSFT: 0.05, ... }
  opts: RankOptions = {}
): RankRow[] {
  const thr = num(opts.threshold, 0);
  const bad = new Set((opts.blacklist || []).map((x) => x.toUpperCase()));
  const arr: RankRow[] = [];
  for (const k in lastWindowReturns) {
    const t = k.toUpperCase();
    if (bad.has(t)) continue;
    const s = num(lastWindowReturns[k]);
    if (Math.abs(s) < thr) continue;
    arr.push({ ticker: t, score: s, rank: 0 });
  }
  arr.sort((a, b) => b.score - a.score);
  for (let i = 0; i < arr.length; i++) arr[i].rank = i + 1;

  if (opts.topK && opts.topK > 0) {
    const k = Math.min(opts.topK, arr.length);
    if (opts.longShort) {
      const top = arr.slice(0, k);
      const bot = arr.slice(-k);
      return uniqueBy([...top, ...bot], (x) => x.ticker).sort((a, b) => b.score - a.score)
        .map((x, i) => ({ ...x, rank: i + 1 }));
    }
    return arr.slice(0, k);
  }
  return arr;
}

/* ---------------------------- Weighting & Rebal ------------------------- */

export type Weights = Record<string, number>;
export type VolMap = Record<string, number>;

export function volTargetWeights(
  tickers: string[],
  vol: VolMap,
  opts: { annFactor?: number; maxWeight?: number } = {}
): Weights {
  const ann = Math.max(1, num(opts.annFactor, 252));
  const cap = isFiniteNum(opts.maxWeight) ? Math.abs(opts.maxWeight!) : Infinity;
  const raw: Record<string, number> = {};
  let sum = 0;
  for (let i = 0; i < tickers.length; i++) {
    const t = tickers[i];
    const d = num(vol[t]);
    if (!(d > 0)) continue;
    const annVol = d * Math.sqrt(ann);
    const w = annVol > 0 ? 1 / annVol : 0;
    raw[t] = w; sum += w;
  }
  if (sum === 0) return {};
  const out: Weights = {};
  let norm = 0;
  for (const k in raw) { out[k] = Math.min(cap, raw[k] / sum); norm += out[k]; }
  for (const k in out) out[k] = out[k] / Math.max(1e-9, norm);
  return out;
}

export type PortState = { weights: Weights; equity: number };
export type RebalanceOpts = {
  volTarget?: number;
  annFactor?: number;
  maxWeight?: number;
  grossCap?: number;
  feeBps?: number;
};

export type Trade = { ticker: string; delta: number };

export function rebalanceMomentum(
  port: PortState,
  longs: string[],
  shorts: string[],
  vol: VolMap,
  opts: RebalanceOpts = {}
) {
  const ann = Math.max(1, num(opts.annFactor, 252));
  const vt = Math.max(0, num(opts.volTarget, 0));
  const fee = Math.max(0, num(opts.feeBps, 0)) / 10_000;
  const grossCap = isFiniteNum(opts.grossCap) ? Math.abs(opts.grossCap!) : 1.0;
  const perCap = isFiniteNum(opts.maxWeight) ? Math.abs(opts.maxWeight!) : 1.0;

  const wL = volTargetWeights(longs, vol, { annFactor: ann, maxWeight: perCap });
  const wS = volTargetWeights(shorts, vol, { annFactor: ann, maxWeight: perCap });

  let comb: Weights = {}; let gross = 0;
  for (const k in wL) { comb[k] = (comb[k] || 0) + wL[k]; gross += wL[k]; }
  for (const k in wS) { comb[k] = (comb[k] || 0) - wS[k]; gross += wS[k]; }

  if (gross > 0) {
    const scaleGross = Math.min(1, grossCap / gross);
    for (const k in comb) comb[k] *= scaleGross;
  }

  if (vt > 0) {
    let v2 = 0;
    for (const k in comb) {
      const d = num(vol[k]); if (!(d > 0)) continue;
      const annVol = d * Math.sqrt(ann);
      v2 += Math.abs(comb[k]) * Math.abs(comb[k]) * (annVol * annVol);
    }
    const cur = Math.sqrt(Math.max(0, v2));
    const scale = cur > 0 ? vt / cur : 1;
    for (const k in comb) comb[k] *= scale;
  }

  const trades: Trade[] = [];
  let turnover = 0;
  for (const k in comb) {
    const d = comb[k] - num(port.weights[k]);
    if (Math.abs(d) > 1e-9) { trades.push({ ticker: k, delta: d }); turnover += Math.abs(d); }
  }
  for (const k in port.weights) {
    if (!(k in comb) && Math.abs(port.weights[k]) > 0) {
      const d = -port.weights[k];
      trades.push({ ticker: k, delta: d }); turnover += Math.abs(d);
    }
  }
  const estCost = turnover * fee * port.equity;

  const after: Weights = { ...port.weights };
  for (let i = 0; i < trades.length; i++) {
    const t = trades[i];
    after[t.ticker] = round6(num(after[t.ticker]) + t.delta);
    if (Math.abs(after[t.ticker]) < 1e-10) delete after[t.ticker];
  }

  return { trades, weightsAfter: after, estCost: round2(estCost) };
}

/* ------------------------------ Backtests ------------------------------- */

export function backtestTSMom(
  prices: Num[],
  opts: { lookback?: number; volTarget?: number; annFactor?: number; feeBps?: number; slipBps?: number; startEquity?: number } = {}
) {
  const L = Math.max(20, (opts.lookback ?? 126) | 0);
  if (prices.length < L + 2) return { equity: [] as { i: number; equity: number }[], stats: emptyStats() };
  const ann = Math.max(1, num(opts.annFactor, 252));
  const fee = Math.max(0, num(opts.feeBps, 0)) / 10_000;
  const slip = Math.max(0, num(opts.slipBps, 0)) / 10_000;
  const vt = Math.max(0, num(opts.volTarget, 0));
  const start = num(opts.startEquity, 1);

  let equity = start;
  let pos = 0; // target exposure [-1,1]
  const curve: { i: number; equity: number }[] = [];

  for (let i = L + 1; i < prices.length; i++) {
    // signal at i-1, apply at i
    const sig = tsMomentumSignal(prices.slice(0, i), { lookback: L, method: "price_return", volAdj: vt > 0, annFactor: ann });
    let target = sig as number;

    if (vt > 0) {
      const r = toRets(prices.slice(0, i));
      const v = stdev(r.slice(-L)) * Math.sqrt(ann);
      const scale = v > 0 ? Math.min(3, vt / v) : 1; // cap scaling
      target *= scale;
      target = clamp(target, -1.5, 1.5);
    }

    const d = target - pos;
    if (Math.abs(d) > 1e-12) {
      const tc = Math.abs(d) * (fee + slip) * equity;
      equity = Math.max(0, equity - tc);
      pos = target;
    }

    const ret = prices[i] / prices[i - 1] - 1;
    equity *= (1 + pos * ret);
    curve.push({ i, equity: round6(equity) });
  }

  const rets = toRets(curve.map(c => c.equity));
  const { annVol, sharpe } = quickStats(rets);
  const { maxDD } = drawdownFromEquity(curve.map(c => c.equity));

  return {
    equity: curve,
    stats: {
      ret: round6(curve.length ? curve[curve.length - 1].equity / start - 1 : 0),
      vol: round6(annVol),
      sharpe: round6(sharpe),
      maxDD: round6(maxDD),
      trades: approxTradesFromCurve(curve),
    },
  };
}

/**
 * Cross-sectional momentum backtest
 * matrix: prices[t][i] where i indexes tickers[], chronological t.
 */
export function backtestCSMom(
  matrix: Num[][],
  tickers: string[],
  opts: {
    lookback?: number; rebalanceEvery?: number; annFactor?: number;
    feeBps?: number; slipBps?: number; startEquity?: number;
    topK?: number; longShort?: boolean; volTarget?: number; maxWeight?: number; grossCap?: number;
    volDaily?: VolMap[]; // optional per-date vol maps
  } = {}
) {
  const T = matrix.length;
  const N = tickers.length;
  if (T < 2 || N === 0) return { equity: [] as { t: number; equity: number }[], log: [] as any[], stats: emptyStats() };

  const L = Math.max(20, (opts.lookback ?? 126) | 0);
  const every = Math.max(1, (opts.rebalanceEvery ?? 21) | 0);
  const ann = Math.max(1, num(opts.annFactor, 252));
  const fee = Math.max(0, num(opts.feeBps, 0)) / 10_000;
  const slip = Math.max(0, num(opts.slipBps, 0)) / 10_000;
  const start = num(opts.startEquity, 1);

  let port: PortState = { weights: {}, equity: start };
  const curve: { t: number; equity: number }[] = [];
  const log: any[] = [];

  for (let t = L; t < T; t++) {
    // rebalance?
    if ((t - L) % every === 0) {
      const map: Record<string, number> = {};
      for (let i = 0; i < N; i++) {
        const px = seriesCol(matrix, i);
        const r = px[t] / px[t - L] - 1;
        map[tickers[i]] = r;
      }
      const ranks = xSectionRanks(map, { topK: opts.topK ?? Math.floor(N / 5), longShort: opts.longShort ?? true });
      const longs = ranks.filter(r => r.score > 0).map(r => r.ticker);
      const shorts = ranks.filter(r => r.score < 0).map(r => r.ticker);

      const vol = opts.volDaily && opts.volDaily[t] ? opts.volDaily[t]! : {};
      const reb = rebalanceMomentum(
        port, longs, shorts, vol,
        { volTarget: opts.volTarget ?? 0, annFactor: ann, maxWeight: opts.maxWeight ?? 0.1, grossCap: opts.grossCap ?? 1, feeBps: opts.feeBps ?? 0 }
      );
      port.weights = reb.weightsAfter;
      port.equity = Math.max(0, port.equity - reb.estCost);
      log.push({ t, event: "rebalance", trades: reb.trades, cost: reb.estCost, weights: { ...port.weights } });
    }

    // daily P&L using asset returns
    let dayRet = 0;
    for (const k in port.weights) {
      const idx = tickers.indexOf(k);
      if (idx < 0) continue;
      const r = matrix[t][idx] / matrix[t - 1][idx] - 1;
      dayRet += port.weights[k] * r;
    }
    // apply costs for a small slippage factor daily if provided
    port.equity *= (1 + dayRet);
    curve.push({ t, equity: round6(port.equity) });
  }

  const rets = toRets(curve.map(x => x.equity));
  const { annVol, sharpe } = quickStats(rets);
  const { maxDD } = drawdownFromEquity(curve.map(x => x.equity));

  return {
    equity: curve,
    log,
    stats: {
      ret: round6(curve.length ? curve[curve.length - 1].equity / start - 1 : 0),
      vol: round6(annVol),
      sharpe: round6(sharpe),
      maxDD: round6(maxDD),
      trades: log.reduce((a, b) => a + (Array.isArray(b.trades) ? b.trades.length : 0), 0),
    },
  };
}

/* --------------------------------- Helpers -------------------------------- */

function sma(p: Num[], w: number) {
  const out: Num[] = []; let s = 0; const q: Num[] = [];
  for (let i = 0; i < p.length; i++) {
    s += p[i]; q.push(p[i]); if (q.length > w) s -= q.shift()!;
    out.push(q.length === w ? s / w : NaN);
  }
  return out;
}

function ema(p: Num[], w: number) {
  const out: Num[] = []; const k = 2 / (w + 1);
  let e = p[0];
  for (let i = 0; i < p.length; i++) { e = i === 0 ? p[0] : k * p[i] + (1 - k) * e; out.push(e); }
  return out;
}

function toRets(p: Num[]) { const out: Num[] = []; for (let i = 1; i < p.length; i++) out.push(p[i] / p[i - 1] - 1); return out; }
function mean(a: Num[]) { let s = 0, n = 0; for (let i = 0; i < a.length; i++) { const v = a[i]; if (isFinite(v)) { s += v; n++; } } return n ? s / n : 0; }
function stdev(a: Num[]) { const m = mean(a); let s = 0, n = 0; for (let i = 0; i < a.length; i++) { const v = a[i]; if (isFinite(v)) { const d = v - m; s += d * d; n++; } } return n > 1 ? Math.sqrt(s / (n - 1)) : 0; }
function seriesCol(matrix: Num[][], col: number) { const out: Num[] = []; for (let r = 0; r < matrix.length; r++) out.push(matrix[r][col]); return out; }

function quickStats(rets: Num[], ann = 252) {
  const m = mean(rets), s = stdev(rets);
  return { annVol: s * Math.sqrt(ann), sharpe: s !== 0 ? (m / s) * Math.sqrt(ann) : 0 };
}

function drawdownFromEquity(eq: Num[]) {
  let peak = -Infinity, maxDD = 0;
  for (let i = 0; i < eq.length; i++) {
    const e = eq[i]; peak = Math.max(peak, e);
    if (peak > 0) { const dd = (peak - e) / peak; if (dd > maxDD) maxDD = dd; }
  }
  return { maxDD };
}

function approxTradesFromCurve(curve: { i: number; equity: number }[]) {
  // crude: count sign changes in equity returns as a proxy (not exact orders)
  if (curve.length < 3) return 0;
  let cnt = 0;
  for (let i = 2; i < curve.length; i++) {
    const r1 = curve[i].equity / curve[i - 1].equity - 1;
    const r0 = curve[i - 1].equity / curve[i - 2].equity - 1;
    if (r1 * r0 < 0) cnt++;
  }
  return cnt;
}

function uniqueBy<T>(arr: T[], key: (x: T) => string): T[] {
  const seen = new Set<string>(); const out: T[] = [];
  for (let i = 0; i < arr.length; i++) {
    const k = key(arr[i]); if (seen.has(k)) continue; seen.add(k); out.push(arr[i]);
  }
  return out;
}

function clamp(x: number, lo: number, hi: number) { return Math.min(hi, Math.max(lo, x)); }
function num(v: any, d = 0) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function isFiniteNum(v: any) { const n = Number(v); return Number.isFinite(n); }
function round2(n: number) { return Math.round(n * 100) / 100; }
function round6(n: number) { return Math.round(n * 1e6) / 1e6; }

function emptyStats() { return { ret: 0, vol: 0, sharpe: 0, maxDD: 0, trades: 0 }; }