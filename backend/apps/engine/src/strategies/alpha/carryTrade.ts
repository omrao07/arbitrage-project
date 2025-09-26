// alpha/carrytrade.ts
// Import-free FX Carry (and generalized asset carry) alpha utilities.
//
// What this gives you
// - fxCarrySignal({ spot, fwd, T } | { rBase, rQuote, T }) → annualized carry
// - rankSignals(universe, { topK | longShort | threshold }) → long/short lists
// - volTargetWeights(returnsCov | stdev map, targets) → risk-balanced weights
// - rebalanceCarry(port, signals, prices, opts) → trade deltas for rebalancing
// - backtestCarry(daily, opts) → daily P&L/portfolio path with simple costs
//
// Notes
// - Carry definition (annualized):
//      carry ≈ (F/S - 1)/T   (forward discount)  ≈ (rBase - rQuote) for FX
//   Positive carry → long BASE/short QUOTE (e.g., long AUDUSD when AUD rate > USD).
// - Everything is synchronous and side-effect free.
// - No external imports. Pure TypeScript, safe for Node or browser.
//
// Types key:
//   Pair "EURUSD" means BASE=EUR, QUOTE=USD. Spot is BASE/QUOTE.
//
// ---------------------------------------------------------------------------

/* ============================== Types =================================== */

export type Num = number;

export type FXInputs =
  | { spot: Num; fwd: Num; T: Num }                 // use forwards
  | { rBase: Num; rQuote: Num; T: Num };            // or rate differential directly

export type CarrySignal = {
  pair: string;        // e.g., "AUDUSD"
  carryAnn: number;    // annualized carry (decimal p.a.)
  T: number;           // years used
};

export type SignalRow = CarrySignal & {
  rank: number;        // 1 = best (highest carry)
};

export type RankOptions = {
  topK?: number;                 // take top K longs (and optionally bottom K shorts if longShort)
  longShort?: boolean;           // if true, create symmetric short leg
  threshold?: number;            // minimum |carry| to include (e.g., 0.005 = 50 bps)
  blacklist?: string[];          // exclude symbols
};

export type Weights = Record<string, number>; // pair -> weight (+ long BASE, - short BASE)

export type VolMap = Record<string, number>;  // pair -> daily stdev (decimal)

export type RebalanceOptions = {
  volTarget?: number;            // portfolio annualized vol target (e.g., 0.10)
  annFactor?: number;            // 252 by default
  maxWeight?: number;            // cap per leg (abs)
  grossCap?: number;             // cap sum of abs weights
  feeBps?: number;               // per-dollar turnover cost (both ways) in bps
};

export type Trade = { pair: string; delta: number }; // + increases long BASE exposure

export type PortfolioState = {
  weights: Weights;     // current weights
  equity: number;       // base currency equity (P&L tracked in same)
};

export type RebalanceResult = {
  trades: Trade[];
  weightsAfter: Weights;
  estCost: number;      // equity points
};

/* ============================ Core Signals ============================== */

/**
 * Compute annualized FX carry. You can provide (spot, fwd, T) OR (rBase, rQuote, T).
 * Returns carry in decimal p.a. (e.g., 0.03 = +3%).
 */
export function fxCarrySignal(input: FXInputs): number {
  const T = Math.max(1e-9, num((input as any).T));
  if (has(input, "spot") && has(input, "fwd")) {
    const S = num((input as any).spot);
    const F = num((input as any).fwd);
    if (S <= 0 || F <= 0) return 0;
    return (F / S - 1) / T; // simple compounding
  }
  const rB = num((input as any).rBase);
  const rQ = num((input as any).rQuote);
  return rB - rQ;
}

/**
 * Build ranked signals from a universe.
 * universe: Array<{ pair, carryAnn, T }> (you can precompute carryAnn or pass raw inputs).
 */
export function rankSignals(
  rows: Array<{ pair: string } & (CarrySignal | FXInputs)>,
  opts: RankOptions = {}
): SignalRow[] {
  const thr = num(opts.threshold, 0);
    const bad = new Set((opts.blacklist || []).map(x => x.toUpperCase()));
  
  const list: SignalRow[] = [];

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const pair = r.pair.toUpperCase();
    if (bad.has(pair)) continue;

    let carryAnn: number;
    let T: number;
    if (has(r as any, "carryAnn")) {
      carryAnn = num((r as any).carryAnn);
      T = Math.max(1e-9, num((r as any).T, 0.25));
    } else {
      carryAnn = fxCarrySignal(r as any);
      T = Math.max(1e-9, num((r as any).T, 0.25));
    }
    if (Math.abs(carryAnn) < thr) continue;

    list.push({ pair, carryAnn, T, rank: 0 });
  }

  list.sort((a, b) => b.carryAnn - a.carryAnn); // highest carry first
  for (let i = 0; i < list.length; i++) list[i].rank = i + 1;

  // If topK set, slice; if longShort, we’ll mirror later in weights
  if (opts.topK && opts.topK > 0) {
    const k = Math.min(opts.topK, list.length);
    if (opts.longShort) {
      // take top K for longs and bottom K for shorts (if available)
      const top = list.slice(0, k);
      const bot = list.slice(-k);
      // ensure uniqueness when overlap (small universes)
      const keep = uniqueBy([...top, ...bot], (x) => x.pair);
      // re-rank for neatness
      keep.sort((a, b) => b.carryAnn - a.carryAnn);
      for (let i = 0; i < keep.length; i++) keep[i].rank = i + 1;
      return keep;
    }
    return list.slice(0, k);
  }
  return list;
}

/* ============================ Weighting Rules =========================== */

/**
 * Risk-budget weights by inverse vol with optional caps.
 * vol: daily stdev map; annFactor converts to annual.
 * If a pair is missing vol, it receives 0 weight.
 * Returns normalized weights summing to 1 (by abs for long/short).
 */
export function volTargetWeights(
  pairs: string[],
  vol: VolMap,
  opts: { annFactor?: number; maxWeight?: number } = {}
): Weights {
  const ann = Math.max(1, num(opts.annFactor, 252));
  const caps = isFiniteNum(opts.maxWeight) ? Math.abs(opts.maxWeight!) : Infinity;

  // inverse annualized vol weights
  const raw: Record<string, number> = {};
  let sum = 0;
  for (let i = 0; i < pairs.length; i++) {
    const p = pairs[i];
    const d = num(vol[p]);
    if (!(d > 0)) continue;
    const annVol = d * Math.sqrt(ann);
    const w = annVol > 0 ? 1 / annVol : 0;
    raw[p] = w;
    sum += w;
  }
  if (sum === 0) return {};

  // normalize & cap
  const out: Weights = {};
  let norm = 0;
  for (const k in raw) {
    out[k] = Math.min(caps, raw[k] / sum);
    norm += out[k];
  }
  // renormalize to sum to 1
  for (const k in out) out[k] = out[k] / Math.max(1e-9, norm);
  return out;
}

/* ============================ Rebalancer ================================ */

/**
 * Turn signals into target weights (longShort supported) and compute trades.
 * price map is not used for sizing (weights are fraction of equity), but
 * included for extensibility (e.g., min notional checks).
 */
export function rebalanceCarry(
  port: PortfolioState,
  signals: SignalRow[],
  vol: VolMap,
  opts: RebalanceOptions = {}
): RebalanceResult {
  const ann = Math.max(1, num(opts.annFactor, 252));
  const vt = Math.max(0, num(opts.volTarget, 0)); // if 0 → no scaling
  const fee = Math.max(0, num(opts.feeBps, 0)) / 10_000;
  const grossCap = isFiniteNum(opts.grossCap) ? Math.abs(opts.grossCap!) : 1.0;
  const perCap = isFiniteNum(opts.maxWeight) ? Math.abs(opts.maxWeight!) : 1.0;

  // Separate long & short by carry sign
  const longs = signals.filter(s => s.carryAnn > 0).map(s => s.pair);
  const shorts = signals.filter(s => s.carryAnn < 0).map(s => s.pair);

  // Risk budgets
  const wLong = volTargetWeights(longs, vol, { annFactor: ann, maxWeight: perCap });
  const wShort = volTargetWeights(shorts, vol, { annFactor: ann, maxWeight: perCap });

  // Combine; longs positive, shorts negative; normalize absolute sum to <= grossCap
  let comb: Weights = {};
  let gross = 0;
  for (const k in wLong) { comb[k] = (comb[k] || 0) + wLong[k]; gross += wLong[k]; }
  for (const k in wShort) { comb[k] = (comb[k] || 0) - wShort[k]; gross += wShort[k]; }

  if (gross > 0) {
    const scaleGross = Math.min(1, grossCap / gross);
    for (const k in comb) comb[k] *= scaleGross;
  }

  // Optional portfolio-level vol target using naive risk proxy:
  // portVol^2 ≈ Σ |w_i|^2 * vol_i^2  (ignoring correlation)
  if (vt > 0) {
    let v2 = 0;
    for (const k in comb) {
      const d = num(vol[k]); if (!(d > 0)) continue;
      const annVol = d * Math.sqrt(ann);
      v2 += Math.abs(comb[k]) * Math.abs(comb[k]) * (annVol * annVol);
    }
    const portAnnVol = Math.sqrt(Math.max(0, v2));
    const scale = portAnnVol > 0 ? vt / portAnnVol : 1;
    for (const k in comb) comb[k] *= scale;
  }

  // Build trades vs existing
  const trades: Trade[] = [];
  let turnover = 0;
  for (const k in comb) {
    const cur = num(port.weights[k]);
    const tgt = comb[k];
    const d = tgt - cur;
    if (Math.abs(d) > 1e-9) {
      trades.push({ pair: k, delta: d });
      turnover += Math.abs(d);
    }
  }
  // Also close stale names not in comb
  for (const k in port.weights) {
    if (!(k in comb) && Math.abs(port.weights[k]) > 0) {
      const d = -port.weights[k];
      trades.push({ pair: k, delta: d });
      turnover += Math.abs(d);
    }
  }

  const estCost = turnover * fee * port.equity; // linear in weights * equity

  // Apply trades to weights (pure result preview)
  const after: Weights = { ...port.weights };
  for (let i = 0; i < trades.length; i++) {
    const t = trades[i];
    after[t.pair] = round6(num(after[t.pair]) + t.delta);
    if (Math.abs(after[t.pair]) < 1e-10) delete after[t.pair];
  }

  return { trades, weightsAfter: after, estCost: round2(estCost) };
}

/* =============================== Backtest =============================== */

export type DailyRow = {
  ts: number;                         // epoch ms
  pairs: Record<string, { spot: number; fwdNext?: number }>; // price book
  rates?: Record<string, { rBase?: number; rQuote?: number }>;
  vol?: VolMap;                       // daily stdevs for sizing on this date (optional)
};

export type BacktestOptions = RebalanceOptions & {
  rebalanceEvery?: number;  // days (default 21)
  startEquity?: number;     // default 1.0
  useForwards?: boolean;    // if true, use fwdNext for carry; else use rates if provided
};

/**
 * Very simple daily backtest:
 * - At rebalance dates, compute signals and weights.
 * - Daily P&L: sum_i weight_i * carry_i/annFactor + mark-to-market from spot drift proxy.
 *   For simplicity, we approximate daily carry as carryAnn/annFactor and ignore FX mtm cross effects.
 * - Costs charged on rebalances using feeBps & turnover.
 */
export function backtestCarry(data: DailyRow[], universe: string[], opts: BacktestOptions = {}) {
  if (!Array.isArray(data) || data.length === 0) return { equity: [] as { ts: number; equity: number }[], log: [] as any[] };

  const ann = Math.max(1, num(opts.annFactor, 252));
  const fee = Math.max(0, num(opts.feeBps, 0)) / 10_000;
  const every = Math.max(1, num(opts.rebalanceEvery, 21));
  const startEq = num(opts.startEquity, 1);

  let port: PortfolioState = { weights: {}, equity: startEq };
  const path: Array<{ ts: number; equity: number }> = [];
  const log: any[] = [];

  for (let d = 0; d < data.length; d++) {
    const row = data[d];

    // Rebalance?
    if (d % every === 0 || d === 0) {
      const sigs: SignalRow[] = [];
      for (let i = 0; i < universe.length; i++) {
        const pair = universe[i];
        const px = row.pairs[pair];
        if (!px || !isFinite(px.spot)) continue;

        let carryAnn = 0, T = 1; // assume 1y tenor by default
        if (opts.useForwards && isFinite(px.fwdNext)) {
          carryAnn = fxCarrySignal({ spot: px.spot, fwd: px.fwdNext!, T });
        } else if (row.rates && row.rates[pair]) {
          const r = row.rates[pair]!;
          carryAnn = fxCarrySignal({ rBase: num(r.rBase), rQuote: num(r.rQuote), T });
        } else {
          // fallback: no signal
          continue;
        }
        sigs.push({ pair, carryAnn, T, rank: 0 });
      }
      // rank, select both sides (long/short)
      const ranked = rankSignals(sigs, { longShort: true });
      const vol = row.vol || {};
      const reb = rebalanceCarry(port, ranked, vol, opts);
      // apply trades (only weights & fee deduction)
      port.weights = reb.weightsAfter;
      port.equity = Math.max(0, port.equity - reb.estCost);
      log.push({ ts: row.ts, event: "rebalance", trades: reb.trades, cost: reb.estCost, weights: { ...port.weights } });
    }

    // Daily P&L: carry accrual
    let pnl = 0;
    for (const k in port.weights) {
      // find current carry for k
      const px = row.pairs[k];
      if (!px) continue;
      let carryAnn = 0;
      if (opts.useForwards && isFinite(px.fwdNext)) {
        carryAnn = fxCarrySignal({ spot: px.spot, fwd: px.fwdNext!, T: 1 });
      } else if (row.rates && row.rates[k]) {
        const r = row.rates[k]!;
        carryAnn = fxCarrySignal({ rBase: num(r.rBase), rQuote: num(r.rQuote), T: 1 });
      }
      pnl += port.weights[k] * (carryAnn / ann) * port.equity;
    }

    port.equity = Math.max(0, port.equity + pnl);
    path.push({ ts: row.ts, equity: round6(port.equity) });
  }

  return { equity: path, log };
}

/* ============================== Utilities =============================== */

function uniqueBy<T>(arr: T[], key: (x: T) => string): T[] {
  const seen = new Set<string>(); const out: T[] = [];
  for (let i = 0; i < arr.length; i++) {
    const k = key(arr[i]); if (seen.has(k)) continue;
    seen.add(k); out.push(arr[i]);
  }
  return out;
}

function has<T extends object>(o: T, k: string) { return Object.prototype.hasOwnProperty.call(o, k); }
function num(v: any, d = 0) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function isFiniteNum(v: any) { const n = Number(v); return Number.isFinite(n); }
function round2(n: number) { return Math.round(n * 100) / 100; }
function round6(n: number) { return Math.round(n * 1e6) / 1e6; }

/* --------------------------------- Example ------------------------------ */
// const sig = fxCarrySignal({ spot: 1.00, fwd: 1.02, T: 1 }); // ≈ +2% carry
// const ranked = rankSignals([{ pair: "AUDUSD", carryAnn: 0.02, T: 1 }, { pair: "EURUSD", carryAnn: -0.01, T: 1 }], { longShort: true });
// const vol: VolMap = { AUDUSD: 0.008, EURUSD: 0.007 };
// const res = rebalanceCarry({ weights: {}, equity: 1_000_000 }, ranked, vol, { volTarget: 0.10, feeBps: 5 });
// console.log(res);