// riskarb/mergerarb.ts
// Import-free utilities for merger/arbitrage (cash & stock deals).
//
// What you get (all pure functions):
// - dealMetrics(d) → spread, downside, base/alt scenarios, exp return, ann IRR
// - probFromSpread(d) → implied probability from market spread
// - stockSwapTerms(d) → map cash/stock mix into effective per-target consideration
// - kellySize({edge, vol}, frac) → Kelly sizing helper (fractional Kelly)
// - basketPlan(deals, opts) → portfolio weights sized by risk/vol/DD target
// - simulate(deals, pathOpts) → simple Monte style path outcomes (close/break)
// - backtest(daily, deals, opts) → timeline P&L from historical priced series
//
// Conventions
// - Prices in same currency. Dates are epoch ms.
// - Cash deal: considerationCash > 0; stock leg optional (mix).
// - Stock deal: set exchangeRatio > 0 and acquirerPx for preview.
// - Use borrowFee for acquirer short borrow (annualized decimal).
//
// NOTE: This is a desk-focused, dependency-free module. No I/O, no imports.

export type Num = number;

export type DealType = "CASH" | "STOCK" | "MIX";
export type Status = "announced" | "pending" | "closed" | "terminated";

export type Deal = {
  id: string;                  // unique key
  tickerTgt: string;           // target ticker
  tickerAcq?: string;          // acquirer ticker (if stock leg)
  type: DealType;
  annDate?: number;            // announced (ms)
  expCloseDate?: number;       // expected close (ms)
  dropDeadDate?: number;       // long stop (ms)

  // Market inputs (latest)
  pxTgt: number;               // target last price
  pxAcq?: number;              // acquirer last price (if stock/mix)
  borrowFeeAcq?: number;       // annualized borrow fee for acquirer short (decimal)
  divYieldAcq?: number;        // annualized dividend yield paid by acquirer (short owes)

  // Terms
  considerationCash?: number;  // cash per target share
  exchangeRatio?: number;      // acquirer shares offered per target share (can be 0 if pure cash)
  collar?: {                   // optional fixed value collar bounds for exchange ratio
    lowerPx?: number;          // acquirer px lower bound for fixed exchange
    upperPx?: number;          // acquirer px upper bound
    fixedAtLower?: number;     // fixed exchange at lower bound
    fixedAtUpper?: number;     // fixed exchange at upper bound
  };

  // Risk modeling
  pClose?: number;             // subjective probability of closing (0..1)
  breakPx?: number;            // assumed break price for target
  closeLagDays?: number;       // settlement lag after closing
  notes?: string;

  // Status
  status?: Status;
};

export type Consideration = {
  cashPerShare: number;
  stockPerShare: number;     // shares of acquirer delivered per target share (effective)
  valuePerShare: number;     // using provided pxAcq
};

export type DealMetrics = {
  id: string;
  type: DealType;
  timeToCloseY: number;        // in years (bounded at ~0 if past)
  spread: number;              // (consid - pxTgt) / pxTgt, using current acquirer price
  downMove: number;            // (breakPx - pxTgt) / pxTgt (negative if below)
  borrowCarryY: number;        // annual carry cost on acquirer short (if any)
  expReturn: number;           // pClose * closeRet + (1-p) * breakRet - carryAdj
  expIRR: number;              // annualized (approx) expected IRR (simple)
  closeRet: number;            // gross return if closes at expCloseDate
  breakRet: number;            // gross return if breaks today (to breakPx)
  consideration: Consideration;
};

export function stockSwapTerms(d: Deal): Consideration {
  const S = num(d.pxAcq, 0);
  // Apply collar if present (very simplified: piecewise const)
  let er = num(d.exchangeRatio, 0);
  if (d.collar && isFiniteNum(S)) {
    const L = num(d.collar.lowerPx, -Infinity);
    const U = num(d.collar.upperPx, Infinity);
    if (S <= L && isFiniteNum(d.collar.fixedAtLower)) er = d.collar.fixedAtLower!;
    else if (S >= U && isFiniteNum(d.collar.fixedAtUpper)) er = d.collar.fixedAtUpper!;
  }
  const cash = num(d.considerationCash, 0);
  const stock = er > 0 ? er : 0;
  const value = cash + stock * S;
  return { cashPerShare: cash, stockPerShare: stock, valuePerShare: value };
}

export function dealMetrics(d: Deal, asOf?: number): DealMetrics {
  const now = asOf ?? Date.now();
  const Tyears = Math.max(1 / 365, yearFrac(now, num(d.expCloseDate, now)));
  const cons = stockSwapTerms(d);
  const pxT = num(d.pxTgt);
  const pxA = num(d.pxAcq, 0);
  const spread = pxT > 0 ? (cons.valuePerShare - pxT) / pxT : 0;

  const breakPx = num(d.breakPx, Math.max(0, pxT * 0.8));
  const breakRet = pxT > 0 ? (breakPx - pxT) / pxT : 0;

  // Closing payoff (ignores borrow if pure cash)
  const closeRetGross = pxT > 0 ? (cons.valuePerShare - pxT) / pxT : 0;

  // Borrow/div carry for stock leg while position is on:
  // If stock leg: arb is long TGT, short ER shares of ACQ ⇒ pay borrow + dividends on short.
  const er = cons.stockPerShare;
  const borrowY = er > 0
    ? Math.max(0, num(d.borrowFeeAcq, 0)) + Math.max(0, num(d.divYieldAcq, 0))
    : 0;

  // Translate carry to per-target equity return per year (approx): carry on short notional ER*pxA vs long pxT.
  const carryOnEquityY = (pxT > 0 && er > 0)
    ? (er * pxA / pxT) * borrowY
    : 0;

  const closeRetNet = closeRetGross - carryOnEquityY * Tyears;

  const p = clamp(num(d.pClose, 0.7), 0, 1);
  const expReturn = p * closeRetNet + (1 - p) * breakRet;
  const expIRR = Tyears > 0 ? expReturn / Tyears : 0;

  return {
    id: d.id,
    type: d.type,
    timeToCloseY: Tyears,
    spread,
    downMove: breakRet,
    borrowCarryY: carryOnEquityY,
    expReturn,
    expIRR,
    closeRet: closeRetNet,
    breakRet,
    consideration: cons,
  };
}

export function probFromSpread(d: Deal, asOf?: number) {
  // Solve for p from: E[ret] = p*(close - carry*T) + (1-p)*break = market spread (observed)
  const m = dealMetrics(d, asOf);
  const S = m.spread;
  const C = m.closeRet; // already net of carry
  const B = m.breakRet;
  const den = C - B;
  const p = den !== 0 ? (S - B) / den : 0.5;
  return clamp(p, 0, 1);
}

/* ------------------------------- Sizing --------------------------------- */

export function kellySize(input: { edge: number; vol: number }, frac = 0.5) {
  // Edge = expected return per unit risk (or per period); vol = stdev of outcome per period.
  // Kelly f* ≈ edge / vol^2. We default to fractional Kelly (e.g., 0.5).
  const e = num(input.edge), v = Math.max(1e-9, num(input.vol));
  const full = e / (v * v);
  return clamp(full * clamp(frac, 0, 1), -2, 2); // cap for sanity
}

export type BasketOpts = {
  maxGross?: number;        // sum(|w|) cap
  maxName?: number;         // per-deal cap
  volMap?: Record<string, number>; // deal id -> vol proxy (annualized)
  kellyFrac?: number;       // 0..1
  floorP?: number;          // ignore deals below this success prob
};

export function basketPlan(deals: Deal[], opts: BasketOpts = {}) {
  const grossCap = Math.max(0.1, num(opts.maxGross, 1));
  const perCap = Math.max(0.01, num(opts.maxName, 0.15));
  const kf = clamp(num(opts.kellyFrac, 0.5), 0, 1);
  const floorP = clamp(num(opts.floorP, 0.5), 0, 1);

  const rows = deals.map(d => {
    const m = dealMetrics(d);
    const p = num(d.pClose, probFromSpread(d));
    const vol = Math.max(0.05, (opts.volMap && opts.volMap[d.id]) || approxDealVol(m));
    const f = kellySize({ edge: m.expIRR, vol }, kf); // use IRR as "edge" proxy
    return { id: d.id, ticker: d.tickerTgt, p, irr: m.expIRR, vol, raw: f };
  });

  // Drop low quality
  const keep = rows.filter(r => r.p >= floorP && isFiniteNum(r.irr) && r.irr > 0);
  if (keep.length === 0) return { weights: {} as Record<string, number>, detail: [] as any[] };

  // Normalize abs weights
  let sumAbs = 0;
  const w0: Record<string, number> = {};
  for (let i = 0; i < keep.length; i++) { sumAbs += Math.abs(keep[i].raw); }
  for (let i = 0; i < keep.length; i++) {
    const w = sumAbs > 0 ? keep[i].raw / sumAbs : 0;
    w0[keep[i].id] = clamp(w, -perCap, perCap);
  }

  // Renormalize to grossCap
  let gross = 0;
  for (const k in w0) gross += Math.abs(w0[k]);
  const scale = gross > 0 ? Math.min(1, grossCap / gross) : 0;
  const weights: Record<string, number> = {};
  for (const k in w0) weights[k] = round6(w0[k] * scale);

  return { weights, detail: keep.map(r => ({ ...r, w: weights[r.id] })) };
}

/* ------------------------------ Simulation ------------------------------ */

export type SimPathOpts = {
  n?: number;                    // paths
  asOf?: number;
  drawClose?: (d: Deal) => boolean;  // RNG for success
};

export function simulate(deals: Deal[], opts: SimPathOpts = {}) {
  const N = Math.max(1, (opts.n ?? 1000) | 0);
  const res: { id: string; mean: number; pLoss: number; paths: number[] }[] = [];

  for (let i = 0; i < deals.length; i++) {
    const d = deals[i];
    const m = dealMetrics(d, opts.asOf);
    const p = clamp(num(d.pClose, probFromSpread(d)), 0, 1);
    const paths: number[] = [];
    let loss = 0, sum = 0;
    for (let k = 0; k < N; k++) {
      const success = opts.drawClose ? opts.drawClose(d) : (rand() < p);
      const r = success ? m.closeRet : m.breakRet;
      paths.push(r);
      if (r < 0) loss++;
      sum += r;
    }
    res.push({ id: d.id, mean: sum / N, pLoss: loss / N, paths });
  }
  return res;
}

/* ------------------------------- Backtest -------------------------------- */

export type DailyPx = {
  ts: number;
  pxT: number;
  pxA?: number;
};

export type DealTape = {
  id: string;
  tickerTgt: string;
  tickerAcq?: string;
  series: DailyPx[];      // chronological
  expCloseDate?: number;
  breakPx?: number;
  exchangeRatio?: number;
  considerationCash?: number;
  borrowFeeAcq?: number;
  divYieldAcq?: number;
  pClose?: number;
};

export type BacktestOpts = {
  startEquity?: number;
  kellyFrac?: number;
  maxGross?: number;
  maxName?: number;
};

export function backtest(tape: DealTape[], opts: BacktestOpts = {}) {
  // Very simple: treat each day as reprice of payoffs; weights set from basketPlan using last obs.
  const start = num(opts.startEquity, 1);
  let equity = start;

  const pointers = tape.map(_ => 0);
  const curve: { ts: number; equity: number }[] = [];

  // Build current deal states
  function snapshotDeals(ts: number): Deal[] {
    const deals: Deal[] = [];
    for (let i = 0; i < tape.length; i++) {
      const rec = tape[i];
      const idx = pointers[i];
      const row = rec.series[idx];
      if (!row) continue;
      deals.push({
        id: rec.id,
        tickerTgt: rec.tickerTgt,
        tickerAcq: rec.tickerAcq,
        type: (num(rec.exchangeRatio, 0) > 0 && num(rec.considerationCash, 0) > 0) ? "MIX" :
              (num(rec.exchangeRatio, 0) > 0 ? "STOCK" : "CASH"),
        expCloseDate: rec.expCloseDate,
        pxTgt: row.pxT,
        pxAcq: row.pxA,
        breakPx: rec.breakPx,
        exchangeRatio: rec.exchangeRatio,
        considerationCash: rec.considerationCash,
        borrowFeeAcq: rec.borrowFeeAcq,
        divYieldAcq: rec.divYieldAcq,
        pClose: rec.pClose,
      });
    }
    return deals;
  }

  // Iterate through timeline day by day (advance each series synchronously)
  let ts = minTs(tape);
  const endTs = maxTs(tape);

  while (ts <= endTs) {
    // advance pointers to ts (or nearest <= ts)
    for (let i = 0; i < tape.length; i++) {
      const arr = tape[i].series;
      while (pointers[i] + 1 < arr.length && arr[pointers[i] + 1].ts <= ts) pointers[i]++;
    }

    // rebalance daily using current snapshot
    const dealsNow = snapshotDeals(ts);
    const plan = basketPlan(dealsNow, {
      kellyFrac: num(opts.kellyFrac, 0.5),
      maxGross: num(opts.maxGross, 1),
      maxName: num(opts.maxName, 0.2),
    });

    // One-day equity change: sum_i w_i * daily ret of deal i proxied by change in theoretical value
    let dayRet = 0;
    for (let i = 0; i < dealsNow.length; i++) {
      const d = dealsNow[i];
      const id = d.id;
      const w = plan.weights[id] || 0;
      const idx = pointers[i];
      const s = tape[i].series;
      if (idx === 0) continue;
      const prev = s[idx - 1], cur = s[idx];
      const prevTheo = stockSwapTerms({ ...d, pxAcq: prev.pxA }).valuePerShare;
      const curTheo = stockSwapTerms({ ...d, pxAcq: cur.pxA }).valuePerShare;
      const prevRet = prev.pxT > 0 ? (prevTheo - prev.pxT) / prev.pxT : 0;
      const curRet  = cur.pxT > 0 ? (curTheo - cur.pxT) / cur.pxT : 0;
      const dSpread = curRet - prevRet; // change in spread as proxy for P&L
      dayRet += w * dSpread;
    }

    equity *= (1 + dayRet);
    curve.push({ ts, equity: round6(equity) });

    // next day
    ts += 86400000;
  }

  return { equity: curve, finalRet: curve.length ? curve[curve.length - 1].equity / start - 1 : 0 };
}

/* -------------------------------- Helpers -------------------------------- */

function approxDealVol(m: DealMetrics) {
  // crude: vol proxy from downside asymmetry + time: use |downMove| / sqrt(T)
  const T = Math.max(1 / 365, m.timeToCloseY);
  return Math.max(0.1, Math.min(1, Math.abs(m.downMove) / Math.sqrt(T)));
}

function yearFrac(t0: number, t1: number) {
  const days = Math.max(0, (t1 - t0) / 86400000);
  return days / 365;
}

function clamp(x: number, lo: number, hi: number) { return Math.min(hi, Math.max(lo, x)); }
function num(v: any, d = 0) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function isFiniteNum(v: any) { const n = Number(v); return Number.isFinite(n); }
function round6(n: number) { return Math.round(n * 1e6) / 1e6; }
function rand() { let x = Math.sin(Date.now() + Math.random() * 1e6) * 1e9; return Math.abs(x % 1); }

function minTs(tape: DealTape[]) {
  let m = Infinity;
  for (let i = 0; i < tape.length; i++) { if (tape[i].series.length) m = Math.min(m, tape[i].series[0].ts); }
  return m === Infinity ? Date.now() : m;
}
function maxTs(tape: DealTape[]) {
  let m = -Infinity;
  for (let i = 0; i < tape.length; i++) { const arr = tape[i].series; if (arr.length) m = Math.max(m, arr[arr.length - 1].ts); }
  return m === -Infinity ? Date.now() : m;
}

/* --------------------------------- Examples -------------------------------- */
// const d: Deal = {
//   id: "ADBE/Figma",
//   tickerTgt: "FIGMA",
//   tickerAcq: "ADBE",
//   type: "STOCK",
//   pxTgt: 90, pxAcq: 500,
//   exchangeRatio: 0.26, considerationCash: 0,
//   pClose: 0.65, breakPx: 40, expCloseDate: Date.now() + 180*86400000,
//   borrowFeeAcq: 0.02, divYieldAcq: 0.005
// };
// console.log(dealMetrics(d), probFromSpread(d));
// console.log(basketPlan([d], { kellyFrac: 0.5 }));