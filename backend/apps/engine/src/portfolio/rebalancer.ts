// portfolio/rebalancer.ts
// Pure, import-free portfolio rebalancer that produces a trade plan
// to move from current positions to target weights with practical
// constraints (bands, lot size, min notional, cash buffer, turnover cap).
//
// Usage:
//   const rb = createRebalancer();
//   const plan = rb.compute({
//     equity: 1_000_000,
//     cash: 50_000,
//     prices: { AAPL: 190, MSFT: 430, NVDA: 1100 },
//     positions: { AAPL: 800, MSFT: 300 }, // shares (qty). Missing = 0
//     targets: { AAPL: 0.30, MSFT: 0.30, NVDA: 0.40 },
//     opts: { bandPct: 0.005, lotSize: { NVDA: 1 }, minNotional: 5000, cashBufferPct: 0.02 }
//   });
//   console.log(plan.orders, plan.summary);
//
// Notes:
// - Weights are target portfolio weights that should sum ≈ 1 (not required; we normalize).
// - Prices must be provided for all symbols that appear in positions or targets.
// - Quantities are rounded to lot sizes; min notionals and turnover caps are applied.
// - Fees/slippage can be estimated for reporting (no external effects).
//
// Outputs:
//   {
//     orders: Array<{symbol, side, qty, pxRef, notional, deltaW, reason}>,
//     summary: { equity, cashBefore, cashAfter, estFees, estSlippage, grossTurnover, netTurnover, buys, sells },
//     diffs:   Array<{symbol, wCur, wTgt, wAfter, $cur, $tgt, $after, delta$}}
//   }

export type RebalanceInput = {
  equity: number;                            // total account equity (cash + positions MV)
  cash: number;                              // available cash before trades
  prices: Record<string, number>;            // symbol -> last price
  positions: Record<string, number>;         // symbol -> current qty (shares/contracts)
  targets: Record<string, number>;           // symbol -> desired weight (will be normalized)
  opts?: Partial<RebalanceOptions>;
};

export type RebalanceOptions = {
  bandPct: number;                           // do nothing if |wCur - wTgt| < band (default 0)
  lotSize: Record<string, number>;           // min qty increments by symbol (default 1)
  minNotional: number;                       // drop orders below this notional (default 0)
  maxTrades: number;                          // cap number of orders (default 1000)
  maxTurnoverPct: number;                    // cap gross turnover as % equity (0=unlimited)
  cashBufferPct: number;                     // keep this % of equity in cash (default 0)
  feeBps: number;                            // per-side fee estimate in bps (basis points)
  slipBps: number;                           // per-side slippage estimate in bps
  roundMode: "floor" | "round" | "ceil";     // qty rounding before lotSize snap (default round)
};

export type Order = {
  symbol: string;
  side: "BUY" | "SELL";
  qty: number;          // integer multiple of lot size
  pxRef: number;        // reference price used for sizing
  notional: number;     // qty * pxRef (signed in summary; positive here)
  deltaW: number;       // target weight change contributed by this order
  reason: "rebalance" | "cash-cap" | "turnover-cap" | "threshold";
};

export type RebalancePlan = {
  orders: Order[];
  summary: {
    equity: number;
    cashBefore: number;
    cashAfter: number;
    estFees: number;
    estSlippage: number;
    grossTurnover: number; // Σ |order notional|
    netTurnover: number;   // buys + sells (same as grossTurnover)
    buys: number;
    sells: number;
  };
  diffs: Array<{
    symbol: string;
    wCur: number;
    wTgt: number;
    wAfter: number;
    $cur: number;
    $tgt: number;
    $after: number;
    delta$: number;
  }>;
};

export function createRebalancer() {
  function compute(input: RebalanceInput): RebalancePlan {
    const equity = num(input.equity);
    const cashBefore = num(input.cash);
    const prices = sanitizePrices(input.prices);
    const positions = sanitizePositions(input.positions);
    const targetsRaw = sanitizeTargets(input.targets);
    const o = normalizeOptions(input.opts);

    // Normalize target weights (ignore negatives/NaNs)
    const targets = normalizeWeights(targetsRaw);

    // Build universe = union of symbols from positions and targets with known prices
    const symbols = Array.from(new Set([...Object.keys(positions), ...Object.keys(targets)]))
      .filter(s => isFinite(prices[s]));

    // Current dollar + weights
    const $cur: Record<string, number> = {};
    for (const s of symbols) $cur[s] = (positions[s] || 0) * prices[s];

    // We may want to reserve a cash buffer
    const cashReserve = equity * o.cashBufferPct;
    const investableEquity = Math.max(0, equity - cashReserve);

    // Desired dollars per symbol on investable equity
    const $tgt: Record<string, number> = {};
    for (const s of symbols) $tgt[s] = targets[s] * investableEquity;

    // Current weights relative to equity
    const wCur: Record<string, number> = {};
    for (const s of symbols) wCur[s] = $cur[s] / Math.max(1e-9, equity);

    // Thresholding: if within band, set target = current (no trade)
    for (const s of symbols) {
      if (Math.abs((targets[s] || 0) - (wCur[s] || 0)) < o.bandPct) {
        $tgt[s] = $cur[s];
      }
    }

    // Dollar deltas and raw qty deltas
    const delta$: Record<string, number> = {};
    const deltaQtyRaw: Record<string, number> = {};
    for (const s of symbols) {
      delta$[s] = $tgt[s] - $cur[s];
      deltaQtyRaw[s] = delta$[s] / prices[s]; // can be fractional
    }

    // Snap quantities to lot sizes with chosen rounding mode
    const ordersDraft: Order[] = [];
    for (const s of symbols) {
      const lot = Math.max(1, Math.floor(o.lotSize[s] || 1));
      let qty = snapQty(deltaQtyRaw[s], lot, o.roundMode);
      if (qty === 0) continue;

      const notional = Math.abs(qty) * prices[s];
      if (notional < o.minNotional) continue;

      const side: "BUY" | "SELL" = qty > 0 ? "BUY" : "SELL";
      const deltaW = (qty * prices[s]) / Math.max(1e-9, equity);

      ordersDraft.push({
        symbol: s,
        side,
        qty: Math.abs(qty),
        pxRef: prices[s],
        notional: Math.abs(qty) * prices[s],
        deltaW,
        reason: "rebalance",
      });
    }

    // Sort by largest absolute weight gap first (greedy execution)
    ordersDraft.sort((a, b) => Math.abs(b.deltaW) - Math.abs(a.deltaW));

    // Apply turnover cap if any
    const grossLimit = o.maxTurnoverPct > 0 ? o.maxTurnoverPct * equity : Infinity;
    let grossUsed = 0;
    const ordersTurnover: Order[] = [];
    for (const ord of ordersDraft) {
      if (ordersTurnover.length >= o.maxTrades) break;
      if (grossUsed + ord.notional <= grossLimit + 1e-9) {
        ordersTurnover.push(ord);
        grossUsed += ord.notional;
      } else {
        // Try a partial qty fit, scaled to remaining turnover budget
        const rem = Math.max(0, grossLimit - grossUsed);
        const lot = Math.max(1, Math.floor(o.lotSize[ord.symbol] || 1));
        let q = Math.floor(rem / ord.pxRef / lot) * lot;
        if (q > 0) {
          ordersTurnover.push({
            ...ord,
            qty: q,
            notional: q * ord.pxRef,
            deltaW: (q * ord.pxRef) / Math.max(1e-9, equity),
            reason: "turnover-cap",
          });
          grossUsed += q * ord.pxRef;
        }
        break; // turnover budget exhausted
      }
    }

    // Apply cash constraint: ensure we don't overspend beyond available cash + sells
    const buys = ordersTurnover.filter(o => o.side === "BUY");
    const sells = ordersTurnover.filter(o => o.side === "SELL");
    const buyNotional = sum(buys.map(o => o.notional));
    const sellNotional = sum(sells.map(o => o.notional));
    const rawCashAfter = cashBefore - buyNotional + sellNotional;

    let orders: Order[] = ordersTurnover;
    let cashAfter = rawCashAfter;

    if (rawCashAfter < 0) {
      // Scale down buys proportionally to fit cash (keeping minNotional & lot sizes)
      const need = -rawCashAfter;
      const scale = Math.max(0, 1 - need / Math.max(1e-9, buyNotional));
      const resized: Order[] = [];

      for (const b of buys) {
        const lot = Math.max(1, Math.floor(o.lotSize[b.symbol] || 1));
        const qScaled = Math.floor(b.qty * scale / lot) * lot;
        if (qScaled <= 0) continue;
        const notional = qScaled * b.pxRef;
        if (notional < o.minNotional) continue;
        resized.push({
          ...b,
          qty: qScaled,
          notional,
          deltaW: (qScaled * b.pxRef) / equity,
          reason: "cash-cap",
        });
      }

      orders = [...resized, ...sells];
      cashAfter = cashBefore - sum(resized.map(o => o.notional)) + sellNotional;
    }

    // Estimate fees & slippage (purely informational)
    const fee = (o.feeBps / 10_000) * sum(orders.map(x => x.notional));
    const slp = (o.slipBps / 10_000) * sum(orders.map(x => x.notional));

    // Compute post-trade weights snapshot (approx, using pxRef and order qty deltas)
    const $after: Record<string, number> = {};
    for (const s of symbols) $after[s] = $cur[s];
    for (const ord of orders) {
      const signed = (ord.side === "BUY" ? 1 : -1) * ord.qty * ord.pxRef;
      $after[ord.symbol] = ($after[ord.symbol] || 0) + signed;
    }
    // Equity approx unchanged (fees/slp reduce cash but ignore here for weights view)
    const wAfter: Record<string, number> = {};
    for (const s of symbols) wAfter[s] = $after[s] / Math.max(1e-9, equity);

    // Diffs for report
    const diffs = symbols.map(s => ({
      symbol: s,
      wCur: round4(wCur[s] || 0),
      wTgt: round4(targets[s] || 0),
      wAfter: round4(wAfter[s] || 0),
      $cur: round2($cur[s] || 0),
      $tgt: round2($tgt[s] || 0),
      $after: round2($after[s] || 0),
      delta$: round2(($tgt[s] || 0) - ($cur[s] || 0)),
    }));

    const summary = {
      equity: round2(equity),
      cashBefore: round2(cashBefore),
      cashAfter: round2(cashAfter - fee - slp), // subtract est. costs
      estFees: round2(fee),
      estSlippage: round2(slp),
      grossTurnover: round2(sum(orders.map(o => o.notional))),
      netTurnover: round2(sum(orders.map(o => o.notional))), // same here (buys+sells absolute)
      buys: sum(orders.map(o => (o.side === "BUY" ? o.notional : 0))),
      sells: sum(orders.map(o => (o.side === "SELL" ? o.notional : 0))),
    };

    // Final order cap (count)
    if (orders.length > o.maxTrades) {
      orders = orders.slice(0, o.maxTrades);
    }

    return {
      orders,
      summary,
      diffs,
    };
  }

  return { compute };
}

/* --------------------------------- Helpers -------------------------------- */

function sanitizePrices(p: Record<string, number>) {
  const out: Record<string, number> = {};
  for (const k in p) {
    const v = Number(p[k]);
    if (Number.isFinite(v) && v > 0) out[k] = v;
  }
  return out;
}

function sanitizePositions(p: Record<string, number>) {
  const out: Record<string, number> = {};
  for (const k in p) {
    const v = Number(p[k]);
    if (Number.isFinite(v)) out[k] = v;
  }
  return out;
}

function sanitizeTargets(t: Record<string, number>) {
  const out: Record<string, number> = {};
  for (const k in t) {
    const v = Number(t[k]);
    if (Number.isFinite(v)) out[k] = v;
  }
  return out;
}

function normalizeWeights(w: Record<string, number>) {
  const out: Record<string, number> = {};
  let sumW = 0;
  for (const k in w) {
    const v = Number(w[k]);
    if (Number.isFinite(v) && v > 0) { out[k] = v; sumW += v; }
  }
  if (sumW <= 0) return {};
  for (const k in out) out[k] = out[k] / sumW;
  return out;
}

function snapQty(q: number, lot: number, mode: "floor" | "round" | "ceil") {
  if (!Number.isFinite(q) || q === 0) return 0;
  const s = Math.sign(q);
  const abs = Math.abs(q);
  let base: number;
  switch (mode) {
    case "floor": base = Math.floor(abs / lot) * lot; break;
    case "ceil":  base = Math.ceil(abs / lot) * lot;  break;
    default:      base = Math.round(abs / lot) * lot; break;
  }
  return s * base;
}

function normalizeOptions(o?: Partial<RebalanceOptions>): RebalanceOptions {
  const d: RebalanceOptions = {
    bandPct: 0,
    lotSize: {},
    minNotional: 0,
    maxTrades: 1000,
    maxTurnoverPct: 0,
    cashBufferPct: 0,
    feeBps: 0,
    slipBps: 0,
    roundMode: "round",
  };
  const x = o || {};
  d.bandPct = num(x.bandPct, d.bandPct);
  d.lotSize = x.lotSize || d.lotSize;
  d.minNotional = num(x.minNotional, d.minNotional);
  d.maxTrades = int(x.maxTrades, d.maxTrades);
  d.maxTurnoverPct = num(x.maxTurnoverPct, d.maxTurnoverPct);
  d.cashBufferPct = num(x.cashBufferPct, d.cashBufferPct);
  d.feeBps = num(x.feeBps, d.feeBps);
  d.slipBps = num(x.slipBps, d.slipBps);
  d.roundMode = (x.roundMode as any) || d.roundMode;
  return d;
}

function sum(arr: number[]) { return arr.reduce((a, b) => a + b, 0); }
function num(v: any, d = 0) { const n = Number(v); return Number.isFinite(n) ? n : d; }
function int(v: any, d = 0) { const n = Number(v); return Number.isFinite(n) ? (n | 0) : d; }
function round2(n: number) { return Math.round(n * 100) / 100; }
function round4(n: number) { return Math.round(n * 10000) / 10000; }