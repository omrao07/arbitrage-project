// portfolio/pnl.ts
// Pure, import-free PnL & positions engine with FIFO lot accounting.
//
// Features
// - FIFO lots per symbol (avg cost derived from remaining lots)
// - Realized & unrealized PnL, fees, exposure, equity
// - Cash balance tracking (buy decreases cash, sell increases cash)
// - Mark-to-market with price ticks
// - Equity curve time-series + returns + drawdown stats
// - Snapshots, dump/load for persistence
//
// Usage:
//   const pnl = createPnL({ base: "USD", initialCash: 100000 });
//   pnl.trade({ symbol: "AAPL", qty: 10, price: 190, fee: 1 });
//   pnl.price({ symbol: "AAPL", price: 195 });
//   console.log(pnl.position("AAPL"));
//   console.log(pnl.totals());

export type PnLOptions = {
  base?: string;          // currency code (display only)
  initialCash?: number;   // starting cash
};

export type Trade = {
  symbol: string;
  qty: number;            // +buy, -sell
  price: number;          // fill price
  fee?: number;           // commissions/fees (always subtract from cash)
  ts?: number;            // epoch ms
};

export type PriceTick = {
  symbol: string;
  price: number;
  ts?: number;
};

export type Lot = {
  qty: number;            // remaining quantity in this lot (>0 for long, <0 for short)
  price: number;          // lot entry price
  ts: number;             // open time
};

export type Position = {
  symbol: string;
  qty: number;
  avgCost: number;        // 0 if flat
  price: number;          // last mark
  mktValue: number;       // qty * price
  unrealized: number;     // (price - avgCost)*qty (sign respects position)
};

export type Totals = {
  base: string;
  cash: number;
  equity: number;         // cash + Σ position market values
  exposure: number;       // Σ |qty*price|
  realized: number;       // cumulative realized PnL
  unrealized: number;     // Σ unrealized across symbols
  gross: number;          // realized + unrealized
  fees: number;           // cumulative fees paid
  ts: number;
};

export function createPnL(opts: PnLOptions = {}) {
  const state = {
    base: opts.base || "USD",
    cash: num(opts.initialCash, 0),
    fees: 0,
    realized: 0,
    prices: new Map<string, { price: number; ts: number }>(),
    lots: new Map<string, Lot[]>(),     // FIFO list per symbol
    // time-series
    curve: [] as { ts: number; equity: number }[],
    rets: [] as { ts: number; ret: number }[], // simple step returns
    dd: { peak: -Infinity, maxDD: 0, curDD: 0 }, // drawdown stats
  };

  /* ------------------------------- Trades -------------------------------- */

  function trade(t: Trade) {
    const ts = t.ts ?? now();
    const fee = num(t.fee, 0);
    const price = num(t.price, 0);
    const qty = num(t.qty, 0);

    if (!t.symbol || !qty || !isFinite(price)) return;

    // Apply fee immediately
    state.cash -= fee;
    state.fees += fee;

    // Update cash for trade notional: buy -> pay cash; sell -> receive cash
    state.cash -= qty * price;

    // Put a mark (if newer) so equity reflects this trade price
    setPrice({ symbol: t.symbol, price, ts });

    // Ensure lot list exists
    const arr = state.lots.get(t.symbol) || [];
    if (qty > 0) {
      // BUY: append a long lot
      arr.push({ qty, price, ts });
    } else if (qty < 0) {
      // SELL: consume long lots (or grow short if over-sell)
      let sellQty = -qty; // positive amount to sell
      // Consume existing long lots first (FIFO)
      while (sellQty > 0 && arr.length > 0 && arr[0].qty > 0) {
        const lot = arr[0];
        const use = Math.min(lot.qty, sellQty);
        // Realize PnL on portion closed
        const pnl = (price - lot.price) * use;
        state.realized += pnl;
        lot.qty -= use;
        sellQty -= use;
        if (lot.qty === 0) arr.shift();
      }
      // If still selling and no/insufficient long lots, we are creating/adding to a SHORT
      if (sellQty > 0) {
        // Represent short lots with negative qty at sell price
        arr.unshift({ qty: -sellQty, price, ts }); // new short opened at current sell price
        sellQty = 0;
      }
    }
    // Clean zero lots
    for (let i = arr.length - 1; i >= 0; i--) {
      if (arr[i].qty === 0) arr.splice(i, 1);
    }
    state.lots.set(t.symbol, arr);

    // Update equity curve after trade
    markEquity(ts);
  }

  /* -------------------------- Prices / Marking --------------------------- */

  function price(p: PriceTick) {
    setPrice(p);
    markEquity(p.ts ?? now());
  }

  function setPrice(p: PriceTick) {
    const ts = p.ts ?? now();
    const prev = state.prices.get(p.symbol);
    if (!prev || ts >= prev.ts) {
      state.prices.set(p.symbol, { price: num(p.price, 0), ts });
    }
  }

  /* ------------------------------ Queries -------------------------------- */

  function position(symbol: string): Position {
    const lots = state.lots.get(symbol) || [];
    const qty = lots.reduce((a, b) => a + b.qty, 0);
    const price = state.prices.get(symbol)?.price ?? 0;

    let avg = 0;
    let rem = Math.abs(qty);
    if (qty > 0) {
      // long: average remaining long lots
      const longs = lots.filter(l => l.qty > 0);
      const tot = longs.reduce((s, l) => s + l.qty, 0);
      if (tot > 0) {
        let cost = 0, left = rem;
        for (const l of longs) {
          const used = Math.min(l.qty, left);
          cost += used * l.price;
          left -= used;
          if (left <= 0) break;
        }
        avg = tot > 0 ? cost / Math.max(1, Math.abs(qty)) : 0;
      }
    } else if (qty < 0) {
      // short: average remaining short lots (qty negative)
      const shorts = lots.filter(l => l.qty < 0);
      const tot = shorts.reduce((s, l) => s + (-l.qty), 0);
      if (tot > 0) {
        let credit = 0, left = Math.abs(qty);
        for (const l of shorts) {
          const used = Math.min(-l.qty, left);
          credit += used * l.price;
          left -= used;
          if (left <= 0) break;
        }
        avg = tot > 0 ? credit / Math.max(1, Math.abs(qty)) : 0;
      }
    }

    const mktValue = qty * price;
    const unreal =
      qty >= 0
        ? (price - avg) * qty
        : (avg - price) * Math.abs(qty); // for shorts, gain if price < avg

    return {
      symbol,
      qty,
      avgCost: round(avg),
      price: round(price),
      mktValue: round(mktValue),
      unrealized: round(unreal),
    };
  }

  function positions(): Position[] {
    const out: Position[] = [];
    for (const sym of uniqueSyms()) {
      if (!state.lots.has(sym)) {
        out.push({
          symbol: sym,
          qty: 0,
          avgCost: 0,
          price: round(state.prices.get(sym)!.price),
          mktValue: 0,
          unrealized: 0,
        });
      }
    }
    // stable sort by symbol
    out.sort((a, b) => (a.symbol < b.symbol ? -1 : a.symbol > b.symbol ? 1 : 0));
    return out;
  }

  function totals(): Totals {
    const ts = now();
    let exposure = 0;
    let unreal = 0;
    for (const sym of uniqueSyms()) {
      const p = position(sym);
      exposure += Math.abs(p.mktValue);
      unreal += p.unrealized;
    }
    const equity = state.cash + sumMktValue();
    return {
      base: state.base,
      cash: round(state.cash),
      equity: round(equity),
      exposure: round(exposure),
      realized: round(state.realized),
      unrealized: round(unreal),
      gross: round(state.realized + unreal),
      fees: round(state.fees),
      ts,
    };
  }

  function snapshot() {
    return {
      base: state.base,
      cash: state.cash,
      fees: state.fees,
      realized: state.realized,
      positions: positions(),
      totals: totals(),
      equityCurve: state.curve.slice(-500), // recent
      returns: state.rets.slice(-500),
      drawdown: { ...state.dd },
    };
  }

  /* --------------------------- Equity & Stats ---------------------------- */

  function sumMktValue(): number {
    let mv = 0;
    for (const sym of uniqueSyms()) {
      const pos = position(sym);
      mv += pos.mktValue;
    }
    return mv;
  }

  function markEquity(ts: number) {
    const eq = state.cash + sumMktValue();
    const last = state.curve[state.curve.length - 1];
    state.curve.push({ ts, equity: eq });

    if (last) {
      const ret = safeDiv(eq - last.equity, Math.max(1e-9, Math.abs(last.equity)));
      state.rets.push({ ts, ret });
    } else {
      state.rets.push({ ts, ret: 0 });
    }

    // drawdown stats (relative to equity peaks)
    if (eq > state.dd.peak) state.dd.peak = eq;
    const dd = state.dd.peak > 0 ? (state.dd.peak - eq) / state.dd.peak : 0;
    state.dd.curDD = Math.max(0, dd);
    if (state.dd.curDD > state.dd.maxDD) state.dd.maxDD = state.dd.curDD;
  }

  function equityCurve() {
    return state.curve.slice();
  }

  function returns() {
    return state.rets.slice();
  }

  function drawdown() {
    return { ...state.dd };
  }

  /* --------------------------- Persistence ------------------------------ */

  function dump(): string {
    const obj: any = {
      base: state.base,
      cash: state.cash,
      fees: state.fees,
      realized: state.realized,
      prices: Array.from(state.prices.entries()),
      lots: Array.from(state.lots.entries()),
      curve: state.curve,
      rets: state.rets,
      dd: state.dd,
    };
    try { return JSON.stringify(obj); } catch { return "{}"; }
  }

  function load(json: string): boolean {
    try {
      const o = JSON.parse(json || "{}");
      state.base = String(o.base || "USD");
      state.cash = num(o.cash, 0);
      state.fees = num(o.fees, 0);
      state.realized = num(o.realized, 0);

      state.prices = new Map((o.prices || []).map((it: any) => [String(it[0]), { price: num(it[1]?.price, 0), ts: num(it[1]?.ts, now()) }]));
      state.lots = new Map((o.lots || []).map((it: any) => [String(it[0]), (Array.isArray(it[1]) ? it[1] : []).map((x: any) => ({
        qty: num(x.qty, 0), price: num(x.price, 0), ts: num(x.ts, now()),
      }))]));

      state.curve = Array.isArray(o.curve) ? o.curve : [];
      state.rets = Array.isArray(o.rets) ? o.rets : [];
      state.dd = o.dd && typeof o.dd === "object" ? {
        peak: num(o.dd.peak, -Infinity),
        maxDD: num(o.dd.maxDD, 0),
        curDD: num(o.dd.curDD, 0),
      } : { peak: -Infinity, maxDD: 0, curDD: 0 };

      return true;
    } catch {
      return false;
    }
  }

  /* -------------------------------- Utils -------------------------------- */

  function uniqueSyms(): string[] {
    const s = new Set<string>();
    
    return Array.from(s);
  }

  function now() { return Date.now(); }
  function num(v: any, d: number) { const n = Number(v); return Number.isFinite(n) ? n : d; }
  function round(n: number) { return Math.round(n * 1e6) / 1e6; }
  function safeDiv(a: number, b: number) { return b === 0 ? 0 : a / b; }

  /* -------------------------------- API ---------------------------------- */

  return {
    // events
    trade,
    price,
    // queries
    position,
    positions,
    totals,
    equityCurve,
    returns,
    drawdown,
    snapshot,
    // persistence
    dump,
    load,
  };
}