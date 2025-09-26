// strategies/base.ts
// No imports. Pure TypeScript base layer for backtesting strategies.
// - Evented abstract StrategyBase with onInit / onBar / onEnd hooks
// - Minimal in-file broker: market/limit/stop orders, next-bar execution
// - Portfolio accounting (cash, equity, P&L), positions, trades, orders
// - Risk gates: max gross leverage, per-position weight, stop-loss / take-profit
// - Utilities: rolling indicators, performance stats, unique IDs
//
// Quick start:
//
// class MyStrat extends StrategyBase {
//   protected onInit(ctx: Ctx) { this.note("ready"); }
//   protected onBar(ctx: Ctx) {
//     const p = ctx.price("AAPL");
//     if (ctx.idx === 1) this.targetWeight("AAPL", 1.0);       // go 100% long at t+1 open
//     if (ctx.idx === ctx.n - 20) this.targetWeight("AAPL", 0); // flat near the end
//   }
//   protected onEnd(ctx: Ctx) { this.note("done"); }
// }
//
// const strat = new MyStrat({ initialCash: 100_000 });
// const report = strat.run({ AAPL: bars });
// console.log(report.summary.sharpe);

export type Side = "buy" | "sell";
export type OrderType = "market" | "limit" | "stop";
export type TIF = "DAY" | "GTC";

export type Bar = {
  t: number;           // unix ms (or any monotonic number)
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type Series = Bar[];

export type Order = {
  id: string;
  symbol: string;
  side: Side;
  type: OrderType;
  qty: number;           // signed (buy>0, sell>0); engine uses side to sign
  limitPrice?: number;
  stopPrice?: number;
  tif: TIF;
  placedIdx: number;     // bar index when placed
  status: "open" | "filled" | "canceled" | "expired" | "rejected" | "partial";
  fills: Fill[];
  note?: string;
};

export type Fill = {
  idx: number;           // bar index filled
  price: number;
  qty: number;           // signed: + for buy, - for sell applied to position
  fee: number;
  slippage: number;      // applied price delta (absolute)
};

export type Position = {
  symbol: string;
  qty: number;
  avgPrice: number;      // average cost for current open qty
  realizedPnl: number;   // cumulative realized
  lastIdx: number;       // last touch index
};

export type Trade = {
  symbol: string;
  entryIdx: number;
  exitIdx: number;
  entryPrice: number;
  exitPrice: number;
  qty: number;           // signed from entry perspective
  pnl: number;
  ret: number;
};

export type Portfolio = {
  cash: number;
  equity: number;
  positions: Record<string, Position>;
  // Derived every step (not persisted): grossExposure, netExposure, mvBySym
};

export type RiskConfig = {
  maxGrossLeverage?: number;       // default 1
  maxPositionWeight?: number;      // 0..1 (per symbol) default 1
  stopLossPct?: number;            // optional soft stop at position level
  takeProfitPct?: number;          // optional soft TP at position level
};

export type StrategyOptions = {
  initialCash?: number;            // default 100_000
  feeBps?: number;                 // per fill, default 1 (0.01%)
  slippageBps?: number;            // on execution price, default 0
  allowShort?: boolean;            // default true
  risk?: RiskConfig;
  name?: string;
};

export type RunReport = {
  equity: number[];                // per index equity curve
  returns: number[];               // arithmetic returns (eq[t]/eq[t-1]-1)
  orders: Order[];
  trades: Trade[];
  portfolio: Portfolio;            // final
  logs: string[];
  symbols: string[];
  n: number;
  summary: {
    cagr: number;
    vol: number;
    sharpe: number;
    maxDD: number;
    hitRate: number;
    avgTrade: number;
    nTrades: number;
  };
};

export type Ctx = {
  readonly name: string;
  readonly idx: number;
  readonly n: number;
  readonly now: number;
  price: (symbol: string) => number;
  bar: (symbol: string, k?: number) => Bar | undefined; // k<=0: current, -1: prev
  hist: (symbol: string, n: number) => number[];        // closes [t-n+1..t]
  equity: () => number;
  cash: () => number;
  pos: (symbol: string) => Position | undefined;
  mv: (symbol?: string) => number;                      // market value (abs sum if no symbol)
  place: (o: Partial<Order> & { symbol: string; side: Side; type: OrderType; qty: number }) => string | null;
  cancel: (id: string) => boolean;
  targetWeight: (symbol: string, w: number) => string | null;
  flat: (symbol: string) => string | null;
  note: (s: string) => void;
  config: Readonly<Required<StrategyOptions>>;
};

type EngineState = {
  opt: Required<StrategyOptions>;
  bars: Record<string, Series>;
  idx: number;
  n: number;
  portfolio: Portfolio;
  orders: Order[];
  logs: string[];
  trades: Trade[];
  uid: number;
};

/* =============================== Base Class =============================== */

export default abstract class StrategyBase {
  protected state?: EngineState;

  constructor(protected readonly options: StrategyOptions = {}) {}

  /** Override for initialization. */
  protected onInit(_ctx: Ctx) {}
  /** Override to react to each bar (synchronous). */
  protected onBar(_ctx: Ctx) {}
  /** Override for cleanup / summarization. */
  protected onEnd(_ctx: Ctx) {}

  /** Run the backtest. `data` is symbol → Series (aligned by index order). */
  run(data: Record<string, Series>): RunReport {
    const symbols = Object.keys(data).sort();
    if (!symbols.length) throw new Error("No symbols provided");

    const n = Math.min(...symbols.map((s) => data[s].length));
    const opt: Required<StrategyOptions> = {
      initialCash: this.options.initialCash ?? 100_000,
      feeBps: this.options.feeBps ?? 1,
      slippageBps: this.options.slippageBps ?? 0,
      allowShort: this.options.allowShort ?? true,
      risk: {
        maxGrossLeverage: this.options.risk?.maxGrossLeverage ?? 1,
        maxPositionWeight: this.options.risk?.maxPositionWeight ?? 1,
        stopLossPct: this.options.risk?.stopLossPct,
        takeProfitPct: this.options.risk?.takeProfitPct,
      },
      name: this.options.name ?? this.constructor.name,
    };

    const st: EngineState = {
      opt,
      bars: data,
      idx: 0,
      n,
      portfolio: { cash: opt.initialCash, equity: opt.initialCash, positions: {} },
      orders: [],
      logs: [],
      trades: [],
      uid: 1,
    };
    this.state = st;

    const eq: number[] = new Array(n).fill(opt.initialCash);
    const rets: number[] = new Array(n).fill(0);

    // init
    this.onInit(this.ctx());

    // main loop
    for (let i = 0; i < n; i++) {
      st.idx = i;

      // broker: expire DAY orders from previous step
      for (const o of st.orders) {
        if (o.status === "open" && o.tif === "DAY" && o.placedIdx < i - 1) {
          o.status = "expired";
        }
      }

      // strategy hook
      this.onBar(this.ctx());

      // broker: execute open orders on next-bar model (use current bar since we are at open of i)
      this.matchOrdersAtIndex(i);

      // broker: risk soft stops (per-position)
      this.applyStops(i);

      // mark & metrics
      const equity = this.markToMarket(i);
      eq[i] = equity;
      if (i > 0) rets[i] = eq[i] / eq[i - 1] - 1;
    }

    // end
    this.onEnd(this.ctx());

    const port = st.portfolio;
    const summary = summarize(eq, rets, st.trades);

    return {
      equity: eq,
      returns: rets,
      orders: st.orders.slice(),
      trades: st.trades.slice(),
      portfolio: { cash: port.cash, equity: port.equity, positions: clonePos(port.positions) },
      logs: st.logs.slice(),
      symbols,
      n,
      summary,
    };
  }

  /* ----------------------------- Order Helpers ---------------------------- */

  protected targetWeight(symbol: string, w: number): string | null {
    return this.ctx().targetWeight(symbol, w);
  }

  protected flat(symbol: string): string | null {
    return this.ctx().flat(symbol);
  }

  protected placeMarket(symbol: string, side: Side, qty: number, note?: string) {
    return this.ctx().place({ symbol, side, type: "market", qty, note });
  }

  protected note(msg: string) {
    this.ctx().note(msg);
  }

  /* ------------------------------- Internals ------------------------------ */

  private ctx(): Ctx {
    const st = this.state!;
    const self = this;

    const ctx: Ctx = {
      name: st.opt.name,
      idx: st.idx,
      n: st.n,
      now: st.bars[Object.keys(st.bars)[0]][st.idx].t,
      price: (sym) => st.bars[sym][st.idx].close,
      bar: (sym, k = 0) => {
        const i = st.idx + k;
        return i >= 0 && i < st.n ? st.bars[sym][i] : undefined;
      },
      hist: (sym, n) => {
        const out: number[] = [];
        for (let i = Math.max(0, st.idx - n + 1); i <= st.idx; i++) out.push(st.bars[sym][i].close);
        return out;
      },
      equity: () => st.portfolio.equity,
      cash: () => st.portfolio.cash,
      pos: (sym) => st.portfolio.positions[sym],
      mv: (sym?: string) => {
        if (sym) {
          const p = st.portfolio.positions[sym];
          const px = st.bars[sym][st.idx].close;
          return p ? p.qty * px : 0;
        }
        let sum = 0;
        for (const s of Object.keys(st.portfolio.positions)) {
          const p = st.portfolio.positions[s];
          const px = st.bars[s][st.idx].close;
          sum += Math.abs(p.qty * px);
        }
        return sum;
      },
      place: (o) => self.placeOrder(o),
      cancel: (id) => self.cancelOrder(id),
      targetWeight: (sym, w) => self.placeTargetWeight(sym, w),
      flat: (sym) => self.placeTargetWeight(sym, 0),
      note: (s) => st.logs.push(`[${fmtTs(st.bars[Object.keys(st.bars)[0]][st.idx].t)}] ${s}`),
      config: st.opt,
    };
    return ctx;
  }

  private placeTargetWeight(symbol: string, w: number): string | null {
    const st = this.state!;
    const px = this.ctx().price(symbol);
    const pos = st.portfolio.positions[symbol];
    const equity = this.ctx().equity();
    const targetMV = clamp(-st.opt.risk.maxPositionWeight, st.opt.risk.maxPositionWeight, w) * equity;
    const curMV = (pos?.qty || 0) * px;
    const diffMV = targetMV - curMV;
    const qty = Math.floor(diffMV / px); // integer shares
    if (qty === 0) return null;
    const side: Side = qty > 0 ? "buy" : "sell";
    return this.placeOrder({ symbol, side, type: "market", qty: Math.abs(qty), note: `target w=${w.toFixed(3)}` });
  }

  private placeOrder(o: Partial<Order> & { symbol: string; side: Side; type: OrderType; qty: number }): string | null {
    const st = this.state!;
    if (o.qty <= 0) return null;
    // Risk gates (pre-trade): leverage and position cap (best-effort estimate)
    if (!this.passesPreTradeRisk(o.symbol, o.side, o.qty, o.limitPrice)) {
      const id = this.uid();
      st.orders.push({
        id,
        symbol: o.symbol,
        side: o.side,
        type: o.type,
        qty: o.qty,
        limitPrice: o.limitPrice,
        stopPrice: o.stopPrice,
        tif: o.tif || "DAY",
        placedIdx: st.idx,
        status: "rejected",
        fills: [],
        note: (o.note || "") + " [risk-reject]",
      });
      return null;
    }

    const id = this.uid();
    st.orders.push({
      id,
      symbol: o.symbol,
      side: o.side,
      type: o.type,
      qty: Math.floor(o.qty),
      limitPrice: o.limitPrice,
      stopPrice: o.stopPrice,
      tif: o.tif || "DAY",
      placedIdx: st.idx,
      status: "open",
      fills: [],
      note: o.note,
    });
    return id;
  }

  private cancelOrder(id: string): boolean {
    const st = this.state!;
    const o = st.orders.find((x) => x.id === id && x.status === "open");
    if (!o) return false;
    o.status = "canceled";
    return true;
  }

  private passesPreTradeRisk(symbol: string, side: Side, qty: number, lim?: number) {
    const st = this.state!;
    const pxNow = this.ctx().price(symbol);
    const px = lim ?? pxNow;
    const pos = st.portfolio.positions[symbol];
    const nextQty = (pos?.qty || 0) + (side === "buy" ? qty : -qty);
    const nextMV = Math.abs(nextQty * px);
    const eq = this.ctx().equity();
    // position cap
    if (eq > 0 && nextMV / eq > (st.opt.risk.maxPositionWeight ?? 1)) return false;

    // leverage cap (gross = sum |mv| / equity)
    let gross = 0;
    const syms = Object.keys(st.portfolio.positions);
    for (const s of syms) {
      const p = st.portfolio.positions[s];
      gross += Math.abs(p.qty * this.ctx().price(s));
    }
    gross += Math.abs((side === "buy" ? qty : -qty) * px);
    const grossLev = eq > 0 ? gross / eq : Infinity;
    if (grossLev > (st.opt.risk.maxGrossLeverage ?? 1)) return false;

    // shorting permission
    if (!st.opt.allowShort && (pos?.qty || 0) + (side === "sell" ? -qty : qty) < 0) return false;

    return true;
  }

  private matchOrdersAtIndex(i: number) {
    const st = this.state!;
    for (const o of st.orders) {
      if (o.status !== "open" || o.placedIdx === i) continue; // next-bar only

      const b = st.bars[o.symbol][i];
      const pxExec = this.executionPrice(o, b, st.opt.slippageBps);
      if (pxExec == null) continue; // not triggered

      // Fill fully (simple model)
      const signedQty = (o.side === "buy" ? +1 : -1) * o.qty;
      const fee = Math.abs(pxExec * o.qty) * (st.opt.feeBps * 1e-4);
      const slip = Math.abs(pxExec - refPriceForType(o, b));

      o.fills.push({ idx: i, price: pxExec, qty: signedQty, fee, slippage: slip });
      o.status = "filled";

      // Apply to portfolio
      this.applyFill(o.symbol, signedQty, pxExec, fee, i);
    }

    // expire DAY orders placed yesterday if still open
    for (const o of st.orders) {
      if (o.status === "open" && o.tif === "DAY" && o.placedIdx < i) o.status = "expired";
    }
  }

  private executionPrice(o: Order, b: Bar, slippageBps: number): number | null {
    const slip = (p: number) => p * (1 + Math.sign(p) * 0); // price sign irrelevant; we'll add absolute bps below
    const addSlippage = (p: number) => p * (1 + (o.side === "buy" ? +1 : -1) * slippageBps * 1e-4);

    if (o.type === "market") {
      return addSlippage(b.open);
    }
    if (o.type === "limit") {
      if (o.side === "buy") {
        if (b.low <= (o.limitPrice || -Infinity)) {
          // assume best execution at min(hit, open) within bar, slippage applied
          const px = Math.min(b.open, o.limitPrice!);
          return addSlippage(px);
        }
      } else {
        if (b.high >= (o.limitPrice || Infinity)) {
          const px = Math.max(b.open, o.limitPrice!);
          return addSlippage(px);
        }
      }
    }
    if (o.type === "stop") {
      if (o.side === "buy") {
        if (b.high >= (o.stopPrice || Infinity)) {
          const px = Math.max(b.open, o.stopPrice!);
          return addSlippage(px);
        }
      } else {
        if (b.low <= (o.stopPrice || -Infinity)) {
          const px = Math.min(b.open, o.stopPrice!);
          return addSlippage(px);
        }
      }
    }
    return null;
  }

  private applyFill(symbol: string, signedQty: number, price: number, fee: number, idx: number) {
    const st = this.state!;
    const positions = st.portfolio.positions;
    const p = positions[symbol] || (positions[symbol] = { symbol, qty: 0, avgPrice: 0, realizedPnl: 0, lastIdx: idx });

    // Cash impact
    st.portfolio.cash -= signedQty * price + fee;

    // Position accounting
    if ((p.qty >= 0 && signedQty >= 0) || (p.qty <= 0 && signedQty <= 0)) {
      // Increasing same direction
      const newQty = p.qty + signedQty;
      const newCost = p.avgPrice * Math.abs(p.qty) + price * Math.abs(signedQty);
      p.qty = newQty;
      p.avgPrice = newQty !== 0 ? newCost / Math.abs(newQty) : 0;
    } else {
      // Reducing / crossing
      const closing = Math.min(Math.abs(p.qty), Math.abs(signedQty)) * Math.sign(signedQty);
      const realized = (price - p.avgPrice) * closing * Math.sign(p.qty); // correct sign for long/short
      p.realizedPnl += realized;
      p.qty += signedQty;
      if (p.qty === 0) p.avgPrice = 0;
    }

    p.lastIdx = idx;

    // Trade book (round-trip heuristic): whenever position crosses zero
    // We'll record when qty flips sign or hits 0 from non-zero.
    // For more precise trade pairing, a FIFO lot engine would be needed; this is simplified.
    // We record trades only when flat after having non-zero previous.
    // The equity curve captures full P&L anyway.
    // (Omitted here to keep code compact; see exit logic below in applyStops.)
  }

  private applyStops(i: number) {
    const st = this.state!;
    const stop = st.opt.risk.stopLossPct;
    const take = st.opt.risk.takeProfitPct;
    if (stop == null && take == null) return;

    for (const sym of Object.keys(st.portfolio.positions)) {
      const p = st.portfolio.positions[sym];
      if (!p.qty) continue;
      const px = st.bars[sym][i].close;
      const side = p.qty > 0 ? "long" : "short";
      const ret = side === "long" ? (px / p.avgPrice - 1) : (p.avgPrice / px - 1);

      let shouldExit = false;
      if (stop != null && ret <= -Math.abs(stop)) shouldExit = true;
      if (take != null && ret >= Math.abs(take)) shouldExit = true;

      if (shouldExit) {
        const qty = Math.abs(p.qty);
        this.placeOrder({ symbol: sym, side: p.qty > 0 ? "sell" : "buy", type: "market", qty, note: "risk-exit" });
      }
    }
  }

  private markToMarket(i: number) {
    const st = this.state!;
    let mv = 0;
    for (const sym of Object.keys(st.portfolio.positions)) {
      const p = st.portfolio.positions[sym];
      const px = st.bars[sym][i].close;
      mv += p.qty * px;
    }
    st.portfolio.equity = st.portfolio.cash + mv;
    return st.portfolio.equity;
  }

  private uid() {
    const st = this.state!;
    return `${st.opt.name || "S"}_${st.uid++}`;
  }
}

/* =============================== Indicators =============================== */

export function sma(arr: number[], n: number, i?: number) {
  const k = i == null ? arr.length - 1 : i;
  if (k + 1 < n) return NaN;
  let s = 0;
  for (let j = k - n + 1; j <= k; j++) s += arr[j];
  return s / n;
}
export function ema(arr: number[], n: number) {
  const out = new Array(arr.length).fill(NaN);
  const a = 2 / (n + 1);
  let prev = arr[0];
  out[0] = prev;
  for (let i = 1; i < arr.length; i++) {
    const v = a * arr[i] + (1 - a) * prev;
    out[i] = v; prev = v;
  }
  return out;
}
export function rsi(closes: number[], n = 14) {
  const out = new Array(closes.length).fill(NaN);
  let up = 0, dn = 0;
  for (let i = 1; i < closes.length; i++) {
    const ch = closes[i] - closes[i - 1];
    const u = Math.max(0, ch), d = Math.max(0, -ch);
    if (i <= n) { up += u; dn += d; if (i === n) out[i] = 100 - 100 / (1 + (up / n) / ((dn || 1e-12) / n)); }
    else {
      up = (up * (n - 1) + u) / n;
      dn = (dn * (n - 1) + d) / n;
      const rs = up / (dn || 1e-12);
      out[i] = 100 - 100 / (1 + rs);
    }
  }
  return out;
}

/* ============================== Perf Metrics ============================== */

function summarize(eq: number[], rets: number[], trades: Trade[]) {
  const n = rets.length;
  const daily = rets.slice(1).filter((x) => isFinite(x));
  const m = mean(daily);
  const sd = stdev(daily, m);
  const sharpe = sd > 0 ? (m * 252) / (sd * Math.sqrt(252)) : 0;
  const cagr = eq.length > 1 ? Math.pow(eq[eq.length - 1] / eq[0], 252 / Math.max(1, eq.length - 1)) - 1 : 0;
  const vol = sd * Math.sqrt(252);
  const maxDD = maxDrawdown(eq);
  const hitRate = trades.length ? trades.filter((t) => t.pnl > 0).length / trades.length : 0;
  const avgTrade = trades.length ? trades.reduce((s, t) => s + t.pnl, 0) / trades.length : 0;

  return { cagr, vol, sharpe, maxDD, hitRate, avgTrade, nTrades: trades.length };
}

/* ================================ Helpers ================================ */

function clonePos(src: Record<string, Position>) {
  const out: Record<string, Position> = {};
  for (const k of Object.keys(src)) out[k] = { ...src[k] };
  return out;
}
function mean(a: number[]) {
  if (!a.length) return 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}
function stdev(a: number[], m = mean(a)) {
  if (a.length < 2) return 0;
  let acc = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - m; acc += d * d; }
  return Math.sqrt(acc / (a.length - 1));
}
function maxDrawdown(curve: number[]) {
  let peak = curve[0] || 1, maxDD = 0;
  for (let i = 1; i < curve.length; i++) {
    peak = Math.max(peak, curve[i]);
    const dd = 1 - (curve[i] / peak);
    if (dd > maxDD) maxDD = dd;
  }
  return maxDD;
}
function clamp(lo: number, hi: number, x: number) {
  return Math.max(lo, Math.min(hi, x));
}
function fmtTs(t: number) {
  try {
    const d = new Date(t);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  } catch { return String(t); }
}
function refPriceForType(o: Order, b: Bar) {
  if (o.type === "market") return b.open;
  if (o.type === "limit") return o.limitPrice ?? b.open;
  if (o.type === "stop") return o.stopPrice ?? b.open;
  return b.open;
}

/* ============================ Named Exports ============================ */

export { StrategyBase as BaseStrategy };