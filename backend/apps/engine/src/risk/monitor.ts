// risk/monitor.ts
// Import-free real-time risk monitor.
//
// What it does
// - Tracks running equity, exposure, leverage, PnL, drawdown, realized PnL
// - Maintains rolling equity/returns windows for quick stats (vol, Sharpe)
// - Per-symbol sizing metrics (qty, notional) from {positions, prices}
// - Built-in limit engine (same fields as risk/limits.ts) with per-symbol + global limits
// - Event hooks: "update" (every snapshot) and "violation" (on any breach)
//
// Usage:
//   const rm = createRiskMonitor({ retWindow: 252 });
//   rm.setLimits("global", { maxLeverage: 3, maxDrawdown: 0.2 });
//   rm.setLimits("AAPL", { maxPosition: 5000, maxNotional: 1_000_000 });
//   rm.update({ positions: { AAPL: 1200 }, prices: { AAPL: 190 }, equity: 250_000 });
//   const snap = rm.check();  // { ok, violations, metrics, snapshot }

export type LimitConfig = {
  maxPosition?: number;
  maxNotional?: number;
  maxLoss?: number;            // cumulative realized loss (negative number)
  maxDrawdown?: number;        // fraction; e.g. 0.2 = 20%
  maxGrossExposure?: number;   // absolute $
  maxLeverage?: number;        // gross / equity
  maxOrdersPerMin?: number;
};

export type Snapshot = {
  ts?: number;
  equity?: number;
  realized?: number;
  positions?: Record<string, number>; // qty
  prices?: Record<string, number>;    // last
  ordersLastMin?: number;
};

export type Metrics = {
  ts: number;
  equity: number;
  realized: number;
  peakEquity: number;
  dd: number;           // current drawdown
  maxDD: number;        // max drawdown since start (fraction)
  longMV: number;
  shortMV: number;
  gross: number;
  net: number;
  leverage: number;
  vol: number;          // rolling annualized vol (using returns)
  sharpe: number;       // simple Sharpe (rf=0) on rolling window
};

export type CheckResult = {
  ok: boolean;
  violations: Array<{ key: string; field: keyof LimitConfig; value: number; limit: number }>;
  metrics: Metrics;
  snapshot: Required<Snapshot>;
};

export type RiskMonitorAPI = {
  // data
  update(s: Snapshot): Metrics;                          // merge snapshot -> recompute metrics
  snapshot(): Required<Snapshot>;
  metrics(): Metrics;

  // limits
  setLimits(key: string, cfg: LimitConfig): void;        // "global" or symbol
  getLimits(key: string): LimitConfig | undefined;
  allLimits(): Record<string, LimitConfig>;
  clearLimits(): void;

  // checks
  check(): CheckResult;                                  // run checks using current state

  // events
  on(ev: "update" | "violation", fn: (arg: any) => void): () => void;

  // maintenance
  reset(): void;                                         // clears time series & peaks, keeps limits
};

export function createRiskMonitor(opts: { retWindow?: number; annFactor?: number } = {}): RiskMonitorAPI {
  const cfg = {
    window: Math.max(4, (opts.retWindow || 252) | 0),
    ann: Math.max(1, (opts.annFactor || 252) | 0),
  };

  // ---------- state ----------
  const limits = new Map<string, LimitConfig>();

  const snap: Required<Snapshot> = {
    ts: Date.now(),
    equity: 0,
    realized: 0,
    positions: {},
    prices: {},
    ordersLastMin: 0,
  };

  let peakEquity = -Infinity;
  let maxDD = 0;

  // rolling equity & returns
  const eq: number[] = [];     // equity time series
  const rets: number[] = [];   // step returns (simple)
  let vol = 0;
  let sharpe = 0;

  // listeners
  const listeners = { update: new Set<(x:any)=>void>(), violation: new Set<(x:any)=>void>() };
  const emit = (ch: "update" | "violation", payload: any) => {
    const ls = listeners[ch]; if (!ls) return;
    for (const fn of Array.from(ls)) { try { fn(payload); } catch {} }
  };

  // ---------- core calc ----------
  function exposure(positions: Record<string, number>, prices: Record<string, number>) {
    let longMV = 0, shortMV = 0;
    for (const k in positions) {
      const q = Number(positions[k]); const p = Number(prices[k]);
      if (!isFinite(q) || !isFinite(p)) continue;
      const mv = q * p;
      if (mv > 0) longMV += mv; else if (mv < 0) shortMV += -mv;
    }
    const gross = longMV + shortMV;
    const net = longMV - shortMV;
    const lev = snap.equity > 0 ? gross / snap.equity : 0;
    return { longMV, shortMV, gross, net, lev };
  }

  function pushReturn(newEq: number) {
    const lastEq = eq.length ? eq[eq.length - 1] : 0;
    eq.push(newEq);
    if (eq.length > cfg.window + 1) eq.shift();
    if (lastEq > 0) {
      rets.push((newEq - lastEq) / lastEq);
      if (rets.length > cfg.window) rets.shift();
    } else {
      // prime the series with a zero return
      rets.push(0);
    }
    // rolling vol + Sharpe
    const sigma = stdev(rets);
    vol = sigma * Math.sqrt(cfg.ann);
    const m = mean(rets);
    sharpe = sigma === 0 ? 0 : (m / sigma) * Math.sqrt(cfg.ann);
  }

  function update(s: Snapshot): Metrics {
    // merge snapshot
    if (s.ts != null) snap.ts = Number(s.ts);
    if (s.equity != null) snap.equity = Number(s.equity);
    if (s.realized != null) snap.realized = Number(s.realized);
    if (s.ordersLastMin != null) snap.ordersLastMin = Number(s.ordersLastMin);

    if (s.positions) {
      const out: Record<string, number> = {};
      for (const k in s.positions) {
        const v = Number(s.positions[k]); if (isFinite(v)) out[k] = v;
      }
      snap.positions = out;
    }
    if (s.prices) {
      const out: Record<string, number> = {};
      for (const k in s.prices) {
        const v = Number(s.prices[k]); if (isFinite(v) && v > 0) out[k] = v;
      }
      snap.prices = out;
    }

    // equity series / peaks / dd
    const ts = isFinite(snap.ts) ? snap.ts : Date.now();
    if (eq.length === 0 || eq[eq.length - 1] !== snap.equity) pushReturn(snap.equity);
    if (snap.equity > peakEquity) peakEquity = snap.equity;
    const dd = peakEquity > 0 ? Math.max(0, (peakEquity - snap.equity) / peakEquity) : 0;
    if (dd > maxDD) maxDD = dd;

    const ex = exposure(snap.positions, snap.prices);

    const m: Metrics = {
      ts,
      equity: round2(snap.equity),
      realized: round2(snap.realized),
      peakEquity: round2(peakEquity > 0 ? peakEquity : snap.equity),
      dd: round4(dd),
      maxDD: round4(maxDD),
      longMV: round2(ex.longMV),
      shortMV: round2(ex.shortMV),
      gross: round2(ex.gross),
      net: round2(ex.net),
      leverage: round4(ex.lev),
      vol: round4(vol),
      sharpe: round4(sharpe),
    };

    emit("update", { metrics: m, snapshot: { ...snap } });
    return m;
  }

  function snapshot() { return { ...snap }; }
  function metrics() {
    const ex = exposure(snap.positions, snap.prices);
    const dd = peakEquity > 0 ? Math.max(0, (peakEquity - snap.equity) / peakEquity) : 0;
    return {
      ts: snap.ts,
      equity: round2(snap.equity),
      realized: round2(snap.realized),
      peakEquity: round2(peakEquity > 0 ? peakEquity : snap.equity),
      dd: round4(dd),
      maxDD: round4(maxDD),
      longMV: round2(ex.longMV),
      shortMV: round2(ex.shortMV),
      gross: round2(ex.gross),
      net: round2(ex.net),
      leverage: round4(ex.lev),
      vol: round4(vol),
      sharpe: round4(sharpe),
    } as Metrics;
  }

  // ---------- limits ----------
  function setLimits(key: string, cfg: LimitConfig) { limits.set(key, { ...cfg }); }
  function getLimits(key: string) { return limits.get(key); }
  function allLimits() { const o: Record<string, LimitConfig> = {}; limits.forEach((v,k)=>o[k]={...v}); return o; }
  function clearLimits() { limits.clear(); }

  function check(): CheckResult {
    const viols: CheckResult["violations"] = [];
    const ex = exposure(snap.positions, snap.prices);
    const dd = peakEquity > 0 ? (peakEquity - snap.equity) / peakEquity : 0;

    // helper
    function apply(key: string, cfg?: LimitConfig, fields?: { qty?: number; notional?: number }) {
      if (!cfg) return;
      if (cfg.maxPosition != null && fields && fields.qty != null && Math.abs(fields.qty) > cfg.maxPosition) {
        viols.push({ key, field: "maxPosition", value: fields.qty, limit: cfg.maxPosition });
      }
      if (cfg.maxNotional != null && fields && fields.notional != null && Math.abs(fields.notional) > cfg.maxNotional) {
        viols.push({ key, field: "maxNotional", value: fields.notional, limit: cfg.maxNotional });
      }
      if (cfg.maxLoss != null && snap.realized < cfg.maxLoss) {
        viols.push({ key, field: "maxLoss", value: snap.realized, limit: cfg.maxLoss });
      }
      if (cfg.maxDrawdown != null && dd > cfg.maxDrawdown) {
        viols.push({ key, field: "maxDrawdown", value: dd, limit: cfg.maxDrawdown });
      }
      if (cfg.maxGrossExposure != null && ex.gross > cfg.maxGrossExposure) {
        viols.push({ key, field: "maxGrossExposure", value: ex.gross, limit: cfg.maxGrossExposure });
      }
      if (cfg.maxLeverage != null && snap.equity > 0 && ex.lev > cfg.maxLeverage) {
        viols.push({ key, field: "maxLeverage", value: ex.lev, limit: cfg.maxLeverage });
      }
      if (cfg.maxOrdersPerMin != null && snap.ordersLastMin > cfg.maxOrdersPerMin) {
        viols.push({ key, field: "maxOrdersPerMin", value: snap.ordersLastMin, limit: cfg.maxOrdersPerMin });
      }
    }

    // per-symbol checks
    for (const sym in snap.positions) {
      const cfg = limits.get(sym);
      if (!cfg) continue;
      const qty = Number(snap.positions[sym]) || 0;
      const notional = Math.abs(qty) * (Number(snap.prices[sym]) || 0);
      apply(sym, cfg, { qty, notional });
    }

    // global checks
    apply("global", limits.get("global"));

    const m = metrics();
    const out: CheckResult = { ok: viols.length === 0, violations: viols, metrics: m, snapshot: { ...snap } };
    if (!out.ok) emit("violation", out);
    return out;
  }

  // ---------- events ----------
  function on(ev: "update" | "violation", fn: (arg: any) => void) {
    const set = listeners[ev];
    if (!set) return () => {};
    set.add(fn);
    return () => set.delete(fn);
  }

  function reset() {
    eq.length = 0; rets.length = 0;
    peakEquity = -Infinity; maxDD = 0;
    vol = 0; sharpe = 0;
  }

  return {
    update,
    snapshot,
    metrics,
    setLimits,
    getLimits,
    allLimits,
    clearLimits,
    check,
    on,
    reset,
  };
}

/* ------------------------------ math utils ------------------------------ */

function mean(a: number[]) {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) { const v = a[i]; if (isFinite(v)) { s += v; n++; } }
  return n ? s / n : 0;
}
function stdev(a: number[]) {
  const m = mean(a); let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) { const v = a[i]; if (isFinite(v)) { const d = v - m; s += d * d; n++; } }
  return n > 1 ? Math.sqrt(s / (n - 1)) : 0;
}
function round2(n: number) { return Math.round(n * 100) / 100; }
function round4(n: number) { return Math.round(n * 10000) / 10000; }