// risk/src/stress.ts
// Pure, zero-import stress-testing utilities.
// Works standalone or alongside risk/src/index.ts. No external deps.
//
// What you get:
//  • buildExposures(positions, factorLoadings?)        → Exposure[]
//  • runFactorStress(exposures, scenario)              → StressResult
//  • runPriceStress(positions, scenario)               → StressResultRich
//  • runCombinedStress(positions, scenario, opts?)     → StressResultRich
//  • composeScenarios(...scenarios) / scaleScenario(s, k)
//  • convenience scenario builders: equityCrash(%), parallelRate(bps), fxMove(%)
//  • tiny helpers: pct(), bps(), clamp(), etc.
//
// Conventions
//  • Factor shocks are *returns* (e.g., -0.05 for -5%).
//  • Price shocks are *returns* per security id (same convention).
//  • P&L is monetary, using position.price * qty * return.
//  • If you only have factor shocks, P&L is Exposure (monetary) × factor return.
//  • If you also provide factor loadings per security, factor P&L is
//    distributed to securities for contribBySecurity.

export type Dict<T = any> = { [k: string]: T };

// ────────────────────────────────────────────────────────────────────────────
// Types (kept compatible with risk/src/index.ts)
// ────────────────────────────────────────────────────────────────────────────

export type Position = {
  id: string;
  qty: number;
  price?: number;
  sector?: string;
  assetClass?: string;
  factorLoads?: Record<string, number>; // optional per-security factor loadings
  meta?: Dict;
};

export type Exposure = { factor: string; value: number };

export type FactorLoadings = Record<string, Record<string, number>>; // factor -> (securityId -> loading)

export type FactorScenario = {
  name: string;
  factors: Record<string, number>; // factor -> return (e.g., -0.03 = -3%)
  description?: string;
  meta?: Dict;
};

export type PriceScenario = {
  name: string;
  prices: Record<string, number>; // securityId -> return (e.g., -0.10 = -10%)
  description?: string;
  meta?: Dict;
};

export type CombinedScenario = {
  name: string;
  factors?: FactorScenario["factors"];
  prices?: PriceScenario["prices"];
  description?: string;
  meta?: Dict;
};

export type StressResult = {
  name: string;
  pnl: number;                              // total portfolio P&L
  contribByFactor: Record<string, number>;  // factor -> P&L
};

export type StressResultRich = StressResult & {
  contribBySecurity?: Record<string, number>;         // securityId -> P&L
  contribByGroup?: { sector?: Record<string, number>; assetClass?: Record<string, number> };
  assumptions?: Dict;                                  // notes on slippage, fees, caps etc.
};

// Options for combined stress execution
export type StressOptions = {
  // When distributing factor shock to securities:
  factorLoadings?: FactorLoadings; // used if positions lack factorLoads
  // Post-processing adjustments:
  slippageBps?: number;            // subtract |gross shocked notional| * bps/10_000
  perTradeFee?: number;            // subtract flat fee per *impacted* security (prices or factor implied)
  capLoss?: number;                // optional lower cap on total P&L (e.g., -Infinity by default)
  floorGain?: number;              // optional upper cap on total P&L
};

// ────────────────────────────────────────────────────────────────────────────
// Core builders
// ────────────────────────────────────────────────────────────────────────────

export function buildExposures(positions: Position[], globalLoads?: FactorLoadings): Exposure[] {
  const map: Record<string, number> = Object.create(null);
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const worth = (p.price || 0) * p.qty;
    const loads = p.factorLoads || mergeLoadsFor(p.id, globalLoads);
    if (!loads) continue;
    for (const f in loads) {
      const w = Number(loads[f]) || 0;
      if (!map[f]) map[f] = 0;
      map[f] += worth * w;
    }
  }
  const out: Exposure[] = [];
  for (const f in map) out.push({ factor: f, value: map[f] });
  out.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return out;
}

export function runFactorStress(exposures: Exposure[], scenario: FactorScenario): StressResult {
  let total = 0;
  const contrib: Record<string, number> = {};
  for (let i = 0; i < exposures.length; i++) {
    const e = exposures[i];
    const r = Number(scenario.factors[e.factor] || 0);
    const pnl = e.value * r;
    contrib[e.factor] = (contrib[e.factor] || 0) + pnl;
    total += pnl;
  }
  return { name: scenario.name, pnl: round2(total), contribByFactor: squashZeros(contrib) };
}

export function runPriceStress(positions: Position[], scenario: PriceScenario): StressResultRich {
  let total = 0;
  const bySec: Record<string, number> = {};
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const r = Number(scenario.prices[p.id] || 0);
    if (!r) continue;
    const pnl = (p.price || 0) * p.qty * r;
    bySec[p.id] = (bySec[p.id] || 0) + pnl;
    total += pnl;
  }
  return {
    name: scenario.name,
    pnl: round2(total),
    contribByFactor: {},
    contribBySecurity: squashZeros(bySec),
    contribByGroup: groupBreakdown(positions, bySec)
  };
}

export function runCombinedStress(
  positions: Position[],
  scenario: CombinedScenario,
  opts?: StressOptions
): StressResultRich {
  const name = scenario.name;
  const pxPart = scenario.prices ? runPriceStress(positions, { name, prices: scenario.prices }) : undefined;

  // Factor part
  let facPart: StressResult | undefined;
  let facToSec: Record<string, number> | undefined;

  if (scenario.factors && Object.keys(scenario.factors).length) {
    const exps = buildExposures(positions, opts?.factorLoadings);
    facPart = runFactorStress(exps, { name, factors: scenario.factors });

    // Distribute factor P&L to securities proportional to |worth * loading|
    facToSec = distributeFactorPnlToSecurities(positions, scenario.factors, opts?.factorLoadings);
  }

  // Combine security contributions
  const bySec: Record<string, number> = {};
  if (pxPart?.contribBySecurity) addInto(bySec, pxPart.contribBySecurity);
  if (facToSec) addInto(bySec, facToSec);

  // Factor contributions (from facPart; price-only contributes none)
  const byFactor: Record<string, number> = facPart ? { ...facPart.contribByFactor } : {};

  // Totals and adjustments
  let total = (pxPart?.pnl || 0) + (facPart?.pnl || 0);

  // Apply optional costs/slippage caps
  const assumptions: Dict = {};
  if (opts?.slippageBps && opts.slippageBps !== 0) {
    const impacted = Object.keys(bySec);
    let grossNotional = 0;
    for (let i = 0; i < positions.length; i++) {
      const id = positions[i].id;
      if (!impacted.includes(id)) continue;
      grossNotional += Math.abs((positions[i].price || 0) * positions[i].qty);
    }
    const slip = grossNotional * (opts.slippageBps / 10_000);
    total -= slip;
    assumptions.slippageBps = opts.slippageBps;
    assumptions.slippageCost = round2(slip);
  }
  if (opts?.perTradeFee && opts.perTradeFee > 0) {
    const impactedCount = Object.keys(bySec).length;
    const fee = impactedCount * opts.perTradeFee;
    total -= fee;
    assumptions.perTradeFee = opts.perTradeFee;
    assumptions.tradeCount = impactedCount;
    assumptions.feeCost = round2(fee);
  }
  if (typeof opts?.capLoss === "number") total = Math.max(total, opts.capLoss);
  if (typeof opts?.floorGain === "number") total = Math.min(total, opts.floorGain);

  return {
    name,
    pnl: round2(total),
    contribByFactor: squashZeros(byFactor),
    contribBySecurity: squashZeros(bySec),
    contribByGroup: groupBreakdown(positions, bySec),
    assumptions
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Scenario utilities (compose/scale/normalize)
// ────────────────────────────────────────────────────────────────────────────

export function composeScenarios(...scenarios: CombinedScenario[]): CombinedScenario {
  const name = scenarios.map(s => s.name).join(" + ");
  const fac: Record<string, number> = {};
  const px: Record<string, number> = {};
  for (let i = 0; i < scenarios.length; i++) {
    const s = scenarios[i];
    if (s.factors) for (const k in s.factors) fac[k] = (fac[k] || 0) + Number(s.factors[k] || 0);
    if (s.prices) for (const k in s.prices) px[k] = (px[k] || 0) + Number(s.prices[k] || 0);
  }
  return { name, factors: fac, prices: px };
}

export function scaleScenario(s: CombinedScenario, k: number): CombinedScenario {
  const fac: Record<string, number> = {};
  const px: Record<string, number> = {};
  if (s.factors) for (const f in s.factors) fac[f] = Number(s.factors[f]) * k;
  if (s.prices) for (const id in s.prices) px[id] = Number(s.prices[id]) * k;
  return { name: `${s.name} x${k}`, factors: fac, prices: px, description: s.description, meta: s.meta };
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience scenario builders
// ────────────────────────────────────────────────────────────────────────────

/** Equity market crash of `drop` (e.g., equityCrash(-0.1) for -10%). */
export function equityCrash(drop: number): CombinedScenario {
  return { name: `Equity ${pctFmt(drop)}`, factors: { MKT: drop } };
}

/** Parallel rate shift in basis points applied to RATE factor (e.g., +100 bps). */
export function parallelRate(basisPoints: number): CombinedScenario {
  return { name: `Rates ${bpsFmt(basisPoints)}`, factors: { RATE: basisPoints / 10_000 } };
}

/** FX move on a given factor key, e.g., "USDJPY" or "EURUSD". */
export function fxMove(pair: string, pctMove: number): CombinedScenario {
  const f: Record<string, number> = {}; f[pair] = pctMove;
  return { name: `FX ${pair} ${pctFmt(pctMove)}`, factors: f };
}

// ────────────────────────────────────────────────────────────────────────────
// Internals
// ────────────────────────────────────────────────────────────────────────────

function mergeLoadsFor(id: string, gl?: FactorLoadings): Record<string, number> | undefined {
  if (!gl) return undefined;
  const res: Record<string, number> = {};
  for (const f in gl) {
    const v = gl[f]?.[id];
    if (v == null) continue;
    res[f] = Number(v) || 0;
  }
  return Object.keys(res).length ? res : undefined;
}

/** Distribute factor P&L to securities ∝ |worth * loading| per factor, summed. */
function distributeFactorPnlToSecurities(
  positions: Position[],
  factorShock: Record<string, number>,
  globalLoads?: FactorLoadings
): Record<string, number> {
  // First compute each security's implied P&L from all factor shocks
  const bySec: Record<string, number> = {};
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const worth = (p.price || 0) * p.qty;
    const loads = p.factorLoads || mergeLoadsFor(p.id, globalLoads);
    if (!loads || !worth) continue;

    let pnl = 0;
    for (const f in factorShock) {
      const r = Number(factorShock[f] || 0);
      const w = Number(loads[f] || 0);
      pnl += worth * w * r;
    }
    if (pnl !== 0) bySec[p.id] = (bySec[p.id] || 0) + pnl;
  }
  return bySec;
}

function groupBreakdown(positions: Position[], bySecurity: Record<string, number>): {
  sector?: Record<string, number>;
  assetClass?: Record<string, number>;
} {
  const sector: Record<string, number> = {};
  const asset: Record<string, number> = {};
  const meta: Record<string, Position> = {};
  for (let i = 0; i < positions.length; i++) meta[positions[i].id] = positions[i];

  for (const id in bySecurity) {
    const pnl = bySecurity[id];
    const p = meta[id];
    const s = p?.sector || "UNKNOWN";
    const a = p?.assetClass || "UNKNOWN";
    sector[s] = (sector[s] || 0) + pnl;
    asset[a] = (asset[a] || 0) + pnl;
  }

  return { sector: squashZeros(sector), assetClass: squashZeros(asset) };
}

function addInto(dst: Record<string, number>, src: Record<string, number>) {
  for (const k in src) dst[k] = (dst[k] || 0) + Number(src[k] || 0);
}

function squashZeros<T extends Record<string, number>>(m: T): T {
  const out: any = {};
  for (const k in m) if (m[k]) out[k] = round2(m[k]);
  return out;
}

// formatting helpers
function pctFmt(x: number): string {
  return `${round2(x * 100)}%`;
}
function bpsFmt(x: number): string {
  return `${round2(x)}bps`;
}

// math/helpers
function round2(x: number): number { return Math.round(x * 100) / 100; }

// Optional user helpers (exported for convenience)
export function pct(x: number | string): number {
  if (typeof x === "number") return x;
  const s = x.trim();
  if (s.endsWith("%")) return Number(s.slice(0, -1)) / 100;
  return Number(s);
}
export function bps(x: number | string): number {
  if (typeof x === "number") return x / 10_000;
  const s = x.trim().toLowerCase().replace("bps", "");
  return Number(s) / 10_000;
}
export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */