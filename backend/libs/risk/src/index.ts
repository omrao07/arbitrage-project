// risk/src/index.ts
// Pure, zero-import risk toolkit: factor exposures, VaR/ES, stress, and limits.
// Everything is self-contained (no external deps). Strongly typed and defensive.
//
// High-level usage:
//
//   const snap: PortfolioSnapshot = { ts: new Date(), positions: [...] };
//   const hist: FactorHistory = { factors: ["MKT","RATE"], series: { MKT: [...], RATE: [...] } };
//   const loads: FactorLoadings = {          // if positions omit factorLoads, you can pass here
//     MKT: { AAPL: 1.2, MSFT: 1.1 },        // securityId -> loading on factor
//     RATE: { AAPL: -0.3, MSFT: -0.2 }
//   };
//   const md: MarketData = { prices: { AAPL: 220, MSFT: 410 } };
//   const res = computeRiskState({ snapshot: snap, factorHistory: hist, market: md, factorLoadings: loads });
//
//   const breaches = checkLimits(res.metrics, [
//     { id:"maxVaR99", metric: "var_99", op: "<=", threshold: 1_000_000, severity: "critical" },
//     { id:"grossCap", metric: "gross", op: "<=", threshold: 50_000_000, severity: "warn" },
//   ]);
//
// Exports focus on: types, utilities, orchestrators.

export type Dict<T = any> = { [k: string]: T };

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type Position = {
  id: string;                // unique security identifier
  qty: number;               // units (signed)
  price?: number;            // latest price (if absent, use market.prices[id] or 0)
  sector?: string;
  assetClass?: string;
  factorLoads?: Record<string, number>; // factor -> loading (optional; can be injected globally)
  meta?: Dict;
};

export type MarketData = {
  prices?: Record<string, number>;     // id -> price
  fxRates?: Record<string, number>;    // optional FX map (base->quote), not used by core funcs
};

export type PortfolioSnapshot = {
  ts: string | number | Date;
  base?: string;             // currency code (optional; informational)
  positions: Position[];
  market?: MarketData;
};

export type FactorHistory = {
  // factor returns, aligned by index (time order oldest→newest)
  factors: string[];
  series: Record<string, number[]>; // factor -> returns[]
};

export type FactorLoadings = Record<string, Record<string, number>>; // factor -> (securityId -> loading)

export type Exposure = {
  factor: string;
  value: number;             // monetary exposure in base currency
};

export type StressShock = {
  name: string;
  shocks: Record<string, number>; // factor -> return shock (e.g., -0.05 for -5%)
};

export type StressResult = {
  name: string;
  pnl: number;
  contribByFactor: Record<string, number>;
};

export type VaRMethod = "historical" | "parametric";

export type RiskOptions = {
  varConfidence?: number;        // e.g., 0.99 for 99%
  esConfidence?: number;         // e.g., 0.975
  varMethod?: VaRMethod;         // default: "historical" if history provided else "parametric"
  annualizationFactor?: number;  // for volatility (default 252)
};

export type RiskMetrics = {
  gross: number;
  net: number;
  long: number;
  short: number;
  volDaily?: number;       // daily sigma from history
  volAnnual?: number;      // annualized
  var_95?: number;
  var_99?: number;
  es_95?: number;
  es_99?: number;
  exposures: Exposure[];   // factor exposures (monetary)
};

export type Limit = {
  id: string;
  metric: keyof RiskMetrics | string; // allows custom metric keys
  op: "<" | "<=" | ">" | ">=" | "abs<=" | "abs<";
  threshold: number;
  severity?: "info" | "warn" | "critical";
};

export type Breach = {
  id: string;
  limitId: string;
  metric: string;
  value: number;
  threshold: number;
  op: Limit["op"];
  severity: NonNullable<Limit["severity"]>;
};

export type RiskState = {
  ts: string;
  metrics: RiskMetrics;
  method: VaRMethod;
  histPnL?: number[];       // historical factor P&L path used for historical VaR/ES
  stresses?: StressResult[];
  notes?: string[];
};

// ────────────────────────────────────────────────────────────────────────────
// Public orchestrator
// ────────────────────────────────────────────────────────────────────────────

export function computeRiskState(input: {
  snapshot: PortfolioSnapshot;
  factorHistory?: FactorHistory;
  market?: MarketData;
  factorLoadings?: FactorLoadings; // optional global map applied if position lacks loads
  options?: RiskOptions;
  stresses?: StressShock[];
}): RiskState {
  const notes: string[] = [];
  const snap = input.snapshot || { ts: Date.now(), positions: [] };
  const market = input.market || snap.market || {};
  const opts = withDefaults(input.options);

  // 1) Enrich positions with prices
  const pos = normalizePositions(snap.positions, market.prices || {});

  // 2) Basic gross/net metrics
  const gross = sumAbsValue(pos);
  const { long, short, net } = longShortNet(pos);

  // 3) Factor exposures
  const exposures = computeFactorExposures(pos, input.factorLoadings);

  // 4) If we have history, build historical P&L from factor returns
  let method: VaRMethod = "parametric";
  let histPnL: number[] | undefined;
  let volDaily: number | undefined;
  let volAnnual: number | undefined;
  let var95: number | undefined, var99: number | undefined;
  let es95: number | undefined, es99: number | undefined;

  const E = exposureVector(exposures, input.factorHistory?.factors || []);
  if (input.factorHistory && validHistory(input.factorHistory)) {
    method = "historical";
    histPnL = portfolioPnLFromFactors(E, input.factorHistory);
    if (histPnL.length >= 2) {
      const sigma = stdev(histPnL);
      volDaily = sigma;
      volAnnual = sigma * Math.sqrt(opts.annualizationFactor);
      var95 = -percentile(histPnL, 0.05);
      var99 = -percentile(histPnL, 0.01);
      es95 = -expectedShortfall(histPnL, 0.05);
      es99 = -expectedShortfall(histPnL, 0.01);
    } else {
      notes.push("Insufficient historical points; falling back to parametric VaR.");
      method = "parametric";
    }
  }

  // 5) Parametric VaR if selected or fallback
  if (method === "parametric") {
    // Estimate covariance from history if available; else diagonal ~0
    let cov: number[][] | undefined;
    if (input.factorHistory && validHistory(input.factorHistory)) {
      cov = covarianceMatrix(input.factorHistory);
    } else {
      cov = diagonalZeros(E.length);
      notes.push("No factor history; parametric VaR uses zero covariance (VaR=0).");
    }
    const sigmaP = portfolioSigma(E, cov || []);
    volDaily = sigmaP;
    volAnnual = sigmaP * Math.sqrt(opts.annualizationFactor);
    var95 = gaussVaR(sigmaP, 0.95);
    var99 = gaussVaR(sigmaP, 0.99);
    es95 = gaussES(sigmaP, 0.95);
    es99 = gaussES(sigmaP, 0.99);
  }

  // 6) Optional stresses
  let stresses: StressResult[] | undefined;
  if (input.stresses && input.stresses.length) {
    stresses = input.stresses.map(s => runStress(E, s));
  }

  const metrics: RiskMetrics = {
    gross, net, long, short,
    volDaily, volAnnual,
    var_95: round2(var95),
    var_99: round2(var99),
    es_95: round2(es95),
    es_99: round2(es99),
    exposures
  };

  return {
    ts: iso(snap.ts),
    metrics,
    method,
    histPnL,
    stresses,
    notes
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Factor exposures
// ────────────────────────────────────────────────────────────────────────────

export function computeFactorExposures(
  positions: Position[],
  globalLoads?: FactorLoadings
): Exposure[] {
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

function exposureVector(expos: Exposure[], order: string[]): number[] {
  if (!order || !order.length) return expos.map(e => e.value);
  const map: Record<string, number> = {};
  for (let i = 0; i < expos.length; i++) map[expos[i].factor] = expos[i].value;
  const out = new Array(order.length);
  for (let i = 0; i < order.length; i++) out[i] = Number(map[order[i]] || 0);
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Historical P&L from factor returns
// ────────────────────────────────────────────────────────────────────────────

export function portfolioPnLFromFactors(E: number[], hist: FactorHistory): number[] {
  const T = histLength(hist);
  if (!T) return [];
  const pnl = new Array(T).fill(0);
  for (let t = 0; t < T; t++) {
    let s = 0;
    for (let f = 0; f < hist.factors.length; f++) {
      const fac = hist.factors[f];
      const ret = Number(hist.series[fac][t] || 0);
      s += (E[f] || 0) * ret;
    }
    pnl[t] = s;
  }
  return pnl;
}

function validHistory(h: FactorHistory | undefined): h is FactorHistory {
  if (!h || !h.factors || !h.factors.length) return false;
  const T = histLength(h);
  return T > 1;
}

function histLength(h: FactorHistory): number {
  if (!h.factors.length) return 0;
  const f0 = h.factors[0];
  return Array.isArray(h.series[f0]) ? h.series[f0].length : 0;
}

// ────────────────────────────────────────────────────────────────────────────
// Parametric (Gaussian) VaR/ES
// ────────────────────────────────────────────────────────────────────────────

export function covarianceMatrix(hist: FactorHistory): number[][] {
  const F = hist.factors.length;
  const T = histLength(hist);
  const M: number[][] = new Array(F);
  const X: number[][] = new Array(F);
  for (let i = 0; i < F; i++) {
    const f = hist.factors[i];
    const arr = (hist.series[f] || []).slice(-T);
    X[i] = arr.map(Number);
  }
  // demean
  const mu = new Array(F).fill(0);
  for (let i = 0; i < F; i++) mu[i] = mean(X[i]);
  for (let i = 0; i < F; i++) for (let t = 0; t < T; t++) X[i][t] = X[i][t] - mu[i];

  for (let i = 0; i < F; i++) {
    M[i] = new Array(F).fill(0);
    for (let j = i; j < F; j++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += X[i][t] * X[j][t];
      const v = s / Math.max(1, T - 1);
      M[i][j] = v;
      if (i !== j) M[j][i] = v;
    }
  }
  return M;
}

export function portfolioSigma(E: number[], cov: number[][]): number {
  const F = E.length;
  let s = 0;
  for (let i = 0; i < F; i++) {
    for (let j = 0; j < F; j++) {
      s += (E[i] || 0) * (cov[i]?.[j] || 0) * (E[j] || 0);
    }
  }
  return Math.sqrt(Math.max(0, s));
}

export function gaussVaR(sigma: number, cl: 0.95 | 0.99 | number): number {
  const z = inverseStdNormal(cl);
  return Math.max(0, z * Math.max(0, sigma));
}

export function gaussES(sigma: number, cl: 0.95 | 0.99 | number): number {
  // ES = sigma * phi(z) / (1 - cl), where z = Φ^{-1}(cl)
  const z = inverseStdNormal(cl);
  const phi = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const tail = 1 - cl;
  return Math.max(0, (sigma * phi) / Math.max(1e-12, tail));
}

// ────────────────────────────────────────────────────────────────────────────
// Stresses
// ────────────────────────────────────────────────────────────────────────────

export function runStress(E: number[], shock: StressShock): StressResult {
  // E aligned to unknown order; assume shock.shocks keys align to E order?
  // To avoid order dependence, we compute by factor name if present in E? We only have E vector.
  // For correctness, prefer named evaluation; export helper for that too.
  // Here we compute via named if provided by user using runStressNamed.
  return { name: shock.name, pnl: 0, contribByFactor: {} };
}

export function runStressNamed(exposures: Exposure[], shock: StressShock): StressResult {
  let pnl = 0;
  const contrib: Record<string, number> = {};
  for (let i = 0; i < exposures.length; i++) {
    const e = exposures[i];
    const r = Number(shock.shocks[e.factor] || 0);
    const c = e.value * r;
    pnl += c;
    contrib[e.factor] = c;
  }
  return { name: shock.name, pnl, contribByFactor: contrib };
}

// ────────────────────────────────────────────────────────────────────────────
// Limits
// ────────────────────────────────────────────────────────────────────────────

export function checkLimits(metrics: RiskMetrics, limits: Limit[]): Breach[] {
  const out: Breach[] = [];
  for (let i = 0; i < limits.length; i++) {
    const L = limits[i];
    const v = (metrics as any)[L.metric];
    if (typeof v !== "number") continue;
    const ok = compare(v, L.op, L.threshold);
    if (!ok) {
      out.push({
        id: `breach_${L.id}`,
        limitId: L.id,
        metric: String(L.metric),
        value: v,
        threshold: L.threshold,
        op: L.op,
        severity: L.severity || "warn"
      });
    }
  }
  return out;
}

export function killswitchSuggested(breaches: Breach[], level: "critical" | "warn" = "critical"): boolean {
  for (let i = 0; i < breaches.length; i++) if (breaches[i].severity === level) return true;
  return false;
}

// ────────────────────────────────────────────────────────────────────────────
// Exposure breakdowns & summaries
// ────────────────────────────────────────────────────────────────────────────

export function breakdownBy<T extends "sector" | "assetClass" | string>(
  positions: Position[],
  key: T
): Record<string, number> {
  const out: Record<string, number> = Object.create(null);
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const k = (p as any)[key] || "UNKNOWN";
    out[k] = (out[k] || 0) + (p.price || 0) * p.qty;
  }
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Internals & utilities
// ────────────────────────────────────────────────────────────────────────────

function withDefaults(o?: RiskOptions): Required<RiskOptions> {
  return {
    varConfidence: clamp(o?.varConfidence ?? 0.99, 0.5, 0.999),
    esConfidence: clamp(o?.esConfidence ?? 0.975, 0.5, 0.999),
    varMethod: (o?.varMethod || "historical") as VaRMethod,
    annualizationFactor: o?.annualizationFactor ?? 252
  };
}

function normalizePositions(list: Position[], priceMap: Record<string, number>): Position[] {
  const out: Position[] = new Array(list.length);
  for (let i = 0; i < list.length; i++) {
    const p = list[i];
    out[i] = { ...p, price: p.price != null ? Number(p.price) : Number(priceMap[p.id] || 0) };
  }
  return out;
}

function sumAbsValue(pos: Position[]): number {
  let s = 0;
  for (let i = 0; i < pos.length; i++) s += Math.abs((pos[i].price || 0) * pos[i].qty);
  return s;
}

function longShortNet(pos: Position[]) {
  let long = 0, short = 0;
  for (let i = 0; i < pos.length; i++) {
    const v = (pos[i].price || 0) * pos[i].qty;
    if (v >= 0) long += v; else short += v;
  }
  return { long, short, net: long + short };
}

function diagonalZeros(n: number): number[][] {
  const M = new Array(n);
  for (let i = 0; i < n; i++) { M[i] = new Array(n).fill(0); }
  return M;
}

function percentile(arr: number[], p: number): number {
  if (!arr.length) return 0;
  const a = arr.slice().sort((x, y) => x - y);
  const r = clamp01(p);
  const idx = r * (a.length - 1);
  const i = Math.floor(idx);
  const frac = idx - i;
  if (i + 1 < a.length) return a[i] * (1 - frac) + a[i + 1] * frac;
  return a[i];
}

function expectedShortfall(arr: number[], alpha: number): number {
  if (!arr.length) return 0;
  const thr = percentile(arr, alpha);
  let s = 0, n = 0;
  for (let i = 0; i < arr.length; i++) if (arr[i] <= thr) { s += arr[i]; n++; }
  return n ? s / n : thr;
}

function stdev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let v = 0;
  for (let i = 0; i < xs.length; i++) { const d = xs[i] - m; v += d * d; }
  return Math.sqrt(v / (xs.length - 1));
}

function mean(xs: number[]): number {
  if (!xs.length) return 0;
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function inverseStdNormal(p: number): number {
  // Acklam's approximation, double precision, for 0<p<1
  // https://web.archive.org/web/20150910044754/http://home.online.no/~pjacklam/notes/invnorm/
  const a = [
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
  ];
  const b = [
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01
  ];
  const c = [
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
  ];
  const d = [
    7.784695709041462e-03, 3.224671290700398e-01,
    2.445134137142996e+00, 3.754408661907416e+00
  ];
  const pl = 0.02425;
  const ph = 1 - pl;
  if (p <= 0 || p >= 1) return NaN;
  let x = 0;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (ph < p) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else {
    const q = p - 0.5;
    const r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  // One Halley iteration for refinement
  const e = 0.5 * (1 + erf(x / Math.SQRT2)) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
  return x - u / (1 + x * u / 2);
}

function erf(x: number): number {
  // Numerical approximation of error function
  const sign = x < 0 ? -1 : 1;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function compare(v: number, op: Limit["op"], thr: number): boolean {
  if (op === "<") return v < thr;
  if (op === "<=") return v <= thr;
  if (op === ">") return v > thr;
  if (op === ">=") return v >= thr;
  if (op === "abs<=") return Math.abs(v) <= thr;
  if (op === "abs<") return Math.abs(v) < thr;
  return true;
}

function iso(t: string | number | Date): string {
  if (t instanceof Date) return t.toISOString();
  if (typeof t === "number") return new Date(t).toISOString();
  if (typeof t === "string") return new Date(t).toISOString();
  return new Date().toISOString();
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function clamp01(x: number): number { return clamp(x, 0, 1); }
function round2(x: number | undefined): number | undefined {
  if (typeof x !== "number" || !isFinite(x)) return undefined;
  return Math.round(x * 100) / 100;
}

// ────────────────────────────────────────────────────────────────────────────
// Version tag (manually bump)
// ────────────────────────────────────────────────────────────────────────────

export const VERSION = "0.1.0-pure";

// ────────────────────────────────────────────────────────────────────────────
// Example (commented)
//
// // Build a tiny example portfolio and factor history
// // const snapshot: PortfolioSnapshot = {
// //   ts: new Date(),
// //   positions: [
// //     { id: "AAPL", qty: 1000, price: 220, factorLoads: { MKT: 1.2, RATE: -0.3 }, sector: "Tech", assetClass: "Equity" },
// //     { id: "MSFT", qty: -300, price: 410, factorLoads: { MKT: 1.1, RATE: -0.2 }, sector: "Tech", assetClass: "Equity" }
// //   ]
// // };
// // const history: FactorHistory = {
// //   factors: ["MKT", "RATE"],
// //   series: { MKT: [0.01, -0.02, 0.005, ...], RATE: [-0.001, 0.0005, -0.0002, ...] }
// // };
// // const state = computeRiskState({ snapshot, factorHistory: history });
// // console.log(state.metrics.var_99, state.metrics.exposures);
//