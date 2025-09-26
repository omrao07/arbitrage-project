// macro/currencybasis.ts
// Import-free helpers for Cross-Currency Basis (XCCY basis) and CIP checks.
//
// What this file gives you
// - tenorToYears("3M") → 0.25
// - cipForward(spot, rBase, rQuote, tenorYears) → theoretical forward
// - basisFromForward(spot, fwd, rBase, rQuote, tenorYears) → annualized basis
// - impliedRateFromFwd({spot, fwd, rBase, tenorYears, side}) → quote/base implied
// - buildBasisCurve({pair, spot, forwards, rates}) → term structure in bps
// - detectDeviations(curve, {absBp, z}) → flags large CIP breaches
//
// Conventions
// - Pair is BASEQUOTE (e.g., EURUSD). rBase is the short rate for BASE ccy,
//   rQuote is the short rate for QUOTE ccy. Rates are simple annual (ACT/365
//   or ACT/360 consistent across inputs).
// - Basis definition here is the annualized additive spread that must be added
//   to the QUOTE rate to make CIP hold: F = S * (1 + (rBase - (rQuote + b)) * T).
//   Positive “b” = quote currency scarcity (USD scarcity in EURUSD if b>0).
//
// Units
// - Rates: decimals (0.05 = 5%)
// - Basis: returned both in decimal and basis points (bp = 1e-4)

export type Tenor = "1W" | "2W" | "1M" | "2M" | "3M" | "6M" | "9M" | "1Y" | string;

export type RateCurve = {
  ccy: string;                                      // e.g., "USD"
  points: Array<{ tenor: Tenor; rate: number }>;    // simple annualized
};

export type BasisPoint = {
  tenor: Tenor;
  T: number;                // years
  spot: number;             // S
  fwd: number;              // outright F
  rBase: number;            // rate(base)
  rQuote: number;           // rate(quote)
  basis: number;            // decimal p.a.
  basisBp: number;          // basis * 10_000
  fwdPoints: number;        // F - S
  cipTheo: number;          // S*(1+(rBase-rQuote)*T)
  breachBp: number;         // (F - CIP_theo) / (S*T) in bp, sign aligned with 'basis'
};

export type BasisCurve = {
  pair: string;
  items: BasisPoint[];
  summary: {
    avgBp: number;
    minBp: number;
    maxBp: number;
    stdevBp: number;
  };
};

export function tenorToYears(t: Tenor): number {
  const m = String(t).trim().toUpperCase().match(/^(\d+)([DWMY])$/);
  if (!m) return 0;
  const n = Number(m[1]);
  switch (m[2]) {
    case "D": return n / 365;
    case "W": return (7 * n) / 365;
    case "M": return n / 12;
    case "Y": return n;
    default: return 0;
  }
}

// Covered Interest Parity theoretical forward
export function cipForward(spot: number, rBase: number, rQuote: number, T: number): number {
  return spot * (1 + (rBase - rQuote) * T);
}

// From an observed forward, infer the annualized XCCY basis “b” applied to QUOTE leg
// such that: F = S * (1 + (rBase - (rQuote + b)) * T)
export function basisFromForward(spot: number, fwd: number, rBase: number, rQuote: number, T: number) {
  if (!isFiniteInputs([spot, fwd, rBase, rQuote, T]) || spot <= 0 || T <= 0) return { basis: 0, basisBp: 0 };
  const lhs = fwd / spot - 1;            // ≈ (rBase - (rQuote + b)) * T
  const b = rBase - rQuote - lhs / T;    // annualized basis on QUOTE side
  return { basis: b, basisBp: b * 10_000 };
}

// Given a forward, back out the implied rate for either leg
// side = "quote" returns implied rQuote; side = "base" returns implied rBase.
export function impliedRateFromFwd(args: {
  spot: number; fwd: number; rBase?: number; rQuote?: number; T: number; side: "quote" | "base";
}) {
  const { spot, fwd, rBase = 0, rQuote = 0, T, side } = args;
  if (!isFiniteInputs([spot, fwd, rBase, rQuote, T]) || spot <= 0 || T <= 0) return NaN;
  const lhs = fwd / spot - 1; // (rBase - rQuote_eff) * T
  if (side === "quote") return rBase - lhs / T;
  return rQuote + lhs / T;
}

// Build a term structure of basis points across tenors.
// Inputs:
// - pair: "EURUSD" (BASEQUOTE)
// - spot: number
// - forwards: map tenor->forward outright (e.g., {"1M": 1.0912, "3M": 1.0950})
// - rates: { base: rate or curve, quote: rate or curve } (numbers = flat)
export function buildBasisCurve(input: {
  pair: string;
  spot: number;
  forwards: Record<string, number>;
  rates: { base: number | RateCurve; quote: number | RateCurve };
}): BasisCurve {
  const pair = input.pair.toUpperCase();
  const spot = Number(input.spot);
  const tenors = Object.keys(input.forwards || {}).sort(sortTenors);
  const items: BasisPoint[] = [];

  for (let i = 0; i < tenors.length; i++) {
    const t = tenors[i];
    const T = tenorToYears(t);
    const fwd = Number((input.forwards as any)[t]);
    const rB = rateAt(input.rates.base, t);
    const rQ = rateAt(input.rates.quote, t);
    const theo = cipForward(spot, rB, rQ, T);
    const { basis, basisBp } = basisFromForward(spot, fwd, rB, rQ, T);
    const breach = ((fwd - theo) / (spot * T)) * 10_000; // bp deviation from pure CIP
    items.push({
      tenor: t, T, spot, fwd,
      rBase: rB, rQuote: rQ,
      basis, basisBp,
      fwdPoints: fwd - spot,
      cipTheo: theo,
      breachBp: breach,
    });
  }

  const arr = items.map(x => x.basisBp);
  const summary = {
    avgBp: round(mean(arr)),
    minBp: round(min(arr)),
    maxBp: round(max(arr)),
    stdevBp: round(stdev(arr)),
  };

  return { pair, items, summary };
}

// Simple deviation detector: flag tenors where |breach| > absBp OR z-score > z
export function detectDeviations(curve: BasisCurve, opts: { absBp?: number; z?: number } = {}) {
  const absBp = Number(opts.absBp ?? 10); // default 10 bp
  const z = Number(opts.z ?? 3);
  const xs = curve.items.map(x => x.breachBp);
  const m = mean(xs), s = stdev(xs) || 1e-9;
  const flags = [];
  for (let i = 0; i < curve.items.length; i++) {
    const b = curve.items[i].breachBp;
    const zi = (b - m) / s;
    if (Math.abs(b) >= absBp || Math.abs(zi) >= z) {
      flags.push({ tenor: curve.items[i].tenor, breachBp: round(b), z: round(zi) });
    }
  }
  return { ok: flags.length === 0, flags, meanBp: round(m), stdevBp: round(s) };
}

/* ------------------------------- Utilities ------------------------------- */

function rateAt(r: number | RateCurve, tenor: Tenor): number {
  if (typeof r === "number") return r;
  // nearest tenor (by year distance)
  const T = tenorToYears(tenor);
  let best = 0, bestDiff = Infinity;
  for (let i = 0; i < r.points.length; i++) {
    const Ti = tenorToYears(r.points[i].tenor);
    const d = Math.abs(Ti - T);
    if (d < bestDiff) { bestDiff = d; best = r.points[i].rate; }
  }
  return best;
}

function sortTenors(a: string, b: string) {
  return tenorToYears(a as Tenor) - tenorToYears(b as Tenor);
}

function isFiniteInputs(xs: any[]) {
  for (let i = 0; i < xs.length; i++) if (!Number.isFinite(Number(xs[i]))) return false;
  return true;
}

/* ------------------------------- Math bits ------------------------------- */

function mean(a: number[]) { if (a.length === 0) return 0; let s = 0, n = 0; for (let i=0;i<a.length;i++){ const v=a[i]; if (isFinite(v)) { s+=v; n++; } } return n? s/n:0; }
function stdev(a: number[]) { const m=mean(a); let s=0,n=0; for (let i=0;i<a.length;i++){ const v=a[i]; if (isFinite(v)){ const d=v-m; s+=d*d; n++; } } return n>1? Math.sqrt(s/(n-1)):0; }
function min(a: number[]) { let m=Infinity; for (let i=0;i<a.length;i++){ const v=a[i]; if (isFinite(v) && v<m) m=v; } return m===Infinity?0:m; }
function max(a: number[]) { let m=-Infinity; for (let i=0;i<a.length;i++){ const v=a[i]; if (isFinite(v) && v>m) m=v; } return m===-Infinity?0:m; }
function round(n: number) { return Math.round(n * 100) / 100; }

/* ----------------------------- Tiny example ------------------------------ */
// const curve = buildBasisCurve({
//   pair: "EURUSD",
//   spot: 1.0850,
//   forwards: { "1M": 1.0862, "3M": 1.0890, "6M": 1.0925, "1Y": 1.1000 },
//   rates: {
//     base: { ccy: "EUR", points: [{tenor:"1M",rate:0.037},{tenor:"3M",rate:0.0375},{tenor:"6M",rate:0.038},{tenor:"1Y",rate:0.0385}] },
//     quote:{ ccy: "USD", points: [{tenor:"1M",rate:0.052},{tenor:"3M",rate:0.0515},{tenor:"6M",rate:0.051},{tenor:"1Y",rate:0.0505}] },
//   },
// });
// console.log(curve, detectDeviations(curve, { absBp: 5, z: 2.5 }));