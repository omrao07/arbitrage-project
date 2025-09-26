// fetchbondsserver.ts
// Zero-import, deterministic mock bond data generator (gov curve, quotes, spreads, history).
// Works server-side or edge. No external libs.
//
// Exports:
//   - types
//   - fetchBondSnapshot(params)
//   - handleGET(url) helper for route handlers

/* --------------------------------- Types --------------------------------- */

export type CurvePoint = {
  tenor: string;     // "2Y", "10Y", etc
  years: number;     // 2, 10, ...
  yield: number;     // decimal (0.0425 = 4.25%)
};

export type BondQuote = {
  symbol: string;        // e.g., "UST 10Y"
  isin?: string;         // mock ISIN-ish
  tenorYears: number;    // 10, 2, etc
  coupon: number;        // decimal (0.03 = 3%)
  maturity: string;      // ISO date
  cleanPrice: number;    // per 100 par
  ytm: number;           // decimal
  duration: number;      // Modified duration (approx)
  convexity: number;     // Convexity (approx)
  bid: number; ask: number;
  bidSize: number; askSize: number;
};

export type SpreadTable = {
  AAA: number; AA: number; A: number; BBB: number; BB: number; B: number;
};

export type HistoryPoint = { t: string; tenor: number; yield: number };

export type BondSnapshot = {
  country: string;       // "US", "GB", "DE", "IN", etc
  currency: string;      // "USD", "EUR", ...
  asOf: string;          // ISO timestamp
  govCurve: CurvePoint[];
  quotes: BondQuote[];
  spreads: SpreadTable;  // spread vs gov in decimal (0.012 = 120 bps)
  history: HistoryPoint[]; // trailing days for 2Y/10Y/30Y
};

export type BondParams = {
  country?: string;        // default "US"
  seed?: number;           // extra entropy
  tenors?: number[];       // years; default common set
  daysHistory?: number;    // default 90
};

/* --------------------------------- API ---------------------------------- */

export async function fetchBondSnapshot(params?: BondParams): Promise<BondSnapshot> {
  const p = withDefaults(params);
  const asOf = new Date();
  const rng = mulberry32(hash(`${p.country}|${p.seed}|${asOf.toISOString().slice(0,10)}`));

  // Base policy/ref level by country (very rough & deterministic)
  const cfg = countryConfig(p.country);
  const base = cfg.baseRate; // e.g., US ~5.25% policy anchor

  // Build gov curve (smooth-ish monotone with hump option)
  const govCurve = buildGovCurve(p.tenors, base, cfg.termSlope, cfg.curveHump, rng);

  // Build quotes (UST-like) around curve
  const quotes = govCurve.map((pt) => {
    const coupon = clamp(round((pt.yield + (rng() - 0.5) * 0.01), 4), 0.001, 0.15); // coupon ~ yld ±50bps
    const mat = addYears(asOf, pt.years);
    // Price from yield with semiannual, 30/360 approx
    const clean = round(priceFromYield(100, coupon, pt.yield, pt.years, 2), 3);
    const dur = round(modDuration(100, coupon, pt.yield, pt.years, 2), 3);
    const conv = round(convexity(100, coupon, pt.yield, pt.years, 2), 3);
    const mid = clean;
    const spr = 0.05 + rng() * 0.10; // 5–15 bps half-spread
    const bid = round(mid - spr, 3), ask = round(mid + spr, 3);
    return {
      symbol: `${cfg.govTicker} ${pt.tenor}`,
      isin: mockISIN(cfg.isinPrefix, pt.years),
      tenorYears: pt.years,
      coupon,
      maturity: mat.toISOString().slice(0, 10),
      cleanPrice: clean,
      ytm: pt.yield,
      duration: dur,
      convexity: conv,
      bid, ask,
      bidSize: 1_000 + Math.floor(rng() * 9_000),
      askSize: 1_000 + Math.floor(rng() * 9_000),
    } as BondQuote;
  });

  // Simple credit spread table (country-dependent bias)
  const spreads = makeSpreads(cfg.creditBias, rng);

  // History for selected tenors
  const historyTenors = [2, 10, 30].filter((t) => p.tenors.includes(t));
  const history = makeHistory(historyTenors, p.daysHistory, base, cfg.termSlope, cfg.curveHump, rng);

  return {
    country: p.country,
    currency: cfg.currency,
    asOf: asOf.toISOString(),
    govCurve,
    quotes,
    spreads,
    history,
  };
}

/**
 * Optional helper to expose via a Next.js Route Handler.
 * Example:
 *   export const dynamic = "force-dynamic";
 *   export async function GET(req: Request) { return handleGET(req.url); }
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const country = (url.searchParams.get("country") || "US").toUpperCase();
  const seed = num(url.searchParams.get("seed"));
  const days = num(url.searchParams.get("daysHistory"));
  const tenors = url.searchParams.getAll("tenor").map(Number).filter((x) => Number.isFinite(x));
  const snap = await fetchBondSnapshot({
    country,
    seed: seed ?? 0,
    daysHistory: days ?? undefined,
    tenors: tenors.length ? tenors : undefined,
  });
  return new Response(JSON.stringify(snap, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

/* ------------------------------- Generators ------------------------------ */

function buildGovCurve(tenors: number[], base: number, slope: number, hump: number, rng: () => number): CurvePoint[] {
  // Parametric: y(T) = base + slope * f1(T) + hump * f2(T) + noise
  // f1 increases with tenor (term premium), f2 bell-shaped around 5y–7y.
  return tenors
    .map((T) => {
      const f1 = Math.log(1 + T) / Math.log(11);           // 0 → 1 over [0,10+]
      const f2 = Math.exp(-Math.pow((T - 6) / 4, 2));      // hump near ~6y
      const noise = (rng() - 0.5) * 0.0008;                // ±8 bps jitter
      const y = clamp(base + slope * f1 + hump * f2 + noise, -0.01, 0.20);
      return {
        tenor: fmtTenor(T),
        years: T,
        yield: round(y, 6),
      };
    })
    .sort((a, b) => a.years - b.years);
}

function makeSpreads(bias: number, rng: () => number): SpreadTable {
  const base = bias; // country credit environment (e.g., DM ~ 100bps BBB)
  const mk = (x: number) => round(Math.max(0, base + x + (rng() - 0.5) * 0.001), 6);
  return {
    AAA: mk(-0.006),
    AA:  mk(-0.004),
    A:   mk(-0.002),
    BBB: mk(+0.000),
    BB:  mk(+0.008),
    B:   mk(+0.016),
  };
}

function makeHistory(tenors: number[], days: number, base: number, slope: number, hump: number, rng: () => number): HistoryPoint[] {
  const now = new Date();
  const out: HistoryPoint[] = [];
  // Small random walk around parametric level, correlated across tenors
  let shock = 0;
  for (let d = days - 1; d >= 0; d--) {
    const date = new Date(now.getTime() - d * 24 * 3600 * 1000);
    shock = shock * 0.95 + (rng() - 0.5) * 0.0007; // AR(1) drift
    for (const T of tenors) {
      const f1 = Math.log(1 + T) / Math.log(11);
      const f2 = Math.exp(-Math.pow((T - 6) / 4, 2));
      const level = base + slope * f1 + hump * f2 + shock + (rng() - 0.5) * 0.0004;
      out.push({ t: date.toISOString().slice(0, 10), tenor: T, yield: round(clamp(level, -0.01, 0.2), 6) });
    }
  }
  return out;
}

/* --------------------------- Pricing & Risk (SA) -------------------------- */
/** Semiannual coupon bond math with simple day count (30/360), clean price. */

function priceFromYield(par: number, cpn: number, ytm: number, years: number, freq: 1 | 2 = 2): number {
  const n = Math.max(1, Math.round(years * freq));
  const i = ytm / freq;
  const c = cpn * par / freq;
  let pv = 0;
  for (let k = 1; k <= n; k++) pv += c / Math.pow(1 + i, k);
  pv += par / Math.pow(1 + i, n);
  return pv; // clean (ignoring accrual for simplicity)
}

function yieldFromPrice(par: number, cpn: number, price: number, years: number, freq: 1 | 2 = 2): number {
  // Newton-Raphson with fallback bisection
  let y = cpn + 0.005; // start near coupon
  for (let it = 0; it < 20; it++) {
    const f = priceFromYield(par, cpn, y, years, freq) - price;
    const df = (priceFromYield(par, cpn, y + 1e-5, years, freq) - priceFromYield(par, cpn, y - 1e-5, years, freq)) / 2e-5;
    if (Math.abs(df) < 1e-8) break;
    const step = f / df;
    y -= clamp(step, -0.05, 0.05);
    if (Math.abs(f) < 1e-6) break;
  }
  return clamp(y, -0.01, 0.30);
}

function modDuration(par: number, cpn: number, ytm: number, years: number, freq: 1 | 2 = 2): number {
  // Modified duration (approx) via cashflow formula
  const n = Math.max(1, Math.round(years * freq));
  const i = ytm / freq;
  const c = cpn * par / freq;
  const pv = priceFromYield(par, cpn, ytm, years, freq);
  let D = 0;
  for (let k = 1; k <= n; k++) {
    const t = k / freq;
    const cf = (k === n ? c + par : c);
    D += (t * cf) / Math.pow(1 + i, k);
  }
  const macaulay = D / pv;
  return macaulay / (1 + i);
}

function convexity(par: number, cpn: number, ytm: number, years: number, freq: 1 | 2 = 2): number {
  const n = Math.max(1, Math.round(years * freq));
  const i = ytm / freq;
  const c = cpn * par / freq;
  let C = 0;
  for (let k = 1; k <= n; k++) {
    const t = k / freq;
    const cf = (k === n ? c + par : c);
    C += (cf * t * (t + 1 / freq)) / Math.pow(1 + i, k + 2);
  }
  return C;
}

/* ------------------------------ Config/Utils ----------------------------- */

function withDefaults(p?: BondParams) {
  return {
    country: (p?.country || "US").toUpperCase(),
    seed: p?.seed ?? 0,
    tenors: p?.tenors ?? [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    daysHistory: p?.daysHistory ?? 90,
  };
}

function countryConfig(country: string) {
  // Very rough macro presets
  switch (country) {
    case "US": return { currency: "USD", govTicker: "UST", isinPrefix: "US", baseRate: 0.0525, termSlope: -0.004, curveHump: 0.0025, creditBias: 0.010 };
    case "GB": return { currency: "GBP", govTicker: "UKT", isinPrefix: "GB", baseRate: 0.0525, termSlope: -0.003, curveHump: 0.0020, creditBias: 0.011 };
    case "DE": return { currency: "EUR", govTicker: "DBR", isinPrefix: "DE", baseRate: 0.0375, termSlope: -0.002, curveHump: 0.0015, creditBias: 0.009 };
    case "JP": return { currency: "JPY", govTicker: "JGB", isinPrefix: "JP", baseRate: 0.0050, termSlope: +0.001, curveHump: 0.0005, creditBias: 0.012 };
    case "IN": return { currency: "INR", govTicker: "GS",  isinPrefix: "IN", baseRate: 0.0660, termSlope: +0.001, curveHump: 0.0020, creditBias: 0.018 };
    default:   return { currency: "USD", govTicker: country + "T", isinPrefix: country.slice(0,2), baseRate: 0.05, termSlope: -0.002, curveHump: 0.002, creditBias: 0.012 };
  }
}

function mockISIN(prefix: string, years: number) {
  const body = Math.abs(hash(prefix + years)).toString().slice(0, 9).padStart(9, "0");
  const chk = (body.split("").reduce((s, d) => s + (d.charCodeAt(0) % 9), 0) % 10).toString();
  return `${prefix}${body}${chk}`.slice(0, 12);
}

function addYears(d: Date, years: number) {
  const dt = new Date(d);
  dt.setFullYear(dt.getFullYear() + Math.floor(years));
  // push remaining months (e.g., 0.5Y → 6m)
  const months = Math.round((years - Math.floor(years)) * 12);
  dt.setMonth(dt.getMonth() + months);
  return dt;
}

function fmtTenor(y: number) {
  if (y < 1) return `${Math.round(y * 12)}M`;
  return `${Math.round(y)}Y`;
}

function num(s: string | null) { if (s == null) return undefined; const n = Number(s); return Number.isFinite(n) ? n : undefined; }
function round(n: number, d = 6) { const p = 10 ** d; return Math.round(n * p) / p; }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }

// Deterministic RNG
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}
function hash(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

/* --------------------------------- Notes ---------------------------------
- This module synthesizes plausible curves/quotes only. Do not use for trading.
- Price/yield math assumes clean price, semiannual coupons, simple accrual ignored.
- Duration/convexity are approximations to keep things fast and dependency-free.
--------------------------------------------------------------------------- */