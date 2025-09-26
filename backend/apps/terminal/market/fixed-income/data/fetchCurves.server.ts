// fetchcurves.server.ts
// Zero-import, deterministic mock government yield curves (multi-country).
// Works server-side or edge. No external libs.
//
// Exports:
//  - types (CurvePoint, CurveSeries, CurveSnapshot)
//  - fetchCurves(params)
//  - handleGET(url) helper for use in a route handler

/* --------------------------------- Types --------------------------------- */

export type CurvePoint = {
  tenor: string;   // "3M","2Y","10Y","30Y"
  years: number;   // numeric tenor in years
  yield: number;   // decimal (0.0425 = 4.25%)
};

export type CurveSeries = {
  country: string;     // "US","GB","DE","JP","IN", etc.
  currency: string;    // "USD","GBP","EUR","JPY","INR", ...
  asOf: string;        // ISO timestamp
  points: CurvePoint[]; // sorted by years asc
};

export type HistoryPoint = {
  t: string;           // ISO date (YYYY-MM-DD)
  years: number;       // tenor
  yield: number;       // decimal
  country: string;
};

export type CurveSnapshot = {
  asOf: string;
  tenors: number[];
  countries: string[];
  series: CurveSeries[];
  history: HistoryPoint[];   // compact history for requested tenors across countries
  meta: Record<string, { currency: string }>;
};

export type CurveParams = {
  countries?: string[];     // defaults ["US","GB","DE","JP","IN"]
  tenors?: number[];        // defaults [0.25,0.5,1,2,3,5,7,10,20,30]
  daysHistory?: number;     // defaults 120
  seed?: number;            // optional extra entropy
  model?: "param" | "nss";  // curve model (param = simple; nss = Nelson-Siegel-Svensson-ish)
};

/* ---------------------------------- API ---------------------------------- */

export async function fetchCurves(p?: CurveParams): Promise<CurveSnapshot> {
  const cfg = withDefaults(p);
  const asOf = new Date();
  const series: CurveSeries[] = [];
  const hist: HistoryPoint[] = [];
  const meta: Record<string, { currency: string }> = {};

  for (const c of cfg.countries) {
    const cc = countryConfig(c);
    meta[c] = { currency: cc.currency };

    // Build curve
    const rng = mulberry32(hash(`${c}|${cfg.seed}|${asOf.toISOString().slice(0,10)}`));
    const pts =
      cfg.model === "nss"
        ? buildNSSCurve(cfg.tenors, cc.baseRate, cc.termSlope, cc.curveHump, rng)
        : buildParamCurve(cfg.tenors, cc.baseRate, cc.termSlope, cc.curveHump, rng);

    series.push({
      country: c,
      currency: cc.currency,
      asOf: asOf.toISOString(),
      points: pts,
    });

    // History
    const hRng = mulberry32(hash(`${c}|hist|${cfg.seed}`));
    hist.push(
      ...makeHistory(c, cfg.tenors, cfg.daysHistory, cc.baseRate, cc.termSlope, cc.curveHump, hRng, cfg.model)
    );
  }

  return {
    asOf: asOf.toISOString(),
    tenors: cfg.tenors.slice(),
    countries: cfg.countries.slice(),
    series,
    history: hist,
    meta,
  };
}

/**
 * Optional helper for Next.js Route Handler.
 * Example:
 *   export const dynamic = "force-dynamic";
 *   export async function GET(req: Request) { return handleGET(req.url); }
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const countries = (url.searchParams.getAll("country").length
    ? url.searchParams.getAll("country")
    : (url.searchParams.get("countries") || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
  ).map((s) => s.toUpperCase());
  const tenors = url.searchParams
    .getAll("tenor")
    .map(Number)
    .filter((x) => Number.isFinite(x));
  const daysHistory = num(url.searchParams.get("daysHistory"));
  const seed = num(url.searchParams.get("seed"));
  const model = (url.searchParams.get("model") as "param" | "nss") || undefined;

  const snap = await fetchCurves({
    countries: countries.length ? countries : undefined,
    tenors: tenors.length ? tenors : undefined,
    daysHistory: daysHistory ?? undefined,
    seed: seed ?? undefined,
    model,
  });

  return new Response(JSON.stringify(snap, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

/* -------------------------------- Builders -------------------------------- */

function buildParamCurve(
  tenors: number[],
  base: number,
  slope: number,
  hump: number,
  rng: () => number
): CurvePoint[] {
  // Smooth parametric level: base + slope*log(1+T)/log(11) + hump*exp(-((T-6)/4)^2) + noise
  return tenors
    .map((T) => {
      const f1 = Math.log(1 + T) / Math.log(11);     // term premium proxy
             // belly hump near 6Y
      const noise = (rng() - 0.5) * 0.0009;          // ±9 bps jitter
      const y = clamp(base + slope * f1 + hump *  + noise, -0.01, 0.20);
      return { tenor: fmtTenor(T), years: T, yield: round(y, 6) };
    })
    .sort((a, b) => a.years - b.years);
}

function buildNSSCurve(
  tenors: number[],
  base: number,
  slope: number,
  hump: number,
  rng: () => number
): CurvePoint[] {
  // Lightweight NSS-like: y(T) = beta0 + beta1 * ((1 - e^(-T/t1)) / (T/t1)) + beta2 * [((1 - e^(-T/t1)) / (T/t1)) - e^(-T/t1)]
  // plus a small long-term factor beta3 * [((1 - e^(-T/t2)) / (T/t2)) - e^(-T/t2)]
  const beta0 = base;
  const beta1 = slope * 1.4;
  const beta2 = hump * 3.0;
  const beta3 = hump * 1.2;
  const t1 = 2.5 + (rng() - 0.5) * 0.6;
  const t2 = 8.0 + (rng() - 0.5) * 1.5;

  const f = (T: number) => {
    const f1 = (1 - Math.exp(-T / t1)) / (T / t1);
    const f2 = f1 - Math.exp(-T / t1);
    const f3 = (1 - Math.exp(-T / t2)) / (T / t2) - Math.exp(-T / t2);
    const eps = (rng() - 0.5) * 0.0007; // ±7 bps
    return clamp(beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3 + eps, -0.01, 0.20);
  };

  return tenors
    .map((T) => ({ tenor: fmtTenor(T), years: T, yield: round(f(T), 6) }))
    .sort((a, b) => a.years - b.years);
}

function makeHistory(
  country: string,
  tenors: number[],
  days: number,
  base: number,
  slope: number,
  hump: number,
  rng: () => number,
  model: "param" | "nss"
): HistoryPoint[] {
  const out: HistoryPoint[] = [];
  const today = new Date();
  // Shared shock process so tenors move together but not identically
  let shock = 0;
  for (let d = days - 1; d >= 0; d--) {
    const date = new Date(today.getTime() - d * 24 * 3600 * 1000);
    shock = shock * 0.96 + (rng() - 0.5) * 0.0006;
    const rngD = mulberry32(hash(`${country}|${d}`));
    for (const T of tenors) {
      const eps = (rngD() - 0.5) * 0.0004;
      let y: number;
      if (model === "nss") {
        // reuse the nss structure with slow drift
        const t1 = 2.5, t2 = 8.0;
        const f1 = (1 - Math.exp(-T / t1)) / (T / t1);
        const f2 = f1 - Math.exp(-T / t1);
        const f3 = (1 - Math.exp(-T / t2)) / (T / t2) - Math.exp(-T / t2);
        const beta0 = base, beta1 = slope * 1.4, beta2 = hump * 3.0, beta3 = hump * 1.2;
        y = beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3 + shock + eps;
      } else {
        const f1 = Math.log(1 + T) / Math.log(11);
        
       
        y = base + slope * f1 + hump *  + shock + eps;
      }
      out.push({ t: date.toISOString().slice(0, 10), years: T, yield: round(clamp(y, -0.01, 0.2), 6), country });
    }
  }
  return out;
}

/* --------------------------------- Config -------------------------------- */

function withDefaults(p?: CurveParams) {
  return {
    countries: (p?.countries && p.countries.length ? dedupe(p.countries.map((x) => x.toUpperCase())) : ["US", "GB", "DE", "JP", "IN"]) as string[],
    tenors: p?.tenors?.length ? [...p.tenors].sort((a, b) => a - b) : [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
    daysHistory: p?.daysHistory ?? 120,
    seed: p?.seed ?? 0,
    model: p?.model ?? "param",
  };
}

function countryConfig(country: string) {
  switch (country.toUpperCase()) {
    case "US": return { currency: "USD", baseRate: 0.0525, termSlope: -0.0035, curveHump: 0.0024 };
    case "GB": return { currency: "GBP", baseRate: 0.0525, termSlope: -0.0028, curveHump: 0.0021 };
    case "DE": return { currency: "EUR", baseRate: 0.0375, termSlope: -0.0020, curveHump: 0.0014 };
    case "JP": return { currency: "JPY", baseRate: 0.0050, termSlope: +0.0010, curveHump: 0.0006 };
    case "IN": return { currency: "INR", baseRate: 0.0660, termSlope: +0.0012, curveHump: 0.0020 };
    case "CA": return { currency: "CAD", baseRate: 0.0500, termSlope: -0.0025, curveHump: 0.0018 };
    case "AU": return { currency: "AUD", baseRate: 0.0425, termSlope: -0.0018, curveHump: 0.0015 };
    default:   return { currency: "USD", baseRate: 0.05,   termSlope: -0.0020, curveHump: 0.0018 };
  }
}

/* --------------------------------- Utils --------------------------------- */

function fmtTenor(y: number) { return y < 1 ? `${Math.round(y * 12)}M` : `${Math.round(y)}Y`; }
function dedupe<T>(a: T[]) { return Array.from(new Set(a)); }
function round(n: number, d = 6) { const p = 10 ** d; return Math.round(n * p) / p; }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
function num(s: string | null) { if (s == null) return undefined; const n = Number(s); return Number.isFinite(n) ? n : undefined; }

// Deterministic RNG & hash
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
- These curves are synthetic and for UI/testing only. Not for pricing/trading.
- "param" model is simple, fast, and monotone-ish; "nss" is smoother in the belly.
- History is generated via a low-variance AR(1) shock; tenors are correlated.
--------------------------------------------------------------------------- */