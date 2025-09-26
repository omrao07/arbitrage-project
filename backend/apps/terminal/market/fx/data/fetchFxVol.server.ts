// fetchfxvol.server.ts
// Zero-import, deterministic mock FX implied volatility generator.
// Works on server or edge. No external libs.
//
// Exports:
//  - types
//  - fetchFxVol(params)
//  - handleGET(url) helper for Next.js Route Handlers
//
// Notes:
//  * Generates a vanilla delta-smile surface per pair across standard tenors
//  * Tenors in days; deltas as absolute (e.g., 0.10, 0.25, 0.5 ATM, 0.75, 0.90)
//  * History for ATM vol term (e.g., 1M ATM) via gentle AR(1) walk
//  * All numbers are synthetic — for UI/testing only

/* --------------------------------- Types --------------------------------- */

export type FxVolPoint = {
  tenorDays: number; // e.g., 7, 30, 90, 180, 365
  delta: number;     // 0.10..0.90 (0.50 = ATM)
  vol: number;       // decimal (0.12 = 12%)
};

export type FxVolSurface = {
  pair: string;           // "EURUSD"
  asOf: string;           // ISO timestamp
  deltas: number[];       // sorted
  tenors: number[];       // sorted (days)
  points: FxVolPoint[];   // grid (tenors x deltas)
  params: {
    atm: Record<number, number>; // by tenorDays
    rr25: Record<number, number>; // 25Δ risk reversal (call - put)
    bf25: Record<number, number>; // 25Δ butterfly
  };
};

export type FxVolHistoryPoint = {
  t: string;   // ISO date
  pair: string;
  tenorDays: number;
  delta: number;
  vol: number;
};

export type FxVolSnapshot = {
  asOf: string;
  surfaces: FxVolSurface[];
  history: FxVolHistoryPoint[]; // ATM 1M history for each pair
};

export type FxVolParams = {
  pairs?: string[];        // e.g., ["EURUSD","USDJPY"]
  tenors?: number[];       // days; default common set
  deltas?: number[];       // default [0.1,0.25,0.5,0.75,0.9]
  daysHistory?: number;    // default 120
  seed?: number;           // extra entropy
};

/* ---------------------------------- API ---------------------------------- */

export async function fetchFxVol(p?: FxVolParams): Promise<FxVolSnapshot> {
  const cfg = withDefaults(p);
  const asOf = new Date();
  const surfaces: FxVolSurface[] = [];

  for (const pr of cfg.pairs) {
    const pair = normalizePair(pr);
    if (!pair) continue;

    // Pair “character” influences atm level, smile skew, and term slope
    const char = pairCharacter(pair);
    const dayKey = asOf.toISOString().slice(0, 10);
    const rng = mulberry32(hash(`vol|${pair}|${cfg.seed}|${dayKey}`));

    // Base ATM term structure (days -> vol)
    const atm: Record<number, number> = {};
    for (const T of cfg.tenors) {
      const yr = Math.max(1 / 365, T / 365);
      const termSlope = char.termSlope;        // +steeper for EM, flatter for majors
      const base = char.baseATM;               // pair-level base
      const shape = base * (1 + termSlope * Math.log(1 + yr * 4));
      const jitter = (rng() - 0.5) * 0.01;     // ±1 vol pt
      atm[T] = clamp(shape + jitter, 0.02, 1.20);
    }

    // Smile params by tenor: rr25, bf25
    const rr25: Record<number, number> = {};
    const bf25: Record<number, number> = {};
    for (const T of cfg.tenors) {
      const skew = char.skewBase * (1 + (rng() - 0.5) * 0.25);  // bias per pair
      const smile = char.bfBase  * (1 + (rng() - 0.5) * 0.30);
      // Short tenors typically have a bit more skew noise
      const scale = 1 + (30 / (T + 30));
      rr25[T] = clamp(skew * scale, -0.10, 0.10); // ±10 vol pts (extreme)
      bf25[T] = clamp(Math.abs(smile) * (0.7 + 0.6 * rng()), 0.000, 0.15);
    }

    // Build full grid with SABR-lite / market-ish decomposition:
    //  v(δ) ≈ ATM + 0.5 * BF25 * smileShape(δ) + 0.5 * RR25 * skewShape(δ)
    const pts: FxVolPoint[] = [];
    for (const T of cfg.tenors) {
      for (const d of cfg.deltas) {
        const v =
          atm[T] +
          0.5 * bf25[T] * butterflyShape(d) +
          0.5 * rr25[T] * skewShape(d);
        pts.push({ tenorDays: T, delta: d, vol: clamp(v, 0.02, 2.0) });
      }
    }

    surfaces.push({
      pair,
      asOf: asOf.toISOString(),
      deltas: cfg.deltas.slice(),
      tenors: cfg.tenors.slice(),
      points: pts,
      params: { atm, rr25, bf25 },
    });
  }

  // History: ATM 1M (30d) path per pair
  const hist: FxVolHistoryPoint[] = [];
  for (const pr of cfg.pairs) {
    const pair = normalizePair(pr);
    if (!pair) continue;
    const rng = mulberry32(hash(`hist|${pair}|${cfg.seed}`));
    const base = pairCharacter(pair).baseATM;
    let lvl = base;
    let drift = 0;
    for (let i = cfg.daysHistory - 1; i >= 0; i--) {
      const d = new Date(asOf.getTime() - i * 86400000);
      drift = 0.92 * drift + (rng() - 0.5) * 0.002; // slow drift
      lvl = clamp(lvl * Math.exp((rng() - 0.5) * 0.02 + drift), 0.02, 1.5);
      hist.push({
        t: d.toISOString().slice(0, 10),
        pair,
        tenorDays: 30,
        delta: 0.5,
        vol: round(lvl, 4),
      });
    }
  }

  return { asOf: asOf.toISOString(), surfaces, history: hist };
}

/**
 * Optional helper for Next.js Route Handlers:
 *   export const dynamic = "force-dynamic";
 *   export async function GET(req: Request) { return handleGET(req.url); }
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const pairs =
    url.searchParams.getAll("pair").length
      ? url.searchParams.getAll("pair")
      : (url.searchParams.get("pairs") || "")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
  const tenors = (url.searchParams.get("tenors") || "")
    .split(",")
    .map((s) => +s.trim())
    .filter((n) => Number.isFinite(n) && n > 0);
  const deltas = (url.searchParams.get("deltas") || "")
    .split(",")
    .map((s) => +s.trim())
    .filter((n) => Number.isFinite(n) && n > 0 && n < 1);
  const daysHistory = num(url.searchParams.get("daysHistory"));
  const seed = num(url.searchParams.get("seed"));

  const snap = await fetchFxVol({
    pairs: pairs.length ? pairs : undefined,
    tenors: tenors.length ? tenors : undefined,
    deltas: deltas.length ? deltas : undefined,
    daysHistory: daysHistory ?? undefined,
    seed: seed ?? undefined,
  });

  return new Response(JSON.stringify(snap, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

/* ----------------------------- Market Shapes ----------------------------- */

// Symmetric “smile” shape around ATM using deltas in [0,1]; 0.5 is ATM.
function butterflyShape(delta: number) {
  const x = clamp(delta, 0.0001, 0.9999);
  // Quadratic bowl with min at ATM, normalized so BF impacts 25Δs most.
  // f(0.25) ≈ f(0.75) ≈ 1, f(0.5)=0
  const c = 16; // curvature
  return c * (x - 0.5) ** 2;
}

// Skew shape: negative for puts (delta < 0.5), positive for calls (delta > 0.5).
// Normalized so RR25 adds/subtracts roughly at 25Δ/75Δ.
function skewShape(delta: number) {
  const x = clamp(delta, 0.0001, 0.9999);
  // Smooth odd function: tanh-like around ATM
  return Math.tanh((x - 0.5) * 4);
}

/* --------------------------- Pair Characteristics -------------------------- */

function pairCharacter(pair: string) {
  // Simple heuristics: majors lower ATM; JPY/crosses a bit higher skew.
  const p = pair.toUpperCase();
  if (/^USDJPY$/.test(p)) return { baseATM: 0.115, skewBase: -0.010, bfBase: 0.020, termSlope: 0.10 };
  if (/^EURUSD$/.test(p)) return { baseATM: 0.095, skewBase: -0.006, bfBase: 0.018, termSlope: 0.06 };
  if (/^GBPUSD$/.test(p)) return { baseATM: 0.105, skewBase: -0.008, bfBase: 0.020, termSlope: 0.08 };
  if (/^AUDUSD$/.test(p)) return { baseATM: 0.115, skewBase: -0.004, bfBase: 0.022, termSlope: 0.10 };
  if (/^USDCAD$/.test(p)) return { baseATM: 0.110, skewBase: +0.003, bfBase: 0.018, termSlope: 0.09 };
  if (/^NZDUSD$/.test(p)) return { baseATM: 0.120, skewBase: -0.003, bfBase: 0.022, termSlope: 0.10 };
  if (/^EURGBP$/.test(p)) return { baseATM: 0.070, skewBase: -0.004, bfBase: 0.015, termSlope: 0.05 };
  if (/^EURJPY$/.test(p)) return { baseATM: 0.110, skewBase: -0.012, bfBase: 0.022, termSlope: 0.10 };
  if (/^GBPJPY$/.test(p)) return { baseATM: 0.125, skewBase: -0.015, bfBase: 0.026, termSlope: 0.12 };
  if (/^USDINR$/.test(p)) return { baseATM: 0.065, skewBase: +0.004, bfBase: 0.012, termSlope: 0.14 };
  if (/^USDCNH$/.test(p)) return { baseATM: 0.070, skewBase: +0.005, bfBase: 0.012, termSlope: 0.16 };
  // default cross
  return { baseATM: 0.10, skewBase: -0.006, bfBase: 0.018, termSlope: 0.09 };
}

/* --------------------------------- Defaults -------------------------------- */

function withDefaults(p?: FxVolParams) {
  return {
    pairs: (p?.pairs?.length ? p.pairs : ["EURUSD", "USDJPY", "GBPUSD"]) as string[],
    tenors: p?.tenors?.length ? sortNums(p.tenors) : [7, 14, 30, 60, 90, 180, 365],
    deltas: p?.deltas?.length ? sortNums(p.deltas) : [0.10, 0.25, 0.50, 0.75, 0.90],
    daysHistory: p?.daysHistory ?? 120,
    seed: p?.seed ?? 0,
  };
}

/* ---------------------------------- Utils ---------------------------------- */

function normalizePair(p: string) {
  const s = (p || "").toUpperCase().replace(/[^A-Z]/g, "");
  return s.length === 6 ? s : "";
}
function sortNums(a: number[]) { return [...a].sort((x, y) => x - y); }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
function round(n: number, d = 6) { const p = 10 ** d; return Math.round(n * p) / p; }
function num(s: string | null) { if (s == null) return undefined; const n = Number(s); return Number.isFinite(n) ? n : undefined; }

/* ------------------------ Deterministic RNG & Hash ------------------------- */

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