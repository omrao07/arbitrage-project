// fetchfxspot.server.ts (clean rewrite)
// Zero-import, deterministic mock FX spot + history generator.
// Works on server or edge. No external libs.
//
// Exports:
//   - types
//   - fetchFxSpot(params)
//   - handleGET(url) helper for Next.js Route Handlers
//
// Design notes (fixes vs older versions):
//   * Single, consistent convention: **usdPer[CCY] = USD per 1 unit of CCY**
//     → Cross mid = usdPer[base] / usdPer[quote] (no special cases).
//   * Stable pips & rounding: JPY quotes use 0.01 pip, others 0.0001.
//   * Bid/Ask guaranteed so that (ask - bid) / pip == spreadPips exactly.
//   * Deterministic “as of day” snapshot + gentle AR(1) history.
//   * Accepts pairs like "EURUSD", case/sep-insensitive ("eur/usd" ok).

/* --------------------------------- Types --------------------------------- */

export type FxQuote = {
  pair: string;       // "EURUSD"
  base: string;       // "EUR"
  quote: string;      // "USD"
  spot: number;       // mid (e.g., 1.0832)
  bid: number;
  ask: number;
  spreadPips: number; // integer pips
  ts: string;         // ISO timestamp
};

export type FxHistoryPoint = { t: string; pair: string; mid: number };

export type FxSnapshot = {
  asOf: string;              // ISO
  quotes: FxQuote[];
  history: FxHistoryPoint[]; // trailing daily mids for each requested pair
  meta: { pip: Record<string, number> };
};

export type FxParams = {
  pairs?: string[];       // e.g., ["EURUSD","USDJPY", ...]
  seed?: number;          // deterministic randomness seed
  daysHistory?: number;   // default 120
  overridesUsdPer?: Partial<Record<string, number>>; // override usdPer anchors
};

/* ---------------------------------- API ---------------------------------- */

export async function fetchFxSpot(p?: FxParams): Promise<FxSnapshot> {
  const cfg = withDefaults(p);
  const now = new Date();
  const dayKey = now.toISOString().slice(0, 10); // freeze within the day
  const rngDay = mulberry32(hash(`fx|snap|${cfg.seed}|${dayKey}`));

  // Build usdPer anchors (USD per 1 unit of CCY)
  // Example levels around late-2025 ballpark; slight daily jitter for realism.
  const usdPer = makeUsdPerAnchors(rngDay, cfg.overridesUsdPer);

  // Normalize pairs
  const pairs = dedupe(
    (cfg.pairs.length ? cfg.pairs : defaultPairs()).map(normalizePair).filter(Boolean) as string[]
  );

  // Quotes
  const quotes: FxQuote[] = pairs.map((pair) => quoteForPair(pair, usdPer, rngDay, now));

  // History: ATM mids via gentle AR(1) log-walk around current cross
  const history = makeHistory(pairs, usdPer, cfg.daysHistory, cfg.seed);

  return {
    asOf: now.toISOString(),
    quotes,
    history,
    meta: { pip: Object.fromEntries(pairs.map((pr) => [pr, pipSize(pr)])) },
  };
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
  const daysHistory = num(url.searchParams.get("daysHistory"));
  const seed = num(url.searchParams.get("seed"));

  const snap = await fetchFxSpot({
    pairs: pairs.length ? pairs : undefined,
    daysHistory: daysHistory ?? undefined,
    seed: seed ?? undefined,
  });

  return new Response(JSON.stringify(snap, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

/* --------------------------------- Quotes --------------------------------- */

function quoteForPair(pair: string, usdPer: Record<string, number>, rng: () => number, now: Date): FxQuote {
  const { base, quote } = splitPair(pair);
  const mid = crossMid(usdPer, base, quote, rng);
  const pip = pipSize(pair);
  const spreadPips = pipSpread(pair, rng);   // integer
  const half = (spreadPips * pip) / 2;

  // Round bid/ask so that exact pip math holds even after rounding
  const bid = roundFX(mid - half, pair);
  const ask = roundFX(bid + spreadPips * pip, pair); // enforce exact width
  const spot = roundFX((bid + ask) / 2, pair);

  return {
    pair,
    base,
    quote,
    spot,
    bid,
    ask,
    spreadPips,
    ts: now.toISOString(),
  };
}

function crossMid(usdPer: Record<string, number>, base: string, quote: string, rng: () => number) {
  // USD per base divided by USD per quote = quote units per base
  const b = usdPer[base] ?? 1;
  const q = usdPer[quote] ?? 1;
  const bothMajor = isMajor(base) && isMajor(quote);
  const vol = bothMajor ? 0.0012 : 0.003; // tiny day jitter
  // Multiplicative noise
  const n = (rng() - 0.5) * vol;
  return (b / q) * Math.exp(n);
}

/* --------------------------------- History -------------------------------- */

function makeHistory(pairs: string[], usdPer: Record<string, number>, days: number, seed: number): FxHistoryPoint[] {
  const out: FxHistoryPoint[] = [];
  const today = new Date();

  for (const pairRaw of pairs) {
    const pair = normalizePair(pairRaw);
    const { base, quote } = splitPair(pair);
    const rng = mulberry32(hash(`fxhist|${pair}|${seed}`));
    let lvl = crossMid(usdPer, base, quote, rng); // start near “today”

    // AR(1) + lognormal step
    const bothMajor = isMajor(base) && isMajor(quote);
    const vol = bothMajor ? 0.008 : 0.015; // daily sigma
    let drift = 0;
    for (let d = days - 1; d >= 0; d--) {
      const date = new Date(today.getTime() - d * 86400000);
      drift = 0.93 * drift + (rng() - 0.5) * vol * 0.15;
      const shock = (rng() - 0.5) * vol + drift;
      lvl = Math.max(1e-9, lvl * Math.exp(shock));
      out.push({ t: date.toISOString().slice(0, 10), pair, mid: roundFX(lvl, pair) });
    }
  }
  return out;
}

/* -------------------------------- Anchors --------------------------------- */

function makeUsdPerAnchors(rng: () => number, overrides?: Partial<Record<string, number>>): Record<string, number> {
  // USD per 1 CCY (so EUR≈1.08 means 1 EUR = 1.08 USD; JPY≈0.00645 means 1 JPY = 0.00645 USD)
  const base: Record<string, number> = {
    USD: 1.0,
    EUR: 1.08  + (rng() - 0.5) * 0.010, // ±1c
    GBP: 1.27  + (rng() - 0.5) * 0.012,
    CHF: 1.12  + (rng() - 0.5) * 0.010,
    AUD: 0.66  + (rng() - 0.5) * 0.010,
    NZD: 0.60  + (rng() - 0.5) * 0.010,
    CAD: 0.74  + (rng() - 0.5) * 0.010, // USDCAD ~ 1/0.74 ≈ 1.35
    JPY: 0.00645 + (rng() - 0.5) * 0.00006, // USD/JPY ~ 155
    CNH: 0.137 + (rng() - 0.5) * 0.003,
    INR: 0.0120 + (rng() - 0.5) * 0.0002,
    SEK: 0.0926 + (rng() - 0.5) * 0.002,
    NOK: 0.0952 + (rng() - 0.5) * 0.003,
    MXN: 0.0588 + (rng() - 0.5) * 0.002,
    ZAR: 0.0556 + (rng() - 0.5) * 0.002,
    SGD: 0.74  + (rng() - 0.5) * 0.010,
    HKD: 0.128 + (rng() - 0.5) * 0.002,
  };
  for (const [k, v] of Object.entries(overrides || {})) {
    if (Number.isFinite(v as number)) base[k.toUpperCase()] = v as number;
  }
  return base;
}

/* ------------------------------- Conventions ------------------------------- */

function pipSize(pair: string) {
  const q = pair.slice(3, 6).toUpperCase();
  return q === "JPY" ? 0.01 : 0.0001;
}

function dp(pair: string) {
  // Show one decimal more than pip (like many venues do)
  const q = pair.slice(3, 6).toUpperCase();
  return q === "JPY" ? 3 : 5;
}

function pipSpread(pair: string, rng: () => number) {
  // Integer pips; tighter for majors
  const majors = /^(EURUSD|USDJPY|GBPUSD|USDCHF|AUDUSD|USDCAD|NZDUSD|EURGBP|EURJPY|GBPJPY)$/i;
  const isMaj = majors.test(pair);
  const base = isMaj ? 1 : 6;
  const jitter = isMaj ? Math.floor(rng() * 2) : Math.floor(rng() * 5); // 0–1 or 0–4
  return Math.max(1, base + jitter);
}

function roundFX(x: number, pair: string) {
  const d = dp(pair);
  const p = 10 ** d;
  return Math.round(x * p) / p;
}

/* --------------------------------- Helpers -------------------------------- */

function splitPair(pair: string) {
  const s = normalizePair(pair);
  const base = s.slice(0, 3).toUpperCase();
  const quote = s.slice(3, 6).toUpperCase();
  return { base, quote };
}

function normalizePair(p: string) {
  const s = (p || "").toUpperCase().replace(/[^A-Z]/g, "");
  return s.length === 6 ? s : "";
}

function defaultPairs() {
  return ["EURUSD","USDJPY","GBPUSD","USDCHF","AUDUSD","USDCAD","NZDUSD","EURGBP","EURJPY","GBPJPY","USDCNH","USDINR","USDSEK","USDNOK","USDMXN","USDZAR","USDSGD","USDHKD"];
}

function isMajor(ccy: string) {
  const m = new Set(["USD","EUR","JPY","GBP","CHF","AUD","NZD","CAD","SEK","NOK","SGD","HKD"]);
  return m.has(ccy.toUpperCase());
}

/* ---------------------------------- Utils ---------------------------------- */

function num(s: string | null) { if (s == null) return undefined; const n = Number(s); return Number.isFinite(n) ? n : undefined; }
function dedupe<T>(a: T[]) { return Array.from(new Set(a)); }

/* ----------------------- Deterministic RNG & Hash -------------------------- */

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

/* -------------------------------- Defaults -------------------------------- */

function withDefaults(p?: FxParams) {
  return {
    pairs: (p?.pairs?.length ? p.pairs : []) as string[],
    seed: p?.seed ?? 0,
    daysHistory: p?.daysHistory ?? 120,
    overridesUsdPer: p?.overridesUsdPer ?? {},
  };
}

/* ---------------------------------- Notes ----------------------------------
- All prices are SYNTHETIC for UI/UX/testing. Do not use for trading.
- If you need inverse pairs, provide them directly (e.g., "USDEUR"); this
  module will handle any 6-letter CCY pairs deterministically.
---------------------------------------------------------------------------- */