// feftchcryptosnap.server.ts
// Pure TypeScript utility — no imports, no Next.js types required.

export type CryptoSnap = {
  symbol: string;          // e.g., "BTC", "ETH"
  base: string;            // same as symbol (for clarity)
  quote: string;           // e.g., "USD", "INR"
  price: number;           // last price in quote currency
  change24hPct: number;    // % change over 24h
  high24h: number;         // high in last 24h
  low24h: number;          // low in last 24h
  open24h: number;         // open price 24h ago
  volume24h: number;       // notional or base * price (mocked)
  marketCap: number;       // mocked market cap
  ts: string;              // ISO timestamp
  source: "mock";          // set your real source later
};

type Params = {
  symbols?: string[];     // e.g., ["BTC","ETH","SOL"]
  quote?: string;         // "USD" | "INR" etc.
  seed?: number;          // optional to tweak mock determinism
};

/* --------------------------- tiny deterministic rng --------------------------- */
function seededRandom(seed: number) {
  // Mulberry32
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function strSeed(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/* ------------------------------ base price map ------------------------------- */
// Rough anchor prices to shape the mocks
const BASE_USD: Record<string, number> = {
  BTC: 65000,
  ETH: 3400,
  SOL: 160,
  XRP: 0.6,
  ADA: 0.45,
  DOGE: 0.12,
  MATIC: 0.8,
  BNB: 520,
  AVAX: 30,
  DOT: 6.2,
};

function fxUSD(quote: string): number {
  // very rough FX for mock (USD→quote). Tweak or wire to FX later.
  const q = quote.toUpperCase();
  if (q === "USD") return 1;
  if (q === "INR") return 83;
  if (q === "EUR") return 0.92;
  if (q === "JPY") return 146;
  return 1; // default
}

/* ------------------------------ mock generator ------------------------------- */
function generateSnap(sym: string, quote = "USD", seed = 0): CryptoSnap {
  const S = sym.toUpperCase();
  const baseUSD = BASE_USD[S] ?? 10 + (strSeed(S) % 400); // unknowns: 10..410 USD
  const fx = fxUSD(quote);
  // rng seeded by symbol + optional external seed + current date (to evolve daily)
  const dayKey = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const rng = seededRandom(strSeed(S + "|" + dayKey) ^ (seed >>> 0));

  // 24h change between -8%..+8% biased to center
  const changeSign = rng() < 0.5 ? -1 : 1;
  const mag = Math.pow(rng(), 1.5) * 0.08; // skew small
  const changePct = changeSign * mag; // -0.08..+0.08

  const openUsd = baseUSD * (1 - changePct);
  const lastUsd = baseUSD;
  const highUsd = Math.max(lastUsd, openUsd) * (1 + rng() * 0.02);
  const lowUsd = Math.min(lastUsd, openUsd) * (1 - rng() * 0.02);

  const mcapUsd = baseUSD * (5_000_000 + Math.floor(rng() * 50_000_000)); // arbitrary
  const volUsd = (lastUsd * (50_000 + Math.floor(rng() * 1_500_000)));   // arbitrary

  return {
    symbol: S,
    base: S,
    quote: quote.toUpperCase(),
    price: round(lastUsd * fx, 2),
    change24hPct: round(changePct * 100, 2),
    high24h: round(highUsd * fx, 2),
    low24h: round(lowUsd * fx, 2),
    open24h: round(openUsd * fx, 2),
    volume24h: round(volUsd * fx, 0),
    marketCap: round(mcapUsd * fx, 0),
    ts: new Date().toISOString(),
    source: "mock",
  };
}

const round = (n: number, d = 2) => {
  const p = 10 ** d;
  return Math.round(n * p) / p;
};

/* ---------------------------------- API ---------------------------------- */
/**
 * Fetch a batch of crypto snapshots.
 * Replace generateSnap(...) with real exchange/aggregator calls later.
 */
export async function fetchCryptoSnap(params?: Params): Promise<CryptoSnap[]> {
  const { symbols = ["BTC", "ETH", "SOL"], quote = "USD", seed = 0 } = params ?? {};
  return symbols.map((s) => generateSnap(s, quote, seed));
}

/* ---------------------------- Example (server) ----------------------------
const snaps = await fetchCryptoSnap({ symbols: ["BTC","ETH"], quote: "INR" });
// -> [{ symbol:"BTC", quote:"INR", price: …, change24hPct: …, … }, …]
--------------------------------------------------------------------------- */

/* ----------------------- Optional: single-symbol helper ----------------------- */
export async function fetchOneCryptoSnap(symbol: string, quote = "USD", seed = 0): Promise<CryptoSnap> {
  return generateSnap(symbol, quote, seed);
}