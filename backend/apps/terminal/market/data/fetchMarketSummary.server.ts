// fetch-marketsummary.server.ts
// Zero-import, framework-agnostic server utility that returns a market summary snapshot.
// Deterministic-ish mocks (seed + day) so UI is stable across reloads.
// Replace the mock generators with real data sources when wiring up.

// ----------------------------- Types -----------------------------

export type IndexRow = {
  symbol: string;          // e.g., "NIFTY50", "SPX", "NDX"
  name: string;            // "Nifty 50"
  price: number;           // last
  changePct: number;       // % change today
  high: number;
  low: number;
  prevClose: number;
  volume?: number;         // optional
  currency: string;        // "INR", "USD"
};

export type SectorMove = {
  sector: string;          // "IT", "Energy", "Banks"
  changePct: number;       // % change today
  advancers: number;       // # of rising constituents
  decliners: number;       // # of falling constituents
};

export type FxPair = {
  base: string;            // "USD"
  quote: string;           // "INR"
  rate: number;            // e.g., 83.15
  changePct: number;       // daily %
};

export type CommodityRow = {
  symbol: string;          // "BRENT", "WTI", "XAU", "XAG"
  name: string;            // "Brent Crude"
  price: number;
  changePct: number;
  unit?: string;           // "USD/bbl", "USD/oz"
};

export type CryptoRow = {
  symbol: string;          // "BTC", "ETH"
  price: number;           // in quoteCurrency
  change24hPct: number;
  marketCap?: number;
  volume24h?: number;
};

export type MarketSummary = {
  ts: string;              // ISO timestamp
  region: "IN" | "US" | "EU" | "GLOBAL";
  currency: string;        // reference display currency ("INR" or "USD")
  indices: IndexRow[];
  sectors: SectorMove[];
  fx: FxPair[];
  commodities: CommodityRow[];
  crypto: CryptoRow[];
};

// ----------------------------- Params -----------------------------

export type MarketSummaryParams = {
  region?: "IN" | "US" | "EU" | "GLOBAL";
  currency?: "INR" | "USD";
  seed?: number;                 // tweak mock randomness
  // Optional inclusion filters (leave undefined to include all)
  includeIndices?: boolean;
  includeSectors?: boolean;
  includeFx?: boolean;
  includeCommodities?: boolean;
  includeCrypto?: boolean;
};

// ------------------------ Deterministic RNG ------------------------

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

const round = (n: number, d = 2) => {
  const p = 10 ** d;
  return Math.round(n * p) / p;
};

// --------------------------- Mock Anchors ---------------------------

const INDEX_ANCHORS_USD: Record<string, { price: number; name: string; ccy: string }> = {
  SPX: { price: 5600, name: "S&P 500", ccy: "USD" },
  NDX: { price: 20000, name: "Nasdaq 100", ccy: "USD" },
  DJIA: { price: 42000, name: "Dow Jones", ccy: "USD" },
  NIFTY50: { price: 24500, name: "Nifty 50", ccy: "INR" },
  BANKNIFTY: { price: 52000, name: "Bank Nifty", ccy: "INR" },
  DAX: { price: 19000, name: "DAX 40", ccy: "EUR" },
};

const SECTORS = ["IT", "Banks", "Energy", "Healthcare", "FMCG", "Auto", "Metals", "Realty"];

const FX_PAIRS: Array<[string, string, number]> = [
  ["USD", "INR", 83.2],
  ["EUR", "USD", 1.09],
  ["USD", "JPY", 146],
  ["GBP", "USD", 1.27],
  ["USD", "CNY", 7.2],
];

const COMMODS: Array<[string, string, number, string]> = [
  ["BRENT", "Brent Crude", 86, "USD/bbl"],
  ["WTI", "WTI Crude", 82, "USD/bbl"],
  ["XAU", "Gold", 2400, "USD/oz"],
  ["XAG", "Silver", 28, "USD/oz"],
  ["XCU", "Copper", 4.2, "USD/lb"],
];

const CRYPTOS: Array<[string, number, number]> = [
  ["BTC", 65000, 1.2e12],
  ["ETH", 3400, 4.2e11],
  ["SOL", 160, 7.2e10],
  ["XRP", 0.6, 3.1e10],
];

// very rough FX for mock conversion (USD → target)
function fxUSD(quote: string): number {
  const q = quote.toUpperCase();
  if (q === "USD") return 1;
  if (q === "INR") return 83;
  if (q === "EUR") return 0.92;
  if (q === "JPY") return 146;
  if (q === "GBP") return 0.79;
  return 1;
}

// --------------------------- Generators ----------------------------

function genIndex(symbol: string, base: number, name: string, ccy: string, rng: () => number, displayCcy: string): IndexRow {
  // daily % move in ±2.2% with small bias to 0
  const mag = Math.pow(rng(), 1.6) * 2.2;
  const sign = rng() < 0.5 ? -1 : 1;
  const pct = round(sign * mag, 2);
  const prev = base / (1 + pct / 100);
  const high = Math.max(base, prev) * (1 + rng() * 0.004);
  const low = Math.min(base, prev) * (1 - rng() * 0.004);

  // convert to display currency if needed (only for presentation; not precise)
  const fx = ccy === displayCcy ? 1 : (ccy === "USD" ? fxUSD(displayCcy) : 1);
  return {
    symbol,
    name,
    price: round(base * fx, 2),
    changePct: pct,
    high: round(high * fx, 2),
    low: round(low * fx, 2),
    prevClose: round(prev * fx, 2),
    volume: Math.floor(50_000_000 + rng() * 300_000_000),
    currency: displayCcy,
  };
}

function genSector(name: string, rng: () => number): SectorMove {
  const mag = Math.pow(rng(), 1.6) * 3.0;
  const pct = (rng() < 0.5 ? -1 : 1) * mag;
  const total = 50 + Math.floor(rng() * 100);
  const advRatio = (pct >= 0 ? 0.55 : 0.45) + (rng() - 0.5) * 0.1;
  const adv = Math.max(0, Math.min(total, Math.round(total * advRatio)));
  return {
    sector: name,
    changePct: round(pct, 2),
    advancers: adv,
    decliners: total - adv,
  };
}

function genFx([b, q, base]: [string, string, number], rng: () => number): FxPair {
  const pct = (rng() - 0.5) * 0.8; // ±0.4%
  const rate = base * (1 + pct / 100);
  return {
    base: b,
    quote: q,
    rate: round(rate, 4),
    changePct: round(pct, 3),
  };
}

function genCommodity([sym, name, base, unit]: [string, string, number, string], rng: () => number): CommodityRow {
  const pct = (rng() - 0.5) * 2.5; // ±1.25%
  const price = base * (1 + pct / 100);
  return {
    symbol: sym,
    name,
    price: round(price, 2),
    changePct: round(pct, 2),
    unit,
  };
}

function genCrypto([sym, base, mcap]: [string, number, number], rng: () => number, quote = "USD"): CryptoRow {
  const fx = fxUSD(quote);
  const pct = (rng() - 0.5) * 8; // ±4%
  const price = base * (1 + pct / 100);
  const vol = (price * 10_000 + rng() * 50_000_000) * (quote === "USD" ? 1 : fx);
  return {
    symbol: sym,
    price: round(price * fx, 2),
    change24hPct: round(pct, 2),
    marketCap: Math.round(mcap * fx),
    volume24h: Math.round(vol),
  };
}

// ------------------------------ API -------------------------------

export async function fetchMarketSummary(params?: MarketSummaryParams): Promise<MarketSummary> {
  const region = params?.region ?? "GLOBAL";
  const currency = params?.currency ?? (region === "IN" ? "INR" : "USD");
  const seedBase = (params?.seed ?? 0) ^ strSeed(region + "|" + currency);
  const dayKey = new Date().toISOString().slice(0, 10);
  const rng = seededRandom(seedBase ^ strSeed(dayKey));

  const includeIndices = params?.includeIndices ?? true;
  const includeSectors = params?.includeSectors ?? true;
  const includeFx = params?.includeFx ?? true;
  const includeCommodities = params?.includeCommodities ?? true;
  const includeCrypto = params?.includeCrypto ?? true;

  // Indices selection by region
  const idxSymbols =
    region === "IN"
      ? ["NIFTY50", "BANKNIFTY"]
      : region === "US"
      ? ["SPX", "NDX", "DJIA"]
      : region === "EU"
      ? ["DAX", "SPX"]
      : ["SPX", "NDX", "NIFTY50", "DAX"];

  const indices: IndexRow[] = includeIndices
    ? idxSymbols.map((s) => {
        const meta = INDEX_ANCHORS_USD[s];
        const base = meta.price;
        return genIndex(s, base, meta.name, meta.ccy, rng, currency);
      })
    : [];

  const sectors: SectorMove[] = includeSectors ? SECTORS.map((s) => genSector(s, rng)) : [];

  const fx: FxPair[] = includeFx ? FX_PAIRS.map((p) => genFx(p, rng)) : [];

  const commodities: CommodityRow[] = includeCommodities ? COMMODS.map((c) => genCommodity(c, rng)) : [];

  const crypto: CryptoRow[] = includeCrypto ? CRYPTOS.map((r) => genCrypto(r, rng, currency)) : [];

  return {
    ts: new Date().toISOString(),
    region,
    currency,
    indices,
    sectors,
    fx,
    commodities,
    crypto,
  };
}

// --------------------------- Example usage ---------------------------
//
// const snap = await fetchMarketSummary({ region: "IN", currency: "INR" });
// // snap.indices -> array of IndexRow in INR
// // snap.sectors -> sector breadth
// // snap.fx      -> major FX pairs
// // snap.crypto  -> BTC/ETH… in INR (mocked)
//
// --------------------------------------------------------------------