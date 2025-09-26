// runequityscreenserver.tsx
// Zero-import synthetic “equity screener” that filters a mock universe on the server.
// Use this for wiring with ScreenerForm. Replace the mock generator with your DB/feed.
//
// Exports:
//   - types
//   - runEquityScreen(input)
//   - handleGET(url) for Next.js Route Handlers

/* ---------------------------------- Types ---------------------------------- */

export type EquityRow = {
  symbol: string;
  name: string;
  sector: string;
  country: string;
  price: number;
  marketCap: number;
  volume: number;
  pe?: number;
  pb?: number;
  ps?: number;
  yield?: number;
  beta?: number;
  rsi?: number;
  ivPct?: number;
  score?: number;
};

export type EquityScreenInput = {
  symbols?: string[];
  sectors?: string[];
  countries?: string[];
  price?: [number | "", number | ""];
  marketCap?: [number | "", number | ""];
  volume?: [number | "", number | ""];
  pe?: [number | "", number | ""];
  pb?: [number | "", number | ""];
  ps?: [number | "", number | ""];
  yield?: [number | "", number | ""];
  beta?: [number | "", number | ""];
  rsi?: [number | "", number | ""];
  ivPct?: [number | "", number | ""];
  score?: [number | "", number | ""];
  sort?: { key: keyof EquityRow; dir: "asc" | "desc" };
  limit?: number;
  seed?: number;
};

export type EquityScreenResult = {
  asOf: string;
  total: number;
  rows: EquityRow[];
  query: EquityScreenInput;
};

/* ---------------------------------- API ---------------------------------- */

export async function runEquityScreen(input: EquityScreenInput = {}): Promise<EquityScreenResult> {
  const now = new Date().toISOString();
  const rng = mulberry32(input.seed ?? 42);
  const universe = buildMockUniverse(2000, rng);

  const syms = new Set((input.symbols || []).map((s) => s.trim().toUpperCase()).filter(Boolean));
  const secSet = new Set((input.sectors || []).map((s) => s.toLowerCase()));
  const ctySet = new Set((input.countries || []).map((s) => s.toUpperCase()));

  let rows = universe.filter((e) => {
    if (syms.size && !syms.has(e.symbol.toUpperCase())) return false;
    if (secSet.size && !secSet.has(e.sector.toLowerCase())) return false;
    if (ctySet.size && !ctySet.has(e.country.toUpperCase())) return false;
    if (!rangeOk(e.price, input.price)) return false;
    if (!rangeOk(e.marketCap, input.marketCap)) return false;
    if (!rangeOk(e.volume, input.volume)) return false;
    if (!rangeOk(e.pe, input.pe)) return false;
    if (!rangeOk(e.pb, input.pb)) return false;
    if (!rangeOk(e.ps, input.ps)) return false;
    if (!rangeOk(e.yield, input.yield)) return false;
    if (!rangeOk(e.beta, input.beta)) return false;
    if (!rangeOk(e.rsi, input.rsi)) return false;
    if (!rangeOk(e.ivPct, input.ivPct)) return false;
    if (!rangeOk(e.score, input.score)) return false;
    return true;
  });

  const sortKey = input.sort?.key || "marketCap";
  const sortDir = input.sort?.dir || "desc";
  rows.sort((a, b) => cmp((a as any)[sortKey], (b as any)[sortKey], sortDir));

  const limit = Math.max(1, Math.min(input.limit ?? 500, 5000));
  const out = rows.slice(0, limit);

  return { asOf: now, total: rows.length, rows: out, query: input };
}

/**
 * Optional Next.js Route Handler:
 *   export async function GET(req: Request) { return handleGET(req.url); }
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const getArr = (k: string) => (url.searchParams.get(k) || "").split(",").map((s) => s.trim()).filter(Boolean);
  const getPair = (k: string): [number | "", number | ""] | undefined => {
    const v = getArr(k);
    if (!v.length) return undefined;
  };
  const sortRaw = url.searchParams.get("sort") || "";
  const sort = sortRaw
    ? { key: sortRaw.split(":")[0] as keyof EquityRow, dir: (sortRaw.split(":")[1] as "asc" | "desc") || "desc" }
    : undefined;

  const input: EquityScreenInput = {
    symbols: getArr("symbols"),
    sectors: getArr("sectors"),
    countries: getArr("countries"),
    price: getPair("price"),
    marketCap: getPair("marketCap"),
    volume: getPair("volume"),
    pe: getPair("pe"),
    pb: getPair("pb"),
    ps: getPair("ps"),
    yield: getPair("yield"),
    beta: getPair("beta"),
    rsi: getPair("rsi"),
    ivPct: getPair("ivPct"),
    score: getPair("score"),
    sort,
    limit: toNum(url.searchParams.get("limit")),
    seed: toNum(url.searchParams.get("seed")),
  };

  const res = await runEquityScreen(input);
  return new Response(JSON.stringify(res, null, 2), { headers: { "content-type": "application/json; charset=utf-8" } });
}

/* ------------------------------ Mock Universe ------------------------------ */

function buildMockUniverse(n: number, rng: () => number): EquityRow[] {
  const sectors = ["Tech","Financials","Health Care","Energy","Materials","Industrials","Utilities","Consumer","Real Estate"];
  const countries = ["US","GB","DE","FR","JP","CN","IN","BR","CA","AU"];
  const out: EquityRow[] = [];

  for (let i = 0; i < n; i++) {
    const symbol = genSym(i);
    const name = `Company ${i}`;
    const sector = pick(sectors, rng);
    const country = pick(countries, rng);
    const price = round(5 + rng() * 995, 2);
    const marketCap = Math.floor(100e6 + rng() * 900e9);
    const volume = Math.floor(100e3 + rng() * 5e7);
    const pe = round(5 + rng() * 35, 1);
    const pb = round(0.5 + rng() * 8, 1);
    const ps = round(0.5 + rng() * 12, 1);
    const yieldPct = round(rng() * 6, 2);
    const beta = round(0.2 + rng() * 2, 2);
    const rsi = round(20 + rng() * 60, 1);
    const ivPct = round(10 + rng() * 70, 1);
    const score = round(rng() * 100, 1);

    out.push({ symbol, name, sector, country, price, marketCap, volume, pe, pb, ps, yield: yieldPct, beta, rsi, ivPct, score });
  }
  return out;
}

/* --------------------------------- Helpers -------------------------------- */

function rangeOk(v?: number, p?: [number | "", number | ""]) {
  if (v == null) return false;
  if (!p) return true;
  const [lo, hi] = p;
  if (lo !== "" && v < Number(lo)) return false;
  if (hi !== "" && v > Number(hi)) return false;
  return true;
}

function cmp(a: any, b: any, dir: "asc" | "desc") {
  const ax = a ?? -Infinity, bx = b ?? -Infinity;
  if (ax === bx) return 0;
  const r = ax > bx ? 1 : -1;
  return dir === "asc" ? r : -r;
}

function genSym(i: number) {
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  return Array.from({ length: 3 }, (_, k) => letters[(i + k * 7) % letters.length]).join("");
}
function pick<T>(arr: T[], rng: () => number): T { return arr[Math.floor(rng() * arr.length)]; }
function round(x: number, d = 2) { const p = 10**d; return Math.round(x * p)/p; }
function toNum(s: string | null): number | undefined {
  if (!s) return undefined;
  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function() {
    t += 0x6D2B79F5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------------------------- Notes ----------------------------------
- This is a synthetic mock screener for development/demo.
- Replace buildMockUniverse() with a query to your equities DB or data feed.
---------------------------------------------------------------------------- */