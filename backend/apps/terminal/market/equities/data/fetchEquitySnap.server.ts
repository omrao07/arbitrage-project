// fetchequitysnap.server.ts
// Zero-import, framework-agnostic mock “equity snapshot” generator.
// Drop this anywhere (server or edge). Deterministic per (symbol + day).
//
// Exports:
//   - types (EquitySnap, Quote, Bar, Fundamentals, OrderBook)
//   - fetchEquitySnap(params)
//   - optional handleGET(url) helper to use as a Next.js route handler later.

/* ------------------------------- Types -------------------------------- */

export type Bar = {
  t: string;    // ISO timestamp
  o: number;    // open
  h: number;    // high
  l: number;    // low
  c: number;    // close
  v: number;    // volume
};

export type Quote = {
  price: number;
  prevClose: number;
  change: number;      // price - prevClose
  changePct: number;   // change / prevClose
  open: number;
  dayHigh: number;
  dayLow: number;
  bid: number;
  bidSize: number;
  ask: number;
  askSize: number;
  volume: number;
  vwap: number;
};

export type Fundamentals = {
  marketCap: number;   // $
  sharesOut: number;   // #
  pe: number;          // trailing PE
  eps: number;         // $
  beta: number;
  divYield: number;    // decimal (0.015 = 1.5%)
  week52High: number;
  week52Low: number;
  currency: string;    // e.g., "USD"
  exchange: string;    // e.g., "NASDAQ"
  sector: string;
  industry: string;
};

export type OrderBook = {
  bids: Array<{ price: number; size: number }>;
  asks: Array<{ price: number; size: number }>;
  ts: string; // ISO
};

export type EquitySnap = {
  symbol: string;
  quote: Quote;
  fundamentals: Fundamentals;
  intraday: Bar[];     // today’s 1-min bars (or coarser if minutes<)
  lastUpdated: string; // ISO
};

export type SnapParams = {
  symbol?: string;     // default "AAPL"
  seed?: number;       // extra entropy
  minutes?: number;    // bars to generate (default 120)
  tzOffsetMin?: number;// minutes offset for session start (default -240 for ET)
  price?: number;      // override last price
  prevClose?: number;  // override previous close
};

/* ------------------------------- API ---------------------------------- */

export async function fetchEquitySnap(p?: SnapParams): Promise<EquitySnap> {
  const cfg = defaults(p);
  const dayKey = new Date().toISOString().slice(0, 10);
  const rng = mulberry32(hash(`${cfg.symbol}|${dayKey}|${cfg.seed}`));

  // Base levels
  const prevClose = cfg.prevClose ?? round((cfg.price ?? basePrice(cfg.symbol)) * (0.98 + rng() * 0.04), 2);
  const last = cfg.price ?? round(prevClose * (0.985 + rng() * 0.03), 2);
  const open = round(prevClose * (0.99 + rng() * 0.02), 2);

  // Session timeline
  const now = new Date();
  const sessionStart = withTime(now, 9, 30, 0, cfg.tzOffsetMin); // default ~09:30 local TZ
  const bars = makeBars({
    start: sessionStart,
    count: cfg.minutes,
    prevClose,
    last,
    rng,
  });

  // Day range & vwap
  const dayHigh = Math.max(...bars.map((b) => b.h), last, open);
  const dayLow = Math.min(...bars.map((b) => b.l), last, open);
  const volume = Math.max(1, Math.round(bars.reduce((s, b) => s + b.v, 0)));
  const vwap = round(bars.reduce((s, b) => s + b.c * b.v, 0) / Math.max(1, volume), 4);

  // Quote
  const q: Quote = {
    price: last,
    prevClose,
    change: round(last - prevClose, 2),
    changePct: round((last - prevClose) / (prevClose || 1), 4),
    open,
    dayHigh: round(dayHigh, 2),
    dayLow: round(dayLow, 2),
    bid: round(last - 0.02, 2),
    bidSize: 100 + Math.floor(rng() * 900),
    ask: round(last + 0.02, 2),
    askSize: 100 + Math.floor(rng() * 900),
    volume,
    vwap,
  };

  // Fundamentals (mock but coherent)
  const fundamentals = makeFundamentals(cfg.symbol, last, rng);

  return {
    symbol: cfg.symbol,
    quote: q,
    fundamentals,
    intraday: bars,
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * Optional helper to wire as a Next.js Route Handler:
 * export const dynamic = "force-dynamic";
 * export async function GET(req: Request) { return handleGET(req.url); }
 */
export async function handleGET(urlOrReqUrl: string): Promise<Response> {
  const url = new URL(urlOrReqUrl);
  const symbol = url.searchParams.get("symbol") || undefined;
  const minutes = num(url.searchParams.get("minutes"));
  const seed = num(url.searchParams.get("seed"));
  const tz = num(url.searchParams.get("tz"));
  const snap = await fetchEquitySnap({ symbol, minutes: minutes ?? undefined, seed: seed ?? undefined, tzOffsetMin: tz ?? undefined });
  return new Response(JSON.stringify(snap, null, 2), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

/* ----------------------------- Generators ----------------------------- */

function makeBars({
  start,
  count,
  prevClose,
  last,
  rng,
}: {
  start: Date;
  count: number;
  prevClose: number;
  last: number;
  rng: () => number;
}): Bar[] {
  const n = Math.max(10, count);
  const bars: Bar[] = [];
  // Build a smooth random walk from prevClose to last across n bars
  const t0 = new Date(start);
  let px = prevClose;
  let volBase = Math.max(5000, Math.round(30000 * (0.8 + rng() * 0.6)));
  for (let i = 0; i < n; i++) {
    const t = new Date(t0.getTime() + i * 60_000); // 1-minute spacing
    // drift towards target (last) plus noise
    const blend = (i + 1) / n;
    const target = prevClose + (last - prevClose) * blend;
    const noise = (rng() - 0.5) * prevClose * 0.0025; // ~±25 bps noise
    const next = clamp(target + noise, 0.01, 1e9);

    const o = px;
    const c = round(next, 4);
    const h = round(Math.max(o, c) * (1 + (rng() * 0.0015)), 4);
    const l = round(Math.min(o, c) * (1 - (rng() * 0.0015)), 4);
    const v = Math.max(1, Math.round(volBase * (0.6 + 0.8 * rng())));

    bars.push({ t: t.toISOString(), o: round(o, 4), h, l, c, v });
    px = c;
  }
  return bars;
}

function makeFundamentals(symbol: string, last: number, rng: () => number): Fundamentals {
  const sharesOut = 1e9 * (0.5 + rng()); // 0.5–1.5B
  const eps = round((1 + rng() * 9) / 12, 2); // $0.08–$0.83 (quarterly-ish)
  const pe = round(Math.max(5, Math.min(80, (last / Math.max(eps, 0.01)) * (0.6 + rng() * 0.8))), 2);
  const marketCap = round(sharesOut * last, 0);
  const beta = round(0.6 + rng() * 1.2, 2);
  const divYield = round(rng() * 0.03, 4); // 0–3%
  const hi = round(last * (1.1 + rng() * 0.15), 2);
  const lo = round(last * (0.8 - rng() * 0.1), 2);

  return {
    marketCap,
    sharesOut: Math.round(sharesOut),
    pe,
    eps,
    beta,
    divYield,
    week52High: Math.max(hi, lo, last),
    week52Low: Math.min(hi, lo, last),
    currency: "USD",
    exchange: guessExchange(symbol),
    sector: guessSector(symbol),
    industry: guessIndustry(symbol),
  };
}

/* ------------------------------- Utilities ---------------------------- */

function defaults(p?: SnapParams) {
  return {
    symbol: (p?.symbol || "AAPL").toUpperCase(),
    seed: p?.seed ?? 0,
    minutes: p?.minutes ?? 120,
    tzOffsetMin: p?.tzOffsetMin ?? -240, // (approx) US/Eastern
    price: p?.price,
    prevClose: p?.prevClose,
  };
}

function basePrice(symbol: string) {
  // deterministic base price from symbol string
  const h = hash(symbol);
  return 50 + (h % 500); // $50–$550
}

function withTime(now: Date, hh: number, mm: number, ss: number, tzOffsetMin: number) {
  // Create a date that roughly maps to local clock hh:mm:ss but keep it simple
  const d = new Date(now);
  d.setHours(hh, mm, ss, 0);
  // Adjust by offset minutes if you want sessions in a different TZ
  d.setMinutes(d.getMinutes() + tzOffsetMin);
  return d;
}

function num(s: string | null) {
  if (s == null) return undefined;
  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}

function round(n: number, d = 2) {
  const p = 10 ** d;
  return Math.round(n * p) / p;
}
function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}

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
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function guessExchange(symbol: string) {
  if (/^\^?GSPC|^SPY$/.test(symbol)) return "NYSE Arca";
  if (/^AAPL|MSFT|GOOGL|AMZN|META|NVDA|TSLA|NFLX/.test(symbol)) return "NASDAQ";
  return "NYSE";
}
function guessSector(symbol: string) {
  if (/AAPL|MSFT|NVDA/.test(symbol)) return "Information Technology";
  if (/AMZN|WMT|TGT/.test(symbol)) return "Consumer Discretionary";
  if (/XOM|CVX/.test(symbol)) return "Energy";
  if (/JPM|GS|BAC/.test(symbol)) return "Financials";
  return "Industrials";
}
function guessIndustry(symbol: string) {
  if (/AAPL|MSFT/.test(symbol)) return "Consumer Electronics";
  if (/NVDA/.test(symbol)) return "Semiconductors";
  if (/AMZN/.test(symbol)) return "Internet & Direct Marketing";
  return "Diversified";
}