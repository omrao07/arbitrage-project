// apps/terminal/app/(workspaces)/market/_data/fetchTopMovers.server.tsx
// Server-only helper to fetch top gainers/losers/actives across universes.
//
// Usage (server comp/action):
//   const movers = await fetchTopMovers({ universe: "NIFTY50", direction: "gainers", limit: 25 });

import "server-only";

/* -------------------- types -------------------- */
export type TopMover = {
  symbol: string;          // e.g., "AAPL"
  name?: string;           // "Apple Inc."
  exchange?: string;       // "NASDAQ"
  assetClass?: string;     // "Equity" | "ETF" | "FX" | "Crypto" | "Futures" | ...
  last?: number;           // last price
  change?: number;         // absolute change
  pctChange?: number;      // % change (e.g., 2.31 for +2.31%)
  open?: number;
  high?: number;
  low?: number;
  prevClose?: number;
  volume?: number;         // shares/contracts traded
  turnover?: number;       // notional traded (price * volume)
  marketCap?: number;
  currency?: string;       // "USD", "INR", ...
  sector?: string;
  industry?: string;
  time?: string;           // ISO timestamp of quote
};

export type FetchTopMoversReq = {
  universe?: string;       // e.g., "NIFTY50", "SPX", "NASDAQ100", "FTSE100"
  watchlistId?: string;    // custom watchlist identifier
  assetClass?: string;     // filter (Equity/ETF/FX/Crypto/Futures/Options)
  region?: string;         // "US" | "IN" | "EU" | "JP" | ...
  session?: "regular" | "pre" | "post"; // trading session
  direction?: "gainers" | "losers" | "actives"; // top list flavor
  sortBy?: "pctChange" | "change" | "volume" | "turnover"; // default pctChange
  limit?: number;          // default 50
  offset?: number;         // pagination
  since?: string;          // ISO date/time window start (optional)
  until?: string;          // ISO date/time window end (optional)
};

/* -------------------- endpoint resolution -------------------- */
const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.API_BASE_URL ||
  "";

/** Build endpoint: prefer external backend, fallback to local Next route */
function endpoint(): string {
  if (BACKEND_URL) {
    return `${BACKEND_URL.replace(/\/+$/, "")}/market/top-movers`;
  }
  return "/api/market/top-movers";
}

/* -------------------- main fetcher -------------------- */
export async function fetchTopMovers(req: FetchTopMoversReq = {}): Promise<TopMover[]> {
  const payload: FetchTopMoversReq = {
    direction: "gainers",
    sortBy: "pctChange",
    limit: 50,
    offset: 0,
    session: "regular",
    ...req,
  };

  const res = await fetch(endpoint(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`fetchTopMovers: ${res.status} ${res.statusText} ${text}`);
  }

  const raw = await res.json();
  // Supported backend shapes:
  // 1) { items: [...] } where items are TopMover-like
  // 2) [ ... ] array of TopMover-like
  // 3) { symbols:[...], fields:[...], data:[[...], ...] } columnar
  if (Array.isArray(raw)) return raw.map(normalizeMover);
  if (Array.isArray(raw?.items)) return raw.items.map(normalizeMover);
  if (raw?.symbols && raw?.fields && raw?.data) {
    return normalizeColumnar(raw.symbols, raw.fields, raw.data);
  }
  // Fallback empty
  return [];
}

/* -------------------- normalization -------------------- */
function normalizeMover(x: any): TopMover {
  const last = num(x.last ?? x.price ?? x.ltp);
  const prevClose = num(x.prevClose ?? x.pc);
  const open = num(x.open);
  const change = num(
    x.change ??
      (last != null && prevClose != null ? last - prevClose : undefined)
  );
  const pctChange = num(
    x.pctChange ??
      x.percentChange ??
      (last != null && prevClose != null && prevClose !== 0
        ? ((last - prevClose) / prevClose) * 100
        : undefined)
  );

  return {
    name: str(x.name),
    exchange: str(x.exchange ?? x.venue),
    assetClass: str(x.assetClass ?? x.class),
    last,
    change,
    pctChange,
    open,
    high: num(x.high),
    low: num(x.low),
    prevClose,
    volume: num(x.volume ?? x.vol),
    marketCap: num(x.marketCap ?? x.mcap),
    currency: str(x.currency ?? x.ccy),
    sector: str(x.sector),
    industry: str(x.industry),
    time: str(x.time ?? x.timestamp),
    symbol: ""
};
}

function normalizeColumnar(symbols: any[], fields: any[], data: any[][]): TopMover[] {
  const F = fields.map(String);
  const idx = (k: string) => F.indexOf(k);
  const ix = {
    name: pickIndex(F, ["name", "Name"]),
    exchange: pickIndex(F, ["exchange", "venue"]),
    assetClass: pickIndex(F, ["assetClass", "class"]),
    last: pickIndex(F, ["last", "price", "ltp"]),
    change: pickIndex(F, ["change"]),
    pctChange: pickIndex(F, ["pctChange", "percentChange"]),
    open: pickIndex(F, ["open"]),
    high: pickIndex(F, ["high"]),
    low: pickIndex(F, ["low"]),
    prevClose: pickIndex(F, ["prevClose", "pc"]),
    volume: pickIndex(F, ["volume", "vol"]),
    turnover: pickIndex(F, ["turnover"]),
    marketCap: pickIndex(F, ["marketCap", "mcap"]),
    currency: pickIndex(F, ["currency", "ccy"]),
    sector: pickIndex(F, ["sector"]),
    industry: pickIndex(F, ["industry"]),
    time: pickIndex(F, ["time", "timestamp"]),
  };

  return symbols.map((sym: any, r: number) => {
    const row = data[r] || [];
    const g = (i: number | null) => (i == null || i < 0 ? undefined : row[i]);
    return {
      symbol: String(sym),
      name: str(g(ix.name)),
      exchange: str(g(ix.exchange)),
      assetClass: str(g(ix.assetClass)),
      last: num(g(ix.last)),
      change: num(g(ix.change)),
      pctChange: num(g(ix.pctChange)),
      open: num(g(ix.open)),
      high: num(g(ix.high)),
      low: num(g(ix.low)),
      prevClose: num(g(ix.prevClose)),
      volume: num(g(ix.volume)),
      turnover: num(g(ix.turnover)),
      marketCap: num(g(ix.marketCap)),
      currency: str(g(ix.currency)),
      sector: str(g(ix.sector)),
      industry: str(g(ix.industry)),
      time: str(g(ix.time)),
    };
  });
}

/* -------------------- tiny utils -------------------- */
function pickIndex(fields: string[], candidates: string[]): number | null {
  for (const name of candidates) {
    const i = fields.findIndex((f) => f.toLowerCase() === name.toLowerCase());
    if (i >= 0) return i;
  }
  return null;
}
function num(v: any): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
function str(v: any): string | undefined {
  return v == null ? undefined : String(v);
}