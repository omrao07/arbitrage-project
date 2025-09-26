// data/feeds/equities.ts
// Equities market data feed adapter (pure Node, no imports)
//
// Stubbed adapter exposing common endpoints you can later wire to a real
// provider (Polygon, Alpha Vantage, Tiingo, IEX, NSE/BSE, Yahoo, etc.).
//
// Exposed methods:
// - isConnected()
// - connect()/disconnect()
// - getQuote(ticker: string)
// - getQuotes(tickers?: string[])
// - getOrderBook(ticker: string, depth?: number)
// - getPeers(ticker: string)
// - getNews(ticker: string, limit?: number)
// - getSnapshot(ticker: string)   // OHLC, volume, day change, market cap
// - search(query: string)         // simple symbol search (stub)

export function EquitiesFeed(opts: any = {}) {
  const state = {
    name: "equities-feed",
    connected: false,
    lastUpdate: 0,
    latencyMs: Number(opts.latencyMs ?? 35),
    currency: "USD",
  };

  // Small demo universe (add your own)
  const universe = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "TSLA", "NVDA", "NFLX", "AMD", "INTC",
    "TCS.NS", "INFY.NS", "RELIANCE.NS"
  ];

  function isConnected() {
    return state.connected;
  }

  async function connect() {
    state.connected = true;
    state.lastUpdate = Date.now();
    return { ok: true, msg: "connected to equities feed" };
  }

  async function disconnect() {
    state.connected = false;
    return { ok: true, msg: "disconnected" };
  }

  /* ------------------------------ Quotes ------------------------------ */

  async function getQuote(ticker: string) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const base = basePrice(ticker);
    const px = jiggle(base, 0.004); // ±0.4%
    return {
      ok: true,
      ts: iso(),
      ticker,
      price: round(px),
      currency: cc(ticker),
      bid: round(px - tickSize(px)),
      ask: round(px + tickSize(px)),
      bidSize: lot(px),
      askSize: lot(px),
      dayChangePct: round(((px - base) / base) * 100),
    };
  }

  async function getQuotes(tickers?: string[]) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const list = (tickers && tickers.length ? tickers : universe).slice(0, 100);
    const quotes: Record<string, any> = {};
    for (const t of list) {
      const q = await getQuote(t);
      quotes[t] = q.ok ? {
        price: (q as any).price,
        bid: (q as any).bid,
        ask: (q as any).ask,
        bidSize: (q as any).bidSize,
        askSize: (q as any).askSize,
        currency: (q as any).currency,
        ts: (q as any).ts,
      } : { error: "na" };
    }
    return { ok: true, quotes };
  }

  /* ---------------------------- Order Book ---------------------------- */

  async function getOrderBook(ticker: string, depth: number = 10) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const mid = basePrice(ticker);
    const px = jiggle(mid, 0.002);
    const step = tickSize(px);
    const bids: [number, number][] = [];
    const asks: [number, number][] = [];

    for (let i = 0; i < depth; i++) {
      bids.push([round(px - step * (i + 1)), lot(px)]);
      asks.push([round(px + step * (i + 1)), lot(px)]);
    }

    return {
      ok: true,
      ts: iso(),
      ticker,
      bids, // [price, size]
      asks,
      mid: round(px),
    };
  }

  /* ------------------------------ Peers ------------------------------- */

  async function getPeers(ticker: string) {
    await delay(state.latencyMs);
    const map: Record<string, string[]> = {
      AAPL: ["MSFT", "GOOGL", "NVDA", "AMD"],
      MSFT: ["AAPL", "GOOGL", "AMZN", "META"],
      AMZN: ["WMT", "COST", "GOOGL", "META"],
      GOOGL: ["META", "MSFT", "AAPL", "AMZN"],
      META: ["GOOGL", "SNAP", "PINS", "MSFT"],
      TSLA: ["GM", "F", "NIO", "RIVN"],
      NVDA: ["AMD", "INTC", "AVGO", "QCOM"],
      TCS_NS: ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
      "TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
      "INFY.NS": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
      "RELIANCE.NS": ["ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS"],
    };
    return { ok: true, ticker, peers: map[ticker] || samplePeers(ticker) };
  }

  /* ------------------------------- News -------------------------------- */

  async function getNews(ticker: string, limit: number = 8) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const items = [];
    for (let i = 0; i < Math.min(limit, 20); i++) {
      items.push({
        id: uid(),
        ticker,
        headline: `(${ticker}) Update ${i + 1}: ${newsHeadline(ticker)}`,
        source: pick(["Reuters", "Bloomberg", "WSJ", "CNBC", "ET Markets"]),
        ts: iso(),
        url: `https://news.example/${ticker}/${uid()}`, // stub
      });
    }
    return { ok: true, items };
  }

  /* ------------------------------ Snapshot ----------------------------- */

  async function getSnapshot(ticker: string) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const ohlc = stubOHLC(basePrice(ticker));
    const cap = marketCapGuess(ticker, ohlc.close);
    return {
      ok: true,
      ts: iso(),
      ticker,
      currency: cc(ticker),
      ohlc,
      volume: volGuess(ticker),
      dayChangePct: round(((ohlc.close - ohlc.open) / ohlc.open) * 100),
      marketCap: cap,
      floatShares: floatGuess(ticker),
    };
  }

  /* ------------------------------- Search ------------------------------ */

  async function search(query: string) {
    await delay(20);
    const q = (query || "").toUpperCase();
    const matches = universe.filter(t => t.includes(q)).slice(0, 12).map(t => ({
      ticker: t,
      name: companyName(t),
      currency: cc(t),
      exchange: exg(t),
    }));
    return { ok: true, results: matches };
  }

  /* ------------------------------- API --------------------------------- */

  return {
    isConnected,
    connect,
    disconnect,
    getQuote,
    getQuotes,
    getOrderBook,
    getPeers,
    getNews,
    getSnapshot,
    search,
  };
}

/* --------------------------- Stubs / Helpers --------------------------- */

function iso() {
  try { return new Date().toISOString(); } catch { return "" + Date.now(); }
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function round(n: number) {
  return Math.round(n * 100) / 100;
}

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function jiggle(mid: number, pct: number) {
  const delta = mid * pct * (Math.random() - 0.5) * 2;
  return Math.max(0.01, mid + delta);
}

function tickSize(px: number) {
  if (px < 1) return 0.001;
  if (px < 10) return 0.01;
  if (px < 100) return 0.05;
  return 0.1;
}

function lot(px: number) {
  if (px < 5) return Math.floor(500 + Math.random() * 500);
  if (px < 50) return Math.floor(200 + Math.random() * 300);
  return Math.floor(50 + Math.random() * 150);
}

function basePrice(ticker: string): number {
  const map: Record<string, number> = {
    AAPL: 190, MSFT: 430, AMZN: 185, GOOGL: 165, META: 500,
    TSLA: 260, NVDA: 1100, NFLX: 640, AMD: 160, INTC: 42,
    "TCS.NS": 3800, "INFY.NS": 1700, "RELIANCE.NS": 2900
  };
  return map[ticker] ?? 100;
}

function cc(ticker: string) {
  return ticker.endsWith(".NS") ? "INR" : "USD";
}

function exg(ticker: string) {
  return ticker.endsWith(".NS") ? "NSE" : "NASDAQ";
}

function companyName(ticker: string) {
  const map: Record<string, string> = {
    AAPL: "Apple Inc.",
    MSFT: "Microsoft Corporation",
    AMZN: "Amazon.com, Inc.",
    GOOGL: "Alphabet Inc.",
    META: "Meta Platforms, Inc.",
    TSLA: "Tesla, Inc.",
    NVDA: "NVIDIA Corporation",
    NFLX: "Netflix, Inc.",
    AMD: "Advanced Micro Devices, Inc.",
    INTC: "Intel Corporation",
    "TCS.NS": "Tata Consultancy Services Ltd",
    "INFY.NS": "Infosys Ltd",
    "RELIANCE.NS": "Reliance Industries Ltd",
  };
  return map[ticker] ?? `${ticker} Corp.`;
}

function marketCapGuess(ticker: string, price: number) {
  // ballpark float shares * price
  return Math.round(floatGuess(ticker) * price);
}

function floatGuess(ticker: string) {
  const map: Record<string, number> = {
    AAPL: 15_500_000_000,
    MSFT: 7_450_000_000,
    AMZN: 10_300_000_000,
    GOOGL: 5_900_000_000,
    META: 2_600_000_000,
    TSLA: 3_200_000_000,
    NVDA: 2_500_000_000,
    NFLX: 420_000_000,
    AMD: 1_600_000_000,
    INTC: 4_200_000_000,
    "TCS.NS": 3_650_000_000,
    "INFY.NS": 4_100_000_000,
    "RELIANCE.NS": 6_800_000_000,
  };
  // default small-cap-ish
  return map[ticker] ?? 150_000_000;
}

function volGuess(ticker: string) {
  const base = floatGuess(ticker);
  // 0.1% – 1% of float traded per day (rough)
  const pct = 0.001 + Math.random() * 0.009;
  return Math.round(base * pct);
}

function stubOHLC(mid: number) {
  const open = jiggle(mid, 0.004);
  const high = open * (1 + Math.random() * 0.01);
  const low  = open * (1 - Math.random() * 0.01);
  const close = jiggle((high + low) / 2, 0.004);
  return {
    open: round(open),
    high: round(Math.max(high, close)),
    low: round(Math.min(low, close)),
    close: round(close),
  };
}

function newsHeadline(ticker: string) {
  const verbs = ["surges", "slips", "holds steady", "beats", "misses", "guides up", "guides down", "announces buyback", "faces probe", "launches product"];
  const ctx = ["after earnings", "on analyst call", "amid macro jitters", "as sector rallies", "ahead of FOMC", "after rating upgrade", "post M&A chatter"];
  return `${ticker} ${pick(verbs)} ${pick(ctx)}`;
}

function samplePeers(ticker: string) {
  // crude peers via same starting letter
  const letter = (ticker.replace(".NS","")[0] || "A").toUpperCase();
  const pool = ["AAPL","AMZN","AMD","ADBE","ABNB","MSFT","META","MA","NVDA","NFLX","NKE","NEE","GOOGL","GM","GE","TSLA","TM","TCS.NS","INFY.NS","RELIANCE.NS"];
  const out: string[] = [];
  for (const t of pool) if (t[0] === letter && t !== ticker) out.push(t);
  while (out.length < 4) out.push(pick(pool));
  return Array.from(new Set(out)).slice(0, 4);
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}