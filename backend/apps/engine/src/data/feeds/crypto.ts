// data/feeds/crypto.ts
// Crypto market data feed adapter (pure Node, no imports)
//
// Stubbed adapter exposing common endpoints you can wire to a real exchange
// (Binance, Bybit, Deribit, Coinbase, OKX, etc.). Replace the internals with
// HTTPS/fetch calls when you’re ready.
//
// Exposed methods:
// - isConnected()
// - connect()/disconnect()
// - getSpot(tickers?: string[])
// - getFundingRates(symbols?: string[])
// - getOpenInterest(symbols?: string[])
// - getPerpBasis(symbol: string)             // spot vs perp basis snapshot
// - getOrderBook(symbol: string, depth?: n)  // lightweight L2 book
// - getMetadata(symbol: string)

export function CryptoFeed(opts: any = {}) {
  const state = {
    name: "crypto-feed",
    connected: false,
    lastUpdate: 0,
    latencyMs: Number(opts.latencyMs ?? 40),
    base: "USD",
  };

  const universe = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
  ];

  function isConnected() {
    return state.connected;
  }

  async function connect() {
    state.connected = true;
    state.lastUpdate = Date.now();
    return { ok: true, msg: "connected to crypto feed" };
  }

  async function disconnect() {
    state.connected = false;
    return { ok: true, msg: "disconnected" };
  }

  // -------- Spot Prices -------------------------------------------------

  async function getSpot(tickers?: string[]) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const list = (tickers && tickers.length ? tickers : universe).slice(0, 50);
    const prices: Record<string, number> = {};
    for (const t of list) prices[t] = stubSpot(t);

    return {
      ok: true,
      base: state.base,
      ts: iso(),
      prices,
    };
  }

  // -------- Funding Rates (perp) ---------------------------------------

  async function getFundingRates(symbols?: string[]) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const list = (symbols && symbols.length ? symbols : universe).slice(0, 50);
    const rates: Record<string, number> = {};
    for (const s of list) rates[s] = stubFunding(s);

    return {
      ok: true,
      ts: iso(),
      rates,                 // 8h funding rate (fraction, e.g., 0.0001 = 1 bps)
      intervalHours: 8,
    };
  }

  // -------- Open Interest ----------------------------------------------

  async function getOpenInterest(symbols?: string[]) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const list = (symbols && symbols.length ? symbols : universe).slice(0, 50);
    const oi: Record<string, number> = {};
    for (const s of list) oi[s] = stubOpenInterest(s);

    return {
      ok: true,
      ts: iso(),
      oi, // notional in base currency (approx)
      currency: state.base,
    };
  }

  // -------- Perp Basis (spot vs perp) ----------------------------------

  async function getPerpBasis(symbol: string) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const spot = stubSpot(symbol);
    const perp = spot * (1 + stubFunding(symbol) * 20); // fake small skew from funding
    const basisAbs = perp - spot;
    const basisPct = basisAbs / spot;

    return {
      ok: true,
      ts: iso(),
      symbol,
      spot,
      perp,
      basisAbs,
      basisPct,
    };
  }

  // -------- Order Book (L2) --------------------------------------------

  async function getOrderBook(symbol: string, depth: number = 10) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const mid = stubSpot(symbol);
    const tick = mid * 0.0005; // 5 bps tick
    const bids: [number, number][] = [];
    const asks: [number, number][] = [];

    for (let i = 0; i < depth; i++) {
      bids.push([round(mid - tick * (i + 1)), stubSize(symbol)]);
      asks.push([round(mid + tick * (i + 1)), stubSize(symbol)]);
    }

    return {
      ok: true,
      ts: iso(),
      symbol,
      bids, // [price, size]
      asks,
    };
  }

  // -------- Metadata ----------------------------------------------------

  async function getMetadata(symbol: string) {
    const meta: Record<string, any> = {
      BTCUSDT: { base: "BTC", quote: "USDT", lot: 0.0001, minNotional: 5 },
      ETHUSDT: { base: "ETH", quote: "USDT", lot: 0.001, minNotional: 5 },
    };
    return {
      ok: true,
      symbol,
      ...meta[symbol],
    };
  }

  return {
    isConnected,
    connect,
    disconnect,
    getSpot,
    getFundingRates,
    getOpenInterest,
    getPerpBasis,
    getOrderBook,
    getMetadata,
  };
}

/* ---------------------------- Stubs/Helpers ---------------------------- */

function iso() {
  try { return new Date().toISOString(); } catch { return "" + Date.now(); }
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function round(n: number) {
  return Math.round(n * 100) / 100;
}

function stubSpot(sym: string): number {
  // quick and dirty symbol → base price mapping
  const base: Record<string, number> = {
    BTCUSDT: 65000,
    ETHUSDT: 3200,
    SOLUSDT: 180,
    BNBUSDT: 580,
    XRPUSDT: 0.6,
    ADAUSDT: 0.45,
    DOGEUSDT: 0.17,
    AVAXUSDT: 40,
    DOTUSDT: 7.5,
    MATICUSDT: 0.8,
  };
  const mid = base[sym] ?? 100;
  // add tiny random walk so values move a bit
  const jitter = (Math.random() - 0.5) * mid * 0.002; // ±0.2%
  return Math.max(0.0001, mid + jitter);
}

function stubFunding(_sym: string): number {
  // random funding rate around 0, typically within ±5 bps per 8h
  return (Math.random() - 0.5) * 0.001; // ±0.10% (10 bps) wide for variety
}

function stubOpenInterest(sym: string): number {
  // rough notional size by asset
  const scale: Record<string, number> = {
    BTCUSDT: 3_500_000_000,
    ETHUSDT: 1_200_000_000,
    SOLUSDT: 800_000_000,
    BNBUSDT: 600_000_000,
  };
  const base = scale[sym] ?? 250_000_000;
  // jitter ±10%
  return Math.max(0, base * (0.9 + Math.random() * 0.2));
}

function stubSize(symbol: string): number {
    throw new Error("Function not implemented.");
}
