// data/feeds/fx.ts
// FX market data feed adapter (pure Node, no imports)
//
// Stubbed adapter exposing common endpoints you can later wire to real sources
// (Refinitiv, Bloomberg, Oanda, Polygon, TwelveData, etc.).
//
// Exposed methods:
// - isConnected()
// - connect()/disconnect()
// - getSpot(pairs?: string[])                 // e.g., ["EURUSD","USDJPY"]
// - getForwards(pair: string, tenors?: string[]) // FX forward points & outright
// - getCarry(pair: string)                    // simple interest-rate carry proxy
// - getVolSurface(pair: string)               // stub IV surface (tenor x delta)
// - getMetadata(pair: string)

export function FXFeed(opts: any = {}) {
  const state = {
    name: "fx-feed",
    connected: false,
    lastUpdate: 0,
    latencyMs: Number(opts.latencyMs ?? 30),
    baseCcy: "USD",
  };

  // Small demo universe
  const universe = ["EURUSD", "GBPUSD", "USDJPY", "USDINR", "AUDUSD", "USDCAD"];

  function isConnected() {
    return state.connected;
  }

  async function connect() {
    state.connected = true;
    state.lastUpdate = Date.now();
    return { ok: true, msg: "connected to fx feed" };
  }

  async function disconnect() {
    state.connected = false;
    return { ok: true, msg: "disconnected" };
  }

  /* ------------------------------- Spot -------------------------------- */

  async function getSpot(pairs?: string[]) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const list = (pairs && pairs.length ? pairs : universe).slice(0, 50);
    const quotes: Record<string, { bid: number; ask: number; mid: number; ts: string }> = {};

    for (const p of list) {
      const mid = stubMid(p);
      const spr = spread(p, mid);
      const bid = round(mid - spr / 2);
      const ask = round(mid + spr / 2);
      quotes[p] = { bid, ask, mid: round(mid), ts: iso() };
    }

    return { ok: true, quotes };
  }

  /* ------------------------------ Forwards ------------------------------ */

  async function getForwards(pair: string, tenors?: string[]) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const mids = await getSpot([pair]);
    if (!(mids as any).ok) return { ok: false, error: "spot unavailable" };
    const spot = (mids as any).quotes[pair]?.mid ?? stubMid(pair);

    const T = (tenors && tenors.length ? tenors : ["1W", "1M", "3M", "6M", "1Y"]).slice(0, 10);
    const pts: Record<string, { points: number; outright: number }> = {};

    // very rough CIP-ish forward points using stub short rates
    const { rBase, rQuote } = stubRates(pair);
    for (const t of T) {
      const y = tenorToYears(t);
      const forward = spot * (1 + (rBase - rQuote) * y);
      const points = forward - spot;
      pts[t] = { points: round(points), outright: round(forward) };
    }
    return { ok: true, pair, spot: round(spot), forwards: pts, ts: iso() };
  }

  /* ------------------------------- Carry -------------------------------- */

  async function getCarry(pair: string) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const { rBase, rQuote } = stubRates(pair);
    // carry ≈ (rBase - rQuote)
    const carry = rBase - rQuote;
    return {
      ok: true,
      pair,
      annualized: roundPct(carry),
      rBase: roundPct(rBase),
      rQuote: roundPct(rQuote),
      comment: carry >= 0 ? "positive carry (long base)" : "negative carry (long base)",
      ts: iso(),
    };
  }

  /* ---------------------------- Vol Surface ----------------------------- */

  async function getVolSurface(pair: string) {
    if (!state.connected) return { ok: false, error: "not connected" };
    await delay(state.latencyMs);

    const tenors = ["1W", "1M", "3M", "6M", "1Y"];
    const deltas = ["10P", "25P", "ATM", "25C", "10C"]; // P=put wing, C=call wing
    const baseVol = baseIV(pair);

    const surface: Record<string, Record<string, number>> = {};
    for (const t of tenors) {
      surface[t] = {};
      for (const d of deltas) {
        const smileAdj =
          d === "ATM" ? 0 :
          d.endsWith("P") ? 0.02 : 0.015; // puts a tad higher skew than calls
        const termAdj = termBump(t);
        // jitter a bit
        const v = Math.max(0.03, baseVol + termAdj + (d === "ATM" ? 0 : smileAdj) + (Math.random() - 0.5) * 0.004);
        surface[t][d] = roundPct(v);
      }
    }

    return { ok: true, pair, surface, ts: iso() };
  }

  /* ------------------------------ Metadata ------------------------------ */

  async function getMetadata(pair: string) {
    await delay(5);
    const meta: Record<string, any> = {
      EURUSD: { base: "EUR", quote: "USD", pip: 0.0001, session: "24x5" },
      GBPUSD: { base: "GBP", quote: "USD", pip: 0.0001, session: "24x5" },
      USDJPY: { base: "USD", quote: "JPY", pip: 0.01, session: "24x5" },
      USDINR: { base: "USD", quote: "INR", pip: 0.0025, session: "IST 9:00–17:00 + offshore" },
      AUDUSD: { base: "AUD", quote: "USD", pip: 0.0001, session: "24x5" },
      USDCAD: { base: "USD", quote: "CAD", pip: 0.0001, session: "24x5" },
    };
    return { ok: true, pair, ...(meta[pair] || guessMeta(pair)) };
  }

  /* -------------------------------- API -------------------------------- */

  return {
    isConnected,
    connect,
    disconnect,
    getSpot,
    getForwards,
    getCarry,
    getVolSurface,
    getMetadata,
  };
}

/* --------------------------- Stubs / Helpers --------------------------- */

function iso() { try { return new Date().toISOString(); } catch { return "" + Date.now(); } }
function delay(ms: number) { return new Promise((r) => setTimeout(r, ms)); }
function round(n: number) { return Math.round(n * 1e6) / 1e6; }
function roundPct(n: number) { return Math.round(n * 10000) / 10000; } // 4 dp (e.g., 0.0523)

function tenorToYears(t: string): number {
  const m = String(t).toUpperCase().match(/^(\d+)([DWMY])$/);
  if (!m) return 0.25;
  const n = Number(m[1]); const u = m[2];
  switch (u) {
    case "D": return n / 365;
    case "W": return (n * 7) / 365;
    case "M": return n / 12;
    case "Y": return n;
    default: return 0.25;
  }
}

function stubMid(pair: string): number {
  const base: Record<string, number> = {
    EURUSD: 1.085,
    GBPUSD: 1.27,
    USDJPY: 150.2,
    USDINR: 83.2,
    AUDUSD: 0.66,
    USDCAD: 1.35,
  };
  const mid = base[pair] ?? 1.0;
  // small random walk ±0.1%
  const jitter = mid * 0.001 * (Math.random() - 0.5);
  return Math.max(0.0001, mid + jitter);
}

function spread(pair: string, mid: number): number {
  // crude spread model (tighter for majors)
  if (pair === "USDINR") return Math.max(0.01, mid * 0.0004); // ~4 bps
  if (pair === "USDJPY") return Math.max(0.006, mid * 0.0003);
  return Math.max(0.00008, mid * 0.00015);
}

function stubRates(pair: string): { rBase: number; rQuote: number } {
  // very rough short-rate guesses (annualized)
  // Pair notation: BASEQUOTE (e.g., EURUSD => rBase=EUR rate, rQuote=USD rate)
  const baseCcy = pair.slice(0, 3);
  const quoteCcy = pair.slice(3, 6);
  const r: Record<string, number> = {
    USD: 0.052, EUR: 0.037, GBP: 0.053, JPY: 0.005, INR: 0.065, AUD: 0.045, CAD: 0.047,
  };
  return { rBase: r[baseCcy] ?? 0.03, rQuote: r[quoteCcy] ?? 0.03 };
}

function baseIV(pair: string): number {
  // base ATM IV level (annualized) by pair
  const map: Record<string, number> = {
    EURUSD: 0.085,
    GBPUSD: 0.095,
    USDJPY: 0.10,
    USDINR: 0.07,
    AUDUSD: 0.11,
    USDCAD: 0.095,
  };
  return map[pair] ?? 0.09;
}

function termBump(tenor: string): number {
  switch (tenor) {
    case "1W": return -0.015;
    case "1M": return -0.005;
    case "3M": return 0.0;
    case "6M": return 0.004;
    case "1Y": return 0.008;
    default: return 0;
  }
}

function guessMeta(pair: string) {
  const b = pair.slice(0, 3), q = pair.slice(3, 6);
  return { base: b, quote: q, pip: b === "USD" || q === "JPY" ? 0.01 : 0.0001, session: "24x5" };
}