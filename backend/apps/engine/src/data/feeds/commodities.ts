// data/feeds/commodities.ts
// Commodity market data feed adapter (pure Node, no imports)
//
// Provides a simple interface to fetch commodity spot prices,
// futures chain snapshots, and basic metadata. This is stubbed
// with demo data — replace with a real API (Bloomberg, Quandl,
// CME, etc.) for production use.

export function CommoditiesFeed(opts: any = {}) {
  const state = {
    name: "commodities-feed",
    connected: false,
    lastUpdate: 0,
    latencyMs: 50,
  };

  function isConnected() {
    return state.connected;
  }

  async function connect() {
    // stub connect
    state.connected = true;
    state.lastUpdate = Date.now();
    return { ok: true, msg: "connected to commodities feed" };
  }

  async function disconnect() {
    state.connected = false;
    return { ok: true, msg: "disconnected" };
  }

  async function getSpotPrices() {
    if (!state.connected) return { ok: false, error: "not connected" };

    // stub: pretend API latency
    await delay(state.latencyMs);

    return {
      ok: true,
      prices: {
        GOLD: 1923.45,
        SILVER: 24.18,
        CRUDE_OIL: 83.5,
        NATGAS: 2.75,
        COPPER: 3.89,
      },
      ts: new Date().toISOString(),
    };
  }

  async function getFuturesChain(symbol: string) {
    if (!state.connected) return { ok: false, error: "not connected" };

    await delay(state.latencyMs);

    // stub futures chain (monthly maturities)
    const today = new Date();
    const chain = [];
    for (let i = 1; i <= 6; i++) {
      const d = new Date(today);
      d.setMonth(today.getMonth() + i);
      chain.push({
        expiry: d.toISOString().slice(0, 10),
        price: 1900 + Math.random() * 100,
      });
    }

    return { ok: true, symbol, chain };
  }

  async function getMetadata(symbol: string) {
    return {
      ok: true,
      symbol,
      description: {
        GOLD: "Gold (XAU)",
        SILVER: "Silver (XAG)",
        CRUDE_OIL: "Crude Oil (WTI)",
        NATGAS: "Natural Gas (Henry Hub)",
        COPPER: "Copper (COMEX)",
      }[symbol] || "Unknown commodity",
    };
  }

  return {
    isConnected,
    connect,
    disconnect,
    getSpotPrices,
    getFuturesChain,
    getMetadata,
  };
}

/* ---------------- Helper ---------------- */

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}