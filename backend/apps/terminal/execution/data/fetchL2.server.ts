"use server";

/**
 * fetchL2.server.ts
 * - Server action to fetch L2 order book for a symbol
 * - Normalizes bids/asks, sorts, and returns top N
 * - Replace the mock section with your broker/market data adapter
 */

export type L2Level = {
  price: number;
  size: number;
  orders?: number;
};

export type L2Snapshot = {
  symbol: string;
  ts: number;           // epoch ms
  bids: L2Level[];      // sorted desc by price
  asks: L2Level[];      // sorted asc by price
};

export interface FetchL2Input {
  symbol: string;
  depth?: number;       // number of levels per side
}

export async function fetchL2(input: FetchL2Input): Promise<L2Snapshot> {
  const { symbol, depth = 10 } = input;
  if (!symbol) throw new Error("symbol is required");

  // TODO: swap this out with your adapter (websocket snapshot, REST call, etc.)
  // Example: await broker.getOrderBook(symbol, { depth });
  const mock = mockOrderBook(symbol, depth);

  // normalize: ensure arrays, strip invalid levels
  const cleanBids = (mock.bids || [])
    .filter(valid)
    .sort((a, b) => b.price - a.price)
    .slice(0, depth);

  const cleanAsks = (mock.asks || [])
    .filter(valid)
    .sort((a, b) => a.price - b.price)
    .slice(0, depth);

  return {
    symbol,
    ts: Date.now(),
    bids: cleanBids,
    asks: cleanAsks,
  };
}

/* ---------------- helpers ---------------- */

function valid(l: any): l is L2Level {
  return l && Number.isFinite(l.price) && Number.isFinite(l.size) && l.size > 0;
}

// mock generator: creates fake L2 book around a ref price
function mockOrderBook(symbol: string, depth: number): { bids: L2Level[]; asks: L2Level[] } {
  const ref = 100 + (symbol.charCodeAt(0) % 50); // vary by symbol
  const bids: L2Level[] = [];
  const asks: L2Level[] = [];
  for (let i = 0; i < depth; i++) {
    bids.push({ price: +(ref - i * 0.01).toFixed(2), size: Math.floor(Math.random() * 1000) + 1, orders: Math.floor(Math.random() * 5) + 1 });
    asks.push({ price: +(ref + i * 0.01).toFixed(2), size: Math.floor(Math.random() * 1000) + 1, orders: Math.floor(Math.random() * 5) + 1 });
  }
  return { bids, asks };
}