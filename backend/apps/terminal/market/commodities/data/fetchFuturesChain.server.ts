// fetchFuturesChain.server.ts
// Server function to fetch futures chain data (mock or via API)

// Type for a single futures contract
export type FuturesContract = {
  symbol: string;
  expiry: string; // ISO date
  strike: number;
  type: "CALL" | "PUT" | "FUT";
  last: number;
  change: number;
  volume: number;
  oi: number; // open interest
};

// Mock generator
function generateFuturesChain(baseSymbol = "NIFTY", basePrice = 22000): FuturesContract[] {
  const expiries = ["2025-09-25", "2025-10-02", "2025-10-30"];
  const strikes = Array.from({ length: 9 }, (_, i) => basePrice - 400 + i * 100);

  const contracts: FuturesContract[] = [];

  for (const expiry of expiries) {
    contracts.push({
      symbol: `${baseSymbol}-${expiry}-FUT`,
      expiry,
      strike: basePrice,
      type: "FUT",
      last: basePrice + (Math.random() - 0.5) * 200,
      change: (Math.random() - 0.5) * 50,
      volume: Math.floor(Math.random() * 100000),
      oi: Math.floor(Math.random() * 500000),
    });

    for (const strike of strikes) {
      contracts.push({
        symbol: `${baseSymbol}-${expiry}-C${strike}`,
        expiry,
        strike,
        type: "CALL",
        last: Math.max(1, (basePrice - strike) * 0.5 + Math.random() * 50),
        change: (Math.random() - 0.5) * 10,
        volume: Math.floor(Math.random() * 20000),
        oi: Math.floor(Math.random() * 100000),
      });
      contracts.push({
        symbol: `${baseSymbol}-${expiry}-P${strike}`,
        expiry,
        strike,
        type: "PUT",
        last: Math.max(1, (strike - basePrice) * 0.5 + Math.random() * 50),
        change: (Math.random() - 0.5) * 10,
        volume: Math.floor(Math.random() * 20000),
        oi: Math.floor(Math.random() * 100000),
      });
    }
  }

  return contracts;
}

export async function fetchFuturesChain(symbol = "NIFTY"): Promise<FuturesContract[]> {
  return generateFuturesChain(symbol);
}