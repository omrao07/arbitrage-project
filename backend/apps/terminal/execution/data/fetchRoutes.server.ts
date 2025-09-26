"use server";

/**
 * fetchRoutes.server.ts
 * - Returns execution routes / venues for a given account/symbol
 * - Replace mockRoutes() with your broker/OMS adapter
 */

export type RouteStatus = "live" | "delayed" | "offline";

export interface RouteInfo {
  id: string;                // e.g., "SMART", "NYSE", "NASDAQ", "DARK_ARCA"
  label: string;             // user-facing label
  description?: string;      // optional details
  status: RouteStatus;       // venue health
  assetClasses: string[];    // supported: ["Equity","ETF","Options","Futures","FX","Crypto"]
  supportsIOC?: boolean;
  supportsFOK?: boolean;
  supportsPostOnly?: boolean;
  supportsReduceOnly?: boolean;
  lotSize?: number;          // min order increment if applicable
  feeBps?: number;           // indicative fee in basis points (optional)
  region?: string;           // e.g., "US", "EU", "IN"
  latencyMs?: number;        // indicative venue latency
}

export interface FetchRoutesInput {
  accountId?: string;
  symbol?: string;
  assetClass?: string;       // filter (e.g., "Equity")
  region?: string;           // filter by region
  includeOffline?: boolean;  // default false
}

export interface FetchRoutesResult {
  ts: number;
  routes: RouteInfo[];
}

export async function fetchRoutes(input: FetchRoutesInput = {}): Promise<FetchRoutesResult> {
  const { accountId, symbol, assetClass, region, includeOffline = false } = input;

  // TODO: replace with real adapter call, e.g.:
  // const routes = await broker.getRoutes({ accountId, symbol });
  const routes = mockRoutes(accountId, symbol);

  const filtered = routes.filter((r) => {
    if (assetClass && !r.assetClasses.includes(assetClass)) return false;
    if (region && r.region !== region) return false;
    if (!includeOffline && r.status === "offline") return false;
    return true;
  });

  return {
    ts: Date.now(),
    routes: filtered,
  };
}

/* ---------------- mock adapter ---------------- */

function mockRoutes(accountId?: string, symbol?: string): RouteInfo[] {
  // lightweight variation by symbol/account for realism
  const baseEquity = [
    {
      id: "SMART",
      label: "Smart Router",
      description: "Auto-select best venue across lit/dark pools",
      status: "live",
      assetClasses: ["Equity", "ETF"],
      supportsIOC: true,
      supportsFOK: true,
      supportsPostOnly: true,
      lotSize: 1,
      feeBps: 0.15,
      region: "US",
      latencyMs: 3,
    },
    {
      id: "NYSE",
      label: "NYSE",
      description: "New York Stock Exchange lit book",
      status: "live",
      assetClasses: ["Equity", "ETF"],
      supportsIOC: true,
      supportsFOK: true,
      lotSize: 1,
      feeBps: 0.18,
      region: "US",
      latencyMs: 4,
    },
    {
      id: "NASDAQ",
      label: "NASDAQ",
      description: "NASDAQ continuous book",
      status: "live",
      assetClasses: ["Equity", "ETF"],
      supportsIOC: true,
      supportsFOK: true,
      lotSize: 1,
      feeBps: 0.17,
      region: "US",
      latencyMs: 4,
    },
    {
      id: "ARCA_DARK",
      label: "ARCA Dark",
      description: "Hidden liquidity, midpoint pricing",
      status: "delayed",
      assetClasses: ["Equity", "ETF"],
      supportsIOC: true,
      supportsPostOnly: true,
      lotSize: 1,
      feeBps: 0.12,
      region: "US",
      latencyMs: 7,
    },
    {
      id: "IEX",
      label: "IEX",
      description: "Lit venue with speed bump",
      status: "live",
      assetClasses: ["Equity", "ETF"],
      supportsIOC: true,
      lotSize: 1,
      feeBps: 0.16,
      region: "US",
      latencyMs: 5,
    },
    {
      id: "DARK_GENERIC",
      label: "Dark Pool (Generic)",
      description: "Crossing network; conditional only",
      status: "offline",
      assetClasses: ["Equity", "ETF"],
      supportsPostOnly: true,
      lotSize: 1,
      feeBps: 0.1,
      region: "US",
      latencyMs: 10,
    },
  ] as RouteInfo[];

  const futs = [
    {
      id: "CME_GLOBEX",
      label: "CME Globex",
      description: "US futures (CME/CBOT/NYMEX/COMEX)",
      status: "live",
      assetClasses: ["Futures"],
      supportsIOC: true,
      supportsFOK: true,
      lotSize: 1,
      feeBps: 0.0,
      region: "US",
      latencyMs: 6,
    },
  ] as RouteInfo[];

  const fx = [
    {
      id: "FX_ECN",
      label: "FX ECN",
      description: "Multi-bank ECN with smart streams",
      status: "live",
      assetClasses: ["FX"],
      supportsIOC: true,
      supportsFOK: true,
      lotSize: 1000,
      feeBps: 0.0,
      region: "Global",
      latencyMs: 8,
    },
  ] as RouteInfo[];

  const crypto = [
    {
      id: "BINANCE",
      label: "Binance",
      description: "Crypto spot/perp",
      status: "live",
      assetClasses: ["Crypto"],
      supportsIOC: true,
      supportsFOK: true,
      supportsReduceOnly: true,
      lotSize: 0.0001,
      feeBps: 2.5,
      region: "Global",
      latencyMs: 20,
    },
    {
      id: "COINBASE",
      label: "Coinbase",
      description: "Crypto spot",
      status: "live",
      assetClasses: ["Crypto"],
      supportsIOC: true,
      lotSize: 0.0001,
      feeBps: 3.0,
      region: "US",
      latencyMs: 22,
    },
  ] as RouteInfo[];

  const opt = [
    {
      id: "OPRA_SMART",
      label: "Options Smart",
      description: "Routes across OPRA venues",
      status: "live",
      assetClasses: ["Options"],
      supportsIOC: true,
      lotSize: 1,
      feeBps: 0.0,
      region: "US",
      latencyMs: 7,
    },
  ] as RouteInfo[];

  const all: RouteInfo[] = [
    ...baseEquity,
    ...futs,
    ...fx,
    ...crypto,
    ...opt,
  ];

  // tiny flavor by symbol/account (optional)
  if (symbol?.endsWith("-USD")) {
    // crypto symbols → prioritize crypto venues on top
    return [...crypto, ...baseEquity, ...futs, ...fx, ...opt];
  }
  if (accountId?.startsWith("IN-")) {
    // pretend US lit venues are delayed for this region
    return all.map((r) =>
      r.region === "US" && (r.id === "NYSE" || r.id === "NASDAQ")
        ? { ...r, status: "delayed" as RouteStatus }
        : r
    );
  }
  return all;
}