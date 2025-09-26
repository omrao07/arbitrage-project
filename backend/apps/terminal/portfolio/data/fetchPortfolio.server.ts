// lib/fetchportfolio.server.ts
// No imports. Server-side helper to build a normalized portfolio snapshot
// from either provided objects or remote JSON endpoints.
// - Graceful fallbacks: any missing piece defaults to empty
// - Symbols are uppercased and trimmed
// - Returns computed totals, P&L, weights, and simple exposures.
//
// Example usage:
//
//   const snap = await fetchPortfolio({
//     endpoints: {
//       positions: "https://api.example.com/positions",  // GET -> [{symbol, qty, avgCost?, name?, sector?, currency?}]
//       prices:    "https://api.example.com/prices",     // GET -> { "AAPL": 226.4, ... }
//       cash:      "https://api.example.com/cash"        // GET -> { cash: 1200.50 }
//     },
//     headers: { Authorization: `Bearer ${token}` },
//     timeoutMs: 4000,
//   });
//
//   // or provide data inline (remote fetch skipped)
//   const snap = await fetchPortfolio({
//     positions: [{ symbol: "AAPL", qty: 10, avgCost: 200, sector: "Tech" }],
//     prices: { AAPL: 226.4 },
//     cash: 1000,
//   });

"use server";

/* =================================== Types =================================== */

export type PositionInput = {
  symbol: string;
  qty: number;
  avgCost?: number;
  name?: string;
  sector?: string;
  currency?: string; // e.g., "USD", "INR"
};

export type FetchPortfolioInput = {
  // Inline data (optional). If provided, these take precedence over endpoints.
  positions?: PositionInput[];
  prices?: Record<string, number>;
  cash?: number;

  // Remote endpoints (optional). All are simple GET returning JSON.
  endpoints?: {
    positions?: string; // -> PositionInput[]
    prices?: string;    // -> Record<string, number>
    cash?: string;      // -> { cash: number } or number
  };

  headers?: Record<string, string>; // headers for fetch calls
  timeoutMs?: number;               // default 3500
  userAgent?: string;               // optional User-Agent override
};

export type PositionRow = {
  symbol: string;
  name?: string;
  sector?: string;
  currency?: string;
  qty: number;
  price: number;
  avgCost: number;
  value: number;
  cost: number;
  pnl: number;
  pnlPct: number;
  weight: number; // out of equity market value (ex-cash)
};

export type PortfolioSnapshot = {
  positions: PositionRow[];
  cash: number;
  totals: {
    value: number;     // equity MV (sum of positions.value)
    cost: number;
    pnl: number;
    pnlPct: number;    // cost-based
    gross: number;     // value + cash
    net: number;       // same as gross (placeholder for fees etc.)
  };
  exposures: {
    sector: Record<string, number>;   // weights 0..1 of equity MV
    currency: Record<string, number>; // weights 0..1 of equity MV
  };
  symbols: string[];
  updatedAt: string; // ISO
  errors: string[];
};

/* =================================== Public API =================================== */

export async function fetchPortfolio(input: FetchPortfolioInput | FormData): Promise<PortfolioSnapshot> {
  const nowIso = new Date().toISOString();
  const errors: string[] = [];

  // -------- normalize input or fetch endpoints --------
  let inPositions: PositionInput[] | undefined;
  let inPrices: Record<string, number> | undefined;
  let inCash: number | undefined;

  if (isFormData(input)) {
    // Accept JSON strings in FormData under keys: positions, prices, cash
    inPositions = readJson<PositionInput[]>(input.get("positions"));
    inPrices = readJson<Record<string, number>>(input.get("prices"));
    const cashObj = readJson<{ cash?: number }>(input.get("cash"));
    inCash = num(input.get("cash"));
    if (!Number.isFinite(inCash as number) && cashObj && Number.isFinite(cashObj.cash as number)) {
      inCash = cashObj.cash!;
    }
  } else {
    inPositions = input.positions;
    inPrices = input.prices;
    inCash = input.cash;
  }

  // If any piece is still missing, try endpoints
  const eps = isFormData(input) ? undefined : input.endpoints;
  const headers = isFormData(input) ? undefined : input.headers;
  const timeoutMs = isFormData(input) ? undefined : input.timeoutMs;
  const userAgent = isFormData(input) ? undefined : input.userAgent;

  if (!inPositions && eps?.positions) {
    try {
      const arr = await fetchJson(eps.positions, timeoutMs ?? 3500, headers, userAgent);
      if (Array.isArray(arr)) inPositions = arr as PositionInput[];
      else errors.push("positions endpoint returned non-array");
    } catch (e: any) {
      errors.push(`positions fetch failed: ${e?.message || "error"}`);
    }
  }
  if (!inPrices && eps?.prices) {
    try {
      const obj = await fetchJson(eps.prices, timeoutMs ?? 3500, headers, userAgent);
      if (obj && typeof obj === "object") inPrices = obj as Record<string, number>;
      else errors.push("prices endpoint returned non-object");
    } catch (e: any) {
      errors.push(`prices fetch failed: ${e?.message || "error"}`);
    }
  }
  if (typeof inCash !== "number" && eps?.cash) {
    try {
      const val = await fetchJson(eps.cash, timeoutMs ?? 3500, headers, userAgent);
      if (typeof val === "number") inCash = val;
      else if (val && typeof val === "object" && Number.isFinite((val as any).cash)) inCash = (val as any).cash;
      else errors.push("cash endpoint returned invalid payload");
    } catch (e: any) {
      errors.push(`cash fetch failed: ${e?.message || "error"}`);
    }
  }

  // -------- safe defaults --------
  const positions = Array.isArray(inPositions) ? inPositions : [];
  const priceMap = inPrices || {};
  const cash = Number.isFinite(inCash as number) ? (inCash as number) : 0;

  // -------- normalize + compute --------
  const priceU = upperKeys(priceMap);
  const rows: PositionRow[] = [];

  for (const p of positions) {
    const sym = (p.symbol || "").trim().toUpperCase();
    if (!sym) continue;

    const qty = clampNonNeg(num(p.qty), 0);
    const price = clampNonNeg(num(priceU[sym]), 0);
    const avg = clampNonNeg(num(p.avgCost), 0);
    const value = qty * price;
    const cost = qty * avg;
    const pnl = value - cost;
    const pnlPct = cost > 0 ? pnl / cost : 0;

    rows.push({
      symbol: sym,
      name: str(p.name),
      sector: str(p.sector),
      currency: str(p.currency),
      qty, price, avgCost: avg,
      value, cost, pnl, pnlPct,
      weight: 0, // filled later
    });
  }

  // Totals + weights
  const totalValue = rows.reduce((s, r) => s + r.value, 0);
  const totalCost = rows.reduce((s, r) => s + r.cost, 0);
  const totalPnL = totalValue - totalCost;
  const pnlPct = totalCost > 0 ? totalPnL / totalCost : 0;

  rows.forEach((r) => (r.weight = totalValue > 0 ? r.value / totalValue : 0));

  // Exposures (sector, currency) — weights out of equity MV
  const sector: Record<string, number> = {};
  const currency: Record<string, number> = {};
  for (const r of rows) {
    const w = r.weight;
    const sKey = (r.sector || "Unclassified").trim() || "Unclassified";
    const cKey = (r.currency || "—").trim() || "—";
    sector[sKey] = (sector[sKey] || 0) + w;
    currency[cKey] = (currency[cKey] || 0) + w;
  }

  // Final snapshot
  const snapshot: PortfolioSnapshot = {
    positions: rows,
    cash,
    totals: {
      value: totalValue,
      cost: totalCost,
      pnl: totalPnL,
      pnlPct,
      gross: totalValue + cash,
      net: totalValue + cash,
    },
    exposures: { sector, currency },
    symbols: rows.map((r) => r.symbol),
    updatedAt: nowIso,
    errors,
  };

  return snapshot;
}

/* =============================== Helpers =============================== */

async function fetchJson(
  url: string,
  timeoutMs: number,
  headers?: Record<string, string>,
  userAgent?: string,
): Promise<any> {
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : (null as any);
  const timer = ctrl ? setTimeout(() => ctrl.abort(), Math.max(500, Math.min(20000, timeoutMs))) : null;
  try {
    const res = await fetch(url, {
      signal: ctrl?.signal,
      headers: {
        ...(headers || {}),
        ...(userAgent ? { "user-agent": userAgent } : {}),
        accept: "application/json, text/plain, */*",
      },
      // @ts-ignore
      cache: "no-store",
    } as any);
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const txt = await res.text();
    // Try JSON first; if non-JSON, still attempt to parse
    if (/application\/json|text\/json/.test(ct) || /^[\[{]/.test(txt.trim())) {
      return JSON.parse(txt);
    }
    // Not JSON — return as-is (caller may handle number / object shapes)
    return txt;
  } finally {
    if (timer) clearTimeout(timer as any);
  }
}

function isFormData(x: any): x is FormData {
  return typeof x === "object" && x?.constructor?.name === "FormData";
}
function readJson<T>(v: any): T | undefined {
  const s = str(v);
  if (!s) return undefined;
  try { return JSON.parse(s) as T; } catch { return undefined; }
}
function str(v: any): string | undefined {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}
function num(v: any): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}
function clampNonNeg(n: number, fallback = 0) {
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}
function upperKeys<T extends Record<string, any>>(obj: T) {
  const out: any = {};
  for (const k of Object.keys(obj || {})) out[String(k).toUpperCase()] = obj[k];
  return out as T;
}

/* ========================== Convenience variants ========================== */

/** Minimal inline-friendly variant. */
export async function buildSnapshot(
  positions: PositionInput[],
  prices: Record<string, number>,
  cash = 0,
): Promise<PortfolioSnapshot> {
  return fetchPortfolio({ positions, prices, cash });
}

/** Fetch using only endpoints + optional headers/timeouts. */
export async function fetchPortfolioFromEndpoints(
  endpoints: NonNullable<FetchPortfolioInput["endpoints"]>,
  headers?: Record<string, string>,
  timeoutMs?: number,
): Promise<PortfolioSnapshot> {
  return fetchPortfolio({ endpoints, headers, timeoutMs });
}
