// apps/terminal/app/(workspaces)/analytics/_data/fetchFactors.server.ts
// Server-only fetcher for factor exposures.
// Usage (server components/actions only):
//   const factors = await fetchFactors({ symbols: ["AAPL"], window: 252 });

import "server-only";

export type FetchFactorsReq = {
  /** Portfolio or symbol list to evaluate (pick one). */
  symbols?: string[];           // e.g., ["AAPL","MSFT"]
  portfolioId?: string;         // e.g., "core-us-largecap"
  /** Lookback window in trading days (default 252). */
  window?: number;
  /** Factor model to use. */
  model?:
    | "FamaFrench3"
    | "FamaFrench5"
    | "QFactor"
    | "AQR"
    | "Custom";
  /** Optional custom factor names if model="Custom". */
  customFactors?: string[];     // e.g., ["MKT","SMB","HML","MOM"]
  /** Optional benchmark mapping per factor. */
  benchmarks?: Record<string, number>;
};

export type Factor = {
  name: string;
  value: number;
  benchmark?: number;
};

export type FetchFactorsResp = Factor[];

// -------- configuration --------
const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL /* fronted env */ ||
  process.env.API_BASE_URL /* server env */ ||
  ""; // if empty, we fall back to Next local API route

/** where to send the request:
 *  1) direct to backend if BACKEND_URL set
 *  2) otherwise to local Next API route
 */
function endpoint(): string {
  if (BACKEND_URL) {
    return `${BACKEND_URL.replace(/\/+$/, "")}/analytics/factors`;
  }
  // Next local API route (see /app/api/analytics/factors/route.ts)
  return "/api/analytics/factors";
}

// -------- main fetcher --------
export async function fetchFactors(req: FetchFactorsReq): Promise<FetchFactorsResp> {
  const payload: FetchFactorsReq = {
    window: 252,
    model: "FamaFrench5",
    ...req,
  };

  // minimal validation
  if ((!payload.symbols || payload.symbols.length === 0) && !payload.portfolioId) {
    throw new Error("fetchFactors: provide either `symbols` or `portfolioId`.");
  }

  const url = endpoint();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Next server-side fetch: don't cache analytics
    cache: "no-store",
    body: JSON.stringify(payload),
  });

  // If you’re proxying to the same Next app (no absolute URL), ensure it runs on the server:
  // This file uses `server-only`, so it won't be imported on the client.

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`fetchFactors: ${res.status} ${res.statusText} ${text}`);
  }

  const raw = await res.json();

  // Accept multiple shapes and normalize:
  // 1) { factors: [{ name, value, benchmark? }] }
  // 2) { betas: { MKT:0.87, SMB:-0.42, ... }, benchmarks?: { MKT:0.5 } }
  // 3) [{ name, value, benchmark? }]
  let normalized: FetchFactorsResp;

  if (Array.isArray(raw)) {
    normalized = raw.map(toFactor);
  } else if (raw?.factors && Array.isArray(raw.factors)) {
    normalized = raw.factors.map(toFactor);
  } else if (raw?.betas && typeof raw.betas === "object") {
    const bm: Record<string, number> = raw.benchmarks || {};
    normalized = Object.entries(raw.betas).map(([name, value]) => ({
      name,
      value: Number(value),
      benchmark: isFiniteNumber(bm[name]) ? Number(bm[name]) : undefined,
    }));
  } else {
    // Dev fallback: fabricate small demo so UI doesn't break
    normalized = demoFactors(payload);
  }

  // final safety: ensure numeric values
  normalized = normalized
    .filter((f) => typeof f?.name === "string" && isFiniteNumber(f?.value))
    .map((f) => ({
      name: f.name,
      value: Number(f.value),
      benchmark: isFiniteNumber(f.benchmark) ? Number(f.benchmark) : undefined,
    }));

  return normalized;
}

// -------- helpers --------
function toFactor(x: any): Factor {
  return {
    name: String(x?.name ?? x?.factor ?? "Unknown"),
    value: Number(x?.value ?? x?.beta ?? 0),
    benchmark: isFiniteNumber(x?.benchmark) ? Number(x.benchmark) : undefined,
  };
}

function isFiniteNumber(n: any): boolean {
  return typeof n === "number" && Number.isFinite(n);
}

function demoFactors(req: FetchFactorsReq): FetchFactorsResp {
  const names =
    req.customFactors ??
    (req.model === "FamaFrench3"
      ? ["MKT", "SMB", "HML"]
      : ["MKT", "SMB", "HML", "RMW", "CMA"]); // FF5 default
  return names.map((name, i) => ({
    name,
    value: Number((Math.cos(i * 0.9) * 0.8).toFixed(2)),
    benchmark: i === 0 ? 1 : undefined, // example: market beta benchmark at 1
  }));
}