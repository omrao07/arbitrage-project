import "server-only";

export type Position = {
  dayPnl: null;
  dayPct: null;
  id: string;
  symbol: string;
  name?: string;
  assetClass?: string;
  region?: string;
  quantity: number;
  avgPrice?: number;
  marketPrice?: number;
  marketValue?: number;
  pnl?: number;
  currency?: string;
  tags?: string[];
  updatedAt?: string;
};

export type FetchPositionsReq = {
  accountId?: string;
  strategy?: string;
  date?: string;
  region?: string;
  assetClass?: string;
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.API_BASE_URL ||
  "";

function endpoint(): string {
  return BACKEND_URL
    ? `${BACKEND_URL.replace(/\/+$/, "")}/portfolio/positions`
    : "/api/portfolio/positions";
}

export async function fetchPositions(req: FetchPositionsReq = {}): Promise<Position[]> {
  const res = await fetch(endpoint(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`fetchPositions: ${res.status} ${res.statusText} ${text}`);
  }

  const raw = await res.json();
  if (Array.isArray(raw)) return raw.map(normalize);
  if (raw?.positions && Array.isArray(raw.positions)) return raw.positions.map(normalize);
  return [];
}

function normalize(r: any): Position {
  const qty = Number(r.quantity ?? r.qty ?? 0);
  const mkt = Number(r.marketPrice ?? r.price ?? 0);

  return {
    id: String(r.id ?? r.symbol ?? ""),
    symbol: String(r.symbol ?? ""),
    name: r.name ?? "",
    assetClass: r.assetClass ?? r.class ?? "Other",
    region: r.region,
    quantity: qty,
    avgPrice: num(r.avgPrice ?? r.cost),
    marketPrice: num(mkt),
    marketValue: Number.isFinite(qty * mkt) ? qty * mkt : num(r.marketValue),
    pnl: num(r.pnl ?? r.unrealized),
    currency: r.currency ?? "USD",
    tags: r.tags ?? [],
    updatedAt: r.updatedAt ?? new Date().toISOString(),
    dayPnl: null,
    dayPct: null,
   
};
}

function num(v: any): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}