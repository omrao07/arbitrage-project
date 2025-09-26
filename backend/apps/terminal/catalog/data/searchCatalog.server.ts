// apps/terminal/app/(workspaces)/catalog/_data/searchCatalog.server.ts
// Server-only helper to search the catalog.
// Usage (server component / server action):
//   const results = await searchCatalog({ query: "launchpad", platform: "Bloomberg" });

import "server-only";

export type CatalogItem = {
  id: string;
  platform: "Bloomberg" | "Koyfin" | "Hammer" | string;
  category: string;
  assetClass: string;
  subClass?: string;
  type: "function" | "feature" | "ticker" | "screen" | string;
  code: string;
  name: string;
  description: string;
  aliases?: string[];
  tags?: string[];
  region?: string;
  overlaps?: string[];
  source?: string;
};

export type SearchCatalogReq = {
  query?: string;         // free text search
  platform?: string;      // filter by platform
  category?: string;      // filter by category
  assetClass?: string;    // filter by asset class
  uniqueOnly?: boolean;   // only return unique features
  limit?: number;         // max results
  offset?: number;        // for pagination
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.API_BASE_URL ||
  "";

/** build endpoint for backend vs local route */
function endpoint(): string {
  if (BACKEND_URL) {
    return `${BACKEND_URL.replace(/\/+$/, "")}/catalog/search`;
  }
  // local Next route fallback
  return "/api/catalog/search";
}

export async function searchCatalog(req: SearchCatalogReq): Promise<CatalogItem[]> {
  const payload: SearchCatalogReq = {
    limit: 50,
    offset: 0,
    ...req,
  };

  const url = endpoint();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`searchCatalog: ${res.status} ${res.statusText} ${text}`);
  }

  const raw = await res.json();
  if (Array.isArray(raw)) return raw.map(normalize);
  if (raw?.items && Array.isArray(raw.items)) return raw.items.map(normalize);
  return [];
}

/** normalize arbitrary backend shapes into CatalogItem */
function normalize(raw: any): CatalogItem {
  return {
    id: String(raw.id ?? raw.code ?? "unknown"),
    platform: raw.platform ?? "Other",
    category: raw.category ?? "Other",
    assetClass: raw.assetClass ?? "Other",
    subClass: raw.subClass,
    type: raw.type ?? "feature",
    code: raw.code ?? "",
    name: raw.name ?? "",
    description: raw.description ?? "",
    aliases: raw.aliases ?? [],
    tags: raw.tags ?? [],
    region: raw.region,
    overlaps: raw.overlaps ?? [],
    source: raw.source,
  };
}