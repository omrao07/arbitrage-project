// apps/terminal/app/(workspaces)/catalog/_data/getItem.server.ts
// Server-only helper to fetch a single catalog item by id.
//
// Usage (server components/actions):
//   const item = await getItem("bloomberg-extra-utilities.EASY");

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

const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.API_BASE_URL ||
  "";

/** Build endpoint for backend vs local route */
function endpoint(id: string): string {
  if (BACKEND_URL) {
    return `${BACKEND_URL.replace(/\/+$/, "")}/catalog/item/${encodeURIComponent(id)}`;
  }
  // local Next route fallback: /app/api/catalog/[id]/route.ts
  return `/api/catalog/${encodeURIComponent(id)}`;
}

export async function getItem(id: string): Promise<CatalogItem | null> {
  if (!id) throw new Error("getItem: missing id");

  const url = endpoint(id);
  const res = await fetch(url, {
    method: "GET",
    headers: { "Accept": "application/json" },
    cache: "no-store",
  });

  if (!res.ok) {
    if (res.status === 404) return null;
    const text = await res.text().catch(() => "");
    throw new Error(`getItem: ${res.status} ${res.statusText} ${text}`);
  }

  const raw = await res.json();
  return normalize(raw);
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