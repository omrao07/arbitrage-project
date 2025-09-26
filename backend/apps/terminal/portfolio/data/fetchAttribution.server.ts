// lib/fetchattribution.server.ts
// No imports. Server-side helpers to build normalized "attribution sources"
// from a list of item URLs (e.g., news articles). Light HTML scraping is used
// to derive site name, favicon, and license hints. Results are deduped by origin.
//
// Usage:
//   const { sources } = await fetchAttribution([
//     { url: "https://www.reuters.com/markets/asia/...", source: "Reuters" },
//     { url: "https://www.bloomberg.com/news/articles/...", note: "Paywalled" },
//   ]);
//
//   // pass `sources` straight to <AttributionPanel sources={sources} />
//
// Notes:
// - Network requests are time-limited; if a site doesn't respond quickly we
//   fall back to sensible defaults (hostname, /favicon.ico, unknown license).
// - We cache basic site metadata per-origin for this server's process lifetime.

"use server";

/* ===================== Types ===================== */

export type RawAttribution = {
  url?: string;       // web page for the item (article, doc, dataset, etc.)
  source?: string;    // preferred human name for the site/publisher
  license?: string;   // explicit license string if known
  note?: string;      // short descriptor (e.g., "Paywalled", "Press Release")
  logoUrl?: string;   // explicit logo/fav icon if known
};

export type AttributionSource = {
  name: string;       // required: human-readable source name
  url?: string;       // homepage or canonical URL for the source
  license?: string;   // best-effort detected or provided
  logoUrl?: string;   // small icon/logo
  note?: string;      // optional note
};

export type FetchAttributionResult = {
  sources: AttributionSource[];
  errors: string[]; // non-fatal info about fetch failures
};

export type FetchOpts = {
  timeoutMs?: number; // default 3500
  userAgent?: string; // optional UA header
};

/* ===================== Public API ===================== */

export async function fetchAttribution(
  items: RawAttribution[],
  opts?: FetchOpts,
): Promise<FetchAttributionResult> {
  const errors: string[] = [];
  const byOrigin = new Map<string, RawAttribution[]>();

  for (const it of items || []) {
    const u = safeUrl(it.url);
    if (!u) continue;
    const origin = u.origin;
    if (!byOrigin.has(origin)) byOrigin.set(origin, []);
    byOrigin.get(origin)!.push(it);
  }

  const tasks = Array.from(byOrigin.entries()).map(async ([origin, group]) => {
    const bestHint = pickBestHint(group);
    try {
      const meta = await getSiteMeta(origin, opts);
      return toAttribution(origin, bestHint, meta);
    } catch (e: any) {
      errors.push(`${origin}: ${e?.message || "fetch failed"}`);
      // fallback attribution even if fetch failed
      return toAttribution(origin, bestHint, { name: hostLabel(origin), home: origin, icon: `${origin}/favicon.ico` });
    }
  });

  const sources = (await Promise.all(tasks))
    .filter(Boolean) as AttributionSource[];

  // If caller provided items with no URLs, include them as standalone rows
  for (const it of items || []) {
    if (!it.url && it.source) {
      sources.push({
        name: it.source,
        license: it.license,
        logoUrl: it.logoUrl,
        note: it.note,
      });
    }
  }

  // Dedupe by (name, url) with stable order
  const seen = new Set<string>();
  const deduped: AttributionSource[] = [];
  for (const s of sources) {
    const k = `${(s.name || "").toLowerCase()}|${(s.url || "").toLowerCase()}`;
    if (seen.has(k)) continue;
    seen.add(k);
    deduped.push(s);
  }

  return { sources: deduped, errors };
}

/* ===================== Cache & Fetch ===================== */

type SiteMeta = {
  name?: string;   // site/publisher name
  home?: string;   // canonical homepage (origin)
  icon?: string;   // absolute URL to an icon
  license?: string;
};

const metaCache = new Map<string, SiteMeta>(); // key = origin

async function getSiteMeta(origin: string, opts?: FetchOpts): Promise<SiteMeta> {
  if (metaCache.has(origin)) return metaCache.get(origin)!;

  const timeout = Math.max(500, Math.min(20_000, opts?.timeoutMs ?? 3500));
  const home = origin; // start with origin (e.g., https://example.com)

  // Try homepage HTML for meta tags
  const html = await fetchText(home, timeout, opts?.userAgent);
  const name =
    pickMeta(html, /property=["']og:site_name["']\s+content=["']([^"']+)["']/i) ||
    pickMeta(html, /name=["']application-name["']\s+content=["']([^"']+)["']/i) ||
    pickTitle(html) ||
    hostLabel(origin);

  const icon =
    absolutize(findIconHref(html), origin) ||
    `${origin}/favicon.ico`;

  const license =
    detectLicense(html) || undefined;

  const meta: SiteMeta = { name, home, icon, license };
  metaCache.set(origin, meta);
  return meta;
}

async function fetchText(url: string, timeoutMs: number, userAgent?: string): Promise<string> {
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : (null as any);
  const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
  try {
    const res = await fetch(url, {
      signal: ctrl?.signal,
      headers: userAgent ? { "user-agent": userAgent } : undefined,
      // Avoid huge transfers by hinting—servers may still ignore it.
      // @ts-ignore
      cache: "force-cache",
    } as any);
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    if (!/text\/html|application\/xhtml\+xml/.test(ct)) {
      // not HTML; just return a tiny string
      return "";
    }
    const text = await res.text();
    // Cap processing to the first ~300k chars for speed
    return text.length > 300_000 ? text.slice(0, 300_000) : text;
  } finally {
    if (timer) clearTimeout(timer as any);
  }
}

/* ===================== Extraction helpers ===================== */

function pickMeta(html: string, re: RegExp): string | undefined {
  const m = re.exec(html);
  return m?.[1]?.trim() || undefined;
}

function pickTitle(html: string): string | undefined {
  const m = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html || "");
  if (!m) return undefined;
  const t = decodeHtml(stripTags(m[1]).trim());
  // titles often have pipes/dashes — take the site-like chunk if it's last
  const parts = t.split(/[\|\-–—»]+/).map((x) => x.trim()).filter(Boolean);
  return parts.length > 1 ? parts[parts.length - 1] : t || undefined;
}

function findIconHref(html: string): string | undefined {
  // Prefer high-res touch icons, then SVG/ICO "icon" rels
  const linkRegexes = [
    /<link[^>]+rel=["'](?:apple-touch-icon(?:-precomposed)?)["'][^>]*>/gi,
    /<link[^>]+rel=["']icon["'][^>]*>/gi,
    /<link[^>]+rel=["']shortcut icon["'][^>]*>/gi,
    /<link[^>]+rel=["']mask-icon["'][^>]*>/gi,
  ];
  for (const re of linkRegexes) {
    const iter = html.match(re);
    if (!iter) continue;
    // choose the first with sizes >= 48 if specified
    let best: string | undefined;
    for (const tag of iter) {
      const href = attr(tag, "href");
      if (!href) continue;
      const sizes = (attr(tag, "sizes") || "").toLowerCase();
      if (!best) best = href;
      if (/\b(48x48|57x57|60x60|64x64|72x72|96x96|120x120|128x128|144x144|152x152|180x180|192x192|256x256|512x512)\b/.test(sizes)) {
        return href;
      }
    }
    if (best) return best;
  }
  return undefined;
}

function detectLicense(html: string): string | undefined {
  const s = (html || "").toLowerCase();
  // Common Creative Commons patterns
  const cc = /creativecommons\.org\/licenses\/([a-z\-]+)\/([\d.]+)/.exec(s);
  if (cc) {
    const code = cc[1].toUpperCase().replace(/-/g, "-");
    const ver = cc[2];
    return `CC ${code} ${ver}`;
  }
  if (/all rights reserved/.test(s)) return "All rights reserved";
  const meta = pickMeta(html, /name=["']copyright["']\s+content=["']([^"']+)["']/i)
            || pickMeta(html, /property=["']article:author["']\s+content=["']([^"']+)["']/i);
  return meta ? String(meta) : undefined;
}

function attr(tagHtml: string, name: string): string | undefined {
  const re = new RegExp(`${name}\\s*=\\s*("([^"]+)"|'([^']+)'|([^\\s>]+))`, "i");
  const m = re.exec(tagHtml);
  return (m?.[2] || m?.[3] || m?.[4] || "").trim() || undefined;
}

function stripTags(s: string) {
  return s.replace(/<[^>]*>/g, " ");
}
function decodeHtml(s: string) {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"');
}

function safeUrl(u?: string): URL | undefined {
  if (!u) return undefined;
  try {
    const url = new URL(u);
    if (!/^https?:$/i.test(url.protocol)) return undefined;
    return url;
  } catch {
    return undefined;
  }
}
function absolutize(href?: string, base?: string): string | undefined {
  if (!href) return undefined;
  try {
    return new URL(href, base).toString();
  } catch {
    return undefined;
  }
}
function hostLabel(origin: string) {
  try {
    const h = new URL(origin).hostname.replace(/^www\./, "");
    return h.charAt(0).toUpperCase() + h.slice(1);
  } catch {
    return origin;
  }
}

/* ===================== Assembly ===================== */

function pickBestHint(group: RawAttribution[]): RawAttribution {
  // Prefer one with explicit source/license/logo/note; otherwise any
  const withName = group.find((g) => g.source) || group[0];
  const merged: RawAttribution = { ...withName };
  // If any entry provides license/logo/note, keep them
  for (const g of group) {
    merged.license = merged.license || g.license;
    merged.logoUrl = merged.logoUrl || g.logoUrl;
    merged.note = merged.note || g.note;
  }
  // Always keep the first URL
  merged.url = merged.url || group[0]?.url;
  return merged;
}

function toAttribution(origin: string, hint: RawAttribution, meta: SiteMeta): AttributionSource {
  return {
    name: hint.source || meta.name || hostLabel(origin),
    url: meta.home || origin,
    license: hint.license || meta.license,
    logoUrl: hint.logoUrl || meta.icon || `${origin}/favicon.ico`,
    note: hint.note,
  };
}

/* ===================== Convenience variants ===================== */

/** Build attributions directly from a list of article URLs (strings). */
export async function fetchAttributionFromUrls(
  urls: string[],
  opts?: FetchOpts,
): Promise<FetchAttributionResult> {
  const raws: RawAttribution[] = (urls || []).map((u) => ({ url: u }));
  return fetchAttribution(raws, opts);
}

/** Given generic "items" that have `url` and optional `source`, map and fetch. */
export async function fetchAttributionForItems<T extends { url?: string; source?: string; note?: string; license?: string; logoUrl?: string }>(
  items: T[],
  opts?: FetchOpts,
): Promise<FetchAttributionResult> {
  const raws: RawAttribution[] = (items || []).map((it) => ({
    url: it.url,
    source: it.source,
    note: (it as any).note,
    license: (it as any).license,
    logoUrl: (it as any).logoUrl,
  }));
  return fetchAttribution(raws, opts);
}
