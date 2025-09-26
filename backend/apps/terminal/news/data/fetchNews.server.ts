// app/market/fetchnews.server.ts
// No imports. Server-only util + Server Action to fetch/normalize news from JSON endpoints.
// If no endpoints are configured, returns a small mocked feed.
// Configure endpoints via env: NEWS_ENDPOINTS="https://api.example.com/news,https://api2.example.com/feed"
// Each endpoint should return an array of objects. Field names are inferred (title/headline, url/link, etc).

"use server";

/* ---------------- Types ---------------- */
export type Item = {
  id: string;
  title: string;
  summary?: string;
  url?: string;
  source: string;
  category: "Equities" | "FX" | "Fixed Income" | "Derivatives" | "Macro";
  symbols?: string[];
  sentiment?: -1 | 0 | 1;
  ts: number; // epoch ms
};

export type Params = {
  q?: string;
  category?: Item["category"] | "All";
  from?: number; // epoch ms
  to?: number;   // epoch ms
  symbols?: string[];
  limit?: number;
  sources?: string[]; // override endpoints
  timeoutMs?: number; // per-request timeout
  cacheTtlMs?: number;
};

/* ---------------- In-memory cache ---------------- */
declare global {
  // eslint-disable-next-line no-var
  var __NEWS_CACHE: Map<string, { ts: number; data: Item[] }> | undefined;
  // eslint-disable-next-line no-var
  var __NEWS_ENDPOINTS: string[] | undefined;
}
if (!globalThis.__NEWS_CACHE) globalThis.__NEWS_CACHE = new Map();
const CACHE = globalThis.__NEWS_CACHE!;

/* ---------------- Endpoint registry ---------------- */
function envSources(): string[] {
  if (!globalThis.__NEWS_ENDPOINTS) {
    const raw =
      (typeof process !== "undefined" &&
        (process as any).env &&
        (process as any).env.NEWS_ENDPOINTS) ||
      "";
    globalThis.__NEWS_ENDPOINTS = raw
      .split(",")
      .map((s: string) => s.trim())
      .filter(Boolean);
  }
  return globalThis.__NEWS_ENDPOINTS!;
}

/* ---------------- Public API ---------------- */
export async function fetchNews(params: Params = {}): Promise<Item[]> {
  const p = normalizeParams(params);
  const key = JSON.stringify(p);

  // cache
  const cached = CACHE.get(key);
  if (cached && Date.now() - cached.ts < p.cacheTtlMs) {
    return cached.data;
  }

  const urls = (p.sources && p.sources.length ? p.sources : envSources()).filter(Boolean);

  let rawItems: Item[] = [];
  if (urls.length > 0) {
    // fetch all endpoints with timeout & ignore failures
    const results = await Promise.all(
      urls.map((u) => fetchJsonSafe(u, p.timeoutMs).then((j) => normalizePayload(j, u)).catch(() => [] as Item[])),
    );
    rawItems = results.flat();
  } else {
    // fallback when nothing configured
    rawItems = mockFeed();
  }

  const filtered = applyFilters(rawItems, p);
  const deduped = dedupe(filtered);
  const sorted = deduped.sort((a, b) => b.ts - a.ts);
  const limited = p.limit ? sorted.slice(0, p.limit) : sorted;

  CACHE.set(key, { ts: Date.now(), data: limited });
  return limited;
}

/* Server Action variant that reads FormData (for <form action={...}>) */
export async function fetchNewsAction(fd: FormData): Promise<{ ok: true; data: Item[] } | { ok: false; error: string }> {
  "use server";
  try {
    const p: Params = {
      q: str(fd.get("q")),
      category: (str(fd.get("category")) as any) || "All",
      from: num(fd.get("from")),
      to: num(fd.get("to")),
      symbols: splitSyms(str(fd.get("symbols"))),
      limit: num(fd.get("limit")),
      timeoutMs: num(fd.get("timeoutMs")) ?? 4000,
      cacheTtlMs: num(fd.get("cacheTtlMs")) ?? 20000,
      sources: splitUrls(str(fd.get("sources"))) || undefined,
    };
    const data = await fetchNews(p);
    return { ok: true, data };
  } catch (e: any) {
    return { ok: false, error: e?.message || "fetch failed" };
  }
}

/* ---------------- Internals ---------------- */
function normalizeParams(p: Params): Required<Params> {
  return {
    q: (p.q || "").trim(),
    category: (p.category as any) || "All",
    from: Number.isFinite(p.from!) ? (p.from as number) : 0,
    to: Number.isFinite(p.to!) ? (p.to as number) : 0,
    symbols: (p.symbols || []).map((s) => s.trim().toUpperCase()).filter(Boolean),
    limit: p.limit && p.limit > 0 ? p.limit : 100,
    sources: (p.sources || []).filter(Boolean),
    timeoutMs: p.timeoutMs && p.timeoutMs > 0 ? p.timeoutMs : 4000,
    cacheTtlMs: p.cacheTtlMs && p.cacheTtlMs >= 0 ? p.cacheTtlMs : 20000,
  };
}

async function fetchJsonSafe(url: string, timeoutMs = 4000): Promise<any> {
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : (null as any);
  const id = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
  try {
    const res = await fetch(url, { signal: ctrl?.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ct = res.headers.get("content-type") || "";
    if (!/json/i.test(ct)) {
      // attempt to parse as text then JSON.parse if applicable
      const txt = await res.text();
      try {
        return JSON.parse(txt);
      } catch {
        // unsupported format
        return [];
      }
    }
    return await res.json();
  } finally {
    if (id) clearTimeout(id as any);
  }
}

function normalizePayload(payload: any, url: string): Item[] {
  // Expect: array of objects; try to infer common field names.
  const arr = Array.isArray(payload) ? payload : Array.isArray(payload?.items) ? payload.items : [];
  const host = safeHost(url) || "source";
  const out: Item[] = [];

  for (const r of arr) {
    const title = str(r.title) || str(r.headline) || str(r.name) || "";
    if (!title) continue;

    const link = str(r.url) || str(r.link) || undefined;
    const source =
      str(r.source) ||
      str(r.publisher) ||
      (link ? safeHost(link) : host) ||
      host;

    const cat =
      mapCategory(str(r.category)) ||
      mapCategory(str(r.section)) ||
      ("Macro" as Item["category"]);

    const summary = str(r.summary) || str(r.description) || undefined;

    const syms = (arrify(r.symbols || r.tickers) as string[])
      .map((s) => String(s).toUpperCase().trim())
      .filter(Boolean);

    const ts =
      num(r.ts) ||
      num(r.time) ||
      num(r.published) ||
      (str(r.published_at) ? Date.parse(String(r.published_at)) : 0) ||
      (str(r.pubDate) ? Date.parse(String(r.pubDate)) : 0) ||
      Date.now();

    const sentiment: -1 | 0 | 1 =
      num(r.sentiment) as any ??
      quickSentiment(title + " " + (summary || ""));

    out.push({
      id: str(r.id) || hashId(title + (link || "") + source + String(ts)),
      title,
      summary,
      url: link,
      source,
      category: cat,
      symbols: syms.length ? syms : undefined,
      sentiment,
      ts: ts > 0 ? ts : Date.now(),
    });
  }

  return out;
}

function applyFilters(items: Item[], p: Required<Params>): Item[] {
  const q = p.q.toLowerCase();
  return items.filter((i) => {
    if (p.category !== "All" && i.category !== p.category) return false;
    if (p.from && i.ts < p.from) return false;
    if (p.to && i.ts > p.to) return false;

    if (p.symbols.length) {
      const has = (i.symbols || []).some((s) => p.symbols.includes(s.toUpperCase()));
      if (!has) return false;
    }

    if (q) {
      const hay = `${i.title} ${i.summary || ""} ${i.source} ${(i.symbols || []).join(" ")}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }

    return true;
  });
}

function dedupe(items: Item[]): Item[] {
  const seen = new Set<string>();
  const out: Item[] = [];
  for (const it of items) {
    const k = (it.id || "") + "|" + (it.url || "") + "|" + it.title;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(it);
  }
  return out;
}

/* ---------------- Tiny helpers ---------------- */
function str(v: any): string | undefined {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}
function num(v: any): number | undefined {
  if (v == null || v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}
function arrify(v: any): any[] {
  return Array.isArray(v) ? v : v == null ? [] : [v];
}
function safeHost(url?: string) {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, "") : undefined;
  } catch {
    return undefined;
  }
}
function mapCategory(s?: string): Item["category"] | undefined {
  if (!s) return undefined;
  const t = s.toLowerCase();
  if (/equit|stock|share/.test(t)) return "Equities";
  if (/\bfx\b|forex|currency|usd|eur|inr|jpy|gbp|cny/.test(t)) return "FX";
  if (/fixed|bond|yield|gilt|10y|sovereign/.test(t)) return "Fixed Income";
  if (/deriv|option|futures|opec|brent|crude|hedge/.test(t)) return "Derivatives";
  return "Macro";
}
function quickSentiment(text: string): -1 | 0 | 1 {
  const s = text.toLowerCase();
  let score = 0;
  ["beats", "surge", "rally", "growth", "upgrade", "strong", "improve"].forEach((w) => (score += s.includes(w) ? 1 : 0));
  ["miss", "cuts", "drop", "fall", "downgrade", "weak", "risk"].forEach((w) => (score -= s.includes(w) ? 1 : 0));
  return score > 0 ? 1 : score < 0 ? -1 : 0;
}
function hashId(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return "n_" + (h >>> 0).toString(36);
}
function splitSyms(s?: string | null): string[] | undefined {
  if (!s) return undefined;
  return s.split(/[\s,|]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
}
function splitUrls(s?: string | null): string[] | undefined {
  if (!s) return undefined;
  return s.split(/[,\s]+/).map((t) => t.trim()).filter(Boolean);
}

/* ---------------- Fallback feed ---------------- */
function mockFeed(): Item[] {
  const now = Date.now();
  const make = (
    id: string,
    title: string,
    source: string,
    category: Item["category"],
    minsAgo: number,
    summary?: string,
    symbols?: string[],
    url?: string,
  ): Item => ({
    id, title, source, category, summary, symbols, url,
    sentiment: quickSentiment(title + " " + (summary || "")),
    ts: now - minsAgo * 60_000,
  });

  return [
    make("m1", "RBI seen holding rates; commentary hints at durable pause", "Economic Times", "Macro", 8, "Cautious tone on inflation."),
    make("m2", "Infosys rallies on strong deal wins; guidance intact", "Mint", "Equities", 16, "Margins guided to improve.", ["INFY"]),
    make("m3", "USD/JPY slips as UST yields ease; risk appetite improves", "FXStreet", "FX", 34, undefined, ["USD/JPY"]),
    make("m4", "India 10Y yields steady; auction demand robust", "Bloomberg", "Fixed Income", 52, "Supply well absorbed."),
    make("m5", "Brent spikes as OPEC+ chatter revives risk premium", "Reuters", "Derivatives", 77, "Tighter balances into Q4.", ["BRENT"]),
  ];
}
