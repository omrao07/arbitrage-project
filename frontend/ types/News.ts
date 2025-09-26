// frontend/types/News.ts
// Strong types for news ingestion, normalization, sentiment, and linking to strategies.

export type ISODateTime = string;

/** Supported providers/sources (extend as needed). */
export enum NewsSource {
  Bloomberg = "Bloomberg",
  Reuters = "Reuters",
  WSJ = "WSJ",
  FT = "FT",
  AP = "AP",
  Yahoo = "Yahoo",
  GDELT = "GDELT",
  RSS = "RSS",
  Twitter = "Twitter",
  Custom = "Custom",
}

/** Lightweight sentiment container (model-agnostic). */
export interface Sentiment {
  score: number;       // -1..+1
  confidence?: number; // 0..1
  label?: "neg" | "neu" | "pos";
  model?: string;      // e.g. "finbert-v1"
}

/** Tagging for quick filtering and strategy routing. */
export type TopicTag =
  | "CPI" | "PPI" | "Payrolls" | "UnemploymentRate" | "GDP"
  | "FOMC" | "ECB" | "BOJ" | "BOE"
  | "Inflation" | "Deflation" | "Growth" | "Recession"
  | "Energy" | "Oil" | "Gas" | "Utilities"
  | "Semis" | "AI" | "Cloud" | "Retail" | "Autos" | "Banks"
  | "FX" | "Rates" | "Credit" | "CDS" | "HY" | "IG"
  | "Geopolitics" | "War" | "Sanctions"
  | "Earnings" | "Guidance" | "M&A" | "Buyback" | "Dividends"
  | "SupplyChain" | "China" | "Europe" | "US" | "Japan" | "EM";

/** Normalized news item. */
export interface NewsItem {
  id: string;                 // provider id or hash
  asOf: ISODateTime;          // publication time (UTC)
  source: NewsSource | string;
  headline: string;
  url?: string;
  body?: string;              // optional short body/summary
  tickers?: string[];         // ["AAPL","NVDA","ES1","USDJPY"]
  countries?: string[];       // ISO codes
  sectors?: string[];         // GICS-ish
  topics?: TopicTag[];        // tags for routing
  sentiment?: Sentiment;
  importance?: number;        // 0..1 (derived from provider priority + engagement)
  // vendor-specific raw
  raw?: Record<string, unknown>;
}

/** Macro calendar event (CPI, payrolls, central bank, auctions, etc.). */
export interface MacroEvent {
  id: string;
  asOf: ISODateTime;            // event time
  name: string;                 // "US CPI", "FOMC Rate Decision"
  country?: string;             // "US","JP","EU"
  actual?: number | string;
  consensus?: number | string;
  previous?: number | string;
  unit?: string;                // "% y/y", "k", "bps"
  surprise?: number;            // standardized z-score if you compute it
  url?: string;
  topics?: TopicTag[];
  importance?: number;          // 0..1
  raw?: Record<string, unknown>;
}

/** Query contract for your news bridge API. */
export interface NewsQuery {
  q?: string;                     // full-text search
  tickers?: string[];
  topics?: TopicTag[];
  sources?: (NewsSource | string)[];
  from?: ISODateTime;
  to?: ISODateTime;
  limit?: number;
  offset?: number;
}

/** Response shapes from backend. */
export interface NewsResponse {
  items: NewsItem[];
  total?: number;
  nextOffset?: number;
  asOf: ISODateTime;
}

export interface MacroResponse {
  events: MacroEvent[];
  asOf: ISODateTime;
}

/** Link a strategy to a stream of news. */
export interface StrategyNewsLink {
  strategyId: string;
  watchTickers?: string[];
  watchTopics?: TopicTag[];
  minImportance?: number; // default 0.3
}

/* ------------------------------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------------------------------ */

/** Simple sentiment post-processor that fills label & clamps ranges. */
export function normalizeSentiment(s?: Sentiment): Sentiment | undefined {
  if (!s) return undefined;
  const score = Math.max(-1, Math.min(1, Number(s.score ?? 0)));
  const confidence = s.confidence != null ? Math.max(0, Math.min(1, Number(s.confidence))) : undefined;
  let label: Sentiment["label"];
  if (score > 0.15) label = "pos";
  else if (score < -0.15) label = "neg";
  else label = "neu";
  return { score, confidence, label, model: s.model };
}

/** Basic deduper by (source,id) or (headline+asOf). */
export function dedupeNews(items: NewsItem[]): NewsItem[] {
  const seen = new Set<string>();
  const out: NewsItem[] = [];
  for (const it of items) {
    const key = `${it.source}:${it.id || `${it.headline}|${it.asOf}`}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(it);
  }
  return out;
}

/** Filter by watchlist (tickers/topics) and minimum importance. */
export function filterForStrategy(items: NewsItem[], link: StrategyNewsLink): NewsItem[] {
  const minImp = link.minImportance ?? 0;
  const tickers = new Set((link.watchTickers ?? []).map(x => x.toUpperCase()));
  const topics = new Set(link.watchTopics ?? []);
  return items.filter(it => {
    const impOK = (it.importance ?? 0) >= minImp;
    const tickOK = !tickers.size || (it.tickers ?? []).some(t => tickers.has(t.toUpperCase()));
    const topOK = !topics.size || (it.topics ?? []).some(t => topics.has(t));
    return impOK && tickOK && topOK;
  });
}

/** Compact mapper from arbitrary vendor record into our NewsItem. */
export function toNewsItem(raw: any, source: NewsSource | string): NewsItem {
  const id = String(raw.id ?? raw.guid ?? raw.hash ?? `${raw.title ?? raw.headline}-${raw.publishedAt ?? raw.time}`);
  const headline = String(raw.headline ?? raw.title ?? "");
  const asOf = String(raw.asOf ?? raw.publishedAt ?? raw.pubDate ?? new Date().toISOString());
  const url = raw.url ?? raw.link;
  const body = raw.body ?? raw.summary ?? raw.description;
  const tickers: string[] = Array.from(new Set<string>((raw.tickers ?? raw.symbols ?? []).map((t: string) => t.toUpperCase())));
  const countries: string[] = raw.countries ?? [];
  const sectors: string[] = raw.sectors ?? [];
  const topics: TopicTag[] = (raw.topics ?? []) as TopicTag[];
  const importance = raw.importance ?? raw.priority ?? 0;
  const sentiment = normalizeSentiment(raw.sentiment);
  return { id, asOf, source, headline, url, body, tickers, countries, sectors, topics, importance, sentiment, raw };
}

/** Merge + sort news feed (descending time). */
export function mergeAndSortNews(...lists: NewsItem[][]): NewsItem[] {
  const merged = dedupeNews(lists.flat());
  return merged.sort((a, b) => (a.asOf < b.asOf ? 1 : -1));
}

/** Group news by ticker for quick display. */
export function groupByTicker(items: NewsItem[]): Record<string, NewsItem[]> {
  const out: Record<string, NewsItem[]> = {};
  for (const it of items) {
    const keys = (it.tickers && it.tickers.length ? it.tickers : ["__UNTAGGED__"]);
    for (const k of keys) {
      const u = k.toUpperCase();
      (out[u] ||= []).push(it);
    }
  }
  return out;
}

/** Compute a lightweight per-ticker sentiment snapshot. */
export function tickerSentiment(items: NewsItem[]): Record<string, { score: number; n: number }> {
  const acc: Record<string, { s: number; n: number }> = {};
  for (const it of items) {
    const sc = it.sentiment?.score ?? 0;
    for (const k of it.tickers ?? []) {
      const u = k.toUpperCase();
      (acc[u] ||= { s: 0, n: 0 });
      acc[u].s += sc;
      acc[u].n += 1;
    }
  }
  const out: Record<string, { score: number; n: number }> = {};
  for (const [k, v] of Object.entries(acc)) out[k] = { score: v.n ? v.s / v.n : 0, n: v.n };
  return out;
}

/* ------------------------------------------------------------------------------------------------
 * In-memory cache (optional for the UI)
 * ------------------------------------------------------------------------------------------------ */

export interface NewsCache {
  put(key: string, data: NewsResponse | MacroResponse, ttlMs?: number): void;
  get<T extends NewsResponse | MacroResponse>(key: string): T | undefined;
}

export function createMemoryNewsCache(): NewsCache {
  const store = new Map<string, { exp: number; data: any }>();
  return {
    put(key, data, ttlMs = 60_000) {
      store.set(key, { exp: Date.now() + ttlMs, data });
    },
    get(key) {
      const hit = store.get(key);
      if (!hit) return undefined;
      if (Date.now() > hit.exp) { store.delete(key); return undefined; }
      return hit.data;
    }
  };
}

/* ------------------------------------------------------------------------------------------------
 * Example fetcher (you can replace with your data/news_bridge.py endpoint)
 * ------------------------------------------------------------------------------------------------ */

export async function fetchNews(apiBase: string, q: NewsQuery, cache?: NewsCache): Promise<NewsResponse> {
  const key = `news:${JSON.stringify(q)}`;
  const cached = cache?.get<NewsResponse>(key);
  if (cached) return cached;

  const params = new URLSearchParams();
  if (q.q) params.set("q", q.q);
  if (q.tickers?.length) params.set("tickers", q.tickers.join(","));
  if (q.topics?.length) params.set("topics", q.topics.join(","));
  if (q.sources?.length) params.set("sources", q.sources.join(","));
  if (q.from) params.set("from", q.from);
  if (q.to) params.set("to", q.to);
  if (q.limit) params.set("limit", String(q.limit));
  if (q.offset) params.set("offset", String(q.offset));

  const res = await fetch(`${apiBase.replace(/\/$/, "")}/news?${params.toString()}`);
  if (!res.ok) throw new Error(`news fetch failed: ${res.status}`);
  const json = await res.json();
  // assume backend returns items already normalized; if not, map with toNewsItem()
  const out: NewsResponse = {
    items: (json.items ?? []).map((x: any) => toNewsItem(x, x.source ?? NewsSource.Custom)),
    total: json.total,
    nextOffset: json.nextOffset,
    asOf: json.asOf ?? new Date().toISOString(),
  };
  cache?.put(key, out, 45_000);
  return out;
}