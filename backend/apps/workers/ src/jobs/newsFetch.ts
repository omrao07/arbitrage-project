// jobs/newsfetch.ts
// Pure TypeScript. No imports. Minimal logger + ULID + paged news fetcher.

/* =============================== Minimal Logger =============================== */

type LogLevel = "debug" | "info" | "warn" | "error";

function iso(ts = Date.now()) { return new Date(ts).toISOString(); }

function createLogger(name = "news") {
  function line(lvl: LogLevel, msg: string, data?: Record<string, unknown>) {
    const payload = data && Object.keys(data).length ? " " + JSON.stringify(data) : "";
    const txt = `[${iso()}] ${lvl.toUpperCase()} ${name} ${msg}${payload}`;
    if (lvl === "error") console.error(txt);
    else if (lvl === "warn") console.warn(txt);
    else console.log(txt);
  }
  function time(label: string, base?: Record<string, unknown>, level: LogLevel = "info") {
    const start = Date.now();
    return (extra?: Record<string, unknown>) => {
      const ms = Date.now() - start;
      line(level, label, { ...(base || {}), ...(extra || {}), ms });
    };
  }
  return {
    debug: (m: string, d?: Record<string, unknown>) => line("debug", m, d),
    info:  (m: string, d?: Record<string, unknown>) => line("info", m, d),
    warn:  (m: string, d?: Record<string, unknown>) => line("warn", m, d),
    error: (m: string, d?: Record<string, unknown>) => line("error", m, d),
    time,
  };
}
const logger = createLogger("news");

/* =============================== Minimal ULID =============================== */

type Bytes = Uint8Array;
const B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

function rngBytes(n: number): Bytes {
  const a = new Uint8Array(n);
  const g: any = (globalThis as any);
  const c = g?.crypto || g?.msCrypto;
  if (c && typeof c.getRandomValues === "function") {
    c.getRandomValues(a); return a;
  }
  for (let i = 0; i < n; i++) a[i] = Math.floor(Math.random() * 256);
  return a;
}
function ulidEncodeTime(ms: number): string {
  let t = Math.max(0, Math.floor(ms));
  const out = new Array(10);
  for (let i = 9; i >= 0; i--) { out[i] = B32[t % 32]; t = Math.floor(t / 32); }
  return out.join("");
}
function ulidEncodeRand(bytes: Bytes): string {
  // 10 bytes -> 16 chars (5-bit groups)
  let s = "", acc = 0, bits = 0;
  for (let i = 0; i < bytes.length; i++) {
    acc = (acc << 8) | bytes[i]; bits += 8;
    while (bits >= 5) { bits -= 5; s += B32[(acc >>> bits) & 31]; }
  }
  if (bits > 0) s += B32[(acc << (5 - bits)) & 31];
  return s.slice(0, 16);
}
let __ulid_t = 0;
let __ulid_r = rngBytes(10);
function ulidMonotonic(dateMs?: number): string {
  const t = dateMs ?? Date.now();
  let r = rngBytes(10);
  if (t === __ulid_t) {
    r = __ulid_r.slice();
    for (let i = r.length - 1; i >= 0; i--) { r[i] = (r[i] + 1) & 0xff; if (r[i] !== 0) break; }
  }
  __ulid_t = t; __ulid_r = r;
  return ulidEncodeTime(t) + ulidEncodeRand(r);
}

/* =================================== Types =================================== */

export type NewsItem = {
  id: string;
  title: string;
  url?: string;
  source?: string;
  publishedAt: number;        // epoch ms
  topics?: string[];
  symbols?: string[];
  score?: number;             // optional relevance/priority
  summary?: string;
  raw?: any;                  // original vendor payload for audit
};

export type FetchPageArgs = { sinceMs?: number; cursor?: string; signal: AbortSignal };
export type FetchPageResult = { items: any[]; nextCursor?: string };

export type Options = {
  /** Pull one page from your provider. Use `cursor` for pagination or `sinceMs` for recency. */
  fetchPage: (args: FetchPageArgs) => Promise<FetchPageResult>;

  /** Convert raw vendor item -> normalized NewsItem (id optional; fallback generated). */
  normalize: (raw: any) => Partial<NewsItem> & { title?: string; publishedAt?: number };

  /** Upsert/save normalized item to your store (DB/cache/search). */
  persist?: (item: NewsItem) => Promise<void>;

  /** Optional mutate step before saving (e.g., enrich with tags). */
  beforeSave?: (item: NewsItem) => Promise<NewsItem | void> | NewsItem | void;

  /** Drop items that don’t pass. */
  filter?: (item: NewsItem) => boolean;

  /** Minimum score to keep (if item.score provided). */
  minScore?: number;

  /** Upper bounds to limit each run. */
  maxPages?: number;        // default 3
  maxItems?: number;        // default 200

  /** Do not reprocess items newer than this window multiple times (based on id/url). */
  dedupeWindowMs?: number;  // default 2 days

  /** Optional recency cut (ignore older than sinceMs). If omitted, auto = now - 24h. */
  sinceMs?: number;
};

/* =============================== In-memory state =============================== */

const seenKeyToTs = new Map<string, number>(); // key → publishedAt
function sweepSeen(now: number, windowMs: number) {
  const cutoff = now - Math.max(1, windowMs);
 
}

/* ================================= Utilities ================================= */

function trim(s?: string): string | undefined {
  if (s == null) return undefined;
  const t = String(s).trim();
  return t ? t : undefined;
}
function canonicalKey(n: Partial<NewsItem>): string {
  const url = trim(n.url);
  const id = trim(n.id);
  return (id || url || ulidMonotonic())!;
}
function sanitizeTitle(t: string): string {
  return t.replace(/\s+/g, " ").replace(/\u0000/g, "").trim();
}
function stripHtml(s?: string): string | undefined {
  if (!s) return s;
  return s.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

/* ================================= Defaults ================================= */

const NoopOptions: Options = {
  async fetchPage() { return { items: [], nextCursor: undefined }; },
  normalize(raw: any) {
    const title = sanitizeTitle(String(raw?.title || raw?.headline || "Untitled"));
    const publishedAt = Number(raw?.publishedAt || raw?.time || Date.now());
    const url = trim(raw?.url || raw?.link);
    return { title, publishedAt, url, source: raw?.source, raw };
  },
  async persist(_item: NewsItem) { /* no-op */ },
  beforeSave: undefined,
  filter: undefined,
  minScore: undefined,
  maxPages: 3,
  maxItems: 200,
  dedupeWindowMs: 2 * 24 * 60 * 60 * 1000, // 2 days
  sinceMs: undefined,
};

/* ================================== Runner ================================== */

export async function runNewsFetch(
  signal: AbortSignal,
  options?: Partial<Options>
): Promise<{ fetched: number; saved: number; skipped: number; latestMs: number | null }> {
  const opts: Options = { ...NoopOptions, ...(options as any) };
  const started = Date.now();
  const maxPages = Math.max(1, opts.maxPages ?? NoopOptions.maxPages!);
  const maxItems = Math.max(1, opts.maxItems ?? NoopOptions.maxItems!);
  const dedupeWindowMs = Math.max(1, opts.dedupeWindowMs ?? NoopOptions.dedupeWindowMs!);
  const sinceCut = typeof opts.sinceMs === "number" ? opts.sinceMs : started - 24 * 60 * 60 * 1000;

  sweepSeen(started, dedupeWindowMs);

  let cursor: string | undefined = undefined;
  let fetched = 0, saved = 0, skipped = 0;
  let latestMs: number | null = null;

  const done = logger.time("news-fetch", { maxPages, maxItems, sinceMs: sinceCut });

  for (let page = 0; page < maxPages; page++) {
    if (signal.aborted) break;

    let pageRes: FetchPageResult;
    try {
      pageRes = await opts.fetchPage({ sinceMs: sinceCut, cursor, signal });
    } catch (e: any) {
      logger.error("fetchPage failed", { err: e?.message || String(e), page });
      break;
    }

    const raws = Array.isArray(pageRes.items) ? pageRes.items : [];
    if (raws.length === 0) { cursor = undefined; break; }

    for (let i = 0; i < raws.length; i++) {
      if (signal.aborted) break;
      if (saved >= maxItems) break;

      const raw = raws[i];
      const norm = opts.normalize(raw) || {};
      if (!norm.title) { skipped++; continue; }

      const item: NewsItem = {
        id: canonicalKey(norm),
        title: sanitizeTitle(norm.title),
        url: trim(norm.url),
        source: trim(norm.source),
        publishedAt: Number(norm.publishedAt || Date.now()),
        topics: (norm.topics || []) as string[],
        symbols: (norm.symbols || []) as string[],
        score: typeof norm.score === "number" ? norm.score : undefined,
        summary: stripHtml(norm.summary),
        raw: norm.raw != null ? norm.raw : raw,
      };

      if (item.publishedAt && (latestMs == null || item.publishedAt > latestMs)) latestMs = item.publishedAt;

      // recency gate
      if (item.publishedAt < sinceCut) { skipped++; continue; }

      // score gate
      if (opts.minScore != null && typeof item.score === "number" && item.score < opts.minScore) { skipped++; continue; }

      // dedupe (id/url within window)
      const key = item.id || item.url || "";
      const seenTs = seenKeyToTs.get(key);
      if (seenTs && started - seenTs < dedupeWindowMs) { skipped++; continue; }

      // user filter
      if (opts.filter && !opts.filter(item)) { skipped++; continue; }

      // beforeSave (mutate)
      let toSave = item;
      try {
        const maybe = opts.beforeSave && (await opts.beforeSave(item));
        if (maybe && typeof maybe === "object") toSave = maybe as NewsItem;
      } catch (e: any) {
        logger.warn("beforeSave failed", { err: e?.message || String(e) });
      }

      // persist
      try {
        if (opts.persist) await opts.persist(toSave);
        seenKeyToTs.set(key, toSave.publishedAt || started);
        saved++;
      } catch (e: any) {
        logger.error("persist failed", { err: e?.message || String(e) });
        skipped++;
      }
    }

    fetched += raws.length;
    cursor = pageRes.nextCursor;
    if (!cursor) break;
  }

  done({ fetched, saved, skipped, latestMs });

  return { fetched, saved, skipped, latestMs };
}

/* ============================== Example (optional) ============================== */
/*
  // Example wiring with a generic JSON API that supports ?since and ?cursor:
  await runNewsFetch(signal, {
    async fetchPage({ sinceMs, cursor, signal }) {
      const base = "https://api.example.com/news";
      const url = new URL(base);
      if (sinceMs) url.searchParams.set("since", String(Math.floor(sinceMs / 1000)));
      if (cursor) url.searchParams.set("cursor", cursor);
      const res = await fetch(url.toString(), { signal as any });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const json = await res.json();
      return { items: json.items || [], nextCursor: json.nextCursor };
    },
    normalize(raw) {
      return {
        id: trim(raw.id),
        title: String(raw.title || raw.headline || ""),
        url: trim(raw.url),
        source: trim(raw.source || raw.publisher),
        publishedAt: Number(raw.publishedAt || raw.time || Date.parse(raw.date) || Date.now()),
        symbols: Array.isArray(raw.tickers) ? raw.tickers : undefined,
        topics: Array.isArray(raw.topics) ? raw.topics : undefined,
        score: typeof raw.score === "number" ? raw.score : undefined,
        summary: raw.summary,
        raw,
      };
    },
    async persist(item) {
      // save to your DB/cache
      console.log("SAVE", item.id, item.title);
    },
    minScore: 0.2,
    maxPages: 5,
    maxItems: 300,
    dedupeWindowMs: 24 * 60 * 60 * 1000,
    sinceMs: Date.now() - 12 * 60 * 60 * 1000, // last 12h
  });
*/