// core/bloomberg.ts
// Pure TypeScript (zero imports). Bloomberg-normalization scaffolding WITHOUT any external API calls.
// Plug a real HTTP transport later. This file gives you:
//  - Types for requests/responses
//  - Rate limiter + backoff
//  - Entitlements gate
//  - Field mapping (provider <-> catalog) + FX/unit/tz normalization
//  - Pagination harness
//  - Historical snapshot normalization helpers
//  - Deterministic simulators for local dev (so pages render before adapters)

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type Dict<T = any> = { [k: string]: T };

export type HttpRequest = {
  method: "GET" | "POST";
  url: string;
  headers?: Dict<string>;
  body?: string;
  timeoutMs?: number;
};

export type HttpResponse = {
  status: number;
  headers: Dict<string>;
  body: string; // provider JSON/text
};

export type Transport = (req: HttpRequest) => Promise<HttpResponse>;

export type BBGCredential = {
  // Placeholder for whatever auth schema you use (session, certs, etc.)
  token?: string;      // e.g., "u:<user>|r:<role>|exp:<iso>|sig:<hex>"
  app?: string;        // app name
  user?: string;
  entitlements?: string[]; // permitted field groups
};

export type Pagination = {
  // Generic pagination surface
  next?: string;           // provider continuation token
  limit?: number;
};

export type RateRule = {
  // Simple token bucket limits
  capacity: number;  // max tokens
  refillPerSec: number; // tokens per second
};

export type Backoff = {
  // Exponential backoff config
  minMs: number;
  maxMs: number;
  factor: number;
};

export type BBGReq = {
  dataset: "px-hist" | "px-ref" | "fundamentals" | "news";
  symbols: string[];
  fields: string[];
  start?: string; // ISO
  end?: string;   // ISO
  overrides?: Dict;
  pagination?: Pagination;
};

export type BBGRow = Dict; // raw provider row (flattened)

// Normalized catalog row (subset used widely)
export type CatalogRow = {
  asOf?: string;     // ISO
  ticker?: string;
  currency?: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  // Optional common bits:
  name?: string;
  mic?: string;
  figi?: string;
  isin?: string;
  bbgid?: string;
  // Extra numeric/string fields allowed:
  [k: string]: any;
};

export type NormalizeCtx = {
  baseCcy?: string;            // normalize monetary to this (default USD)
  fx?: Record<string, number>; // e.g., { EURUSD: 1.075 }
  tz?: string;                 // "UTC" expected downstream; we snap to ISO
};

// ──────────────────────────────────────────────────────────────────────────────
// Rate limiter & backoff
// ──────────────────────────────────────────────────────────────────────────────

export class TokenBucket {
  private tokens: number;
  private lastRefill: number;
  constructor(private rule: RateRule) {
    this.tokens = rule.capacity;
    this.lastRefill = Date.now();
  }
  async take(n = 1) {
    for (;;) {
      this.refill();
      if (this.tokens >= n) {
        this.tokens -= n;
        return;
      }
      await sleep(50);
    }
  }
  private refill() {
    const now = Date.now();
    const delta = (now - this.lastRefill) / 1000;
    if (delta <= 0) return;
    this.tokens = Math.min(this.rule.capacity, this.tokens + delta * this.rule.refillPerSec);
    this.lastRefill = now;
  }
}

export async function backoffRetry<T>(
  fn: () => Promise<T>,
  cfg: Backoff = { minMs: 200, maxMs: 8_000, factor: 2 },
  attempts = 5
): Promise<T> {
  let wait = cfg.minMs;
  for (let i = 0; i < attempts; i++) {
    try { return await fn(); } catch (e) {
      if (i === attempts - 1) throw e;
      await sleep(wait);
      wait = Math.min(cfg.maxMs, Math.max(cfg.minMs, Math.floor(wait * cfg.factor)));
    }
  }
  // unreachable
  // @ts-ignore
  return undefined;
}

// ──────────────────────────────────────────────────────────────────────────────
// Entitlements (placeholder)
// ──────────────────────────────────────────────────────────────────────────────

export function checkEntitlements(creds: BBGCredential, fields: string[]): { ok: boolean; missing?: string[] } {
  // Extremely simple: every field must start with a group you own; e.g., "PX_*" => "px"
  // Customize to your contract. This is just a guardrail.
  const owned = new Set((creds.entitlements || []).map(s => s.toLowerCase()));
  const miss: string[] = [];
  for (const f of fields) {
    const g = (f.split("_")[0] || "").toLowerCase(); // "PX_LAST" -> "px"
    if (!owned.has(g) && owned.size) miss.push(f);
  }
  return miss.length ? { ok: false, missing: miss } : { ok: true };
}

// ──────────────────────────────────────────────────────────────────────────────
// Field mapping & normalization
// ──────────────────────────────────────────────────────────────────────────────

// Common Bloomberg field aliases
export const BBG = {
  PX_OPEN: "PX_OPEN",
  PX_HIGH: "PX_HIGH",
  PX_LOW: "PX_LOW",
  PX_LAST: "PX_LAST",
  PX_VOLUME: "PX_VOLUME",
  CRNCY: "CRNCY",
  TICKER: "TICKER",
  NAME: "NAME",
  ID_MIC_PRIM_EXCH: "ID_MIC_PRIM_EXCH",
  ID_ISIN: "ID_ISIN",
  ID_BB_UNIQUE: "ID_BB_UNIQUE",
  ID_BB_GLOBAL: "ID_BB_GLOBAL", // FIGI
  DS002: "DS002", // example fundamentals dataset id
} as const;

// Default catalog -> provider mapping
export const CatalogToBBG: Record<string, string> = {
  open: BBG.PX_OPEN,
  high: BBG.PX_HIGH,
  low: BBG.PX_LOW,
  close: BBG.PX_LAST,
  volume: BBG.PX_VOLUME,
  currency: BBG.CRNCY,
  ticker: BBG.TICKER,
  name: BBG.NAME,
  mic: BBG.ID_MIC_PRIM_EXCH,
  isin: BBG.ID_ISIN,
  bbgid: BBG.ID_BB_UNIQUE,
  figi: BBG.ID_BB_GLOBAL,
};

export type FieldSpec = {
  name: string;                            // catalog field name
  unit?: "USD"|"EUR"|"JPY"|"pct"|"bps"|"shares"|"number";
  type?: "number"|"string"|"date";
};

// Minimal catalog specs for common columns (extend as needed)
export const DefaultSpecs: Record<string, FieldSpec> = {
  open:   { name: "open",   unit: "USD", type: "number" },
  high:   { name: "high",   unit: "USD", type: "number" },
  low:    { name: "low",    unit: "USD", type: "number" },
  close:  { name: "close",  unit: "USD", type: "number" },
  volume: { name: "volume", unit: "shares", type: "number" },
  currency: { name: "currency", type: "string" },
  ticker:   { name: "ticker",   type: "string" },
  name:     { name: "name",     type: "string" },
  mic:      { name: "mic",      type: "string" },
  figi:     { name: "figi",     type: "string" },
  isin:     { name: "isin",     type: "string" },
  bbgid:    { name: "bbgid",    type: "string" },
};

export function normalizeRow(
  src: BBGRow,
  mapping: Record<string,string> = CatalogToBBG,
  specs: Record<string, FieldSpec> = DefaultSpecs,
  ctx: NormalizeCtx = { baseCcy: "USD", fx: {}, tz: "UTC" }
): CatalogRow {
  const out: CatalogRow = {};
  for (const catName in mapping) {
    const providerKey = mapping[catName];
    const spec = specs[catName] || { name: catName };
    const raw = src[providerKey];
    (out as any)[catName] = normVal(raw, spec, src, ctx);
  }
  // stamp asOf if present in raw
  const asOf = src["asOf"] || src["EFFECTIVE_DATE"] || src["TRADE_DATE"] || src["date"];
  if (asOf) out.asOf = toISO(asOf);
  return out;
}

function normVal(v: any, spec: FieldSpec, row: Dict, ctx: NormalizeCtx): any {
  if (v == null) return v;
  if (spec.unit) {
    if (isMoney(spec.unit)) {
      const srcCcy = row[BBG.CRNCY] || row["currency"] || ctx.baseCcy || "USD";
      return toBaseCcy(Number(v), String(srcCcy), ctx.baseCcy || "USD", ctx.fx || {});
    }
    if (spec.unit === "pct") return Number(v);           // 0.0123 = 1.23%
    if (spec.unit === "bps") return Number(v) / 10_000;  // -> fraction
  }
  if (spec.type === "number") return Number(v);
  if (spec.type === "date") return toISO(v);
  return String(v);
}

function isMoney(u: string) { return u === "USD" || u === "EUR" || u === "JPY"; }
function toBaseCcy(amount: number, from: string, base: string, fx: Record<string, number>): number {
  if (from === base) return amount;
  const p1 = (from + base).toUpperCase();
  const p2 = (base + from).toUpperCase();
  if (fx[p1]) return amount * fx[p1];
  if (fx[p2]) return amount / fx[p2];
  return amount; // fallback
}
function toISO(d: any): string {
  try { return new Date(d).toISOString(); } catch { return String(d); }
}

// ──────────────────────────────────────────────────────────────────────────────
/** Pagination harness: you pass a page fetcher; we loop w/ next token until done. */
export async function paginate<T>(
  fetchPage: (next?: string) => Promise<{ rows: T[]; next?: string }>,
  cap = 1_000
): Promise<T[]> {
  const out: T[] = [];
  let next: string | undefined = undefined;
  for (;;) {
    const { rows, next: n } = await fetchPage(next);
    for (const r of rows) {
      out.push(r);
      if (out.length >= cap) return out;
    }
    if (!n) break;
    next = n;
  }
  return out;
}

// ──────────────────────────────────────────────────────────────────────────────
// Adapter shell (no actual HTTP). Wire a real transport later.
// ──────────────────────────────────────────────────────────────────────────────

export class BloombergAdapter {
  private limiter: TokenBucket;
  private backoff: Backoff;
  private transport?: Transport;

  constructor(
    private creds: BBGCredential,
    opts?: { rate?: RateRule; backoff?: Backoff; transport?: Transport }
  ) {
    this.limiter = new TokenBucket(opts?.rate || { capacity: 5, refillPerSec: 5 });
    this.backoff = opts?.backoff || { minMs: 200, maxMs: 5000, factor: 2 };
    this.transport = opts?.transport; // can be injected later
  }

  setTransport(t: Transport) { this.transport = t; }

  /** Build a request payload (no send). Customize this to your gateway. */
  buildRequest(req: BBGReq): HttpRequest {
    // This is a placeholder that you’ll adapt to your Bloomberg access layer.
    // Example: POST /bbg/query with symbols/fields and date range in body.
    const url = "https://YOUR-BLOOMBERG-GATEWAY/placeholder";
    const body = JSON.stringify({
      dataset: req.dataset,
      symbols: req.symbols,
      fields: req.fields,
      start: req.start,
      end: req.end,
      overrides: req.overrides,
      page: req.pagination?.next,
      limit: req.pagination?.limit || 500
    });
    const headers: Dict<string> = {
      "content-type": "application/json",
      "x-app": this.creds.app || "app",
      "x-user": this.creds.user || "user",
      "authorization": this.creds.token ? `Bearer ${this.creds.token}` : ""
    };
    return { method: "POST", url, headers, body, timeoutMs: 30_000 };
  }

  /** Execute with limiter + backoff. Requires `transport` to be set. */
  private async exec<T = any>(req: HttpRequest): Promise<T> {
    if (!this.transport) {
      // For local dev: return deterministic simulator so UI works.
      return this.simulate<T>(req);
    }
    await this.limiter.take(1);
    const res = await backoffRetry(async () => this.transport!(req), this.backoff, 5);
    if (res.status >= 400) throw new Error(`BBG_HTTP_${res.status}`);
    return safeJSON<T>(res.body);
  }

  /** High-level: historical price fetch → normalized rows */
  async historical(req: Omit<BBGReq, "dataset"> & { dataset?: "px-hist" }, ctx?: NormalizeCtx): Promise<CatalogRow[]> {
    const fields = req.fields && req.fields.length ? req.fields : [
      BBG.PX_OPEN, BBG.PX_HIGH, BBG.PX_LOW, BBG.PX_LAST, BBG.PX_VOLUME, BBG.CRNCY, BBG.TICKER
    ];
    const ent = checkEntitlements(this.creds, fields);
    if (!ent.ok) throw new Error("entitlements_missing:" + (ent.missing || []).join(","));

    const built = this.buildRequest({ ...req, dataset: "px-hist" });
    const data = await this.exec<{ rows: BBGRow[]; next?: string }>(built);
    const out: CatalogRow[] = [];
    for (const r of (data.rows || [])) out.push(normalizeRow(r, CatalogToBBG, DefaultSpecs, ctx));
    // handle pagination automatically if gateway returns next
    let next = data.next;
    let guard = 6; // safety cap
    while (next && guard-- > 0) {
      const more = await this.exec<{ rows: BBGRow[]; next?: string }>({
        ...built, body: patchJsonBody(built.body || "{}", { page: next })
      });
      for (const r of (more.rows || [])) out.push(normalizeRow(r, CatalogToBBG, DefaultSpecs, ctx));
      next = more.next;
    }
    return out;
  }

  /** High-level: reference data (identifiers, currency, name, etc.) */
  async reference(req: Omit<BBGReq, "dataset"> & { dataset?: "px-ref" }): Promise<CatalogRow[]> {
    const fields = req.fields && req.fields.length ? req.fields : [
      BBG.TICKER, BBG.CRNCY, BBG.NAME, BBG.ID_MIC_PRIM_EXCH, BBG.ID_ISIN, BBG.ID_BB_GLOBAL, BBG.ID_BB_UNIQUE
    ];
    const ent = checkEntitlements(this.creds, fields);
    if (!ent.ok) throw new Error("entitlements_missing:" + (ent.missing || []).join(","));

    const built = this.buildRequest({ ...req, dataset: "px-ref" });
    const data = await this.exec<{ rows: BBGRow[]; next?: string }>(built);
    const out: CatalogRow[] = [];
    for (const r of (data.rows || [])) out.push(normalizeRow(r, CatalogToBBG, DefaultSpecs, { baseCcy: "USD", fx: {}, tz: "UTC" }));
    return out;
  }

  /** Deterministic simulation so the UI renders without network. */
  private async simulate<T = any>(req: HttpRequest): Promise<T> {
    const body = safeJSON<any>(req.body || "{}");
    const { dataset, symbols, start, end } = body || {};
    const rows: any[] = [];

    const startTs = start ? Date.parse(start) : Date.now() - 10 * 24 * 3600 * 1000;
    const endTs = end ? Date.parse(end) : Date.now();
    const days = Math.max(2, Math.min(120, Math.floor((endTs - startTs) / (24 * 3600 * 1000))));
    for (const sym of (symbols || ["SPY"])) {
      if (dataset === "px-ref") {
        rows.push({
          [BBG.TICKER]: sym,
          [BBG.CRNCY]: "USD",
          [BBG.NAME]: `Sim ${sym}`,
          [BBG.ID_MIC_PRIM_EXCH]: "XNAS",
          [BBG.ID_ISIN]: "US" + hash(sym).slice(0,10),
          [BBG.ID_BB_GLOBAL]: "BBG" + hash(sym).slice(0,9),
          [BBG.ID_BB_UNIQUE]: "BBU" + hash(sym).slice(0,9),
          asOf: new Date().toISOString(),
        });
      } else {
        // generate OHLC
        let p = 100 + (hashNum(sym) % 50);
        for (let i = 0; i < days; i++) {
          p *= 0.99 + ((i * 9301 + 49297) % 1000) / 1000 * 0.002; // pseudo path
          const o = p * (0.995 + (i%7)*0.0005);
          const h = o * (1.01 + (i%5)*0.0008);
          const l = o * (0.99 - (i%3)*0.0007);
          const c = (o + h + l) / 3;
          const v = 1_000_000 + (hashNum(sym + i) % 200_000);
          rows.push({
            [BBG.TICKER]: sym,
            [BBG.CRNCY]: "USD",
            [BBG.PX_OPEN]: round(o),
            [BBG.PX_HIGH]: round(h),
            [BBG.PX_LOW]: round(l),
            [BBG.PX_LAST]: round(c),
            [BBG.PX_VOLUME]: v,
            asOf: new Date(startTs + i * 24 * 3600 * 1000).toISOString(),
          });
        }
      }
    }
    const resp = { status: 200, headers: {}, body: JSON.stringify({ rows }) };
    // @ts-ignore
    return respTo<T>(resp);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

function safeJSON<T=any>(s: string): T {
  try { return JSON.parse(s) as T; } catch { return {} as T; }
}

function patchJsonBody(s: string, patch: Dict): string {
  const o = safeJSON<Dict>(s || "{}");
  for (const k in patch) o[k] = patch[k];
  return JSON.stringify(o);
}

function respTo<T>(r: HttpResponse): T {
  try { return JSON.parse(r.body) as T; } catch { return ({} as any as T); }
}

function round(x: number) { return Math.round(x * 100) / 100; }

function hash(s: string): string {
  let x = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    x ^= s.charCodeAt(i);
    x = Math.imul(x, 16777619) >>> 0;
  }
  return ("00000000" + x.toString(16)).slice(-8) + ("00000000" + ((x ^ 0xa5a5a5a5) >>> 0).toString(16)).slice(-8);
}
function hashNum(s: string): number {
  const h = hash(s);
  return parseInt(h.slice(0,8), 16) >>> 0;
}