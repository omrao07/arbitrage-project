// core/hammer.ts
// Hammer Pro adapter scaffold (pure TypeScript, ZERO imports).
// What you get here (no external API calls included):
//  • Types for transport/requests/responses
//  • Token-bucket rate limiter + exponential backoff
//  • Simple entitlements/scopes gate
//  • Provider→catalog field mapping + FX/unit/tz normalization
//  • Pagination harness
//  • Historical + reference fetchers that normalize to catalog rows
//  • Deterministic simulator so UI works before wiring real network

/* ─────────────────────────── Types ─────────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type HttpRequest = {
  method: "GET" | "POST";
  url: string;
  headers?: Dict<string>;
  body?: string;
  timeoutMs?: number;
};
export type HttpResponse = { status: number; headers: Dict<string>; body: string };
export type Transport = (req: HttpRequest) => Promise<HttpResponse>;

export type HammerCredential = {
  token?: string;         // e.g., "u:you|r:analyst|exp:...|sig:..."
  app?: string;
  user?: string;
  scopes?: string[];      // e.g., ["prices.read","ref.read","fund.read","news.read"]
};

export type RateRule = { capacity: number; refillPerSec: number };
export type Backoff  = { minMs: number; maxMs: number; factor: number };

export type HPReq = {
  dataset: "prices" | "reference" | "fundamentals" | "news";
  symbols: string[];
  fields?: string[];
  start?: string; // ISO
  end?: string;   // ISO
  overrides?: Dict;
  pagination?: { next?: string; limit?: number };
};

export type HPRawRow = Dict;

export type CatalogRow = {
  asOf?: string;
  ticker?: string;
  currency?: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  name?: string;
  mic?: string;
  figi?: string;
  isin?: string;
  bbgid?: string;
  [k: string]: any;
};

export type NormalizeCtx = {
  baseCcy?: string;            // normalize monetary to this (default USD)
  fx?: Record<string, number>; // e.g., { EURUSD: 1.075 }
  tz?: string;                 // "UTC" target; we output ISO strings
};

/* ───────────────────── Token bucket & backoff ───────────────────── */

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
      if (this.tokens >= n) { this.tokens -= n; return; }
      await sleep(50);
    }
  }
  private refill() {
    const now = Date.now();
    const d = (now - this.lastRefill) / 1000;
    if (d <= 0) return;
    this.tokens = Math.min(this.rule.capacity, this.tokens + d * this.rule.refillPerSec);
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
  // @ts-ignore
  return undefined;
}

/* ───────────────────────── Entitlements ───────────────────────── */

export function checkScopes(creds: HammerCredential, need: string[]): { ok: boolean; missing?: string[] } {
  const have = new Set((creds.scopes || []).map(s => s.toLowerCase()));
  const miss: string[] = [];
  for (const s of need) if (!have.has(s.toLowerCase())) miss.push(s);
  return miss.length ? { ok: false, missing: miss } : { ok: true };
}

/* ───────────────── Provider field aliases & mapping ───────────────── */

export const HP = {
  OPEN:   "hp_open",
  HIGH:   "hp_high",
  LOW:    "hp_low",
  CLOSE:  "hp_close",
  VOLUME: "hp_volume",
  CCY:    "ccy",
  SYMBOL: "symbol",
  NAME:   "name",
  MIC:    "mic",
  ISIN:   "isin",
  FIGI:   "figi",
  UID:    "uid",       // hammer internal id
  DATE:   "date",      // ISO or epoch ms
} as const;

export type FieldSpec = {
  name: string;
  unit?: "USD"|"EUR"|"JPY"|"pct"|"bps"|"shares"|"number";
  type?: "number"|"string"|"date";
};

export const DefaultSpecs: Record<string, FieldSpec> = {
  open:   { name: "open",   unit: "USD",    type: "number" },
  high:   { name: "high",   unit: "USD",    type: "number" },
  low:    { name: "low",    unit: "USD",    type: "number" },
  close:  { name: "close",  unit: "USD",    type: "number" },
  volume: { name: "volume", unit: "shares", type: "number" },
  currency:{ name: "currency", type:"string" },
  ticker:  { name: "ticker",  type:"string" },
  name:    { name: "name",    type:"string" },
  mic:     { name: "mic",     type:"string" },
  isin:    { name: "isin",    type:"string" },
  figi:    { name: "figi",    type:"string" },
  bbgid:   { name: "bbgid",   type:"string" },
  asOf:    { name: "asOf",    type:"date"   },
};

export const CatalogToHP: Record<string, string> = {
  open: HP.OPEN,
  high: HP.HIGH,
  low:  HP.LOW,
  close: HP.CLOSE,
  volume: HP.VOLUME,
  currency: HP.CCY,
  ticker: HP.SYMBOL,
  name:   HP.NAME,
  mic:    HP.MIC,
  isin:   HP.ISIN,
  figi:   HP.FIGI,
  asOf:   HP.DATE,
};

/* ───────────────────── Normalization utilities ───────────────────── */

export function normalizeRow(
  src: HPRawRow,
  mapping: Record<string,string> = CatalogToHP,
  specs: Record<string, FieldSpec> = DefaultSpecs,
  ctx: NormalizeCtx = { baseCcy: "USD", fx: {}, tz: "UTC" }
): CatalogRow {
  const out: CatalogRow = {};
  for (const cat in mapping) {
    const key  = mapping[cat];
    const spec = specs[cat] || { name: cat };
    const raw  = src[key];
    (out as any)[cat] = normVal(raw, spec, src, ctx);
  }
  // asOf fallback
  if (!out.asOf) {
    const a = src[HP.DATE] || src["asOf"] || src["timestamp"] || src["ts"];
    if (a) out.asOf = toISO(a);
  }
  return out;
}

function normVal(v: any, spec: FieldSpec, row: Dict, ctx: NormalizeCtx): any {
  if (v == null) return v;
  if (spec.unit) {
    if (isMoney(spec.unit)) {
      const from = row[HP.CCY] || row["currency"] || ctx.baseCcy || "USD";
      return toBaseCcy(Number(v), String(from), ctx.baseCcy || "USD", ctx.fx || {});
    }
    if (spec.unit === "pct") return Number(v);          // 0.0123 = 1.23%
    if (spec.unit === "bps") return Number(v) / 10_000; // -> fraction
  }
  if (spec.type === "number") return Number(v);
  if (spec.type === "date")   return toISO(v);
  return String(v);
}

function isMoney(u:string){ return u==="USD"||u==="EUR"||u==="JPY"; }
function toBaseCcy(amount:number, from:string, base:string, fx:Record<string,number>):number{
  if (from === base) return amount;
  const p1 = (from+base).toUpperCase();
  const p2 = (base+from).toUpperCase();
  if (fx[p1]) return amount * fx[p1];
  if (fx[p2]) return amount / fx[p2];
  return amount;
}
function toISO(d:any):string {
  try { return new Date(d).toISOString(); } catch { return String(d); }
}

/* ───────────────────────── Pagination harness ───────────────────────── */

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

/* ───────────────────────── Adapter shell ───────────────────────── */

export class HammerAdapter {
  private limiter: TokenBucket;
  private backoff: Backoff;
  private transport?: Transport;

  constructor(
    private creds: HammerCredential,
    opts?: { rate?: RateRule; backoff?: Backoff; transport?: Transport }
  ) {
    this.limiter = new TokenBucket(opts?.rate || { capacity: 10, refillPerSec: 10 });
    this.backoff = opts?.backoff || { minMs: 200, maxMs: 5_000, factor: 2 };
    this.transport = opts?.transport; // inject when you wire network
  }

  setTransport(t: Transport){ this.transport = t; }

  /** Build a request for your internal Hammer gateway (placeholder). */
  buildRequest(req: HPReq): HttpRequest {
    const url = "https://YOUR-HAMMER-GATEWAY/placeholder";
    const body = JSON.stringify({
      dataset: req.dataset,
      symbols: req.symbols,
      fields:  req.fields,
      start:   req.start,
      end:     req.end,
      overrides: req.overrides,
      page:    req.pagination?.next,
      limit:   req.pagination?.limit || 500
    });
    const headers: Dict<string> = {
      "content-type": "application/json",
      "x-app":  this.creds.app  || "app",
      "x-user": this.creds.user || "user",
      "authorization": this.creds.token ? `Bearer ${this.creds.token}` : ""
    };
    return { method: "POST", url, headers, body, timeoutMs: 30_000 };
  }

  private async exec<T = any>(req: HttpRequest): Promise<T> {
    if (!this.transport) {
      // No transport yet: simulate deterministic payload for UI/dev.
      return this.simulate<T>(req);
    }
    await this.limiter.take(1);
    const res = await backoffRetry(async () => this.transport!(req), this.backoff, 5);
    if (res.status >= 400) throw new Error(`HAMMER_HTTP_${res.status}`);
    return safeJSON<T>(res.body);
  }

  /* ───────────── High-level fetchers ───────────── */

  async prices(req: Omit<HPReq,"dataset"> & { dataset?: "prices" }, ctx?: NormalizeCtx): Promise<CatalogRow[]> {
    const ent = checkScopes(this.creds, ["prices.read"]);
    if (!ent.ok) throw new Error("entitlements_missing:" + (ent.missing || []).join(","));
    const fields = req.fields && req.fields.length
      ? req.fields
      : [HP.OPEN, HP.HIGH, HP.LOW, HP.CLOSE, HP.VOLUME, HP.CCY, HP.SYMBOL, HP.DATE];

    const built = this.buildRequest({ ...req, dataset: "prices", fields });
    const first = await this.exec<{ rows: HPRawRow[]; next?: string }>(built);

    const out: CatalogRow[] = [];
    for (const r of (first.rows || [])) out.push(normalizeRow(r, CatalogToHP, DefaultSpecs, ctx));

    let next = first.next; let guard = 6;
    while (next && guard-- > 0) {
      const more = await this.exec<{ rows: HPRawRow[]; next?: string }>({
        ...built, body: patchJsonBody(built.body || "{}", { page: next })
      });
      for (const r of (more.rows || [])) out.push(normalizeRow(r, CatalogToHP, DefaultSpecs, ctx));
      next = more.next;
    }
    return out;
  }

  async reference(req: Omit<HPReq,"dataset"> & { dataset?: "reference" }, ctx?: NormalizeCtx): Promise<CatalogRow[]> {
    const ent = checkScopes(this.creds, ["ref.read"]);
    if (!ent.ok) throw new Error("entitlements_missing:" + (ent.missing || []).join(","));
    const fields = req.fields && req.fields.length
      ? req.fields
      : [HP.SYMBOL, HP.CCY, HP.NAME, HP.MIC, HP.ISIN, HP.FIGI, HP.UID, HP.DATE];

    const built = this.buildRequest({ ...req, dataset: "reference", fields });
    const data = await this.exec<{ rows: HPRawRow[] }>(built);
    const out: CatalogRow[] = [];
    for (const r of (data.rows || [])) out.push(normalizeRow(r, CatalogToHP, DefaultSpecs, ctx || { baseCcy:"USD", fx:{}, tz:"UTC" }));
    return out;
  }

  /* ───────────── Deterministic simulator ───────────── */

  private async simulate<T = any>(req: HttpRequest): Promise<T> {
    const body = safeJSON<any>(req.body || "{}");
    const { dataset, symbols, start, end } = body || {};
    const rows: any[] = [];

    const startTs = start ? Date.parse(start) : Date.now() - 10 * 24 * 3600 * 1000;
    const endTs   = end   ? Date.parse(end)   : Date.now();
    const days = Math.max(2, Math.min(120, Math.floor((endTs - startTs) / (24 * 3600 * 1000))));

    for (const sym of (symbols || ["SPY"])) {
      if (dataset === "reference") {
        rows.push({
          [HP.SYMBOL]: sym,
          [HP.CCY]: "USD",
          [HP.NAME]: `Hammer ${sym}`,
          [HP.MIC]: "XNAS",
          [HP.ISIN]: "US" + hash(sym).slice(0,10),
          [HP.FIGI]: "BBG" + hash(sym).slice(0,9),
          [HP.UID]:  "HM"  + hash(sym).slice(0,9),
          [HP.DATE]: new Date().toISOString(),
        });
      } else {
        let p = 90 + (hashNum(sym) % 60);
        for (let i = 0; i < days; i++) {
          p *= 0.995 + ((i * 48271 + 40692) % 1000) / 1000 * 0.003; // pseudo path
          const o = p * (0.996 + (i%7)*0.0006);
          const h = o * (1.010 + (i%5)*0.0009);
          const l = o * (0.990 - (i%3)*0.0006);
          const c = (o + h + l) / 3;
          const v = 900_000 + (hashNum(sym + ":" + i) % 250_000);
          rows.push({
            [HP.SYMBOL]: sym,
            [HP.CCY]: "USD",
            [HP.OPEN]: round(o),
            [HP.HIGH]: round(h),
            [HP.LOW]:  round(l),
            [HP.CLOSE]:round(c),
            [HP.VOLUME]: v,
            [HP.DATE]: new Date(startTs + i * 24 * 3600 * 1000).toISOString(),
          });
        }
      }
    }

    const resp = { status: 200, headers: {}, body: JSON.stringify({ rows }) };
    // @ts-ignore
    return respTo<T>(resp);
  }
}

/* ────────────────────────── Helpers ────────────────────────── */

function sleep(ms:number){ return new Promise(r=>setTimeout(r,ms)); }

function safeJSON<T=any>(s:string):T { try { return JSON.parse(s) as T; } catch { return {} as T; } }

function patchJsonBody(s:string, patch:Dict):string {
  const o = safeJSON<Dict>(s || "{}");
  for (const k in patch) o[k] = patch[k];
  return JSON.stringify(o);
}

function respTo<T>(r: HttpResponse): T {
  try { return JSON.parse(r.body) as T; } catch { return ({} as any as T); }
}

function round(x:number){ return Math.round(x*100)/100; }
function hash(s:string){ let x=2166136261>>>0; for(let i=0;i<s.length;i++){ x^=s.charCodeAt(i); x=Math.imul(x,16777619)>>>0; } return ("00000000"+x.toString(16)).slice(-8)+("00000000"+((x^0xa5a5a5a5)>>>0).toString(16)).slice(-8); }
function hashNum(s:string){ const h=hash(s); return parseInt(h.slice(0,8),16)>>>0; }