// adapters/src/koyfin.ts
// Zero-dependency Koyfin REST client + light normalizers.
// - No imports
// - Pluggable transport (so you can wire fetch/axios/got/etc. outside)
// - Helpers for: building requests, mapping vendor→canonical, and coercing types

/* ────────────────────────────────────────────────────────────────────────── *
 * Types
 * ────────────────────────────────────────────────────────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type HttpMethod = "GET" | "POST";

export type RestRequest = {
  method: HttpMethod;
  /** e.g. "/v1/options/eod" (joined with baseUrl by the client) */
  path: string;
  /** query params (values will be url-encoded) */
  query?: Dict<string | number | boolean | string[] | number[]>;
  /** headers merged on top of default headers */
  headers?: Dict<string>;
  /** JSON-serializable body (for POST) */
  body?: any;
};

export type RestResponse<T = any> = {
  ok: boolean;
  status?: number;
  data?: T;
  error?: string;
  headers?: Dict<string>;
};

export type Transport = (req: RestRequest & { url: string }) => Promise<RestResponse>;

/* ────────────────────────────────────────────────────────────────────────── *
 * Client
 * ────────────────────────────────────────────────────────────────────────── */

export type KoyfinClientOpts = {
  /** e.g., "https://api.koyfin.com" (no trailing slash needed). Defaults to "" (path must be absolute). */
  baseUrl?: string;
  /** If set, adds Authorization: Bearer <apiKey> */
  apiKey?: string;
  /** Supply your own transport (recommended). */
  transport: Transport;
  /** Extra headers added to every request (optional) */
  defaultHeaders?: Dict<string>;
};

export class KoyfinClient {
  private baseUrl: string;
  private apiKey?: string;
  private transport: Transport;
  private defaultHeaders: Dict<string>;

  constructor(opts: KoyfinClientOpts) {
    if (!opts || typeof opts.transport !== "function") {
      throw new Error("KoyfinClient requires a transport(req) => Promise<RestResponse>.");
    }
    this.baseUrl = (opts.baseUrl || "").replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.transport = opts.transport;
    this.defaultHeaders = Object.assign(
      { "Content-Type": "application/json" },
      opts.defaultHeaders || {}
    );
  }

  async request<T = any>(req: RestRequest): Promise<RestResponse<T>> {
    const url = this.buildUrl(req.path, req.query);
    const headers = Object.assign({}, this.defaultHeaders, req.headers || {});
    if (this.apiKey && !headers["Authorization"]) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    const wireReq: RestRequest & { url: string } = {
      method: req.method,
      path: req.path,
      query: req.query,
      headers,
      body: req.body,
      url
    };
    return this.transport(wireReq) as Promise<RestResponse<T>>;
  }

  /* ----------------------------- High-level APIs ----------------------------- */

  /**
   * Options EOD chain + greeks
   * GET /v1/options/eod?symbols=AAPL,MSFT&date=YYYY-MM-DD
   */
  async getOptionsEod(params: {
    symbols: string[];       // underlying tickers/symbols
    date?: string;           // YYYY-MM-DD (UTC)
    fields?: string[];       // optional fields filter if endpoint supports
  }): Promise<RestResponse<{ rows: any[] }>> {
    const q: Dict = { symbols: params.symbols.join(",") };
    if (params.date) q.date = normalizeDate(params.date) || params.date;
    if (params.fields && params.fields.length) q.fields = params.fields.join(",");
    return this.request<{ rows: any[] }>({ method: "GET", path: "/v1/options/eod", query: q });
  }

  /**
   * Equities EOD OHLCV (common convenience)
   * GET /v1/equities/eod?tickers=AAPL,MSFT&date=YYYY-MM-DD
   */
  async getEquitiesEod(params: {
    tickers: string[];
    date?: string;
    fields?: string[];
  }): Promise<RestResponse<{ rows: any[] }>> {
    const q: Dict = { tickers: params.tickers.join(",") };
    if (params.date) q.date = normalizeDate(params.date) || params.date;
    if (params.fields && params.fields.length) q.fields = params.fields.join(",");
    return this.request<{ rows: any[] }>({ method: "GET", path: "/v1/equities/eod", query: q });
  }

  /**
   * News headlines (normalized metadata)
   * GET /v1/news/headlines?tickers=AAPL,MSFT&limit=...
   */
  async getNewsHeadlines(params: {
    tickers?: string[];
    limit?: number;
    from?: string;    // ISO or YYYY-MM-DD
    to?: string;      // ISO or YYYY-MM-DD
  }): Promise<RestResponse<{ rows: any[] }>> {
    const q: Dict = {};
    if (params.tickers && params.tickers.length) q.tickers = params.tickers.join(",");
    if (params.limit != null) q.limit = params.limit;
    if (params.from) q.from = toISO(params.from);
    if (params.to) q.to = toISO(params.to);
    return this.request<{ rows: any[] }>({ method: "GET", path: "/v1/news/headlines", query: q });
  }

  /* ----------------------------- URL helpers -------------------------------- */

  private buildUrl(path: string, query?: Dict): string {
    const p = (path || "").startsWith("http") ? path : `${this.baseUrl}${ensureLeadingSlash(path)}`;
    if (!query || !Object.keys(query).length) return p;
    const qs = encodeQuery(query);
    return qs ? `${p}?${qs}` : p;
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Vendor → Canonical field maps (extend as needed)
 * ────────────────────────────────────────────────────────────────────────── */

export const Aliases = {
  optionsGreeks: {
    expiration: "expiration_dt",
    expiration_dt: "expiration_dt",
    dte: "dte",
    strike_price: "strike",
    strike: "strike",
    put_call: "right",
    style: "exercise_style",
    settlement_type: "settlement",
    contract_multiplier: "multiplier",
    currency: "currency",
    bid: "bid",
    ask: "ask",
    mid: "mid",
    last: "last",
    mark: "mark",
    vol: "volume",
    volume: "volume",
    openInterest: "open_interest",
    open_interest: "open_interest",
    underlyingClose: "underlying_close",
    underlying_close: "underlying_close",
    iv: "iv",
    delta: "delta",
    gamma: "gamma",
    vega: "vega",
    theta: "theta",
    rho: "rho",
    tenor: "surface_tenor",
    option_id: "option_id",
    underlying: "underlying_ticker",
    underlying_ticker: "underlying_ticker",
    dt: "dt"
  } as Dict<string>,
  equitiesEod: {
    date: "dt",
    dt: "dt",
    ticker: "ticker",
    px_open: "px_open",
    px_high: "px_high",
    px_low: "px_low",
    px_close: "px_close",
    volume: "volume",
    currency: "currency"
  } as Dict<string>,
  newsHeadlines: {
    ts: "ts",
    dt: "dt",
    id: "id",
    source: "source_name",
    title: "title",
    abstract: "summary",
    url: "url",
    symbols: "tickers",
    sentiment: "sentiment",
    importance: "importance",
    lang: "lang"
  } as Dict<string>
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Normalization helpers
 * ────────────────────────────────────────────────────────────────────────── */

export function mapFields(row: Dict, aliases: Dict<string>): Dict {
  const out: Dict = {};
  for (const k in row) {
    const canon = aliases[k] || aliases[k.toLowerCase?.() as string] || k;
    out[canon] = row[k];
  }
  return out;
}

export function coerceNumber(v: any): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number" && isFinite(v)) return v;
  const n = Number(String(v).replace(/[_ ,]/g, ""));
  return isFinite(n) ? n : null;
}

export function coercePercentFraction(v: any): number | null {
  const n = coerceNumber(v);
  if (n == null) return null;
  return Math.abs(n) > 1.5 ? n / 100 : n;
}

export function normalizeDate(v: any): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "string") {
    const s = v.trim();
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[1]}-${m[2]}-${m[3]}`;
    const t = Date.parse(s);
    if (!isNaN(t)) {
      const d = new Date(t);
      return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
    }
    return null;
  }
  if (v instanceof Date) {
    return `${v.getUTCFullYear()}-${pad2(v.getUTCMonth() + 1)}-${pad2(v.getUTCDate())}`;
  }
  return null;
}

export function toISO(v: any): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "string") {
    // If date-only, append T00:00:00Z
    if (/^\d{4}-\d{2}-\d{2}$/.test(v)) return `${v}T00:00:00Z`;
    const t = Date.parse(v);
    return isNaN(t) ? null : new Date(t).toISOString();
  }
  if (typeof v === "number" && isFinite(v)) return new Date(v).toISOString();
  if (v instanceof Date) return v.toISOString();
  return null;
}

function pad2(n: number): string { return String(n).padStart(2, "0"); }

/* ────────────────────────────────────────────────────────────────────────── *
 * Canonical normalizers (schema-lite)
 * ────────────────────────────────────────────────────────────────────────── */

export function normalizeOptionRow(row: Dict): Dict {
  const r = mapFields(row, Aliases.optionsGreeks);

  // Enum fixups
  if (r.right) {
    const s = String(r.right).toUpperCase();
    r.right = s === "CALL" ? "C" : (s === "PUT" ? "P" : s);
  }

  // Numbers
  r.strike = coerceNumber(r.strike);
  r.multiplier = coerceNumber(r.multiplier);
  r.bid = coerceNumber(r.bid);
  r.ask = coerceNumber(r.ask);
  r.mid = coerceNumber(r.mid);
  r.last = coerceNumber(r.last);
  r.mark = coerceNumber(r.mark);
  r.volume = coerceNumber(r.volume);
  r.open_interest = coerceNumber(r.open_interest);
  r.underlying_close = coerceNumber(r.underlying_close);

  // Greeks & vol
  r.iv = coercePercentFraction(r.iv);
  r.delta = coerceNumber(r.delta);
  r.gamma = coerceNumber(r.gamma);
  r.vega = coerceNumber(r.vega);
  r.theta = coerceNumber(r.theta);
  r.rho = coerceNumber(r.rho);

  // Dates
  r.dt = normalizeDate(r.dt);
  r.expiration_dt = normalizeDate(r.expiration_dt);
  r.dte = r.dte == null ? null : Math.trunc(Number(r.dte));

  return r;
}

export function normalizeEquityEodRow(row: Dict): Dict {
  const r = mapFields(row, Aliases.equitiesEod);
  r.dt = normalizeDate(r.dt);
  r.px_open = coerceNumber(r.px_open);
  r.px_high = coerceNumber(r.px_high);
  r.px_low = coerceNumber(r.px_low);
  r.px_close = coerceNumber(r.px_close);
  r.volume = coerceNumber(r.volume);
  return r;
}

export function normalizeNewsRow(row: Dict): Dict {
  const r = mapFields(row, Aliases.newsHeadlines);
  r.ts = toISO(r.ts);
  r.dt = normalizeDate(r.dt);
  // normalize tickers to string[]
  if (r.tickers != null) {
    r.tickers = Array.isArray(r.tickers)
      ? r.tickers.map((x: any) => String(x))
      : String(r.tickers).split(/[|,;]\s*/).filter(Boolean);
  }
  r.sentiment = r.sentiment == null ? null : Number(r.sentiment);
  r.importance = r.importance == null ? null : Number(r.importance);
  return r;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Convenience: batch normalize arrays
 * ────────────────────────────────────────────────────────────────────────── */

export function normalizeOptionsArray(rows: any[]): any[] {
  const out: any[] = [];
  if (!Array.isArray(rows)) return out;
  for (let i = 0; i < rows.length; i++) out.push(normalizeOptionRow(rows[i]));
  return out;
}

export function normalizeEquityEodArray(rows: any[]): any[] {
  const out: any[] = [];
  if (!Array.isArray(rows)) return out;
  for (let i = 0; i < rows.length; i++) out.push(normalizeEquityEodRow(rows[i]));
  return out;
}

export function normalizeNewsArray(rows: any[]): any[] {
  const out: any[] = [];
  if (!Array.isArray(rows)) return out;
  for (let i = 0; i < rows.length; i++) out.push(normalizeNewsRow(rows[i]));
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Query & utility helpers
 * ────────────────────────────────────────────────────────────────────────── */

export function encodeQuery(q: Dict | undefined): string {
  if (!q) return "";
  const parts: string[] = [];
  for (const k in q) {
    const v = (q as any)[k];
    if (v == null) continue;
    if (Array.isArray(v)) {
      if (v.length === 0) continue;
      parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v.join(","))}`);
    } else {
      parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
    }
  }
  return parts.join("&");
}

export function ensureLeadingSlash(p: string): string {
  if (!p) return "/";
  return p.startsWith("/") ? p : `/${p}`;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Minimal echo transport (for tests)
 * ────────────────────────────────────────────────────────────────────────── */

export function makeEchoTransport(sample: any): Transport {
  return async function echo(req: RestRequest & { url: string }): Promise<RestResponse> {
    return { ok: true, status: 200, data: typeof sample === "function" ? sample(req) : sample };
  };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Example usage (commented)
 *
 * // const client = new KoyfinClient({
 * //   baseUrl: "https://api.koyfin.com",
 * //   apiKey: process.env.KOYFIN_API_KEY!,
 * //   transport: async (req) => {
 * //     const res = await fetch(req.url, {
 * //       method: req.method,
 * //       headers: req.headers,
 * //       body: req.method === "POST" && req.body != null ? JSON.stringify(req.body) : undefined
 * //     });
 * //     const data = await res.json().catch(() => ({}));
 * //     return { ok: res.ok, status: res.status, data, error: res.ok ? undefined : String(data?.error || res.statusText) };
 * //   }
 * // });
 * //
 * // const raw = await client.getOptionsEod({ symbols: ["AAPL","MSFT"], date: "2025-01-03" });
 * // const rows = normalizeOptionsArray(raw.data?.rows || []);
 * // console.log(rows[0]);
 * ────────────────────────────────────────────────────────────────────────── */