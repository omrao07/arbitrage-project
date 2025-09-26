// src/bloomberg.ts
// Zero-dependency helpers for working with Bloomberg-described datasets
// (as per catalog/catalog.schema.json). No imports used.
// - Type guards & schema-lite validation
// - Endpoint & field extraction for BPipe endpoints
// - Normalization utilities (dates, numbers, pct fractions, enums)
// - Vendor→canonical alias mapping
// - Pluggable transport so callers can wire any BPipe/REST client

// ------------------------------
// Catalog descriptor primitives
// ------------------------------
export type PrimitiveType =
  | "string" | "bool"
  | "int32" | "int64"
  | "float32" | "float64"
  | "date" | "timestamp";

export type Column = {
  name: string;
  type: PrimitiveType | `${PrimitiveType}[]`;
  unit?: string;               // e.g., "USD" | "pct" | "shares"
  nullable?: boolean;
  quality?: string[];
  desc?: string;
};

export type Endpoint = {
  kind: "file" | "rest" | "bpipe";
  path: string;                // URL / pattern / mnemonic list (space-delimited)
  params?: { [k: string]: any };
};

export type DatasetDescriptor = {
  id: string;
  title: string;
  vendor: string;
  source: "file" | "rest" | "bpipe";
  version: string;             // semver
  frequency: "minutely" | "hourly" | "daily" | "weekly" | "monthly" | "quarterly" | "annual" | "ad_hoc";
  description: string;
  primary_key: string[];
  partitions?: string[];
  columns: Column[];
  endpoints: Endpoint[];
  lineage?: string[];
  tags?: string[];
};

// ------------------------------
// Lightweight validation
// ------------------------------
export function validateDescriptorBasic(d: any): d is DatasetDescriptor {
  const okString = (v: any) => typeof v === "string" && v.length > 0;
  const okArray = (v: any) => Array.isArray(v);
  return (
    d && okString(d.id) && okString(d.title) && okString(d.vendor) &&
    okString(d.source) && okString(d.version) && okString(d.frequency) &&
    okString(d.description) && okArray(d.primary_key) &&
    okArray(d.columns) && okArray(d.endpoints)
  );
}

// ------------------------------
// Endpoint helpers
// ------------------------------
export function getEndpoint(d: DatasetDescriptor, kind: Endpoint["kind"]): Endpoint | undefined {
  for (let i = 0; i < d.endpoints.length; i++) {
    const e = d.endpoints[i];
    if (e.kind === kind) return e;
  }
  return undefined;
}

// Extract space-delimited BPipe mnemonics from endpoint.path
export function getBpipeFields(ep?: Endpoint): string[] {
  if (!ep || ep.kind !== "bpipe") return [];
  const s = String(ep.path || "").trim();
  if (!s) return [];
  // allow either space or comma separated
  return s.split(/[\s,]+/).filter(Boolean);
}

// ------------------------------
// Alias mapping (vendor → canonical)
// ------------------------------
export type AliasMap = { [vendorField: string]: string }; // e.g., { PX_LAST: "px_close" }

export function mapVendorToCanonical<T extends Record<string, any>>(row: T, aliases: AliasMap): Record<string, any> {
  const out: Record<string, any> = {};
  for (const k in row) {
    const canon = aliases[k] || k;
    // prefer not to overwrite if alias collides; last-write wins by design
    out[canon] = row[k];
  }
  return out;
}

// ------------------------------
// Normalization utilities
// ------------------------------
function isArrayType(t: string): boolean { return /\[\]$/.test(t); }
function baseType(t: string): PrimitiveType { return (t.replace(/\[\]$/, "") as PrimitiveType); }

function parseDateStrict(v: any): string | null {
  // Expect YYYY-MM-DD; also accept Date or ISO
  if (v == null || v === "") return null;
  if (typeof v === "string") {
    const s = v.trim();
    // normalize ISO -> YYYY-MM-DD
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[1]}-${m[2]}-${m[3]}`;
    // try JS Date fallback
    const t = Date.parse(s);
    if (!isNaN(t)) {
      const d = new Date(t);
      const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
      const dd = String(d.getUTCDate()).padStart(2, "0");
      return `${d.getUTCFullYear()}-${mm}-${dd}`;
    }
    return null;
  }
  if (v instanceof Date) {
    const mm = String(v.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(v.getUTCDate()).padStart(2, "0");
    return `${v.getUTCFullYear()}-${mm}-${dd}`;
  }
  return null;
}

function parseTimestampISO(v: any): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "string") {
    // If date-only, append T00:00:00Z
    if (/^\d{4}-\d{2}-\d{2}$/.test(v)) return `${v}T00:00:00Z`;
    // If no timezone, assume Z
    if (/^\d{4}-\d{2}-\d{2}T/.test(v) && !/[zZ]|[+\-]\d{2}:?\d{2}$/.test(v)) return `${v}Z`;
    // Basic sanity
    const t = Date.parse(v);
    if (!isNaN(t)) return new Date(t).toISOString();
    return null;
  }
  if (typeof v === "number" && isFinite(v)) {
    return new Date(v).toISOString();
  }
  if (v instanceof Date) return v.toISOString();
  return null;
}

function toNumber(v: any): number | null {
  if (v == null || v === "") return null;
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v === "boolean") return v ? 1 : 0;
  const n = Number(String(v).replace(/[_ ,]/g, ""));
  return isFinite(n) ? n : null;
}

function coerceScalar(v: any, t: PrimitiveType, unit?: string): any {
  switch (t) {
    case "string": return v == null ? null : String(v);
    case "bool": {
      if (typeof v === "boolean") return v;
      if (v == null || v === "") return null;
      const s = String(v).toLowerCase();
      if (s === "true" || s === "1" || s === "y" || s === "yes") return true;
      if (s === "false" || s === "0" || s === "n" || s === "no") return false;
      return null;
    }
    case "int32":
    case "int64": {
      const n = toNumber(v);
      return n == null ? null : Math.trunc(n);
    }
    case "float32":
    case "float64": {
      let n = toNumber(v);
      if (n == null) return null;
      // Heuristic: if unit === 'pct' we store FRACTIONS (0..1). If vendor sent "12" (12%) scale down.
      if (unit === "pct") {
        if (Math.abs(n) > 1.5) n = n / 100;
      }
      return n;
    }
    case "date": return parseDateStrict(v);
    case "timestamp": return parseTimestampISO(v);
  }
}

function coerceValue(v: any, typeSpec: string, unit?: string): any {
  if (!isArrayType(typeSpec)) return coerceScalar(v, baseType(typeSpec), unit);
  // array: accept arrays or delimited strings
  const bt = baseType(typeSpec);
  const arr: any[] =
    Array.isArray(v) ? v
      : (typeof v === "string" ? v.split(/[|;,]\s*/).filter(Boolean) : (v == null ? [] : [v]));
  const out: any[] = [];
  for (let i = 0; i < arr.length; i++) {
    const cv = coerceScalar(arr[i], bt, unit);
    if (cv !== null && cv !== undefined) out.push(cv);
  }
  return out;
}

export type NormalizeOptions = {
  strictNulls?: boolean;        // if true, fail when non-nullable coerces to null
  onError?: (field: string, value: any, reason: string) => void;
  enumRightCanonical?: boolean; // map CALL/PUT → C/P
};

export function normalizeRowToColumns(
  row: Record<string, any>,
  columns: Column[],
  opts: NormalizeOptions = {}
): Record<string, any> {
  const out: Record<string, any> = {};
  for (let i = 0; i < columns.length; i++) {
    const c = columns[i];
    let v = row[c.name];
    // special: normalize options 'right'
    if (c.name === "right" && opts.enumRightCanonical && v != null) {
      const s = String(v).toUpperCase();
      if (s === "CALL") v = "C";
      else if (s === "PUT") v = "P";
    }
    const cv = coerceValue(v, c.type, c.unit);
    if (cv === null || (Array.isArray(cv) && cv.length === 0)) {
      if (opts.strictNulls && c.nullable !== true) {
        if (opts.onError) opts.onError(c.name, v, "null_not_allowed");
      }
      // leave as undefined to avoid writing nulls unless nullable
      if (c.nullable) out[c.name] = null;
    } else {
      out[c.name] = cv;
    }
  }
  return out;
}

// ------------------------------
// BPipe request builder (logical)
// ------------------------------
export type BpipeRequest = {
  fields: string[];            // e.g., ["PX_OPEN","PX_LAST",...]
  universe: string[];          // list of identifiers (tickers/BBGTickers/etc.)
  options?: { [k: string]: any };
};

export function buildBpipeRequest(
  descriptor: DatasetDescriptor,
  universe: string[],
  overrideFields?: string[]
): BpipeRequest {
  const ep = getEndpoint(descriptor, "bpipe");
  const fields = (overrideFields && overrideFields.length > 0)
    ? overrideFields.slice()
    : getBpipeFields(ep);
  return { fields, universe: universe.slice(), options: (ep && ep.params) ? ep.params : {} };
}

// ------------------------------
// Pluggable transport & client
// ------------------------------
export type TransportResponse = {
  ok: boolean;
  status?: number;
  error?: string;
  // Normalize to array of { id, data: { field:value } }
  rows?: Array<{ id: string; data: Record<string, any> }>;
};

export type Transport = (req: BpipeRequest) => Promise<TransportResponse>;

/**
 * BloombergClient is a tiny orchestrator around:
 *  - alias mapping
 *  - type normalization to a catalog descriptor
 *  - pluggable transport (user supplies actual BPipe/EMSX/REST implementation)
 */
export class BloombergClient {
  private transport: Transport;

  constructor(opts: { transport: Transport }) {
    if (!opts || typeof opts.transport !== "function") {
      throw new Error("BloombergClient requires a transport(req) => Promise<TransportResponse>.");
    }
    this.transport = opts.transport;
  }

  async fetchAndNormalize(args: {
    descriptor: DatasetDescriptor;
    universe: string[];
    aliases?: AliasMap;
    overrideFields?: string[];
    normalize?: NormalizeOptions;
    attachIdAs?: string; // when vendor returns rows keyed by 'id', copy to this column (e.g., 'ticker')
  }): Promise<{ rows: Record<string, any>[]; meta: { requestedFields: string[] } }> {
    const { descriptor, universe, aliases = {}, overrideFields, normalize, attachIdAs } = args;

    if (!validateDescriptorBasic(descriptor)) {
      throw new Error("Invalid descriptor (basic checks failed).");
    }
    if (!Array.isArray(universe) || universe.length === 0) {
      return { rows: [], meta: { requestedFields: [] } };
    }

    const req = buildBpipeRequest(descriptor, universe, overrideFields);
    const rsp = await this.transport(req);
    if (!rsp || !rsp.ok) {
      const msg = rsp && rsp.error ? rsp.error : "Transport error";
      throw new Error(`Bloomberg transport failed: ${msg}`);
    }

    const out: Record<string, any>[] = [];
    const cols = descriptor.columns;

    const onError = (field: string, value: any, reason: string) => {
      // Soft log to console; caller can override via normalize.onError
      try { /* eslint-disable no-console */ console.warn?.("[normalizeRow]", field, reason, value); } catch { /* ignore */ }
      if (normalize && normalize.onError) normalize.onError(field, value, reason);
    };

    for (let i = 0; i < (rsp.rows || []).length; i++) {
      const r = rsp.rows![i]; // { id, data }
      const vendorRow = r.data || {};
      const canonRow = mapVendorToCanonical(vendorRow, aliases);
      if (attachIdAs) {
        // Only set attachIdAs when not already present
        if (canonRow[attachIdAs] == null || canonRow[attachIdAs] === "") {
          canonRow[attachIdAs] = r.id;
        }
      }
      const norm = normalizeRowToColumns(canonRow, cols, {
        enumRightCanonical: true,
        strictNulls: false,
        onError
      });
      out.push(norm);
    }

    return { rows: out, meta: { requestedFields: req.fields } };
  }
}

// ------------------------------
// Misc helpers
// ------------------------------
export function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  if (!Array.isArray(arr) || size <= 0) return out;
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

export function pick<T extends Record<string, any>>(obj: T, keys: string[]): Partial<T> {
  const o: Partial<T> = {};
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    if (k in obj) (o as any)[k] = obj[k];
  }
  return o;
}

export function ensurePrimaryKey(row: Record<string, any>, pk: string[]): string {
  const parts: string[] = [];
  for (let i = 0; i < pk.length; i++) {
    const k = pk[i];
    const v = row[k];
    parts.push(`${v === undefined ? "" : String(v)}`);
  }
  return parts.join("|");
}

// ------------------------------
// Example alias packs (optional)
// ------------------------------
export const Aliases = {
  equitiesEod: {
    PX_OPEN: "px_open",
    PX_HIGH: "px_high",
    PX_LOW: "px_low",
    PX_LAST: "px_close",
    VOLUME: "volume",
    DVD_AMT: "dividend_amt",
    CRNCY: "currency"
  } as AliasMap,
  optionsGreeks: {
    OPT_EXPIRE_DT: "expiration_dt",
    OPT_STRIKE_PX: "strike",
    OPT_PUT_CALL: "right",
    EXERCISE_STYLE: "exercise_style",
    MULTIPLIER: "multiplier",
    CRNCY: "currency",
    BID: "bid",
    ASK: "ask",
    PX_LAST: "last",
    MID: "mid",
    THEO: "mark",
    OPEN_INT: "open_interest",
    VOLUME: "volume",
    UNDERLYING_PX_CLOSE: "underlying_close",
    IVOL_MID: "iv",
    DELTA: "delta",
    GAMMA: "gamma",
    VEGA: "vega",
    THETA: "theta",
    RHO: "rho"
  } as AliasMap
};

// ------------------------------
// Minimal fake transport (for tests)
// ------------------------------
export function makeEchoTransport(sample: Array<{ id: string; data: Record<string, any> }>): Transport {
  return async function echoTransport(_req: BpipeRequest): Promise<TransportResponse> {
    return { ok: true, rows: sample };
  };
}

// ------------------------------
// Usage sketch (commented)
//
// const client = new BloombergClient({ transport: realBpipeTransport });
// const { rows } = await client.fetchAndNormalize({
//   descriptor,                      // DatasetDescriptor (validated elsewhere)
//   universe: ["AAPL US Equity","MSFT US Equity"],
//   aliases: Aliases.equitiesEod,
//   attachIdAs: "bbg_ticker"
// });
// ------------------------------