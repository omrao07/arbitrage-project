// adapters/src/types.ts
// Shared primitive types & small helpers for adapters.
// Pure TypeScript. No imports. No side effects.

/* ────────────────────────────────────────────────────────────────────────── *
 * Basic utility types
 * ────────────────────────────────────────────────────────────────────────── */

export type Dict<T = any> = { [k: string]: T };
export type Maybe<T> = T | null | undefined;

export type JSONPrimitive = string | number | boolean | null;
export type JSONValue = JSONPrimitive | JSONObject | JSONArray;
export type JSONObject = { [k: string]: JSONValue };
export type JSONArray = JSONValue[];

export type Bytes = Uint8Array;

/* ────────────────────────────────────────────────────────────────────────── *
 * Logging
 * ────────────────────────────────────────────────────────────────────────── */

export type Logger = {
  info?: (...args: any[]) => void;
  warn?: (...args: any[]) => void;
  error?: (...args: any[]) => void;
  debug?: (...args: any[]) => void;
};

export const NullLogger: Readonly<Logger> = Object.freeze({
  info: () => {},
  warn: () => {},
  error: () => {},
  debug: () => {}
});

/* ────────────────────────────────────────────────────────────────────────── *
 * Adapter contracts
 * ────────────────────────────────────────────────────────────────────────── */

export type AdapterContext = {
  /** UTC ISO timestamp when the run started */
  startedAt?: string;
  /** Arbitrary environment bag (e.g., { tz, region, dryRun, ... }) */
  env?: Dict;
  /** Version/revision string for tracing */
  version?: string;
  /** Optional logger hook; defaults to console.* or NullLogger by caller */
  log?: Logger;
};

export type AdapterResult<O = any> = {
  ok: boolean;
  /** Primary payload; for row transforms prefer `{ rows: any[] }`. */
  data?: O;
  /** Free-form metadata (timings, counts, provenance, etc.) */
  meta?: Dict;
  /** Error message if ok=false */
  error?: string;
};

export type Adapter<I = any, O = any> =
  (input: I, ctx: AdapterContext) => AdapterResult<O> | Promise<AdapterResult<O>>;

export type RowsPayload = { rows: any[]; meta?: Dict };

/* ────────────────────────────────────────────────────────────────────────── *
 * Pipeline contracts
 * ────────────────────────────────────────────────────────────────────────── */

export type PipelineStep = string;

export type PipelineResult<T = any> = AdapterResult<T> & {
  steps?: Array<{ name: string; ok: boolean; meta?: Dict; error?: string }>;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * HTTP / REST transport (generic)
 * ────────────────────────────────────────────────────────────────────────── */

export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE" | "PATCH" | "HEAD";

export type RestRequest = {
  method: HttpMethod;
  path: string;  // e.g. "/v1/options/eod"
  query?: Dict<string | number | boolean | string[] | number[]>;
  headers?: Dict<string>;
  body?: any;    // JSON-serializable
};

export type RestResponse<T = any> = {
  ok: boolean;
  status?: number;
  data?: T;
  error?: string;
  headers?: Dict<string>;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Storage contracts (S3/GCS/memory compatible)
 * ────────────────────────────────────────────────────────────────────────── */

export type PutInput = string | Bytes | ArrayBuffer | Dict | any[];

export type PutOptions = {
  contentType?: string;
  metadata?: Dict<string>;
  ifNoneMatch?: boolean; // 412 if exists
};

export type StoredObject = {
  key: string;
  size: number;
  contentType?: string;
  contentEncoding?: string;
  etag?: string;                 // weak hash (hex)
  lastModified?: string;         // ISO-8601 UTC
  metadata?: Dict<string>;
};

export type HeadResponse = {
  ok: boolean;
  status?: number;
  object?: StoredObject;
  error?: string;
};

export type GetResponse = HeadResponse & {
  content?: Bytes;
};

export type ListOptions = {
  prefix?: string;
  delimiter?: string;            // "/" for shallow listings
  limit?: number;
  cursor?: string;               // opaque paging cursor
};

export type ListResponse = {
  ok: boolean;
  items: StoredObject[];
  nextCursor?: string;
  truncated?: boolean;
  error?: string;
};

export interface Storage {
  put(key: string, body: PutInput, opts?: PutOptions): Promise<HeadResponse>;
  head(key: string): Promise<HeadResponse>;
  get(key: string): Promise<GetResponse>;
  delete(key: string): Promise<{ ok: boolean; status?: number; error?: string }>;
  list(opts?: ListOptions): Promise<ListResponse>;
  exists(key: string): Promise<boolean>;
  signUrl?(
    key: string,
    opts?: { method?: "GET" | "PUT"; ttlSec?: number; contentType?: string }
  ): Promise<{ url: string; expiresAt?: string }>;
  copy?(
    srcKey: string,
    dstKey: string,
    opts?: { overwrite?: boolean; metadata?: Dict<string> }
  ): Promise<{ ok: boolean; status?: number; error?: string }>;
  move?(
    srcKey: string,
    dstKey: string,
    opts?: { overwrite?: boolean; metadata?: Dict<string> }
  ): Promise<{ ok: boolean; status?: number; error?: string }>;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Catalog / schema-lite (mirrors catalog.schema.json)
 * ────────────────────────────────────────────────────────────────────────── */

export type PrimitiveType =
  | "string" | "bool"
  | "int32" | "int64"
  | "float32" | "float64"
  | "date" | "timestamp";

export type Column = {
  name: string;
  type: PrimitiveType | `${PrimitiveType}[]`;
  unit?: string;               // "USD" | "pct" | "shares" | "contracts"
  nullable?: boolean;
  quality?: string[];
  desc?: string;
};

export type Endpoint = {
  kind: "file" | "rest" | "bpipe";
  path: string;                // URL / pattern / space-delimited mnemonics
  params?: Dict<any>;
};

export type Frequency =
  | "minutely" | "hourly" | "daily" | "weekly"
  | "monthly" | "quarterly" | "annual" | "ad_hoc";

export type DatasetDescriptor = {
  id: string;
  title: string;
  vendor: string;
  source: "file" | "rest" | "bpipe";
  version: string;             // semver "x.y.z"
  frequency: Frequency;
  description: string;
  primary_key: string[];
  partitions?: string[];
  columns: Column[];
  endpoints: Endpoint[];
  lineage?: string[];
  tags?: string[];
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Vendor normalization helpers (types only)
 * ────────────────────────────────────────────────────────────────────────── */

export type AliasMap = { [vendorField: string]: string };

export type NormalizeOptions = {
  strictNulls?: boolean;
  enumRightCanonical?: boolean; // CALL/PUT → C/P
  onError?: (field: string, value: any, reason: string) => void;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Tiny helpers (no side effects)
 * ────────────────────────────────────────────────────────────────────────── */

export function ensureArray<T = any>(v: any): T[] {
  return Array.isArray(v) ? (v as T[]) : (v == null ? [] : [v as T]);
}

export function pick<T extends Dict>(obj: T, keys: string[]): Partial<T> {
  const o: Partial<T> = {};
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    if (k in obj) (o as any)[k] = obj[k];
  }
  return o;
}

export function omit<T extends Dict>(obj: T, keys: string[]): Partial<T> {
  const set: Dict<boolean> = {};
  for (let i = 0; i < keys.length; i++) set[keys[i]] = true;
  const o: Partial<T> = {};
  for (const k in obj) if (!set[k]) (o as any)[k] = obj[k];
  return o;
}

export function isOk<T = any>(r: AdapterResult<T>): r is AdapterResult<T> & { ok: true } {
  return !!r && r.ok === true;
}

export function isErr<T = any>(r: AdapterResult<T>): r is AdapterResult<T> & { ok: false; error: string } {
  return !!r && r.ok === false;
}

export function stableStringify(v: any): string {
  try { return JSON.stringify(sortKeysDeep(v)); } catch { return String(v); }
}

function sortKeysDeep(v: any): any {
  if (Array.isArray(v)) return v.map(sortKeysDeep);
  if (v && typeof v === "object") {
    const out: Dict = {};
    const ks = Object.keys(v).sort();
    for (let i = 0; i < ks.length; i++) out[ks[i]] = sortKeysDeep(v[ks[i]]);
    return out;
  }
  return v;
}

export function assertNever(x: never, msg?: string): never {
  throw new Error(msg || `Unexpected value: ${String(x)}`);
}