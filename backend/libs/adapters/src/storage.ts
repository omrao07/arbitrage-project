// adapters/src/storage.ts
// Zero-dependency storage primitives + in-memory implementation.
// No imports. Pure TypeScript that compiles in Node or browser.
//
// What you get:
//  - A generic Storage interface (put/get/head/list/delete/exists).
//  - A MemoryStorage (great for tests, workers, or local dev).
//  - Helpers for UTF-8, JSON/CSV, content-type guessing, stable stringify.
//  - Tiny hash/etag, path normalization, and optional retry wrappers.

export type Dict<T = any> = { [k: string]: T };

// ------------------------------- Types --------------------------------------

export type Bytes = Uint8Array;

export type PutInput = string | Bytes | ArrayBuffer | Dict | any[];

export type PutOptions = {
  contentType?: string;
  metadata?: Dict<string>;
  /** if true and key exists, do not overwrite (HTTP 412-like) */
  ifNoneMatch?: boolean;
};

export type StoredObject = {
  key: string;
  size: number;
  contentType?: string;
  contentEncoding?: string;
  etag?: string;                 // weak hash (hex)
  lastModified?: string;         // ISO8601 UTC
  metadata?: Dict<string>;
};

export type HeadResponse = {
  ok: boolean;
  status?: number;               // 200, 404, 412, etc (informational)
  object?: StoredObject;
  error?: string;
};

export type GetResponse = HeadResponse & {
  content?: Bytes;
};

export type ListOptions = {
  prefix?: string;
  delimiter?: string;            // if set (e.g., "/"), returns only shallow matches
  limit?: number;                // max items (paging)
  cursor?: string;               // opaque cursor returned by previous list
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
  /** Optional helpers (not required by all impls) */
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

// -------------------------- In-memory storage -------------------------------

type MemRow = {
  key: string;
  bytes: Bytes;
  meta: StoredObject;
};

export class MemoryStorage implements Storage {
  private map: { [k: string]: MemRow } = Object.create(null);

  constructor(initial?: Array<{ key: string; body: PutInput; options?: PutOptions }>) {
    if (Array.isArray(initial)) {
      for (let i = 0; i < initial.length; i++) {
        const it = initial[i];
        // fire and forget; ignore errors during boot
        this.putSync(it.key, it.body, it.options);
      }
    }
  }

  async put(key: string, body: PutInput, opts?: PutOptions): Promise<HeadResponse> {
    try {
      return this.putSync(key, body, opts);
    } catch (e: any) {
      return { ok: false, status: 500, error: stringifyError(e) };
    }
  }

  private putSync(key: string, body: PutInput, opts?: PutOptions): HeadResponse {
    key = normKey(key);
    if (!key) return { ok: false, status: 400, error: "Invalid key" };
    if (opts && opts.ifNoneMatch && this.map[key]) {
      return { ok: false, status: 412, error: "Precondition failed (exists)" };
    }

    const bytes = toBytes(body);
    const ct = (opts && opts.contentType) || guessContentType(key);
    const now = nowIso();
    const etag = hashHex(bytes);

    const meta: StoredObject = {
      key,
      size: bytes.length,
      contentType: ct,
      etag,
      lastModified: now,
      metadata: opts && opts.metadata ? shallowClone(opts.metadata) : undefined
    };

    this.map[key] = { key, bytes, meta };
    return { ok: true, status: 200, object: shallowClone(meta) };
  }

  async head(key: string): Promise<HeadResponse> {
    key = normKey(key);
    const row = this.map[key];
    if (!row) return { ok: false, status: 404, error: "Not found" };
    return { ok: true, status: 200, object: shallowClone(row.meta) };
  }

  async get(key: string): Promise<GetResponse> {
    key = normKey(key);
    const row = this.map[key];
    if (!row) return { ok: false, status: 404, error: "Not found" };
    return {
      ok: true,
      status: 200,
      content: row.bytes.slice(0), // copy
      object: shallowClone(row.meta)
    };
  }

  async delete(key: string): Promise<{ ok: boolean; status?: number; error?: string }> {
    key = normKey(key);
    if (!this.map[key]) return { ok: false, status: 404, error: "Not found" };
    delete this.map[key];
    return { ok: true, status: 200 };
  }

  async exists(key: string): Promise<boolean> {
    return !!this.map[normKey(key)];
  }

  async list(opts?: ListOptions): Promise<ListResponse> {
    const prefix = (opts && opts.prefix ? normKey(opts.prefix) : "") || "";
    const delimiter = opts && opts.delimiter ? String(opts.delimiter) : "";
    const limit = Math.max(0, Math.trunc((opts && opts.limit) || 0));
    const cursor = opts && opts.cursor ? String(opts.cursor) : undefined;

    // gather keys
    const allKeys = Object.keys(this.map).sort();
    let start = 0;
    if (cursor) {
      // cursor is the last key of the previous page
      const idx = allKeys.indexOf(cursor);
      start = idx >= 0 ? idx + 1 : 0;
    }

    const items: StoredObject[] = [];
    for (let i = start; i < allKeys.length; i++) {
      const k = allKeys[i];
      if (prefix && !k.startsWith(prefix)) continue;
      if (delimiter) {
        // If any remaining part after prefix contains the delimiter, skip (only shallow listing)
        const tail = k.slice(prefix.length);
        if (tail.includes(delimiter)) continue;
      }
      items.push(shallowClone(this.map[k].meta));
      if (limit && items.length >= limit) {
        const lastKey = k;
        return { ok: true, items, truncated: true, nextCursor: lastKey };
      }
    }
    return { ok: true, items, truncated: false };
  }

  // Convenience: fake signed URL (mem://)
  async signUrl(
    key: string,
    opts?: { method?: "GET" | "PUT"; ttlSec?: number; contentType?: string }
  ): Promise<{ url: string; expiresAt?: string }> {
    const ttl = opts && opts.ttlSec ? Math.max(1, Math.trunc(opts.ttlSec)) : 300;
    const exp = new Date(Date.now() + ttl * 1000).toISOString();
    return { url: `mem://${encodeURIComponent(normKey(key))}`, expiresAt: exp };
  }

  // Optional helpers for copy/move (local in-map ops)
  async copy(
    srcKey: string,
    dstKey: string,
    opts?: { overwrite?: boolean; metadata?: Dict<string> }
  ): Promise<{ ok: boolean; status?: number; error?: string }> {
    srcKey = normKey(srcKey);
    dstKey = normKey(dstKey);
    const src = this.map[srcKey];
    if (!src) return { ok: false, status: 404, error: "Source not found" };
    if (!opts?.overwrite && this.map[dstKey]) return { ok: false, status: 412, error: "Destination exists" };
    const meta: StoredObject = {
      key: dstKey,
      size: src.bytes.length,
      contentType: src.meta.contentType,
      etag: hashHex(src.bytes),
      lastModified: nowIso(),
      metadata: opts && opts.metadata ? shallowClone(opts.metadata) : shallowClone(src.meta.metadata || {})
    };
    this.map[dstKey] = { key: dstKey, bytes: src.bytes.slice(0), meta };
    return { ok: true, status: 200 };
  }

  async move(
    srcKey: string,
    dstKey: string,
    opts?: { overwrite?: boolean; metadata?: Dict<string> }
  ): Promise<{ ok: boolean; status?: number; error?: string }> {
    const res = await this.copy(srcKey, dstKey, opts);
    if (!res.ok) return res;
    await this.delete(srcKey);
    return { ok: true, status: 200 };
  }
}

// ----------------------------- Proxy wrapper --------------------------------
// Use this to adapt S3/GCS/Azure/etc. by providing functions. No imports needed.

export type StorageDriver = Partial<Storage>;

export class ProxyStorage implements Storage {
  private d: StorageDriver;
  constructor(driver: StorageDriver) {
    this.d = driver || {};
  }
  put(key: string, body: PutInput, opts?: PutOptions): Promise<HeadResponse> {
    if (!this.d.put) return Promise.resolve({ ok: false, status: 501, error: "put not implemented" });
    return this.d.put.call(this.d, key, body, opts);
  }
  head(key: string): Promise<HeadResponse> {
    if (!this.d.head) return Promise.resolve({ ok: false, status: 501, error: "head not implemented" });
    return this.d.head.call(this.d, key);
  }
  get(key: string): Promise<GetResponse> {
    if (!this.d.get) return Promise.resolve({ ok: false, status: 501, error: "get not implemented" });
    return this.d.get.call(this.d, key);
  }
  delete(key: string) {
    if (!this.d.delete) return Promise.resolve({ ok: false, status: 501, error: "delete not implemented" });
    return (this.d.delete as any).call(this.d, key);
  }
  list(opts?: ListOptions) {
    if (!this.d.list) return Promise.resolve({ ok: false, items: [], error: "list not implemented" });
    return this.d.list.call(this.d, opts);
  }
  exists(key: string) {
    if (this.d.exists) return this.d.exists.call(this.d, key);
    // default via head
    return this.head(key).then(r => r.ok);
  }
  signUrl(key: string, opts?: { method?: "GET" | "PUT"; ttlSec?: number; contentType?: string }) {
    if (!this.d.signUrl) return Promise.resolve({ url: `unsupported://${encodeURIComponent(key)}` });
    return this.d.signUrl.call(this.d, key, opts);
  }
  copy(srcKey: string, dstKey: string, opts?: { overwrite?: boolean; metadata?: Dict<string> }) {
    if (!this.d.copy) return Promise.resolve({ ok: false, status: 501, error: "copy not implemented" });
    return this.d.copy.call(this.d, srcKey, dstKey, opts);
  }
  move(srcKey: string, dstKey: string, opts?: { overwrite?: boolean; metadata?: Dict<string> }) {
    if (!this.d.move) return Promise.resolve({ ok: false, status: 501, error: "move not implemented" });
    return this.d.move.call(this.d, srcKey, dstKey, opts);
  }
}

// ------------------------------ Utilities -----------------------------------

export function normKey(p: string): string {
  if (!p) return "";
  let s = String(p);
  s = s.replace(/\\/g, "/");           // windows → POSIX
  s = s.replace(/\/{2,}/g, "/");       // collapse multiple slashes
  s = s.replace(/^\.\//, "");          // strip leading "./"
  // do not resolve ".." to keep semantics simple/safe
  return s;
}

export function nowIso(): string { return new Date().toISOString(); }

export function guessContentType(key: string): string {
  const k = key.toLowerCase();
  if (ends(k, ".json")) return "application/json";
  if (ends(k, ".csv"))  return "text/csv";
  if (ends(k, ".tsv"))  return "text/tab-separated-values";
  if (ends(k, ".parquet")) return "application/octet-stream";
  if (ends(k, ".yaml") || ends(k, ".yml")) return "application/yaml";
  if (ends(k, ".txt"))  return "text/plain; charset=utf-8";
  if (ends(k, ".md"))   return "text/markdown; charset=utf-8";
  if (ends(k, ".html") || ends(k, ".htm")) return "text/html; charset=utf-8";
  if (ends(k, ".js"))   return "application/javascript";
  if (ends(k, ".ts") || ends(k, ".tsx")) return "text/plain; charset=utf-8";
  if (ends(k, ".png"))  return "image/png";
  if (ends(k, ".jpg") || ends(k, ".jpeg")) return "image/jpeg";
  if (ends(k, ".webp")) return "image/webp";
  if (ends(k, ".gif"))  return "image/gif";
  if (ends(k, ".pdf"))  return "application/pdf";
  return "application/octet-stream";
}

function ends(s: string, suffix: string): boolean { return s.slice(-suffix.length) === suffix; }

export function toBytes(input: PutInput): Bytes {
  if (input == null) return new Uint8Array(0);
  if (input instanceof Uint8Array) return input;
  if (typeof ArrayBuffer !== "undefined" && input instanceof ArrayBuffer) return new Uint8Array(input);
  if (typeof input === "string") return utf8Encode(input);
  if (typeof input === "object") {
    // objects/arrays → JSON bytes (stable)
    return utf8Encode(stableStringify(input));
  }
  // fallback
  return utf8Encode(String(input));
}

export function utf8Encode(s: string): Bytes {
  if (typeof (globalThis as any).TextEncoder !== "undefined") {
    try { return new (globalThis as any).TextEncoder().encode(s); } catch { /* noop */ }
  }
  // very simple fallback (BMP only)
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

export function utf8Decode(b: Bytes | ArrayBuffer | null | undefined): string {
  if (!b) return "";
  const u8 = b instanceof Uint8Array ? b : new Uint8Array(b);
  if (typeof (globalThis as any).TextDecoder !== "undefined") {
    try { return new (globalThis as any).TextDecoder("utf-8", { fatal: false }).decode(u8); } catch { /* noop */ }
  }
  let s = "";
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return s;
}

export function shallowClone<T extends object>(o: T | undefined): T | undefined {
  if (!o) return o;
  const out: any = {};
  for (const k in o) out[k] = (o as any)[k];
  return out as T;
}

export function hashHex(bytes: Bytes): string {
  // Tiny fast non-crypto hash (FNV-1a 32-bit)
  let h = 0x811c9dc5 >>> 0;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = (h * 0x01000193) >>> 0;
  }
  return ("00000000" + h.toString(16)).slice(-8);
}

export function stableStringify(v: any): string {
  try { return JSON.stringify(sortKeysDeep(v)); } catch { return String(v); }
}

function sortKeysDeep(v: any): any {
  if (Array.isArray(v)) return v.map(sortKeysDeep);
  if (v && typeof v === "object") {
    const o: Dict = {};
    const keys = Object.keys(v).sort();
    for (let i = 0; i < keys.length; i++) o[keys[i]] = sortKeysDeep(v[keys[i]]);
    return o;
  }
  return v;
}

// -------------------------- JSON / text helpers -----------------------------

export async function putJSON(store: Storage, key: string, obj: any, opts?: PutOptions): Promise<HeadResponse> {
  const ct = (opts && opts.contentType) || "application/json";
  return store.put(key, stableStringify(obj), { ...(opts || {}), contentType: ct });
}

export async function getJSON<T = any>(store: Storage, key: string): Promise<{ ok: boolean; data?: T; error?: string; status?: number }> {
  const r = await store.get(key);
  if (!r.ok || !r.content) return { ok: false, error: r.error, status: r.status };
  try {
    return { ok: true, data: JSON.parse(utf8Decode(r.content)) as T, status: 200 };
  } catch (e: any) {
    return { ok: false, error: "JSON parse failed: " + stringifyError(e), status: 422 };
  }
}

export async function putText(store: Storage, key: string, text: string, opts?: PutOptions): Promise<HeadResponse> {
  const ct = (opts && opts.contentType) || "text/plain; charset=utf-8";
  return store.put(key, text, { ...(opts || {}), contentType: ct });
}

export async function getText(store: Storage, key: string): Promise<{ ok: boolean; text?: string; status?: number; error?: string }> {
  const r = await store.get(key);
  if (!r.ok || !r.content) return { ok: false, error: r.error, status: r.status };
  return { ok: true, text: utf8Decode(r.content), status: 200 };
}

// ------------------------------- CSV utils ----------------------------------

export type CsvOptions = {
  /** if rows are objects and headers omitted, we infer and sort keys */
  headers?: string[];
  /** newline; default "\n" */
  newline?: string;
  /** delimiter; default "," */
  delimiter?: string;
};

export function toCSV(rows: any[], opts?: CsvOptions): string {
  const nl = (opts && opts.newline) || "\n";
  const d = (opts && opts.delimiter) || ",";
  if (!Array.isArray(rows) || rows.length === 0) return "";
  if (Array.isArray(rows[0])) {
    // rows of arrays
    return (opts?.headers ? [opts.headers.join(d)] : [])
      .concat((rows as any[]).map(r => (r as any[]).map(csvCell).join(d)))
      .join(nl);
  }
  // rows of objects
  const headers = (opts && opts.headers) || inferHeadersFromObjects(rows);
  const lines: string[] = [headers.join(d)];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const cells: string[] = [];
    for (let j = 0; j < headers.length; j++) cells.push(csvCell(r[headers[j]]));
    lines.push(cells.join(d));
  }
  return lines.join(nl);
}

function csvCell(v: any): string {
  if (v == null) return "";
  const s = typeof v === "string" ? v : (typeof v === "number" || typeof v === "boolean") ? String(v) : JSON.stringify(v);
  // escape: wrap with quotes if contains comma, quote, newline
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function inferHeadersFromObjects(rows: any[]): string[] {
  const set: Dict<boolean> = Object.create(null);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    for (const k in r) set[k] = true;
  }
  return Object.keys(set).sort();
}

export async function putCSV(store: Storage, key: string, rows: any[], opts?: PutOptions & CsvOptions): Promise<HeadResponse> {
  const csv = toCSV(rows, opts);
  const ct = (opts && opts.contentType) || "text/csv";
  return store.put(key, csv, { ...(opts || {}), contentType: ct });
}

// ------------------------------ Retry helpers --------------------------------

export type RetryOpts = { retries?: number; baseMs?: number; maxMs?: number };

export async function withRetry<T>(fn: () => Promise<T>, opts?: RetryOpts): Promise<T> {
  const retries = Math.max(0, Math.trunc(opts?.retries ?? 3));
  const base = Math.max(1, Math.trunc(opts?.baseMs ?? 100));
  const max = Math.max(base, Math.trunc(opts?.maxMs ?? 2000));
  let attempt = 0;
  let lastErr: any;
  while (attempt <= retries) {
    try { return await fn(); } catch (e) {
      lastErr = e;
      if (attempt === retries) break;
      const backoff = Math.min(max, base * Math.pow(2, attempt)) + jitterMs(50);
      await sleep(backoff);
      attempt++;
    }
  }
  throw lastErr;
}

function sleep(ms: number): Promise<void> { return new Promise(res => setTimeout(res, ms)); }
function jitterMs(n: number): number { return Math.floor(Math.random() * n); }

export function stringifyError(e: any): string {
  if (!e) return "Unknown error";
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message || String(e);
  try { return JSON.stringify(e); } catch { return String(e); }
}

// ------------------------------- Examples ------------------------------------
/*
const store = new MemoryStorage();
await putJSON(store, "catalog/eq_px_eod/2025-01-03.json", { dt: "2025-01-03", rows: 123 });
const got = await getJSON(store, "catalog/eq_px_eod/2025-01-03.json");
console.log(got.ok, got.data);

await putCSV(store, "tmp/test.csv", [{ a: 1, b: "x" }, { a: 2, b: "y" }]);
const list = await store.list({ prefix: "tmp/" });
console.log(list.items.map(i => i.key));
*/