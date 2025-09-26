// loader/reader.ts
// Universal, zero-dependency reader for text/bytes from many sources.
// Pure TypeScript (no imports). Works in Node 18+ and modern browsers.
//
// Supports sources:
//   • string:
//       - "https://..." | "http://..."  → fetched via global fetch or Node http/https
//       - "data:..."                     → data URL decoded locally
//       - "file:///..."                  → Node fs read (if available)
//       - other                          → treated as filesystem path (Node) or literal text (browser)
//   • Uint8Array / ArrayBuffer           → bytes as-is
//   • Blob / File (browser)              → bytes via arrayBuffer()
//   • ReadableStream (browser)           → fully read to bytes
//   • Node Readable stream               → fully read to bytes (using dynamic require('stream') semantics)
//
// Notes:
//   - No gzip/zip decompression (we only sniff & mark contentType).
//   - For unknown strings in browsers (no fs), we return them as literal text.
//   - For Node, unknown strings are attempted as paths first; if read fails, fall back to literal text.

export type Dict<T = any> = { [k: string]: T };

export type ReadOptions = {
  encoding?: "utf-8" | "utf8";     // only utf-8 supported
  maxBytes?: number;               // safety cap while streaming (default 200MB)
  timeoutMs?: number;              // only honored for fetch/http path
  asTextHint?: boolean;            // if true, prefer treating string as literal text (skip fs)
};

export type ReadResult = {
  ok: boolean;
  sourceType?: "url" | "path" | "dataurl" | "blob" | "bytes" | "text" | "stream";
  bytes?: Uint8Array;
  text?: string;                   // utf-8 if not binary/gzip
  contentType?: string;            // best-effort guess/sniff
  filename?: string;               // best-effort (from URL/path/DataURL name=)
  size?: number;                   // bytes length if known
  meta?: Dict;
  error?: string;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Public API
 * ────────────────────────────────────────────────────────────────────────── */

export async function readBytes(
  src: string | Uint8Array | ArrayBuffer | Blob | ReadableStream<any> | any,
  options?: ReadOptions
): Promise<ReadResult> {
  const opts = withDefaults(options);

  try {
    // Bytes-like
    if (src instanceof Uint8Array) {
      const ct = guessContentType(undefined, undefined, src);
      return done("bytes", src, undefined, ct, undefined);
    }
    if (isArrayBuffer(src)) {
      const u8 = new Uint8Array(src as ArrayBuffer);
      const ct = guessContentType(undefined, undefined, u8);
      return done("bytes", u8, undefined, ct, undefined);
    }

    // Browser Blob / File
    if (isBlob(src)) {
      const ab = await (src as Blob).arrayBuffer();
      const u8 = new Uint8Array(ab);
      const name = (src as any).name || undefined;
      const ct = guessContentType(name, (src as any).type || undefined, u8);
      return done("blob", u8, undefined, ct, name);
    }

    // ReadableStream (browser WHATWG)
    if (isReadableStream(src)) {
      const u8 = await readBrowserStream(src as ReadableStream, opts.maxBytes);
      const ct = guessContentType(undefined, undefined, u8);
      return done("stream", u8, undefined, ct, undefined);
    }

    // Node Readable stream
    if (isNodeReadable(src)) {
      const u8 = await readNodeStream(src, opts.maxBytes);
      const ct = guessContentType(undefined, undefined, u8);
      return done("stream", u8, undefined, ct, undefined);
    }

    // String sources
    if (typeof src === "string") {
      const s = src.trim();

      // Data URL
      if (isDataUrl(s)) {
        const d = decodeDataUrl(s);
        const ct = d.contentType || guessContentType(undefined, undefined, d.bytes);
        return done("dataurl", d.bytes, undefined, ct, d.filename);
      }

      // HTTP(S) URL
      if (isHttpUrl(s)) {
        const { bytes, contentType, filename } = await fetchUrlBytes(s, opts);
        const ct = contentType || guessContentType(filename, undefined, bytes);
        return done("url", bytes, undefined, ct, filename ?? guessNameFromUrl(s));
      }

      // file:// URL
      if (isFileUrl(s)) {
        const path = fileUrlToPathSafe(s);
        if (path) {
          const { bytes, err } = await readFileBytes(path);
          if (err) return fail(`FS read error: ${err}`);
          const ct = guessContentType(path, undefined, bytes!);
          return done("path", bytes!, undefined, ct, basename(path));
        }
      }

      // Plain string → try Node fs path (unless asTextHint)
      if (!opts.asTextHint && isNodeEnv()) {
        const { bytes, err } = await readFileBytes(s);
        if (!err && bytes) {
          const ct = guessContentType(s, undefined, bytes);
          return done("path", bytes, undefined, ct, basename(s));
        }
        // Fall back to literal text if fs failed
      }

      // Literal text
      const enc = opts.encoding;
      const u8 = encodeUtf8(s);
      const ct = guessContentType(undefined, "text/plain; charset=utf-8", u8);
      return done("text", u8, s, ct, undefined);
    }

    // Unknown → stringify
    const txt = safeString(src);
    const u8 = encodeUtf8(txt);
    return done("text", u8, txt, "text/plain; charset=utf-8", undefined);

  } catch (e: any) {
    return fail(normalizeErr(e));
  }
}

export async function readText(
  src: Parameters<typeof readBytes>[0],
  options?: ReadOptions
): Promise<ReadResult> {
  const res = await readBytes(src, options);
  if (!res.ok) return res;

  // If already has text, return
  if (typeof res.text === "string") return res;

  // Avoid decoding obvious binary/gzip
  if (looksGzip(res.bytes)) {
    return { ...res, text: undefined, contentType: res.contentType ?? "application/gzip" };
  }
  if (looksPdf(res.bytes)) {
    return { ...res, text: undefined, contentType: res.contentType ?? "application/pdf" };
  }

  const text = decodeUtf8(res.bytes!);
  return { ...res, text, contentType: res.contentType ?? "text/plain; charset=utf-8" };
}

/**
 * Convenience: read and return best text, falling back to bytes if not decodable.
 * Same as readText but guarantees either `text` or `bytes`.
 */
export async function readAuto(
  src: Parameters<typeof readBytes>[0],
  options?: ReadOptions
): Promise<ReadResult> {
  const res = await readText(src, options);
  if (res.ok && (res.text == null || res.text === "")) {
    // Keep bytes as-is; caller can decide what to do
  }
  return res;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals: environment checks
 * ────────────────────────────────────────────────────────────────────────── */

function isNodeEnv(): boolean {
  // Heuristic: presence of process.versions.node
  const g: any = globalThis as any;
  return !!(g && g.process && g.process.versions && g.process.versions.node);
}

function nodeRequireSafe(mod: string): any | undefined {
  try {
    const g: any = globalThis as any;
    const req = g.require || (g.module && g.module.require);
    if (typeof req === "function") return req(mod);
  } catch {}
  return undefined;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals: type guards
 * ────────────────────────────────────────────────────────────────────────── */

function isArrayBuffer(x: any): x is ArrayBuffer {
  return typeof ArrayBuffer !== "undefined" && x instanceof ArrayBuffer;
}
function isBlob(x: any): x is Blob {
  return typeof Blob !== "undefined" && x instanceof Blob;
}
function isReadableStream(x: any): x is ReadableStream<any> {
  return typeof ReadableStream !== "undefined" && x instanceof ReadableStream;
}
function isNodeReadable(x: any): boolean {
  // duck-typing on .pipe or .on('data')
  return !!(x && typeof x === "object" && (typeof x.pipe === "function" || typeof x.on === "function"));
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals: string classifiers
 * ────────────────────────────────────────────────────────────────────────── */

function isHttpUrl(s: string): boolean { return /^https?:\/\//i.test(s); }
function isDataUrl(s: string): boolean { return /^data:/i.test(s); }
function isFileUrl(s: string): boolean { return /^file:\/\//i.test(s); }

function guessNameFromUrl(url: string): string | undefined {
  try {
    const u = new URL(url);
    const path = u.pathname || "";
    const last = path.split("/").filter(Boolean).pop();
    return last || undefined;
  } catch { return undefined; }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals: URL / FS readers
 * ────────────────────────────────────────────────────────────────────────── */

async function fetchUrlBytes(url: string, opts: Required<Pick<ReadOptions, "maxBytes" | "timeoutMs">>): Promise<{ bytes: Uint8Array; contentType?: string; filename?: string }> {
  const g: any = globalThis as any;

  // Prefer global fetch if available
  if (typeof g.fetch === "function") {
    const ctrl = typeof g.AbortController === "function" ? new g.AbortController() : undefined;
    const timer = opts.timeoutMs > 0 ? setTimeout(() => ctrl?.abort(), opts.timeoutMs) : undefined;
    try {
      const r = await g.fetch(url, { signal: ctrl?.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const ct = r.headers && typeof r.headers.get === "function" ? r.headers.get("content-type") || undefined : undefined;
      const disp = r.headers && typeof r.headers.get === "function" ? r.headers.get("content-disposition") || "" : "";
      const fname = extractFilenameFromDisposition(disp) || guessNameFromUrl(url);
      const u8 = new Uint8Array(await r.arrayBuffer());
      if (opts.maxBytes && u8.byteLength > opts.maxBytes) throw new Error(`Response too large (${u8.byteLength} > ${opts.maxBytes})`);
      return { bytes: u8, contentType: ct || undefined, filename: fname || undefined };
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  // Node http/https fallback
  const isHttps = /^https:\/\//i.test(url);
  const mod = nodeRequireSafe(isHttps ? "https" : "http");
  if (!mod) throw new Error("No fetch/http available in this environment.");
  const { bytes, headers } = await new Promise<{ bytes: Uint8Array; headers: any }>((resolve, reject) => {
    const req = mod.get(url, (res: any) => {
      if ((res.statusCode || 200) >= 300 && (res.statusCode || 200) < 400 && res.headers && res.headers.location) {
        // follow one redirect
        mod.get(res.headers.location, (res2: any) => pipeAndCollect(res2, opts.maxBytes, resolve, reject));
      } else {
        pipeAndCollect(res, opts.maxBytes, resolve, reject);
      }
    });
    req.on("error", reject);
    if (opts.timeoutMs > 0) req.setTimeout(opts.timeoutMs, () => { req.destroy(new Error("Request timeout")); });
  });
  const ct = headers && (headers["content-type"] || headers["Content-Type"]);
  const disp = headers && (headers["content-disposition"] || headers["Content-Disposition"]) || "";
  const fname = extractFilenameFromDisposition(String(disp)) || guessNameFromUrl(url);
  return { bytes, contentType: ct ? String(ct) : undefined, filename: fname || undefined };
}

function pipeAndCollect(res: any, maxBytes: number, resolve: (x: any) => void, reject: (e: any) => void) {
  const chunks: Uint8Array[] = [];
  let total = 0;
  res.on("data", (c: Buffer) => {
    const u8 = new Uint8Array(c.buffer, c.byteOffset, c.byteLength);
    total += u8.byteLength;
    if (maxBytes && total > maxBytes) {
      res.destroy(new Error(`Response too large (${total} > ${maxBytes})`));
      return;
    }
    chunks.push(new Uint8Array(u8));
  });
  res.on("end", () => {
    resolve({ bytes: concat(chunks), headers: res.headers });
  });
  res.on("error", reject);
}

async function readFileBytes(path: string): Promise<{ bytes?: Uint8Array; err?: string }> {
  try {
    const fs = nodeRequireSafe("fs");
    if (!fs) return { err: "fs module not available" };
    const prom = fs.promises || nodeRequireSafe("fs/promises");
    const buf = await (prom.readFile ? prom.readFile(path) : new Promise((resolve, reject) => fs.readFile(path, (e: any, d: any) => e ? reject(e) : resolve(d))));
    const u8 = buf instanceof Uint8Array ? new Uint8Array(buf) : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    return { bytes: u8 };
  } catch (e: any) {
    return { err: normalizeErr(e) };
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals: data URL
 * ────────────────────────────────────────────────────────────────────────── */

function decodeDataUrl(dataUrl: string): { bytes: Uint8Array; contentType?: string; filename?: string } {
  // data:[<mediatype>][;base64],<data>
  const m = dataUrl.match(/^data:([^,]*?),(.*)$/i);
  if (!m) throw new Error("Invalid data URL");
  const meta = m[1] || "";
  const data = m[2] || "";
  const isB64 = /;base64/i.test(meta);
  const ctMatch = meta.match(/^([^;]+)/);
  const contentType = ctMatch ? ctMatch[1] : undefined;
  const nameMatch = meta.match(/name=([^;]+)/i);
  const filename = nameMatch ? safeUnquote(decodeURIComponent(nameMatch[1])) : undefined;

  let bytes: Uint8Array;
  if (isB64) {
    const b = data.replace(/\s/g, "");
    bytes = base64ToBytes(b);
  } else {
    bytes = encodeUtf8(decodeURIComponent(data));
  }
  return { bytes, contentType, filename };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals: streams
 * ────────────────────────────────────────────────────────────────────────── */

async function readBrowserStream(stream: ReadableStream, maxBytes: number): Promise<Uint8Array> {
  const reader = (stream as any).getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const u8 = value instanceof Uint8Array ? value : new Uint8Array(value);
    total += u8.byteLength;
    if (maxBytes && total > maxBytes) throw new Error(`Stream too large (${total} > ${maxBytes})`);
    chunks.push(u8);
  }
  return concat(chunks);
}

async function readNodeStream(stream: any, maxBytes: number): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  let total = 0;
  await new Promise<void>((resolve, reject) => {
    stream.on("data", (c: any) => {
      const u8 = c instanceof Uint8Array ? new Uint8Array(c) : new Uint8Array(c.buffer, c.byteOffset, c.byteLength);
      total += u8.byteLength;
      if (maxBytes && total > maxBytes) { stream.destroy(); reject(new Error(`Stream too large (${total} > ${maxBytes})`)); return; }
      chunks.push(u8);
    });
    stream.on("end", () => resolve());
    stream.on("error", (e: any) => reject(e));
  });
  return concat(chunks);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Internals: util & sniffers
 * ────────────────────────────────────────────────────────────────────────── */

function withDefaults(opts?: ReadOptions): Required<Pick<ReadOptions, "encoding" | "maxBytes" | "timeoutMs">> & ReadOptions {
  return {
    encoding: opts?.encoding || "utf-8",
    maxBytes: typeof opts?.maxBytes === "number" ? opts!.maxBytes! : 200 * 1024 * 1024, // 200MB
    timeoutMs: typeof opts?.timeoutMs === "number" ? opts!.timeoutMs! : 30_000,
    ...opts
  };
}

function encodeUtf8(s: string): Uint8Array {
  if (typeof (globalThis as any).TextEncoder !== "undefined") {
    try { return new (globalThis as any).TextEncoder().encode(s); } catch {}
  }
  // Fallback (latin-ish)
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

function decodeUtf8(u8: Uint8Array): string {
  if (typeof (globalThis as any).TextDecoder !== "undefined") {
    try { return new (globalThis as any).TextDecoder("utf-8", { fatal: false }).decode(u8); } catch {}
  }
  let s = "";
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return s;
}

function concat(chunks: Uint8Array[]): Uint8Array {
  let total = 0;
  for (let i = 0; i < chunks.length; i++) total += chunks[i].byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (let i = 0; i < chunks.length; i++) { out.set(chunks[i], off); off += chunks[i].byteLength; }
  return out;
}

function safeString(v: any): string {
  try { return typeof v === "string" ? v : JSON.stringify(v); } catch { return String(v); }
}

function looksGzip(bytes?: Uint8Array): boolean {
  return !!(bytes && bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b);
}
function looksPdf(bytes?: Uint8Array): boolean {
  return !!(bytes && bytes.length >= 4 && bytes[0] === 0x25 && bytes[1] === 0x50 && bytes[2] === 0x44 && bytes[3] === 0x46); // %PDF
}

function guessContentType(name?: string, headerCT?: string, bytes?: Uint8Array): string | undefined {
  if (headerCT) return headerCT;
  if (name) {
    const ext = name.toLowerCase().split(".").pop() || "";
    if (ext === "csv") return "text/csv; charset=utf-8";
    if (ext === "tsv") return "text/tab-separated-values; charset=utf-8";
    if (ext === "json") return "application/json; charset=utf-8";
    if (ext === "jsonl" || ext === "ndjson") return "application/x-ndjson; charset=utf-8";
    if (ext === "yaml" || ext === "yml") return "application/yaml; charset=utf-8";
    if (ext === "txt" || ext === "log") return "text/plain; charset=utf-8";
    if (ext === "pdf") return "application/pdf";
    if (ext === "gz") return "application/gzip";
  }
  if (bytes && looksGzip(bytes)) return "application/gzip";
  if (bytes && looksPdf(bytes)) return "application/pdf";
  return undefined;
}

function extractFilenameFromDisposition(disp: string): string | undefined {
  // content-disposition: attachment; filename="foo.csv"; filename*=UTF-8''foo.csv
  const star = disp.match(/filename\*\s*=\s*([^']*)''([^;]+)/i);
  if (star) {
    try { return decodeURIComponent(star[2]); } catch { return star[2]; }
  }
  const plain = disp.match(/filename\s*=\s*("?)([^";]+)\1/i);
  return plain ? plain[2] : undefined;
}

function safeUnquote(s: string): string {
  if (!s) return s;
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
    return s.slice(1, -1);
  }
  return s;
}

function fileUrlToPathSafe(fileUrl: string): string | undefined {
  try {
    const u = new URL(fileUrl);
    if (u.protocol !== "file:") return undefined;
    let p = decodeURIComponent(u.pathname);
    // Windows drive letters: file:///C:/path → /C:/path
    if (/^\/[A-Za-z]:\//.test(p)) p = p.slice(1);
    return p;
  } catch { return undefined; }
}

function basename(p: string): string {
  const s = String(p);
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  return i >= 0 ? s.slice(i + 1) : s;
}

function base64ToBytes(b64: string): Uint8Array {
  const g: any = globalThis as any;
  if (typeof g.atob === "function") {
    const bin = g.atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  // Node fallback
  try {
    const buf = (g.Buffer && typeof g.Buffer.from === "function") ? g.Buffer.from(b64, "base64") : undefined;
    if (buf) return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } catch {}
  // Manual (slow) fallback
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const lookup: number[] = []; for (let i = 0; i < chars.length; i++) lookup[chars.charCodeAt(i)] = i;
  let buffer = 0, bits = 0, out: number[] = [];
  for (let i = 0; i < b64.length; i++) {
    const c = b64.charCodeAt(i);
    if (c === 43 || c === 47 || (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122)) {
      buffer = (buffer << 6) | lookup[c]; bits += 6;
      if (bits >= 8) { bits -= 8; out.push((buffer >> bits) & 0xff); }
    }
  }
  return new Uint8Array(out);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Result builders
 * ────────────────────────────────────────────────────────────────────────── */

function done(sourceType: ReadResult["sourceType"], bytes: Uint8Array, text: string | undefined, contentType?: string, filename?: string): ReadResult {
  return { ok: true, sourceType, bytes, text, contentType, filename, size: bytes.byteLength };
}
function fail(error: string): ReadResult {
  return { ok: false, error };
}
function normalizeErr(e: any): string { return e instanceof Error ? (e.message || String(e)) : String(e); }

/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */