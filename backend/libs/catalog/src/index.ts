// src/index.ts
// Pure, zero-import public entrypoint.
// It bundles a tiny loader registry + simple resolver + convenience helpers,
// all using dependency injection (you pass the reader/parser).
//
// Why DI? Because you asked for *no imports*. So you give us:
//   - ctx.read(src, opts?)  -> { ok, text?/bytes?, contentType?, filename?, ... }
//   - ctx.parse(input, opts?) -> { ok, format?, rows?, columns?, meta? }
//
// Then:
//   const reg = createRegistry({ read, parse });
//   reg.register(yamlPrefLoader());
//   reg.register(tabularPrefLoader());
//   const out = await reg.load("data.csv");
//
// Or end-to-end:
//   const results = await loadAny(["data.yaml", "https://host/file.csv"], { ctx: { read, parse } });

/* ────────────────────────────────────────────────────────────────────────── *
 * Types
 * ────────────────────────────────────────────────────────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type ReadOptions = {
  encoding?: "utf-8" | "utf8";
  maxBytes?: number;
  timeoutMs?: number;
  asTextHint?: boolean;
};
export type ReadResult = {
  ok: boolean;
  sourceType?: "url" | "path" | "dataurl" | "blob" | "bytes" | "text" | "stream";
  bytes?: Uint8Array;
  text?: string;
  contentType?: string;
  filename?: string;
  size?: number;
  meta?: Dict;
  error?: string;
};

export type ParseOptions = {
  format?: "auto" | "json" | "ndjson" | "csv" | "tsv" | "yaml";
  filename?: string;
  delimiter?: string;
  header?: boolean;
  comment?: string;
  skipEmpty?: boolean;
  autoType?: boolean;
  nulls?: string[];
  sampleSize?: number;
};
export type Column = { name: string; type: "string" | "bool" | "int" | "float" | "date" | "timestamp" | "mixed" };
export type ParseResult = {
  ok: boolean;
  format?: "json" | "ndjson" | "csv" | "tsv" | "yaml";
  rows?: any[];
  columns?: Column[];
  meta?: Dict;
  error?: string;
};

export type RegistryContext = {
  read: (src: any, opts?: ReadOptions) => Promise<ReadResult>;
  parse: (input: string | Uint8Array | ArrayBuffer, opts?: ParseOptions) => ParseResult;
};

export type LoadResult = {
  ok: boolean;
  id?: string;
  format?: ParseResult["format"];
  rows?: any[];
  columns?: Column[];
  meta?: Dict;
  error?: string;
};

export type Loader = {
  id: string;
  match: (src: any, hint: { filename?: string; contentType?: string; sourceType?: string }) => number;
  load: (
    src: any,
    ctx: RegistryContext,
    hint: { filename?: string; contentType?: string; sourceType?: string },
    opts?: { read?: ReadOptions; parse?: ParseOptions }
  ) => Promise<LoadResult>;
};

export type LoadOptions = { read?: ReadOptions; parse?: ParseOptions };
export type Registry = {
  context(): RegistryContext;
  list(): Loader[];
  register(loader: Loader): void;
  unregister(id: string): void;
  resolve(src: any, hint?: { filename?: string; contentType?: string; sourceType?: string }): Loader;
  load(src: any, opts?: LoadOptions): Promise<LoadResult>;
  loadAll(sources: any[], opts?: LoadOptions): Promise<LoadResult[]>;
};

export type ResolveOptions = {
  // If true, try to classify strings; else treat all strings as text.
  classifyStrings?: boolean; // default true
};

export type ResolvedSource = {
  src: any;
  hint: {
    filename?: string;
    contentType?: string;
    sourceType?: "url" | "dataurl" | "path" | "blob" | "bytes" | "text" | "stream";
    path?: string;
  };
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Registry (generic fallback + preference loaders)
 * ────────────────────────────────────────────────────────────────────────── */

export function createRegistry(ctx: RegistryContext): Registry {
  const loaders: Loader[] = [];

  const generic: Loader = {
    id: "generic/auto",
    match: () => 0.0001,
    load: async (src, context, hint, opts) => {
      try {
        const readRes = await context.read(src, opts?.read);
        if (!readRes.ok) return { ok: false, id: "generic/auto", error: readRes.error || "Read failed" };
        const input: string | Uint8Array =
          typeof readRes.text === "string" ? readRes.text : (readRes.bytes as Uint8Array);

        const parseRes = context.parse(input, {
          ...(opts?.parse || {}),
          filename: (opts?.parse && opts.parse.filename) || readRes.filename || hint.filename
        });

        if (!parseRes?.ok) return { ok: false, id: "generic/auto", error: parseRes?.error || "Parse failed" };

        return {
          ok: true,
          id: "generic/auto",
          format: parseRes.format,
          rows: parseRes.rows || [],
          columns: parseRes.columns || [],
          meta: {
            ...(parseRes.meta || {}),
            _source: {
              filename: readRes.filename || hint.filename,
              contentType: readRes.contentType || hint.contentType,
              sourceType: readRes.sourceType || hint.sourceType,
              size: readRes.size
            }
          }
        };
      } catch (e: any) {
        return { ok: false, id: "generic/auto", error: normalizeErr(e) };
      }
    }
  };

  loaders.push(generic);

  function context(): RegistryContext { return ctx; }
  function list(): Loader[] { return loaders.slice(); }

  function register(loader: Loader): void {
    const i = loaders.findIndex(l => l.id === loader.id);
    if (i >= 0) loaders.splice(i, 1);
    const gi = loaders.findIndex(l => l.id === "generic/auto");
    const at = gi >= 0 ? gi : loaders.length;
    loaders.splice(at, 0, loader);
  }

  function unregister(id: string): void {
    const i = loaders.findIndex(l => l.id === id);
    if (i >= 0) loaders.splice(i, 1);
  }

  function resolve(src: any, hint?: { filename?: string; contentType?: string; sourceType?: string }): Loader {
    const scorePairs: Array<{ s: number; l: Loader }> = [];
    for (let i = 0; i < loaders.length; i++) {
      try {
        const s = clamp01(loaders[i].match(src, hint || {}));
        if (s > 0) scorePairs.push({ s, l: loaders[i] });
      } catch {}
    }
    if (!scorePairs.length) return generic;
    scorePairs.sort((a, b) => b.s - a.s);
    return scorePairs[0].l;
  }

  async function load(src: any, opts?: LoadOptions): Promise<LoadResult> {
    const hint = await quickHint(ctx, src, opts?.read);
    const loader = resolve(src, hint);
    return loader.load(src, ctx, hint, { read: opts?.read, parse: opts?.parse });
  }

  async function loadAll(sources: any[], opts?: LoadOptions): Promise<LoadResult[]> {
    const out: LoadResult[] = [];
    for (let i = 0; i < sources.length; i++) out.push(await load(sources[i], opts));
    return out;
  }

  return { context, list, register, unregister, resolve, load, loadAll };
}

export function yamlPrefLoader(): Loader {
  return {
    id: "prefer/yaml",
    match: (_src, hint) => {
      const f = (hint.filename || "").toLowerCase();
      if (!f) return 0;
      if (f.endsWith(".yaml") || f.endsWith(".yml")) return 0.9;
      if (f.includes("/meta/") || f.startsWith("meta_") || f.startsWith("meta-")) return 0.6;
      return 0;
    },
    load: async (src, ctx, hint, opts) => {
      const r = await ctx.read(src, opts?.read);
      if (!r.ok) return { ok: false, id: "prefer/yaml", error: r.error || "Read failed" };
      const input = typeof r.text === "string" ? r.text : (r.bytes as Uint8Array);
      const p = ctx.parse(input, {
        ...(opts?.parse || {}),
        format: "yaml",
        filename: (opts?.parse && opts.parse.filename) || r.filename || hint.filename
      });
      if (!p.ok) return { ok: false, id: "prefer/yaml", error: p.error || "Parse failed" };
      return {
        ok: true,
        id: "prefer/yaml",
        format: p.format,
        rows: p.rows || [],
        columns: p.columns || [],
        meta: { ...(p.meta || {}), _source: { filename: r.filename || hint.filename, contentType: r.contentType } }
      };
    }
  };
}

export function tabularPrefLoader(): Loader {
  return {
    id: "prefer/csv-tsv",
    match: (_src, hint) => {
      const f = (hint.filename || "").toLowerCase();
      if (!f) return 0;
      if (f.endsWith(".csv")) return 0.8;
      if (f.endsWith(".tsv")) return 0.8;
      if (f.endsWith(".txt")) return 0.4;
      return 0;
    },
    load: async (src, ctx, hint, opts) => {
      const r = await ctx.read(src, opts?.read);
      if (!r.ok) return { ok: false, id: "prefer/csv-tsv", error: r.error || "Read failed" };
      const input = typeof r.text === "string" ? r.text : (r.bytes as Uint8Array);
      const p = ctx.parse(input, {
        ...(opts?.parse || {}),
        filename: (opts?.parse && opts.parse.filename) || r.filename || hint.filename
      });
      if (!p.ok) return { ok: false, id: "prefer/csv-tsv", error: p.error || "Parse failed" };
      return {
        ok: true,
        id: "prefer/csv-tsv",
        format: p.format,
        rows: p.rows || [],
        columns: p.columns || [],
        meta: { ...(p.meta || {}), _source: { filename: r.filename || hint.filename, contentType: r.contentType } }
      };
    }
  };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Minimal resolver (no FS/globs — classification only)
 * ────────────────────────────────────────────────────────────────────────── */

export async function resolve(
  inputs: any | any[],
  options?: ResolveOptions
): Promise<ResolvedSource[]> {
  const classify = options?.classifyStrings !== false;
  const list = Array.isArray(inputs) ? inputs.slice() : [inputs];
  const out: ResolvedSource[] = [];
  const seen = new Set<string>();

  for (let i = 0; i < list.length; i++) {
    const it = list[i];

    if (it instanceof Uint8Array || isArrayBuffer(it)) {
      const bytes = it instanceof Uint8Array ? it : new Uint8Array(it as ArrayBuffer);
      pushUnique(out, seen, { src: bytes, hint: { sourceType: "bytes", contentType: guessContentType(undefined, undefined, bytes) } });
      continue;
    }
    if (isBlob(it)) {
      pushUnique(out, seen, { src: it, hint: { sourceType: "blob", filename: (it as any).name, contentType: (it as any).type } });
      continue;
    }
    if (typeof it === "string") {
      if (!classify) {
        pushUnique(out, seen, { src: it, hint: { sourceType: "text", contentType: "text/plain; charset=utf-8" } });
        continue;
      }
      const s = it.trim();
      if (isHttpUrl(s)) {
        pushUnique(out, seen, { src: s, hint: { sourceType: "url", filename: guessNameFromUrl(s) } });
        continue;
      }
      if (isDataUrl(s)) {
        const meta = s.slice(5, s.indexOf(","));
        const ct = meta.split(";")[0] || undefined;
        const nameMatch = meta.match(/name=([^;]+)/i);
        const filename = nameMatch ? safeUnquote(decodeURIComponent(nameMatch[1])) : undefined;
        pushUnique(out, seen, { src: s, hint: { sourceType: "dataurl", filename, contentType: ct } });
        continue;
      }
      if (isFileUrl(s)) {
        const p = fileUrlToPathSafe(s);
        pushUnique(out, seen, { src: s, hint: { sourceType: "path", filename: basename(p || s) } });
        continue;
      }
      // Treat as OS path heuristic if contains separators or known extension
      if (/[\\/]/.test(s) || /\.[A-Za-z0-9]+$/.test(s)) {
        pushUnique(out, seen, { src: s, hint: { sourceType: "path", filename: basename(s) } });
        continue;
      }
      // Fallback: literal text
      pushUnique(out, seen, { src: s, hint: { sourceType: "text", contentType: "text/plain; charset=utf-8" } });
      continue;
    }

    // Fallback: stringify object
    pushUnique(out, seen, { src: safeString(it), hint: { sourceType: "text", contentType: "text/plain; charset=utf-8" } });
  }

  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * End-to-end helper (DI!)
 * ────────────────────────────────────────────────────────────────────────── */

export async function loadAny(
  inputs: any | any[],
  opts: {
    ctx: RegistryContext;           // REQUIRED (read/parse)
    registry?: Registry;            // optional custom registry
    resolve?: ResolveOptions;
    load?: LoadOptions;
    usePrefLoaders?: boolean;       // default true: yaml + csv/tsv
  }
): Promise<Array<LoadResult & { hint?: ResolvedSource["hint"] }>> {
  const sources = await resolve(inputs, opts?.resolve);
  const reg = opts.registry || createRegistry(opts.ctx);
  const usePrefs = opts.usePrefLoaders !== false;

  // Ensure preference loaders are present (avoid duplicates)
  if (usePrefs) {
    if (!reg.list().some(l => l.id === "prefer/yaml")) reg.register(yamlPrefLoader());
    if (!reg.list().some(l => l.id === "prefer/csv-tsv")) reg.register(tabularPrefLoader());
  }

  const out: Array<LoadResult & { hint?: ResolvedSource["hint"] }> = [];
  for (let i = 0; i < sources.length; i++) {
    const s = sources[i];
    const res = await reg.load(s.src, opts?.load);
    out.push({ ...res, hint: s.hint });
  }
  return out;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Tiny shared helpers
 * ────────────────────────────────────────────────────────────────────────── */

async function quickHint(_ctx: RegistryContext, src: any, _readOpts?: ReadOptions): Promise<{ filename?: string; contentType?: string; sourceType?: string }> {
  try {
    if (typeof src === "string") {
      const s = src.trim();
      if (isHttpUrl(s)) return { filename: guessNameFromUrl(s), sourceType: "url" };
      if (isDataUrl(s)) {
        const meta = s.slice(5, s.indexOf(","));
        const ct = meta.split(";")[0] || undefined;
        const nameMatch = meta.match(/name=([^;]+)/i);
        const filename = nameMatch ? safeUnquote(decodeURIComponent(nameMatch[1])) : undefined;
        return { filename, contentType: ct, sourceType: "dataurl" };
      }
      if (isFileUrl(s)) return { filename: basename(fileUrlToPathSafe(s) || s), sourceType: "path" };
      return { filename: basename(s), sourceType: "path" };
    }
    return {};
  } catch { return {}; }
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
function normalizeErr(e: any): string { return e instanceof Error ? (e.message || String(e)) : String(e); }

function isHttpUrl(s: string): boolean { return /^https?:\/\//i.test(s); }
function isDataUrl(s: string): boolean { return /^data:/i.test(s); }
function isFileUrl(s: string): boolean { return /^file:\/\//i.test(s); }

function isArrayBuffer(x: any): x is ArrayBuffer {
  return typeof ArrayBuffer !== "undefined" && x instanceof ArrayBuffer;
}
function isBlob(x: any): x is Blob {
  return typeof Blob !== "undefined" && x instanceof Blob;
}

function basename(p: string): string {
  const s = String(p).replace(/\\/g, "/");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}
function guessNameFromUrl(url: string): string | undefined {
  try { const u = new URL(url); return basename(u.pathname || ""); } catch { return undefined; }
}
function fileUrlToPathSafe(fileUrl: string): string | undefined {
  try {
    const u = new URL(fileUrl);
    if (u.protocol !== "file:") return undefined;
    let p = decodeURIComponent(u.pathname);
    if (/^\/[A-Za-z]:\//.test(p)) p = p.slice(1); // windows
    return p;
  } catch { return undefined; }
}
function safeUnquote(s: string): string {
  if (!s) return s;
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) return s.slice(1, -1);
  return s;
}
function safeString(v: any): string {
  try { return typeof v === "string" ? v : JSON.stringify(v); } catch { return String(v); }
}

function guessContentType(name?: string, headerCT?: string, bytes?: Uint8Array): string | undefined {
  if (headerCT) return headerCT;
  if (name) {
    const ext = (name.slice(name.lastIndexOf(".")) || "").toLowerCase();
    if (ext === ".csv") return "text/csv; charset=utf-8";
    if (ext === ".tsv") return "text/tab-separated-values; charset=utf-8";
    if (ext === ".json") return "application/json; charset=utf-8";
    if (ext === ".jsonl" || ext === ".ndjson") return "application/x-ndjson; charset=utf-8";
    if (ext === ".yaml" || ext === ".yml") return "application/yaml; charset=utf-8";
    if (ext === ".txt" || ext === ".log") return "text/plain; charset=utf-8";
    if (ext === ".pdf") return "application/pdf";
    if (ext === ".gz") return "application/gzip";
  }
  if (bytes && bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) return "application/gzip";
  if (bytes && bytes.length >= 4 && bytes[0] === 0x25 && bytes[1] === 0x50 && bytes[2] === 0x44 && bytes[3] === 0x46) return "application/pdf";
  return undefined;
}

function keyOf(r: ResolvedSource): string {
  const h = r.hint;
  if (h.sourceType === "path" && h.path) return "path:" + h.path;
  if (h.sourceType === "url") return "url:" + (r.src as string);
  if (h.sourceType === "dataurl") return "dataurl:" + (r.src as string).slice(0, 64);
  if (h.sourceType === "blob") return "blob:" + (h.filename || "");
  if (h.sourceType === "bytes") return "bytes:" + (h.filename || "") + ":" + (h.contentType || "");
  if (h.sourceType === "text") return "text:" + String(r.src).slice(0, 64);
  return JSON.stringify(h) + ":" + String(r.src).slice(0, 32);
}
function pushUnique(out: ResolvedSource[], seen: Set<string>, r: ResolvedSource): void {
  const k = keyOf(r);
  if (seen.has(k)) return;
  seen.add(k);
  out.push(r);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Version
 * ────────────────────────────────────────────────────────────────────────── */

export const VERSION = "0.2.0-pure";