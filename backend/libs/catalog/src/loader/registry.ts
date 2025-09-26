// loader/registry.ts
// Tiny, zero-dependency loader registry.
// It does NOT import parser/reader to keep things pure.
// Instead, you pass the functions from ./parser and ./reader via the context.
//
// Usage (example):
//   import { readAuto } from "./reader";   // <-- in YOUR codebase, not here
//   import { parse } from "./parser";
//   const reg = createRegistry({ read: readAuto, parse });
//   const result = await reg.load("data.csv");
//
// If no custom loader matches, a built-in GenericLoader will:
//   1) read bytes/text via ctx.read()
//   2) parse using ctx.parse()
//   3) return { ok, rows, columns, format, meta }

export type Dict<T = any> = { [k: string]: T };

// Mirror the minimal shapes from reader.ts and parser.ts (no imports here)
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

// Context you must provide when creating the registry.
export type RegistryContext = {
  // A function like reader.readAuto (bytes→maybe text)
  read: (src: any, opts?: ReadOptions) => Promise<ReadResult>;
  // A function like parser.parse
  parse: (input: string | Uint8Array | ArrayBuffer, opts?: ParseOptions) => ParseResult;
};

// What a Loader returns
export type LoadResult = {
  ok: boolean;
  id?: string;               // loader id used
  format?: ParseResult["format"];
  rows?: any[];
  columns?: Column[];
  meta?: Dict;
  error?: string;
};

// A Loader decides if it can handle a source and then loads it
export type Loader = {
  /** Unique id for the loader */
  id: string;
  /**
   * Return a match score: number in [0, 1]; 0 = no match, 1 = perfect match.
   * You receive the raw src and a small hint object.
   * Keep it cheap: string checks, file extensions, mime hints, etc.
   */
  match: (src: any, hint: { filename?: string; contentType?: string; sourceType?: string }) => number;
  /**
   * Execute the load. You get the registry context with read/parse fns,
   * plus the same hint. Return a LoadResult.
   */
  load: (src: any, ctx: RegistryContext, hint: { filename?: string; contentType?: string; sourceType?: string }, opts?: { read?: ReadOptions; parse?: ParseOptions }) => Promise<LoadResult>;
};

// Options for registry.load calls
export type LoadOptions = {
  read?: ReadOptions;
  parse?: ParseOptions;
};

// Registry interface
export type Registry = {
  context(): RegistryContext;
  list(): Loader[];
  register(loader: Loader): void;
  unregister(id: string): void;
  resolve(src: any, hint?: { filename?: string; contentType?: string; sourceType?: string }): Loader;
  load(src: any, opts?: LoadOptions): Promise<LoadResult>;
  loadAll(sources: any[], opts?: LoadOptions): Promise<LoadResult[]>;
};

/* ────────────────────────────────────────────────────────────────────────── *
 * Factory
 * ────────────────────────────────────────────────────────────────────────── */

export function createRegistry(ctx: RegistryContext): Registry {
  const loaders: Loader[] = [];

  // Built-in generic loader (fallback)
  const generic: Loader = {
    id: "generic/auto",
    match: (_src, _hint) => 0.0001, // very low score; used only when nothing else matches
    load: async (src, context, hint, opts) => {
      try {
        const readRes = await context.read(src, opts?.read);
        if (!readRes.ok) {
          return { ok: false, id: "generic/auto", error: readRes.error || "Read failed" };
        }
        const parseInput: string | Uint8Array =
          typeof readRes.text === "string" ? readRes.text : (readRes.bytes as Uint8Array);

        const parseRes = context.parse(parseInput, {
          ...(opts?.parse || {}),
          filename: (opts?.parse && opts.parse.filename) || readRes.filename || hint.filename
        });

        if (!parseRes || parseRes.ok !== true) {
          return {
            ok: false,
            id: "generic/auto",
            error: (parseRes && parseRes.error) || "Parse failed"
          };
        }
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

  // Default registry has the generic fallback registered last
  loaders.push(generic);

  function context(): RegistryContext { return ctx; }
  function list(): Loader[] { return loaders.slice(); }

  function register(loader: Loader): void {
    // ensure id unique; remove old
    const i = loaders.findIndex(l => l.id === loader.id);
    if (i >= 0) loaders.splice(i, 1);
    // Put BEFORE generic fallback (end of array is generic)
    const gi = loaders.findIndex(l => l.id === "generic/auto");
    const insertAt = gi >= 0 ? gi : loaders.length;
    loaders.splice(insertAt, 0, loader);
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
      } catch {
        // ignore faulty matchers
      }
    }
    if (scorePairs.length === 0) return generic;
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
    for (let i = 0; i < sources.length; i++) {
      out.push(await load(sources[i], opts));
    }
    return out;
  }

  return { context, list, register, unregister, resolve, load, loadAll };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Helper: quick hint (filename, contentType, sourceType) without heavy work
 * ────────────────────────────────────────────────────────────────────────── */

async function quickHint(ctx: RegistryContext, src: any, readOpts?: ReadOptions): Promise<{ filename?: string; contentType?: string; sourceType?: string }> {
  // Heuristics that are cheap. If src is a string URL/path, we can derive filename.
  try {
    if (typeof src === "string") {
      const s = src.trim();
      if (/^https?:\/\//i.test(s)) {
        return { filename: guessNameFromUrl(s), contentType: undefined, sourceType: "url" };
      }
      if (/^data:/i.test(s)) {
        // data URLs often embed mime and name
        const meta = s.slice(5, s.indexOf(",")); // after 'data:'
        const ct = meta.split(";")[0] || undefined;
        const nameMatch = meta.match(/name=([^;]+)/i);
        const filename = nameMatch ? safeUnquote(decodeURIComponent(nameMatch[1])) : undefined;
        return { filename, contentType: ct || undefined, sourceType: "dataurl" };
      }
      if (/^file:\/\//i.test(s)) {
        const filename = basename(fileUrlToPathSafe(s) || s);
        return { filename, contentType: undefined, sourceType: "path" };
      }
      // treat as path or literal — filename is last segment
      return { filename: basename(s), contentType: undefined, sourceType: "path" };
    }
    // For blobs/streams we could read headers, but keep it cheap → no IO.
    return {};
  } catch {
    return {};
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Convenience: build a couple of common loaders
 * (You can ignore these if the generic one is enough.)
 * ────────────────────────────────────────────────────────────────────────── */

/** Prefer YAML for files under meta/ or ending with .yaml/.yml */
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
      // Force YAML parse format but still rely on reader to fetch bytes/text
      const readRes = await ctx.read(src, opts?.read);
      if (!readRes.ok) return { ok: false, id: "prefer/yaml", error: readRes.error || "Read failed" };
      const input = typeof readRes.text === "string" ? readRes.text : (readRes.bytes as Uint8Array);
      const parseRes = ctx.parse(input, {
        ...(opts?.parse || {}),
        format: "yaml",
        filename: (opts?.parse && opts.parse.filename) || readRes.filename || hint.filename
      });
      if (!parseRes.ok) return { ok: false, id: "prefer/yaml", error: parseRes.error || "Parse failed" };
      return {
        ok: true,
        id: "prefer/yaml",
        format: parseRes.format,
        rows: parseRes.rows || [],
        columns: parseRes.columns || [],
        meta: { ...(parseRes.meta || {}), _source: { filename: readRes.filename || hint.filename, contentType: readRes.contentType } }
      };
    }
  };
}

/** Prefer CSV/TSV for files ending with .csv/.tsv/.txt where delimiter looks tab/comma */
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
      const readRes = await ctx.read(src, opts?.read);
      if (!readRes.ok) return { ok: false, id: "prefer/csv-tsv", error: readRes.error || "Read failed" };
      const input = typeof readRes.text === "string" ? readRes.text : (readRes.bytes as Uint8Array);
      const parseRes = ctx.parse(input, {
        ...(opts?.parse || {}),
        filename: (opts?.parse && opts.parse.filename) || readRes.filename || hint.filename
      });
      if (!parseRes.ok) return { ok: false, id: "prefer/csv-tsv", error: parseRes.error || "Parse failed" };
      return {
        ok: true,
        id: "prefer/csv-tsv",
        format: parseRes.format,
        rows: parseRes.rows || [],
        columns: parseRes.columns || [],
        meta: { ...(parseRes.meta || {}), _source: { filename: readRes.filename || hint.filename, contentType: readRes.contentType } }
      };
    }
  };
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Small shared helpers (no imports)
 * ────────────────────────────────────────────────────────────────────────── */

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }

function guessNameFromUrl(url: string): string | undefined {
  try {
    const u = new URL(url);
    const path = u.pathname || "";
    const last = path.split("/").filter(Boolean).pop();
    return last || undefined;
  } catch { return undefined; }
}
function safeUnquote(s: string): string {
  if (!s) return s;
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) return s.slice(1, -1);
  return s;
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
function basename(p: string): string {
  const s = String(p);
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  return i >= 0 ? s.slice(i + 1) : s;
}
function normalizeErr(e: any): string { return e instanceof Error ? (e.message || String(e)) : String(e); }

/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */