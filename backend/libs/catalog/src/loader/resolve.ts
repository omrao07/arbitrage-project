// loader/resolve.ts
// Resolve mixed "inputs" (paths, urls, globs, blobs, bytes) into a flat,
// de-duplicated list of sources with lightweight hints (filename, type).
// Zero-dependency; works in Node and browsers. No imports.

export type Dict<T = any> = { [k: string]: T };

export type ResolveOptions = {
  /** Base directory for relative paths/globs (Node only). Default: process.cwd() if available, else "/" */
  cwd?: string;
  /** Expand globs/directories on the filesystem (Node only). Default: true */
  expand?: boolean;
  /** Recurse into subdirectories when a directory is passed. Default: true */
  recurse?: boolean;
  /** Include filters (glob patterns) applied to file paths relative to cwd */
  include?: string[];
  /** Exclude filters (glob patterns) applied after includes */
  exclude?: string[];
  /** Only keep files with these extensions (e.g. [".csv",".yaml"]). Case-insensitive. */
  extensions?: string[];
  /** Follow symlinks when walking (Node only). Default: true */
  followSymlinks?: boolean;
  /** Safety cap when expanding directories/globs */
  maxFiles?: number;
};

export type ResolvedSource = {
  src: any;
  hint: {
    filename?: string;
    contentType?: string;
    sourceType?: "url" | "dataurl" | "path" | "blob" | "bytes" | "text" | "stream";
    path?: string; // absolute or normalized path when available
  };
};

/** Entry point: resolve one or many inputs into normalized sources. */
export async function resolve(
  inputs: any | any[],
  options?: ResolveOptions
): Promise<ResolvedSource[]> {
  const opts = withDefaults(options);
  const list = Array.isArray(inputs) ? inputs.slice() : [inputs];
  const out: ResolvedSource[] = [];
  const seen = new Set<string>(); // to dedupe by key

  for (let i = 0; i < list.length; i++) {
    const it = list[i];

    // Bytes-like
    if (it instanceof Uint8Array || isArrayBuffer(it)) {
      const bytes = it instanceof Uint8Array ? it : new Uint8Array(it as ArrayBuffer);
      const ct = guessContentType(undefined, undefined, bytes);
      pushUnique(out, seen, { src: bytes, hint: { filename: undefined, contentType: ct, sourceType: "bytes" } });
      continue;
    }

    // Blob / File
    if (isBlob(it)) {
      const name = (it as any).name || undefined;
      const ct = (it as any).type || undefined;
      pushUnique(out, seen, { src: it, hint: { filename: name, contentType: ct, sourceType: "blob" } });
      continue;
    }

    // Streams (WHATWG or Node)
    if (isReadableStream(it) || isNodeReadable(it)) {
      pushUnique(out, seen, { src: it, hint: { sourceType: "stream" } });
      continue;
    }

    // Strings: URLs, data URLs, paths, globs, or literal text (browser)
    if (typeof it === "string") {
      const s = it.trim();

      // URL
      if (isHttpUrl(s)) {
        const filename = guessNameFromUrl(s);
        pushUnique(out, seen, {
          src: s,
          hint: { filename, contentType: undefined, sourceType: "url" }
        });
        continue;
      }

      // data URL
      if (isDataUrl(s)) {
        const meta = s.slice(5, s.indexOf(",")); // after 'data:'
        const ct = meta.split(";")[0] || undefined;
        const nameMatch = meta.match(/name=([^;]+)/i);
        const filename = nameMatch ? safeUnquote(decodeURIComponent(nameMatch[1])) : undefined;
        pushUnique(out, seen, { src: s, hint: { filename, contentType: ct, sourceType: "dataurl" } });
        continue;
      }

      // file:// URL → path (Node)
      if (isFileUrl(s)) {
        const p = fileUrlToPathSafe(s);
        if (p && isNodeEnv()) {
          await expandPathLike(p, out, seen, opts);
          continue;
        }
        // In browser, treat as text fallback
      }

      // Path or glob (Node), literal text (browser/no-fs)
      if (isNodeEnv() && opts.expand !== false) {
        const looksLikeGlob = hasGlobMeta(s);
        const baseCwd = opts.cwd || nodeCwd() || "/";
        const abs = isAbsPath(s) ? s : joinPath(baseCwd, s);
        if (looksLikeGlob) {
          const files = await expandGlob(abs, opts);
          for (let f of files) await expandPathLike(f, out, seen, opts);
          continue;
        }
        // Not a glob: expand file/dir if exists; else treat as text
        const fs = nodeRequireSafe("fs");
        const path = await statSafe(abs);
        if (path?.isFile) {
          await expandPathLike(abs, out, seen, opts);
          continue;
        } else if (path?.isDir) {
          const files = await walkDir(abs, opts);
          for (let f of files) await expandPathLike(f, out, seen, opts);
          continue;
        }
      }

      // Literal text
      const ct = "text/plain; charset=utf-8";
      pushUnique(out, seen, { src: s, hint: { filename: undefined, contentType: ct, sourceType: "text" } });
      continue;
    }

    // Fallback: unknown object → stringify as text
    const txt = safeString(it);
    pushUnique(out, seen, { src: txt, hint: { sourceType: "text", contentType: "text/plain; charset=utf-8" } });
  }

  // Post-filter on extensions/include/exclude
  const filtered = out.filter((r) => {
    const p = (r.hint.path || r.hint.filename || "").toLowerCase();
    // extensions
    if (opts.extensions && opts.extensions.length) {
      const okExt = opts.extensions.some((e) => p.endsWith(e.toLowerCase()));
      // If we don't have a path/filename (e.g. bytes/blob/url without name), keep it
      if (p && !okExt) return false;
    }
    // include
    if (opts.include && opts.include.length) {
      const inc = opts.include.some((g) => globRe(g).test(relFromCwd(p, opts)));
      if (!inc && p) return false;
    }
    // exclude
    if (opts.exclude && opts.exclude.length) {
      const exc = opts.exclude.some((g) => globRe(g).test(relFromCwd(p, opts)));
      if (exc) return false;
    }
    return true;
  });

  return filtered;
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Node-only expansion helpers (guarded)
 * ────────────────────────────────────────────────────────────────────────── */

async function expandPathLike(absPath: string, out: ResolvedSource[], seen: Set<string>, opts: RequiredOpts): Promise<void> {
  const filename = basename(absPath);
  const ct = guessContentType(filename);
  pushUnique(out, seen, {
    src: absPath,
    hint: { filename, contentType: ct, sourceType: "path", path: normalizePath(absPath) }
  });
}

async function expandGlob(absPattern: string, opts: RequiredOpts): Promise<string[]> {
  // Very small glob expansion: we walk from the nearest parent directory
  // and match with a generated RegExp. Supports **, *, ?, and character classes are NOT supported.
  const root = globBase(absPattern);
  const re = globRe(absPattern);
  const files = await walkDir(root, opts);
  return files.filter((f) => re.test(normalizePath(f)));
}

async function walkDir(rootAbs: string, opts: RequiredOpts): Promise<string[]> {
  const fs = nodeRequireSafe("fs");
  const fsp = (fs && (fs.promises || nodeRequireSafe("fs/promises"))) || undefined;
  if (!fs || !fsp) return [];
  const out: string[] = [];
  const stack: string[] = [rootAbs];
  const follow = opts.followSymlinks !== false;
  const limit = Math.max(0, opts.maxFiles || 0);

  while (stack.length) {
    const dir = stack.pop()!;
    let names: string[] = [];
    try {
      names = await fsp.readdir(dir);
    } catch { continue; }

    for (let i = 0; i < names.length; i++) {
      const full = joinPath(dir, names[i]);
      let st: any;
      try {
        st = await fsp.lstat(full);
      } catch { continue; }

      if (st.isSymbolicLink()) {
        if (!follow) continue;
        try { st = await fsp.stat(full); } catch { continue; }
      }
      if (st.isDirectory()) {
        if (opts.recurse !== false) stack.push(full);
        continue;
      }
      if (st.isFile()) {
        out.push(full);
        if (limit && out.length >= limit) return out;
      }
    }
  }
  return out;
}

async function statSafe(absPath: string): Promise<{ isFile: boolean; isDir: boolean } | null> {
  const fs = nodeRequireSafe("fs");
  const fsp = (fs && (fs.promises || nodeRequireSafe("fs/promises"))) || undefined;
  if (!fs || !fsp) return null;
  try {
    const st = await fsp.lstat(absPath);
    if (st.isSymbolicLink()) {
      try {
        const st2 = await fsp.stat(absPath);
        return { isFile: st2.isFile(), isDir: st2.isDirectory() };
      } catch { return null; }
    }
    return { isFile: st.isFile(), isDir: st.isDirectory() };
  } catch {
    return null;
  }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Glob basics
 * ────────────────────────────────────────────────────────────────────────── */

function hasGlobMeta(p: string): boolean {
  return /[*?[\]{}]/.test(p) || p.includes("**");
}

function globBase(patternAbs: string): string {
  // take everything up to the first glob meta as base
  const parts = splitPath(patternAbs);
  const out: string[] = [];
  for (let i = 0; i < parts.length; i++) {
    const seg = parts[i];
    if (hasGlobMeta(seg)) break;
    out.push(seg);
  }
  const base = joinParts(out);
  return base || "/";
}

function globRe(pattern: string): RegExp {
  const norm = normalizePath(pattern);
  const esc = (s: string) => s.replace(/[-/\\^$+?.()|[\]{}]/g, "\\$&"); // escape regex meta (leave * and ? custom)
  const parts = norm.split("/");
  const reParts: string[] = [];
  for (let i = 0; i < parts.length; i++) {
    const seg = parts[i];
    if (seg === "**") { reParts.push("(?:.*)"); continue; }
    let r = "";
    for (let j = 0; j < seg.length; j++) {
      const c = seg[j];
      if (c === "*") r += "[^/]*";
      else if (c === "?") r += "[^/]";
      else r += esc(c);
    }
    reParts.push(r);
  }
  const full = "^" + reParts.join("/") + "$";
  return new RegExp(full, "i");
}

/* ────────────────────────────────────────────────────────────────────────── *
 * Environment + tiny path utilities (POSIX-ish)
 * ────────────────────────────────────────────────────────────────────────── */

type RequiredOpts = Required<ResolveOptions>;
function withDefaults(opts?: ResolveOptions): RequiredOpts {
  // Default cwd to process.cwd() if available; else "/"
  // Other defaults: expand=true, recurse=true, followSymlinks=true, maxFiles=50000
  const cwd = opts?.cwd ?? nodeCwd() ?? "/";
  return {
    cwd,
    expand: opts?.expand !== false,
    recurse: opts?.recurse !== false,
    include: opts?.include || [],
    exclude: opts?.exclude || [],
    extensions: opts?.extensions || [],
    followSymlinks: opts?.followSymlinks !== false,
    maxFiles: typeof opts?.maxFiles === "number" ? opts!.maxFiles! : 50_000
  };
}

function isNodeEnv(): boolean {
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
function nodeCwd(): string | undefined {
  try { const g: any = globalThis as any; return g.process && typeof g.process.cwd === "function" ? g.process.cwd() : undefined; } catch { return undefined; }
}

function isHttpUrl(s: string): boolean { return /^https?:\/\//i.test(s); }
function isDataUrl(s: string): boolean { return /^data:/i.test(s); }
function isFileUrl(s: string): boolean { return /^file:\/\//i.test(s); }

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
  return !!(x && typeof x === "object" && (typeof x.pipe === "function" || typeof x.on === "function"));
}

function guessNameFromUrl(url: string): string | undefined {
  try {
    const u = new URL(url);
    const path = u.pathname || "";
    const last = path.split("/").filter(Boolean).pop();
    return last || undefined;
  } catch { return undefined; }
}

function fileUrlToPathSafe(fileUrl: string): string | undefined {
  try {
    const u = new URL(fileUrl);
    if (u.protocol !== "file:") return undefined;
    let p = decodeURIComponent(u.pathname);
    if (/^\/[A-Za-z]:\//.test(p)) p = p.slice(1); // windows drive letters
    return p;
  } catch { return undefined; }
}

function isAbsPath(p: string): boolean {
  return /^([A-Za-z]:[\\/]|\/)/.test(p);
}
function splitPath(p: string): string[] {
  return normalizePath(p).split("/").filter((s) => s.length > 0);
}
function joinParts(parts: string[]): string {
  let s = parts.join("/");
  if (!s.startsWith("/")) s = "/" + s;
  return s;
}
function normalizePath(p: string): string {
  let s = p.replace(/\\/g, "/");
  // collapse // -> /
  s = s.replace(/\/{2,}/g, "/");
  // resolve ./ and ../
  const segs = s.split("/");
  const out: string[] = [];
  for (let i = 0; i < segs.length; i++) {
    const g = segs[i];
    if (!g || g === ".") continue;
    if (g === "..") { out.pop(); continue; }
    out.push(g);
  }
  if (s.startsWith("/")) return "/" + out.join("/");
  return out.join("/") || ".";
}
function joinPath(a: string, b: string): string {
  if (isAbsPath(b)) return normalizePath(b);
  return normalizePath((a.endsWith("/") ? a : a + "/") + b);
}
function basename(p: string): string {
  const s = p.replace(/\\/g, "/");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}
function relFromCwd(p: string, opts: RequiredOpts): string {
  const base = normalizePath(opts.cwd);
  const full = normalizePath(p);
  if (!p || !base || !full.startsWith(base)) return p || "";
  const rel = full.slice(base.length);
  return rel.startsWith("/") ? rel.slice(1) : rel;
}
function extname(p: string): string {
  const b = basename(p);
  const i = b.lastIndexOf(".");
  return i >= 0 ? b.slice(i) : "";
}

function guessContentType(name?: string, headerCT?: string, bytes?: Uint8Array): string | undefined {
  if (headerCT) return headerCT;
  if (name) {
    const ext = (extname(name) || "").toLowerCase();
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

function safeUnquote(s: string): string {
  if (!s) return s;
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) return s.slice(1, -1);
  return s;
}
function safeString(v: any): string {
  try { return typeof v === "string" ? v : JSON.stringify(v); } catch { return String(v); }
}

/* ────────────────────────────────────────────────────────────────────────── *
 * De-dupe management
 * ────────────────────────────────────────────────────────────────────────── */

function keyOf(r: ResolvedSource): string {
  const h = r.hint;
  if (h.sourceType === "path" && h.path) return "path:" + h.path;
  if (h.sourceType === "url") return "url:" + (r.src as string);
  if (h.sourceType === "dataurl") return "dataurl:" + (r.src as string).slice(0, 64); // prefix
  if (h.sourceType === "blob") return "blob:" + (h.filename || "");
  if (h.sourceType === "bytes") return "bytes:" + (h.filename || "") + ":" + (h.contentType || "");
  if (h.sourceType === "text") return "text:" + String(r.src).slice(0, 64);
  if (h.sourceType === "stream") return "stream:" + (h.filename || "");
  return JSON.stringify(h) + ":" + String(r.src).slice(0, 32);
}

function pushUnique(out: ResolvedSource[], seen: Set<string>, r: ResolvedSource): void {
  const k = keyOf(r);
  if (seen.has(k)) return;
  seen.add(k);
  out.push(r);
}

/* ────────────────────────────────────────────────────────────────────────── *
 * END
 * ────────────────────────────────────────────────────────────────────────── */