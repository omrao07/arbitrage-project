// core/fs.ts
// Minimal filesystem storage driver (text + bytes). Pure TypeScript, zero imports.
// Works in Node via built-in 'fs' and 'path' accessed through require().
//
// API:
//   const fs = new FS("lake");
//   await fs.writeText("raw/px-daily/dt=2025-09-26/symbol=AAPL.jsonl", "line\n");
//   const s = await fs.readText("raw/px-daily/dt=2025-09-26/symbol=AAPL.jsonl");
//   const list = await fs.list("raw/px-daily/");
//   await fs.writeBytes("bin/blob", new Uint8Array([1,2,3]));
//   await fs.del("bin/blob");
//
// Notes:
// - Keys always use forward slashes ('/'). They are joined under the provided root.
// - Writes are atomic (temp file + rename).
// - list() is recursive when prefix maps to a directory; otherwise it filters by prefix.
// - No external imports; safe for your “no imports” constraint.

export type ListEntry = {
  key: string;        // relative key (forward slashes)
  size: number;       // bytes
  modified?: string;  // ISO mtime
  isDir?: boolean;
};

export class FS {
  private _fs: any;
  private _path: any;
  private root: string;

  constructor(root: string = "lake") {
    // lazy bind Node built-ins via require (no import needed)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const req: any = (typeof require !== "undefined") ? require : null;
    if (!req) throw new Error("FS requires Node.js runtime");
    this._fs = req("fs");
    this._path = req("path");

    this.root = root;
    if (!this._fs.existsSync(this.root)) this._fs.mkdirSync(this.root, { recursive: true });
  }

  /** Normalize a storage key into an absolute filesystem path under root. */
  private pathOf(key: string): string {
    const norm = normalizeKey(key);
    return this._path.join(this.root, ...norm.split("/").filter(Boolean));
  }

  /** Ensure directory exists for a given key (file path). */
  private ensureDirFor(key: string): void {
    const p = this.pathOf(key);
    const dir = this._path.dirname(p);
    if (!this._fs.existsSync(dir)) this._fs.mkdirSync(dir, { recursive: true });
  }

  /** Exists check for a given key (file or directory). */
  exists(key: string): boolean {
    try { return this._fs.existsSync(this.pathOf(key)); } catch { return false; }
  }

  /** Read text file; returns null if missing. */
  async readText(key: string): Promise<string | null> {
    const p = this.pathOf(key);
    if (!this._fs.existsSync(p)) return null;
    return this._fs.readFileSync(p, "utf8");
  }

  /** Write text atomically. */
  async writeText(key: string, text: string): Promise<void> {
    this.ensureDirFor(key);
    const p = this.pathOf(key);
    atomicWrite(this._fs, p, BufferFrom(text, "utf8"));
  }

  /** Append text (creates file if missing). */
  async appendText(key: string, text: string): Promise<void> {
    this.ensureDirFor(key);
    const p = this.pathOf(key);
    this._fs.appendFileSync(p, BufferFrom(text, "utf8"));
  }

  /** Read bytes; returns null if missing. */
  async readBytes(key: string): Promise<Uint8Array | null> {
    const p = this.pathOf(key);
    if (!this._fs.existsSync(p)) return null;
    const buf = this._fs.readFileSync(p);
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  /** Write bytes atomically. */
  async writeBytes(key: string, data: Uint8Array): Promise<void> {
    this.ensureDirFor(key);
    const p = this.pathOf(key);
    atomicWrite(this._fs, p, BufferFrom(data));
  }

  /** Delete a file (idempotent). */
  async del(key: string): Promise<void> {
    const p = this.pathOf(key);
    if (this._fs.existsSync(p)) this._fs.unlinkSync(p);
  }

  /**
   * List files under a prefix (recursive if the prefix is a directory).
   * Returns entries with keys relative to the storage root.
   */
  async list(prefix: string = ""): Promise<ListEntry[]> {
    const rel = normalizeKey(prefix);
    const abs = this.pathOf(rel);
    const out: ListEntry[] = [];

    const pushEntry = (fp: string) => {
      const st = this._fs.statSync(fp);
      const key = relKey(this.root, fp, this._path);
      out.push({
        key,
        size: st.isDirectory() ? 0 : st.size,
        modified: st.mtime ? st.mtime.toISOString() : undefined,
        isDir: st.isDirectory()
      });
    };

    if (this._fs.existsSync(abs) && this._fs.statSync(abs).isDirectory()) {
      walkDir(abs, this._fs, this._path, (fp: string) => pushEntry(fp));
      // filter out the directory itself; only return nested
      return out.filter(e => e.key !== stripTrailingSlash(rel));
    }

    // If not a directory, walk the whole root and filter by prefix.
    walkDir(this.root, this._fs, this._path, (fp: string) => {
      const key = relKey(this.root, fp, this._path);
      if (key.startsWith(rel)) {
        const st = this._fs.statSync(fp);
        out.push({
          key,
          size: st.isDirectory() ? 0 : st.size,
          modified: st.mtime ? st.mtime.toISOString() : undefined,
          isDir: st.isDirectory()
        });
      }
    });
    return out;
  }

  /** Make a directory tree by key (useful for preparing partitions). */
  async ensureDir(key: string): Promise<void> {
    const p = this.pathOf(key);
    if (!this._fs.existsSync(p)) this._fs.mkdirSync(p, { recursive: true });
  }

  /** Read JSON (object) or null if missing/invalid. */
  async readJSON<T = any>(key: string): Promise<T | null> {
    const t = await this.readText(key);
    if (t == null) return null;
    try { return JSON.parse(t) as T; } catch { return null; }
  }

  /** Write JSON atomically with trailing newline. */
  async writeJSON(key: string, obj: any): Promise<void> {
    const txt = JSON.stringify(obj) + "\n";
    await this.writeText(key, txt);
  }
}

/* ───────────────────────── Helpers ───────────────────────── */

function normalizeKey(key: string): string {
  const s = String(key || "").replace(/\\/g, "/").replace(/^\/+/, "");
  return s;
}

function stripTrailingSlash(s: string): string {
  return s.endsWith("/") ? s.slice(0, -1) : s;
}

function walkDir(base: string, _fs: any, _path: any, cb: (absPath: string) => void) {
  const st = _fs.statSync(base);
  if (st.isFile()) { cb(base); return; }
  const stack: string[] = [base];
  while (stack.length) {
    const dir = stack.pop() as string;
    for (const name of _fs.readdirSync(dir)) {
      const p = _path.join(dir, name);
      const s = _fs.statSync(p);
      if (s.isDirectory()) { stack.push(p); }
      else { cb(p); }
    }
  }
}

function relKey(root: string, absPath: string, _path: any): string {
  const rel = _path.relative(root, absPath);
  return rel.split(_path.sep).join("/");
}

function atomicWrite(_fs: any, finalPath: string, buf: any) {
  const dir = require("path").dirname(finalPath);
  const tmp = require("path").join(dir, ".tmp-" + Date.now() + "-" + Math.random().toString(36).slice(2));
  _fs.writeFileSync(tmp, buf);
  _fs.renameSync(tmp, finalPath);
}

function BufferFrom(data: any, enc?: any): any {
  // Use Node's Buffer if available; otherwise build from Uint8Array
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const B: any = (typeof Buffer !== "undefined") ? Buffer : null;
  if (B && enc) return B.from(data, enc);
  if (B) return B.from(data);
  if (typeof data === "string") {
    // manual encode utf8
    const te = new TextEncoder(); return te.encode(data);
  }
  if (data instanceof Uint8Array) return data;
  return new Uint8Array(data);
}