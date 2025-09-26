// core/cursor.ts
// Checkpoint cursors for backfills & incrementals (pure TS, zero imports).
// Features:
//  - read/write cursor value per (dataset, provider)
//  - full record read/write (with metadata)
//  - CAS (compare-and-swap) for safe concurrent updates
//  - optimistic lock with TTL (file-backed lock token)
//  - append-only history (.jsonl) + pruning
//  - optional in-memory backend for local/dev
//
// You provide a tiny TextIO (read/write/del/list). If not provided, uses memory.

export type CursorRecord = {
  dataset: string;
  provider: string;
  value: string;            // opaque cursor: ISO ts, seq id, etc.
  updated: string;          // ISO timestamp
  meta?: Record<string, any>;
};

export type LockInfo = {
  token: string;
  owner?: string;
  since: string;            // ISO created time
  ttlMs: number;            // lease length
};

export type TextIO = {
  read(key: string): Promise<string | null>;
  write(key: string, val: string): Promise<void>;
  del(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>; // return keys under prefix
};

export type CursorStoreOpts = {
  basePrefix?: string;      // default "checkpoints"
  io?: TextIO;              // storage backend
};

export class CursorStore {
  private base: string;
  private io: TextIO;

  constructor(opts: CursorStoreOpts = {}) {
    this.base = (opts.basePrefix || "checkpoints").replace(/\/+$/,"");
    this.io = opts.io || MemoryTextIO();
  }

  /* ───────────── Keys ───────────── */

  key(ds: string, provider: string) {
    return `${this.base}/${sanitize(provider)}/${sanitize(ds)}.cursor.json`;
  }
  histKey(ds: string, provider: string) {
    return `${this.base}/${sanitize(provider)}/${sanitize(ds)}.history.jsonl`;
  }
  lockKey(ds: string, provider: string) {
    return `${this.base}/${sanitize(provider)}/${sanitize(ds)}.lock.json`;
  }

  /* ───────────── Read / Write ───────────── */

  /** Read only the cursor value ("" if missing). */
  async readValue(dataset: string, provider: string): Promise<string> {
    const rec = await this.read(dataset, provider);
    return rec?.value || "";
  }

  /** Read full record (null if missing). */
  async read(dataset: string, provider: string): Promise<CursorRecord | null> {
    const raw = await this.io.read(this.key(dataset, provider));
    if (!raw) return null;
    const rec = safeJSON<CursorRecord>(raw);
    if (!rec || typeof rec.value !== "string") return null;
    return rec;
  }

  /** Write new cursor value (overwrites). */
  async writeValue(dataset: string, provider: string, value: string, meta?: Record<string, any>): Promise<void> {
    const rec: CursorRecord = {
      dataset, provider, value,
      updated: new Date().toISOString(),
      meta: meta && Object.keys(meta).length ? meta : undefined
    };
    await this.io.write(this.key(dataset, provider), JSON.stringify(rec));
    await this.appendHistory(dataset, provider, { type: "set", at: rec.updated, value, meta });
  }

  /** Write full record (overwrites). */
  async write(dataset: string, provider: string, record: Partial<CursorRecord> & { value: string }): Promise<void> {
    const rec: CursorRecord = {
      dataset,
      provider,
      value: record.value,
      updated: record.updated || new Date().toISOString(),
      meta: record.meta
    };
    await this.io.write(this.key(dataset, provider), JSON.stringify(rec));
    await this.appendHistory(dataset, provider, { type: "set", at: rec.updated, value: rec.value, meta: rec.meta });
  }

  /** Compare-and-swap by expected current value. Returns true if swapped. */
  async cas(dataset: string, provider: string, expect: string, next: string, meta?: Record<string, any>): Promise<boolean> {
    const cur = await this.read(dataset, provider);
    if ((cur?.value || "") !== (expect || "")) return false;
    await this.writeValue(dataset, provider, next, meta);
    return true;
  }

  /** Delete cursor + lock (does not delete history). */
  async clear(dataset: string, provider: string): Promise<void> {
    await this.io.del(this.key(dataset, provider));
    await this.io.del(this.lockKey(dataset, provider));
    await this.appendHistory(dataset, provider, { type: "clear", at: new Date().toISOString() });
  }

  /* ───────────── Locking ───────────── */

  /**
   * Acquire a lock (optimistic): writes a lock file with token & ttl.
   * If an unexpired lock exists with different token, throws.
   * Returns the token (use it to release).
   */
  async lock(dataset: string, provider: string, ttlMs = 60_000, owner?: string): Promise<string> {
    const lk = this.lockKey(dataset, provider);
    const now = Date.now();
    const existingRaw = await this.io.read(lk);
    const nowIso = new Date(now).toISOString();
    if (existingRaw) {
      const li = safeJSON<LockInfo>(existingRaw);
      if (li && li.token && !isExpired(li, now)) {
        throw err(423, "locked"); // 423 Locked
      }
    }
    const token = genToken();
    const lock: LockInfo = { token, owner, since: nowIso, ttlMs };
    await this.io.write(lk, JSON.stringify(lock));
    await this.appendHistory(dataset, provider, { type: "lock", at: nowIso, ttlMs, owner, token });
    return token;
  }

  /** Refresh an existing lock (same token). Throws if token mismatch. */
  async refreshLock(dataset: string, provider: string, token: string, ttlMs = 60_000): Promise<void> {
    const lk = this.lockKey(dataset, provider);
    const raw = await this.io.read(lk);
    if (!raw) throw err(409, "no_lock");
    const li = safeJSON<LockInfo>(raw) || ({} as LockInfo);
    if (li.token !== token) throw err(423, "locked_other");
    const now = new Date().toISOString();
    const next: LockInfo = { token, owner: li.owner, since: now, ttlMs };
    await this.io.write(lk, JSON.stringify(next));
    await this.appendHistory(dataset, provider, { type: "lock_refresh", at: now, ttlMs });
  }

  /** Release lock if token matches (idempotent). */
  async unlock(dataset: string, provider: string, token: string): Promise<void> {
    const lk = this.lockKey(dataset, provider);
    const raw = await this.io.read(lk);
    if (!raw) return;
    const li = safeJSON<LockInfo>(raw) || ({} as LockInfo);
    if (li.token && li.token !== token) throw err(423, "locked_other");
    await this.io.del(lk);
    await this.appendHistory(dataset, provider, { type: "unlock", at: new Date().toISOString() });
  }

  /**
   * Perform a critical section with lock.
   * Automatically unlocks on success/failure.
   */
  async withLock<T>(
    dataset: string,
    provider: string,
    ttlMs: number,
    fn: () => Promise<T> | T,
    owner?: string
  ): Promise<T> {
    const token = await this.lock(dataset, provider, ttlMs, owner);
    try {
      const res = await fn();
      await this.unlock(dataset, provider, token);
      return res;
    } catch (e) {
      try { await this.unlock(dataset, provider, token); } catch {}
      throw e;
    }
  }

  /* ───────────── History ───────────── */

  /** Append one line JSON event to history. */
  async appendHistory(dataset: string, provider: string, event: Record<string, any>): Promise<void> {
    const key = this.histKey(dataset, provider);
    const now = new Date().toISOString();
    const line = JSON.stringify({ ts: now, ...event }) + "\n";
    const prev = (await this.io.read(key)) || "";
    await this.io.write(key, prev + line);
  }

  /** Prune history to keep last `maxLines` entries (default 10k). */
  async pruneHistory(dataset: string, provider: string, maxLines = 10_000): Promise<void> {
    const key = this.histKey(dataset, provider);
    const raw = await this.io.read(key);
    if (!raw) return;
    const lines = raw.split(/\r?\n/).filter(Boolean);
    if (lines.length <= maxLines) return;
    const keep = lines.slice(-maxLines).join("\n") + "\n";
    await this.io.write(key, keep);
  }

  /* ───────────── Bulk helpers ───────────── */

  /** List all cursor keys under base prefix (provider optional). */
  async list(provider?: string): Promise<string[]> {
    const prefix = provider ? `${this.base}/${sanitize(provider)}/` : `${this.base}/`;
    return this.io.list(prefix);
  }
}

/* ───────────── In-memory TextIO (default) ───────────── */

export function MemoryTextIO(): TextIO {
  const mem = new Map<string, string>();
  return {
    async read(k) { return mem.has(k) ? (mem.get(k) as string) : null; },
    async write(k, v) { mem.set(k, String(v)); },
    async del(k) { mem.delete(k); },
    async list(prefix) {
      const out: string[] = [];
      for (const k of mem.keys()) if (k.startsWith(prefix)) out.push(k);
      return out;
    }
  };
}

/* ───────────── Utilities ───────────── */

function sanitize(s: string) {
  return String(s || "")
    .replace(/[^A-Za-z0-9._\-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "") || "na";
}

function safeJSON<T = any>(s: string): T {
  try { return JSON.parse(s) as T; } catch { return {} as any as T; }
}

function genToken(): string {
  // Non-crypto token; replace with secure RNG if needed.
  let x = 2166136261 >>> 0;
  const n = Date.now().toString() + Math.random().toString(36);
  for (let i = 0; i < n.length; i++) {
    x ^= n.charCodeAt(i);
    x = Math.imul(x, 16777619) >>> 0;
  }
  return ("t_" + x.toString(16) + "_" + (Math.floor(Math.random() * 1e9)).toString(36));
}

function isExpired(li: LockInfo, nowMs: number): boolean {
  const since = Date.parse(li.since || "");
  if (!Number.isFinite(since)) return true;
  return nowMs > since + (li.ttlMs || 0);
}

function err(code: number, reason: string) {
  const e: any = new Error(reason); e.code = code; e.reason = reason; return e;
}