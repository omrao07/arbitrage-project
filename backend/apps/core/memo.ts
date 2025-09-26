// core/memo.ts
// Zero-import cache + memoization utilities.
// Features:
//  • LRU cache with TTL and optional SWR (stale-while-revalidate)
//  • Async in-flight deduping (request coalescing)
//  • Namespaces + tags + bulk invalidation
//  • Size limits (entries/bytes) + stats
//  • Flexible memoize() wrapper for sync/async functions
//  • Stable key serializer (order-independent JSON)
// Pure TypeScript; safe to drop anywhere.

/* ───────────────────────── Types ───────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type CacheEntry<T = any> = {
  v: T;
  /** absolute expiry (ms epoch). If now > exp → expired (miss) */
  exp: number;
  /** absolute staleness (ms epoch). If now > exp but ≤ swr, return stale */
  swr?: number;
  /** approximate size in bytes */
  bytes: number;
  /** optional tags for group invalidation */
  tags?: string[];
};

export type CacheStats = {
  hits: number;
  stale: number;
  misses: number;
  sets: number;
  evictions: number;
  del: number;
  entries: number;
  bytes: number;
};

export type SetOpts = {
  ttlMs?: number;          // default 5 min
  swrMs?: number;          // default 0 (disabled)
  tags?: string[];
};

export type MemoOpts<Args extends any[] = any[], Ret = any> = {
  key?: (...args: Args) => string;           // custom key builder
  ttlMs?: number;                            // default 5 min
  swrMs?: number;                            // return stale while background refresh
  dedupe?: boolean;                          // coalesce concurrent calls (default true)
  cache?: LRUCache<Ret>;
  onError?: (e: any, key: string) => void;
  /** If true, when stale is served we trigger a background refresh (default true if swrMs>0) */
  refreshOnStale?: boolean;
};

/* ───────────────────────── LRU Cache ───────────────────────── */

export class LRUCache<V = any> {
  private maxEntries: number;
  private maxBytes: number;
  private map = new Map<string, Node<V>>();
  private head: Node<V> | null = null;
  private tail: Node<V> | null = null;
  private _bytes = 0;
  private _stats: CacheStats = { hits:0, stale:0, misses:0, sets:0, evictions:0, del:0, entries:0, bytes:0 };
  private tagIx = new Map<string, Set<string>>(); // tag -> keys

  constructor(opts?: { maxEntries?: number; maxBytes?: number }) {
    this.maxEntries = Math.max(1, opts?.maxEntries ?? 5_000);
    this.maxBytes   = Math.max(0, opts?.maxBytes   ?? 0); // 0 = unlimited
  }

  get stats(): CacheStats {
    return { ...this._stats, entries: this.map.size, bytes: this._bytes };
  }

  /** Get value; respects TTL/SWR. Returns undefined if expired beyond SWR. */
  get(key: string): V | undefined {
    const n = this.map.get(key);
    if (!n) { this._stats.misses++; return undefined; }
    const now = Date.now();
    if (now > n.e.exp) {
      if (n.e.swr && now <= n.e.swr) {
        // stale hit (SWR window)
        this.touch(n);
        this._stats.stale++;
        return n.e.v;
      }
      // hard expired → delete
      this.unlink(n);
      this.map.delete(key);
      this._bytes -= n.e.bytes;
      this._stats.misses++;
      return undefined;
    }
    // fresh
    this.touch(n);
    this._stats.hits++;
    return n.e.v;
  }

  /** Get with metadata (useful for memo to know staleness). */
  getMeta(key: string): { value?: V; stale: boolean; hit: boolean } {
    const n = this.map.get(key);
    if (!n) return { hit: false, stale: false, value: undefined };
    const now = Date.now();
    if (now <= n.e.exp) {
      this.touch(n);
      this._stats.hits++;
      return { hit: true, stale: false, value: n.e.v };
    }
    if (n.e.swr && now <= n.e.swr) {
      this.touch(n);
      this._stats.stale++;
      return { hit: true, stale: true, value: n.e.v };
    }
    // hard expired
    this.unlink(n);
    this.map.delete(key);
    this._bytes -= n.e.bytes;
    this._stats.misses++;
    return { hit: false, stale: false, value: undefined };
  }

  /** Set value with TTL/SWR and optional tags. */
  set(key: string, val: V, opts?: SetOpts): void {
    const ttl = Math.max(1, opts?.ttlMs ?? 5 * 60_000);
    const swr = Math.max(0, opts?.swrMs ?? 0);
    const now = Date.now();
    const entry: CacheEntry<V> = {
      v: val,
      exp: now + ttl,
      swr: swr ? now + ttl + swr : undefined,
      bytes: estimateSize(val) + key.length + 32,
      tags: opts?.tags?.slice()
    };

    let n = this.map.get(key);
    if (n) {
      // update
      this._bytes -= n.e.bytes;
      n.e = entry;
      this._bytes += n.e.bytes;
      this.touch(n);
    } else {
      // insert
      n = { k: key, e: entry, prev: null, next: null };
      this.linkHead(n);
      this.map.set(key, n);
      this._bytes += n.e.bytes;
    }
    this._stats.sets++;

    // tag index
    if (entry.tags) {
      for (const t of entry.tags) {
        if (!this.tagIx.has(t)) this.tagIx.set(t, new Set());
        this.tagIx.get(t)!.add(key);
      }
    }

    // enforce limits
    this.evictIfNeeded();
  }

  has(key: string): boolean {
    return this.map.has(key);
  }

  /** Delete a key; returns true if existed. */
  del(key: string): boolean {
    const n = this.map.get(key);
    if (!n) return false;
    this.unlink(n);
    this.map.delete(key);
    this._bytes -= n.e.bytes;
    this._stats.del++;
    // tags
    if (n.e.tags) for (const t of n.e.tags) this.tagIx.get(t)?.delete(key);
    return true;
  }

  /** Invalidate all entries with any of the given tags. */
  invalidateByTags(tags: string[]): number {
    let count = 0;
    for (const t of tags) {
      const set = this.tagIx.get(t);
      if (!set) continue;
      for (const k of set) if (this.del(k)) count++;
      this.tagIx.delete(t);
    }
    return count;
  }

  clear(): void {
    this.map.clear();
    this.head = this.tail = null;
    this._bytes = 0;
    this.tagIx.clear();
  }

  /* ── LRU internals ── */
  private touch(n: Node<V>) {
    if (this.head === n) return;
    this.unlink(n);
    this.linkHead(n);
  }
  private linkHead(n: Node<V>) {
    n.prev = null; n.next = this.head;
    if (this.head) this.head.prev = n;
    this.head = n;
    if (!this.tail) this.tail = n;
  }
  private unlink(n: Node<V>) {
    if (n.prev) n.prev.next = n.next; else this.head = n.next;
    if (n.next) n.next.prev = n.prev; else this.tail = n.prev;
    n.prev = n.next = null;
  }
  private evictIfNeeded() {
    // evict by count
    while (this.map.size > this.maxEntries) this.evictOne();
    // evict by bytes
    if (this.maxBytes > 0) {
      while (this._bytes > this.maxBytes && this.tail) this.evictOne();
    }
  }
  private evictOne() {
    const n = this.tail;
    if (!n) return;
    this.del(n.k);
    this._stats.evictions++;
  }
}

type Node<V> = { k: string; e: CacheEntry<V>; prev: Node<V> | null; next: Node<V> | null };

/* ───────────────────────── Memoization ───────────────────────── */

/**
 * Memoize a sync or async function. Works with TTL and SWR, and coalesces concurrent calls.
 * Example:
 *   const cache = new LRUCache<any>({ maxEntries: 2000 });
 *   const getPx = memoize(async (sym) => fetchPx(sym), { ttlMs: 60_000, swrMs: 30_000, cache });
 *   await getPx("AAPL");
 */
export function memoize<Args extends any[], Ret>(
  fn: (...args: Args) => Ret | Promise<Ret>,
  opts?: MemoOpts<Args, Ret>
): (...args: Args) => Promise<Ret> {
  const cache = opts?.cache || new LRUCache<Ret>();
  const buildKey = opts?.key || ((...a: Args) => stableKey(a));
  const dedupe = opts?.dedupe !== false; // default true
  const refreshOnStale = opts?.refreshOnStale ?? (Boolean(opts?.swrMs) || false);
  const inflight = new Map<string, Promise<Ret>>();

  return async function wrapped(...args: Args): Promise<Ret> {
    const key = buildKey(...args);
    // fast path: cache
    const meta = cache.getMeta(key);
    if (meta.hit && !meta.stale) return meta.value as Ret;

    // stale: optionally kick background refresh and return stale
    if (meta.hit && meta.stale) {
      if (refreshOnStale) triggerRefresh(key, args);
      return meta.value as Ret;
    }

    // miss: run with dedupe
    if (dedupe) {
      const existing = inflight.get(key);
      if (existing) return existing;
    }
    const p = runAndStore(key, args);
    if (dedupe) inflight.set(key, p);
    try {
      const val = await p;
      return val;
    } finally {
      inflight.delete(key);
    }
  };

  async function runAndStore(key: string, args: Args): Promise<Ret> {
    try {
      const val = await Promise.resolve(fn(...args));
      cache.set(key, val, { ttlMs: opts?.ttlMs, swrMs: opts?.swrMs });
      return val;
    } catch (e) {
      opts?.onError?.(e, key);
      throw e;
    }
  }

  function triggerRefresh(key: string, args: Args) {
    if (inflight.has(key)) return; // already refreshing
    const p = runAndStore(key, args);
    inflight.set(key, p);
    p.finally(() => inflight.delete(key));
  }
}

/* ───────────────────────── Namespaces ───────────────────────── */

/** Build a namespaced key helper: nsKey("user", 42) -> "ns:user|42" */
export function ns(ns: string, sep = "|") {
  const prefix = String(ns || "ns");
  return (...parts: any[]) => {
    const body = parts.map(seg => segToStr(seg)).join(sep);
    return prefix + ":" + body;
  };
}

/** Invalidate all keys carrying any of the given tags across caches you manage. */
export function invalidateTagged(caches: LRUCache<any>[], ...tags: string[]): number {
  let tot = 0;
  for (const c of caches) tot += c.invalidateByTags(tags);
  return tot;
}

/* ───────────────────────── Helpers ───────────────────────── */

function estimateSize(val: any): number {
  try {
    if (typeof val === "string") return val.length * 2;
    if (typeof val === "number") return 16;
    if (typeof val === "boolean") return 4;
    if (val == null) return 0;
    return JSON.stringify(val).length;
  } catch { return 128; }
}

function stableKey(x: any): string {
  // Order-independent JSON stringify
  return "k:" + stableJson(x);
}

function stableJson(x: any): string {
  if (x == null) return "null";
  const t = typeof x;
  if (t === "number" || t === "boolean") return JSON.stringify(x);
  if (t === "string") return JSON.stringify(x);
  if (Array.isArray(x)) return "[" + x.map(stableJson).join(",") + "]";
  const keys = Object.keys(x).sort();
  const parts: string[] = [];
  for (const k of keys) parts.push(JSON.stringify(k) + ":" + stableJson(x[k]));
  return "{" + parts.join(",") + "}";
}

function segToStr(v: any): string {
  if (v == null) return "~";
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") return String(v);
  return stableJson(v);
}

/* ───────────────────────── Tiny self-test ───────────────────────── */

export function __selftest__(): string {
  const c = new LRUCache<number>({ maxEntries: 2 });
  c.set("a", 1, { ttlMs: 50 });
  if (c.get("a") !== 1) return "miss_a";
  const keyer = ns("px");
  const k = keyer("AAPL", { w: "1d" });
  c.set(k, 42, { ttlMs: 1, swrMs: 50 });
  const m1 = c.getMeta(k);
  if (!m1.hit) return "miss_ns";
  // expire ttl but within swr → stale should serve
  (c as any).map.get(k).e.exp = Date.now() - 1;
  const m2 = c.getMeta(k);
  if (!m2.stale) return "no_stale";
  // memoize dedupe
  let runs = 0;
  const slow = memoize(async (x:number)=>{ runs++; await sleep(10); return x*2; }, { ttlMs: 50, swrMs: 50, cache: new LRUCache() });
  const p = Promise.all([slow(3), slow(3), slow(3)]);
  return p.then(vals=>{
    if (runs !== 1) return "no_dedupe";
    if (vals[0] !== 6) return "bad_val";
    return "ok";
  }) as unknown as string;
}

function sleep(ms:number){ return new Promise(r=>setTimeout(r,ms)); }