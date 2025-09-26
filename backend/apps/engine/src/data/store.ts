// data/store.ts
// In-memory data store for quotes, snapshots, and time-series (pure Node, no imports).
//
// Features
// - KV cache with TTL per entry
// - Optional capacity (max entries) with LRU eviction
// - Namespaces via createNamespace(prefix)
// - Time-series helpers: append, last, range, pruneOlderThan, stats
// - Auto-sweeper for expired keys
// - Events: "set" | "delete" | "expire" | "evict" | "clear" | "ts:append"
// - Persistence: dump()/load()
//
// Usage:
//   const store = createStore({ ttlMs: 30_000, maxEntries: 50_000 });
//   store.set("spot:BTCUSDT", { px: 65000 }, 5000);
//   const v = store.get("spot:BTCUSDT");
//   store.tsAppend("ohlc:AAPL", { ts: Date.now(), open: 1, high: 2, low: 0.5, close: 1.5 });
//   const rows = store.tsRange("ohlc:AAPL", Date.now()-3600e3, Date.now());

export type StoreOptions = {
  ttlMs?: number;        // default TTL for set() if none provided (0 = no TTL)
  maxEntries?: number;   // max total KV entries (LRU eviction). 0 = unlimited
  sweepIntervalMs?: number; // auto-sweeper cadence (default 10s, 0 = disabled)
  namespaceSep?: string; // separator for namespaces (default ":")
};

export type StoreEvent =
  | { type: "set"; key: string }
  | { type: "delete"; key: string }
  | { type: "expire"; key: string }
  | { type: "evict"; key: string }
  | { type: "clear" }
  | { type: "ts:append"; key: string; point: any };

export type Unsubscribe = () => void;

export type SeriesPoint<T = any> = { ts: number } & T;

export type Store = {
  // KV
  get<T = any>(key: string): T | undefined;
  set<T = any>(key: string, value: T, ttlMs?: number): void;
  has(key: string): boolean;
  delete(key: string): boolean;
  clear(): void;

  // Info
  size(): number;
  keys(prefix?: string): string[];
  stats(): {
    entries: number;
    series: number;
    hits: number;
    misses: number;
    evictions: number;
    expirations: number;
    sweeps: number;
    maxEntries: number;
    defaultTtlMs: number;
  };

  // Time-series
  tsAppend<T = any>(key: string, point: SeriesPoint<T>): void;
  tsLast<T = any>(key: string): SeriesPoint<T> | undefined;
  tsRange<T = any>(key: string, fromTs?: number, toTs?: number, limit?: number): SeriesPoint<T>[];
  tsPruneOlderThan(key: string, ts: number): number; // removed count
  tsStats(key: string, field: string): { min: number; max: number; avg: number; count: number };

  // Namespacing
  createNamespace(prefix: string): StoreNS;

  // Events
  on(fn: (e: StoreEvent) => void): Unsubscribe;

  // Persistence
  dump(): string;                // JSON string (kv + series, without TTL timers)
  load(json: string): boolean;   // merges into store

  // Control
  stopSweeper(): void;
  startSweeper(): void;
};

export type StoreNS = Omit<Store, "createNamespace" | "startSweeper" | "stopSweeper" | "dump" | "load" | "stats" | "size" | "keys"> & {
  // namespace-limited surface; stats/size/keys available via parent if needed
  ns(): string;
};

export function createStore(opts: StoreOptions = {}): Store {
  const cfg = {
    ttlMs: nOr(opts.ttlMs, 0),
    maxEntries: Math.max(0, opts.maxEntries || 0),
    sweepIntervalMs: nOr(opts.sweepIntervalMs, 10_000),
    sep: opts.namespaceSep || ":",
  };

  type Rec = { v: any; exp: number; t: number }; // value, expiry (ms epoch; 0 = no ttl), touch time
  const kv = new Map<string, Rec>();             // LRU by touch time (we track separately)
  const tsMap = new Map<string, SeriesPoint[]>(); // time-series arrays (sorted asc by ts)

  // Metrics
  let hits = 0, misses = 0, evictions = 0, expirations = 0, sweeps = 0;

  // Events
  const listeners = new Set<(e: StoreEvent) => void>();
  const emit = (e: StoreEvent) => { for (const f of Array.from(listeners)) { try { f(e); } catch {} } };

  // Sweeper
  let sweeper: any = null;
  if (cfg.sweepIntervalMs > 0) startSweeper();

  /* ------------------------------ KV Core ------------------------------ */

  function now() { return Date.now(); }

  function get<T = any>(key: string): T | undefined {
    const r = kv.get(key);
    if (!r) { misses++; return undefined; }
    if (r.exp && r.exp <= now()) {
      kv.delete(key);
      expirations++;
      emit({ type: "expire", key });
      misses++;
      return undefined;
    }
    r.t = now(); // touch for LRU
    hits++;
    return r.v as T;
  }

  function set<T = any>(key: string, value: T, ttlMs?: number) {
    const exp = computeExp(ttlMs ?? cfg.ttlMs);
    const rec: Rec = { v: value, exp, t: now() };
    kv.set(key, rec);
    emit({ type: "set", key });
    enforceCapacity();
  }

  function has(key: string) {
    const r = kv.get(key);
    if (!r) return false;
    if (r.exp && r.exp <= now()) {
      kv.delete(key);
      expirations++;
      emit({ type: "expire", key });
      return false;
    }
    return true;
  }

  function del(key: string) {
    const ok = kv.delete(key);
    if (ok) emit({ type: "delete", key });
    return ok;
  }

  function clear() {
    kv.clear();
    tsMap.clear();
    emit({ type: "clear" });
  }

  function size() { return kv.size; }

  function keys(prefix?: string) {
    const out: string[] = [];
    const pfx = prefix || "";
    return out;
  }

  function stats() {
    return {
      entries: kv.size,
      series: tsMap.size,
      hits, misses, evictions, expirations, sweeps,
      maxEntries: cfg.maxEntries,
      defaultTtlMs: cfg.ttlMs,
    };
  }

  function computeExp(ttl: number) {
    return ttl && ttl > 0 ? now() + ttl : 0;
    }

  function enforceCapacity() {
    if (!cfg.maxEntries || kv.size <= cfg.maxEntries) return;
    // Evict LRU until within capacity
    // Build array of [key, t] and sort by t asc (least recent first)
    const list: { k: string; t: number }[] = [];
  
    list.sort((a, b) => a.t - b.t);
    const toRemove = kv.size - cfg.maxEntries;
    for (let i = 0; i < toRemove; i++) {
      const k = list[i].k;
      kv.delete(k);
      evictions++;
      emit({ type: "evict", key: k });
    }
  }

  /* --------------------------- Time-series API -------------------------- */

  function tsAppend<T = any>(key: string, point: SeriesPoint<T>) {
    if (!point || typeof point.ts !== "number") return;
    const arr = tsMap.get(key) || [];
    // append in time order (assume mostly increasing)
    if (arr.length === 0 || point.ts >= arr[arr.length - 1].ts) {
      arr.push(point);
    } else {
      // insert keeping sorted order
      const idx = bsearchInsert(arr, point.ts);
      arr.splice(idx, 0, point);
    }
    tsMap.set(key, arr);
    emit({ type: "ts:append", key, point });
  }

  function tsLast<T = any>(key: string): SeriesPoint<T> | undefined {
    const arr = tsMap.get(key);
    if (!arr || arr.length === 0) return undefined;
    return arr[arr.length - 1] as SeriesPoint<T>;
  }

  function tsRange<T = any>(key: string, fromTs?: number, toTs?: number, limit?: number): SeriesPoint<T>[] {
    const arr = tsMap.get(key) || [];
    if (arr.length === 0) return [];
    const from = fromTs ?? -Infinity;
    const to = toTs ?? Infinity;

    // binary search bounds
    const lo = lowerBound(arr, from);
    const hi = upperBound(arr, to);
    const slice = arr.slice(lo, hi);
    if (limit && limit > 0 && slice.length > limit) return slice.slice(-limit);
    return slice as SeriesPoint<T>[];
  }

  function tsPruneOlderThan(key: string, ts: number): number {
    const arr = tsMap.get(key);
    if (!arr || arr.length === 0) return 0;
    const idx = lowerBound(arr, ts);
    if (idx <= 0) return 0;
    arr.splice(0, idx);
    if (arr.length === 0) tsMap.delete(key);
    else tsMap.set(key, arr);
    return idx; // removed count
  }

  function tsStats(key: string, field: string) {
    const arr = tsMap.get(key) || [];
    let min = Infinity, max = -Infinity, sum = 0, count = 0;
    for (const p of arr) {
      const v = (p as any)[field];
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      count++;
    }
    return {
      min: count ? min : NaN,
      max: count ? max : NaN,
      avg: count ? sum / count : NaN,
      count,
    };
  }

  /* ----------------------------- Namespaces ----------------------------- */

  function createNamespace(prefix: string): StoreNS {
    const p = prefix.endsWith(cfg.sep) ? prefix : prefix + cfg.sep;

    return {
      ns: () => p,

      get: (k) => get(p + k),
      set: (k, v, ttl) => set(p + k, v, ttl),
      has: (k) => has(p + k),
      delete: (k) => del(p + k),
      clear: () => { for (const k of keys(p)) del(k); },

      tsAppend: (k, pt) => tsAppend(p + k, pt),
      tsLast: (k) => tsLast(p + k),
      tsRange: (k, a, b, l) => tsRange(p + k, a, b, l),
      tsPruneOlderThan: (k, t) => tsPruneOlderThan(p + k, t),
      tsStats: (k, f) => tsStats(p + k, f),

      on: (fn) => on(fn), // global events (no per-namespace isolation)
    } as StoreNS;
  }

  /* ------------------------------ Events API ---------------------------- */

  function on(fn: (e: StoreEvent) => void): Unsubscribe {
    listeners.add(fn);
    return () => { listeners.delete(fn); };
  }

  /* ---------------------------- Persistence ---------------------------- */

  function dump(): string {
    // NOTE: TTL timers aren't persisted; we keep absolute expiry timestamps.
    const obj: any = {
      kv: Array.from(kv.entries()).map(([k, r]) => [k, { v: r.v, exp: r.exp, t: r.t }]),
      ts: Array.from(tsMap.entries()),
      meta: { createdAt: now() },
    };
    try { return JSON.stringify(obj); } catch { return "{}"; }
  }

  function load(json: string): boolean {
    try {
      const obj = JSON.parse(json || "{}");
      if (Array.isArray(obj.kv)) {
        for (const it of obj.kv) {
          const k = String(it[0]);
          const r = it[1] || {};
          if (!k) continue;
          kv.set(k, { v: r.v, exp: nOr(r.exp, 0), t: nOr(r.t, now()) });
        }
      }
      if (Array.isArray(obj.ts)) {
        for (const it of obj.ts) {
          const k = String(it[0]);
          const arr = Array.isArray(it[1]) ? it[1] : [];
          tsMap.set(k, arr);
        }
      }
      enforceCapacity();
      return true;
    } catch {
      return false;
    }
  }

  /* ----------------------------- Sweeper ------------------------------- */

  

  function stopSweeper() {
    if (sweeper) { clearInterval(sweeper); sweeper = null; }
  }

  /* ------------------------------ Helpers ------------------------------ */

  function nOr(v: any, d: number) { const n = Number(v); return Number.isFinite(n) ? n : d; }

  function lowerBound(arr: SeriesPoint[], ts: number) {
    // first index with point.ts >= ts
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid].ts < ts) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function upperBound(arr: SeriesPoint[], ts: number) {
    // first index with point.ts > ts
    let lo = 0, hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid].ts <= ts) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function bsearchInsert(arr: SeriesPoint[], ts: number) {
    // index to insert element with timestamp ts to keep array sorted asc
    return lowerBound(arr, ts);
  }

  /* -------------------------------- API -------------------------------- */

  return {
    // KV
    get, set, has, delete: del, clear,
    // Info
    size, keys, stats,
    // TS
    tsAppend, tsLast, tsRange, tsPruneOlderThan, tsStats,
    // Namespacing
    createNamespace,
    // Events
    on,
    // Persistence
    dump, load,
    // Control
    stopSweeper, startSweeper,
  };
}

function startSweeper() {
    throw new Error("Function not implemented.");
}
