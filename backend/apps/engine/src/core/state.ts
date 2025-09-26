// core/state.ts
// Global in-memory state manager for the engine (pure Node, no imports).
//
// - Central key-value store (shallow + nested paths).
// - Immutable snapshot queries.
// - Event-driven updates via .on("change", fn).
// - Helpers: get(), set(), update(), remove(), clear(), snapshot().
// - Supports dot.path for nested objects.
// - Designed for lightweight real-time state (not a DB).
//
// Example:
//   const state = createState();
//   state.on("change", e => console.log("changed:", e.key, e.value));
//   state.set("risk.limit", 100);
//   console.log(state.get("risk.limit")); // 100

export type StateEvent = {
  type: "set" | "update" | "remove" | "clear";
  key?: string;
  value?: any;
  prev?: any;
  ts: string;
};

export type StateListener = (e: StateEvent) => void;

export type State = {
  get<T = any>(key: string, fallback?: T): T | undefined;
  set<T = any>(key: string, value: T): void;
  update<T = any>(key: string, fn: (prev: T) => T): void;
  remove(key: string): void;
  clear(): void;
  snapshot(): any;
  on(ev: "change", fn: StateListener): () => void;
};

export function createState(initial: any = {}): State {
  let store: any = deepClone(initial);
  const listeners = new Set<StateListener>();

  function ts() {
    try { return new Date().toISOString(); } catch { return "" + Date.now(); }
  }

  function emit(ev: StateEvent) {
    for (const fn of Array.from(listeners)) {
      try { fn(ev); } catch {}
    }
  }

  function pathGet(obj: any, path: string): any {
    if (!path) return obj;
    const parts = path.split(".");
    let cur = obj;
    for (const p of parts) {
      if (cur == null) return undefined;
      cur = cur[p];
    }
    return cur;
  }

  function pathSet(obj: any, path: string, val: any): any {
    const parts = path.split(".");
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      if (!cur[p] || typeof cur[p] !== "object") cur[p] = {};
      cur = cur[p];
    }
    cur[parts[parts.length - 1]] = val;
    return obj;
  }

  function pathRemove(obj: any, path: string): any {
    const parts = path.split(".");
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      if (!cur[p] || typeof cur[p] !== "object") return obj;
      cur = cur[p];
    }
    delete cur[parts[parts.length - 1]];
    return obj;
  }

  function get<T = any>(key: string, fallback?: T): T | undefined {
    const v = pathGet(store, key);
    return v !== undefined ? v : fallback;
  }

  function set<T = any>(key: string, value: T) {
    const prev = pathGet(store, key);
    store = pathSet(store, key, value);
    emit({ type: "set", key, value, prev, ts: ts() });
  }

  function update<T = any>(key: string, fn: (prev: T) => T) {
    const prev = pathGet(store, key);
    const next = fn(prev);
    store = pathSet(store, key, next);
    emit({ type: "update", key, value: next, prev, ts: ts() });
  }

  function remove(key: string) {
    const prev = pathGet(store, key);
    store = pathRemove(store, key);
    emit({ type: "remove", key, prev, ts: ts() });
  }

  function clear() {
    store = {};
    emit({ type: "clear", ts: ts() });
  }

  function snapshot() {
    return deepClone(store);
  }

  function on(ev: "change", fn: StateListener) {
    if (ev !== "change") return () => {};
    listeners.add(fn);
    return () => { listeners.delete(fn); };
  }

  return { get, set, update, remove, clear, snapshot, on };
}

/* ------------------------- Helpers ------------------------- */

function deepClone<T>(x: T): T {
  try { return JSON.parse(JSON.stringify(x)); }
  catch { return x; }
}