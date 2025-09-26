/**
 * actions/connections.ts
 * Pure, import-free utilities for managing data/API connections with robust persistence.
 *
 * - No external imports. Works in browser (localStorage) and Node (in-memory fallback).
 * - Strong normalization/validation; safe JSON parsing.
 * - Tiny event system (subscribe/unsubscribe) with typed events.
 * - CRUD, bulk ops, reorder, search, import/export, and soft migrations.
 * - Optional custom storage (e.g., to wire up your own backend).
 */

/* ================================ Types ================================== */

export type ConnectionType = "API" | "DB" | "File" | "Other";
export type ConnectionStatus = "connected" | "disconnected";

export type Connection = {
  id: string;
  name: string;
  type: ConnectionType;
  status: ConnectionStatus;
  meta?: Record<string, unknown>;     // optional addons (e.g., baseUrl)
  createdAt?: string;                 // ISO
  updatedAt?: string;                 // ISO
};

export type ConnectionsEvent =
  | { type: "connections/updated"; payload: Connection[] }
  | { type: "connections/added"; payload: Connection }
  | { type: "connections/changed"; payload: Connection }
  | { type: "connections/removed"; payload: { id: string } }
  | { type: "connections/reordered"; payload: Connection[] };

export type Unsubscribe = () => void;

export type StorageDriver = {
  getItem(key: string): string | null;
  setItem(key: string, val: string): void;
};

/* ============================== Constants ================================ */

const STORAGE_KEY = "settings:connections:v2"; // bump when structure changes

/* ============================ Storage Layer ============================== */

// In-memory fallback for non-browser environments or storage failures.
const memoryStore: Record<string, string> = {};

const defaultStorage: StorageDriver = (() => {
  try {
    if (typeof localStorage !== "undefined") {
      return {
        getItem: (k) => localStorage.getItem(k),
        setItem: (k, v) => localStorage.setItem(k, v),
      };
    }
  } catch {}
  return {
    getItem: (k) => (k in memoryStore ? memoryStore[k] : null),
    setItem: (k, v) => { memoryStore[k] = v; },
  };
})();

let driver: StorageDriver = defaultStorage;

/** Optionally provide your own storage driver (e.g., wrapper around an API). */
export function useConnectionsStorage(custom: StorageDriver | null | undefined) {
  driver = custom || defaultStorage;
}

/* ============================== Event Bus ================================ */

const listeners = new Set<(e: ConnectionsEvent) => void>();

export function onConnectionsEvent(fn: (e: ConnectionsEvent) => void): Unsubscribe {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit(e: ConnectionsEvent) {
  for (const fn of listeners) {
    try { fn(e); } catch {}
  }
}

/* ============================ Read / Write =============================== */

type Persisted = { version: number; items: Connection[] };

function load(): Connection[] {
  const raw = safeGet(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as Persisted | Connection[];
    // v1 was raw array; v2 uses {version, items}
    const items = Array.isArray(parsed) ? parsed : Array.isArray(parsed.items) ? parsed.items : [];
    return normalizeList(items);
  } catch {
    return [];
  }
}

function save(list: Connection[]) {
  const payload: Persisted = { version: 2, items: list };
  safeSet(STORAGE_KEY, JSON.stringify(payload));
}

function safeGet(k: string): string | null {
  try { return driver.getItem(k); } catch { return null; }
}
function safeSet(k: string, v: string) {
  try { driver.setItem(k, v); } catch {}
}

/* ================================ API =================================== */

/** Return all connections. */
export function listConnections(): Connection[] {
  return load();
}

/** Look up a single connection by id. */
export function getConnection(id: string): Connection | null {
  return load().find((c) => c.id === id) || null;
}

/** Add a new connection. */
export function addConnection(input: {
  name: string;
  type?: string;                          // tolerant input
  status?: string;                        // tolerant input
  meta?: Record<string, unknown>;
}): Connection[] {
  const now = isoNow();
  const next: Connection = {
    id: genId("c_"),
    name: sanitizeName(input.name),
    type: normalizeType(input.type),
    status: normalizeStatus(input.status),
    meta: isObject(input.meta) ? input.meta : undefined,
    createdAt: now,
    updatedAt: now,
  };

  const list = [...load(), next];
  save(list);
  emit({ type: "connections/added", payload: next });
  emit({ type: "connections/updated", payload: list });
  return list;
}

/** Update a connection (name/type/status/meta). */
export function updateConnection(
  id: string,
  patch: Partial<Pick<Connection, "name" | "type" | "status" | "meta">>
): Connection[] {
  const list = load();
  let changed: Connection | undefined;

  const out = list.map((c) => {
    if (c.id !== id) return c;
    const next: Connection = {
      ...c,
      name: patch.name != null ? sanitizeName(String(patch.name)) : c.name,
      type: patch.type != null ? normalizeType(patch.type) : c.type,
      status: patch.status != null ? normalizeStatus(patch.status) : c.status,
      meta: patch.meta != null ? (isObject(patch.meta) ? patch.meta : c.meta) : c.meta,
      updatedAt: isoNow(),
    };
    changed = next;
    return next;
  });

  save(out);
  if (changed) {
    emit({ type: "connections/changed", payload: changed });
    emit({ type: "connections/updated", payload: out });
  }
  return out;
}

/** Toggle connection status connected⟷disconnected. */
export function toggleConnection(id: string): Connection[] {
  const c = getConnection(id);
  if (!c) return load();
  const status: ConnectionStatus = c.status === "connected" ? "disconnected" : "connected";
  return updateConnection(id, { status });
}



/** Remove a connection by id. */
export function removeConnection(id: string): Connection[] {
  const list = load();
  const out = list.filter((c) => c.id !== id);
  if (out.length === list.length) return out;
  save(out);
  emit({ type: "connections/removed", payload: { id } });
  emit({ type: "connections/updated", payload: out });
  return out;
}

/** Replace the entire list (e.g., import). */
export function replaceConnections(next: unknown): Connection[] {
  const normalized = normalizeList(Array.isArray(next) ? next as any[] : []);
  save(normalized);
  emit({ type: "connections/updated", payload: normalized });
  return normalized;
}

/** Reorder list by providing an array of ids in the desired order. */
export function reorderConnections(order: string[]): Connection[] {
  const list = load();
  const map = new Map(list.map((c) => [c.id, c]));
  const seen = new Set(order);
  const reordered: Connection[] = [];

  for (const id of order) {
    const item = map.get(id);
    if (item) reordered.push(item);
  }
  // append any missing (stable)
  for (const c of list) if (!seen.has(c.id)) reordered.push(c);

  save(reordered);
  emit({ type: "connections/reordered", payload: reordered });
  emit({ type: "connections/updated", payload: reordered });
  return reordered;
}

/** Find connections by fuzzy text (matches name, type, status, id). */
export function searchConnections(q: string): Connection[] {
  const s = String(q || "").trim().toLowerCase();
  if (!s) return load();
  const hay = (c: Connection) =>
    `${c.name} ${c.type} ${c.status} ${c.id}`.toLowerCase();
  return load().filter((c) => hay(c).includes(s));
}

/** Export as pretty JSON. */
export function exportConnections(): string {
  const payload = { version: 2, ts: isoNow(), connections: load() };
  return JSON.stringify(payload, null, 2);
}

/** Import from string or object. Ignores invalid entries, dedupes by id. */
export function importConnections(payload: string | { connections?: unknown }): Connection[] {
  let list: unknown;
  try {
    const obj = typeof payload === "string" ? JSON.parse(payload) : payload;
    list = (obj as any)?.connections;
  } catch {}
  return replaceConnections(Array.isArray(list) ? list : []);
}

/* ================================ Utils ================================== */

function normalizeList(list: any[]): Connection[] {
  const out: Connection[] = [];
  const seen = new Set<string>();

  for (const raw of Array.isArray(list) ? list : []) {
    const id = String(raw?.id || genId("c_"));
    if (seen.has(id)) continue;
    seen.add(id);

    const name = sanitizeName(raw?.name ?? "Untitled");
    const type = normalizeType(raw?.type);
    const status = normalizeStatus(raw?.status);
    const meta = isObject(raw?.meta) ? (raw.meta as Record<string, unknown>) : undefined;

    const createdAt = toIsoOrNull(raw?.createdAt) || isoNow();
    const updatedAt = toIsoOrNull(raw?.updatedAt) || createdAt;

    out.push({ id, name, type, status, meta, createdAt, updatedAt });
  }
  return out;
}

function normalizeType(t: unknown): ConnectionType {
  const v = String(t ?? "").trim().toUpperCase();
  if (v === "API") return "API";
  if (v === "DB") return "DB";
  if (v === "FILE") return "File";
  if (v === "OTHER") return "Other";
  return "Other";
}

function normalizeStatus(s: unknown): ConnectionStatus {
  const v = String(s ?? "").trim().toLowerCase();
  return v === "connected" ? "connected" : "disconnected";
}

function sanitizeName(n: unknown): string {
  const s = String(n ?? "").trim();
  return s || "Untitled";
}

function isObject(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === "object" && !Array.isArray(x);
}

function genId(prefix = ""): string {
  return prefix + Math.random().toString(36).slice(2, 10);
}

function isoNow(): string {
  try { return new Date().toISOString(); } catch { return "" + Date.now(); }
}

function toIsoOrNull(x: unknown): string | null {
  if (!x) return null;
  const s = String(x);
  const t = Date.parse(s);
  return isFinite(t) ? new Date(t).toISOString() : null;
}