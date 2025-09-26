/**
 * actions/flags.ts
 * Pure, import-free utilities to manage feature flags with persistence + events.
 *
 * - No external deps. Works in browser (localStorage) and Node (memory fallback).
 * - CRUD: list, add, set, reset, import, export.
 * - Safe JSON parsing, normalization, deduplication.
 * - Tiny event system via emitSettingsEvent from dispatcher.ts.
 */

import { emitSettingsEvent } from "./dispatcher";

/* ================================ Types ================================== */

export type Flag = {
  key: string;                     // machine id
  label?: string;                   // display name
  description?: string;
  category?: string;
  default: boolean;
  on: boolean;
  createdAt?: string;               // ISO
  updatedAt?: string;               // ISO
};

export type FlagsEvent =
  | { type: "flags/updated"; payload: Flag[] }
  | { type: "flags/added"; payload: Flag }
  | { type: "flags/changed"; payload: Flag }
  | { type: "flags/removed"; payload: { key: string } };

const STORAGE_KEY = "settings:flags:v1";

/* ============================= Storage Layer ============================= */

const memoryStore: Record<string, string> = {};

function safeGet(k: string): string | null {
  try {
    if (typeof localStorage !== "undefined") return localStorage.getItem(k);
  } catch {}
  return memoryStore[k] ?? null;
}
function safeSet(k: string, v: string) {
  try {
    if (typeof localStorage !== "undefined") return localStorage.setItem(k, v);
  } catch {}
  memoryStore[k] = v;
}

/* ================================ Core =================================== */

export function listFlags(): Flag[] {
  try {
    const raw = safeGet(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return normalizeList(Array.isArray(parsed) ? parsed : parsed.items);
  } catch {
    return [];
  }
}

export function addFlag(input: {
  key: string;
  label?: string;
  description?: string;
  category?: string;
  default?: boolean;
  on?: boolean;
}): Flag[] {
  const list = listFlags();
  if (list.some((f) => f.key === input.key)) return list;
  const now = isoNow();
  const f: Flag = {
    key: sanitizeKey(input.key),
    label: input.label?.trim() || input.key,
    description: input.description?.trim(),
    category: input.category?.trim() || "General",
    default: !!input.default,
    on: input.on != null ? !!input.on : !!input.default,
    createdAt: now,
    updatedAt: now,
  };
  const out = [...list, f];
  persist(out);
  emitSettingsEvent({ type: "flags/added", payload: f });
  emitSettingsEvent({ type: "flags/updated", payload: out });
  return out;
}

export function setFlag(key: string, on: boolean): Flag[] {
  let changed: Flag | undefined;
  const out = listFlags().map((f) => {
    if (f.key !== key) return f;
    changed = { ...f, on: !!on, updatedAt: isoNow() };
    return changed;
  });
  persist(out);
  if (changed) {
    emitSettingsEvent({ type: "flags/changed", payload: changed });
    emitSettingsEvent({ type: "flags/updated", payload: out });
  }
  return out;
}

export function removeFlag(key: string): Flag[] {
  const list = listFlags();
  const out = list.filter((f) => f.key !== key);
  if (out.length === list.length) return out;
  persist(out);
  emitSettingsEvent({ type: "flags/removed", payload: { key } });
  emitSettingsEvent({ type: "flags/updated", payload: out });
  return out;
}

export function resetFlagsToDefault(): Flag[] {
  const out = listFlags().map((f) => ({ ...f, on: f.default, updatedAt: isoNow() }));
  persist(out);
  emitSettingsEvent({ type: "flags/updated", payload: out });
  return out;
}

export function replaceFlags(next: unknown): Flag[] {
  const normalized = normalizeList(Array.isArray(next) ? (next as any[]) : []);
  persist(normalized);
  emitSettingsEvent({ type: "flags/updated", payload: normalized });
  return normalized;
}

export function importFlags(payload: string | { flags?: unknown }): Flag[] {
  let arr: unknown;
  try {
    const obj = typeof payload === "string" ? JSON.parse(payload) : payload;
    arr = (obj as any)?.flags;
  } catch {}
  return replaceFlags(Array.isArray(arr) ? arr : []);
}

export function exportFlags(): string {
  const payload = { ts: isoNow(), flags: listFlags() };
  return JSON.stringify(payload, null, 2);
}

/* ================================ Utils ================================== */

function persist(list: Flag[]) {
  safeSet(STORAGE_KEY, JSON.stringify(list));
}

function normalizeList(list: any[]): Flag[] {
  const out: Flag[] = [];
  const seen = new Set<string>();
  for (const raw of Array.isArray(list) ? list : []) {
    const key = sanitizeKey(raw?.key || "");
    if (!key || seen.has(key)) continue;
    seen.add(key);
    const now = isoNow();
    out.push({
      key,
      label: raw?.label?.trim() || key,
      description: raw?.description?.trim(),
      category: raw?.category?.trim() || "General",
      default: !!raw?.default,
      on: raw?.on != null ? !!raw.on : !!raw?.default,
      createdAt: toIsoOrNull(raw?.createdAt) || now,
      updatedAt: toIsoOrNull(raw?.updatedAt) || now,
    });
  }
  return out;
}

function sanitizeKey(k: string): string {
  return String(k || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_\-]/g, "_")
    .slice(0, 64);
}

function isoNow() {
  try { return new Date().toISOString(); } catch { return "" + Date.now(); }
}

function toIsoOrNull(x: unknown): string | null {
  if (!x) return null;
  const t = Date.parse(String(x));
  return isFinite(t) ? new Date(t).toISOString() : null;
}