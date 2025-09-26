/**
 * actions/layouts.ts
 * Pure, import-free utilities to manage layout presets with persistence + events.
 *
 * - Works in browser (localStorage) or Node (memory fallback).
 * - CRUD: list, add, rename, remove, reorder, load, import/export.
 * - Emits events via dispatcher.
 */

import { emitSettingsEvent } from "./dispatcher";

/* ================================ Types ================================== */

export type LayoutPreset = {
  id: string;
  name: string;
  icon?: string;
  layout: any;
  createdAt?: string;
  updatedAt?: string;
};

export type LayoutsEvent =
  | { type: "layouts/updated"; payload: LayoutPreset[] }
  | { type: "layouts/added"; payload: LayoutPreset }
  | { type: "layouts/changed"; payload: LayoutPreset }
  | { type: "layouts/removed"; payload: { id: string } }
  | { type: "layouts/loaded"; payload: LayoutPreset }
  | { type: "layouts/reordered"; payload: LayoutPreset[] };

const STORAGE_KEY = "settings:layouts:v1";

/* ============================= Storage Layer ============================= */

const memoryStore: Record<string, string> = {};

function safeGet(k: string): string | null {
  try { if (typeof localStorage !== "undefined") return localStorage.getItem(k); } catch {}
  return memoryStore[k] ?? null;
}
function safeSet(k: string, v: string) {
  try { if (typeof localStorage !== "undefined") return localStorage.setItem(k, v); } catch {}
  memoryStore[k] = v;
}

/* ================================ Core =================================== */

export function listPresets(): LayoutPreset[] {
  try {
    const raw = safeGet(STORAGE_KEY);
    if (!raw) return seed();
    const parsed = JSON.parse(raw);
    const arr = Array.isArray(parsed) ? parsed : parsed?.items;
    const list = normalizeList(Array.isArray(arr) ? arr : []);
    return list.length ? list : seed();
  } catch {
    return seed();
  }
}

export function addPreset(name: string, layout: any = {}, icon = "🧩"): LayoutPreset[] {
  const now = isoNow();
  const preset: LayoutPreset = {
    id: genId("p_"),
    name: String(name || "Untitled"),
    icon,
    layout,
    createdAt: now,
    updatedAt: now,
  };
  const out = [...listPresets(), preset];
  persist(out);
  emitSettingsEvent({ type: "layouts/added", payload: preset });
  emitSettingsEvent({ type: "layouts/updated", payload: out });
  return out;
}

export function renamePreset(id: string, name: string): LayoutPreset[] {
  let changed: LayoutPreset | undefined;
  const out = listPresets().map((p) => {
    if (p.id !== id) return p;
    changed = { ...p, name: String(name || p.name), updatedAt: isoNow() };
    return changed;
  });
  persist(out);
  if (changed) {
    emitSettingsEvent({ type: "layouts/changed", payload: changed });
    emitSettingsEvent({ type: "layouts/updated", payload: out });
  }
  return out;
}

export function removePreset(id: string): LayoutPreset[] {
  const list = listPresets();
  const out = list.filter((p) => p.id !== id);
  if (out.length === list.length) return out;
  persist(out);
  emitSettingsEvent({ type: "layouts/removed", payload: { id } });
  emitSettingsEvent({ type: "layouts/updated", payload: out });
  return out;
}

export function reorderPresets(order: string[]): LayoutPreset[] {
  const list = listPresets();
  const map = new Map(list.map((p) => [p.id, p]));
  const seen = new Set(order);
  const reordered: LayoutPreset[] = [];
  for (const id of order) {
    const item = map.get(id);
    if (item) reordered.push(item);
  }
  for (const p of list) if (!seen.has(p.id)) reordered.push(p);
  persist(reordered);
  emitSettingsEvent({ type: "layouts/reordered", payload: reordered });
  emitSettingsEvent({ type: "layouts/updated", payload: reordered });
  return reordered;
}

export function loadPreset(id: string): LayoutPreset | null {
  const p = listPresets().find((x) => x.id === id) || null;
  if (p) emitSettingsEvent({ type: "layouts/loaded", payload: p });
  return p;
}

export function replacePresets(next: unknown): LayoutPreset[] {
  const normalized = normalizeList(Array.isArray(next) ? (next as any[]) : []);
  persist(normalized);
  emitSettingsEvent({ type: "layouts/updated", payload: normalized });
  return normalized;
}

export function exportPresets(): string {
  const payload = { ts: isoNow(), presets: listPresets() };
  return JSON.stringify(payload, null, 2);
}

export function importPresets(payload: string | { presets?: unknown }): LayoutPreset[] {
  let arr: unknown;
  try {
    const obj = typeof payload === "string" ? JSON.parse(payload) : payload;
    arr = (obj as any)?.presets;
  } catch {}
  return replacePresets(Array.isArray(arr) ? arr : []);
}

/* ================================ Utils ================================== */

function persist(list: LayoutPreset[]) {
  safeSet(STORAGE_KEY, JSON.stringify(list));
}

function normalizeList(list: any[]): LayoutPreset[] {
  const out: LayoutPreset[] = [];
  const seen = new Set<string>();
  for (const raw of Array.isArray(list) ? list : []) {
    const id = String(raw?.id || genId("p_"));
    if (seen.has(id)) continue;
    seen.add(id);
    const now = isoNow();
    out.push({
      id,
      name: String(raw?.name || id),
      icon: raw?.icon || "📐",
      layout: raw?.layout || {},
      createdAt: toIsoOrNull(raw?.createdAt) || now,
      updatedAt: toIsoOrNull(raw?.updatedAt) || now,
    });
  }
  return out;
}

function seed(): LayoutPreset[] {
  const now = isoNow();
  return [
    { id: "default", name: "Default layout", icon: "📊", layout: {}, createdAt: now, updatedAt: now },
  ];
}

function genId(prefix = "") { return prefix + Math.random().toString(36).slice(2, 8); }
function isoNow() { try { return new Date().toISOString(); } catch { return "" + Date.now(); } }
function toIsoOrNull(x: unknown): string | null {
  if (!x) return null;
  const t = Date.parse(String(x));
  return isFinite(t) ? new Date(t).toISOString() : null;
}