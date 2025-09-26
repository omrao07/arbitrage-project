/**
 * actions/hotkeys.ts
 * Pure, import-free utilities to manage keyboard shortcuts with persistence + events.
 *
 * - No external deps. Works in browser (localStorage) and Node (memory fallback).
 * - CRUD + reorder + search + conflict detection.
 * - Robust combo normalization (Ctrl/Cmd/Alt/Shift + Key, function keys, symbols).
 * - Import/Export helpers and safe JSON handling.
 */

import { emitSettingsEvent } from "./dispatcher";

/* ================================ Types ================================== */

export type Hotkey = {
  id: string;                 // stable id for the action (unique)
  label: string;              // user-facing label
  combo: string;              // normalized combo string, e.g. "Ctrl+Shift+K", "Cmd+S", "Slash", "Unassigned"
  category?: string;          // grouping (e.g., "Global", "Editor")
  description?: string;       // help text
  createdAt?: string;         // ISO
  updatedAt?: string;         // ISO
};

export type HotkeysEvent =
  | { type: "hotkeys/updated"; payload: Hotkey[] }
  | { type: "hotkeys/added"; payload: Hotkey }
  | { type: "hotkeys/changed"; payload: Hotkey }
  | { type: "hotkeys/removed"; payload: { id: string } }
  | { type: "hotkeys/reordered"; payload: Hotkey[] };

const STORAGE_KEY = "settings:hotkeys:v1";

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

export function listHotkeys(): Hotkey[] {
  try {
    const raw = safeGet(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    const arr = Array.isArray(parsed) ? parsed : parsed?.items;
    return normalizeList(Array.isArray(arr) ? arr : []);
  } catch {
    return [];
  }
}

export function addHotkey(partial?: Partial<Hotkey>): Hotkey[] {
  const list = listHotkeys();
  const now = isoNow();
  const hk: Hotkey = {
    id: partial?.id?.trim() || genId("hk_"),
    label: (partial?.label || "New action").trim(),
    combo: normalizeCombo(partial?.combo || "Unassigned"),
    category: partial?.category?.trim() || "Custom",
    description: partial?.description?.trim() || "",
    createdAt: now,
    updatedAt: now,
  };
  const out = [...list, hk];
  persist(out);
  emitSettingsEvent({ type: "hotkeys/added", payload: hk });
  emitSettingsEvent({ type: "hotkeys/updated", payload: out });
  return out;
}

export function updateHotkey(id: string, patch: Partial<Hotkey>): Hotkey[] {
  let changed: Hotkey | undefined;
  const out = listHotkeys().map((h) => {
    if (h.id !== id) return h;
    const next: Hotkey = {
      ...h,
      label: patch.label != null ? String(patch.label).trim() : h.label,
      combo: patch.combo != null ? normalizeCombo(patch.combo) : h.combo,
      category: patch.category != null ? String(patch.category).trim() : h.category,
      description: patch.description != null ? String(patch.description).trim() : h.description,
      updatedAt: isoNow(),
    };
    changed = next;
    return next;
  });
  persist(out);
  if (changed) {
    emitSettingsEvent({ type: "hotkeys/changed", payload: changed });
    emitSettingsEvent({ type: "hotkeys/updated", payload: out });
  }
  return out;
}

export function removeHotkey(id: string): Hotkey[] {
  const list = listHotkeys();
  const out = list.filter((h) => h.id !== id);
  if (out.length === list.length) return out;
  persist(out);
  emitSettingsEvent({ type: "hotkeys/removed", payload: { id } });
  emitSettingsEvent({ type: "hotkeys/updated", payload: out });
  return out;
}

export function replaceHotkeys(next: unknown): Hotkey[] {
  const normalized = normalizeList(Array.isArray(next) ? (next as any[]) : []);
  persist(normalized);
  emitSettingsEvent({ type: "hotkeys/updated", payload: normalized });
  return normalized;
}

/** Move an item to a new index (bounds are clamped). */
export function moveHotkey(id: string, newIndex: number): Hotkey[] {
  const list = listHotkeys();
  const idx = list.findIndex((h) => h.id === id);
  if (idx < 0) return list;
  const item = list[idx];
  const arr = [...list];
  arr.splice(idx, 1);
  const ni = Math.max(0, Math.min(arr.length, newIndex | 0));
  arr.splice(ni, 0, item);
  persist(arr);
  emitSettingsEvent({ type: "hotkeys/reordered", payload: arr });
  emitSettingsEvent({ type: "hotkeys/updated", payload: arr });
  return arr;
}

/** Return a map from combo -> array of hotkey ids that share it (conflicts). */
export function conflicts(list: Hotkey[] = listHotkeys()): Record<string, string[]> {
  const map: Record<string, string[]> = {};
  for (const it of list) {
    const k = normalizeCombo(it.combo);
    if (k === "Unassigned") continue;
    (map[k] ||= []).push(it.id);
  }
  for (const k of Object.keys(map)) if (map[k].length < 2) delete map[k];
  return map;
}

/** Find the first hotkey that matches a combo string (after normalization). */
export function findHotkeyByCombo(combo: string, list: Hotkey[] = listHotkeys()): Hotkey | null {
  const key = normalizeCombo(combo);
  return list.find((h) => normalizeCombo(h.combo) === key) || null;
}

/** Simple fuzzy search across label, combo, id, category, description. */
export function searchHotkeys(q: string): Hotkey[] {
  const s = (q || "").trim().toLowerCase();
  if (!s) return listHotkeys();
  const hay = (h: Hotkey) =>
    `${h.label} ${h.combo} ${h.id} ${h.category || ""} ${h.description || ""}`.toLowerCase();
  return listHotkeys().filter((h) => hay(h).includes(s));
}

/** Export hotkeys as pretty JSON. */
export function exportHotkeys(): string {
  const payload = { ts: isoNow(), hotkeys: listHotkeys() };
  return JSON.stringify(payload, null, 2);
}

/** Import hotkeys from a string or object { hotkeys: Hotkey[] }. */
export function importHotkeys(payload: string | { hotkeys?: unknown }): Hotkey[] {
  let arr: unknown;
  try {
    const obj = typeof payload === "string" ? JSON.parse(payload) : payload;
    arr = (obj as any)?.hotkeys;
  } catch {}
  return replaceHotkeys(Array.isArray(arr) ? arr : []);
}

/* =========================== Combo Utilities ============================= */

/**
 * Normalize a combo string to a canonical form:
 *   Order: Ctrl, Cmd, Alt, Shift, Key
 *   Examples: "Ctrl+Shift+K", "Cmd+S", "Slash", "F1", "Unassigned"
 */
export function normalizeCombo(combo: string): string {
  if (!combo || combo.toLowerCase() === "unassigned") return "Unassigned";
  const parts = String(combo)
    .split(/[\+\-\s]+/)
    .map((p) => p.trim())
    .filter(Boolean);

  let ctrl = false, cmd = false, alt = false, shift = false, key = "";
  for (const p of parts) {
    const u = p.toLowerCase();
    if (u === "ctrl" || u === "control") ctrl = true;
    else if (u === "cmd" || u === "meta" || u === "command") cmd = true;
    else if (u === "alt" || u === "option") alt = true;
    else if (u === "shift") shift = true;
    else key = normalizeKeyName(p);
  }
  if (!key) return "Unassigned";
  const mods = [];
  
  return [...mods, key].join("+");
}

/** Convert KeyboardEvent to a normalized combo string. */
export function comboFromEvent(e: KeyboardEvent): string {
  const mods = [];
  

  let key = e.key;
  if (key.length === 1) key = /^[a-z]$/i.test(key) ? key.toUpperCase() : key;

  const special: Record<string, string> = {
    " ": "Space",
    "/": "Slash",
    "\\": "Backslash",
    ".": "Period",
    ",": "Comma",
    ";": "Semicolon",
    "'": "Quote",
    "`": "Backquote",
    "-": "Minus",
    "=": "Equal",
    "Escape": "Esc",
    "ArrowUp": "ArrowUp",
    "ArrowDown": "ArrowDown",
    "ArrowLeft": "ArrowLeft",
    "ArrowRight": "ArrowRight",
    "Tab": "Tab",
    "Enter": "Enter",
    "Backspace": "Backspace",
    "Delete": "Delete",
    "Home": "Home",
    "End": "End",
    "PageUp": "PageUp",
    "PageDown": "PageDown",
    "Insert": "Insert",
  };
  if (special[key]) key = special[key];

  const f = key.toUpperCase().match(/^F([1-9]|1[0-2])$/);
  if (f) key = `F${f[1]}`;

  if (["Control", "Meta", "Alt", "Shift"].includes(key)) return "Unassigned";
  return key ? [...mods, key].join("+") : "Unassigned";
}

/** Pretty-print a combo for display (already normalized in practice). */
export function prettyCombo(combo: string): string {
  const k = normalizeCombo(combo);
  return k === "Unassigned" ? "Unassigned" : k;
}

/* ================================ Utils ================================== */

function persist(list: Hotkey[]) {
  safeSet(STORAGE_KEY, JSON.stringify(list));
}

function normalizeList(list: any[]): Hotkey[] {
  const out: Hotkey[] = [];
  const seen = new Set<string>();
  for (const raw of Array.isArray(list) ? list : []) {
    const id = String(raw?.id || genId("hk_"));
    if (seen.has(id)) continue;
    seen.add(id);

    const now = isoNow();
    out.push({
      id,
      label: String(raw?.label || id),
      combo: normalizeCombo(raw?.combo || "Unassigned"),
      category: raw?.category?.trim() || "General",
      description: raw?.description?.trim() || "",
      createdAt: toIsoOrNull(raw?.createdAt) || now,
      updatedAt: toIsoOrNull(raw?.updatedAt) || now,
    });
  }
  return out;
}

function genId(prefix = "") { return prefix + Math.random().toString(36).slice(2, 8); }
function isoNow() { try { return new Date().toISOString(); } catch { return "" + Date.now(); } }
function toIsoOrNull(x: unknown): string | null {
  if (!x) return null;
  const t = Date.parse(String(x));
  return isFinite(t) ? new Date(t).toISOString() : null;
}

function normalizeKeyName(k: string) {
  const up = k.trim();
  const map: Record<string, string> = {
    " ": "Space",
    "Space": "Space",
    "Spacebar": "Space",
    "/": "Slash",
    "\\": "Backslash",
    ".": "Period",
    ",": "Comma",
    ";": "Semicolon",
    "'": "Quote",
    "`": "Backquote",
    "-": "Minus",
    "=": "Equal",
  };
  if (map[up]) return map[up];
  const F = up.toUpperCase().match(/^F([1-9]|1[0-2])$/);
  if (F) return `F${F[1]}`;
  if (/^[A-Za-z]$/.test(up)) return up.toUpperCase();
  if (/^\d$/.test(up)) return up;
  // arrows, nav keys, etc. (TitleCase)
  const title = up.charAt(0).toUpperCase() + up.slice(1);
  return title;
}