/**
 * actions/profile.ts
 * Pure, import-free utilities to manage a single user profile with persistence + events.
 *
 * - No external deps. Works in browser (localStorage) and Node (memory fallback).
 * - Read/update/clear, password change placeholder, import/export.
 * - Strong normalization + validation with friendly helpers.
 *
 * Events emitted (via dispatcher):
 *  - { type: "profile/saved", payload: Profile }
 *  - { type: "profile/changed", payload: Profile }
 *  - { type: "profile/cleared" }
 *  - { type: "profile/passwordChanged" }
 */

import { emitSettingsEvent } from "./dispatcher";

/* ================================ Types ================================== */

export type ThemeMode = "system" | "light" | "dark";

export type Profile = {
  name: string;
  handle: string;          // a-z 0-9 _ . -
  email: string;
  avatarDataUrl?: string;  // base64 data URL (client-side only)
  timezone: string;        // IANA TZ
  locale: string;          // e.g. en-US
  theme: ThemeMode;
  createdAt?: string;      // ISO
  updatedAt?: string;      // ISO
};

export type Result<T> = { ok: true; data: T } | { ok: false; error: string };

const STORAGE_KEY = "settings:profile:v1";

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
function safeDel(k: string) {
  try { if (typeof localStorage !== "undefined") return localStorage.removeItem(k); } catch {}
  delete memoryStore[k];
}

/* ================================ Core =================================== */

/** Get the saved profile, or a normalized default if none exists. */
export function getProfile(): Profile {
  try {
    const raw = safeGet(STORAGE_KEY);
    if (!raw) return seed();
    const parsed = JSON.parse(raw);
    return normalize(parsed);
  } catch {
    return seed();
  }
}

/** Replace the entire profile after validating it. */
export function saveProfile(p: Profile): Result<Profile> {
  const norm = normalize(p);
  const err = validate(norm);
  if (err) return { ok: false, error: err };
  persist(norm, /*emitChange*/ true, /*saved*/ true);
  return { ok: true, data: norm };
}

/** Patch specific fields; returns the updated profile. */
export function updateProfile(patch: Partial<Profile>): Result<Profile> {
  const merged = { ...getProfile(), ...patch };
  const norm = normalize(merged);
  const err = validate(norm);
  if (err) return { ok: false, error: err };
  persist(norm, /*emitChange*/ true);
  return { ok: true, data: norm };
}

/** Clear the stored profile (reverts to defaults). */
export function clearProfile(): void {
  safeDel(STORAGE_KEY);
  emitSettingsEvent({ type: "profile/cleared" });
}

/** Convenience setters */
export function setAvatarDataUrl(dataUrl: string | undefined | null): Result<Profile> {
  return updateProfile({ avatarDataUrl: dataUrl || "" });
}
export function setTheme(theme: ThemeMode): Result<Profile> {
  return updateProfile({ theme: normalizeTheme(theme) });
}

/**
 * Change password placeholder.
 * NOTE: This only validates locally; wire to your backend in real apps.
 */
export async function changePassword(current: string, next: string, confirm: string): Promise<Result<true>> {
  if (!current || !next || !confirm) return { ok: false, error: "All fields are required." };
  if (next.length < 8) return { ok: false, error: "Password must be at least 8 characters." };
  if (next !== confirm) return { ok: false, error: "Passwords do not match." };
  emitSettingsEvent({ type: "profile/passwordChanged" });
  return { ok: true, data: true };
}

/** Export as pretty JSON. */
export function exportProfile(): string {
  const payload = { ts: isoNow(), profile: getProfile() };
  return JSON.stringify(payload, null, 2);
}

/** Import from string or object; invalid fields are ignored. */
export function importProfile(payload: string | { profile?: unknown }): Result<Profile> {
  let obj: unknown;
  try {
    const parsed = typeof payload === "string" ? JSON.parse(payload) : payload;
    obj = (parsed as any)?.profile ?? parsed;
  } catch {
    // ignore
  }
  const norm = normalize(obj);
  const err = validate(norm);
  if (err) return { ok: false, error: err };
  persist(norm, /*emitChange*/ true, /*saved*/ true);
  return { ok: true, data: norm };
}

/* ================================ Utils ================================== */

function persist(p: Profile, emitChange = false, saved = false) {
  const withTimestamps: Profile = { ...p, updatedAt: isoNow(), createdAt: p.createdAt || isoNow() };
  safeSet(STORAGE_KEY, JSON.stringify(withTimestamps));
  if (emitChange) {
    emitSettingsEvent({ type: saved ? "profile/saved" : "profile/changed", payload: withTimestamps });
  }
}

function seed(): Profile {
  const now = isoNow();
  return {
    name: "",
    handle: "",
    email: "",
    avatarDataUrl: "",
    timezone: guessTZ(),
    locale: guessLocale(),
    theme: "system",
    createdAt: now,
    updatedAt: now,
  };
}

function normalize(x: any): Profile {
  const now = isoNow();
  const theme = normalizeTheme(x?.theme);
  const handle = sanitizeHandle(x?.handle);
  const email = String(x?.email ?? "");
  return {
    name: String(x?.name ?? "").trim(),
    handle,
    email,
    avatarDataUrl: typeof x?.avatarDataUrl === "string" ? x.avatarDataUrl : "",
    timezone: String(x?.timezone || guessTZ()),
    locale: String(x?.locale || guessLocale()),
    theme,
    createdAt: toIsoOrNull(x?.createdAt) || now,
    updatedAt: toIsoOrNull(x?.updatedAt) || now,
  };
}

function validate(p: Profile): string | null {
  if (!p.name.trim()) return "Name is required.";
  if (!/^[a-z0-9_][a-z0-9_.\-]{1,31}$/i.test(p.handle || "")) {
    return "Handle must be 2–32 chars (letters, numbers, _ . -) and start with a letter/number/_";
  }
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(p.email || "")) return "Invalid email address.";
  return null;
}

function sanitizeHandle(v: unknown): string {
  const s = String(v ?? "").replace(/[^a-z0-9_.\-]/gi, "").slice(0, 32);
  // ensure first char is [A-Za-z0-9_]; if not, prefix underscore
  return /^[A-Za-z0-9_]/.test(s) ? s : (s ? "_" + s : "");
}

function normalizeTheme(t: unknown): ThemeMode {
  const v = String(t ?? "system").toLowerCase();
  return v === "light" ? "light" : v === "dark" ? "dark" : "system";
}

function guessTZ() { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch { return "UTC"; } }
function guessLocale() { try { return navigator.language || "en-US"; } catch { return "en-US"; } }
function isoNow() { try { return new Date().toISOString(); } catch { return "" + Date.now(); } }
function toIsoOrNull(x: unknown): string | null {
  if (!x) return null;
  const t = Date.parse(String(x));
  return isFinite(t) ? new Date(t).toISOString() : null;
}