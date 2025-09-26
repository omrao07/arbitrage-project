/**
 * actions/storage.ts
 * Tiny wrapper for safe JSON storage (browser localStorage or Node fallback).
 *
 * - No external deps. Works everywhere.
 * - Provides get/set/remove/clear for JSON values.
 * - Allows custom storage driver injection (e.g., API backend).
 */

/* ================================ Types ================================== */

export type StorageDriver = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  clear(): void;
};

const memoryStore: Record<string, string> = {};

const memoryDriver: StorageDriver = {
  getItem: (k) => (k in memoryStore ? memoryStore[k] : null),
  setItem: (k, v) => { memoryStore[k] = v; },
  removeItem: (k) => { delete memoryStore[k]; },
  clear: () => { for (const k in memoryStore) delete memoryStore[k]; },
};

const localDriver: StorageDriver | null = (() => {
  try {
    if (typeof localStorage !== "undefined") {
      return {
        getItem: (k) => localStorage.getItem(k),
        setItem: (k, v) => localStorage.setItem(k, v),
        removeItem: (k) => localStorage.removeItem(k),
        clear: () => localStorage.clear(),
      };
    }
  } catch {}
  return null;
})();

let driver: StorageDriver = localDriver || memoryDriver;

/** Override the storage driver (e.g., to use an API). */
export function setStorageDriver(custom: StorageDriver | null | undefined) {
  driver = custom || localDriver || memoryDriver;
}

/* ============================== JSON Helpers ============================= */

/** Load JSON value or return fallback. */
export function loadJSON<T>(key: string, fallback: T): T {
  try {
    const raw = driver.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

/** Save JSON value. */
export function saveJSON<T>(key: string, value: T): void {
  try {
    driver.setItem(key, JSON.stringify(value));
  } catch {}
}

/** Remove a key. */
export function removeJSON(key: string): void {
  try { driver.removeItem(key); } catch {}
}

/** Clear all keys. */
export function clearStorage(): void {
  try { driver.clear(); } catch {}
}

/* ============================== Key Constants ============================= */

export const STORAGE_KEYS = {
  profile: "settings:profile",
  connections: "settings:connections",
  flags: "settings:flags",
  hotkeys: "settings:hotkeys",
  layouts: "settings:layouts",
};