/**
 * actions/types.ts
 * Shared type definitions for all settings/actions modules.
 */

/* =============================== Results ================================ */

export type Result<T> = { ok: true; data: T } | { ok: false; error: string };

/* =============================== Profile ================================ */

export type ThemeMode = "system" | "light" | "dark";

export type Profile = {
  name: string;
  handle: string;              // user handle: a-z0-9_.-
  email: string;
  avatarDataUrl?: string;      // base64 (optional)
  timezone: string;            // IANA TZ
  locale: string;              // e.g. en-US
  theme: ThemeMode;
  createdAt?: string;          // ISO
  updatedAt?: string;          // ISO
};

/* ============================== Connections ============================= */

export type ConnectionType = "API" | "DB" | "File" | "Other";
export type ConnectionStatus = "connected" | "disconnected";

export type Connection = {
  id: string;
  name: string;
  type: ConnectionType;
  status: ConnectionStatus;
  meta?: Record<string, unknown>;
  createdAt?: string;
  updatedAt?: string;
};

/* ================================ Flags ================================= */

export type Flag = {
  key: string;
  label?: string;
  description?: string;
  category?: string;
  default: boolean;
  on: boolean;
  createdAt?: string;
  updatedAt?: string;
};

/* =============================== Hotkeys ================================ */

export type Hotkey = {
  id: string;
  label: string;
  combo: string;                 // normalized combo string
  category?: string;
  description?: string;
  createdAt?: string;
  updatedAt?: string;
};

/* =============================== Layouts ================================ */

export type LayoutPreset = {
  id: string;
  name: string;
  icon?: string;
  layout: any;
  createdAt?: string;
  updatedAt?: string;
};

/* ============================ Settings Events =========================== */

export type SettingsEvent =
  | { type: "profile/saved"; payload: Profile }
  | { type: "profile/changed"; payload: Profile }
  | { type: "profile/cleared" }
  | { type: "profile/passwordChanged" }
  | { type: "connections/updated"; payload: Connection[] }
  | { type: "connections/added"; payload: Connection }
  | { type: "connections/changed"; payload: Connection }
  | { type: "connections/removed"; payload: { id: string } }
  | { type: "connections/reordered"; payload: Connection[] }
  | { type: "flags/updated"; payload: Flag[] }
  | { type: "flags/added"; payload: Flag }
  | { type: "flags/changed"; payload: Flag }
  | { type: "flags/removed"; payload: { key: string } }
  | { type: "hotkeys/updated"; payload: Hotkey[] }
  | { type: "hotkeys/added"; payload: Hotkey }
  | { type: "hotkeys/changed"; payload: Hotkey }
  | { type: "hotkeys/removed"; payload: { id: string } }
  | { type: "hotkeys/reordered"; payload: Hotkey[] }
  | { type: "layouts/updated"; payload: LayoutPreset[] }
  | { type: "layouts/added"; payload: LayoutPreset }
  | { type: "layouts/changed"; payload: LayoutPreset }
  | { type: "layouts/removed"; payload: { id: string } }
  | { type: "layouts/loaded"; payload: LayoutPreset }
  | { type: "layouts/reordered"; payload: LayoutPreset[] };