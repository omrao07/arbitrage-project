// actions/savealert.action.ts (or app/actions/savealert.action.ts)
"use server";

/**
 * A dependency-free Server Action to save a market alert.
 * - Accepts either a typed object or FormData from a <form action={saveAlertAction}>.
 * - No external imports; uses in-memory store (replace persistAlert with your DB).
 */

type Condition =
  | ">"
  | "<"
  | ">="
  | "<="
  | "=="
  | "crosses_up"
  | "crosses_down";

type Channel = "ui" | "email" | "sms" | "webhook";

type Recurrence = "once" | "every_time" | "once_per_day";

export type AlertInput = {
  symbol: string;              // e.g., "RELIANCE", "NIFTY", "USD/INR"
  condition: Condition;        // e.g., ">="
  value: number;               // threshold (price/level)
  note?: string;
  channel?: Channel;           // default "ui"
  webhookUrl?: string;         // used when channel==="webhook"
  expiresAt?: string;          // ISO string
  recurrence?: Recurrence;     // default "once"
  userId?: string;             // provide from session; else "anon"
  metadata?: Record<string, any>; // free-form
};

export type Alert = {
  id: string;
  userId: string;
  symbol: string;
  condition: Condition;
  value: number;
  note?: string;
  channel: Channel;
  webhookUrl?: string;
  expiresAt?: string; // ISO
  recurrence: Recurrence;
  status: "active" | "paused" | "expired";
  createdAt: string; // ISO
  updatedAt: string; // ISO
  metadata?: Record<string, any>;
};

export type ActionResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: string; fieldErrors?: Record<string, string> };

/* ---------------- In-memory store (replace with your DB) ---------------- */
declare global {
  // eslint-disable-next-line no-var
  var __ALERT_STORE: Map<string, Alert[]> | undefined;
}
if (!globalThis.__ALERT_STORE) {
  globalThis.__ALERT_STORE = new Map();
}
const STORE = globalThis.__ALERT_STORE!;

/* ---------------- Public server action ---------------- */
export async function saveAlertAction(
  input: Partial<AlertInput> | FormData,
): Promise<ActionResult<Alert>> {
  try {
    const parsed = parseInput(input);
    const val = validateInput(parsed);
    if (!val.valid) {
      return { ok: false, error: "Validation failed", fieldErrors: val.errors };
    }

    const nowIso = new Date().toISOString();
    const id = (globalThis.crypto?.randomUUID?.() ?? randomId());
    const alert: Alert = {
      id,
      userId: parsed.userId || "anon",
      symbol: normalizeSymbol(parsed.symbol!),
      condition: parsed.condition!,
      value: Number(parsed.value),
      note: parsed.note?.slice(0, 280),
      channel: parsed.channel || "ui",
      webhookUrl: parsed.webhookUrl,
      expiresAt: parsed.expiresAt,
      recurrence: parsed.recurrence || "once",
      status: "active",
      createdAt: nowIso,
      updatedAt: nowIso,
      metadata: parsed.metadata,
    };

    await persistAlert(alert);
    return { ok: true, data: alert };
  } catch (e: any) {
    return { ok: false, error: e?.message || "Failed to save alert" };
  }
}

/* ---------------- Helpers ---------------- */
function parseInput(input: Partial<AlertInput> | FormData): Partial<AlertInput> {
  if (isFormData(input)) {
    const fd = input as FormData;
    const obj: Partial<AlertInput> = {
      symbol: str(fd.get("symbol")),
      condition: str(fd.get("condition")) as Condition,
      value: num(fd.get("value")),
      note: str(fd.get("note")),
      channel: (str(fd.get("channel")) as Channel) || "ui",
      webhookUrl: str(fd.get("webhookUrl")),
      expiresAt: iso(str(fd.get("expiresAt"))),
      recurrence: (str(fd.get("recurrence")) as Recurrence) || "once",
      userId: str(fd.get("userId")) || "anon",
      metadata: tryJson(str(fd.get("metadata"))),
    };
    return obj;
  }
  const obj = input as Partial<AlertInput>;
  // Coerce types safely
  return {
    ...obj,
    symbol: obj.symbol ? String(obj.symbol) : undefined,
    condition: obj.condition as Condition | undefined,
    value: obj.value != null ? Number(obj.value) : (undefined as any),
    note: obj.note ? String(obj.note) : undefined,
    channel: (obj.channel as Channel) || "ui",
    webhookUrl: obj.webhookUrl ? String(obj.webhookUrl) : undefined,
    expiresAt: obj.expiresAt ? iso(String(obj.expiresAt)) : undefined,
    recurrence: (obj.recurrence as Recurrence) || "once",
    userId: obj.userId ? String(obj.userId) : "anon",
    metadata: typeof obj.metadata === "string" ? tryJson(obj.metadata) : obj.metadata,
  };
}

function validateInput(input: Partial<AlertInput>): {
  valid: boolean;
  errors?: Record<string, string>;
} {
  const errors: Record<string, string> = {};

  const symbol = input.symbol?.trim();
  if (!symbol) {
    errors.symbol = "Symbol is required.";
  } else if (!/^[A-Z0-9/\-_.]{1,32}$/i.test(symbol)) {
    errors.symbol = "Invalid symbol format.";
  }

  const cond = input.condition;
  if (!cond || !isCondition(cond)) {
    errors.condition = "Invalid or missing condition.";
  }

  const value = Number(input.value);
  if (!Number.isFinite(value)) {
    errors.value = "Value must be a finite number.";
  }

  if (input.channel && !isChannel(input.channel)) {
    errors.channel = "Invalid channel.";
  }

  if (input.channel === "webhook" && input.webhookUrl) {
    try {
      const u = new URL(input.webhookUrl);
      if (!/^https?:$/.test(u.protocol)) {
        errors.webhookUrl = "Webhook must be http(s).";
      }
    } catch {
      errors.webhookUrl = "Invalid webhook URL.";
    }
  }

  if (input.expiresAt) {
    const t = Date.parse(input.expiresAt);
    if (Number.isNaN(t)) {
      errors.expiresAt = "expiresAt must be ISO datetime.";
    } else if (t <= Date.now()) {
      errors.expiresAt = "expiresAt must be in the future.";
    }
  }

  return { valid: Object.keys(errors).length === 0, errors };
}

async function persistAlert(alert: Alert): Promise<void> {
  const list = STORE.get(alert.userId) ?? [];
  // upsert by ID if already exists
  const idx = list.findIndex((a) => a.id === alert.id);
  const arr = idx >= 0 ? (list.splice(idx, 1, alert), list) : (list.push(alert), list);
  STORE.set(alert.userId, arr);
}

function normalizeSymbol(s: string): string {
  return s.trim().toUpperCase();
}

function isFormData(x: any): x is FormData {
  return typeof x === "object" && x?.constructor?.name === "FormData";
}

function str(v: FormDataEntryValue | null): string | undefined {
  if (v == null) return undefined;
  return String(v);
}

function num(v: FormDataEntryValue | null): number | undefined {
  if (v == null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function iso(v?: string): string | undefined {
  if (!v) return undefined;
  const t = Date.parse(v);
  return Number.isNaN(t) ? undefined : new Date(t).toISOString();
}

function tryJson(s?: string): Record<string, any> | undefined {
  if (!s) return undefined;
  try {
    const o = JSON.parse(s);
    return typeof o === "object" && o ? o : undefined;
  } catch {
    return undefined;
  }
}

function randomId(): string {
  // fallback if crypto.randomUUID is unavailable
  return "alrt_" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

function isCondition(x: any): x is Condition {
  return (
    x === ">" ||
    x === "<" ||
    x === ">=" ||
    x === "<=" ||
    x === "==" ||
    x === "crosses_up" ||
    x === "crosses_down"
  );
}

function isChannel(x: any): x is Channel {
  return x === "ui" || x === "email" || x === "sms" || x === "webhook";
}

/* ---------------- Optional: tiny query helpers (no imports) ---------------- */
export async function listAlertsAction(userId = "anon"): Promise<Alert[]> {
  return STORE.get(userId) ?? [];
}

export async function pauseAlertAction(id: string, userId = "anon"): Promise<ActionResult<Alert>> {
  const list = STORE.get(userId) ?? [];
  const idx = list.findIndex((a) => a.id === id);
  if (idx < 0) return { ok: false, error: "Alert not found" };
  list[idx] = { ...list[idx], status: "paused", updatedAt: new Date().toISOString() };
  STORE.set(userId, list);
  return { ok: true, data: list[idx] };
}

export async function resumeAlertAction(id: string, userId = "anon"): Promise<ActionResult<Alert>> {
  const list = STORE.get(userId) ?? [];
  const idx = list.findIndex((a) => a.id === id);
  if (idx < 0) return { ok: false, error: "Alert not found" };
  list[idx] = { ...list[idx], status: "active", updatedAt: new Date().toISOString() };
  STORE.set(userId, list);
  return { ok: true, data: list[idx] };
}

export async function deleteAlertAction(id: string, userId = "anon"): Promise<ActionResult<null>> {
  const list = STORE.get(userId) ?? [];
  const next = list.filter((a) => a.id !== id);
  if (next.length === list.length) return { ok: false, error: "Alert not found" };
  STORE.set(userId, next);
  return { ok: true, data: null };
}
