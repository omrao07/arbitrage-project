// lib/env.ts
// Pure env helpers (no imports). Safe defaults; works in Node/Workers.

/* ================================ helpers ================================ */

function getEnv(name: string): string | undefined {
  try {
    const g: any = (globalThis as any);
    if (g && g.process && g.process.env && typeof g.process.env[name] === "string") {
      return g.process.env[name] as string;
    }
  } catch { /* ignore */ }
  return undefined;
}

export function str(name: string, def = ""): string {
  const v = getEnv(name);
  return v != null ? String(v) : def;
}

export function bool(name: string, def = false): boolean {
  const v = getEnv(name);
  if (v == null) return def;
  const s = String(v).trim().toLowerCase();
  if (s === "1" || s === "true" || s === "yes" || s === "on") return true;
  if (s === "0" || s === "false" || s === "no" || s === "off") return false;
  return def;
}

export function num(name: string, def: number): number {
  const v = Number(getEnv(name));
  return Number.isFinite(v) ? v : def;
}

export function json<T = unknown>(name: string, def: T): T {
  const v = getEnv(name);
  if (v == null || v.trim() === "") return def;
  try { return JSON.parse(v) as T; } catch { return def; }
}

export function list(name: string, def: string[] = []): string[] {
  const v = getEnv(name);
  if (v == null) return def;
  return v.split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
}

/* -------- optional: duration parsing ("5m", "2h", "1500ms", "1:30:00") -------- */

function parseDurationMs(input?: string): number | undefined {
  if (!input) return undefined;
  const s = String(input).trim();
  // unit form
  const re = /(-?\d+(?:\.\d+)?)\s*(ms|msec|s|sec|m|min|h|hr|d|w)\b/gi;
  let ms = 0, matched = false, m: RegExpExecArray | null;
  const mult: Record<string, number> = {
    ms: 1, msec: 1,
    s: 1000, sec: 1000,
    m: 60_000, min: 60_000,
    h: 3_600_000, hr: 3_600_000,
    d: 86_400_000,
    w: 604_800_000,
  };
  while ((m = re.exec(s))) { matched = true; ms += parseFloat(m[1]) * (mult[m[2].toLowerCase()] || 0); }
  if (matched) return ms;

  // clock H:MM[:SS[.mmm]]
  if (/^\d{1,2}:\d{2}(:\d{2}(\.\d{1,3})?)?$/.test(s)) {
    const parts = s.split(":");
    let h = 0, mm = 0, ss = 0, milli = 0;
    if (parts.length === 2) {
      h = parseInt(parts[0], 10) || 0;
      mm = parseInt(parts[1], 10) || 0;
    } else {
      h = parseInt(parts[0], 10) || 0;
      mm = parseInt(parts[1], 10) || 0;
      const sp = parts[2].split(".");
      ss = parseInt(sp[0], 10) || 0;
      milli = sp[1] ? parseInt(sp[1].slice(0, 3).padEnd(3, "0"), 10) : 0;
    }
    return h * 3_600_000 + mm * 60_000 + ss * 1000 + milli;
  }

  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}

export function duration(name: string, defMs: number): number {
  const v = getEnv(name);
  if (v == null || v.trim() === "") return defMs;
  const ms = parseDurationMs(v);
  return ms != null ? ms : defMs;
}

/* ================================ config ================================ */

export const cfg = {
  // feature toggles
  newsEnabled: bool("WORKERS_NEWS_ENABLED", true),
  alertsEnabled: bool("WORKERS_ALERTS_ENABLED", true),
  stressEnabled: bool("WORKERS_STRESS_ENABLED", true),

  // schedules (accept "300000", "5m", "00:05:00")
  newsEveryMs: duration("NEWS_INTERVAL_MS", 5 * 60_000),
  alertsEveryMs: duration("ALERTS_INTERVAL_MS", 60_000),
  stressEveryMs: duration("STRESS_INTERVAL_MS", 15 * 60_000),

  // logging
  logLevel: str("LOG_LEVEL", "info"),
  logName:  str("LOG_NAME", "workers"),

  // misc buckets for extension
  extra: json<Record<string, unknown>>("WORKERS_EXTRA_JSON", {}),
  allowlistTopics: list("WORKERS_TOPICS", []),
};

export default cfg;