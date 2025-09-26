// utils/time.ts
// Pure, dependency-free time/date helpers (TS targets < ES2020 OK). No imports.

export const MS = 1;
export const SEC = 1000 * MS;
export const MIN = 60 * SEC;
export const HOUR = 60 * MIN;
export const DAY = 24 * HOUR;
export const WEEK = 7 * DAY;

/* ------------------------------- basics -------------------------------- */

export function now(): number { return Date.now(); }

export function toISO(ms?: number, utc: boolean = true): string {
  const d = ms != null ? new Date(ms) : new Date();
  return utc ? d.toISOString() : toISOLocal(d);
}

export function fromISO(iso: string): number | undefined {
  const t = Date.parse(iso);
  return isFinite(t) ? t : undefined;
}

export function toISOLocal(d: Date): string {
  const pad2 = (n: number) => (n < 10 ? "0" + n : "" + n);
  const pad3 = (n: number) => (n < 10 ? "00" + n : n < 100 ? "0" + n : "" + n);
  const off = -d.getTimezoneOffset();
  const sign = off >= 0 ? "+" : "-";
  const hh = pad2(Math.floor(Math.abs(off) / 60));
  const mm = pad2(Math.abs(off) % 60);
  return (
    d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate()) +
    "T" + pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds()) +
    "." + pad3(d.getMilliseconds()) + sign + hh + ":" + mm
  );
}

/* ------------------------------- formatting ------------------------------ */

export type DatePattern =
  | "YYYY-MM-DD"
  | "YYYY/MM/DD"
  | "DD-MM-YYYY"
  | "YYYY-MM-DD HH:mm"
  | "YYYY-MM-DD HH:mm:ss"
  | "HH:mm"
  | "HH:mm:ss"
  | string;

/** Tokens: YYYY, MM, DD, HH, mm, ss, SSS. */
export function format(ts: number | Date, pattern: DatePattern = "YYYY-MM-DD HH:mm:ss", utc: boolean = false): string {
  const d = ts instanceof Date ? ts : new Date(ts);
  const g = utc
    ? { Y: d.getUTCFullYear(), M: d.getUTCMonth() + 1, D: d.getUTCDate(), h: d.getUTCHours(), m: d.getUTCMinutes(), s: d.getUTCSeconds(), S: d.getUTCMilliseconds() }
    : { Y: d.getFullYear(),     M: d.getMonth() + 1,     D: d.getDate(),     h: d.getHours(),     m: d.getMinutes(),     s: d.getSeconds(),     S: d.getMilliseconds() };
  const pad2 = (n: number) => (n < 10 ? "0" + n : "" + n);
  const pad3 = (n: number) => (n < 10 ? "00" + n : n < 100 ? "0" + n : "" + n);
  return String(pattern)
    .replace(/YYYY/g, String(g.Y))
    .replace(/MM/g, pad2(g.M))
    .replace(/DD/g, pad2(g.D))
    .replace(/HH/g, pad2(g.h))
    .replace(/mm/g, pad2(g.m))
    .replace(/ss/g, pad2(g.s))
    .replace(/SSS/g, pad3(g.S));
}

/* ---------------------------- duration parsing --------------------------- */

export type ParseDurationOptions = { allowClock?: boolean };

/**
 * Parse human duration strings into milliseconds.
 * Supports "1h 30m", "2d", "500ms", "1.5h", "3m45s", and clocks "1:23", "1:23:45(.250)".
 */
export function parseDuration(input: string, opts?: ParseDurationOptions): number | undefined {
  if (!input) return undefined;
  const s = String(input).trim();

  // unit form
  const re = /(-?\d+(?:\.\d+)?)\s*(ms|msec|s|sec|m|min|h|hr|d|w)\b/gi;
  let ms = 0, matched = false, m: RegExpExecArray | null;
  const mult: Record<string, number> = { ms: MS, msec: MS, s: SEC, sec: SEC, m: MIN, min: MIN, h: HOUR, hr: HOUR, d: DAY, w: WEEK };
  while ((m = re.exec(s))) { matched = true; ms += parseFloat(m[1]) * (mult[m[2].toLowerCase()] || 0); }
  if (matched) return ms;

  // clock form
  const allowClock = opts && typeof opts.allowClock === "boolean" ? !!opts.allowClock : true;
  if (allowClock && /^\d{1,2}:\d{2}(:\d{2}(\.\d{1,3})?)?$/.test(s)) {
    const parts = s.split(":");
    let h = 0, mm = 0, ss = 0, milli = 0;
    if (parts.length === 2) {
      h = parseInt(parts[0], 10) || 0; mm = parseInt(parts[1], 10) || 0;
    } else {
      h = parseInt(parts[0], 10) || 0;
      mm = parseInt(parts[1], 10) || 0;
      const sp = parts[2].split(".");
      ss = parseInt(sp[0], 10) || 0;
      milli = sp[1] ? parseInt(sp[1].slice(0, 3).padEnd(3, "0"), 10) : 0;
    }
    return h * HOUR + mm * MIN + ss * SEC + milli;
  }

  // numeric: ms
  const n = Number(s);
  return isFinite(n) ? n : undefined;
}

export type FormatDurationOptions = {
  style?: "compact" | "colon" | "long";
  maxUnits?: number;
  includeMs?: boolean;
  forceHours?: boolean;
};

/** Format milliseconds nicely. Negative values get a leading "-". */
export function formatDuration(ms: number, opts?: FormatDurationOptions): string {
  const style = (opts && opts.style) || "compact";
  const maxUnits = (opts && opts.maxUnits) || 2;
  const includeMs = !!(opts && opts.includeMs);
  const forceH = !!(opts && opts.forceHours);

  const neg = ms < 0; const t = Math.abs(ms | 0);
  const d = Math.floor(t / DAY);
  const h = Math.floor((t % DAY) / HOUR);
  const m = Math.floor((t % HOUR) / MIN);
  const s = Math.floor((t % MIN) / SEC);
  const milli = t % 1000;

  if (style === "colon") {
    const pad2 = (n: number) => (n < 10 ? "0" + n : "" + n);
    const base = (forceH || d > 0 || h > 0 ? pad2(d * 24 + h) + ":" : "") + pad2(m) + ":" + pad2(s);
    const msPart = includeMs ? "." + (milli < 10 ? "00" + milli : milli < 100 ? "0" + milli : "" + milli) : "";
    return (neg ? "-" : "") + base + msPart;
  }

  const parts: string[] = [];
  const push = (v: number, u: string, long: string) => {
    if (!v) return;
    parts.push(style === "long" ? v + " " + (v === 1 ? long : long + "s") : v + u);
  };
  push(d, "d", "day"); push(h, "h", "hour"); push(m, "m", "minute"); push(s, "s", "second");
  if (includeMs && (milli || parts.length === 0)) push(milli, "ms", "millisecond");

  const shown = parts.slice(0, Math.max(1, maxUnits));
  return (neg ? "-" : "") + (shown.length ? shown.join(" ") : (style === "long" ? "0 seconds" : "0s"));
}

/* ---------------------------- calendar helpers --------------------------- */

export function startOfDay(ts?: number, utc: boolean = false): number {
  const d = ts != null ? new Date(ts) : new Date();
  return utc
    ? Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 0, 0, 0, 0)
    : new Date(d.getFullYear(), d.getMonth(), d.getDate(), 0, 0, 0, 0).getTime();
}
export function endOfDay(ts?: number, utc: boolean = false): number {
  return startOfDay(ts, utc) + DAY - 1;
}
export function add(ms: number, opts: { days?: number; hours?: number; minutes?: number; seconds?: number; millis?: number }): number {
  return ms + (opts.days || 0) * DAY + (opts.hours || 0) * HOUR + (opts.minutes || 0) * MIN + (opts.seconds || 0) * SEC + (opts.millis || 0);
}
export function diff(a: number, b: number, unit: "ms" | "s" | "m" | "h" | "d" = "ms"): number {
  const raw = a - b;
  return unit === "s" ? raw / SEC : unit === "m" ? raw / MIN : unit === "h" ? raw / HOUR : unit === "d" ? raw / DAY : raw;
}
export function isSameDay(a: number, b: number, utc: boolean = false): boolean {
  const da = new Date(a), db = new Date(b);
  return utc
    ? (da.getUTCFullYear() === db.getUTCFullYear() && da.getUTCMonth() === db.getUTCMonth() && da.getUTCDate() === db.getUTCDate())
    : (da.getFullYear() === db.getFullYear() && da.getMonth() === db.getMonth() && da.getDate() === db.getDate());
}

/* ----------------------------- relative time ----------------------------- */

export function relativeTime(targetMs: number, baseMs?: number): string {
  const base = baseMs != null ? baseMs : Date.now();
  const delta = targetMs - base;
  const ad = Math.abs(delta);
  const choose = (n: number, u: string) => (delta >= 0 ? "in " + n + u : n + u + " ago");

  if (ad < 45 * SEC) return choose(Math.round(ad / SEC) || 0, "s");
  if (ad < 45 * MIN) return choose(Math.round(ad / MIN), "m");
  if (ad < 22 * HOUR) return choose(Math.round(ad / HOUR), "h");
  if (ad < 26 * DAY) return choose(Math.round(ad / DAY), "d");
  if (ad < 11 * WEEK) return choose(Math.round(ad / WEEK), "w");
  return (delta >= 0 ? "on " : "") + format(targetMs, "YYYY-MM-DD", true);
}

/* ---------------------------- async conveniences ------------------------- */

export function sleep(msDelay: number): Promise<void> {
  return new Promise(function (res) { setTimeout(res, Math.max(0, msDelay | 0)); });
}
export const delay = sleep;

export function withTimeout<T>(p: Promise<T>, msDelay: number, message?: string): Promise<T> {
  return new Promise(function (resolve, reject) {
    const t = setTimeout(function () { reject(new Error(message || "Timeout after " + msDelay + "ms")); }, Math.max(0, msDelay | 0));
    p.then(function (v) { clearTimeout(t); resolve(v); }, function (e) { clearTimeout(t); reject(e); });
  });
}

/* --------------------------- debounce / throttle ------------------------- */

export type Debounced<F extends (...args: any[]) => any> =
  ((...args: Parameters<F>) => void) & { cancel: () => void; flush: () => void };

export function debounce<F extends (...args: any[]) => any>(
  fn: F,
  wait: number,
  immediate?: boolean
): Debounced<F> {
  let timer: any = null;
  let lastArgs: any[] | null = null;
  let lastThis: any = null;

  function wrapper(this: any, ...args: any[]) {
    lastArgs = args; lastThis = this;
    const callNow = immediate && !timer;
    if (timer) clearTimeout(timer);
    timer = setTimeout(function () {
      timer = null;
      if (!immediate && lastArgs) { fn.apply(lastThis, lastArgs); lastArgs = null; lastThis = null; }
    }, Math.max(0, wait | 0));
    if (callNow) { fn.apply(lastThis, lastArgs as any); lastArgs = null; lastThis = null; }
  }

  (wrapper as any).cancel = function () { if (timer) { clearTimeout(timer); timer = null; } lastArgs = null; lastThis = null; };
  (wrapper as any).flush = function () { if (timer) { clearTimeout(timer); timer = null; } if (lastArgs) { fn.apply(lastThis, lastArgs); lastArgs = null; lastThis = null; } };

  return wrapper as Debounced<F>;
}

export type Throttled<F extends (...args: any[]) => any> =
  ((...args: Parameters<F>) => void) & { cancel: () => void };

export function throttle<F extends (...args: any[]) => any>(
  fn: F,
  wait: number,
  leading: boolean = true,
  trailing: boolean = true
): Throttled<F> {
  let timer: any = null;
  let lastExec = 0;
  let pendingArgs: any[] | null = null;
  let pendingThis: any = null;

  function invoke() {
    lastExec = Date.now();
    fn.apply(pendingThis, pendingArgs as any);
    pendingArgs = null; pendingThis = null;
  }

  function wrapper(this: any, ...args: any[]) {
    const nowTs = Date.now();
    if (!lastExec && !leading) lastExec = nowTs;
    const remaining = wait - (nowTs - lastExec);
    pendingArgs = args; pendingThis = this;

    if (remaining <= 0 || remaining > wait) {
      if (timer) { clearTimeout(timer); timer = null; }
      invoke();
    } else if (trailing && !timer) {
      timer = setTimeout(function () { timer = null; if (trailing && pendingArgs) invoke(); }, remaining);
    }
  }

  (wrapper as any).cancel = function () { if (timer) { clearTimeout(timer); timer = null; } lastExec = 0; pendingArgs = null; pendingThis = null; };

  return wrapper as Throttled<F>;
}

/* -------------------------------- exports -------------------------------- */

const time = {
  MS, SEC, MIN, HOUR, DAY, WEEK,
  now, toISO, fromISO, toISOLocal, format,
  parseDuration, formatDuration,
  startOfDay, endOfDay, add, diff, isSameDay,
  relativeTime, sleep, delay, withTimeout,
  debounce, throttle,
};

export default time;