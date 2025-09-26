// utils/logger.ts
// Pure, dependency-free logger for browser/Node.
// - Levels: trace, debug, info, warn, error, fatal
// - Pretty or JSON output
// - Namespaces (per-logger "name")
// - Child loggers with merged context
// - Timers: log.time('label') -> done(extra?)
// - In-memory ring buffer (history)
// No imports. Works on old TS targets (no BigInt, no Node types required).

/* ================================= Types ================================ */

export type LogLevelName = "trace" | "debug" | "info" | "warn" | "error" | "fatal" | "silent";

export type LogRecord = {
  time: string;               // ISO timestamp
  ts: number;                 // epoch ms
  level: LogLevelName;
  name?: string;
  msg?: string;
  data?: Record<string, unknown>;
};

export type LoggerOptions = {
  name?: string;              // namespace
  level?: LogLevelName;       // default from env or "info"
  json?: boolean;             // force JSON lines (default from env LOG_JSON)
  pretty?: boolean;           // force pretty text (default true for TTY)
  colors?: boolean;           // enable ANSI colors (default true for TTY)
  enabledNamespaces?: string[]; // only log when logger.name starts with any of these
  useUTC?: boolean;           // pretty mode timestamp uses UTC (default true)
  bufferSize?: number;        // history ring buffer size (default 200)
  // static fields added to every log from this logger
  context?: Record<string, unknown>;
};

export type Timer = (extra?: Record<string, unknown>) => void;

export type Logger = {
  // basic
  level(): LogLevelName;
  setLevel(lvl: LogLevelName): void;
  isEnabled(lvl: LogLevelName): boolean;

  // log
  log(lvl: LogLevelName, msg: string, data?: Record<string, unknown>): void;
  trace(msg: string, data?: Record<string, unknown>): void;
  debug(msg: string, data?: Record<string, unknown>): void;
  info(msg: string, data?: Record<string, unknown>): void;
  warn(msg: string, data?: Record<string, unknown>): void;
  error(msg: string, data?: Record<string, unknown>): void;
  fatal(msg: string, data?: Record<string, unknown>): void;

  // helpers
  child(nameOrCtx?: string | Record<string, unknown>): Logger;
  time(label: string, base?: Record<string, unknown>, level?: LogLevelName): Timer;
  history(): LogRecord[]; // returns a copy of the ring buffer
  name(): string | undefined;
};

/* ============================ Implementation ============================ */

const LEVELS: LogLevelName[] = ["trace", "debug", "info", "warn", "error", "fatal", "silent"];
const LEVEL_NUM: Record<LogLevelName, number> = {
  trace: 10, debug: 20, info: 30, warn: 40, error: 50, fatal: 60, silent: 1000,
};

function env(k: string): string | undefined {
  try {
    const g: any = (globalThis as any);
    if (g && g.process && g.process.env && typeof g.process.env[k] === "string") {
      return g.process.env[k];
    }
    // Deno
    if (g && g.Deno && g.Deno.env && typeof g.Deno.env.get === "function") {
      return g.Deno.env.get(k);
    }
  } catch { /* ignore */ }
  return undefined;
}

function isTTY(): boolean {
  try {
    const g: any = (globalThis as any);
    return !!(g && g.process && g.process.stdout && g.process.stdout.isTTY);
  } catch { return false; }
}

function parseBool(v?: string): boolean | undefined {
  if (v == null) return undefined;
  const s = String(v).trim().toLowerCase();
  if (s === "1" || s === "true" || s === "yes" || s === "on") return true;
  if (s === "0" || s === "false" || s === "no" || s === "off") return false;
  return undefined;
}

function parseLevel(v?: string): LogLevelName | undefined {
  if (!v) return undefined;
  const s = String(v).toLowerCase();
  for (let i = 0; i < LEVELS.length; i++) if (LEVELS[i] === s) return LEVELS[i];
  return undefined;
}

function nowMs(): number { return Date.now(); }

function iso(ms: number): string { return new Date(ms).toISOString(); }

function pad2(n: number) { return n < 10 ? "0" + n : "" + n; }

function localTimestamp(ms: number, useUTC: boolean): string {
  const d = new Date(ms);
  if (useUTC) {
    return (
      d.getUTCFullYear() + "-" + pad2(d.getUTCMonth() + 1) + "-" + pad2(d.getUTCDate()) + " " +
      pad2(d.getUTCHours()) + ":" + pad2(d.getUTCMinutes()) + ":" + pad2(d.getUTCSeconds())
    );
  }
  return (
    d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate()) + " " +
    pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds())
  );
}

// Safe JSON stringify (handles circular refs)
function safeStringify(obj: any): string {
  const seen: any[] = [];
  return JSON.stringify(
    obj,
    function (_k, v) {
      if (typeof v === "object" && v !== null) {
        if (seen.indexOf(v) >= 0) return "[Circular]";
        seen.push(v);
      }
      if (typeof v === "bigint") return String(v);
      if (v instanceof Error) {
        return { name: v.name, message: v.message, stack: v.stack };
      }
      return v;
    }
  );
}

// key=value rendering for pretty mode
function kv(data?: Record<string, unknown>): string {
  if (!data) return "";
  const parts: string[] = [];
  const keys = Object.keys(data);
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const v: any = (data as any)[k];
    let s: string;
    if (v == null) s = "null";
    else if (typeof v === "string") {
      const needsQuote = /\s|["=]/.test(v);
      s = needsQuote ? JSON.stringify(v) : v;
    } else if (typeof v === "number" || typeof v === "boolean") {
      s = String(v);
    } else if (v instanceof Error) {
      s = JSON.stringify({ name: v.name, message: v.message });
    } else {
      s = safeStringify(v);
    }
    parts.push(k + "=" + s);
  }
  return parts.join(" ");
}

const ANSI: Record<string, string> = {
  reset: "\x1b[0m",
  gray: "\x1b[90m",
  cyan: "\x1b[36m",
  blue: "\x1b[34m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  magenta: "\x1b[35m",
  bold: "\x1b[1m",
};

function colorize(enabled: boolean, text: string, c: string): string {
  return enabled ? (ANSI[c] || "") + text + ANSI.reset : text;
}

function defaultEnabledNamespaces(): string[] {
  const ns = env("LOG_NS");
  if (!ns) return [];
  // comma or space separated
  return ns.split(/[, ]+/).map(s => s.trim()).filter(Boolean);
}

/* ============================== Factory ================================ */

export function createLogger(opts?: LoggerOptions): Logger {
  const tty = isTTY();
  const envLevel = parseLevel(env("LOG_LEVEL"));
  const envJson = parseBool(env("LOG_JSON"));
  const envPretty = parseBool(env("LOG_PRETTY"));
  const envColors = parseBool(env("LOG_COLORS"));

  const state = {
    name: opts && opts.name ? String(opts.name) : undefined,
    level: (opts && opts.level) || envLevel || "info",
    json: typeof (opts && opts.json) === "boolean" ? !!opts!.json : (envJson ?? false),
    pretty: typeof (opts && opts.pretty) === "boolean" ? !!opts!.pretty : (envPretty ?? true),
    colors: typeof (opts && opts.colors) === "boolean" ? !!opts!.colors : (envColors ?? tty),
    enabledNS: (opts && opts.enabledNamespaces) || defaultEnabledNamespaces(),
    useUTC: (opts && typeof opts.useUTC === "boolean") ? !!opts!.useUTC : true,
    ctx: (opts && opts.context) ? shallowClone(opts.context) : {},
    buf: new RingBuffer<LogRecord>((opts && opts.bufferSize) || 200),
  };

  function nsAllowed(name?: string): boolean {
    if (!state.enabledNS || state.enabledNS.length === 0) return true;
    if (!name) return false;
    for (let i = 0; i < state.enabledNS.length; i++) {
      const want = state.enabledNS[i];
      if (want === "*") return true;
      // simple prefix match; support trailing '*' wildcard
      if (want.charAt(want.length - 1) === "*") {
        const p = want.slice(0, want.length - 1);
        if (name.indexOf(p) === 0) return true;
      } else if (name === want || name.indexOf(want) === 0) {
        return true;
      }
    }
    return false;
  }

  function isEnabled(lvl: LogLevelName): boolean {
    return LEVEL_NUM[lvl] >= LEVEL_NUM[state.level] &&
           nsAllowed(state.name) &&
           state.level !== "silent";
  }

  function write(rec: LogRecord) {
    state.buf.push(rec);
    if (state.json) {
      const obj: any = { time: rec.time, level: rec.level, msg: rec.msg };
      if (state.name) obj.name = state.name;
      if (state.ctx && Object.keys(state.ctx).length) obj.ctx = state.ctx;
      if (rec.data && Object.keys(rec.data).length) obj.data = rec.data;
      // eslint-disable-next-line no-console
      console.log(safeStringify(obj));
      return;
    }

    // Pretty line
    const t = localTimestamp(rec.ts, state.useUTC);
    const lvlPadded = padLevel(rec.level);
    const name = state.name ? colorize(state.colors, state.name, "cyan") + " " : "";
    const line =
      colorize(state.colors, t, "gray") + " " +
      colorize(state.colors, lvlColor(rec.level, state.colors), lvlColor(rec.level, false)) + " " +
      name +
      (rec.msg || "") +
      (rec.data ? " " + kv(rec.data) : "") +
      (state.ctx && Object.keys(state.ctx).length ? " " + kv(state.ctx) : "");

    // eslint-disable-next-line no-console
    (rec.level === "error" || rec.level === "fatal") ? console.error(line) :
    (rec.level === "warn") ? console.warn(line) : console.log(line);
  }

  function log(lvl: LogLevelName, msg: string, data?: Record<string, unknown>) {
    if (!isEnabled(lvl)) return;

    const ts = nowMs();
    const rec: LogRecord = {
      time: iso(ts),
      ts,
      level: lvl,
      name: state.name,
      msg,
      data: merge(state.ctx, data),
    };
    write(rec);
  }

  const api: Logger = {
    level() { return state.level; },
    setLevel(lvl: LogLevelName) { if (LEVEL_NUM[lvl] != null) state.level = lvl; },
    isEnabled,

    log,
    trace(m, d) { log("trace", m, d); },
    debug(m, d) { log("debug", m, d); },
    info(m, d)  { log("info",  m, d); },
    warn(m, d)  { log("warn",  m, d); },
    error(m, d) { log("error", m, d); },
    fatal(m, d) { log("fatal", m, d); },

    child(nameOrCtx?: string | Record<string, unknown>) {
      const childOpts: LoggerOptions = {
        name: state.name ? (typeof nameOrCtx === "string" ? state.name + ":" + nameOrCtx : state.name) : (typeof nameOrCtx === "string" ? nameOrCtx : undefined),
        level: state.level,
        json: state.json,
        pretty: state.pretty,
        colors: state.colors,
        enabledNamespaces: state.enabledNS.slice(),
        useUTC: state.useUTC,
        bufferSize: state.buf.size,
        context: merge(state.ctx, typeof nameOrCtx === "object" ? nameOrCtx : undefined),
      };
      return createLogger(childOpts);
    },

    time(label: string, base?: Record<string, unknown>, level?: LogLevelName): Timer {
      const start = nowMs();
      const lvl = level || "debug";
      return function done(extra?: Record<string, unknown>) {
        const dur = nowMs() - start;
        const data = merge(base, extra, { ms: dur, label: label });
        log(lvl, label, data);
      };
    },

    history() { return state.buf.toArray(); },
    name() { return state.name; },
  };

  return api;
}

/* =============================== Utilities ============================== */

class RingBuffer<T> {
  private arr: T[];
  private idx = 0;
  public readonly size: number;

  constructor(n: number) {
    this.size = Math.max(1, Math.floor(n));
    this.arr = new Array(this.size);
  }
  push(x: T) {
    this.arr[this.idx] = x;
    this.idx = (this.idx + 1) % this.size;
  }
  toArray(): T[] {
    const out: T[] = [];
    const start = this.idx;
    for (let i = 0; i < this.size; i++) {
      const v = this.arr[(start + i) % this.size];
      if (v != null) out.push(v);
    }
    return out;
  }
}

function shallowClone<T extends Record<string, unknown>>(o: T | undefined): T {
  const r: any = {};
  if (!o) return r as T;
  const keys = Object.keys(o);
  for (let i = 0; i < keys.length; i++) r[keys[i]] = (o as any)[keys[i]];
  return r as T;
}

function merge(...objs: Array<Record<string, unknown> | undefined>): Record<string, unknown> | undefined {
  let out: any = undefined;
  for (let i = 0; i < objs.length; i++) {
    const o = objs[i];
    if (!o) continue;
    if (!out) out = {};
    const keys = Object.keys(o);
    for (let j = 0; j < keys.length; j++) out[keys[j]] = (o as any)[keys[j]];
  }
  return out;
}

function padLevel(lvl: LogLevelName): string {
  // align to 5 chars for pretty output
  const m = { trace: "TRACE", debug: "DEBUG", info: "INFO ", warn: "WARN ", error: "ERROR", fatal: "FATAL", silent: "     " };
  return m[lvl] || lvl.toUpperCase();
}

function lvlColor(lvl: LogLevelName, colored: boolean): string {
  const label = padLevel(lvl);
  if (!colored) return label;
  switch (lvl) {
    case "trace": return ANSI.gray + label + ANSI.reset;
    case "debug": return ANSI.blue + label + ANSI.reset;
    case "info":  return ANSI.green + label + ANSI.reset;
    case "warn":  return ANSI.yellow + label + ANSI.reset;
    case "error": return ANSI.red + label + ANSI.reset;
    case "fatal": return ANSI.magenta + ANSI.bold + label + ANSI.reset;
    default: return label;
  }
}

/* ============================ Default Export ============================ */

// A convenient default logger (name inferred from env LOG_NAME or "app")
const defaultName = env("LOG_NAME") || "app";
const defaultLogger = createLogger({ name: defaultName });

export { defaultLogger as logger };
export default defaultLogger;