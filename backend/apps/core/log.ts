// core/log.ts
// Zero-import structured logger with:
//  • levels: debug/info/warn/error (+ threshold)
//  • JSON records (console sink by default)
//  • child contexts (trace/span)
//  • timers (duration_ms)
//  • rate limiting per (event, level)
//  • sampling
//  • redaction (mask keys)
//  • ring buffer snapshot (in-mem)
// All pure TypeScript. Drop in anywhere.

export type LogLevel = "debug" | "info" | "warn" | "error";

export type LogRecord = {
  ts: string;              // ISO time
  level: LogLevel;
  event: string;           // short, machine-friendly event name
  msg?: string;            // optional human message
  data?: Record<string, any>;
  trace?: string;          // trace id
  span?: string;           // span id
  ctx?: Record<string, any>; // bound context
  duration_ms?: number;    // for timers
  seq?: number;            // monotonically increasing sequence
};

export type LogSink = (rec: LogRecord) => void | Promise<void>;

export type LogConfig = {
  level?: LogLevel;                  // minimum level to emit (default "info")
  redactKeys?: string[];             // keys to mask in data/ctx
  rateLimitMs?: number;              // min interval per (level,event) (default 0 = off)
  sample?: Partial<Record<LogLevel, number>>; // 0..1 sampling per level (default 1)
  bufferSize?: number;               // ring buffer size for snapshot (default 500)
  sink?: LogSink;                    // custom sink
  pretty?: boolean;                  // pretty print (dev)
};

type RLKey = `${LogLevel}:${string}`;

const LEVEL_RANK: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

export class Logger {
  private cfg: Required<LogConfig>;
  private ctx: Record<string, any>;
  private seq = 0;
  private rateLast = new Map<RLKey, number>();
  private buf: LogRecord[];

  constructor(config?: LogConfig, boundCtx?: Record<string, any>, sharedState?: { rateLast?: Map<RLKey, number>, seq?: number, buf?: LogRecord[] }) {
    this.cfg = normalizeConfig(config);
    this.ctx = sanitize(boundCtx || {});
    // share state across children:
    this.rateLast = sharedState?.rateLast || this.rateLast;
    if (sharedState?.seq != null) this.seq = sharedState.seq;
    if (sharedState?.buf) this.buf = sharedState.buf; else this.buf = new Array<LogRecord>();
  }

  /** Bind more context and/or trace/span, return child logger. */
  with(extra: Record<string, any>): Logger {
    const nextCtx = { ...this.ctx, ...sanitize(extra) };
    const child = new Logger(this.cfg, nextCtx, { rateLast: this.rateLast, seq: this.seq, buf: this.buf });
    return child;
  }

  /** Start/continue a trace. If trace not provided, generates one. */
  trace(trace?: string): Logger {
    return this.with({ trace: trace || genId("tr_") });
  }

  /** Create a child span under the current trace. */
  span(name?: string): Logger {
    const spanId = genId("sp_");
    const label = name ? { span_name: String(name) } : {};
    return this.with({ span: spanId, ...label });
  }

  setLevel(level: LogLevel) { this.cfg.level = level; }
  setSink(sink: LogSink) { this.cfg.sink = sink; }
  setPretty(on: boolean) { this.cfg.pretty = !!on; }
  setSample(level: LogLevel, p: number) { (this.cfg.sample as any)[level] = clamp01(p); }
  setRateLimit(ms: number) { this.cfg.rateLimitMs = Math.max(0, ms|0); }
  setRedactions(keys: string[]) { this.cfg.redactKeys = (keys||[]).slice(); }

  debug(event: string, data?: any, msg?: string) { this.emit("debug", event, data, msg); }
  info(event: string,  data?: any, msg?: string) { this.emit("info",  event, data, msg); }
  warn(event: string,  data?: any, msg?: string) { this.emit("warn",  event, data, msg); }
  error(event: string, data?: any, msg?: string) { this.emit("error", event, data, msg); }

  /** Timer helper: returns a done(ok?, extra?) function that logs duration. */
  timer(event: string, base?: any) {
    const t0 = nowMs();
    const self = this;
    return function done(ok: boolean = true, extra?: any, level: LogLevel = ok ? "info" : "error") {
      const dur = Math.max(0, nowMs() - t0);
      const data = { ...(base||{}), ...(extra||{}), ok };
      (self as Logger).emit(level, event, data, undefined, dur);
      return dur;
    };
  }

  /** Snapshot last N records kept in the ring buffer. */
  snapshot(): LogRecord[] {
    return this.buf.slice();
  }

  /* ─────────── internals ─────────── */

  private emit(level: LogLevel, event: string, data?: any, msg?: string, duration_ms?: number) {
    if (LEVEL_RANK[level] < LEVEL_RANK[this.cfg.level]) return;
    
    if (!event || typeof event !== "string") event = "event";

    // assemble record

    const now = new Date();
    const rec: LogRecord = {
      ts: now.toISOString(),
      level,
      event: String(event || "event"),
      msg: msg != null ? String(msg) : undefined,
      data: redact(copySafe(flatten(data)), this.cfg.redactKeys),
      ctx: redact(copySafe(this.ctx), this.cfg.redactKeys),
      trace: (this.ctx && this.ctx.trace) || undefined,
      span:  (this.ctx && this.ctx.span)  || undefined,
      duration_ms,
      seq: ++this.seq
    };

    if (this.cfg.rateLimitMs > 0 && !this.rateOk(level, event, now.getTime())) {
      return; // dropped due to rate limit
    }

    // ring buffer (bounded)
    const cap = this.cfg.bufferSize;
    if (cap > 0) {
      if (this.buf.length >= cap) this.buf.shift();
      this.buf.push(rec);
    }

    // sink
    try {
      if (this.cfg.pretty) prettySink(rec);
      else jsonSink(rec);
      if (this.cfg.sink) this.cfg.sink(rec);
    } catch { /* never throw from logger */ }
  }

  private rateOk(level: LogLevel, event: string, ts: number): boolean {
    const k: RLKey = `${level}:${event}`;
    const last = this.rateLast.get(k) || 0;
    if (ts - last < this.cfg.rateLimitMs) return false;
    this.rateLast.set(k, ts);
    return true;
  }
}

/* ─────────── factory ─────────── */

export function createLogger(config?: LogConfig): Logger {
  return new Logger(config);
}

/* ─────────── default global logger ─────────── */

export const log = createLogger();

/* ─────────── sinks ─────────── */

function jsonSink(rec: LogRecord) {
  // eslint-disable-next-line no-console
  console.log(stringifySafe(rec));
}

function prettySink(rec: LogRecord) {
  const lvl = rec.level.toUpperCase().padEnd(5);
  const base = `[${rec.ts}] ${lvl} ${rec.event}`;
  const tail = rec.msg ? ` — ${rec.msg}` : "";
  // eslint-disable-next-line no-console
  console.log(`${base}${tail}`);
  if (rec.trace || rec.span) {
    // eslint-disable-next-line no-console
    console.log(`  trace=${rec.trace || "-"} span=${rec.span || "-"}`);
  }
  if (rec.duration_ms != null) {
    // eslint-disable-next-line no-console
    console.log(`  duration=${rec.duration_ms}ms`);
  }
  if (rec.data && Object.keys(rec.data).length) {
    // eslint-disable-next-line no-console
    console.log("  data:", rec.data);
  }
  if (rec.ctx && Object.keys(rec.ctx).length) {
    // eslint-disable-next-line no-console
    console.log("  ctx :", rec.ctx);
  }
}

/* ─────────── helpers ─────────── */

function normalizeConfig(cfg?: LogConfig): Required<LogConfig> {
  return {
    level: cfg?.level ?? "info",
    redactKeys: (cfg?.redactKeys || []).slice(),
    rateLimitMs: cfg?.rateLimitMs ?? 0,
    sample: {
      debug: clamp01(cfg?.sample?.debug ?? 1),
      info:  clamp01(cfg?.sample?.info  ?? 1),
      warn:  clamp01(cfg?.sample?.warn  ?? 1),
      error: clamp01(cfg?.sample?.error ?? 1),
    },
    bufferSize: cfg?.bufferSize ?? 500,
    sink: cfg?.sink ?? (() => {}),
    pretty: !!cfg?.pretty,
  };
}

function clamp01(x: any): number {
  const n = Number(x);
  if (!isFinite(n)) return 1;
  return Math.max(0, Math.min(1, n));
}

function samplePass(p: number): boolean {
  if (p >= 1) return true;
  if (p <= 0) return false;
  return Math.random() < p;
}

function sanitize(obj: any): Record<string, any> {
  if (!obj || typeof obj !== "object") return {};
  const out: Record<string, any> = Object.create(null);
  for (const k in obj) {
    const v = (obj as any)[k];
    if (v === undefined) continue;
    out[k] = v;
  }
  return out;
}

function flatten(x: any): any {
  if (!x || typeof x !== "object") return x;
  if (x instanceof Error) return serializeError(x);
  // shallow flatten to plain object (avoid prototypes)
  const out: any = Object.create(null);
  for (const k in x) {
    const v = (x as any)[k];
    out[k] = (v instanceof Error) ? serializeError(v) : v;
  }
  return out;
}

function copySafe(x: any): any {
  try { return JSON.parse(JSON.stringify(x)); } catch { return x; }
}

function serializeError(e: any) {
  const obj: any = {
    name: String(e?.name || "Error"),
    message: String(e?.message || ""),
  };
  const st = String(e?.stack || "");
  obj.stack = st.length > 2000 ? st.slice(0, 2000) + "…" : st;
  for (const k in e) if (!(k in obj)) obj[k] = e[k];
  return obj;
}

function redact<T extends Record<string, any>>(obj: T, keys: string[] = []): T {
  if (!obj || typeof obj !== "object" || !keys.length) return obj;
  const set = new Set(keys.map(k => k.toLowerCase()));
  const out: any = Array.isArray(obj) ? [] : Object.create(null);
  for (const k in obj) {
    const v = (obj as any)[k];
    if (set.has(k.toLowerCase())) out[k] = mask(v);
    else if (v && typeof v === "object") out[k] = redact(v as any, keys);
    else out[k] = v;
  }
  return out as T;
}

function mask(v: any): any {
  if (v == null) return v;
  const s = String(v);
  if (s.length <= 4) return "***";
  return s.slice(0, 2) + "***" + s.slice(-2);
}

function stringifySafe(o: any): string {
  try { return JSON.stringify(o); } catch {
    return JSON.stringify({ ts: new Date().toISOString(), level: "error", event: "logger.stringify.fail" });
  }
}

function genId(prefix = "id_"): string {
  let x = 2166136261 >>> 0;
  const s = prefix + Date.now().toString(36) + ":" + Math.random().toString(36).slice(2);
  for (let i = 0; i < s.length; i++) { x ^= s.charCodeAt(i); x = Math.imul(x, 16777619) >>> 0; }
  return prefix + x.toString(16);
}

function nowMs(): number {
  // Prefer performance.now if available for sub-ms monotonic timing
  // @ts-ignore
  const perf = (typeof performance !== "undefined" && performance && performance.now) ? performance : null;
  if (perf && typeof perf.now === "function") {
    // combine with Date.now() epoch for absolute-ish ms
    const base = Date.now() - perf.now();
    return Math.floor(base + perf.now());
  }
  return Date.now();
}