// jobs/alertsdispatch.ts
// Pure TypeScript. No imports. Minimal logger + ULID + dispatcher.

/* =============================== Minimal Logger =============================== */

type LogLevel = "debug" | "info" | "warn" | "error";

function iso(ts = Date.now()) { return new Date(ts).toISOString(); }

function createLogger(name = "alerts") {
  function line(lvl: LogLevel, msg: string, data?: Record<string, unknown>) {
    const payload = data && Object.keys(data).length ? " " + JSON.stringify(data) : "";
    const txt = `[${iso()}] ${lvl.toUpperCase()} ${name} ${msg}${payload}`;
    if (lvl === "error") console.error(txt);
    else if (lvl === "warn") console.warn(txt);
    else console.log(txt);
  }
  function time(label: string, base?: Record<string, unknown>, level: LogLevel = "info") {
    const start = Date.now();
    return (extra?: Record<string, unknown>) => {
      const ms = Date.now() - start;
      line(level, label, { ...(base || {}), ...(extra || {}), ms });
    };
  }
  return {
    debug: (m: string, d?: Record<string, unknown>) => line("debug", m, d),
    info:  (m: string, d?: Record<string, unknown>) => line("info", m, d),
    warn:  (m: string, d?: Record<string, unknown>) => line("warn", m, d),
    error: (m: string, d?: Record<string, unknown>) => line("error", m, d),
    time,
  };
}
const logger = createLogger("alerts");

/* =============================== Minimal ULID =============================== */

type Bytes = Uint8Array;
const B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

function rngBytes(n: number): Bytes {
  const a = new Uint8Array(n);
  const g: any = (globalThis as any);
  const c = g?.crypto || g?.msCrypto;
  if (c && typeof c.getRandomValues === "function") {
    c.getRandomValues(a); return a;
  }
  for (let i = 0; i < n; i++) a[i] = Math.floor(Math.random() * 256);
  return a;
}
function ulidEncodeTime(ms: number): string {
  let t = Math.max(0, Math.floor(ms));
  const out = new Array(10);
  for (let i = 9; i >= 0; i--) { out[i] = B32[t % 32]; t = Math.floor(t / 32); }
  return out.join("");
}
function ulidEncodeRand(bytes: Bytes): string {
  // 10 bytes -> 16 chars (5-bit groups)
  let s = "", acc = 0, bits = 0;
  for (let i = 0; i < bytes.length; i++) {
    acc = (acc << 8) | bytes[i]; bits += 8;
    while (bits >= 5) { bits -= 5; s += B32[(acc >>> bits) & 31]; }
  }
  if (bits > 0) s += B32[(acc << (5 - bits)) & 31];
  return s.slice(0, 16);
}
let __ulid_t = 0;
let __ulid_r = rngBytes(10);
function ulidMonotonic(dateMs?: number): string {
  const t = dateMs ?? Date.now();
  let r = rngBytes(10);
  if (t === __ulid_t) {
    r = __ulid_r.slice();
    for (let i = r.length - 1; i >= 0; i--) { r[i] = (r[i] + 1) & 0xff; if (r[i] !== 0) break; }
  }
  __ulid_t = t; __ulid_r = r;
  return ulidEncodeTime(t) + ulidEncodeRand(r);
}

/* ================================= Types ================================== */

export type Severity = "info" | "low" | "medium" | "high" | "critical";

export type AlertCandidate = {
  ruleId: string;
  scopeKey?: string;
  title: string;
  data?: Record<string, unknown>;
  severity: Severity;
  value?: number;
  threshold?: number;
  cmp?: "gte" | "lte" | "gt" | "lt" | "eq";
  detectedAt?: number;
  channels?: Partial<NotifyTargets>;
};

export type NotifyTargets = {
  email: boolean;
  webhook: boolean;
  sms: boolean;
  push: boolean;
};

export type NotificationPayload = {
  id: string;
  ts: number;
  ruleId: string;
  scopeKey?: string;
  title: string;
  severity: Severity;
  data?: Record<string, unknown>;
  value?: number;
  threshold?: number;
  cmp?: string;
  targets: NotifyTargets;
};

export type Options = {
  fetchCandidates: (signal: AbortSignal) => Promise<AlertCandidate[]>;
  send: (payload: NotificationPayload, signal: AbortSignal) => Promise<void>;
  persist?: (payload: NotificationPayload) => Promise<void>;
  dedupeKey?: (c: AlertCandidate) => string;
  defaults?: { targets?: Partial<Record<Severity, Partial<NotifyTargets>>> };
  maxPerRun?: number;
  cooldownMs?: number;
};

/* ============================ In-memory cooldown ============================ */

const lastSentAt = new Map<string, number>();
function sweepCooldown(now: number, cooldown: number) {
  const cutoff = now - Math.max(1, cooldown);
 
}

/* ================================= Helpers ================================= */

const DEFAULT_TARGETS: NotifyTargets = { email: true, webhook: true, sms: false, push: false };

function mergeTargets(base?: Partial<NotifyTargets>, override?: Partial<NotifyTargets>): NotifyTargets {
  const b = { ...DEFAULT_TARGETS, ...(base || {}) };
  return { ...b, ...(override || {}) } as NotifyTargets;
}
function buildTargets(c: AlertCandidate, opts?: Options): NotifyTargets {
  const sev = c.severity || "info";
  const bySev = opts?.defaults?.targets?.[sev] || {};
  return mergeTargets(bySev, c.channels);
}
function defaultDedupeKey(c: AlertCandidate): string {
  return `${c.ruleId}|${c.scopeKey || "global"}`;
}
function makePayload(c: AlertCandidate, targets: NotifyTargets): NotificationPayload {
  return {
    id: ulidMonotonic(),
    ts: Date.now(),
    ruleId: c.ruleId,
    scopeKey: c.scopeKey,
    title: c.title,
    severity: c.severity,
    data: c.data,
    value: c.value,
    threshold: c.threshold,
    cmp: c.cmp,
    targets,
  };
}

/* ================================ Defaults ================================= */

const NoopOptions: Options = {
  async fetchCandidates() { return []; },
  async send(p) { console.log("[alerts] send noop", JSON.stringify(p)); },
  async persist(_p) { /* noop */ },
  dedupeKey: defaultDedupeKey,
  defaults: { targets: {} },
  maxPerRun: 100,
  cooldownMs: 10 * 60_000,
};

/* ================================== Main ================================== */

export async function runAlertsDispatch(
  signal: AbortSignal,
  options?: Partial<Options>
): Promise<{ sent: number; skippedCooldown: number; total: number }> {
  const opts: Options = { ...NoopOptions, ...(options as any) };
  const started = Date.now();
  const cooldown = Math.max(1, opts.cooldownMs ?? NoopOptions.cooldownMs!);
  const maxPerRun = Math.max(1, opts.maxPerRun ?? NoopOptions.maxPerRun!);
  const keyFn = opts.dedupeKey || defaultDedupeKey;

  const done = logger.time("alerts-dispatch", { cooldownMs: cooldown, maxPerRun });

  const candidates = await opts.fetchCandidates(signal);
  if (signal.aborted) return { sent: 0, skippedCooldown: 0, total: candidates.length };

  sweepCooldown(started, cooldown);

  let skippedCooldown = 0;
  const queue: NotificationPayload[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const key = keyFn(c);
    const last = lastSentAt.get(key) || 0;
    if (started - last < cooldown) { skippedCooldown++; continue; }

    const targets = buildTargets(c, opts);
    const payload = makePayload(c, targets);
    queue.push(payload);

    if (queue.length >= maxPerRun) break;
  }

  let sent = 0;
  for (let i = 0; i < queue.length; i++) {
    if (signal.aborted) break;
    const p = queue[i];
    try {
      await opts.send(p, signal);
      lastSentAt.set(`${p.ruleId}|${p.scopeKey || "global"}`, p.ts);
      if (opts.persist) {
        try { await opts.persist(p); } catch (e: any) { logger.warn("persist failed", { err: e?.message || String(e) }); }
      }
      sent++;
    } catch (e: any) {
      logger.error("send failed", { err: e?.message || String(e), ruleId: p.ruleId, scopeKey: p.scopeKey });
    }
  }

  done({ total: candidates.length, queued: queue.length, sent, skippedCooldown });
  return { sent, skippedCooldown, total: candidates.length };
}