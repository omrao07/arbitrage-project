// jobs/stresstrigger.ts
// Pure TypeScript. No imports. Minimal logger + ULID + concurrent stress trigger.

/* =============================== Minimal Logger =============================== */

type LogLevel = "debug" | "info" | "warn" | "error";

function iso(ts = Date.now()) { return new Date(ts).toISOString(); }

function createLogger(name = "stress") {
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
const logger = createLogger("stress");

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

/* ================================== Types ================================== */

export type Target = {
  portfolioId: string;
  accountId?: string;
  householdId?: string;
  // arbitrary labels you may need downstream
  labels?: Record<string, string | number | boolean>;
};

export type StressScenario = {
  id: string;
  name?: string;
  params?: Record<string, unknown>;
  // Optional ordering or importance
  priority?: number;
};

export type StressJob = {
  id: string;                 // local job id (ulid)
  target: Target;
  scenario: StressScenario;
  createdAt: number;
  attempts: number;
};

export type TriggerResult = {
  ok: boolean;
  externalId?: string;        // id from remote system
  statusUrl?: string;
  error?: string;
};

export type RetryPolicy = {
  retries?: number;           // default 2 (total attempts = retries+1)
  baseMs?: number;            // default 500ms (exponential backoff)
  maxMs?: number;             // default 5000ms
};

export type Options = {
  /** Collect portfolios/targets to stress. */
  enumerateTargets: (signal: AbortSignal) => Promise<Target[]>;

  /** For a given target, list scenarios to run. */
  listScenarios: (target: Target, signal: AbortSignal) => Promise<StressScenario[]>;

  /** Fire a single stress job to your backend/action. */
  trigger: (job: StressJob, signal: AbortSignal) => Promise<TriggerResult>;

  /** Optional hook after each job completes (success or failure). */
  onResult?: (job: StressJob, res: TriggerResult) => Promise<void> | void;

  /** Persist an audit event (queued/sent/failed). */
  persist?: (event: { type: "queued" | "sent" | "failed" | "skipped"; job: StressJob; info?: any }) => Promise<void> | void;

  /** Drop if we recently fired the same (target,scenario) within cooldown. */
  cooldownMs?: number;        // default 15 min

  /** Limit total jobs per run. */
  maxPerRun?: number;         // default 100

  /** Concurrency for firing remote triggers. */
  concurrency?: number;       // default 3

  /** Retry policy for trigger errors. */
  retry?: RetryPolicy;

  /** Optional filter to skip some jobs before queueing. */
  shouldTrigger?: (target: Target, scenario: StressScenario) => boolean;
};

/* ================================ In-memory ================================= */

const lastTriggered = new Map<string, number>(); // key: tgt|scn -> ts

function keyFor(target: Target, scenario: StressScenario): string {
  const t = target.portfolioId || target.accountId || target.householdId || "unknown";
  return `${t}|${scenario.id}`;
}
function sweepCooldown(now: number, cooldownMs: number) {
  const cutoff = now - Math.max(1, cooldownMs);
 
}

/* ================================= Helpers ================================= */

function sleep(ms: number): Promise<void> {
  return new Promise(function (res) { setTimeout(res, Math.max(0, ms | 0)); });
}

async function withRetry<T>(
  fn: () => Promise<T>,
  signal: AbortSignal,
  policy?: RetryPolicy
): Promise<T> {
  const retries = Math.max(0, (policy?.retries ?? 2) | 0);
  const base = Math.max(1, (policy?.baseMs ?? 500) | 0);
  const maxD = Math.max(base, (policy?.maxMs ?? 5000) | 0);

  let attempt = 0;
  let lastErr: any = null;
  while (attempt <= retries) {
    if (signal.aborted) throw new Error("aborted");
    try {
      return await fn();
    } catch (e: any) {
      lastErr = e;
      attempt++;
      if (attempt > retries) break;
      const delay = Math.min(maxD, base * Math.pow(2, attempt - 1));
      await sleep(delay);
    }
  }
  throw lastErr || new Error("failed");
}

async function runPool<T, R>(
  items: T[],
  limit: number,
  worker: (item: T) => Promise<R>,
  signal: AbortSignal
): Promise<R[]> {
  const n = Math.max(1, limit | 0);
  const results: R[] = new Array(items.length) as any;
  let idx = 0;

  async function next(): Promise<void> {
    if (signal.aborted) return;
    const i = idx++;
    if (i >= items.length) return;
    try {
      results[i] = await worker(items[i]);
    } catch (e: any) {
      (results as any)[i] = e;
    }
    if (!signal.aborted) return next();
  }

  const runners = new Array(Math.min(n, items.length)).fill(0).map(() => next());
  await Promise.all(runners);
  return results;
}

/* ================================= Defaults ================================= */

const Defaults: Required<Pick<Options,
  "cooldownMs" | "maxPerRun" | "concurrency" | "retry"
>> = {
  cooldownMs: 15 * 60_000,
  maxPerRun: 100,
  concurrency: 3,
  retry: { retries: 2, baseMs: 500, maxMs: 5000 },
};

const NoopOptions: Options = {
  async enumerateTargets() { return []; },
  async listScenarios(_t) { return []; },
  async trigger(_job) { logger.debug("trigger noop"); return { ok: true }; },
  onResult: undefined,
  persist: undefined,
  cooldownMs: Defaults.cooldownMs,
  maxPerRun: Defaults.maxPerRun,
  concurrency: Defaults.concurrency,
  retry: Defaults.retry,
  shouldTrigger: undefined,
};

/* ================================== Main ================================== */

export async function runStressTrigger(
  signal: AbortSignal,
  options?: Partial<Options>
): Promise<{ queued: number; sent: number; failed: number; skippedCooldown: number }> {
  const opts: Options = { ...NoopOptions, ...(options as any) };
  const started = Date.now();
  const cooldown = Math.max(1, opts.cooldownMs ?? Defaults.cooldownMs);
  const maxPerRun = Math.max(1, opts.maxPerRun ?? Defaults.maxPerRun);
  const concurrency = Math.max(1, opts.concurrency ?? Defaults.concurrency);
  const retry = opts.retry || Defaults.retry;

  sweepCooldown(started, cooldown);
  const done = logger.time("stress-trigger", { cooldownMs: cooldown, maxPerRun, concurrency });

  // 1) Enumerate targets
  const targets = await opts.enumerateTargets(signal);
  if (signal.aborted) return { queued: 0, sent: 0, failed: 0, skippedCooldown: 0 };

  // 2) Expand scenarios and build candidate jobs
  const candidates: StressJob[] = [];
  for (let ti = 0; ti < targets.length; ti++) {
    if (signal.aborted || candidates.length >= maxPerRun) break;
    const t = targets[ti];
    let scenarios: StressScenario[] = [];
    try {
      scenarios = await opts.listScenarios(t, signal);
    } catch (e: any) {
      logger.warn("listScenarios failed", { err: e?.message || String(e), portfolioId: t.portfolioId });
      continue;
    }
    // sort by priority if provided
    scenarios.sort(function (a, b) {
      const pa = (a.priority == null ? 0 : a.priority) as number;
      const pb = (b.priority == null ? 0 : b.priority) as number;
      return pb - pa;
    });

    for (let si = 0; si < scenarios.length; si++) {
      if (candidates.length >= maxPerRun) break;
      const scn = scenarios[si];

      if (opts.shouldTrigger && !opts.shouldTrigger(t, scn)) continue;

      const key = keyFor(t, scn);
      const last = lastTriggered.get(key) || 0;
      if (started - last < cooldown) continue;

      const job: StressJob = {
        id: ulidMonotonic(),
        target: t,
        scenario: scn,
        createdAt: Date.now(),
        attempts: 0,
      };
      candidates.push(job);
      if (opts.persist) {
        try { await opts.persist({ type: "queued", job }); } catch (_) { /* ignore */ }
      }
    }
  }

  const queued = candidates.length;
  if (queued === 0) { done({ queued: 0, sent: 0, failed: 0, skippedCooldown: 0 }); return { queued: 0, sent: 0, failed: 0, skippedCooldown: 0 }; }

  // 3) Split into cooldown-safe and skipped, re-check to report stats
  let skippedCooldown = 0;
  const queue: StressJob[] = [];
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const k = keyFor(c.target, c.scenario);
    const last = lastTriggered.get(k) || 0;
    if (started - last < cooldown) skippedCooldown++;
    else queue.push(c);
  }

  // 4) Fire with concurrency + retries
  let sent = 0, failed = 0;

  await runPool(queue, concurrency, async (job) => {
    if (signal.aborted) return;

    try {
      const res = await withRetry(async () => {
        job.attempts++;
        return opts.trigger(job, signal);
      }, signal, retry);

      if (res && res.ok) {
        sent++;
        lastTriggered.set(keyFor(job.target, job.scenario), Date.now());
        if (opts.persist) { try { await opts.persist({ type: "sent", job, info: res }); } catch (_) { /* ignore */ } }
        if (opts.onResult) { try { await opts.onResult(job, res); } catch (_) { /* ignore */ } }
      } else {
        failed++;
        if (opts.persist) { try { await opts.persist({ type: "failed", job, info: res }); } catch (_) { /* ignore */ } }
        if (opts.onResult) { try { await opts.onResult(job, res || { ok: false, error: "unknown" }); } catch (_) { /* ignore */ } }
      }
    } catch (e: any) {
      failed++;
      const info = { ok: false, error: e?.message || String(e) };
      if (opts.persist) { try { await opts.persist({ type: "failed", job, info }); } catch (_) { /* ignore */ } }
      if (opts.onResult) { try { await opts.onResult(job, info); } catch (_) { /* ignore */ } }
      logger.error("trigger failed", { err: info.error, portfolioId: job.target.portfolioId, scenario: job.scenario.id });
    }
  }, signal);

  done({ queued, sent, failed, skippedCooldown });
  return { queued, sent, failed, skippedCooldown };
}

/* ============================== Example wiring ============================== */
/*
await runStressTrigger(signal, {
  enumerateTargets: async () => [
    { portfolioId: "P1001" },
    { portfolioId: "P1002" }
  ],
  listScenarios: async (t) => [
    { id: "HIST_2020_COVID", name: "Mar-2020 Shock", priority: 10 },
    { id: "RATE_UP_200BP", name: "+200bp Parallel", priority: 5, params: { curve: "parallel", bp: 200 } }
  ],
  trigger: async (job) => {
    // Call your triggerstress.action endpoint here
    // Example pseudo-call using global fetch (Node 18+ / browsers):
    // const res = await fetch("https://internal/api/triggerstress", {
    //   method: "POST",
    //   headers: { "content-type": "application/json" },
    //   body: JSON.stringify({
    //     portfolioId: job.target.portfolioId,
    //     scenarioId: job.scenario.id,
    //     params: job.scenario.params
    //   })
    // });
    // if (!res.ok) return { ok: false, error: "HTTP " + res.status };
    // const json = await res.json();
    // return { ok: true, externalId: json.id, statusUrl: json.statusUrl };
    return { ok: true, externalId: "demo-" + job.id };
  },
  cooldownMs: 10 * 60_000,
  maxPerRun: 50,
  concurrency: 4,
  retry: { retries: 3, baseMs: 400, maxMs: 4000 },
  shouldTrigger: (t, s) => true,
  persist: async (e) => { /* write to DB/log */ 
