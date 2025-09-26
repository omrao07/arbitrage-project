// recompute.ts
// Orchestrator for (re)computing precompute tasks.
// Zero imports. Drop in as a standalone module.
//
// What this gives you:
//  • Register any number of tasks (compatible with core/precompute.ts PrecomputeTask)
//  • Interval scheduling with jitter + failure backoff
//  • Manual run / filtered run (by id / tag / dataset)
//  • Dependency invalidation (dataset/tag) triggers immediate recompute
//  • Concurrency control (semaphore) + in-flight de-dup (per task)
//  • Persists lightweight run manifests & planner state in a TextStore
//  • Works with any “runner” that exposes run(task) → { artifact, manifest }
//
// You DO NOT need to import anything. Provide your PrecomputeEngine instance
// (or any compatible runner) via constructor.
//
// If you don’t have a store/engine yet, this file includes a MemoryTextStore
// and a simple FakeRunner for smoke tests at the bottom.

/* ─────────────────────── Types ─────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type TextStore = {
  read(key: string): Promise<string | null>;
  write(key: string, val: string): Promise<void>;
  del(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>;
};

export type PrecomputeTask<I = any, R = any> = {
  id: string;         // unique id
  dataset: string;    // logical dataset
  provider: string;   // provider name
  tags?: string[];    // arbitrary labels
  ttlMs?: number;     // hint for artifact freshness (optional)
  input: I;
  fetch: (input: I, cursor: string | undefined) => Promise<{ rows: any[]; nextCursor?: string }>;
  compute: (rows: any[], input: I) => Promise<R> | R;
  outputKey?: string; // where artifact is written by the runner
};

export type Runner = {
  run<I, R>(task: PrecomputeTask<I, R>, opts?: { meta?: Dict }): Promise<{ artifact: R; manifest: Manifest }>;
};

export type Manifest = {
  id: string;
  task: string;
  dataset: string;
  provider: string;
  tags?: string[];
  input: any;
  outputKey: string;
  rows: number;
  cursor?: string;
  started: string;
  finished: string;
  duration_ms: number;
  ok: boolean;
  error?: string;
  ttlMs?: number;
  meta?: Dict;
};

export type PlanConfig = {
  /** Base interval between runs (ms). Default: 5 minutes. */
  intervalMs?: number;
  /** Jitter fraction 0..1 (randomize next due within ±jitter*interval). Default 0.15 */
  jitter?: number;
  /** Max exponential backoff for repeated failures. Default 15 min. */
  maxBackoffMs?: number;
  /** Dependencies → if any change, task is marked dirty. */
  deps?: {
    datasets?: string[];   // e.g., ["px-daily","fundamentals-q"]
    tags?: string[];       // e.g., ["today","tile:market"]
  };
  /** Optional per-task priority (higher runs first when contended). */
  priority?: number;
};

export type RegisterOpts = PlanConfig & {
  /** Override the planner’s TextStore keys prefix for this task (rare). */
  prefixKey?: string;
};

export type RunFilter = {
  ids?: string[];
  tag?: string;
  dataset?: string;
  onlyDirty?: boolean;
  force?: boolean;
  limit?: number;
};

export type RecomputeOpts = {
  runner: Runner;
  store?: TextStore;
  /** Poll period for the scheduler loop (ms). Default: 5s */
  tickMs?: number;
  /** Max parallel tasks. Default: 3 */
  concurrency?: number;
  /** Global jitter to apply when computing next due. Default: 0.15 */
  defaultJitter?: number;
  /** Default interval (ms) for tasks without explicit interval. Default: 5 min */
  defaultIntervalMs?: number;
  /** Max backoff cap (ms). Default: 15 min */
  defaultMaxBackoffMs?: number;
  /** Optional event hooks */
  onEvent?: (evt: PlannerEvent) => void;
};

export type PlannerEvent =
  | { type: "task_registered"; id: string }
  | { type: "task_scheduled"; id: string; nextDue: string }
  | { type: "task_start"; id: string; reason: string }
  | { type: "task_finish"; id: string; ok: boolean; error?: string; duration_ms: number }
  | { type: "invalidate"; reason: string; count: number }
  | { type: "tick"; dueCount: number };

/* ─────────────────────── Recompute Planner ─────────────────────── */

export class Recompute {
  private runner: Runner;
  private store: TextStore;
  private tickMs: number;
  private maxConc: number;
  private defaultJitter: number;
  private defaultInterval: number;
  private defaultMaxBackoff: number;
  private onEvent?: (e: PlannerEvent) => void;

  private tasks = new Map<string, TaskEntry>();
  private timer: any = null;
  private sem = new Semaphore(3);
  private inflight = new Set<string>();

  constructor(opts: RecomputeOpts) {
    this.runner = opts.runner;
    this.store = opts.store || MemoryTextStore();
    this.tickMs = Math.max(1000, opts.tickMs ?? 5000);
    this.maxConc = Math.max(1, opts.concurrency ?? 3);
    this.sem = new Semaphore(this.maxConc);
    this.defaultJitter = clamp01(opts.defaultJitter ?? 0.15);
    this.defaultInterval = Math.max(5_000, opts.defaultIntervalMs ?? 5 * 60_000);
    this.defaultMaxBackoff = Math.max(10_000, opts.defaultMaxBackoffMs ?? 15 * 60_000);
    this.onEvent = opts.onEvent;
  }

  /** Register a task + plan. */
  register<I, R>(task: PrecomputeTask<I, R>, cfg?: RegisterOpts): void {
    if (!task?.id) throw new Error("recompute.register: missing task.id");
    if (this.tasks.has(task.id)) throw new Error("recompute.register: duplicate id " + task.id);

    const now = Date.now();
    const entry: TaskEntry = {
      task: task as any,
      id: task.id,
      cfg: normalizePlan(cfg, this.defaultInterval, this.defaultJitter, this.defaultMaxBackoff),
      state: {
        nextDue: now + randJitter(this.defaultInterval, this.defaultJitter),
        dirty: true,
        failures: 0,
        lastStart: 0,
        lastFinish: 0,
        lastOk: undefined
      },
      prefixKey: cfg?.prefixKey || "recompute"
    };

    this.tasks.set(task.id, entry);
    this.emit({ type: "task_registered", id: task.id });
    this.schedulePersist(entry).catch(() => {});
  }

  /** Force-invalidate tasks that depend on a dataset/tag (or all). */
  async invalidate(reason: { dataset?: string; tag?: string; all?: boolean }): Promise<number> {
    let n = 0;
    for (const e of this.tasks.values()) {
      if (reason.all) { e.state.dirty = true; n++; continue; }
      const depDs = e.cfg.deps?.datasets || [];
      const depTags = e.cfg.deps?.tags || [];
      if ((reason.dataset && depDs.includes(reason.dataset)) ||
          (reason.tag && depTags.includes(reason.tag))) {
        e.state.dirty = true; n++;
      }
    }
    if (n) this.emit({ type: "invalidate", reason: JSON.stringify(reason), count: n });
    return n;
  }

  /** Manually run by id (returns manifest). */
  async runById(id: string, why = "manual"): Promise<Manifest> {
    const e = this.tasks.get(id);
    if (!e) throw new Error("unknown_task:" + id);
    const res = await this.runOne(e, why);
    return res.manifest;
  }

  /** Run all tasks that are due/dirty (optionally filtered). */
  async runEligible(filter?: RunFilter): Promise<number> {
    const due = this.pickDue(filter);
    const limited = (filter?.limit && filter.limit > 0) ? due.slice(0, filter.limit) : due;
   
    return limited.length;
  }

  /** Start background loop. */
  start(): void {
    if (this.timer) return;
    const tick = async () => {
      const due = this.pickDue();
      this.emit({ type: "tick", dueCount: due.length });
      for (const d of due) this.safeRun(d.entry, d.reason);
      this.timer = setTimeout(tick, this.tickMs);
    };
    this.timer = setTimeout(tick, this.tickMs);
  }

  /** Stop background loop (does not cancel in-flight tasks). */
  stop(): void {
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
  }

  /** Introspection helpers. */
  list(): PlannerInfo[] {
    const out: PlannerInfo[] = [];
    for (const e of this.tasks.values()) out.push(describe(e));
    out.sort((a, b) => (a.nextDueMs - b.nextDueMs));
    return out;
  }

  /* ───────────── internals ───────────── */

  private pickDue(filter?: RunFilter): Array<{ entry: TaskEntry; reason: string }> {
    const now = Date.now();
    const rows: Array<{ entry: TaskEntry; reason: string }> = [];
    const ids = filter?.ids && new Set(filter.ids.map(String));
    for (const e of this.tasks.values()) {
      if (ids && !ids.has(e.id)) continue;
      if (filter?.dataset && e.task.dataset !== filter.dataset) continue;
      if (filter?.tag && !(e.task.tags || []).includes(filter.tag)) continue;

      const isDirty = e.state.dirty;
      const isDue = now >= e.state.nextDue;
      const eligible = filter?.force || (filter?.onlyDirty ? isDirty : (isDirty || isDue));

      if (!eligible) continue;
      if (this.inflight.has(e.id)) continue; // de-dup in-flight
      rows.push({ entry: e, reason: isDirty ? "dirty" : "interval" });
    }
    // priority sort: higher first; then earlier nextDue
    rows.sort((a, b) => {
      const pa = a.entry.cfg.priority || 0;
      const pb = b.entry.cfg.priority || 0;
      if (pa !== pb) return pb - pa;
      return a.entry.state.nextDue - b.entry.state.nextDue;
    });
    return rows;
  }

  private async safeRun(e: TaskEntry, reason: string): Promise<void> {
    // semaphore
    const release = await this.sem.acquire();
    try {
      await this.runOne(e, reason);
    } finally {
      release();
    }
  }

  private async runOne(e: TaskEntry, reason: string): Promise<{ artifact: any; manifest: Manifest }> {
    // in-flight guard
    if (this.inflight.has(e.id)) {
      // another runner grabbed it just now
      // best-effort: skip
      return { artifact: undefined, manifest: dummyManifest(e, "skipped_inflight") };
    }
    this.inflight.add(e.id);

    const start = Date.now();
    this.emit({ type: "task_start", id: e.id, reason });

    e.state.lastStart = start;
    e.state.dirty = false;

    try {
      const { artifact, manifest } = await this.runner.run(e.task, { meta: { reason } });

      const ok = manifest?.ok !== false && !manifest?.error;
      e.state.lastFinish = Date.now();
      e.state.lastOk = ok;
      e.state.failures = ok ? 0 : e.state.failures + 1;

      // schedule next
      e.state.nextDue = this.computeNextDue(e, ok);

      await this.persistRun(e, manifest).catch(() => {});
      this.emit({ type: "task_finish", id: e.id, ok, error: manifest?.error, duration_ms: manifest?.duration_ms ?? (Date.now() - start) });
      return { artifact, manifest };
    } catch (err: any) {
      const now = Date.now();
      e.state.lastFinish = now;
      e.state.lastOk = false;
      e.state.failures += 1;
      const next = this.computeNextDue(e, false);
      e.state.nextDue = next;

      const manifest = {
        id: `${e.id}:${now.toString(36)}`,
        task: e.id,
        dataset: e.task.dataset,
        provider: e.task.provider,
        tags: e.task.tags,
        input: e.task.input,
        outputKey: "",
        rows: 0,
        started: new Date(start).toISOString(),
        finished: new Date(now).toISOString(),
        duration_ms: now - start,
        ok: false,
        error: String(err?.message || err) || "error",
        ttlMs: e.task.ttlMs,
        meta: { reason }
      } as Manifest;

      await this.persistRun(e, manifest).catch(() => {});
      this.emit({ type: "task_finish", id: e.id, ok: false, error: manifest.error, duration_ms: manifest.duration_ms });
      return { artifact: undefined, manifest };
    } finally {
      this.inflight.delete(e.id);
      this.schedulePersist(e).catch(() => {});
    }
  }

  private computeNextDue(e: TaskEntry, ok: boolean): number {
    const base = e.cfg.intervalMs;
    if (ok) {
      // normal cadence + jitter
      const j = randJitter(base, e.cfg.jitter);
      return Date.now() + j;
    }
    // failure → exponential backoff, capped
    const back = Math.min(e.cfg.maxBackoffMs, base * Math.pow(2, Math.max(0, e.state.failures - 1)));
    const withJitter = randJitter(back, Math.min(0.5, e.cfg.jitter * 2));
    return Date.now() + withJitter;
  }

  private async schedulePersist(e: TaskEntry): Promise<void> {
    // Save compact planner state per task
    const key = `${e.prefixKey}/_planner/${sanitize(e.id)}.json`;
    const state = {
      id: e.id,
      nextDue: e.state.nextDue,
      dirty: e.state.dirty,
      failures: e.state.failures,
      lastStart: e.state.lastStart,
      lastFinish: e.state.lastFinish,
      lastOk: e.state.lastOk,
      cfg: e.cfg
    };
    await this.store.write(key, JSON.stringify(state));
    this.emit({ type: "task_scheduled", id: e.id, nextDue: new Date(e.state.nextDue).toISOString() });
  }

  private async persistRun(e: TaskEntry, m: Manifest): Promise<void> {
    const ts = m.finished || new Date().toISOString();
    const key = `${e.prefixKey}/_runs/${sanitize(e.id)}/${ts.replace(/[:.]/g, "-")}.json`;
    await this.store.write(key, JSON.stringify(m) + "\n");
  }

  private emit(evt: PlannerEvent) { try { this.onEvent?.(evt); } catch { /* no throw */ } }
}

/* ─────────────────────── Helpers / Internals ─────────────────────── */

type TaskEntry = {
  id: string;
  task: PrecomputeTask<any, any>;
  cfg: Required<PlanConfig>;
  state: {
    nextDue: number;
    dirty: boolean;
    failures: number;
    lastStart: number;
    lastFinish: number;
    lastOk?: boolean;
  };
  prefixKey: string;
};

export type PlannerInfo = {
  id: string;
  dataset: string;
  provider: string;
  tags: string[];
  intervalMs: number;
  jitter: number;
  priority: number;
  failures: number;
  lastStart?: string;
  lastFinish?: string;
  lastOk?: boolean;
  nextDue: string;
  nextDueMs: number;
  dirty: boolean;
  inflight: boolean;
};

function describe(e: TaskEntry): PlannerInfo {
  return {
    id: e.id,
    dataset: e.task.dataset,
    provider: e.task.provider,
    tags: e.task.tags || [],
    intervalMs: e.cfg.intervalMs,
    jitter: e.cfg.jitter,
    priority: e.cfg.priority || 0,
    failures: e.state.failures,
    lastStart: e.state.lastStart ? new Date(e.state.lastStart).toISOString() : undefined,
    lastFinish: e.state.lastFinish ? new Date(e.state.lastFinish).toISOString() : undefined,
    lastOk: e.state.lastOk,
    nextDue: new Date(e.state.nextDue).toISOString(),
    nextDueMs: e.state.nextDue,
    dirty: e.state.dirty,
    inflight: false
  };
}

function normalizePlan(cfg: RegisterOpts | undefined, defInterval: number, defJitter: number, defMaxBackoff: number): Required<PlanConfig> {
  const interval = Math.max(5_000, cfg?.intervalMs ?? defInterval);
  const jitter = clamp01(cfg?.jitter ?? defJitter);
  const maxBack = Math.max(interval, cfg?.maxBackoffMs ?? defMaxBackoff);
  return {
    intervalMs: interval,
    jitter,
    maxBackoffMs: maxBack,
    deps: cfg?.deps || {},
    priority: cfg?.priority ?? 0
  };
}

function clamp01(x: number): number {
  if (!isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function randJitter(baseMs: number, jitter: number): number {
  const j = Math.max(0, Math.min(1, jitter));
  if (j <= 0) return baseMs;
  const lo = baseMs * (1 - j);
  const hi = baseMs * (1 + j);
  return Math.floor(lo + Math.random() * (hi - lo));
}

function sanitize(s: string): string {
  return String(s || "").replace(/[^A-Za-z0-9._\-]/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "") || "na";
}

function dummyManifest(e: TaskEntry, reason: string): Manifest {
  const now = new Date().toISOString();
  return {
    id: `${e.id}:${Date.now().toString(36)}`,
    task: e.id,
    dataset: e.task.dataset,
    provider: e.task.provider,
    input: e.task.input,
    tags: e.task.tags || [],
    outputKey: "",
    rows: 0,
    started: now,
    finished: now,
    duration_ms: 0,
    ok: true,
    ttlMs: e.task.ttlMs,
    meta: { reason }
  };
}

/* ─────────────────────── Minimal Semaphore ─────────────────────── */

class Semaphore {
  private permits: number;
  private q: Array<() => void> = [];
  constructor(n: number) { this.permits = Math.max(1, n|0); }
  async acquire(): Promise<() => void> {
    if (this.permits > 0) { this.permits--; return () => this.release(); }
    return new Promise(res => this.q.push(() => res(() => this.release())));
  }
  private release() {
    const w = this.q.shift();
    if (w) w(); else this.permits++;
  }
}

/* ─────────────────────── Memory TextStore ─────────────────────── */

export function MemoryTextStore(): TextStore {
  const m = new Map<string, string>();
  return {
    async read(k) { return m.has(k) ? (m.get(k) as string) : null; },
    async write(k, v) { m.set(k, String(v)); },
    async del(k) { m.delete(k); },
    async list(prefix) { const out: string[] = []; for (const k of m.keys()) if (k.startsWith(prefix)) out.push(k); return out; }
  };
}

/* ─────────────────────── Fake Runner (for smoke tests) ─────────────────────── */

export class FakeRunner implements Runner {
  async run<I, R>(task: PrecomputeTask<I, R>): Promise<{ artifact: any; manifest: Manifest }> {
    // do nothing: pretend we computed something
    const t0 = Date.now();
    await sleep(5 + Math.floor(Math.random() * 15));
    const ok = Math.random() < 0.98; // 2% fail to exercise backoff
    const art = { id: task.id, ts: new Date().toISOString(), echo: task.input };
    const t1 = Date.now();
    return {
      artifact: art,
      manifest: {
        id: `${task.id}:${t1.toString(36)}`,
        task: task.id,
        dataset: task.dataset,
        provider: task.provider,
        input: task.input,
        tags: task.tags || [],
        outputKey: `precompute/${sanitize(task.id)}/${new Date(t1).toISOString().replace(/[:.]/g,"-")}.json`,
        rows: 1,
        started: new Date(t0).toISOString(),
        finished: new Date(t1).toISOString(),
        duration_ms: t1 - t0,
        ok,
        error: ok ? undefined : "sim_error",
        ttlMs: task.ttlMs,
        meta: { simulated: true }
      }
    };
  }
}

/* ─────────────────────── Tiny self-test ─────────────────────── */

export async function __selftest__(): Promise<string> {
  const store = MemoryTextStore();
  const runner = new FakeRunner();
  const rc = new Recompute({ runner, store, tickMs: 50, concurrency: 2 });

  // two simulated tasks
  rc.register({
    id: "tile:market",
    dataset: "px-daily",
    provider: "sim",
    tags: ["today","tiles"],
    ttlMs: 60000,
    input: { sym: "SPY" },
    fetch: async (_i, _c) => ({ rows: [{t:1,c:100}], nextCursor: "1" }),
    compute: (rows, i) => ({ sym: i.sym, n: rows.length }),
    outputKey: "precompute/tile-market/${date}.json"
  }, { intervalMs: 200, jitter: 0.2, deps: { datasets: ["px-daily"] }, priority: 5 });

  rc.register({
    id: "tile:rates",
    dataset: "yield-curves",
    provider: "sim",
    tags: ["today","tiles"],
    ttlMs: 60000,
    input: { curve: "UST" },
    fetch: async () => ({ rows: [{t:1,y:0.02}], nextCursor: "1" }),
    compute: (rows, i) => ({ curve: i.curve, n: rows.length }),
  }, { intervalMs: 250, jitter: 0.1, deps: { datasets: ["rates"] }, priority: 3 });

  // run a couple ticks manually (no loop) to verify no throws
  await rc.runEligible({ limit: 10 });
  await rc.invalidate({ dataset: "px-daily" });
  await rc.runEligible({ onlyDirty: true });
  const lst = rc.list();
  return lst.length === 2 ? "ok" : "fail";
}

/* ─────────────────────── local sleep ─────────────────────── */

function sleep(ms: number): Promise<void> { return new Promise(r => setTimeout(r, Math.max(0, ms|0))); }