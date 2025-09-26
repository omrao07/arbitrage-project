// core/scheduler.ts
// Lightweight, import-free job scheduler with intervals, delays, retries,
// backoff, timeouts, jitter, concurrency control, and event hooks.
//
// - No imports; drop-in anywhere.
// - Schedule repeating or one-shot jobs.
// - Start/stop/pause/resume; runNow(); remove(); enable/disable.
// - Observability via .on(event, handler) and .stats().
//
// Events:
//   "start"                 -> {}
//   "stop"                  -> {}
//   "pause"                 -> {}
//   "resume"                -> {}
//   "drain"                 -> { pending, running }
//   "task:add"              -> { id, name }
//   "task:remove"           -> { id }
//   "task:enable"           -> { id }
//   "task:disable"          -> { id }
//   "task:scheduled"        -> { id, nextRun }
//   "task:run:start"        -> { id, attempt }
//   "task:run:success"      -> { id, ms, attempt, result }
//   "task:run:failure"      -> { id, ms, attempt, error, willRetry }
//   "task:skipped"          -> { id, reason }
//
// Example:
//   const sch = createScheduler({ tickMs: 250, concurrency: 4 });
//   const id = sch.add({
//     name: "heartbeat",
//     intervalMs: 1000,
//     runOnStart: true,
//     fn: async () => console.log("beat"),
//   });
//   sch.start();

export type SchedulerOptions = {
  tickMs?: number;           // scheduler tick (polling) frequency
  concurrency?: number;      // max parallel tasks
  defaultTimeoutMs?: number; // per-run timeout (0=none)
  defaultRetry?: number;     // default retries
  defaultBackoff?: "none" | "fixed" | "exp";
  defaultBackoffMs?: number; // base backoff ms
  defaultJitter?: boolean;   // add jitter to backoff
};

export type TaskFn<T = unknown> = () => Promise<T> | T;

export type TaskSpec = {
  name?: string;
  fn: TaskFn<any>;
  intervalMs?: number;       // if set: repeating run every interval
  delayMs?: number;          // initial delay for first run (one-shot if no interval)
  runAt?: number;            // absolute epoch ms for first/only run
  runOnStart?: boolean;      // run immediately on scheduler.start()
  enabled?: boolean;         // default true
  timeoutMs?: number;        // override run timeout
  retry?: number;            // max retries
  backoff?: "none" | "fixed" | "exp";
  backoffMs?: number;
  jitter?: boolean;
};

export type Task = TaskSpec & {
  id: string;
  createdAt: number;
  lastRun?: number;
  nextRun?: number;
  attempts: number;
  enabled: boolean;
  running: boolean;
  _timer?: any; // internal
};

export type SchedulerEventMap = {
  start:   (e: {}) => void;
  stop:    (e: {}) => void;
  pause:   (e: {}) => void;
  resume:  (e: {}) => void;
  drain:   (e: { pending: number; running: number }) => void;

  "task:add":       (e: { id: string; name?: string }) => void;
  "task:remove":    (e: { id: string }) => void;
  "task:enable":    (e: { id: string }) => void;
  "task:disable":   (e: { id: string }) => void;
  "task:scheduled": (e: { id: string; nextRun?: number }) => void;
  "task:skipped":   (e: { id: string; reason: string }) => void;

  "task:run:start":   (e: { id: string; attempt: number }) => void;
  "task:run:success": (e: { id: string; ms: number; attempt: number; result: unknown }) => void;
  "task:run:failure": (e: { id: string; ms: number; attempt: number; error: unknown; willRetry: boolean }) => void;
};

export type Scheduler = {
  start(): void;
  stop(): void;
  pause(): void;
  resume(): void;

  add(spec: TaskSpec): string;
  remove(id: string): boolean;
  enable(id: string): boolean;
  disable(id: string): boolean;
  runNow(id: string): boolean;

  list(): Task[];
  stats(): {
    tickMs: number;
    concurrency: number;
    running: number;
    queued: number;
    tasks: number;
    started: boolean;
    paused: boolean;
  };

  on<K extends keyof SchedulerEventMap>(ev: K, fn: SchedulerEventMap[K]): () => void;
};

export function createScheduler(opts?: SchedulerOptions): Scheduler {
  const cfg = normalizeOptions(opts);

  const tasks = new Map<string, Task>();
  const listeners: Partial<{ [K in keyof SchedulerEventMap]: Set<SchedulerEventMap[K]> }> = {};
  let tickHandle: any = null;
  let started = false;
  let paused = false;

  // run queue + concurrency gate
  const queue: string[] = [];
  const running = new Map<string, { startedAt: number; timer?: any }>();

  // ------------------------------ Events ------------------------------

  function on<K extends keyof SchedulerEventMap>(ev: K, fn: SchedulerEventMap[K]) {
    
    return () => { try { (listeners[ev] as Set<any>).delete(fn as any); } catch {} };
  }

  function emit<K extends keyof SchedulerEventMap>(ev: K, data: Parameters<SchedulerEventMap[K]>[0]) {
    const set = listeners[ev]; if (!set || set.size === 0) return;
    for (const fn of Array.from(set)) {
      try { (fn as any)(data); } catch {}
    }
  }

  // ----------------------------- Control ------------------------------

  function start() {
    if (started) return;
    started = true;
    paused = false;
    scheduleAll();
    tickHandle = setInterval(tick, cfg.tickMs);
    emit("start", {});
  }

  function stop() {
    if (!started) return;
    started = false;
    paused = false;
    if (tickHandle) { clearInterval(tickHandle); tickHandle = null; }
    // Do not kill running tasks; we just stop scheduling further runs.
    emit("stop", {});
  }

  function pause() {
    if (!started || paused) return;
    paused = true;
    emit("pause", {});
  }

  function resume() {
    if (!started || !paused) return;
    paused = false;
    emit("resume", {});
    tick();
  }

  // ------------------------------ Tasks --------------------------------

  function add(spec: TaskSpec): string {
    const id = uid();
    const now = Date.now();
    const t: Task = {
      id,
      name: spec.name || "task:" + id.slice(-4),
      fn: spec.fn,
      intervalMs: nOrU(spec.intervalMs),
      delayMs: nOrU(spec.delayMs),
      runAt: nOrU(spec.runAt),
      runOnStart: !!spec.runOnStart,
      enabled: spec.enabled !== false,
      timeoutMs: nOrU(spec.timeoutMs) ?? cfg.defaultTimeoutMs,
      retry: nOrU(spec.retry) ?? cfg.defaultRetry,
      backoff: spec.backoff || cfg.defaultBackoff,
      backoffMs: nOrU(spec.backoffMs) ?? cfg.defaultBackoffMs,
      jitter: spec.jitter !== false, // default true
      createdAt: now,
      attempts: 0,
      running: false,
    };

    // compute initial nextRun
    t.nextRun = computeFirstNextRun(t, now, started);

    tasks.set(id, t);
    emit("task:add", { id, name: t.name });
    emit("task:scheduled", { id, nextRun: t.nextRun });
    return id;
  }

  function remove(id: string): boolean {
    // cannot remove if currently running; just mark disabled
    const r = tasks.get(id);
    if (!r) return false;
    if (running.has(id)) {
      r.enabled = false;
      emit("task:disable", { id });
      return true;
    }
    const ok = tasks.delete(id);
    if (ok) emit("task:remove", { id });
    // also sweep queue
    for (let i = queue.length - 1; i >= 0; i--) if (queue[i] === id) queue.splice(i, 1);
    return ok;
  }

  function enable(id: string): boolean {
    const t = tasks.get(id);
    if (!t) return false;
    if (t.enabled) return true;
    t.enabled = true;
    if (!t.nextRun) t.nextRun = Date.now(); // schedule asap
    emit("task:enable", { id });
    emit("task:scheduled", { id, nextRun: t.nextRun });
    return true;
  }

  function disable(id: string): boolean {
    const t = tasks.get(id);
    if (!t) return false;
    if (!t.enabled) return true;
    t.enabled = false;
    emit("task:disable", { id });
    return true;
  }

  function runNow(id: string): boolean {
    const t = tasks.get(id);
    if (!t) return false;
    t.nextRun = Date.now();
    emit("task:scheduled", { id, nextRun: t.nextRun });
    tick();
    return true;
  }

  function list(): Task[] {
    return Array.from(tasks.values()).map((t) => ({ ...t }));
  }

  // --------------------------- Scheduling Loop ---------------------------

  function scheduleAll() {
    const now = Date.now();
   
  }

  function tick() {
    if (!started || paused) return;

    const now = Date.now();

    
    // run up to concurrency
    while (running.size < cfg.concurrency && queue.length > 0) {
      const id = queue.shift()!;
      const t = tasks.get(id);
      if (!t || !t.enabled) {
        emit("task:skipped", { id, reason: "missing or disabled" });
        continue;
      }
      launch(t);
    }

    maybeDrain();
  }

  function launch(t: Task) {
    const attemptsUsed = t.attempts; // attempts used in *current* run (reset after success/interval)
    const attempt = attemptsUsed + 1;

    t.running = true;
    emit("task:run:start", { id: t.id, attempt });

    const startedAt = Date.now();
    const rec = { startedAt, timer: undefined as any };
    running.set(t.id, rec);

    // timeout
    if (t.timeoutMs && t.timeoutMs > 0) {
      rec.timer = setTimeout(() => {
        finalize(false, new Error("scheduler: timeout"));
      }, t.timeoutMs);
    }

    // execute
    let finished = false;
    const finalize = (ok: boolean, val: unknown) => {
      if (finished) return;
      finished = true;
      clearTimeout(rec.timer);
      running.delete(t.id);

      const ms = Date.now() - startedAt;
      t.lastRun = startedAt;

      if (ok) {
        // success path
        emit("task:run:success", { id: t.id, ms, attempt, result: val });
        t.attempts = 0; // reset attempts on success
        // compute next run for repeating tasks
        if (t.intervalMs && t.intervalMs > 0 && t.enabled) {
          t.nextRun = Date.now() + t.intervalMs;
          emit("task:scheduled", { id: t.id, nextRun: t.nextRun });
        }
        t.running = false;
        tick();
        maybeDrain();
      } else {
        // failure path
        const remaining = (t.retry ?? cfg.defaultRetry) - attemptsUsed;
        const willRetry = remaining > 0 && t.enabled;

        emit("task:run:failure", { id: t.id, ms, attempt, error: val, willRetry });

        if (willRetry) {
          t.attempts = attemptsUsed + 1;
          const delay = computeBackoff(t.backoff || cfg.defaultBackoff, t.backoffMs ?? cfg.defaultBackoffMs, attempt, t.jitter !== false);
          setTimeout(() => {
            if (!t.enabled) {
              t.running = false;
              emit("task:skipped", { id: t.id, reason: "disabled before retry" });
              maybeDrain();
              return;
            }
            // re-run immediately by enqueuing now
            queue.unshift(t.id);
            t.running = false;
            tick();
          }, delay);
          return;
        }

        // give up
        t.attempts = 0;
        t.running = false;
        // for repeating tasks, schedule next interval despite failure
        if (t.intervalMs && t.intervalMs > 0 && t.enabled) {
          t.nextRun = Date.now() + t.intervalMs;
          emit("task:scheduled", { id: t.id, nextRun: t.nextRun });
        }
        tick();
        maybeDrain();
      }
    };

    try {
      const r = t.fn();
      if (r && typeof (r as any).then === "function") {
        (r as Promise<unknown>).then(
          (val) => finalize(true, val),
          (err) => finalize(false, err)
        );
      } else {
        finalize(true, r);
      }
    } catch (err) {
      finalize(false, err);
    }
  }

  function maybeDrain() {
    if (queue.length === 0 && running.size === 0) {
      emit("drain", { pending: 0, running: 0 });
    }
  }

  // ------------------------------ Public ------------------------------

  function stats() {
    return {
      tickMs: cfg.tickMs,
      concurrency: cfg.concurrency,
      running: running.size,
      queued: queue.length,
      tasks: tasks.size,
      started,
      paused,
    };
  }

  return {
    start,
    stop,
    pause,
    resume,
    add,
    remove,
    enable,
    disable,
    runNow,
    list,
    stats,
    on,
  };
}

/* ----------------------------- Utilities ----------------------------- */

function normalizeOptions(opts?: SchedulerOptions) {
  const o = opts || {};
  return {
    tickMs: clampInt(o.tickMs ?? 250, 10, 10_000),
    concurrency: clampInt(o.concurrency ?? 4, 1, 1024),
    defaultTimeoutMs: clampInt(o.defaultTimeoutMs ?? 0, 0, 86_400_000),
    defaultRetry: clampInt(o.defaultRetry ?? 0, 0, 100),
    defaultBackoff: (o.defaultBackoff || "fixed") as "none" | "fixed" | "exp",
    defaultBackoffMs: clampInt(o.defaultBackoffMs ?? 250, 0, 60_000),
    defaultJitter: o.defaultJitter !== false, // default true
  };
}

function computeFirstNextRun(t: Task, now: number, started: boolean): number | undefined {
  if (!t.enabled) return undefined;
  if (t.runOnStart && started) return now;
  if (t.runAt != null) return Math.max(now, t.runAt);
  if (t.delayMs != null) return now + Math.max(0, t.delayMs);
  if (t.intervalMs && t.intervalMs > 0) return now + t.intervalMs;
  // one-shot with no time hint -> run immediately when started
  return started ? now : undefined;
}

function computeBackoff(kind: "none" | "fixed" | "exp", baseMs: number, attempt: number, jitter: boolean) {
  let d = 0;
  switch (kind) {
    case "none": d = 0; break;
    case "fixed": d = baseMs; break;
    case "exp": {
      const pow = Math.min(10, attempt - 1); // cap exponent
      d = baseMs * Math.pow(2, pow);
      break;
    }
  }
  if (jitter && d > 0) d += Math.floor(Math.random() * Math.min(d, 500));
  return d;
}

function nOrU(x: any): number | undefined {
  const n = Number(x);
  return Number.isFinite(n) ? n : undefined;
}

function clampInt(n: number, min: number, max: number) {
  n = Number(n) | 0;
  return Math.max(min, Math.min(max, n));
}

function uid() { return Math.random().toString(36).slice(2, 10); }