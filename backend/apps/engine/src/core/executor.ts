// core/executor.ts
// Minimal, import-free async job executor with queue, concurrency, retries,
// backoff, timeouts, cancellation, and lightweight event hooks.
//
// - No imports. Pure TS/JS.
// - Submit any function: () => Promise<T> | T
// - Control: pause/resume, cancel(id), flush(), drain()
// - Policies: concurrency, retry, backoff, timeout
// - Observability: on(event, fn), stats()
//
// Events:
//   "submit"   -> { id, ctx }
//   "start"    -> { id, attempt }
//   "success"  -> { id, result, ms, attempt }
//   "failure"  -> { id, error, ms, attempt, willRetry }
//   "cancel"   -> { id, reason }
//   "drain"    -> { pending: 0 }
//   "pause"    -> {}
//   "resume"   -> {}
//
// Example:
//   const ex = createExecutor({ concurrency: 4, retry: 2, timeoutMs: 5000 });
//   ex.on("success", e => console.log("ok", e.id));
//   const id = ex.submit(async () => { /* ... */ return 42; });
//   await ex.drain();

export type ExecutorOptions = {
  concurrency?: number;         // max parallel jobs (default 4)
  retry?: number;               // max retries on failure (default 0)
  timeoutMs?: number;           // per-attempt timeout (0 = no timeout)
  backoff?: "none" | "fixed" | "exp";  // backoff strategy between retries
  backoffMs?: number;           // base backoff milliseconds (default 250)
  jitter?: boolean;             // add jitter to backoff (default true)
  name?: string;                // optional label for stats
};

export type JobContext = {
  id: string;
  submittedAt: number;   // ts in ms
  attempts: number;      // attempts used so far
  maxRetry: number;      // from options
  status: "queued" | "running" | "done" | "failed" | "canceled";
  reason?: string;       // cancel reason
};

export type JobFn<T = unknown> = (ctx: JobContext) => Promise<T> | T;

export type ExecutorEventMap = {
  submit:  (e: { id: string; ctx: JobContext }) => void;
  start:   (e: { id: string; attempt: number }) => void;
  success: (e: { id: string; result: unknown; ms: number; attempt: number }) => void;
  failure: (e: { id: string; error: unknown; ms: number; attempt: number; willRetry: boolean }) => void;
  cancel:  (e: { id: string; reason?: string }) => void;
  drain:   (e: { pending: number }) => void;
  pause:   (e: {}) => void;
  resume:  (e: {}) => void;
};

export type Executor = {
  submit<T = unknown>(fn: JobFn<T>): string;
  wrap<T = unknown>(thunk: () => Promise<T> | T): string;  // sugar for submit
  cancel(id: string, reason?: string): boolean;
  pause(): void;
  resume(): void;
  flush(): void;             // clear queued (does not cancel running)
  drain(): Promise<void>;    // resolves when queue and running are empty
  on<K extends keyof ExecutorEventMap>(ev: K, fn: ExecutorEventMap[K]): () => void;
  stats(): {
    name: string;
    queued: number;
    running: number;
    completed: number;
    failed: number;
    canceled: number;
    concurrency: number;
    retry: number;
    timeoutMs: number;
  };
};

export function createExecutor(opts?: ExecutorOptions): Executor {
  const cfg = normalizeOptions(opts);

  // State
  const queue: { id: string; fn: JobFn; ctx: JobContext }[] = [];
  const running = new Map<string, { ctx: JobContext; startedAt: number; timer?: any }>();
  const canceled = new Set<string>();

  let completed = 0;
  let failed = 0;
  let canceledCount = 0;
  let paused = false;

  // Events
  const listeners: { [K in keyof ExecutorEventMap]?: Set<ExecutorEventMap[K]> } = {};

  function on<K extends keyof ExecutorEventMap>(ev: K, fn: ExecutorEventMap[K]) {
   
    return () => { try { (listeners[ev] as Set<any>).delete(fn as any); } catch {} };
  }
  function emit<K extends keyof ExecutorEventMap>(ev: K, data: Parameters<ExecutorEventMap[K]>[0]) {
    const set = listeners[ev]; if (!set || set.size === 0) return;
    for (const fn of Array.from(set)) {
      try { (fn as any)(data); } catch (err) { /* swallow */ }
    }
  }

  function submit<T>(fn: JobFn<T>): string {
    const id = uid();
    const ctx: JobContext = {
      id,
      submittedAt: Date.now(),
      attempts: 0,
      maxRetry: cfg.retry,
      status: "queued",
    };
    queue.push({ id, fn, ctx });
    emit("submit", { id, ctx });
    tick();
    return id;
  }

  function wrap<T>(thunk: () => Promise<T> | T): string {
    return submit<T>(() => thunk());
  }

  function cancel(id: string, reason?: string): boolean {
    // If in queue, remove it
    const idx = queue.findIndex((q) => q.id === id);
    if (idx >= 0) {
      queue[idx].ctx.status = "canceled";
      queue[idx].ctx.reason = reason || "canceled";
      queue.splice(idx, 1);
      canceled.add(id);
      canceledCount++;
      emit("cancel", { id, reason });
      return true;
    }
    // If running, mark it; runner checks before retrying
    if (running.has(id)) {
      canceled.add(id);
      const r = running.get(id)!;
      r.ctx.reason = reason || "canceled";
      // We cannot forcibly abort arbitrary job functions, but we will:
      // - NOT schedule retries
      // - Mark as canceled upon completion/timeout
      return true;
    }
    return false;
  }

  function pause() {
    if (paused) return;
    paused = true;
    emit("pause", {});
  }

  function resume() {
    if (!paused) return;
    paused = false;
    emit("resume", {});
    tick();
  }

  function flush() {
    // clear queued (do not touch running)
    while (queue.length) {
      const q = queue.pop()!;
      q.ctx.status = "canceled";
      q.ctx.reason = "flushed";
      canceled.add(q.id);
      canceledCount++;
      emit("cancel", { id: q.id, reason: "flushed" });
    }
    maybeDrain();
  }

  function drain(): Promise<void> {
    if (queue.length === 0 && running.size === 0) return Promise.resolve();
    return new Promise<void>((resolve) => {
      const off = on("drain", () => { off(); resolve(); });
      maybeDrain();
    });
  }

  function stats() {
    return {
      name: cfg.name,
      queued: queue.length,
      running: running.size,
      completed,
      failed,
      canceled: canceledCount,
      concurrency: cfg.concurrency,
      retry: cfg.retry,
      timeoutMs: cfg.timeoutMs,
    };
  }

  // ---------------------------- Internal --------------------------------

  function tick() {
    if (paused) return;
    while (running.size < cfg.concurrency && queue.length > 0) {
      const job = queue.shift()!;
      run(job);
    }
    maybeDrain();
  }

  function run(job: { id: string; fn: JobFn; ctx: JobContext }) {
    if (canceled.has(job.id)) {
      job.ctx.status = "canceled";
      emit("cancel", { id: job.id, reason: job.ctx.reason });
      maybeDrain();
      return;
    }

    job.ctx.status = "running";
    job.ctx.attempts++;
    const attempt = job.ctx.attempts;
    const startedAt = Date.now();
    emit("start", { id: job.id, attempt });

    const rec = { ctx: job.ctx, startedAt, timer: undefined as any };
    running.set(job.id, rec);

    let finished = false;

    const finalize = (okRes: boolean, val: unknown) => {
      if (finished) return;
      finished = true;
      clearTimeout(rec.timer);
      running.delete(job.id);

      const ms = Date.now() - startedAt;

      if (canceled.has(job.id)) {
        // treat as canceled regardless of outcome
        job.ctx.status = "canceled";
        canceledCount++;
        emit("cancel", { id: job.id, reason: job.ctx.reason });
      } else if (okRes) {
        job.ctx.status = "done";
        completed++;
        emit("success", { id: job.id, result: val, ms, attempt });
      } else {
        // Failure path
        const err = val;
        const remaining = job.ctx.maxRetry - (attempt - 1);
        const willRetry = !canceled.has(job.id) && remaining > 0;

        emit("failure", { id: job.id, error: err, ms, attempt, willRetry });

        if (willRetry) {
          // schedule retry with backoff
          const delay = computeBackoff(cfg, attempt);
          setTimeout(() => {
            if (canceled.has(job.id)) {
              job.ctx.status = "canceled";
              emit("cancel", { id: job.id, reason: job.ctx.reason });
              maybeDrain();
            } else {
              job.ctx.status = "queued";
              queue.unshift(job); // retry sooner
              tick();
            }
          }, delay);
          return;
        }

        job.ctx.status = "failed";
        failed++;
      }

      tick();
      maybeDrain();
    };

    // Timeout per attempt
    if (cfg.timeoutMs > 0) {
      rec.timer = setTimeout(() => {
        finalize(false, new Error("executor: timeout"));
      }, cfg.timeoutMs);
    }

    // Execute job
    try {
      const res = job.fn(job.ctx);
      if (res && typeof (res as any).then === "function") {
        (res as Promise<unknown>).then(
          (val) => finalize(true, val),
          (err) => finalize(false, err)
        );
      } else {
        finalize(true, res);
      }
    } catch (err) {
      finalize(false, err);
    }
  }

  function maybeDrain() {
    if (queue.length === 0 && running.size === 0) {
      emit("drain", { pending: 0 });
    }
  }

  return {
    submit,
    wrap,
    cancel,
    pause,
    resume,
    flush,
    drain,
    on,
    stats,
  };
}

/* ----------------------------- Utilities ------------------------------ */

function normalizeOptions(opts?: ExecutorOptions) {
  const o = opts || {};
  return {
    name: String(o.name || "executor"),
    concurrency: clampInt(o.concurrency ?? 4, 1, 1024),
    retry: clampInt(o.retry ?? 0, 0, 100),
    timeoutMs: clampInt(o.timeoutMs ?? 0, 0, 24 * 60 * 60 * 1000),
    backoff: (o.backoff || "fixed") as "none" | "fixed" | "exp",
    backoffMs: clampInt(o.backoffMs ?? 250, 0, 60_000),
    jitter: o.jitter !== false, // default true
  };
}

function computeBackoff(cfg: ReturnType<typeof normalizeOptions>, attempt: number) {
  const base = cfg.backoffMs;
  let delay = 0;
  switch (cfg.backoff) {
    case "none":
      delay = 0;
      break;
    case "fixed":
      delay = base;
      break;
    case "exp":
      // exponential backoff grows with attempts (1,2,4,8...) * base
      const pow = Math.min(10, attempt - 1); // cap exponent
      delay = base * Math.pow(2, pow);
      break;
  }
  if (cfg.jitter && delay > 0) {
    const jitter = Math.floor(Math.random() * Math.min(delay, 500));
    delay += jitter;
  }
  return delay;
}

function clampInt(n: number, min: number, max: number) {
  n = Number(n) | 0;
  return Math.max(min, Math.min(max, n));
}

function uid() {
  return Math.random().toString(36).slice(2, 10);
}