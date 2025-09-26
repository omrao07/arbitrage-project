// lib/runner.ts
// Pure TypeScript job runner. No imports. Works in Node 18+ / browsers.

/* ============================== Minimal Logger ============================== */

type LogLevel = "debug" | "info" | "warn" | "error";

function iso(ts = Date.now()) { return new Date(ts).toISOString(); }

function createLogger(name = "runner") {
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

/* ============================== Abort polyfill ============================= */

type AnyAbortController = { abort: () => void; signal: AbortSignal };
function newAbortController(): AnyAbortController {
  const AC: any = (globalThis as any).AbortController;
  if (typeof AC === "function") return new AC();
  // Minimal polyfill: only supports .aborted flag
  let aborted = false;
  const signal: any = {};
  Object.defineProperty(signal, "aborted", { get: () => aborted });
  return {
    abort() { aborted = true; },
    signal: signal as AbortSignal,
  };
}

/* ================================== Types ================================== */

export type JobFn = (signal: AbortSignal) => Promise<void> | void;

export type Job = {
  name: string;
  everyMs: number;      // base interval in ms
  jitterMs?: number;    // optional random 0..jitter added each cycle
  enabled?: boolean;    // default true
  run: JobFn;           // the work
};

export type RunnerOptions = {
  name?: string;        // logger namespace
  level?: LogLevel;     // (kept for future; minimal logger logs everything)
};

export type JobStats = {
  running: boolean;
  runs: number;
  fails: number;
  lastRun?: number;     // epoch ms
  lastOk?: boolean;
  nextInMs?: number;    // scheduled delay remaining (best-effort)
};

/* ================================= Runner ================================= */

type State = {
  timer?: any;
  running: boolean;
  abort?: AnyAbortController;
  runs: number;
  fails: number;
  lastRun?: number;
  lastOk?: boolean;
  nextDue?: number;     // epoch ms
  enabled: boolean;
};

export class Runner {
  private readonly log = createLogger("runner");
  private readonly states: Record<string, State> = {};
  private readonly jobs: Record<string, Job> = {};

  constructor(opts?: RunnerOptions) {
    if (opts?.name) (this as any).log = createLogger(String(opts.name));
  }

  /** Register and start a repeating job. Kicks immediately, then schedules. */
  register(job: Job): void {
    const j: Job = {
      name: job.name,
      everyMs: Math.max(1, job.everyMs | 0),
      jitterMs: Math.max(0, (job.jitterMs ?? 0) | 0),
      enabled: job.enabled !== false,
      run: job.run,
    };
    this.jobs[j.name] = j;

    if (!this.states[j.name]) {
      this.states[j.name] = { running: false, runs: 0, fails: 0, enabled: !!j.enabled };
    } else {
      this.states[j.name].enabled = !!j.enabled;
    }

    if (!j.enabled) {
      this.log.info("register(disabled)", { job: j.name });
      return;
    }

    const tick = async () => {
      const st = this.states[j.name];
      if (!st || !st.enabled) return;

      if (st.running) {
        this.log.warn("skip(overlap)", { job: j.name });
        this._scheduleNext(j); // still schedule the next tick
        return;
      }

      st.running = true;
      st.abort = newAbortController();
      st.lastRun = Date.now();
      const done = this.log.time(`run:${j.name}`, { job: j.name });

      try {
        await Promise.resolve(j.run(st.abort.signal));
        st.runs++;
        st.lastOk = true;
        done({ ok: true });
      } catch (e: any) {
        st.fails++;
        st.lastOk = false;
        this.log.error(`fail:${j.name}`, { err: e?.message || String(e) });
        done({ ok: false });
      } finally {
        st.running = false;
        this._scheduleNext(j);
      }
    };

    // kick immediately, then schedule subsequent runs
    tick().catch(() => { /* already logged */ });
  }

  /** Stop a specific job's schedule and abort if it's running. */
  stop(name: string): void {
    const st = this.states[name];
    if (!st) return;
    if (st.timer) { clearTimeout(st.timer); st.timer = undefined; }
    if (st.running && st.abort) { try { st.abort.abort(); } catch { /* ignore */ } }
    st.enabled = false;
    this.log.info("stopped", { job: name });
  }

  /** Stop all jobs and abort any in-flight work. */
  stopAll(): void {
    const names = Object.keys(this.states);
    for (let i = 0; i < names.length; i++) this.stop(names[i]);
    this.log.info("stopped(all)");
  }

  /** Manually run a job once (without altering its schedule). */
  async runOnce(name: string, signal?: AbortSignal): Promise<void> {
    const j = this.jobs[name]; const st = this.states[name];
    if (!j || !st) return;
    if (st.running) { this.log.warn("runOnce(skip:running)", { job: name }); return; }

    st.running = true;
    const ac = signal ? null : newAbortController();
    const sig = signal || (ac as AnyAbortController)!.signal;

    const done = this.log.time(`run:${name}:manual`, { job: name });
    try {
      await Promise.resolve(j.run(sig));
      st.runs++; st.lastOk = true; st.lastRun = Date.now();
      done({ ok: true });
    } catch (e: any) {
      st.fails++; st.lastOk = false; st.lastRun = Date.now();
      this.log.error(`fail:${name}:manual`, { err: e?.message || String(e) });
      done({ ok: false });
    } finally {
      st.running = false;
      if (ac) ac.abort(); // release references
    }
  }

  /** Get stats snapshot for a job, or all jobs if name omitted. */
  stats(name?: string): Record<string, JobStats> | JobStats | undefined {
    const toStats = (st: State): JobStats => ({
      running: st.running,
      runs: st.runs,
      fails: st.fails,
      lastRun: st.lastRun,
      lastOk: st.lastOk,
      nextInMs: st.nextDue ? Math.max(0, st.nextDue - Date.now()) : undefined,
    });

    if (name) {
      const st = this.states[name];
      return st ? toStats(st) : undefined;
    }
    const out: Record<string, JobStats> = {};
    const keys = Object.keys(this.states);
    for (let i = 0; i < keys.length; i++) out[keys[i]] = toStats(this.states[keys[i]]);
    return out;
  }

  /** Internal: schedule the next timer for a job. */
  private _scheduleNext(j: Job) {
    const st = this.states[j.name];
    if (!st || !st.enabled) return;

    // clear old timer if any
    if (st.timer) { clearTimeout(st.timer); st.timer = undefined; }

    const jitter = j.jitterMs ? Math.floor(Math.random() * j.jitterMs) : 0;
    const delay = Math.max(1, j.everyMs + jitter);
    st.nextDue = Date.now() + delay;

    st.timer = setTimeout(() => {
      // guard: job may have been stopped
      if (!this.states[j.name] || !this.states[j.name].enabled) return;
      // re-tick
      const run = async () => {
        const curr = this.states[j.name];
        if (!curr) return;
        if (curr.running) {
          this.log.warn("skip(overlap)", { job: j.name });
          this._scheduleNext(j);
          return;
        }
        curr.running = true;
        curr.abort = newAbortController();
        curr.lastRun = Date.now();
        const done = this.log.time(`run:${j.name}`, { job: j.name });
        try {
          await Promise.resolve(j.run(curr.abort.signal));
          curr.runs++;
          curr.lastOk = true;
          done({ ok: true });
        } catch (e: any) {
          curr.fails++;
          curr.lastOk = false;
          this.log.error(`fail:${j.name}`, { err: e?.message || String(e) });
          done({ ok: false });
        } finally {
          curr.running = false;
          this._scheduleNext(j);
        }
      };
      run().catch(() => { /* already logged */ });
    }, delay);
  }
}

export default Runner;