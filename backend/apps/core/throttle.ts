// core/throttles.ts
// Pure TypeScript utilities (ZERO imports) for:
//  • Token bucket & sliding-window rate limiters
//  • Semaphore/Mutex (concurrency control) + withPermit()
//  • Bulkhead queue (concurrency + bounded queue + shedding)
//  • Throttle / Debounce wrappers (sync or async fns)
//  • Retry with jittered backoff (exponential, linear, decorrelated)
//  • Circuit breaker (open/half-open/closed)
//  • Timeout & deadline helpers
//  • Per-key limiters (lazy buckets)
// Drop-in safe for "no imports" constraint.

/* ───────────────────────── Common Types ───────────────────────── */

export type Dict<T = any> = { [k: string]: T };

export type Clock = { now(): number };
const DefaultClock: Clock = { now: () => Date.now() };

function sleep(ms: number): Promise<void> { return new Promise(r => setTimeout(r, ms)); }
function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }
function rand() { return Math.random(); }

/* ───────────────────────── Token Bucket ───────────────────────── */

export class TokenBucket {
  private tokens: number;
  private lastRefill: number;
  private pausedUntil = 0;

  constructor(
    private capacity: number,
    private refillPerSec: number,
    private clock: Clock = DefaultClock
  ) {
    this.capacity = Math.max(0.0001, capacity);
    this.refillPerSec = Math.max(0, refillPerSec);
    this.tokens = this.capacity;
    this.lastRefill = clock.now();
  }

  /** Non-blocking: consume if available, else false. */
  tryTake(n = 1): boolean {
    this.refill();
    if (this.tokens + 1e-9 < n || this.clock.now() < this.pausedUntil) return false;
    this.tokens -= n;
    return true;
  }

  /** Blocking: wait until n tokens available, then consume. */
  async take(n = 1): Promise<void> {
    for (;;) {
      if (this.tryTake(n)) return;
      const wait = this.timeToAvailable(n);
      await sleep(Math.max(1, wait));
    }
  }

  /** Pause bucket until future time (ms epoch). */
  pauseUntil(tsMs: number) { this.pausedUntil = Math.max(this.pausedUntil, tsMs | 0); }

  /** Estimated ms until n tokens are available (0 if now). */
  timeToAvailable(n = 1): number {
    this.refill();
    if (this.clock.now() < this.pausedUntil) return this.pausedUntil - this.clock.now();
    if (this.tokens >= n) return 0;
    const need = n - this.tokens;
    const rate = this.refillPerSec / 1000; // tokens per ms
    if (rate <= 0) return 1e12; // effectively infinite
    return Math.ceil(need / rate);
  }

  private refill() {
    const now = this.clock.now();
    const dt = Math.max(0, now - this.lastRefill); // ms
    if (dt <= 0) return;
    const add = (this.refillPerSec / 1000) * dt;
    this.tokens = Math.min(this.capacity, this.tokens + add);
    this.lastRefill = now;
  }
}

/* ───────────────────────── Sliding-Window Limiter ───────────────────────── */

export class SlidingWindowLimiter {
  private hits: number[] = [];
  constructor(private limit: number, private windowMs: number, private clock: Clock = DefaultClock) {
    this.limit = Math.max(1, limit|0);
    this.windowMs = Math.max(1, windowMs|0);
  }

  /** Try to record an event; returns true if under limit. */
  try(): boolean {
    const now = this.clock.now();
    this.gc(now);
    if (this.hits.length >= this.limit) return false;
    this.hits.push(now);
    return true;
  }

  /** Wait until a slot is available, then record. */
  async acquire(): Promise<void> {
    for (;;) {
      if (this.try()) return;
      const now = this.clock.now();
      this.gc(now);
      const oldest = this.hits[0] || now;
      const wait = Math.max(1, oldest + this.windowMs - now);
      await sleep(wait);
    }
  }

  private gc(now: number) {
    const cutoff = now - this.windowMs;
    while (this.hits.length && this.hits[0] < cutoff) this.hits.shift();
  }
}

/* ───────────────────────── Semaphore / Mutex ───────────────────────── */

export class Semaphore {
  private queue: Array<(v?: unknown) => void> = [];
  private permits: number;

  constructor(maxConcurrency: number) {
    this.permits = Math.max(0, maxConcurrency|0);
  }

  /** Acquire 1 permit; resolves to a release() fn. */
  async acquire(): Promise<() => void> {
    if (this.permits > 0) { this.permits--; return () => this.release(); }
    return new Promise<() => void>(resolve => {
      this.queue.push(() => resolve(() => this.release()));
    });
  }

  /** Try acquire immediately; returns release() or null. */
  tryAcquire(): (() => void) | null {
    if (this.permits > 0) { this.permits--; return () => this.release(); }
    return null;
  }

  /** Run fn within a permit (always releases). */
  async withPermit<T>(fn: () => Promise<T> | T): Promise<T> {
    const release = await this.acquire();
    try { return await Promise.resolve(fn()); }
    finally { release(); }
  }

  private release() {
    const waiter = this.queue.shift();
    if (waiter) waiter();
    else this.permits++;
  }

  /** Current number of queued waiters. */
  get queued(): number { return this.queue.length; }
  /** Current available permits. */
  get available(): number { return this.permits; }
}

export class Mutex extends Semaphore {
  constructor() { super(1); }
}

/* ───────────────────────── Bulkhead Queue (bounded) ───────────────────────── */

export type BulkheadOpts = {
  maxConcurrency: number;
  maxQueue: number;           // pending tasks allowed
  shedPolicy?: "reject" | "oldest" | "newest";
};

export class Bulkhead {
  private sem: Semaphore;
  private queue: Array<{
    job: () => Promise<any> | any;
    resolve: (v: any) => void;
    reject: (e: any) => void;
  }> = [];
  private opts: Required<BulkheadOpts>;

  constructor(opts: BulkheadOpts) {
    this.opts = {
      maxConcurrency: Math.max(1, opts.maxConcurrency|0),
      maxQueue: Math.max(0, opts.maxQueue|0),
      shedPolicy: opts.shedPolicy || "reject"
    };
    this.sem = new Semaphore(this.opts.maxConcurrency);
  }

  /** Submit a job; may reject immediately if queue full (per shedPolicy). */
  submit<T>(job: () => Promise<T> | T): Promise<T> {
    // if a permit is free, run immediately
    const release = this.sem.tryAcquire();
    if (release) return this.runNow(job, release);

    // else enqueue
    if (this.queue.length >= this.opts.maxQueue) {
      if (this.opts.shedPolicy === "reject") {
        return Promise.reject(new Error("bulkhead.queue_full"));
      }
      if (this.opts.shedPolicy === "oldest") this.queue.shift();
      if (this.opts.shedPolicy === "newest") this.queue.pop();
    }

    return new Promise<T>((resolve, reject) => {
      this.queue.push({ job: job as any, resolve, reject });
      this.drain();
    });
  }

  private async runNow<T>(job: () => Promise<T> | T, release: () => void): Promise<T> {
    try { return await Promise.resolve(job()); }
    finally { release(); this.drain(); }
  }

  private async drain() {
    while (this.queue.length && this.sem.available > 0) {
      const next = this.queue.shift()!;
      const rel = this.sem.tryAcquire()!;
      this.runNow(next.job, rel).then(next.resolve, next.reject);
    }
  }

  get depth(): number { return this.queue.length; }
  get concurrency(): number { return this.opts.maxConcurrency; }
}

/* ───────────────────────── Throttle / Debounce ───────────────────────── */

export type ThrottleOptions = { leading?: boolean; trailing?: boolean };

export function throttle<F extends (...args: any[]) => any>(
  fn: F,
  waitMs: number,
  opts: ThrottleOptions = { leading: true, trailing: true }
): (...args: Parameters<F>) => ReturnType<F> | undefined {
  let lastCall = 0;
  let timeout: any = null;
  let lastArgs: any[] | null = null;
  let lastThis: any = null;
  const leading = opts.leading !== false;
  const trailing = opts.trailing !== false;

  function invoke() {
    lastCall = Date.now();
    // @ts-ignore
    const r = fn.apply(lastThis, lastArgs || []);
    lastArgs = lastThis = null;
    return r;
  }

  return function(this: any, ...args: any[]) {
    const now = Date.now();
    const remaining = waitMs - (now - lastCall);
    lastArgs = args; lastThis = this;

    if (remaining <= 0 || remaining > waitMs) {
      if (timeout) { clearTimeout(timeout); timeout = null; }
      if (leading || lastCall !== 0) return invoke();
      lastCall = now;
      return undefined;
    }

    if (trailing && !timeout) {
      timeout = setTimeout(() => {
        timeout = null;
        if (trailing && lastArgs) invoke();
      }, remaining);
    }
    return undefined;
  };
}

export type DebounceOptions = { leading?: boolean; maxWaitMs?: number };

export function debounce<F extends (...args: any[]) => any>(
  fn: F,
  waitMs: number,
  opts: DebounceOptions = {}
): (...args: Parameters<F>) => Promise<ReturnType<F>> {
  let timeout: any = null;
  let lastInvoke = 0;
  let pendingResolve: ((v: any) => void) | null = null;
  const leading = !!opts.leading;
  const maxWait = Math.max(0, opts.maxWaitMs ?? 0);

  return function(this: any, ...args: any[]) {
    if (timeout) clearTimeout(timeout);

    const now = Date.now();
    const shouldInvokeNow = leading && (now - lastInvoke) >= waitMs;

    return new Promise<ReturnType<F>>(resolve => {
      pendingResolve = resolve;

      const invoke = () => {
        lastInvoke = Date.now();
        const r = fn.apply(this, args);
        Promise.resolve(r).then(v => pendingResolve && pendingResolve(v));
        pendingResolve = null;
      };

      if (shouldInvokeNow) {
        invoke();
      } else {
        const delay = waitMs;
        timeout = setTimeout(invoke, delay);
        if (maxWait > 0) {
          const elapsed = now - lastInvoke;
          const remaining = Math.max(0, maxWait - elapsed);
          if (remaining <= delay) {
            clearTimeout(timeout);
            timeout = setTimeout(invoke, remaining);
          }
        }
      }
    });
  };
}

/* ───────────────────────── Retry & Backoff ───────────────────────── */

export type BackoffStrategy = (attempt: number) => number; // ms to wait BEFORE attempt (attempt starts at 1)
export type RetryOpts = {
  attempts?: number;                 // total attempts including first (default 5)
  shouldRetry?: (e: any) => boolean; // default always retry
  backoff?: BackoffStrategy;         // default exp backoff with jitter
  onRetry?: (e: any, attempt: number, delayMs: number) => void;
  maxDelayMs?: number;
};

export const backoff = {
  exponential(baseMs = 200, factor = 2, jitter = 0.25, capMs = 30_000): BackoffStrategy {
    return (attempt: number) => {
      const raw = baseMs * Math.pow(factor, Math.max(0, attempt - 1));
      const jittered = raw * (1 - jitter + 2 * jitter * rand());
      return Math.min(capMs, Math.floor(jittered));
    };
  },
  linear(stepMs = 200, jitter = 0.25, capMs = 30_000): BackoffStrategy {
    return (attempt: number) => {
      const raw = stepMs * Math.max(1, attempt);
      const jittered = raw * (1 - jitter + 2 * jitter * rand());
      return Math.min(capMs, Math.floor(jittered));
    };
  },
  decorrelated(baseMs = 200, maxMs = 30_000): BackoffStrategy {
    let prev = baseMs;
    return (_attempt: number) => {
      const next = Math.min(maxMs, Math.floor(baseMs + rand() * prev * 3));
      prev = next;
      return next;
    };
  }
};

export async function retry<T>(fn: () => Promise<T> | T, opts?: RetryOpts): Promise<T> {
  const attempts = Math.max(1, opts?.attempts ?? 5);
  const should = opts?.shouldRetry ?? (() => true);
  const bo = opts?.backoff ?? backoff.exponential();
  for (let i = 1; i <= attempts; i++) {
    try { return await Promise.resolve(fn()); }
    catch (e) {
      if (i >= attempts || !should(e)) throw e;
      const delay = clamp(bo(i), 0, opts?.maxDelayMs ?? 60_000);
      try { opts?.onRetry?.(e, i, delay); } catch {}
      await sleep(delay);
    }
  }
  // @ts-ignore
  return undefined;
}

/* ───────────────────────── Circuit Breaker ───────────────────────── */

export type BreakerState = "closed" | "open" | "half_open";
export type BreakerOpts = {
  failureThreshold?: number;   // open when failures >= this within window (default 5)
  windowMs?: number;           // rolling window for counts (default 30_000)
  cooldownMs?: number;         // time to stay open before half-open (default 10_000)
  halfOpenMaxInFlight?: number;// allow this many test calls in half-open (default 1)
  isFailure?: (e: any) => boolean; // classify failures
};

export class CircuitBreaker {
  private state: BreakerState = "closed";
  private openedAt = 0;
  private windowMs: number;
  private failThresh: number;
  private cooldownMs: number;
  private halfOpenMax: number;
  private fails: number[] = [];
  private inFlightHalf = 0;
  private isFailure: (e:any)=>boolean;

  constructor(opts?: BreakerOpts, private clock: Clock = DefaultClock) {
    this.failThresh = Math.max(1, opts?.failureThreshold ?? 5);
    this.windowMs = Math.max(1000, opts?.windowMs ?? 30_000);
    this.cooldownMs = Math.max(1000, opts?.cooldownMs ?? 10_000);
    this.halfOpenMax = Math.max(1, opts?.halfOpenMaxInFlight ?? 1);
    this.isFailure = opts?.isFailure || (() => true);
  }

  /** Wrap a call; throws BreakerOpenError if open & not ready. */
  async exec<T>(fn: () => Promise<T> | T): Promise<T> {
    const now = this.clock.now();
    this.roll(now);

    if (this.state === "open") {
      if (now - this.openedAt >= this.cooldownMs) {
        this.state = "half_open";
        this.inFlightHalf = 0;
      } else {
        throw new BreakerOpenError("circuit_open");
      }
    }

    if (this.state === "half_open") {
      if (this.inFlightHalf >= this.halfOpenMax) throw new BreakerOpenError("half_open_saturated");
      this.inFlightHalf++;
      try {
        const res = await Promise.resolve(fn());
        this.success(now);
        return res;
      } catch (e) {
        this.failure(now, e);
        throw e;
      } finally {
        this.inFlightHalf = Math.max(0, this.inFlightHalf - 1);
      }
    }

    // closed
    try {
      const res = await Promise.resolve(fn());
      this.success(now);
      return res;
    } catch (e) {
      this.failure(now, e);
      throw e;
    }
  }

  private success(now: number) {
    if (this.state === "half_open") {
      // success closes the breaker
      this.state = "closed";
      this.fails = [];
    }
  }

  private failure(now: number, e: any) {
    if (!this.isFailure(e)) return;
    this.fails.push(now);
    this.roll(now);
    if (this.state === "half_open") {
      this.trip(now);
    } else if (this.state === "closed" && this.fails.length >= this.failThresh) {
      this.trip(now);
    }
  }

  private trip(now: number) {
    this.state = "open";
    this.openedAt = now;
  }

  private roll(now: number) {
    const cutoff = now - this.windowMs;
    while (this.fails.length && this.fails[0] < cutoff) this.fails.shift();
  }

  getState(): BreakerState { return this.state; }
}

export class BreakerOpenError extends Error {
  constructor(msg = "circuit_open") { super(msg); this.name = "BreakerOpenError"; }
}

/* ───────────────────────── Timeout / Deadline ───────────────────────── */

export function withTimeout<T>(p: Promise<T>, ms: number, message = "timeout"): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(message)), Math.max(1, ms|0));
    p.then(v => { clearTimeout(t); resolve(v); }, e => { clearTimeout(t); reject(e); });
  });
}

export function deadlineMs(msFromNow: number): number { return Date.now() + Math.max(0, msFromNow|0); }
export function timeLeft(deadlineEpochMs: number): number { return Math.max(0, deadlineEpochMs - Date.now()); }

/* ───────────────────────── Per-key Limiters ───────────────────────── */

export class PerKeyBuckets {
  private map = new Map<string, TokenBucket>();
  constructor(private mk: (key: string) => TokenBucket) {}

  get(key: string): TokenBucket {
    const k = String(key);
    let b = this.map.get(k);
    if (!b) { b = this.mk(k); this.map.set(k, b); }
    return b;
  }

  async take(key: string, n = 1) { return this.get(key).take(n); }
  tryTake(key: string, n = 1) { return this.get(key).tryTake(n); }
}

/* ───────────────────────── Tiny self-test (optional) ───────────────────────── */

export async function __selftest__(): Promise<string> {
  // Token bucket
  const tb = new TokenBucket(2, 2); // 2 tokens, 2/sec
  const t0 = tb.tryTake(2);
  const t1 = tb.tryTake(1); // should be false immediately
  if (!t0 || t1) return "tb_fail";
  await tb.take(1); // wait ~0.5s
  // Semaphore
  const sem = new Semaphore(2);
  let inFlight = 0, maxIF = 0;
  const run = async () => sem.withPermit(async () => { inFlight++; maxIF = Math.max(maxIF, inFlight); await sleep(20); inFlight--; });
  await Promise.all([run(), run(), run(), run()]);
  if (maxIF > 2) return "sem_fail";
  // Debounce
  let hits = 0;
  const deb = debounce(() => { hits++; return hits; }, 30, { maxWaitMs: 50 });
  await Promise.all([deb(), deb(), deb()]);
  if (hits !== 1) return "debounce_fail";
  // Retry
  let tries = 0;
  await retry(async () => { tries++; if (tries < 3) throw new Error("x"); return 42; }, { attempts: 5 });
  if (tries !== 3) return "retry_fail";
  // Breaker
  const br = new CircuitBreaker({ failureThreshold: 2, windowMs: 500, cooldownMs: 100 });
  let threw = false;
  try { await br.exec(async () => { throw new Error("boom"); }); } catch {}
  try { await br.exec(async () => { throw new Error("boom"); }); } catch {}
  try { await br.exec(async () => 1); } catch (e) { threw = e instanceof BreakerOpenError; }
  if (!threw) return "breaker_fail";
  return "ok";
}