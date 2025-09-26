// utils/retry.ts

/**
 * Sleep helper.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Exponential backoff with jitter.
 * attempt = 1 → baseDelay
 * attempt = 2 → baseDelay * 2
 * capped at maxDelay
 * random jitter ±20%
 */
export function backoffDelay(
  attempt: number,
  baseDelay: number,
  maxDelay: number
): number {
  const exp = Math.min(baseDelay * Math.pow(2, attempt - 1), maxDelay);
  const jitter = exp * (0.8 + Math.random() * 0.4);
  return jitter;
}

export type RetryOptions = {
  retries?: number;       // max retries (default 3)
  baseDelayMs?: number;   // base backoff (default 200ms)
  maxDelayMs?: number;    // cap (default 3000ms)
  onRetry?: (err: any, attempt: number) => void;
};

/**
 * Retry a promise-returning function with backoff + jitter.
 */
export async function retry<T>(
  fn: () => Promise<T>,
  opts: RetryOptions = {}
): Promise<T> {
  const retries = opts.retries ?? 3;
  const base = opts.baseDelayMs ?? 200;
  const cap = opts.maxDelayMs ?? 3000;

  let lastErr: any = null;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      opts.onRetry?.(err, attempt);

      if (attempt < retries) {
        const delay = backoffDelay(attempt, base, cap);
        await sleep(delay);
      }
    }
  }

  throw lastErr;
}

/* ---------------- Example usage ---------------- */
if (require.main === module) {
  (async () => {
    let count = 0;
    try {
      const result = await retry(
        async () => {
          count++;
          if (count < 3) throw new Error("fail");
          return "success!";
        },
        { retries: 5, onRetry: (e, a) => console.log(`retry #${a}:`, e.message) }
      );
      console.log("final result:", result);
    } catch (e) {
      console.error("failed after retries:", e);
    }
  })();
}