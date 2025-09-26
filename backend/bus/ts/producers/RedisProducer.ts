// producers/redis_producer.ts
import { createClient, RedisClientType } from "redis";

export type RedisProducerOpts = {
  url?: string;                 // e.g., "redis://127.0.0.1:6379"
  username?: string;
  password?: string;
  db?: number;
  retries?: number;             // default 3
  baseDelayMs?: number;         // default 200
  maxDelayMs?: number;          // default 3000
  defaultHeaders?: Record<string, string>; // added to message fields for streams
};

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
function jittered(attempt: number, base: number, cap: number) {
  const raw = Math.min(base * Math.pow(2, attempt - 1), cap);
  return raw * (0.8 + Math.random() * 0.4);
}

export class RedisProducer {
  private client: RedisClientType;
  private readonly opts: Required<Omit<RedisProducerOpts, "url"|"username"|"password"|"db">> &
    Pick<RedisProducerOpts, "url"|"username"|"password"|"db"|"defaultHeaders">;

  constructor(opts: RedisProducerOpts = {}) {
    this.opts = {
      url: opts.url ?? process.env.REDIS_URL ?? "redis://127.0.0.1:6379",
      username: opts.username ?? process.env.REDIS_USERNAME,
      password: opts.password ?? process.env.REDIS_PASSWORD,
      db: typeof opts.db === "number" ? opts.db : (process.env.REDIS_DB ? Number(process.env.REDIS_DB) : undefined),
      retries: opts.retries ?? 3,
      baseDelayMs: opts.baseDelayMs ?? 200,
      maxDelayMs: opts.maxDelayMs ?? 3000,
      defaultHeaders: { "content-type": "application/json", ...(opts.defaultHeaders ?? {}) },
    };

    this.client = createClient({
      url: this.opts.url,
      username: this.opts.username,
      password: this.opts.password,
      database: this.opts.db,
    });

    this.client.on("error", (e) => {
      // don't throw here; let callers decide on fatality
      console.error("[redis-producer] client error:", e);
    });
  }

  /** Connect once at startup */
  async start(): Promise<void> {
    await this.client.connect();
    process.once("SIGINT", () => void this.close());
    process.once("SIGTERM", () => void this.close());
  }

  /** Graceful shutdown */
  async close(): Promise<void> {
    try { await this.client.quit(); } catch (e) {
      // if quit fails (network), force close
      try { await this.client.disconnect(); } catch {}
    }
  }

  /**
   * Publish to classic Redis Pub/Sub (fire-and-forget).
   * `value` can be object or string; objects are JSON-stringified.
   */
  async publish(channel: string, value: unknown): Promise<void> {
    const payload = typeof value === "string" ? value : JSON.stringify(value);
    let lastErr: any = null;

    for (let attempt = 1; attempt <= this.opts.retries; attempt++) {
      try {
        await this.client.publish(channel, payload);
        return;
      } catch (e) {
        lastErr = e;
        if (attempt >= this.opts.retries) break;
        await sleep(jittered(attempt, this.opts.baseDelayMs, this.opts.maxDelayMs));
      }
    }
    throw lastErr;
  }

  /**
   * Append a record to a Redis Stream with XADD.
   * Fields are stored as strings; we add default headers and JSON payload.
   * Returns the generated message ID (e.g., "1715100640000-0").
   */
  async xadd(params: {
    stream: string;
    value: unknown;                            // will be JSON-stringified
    id?: string;                               // default "*"
    headers?: Record<string, string>;          // extra key/values
    maxLenApprox?: number;                     // use approximate trim (~) if provided
  }): Promise<string> {
    const { stream, value, id = "*", headers = {}, maxLenApprox } = params;
    const fields: Record<string, string> = {
      ...this.opts.defaultHeaders,
      ...headers,
      payload: typeof value === "string" ? value : JSON.stringify(value),
      ts: new Date().toISOString(),
    };

    let lastErr: any = null;
    for (let attempt = 1; attempt <= this.opts.retries; attempt++) {
      try {
        // Build XADD args
        const args: (string | number)[] = [stream];
        if (maxLenApprox && maxLenApprox > 0) {
          args.push("MAXLEN", "~", String(maxLenApprox));
        }
        args.push(id);

        // Flatten field map to [k1, v1, k2, v2, ...]
        for (const [k, v] of Object.entries(fields)) {
          args.push(k, v);
        }

        const msgId = await (this.client as any).xAdd(...args);
        return msgId as string;
      } catch (e) {
        lastErr = e;
        if (attempt >= this.opts.retries) break;
        await sleep(jittered(attempt, this.opts.baseDelayMs, this.opts.maxDelayMs));
      }
    }
    throw lastErr;
  }

  /**
   * Pipeline multiple stream appends efficiently.
   * Returns array of message IDs in the same order as records.
   */
  async xaddBatch(records: Array<{
    stream: string;
    value: unknown;
    id?: string;
    headers?: Record<string, string>;
    maxLenApprox?: number;
  }>): Promise<string[]> {
    const multi = this.client.multi();

    for (const r of records) {
      const fields: Record<string, string> = {
        ...this.opts.defaultHeaders,
        ...(r.headers || {}),
        payload: typeof r.value === "string" ? r.value : JSON.stringify(r.value),
        ts: new Date().toISOString(),
      };

      const args: (string | number)[] = [r.stream];
      if (r.maxLenApprox && r.maxLenApprox > 0) args.push("MAXLEN", "~", String(r.maxLenApprox));
      args.push(r.id ?? "*");
      for (const [k, v] of Object.entries(fields)) args.push(k, v);

      // @ts-ignore redis client types don't expose xAdd in multi as a typed method
      (multi as any).xAdd(...args);
    }

    const res = await multi.exec();
    // res is (string | Error)[] — normalize to string[] or throw first error
    const ids: string[] = [];
    if (!res) throw new Error("[redis-producer] MULTI/EXEC returned null");
    for (const item of res) {
      if (item instanceof Error) throw item;
      ids.push(String(item));
    }
    return ids;
  }
}

/* ---------------- Example runner ---------------- */
if (require.main === module) {
  (async () => {
    const p = new RedisProducer();
    await p.start();

    // Pub/Sub
    await p.publish("news.signals.v1", { type: "signal", id: "sig-1", symbol: "AAPL", score: 0.78 });

    // Streams
    const id1 = await p.xadd({
      stream: "market.ticks",
      value: { type: "tick", symbol: "AAPL", price: 191.25 },
      maxLenApprox: 100000, // keep last 100k entries approx
      headers: { "x-event-type": "market" },
    });
    console.log("[redis-producer] XADD id:", id1);

    const ids = await p.xaddBatch([
      { stream: "market.ticks", value: { type: "tick", symbol: "MSFT", price: 420.1 } },
      { stream: "market.ticks", value: { type: "tick", symbol: "GOOGL", price: 130.0 } },
    ]);
    console.log("[redis-producer] XADD batch ids:", ids);

    await p.close();
    // eslint-disable-next-line no-process-exit
    process.exit(0);
  })().catch((e) => {
    console.error("[redis-producer] fatal:", e);
    process.exit(1);
  });
}