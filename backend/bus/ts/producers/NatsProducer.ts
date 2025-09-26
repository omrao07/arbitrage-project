// producers/nats_producer.ts
import { connect, headers, JSONCodec, NatsConnection, JetStreamClient } from "nats";
import * as os from "node:os";

export type NATSProducerOpts = {
  servers?: string;                  // "nats://127.0.0.1:4222,nats://host2:4222"
  name?: string;                     // client name
  // JetStream is lazy-initialized when first used
  defaultHeaders?: Record<string, string>;
  // retry/backoff (for publish failures)
  retries?: number;                  // default 3
  baseDelayMs?: number;              // default 200
  maxDelayMs?: number;               // default 3000
};

const jc = JSONCodec();

function backoff(attempt: number, base: number, cap: number): number {
  const raw = Math.min(base * Math.pow(2, attempt - 1), cap);
  return raw * (0.8 + Math.random() * 0.4); // jitter
}

function toNatsHeaders(h?: Record<string, string>) {
  const nh = headers();
  if (h) for (const [k, v] of Object.entries(h)) nh.set(k, String(v));
  return nh;
}

export class NATSJsonProducer {
  private nc!: NatsConnection;
  private js?: JetStreamClient;
  private opts: Required<NATSProducerOpts>;

  constructor(opts: NATSProducerOpts = {}) {
    this.opts = {
      servers: opts.servers ?? process.env.NATS_SERVERS ?? "nats://127.0.0.1:4222",
      name: opts.name ?? process.env.NATS_CLIENT_NAME ?? `${os.hostname()}-producer`,
      defaultHeaders: { "content-type": "application/json", ...(opts.defaultHeaders ?? {}) },
      retries: opts.retries ?? 3,
      baseDelayMs: opts.baseDelayMs ?? 200,
      maxDelayMs: opts.maxDelayMs ?? 3000,
    };
  }

  /** Connect once at startup */
  async start(): Promise<void> {
    this.nc = await connect({
      servers: this.opts.servers.split(","),
      name: this.opts.name,
    });
    process.once("SIGINT", () => this.close());
    process.once("SIGTERM", () => this.close());
  }

  /** Lazy JetStream init */
  private ensureJs(): JetStreamClient {
    if (!this.js) this.js = this.nc.jetstream();
    return this.js!;
  }

  /** Graceful shutdown */
  async close(): Promise<void> {
    try { await this.nc?.drain(); } catch {}
    try { await this.nc?.close(); } catch {}
  }

  /**
   * Plain pub/sub publish (no persistence/acks). Fire-and-forget.
   */
  async publish(subject: string, value: unknown, hdrs?: Record<string, string>): Promise<void> {
    const h = { ...this.opts.defaultHeaders, ...(hdrs || {}) };
    const nh = toNatsHeaders(h);

    const payload = jc.encode(value);
    let lastErr: any = null;

    for (let attempt = 1; attempt <= this.opts.retries; attempt++) {
      try {
        await this.nc.publish(subject, payload, { headers: nh });
        return;
      } catch (e) {
        lastErr = e;
        if (attempt >= this.opts.retries) break;
        const d = backoff(attempt, this.opts.baseDelayMs, this.opts.maxDelayMs);
        await new Promise((r) => setTimeout(r, d));
      }
    }
    throw lastErr;
  }

  /**
   * JetStream publish (persisted). Returns the server ack (sequence, stream, etc.).
   * Set msgId for idempotency (dedupe) within JS.
   */
  async publishJs(params: {
    subject: string;
    value: unknown;
    headers?: Record<string, string>;
    msgId?: string;              // JetStream de-duplication key
    expectStream?: string;       // optional stream name for expectations
    timeoutMs?: number;          // ack timeout
  }) {
    const js = this.ensureJs();
    const {
      subject, value, headers: hdrs, msgId, expectStream, timeoutMs = 2000,
    } = params;

    const h = { ...this.opts.defaultHeaders, ...(hdrs || {}) };
    const nh = toNatsHeaders(h);
    const payload = jc.encode(value);

    let lastErr: any = null;
    for (let attempt = 1; attempt <= this.opts.retries; attempt++) {
      try {
        const pa = await js.publish(subject, payload, {
          headers: nh,
          msgID: msgId,
          expect: expectStream ? { streamName: expectStream } : undefined,
          timeout: timeoutMs,
        });
        return pa; // { stream, seq, duplicate, ... }
      } catch (e) {
        lastErr = e;
        if (attempt >= this.opts.retries) break;
        const d = backoff(attempt, this.opts.baseDelayMs, this.opts.maxDelayMs);
        await new Promise((r) => setTimeout(r, d));
      }
    }
    throw lastErr;
  }
}

// ---------------- Example runner ----------------
if (require.main === module) {
  (async () => {
    const producer = new NATSJsonProducer();
    await producer.start();

    // Plain pub/sub
    await producer.publish("market.ticks.v1", { type: "tick", symbol: "AAPL", price: 191.25, ts: new Date().toISOString() });

    // JetStream (persisted)
    const ack = await producer.publishJs({
      subject: "news.headlines.v1",
      value: { type: "headline", id: "h-1", headline: "CPI cools", ts: new Date().toISOString() },
      msgId: "news-h-1",               // idempotent
      expectStream: process.env.NATS_STREAM || undefined,
    });
    console.log("[nats-producer] JS ack:", ack);

    await producer.close();
    // eslint-disable-next-line no-process-exit
    process.exit(0);
  })().catch(async (e) => {
    console.error("[nats-producer] fatal:", e);
    process.exit(1);
  });
}