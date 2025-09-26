// consumers/redis_consumer.ts
import { createClient, RedisClientType } from "redis";

export type RedisConsumerOpts = {
  stream: string;              // e.g. "market_ticks"
  group: string;               // consumer group name
  consumer: string;            // consumer instance name
  handler: (msg: {
    id: string;
    values: Record<string, string>;
  }) => Promise<void> | void;
  blockMs?: number;            // how long to block while reading (default 5000)
  count?: number;              // max messages to read at once
  redisUrl?: string;           // override redis connection url
};

export class RedisStreamConsumer {
  private client: RedisClientType;
  private opts: RedisConsumerOpts;

  constructor(opts: RedisConsumerOpts) {
    this.opts = opts;
    this.client = createClient({
      url: opts.redisUrl || process.env.REDIS_URL || "redis://127.0.0.1:6379",
    });
  }

  async connect() {
    this.client.on("error", (err) => console.error("[Redis] Client error", err));
    await this.client.connect();
    console.log("[Redis] Connected");

    // Create consumer group if it doesn’t exist
    try {
      await this.client.xGroupCreate(
        this.opts.stream,
        this.opts.group,
        "0",
        { MKSTREAM: true }
      );
      console.log(`[Redis] Consumer group '${this.opts.group}' created on stream '${this.opts.stream}'`);
    } catch (err: any) {
      if (err?.message?.includes("BUSYGROUP")) {
        console.log(`[Redis] Consumer group '${this.opts.group}' already exists`);
      } else {
        throw err;
      }
    }
  }

  async consume() {
    const { stream, group, consumer, handler, blockMs = 5000, count = 1 } = this.opts;

    console.log(`[Redis] Consumer '${consumer}' listening on stream '${stream}' group '${group}'`);

    while (true) {
      try {
        const response = await this.client.xReadGroup(
          group,
          consumer,
          [{ key: stream, id: ">" }],
          { COUNT: count, BLOCK: blockMs }
        );

        if (response) {
          for (const record of response) {
            for (const message of record.messages) {
              await handler({ id: message.id, values: message.message });
              // Acknowledge
              await this.client.xAck(stream, group, message.id);
            }
          }
        }
      } catch (err) {
        console.error("[Redis] Error consuming messages:", err);
        await new Promise((res) => setTimeout(res, 1000)); // backoff
      }
    }
  }

  async disconnect() {
    await this.client.quit();
    console.log("[Redis] Disconnected");
  }
}

// ---- Example runner ----
if (require.main === module) {
  (async () => {
    const consumer = new RedisStreamConsumer({
      stream: "market_ticks",
      group: "analytics",
      consumer: "c1",
      handler: async (msg) => {
        console.log(`[Redis] Got message ${msg.id}:`, msg.values);
      },
    });
    await consumer.connect();
    await consumer.consume();
  })();
}