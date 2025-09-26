// consumers/kafka_consumer.ts
import { Kafka, logLevel, Consumer, EachMessagePayload } from "kafkajs";
import * as os from "node:os";

export type KafkaConsumerOpts = {
  brokers?: string[];
  groupId?: string;
  clientId?: string;
  topics: string[];
  fromBeginning?: boolean;
  concurrency?: number; // partitionsConsumedConcurrently
  handler: (msg: {
    topic: string;
    partition: number;
    key: string | null;
    timestamp: number;
    headers: Record<string, string>;
    value: any;          // decoded JSON; falls back to raw string
    raw: Buffer;         // raw bytes if you need them
  }) => Promise<void> | void;
  logLevel?: logLevel;
};

function normHeaders(h?: Record<string, Buffer | string | undefined>): Record<string, string> {
  const out: Record<string, string> = {};
  if (!h) return out;
  for (const [k, v] of Object.entries(h)) {
    if (v === undefined) continue;
    out[k.toLowerCase()] = Buffer.isBuffer(v) ? v.toString("utf8") : String(v);
  }
  return out;
}

function decodeJson(raw: Buffer): any {
  try {
    return JSON.parse(raw.toString("utf8"));
  } catch {
    // Not JSON -> return string (or keep Buffer if you prefer)
    return raw.toString("utf8");
  }
}

export class KafkaJsonConsumer {
  private kafka: Kafka;
  private consumer: Consumer;
  private opts: KafkaConsumerOpts;

  constructor(opts: KafkaConsumerOpts) {
    const {
      brokers = (process.env.KAFKA_BROKERS || "localhost:9092").split(","),
      groupId = process.env.KAFKA_GROUP_ID || "hyper-os-consumer",
      clientId = process.env.KAFKA_CLIENT_ID || `${os.hostname()}-${process.pid}`,
      logLevel: lvl = logLevel.INFO,
    } = opts;

    this.kafka = new Kafka({ brokers, clientId, logLevel: lvl });
    this.consumer = this.kafka.consumer({ groupId, allowAutoTopicCreation: false });
    this.opts = opts;
  }

  async start(): Promise<void> {
    await this.consumer.connect();
    for (const t of this.opts.topics) {
      await this.consumer.subscribe({ topic: t, fromBeginning: !!this.opts.fromBeginning });
    }

    await this.consumer.run({
      partitionsConsumedConcurrently: Math.max(1, this.opts.concurrency ?? 1),
      eachMessage: async (payload: EachMessagePayload) => this.onMessage(payload),
    });

    process.once("SIGINT", () => this.stop());
    process.once("SIGTERM", () => this.stop());
  }

  async stop(): Promise<void> {
    try { await this.consumer.disconnect(); } catch { /* noop */ }
  }

  private async onMessage({ topic, partition, message }: EachMessagePayload) {
    const raw = message.value ?? Buffer.alloc(0);
    const headers = normHeaders(message.headers as any);
    const key = message.key ? message.key.toString("utf8") : null;
    const timestamp = Number(message.timestamp);

    const value = decodeJson(raw);

    await this.opts.handler({
      topic,
      partition,
      key,
      timestamp,
      headers,
      value,
      raw,
    });
  }
}

// -------- Example run --------
if (require.main === module) {
  (async () => {
    const c = new KafkaJsonConsumer({
      topics: (process.env.KAFKA_TOPICS || "market.ticks.v1").split(","),
      handler: async (msg) => {
        console.log(
          `[${msg.topic}] p${msg.partition} key=${msg.key} ts=${msg.timestamp} value=`,
          typeof msg.value === "object" ? JSON.stringify(msg.value) : msg.value
        );
      },
    });
    await c.start();
  })().catch((e) => {
    console.error("Fatal:", e);
    process.exit(1);
  });
}