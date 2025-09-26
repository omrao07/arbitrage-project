// producers/kafka_producer.ts
import {
  Kafka,
  Producer,
  logLevel,
  IHeaders,
  CompressionTypes,
  Message,
} from "kafkajs";
import * as os from "node:os";

export type KafkaProducerOpts = {
  brokers?: string[];                 // ["host1:9092","host2:9092"]
  clientId?: string;                  // default: "<hostname>-<pid>"
  acks?: 0 | 1 | -1;                  // 0=none, 1=leader, -1=all (default -1)
  idempotent?: boolean;               // default true (requires acks=-1)
  compression?: "none" | "gzip" | "snappy" | "lz4" | "zstd";
  retries?: number;                   // producer-level retries (default 5)
  logLevel?: logLevel;                // default INFO
  allowAutoTopicCreation?: boolean;   // default false
};

const DEFAULT_HEADERS: IHeaders = {
  "content-type": "application/json",
  "x-producer": "kafkajs-json",
};

function encode(value: unknown): Buffer {
  return Buffer.from(JSON.stringify(value));
}

function toCompression(t: KafkaProducerOpts["compression"]) {
  switch ((t || "none").toLowerCase()) {
    case "gzip":   return CompressionTypes.GZIP;
    case "snappy": return CompressionTypes.Snappy;
    case "lz4":    return CompressionTypes.LZ4;
    case "zstd":   return CompressionTypes.ZSTD;
    default:       return CompressionTypes.None;
  }
}

export class KafkaJsonProducer {
  private kafka: Kafka;
  private producer: Producer;
  private readonly opts: Required<Omit<KafkaProducerOpts, "compression">> & {
    compression: KafkaProducerOpts["compression"];
  };

  constructor(opts: KafkaProducerOpts = {}) {
    const brokers = opts.brokers ?? (process.env.KAFKA_BROKERS || "localhost:9092").split(",");
    const clientId = opts.clientId ?? process.env.KAFKA_CLIENT_ID ?? `${os.hostname()}-${process.pid}`;
    const acks = opts.acks ?? -1;
    const idempotent = opts.idempotent ?? true;
    const retries = opts.retries ?? 5;
    const lvl = opts.logLevel ?? logLevel.INFO;

    this.kafka = new Kafka({ brokers, clientId, logLevel: lvl });

    this.producer = this.kafka.producer({
      allowAutoTopicCreation: opts.allowAutoTopicCreation ?? false,
      idempotent,
      maxInFlightRequests: idempotent ? 5 : 1,
      retry: { retries },
    });

    this.opts = {
      brokers,
      clientId,
      acks,
      idempotent,
      retries,
      logLevel: lvl,
      allowAutoTopicCreation: opts.allowAutoTopicCreation ?? false,
      compression: opts.compression ?? (process.env.KAFKA_COMPRESSION as any) ?? "none",
    };
  }

  /** Connect once at startup */
  async start(): Promise<void> {
    await this.producer.connect();
    process.once("SIGINT", () => void this.close());
    process.once("SIGTERM", () => void this.close());
  }

  /** Flush + disconnect safely */
async close(): Promise<void> {
  try {
    // flush() is deprecated / removed in newer kafkajs
    await this.producer.disconnect();
    console.log("[kafka-producer] disconnected cleanly");
  } catch (err) {
    console.error("[kafka-producer] error during disconnect:", err);
  }
}
  /** Publish one JSON record */
  async publish(params: {
    topic: string;
    value: unknown;                         // JSON-serializable
    key?: string | Buffer | null;
    headers?: IHeaders;
    partition?: number;
  }): Promise<void> {
    const { topic, value, key = null, headers = {}, partition } = params;

    const msg: Message = {
      value: encode(value),
      key: typeof key === "string" ? Buffer.from(key) : key ?? undefined,
      headers: { ...DEFAULT_HEADERS, ...headers },
      partition,
    };

    await this.producer.send({
      topic,
      acks: this.opts.acks,
      compression: toCompression(this.opts.compression),
      messages: [msg],
    });
  }

  /** Publish a batch (same topic) */
  async publishBatch(params: {
    topic: string;
    records: Array<{ value: unknown; key?: string | Buffer | null; headers?: IHeaders; partition?: number }>;
  }): Promise<void> {
    const { topic, records } = params;
    const messages: Message[] = records.map(r => ({
      value: encode(r.value),
      key: typeof r.key === "string" ? Buffer.from(r.key) : r.key ?? undefined,
      headers: { ...DEFAULT_HEADERS, ...(r.headers || {}) },
      partition: r.partition,
    }));

    await this.producer.send({
      topic,
      acks: this.opts.acks,
      compression: toCompression(this.opts.compression),
      messages,
    });
  }
}

/* ---------------- Example runner ---------------- */
if (require.main === module) {
  (async () => {
    const producer = new KafkaJsonProducer();
    await producer.start();

    // single
    await producer.publish({
      topic: process.env.KAFKA_TOPIC || "market.ticks.v1",
      key: "AAPL",
      value: { type: "tick", symbol: "AAPL", price: 191.25, ts: new Date().toISOString() },
      headers: { "x-event-type": "market" },
    });

    // batch
    await producer.publishBatch({
      topic: process.env.KAFKA_TOPIC || "market.ticks.v1",
      records: [
        { key: "MSFT",  value: { type: "tick", symbol: "MSFT",  price: 420.1, ts: new Date().toISOString() } },
        { key: "GOOGL", value: { type: "tick", symbol: "GOOGL", price: 130.0, ts: new Date().toISOString() } },
      ],
    });

    await producer.close();
    // eslint-disable-next-line no-process-exit
    process.exit(0);
  })().catch(async (e) => {
    console.error("[kafka-producer] fatal:", e);
    process.exit(1);
  });
}