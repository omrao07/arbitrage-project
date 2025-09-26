import { consumerOpts, JsMsg, JetStreamClient, NatsConnection, connect } from "nats";

class NatsJetStreamConsumer {
  private nc!: NatsConnection;
  private js!: JetStreamClient;

  async connect() {
    this.nc = await connect({ servers: process.env.NATS_SERVERS || "nats://127.0.0.1:4222" });
    this.js = this.nc.jetstream();
    console.log("[NATS] Connected to JetStream");
  }

  async subscribe(subject: string, durableName = "ts-consumer") {
    const opts = consumerOpts();
    opts.durable(durableName);
    opts.manualAck(); // we’ll handle ack manually
    opts.ackExplicit();
    opts.deliverTo(`deliver-${durableName}-${Date.now()}`);

    const sub = await this.js.subscribe(subject, opts);

    (async () => {
      for await (const m of sub) {
        const jsMsg = m as JsMsg;

        try {
          await this.handleMessage(
            jsMsg.data,
            jsMsg.headers,
            jsMsg.seq,
            jsMsg.subject,
            async () => jsMsg.ack(),
            async (delayMs?: number) => jsMsg.nak(delayMs)
          );
        } catch (err) {
          console.error("[NATS] Error handling message:", err);
          await jsMsg.nak(500); // retry after 500ms
        }
      }
    })();
  }

  // Example message handler — extend as needed
  private async handleMessage(
    data: Uint8Array,
    headers: any,
    seq: number,
    subject: string,
    ack: () => Promise<void>,
    nak: (delayMs?: number) => Promise<void>
  ) {
    console.log(`[NATS] [${subject}] #${seq}:`, new TextDecoder().decode(data));
    await ack();
  }
}

// Run directly
if (require.main === module) {
  (async () => {
    const consumer = new NatsJetStreamConsumer();
    await consumer.connect();
    await consumer.subscribe("market.ticks.v1", "ticks-durable");
  })();
}