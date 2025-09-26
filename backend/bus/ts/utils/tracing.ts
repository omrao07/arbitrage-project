// utils/tracing.ts
import { context, trace, Span, Tracer, SpanStatusCode } from "@opentelemetry/api";

let tracer: Tracer | null = null;

/**
 * Initialize OpenTelemetry tracer.
 * Call this once in app bootstrap (e.g. index.ts)
 */
export function initTracer(serviceName: string): Tracer {
  tracer = trace.getTracer(serviceName);
  return tracer;
}

/**
 * Get current tracer or fallback.
 */
export function getTracer(): Tracer {
  if (!tracer) {
    tracer = trace.getTracer("default-service");
  }
  return tracer!;
}

/**
 * Start a new span with automatic context handling.
 */
export function startSpan<T>(
  name: string,
  fn: (span: Span) => Promise<T> | T
): Promise<T> {
  const tracer = getTracer();

  return context.with(trace.setSpan(context.active(), tracer.startSpan(name)), async () => {
    const span = trace.getSpan(context.active())!;
    try {
      const result = await fn(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (err: any) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
      span.recordException(err);
      throw err;
    } finally {
      span.end();
    }
  });
}

/**
 * Add an event or metadata to an active span.
 */
export function addSpanEvent(span: Span | undefined, event: string, attrs?: Record<string, any>) {
  if (span) {
    span.addEvent(event, attrs);
  }
}

/**
 * Example usage with Kafka producer:
 *
 * await startSpan("kafka.produce", async (span) => {
 *   span.setAttribute("topic", "trades");
 *   await producer.send({ topic: "trades", messages: [{ value: "hello" }] });
 * });
 */

if (require.main === module) {
  // Quick self-test
  initTracer("test-service");
  startSpan("test-operation", async (span) => {
    addSpanEvent(span, "custom-event", { foo: "bar" });
    console.log("tracing test run");
  });
}