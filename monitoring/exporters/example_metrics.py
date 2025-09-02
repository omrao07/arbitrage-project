# platform/example_metrics.py
"""
Example usage of the metrics layer (Prometheus + OTEL).
Run this standalone to see counters, gauges, histograms updating.
"""

import random
import time
from platform import bootstrap # type: ignore

SERVICE = "example-metrics"


def main():
    # init() will set up redis, tracer, metrics, etc.
    ctx = bootstrap.init(SERVICE)
    METRICS = ctx["metrics"]

    # Get handles (with labels pre-bound by bootstrap)
    tasks_total = METRICS.tasks_total.labels(SERVICE, "demo")
    task_errors = METRICS.task_errors.labels(SERVICE, "demo")
    latency = METRICS.latency_histogram("demo_latency", buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5))
    gauge = METRICS.gauge("demo_inflight")

    # Loop: simulate 100 iterations
    for i in range(100):
        # start a timer for latency measurement
        with latency.time():
            # simulate some work
            t = random.random()
            gauge.set(i % 5)  # cycle inflight count
            time.sleep(t * 0.2)  # 0..200ms

            # increment counters
            tasks_total.inc()
            if t < 0.1:
                task_errors.inc()  # simulate occasional error

        if i % 10 == 0:
            print(f"iteration {i}, slept {t*0.2:.3f}s")

    print("Done. Metrics are being served on /metrics via Prometheus client.")


if __name__ == "__main__":
    main()