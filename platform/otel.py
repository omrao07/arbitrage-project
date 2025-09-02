# platform/otel.py
from __future__ import annotations

import logging
import os
import socket
import threading
from contextlib import contextmanager
from typing import Optional

# ---- OpenTelemetry (tracing) ----
from opentelemetry import trace 
from opentelemetry.sdk.resources import Resource 
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter 

# ---- Prometheus (metrics) ----
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest,
    start_http_server,  # default serves /metrics
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Singleton guards
_TRACER_INIT_LOCK = threading.Lock()
_TRACER_INITIALIZED = False

_METRICS_INIT_LOCK = threading.Lock()
_METRICS_INITIALIZED = False


class WorkerMetrics:
    """
    Prometheus metrics helper. Exposes:
      - worker_task_latency_seconds (Histogram, labels: worker, task)
      - worker_tasks_total (Counter, labels: worker, task)
      - worker_task_errors_total (Counter, labels: worker, task)
      - worker_dlq_total (Counter, labels: worker)
      - worker_queue_lag_seconds (Gauge, labels: worker)
      - worker_queue_lag_seconds_sum (Gauge, labels: worker)  # to match existing alert rules
    """

    def __init__(self, service_name: str, registry: Optional[CollectorRegistry] = None):
        self.worker = service_name
        self.registry = registry

        # Buckets tuned for sub-second to tens-of-seconds jobs
        self.task_latency = Histogram(
            "worker_task_latency_seconds",
            "Task latency in seconds",
            ["worker", "task"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
            registry=registry,
        )
        self.tasks_total = Counter(
            "worker_tasks_total",
            "Total tasks processed",
            ["worker", "task"],
            registry=registry,
        )
        self.task_errors = Counter(
            "worker_task_errors_total",
            "Total task errors",
            ["worker", "task"],
            registry=registry,
        )
        self.dlq_total = Counter(
            "worker_dlq_total",
            "Messages sent to DLQ",
            ["worker"],
            registry=registry,
        )
        self.queue_lag = Gauge(
            "worker_queue_lag_seconds",
            "Current estimated queue lag in seconds",
            ["worker"],
            registry=registry,
        )
        # To align with previously shared alert/recording rules
        self.queue_lag_sum = Gauge(
            "worker_queue_lag_seconds_sum",
            "Compatibility gauge for queue lag (used by some rules)",
            ["worker"],
            registry=registry,
        )

    @contextmanager
    def latency_timer(self, task: str):
        """Context manager to observe task latency and auto-count success/error."""
        import time

        start = time.perf_counter()
        err = None
        try:
            yield
        except Exception as e:  # record error then re-raise
            err = e
            raise
        finally:
            dur = time.perf_counter() - start
            self.task_latency.labels(self.worker, task).observe(dur)
            self.tasks_total.labels(self.worker, task).inc()
            if err is not None:
                self.task_errors.labels(self.worker, task).inc()

    def inc_task(self, task: str, *, error: bool = False):
        self.tasks_total.labels(self.worker, task).inc()
        if error:
            self.task_errors.labels(self.worker, task).inc()

    def inc_dlq(self, n: float = 1.0):
        self.dlq_total.labels(self.worker).inc(n)

    def set_queue_lag(self, seconds: float):
        self.queue_lag.labels(self.worker).set(max(0.0, float(seconds)))
        # Keep the _sum gauge in sync for compatibility with existing rules
        self.queue_lag_sum.labels(self.worker).set(max(0.0, float(seconds)))


def _init_tracer(service_name: str) -> trace.Tracer:
    global _TRACER_INITIALIZED
    with _TRACER_INIT_LOCK:
        if not _TRACER_INITIALIZED:
            env = os.getenv("DEPLOY_ENV", os.getenv("ENV", "production"))
            svc_ver = os.getenv("SERVICE_VERSION", os.getenv("GIT_SHA", "0.0.0"))
            region = os.getenv("REGION", "")

            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.version": svc_ver,
                    "deployment.environment": env,
                    "service.instance.id": socket.gethostname(),
                    "service.region": region,
                }
            )
            provider = TracerProvider(resource=resource)

            # OTLP HTTP exporter (4318). Override via OTEL_EXPORTER_OTLP_ENDPOINT env.
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

            # Optional console exporter for local debugging
            if os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() in ("1", "true", "yes"):
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

            trace.set_tracer_provider(provider)
            _TRACER_INITIALIZED = True

    return trace.get_tracer(service_name)


def _init_metrics_server(port: int = 9090):
    """
    Start Prometheus metrics HTTP server if not already started.
    prometheus_client.start_http_server serves /metrics on the given port.
    """
    global _METRICS_INITIALIZED
    with _METRICS_INIT_LOCK:
        if not _METRICS_INITIALIZED:
            start_http_server(port)
            _METRICS_INITIALIZED = True
            log.info("Prometheus metrics server started on :%d/metrics", port)


def init(service_name: str, *, metrics_port: Optional[int] = None) -> tuple[trace.Tracer, WorkerMetrics]:
    """
    Initialize tracing + metrics. Idempotent.

    Returns:
        (tracer, metrics) where:
          - tracer = OpenTelemetry tracer
          - metrics = WorkerMetrics helper (Prometheus)
    """
    tracer = _init_tracer(service_name)

    # Prometheus metrics server (enabled by default)
    port = int(metrics_port or os.getenv("METRICS_PORT", "9090"))
    if os.getenv("METRICS_DISABLED", "false").lower() not in ("1", "true", "yes"):
        _init_metrics_server(port)

    # Default registry is fine; start_http_server uses it automatically.
    metrics = WorkerMetrics(service_name=service_name)
    return tracer, metrics