# observability/metrics/bus_exporter.py
"""
Bus metrics exporter for Prometheus / OpenMetrics.

Features
--------
- Redis:
    * list length (LLEN) for queue-style keys
    * stream length (XLEN) for XADD streams
    * redis INFO stats (uptime, mem, ops) as gauges (subset)
- Kafka (optional, via confluent-kafka):
    * partition high/low watermarks (depth = high - low)
    * probe-based "approx backlog" using a throwaway consumer (NOT per-consumer-group lag)
- Prometheus exposition:
    * pull: starts an HTTP server on :<port>/metrics (requires prometheus_client)
    * push: optional Pushgateway
    * fallback: prints OpenMetrics to stdout at intervals if prometheus_client is absent

Usage
-----
from observability.metrics.bus_exporter import BusExporter, RedisConfig, KafkaConfig

exp = BusExporter(
    port=9108,                   # HTTP port for /metrics (if prometheus_client present)
    interval_s=10.0,             # scrape/collect interval
    redis=RedisConfig(url="redis://localhost:6379/0", lists=["orders", "fills"], streams=["md_ticks"]),
    kafka=KafkaConfig(bootstrap="localhost:9092", topics=["orders", "fills"]),
    pushgateway=None             # e.g., "http://localhost:9091" to push instead of serve
)
exp.run_blocking()
"""

from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Optional deps
try:
    from prometheus_client import (
        CollectorRegistry,
        Gauge,
        start_http_server,
        push_to_gateway,
        write_to_textfile,
        CONTENT_TYPE_LATEST,
        generate_latest,# type: ignore
    )  # type: ignore
except Exception:
    CollectorRegistry = None  # type: ignore
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore
    push_to_gateway = None  # type: ignore
    write_to_textfile = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # fallback
    def generate_latest(reg):  # type: ignore
        return b""  # we implement our own fallback printer

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

try:
    from confluent_kafka import Consumer, TopicPartition  # type: ignore
except Exception:
    Consumer = None  # type: ignore
    TopicPartition = None  # type: ignore


# ----------------------------- Config -----------------------------

@dataclass
class RedisConfig:
    url: str                                  # e.g., "redis://localhost:6379/0"
    lists: List[str] = field(default_factory=list)
    streams: List[str] = field(default_factory=list)
    info: bool = True                         # collect basic INFO stats


@dataclass
class KafkaConfig:
    bootstrap: str                            # "host:port"
    topics: List[str] = field(default_factory=list)
    security: Dict[str, str] = field(default_factory=dict)  # extra confluent config (sasl.*, ssl.*)
    # probe settings
    group_id: str = "bus-metrics-probe"
    session_timeout_ms: int = 6000


@dataclass
class BusExporterConfig:
    port: int = 9108                 # HTTP /metrics port (pull mode)
    interval_s: float = 10.0         # collect interval
    pushgateway: Optional[str] = None  # if set, push instead of serving HTTP
    push_job: str = "bus_exporter"
    textfile_path: Optional[str] = None  # optional node_exporter textfile collector path
    # namespaces / labels
    namespace: str = "newsintel"
    labels: Dict[str, str] = field(default_factory=dict)
    # backends
    redis: Optional[RedisConfig] = None
    kafka: Optional[KafkaConfig] = None


# ----------------------------- Exporter -----------------------------

class BusExporter:
    def __init__(self, port: int = 9108, interval_s: float = 10.0,
                 redis: Optional[RedisConfig] = None,
                 kafka: Optional[KafkaConfig] = None,
                 pushgateway: Optional[str] = None,
                 push_job: str = "bus_exporter",
                 textfile_path: Optional[str] = None,
                 namespace: str = "newsintel",
                 labels: Optional[Dict[str, str]] = None):
        self.cfg = BusExporterConfig(
            port=port, interval_s=interval_s, pushgateway=pushgateway, push_job=push_job,
            textfile_path=textfile_path, namespace=namespace, labels=labels or {},
            redis=redis, kafka=kafka
        )
        self._stop = False

        # Prometheus registry & metrics
        self._have_prom = CollectorRegistry is not None
        if self._have_prom:
            self.reg = CollectorRegistry()# type: ignore
            ns = self.cfg.namespace

            # Redis gauges
            self.g_redis_list_len = Gauge(f"{ns}_redis_list_len", "Redis list length",
                                          ["key", *self.cfg.labels.keys()], registry=self.reg) # type: ignore
            self.g_redis_stream_len = Gauge(f"{ns}_redis_stream_len", "Redis stream length",# type: ignore
                                            ["key", *self.cfg.labels.keys()], registry=self.reg)
            self.g_redis_info_uptime = Gauge(f"{ns}_redis_uptime_seconds", "Redis uptime seconds",# type: ignore
                                             list(self.cfg.labels.keys()), registry=self.reg)
            self.g_redis_info_mem = Gauge(f"{ns}_redis_used_memory_bytes", "Redis used memory bytes",# type: ignore
                                          list(self.cfg.labels.keys()), registry=self.reg)
            self.g_redis_info_ops = Gauge(f"{ns}_redis_ops_per_sec", "Redis instantaneous ops per sec",# type: ignore
                                          list(self.cfg.labels.keys()), registry=self.reg)

            # Kafka gauges
            self.g_kafka_depth = Gauge(f"{ns}_kafka_topic_depth", "Kafka topic depth (approx, high-low)",# type: ignore
                                       ["topic", "partition", *self.cfg.labels.keys()], registry=self.reg)
            self.g_kafka_high = Gauge(f"{ns}_kafka_high_watermark", "Kafka high watermark",# type: ignore
                                      ["topic", "partition", *self.cfg.labels.keys()], registry=self.reg)
            self.g_kafka_low = Gauge(f"{ns}_kafka_low_watermark", "Kafka low watermark",# type: ignore
                                     ["topic", "partition", *self.cfg.labels.keys()], registry=self.reg)
        else:
            self.reg = None  # stdout fallback will render minimal metrics itself

        # Connections (lazy)
        self._r = None
        self._k = None

    # ------------------- lifecycle -------------------

    def run_blocking(self) -> None:
        """
        Starts HTTP server (if prometheus_client present) or uses stdout mode, then collects forever.
        """
        if self._have_prom and not self.cfg.pushgateway and not self.cfg.textfile_path:
            # pull model HTTP server
            if start_http_server is None:
                print("[bus_exporter] prometheus_client missing; falling back to stdout", file=sys.stderr)
            else:
                start_http_server(self.cfg.port, registry=self.reg)  # type: ignore
                print(f"[bus_exporter] serving /metrics on :{self.cfg.port}", file=sys.stderr)

        try:
            while not self._stop:
                self.collect_once()
                # push modes
                if self._have_prom and self.cfg.pushgateway:
                    try:
                        push_to_gateway(self.cfg.pushgateway, job=self.cfg.push_job, registry=self.reg)  # type: ignore
                    except Exception as e:
                        print(f"[bus_exporter] pushgateway error: {e}", file=sys.stderr)
                if self._have_prom and self.cfg.textfile_path:
                    try:
                        write_to_textfile(self.cfg.textfile_path, self.reg)  # type: ignore
                    except Exception as e:
                        print(f"[bus_exporter] textfile write error: {e}", file=sys.stderr)

                if not self._have_prom:
                    # stdout OpenMetrics-ish fallback (very simple)
                    self._print_stdout_snapshot()
                time.sleep(max(1.0, self.cfg.interval_s))
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        self._stop = True

    # ------------------- collection -------------------

    def collect_once(self) -> None:
        labels = self.cfg.labels or {}

        # Redis
        if self.cfg.redis:
            if redis is None:
                print("[bus_exporter] redis package not installed; skipping Redis metrics", file=sys.stderr)
            else:
                self._collect_redis(self.cfg.redis, labels)

        # Kafka
        if self.cfg.kafka:
            if Consumer is None:
                print("[bus_exporter] confluent-kafka not installed; skipping Kafka metrics", file=sys.stderr)
            else:
                self._collect_kafka(self.cfg.kafka, labels)

    # ---- Redis ----

    def _redis_client(self, rcfg: RedisConfig):
        if self._r is None:
            self._r = redis.from_url(rcfg.url)  # type: ignore
        return self._r

    def _collect_redis(self, rcfg: RedisConfig, labels: Dict[str, str]) -> None:
        r = self._redis_client(rcfg)
        # lists
        for key in rcfg.lists:
            try:
                ln = r.llen(key)  # type: ignore
                if self._have_prom:
                    self.g_redis_list_len.labels(key=key, **labels).set(float(ln))# type: ignore
            except Exception as e:
                print(f"[bus_exporter] redis LLEN {key} error: {e}", file=sys.stderr)
        # streams
        for key in rcfg.streams:
            try:
                ln = r.xlen(key)  # type: ignore
                if self._have_prom:
                    self.g_redis_stream_len.labels(key=key, **labels).set(float(ln))# type: ignore
            except Exception as e:
                print(f"[bus_exporter] redis XLEN {key} error: {e}", file=sys.stderr)
        # info subset
        if rcfg.info:
            try:
                info = r.info()  # type: ignore
                if self._have_prom:
                    self.g_redis_info_uptime.labels(**labels).set(float(info.get("uptime_in_seconds", 0)))# type: ignore
                    self.g_redis_info_mem.labels(**labels).set(float(info.get("used_memory", 0)))# type: ignore
                    self.g_redis_info_ops.labels(**labels).set(float(info.get("instantaneous_ops_per_sec", 0)))# type: ignore
            except Exception as e:
                print(f"[bus_exporter] redis INFO error: {e}", file=sys.stderr)

    # ---- Kafka ----

    def _kafka_consumer(self, kcfg: KafkaConfig):
        if self._k is None:
            conf = {
                "bootstrap.servers": kcfg.bootstrap,
                "group.id": kcfg.group_id,
                "enable.auto.commit": False,
                "session.timeout.ms": kcfg.session_timeout_ms,
                "auto.offset.reset": "latest",
            }
            conf.update(kcfg.security or {})
            self._k = Consumer(conf)  # type: ignore
        return self._k

    def _collect_kafka(self, kcfg: KafkaConfig, labels: Dict[str, str]) -> None:
        c = self._kafka_consumer(kcfg)
        try:
            # assign partitions per topic (metadata fetch)
            md = c.list_topics(timeout=5.0)  # type: ignore
            for topic in kcfg.topics:
                tmeta = md.topics.get(topic)
                if tmeta is None or tmeta.error is not None:
                    print(f"[bus_exporter] kafka topic metadata error: {getattr(tmeta, 'error', None)}", file=sys.stderr)
                    continue
                for pnum, pmeta in tmeta.partitions.items():
                    tp = TopicPartition(topic, pnum)  # type: ignore
                    low, high = c.get_watermark_offsets(tp, timeout=5.0)  # type: ignore
                    depth = max(0, high - low)
                    part_lbls = {"topic": topic, "partition": str(pnum), **labels}
                    if self._have_prom:
                        self.g_kafka_low.labels(**part_lbls).set(float(low))
                        self.g_kafka_high.labels(**part_lbls).set(float(high))
                        self.g_kafka_depth.labels(**part_lbls).set(float(depth))
        except Exception as e:
            print(f"[bus_exporter] kafka collect error: {e}", file=sys.stderr)

    # ------------------- fallback output -------------------

    def _print_stdout_snapshot(self) -> None:
        """Minimal OpenMetrics-ish dump for environments without prometheus_client."""
        ts = int(time.time())
        def line(name: str, value: float, labels: Dict[str, str]):
            if labels:
                lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
                print(f"{name}{{{lab}}} {value} {ts}")
            else:
                print(f"{name} {value} {ts}")

        ns = self.cfg.namespace
        # we can't access registry values here; we recompute quickly to print (best effort)
        # Re-run light probes (lists/streams only, not heavy Kafka metadata) to avoid extra logic.
        if self.cfg.redis and redis is not None:
            r = self._redis_client(self.cfg.redis)
            for key in self.cfg.redis.lists:
                try:
                    ln = r.llen(key)  # type: ignore
                    line(f"{ns}_redis_list_len", float(ln), {"key": key, **self.cfg.labels})# type: ignore
                except Exception:
                    pass
            for key in self.cfg.redis.streams:
                try:
                    ln = r.xlen(key)  # type: ignore
                    line(f"{ns}_redis_stream_len", float(ln), {"key": key, **self.cfg.labels})# type: ignore
                except Exception:
                    pass


# ----------------------------- CLI -----------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default

def _boolenv(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")

def main() -> int:
    # Env-driven quick start (handy for containers)
    port = int(_env("BUS_EXPORTER_PORT", "9108"))# type: ignore
    interval = float(_env("BUS_EXPORTER_INTERVAL_S", "10.0"))# type: ignore
    ns = _env("BUS_EXPORTER_NAMESPACE", "newsintel") or "newsintel"

    # Labels: BUS_EXPORTER_LABELS="env=dev,service=news-intel"
    raw_labels = _env("BUS_EXPORTER_LABELS", "")
    labels = {}
    if raw_labels:
        for kv in raw_labels.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                labels[k.strip()] = v.strip()

    # Redis
    rurl = _env("BUS_EXPORTER_REDIS_URL")
    rlists = (_env("BUS_EXPORTER_REDIS_LISTS", "") or "").split(",") if _env("BUS_EXPORTER_REDIS_LISTS") else []
    rstreams = (_env("BUS_EXPORTER_REDIS_STREAMS", "") or "").split(",") if _env("BUS_EXPORTER_REDIS_STREAMS") else []
    rcfg = RedisConfig(url=rurl, lists=[x for x in rlists if x], streams=[x for x in rstreams if x]) if rurl else None

    # Kafka
    kboot = _env("BUS_EXPORTER_KAFKA_BOOTSTRAP")
    ktopics = (_env("BUS_EXPORTER_KAFKA_TOPICS", "") or "").split(",") if _env("BUS_EXPORTER_KAFKA_TOPICS") else []
    kcfg = KafkaConfig(bootstrap=kboot, topics=[x for x in ktopics if x]) if kboot else None

    # Push modes
    push = _env("BUS_EXPORTER_PUSHGATEWAY")
    textfile = _env("BUS_EXPORTER_TEXTFILE")

    exp = BusExporter(
        port=port,
        interval_s=interval,
        redis=rcfg,
        kafka=kcfg,
        pushgateway=push,
        textfile_path=textfile,
        namespace=ns,
        labels=labels,
    )
    exp.run_blocking()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())