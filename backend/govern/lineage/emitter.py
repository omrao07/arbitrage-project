# governance/lineage/emitter.py
"""
Lightweight lineage emitter (OpenLineage-style).

Usage
-----
from governance.lineage.emitter import LineageEmitter, Dataset, Job

em = LineageEmitter(
    sink="stdout",                             # "stdout" | "file" | "http" | "kafka"
    file_path="logs/lineage.jsonl",           # for sink="file"
    http_url="https://marquez/api/v1/lineage",# for sink="http"
    kafka_bootstrap="localhost:9092",         # for sink="kafka"
    kafka_topic="lineage.events",
    facets_path="governance/lineage/facets.yaml"
)

run_id = em.new_run_id()
em.emit_start(
    job=Job(namespace="news-intel", name="pipeline_runner"),
    run_id=run_id,
    inputs=[Dataset("rss", "bbc_world"), Dataset("rss", "reuters_markets")],
)
# ... pipeline runs ...
em.emit_complete(
    job=Job(namespace="news-intel", name="pipeline_runner"),
    run_id=run_id,
    outputs=[Dataset("sqlite", "news.sqlite#articles")],
    stats={"rows": 1234}
)
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

# Optional deps (Kafka)
try:
    from confluent_kafka import Producer  # type: ignore
except Exception:
    Producer = None  # type: ignore

# Optional YAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# --- small models ----------------------------------------------------------------

@dataclass
class Dataset:
    namespace: str
    name: str
    facets: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Job:
    namespace: str
    name: str
    facets: Dict[str, Any] = field(default_factory=dict)

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-host"

# --- Emitter ---------------------------------------------------------------------

class LineageEmitter:
    """
    Minimal OpenLineage-like emitter with multiple sinks (stdout/file/http/kafka).
    `facets.yaml` (optional) lets you enrich jobs/datasets with static facets.
    """

    def __init__(
        self,
        *,
        sink: str = "stdout",
        file_path: Optional[str] = None,
        http_url: Optional[str] = None,
        http_headers: Optional[Dict[str, str]] = None,
        kafka_bootstrap: Optional[str] = None,
        kafka_topic: Optional[str] = None,
        kafka_config: Optional[Dict[str, Any]] = None,
        facets_path: Optional[str] = None,
        project: str = "news-intel",
        emitter_name: str = "lineage-emitter",
    ):
        self.sink = sink.lower()
        self.file_path = file_path
        self.http_url = http_url
        self.http_headers = http_headers or {}
        self.kafka_bootstrap = kafka_bootstrap
        self.kafka_topic = kafka_topic
        self.kafka_config = kafka_config or {}
        self.project = project
        self.emitter_name = emitter_name
        self.facets_cfg = self._load_facets(facets_path) if facets_path else {}

        # Prepare sinks
        if self.sink == "file":
            if not self.file_path:
                raise ValueError("file sink requires file_path")
            os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)

        if self.sink == "kafka":
            if Producer is None:
                raise RuntimeError("Kafka sink requires confluent-kafka")
            if not (self.kafka_bootstrap and self.kafka_topic):
                raise ValueError("Kafka sink requires kafka_bootstrap and kafka_topic")
            cfg = {"bootstrap.servers": self.kafka_bootstrap}
            cfg.update(self.kafka_config)
            self._kprod = Producer(cfg)  # type: ignore[attr-defined]

    # ------------- public helpers -------------

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())

    def emit_start(self, *, job: Job, run_id: str, inputs: Optional[List[Dataset]] = None) -> None:
        evt = self._mk_event("START", job, run_id, inputs=inputs or [], outputs=[])
        self._emit(evt)

    def emit_complete(
        self, *, job: Job, run_id: str, outputs: Optional[List[Dataset]] = None, stats: Optional[Dict[str, Any]] = None
    ) -> None:
        evt = self._mk_event("COMPLETE", job, run_id, inputs=[], outputs=outputs or [], stats=stats or {})
        self._emit(evt)

    def emit_fail(self, *, job: Job, run_id: str, error: str, inputs: Optional[List[Dataset]] = None) -> None:
        evt = self._mk_event("FAIL", job, run_id, inputs=inputs or [], outputs=[], error=error)
        self._emit(evt)

    # ------------- internals -------------

    def _mk_event(
        self,
        event_type: str,
        job: Job,
        run_id: str,
        *,
        inputs: List[Dataset],
        outputs: List[Dataset],
        stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Attach facets from config
        job_f = self._facets_for("job", f"{job.namespace}.{job.name}")
        ds_in = [self._dataset_with_facets(d) for d in inputs]
        ds_out = [self._dataset_with_facets(d) for d in outputs]

        evt = {
            "eventType": event_type,
            "eventTime": _now_iso(),
            "producer": f"{self.project}/{self.emitter_name}",
            "host": _hostname(),
            "job": {
                "namespace": job.namespace,
                "name": job.name,
                "facets": {**(job.facets or {}), **job_f},
            },
            "run": {
                "runId": run_id,
                "facets": {
                    "runDetails": {"_producer": self.emitter_name, "ts": _now_iso()},
                },
            },
            "inputs": [asdict(d) for d in ds_in],
            "outputs": [asdict(d) for d in ds_out],
        }
        if stats:
            evt["run"]["facets"]["stats"] = stats
        if error:
            evt["run"]["facets"]["errorMessage"] = {"message": error}
        return evt

    def _dataset_with_facets(self, d: Dataset) -> Dataset:
        key = f"{d.namespace}.{d.name}"
        extra = self._facets_for("dataset", key)
        merged = dict(d.facets or {})
        merged.update(extra)
        return Dataset(d.namespace, d.name, merged)

    def _facets_for(self, kind: str, key: str) -> Dict[str, Any]:
        # facets.yaml structure:
        # job:
        #   news-intel.pipeline_runner:
        #     owner: data-eng
        # dataset:
        #   rss.bbc_world:
        #     retention_days: 7
        cfg = self.facets_cfg.get(kind, {})
        return cfg.get(key, {})

    # ------------- sinks -------------

    def _emit(self, event: Dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False)

        if self.sink == "stdout":
            print(payload)
            return

        if self.sink == "file":
            assert self.file_path
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(payload + "\n")
            return

        if self.sink == "http":
            if not self.http_url:
                raise ValueError("http sink requires http_url")
            self._post_http(self.http_url, payload, headers=self.http_headers)
            return

        if self.sink == "kafka":
            topic = self.kafka_topic or ""
            self._kprod.produce(topic, value=payload.encode("utf-8"))  # type: ignore[attr-defined]
            # give the producer a chance to flush in background
            self._kprod.poll(0)  # type: ignore[attr-defined]
            return

        raise ValueError(f"unknown sink: {self.sink}")

    @staticmethod
    def _post_http(url: str, body: str, headers: Optional[Dict[str, str]] = None, timeout: float = 8.0) -> None:
        # stdlib HTTP POST to avoid extra dependency
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url=url,
            data=body.encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            if getattr(resp, "status", 200) >= 300:
                raise RuntimeError(f"HTTP sink status {resp.status}")

    @staticmethod
    def _load_facets(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        if yaml is None:
            # very small YAML subset: allow JSON as a fallback
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ------------- cleanup -------------

    def close(self) -> None:
        if self.sink == "kafka" and hasattr(self, "_kprod"):
            try:
                self._kprod.flush(5.0)  # type: ignore[attr-defined]
            except Exception:
                pass