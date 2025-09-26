# observability/logs/exporter.py
"""
Unified log exporter (dependency-light).

Backends:
  - stdout: print JSONL to stdout
  - file: append to a local file, simple size-based rotation
  - http: POST newline-delimited JSON to an endpoint
  - loki: push logs to Grafana Loki
  - elasticsearch: bulk ingest to ES _bulk endpoint
  - datadog: http intake (logs)

Usage:
  exp = LogExporter(kind="loki",
                    endpoint="http://localhost:3100/loki/api/v1/push",
                    default_labels={"app":"news-intel","env":"dev"})
  exp.write({"ts": "...", "level": "INFO", "msg": "hello"})
  exp.flush(); exp.close()

All writes are buffered and sent in batches with retries/backoff.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass, field
import threading
import urllib.request
import urllib.error
import urllib.parse

# -------------------- helpers --------------------

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _sleep_backoff(attempt: int, base: float = 0.25) -> None:
    d = base * (2 ** attempt)
    time.sleep(d + random.uniform(0, d * 0.25))

# -------------------- config --------------------

@dataclass
class ExporterConfig:
    kind: str = "stdout"                 # stdout | file | http | loki | elasticsearch | datadog
    endpoint: Optional[str] = None       # URL for http/loki/elasticsearch/datadog
    headers: Dict[str, str] = field(default_factory=dict)

    # buffering
    batch_size: int = 200
    flush_interval_s: float = 2.0
    gzip_http: bool = False

    # file backend
    path: Optional[str] = None
    rotate_mb: int = 50
    rotate_keep: int = 5

    # loki
    default_labels: Dict[str, str] = field(default_factory=dict)  # {"app":"...", "env":"..."}
    loki_label_keys: List[str] = field(default_factory=lambda: ["level", "logger"])  # promoted to labels if present
    loki_stream_label: str = "app"

    # elasticsearch
    index: Optional[str] = None          # e.g., "logs-news-%Y.%m.%d"
    es_pipeline: Optional[str] = None    # optional ingest pipeline

    # datadog
    dd_api_key: Optional[str] = None
    dd_service: Optional[str] = None
    dd_source: Optional[str] = None
    dd_host: Optional[str] = None

    # retries
    max_retries: int = 5
    timeout_s: float = 8.0

# -------------------- exporter --------------------

class LogExporter:
    def __init__(self, **kwargs):
        self.cfg = ExporterConfig(**kwargs)
        self._buf: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = False
        self._thread: Optional[threading.Thread] = None

        if self.cfg.kind == "file":
            if not self.cfg.path:
                raise ValueError("file backend requires path")
            os.makedirs(os.path.dirname(self.cfg.path) or ".", exist_ok=True)

        if self.cfg.kind in ("http", "loki", "elasticsearch", "datadog"):
            if not self.cfg.endpoint:
                # Datadog default if only API key present
                if self.cfg.kind == "datadog":
                    self.cfg.endpoint = "https://http-intake.logs.datadoghq.com/api/v2/logs"
                else:
                    raise ValueError(f"{self.cfg.kind} backend requires endpoint URL")

        # Add defaults for Datadog headers
        if self.cfg.kind == "datadog":
            if self.cfg.dd_api_key:
                self.cfg.headers.setdefault("DD-API-KEY", self.cfg.dd_api_key)
            self.cfg.headers.setdefault("Content-Type", "application/json")

        # start flush thread for timed flushes
        if self.cfg.flush_interval_s > 0:
            self._thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._thread.start()

    # ------------- public -------------

    def write(self, record: Dict[str, Any]) -> None:
        """Accept one structured log record (dict)."""
        with self._lock:
            self._buf.append(record)
            if len(self._buf) >= self.cfg.batch_size:
                batch = self._drain_locked()
        if 'batch' in locals():
            self._ship(batch)

    def write_many(self, records: Iterable[Dict[str, Any]]) -> None:
        with self._lock:
            for r in records:
                self._buf.append(r)
                if len(self._buf) >= self.cfg.batch_size:
                    batch = self._drain_locked()
                    self._ship(batch)
            if len(self._buf) >= self.cfg.batch_size:
                batch = self._drain_locked()
        if 'batch' in locals():
            self._ship(batch)

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            batch = self._drain_locked()
        self._ship(batch)

    def close(self) -> None:
        self._stop = True
        try:
            if self._thread:
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self.flush()

    # ------------- internals -------------

    def _drain_locked(self) -> List[Dict[str, Any]]:
        batch = self._buf
        self._buf = []
        return batch

    def _flush_loop(self) -> None:
        while not self._stop:
            time.sleep(self.cfg.flush_interval_s)
            try:
                self.flush()
            except Exception:
                # never crash on background flush
                pass

    def _ship(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        kind = self.cfg.kind
        if kind == "stdout":
            for r in batch:
                print(_json(r))
            return
        if kind == "file":
            self._ship_file(batch)
            return
        if kind == "http":
            self._ship_http_jsonl(batch)
            return
        if kind == "loki":
            self._ship_loki(batch)
            return
        if kind == "elasticsearch":
            self._ship_es_bulk(batch)
            return
        if kind == "datadog":
            self._ship_datadog(batch)
            return
        raise ValueError(f"unknown backend kind: {kind}")

    # -------- file --------

    def _ship_file(self, batch: List[Dict[str, Any]]) -> None:
        path = self.cfg.path or "logs/app.jsonl"
        # rotate if size exceeds rotate_mb
        try:
            if os.path.exists(path) and os.path.getsize(path) > self.cfg.rotate_mb * 1024 * 1024:
                self._rotate_file(path, self.cfg.rotate_keep)
        except Exception:
            pass
        with open(path, "a", encoding="utf-8") as f:
            for r in batch:
                f.write(_json(r) + "\n")

    @staticmethod
    def _rotate_file(path: str, keep: int) -> None:
        try:
            for i in range(keep - 1, 0, -1):
                src = f"{path}.{i}"
                dst = f"{path}.{i+1}"
                if os.path.exists(src):
                    os.replace(src, dst)
            os.replace(path, f"{path}.1")
        except Exception:
            # rotation best-effort
            pass

    # -------- http generic (JSONL) --------

    def _ship_http_jsonl(self, batch: List[Dict[str, Any]]) -> None:
        body = "\n".join(_json(r) for r in batch) + "\n"
        self._post(self.cfg.endpoint or "", body.encode("utf-8"),
                   headers={"Content-Type": "application/json", **self.cfg.headers})

    # -------- Loki --------

    def _ship_loki(self, batch: List[Dict[str, Any]]) -> None:
        # Loki expects: { "streams": [ { "stream": {label_k:v}, "values": [ [ "<ns>", "<line>" ], ... ] } ] }
        # We group by labels to cut payload size.
        groups: Dict[Tuple[Tuple[str, str], ...], List[Tuple[int, str]]] = {}
        base_labels = dict(self.cfg.default_labels or {})
        for r in batch:
            lbl = dict(base_labels)
            # promote configured keys to labels if present
            for k in self.cfg.loki_label_keys:
                if k in r:
                    lbl[k] = str(r[k])
            # ensure stream label exists (e.g., app)
            lbl.setdefault(self.cfg.loki_stream_label, base_labels.get(self.cfg.loki_stream_label, "app"))
            key = tuple(sorted(lbl.items()))
            # timestamp: nanoseconds string; prefer r["ts"], else now
            ts = r.get("ts") or _now_iso()
            ns = _iso_to_ns(ts)
            line = _json(r)
            groups.setdefault(key, []).append((ns, line))

        streams = []
        for key, pairs in groups.items():
            stream = {k: v for k, v in key}
            values = [[str(ns), line] for ns, line in pairs]
            streams.append({"stream": stream, "values": values})

        payload = _json({"streams": streams}).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.cfg.headers}
        self._post(self.cfg.endpoint or "", payload, headers=headers)

    # -------- Elasticsearch Bulk --------

    def _ship_es_bulk(self, batch: List[Dict[str, Any]]) -> None:
        index = time.strftime(self.cfg.index or "logs-%Y.%m.%d")
        lines: List[str] = []
        meta_base: Dict[str, Any] = {"index": {"_index": index}}
        if self.cfg.es_pipeline:
            meta_base["index"]["pipeline"] = self.cfg.es_pipeline
        for r in batch:
            lines.append(_json(meta_base))
            lines.append(_json(r))
        data = ("\n".join(lines) + "\n").encode("utf-8")
        headers = {"Content-Type": "application/x-ndjson", **self.cfg.headers}
        url = self.cfg.endpoint.rstrip("/") + "/_bulk" # type: ignore
        self._post(url, data, headers=headers)

    # -------- Datadog --------

    def _ship_datadog(self, batch: List[Dict[str, Any]]) -> None:
        # Datadog expects an array of log records; we enrich with ddsource/service/host if provided.
        arr = []
        for r in batch:
            dd = dict(r)
            if self.cfg.dd_source:
                dd.setdefault("ddsource", self.cfg.dd_source)
            if self.cfg.dd_service:
                dd.setdefault("service", self.cfg.dd_service)
            if self.cfg.dd_host:
                dd.setdefault("host", self.cfg.dd_host)
            arr.append(dd)
        data = _json(arr).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.cfg.headers}
        self._post(self.cfg.endpoint or "", data, headers=headers)

    # -------- HTTP POST with retries --------

    def _post(self, url: str, body: bytes, *, headers: Dict[str, str]) -> None:
        # Optionally gzip payload
        if self.cfg.gzip_http:
            bio = io.BytesIO()
            with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
                gz.write(body)
            body = bio.getvalue()
            headers = dict(headers)
            headers["Content-Encoding"] = "gzip"

        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                req = urllib.request.Request(url=url, data=body, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:  # nosec B310
                    status = getattr(resp, "status", 200)
                    if status >= 300:
                        raise urllib.error.HTTPError(url, status, f"status={status}", hdrs=resp.headers, fp=None)
                    return
            except Exception as e:
                last_exc = e
                if attempt >= self.cfg.max_retries:
                    # last resort: write to stderr (never raise)
                    sys.stderr.write(f"[logs/exporter] POST failed ({type(e).__name__}): {e}\n")
                    return
                _sleep_backoff(attempt)

# -------------------- utilities --------------------

def _iso_to_ns(iso_or_ns: Any) -> int:
    """Return UNIX time in nanoseconds for Loki."""
    if isinstance(iso_or_ns, (int, float)) and iso_or_ns > 1e12:
        # looks like nanoseconds already
        return int(iso_or_ns)
    # parse iso8601
    try:
        import datetime as dt
        s = str(iso_or_ns)
        if s.isdigit():
            # seconds
            return int(float(s) * 1e9)
        # strip Z if present
        if s.endswith("Z"):
            s = s[:-1]
        # handle fractional seconds optionally
        fmt = "%Y-%m-%dT%H:%M:%S"
        if "." in s:
            fmt = "%Y-%m-%dT%H:%M:%S.%f"
        ts = dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc).timestamp()
        return int(ts * 1e9)
    except Exception:
        return int(time.time() * 1e9)

# -------------------- self-test --------------------

if __name__ == "__main__":
    # quick demo to stdout
    exp = LogExporter(kind="stdout", batch_size=3, flush_interval_s=1.0)
    for i in range(5):
        exp.write({"ts": _now_iso(), "level": "INFO", "logger": "demo", "msg": f"hello {i}"})
    time.sleep(2.5)
    exp.close()