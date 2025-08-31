
"""
backend/workers/analyst_worker.py

Analyst Worker:
- Loads config from analyst_worker.yaml (path via $ANALYST_CONFIG or defaults alongside file)
- Connects to Redis
- Consumes from configured inbound streams using consumer groups
- Routes messages to task handlers based on "topic" and "stream"
- Handles simple retries and per-task rate limits
- Publishes outputs to outbound streams and UI channel
- Exposes Prometheus metrics if prometheus_client is available

NOTE: Task handlers referenced in YAML ("module", "entry") must be importable.
      Example stubs you can create: research/gnn_correlations_runner.py, research/risk/var_es_runner.py, research/exec/almgren_runner.py, research/anomaly/scan_runner.py
"""

from __future__ import annotations
import os
import sys
import time
import json
import yaml
import signal
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

try:
    import redis  # redis-py
except Exception as e:
    raise RuntimeError("redis-py is required. pip install redis") from e

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server # type: ignore
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False

# ----------------------------- Logging (JSON) -----------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "level": record.levelname,
            "time": int(record.created * 1000),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)

def setup_logging(level: str = "INFO", to_stdout: bool = True, file_path: Optional[str] = None):
    logger = logging.getLogger("analyst")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = JsonFormatter()
    if to_stdout:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# ----------------------------- Config types ------------------------------

@dataclass
class RedisConfig:
    host: str
    port: int
    db: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    consumer_group: str = "analyst_v1"
    consumer_name: str = "analyst-1"

@dataclass
class StreamDef:
    name: str
    start: str = "$"
    batch_max: int = 128
    block_ms: int = 5000

@dataclass
class InOutStreams:
    inbound: Dict[str, StreamDef]
    outbound: Dict[str, StreamDef]

@dataclass
class TaskDef:
    module: str
    entry: str
    timeout_sec: int = 120
    rate_limit_per_min: int = 60
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Route:
    match: Dict[str, Any]  # {stream: "prices", topic: "snapshot", schema?: "..."}
    task: str

@dataclass
class WorkerConfig:
    name: str
    role: str
    runtime: str
    entrypoint: str
    args: List[str]
    resources: Dict[str, Any]
    concurrency: Dict[str, Any]
    retry_policy: Dict[str, Any]

@dataclass
class Config:
    version: int
    worker: WorkerConfig
    io: Dict[str, Any]
    streams: Dict[str, Any]
    pubsub: Dict[str, Any]
    storage: Dict[str, Any]
    routing: Dict[str, Any]
    tasks: Dict[str, TaskDef]
    scheduling: Dict[str, Any]
    logging: Dict[str, Any]
    health: Dict[str, Any]
    metrics: Dict[str, Any]
    security: Dict[str, Any]

# ----------------------------- Utils -------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        text = f.read()
    # env interpolation like ${VAR:-default}
    def _sub_env(value: str) -> str:
        import re
        def repl(m):
            expr = m.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.environ.get(var, default)
            return os.environ.get(expr, "")
        return re.sub(r"\$\{([^}]+)\}", repl, value)

    text = _sub_env(text)
    return yaml.safe_load(text)

def import_callable(module_path: str, func_name: str):
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)
    return fn

def ensure_consumer_group(r: "redis.Redis", stream: str, group: str, logger: logging.Logger):
    try:
        r.xgroup_create(name=stream, groupname=group, id="0-0", mkstream=True)
        logger.info(json.dumps({"event": "xgroup_create", "stream": stream, "group": group}))
    except redis.exceptions.ResponseError as e: # type: ignore
        # Group might exist
        if "BUSYGROUP" in str(e):
            pass
        else:
            raise

def xack_safe(r: "redis.Redis", stream: str, group: str, msg_id: str, logger: logging.Logger):
    try:
        r.xack(stream, group, msg_id)
    except Exception as e:
        logger.error(json.dumps({"event": "xack_error", "stream": stream, "id": msg_id, "error": str(e)}))

def publish_stream(r: "redis.Redis", stream: str, payload: Dict[str, Any]) -> str:
    # Redis Streams fields must be flat string pairs
    flat = {k: json.dumps(v) if not isinstance(v, (str, bytes)) else v for k, v in payload.items()}
    return r.xadd(stream, flat) # type: ignore

def publish_pubsub(r: "redis.Redis", channel: str, payload: Dict[str, Any]) -> int:
    return r.publish(channel, json.dumps(payload)) # type: ignore

# ----------------------------- Metrics -----------------------------------

if PROM_AVAILABLE:
    METRICS = {
        "tasks_total": Counter("analyst_tasks_total", "Total tasks executed", ["task"]),
        "task_failures_total": Counter("analyst_task_failures_total", "Total task failures", ["task"]),
        "msg_consumed": Counter("analyst_messages_consumed_total", "Messages consumed", ["stream"]),
        "msg_published": Counter("analyst_messages_published_total", "Messages published", ["stream"]),
        "task_latency": Histogram("analyst_task_latency_seconds", "Task runtime", ["task"]),
        "backlog": Gauge("analyst_stream_backlog", "Pending entries in stream", ["stream"]),
    }
else:
    class _Noop:
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): pass
        def observe(self, *a, **k): pass
        def set(self, *a, **k): pass
    METRICS = {k: _Noop() for k in ["tasks_total","task_failures_total","msg_consumed","msg_published","task_latency","backlog"]}

# ----------------------------- Rate limit -------------------------------

class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_min = max(1, per_minute)
        self.tokens = self.per_min
        self.last = time.time()

    def allow(self) -> bool:
        now = time.time()
        # refill
        elapsed = now - self.last
        refill = elapsed * (self.per_min / 60.0)
        if refill > 0:
            self.tokens = min(self.per_min, self.tokens + refill)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

# ----------------------------- Worker -----------------------------------

class AnalystWorker:
    def __init__(self, cfg_path: Optional[str] = None):
        cfg_path = cfg_path or os.environ.get("ANALYST_CONFIG") or "analyst_worker.yaml"
        self.cfg_raw = load_yaml_config(cfg_path)

        # Logging
        log_cfg = self.cfg_raw.get("logging", {})
        self.logger = setup_logging(
            level=log_cfg.get("level", "INFO"),
            to_stdout=log_cfg.get("stdout", True),
            file_path=(log_cfg.get("file", {}) or {}).get("path") if (log_cfg.get("file", {}) or {}).get("enabled") else None
        )
        self.logger.info(json.dumps({"event": "boot", "config_path": cfg_path}))

        # Redis
        r_cfg = self.cfg_raw["io"]["redis"]
        self.redis = redis.Redis(
            host=r_cfg["host"],
            port=int(r_cfg["port"]),
            db=int(r_cfg.get("db", 0)),
            username=(r_cfg.get("username") or None) or None,
            password=(r_cfg.get("password") or None) or None,
            decode_responses=True,
        )
        self.group = r_cfg.get("consumer_group", "analyst_v1")
        self.consumer = r_cfg.get("consumer_name", "analyst-1")

        # Streams
        self.inbound = self.cfg_raw["streams"]["inbound"]
        self.outbound = self.cfg_raw["streams"]["outbound"]
        for st_key, st_def in self.inbound.items():
            ensure_consumer_group(self.redis, st_def["name"], self.group, self.logger)

        # Pubsub channel
        self.ui_channel = (self.cfg_raw.get("pubsub", {}).get("channels", {}) or {}).get("ui_bus")

        # Routes -> task map
        self.routes: List[Route] = [Route(**r) for r in self.cfg_raw["routing"]["routes"]]
        self.tasks: Dict[str, TaskDef] = {k: TaskDef(**v) for k, v in self.cfg_raw["tasks"].items()}
        self.rate_limiters: Dict[str, RateLimiter] = {name: RateLimiter(td.rate_limit_per_min) for name, td in self.tasks.items()}

        # Retry policy
        rp = self.cfg_raw["worker"].get("retry_policy", {})
        self.max_retries = int(rp.get("max_retries", 3))
        self.backoff_seconds = int(rp.get("backoff_seconds", 10))

        # Metrics server
        met_cfg = self.cfg_raw.get("metrics", {}).get("prometheus", {})
        if PROM_AVAILABLE and met_cfg.get("enabled"):
            start_http_server(int(met_cfg.get("port", 9093)), addr=met_cfg.get("bind", "0.0.0.0"))
            self.logger.info(json.dumps({"event": "metrics_started", "port": met_cfg.get("port", 9093)}))

        self._shutdown = False
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ---------------------- Core loop ----------------------

    def _on_signal(self, *args):
        self.logger.info(json.dumps({"event": "shutdown_signal"}))
        self._shutdown = True

    def _match_task(self, stream_key: str, topic: Optional[str]) -> Optional[str]:
        for r in self.routes:
            ok_stream = (r.match.get("stream") == stream_key)
            ok_topic = (("topic" not in r.match) or (r.match.get("topic") == topic))
            if ok_stream and ok_topic:
                return r.task
        return None

    def _call_task(self, task_name: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        td = self.tasks[task_name]
        if not self.rate_limiters[task_name].allow():
            raise RuntimeError(f"rate_limited:{task_name}")
        fn = import_callable(td.module, td.entry)
        start = time.time()
        try:
            result = fn(payload=payload, params=td.params, ctx={"redis": self.redis})
            elapsed = time.time() - start
            METRICS["task_latency"].labels(task_name).observe(elapsed)
            METRICS["tasks_total"].labels(task_name).inc()
            # Normalize result: list of outbound messages [{stream, topic, payload, channel?}]
            if result is None:
                return []
            if isinstance(result, dict):
                return [result]
            if isinstance(result, list):
                return result
            # Else wrap
            return [{"stream": None, "topic": "result", "payload": result}]
        except Exception as e:
            METRICS["task_failures_total"].labels(task_name).inc()
            raise

    def _publish_outputs(self, outs: List[Dict[str, Any]]):
        for msg in outs:
            stream_alias = msg.get("stream")  # alias key from outbound section, e.g., "factors"
            topic = msg.get("topic") or "result"
            payload = msg.get("payload") or {}
            ui_payload = msg.get("ui")  # optional small UI update

            if stream_alias:
                stream_name = self.outbound[stream_alias]["name"]
                publish_stream(self.redis, stream_name, {"topic": topic, "payload": payload})
                METRICS["msg_published"].labels(stream_name).inc()
            if ui_payload and self.ui_channel:
                publish_pubsub(self.redis, self.ui_channel, {"topic": topic, "payload": ui_payload})

    def loop(self):
        self.logger.info(json.dumps({"event": "loop_start"}))
        # Build xreadgroup args
        streams = []
        ids = []
        for key, st in self.inbound.items():
            streams.append(st["name"])
            # "$" to get new messages; ">" is special ID for xreadgroup (new only)
            ids.append(">")

        while not self._shutdown:
            try:
                # Calculate block time as min of configured
                block_ms = min([int(st.get("block_ms", 5000)) for st in self.inbound.values()]) if self.inbound else 5000
                resp = self.redis.xreadgroup(self.group, self.consumer, dict(zip(streams, ids)), count=200, block=block_ms)
            except redis.exceptions.ResponseError as e: # type: ignore
                # Group might not exist yet (race), re-ensure
                for st in self.inbound.values():
                    ensure_consumer_group(self.redis, st["name"], self.group, self.logger)
                continue

            if not resp:
                continue

            for stream_name, entries in resp: # type: ignore
                # Find the alias key for this stream
                stream_key = None
                for k, st in self.inbound.items():
                    if st["name"] == stream_name:
                        stream_key = k; break
                if stream_key is None:
                    # Unknown stream was read
                    continue

                METRICS["msg_consumed"].labels(stream_name).inc()

                for msg_id, fields in entries:
                    try:
                        # Expect "topic" and "payload" (JSON) fields; allow raw as fallback
                        topic = None
                        payload = {}
                        if "topic" in fields:
                            try:
                                topic = json.loads(fields["topic"]) if fields["topic"].startswith("{") else fields["topic"]
                            except Exception:
                                topic = fields["topic"]
                        if "payload" in fields:
                            try:
                                payload = json.loads(fields["payload"])
                            except Exception:
                                payload = {"raw": fields["payload"]}
                        else:
                            # flatten any fields
                            payload = fields

                        task_name = self._match_task(stream_key, topic)
                        if not task_name:
                            # No matching route, ack and skip
                            xack_safe(self.redis, stream_name, self.group, msg_id, self.logger)
                            continue

                        # Attach meta
                        payload_meta = {"_stream": stream_key, "_stream_name": stream_name, "_id": msg_id, "_topic": topic}
                        payload = {**payload, **payload_meta}

                        # Retry loop
                        attempt = 0
                        while True:
                            try:
                                outs = self._call_task(task_name, payload)
                                self._publish_outputs(outs)
                                break
                            except Exception as e:
                                attempt += 1
                                self.logger.error(json.dumps({
                                    "event": "task_error",
                                    "task": task_name,
                                    "attempt": attempt,
                                    "error": str(e)
                                }))
                                if attempt > self.max_retries:
                                    break
                                time.sleep(self.backoff_seconds)

                        # Ack regardless (poison-pill handling could be added)
                        xack_safe(self.redis, stream_name, self.group, msg_id, self.logger)

                    except Exception as e:
                        # Ack to avoid stuck messages, but log
                        self.logger.error(json.dumps({"event": "process_error", "stream": stream_name, "id": msg_id, "error": str(e)}))
                        xack_safe(self.redis, stream_name, self.group, msg_id, self.logger)

        self.logger.info(json.dumps({"event": "loop_end"}))

# ----------------------------- Main entry --------------------------------

def main():
    cfg_path = os.environ.get("ANALYST_CONFIG") or "analyst_worker.yaml"
    worker = AnalystWorker(cfg_path)
    worker.loop()

if __name__ == "__main__":
    main()
