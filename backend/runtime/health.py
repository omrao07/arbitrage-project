# backend/engine/health.py
from __future__ import annotations

import os
import time
import socket
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

import redis

from backend.execution.broker_base import BrokerBase, PaperBroker # type: ignore

log = logging.getLogger("health")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

@dataclass
class HealthStatus:
    component: str
    ok: bool
    detail: Optional[str] = None
    latency_ms: Optional[float] = None

@dataclass
class HealthReport:
    ts_ms: int
    statuses: Dict[str, HealthStatus]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "components": {k: asdict(v) for k, v in self.statuses.items()}
        }

# ------------------- Checks -------------------

def check_redis() -> HealthStatus:
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        t0 = time.time()
        pong = r.ping()
        dt = (time.time() - t0) * 1000
        return HealthStatus("redis", ok=pong, detail="pong" if pong else "fail", latency_ms=dt) # type: ignore
    except Exception as e:
        return HealthStatus("redis", ok=False, detail=str(e))

def check_broker(broker: Optional[BrokerBase] = None) -> HealthStatus:
    try:
        if broker is None:
            broker = PaperBroker()
            broker.connect() # type: ignore
        t0 = time.time()
        acct = broker.get_account() # type: ignore
        dt = (time.time() - t0) * 1000
        return HealthStatus("broker", ok=True, detail=f"equity={acct.equity}", latency_ms=dt)
    except Exception as e:
        return HealthStatus("broker", ok=False, detail=str(e))

def check_network(host: str = "8.8.8.8", port: int = 53, timeout: int = 2) -> HealthStatus:
    """
    Simple network reachability check (default: Google DNS).
    """
    try:
        t0 = time.time()
        sock = socket.create_connection((host, port), timeout)
        sock.close()
        dt = (time.time() - t0) * 1000
        return HealthStatus("network", ok=True, detail=f"{host}:{port} reachable", latency_ms=dt)
    except Exception as e:
        return HealthStatus("network", ok=False, detail=str(e))

# ------------------- Main -------------------

def run_checks(broker: Optional[BrokerBase] = None) -> HealthReport:
    statuses = {}
    statuses["redis"] = check_redis()
    statuses["broker"] = check_broker(broker)
    statuses["network"] = check_network()
    return HealthReport(ts_ms=int(time.time() * 1000), statuses=statuses)

# ------------------- Optional HTTP server -------------------

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    @app.get("/health")
    def health_endpoint() -> Dict[str, Any]:
        rep = run_checks()
        return JSONResponse(content=rep.as_dict()) # type: ignore

    def serve(host: str = "0.0.0.0", port: int = 8081): # type: ignore
        log.info(f"Starting health server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

except ImportError:
    # FastAPI not installed — only programmatic use
    def serve(*args, **kwargs):
        log.error("FastAPI/uvicorn not available; pip install fastapi uvicorn to enable HTTP health endpoint.")