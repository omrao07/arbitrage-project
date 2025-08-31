# backend/api/alerts.py
from __future__ import annotations

import os
import time
import json
import hashlib
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field, validator

# ---- Optional deps (Redis & Prometheus) -------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

try:
    from prometheus_client import Counter, Histogram # type: ignore
except Exception:
    Counter = Histogram = None  # type: ignore

router = APIRouter()

# ---- Env / Defaults ---------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ALERTS_STREAM = os.getenv("ALERTS_STREAM", "alerts.stream")
ALERTS_ACK_HASH = os.getenv("ALERTS_ACK_HASH", "alerts.ack")
MAX_STREAM_LEN = int(os.getenv("ALERTS_MAX_STREAM_LEN", "5000"))
THROTTLE_WINDOW_S = int(os.getenv("ALERTS_THROTTLE_WINDOW_S", "10"))  # per key
THROTTLE_MAX_PER_WINDOW = int(os.getenv("ALERTS_THROTTLE_MAX", "5"))

# ---- Prometheus (optional) --------------------------------------------------
if Counter:
    ALERTS_CREATED = Counter("alerts_created_total", "Number of alerts created", ["type"])
    ALERTS_ACKED = Counter("alerts_acked_total", "Number of alerts acknowledged", ["type"])
    ALERTS_WS_CONNECTED = Counter("alerts_ws_connected_total", "Active websockets")
    ALERTS_WS_SENT = Counter("alerts_ws_sent_total", "Alerts pushed over WS", ["type"])
    ALERTS_LATENCY = Histogram("alerts_creation_latency_ms", "Client->server latency (ms)") # type: ignore
else:
    ALERTS_CREATED = ALERTS_ACKED = ALERTS_WS_CONNECTED = ALERTS_WS_SENT = None  # type: ignore
    ALERTS_LATENCY = None  # type: ignore

# ---- Schemas ----------------------------------------------------------------
class AlertCreate(BaseModel):
    type: str = Field(..., description="e.g., vol_spike, price_cross, policy_breach")
    symbol: Optional[str] = Field(None, description="Ticker or instrument")
    severity: str = Field("info", description="info|warn|critical")
    title: str
    message: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    ts_ms: Optional[int] = Field(None, description="Client timestamp (ms epoch)")

    @validator("severity")
    def _sev(cls, v: str) -> str:
        v = v.lower()
        if v not in {"info", "warn", "critical"}:
            raise ValueError("severity must be one of: info|warn|critical")
        return v


class Alert(AlertCreate):
    id: str
    acked: bool = False
    ack_ts_ms: Optional[int] = None


class AckResponse(BaseModel):
    id: str
    acked: bool
    ack_ts_ms: int


# ---- In-memory fallback (ring buffer) ---------------------------------------
class _Ring:
    def __init__(self, cap: int = 1000):
        self.cap = cap
        self.buf: List[Dict[str, Any]] = []
        self.acks: Dict[str, int] = {}

    def append(self, a: Dict[str, Any]):
        self.buf.append(a)
        if len(self.buf) > self.cap:
            self.buf = self.buf[-self.cap :]

    def list(self, limit: int = 100, symbol: Optional[str] = None,
             typ: Optional[str] = None, ack: Optional[bool] = None) -> List[Dict[str, Any]]:
        out = []
        for it in reversed(self.buf):
            if symbol and it.get("symbol") != symbol:
                continue
            if typ and it.get("type") != typ:
                continue
            if ack is not None and bool(it.get("acked", False)) != ack:
                continue
            out.append(it)
            if len(out) >= limit:
                break
        return list(reversed(out))

    def ack(self, aid: str) -> Optional[Dict[str, Any]]:
        for it in reversed(self.buf):
            if it["id"] == aid:
                it["acked"] = True
                it["ack_ts_ms"] = int(time.time() * 1000)
                self.acks[aid] = it["ack_ts_ms"]
                return it
        return None


_mem = _Ring(cap=2000)

# ---- Redis dependency --------------------------------------------------------
async def get_redis() -> Optional[AsyncRedis]: # type: ignore
    if not USE_REDIS:
        return None
    try:
        r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        # quick ping to surface connection issues fast
        await r.ping()
        return r
    except Exception:
        return None


# ---- Helpers ----------------------------------------------------------------
def _mk_id(payload: Dict[str, Any]) -> str:
    # stable-ish id: type|symbol|title|message|bucket_min
    bucket = int(time.time() // 60)  # minute bucket
    base = f"{payload.get('type','')}|{payload.get('symbol','')}|{payload.get('title','')}|{payload.get('message','')}|{bucket}"
    return hashlib.sha1(base.encode()).hexdigest()  # 40 hex chars

_throttle_cache: Dict[str, List[int]] = {}

def _throttle_key(p: Dict[str, Any]) -> str:
    return f"{p.get('type','')}|{p.get('symbol','')}|{p.get('severity','')}"

def _pass_throttle(p: Dict[str, Any]) -> bool:
    now = int(time.time())
    key = _throttle_key(p)
    arr = _throttle_cache.setdefault(key, [])
    # drop old
    arr[:] = [t for t in arr if now - t <= THROTTLE_WINDOW_S]
    if len(arr) >= THROTTLE_MAX_PER_WINDOW:
        return False
    arr.append(now)
    return True


async def _publish_alert_redis(r: AsyncRedis, alert: Dict[str, Any]) -> None: # type: ignore
    # use Redis Streams with capped length
    await r.xadd(ALERTS_STREAM, {"json": json.dumps(alert)}, maxlen=MAX_STREAM_LEN, approximate=True)


async def _ack_redis(r: AsyncRedis, alert_id: str, alert_type: str) -> None: # type: ignore
    # maintain ack map in hash
    ts = int(time.time() * 1000)
    await r.hset(ALERTS_ACK_HASH, alert_id, ts)
    if ALERTS_ACKED:
        ALERTS_ACKED.labels(alert_type).inc()


async def _read_recent_redis(r: AsyncRedis, count: int = 100) -> List[Dict[str, Any]]: # type: ignore
    try:
        # read last N
        entries = await r.xrevrange(ALERTS_STREAM, count=count)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for _id, fields in reversed(entries):
        try:
            obj = json.loads(fields.get("json", "{}"))
            out.append(obj)
        except Exception:
            continue
    return out


# ---- Routes -----------------------------------------------------------------
@router.post("/alerts", response_model=Alert)
async def create_alert(payload: AlertCreate, r: Optional[AsyncRedis] = Depends(get_redis)) -> Alert: # type: ignore
    server_recv_ms = int(time.time() * 1000)
    data = payload.dict()
    data["ts_ms"] = data.get("ts_ms") or server_recv_ms
    data["acked"] = False
    data["ack_ts_ms"] = None
    data["id"] = _mk_id(data)

    if not _pass_throttle(data):
        # silently drop duplicates in a tight window
        raise HTTPException(status_code=429, detail="Throttled: too many similar alerts")

    if ALERTS_CREATED:
        ALERTS_CREATED.labels(data["type"]).inc()
        if payload.ts_ms:
            ALERTS_LATENCY.observe(max(0, server_recv_ms - payload.ts_ms))  # type: ignore

    # publish
    if r:
        await _publish_alert_redis(r, data)
    else:
        _mem.append(data)

    return Alert(**data)


@router.get("/alerts", response_model=List[Alert])
async def list_alerts(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = None,
    type: Optional[str] = Query(None, alias="typ"),
    ack: Optional[bool] = None,
    r: Optional[AsyncRedis] = Depends(get_redis), # type: ignore
) -> List[Alert]:
    rows: List[Dict[str, Any]]
    if r:
        rows = await _read_recent_redis(r, count=limit * 3)  # fetch a bit more, filter below
    else:
        rows = _mem.list(limit=limit * 3, symbol=None, typ=None, ack=None)

    out: List[Alert] = []
    for it in reversed(rows):
        if symbol and it.get("symbol") != symbol:
            continue
        if type and it.get("type") != type:
            continue
        if ack is not None and bool(it.get("acked", False)) != ack:
            continue
        out.append(Alert(**it))
        if len(out) >= limit:
            break
    return list(reversed(out))


@router.post("/alerts/{alert_id}/ack", response_model=AckResponse)
async def ack_alert(alert_id: str, r: Optional[AsyncRedis] = Depends(get_redis)) -> AckResponse: # type: ignore
    ts = int(time.time() * 1000)
    # best-effort: record ack in redis; update memory mirror
    if r:
        await _ack_redis(r, alert_id, alert_type="unknown")
    res = _mem.ack(alert_id)
    if not res:
        # if not in memory, still return ack (idempotent behavior)
        res = {"id": alert_id, "acked": True, "ack_ts_ms": ts}
    return AckResponse(id=alert_id, acked=True, ack_ts_ms=res["ack_ts_ms"])


# ---- WebSocket: live alert stream ------------------------------------------
class _Hub:
    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)
        if ALERTS_WS_CONNECTED:
            ALERTS_WS_CONNECTED.inc()

    def disconnect(self, ws: WebSocket):
        try:
            self.clients.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, msg: Dict[str, Any]):
        dead: List[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

_hub = _Hub()


@router.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await _hub.connect(ws)
    try:
        # initial burst: last 50
        for it in _mem.list(limit=50):
            await ws.send_json(it)

        # live loop: either consume Redis stream or receive no-op pings and stay connected
        last_id = "$"
        r: Optional[AsyncRedis] = None # type: ignore
        if USE_REDIS:
            try:
                r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
                await r.ping() # type: ignore
            except Exception:
                r = None

        while True:
            if r:
                try:
                    resp = await r.xread({ALERTS_STREAM: last_id}, count=50, block=5000)
                    if resp:
                        _, entries = resp[0]
                        for _id, fields in entries:
                            last_id = _id
                            obj = json.loads(fields.get("json", "{}"))
                            await ws.send_json(obj)
                            if ALERTS_WS_SENT:
                                ALERTS_WS_SENT.labels(obj.get("type", "unknown")).inc()
                except Exception:
                    # fall back to heartbeat
                    await ws.send_json({"type": "heartbeat", "ts_ms": int(time.time() * 1000)})
            else:
                # no redis: periodic heartbeat; client can poll /alerts
                await ws.send_json({"type": "heartbeat", "ts_ms": int(time.time() * 1000)})
                await ws.receive_text()  # backpressure; ignore content
    except WebSocketDisconnect:
        _hub.disconnect(ws)
    except Exception:
        _hub.disconnect(ws)