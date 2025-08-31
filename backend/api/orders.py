# backend/api/orders.py
from __future__ import annotations

import os, time, json, hashlib, uuid
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel, Field, validator

# ---------- Optional deps (Redis & Prometheus) ----------
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

# ---------- Env / Defaults ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
INCOMING_ORDERS = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")   # pre-risk
ORDERS_UPDATES  = os.getenv("ORDERS_UPDATES_STREAM", "orders.updates")   # OMS emits acks/fills here
ORDERS_HASH     = os.getenv("ORDERS_HASH", "orders.hash")                # optional mirror
MAX_STREAM_LEN  = int(os.getenv("ORDERS_MAX_STREAM_LEN", "5000"))
HALT_KEY        = os.getenv("GOVERNOR_HALT_KEY", "govern:halt_trading")

# ---------- Metrics (optional) ----------
if Counter:
    ORDERS_CREATED = Counter("orders_created_total", "Orders created", ["side","type"])
    ORDERS_CANCELED = Counter("orders_canceled_total", "Orders canceled")
    ORDERS_REPLACED = Counter("orders_replaced_total", "Orders replaced")
    ORDERS_LATENCY = Histogram("orders_creation_latency_ms", "Client->server latency (ms)") # type: ignore
else:
    ORDERS_CREATED = ORDERS_CANCELED = ORDERS_REPLACED = None  # type: ignore
    ORDERS_LATENCY = None  # type: ignore

# ---------- Schemas ----------
Side = Literal["buy", "sell"]
OrdType = Literal["market", "limit", "post_only", "ioc", "fok"]

class OrderCreate(BaseModel):
    symbol: str
    side: Side
    qty: float = Field(..., gt=0)
    type: OrdType = "market"
    limit_price: Optional[float] = Field(None, gt=0)
    venue: Optional[str] = None
    strategy: Optional[str] = None
    region: Optional[str] = None
    tif: Optional[str] = Field(None, description="GTC|DAY|IOC|FOK")
    client_id: Optional[str] = Field(None, description="Idempotency token")
    meta: Dict[str, Any] = Field(default_factory=dict)
    ts_ms: Optional[int] = None

    @validator("symbol")
    def _sym(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol required")
        return v

    @validator("type")
    def _type_guard(cls, v: str, values):
        if v != "limit" and values.get("limit_price"):
            # ignore stray limit_price on non-limit types
            values["limit_price"] = None
        return v

class OrderReplace(BaseModel):
    qty: Optional[float] = Field(None, gt=0)
    limit_price: Optional[float] = Field(None, gt=0)
    tif: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class Order(BaseModel):
    id: str
    status: str = "accepted"  # accepted|rejected|working|filled|canceled|replaced|partial
    symbol: str
    side: Side
    qty: float
    type: OrdType
    limit_price: Optional[float] = None
    venue: Optional[str] = None
    strategy: Optional[str] = None
    region: Optional[str] = None
    tif: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    ts_ms: int
    last_update_ms: int

# ---------- In-memory fallback ----------
class _Mem:
    def __init__(self, cap: int = 5000):
        self.cap = cap
        self.rows: Dict[str, Dict[str, Any]] = {}

    def upsert(self, o: Dict[str, Any]):
        self.rows[o["id"]] = o
        if len(self.rows) > self.cap:
            # simple trim by oldest update
            for k, _ in sorted(self.rows.items(), key=lambda kv: kv[1].get("last_update_ms", 0))[:100]:
                self.rows.pop(k, None)

    def get(self, oid: str) -> Optional[Dict[str, Any]]:
        return self.rows.get(oid)

    def list(self, limit: int, symbol: Optional[str], status: Optional[str], strategy: Optional[str]) -> List[Dict[str, Any]]:
        items = list(self.rows.values())
        items.sort(key=lambda x: x.get("last_update_ms", 0), reverse=True)
        out: List[Dict[str, Any]] = []
        for it in items:
            if symbol and it.get("symbol") != symbol: continue
            if status and it.get("status") != status: continue
            if strategy and it.get("strategy") != strategy: continue
            out.append(it)
            if len(out) >= limit: break
        return out

_mem = _Mem()

# ---------- Redis dep ----------
async def get_redis() -> Optional[AsyncRedis]: # type: ignore
    if not USE_REDIS:
        return None
    try:
        r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        await r.ping()
        return r
    except Exception:
        return None

# ---------- Helpers ----------
def _mk_id(p: OrderCreate) -> str:
    # idempotency: prefer client_id if provided
    if p.client_id:
        return hashlib.sha1(p.client_id.encode()).hexdigest()
    seed = f"{p.symbol}|{p.side}|{p.qty}|{p.type}|{p.limit_price}|{int(time.time()//60)}|{uuid.uuid4()}"
    return hashlib.sha1(seed.encode()).hexdigest()

async def _halted(r: Optional[AsyncRedis]) -> bool: # type: ignore
    try:
        if r:
            v = await r.get(HALT_KEY)
            return str(v).lower() in {"1","true","yes","halt"}
    except Exception:
        pass
    return False

async def _xadd(r: AsyncRedis, stream: str, payload: Dict[str, Any]) -> None: # type: ignore
    await r.xadd(stream, {"json": json.dumps(payload)}, maxlen=MAX_STREAM_LEN, approximate=True)

# ---------- Routes ----------
@router.post("/orders", response_model=Order)
async def place_order(p: OrderCreate, r: Optional[AsyncRedis] = Depends(get_redis)) -> Order: # type: ignore
    now_ms = int(time.time() * 1000)
    if await _halted(r):
        raise HTTPException(status_code=423, detail="Trading halted by governor")

    oid = _mk_id(p)
    order_payload = {
        "id": oid,
        "ts_ms": p.ts_ms or now_ms,
        "last_update_ms": now_ms,
        "symbol": p.symbol,
        "side": p.side,
        "qty": float(p.qty),
        "typ": p.type,
        "limit_price": p.limit_price,
        "venue": p.venue,
        "strategy": p.strategy,
        "region": p.region,
        "tif": p.tif,
        "meta": p.meta or {},
        "source": "api.orders",
        "op": "new",
    }

    if r:
        try:
            await _xadd(r, INCOMING_ORDERS, order_payload)  # risk → OMS pipeline
            await r.hset(ORDERS_HASH, oid, json.dumps(order_payload))
        except Exception as e:
            # fallback to memory
            pass
    _mem.upsert({**order_payload, "status": "accepted"})

    if ORDERS_CREATED:
        ORDERS_CREATED.labels(p.side, p.type).inc()
        if p.ts_ms:
            ORDERS_LATENCY.observe(max(0, now_ms - p.ts_ms))  # type: ignore

    return Order(
        id=oid, status="accepted", symbol=p.symbol, side=p.side, qty=p.qty,
        type=p.type, limit_price=p.limit_price, venue=p.venue, strategy=p.strategy,
        region=p.region, tif=p.tif, meta=p.meta or {}, ts_ms=order_payload["ts_ms"],
        last_update_ms=now_ms
    )

@router.get("/orders", response_model=List[Order])
async def list_orders(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    strategy: Optional[str] = None,
    r: Optional[AsyncRedis] = Depends(get_redis), # type: ignore
) -> List[Order]:
    # Prefer Redis hash (mirror) if present; otherwise memory.
    rows: List[Dict[str, Any]] = []
    if r:
        try:
            h = await r.hgetall(ORDERS_HASH)
            for _, v in h.items():
                try:
                    rows.append(json.loads(v))
                except Exception:
                    continue
        except Exception:
            pass
    if not rows:
        rows = _mem.list(limit*3, None, None, None)

    # Filter + normalize
    rows.sort(key=lambda x: x.get("last_update_ms", 0), reverse=True)
    out: List[Order] = []
    for it in rows:
        if symbol and it.get("symbol") != symbol: continue
        if status and it.get("status") != status: continue
        if strategy and it.get("strategy") != strategy: continue
        out.append(Order(
            id=it["id"], status=it.get("status","working"), symbol=it["symbol"], side=it["side"],
            qty=float(it["qty"]), type=it.get("typ","market"),
            limit_price=it.get("limit_price"), venue=it.get("venue"),
            strategy=it.get("strategy"), region=it.get("region"), tif=it.get("tif"),
            meta=it.get("meta",{}), ts_ms=int(it.get("ts_ms", it.get("last_update_ms", 0))),
            last_update_ms=int(it.get("last_update_ms", it.get("ts_ms", 0)))
        ))
        if len(out) >= limit: break
    return out

@router.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: str, r: Optional[AsyncRedis] = Depends(get_redis)) -> Order: # type: ignore
    it = None
    if r:
        try:
            raw = await r.hget(ORDERS_HASH, order_id)
            if raw:
                it = json.loads(raw)
        except Exception:
            it = None
    if it is None:
        it = _mem.get(order_id)
    if it is None:
        raise HTTPException(status_code=404, detail="Order not found")
    return Order(
        id=it["id"], status=it.get("status","working"), symbol=it["symbol"], side=it["side"],
        qty=float(it["qty"]), type=it.get("typ","market"),
        limit_price=it.get("limit_price"), venue=it.get("venue"),
        strategy=it.get("strategy"), region=it.get("region"), tif=it.get("tif"),
        meta=it.get("meta",{}), ts_ms=int(it.get("ts_ms", it.get("last_update_ms", 0))),
        last_update_ms=int(it.get("last_update_ms", it.get("ts_ms", 0)))
    )

@router.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str, r: Optional[AsyncRedis] = Depends(get_redis)) -> Dict[str, Any]: # type: ignore
    now = int(time.time() * 1000)
    payload = {"id": order_id, "op": "cancel", "ts_ms": now, "source": "api.orders"}
    if r:
        try:
            await _xadd(r, INCOMING_ORDERS, payload)
        except Exception:
            pass
    row = _mem.get(order_id) or {"id": order_id, "status": "canceled"}
    row["status"] = "canceled"
    row["last_update_ms"] = now
    _mem.upsert(row)
    if ORDERS_CANCELED: ORDERS_CANCELED.inc()
    return {"ok": True, "id": order_id, "status": "canceled", "ts_ms": now}

@router.post("/orders/{order_id}/replace")
async def replace_order(order_id: str, req: OrderReplace, r: Optional[AsyncRedis] = Depends(get_redis)) -> Dict[str, Any]: # type: ignore
    now = int(time.time() * 1000)
    patch = {"id": order_id, "op": "replace", "ts_ms": now, "source": "api.orders"}
    if req.qty is not None: patch["qty"] = float(req.qty)
    if req.limit_price is not None: patch["limit_price"] = float(req.limit_price)
    if req.tif is not None: patch["tif"] = req.tif
    if req.meta: patch["meta"] = req.meta

    if r:
        try:
            await _xadd(r, INCOMING_ORDERS, patch)
        except Exception:
            pass
    row = _mem.get(order_id)
    if row:
        row.update({k:v for k,v in patch.items() if k not in {"op","source"}})
        row["status"] = "replaced"
        row["last_update_ms"] = now
        _mem.upsert(row)
    if ORDERS_REPLACED: ORDERS_REPLACED.inc()
    return {"ok": True, "id": order_id, "status": "replaced", "ts_ms": now, "patch": patch}

# ---------- WebSocket: live updates ----------
class _Hub:
    def __init__(self):
        self.clients: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept(); self.clients.append(ws)
    def disconnect(self, ws: WebSocket):
        try: self.clients.remove(ws)
        except ValueError: pass
    async def broadcast(self, msg: Dict[str, Any]):
        dead=[]
        for ws in self.clients:
            try: await ws.send_json(msg)
            except Exception: dead.append(ws)
        for ws in dead: self.disconnect(ws)

_hub = _Hub()

@router.websocket("/ws/orders")
async def ws_orders(ws: WebSocket):
    await _hub.connect(ws)
    try:
        # initial snapshot from memory
        for it in _mem.list(limit=50, symbol=None, status=None, strategy=None):
            await ws.send_json(it)
        # live tail from Redis stream if available; otherwise heartbeat
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
                    resp = await r.xread({ORDERS_UPDATES: last_id}, count=50, block=5000)
                    if resp:
                        _, entries = resp[0]
                        for _id, fields in entries:
                            last_id = _id
                            obj = json.loads(fields.get("json", "{}"))
                            # mirror to memory for REST consumers
                            oid = obj.get("id") or obj.get("order_id")
                            if oid:
                                row = _mem.get(oid) or {"id": oid}
                                row.update(obj)
                                row["last_update_ms"] = int(time.time() * 1000)
                                _mem.upsert(row)
                            await ws.send_json(obj)
                except Exception:
                    await ws.send_json({"type": "heartbeat", "ts_ms": int(time.time()*1000)})
            else:
                await ws.send_json({"type": "heartbeat", "ts_ms": int(time.time()*1000)})
                await ws.receive_text()
    except WebSocketDisconnect:
        _hub.disconnect(ws)
    except Exception:
        _hub.disconnect(ws)