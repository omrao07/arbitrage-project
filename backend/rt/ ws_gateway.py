# backend/gateway/ws_gateway.py
"""
WebSocket Gateway for the Event Bus
-----------------------------------
Expose your internal Redis streams to browser/desktop UIs via WebSockets.

✓ Subscribe/unsubscribe to multiple streams
✓ Tail Redis Streams (XREAD) with per-conn cursors
✓ Optional publish back into bus (XADD wrapper)
✓ Per-connection rate limiting & backpressure
✓ Heartbeats (ping/pong) and idle timeouts
✓ Simple bearer/API-key auth (shared secret or JWT)
✓ CORS and origin allowlist
✓ Graceful shutdown

Client message schema (JSON)
----------------------------
# subscribe to one or more streams
{"op":"sub", "streams":["insights.events","risk.governor"], "from":"$"}   # from: "$" or "0-0"
# unsubscribe
{"op":"unsub", "streams":["risk.governor"]}
# publish (optional, can be disabled)
{"op":"pub", "stream":"strategy.cmd.momo_in", "payload":{"cmd":"pause"}}
# ping (client-initiated)
{"op":"ping"}

Server -> client messages
-------------------------
{"op":"hello", "sid":"<uuid>", "server_ts": 1699999999999}
{"op":"event", "stream":"insights.events", "id":"1700000-0", "data":{...}}
{"op":"pong"}
{"op":"info"|"warn"|"error", "msg":"...", "code":"..."}
{"op":"bye","reason":"..."}

Env vars
--------
WS_PORT=8080
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
ALLOW_PUBLISH=true|false
WS_TOKEN=...                   # static bearer token (optional)
JWT_PUBLIC_KEY_PATH=...        # optional RSA/EC public key (PEM) for Authorization: Bearer <jwt>
ALLOWED_ORIGINS=https://app.local,https://your.site
IDLE_TIMEOUT_S=180
MAX_SEND_Q=1000                # per-conn outbound buffer
XREAD_BLOCK_MS=800
XREAD_BATCH=200
RATE_MAX_PUB_PER_MIN=120
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# --- Redis (async) ---
try:
    import redis.asyncio as aioredis  # pip install redis>=4.6
except Exception as e:  # pragma: no cover
    aioredis = None
    raise

# --- Optional JWT auth ---
try:
    import jwt  # type: ignore # pip install pyjwt
except Exception:
    jwt = None  # type: ignore

# --- Local bus helpers (optional best-effort) ---
try:
    from backend.bus.streams import publish_stream  # sync; we'll wrap
except Exception:
    publish_stream = None  # type: ignore

# ---------- Config ----------
def _bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in ("1","true","yes","y","on")

PORT               = int(os.getenv("WS_PORT", "8080"))
REDIS_HOST         = os.getenv("REDIS_HOST","localhost")
REDIS_PORT         = int(os.getenv("REDIS_PORT","6379"))
REDIS_DB           = int(os.getenv("REDIS_DB","0"))
ALLOW_PUBLISH      = _bool(os.getenv("ALLOW_PUBLISH","true"))
STATIC_TOKEN       = os.getenv("WS_TOKEN")
JWT_PUBKEY_PATH    = os.getenv("JWT_PUBLIC_KEY_PATH")
ALLOWED_ORIGINS    = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","*").split(",") if o.strip()]
IDLE_TIMEOUT_S     = int(os.getenv("IDLE_TIMEOUT_S","180"))
MAX_SEND_Q         = int(os.getenv("MAX_SEND_Q","1000"))
XREAD_BLOCK_MS     = int(os.getenv("XREAD_BLOCK_MS","800"))
XREAD_BATCH        = int(os.getenv("XREAD_BATCH","200"))
RATE_MAX_PUB       = int(os.getenv("RATE_MAX_PUB_PER_MIN","120"))

# ---------- App ----------
app = FastAPI(title="WS Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Auth ----------
_pubkey_cache: Optional[str] = None
if JWT_PUBKEY_PATH and os.path.exists(JWT_PUBKEY_PATH):
    with open(JWT_PUBKEY_PATH,"r",encoding="utf-8") as f:
        _pubkey_cache = f.read()

async def _auth(websocket: WebSocket) -> Dict[str, Any]:
    """
    Accept either:
      - Query param token=... (or header Authorization: Bearer <token>) matching WS_TOKEN
      - JWT in Authorization header, verified by JWT_PUBLIC_KEY_PATH (if provided)
    On failure: raise and close.
    """
    token = websocket.query_params.get("token")
    auth = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ",1)[1].strip()

    if STATIC_TOKEN:
        if token == STATIC_TOKEN:
            return {"sub":"static","scopes":["*"]}
    # JWT path
    if token and _pubkey_cache and jwt:
        try:
            payload = jwt.decode(token, _pubkey_cache, algorithms=["RS256","ES256","EdDSA"])
            return {"sub": payload.get("sub","jwt"), "scopes": payload.get("scopes", ["*"]), "jwt": payload}
        except Exception:
            pass

    if STATIC_TOKEN or _pubkey_cache:
        # auth required
        await websocket.close(code=4401)
        raise WebSocketDisconnect(code=4401)

    # auth not configured => anonymous ok (read-only)
    return {"sub":"anon","scopes":["read"]}

# ---------- Data types ----------
@dataclass
class ConnState:
    sid: str
    ws: WebSocket
    authed: Dict[str, Any]
    subs: Dict[str, str] = field(default_factory=dict)  # stream -> last_id cursor
    send_q: "asyncio.Queue[Dict[str, Any]]" = field(default_factory=asyncio.Queue)
    last_recv_ts: float = field(default_factory=lambda: time.time())
    last_send_ts: float = field(default_factory=lambda: time.time())
    pub_timestamps: List[float] = field(default_factory=list)  # for rate limiting

    def touch_recv(self):
        self.last_recv_ts = time.time()
    def touch_send(self):
        self.last_send_ts = time.time()

# ---------- Redis client ----------
redis_client: Optional[aioredis.Redis] = None # type: ignore

@app.on_event("startup")
async def _startup():
    global redis_client
    redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True) # type: ignore
    try:
        await redis_client.ping() # type: ignore
    except Exception as e:
        print(f"[ws-gateway] Redis ping failed: {e}")
        raise

@app.on_event("shutdown")
async def _shutdown():
    global redis_client
    if redis_client:
        await redis_client.close()

# ---------- Health ----------
@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"

# ---------- WebSocket endpoint ----------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, user=Depends(_auth)):
    await ws.accept()
    sid = str(uuid.uuid4())
    state = ConnState(sid=sid, ws=ws, authed=user)

    # greet
    await _safe_send(state, {"op":"hello","sid":sid,"server_ts":int(time.time()*1000),"read_only": (not ALLOW_PUBLISH)})

    # tasks: reader (from client), bus_reader (from Redis), sender (to client), heartbeat
    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(_client_reader(state, stop)),
        asyncio.create_task(_bus_reader(state, stop)),
        asyncio.create_task(_sender(state, stop)),
        asyncio.create_task(_heartbeats(state, stop)),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception:
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await _safe_send(state, {"op":"bye","reason":"closing"})
        try:
            await ws.close()
        except Exception:
            pass

# ---------- Task: read messages from client ----------
async def _client_reader(state: ConnState, stop: asyncio.Event):
    ws = state.ws
    while not stop.is_set():
        try:
            raw = await ws.receive_text()
            state.touch_recv()
        except WebSocketDisconnect:
            break
        except Exception:
            break

        try:
            msg = json.loads(raw)
        except Exception:
            await _enqueue(state, {"op":"error","code":"bad_json","msg":"Invalid JSON"})
            continue

        op = str(msg.get("op","")).lower()
        if op == "ping":
            await _enqueue(state, {"op":"pong"})
            continue

        if op == "sub":
            streams = list(msg.get("streams") or [])
            start_from = str(msg.get("from","$"))
            for s in streams:
                # "$" means new messages only; "0-0" means from beginning
                state.subs[s] = start_from
            await _enqueue(state, {"op":"info","msg":f"subscribed:{streams}"})
            continue

        if op == "unsub":
            streams = list(msg.get("streams") or [])
            for s in streams:
                state.subs.pop(s, None)
            await _enqueue(state, {"op":"info","msg":f"unsubscribed:{streams}"})
            continue

        if op == "pub":
            if not ALLOW_PUBLISH:
                await _enqueue(state, {"op":"error","code":"publish_disabled","msg":"Server publish disabled"})
                continue
            # rudimentary rate limit
            now = time.time()
            state.pub_timestamps = [t for t in state.pub_timestamps if now - t < 60.0]
            if len(state.pub_timestamps) >= RATE_MAX_PUB:
                await _enqueue(state, {"op":"warn","code":"rate_limit","msg":"Too many publish calls"})
                continue
            stream = str(msg.get("stream") or "")
            payload = msg.get("payload") or {}
            if not stream or not isinstance(payload, dict):
                await _enqueue(state, {"op":"error","code":"bad_publish","msg":"Missing stream or payload"})
                continue
            state.pub_timestamps.append(now)
            try:
                await _xadd(stream, payload)
                await _enqueue(state, {"op":"info","msg":f"published:{stream}"})
            except Exception as e:
                await _enqueue(state, {"op":"error","code":"publish_failed","msg":str(e)})
            continue

        await _enqueue(state, {"op":"error","code":"unknown_op","msg":op})

# ---------- Task: read from Redis streams ----------
async def _bus_reader(state: ConnState, stop: asyncio.Event):
    if not redis_client:
        await _enqueue(state, {"op":"error","code":"redis_unavailable","msg":"no redis client"})
        return
    while not stop.is_set():
        if not state.subs:
            await asyncio.sleep(0.05)
            continue

        # Build XREAD args
        streams: List[str] = []
        cursors: List[str] = []
        for s, cur in state.subs.items():
            streams.append(s); cursors.append(cur)

        try:
            resp = await redis_client.xread(streams=streams, count=XREAD_BATCH, block=XREAD_BLOCK_MS, latest_ids=cursors)
        except Exception:
            await asyncio.sleep(0.05)
            continue

        if not resp:
            continue

        # resp is list of (stream, [(id, {k:v})...])
        for stream_name, entries in resp:
            last_id = state.subs.get(stream_name, "$")
            for (xid, data) in entries:
                last_id = xid
                # normalize data to JSONable
                try:
                    # best-effort: attempt to parse JSON-y strings
                    norm = {k: _maybe_json(v) for k, v in (data or {}).items()}
                except Exception:
                    norm = data or {}
                await _enqueue(state, {"op":"event","stream":stream_name,"id":xid,"data":norm})
            # advance cursor
            state.subs[stream_name] = last_id

# ---------- Task: sender with backpressure ----------
async def _sender(state: ConnState, stop: asyncio.Event):
    while not stop.is_set():
        try:
            msg = await state.send_q.get()
        except Exception:
            break
        try:
            await state.ws.send_text(json.dumps(msg, separators=(",",":")))
            state.touch_send()
        except Exception:
            break

# ---------- Task: heartbeats & idle timeouts ----------
async def _heartbeats(state: ConnState, stop: asyncio.Event):
    interval = 15.0
    while not stop.is_set():
        await asyncio.sleep(interval)
        # idle timeout
        idle = time.time() - max(state.last_recv_ts, state.last_send_ts)
        if IDLE_TIMEOUT_S > 0 and idle > IDLE_TIMEOUT_S:
            await _enqueue(state, {"op":"bye","reason":"idle_timeout"})
            try:
                await state.ws.close()
            except Exception:
                pass
            break
        # server ping
        await _enqueue(state, {"op":"ping","ts":int(time.time()*1000)})

# ---------- Helpers ----------
async def _enqueue(state: ConnState, payload: Dict[str, Any]):
    if state.send_q.qsize() >= MAX_SEND_Q:
        # blunt backpressure: drop oldest and warn
        try:
            _ = state.send_q.get_nowait()
        except Exception:
            pass
        # also send a warning (best effort)
        payload = {"op":"warn","code":"backpressure","msg":"send queue full; dropping oldest"}
    try:
        await state.send_q.put(payload)
    except Exception:
        pass

async def _safe_send(state: ConnState, payload: Dict[str, Any]):
    try:
        await state.ws.send_text(json.dumps(payload, separators=(",",":")))
        state.touch_send()
    except Exception:
        pass

async def _xadd(stream: str, data: Dict[str, Any]) -> None:
    """
    Publish to Redis Stream. If your local bus helper exists, mirror it.
    """
    if publish_stream:
        try:
            # fire-and-forget; helper may be sync
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, publish_stream, stream, data)
        except Exception:
            pass
    if redis_client:
        # Redis expects flat dict of str->str
        flat = {str(k): json.dumps(v) if isinstance(v,(dict,list)) else str(v) for k, v in data.items()}
        await redis_client.xadd(stream, flat, maxlen=None)

def _maybe_json(v: Any) -> Any:
    if not isinstance(v, str): return v
    s = v.strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return v
    # try numeric
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return v

# ---------- Entrypoint ----------
def main():
    import uvicorn  # pip install uvicorn
    uvicorn.run("backend.gateway.ws_gateway:app", host="0.0.0.0", port=PORT, reload=False)

if __name__ == "__main__":
    main()