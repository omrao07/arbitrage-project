# backend/gateway.py
from __future__ import annotations
import os, json, asyncio, signal, time, traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Set, DefaultDict, Tuple, List
from collections import defaultdict

# ---------- FastAPI / WS ----------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- Redis (graceful if missing) ----------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

APP_NAME = os.getenv("APP_NAME", "ws-gateway")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")  # if set, required for WS & POSTs
WS_PING_SEC = int(os.getenv("WS_PING_SEC", "20"))
MAX_QUEUE = int(os.getenv("WS_MAX_QUEUE", "5000"))  # per-connection outgoing buffer (messages)
MAX_READ_BATCH = int(os.getenv("REDIS_MAX_BATCH", "200"))
XREAD_BLOCK_MS = int(os.getenv("REDIS_BLOCK_MS", "1000"))
STREAM_MAXLEN = int(os.getenv("STREAM_MAXLEN", "50000"))

# Map logical topics -> Redis stream names (override via env)
TOPIC_STREAMS: Dict[str, str] = {
    "candles":      os.getenv("S_CANDLES", "ws.candles"),
    "orderbook":    os.getenv("S_ORDERBOOK", "ws.orderbook"),
    "greeks":       os.getenv("S_GREEKS", "ws.greeks"),
    "alerts":       os.getenv("S_ALERTS", "alerts.events"),
    "fills":        os.getenv("S_FILLS", "orders.filled"),
    "pnl":          os.getenv("S_PNL", "pnl.stream"),
    "news":         os.getenv("S_NEWS", "news.events"),
    "status":       os.getenv("S_STATUS", "ops.status"),
}
# You can add more topics -> streams above

# ---------- App ----------
app = FastAPI(title=APP_NAME, version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Auth ----------
def require_token(req: Request):
    if not AUTH_TOKEN:
        return
    tok = req.headers.get("authorization") or req.query_params.get("token") or ""
    if tok.lower().startswith("bearer "):
        tok = tok.split(" ", 1)[1]
    if tok != AUTH_TOKEN:
        raise HTTPException(401, "unauthorized")

async def ws_require_token(ws: WebSocket):
    if not AUTH_TOKEN:
        return
    # try header, then query ?token=
    tok = ws.headers.get("authorization") or ws.query_params.get("token")
    if tok and tok.lower().startswith("bearer "):
        tok = tok.split(" ", 1)[1]
    if tok != AUTH_TOKEN:
        await ws.close(code=4401)
        raise RuntimeError("unauthorized")

# ---------- State ----------
@dataclass
class Client:
    ws: WebSocket
    topics: Set[str]
    q: asyncio.Queue  # outgoing messages

clients: Set[Client] = set()
topic_to_clients: DefaultDict[str, Set[Client]] = defaultdict(set)

# Redis handle and consumer task
redis: Optional[AsyncRedis] = None # type: ignore
consumer_task: Optional[asyncio.Task] = None
running = True

# Track the last IDs we read per stream
last_ids: Dict[str, str] = {stream: "$" for stream in TOPIC_STREAMS.values()}  # new messages only by default

# ---------- Utilities ----------
def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"

async def _send_safe(cli: Client, payload: Dict[str, Any]):
    """Send to a client with backpressure and disconnect safety."""
    if cli.q.qsize() > MAX_QUEUE:
        # drop oldest to keep UI responsive
        try:
            _ = cli.q.get_nowait()
        except Exception:
            pass
    await cli.q.put(payload)

async def _broadcaster(cli: Client):
    """Per-client sender: drains queue and writes to WS."""
    try:
        while True:
            msg = await cli.q.get()
            await cli.ws.send_text(_json(msg))
    except WebSocketDisconnect:
        pass
    except Exception:
        # swallow, close handled by outer
        pass

async def _heartbeat(ws: WebSocket):
    """Ping to keep proxies happy."""
    while True:
        await asyncio.sleep(WS_PING_SEC)
        try:
            await ws.send_text(_json({"type":"ping","ts":int(time.time()*1000)}))
        except Exception:
            break

def _parse_stream_entry(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expect producers to publish either "json" field with JSON or plain fields.
    """
    if "json" in fields:
        try:
            return json.loads(fields["json"])
        except Exception:
            pass
    # coerce numerics when possible
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, (int, float)):
            out[k] = v
            continue
        s = str(v)
        try:
            if s.isdigit():
                out[k] = int(s)
            else:
                out[k] = float(s)
        except Exception:
            out[k] = s
    return out

async def _ensure_redis():
    global redis
    if not HAVE_REDIS:
        return None
    if redis:
        try:
            await redis.ping()
            return redis
        except Exception:
            pass
    # create new
    try:
        redis = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
        await redis.ping() # type: ignore
        return redis
    except Exception:
        return None

async def _consume_streams():
    """
    Reads from Redis Streams listed in TOPIC_STREAMS and broadcasts to subscribers
    under message["topic"].
    """
    global running
    while running:
        r = await _ensure_redis()
        if not r:
            await asyncio.sleep(1.0)
            continue
        try:
            resp = await r.xread(last_ids, count=MAX_READ_BATCH, block=XREAD_BLOCK_MS)  # type: ignore
        except Exception:
            # redis hiccup; retry
            await asyncio.sleep(0.25)
            continue

        if not resp:
            continue

        for stream_name, entries in resp:
            # advance last_id for this stream
            last_ids[stream_name] = entries[-1][0]
            # find topic string by stream mapping
            topic = None
            for t, s in TOPIC_STREAMS.items():
                if s == stream_name:
                    topic = t
                    break
            if not topic:
                continue

            # dispatch each message to subscribers of topic
            if not topic_to_clients.get(topic):
                continue

            for _id, fields in entries:
                payload = _parse_stream_entry(fields)
                # enforce topic & ts
                if "topic" not in payload:
                    payload["topic"] = topic
                if "ts" not in payload and "ts_ms" not in payload:
                    payload["ts_ms"] = int(time.time() * 1000)
                # fan out
                msg = {"type":"data","topic":topic,"data":payload}
                # Copy the set to avoid mutation during iteration
                targets = list(topic_to_clients[topic])
                for cli in targets:
                    await _send_safe(cli, msg)

# ---------- Routes ----------
@app.get("/health")
async def health():
    ok = True
    r_ok = False
    if HAVE_REDIS:
        try:
            r = await _ensure_redis()
            r_ok = bool(r)
        except Exception:
            r_ok = False
    return JSONResponse({"ok": ok, "redis": r_ok, "topics": list(TOPIC_STREAMS.keys())})

@app.post("/publish/{topic}")
async def publish(topic: str, req: Request, _: Any = Depends(require_token)):
    """
    Lightweight publisher to a stream for dev/testing:
    body can be a JSON object; we wrap in {"json": "..."} and XADD.
    """
    topic = topic.strip().lower()
    if topic not in TOPIC_STREAMS:
        raise HTTPException(404, f"unknown topic '{topic}'")
    body = await req.body()
    try:
        obj = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        raise HTTPException(400, "invalid JSON body")

    r = await _ensure_redis()
    if not r:
        raise HTTPException(503, "redis unavailable")

    obj.setdefault("topic", topic)
    obj.setdefault("ts_ms", int(time.time()*1000))
    try:
        await r.xadd(TOPIC_STREAMS[topic], {"json": json.dumps(obj, ensure_ascii=False)}, maxlen=STREAM_MAXLEN, approximate=True)  # type: ignore
    except Exception as e:
        raise HTTPException(500, f"xadd error: {e}")
    return JSONResponse({"ok": True})

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_require_token(ws)
    await ws.accept()
    cli = Client(ws=ws, topics=set(), q=asyncio.Queue(maxsize=MAX_QUEUE))
    clients.add(cli)

    # per-client sender + heartbeat
    sender = asyncio.create_task(_broadcaster(cli))
    pinger = asyncio.create_task(_heartbeat(ws))

    # send hello
    await _send_safe(cli, {"type":"hello","topics": list(TOPIC_STREAMS.keys()), "ts_ms": int(time.time()*1000)})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await _send_safe(cli, {"type":"error","err":"invalid_json"})
                continue

            mtype = str(msg.get("type") or "")
            if mtype == "subscribe":
                # {"type":"subscribe","topics":["candles","orderbook"]}
                req_topics = msg.get("topics") or []
                added = []
                for t in req_topics:
                    t = str(t).lower()
                    if t in TOPIC_STREAMS:
                        if t not in cli.topics:
                            cli.topics.add(t)
                            topic_to_clients[t].add(cli)
                            added.append(t)
                await _send_safe(cli, {"type":"subscribed","topics":sorted(list(cli.topics)), "added":added})
            elif mtype == "unsubscribe":
                req_topics = msg.get("topics") or []
                removed = []
                for t in req_topics:
                    t = str(t).lower()
                    if t in cli.topics:
                        cli.topics.remove(t)
                        topic_to_clients[t].discard(cli)
                        removed.append(t)
                await _send_safe(cli, {"type":"unsubscribed","topics":sorted(list(cli.topics)), "removed":removed})
            elif mtype == "echo":
                await _send_safe(cli, {"type":"echo","data": msg.get("data")})
            else:
                await _send_safe(cli, {"type":"error","err":"unknown_type","got":mtype})

    except WebSocketDisconnect:
        pass
    except Exception:
        # log server-side, but keep client clean
        traceback.print_exc()
    finally:
        # cleanup
        for t in list(cli.topics):
            topic_to_clients[t].discard(cli)
        clients.discard(cli)
        try:
            sender.cancel()
        except Exception:
            pass
        try:
            pinger.cancel()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def on_start():
    global consumer_task, running
    running = True
    if HAVE_REDIS:
        await _ensure_redis()
    # Start the background consumer
    loop = asyncio.get_running_loop()
    consumer_task = loop.create_task(_consume_streams())

@app.on_event("shutdown")
async def on_stop():
    global consumer_task, running, redis
    running = False
    if consumer_task:
        try:
            consumer_task.cancel()
        except Exception:
            pass
    if redis:
        try:
            await redis.close()
        except Exception:
            pass