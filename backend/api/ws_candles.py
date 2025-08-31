# backend/api/ws_candles.py
from __future__ import annotations

import os
import time
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException
from pydantic import BaseModel

# ---------- Optional deps ----------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

router = APIRouter()

# ---------- Env / defaults ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# incoming ticks stream (if you publish ticks to Redis)
TICKS_STREAM = os.getenv("TICKS_STREAM", "ticks.raw")            # fields: json={ts, symbol, price, size, venue}
# outgoing candles stream (for persistence & other services)
CANDLES_STREAM = os.getenv("CANDLES_STREAM", "candles.agg")      # fields: json={t,o,h,l,c,v,tf,symbol}
MAXLEN = int(os.getenv("CANDLES_MAXLEN", "20000"))

# if you already have a separate candle aggregator emitting bars, set:
EXTERNAL_CANDLES_STREAM = os.getenv("EXTERNAL_CANDLES_STREAM", "")  # e.g., "candles.external"

# supported timeframes in seconds
TF_SEC: Dict[str, int] = {"1m": 60, "5m": 300, "15m": 900}

# ---------- Models ----------
class Candle(BaseModel):
    t: int           # bar start (epoch seconds)
    o: float
    h: float
    l: float
    c: float
    v: float
    tf: str
    symbol: str

class CandleHistoryRequest(BaseModel):
    symbol: str
    tf: str = "1m"
    limit: int = 300  # number of most-recent bars

# ---------- Helpers ----------
def bucket_start(ts_ms: int, tf: str) -> int:
    step = TF_SEC.get(tf, 60)
    return (ts_ms // 1000) // step * step

@dataclass
class _Bar:
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float
    tf: str
    symbol: str
    def to_dict(self) -> Dict[str, Any]:
        return {"t": self.t, "o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v, "tf": self.tf, "symbol": self.symbol}

# in-memory ring per (symbol, tf)
class _Ring:
    def __init__(self, cap: int = 5000):
        self.cap = cap
        self.buf: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    def append(self, symbol: str, tf: str, bar: Dict[str, Any]):
        key = (symbol, tf)
        self.buf[key].append(bar)
        if len(self.buf[key]) > self.cap:
            self.buf[key] = self.buf[key][-self.cap:]

    def last(self, symbol: str, tf: str, n: int) -> List[Dict[str, Any]]:
        key = (symbol, tf)
        arr = self.buf.get(key, [])
        return arr[-n:] if n > 0 else []

_RING = _Ring(cap=10000)

# redis dep
async def get_redis() -> Optional[AsyncRedis]: # type: ignore
    if not USE_REDIS:
        return None
    try:
        r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        await r.ping()
        return r
    except Exception:
        return None

async def _xadd(r: AsyncRedis, stream: str, payload: Dict[str, Any]) -> None: # type: ignore
    await r.xadd(stream, {"json": json.dumps(payload)}, maxlen=MAXLEN, approximate=True)

async def _read_recent_stream(r: AsyncRedis, stream: str, count: int) -> List[Dict[str, Any]]: # type: ignore
    try:
        entries = await r.xrevrange(stream, count=count)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for _id, fields in reversed(entries):
        try:
            out.append(json.loads(fields.get("json", "{}")))
        except Exception:
            continue
    return out

# ---------- REST: history endpoint ----------
@router.get("/candles", response_model=List[Candle])
async def get_candles(
    symbol: str = Query(..., description="Ticker symbol"),
    tf: str = Query("1m", description="1m|5m|15m"),
    limit: int = Query(300, ge=1, le=2000),
    r: Optional[AsyncRedis] = Depends(get_redis), # type: ignore
) -> List[Candle]:
    symbol = symbol.upper()
    if tf not in TF_SEC:
        raise HTTPException(status_code=400, detail=f"Unsupported tf {tf}. Allowed: {list(TF_SEC)}")
    if r and CANDLES_STREAM:
        rows = await _read_recent_stream(r, f"{CANDLES_STREAM}:{symbol}:{tf}", limit)
        if rows:
            return [Candle(**x) for x in rows[-limit:]]
    # fallback to in-memory
    rows = _RING.last(symbol, tf, limit)
    return [Candle(**x) for x in rows]

# ---------- WebSocket hub ----------
class _Hub:
    def __init__(self):
        self.clients: List[Tuple[WebSocket, str, str]] = []  # (ws, symbol, tf)
    async def connect(self, ws: WebSocket, symbol: str, tf: str):
        await ws.accept()
        self.clients.append((ws, symbol, tf))
    def disconnect(self, ws: WebSocket):
        self.clients = [(w,s,t) for (w,s,t) in self.clients if w is not ws]
    async def send(self, symbol: str, tf: str, bar: Dict[str, Any]):
        dead = []
        for (ws, s, t) in self.clients:
            if s == symbol and t == tf:
                try:
                    await ws.send_json(bar)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

_HUB = _Hub()

# ---------- Aggregation core (ticks -> bars) ----------
class _Aggregator:
    """Per-connection lightweight aggregator if you subscribe to tick stream here.
       If you already have backend/ingestion/adapters/candle_aggregator.py emitting bars,
       set EXTERNAL_CANDLES_STREAM and we'll just forward those.
    """
    def __init__(self, symbol: str, tf: str):
        self.symbol = symbol
        self.tf = tf
        self.current: Optional[_Bar] = None

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # tick: {ts(ms), symbol, price, size}
        try:
            sym = (tick.get("symbol") or "").upper()
            if sym != self.symbol:
                return None
            ts_ms = int(tick.get("ts") or tick.get("ts_ms")) # type: ignore
            px = float(tick["price"])
            sz = float(tick.get("size", 0.0))
        except Exception:
            return None

        t0 = bucket_start(ts_ms, self.tf)
        if self.current is None or self.current.t != t0:
            # flush prior bar if any
            if self.current is not None:
                bar = self.current.to_dict()
                self.current = _Bar(t=t0, o=px, h=px, l=px, c=px, v=sz, tf=self.tf, symbol=self.symbol)
                return bar
            # start a new one
            self.current = _Bar(t=t0, o=px, h=px, l=px, c=px, v=sz, tf=self.tf, symbol=self.symbol)
            return None

        # update current
        self.current.h = max(self.current.h, px)
        self.current.l = min(self.current.l, px)
        self.current.c = px
        self.current.v += sz
        return None

    def force_flush(self) -> Optional[Dict[str, Any]]:
        if self.current is None:
            return None
        bar = self.current.to_dict()
        self.current = None
        return bar

# ---------- WebSocket: /ws/candles ----------
@router.websocket("/ws/candles")
async def ws_candles(
    ws: WebSocket,
    symbol: str = Query(...),
    tf: str = Query("1m")
):
    symbol = symbol.upper()
    if tf not in TF_SEC:
        await ws.close(code=1003)
        return

    await _HUB.connect(ws, symbol, tf)

    # initial snapshot to client
    for it in _RING.last(symbol, tf, 200):
        try:
            await ws.send_json(it)
        except Exception:
            pass

    r: Optional[AsyncRedis] = None # type: ignore
    if USE_REDIS:
        try:
            r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
            await r.ping() # type: ignore
        except Exception:
            r = None

    # If we already have a candle stream, just tail it; else, tail ticks and aggregate.
    use_external_bars = bool(EXTERNAL_CANDLES_STREAM)
    last_id_ticks = "$"
    last_id_bars = "$"
    agg = _Aggregator(symbol, tf)

    try:
        while True:
            if r:
                if use_external_bars:
                    # consume external prebuilt candles
                    stream = f"{EXTERNAL_CANDLES_STREAM}:{symbol}:{tf}"
                    resp = await r.xread({stream: last_id_bars}, count=200, block=5000)
                    if resp:
                        _, entries = resp[0]
                        for _id, fields in entries:
                            last_id_bars = _id
                            bar = json.loads(fields.get("json", "{}"))
                            # mirror to memory
                            _RING.append(symbol, tf, bar)
                            # fanout
                            await _HUB.send(symbol, tf, bar)
                else:
                    # aggregate from ticks
                    resp = await r.xread({TICKS_STREAM: last_id_ticks}, count=500, block=5000)
                    if resp:
                        _, entries = resp[0]
                        to_flush: List[Dict[str, Any]] = []
                        for _id, fields in entries:
                            last_id_ticks = _id
                            tick = json.loads(fields.get("json", "{}"))
                            flushed = agg.on_tick(tick)
                            if flushed:
                                to_flush.append(flushed)
                        # flush completed bars
                        for bar in to_flush:
                            _RING.append(symbol, tf, bar)
                            await _HUB.send(symbol, tf, bar)
                            # persist to a per-symbol stream for fast history
                            try:
                                await _xadd(r, f"{CANDLES_STREAM}:{symbol}:{tf}", bar)
                            except Exception:
                                pass
            else:
                # no redis: keep the socket alive with heartbeats
                await ws.send_json({"type": "heartbeat", "ts_ms": int(time.time() * 1000)})
                await ws.receive_text()
    except WebSocketDisconnect:
        _HUB.disconnect(ws)
    except Exception:
        _HUB.disconnect(ws)