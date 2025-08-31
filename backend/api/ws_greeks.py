# backend/api/ws_greeks.py
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException
from pydantic import BaseModel, Field

# ---------- Optional Redis ----------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

router = APIRouter()

# ---------- Env / defaults ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GREEKS_STREAM = os.getenv("GREEKS_STREAM", "options.greeks")   # fields: json={...tick...}
MAXLEN = int(os.getenv("GREEKS_MAXLEN", "20000"))

# ---------- Models ----------
class GreekTick(BaseModel):
    ts: int                                   # ms epoch
    symbol: str                               # underlying symbol (e.g., TSLA)
    expiry: str                               # YYYY-MM-DD
    right: str = Field(..., regex="^[CP]$")   # type: ignore # C or P
    strike: float

    # prices
    mark: Optional[float] = None              # option mark price
    underlying: Optional[float] = None        # spot

    # greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None                # implied vol (decimal, e.g., 0.42)

# ---------- In-memory cache (latest per contract) ----------
Key = Tuple[str, str, str, float]  # (symbol, expiry, right, strike)

@dataclass
class _Chain:
    # latest per contract
    latest: Dict[Key, Dict[str, Any]]
    # index by symbol -> keys
    by_symbol: Dict[str, List[Key]]

    def __init__(self):
        self.latest = {}
        self.by_symbol = {}

    def upsert(self, tick: Dict[str, Any]):
        try:
            symbol = (tick.get("symbol") or "").upper()
            expiry = tick["expiry"]
            right = tick["right"]
            strike = float(tick["strike"])
        except Exception:
            return
        k: Key = (symbol, expiry, right, strike)
        self.latest[k] = tick
        arr = self.by_symbol.setdefault(symbol, [])
        # maintain index once
        if k not in arr:
            arr.append(k)

    def snapshot(
        self,
        symbol: str,
        expiry: Optional[str] = None,
        right: Optional[str] = None,
        strike_min: Optional[float] = None,
        strike_max: Optional[float] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        symbol = symbol.upper()
        keys = self.by_symbol.get(symbol, [])
        out: List[Dict[str, Any]] = []
        for k in keys:
            _sym, _exp, _right, _strike = k
            if expiry and _exp != expiry:
                continue
            if right and _right != right:
                continue
            if strike_min is not None and _strike < strike_min:
                continue
            if strike_max is not None and _strike > strike_max:
                continue
            out.append(self.latest[k])
            if len(out) >= limit:
                break
        # sort by (expiry, right, strike)
        out.sort(key=lambda x: (x.get("expiry",""), x.get("right",""), float(x.get("strike", 0.0))))
        return out

_CHAIN = _Chain()

# ---------- Redis helpers ----------
async def get_redis() -> Optional[AsyncRedis]: # type: ignore
    if not USE_REDIS:
        return None
    try:
        r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        await r.ping()
        return r
    except Exception:
        return None

async def _xread_tail(r: AsyncRedis, stream: str, last_id: str, count: int = 500, block_ms: int = 5000): # type: ignore
    try:
        resp = await r.xread({stream: last_id}, count=count, block=block_ms)
        return resp
    except Exception:
        return None

# ---------- REST: snapshots ----------
@router.get("/greeks/latest", response_model=List[GreekTick])
async def get_greeks_latest(
    symbol: str = Query(...),
    expiry: Optional[str] = None,
    right: Optional[str] = Query(None, regex="^[CP]$"),
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
    limit: int = Query(2000, ge=1, le=5000),
    r: Optional[AsyncRedis] = Depends(get_redis), # type: ignore
) -> List[GreekTick]:
    # We serve from in-memory snapshot (kept fresh by WS consumer / background task).
    # If you want history, add a history endpoint backed by a Redis HASH/Stream.
    rows = _CHAIN.snapshot(symbol, expiry, right, strike_min, strike_max, limit)
    return [GreekTick(**x) for x in rows]

@router.get("/greeks/chain", response_model=List[GreekTick])
async def get_greeks_chain(
    symbol: str = Query(...),
    expiry: Optional[str] = None,
    right: Optional[str] = Query(None, regex="^[CP]$"),
    r: Optional[AsyncRedis] = Depends(get_redis), # type: ignore
) -> List[GreekTick]:
    rows = _CHAIN.snapshot(symbol, expiry, right, None, None, 5000)
    return [GreekTick(**x) for x in rows]

# ---------- WS hub with per-client filters ----------
class _Client:
    def __init__(self, ws: WebSocket, symbol: str, expiry: Optional[str], right: Optional[str],
                 strike_min: Optional[float], strike_max: Optional[float]):
        self.ws = ws
        self.symbol = symbol.upper()
        self.expiry = expiry
        self.right = right
        self.strike_min = strike_min
        self.strike_max = strike_max
    def match(self, t: Dict[str, Any]) -> bool:
        if (t.get("symbol") or "").upper() != self.symbol: return False
        if self.expiry and t.get("expiry") != self.expiry: return False
        if self.right and t.get("right") != self.right: return False
        k = float(t.get("strike", 0.0))
        if self.strike_min is not None and k < self.strike_min: return False
        if self.strike_max is not None and k > self.strike_max: return False
        return True

class _Hub:
    def __init__(self):
        self.clients: List[_Client] = []
    async def connect(self, c: _Client):
        await c.ws.accept()
        self.clients.append(c)
    def disconnect(self, ws: WebSocket):
        self.clients = [c for c in self.clients if c.ws is not ws]
    async def broadcast_match(self, t: Dict[str, Any]):
        dead: List[WebSocket] = []
        for c in self.clients:
            if c.match(t):
                try:
                    await c.ws.send_json(t)
                except Exception:
                    dead.append(c.ws)
        for ws in dead:
            self.disconnect(ws)

_HUB = _Hub()

# ---------- WebSocket: /ws/greeks ----------
@router.websocket("/ws/greeks")
async def ws_greeks(
    ws: WebSocket,
    symbol: str = Query(...),
    expiry: Optional[str] = Query(None),
    right: Optional[str] = Query(None, regex="^[CP]$"),
    strike_min: Optional[float] = Query(None),
    strike_max: Optional[float] = Query(None),
    warmup: int = Query(200, ge=0, le=2000),   # how many cached ticks to send initially
):
    client = _Client(ws, symbol, expiry, right, strike_min, strike_max)
    await _HUB.connect(client)

    # Initial snapshot (latest per matching contract)
    snap = _CHAIN.snapshot(client.symbol, client.expiry, client.right, client.strike_min, client.strike_max, warmup)
    for row in snap:
        try:
            await ws.send_json(row)
        except Exception:
            pass

    r: Optional[AsyncRedis] = None # type: ignore
    if USE_REDIS:
        try:
            r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
            await r.ping() # type: ignore
        except Exception:
            r = None

    last_id = "$"
    try:
        while True:
            if r:
                resp = await _xread_tail(r, GREEKS_STREAM, last_id, count=500, block_ms=5000)
                if not resp:
                    # heartbeat to keep socket healthy when idle
                    await ws.send_json({"type":"heartbeat", "ts": int(time.time()*1000)})
                    continue
                _, entries = resp[0]
                for _id, fields in entries:
                    last_id = _id
                    try:
                        t = json.loads(fields.get("json", "{}"))
                        # normalize fields
                        t["symbol"] = (t.get("symbol") or "").upper()
                        # require core keys
                        if not all(k in t for k in ("symbol","expiry","right","strike")):
                            continue
                        # ensure ts
                        t["ts"] = int(t.get("ts") or t.get("ts_ms") or int(time.time()*1000))
                    except Exception:
                        continue
                    # update cache
                    _CHAIN.upsert(t)
                    # fan out to interested clients
                    await _HUB.broadcast_match(t)
            else:
                # no redis: idle heartbeat; client can poll REST
                await ws.send_json({"type":"heartbeat", "ts": int(time.time()*1000)})
                await ws.receive_text()
    except WebSocketDisconnect:
        _HUB.disconnect(ws)
    except Exception:
        _HUB.disconnect(ws)