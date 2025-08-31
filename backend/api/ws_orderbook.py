# backend/api/ws_orderbook.py
from __future__ import annotations

import os
import json
import time
import bisect
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException

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
ORDERBOOK_STREAM = os.getenv("ORDERBOOK_STREAM", "book.l2")  # you can publish per-symbol as "book.l2:TSLA"
MAXLEN = int(os.getenv("ORDERBOOK_MAXLEN", "5000"))

# ---------- In-memory books ----------
# Each side keeps two parallel arrays (prices, sizes) to maintain sorted levels efficiently.
class _SideBook:
    def __init__(self, side: str):
        self.side = side  # "bids" or "asks"
        self.prices: List[float] = []
        self.sizes: List[float] = []

    def _key(self, px: float) -> float:
        # Bids sorted DESC, Asks sorted ASC
        return -px if self.side == "bids" else px

    def snapshot(self, arr: List[Tuple[float, float]]):
        self.prices = []
        self.sizes = []
        if not arr:
            return
        # normalize + sort
        if self.side == "bids":
            arr = sorted(((float(p), float(s)) for p, s in arr if s > 0), key=lambda x: x[0], reverse=True)
        else:
            arr = sorted(((float(p), float(s)) for p, s in arr if s > 0), key=lambda x: x[0])
        for p, s in arr:
            self.prices.append(p)
            self.sizes.append(s)

    def upsert(self, px: float, sz: float):
        px = float(px); sz = float(sz)
        if sz <= 0:
            # remove level if exists
            i = self._find(px)
            if i is not None:
                del self.prices[i]
                del self.sizes[i]
            return
        # insert/update keeping order
        i = self._find(px)
        if i is not None:
            self.sizes[i] = sz
            return
        # insert new
        key_list = [-p for p in self.prices] if self.side == "bids" else self.prices
        k = -px if self.side == "bids" else px
        idx = bisect.bisect_left(key_list, k)
        self.prices.insert(idx, px)
        self.sizes.insert(idx, sz)

    def _find(self, px: float) -> Optional[int]:
        # binary search by price
        if not self.prices:
            return None
        lo, hi = 0, len(self.prices)-1
        while lo <= hi:
            mid = (lo + hi) // 2
            v = self.prices[mid]
            if v == px: return mid
            if self.side == "bids":
                # descending
                if v < px: hi = mid - 1
                else: lo = mid + 1
            else:
                # ascending
                if v > px: hi = mid - 1
                else: lo = mid + 1
        return None

    def top(self, depth: int) -> List[List[float]]:
        depth = max(1, depth)
        out: List[List[float]] = []
        if self.side == "bids":
            rng = range(0, min(depth, len(self.prices)))
        else:
            rng = range(0, min(depth, len(self.prices)))
        for i in rng:
            out.append([self.prices[i], self.sizes[i]])
        return out


class _Book:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ts: int = 0
        self.bids = _SideBook("bids")
        self.asks = _SideBook("asks")

    def apply_snapshot(self, msg: Dict[str, Any]):
        self.ts = int(msg.get("ts") or msg.get("ts_ms") or int(time.time() * 1000))
        self.bids.snapshot(msg.get("bids") or [])
        self.asks.snapshot(msg.get("asks") or [])

    def apply_delta(self, msg: Dict[str, Any]):
        self.ts = int(msg.get("ts") or msg.get("ts_ms") or int(time.time() * 1000))
        for p, s in (msg.get("bids") or []):
            self.bids.upsert(float(p), float(s))
        for p, s in (msg.get("asks") or []):
            self.asks.upsert(float(p), float(s))

    def top(self, depth: int) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts,
            "bids": self.bids.top(depth),
            "asks": self.asks.top(depth),
        }

# Global book registry
_BOOKS: Dict[str, _Book] = {}

def _get_book(symbol: str) -> _Book:
    sym = symbol.upper()
    if sym not in _BOOKS:
        _BOOKS[sym] = _Book(sym)
    return _BOOKS[sym]

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

async def _xread_tail(r: AsyncRedis, stream: str, last_id: str, count: int = 200, block_ms: int = 5000): # type: ignore
    try:
        return await r.xread({stream: last_id}, count=count, block=block_ms)
    except Exception:
        return None

# ---------- REST: snapshot ----------
@router.get("/orderbook")
async def get_orderbook(
    symbol: str = Query(..., description="Ticker symbol"),
    depth: int = Query(25, ge=1, le=200),
) -> Dict[str, Any]:
    book = _get_book(symbol)
    data = book.top(depth)
    if not data["bids"] and not data["asks"]:
        # likely no data ingested yet
        raise HTTPException(status_code=404, detail=f"No orderbook for {symbol.upper()} (ingestion not started)")
    return data

# ---------- WS hub ----------
class _Hub:
    def __init__(self):
        self.clients: List[Tuple[WebSocket, str, int]] = []  # (ws, symbol, depth)
    async def connect(self, ws: WebSocket, symbol: str, depth: int):
        await ws.accept()
        self.clients.append((ws, symbol, depth))
    def disconnect(self, ws: WebSocket):
        self.clients = [(w,s,d) for (w,s,d) in self.clients if w is not ws]
    async def broadcast_if(self, symbol: str, payload: Dict[str, Any]):
        dead: List[WebSocket] = []
        for (ws, s, d) in self.clients:
            if s == symbol:
                try:
                    # trim to the client depth before sending
                    msg = {
                        "symbol": payload["symbol"],
                        "ts": payload["ts"],
                        "bids": payload["bids"][:d],
                        "asks": payload["asks"][:d],
                    }
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

_HUB = _Hub()

# ---------- WebSocket: /ws/orderbook ----------
@router.websocket("/ws/orderbook")
async def ws_orderbook(
    ws: WebSocket,
    symbol: str = Query(...),
    depth: int = Query(25, ge=1, le=200),
):
    symbol = symbol.upper()
    await _HUB.connect(ws, symbol, depth)

    # Send a warm snapshot if we have something in memory
    init = _get_book(symbol).top(depth)
    if init["bids"] or init["asks"]:
        try:
            await ws.send_json(init)
        except Exception:
            pass

    r: Optional[AsyncRedis] = None # type: ignore
    if USE_REDIS:
        try:
            r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
            await r.ping() # type: ignore
        except Exception:
            r = None

    # If publishers write per-symbol: "book.l2:TSLA"
    # Otherwise, a single stream with mixed symbols is fine; we'll filter.
    stream = f"{ORDERBOOK_STREAM}:{symbol}"
    last_id = "$"

    try:
        while True:
            if r:
                resp = await _xread_tail(r, stream, last_id, count=200, block_ms=5000)
                if not resp:
                    # heartbeat for idle periods
                    await ws.send_json({"type": "heartbeat", "ts": int(time.time()*1000)})
                    continue
                _, entries = resp[0]
                for _id, fields in entries:
                    last_id = _id
                    try:
                        msg = json.loads(fields.get("json", "{}"))
                        # normalize
                        msg["symbol"] = (msg.get("symbol") or symbol).upper()
                        mtype = (msg.get("type") or "delta").lower()
                    except Exception:
                        continue

                    if msg["symbol"] != symbol:
                        # if you're on a mixed stream, ignore different symbols
                        continue

                    book = _get_book(symbol)
                    if mtype == "snapshot":
                        book.apply_snapshot(msg)
                    else:
                        book.apply_delta(msg)

                    await _HUB.broadcast_if(symbol, book.top(depth))
            else:
                # no redis available → keep socket alive
                await ws.send_json({"type": "heartbeat", "ts": int(time.time()*1000)})
                await ws.receive_text()
    except WebSocketDisconnect:
        _HUB.disconnect(ws)
    except Exception:
        _HUB.disconnect(ws)