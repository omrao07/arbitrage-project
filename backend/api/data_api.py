# backend/api/data_api.py
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# ---------- Simple token auth ----------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(x_api_key: Optional[str] = Depends(API_KEY_HEADER)):
    expected = os.getenv("DATA_API_KEY", "").strip()
    if not expected:
        # no key set -> open (dev mode)
        return None
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ---------- In-memory bus (updated by your loop) ----------
class DataBus:
    """
    Thread/async-safe in-memory store the trading loop updates.
    Call these set_* methods from your loop/orchestrator after each bar.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self.account: Dict[str, Any] = {}
        self.positions: Dict[str, Dict[str, float]] = {}     # {symbol: {"qty":..., "avg_price":...}}
        self.prices: Dict[str, float] = {}                   # last marks
        self.pnl_snapshot: Dict[str, Any] = {}               # from PnLAttributor.snapshot()
        self.risk_snapshot: Dict[str, Any] = {}              # from RiskMetrics.snapshot()
        self.tca_snapshot: Dict[str, Any] = {}               # from TCA.snapshot()
        self.news_events: List[Dict[str, Any]] = []          # recent normalized NewsEvent dicts

        # live event stream for websockets (fan-out)
        self._subscribers: List[asyncio.Queue] = []

    async def set_account(self, account: Dict[str, Any]):
        async with self._lock:
            self.account = account

    async def set_positions(self, positions: Dict[str, Dict[str, float]]):
        async with self._lock:
            self.positions = positions

    async def set_prices(self, prices: Dict[str, float]):
        async with self._lock:
            self.prices.update(prices)

    async def set_pnl(self, pnl_snapshot: Dict[str, Any]):
        async with self._lock:
            self.pnl_snapshot = pnl_snapshot

    async def set_risk(self, risk_snapshot: Dict[str, Any]):
        async with self._lock:
            self.risk_snapshot = risk_snapshot

    async def set_tca(self, tca_snapshot: Dict[str, Any]):
        async with self._lock:
            self.tca_snapshot = tca_snapshot

    async def push_news(self, events: List[Dict[str, Any]], keep_last: int = 200):
        async with self._lock:
            self.news_events.extend(events)
            if len(self.news_events) > keep_last:
                self.news_events = self.news_events[-keep_last:]

    # ---- websocket pub/sub ----
    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue):
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def publish_event(self, kind: str, payload: Dict[str, Any]):
        """
        Called by your loop to broadcast live updates to all connected websockets.
        kind: "prices" | "pnl" | "risk" | "tca" | "news" | "positions" | "account"
        """
        msg = {"ts": time.time(), "type": kind, "data": payload}
        for q in list(self._subscribers):
            # don't await put(); drop if queue is too full to avoid backpressure
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

BUS = DataBus()

# ---------- Pydantic response models (lightweight) ----------
class AccountModel(BaseModel):
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"

class PositionModel(BaseModel):
    symbol: str
    qty: float
    avg_price: float

class PriceModel(BaseModel):
    symbol: str
    price: float

class NewsItem(BaseModel):
    source: str
    symbol: Optional[str] = None
    headline: str
    url: str
    published_at: float
    score: Optional[float] = None  # optional sentiment in [-1,1]

# ---------- FastAPI setup ----------
app = FastAPI(title="Trading Data API", version="1.0.0")

# CORS for dashboard dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DATA_API_CORS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/health")
async def health():
    return {"ok": True, "time": time.time()}

# ---------- Account / Positions / Prices ----------
@app.get("/account", response_model=AccountModel)
async def get_account(_: Any = Depends(get_api_key)):
    if not BUS.account:
        return AccountModel(equity=0.0, cash=0.0, buying_power=0.0, currency="USD")
    return BUS.account

@app.get("/positions")
async def get_positions(_: Any = Depends(get_api_key)):
    return BUS.positions

@app.get("/prices")
async def get_prices(symbols: Optional[str] = Query(default=None, description="Comma-separated symbols"),
                     _: Any = Depends(get_api_key)):
    if not symbols:
        return BUS.prices
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    return {s: BUS.prices.get(s) for s in syms}

# ---------- Analytics: PnL / Risk / TCA ----------
@app.get("/pnl")
async def get_pnl(_: Any = Depends(get_api_key)):
    return BUS.pnl_snapshot

@app.get("/risk")
async def get_risk(_: Any = Depends(get_api_key)):
    return BUS.risk_snapshot

@app.get("/tca")
async def get_tca(_: Any = Depends(get_api_key)):
    return BUS.tca_snapshot

# ---------- News ----------
@app.get("/news", response_model=List[NewsItem])
async def get_news(limit: int = 50, _: Any = Depends(get_api_key)):
    items = BUS.news_events[-limit:] if limit > 0 else BUS.news_events
    # ensure shape matches NewsItem
    out: List[Dict[str, Any]] = []
    for e in items:
        out.append({
            "source": e.get("source", ""),
            "symbol": e.get("symbol"),
            "headline": e.get("headline", ""),
            "url": e.get("url", ""),
            "published_at": float(e.get("published_at", time.time())),
            "score": e.get("score"),
        })
    return out

# ---------- WebSocket (live stream) ----------
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    q = await BUS.subscribe()
    try:
        while True:
            # send next event
            msg = await q.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        await BUS.unsubscribe(q)

# ---------- Helpers for your loop to update the bus ----------
# Import and use these from your orchestrator/loop after each bar.

async def update_account(equity: float, cash: float, buying_power: float, currency: str = "USD"):
    payload = {"equity": float(equity), "cash": float(cash), "buying_power": float(buying_power), "currency": currency}
    await BUS.set_account(payload)
    await BUS.publish_event("account", payload)

async def update_positions(positions: Dict[str, Dict[str, float]]):
    await BUS.set_positions(positions)
    await BUS.publish_event("positions", positions)

async def update_prices(prices: Dict[str, float]):
    await BUS.set_prices(prices)
    await BUS.publish_event("prices", prices)

async def update_pnl(pnl_snapshot: Dict[str, Any]):
    await BUS.set_pnl(pnl_snapshot)
    await BUS.publish_event("pnl", pnl_snapshot)

async def update_risk(risk_snapshot: Dict[str, Any]):
    await BUS.set_risk(risk_snapshot)
    await BUS.publish_event("risk", risk_snapshot)

async def update_tca(tca_snapshot: Dict[str, Any]):
    await BUS.set_tca(tca_snapshot)
    await BUS.publish_event("tca", tca_snapshot)

async def append_news(news_events: List[Dict[str, Any]]):
    await BUS.push_news(news_events)
    for e in news_events:
        await BUS.publish_event("news", e)

# ---------- Dev server ----------
# Run with:  uvicorn backend.api.data_api:app --reload --port 8000