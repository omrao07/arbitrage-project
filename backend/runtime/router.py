# backend/api/router.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# --- your modules ---
from backend.bus.streams import publish_stream, hset, hgetall
from backend.execution.order_store import OrderStore # type: ignore
from backend.execution.pricer import Position as PricerPos, Quote, quotes_from_last, mark_portfolio, mtm_dicts # type: ignore
from backend.engine.models import Side, OrderType, TIF # type: ignore

# ------------------- config / streams -------------------
ORDERS_IN     = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")
ORDERS_ACKS   = os.getenv("ORDERS_ACKS_STREAM", "orders.acks")
FILLS_STREAM  = os.getenv("FILLS_STREAM", "fills")
PRICES_HASH   = os.getenv("PRICES_HASH", "px:last")
HEALTH_KEY    = os.getenv("HEALTH_HEARTBEAT_KEY", "engine:hb")
POS_REDIS_KEY = os.getenv("REDIS_POS_KEY", "pos:live")
BASE_CCY      = os.getenv("BASE_CCY", "USD")

DB_PATH       = os.getenv("ORDER_STORE_DB", "runtime/order_store.db")

# ------------------- DI / singletons --------------------
def get_store() -> OrderStore:
    # one SQLite file; FastAPI creates a new instance per request (thread-safe inside)
    return OrderStore(db_path=DB_PATH)

router = APIRouter(prefix="/api", tags=["api"])

# ------------------- pydantic models --------------------

class OrderIn(BaseModel):
    symbol: str = Field(..., description="e.g., RELIANCE.NS")
    side: Side
    qty: float = Field(..., gt=0)
    typ: OrderType = OrderType.MARKET
    limit_price: Optional[float] = Field(None, gt=0)
    tif: Optional[TIF] = None
    strategy: Optional[str] = None
    region: Optional[str] = None
    mark_price: Optional[float] = None
    client_order_id: Optional[str] = None

class OrderAckOut(BaseModel):
    ok: bool
    order_id: Optional[str]
    reason: Optional[str] = None

class HealthOut(BaseModel):
    last_tick_ms: Optional[int] = None
    last_news_ms: Optional[int] = None
    last_risk_ms: Optional[int] = None
    last_oms_ms: Optional[int] = None
    manager_alive_ms: Optional[int] = None
    now_ms: int

class PositionOut(BaseModel):
    symbol: str
    qty: float
    avg_price: float

class PnLDayOut(BaseModel):
    realized: float
    fees: float
    pnl: float

# ------------------- endpoints ---------------------------

@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    hb = hgetall(HEALTH_KEY) or {}
    def _g(k: str) -> Optional[int]:
        try:
            v = hb.get(k)
            return int(v) if v is not None else None
        except Exception:
            return None
    return HealthOut(
        last_tick_ms=_g("last_tick_ms"),
        last_news_ms=_g("last_news_ms"),
        last_risk_ms=_g("last_risk_ms"),
        last_oms_ms=_g("last_oms_ms"),
        manager_alive_ms=_g("manager_alive_ms"),
        now_ms=int(time.time() * 1000),
    )

@router.post("/orders/place", response_model=OrderAckOut)
def place_order(order: OrderIn):
    if order.typ == OrderType.LIMIT and not order.limit_price:
        raise HTTPException(400, "limit_price required for LIMIT order")

    payload: Dict[str, Any] = dict(
        ts_ms=int(time.time() * 1000),
        symbol=order.symbol.upper(),
        side=order.side.value,
        qty=float(order.qty),
        typ=order.typ.value,
        limit_price=order.limit_price,
        tif=(order.tif.value if order.tif else None),
        strategy=order.strategy,
        region=order.region,
        mark_price=order.mark_price,
        client_order_id=order.client_order_id,
    )
    publish_stream(ORDERS_IN, payload)
    # Synchronous ack isn’t available here; we just echo that it was queued.
    return OrderAckOut(ok=True, order_id=None, reason="QUEUED_TO_RISK")

@router.post("/risk/kill")
def kill_switch(on: bool = Query(True, description="true to halt, false to resume")) -> Dict[str, Any]:
    hset("risk:halt", "flag", "true" if on else "false")
    return {"ok": True, "halt": on}

@router.get("/risk/kill")
def get_kill() -> Dict[str, Any]:
    v = (hget("risk:halt", "flag") or "").lower() == "true" # type: ignore
    return {"halt": v}

@router.get("/orders", response_model=List[Dict[str, Any]])
def list_orders(limit: int = 200, status: Optional[str] = Query(None), store: OrderStore = Depends(get_store)):
    return store.get_orders(limit=limit, status=status)

@router.get("/fills", response_model=List[Dict[str, Any]])
def list_fills(limit: int = 200, store: OrderStore = Depends(get_store)):
    return store.get_fills(limit=limit)

@router.get("/positions", response_model=List[PositionOut])
def positions(store: OrderStore = Depends(get_store)):
    rows = store.get_positions()
    return [PositionOut(**r) for r in rows]

@router.get("/pnl/day", response_model=PnLDayOut)
def pnl_day(store: OrderStore = Depends(get_store)):
    return PnLDayOut(**store.get_pnl_day())

@router.get("/mtm")
def mtm(base_ccy: str = Query(BASE_CCY, description="Report currency"), store: OrderStore = Depends(get_store)):
    # positions from SQLite
    pos_rows = store.get_positions()
    positions = [PricerPos(symbol=r["symbol"], qty=float(r["qty"]), avg_price=float(r["avg_price"]), currency=base_ccy) for r in pos_rows]
    # last prices from Redis
    last_map = {k: float(v) for k, v in (hgetall(PRICES_HASH) or {}).items()}
    quotes: Dict[str, Quote] = quotes_from_last(last_map)
    res = mark_portfolio(positions, quotes, base_ccy=base_ccy, fx_rates=None)
    return mtm_dicts(res)

@router.get("/prices/last")
def prices_last() -> Dict[str, float]:
    h = hgetall(PRICES_HASH) or {}
    out: Dict[str, float] = {}
    for k, v in h.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out

# -------- convenience: set last price manually (dev/testing) ----------
class ManualPriceIn(BaseModel):
    symbol: str
    price: float

@router.post("/prices/set")
def set_price(p: ManualPriceIn) -> Dict[str, Any]:
    hset(PRICES_HASH, p.symbol.upper(), float(p.price))
    return {"ok": True}

# -------- optional: annotate a fee cost into pnl_day -------------------
class FeeIn(BaseModel):
    amount: float = Field(..., gt=0, description="absolute currency (same as base)")
@router.post("/pnl/fee")
def add_fee(f: FeeIn, store: OrderStore = Depends(get_store)) -> Dict[str, Any]:
    store.bump_fees(f.amount)
    return {"ok": True}