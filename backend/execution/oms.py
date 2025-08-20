# backend/execution/oms.py
"""
Order Management System (OMS)
- Consumes approved orders from Redis Stream 'orders'
- PAPER mode: simulate fills at last cached price
- BROKER mode: route to exchange adapters (stubs provided)
- Publishes fills to STREAM_FILLS and CHAN_ORDERS for UI
- Maintains positions, PnL, and exposure in Redis

Env (optional):
  OMS_MODE=paper|broker        (default: paper)
  EXEC_SLIPPAGE_BPS=0
  EXEC_FEES_BPS=0
  RISK_MAX_GROSS_USD=100000
  RISK_MAX_POS_PER_STRAT_USD=25000
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import redis

from backend.bus.streams import (
    consume_stream,
    publish_stream,
    publish_pubsub,
    hgetall,
    hset,
    get as kv_get,
    set as kv_set,
    STREAM_ORDERS,
    STREAM_FILLS,
    CHAN_ORDERS,
)

# ---------- Config ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
OMS_MODE   = os.getenv("OMS_MODE", "paper").lower().strip()  # 'paper' or 'broker'

SLIPPAGE_BPS = float(os.getenv("EXEC_SLIPPAGE_BPS", "0"))
FEES_BPS     = float(os.getenv("EXEC_FEES_BPS", "0"))

RISK_MAX_GROSS_USD        = float(os.getenv("RISK_MAX_GROSS_USD", "100000"))
RISK_MAX_POS_PER_STRAT_USD= float(os.getenv("RISK_MAX_POS_PER_STRAT_USD", "25000"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ---------- Helpers ----------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _last_price(symbol: str) -> Optional[float]:
    raw = r.hget("last_price", symbol.upper())
    if not raw: return None
    try:
        return float(json.loads(raw)["price"])
    except Exception:
        return None

def _gross_exposure_usd() -> float:
    v = kv_get("portfolio:gross_usd", "0")
    try: return float(v if isinstance(v, str) else json.loads(v).get("usd", 0.0))
    except Exception: return 0.0

def _set_gross_exposure_usd(x: float) -> None:
    kv_set("portfolio:gross_usd", {"usd": float(x)})

def _bump_usage(strategy: str, symbol: str, usd: float) -> None:
    r.hincrbyfloat("risk:used_by_strategy", strategy, usd)
    r.hincrbyfloat("risk:used_by_symbol", symbol, usd)

def _apply_slippage_fees(px: float, side: str) -> float:
    p = px
    if SLIPPAGE_BPS:
        adj = px * SLIPPAGE_BPS / 10000.0
        p = p + (adj if side == "buy" else -adj)
    if FEES_BPS:
        fee = px * FEES_BPS / 10000.0
        p = p + (fee if side == "buy" else -fee)
    return p

# ---------- Position & PnL ----------
def _load_pos(symbol: str) -> Dict:
    raw = r.hget("positions", symbol)
    if not raw: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}
    try: return json.loads(raw)
    except Exception: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}

def _save_pos(p: Dict) -> None:
    r.hset("positions", p["symbol"], json.dumps(p))

def _load_pos_strategy(symbol: str, strategy: str) -> Dict:
    raw = r.hget(f"positions:by_strategy:{strategy}", symbol)
    if not raw: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}
    try: return json.loads(raw)
    except Exception: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}

def _save_pos_strategy(strategy: str, p: Dict) -> None:
    r.hset(f"positions:by_strategy:{strategy}", p["symbol"], json.dumps(p))

def _update_position(symbol: str, side: str, qty: float, price: float) -> float:
    """Update aggregate position, return realized PnL delta."""
    pos = _load_pos(symbol)
    realized = 0.0
    if side == "buy":
        new_qty = pos["qty"] + qty
        if pos["qty"] <= 0:
            pos["avg_price"] = price if new_qty != 0 else 0.0
        else:
            pos["avg_price"] = (pos["avg_price"]*pos["qty"] + price*qty) / new_qty
        pos["qty"] = new_qty
    else:  # sell
        sell_qty = qty
        if pos["qty"] > 0:
            close = min(pos["qty"], sell_qty)
            realized += (price - pos["avg_price"]) * close
            pos["qty"] -= close
            sell_qty -= close
        if sell_qty > 0:
            pos["avg_price"] = price
            pos["qty"] -= sell_qty
    pos["realized_pnl"] = pos.get("realized_pnl", 0.0) + realized
    _save_pos(pos)
    return realized

def _update_position_strategy(strategy: str, symbol: str, side: str, qty: float, price: float) -> float:
    pos = _load_pos_strategy(symbol, strategy)
    realized = 0.0
    if side == "buy":
        new_qty = pos["qty"] + qty
        if pos["qty"] <= 0:
            pos["avg_price"] = price if new_qty != 0 else 0.0
        else:
            pos["avg_price"] = (pos["avg_price"]*pos["qty"] + price*qty) / new_qty
        pos["qty"] = new_qty
    else:
        sell_qty = qty
        if pos["qty"] > 0:
            close = min(pos["qty"], sell_qty)
            realized += (price - pos["avg_price"]) * close
            pos["qty"] -= close
            sell_qty -= close
        if sell_qty > 0:
            pos["avg_price"] = price
            pos["qty"] -= sell_qty
    pos["realized_pnl"] = pos.get("realized_pnl", 0.0) + realized
    _save_pos_strategy(strategy, pos)
    return realized

def _unrealized_total() -> float:
    total = 0.0
    for sym, raw in hgetall("positions").items():
        try:
            p = json.loads(raw)
            lp = _last_price(sym)
            if lp is None: continue
            total += (lp - float(p.get("avg_price", 0.0))) * float(p.get("qty", 0.0))
        except Exception:
            continue
    return total

def _update_total_pnl() -> None:
    realized = 0.0
    for _, raw in hgetall("positions").items():
        try: realized += float(json.loads(raw).get("realized_pnl", 0.0))
        except Exception: continue
    unreal = _unrealized_total()
    kv_set("pnl", {"realized": realized, "unrealized": unreal, "total": realized + unreal})

# ---------- Broker Adapters (stubs for later) ----------
class BrokerBase:
    def place_order(self, order: Dict) -> Dict:
        raise NotImplementedError

class PaperBroker(BrokerBase):
    """Immediate fill at last price (after slippage/fees)."""
    def place_order(self, order: Dict) -> Dict:
        symbol   = str(order.get("symbol")).upper()
        side     = str(order.get("side")).lower()
        qty      = float(order.get("qty", 0.0))
        strategy = str(order.get("strategy", ""))
        lp = _last_price(symbol)
        if lp is None:
            return {"status": "reject", "reason": "no_market_price", **order}
        exec_px = _apply_slippage_fees(lp, side)
        notional_abs = abs(exec_px * qty)

        # update usage and gross exposure
        _bump_usage(strategy, symbol, notional_abs)
        _set_gross_exposure_usd(_gross_exposure_usd() + notional_abs)

        # positions & pnl
        _ = _update_position_strategy(strategy, symbol, side, qty, exec_px)
        realized_delta = _update_position(symbol, side, qty, exec_px)
        _update_total_pnl()

        fill = {
            "ts_ms": _now_ms(),
            "order_id": order.get("order_id") or f"{strategy}:{symbol}:{_now_ms()}",
            "symbol": symbol, "side": side, "qty": qty, "price": exec_px,
            "strategy": strategy, "status": "filled", "venue": order.get("venue"),
            "realized_delta": realized_delta,
        }
        return {"status": "filled", "fill": fill}

class AlpacaBroker(BrokerBase):
    def place_order(self, order: Dict) -> Dict:
        # TODO: implement real POST /v2/orders using ALPACA_* envs
        return {"status": "reject", "reason": "alpaca_not_configured", **order}

class OandaBroker(BrokerBase):
    def place_order(self, order: Dict) -> Dict:
        # TODO: implement real /v3/accounts/<id>/orders using OANDA_* envs
        return {"status": "reject", "reason": "oanda_not_configured", **order}

class ZerodhaBroker(BrokerBase):
    def place_order(self, order: Dict) -> Dict:
        # TODO: implement real Kite order using ZERODHA_* envs
        return {"status": "reject", "reason": "zerodha_not_configured", **order}

class IBKRBroker(BrokerBase):
    def place_order(self, order: Dict) -> Dict:
        # TODO: implement IBKR TWS/Gateway place order
        return {"status": "reject", "reason": "ibkr_not_configured", **order}

class BinanceBroker(BrokerBase):
    def place_order(self, order: Dict) -> Dict:
        # Optional: place account orders; for now leave unimplemented
        return {"status": "reject", "reason": "binance_trading_not_configured", **order}

# ---------- Router ----------
def _select_broker(order: Dict) -> BrokerBase:
    if OMS_MODE == "broker":
        # Basic routing by region/venue; customize as needed
        region = (order.get("region") or "").upper()
        venue  = (order.get("venue") or "").upper()
        if venue == "ALPACA" or region == "US":
            return AlpacaBroker()
        if venue in ("OANDA",) or region == "FX":
            return OandaBroker()
        if venue in ("ZERODHA",) or region == "IN":
            return ZerodhaBroker()
        if venue in ("IBKR",) or region in ("EU", "JP", "CNHK"):
            return IBKRBroker()
        if venue in ("BINANCE", "BYBIT", "COINBASE") or region == "CRYPTO":
            return BinanceBroker()
    # default
    return PaperBroker()

# ---------- Main loop ----------
def _emit_fill_and_notify(fill: Dict, event: str = "fill") -> None:
    publish_stream(STREAM_FILLS, fill)
    publish_pubsub(CHAN_ORDERS, {"event": event, **fill})

def run() -> None:
    kv_set("oms:alive", {"ts": _now_ms(), "mode": OMS_MODE})
    for _, order in consume_stream(STREAM_ORDERS, start_id="$", block_ms=1000, count=200):
        try:
            if isinstance(order, str):
                order = json.loads(order)

            broker = _select_broker(order)
            res = broker.place_order(order)

            status = res.get("status")
            if status == "filled":
                _emit_fill_and_notify(res["fill"], event="fill")
            else:
                publish_pubsub(CHAN_ORDERS, {"event": "reject", **res})
        except Exception as e:
            publish_pubsub(CHAN_ORDERS, {"event": "error", "component": "oms", "error": str(e)})

if __name__ == "__main__":
    run()