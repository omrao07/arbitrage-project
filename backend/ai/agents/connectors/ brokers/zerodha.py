# backend/ai/agents/connectors/brokers/zerodha.py
from __future__ import annotations

import os
import time
import uuid
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

# ============================================================
# ENV CONFIG
# ============================================================
Z_API_KEY       = os.getenv("ZERODHA_API_KEY", "")
Z_API_SECRET    = os.getenv("ZERODHA_API_SECRET", "")
Z_ACCESS_TOKEN  = os.getenv("ZERODHA_ACCESS_TOKEN", "")  # Typically retrieved via login flow
Z_USER_ID       = os.getenv("ZERODHA_USER_ID", "")
Z_DEFAULT_EXCH  = os.getenv("ZERODHA_DEFAULT_EXCHANGE", "NSE")      # NSE/BSE
Z_DEFAULT_PROD  = os.getenv("ZERODHA_DEFAULT_PRODUCT", "CNC")       # CNC/MIS/NRML
Z_DEFAULT_VAR   = os.getenv("ZERODHA_DEFAULT_VARIETY", "regular")   # regular/amo/bo/oco (bo/oco require permissions)
Z_TIMEOUT_S     = float(os.getenv("ZERODHA_TIMEOUT_S", "8"))
Z_PAPER_SIM     = os.getenv("ZERODHA_SIMULATE", "false").lower() in ("1", "true", "yes", "y")  # force sim even if kiteconnect exists

# ============================================================
# Optional dependency
# ============================================================
try:
    from kiteconnect import KiteConnect  # type: ignore
    _HAS_KITE = True
except Exception:
    _HAS_KITE = False

# ============================================================
# Public result
# ============================================================
@dataclass
class OrderAck:
    order_id: str
    broker: str = "zerodha"
    status: str = "accepted"  # accepted | sent | filled | cancelled | replaced | rejected
    details: Dict[str, Any] = None # type: ignore

# ============================================================
# Zerodha Client (singleton)
# ============================================================
class _ZerodhaClient:
    def __init__(self):
        self._lock = threading.RLock()
        self._kite: Optional["KiteConnect"] = None
        self._connected = False

        # sim fallback state
        self._sim_orders: Dict[str, Dict[str, Any]] = {}
        self._sim_positions: Dict[str, Dict[str, float]] = {}  # symbol -> {"qty":..., "avg_cost":...}
        self._sim_cash: float = 1_000_000.0
        self._sim_last: Dict[str, float] = {}

    # ------------- lifecycle -------------
    def connect(self) -> bool:
        if Z_PAPER_SIM or not _HAS_KITE:
            self._connected = True
            return True

        with self._lock:
            if self._connected and self._kite is not None:
                return True
            if not Z_API_KEY:
                # cannot auth; fall back to sim
                self._connected = True
                self._kite = None
                return True
            try:
                kite = KiteConnect(api_key=Z_API_KEY)
                if Z_ACCESS_TOKEN:
                    kite.set_access_token(Z_ACCESS_TOKEN)
                    self._kite = kite
                    self._connected = True
                    return True
                # No access token provided → treat as connected to simulator for safety
                self._kite = None
                self._connected = True
                return True
            except Exception:
                self._kite = None
                self._connected = True
                return True

    def is_connected(self) -> bool:
        return bool(self._connected)

    def disconnect(self) -> None:
        with self._lock:
            self._kite = None
            self._connected = False

    # ------------- market data -------------
    def last_price(self, symbol: str, exchange: Optional[str] = None) -> Optional[float]:
        if not self.connect() or self._kite is None:
            return self._sim_last.get(symbol)
        try:
            exch = (exchange or Z_DEFAULT_EXCH).upper()
            ins = f"{exch}:{symbol.upper()}"
            q = self._kite.quote([ins])
            px = q[ins]["last_price"] if ins in q else None
            return float(px) if px is not None else None
        except Exception:
            return None

    # Allow external feed to set synthetic last for sim
    def push_price(self, symbol: str, last: float) -> None:
        self._sim_last[symbol] = float(last)

    # ------------- trading -------------
    def submit_order(self,
                     symbol: str,
                     side: str,
                     qty: float,
                     order_type: str = "market",
                     limit_price: Optional[float] = None,
                     tag: Optional[str] = None,
                     venue: Optional[str] = None,
                     product: Optional[str] = None,
                     variety: Optional[str] = None) -> OrderAck:

        if not self.connect():
            return OrderAck(order_id=f"SIM-{uuid.uuid4().hex[:10]}", status="rejected", details={"reason": "not connected"})

        # Simulator path
        if self._kite is None:
            oid = f"SIM-{uuid.uuid4().hex[:10]}"
            now = int(time.time() * 1000)
            ot = order_type.lower()
            self._sim_orders[oid] = {
                "symbol": symbol, "side": side, "qty": float(qty), "remaining": float(qty),
                "type": ot, "limit": float(limit_price) if limit_price else None,
                "status": "sent", "ts": now, "product": product or Z_DEFAULT_PROD, "variety": variety or Z_DEFAULT_VAR,
                "exchange": venue or Z_DEFAULT_EXCH, "tag": tag
            }
            # naive immediate fill for market / cross for limit
            px = self._sim_last.get(symbol, 100.0)
            if ot == "market" or (ot == "limit" and ((side.lower()=="buy" and px <= (limit_price or px)) or (side.lower()=="sell" and px >= (limit_price or px)))):
                self._sim_fill(oid, px)
            return OrderAck(order_id=oid, status="sent", details={"sim": True})

        # Real Kite order
        try:
            exch = (venue or Z_DEFAULT_EXCH).upper()
            product = (product or Z_DEFAULT_PROD).upper()
            variety = (variety or Z_DEFAULT_VAR).lower()
            tx = "BUY" if side.lower() == "buy" else "SELL"
            ot = "MARKET" if order_type.lower() in ("market","mkt") else "LIMIT"

            params = dict(
                variety=variety,
                exchange=exch,
                tradingsymbol=symbol.upper(),
                transaction_type=tx,
                quantity=int(round(qty)),
                product=product,
                order_type=ot,
            )
            if ot == "LIMIT":
                if limit_price is None:
                    raise ValueError("limit_price required for limit order")
                params["price"] = float(limit_price) # type: ignore

            if tag:
                params["tag"] = str(tag)[:8]  # Kite allows short tag (8 chars)

            resp = self._kite.place_order(**params)  # returns order_id
            oid = str(resp["order_id"])
            return OrderAck(order_id=oid, status="sent", details={"exchange": exch, "product": product, "variety": variety})
        except Exception as e:
            return OrderAck(order_id=f"ERR-{uuid.uuid4().hex[:6]}", status="rejected", details={"error": str(e)})

    def cancel_order(self, order_id: str, *, variety: Optional[str] = None) -> bool:
        if not self.connect():
            return False

        if self._kite is None:
            o = self._sim_orders.get(order_id)
            if not o or o["status"] in ("filled","cancelled"):
                return False
            o["status"] = "cancelled"
            o["remaining"] = 0.0
            return True

        try:
            self._kite.cancel_order(variety=(variety or Z_DEFAULT_VAR).lower(), order_id=order_id)
            return True
        except Exception:
            return False

    def replace_order(self, order_id: str, *,
                      new_qty: Optional[int] = None,
                      new_limit: Optional[float] = None,
                      variety: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        if not self.connect():
            return (False, None)

        if self._kite is None:
            o = self._sim_orders.get(order_id)
            if not o or o["status"] in ("filled","cancelled"):
                return (False, None)
            if new_qty is not None:
                filled = o["qty"] - o["remaining"]
                if new_qty < filled:
                    return (False, None)
                o["qty"] = int(new_qty)
                o["remaining"] = int(new_qty - filled)
            if new_limit is not None and o["type"] == "limit":
                o["limit"] = float(new_limit)
            o["status"] = "replaced"
            return (True, order_id)

        try:
            params = dict(variety=(variety or Z_DEFAULT_VAR).lower(), order_id=order_id)
            if new_qty is not None:
                params["quantity"] = int(new_qty) # type: ignore
            if new_limit is not None:
                params["price"] = float(new_limit) # pyright: ignore[reportArgumentType]
            self._kite.modify_order(**params)
            return (True, order_id)
        except Exception:
            return (False, None)

    # ------------- account/positions -------------
    def account_summary(self) -> Dict[str, Any]:
        if not self.connect():
            return {"error": "not connected"}
        if self._kite is None:
            net = self._sim_cash + sum(self._mtm(sym, p["qty"], p["avg"]) for sym, p in self._sim_positions.items())
            return {"account": Z_USER_ID or "SIM", "cash": self._sim_cash, "net_liq": net, "paper": True}
        try:
            # KiteConnect doesn't expose a single "account summary"; use margins
            m = self._kite.margins("equity")
            return {"account": Z_USER_ID or "KITE", "available": m.get("available", {}), "net": m.get("net", None)}
        except Exception as e:
            return {"error": str(e)}

    def positions(self) -> List[Dict[str, Any]]:
        if not self.connect():
            return []
        if self._kite is None:
            out = []
            for sym, p in self._sim_positions.items():
                out.append({"symbol": sym, "qty": p["qty"], "avg_cost": p["avg"]})
            return out
        try:
            pos = self._kite.positions()
            out = []
            for p in pos.get("net", []):
                out.append({"symbol": p.get("tradingsymbol"), "qty": float(p.get("quantity", 0)), "avg_cost": float(p.get("average_price", 0))})
            return out
        except Exception:
            return []

    # ------------- simulator helpers -------------
    def _sim_fill(self, oid: str, px: float) -> None:
        o = self._sim_orders.get(oid)
        if not o or o["remaining"] <= 0:
            return
        qty = o["remaining"]
        o["remaining"] = 0
        o["status"] = "filled"

        # positions & cash
        sym = o["symbol"]; side = o["side"].lower()
        p = self._sim_positions.get(sym) or {"qty": 0.0, "avg": 0.0}
        if side == "buy":
            new_qty = p["qty"] + qty
            p["avg"] = (p["avg"] * p["qty"] + px * qty) / new_qty if new_qty != 0 else 0.0
            p["qty"] = new_qty
            self._sim_cash -= px * qty
        else:
            new_qty = p["qty"] - qty
            # Realized P&L simplified into cash adj:
            realized = (px - p["avg"]) * min(p["qty"], qty)
            self._sim_cash += px * qty + realized
            p["qty"] = new_qty
            if p["qty"] <= 0:
                p["avg"] = 0.0
        self._sim_positions[sym] = p

    def _mtm(self, sym: str, qty: float, avg: float) -> float:
        last = self._sim_last.get(sym, avg)
        return (last - avg) * qty

# ============================================================
# Singleton & thin module API
# ============================================================
_client = _ZerodhaClient()

def connect() -> bool:
    return _client.connect()

def is_connected() -> bool:
    return _client.is_connected()

def disconnect() -> None:
    _client.disconnect()

def submit_order(symbol: str, side: str, qty: float, *,
                 order_type: str = "market",
                 limit_price: Optional[float] = None,
                 tag: Optional[str] = None,
                 venue: Optional[str] = None,
                 product: Optional[str] = None,
                 variety: Optional[str] = None) -> str:
    ack = _client.submit_order(symbol, side, qty, order_type, limit_price, tag, venue, product, variety)
    return ack.order_id

def cancel_order(order_id: str, *, variety: Optional[str] = None) -> bool:
    return _client.cancel_order(order_id, variety=variety)

def replace_order(order_id: str, *, new_qty: Optional[int] = None,
                  new_limit: Optional[float] = None, variety: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    return _client.replace_order(order_id, new_qty=new_qty, new_limit=new_limit, variety=variety)

def account_summary() -> Dict[str, Any]:
    return _client.account_summary()

def positions() -> List[Dict[str, Any]]:
    return _client.positions()

def last_price(symbol: str, venue: Optional[str] = None) -> Optional[float]:
    return _client.last_price(symbol, venue)

# Simulator helper (optional)
def push_price(symbol: str, last: float) -> None:
    _client.push_price(symbol, last)

# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":  # pragma: no cover
    print("connect:", connect(), "kite=", ("yes" if (_HAS_KITE and not Z_PAPER_SIM) else "no/sim"))
    print("acct:", account_summary())
    push_price("RELIANCE", 2500.00)
    oid = submit_order("RELIANCE", "buy", 10, order_type="limit", limit_price=2500.0, tag="DEMO", venue="NSE", product="CNC")
    print("order id:", oid)
    print("positions:", positions())
    print("last price:", last_price("RELIANCE"))