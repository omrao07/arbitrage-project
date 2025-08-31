# backend/ai/agents/connectors/brokers/ibkr.py
from __future__ import annotations

import os
import time
import uuid
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# ============================================================
# Configuration (env)
# ============================================================
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))  # 7497=TWS paper, 7496=TWS live, 4002/4001=GW
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "117"))
IBKR_ACCOUNT = os.getenv("IBKR_ACCOUNT", "")     # optional filter
IBKR_TIMEOUT_S = float(os.getenv("IBKR_TIMEOUT_S", "10"))
IBKR_PAPER = os.getenv("IBKR_PAPER", "true").lower() in ("1","true","yes","y")

# ============================================================
# Optional dependency: ib_insync
# ============================================================
try:
    from ib_insync import IB, Stock, Forex, Contract, MarketOrder, LimitOrder, util  # type: ignore
    _HAS_IB = True
except Exception:
    _HAS_IB = False

# ============================================================
# Public order result type
# ============================================================
@dataclass
class OrderAck:
    order_id: str
    broker: str = "ibkr"
    status: str = "accepted"   # accepted | rejected | sent | filled | cancelled | replaced
    details: Dict[str, Any] = None # type: ignore

# ============================================================
# Utility
# ============================================================
def _now_ms() -> int:
    return int(time.time() * 1000)

# ============================================================
# Contract helpers
# ============================================================
def _equity(symbol: str, exchange: Optional[str] = None, currency: str = "USD") -> Any:
    if _HAS_IB:
        return Stock(symbol.upper(), exchange or "SMART", currency.upper())
    return {"type": "STK", "symbol": symbol.upper(), "exchange": exchange or "SIM", "currency": currency.upper()}

def _fx(pair: str) -> Any:
    if _HAS_IB:
        # pair like "EURUSD"
        base, quote = pair[:3].upper(), pair[3:].upper()
        return Forex(f"{base}{quote}")
    return {"type": "CASH", "symbol": pair.upper()}

def _to_contract(symbol: str, venue: Optional[str]) -> Any:
    if ":" in symbol:
        # Allow "FX:EURUSD" or "STK:AAPL"
        klass, rest = symbol.split(":", 1)
        if klass.upper() == "FX":
            return _fx(rest)
        else:
            return _equity(rest, exchange=venue or "SMART")
    if len(symbol) == 6 and symbol.isalpha():  # simple FX heuristic
        return _fx(symbol)
    return _equity(symbol, exchange=venue or "SMART")

# ============================================================
# IBKR client (singleton)
# ============================================================
class _IBKRClient:
    def __init__(self):
        self._lock = threading.RLock()
        self._connected = False
        self._ib: Optional["IB"] = None
        # local registry for simulator fallback
        self._sim_orders: Dict[str, Dict[str, Any]] = {}

    # ------------- lifecycle -------------
    def connect(self) -> bool:
        with self._lock:
            if not _HAS_IB:
                self._connected = True
                return True
            if self._connected and self._ib and self._ib.isConnected():
                return True
            self._ib = IB()
            try:
                self._ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=IBKR_TIMEOUT_S) # type: ignore
                self._connected = True
                return True
            except Exception:
                self._connected = False
                self._ib = None
                return False

    def is_connected(self) -> bool:
        if not _HAS_IB:
            return True
        return bool(self._ib and self._ib.isConnected())

    def disconnect(self) -> None:
        with self._lock:
            if _HAS_IB and self._ib and self._ib.isConnected():
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
            self._connected = False

    # ------------- trading API -------------
    def submit_order(self, symbol: str, side: str, qty: float,
                     order_type: str = "market",
                     limit_price: Optional[float] = None,
                     tag: Optional[str] = None,
                     venue: Optional[str] = None) -> OrderAck:
        """
        Submit a simple equity/FX order. Returns an OrderAck with broker order id.
        """
        if not self.connect():
            # fallback to simulator
            oid = f"SIM-{uuid.uuid4().hex[:10]}"
            self._sim_orders[oid] = {
                "symbol": symbol, "side": side, "qty": float(qty),
                "type": order_type, "limit": limit_price, "ts": _now_ms(), "status": "accepted", "tag": tag
            }
            return OrderAck(order_id=oid, status="accepted", details={"sim": True})

        if not _HAS_IB or self._ib is None:
            oid = f"SIM-{uuid.uuid4().hex[:10]}"
            return OrderAck(order_id=oid, status="accepted", details={"sim": True, "note": "ib_insync not installed"})

        side = side.lower()
        order_type = order_type.lower()

        contract = _to_contract(symbol, venue)
        if order_type in ("market", "mkt"):
            order = MarketOrder("BUY" if side == "buy" else "SELL", abs(float(qty)))
        elif order_type in ("limit", "lmt"):
            if limit_price is None:
                raise ValueError("limit_price required for limit order")
            order = LimitOrder("BUY" if side == "buy" else "SELL", abs(float(qty)), float(limit_price))
        else:
            raise ValueError(f"unsupported order_type '{order_type}'")

        if tag:
            try:
                order.tif = getattr(order, "tif", "DAY")
                order.orderRef = str(tag)[:32]
            except Exception:
                pass

        with self._lock:
            trade = self._ib.placeOrder(contract, order)
            # IB assigns orderId immediately
            oid = str(getattr(trade, "order", None).orderId if getattr(trade, "order", None) else uuid.uuid4().hex[:8]) # type: ignore
            return OrderAck(
                order_id=oid,
                status="sent",
                details={"orderType": order.orderType, "tif": getattr(order, "tif", "DAY"), "ref": getattr(order, "orderRef", None)}
            )

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected() or not _HAS_IB or self._ib is None:
            # simulate
            o = self._sim_orders.get(order_id)
            if o:
                o["status"] = "cancelled"
                return True
            return False

        with self._lock:
            try:
                # Find trade by id
                for tr in list(self._ib.trades()):
                    if str(tr.order.orderId) == str(order_id):
                        self._ib.cancelOrder(tr.order)
                        return True
            except Exception:
                return False
        return False

    def replace_order(self, order_id: str, *, new_qty: Optional[float] = None,
                      new_limit: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Cancel/replace semantics: modify qty/limit if order is working.
        Returns (ok, new_order_id_or_none).
        """
        if not self.is_connected() or not _HAS_IB or self._ib is None:
            o = self._sim_orders.get(order_id)
            if not o:
                return (False, None)
            if new_qty is not None: o["qty"] = float(new_qty)
            if new_limit is not None: o["limit"] = float(new_limit)
            o["status"] = "replaced"
            return (True, order_id)

        with self._lock:
            try:
                for tr in list(self._ib.trades()):
                    if str(tr.order.orderId) == str(order_id):
                        order = tr.order
                        if new_qty is not None:
                            order.totalQuantity = abs(float(new_qty))
                        if new_limit is not None and hasattr(order, "lmtPrice"):
                            order.lmtPrice = float(new_limit)
                        self._ib.placeOrder(tr.contract, order)
                        return (True, str(order.orderId))
            except Exception:
                return (False, None)
        return (False, None)

    # ------------- account/market data -------------
    def account_summary(self) -> Dict[str, Any]:
        if not self.connect() or not _HAS_IB or self._ib is None:
            # minimal sim snapshot
            return {"account": IBKR_ACCOUNT or "SIM", "cash": 1_000_000.0, "net_liq": 1_000_000.0, "paper": not _HAS_IB}
        with self._lock:
            try:
                tagvals = self._ib.accountSummary()
                out = {}
                for tv in tagvals:
                    if IBKR_ACCOUNT and tv.account != IBKR_ACCOUNT:
                        continue
                    out.setdefault(tv.account, {})[tv.tag] = tv.value
                return out if out else {"note": "no account summary"}
            except Exception as e:
                return {"error": str(e)}

    def positions(self) -> List[Dict[str, Any]]:
        if not self.connect() or not _HAS_IB or self._ib is None:
            return []
        with self._lock:
            try:
                poss = self._ib.positions()
                out = []
                for p in poss:
                    out.append({
                        "account": p.account,
                        "symbol": getattr(p.contract, "symbol", None) or getattr(p.contract, "localSymbol", None),
                        "conId": getattr(p.contract, "conId", None),
                        "qty": float(p.position),
                        "avg_cost": float(p.avgCost),
                    })
                return out
            except Exception as e:
                return [{"error": str(e)}]

    def last_price(self, symbol: str, venue: Optional[str] = None) -> Optional[float]:
        if not self.connect() or not _HAS_IB or self._ib is None:
            return None
        with self._lock:
            try:
                c = _to_contract(symbol, venue)
                ticker = self._ib.reqMktData(c, "", False, False)
                # give IB a moment to populate
                util.sleep(0.3)
                px = None
                if getattr(ticker, "last", None):
                    px = float(ticker.last)
                elif getattr(ticker, "marketPrice", None):
                    px = float(ticker.marketPrice())
                self._ib.cancelMktData(c)
                return px
            except Exception:
                return None

# ============================================================
# Singleton & module-level convenience
# ============================================================
_client = _IBKRClient()

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
                 venue: Optional[str] = None) -> str:
    """
    Thin wrapper to match skills.trading.broker_interface contract.
    Returns broker order id (string).
    """
    ack = _client.submit_order(symbol, side, qty, order_type, limit_price, tag, venue)
    return ack.order_id

def cancel_order(order_id: str) -> bool:
    return _client.cancel_order(order_id)

def replace_order(order_id: str, *, new_qty: Optional[float] = None,
                  new_limit: Optional[float] = None) -> Tuple[bool, Optional[str]]:
    return _client.replace_order(order_id, new_qty=new_qty, new_limit=new_limit)

def account_summary() -> Dict[str, Any]:
    return _client.account_summary()

def positions() -> List[Dict[str, Any]]:
    return _client.positions()

def last_price(symbol: str, venue: Optional[str] = None) -> Optional[float]:
    return _client.last_price(symbol, venue)

# ============================================================
# Quick smoke test
# ============================================================
if __name__ == "__main__":  # pragma: no cover
    print("Connecting to IBKR…", connect(), "(paper:" + str(IBKR_PAPER) + ")")
    print("Connected:", is_connected())
    print("Account summary:", account_summary())
    oid = submit_order("AAPL", "buy", 10, order_type="limit", limit_price=10.00, tag="DEMO")
    print("order id:", oid)
    ok = cancel_order(oid)
    print("cancel:", ok)
    print("positions:", positions()[:2])
    print("last px AAPL:", last_price("AAPL"))
    disconnect()