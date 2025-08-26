# backend/execution_plus/adapters.py
"""
Execution Adapters: unified interface + simple mocks for venues.

Usage:
    from backend.execution_plus.adapters import (
        AdapterBase, AdapterRegistry, Order, OrderResult, Side, OrderType
    )

    adapter = AdapterRegistry.get("BINANCE")  # or load_from_path("engine.adapters.binance")
    q = adapter.get_quote("BTCUSDT")
    res = adapter.place_order(Order(symbol="BTCUSDT", side=Side.BUY, qty=0.01, type=OrderType.MARKET))

Wire this into your Global Arbitrage Router (discovery/routing) so each venue is a pluggable adapter.
"""

from __future__ import annotations

import abc
import enum
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol


# ---------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------

class Side(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    client_id: Optional[str] = None
    venue_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str]
    symbol: str
    side: Side
    filled_qty: float
    avg_price: Optional[float]
    fees: float
    status: str  # "filled" | "partial" | "rejected" | "cancelled" | "accepted"
    ts: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Quote:
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    ts: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VenueConfig:
    id: str
    name: str
    type: str            # equities | crypto | futures
    region: str
    base_currency: str
    maker_fee_bps: float
    taker_fee_bps: float
    min_order_size: float
    max_order_size: float
    avg_latency_ms: int


# ---------------------------------------------------------------------
# Adapter Interface
# ---------------------------------------------------------------------

class AdapterBase(abc.ABC):
    """
    Minimal interface all venue adapters must implement.
    """

    def __init__(self, cfg: VenueConfig):
        self.cfg = cfg

    # --- Market data ---
    @abc.abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abc.abstractmethod
    def get_symbols(self) -> List[str]: ...

    # --- Trading ---
    @abc.abstractmethod
    def place_order(self, order: Order) -> OrderResult: ...

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    # --- Accounts ---
    @abc.abstractmethod
    def get_balance(self) -> Dict[str, float]: ...

    # --- Venue metadata ---
    def fee_bps(self, taker: bool = True) -> float:
        return self.cfg.taker_fee_bps if taker else self.cfg.maker_fee_bps

    def latency_ms(self) -> int:
        return self.cfg.avg_latency_ms


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _now() -> float:
    return time.time()

def _apply_latency(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)

def _fee_amount(notional: float, bps: float) -> float:
    return notional * (bps / 10_000.0)


# ---------------------------------------------------------------------
# Mocks for fast wiring (replace with real SDKs later)
# ---------------------------------------------------------------------

class _MockLimitBook:
    def __init__(self, mid: float):
        self.mid = mid

    def quote(self) -> Tuple[float, float, float]:
        spread = self.mid * 0.0008  # 8 bps spread
        bid = self.mid - spread / 2
        ask = self.mid + spread / 2
        return bid, ask, self.mid

    def slip_fill(self, side: Side, qty: float) -> Tuple[float, float]:
        # naive slippage model: +/- 2 bps per $100k notional
        bps = 2 + max(0, (qty * self.mid) / 100_000.0) * 2
        sign = +1 if side == Side.BUY else -1
        px = self.mid * (1 + sign * bps / 10_000.0)
        return max(0.0, qty), px


class MockAdapter(AdapterBase):
    """
    Venue-agnostic mock that simulates quotes & fills with latency/fees
    derived from VenueConfig. Good enough for router dev & backtests.
    """

    def __init__(self, cfg: VenueConfig, symbols: List[str], seed_mid: Dict[str, float]):
        super().__init__(cfg)
        self._symbols = symbols
        self._books: Dict[str, _MockLimitBook] = {s: _MockLimitBook(seed_mid.get(s, 100.0)) for s in symbols}
        self._balances: Dict[str, float] = {"CASH": 10_000_000.0}
        self._orders: Dict[str, OrderResult] = {}

    # --- Market data ---
    def get_symbols(self) -> List[str]:
        return list(self._symbols)

    def get_quote(self, symbol: str) -> Quote:
        _apply_latency(self.cfg.avg_latency_ms)
        book = self._books.get(symbol)
        if not book:
            return Quote(None, None, None, _now(), raw={"error": "unknown_symbol"})
        # jitter mid
        j = (random.random() - 0.5) * 0.001 * book.mid
        book.mid = max(0.01, book.mid + j)
        bid, ask, mid = book.quote()
        return Quote(bid=bid, ask=ask, mid=mid, ts=_now())

    # --- Trading ---
    def place_order(self, order: Order) -> OrderResult:
        _apply_latency(self.cfg.avg_latency_ms)
        book = self._books.get(order.symbol)
        if not book:
            return OrderResult(False, None, order.symbol, order.side, 0.0, None, 0.0, "rejected", _now(), {"reason": "unknown_symbol"})

        # size limits
        notional_guess = (book.mid if order.limit_price is None else order.limit_price) * order.qty
        if notional_guess < self.cfg.min_order_size or notional_guess > self.cfg.max_order_size:
            return OrderResult(False, None, order.symbol, order.side, 0.0, None, 0.0, "rejected", _now(), {"reason": "size_limits"})

        # fill model
        if order.type == OrderType.MARKET:
            filled_qty, px = book.slip_fill(order.side, order.qty)
        else:
            # limit: fill if favorable vs book
            bid, ask, mid = book.quote()
            want = order.limit_price or mid
            if order.side == Side.BUY and want >= ask:
                filled_qty, px = order.qty, min(want, ask)
            elif order.side == Side.SELL and want <= bid:
                filled_qty, px = order.qty, max(want, bid)
            else:
                return OrderResult(True, f"lim_{int(_now()*1000)}", order.symbol, order.side, 0.0, None, 0.0, "accepted", _now(), {"limit": want})

        notional = filled_qty * px
        fees = _fee_amount(notional, self.fee_bps(taker=True))
        oid = f"ord_{int(_now()*1000)}"
        res = OrderResult(True, oid, order.symbol, order.side, filled_qty, px, fees, "filled", _now())
        self._orders[oid] = res
        # cash update (very rough)
        if order.side == Side.BUY:
            self._balances["CASH"] -= (notional + fees)
        else:
            self._balances["CASH"] += (notional - fees)
        return res

    def cancel_order(self, order_id: str) -> bool:
        _apply_latency(self.cfg.avg_latency_ms // 2)
        # Our mock fills immediately; only accepted limits can be cancelled
        res = self._orders.get(order_id)
        if not res:
            return False
        if res.status == "accepted":
            res.status = "cancelled"
            return True
        return False

    def get_balance(self) -> Dict[str, float]:
        _apply_latency(self.cfg.avg_latency_ms // 4)
        return dict(self._balances)


# ---------------------------------------------------------------------
# Pre-wired venue adapters (mocked) matching your venues.yaml
# ---------------------------------------------------------------------

def _cfg(**kw) -> VenueConfig:
    return VenueConfig(**kw)

# Seed mids for symbols per venue (tweak as needed)
_EQUITY_SEED = {"AAPL": 200.0, "TSLA": 250.0, "NVDA": 120.0, "RELIANCE.NS": 2900.0}
_CRYPTO_SEED = {"BTCUSDT": 65_000.0, "ETHUSDT": 3_200.0}
_FUT_SEED    = {"ESU5": 5400.0, "CLZ5": 80.0}

class NYSEAdapter(MockAdapter): pass
class NSEAdapter(MockAdapter): pass
class LSEAdapter(MockAdapter): pass
class CMEAdapter(MockAdapter): pass
class BinanceAdapter(MockAdapter): pass


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

class AdapterRegistry:
    """
    Simple in-process registry keyed by venue id.
    """
    _REG: Dict[str, AdapterBase] = {}

    @classmethod
    def bootstrap_defaults(cls) -> None:
        if cls._REG:
            return
        cls._REG["NYSE"] = NYSEAdapter(_cfg(
            id="NYSE", name="New York Stock Exchange", type="equities", region="US",
            base_currency="USD", maker_fee_bps=0.5, taker_fee_bps=1.0,
            min_order_size=100.0, max_order_size=1_000_000.0, avg_latency_ms=20
        ), symbols=list(_EQUITY_SEED.keys() - {"RELIANCE.NS"} if hasattr(dict, "keys") else ["AAPL","TSLA","NVDA"]),
        seed_mid=_EQUITY_SEED)

        cls._REG["NSE"] = NSEAdapter(_cfg(
            id="NSE", name="National Stock Exchange of India", type="equities", region="IN",
            base_currency="INR", maker_fee_bps=1.0, taker_fee_bps=1.5,
            min_order_size=10_000.0, max_order_size=50_000_000.0, avg_latency_ms=40
        ), symbols=["RELIANCE.NS"], seed_mid=_EQUITY_SEED)

        cls._REG["LSE"] = LSEAdapter(_cfg(
            id="LSE", name="London Stock Exchange", type="equities", region="UK",
            base_currency="GBP", maker_fee_bps=0.6, taker_fee_bps=1.2,
            min_order_size=1_000.0, max_order_size=5_000_000.0, avg_latency_ms=30
        ), symbols=["AAPL","TSLA"], seed_mid=_EQUITY_SEED)

        cls._REG["CME"] = CMEAdapter(_cfg(
            id="CME", name="Chicago Mercantile Exchange", type="futures", region="US",
            base_currency="USD", maker_fee_bps=0.25, taker_fee_bps=0.25,
            min_order_size=1.0, max_order_size=100_000.0, avg_latency_ms=25
        ), symbols=list(_FUT_SEED.keys()), seed_mid=_FUT_SEED)

        cls._REG["BINANCE"] = BinanceAdapter(_cfg(
            id="BINANCE", name="Binance", type="crypto", region="SG",
            base_currency="USDT", maker_fee_bps=0.1, taker_fee_bps=0.1,
            min_order_size=10.0, max_order_size=10_000_000.0, avg_latency_ms=80
        ), symbols=list(_CRYPTO_SEED.keys()), seed_mid=_CRYPTO_SEED)

    @classmethod
    def get(cls, venue_id: str) -> AdapterBase:
        if not cls._REG:
            cls.bootstrap_defaults()
        try:
            return cls._REG[venue_id.upper()]
        except KeyError:
            raise KeyError(f"Adapter for venue '{venue_id}' not found")

    @classmethod
    def register(cls, venue_id: str, adapter: AdapterBase) -> None:
        cls._REG[venue_id.upper()] = adapter

    @classmethod
    def all(cls) -> Dict[str, AdapterBase]:
        if not cls._REG:
            cls.bootstrap_defaults()
        return dict(cls._REG)


# ---------------------------------------------------------------------
# Optional: load from dotted path (for adapters specified in YAML)
# ---------------------------------------------------------------------

def load_from_path(path: str, cfg: VenueConfig, **kwargs) -> AdapterBase:
    """
    Dynamically load an adapter class from a dotted path string, e.g.:
        "engine.adapters.binance:BinanceREST"
        "engine.adapters.nyse:NYSEFix"
    If no class is given, looks for `Adapter` in the module.
    """
    import importlib
    mod_name, _, cls_name = path.partition(":")
    mod = importlib.import_module(mod_name)
    cls_obj = getattr(mod, cls_name or "Adapter")
    if not issubclass(cls_obj, AdapterBase):
        raise TypeError(f"{cls_obj} is not an AdapterBase")
    return cls_obj(cfg, **kwargs)


# ---------------------------------------------------------------------
# Tiny smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    AdapterRegistry.bootstrap_defaults()
    binance = AdapterRegistry.get("BINANCE")
    print("BINANCE symbols:", binance.get_symbols())
    q = binance.get_quote("BTCUSDT")
    print("Quote BTCUSDT:", q)
    res = binance.place_order(Order(symbol="BTCUSDT", side=Side.BUY, qty=0.01))
    print("Order result:", res)
    print("Balance:", binance.get_balance())