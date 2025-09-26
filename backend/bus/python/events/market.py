# bus/python/events/market.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Base class for all market events
# ---------------------------------------------------------------------
@dataclass
class MarketEvent:
    event_type: str
    ts_event: int  # event timestamp in ms
    ts_ingest: int  # ingestion timestamp in ms
    source: str     # e.g., "binance", "bloomberg", "cme"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarketEvent":
        return cls(**d)

    @classmethod
    def now(cls, event_type: str, source: str, **kwargs) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        return {
            "event_type": event_type,
            "ts_event": now_ms,
            "ts_ingest": now_ms,
            "source": source,
            **kwargs,
        }


# ---------------------------------------------------------------------
# Tick (last price, best bid/ask)
# ---------------------------------------------------------------------
@dataclass
class Tick(MarketEvent):
    symbol: str
    price: float
    size: float
    bid: Optional[float] = None
    ask: Optional[float] = None

    @classmethod
    def create(cls, symbol: str, price: float, size: float,
               bid: Optional[float] = None, ask: Optional[float] = None,
               source: str = "feed") -> "Tick":
        base = MarketEvent.now(event_type="tick", source=source)
        return cls(symbol=symbol, price=price, size=size, bid=bid, ask=ask, **base)


# ---------------------------------------------------------------------
# Trade (executed trade)
# ---------------------------------------------------------------------
@dataclass
class Trade(MarketEvent):
    symbol: str
    price: float
    size: float
    side: str  # "buy" or "sell"
    trade_id: Optional[str] = None

    @classmethod
    def create(cls, symbol: str, price: float, size: float, side: str,
               trade_id: Optional[str] = None, source: str = "feed") -> "Trade":
        base = MarketEvent.now(event_type="trade", source=source)
        return cls(symbol=symbol, price=price, size=size, side=side, trade_id=trade_id, **base)


# ---------------------------------------------------------------------
# OrderBook snapshot (level 2 or level 3)
# ---------------------------------------------------------------------
@dataclass
class OrderBook(MarketEvent):
    symbol: str
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]

    @classmethod
    def create(cls, symbol: str, bids: List[List[float]], asks: List[List[float]],
               source: str = "feed") -> "OrderBook":
        base = MarketEvent.now(event_type="orderbook", source=source)
        return cls(symbol=symbol, bids=bids, asks=asks, **base)


# ---------------------------------------------------------------------
# Derivatives quote (options/futures)
# ---------------------------------------------------------------------
@dataclass
class DerivativeQuote(MarketEvent):
    symbol: str
    expiry: str
    strike: float
    option_type: Optional[str] = None  # "call" or "put"
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None

    @classmethod
    def create(cls, symbol: str, expiry: str, strike: float,
               option_type: Optional[str] = None,
               bid: Optional[float] = None, ask: Optional[float] = None,
               last: Optional[float] = None, source: str = "feed") -> "DerivativeQuote":
        base = MarketEvent.now(event_type="derivative", source=source)
        return cls(symbol=symbol, expiry=expiry, strike=strike,
                   option_type=option_type, bid=bid, ask=ask, last=last, **base)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    tick = Tick.create(symbol="AAPL", price=190.25, size=100, bid=190.20, ask=190.30, source="nasdaq")
    trade = Trade.create(symbol="BTC-USD", price=45000, size=0.5, side="buy", source="binance")
    ob = OrderBook.create(symbol="ETH-USD", bids=[[3500, 2.0]], asks=[[3510, 1.5]], source="coinbase")
    dq = DerivativeQuote.create(symbol="SPY", expiry="2024-12-20", strike=450, option_type="call", bid=2.3, ask=2.5)

    print(tick.to_json())
    print(trade.to_json())
    print(ob.to_json())
    print(dq.to_json())