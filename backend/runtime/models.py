# backend/engine/models.py
from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List


# ----------------------------- helpers ---------------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def ensure_symbol(s: str | None) -> str:
    return (s or "").upper()

def asdict_json(obj) -> Dict[str, Any]:
    """dataclass -> plain dict (ready for JSON)."""
    d = asdict(obj)
    # prune None to keep payloads compact
    return {k: v for k, v in d.items() if v is not None}

def to_json(obj) -> str:
    return json.dumps(asdict_json(obj), separators=(",", ":"))


# ----------------------------- enums -----------------------------------

class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"

class TIF(str, enum.Enum):
    DAY = "day"
    IOC = "ioc"
    GTC = "gtc"


# ----------------------------- market data -----------------------------

@dataclass
class Tick:
    """
    Normalized tick payload used across the engine.
    Minimal but sufficient; extend as needed.
    """
    symbol: str
    price: float
    ts_ms: int = field(default_factory=now_ms)
    venue: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    size: Optional[float] = None

    @staticmethod
    def from_any(d: Dict[str, Any]) -> "Tick":
        return Tick(
            symbol=ensure_symbol(d.get("symbol") or d.get("s")),
            price=float(d.get("price") or d.get("p") or 0.0),
            ts_ms=int(d.get("ts_ms") or d.get("t") or now_ms()),
            venue=d.get("venue"),
            bid=(float(d["bid"]) if d.get("bid") else None),
            ask=(float(d["ask"]) if d.get("ask") else None),
            size=(float(d["size"]) if d.get("size") else None),
        )


# ----------------------------- news ------------------------------------

@dataclass
class NewsEvent:
    """
    Canonical news event (already used by your news adapters and strategies).
    """
    source: str                          # 'yahoo' | 'moneycontrol' | ...
    headline: str
    url: Optional[str] = None
    symbol: Optional[str] = None         # mapped/normalized symbol if known
    score: Optional[float] = None        # sentiment in [-1,1]
    published_at: Optional[int] = None   # epoch seconds or ms (client choice)
    ts_ms: int = field(default_factory=now_ms)

    @staticmethod
    def from_any(d: Dict[str, Any]) -> "NewsEvent":
        sym = d.get("symbol") or d.get("ticker")
        pub = d.get("published_at") or d.get("timestamp")
        # normalize to ms if it looks like seconds
        if pub and pub < 2_000_000_000:  # before year ~2033
            pub = int(pub) * 1000
        return NewsEvent(
            source=str(d.get("source") or "unknown"),
            headline=str(d.get("headline") or d.get("title") or ""),
            url=d.get("url"),
            symbol=ensure_symbol(sym) if sym else None,
            score=(float(d["score"]) if d.get("score") is not None else None),
            published_at=(int(pub) if pub else None),
        )


# ----------------------------- orders / acks / fills -------------------

@dataclass
class OrderMsg:
    """
    Pre-risk order payload (goes on 'orders.incoming').
    Matches your Strategy.order(...) fields.
    """
    symbol: str
    side: Side
    qty: float
    typ: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    tif: Optional[TIF] = None
    strategy: Optional[str] = None
    region: Optional[str] = None
    mark_price: Optional[float] = None
    client_order_id: Optional[str] = None
    ts_ms: int = field(default_factory=now_ms)

    def validate(self) -> None:
        assert self.symbol, "symbol required"
        assert self.side in (Side.BUY, Side.SELL), "invalid side"
        assert self.qty > 0, "qty must be > 0"
        if self.typ == OrderType.LIMIT:
            assert self.limit_price and self.limit_price > 0, "limit order needs limit_price"

@dataclass
class OrderAckMsg:
    """
    Minimal broker/OMS acknowledgement echo for UI/TCA.
    """
    ok: bool
    symbol: str
    strategy: Optional[str] = None
    order_id: Optional[str] = None
    reason: Optional[str] = None
    ts_ms: int = field(default_factory=now_ms)

@dataclass
class FillMsg:
    """
    Fill echo used by TCA, positions, and dashboards.
    """
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    ts_ms: int
    strategy: Optional[str] = None
    venue: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


# ----------------------------- positions / account ---------------------

@dataclass
class PositionState:
    symbol: str
    qty: float
    avg_price: float

@dataclass
class AccountState:
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    currency: str = "USD"


# ----------------------------- signals & risk --------------------------

@dataclass
class StrategySignal:
    strategy: str
    score: float              # [-1,1]
    ts_ms: int = field(default_factory=now_ms)
    meta: Optional[Dict[str, Any]] = None

@dataclass
class RiskSnapshot:
    """
    Lightweight risk metrics snapshot for dashboards.
    """
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    vol: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    ts_ms: int = field(default_factory=now_ms)


# ----------------------------- PnL / attribution -----------------------

@dataclass
class PnLTotals:
    realized: float = 0.0
    unrealized: float = 0.0
    fees: float = 0.0

    @property
    def pnl(self) -> float:
        return self.realized + self.unrealized - self.fees

@dataclass
class PnLAttribution:
    """
    Generalized attribution container for your dashboards (/pnl endpoint).
    """
    totals: PnLTotals = field(default_factory=PnLTotals)
    by_symbol: Dict[str, PnLTotals] = field(default_factory=dict)
    by_strategy: Dict[str, PnLTotals] = field(default_factory=dict)
    by_region: Dict[str, PnLTotals] = field(default_factory=dict)
    ts_ms: int = field(default_factory=now_ms)


# ----------------------------- serialization --------------------------

def dumps(obj: Any) -> str:
    """Alias to to_json for convenience."""
    return to_json(obj)

def loads_tick(s: str | Dict[str, Any]) -> Tick:
    return Tick.from_any(json.loads(s) if isinstance(s, str) else s)

def loads_news(s: str | Dict[str, Any]) -> NewsEvent:
    return NewsEvent.from_any(json.loads(s) if isinstance(s, str) else s)


# ----------------------------- public API ------------------------------

__all__ = [
    # enums
    "Side", "OrderType", "TIF",
    # market data / news
    "Tick", "NewsEvent",
    # trading
    "OrderMsg", "OrderAckMsg", "FillMsg",
    # state
    "PositionState", "AccountState",
    # signals / risk
    "StrategySignal", "RiskSnapshot",
    # pnl
    "PnLTotals", "PnLAttribution",
    # helpers
    "now_ms", "ensure_symbol", "asdict_json", "to_json",
    "dumps", "loads_tick", "loads_news",
]