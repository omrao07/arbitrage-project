from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


# -------- Market Tick --------
class MarketTick(BaseModel):
    symbol: str = Field(..., description="Trading symbol, e.g. BTCUSDT")
    price: float = Field(..., description="Last trade price")
    volume: float = Field(..., description="Last trade volume")
    timestamp: datetime = Field(..., description="Timestamp of the tick (UTC)")
    source: Optional[str] = Field(None, description="Data source, e.g. binance")


# -------- Order --------
class OrderMessage(BaseModel):
    strategy: str = Field(..., description="Strategy ID or name")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="'buy' or 'sell'")
    qty: float = Field(..., description="Quantity to trade")
    price: Optional[float] = Field(None, description="Limit price (None = market)")
    order_type: str = Field(default="market", description="market|limit")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# -------- Position --------
class Position(BaseModel):
    symbol: str
    qty: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float


# -------- Portfolio --------
class Portfolio(BaseModel):
    cash: float
    equity: float
    positions: Dict[str, Position]


# -------- Strategy Status --------
class StrategyStatus(BaseModel):
    name: str
    active: bool
    last_signal_time: Optional[datetime]