# backend/common/schemas.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Literal, Type, TypeVar

SCHEMA_VERSION = "0.5.0"

def now_ms() -> int:
    return int(time.time() * 1000)

def _coerce_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        d = asdict(obj) # type: ignore
    elif isinstance(obj, dict):
        d = dict(obj)
    else:
        raise TypeError("to_dict expects dataclass or dict")
    d["_schema_version"] = SCHEMA_VERSION
    return {k: v for k, v in d.items() if v is not None}

T = TypeVar("T")

def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
    fields = getattr(cls, "__annotations__", {})
    payload = {k: data.get(k) for k in fields.keys()}
    return cls(**payload)  # type: ignore

# ========= Market data =========
Side = Literal["buy", "sell"]
TIF = Literal["DAY", "IOC", "FOK", "GTD"]
AssetType = Literal["equity", "futures", "options", "fx", "crypto", "index"]
Session = Literal["PRE", "REG", "POST"]

@dataclass
class Quote:
    symbol: str
    ts_ms: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    venue: Optional[str] = None
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.ask >= self.bid:
            return 0.5 * (self.bid + self.ask)
        return None
    def spread_bps(self) -> Optional[float]:
        m = self.mid()
        if m and self.ask is not None and self.bid is not None and m > 0:
            return (self.ask - self.bid) / m * 1e4
        return None

@dataclass
class TradeTick:
    symbol: str
    ts_ms: int
    price: float
    size: float
    side: Optional[Literal["buyer", "seller"]] = None
    venue: Optional[str] = None

@dataclass
class Candle:
    symbol: str
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    interval: Optional[str] = None  # "1m","5m","1h",...

@dataclass
class OrderBookLevel:
    price: float
    size: float

@dataclass
class OrderBookSnapshot:
    symbol: str
    ts_ms: int
    bids: List[OrderBookLevel] = field(default_factory=list)  # best→worst
    asks: List[OrderBookLevel] = field(default_factory=list)  # best→worst
    venue: Optional[str] = None
    depth: Optional[int] = None

# ========= Orders / Execution =========
@dataclass
class OrderIntent:
    symbol: str
    side: Side
    qty: float
    urgency: Literal["low", "normal", "high"] = "normal"
    limit_price: Optional[float] = None
    tif: Optional[TIF] = None
    venue_hint: Optional[str] = None
    participation_cap: Optional[float] = None  # POV cap (0..1]
    strategy: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ParentOrder:
    order_id: str
    symbol: str
    side: Side
    qty: float
    created_ms: int = field(default_factory=now_ms)
    limit_price: Optional[float] = None
    tif: TIF = "DAY"
    status: Literal["live", "done", "canceled", "rejected"] = "live"
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChildOrder:
    child_id: str
    parent_id: str
    symbol: str
    side: Side
    qty: float
    created_ms: int = field(default_factory=now_ms)
    venue: Optional[str] = None
    algo: Optional[str] = None
    limit_price: Optional[float] = None
    tif: TIF = "DAY"
    status: Literal["live", "done", "canceled", "rejected"] = "live"
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderAck:
    order_id: str
    ts_ms: int = field(default_factory=now_ms)
    accepted: bool = True
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Fill:
    child_id: str
    parent_id: str
    symbol: str
    ts_ms: int
    price: float
    qty: float
    venue: Optional[str] = None
    fee_bps: Optional[float] = None
    liquidity_flag: Optional[Literal["add", "remove"]] = None

@dataclass
class ExecutionReport:
    parent_id: str
    ts_ms: int
    status: Literal["partial", "filled", "canceled", "rejected"]
    filled_qty: float
    avg_fill_price: Optional[float] = None
    remaining_qty: Optional[float] = None
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# ========= Positions / Portfolio =========
@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    def notional(self) -> float:
        px = self.last_price or self.avg_price or 0.0
        return abs(self.qty) * px

@dataclass
class PortfolioSnapshot:
    cash: float
    nav: float
    leverage: float
    positions: Dict[str, Position] = field(default_factory=dict)
    adv: Dict[str, float] = field(default_factory=dict)
    spread_bps: Dict[str, float] = field(default_factory=dict)
    symbol_weights: Dict[str, float] = field(default_factory=dict)
    var_1d_frac: Optional[float] = None
    drawdown_frac: Optional[float] = None
    ts_ms: int = field(default_factory=now_ms)

# ========= Routing / Policy / Risk =========
@dataclass
class VenueWeight:
    venue: str
    weight: float
    dark: bool = False

@dataclass
class SlicePlan:
    child_qty: float
    interval_ms: int
    max_spread_bps: float
    price_limit: Optional[float] = None
    post_only: bool = False

@dataclass
class PolicyPlan:
    algo: Literal["VWAP", "TWAP", "POV", "AdaptiveVWAP", "IOC", "FOK"]
    tif: TIF
    limit_price: Optional[float]
    max_slippage_bps: float
    max_participation: Optional[float]
    slice: SlicePlan
    venues: List[VenueWeight] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouteDecision:
    ok: bool
    broker: Optional[str] = None
    account: Optional[str] = None
    venues: List[Dict[str, Any]] = field(default_factory=list)  # {"venue","weight"}
    algo_hint: Optional[str] = None
    tif: Optional[TIF] = None
    notes: List[str] = field(default_factory=list)
    rule_id: Optional[str] = None
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskDecision:
    ok: bool
    level: Literal["allow", "warn", "block", "halt"]
    reasons: List[str] = field(default_factory=list)
    soft_caps: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

# ========= News / Sentiment =========
@dataclass
class NewsItem:
    id: str
    ts_ms: int
    source: str                 # "yahoo","moneycontrol","reuters",...
    title: str
    url: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    raw: Optional[str] = None
    lang: Optional[str] = "en"

@dataclass
class SentimentScore:
    news_id: str
    ts_ms: int
    symbol: Optional[str]
    score: float                # [-1..1]
    model: Optional[str] = None
    confidence: Optional[float] = None

# ========= Ledger / Events =========
@dataclass
class LedgerEvent:
    kind: Literal[
        "order.created","order.ack","order.child","order.fill","order.exec",
        "risk.decision","route.decision","policy.plan",
        "position.update","portfolio.snapshot",
        "news.item","sentiment.score"
    ]
    ts_ms: int = field(default_factory=now_ms)
    payload: Dict[str, Any] = field(default_factory=dict)
    ref_id: Optional[str] = None  # e.g., parent_id or news_id
    actor: Optional[str] = None   # strategy/agent
    @staticmethod
    def wrap(obj: Any, kind: Optional[str] = None, ref_id: Optional[str] = None, actor: Optional[str] = None) -> "LedgerEvent":
        k = kind or _infer_kind(obj)
        p = to_dict(obj) if is_dataclass(obj) else (obj if isinstance(obj, dict) else {"value": str(obj)})
        return LedgerEvent(kind=k, payload=p, ref_id=ref_id, actor=actor) # type: ignore

def _infer_kind(obj: Any) -> str:
    name = obj.__class__.__name__.lower() if is_dataclass(obj) else "" # type: ignore
    if "parentorder" in name: return "order.created"
    if "childorder" in name:  return "order.child"
    if "fill" in name:        return "order.fill"
    if "executionreport" in name: return "order.exec"
    if "riskdecision" in name: return "risk.decision"
    if "routedecision" in name: return "route.decision"
    if "policyplan" in name:    return "policy.plan"
    if "position" in name:      return "position.update"
    if "portfoliosnapshot" in name: return "portfolio.snapshot"
    if "newsitem" in name:      return "news.item"
    if "sentimentscore" in name:return "sentiment.score"
    return "unknown"

# ========= Validators =========
def validate_order_intent(oi: OrderIntent) -> List[str]:
    errs: List[str] = []
    if not oi.symbol: errs.append("symbol required")
    if oi.side not in ("buy", "sell"): errs.append("side must be 'buy'|'sell'")
    if oi.qty is None or oi.qty <= 0: errs.append("qty must be > 0")
    if oi.participation_cap is not None and not (0.0 < oi.participation_cap <= 1.0):
        errs.append("participation_cap must be in (0,1]")
    if oi.limit_price is not None and _coerce_float(oi.limit_price) is None:
        errs.append("limit_price must be numeric")
    return errs

def validate_policy_plan(plan: PolicyPlan) -> List[str]:
    errs: List[str] = []
    if plan.slice.child_qty <= 0: errs.append("slice.child_qty must be > 0")
    if plan.slice.interval_ms <= 0: errs.append("slice.interval_ms must be > 0")
    if plan.max_participation is not None and not (0.0 < plan.max_participation <= 1.0):
        errs.append("max_participation must be in (0,1]")
    if plan.venues:
        s = sum(v.weight for v in plan.venues)
        if abs(s - 1.0) > 1e-6: errs.append("venues weights should sum to 1.0")
    return errs

def validate_route_decision(dec: RouteDecision) -> List[str]:
    errs: List[str] = []
    if not dec.ok: return errs
    if not dec.broker: errs.append("broker required when ok=true")
    if dec.venues:
        s = sum(float(v.get("weight", 0.0)) for v in dec.venues)
        if abs(s - 1.0) > 1e-6: errs.append("venue weights should sum to 1.0")
    return errs

def validate_risk_decision(dec: RiskDecision) -> List[str]:
    errs: List[str] = []
    if dec.ok and dec.level != "allow":
        errs.append("ok=True requires level='allow'")
    return errs

# ========= JSON helpers =========
def dumps(obj: Any, *, indent: Optional[int] = None) -> str:
    return json.dumps(to_dict(obj), ensure_ascii=False, indent=indent, default=str)

def loads(cls: type[T], s: str) -> T:
    return from_dict(cls, json.loads(s))

# ========= Smoke test =========
if __name__ == "__main__":  # pragma: no cover
    oi = OrderIntent(symbol="AAPL", side="buy", qty=1000, urgency="normal")
    print("intent errs:", validate_order_intent(oi))

    plan = PolicyPlan(
        algo="VWAP", tif="DAY", limit_price=None, max_slippage_bps=25.0, max_participation=0.1,
        slice=SlicePlan(child_qty=100, interval_ms=4000, max_spread_bps=10.0, post_only=True),
        venues=[VenueWeight("NYSE", 0.6), VenueWeight("ARCA", 0.4)],
        notes=["demo"]
    )
    print("plan errs:", validate_policy_plan(plan))

    dec = RouteDecision(ok=True, broker="ibkr", venues=[{"venue":"NYSE","weight":0.5},{"venue":"ARCA","weight":0.5}], algo_hint="POV", tif="DAY")
    print("route errs:", validate_route_decision(dec))

    ev = LedgerEvent.wrap(plan, actor="execution_agent")
    print("ledger:", dumps(ev)[:120], "...")