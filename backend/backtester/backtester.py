# backend/engine/backtester.py
from __future__ import annotations

import csv, json, math, os, importlib, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Iterable, Tuple, Callable

# ---- optional deps (graceful) ----------------------------------------------
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

# ---- Optional Redis mirror (off by default) --------------------------------
USE_REDIS = False
try:
    from redis import Redis  # type: ignore
    USE_REDIS = True
except Exception:
    USE_REDIS = False
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---- Strategy import helper -------------------------------------------------
def load_strategy(qualname: str, **kwargs) -> "Strategy": # type: ignore
    """
    qualname: 'package.module:ClassName'
    """
    mod_name, cls_name = qualname.split(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)

# ---- Data event shapes ------------------------------------------------------
@dataclass
class Bar:
    ts_ms: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

@dataclass
class News:
    ts_ms: int
    title: str
    symbols: List[str]
    score: Optional[float] = None
    source: Optional[str] = None
    url: Optional[str] = None

# ---- Cost model -------------------------------------------------------------
class CostModel:
    """
    Simple linear+slippage cost model:
      fee_bps:      commission in bps of notional
      spread_bps:   half-spread (one-way) in bps applied on market orders
      impact_bps:   extra slip per 1% ADV traded (toy)
    """
    def __init__(self, fee_bps: float = 0.5, spread_bps: float = 1.0, impact_bps: float = 0.0):
        self.fee_bps = float(fee_bps)
        self.spread_bps = float(spread_bps)
        self.impact_bps = float(impact_bps)

    def estimate_fill_price(self, side: str, mark: float, order_type: str, limit_price: Optional[float]) -> float:
        if order_type == "limit" and limit_price is not None:
            return float(limit_price)
        # market: cross the spread
        slip = (self.spread_bps / 1e4) * mark
        return mark + (slip if side == "buy" else -slip)

    def estimate_fee(self, notional: float, participation_pct: float = 1.0) -> float:
        # very crude: linear fee + small impact
        return abs(notional) * (self.fee_bps / 1e4) + abs(notional) * (self.impact_bps / 1e4) * participation_pct

# ---- Risk checks ------------------------------------------------------------
@dataclass
class RiskLimits:
    max_pos_qty: float = 1e6
    max_notional: float = 1e9
    max_order_qty: float = 1e6

class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.l = limits

    def check(self, symbol: str, side: str, qty: float, price: float, cur_pos: float) -> Tuple[bool, Optional[str]]:
        if qty <= 0 or qty > self.l.max_order_qty:
            return False, "ORDER_QTY"
        new_pos = cur_pos + (qty if side == "buy" else -qty)
        if abs(new_pos) > self.l.max_pos_qty:
            return False, "POS_LIMIT"
        if abs(new_pos * price) > self.l.max_notional:
            return False, "NOTIONAL_LIMIT"
        return True, None

# ---- Broker simulator -------------------------------------------------------
@dataclass
class Order:
    id: str
    ts_ms: int
    strategy: str
    symbol: str
    side: str       # 'buy' | 'sell'
    qty: float
    typ: str        # 'market' | 'limit'
    limit_price: Optional[float] = None
    venue: Optional[str] = None

@dataclass
class Fill:
    ts_ms: int
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: float

class BrokerSim:
    def __init__(self, cost: Optional[CostModel] = None, latency_ms: int = 5, risk: Optional[RiskManager] = None):
        self.cost = cost or CostModel()
        self.latency_ms = int(latency_ms)
        self.risk = risk or RiskManager(RiskLimits())
        self._oid = 0

    def next_id(self) -> str:
        self._oid += 1
        return f"bt-{self._oid:07d}"

    def process(self, order: Order, mark: float, cur_pos: float, adv_participation: float = 0.02) -> Optional[Fill]:
        ok, reason = self.risk.check(order.symbol, order.side, order.qty, mark, cur_pos)
        if not ok:
            return None
        px = self.cost.estimate_fill_price(order.side, mark, order.typ, order.limit_price)
        fee = self.cost.estimate_fee(notional=px * order.qty, participation_pct=adv_participation)
        # one shot fill
        ts = order.ts_ms + self.latency_ms
        return Fill(ts_ms=ts, order_id=order.id, symbol=order.symbol, side=order.side, qty=order.qty, price=px, fee=fee)

# ---- Portfolio & accounting -------------------------------------------------
@dataclass
class Position:
    qty: float = 0.0
    avg_px: float = 0.0

class Book:
    def __init__(self, capital_base: float = 100_000.0):
        self.cash = float(capital_base)
        self.pos: Dict[str, Position] = {}
        self.fees = 0.0
        self.realized = 0.0

    def on_fill(self, f: Fill):
        p = self.pos.setdefault(f.symbol, Position())
        # trade sign
        sgn = +1.0 if f.side == "buy" else -1.0
        notional = sgn * f.qty * f.price
        self.cash -= notional
        self.fees += f.fee
        # realized pnl when flipping through zero
        new_qty = p.qty + sgn * f.qty
        if p.qty != 0 and (p.qty * new_qty) < 0:  # crossed through zero
            crossing_qty = min(abs(p.qty), f.qty)
            self.realized += crossing_qty * (p.avg_px - f.price) * (1 if p.qty < 0 else -1)
        # update position & avg price
        if new_qty == 0:
            p.qty, p.avg_px = 0.0, 0.0
        elif sgn > 0:
            # buying: new weighted avg
            p.avg_px = (p.avg_px * p.qty + f.qty * f.price) / (p.qty + f.qty if p.qty + f.qty != 0 else 1.0)
            p.qty = new_qty
        else:
            # selling: reduce qty, avg stays
            p.qty = new_qty

    def mtm(self, prices: Dict[str, float]) -> Tuple[float, float, float]:
        unreal = 0.0
        gross = 0.0
        for sym, p in self.pos.items():
            px = float(prices.get(sym, p.avg_px or 0.0))
            unreal += (px - p.avg_px) * p.qty
            gross += abs(px * p.qty)
        equity = self.cash + self.realized + unreal - self.fees
        return equity, unreal, gross

# ---- Event loop / runner ----------------------------------------------------
class StrategyHarness:
    """
    Intercepts Strategy.order(...) and forwards to BrokerSim.
    """
    def __init__(self, strategy: "Strategy", broker: BrokerSim, book: Book): # type: ignore
        self.strategy = strategy
        self.broker = broker
        self.book = book
        # monkey-patch .order on the strategy instance
        self.strategy_order = strategy.order
        strategy.order = self._order_proxy  # type: ignore

    def _order_proxy(self, symbol: str, side: str, qty: float | None = None, *, order_type: str = "market", limit_price: float | None = None, venue: Optional[str] = None, mark_price: float | None = None, extra: Optional[Dict[str, Any]] = None) -> None:  # noqa: E501
        q = self.strategy.ctx.default_qty if (qty is None or qty <= 0) else float(qty)
        oid = self.broker.next_id()
        ts = int(time.time() * 1000)
        mark = float(mark_price or 0.0)
        # cur pos for risk evaluation
        cur_pos = self.book.pos.get(symbol, Position()).qty
        fill = self.broker.process(
            Order(id=oid, ts_ms=ts, strategy=self.strategy.ctx.name, symbol=symbol.upper(), side=side.lower(), qty=q, typ=order_type, limit_price=limit_price, venue=venue),
            mark=mark if mark > 0 else mark, cur_pos=cur_pos
        )
        if fill:
            self.book.on_fill(fill)
            self._blotter.append(asdict(fill))

    # def __enter.mechanics(self):
    #     ...

    def attach_blotter(self, blotter: List[Dict[str, Any]]):
        self._blotter = blotter

# ---- Loaders ----------------------------------------------------------------
def load_bars(path_or_df: str | "pd.DataFrame") -> List[Bar]: # type: ignore
    if isinstance(path_or_df, str):
        if path_or_df.endswith(".json") or path_or_df.endswith(".jsonl"):
            rows: List[Dict[str, Any]] = []
            with open(path_or_df, "r") as f:
                if path_or_df.endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                else:
                    rows = json.load(f)
        else:
            # CSV
            rows = []
            with open(path_or_df, "r", newline="") as f:
                for r in csv.DictReader(f):
                    rows.append(r)
        df = pd.DataFrame(rows) if pd is not None else rows  # type: ignore
    else:
        df = path_or_df  # type: ignore

    out: List[Bar] = []
    if pd is not None and isinstance(df, pd.DataFrame):  # type: ignore
        # Expect columns: ts_ms, symbol, open, high, low, close, volume?
        for _, r in df.iterrows():
            out.append(Bar(
                ts_ms=int(r.get("ts_ms") or r.get("timestamp") or r.get("time") or 0),
                symbol=str(r.get("symbol") or r.get("sym") or "").upper(),
                open=float(r.get("open") or r.get("o") or r.get("close")), # type: ignore
                high=float(r.get("high") or r.get("h") or r.get("close")), # type: ignore
                low=float(r.get("low") or r.get("l") or r.get("close")), # type: ignore
                close=float(r.get("close") or r.get("c") or r.get("price") or r.get("px")), # type: ignore
                volume=float(r.get("volume") or r.get("v") or 0.0),
            ))
    else:
        for r in (df if isinstance(df, list) else []):
            out.append(Bar(
                ts_ms=int(r.get("ts_ms") or r.get("timestamp") or 0),
                symbol=str(r.get("symbol") or "").upper(),
                open=float(r.get("open") or r.get("close")), # type: ignore
                high=float(r.get("high") or r.get("close")), # type: ignore
                low=float(r.get("low") or r.get("close")), # type: ignore
                close=float(r.get("close") or r.get("price")), # type: ignore
                volume=float(r.get("volume") or 0.0),
            ))
    out.sort(key=lambda b: (b.ts_ms, b.symbol))
    return out

def load_news(path: Optional[str]) -> List[News]:
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with open(path, "r") as f:
            rows = json.load(f)
    out: List[News] = []
    for r in rows:
        out.append(News(
            ts_ms=int(r.get("ts_ms") or r.get("timestamp") or 0),
            title=str(r.get("title") or r.get("text") or ""),
            symbols=[str(s).upper() for s in (r.get("symbols") or r.get("tickers") or [])],
            score=(r.get("score") if r.get("score") is not None else r.get("sentiment")),
            source=r.get("source"),
            url=r.get("url"),
        ))
    out.sort(key=lambda n: n.ts_ms)
    return out

# ---- Backtest engine --------------------------------------------------------
@dataclass
class Result:
    summary: Dict[str, Any]
    equity_curve: List[Dict[str, Any]]
    blotter: List[Dict[str, Any]]
    positions: List[Dict[str, Any]]

class Backtester:
    def __init__(
        self,
        strategy: "Strategy", # type: ignore
        *,
        capital_base: float = 100_000.0,
        cost: Optional[CostModel] = None,
        latency_ms: int = 5,
        risk_limits: Optional[RiskLimits] = None,
        redis_mirror: bool = False
    ):
        self.book = Book(capital_base=capital_base)
        self.broker = BrokerSim(cost=cost or CostModel(), latency_ms=latency_ms, risk=RiskManager(risk_limits or RiskLimits()))
        self.harness = StrategyHarness(strategy, self.broker, self.book)
        self._blotter: List[Dict[str, Any]] = []
        self.harness.attach_blotter(self._blotter)
        self._curve: List[Dict[str, Any]] = []
        self._pos_snap: List[Dict[str, Any]] = []
        self.strategy = strategy
        self.redis = None
        if redis_mirror and USE_REDIS:
            try:
                self.redis = Redis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            except Exception:
                self.redis = None

    def _mirror(self, stream: str, obj: Dict[str, Any]):
        if not self.redis:
            return
        try:
            self.redis.xadd(stream, {"json": json.dumps(obj)}, maxlen=5000, approximate=True)  # type: ignore
        except Exception:
            pass

    def run(self, bars: List[Bar], news: Optional[List[News]] = None) -> Result:
        news = news or []
        # lifecycle
        if hasattr(self.strategy, "on_start"):
            self.strategy.on_start()

        # timeline merge (bars per symbol + news)
        i_news = 0
        prices: Dict[str, float] = {}
        last_bucket = None

        for b in bars:
            prices[b.symbol] = b.close
            # emit tick to strategy
            tick = {"ts_ms": b.ts_ms, "symbol": b.symbol, "price": b.close, "open": b.open, "high": b.high, "low": b.low, "volume": b.volume}
            # Provide reference price to broker order proxy
            self.harness._order_proxy_mark = b.close  # type: ignore # not used directly but you can extend if needed
            # call on_tick
            self.strategy.on_tick(tick)

            # mirror (optional)
            self._mirror("prices.bars", {"ts_ms": b.ts_ms, "symbol": b.symbol, "close": b.close})

            # flush any news up to now
            while i_news < len(news) and news[i_news].ts_ms <= b.ts_ms:
                ev = news[i_news]
                i_news += 1
                if hasattr(self.strategy, "on_news"):
                    try:
                        self.strategy.on_news(asdict(ev))  # type: ignore
                    except Exception:
                        pass
                self._mirror("features.alt.news", asdict(ev))

            # end-of-bar accounting once per timestamp bucket
            bucket = b.ts_ms
            if last_bucket != bucket:
                equity, unreal, gross = self.book.mtm(prices)
                self._curve.append({"ts_ms": bucket, "equity": equity, "unreal": unreal, "gross": gross, "cash": self.book.cash, "realized": self.book.realized, "fees": self.book.fees})
                snap = {"ts_ms": bucket, "positions": [{"symbol": s, "qty": p.qty, "avg_px": p.avg_px} for s,p in self.book.pos.items() if abs(p.qty) > 1e-9], "prices": dict(prices)}
                self._pos_snap.append(snap)
                self._mirror("positions.snapshots", snap)
                last_bucket = bucket

        # finalize
        if hasattr(self.strategy, "on_stop"):
            try:
                self.strategy.on_stop()
            except Exception:
                pass

        # summary
        eq = [row["equity"] for row in self._curve]
        if eq:
            rtns = [0.0] + [ (eq[i] - eq[i-1]) for i in range(1, len(eq)) ]
            cum_pnl = eq[-1] - eq[0]
            vol = float(np.std(rtns, ddof=0)) if (np is not None and len(rtns) > 2) else 0.0
            peak = -1e18; dd = 0.0
            for v in eq:
                if v > peak: peak = v
                dd = min(dd, v - peak)
            sharpe = (np.mean(rtns)/vol) if (np is not None and vol > 1e-12) else 0.0
        else:
            cum_pnl, vol, dd, sharpe = 0.0, 0.0, 0.0, 0.0

        summary = {
            "trades": len(self._blotter),
            "final_equity": eq[-1] if eq else self.book.cash,
            "pnl_total": round(cum_pnl, 6),
            "fees": round(self.book.fees, 6),
            "vol": round(vol, 6),
            "max_drawdown": round(dd, 6),
            "sharpe_like": float(sharpe) if isinstance(sharpe, (int, float)) else 0.0,
        }
        return Result(summary=summary, equity_curve=self._curve, blotter=self._blotter, positions=self._pos_snap)

# ---- CLI --------------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("backtester")
    ap.add_argument("--bars", type=str, required=True, help="Path to bars CSV/JSON/JSONL with columns ts_ms,symbol,open,high,low,close,volume")
    ap.add_argument("--news", type=str, default=None, help="Optional news JSON/JSONL (ts_ms,title,symbols[],score)")
    ap.add_argument("--strategy", type=str, default="backend.engine.strategy_base:ExampleBuyTheDip", help="Import path 'pkg.module:Class'")
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--fee-bps", type=float, default=0.5)
    ap.add_argument("--spread-bps", type=float, default=1.0)
    ap.add_argument("--impact-bps", type=float, default=0.0)
    ap.add_argument("--latency-ms", type=int, default=5)
    ap.add_argument("--redis-mirror", action="store_true", help="Mirror events to Redis streams for your dashboards")
    args = ap.parse_args()

    strat = load_strategy(args.strategy)
    bt = Backtester(
        strat,
        capital_base=args.capital,
        cost=CostModel(fee_bps=args.fee_bps, spread_bps=args.spread_bps, impact_bps=args.impact_bps),
        latency_ms=args.latency_ms,
        redis_mirror=args.redis_mirror
    )
    bars = load_bars(args.bars)
    news = load_news(args.news)
    res = bt.run(bars, news=news)
    print(json.dumps(res.summary, indent=2))
    # write artifacts
    os.makedirs("artifacts/backtests", exist_ok=True)
    base = os.path.join("artifacts/backtests", f"bt_{int(time.time())}")
    with open(base + "_curve.json", "w") as f:
        json.dump(res.equity_curve, f)
    with open(base + "_blotter.json", "w") as f:
        json.dump(res.blotter, f)
    with open(base + "_positions.json", "w") as f:
        json.dump(res.positions, f)
    print("Artifacts:", base + "_{curve,blotter,positions}.json")

if __name__ == "__main__":
    _cli()