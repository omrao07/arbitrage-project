# backend/backtest/backtester.py
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Literal

# ===== Try to use your shared schemas; else provide minimal shims =====
try:
    from backend.common.schemas import ( # type: ignore
        Candle, TradeTick, Quote, OrderIntent, ParentOrder, ChildOrder, Fill,
        ExecutionReport, Position, PortfolioSnapshot, LedgerEvent, VenueWeight
    )
except Exception:  # minimal local shims so this file stays drop-in
    @dataclass
    class Candle:
        symbol: str; ts_ms: int; open: float; high: float; low: float; close: float
        volume: Optional[float] = None; interval: Optional[str] = None
    @dataclass
    class TradeTick:
        symbol: str; ts_ms: int; price: float; size: float
    @dataclass
    class Quote:
        symbol: str; ts_ms: int; bid: Optional[float]=None; ask: Optional[float]=None
        bid_size: Optional[float]=None; ask_size: Optional[float]=None; venue: Optional[str]=None
        def mid(self) -> Optional[float]:
            if self.bid is not None and self.ask is not None and self.ask >= self.bid:
                return 0.5*(self.bid+self.ask)
            return None
    Side = Literal["buy","sell"]
    TIF = Literal["DAY","IOC","FOK"]
    @dataclass
    class OrderIntent:
        symbol: str; side: Side; qty: float
        urgency: Literal["low","normal","high"]="normal"
        limit_price: Optional[float]=None; tif: Optional[TIF]=None
        venue_hint: Optional[str]=None; participation_cap: Optional[float]=None
        strategy: Optional[str]=None; meta: Dict[str,Any]=field(default_factory=dict)
    @dataclass
    class ParentOrder:
        order_id: str; symbol: str; side: Side; qty: float
        created_ms: int; limit_price: Optional[float]=None; tif: TIF="DAY"
        status: Literal["live","done","canceled","rejected"]="live"
        tags: List[str]=field(default_factory=list); meta: Dict[str,Any]=field(default_factory=dict)
    @dataclass
    class ChildOrder:
        child_id: str; parent_id: str; symbol: str; side: Side; qty: float
        created_ms: int; venue: Optional[str]=None; algo: Optional[str]=None
        limit_price: Optional[float]=None; tif: TIF="DAY"
        status: Literal["live","done","canceled","rejected"]="live"; meta: Dict[str,Any]=field(default_factory=dict)
    @dataclass
    class Fill:
        child_id: str; parent_id: str; symbol: str; ts_ms: int; price: float; qty: float
        venue: Optional[str]=None; fee_bps: Optional[float]=None; liquidity_flag: Optional[Literal["add","remove"]]=None
    @dataclass
    class ExecutionReport:
        parent_id: str; ts_ms: int; status: Literal["partial","filled","canceled","rejected"]
        filled_qty: float; avg_fill_price: Optional[float]=None; remaining_qty: Optional[float]=None
        reason: Optional[str]=None; meta: Dict[str,Any]=field(default_factory=dict)
    @dataclass
    class Position:
        symbol: str; qty: float=0.0; avg_price: float=0.0; last_price: float=0.0
        def notional(self)->float: return abs(self.qty)*(self.last_price or self.avg_price or 0.0)
    @dataclass
    class PortfolioSnapshot:
        cash: float; nav: float; leverage: float; positions: Dict[str,Position]=field(default_factory=dict)
        adv: Dict[str,float]=field(default_factory=dict); spread_bps: Dict[str,float]=field(default_factory=dict)
        symbol_weights: Dict[str,float]=field(default_factory=dict); var_1d_frac: Optional[float]=None
        drawdown_frac: Optional[float]=None; ts_ms: int=0
    @dataclass
    class LedgerEvent:
        kind: str; ts_ms: int; payload: Dict[str,Any]=field(default_factory=dict); ref_id: Optional[str]=None; actor: Optional[str]=None
        @staticmethod
        def wrap(obj: Any, kind: Optional[str]=None, ref_id: Optional[str]=None, actor: Optional[str]=None) -> "LedgerEvent":
            return LedgerEvent(kind or obj.__class__.__name__, ts_ms=getattr(obj,"ts_ms",0), payload=asdict(obj), ref_id=ref_id, actor=actor)
    @dataclass
    class VenueWeight:
        venue: str; weight: float; dark: bool=False

# ===== Small utilities =====
def sign(side: str) -> int: return 1 if side == "buy" else -1
def clamp(x: float, a: float, b: float) -> float: return max(a, min(b, x))

def ms_to_day(ts_ms: int) -> int: return ts_ms // 86_400_000

# ===== Sim clock & events =====
@dataclass(order=True)
class _SchedItem:
    ts_ms: int
    seq: int
    payload: Any=field(compare=False)

@dataclass
class Event:
    kind: Literal["trade","candle","quote","timer"]
    ts_ms: int
    data: Any

class SimClock:
    def __init__(self):
        self.t: int = 0
        self._q: List[_SchedItem] = []
        self._seq: int = 0

    def set(self, t_ms: int) -> None: self.t = int(t_ms)

    def schedule(self, ts_ms: int, payload: Any) -> None:
        heapq.heappush(self._q, _SchedItem(int(ts_ms), self._seq, payload)); self._seq += 1

    def step(self) -> Optional[Any]:
        if not self._q: return None
        item = heapq.heappop(self._q)
        self.t = item.ts_ms
        return item.payload

    def has_events(self) -> bool: return bool(self._q)

# ===== Data feed (merge candles/trades/quotes) =====
class DataFeed:
    def __init__(self, *, candles: Iterable[Candle] | None = None,
                 trades: Iterable[TradeTick] | None = None,
                 quotes: Iterable[Quote] | None = None):
        self._iters: List[Iterator[Tuple[int, Event]]] = []
        if candles: self._iters.append(self._wrap("candle", candles))
        if trades:  self._iters.append(self._wrap("trade", trades))
        if quotes:  self._iters.append(self._wrap("quote", quotes))

    def _wrap(self, kind: str, seq: Iterable[Any]) -> Iterator[Tuple[int, Event]]:
        for obj in seq:
            ts = int(getattr(obj, "ts_ms"))
            yield ts, Event(kind=kind, ts_ms=ts, data=obj) # type: ignore

    def schedule_into(self, clock: SimClock) -> None:
        # merge by timestamp with a small seq to stabilize ordering
        buf: List[Tuple[int, int, Event]] = []
        seq = 0
        for it in self._iters:
            for ts, ev in it:
                buf.append((ts, seq, ev)); seq += 1
        buf.sort(key=lambda x: (x[0], x[1]))
        for ts, _, ev in buf:
            clock.schedule(ts, ev)

# ===== Execution & cost models =====
@dataclass
class CommissionModel:
    per_share: float = 0.0          # e.g., $0.002/share
    min_fee: float = 0.0
    bps: float = 0.0                # basis points of notional

    def fees(self, price: float, qty: float) -> float:
        notional = abs(price * qty)
        f = self.per_share * abs(qty) + self.bps/1e4 * notional
        return max(self.min_fee, f)

@dataclass
class SlippageModel:
    # simple linear model in spread + participation
    k_spread: float = 0.5           # fill at mid +/- k * half-spread (0..1)
    k_participation: float = 0.1    # extra slippage per 10% ADV participation

    def slip_bps(self, half_spread_bps: float, participation: float) -> float:
        return max(0.0, self.k_spread * half_spread_bps + self.k_participation * 100.0 * participation)

@dataclass
class LatencyModel:
    ms: int = 50
    jitter_ms: int = 10
    def delay(self) -> int:
        if self.jitter_ms <= 0: return self.ms
        lo = max(0, self.ms - self.jitter_ms)
        hi = self.ms + self.jitter_ms
        return random.randint(lo, hi)

@dataclass
class ExecConfig:
    commission: CommissionModel = field(default_factory=CommissionModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    latency: LatencyModel = field(default_factory=LatencyModel)
    adv_map: Dict[str, float] = field(default_factory=dict)  # shares/day for participation calc

# ===== Portfolio Accounting =====
class Portfolio:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.positions: Dict[str, Position] = {}
        self._nav_peak: float = cash
        self.nav: float = cash

    def mark(self, symbol: str, price: float) -> None:
        p = self.positions.get(symbol)
        if p:
            p.last_price = price
            self._recalc_nav()

    def on_fill(self, fill: Fill, fees: float) -> None:
        side_sgn = 1 if fill.qty > 0 else -1
        qty = fill.qty
        px = fill.price
        pos = self.positions.get(fill.symbol) or Position(symbol=fill.symbol, qty=0.0, avg_price=0.0, last_price=px)
        # cash: buy -> spend, sell -> receive
        self.cash -= qty * px + fees * (1 if side_sgn > 0 else 1)  # fees always paid from cash
        # position update (classic WAC)
        new_qty = pos.qty + qty
        if abs(new_qty) < 1e-12:
            pos.qty = 0.0; pos.avg_price = 0.0
        elif pos.qty * new_qty > 0:  # same direction -> average
            pos.avg_price = (pos.avg_price * abs(pos.qty) + px * abs(qty)) / abs(new_qty)
            pos.qty = new_qty
        else:
            # crossed through or reduced; avg_price stays if not crossing sign
            pos.qty = new_qty
            if pos.qty == 0:
                pos.avg_price = 0.0
        pos.last_price = px
        self.positions[fill.symbol] = pos
        self._recalc_nav()

    def snapshot(self, ts_ms: int) -> PortfolioSnapshot:
        lev = self._gross_notional() / max(1e-9, self.nav)
        return PortfolioSnapshot(
            cash=self.cash, nav=self.nav, leverage=lev,
            positions=self.positions.copy(), ts_ms=ts_ms
        )

    # internals
    def _gross_notional(self) -> float:
        return sum(abs(p.qty) * (p.last_price or p.avg_price) for p in self.positions.values())

    def _recalc_nav(self) -> None:
        holdings = sum((p.last_price or p.avg_price) * p.qty for p in self.positions.values())
        self.nav = self.cash + holdings
        self._nav_peak = max(self._nav_peak, self.nav)

# ===== Orders & fills inside simulator =====
@dataclass
class SimOrder:
    parent: ParentOrder
    remaining: float
    venue: Optional[str] = None

class ExecutionModel:
    """
    Simple matching against mid/quotes or last trade with slippage and commission.
    Supports market & limit (DAY/IOC). No short-borrow constraints (you can extend).
    """
    def __init__(self, clock: SimClock, portfolio: Portfolio, cfg: ExecConfig):
        self.clock = clock
        self.pf = portfolio
        self.cfg = cfg
        self.live: Dict[str, SimOrder] = {}  # parent_id -> SimOrder
        self.fills_out: List[Fill] = []
        self.reports_out: List[ExecutionReport] = []
        self.last_quote: Dict[str, Quote] = {}
        self.last_trade: Dict[str, TradeTick] = {}
        self._id = 0

    def _gen_id(self) -> str:
        self._id += 1
        return f"ord_{self._id}"

    # ---- order entry from strategy ----
    def submit(self, intent: OrderIntent, ts_ms: int) -> ParentOrder:
        pid = self._gen_id()
        po = ParentOrder(order_id=pid, symbol=intent.symbol, side=intent.side,
                         qty=float(intent.qty), created_ms=ts_ms,
                         limit_price=intent.limit_price, tif=intent.tif or "DAY",
                         meta=intent.meta)
        self.live[pid] = SimOrder(parent=po, remaining=float(intent.qty), venue=intent.venue_hint)
        # schedule an attempt after latency
        self.clock.schedule(ts_ms + self.cfg.latency.delay(), ("try_fill", pid))
        return po

    # ---- market data hooks ----
    def on_quote(self, q: Quote) -> None:
        self.last_quote[q.symbol] = q

    def on_trade(self, t: TradeTick) -> None:
        self.last_trade[t.symbol] = t

    # ---- internal fill attempt ----
    def try_fill(self, pid: str, ts_ms: int) -> None:
        so = self.live.get(pid)
        if not so: return
        sym = so.parent.symbol
        side = so.parent.side
        rem = so.remaining
        if rem <= 0: return

        # choose reference price
        ref_px = None
        half_spread_bps = 0.0
        q = self.last_quote.get(sym)
        if q and q.bid is not None and q.ask is not None and q.ask >= q.bid:
            ref_px = (q.bid + q.ask) * 0.5
            half_spread_bps = (q.ask - q.bid) / ref_px * 1e4 * 0.5
            # enforce limit side vs L1 if limit order
            if so.parent.limit_price is not None:
                if side == "buy" and so.parent.limit_price < (q.ask or ref_px):
                    # not marketable; reschedule on next event
                    self.clock.schedule(ts_ms + 25, ("try_fill", pid))
                    return
                if side == "sell" and so.parent.limit_price > (q.bid or ref_px):
                    self.clock.schedule(ts_ms + 25, ("try_fill", pid))
                    return
        else:
            t = self.last_trade.get(sym)
            if t:
                ref_px = t.price
            # if neither quote nor trade -> no fill yet
            if ref_px is None:
                self.clock.schedule(ts_ms + 25, ("try_fill", pid))
                return

        # compute participation proxy
        adv = self.cfg.adv_map.get(sym, 0.0)
        participation = clamp(abs(rem) / max(1.0, adv), 0.0, 1.0)

        # slippage (bps)
        slip_bps = self.cfg.slippage.slip_bps(half_spread_bps, participation)
        slip = slip_bps / 1e4 * ref_px
        fill_px = ref_px + (slip if side == "buy" else -slip)

        # IOC/partial fill logic (simple): fill all remaining
        fill_qty = rem
        so.remaining = 0.0
        so.parent.status = "done"

        # commission
        fee = self.cfg.commission.fees(fill_px, fill_qty)

        # record fill
        child_id = f"{pid}.1"
        fill = Fill(child_id=child_id, parent_id=pid, symbol=sym, ts_ms=ts_ms,
                    price=fill_px, qty=(+abs(fill_qty) if side=="buy" else -abs(fill_qty)),
                    venue=so.venue or "SIM", fee_bps=None, liquidity_flag=None)
        self.fills_out.append(fill)
        self.pf.on_fill(fill, fee)

        self.reports_out.append(ExecutionReport(parent_id=pid, ts_ms=ts_ms, status="filled",
                                                filled_qty=abs(fill_qty), avg_fill_price=fill_px,
                                                remaining_qty=0.0))

        # remove from live
        self.live.pop(pid, None)

# ===== Strategy adapter =====
class StrategyAPI:
    """
    Thin adapter exposed to your strategy in backtests.
    Provides .order() and .emit() while you keep on_tick/on_bar handlers.
    """
    def __init__(self, name: str, submit_fn: Callable[[OrderIntent,int], ParentOrder]):
        self.name = name
        self._submit = submit_fn

    def order(self, symbol: str, side: str, qty: float,
              *, order_type: str = "market", limit_price: Optional[float] = None,
              tif: Optional[str] = None, meta: Optional[Dict[str,Any]] = None) -> ParentOrder:
        if order_type not in ("market","limit"):
            order_type = "market"
        return self._submit(OrderIntent(
            symbol=symbol, side=side, qty=float(qty), # type: ignore
            limit_price=(None if order_type=="market" else limit_price),
            tif=(tif or "DAY"), strategy=self.name, meta=meta or {} # type: ignore
        ), ts_ms=0)  # type: ignore # ts will be overridden by engine at submit time

# ===== Backtest engine =====
@dataclass
class BacktestResult:
    logs: List[LedgerEvent]
    equity_curve: List[Tuple[int,float]]
    fills: List[Fill]
    reports: List[ExecutionReport]
    final_snapshot: PortfolioSnapshot

class Backtester:
    """
    Event-driven backtester:
      • Merge data (trades/candles/quotes) into a SimClock
      • Drive your strategy callbacks: on_tick(TradeTick) and/or on_bar(Candle)
      • Orders -> ExecutionModel (latency/slippage/fees)
      • Portfolio marked on quotes/trades/candles close
    """
    def __init__(self, *, initial_cash: float = 1_000_000.0, exec_cfg: Optional[ExecConfig]=None):
        self.clock = SimClock()
        self.portfolio = Portfolio(initial_cash)
        self.exec = ExecutionModel(self.clock, self.portfolio, exec_cfg or ExecConfig())
        self.logs: List[LedgerEvent] = []
        self.equity_curve: List[Tuple[int,float]] = []
        self._strategy_handlers: Dict[str, Dict[str, Callable[[Any], None]]] = {}
        self._strategy_api: Dict[str, StrategyAPI] = {}

    # ---- data wiring ----
    def load_data(self, feed: DataFeed) -> None:
        feed.schedule_into(self.clock)

    # ---- strategy wiring ----
    def attach_strategy(self, name: str,
                        on_tick: Optional[Callable[[TradeTick, StrategyAPI], None]] = None,
                        on_bar: Optional[Callable[[Candle, StrategyAPI], None]] = None,
                        on_quote: Optional[Callable[[Quote, StrategyAPI], None]] = None) -> StrategyAPI:
        api = StrategyAPI(name, submit_fn=self._submit_with_ts)
        self._strategy_api[name] = api
        self._strategy_handlers[name] = {"tick": (on_tick or (lambda *_: None)), # type: ignore
                                         "bar":  (on_bar or (lambda *_: None)),
                                         "quote":(on_quote or (lambda *_: None))}
        return api

    def _submit_with_ts(self, intent: OrderIntent, ts_ms: int) -> ParentOrder:
        # called by StrategyAPI; ts_ms is injected by run loop
        intent.meta = {**(intent.meta or {}), "submitted_ms": self.clock.t}
        po = self.exec.submit(intent, ts_ms=self.clock.t)
        self.logs.append(LedgerEvent(kind="order.created", ts_ms=self.clock.t, payload=asdict(po), actor=intent.strategy))
        return po

    # ---- run loop ----
    def run(self) -> BacktestResult:
        # prime equity
        self.equity_curve.append((self.clock.t, self.portfolio.nav))

        while self.clock.has_events():
            item = self.clock.step()
            if isinstance(item, Event):
                self._on_event(item)
            elif isinstance(item, tuple) and item[0] == "try_fill":
                _, pid = item
                self.exec.try_fill(pid, self.clock.t)

            # collect fills/reports emitted during this tick
            while self.exec.fills_out:
                f = self.exec.fills_out.pop(0)
                self.logs.append(LedgerEvent.wrap(f, kind="order.fill", ref_id=f.parent_id))
            while self.exec.reports_out:
                r = self.exec.reports_out.pop(0)
                self.logs.append(LedgerEvent.wrap(r, kind="order.exec", ref_id=r.parent_id))

            # record equity curve at most once per millisecond state
            self.equity_curve.append((self.clock.t, self.portfolio.nav))

        snap = self.portfolio.snapshot(ts_ms=self.clock.t)
        return BacktestResult(
            logs=self.logs,
            equity_curve=self.equity_curve,
            fills=[ev.payload for ev in self.logs if ev.kind=="order.fill"],   # type: ignore
            reports=[ev.payload for ev in self.logs if ev.kind=="order.exec"], # type: ignore
            final_snapshot=snap
        )

    # ---- event handlers ----
    def _on_event(self, ev: Event) -> None:
        # forward quotes/trades to execution model for pricing
        if ev.kind == "quote":
            q: Quote = ev.data
            self.exec.on_quote(q)
            # mark-to-market on mid if available
            m = q.mid()
            if m:
                self.portfolio.mark(q.symbol, m)
            for name, handlers in self._strategy_handlers.items():
                handlers["quote"](q, self._strategy_api[name]) # type: ignore
            self.logs.append(LedgerEvent(kind="quote", ts_ms=ev.ts_ms, payload=asdict(q)))
        elif ev.kind == "trade":
            t: TradeTick = ev.data
            self.exec.on_trade(t)
            self.portfolio.mark(t.symbol, t.price)
            for name, handlers in self._strategy_handlers.items():
                handlers["tick"](t, self._strategy_api[name]) # type: ignore
            self.logs.append(LedgerEvent(kind="trade", ts_ms=ev.ts_ms, payload=asdict(t)))
        elif ev.kind == "candle":
            c: Candle = ev.data
            # mark on close; you can change to OHLC midpoint if you prefer
            self.portfolio.mark(c.symbol, c.close)
            for name, handlers in self._strategy_handlers.items():
                handlers["bar"](c, self._strategy_api[name]) # type: ignore
            self.logs.append(LedgerEvent(kind="candle", ts_ms=ev.ts_ms, payload=asdict(c)))

# ===== Convenience: quick runner with a toy strategy =====
class BuyTheDip:
    """
    Simple demo strategy working on candles:
    - EWMA of close; buy when close < avg by X bps, sell when above.
    """
    def __init__(self, symbol: str, bps: float = 10.0, default_qty: float = 10.0):
        self.symbol = symbol
        self.bps = float(bps)
        self.qty = float(default_qty)
        self.avg = None

    def on_bar(self, bar: Candle, api: StrategyAPI):
        if bar.symbol != self.symbol: return
        # EWMA
        self.avg = bar.close if self.avg is None else 0.98*self.avg + 0.02*bar.close
        diff_bps = (bar.close - self.avg) / self.avg * 1e4
        if diff_bps <= -self.bps:
            api.order(self.symbol, "buy", self.qty, order_type="market")
        elif diff_bps >= self.bps:
            api.order(self.symbol, "sell", self.qty, order_type="market")

def run_demo() -> None:
    # Build synthetic candles for a quick sanity test
    import time as _time, random as _rand
    now = int(_time.time()*1000)
    px = 100.0
    candles: List[Candle] = []
    for i in range(240):  # 240 mins (~4h)
        # random walk
        ch = _rand.uniform(-0.3, 0.3)
        px = max(0.01, px*(1.0 + ch/100))
        o = px*(1.0-_rand.uniform(0,0.05)/100); h=o*(1+_rand.uniform(0,0.1)/100)
        l=o*(1-_rand.uniform(0,0.1)/100); c=px
        candles.append(Candle(symbol="DEMO", ts_ms=now+i*60_000, open=o, high=h, low=l, close=c, volume=1000, interval="1m"))

    feed = DataFeed(candles=candles)
    bt = Backtester(initial_cash=100_000.0, exec_cfg=ExecConfig(
        commission=CommissionModel(per_share=0.0, bps=0.5),  # 0.5 bps all-in
        slippage=SlippageModel(k_spread=0.4, k_participation=0.05),
        latency=LatencyModel(ms=5, jitter_ms=2),
        adv_map={"DEMO": 1_000_000}
    ))
    strat = BuyTheDip("DEMO", bps=8.0, default_qty=25)
    bt.attach_strategy("buy_the_dip", on_bar=strat.on_bar)

    bt.load_data(feed)
    res = bt.run()

    # Tiny report
    print(f"Final NAV: {res.final_snapshot.nav:,.2f}")
    fills = [e for e in res.logs if e.kind=="order.fill"]
    print(f"Fills: {len(fills)}; Equity curve points: {len(res.equity_curve)}")

if __name__ == "__main__":  # pragma: no cover
    run_demo()