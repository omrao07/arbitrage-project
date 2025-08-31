# backend/sim/digital_twin.py
from __future__ import annotations

import os, csv, json, math, time, asyncio, random, contextlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable

# ---------------- Optional deps (graceful) -----------------------------------
HAVE_NUMPY = True
try:
    import numpy as np
except Exception:
    HAVE_NUMPY = False
    np = None  # type: ignore

HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

# ---------------- Env / Streams ---------------------------------------------
REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_BARS   = os.getenv("PRICES_STREAM", "prices.bars")
STREAM_ORD_IN = os.getenv("ORDERS_INCOMING", "orders.incoming")
STREAM_FILLS  = os.getenv("ORDERS_FILLED", "orders.filled")
STREAM_REJ    = os.getenv("ORDERS_REJECTED", "orders.rejected")
STREAM_POS    = os.getenv("POS_SNAPSHOTS", "positions.snapshots")
STREAM_ALERTS = os.getenv("ALERTS_STREAM", "alerts.events")

MAXLEN        = int(os.getenv("SIM_STREAM_MAXLEN", "8000"))

# ---------------- Utils ------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def jload_any(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line=line.strip()
                if line: rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    # CSV
    rows=[]
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

# ---------------- Core datatypes --------------------------------------------
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
class Order:
    id: str
    ts_ms: int
    strategy: str
    symbol: str
    side: str        # 'buy'|'sell'
    qty: float
    typ: str         # 'market'|'limit'
    limit_price: Optional[float] = None
    venue: Optional[str] = "SIM"

@dataclass
class Fill:
    ts_ms: int
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    venue: str = "SIM"

@dataclass
class Rejection:
    ts_ms: int
    order_id: str
    symbol: str
    reason: str
    venue: str = "SIM"

@dataclass
class VenueProfile:
    name: str = "SIM"
    base_latency_ms: int = 5
    jitter_ms: int = 3
    fee_bps: float = 0.3
    half_spread_bps: float = 0.8
    impact_bps: float = 0.0
    outage_prob: float = 0.0          # per order
    reject_prob: float = 0.0
    max_qty: float = 1e9

# ---------------- Publisher (Redis mirror) -----------------------------------
class Bus:
    def __init__(self, url: str = REDIS_URL):
        self.url = url
        self.r: Optional[AsyncRedis] = None # type: ignore

    async def connect(self):
        if not HAVE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(self.url, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def xadd(self, stream: str, obj: Dict[str, Any]):
        if not self.r:
            # graceful: print
            return
        try:
            await self.r.xadd(stream, {"json": json.dumps(obj, ensure_ascii=False)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

# ---------------- Microstructure (toy LOB) -----------------------------------
class LOBSim:
    """
    Minimal limit order book simulator for mark/limit fills.
    """
    def __init__(self, half_spread_bps: float, fee_bps: float, impact_bps: float):
        self.half_spread_bps = float(half_spread_bps)
        self.fee_bps = float(fee_bps)
        self.impact_bps = float(impact_bps)

    def quote(self, mid: float) -> Tuple[float, float]:
        sp = self.half_spread_bps / 1e4 * mid
        return (mid - sp, mid + sp)

    def market_fill(self, side: str, mid: float, qty: float) -> Tuple[float, float]:
        # cross spread + (tiny) impact proportional to qty
        bid, ask = self.quote(mid)
        px = ask if side == "buy" else bid
        px += (self.impact_bps / 1e4) * mid * (1 if side == "buy" else -1) * min(1.0, abs(qty) / 1e4)
        fee = abs(px * qty) * (self.fee_bps / 1e4)
        return float(px), float(fee)

    def limit_fill(self, side: str, mid: float, limit_px: float) -> Optional[Tuple[float, float]]:
        bid, ask = self.quote(mid)
        if side == "buy" and limit_px >= ask:
            fee = abs(limit_px) * (self.fee_bps / 1e4)
            return float(limit_px), float(fee)
        if side == "sell" and limit_px <= bid:
            fee = abs(limit_px) * (self.fee_bps / 1e4)
            return float(limit_px), float(fee)
        return None

# ---------------- Twin market / broker --------------------------------------
class TwinMarket:
    def __init__(self, venue: VenueProfile):
        self.venue = venue
        self.lob = LOBSim(half_spread_bps=venue.half_spread_bps, fee_bps=venue.fee_bps, impact_bps=venue.impact_bps)
        self.prices: Dict[str, float] = {}  # mid per symbol

    def set_price(self, symbol: str, mid: float):
        self.prices[symbol] = float(mid)

    async def execute(self, o: Order) -> Tuple[Optional[Fill], Optional[Rejection], int]:
        # Chaos / outages
        if random.random() < self.venue.outage_prob:
            return None, Rejection(ts_ms=now_ms(), order_id=o.id, symbol=o.symbol, reason="VENUE_OUTAGE", venue=self.venue.name), self.venue.base_latency_ms
        if random.random() < self.venue.reject_prob:
            return None, Rejection(ts_ms=now_ms(), order_id=o.id, symbol=o.symbol, reason="RANDOM_REJECT", venue=self.venue.name), self.venue.base_latency_ms

        mid = self.prices.get(o.symbol)
        if mid is None or mid <= 0:
            return None, Rejection(ts_ms=now_ms(), order_id=o.id, symbol=o.symbol, reason="NO_MARK", venue=self.venue.name), self.venue.base_latency_ms

        if abs(o.qty) > self.venue.max_qty:
            return None, Rejection(ts_ms=now_ms(), order_id=o.id, symbol=o.symbol, reason="QTY_LIMIT", venue=self.venue.name), self.venue.base_latency_ms

        # Fill logic
        if o.typ == "limit" and o.limit_price is not None:
            got = self.lob.limit_fill(o.side, mid, float(o.limit_price))
            if got:
                px, fee = got
            else:
                return None, None, self._latency()
        else:
            px, fee = self.lob.market_fill(o.side, mid, o.qty)

        f = Fill(ts_ms=now_ms(), order_id=o.id, symbol=o.symbol, side=o.side, qty=o.qty, price=px, fee=fee, venue=self.venue.name)
        return f, None, self._latency()

    def _latency(self) -> int:
        base = self.venue.base_latency_ms
        jit  = self.venue.jitter_ms
        return max(0, int(base + random.randint(-jit, jit)))

# ---------------- Agents & book ---------------------------------------------
@dataclass
class Position:
    qty: float = 0.0
    avg_px: float = 0.0

class TwinBook:
    def __init__(self, capital_base: float = 100_000.0):
        self.cash = float(capital_base)
        self.pos: Dict[str, Position] = {}
        self.fees = 0.0
        self.realized = 0.0

    def on_fill(self, f: Fill):
        p = self.pos.setdefault(f.symbol, Position())
        sgn = 1.0 if f.side == "buy" else -1.0
        notional = sgn * f.qty * f.price
        self.cash -= notional
        self.fees += f.fee
        new_qty = p.qty + sgn * f.qty
        if p.qty != 0 and (p.qty * new_qty) < 0:
            crossing = min(abs(p.qty), f.qty)
            self.realized += crossing * (p.avg_px - f.price) * (1 if p.qty < 0 else -1)
        if new_qty == 0:
            p.qty, p.avg_px = 0.0, 0.0
        elif sgn > 0:
            p.avg_px = (p.avg_px * p.qty + f.qty * f.price) / (p.qty + f.qty if p.qty + f.qty != 0 else 1.0)
            p.qty = new_qty
        else:
            p.qty = new_qty

    def equity(self, prices: Dict[str, float]) -> float:
        unreal = 0.0
        for s, p in self.pos.items():
            px = float(prices.get(s, p.avg_px or 0.0))
            unreal += (px - p.avg_px) * p.qty
        return self.cash + self.realized + unreal - self.fees

class Agent:
    """
    Plug-in agent. Provide callables for events you care about.
    """
    def __init__(
        self,
        name: str,
        on_bar: Optional[Callable[[Dict[str, Any], Callable[..., None]], None]] = None,
        on_news: Optional[Callable[[Dict[str, Any], Callable[..., None]], None]] = None
    ):
        self.name = name
        self.on_bar = on_bar
        self.on_news = on_news

# ---------------- Digital twin orchestrator ----------------------------------
@dataclass
class TwinConfig:
    mode: str = "replay"            # 'replay' | 'synthetic'
    bars_path: Optional[str] = None
    news_path: Optional[str] = None
    capital_base: float = 100_000.0
    venue: VenueProfile = field(default_factory=VenueProfile)
    publish_to_redis: bool = True
    bar_interval_ms: int = 1000     # synthetic cadence
    symbols: List[str] = field(default_factory=lambda: ["AAPL","MSFT"])
    drift_bps_per_bar: float = 0.0
    vol_bps_per_sqrtbar: float = 5.0
    shock_prob: float = 0.0
    shock_bps: float = 50.0

class DigitalTwin:
    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self.market = TwinMarket(cfg.venue)
        self.book = TwinBook(cfg.capital_base)
        self.bus = Bus()
        self.agents: List[Agent] = []
        self._oid = 0
        self._running = False
        self._prices: Dict[str, float] = {}

    def register_agent(self, agent: Agent):
        self.agents.append(agent)

    def _next_order_id(self) -> str:
        self._oid += 1
        return f"twin-{self._oid:08d}"

    async def start(self):
        if self.cfg.publish_to_redis:
            await self.bus.connect()
        self._running = True
        if self.cfg.mode == "replay":
            await self._loop_replay()
        else:
            await self._loop_synthetic()

    async def stop(self):
        self._running = False

    # ---- Order entry for agents --------------------------------------------
    async def submit(self, *, symbol: str, side: str, qty: float, typ: str = "market", limit_price: Optional[float] = None, venue: Optional[str] = None, strategy: str = "agent"):  # noqa: E501
        o = Order(
            id=self._next_order_id(),
            ts_ms=now_ms(),
            strategy=strategy,
            symbol=symbol.upper(),
            side=side.lower(),
            qty=float(qty),
            typ=typ,
            limit_price=limit_price,
            venue=venue or self.cfg.venue.name
        )
        # Execute on venue
        fill, rej, lat = await self.market.execute(o)
        await asyncio.sleep(lat / 1000.0)
        if fill:
            self.book.on_fill(fill)
            await self.bus.xadd(STREAM_FILLS, asdict(fill))
        if rej:
            await self.bus.xadd(STREAM_REJ, asdict(rej))
        return fill, rej

    # ---- Replay loop --------------------------------------------------------
    async def _loop_replay(self):
        if not self.cfg.bars_path:
            raise RuntimeError("Twin replay mode requires bars_path")
        bars_raw = jload_any(self.cfg.bars_path)
        bars: List[Bar] = []
        for r in bars_raw:
            bars.append(Bar(
                ts_ms=int(r.get("ts_ms") or r.get("timestamp") or r.get("time") or 0),
                symbol=str(r.get("symbol") or r.get("sym") or "").upper(),
                open=float(r.get("open") or r.get("o") or r.get("close")), # type: ignore
                high=float(r.get("high") or r.get("h") or r.get("close")), # type: ignore
                low=float(r.get("low") or r.get("l") or r.get("close")), # type: ignore
                close=float(r.get("close") or r.get("c") or r.get("price")), # type: ignore
                volume=float(r.get("volume") or r.get("v") or 0.0),
            ))
        bars.sort(key=lambda b: (b.ts_ms, b.symbol))
        news = jload_any(self.cfg.news_path) if self.cfg.news_path else []
        i_news = 0

        for b in bars:
            if not self._running: break
            self._prices[b.symbol] = b.close
            self.market.set_price(b.symbol, b.close)
            await self._emit_bar(b)
            # flush news up to now
            while i_news < len(news) and int(news[i_news].get("ts_ms", 0)) <= b.ts_ms:
                await self._emit_news(news[i_news]); i_news += 1

        await self._snapshot_positions()

    # ---- Synthetic loop -----------------------------------------------------
    async def _loop_synthetic(self):
        # init prices
        for s in self.cfg.symbols:
            self._prices[s] = self._prices.get(s, 100.0)
            self.market.set_price(s, self._prices[s])
        step = self.cfg.bar_interval_ms
        while self._running:
            t0 = now_ms()
            for s in self.cfg.symbols:
                mid = self._prices[s]
                # GBM-ish update
                if HAVE_NUMPY:
                    z = float(np.random.normal()) # type: ignore
                else:
                    z = random.gauss(0.0, 1.0)
                drift = self.cfg.drift_bps_per_bar / 1e4
                vol = self.cfg.vol_bps_per_sqrtbar / 1e4
                mid *= math.exp((drift - 0.5 * vol * vol) + vol * z)
                # occasional shock
                if random.random() < self.cfg.shock_prob:
                    shock = (self.cfg.shock_bps / 1e4)
                    mid *= (1.0 + (shock if random.random() < 0.5 else -shock))
                self._prices[s] = max(0.01, mid)
                self.market.set_price(s, self._prices[s])
                b = Bar(ts_ms=t0, symbol=s, open=mid, high=mid, low=mid, close=mid, volume=0.0)
                await self._emit_bar(b)
            await asyncio.sleep(max(0.0, step/1000.0))

    # ---- Emitters -----------------------------------------------------------
    async def _emit_bar(self, b: Bar):
        await self.bus.xadd(STREAM_BARS, {"ts_ms": b.ts_ms, "symbol": b.symbol, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume})
        # notify agents
        for a in self.agents:
            if a.on_bar:
                try:
                    await maybe_await(a.on_bar(asdict(b), self._agent_order(a.name)))
                except Exception as e:
                    await self.bus.xadd(STREAM_ALERTS, {"ts_ms": now_ms(), "level":"error","msg":f"agent {a.name} on_bar: {e}"})

        # positions snapshot (coarse)
        await self._snapshot_positions()

    async def _emit_news(self, ev: Dict[str, Any]):
        await self.bus.xadd("features.alt.news", ev)
        for a in self.agents:
            if a.on_news:
                try:
                    await maybe_await(a.on_news(ev, self._agent_order(a.name)))
                except Exception as e:
                    await self.bus.xadd(STREAM_ALERTS, {"ts_ms": now_ms(), "level":"error","msg":f"agent {a.name} on_news: {e}"})

    async def _snapshot_positions(self):
        snap = {
            "ts_ms": now_ms(),
            "positions": [{"symbol": s, "qty": p.qty, "avg_px": p.avg_px} for s,p in self.book.pos.items() if abs(p.qty) > 1e-9],
            "prices": dict(self._prices)
        }
        await self.bus.xadd(STREAM_POS, snap)

    # ---- Agent order handle --------------------------------------------------
    def _agent_order(self, strategy_name: str) -> Callable[..., Any]:
        async def _submit(symbol: str, side: str, qty: float, *, typ: str = "market", limit_price: Optional[float] = None, venue: Optional[str] = None):  # noqa: E501
            return await self.submit(symbol=symbol, side=side, qty=qty, typ=typ, limit_price=limit_price, venue=venue, strategy=strategy_name)
        return _submit

# ---------------- Helpers ----------------------------------------------------
async def maybe_await(x):
    if asyncio.iscoroutine(x):
        return await x
    return x

# ---------------- CLI --------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("digital_twin")
    ap.add_argument("--mode", choices=["replay","synthetic"], default="synthetic")
    ap.add_argument("--bars", type=str, default=None, help="CSV/JSON/JSONL bars (replay)")
    ap.add_argument("--news", type=str, default=None, help="JSON/JSONL news (replay)")
    ap.add_argument("--symbols", type=str, default="AAPL,MSFT", help="Comma symbols (synthetic)")
    ap.add_argument("--interval-ms", type=int, default=500)
    ap.add_argument("--venue-latency", type=int, default=5)
    ap.add_argument("--venue-jitter", type=int, default=3)
    ap.add_argument("--half-spread-bps", type=float, default=0.8)
    ap.add_argument("--fee-bps", type=float, default=0.3)
    ap.add_argument("--impact-bps", type=float, default=0.0)
    ap.add_argument("--outage-prob", type=float, default=0.0)
    ap.add_argument("--reject-prob", type=float, default=0.0)
    args = ap.parse_args()

    cfg = TwinConfig(
        mode=args.mode,
        bars_path=args.bars,
        news_path=args.news,
        bar_interval_ms=args.interval_ms,
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        venue=VenueProfile(
            base_latency_ms=args.venue_latency,
            jitter_ms=args.venue_jitter,
            fee_bps=args.fee_bps,
            half_spread_bps=args.half_spread_bps,
            impact_bps=args.impact_bps,
            outage_prob=args.outage_prob,
            reject_prob=args.reject_prob,
            name="SIM"
        )
    )

    async def _run():
        twin = DigitalTwin(cfg)

        # Example agent: momentum tick scalper
        async def on_bar(bar: Dict[str, Any], order: Callable[..., Any]):
            # trivial: buy if price ended with .00-.25 range, sell if .75-.99 (toy)
            px = float(bar["close"]); sym = bar["symbol"]
            frac = (px - math.floor(px))
            if frac < 0.25: await order(sym, "buy", 1.0, typ="market")
            elif frac > 0.75: await order(sym, "sell", 1.0, typ="market")

        twin.register_agent(Agent("toy_scalper", on_bar=on_bar)) # type: ignore
        try:
            await twin.start()
        except KeyboardInterrupt:
            await twin.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()