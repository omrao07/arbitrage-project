# backend/ops/shadow_runner.py
from __future__ import annotations

import os, time, math, json, asyncio, importlib, csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable

# ---------- Optional deps (graceful) -----------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

# ---------- Env / Streams ----------------------------------------------------
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
S_BARS         = os.getenv("PRICES_STREAM", "prices.bars")
S_NEWS         = os.getenv("NEWS_STREAM", "features.alt.news")
S_ORDERS_LIVE  = os.getenv("ORDERS_FILLED", "orders.filled")     # optional for comparison
S_SHADOW_ORD   = os.getenv("SHADOW_ORDERS", "shadow.orders")     # simulated fills
S_SHADOW_POS   = os.getenv("SHADOW_POS", "shadow.positions")
S_SHADOW_PNL   = os.getenv("SHADOW_PNL", "shadow.pnl")
S_SHADOW_EVT   = os.getenv("SHADOW_EVENTS", "shadow.events")
MAXLEN         = int(os.getenv("SHADOW_MAXLEN", "10000"))

def now_ms() -> int: return int(time.time() * 1000)

# ---------- Minimal cost & broker sim ---------------------------------------
@dataclass
class CostModel:
    fee_bps: float = 0.3
    half_spread_bps: float = 0.8
    impact_bps: float = 0.0  # per 1% ADV (toy)

    def market_fill(self, side: str, mark: float, qty: float) -> Tuple[float, float]:
        slip = (self.half_spread_bps / 1e4) * mark
        px = mark + (slip if side == "buy" else -slip)
        fee = abs(px * qty) * (self.fee_bps / 1e4)
        return float(px), float(fee)

    def limit_fill(self, side: str, mark: float, limit_px: float) -> Optional[Tuple[float,float]]:
        # fill if crossing best
        sp = (self.half_spread_bps / 1e4) * mark
        bid, ask = mark - sp, mark + sp
        if side == "buy" and limit_px >= ask:  return float(limit_px), abs(limit_px * self.fee_bps / 1e4)
        if side == "sell" and limit_px <= bid: return float(limit_px), abs(limit_px * self.fee_bps / 1e4)
        return None

@dataclass
class Position:
    qty: float = 0.0
    avg_px: float = 0.0

class ShadowBook:
    def __init__(self, capital: float = 100_000.0):
        self.cash = float(capital)
        self.pos: Dict[str, Position] = {}
        self.realized = 0.0
        self.fees = 0.0

    def on_fill(self, *, symbol: str, side: str, qty: float, price: float, fee: float):
        p = self.pos.setdefault(symbol, Position())
        sgn = 1.0 if side == "buy" else -1.0
        notional = sgn * qty * price
        self.cash -= notional
        self.fees += fee
        new_qty = p.qty + sgn * qty
        # realize if crossing through zero
        if p.qty != 0 and (p.qty * new_qty) < 0:
            crossing = min(abs(p.qty), qty)
            self.realized += crossing * (p.avg_px - price) * (1 if p.qty < 0 else -1)
        if new_qty == 0:
            p.qty, p.avg_px = 0.0, 0.0
        elif sgn > 0:  # buying
            p.avg_px = (p.avg_px * p.qty + qty * price) / (p.qty + qty if (p.qty + qty) != 0 else 1.0)
            p.qty = new_qty
        else:          # selling
            p.qty = new_qty

    def equity(self, marks: Dict[str, float]) -> float:
        unreal = 0.0
        for s, p in self.pos.items():
            px = float(marks.get(s, p.avg_px or 0.0))
            unreal += (px - p.avg_px) * p.qty
        return self.cash + self.realized + unreal - self.fees

# ---------- Strategy harness (monkey-patch .order) ---------------------------
def load_strategy(qualname: str, **kwargs):
    """
    qualname: 'pkg.module:ClassName'
    """
    mod_name, cls_name = qualname.split(":")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)

class StrategyHarness:
    """
    Intercepts Strategy.order(...) and emits simulated fills using latest marks.
    """
    def __init__(self, strategy: "Strategy", book: ShadowBook, cost: CostModel, publish: Callable[[str, Dict[str,Any]], None], default_qty: float = 1.0, venue_latency_ms: int = 5): # type: ignore
        self.strategy = strategy
        self.book = book
        self.cost = cost
        self.pub = publish
        self.default_qty = float(default_qty)
        self.venue_latency_ms = int(venue_latency_ms)
        self._oid = 0
        # monkey patch
        self._orig_order = strategy.order
        strategy.order = self._order_proxy  # type: ignore
        self._marks: Dict[str,float] = {}

    def update_mark(self, symbol: str, px: float):
        self._marks[symbol] = float(px)

    def _next_id(self) -> str:
        self._oid += 1
        return f"sh-{self._oid:08d}"

    def _order_proxy(
        self, symbol: str, side: str, qty: float | None = None, *,
        order_type: str = "market",
        limit_price: float | None = None,
        venue: Optional[str] = None,
        mark_price: float | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        sym = symbol.upper()
        q = self.default_qty if (qty is None or qty <= 0) else float(qty)
        mark = float(mark_price if (mark_price and mark_price > 0) else self._marks.get(sym, 0.0))
        oid = self._next_id()
        # simulate fill
        if order_type == "limit" and limit_price is not None:
            got = self.cost.limit_fill(side, mark, float(limit_price))
            if not got:
                self.pub(S_SHADOW_EVT, {"ts_ms": now_ms(), "level": "info", "msg": "limit not marketable", "symbol": sym, "limit": limit_price, "mark": mark, "order_id": oid})
                return
            px, fee = got
        else:
            px, fee = self.cost.market_fill(side, mark, q)

        ts = now_ms() + self.venue_latency_ms
        # book & publish
        self.book.on_fill(symbol=sym, side=side, qty=q, price=px, fee=fee)
        fill = {"ts_ms": ts, "order_id": oid, "symbol": sym, "side": side, "qty": q, "price": px, "fee": fee, "strategy": self.strategy.ctx.name}
        self.pub(S_SHADOW_ORD, fill)

# ---------- Redis bus (graceful) --------------------------------------------
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

    async def publish(self, stream: str, obj: Dict[str,Any]):
        if not self.r:
            # fallback: print nothing to keep quiet in prod
            return
        try:
            await self.r.xadd(stream, {"json": json.dumps(obj, ensure_ascii=False)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

    async def read(self, last_ids: Dict[str,str], count: int = 500, block_ms: int = 1000):
        if not self.r:
            await asyncio.sleep(block_ms/1000.0)
            return []
        try:
            return await self.r.xread(last_ids, count=count, block=block_ms)  # type: ignore
        except Exception:
            return []

# ---------- Shadow Runner ----------------------------------------------------
@dataclass
class ShadowConfig:
    strategy_path: str = "backend.engine.strategy_base:ExampleBuyTheDip"
    capital_base: float = 100_000.0
    default_qty: float = 1.0
    fee_bps: float = 0.3
    half_spread_bps: float = 0.8
    impact_bps: float = 0.0
    venue_latency_ms: int = 5
    subscribe_news: bool = True
    compare_with_live: bool = False    # read S_ORDERS_LIVE to diff
    write_artifacts: bool = True
    artifacts_dir: str = "artifacts/shadow"
    symbols_whitelist: Optional[List[str]] = None

class ShadowRunner:
    def __init__(self, cfg: ShadowConfig):
        self.cfg = cfg
        self.bus = Bus()
        self.cost = CostModel(cfg.fee_bps, cfg.half_spread_bps, cfg.impact_bps)
        self.book = ShadowBook(cfg.capital_base)
        # load strategy
        self.strategy = load_strategy(cfg.strategy_path)
        self.harness = StrategyHarness(
            self.strategy, self.book, self.cost, publish=self._publish,  # type: ignore
            default_qty=cfg.default_qty, venue_latency_ms=cfg.venue_latency_ms
        )
        self._marks: Dict[str,float] = {}
        self._curve: List[Dict[str,Any]] = []
        self._blotter: List[Dict[str,Any]] = []
        self._last_ids = {S_BARS: "$"}
        if cfg.subscribe_news:
            self._last_ids[S_NEWS] = "$"
        if cfg.compare_with_live:
            self._last_ids[S_ORDERS_LIVE] = "$"
        # artifacts
        self.run_id = f"shadow_{int(time.time())}"
        self.art_dir = os.path.join(cfg.artifacts_dir, self.run_id)
        if cfg.write_artifacts:
            os.makedirs(self.art_dir, exist_ok=True)

    async def _publish(self, stream: str, obj: Dict[str,Any]):
        # capture blotter / curve when relevant
        if stream == S_SHADOW_ORD:
            self._blotter.append(obj)
        await self.bus.publish(stream, obj)

    async def start(self):
        await self.bus.connect()
        # strategy lifecycle
        if hasattr(self.strategy, "on_start"):
            try: self.strategy.on_start()
            except Exception: pass

        try:
            await self._loop()
        finally:
            if hasattr(self.strategy, "on_stop"):
                try: self.strategy.on_stop()
                except Exception: pass
            await self._write_artifacts()

    async def _loop(self):
        # main pump: bars (+ optional news, live fills for diff)
        while True:
            resp = await self.bus.read(self._last_ids, count=600, block_ms=1000)
            if not resp:
                # periodic equity snapshot even if idle
                await self._snapshot_equity()
                continue

            for stream, entries in resp:
                self._last_ids[stream] = entries[-1][0]
                if stream == S_BARS:
                    for _id, fields in entries:
                        tick = _parse_json_or_fields(fields)
                        sym = str(tick.get("symbol","")).upper()
                        if self.cfg.symbols_whitelist and sym not in self.cfg.symbols_whitelist:
                            continue
                        px = float(tick.get("close") or tick.get("price") or 0.0)
                        if px <= 0: 
                            continue
                        self._marks[sym] = px
                        self.harness.update_mark(sym, px)
                        # call strategy
                        try:
                            self.strategy.on_tick({"ts_ms": int(tick.get("ts_ms") or now_ms()), "symbol": sym, "price": px})
                        except Exception as e:
                            await self.bus.publish(S_SHADOW_EVT, {"ts_ms": now_ms(), "level":"error", "msg": f"on_tick: {e}"})
                    await self._snapshot_equity()

                elif stream == S_NEWS and self.cfg.subscribe_news:
                    for _id, fields in entries:
                        ev = _parse_json_or_fields(fields)
                        if hasattr(self.strategy, "on_news"):
                            try:
                                self.strategy.on_news(ev)  # type: ignore
                            except Exception as e:
                                await self.bus.publish(S_SHADOW_EVT, {"ts_ms": now_ms(), "level":"error", "msg": f"on_news: {e}"})

                elif stream == S_ORDERS_LIVE and self.cfg.compare_with_live:
                    # compare latest equity vs live fills (optional; here just publish diff stubs)
                    for _id, fields in entries:
                        live = _parse_json_or_fields(fields)
                        await self.bus.publish(S_SHADOW_EVT, {"ts_ms": now_ms(), "level":"info", "msg":"live_order_seen", "live": live})

    async def _snapshot_equity(self):
        eq = self.book.equity(self._marks)
        snap = {
            "ts_ms": now_ms(),
            "equity": eq,
            "cash": self.book.cash,
            "realized": self.book.realized,
            "fees": self.book.fees,
        }
        # positions
        pos = [{"symbol": s, "qty": p.qty, "avg_px": p.avg_px} for s,p in self.book.pos.items() if abs(p.qty) > 1e-9]
        await self.bus.publish(S_SHADOW_PNL, snap)
        await self.bus.publish(S_SHADOW_POS, {"ts_ms": snap["ts_ms"], "positions": pos})
        self._curve.append({"ts_ms": snap["ts_ms"], "equity": eq, "cash": self.book.cash, "realized": self.book.realized, "fees": self.book.fees})

    async def _write_artifacts(self):
        if not self.cfg.write_artifacts:
            return
        # curve & blotter & positions snapshots
        with open(os.path.join(self.art_dir, "curve.json"), "w") as f:
            json.dump(self._curve, f)
        with open(os.path.join(self.art_dir, "blotter.json"), "w") as f:
            json.dump(self._blotter, f)
        # last positions snapshot
        last_pos = []
        for s,p in self.book.pos.items():
            last_pos.append({"symbol": s, "qty": p.qty, "avg_px": p.avg_px})
        with open(os.path.join(self.art_dir, "positions.json"), "w") as f:
            json.dump(last_pos, f)

# ---------- Helpers ----------------------------------------------------------
def _parse_json_or_fields(fields: Dict[str,Any]) -> Dict[str,Any]:
    raw = fields.get("json")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    # else treat as flat
    return {k: _coerce(v) for k,v in fields.items()}

def _coerce(v: Any) -> Any:
    # try numeric
    try:
        if isinstance(v, str) and v.strip().isdigit():
            return int(v)
        return float(v)
    except Exception:
        return v

# ---------- CLI --------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("shadow_runner")
    ap.add_argument("--strategy", type=str, default="backend.engine.strategy_base:ExampleBuyTheDip")
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--qty", type=float, default=1.0)
    ap.add_argument("--fee-bps", type=float, default=0.3)
    ap.add_argument("--spread-bps", type=float, default=0.8)
    ap.add_argument("--impact-bps", type=float, default=0.0)
    ap.add_argument("--latency-ms", type=int, default=5)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--compare-live", action="store_true")
    ap.add_argument("--symbols", type=str, default=None, help="Comma whitelist, e.g. AAPL,MSFT,RELIANCE")
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()

    cfg = ShadowConfig(
        strategy_path=args.strategy,
        capital_base=args.capital,
        default_qty=args.qty,
        fee_bps=args.fee_bps,
        half_spread_bps=args.spread_bps,
        impact_bps=args.impact_bps,
        venue_latency_ms=args.latency_ms,
        subscribe_news=(not args.no_news),
        compare_with_live=args.compare_live,
        write_artifacts=(not args.no_artifacts),
        symbols_whitelist=[s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    )

    async def _run():
        sr = ShadowRunner(cfg)
        await sr.start()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()