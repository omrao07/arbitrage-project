# backend/analytics/attribution.py
from __future__ import annotations

import os, json, time, math, datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable, DefaultDict
from collections import defaultdict

# -------- Optional Redis -----------------------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

# -------- Env / Streams ------------------------------------------------------
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FILLS_STREAM     = os.getenv("FILLS_STREAM", "orders.updates")    # OMS emits fills/partials
MARKS_STREAM     = os.getenv("MARKS_STREAM", "prices.marks")      # {symbol, ts_ms, price, currency?}
FX_STREAM        = os.getenv("FX_STREAM", "fx.rates")             # {pair:'EURUSD', ts_ms, mid}
ATTRIB_STREAM    = os.getenv("ATTRIB_STREAM", "pnl.attribution")  # rollup out
MAXLEN           = int(os.getenv("ATTRIB_MAXLEN", "20000"))
HOME_CCY         = os.getenv("HOME_CCY", "USD")

# -------- Cost Model hook (optional) ----------------------------------------
class CostModel:
    """
    Override/replace with your own (import from backend/oms/cost_model.py if present).
    Must implement cost(fill) and slippage(fill, mark_ref).
    """
    def cost(self, fill: Dict[str, Any]) -> float:
        # default: flat 1 bps + $0.005 per share
        qty = abs(float(fill.get("qty", 0)))
        px  = float(fill.get("price", 0))
        adval = qty * px
        return 0.0001 * adval + 0.005 * qty

    def slippage(self, fill: Dict[str, Any], ref_px: Optional[float]) -> float:
        if ref_px is None: return 0.0
        side = str(fill.get("side","buy")).lower()
        qty  = float(fill.get("qty", 0))
        px   = float(fill.get("price", 0))
        slip = (px - ref_px) * qty if side == "buy" else (ref_px - px) * qty
        return slip

# -------- Data keys / rows ---------------------------------------------------
def _yyyymmdd(ts_ms: int, tz: Optional[dt.tzinfo] = None) -> str:
    tz = tz or dt.timezone.utc
    return dt.datetime.fromtimestamp(ts_ms/1000, tz).strftime("%Y-%m-%d")

@dataclass(frozen=True)
class Key:
    d: str               # trade/mark day (YYYY-MM-DD) in HOME_CCY timezone
    strategy: str
    symbol: str
    region: str = ""
    book: str = "default"
    tags: Tuple[str, ...] = ()   # arbitrary labels (sector, factor bucket, etc.)
    ccy: str = HOME_CCY          # native security currency (pre-FX)

@dataclass
class Bucket:
    realized: float = 0.0         # sum of signed (fill_px - avg_cost)*qty closed
    unrealized: float = 0.0       # (mark - avg_cost)*open_qty
    fees: float = 0.0
    slippage: float = 0.0
    financing: float = 0.0        # carry/borrow if provided externally
    fx_pnl: float = 0.0           # FX translation component
    gross: float = 0.0            # |notional| traded today
    trades: int = 0
    qty_open: float = 0.0         # end-of-day open position (signed)
    avg_cost: float = 0.0         # running average cost (signed / position-aware)
    last_mark: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total"] = self.realized + self.unrealized - self.fees - self.financing - self.slippage + self.fx_pnl
        return d

# -------- Engine -------------------------------------------------------------
class AttributionEngine:
    def __init__(self, home_ccy: str = HOME_CCY, cost_model: Optional[CostModel] = None):
        self.home = home_ccy.upper()
        self.cost = cost_model or CostModel()
        # state
        self.buckets: DefaultDict[Key, Bucket] = defaultdict(Bucket)
        self.positions: Dict[Tuple[str, str], Tuple[float, float]] = {}  # (symbol, book) -> (qty, avg_cost)
        self.fx: Dict[Tuple[str, str], float] = {}  # (from, to) -> rate (from->to)
        self.ref_prices: Dict[str, float] = {}      # for slippage refs / last seen mark

    # ---- FX helpers ---------------------------------------------------------
    def set_fx(self, pair: str, mid: float):
        pair = pair.upper().replace("/", "")
        if len(pair) != 6: return
        base, quote = pair[:3], pair[3:]
        if mid <= 0: return
        self.fx[(base, quote)] = mid
        self.fx[(quote, base)] = 1.0 / mid
        self.fx[(base, base)] = 1.0
        self.fx[(quote, quote)] = 1.0

    def fx_conv(self, amt: float, from_ccy: str, to_ccy: Optional[str] = None) -> float:
        to_ccy = (to_ccy or self.home).upper()
        from_ccy = (from_ccy or self.home).upper()
        if from_ccy == to_ccy:
            return amt
        rate = self.fx.get((from_ccy, to_ccy))
        if rate is None:
            # unknown → assume 1.0 (safe fallback; better: persist last)
            return amt
        return amt * rate

    # ---- Fill handling ------------------------------------------------------
    def on_fill(self, ev: Dict[str, Any]) -> None:
        """
        ev: {
          id, ts_ms, symbol, side, qty, price, strategy?, region?, book?, ccy?, ref_px?, fees?, financing?
          tags?: [..]
        }
        """
        ts_ms = int(ev.get("ts_ms") or time.time()*1000)
        day = _yyyymmdd(ts_ms)
        sym = str(ev.get("symbol","")).upper()
        if not sym: return
        side = str(ev.get("side","buy")).lower()
        qty  = float(ev.get("qty") or 0.0) * (1.0 if side=="buy" else -1.0)
        px   = float(ev.get("price") or 0.0)
        strat = str(ev.get("strategy") or "unknown")
        region = str(ev.get("region") or "")
        book = str(ev.get("book") or "default")
        ccy = str(ev.get("ccy") or self.home).upper()
        tags = tuple(sorted([str(t) for t in (ev.get("tags") or [])]))

        k = Key(d=day, strategy=strat, symbol=sym, region=region, book=book, tags=tags, ccy=ccy)
        b = self.buckets[k]

        # transaction costs & slippage
        ref_px = ev.get("ref_px", self.ref_prices.get(sym))
        fees = float(ev.get("fees") or self.cost.cost({"qty":abs(qty), "price":px}))
        slip = float(ev.get("slippage") or self.cost.slippage({"qty":abs(qty), "price":px, "side":side}, ref_px))

        # update running position
        pos_key = (sym, book)
        pos_qty, pos_cost = self.positions.get(pos_key, (0.0, 0.0))  # avg_cost is pos_cost
        new_qty = pos_qty + qty

        realized = 0.0
        if pos_qty == 0.0 or (pos_qty > 0 and qty > 0) or (pos_qty < 0 and qty < 0):
            # increase/add to position → new weighted average
            new_cost = _wa(pos_qty, pos_cost, qty, px)
        else:
            # closing (fully or partially) → realize PnL on closed portion
            closing_qty = -qty if abs(qty) < abs(pos_qty) and ((pos_qty>0 and qty<0) or (pos_qty<0 and qty>0)) else pos_qty
            closed = min(abs(qty), abs(pos_qty))
            sign = 1.0 if pos_qty > 0 else -1.0
            realized = (px - pos_cost) * (closed * sign)  # sign aligns P&L with position
            # if flip past zero, compute new avg from residual open + new open
            if new_qty != 0:
                # residual open at px as new cost if we crossed through zero
                new_cost = px
            else:
                new_cost = 0.0

        # update state
        self.positions[pos_key] = (new_qty, new_cost)
        b.realized += self.fx_conv(realized, ccy, self.home)
        b.fees     += self.fx_conv(fees, ccy, self.home)
        b.slippage += self.fx_conv(slip, ccy, self.home)
        b.gross    += abs(qty) * px
        b.trades   += 1
        b.qty_open = new_qty
        b.avg_cost = new_cost
        if self.ref_prices.get(sym) is None:
            self.ref_prices[sym] = px
        b.last_mark = self.ref_prices.get(sym, px)

    # ---- Mark handling ------------------------------------------------------
    def on_mark(self, ev: Dict[str, Any]) -> None:
        """
        ev: {ts_ms, symbol, price, ccy?}
        Revalues ALL books holding symbol at this price.
        """
        ts_ms = int(ev.get("ts_ms") or time.time()*1000)
        day = _yyyymmdd(ts_ms)
        sym = str(ev.get("symbol","")).upper()
        if not sym: return
        px = float(ev.get("price") or 0.0)
        ccy = str(ev.get("ccy") or self.home).upper()
        self.ref_prices[sym] = px

        # revalue for each book that holds the symbol
        for (s, book), (qty, avg_cost) in list(self.positions.items()):
            if s != sym: continue
            # find all buckets for this symbol/book on that day (multiple strategies/tags possible)
            for k, b in self.buckets.items():
                if k.symbol != sym: continue
                if k.book != book: continue
                if k.d != day: continue
                # unrealized at mark
                b.unrealized = self.fx_conv((px - b.avg_cost) * b.qty_open, ccy, self.home)
                b.last_mark = px

    # ---- Financing / FX adjustments ----------------------------------------
    def on_financing(self, ev: Dict[str, Any]) -> None:
        """
        ev: {ts_ms, symbol?, book?, amount, ccy?}  (positive = cost, negative = rebate)
        """
        ts_ms = int(ev.get("ts_ms") or time.time()*1000)
        day = _yyyymmdd(ts_ms)
        sym = str(ev.get("symbol") or "").upper()
        book = str(ev.get("book") or "default")
        ccy = str(ev.get("ccy") or self.home).upper()
        amt = float(ev.get("amount") or 0.0)

        # apply to all strategies for that symbol/book/day (or symbol empty → portfolio-level bucket)
        for k, b in self.buckets.items():
            if k.d != day: continue
            if sym and k.symbol != sym: continue
            if k.book != book: continue
            b.financing += self.fx_conv(amt, ccy, self.home)

    def on_fx(self, ev: Dict[str, Any]) -> None:
        """
        ev: {pair:'EURUSD', ts_ms, mid}
        """
        pair = str(ev.get("pair",""))
        mid = float(ev.get("mid") or 0.0)
        if pair and mid > 0:
            self.set_fx(pair, mid)

    # ---- Reports ------------------------------------------------------------
    def rows(self) -> List[Dict[str, Any]]:
        out = []
        for k, b in self.buckets.items():
            row = {
                "date": k.d, "strategy": k.strategy, "symbol": k.symbol,
                "region": k.region, "book": k.book, "tags": list(k.tags), "ccy": k.ccy,
                **b.to_dict(),
            }
            out.append(row)
        return out

    def rollup(self, group_by: Iterable[str]) -> List[Dict[str, Any]]:
        """
        group_by: e.g., ["date","strategy"] or ["date","book","region"]
        """
        gb = tuple(group_by)
        table: DefaultDict[Tuple[Any,...], Bucket] = defaultdict(Bucket)
        for r in self.rows():
            key = tuple(r[g] for g in gb)
            agg = table[key]
            agg.realized   += r["realized"]
            agg.unrealized += r["unrealized"]
            agg.fees       += r["fees"]
            agg.slippage   += r["slippage"]
            agg.financing  += r["financing"]
            agg.fx_pnl     += r["fx_pnl"]
            agg.gross      += r["gross"]
            agg.trades     += r["trades"]
        out = []
        for k, b in table.items():
            row = {g: v for g, v in zip(gb, k)}
            row.update(b.to_dict())
            out.append(row)
        # sort by date then desc total
        out.sort(key=lambda x: (x.get("date",""), -x.get("total",0.0)))
        return out

# -------- Utilities ----------------------------------------------------------
def _wa(q1: float, c1: float, q2: float, p2: float) -> float:
    """
    Weighted average cost for running position; sign-aware.
    If adding to same-side position: (q1*c1 + q2*p2) / (q1+q2), using absolute shares.
    If closing, caller decides whether to reset to p2 when crossing through zero.
    """
    same_side = (q1 >= 0 and q2 >= 0) or (q1 <= 0 and q2 <= 0)
    if not same_side:
        # adding opposite sign → caller handles; return current
        return c1
    a1, a2 = abs(q1), abs(q2)
    if a1 + a2 == 0: return 0.0
    return (a1 * c1 + a2 * p2) / (a1 + a2)

# -------- Stream worker (optional) ------------------------------------------
async def run_worker(engine: Optional[AttributionEngine] = None):
    """
    Tails FILLS_STREAM, MARKS_STREAM, FX_STREAM; periodically publishes rollups to ATTRIB_STREAM.
    """
    if not USE_REDIS:
        raise RuntimeError("Redis not available")
    r: AsyncRedis = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
    await r.ping()
    e = engine or AttributionEngine()

    last_fills = "$"; last_marks = "$"; last_fx = "$"
    last_flush = time.time()

    while True:
        try:
            resp = await r.xread({
                FILLS_STREAM: last_fills,
                MARKS_STREAM: last_marks,
                FX_STREAM:    last_fx
            }, count=500, block=5000)
            if not resp:
                # timed flush
                if time.time() - last_flush > 5.0:
                    await _flush_rollups(r, e)
                    last_flush = time.time()
                continue

            for stream, entries in resp:
                for _id, fields in entries:
                    if stream == FILLS_STREAM:
                        last_fills = _id
                        try:
                            ev = json.loads(fields.get("json","{}"))
                            if (ev.get("status") or "").lower() in {"filled","partial"}:
                                e.on_fill(ev)
                        except Exception:
                            continue
                    elif stream == MARKS_STREAM:
                        last_marks = _id
                        try:
                            ev = json.loads(fields.get("json","{}"))
                            e.on_mark(ev)
                        except Exception:
                            continue
                    elif stream == FX_STREAM:
                        last_fx = _id
                        try:
                            ev = json.loads(fields.get("json","{}"))
                            e.on_fx(ev)
                        except Exception:
                            continue

            # periodic flush
            if time.time() - last_flush > 2.0:
                await _flush_rollups(r, e)
                last_flush = time.time()

        except Exception:
            await _sleep(0.5)

async def _flush_rollups(r: AsyncRedis, e: AttributionEngine): # type: ignore
    # Publish per-strategy daily rollup
    rows = e.rollup(["date","strategy"])
    if not rows: return
    payload = {"ts_ms": int(time.time()*1000), "rows": rows, "home_ccy": e.home}
    try:
        await r.xadd(ATTRIB_STREAM, {"json": json.dumps(payload)}, maxlen=MAXLEN, approximate=True)
    except Exception:
        pass

async def _sleep(sec: float):
    try:
        import asyncio
        await asyncio.sleep(sec)
    except Exception:
        time.sleep(sec)

# -------- Minimal self-test --------------------------------------------------
def _demo():
    eng = AttributionEngine()
    # FX
    eng.set_fx("EURUSD", 1.10)
    now = int(time.time()*1000)
    # Fills: buy 100 AAPL @ 100, later sell 60 @ 102
    eng.on_fill({"ts_ms": now, "symbol":"AAPL", "side":"buy", "qty":100, "price":100, "strategy":"alpha1"})
    eng.on_mark({"ts_ms": now, "symbol":"AAPL", "price":101})
    eng.on_fill({"ts_ms": now+1000, "symbol":"AAPL", "side":"sell", "qty":60, "price":102, "strategy":"alpha1"})
    eng.on_mark({"ts_ms": now+2000, "symbol":"AAPL", "price":103})
    # Print rollup
    print(eng.rollup(["date","strategy"]))

if __name__ == "__main__":
    _demo()