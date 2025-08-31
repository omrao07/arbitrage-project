# backend/analytics/index_builder.py
from __future__ import annotations

import os, json, time, math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any

# ---------- Optional Redis (graceful fallback) ----------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PRICES_STREAM    = os.getenv("PRICES_STREAM", "prices.marks")   # {symbol, price, ccy?, ts_ms}
FX_STREAM        = os.getenv("FX_STREAM", "fx.rates")           # {pair:'EURUSD', mid, ts_ms}
INDEX_LEVELS_OUT = os.getenv("INDEX_LEVELS_OUT", "index.levels")
MAXLEN           = int(os.getenv("INDEX_LEVELS_MAXLEN", "20000"))
HOME_CCY         = os.getenv("HOME_CCY", "USD")

# ---------- Small utils ----------
def now_ms() -> int: return int(time.time() * 1000)
def yyyymmdd(ts_ms: int) -> str:
    t = time.gmtime(ts_ms/1000)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"

# ---------- Specs / Models ----------
@dataclass
class Constituent:
    symbol: str
    weight: float                 # target weight in fraction (0..1)
    ccy: str = HOME_CCY           # local security currency
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WeightChange:
    ts_ms: int                    # effective timestamp (open of day is common)
    weights: Dict[str, float]     # symbol -> target weight (fractions)

@dataclass
class CorpAction:
    ts_ms: int
    symbol: str
    kind: str                     # 'split' | 'cash_div'
    ratio: Optional[float] = None # for split (e.g., 2.0 for 2-for-1)
    amount: Optional[float] = None# for cash dividend in local CCY

@dataclass
class IndexSpec:
    name: str
    base_level: float = 1000.0
    base_ts_ms: int = 0           # if 0, first observed mark becomes base
    home_ccy: str = HOME_CCY
    method: str = "weighted"      # 'weighted' | 'equal' | 'price'
    total_return: bool = True     # include dividends
    cap_limit: Optional[float] = None   # e.g., 0.10 for 10% cap at rebalance
    rebalance_rule: Optional[str] = "monthly"  # 'monthly' | 'quarterly' | None
    drift_tol: Optional[float] = 0.02   # rebalance if |w_t - w_target| > 2%
    # Point-in-time weight history (effective schedules)
    schedule: List[WeightChange] = field(default_factory=list)

# ---------- Price/FX providers (callable hooks) ----------
PriceProvider = Callable[[str, int], Optional[Tuple[float, str]]]         # (price, ccy) at ts_ms
FXProvider    = Callable[[str, str, int], Optional[float]]                # rate from->to at ts_ms

# ---------- Index state ----------
@dataclass
class IndexState:
    ts_ms: int = 0
    level_price: float = 0.0      # price return index level
    level_total: float = 0.0      # total return index level
    divisor: float = 1.0          # maintains continuity on rebalances/CA
    holdings: Dict[str, float] = field(default_factory=dict)  # notional allocation in home CCY
    weights_target: Dict[str, float] = field(default_factory=dict)
    last_prices: Dict[str, float] = field(default_factory=dict)  # in HOME CCY
    accrued_div: float = 0.0      # dividends since last level update (home CCY)

# ---------- Core builder ----------
class IndexBuilder:
    """
    Computes custom index levels (price & total return), handles rebalances,
    corporate actions, dividends and FX.

    Typical usage:
      ib = IndexBuilder(spec, price_provider, fx_provider)
      ib.initialize(first_ts)
      ib.on_mark({'symbol':'AAPL','price':190,'ccy':'USD','ts_ms':...})
      level = ib.level()
    """
    def __init__(self, spec: IndexSpec, get_price: PriceProvider, get_fx: FXProvider):
        self.spec = spec
        self.get_price = get_price
        self.get_fx = get_fx
        self.st = IndexState()
        self.fx_cache: Dict[Tuple[str,str], float] = {}  # last seen fx from->to

    # ---- helpers ----
    def _fx(self, from_ccy: str, to_ccy: Optional[str], ts_ms: int) -> float:
        to = (to_ccy or self.spec.home_ccy).upper()
        fr = (from_ccy or to).upper()
        if fr == to: return 1.0
        key = (fr, to)
        r = self.fx_cache.get(key)
        if r is not None:
            return r
        rate = self.get_fx(fr, to, ts_ms) or 1.0
        self.fx_cache[key] = rate
        self.fx_cache[(to, fr)] = 1.0 / rate if rate != 0 else 1.0
        return rate

    def _current_targets(self, ts_ms: int) -> Dict[str, float]:
        """Point-in-time target weights (sum≈1)."""
        sched = sorted(self.spec.schedule, key=lambda w: w.ts_ms)
        active = {}
        for w in sched:
            if w.ts_ms <= ts_ms:
                active = dict(w.weights)
            else:
                break
        if not active:
            # if no schedule provided, assume equal weights over initial constituents from spec.schedule[0]
            if self.spec.schedule:
                first = self.spec.schedule[0].weights
                n = max(1, len(first))
                return {k: 1.0/n for k in first}
            return {}
        # normalize
        s = sum(max(0.0, v) for v in active.values()) or 1.0
        return {k: max(0.0, v)/s for k, v in active.items()}

    # ---- lifecycle ----
    def initialize(self, ts_ms: int):
        self.st = IndexState(ts_ms=ts_ms, level_price=self.spec.base_level or 1000.0,
                             level_total=self.spec.base_level or 1000.0, divisor=1.0)
        self.st.weights_target = self._current_targets(ts_ms)

    def _maybe_rebalance(self, ts_ms: int):
        # calendar trigger
        if self.spec.rebalance_rule:
            d_prev = yyyymmdd(self.st.ts_ms)
            d_now  = yyyymmdd(ts_ms)
            if d_prev != d_now:
                if self.spec.rebalance_rule == "monthly":
                    if d_prev[:7] != d_now[:7]:   # month changed
                        self._rebalance_to_targets(ts_ms)
                elif self.spec.rebalance_rule == "quarterly":
                    if d_prev[:7] != d_now[:7]:
                        # simple: rebalance in months 03/06/09/12
                        if d_now[5:7] in {"03","06","09","12"}:
                            self._rebalance_to_targets(ts_ms)
        # drift trigger
        if self.st.holdings and self.spec.drift_tol:
            tot = sum(self.st.holdings.values())
            if tot > 0:
                weights_now = {k: v/tot for k, v in self.st.holdings.items()}
                for k, wt in self.st.weights_target.items():
                    if abs(weights_now.get(k, 0.0) - wt) > self.spec.drift_tol:
                        self._rebalance_to_targets(ts_ms)
                        break

    def _apply_caps(self, w: Dict[str, float]) -> Dict[str, float]:
        """Apply simple cap limit (e.g., 10%) and renormalize the rest."""
        cap = self.spec.cap_limit
        if not cap or cap <= 0:
            return w
        w = dict(w)
        over = {k: min(v, cap) for k, v in w.items()}
        cut = sum(w.values()) - sum(over.values())
        if cut <= 1e-12:
            return over
        # redistribute proportionally to those below cap
        below = {k: over[k] for k in over if over[k] < cap - 1e-12}
        if not below:
            return over
        s = sum(below.values()) or 1e-12
        for k in below:
            add = cut * (over[k] / s)
            over[k] += add
        s2 = sum(over.values()) or 1.0
        return {k: v/s2 for k, v in over.items()}

    def _rebalance_to_targets(self, ts_ms: int):
        self.st.weights_target = self._current_targets(ts_ms)
        self.st.weights_target = self._apply_caps(self.st.weights_target)
        # set holdings to match targets at current total notional
        tot_notional = sum(self.st.holdings.values()) or 1.0
        self.st.holdings = {k: tot_notional * wt for k, wt in self.st.weights_target.items()}

    # ---- market updates ----
    def on_mark(self, symbol: str, price: float, ccy: Optional[str], ts_ms: int):
        """Consume a price tick; update last_prices (home CCY)."""
        if ts_ms <= 0 or price <= 0: return
        fx = self._fx((ccy or self.spec.home_ccy), self.spec.home_ccy, ts_ms)
        self.st.last_prices[symbol] = float(price) * fx

    def on_fx(self, pair: str, mid: float, ts_ms: int):
        """Update FX cache from stream events (e.g., EURUSD)."""
        pair = (pair or "").upper().replace("/", "")
        if len(pair) != 6 or mid <= 0:
            return
        base, quote = pair[:3], pair[3:]
        self.fx_cache[(base, quote)] = mid
        self.fx_cache[(quote, base)] = 1.0 / mid

    def on_corporate_action(self, ca: CorpAction):
        """Splits adjust holdings & last_prices; cash_div adds to accrued_div for TR."""
        if ca.kind == "split" and ca.ratio and ca.ratio > 0:
            r = float(ca.ratio)
            # Adjust price inversely; holdings (notional) unchanged because we store in currency
            # If you model *units*, do: units *= r; price /= r. Here: leave last price alone (currency basis).
            pass
        elif ca.kind == "cash_div" and ca.amount and ca.amount > 0:
            px = self.st.last_prices.get(ca.symbol)
            if px:
                # dividend per share * (notional/px) = cash in home CCY
                units = (self.st.holdings.get(ca.symbol, 0.0) / max(px, 1e-12))
                fx = self._fx(self.spec.home_ccy, self.spec.home_ccy, ca.ts_ms)  # 1.0 but placeholder for foreign divs
                self.st.accrued_div += float(ca.amount) * units * fx

    # ---- level computation ----
    def _ensure_initialized(self, ts_ms: int):
        if self.st.ts_ms == 0:
            self.initialize(ts_ms)

    def step(self, ts_ms: int) -> Dict[str, Any]:
        """
        Revalue holdings at latest prices, update index levels.
        Returns snapshot dict with price & total return levels.
        """
        self._ensure_initialized(ts_ms)

        # On first call, set holdings by targets using base index notional
        if not self.st.holdings:
            # seed total notional = base_level (arbitrary scale)
            tot = self.spec.base_level
            targets = self._apply_caps(self.st.weights_target or {})
            self.st.holdings = {k: tot * wt for k, wt in targets.items()}

        # Rebalance if needed
        self._maybe_rebalance(ts_ms)

        # Revalue notional (home CCY)
        tot_value = 0.0
        for sym, notional in list(self.st.holdings.items()):
            px = self.st.last_prices.get(sym)
            if px is None or px <= 0:
                continue
            # notional is already in currency; to maintain weights we just sum
            tot_value += float(notional)

        if tot_value <= 0:
            tot_value = sum(self.st.holdings.values())

        # Price return level:
        # simple: level scales by total notional / divisor
        if self.st.level_price <= 0:
            self.st.level_price = self.spec.base_level
        if self.st.level_total <= 0:
            self.st.level_total = self.spec.base_level

        # Use divisor so that when holdings change (rebalance), continuity is preserved.
        level_px = tot_value / max(self.st.divisor, 1e-12)

        # Total return adds accrued dividends as if reinvested at step
        level_tr = (tot_value + self.st.accrued_div) / max(self.st.divisor, 1e-12)
        self.st.accrued_div = 0.0

        self.st.level_price = level_px
        self.st.level_total = level_tr
        self.st.ts_ms = ts_ms

        return {
            "ts_ms": ts_ms,
            "name": self.spec.name,
            "home_ccy": self.spec.home_ccy,
            "price_level": round(self.st.level_price, 8),
            "total_return_level": round(self.st.level_total, 8),
            "divisor": round(self.st.divisor, 12),
        }

    def level(self) -> Tuple[float, float]:
        return self.st.level_price, self.st.level_total

    # ---- backfill ----
    def backfill(
        self,
        start_ts_ms: int,
        end_ts_ms: int,
        step_ms: int,
        symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Offline backfill using providers.
        """
        out: List[Dict[str, Any]] = []
        t = start_ts_ms
        self.initialize(start_ts_ms)
        base_set = False

        symbols = symbols or list((self.spec.schedule[0].weights if self.spec.schedule else {}).keys())

        while t <= end_ts_ms:
            # pull prices
            for sym in symbols:
                px_ccy = self.get_price(sym, t)
                if px_ccy is None:
                    continue
                px, ccy = px_ccy
                self.on_mark(sym, px, ccy, t)
            snap = self.step(t)
            if not base_set and (self.spec.base_ts_ms and t >= self.spec.base_ts_ms):
                # re-anchor divisor to match base_level exactly
                if snap["price_level"] != 0:
                    self.st.divisor *= snap["price_level"] / max(self.spec.base_level, 1e-12)
                    snap = self.step(t)
                base_set = True
            out.append(snap)
            t += step_ms
        return out

# ---------- Redis worker (optional) ----------
class IndexWorker:
    """
    Live worker: tails price + fx streams, steps the index each tick and publishes levels.
    """
    def __init__(self, builder: IndexBuilder):
        self.b = builder
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.last_prices_id = "$"
        self.last_fx_id = "$"

    async def connect(self):
        if not USE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def run(self):
        await self.connect()
        if not self.r:
            print("[index_builder] no redis; worker idle")
            return

        self.b.initialize(now_ms())

        while True:
            try:
                resp = await self.r.xread(
                    {PRICES_STREAM: self.last_prices_id, FX_STREAM: self.last_fx_id},
                    count=500, block=5000
                )  # type: ignore
                tick = False
                if not resp:
                    # heartbeat step each 5s if idle
                    snap = self.b.step(now_ms())
                    await self._publish(snap)
                    continue

                for stream, entries in resp:
                    for _id, fields in entries:
                        if stream == PRICES_STREAM:
                            self.last_prices_id = _id
                            j = json.loads(fields.get("json", "{}"))
                            sym = str(j.get("symbol","")).upper()
                            px = float(j.get("price") or 0.0)
                            ccy = str(j.get("ccy") or self.b.spec.home_ccy).upper()
                            ts = int(j.get("ts_ms") or now_ms())
                            if sym and px > 0:
                                self.b.on_mark(sym, px, ccy, ts)
                                tick = True
                        elif stream == FX_STREAM:
                            self.last_fx_id = _id
                            j = json.loads(fields.get("json", "{}"))
                            pair = str(j.get("pair","")).upper()
                            mid = float(j.get("mid") or 0.0)
                            ts = int(j.get("ts_ms") or now_ms())
                            if pair and mid > 0:
                                self.b.on_fx(pair, mid, ts)

                if tick:
                    snap = self.b.step(now_ms())
                    await self._publish(snap)

            except Exception as e:
                await self._publish({"ts_ms": now_ms(), "name": self.b.spec.name, "error": str(e)})
                await asyncio_sleep(0.5)

    async def _publish(self, snap: Dict[str, Any]):
        if self.r:
            try:
                await self.r.xadd(INDEX_LEVELS_OUT, {"json": json.dumps(snap)}, maxlen=MAXLEN, approximate=True)  # type: ignore
            except Exception:
                pass
        else:
            print("[index_builder]", snap)

# ---------- Async helper ----------
async def asyncio_sleep(sec: float):
    try:
        import asyncio
        await asyncio.sleep(sec)
    except Exception:
        time.sleep(sec)

# ---------- Quick demo providers ----------
def demo_price_provider(symbol: str, ts_ms: int) -> Optional[Tuple[float, str]]:
    # toy: sin wave around 100
    base = 100.0 + 5.0*math.sin(ts_ms/86400000*2*math.pi)
    # simple symbol-specific offset
    off = sum(ord(c) for c in symbol) % 20 - 10
    return base + off, "USD"

def demo_fx_provider(fr: str, to: str, ts_ms: int) -> Optional[float]:
    if fr == to: return 1.0
    # toy constant
    if fr == "EUR" and to == "USD": return 1.10
    if fr == "USD" and to == "EUR": return 0.91
    return 1.0

# ---------- CLI ----------
def _demo():
    # Define an index spec with two rebalances and caps
    spec = IndexSpec(
        name="MY_ALPHA_10",
        base_level=1000.0,
        home_ccy="USD",
        total_return=True,
        cap_limit=0.10,
        rebalance_rule="monthly",
        schedule=[
            WeightChange(ts_ms=1, weights={"AAPL":0.4, "MSFT":0.3, "AMZN":0.3}),
            WeightChange(ts_ms=9999999999999, weights={"AAPL":0.34, "MSFT":0.33, "AMZN":0.33})
        ]
    )
    ib = IndexBuilder(spec, demo_price_provider, demo_fx_provider)
    start = int(time.time()*1000) - 10*86400000
    end   = int(time.time()*1000)
    rows = ib.backfill(start, end, step_ms=86400000)
    print("Backfilled", len(rows), "days. Last:", rows[-1])

if __name__ == "__main__":
    import argparse, asyncio
    ap = argparse.ArgumentParser("index_builder")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
    elif args.worker:
        spec = IndexSpec(name=os.getenv("INDEX_NAME","MY_INDEX"))
        ib = IndexBuilder(spec, demo_price_provider, demo_fx_provider)
        asyncio.run(IndexWorker(ib).run())
    else:
        _demo()