# backend/strategies/hedger.py
from __future__ import annotations

"""
Hedger
------
Portfolio-level hedging agent that:
- Listens to risk/position snapshots (Redis Streams or direct calls)
- Applies simple policy rules (vol target, delta neutrality, drawdown / ES protection)
- Uses hedge recipes (collars, protective puts, futures overlays, delta hedge) to
  generate order intents
- Publishes intents to the pre-risk order stream your stack already uses

Dependencies: standard library (+ optional redis)
Pairs with: backend/strategies/hedge_recipes.py

Streams (override via env):
  positions.snapshots : {"ts_ms", "book", "positions":[{"symbol","qty"},...], "prices":{"SYM":px}, "beta_to_index"?, ...}
  risk.metrics        : {"ts_ms","book","drawdown_pct","vol_20d","es_975", ...}
  hedge.commands      : {"ts_ms","action":"rebalance|delta|protect|vol","book":"...","args":{...}}
  orders.incoming     : order-intents for Risk→OMS

Typical wiring:
  python -m backend.strategies.hedger
"""

import os
import json
import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable

# ---------- Optional Redis (graceful if missing) ----------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

# ---------- Local bus fallbacks (publish/consume) ----------
def _now_ms() -> int: return int(time.time() * 1000)

# If you already have backend.bus.streams, use it; else minimal fallbacks.
try:
    from backend.bus.streams import publish_stream as _publish_stream
    from backend.bus.streams import consume_stream as _consume_stream
except Exception:
    _publish_stream = None
    _consume_stream = None

# ---------- Hedge recipes ----------
from backend.strategies.hedge_recipes import ( # type: ignore
    HedgeKitchen, ChainSnapshot, OptionQuote, dummy_accessor
)

# ---------- Env / Streams ----------
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379/0")

POSITIONS_STREAM   = os.getenv("POSITIONS_STREAM", "positions.snapshots")
RISK_STREAM        = os.getenv("RISK_STREAM", "risk.metrics")
COMMANDS_STREAM    = os.getenv("HEDGE_COMMANDS_STREAM", "hedge.commands")
ORDERS_STREAM      = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")
EVENTS_STREAM      = os.getenv("HEDGE_EVENTS_STREAM", "hedge.events")

MAXLEN             = int(os.getenv("HEDGE_MAXLEN", "20000"))

INDEX_FUTURE       = os.getenv("HEDGE_INDEX_FUTURE", "ESZ5")     # example
INDEX_CONTRACT_VAL = float(os.getenv("HEDGE_INDEX_CONTRACT_VAL", "50000"))  # $/contract

CONTRACT_MULT      = int(os.getenv("HEDGE_CONTRACT_MULT", "100"))  # equity options multiplier

# ---------- Policy model ----------
@dataclass
class HedgePolicy:
    enabled: bool = True

    # Vol targeting via futures overlay
    vol_target: Optional[float] = 0.20           # e.g., 20% annualized (or daily stdev proxy)
    vol_tolerance: float = 0.02                   # deadband around target
    vol_beta_to_index: float = 1.0               # portfolio beta to chosen index
    index_future: str = INDEX_FUTURE
    index_contract_value: float = INDEX_CONTRACT_VAL

    # Delta neutrality (per symbol or book-level quick control)
    delta_target: Optional[float] = None         # None → disable; else target net delta (shares)
    delta_threshold: float = 0.0                 # act if |delta - target| > threshold

    # Protection triggers (drawdown / ES)
    drawdown_trigger: Optional[float] = 0.08     # 8% drawdown → hedge
    es_trigger: Optional[float] = None           # e.g., ES_97.5% > x → hedge

    # Which recipe & params when protection triggers
    protection_recipe: str = "collar"            # "collar" | "protective_put" | "put_spread" | "tail_put"
    put_delta: float = -0.25
    call_delta: float = 0.20
    expiry_days: int = 45
    ratio: float = 1.0

    # Universe filter: only symbols to hedge (optional)
    symbols: Optional[List[str]] = None

# ---------- Simple portfolio types ----------
@dataclass
class Position:
    symbol: str
    qty: float

@dataclass
class Snapshot:
    ts_ms: int
    book: str
    positions: List[Position]
    prices: Dict[str, float]
    beta_to_index: float = 1.0
    portfolio_value: Optional[float] = None

@dataclass
class RiskMetrics:
    ts_ms: int
    book: str
    vol_20d: Optional[float] = None          # realized/forecast vol proxy
    drawdown_pct: Optional[float] = None     # 0.08 for -8%
    es_975: Optional[float] = None           # Expected Shortfall at 97.5%
    delta_book: Optional[float] = None       # optional net delta at book-level

# ---------- Hedger core ----------
class Hedger:
    def __init__(
        self,
        *,
        chain_accessor: Callable[[str, Optional[int]], ChainSnapshot] = dummy_accessor,
        policy: Optional[HedgePolicy] = None
    ):
        self.kitchen = HedgeKitchen(chain_accessor, contract_multiplier=CONTRACT_MULT)
        self.policy = policy or HedgePolicy()
        self._r: Optional[AsyncRedis] = None # type: ignore

    # ---- Connectivity ----
    async def connect(self):
        if not USE_REDIS:
            return
        try:
            self._r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self._r.ping() # type: ignore
        except Exception:
            self._r = None

    # ---- Publish/Log helpers ----
    async def _publish_orders(self, intents: List[Dict[str, Any]]):
        if _publish_stream:
            for it in intents:
                _publish_stream(ORDERS_STREAM, it)
            return
        if self._r:
            for it in intents:
                await self._r.xadd(ORDERS_STREAM, {"json": json.dumps(it)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        else:
            print("[hedger] (no-redis) ORDER:", intents)

    async def _emit_event(self, obj: Dict[str, Any]):
        obj = {"ts_ms": _now_ms(), **obj}
        if self._r:
            try:
                await self._r.xadd(EVENTS_STREAM, {"json": json.dumps(obj)}, maxlen=MAXLEN, approximate=True)  # type: ignore
            except Exception:
                pass
        else:
            print("[hedger] EVENT:", obj)

    # ---- Decisions ----
    async def evaluate(self, snap: Snapshot, risk: Optional[RiskMetrics]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Build hedging order-intents given the latest snapshot & risk metrics.
        Returns (orders, notes). Caller publishes.
        """
        if not self.policy.enabled:
            return [], ["hedger disabled"]

        notes: List[str] = []
        intents: List[Dict[str, Any]] = []

        # Compute portfolio value if missing
        pv = snap.portfolio_value
        if pv is None:
            pv = 0.0
            for p in snap.positions:
                px = float(snap.prices.get(p.symbol, 0.0) or 0.0)
                pv += abs(p.qty) * px
        pv = float(pv or 0.0)

        # 1) Vol targeting via index futures overlay
        if risk and self.policy.vol_target is not None and risk.vol_20d is not None:
            # simple deadband
            over = (risk.vol_20d - self.policy.vol_target)
            if abs(over) > self.policy.vol_tolerance and pv > 0:
                h_ints, h_notes = self.kitchen.vol_target_overlay(
                    book_vol=float(risk.vol_20d),
                    target_vol=float(self.policy.vol_target),
                    beta_to_index=float(self.policy.vol_beta_to_index or snap.beta_to_index or 1.0),
                    index_future=self.policy.index_future,
                    index_contract_value=float(self.policy.index_contract_value),
                    portfolio_value=pv,
                    meta={"book": snap.book, "why": "vol_target"}
                )
                for it in h_ints:
                    intents.append(self._intent_to_order(it, strategy="hedger_vol"))
                notes.extend(h_notes)

        # 2) Book delta hedge (optional)
        if self.policy.delta_target is not None and risk and risk.delta_book is not None:
            cur_delta = float(risk.delta_book)
            if abs(cur_delta - float(self.policy.delta_target)) > float(self.policy.delta_threshold or 0.0):
                # use basket hedge: pick the largest symbol by exposure to nudge
                top = max(snap.positions, key=lambda p: abs(p.qty) * float(snap.prices.get(p.symbol, 0.0) or 0.0), default=None)
                if top:
                    d_ints, d_notes = self.kitchen.delta_hedge(
                        symbol=top.symbol,
                        pos_shares=top.qty,
                        target_delta=float(self.policy.delta_target),
                        current_delta=cur_delta,
                        spot=float(snap.prices.get(top.symbol, 0.0) or 0.0),
                        meta={"book": snap.book, "why": "delta_book"}
                    )
                    for it in d_ints:
                        intents.append(self._intent_to_order(it, strategy="hedger_delta"))
                    notes.extend(d_notes)

        # 3) Protection trigger (drawdown / ES)
        trigger = False
        reason = None
        if risk:
            if self.policy.drawdown_trigger is not None and risk.drawdown_pct is not None:
                if float(risk.drawdown_pct) >= float(self.policy.drawdown_trigger):
                    trigger = True
                    reason = f"drawdown≥{self.policy.drawdown_trigger:.2%}"
            if (not trigger) and (self.policy.es_trigger is not None) and (risk.es_975 is not None):
                if float(risk.es_975) >= float(self.policy.es_trigger):
                    trigger = True
                    reason = f"ES≥{self.policy.es_trigger}"

        if trigger:
            # choose symbols to protect (filtered/universe)
            symbols = self.policy.symbols or [p.symbol for p in snap.positions if p.qty != 0]
            # pick top N exposures (keep it small here; can extend)
            # sort by |notional|
            symbols = sorted(symbols, key=lambda s: abs(float(snap.prices.get(s, 0.0) or 0.0) * _qty_of(snap.positions, s)), reverse=True)
            symbols = symbols[:5]  # cap to 5 names per fire to avoid burst

            for sym in symbols:
                qty = _qty_of(snap.positions, sym)
                if qty == 0:
                    continue
                spot = float(snap.prices.get(sym, 0.0) or 0.0)
                if spot <= 0:
                    continue

                if self.policy.protection_recipe == "protective_put":
                    r_ints, r_notes = self.kitchen.protective_put(
                        symbol=sym, pos_shares=abs(qty), spot=spot,
                        put_delta=self.policy.put_delta,
                        expiry_days=self.policy.expiry_days,
                        ratio=self.policy.ratio,
                        meta={"book": snap.book, "why": reason}
                    )
                elif self.policy.protection_recipe == "put_spread":
                    r_ints, r_notes = self.kitchen.put_spread(
                        symbol=sym, pos_shares=abs(qty), spot=spot,
                        long_put_delta=self.policy.put_delta,
                        short_put_delta=min(-0.1, self.policy.put_delta/2),
                        expiry_days=self.policy.expiry_days,
                        ratio=self.policy.ratio,
                        meta={"book": snap.book, "why": reason}
                    )
                elif self.policy.protection_recipe == "tail_put":
                    r_ints, r_notes = self.kitchen.tail_put(
                        symbol=sym, pos_shares=abs(qty),
                        moneyness=0.8, expiry_days=max(60, self.policy.expiry_days),
                        ratio=max(0.3, self.policy.ratio/2),
                        meta={"book": snap.book, "why": reason}
                    )
                else:  # collar (default)
                    r_ints, r_notes = self.kitchen.collar(
                        symbol=sym, pos_shares=abs(qty), spot=spot,
                        put_delta=self.policy.put_delta, call_delta=self.policy.call_delta,
                        expiry_days=self.policy.expiry_days, ratio=self.policy.ratio,
                        meta={"book": snap.book, "why": reason}
                    )

                for it in r_ints:
                    intents.append(self._intent_to_order(it, strategy="hedger_protect"))
                notes.extend([f"{sym}: " + n for n in r_notes])

        return intents, notes

    def _intent_to_order(self, it, *, strategy: str) -> Dict[str, Any]:
        """
        Map OrderIntent (from recipes) → your order stream schema.
        """
        payload = {
            "ts_ms": _now_ms(),
            "strategy": strategy,
            "symbol": it.symbol,
            "side": it.side,
            "qty": it.qty,
            "typ": getattr(it, "typ", "market"),
            "limit_price": getattr(it, "limit_price", None),
            "venue": getattr(it, "venue", None),
            "asset_class": getattr(it, "asset_class", None),
            "meta": getattr(it, "meta", {}) or {},
        }
        return payload

    # ---- Worker loop (Redis-based) ----
    async def run_forever(self):
        """
        Tails POSITIONS_STREAM + RISK_STREAM; runs evaluate() when both present per book.
        Emits decisions to ORDERS_STREAM and notes to EVENTS_STREAM.
        """
        await self.connect()
        if not self._r and not _consume_stream:
            print("[hedger] No Redis and no bus.consume_stream; cannot run worker loop.")
            return

        last_pos = "$"; last_risk = "$"
        # per-book caches
        pos_cache: Dict[str, Snapshot] = {}
        risk_cache: Dict[str, RiskMetrics] = {}

        while True:
            try:
                if _consume_stream:
                    # If you have your bus wrapper, you can swap to it; here we stick with Redis
                    pass
                if not self._r:
                    await asyncio.sleep(1.0)
                    continue

                resp = await self._r.xread({POSITIONS_STREAM: last_pos, RISK_STREAM: last_risk}, count=200, block=5000)  # type: ignore
                if not resp:
                    continue
                for stream, entries in resp:
                    for _id, fields in entries:
                        if stream == POSITIONS_STREAM:
                            last_pos = _id
                            snap = _parse_snapshot(fields)
                            if snap:
                                pos_cache[snap.book] = snap
                        elif stream == RISK_STREAM:
                            last_risk = _id
                            met = _parse_risk(fields)
                            if met:
                                risk_cache[met.book] = met

                # evaluate books that have both
                for book in list(set(pos_cache.keys()) & set(risk_cache.keys())):
                    snap = pos_cache.get(book)
                    met = risk_cache.get(book)
                    if not snap:
                        continue
                    intents, notes = await self.evaluate(snap, met)
                    if intents:
                        await self._publish_orders(intents)
                    if notes:
                        await self._emit_event({"book": book, "notes": notes, "kind": "decision"})

            except Exception as e:
                await self._emit_event({"lvl": "error", "msg": str(e)})
                await asyncio.sleep(0.5)

# ---------- Parsers for stream payloads ----------
def _parse_snapshot(fields: Dict[str, Any]) -> Optional[Snapshot]:
    try:
        j = json.loads(fields.get("json", "{}"))
        ps = [Position(symbol=str(p["symbol"]).upper(), qty=float(p["qty"])) for p in j.get("positions", [])]
        prices = {str(k).upper(): float(v) for k, v in (j.get("prices") or {}).items()}
        return Snapshot(
            ts_ms=int(j.get("ts_ms") or _now_ms()),
            book=str(j.get("book") or "main"),
            positions=ps,
            prices=prices,
            beta_to_index=float(j.get("beta_to_index") or 1.0),
            portfolio_value=float(j.get("portfolio_value") or 0.0) if j.get("portfolio_value") is not None else None
        )
    except Exception:
        return None

def _parse_risk(fields: Dict[str, Any]) -> Optional[RiskMetrics]:
    try:
        j = json.loads(fields.get("json", "{}"))
        return RiskMetrics(
            ts_ms=int(j.get("ts_ms") or _now_ms()),
            book=str(j.get("book") or "main"),
            vol_20d=float(j["vol_20d"]) if j.get("vol_20d") is not None else None,
            drawdown_pct=float(j["drawdown_pct"]) if j.get("drawdown_pct") is not None else None,
            es_975=float(j["es_975"]) if j.get("es_975") is not None else None,
            delta_book=float(j["delta_book"]) if j.get("delta_book") is not None else None
        )
    except Exception:
        return None

def _qty_of(positions: List[Position], symbol: str) -> float:
    for p in positions:
        if p.symbol.upper() == symbol.upper():
            return float(p.qty)
    return 0.0

# ---------- Quick CLI ----------
async def _demo_once():
    """
    Fire a single evaluation with synthetic data (no Redis required).
    """
    hedger = Hedger()
    # synthetic book
    snap = Snapshot(
        ts_ms=_now_ms(),
        book="main",
        positions=[Position("AAPL", 10000), Position("MSFT", 8000)],
        prices={"AAPL": 200.0, "MSFT": 400.0},
        beta_to_index=1.1
    )
    risk = RiskMetrics(
        ts_ms=_now_ms(),
        book="main",
        vol_20d=0.28,          # above 0.20 target → short futures overlay
        drawdown_pct=0.09,     # 9% drawdown → protection trigger
        es_975=0.0,
        delta_book=5000.0
    )
    intents, notes = await hedger.evaluate(snap, risk)
    print("NOTES:")
    for n in notes: print(" -", n)
    print("ORDERS:")
    for it in intents: print(" -", json.dumps(it))

def _main():
    import argparse, asyncio as aio
    ap = argparse.ArgumentParser("hedger")
    ap.add_argument("--demo", action="store_true", help="run one-shot demo without Redis")
    args = ap.parse_args()
    if args.demo:
        aio.run(_demo_once())
        return
    hedger = Hedger()
    aio.run(hedger.run_forever())

if __name__ == "__main__":
    _main()