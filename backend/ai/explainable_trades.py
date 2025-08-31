# backend/ai/explainable_trades.py
from __future__ import annotations

import os
import json
import time
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Optional Redis
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    AsyncRedis = None  # type: ignore
    USE_REDIS = False

# ---------- Env / streams ----------------------------------------------------
REDIS_URL              = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ORDERS_UPDATES_STREAM  = os.getenv("ORDERS_UPDATES_STREAM", "orders.updates")  # OMS emits acks/fills here
SIGNALS_STREAM         = os.getenv("SIGNALS_STREAM", "signals.analyst")        # Analyst/alpha signals
POLICY_DECISIONS       = os.getenv("POLICY_DECISIONS_STREAM", "govern.decisions")
RISK_STATE_HASH        = os.getenv("RISK_STATE_HASH", "risk.state")            # HGETALL snapshot: pos, dd, VaR, limits
FEATURE_STORE_STREAM   = os.getenv("FEATURE_STORE_STREAM", "features.store")   # optional numeric features
EXPLAIN_STREAM         = os.getenv("EXPLAIN_TRADES_STREAM", "explain.trades")  # output
MAXLEN                 = int(os.getenv("EXPLAIN_MAXLEN", "20000"))

# ---------- Simple LLM hook (optional) ---------------------------------------
class LLMProvider:
    """Swap with your real provider if desired (OpenAI, local LLM, etc.)."""
    def summarize(self, bullets: List[str], limit: int = 320) -> str:
        # Offline fallback: join bullets and trim
        txt = " ".join(bullets)
        return (txt[: limit - 3] + "...") if len(txt) > limit else txt

# ---------- Data models ------------------------------------------------------
@dataclass
class TradeEvent:
    """Normalized order update/fill from OMS."""
    id: str
    ts_ms: int
    symbol: str
    side: str                # 'buy' | 'sell'
    qty: float
    status: str              # working|partial|filled|canceled|rejected
    price: Optional[float] = None
    venue: Optional[str] = None
    strategy: Optional[str] = None
    region: Optional[str] = None
    meta: Dict[str, Any] = None # type: ignore

@dataclass
class ContextSnapshot:
    """State around the moment of execution used for explanation."""
    spot: Optional[float] = None
    vol: Optional[float] = None
    spread_bps: Optional[float] = None
    position: Optional[float] = None
    exposure_usd: Optional[float] = None
    dd: Optional[float] = None              # drawdown fraction (e.g., 0.06)
    var_1d: Optional[float] = None
    policy_blocked: bool = False
    policy_reason: Optional[str] = None
    governor_halt: bool = False
    features: Dict[str, float] = None       # type: ignore # last numeric features (e.g., analyst_confidence)
    signal: Optional[Dict[str, Any]] = None # last strategy/analyst signal

@dataclass
class Explanation:
    id: str
    order_id: str
    ts_ms: int
    symbol: str
    side: str
    qty: float
    fill_price: Optional[float]
    summary: str
    bullets: List[str]
    checklist: Dict[str, bool]
    drivers: Dict[str, Any]          # factors/features that drove decision
    context: Dict[str, Any]          # snapshot fields for audit
    meta: Dict[str, Any]

# ---------- Small caches (latest state by symbol/strategy) -------------------
class _LRU:
    def __init__(self, cap: int = 5000):
        self.cap = cap
        self.map: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
    def set(self, k: str, v: Dict[str, Any]):
        if k in self.map:
            self.order.remove(k)
        self.map[k] = v
        self.order.append(k)
        if len(self.order) > self.cap:
            old = self.order.pop(0)
            self.map.pop(old, None)
    def get(self, k: str) -> Optional[Dict[str, Any]]:
        return self.map.get(k)

_last_signal_by_symbol = _LRU()
_last_features_by_symbol = _LRU()

# ---------- Redis helpers ----------------------------------------------------
async def _redis() -> Optional[AsyncRedis]: # type: ignore
    if not USE_REDIS: return None
    try:
        r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        await r.ping()
        return r
    except Exception:
        return None

async def _xread(r: AsyncRedis, stream: str, last_id: str, count: int = 200, block_ms: int = 5000): # type: ignore
    try:
        return await r.xread({stream: last_id}, count=count, block=block_ms)
    except Exception:
        return None

async def _xadd(r: AsyncRedis, stream: str, payload: Dict[str, Any]): # type: ignore
    try:
        await r.xadd(stream, {"json": json.dumps(payload)}, maxlen=MAXLEN, approximate=True)
    except Exception:
        pass

async def _hgetall(r: AsyncRedis, key: str) -> Dict[str, Any]: # type: ignore
    try:
        return await r.hgetall(key) or {}
    except Exception:
        return {}

# ---------- Context builders -------------------------------------------------
def _bps(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        return (float(a) - float(b)) / float(b) * 1e4 # type: ignore
    except Exception:
        return None

def _order_to_trade(ev: Dict[str, Any]) -> TradeEvent:
    return TradeEvent(
        id=str(ev.get("id") or ev.get("order_id")),
        ts_ms=int(ev.get("ts_ms") or int(time.time() * 1000)),
        symbol=(ev.get("symbol") or "").upper(),
        side=str(ev.get("side") or ev.get("direction") or "buy").lower(),
        qty=float(ev.get("qty") or ev.get("quantity") or 0),
        status=str(ev.get("status") or "working"),
        price=(float(ev["fill_price"]) if "fill_price" in ev else float(ev.get("price") or ev.get("mark_price") or 0.0)) or None,
        venue=ev.get("venue"),
        strategy=ev.get("strategy"),
        region=ev.get("region"),
        meta=ev.get("meta") or {},
    )

async def _gather_context(r: Optional[AsyncRedis], t: TradeEvent) -> ContextSnapshot: # type: ignore
    # Start blank; enrich from caches and Redis if available
    ctx = ContextSnapshot(features={}, signal={})
    # recent signal/features
    sig = _last_signal_by_symbol.get(t.symbol)
    if sig: ctx.signal = sig
    feats = _last_features_by_symbol.get(t.symbol)
    if feats: ctx.features = feats

    # risk state snapshot (positions, dd, var, policy flags)
    if r:
        rs = await _hgetall(r, RISK_STATE_HASH)
        # Expect optional fields like pos:{SYM}, spot:{SYM}, vol:{SYM}, dd, var_1d, halt, policy_block
        try: ctx.position = float(rs.get(f"pos:{t.symbol}", "nan"))
        except Exception: pass
        try: ctx.spot = float(rs.get(f"spot:{t.symbol}", "nan"))
        except Exception: pass
        try: ctx.vol = float(rs.get(f"vol:{t.symbol}", "nan"))
        except Exception: pass
        try: ctx.exposure_usd = float(rs.get(f"exposure:{t.symbol}", "nan"))
        except Exception: pass
        try: ctx.dd = float(rs.get("dd", "nan"))
        except Exception: pass
        try: ctx.var_1d = float(rs.get("var_1d", "nan"))
        except Exception: pass
        halt = str(rs.get("govern:halt_trading", "false")).lower() in {"1","true","yes","halt"}
        ctx.governor_halt = halt
        # policy decisions stream can be mirrored into the hash by your governor; keep reason if present
        ctx.policy_blocked = str(rs.get("policy:block", "false")).lower() in {"1","true"}
        ctx.policy_reason = rs.get("policy:reason")

    # derive spread if signal has best bid/ask
    try:
        bid = float(ctx.signal.get("best_bid")) if ctx.signal else math.nan # type: ignore
        ask = float(ctx.signal.get("best_ask")) if ctx.signal else math.nan # type: ignore
        if bid and ask and ask > 0:
            ctx.spread_bps = (ask - bid) / ((ask + bid) / 2.0) * 1e4
    except Exception:
        pass

    return ctx

# ---------- Explanation core -------------------------------------------------
def _mk_summary(t: TradeEvent, c: ContextSnapshot) -> str:
    bits = []
    s_side = "Bought" if t.side == "buy" else "Sold"
    px = f" @ {t.price:.4f}" if t.price else ""
    bits.append(f"{s_side} {t.qty:g} {t.symbol}{px}.")
    if c.signal and "direction" in c.signal:
        dir_word = c.signal.get("direction")
        conf = c.signal.get("confidence")
        if conf is not None:
            bits.append(f"Signal: {dir_word} (conf {float(conf):.2f}).")
        else:
            bits.append(f"Signal: {dir_word}.")
    if c.position is not None:
        bits.append(f"New pos≈ {c.position:+g}.")
    if c.var_1d is not None:
        bits.append(f"Portfolio VaR≈ {c.var_1d:.2f}.")
    if c.policy_blocked:
        bits.append(f"Policy override: {c.policy_reason or 'blocked by policy'}.")
    return " ".join(bits)

def _mk_bullets(t: TradeEvent, c: ContextSnapshot) -> List[str]:
    b: List[str] = []
    if c.signal:
        if "rationale" in c.signal:
            b.append(c.signal["rationale"])
        s = c.signal.get("score")
        if s is not None:
            b.append(f"Signal score {float(s):+.2f}.")
        ac = c.features.get("analyst_confidence") if c.features else None
        if ac is not None:
            b.append(f"Analyst confidence {float(ac):.2f}.")
    if c.spot and t.price:
        dev = _bps(t.price, c.spot)
        if dev is not None:
            b.append(f"Fill vs spot: {dev:+.1f} bps.")
    if c.spread_bps is not None:
        b.append(f"Spread ~{c.spread_bps:.1f} bps.")
    if c.dd is not None:
        b.append(f"Drawdown {c.dd*100:.1f}% {'(risk-on)' if c.dd < 0.05 else '(caution)'}")
    if c.policy_blocked:
        b.append(f"Policy decision: {c.policy_reason or 'blocked'}.")
    return b[:8]

def _mk_checklist(t: TradeEvent, c: ContextSnapshot) -> Dict[str, bool]:
    return {
        "governor_halt": not c.governor_halt,
        "policy_allow": not c.policy_blocked,
        "size_nonzero": t.qty > 0,
        "has_symbol": bool(t.symbol),
        "price_ok": (t.price or 0.0) >= 0.0,
    }

def _mk_drivers(t: TradeEvent, c: ContextSnapshot) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if c.signal:
        d["signal"] = {k: c.signal[k] for k in ("direction","score","confidence") if k in c.signal}
    if c.features:
        d["features"] = c.features
    if c.spot is not None:
        d["spot"] = c.spot
    if c.vol is not None:
        d["vol"] = c.vol
    if c.var_1d is not None:
        d["var_1d"] = c.var_1d
    if c.position is not None:
        d["position"] = c.position
    return d

def _exp_id(order_id: str, ts_ms: int) -> str:
    return hashlib.sha1(f"{order_id}|{ts_ms}".encode()).hexdigest()

# ---------- Orchestrator -----------------------------------------------------
class ExplainableTrades:
    def __init__(self, use_llm: bool = False):
        self.llm = LLMProvider() if use_llm else None
        self.r: Optional[AsyncRedis] = None # type: ignore

    async def connect(self):
        self.r = await _redis()

    async def run_forever(self):
        if self.r is None:
            await self.connect()
        r = self.r
        if r is None:
            # Without Redis we cannot tail order updates; idle
            while True:
                await asyncio_sleep(1.0)

        last_orders = "$"
        last_signals = "$"
        last_features = "$"
        last_policy = "$"

        # warm small reads (non-blocking) to avoid initial miss
        try:
            await r.xread({ORDERS_UPDATES_STREAM: last_orders}, count=1, block=10)  # type: ignore
        except Exception:
            pass

        while True:
            try:
                # multiplex reads from multiple streams
                resp = await r.xread({  # type: ignore
                    ORDERS_UPDATES_STREAM: last_orders,
                    SIGNALS_STREAM: last_signals,
                    FEATURE_STORE_STREAM: last_features,
                    POLICY_DECISIONS: last_policy,
                }, count=200, block=5000)

                if not resp:
                    continue

                # xread returns list of (stream, entries)
                for stream_key, entries in resp:
                    if stream_key == ORDERS_UPDATES_STREAM:
                        for _id, fields in entries:
                            last_orders = _id
                            await self._handle_order_update(fields)
                    elif stream_key == SIGNALS_STREAM:
                        for _id, fields in entries:
                            last_signals = _id
                            _ingest_signal(fields)
                    elif stream_key == FEATURE_STORE_STREAM:
                        for _id, fields in entries:
                            last_features = _id
                            _ingest_features(fields)
                    elif stream_key == POLICY_DECISIONS:
                        last_policy = entries[-1][0] if entries else last_policy
            except Exception:
                # light backoff
                await asyncio_sleep(0.5)

    async def _handle_order_update(self, fields: Dict[str, Any]):
        try:
            ev = json.loads(fields.get("json", "{}"))
        except Exception:
            return
        t = _order_to_trade(ev)

        # Only explain fills or meaningful state changes
        if t.status not in {"filled", "partial", "replaced", "working"}:
            return

        r = self.r
        ctx = await _gather_context(r, t)

        # Build explanation
        bullets = _mk_bullets(t, ctx)
        summary = _mk_summary(t, ctx)
        if self.llm:
            # optional: compress bullets to a neat sentence
            summary = self.llm.summarize([summary] + bullets, limit=320)

        check = _mk_checklist(t, ctx)
        drivers = _mk_drivers(t, ctx)

        exp = Explanation(
            id=_exp_id(t.id, t.ts_ms),
            order_id=t.id,
            ts_ms=t.ts_ms,
            symbol=t.symbol,
            side=t.side,
            qty=t.qty,
            fill_price=t.price,
            summary=summary,
            bullets=bullets,
            checklist=check,
            drivers=drivers,
            context={
                "spot": ctx.spot, "vol": ctx.vol, "spread_bps": ctx.spread_bps,
                "position": ctx.position, "exposure_usd": ctx.exposure_usd,
                "dd": ctx.dd, "var_1d": ctx.var_1d,
                "policy_blocked": ctx.policy_blocked, "policy_reason": ctx.policy_reason,
                "governor_halt": ctx.governor_halt,
            },
            meta={"source": "explainable_trades", "strategy": t.strategy, "venue": t.venue, **(t.meta or {})},
        )

        payload = asdict(exp)
        if r:
            await _xadd(r, EXPLAIN_STREAM, payload)
        else:
            print("[explain.trades]", json.dumps(payload)[:800])

# ---------- Ingest helpers for caches ---------------------------------------
def _ingest_signal(fields: Dict[str, Any]) -> None:
    try:
        j = json.loads(fields.get("json", "{}"))
        sym = (j.get("symbol") or "").upper()
        if not sym:
            return
        keep = {k: j.get(k) for k in ("direction","confidence","score","rationale","best_bid","best_ask")}
        _last_signal_by_symbol.set(sym, keep)
    except Exception:
        pass

def _ingest_features(fields: Dict[str, Any]) -> None:
    try:
        j = json.loads(fields.get("json", "{}"))
        sym = (j.get("symbol") or "").upper()
        feats = j.get("features")
        if not sym or not isinstance(feats, dict):
            return
        _last_features_by_symbol.set(sym, {k: float(v) for k, v in feats.items() if _is_float(v)})
    except Exception:
        pass

def _is_float(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

# ---------- tiny asyncio util -----------------------------------------------
async def asyncio_sleep(sec: float):
    try:
        import asyncio
        await asyncio.sleep(sec)
    except Exception:
        time.sleep(sec)

# ---------- CLI --------------------------------------------------------------
async def _amain():
    et = ExplainableTrades(use_llm=False)
    await et.connect()
    print("[explainable_trades] started. streams:",
          ORDERS_UPDATES_STREAM, SIGNALS_STREAM, FEATURE_STORE_STREAM, "->", EXPLAIN_STREAM)
    await et.run_forever()

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass