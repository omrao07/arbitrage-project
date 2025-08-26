# backend/risk/risk_manager.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

# ----------------------- Soft dependencies (kept optional) -------------------
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

try:
    # Your bus helpers; we keep them soft so this file runs standalone.
    # Expected signatures:
    #   publish_stream(stream: str, payload: dict) -> None
    #   consume_stream(stream: str, start_id: str = '$', block_ms: int = 1000, count: int = 200) -> Iterator[(id, dict)]
    #   hset(key: str, field: str, value: Any) -> None
    from backend.bus.streams import publish_stream, consume_stream, hset  # type: ignore
except Exception:
    def publish_stream(_s, _p): pass
    def hset(_k, _f, _v): pass
    def consume_stream(_s, start_id: str = "$", block_ms: int = 1000, count: int = 200):
        while False:
            yield "0-0", {}

try:
    # Governor from earlier file
    from backend.risk.governor import Governor, Policy as GovPolicy  # type: ignore
except Exception:
    Governor = None
    class GovPolicy: ...

try:
    # Optional: adversary toxicity pre-checks
    from backend.risk.adversary import AdversarySuite, default_suite, GuardrailPolicy  # type: ignore
except Exception:
    AdversarySuite = None
    def default_suite(*a, **k): return None
    class GuardrailPolicy:
        def decide(self, precheck, *, default="CROSS"): return {"mode": default, "tox": 0.0}

try:
    # Optional: sizing helper
    from backend.risk.liquidity_surface import LiquiditySurface  # type: ignore
except Exception:
    LiquiditySurface = None

# ----------------------- Env / streams ---------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

INCOMING_ORDERS = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")      # from strategies
VALIDATED_ORDERS = os.getenv("RISK_OUT_STREAM", "orders.validated")          # to OMS/router
REJECTED_ORDERS  = os.getenv("RISK_REJECT_STREAM", "orders.rejected")        # audit for UI
HEALTH_STREAM    = os.getenv("RISK_HEALTH_STREAM", "risk.health")            # heartbeats

_R = None
if redis:
    try:
        _R = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    except Exception:
        _R = None

# ----------------------- Data models -----------------------------------------

@dataclass
class OrderDecision:
    allowed: bool
    reason: str = "ok"
    scaled_qty: float = 0.0
    scale: float = 1.0
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ----------------------- NBBO cache (tiny helper) ----------------------------

class NbboCache:
    """
    Minimal NBBO cache the router can update. Risk manager uses it for spread guards.
    """
    def __init__(self):
        self._m: Dict[str, Dict[str, float]] = {}

    def update(self, symbol: str, bid: float, ask: float, ts: Optional[float] = None) -> None:
        self._m[symbol.upper()] = {"bid": float(bid), "ask": float(ask), "ts": float(ts or time.time())}

    def get(self, symbol: str) -> Optional[Dict[str, float]]:
        return self._m.get(symbol.upper())

# ----------------------- Risk Manager ----------------------------------------

class RiskManager:
    """
    Central gate between strategies and OMS.

    Pipeline per order:
      1) Normalize & enrich (compute notional, pos_after if provided)
      2) (optional) Adversary pre-check → toxicity score
      3) Governor.check_and_scale(...) → Decision (allow/scale/reject/kill)
      4) Publish to VALIDATED_ORDERS or REJECTED_ORDERS

    Also ingests fills/marks via methods so governor stats stay current.
    """

    def __init__(
        self,
        *,
        governor: Optional[Governor] = None, # type: ignore
        adv_suite: Optional[AdversarySuite] = None, # type: ignore
        guard_policy: Optional[GuardrailPolicy] = None,
        liq_surface: Optional[LiquiditySurface] = None, # type: ignore
        nbbo: Optional[NbboCache] = None,
    ):
        self.gov = governor or (Governor(GovPolicy()) if Governor else None)
        self.advs = adv_suite or default_suite()
        self.guard = guard_policy or GuardrailPolicy()
        self.liq = liq_surface
        self.nbbo = nbbo or NbboCache()

        # lightweight position/exposure view (for pos_notional_after)
        self._pos_notional: Dict[Tuple[str, str], float] = {}  # (strategy, symbol) -> notional (signed)
        self._last_heartbeat = 0.0

    # -------------------- External state updates -----------------------------

    def ingest_fill(self, *, strategy: str, symbol: str, qty: float, price: float, side: str, fees: float = 0.0) -> None:
        """
        Update governor stats and local pos view after a confirmed trade.
        """
        signed = float(qty) if side.lower() == "buy" else -float(qty)
        key = (strategy, symbol.upper())
        cur = self._pos_notional.get(key, 0.0)
        self._pos_notional[key] = cur + signed * float(price)

        if self.gov:
            self.gov.ingest_fill(strategy=strategy, symbol=symbol, qty=qty, price=price, side=side, fees=fees)

    def ingest_mark(self, *, strategy: Optional[str] = None, pnl_delta: float = 0.0, dd_frac: Optional[float] = None) -> None:
        """
        Forward PnL marks to governor (e.g., from PnL engine).
        """
        if self.gov:
            self.gov.ingest_mark(strategy=strategy, pnl_delta=float(pnl_delta), dd_frac=dd_frac)

    def update_outstanding(self, *, strategy: str, gross_outstanding: float) -> None:
        if self.gov:
            self.gov.update_outstanding(strategy=strategy, gross_outstanding=gross_outstanding)

    # -------------------- Core processing ------------------------------------

    def process_order(self, order: Dict[str, Any]) -> OrderDecision:
        """
        Enrich → guardrails → governor → (scale|reject).
        `order` expected keys:
            strategy, symbol, side, qty, price (or mark_price), typ ("market"/"limit")
            optional: region, venue, notional, vol_daily
        """
        o = dict(order)  # shallow copy
        strat = str(o.get("strategy",""))
        symbol = str(o.get("symbol","")).upper()
        side = str(o.get("side","")).lower()
        qty  = float(o.get("qty", 0.0))
        mark = float(o.get("mark_price") or o.get("price") or 0.0)
        notional = float(o.get("notional") or abs(qty * mark))

        # pos after (signed notional) for position caps if provided by router; else rough estimate
        key = (strat, symbol)
        cur_notional = self._pos_notional.get(key, 0.0)
        signed = qty if side == "buy" else -qty
        pos_after = cur_notional + signed * mark
        o["pos_notional_after"] = pos_after
        o["notional"] = notional
        o["price"] = mark  # ensure field present

        # pick NBBO if available
        nbbo = self.nbbo.get(symbol)

        # toxicity pre-check (optional)
        tox = None
        if self.advs:
            try:
                pre = self.advs.pre_trade_check(_as_order(o), _as_nbbo(nbbo))
                tox = float(pre.get("toxicity", 0.0))
            except Exception:
                tox = None

        # Let the Governor decide
        if not self.gov:
            # permissive fallback
            return OrderDecision(True, "no_governor", scaled_qty=qty, scale=1.0, notes={"fallback": True})

        dec = self.gov.check_and_scale(o, nbbo=nbbo, adversary_toxicity=tox)
        if not dec.allowed:
            # Audit reject
            self._emit_reject(o, dec.reason, extra={"notes": dec.notes, "tox": tox})
            return OrderDecision(False, dec.reason, 0.0, 0.0, notes=dec.notes)

        scaled_qty = qty * float(dec.scale or 1.0)
        o["qty"] = scaled_qty
        o["scaled_from_qty"] = qty
        o["scale"] = dec.scale

        # Optional: clip single-child size via liquidity surface cap (if supplied)
        if self.liq is not None:
            try:
                adv = float(o.get("adv_notional") or 0.0)
                vol = float(o.get("vol_daily") or 0.0)
                cap_bps = float(os.getenv("RISK_IMPACT_CAP_BPS", "10.0"))
                child_cap_notional = self.liq.max_child_notional_for_cap(
                    adv_notional=adv, vol_daily=vol, cap_bps=cap_bps
                )
                if child_cap_notional > 0 and o["notional"] > child_cap_notional:
                    # shrink qty to impact cap
                    new_qty = child_cap_notional / max(1e-9, mark)
                    o["qty"] = new_qty
                    o["scale"] = new_qty / max(1e-9, qty)
                    o.setdefault("notes", {})["liquidity_cap_bps"] = cap_bps
            except Exception:
                pass

        # Push to OMS
        publish_stream(VALIDATED_ORDERS, o)
        return OrderDecision(True, "ok", scaled_qty=float(o["qty"]), scale=float(o.get("scale", 1.0)), notes={"tox": tox})

    # -------------------- Streams runner -------------------------------------

    def run(self, start_id: str = "$") -> None:
        """
        Long-running worker:
          - consumes INCOMING_ORDERS
          - processes, publishes to VALIDATED_ORDERS or REJECTED_ORDERS
        """
        # heartbeat mark-up so your UI knows it's alive
        self._heartbeat("starting")
        for _, msg in consume_stream(INCOMING_ORDERS, start_id=start_id, block_ms=1000, count=200): # type: ignore
            if not msg:
                self._heartbeat("idle")
                continue
            try:
                if isinstance(msg, str):
                    msg = json.loads(msg)
                decision = self.process_order(msg)
                # small health ping each batch
                self._heartbeat("ok")
            except Exception as e:
                self._emit_reject(msg if isinstance(msg, dict) else {"raw": str(msg)}, "exception", extra={"err": str(e)})
                self._heartbeat("error")

    # -------------------- Internals ------------------------------------------

    def _emit_reject(self, order: Dict[str, Any], reason: str, *, extra: Dict[str, Any] | None = None) -> None:
        payload = {
            "ts_ms": int(time.time()*1000),
            "reason": reason,
            "order": order,
            "extra": extra or {},
        }
        try:
            publish_stream(REJECTED_ORDERS, payload)
        except Exception:
            pass
        if _R:
            try:
                _R.lpush("risk:rejects", json.dumps(payload))
                _R.ltrim("risk:rejects", 0, 999)
            except Exception:
                pass

    def _heartbeat(self, status: str) -> None:
        now = time.time()
        if now - self._last_heartbeat < 3.0:
            return
        self._last_heartbeat = now
        try:
            publish_stream(HEALTH_STREAM, {"ts_ms": int(now*1000), "service": "risk_manager", "status": status})
        except Exception:
            pass

# ----------------------- Tiny helpers ----------------------------------------

def _as_nbbo(nbbo: Optional[Dict[str, float]]) -> Any:
    if not nbbo:
        return {"bid": None, "ask": None}
    return {"bid": float(nbbo.get("bid") or 0.0), "ask": float(nbbo.get("ask") or 0.0), "ts": float(nbbo.get("ts") or time.time())}

def _as_order(o: Dict[str, Any]) -> Any:
    # minimal Order-like view for adversary precheck
    return {
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "qty": float(o.get("qty") or 0.0),
        "tif": o.get("tif", "IOC" if o.get("typ","market") == "market" else "DAY"),
        "limit_price": o.get("limit_price"),
        "meta": o.get("meta", {}),
    }

# ----------------------- CLI smoke test --------------------------------------

if __name__ == "__main__":
    """
    Quick smoke:
      export REDIS_HOST=... if needed
      python -m backend.risk.risk_manager
    Then publish a test order into 'orders.incoming' using redis-cli or your code.
    """
    rm = RiskManager()
    try:
        # mark a small NBBO so spread guards can work if Governor uses them
        rm.nbbo.update("AAPL", bid=192.00, ask=192.02)
        # register a strategy with the governor (optional)
        if rm.gov:
            rm.gov.register_strategy("alpha.example")

        print("[risk_manager] listening on stream:", INCOMING_ORDERS)
        rm.run("$")  # new messages only
    except KeyboardInterrupt:
        print("\n[risk_manager] stopped.")