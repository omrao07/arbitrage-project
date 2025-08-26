# backend/oms/router.py
"""
Smart Order Router (SOR)
------------------------
Route parent orders to venues/brokers with cost/latency/liquidity awareness.

Key features
- Risk pre-check hook (backend.oms.risk_manager)
- Venue scoring: combines liquidity signal, latency estimate, and cost model bps
- Slicing: TWAP/VWAP/POV/Immediate; min child size; participation caps
- Time-in-force: DAY / IOC / FOK / POST_ONLY (maker)
- Throttling per broker/venue to avoid API spam
- Persistence to order_store + eventing on bus

Public API
----------
router = Router()
child_ids = router.route(parent_order_dict)

CLI probe
---------
python -m backend.oms.router --probe
"""

from __future__ import annotations

import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# -------- optional deps / project glue (graceful fallbacks) ----------
try:
    from backend.utils.throttle import throttle # type: ignore
except Exception:
    class _Dummy:  # fallback no-op
        def limit(self, *a, **k):
            def deco(fn): return fn
            return deco
        def wait(self, *a, **k): pass
    throttle = _Dummy()  # type: ignore

try:
    from backend.oms.cost_model import CostModel
except Exception:
    CostModel = None  # type: ignore

try:
    import backend.oms.broker_interface as broker_interface # type: ignore
except Exception:
    broker_interface = None

try:
    import backend.oms.order_store as order_store # type: ignore
except Exception:
    order_store = None

try:
    import backend.oms.risk_manager as risk_manager # type: ignore
except Exception:
    risk_manager = None

# optional data adapters
try:
    import backend.data.liquidity_adapter as liquidity_adapter  # type: ignore # exposes get_book/imbalance/liquidity_hint
except Exception:
    liquidity_adapter = None
try:
    import backend.data.latency_adapter as latency_adapter  # type: ignore # exposes get_latency_ms(venue)
except Exception:
    latency_adapter = None

# bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None

# ---------- small utils ----------
def _utc_ms() -> int: return int(time.time() * 1000)
def _uid(prefix: str) -> str: return f"{prefix}_{uuid.uuid4().hex[:12]}"
def _upper(x: Optional[str]) -> Optional[str]: return x.upper() if isinstance(x, str) else x

# ---------- config model ----------
@dataclass
class VenueConfig:
    name: str
    broker: str
    max_participation: float = 0.10     # cap of ADV participation per order (heuristic)
    min_child_qty: float = 1.0
    tif_default: str = "IOC"            # DAY|IOC|FOK|POST_ONLY
    allow_dark: bool = False
    maker_rebate_bps: float = 0.0       # positive = rebate, negative = fee
    taker_fee_bps: float = 0.5
    priority: int = 50                  # tie-break; lower is better

@dataclass
class RouterConfig:
    venues: List[VenueConfig] = field(default_factory=list)
    algo: str = "POV"                   # POV|TWAP|VWAP|IMMEDIATE
    pov: float = 0.10                   # slice as % of est. interval volume
    twap_slices: int = 5
    clock_s: int = 2                    # pacing between child orders
    max_children: int = 50
    max_total_slippage_bps: float = 25.0
    hard_qty_cap: Optional[float] = None
    use_cost_model: bool = True
    region: Optional[str] = None

    @staticmethod
    def default() -> "RouterConfig":
        return RouterConfig(
            venues=[
                VenueConfig(name="NSE",  broker="zerodha", tif_default="IOC", min_child_qty=1, taker_fee_bps=0.3),
                VenueConfig(name="BSE",  broker="zerodha", tif_default="IOC", min_child_qty=1, taker_fee_bps=0.5),
                VenueConfig(name="NASDAQ", broker="ibkr",   tif_default="IOC", min_child_qty=1, taker_fee_bps=0.3),
            ],
            algo=os.getenv("ROUTER_ALGO", "POV"),
            pov=float(os.getenv("ROUTER_POV", "0.10")),
            twap_slices=int(os.getenv("ROUTER_TWAP_SLICES", "5")),
            clock_s=int(os.getenv("ROUTER_CLOCK_S", "2")),
            max_children=int(os.getenv("ROUTER_MAX_CHILDREN", "50")),
            max_total_slippage_bps=float(os.getenv("ROUTER_MAX_SLIP_BPS", "25.0")),
            use_cost_model=True,
            region=os.getenv("ROUTER_REGION") or None,
        )

# ---------- router core ----------
class Router:
    def __init__(self, cfg: Optional[RouterConfig] = None, cost_model: Optional[Any] = None):
        self.cfg = cfg or RouterConfig.default()
        self.cm = cost_model or (CostModel() if (CostModel is not None and self.cfg.use_cost_model) else None)

    # -------- public API --------
    def route(self, parent: Dict[str, Any]) -> List[str]:
        """
        parent fields expected (dict): symbol, side, qty, typ, limit_price?, strategy?, region?, venue?
        Returns list of child order ids.
        """
        sym = _upper(parent.get("symbol"))
        side = str(parent.get("side","buy")).lower()
        qty  = float(parent.get("qty") or 0)
        if qty <= 0 or not sym:
            return []

        # risk pre-check (best-effort)
        if risk_manager and hasattr(risk_manager, "precheck"):
            ok, reason = True, None
            try:
                ok, reason = risk_manager.precheck(parent)
            except Exception:
                ok, reason = True, None
            if not ok:
                self._emit("rejected", {"parent": parent, "reason": reason})
                return []

        # optional hard cap
        if self.cfg.hard_qty_cap and qty > self.cfg.hard_qty_cap:
            qty = self.cfg.hard_qty_cap

        # plan slices
        plan = self._build_plan(parent)

        child_ids: List[str] = []
        for slc in plan:
            venue = slc["venue"]
            vcfg = self._venue_cfg(venue)
            if not vcfg:
                continue

            # score venue (liquidity, latency, cost)
            score = self._score_venue(sym, side, vcfg, parent, slc)
            if score is None:
                continue

            child = {
                "id": _uid("child"),
                "parent_id": parent.get("id") or parent.get("parent_id") or _uid("parent"),
                "ts_ms": _utc_ms(),
                "symbol": sym,
                "side": side,
                "qty": slc["qty"],
                "typ": parent.get("typ") or "market",
                "limit_price": parent.get("limit_price"),
                "venue": vcfg.name,
                "broker": vcfg.broker,
                "tif": parent.get("tif") or vcfg.tif_default,
                "strategy": parent.get("strategy") or "unknown",
                "meta": {
                    "algo": self.cfg.algo,
                    "slice_idx": slc["i"],
                    "slices": slc["n"],
                    "venue_score": score,
                },
            }

            # persist locally
            if order_store and hasattr(order_store, "add_order"):
                try:
                    order_store.add_order(child)
                except Exception:
                    pass

            # publish event
            self._emit("dispatch", {"child": child})

            # send to broker (throttled per broker)
            placed_ok = self._send(child)
            if placed_ok:
                child_ids.append(child["id"])

            # pacing
            time.sleep(max(0.0, self.cfg.clock_s))

        return child_ids

    # -------- planning & scoring --------
    def _build_plan(self, parent: Dict[str, Any]) -> List[Dict[str, Any]]:
        qty = float(parent.get("qty") or 0)
        venues = self._allowed_venues(parent)
        if not venues or qty <= 0:
            return []

        algo = (parent.get("algo") or self.cfg.algo or "POV").upper()
        pieces: List[Tuple[str, float]] = []

        if algo == "IMMEDIATE":
            # one shot: send to best venue later
            best = venues[0].name
            pieces.append((best, qty))
        elif algo == "TWAP":
            n = max(1, min(self.cfg.twap_slices, self.cfg.max_children))
            per = qty / n
            for i in range(n):
                pieces.append((venues[i % len(venues)].name, per))
        elif algo == "VWAP":
            # if we lack real intraday curve, approximate with TWAP for now
            n = max(1, min(self.cfg.twap_slices, self.cfg.max_children))
            per = qty / n
            for i in range(n):
                pieces.append((venues[i % len(venues)].name, per))
        else:  # POV default
            pov = min(0.99, max(0.01, float(parent.get("pov") or self.cfg.pov)))
            # Estimate interval volume per venue
            for v in venues:
                est_vol = self._venue_volume_hint(parent, v)
                slice_qty = max(v.min_child_qty, pov * est_vol)
                pieces.append((v.name, min(slice_qty, qty)))  # will be truncated later

        # enforce min child, participation cap, and total <= qty
        plan: List[Dict[str, Any]] = []
        rem = qty
        i = 0
        for venue_name, q in pieces:
            vcfg = self._venue_cfg(venue_name)
            if not vcfg or rem <= 0:
                break
            # cap by participation (if we have ADV hint)
            cap = self._cap_by_participation(parent, vcfg, q)
            child_q = max(vcfg.min_child_qty, min(rem, cap))
            if child_q <= 0:
                continue
            i += 1
            plan.append({"venue": venue_name, "qty": child_q, "i": i, "n": len(pieces)})
            rem -= child_q
            if len(plan) >= self.cfg.max_children:
                break

        if rem > 0 and plan:
            # top up last slice with the remainder
            plan[-1]["qty"] += rem

        return plan

    def _allowed_venues(self, parent: Dict[str, Any]) -> List[VenueConfig]:
        want = _upper(parent.get("venue"))  # if caller forces a venue
        if want:
            v = self._venue_cfg(want)
            return [v] if v else []
        # region filter if provided
        if self.cfg.region:
            return [v for v in self.cfg.venues if self.cfg.region.upper() in (v.name.upper(), v.broker.upper())]
        return sorted(self.cfg.venues, key=lambda v: (v.priority, v.name))

    def _venue_cfg(self, name: str) -> Optional[VenueConfig]:
        for v in self.cfg.venues:
            if v.name.upper() == name.upper():
                return v
        return None

    def _venue_volume_hint(self, parent: Dict[str, Any], vcfg: VenueConfig) -> float:
        sym = _upper(parent.get("symbol"))
        # Try adapter first
        if liquidity_adapter and hasattr(liquidity_adapter, "liquidity_hint"):
            try:
                hint = liquidity_adapter.liquidity_hint(sym, vcfg.name)
                if hint: return float(hint)
            except Exception:
                pass
        # Fallback: rough function of qty
        return max(vcfg.min_child_qty, 0.2 * float(parent.get("qty") or 0))

    def _cap_by_participation(self, parent: Dict[str, Any], vcfg: VenueConfig, proposed_qty: float) -> float:
        # If we had ADV per-venue we’d use it; approximate with parent qty for now
        max_part = float(vcfg.max_participation or 0.1)
        adv_hint = float(parent.get("adv_hint") or (10 * float(parent.get("qty") or 0)))
        cap = max_part * adv_hint
        return min(proposed_qty, cap)

    def _score_venue(self, sym: str, side: str, vcfg: VenueConfig, parent: Dict[str, Any], slc: Dict[str, Any]) -> Optional[float]:
        # liquidity score (more is better)
        liq = 0.0
        if liquidity_adapter and hasattr(liquidity_adapter, "liquidity_hint"):
            try:
                liq = float(liquidity_adapter.liquidity_hint(sym, vcfg.name) or 0.0)
            except Exception:
                liq = 0.0

        # latency penalty
        lat_ms = 10.0
        if latency_adapter and hasattr(latency_adapter, "get_latency_ms"):
            try:
                lat_ms = float(latency_adapter.get_latency_ms(vcfg.name) or 10.0)
            except Exception:
                lat_ms = 10.0
        lat_pen = min(10.0, lat_ms / 10.0)  # normalize ~[0..10]

        # cost (bps) → penalty (higher cost => lower score)
        cost_pen = 0.0
        if self.cm:
            try:
                est = self.cm.estimate(
                    side=side, qty=slc["qty"], price=float(parent.get("mark_price") or parent.get("limit_price") or 0) or 1.0,
                    symbol=sym, venue=vcfg.name, instrument_type=parent.get("instrument_type","equity"),
                    notional_ccy=parent.get("currency","USD"), broker=vcfg.broker, spread=parent.get("spread"),
                    adv=parent.get("adv_hint"), vol=parent.get("vol_hint"),
                )
                cost_pen = max(0.0, est.total_bps)
            except Exception:
                cost_pen = vcfg.taker_fee_bps

        # maker rebate if POST_ONLY
        tif = (parent.get("tif") or vcfg.tif_default or "IOC").upper()
        maker_bonus = vcfg.maker_rebate_bps if tif == "POST_ONLY" else 0.0

        # final score
        score = (liq / 1e6) - (lat_pen * 0.1) - (cost_pen * 0.01) + (maker_bonus * 0.01)
        return score

    # -------- sending ----------
    @throttle.limit("router.send", calls=10, per=1)  # 10 child sends per second globally
    def _send(self, child: Dict[str, Any]) -> bool:
        broker = child.get("broker") or "paper"
        # additional throttling per broker
        throttle.wait(f"router.{broker}", calls=5, per=1.0)

        ok = False
        try:
            if broker_interface and hasattr(broker_interface, "place_order"):
                resp = broker_interface.place_order(broker=broker, order=child)
                ok = bool(resp and resp.get("accepted", True))
                if order_store and hasattr(order_store, "attach_broker_id") and resp and resp.get("order_id"):
                    order_store.attach_broker_id(child["id"], resp["order_id"])
        except Exception as e:
            self._emit("error", {"child": child, "error": str(e)})
            ok = False

        if ok:
            self._emit("sent", {"child": child})
            if order_store and hasattr(order_store, "update_order_status"):
                try:
                    order_store.update_order_status(child["id"], "open")
                except Exception:
                    pass
        else:
            if order_store and hasattr(order_store, "update_order_status"):
                try:
                    order_store.update_order_status(child["id"], "rejected")
                except Exception:
                    pass
        return ok

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        if not publish_stream:
            return
        try:
            publish_stream("oms.router", {"ts_ms": _utc_ms(), "kind": kind, **payload})
        except Exception:
            pass


# -------------- CLI probe ----------------
def _probe():
    r = Router()
    parent = {
        "id": _uid("parent"),
        "symbol": "RELIANCE.NS",
        "side": "buy",
        "qty": 2500,
        "typ": "limit",
        "limit_price": 2500.0,
        "strategy": "momo_in",
        "instrument_type": "equity",
        "currency": "INR",
        "adv_hint": 2_000_000,
        "vol_hint": 0.22,
        "spread": 0.05,
        "tif": "IOC",
        "algo": "POV",
    }
    ids = r.route(parent)
    print("Dispatched children:", ids)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SOR probe")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        _probe()