# backend/execution/router.py
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal, Any, Tuple, Callable

try:
    # Prefer your concrete interface if present
    from backend.brokers.broker_interface import BrokerInterface, Order, Fill # type: ignore
except Exception:
    # Minimal shims so this file stays importable
    from dataclasses import dataclass
    Side = Literal["buy","sell"]
    @dataclass
    class Order:
        id: str
        symbol: str
        side: Side # type: ignore
        qty: float
        order_type: str = "market"
        limit_price: Optional[float] = None
        tif: str = "DAY"
        extra: Dict[str, Any] = None # type: ignore
    class BrokerInterface:  # type: ignore
        def place_order(self, order: Order) -> str: ...
        def cancel_order(self, order_id: str) -> bool: ...
        def get_order_status(self, order_id: str) -> Dict[str, Any]: ...
        def get_positions(self) -> Dict[str, Any]: ...
        def get_account(self) -> Dict[str, Any]: ...

Side = Literal["buy","sell"]

# ---------------------------------------------------------------------
# Venue & scoring models
# ---------------------------------------------------------------------

@dataclass
class VenueInfo:
    id: str
    broker_key: str                 # which broker to send to
    latency_ms: float = 1.0
    fee_bps: float = 0.0            # taker fees (bps of notional)
    liquidity: float = 1.0          # 0..1 (relative book depth/ADV)
    toxicity: float = 0.2           # 0..1 (higher = worse)
    dark_pool: bool = False
    supports_ioc: bool = True
    supports_fok: bool = True
    supports_market: bool = True
    supports_limit: bool = True
    enabled: bool = True
    last_ok_ts: float = 0.0         # health heartbeat
    error_streak: int = 0

@dataclass
class RouteIntent:
    symbol: str
    side: Side # type: ignore
    qty: float
    order_type: str = "market"        # "market" | "limit" | "ioc" | "fok"
    limit_price: Optional[float] = None
    urgency: float = 0.5              # 0..1
    allow_dark: bool = True
    min_child_qty: float = 1.0
    max_splits: int = 3               # max venues to split across

@dataclass
class RouteDecision:
    chosen: List[Tuple[VenueInfo, float]]  # [(venue, child_qty), ...]
    expected_cost_bps: float
    score_breakdown: Dict[str, float]
    notes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouterConfig:
    liquidity_weight: float = 0.5
    latency_weight: float = 0.2
    fee_weight: float = 0.1
    toxicity_weight: float = 0.2
    prefer_dark_when_large: bool = True
    large_vs_adv: float = 0.05            # 5% ADV threshold for dark bias
    health_backoff_sec: float = 10.0
    max_error_streak: int = 3
    throttle_per_sec: int = 40            # per venue
    burst: int = 20

# ---------------------------------------------------------------------
# Simple token-bucket per venue for throttle
# ---------------------------------------------------------------------

class _Bucket:
    def __init__(self, rate_per_sec: int, burst: int):
        self.rate = max(1, rate_per_sec)
        self.capacity = max(1, burst)
        self.tokens = self.capacity
        self.ts = time.time()
    def allow(self) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
        self.ts = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------

class Router:
    """
    Smart Order Router:
      • Scores venues with configurable weights
      • Optional dark-pool bias for large orders
      • Health & throttle per venue
      • Split routing (top-K) with proportional allocation
      • Failover on error; minimal retries
    """

    def __init__(self, cfg: RouterConfig | None = None):
        self.cfg = cfg or RouterConfig()
        self._venues: Dict[str, VenueInfo] = {}
        self._brokers: Dict[str, BrokerInterface] = {}
        self._buckets: Dict[str, _Bucket] = {}
        self._adv_lookup: Callable[[str], float] = lambda sym: 1_000_000.0  # override with real ADV

    # --- registry ----------------------------------------------------

    def register_broker(self, key: str, broker: BrokerInterface) -> None:
        self._brokers[key] = broker

    def upsert_venue(self, v: VenueInfo) -> None:
        self._venues[v.id] = v
        if v.id not in self._buckets:
            self._buckets[v.id] = _Bucket(self.cfg.throttle_per_sec, self.cfg.burst)

    def set_adv_provider(self, fn: Callable[[str], float]) -> None:
        """fn(symbol) -> ADV float"""
        self._adv_lookup = fn

    def mark_venue_ok(self, venue_id: str) -> None:
        if venue_id in self._venues:
            vi = self._venues[venue_id]
            vi.last_ok_ts = time.time()
            vi.error_streak = 0

    def mark_venue_error(self, venue_id: str) -> None:
        if venue_id in self._venues:
            vi = self._venues[venue_id]
            vi.error_streak += 1
            if vi.error_streak >= self.cfg.max_error_streak:
                vi.enabled = False

    # --- scoring -----------------------------------------------------

    def _score(self, v: VenueInfo, intent: RouteIntent, *, is_large: bool) -> Tuple[float, Dict[str, float]]:
        if not v.enabled:
            return -1e9, {"disabled": 1.0}

        # health gate
        if v.error_streak >= self.cfg.max_error_streak:
            return -1e9, {"sick": 1.0}
        if (time.time() - v.last_ok_ts) > max(1.0, 5.0 * self.cfg.health_backoff_sec) and v.last_ok_ts > 0:
            # stale health – soft penalty
            pass

        # feature transforms (higher is better)
        liq = max(0.0, min(1.0, v.liquidity))
        lat = 1.0 / (1.0 + max(0.0, v.latency_ms))          # lower latency → higher score
        fee = 1.0 / (1.0 + max(0.0, v.fee_bps))             # lower fees → higher
        tox = 1.0 - max(0.0, min(1.0, v.toxicity))          # less toxic → higher

        # dark bias for large orders
        dark_bonus = 0.0
        if intent.allow_dark and self.cfg.prefer_dark_when_large and is_large and v.dark_pool:
            dark_bonus = 0.10  # small additive bump

        score = (
            self.cfg.liquidity_weight * liq +
            self.cfg.latency_weight  * lat +
            self.cfg.fee_weight      * fee +
            self.cfg.toxicity_weight * tox +
            dark_bonus
        )
        return score, {"liq": liq, "lat": lat, "fee": fee, "tox": tox, "dark_bonus": dark_bonus}

    # --- entry points ------------------------------------------------

    def decide(self, intent: RouteIntent) -> RouteDecision:
        """Choose one or more venues and child sizes (no side effects)."""
        adv = max(1.0, self._adv_lookup(intent.symbol))
        is_large = (intent.qty / adv) >= self.cfg.large_vs_adv

        # filter by feature support
        cand: List[VenueInfo] = []
        for v in self._venues.values():
            if not v.enabled:
                continue
            if intent.order_type in ("market",) and not v.supports_market:  continue
            if intent.order_type in ("limit",)  and not v.supports_limit:   continue
            if intent.order_type == "ioc" and not v.supports_ioc:           continue
            if intent.order_type == "fok" and not v.supports_fok:           continue
            if v.dark_pool and not intent.allow_dark:                       continue
            cand.append(v)

        if not cand:
            return RouteDecision(chosen=[], expected_cost_bps=1e9, score_breakdown={}, notes={"reason": "no_venues"})

        scored: List[Tuple[VenueInfo, float, Dict[str,float]]] = []
        for v in cand:
            s, bd = self._score(v, intent, is_large=is_large)
            scored.append((v, s, bd))
        scored.sort(key=lambda x: x[1], reverse=True)

        k = max(1, min(intent.max_splits, len(scored)))
        top = scored[:k]

        # proportional allocation by (normalized) score and venue liquidity
        total_score = sum(max(1e-9, s if s > 0 else 0.0) for _, s, _ in top)
        # mix with liquidity weight for robustness
        chosen: List[Tuple[VenueInfo, float]] = []
        qty_left = intent.qty
        breakdown = {}
        for v, s, bd in top:
            w = (max(0.0, s) / total_score) if total_score > 0 else (1.0 / k)
            w = 0.5 * w + 0.5 * max(0.0, min(1.0, v.liquidity)) / sum(max(1e-9, t[0].liquidity) for t in top)
            q_child = max(0.0, qty_left * w)
            if q_child >= intent.min_child_qty:
                chosen.append((v, q_child))
                breakdown[v.id] = bd
        # ensure we don’t drop dust; allocate any rounding residue to best venue
        residue = intent.qty - sum(q for _, q in chosen)
        if residue >= intent.min_child_qty and chosen:
            chosen[0] = (chosen[0][0], chosen[0][1] + residue)

        # rough expected cost in bps (fee + 0.5*spread proxy via liquidity)
        exp_bps = 0.0
        for v, q in chosen:
            # lower liquidity → higher slippage proxy
            liq_pen = (1.0 / max(1e-6, v.liquidity)) - 1.0
            bps = v.fee_bps + 5.0 * liq_pen + 0.1 * v.latency_ms + 10.0 * max(0.0, v.toxicity - 0.2)
            exp_bps += (q / max(1e-9, intent.qty)) * bps

        return RouteDecision(chosen=chosen, expected_cost_bps=exp_bps, score_breakdown={"venues": breakdown}, # type: ignore
                             notes={"is_large": is_large, "adv": adv})

    def route(self, intent: RouteIntent) -> Dict[str, Any]:
        """
        Execute according to decision:
          - per-venue throttle
          - place orders via mapped broker
          - minimal failover on error
        Returns a summary with child order ids and any failures.
        """
        decision = self.decide(intent)
        out = {
            "decision": {
                "expected_cost_bps": decision.expected_cost_bps,
                "chosen": [(v.id, q) for v, q in decision.chosen],
                "notes": decision.notes
            },
            "children": [],   # [{venue, order_id, qty, status}, ...]
            "failures": []
        }
        if not decision.chosen:
            out["failures"].append({"reason": "no_venues"})
            return out

        for v, q_child in decision.chosen:
            # throttle
            bucket = self._buckets.get(v.id)
            if bucket and not bucket.allow():
                out["failures"].append({"venue": v.id, "reason": "throttled"})
                continue
            # broker
            br = self._brokers.get(v.broker_key)
            if br is None:
                out["failures"].append({"venue": v.id, "reason": f"broker_missing:{v.broker_key}"})
                continue

            # craft child order
            typ = intent.order_type
            tif = "IOC" if typ in ("ioc", "fok") else "DAY"
            o = Order(
                id="",
                symbol=intent.symbol,
                side=intent.side,
                qty=q_child,
                order_type=("market" if typ in ("ioc","fok","market") else "limit"),
                limit_price=intent.limit_price,
                tif=tif,
                extra={"venue": v.id, "route": "SOR", "urgency": intent.urgency}
            )

            try:
                oid = br.place_order(o)
                self.mark_venue_ok(v.id)
                out["children"].append({"venue": v.id, "order_id": oid, "qty": q_child, "status": "sent"})
            except Exception as e:
                self.mark_venue_error(v.id)
                out["failures"].append({"venue": v.id, "reason": f"place_error:{e}"})
                # simple failover: push this qty to the best remaining enabled venue once
                failover = self._best_failover(v_exclude=v.id, intent=intent, qty=q_child)
                if failover:
                    v2, q2 = failover
                    try:
                        oid2 = self._brokers[v2.broker_key].place_order(Order(
                            id="", symbol=intent.symbol, side=intent.side, qty=q2,
                            order_type=o.order_type, limit_price=o.limit_price, tif=o.tif,
                            extra={"venue": v2.id, "route": "SOR-failover", "urgency": intent.urgency}
                        ))
                        self.mark_venue_ok(v2.id)
                        out["children"].append({"venue": v2.id, "order_id": oid2, "qty": q2, "status": "sent"})
                    except Exception as e2:
                        self.mark_venue_error(v2.id)
                        out["failures"].append({"venue": v2.id, "reason": f"failover_error:{e2}"})
        return out

    # --- helpers -----------------------------------------------------

    def _best_failover(self, v_exclude: str, intent: RouteIntent, qty: float) -> Optional[Tuple[VenueInfo, float]]:
        # Re-score without the excluded venue; pick the next best single venue
        tmp = RouteIntent(**{**asdict(intent), "qty": qty})
        cand = [v for v in self._venues.values() if v.id != v_exclude and v.enabled]
        if not cand:
            return None
        # quick pick by same scoring
        adv = max(1.0, self._adv_lookup(intent.symbol))
        is_large = (qty / adv) >= self.cfg.large_vs_adv
        best = None
        best_s = -1e9
        for v in cand:
            s, _ = self._score(v, tmp, is_large=is_large)
            if s > best_s:
                best_s = s
                best = v
        return (best, qty) if best else None