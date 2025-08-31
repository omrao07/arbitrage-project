# backend/execution/execution_policy.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# --- Optional helpers (safe fallbacks) ---------------------------------
try:
    from backend.ai.agents.core.toolbelt import load_yaml, now_ms, percentile, ewma
except Exception:
    import time as _t
    def load_yaml(path: str) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    def now_ms() -> int: return int(_t.time()*1000)
    def percentile(values, q): 
        if not values: return float("nan")
        v=sorted(values); k=(len(v)-1)*(q/100.0); f=int(k); c=min(f+1,len(v)-1)
        return v[f] if f==c else v[f]+(v[c]-v[f])*(k-f)
    def ewma(prev, x, alpha): return (1-alpha)*prev + alpha*x

# -----------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------
@dataclass
class OrderIntent:
    symbol: str
    side: str                     # "buy" | "sell"
    qty: float
    urgency: str = "normal"       # "low" | "normal" | "high"
    participation_cap: Optional[float] = None   # 0.0-1.0 (for POV)
    limit_price: Optional[float] = None
    tif: Optional[str] = None     # "IOC" | "FOK" | "DAY" | "GTD"
    venue_hint: Optional[str] = None
    strategy: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketContext:
    last: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    vwap_5m: Optional[float] = None
    vol_5m: Optional[float] = None              # stdev in price terms (not %)
    adv_20d: Optional[float] = None             # average daily volume (shares)
    est_intraday_vol: Optional[float] = None    # % sigma for day
    lit_venues: List[str] = field(default_factory=list)
    dark_venues: List[str] = field(default_factory=list)
    time_ms: int = field(default_factory=now_ms)
    session: str = "REG"                        # "PRE"|"REG"|"POST"
    toxicity: Dict[str, float] = field(default_factory=dict)  # venue->score [0..1]
    fees_bps: Dict[str, float] = field(default_factory=dict)  # venue->bps

@dataclass
class SlicePlan:
    child_qty: float
    interval_ms: int
    max_spread_bps: float
    price_limit: Optional[float] = None
    post_only: bool = False

@dataclass
class VenueWeight:
    venue: str
    weight: float        # 0..1
    dark: bool = False

@dataclass
class PolicyPlan:
    algo: str                          # "VWAP" | "TWAP" | "POV" | "AdaptiveVWAP" | "IOC" | "FOK"
    tif: str                           # "DAY" | "IOC" | "FOK" | "GTD"
    limit_price: Optional[float]
    max_slippage_bps: float
    max_participation: Optional[float]
    slice: SlicePlan
    venues: List[VenueWeight]
    notes: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------
# ExecutionPolicy
# -----------------------------------------------------------------------
class ExecutionPolicy:
    """
    Rule-based planner that maps OrderIntent + MarketContext -> PolicyPlan.
    Loads thresholds from policy.yaml if present, otherwise uses sane defaults.
    """

    def __init__(self, config_path: str | None = None):
        cfg = load_yaml(config_path) if config_path else {}
        self.cfg = {
            "spread_bps_twap": 5.0,
            "spread_bps_vwap": 10.0,
            "spread_bps_pov": 12.0,
            "slippage_bps_cap": 25.0,
            "pov_default": 0.1,              # 10% of market volume
            "pov_high_urgency": 0.25,
            "slice_seconds_low": 8,
            "slice_seconds_normal": 4,
            "slice_seconds_high": 1,
            "dark_cap": 0.5,                 # at most 50% to dark pools unless high urgency
            "toxicity_cutoff": 0.7,          # avoid venues > 0.7 toxicity unless no alternative
            "fees_bias_bps": 0.5,            # prefer cheaper venues
            "prepost_aggr": "IOC",           # off-session behavior
        } | (cfg.get("execution_policy", {}) if isinstance(cfg.get("execution_policy", {}), dict) else {})

    # -------- Public API --------
    def evaluate(self, oi: OrderIntent, mc: MarketContext) -> PolicyPlan:
        side_mult = 1.0 if oi.side.lower() == "buy" else -1.0

        # 1) Determine baseline algo from urgency + session + constraints
        algo, tif = self._choose_algo_tif(oi, mc)

        # 2) Limit price and slippage caps
        limit_px, max_slip = self._limit_and_slippage(oi, mc, side_mult, algo)

        # 3) Slice cadence
        slice_plan = self._slice_plan(oi, mc, algo, limit_px)

        # 4) Venue mix (lit/dark weighting with toxicity & fees)
        venues = self._venue_mix(oi, mc, urgency=oi.urgency)

        # 5) POV participation (if applicable)
        pov = None
        if algo.upper() == "POV":
            pov = min(max(oi.participation_cap or self._pov_default(oi), 0.01), 0.9)

        notes = self._make_notes(oi, mc, algo, limit_px, pov)

        return PolicyPlan(
            algo=algo,
            tif=tif,
            limit_price=limit_px,
            max_slippage_bps=max_slip,
            max_participation=pov,
            slice=slice_plan,
            venues=venues,
            notes=notes,
            meta={
                "symbol": oi.symbol,
                "side": oi.side,
                "qty": oi.qty,
                "session": mc.session,
                "urgency": oi.urgency,
            },
        )

    # -------- Internals --------
    def _choose_algo_tif(self, oi: OrderIntent, mc: MarketContext) -> Tuple[str, str]:
        # Off-session or very wide spreads -> more aggressive IOC/FOK
        spread_bps = self._spread_bps(mc)
        if mc.session != "REG":
            algo = self.cfg["prepost_aggr"]
            tif = "IOC" if algo != "FOK" else "FOK"
            return algo, tif

        if oi.tif in ("IOC", "FOK"):
            return oi.tif, oi.tif

        # Urgency heuristics
        if oi.urgency == "high":
            # If tight spread -> AdaptiveVWAP; else POV to chase liquidity
            if spread_bps is not None and spread_bps <= self.cfg["spread_bps_vwap"]:
                return "AdaptiveVWAP", "DAY"
            return "POV", "DAY"

        if oi.urgency == "low":
            # Gentle TWAP if spreads tight; otherwise VWAP
            if spread_bps is not None and spread_bps <= self.cfg["spread_bps_twap"]:
                return "TWAP", "DAY"
            return "VWAP", "DAY"

        # normal
        if spread_bps is not None and spread_bps <= self.cfg["spread_bps_vwap"]:
            return "VWAP", "DAY"
        return "POV", "DAY"

    def _limit_and_slippage(self, oi: OrderIntent, mc: MarketContext, side_mult: float, algo: str) -> Tuple[Optional[float], float]:
        # baseline slippage cap
        max_slip = float(self.cfg["slippage_bps_cap"])
        # limit price: if user provided → respect; else anchor around last/quote
        if oi.limit_price is not None:
            return float(oi.limit_price), max_slip

        px = mc.last
        # If we have quotes, bias limit toward best for post-only/passive where possible
        if algo.upper() in ("VWAP","TWAP","ADAPTIVEVWAP"):
            if oi.side.lower() == "buy" and mc.bid:  # try to sit near bid
                return float(mc.bid), max_slip
            if oi.side.lower() == "sell" and mc.ask:
                return float(mc.ask), max_slip

        # otherwise no explicit cap (marketable), but keep slippage guard
        return None, max_slip

    def _slice_plan(self, oi: OrderIntent, mc: MarketContext, algo: str, limit_price: Optional[float]) -> SlicePlan:
        # cadence by urgency
        sec = {
            "low": self.cfg["slice_seconds_low"],
            "normal": self.cfg["slice_seconds_normal"],
            "high": self.cfg["slice_seconds_high"],
        }.get(oi.urgency, self.cfg["slice_seconds_normal"])

        # default max spread gate depends on algo
        spread_gate = {
            "VWAP": self.cfg["spread_bps_vwap"],
            "ADAPTIVEVWAP": self.cfg["spread_bps_vwap"],
            "TWAP": self.cfg["spread_bps_twap"],
            "POV": self.cfg["spread_bps_pov"],
            "IOC": 999.0,
            "FOK": 999.0,
        }.get(algo.upper(), self.cfg["spread_bps_vwap"])

        # child size: simple rule (could use ADV and vol in future)
        base_child = max(1.0, oi.qty * (0.05 if oi.urgency == "high" else 0.02 if oi.urgency == "normal" else 0.01))
        return SlicePlan(
            child_qty=min(base_child, oi.qty),
            interval_ms=int(sec * 1000),
            max_spread_bps=float(spread_gate),
            price_limit=limit_price,
            post_only=(algo.upper() in ("VWAP","TWAP","ADAPTIVEVWAP"))
        )

    def _venue_mix(self, oi: OrderIntent, mc: MarketContext, *, urgency: str) -> List[VenueWeight]:
        venues: List[VenueWeight] = []
        lit = list(mc.lit_venues or [])
        dark = list(mc.dark_venues or [])

        # Start with lit venues equally weighted, then bias by fees and toxicity
        if lit:
            base_w = 1.0 / len(lit)
            for v in lit:
                tox = mc.toxicity.get(v, 0.0)
                fee = mc.fees_bps.get(v, 0.0)
                w = base_w
                # Penalize high toxicity
                if tox > self.cfg["toxicity_cutoff"]:
                    w *= 0.5
                # Prefer cheaper venues (reduce weight by fee bias)
                w *= max(0.2, 1.0 - (fee / max(1e-6, self.cfg["fees_bias_bps"])))
                venues.append(VenueWeight(venue=v, weight=w, dark=False))

        # Normalize
        s = sum(v.weight for v in venues) or 1.0
        for v in venues:
            v.weight /= s

        # Mix in dark if available and allowed
        dark_cap = (0.8 if urgency == "high" else self.cfg["dark_cap"])
        if dark and dark_cap > 0:
            add = min(dark_cap, 1.0)
            # scale down lit weights
            for v in venues:
                v.weight *= (1.0 - add)
            # spread dark evenly
            each = add / len(dark)
            for dv in dark:
                venues.append(VenueWeight(venue=dv, weight=each, dark=True))

        # Venue hint override
        if oi.venue_hint:
            # bump hinted venue by +50% weight (renormalize)
            for v in venues:
                if v.venue == oi.venue_hint:
                    v.weight *= 1.5
            s = sum(v.weight for v in venues) or 1.0
            for v in venues:
                v.weight /= s

        return venues

    def _pov_default(self, oi: OrderIntent) -> float:
        return self.cfg["pov_high_urgency"] if oi.urgency == "high" else self.cfg["pov_default"]

    def _spread_bps(self, mc: MarketContext) -> Optional[float]:
        if mc.spread is not None and mc.last:
            return float(mc.spread / mc.last * 1e4)
        if mc.bid and mc.ask and mc.last:
            return float((mc.ask - mc.bid) / mc.last * 1e4)
        return None

    def _make_notes(self, oi: OrderIntent, mc: MarketContext, algo: str, limit_px: Optional[float], pov: Optional[float]) -> List[str]:
        notes = []
        sbps = self._spread_bps(mc)
        if sbps is not None:
            notes.append(f"spread={sbps:.1f} bps")
        notes.append(f"session={mc.session}")
        if limit_px is not None:
            notes.append(f"limit={limit_px}")
        if pov is not None:
            notes.append(f"POV cap={pov*100:.1f}%")
        if oi.tif:
            notes.append(f"TIF override={oi.tif}")
        if oi.venue_hint:
            notes.append(f"venue_hint={oi.venue_hint}")
        return notes

# -----------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    policy = ExecutionPolicy()  # or ExecutionPolicy("./policy.yaml")

    oi = OrderIntent(symbol="AAPL", side="buy", qty=5000, urgency="normal")
    mc = MarketContext(
        last=190.0, bid=189.98, ask=190.02,
        lit_venues=["NYSE","ARCA"], dark_venues=["DARK1"],
        toxicity={"NYSE":0.2,"ARCA":0.4,"DARK1":0.1},
        fees_bps={"NYSE":0.2,"ARCA":0.5,"DARK1":0.0},
        session="REG"
    )
    plan = policy.evaluate(oi, mc)
    from pprint import pprint
    pprint(asdict(plan))