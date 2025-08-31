# backend/execution/routing_rules.py
from __future__ import annotations

import fnmatch
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# ---- tiny YAML loader (safe fallback) ----------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def now_ms() -> int: return int(time.time() * 1000)

# ================= Models =================
@dataclass
class RouteContext:
    symbol: str
    side: str                      # "buy" | "sell"
    qty: float
    notional: Optional[float] = None
    asset: Optional[str] = None    # "equity"|"futures"|"options"|"fx"|"crypto"
    region: Optional[str] = None   # "US"|"IN"|"EU"|"SG"...
    session: str = "REG"           # "PRE"|"REG"|"POST"
    spread_bps: Optional[float] = None
    toxicity: Dict[str, float] = field(default_factory=dict)  # venue->0..1
    fees_bps: Dict[str, float] = field(default_factory=dict)  # venue->bps
    lit_venues: List[str] = field(default_factory=list)
    dark_venues: List[str] = field(default_factory=list)
    time_ms: int = field(default_factory=now_ms)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RouteDecision:
    ok: bool
    broker: Optional[str] = None
    account: Optional[str] = None
    venues: List[Dict[str, Any]] = field(default_factory=list)   # [{"venue":"NYSE","weight":0.6},...]
    algo_hint: Optional[str] = None                              # "VWAP"|"POV"|...
    tif: Optional[str] = None                                    # "DAY"|"IOC"|...
    notes: List[str] = field(default_factory=list)
    rule_id: Optional[str] = None
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

@dataclass
class Rule:
    id: str
    priority: int = 100
    mode: str = "first"              # "first"|"best"
    when: Dict[str, Any] = field(default_factory=dict)
    route: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0                # weight in "best" mode

# ================= Engine =================
class RoutingRules:
    """
    Lightweight, config-driven router.
    Load with: rr = RoutingRules("./policy.yaml")
    Decide with: dec = rr.decide(ctx)
    """

    def __init__(self, policy_path: Optional[str] = None, *, section: str = "routing_rules"):
        self.rules: List[Rule] = []
        self.mode_global = "first"
        if policy_path:
            self.load(policy_path, section=section)

    # ---------- Public ----------
    def load(self, policy_path: str, *, section: str = "routing_rules") -> None:
        data = _load_yaml(policy_path)
        cfg = data.get(section, {})
        if isinstance(cfg, dict):
            self.mode_global = str(cfg.get("mode", "first")).lower()
            raw_rules = cfg.get("rules", [])
        elif isinstance(cfg, list):
            raw_rules = cfg
        else:
            raw_rules = []

        rules: List[Rule] = []
        for i, r in enumerate(raw_rules):
            rid = str(r.get("id") or f"rule_{i+1}")
            rules.append(Rule(
                id=rid,
                priority=int(r.get("priority", 100)),
                mode=str(r.get("mode", self.mode_global)).lower(),
                when=r.get("when", {}) or {},
                route=r.get("route", {}) or {},
                score=float(r.get("score", 1.0)),
            ))
        self.rules = sorted(rules, key=lambda x: (x.priority, x.id))

    def decide(self, ctx: RouteContext) -> RouteDecision:
        matches: List[Tuple[Rule, float, List[str]]] = []
        for rule in self.rules:
            ok, score, notes = self._match(rule.when, ctx)
            if ok:
                matches.append((rule, score * rule.score, notes))

        if not matches:
            return RouteDecision(ok=False, notes=["no_rule_matched"], meta={"ctx": _preview_ctx(ctx)})

        # mode selection
        if any(r.mode == "first" for r, _, _ in matches) and self.mode_global == "first":
            rule, sc, notes = matches[0]
        else:
            # best score (ties → lowest priority)
            rule, sc, notes = sorted(matches, key=lambda t: (-t[1], t[0].priority))[0]

        decision = self._build_decision(rule, ctx, notes, sc)
        return decision

    # ---------- Matchers ----------
    def _match(self, when: Dict[str, Any], ctx: RouteContext) -> Tuple[bool, float, List[str]]:
        score = 0.0
        notes: List[str] = []

        def bump(s: float, note: str):
            nonlocal score; score += s; notes.append(note)

        # symbol patterns / regex
        if "symbol_glob" in when:
            globs = _listify(when["symbol_glob"])
            if not any(fnmatch.fnmatch(ctx.symbol, g) for g in globs): return (False, 0.0, notes)
            bump(1.0, f"symbol_glob={globs}")
        if "symbol_regex" in when:
            rx = re.compile(str(when["symbol_regex"]))
            if not rx.search(ctx.symbol): return (False, 0.0, notes)
            bump(1.0, f"symbol_regex={when['symbol_regex']}")

        # asset / region / session
        if not _opt_eq(when.get("asset"), ctx.asset): return (False, 0.0, notes)
        if not _opt_eq(when.get("region"), ctx.region): return (False, 0.0, notes)
        if not _opt_in(when.get("session"), ctx.session): return (False, 0.0, notes)
        if when.get("side") and str(ctx.side).lower() != str(when["side"]).lower():
            return (False, 0.0, notes)

        # qty / notional ranges
        if not _range_ok(when.get("min_qty"), when.get("max_qty"), ctx.qty): return (False, 0.0, notes)
        if ctx.notional is not None and not _range_ok(when.get("min_notional"), when.get("max_notional"), ctx.notional):
            return (False, 0.0, notes)

        # market microstructure gates
        if ctx.spread_bps is not None:
            if not _range_ok(when.get("min_spread_bps"), when.get("max_spread_bps"), ctx.spread_bps):
                return (False, 0.0, notes)

        # venue toxicity / fees thresholds
        tox_max = when.get("max_venue_toxicity")
        if tox_max is not None:
            # require at least one venue <= tox_max
            if ctx.toxicity and not any(v <= float(tox_max) for v in ctx.toxicity.values()):
                return (False, 0.0, notes)

        fee_max = when.get("max_venue_fee_bps")
        if fee_max is not None:
            if ctx.fees_bps and not any(f <= float(fee_max) for f in ctx.fees_bps.values()):
                return (False, 0.0, notes)

        # time windows (24h clock, local process time)
        if "time_window" in when:
            hh = _now_hhmm()
            for w in _listify(when["time_window"]):
                if _hhmm_in_window(hh, str(w)):
                    bump(0.5, f"time_window={w}")
                    break
            else:
                return (False, 0.0, notes)

        # venue allow/deny
        if "allow_venues" in when:
            allow = set(_listify(when["allow_venues"]))
            if ctx.lit_venues or ctx.dark_venues:
                have = set(ctx.lit_venues + ctx.dark_venues)
                if not allow.intersection(have):
                    return (False, 0.0, notes)
            bump(0.5, f"allow_venues∩have")
        if "deny_venues" in when:
            deny = set(_listify(when["deny_venues"]))
            if any(v in deny for v in (ctx.lit_venues + ctx.dark_venues)):
                return (False, 0.0, notes)

        # simple boosts
        if when.get("boost_if_dark") and ctx.dark_venues:
            bump(0.2, "boost_dark")
        if when.get("boost_if_lit") and ctx.lit_venues:
            bump(0.2, "boost_lit")

        return (True, score, notes)

    # ---------- Decision builder ----------
    def _build_decision(self, rule: Rule, ctx: RouteContext, notes: List[str], score: float) -> RouteDecision:
        r = rule.route
        dec = RouteDecision(
            ok=True,
            broker=r.get("broker"),
            account=r.get("account"),
            algo_hint=r.get("algo"),
            tif=r.get("tif"),
            notes=notes + [f"match={rule.id}", f"score={round(score,3)}"],
            rule_id=rule.id,
            score=score,
            meta={"symbol": ctx.symbol, "side": ctx.side, "qty": ctx.qty, "session": ctx.session},
        )

        # venue weights (lit/dark / explicit)
        venues: List[Dict[str, Any]] = []
        if "venues" in r:
            for v in _listify(r["venues"]):
                if isinstance(v, str):
                    venues.append({"venue": v, "weight": 0.0})
                elif isinstance(v, dict) and "venue" in v:
                    venues.append({"venue": v["venue"], "weight": float(v.get("weight", 0.0))})
        else:
            # derive simple equal weights from context (filter by deny list if any)
            ven = list(ctx.lit_venues) + list(ctx.dark_venues)
            deny = set(_listify(rule.when.get("deny_venues", [])))
            ven = [v for v in ven if v not in deny]
            if ven:
                w = 1.0 / len(ven)
                venues = [{"venue": v, "weight": w} for v in ven]

        # normalize weights if any > 0
        s = sum(v.get("weight", 0.0) for v in venues)
        if s > 0:
            for v in venues:
                v["weight"] = v.get("weight", 0.0) / s
        dec.venues = venues
        return dec

# ================= Helpers =================
def _preview_ctx(ctx: RouteContext) -> Dict[str, Any]:
    return {
        "symbol": ctx.symbol, "side": ctx.side, "qty": ctx.qty, "asset": ctx.asset,
        "region": ctx.region, "session": ctx.session, "lit": ctx.lit_venues, "dark": ctx.dark_venues
    }

def _listify(x: Any) -> List[Any]:
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _opt_eq(cond: Optional[str | List[str]], val: Optional[str]) -> bool:
    if cond is None: return True
    if isinstance(cond, list):
        return (val in cond)
    return str(val).lower() == str(cond).lower()

def _opt_in(cond: Optional[str | List[str]], val: Optional[str]) -> bool:
    if cond is None: return True
    if isinstance(cond, list):
        return str(val).upper() in {str(c).upper() for c in cond}
    return str(val).upper() == str(cond).upper()

def _range_ok(min_v: Any, max_v: Any, val: Optional[float]) -> bool:
    if val is None: return True
    if min_v is not None and float(val) < float(min_v): return False
    if max_v is not None and float(val) > float(max_v): return False
    return True

def _now_hhmm() -> int:
    t = time.localtime()
    return t.tm_hour * 100 + t.tm_min

def _hhmm_in_window(hhmm: int, window: str) -> bool:
    """
    window examples: "0930-1600", "1900-0459" (wrap past midnight)
    """
    try:
        a, b = window.split("-", 1)
        a, b = int(a), int(b)
        if a <= b:
            return a <= hhmm <= b
        # wrap
        return hhmm >= a or hhmm <= b
    except Exception:
        return True

# ================= Smoke test =================
if __name__ == "__main__":  # pragma: no cover
    # Example inline rules (works even without policy.yaml)
    rr = RoutingRules()
    rr.rules = [
        Rule(
            id="IN_equities_day",
            priority=10,
            when={"asset":"equity","region":"IN","session":["REG"],"symbol_glob":["NSE:*","BSE:*"],"max_spread_bps":20},
            route={"broker":"zerodha","account":"main","algo":"VWAP","tif":"DAY","venues":["NSE","BSE"]}
        ),
        Rule(
            id="US_large_notional_dark_bias",
            priority=20,
            when={"asset":"equity","region":"US","min_notional":250_000,"boost_if_dark":True},
            route={"broker":"ibkr","account":"pm1","algo":"POV","tif":"DAY"}
        ),
    ]

    ctx = RouteContext(
        symbol="NSE:RELIANCE", side="buy", qty=5000, notional=1_000_000,
        asset="equity", region="IN", session="REG", spread_bps=8.0,
        lit_venues=["NSE","BSE"], dark_venues=[]
    )
    dec = rr.decide(ctx)
    from pprint import pprint
    pprint(dec.to_dict())