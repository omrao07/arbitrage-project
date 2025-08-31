# backend/risk/policy_engine.py
from __future__ import annotations

import os, time, json, math, re, threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Callable, Tuple

# ----- optional deps (graceful) ----------------------------------------------
HAVE_YAML = True
try:
    import yaml  # type: ignore
except Exception:
    HAVE_YAML = False
    yaml = None  # type: ignore

HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

# ----- env / streams ---------------------------------------------------------
REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_AUDIT  = os.getenv("RISK_AUDIT_STREAM", "risk.audit")
HASH_LIMITS   = os.getenv("RISK_LIMITS_HASH", "risk:limits")      # used for dynamic caps
KEY_KILL      = os.getenv("RISK_KILLSWITCH_KEY", "risk:killswitch")
CFG_PATH      = os.getenv("POLICY_CFG", "configs/policy_engine.yaml")

# ----- data models -----------------------------------------------------------
@dataclass
class Order:
    strategy: str
    symbol: str
    side: str                 # buy/sell
    qty: float
    typ: str = "market"       # market/limit
    limit_price: Optional[float] = None
    venue: Optional[str] = None
    region: Optional[str] = None
    account: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Decision:
    action: str               # "allow" | "deny" | "modify"
    reason: str
    order: Order
    changes: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)  # detailed results
    ts_ms: int = field(default_factory=lambda: int(time.time()*1000))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["order"] = asdict(self.order)
        return d

# ----- tiny backend wrapper ---------------------------------------------------
class _Backend:
    def __init__(self, redis_url: Optional[str] = None):
        self.mem: Dict[str, Any] = {}
        self.redis = None
        if HAVE_REDIS:
            try:
                self.redis = Redis.from_url(redis_url or REDIS_URL, decode_responses=True)  # type: ignore
                self.redis.ping()
            except Exception:
                self.redis = None

    def get(self, key: str) -> Optional[str]:
        if self.redis:
            try: return self.redis.get(key)  # type: ignore
            except Exception: pass
        v = self.mem.get(key)
        return None if v is None else str(v)

    def hgetall(self, key: str) -> Dict[str, str]:
        if self.redis:
            try: return self.redis.hgetall(key) or {}  # type: ignore
            except Exception: pass
        return dict(self.mem.get(key) or {})

    def xadd(self, stream: str, obj: Dict[str, Any], maxlen: int = 50_000) -> None:
        if self.redis:
            try:
                self.redis.xadd(stream, {"json": json.dumps(obj)}, maxlen=maxlen, approximate=True)  # type: ignore
                return
            except Exception:
                pass
        # mem fallback: ignore

# ----- core: registry of checks ----------------------------------------------
CheckFn = Callable[['PolicyEngine', Order, Dict[str, Any]], Tuple[bool, Dict[str, Any]]]
# returns (passed, details) — if passed=False and details.get("fatal",True) → deny; else can be modify

def _sgn(side: str) -> int:
    s = (side or "").lower()
    return +1 if s in ("buy","b","long") else -1

def _num(x) -> float:
    try: return float(x)
    except Exception: return 0.0

class PolicyEngine:
    """
    Evaluate an order against configured checks.
    YAML structure (configs/policy_engine.yaml):

    defaults:
      price_band_bps: 300      # +/- 3% from reference
      max_order_notional: 2_000_000
      max_qty: 250000
      allowed_venues: ["XNSE","BSE","XNAS","BATS"]
      throttle:
        per_symbol_per_min: 50
        per_strategy_per_min: 200
    symbols:
      RELIANCE:
        price_band_bps: 150
        max_qty: 100000
    strategies:
      alpha_dip:
        max_order_notional: 500000
        allowed_venues: ["XNSE"]
    regions:
      IN:
        market_hours: "09:15-15:30"   # local exchange time
    hard_blocks:
      - { field: "side", regex: "short", reason: "shorting disabled in cash" }
      - { field: "symbol", regex: "^BANKNIFTY$", reason: "blocked instrument" }
    """

    def __init__(self, cfg_path: str = CFG_PATH, *, redis_url: Optional[str] = None):
        self.cfg_path = cfg_path
        self.cfg = self._load_cfg(cfg_path)
        self.b = _Backend(redis_url)
        self.checks: List[Tuple[str, CheckFn]] = []
        self._register_builtin_checks()

        # in-mem rolling counters for throttles (fallback if no Redis rate limiter)
        self._roll: Dict[str, List[int]] = {}

    # ----- config load -----
    def _load_cfg(self, path: str) -> Dict[str, Any]:
        if HAVE_YAML and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f) or {} # type: ignore
                return obj
        # minimal default
        return {
            "defaults": {
                "price_band_bps": 300,
                "max_order_notional": 2_000_000,
                "max_qty": 250_000,
                "allowed_venues": [],
                "throttle": {"per_symbol_per_min": 120, "per_strategy_per_min": 600},
            },
            "symbols": {},
            "strategies": {},
            "regions": {},
            "hard_blocks": [],
        }

    # ----- utilities -----
    def ref_price(self, order: Order) -> Optional[float]:
        # Try order.meta first; else from limits hash (e.g., mid)
        px = order.meta.get("ref_px") if order.meta else None
        if px: 
            try: return float(px)
            except Exception: pass
        # fallback: read from limits hash if some producer writes mids there (optional)
        return None

    def _roll_hit(self, key: str, window_sec: int, limit: int) -> Tuple[bool, int]:
        """Return (limit_exceeded, count_in_window)."""
        now = int(time.time())
        arr = self._roll.setdefault(key, [])
        # prune
        while arr and arr[0] < now - window_sec:
            arr.pop(0)
        arr.append(now)
        return (len(arr) > limit, len(arr))

    def _limits_overlay(self, order: Order) -> Dict[str, Any]:
        """Combine defaults + symbol + strategy + region overlays."""
        d = dict(self.cfg.get("defaults", {}))
        sym = (self.cfg.get("symbols", {}) or {}).get(order.symbol.upper(), {})
        strat = (self.cfg.get("strategies", {}) or {}).get(order.strategy, {})
        reg = (self.cfg.get("regions", {}) or {}).get((order.region or "").upper(), {})
        # shallow overlay
        for layer in (sym, strat, reg):
            for k, v in (layer or {}).items():
                d[k] = v
        return d

    # ----- check registry -----
    def _register(self, name: str, fn: CheckFn):
        self.checks.append((name, fn))

    def _register_builtin_checks(self):
        self._register("kill_switch", self._check_kill) # type: ignore
        self._register("hard_blocks", self._check_hard_blocks) # type: ignore
        self._register("venue_allow", self._check_venue) # type: ignore
        self._register("notional_caps", self._check_notional) # type: ignore
        self._register("qty_caps", self._check_qty) # type: ignore
        self._register("price_band", self._check_price_band) # type: ignore
        self._register("throttle", self._check_throttle) # type: ignore
        self._register("dynamic_limits", self._check_dynamic_limits) # type: ignore
        self._register("quarantine", self._check_quarantine) # type: ignore

    # ----- checks (return (passed, details)) ---------------------------------
    def _check_kill(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        ks = (self.b.get(KEY_KILL) or "off").lower() == "on"
        return (not ks, {"fatal": True, "kill": ks, "reason": "global_kill_on" if ks else ""})

    def _check_hard_blocks(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        for rule in self.cfg.get("hard_blocks", []) or []:
            field = (rule.get("field") or "").strip()
            pattern = rule.get("regex")
            if not field or not pattern: 
                continue
            val = str(getattr(order, field, "") or order.meta.get(field, ""))
            if re.search(pattern, val, flags=re.IGNORECASE):
                return False, {"fatal": True, "rule": rule, "reason": rule.get("reason") or "hard_block"}
        return True, {}

    def _check_venue(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        allowed = lim.get("allowed_venues") or []
        if allowed and (order.venue or "") not in allowed:
            return False, {"fatal": True, "reason": f"venue_not_allowed:{order.venue}", "allowed": allowed}
        return True, {}

    def _check_notional(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        max_notional = float(lim.get("max_order_notional", 0) or 0)
        if max_notional <= 0:
            return True, {}
        px = float(order.limit_price or order.meta.get("ref_px") or 0.0)
        notional = abs(order.qty) * (px if px>0 else 0.0)
        if notional > max_notional:
            # modify: cap qty
            cap_qty = math.floor(max_notional / max(1e-9, px))
            return False, {"fatal": False, "reason": "cap_notional", "cap_qty": cap_qty, "max_notional": max_notional}
        return True, {}

    def _check_qty(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        mx = float(lim.get("max_qty", 0) or 0)
        if mx > 0 and abs(order.qty) > mx:
            return False, {"fatal": False, "reason": "cap_qty", "cap_qty": math.copysign(mx, order.qty)}
        return True, {}

    def _check_price_band(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        # Allow market orders if no reference price available
        band_bps = float(lim.get("price_band_bps", 0) or 0)
        if band_bps <= 0 or order.typ.lower() == "market":
            return True, {}
        ref = self.ref_price(order)
        if not ref:
            return True, {"note":"no_ref_price"}
        lp = float(order.limit_price or 0.0)
        if lp <= 0: 
            return True, {}
        s = _sgn(order.side)
        # buy limit cannot exceed ref*(1+band); sell limit cannot be below ref*(1-band)
        max_buy = ref * (1 + band_bps/1e4)
        min_sell = ref * (1 - band_bps/1e4)
        ok = (s>0 and lp <= max_buy) or (s<0 and lp >= min_sell)
        det = {"ref_px": ref, "limit_px": lp, "band_bps": band_bps, "max_buy": max_buy, "min_sell": min_sell}
        return (ok, {"fatal": True, "reason": "price_band_violation", **det} if not ok else det)

    def _check_throttle(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        th = lim.get("throttle") or {}
        sym_lim = int(th.get("per_symbol_per_min", 0) or 0)
        strat_lim = int(th.get("per_strategy_per_min", 0) or 0)
        hit1 = hit2 = False
        c1 = c2 = 0
        if sym_lim > 0:
            hit1, c1 = self._roll_hit(f"s:{order.symbol}", 60, sym_lim)
        if strat_lim > 0:
            hit2, c2 = self._roll_hit(f"st:{order.strategy}", 60, strat_lim)
        if hit1 or hit2:
            return False, {"fatal": True, "reason": "throttle", "symbol_count": c1, "strategy_count": c2}
        return True, {}

    def _check_dynamic_limits(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Consult dynamic limits written by the dynamic governor:
          - global: HASH_LIMITS
          - per strategy: f"{HASH_LIMITS}:{strategy}"
        Enforce `halt_new`, `disabled`, `max_order_val_usd`, `throttle_bps` (convert to no-op here), etc.
        """
        # global gates
        g = self.b.hgetall(HASH_LIMITS) or {}
        if (g.get("halt_new") or "").lower() in ("1","true","on","yes"):
            return False, {"fatal": True, "reason":"halt_new_global"}
        # per-strategy
        s = self.b.hgetall(f"{HASH_LIMITS}:{order.strategy}") or {}
        if (s.get("disabled") or "").lower() in ("1","true","on","yes"):
            return False, {"fatal": True, "reason": f"strategy_disabled:{order.strategy}"}
        # optional per-order cap
        cap = s.get("max_order_val_usd") or g.get("max_order_val_usd")
        if cap:
            try:
                capf = float(cap)
                px = float(order.limit_price or order.meta.get("ref_px") or 0.0)
                if px>0 and abs(order.qty) * px > capf:
                    nq = math.floor(capf / px) * (1 if order.qty>=0 else -1)
                    return False, {"fatal": False, "reason": "cap_dyn_order_val", "cap_qty": nq, "max_order_val_usd": capf}
            except Exception:
                pass
        return True, {}

    def _check_quarantine(self, order: Order, lim: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        # Simple flag: set in order.meta or in limits hash (e.g., per symbol)
        if bool(order.meta.get("quarantined")):
            return False, {"fatal": True, "reason": "quarantined_order"}
        return True, {}

    # ----- evaluate -----------------------------------------------------------
    def evaluate(self, order_dict: Dict[str, Any]) -> Decision:
        order = Order(**order_dict)
        lim = self._limits_overlay(order)

        changes: Dict[str, Any] = {}
        details: List[Dict[str, Any]] = []
        hard_denied = False
        deny_reason = ""

        for name, fn in self.checks:
            ok, det = fn(order, lim) # type: ignore
            det = det or {}
            det["check"] = name
            details.append(det)
            if not ok:
                if det.get("fatal", True):
                    hard_denied = True
                    deny_reason = det.get("reason") or name
                    break
                # soft failure → propose modification
                if "cap_qty" in det:
                    changes["qty"] = float(det["cap_qty"])
                if det.get("reason") == "cap_notional" and "cap_qty" in det:
                    changes["qty"] = float(det["cap_qty"])

        if hard_denied:
            return Decision(action="deny", reason=deny_reason, order=order, changes={}, checks=details)

        if changes:
            # apply modifications
            new = {**asdict(order), **changes}
            mod_order = Order(**new)
            dec = Decision(action="modify", reason="modified_to_fit_limits", order=mod_order, changes=changes, checks=details)
        else:
            dec = Decision(action="allow", reason="ok", order=order, changes={}, checks=details)

        # audit
        self._audit(dec)
        return dec

    def _audit(self, decision: Decision) -> None:
        try:
            self.b.xadd(STREAM_AUDIT, {"type":"policy", **decision.to_dict()})
        except Exception:
            pass

# ----- CLI -------------------------------------------------------------------
def _cli():
    import argparse, sys
    ap = argparse.ArgumentParser("policy_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="Evaluate a single order JSON")
    e.add_argument("--order-json", required=True, help='Path to JSON like {"strategy":"alpha","symbol":"AAPL","side":"buy","qty":100,...}')
    e.add_argument("--cfg", default=CFG_PATH)

    args = ap.parse_args()
    if args.cmd == "eval":
        with open(args.order_json, "r", encoding="utf-8") as f:
            o = json.load(f)
        pe = PolicyEngine(cfg_path=args.cfg)
        dec = pe.evaluate(o)
        print(json.dumps(dec.to_dict(), indent=2))

if __name__ == "__main__":
    _cli()