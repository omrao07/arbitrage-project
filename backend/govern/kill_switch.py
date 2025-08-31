# backend/risk/kill_switch.py
from __future__ import annotations

import os, time, json, threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# ---------- optional Redis (graceful fallback) ----------
HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

# ---------- env / defaults ----------
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KEY_GLOBAL      = os.getenv("RISK_KILLSWITCH_KEY", "risk:killswitch")           # "on"/"off"
KEY_PREFIX_SCOP = os.getenv("RISK_KILLSWITCH_SCOPE_PREFIX", "risk:kills:")      # risk:kills:<scope>
STREAM_COMMANDS = os.getenv("RISK_COMMANDS_STREAM", "risk.commands")            # broadcast decisions
DEFAULT_TTL_SEC = int(os.getenv("RISK_KILL_TTL_SEC", "0"))  # 0 = no TTL (sticky until turned off)

# ---------- data model ----------
@dataclass
class KillState:
    on: bool
    ts_ms: int
    reason: str = ""
    by: str = ""         # operator/service
    ttl_sec: int = 0     # 0 = sticky
    scope: str = "global"  # "global" or "strategy:<name>" or "venue:<id>" etc.

    def to_kv(self) -> Dict[str, str]:
        return {
            "on": "on" if self.on else "off",
            "ts_ms": str(self.ts_ms),
            "reason": self.reason,
            "by": self.by,
            "ttl_sec": str(int(self.ttl_sec or 0)),
            "scope": self.scope,
        }

# ---------- backend ----------
class _Backend:
    """Simple key/value + stream wrapper with Redis (or memory fallback)."""
    def __init__(self, redis_url: Optional[str] = None):
        self.mem: Dict[str, Dict[str, str]] = {}
        self.redis = None
        if HAVE_REDIS:
            try:
                self.redis = Redis.from_url(redis_url or REDIS_URL, decode_responses=True)  # type: ignore
                # touch to verify
                self.redis.ping()
            except Exception:
                self.redis = None

    def hgetall(self, key: str) -> Dict[str, str]:
        if self.redis:
            try:
                return self.redis.hgetall(key) or {} # type: ignore
            except Exception:
                pass
        return dict(self.mem.get(key) or {})

    def hset(self, key: str, mapping: Dict[str, str]) -> None:
        if self.redis:
            try:
                self.redis.hset(key, mapping=mapping)  # type: ignore
                return
            except Exception:
                pass
        self.mem[key] = dict(mapping)

    def set_ttl(self, key: str, ttl_sec: int) -> None:
        if ttl_sec and self.redis:
            try:
                self.redis.expire(key, int(ttl_sec))  # type: ignore
                return
            except Exception:
                pass
        # memory fallback TTL via background thread
        if ttl_sec:
            def _expire():
                time.sleep(ttl_sec)
                self.mem.pop(key, None)
            threading.Thread(target=_expire, daemon=True).start()

    def xadd(self, stream: str, obj: Dict[str, Any], maxlen: int = 50_000) -> None:
        if self.redis:
            try:
                self.redis.xadd(stream, {"json": json.dumps(obj)}, maxlen=maxlen, approximate=True)  # type: ignore
                return
            except Exception:
                pass
        # memory fallback: ignore (no stream)

# ---------- KillSwitch ----------
class KillSwitch:
    """
    Global + scoped kill switch with TTL and audit info.
    Use in your router/executor to hard-block order flow when engaged.
    """
    def __init__(self, *, redis_url: Optional[str] = None, default_ttl_sec: int = DEFAULT_TTL_SEC):
        self.b = _Backend(redis_url)
        self.default_ttl = max(0, int(default_ttl_sec))

    # ---- keys / scopes ----
    @staticmethod
    def scope_key(scope: str) -> str:
        s = (scope or "global").strip().lower()
        return KEY_GLOBAL if s == "global" else (KEY_PREFIX_SCOP + s)

    # ---- state ops ----
    def get(self, scope: str = "global") -> KillState:
        kv = self.b.hgetall(self.scope_key(scope))
        if not kv:
            return KillState(on=False, ts_ms=int(time.time()*1000), scope=scope)
        return KillState(
            on=str(kv.get("on","off")) == "on",
            ts_ms=int(kv.get("ts_ms", str(int(time.time()*1000)))),
            reason=str(kv.get("reason","")),
            by=str(kv.get("by","")),
            ttl_sec=int(kv.get("ttl_sec","0") or 0),
            scope=str(kv.get("scope") or scope),
        )

    def is_on(self, scope: str = "global") -> bool:
        st = self.get(scope)
        if st.on:
            return True
        # if scope off, inherit from global
        if scope != "global":
            return self.get("global").on
        return False

    def turn_on(self, *, scope: str = "global", reason: str = "", by: str = "", ttl_sec: Optional[int] = None) -> KillState:
        ttl = self.default_ttl if ttl_sec is None else max(0, int(ttl_sec))
        st = KillState(on=True, ts_ms=int(time.time()*1000), reason=reason, by=by, ttl_sec=ttl, scope=scope)
        key = self.scope_key(scope)
        self.b.hset(key, st.to_kv())
        self.b.set_ttl(key, ttl)
        self._broadcast(st)
        return st

    def turn_off(self, *, scope: str = "global", by: str = "") -> KillState:
        st = KillState(on=False, ts_ms=int(time.time()*1000), reason="manual_off", by=by, ttl_sec=0, scope=scope)
        key = self.scope_key(scope)
        self.b.hset(key, st.to_kv())
        # if Redis, we keep key but set on=off; memory fallback just overwrites
        self._broadcast(st)
        return st

    def toggle(self, *, scope: str = "global", reason: str = "", by: str = "", ttl_sec: Optional[int] = None) -> KillState:
        return self.turn_off(scope=scope, by=by) if self.is_on(scope) else self.turn_on(scope=scope, reason=reason, by=by, ttl_sec=ttl_sec)

    # ---- helpers / integration ----
    def guard(self, *, scope: str = "global"):
        """
        Decorator/context-manager: block function if kill is ON (scope or global).
        Usage:
            @ks.guard(scope="strategy:alpha1")
            def place_order(...): ...
        """
        ks = self
        class _Guard:
            def __init__(self, f=None): self.f = f
            def __call__(self, *a, **kw):
                if ks.is_on(scope): raise RuntimeError(f"killswitch[{scope}] is ON")
                return self.f(*a, **kw) # type: ignore
            def __enter__(self): 
                if ks.is_on(scope): raise RuntimeError(f"killswitch[{scope}] is ON")
                return self
            def __exit__(self, exc_type, exc, tb): return False
        def deco(f):
            g = _Guard(f)
            return g
        return deco

    def _broadcast(self, st: KillState) -> None:
        payload = {"type":"kill_switch", **asdict(st)}
        try:
            self.b.xadd(STREAM_COMMANDS, payload)
        except Exception:
            pass

# ---------- CLI ----------
def _cli():
    import argparse, sys
    ap = argparse.ArgumentParser("kill_switch")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("get", help="Show kill state")
    g.add_argument("--scope", default="global")

    onp = sub.add_parser("on", help="Turn ON kill switch")
    onp.add_argument("--scope", default="global")
    onp.add_argument("--reason", default="")
    onp.add_argument("--by", default="cli")
    onp.add_argument("--ttl", type=int, default=None, help="TTL seconds (0=sticky)")

    offp = sub.add_parser("off", help="Turn OFF kill switch")
    offp.add_argument("--scope", default="global")
    offp.add_argument("--by", default="cli")

    tog = sub.add_parser("toggle", help="Toggle kill switch")
    tog.add_argument("--scope", default="global")
    tog.add_argument("--reason", default="")
    tog.add_argument("--by", default="cli")
    tog.add_argument("--ttl", type=int, default=None)

    args = ap.parse_args()
    ks = KillSwitch()

    if args.cmd == "get":
        st = ks.get(args.scope)
        print(json.dumps(asdict(st), indent=2))
    elif args.cmd == "on":
        st = ks.turn_on(scope=args.scope, reason=args.reason, by=args.by, ttl_sec=args.ttl)
        print(json.dumps(asdict(st), indent=2))
    elif args.cmd == "off":
        st = ks.turn_off(scope=args.scope, by=args.by)
        print(json.dumps(asdict(st), indent=2))
    elif args.cmd == "toggle":
        st = ks.toggle(scope=args.scope, reason=args.reason, by=args.by, ttl_sec=args.ttl)
        print(json.dumps(asdict(st), indent=2))

if __name__ == "__main__":
    _cli()