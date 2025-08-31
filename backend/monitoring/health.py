# backend/ops/chaos.py
from __future__ import annotations

import os, sys, time, json, math, random, threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Callable, Tuple, List

# ---------- optional Redis (graceful fallback) ----------
HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

# ---------- env / defaults ----------
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
IN_TICKS_STREAM  = os.getenv("TICKS_STREAM", "md.trades")
OUT_TICKS_STREAM = os.getenv("TICKS_STREAM_OUT", IN_TICKS_STREAM)  # default in-place
KILLSWITCH_KEY   = os.getenv("RISK_KILLSWITCH_KEY", "risk:killswitch")

# Global enable gate (can be flipped at runtime by CLI subcommands)
CHAOS_ENABLED = (os.getenv("CHAOS_ENABLED") or "false").lower() in ("1","true","on","yes")

# ---------- Models ----------
@dataclass
class ChaosProfile:
    enabled: bool = False
    # timing
    delay_ms_mean: float = 0.0
    delay_ms_jitter: float = 0.0    # +/- jitter around mean
    # reliability
    drop_prob: float = 0.0          # 0..1
    dup_prob: float = 0.0           # 0..1 (emit twice)
    fail_prob: float = 0.0          # raises RuntimeError in decorated call
    # numeric jitter
    jitter_pct: float = 0.0         # e.g., 0.001 = 10 bps (±)
    jitter_fields: Tuple[str, ...] = ("price", "p", "limit_price", "qty", "size", "q")
    # bounds (optional guards)
    max_delay_ms: int = 10_000

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------- Backend helpers ----------
class _KV:
    def __init__(self, url: Optional[str] = None):
        self.r = None
        if HAVE_REDIS:
            try:
                self.r = Redis.from_url(url or REDIS_URL, decode_responses=True)  # type: ignore
                self.r.ping()
            except Exception:
                self.r = None
    def xadd(self, stream: str, obj: Dict[str, Any], maxlen: int = 200_000):
        if self.r:
            try:
                self.r.xadd(stream, {"json": json.dumps(obj)}, maxlen=maxlen, approximate=True)  # type: ignore
                return
            except Exception:
                pass
        # fallback: print
        print(f"[chaos:xadd:{stream}] {obj}")
    def set(self, key: str, val: str): 
        if self.r:
            try: self.r.set(key, val)  # type: ignore
            except Exception: pass

# ---------- Core engine ----------
class ChaosEngine:
    """
    Stateless random fault injector controlled by a ChaosProfile.
    Use directly (per message) or via decorator/context manager.
    """
    def __init__(self, profile: Optional[ChaosProfile] = None):
        self.profile = profile or ChaosProfile(enabled=False)

    # ---- per-call effects ---------------------------------------------------
    def maybe_delay(self):
        if not self.profile.enabled: return
        mu = max(0.0, float(self.profile.delay_ms_mean))
        jit = max(0.0, float(self.profile.delay_ms_jitter))
        if mu <= 0 and jit <= 0: return
        d = mu + random.uniform(-jit, +jit)
        d = max(0.0, min(d, float(self.profile.max_delay_ms)))
        if d > 0: time.sleep(d / 1000.0)

    def maybe_fail(self):
        if self.profile.enabled and random.random() < self.profile.fail_prob:
            raise RuntimeError("ChaosEngine induced failure")

    def maybe_drop(self) -> bool:
        """Return True if the message should be dropped."""
        return self.profile.enabled and (random.random() < self.profile.drop_prob)

    def maybe_dup(self) -> bool:
        """Return True if the message should be duplicated."""
        return self.profile.enabled and (random.random() < self.profile.dup_prob)

    def jitter_number(self, x: Any) -> Any:
        if not self.profile.enabled: return x
        try:
            x = float(x)
        except Exception:
            return x
        pct = float(self.profile.jitter_pct or 0.0)
        if pct <= 0: return x
        # symmetric multiplicative jitter
        eps = 1.0 + random.uniform(-pct, +pct)
        return x * eps

    def jitter_payload(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        if not self.profile.enabled: return obj
        out = dict(obj)
        for k in self.profile.jitter_fields:
            if k in out:
                out[k] = self.jitter_number(out[k])
        return out

    # ---- decorator / context manager ---------------------------------------
    def guard(self, *, apply_delay=True):
        """
        Decorator/context: inject delay/failure before running function.
        """
        eng = self
        class _Guard:
            def __init__(self, f=None): self.f = f
            def __call__(self, *a, **k):
                if not eng.profile.enabled: return self.f(*a, **k) # type: ignore
                if apply_delay: eng.maybe_delay()
                eng.maybe_fail()
                return self.f(*a, **k) # type: ignore
            def __enter__(self):
                if apply_delay: eng.maybe_delay()
                eng.maybe_fail()
                return self
            def __exit__(self, exc_type, exc, tb): return False
        def deco(f):
            return _Guard(f)
        return deco

# ---------- Wrappers for streams/gateways -----------------------------------
class StreamChaosProxy:
    """
    Wrap a publish/consume path to inject chaos into message flow.
    """
    def __init__(self, profile: ChaosProfile, *, redis_url: Optional[str] = None):
        self.eng = ChaosEngine(profile)
        self.kv = _KV(redis_url)

    def publish(self, stream: str, payload: Dict[str, Any]):
        # maybe drop outright
        if self.eng.maybe_drop(): 
            return
        # jitter numeric fields
        jittered = self.eng.jitter_payload(payload)
        # delay
        self.eng.maybe_delay()
        # publish
        self.kv.xadd(stream, jittered)
        # duplicate?
        if self.eng.maybe_dup():
            self.kv.xadd(stream, jittered)

# ---------- Pre-canned drills -----------------------------------------------
def drill_flash_crash(*, symbol: str, px0: float, drop_pct: float = 0.07, steps: int = 10, interval_ms: int = 100,
                      stream: str = OUT_TICKS_STREAM, size: float = 1.0):
    """
    Publish a staged flash-crash (and rebound) to ticks stream.
    """
    kv = _KV()
    now = int(time.time()*1000)
    # down
    for i in range(steps):
        px = px0 * (1 - drop_pct * ((i+1)/steps))
        kv.xadd(stream, {"type":"trade","json": json.dumps({"ts_ms": now+i*interval_ms, "symbol": symbol, "price": px, "size": size})})
        time.sleep(interval_ms/1000.0)
    # rebound
    for i in range(steps):
        px = px0 * (1 - drop_pct * (1 - (i+1)/steps))
        kv.xadd(stream, {"type":"trade","json": json.dumps({"ts_ms": now+(steps+i)*interval_ms, "symbol": symbol, "price": px, "size": size})})
        time.sleep(interval_ms/1000.0)

def drill_spike(*, symbol: str, px0: float, up_pct: float = 0.05, steps: int = 5, interval_ms: int = 100,
                stream: str = OUT_TICKS_STREAM, size: float = 1.0):
    kv = _KV()
    now = int(time.time()*1000)
    for i in range(steps):
        px = px0 * (1 + up_pct * ((i+1)/steps))
        kv.xadd(stream, {"type":"trade","json": json.dumps({"ts_ms": now+i*interval_ms, "symbol": symbol, "price": px, "size": size})})
        time.sleep(interval_ms/1000.0)

def drill_toggle_kill(on: bool, reason: str = "chaos_drill"):
    kv = _KV()
    kv.set(KILLSWITCH_KEY, "on" if on else "off")
    kv.xadd("risk.commands", {"type":"kill_switch","json": json.dumps({"on": on, "reason": reason, "ts_ms": int(time.time()*1000)})})

def flood(*, stream: str, n: int = 10_000, symbol: str = "TEST", px: float = 100.0, size: float = 1.0):
    kv = _KV()
    now = int(time.time()*1000)
    for i in range(n):
        kv.xadd(stream, {"type":"trade","json": json.dumps({"ts_ms": now+i, "symbol": symbol, "price": px, "size": size})})

# ---------- CLI --------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("chaos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # live proxy
    runp = sub.add_parser("proxy", help="Read from IN_TICKS_STREAM, inject chaos, write to OUT_TICKS_STREAM")
    runp.add_argument("--delay-mean", type=float, default=float(os.getenv("CHAOS_DELAY_MS", "0")))
    runp.add_argument("--delay-jitter", type=float, default=float(os.getenv("CHAOS_JITTER_MS", "0")))
    runp.add_argument("--drop", type=float, default=float(os.getenv("CHAOS_DROP_PROB", "0")))
    runp.add_argument("--dup", type=float, default=float(os.getenv("CHAOS_DUP_PROB", "0")))
    runp.add_argument("--fail", type=float, default=float(os.getenv("CHAOS_FAIL_PROB", "0")))
    runp.add_argument("--jitter-pct", type=float, default=float(os.getenv("CHAOS_JITTER_PCT", "0")))
    runp.add_argument("--fields", default="price,p,limit_price,qty,size,q")
    runp.add_argument("--in", dest="in_stream", default=IN_TICKS_STREAM)
    runp.add_argument("--out", dest="out_stream", default=OUT_TICKS_STREAM)

    # drills
    fc = sub.add_parser("flash-crash")
    fc.add_argument("--symbol", required=True)
    fc.add_argument("--px0", type=float, required=True)
    fc.add_argument("--drop", type=float, default=0.07)
    fc.add_argument("--steps", type=int, default=10)
    fc.add_argument("--interval-ms", type=int, default=100)

    sp = sub.add_parser("spike")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--px0", type=float, required=True)
    sp.add_argument("--up", type=float, default=0.05)
    sp.add_argument("--steps", type=int, default=5)
    sp.add_argument("--interval-ms", type=int, default=100)

    ks = sub.add_parser("kill")
    ks.add_argument("--on", action="store_true")
    ks.add_argument("--off", action="store_true")

    fl = sub.add_parser("flood")
    fl.add_argument("--stream", default=OUT_TICKS_STREAM)
    fl.add_argument("--n", type=int, default=10000)
    fl.add_argument("--symbol", default="TEST")
    fl.add_argument("--px", type=float, default=100.0)
    fl.add_argument("--size", type=float, default=1.0)

    args = ap.parse_args()

    if args.cmd == "proxy":
        if not HAVE_REDIS:
            raise SystemExit("Redis not available for proxy mode.")
        prof = ChaosProfile(
            enabled=True,
            delay_ms_mean=args.delay_mean,
            delay_ms_jitter=args.delay_jitter,
            drop_prob=args.drop,
            dup_prob=args.dup,
            fail_prob=args.fail,
            jitter_pct=args.jitter_pct,
            jitter_fields=tuple([s.strip() for s in args.fields.split(",") if s.strip()]),
        )
        proxy = StreamChaosProxy(prof)
        r = Redis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
        last_id = "$"
        print(f"[chaos] proxy {args.in_stream} -> {args.out_stream} with {prof.as_dict()}")
        while True:
            resp = r.xread({args.in_stream: last_id}, count=500, block=1000)  # type: ignore
            if not resp:
                continue
            _, entries = resp[0] # type: ignore
            last_id = entries[-1][0]
            for _id, fields in entries:
                raw = fields.get("json") or fields.get("data") or ""
                try:
                    msg = json.loads(raw) if raw else fields
                except Exception:
                    msg = fields
                proxy.publish(args.out_stream, msg)

    elif args.cmd == "flash-crash":
        drill_flash_crash(symbol=args.symbol, px0=args.px0, drop_pct=args.drop, steps=args.steps, interval_ms=args.interval_ms)

    elif args.cmd == "spike":
        drill_spike(symbol=args.symbol, px0=args.px0, up_pct=args.up, steps=args.steps, interval_ms=args.interval_ms)

    elif args.cmd == "kill":
        if args.on == args.off:
            raise SystemExit("Specify --on or --off")
        drill_toggle_kill(on=args.on, reason="cli")

    elif args.cmd == "flood":
        flood(stream=args.stream, n=args.n, symbol=args.symbol, px=args.px, size=args.size)

if __name__ == "__main__":
    _cli()