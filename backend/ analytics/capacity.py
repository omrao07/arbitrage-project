# backend/risk/capacity.py
from __future__ import annotations

import os, time, json, math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

# ---------- Optional Redis ----------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---------- Keys / Streams ----------
K_POLICY     = os.getenv("CAP_POLICY_HASH",   "risk.capacity.policy")      # HSET <scope> -> json(policy)
K_STATE      = os.getenv("CAP_STATE_HASH",    "risk.capacity.state")       # HSET <scope> -> json(state)
K_THROTTLE   = os.getenv("CAP_THROTTLE_HASH", "risk.capacity.throttle")    # HSET <scope> -> json(bucket)
K_AUDIT      = os.getenv("CAP_AUDIT_STREAM",  "risk.capacity.audit")       # XADD audit
MAXLEN       = int(os.getenv("CAP_AUDIT_MAXLEN", "20000"))

# ---------- Models ----------
@dataclass
class CapacityPolicy:
    # hard limits
    max_notional_per_order: float | None = None     # e.g., 250_000
    max_notional_per_day: float | None = None       # strategy/symbol/day
    max_open_orders: int | None = None              # concurrency cap
    # participation
    max_participation_adv: float | None = None      # e.g., 0.08  (8% of ADV)
    max_participation_live: float | None = None     # e.g., 0.15  (15% of today live vol)
    max_participation_bar: float | None = None      # e.g., 0.20  (20% of bar vol)
    # throttling
    rate_per_sec: float | None = None               # average orders/sec
    burst: int | None = None                        # token bucket burst
    # safety
    enabled: bool = True

@dataclass
class CapacityState:
    ts_ms: int
    day: str
    used_notional_day: float = 0.0
    open_orders: int = 0
    live_volume_today: float = 0.0     # shares today
    bar_volume: float = 0.0            # shares in current bar
    adv_shares: float = 0.0            # 20d ADV or similar baseline
    last_bar_bucket: int = 0           # epoch bucket id for bar alignment

@dataclass
class Decision:
    allow: bool
    reason: str
    caps: Dict[str, Any]
    snapshot: Dict[str, Any]

# ---------- Utilities ----------
def now_ms() -> int: return int(time.time() * 1000)
def yyyymmdd(ts_ms: int | None = None) -> str:
    t = time.gmtime((ts_ms or now_ms())/1000)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"

def scope_key(strategy: str, symbol: str, venue: str | None = None) -> str:
    symbol = symbol.upper()
    strategy = strategy.lower()
    return f"{strategy}:{symbol}:{(venue or 'ANY').upper()}"

def _bar_bucket(ts_ms: int, sec: int = 60) -> int:
    return (ts_ms // 1000) // sec

# ---------- Store (Redis-backed, memory fallback) ----------
class Store:
    def __init__(self):
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.mem_policy: Dict[str, Dict[str, Any]] = {}
        self.mem_state: Dict[str, Dict[str, Any]] = {}
        self.mem_throttle: Dict[str, Dict[str, Any]] = {}

    async def connect(self):
        if not USE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    # policy
    async def load_policy(self, scope: str) -> Dict[str, Any] | None:
        if self.r:
            js = await self.r.hget(K_POLICY, scope)  # type: ignore
            if js: return json.loads(js)
        return self.mem_policy.get(scope)

    async def save_policy(self, scope: str, policy: CapacityPolicy):
        js = json.dumps(asdict(policy))
        if self.r:
            await self.r.hset(K_POLICY, scope, js)  # type: ignore
        self.mem_policy[scope] = json.loads(js)

    # state
    async def load_state(self, scope: str) -> Dict[str, Any] | None:
        if self.r:
            js = await self.r.hget(K_STATE, scope)  # type: ignore
            if js: return json.loads(js)
        return self.mem_state.get(scope)

    async def save_state(self, scope: str, st: CapacityState):
        js = json.dumps(asdict(st))
        if self.r:
            await self.r.hset(K_STATE, scope, js)  # type: ignore
        self.mem_state[scope] = json.loads(js)

    # throttle bucket (token bucket)
    async def load_bucket(self, scope: str) -> Dict[str, Any] | None:
        if self.r:
            js = await self.r.hget(K_THROTTLE, scope)  # type: ignore
            if js: return json.loads(js)
        return self.mem_throttle.get(scope)

    async def save_bucket(self, scope: str, b: Dict[str, Any]):
        js = json.dumps(b)
        if self.r:
            await self.r.hset(K_THROTTLE, scope, js)  # type: ignore
        self.mem_throttle[scope] = json.loads(js)

    async def audit(self, obj: Dict[str, Any]):
        if self.r:
            try:
                await self.r.xadd(K_AUDIT, {"json": json.dumps(obj)}, maxlen=MAXLEN, approximate=True)  # type: ignore
            except Exception:
                pass

# ---------- Core: CapacityManager ----------
class CapacityManager:
    """
    Enforce capacity before routing orders. Typical call flow:

        cm = CapacityManager()
        await cm.init()

        # when an order is about to be sent
        d = await cm.check_and_reserve(
            strategy="mm_core", symbol="TSLA", venue="NASDAQ",
            side="buy", qty=200, price=242.10,
            context={"adv": 25_000_000, "today_vol": 4_200_000, "bar_vol": 120_000}
        )
        if d.allow:
            # send order; on open/ack:
            await cm.on_order_open(scope, +1)
        else:
            # reject / downsize based on d.reason

        # when order fills:
        await cm.on_fill(scope, filled_qty, fill_price)

        # when order cancels:
        await cm.on_order_open(scope, -1)  # reduce open count
    """
    def __init__(self, bar_seconds: int = 60):
        self.store = Store()
        self.bar_seconds = bar_seconds

    async def init(self):
        await self.store.connect()

    # ----- Policy CRUD -------------------------------------------------------
    async def upsert_policy(self, strategy: str, symbol: str, venue: str | None, policy: CapacityPolicy):
        scope = scope_key(strategy, symbol, venue)
        await self.store.save_policy(scope, policy)
        # also initialize state if missing
        st = await self.store.load_state(scope)
        if not st:
            await self.store.save_state(scope, CapacityState(ts_ms=now_ms(), day=yyyymmdd()))

    # ----- Live data hooks ---------------------------------------------------
    async def on_market_volume(self, strategy: str, symbol: str, venue: str | None,
                               today_vol_shares: float, bar_vol_shares: float, adv_shares: float | None = None):
        scope = scope_key(strategy, symbol, venue)
        st = await self._get_state(scope)
        ts = now_ms()
        day = yyyymmdd(ts)
        if st.day != day:
            st = CapacityState(ts_ms=ts, day=day)
        st.ts_ms = ts
        st.live_volume_today = max(float(today_vol_shares), 0.0)
        st.bar_volume = max(float(bar_vol_shares), 0.0)
        if adv_shares is not None:
            st.adv_shares = max(float(adv_shares), 0.0)
        st.last_bar_bucket = _bar_bucket(ts, self.bar_seconds)
        await self.store.save_state(scope, st)

    async def on_order_open(self, strategy: str, symbol: str, venue: str | None, delta_open: int):
        scope = scope_key(strategy, symbol, venue)
        st = await self._get_state(scope)
        ts = now_ms(); day = yyyymmdd(ts)
        if st.day != day:
            st = CapacityState(ts_ms=ts, day=day)
        st.open_orders = max(0, int(st.open_orders) + int(delta_open))
        st.ts_ms = ts
        await self.store.save_state(scope, st)

    async def on_fill(self, strategy: str, symbol: str, venue: str | None, filled_qty: float, price: float):
        scope = scope_key(strategy, symbol, venue)
        st = await self._get_state(scope)
        ts = now_ms(); day = yyyymmdd(ts)
        if st.day != day:
            st = CapacityState(ts_ms=ts, day=day)
        notional = abs(float(filled_qty)) * float(price)
        st.used_notional_day += notional
        st.ts_ms = ts
        await self.store.save_state(scope, st)

    # ----- Throttle (token bucket) ------------------------------------------
    async def _throttle_take(self, scope: str, rate_per_sec: float | None, burst: int | None) -> Tuple[bool, Dict[str, Any]]:
        """
        Leaky/token bucket: allow if tokens >= 1 after refill. Refill amount = rate * dt.
        State: {tokens, last_ts}
        """
        if not rate_per_sec or rate_per_sec <= 0:
            return True, {"tokens": float("inf")}
        burst = int(burst or max(1, math.ceil(rate_per_sec)))
        b = await self.store.load_bucket(scope) or {"tokens": float(burst), "last_ts": now_ms()}
        now = now_ms()
        dt = max(0.0, (now - float(b.get("last_ts", now))) / 1000.0)
        tokens = float(b.get("tokens", burst))
        tokens = min(burst, tokens + rate_per_sec * dt)
        if tokens < 1.0:
            # deny this instant
            b.update({"tokens": tokens, "last_ts": now})
            await self.store.save_bucket(scope, b)
            return False, {"tokens": tokens, "burst": burst, "rate": rate_per_sec}
        # consume
        tokens -= 1.0
        b.update({"tokens": tokens, "last_ts": now})
        await self.store.save_bucket(scope, b)
        return True, {"tokens": tokens, "burst": burst, "rate": rate_per_sec}

    # ----- Main check --------------------------------------------------------
    async def check_and_reserve(
        self,
        strategy: str,
        symbol: str,
        venue: str | None,
        side: str,
        qty: float,
        price: float,
        *,
        context: Optional[Dict[str, Any]] = None
    ) -> Decision:
        """
        Evaluate a proposed order against capacity. If ALLOW, reservation is implicit for:
          - token bucket (consumes 1 token)
        Other state (open orders / notional) is updated when OMS ACKs or FILLS via on_order_open/on_fill.
        """
        scope = scope_key(strategy, symbol, venue)
        pol = await self._get_policy(scope)
        st  = await self._get_state(scope)

        ts = now_ms()
        # refresh intraday context if passed in (so the check is stateless to external feeds)
        ctx = context or {}
        if "today_vol" in ctx or "bar_vol" in ctx or "adv" in ctx:
            await self.on_market_volume(strategy, symbol, venue,
                                        ctx.get("today_vol") or st.live_volume_today,
                                        ctx.get("bar_vol") or st.bar_volume,
                                        ctx.get("adv") or st.adv_shares)
            st = await self._get_state(scope)

        # disabled
        if not pol.enabled:
            return await self._decision(scope, True, "policy_disabled", pol, st)

        notional = abs(float(qty)) * float(price)
        # ---- hard caps
        if pol.max_notional_per_order and notional > pol.max_notional_per_order:
            return await self._decision(scope, False, f"over_notional_per_order>{pol.max_notional_per_order}", pol, st)

        if pol.max_notional_per_day and (st.used_notional_day + notional) > pol.max_notional_per_day:
            return await self._decision(scope, False, "over_notional_per_day", pol, st)

        if pol.max_open_orders and st.open_orders >= pol.max_open_orders:
            return await self._decision(scope, False, "over_concurrency_open_orders", pol, st)

        # ---- participation caps
        # ADV (baseline)
        if pol.max_participation_adv and st.adv_shares > 0:
            allowed_shares = pol.max_participation_adv * st.adv_shares
            if (st.used_notional_day / max(price, 1e-9)) + abs(qty) > allowed_shares:
                return await self._decision(scope, False, "over_participation_adv", pol, st)

        # live day volume
        if pol.max_participation_live and st.live_volume_today > 0:
            allowed = pol.max_participation_live * st.live_volume_today
            if (st.used_notional_day / max(price, 1e-9)) + abs(qty) > allowed:
                return await self._decision(scope, False, "over_participation_live", pol, st)

        # bar volume (per minute or bar_seconds)
        cur_bucket = _bar_bucket(ts, self.bar_seconds)
        if st.last_bar_bucket != cur_bucket:
            # rotate bar counter
            st.bar_volume = ctx.get("bar_vol", st.bar_volume)
            st.last_bar_bucket = cur_bucket
            await self.store.save_state(scope, st)

        if pol.max_participation_bar and st.bar_volume > 0:
            allowed = pol.max_participation_bar * st.bar_volume
            if abs(qty) > allowed:
                return await self._decision(scope, False, "over_participation_bar", pol, st)

        # ---- throttle
        ok, tk = await self._throttle_take(scope, pol.rate_per_sec, pol.burst)
        if not ok:
            return await self._decision(scope, False, "throttle", pol, st, extra=tk)

        return await self._decision(scope, True, "ok", pol, st)

    # ----- Helpers -----------------------------------------------------------
    async def _get_policy(self, scope: str) -> CapacityPolicy:
        raw = await self.store.load_policy(scope)
        if raw:
            return CapacityPolicy(**raw)
        # default safe policy (no caps but enabled)
        pol = CapacityPolicy(enabled=True)
        await self.store.save_policy(scope, pol)
        return pol

    async def _get_state(self, scope: str) -> CapacityState:
        raw = await self.store.load_state(scope)
        if raw:
            return CapacityState(**raw)
        st = CapacityState(ts_ms=now_ms(), day=yyyymmdd())
        await self.store.save_state(scope, st)
        return st

    async def _decision(self, scope: str, allow: bool, reason: str,
                        pol: CapacityPolicy, st: CapacityState, extra: Dict[str, Any] | None = None) -> Decision:
        snap = {
            "policy": asdict(pol),
            "state": asdict(st),
            "extra": extra or {},
        }
        obj = {"ts_ms": now_ms(), "scope": scope, "allow": allow, "reason": reason, **snap}
        await self.store.audit(obj)
        return Decision(allow=allow, reason=reason, caps=pol.__dict__, snapshot=snap)