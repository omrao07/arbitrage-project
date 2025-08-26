# backend/risk/governor.py
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

# -------- Optional Redis + your stream helpers (kept soft) -------------------
try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

try:
    # Your earlier helper signatures:
    #   publish_stream(stream, payload)
    #   hset(key, field, value)
    from backend.bus.streams import publish_stream, hset  # type: ignore
except Exception:
    def publish_stream(_s, _p): pass
    def hset(_k, _f, _v): pass

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_R = None
if redis:
    try:
        _R = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    except Exception:
        _R = None

# -----------------------------------------------------------------------------


@dataclass
class Policy:
    """Static knobs (tunable at runtime via set_policy). All amounts in base CCY unless specified."""
    enabled: bool = True

    # Per-strategy limits
    max_gross_notional: float = 5_000_000.0     # cap on |qty*price| outstanding
    max_daily_turnover: float = 10_000_000.0    # buy+sell notional per day
    max_position_notional: float = 2_000_000.0  # per-symbol position cap
    max_order_notional: float = 250_000.0       # single child cap
    max_orders_per_min: int = 120               # spam guard
    min_clip_notional: float = 500.0            # tiny order trash filter

    # Risk throttles (per-strategy unless noted)
    daily_pnl_floor: float = -50_000.0          # stop-loss (strategy)
    daily_pnl_kill: float = -100_000.0          # kill-switch (strategy)
    global_daily_pnl_kill: float = -300_000.0   # kill-switch (global)
    rolling_dd_limit: float = 0.10              # 10% drawdown -> hard stop
    throttle_on_dd: float = 0.05                # 5% dd -> scale & slow

    # Scaling
    base_size_scale: float = 1.0                # master multiplier for size
    dd_scale_floor: float = 0.25                # min scale under stress
    vol_scale_ref: float = 0.02                 # 2% daily vol reference
    vol_scale_power: float = 0.5                # impact of vol on size

    # Market microstructure guards
    max_spread_bps_to_cross: float = 15.0       # reject crossing wide markets
    max_adversary_toxicity: float = 0.70        # if plugged into adversary suite
    news_halt_enabled: bool = True              # if a symbol is flagged -> block

    # Cooldowns / circuit breakers
    post_reject_cooldown_s: int = 5
    post_kill_cooldown_s: int = 300
    symbol_halt_cooldown_s: int = 120

    # Streams / keys
    audit_stream: str = os.getenv("GOV_AUDIT_STREAM", "governor.audit")
    news_halt_key: str = "halt:news:symbols"    # Redis set of halted symbols (optional)


@dataclass
class Stat:
    """Mutable, rolling stats (kept in memory + optionally mirrored in Redis)."""
    pnl_today: float = 0.0
    dd_frac: float = 0.0                 # 0.08 = 8% drawdown
    turnover_today: float = 0.0
    gross_outstanding: float = 0.0
    orders_last_min: int = 0
    last_reject_ts: float = 0.0
    last_kill_ts: float = 0.0


@dataclass
class Decision:
    allowed: bool
    reason: str = "ok"
    scale: float = 1.0
    notes: Dict[str, Any] = field(default_factory=dict)


class Governor:
    """
    Lightweight pre-trade risk gate + scaler.
    Call `check_and_scale(order, context)` before submitting to OMS.
    Maintain per-strategy & global stats via `ingest_fill/ingest_mark` periodically.
    """
    def __init__(self, policy: Optional[Policy] = None):
        self.policy = policy or Policy()
        self._per_strat: Dict[str, Stat] = {}
        self._global = Stat()
        self._per_symbol_halt: Dict[str, float] = {}
        self._last_min_bucket = int(time.time() // 60)

    # ------------------- Public API -------------------

    def set_policy(self, **updates) -> None:
        for k, v in updates.items():
            if hasattr(self.policy, k):
                setattr(self.policy, k, v)

    def register_strategy(self, name: str) -> None:
        self._per_strat.setdefault(name, Stat())
        hset("strategy:enabled", name, "true")

    def reset_day(self) -> None:
        self._global = Stat()
        for s in self._per_strat.values():
            s.pnl_today = s.turnover_today = s.orders_last_min = 0.0 # type: ignore
        self._per_symbol_halt.clear()

    def ingest_fill(self, *, strategy: str, symbol: str, qty: float, price: float, side: str, fees: float = 0.0) -> None:
        """
        Update turnover & PnL after a trade is confirmed.
        For simplicity: signed pnl from trade vs mark handled in ingest_mark.
        """
        st = self._per_strat.setdefault(strategy, Stat())
        notional = abs(qty * price)
        st.turnover_today += notional
        self._global.turnover_today += notional

    def ingest_mark(self, *, strategy: Optional[str] = None, pnl_delta: float = 0.0, dd_frac: Optional[float] = None) -> None:
        """
        Periodic mark-to-market: feed incremental PnL and (optionally) updated drawdown fraction.
        """
        if strategy:
            st = self._per_strat.setdefault(strategy, Stat())
            st.pnl_today += pnl_delta
            if dd_frac is not None:
                st.dd_frac = dd_frac
        self._global.pnl_today += pnl_delta

    def update_outstanding(self, *, strategy: str, gross_outstanding: float) -> None:
        self._per_strat.setdefault(strategy, Stat()).gross_outstanding = max(0.0, gross_outstanding)

    def set_symbol_halt(self, symbol: str, halted: bool = True) -> None:
        if halted:
            self._per_symbol_halt[symbol.upper()] = time.time()
            if _R and self.policy.news_halt_key:
                try: _R.sadd(self.policy.news_halt_key, symbol.upper())
                except Exception: pass
        else:
            self._per_symbol_halt.pop(symbol.upper(), None)
            if _R and self.policy.news_halt_key:
                try: _R.srem(self.policy.news_halt_key, symbol.upper())
                except Exception: pass

    def check_and_scale(
        self,
        order: Dict[str, Any],
        *,
        nbbo: Optional[Dict[str, float]] = None,
        adversary_toxicity: Optional[float] = None,
    ) -> Decision:
        """
        Enforce limits & return possibly-scaled size.
        `order` must include: strategy, symbol, side, qty, price (mark), notional (optional).
        """
        p = self.policy
        if not p.enabled:
            return Decision(True, "disabled")

        now = time.time()
        strat = str(order.get("strategy", "") or "")
        symbol = str(order.get("symbol", "") or "").upper()
        side = str(order.get("side", "") or "").lower()
        price = float(order.get("mark_price") or order.get("price") or 0.0)
        qty = float(order.get("qty") or 0.0)
        notional = float(order.get("notional") or abs(qty * price))

        if qty <= 0 or notional <= 0:
            return Decision(False, "empty_order")

        st = self._per_strat.setdefault(strat, Stat())

        # minute bucket
        bucket = int(now // 60)
        if bucket != self._last_min_bucket:
            for s in self._per_strat.values():
                s.orders_last_min = 0
            self._last_min_bucket = bucket

        # cooldowns
        if now - st.last_kill_ts < p.post_kill_cooldown_s:
            return self._reject(strat, "cooldown_after_kill", now, {"cooldown_s": p.post_kill_cooldown_s})
        if now - st.last_reject_ts < p.post_reject_cooldown_s:
            return self._reject(strat, "cooldown_after_reject", now, {"cooldown_s": p.post_reject_cooldown_s})

        # global kill
        if self._global.pnl_today <= p.global_daily_pnl_kill:
            return self._kill(strat, "global_pnl_kill", now, {"pnl": self._global.pnl_today})

        # news / symbol halts
        if p.news_halt_enabled:
            halted = symbol in self._per_symbol_halt
            if not halted and _R and p.news_halt_key:
                try:
                    halted = bool(_R.sismember(p.news_halt_key, symbol))
                except Exception:
                    halted = False
            if halted:
                ts = self._per_symbol_halt.get(symbol, 0.0)
                if now - ts < p.symbol_halt_cooldown_s:
                    return self._reject(strat, "symbol_halt", now, {"symbol": symbol})

        # microstructure: spread guard
        if nbbo and nbbo.get("bid") and nbbo.get("ask"):
            spread_bps = 1e4 * (nbbo["ask"] - nbbo["bid"]) / max(1e-9, (nbbo["ask"] + nbbo["bid"]) / 2.0)
            if side in ("buy", "sell") and spread_bps > p.max_spread_bps_to_cross and order.get("typ", "market") == "market":
                return self._reject(strat, "wide_spread", now, {"spread_bps": round(spread_bps, 2)})

        # adversary toxicity
        if adversary_toxicity is not None and adversary_toxicity >= p.max_adversary_toxicity:
            return self._reject(strat, "adversary_toxicity", now, {"tox": adversary_toxicity})

        # per-strategy hard stops
        if st.pnl_today <= p.daily_pnl_kill or st.dd_frac >= p.rolling_dd_limit:
            return self._kill(strat, "strategy_kill", now, {"pnl": st.pnl_today, "dd": st.dd_frac})

        # spam / turnover / size caps
        st.orders_last_min += 1
        if st.orders_last_min > p.max_orders_per_min:
            return self._reject(strat, "rate_limit_orders_per_min", now, {"count": st.orders_last_min})
        if notional < p.min_clip_notional:
            return self._reject(strat, "too_small", now, {"notional": notional})
        if notional > p.max_order_notional:
            return self._reject(strat, "order_cap", now, {"notional": notional, "cap": p.max_order_notional})
        if st.turnover_today + notional > p.max_daily_turnover:
            return self._reject(strat, "turnover_cap", now, {"turnover": st.turnover_today, "cap": p.max_daily_turnover})
        if st.gross_outstanding + notional > p.max_gross_notional:
            return self._reject(strat, "gross_cap", now, {"gross": st.gross_outstanding, "cap": p.max_gross_notional})
        if order.get("pos_notional_after", 0.0) and order["pos_notional_after"] > p.max_position_notional:
            return self._reject(strat, "position_cap", now, {"pos_after": order["pos_notional_after"], "cap": p.max_position_notional})

        # soft throttles → scale size
        scale = p.base_size_scale

        # throttle on drawdown
        if st.dd_frac >= p.throttle_on_dd:
            # quadratic approach to floor as dd→limit
            t = min(1.0, (st.dd_frac - p.throttle_on_dd) / max(1e-9, (p.rolling_dd_limit - p.throttle_on_dd)))
            scale *= max(p.dd_scale_floor, 1.0 - t * t)

        # vol-aware scaling (if order has vol_daily)
        vol = float(order.get("vol_daily") or 0.0)
        if vol > 0 and p.vol_scale_ref > 0:
            # size ∝ (vol_ref / vol)^power
            scale *= (p.vol_scale_ref / max(1e-9, vol)) ** p.vol_scale_power

        # clamp scaled notional
        scaled_notional = notional * scale
        if scaled_notional > p.max_order_notional:
            scale *= p.max_order_notional / max(1e-9, scaled_notional)

        # audit
        self._audit("allow", strat, symbol, notional, scale, extra={"pnl": st.pnl_today, "dd": st.dd_frac})

        return Decision(True, "ok", scale=scale)

    # ------------------- Internals / helpers -------------------

    def _reject(self, strat: str, reason: str, now: float, extra: Dict[str, Any]) -> Decision:
        self._per_strat.setdefault(strat, Stat()).last_reject_ts = now
        self._audit("reject", strat, extra.get("symbol", ""), extra.get("notional", 0.0), 0.0, extra, reason)
        return Decision(False, reason, scale=0.0, notes=extra)

    def _kill(self, strat: str, reason: str, now: float, extra: Dict[str, Any]) -> Decision:
        st = self._per_strat.setdefault(strat, Stat())
        st.last_kill_ts = now
        hset("strategy:enabled", strat, "false")
        self._audit("kill", strat, extra.get("symbol", ""), extra.get("notional", 0.0), 0.0, extra, reason)
        return Decision(False, reason, scale=0.0, notes=extra)

    def _audit(self, action: str, strat: str, symbol: str, notional: float, scale: float, extra: Dict[str, Any], reason: str = "ok") -> None:
        payload = {
            "ts_ms": int(time.time() * 1000),
            "action": action, "strategy": strat, "symbol": symbol, "notional": float(notional),
            "scale": float(scale), "reason": reason, "extra": extra
        }
        try:
            publish_stream(self.policy.audit_stream, payload)
        except Exception:
            pass
        if _R:
            try:
                _R.lpush(f"gov:audit:{strat}", json.dumps(payload))
                _R.ltrim(f"gov:audit:{strat}", 0, 999)
            except Exception:
                pass


# ---------------------- Tiny smoke test --------------------------------------
if __name__ == "__main__":
    gov = Governor()
    gov.register_strategy("alpha.news")
    gov.ingest_mark(strategy="alpha.news", pnl_delta=-20_000, dd_frac=0.03)

    # wide market example
    nbbo = {"bid": 100.0, "ask": 100.30}
    o = {"strategy": "alpha.news", "symbol": "AAPL", "side": "buy", "qty": 200, "price": 100.30, "typ": "market"}
    print("Decision(wide):", gov.check_and_scale(o, nbbo=nbbo))

    # normal market, scaled by DD & vol
    nbbo2 = {"bid": 100.0, "ask": 100.02}
    o2 = {"strategy": "alpha.news", "symbol": "AAPL", "side": "buy", "qty": 1000, "price": 100.02, "vol_daily": 0.03}
    print("Decision(ok):", gov.check_and_scale(o2, nbbo=nbbo2))