# backend/execution/twap.py
from __future__ import annotations

import math, time, uuid, random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple

# -------- helpers ------------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def sgn(side: str) -> int: return +1 if str(side).lower() in ("buy","b","long") else -1

# -------- config & state -----------------------------------------------------
@dataclass
class TWAPConfig:
    symbol: str
    side: str                 # "buy" | "sell"
    target_qty: float         # absolute quantity
    start_ms: int             # epoch ms to start
    end_ms: int               # epoch ms to finish
    min_clip: float = 1.0
    max_clip: float = 50_000.0
    # How often to consider sending a child
    tick_interval_ms: int = 1000
    # Price guard: pause if EWMA mid drifts worse than this from start (bps)
    price_limit_bps: float = 25.0
    # Randomization of clip size to avoid signaling
    clip_jitter_pct: float = 0.15           # ±15% jitter
    # Catch-up aggressiveness if behind schedule (0..1)
    catchup_factor: float = 0.6
    # Optional deadline wrapper (if end_ms missed, stop vs. force):
    allow_catchup_after_deadline: bool = False
    # Optional venue / tags
    venue: Optional[str] = None
    strategy_tag: str = "exec_twap"

@dataclass
class TWAPState:
    started_ms: int
    ewma_px: float = 0.0
    start_px: float = 0.0
    filled_qty: float = 0.0
    last_child_ms: int = 0
    paused: bool = False
    done: bool = False
    last_reason: str = ""

# -------- user adapter types -------------------------------------------------
PublishOrderFn = Callable[[Dict[str, Any]], None]
GetQuarantineFn = Callable[[str, Optional[str], Optional[str], Optional[str]], bool]
# signature: (symbol, strategy, account, venue) -> True if blocked

# -------- executor -----------------------------------------------------------
class TWAPExecutor:
    """
    Time-slicing execution:
      - Call .on_mid(price) with periodic mid/last prices (optional but recommended for drift guard).
      - Call .on_fill(qty, price) when your child fills.
      - Call .tick() every cfg.tick_interval_ms to let it send the next child order.
    """
    def __init__(
        self,
        cfg: TWAPConfig,
        *,
        publish_order: PublishOrderFn,
        get_quarantine: Optional[GetQuarantineFn] = None,
        account: Optional[str] = None,
    ):
        assert cfg.end_ms > cfg.start_ms, "end_ms must be greater than start_ms"
        assert cfg.target_qty > 0, "target_qty must be > 0"
        self.cfg = cfg
        self.pub = publish_order
        self.qcheck = get_quarantine or (lambda sym, strat, acct, ven: False)
        self.account = account
        self.state = TWAPState(started_ms=now_ms())
        self._id = "twap-" + uuid.uuid4().hex[:8]

    # ---- ingestion ----------------------------------------------------------
    def on_mid(self, price: float):
        """Feed mid/last to track drift and EWMA."""
        if price <= 0: return
        if self.state.ewma_px <= 0:
            self.state.ewma_px = price
            self.state.start_px = price
        else:
            # light EWMA
            a = 0.15
            self.state.ewma_px = (1-a) * self.state.ewma_px + a * price

    def on_fill(self, qty: float, price: float):
        """Call when your child order fills (partial or full)."""
        if qty == 0: return
        self.state.filled_qty += abs(qty)

    # ---- internals ----------------------------------------------------------
    def _schedule_ratio(self, t_ms: int) -> float:
        """Return planned completed ratio (0..1) given current clock."""
        if t_ms <= self.cfg.start_ms: return 0.0
        if t_ms >= self.cfg.end_ms: return 1.0
        return (t_ms - self.cfg.start_ms) / float(self.cfg.end_ms - self.cfg.start_ms)

    def _planned_qty_by_now(self, t_ms: int) -> float:
        return self._schedule_ratio(t_ms) * self.cfg.target_qty

    def _remaining_time_ms(self, t_ms: int) -> int:
        return max(0, self.cfg.end_ms - t_ms)

    def _should_pause(self) -> Tuple[bool, str]:
        # quarantine?
        if self.qcheck(self.cfg.symbol, self.cfg.strategy_tag, self.account, self.cfg.venue):
            return True, "quarantined"
        # price drift guard
        px0 = self.state.start_px if self.state.start_px > 0 else self.state.ewma_px
        px  = self.state.ewma_px
        if px0 > 0 and px > 0:
            drift_bps = abs((px - px0) / px0) * 1e4
            if drift_bps >= self.cfg.price_limit_bps:
                return True, f"price_drift_{drift_bps:.1f}bps"
        return False, ""

    def _child_size(self, t_ms: int) -> Tuple[float, str]:
        """
        Decide next child clip based on:
         - straight TWAP quota since last send,
         - catch-up if behind schedule,
         - clip bounds and jitter.
        """
        cfg, st = self.cfg, self.state

        # Remaining parent quantity
        remaining = max(0.0, cfg.target_qty - st.filled_qty)
        if remaining <= 0:
            return 0.0, "done"

        # Where should we be?
        plan_cum = self._planned_qty_by_now(t_ms)
        behind = max(0.0, plan_cum - st.filled_qty)

        # Baseline per-tick TWAP: spread remaining evenly across remaining ticks
        ticks_left = max(1, math.ceil(self._remaining_time_ms(t_ms) / max(1, cfg.tick_interval_ms)))
        base = remaining / float(ticks_left)

        # Catch-up share of behind amount
        catchup = self.cfg.catchup_factor * behind

        clip = base + catchup
        # Jitter to reduce footprint predictability
        if cfg.clip_jitter_pct > 0:
            j = 1.0 + random.uniform(-cfg.clip_jitter_pct, cfg.clip_jitter_pct)
            clip *= j

        # Safety clamps
        clip = clamp(clip, cfg.min_clip, cfg.max_clip)
        clip = min(clip, remaining)

        return float(clip), f"base={base:.3f} behind={behind:.3f} ticks_left={ticks_left}"

    def _deadline_reached(self, t_ms: int) -> bool:
        return t_ms >= self.cfg.end_ms

    # ---- public brain -------------------------------------------------------
    def tick(self) -> Optional[Dict[str, Any]]:
        """
        Call every cfg.tick_interval_ms. If a child should be placed, publishes
        via `publish_order(payload)` and returns the payload; else returns None.
        """
        if self.state.done:
            return None

        t = now_ms()

        # Before start: do nothing
        if t < self.cfg.start_ms:
            self.state.last_reason = "pre_window"
            return None

        # Finished?
        if self.state.filled_qty >= self.cfg.target_qty:
            self.state.done = True
            self.state.last_reason = "completed"
            return None

        # Deadline reached?
        if self._deadline_reached(t) and not self.cfg.allow_catchup_after_deadline:
            self.state.done = True
            self.state.last_reason = "deadline_stop"
            return None

        # Pause checks
        pause, reason = self._should_pause()
        if pause:
            self.state.paused = True
            self.state.last_reason = reason
            return None
        self.state.paused = False

        # Decide next child
        clip, explain = self._child_size(t)
        if clip <= 0:
            self.state.last_reason = "noop_" + explain
            return None

        payload = {
            "ts_ms": t,
            "id": f"{self._id}-{t}",
            "strategy": self.cfg.strategy_tag,
            "symbol": self.cfg.symbol.upper(),
            "side": self.cfg.side.lower(),
            "qty": round(clip, 6),
            "typ": "market",                # change to "limit" and add limit_price if you peg
            "venue": self.cfg.venue or "",
            "meta": {
                "algo": "TWAP",
                "explain": explain,
                "start_ms": self.cfg.start_ms,
                "end_ms": self.cfg.end_ms,
            }
        }

        try:
            self.pub(payload)
            self.state.last_child_ms = t
            self.state.last_reason = "child_sent"
        except Exception as e:
            self.state.last_reason = f"pub_err:{e}"
            return None
        return payload

    def status(self) -> Dict[str, Any]:
        t = now_ms()
        return {
            "symbol": self.cfg.symbol.upper(),
            "side": self.cfg.side.lower(),
            "target_qty": self.cfg.target_qty,
            "filled_qty": round(self.state.filled_qty, 6),
            "remaining_qty": round(max(0.0, self.cfg.target_qty - self.state.filled_qty), 6),
            "planned_qty_by_now": round(self._planned_qty_by_now(t), 6),
            "progress_ratio": round(self._schedule_ratio(t), 6),
            "paused": self.state.paused,
            "done": self.state.done,
            "last_reason": self.state.last_reason,
            "ewma_px": round(self.state.ewma_px, 8),
            "start_px": round(self.state.start_px, 8),
            "window": {"start_ms": self.cfg.start_ms, "end_ms": self.cfg.end_ms},
        }

# -------- example wiring (optional) -----------------------------------------
if __name__ == "__main__":
    sent = []
    def publish(o):
        print("[TWAP] child:", o)
        sent.append(o)

    def quarantined(sym, strat, acct, ven):
        return False

    now = now_ms()
    cfg = TWAPConfig(
        symbol="AAPL",
        side="buy",
        target_qty=12_000,
        start_ms=now + 2000,
        end_ms=now + 62_000,
        tick_interval_ms=1000,
        min_clip=200, max_clip=2000,
        price_limit_bps=30.0,
        clip_jitter_pct=0.1,
    )
    tw = TWAPExecutor(cfg, publish_order=publish, get_quarantine=quarantined)

    # simulate
    for i in range(80):
        # feed a slowly drifting mid
        tw.on_mid(200.0 + 0.01*i)
        time.sleep(0.05)
        tw.tick()
        # fake fill ~= 70% of last child
        if sent:
            tw.on_fill(qty=sent[-1]["qty"] * 0.7, price=200.0 + 0.01*i)

    print("STATUS:", tw.status())