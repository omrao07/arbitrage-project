# backend/execution/vwap.py
from __future__ import annotations

import math, time, uuid, random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple, List

# ---------- helpers ----------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def sgn(side: str) -> int: return +1 if str(side).lower() in ("buy","b","long") else -1

# ---------- config & state ---------------------------------------------------
@dataclass
class VWAPConfig:
    symbol: str
    side: str                  # "buy" | "sell"
    target_qty: float          # absolute quantity
    start_ms: int              # execution window start (epoch ms)
    end_ms: int                # execution window end (epoch ms)
    mode: str = "adaptive"     # "historical" | "adaptive"
    # Historical mode: cumulative volume curve points (time_frac, cum_frac), both in [0,1]
    # e.g., [(0.00,0.00),(0.10,0.18),(0.25,0.35),(0.50,0.62),(0.75,0.83),(1.00,1.00)]
    ucurve: Optional[List[Tuple[float, float]]] = None

    tick_interval_ms: int = 1000
    min_clip: float = 1.0
    max_clip: float = 50_000.0
    clip_jitter_pct: float = 0.15        # ±15% randomization
    catchup_factor: float = 0.6          # 0..1 how aggressively close schedule gap

    # Guards
    price_limit_bps: float = 25.0        # pause when EWMA mid drifts worse than this from start
    allow_catchup_after_deadline: bool = False

    # Optional tags
    venue: Optional[str] = None
    strategy_tag: str = "exec_vwap"

@dataclass
class VWAPState:
    started_ms: int
    ewma_px: float = 0.0
    start_px: float = 0.0
    filled_qty: float = 0.0

    # tape tracking (observed)
    printed_qty_total: float = 0.0         # cumulative market prints within window
    last_trade_ms: int = 0

    last_child_ms: int = 0
    paused: bool = False
    done: bool = False
    last_reason: str = ""

# ---------- user adapter types ----------------------------------------------
PublishOrderFn = Callable[[Dict[str, Any]], None]
GetQuarantineFn = Callable[[str, Optional[str], Optional[str], Optional[str]], bool]
# signature: (symbol, strategy, account, venue) -> True if blocked

# ---------- executor ---------------------------------------------------------
class VWAPExecutor:
    """
    VWAP execution:
      - Call .on_trade(price, qty, ts_ms) for each **market print** (or per-second aggregate).
      - (Optional) Call .on_mid(price) periodically to feed price EWMA for drift guard.
      - Call .on_fill(qty, price) when your **child fills** occur.
      - Call .tick() every cfg.tick_interval_ms; it will send a child via your publish callable.
    """
    def __init__(
        self,
        cfg: VWAPConfig,
        *,
        publish_order: PublishOrderFn,
        get_quarantine: Optional[GetQuarantineFn] = None,
        account: Optional[str] = None,
    ):
        assert cfg.end_ms > cfg.start_ms, "end_ms must be > start_ms"
        assert cfg.target_qty > 0, "target_qty must be > 0"
        if cfg.mode == "historical":
            assert cfg.ucurve and len(cfg.ucurve) >= 2, "historical mode requires ucurve points"
            # ensure sorted and clamped
            curve = sorted((max(0.0,min(1.0,t)), max(0.0,min(1.0,c))) for t,c in cfg.ucurve)
            if curve[0][0] > 0.0: curve.insert(0,(0.0,0.0))
            if curve[-1][0] < 1.0: curve.append((1.0,1.0))
            self.ucurve = curve
        else:
            self.ucurve = None

        self.cfg = cfg
        self.pub = publish_order
        self.qcheck = get_quarantine or (lambda sym, strat, acct, ven: False)
        self.account = account
        self.state = VWAPState(started_ms=now_ms())
        self._id = "vwap-" + uuid.uuid4().hex[:8]

    # ---- ingestion ----------------------------------------------------------
    def on_trade(self, price: float, qty: float, ts_ms: Optional[int] = None):
        """Feed market prints volume (best with per-trade or per-second aggregate)."""
        ts = ts_ms or now_ms()
        if ts < self.cfg.start_ms or ts > self.cfg.end_ms:
            return
        if qty <= 0 or price <= 0:
            return
        self.state.printed_qty_total += float(qty)
        self.state.last_trade_ms = ts
        # seed EWMA from tape if not using on_mid
        if self.state.ewma_px <= 0:
            self.state.ewma_px = price
            self.state.start_px = price if self.state.start_px <= 0 else self.state.start_px
        else:
            a = 0.15
            self.state.ewma_px = (1-a)*self.state.ewma_px + a*price

    def on_mid(self, price: float):
        """(Optional) Feed mid/last to track drift and EWMA even without trades."""
        if price <= 0: return
        if self.state.ewma_px <= 0:
            self.state.ewma_px = price
            self.state.start_px = price
        else:
            a = 0.15
            self.state.ewma_px = (1-a)*self.state.ewma_px + a*price

    def on_fill(self, qty: float, price: float):
        """Call when your child order fills (partial or full)."""
        if qty == 0: return
        self.state.filled_qty += abs(qty)

    # ---- planning -----------------------------------------------------------
    def _time_frac(self, t_ms: int) -> float:
        return 0.0 if t_ms <= self.cfg.start_ms else (
            1.0 if t_ms >= self.cfg.end_ms else (t_ms - self.cfg.start_ms) / float(self.cfg.end_ms - self.cfg.start_ms)
        )

    def _hist_cum_frac(self, time_frac: float) -> float:
        """Interpolate cumulative volume fraction from u-curve at time_frac [0..1]."""
        curve = self.ucurve or [(0.0,0.0),(1.0,1.0)]
        # binary/linear search
        for i in range(1, len(curve)):
            t0,c0 = curve[i-1]; t1,c1 = curve[i]
            if time_frac <= t1:
                if t1 == t0: return c1
                w = (time_frac - t0) / (t1 - t0)
                return c0 + w*(c1 - c0)
        return 1.0

    def _planned_cum_qty(self, t_ms: int) -> float:
        """
        Planned cumulative qty by now:
         - historical: Q * cum_frac_from_ucurve(time)
         - adaptive:   target_participation ~= printed / expected_total → we don't know expected;
                       simplest: set plan proportional to *observed* printed vs observed end → use time_frac as guard.
        """
        Q = self.cfg.target_qty
        if self.cfg.mode == "historical":
            frac = self._hist_cum_frac(self._time_frac(t_ms))
            return frac * Q
        else:
            # adaptive: mirror observed cumulative **volume share**.
            # Without absolute market volume target, aim to distribute Q uniformly over time
            # but scale with observed prints to avoid over-trading during lulls:
            # plan = Q * time_frac * α + Q * (printed/ max(printed_at_end, tiny)) * (1-α)
            # Since printed_at_end unknown mid-run, we fallback to linear time with mild weight from prints.
            tf = self._time_frac(t_ms)
            # give 70% weight to linear time, 30% to observed liquidity proxy
            liq = 1.0 if self.state.printed_qty_total <= 0 else min(1.0, (self.state.printed_qty_total / (self.state.printed_qty_total + 1e-9)))
            frac = 0.7*tf + 0.3*liq
            return frac * Q

    def _remaining_ticks(self, t_ms: int) -> int:
        left = max(0, self.cfg.end_ms - t_ms)
        return max(1, math.ceil(left / max(1, self.cfg.tick_interval_ms)))

    def _child_size(self, t_ms: int) -> Tuple[float, str]:
        st, cfg = self.state, self.cfg
        remaining = max(0.0, cfg.target_qty - st.filled_qty)
        if remaining <= 0:
            return 0.0, "done"

        # schedule tracking
        plan_cum = self._planned_cum_qty(t_ms)
        behind = max(0.0, plan_cum - st.filled_qty)

        # baseline TWAP-ish split of remaining over remaining ticks
        ticks_left = self._remaining_ticks(t_ms)
        base = remaining / float(ticks_left)

        # add catch-up share of the schedule gap
        catchup = cfg.catchup_factor * behind

        clip = base + catchup

        # randomization
        if cfg.clip_jitter_pct > 0:
            clip *= 1.0 + random.uniform(-cfg.clip_jitter_pct, cfg.clip_jitter_pct)

        clip = clamp(clip, cfg.min_clip, cfg.max_clip)
        clip = min(clip, remaining)

        return float(clip), f"base={base:.3f} behind={behind:.3f} ticks_left={ticks_left}"

    # ---- guards & deadline ---------------------------------------------------
    def _should_pause(self) -> Tuple[bool, str]:
        # quarantine gate
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

    def _deadline_reached(self, t_ms: int) -> bool:
        return t_ms >= self.cfg.end_ms

    # ---- main loop ----------------------------------------------------------
    def tick(self) -> Optional[Dict[str, Any]]:
        """
        Call every cfg.tick_interval_ms. Emits a child order payload via publish_order()
        when appropriate and returns that payload; else returns None.
        """
        if self.state.done: return None

        t = now_ms()
        if t < self.cfg.start_ms:
            self.state.last_reason = "pre_window"
            return None

        if self.state.filled_qty >= self.cfg.target_qty:
            self.state.done = True
            self.state.last_reason = "completed"
            return None

        if self._deadline_reached(t) and not self.cfg.allow_catchup_after_deadline:
            self.state.done = True
            self.state.last_reason = "deadline_stop"
            return None

        pause, reason = self._should_pause()
        if pause:
            self.state.paused = True
            self.state.last_reason = reason
            return None
        self.state.paused = False

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
            "typ": "market",           # change to "limit" and add limit_price if you peg to mid/tick
            "venue": self.cfg.venue or "",
            "meta": {
                "algo": "VWAP",
                "mode": self.cfg.mode,
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
        planned = self._planned_cum_qty(t)
        return {
            "symbol": self.cfg.symbol.upper(),
            "side": self.cfg.side.lower(),
            "target_qty": self.cfg.target_qty,
            "filled_qty": round(self.state.filled_qty, 6),
            "remaining_qty": round(max(0.0, self.cfg.target_qty - self.state.filled_qty), 6),
            "printed_qty_total": round(self.state.printed_qty_total, 6),
            "planned_cum_qty": round(planned, 6),
            "behind_qty": round(max(0.0, planned - self.state.filled_qty), 6),
            "paused": self.state.paused,
            "done": self.state.done,
            "last_reason": self.state.last_reason,
            "ewma_px": round(self.state.ewma_px, 8),
            "start_px": round(self.state.start_px, 8),
            "window": {"start_ms": self.cfg.start_ms, "end_ms": self.cfg.end_ms},
            "mode": self.cfg.mode,
        }

# ---------- example wiring (optional) ----------------------------------------
if __name__ == "__main__":
    sent = []
    def publish(o):
        print("[VWAP] child:", o)
        sent.append(o)

    def quarantined(sym, strat, acct, ven):
        return False

    now = now_ms()
    # historical example u-curve (sparse & illustrative only)
    curve = [(0.00,0.00),(0.10,0.18),(0.25,0.35),(0.50,0.62),(0.75,0.83),(1.00,1.00)]
    cfg = VWAPConfig(
        symbol="AAPL",
        side="buy",
        target_qty=15_000,
        start_ms=now + 2000,
        end_ms=now + 62_000,
        mode="historical",
        ucurve=curve,
        min_clip=200, max_clip=2500, clip_jitter_pct=0.10
    )
    ex = VWAPExecutor(cfg, publish_order=publish, get_quarantine=quarantined)

    # simulate ~per-second tape and heartbeats
    for i in range(80):
        px = 200.0 + 0.02*i
        ex.on_mid(px)
        # pretend 120–180 shares printed this second
        ex.on_trade(px, qty=120 + (i % 60), ts_ms=now + 2000 + i*1000)
        time.sleep(0.03)
        ex.tick()
        # fake ~70% of last child fills immediately
        if sent:
            ex.on_fill(qty=sent[-1]["qty"] * 0.7, price=px)

    print("STATUS:", ex.status())