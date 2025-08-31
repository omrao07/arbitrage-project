# backend/execution/pov.py
from __future__ import annotations

import math, time, uuid, dataclasses
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple, Any

# ---------- helpers ----------
def now_ms() -> int: return int(time.time() * 1000)
def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def sgn(side: str) -> int: return +1 if str(side).lower() in ("buy","b","long") else -1

# ---------- config & state ----------
@dataclass
class POVConfig:
    symbol: str
    side: str                 # "buy" | "sell"
    target_qty: float         # absolute target
    target_participation: float = 0.10   # 10% of prints volume
    min_clip: float = 1.0                 # min child qty
    max_clip: float = 10_000.0            # max child qty
    max_participation: float = 0.25       # hard cap (safety)
    catchup_factor: float = 0.50          # how aggressively catch up when behind
    price_limit_bps: float = 20.0         # pause if mid moves worse than this from start
    vol_pause_mult: float = 3.0           # pause if last-1min vol > mult * ewma
    ewma_alpha: float = 0.15              # for vol & price ewma
    tick_interval_ms: int = 500           # scheduler heartbeat
    venue: Optional[str] = None
    strategy_tag: str = "exec_pov"
    good_til_ms: Optional[int] = None     # optional absolute deadline (ms since epoch)
    allow_catchup_after_deadline: bool = False

@dataclass
class POVState:
    started_ms: int
    decided_px: float
    ewma_px: float
    ewma_vol_per_s: float = 0.0
    filled_qty: float = 0.0
    printed_qty: float = 0.0      # cumulative market prints seen
    last_trade_ts_ms: int = 0
    paused: bool = False
    done: bool = False
    last_reason: str = ""
    last_child_ms: int = 0

# ---------- types for adapters ----------
PublishOrderFn = Callable[[Dict[str, Any]], None]
GetQuarantineFn = Callable[[str, Optional[str], Optional[str], Optional[str]], bool]
# signature: (symbol, strategy, account, venue) -> bool (True if blocked)

class POVExecutor:
    """
    Drop-in POV execution:
      - call .on_trade(...) for each market trade (price, qty)
      - call .on_fill(...) for your child fills
      - call .tick() from a scheduler (e.g., every cfg.tick_interval_ms)
      - it will compute the desired clip and call publish_order(payload)
    """
    def __init__(
        self,
        cfg: POVConfig,
        *,
        publish_order: PublishOrderFn,
        get_quarantine: Optional[GetQuarantineFn] = None,
        account: Optional[str] = None,
    ):
        self.cfg = cfg
        self.pub = publish_order
        self.qcheck = get_quarantine or (lambda sym, strat, acct, ven: False)
        self.account = account
        self.state = POVState(
            started_ms=now_ms(),
            decided_px=0.0,
            ewma_px=0.0,
        )
        self._id = "pov-" + uuid.uuid4().hex[:8]

    # ---------- event ingestion ----------
    def on_trade(self, price: float, qty: float, ts_ms: Optional[int] = None):
        """
        Feed *market prints* here (tape). Use last trade price & size.
        """
        ts = ts_ms or now_ms()
        if qty <= 0 or price <= 0: 
            return
        self.state.printed_qty += float(qty)
        # EWMA price
        if self.state.ewma_px <= 0:
            self.state.ewma_px = price
            self.state.decided_px = price if self.state.decided_px <= 0 else self.state.decided_px
        else:
            a = self.cfg.ewma_alpha
            self.state.ewma_px = (1-a)*self.state.ewma_px + a*price

        # crude vol per second over last heartbeat: treat each trade as instantaneous
        # Maintain a lightweight EWMA of "trades per second × size" using delta time
        if self.state.last_trade_ts_ms > 0:
            dt_s = max(1e-3, (ts - self.state.last_trade_ts_ms) / 1000.0)
            inst_rate = qty / dt_s
            if self.state.ewma_vol_per_s <= 0:
                self.state.ewma_vol_per_s = inst_rate
            else:
                a = self.cfg.ewma_alpha
                self.state.ewma_vol_per_s = (1-a)*self.state.ewma_vol_per_s + a*inst_rate
        self.state.last_trade_ts_ms = ts

    def on_fill(self, qty: float, price: float):
        """Call when *your child order* fills (partial or full)."""
        if qty == 0:
            return
        self.state.filled_qty += abs(qty)

    # ---------- main brain ----------
    def _target_child_qty(self) -> Tuple[float, str]:
        """
        Decide next child quantity based on:
           - target participation vs observed prints
           - catch-up when behind
           - safety limits (max participation, clip bounds)
        Returns (clip_qty, reason_str).
        """
        cfg, st = self.cfg, self.state
        remaining = max(0.0, cfg.target_qty - st.filled_qty)
        if remaining <= 0:
            return 0.0, "done"

        # projected market volume until next tick (use ewma vol rate × horizon)
        horizon_s = max(0.2, cfg.tick_interval_ms / 1000.0)
        proj_mkt = max(0.0, st.ewma_vol_per_s * horizon_s)

        base = cfg.target_participation * proj_mkt
        # where should we be by participation overall?
        # Desired cumulative fill ≈ cfg.target_participation × printed
        desired_cum = cfg.target_participation * max(st.printed_qty, 1.0)
        shortfall = max(0.0, desired_cum - st.filled_qty)

        # combine: child aims to do base + catch-up on shortfall over the next N slices
        catch_up = cfg.catchup_factor * shortfall
        clip = base + catch_up
        clip = clamp(clip, cfg.min_clip, cfg.max_clip)
        clip = min(clip, remaining)

        # hard cap by instantaneous participation: clip should not exceed max_participation of projected vol
        if proj_mkt > 0:
            cap = cfg.max_participation * proj_mkt
            clip = min(clip, cap)

        return max(0.0, float(clip)), f"base={base:.3f} shortfall={shortfall:.3f} proj={proj_mkt:.3f}"

    def _should_pause(self) -> Tuple[bool, str]:
        cfg, st = self.cfg, self.state
        # quarantine?
        if self.qcheck(cfg.symbol, cfg.strategy_tag, self.account, cfg.venue):
            return True, "quarantined"

        # price drift vs start
        px0 = st.decided_px if st.decided_px > 0 else st.ewma_px
        px  = st.ewma_px
        if px0 > 0 and px > 0:
            drift_bps = abs((px - px0) / px0) * 1e4
            if drift_bps >= cfg.price_limit_bps:
                return True, f"price_drift_{drift_bps:.1f}bps"

        # (optional) vol spike guard: compare instantaneous rate to ewma baseline
        # Here we only have ewma; treat spikes via low projected volume (no need to pause unless you want)
        return False, ""

    def _deadline_reached(self) -> bool:
        if self.cfg.good_til_ms is None:
            return False
        return now_ms() >= self.cfg.good_til_ms

    def tick(self) -> Optional[Dict[str, Any]]:
        """
        Call every cfg.tick_interval_ms. When it decides to place a child,
        it publishes via `publish_order(payload)` and returns the payload.
        Otherwise returns None.
        """
        if self.state.done:
            return None

        # complete?
        if self.state.filled_qty >= self.cfg.target_qty:
            self.state.done = True
            self.state.last_reason = "completed"
            return None

        # deadline handling
        if self._deadline_reached() and not self.cfg.allow_catchup_after_deadline:
            # stop placing more children; leave remainder
            self.state.done = True
            self.state.last_reason = "deadline_reached"
            return None

        # pause logic
        pause, reason = self._should_pause()
        if pause:
            self.state.paused = True
            self.state.last_reason = reason
            return None
        self.state.paused = False

        # compute next child clip
        clip, explain = self._target_child_qty()
        if clip <= 0:
            self.state.last_reason = "noop_" + explain
            return None

        child_qty = round(clip, 6)
        side = self.cfg.side.lower()
        payload = {
            "ts_ms": now_ms(),
            "id": f"{self._id}-{int(time.time()*1000)}",
            "strategy": self.cfg.strategy_tag,
            "symbol": self.cfg.symbol.upper(),
            "side": side,
            "qty": child_qty,
            "typ": "market",          # or "limit" if you supply limit logic externally
            "venue": self.cfg.venue or "",
            "meta": {
                "algo": "POV",
                "explain": explain,
                "target_participation": self.cfg.target_participation,
                "max_participation": self.cfg.max_participation,
            }
        }
        # publish to your OMS / order bus
        try:
            self.pub(payload)
            self.state.last_child_ms = payload["ts_ms"]
            self.state.last_reason = "child_sent"
        except Exception as e:
            self.state.last_reason = f"pub_err:{e}"
            return None
        return payload

    # ---------- status ----------
    def status(self) -> Dict[str, Any]:
        cfg, st = self.cfg, self.state
        remaining = max(0.0, cfg.target_qty - st.filled_qty)
        desired_cum = cfg.target_participation * max(st.printed_qty, 1.0)
        part_now = (st.filled_qty / st.printed_qty) if st.printed_qty > 0 else 0.0
        return {
            "symbol": cfg.symbol.upper(),
            "side": cfg.side.lower(),
            "target_qty": cfg.target_qty,
            "filled_qty": round(st.filled_qty, 6),
            "remaining_qty": round(remaining, 6),
            "printed_qty": round(st.printed_qty, 6),
            "target_participation": cfg.target_participation,
            "live_participation": round(part_now, 6),
            "ewma_px": round(st.ewma_px, 8),
            "ewma_vol_per_s": round(st.ewma_vol_per_s, 6),
            "paused": st.paused,
            "done": st.done,
            "last_reason": st.last_reason,
            "good_til_ms": cfg.good_til_ms,
        }

# ---------- example wiring (optional) ----------
if __name__ == "__main__":
    # Minimal demo with fake tape
    sent = []
    def publish(o): 
        print("[POV] child:", o)
        sent.append(o)

    def quar(sym, strat, acct, ven): 
        return False

    cfg = POVConfig(symbol="AAPL", side="buy", target_qty=10_000, target_participation=0.10, min_clip=50, max_clip=2000)
    pov = POVExecutor(cfg, publish_order=publish, get_quarantine=quar)

    # simulate 60 trades, avg 100 shares each, ~500ms apart
    t0 = now_ms()
    for i in range(60):
        pov.on_trade(price=200.0 + 0.02*i, qty=100, ts_ms=t0 + i*500)
        if i % 1 == 0:
            pov.tick()
        # fake fills: assume 70% of what we send fills immediately in this toy demo
        if sent:
            last = sent[-1]["qty"]
            pov.on_fill(qty=last*0.7, price=200.0 + 0.02*i)

    print("STATUS:", pov.status())