# backend/engine/strategies/tail_hedger.py
from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset

# ---------- Redis ----------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

KEY_NLV = os.getenv("PORTFOLIO_NLV_KEY", "portfolio:nlv")  # HGET <KEY_NLV> value -> float


# ---------- Config ----------
@dataclass
class TailHedgeConfig:
    # Underlying you want to protect (driver signal)
    underlying: str = "SPY"

    # What to TRADE for protection (cash proxy). Examples:
    #  - "VIXY", "VXX" (long vol ETNs/ETFs)
    #  - "SH" (ProShares Short S&P 500), "SDS" (2x short)
    #  - Or just short the underlying (set proxy_asset=None to short underlying)
    proxy_asset: Optional[str] = "VIXY"

    # Signal model params (EWMA + drawdown + momentum + gap)
    ewma_fast: float = 0.2
    ewma_slow: float = 0.05
    vol_alpha: float = 0.08         # EWMA of squared returns
    dd_window: int = 20 * 60 * 1000 # (ms) trailing peak window ~20 min on tick stream (adjust as needed)

    # Risk score weights (0..1; they will be renormalized)
    w_momo: float = 0.35            # downside momentum (fast<slow)
    w_vol: float  = 0.35            # realized vol spike
    w_dd: float   = 0.20            # drawdown from recent peak
    w_gap: float  = 0.10            # single-bar large negative return

    # Triggers / hysteresis
    enter_threshold: float = 0.55   # start (or add) hedge when risk_score >= this
    peel_threshold: float  = 0.30   # peel/scale-down when risk_score <= this
    max_hedge_frac: float  = 0.40   # cap: ≤ 40% of NAV in hedge notional (proxy or short)
    step_frac: float       = 0.10   # each action adjusts hedge toward target by this fraction

    # Sizing
    beta_proxy: float = +3.0        # rough beta of proxy vs underlying returns (+ for VIXY ~ +3 to +5 on selloffs)
    beta_underlying_short: float = -1.0  # if no proxy_asset and we short the underlying
    notional_clip: float = 250_000.0      # per action cap
    default_qty: float = 1.0

    # Execution / safety
    cooldown_ms: int = 5_000
    hard_kill: bool = False
    venues: tuple[str, ...] = ("IBKR", "PAPER")


class TailHedger(Strategy):
    """
    Automatic tail hedge:
      - Computes a blended risk score from downside momentum, realized vol spike,
        recent drawdown, and single-bar gap.
      - When score high → increase protection (buy VIXY/VXX/SDS/SH OR short underlying).
      - When score low → peel protection.
      - Sizes from portfolio NAV with beta adjustment and caps.

    Tick tolerance:
      {symbol|s, price|p|mid} or {bid, ask}; we subscribe to the underlying and (if proxy_asset) its price too.
    """

    def __init__(self, name="policy_tail_hedger", region=None, cfg: Optional[TailHedgeConfig] = None):
        cfg = cfg or TailHedgeConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg
        self.und = cfg.underlying.upper()
        self.proxy = cfg.proxy_asset.upper() if cfg.proxy_asset else None

        # State
        self._last_px: float = 0.0
        self._ewma_fast: float = 0.0
        self._ewma_slow: float = 0.0
        self._vol_ewma: float = 1e-8
        self._peak_px: float = 0.0
        self._peak_ts: int = 0
        self._last_ret: float = 0.0

        # Hedge state (our local target notionals; OMS/positions are source of truth)
        self._target_hedge_notional: float = 0.0
        self._last_act_ms: int = 0

    # -------- lifecycle --------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["risk", "tail_hedge", "overlay"],
            "region": self.ctx.region or "US",
            "underlying": self.und,
            "proxy": self.proxy or "SHORT_UNDERLYING",
            "notes": "Automatic tail hedge overlay sized by NAV; enters on risk spike and peels on calm."
        })

    # -------- helpers --------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _safe_float(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d

    def _nlv(self) -> float:
        try:
            v = _r.hget(KEY_NLV, "value")
            return float(v) if v is not None else 0.0 # type: ignore
        except Exception:
            return 0.0

    def _update_filters(self, px: float, now: int) -> None:
        if self._ewma_fast == 0.0:
            self._ewma_fast = px
            self._ewma_slow = px
            self._peak_px = px
            self._peak_ts = now
            self._last_px = px
            return

        a_f = max(1e-4, min(0.99, self.cfg.ewma_fast))
        a_s = max(1e-4, min(0.99, self.cfg.ewma_slow))
        self._ewma_fast = (1 - a_f) * self._ewma_fast + a_f * px
        self._ewma_slow = (1 - a_s) * self._ewma_slow + a_s * px

        # simple return
        ret = (px / max(self._last_px, 1e-12)) - 1.0 if self._last_px > 0 else 0.0
        self._last_ret = ret

        # vol ewma on returns
        a_v = max(1e-5, min(0.5, self.cfg.vol_alpha))
        self._vol_ewma = (1 - a_v) * self._vol_ewma + a_v * (ret * ret)

        # rolling peak for drawdown (time-limited)
        if px > self._peak_px or (now - self._peak_ts) > self.cfg.dd_window:
            self._peak_px = px
            self._peak_ts = now

        self._last_px = px

    def _risk_score(self) -> float:
        """
        Blend of:
          - Momentum (fast < slow → bearish)
          - Realized vol (vol_ewma)
          - Drawdown from recent peak
          - Gap (negative last return)
        Returns 0..1
        """
        if self._ewma_slow <= 0:
            return 0.0

        # normalize components to 0..1
        momo = max(0.0, min(1.0, (self._ewma_slow - self._ewma_fast) / self._ewma_slow * 50.0))  # ~2% gap → 1.0
        vol = max(0.0, min(1.0, math.sqrt(self._vol_ewma) * 20.0))  # ~5% daily vol → ~1.0 (tunable)
        dd = 0.0
        if self._peak_px > 0:
            dd = max(0.0, min(1.0, (self._peak_px - self._last_px) / self._peak_px * 10.0))  # 10% dd → 1.0
        gap = max(0.0, min(1.0, -self._last_ret * 50.0))  # -2% bar → 1.0

        # weights renormalized
        wsum = max(1e-9, self.cfg.w_momo + self.cfg.w_vol + self.cfg.w_dd + self.cfg.w_gap)
        score = (
            self.cfg.w_momo * momo +
            self.cfg.w_vol  * vol  +
            self.cfg.w_dd   * dd   +
            self.cfg.w_gap  * gap
        ) / wsum

        return max(0.0, min(1.0, score))

    def _proxy_price(self, tick: Dict[str, Any]) -> Optional[float]:
        if not self.proxy:
            return None
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym != self.proxy:
            return None
        px = tick.get("price") or tick.get("p") or tick.get("mid")
        if px is None:
            bid = tick.get("bid"); ask = tick.get("ask")
            if bid and ask:
                try:
                    px = 0.5 * (float(bid) + float(ask))
                except Exception:
                    px = None
        try:
            return float(px) if px is not None else None
        except Exception:
            return None

    # -------- execution sizing --------
    def _desired_hedge_notional(self, nav: float, score: float) -> float:
        """
        Hedge target notional = NAV * max_hedge_frac * f(score)
        with a smooth ramp starting at enter_threshold and saturating near 1.0
        """
        if score <= self.cfg.peel_threshold:
            return 0.0
        # smooth ramp
        lo, hi = self.cfg.enter_threshold, 1.00
        x = 0.0 if score <= lo else min(1.0, (score - lo) / max(1e-9, hi - lo))
        target_frac = x * self.cfg.max_hedge_frac
        return max(0.0, min(self.cfg.max_hedge_frac, target_frac)) * max(0.0, nav)

    def _act(self, side: str, symbol: str, qty: float, mark: float, reason: str, extra: Dict[str, Any]) -> None:
        self.order(symbol, side=side, qty=qty, order_type="market", mark_price=mark, extra=extra)

    # -------- main --------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym not in (self.und, self.proxy or ""):
            return

        # update prices/filters only on underlying ticks
        if sym == self.und:
            # normalize price
            px = tick.get("price") or tick.get("p") or tick.get("mid")
            if px is None:
                bid = tick.get("bid"); ask = tick.get("ask")
                if bid and ask and float(bid) > 0 and float(ask) > 0:
                    px = 0.5 * (float(bid) + float(ask))
            if px is None:
                return
            try:
                px = float(px)
            except Exception:
                return
            if px <= 0:
                return

            now = self._now_ms()
            self._update_filters(px, now)

            # compute risk score and emit
            score = self._risk_score()
            self.emit_signal(score * 2.0 - 1.0)  # map 0..1 → -1..+1 for visual consistency

            # cooldown
            if now - self._last_act_ms < self.cfg.cooldown_ms:
                return

            nav = self._nlv()
            if nav <= 0:
                return

            # target hedge notional and step toward it
            target = self._desired_hedge_notional(nav, score)
            delta = target - self._target_hedge_notional
            step = self.cfg.step_frac * nav
            step = max(-abs(step), min(abs(step), delta))  # clamp step
            if abs(step) < 1e-6:
                return

            # Decide instrument & side
            if self.proxy:
                # Buy proxy when risk up (positive notional), sell when peeling (negative step)
                proxy_px = None  # we’ll allow market mark from underlying if proxy tick not seen
                proxy_px = proxy_px or px  # conservative
                qty = max(1.0, min(abs(step), self.cfg.notional_clip) / max(proxy_px, 1e-9))
                side = "buy" if step > 0 else "sell"
                self._act(side, self.proxy, qty, proxy_px,
                          reason="tail_hedge_adjust",
                          extra={"score": score, "target": target, "step": step, "nav": nav})
            else:
                # No proxy: short/cover the underlying
                qty = max(1.0, min(abs(step), self.cfg.notional_clip) / max(px, 1e-9))
                side = "sell" if step > 0 else "buy"
                self._act(side, self.und, qty, px,
                          reason="tail_hedge_adjust_short_udl",
                          extra={"score": score, "target": target, "step": step, "nav": nav})

            self._target_hedge_notional += step
            self._last_act_ms = now

        else:
            # proxy price tick (optional; not strictly needed for actions)
            _ = self._proxy_price(tick)  # kept for future enhancements
            return


# ---------------- optional runner ----------------
if __name__ == "__main__":
    """
    Example:
      export REDIS_HOST=localhost REDIS_PORT=6379
      HSET portfolio:nlv value 1_000_000
      python -m backend.engine.strategies.tail_hedger
    Attach via your runner: strat.run(stream="ticks.equities.us")
    """
    strat = TailHedger()
    # strat.run(stream="ticks.equities.us")