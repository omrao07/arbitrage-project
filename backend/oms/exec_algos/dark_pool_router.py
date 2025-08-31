# backend/engine/strategies/dark_pool_router.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


# -------------------- Config --------------------

@dataclass
class DarkPoolRouterConfig:
    symbol: str = "SPY"
    side: str = "buy"                      # "buy" | "sell"
    parent_qty: float = 50_000.0

    # Venues
    dark_venues: tuple[str, ...] = ("DBKX", "MSPL", "BATS-D", "LQDX")
    lit_venues:  tuple[str, ...] = ("NYSE", "NASDAQ", "ARCA", "BATS")

    # Child order pacing / sizing
    child_interval_ms: int = 1200
    min_child_qty: float = 300.0
    max_child_qty: float = 5_000.0
    min_exec_size: float = 200.0          # minimum execution size constraint for dark IOC
    batch_ratio: float = 0.15             # portion of remaining qty per child

    # Pricing
    midpeg_improve_bps: float = 0.1       # post at mid -/+ tiny improve vs touch for lit
    fade_on_impact_bps: float = 1.0       # avoid crossing if slippage exceeds this
    reject_on_wide_spread_bps: float = 8  # skip dark posting if spread very wide

    # Learning / exploration (per-venue UCB1)
    ucb_explore_c: float = 1.2            # exploration strength
    decay: float = 0.995                  # EWMA decay for venue statistics

    # Toxicity / Safety
    pause_on_vol_z: float = 4.0           # pause when short-horizon vol spikes
    hard_kill: bool = False
    notional_cap: float = 2_000_000.0
    kill_pct_complete: float = 1.02       # stop after slight overshoot

    # OMS flags (your adapter can map these)
    tag_dark_ioc: str = "DARK_IOC"
    tag_mid_peg: str = "MID_PEG"


# -------------------- Strategy --------------------

class DarkPoolRouter(Strategy):
    """
    Adaptive dark-pool router:
      • Learns fill quality per dark venue with a simple UCB1 bandit.
      • Posts **IOC dark** first (min size), then **lit mid-peg/near-touch** if toxicity is high or no dark fill.
      • Backs off when spreads widen or short-horizon volatility (toxicity) spikes.
    Expected market data tick:
      {symbol|s, bid, ask, last|price|mid, size?}
    OMS should call `on_fill(qty, price, venue=None)` to feed learning.
    """

    def __init__(self, name="exec_dark_pool_router", region=None, cfg: Optional[DarkPoolRouterConfig] = None):
        cfg = cfg or DarkPoolRouterConfig()
        super().__init__(name=name, region=region, default_qty=cfg.min_child_qty)
        self.cfg = cfg
        self.sym = cfg.symbol.upper()
        self.buy = (cfg.side.lower() == "buy")

        # market state
        self.bid = 0.0
        self.ask = 0.0
        self.mid = 0.0
        self.last_px = 0.0
        self.ret2_ewma = 0.0     # short-horizon variance proxy
        self.spread_ewma = 1.5

        # progress
        self.filled_qty = 0.0
        self.last_child_ms = 0
        self.dead = False

        # learning store: per venue -> {n, reward, last_ts}
        self.venues: Dict[str, Dict[str, float]] = {
            v: {"n": 1e-6, "reward": 0.0, "last": 0.0} for v in (cfg.dark_venues + cfg.lit_venues)
        }

    # ---------- lifecycle ----------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["execution", "router", "dark_pool"],
            "underlying": self.sym,
            "notes": "UCB1 dark router with volatility/spread toxicity guard and lit fallback."
        })

    # ---------- utils ----------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _remaining(self) -> float:
        return max(0.0, self.cfg.parent_qty - self.filled_qty)

    def _child_qty(self) -> float:
        rem = self._remaining()
        if rem <= 0:
            return 0.0
        qty = max(self.cfg.min_child_qty, min(self.cfg.max_child_qty, rem * self.cfg.batch_ratio))
        # notional cap
        if self.mid > 0 and qty * self.mid > self.cfg.notional_cap:
            qty = max(self.cfg.min_child_qty, self.cfg.notional_cap / self.mid)
        return qty

    def _update_md(self, t: Dict[str, Any]) -> None:
        b, a = t.get("bid"), t.get("ask")
        p = t.get("last") or t.get("price") or t.get("p") or t.get("mid")
        if b is not None and a is not None:
            try:
                b = float(b); a = float(a)
                if b > 0 and a > 0:
                    self.bid, self.ask = b, a
                    self.mid = 0.5 * (b + a)
                    spread_bps = (a - b) / max(1e-9, self.mid) * 1e4
                    self.spread_ewma = 0.95 * self.spread_ewma + 0.05 * spread_bps
                    if self.last_px > 0:
                        r = (self.mid / self.last_px) - 1.0
                        self.ret2_ewma = 0.97 * self.ret2_ewma + 0.03 * (r * r)
                    self.last_px = self.mid
            except Exception:
                pass
        elif p is not None:
            try:
                px = float(p)
                if px > 0:
                    self.mid = px if self.mid == 0 else self.mid  # keep first seen as baseline
            except Exception:
                pass

    # ---------- learning ----------
    def _ucb_score(self, venue: str, t: float) -> float:
        v = self.venues[venue]
        n = max(1e-6, v["n"])
        r = v["reward"]
        total_n = sum(max(1e-6, self.venues[x]["n"]) for x in self.venues)
        explore = self.cfg.ucb_explore_c * math.sqrt(math.log(max(1.0, total_n)) / n)
        return (r / n) + explore

    def _choose_dark(self, now: int) -> Optional[str]:
        # pick highest UCB among dark venues
        if not self.cfg.dark_venues:
            return None
        best, score = None, -1e9
        for v in self.cfg.dark_venues:
            s = self._ucb_score(v, now)
            if s > score:
                best, score = v, s
        return best

    def _reward_proxy(self, fill_px: float, venue: str) -> float:
        """
        Positive reward if we bettered the touch for our side; small penalty otherwise.
        Can be replaced by TCA alpha/slippage delta if you have it.
        """
        if self.bid <= 0 or self.ask <= 0 or fill_px <= 0:
            return 0.0
        if self.buy:
            edge_bps = (min(fill_px, self.mid) - self.mid) / max(1e-9, self.mid) * 1e4  # negative is good (below mid)
        else:
            edge_bps = (self.mid - max(fill_px, self.mid)) / max(1e-9, self.mid) * 1e4  # negative is good (above mid)
        # invert: more positive reward when negative edge_bps (price better than mid)
        return -edge_bps

    def on_fill(self, qty: float, price: float, venue: Optional[str] = None) -> None:
        """
        OMS should invoke this with the venue string in `venue` (if known).
        """
        self.filled_qty += max(0.0, qty)
        if not venue:
            return
        v = self.venues.get(venue)
        if not v:
            return
        # EWMA decay then add reward
        v["reward"] *= self.cfg.decay
        v["n"] = v.get("n", 0.0) * self.cfg.decay + 1.0
        v["reward"] += self._reward_proxy(price, venue)
        v["last"] = self._now_ms()

    # ---------- routing ----------
    def _post_dark_ioc(self, venue: str, qty: float) -> None:
        """
        Post a DARK IOC child with min-exec-size. Your OMS maps extras to broker flags.
        """
        if qty < self.cfg.min_exec_size:
            qty = self.cfg.min_exec_size
        extra = {
            "reason": "dark_router",
            "venue": venue,
            "exec": self.cfg.tag_dark_ioc,
            "min_exec_size": self.cfg.min_exec_size,
        }
        side = "buy" if self.buy else "sell"
        # price = mid reference; dark pools typically peg → mark only
        self.order(self.sym, side, qty=qty, order_type="market", mark_price=self.mid, extra=extra)

    def _post_lit_midpeg(self, qty: float) -> None:
        """
        Post a near-mid peg on lit (improve from mid toward our favor by tiny amount).
        """
        if self.mid <= 0:
            return
        improve = self.mid * (self.cfg.midpeg_improve_bps / 1e4)
        side = "buy" if self.buy else "sell"
        limit = self.mid - improve if self.buy else self.mid + improve
        self.order(self.sym, side, qty=qty, order_type="limit", limit_price=limit,
                   extra={"reason": "dark_router_fallback", "exec": self.cfg.tag_mid_peg})

    # ---------- main ----------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.dead or self.cfg.hard_kill:
            return
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym != self.sym:
            return

        self._update_md(tick)
        now = self._now_ms()

        # Completion / kill
        if self.filled_qty >= self.cfg.parent_qty * self.cfg.kill_pct_complete:
            self.dead = True
            self.emit_signal(0.0)
            return

        # Toxicity guard
        vol_z = math.sqrt(self.ret2_ewma) * 100.0
        if vol_z >= self.cfg.pause_on_vol_z:
            return

        # pacing
        if now - self.last_child_ms < self.cfg.child_interval_ms:
            return

        qty = self._child_qty()
        if qty <= 0 or self.mid <= 0:
            return

        # If spread very wide or we are already getting negative slippage, prefer lit peg
        spread_bps = (self.ask - self.bid) / max(1e-9, self.mid) * 1e4 if (self.bid > 0 and self.ask > 0) else self.spread_ewma
        if spread_bps >= self.cfg.reject_on_wide_spread_bps:
            self._post_lit_midpeg(qty)
            self.last_child_ms = now
            return

        # Dark first: pick venue via UCB, otherwise fallback to lit
        venue = self._choose_dark(now)
        use_dark = bool(venue)

        # If last few moves are strongly against our side (impact risk), prefer lit peg
        if self.buy and self.ask > 0 and self.mid > 0:
            bps_vs_ask = (self.ask - self.mid) / self.mid * 1e4
            if bps_vs_ask > self.cfg.fade_on_impact_bps:
                use_dark = False
        elif (not self.buy) and self.bid > 0 and self.mid > 0:
            bps_vs_bid = (self.mid - self.bid) / self.mid * 1e4
            if bps_vs_bid > self.cfg.fade_on_impact_bps:
                use_dark = False

        if use_dark and venue:
            self._post_dark_ioc(venue, qty)
        else:
            self._post_lit_midpeg(qty)

        self.last_child_ms = now

        # health signal: remaining fraction (buy positive, sell negative)
        rem_frac = self._remaining() / max(1.0, self.cfg.parent_qty)
        self.emit_signal(rem_frac if self.buy else -rem_frac)


# ---------------- runner (optional) ----------------

if __name__ == "__main__":
    """
    Example:
        cfg = DarkPoolRouterConfig(symbol="AAPL", side="buy", parent_qty=80_000)
        strat = DarkPoolRouter(cfg=cfg)
        # strat.run(stream="ticks.equities.us")
    """
    strat = DarkPoolRouter()
    # strat.run(stream="ticks.equities.us")