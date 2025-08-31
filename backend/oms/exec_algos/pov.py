# backend/engine/strategies/pov.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


# -------------------- Config --------------------

@dataclass
class POVConfig:
    symbol: str = "SPY"
    side: str = "buy"                    # "buy" | "sell"
    target_qty: float = 50_000.0         # max we intend to execute (optional cap)

    # Core participation
    participation: float = 0.10          # 10% of traded volume (0..1)
    min_child_qty: float = 200.0
    max_child_qty: float = 5_000.0
    child_interval_ms: int = 1000        # min spacing between children

    # Microstructure & adaptivity
    post_improve_bps: float = 0.3        # passive: mid ± improvement
    widen_spread_bps: float = 8.0        # if spread above → be conservative (post smaller / skip)
    take_on_momentum: bool = True        # allow market if flow strongly with us
    momentum_window_ms: int = 1500
    momentum_min_trades: int = 5

    # Safety
    notional_cap: float = 2_000_000.0
    pause_on_vol_z: float = 4.0          # pause if short-horizon vol z too high
    kill_pct_complete: float = 1.02
    hard_kill: bool = False


# -------------------- Strategy --------------------

class POV(Strategy):
    """
    Participation-of-Volume execution:
      - Tracks tape prints and posts child orders sized as (participation * recent traded volume).
      - Prefers passive mid-peg with tiny improvement; can go market when momentum is strong in our favor.
      - Respects pacing, notional caps, and simple toxicity guards (volatility/spread).
    Tick tolerance:
      - Market data ticks: {symbol|s, bid, ask, last|price|p|mid, size|q|qty}
      - OMS should call on_fill(qty, price) to update progress (optional if your router does it elsewhere).
    """

    def __init__(self, name="exec_pov", region=None, cfg: Optional[POVConfig] = None):
        cfg = cfg or POVConfig()
        super().__init__(name=name, region=region, default_qty=cfg.min_child_qty)
        self.cfg = cfg
        self.sym = cfg.symbol.upper()
        self.buy = (cfg.side.lower() == "buy")

        # microstructure state
        self.bid = 0.0
        self.ask = 0.0
        self.mid = 0.0
        self.last_mid = 0.0
        self.spread_ewma = 2.0
        self.ret2_ewma = 0.0  # short-horizon variance proxy

        # volume/tape
        self.session_vol = 0.0
        self.window_vol = 0.0
        self.window_start_ms = self._now_ms()
        self.window_ms = 2000  # rolling window for sizing (2s default)

        # simple trade tape for momentum
        self.tape = []  # list[(ts_ms, side(+1 buy/-1 sell), px, sz)]

        # progress / pacing
        self.filled_qty = 0.0
        self.last_child_ms = 0
        self.dead = False

    # ---------- lifecycle ----------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["execution", "POV"],
            "underlying": self.sym,
            "notes": "Fixed participation-of-volume with mid-peg posting and optional momentum-taking."
        })

    # ---------- utils ----------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _sf(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d

    def _update_md_and_volume(self, t: Dict[str, Any]) -> None:
        # Spread & mid
        b, a = t.get("bid"), t.get("ask")
        if b is not None and a is not None:
            b = self._sf(b); a = self._sf(a)
            if b > 0 and a > 0:
                self.bid, self.ask = b, a
                self.mid = 0.5 * (b + a)
                spread_bps = (a - b) / max(1e-9, self.mid) * 1e4
                self.spread_ewma = 0.95 * self.spread_ewma + 0.05 * spread_bps
                if self.last_mid > 0:
                    r = (self.mid / self.last_mid) - 1.0
                    self.ret2_ewma = 0.97 * self.ret2_ewma + 0.03 * (r * r)
                self.last_mid = self.mid
        else:
            # fallback mid from last/price
            p = t.get("last") or t.get("price") or t.get("p") or t.get("mid")
            if p is not None:
                px = self._sf(p)
                if px > 0 and self.mid == 0.0:
                    self.mid = px
                    self.last_mid = px

        # Volume from prints
        p = t.get("last") or t.get("price") or t.get("p") or t.get("mid")
        q = t.get("size") or t.get("q") or t.get("qty")
        if p is not None and q is not None:
            px = self._sf(p); sz = self._sf(q)
            if px > 0 and sz > 0:
                self.session_vol += sz
                # maintain rolling window volume
                now = self._now_ms()
                self.window_vol += sz
                # decay window by time (coarse)
                if now - self.window_start_ms > self.window_ms:
                    self.window_vol = max(0.0, self.window_vol * 0.5)
                    self.window_start_ms = now

                # momentum tape (if side known, else infer from price vs mid)
                side = 0
                if self.bid and self.ask:
                    # naive: trade at/above ask → buyer; at/below bid → seller
                    if px >= self.ask:
                        side = +1
                    elif px <= self.bid:
                        side = -1
                self.tape.append((now, side, px, sz))
                # trim
                cutoff = now - max(self.cfg.momentum_window_ms, 500)
                self.tape = [x for x in self.tape if x[0] >= cutoff]

    def _child_qty(self) -> float:
        # base on participation of the recent window volume
        base = self.cfg.participation * max(0.0, self.window_vol)
        qty = max(self.cfg.min_child_qty, min(self.cfg.max_child_qty, base))
        # cap by remaining target (optional)
        rem = max(0.0, self.cfg.target_qty - self.filled_qty)
        qty = min(qty, rem) if self.cfg.target_qty > 0 else qty
        # notional cap
        if self.mid > 0 and qty * self.mid > self.cfg.notional_cap:
            qty = max(self.cfg.min_child_qty, self.cfg.notional_cap / self.mid)
        return max(0.0, qty)

    def _momentum_bias(self) -> float:
        """Return [-1, +1] bias from recent tape aggressor sides."""
        if not self.tape:
            return 0.0
        winsz = self.cfg.momentum_window_ms
        now = self._now_ms()
        recent = [sgn for ts, sgn, *_ in self.tape if (now - ts) <= winsz and sgn != 0]
        if not recent:
            return 0.0
        buys = sum(1 for s in recent if s > 0)
        sells = len(recent) - buys
        return (buys - sells) / max(1, len(recent))

    # ---------- execution ----------
    def _post_passive(self, side: str, qty: float) -> None:
        if self.mid <= 0:
            return
        improve = self.mid * (self.cfg.post_improve_bps / 1e4)
        limit = self.mid - improve if side == "buy" else self.mid + improve
        self.order(self.sym, side, qty=qty, order_type="limit", limit_price=limit,
                   extra={"reason": "pov_passive_midpeg", "participation": self.cfg.participation})

    def _go_aggressive(self, side: str, qty: float) -> None:
        if self.mid <= 0:
            return
        self.order(self.sym, side, qty=qty, order_type="market", mark_price=self.mid,
                   extra={"reason": "pov_take", "participation": self.cfg.participation})

    # ---------- main ----------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.dead or self.cfg.hard_kill:
            return
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym != self.sym:
            return

        self._update_md_and_volume(tick)
        now = self._now_ms()

        # completion/kill
        if self.cfg.target_qty > 0 and self.filled_qty >= self.cfg.target_qty * self.cfg.kill_pct_complete:
            self.dead = True
            self.emit_signal(0.0)
            return

        # toxicity guard
        vol_z = math.sqrt(self.ret2_ewma) * 100.0
        if vol_z >= self.cfg.pause_on_vol_z:
            return

        # pacing
        if now - self.last_child_ms < self.cfg.child_interval_ms:
            return

        qty = self._child_qty()
        if qty <= 0 or self.mid <= 0:
            return

        # spread guard
        spread_bps = (self.ask - self.bid) / max(1e-9, self.mid) * 1e4 if (self.bid > 0 and self.ask > 0) else self.spread_ewma
        side = "buy" if self.buy else "sell"

        # decide aggression:
        aggressive = False
        if self.cfg.take_on_momentum:
            bias = self._momentum_bias()
            if (self.buy and bias > 0.4) or ((not self.buy) and bias < -0.4):
                aggressive = True
        # if spread is too wide, avoid aggression regardless
        if spread_bps >= self.cfg.widen_spread_bps:
            aggressive = False

        if aggressive:
            self._go_aggressive(side, qty)
        else:
            self._post_passive(side, qty)

        self.last_child_ms = now

        # health signal: remaining fraction (positive for buy, negative for sell)
        if self.cfg.target_qty > 0:
            rem_frac = max(0.0, self.cfg.target_qty - self.filled_qty) / max(1.0, self.cfg.target_qty)
            self.emit_signal(rem_frac if self.buy else -rem_frac)

    # OMS can call this to keep progress accurate
    def on_fill(self, qty: float, price: float) -> None:
        q = max(0.0, float(qty))
        self.filled_qty += q
        # Optional: fold fills into session volume/tape stats if you want
        if price and q:
            self.session_vol += q


# ---------------- runner (optional) ----------------

if __name__ == "__main__":
    """
    Example:
        cfg = POVConfig(symbol="AAPL", side="buy", participation=0.12, target_qty=100_000)
        strat = POV(cfg=cfg)
        # strat.run(stream="ticks.equities.us")
    """
    strat = POV()
    # strat.run(stream="ticks.equities.us")