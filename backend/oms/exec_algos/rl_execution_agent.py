# backend/engine/strategies/rl_execution_agent.py
from __future__ import annotations

import math
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset
from backend.microstructure.queue_position import QueueTracker # type: ignore


# ----------------------------- Config -----------------------------

@dataclass
class RLExecConfig:
    symbol: str = "SPY"
    side: str = "buy"                      # "buy" | "sell"
    parent_qty: float = 50_000.0

    # action set
    improve_bps: float = 0.4               # for post_improve / replace_improve
    dark_min_exec: float = 200.0           # min size for dark IOC
    actions: Tuple[str, ...] = (
        "post_midpeg",         # limit at mid +/- tiny improve
        "post_improve",        # limit further inside touch (improves queue priority)
        "take_market",         # market/near-touch
        "dark_ioc",            # route IOC to dark pool
        "hold"                 # do nothing this step
    )

    # pacing / sizing
    child_interval_ms: int = 1200
    min_child_qty: float = 300.0
    max_child_qty: float = 5_000.0
    batch_ratio: float = 0.15              # portion of remaining to attempt per decision

    # risk & guards
    notional_cap: float = 2_000_000.0
    pause_on_vol_z: float = 4.5
    widen_spread_bps: float = 9.0
    kill_pct_complete: float = 1.02
    hard_kill: bool = False

    # bandit / Q-learning
    epsilon: float = 0.08                   # exploration rate
    alpha: float = 0.25                     # learning rate
    gamma: float = 0.60                     # discount for 1-step TD
    state_bins: Tuple[int, int, int, int] = (6, 6, 5, 4)
    # state features: [spread_bps, vol_z, book_imbalance, queue_hint]
    #   spread_bps       : (ask-bid)/mid * 1e4
    #   vol_z            : sqrt(ret2_ewma)*100 (%-like)
    #   book_imbalance   : (bid_sz-ask_sz)/(bid_sz+ask_sz) in [-1,1], fallback 0
    #   queue_hint       : normalized queue_ahead/self_qty in [0, +inf) → clipped to [0,3]

    # reward shaping
    bench: str = "mid"                      # "mid" | "vwap"
    fill_weight: float = 0.5                # reward includes fill ratio
    impact_weight: float = 0.2              # penalize adverse move after action
    time_penalty: float = 0.0005            # tiny penalty per decision tick to encourage finishing

    # venues/tags (mapped by your OMS)
    dark_venues: Tuple[str, ...] = ("DBKX", "MSPL", "BATS-D", "LQDX")
    tag_dark_ioc: str = "DARK_IOC"


# ------------------------------ Agent ------------------------------

class RLExecutionAgent(Strategy):
    """
    RL execution agent (contextual bandit / 1-step Q-learning):
      • Builds a compact state vector from microstructure.
      • Chooses among {post_midpeg, post_improve, take_market, dark_ioc, hold}.
      • Receives reward from realized slippage vs benchmark and slight time/impact penalties.
      • Learns online (tabular Q, discretized state).
    Expects:
      market data ticks: {symbol|s, bid, ask, last|price|p|mid, size|q|qty, bid_size?, ask_size?}
      OMS events       : call on_order_ack / on_fill from your adapter (quantities/prices/venue).
    """

    def __init__(self, name="exec_rl_agent", region=None, cfg: Optional[RLExecConfig] = None):
        cfg = cfg or RLExecConfig()
        super().__init__(name=name, region=region, default_qty=cfg.min_child_qty)
        self.cfg = cfg
        self.sym = cfg.symbol.upper()
        self.buy = (cfg.side.lower() == "buy")

        # microstructure
        self.bid = 0.0
        self.ask = 0.0
        self.mid = 0.0
        self.last_mid = 0.0
        self.bid_sz = 0.0
        self.ask_sz = 0.0
        self.ret2_ewma = 0.0
        self.spread_ewma = 2.0

        # VWAP agg
        self.vwap_num = 0.0
        self.vwap_den = 0.0

        # progress
        self.filled_qty = 0.0
        self.last_child_ms = 0
        self.dead = False

        # queue tracker
        self.qt = QueueTracker()
        self.last_order_id: Optional[str] = None

        # Q-table: dict[(s0,s1,s2,s3)][action_index] -> value
        self.Q: Dict[Tuple[int, int, int, int], List[float]] = {}

        # Last decision for TD update
        self._last_state: Optional[Tuple[int, int, int, int]] = None
        self._last_action: Optional[int] = None
        self._last_bench_px: float = 0.0
        self._last_time_ms: int = 0

    # ---------------- lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["execution", "RL", "bandit", "adaptive"],
            "underlying": self.sym,
            "notes": "Contextual bandit / tabular Q-learning over microstructure for execution decisions."
        })

    # ---------------- utilities ----------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _sf(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d

    def _rolling_vwap(self) -> float:
        return self.vwap_num / self.vwap_den if self.vwap_den > 0 else (self.mid or 0.0)

    # --------- state features + discretization ----------
    def _features(self) -> Tuple[float, float, float, float]:
        # spread (bps)
        spread_bps = 0.0
        if self.bid > 0 and self.ask > 0:
            spread_bps = (self.ask - self.bid) / max(1e-9, self.mid) * 1e4

        # vol z (rough)
        vol_z = math.sqrt(max(0.0, self.ret2_ewma)) * 100.0

        # imbalance
        imb = 0.0
        tot = self.bid_sz + self.ask_sz
        if tot > 0:
            imb = (self.bid_sz - self.ask_sz) / tot  # [-1, +1]

        # queue hint (need a working order to be meaningful)
        qa_norm = 0.0
        ref_qty = max(1.0, self.cfg.min_child_qty)
        qa = self.qt.queue_ahead(self.sym, "buy" if self.buy else "sell")
        if qa is not None:
            qa_norm = min(3.0, max(0.0, qa / ref_qty))

        return spread_bps, vol_z, imb, qa_norm

    def _disc(self, x: float, lo: float, hi: float, n: int) -> int:
        if n <= 1:
            return 0
        if x <= lo:
            return 0
        if x >= hi:
            return n - 1
        z = (x - lo) / (hi - lo)
        return int(z * n)

    def _state(self) -> Tuple[int, int, int, int]:
        # heuristic bin ranges
        spread_bps, vol_z, imb, qa_norm = self._features()
        s_bins, v_bins, i_bins, q_bins = self.cfg.state_bins
        s = self._disc(spread_bps, 0.0, 20.0, s_bins)          # 0..20 bps
        v = self._disc(vol_z, 0.0, 3.0, v_bins)                 # 0..3 vol-z
        i = self._disc((imb + 1.0) * 0.5, 0.0, 1.0, i_bins)     # map [-1,1]→[0,1]
        q = self._disc(qa_norm, 0.0, 3.0, q_bins)               # 0..3x ref qty ahead
        return (s, v, i, q)

    def _ensure_state(self, s: Tuple[int, int, int, int]) -> None:
        if s not in self.Q:
            self.Q[s] = [0.0 for _ in self.cfg.actions]

    # ---------------- reward ----------------
    def _benchmark_px(self) -> float:
        if self.cfg.bench == "vwap":
            px = self._rolling_vwap()
            return px if px > 0 else (self.mid or 0.0)
        return self.mid or 0.0

    def _apply_reward(self, final: bool = False) -> None:
        # Needs last decision context and observed outcome (fills + microstructure)
        if self._last_state is None or self._last_action is None or self._last_bench_px <= 0:
            return

        # Compute realized average fill since last decision (approx via VWAP diff on fills)
        # For simplicity, reward is only computed when we see at least some fill progress OR on final.
        # You may wire precise fill accounting via OMS to store per-decision fill PX/QTY.
        # Here, we approximate using current VWAP aggregates and filled_qty delta.
        now = self._now_ms()
        dt_s = max(0.001, (now - self._last_time_ms) / 1000.0)

        # slippage proxy:
        #  - For BUY: positive reward if fills <= benchmark (we paid below bench)
        #  - For SELL: positive reward if fills >= benchmark
        bench = self._last_bench_px
        last_mid_now = self.mid or bench
        # impact: if price moved against us after action, subtract a small penalty
        impact = 0.0
        if self.buy:
            impact = max(0.0, (last_mid_now - bench) / max(1e-9, bench)) * self.cfg.impact_weight
        else:
            impact = max(0.0, (bench - last_mid_now) / max(1e-9, bench)) * self.cfg.impact_weight

        # Fill ratio estimate (did we advance?)
        # In practice, call _note_fill() per decision window; here we use change in filled qty via OMS callbacks.
        fill_ratio = 0.0  # strategy doesn’t track per-window fills; optional future hook

        # base reward – neutral small negative to avoid endless holding
        reward = -self.cfg.time_penalty * dt_s

        # slippage reward only if we have a last trade price (mid proxy)
        if self.buy:
            # positive if current is <= bench (we didn't chase worse)
            edge = max(-0.02, min(0.02, (bench - last_mid_now) / max(1e-9, bench)))
        else:
            edge = max(-0.02, min(0.02, (last_mid_now - bench) / max(1e-9, bench)))
        reward += edge
        reward -= impact
        reward += self.cfg.fill_weight * fill_ratio

        # 1-step TD update
        s = self._last_state
        a = self._last_action
        self._ensure_state(s)
        q_sa = self.Q[s][a]

        s_next = self._state()
        self._ensure_state(s_next)
        max_next = max(self.Q[s_next]) if not final else 0.0

        td_target = reward + self.cfg.gamma * max_next
        self.Q[s][a] = (1 - self.cfg.alpha) * q_sa + self.cfg.alpha * td_target

        # clear context
        self._last_state = None
        self._last_action = None
        self._last_bench_px = 0.0
        self._last_time_ms = now

    # ---------------- action execution ----------------
    def _child_qty(self) -> float:
        rem = max(0.0, self.cfg.parent_qty - self.filled_qty)
        if rem <= 0:
            return 0.0
        qty = max(self.cfg.min_child_qty, min(self.cfg.max_child_qty, rem * self.cfg.batch_ratio))
        if self.mid > 0 and qty * self.mid > self.cfg.notional_cap:
            qty = max(self.cfg.min_child_qty, self.cfg.notional_cap / self.mid)
        return max(0.0, qty)

    def _post_midpeg(self, qty: float) -> None:
        if self.mid <= 0 or qty <= 0: return
        improve = self.mid * (0.1 / 1e4)  # tiny default improve
        limit = self.mid - improve if self.buy else self.mid + improve
        self.order(self.sym, "buy" if self.buy else "sell", qty=qty, order_type="limit",
                   limit_price=limit, extra={"reason": "rl_post_midpeg"})

    def _post_improve(self, qty: float) -> None:
        if self.mid <= 0 or qty <= 0: return
        improve = self.mid * (self.cfg.improve_bps / 1e4)
        limit = self.mid - improve if self.buy else self.mid + improve
        self.order(self.sym, "buy" if self.buy else "sell", qty=qty, order_type="limit",
                   limit_price=limit, extra={"reason": "rl_post_improve", "improve_bps": self.cfg.improve_bps})

    def _take_market(self, qty: float) -> None:
        if self.mid <= 0 or qty <= 0: return
        self.order(self.sym, "buy" if self.buy else "sell", qty=qty, order_type="market",
                   mark_price=self.mid, extra={"reason": "rl_take_market"})

    def _dark_ioc(self, qty: float) -> None:
        if self.mid <= 0 or qty <= 0: return
        qty = max(qty, self.cfg.dark_min_exec)
        self.order(self.sym, "buy" if self.buy else "sell", qty=qty, order_type="market",
                   mark_price=self.mid,
                   extra={"reason": "rl_dark_ioc", "exec": self.cfg.tag_dark_ioc, "venue_hint": self.cfg.dark_venues[0]})

    def _act(self, a_idx: int) -> None:
        qty = self._child_qty()
        if qty <= 0:
            return
        act = self.cfg.actions[a_idx]
        if act == "post_midpeg":
            self._post_midpeg(qty)
        elif act == "post_improve":
            self._post_improve(qty)
        elif act == "take_market":
            # guard: avoid taking on very wide spread
            if self.bid > 0 and self.ask > 0:
                spread_bps = (self.ask - self.bid) / max(1e-9, self.mid) * 1e4
                if spread_bps >= self.cfg.widen_spread_bps:
                    return
            self._take_market(qty)
        elif act == "dark_ioc":
            # skip if spread is extreme (toxicity)
            if self.bid > 0 and self.ask > 0:
                spread_bps = (self.ask - self.bid) / max(1e-9, self.mid) * 1e4
                if spread_bps >= self.cfg.widen_spread_bps:
                    return
            self._dark_ioc(qty)
        else:
            # hold: no action this tick
            pass

    # ---------------- main loop ----------------
    def _update_md(self, t: Dict[str, Any]) -> None:
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
            p = t.get("last") or t.get("price") or t.get("p") or t.get("mid")
            if p is not None:
                px = self._sf(p)
                if px > 0 and self.mid == 0.0:
                    self.mid = px
                    self.last_mid = px

        # sizes (optional fields)
        self.bid_sz = self._sf(t.get("bid_size"), self.bid_sz)
        self.ask_sz = self._sf(t.get("ask_size"), self.ask_sz)

        # tape for VWAP
        p = t.get("last") or t.get("price") or t.get("p") or t.get("mid")
        q = t.get("size") or t.get("q") or t.get("qty")
        if p is not None and q is not None:
            px = self._sf(p); sz = self._sf(q)
            if px > 0 and sz > 0:
                self.vwap_num += px * sz
                self.vwap_den += sz

        # feed queue tracker best level (if available)
        if self.bid > 0 and self.ask > 0:
            self.qt.on_l2(self.sym, bids=((self.bid, self.bid_sz or 0.0),), asks=((self.ask, self.ask_sz or 0.0),))

    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.dead or self.cfg.hard_kill:
            return
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym != self.sym:
            return

        self._update_md(tick)
        now = self._now_ms()

        # completion/kill
        if self.filled_qty >= self.cfg.parent_qty * self.cfg.kill_pct_complete:
            self.dead = True
            self.emit_signal(0.0)
            # finalize reward for the last decision
            self._apply_reward(final=True)
            return

        # toxicity guard
        vol_z = math.sqrt(max(0.0, self.ret2_ewma)) * 100.0
        if vol_z >= self.cfg.pause_on_vol_z:
            return

        # pacing
        if now - self.last_child_ms < self.cfg.child_interval_ms:
            return

        # TD update from last step before taking a new action
        self._apply_reward(final=False)

        # choose action ε-greedily
        s = self._state()
        self._ensure_state(s)
        import random
        if random.random() < self.cfg.epsilon:
            a_idx = random.randrange(len(self.cfg.actions))
        else:
            vals = self.Q[s]
            a_idx = max(range(len(vals)), key=lambda i: vals[i])

        # benchmark snapshot for this decision
        self._last_state = s
        self._last_action = a_idx
        self._last_bench_px = self._benchmark_px()
        self._last_time_ms = now

        # act
        self._act(a_idx)
        self.last_child_ms = now

        # signal = remaining fraction (buy positive, sell negative)
        rem_frac = max(0.0, self.cfg.parent_qty - self.filled_qty) / max(1.0, self.cfg.parent_qty)
        self.emit_signal(rem_frac if self.buy else -rem_frac)

    # ---------------- OMS hooks ----------------
    def on_order_ack(self, order_id: str, side: Optional[str], price: float, qty: float) -> None:
        """
        Call from your OMS when a child goes live.
        """
        s = "buy" if (side or ("buy" if self.buy else "sell")).lower() == "buy" else "sell"
        self.last_order_id = order_id
        self.qt.on_order_ack(self.sym, order_id=order_id, side=s, price=float(price), qty=float(qty))

    def on_replace_ack(self, order_id: str, new_price: float, new_qty: Optional[float] = None) -> None:
        self.qt.on_replace_ack(self.sym, order_id=order_id, new_price=float(new_price), new_qty=new_qty)

    def on_cancel_ack(self, order_id: str) -> None:
        self.qt.on_cancel_ack(self.sym, order_id=order_id)

    def on_fill(self, qty: float, price: float, venue: Optional[str] = None) -> None:
        """
        OMS should invoke this for fills. We update filled_qty, VWAP aggregates, and queue tracker.
        """
        q = max(0.0, float(qty))
        px = max(0.0, float(price))
        self.filled_qty += q
        if px > 0 and q > 0:
            self.vwap_num += px * q
            self.vwap_den += q
        if self.last_order_id:
            self.qt.on_fill(self.sym, order_id=self.last_order_id, fill_qty=q, fill_price=px)

    # ---------------- snapshot/export (optional) ----------------
    def q_snapshot(self) -> Dict[str, Any]:
        """
        Export a compact JSON snapshot of Q-values for dashboards.
        """
        out = {
            "actions": list(self.cfg.actions),
            "states": [],
        }
        for st, vals in list(self.Q.items())[:2048]:
            out["states"].append({"s": st, "q": vals})
        return out


# ---------------- runner (optional) ----------------

if __name__ == "__main__":
    """
    Example wiring (your orchestrator should set this up and feed ticks + OMS callbacks):
        cfg = RLExecConfig(symbol="AAPL", side="buy", parent_qty=100_000)
        strat = RLExecutionAgent(cfg=cfg)
        # strat.run(stream="ticks.equities.us")
    """
    strat = RLExecutionAgent()
    # strat.run(stream="ticks.equities.us")