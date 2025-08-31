# backend/ai/agents/rl/reward_shaping.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List
import math
import time

# -------------------- helpers --------------------
def now_ms() -> int: return int(time.time() * 1000)

def bps(a: float, b: float) -> float:
    """Return (a-b)/b in basis points; safe for b=0."""
    if b == 0: return 0.0
    return (a - b) / b * 1e4

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b not in (0.0, 0) else default

class EWMA:
    def __init__(self, alpha: float = 0.1):
        self.a = float(alpha); self.m = None
    def update(self, x: float) -> float:
        self.m = x if self.m is None else (1 - self.a) * self.m + self.a * x
        return self.m or 0.0
    @property
    def value(self) -> float:
        return 0.0 if self.m is None else self.m

class EWMAVar:
    """Welford-like EWMA variance for normalization."""
    def __init__(self, alpha: float = 0.1):
        self.a = float(alpha); self.mean = None; self.var = 0.0
    def update(self, x: float) -> float:
        if self.mean is None:
            self.mean = x; self.var = 0.0
        else:
            delta = x - self.mean
            self.mean = (1 - self.a) * self.mean + self.a * x
            self.var = (1 - self.a) * (self.var + self.a * delta * delta)
        return self.var
    def z(self, x: float, eps: float = 1e-12) -> float:
        if self.mean is None: return 0.0
        sd = math.sqrt(max(self.var, eps))
        return (x - self.mean) / (sd if sd > 0 else 1.0)

# -------------------- data schemas --------------------
@dataclass
class StepMetrics:
    # Execution / fills
    arrival_px: float                    # price when parent order submitted
    fill_px: Optional[float] = None      # VWAP of fills in this step (None if no fills)
    filled_qty: float = 0.0              # filled in this step
    parent_remaining: float = 0.0
    market_vwap_px: Optional[float] = None  # reference VWAP for period (if known)
    queue_pos: Optional[float] = None    # 0..1 where 0=head, 1=tail (smaller better)
    spread_bps: Optional[float] = None   # current bid-ask spread in bps
    latency_ms: Optional[int] = None
    venue_fee_bps: Optional[float] = None
    venue_toxicity: Optional[float] = None  # 0..1
    # Risk / realized
    mark_px: Optional[float] = None      # mark at step end
    inventory: float = 0.0               # +long/-short after step (for MM)
    pnl_step: Optional[float] = None     # realized + unrealized change
    # Penalties / flags
    safety_violation: bool = False
    policy_breach: bool = False

@dataclass
class EpisodeSummary:
    parent_qty: float
    final_vwap_px: Optional[float] = None   # VWAP of all fills
    benchmark_px: Optional[float] = None    # e.g., arrival, prev_close, target VWAP
    realized_pnl: Optional[float] = None
    max_drawdown: Optional[float] = None
    fill_ratio: float = 0.0
    slippage_bps: Optional[float] = None
    shortfall_bps: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    violations: int = 0
    notes: Dict[str, float] = field(default_factory=dict)

# -------------------- Reward Shaper --------------------
class RewardShaper:
    """
    General-purpose reward shaping for:
      • Execution RL (min shortfall/slippage, maximize timely fills, avoid toxic venues, low latency)
      • Market-making (inventory-adjusted spread capture)
      • Arbitrage (edge realization minus latency/leg risk)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, *,
                 clip: float = 10.0, normalize: bool = True, name: str = "execution"):
        # Default weights tuned for execution learning
        base = {
            "neg_shortfall_bps": 0.40,     # reward = - shortfall (bps)  (buy: fill_px-arrival)
            "neg_slip_bps":      0.25,     # reward = - (fill_px - market_vwap) bps
            "fill_ratio":        0.20,     # reward for filling when remaining > 0
            "low_latency":       0.05,     # reward for quick feedback
            "low_toxicity":      0.04,     # reward for cleaner venues
            "queue_advantage":   0.03,     # reward for being near head of queue
            "low_fees":          0.03,     # lower fee venues
            "risk_penalty":     -0.10,     # penalize |inventory| and pnl volatility
            "violations":       -1.00,     # hard penalty for safety/policy violations (per step)
        }
        self.w = base if weights is None else {**base, **weights}
        self.clip = float(clip)
        self.normalize = bool(normalize)

        # Running normalizers
        self._z_shortfall = EWMAVar(0.05)
        self._z_slip = EWMAVar(0.05)
        self._z_latency = EWMAVar(0.05)
        self._z_inv = EWMAVar(0.05)
        self._z_pnl = EWMAVar(0.05)

        self.name = name
        self._last_ts = now_ms()

    # -------------- reward per step --------------
    def step_reward(self, m: StepMetrics, *, side: str, parent_qty: float) -> float:
        """
        side: "buy" or "sell"
        parent_qty: total intended size (for fill ratio scaling)
        """
        sgn = 1.0 if side.lower() == "buy" else -1.0
        r = 0.0

        # --- Shortfall vs arrival (implementation shortfall)
        shortfall_bps = None
        if m.fill_px is not None and math.isfinite(m.arrival_px) and m.arrival_px != 0:
            # For buy, (fill - arrival) positive is BAD; negate below
            diff = (m.fill_px - m.arrival_px) * sgn
            shortfall_bps = diff / m.arrival_px * 1e4
            sf = shortfall_bps
            if self.normalize: sf = self._z_shortfall.z(shortfall_bps)
            r += self.w["neg_shortfall_bps"] * (-sf)

        # --- Slippage vs market VWAP reference (period micro-benchmark)
        slip_bps = None
        if m.fill_px is not None and m.market_vwap_px:
            diff = (m.fill_px - m.market_vwap_px) * sgn
            slip_bps = diff / m.market_vwap_px * 1e4
            sp = slip_bps
            if self.normalize: sp = self._z_slip.z(slip_bps)
            r += self.w["neg_slip_bps"] * (-sp)

        # --- Fill ratio (encourage progress when remaining is large)
        if parent_qty > 0:
            remaining = max(0.0, m.parent_remaining)
            need = max(1e-9, remaining + m.filled_qty)  # remaining before this step
            step_fill_ratio = safe_div(m.filled_qty, parent_qty, 0.0)
            urgency_scale = min(1.0, safe_div(need, parent_qty, 0.0) * 2.0)
            r += self.w["fill_ratio"] * step_fill_ratio * urgency_scale

        # --- Queue position advantage (closer to head better)
        if m.queue_pos is not None:
            # queue_pos in [0,1]; reward (1 - q)
            r += self.w["queue_advantage"] * (1.0 - max(0.0, min(1.0, m.queue_pos)))

        # --- Low latency reward
        if m.latency_ms is not None and m.latency_ms >= 0:
            lat = m.latency_ms
            lat_norm = -self._z_latency.z(lat) if self.normalize else -lat / 1000.0
            r += self.w["low_latency"] * lat_norm

        # --- Venue toxicity (prefer cleaner venues)
        if m.venue_toxicity is not None:
            r += self.w["low_toxicity"] * (1.0 - max(0.0, min(1.0, m.venue_toxicity)))

        # --- Fees (prefer cheaper venues)
        if m.venue_fee_bps is not None:
            # negative contribution for higher fees
            r += self.w["low_fees"] * (-float(m.venue_fee_bps) / 10.0)

        # --- Risk penalty (inventory magnitude + pnl variance proxy)
        inv_pen = 0.0
        if m.inventory is not None:
            inv_pen += abs(m.inventory)
        if m.pnl_step is not None:
            inv_pen += abs(self._z_pnl.z(m.pnl_step)) if self.normalize else abs(m.pnl_step)
        if inv_pen:
            r += self.w["risk_penalty"] * inv_pen

        # --- Hard penalties
        if m.safety_violation:
            r += self.w["violations"] * 1.0
        if m.policy_breach:
            r += self.w["violations"] * 0.5

        # Clip to keep stable
        r = max(-self.clip, min(self.clip, r))

        # Update normalizers (after usage to avoid leakage)
        if shortfall_bps is not None: self._z_shortfall.update(shortfall_bps)
        if slip_bps is not None: self._z_slip.update(slip_bps)
        if m.latency_ms is not None: self._z_latency.update(float(m.latency_ms))
        self._z_inv.update(abs(m.inventory))

        self._last_ts = now_ms()
        return float(r)

    # -------------- episode terminal bonus --------------
    def episode_bonus(self, ep: EpisodeSummary, *, side: str) -> float:
        """
        Terminal shaping after episode finishes (parent order complete or timed out).
        Encourages hitting benchmark and finishing the order.
        """
        sgn = 1.0 if side.lower() == "buy" else -1.0
        R = 0.0

        # Hit benchmark (arrival / target VWAP)
        if ep.final_vwap_px and ep.benchmark_px:
            diff_bps = (ep.final_vwap_px - ep.benchmark_px) / ep.benchmark_px * 1e4 * sgn
            z = self._z_slip.z(diff_bps) if self.normalize else diff_bps
            R += 1.0 * (-z)   # strong bonus for beating benchmark

        # Completion
        R += 0.5 * (1.0 if ep.fill_ratio >= 0.999 else - (1.0 - ep.fill_ratio))

        # Latency preference (lower is better)
        if ep.avg_latency_ms is not None:
            R += 0.1 * (-self._z_latency.z(ep.avg_latency_ms))

        # Violations
        if ep.violations:
            R -= 1.0 * ep.violations

        # Drawdown aversion
        if ep.max_drawdown is not None:
            R -= 0.2 * abs(ep.max_drawdown)

        return max(-self.clip, min(self.clip, R))

    # -------------- presets --------------
    @staticmethod
    def preset_execution() -> "RewardShaper":
        return RewardShaper()

    @staticmethod
    def preset_market_making() -> "RewardShaper":
        w = {
            # Encourage spread capture net of inventory risk
            "neg_shortfall_bps": 0.00,
            "neg_slip_bps":      0.10,
            "fill_ratio":        0.10,
            "low_latency":       0.10,
            "low_toxicity":      0.05,
            "queue_advantage":   0.15,
            "low_fees":          0.05,
            "risk_penalty":     -0.30,   # strong penalty for large |inventory|
            "violations":       -1.00,
        }
        return RewardShaper(weights=w, clip=10.0, normalize=True, name="market_making")

    @staticmethod
    def preset_arbitrage() -> "RewardShaper":
        w = {
            "neg_shortfall_bps": 0.05,  # small relative to realized edge
            "neg_slip_bps":      0.10,
            "fill_ratio":        0.10,
            "low_latency":       0.30,  # critical for arb
            "low_toxicity":      0.05,
            "queue_advantage":   0.10,
            "low_fees":          0.10,
            "risk_penalty":     -0.20,
            "violations":       -1.00,
        }
        return RewardShaper(weights=w, clip=10.0, normalize=True, name="arbitrage")

# -------------------- quick smoke tests --------------------
if __name__ == "__main__":  # pragma: no cover
    # Execution example (BUY parent)
    sh = RewardShaper.preset_execution()
    steps: List[StepMetrics] = []
    parent = 10_000
    remaining = parent
    for i in range(10):
        filled = 1000 if i < 8 else (parent - (i * 1000))  # last 2 steps smaller
        remaining = max(0.0, remaining - filled)
        sm = StepMetrics(
            arrival_px=100.00,
            fill_px=100.01 if i % 2 == 0 else 99.99,
            filled_qty=filled,
            parent_remaining=remaining,
            market_vwap_px=100.00,
            queue_pos=0.3,
            spread_bps=5.0,
            latency_ms=20 + i,
            venue_fee_bps=0.2,
            venue_toxicity=0.2,
            mark_px=100.02,
            inventory=remaining * 0.0,
            pnl_step=(0.02 if i % 2 == 0 else -0.01),
            safety_violation=False,
            policy_breach=False,
        )
        r = sh.step_reward(sm, side="buy", parent_qty=parent)
        steps.append(sm)
        print(f"step {i} reward={r:.4f}")

    ep = EpisodeSummary(
        parent_qty=parent,
        final_vwap_px=100.002,
        benchmark_px=100.0,
        realized_pnl=12.0,
        max_drawdown=0.005,
        fill_ratio=0.998,
        slippage_bps=-0.2,
        shortfall_bps=-0.2,
        avg_latency_ms=25.0,
        violations=0
    )
    R = sh.episode_bonus(ep, side="buy")
    print("episode bonus:", round(R, 4))