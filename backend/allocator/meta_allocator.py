# backend/allocator/meta_allocator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math
import time

@dataclass
class StratMetrics:
    """
    Rolling, *live* metrics per strategy (all bps relative to NAV unless noted).
    """
    sharpe: float            # rolling Sharpe (e.g., 20d)
    drawdown: float          # peak-to-trough as +fraction (0.12 = -12%)
    var99_bps: float         # daily VaR@99 (bps)
    es97_bps: float          # daily ES@97 (bps)
    vol_bps: float           # daily realized vol (bps)
    live_pnl_bps: float      # short-horizon realized pnl (bps)
    pulls: int               # number of allocation rounds observed
    reward_bps: float        # bandit reward proxy (e.g., alpha per risk unit)

@dataclass
class Limits:
    """
    Per-allocation hard/soft constraints.
    - max_w: per strategy weight cap (fraction of NAV)
    - max_port_var_bps: portfolio VaR cap used for global scaling (caller passes actual VaR)
    - max_strat_es_bps: hard ES cap *per strategy*
    - max_strat_dd: hard drawdown cap *per strategy* (fraction, e.g., 0.15 for 15%)
    """
    max_w: float = 0.10
    max_port_var_bps: float = 300.0
    max_strat_es_bps: float = 120.0
    max_strat_dd: float = 0.15

def ucb_score(m: StratMetrics, t_rounds: int, c: float = 2.0) -> float:
    """
    Upper-Confidence-Bound score on risk-adjusted reward.
    Exploration bonus shrinks as pulls increase.
    """
    base = m.reward_bps / max(1.0, m.vol_bps)  # crude risk-normalization
    bonus = math.sqrt((c * math.log(max(2, t_rounds))) / max(1, m.pulls))
    return base + bonus

def risk_penalty(m: StratMetrics, lim: Limits) -> float:
    """
    Soft penalties near ES/DD limits; hard zero when breached.
    """
    if m.es97_bps > lim.max_strat_es_bps:
        return 0.0
    if m.drawdown > lim.max_strat_dd:
        return 0.0
    p = 1.0
    if m.es97_bps > 0.8 * lim.max_strat_es_bps:
        p *= 0.6
    if m.drawdown > 0.8 * lim.max_strat_dd:
        p *= 0.6
    return p

def allocate(strats: Dict[str, StratMetrics], lim: Limits, last_weights: Dict[str, float], port_var_bps: float) -> Dict[str, float]:
    """
    Compute new per-strategy weights subject to risk guards.
    Steps:
      1) UCB scores × risk penalties
      2) Per-strategy cap
      3) Normalize
      4) Global scale if portfolio VaR exceeds cap
      5) Inertia blend with last_weights to reduce churn
    Returns:
      Dict[str, float] weights (sum ≤ 1)
    """
    if not strats:
        return {}

    # 1) UCB × penalties
    t_rounds = sum(m.pulls for m in strats.values()) + 1
    scores = {k: ucb_score(m, t_rounds) * risk_penalty(m, lim) for k, m in strats.items()}
    scores = {k: (v if v > 0 else 0.0) for k, v in scores.items()}
    ssum = sum(scores.values()) or 1.0

    # 2) raw proportional + per-strategy cap
    w = {k: min(lim.max_w, max(0.0, v / ssum)) for k, v in scores.items()}

    # 3) normalize after capping
    norm = sum(w.values()) or 1.0
    w = {k: v / norm for k, v in w.items()}

    # 4) global scale by VaR if needed (caller supplies current port_var_bps)
    if port_var_bps > lim.max_port_var_bps:
        scale = lim.max_port_var_bps / max(1e-6, port_var_bps)
        w = {k: v * scale for k, v in w.items()}

    # 5) inertia to reduce turnover/churn
    alpha = 0.20  # 20% step toward target
    out = {k: (1 - alpha) * last_weights.get(k, 0.0) + alpha * w.get(k, 0.0) for k in strats.keys()}

    # final normalize safety
    s = sum(out.values())
    if s > 1.0:
        out = {k: v / s for k, v in out.items()}
    return out

def decision_proof(weights: Dict[str, float], lim: Limits, port_var_bps: float) -> Dict[str, str]:
    """
    Machine-checkable proof that key constraints have been enforced.
    """
    ok = (port_var_bps <= lim.max_port_var_bps) and all(v <= lim.max_w + 1e-9 for v in weights.values())
    return {
        "ts": str(int(time.time() * 1000)),
        "proof_type": "allocator_limits_check",
        "verdict": "PASS" if ok else "FAIL",
        "port_var_bps": f"{port_var_bps:.2f}",
        "max_port_var_bps": f"{lim.max_port_var_bps:.2f}",
        "max_w": f"{lim.max_w:.4f}",
    }