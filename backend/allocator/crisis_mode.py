# backend/allocator/crisis_mode.py
"""
Crisis Mode Controller
----------------------
Purpose:
  • Detect systemic stress early (tail risk, regime, drawdown, liquidity).
  • Enter crisis mode with hysteresis; de-gear portfolio, raise cash, add hedges.
  • Exit only after conditions normalize for sustained periods.
  • Produce machine-checkable proof; (optionally) append Merkle-ledger entry.

Inputs (summarized):
  - current_weights: Dict[str, float]          # per-strategy weights (sum ≤ 1)
  - port_metrics: PortfolioMetrics             # live risk & health
  - regime_hint: Dict[str, float]              # from RegimeOracle.regime_signal() (risk_multiplier, hedge_bias)
  - cfg: CrisisConfig                          # thresholds & actions
  - ledger_path: Optional[str]                 # append decision to audit/merkle ledger

Outputs:
  - CrisisDecision {active, new_weights, hedge_plan, proof}

Safe defaults:
  - If dynamic_hedges is not importable, we fallback to a simple index-put overlay plan.
  - If OMS is not available, we only return the plan (no side effects).

Recommended wiring:
  1) Call `decide_and_apply(...)` before capital routing commit.
  2) If `decision.active` is True, use `decision.new_weights` for allocator commit and
     submit hedges via your OMS (or pass an `oms_place_fn` callback).

All risk quantities are daily unless specified. All 'bps' refer to bps of NAV.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------- Data classes -----------------------------------

@dataclass
class PortfolioMetrics:
    var99_bps: float                # daily portfolio VaR@99 (bps)
    es97_bps: float                 # daily portfolio ES@97 (bps)
    drawdown: float                 # peak-to-trough as +fraction (0.10 => -10%)
    realized_vol_bps: float         # daily realized vol in bps of NAV
    liquidity_score: float          # 0..1 (1 = excellent, 0 = frozen book)
    data_health_score: float        # 0..1 (1 = perfect, 0 = broken feeds)

@dataclass
class CrisisConfig:
    # Activation thresholds
    tail_prob_thresh: float = 0.25      # Tail-AI: P(|r| >= 3-5σ) next 1-3d
    dd_enter: float = 0.08              # enter if drawdown ≥ 8%
    dd_exit: float = 0.05               # exit only if drawdown ≤ 5%
    var_cap_bps: float = 300.0          # absolute hard cap
    var_enter_frac: float = 0.85        # enter if VaR >= 85% of cap + other stressors
    liq_floor: float = 0.35             # enter if liquidity_score ≤ 0.35
    data_floor: float = 0.70            # enter if data_health ≤ 0.70

    # Hysteresis windows (minutes)
    sustain_enter_min: int = 10
    sustain_exit_min: int = 120

    # Actions when active
    gross_scale: float = 0.55           # scale gross exposure to 55%
    min_cash_buf: float = 0.08          # keep ≥8% NAV in cash
    per_strategy_cap: float = 0.06      # tighter cap while in crisis
    hedge_budget_bps: int = 25          # spend up to 25bps NAV on hedges
    prefer_overlays: Tuple[str, ...] = ("index_put_spread", "long_vix", "fx")

    # Proof/audit
    attach_inputs_in_proof: bool = True


@dataclass
class CrisisDecision:
    active: bool
    reason: str
    score: float
    new_weights: Dict[str, float]
    hedge_plan: List[Dict[str, Any]]
    proof: Dict[str, Any]
    ts_ms: int


# ----------------------------- Core logic -------------------------------------

def _composite_score(
    *,
    tail_prob: float,
    regime_hint: Dict[str, float],
    pm: PortfolioMetrics,
) -> float:
    """
    Blend tail probability, regime 'panic' bias (via hedge_bias > 0),
    VaR utilization, drawdown, and liquidity/data health into a 0..1 score.
    """
    # Normalize components 0..1
    var_util = min(1.0, pm.var99_bps / max(1e-6, pm.es97_bps if pm.es97_bps > 0 else pm.var99_bps))
    dd = min(1.0, pm.drawdown / 0.25)  # clamp 25% dd ⇒ 1.0
    liq = 1.0 - pm.liquidity_score     # lower liquidity ⇒ higher stress
    data = 1.0 - pm.data_health_score

    # Regime hedge bias: map -1..+1 to 0..1
    hb = float(regime_hint.get("hedge_bias", 0.0))
    hb01 = 0.5 * (hb + 1.0)

    # Weighted blend (tune as needed)
    w_tail, w_var, w_dd, w_liq, w_data, w_reg = 0.28, 0.22, 0.22, 0.12, 0.06, 0.10
    score = (
        w_tail * min(1.0, tail_prob) +
        w_var  * var_util +
        w_dd   * dd +
        w_liq  * liq +
        w_data * data +
        w_reg  * hb01
    )
    return float(np.clip(score, 0.0, 1.0))


def _should_enter(score: float, pm: PortfolioMetrics, cfg: CrisisConfig, tail_prob: float) -> Tuple[bool, str]:
    reasons = []
    if tail_prob >= cfg.tail_prob_thresh:
        reasons.append(f"tail_prob={tail_prob:.3f}>=thresh")
    if pm.drawdown >= cfg.dd_enter:
        reasons.append(f"dd={pm.drawdown:.3f}>=dd_enter")
    if (pm.var99_bps >= cfg.var_enter_frac * cfg.var_cap_bps) or (pm.var99_bps >= cfg.var_cap_bps):
        reasons.append(f"var={pm.var99_bps:.1f}bps near/cross cap")
    if pm.liquidity_score <= cfg.liq_floor:
        reasons.append(f"liq={pm.liquidity_score:.2f}<=liq_floor")
    if pm.data_health_score <= cfg.data_floor:
        reasons.append(f"data={pm.data_health_score:.2f}<=data_floor")

    if score >= 0.58 and reasons:
        return True, " & ".join(reasons) + f" | score={score:.2f}"
    return False, ""


def _should_exit(score: float, pm: PortfolioMetrics, cfg: CrisisConfig) -> Tuple[bool, str]:
    # Exit only when stress is clearly reduced
    conds = [
        score <= 0.32,
        pm.drawdown <= cfg.dd_exit,
        pm.var99_bps <= 0.70 * cfg.var_cap_bps,
        pm.liquidity_score >= 0.60,
        pm.data_health_score >= 0.90,
    ]
    if all(conds):
        return True, f"score={score:.2f}, dd={pm.drawdown:.3f}, var={pm.var99_bps:.1f}bps normalized"
    return False, ""


def _scale_weights_for_crisis(weights: Dict[str, float], gross_scale: float, min_cash: float, per_cap: float) -> Dict[str, float]:
    """
    Scale down all strategy weights proportionally, clamp per-strategy caps, and
    ensure a minimum cash buffer.
    """
    # Proportional scale
    w = {k: max(0.0, v * gross_scale) for k, v in weights.items()}
    # Per-strategy clamp
    w = {k: min(per_cap, v) for k, v in w.items()}

    # Normalize to (1 - min_cash)
    gross = sum(w.values())
    target_gross = max(0.0, 1.0 - min_cash)
    if gross > 0:
        scale = target_gross / gross
        w = {k: v * scale for k, v in w.items()}
    return w


def _derive_hedge_plan(
    *,
    hedge_bias: float,
    budget_bps: int,
    prefer_overlays: Tuple[str, ...],
    region_mix: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Construct a hedge overlay plan. If risk/dynamic_hedges is available, use it.
    Otherwise, fallback to a sensible static plan: equity index puts + long vol + FX hedge.
    """
    try:
        # Optional advanced hedger (if implemented in your stack)
        from backend.risk.dynamic_hedges import suggest_overlays  # type: ignore
        return suggest_overlays(
            budget_bps=budget_bps,
            hedge_bias=hedge_bias,
            prefer=prefer_overlays,
            region_mix=region_mix or {},
        )
    except Exception:
        # Fallback static plan
        plan: List[Dict[str, Any]] = []
        # Split budget: 60% index puts, 30% long vol, 10% FX (if bias > 0)
        put_bps = int(0.60 * budget_bps)
        vix_bps = int(0.30 * budget_bps)
        fx_bps  = max(0, budget_bps - put_bps - vix_bps)

        plan.append({"type": "index_put_spread", "notional_bps": put_bps, "tenor_d": 21, "underlyings": ["US_EQ", "EU_EQ", "JP_EQ"]})
        plan.append({"type": "long_vix", "notional_bps": vix_bps, "tenor_d": 14})
        if hedge_bias > 0.25 and fx_bps > 0:
            plan.append({"type": "fx", "pair": "USD/JPY", "direction": "long_USD", "notional_bps": fx_bps, "tenor_d": 7})
        return plan


def _proof_bundle(
    *,
    active: bool,
    reason: str,
    score: float,
    tail_prob: float,
    regime_hint: Dict[str, float],
    pm: PortfolioMetrics,
    cfg: CrisisConfig,
    weights_before: Dict[str, float],
    weights_after: Dict[str, float],
    hedge_plan: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out = {
        "ts": int(time.time() * 1000),
        "proof_type": "crisis_mode_decision",
        "verdict": "ACTIVE" if active else "INACTIVE",
        "reason": reason,
        "score": round(score, 4),
        "tail_prob": round(float(tail_prob), 4),
        "regime_hint": {k: round(float(v), 4) for k, v in regime_hint.items()},
        "portfolio": {
            "var99_bps": round(pm.var99_bps, 2),
            "es97_bps": round(pm.es97_bps, 2),
            "drawdown": round(pm.drawdown, 4),
            "realized_vol_bps": round(pm.realized_vol_bps, 2),
            "liquidity_score": round(pm.liquidity_score, 3),
            "data_health_score": round(pm.data_health_score, 3),
        },
        "weights_before_sum": round(sum(weights_before.values()), 6),
        "weights_after_sum": round(sum(weights_after.values()), 6),
        "hedge_plan": hedge_plan,
        "actions": {
            "gross_scale": cfg.gross_scale,
            "min_cash_buf": cfg.min_cash_buf,
            "per_strategy_cap": cfg.per_strategy_cap,
            "hedge_budget_bps": cfg.hedge_budget_bps,
        },
    }
    if not cfg.attach_inputs_in_proof:
        out.pop("hedge_plan", None)
    return out


# ----------------------------- Public API -------------------------------------

def decide_and_apply(
    *,
    current_weights: Dict[str, float],
    tail_prob: float,
    regime_hint: Dict[str, float],
    port_metrics: PortfolioMetrics,
    cfg: Optional[CrisisConfig] = None,
    region_mix: Optional[Dict[str, float]] = None,
    sustain_tracker: Optional[Dict[str, Any]] = None,
    ledger_path: Optional[str] = None,
    oms_place_fn: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> CrisisDecision:
    """
    Decide crisis activation and compute new weights + hedge plan.

    sustain_tracker:
      A mutable dict you persist across calls: {"enter_since": ts_ms or None, "exit_since": ts_ms or None, "active": bool}

    oms_place_fn:
      Optional callback to actually place hedges; receives hedge_plan list. If it raises, we still return the decision.

    Returns:
      CrisisDecision
    """
    cfg = cfg or CrisisConfig()
    sustain_tracker = sustain_tracker if sustain_tracker is not None else {"active": False, "enter_since": None, "exit_since": None}

    # 1) compute composite stress score
    score = _composite_score(tail_prob=tail_prob, regime_hint=regime_hint, pm=port_metrics)

    # 2) hysteresis gate
    now = int(time.time() * 1000)
    active_now = bool(sustain_tracker.get("active", False))
    enter_since = sustain_tracker.get("enter_since")
    exit_since = sustain_tracker.get("exit_since")

    do_enter, why_enter = _should_enter(score, port_metrics, cfg, tail_prob)
    do_exit,  why_exit  = _should_exit(score, port_metrics, cfg)

    if not active_now and do_enter:
        enter_since = enter_since or now
        mins = (now - enter_since) / 60000.0
        if mins >= cfg.sustain_enter_min:
            active = True
            reason = f"ENTER: {why_enter} sustained {mins:.1f}m"
            enter_since = None
            exit_since = None
        else:
            # Not yet sustained; keep inactive but remember start
            active = False
            reason = f"ARMED_ENTER: {why_enter} ({mins:.1f}m/{cfg.sustain_enter_min}m)"
    elif active_now and do_exit:
        exit_since = exit_since or now
        mins = (now - exit_since) / 60000.0
        if mins >= cfg.sustain_exit_min:
            active = False
            reason = f"EXIT: {why_exit} sustained {mins:.1f}m"
            enter_since = None
            exit_since = None
        else:
            active = True
            reason = f"HOLD_CRISE: {why_exit} ({mins:.1f}m/{cfg.sustain_exit_min}m)"
    else:
        active = active_now
        reason = "MAINTAIN_ACTIVE" if active_now else "MAINTAIN_INACTIVE"
        # reset arming timers if conditions not continuously met
        if not do_enter:
            enter_since = None
        if not do_exit:
            exit_since = None

    # 3) compute new weights & hedges if active
    weights_after = dict(current_weights)
    hedge_plan: List[Dict[str, Any]] = []
    if active:
        weights_after = _scale_weights_for_crisis(
            current_weights, gross_scale=cfg.gross_scale, min_cash=cfg.min_cash_buf, per_cap=cfg.per_strategy_cap
        )
        hedge_plan = _derive_hedge_plan(
            hedge_bias=float(regime_hint.get("hedge_bias", 0.0)),
            budget_bps=cfg.hedge_budget_bps,
            prefer_overlays=cfg.prefer_overlays,
            region_mix=region_mix,
        )

    # 4) build proof & ledger append
    proof = _proof_bundle(
        active=active,
        reason=reason,
        score=score,
        tail_prob=tail_prob,
        regime_hint=regime_hint,
        pm=port_metrics,
        cfg=cfg,
        weights_before=current_weights,
        weights_after=weights_after,
        hedge_plan=hedge_plan,
    )

    if ledger_path:
        try:
            from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
            MerkleLedger(ledger_path).append({"type": "crisis_mode", "decision": proof, "weights": weights_after})
        except Exception:
            # Audit failure is non-fatal
            pass

    # 5) optionally place hedges
    if active and oms_place_fn and hedge_plan:
        try:
            oms_place_fn(hedge_plan)
        except Exception:
            # surface via proof note, but do not crash allocator loop
            proof["hedge_submit_error"] = True

    # 6) persist sustain state back to caller
    sustain_tracker["active"] = active
    sustain_tracker["enter_since"] = enter_since
    sustain_tracker["exit_since"] = exit_since

    return CrisisDecision(
        active=active,
        reason=reason,
        score=round(score, 4),
        new_weights=weights_after,
        hedge_plan=hedge_plan,
        proof=proof,
        ts_ms=proof["ts"],
    )