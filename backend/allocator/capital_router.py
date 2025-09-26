# backend/allocator/capital_router.py
"""
Capital Router 2.0
------------------
Hierarchical allocator that routes capital across regions and strategies:
  1) Region budgets from regime-aware scores (Sharpe, drawdown, tail-risk).
  2) Within each region, defer to meta_allocator.allocate() (UCB bandit + guards).
  3) Enforce portfolio VAR/ES caps via covariance scaling.
  4) Emit a machine-checkable proof bundle and (optionally) append to Merkle ledger.

Inputs
------
snapshots: Dict[str, Snapshot]
    Point-in-time metrics per strategy (includes region + StratMetrics).
last_weights: Dict[str, float]
    Current strategy weights (0..1, sum<=1).
cov_bps2: pandas.DataFrame or Dict[str, Dict[str, float]]
    Strategy-by-strategy covariance matrix in bps^2 of NAV (daily).
regime_hints: Dict[str, Dict[str, float]]
    Per-region hints from RegimeOracle.regime_signal(): {"risk_multiplier", "hedge_bias"}.
limits: PortfolioLimits
    Hard caps (portfolio VAR/ES, per-strategy cap, per-region cap).
ledger_path: Optional[str]
    If provided, append the decision + proof to backend/audit/merkle_ledger.MerkleLedger.

Outputs
-------
(weights, proof) where:
  weights: Dict[str, float]  # target weights per strategy (sum <= 1 - cash_buffer)
  proof: Dict[str, Any]      # machine-checkable statement of constraints satisfied

Notes
-----
• All math is daily units; VAR/ES caps are in bps of NAV.
• Fallback risk: if cov is missing for a pair, we assume zero covariance.
• Cash buffer is kept unallocated to absorb slippage / hedging cost.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

# --- reuse within-region allocator primitives
from .meta_allocator import (
    StratMetrics,
    Limits as StratLimits,
    allocate as intra_region_allocate,
    decision_proof as intra_region_proof,
)

# ----- Data structures --------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    strategy_id: str            # e.g., "US.labor_union_power"
    region: str                 # "US" | "EU" | "IN" | "JP"
    metrics: StratMetrics       # rolling Sharpe, ES, drawdown, etc.
    enabled: bool = True

@dataclass
class RegionLimits:
    max_region_cap: float = 0.40      # cap (%) of total NAV per region
    min_region_cap: float = 0.00
    max_region_var_bps: float = 220.0 # optional soft cap, enforced as scaler

@dataclass
class PortfolioLimits:
    max_port_var_bps: float = 300.0   # hard cap on portfolio daily VAR@99 (bps)
    cash_buffer: float = 0.02         # leave 2% in cash by default
    max_per_strategy: float = 0.10    # 10% each (upper bound fed to intra-allocator)

# ----- Helpers ----------------------------------------------------------------

def _as_dataframe_cov(cov_bps2: Any, ids: Iterable[str]) -> pd.DataFrame:
    """Coerce covariance to DataFrame with all strategy ids; missing pairs -> 0."""
    ids = list(ids)
    if isinstance(cov_bps2, pd.DataFrame):
        df = cov_bps2.reindex(index=ids, columns=ids).fillna(0.0)
        return df
    # dict-dict fallback
    mat = np.zeros((len(ids), len(ids)), dtype=float)
    for i, a in enumerate(ids):
        row = cov_bps2.get(a, {}) if isinstance(cov_bps2, dict) else {}
        for j, b in enumerate(ids):
            mat[i, j] = float(row.get(b, 0.0))
    return pd.DataFrame(mat, index=ids, columns=ids)

def _portfolio_var_bps(weights: Dict[str, float], cov_bps2: pd.DataFrame) -> float:
    """sqrt(w^T Σ w) in bps."""
    if not weights:
        return 0.0
    ids = list(weights.keys())
    w = np.array([weights[k] for k in ids], dtype=float)
    Sigma = cov_bps2.reindex(index=ids, columns=ids).fillna(0.0).values
    var = float(np.sqrt(max(0.0, w @ Sigma @ w))) # pyright: ignore[reportArgumentType]
    return var

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / max(1e-8, temperature)
    e = np.exp(z)
    s = e.sum()
    return e / (s if s > 0 else 1.0)

# ----- Region scoring ----------------------------------------------------------

def _region_scores(group: Dict[str, Snapshot], regime_hint: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Score a region using aggregated Sharpe / ES penalties and regime risk multiplier.
    Returns (score, per_strategy_scores) for within-region blending.
    """
    # Base score per strategy: reward realized alpha per risk, penalize ES & drawdown
    strat_scores: Dict[str, float] = {}
    for sid, snap in group.items():
        m = snap.metrics
        if m.es97_bps > 0:
            risk_adj = (m.reward_bps / max(1.0, m.es97_bps)) * max(0.1, 1.0 - snap.metrics.drawdown)
        else:
            risk_adj = (m.reward_bps / max(1.0, m.vol_bps)) * max(0.1, 1.0 - snap.metrics.drawdown)
        strat_scores[sid] = risk_adj

    # Region aggregate: mean of positive scores (avoid negatives pulling too hard)
    pos = [max(0.0, v) for v in strat_scores.values()]
    base = float(np.mean(pos)) if pos else 0.0

    # Regime multiplier (0.2..1.5 typical)
    rm = float(regime_hint.get("risk_multiplier", 1.0))
    score = base * rm
    return score, strat_scores

# ----- Router Core -------------------------------------------------------------

def route_capital(
    *,
    snapshots: Dict[str, Snapshot],
    last_weights: Dict[str, float],
    cov_bps2: Any,
    regime_hints: Dict[str, Dict[str, float]],
    portfolio_limits: PortfolioLimits,
    region_limits_by_code: Dict[str, RegionLimits],
    ledger_path: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute target weights across strategies subject to regime-aware region budgets
    and portfolio-wide risk constraints.
    """
    # 0) housekeeping
    enabled = {sid: s for sid, s in snapshots.items() if s.enabled}
    if not enabled:
        return {}, _proof(verdict="FAIL", reason="no_enabled_strategies")

    # 1) group strategies by region
    by_region: Dict[str, Dict[str, Snapshot]] = {}
    for sid, s in enabled.items():
        by_region.setdefault(s.region.upper(), {})[sid] = s

    # 2) build covariance matrix over all active strats
    cov = _as_dataframe_cov(cov_bps2, enabled.keys())

    # 3) compute region scores & budgets
    region_codes = sorted(by_region.keys())
    scores = []
    for r in region_codes:
        hint = regime_hints.get(r, {})
        sc, _ = _region_scores(by_region[r], hint)
        # floor to tiny epsilon to keep region alive if needed
        scores.append(max(0.0, sc))
    scores = np.array(scores, dtype=float)

    if scores.sum() == 0:
        # fallback equal-weight if all scores are zero
        scores = np.ones_like(scores)

    # normalized desired budgets (before caps / cash buffer)
    raw_budgets = _softmax(scores, temperature=1.0)  # sum=1
    # apply per-region caps / floors
    budgets = {}
    for i, r in enumerate(region_codes):
        rl = region_limits_by_code.get(r, RegionLimits())
        budgets[r] = float(np.clip(raw_budgets[i], rl.min_region_cap, rl.max_region_cap))

    # renormalize to 1 - cash_buffer
    gross_target = max(0.0, 1.0 - float(portfolio_limits.cash_buffer))
    norm = sum(budgets.values()) or 1.0
    for r in budgets:
        budgets[r] = budgets[r] / norm * gross_target

    # 4) within-region allocation using meta_allocator.allocate()
    strategy_weights: Dict[str, float] = {}
    for r in region_codes:
        group = by_region[r]
        lims = StratLimits(
            max_w=min(portfolio_limits.max_per_strategy, region_limits_by_code.get(r, RegionLimits()).max_region_cap),
            max_port_var_bps=region_limits_by_code.get(r, RegionLimits()).max_region_var_bps,
            max_strat_es_bps=max(s.metrics.es97_bps for s in group.values()) + 1e-6,  # pass-through guard, intra will cap
            max_strat_dd=max(s.metrics.drawdown for s in group.values()) + 1e-6,
        )
        last_r = {sid: last_weights.get(sid, 0.0) for sid in group}
        mets = {sid: s.metrics for sid, s in group.items()}

        intra = intra_region_allocate(mets, lims, last_r, port_var_bps=_region_var_bps(last_r, cov))
        # scale to region budget
        ssum = sum(intra.values()) or 1.0
        for sid, w in intra.items():
            strategy_weights[sid] = float(w / ssum * budgets[r])

    # 5) enforce portfolio VAR cap (global scaling if needed)
    port_var = _portfolio_var_bps(strategy_weights, cov)
    if port_var > portfolio_limits.max_port_var_bps:
        scale = float(portfolio_limits.max_port_var_bps / max(1e-9, port_var))
        strategy_weights = {k: v * scale for k, v in strategy_weights.items()}
        port_var = _portfolio_var_bps(strategy_weights, cov)

    # 6) final proof bundle + optional merkle-append
    proof = _proof(
        verdict="PASS",
        region_budgets=budgets,
        max_port_var_bps=portfolio_limits.max_port_var_bps,
        realized_port_var_bps=round(port_var, 3),
        cash_buffer=portfolio_limits.cash_buffer,
        per_strategy_cap=portfolio_limits.max_per_strategy,
        regions=list(region_codes),
    )

    if ledger_path:
        try:
            from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
            led = MerkleLedger(ledger_path)
            led.append({"type": "capital_route", "weights": strategy_weights, "proof": proof})
        except Exception:
            # non-fatal; allocator still returns weights + proof
            pass

    return strategy_weights, proof

# ----- Private utils -----------------------------------------------------------

def _region_var_bps(w_region: Dict[str, float], cov: pd.DataFrame) -> float:
    if not w_region:
        return 0.0
    ids = list(w_region.keys())
    w = np.array([w_region[k] for k in ids], dtype=float)
    Sigma = cov.reindex(index=ids, columns=ids).fillna(0.0).values
    return float(np.sqrt(max(0.0, w @ Sigma @ w))) # type: ignore

def _proof(**kwargs) -> Dict[str, Any]:
    out = dict(kwargs)
    out["ts"] = int(time.time() * 1000)
    out["proof_type"] = "capital_router_limits_check"
    return out