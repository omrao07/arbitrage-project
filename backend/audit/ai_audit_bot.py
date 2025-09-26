# backend/allocator/risk_budget.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Iterable
import math
import time

import numpy as np
import pandas as pd

# Reuse StratMetrics definition from meta_allocator for consistency
try:
    from .meta_allocator import StratMetrics # type: ignore
except Exception:
    @dataclass
    class StratMetrics:  # fallback typing if imported standalone
        sharpe: float
        drawdown: float
        var99_bps: float
        es97_bps: float
        vol_bps: float
        live_pnl_bps: float
        pulls: int
        reward_bps: float


# ----------------------- Data structures -----------------------

@dataclass
class PortfolioRiskLimits:
    """All caps are daily in bps of NAV unless noted."""
    total_es_budget_bps: float = 220.0     # portfolio ES budget to distribute
    max_port_var_bps: float = 300.0        # hard VaR cap (guard check)
    cash_buffer: float = 0.02              # keep some dry powder

@dataclass
class RegionCaps:
    max_region_risk_bps: float = 120.0     # cap risk assigned to region
    min_region_risk_bps: float = 0.0

@dataclass
class StrategyCaps:
    max_strat_risk_bps: float = 60.0       # per-strategy ES cap
    min_strat_risk_bps: float = 0.0

@dataclass
class Inputs:
    """Container for a single snapshot of the world."""
    strat_region: Dict[str, str]                    # "US.labor_union_power" -> "US"
    metrics: Dict[str, StratMetrics]                # per strategy live metrics
    regime_hints: Dict[str, Dict[str, float]]       # region -> {"risk_multiplier","hedge_bias"}
    cov_bps2: Any                                   # strategy covariance (DataFrame or dict-of-dicts)

@dataclass
class Result:
    risk_budget_bps: Dict[str, float]               # ES budget per strategy (bps of NAV)
    weights: Dict[str, float]                       # target weights per strategy (sum <= 1 - cash)
    proofs: Dict[str, Any]                          # machine-checkable proofs
    region_risk_bps: Dict[str, float]               # region totals for visibility
    port_var_bps: float                             # resulting portfolio VaR estimate


# ----------------------- Core helpers -----------------------

def _as_dataframe_cov(cov_bps2: Any, ids: Iterable[str]) -> pd.DataFrame:
    ids = list(ids)
    if isinstance(cov_bps2, pd.DataFrame):
        return cov_bps2.reindex(index=ids, columns=ids).fillna(0.0)
    # dict-of-dicts fallback
    mat = np.zeros((len(ids), len(ids)), dtype=float)
    for i, a in enumerate(ids):
        row = cov_bps2.get(a, {}) if isinstance(cov_bps2, dict) else {}
        for j, b in enumerate(ids):
            mat[i, j] = float(row.get(b, 0.0))
    return pd.DataFrame(mat, index=ids, columns=ids)

def _portfolio_var_bps(weights: Dict[str, float], cov_bps2: pd.DataFrame) -> float:
    if not weights:
        return 0.0
    ids = list(weights.keys())
    w = np.array([weights[k] for k in ids], dtype=float)
    Sigma = cov_bps2.reindex(index=ids, columns=ids).fillna(0.0).values
    return float(np.sqrt(max(0.0, w @ Sigma @ w))) # type: ignore

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / max(1e-8, temperature)
    e = np.exp(z)
    s = e.sum()
    return e / (s if s > 0 else 1.0)

def _risk_per_weight(m: StratMetrics) -> float:
    """
    Conservative estimate of ES per unit weight (bps of NAV).
    Prefer ES; fall back to 2.33 * vol (approx VaR@99 assuming normal)
    """
    if m.es97_bps and m.es97_bps > 0:
        return float(m.es97_bps)
    if m.var99_bps and m.var99_bps > 0:
        return float(m.var99_bps)
    return float(max(1.0, 2.33 * m.vol_bps))


# ----------------------- Region scoring -----------------------

def _region_score(strat_ids: Iterable[str], metrics: Dict[str, StratMetrics], hint: Dict[str, float]) -> float:
    """
    Region desirability: mean positive risk-adjusted rewards × regime multiplier.
    Reward proxy: reward_bps / vol_bps, penalized by drawdown.
    """
    vals = []
    for sid in strat_ids:
        m = metrics[sid]
        base = (m.reward_bps / max(1.0, m.vol_bps)) * max(0.1, 1.0 - m.drawdown)
        if m.es97_bps > 0 and m.es97_bps > 120.0:
            base *= 0.3  # penalize very risky
        vals.append(max(0.0, base))
    base_score = float(np.mean(vals)) if vals else 0.0
    rm = float(hint.get("risk_multiplier", 1.0))
    return base_score * rm


# ----------------------- Public API -----------------------

def compute_risk_budgets(
    *,
    inp: Inputs,
    port_limits: PortfolioRiskLimits,
    region_caps: Dict[str, RegionCaps],
    strat_caps: StrategyCaps,
    gross_cap: float = 1.0,
) -> Result:
    """
    1) Allocate ES budget to regions by region score × regime multiplier with caps/floors.
    2) Within each region, allocate ES to strategies by positive RA-reward with caps/floors.
    3) Convert each strategy's ES budget to a target weight via its ES-per-weight.
    4) Enforce portfolio VaR cap by proportional scaling if necessary.
    """
    # Filter enabled strategies (by having metrics)
    strat_ids = [sid for sid in inp.metrics.keys() if sid in inp.strat_region]
    if not strat_ids:
        return Result({}, {}, {"verdict": "FAIL", "reason": "no_strategies"}, {}, 0.0)

    # Group by region
    by_region: Dict[str, list] = {}
    for sid in strat_ids:
        r = inp.strat_region[sid].upper()
        by_region.setdefault(r, []).append(sid)

    # Region scores
    regions = sorted(by_region.keys())
    scores = np.array([
        _region_score(by_region[r], inp.metrics, inp.regime_hints.get(r, {}))
        for r in regions
    ], dtype=float)

    if scores.sum() <= 1e-12:
        scores = np.ones_like(scores)

    raw = _softmax(scores, temperature=1.0)  # normalized 0..1
    # Apply region risk caps/floors and renormalize to total ES budget
    region_budget = {}
    tot = float(port_limits.total_es_budget_bps)
    # First pass: clamp to caps
    clamped = []
    for i, r in enumerate(regions):
        rc = region_caps.get(r, RegionCaps())
        v = float(np.clip(raw[i] * tot, rc.min_region_risk_bps, rc.max_region_risk_bps))
        region_budget[r] = v
        clamped.append(v)
    # If sum != tot, proportionally scale within cap room
    s = sum(clamped)
    if s > 0 and abs(s - tot) > 1e-6:
        scale = tot / s
        for r in regions:
            rb = region_budget[r] * scale
            # respect caps again
            rc = region_caps.get(r, RegionCaps())
            region_budget[r] = float(np.clip(rb, rc.min_region_risk_bps, rc.max_region_risk_bps))

    # Strategy risk budgets within each region
    strat_risk: Dict[str, float] = {}
    for r in regions:
        # strategy weights by positive RA-reward, penalize near ES/DD limits
        vals = []
        ids = by_region[r]
        for sid in ids:
            m = inp.metrics[sid]
            score = (m.reward_bps / max(1.0, m.vol_bps)) * max(0.1, 1.0 - m.drawdown)
            # soft penalties
            if m.es97_bps > 0.8 * strat_caps.max_strat_risk_bps:
                score *= 0.6
            vals.append(max(0.0, score))
        arr = np.array(vals, dtype=float)
        if arr.sum() <= 0:
            arr = np.ones_like(arr)

        # turn into per-strategy ES budgets
        weights = arr / arr.sum()
        region_total = region_budget[r]
        for w, sid in zip(weights, ids):
            es_bps = float(np.clip(w * region_total, strat_caps.min_strat_risk_bps, strat_caps.max_strat_risk_bps))
            strat_risk[sid] = es_bps

    # Convert ES budgets to target weights using each strategy's ES-per-weight
    weights: Dict[str, float] = {}
    for sid, es_budget in strat_risk.items():
        m = inp.metrics[sid]
        es_per_w = max(1e-6, _risk_per_weight(m))
        w = float(es_budget / es_per_w)
        weights[sid] = max(0.0, w)

    # Normalize weights to ≤ gross_cap * (1 - cash_buffer)
    gross_target = max(0.0, gross_cap * (1.0 - port_limits.cash_buffer))
    s = sum(weights.values())
    if s > 0 and s > gross_target:
        scale = gross_target / s
        weights = {k: v * scale for k, v in weights.items()}

    # Portfolio VaR guard (scale proportionally if needed)
    cov = _as_dataframe_cov(inp.cov_bps2, weights.keys())
    port_var = _portfolio_var_bps(weights, cov)
    if port_var > port_limits.max_port_var_bps:
        scale = float(port_limits.max_port_var_bps / max(1e-9, port_var))
        weights = {k: v * scale for k, v in weights.items()}
        port_var = _portfolio_var_bps(weights, cov)

    # Proof bundle
    proof = _proof_bundle(
        region_budget=region_budget,
        strat_risk=strat_risk,
        weights=weights,
        port_var_bps=port_var,
        limits=port_limits,
    )

    return Result(
        risk_budget_bps=strat_risk,
        weights=weights,
        proofs=proof,
        region_risk_bps=region_budget,
        port_var_bps=port_var,
    )


# ----------------------- Proofs & Audit -----------------------

def _proof_bundle(*, region_budget: Dict[str, float], strat_risk: Dict[str, float], weights: Dict[str, float],
                  port_var_bps: float, limits: PortfolioRiskLimits) -> Dict[str, Any]:
    ok_var = port_var_bps <= limits.max_port_var_bps + 1e-9
    gross = sum(weights.values())
    out = {
        "ts": int(time.time() * 1000),
        "proof_type": "risk_budget_allocation",
        "verdict": "PASS" if ok_var and gross <= (1.0 - limits.cash_buffer + 1e-9) else "WARN",
        "port_var_bps": round(port_var_bps, 3),
        "max_port_var_bps": limits.max_port_var_bps,
        "cash_buffer": limits.cash_buffer,
        "gross_sum": round(gross, 6),
        "region_budget_bps": {k: round(v, 3) for k, v in region_budget.items()},
        "strat_risk_bps": {k: round(v, 3) for k, v in strat_risk.items()},
    }
    return out


# ----------------------- Convenience: with ledger -----------------------

def compute_with_ledger(
    *,
    inp: Inputs,
    port_limits: PortfolioRiskLimits,
    region_caps: Dict[str, RegionCaps],
    strat_caps: StrategyCaps,
    ledger_path: str,
    gross_cap: float = 1.0,
) -> Result:
    res = compute_risk_budgets(
        inp=inp,
        port_limits=port_limits,
        region_caps=region_caps,
        strat_caps=strat_caps,
        gross_cap=gross_cap,
    )
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "risk_budget", "proofs": res.proofs, "weights": res.weights})
    except Exception:
        pass
    return res