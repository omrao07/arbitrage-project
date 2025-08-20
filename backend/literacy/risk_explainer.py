# backend/literacy/risk_explainer.py
"""
Risk Explainer
--------------
Transforms quantitative risk metrics into human-friendly explanations
for your Literacy Mode and dashboards.

Inputs (any subset):
- risk_snapshot: dict from RiskMetrics.snapshot() or similar:
    {
      "total": {"sharpe":..., "vol":..., "max_drawdown":..., "var_95":...},
      "mean_rev": {...}, ...
    }
  or just a dict[strategy]->metrics dict.

- pnl_snapshot: dict from PnLAttributor.snapshot()

- exposures:
    {
      "regions": {"india": 0.55, "us": 0.45},
      "sectors": {"tech": 0.40, "energy": 0.10, ...},
      "gross": 1.25,     # sum |weights|
      "net": 0.35        # long - short
    }

- leverage: float (assets/equity). If missing, inferred from exposures["gross"] when sensible.

Public:
- explain(risk_snapshot, pnl_snapshot=None, exposures=None, leverage=None) -> dict
- scenario_shock(prices, positions, shocks) -> dict
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple


# ---------------- thresholds (tweak to taste) ----------------

DD_WARN = -0.03       # drawdown more negative than this -> warning
DD_ALERT = -0.08
VAR_WARN = 0.02       # daily VaR >= 2% of equity -> warning
VAR_ALERT = 0.04
VOL_WARN = 0.25       # annualized vol >= 25% -> warning
VOL_ALERT = 0.40
LEVERAGE_WARN = 1.5
LEVERAGE_ALERT = 2.5
CONC_WARN = 0.60      # >60% in a single region/sector -> warning
CONC_ALERT = 0.75

# ---------------- helpers ----------------

def _lvl(value: float, warn: float, alert: float, reverse: bool = False) -> str:
    """
    Map a metric to 'ok'|'warning'|'alert'.
    reverse=True means lower is worse (not used here).
    """
    v = -value if reverse else value
    if v >= alert:
        return "alert"
    if v >= warn:
        return "warning"
    return "ok"

def _pct(x: float) -> str:
    return f"{x*100:.2f}%"

def _safe(d: Optional[Dict], *path, default=None):
    cur = d or {}
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# ---------------- main API ----------------

def explain(
    risk_snapshot: Optional[Dict[str, Any]] = None,
    pnl_snapshot: Optional[Dict[str, Any]] = None,
    exposures: Optional[Dict[str, Dict[str, float]]] = None,
    leverage: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return structured, plain-English risk explanations:
    {
      "headline": "...",
      "sections": [
        {"title": "Drawdown", "level": "warning", "detail": "...", "tip": "..."},
        {"title": "VaR", ...},
        {"title": "Volatility", ...},
        {"title": "Leverage", ...},
        {"title": "Concentration", ...}
      ],
      "summary": "two-line summary"
    }
    """
    # Try to extract a "total" metrics dict
    total = None
    if risk_snapshot:
        if "total" in risk_snapshot and isinstance(risk_snapshot["total"], dict):
            total = risk_snapshot["total"]
        else:
            # average across strategies if no explicit total
            vals = [v for v in risk_snapshot.values() if isinstance(v, dict)]
            if vals:
                import statistics as _st
                total = {
                    "sharpe": _st.fmean([v.get("sharpe", 0.0) for v in vals]),
                    "vol": _st.fmean([v.get("vol", 0.0) for v in vals]),
                    "max_drawdown": _st.fmean([v.get("max_drawdown", 0.0) for v in vals]),
                    "var_95": _st.fmean([v.get("var_95", 0.0) for v in vals]),
                }
    total = total or {}

    dd = float(total.get("max_drawdown", 0.0) or 0.0)  # negative number (e.g., -0.12)
    var = abs(float(total.get("var_95", 0.0) or 0.0))  # positive magnitude
    vol = float(total.get("vol", 0.0) or 0.0)          # annualized

    # Leverage inference if not provided
    if leverage is None and exposures and isinstance(exposures.get("gross"), (int, float)):
        leverage = float(exposures["gross"])
    lev = float(leverage or 1.0)

    # Concentration check (region + sector)
    conc_items = []
    for bucket_name in ("regions", "sectors"):
        b = exposures.get(bucket_name) if exposures else None
        if isinstance(b, dict) and b:
            top_key, top_w = max(b.items(), key=lambda kv: kv[1])
            conc_items.append((bucket_name, top_key, float(top_w)))
    conc_items = conc_items or [("regions", None, 0.0)]

    # -------- Build sections --------
    sections = []

    # Drawdown
    dd_level = "ok"
    if dd <= DD_ALERT:
        dd_level = "alert"
    elif dd <= DD_WARN:
        dd_level = "warning"
    sections.append({
        "title": "Drawdown",
        "level": dd_level,
        "detail": f"Max drawdown is {dd:.2%} from the peak.",
        "tip": "Drawdown measures the worst fall from a prior high. Reduce size or diversify to tame large drawdowns."
    })

    # VaR
    var_level = _lvl(var, VAR_WARN, VAR_ALERT)
    sections.append({
        "title": "Value-at-Risk (95%)",
        "level": var_level,
        "detail": f"Estimated 1‑day loss at 95% confidence is about {_pct(var)} of equity.",
        "tip": "VaR grows when volatility rises or leverage increases. Keep daily VaR within your comfort band."
    })

    # Volatility
    vol_level = _lvl(vol, VOL_WARN, VOL_ALERT)
    sections.append({
        "title": "Volatility (annualized)",
        "level": vol_level,
        "detail": f"Annualized volatility ~ {_pct(vol)}.",
        "tip": "Higher volatility means larger swings. Consider lowering gross exposure or hedging highly volatile names."
    })

    # Leverage
    lev_level = _lvl(lev, LEVERAGE_WARN, LEVERAGE_ALERT)
    sections.append({
        "title": "Leverage",
        "level": lev_level,
        "detail": f"Effective leverage ≈ {lev:.2f}× gross exposure.",
        "tip": "Leverage amplifies both gains and losses. Keep dry powder for adverse moves; avoid cascading margin calls."
    })

    # Concentration
    conc_level = "ok"
    conc_detail = "No significant concentration detected."
    conc_tip = "Diversification across regions and sectors helps reduce idiosyncratic shocks."
    for bucket, key, w in conc_items:
        if w >= CONC_ALERT:
            conc_level = "alert"
            conc_detail = f"Very concentrated: {bucket[:-1].capitalize()} '{key}' is {_pct(w)} of portfolio."
            conc_tip = "Trim positions in the concentrated bucket and reallocate to independent exposures."
            break
        elif w >= CONC_WARN:
            conc_level = "warning"
            conc_detail = f"High concentration: {bucket[:-1].capitalize()} '{key}' is {_pct(w)} of portfolio."
            conc_tip = "Consider redistributing weight to reduce single‑bucket risk."
            break
    sections.append({
        "title": "Concentration",
        "level": conc_level,
        "detail": conc_detail,
        "tip": conc_tip
    })

    # Headline + summary
    levels = [s["level"] for s in sections]
    if "alert" in levels:
        headline = "Risk elevated — action recommended"
    elif "warning" in levels:
        headline = "Risk higher than usual — monitor closely"
    else:
        headline = "Risk well‑balanced"

    summary = _make_summary(dd, var, vol, lev, conc_items, levels)

    return {
        "headline": headline,
        "sections": sections,
        "summary": summary,
    }


# ---------------- scenario shocks ----------------

def scenario_shock(
    prices: Dict[str, float],
    positions: Dict[str, float],
    shocks: Dict[str, float],
) -> Dict[str, Any]:
    """
    Apply simple percentage shocks to prices and estimate PnL impact.

    Args:
      prices: {symbol: last_price}
      positions: {symbol: qty}  (positive long, negative short)
      shocks: either
        - {"ALL": -0.02}  (apply to all symbols)
        - {"india": -0.03, "us": -0.01} with symbol prefixes like "RELIANCE.NS" -> infer region from suffix
        - {"RELIANCE.NS": -0.05, "AAPL": -0.02} per-symbol overrides

    Returns: {"pnl": float, "by_symbol": {sym: pnl}, "assumptions": {...}}
    """
    def _region(sym: str) -> str:
        if sym.endswith(".NS"): return "india"
        if sym.endswith(".BO"): return "india"
        return "us"  # naive default

    by_symbol = {}
    total = 0.0
    for sym, qty in positions.items():
        px = prices.get(sym)
        if px is None:
            continue
        shock = 0.0
        if sym in shocks:
            shock = shocks[sym]
        elif "ALL" in shocks:
            shock = shocks["ALL"]
        else:
            reg = _region(sym)
            if reg in shocks:
                shock = shocks[reg]
        new_px = px * (1.0 + shock)
        pnl = (new_px - px) * qty
        by_symbol[sym] = pnl
        total += pnl

    return {
        "pnl": total,
        "by_symbol": by_symbol,
        "assumptions": {"shocks": shocks},
    }


# ---------------- summary writer ----------------

def _make_summary(dd: float, var: float, vol: float, lev: float, conc_items, levels) -> str:
    bits = []
    if dd <= DD_WARN:
        bits.append(f"Drawdown {dd:.1%}")
    if var >= VAR_WARN:
        bits.append(f"VaR ~ {_pct(var)}")
    if vol >= VOL_WARN:
        bits.append(f"Vol {_pct(vol)}")
    if lev >= LEVERAGE_WARN:
        bits.append(f"Lev {lev:.1f}×")
    for bucket, key, w in conc_items:
        if w >= CONC_WARN:
            bits.append(f"Concentration {_pct(w)} in {key}")
            break
    if not bits:
        return "Risks look balanced with moderate volatility and diversified exposures."
    return " | ".join(bits)