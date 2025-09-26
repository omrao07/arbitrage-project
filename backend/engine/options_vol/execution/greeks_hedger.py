# engines/options/hedging/greeks_hedger.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioGreeks:
    """
    Portfolio-level Greeks in currency units (after multipliers & contracts).
    """
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0         # per 1.00 volatility (not 1 vol-point)
    theta: float = 0.0        # per year
    vanna: float = 0.0
    vomma: float = 0.0

    def as_vector(self, fields: List[str]) -> np.ndarray:
        return np.array([getattr(self, f, 0.0) for f in fields], dtype=float)

@dataclass(frozen=True)
class HedgeInstrument:
    """
    One instrument available for hedging (underlying share, future, or option).
    greeks_* are PER UNIT of this instrument (1 share, 1 contract).
      - For options, 'unit' means 1 contract (including multiplier effect).
      - For underlying, gamma/vega/theta are ~0; delta ≈ 1 * multiplier.
    """
    symbol: str
    price: float                            # mid price (for notional/cost)
    greeks: Dict[str, float]                # e.g. {"delta": 50.0, "gamma": 120.0, "vega": 250.0, "theta": -30.0}
    min_qty: float = -np.inf                # lower bound (negative = short allowed)
    max_qty: float = np.inf                 # upper bound
    lot_size: float = 1.0                   # e.g., 1 share, 1 contract, 100 shares lot, etc.
    trade_cost_bps: float = 0.0             # symmetric round-trip cost bps of notional for regularization weight

    def greeks_vector(self, fields: List[str]) -> np.ndarray:
        return np.array([self.greeks.get(f, 0.0) for f in fields], dtype=float)

# ---------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------

def _project_to_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

def _round_to_lots(x: np.ndarray, lots: np.ndarray) -> np.ndarray:
    lots = np.where(lots <= 0, 1.0, lots)
    return np.round(x / lots) * lots

def _ridge_ls(G: np.ndarray, b: np.ndarray, W: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """
    Solve: minimize || W (G x - b) ||^2 + x^T L2 x
    → (G^T W^T W G + L2) x = G^T W^T W b
    """
    GTWG = G.T @ (W @ W) @ G
    rhs  = G.T @ (W @ W) @ b
    A = GTWG + L2
    # Stable solve
    return np.linalg.solve(A, rhs)

def build_hedge(
    *,
    portfolio: PortfolioGreeks,
    instruments: List[HedgeInstrument],
    greek_fields: List[str] = ("delta","gamma","vega","theta"), # type: ignore
    tolerances: Dict[str, float] = None,      # e.g., {"delta": 1e-2, "gamma": 50, "vega": 200, "theta": 50} # type: ignore
    cost_weight: float = 1e-6,                # weight on notional cost (as L2 on qty scaled by price)
    qty_l2_weight: float = 1e-9,              # tiny L2 on raw size (numerical regularizer)
    max_iters: int = 8,                       # projection/rounding refinement passes
    enforce_lots: bool = True,
    integer_after_projection: bool = True,
) -> Dict[str, object]:
    """
    Compute hedge quantities for the given instruments to offset the portfolio Greeks.
    Approach:
      - Solve weighted ridge least-squares to match -portfolio Greeks across selected fields.
      - Add L2 on (price * qty) to discourage expensive hedges (cost_weight).
      - Project into box bounds; optionally round to lot sizes; re-solve locally a few times.

    Returns:
      {
        'qty': Series (per symbol),
        'post_greeks': dict (remaining greek exposure after hedge),
        'matched_error': Series (per greek),
        'cost_notional$': float (approx notional traded),
        'G': DataFrame (instrument greek matrix),
      }
    """
    if tolerances is None:
        tolerances = {f: 1.0 for f in greek_fields}

    # Assemble matrices
    n = len(instruments)
    m = len(greek_fields)

    G = np.vstack([inst.greeks_vector(list(greek_fields)) for inst in instruments]).T  # (m x n)
    b = -portfolio.as_vector(list(greek_fields))  # target greek change

    # Weight residuals by inverse tolerance so tighter tolerance gets higher weight
    tol_vec = np.array([max(1e-12, float(tolerances.get(f, 1.0))) for f in greek_fields], dtype=float)
    W = np.diag(1.0 / tol_vec)

    # Cost-aware L2 (scaled by price & optional trade_cost_bps)
    prices = np.array([inst.price for inst in instruments], dtype=float)
    cost_scale = (prices / max(np.nanmedian(prices), 1e-9))  # type: ignore # scale-invariant
    bps_vec = np.array([inst.trade_cost_bps for inst in instruments], dtype=float)
    L2_diag = qty_l2_weight + cost_weight * (cost_scale**2) * (1.0 + bps_vec*1e-4)
    L2 = np.diag(L2_diag)

    # Unconstrained ridge solution
    x = _ridge_ls(G, b, W, L2)

    # Iterate projection → local re-solve in active set
    lo = np.array([inst.min_qty for inst in instruments], dtype=float)
    hi = np.array([inst.max_qty for inst in instruments], dtype=float)
    lots = np.array([inst.lot_size for inst in instruments], dtype=float)

    for _ in range(max_iters):
        x = _project_to_bounds(x, lo, hi)
        if enforce_lots:
            x = _round_to_lots(x, lots)
        # Active set: variables not pinned to bounds (within half lot)
        pinned = (np.isclose(x, lo, atol=0.5*lots)) | (np.isclose(x, hi, atol=0.5*lots))
        free_idx = np.where(~pinned)[0]
        if free_idx.size == 0:
            break
        Gf = G[:, free_idx]
        L2f = np.diag(L2_diag[free_idx])
        # residual target after pinned components
        r = b - G @ x
        # solve for update on free vars
        dx = _ridge_ls(Gf, r, W, L2f)
        x[free_idx] += dx

    if integer_after_projection and enforce_lots:
        x = _round_to_lots(_project_to_bounds(x, lo, hi), lots)

    # Build outputs
    symbols = [inst.symbol for inst in instruments]
    qty = pd.Series(x, index=symbols, dtype=float)

    # Post-hedge greeks & errors
    achieved = G @ x
    err = achieved - b   # residual = Gx - b → how far from target
    matched = pd.Series(err, index=greek_fields, dtype=float)

    # Remaining portfolio exposure (post-hedge)
    remaining = portfolio.as_vector(list(greek_fields)) + (-b + achieved)  # = initial + delta_from_hedge
    post = {g: float(v) for g, v in zip(greek_fields, remaining)}

    # Notional proxy for cost
    notionals = np.abs(qty.values) * prices # type: ignore
    approx_cost = float(np.sum(notionals * (bps_vec * 1e-4)))

    return {
        "qty": qty.sort_index(),
        "post_greeks": post,
        "matched_error": matched,
        "cost_notional$": float(notionals.sum()),
        "approx_tc$": approx_cost,
        "G": pd.DataFrame(G, index=greek_fields, columns=symbols),
    }

# ---------------------------------------------------------------------
# Convenience: build PortfolioGreeks from option book snapshot
# ---------------------------------------------------------------------

def portfolio_greeks_from_book(
    *,
    positions: pd.Series,                  # contracts by option_id (can include 'UNDERLYING' row for shares)
    greeks_table: pd.DataFrame,            # rows: option_id (and/or 'UNDERLYING'), cols: ['delta','gamma','vega','theta','vanna','vomma']
    multipliers: Dict[str, float] = None,  # option_id → contract multiplier (e.g., 100 for US equity options) # type: ignore
) -> PortfolioGreeks:
    if multipliers is None:
        multipliers = {}
    aligned = greeks_table.reindex(positions.index).fillna(0.0)
    mult = pd.Series({k: multipliers.get(k, 1.0) for k in positions.index})
    # Total Greeks = Σ (contracts * multiplier * greek_per_unit)
    totals = (aligned.mul(positions, axis=0).mul(mult, axis=0)).sum()
    return PortfolioGreeks(
        delta=float(totals.get("delta", 0.0)),
        gamma=float(totals.get("gamma", 0.0)),
        vega=float(totals.get("vega", 0.0)),
        theta=float(totals.get("theta", 0.0)),
        vanna=float(totals.get("vanna", 0.0)),
        vomma=float(totals.get("vomma", 0.0)),
    )

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example: hedge SPX book with shares + two options
    # Portfolio exposures (already in $ greeks):
    port = PortfolioGreeks(delta=+250_000, gamma=-8_000, vega=+50_000, theta=-7_500)

    # Hedging instruments:
    inst = [
        # Underlying (SPY proxy): delta ≈ +100 per 1-lot (100 shares)
        HedgeInstrument(symbol="SPY_shares", price=550.0, greeks={"delta": 100.0}, min_qty=-50_000, max_qty=50_000, lot_size=1, trade_cost_bps=0.5),
        # Short-dated call: +delta, +gamma, +vega, negative theta
        HedgeInstrument(symbol="C_atm_M1", price=25.0, greeks={"delta": 55.0, "gamma": 600.0, "vega": 900.0, "theta": -40.0}, min_qty=-5_000, max_qty=5_000, lot_size=1.0, trade_cost_bps=8.0),
        # Short-dated put: -delta, +gamma, +vega, negative theta
        HedgeInstrument(symbol="P_atm_M1", price=24.0, greeks={"delta": -45.0, "gamma": 650.0, "vega": 850.0, "theta": -38.0}, min_qty=-5_000, max_qty=5_000, lot_size=1.0, trade_cost_bps=8.0),
    ]

    targets = ("delta","gamma","vega","theta")
    tol = {"delta": 1_000.0, "gamma": 200.0, "vega": 500.0, "theta": 200.0}

    out = build_hedge(
        portfolio=port,
        instruments=inst,
        greek_fields=list(targets),
        tolerances=tol,
        cost_weight=1e-8,
        qty_l2_weight=1e-10,
        enforce_lots=True,
        integer_after_projection=True,
    )

    print("Hedge quantities:\n", out["qty"])
    print("\nPost-hedge greeks:\n", out["post_greeks"])
    print("\nMatched error:\n", out["matched_error"])
    print("\nApprox traded notional (USD):", round(out["cost_notional$"], 2)) # type: ignore