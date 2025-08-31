# backend/risk/lvar.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Literal

Side = Literal["buy", "sell"]

# ---------------------------------------------------------------------
# Inputs / Profiles
# ---------------------------------------------------------------------

@dataclass
class LiquidityProfile:
    """
    Inputs you usually know or can estimate per symbol:
      • adv           : average daily volume (shares or contracts)
      • spread_bps    : quoted spread in bps of mid (typical)
      • vol_day       : daily volatility of returns (stdev, e.g. 0.02 = 2%)
      • fee_bps       : commissions + taxes, in bps notional
      • impact_eta    : temporary impact coefficient (bps per 1.0 participation)
      • impact_gamma  : permanent impact coefficient (bps per 1.0 participation)
    """
    adv: float
    spread_bps: float
    vol_day: float
    fee_bps: float = 0.0
    impact_eta: float = 25.0
    impact_gamma: float = 5.0

@dataclass
class SlicePlan:
    """
    Slicing plan for liquidation/execution.
      • horizon_min  : total liquidation horizon in minutes
      • slices       : number of equal time slices (n child orders)
      • part_cap     : max participation of ADV per *slice* (0..1)
    """
    horizon_min: int
    slices: int
    part_cap: float = 0.15  # e.g., max 15% of expected volume per slice

    def dt_min(self) -> float:
        return max(1.0, self.horizon_min / max(1, self.slices))

# ---------------------------------------------------------------------
# Cost components (Almgren–Chriss style + frictions)
# ---------------------------------------------------------------------

def half_spread_cost_bps(spread_bps: float) -> float:
    """
    Expected crossing cost ≈ half the spread (in bps).
    We keep it simple and let the stochastic part sit in the variance term.
    """
    return max(0.0, 0.5 * spread_bps)

def temporary_impact_bps(eta: float, participation: float) -> float:
    """
    Temporary impact (bps) per slice; linear in participation for robustness.
    """
    return max(0.0, eta * participation)

def permanent_impact_bps(gamma: float, participation: float) -> float:
    """
    Permanent impact (bps) accrued as you move through the book/day.
    """
    return max(0.0, gamma * participation)

def schedule_variance_bps(vol_day: float, horizon_min: float) -> float:
    """
    Variance proxy of price noise over execution horizon (in bps).
    Uses sqrt-time scaling from daily vol to horizon:
      σ_h ≈ vol_day * sqrt(horizon_min / (6.5h * 60))
    Returns *bps* stdev (not variance) for convenience.
    """
    day_min = 6.5 * 60.0  # equities trading day; edit if needed
    sigma_h = vol_day * math.sqrt(max(1e-9, horizon_min / day_min))
    return sigma_h * 1e4  # to bps

# ---------------------------------------------------------------------
# LVaR core (parametric)
# ---------------------------------------------------------------------

@dataclass
class LVaRBreakdown:
    """
    All cost numbers are expressed in *bps of notional* for comparability.
    """
    expected_bps: float            # E[cost] across slices
    stdev_bps: float               # σ(cost) across slices (noise)
    lv_ar_bps: float               # LVaR_α (positive = loss bps)
    total_fees_bps: float          # commissions/taxes
    tmp_impact_bps: float          # expected temp impact component
    perm_impact_bps: float         # permanent impact component
    spread_cross_bps: float        # spread half-cost component
    alpha: float
    horizon_min: float
    slices: int
    participation_path: List[float]

def lvar_parametric(
    *,
    side: Side,
    qty: float,
    px: float,
    profile: LiquidityProfile,
    plan: SlicePlan,
    alpha: float = 0.975,
    expected_volume_per_min: Optional[float] = None
) -> LVaRBreakdown:
    """
    Parametric LVaR under simple execution + price noise assumptions.

    Model:
      cost_per_slice_bps ≈ half_spread + eta * part + gamma * cum_part   (+/- noise)
      noise stdev over horizon in bps ≈ schedule_variance_bps(vol_day, H)

    Participation path:
      per-slice expected mkt volume = ADV / day_min * dt_min
      cap trade size per slice to (part_cap * expected slice volume)
      -> realize a per-slice participation p_i and cumulative c_i
    """
    assert qty >= 0 and px > 0
    H = float(plan.horizon_min)
    n = int(max(1, plan.slices))
    dt = plan.dt_min()

    # Expected market volume per minute
    day_min = 6.5 * 60.0
    evpm = expected_volume_per_min or (profile.adv / day_min)
    slice_mkt_vol = evpm * dt

    # Build participation schedule: equal-size child orders, capped by part_cap per slice
    child = qty / n
    parts: List[float] = []
    traded_total = 0.0
    for i in range(n):
        cap_qty = plan.part_cap * slice_mkt_vol
        q_i = min(child, cap_qty)
        # If we still have leftover not executable within cap, push to later slices equally
        # (simple pass; in practice you’d re-solve/expand horizon).
        traded_total += q_i
        parts.append(min(1.0, q_i / max(1.0, slice_mkt_vol)))

    # If capped too hard, we might not liquidate completely; warn via participation list sum
    # (we still price the portion executed).
    executed_qty = min(qty, sum(p * slice_mkt_vol for p in parts))

    # Costs (bps)
    spread_bps = half_spread_cost_bps(profile.spread_bps)
    tmp_bps = 0.0
    perm_bps = 0.0
    cum_part = 0.0
    for p in parts:
        tmp_bps += temporary_impact_bps(profile.impact_eta, p)
        cum_part += p
        perm_bps += permanent_impact_bps(profile.impact_gamma, cum_part / max(1e-9, n))  # amortize across slices

    # Expected cost is additive; fees once on executed notional
    exp_bps = spread_bps + (tmp_bps + perm_bps) / max(1, n) + profile.fee_bps

    # Noise: treat it as Normal with stdev equal to price noise over horizon
    stdev_exec_bps = schedule_variance_bps(profile.vol_day, H)

    # Quantile
    p = 1.0 - float(alpha)
    z = _inv_norm_cdf(1.0 - p)  # right tail for loss (positive cost)
    lv_ar_bps = exp_bps + z * stdev_exec_bps

    return LVaRBreakdown(
        expected_bps=exp_bps,
        stdev_bps=stdev_exec_bps,
        lv_ar_bps=lv_ar_bps,
        total_fees_bps=profile.fee_bps,
        tmp_impact_bps=tmp_bps / max(1, n),
        perm_impact_bps=perm_bps / max(1, n),
        spread_cross_bps=spread_bps,
        alpha=alpha,
        horizon_min=H,
        slices=n,
        participation_path=parts
    )

# ---------------------------------------------------------------------
# Orderbook-based LVaR (depth takeout)
# ---------------------------------------------------------------------

@dataclass
class BookLevel:
    price: float
    size: float

@dataclass
class BookSide:
    levels: List[BookLevel]  # sorted: bids desc, asks asc

def lvar_orderbook(
    *,
    side: Side,
    qty: float,
    mid: float,
    book: BookSide
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Consume an L2 snapshot and compute *deterministic* slippage cost in bps
    to execute `qty` at market now.
    Returns (cost_bps, fills) where fills = [(px, size), ...]
    """
    assert mid > 0 and qty >= 0
    remaining = qty
    notional = 0.0
    paid = 0.0
    fills: List[Tuple[float, float]] = []

    if side == "buy":
        # cross asks (ascending)
        for lvl in book.levels:
            take = min(remaining, lvl.size)
            if take <= 0: break
            paid += take * lvl.price
            notional += take * mid
            fills.append((lvl.price, take))
            remaining -= take
    else:
        # sell into bids (descending)
        for lvl in book.levels:
            take = min(remaining, lvl.size)
            if take <= 0: break
            paid += take * lvl.price
            notional += take * mid
            fills.append((lvl.price, take))
            remaining -= take

    if remaining > 1e-9:
        # insufficient depth; treat missing as infinite cost; return huge bps
        return 1e9, fills

    vwpx = paid / max(1e-12, sum(s for _, s in fills))
    if side == "buy":
        slip = (vwpx - mid) / mid * 1e4  # bps
    else:
        slip = (mid - vwpx) / mid * 1e4

    return max(0.0, slip), fills

# ---------------------------------------------------------------------
# Portfolio aggregation helpers
# ---------------------------------------------------------------------

@dataclass
class LVaRPortfolioItem:
    symbol: str
    side: Side
    qty: float
    px: float
    lvar_bps: float
    expected_bps: float
    stdev_bps: float
    notional: float

@dataclass
class LVaRPortfolioSummary:
    items: List[LVaRPortfolioItem]
    total_expected: float          # currency
    total_lvar: float              # currency
    total_notional: float
    alpha: float

def aggregate_portfolio(
    items: Iterable[Tuple[str, Side, float, float, LVaRBreakdown]]
) -> LVaRPortfolioSummary:
    out: List[LVaRPortfolioItem] = []
    tot_exp = tot_lvar = tot_not = 0.0
    alpha = None
    for sym, side, q, px, br in items:
        notional = abs(q * px)
        exp_ccy = notional * br.expected_bps / 1e4
        lvar_ccy = notional * br.lv_ar_bps / 1e4
        out.append(LVaRPortfolioItem(
            symbol=sym, side=side, qty=q, px=px,
            lvar_bps=br.lv_ar_bps, expected_bps=br.expected_bps, stdev_bps=br.stdev_bps,
            notional=notional
        ))
        tot_exp += exp_ccy
        tot_lvar += lvar_ccy
        tot_not += notional
        alpha = br.alpha
    return LVaRPortfolioSummary(out, tot_exp, tot_lvar, tot_not, alpha or 0.975)

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _phi(z: float) -> float:
    return math.exp(-0.5*z*z) / math.sqrt(2.0*math.pi)

def _ncdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _inv_norm_cdf(p: float, lo: float = -8.0, hi: float = 8.0, tol: float = 1e-7) -> float:
    """Simple binary-search inverse Φ to avoid external deps."""
    if p <= 0.0: return -8.0
    if p >= 1.0: return 8.0
    a, b = lo, hi
    while b - a > tol:
        m = 0.5 * (a + b)
        if _ncdf(m) < p: a = m
        else: b = m
    return 0.5 * (a + b)

# ---------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    # Parametric example
    prof = LiquidityProfile(
        adv=5_000_000,    # shares/day
        spread_bps=2.0,   # 2 bps
        vol_day=0.018,    # 1.8% daily vol
        fee_bps=0.3,      # fees
        impact_eta=30.0,  # temporary
        impact_gamma=6.0  # permanent
    )
    plan = SlicePlan(horizon_min=60, slices=12, part_cap=0.15)
    br = lvar_parametric(side="buy", qty=200_000, px=50.0, profile=prof, plan=plan, alpha=0.975)
    print("Param LVaR bps:", round(br.lv_ar_bps, 2), " Expected bps:", round(br.expected_bps, 2), "σ:", round(br.stdev_bps, 2))

    # Orderbook example (toy)
    mid = 100.0
    book = BookSide(levels=[
        BookLevel(price=100.02, size=2_000),
        BookLevel(price=100.03, size=3_000),
        BookLevel(price=100.05, size=5_000),
        BookLevel(price=100.07, size=10_000),
    ])
    slip_bps, fills = lvar_orderbook(side="buy", qty=8_500, mid=mid, book=book)
    print("OB slip bps:", round(slip_bps, 2), "fills:", fills)

    # Aggregate two legs
    br2 = lvar_parametric(side="sell", qty=120_000, px=70.0, profile=prof, plan=plan, alpha=0.99)
    summ = aggregate_portfolio([
        ("AAA","buy",200_000,50.0,br),
        ("BBB","sell",120_000,70.0,br2),
    ])
    print("Portfolio LVaR (ccy):", round(summ.total_lvar, 2), "@alpha", summ.alpha)