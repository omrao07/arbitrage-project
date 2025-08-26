# backend/risk/liquidity_spiral.py
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

# Soft deps to play nicely with the rest of your codebase
try:
    from backend.execution.pricer import Quote # type: ignore
except Exception:
    @dataclass
    class Quote:
        symbol: str
        bid: float | None = None
        ask: float | None = None
        last: float | None = None
        def mid(self) -> Optional[float]:
            if self.bid and self.ask:
                return (self.bid + self.ask) / 2.0
            return self.last

# =============================================================================
# Models
# =============================================================================

@dataclass
class AssetSpec:
    """
    ADV: average daily volume (notional, in base CCY) to scale market impact
    vol: daily volatility (as decimal, e.g., 0.02 for 2%)
    """
    symbol: str
    adv: float                     # base-ccy ADV (e.g., USD)
    vol: float                     # daily sigma
    init_price: float              # starting clean price (mid)
    impact_lambda: float = 0.1     # permanent impact per 100% ADV sold (Kyle-ish)
    impact_kappa: float = 0.3      # temporary impact share (0..1); permanent = (1-kappa)
    fee_bps: float = 1.0           # round-trip fees/slippage bps used for proceeds haircut
    min_tick: float = 0.0          # optional

@dataclass
class Position:
    symbol: str
    qty: float                     # + long, - short
    avg_price: float
    margin_haircut: float          # initial margin/haircut (0..1 of gross notional)
    repo_haircut: float = 0.0      # additional funding haircut if financed
    financed: bool = True          # if True, funding haircuts apply (repo/prime)

@dataclass
class FundingLine:
    """
    A credit line used for margin + financing.
    """
    name: str
    limit: float                   # maximum drawable
    rate: float                    # daily rate (simple)
    drawn: float = 0.0             # current utilization

@dataclass
class SpiralParams:
    """
    Controls the spiral dynamics.
    """
    base_ccy: str = "USD"
    stress_days: int = 5                   # simulate over N days
    liquidation_horizon_days: int = 3      # how fast you can liquidate (equal slices)
    min_cash_buffer: float = 0.0           # operational buffer you try not to breach
    var_conf: float = 0.99                 # drives procyclical margin
    var_mult: float = 3.0                  # how aggressively brokers scale haircuts with VaR
    margin_add_on_bps_per_sigma: float = 50.0  # extra haircut (bps) per 1σ vol jump
    leverage_cap: float = 10.0             # hard cap; above → forced delever
    delever_priority: str = "worst_liquidity_first"  # or "proportional"
    max_iterations: int = 50               # per day fixed-point iterations
    tol: float = 1e-6

@dataclass
class SpiralState:
    day: int
    cash: float
    equity: float
    nav: float
    leverage: float
    margin_required: float
    shortfall: float
    prices: Dict[str, float]
    holdings: Dict[str, float]
    sold_today: Dict[str, float]
    drawn_funding: float

@dataclass
class SpiralResult:
    states: List[SpiralState]
    terminal: SpiralState
    metrics: Dict[str, float]
    params: Dict[str, float | int | str]

# =============================================================================
# Core mechanics
# =============================================================================

def _impact_price(p0: float, sell_notional: float, spec: AssetSpec) -> float:
    """
    Simple permanent+temporary impact model.
    sell_notional as fraction of ADV: q = N / ADV
    dP/P ≈ - lambda * q   (permanent)
    Execution mid used for proceeds includes temporary component (kappa).
    """
    if p0 <= 0 or spec.adv <= 0:
        return max(0.0, p0)
    q = max(0.0, sell_notional / spec.adv)
    d_perm = spec.impact_lambda * q
    p_new = p0 * max(0.0, 1.0 - d_perm)
    if spec.min_tick > 0:
        p_new = round(p_new / spec.min_tick) * spec.min_tick
    return max(0.0, p_new)

def _exec_proceeds(sell_qty: float, px_before: float, spec: AssetSpec) -> float:
    """
    Execution price includes temporary impact share (kappa) and fees.
    """
    if sell_qty <= 0:
        return 0.0
    notional = sell_qty * px_before
    q = notional / max(1e-12, spec.adv)
    d_tmp = spec.impact_kappa * spec.impact_lambda * q
    exec_px = px_before * max(0.0, 1.0 - d_tmp)
    fees = (spec.fee_bps / 1e4) * sell_qty * exec_px
    return max(0.0, sell_qty * exec_px - fees)

def _portfolio_values(positions: Dict[str, Position], prices: Dict[str, float]) -> Tuple[float, float]:
    mv = 0.0
    gross = 0.0
    for sym, pos in positions.items():
        px = prices.get(sym, pos.avg_price)
        mv += pos.qty * px
        gross += abs(pos.qty * px)
    return mv, gross

def _margin_required(positions: Dict[str, Position], prices: Dict[str, float], vol_bump: float, sp: SpiralParams) -> float:
    """
    Haircuts scale up with vol via a simple procyclical rule:
        eff_haircut = margin_haircut + repo_haircut + (sigma * var_mult * Phi^{-1}(conf)) * add_on_bps
    """
    # inverse normal quantile approximation
    z = _norm_cdf_inv(sp.var_conf)
    req = 0.0
    for sym, pos in positions.items():
        px = prices.get(sym, pos.avg_price)
        base_h = pos.margin_haircut + (pos.repo_haircut if pos.financed else 0.0)
        addon = (sp.margin_add_on_bps_per_sigma / 1e4) * (vol_bump * z * sp.var_mult)
        h = min(0.99, max(base_h, base_h + addon))
        req += abs(pos.qty * px) * h
    return req

def _norm_cdf_inv(p: float) -> float:
    # Acklam inverse CDF (same as in your shock_models) – trimmed for brevity
    p = max(1e-12, min(1 - 1e-12, p))
    a = (-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00)
    pl, ph = 0.02425, 1-0.02425
    if p < pl:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > ph:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

# =============================================================================
# Simulator
# =============================================================================

def simulate_spiral(
    *,
    assets: Dict[str, AssetSpec],
    positions: Dict[str, Position],
    starting_cash: float,
    funding: Optional[FundingLine] = None,
    params: Optional[SpiralParams] = None,
) -> SpiralResult:
    """
    Discrete-time (days) simulator of a liquidity spiral:
      1) Mark portfolio → equity/NAV/leverage.
      2) Update procyclical margins (haircuts ↑ with vol bump).
      3) If cash < margin + buffer -> sell to raise cash.
      4) Sales cause market impact → prices ↓ → mark-to-market loss.
      5) Iterate inside a day until fixed point (or max iter).
      6) Repeat for 'stress_days'.
    """
    sp = params or SpiralParams()
    prices = {s: assets[s].init_price for s in assets}
    pos = {k: Position(**asdict(v)) for k, v in positions.items()}  # deep-copy
    cash = float(starting_cash)
    states: List[SpiralState] = []

    for day in range(1, sp.stress_days + 1):
        # base vol bump can be proportional to asset sigma (1-day shock)
        vol_bump = max(a.vol for a in assets.values())
        sold_today: Dict[str, float] = {s: 0.0 for s in pos}

        # Fixed-point inner loop (solve for price/quantity given constraints)
        for _it in range(sp.max_iterations):
            mv, gross = _portfolio_values(pos, prices)
            equity = cash + mv - (funding.drawn if funding else 0.0)
            nav = equity
            leverage = (gross / max(1e-9, equity)) if equity > 0 else float("inf")
            mreq = _margin_required(pos, prices, vol_bump, sp)
            shortfall = max(0.0, mreq + sp.min_cash_buffer - cash)

            # Stop if constraints satisfied and leverage ok
            if shortfall <= sp.tol and (math.isfinite(leverage) and leverage <= sp.leverage_cap + 1e-9):
                break

            # Determine liquidation need for this inner step
            need = shortfall
            if math.isfinite(leverage) and leverage > sp.leverage_cap:
                # sell enough gross to bring leverage down towards cap (heuristic)
                target_gross = sp.leverage_cap * max(1e-9, equity)
                need = max(need, (gross - target_gross) * 0.5)  # sell half the excess this step

            if need <= sp.tol:
                break

            # Allocate sales by rule
            sales_plan: Dict[str, float] = {}
            if sp.delever_priority == "worst_liquidity_first":
                # rank by (vol * impact / ADV) descending
                ranked = sorted(pos.keys(), key=lambda s: (assets[s].vol * assets[s].impact_lambda / max(assets[s].adv,1e-9)), reverse=True)
            else:
                ranked = list(pos.keys())

            # sell across horizon slices
            slice_cap = 1.0 / max(1, sp.liquidation_horizon_days)
            to_raise = need

            for s in ranked:
                if to_raise <= 0:
                    break
                p = pos[s]
                if p.qty <= 0:
                    continue
                spec = assets[s]
                px = prices[s]
                # max notional we *can* sell this inner step (heuristic: slice of position)
                max_sell_qty = p.qty * slice_cap
                if max_sell_qty <= 0:
                    continue
                # choose qty such that proceeds approximate the remaining need
                # back-of-envelope exec price ~ current px * (1 - kappa*lambda*q)
                # start with linear guess; clamp to max
                guess_qty = min(max_sell_qty, to_raise / max(px * (1 - spec.impact_kappa * spec.impact_lambda * (px * max_sell_qty / max(spec.adv,1e-9)) / max(px,1e-9)), 1e-9))
                if guess_qty <= 0:
                    continue

                proceeds = _exec_proceeds(guess_qty, px, spec)
                if proceeds <= 0:
                    continue

                # apply: update cash, reduce qty, update price with permanent impact
                cash += proceeds
                p.qty -= guess_qty
                sold_today[s] += guess_qty
                # permanent impact from this *step*
                prices[s] = _impact_price(px, guess_qty * px, spec)

                to_raise = max(0.0, to_raise - proceeds)

            # If we couldn't raise anything (illiquidity), break to avoid infinite loop
            if need > 0 and abs(need - to_raise) / max(1.0, need) > 0.999:
                break

        # accrue interest on funding (if any)
        if funding:
            funding.drawn = min(funding.limit, max(0.0, funding.drawn))  # keep sane
            interest = funding.drawn * funding.rate
            cash -= interest

        # End-of-day snapshot
        mv, gross = _portfolio_values(pos, prices)
        equity = cash + mv - (funding.drawn if funding else 0.0)
        nav = equity
        leverage = (gross / max(1e-9, equity)) if equity > 0 else float("inf")
        mreq = _margin_required(pos, prices, vol_bump, sp)
        shortfall = max(0.0, mreq + sp.min_cash_buffer - cash)

        states.append(SpiralState(
            day=day,
            cash=cash,
            equity=equity,
            nav=nav,
            leverage=leverage if math.isfinite(leverage) else float("inf"),
            margin_required=mreq,
            shortfall=shortfall,
            prices={k: float(v) for k, v in prices.items()},
            holdings={k: float(v.qty) for k, v in pos.items()},
            sold_today={k: float(v) for k, v in sold_today.items()},
            drawn_funding=(funding.drawn if funding else 0.0),
        ))

    terminal = states[-1]
    metrics = {
        "days": len(states),
        "nav_drawdown_pct": 0.0 if not states else 100.0 * (states[0].nav - terminal.nav) / max(1e-9, states[0].nav),
        "max_leverage": max(s.leverage for s in states if math.isfinite(s.leverage)),
        "cum_sold_notional": sum(sum(st.sold_today[s] * states[st.day-1].prices[s] for s in st.sold_today) for st in states),
        "final_margin_shortfall": terminal.shortfall,
    }
    return SpiralResult(states=states, terminal=terminal, metrics=metrics, params=asdict(sp))

# =============================================================================
# Tiny smoke test
# =============================================================================

if __name__ == "__main__":
    # 2-asset book: liquid (AAPL-like) and illiquid smallcap
    assets = {
        "LIQ": AssetSpec(symbol="LIQ", adv=50_000_000.0, vol=0.02, init_price=100.0, impact_lambda=0.10, impact_kappa=0.4, fee_bps=1.0),
        "ILL": AssetSpec(symbol="ILL", adv=2_000_000.0,  vol=0.05, init_price=25.0,  impact_lambda=0.35, impact_kappa=0.6, fee_bps=3.0),
    }
    positions = {
        "LIQ": Position(symbol="LIQ", qty=300_000, avg_price=95.0, margin_haircut=0.20, financed=True, repo_haircut=0.05),
        "ILL": Position(symbol="ILL", qty=600_000, avg_price=22.0, margin_haircut=0.35, financed=True, repo_haircut=0.10),
    }
    funding = FundingLine(name="PB", limit=50_000_000.0, rate=0.0002, drawn=20_000_000.0)
    res = simulate_spiral(
        assets=assets, positions=positions, starting_cash=5_000_000.0,
        funding=funding, params=SpiralParams(stress_days=5, liquidation_horizon_days=3, leverage_cap=8.0)
    )
    print("Terminal NAV:", round(res.terminal.nav, 2))
    print("NAV DD %:", round(res.metrics["nav_drawdown_pct"], 2))
    print("Max Lev:", round(res.metrics["max_leverage"], 2))
    print("Final Shortfall:", round(res.metrics["final_margin_shortfall"], 2))