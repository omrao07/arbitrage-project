# backend/risk/lvar.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Literal

# ----------------------------- helpers -----------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def _stdev(xs: List[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - ddof)
    return math.sqrt(var)

def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    if q <= 0: return xs[0]
    if q >= 1: return xs[-1]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi: return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w

def _norm_ppf(p: float) -> float:
    # Acklam/Beasley–Springer inverse normal CDF
    if p <= 0.0: return -1e9
    if p >= 1.0: return +1e9
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425; phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def _bps(x: float) -> float:
    return 1e4 * x

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------- config ------------------------------

ImpactModel = Literal["sqrt", "amihud"]

@dataclass
class LVaRParams:
    alpha: float = 0.99                 # VaR/ES confidence (right-side: 99% → left-tail 1%)
    horizon_days: int = 1               # statistical horizon for price VaR
    liquidation_days: int = 3           # number of days to unwind position
    participation: float = 0.1          # max %ADV you will trade per day (0..1)
    eff_spread_bps: float = 8.0         # effective half-spread in bps (per take); maker lowers it
    price_impact_model: ImpactModel = "sqrt"
    # Market impact coefficients:
    # - sqrt: impact_bps = k * sqrt( (qty / (ADV * horizon_fraction)) )
    k_sqrt: float = 25.0
    # - amihud: impact_bps = k * |notional| / (ADV_$)
    k_amihud: float = 1e-3              # choose ~1e-3..1e-2 depending on units
    # Spread scaling under stress/liquidity drought
    widen_spread_mult: float = 1.0
    # ADV haircut under stress (e.g., 0.6 = 40% drop in liquidity)
    adv_haircut: float = 1.0

# ----------------------------- results -----------------------------

@dataclass
class LVaRBreakdown:
    alpha: float
    horizon_days: int
    liquidation_days: int
    position_qty: float
    mid_px: float
    notional: float
    price_var_cash: float
    spread_cost_cash: float
    impact_cost_cash: float
    total_lvar_cash: float
    # in bps of notional (positive = loss)
    price_var_bps: float
    spread_cost_bps: float
    impact_cost_bps: float
    total_lvar_bps: float

# ----------------------------- core -------------------------------

class LiquidityVaR:
    """
    Liquidity-adjusted VaR:
      LVaR ≈ PriceVaR + SpreadCost (liquidation schedule) + MarketImpact (non-linear).

    You pass returns (fraction space), position, price, ADV, and params.
    """

    # -------- price VaR (hist or gaussian) --------
    @staticmethod
    def price_var_cash(
        returns: Iterable[float],
        notional: float,
        alpha: float = 0.99,
        method: str = "historical"
    ) -> Tuple[float, float]:
        """
        Returns (VaR_cash, var_ret).
        var_ret is negative (e.g., -0.025 means -2.5%).
        """
        r = [float(x) for x in returns]
        n = len(r)
        if n == 0:
            return (0.0, 0.0)
        if method == "gaussian":
            mu = _mean(r); sd = _stdev(r) if n > 1 else 0.0
            q = 1 - alpha
            z = _norm_ppf(q)  # negative
            var_ret = mu + sd * z
        else:
            rs = sorted(r)
            var_ret = _quantile(rs, 1 - alpha)
        var_cash = -var_ret * notional  # positive loss
        return (max(0.0, var_cash), var_ret)

    # -------- liquidation schedule --------
    @staticmethod
    def liquidation_slices(
        position_qty: float,
        adv_qty_per_day: float,
        liquidation_days: int,
        participation: float
    ) -> List[float]:
        """
        Daily slice quantities constrained by participation * ADV.
        """
        cap = max(1e-9, participation) * max(1e-9, adv_qty_per_day)
        remaining = abs(position_qty)
        slices: List[float] = []
        for _ in range(max(1, liquidation_days)):
            q = min(remaining, cap)
            slices.append(q)
            remaining -= q
        if remaining > 1e-9:  # need extra days if position > schedule capacity
            while remaining > 1e-9:
                q = min(remaining, cap)
                slices.append(q)
                remaining -= q
        return slices

    # -------- spread cost over schedule --------
    @staticmethod
    def spread_cost(
        slices_qty: List[float],
        mid_px: float,
        eff_spread_bps: float
    ) -> float:
        """
        Cost of crossing the spread for each slice.
        We use **half-spread per side** ~ eff_spread_bps (typical effective take).
        """
        total_shares = sum(slices_qty)
        loss_bps = eff_spread_bps  # per notional
        notional = mid_px * total_shares
        return (loss_bps / 1e4) * notional

    # -------- market impact over schedule --------
    @staticmethod
    def impact_cost(
        slices_qty: List[float],
        mid_px: float,
        adv_qty_per_day: float,
        params: LVaRParams
    ) -> float: # type: ignore
        """
        Temporary+permanent impact proxy aggregated over slices.
        - sqrt model: impact_bps = k * sqrt(q / ADV_slice)
        - amihud model: impact_bps = k * (q*px) / ADV_$  (linear in notional liquidity)
        """
        loss_cash = 0.0
        adv_q = max(1e-9, adv_qty_per_day * params.adv_haircut)
        for q in slices_qty:
            if q <= 0: 
                continue
            if params.price_impact_model == "amihud":
                adv_$ = adv_q * mid_px # type: ignore
                impact_bps = params.k_amihud * (q * mid_px) / max(1e-9, adv_$) * 1e4 # type: ignore
            else:
                # sqrt-law impact, using daily ADV as liquidity bucket
                impact_bps = params.k_sqrt * math.sqrt(q / adv_q)
            loss_cash += (impact_bps / 1e4) * (q * mid_px)
        return loss_cash

    # -------- main API: single instrument --------
    @staticmethod
    def estimate_single(
        returns: Iterable[float],
        *,
        position_qty: float,
        mid_px: float,
        adv_qty_per_day: float,
        params: Optional[LVaRParams] = None,
        method: str = "historical"
    ) -> LVaRBreakdown:
        """
        Compute LVaR for one instrument.
        - returns: daily returns (fraction) aligned to price VaR horizon (we assume 1d).
        - position_qty: signed shares/contracts; we assess liquidation cost on |qty|.
        - mid_px: current mid price.
        - adv_qty_per_day: ADV in shares/contracts.
        """
        P = params or LVaRParams()
        # Notional at risk
        notional = abs(position_qty) * mid_px

        # Price VaR on horizon_days (if horizon>1d, scale returns via sqrt-time if gaussian)
        r = [float(x) for x in returns]
        if method == "gaussian" and P.horizon_days > 1:
            mu = _mean(r); sd = _stdev(r)
            if sd > 0:
                # simulate horizon via normal scaling
                z = _norm_ppf(1 - P.alpha)
                var_ret = mu * P.horizon_days + sd * math.sqrt(P.horizon_days) * z
                price_var_cash = max(0.0, -var_ret * notional)
            else:
                price_var_cash, var_ret = (0.0, 0.0)
        else:
            price_var_cash, var_ret = LiquidityVaR.price_var_cash(r, notional, alpha=P.alpha, method=method)

        # Liquidation schedule (respect participation and ADV haircut)
        slices = LiquidityVaR.liquidation_slices(
            position_qty=position_qty,
            adv_qty_per_day=adv_qty_per_day * P.adv_haircut,
            liquidation_days=P.liquidation_days,
            participation=P.participation
        )

        # Spread/impact with stress scalers
        spread_cash = LiquidityVaR.spread_cost(slices, mid_px, P.eff_spread_bps * P.widen_spread_mult)
        impact_cash = LiquidityVaR.impact_cost(slices, mid_px, adv_qty_per_day, P)

        total_cash = price_var_cash + spread_cash + impact_cash

        return LVaRBreakdown(
            alpha=P.alpha,
            horizon_days=P.horizon_days,
            liquidation_days=len(slices),
            position_qty=position_qty,
            mid_px=mid_px,
            notional=notional,
            price_var_cash=price_var_cash,
            spread_cost_cash=spread_cash,
            impact_cost_cash=impact_cash,
            total_lvar_cash=total_cash,
            price_var_bps=_bps(price_var_cash / notional) if notional > 0 else 0.0,
            spread_cost_bps=_bps(spread_cash / notional) if notional > 0 else 0.0,
            impact_cost_bps=_bps(impact_cash / notional) if notional > 0 else 0.0,
            total_lvar_bps=_bps(total_cash / notional) if notional > 0 else 0.0,
        )

    # -------- portfolio (independent assets; simple sum) --------
    @staticmethod
    def estimate_portfolio(
        returns_by_asset: Dict[str, List[float]],
        positions: Dict[str, Tuple[float, float, float]],  # asset -> (qty, mid_px, adv_qty/day)
        params_by_asset: Optional[Dict[str, LVaRParams]] = None,
        method: str = "historical"
    ) -> Dict[str, object]:
        """
        Independent-asset aggregation (conservative).
        positions[k] = (qty, mid, adv_q)
        """
        out: Dict[str, LVaRBreakdown] = {}
        total_cash = 0.0
        for k, (qty, px, advq) in positions.items():
            Pk = (params_by_asset or {}).get(k, LVaRParams())
            br = LiquidityVaR.estimate_single(
                returns_by_asset.get(k, []),
                position_qty=qty, mid_px=px, adv_qty_per_day=advq,
                params=Pk, method=method
            )
            out[k] = br
            total_cash += br.total_lvar_cash
        return {
            "by_asset": out,
            "portfolio_total_cash": total_cash,
            "portfolio_total_bps": _bps(total_cash / max(1e-9, sum(abs(q)*p for q, p, _ in positions.values()))),
            "alpha": (params_by_asset or {}).get("alpha", LVaRParams().alpha)
        }

    # -------- stress knob --------
    @staticmethod
    def stress_single(
        returns: Iterable[float],
        *,
        position_qty: float,
        mid_px: float,
        adv_qty_per_day: float,
        spread_widen_mult: float = 2.0,
        adv_drop_mult: float = 0.5,
        method: str = "historical",
        base_params: Optional[LVaRParams] = None
    ) -> LVaRBreakdown:
        P = base_params or LVaRParams()
        P2 = LVaRParams(
            alpha=P.alpha,
            horizon_days=P.horizon_days,
            liquidation_days=P.liquidation_days,
            participation=P.participation,
            eff_spread_bps=P.eff_spread_bps,
            price_impact_model=P.price_impact_model,
            k_sqrt=P.k_sqrt,
            k_amihud=P.k_amihud,
            widen_spread_mult=spread_widen_mult,
            adv_haircut=adv_drop_mult
        )
        return LiquidityVaR.estimate_single(
            returns,
            position_qty=position_qty,
            mid_px=mid_px,
            adv_qty_per_day=adv_qty_per_day,
            params=P2,
            method=method
        )


# ----------------------------- demo ------------------------------
if __name__ == "__main__":
    # Example: 1y daily returns (toy)
    import random
    random.seed(7)
    rets = [random.gauss(0.0002, 0.012) for _ in range(252)]

    qty = 150_000          # shares
    px  = 25.0             # $
    adv = 5_000_000        # shares/day

    params = LVaRParams(
        alpha=0.99,
        horizon_days=1,
        liquidation_days=3,
        participation=0.12,
        eff_spread_bps=6.0,
        price_impact_model="sqrt",
        k_sqrt=20.0
    )

    br = LiquidityVaR.estimate_single(rets, position_qty=qty, mid_px=px, adv_qty_per_day=adv, params=params)
    print("LVaR (cash):", round(br.total_lvar_cash, 2), "USD")
    print("  price VaR :", round(br.price_var_cash, 2))
    print("  spread    :", round(br.spread_cost_cash, 2))
    print("  impact    :", round(br.impact_cost_cash, 2))
    print("LVaR (bps) :", round(br.total_lvar_bps, 2), "bps")

    stress = LiquidityVaR.stress_single(
        rets, position_qty=qty, mid_px=px, adv_qty_per_day=adv,
        spread_widen_mult=2.5, adv_drop_mult=0.4
    )
    print("Stress LVaR (bps):", round(stress.total_lvar_bps, 2))
    )