# backend/tca/cost_model_v2.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Tuple, Any

Side = Literal["buy", "sell"]
AssetClass = Literal["equity", "futures", "options", "crypto"]


# ----------------------------- Helpers -----------------------------

def _bps(x: float) -> float:
    return 1e4 * x

def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0

_now_ms = lambda: int(time.time() * 1000)


# ----------------------------- Config ------------------------------

@dataclass
class VenueFees:
    """Explicit fees/rebates; if cents provided we convert using reference price."""
    maker_rebate_bps: float = 0.0
    taker_fee_bps: float = 0.0
    maker_rebate_cents: float = 0.0
    taker_fee_cents: float = 0.0
    stamp_duty_bps: float = 0.0         # e.g., some markets
    sec_fee_bps: float = 0.0            # US SEC fee for sells (tiny), etc.


@dataclass
class CostModelParams:
    # ---- Spread component (crossing/price improvement) ----
    # Effective spread share we expect to pay when taking; reduced if passive.
    eff_spread_share_take: float = 0.75     # e.g., pay ~75% of quoted spread when taking
    eff_spread_share_post: float = 0.15     # price improvement / queue jump loss when posting
    # Maker fill risk multiplier if you rarely get full passive fills:
    maker_fill_discount: float = 0.65       # <1 means you often have to chase

    # ---- Temporary market impact: Almgren–Chriss-style ----
    # temp = k_temp * sigma * (q / (ADV * T))^alpha  (in % of price)
    alpha_temp: float = 0.6                 # concave for child orders
    k_temp: float = 1.0                     # baseline; will be scaled per asset class
    # Participation scaling (higher POV increases temp impact)
    pov_elasticity: float = 0.35

    # ---- Permanent impact ----
    # perm = k_perm * sigma * (q / ADV)^beta
    beta_perm: float = 0.5
    k_perm: float = 0.25

    # ---- Microstructure hazard adjustments ----
    # Toxicity / volatility multipliers (0..1 score mapped to [1, 1+…])
    tox_temp_up: float = 0.75               # extra temp impact at toxicity=1 (i.e., ×(1+tox_temp_up))
    tox_spread_up: float = 0.50             # spread inflation at toxicity=1
    # Wide-spread / thin-book guard
    widen_spread_bps: float = 10.0          # treat anything wider as “costly to take”

    # ---- Options specific slippage bump (per 10% of IV) ----
    option_iv_bump_bps_per_10pct: float = 3.0

    # ---- Crypto specific (higher taker, funding/borrow) ----
    crypto_extra_taker_bps: float = 2.0

    # ---- Borrow/funding (equity shorting / futures funding) ----
    borrow_bps_daily: float = 0.0           # set via runtime for hard-to-borrow
    funding_bps_daily: float = 0.0          # futures/crypto funding estimate

    # ---- Smoothing for online calibration ----
    ewma_alpha: float = 0.12


@dataclass
class MarketContext:
    """
    Snapshot of market+order used to estimate cost for one child or a block.
    """
    asset: AssetClass
    side: Side
    mid_px: float
    spread_bps: float
    sigma_daily: float                      # daily vol (as fraction, e.g., 0.02 for 2%)
    adv_shares: float                       # average daily volume (shares/contracts)
    q_child: float                          # shares/contracts to trade for this decision
    T_child_minutes: float                  # horizon for this slice (minutes)
    participation: float                    # 0..1 POV estimate during slice
    venue: Optional[str] = None             # for fee table / toxicity
    toxicity: float = 0.0                   # 0..1 from your (venue) toxicity module
    passive: bool = False                   # are we posting (expected maker)?
    ref_px_for_cents: Optional[float] = None
    option_iv: Optional[float] = None       # implied vol (0.4 = 40%) for options
    borrow_bps_daily: Optional[float] = None
    funding_bps_daily: Optional[float] = None


# --------------------------- Main Model ----------------------------

class CostModelV2:
    """
    Estimates **total cost in bps** and decomposes it:
      total_bps = explicit_fees + spread_component + temporary_impact + permanent_impact
                  + borrow/funding (pro-rated) + special asset-class bumps
    Also supports **online calibration** using realized slippage vs mid/VWAP.
    """

    def __init__(self, params: Optional[CostModelParams] = None, fees: Optional[Dict[str, VenueFees]] = None):
        self.p = params or CostModelParams()
        self.fees = fees or {}
        # simple online calibration (per asset class): scale multipliers
        self.temp_scale: Dict[AssetClass, float] = {k: 1.0 for k in ("equity", "futures", "options", "crypto")}
        self.perm_scale: Dict[AssetClass, float] = {k: 1.0 for k in ("equity", "futures", "options", "crypto")}

    # ----------- explicit fees / rebates / levies -----------
    def _explicit_bps(self, ctx: MarketContext) -> float:
        v = self.fees.get(ctx.venue or "", VenueFees())
        ref_px = ctx.ref_px_for_cents or ctx.mid_px or 1.0
        maker_bps = v.maker_rebate_bps or _bps((v.maker_rebate_cents / 100.0) / ref_px)
        taker_bps = v.taker_fee_bps    or _bps((v.taker_fee_cents    / 100.0) / ref_px)
        duty_bps  = v.stamp_duty_bps + v.sec_fee_bps

        # crypto extra taker bump
        if ctx.asset == "crypto":
            taker_bps += self.p.crypto_extra_taker_bps

        # pick maker/taker depending on passivity
        fee_bps = -maker_bps if ctx.passive else taker_bps
        return float(fee_bps + duty_bps)

    # -------------------- spread component -------------------
    def _spread_bps_component(self, ctx: MarketContext) -> float:
        # Inflate spread when toxicity is high
        spread_bps = ctx.spread_bps * (1.0 + self.p.tox_spread_up * _clip(ctx.toxicity, 0.0, 1.0))
        # Maker vs taker share
        if ctx.passive:
            eff = self.p.eff_spread_share_post
            # discount further by expected maker fill fraction (you might still have to chase)
            eff = eff * self.p.maker_fill_discount
        else:
            eff = self.p.eff_spread_share_take
            # If spread already very wide, nudge toward full touch cost
            if spread_bps >= self.p.widen_spread_bps:
                eff = max(eff, 0.95)
        return spread_bps * eff

    # ---------------- temporary impact (bps) ------------------
    def _temp_impact_bps(self, ctx: MarketContext) -> float:
        if ctx.adv_shares <= 0 or ctx.q_child <= 0:
            return 0.0
        # Convert daily sigma to slice sigma via sqrt(time)
        T_days = max(1e-6, ctx.T_child_minutes / (60.0 * 24.0))
        sigma_slice = ctx.sigma_daily * math.sqrt(T_days)

        # Participation & size factor
        size_ratio = ctx.q_child / max(1e-9, ctx.adv_shares * max(T_days, 1e-6))
        size_ratio = max(1e-9, size_ratio)

        # Baseline AC-style temp impact (in % of price)
        temp_pct = (self.p.k_temp * self.temp_scale[ctx.asset]) * sigma_slice * (size_ratio ** self.p.alpha_temp)

        # Participation elasticity (higher POV → more impact)
        pov_adj = (1.0 + self.p.pov_elasticity * _clip(ctx.participation, 0.0, 1.0))

        # Toxicity multiplier
        tox_adj = (1.0 + self.p.tox_temp_up * _clip(ctx.toxicity, 0.0, 1.0))

        return _bps(temp_pct * pov_adj * tox_adj)

    # ---------------- permanent impact (bps) ------------------
    def _perm_impact_bps(self, ctx: MarketContext) -> float:
        if ctx.adv_shares <= 0 or ctx.q_child <= 0:
            return 0.0
        size_ratio = ctx.q_child / max(1e-9, ctx.adv_shares)
        perm_pct = (self.p.k_perm * self.perm_scale[ctx.asset]) * ctx.sigma_daily * (size_ratio ** self.p.beta_perm)
        return _bps(perm_pct)

    # --------------- borrow / funding (bps) -------------------
    def _carry_bps(self, ctx: MarketContext) -> float:
        borrow = self.p.borrow_bps_daily if ctx.borrow_bps_daily is None else ctx.borrow_bps_daily
        funding = self.p.funding_bps_daily if ctx.funding_bps_daily is None else ctx.funding_bps_daily
        # Pro-rate to the slice horizon (in days). Sign: borrow/funding are costs regardless of side here.
        T_days = max(1e-6, ctx.T_child_minutes / (60.0 * 24.0))
        return (borrow + funding) * T_days

    # ---------------- options slippage bump -------------------
    def _options_bump_bps(self, ctx: MarketContext) -> float:
        if ctx.asset != "options" or not ctx.option_iv:
            return 0.0
        # small linear bump with IV level
        return (ctx.option_iv * 10.0) * self.p.option_iv_bump_bps_per_10pct

    # ----------------------- API ------------------------------
    def estimate_bps(self, ctx: MarketContext) -> Dict[str, float]:
        """
        Return a breakdown dict in **bps**:
            {'explicit': .., 'spread': .., 'temp': .., 'perm': .., 'carry': .., 'special': .., 'total': ..}
        """
        explicit_bps = self._explicit_bps(ctx)
        spread_bps   = self._spread_bps_component(ctx)
        temp_bps     = self._temp_impact_bps(ctx)
        perm_bps     = self._perm_impact_bps(ctx)
        carry_bps    = self._carry_bps(ctx)
        special_bps  = self._options_bump_bps(ctx)

        total = explicit_bps + spread_bps + temp_bps + perm_bps + carry_bps + special_bps
        # For SELL, interpret “bps cost” the same way (positive = you pay); callers can apply sign if needed.
        return {
            "explicit": explicit_bps,
            "spread": spread_bps,
            "temp": temp_bps,
            "perm": perm_bps,
            "carry": carry_bps,
            "special": special_bps,
            "total": total,
        }

    # --------------- Convenience: child notional --------------
    def estimate_cash_cost(self, ctx: MarketContext) -> Dict[str, float]:
        """
        Convert the bps breakdown to **cash cost** using mid price and quantity.
        """
        br = self.estimate_bps(ctx)
        notional = ctx.mid_px * ctx.q_child
        out = {}
        for k, v in br.items():
            out[k] = (v / 1e4) * notional
        out["notional"] = notional
        return out

    # --------------- Online calibration hooks -----------------
    def update_from_fill(self,
                         asset: AssetClass,
                         realized_slip_bps_vs_mid: float,
                         intended_temp_share: float = 0.7) -> None:
        """
        Online EWMA calibration using realized slippage (bps) vs mid at send.
        Split the slippage attribution between temp & spread by intended_temp_share.
        Example:
            realized +8 bps on an equity child with intended_temp_share=0.7
            -> assign 5.6 bps to temp; 2.4 bps to “other” (spread, venue, etc.)
            We scale temp multiplier to move model prediction toward observed.
        """
        a = _clip(self.p.ewma_alpha, 0.01, 0.5)
        obs_temp = realized_slip_bps_vs_mid * _clip(intended_temp_share, 0.0, 1.0)
        # Keep temp/perm separate; here we just nudge temp scale
        cur = self.temp_scale[asset]
        # Target scale factor that would have matched obs (coarse): if temp too low, scale up, etc.
        target_scale = _clip(1.0 + obs_temp / 20.0, 0.5, 2.5)   # 20 bps reference width
        self.temp_scale[asset] = (1 - a) * cur + a * target_scale

    def set_fees(self, venue: str, fees: VenueFees) -> None:
        self.fees[venue] = fees

    # ------------------- Pretty print -------------------------
    @staticmethod
    def summarize_breakdown(bps_breakdown: Dict[str, float]) -> str:
        parts = ["explicit", "spread", "temp", "perm", "carry", "special"]
        lines = []
        for k in parts:
            lines.append(f"{k:8s}: {bps_breakdown.get(k, 0.0):7.3f} bps")
        lines.append(f"{'total':8s}: {bps_breakdown.get('total', 0.0):7.3f} bps")
        return "\n".join(lines)


# ------------------------------ Demo ------------------------------
if __name__ == "__main__":
    # Example usage
    model = CostModelV2()

    # Optional: set fees for venues
    model.set_fees("NASDAQ", VenueFees(taker_fee_bps=2.5, maker_rebate_bps= -0.2))  # negative rebate = you earn

    ctx = MarketContext(
        asset="equity",
        side="buy",
        mid_px=100.00,
        spread_bps=4.0,             # 4 bps quoted spread
        sigma_daily=0.02,           # 2% daily vol
        adv_shares=20_000_000,
        q_child=10_000,             # 10k shares
        T_child_minutes=5.0,        # execute over 5 minutes
        participation=0.1,          # ~10% POV
        venue="NASDAQ",
        toxicity=0.3,               # from your venue tox scorer
        passive=False,
    )

    bps = model.estimate_bps(ctx)
    print(CostModelV2.summarize_breakdown(bps))

    # If you log realized slippage, feed it back:
    model.update_from_fill(asset="equity", realized_slip_bps_vs_mid=6.8, intended_temp_share=0.7)