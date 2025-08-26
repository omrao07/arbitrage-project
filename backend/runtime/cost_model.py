# backend/execution/cost_model.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

# ------------------- Env overrides / defaults -------------------

# Generic (apply to all venues if not overridden)
ENV_SLIPPAGE_BPS_DEFAULT = float(os.getenv("COST_DEFAULT_SLIPPAGE_BPS", "1.0"))  # extra slippage cushion
ENV_SPREAD_BPS_FLOOR     = float(os.getenv("COST_SPREAD_FLOOR_BPS", "4.0"))     # floor on quoted spread if unknown
ENV_IMPACT_K             = float(os.getenv("COST_IMPACT_K", "12.0"))            # square-root impact coefficient (bps)
ENV_IMPACT_ADV_WINDOW    = int(os.getenv("COST_IMPACT_ADV_WINDOW", "20"))       # docs only (your adv calc elsewhere)

# Venue presets (rough; adjust to your account—these are not broker advice)
PRESETS: Dict[str, Dict[str, float]] = {
    # US equities via IBKR (illustrative retail tier)
    "us_ibkr": dict(
        commission_per_share=0.0035,          # $0.0035/sh, min $0.35
        commission_min=0.35,
        sec_fee_bps=0.22 / 100.0,             # sell only (approx, varies)
        finra_trading_activity_fee_per_share=0.000166,  # sell only (approx cap)
        taker_fee_bps=0.0,                    # most routed smart orders: treated in realized slippage
        maker_rebate_bps=0.0,                 # assume neutral for simplicity
        stamp_duty_bps=0.0,
        stt_bps=0.0,
        exchange_txn_bps=0.0,
        gst_bps=0.0,
        sebi_bps=0.0,
    ),
    # India cash equities via Zerodha/NSE (approx public schedule; update to exact)
    "in_zerodha_cash": dict(
        # Brokerage: 0.03% or ₹20 per executed order (whichever lower) – assume % for sizing
        brokerage_bps=3.0,                    # 0.03% = 3 bps
        brokerage_per_order_cap=20.0,         # cap in INR
        stt_bps=10.0,                         # 0.1% on buy+sell (delivery) – adjust for intraday rules
        exchange_txn_bps=0.00325,             # exchange txn charges (per leg) – varies by segment
        sebi_bps=0.001,                       # SEBI charges
        gst_bps=1.8,                          # 18% GST on brokerage + exchange txn (bps view)
        stamp_duty_bps=0.015,                 # buy side only (state dependent)
        taker_fee_bps=0.0,
        maker_rebate_bps=0.0,
        commission_per_share=0.0,             # not used for IN
        commission_min=0.0,
    ),
}

# ------------------- Data classes -------------------

@dataclass
class CostInputs:
    symbol: str
    side: str                 # "buy" | "sell"
    price: float              # reference (mid or expected fill)
    qty: float
    adv: Optional[float] = None        # average daily volume (shares); optional for impact
    spread_bps: Optional[float] = None # quoted or modeled spread in bps
    maker: bool = False                # if True, apply maker rebate; else taker fee
    venue: str = "us_ibkr"             # key from PRESETS
    tif: Optional[str] = None          # "ioc"/"day"/"gtc" (informational)

@dataclass
class CostBreakdown:
    spread_cost: float = 0.0       # $ (or INR)
    impact_cost: float = 0.0
    commissions: float = 0.0
    taxes_fees: float = 0.0        # regulatory/venue taxes (STT, SEC, GST, etc.)
    total: float = 0.0
    total_bps: float = 0.0         # relative to notional
    est_fill_price: float = 0.0

# ------------------- Core formulas -------------------

def _notional(price: float, qty: float) -> float:
    return float(price) * float(qty)

def _spread_cost(price: float, qty: float, spread_bps: float, side: str, maker: bool) -> float:
    """
    Half-spread paid as taker; close to zero if true maker.
    For safety we still allocate tiny cost when maker due to queue risk.
    """
    half = max(spread_bps, ENV_SPREAD_BPS_FLOOR) * 0.5 * 1e-4
    # When maker, assume 15% of half-spread on average due to adverse selection/queue jump.
    eff_half = half if not maker else half * 0.15
    return _notional(price, qty) * eff_half

def _impact_cost(price: float, qty: float, adv: Optional[float], k_bps: float = ENV_IMPACT_K) -> float:
    """
    Square-root market impact in bps: k * sqrt(qty/ADV)
    If ADV unknown -> assume small impact using qty normalization per 1e6 notional.
    """
    if qty <= 0:
        return 0.0
    if adv and adv > 0:
        bps = k_bps * math.sqrt(min(qty / adv, 1.0))
    else:
        # Fallback: scale to notional; assume 5 bps per $1mm notional sqrt
        bps = max(1.0, 5.0 * math.sqrt(_notional(price, qty) / 1_000_000.0))
    return _notional(price, qty) * (bps * 1e-4)

def _commission_and_fees_us_ibkr(inp: CostInputs) -> float:
    p = PRESETS["us_ibkr"]
    per_share = p["commission_per_share"] * inp.qty
    min_fee = p["commission_min"]
    comm = max(per_share, min_fee)

    extra = 0.0
    if inp.side.lower() == "sell":
        extra += _notional(inp.price, inp.qty) * p["sec_fee_bps"] * 1e-2
        extra += p["finra_trading_activity_fee_per_share"] * inp.qty
    # maker/taker left neutral here; typically realized via fill price
    return comm + extra

def _commission_and_fees_in_zerodha(inp: CostInputs) -> float:
    p = PRESETS["in_zerodha_cash"]
    notional = _notional(inp.price, inp.qty)

    # Brokerage: min(percent_of_notional, cap per order)
    brokerage = min(notional * p["brokerage_bps"] * 1e-4, p["brokerage_per_order_cap"])

    # Exchange txn + SEBI
    exch = notional * p["exchange_txn_bps"] * 1e-4
    sebi = notional * p["sebi_bps"] * 1e-4

    # STT (both sides for delivery; if intraday/spec, you may set buy or sell only)
    stt = notional * p["stt_bps"] * 1e-4

    # Stamp duty (usually buy only; keep both sides small for conservatism)
    stamp = notional * p["stamp_duty_bps"] * 1e-4

    # GST 18% on (brokerage + exchange txn)
    gst = 0.18 * (brokerage + exch)

    return brokerage + exch + sebi + stt + stamp + gst

def _maker_taker_adjustment(inp: CostInputs) -> float:
    # If you want to model explicit maker rebates or taker fees separate from spread
    p = PRESETS.get(inp.venue, {})
    bps = 0.0
    if inp.maker:
        bps -= float(p.get("maker_rebate_bps", 0.0))
    else:
        bps += float(p.get("taker_fee_bps", 0.0))
    return _notional(inp.price, inp.qty) * (bps * 1e-4)

def _venue_fees(inp: CostInputs) -> float:
    if inp.venue == "us_ibkr":
        return _commission_and_fees_us_ibkr(inp)
    if inp.venue == "in_zerodha_cash":
        return _commission_and_fees_in_zerodha(inp)
    # default: none
    return 0.0

# ------------------- Public API -------------------

def estimate_cost(inp: CostInputs) -> CostBreakdown:
    """
    Build a full cost breakdown and an estimated executable fill price (mid ± costs).
    """
    spread_bps = inp.spread_bps if inp.spread_bps is not None else ENV_SPREAD_BPS_FLOOR
    spread = _spread_cost(inp.price, inp.qty, spread_bps, inp.side, maker=inp.maker)

    impact = _impact_cost(inp.price, inp.qty, inp.adv, k_bps=ENV_IMPACT_K)

    fees = _venue_fees(inp)
    maker_taker = _maker_taker_adjustment(inp)

    slippage_pad = _notional(inp.price, inp.qty) * (ENV_SLIPPAGE_BPS_DEFAULT * 1e-4)

    total = spread + impact + fees + maker_taker + slippage_pad
    nb = _notional(inp.price, inp.qty)
    total_bps = (total / nb) * 1e4 if nb > 0 else 0.0

    # Est fill price (mid ± bps on the side you cross; maker modeled with tiny spread fraction)
    side_sign = +1 if inp.side.lower() == "buy" else -1
    # Convert only the *price-affecting* components to bps (spread+impact ± maker/taker + slippage_pad)
    px_affect_bps = (spread + impact + maker_taker + slippage_pad) / max(nb, 1e-9) * 1e4
    est_fill = inp.price * (1.0 + side_sign * (px_affect_bps * 1e-4))

    return CostBreakdown(
        spread_cost=spread,
        impact_cost=impact,
        commissions=fees,
        taxes_fees=0.0,  # folded into 'commissions' for presets; split if you want finer granularity
        total=total,
        total_bps=total_bps,
        est_fill_price=est_fill,
    )

# ------------------- Class wrapper -------------------

class CostModel:
    """
    Reusable cost model with optional per-symbol overrides (spread/ADV),
    handy for pre-trade sizing and TCA benchmarks.
    """
    def __init__(self, venue: str = "us_ibkr"):
        self.venue = venue
        self.spread_overrides_bps: Dict[str, float] = {}
        self.adv_overrides: Dict[str, float] = {}

    def set_spread(self, symbol: str, spread_bps: float) -> None:
        self.spread_overrides_bps[symbol.upper()] = float(spread_bps)

    def set_adv(self, symbol: str, adv_shares: float) -> None:
        self.adv_overrides[symbol.upper()] = float(adv_shares)

    def cost_for(self, symbol: str, side: str, qty: float, price: float, *,
                 maker: bool = False, spread_bps: Optional[float] = None,
                 adv: Optional[float] = None, tif: Optional[str] = None) -> CostBreakdown:
        sp = spread_bps if spread_bps is not None else self.spread_overrides_bps.get(symbol.upper())
        av = adv if adv is not None else self.adv_overrides.get(symbol.upper())
        return estimate_cost(CostInputs(
            symbol=symbol, side=side, price=price, qty=qty,
            adv=av, spread_bps=sp, maker=maker, venue=self.venue, tif=tif
        ))