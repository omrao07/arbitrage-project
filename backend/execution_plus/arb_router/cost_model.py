# backend/execution_plus/cost_model.py
"""
Execution cost model(s).

Purpose
-------
Provide pre‑trade and immediate post‑trade cost estimates for routing/venue choice.

Inputs:
  - AdapterBase (for venue cfg: fee bps, latency, base currency)
  - Order (side, qty, type, limit)
  - Quote (bid/ask/mid)
  - Optional market parameters (spread_bps, impact params, fx rates)

Outputs:
  - CostBreakdown: fees, slippage (spread + impact), latency/adverse selection, fx, total

Env knobs (defaults shown):
  COST_SPREAD_BPS=8                 # fallback spread if quote missing
  COST_IMPACT_COEFF=9e-4            # square‑root impact coefficient k
  COST_IMPACT_VOL_DOLLARS=5_000_000 # daily $ volume per symbol (fallback)
  COST_LATENCY_BPS_PER_100MS=0.6    # adverse selection drift per 100ms
  COST_FX_USD_PER_BASE=1.0          # FX rate (USD/base) fallback if needed

Notes
-----
- Square‑root impact:  k * sign * (notional / vol_dollars)**0.5 * mid
- Latency drift:       (latency_ms/100ms) * coeff * mid
- Slippage for MARKET: half‑spread + impact
- Slippage for LIMIT:  0 if price is marketable; else probability_of_fill * same formula
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .adapters import AdapterBase, Order, OrderType, Side, Quote # type: ignore


# ----------------------------- dataclasses -----------------------------

@dataclass
class CostBreakdown:
    notional: float                  # side‑agnostic absolute notional in venue ccy
    fees: float                      # venue taker/maker fees
    spread_slippage: float           # half‑spread or better based on order type
    impact: float                    # market impact component
    latency_adverse: float           # drift while waiting/latency
    fx: float                        # FX conversion cost to P&L currency (USD)
    total: float                     # sum of above (in USD by default)
    meta: Dict[str, Any]             # parameters used


# ----------------------------- interface -------------------------------

class AbstractCostModel:
    def estimate(
        self,
        adapter: AdapterBase,
        order: Order,
        quote: Quote,
        *,
        vol_dollars: Optional[float] = None,
        fx_usd_per_base: Optional[float] = None,
        taker: Optional[bool] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> CostBreakdown:
        raise NotImplementedError


# ----------------------------- helpers ---------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _half_spread_from_quote(q: Quote, fallback_bps: float) -> float:
    if q.bid is not None and q.ask is not None and q.bid > 0 and q.ask > q.bid:
        return 0.5 * (q.ask - q.bid)
    # fall back to bps of mid
    mid = q.mid or ((q.bid or 0.0) + (q.ask or 0.0)) / 2.0
    return (fallback_bps / 10_000.0) * max(mid, 0.0)

def _order_notional(order: Order, q: Quote) -> float:
    px = order.limit_price if (order.type == OrderType.LIMIT and order.limit_price) else (q.mid or q.ask or q.bid or 0.0)
    return abs(float(order.qty) * float(px))

def _p_fill_limit(order: Order, q: Quote) -> float:
    """
    Very rough: if BUY and limit >= ask or SELL and limit <= bid -> marketable (p~1).
    Otherwise drop off linearly as limit moves away from best by half‑spread.
    """
    if order.type != OrderType.LIMIT or order.limit_price is None:
        return 1.0
    bid, ask, mid = q.bid, q.ask, q.mid
    hs = _half_spread_from_quote(q, _env_float("COST_SPREAD_BPS", 8.0))
    if bid is None or ask is None:
        return 0.5
    lp = float(order.limit_price)
    if order.side == Side.BUY:
        if lp >= ask:  # marketable
            return 1.0
        # scale by distance from ask over half‑spread
        return max(0.0, 1.0 - (ask - lp) / max(hs, 1e-9))
    else:
        if lp <= bid:
            return 1.0
        return max(0.0, 1.0 - (lp - bid) / max(hs, 1e-9))

def _sign(side: Side) -> int:
    return +1 if side == Side.BUY else -1


# ----------------------------- default model ---------------------------

class DefaultCostModel(AbstractCostModel):
    def __init__(
        self,
        *,
        spread_bps_fallback: float = None, # type: ignore
        impact_coeff: float = None, # type: ignore
        impact_vol_dollars: float = None, # type: ignore
        latency_bps_per_100ms: float = None, # type: ignore
    ) -> None:
        self.spread_bps_fallback = spread_bps_fallback if spread_bps_fallback is not None else _env_float("COST_SPREAD_BPS", 8.0)
        self.impact_coeff = impact_coeff if impact_coeff is not None else _env_float("COST_IMPACT_COEFF", 9e-4)
        self.impact_vol_dollars = impact_vol_dollars if impact_vol_dollars is not None else _env_float("COST_IMPACT_VOL_DOLLARS", 5_000_000.0)
        self.latency_bps_per_100ms = latency_bps_per_100ms if latency_bps_per_100ms is not None else _env_float("COST_LATENCY_BPS_PER_100MS", 0.6)

    def estimate(
        self,
        adapter: AdapterBase,
        order: Order,
        quote: Quote,
        *,
        vol_dollars: Optional[float] = None,
        fx_usd_per_base: Optional[float] = None,
        taker: Optional[bool] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> CostBreakdown:
        """
        Returns CostBreakdown with all values in **USD** (using fx_usd_per_base).
        """
        extras = dict(extras or {})
        mid = quote.mid or ((quote.bid or 0.0) + (quote.ask or 0.0)) / 2.0
        notional_base = _order_notional(order, quote)  # in venue base ccy
        fx = fx_usd_per_base if fx_usd_per_base is not None else _env_float("COST_FX_USD_PER_BASE", 1.0)
        notional_usd = notional_base * fx

        # --- fees ---
        # Assume taker for MARKET, maker for non‑marketable LIMIT; allow override
        if taker is None:
            if order.type == OrderType.MARKET:
                taker = True
            else:
                taker = _p_fill_limit(order, quote) >= 1.0  # marketable limit counts as taker
        bps = adapter.fee_bps(taker=bool(taker))
        fees_base = notional_base * (bps / 10_000.0)
        fees_usd = fees_base * fx

        # --- spread slippage ---
        half_spread = _half_spread_from_quote(quote, self.spread_bps_fallback)
        if order.type == OrderType.MARKET:
            spread_slip_base = half_spread * order.qty  # paying half‑spread expected
        else:
            p_fill = _p_fill_limit(order, quote)
            # if non‑marketable, expected slippage reduces linearly with fill probability
            spread_slip_base = p_fill * half_spread * order.qty
        spread_slip_usd = spread_slip_base * fx

        # --- impact (square‑root model) ---
        V = max(1.0, float(vol_dollars or self.impact_vol_dollars))
        k = float(self.impact_coeff)
        # price move in $ terms ≈ k * sign * sqrt(notional_usd / V) * mid
        # cost is |move| * qty ; using mid as reference
        impact_move_per_share = k * math.sqrt(max(0.0, notional_usd / V)) * (1.0 if mid <= 0 else 1.0)
        impact_base = abs(impact_move_per_share) * order.qty
        impact_usd = impact_base * fx

        # --- latency / adverse selection ---
        lat_ms = max(0, int(adapter.latency_ms()))
        lat_bps = (lat_ms / 100.0) * self.latency_bps_per_100ms
        lat_move_per_share = (lat_bps / 10_000.0) * (mid or 0.0)
        latency_base = abs(lat_move_per_share) * order.qty
        latency_usd = latency_base * fx

        # --- fx cost (placeholder) ---
        # If venue base currency != USD, charge a tiny conversion spread (1 bps of notional) unless fx==1.0
        fx_cost_usd = 0.0 if abs(fx - 1.0) < 1e-9 else (notional_usd * 0.0001)

        total_usd = fees_usd + spread_slip_usd + impact_usd + latency_usd + fx_cost_usd

        meta = {
            "mid": mid,
            "fee_bps": bps,
            "half_spread_fallback_bps": self.spread_bps_fallback,
            "impact_coeff": self.impact_coeff,
            "impact_vol_dollars": V,
            "latency_ms": lat_ms,
            "latency_bps_per_100ms": self.latency_bps_per_100ms,
            "fx_usd_per_base": fx,
            "taker": taker,
            "p_fill_limit": _p_fill_limit(order, quote) if order.type == OrderType.LIMIT else 1.0,
        }
        meta.update(extras)

        return CostBreakdown(
            notional=notional_usd,
            fees=fees_usd,
            spread_slippage=spread_slip_usd,
            impact=impact_usd,
            latency_adverse=latency_usd,
            fx=fx_cost_usd,
            total=total_usd,
            meta=meta,
        )


# ----------------------------- convenience ----------------------------

def get_default_model() -> DefaultCostModel:
    return DefaultCostModel()