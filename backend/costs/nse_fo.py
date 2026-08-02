"""
Canonical NSE F&O transaction-cost model — the single source of rate truth.

WHY THIS EXISTS
---------------
The backtesters in this repo charge a flat `fee_bps` (0.3 bps in
`backtester/shadow_runner.py`, 5.0 bps in `backtester/backtest_engine.py`).
The real cost of selling an NSE option is dominated by STT at 0.15% of premium
= 150 bps on that leg alone. A backtest charging 0.3 bps against a ~20 bps
reality overstates P&L by more than an order of magnitude, which is enough to
turn a losing short-premium strategy into a "profitable" one on paper.

RATE PROVENANCE AND STALENESS  ** READ THIS **
----------------------------------------------
The rates below are transcribed from the D-Strategies plan (Part 3 / Appendix B,
July 2026), which states F&O STT was hiked effective 1 April 2026:
    options sell 0.10% -> 0.15%,  futures sell 0.02% -> 0.05%.

These numbers are PERISHABLE and are NOT independently verified here. Every one
of them changed at least once in the 18 months before that document was written.
Before you trust a P&L number produced with this module:

  1. Reconcile `option_costs()` against a real Zerodha contract note, to the
     rupee, on a sample of your own trades.
  2. Re-check the rates each Union Budget cycle.
  3. Fix them HERE, not in a strategy file. Nothing else may hardcode a rate.

`verify_against_contract_note()` at the bottom exists to make step 1 mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, int, np.ndarray]

# ── Rates: mid-2026. CENTRALISED. Re-verify each budget cycle. ───────────────
RATES_AS_OF = "2026-07"
RATES_SOURCE = "D-Strategies plan Part 3 / Appendix B — UNVERIFIED, reconcile to a contract note"

STT_OPT_SELL = 0.0015       # 0.15% of premium,  SELL only (eff 1-Apr-2026)
STT_FUT_SELL = 0.0005       # 0.05% of turnover, SELL only (eff 1-Apr-2026)
EXCH_OPT = 0.0003503        # 0.03503% of premium,  both sides
EXCH_FUT = 0.0000173        # 0.00173% of turnover, both sides
SEBI_FEE = 0.000001         # 0.0001%, both sides
GST_RATE = 0.18             # 18% of (brokerage + exchange txn + SEBI fee)
STAMP_OPT_BUY = 0.00003     # 0.003% of premium,  BUY only
STAMP_FUT_BUY = 0.00002     # 0.002% of turnover, BUY only
BROKERAGE_FLAT = 20.0       # INR per executed order (Zerodha)
BROKERAGE_FUT_PCT = 0.0003  # futures: min(0.03% of turnover, flat)


@dataclass(frozen=True, slots=True)
class Costs:
    """Per-leg cost breakdown in INR. `total` is what actually leaves the account."""

    brokerage: float
    stt: float
    exchange_txn: float
    sebi_fee: float
    gst: float
    stamp_duty: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi_fee
            + self.gst
            + self.stamp_duty
        )

    def bps_of(self, notional: float) -> float:
        """Total cost expressed in bps of a given notional (for comparability)."""
        return 0.0 if notional <= 0 else self.total / notional * 1e4


def _validate(value: np.ndarray, what: str) -> np.ndarray:
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(
            f"Non-finite/non-positive {what} in cost calc — data integrity failure, HALT"
        )
    return value


# ── Vectorised (the form backtests must use) ────────────────────────────────

def option_costs_vec(
    premium: ArrayLike,
    lots: ArrayLike,
    lot_size: int,
    is_sell: ArrayLike,
) -> np.ndarray:
    """
    Per-leg option cost in INR. `premium` is per-share; V = premium*lots*lot_size.

    STT is charged on the SELL leg only and stamp duty on the BUY leg only —
    that asymmetry is why short-premium strategies eat the STT hike on entry.
    """
    prem = _validate(np.asarray(premium, dtype=float), "premium")
    n_lots = _validate(np.asarray(lots, dtype=float), "lots")
    sell = np.asarray(is_sell, dtype=bool)
    if lot_size <= 0:
        raise ValueError(f"lot_size must be positive, got {lot_size}")

    V = prem * n_lots * lot_size
    brokerage = np.full_like(V, BROKERAGE_FLAT)
    stt = np.where(sell, STT_OPT_SELL * V, 0.0)
    exch = EXCH_OPT * V
    sebi = SEBI_FEE * V
    gst = GST_RATE * (brokerage + exch + sebi)
    stamp = np.where(sell, 0.0, STAMP_OPT_BUY * V)
    return brokerage + stt + exch + sebi + gst + stamp


def future_costs_vec(
    price: ArrayLike,
    lots: ArrayLike,
    lot_size: int,
    is_sell: ArrayLike,
) -> np.ndarray:
    """Per-leg futures cost in INR. Brokerage is min(0.03% of turnover, ₹20)."""
    px = _validate(np.asarray(price, dtype=float), "price")
    n_lots = _validate(np.asarray(lots, dtype=float), "lots")
    sell = np.asarray(is_sell, dtype=bool)
    if lot_size <= 0:
        raise ValueError(f"lot_size must be positive, got {lot_size}")

    V = px * n_lots * lot_size
    brokerage = np.minimum(BROKERAGE_FUT_PCT * V, BROKERAGE_FLAT)
    stt = np.where(sell, STT_FUT_SELL * V, 0.0)
    exch = EXCH_FUT * V
    sebi = SEBI_FEE * V
    gst = GST_RATE * (brokerage + exch + sebi)
    stamp = np.where(sell, 0.0, STAMP_FUT_BUY * V)
    return brokerage + stt + exch + sebi + gst + stamp


# ── Scalar (for reconciliation against a contract note) ─────────────────────

def option_costs(premium: float, lots: int, lot_size: int, side: str) -> Costs:
    """Itemised single-leg option cost. `side` is 'BUY' or 'SELL'."""
    s = side.upper()
    if s not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    is_sell = s == "SELL"

    V = float(_validate(np.asarray(premium, float), "premium")) * lots * lot_size
    brokerage = BROKERAGE_FLAT
    exch = EXCH_OPT * V
    sebi = SEBI_FEE * V
    return Costs(
        brokerage=brokerage,
        stt=STT_OPT_SELL * V if is_sell else 0.0,
        exchange_txn=exch,
        sebi_fee=sebi,
        gst=GST_RATE * (brokerage + exch + sebi),
        stamp_duty=0.0 if is_sell else STAMP_OPT_BUY * V,
    )


def future_costs(price: float, lots: int, lot_size: int, side: str) -> Costs:
    """Itemised single-leg futures cost. `side` is 'BUY' or 'SELL'."""
    s = side.upper()
    if s not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    is_sell = s == "SELL"

    V = float(_validate(np.asarray(price, float), "price")) * lots * lot_size
    brokerage = min(BROKERAGE_FUT_PCT * V, BROKERAGE_FLAT)
    exch = EXCH_FUT * V
    sebi = SEBI_FEE * V
    return Costs(
        brokerage=brokerage,
        stt=STT_FUT_SELL * V if is_sell else 0.0,
        exchange_txn=exch,
        sebi_fee=sebi,
        gst=GST_RATE * (brokerage + exch + sebi),
        stamp_duty=0.0 if is_sell else STAMP_FUT_BUY * V,
    )


def option_round_trip(
    entry_premium: float,
    exit_premium: float,
    lots: int,
    lot_size: int,
    short: bool = True,
) -> float:
    """
    All-in INR cost of open+close. `short=True` sells to open, buys to close —
    the short-premium case that pays STT on the (larger) entry premium.
    """
    open_side, close_side = ("SELL", "BUY") if short else ("BUY", "SELL")
    return (
        option_costs(entry_premium, lots, lot_size, open_side).total
        + option_costs(exit_premium, lots, lot_size, close_side).total
    )


# ── Slippage (separate from statutory costs) ────────────────────────────────

def slippage(
    mid: ArrayLike,
    half_spread_bps: ArrayLike,
    participation: ArrayLike = 0.0,
    impact_coef: float = 0.1,
) -> np.ndarray:
    """
    Half-spread + square-root market impact (Almgren-style), in price units.

    `half_spread_bps` must be moneyness- and regime-conditional, not a constant:
    on OTM Nifty weeklies near expiry the bid-ask can be 5-15% of premium, so a
    backtest filling at mid is fantasy. Calibrate `impact_coef` from your own
    shadow-engine fills — a textbook number here is a guess.
    """
    m = np.asarray(mid, dtype=float)
    spread_cost = m * (np.asarray(half_spread_bps, dtype=float) / 1e4)
    part = np.clip(np.asarray(participation, dtype=float), 0.0, None)
    return spread_cost + m * impact_coef * np.sqrt(part)


# ── Reconciliation harness ──────────────────────────────────────────────────

def verify_against_contract_note(
    premium: float,
    lots: int,
    lot_size: int,
    side: str,
    broker_total_inr: float,
    tolerance_inr: float = 1.0,
) -> dict:
    """
    Compare this model to a real broker charge. Phase-2 gate in the plan is
    "cost model matches a real Zerodha contract note to the rupee".

    Returns the itemised diff; `matches` is False if you may not trust any
    backtest P&L produced with these rates.
    """
    modelled = option_costs(premium, lots, lot_size, side)
    diff = modelled.total - broker_total_inr
    return {
        "modelled_total": round(modelled.total, 2),
        "broker_total": round(broker_total_inr, 2),
        "difference_inr": round(diff, 2),
        "matches": abs(diff) <= tolerance_inr,
        "breakdown": {
            "brokerage": round(modelled.brokerage, 2),
            "stt": round(modelled.stt, 2),
            "exchange_txn": round(modelled.exchange_txn, 2),
            "sebi_fee": round(modelled.sebi_fee, 4),
            "gst": round(modelled.gst, 2),
            "stamp_duty": round(modelled.stamp_duty, 2),
        },
        "rates_as_of": RATES_AS_OF,
        "rates_source": RATES_SOURCE,
    }
