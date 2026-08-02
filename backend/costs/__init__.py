"""
Transaction-cost models. `nse_fo` is the single source of rate truth for
NSE F&O — no strategy, backtester, or notebook may hardcode a rate.
"""

from backend.costs.nse_fo import (
    BROKERAGE_FLAT,
    EXCH_FUT,
    EXCH_OPT,
    GST_RATE,
    RATES_AS_OF,
    RATES_SOURCE,
    SEBI_FEE,
    STAMP_FUT_BUY,
    STAMP_OPT_BUY,
    STT_FUT_SELL,
    STT_OPT_SELL,
    Costs,
    future_costs,
    future_costs_vec,
    option_costs,
    option_costs_vec,
    option_round_trip,
    slippage,
    verify_against_contract_note,
)

__all__ = [
    "Costs",
    "option_costs",
    "option_costs_vec",
    "future_costs",
    "future_costs_vec",
    "option_round_trip",
    "slippage",
    "verify_against_contract_note",
    "RATES_AS_OF",
    "RATES_SOURCE",
    "STT_OPT_SELL",
    "STT_FUT_SELL",
    "EXCH_OPT",
    "EXCH_FUT",
    "SEBI_FEE",
    "GST_RATE",
    "STAMP_OPT_BUY",
    "STAMP_FUT_BUY",
    "BROKERAGE_FLAT",
]
