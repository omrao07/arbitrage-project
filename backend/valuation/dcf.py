#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dcf.py
------
Discounted Cash Flow (DCF) valuation model.

Supports:
  - Forecasting free cash flows (FCF)
  - Terminal Value via Gordon Growth (perpetuity) or Exit Multiple
  - Discounting using WACC
  - Deriving Enterprise Value (EV), Equity Value, and implied Share Price

Usage:
    from dcf import run_dcf

    fcfs = [500, 550, 600, 650, 700]  # forecasted FCFs in millions
    ev, eq_val, price = run_dcf(
        fcfs=fcfs,
        wacc=0.09,
        terminal_growth=0.025,
        net_debt=2000,
        shares_outstanding=500
    )
    print("Enterprise Value:", ev)
    print("Equity Value:", eq_val)
    print("Implied Price:", price)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Literal, Tuple


@dataclass
class DCFInputs:
    fcfs: List[float]                   # Forecasted free cash flows (millions)
    wacc: float                         # Weighted average cost of capital (decimal, e.g. 0.09 = 9%)
    terminal_growth: float = 0.02       # Long-term growth rate (decimal)
    exit_multiple: float | None = None  # Exit EV/EBITDA multiple (if using multiple method)
    ebitda_terminal: float | None = None# EBITDA for terminal year (if using multiple method)
    net_debt: float = 0.0               # Net debt (debt - cash), millions
    shares_outstanding: float = 1.0     # Number of shares outstanding (millions)
    terminal_method: Literal["perpetuity", "multiple"] = "perpetuity"


def discount_cashflows(fcfs: List[float], wacc: float) -> List[float]:
    """Discount each forecasted cash flow back to present value."""
    return [fcf / ((1 + wacc) ** (t+1)) for t, fcf in enumerate(fcfs)]


def terminal_value(inputs: DCFInputs) -> float:
    """Calculate terminal value using perpetuity growth or exit multiple."""
    if inputs.terminal_method == "perpetuity":
        fcf_last = inputs.fcfs[-1]
        return fcf_last * (1 + inputs.terminal_growth) / (inputs.wacc - inputs.terminal_growth)
    elif inputs.terminal_method == "multiple":
        if inputs.exit_multiple is None or inputs.ebitda_terminal is None:
            raise ValueError("Exit multiple method requires exit_multiple and ebitda_terminal")
        return inputs.exit_multiple * inputs.ebitda_terminal
    else:
        raise ValueError(f"Unknown terminal method: {inputs.terminal_method}")


def run_dcf(
    fcfs: List[float],
    wacc: float,
    terminal_growth: float = 0.02,
    exit_multiple: float | None = None,
    ebitda_terminal: float | None = None,
    net_debt: float = 0.0,
    shares_outstanding: float = 1.0,
    terminal_method: Literal["perpetuity","multiple"] = "perpetuity"
) -> Tuple[float, float, float]:
    """
    Run a DCF valuation and return (Enterprise Value, Equity Value, Price per share).
    """
    inputs = DCFInputs(
        fcfs=fcfs,
        wacc=wacc,
        terminal_growth=terminal_growth,
        exit_multiple=exit_multiple,
        ebitda_terminal=ebitda_terminal,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
        terminal_method=terminal_method,
    )

    # 1. PV of forecasted FCFs
    pv_fcfs = discount_cashflows(inputs.fcfs, inputs.wacc)

    # 2. Terminal value
    tv = terminal_value(inputs)
    pv_tv = tv / ((1 + inputs.wacc) ** len(inputs.fcfs))

    # 3. Enterprise Value
    ev = sum(pv_fcfs) + pv_tv

    # 4. Equity Value
    eq_val = ev - inputs.net_debt

    # 5. Per share
    price = eq_val / inputs.shares_outstanding

    return ev, eq_val, price


# -----------------------
# CLI helper
# -----------------------

if __name__ == "__main__":
    # Demo run
    fcfs = [500, 550, 600, 650, 700]  # forecasted FCFs (millions)
    ev, eq_val, price = run_dcf(
        fcfs=fcfs, # type: ignore
        wacc=0.09,
        terminal_growth=0.025,
        net_debt=2000,
        shares_outstanding=500
    )
    print(f"Enterprise Value: {ev:,.2f}m")
    print(f"Equity Value:     {eq_val:,.2f}m")
    print(f"Price/Share:      {price:,.2f}")