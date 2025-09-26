#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompts.py
----------
Centralized prompt templates and prompt-builder utilities.

Why?
- Avoid hardcoding long prompt strings in multiple places
- Consistent style across valuation models, strategy explainers, risk notes, etc.
- Easy to extend: add new prompt sections as constants or functions

Usage
-----
from prompts import PROMPTS, build_strategy_prompt

text = build_strategy_prompt("yen carry trade", context={"region":"Japan", "asset":"FX"})
response = llm.generate(text)
"""

from __future__ import annotations
from typing import Dict, Any

# ----------------------------------------------------------------------
# Static templates
# ----------------------------------------------------------------------

PROMPTS: Dict[str, str] = {
    "valuation_dcf": (
        "You are a financial analyst trained in DCF valuation. "
        "Explain step by step the inputs, formulas, and risks. "
        "Use clear structure: Assumptions, Formulas, Results, Limitations."
    ),
    "valuation_relative": (
        "You are comparing valuation multiples across sectors. "
        "Explain EV/EBITDA, P/E, P/B, and Price/Sales. "
        "Highlight why certain multiples apply better in specific industries."
    ),
    "risk_management": (
        "You are a CRO at a hedge fund. "
        "Summarize the risk exposures in VaR, stress, and liquidity. "
        "Provide actionable insights for portfolio hedging."
    ),
    "macro_strategy": (
        "You are a macro strategist at a hedge fund. "
        "Explain the economic rationale, triggers, and risks of the given strategy."
    ),
    "technical_analysis": (
        "You are a market technician. "
        "Describe patterns, momentum indicators, and entry/exit rules "
        "for the given asset and timeframe."
    ),
    "fundamental_analysis": (
        "You are a fundamental analyst. "
        "Provide drivers of revenue, margins, and cashflows for the given company."
    ),
    "regression_model": (
        "You are a quant researcher. "
        "Interpret regression output (coefficients, t-stats, R²). "
        "Explain what it implies for trading signals."
    ),
    "explain_code": (
        "You are a senior quant developer. "
        "Explain the given Python/TSX code in plain English for an intern. "
        "Highlight risks, assumptions, and performance concerns."
    ),
    "strategy_summary": (
        "Summarize the given trading strategy in 5 bullets: "
        "Mechanics, Inputs, Signals, Risks, Edge."
    ),
}

# ----------------------------------------------------------------------
# Prompt builder utilities
# ----------------------------------------------------------------------

def build_strategy_prompt(name: str, context: Dict[str, Any] | None = None) -> str:
    """
    Build a macro/strategy explainer prompt.
    name    : strategy name (e.g., "yen carry trade", "nuclear restarts")
    context : extra dict of parameters (e.g., {"region": "Japan", "asset": "FX"})
    """
    base = PROMPTS["macro_strategy"]
    ctx_str = ""
    if context:
        ctx_str = " Context: " + ", ".join(f"{k}={v}" for k, v in context.items())
    return f"{base}\n\nStrategy: {name}.{ctx_str}"

def build_valuation_prompt(kind: str, ticker: str, context: Dict[str, Any] | None = None) -> str:
    """
    Build valuation prompt for 'dcf' or 'relative'.
    """
    if kind not in ("dcf", "relative"):
        raise ValueError("kind must be 'dcf' or 'relative'")
    base = PROMPTS[f"valuation_{kind}"]
    ctx_str = f" Company: {ticker}."
    if context:
        ctx_str += " " + " ".join(f"{k}={v}" for k, v in context.items())
    return f"{base}\n\n{ctx_str}"

def build_risk_prompt(context: Dict[str, Any]) -> str:
    """
    Build risk management prompt with portfolio context.
    context should include e.g. gross_exposure, var_95, stress_loss.
    """
    base = PROMPTS["risk_management"]
    ctx_str = " Context: " + ", ".join(f"{k}={v}" for k, v in context.items())
    return f"{base}\n\n{ctx_str}"

# ----------------------------------------------------------------------
# Example run
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Example Strategy Prompt:\n")
    print(build_strategy_prompt("Yen Carry Trade", {"region": "Japan", "asset": "FX"}))

    print("\nExample Valuation Prompt:\n")
    print(build_valuation_prompt("dcf", "AAPL", {"horizon":"5y", "wacc":"8%"}))