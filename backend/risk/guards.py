#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guards.py
---------
Centralized runtime guardrails: inputs, portfolio limits, and risk checks.

Usage
-----
from guards import Guards, GuardError
g = Guards(risk_limits={"gross": 2.0, "net": 1.0, "max_drawdown": 0.20})
g.check_weights({"AAPL": 0.6, "TSLA": -0.4})
g.check_signal({"AAPL": 1.2, "TSLA": -0.8})
g.check_order({"symbol":"AAPL","qty":100,"price":150})
g.check_drawdown([100_000, 101_000, 97_000, 103_000])
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Sequence

import numpy as np


class GuardError(Exception):
    """Raised when a guard check fails."""


class Guards:
    def __init__(self,
                 risk_limits: Optional[Dict[str, float]] = None,
                 allow_empty: bool = False):
        """
        risk_limits: optional keys (floats):
            - gross: max sum(|weights|)
            - net:   max |sum(weights)|
            - max_drawdown: max allowed peak-to-trough loss as fraction (e.g., 0.2 for -20%)
        allow_empty: if True, skip empty-dict/sequence errors (useful in dry runs)
        """
        self.risk_limits = risk_limits or {}
        self.allow_empty = bool(allow_empty)

    # ------------------------------------------------------------------ #
    # General primitives
    # ------------------------------------------------------------------ #

    def not_none(self, x: Any, name: str = "value") -> Any:
        if x is None:
            raise GuardError(f"{name} must not be None")
        return x

    def finite(self, x: float, name: str = "value") -> float:
        if x is None or not math.isfinite(float(x)):
            raise GuardError(f"{name} must be finite (got {x})")
        return float(x)

    def positive(self, x: float, name: str = "value", strict: bool = True) -> float:
        xv = self.finite(x, name)
        if strict and xv <= 0:
            raise GuardError(f"{name} must be > 0 (got {xv})")
        if not strict and xv < 0:
            raise GuardError(f"{name} must be >= 0 (got {xv})")
        return xv

    def between(self, x: float, lo: float, hi: float, name: str = "value") -> float:
        xv = self.finite(x, name)
        if not (lo <= xv <= hi):
            raise GuardError(f"{name}={xv} out of bounds [{lo}, {hi}]")
        return xv

    # ------------------------------------------------------------------ #
    # Portfolio weights & exposures
    # ------------------------------------------------------------------ #

    def check_weights(self, weights: Dict[str, float]) -> None:
        """
        Validate portfolio weights:
          - non-empty (unless allow_empty)
          - finite values
          - gross exposure <= limit
          - |net exposure| <= limit
        """
        if not weights:
            if self.allow_empty:
                return
            raise GuardError("weights dict is empty")

        vals = np.array(list(weights.values()), dtype=float)
        if not np.all(np.isfinite(vals)):
            raise GuardError("weights contain NaN/inf")

        gross = float(np.abs(vals).sum())
        net = float(vals.sum())

        if "gross" in self.risk_limits and gross > float(self.risk_limits["gross"]) + 1e-12:
            raise GuardError(f"Gross exposure {gross:.4f} > limit {float(self.risk_limits['gross']):.4f}")
        if "net" in self.risk_limits and abs(net) > float(self.risk_limits["net"]) + 1e-12:
            raise GuardError(f"Net exposure {net:.4f} > limit {float(self.risk_limits['net']):.4f}")

    # Optional: per-asset cap (e.g., |w_i| <= cap)
    def cap_per_asset(self, weights: Dict[str, float], cap: float = 0.1) -> None:
        cap = float(cap)
        for k, v in weights.items():
            if abs(float(v)) > cap + 1e-12:
                raise GuardError(f"Weight cap exceeded for {k}: {v:.4f} > {cap:.4f}")

    # ------------------------------------------------------------------ #
    # PnL / equity & drawdown
    # ------------------------------------------------------------------ #

    def check_drawdown(self, equity_curve: Sequence[float]) -> None:
        """
        Enforce max_drawdown limit if configured.
        equity_curve: sequence of portfolio values over time.
        """
        if (equity_curve is None or len(equity_curve) == 0):
            if self.allow_empty:
                return
            raise GuardError("equity_curve is empty")

        arr = np.asarray(equity_curve, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise GuardError("equity_curve contains NaN/inf")

        roll_max = np.maximum.accumulate(arr)
        dd = arr / (roll_max + 1e-12) - 1.0
        maxdd = float(dd.min())

        if "max_drawdown" in self.risk_limits:
            lim = float(self.risk_limits["max_drawdown"])
            if maxdd < -abs(lim):
                raise GuardError(f"Max drawdown {maxdd:.4f} < -{abs(lim):.4f} limit")

    # ------------------------------------------------------------------ #
    # Signals
    # ------------------------------------------------------------------ #

    def check_signal(self,
                     signal: Dict[str, float],
                     min_val: float = -10.0,
                     max_val: float = 10.0) -> None:
        """
        Validate trading signal dict: finite & bounded per name.
        """
        if not signal:
            if self.allow_empty:
                return
            raise GuardError("signal dict is empty")
        for k, v in signal.items():
            if not math.isfinite(float(v)):
                raise GuardError(f"Signal {k} invalid (NaN/inf): {v}")
            if not (min_val <= float(v) <= max_val):
                raise GuardError(f"Signal {k}={v} outside range [{min_val}, {max_val}]")

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #

    def check_order(self, order: Dict[str, Any]) -> None:
        """
        Validate a single order {symbol, qty, price}.
        """
        for key in ("symbol", "qty", "price"):
            if key not in order:
                raise GuardError(f"Order missing '{key}'")

        qty = self.finite(order["qty"], "qty")
        price = self.finite(order["price"], "price")
        if qty == 0:
            raise GuardError("Order qty must not be 0")
        if price <= 0:
            raise GuardError("Order price must be > 0")

    def check_orders(self, orders: List[Dict[str, Any]]) -> None:
        """
        Validate a batch of orders.
        """
        if not orders:
            if self.allow_empty:
                return
            raise GuardError("orders list is empty")
        for od in orders:
            self.check_order(od)

    # ------------------------------------------------------------------ #
    # Turnover / trade-rate guard (optional)
    # ------------------------------------------------------------------ #

    def check_turnover(self, traded_notional: float, equity: float, cap: float = 0.5) -> None:
        """
        Enforce per-step turnover cap: traded_notional / equity <= cap
        """
        tn = self.positive(traded_notional, "traded_notional", strict=True)
        eq = self.positive(equity, "equity", strict=True)
        if (tn / eq) > float(cap) + 1e-12:
            raise GuardError(f"Turnover {(tn/eq):.4f} > cap {float(cap):.4f}")

    # ------------------------------------------------------------------ #
    # Price & data sanity
    # ------------------------------------------------------------------ #

    def check_price_series(self, px: Sequence[float], name: str = "price") -> None:
        if (px is None or len(px) == 0):
            if self.allow_empty:
                return
            raise GuardError(f"{name} series is empty")
        arr = np.asarray(px, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise GuardError(f"{name} series contains NaN/inf")
        if (arr <= 0).any():
            raise GuardError(f"{name} series has non-positive values")

    # Convenience wrapper to assert no-op (for integration hooks)
    def ok(self) -> None:
        """No-op to anchor guard usage in pipelines."""
        return


# ------------------------------- quick demo ---------------------------------- #
if __name__ == "__main__":
    g = Guards(risk_limits={"gross": 2.0, "net": 1.0, "max_drawdown": 0.20})

    # weights
    g.check_weights({"AAPL": 0.6, "TSLA": -0.4})
    g.cap_per_asset({"AAPL": 0.6, "TSLA": -0.4}, cap=0.7)

    # signals
    g.check_signal({"AAPL": 1.2, "TSLA": -0.8}, min_val=-5, max_val=5)

    # orders
    g.check_order({"symbol": "AAPL", "qty": 100, "price": 150.0})
    g.check_orders([
        {"symbol": "AAPL", "qty": 10, "price": 150.0},
        {"symbol": "TSLA", "qty": -5, "price": 250.0},
    ])

    # equity / dd
    g.check_drawdown([100_000, 101_000, 98_000, 104_000, 96_000, 110_000])

    # price series
    g.check_price_series([100, 101, 103, 102])

    print("All guards passed ✅")