# engines/equity_ls/runner.py
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Literal

from engines.equity_ls.signals import momentum, value, quality, sector_rotation # type: ignore
from engines.equity_ls.execution.allocator import allocate_from_scores, generate_orders # type: ignore
from engines.equity_ls.execution.slippage import apply_to_orders # type: ignore
from engines.equity_ls.execution.order_router import ( # type: ignore
    Order, default_router, ExecutionReport
)
from engines.equity_ls.backtest.pnl import run_equity_ls_pnl # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SignalType = Literal["momentum", "value", "quality", "sector_rotation"]


def build_signal(
    signal_type: SignalType,
    prices: pd.DataFrame,
    fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    **kwargs,
) -> pd.Series:
    """Dispatch to the correct signal module."""
    if signal_type == "momentum":
        return momentum.build_signal(prices, sector_map=sector_map, **kwargs)
    elif signal_type == "value":
        return value.build_signal(fundamentals=fundamentals, prices=prices, sector_map=sector_map, **kwargs)
    elif signal_type == "quality":
        return quality.build_signal(fundamentals=fundamentals, sector_map=sector_map, **kwargs)
    elif signal_type == "sector_rotation":
        return sector_rotation.build_signal(prices, sectors_map=sector_map or sector_rotation.DEFAULT_SECTORS, **kwargs)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


def run_equity_ls_strategy(
    signal_type: SignalType,
    prices: pd.DataFrame,
    fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    current_pos: Optional[pd.Series] = None,
    nav: float = 1_000_000.0,
    adv_map: Optional[Dict[str, float]] = None,
    slippage_model: str = "sqrt_impact",
    **kwargs,
) -> Dict:
    """
    Full pipeline:
      1. Build signal (momentum/value/quality/sector_rotation)
      2. Map to weights, dollars, shares
      3. Generate orders vs. current positions
      4. Apply slippage model
      5. Route orders via broker (paper by default)
      6. Run PnL backtest
    Returns: dict with weights, orders, reports, pnl
    """
    logger.info(f"Running equity_ls strategy: {signal_type}")

    # --- Step 1: Signal ---
    w = build_signal(signal_type, prices, fundamentals=fundamentals, sector_map=sector_map, **kwargs)
    if w.empty:
        return {"weights": w, "orders": pd.DataFrame(), "reports": [], "pnl": {}}

    # --- Step 2: Weights → target shares ---
    last_prices = prices.iloc[-1]
    weights, dollars, tgt_shares = allocate_from_scores(w, last_prices, nav=nav)

    # --- Step 3: Generate orders ---
    cur = current_pos if current_pos is not None else pd.Series(0.0, index=w.index)
    orders = generate_orders(cur, tgt_shares, last_prices, min_notional=500, adv_usd=pd.Series(adv_map))

    if orders.empty:
        return {"weights": w, "orders": orders, "reports": [], "pnl": {}}

    # --- Step 4: Slippage costs ---
    orders_slipped = apply_to_orders(
        orders,
        model=slippage_model,
        params={"fee_bps": 0.3, "k_impact": 0.1, "floor_bps": 0.5},
    )

    # --- Step 5: Route via broker ---
    router = default_router()
    parent_orders = [
        Order(
            ticker=t,
            side=("BUY" if row.order_shares > 0 else "SELL"),
            qty=abs(float(row.order_shares)),
            price=float(row.price),
            order_type="MKT",
        )
        for t, row in orders.iterrows()
    ]
    reports: list[ExecutionReport] = router.route_batch(parent_orders, last_prices.to_dict(), adv_map)

    # --- Step 6: Backtest PnL (toy, single-period) ---
    pnl = run_equity_ls_pnl(
        prices=prices,
        positions=pd.DataFrame({c: [tgt_shares.get(c, 0.0)] for c in w.index}, index=[prices.index[-1]]),
        trades=pd.DataFrame(0.0, index=[prices.index[-1]], columns=w.index),
        borrow_bps=50.0,
        div_yield_bps=150.0,
        fee_bps=0.5,
        slippage_bps=2.0,
        cash_equity=nav,
    )

    return {
        "weights": w,
        "orders": orders_slipped,
        "reports": reports,
        "pnl": pnl,
    }


if __name__ == "__main__":
    # --- Demo with toy prices ---
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(0.0002 + 0.015 * rng.standard_normal((len(idx), len(tickers))), axis=0)), # type: ignore
        index=idx,
        columns=tickers,
    )

    out = run_equity_ls_strategy(
        signal_type="momentum",
        prices=prices,
        nav=1_000_000,
        adv_map={t: 1e7 for t in tickers},
    )

    print("Final weights:\n", out["weights"].head())
    print("Orders:\n", out["orders"].head())
    print("Exec reports:", [r.status for r in out["reports"]])
    print("PnL summary:\n", out["pnl"]["summary"].tail())