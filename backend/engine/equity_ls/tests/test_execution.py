# engines/equity_ls/tests/test_execution.py
import pandas as pd
import numpy as np

import pytest # type: ignore

from engines.equity_ls.execution.allocator import ( # type: ignore
    allocate_from_scores,
    generate_orders,
)
from engines.equity_ls.execution.order_router import ( # type: ignore
    Order, default_router, PaperBroker
)
from engines.equity_ls.execution.slippage import apply_to_orders # type: ignore


# ------------------------ Helpers ------------------------

def _toy_scores_prices():
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    scores = pd.Series([1.2, -0.5, 0.7, -1.1], index=tickers)
    prices = pd.Series([100.0, 50.0, 200.0, 10.0], index=tickers)
    return scores, prices


# ------------------------ Allocator ------------------------

def test_allocate_from_scores_shapes_and_normalization():
    scores, prices = _toy_scores_prices()
    w, dollars, shares = allocate_from_scores(scores, prices, nav=1_000_000)

    # sanity checks
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert all(idx in scores.index for idx in w.index)
    assert shares.index.equals(scores.index)
    assert (dollars.abs().sum() > 0)


def test_generate_orders_with_adv_and_min_notional():
    scores, prices = _toy_scores_prices()
    _, _, tgt_shares = allocate_from_scores(scores, prices, nav=500_000)

    current = pd.Series([0, 0, 0, 0], index=scores.index)
    adv = pd.Series([1e7, 1e7, 1e7, 1e7], index=scores.index)

    orders = generate_orders(
        current_pos_shares=current,
        target_pos_shares=tgt_shares,
        last_prices=prices,
        min_notional=1000,
        adv_usd=adv,
        max_participation=0.1,
    )

    assert not orders.empty
    assert {"order_shares", "order_notional", "side"}.issubset(orders.columns)
    # orders should skip dust trades
    assert (orders["order_notional"] >= 1000).all()


# ------------------------ Order Router ------------------------

def test_order_router_paper_end_to_end():
    scores, prices = _toy_scores_prices()
    _, _, tgt_shares = allocate_from_scores(scores, prices, nav=200_000)
    current = pd.Series([0, 0, 0, 0], index=scores.index)

    orders = generate_orders(current, tgt_shares, prices, min_notional=500)

    router = default_router()
    last_prices = prices.to_dict()
    adv_map = {t: 1e7 for t in prices.index}

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

    reports = router.route_batch(parent_orders, last_prices, adv_map)
    assert all(r.status in ("FILLED", "REJECTED") for r in reports)
    assert all(r.ticker in prices.index for r in reports)


# ------------------------ Slippage ------------------------

def test_apply_slippage_models_return_columns():
    scores, prices = _toy_scores_prices()
    _, _, tgt_shares = allocate_from_scores(scores, prices, nav=100_000)
    current = pd.Series([0, 0, 0, 0], index=scores.index)

    orders = generate_orders(current, tgt_shares, prices, min_notional=100)
    orders["adv_usd"] = 1e7
    orders["spread_bps"] = 2.0
    orders["daily_vol"] = 0.02

    for model in ["flat_bps", "sqrt_impact", "amihud", "spread_vol"]:
        out = apply_to_orders(orders, model=model)
        assert "cost_usd" in out.columns
        assert "slippage_bps" in out.columns
        assert (out["cost_usd"] >= 0).all()