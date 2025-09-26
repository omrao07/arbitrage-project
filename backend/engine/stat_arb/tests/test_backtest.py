# engines/stat_arb/tests/test_backtest.py
import numpy as np
import pandas as pd

from engines.stat_arb.backtest.pnl import ( # type: ignore
    compute_pair_pnl, compute_portfolio_pnl, PairInputs, PairSpec
)
from engines.stat_arb.backtest.simulator import simulate_pairs, PairConfig, SimConfig # type: ignore
from engines.stat_arb.execution.allocator import ( # type: ignore
    PairTarget, AllocConfig, generate_orders_from_pairs, allocate_from_units
)


# ----------------------- Helpers -----------------------

def _toy_prices(n_days=260, tickers=("AAA","BBB","CCC","DDD"), seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    X = np.zeros((len(idx), len(tickers)))
    X[0] = 100.0
    # correlated random walks in two pairs: AAA~BBB, CCC~DDD
    cov = np.array([[0.0002, 0.00018, 0.0, 0.0],
                    [0.00018, 0.00022, 0.0, 0.0],
                    [0.0, 0.0, 0.00024, 0.00019],
                    [0.0, 0.0, 0.00019, 0.00021]])
    L = np.linalg.cholesky(cov + 1e-9*np.eye(4))
    drift = np.array([0.0002, 0.00019, 0.00021, 0.00018])
    for t in range(1, len(idx)):
        z = rng.standard_normal(4)
        ret = drift + (L @ z)
        X[t] = X[t-1] * (1.0 + ret)
    return pd.DataFrame(X, index=idx, columns=list(tickers))


# ----------------------- Pair P&L (single) -----------------------

def test_compute_pair_pnl_basic_paths():
    px = _toy_prices()
    y = px["AAA"]; x = px["BBB"]

    # Simple rule: +1 unit when last spread z < -0.5 else 0 (rough)
    spread = y - (y.rolling(60).cov(x) / x.rolling(60).var().replace(0, np.nan)).ffill().bfill().iloc[-1] * x
    z = (spread - spread.rolling(60).mean()) / (spread.rolling(60).std() + 1e-12)
    units = (z < -0.5).astype(float).replace(0.0, np.nan).ffill().fillna(0.0).to_frame("u")

    res = compute_pair_pnl(PairInputs(
        price_y=y, price_x=x,
        spread_units=units,
        fee_bps=0.3, slippage_bps=1.5,
        borrow_bps_y=50.0, borrow_bps_x=50.0,
        div_yield_bps_y=150.0, div_yield_bps_x=150.0,
    ))
    summary = res["summary"]

    assert not summary.empty
    for col in ["pnl$", "price_pnl$", "carry_pnl$", "costs$", "ret_net", "gross_exposure$"]:
        assert col in summary.columns
        assert summary[col].notna().all()


# ----------------------- Portfolio aggregation -----------------------

def test_compute_portfolio_pnl_aggregates():
    px = _toy_prices()
    pairs = {
        "A_B": PairSpec(price_y=px["AAA"], price_x=px["BBB"], spread_units=pd.DataFrame(0.5, index=px.index, columns=["u"])),
        "C_D": PairSpec(price_y=px["CCC"], price_x=px["DDD"], spread_units=pd.DataFrame(-0.3, index=px.index, columns=["u"])),
    }
    out = compute_portfolio_pnl(pairs)

    port = out["summary"]
    assert {"pnl$", "price_pnl$", "carry_pnl$", "costs$", "turnover", "ret_net"}.issubset(port.columns)
    assert port.notna().all().all()
    # sanity: gross exposure positive sometimes
    assert (port["gross_exposure$"] > 0).any()


# ----------------------- Simulator end-to-end -----------------------

def test_simulate_pairs_end_to_end():
    px = _toy_prices()
    configs = [
        PairConfig(y="AAA", x="BBB", entry_z=1.0, exit_z=0.25, max_units=1.0),
        PairConfig(y="CCC", x="DDD", entry_z=1.2, exit_z=0.25, max_units=1.5),
    ]
    out = simulate_pairs(px, configs, SimConfig(rebalance="W-FRI"))
    assert "portfolio" in out and "by_pair" in out
    port = out["portfolio"]
    assert not port.empty and port["ret_net"].notna().all()

    # Ensure per-pair artifacts are present
    for name, art in out["by_pair"].items():
        for k in ["units", "beta", "spread", "z", "shares_y", "shares_x", "trades_y", "trades_x", "summary"]:
            assert k in art
            assert not art[k].empty


# ----------------------- Allocator & Router wiring shape checks -----------------------

def test_allocator_from_units_and_generate_orders():
    px = _toy_prices()
    last = px.iloc[-1]

    units = {"AAA/BBB": 0.8, "CCC/DDD": -0.6}
    betas = {"AAA/BBB": 1.1, "CCC/DDD": 0.9}
    legs  = {"AAA/BBB": ("AAA", "BBB"), "CCC/DDD": ("CCC", "DDD")}

    current = pd.Series(0.0, index=last.index)
    adv = pd.Series(1e8, index=last.index)

    orders = allocate_from_units(
        units_by_pair=units,
        betas_by_pair=betas,
        legs_by_pair=legs,
        current_pos_shares=current,
        last_prices=last,
        adv_usd=adv,
        cfg=AllocConfig(lot_size=1, min_notional=500.0, max_participation=0.10),
    )
    assert not orders.empty
    assert {"order_shares", "order_notional", "side", "price"}.issubset(orders.columns)
    assert (orders["order_notional"] >= 500.0).all()

    # Direct generator path equivalence
    pair_targets = [
        PairTarget(name="AAA/BBB", y_ticker="AAA", x_ticker="BBB", units=0.8, beta=1.1),
        PairTarget(name="CCC/DDD", y_ticker="CCC", x_ticker="DDD", units=-0.6, beta=0.9),
    ]
    orders2 = generate_orders_from_pairs(
        pair_targets=pair_targets,
        current_pos_shares=current,
        last_prices=last,
        adv_usd=adv,
        cfg=AllocConfig(lot_size=1, min_notional=500.0, max_participation=0.10),
    )
    assert not orders2.empty
    # tickers and sides should match
    assert set(orders.index) == set(orders2.index)