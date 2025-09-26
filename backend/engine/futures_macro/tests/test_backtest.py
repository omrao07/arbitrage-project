# engines/futures_macro/tests/test_backtest.py
import numpy as np
import pandas as pd

from engines.futures_macro.backtest.pnl import ( # type: ignore
    ContractSpec, FeeSpec, SlippageSpec,
    compute_futures_pnl, make_trades_from_positions, target_positions_to_trades
)

# ------------------------- helpers -------------------------

def _randwalk(idx, mu=0.0002, sig=0.015, start=100.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(len(idx))
    x[0] = start
    for t in range(1, len(idx)):
        x[t] = x[t-1] * (1 + mu + sig * rng.standard_normal())
    return pd.Series(x, index=idx)

def _toy_prices():
    idx = pd.date_range("2024-01-02", periods=260, freq="B")
    prices = pd.DataFrame({
        "ES": _randwalk(idx, 0.0002, 0.012, 4800, 1),   # USD
        "NQ": _randwalk(idx, 0.00025, 0.015, 17000, 2), # USD
        "CL": _randwalk(idx, 0.0001, 0.02, 80, 3),      # USD
        "FESX": _randwalk(idx, 0.00018, 0.013, 4400, 4),# assume EUR-quoted (EURO STOXX-like)
    }, index=idx)
    return prices

def _toy_specs():
    return {
        "ES":   ContractSpec(symbol="ES",   multiplier=50.0,   tick_size=0.25,   currency="USD"),
        "NQ":   ContractSpec(symbol="NQ",   multiplier=20.0,   tick_size=0.25,   currency="USD"),
        "CL":   ContractSpec(symbol="CL",   multiplier=1000.0, tick_size=0.01,   currency="USD"),
        "FESX": ContractSpec(symbol="FESX", multiplier=10.0,   tick_size=1.0,    currency="EUR"),
    }

# ------------------------- core tests -------------------------

def test_basic_pnl_shapes_and_columns():
    prices = _toy_prices()
    specs = _toy_specs()

    # Build a simple weekly-rebalanced target: flip signs across symbols
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pos.loc[pos.index[::5], "ES"] = 2.0
    pos.loc[pos.index[::5], "NQ"] = -1.0
    pos.loc[pos.index[::5], "CL"] = 3.0
    pos.loc[pos.index[::5], "FESX"] = 5.0
    pos = pos.ffill().fillna(0.0)

    trades = make_trades_from_positions(pos)

    # FX: provide EURUSD (USD per 1 EUR). USD column not required (assumed 1.0)
    eurusd = pd.Series(1.05 + 0.02*np.sin(np.linspace(0, 6, len(prices))), index=prices.index)
    fx = pd.DataFrame({"EUR": eurusd}, index=prices.index)

    out = compute_futures_pnl(
        prices=prices,
        positions=pos,
        trades=trades,
        contract_specs=specs,
        fee_spec=FeeSpec(commission_per_contract=1.25, exchange_fee_per_contract=0.35),
        slippage=SlippageSpec(model="ticks", slippage_ticks=0.25),
        fx_rates=fx,
        collateral_apy=0.045,
        start_nav=1_000_000.0,
    )

    # Summary checks
    summary = out["summary"]
    required_cols = {"price_pnl$", "fees$", "slippage$", "carry_collateral$", "pnl$", "nav", "ret_net", "gross_notional$"}
    assert required_cols.issubset(summary.columns)
    assert summary.notna().all().all()
    # Some economic sanity
    assert (summary["fees$"] >= 0).all()
    assert (summary["slippage$"] >= 0).all()
    assert (summary["gross_notional$"] >= 0).all()
    # NAV is finite and evolves
    assert np.isfinite(summary["nav"]).all()
    assert (summary["nav"] != summary["nav"].iloc[0]).any()

    # Per-symbol table has multiindex columns
    per_sym = out["per_symbol"]
    assert isinstance(per_sym.columns, pd.MultiIndex)
    for k in ["price_pnl$", "traded_contracts", "traded_notional$"]:
        assert (k in [c[1] for c in per_sym.columns])


def test_slippage_bps_path_and_costs_nonnegative():
    prices = _toy_prices()
    specs = _toy_specs()

    # One-shot rebalance mid-sample to generate trades once
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    mid = prices.index[len(prices)//2]
    pos.loc[mid:, "ES"] = 10.0
    pos.loc[mid:, "CL"] = -4.0
    pos = pos.ffill().fillna(0.0)
    trades = make_trades_from_positions(pos)

    out = compute_futures_pnl(
        prices=prices,
        positions=pos,
        trades=trades,
        contract_specs=specs,
        fee_spec=FeeSpec(commission_per_contract=1.0, exchange_fee_per_contract=0.5),
        slippage=SlippageSpec(model="bps", slippage_bps=1.0),
        fx_rates=None,
        collateral_apy=0.0,
        start_nav=2_000_000.0,
    )
    summary = out["summary"]
    assert (summary["fees$"] >= 0).all()
    assert (summary["slippage$"] >= 0).all()
    # On non-trade days, costs should be zero
    costs = (summary["fees$"] + summary["slippage$"])
    trade_days = trades.abs().sum(axis=1) > 0
    assert (costs[~trade_days] == 0).all()


def test_fx_conversion_effect():
    """
    If a non-USD symbol (EUR-quoted) rallies, its USD P&L should be impacted by EURUSD.
    We bump both the EUR-quoted future and EURUSD near the end and ensure price P&L reflects it.
    """
    prices = _toy_prices()
    specs = _toy_specs()

    # Long 5 contracts of FESX throughout
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pos["FESX"] = 5.0
    pos = pos.ffill()
    trades = make_trades_from_positions(pos)

    eurusd = pd.Series(1.05, index=prices.index)
    fx = pd.DataFrame({"EUR": eurusd}, index=prices.index)

    # Baseline run
    base = compute_futures_pnl(
        prices=prices,
        positions=pos,
        trades=trades,
        contract_specs=specs,
        fx_rates=fx,
        start_nav=1_000_000.0,
    )["summary"]["pnl$"].sum()

    # Shock both FESX price and EURUSD up at the last bar → more USD P&L
    prices2 = prices.copy()
    prices2.iloc[-1, prices.columns.get_loc("FESX")] *= 1.02 # type: ignore
    fx2 = fx.copy()
    fx2.iloc[-1, fx.columns.get_loc("EUR")] *= 1.02 # type: ignore

    shocked = compute_futures_pnl(
        prices=prices2,
        positions=pos,
        trades=trades,
        contract_specs=specs,
        fx_rates=fx2,
        start_nav=1_000_000.0,
    )["summary"]["pnl$"].sum()

    assert shocked > base


def test_target_positions_to_trades_exact_diff():
    current = pd.Series({"ES": 3.0, "NQ": -2.0, "CL": 0.0})
    target  = pd.Series({"ES": 1.0, "NQ": 0.0, "CL": -4.0})
    trades = target_positions_to_trades(current_positions=current, target_positions=target)
    # ES: sell 2, NQ: buy 2, CL: sell 4
    assert trades["ES"] == (1.0 - 3.0)
    assert trades["NQ"] == (0.0 - (-2.0))
    assert trades["CL"] == (-4.0 - 0.0)