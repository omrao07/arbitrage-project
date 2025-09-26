# engines/equity_ls/tests/test_backtest.py
import numpy as np
import pandas as pd

from engines.equity_ls.backtest.pnl import compute_pnl, run_equity_ls_pnl # type: ignore
from engines.equity_ls.backtest.risk import ( # type: ignore
    size_equity_ls_weights,
    risk_report_from_positions,
)
from engines.equity_ls.backtest.simulator import simulate_from_scores # type: ignore


def _toy_prices(n_days=260, tickers=("AAA", "BBB", "CCC", "DDD"), seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    drift = rng.normal(0.0002, 0.00005, size=len(tickers))
    vol = rng.normal(0.015, 0.003, size=len(tickers))
    X = np.zeros((len(idx), len(tickers)))
    X[0] = 100.0
    for t in range(1, len(idx)):
        ret = drift + vol * rng.standard_normal(len(tickers))
        X[t] = X[t - 1] * (1.0 + ret)
    return pd.DataFrame(X, index=idx, columns=list(tickers))


def test_compute_pnl_basic_runs():
    prices = _toy_prices()
    # simple long/short: long first two, short next two, constant shares
    shares = pd.DataFrame(
        10.0,
        index=prices.index,
        columns=prices.columns,
    )
    shares.iloc[:, 2:] *= -1.0
    trades = shares.diff().fillna(shares.iloc[0])

    summary, per_ticker = compute_pnl(
        prices=prices,
        positions=shares,
        trades=trades,
        borrow_bps=50.0,
        div_yield_bps=150.0,
        fee_bps=0.5,
        slippage_bps=2.0,
    )

    assert not summary.empty
    for col in ["pnl$", "price_pnl$", "carry_pnl$", "fees_slippage$", "ret_net"]:
        assert col in summary.columns
        assert summary[col].notna().all()

    assert per_ticker.shape == prices.shape


def test_risk_sizer_pipeline():
    prices = _toy_prices()
    # raw cross-sectional score = last 60d momentum z-score
    r = prices.pct_change()
    mom = (r.rolling(60).mean().iloc[-1] / (r.rolling(60).std().iloc[-1] + 1e-9)).dropna()

    # fake sector map
    tickers = list(prices.columns)
    sector_map = {t: ("Tech" if i % 2 == 0 else "Health") for i, t in enumerate(tickers)}

    # asset returns for vol targeting
    asset_returns = prices.pct_change().fillna(0.0)

    pack = size_equity_ls_weights(
        raw_scores=mom,
        prices=prices.iloc[-1],
        sector_map=sector_map,
        market_betas=None,
        asset_returns=asset_returns,
        market_returns=None,
        max_name=0.15,
        max_gross=1.0,
        max_net=0.05,
        cap_per_sector=0.70,
        target_vol=0.10,
        vol_lookback=63,
    )

    w = pack["weights"]
    assert abs(w.abs().sum() - 1.0) < 1e-6
    assert w.index.isin(prices.columns).all()
    assert pack["diagnostics"]["gross"] <= 1.0 + 1e-9


def test_risk_report_shapes():
    prices = _toy_prices()
    shares = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    shares.iloc[:, 0] = 10.0  # long AAA
    shares.iloc[:, 1] = -5.0  # short BBB

    sector_map = {c: ("Tech" if c in ("AAA", "CCC") else "Health") for c in prices.columns}

    report = risk_report_from_positions(
        positions=shares,
        prices=prices,
        factor_exposures=None,
        sector_map=sector_map,
    )
    assert {"gross$", "net$", "concentration_HHI"}.issubset(set(report.columns))
    assert (report.select_dtypes(include=[float]) == report.select_dtypes(include=[float])).all().all()  # no NaN


def test_simulator_from_scores_end_to_end():
    prices = _toy_prices()

    def scores_func(px: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
        r = px.pct_change()
        s = r.rolling(40).mean().iloc[-1] / (r.rolling(40).std().iloc[-1] + 1e-9)
        return s

    out = simulate_from_scores(
        prices=prices,
        scores_func=scores_func,
        start_nav=1_000_000.0,
        gross_target=1.0,
        per_name_cap=0.25,
        rebalance="W-FRI",
        borrow_bps=50.0,
        div_yield_bps=150.0,
        fee_bps=0.5,
        slippage_bps=2.0,
        slippage_mode="flat",
    )

    # sanity checks
    assert set(["positions", "trades", "summary", "per_ticker_pnl", "nav"]).issubset(out.keys())
    assert out["positions"].shape == prices.shape
    assert out["trades"].shape == prices.shape
    assert out["summary"]["nav"].iloc[-1] > 0
    assert out["summary"]["ret_net"].notna().all()


def test_run_equity_ls_pnl_wrapper():
    prices = _toy_prices()
    # build a naive alternating long/short book
    pos = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    pos.iloc[:, ::2] = 10.0
    pos.iloc[:, 1::2] = -10.0
    trades = pos.diff().fillna(pos.iloc[0])

    out = run_equity_ls_pnl(
        prices=prices,
        positions=pos,
        trades=trades,
        borrow_bps=50.0,
        div_yield_bps=150.0,
        fee_bps=0.5,
        slippage_bps=2.0,
        cash_equity=1_000_000.0,
    )
    assert "summary" in out and "per_ticker_pnl" in out
    assert out["summary"]["ret_net"].notna().all()