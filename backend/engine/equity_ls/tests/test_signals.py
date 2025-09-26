# engines/equity_ls/tests/test_signals.py
import numpy as np
import pandas as pd

from engines.equity_ls.signals.momentum import build_signal as build_mom # type: ignore
from engines.equity_ls.signals.value import build_signal as build_val, proxy_book_to_price # type: ignore
from engines.equity_ls.signals.quality import build_signal as build_qual, proxy_quality_from_prices # type: ignore
from engines.equity_ls.signals.sector_rotation import build_signal as build_sector, DEFAULT_SECTORS # type: ignore


# ---------------------- Helpers ----------------------

def _toy_prices(n_days=260, tickers=None, seed=7):
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
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


def _sector_map(tickers):
    # simple alternating sector map
    sectors = ["Tech", "Health", "Financials"]
    return {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}


def _toy_fundamentals(prices: pd.DataFrame):
    """Create simple, consistent fundamentals DataFrames keyed by name (date x ticker)."""
    idx = prices.index
    cols = prices.columns
    # make level-ish fundamentals with mild cross-section dispersion
    base = pd.DataFrame(1.0, index=idx, columns=cols)
    # ROE ~ 10-20%
    roe = base * 0.10 + (np.arange(len(cols)) / (100.0 * len(cols)))
    # GrossMargin ~ 30-60%
    gm = base * 0.30 + (np.arange(len(cols)) / (50.0 * len(cols)))
    # Stability (higher better): inverse of realized vol proxy
    stab = 1.0 / (prices.pct_change().rolling(40).std().replace(0, np.nan)).fillna(0.1)
    # Accruals (lower better)
    accr = base * 0.05 + 0.002 * np.random.default_rng(1).standard_normal((len(idx), len(cols)))
    # LeverageChange (lower better)
    lev = base * 0.01 + 0.001 * np.random.default_rng(2).standard_normal((len(idx), len(cols)))
    # NetIssuance (lower better)
    iss = base * 0.00 + 0.002 * np.random.default_rng(3).standard_normal((len(idx), len(cols)))
    # BuybackYield (higher better)
    bby = base * 0.01 + 0.001 * np.random.default_rng(4).standard_normal((len(idx), len(cols)))

    # Value building blocks
    book = base * 20.0 * (1 + np.arange(len(cols)) / (5.0 * len(cols)))
    earn = base * 3.0 * (1 + np.arange(len(cols)) / (8.0 * len(cols)))
    fcf = base * 2.0
    ebitda = base * 4.0
    sales = base * 30.0
    ev = base * 200.0 * (1 + np.arange(len(cols)) / (10.0 * len(cols)))

    funds = {
        "ROE": roe,
        "GrossMargin": gm,
        "Stability": stab,
        "Accruals": accr,
        "LeverageChange": lev,
        "NetIssuance": iss,
        "BuybackYield": bby,
        "BookEquity": book,
        "Earnings": earn,
        "FreeCashFlow": fcf,
        "EBITDA": ebitda,
        "Sales": sales,
        "EnterpriseValue": ev,
        "Price": prices,
    }
    return funds


# ---------------------- Tests: Momentum ----------------------

def test_momentum_cross_sectional_unit_gross_and_caps():
    prices = _toy_prices()
    sector_map = _sector_map(prices.columns)
    w = build_mom(
        prices,
        method="cross",
        lookbacks=(63, 126, 252),
        vol_scale_lookback=20,
        sector_map=sector_map,
        long_cap=0.06,
        short_cap=-0.06,
        unit_gross=1.0,
        delay=1,
    )
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.abs() <= 0.0600001).all()
    assert w.index.isin(prices.columns).all()


def test_momentum_timeseries_signs_and_unit_gross():
    prices = _toy_prices()
    w = build_mom(
        prices,
        method="timeseries",
        lookback_ts=126,
        vol_target_ts=0.20,
        cap_per_name_ts=0.05,
        unit_gross=1.0,
        delay=1,
    )
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.abs() <= 0.0500001).all()


# ---------------------- Tests: Value ----------------------

def test_value_composite_from_fundamentals_sector_neutral():
    prices = _toy_prices()
    funds = _toy_fundamentals(prices)
    sector_map = _sector_map(prices.columns)
    w = build_val(
        fundamentals=funds,
        sector_map=sector_map,
        long_cap=0.05,
        short_cap=-0.05,
        unit_gross=1.0,
        delay=1,
    )
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.abs() <= 0.0500001).all()
    assert set(w.index) == set(prices.columns)


def test_value_proxy_book_to_price():
    prices = _toy_prices()
    # create synthetic book value per share, proportional to price with noise
    bps = prices / 5.0 * (1.0 + 0.05 * np.random.default_rng(0).standard_normal(prices.shape))
    w = proxy_book_to_price(prices, book_value_ps=bps, unit_gross=1.0)
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.index.isin(prices.columns)).all()


# ---------------------- Tests: Quality ----------------------

def test_quality_composite_from_fundamentals():
    prices = _toy_prices()
    funds = _toy_fundamentals(prices)
    sector_map = _sector_map(prices.columns)
    w = build_qual(
        fundamentals=funds,
        sector_map=sector_map,
        long_cap=0.05,
        short_cap=-0.05,
        unit_gross=1.0,
        delay=1,
    )
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.abs() <= 0.0500001).all()
    assert set(w.index) == set(prices.columns)


def test_quality_proxy_from_prices_only():
    prices = _toy_prices()
    shares_out = pd.DataFrame(1e9, index=prices.index, columns=prices.columns)  # flat SO to exercise code path
    w = proxy_quality_from_prices(
        prices=prices,
        shares_out=shares_out,
        unit_gross=1.0,
        delay=1,
    )
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.index.isin(prices.columns)).all()


# ---------------------- Tests: Sector Rotation ----------------------

def test_sector_rotation_signal_shapes():
    # build synthetic sector ETF prices for DEFAULT_SECTORS tickers
    tickers = list(DEFAULT_SECTORS.keys())
    prices = _toy_prices(tickers=tickers)
    w = build_sector(
        prices=prices,
        sectors_map=DEFAULT_SECTORS,
        lookbacks=(20, 60, 120),
        vol_lb=20,
        delay=1,
        long_cap=0.30,
        short_cap=-0.30,
        target_gross=1.0,
        neutralize_sum=True,
    )
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.index.isin(tickers)).all()
    assert (w.abs() <= 0.3000001).all()


def test_sector_rotation_with_tilts_and_regime_bias():
    tickers = list(DEFAULT_SECTORS.keys())
    prices = _toy_prices(tickers=tickers)
    w = build_sector(
        prices=prices,
        sectors_map=DEFAULT_SECTORS,
        macro_tilts={"Energy": 0.2, "Defensive": 0.1},
        regime_bias="risk_on",
        target_gross=1.0,
        delay=1,
    )
    assert abs(w.abs().sum() - 1.0) < 1e-8
    assert (w.index.isin(tickers)).all()