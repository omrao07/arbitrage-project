#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nikkei_vs_topix.py

Compare Nikkei 225 vs TOPIX:
- Fetches prices (Yahoo Finance) for Nikkei (^N225) and TOPIX (^TOPX by default).
- Computes normalized index (base=100), daily/weekly/monthly returns.
- Performance stats (CAGR, Vol, Sharpe, Max DD, Calmar).
- Rolling correlation/beta (OLS on log-returns).
- Optional simple pairs-trade signals using z-scored spread via rolling OLS hedge ratio.
- Saves artifacts (prices, returns, metrics, signals) to ./artifacts/nikkei_topix_YYYYMMDD_HHMMSS/.
- Plots to PNG (if --plot).

Notes:
- If ^TOPX is unavailable in your region, use a TOPIX ETF like 1306.T or 1348.T via --topix-ticker.
- Similarly, you can swap Nikkei to 1321.T (Nikkei 225 ETF) via --nikkei-ticker.

Usage:
    python nikkei_vs_topix.py --start 2010-01-01 --end 2025-09-05 --interval 1d --plot --signals
"""

import argparse
import os
import sys
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    print("yfinance is required. Install with: pip install yfinance pandas numpy matplotlib statsmodels")
    raise

# Matplotlib is optional (only if --plot)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Statsmodels for OLS (optional; fallback to numpy if missing)
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# ----------------------------- Utilities -----------------------------

def annualization_factor(interval: str) -> int:
    interval = interval.lower()
    if interval in ("1d", "1wk", "1mo"):
        return {"1d": 252, "1wk": 52, "1mo": 12}[interval]
    # Fallback
    return 252


def to_returns(prices: pd.Series) -> pd.Series:
    return prices.ffill().pct_change().dropna()


def cagr(series: pd.Series, periods_per_year: int) -> float:
    if series.empty:
        return np.nan
    start, end = series.iloc[0], series.iloc[-1]
    n_periods = len(series) - 1
    years = n_periods / periods_per_year if periods_per_year > 0 else np.nan
    if years is None or years <= 0 or start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1


def max_drawdown(nav: pd.Series) -> Tuple[float, float, float]:
    """Returns (max_dd, peak_date, trough_date) where max_dd is negative."""
    if nav.empty:
        return np.nan, np.nan, np.nan
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    trough_idx = dd.idxmin()
    max_dd = dd.loc[trough_idx]
    peak_idx = nav.loc[:trough_idx].idxmax()
    return float(max_dd), peak_idx, trough_idx


def sharpe(returns: pd.Series, rf: float, periods_per_year: int) -> float:
    if returns.empty:
        return np.nan
    # Convert annual rf to per-period
    rf_per_period = (1 + rf) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    mu = excess.mean()
    sigma = excess.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return (mu / sigma) * math.sqrt(periods_per_year)


def calmar(cagr_val: float, max_dd_val: float) -> float:
    if max_dd_val == 0 or np.isnan(max_dd_val):
        return np.nan
    return cagr_val / abs(max_dd_val)


def rolling_beta_x_on_y(x_ret: pd.Series, y_ret: pd.Series, window: int = 63) -> pd.Series:
    """Beta of x ~ a + b*y computed rolling with OLS. Returns b."""
    if len(x_ret) != len(y_ret):
        x_ret, y_ret = x_ret.align(y_ret, join="inner")
    if len(x_ret) < window:
        return pd.Series(index=x_ret.index, dtype=float)

    betas = pd.Series(index=x_ret.index, dtype=float)
    if HAVE_SM:
        for i in range(window, len(x_ret) + 1):
            y = x_ret.iloc[i - window:i]
            X = sm.add_constant(y_ret.iloc[i - window:i])
            res = sm.OLS(y.values, X.values).fit()
            betas.iloc[i - 1] = res.params[-1]
    else:
        for i in range(window, len(x_ret) + 1):
            y = x_ret.iloc[i - window:i].values
            x = y_ret.iloc[i - window:i].values
            cov = np.cov(x, y, ddof=0)
            var = cov[0, 0]
            betas.iloc[i - 1] = cov[0, 1] / var if var != 0 else np.nan
    return betas


def zscore(series: pd.Series, window: int = 63) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd


# ----------------------------- Core Logic -----------------------------

@dataclass
class Config:
    nikkei_ticker: str = "^N225"
    topix_ticker: str = "^TOPX"
    start: str = "2010-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    rf_annual: float = 0.01
    plot: bool = False
    signals: bool = False
    outdir: str = "./artifacts"


def fetch_prices(cfg: Config) -> pd.DataFrame:
    tickers = [cfg.nikkei_ticker, cfg.topix_ticker]
    data = yf.download(
        tickers=tickers,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Unify to a simple DataFrame with Close columns
    closes = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                closes[t] = data[t]["Close"]
            else:
                # Single ticker case, but we passed two – handle robustly
                if t in data.columns:
                    closes[t] = data[t]
            # Some regions return 'Adj Close'
            if t not in closes and (t, "Adj Close") in data.columns:
                closes[t] = data[(t, "Adj Close")]
        except Exception:
            pass

    if not closes:
        raise RuntimeError("Failed to fetch any price series. Check tickers or connection.")
    df = pd.DataFrame(closes).dropna(how="all").ffill().dropna()
    if df.empty or df.shape[1] < 2:
        raise RuntimeError("Could not retrieve both series. Try alternative TOPIX/Nikkei ETFs (e.g., 1306.T, 1321.T).")
    return df


def compute_metrics(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pf = df.copy()
    base = pf.iloc[0]
    norm = pf / base * 100.0  # index=100

    # Returns
    ret = pf.pct_change().dropna()
    ann = annualization_factor(cfg.interval)

    # Stats per asset
    stats_rows = []
    for col in pf.columns:
        nav = (1 + ret[col]).cumprod()
        met = {
            "ticker": col,
            "start": pf.index.min().date().isoformat(),
            "end": pf.index.max().date().isoformat(),
            "periods": len(ret[col]),
            "CAGR": cagr(nav, ann),
            "Vol_Ann": ret[col].std(ddof=0) * math.sqrt(ann),
            "Sharpe": sharpe(ret[col], cfg.rf_annual, ann),
        }
        mdd, peak, trough = max_drawdown(nav)
        met["MaxDD"] = mdd
        met["DD_Peak"] = str(peak) if isinstance(peak, pd.Timestamp) else ""
        met["DD_Trough"] = str(trough) if isinstance(trough, pd.Timestamp) else ""
        met["Calmar"] = calmar(met["CAGR"], met["MaxDD"])
        stats_rows.append(met)

    stats = pd.DataFrame(stats_rows).set_index("ticker")

    # Rolling correlation/beta (Nikkei on TOPIX)
    nik, top = pf.columns[0], pf.columns[1]
    r_nik = ret[nik]
    r_top = ret[top]
    roll_win = 126  # ~6 months of trading days

    roll_corr = r_nik.rolling(roll_win).corr(r_top).rename("rolling_corr")
    roll_beta = rolling_beta_x_on_y(r_nik, r_top, window=roll_win).rename("rolling_beta")
    roll = pd.concat([roll_corr, roll_beta], axis=1)

    return norm, ret, stats, roll


def pairs_signals(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simple mean-reversion spread:
    - Rolling OLS to estimate hedge ratio: Nikkei = a + b * TOPIX
    - Spread = Nikkei - b*TOPIX, z-scored
    - Enter when |z| > 2: if z>2 short spread (short Nikkei/long TOPIX), if z<-2 long spread.
    - Exit when |z| < 0.5
    """
    nik, top = df.columns[0], df.columns[1]
    px = df[[nik, top]].dropna()
    logp = np.log(px)

    win = 126
    betas = []
    intercepts = []
    idx = px.index

    if HAVE_SM:
        for i in range(len(px)):
            if i < win:
                betas.append(np.nan)
                intercepts.append(np.nan)
                continue
            y = logp[nik].iloc[i - win:i].values
            X = sm.add_constant(logp[top].iloc[i - win:i].values)
            res = sm.OLS(y, X).fit()
            intercepts.append(res.params[0])
            betas.append(res.params[1])
    else:
        for i in range(len(px)):
            if i < win:
                betas.append(np.nan)
                intercepts.append(np.nan)
                continue
            y = logp[nik].iloc[i - win:i].values
            x = logp[top].iloc[i - win:i].values
            # OLS via closed-form
            x1 = np.vstack([np.ones_like(x), x]).T
            try:
                params = np.linalg.lstsq(x1, y, rcond=None)[0]
                intercepts.append(params[0])
                betas.append(params[1])
            except Exception:
                intercepts.append(np.nan)
                betas.append(np.nan)

    hr = pd.Series(betas, index=idx, name="hedge_beta")
    inter = pd.Series(intercepts, index=idx, name="hedge_intercept")

    spread = logp[nik] - (inter + hr * logp[top])
    z = zscore(spread, window=win).rename("zscore")

    # Signal logic
    upper, lower, exit_band = 2.0, -2.0, 0.5
    state = 0  # -1 short spread, +1 long spread
    sig = []
    for val in z.values:
        if np.isnan(val):
            sig.append(0)
            continue
        if state == 0:
            if val > upper:
                state = -1
            elif val < lower:
                state = +1
        else:
            if abs(val) < exit_band:
                state = 0
        sig.append(state)

    signal = pd.Series(sig, index=z.index, name="signal")  # -1 short spread, +1 long spread, 0 flat

    # P&L approximation:
    # Spread return ≈ d(log NIK) - beta * d(log TOP)
    lr = logp.diff().dropna()
    hedgeb = hr.shift(1).reindex(lr.index)  # use prior day's hedge ratio
    spread_ret = (lr[nik] - hedgeb * lr[top]).rename("spread_ret")
    strat_ret = (signal.shift(1).reindex(lr.index).fillna(0) * spread_ret).rename("strategy_ret")

    nav = (1 + strat_ret).cumprod().rename("strategy_nav")
    ann = annualization_factor("1d")
    stats = {
        "trades": int((np.abs(pd.Series(sig).diff()) > 0).sum() // 2),
        "CAGR": cagr(nav, ann),
        "Vol_Ann": strat_ret.std(ddof=0) * math.sqrt(ann),
        "Sharpe(0%)": sharpe(strat_ret, 0.0, ann),
        "MaxDD": max_drawdown(nav)[0],
    }

    out = pd.concat([hr, inter, spread.rename("spread"), z, signal, spread_ret, strat_ret, nav], axis=1)
    return out, stats


def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"nikkei_topix_{ts}")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def plot_all(norm: pd.DataFrame, roll: pd.DataFrame, signal_df: Optional[pd.DataFrame], outdir: str):
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return

    # 1) Normalized Index
    fig1 = plt.figure(figsize=(10, 5))
    norm.plot(ax=plt.gca(), linewidth=1.2)
    plt.title("Nikkei vs TOPIX — Normalized (Index=100 at start)")
    plt.ylabel("Index (start=100)")
    plt.xlabel("Date")
    plt.tight_layout()
    fig1.savefig(os.path.join(outdir, "normalized_index.png"), dpi=150)
    plt.close(fig1)

    # 2) Rolling Correlation & Beta
    fig2 = plt.figure(figsize=(10, 5))
    roll["rolling_corr"].plot(ax=plt.gca(), linewidth=1.0, label="Rolling Corr (126d)")
    plt.title("Rolling Correlation (Nikkei vs TOPIX)")
    plt.ylabel("Correlation")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "rolling_corr.png"), dpi=150)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(10, 5))
    roll["rolling_beta"].plot(ax=plt.gca(), linewidth=1.0, label="Rolling Beta (Nikkei on TOPIX)")
    plt.title("Rolling Beta (Nikkei on TOPIX)")
    plt.ylabel("Beta")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    fig3.savefig(os.path.join(outdir, "rolling_beta.png"), dpi=150)
    plt.close(fig3)

    # 3) Signals (if any)
    if signal_df is not None and not signal_df.empty:
        fig4 = plt.figure(figsize=(10, 5))
        signal_df["zscore"].plot(ax=plt.gca(), linewidth=1.0, label="Spread Z-score")
        plt.axhline(2.0, linestyle="--")
        plt.axhline(-2.0, linestyle="--")
        plt.axhline(0.5, linestyle=":")
        plt.axhline(-0.5, linestyle=":")
        plt.title("Spread Z-score with Bands")
        plt.ylabel("Z")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        fig4.savefig(os.path.join(outdir, "spread_zscore.png"), dpi=150)
        plt.close(fig4)

        fig5 = plt.figure(figsize=(10, 5))
        signal_df["strategy_nav"].dropna().plot(ax=plt.gca(), linewidth=1.2, label="Pairs Strategy NAV")
        plt.title("Pairs Strategy NAV (toy)")
        plt.ylabel("NAV")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        fig5.savefig(os.path.join(outdir, "strategy_nav.png"), dpi=150)
        plt.close(fig5)


def main():
    parser = argparse.ArgumentParser(description="Nikkei vs TOPIX comparison & (optional) pairs signals.")
    parser.add_argument("--nikkei-ticker", type=str, default="^N225", help="Nikkei 225 ticker (default ^N225)")
    parser.add_argument("--topix-ticker", type=str, default="^TOPX", help="TOPIX ticker or ETF (default ^TOPX, try 1306.T or 1348.T if needed)")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--interval", type=str, default="1d", choices=["1d", "1wk", "1mo"])
    parser.add_argument("--rf", type=float, default=0.01, help="Annual risk-free rate for Sharpe (default 0.01 = 1%)")
    parser.add_argument("--plot", action="store_true", help="Save plots")
    parser.add_argument("--signals", action="store_true", help="Compute simple pairs-trade signals and PnL")
    parser.add_argument("--outdir", type=str, default="./artifacts", help="Base output folder")
    args = parser.parse_args()

    cfg = Config(
        nikkei_ticker=args.nikkei_ticker,
        topix_ticker=args.topix_ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        rf_annual=args.rf,
        plot=args.plot,
        signals=args.signals,
        outdir=args.outdir,
    )

    outdir = ensure_outdir(cfg.outdir)
    print(f"[INFO] Writing artifacts to: {outdir}")

    df = fetch_prices(cfg)
    df.to_csv(os.path.join(outdir, "prices.csv"))
    print(f"[OK] Fetched {df.shape[0]} rows for {list(df.columns)}")

    norm, ret, stats, roll = compute_metrics(df, cfg)
    norm.to_csv(os.path.join(outdir, "normalized_index.csv"))
    ret.to_csv(os.path.join(outdir, "returns.csv"))
    stats.to_csv(os.path.join(outdir, "performance_stats.csv"))
    roll.to_csv(os.path.join(outdir, "rolling_corr_beta.csv"))

    print("\n=== Performance (annualized) ===")
    print(stats.round(4).to_string())

    signal_df = None
    if cfg.signals:
        signal_df, sig_stats = pairs_signals(df, cfg)
        signal_df.to_csv(os.path.join(outdir, "pairs_signals.csv"))
        print("\n=== Pairs Strategy (toy) ===")
        for k, v in sig_stats.items():
            print(f"{k:>10}: {v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k:>10}: {v}")

    if cfg.plot:
        plot_all(norm, roll, signal_df, outdir)
        print("[OK] Plots saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()