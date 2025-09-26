#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nuclear_restarts.py

Event study around Japan nuclear reactor restarts (or any policy/event dates).
- You provide event dates via --events "YYYY-MM-DD,YYYY-MM-DD,..." or --events-file path/to/events.csv
  (one ISO date per line, or a CSV with a 'date' column).
- Script downloads prices (via yfinance) for utilities, benchmark, and optional proxies.
- Computes benchmark-adjusted returns (utilities minus TOPIX ETF) OR market model (OLS) if --market-model.
- Outputs:
    ./artifacts/nuclear_restarts_YYYYMMDD_HHMMSS/
        prices.csv
        daily_returns.csv
        event_window_{Tpre}_{Tpost}.csv     (per-ticker average over events)
        AAR_CAAR.csv                        (average/ cumulative average returns over time)
        per_event_CAR.csv                   (CAR per event & ticker)
        summary_stats.csv                   (t-tests & effect sizes)
        plots/*.png                         (AAR/CAAR and per-ticker heatmaps)

Defaults (feel free to change tickers):
    Utilities: 9501.T (TEPCO), 9503.T (Kansai Electric), 9502.T (Chubu Electric)
    Benchmark: 1306.T (TOPIX ETF). If unavailable, try ^TOPX.
    Proxies  : CL=F (WTI), NG=F (US NatGas), USDJPY=X (FX).

Examples:
    python nuclear_restarts.py --events "2016-08-12,2017-10-10,2018-05-09" --plot
    python nuclear_restarts.py --events-file restarts.csv --market-model --tpre 10 --tpost 20 --plot
    python nuclear_restarts.py --utilities 9501.T,9503.T,9502.T,9531.T --benchmark 1306.T --proxies CL=F,NG=F,USDJPY=X --plot
"""

import argparse
import os
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Please install dependencies: pip install yfinance pandas numpy matplotlib statsmodels")

# matplotlib is optional (only used with --plot)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# statsmodels for market model (optional)
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# -------------------- Config & IO --------------------

@dataclass
class Config:
    utilities: List[str]
    benchmark: str
    proxies: List[str]
    start: str
    end: Optional[str]
    tpre: int
    tpost: int
    use_market_model: bool
    plot: bool
    outdir: str


def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"nuclear_restarts_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def parse_events(events_str: Optional[str], events_file: Optional[str]) -> List[pd.Timestamp]:
    dates: List[pd.Timestamp] = []
    if events_str:
        for s in events_str.split(","):
            s = s.strip()
            if not s:
                continue
            dates.append(pd.to_datetime(s).normalize())
    if events_file:
        df = pd.read_csv(events_file)
        if "date" in df.columns:
            col = "date"
        else:
            # assume first column
            col = df.columns[0]
        dates.extend(pd.to_datetime(df[col]).dt.normalize().tolist())
    # unique + sorted
    dates = sorted(list({d for d in dates if pd.notna(d)}))
    if not dates:
        raise SystemExit("No event dates provided. Use --events or --events-file.")
    return dates


# -------------------- Data --------------------

def fetch_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
        interval="1d",
    )

    # Flatten to Close only
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in df.columns:
                out[t] = df[(t, "Close")]
            elif (t, "Adj Close") in df.columns:
                out[t] = df[(t, "Adj Close")]
    else:
        # single ticker case
        t = tickers[0]
        out[t] = df["Close"] if "Close" in df.columns else df["Adj Close"]

    px = pd.DataFrame(out).sort_index().ffill().dropna(how="all")
    return px


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")


# -------------------- Models --------------------

def benchmark_adjusted(ri: pd.Series, rb: pd.Series) -> pd.Series:
    """Simple excess return over benchmark."""
    s = ri.align(rb, join="inner")
    return s[0] - s[1]


def market_model_abnormal(ri: pd.Series, rb: pd.Series, est_win: int = 120) -> pd.Series:
    """
    Market model abnormal returns using rolling OLS (estimated on trailing est_win days).
    AR_t = r_i,t - (alpha_hat + beta_hat * r_b,t)
    """
    if not HAVE_SM:
        # Fallback: beta via cov/var, alpha as mean difference
        ar = []
        idx = ri.index.intersection(rb.index)
        ri = ri.reindex(idx)
        rb = rb.reindex(idx)
        for i in range(len(idx)):
            if i < est_win:
                ar.append(np.nan)
                continue
            r_i = ri.iloc[i - est_win:i].values
            r_b = rb.iloc[i - est_win:i].values
            var = np.var(r_b, ddof=0)
            beta = np.cov(r_b, r_i, ddof=0)[0, 1] / var if var > 0 else 0.0
            alpha = r_i.mean() - beta * r_b.mean()
            ar.append(ri.iloc[i] - (alpha + beta * rb.iloc[i]))
        return pd.Series(ar, index=idx, name="ar")

    # Using statsmodels
    idx = ri.index.intersection(rb.index)
    ri = ri.reindex(idx)
    rb = rb.reindex(idx)
    ar = pd.Series(index=idx, dtype=float)
    for i in range(len(idx)):
        if i < est_win:
            ar.iloc[i] = np.nan
            continue
        y = ri.iloc[i - est_win:i].values
        X = sm.add_constant(rb.iloc[i - est_win:i].values)
        res = sm.OLS(y, X).fit()
        alpha, beta = res.params[0], res.params[1]
        ar.iloc[i] = ri.iloc[i] - (alpha + beta * rb.iloc[i])
    ar.name = "ar"
    return ar


# -------------------- Event Study --------------------

def event_window(df: pd.DataFrame, event_dates: List[pd.Timestamp], tpre: int, tpost: int) -> pd.DataFrame:
    """
    Build a panel with index = relative day (-tpre..tpost) and columns = MultiIndex (event_id, ticker)
    containing the series aligned around each event date (t=0 is the event day, or nearest trading day).
    """
    # Ensure a complete business day index to find nearest trading dates
    idx = df.index

    panels = []
    for eid, ed in enumerate(event_dates, start=1):
        # find nearest trading day on/after ed (if holiday)
        if ed not in idx:
            # next valid trading day
            after_idx = idx[idx >= ed]
            if after_idx.empty:
                continue
            ed_trading = after_idx[0]
        else:
            ed_trading = ed

        center_loc = idx.get_indexer([ed_trading])[0]
        start_loc = max(0, center_loc - tpre)
        end_loc = min(len(idx) - 1, center_loc + tpost)
        window = df.iloc[start_loc:end_loc + 1].copy()

        # Build relative time index
        rel_index = np.arange(start_loc - center_loc, end_loc - center_loc + 1)
        window.index = rel_index

        # Reindex to full range [-tpre, tpost] to pad missing edges with NaN
        window = window.reindex(np.arange(-tpre, tpost + 1))

        # Add MultiIndex columns: (event_eid, columnname)
        window.columns = pd.MultiIndex.from_product([[f"event_{eid}_{ed_trading.date()}"], window.columns])
        panels.append(window)

    if not panels:
        raise RuntimeError("No event windows constructed (dates may be out of range).")
    panel = pd.concat(panels, axis=1)
    return panel


def agg_aar_caar(panel_ar: pd.DataFrame, utilities: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute AAR (Average Abnormal Return) and CAAR (Cumulative AAR) across events and utilities.
    """
    # average across events for each ticker
    # columns: MultiIndex (event, ticker)
    # We'll average first across events, then across utilities (equal-weight).
    # 1) Per-ticker AAR: mean over events for each ticker
    per_ticker_aar = {}
    for t in utilities:
        cols = [c for c in panel_ar.columns if isinstance(c, tuple) and c[1] == t]
        if not cols:
            continue
        per_ticker_aar[t] = panel_ar[cols].mean(axis=1)
    per_ticker_aar = pd.DataFrame(per_ticker_aar)

    # 2) AAR across utilities
    AAR = per_ticker_aar.mean(axis=1).to_frame("AAR")

    # 3) CAAR
    CAAR = AAR.cumsum().rename(columns={"AAR": "CAAR"})
    return AAR, CAAR


def per_event_car(panel_ar: pd.DataFrame, utilities: List[str]) -> pd.DataFrame:
    """
    CAR per event & ticker over the event window (sum of AR from -tpre..tpost).
    """
    rows = []
    for col in panel_ar.columns:
        ev, ticker = col
        if ticker not in utilities:
            continue
        series = panel_ar[col].dropna()
        car = series.sum()
        rows.append({"event": ev, "ticker": ticker, "CAR": car})
    return pd.DataFrame(rows)


def ttest_zero(x: pd.Series) -> Tuple[float, float]:
    """
    One-sample t-test against zero. Returns (t_stat, p_value).
    """
    x = x.dropna()
    if len(x) < 3:
        return np.nan, np.nan
    mean = x.mean()
    sd = x.std(ddof=1)
    t = mean / (sd / math.sqrt(len(x))) if sd > 0 else np.nan
    # two-sided p using Student's t with n-1 df
    from scipy.stats import t as student_t  # scipy is common; if missing, we fallback
    try:
        p = 2 * (1 - student_t.cdf(abs(t), df=len(x) - 1))
    except Exception:
        p = np.nan
    return t, p


def summarize_effects(AAR: pd.DataFrame, CAAR: pd.DataFrame, per_event: pd.DataFrame) -> pd.DataFrame:
    """
    Summaries: AAR at t=0, CAAR at tpost, distribution of per-event CARs (mean, t-test, Cohen's d).
    """
    # AAR at event day (0)
    aar0 = AAR.loc[0, "AAR"] if 0 in AAR.index else np.nan
    # CAAR at last day
    caar_last = CAAR.iloc[-1, 0] if not CAAR.empty else np.nan

    # Per-event CAR stats
    car_mean = per_event["CAR"].mean()
    car_sd = per_event["CAR"].std(ddof=1)
    d = car_mean / car_sd if car_sd and not np.isnan(car_sd) and car_sd != 0 else np.nan
    try:
        tstat, pval = ttest_zero(per_event["CAR"])
    except Exception:
        tstat, pval = np.nan, np.nan

    summ = pd.DataFrame(
        [{
            "AAR_t0": aar0,
            "CAAR_end": caar_last,
            "CAR_mean": car_mean,
            "CAR_sd": car_sd,
            "Cohen_d": d,
            "t_stat": tstat,
            "p_value": pval,
            "n_events": per_event["event"].nunique() if not per_event.empty else 0
        }]
    )
    return summ


# -------------------- Plotting --------------------

def plot_aar_caar(AAR: pd.DataFrame, CAAR: pd.DataFrame, outdir: str, title_suffix: str = ""):
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return
    fig1 = plt.figure(figsize=(9, 5))
    AAR["AAR"].plot(ax=plt.gca(), linewidth=1.2)
    plt.axvline(0, linestyle="--")
    plt.title(f"AAR around Event Day {title_suffix}".strip())
    plt.xlabel("Event Time (days)")
    plt.ylabel("Average Abnormal Return")
    plt.tight_layout()
    fig1.savefig(os.path.join(outdir, "plots", "AAR.png"), dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9, 5))
    CAAR["CAAR"].plot(ax=plt.gca(), linewidth=1.2)
    plt.axvline(0, linestyle="--")
    plt.title(f"CAAR around Event Day {title_suffix}".strip())
    plt.xlabel("Event Time (days)")
    plt.ylabel("Cumulative AAR")
    plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "plots", "CAAR.png"), dpi=150)
    plt.close(fig2)


def plot_heatmap(panel_ar: pd.DataFrame, utilities: List[str], outdir: str):
    if plt is None:
        return
    # Build a matrix: index = event time, columns = (event,ticker) concatenated labels
    mat = panel_ar.copy()
    mat.columns = [f"{ev}|{tic}" for ev, tic in mat.columns]
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(mat.T.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(mat.columns)))
    ax.set_yticklabels(mat.columns, fontsize=7)
    ax.set_xticks(range(len(mat.index)))
    ax.set_xticklabels(mat.index, fontsize=7)
    ax.set_title("Abnormal Returns Heatmap (events × tickers)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "plots", "heatmap_abnormal_returns.png"), dpi=160)
    plt.close(fig)


# -------------------- Main --------------------

def main():
    p = argparse.ArgumentParser(description="Event study for nuclear restart dates on JP utilities.")
    p.add_argument("--utilities", type=str,
                   default="9501.T,9503.T,9502.T",
                   help="Comma-separated tickers for utilities (default: 9501.T,9503.T,9502.T)")
    p.add_argument("--benchmark", type=str, default="1306.T",
                   help="Benchmark ticker (default: 1306.T TOPIX ETF). Try ^TOPX if needed.")
    p.add_argument("--proxies", type=str, default="CL=F,NG=F,USDJPY=X",
                   help="Optional proxies (oil, gas, FX), comma-separated (default: CL=F,NG=F,USDJPY=X)")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--events", type=str, default=None,
                   help='Comma-separated ISO dates, e.g., "2016-08-12,2017-10-10"')
    p.add_argument("--events-file", type=str, default=None,
                   help="CSV file with a 'date' column or single column of dates")
    p.add_argument("--tpre", type=int, default=5, help="Pre-event window (trading days)")
    p.add_argument("--tpost", type=int, default=10, help="Post-event window (trading days)")
    p.add_argument("--market-model", action="store_true",
                   help="Use market model abnormal returns (rolling OLS) instead of simple benchmark-adjusted.")
    p.add_argument("--plot", action="store_true", help="Save plots (PNG)")
    p.add_argument("--outdir", type=str, default="./artifacts")
    args = p.parse_args()

    cfg = Config(
        utilities=[t.strip() for t in args.utilities.split(",") if t.strip()],
        benchmark=args.benchmark.strip(),
        proxies=[t.strip() for t in args.proxies.split(",") if t.strip()],
        start=args.start,
        end=args.end,
        tpre=args.tpre,
        tpost=args.tpost,
        use_market_model=bool(args.market_model),
        plot=bool(args.plot),
        outdir=args.outdir,
    )

    # Parse events
    events = parse_events(args.events, args.events_file)
    print(f"[INFO] Using {len(events)} event(s): {[d.date().isoformat() for d in events]}")

    # Compute a safe start date with estimation window
    est_win = 120 if cfg.use_market_model else 0
    start_dt = pd.to_datetime(cfg.start) - pd.Timedelta(days=(cfg.tpre + est_win + 30))
    start_str = start_dt.date().isoformat()

    # download prices
    tickers = cfg.utilities + [cfg.benchmark] + cfg.proxies
    tickers = sorted(list({t for t in tickers if t}))
    print(f"[INFO] Downloading {len(tickers)} ticker(s): {tickers}")
    px = fetch_prices(tickers, start=start_str, end=cfg.end)
    px.to_csv(os.path.join(ensure_outdir(cfg.outdir), "tmp.csv"))  # temp to get the auto folder name
    outdir = os.path.dirname(os.path.join(cfg.outdir, sorted(os.listdir(cfg.outdir))[-1], "x"))
    # overwrite outdir with the actual folder (more robust approach below)
    outdir = ensure_outdir(cfg.outdir)  # final outdir
    print(f"[INFO] Writing artifacts to: {outdir}")

    px.to_csv(os.path.join(outdir, "prices.csv"))
    r = daily_returns(px)
    r.to_csv(os.path.join(outdir, "daily_returns.csv"))

    # Split series
    r_b = r.get(cfg.benchmark)
    if r_b is None:
        raise RuntimeError(f"Benchmark {cfg.benchmark} not found in returns. Columns: {list(r.columns)}")

    # Abnormal returns per utility
    ar_df = pd.DataFrame(index=r.index)
    for t in cfg.utilities:
        if t not in r.columns:
            print(f"[WARN] {t} not found in returns. Skipping.")
            continue
        if cfg.use_market_model:
            ar = market_model_abnormal(r[t].dropna(), r_b.dropna(), est_win=120)
        else:
            ar = benchmark_adjusted(r[t].dropna(), r_b.dropna())
        ar_df[t] = ar

    # Build event window panels
    panel_ar = event_window(ar_df[cfg.utilities].dropna(how="all"), events, cfg.tpre, cfg.tpost)
    panel_ar.to_csv(os.path.join(outdir, f"event_window_{cfg.tpre}_{cfg.tpost}.csv"))

    # AAR/CAAR
    AAR, CAAR = agg_aar_caar(panel_ar, cfg.utilities)
    AAR.to_csv(os.path.join(outdir, "AAR.csv"))
    CAAR.to_csv(os.path.join(outdir, "CAAR.csv"))

    # Per-event CAR
    per_event = per_event_car(panel_ar, cfg.utilities)
    per_event.to_csv(os.path.join(outdir, "per_event_CAR.csv"), index=False)

    # Summary stats
    summary = summarize_effects(AAR, CAAR, per_event)
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"), index=False)

    # Save proxies (for your own macro overlays)
    proxy_cols = [c for c in cfg.proxies if c in px.columns]
    if proxy_cols:
        px[proxy_cols].to_csv(os.path.join(outdir, "proxies_prices.csv"))

    print("\n=== Summary ===")
    print(summary.round(4).to_string(index=False))

    # Plots
    if cfg.plot:
        plot_aar_caar(AAR, CAAR, outdir, title_suffix=f"(t=[-{cfg.tpre}, +{cfg.tpost}])")
        plot_heatmap(panel_ar, cfg.utilities, outdir)
        print("[OK] Plots saved to:", os.path.join(outdir, "plots"))

    print("\nDone.")


if __name__ == "__main__":
    main()