#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offshore_wind_projects.py

Event study for offshore-wind project milestones (auction awards, FID, financial close, grid connection, etc.)
on impacted equities (developers, OEMs, EPC/maritime, utilities).

INPUTS
- --events-file: CSV with at least a 'date' column (ISO). Optional columns: 'project','country','type','notes','tickers'
    Example rows:
        date,project,country,type,notes,tickers
        2023-12-12,Sai Kung OWF,Hong Kong,award,Phase 1,0008.HK;VWS.CO
        2024-03-18,Somanhalli OWF,India,FID,,ADANIGREEN.NS;532540.BO
  If 'tickers' is empty, the script can auto-map from --map-file.

- --map-file: CSV mapping rules to infer tickers impacted by an event:
    Columns: 'country','role','tickers'
    Example:
        country,role,tickers
        Japan,developer,9501.T;9503.T
        Global,OEM,VWS.CO;ENR.DE;GE
    If provided, we'll union tickers from (a) the event's explicit 'tickers' column and (b) all rows whose
    'country' matches event.country or is 'Global'. You may also add a 'type' column to restrict by event type.

- --benchmark: Market or sector proxy for abnormal returns (default: ^STOXX50E or ^GSPC etc. choose below).

WHAT IT DOES
1) Loads events; resolves affected tickers per event (union of explicit and mapped).
2) Downloads daily prices for all unique tickers + benchmark (via yfinance).
3) Computes daily returns and abnormal returns (benchmark-adjusted or market-model OLS with --market-model).
4) Constructs event windows [-tpre, +tpost] per event×ticker, then aggregates AAR (avg abnormal return)
   and CAAR (cumulative AAR). Also exports per-event CARs and summary stats.
5) Optional faceted stats by role/country if present in inputs.
6) Saves CSV artifacts & PNG plots.

USAGE
    python offshore_wind_projects.py \
        --events-file events.csv \
        --map-file mapping.csv \
        --benchmark ^STOXX50E \
        --tpre 5 --tpost 10 \
        --market-model --plot

DEPENDENCIES
    pip install yfinance pandas numpy matplotlib statsmodels

NOTES
- Choose a benchmark that best reflects your universe (examples below).
- If a ticker fails to download, it’s skipped with a warning.
"""

import argparse
import os
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Set

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install deps: pip install yfinance pandas numpy matplotlib statsmodels")

# Optional libs
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# ---------------------------- Config ----------------------------

@dataclass
class Config:
    events_file: str
    map_file: Optional[str]
    benchmark: str
    start: str
    end: Optional[str]
    tpre: int
    tpost: int
    use_market_model: bool
    plot: bool
    outdir: str


# ---------------------------- IO utils ----------------------------

def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"offshore_wind_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise SystemExit("events-file must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Normalise expected columns
    for col in ["project", "country", "type", "notes", "tickers", "role"]:
        if col not in df.columns:
            df[col] = np.nan
    return df.sort_values("date").reset_index(drop=True)


def load_map(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    m = pd.read_csv(path)
    cols = set(c.lower() for c in m.columns)
    if "tickers" not in cols:
        raise SystemExit("map-file must have a 'tickers' column; optional: 'country','role','type'")
    # Normalize optional columns
    for need in ["country", "role", "type"]:
        if need not in m.columns:
            m[need] = np.nan
    return m


def split_tickers(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    # Allow comma or semicolon separation
    parts = [x.strip() for x in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def resolve_event_tickers(ev: pd.Series, mapper: Optional[pd.DataFrame]) -> List[str]:
    explicit = set(split_tickers(ev.get("tickers", "")))
    if mapper is None:
        return sorted(list(explicit))
    cand = mapper.copy()
    # Filters: country (exact or Global), type (match if provided)
    if pd.notna(ev.get("country")):
        cand = cand[(cand["country"].fillna("").str.lower() == str(ev["country"]).lower()) |
                    (cand["country"].fillna("").str.lower() == "global")]
    else:
        cand = cand[cand["country"].fillna("").str.lower().isin(["", "global"])]

    if "type" in mapper.columns and pd.notna(ev.get("type")):
        # include rows where type matches OR mapper type is NaN/blank
        cand = cand[(cand["type"].fillna("").str.lower() == str(ev["type"]).lower()) |
                    (cand["type"].fillna("") == "")]

    mapped: Set[str] = set()
    for _, row in cand.iterrows():
        mapped |= set(split_tickers(row.get("tickers", "")))

    union = sorted(list(explicit | mapped))
    return union


# ---------------------------- Data / Returns ----------------------------

def fetch_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in data.columns:
                out[t] = data[(t, "Close")]
            elif (t, "Adj Close") in data.columns:
                out[t] = data[(t, "Adj Close")]
            else:
                # silently skip; we'll warn later
                pass
    else:
        # Single ticker case
        t = tickers[0]
        out[t] = data.get("Close", data.get("Adj Close"))
    px = pd.DataFrame(out).sort_index().ffill()
    # Drop columns entirely missing
    missing = [t for t in tickers if t not in px.columns]
    for t in missing:
        print(f"[WARN] Missing prices for {t}; skipping.")
    px = px.dropna(how="all", axis=1)
    return px


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")


def market_model_abnormal(ri: pd.Series, rb: pd.Series, est_win: int = 120) -> pd.Series:
    idx = ri.index.intersection(rb.index)
    ri = ri.reindex(idx)
    rb = rb.reindex(idx)
    ar = pd.Series(index=idx, dtype=float)
    if HAVE_SM:
        for i in range(len(idx)):
            if i < est_win:
                ar.iloc[i] = np.nan
                continue
            y = ri.iloc[i - est_win:i].values
            X = sm.add_constant(rb.iloc[i - est_win:i].values)
            res = sm.OLS(y, X).fit()
            alpha, beta = res.params[0], res.params[1]
            ar.iloc[i] = ri.iloc[i] - (alpha + beta * rb.iloc[i])
    else:
        for i in range(len(idx)):
            if i < est_win:
                ar.iloc[i] = np.nan
                continue
            r_i = ri.iloc[i - est_win:i].values
            r_b = rb.iloc[i - est_win:i].values
            var = np.var(r_b, ddof=0)
            beta = (np.cov(r_b, r_i, ddof=0)[0, 1] / var) if var > 0 else 0.0
            alpha = r_i.mean() - beta * r_b.mean()
            ar.iloc[i] = ri.iloc[i] - (alpha + beta * rb.iloc[i])
    ar.name = "ar"
    return ar


def benchmark_adjusted(ri: pd.Series, rb: pd.Series) -> pd.Series:
    s = ri.align(rb, join="inner")
    return s[0] - s[1]


# ---------------------------- Event windows & Aggregation ----------------------------

def build_event_windows(ar_df: pd.DataFrame,
                        events: pd.DataFrame,
                        per_event_tickers: Dict[int, List[str]],
                        tpre: int, tpost: int) -> pd.DataFrame:
    """
    Panel of abnormal returns around events.
    Index = relative day [-tpre..tpost]
    Columns = MultiIndex (event_id, ticker)
    """
    idx_prices = ar_df.index
    panels = []
    for eid in range(len(events)):
        edate = events.loc[eid, "date"]
        # Choose nearest trading day on/after date
        if edate not in idx_prices:
            after = idx_prices[idx_prices >= edate]
            if after.empty:
                continue
            ed_trade = after[0]
        else:
            ed_trade = edate
        center = idx_prices.get_indexer([ed_trade])[0]
        start = max(0, center - tpre)
        end = min(len(idx_prices) - 1, center + tpost)
        tickers = [t for t in per_event_tickers.get(eid, []) if t in ar_df.columns]
        if not tickers:
            continue
        win = ar_df.loc[idx_prices[start]:idx_prices[end], tickers].copy()
        rel_idx = np.arange(start - center, end - center + 1)
        win.index = rel_idx
        win = win.reindex(np.arange(-tpre, tpost + 1))
        win.columns = pd.MultiIndex.from_product([[f"event_{eid+1}_{str(ed_trade.date())}"], tickers])
        panels.append(win)
    if not panels:
        raise RuntimeError("No event windows constructed (check dates/tickers).")
    return pd.concat(panels, axis=1)


def aggregate_aar_caar(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Average across all events and tickers equally
    AAR = panel.mean(axis=1).to_frame("AAR")
    CAAR = AAR.cumsum().rename(columns={"AAR": "CAAR"})
    return AAR, CAAR


def per_event_car(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in panel.columns:
        event_key, ticker = col
        series = panel[col].dropna()
        rows.append({"event": event_key, "ticker": ticker, "CAR": float(series.sum())})
    return pd.DataFrame(rows)


def ttest_zero(x: pd.Series) -> Tuple[float, float]:
    x = x.dropna()
    if len(x) < 3:
        return np.nan, np.nan
    m = x.mean()
    s = x.std(ddof=1)
    t = m / (s / math.sqrt(len(x))) if (s and not np.isnan(s) and s != 0) else np.nan
    try:
        from scipy.stats import t as student_t
        p = 2 * (1 - student_t.cdf(abs(t), df=len(x) - 1))
    except Exception:
        p = np.nan
    return t, p


# ---------------------------- Plotting ----------------------------

def plot_aar_caar(AAR: pd.DataFrame, CAAR: pd.DataFrame, outdir: str, title_suffix: str = ""):
    if plt is None:
        print("[INFO] matplotlib not available; skipping plots.")
        return
    fig1 = plt.figure(figsize=(9, 5))
    AAR["AAR"].plot(ax=plt.gca(), linewidth=1.2)
    plt.axvline(0, linestyle="--")
    plt.title(f"AAR around event day {title_suffix}".strip())
    plt.xlabel("Event time (days)")
    plt.ylabel("Average abnormal return")
    plt.tight_layout()
    fig1.savefig(os.path.join(outdir, "plots", "AAR.png"), dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9, 5))
    CAAR["CAAR"].plot(ax=plt.gca(), linewidth=1.2)
    plt.axvline(0, linestyle="--")
    plt.title(f"CAAR around event day {title_suffix}".strip())
    plt.xlabel("Event time (days)")
    plt.ylabel("Cumulative AAR")
    plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "plots", "CAAR.png"), dpi=150)
    plt.close(fig2)


def plot_heatmap(panel: pd.DataFrame, outdir: str):
    if plt is None:
        return
    M = panel.copy()
    M.columns = [f"{ev}|{tic}" for ev, tic in M.columns]
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(M.T.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(M.columns)))
    ax.set_yticklabels(M.columns, fontsize=7)
    ax.set_xticks(range(len(M.index)))
    ax.set_xticklabels(M.index, fontsize=7)
    ax.set_title("Abnormal returns heatmap (events × tickers)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "plots", "heatmap_abnormal_returns.png"), dpi=160)
    plt.close(fig)


# ---------------------------- Main ----------------------------

def main():
    p = argparse.ArgumentParser(description="Event study for offshore-wind project milestones.")
    p.add_argument("--events-file", type=str, required=True,
                   help="CSV with 'date' and optional 'project','country','type','notes','tickers'")
    p.add_argument("--map-file", type=str, default=None,
                   help="CSV mapping (country,role,type?) -> tickers (semicolon/comma separated)")
    p.add_argument("--benchmark", type=str, default="^STOXX50E",
                   help="Benchmark ticker (e.g., ^STOXX50E, ^GSPC, ^N225, 1306.T, ^NSEI)")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--tpre", type=int, default=5)
    p.add_argument("--tpost", type=int, default=10)
    p.add_argument("--market-model", action="store_true",
                   help="Use market-model (rolling OLS) abnormal returns; else simple benchmark-adjusted")
    p.add_argument("--plot", action="store_true", help="Write PNG plots")
    p.add_argument("--outdir", type=str, default="./artifacts")
    args = p.parse_args()

    cfg = Config(
        events_file=args.events_file,
        map_file=args.map_file,
        benchmark=args.benchmark,
        start=args.start,
        end=args.end,
        tpre=args.tpre,
        tpost=args.tpost,
        use_market_model=bool(args.market_model),
        plot=bool(args.plot),
        outdir=args.outdir,
    )

    outdir = ensure_outdir(cfg.outdir)
    print(f"[INFO] Writing artifacts to: {outdir}")

    events = load_events(cfg.events_file)
    mapper = load_map(cfg.map_file)

    # Resolve tickers per event
    per_event_tickers: Dict[int, List[str]] = {}
    all_tickers: Set[str] = set()
    for i in range(len(events)):
        tickers = resolve_event_tickers(events.loc[i], mapper)
        if not tickers:
            print(f"[WARN] Event {i+1} {events.loc[i,'date'].date()} has no tickers; skipped in analysis.")
        per_event_tickers[i] = tickers
        all_tickers |= set(tickers)

    # Add benchmark
    if cfg.benchmark:
        all_tickers |= {cfg.benchmark}

    if not all_tickers:
        raise SystemExit("No tickers to analyze. Provide tickers in events or a proper map-file.")

    # Extend start date for market-model estimation
    est_pad_days = 200 if cfg.use_market_model else 30
    start_ext = (pd.to_datetime(cfg.start) - pd.Timedelta(days=cfg.tpre + est_pad_days)).date().isoformat()

    print(f"[INFO] Downloading prices for {len(all_tickers)} tickers...")
    px = fetch_prices(sorted(all_tickers), start_ext, cfg.end)
    px.to_csv(os.path.join(outdir, "prices.csv"))

    # Returns
    r = daily_returns(px)
    r.to_csv(os.path.join(outdir, "daily_returns.csv"))

    # Abnormal returns
    if cfg.benchmark not in r.columns:
        raise SystemExit(f"Benchmark {cfg.benchmark} not in price data. Available columns: {list(r.columns)}")

    rb = r[cfg.benchmark].dropna()
    ar_df = pd.DataFrame(index=r.index)
    for t in sorted(all_tickers):
        if t == cfg.benchmark or t not in r.columns:
            continue
        if cfg.use_market_model:
            ar = market_model_abnormal(r[t].dropna(), rb)
        else:
            ar = benchmark_adjusted(r[t].dropna(), rb)
        ar_df[t] = ar

    # Build event windows
    panel = build_event_windows(ar_df, events, per_event_tickers, cfg.tpre, cfg.tpost)
    panel.to_csv(os.path.join(outdir, f"event_window_{cfg.tpre}_{cfg.tpost}.csv"))

    # AAR & CAAR (overall)
    AAR, CAAR = aggregate_aar_caar(panel)
    AAR.to_csv(os.path.join(outdir, "AAR.csv"))
    CAAR.to_csv(os.path.join(outdir, "CAAR.csv"))

    # Per-event CARs
    per_event = per_event_car(panel)
    per_event.to_csv(os.path.join(outdir, "per_event_CAR.csv"), index=False)

    # Summary stats
    # AAR at t=0
    aar0 = AAR.loc[0, "AAR"] if 0 in AAR.index else np.nan
    caar_end = CAAR.iloc[-1, 0] if not CAAR.empty else np.nan
    tstat, pval = ttest_zero(per_event["CAR"]) if not per_event.empty else (np.nan, np.nan)
    summary = pd.DataFrame([{
        "AAR_t0": aar0,
        "CAAR_end": caar_end,
        "CAR_mean": per_event["CAR"].mean() if not per_event.empty else np.nan,
        "CAR_sd": per_event["CAR"].std(ddof=1) if not per_event.empty else np.nan,
        "t_stat": tstat,
        "p_value": pval,
        "n_events": per_event["event"].nunique() if not per_event.empty else 0
    }])
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"), index=False)

    print("\n=== Summary ===")
    print(summary.round(4).to_string(index=False))

    # Optional faceting if events have 'country' or 'type'
    for facet_col in ["country", "type"]:
        if facet_col in events.columns and events[facet_col].notna().any():
            # Subset columns for events matching facet value
            vals = sorted([v for v in events[facet_col].dropna().unique()])
            for val in vals:
                cols = [c for c in panel.columns if c[0].startswith("event_")]
                keep_events = [f"event_{i+1}_" for i in range(len(events)) if str(events.loc[i, facet_col]) == str(val)]
                sub_cols = [c for c in panel.columns if any(c[0].startswith(k) for k in keep_events)]
                if not sub_cols:
                    continue
                sub_panel = panel[sub_cols]
                sub_AAR, sub_CAAR = aggregate_aar_caar(sub_panel)
                sub_AAR.to_csv(os.path.join(outdir, f"AAR_{facet_col}_{val}.csv"))
                sub_CAAR.to_csv(os.path.join(outdir, f"CAAR_{facet_col}_{val}.csv"))

    if cfg.plot:
        plot_aar_caar(AAR, CAAR, outdir, title_suffix=f"(t=[-{cfg.tpre}, +{cfg.tpost}])")
        plot_heatmap(panel, outdir)
        print("[OK] Plots saved to:", os.path.join(outdir, "plots"))

    print("\nDone.")


if __name__ == "__main__":
    main()