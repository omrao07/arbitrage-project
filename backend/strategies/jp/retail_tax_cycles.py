#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retail_tax_cycles.py

Event study for retail tax-cycle shocks:
- VAT/GST increases/decreases
- State sales-tax holidays
- Excise changes that materially affect retail pricing

INPUTS
- --events-file: CSV with at least 'date'. Optional: 'country','state','type','notes','tickers','tax_change_bps'
    Examples of 'type': vat_hike, vat_cut, gst_hike, gst_cut, tax_holiday, excise_hike, excise_cut
    'tax_change_bps' = +100 (for +1%), -200 (for -2%), etc.

- --map-file (optional): CSV to infer tickers by geography/type
    Columns (flexible): 'country','state','type','tickers'
    Rows with blank values act like wildcards; 'Global' country acts globally.

WHAT IT DOES
1) Loads events & resolves impacted tickers per event (explicit tickers ∪ mapped tickers).
2) Downloads daily prices via yfinance for all unique tickers + chosen benchmark.
3) Computes abnormal returns (benchmark-adjusted or market-model with --market-model).
4) Builds per-event windows [-tpre, +tpost]; exports AAR/CAAR, per-event CAR, summary stats.
5) Optional cross-sectional OLS of CAR on tax_change_bps and type/country/state dummies.
6) Writes plots (AAR/CAAR, heatmap).

USAGE
    python retail_tax_cycles.py \
        --events-file events.csv \
        --map-file mapping.csv \
        --benchmark ^NSEI \
        --tpre 5 --tpost 10 \
        --market-model --plot

DEPENDENCIES
    pip install yfinance pandas numpy matplotlib statsmodels
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
    outdir = os.path.join(base, f"retail_tax_cycles_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise SystemExit("events-file must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Normalize optional columns
    for col in ["country", "state", "type", "notes", "tickers", "tax_change_bps"]:
        if col not in df.columns:
            df[col] = np.nan
    # Coerce tax_change_bps to float
    def _to_bps(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace("%", "")
        try:
            # If given in %, convert to bps
            if float(s) <= 20 and "." in s:
                return float(s) * 100.0  # e.g., 1.5% -> 150 bps
            return float(s)
        except Exception:
            return np.nan
    df["tax_change_bps"] = df["tax_change_bps"].apply(_to_bps)
    return df.sort_values("date").reset_index(drop=True)


def load_map(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    m = pd.read_csv(path)
    # Ensure required column
    if "tickers" not in m.columns:
        raise SystemExit("map-file must include 'tickers'; optional: 'country','state','type'")
    for col in ["country", "state", "type"]:
        if col not in m.columns:
            m[col] = np.nan
    return m


def split_tickers(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    parts = [x.strip() for x in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def resolve_event_tickers(ev: pd.Series, mapper: Optional[pd.DataFrame]) -> List[str]:
    explicit = set(split_tickers(ev.get("tickers", "")))
    if mapper is None:
        return sorted(list(explicit))
    cand = mapper.copy()

    # Country filter (exact or Global/blank)
    c = str(ev.get("country", "")).strip().lower()
    cand = cand[
        (cand["country"].fillna("").str.lower().isin(["", "global"])) |
        (cand["country"].fillna("").str.lower() == c)
    ]

    # State filter (exact or blank). Only apply if provided
    if pd.notna(ev.get("state")) and str(ev["state"]).strip():
        s = str(ev["state"]).strip().lower()
        cand = cand[(cand["state"].fillna("").str.lower().isin([""])) | (cand["state"].fillna("").str.lower() == s)]

    # Type filter (match or wildcard)
    if pd.notna(ev.get("type")) and str(ev["type"]).strip():
        t = str(ev["type"]).strip().lower()
        cand = cand[(cand["type"].fillna("").str.lower().isin([""])) | (cand["type"].fillna("").str.lower() == t)]

    mapped: Set[str] = set()
    for _, row in cand.iterrows():
        mapped |= set(split_tickers(row.get("tickers", "")))

    return sorted(list(explicit | mapped))


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
        t = tickers[0]
        out[t] = data.get("Close", data.get("Adj Close"))
    px = pd.DataFrame(out).sort_index().ffill()
    # Warn missing
    missing = [t for t in tickers if t not in px.columns]
    for t in missing:
        print(f"[WARN] Missing prices for {t}; skipping.")
    return px.dropna(how="all", axis=1)


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
    AAR = panel.mean(axis=1).to_frame("AAR")
    CAAR = AAR.cumsum().rename(columns={"AAR": "CAAR"})
    return AAR, CAAR


def per_event_car(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in panel.columns:
        ev, tic = col
        series = panel[col].dropna()
        rows.append({"event": ev, "ticker": tic, "CAR": float(series.sum())})
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


# ---------------------------- Cross-sectional (optional) ----------------------------

def cross_sectional_ols(per_event_df: pd.DataFrame, events: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Regress CAR on tax_change_bps and dummies for type/country/state.
    Returns coeffs with t-stats (if statsmodels available).
    """
    if not HAVE_SM or per_event_df.empty:
        return None

    # Extract event_id from "event_#_YYYY-MM-DD"
    def _eid(s):
        try:
            return int(str(s).split("_")[1])
        except Exception:
            return np.nan

    df = per_event_df.copy()
    df["eid"] = df["event"].apply(_eid)
    ev = events.copy()
    ev = ev.assign(eid=np.arange(1, len(ev) + 1))

    df = df.merge(ev[["eid", "country", "state", "type", "tax_change_bps"]], on="eid", how="left")

    X = pd.DataFrame(index=df.index)
    X["tax_change_bps"] = df["tax_change_bps"].fillna(0.0)

    if "type" in df.columns:
        X = pd.concat([X, pd.get_dummies(df["type"].fillna("NA"), prefix="type", drop_first=True)], axis=1)
    if "country" in df.columns:
        X = pd.concat([X, pd.get_dummies(df["country"].fillna("NA"), prefix="country", drop_first=True)], axis=1)
    if "state" in df.columns:
        X = pd.concat([X, pd.get_dummies(df["state"].fillna("NA"), prefix="state", drop_first=True)], axis=1)

    X = sm.add_constant(X)
    y = df["CAR"].astype(float)
    model = sm.OLS(y, X, missing="drop").fit()
    out = pd.DataFrame({"coef": model.params, "t": model.tvalues, "p": model.pvalues})
    out.index.name = "var"
    return out


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
    p = argparse.ArgumentParser(description="Event study for retail tax-cycle shocks.")
    p.add_argument("--events-file", type=str, required=True,
                   help="CSV with 'date' and optional 'country','state','type','notes','tickers','tax_change_bps'")
    p.add_argument("--map-file", type=str, default=None,
                   help="CSV mapping to tickers: required 'tickers'; optional 'country','state','type'")
    p.add_argument("--benchmark", type=str, default="^NSEI",
                   help="Benchmark ticker (e.g., ^NSEI, ^BSESN, ^GSPC, ^N225, ^STOXX50E)")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--tpre", type=int, default=5)
    p.add_argument("--tpost", type=int, default=10)
    p.add_argument("--market-model", action="store_true",
                   help="Use market-model (rolling OLS) abnormal returns; else benchmark-adjusted")
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

    # Resolve tickers per event & gather universe
    per_event_tickers: Dict[int, List[str]] = {}
    all_tickers: Set[str] = set()
    for i in range(len(events)):
        tickers = resolve_event_tickers(events.loc[i], mapper)
        if not tickers:
            print(f"[WARN] Event {i+1} {events.loc[i,'date'].date()} has no tickers; may be skipped.")
        per_event_tickers[i] = tickers
        all_tickers |= set(tickers)

    # Include benchmark
    if cfg.benchmark:
        all_tickers |= {cfg.benchmark}
    if not all_tickers:
        raise SystemExit("No tickers to analyze. Provide tickers in events or a map-file.")

    # Extend start for estimation if using market model
    est_pad_days = 200 if cfg.use_market_model else 30
    start_ext = (pd.to_datetime(cfg.start) - pd.Timedelta(days=cfg.tpre + est_pad_days)).date().isoformat()

    print(f"[INFO] Downloading prices for {len(all_tickers)} tickers...")
    px = fetch_prices(sorted(all_tickers), start_ext, cfg.end)
    px.to_csv(os.path.join(outdir, "prices.csv"))

    r = daily_returns(px)
    r.to_csv(os.path.join(outdir, "daily_returns.csv"))

    if cfg.benchmark not in r.columns:
        raise SystemExit(f"Benchmark {cfg.benchmark} not found in returns. Columns: {list(r.columns)}")

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

    # Event windows
    panel = build_event_windows(ar_df, events, per_event_tickers, cfg.tpre, cfg.tpost)
    panel.to_csv(os.path.join(outdir, f"event_window_{cfg.tpre}_{cfg.tpost}.csv"))

    # AAR/CAAR
    AAR, CAAR = aggregate_aar_caar(panel)
    AAR.to_csv(os.path.join(outdir, "AAR.csv"))
    CAAR.to_csv(os.path.join(outdir, "CAAR.csv"))

    # Per-event CAR
    per_event = per_event_car(panel)
    per_event.to_csv(os.path.join(outdir, "per_event_CAR.csv"), index=False)

    # Summary stats
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

    # Cross-sectional OLS (optional)
    xsec = cross_sectional_ols(per_event, events)
    if xsec is not None:
        xsec.to_csv(os.path.join(outdir, "cross_sectional_ols.csv"))
        print("\n=== Cross-sectional OLS (CAR drivers) ===")
        print(xsec.round(4).to_string())

    # Plots
    if cfg.plot:
        plot_aar_caar(AAR, CAAR, outdir, title_suffix=f"(t=[-{cfg.tpre}, +{cfg.tpost}])")
        plot_heatmap(panel, outdir)
        print("[OK] Plots saved to:", os.path.join(outdir, "plots"))

    print("\nDone.")


if __name__ == "__main__":
    main()