#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# earnings_drift.py
#
# Post-Earnings Announcement Drift (PEAD) toolkit
# ------------------------------------------------
# - Ingests an earnings events file (ticker, date, surprise, etc.)
# - Downloads prices (yfinance) for tickers + a benchmark (e.g., SPY)
# - Builds event windows and computes AAR/CAAR around earnings
# - Ranks events into surprise quantiles (or by day-0 abnormal return)
# - Estimates PEAD by quantile and Q5-Q1 long-short from t=hold_start..hold_end
# - Optional calendar-time portfolio: average return of overlapping event portfolios
# - Exports tidy CSVs and optional PNG plots
#
# Usage
# -----
# python earnings_drift.py \
#   --events-file events.csv \
#   --benchmark SPY \
#   --tpre 1 --tpost 60 \
#   --hold-start 2 --hold-end 60 \
#   --quantiles 5 \
#   --sort-by surprise \
#   --plot
#
# events.csv (minimum columns)
# ----------------------------
# date,ticker,surprise   # date=YYYY-MM-DD ; surprise can be % or z-score (higher=more positive)
# Optional columns: direction, rev_surprise, guide, mcap, notes
#
# Outputs
# -------
# outdir/
#   prices.csv
#   returns.csv
#   event_panel_raw.csv            (simple returns by relative day per event)
#   event_panel_abn.csv            (benchmark-adjusted)
#   aar_caar_by_quantile.csv
#   long_short_curve.csv
#   calendar_time_portfolios.csv
#   summary.csv
#   plots/*.png  (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy yfinance matplotlib

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    raise SystemExit("Please install yfinance: pip install yfinance")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------------------- Config ----------------------------

@dataclass
class Config:
    events_file: str
    benchmark: str
    start: Optional[str]
    end: Optional[str]
    tpre: int
    tpost: int
    hold_start: int
    hold_end: int
    quantiles: int
    sort_by: str  # 'surprise' or 'day0_abn'
    drop_zero_surprise: bool
    plot: bool
    outdir: str


# ---------------------------- Helpers ----------------------------

def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"earnings_drift_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise SystemExit("events-file must have columns: date, ticker (and ideally surprise)")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "surprise" not in df.columns:
        df["surprise"] = np.nan
    else:
        # coerce to numeric (strip % if present)
        df["surprise"] = pd.to_numeric(df["surprise"].astype(str).str.replace("%","", regex=False), errors="coerce")
    # Drop obvious bad rows
    df = df.dropna(subset=["date","ticker"]).reset_index(drop=True)
    return df


def yf_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                sub = data[t][["Open","High","Low","Close","Volume"]].copy()
                sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
                frames.append(sub)
        px = pd.concat(frames, axis=1).sort_index()
    else:
        # single ticker case
        t = tickers[0]
        sub = data[["Open","High","Low","Close","Volume"]].copy()
        sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
        px = sub.sort_index()
    return px


def simple_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def abnormal_returns(stock: pd.Series, bench: pd.Series) -> pd.Series:
    return stock.align(bench, join="inner")[0] - stock.align(bench, join="inner")[1]


def snap_to_trading_day(idx: pd.DatetimeIndex, d: pd.Timestamp) -> Optional[pd.Timestamp]:
    if d in idx:
        return d
    nxt = idx[idx >= d]
    return None if nxt.empty else nxt[0]


# ---------------------------- Event windows ----------------------------

def build_event_panel(returns: pd.DataFrame, events: pd.DataFrame,
                      tpre: int, tpost: int) -> Tuple[pd.DataFrame, Dict[int, Tuple[str, pd.Timestamp]]]:
    """
    returns: DataFrame of returns (columns = tickers)
    events : DataFrame with 'date', 'ticker'
    Returns: (panel, mapping) where panel index = [-tpre..tpost], columns = event_id,
             and mapping maps event_id -> (ticker, snapped_date)
    """
    idx = returns.index
    panels = []
    meta = {}
    eid = 1
    for _, row in events.iterrows():
        t = row["ticker"]
        d = row["date"]
        if t not in returns.columns:  # may be missing (yfinance naming differences)
            continue
        sd = snap_to_trading_day(idx, d)
        if sd is None:
            continue
        center = idx.get_indexer([sd])[0]
        start = max(0, center - tpre)
        end = min(len(idx) - 1, center + tpost)
        rel_idx = np.arange(start - center, end - center + 1)
        win = returns[t].iloc[start:end+1].copy()
        win.index = rel_idx
        win.name = eid
        panels.append(win)
        meta[eid] = (t, sd)
        eid += 1
    if not panels:
        raise RuntimeError("No event windows could be built. Check tickers/dates coverage.")
    panel = pd.concat(panels, axis=1).reindex(np.arange(-tpre, tpost + 1))
    return panel, meta


def quantile_buckets(events: pd.DataFrame, var: pd.Series, q: int) -> pd.Series:
    # Rank into q quantiles; ties='first' to avoid pile-ups; drop NaNs → they’ll be NaN
    try:
        buckets = pd.qcut(var, q, labels=[f"Q{i}" for i in range(1, q+1)])
    except Exception:
        # Fallback: rank then cut
        ranks = var.rank(method="first", na_option="keep")
        buckets = pd.qcut(ranks, q, labels=[f"Q{i}" for i in range(1, q+1)])
    return buckets


def make_aar_caar(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aar = panel.mean(axis=1).to_frame("AAR")
    caar = aar.cumsum().rename(columns={"AAR": "CAAR"})
    return aar, caar


# ---------------------------- Calendar-time portfolio ----------------------------

def calendar_time_returns(panel: pd.DataFrame, meta: Dict[int, Tuple[str, pd.Timestamp]],
                          hold_start: int, hold_end: int) -> pd.Series:
    """
    Converts event-relative returns into a calendar-time portfolio:
    each event contributes its stock's simple return on calendar days from
    t=hold_start..hold_end; equal-weight across active events each day.
    """
    # Build a long table of (calendar_date, event_id, ret)
    rows = []
    for eid in panel.columns:
        t0_idx = panel.index
        for tau in range(hold_start, hold_end + 1):
            if tau not in t0_idx: continue
            r = panel.loc[tau, eid]
            if pd.isna(r): continue
            stock, sd = meta[eid]
            cal_date = sd + timedelta(days=int(tau))  # approximate; will realign after
            rows.append({"date": cal_date, "event_id": eid, "ret": r})
    if not rows:
        return pd.Series(dtype=float)
    long = pd.DataFrame(rows)
    # Group by date, average across events
    daily = long.groupby("date")["ret"].mean().sort_index()
    return daily


# ---------------------------- Pipeline ----------------------------

def run(cfg: Config):
    # Load events
    ev = load_events(cfg.events_file)

    # Choose initial price range so we have pre/post windows
    start_auto = (ev["date"].min() - pd.Timedelta(days=cfg.tpre + 10)).date().isoformat()
    start = cfg.start or start_auto
    end = cfg.end  # yfinance handles None as 'today'
    tickers = sorted(ev["ticker"].unique().tolist())
    universe = tickers + [cfg.benchmark]

    # Download
    px = yf_prices(universe, start, end)
    if px.empty:
        raise SystemExit("No price data downloaded. Check tickers, benchmark, and dates.")
    # Save for reference
    flat = pd.DataFrame(index=px.index)
    for t in universe:
        if t in px.columns.get_level_values(0):
            for f in ["Open","High","Low","Close","Volume"]:
                if (t, f) in px.columns:
                    flat[f"{t}_{f}"] = px[(t, f)]
    flat.to_csv(os.path.join(cfg.outdir, "prices.csv"))

    # Returns (simple)
    rets = {}
    for t in universe:
        if (t, "Close") in px.columns:
            rets[t] = simple_returns(px[(t, "Close")].astype(float))
    rets = pd.DataFrame(rets).dropna(how="all")
    rets.to_csv(os.path.join(cfg.outdir, "returns.csv"))

    # Abnormal returns (stock minus benchmark)
    if cfg.benchmark in rets.columns:
        abn = rets.sub(rets[cfg.benchmark], axis=0)
    else:
        abn = rets.copy()

    # Build event panels (raw & abnormal)
    # Filter events whose ticker missing in returns
    ev = ev[ev["ticker"].isin([c for c in rets.columns if c != cfg.benchmark])].copy()
    if ev.empty:
        raise SystemExit("No valid events after matching tickers to downloaded data.")

    raw_panel, meta = build_event_panel(rets.drop(columns=[cfg.benchmark], errors="ignore"), ev, cfg.tpre, cfg.tpost)
    abn_panel, _ = build_event_panel(abn.drop(columns=[cfg.benchmark], errors="ignore"), ev, cfg.tpre, cfg.tpost)
    raw_panel.to_csv(os.path.join(cfg.outdir, "event_panel_raw.csv"))
    abn_panel.to_csv(os.path.join(cfg.outdir, "event_panel_abn.csv"))

    # Day-0 abnormal for sorting (if needed)
    day0_abn = abn_panel.loc[0].rename("day0_abn")

    # Attach event_id back to events
    ev = ev.reset_index(drop=True)
    ev["event_id"] = range(1, len(ev) + 1)
    # Align: keep only events that survived panel building
    valid_ids = [c for c in abn_panel.columns]
    ev = ev[ev["event_id"].isin(valid_ids)].copy()
    ev = ev.set_index("event_id").sort_index()

    # Sorting variable
    if cfg.sort_by == "surprise" and ("surprise" in ev.columns) and not ev["surprise"].isna().all():
        sort_var = ev["surprise"]
        if cfg.drop_zero_surprise:
            ev = ev[~(ev["surprise"] == 0) & ~(ev["surprise"].isna())]
            # Refilter panel columns
            keep_cols = ev.index.tolist()
            abn_panel = abn_panel[keep_cols]
            raw_panel = raw_panel[keep_cols]
            day0_abn = day0_abn.reindex(keep_cols)
    else:
        # Fallback to day-0 abnormal return as the “surprise proxy”
        cfg.sort_by = "day0_abn"
        sort_var = day0_abn

    # Quantile labels
    buckets = quantile_buckets(ev.reset_index(), sort_var, cfg.quantiles)
    # Reindex back to event_id
    ev["bucket"] = pd.Series(buckets.values, index=ev.index)
    if ev["bucket"].isna().any():
        # Drop NaN buckets (insufficient info)
        drop_ids = ev[ev["bucket"].isna()].index.tolist()
        ev = ev.drop(index=drop_ids)
        abn_panel = abn_panel.drop(columns=drop_ids, errors="ignore")
        raw_panel = raw_panel.drop(columns=drop_ids, errors="ignore")

    # AAR/CAAR by quantile (abnormal)
    aar_caar_rows = []
    for label in sorted(ev["bucket"].dropna().unique().tolist()):
        ids = ev[ev["bucket"] == label].index.tolist()
        if not ids:
            continue
        aar, caar = make_aar_caar(abn_panel[ids])
        tmp = pd.DataFrame({"tau": aar.index, "metric": "AAR", "bucket": label, "value": aar["AAR"].values})
        tmp2 = pd.DataFrame({"tau": caar.index, "metric": "CAAR", "bucket": label, "value": caar["CAAR"].values})
        aar_caar_rows.append(tmp); aar_caar_rows.append(tmp2)
    aar_caar = pd.concat(aar_caar_rows, ignore_index=True)
    aar_caar.to_csv(os.path.join(cfg.outdir, "aar_caar_by_quantile.csv"), index=False)

    # Long-short curve (Q5 - Q1) using CAAR within quantiles
    def q_label(i): return f"Q{i}"
    top = q_label(cfg.quantiles)
    bot = q_label(1)
    c_top = aar_caar[(aar_caar["metric"] == "CAAR") & (aar_caar["bucket"] == top)].set_index("tau")["value"]
    c_bot = aar_caar[(aar_caar["metric"] == "CAAR") & (aar_caar["bucket"] == bot)].set_index("tau")["value"]
    ls = (c_top - c_bot).to_frame("LS_CAAR")
    ls.index.name = "tau"
    ls.to_csv(os.path.join(cfg.outdir, "long_short_curve.csv"))

    # Calendar-time portfolios for top & bottom buckets (abnormal returns)
    cal_top = calendar_time_returns(abn_panel[ev[ev["bucket"] == top].index], meta, cfg.hold_start, cfg.hold_end)
    cal_bot = calendar_time_returns(abn_panel[ev[ev["bucket"] == bot].index], meta, cfg.hold_start, cfg.hold_end)
    cal_df = pd.DataFrame({"TOP_eqw": cal_top, "BOT_eqw": cal_bot})
    cal_df["LS_eqw"] = cal_df["TOP_eqw"] - cal_df["BOT_eqw"]
    cal_df = cal_df.dropna(how="all").sort_index()
    cal_df.to_csv(os.path.join(cfg.outdir, "calendar_time_portfolios.csv"))

    # Summary stats over the hold window (abnormal)
    def window_stat(ids: List[int]) -> Dict[str, float]:
        # Sum abnormal returns from hold_start..hold_end per event, then average across events
        sub = abn_panel[ids].loc[cfg.hold_start:cfg.hold_end]
        # event-level CAR
        car = sub.sum(axis=0)
        return {
            "N_events": float(len(ids)),
            "mean_CAR": float(car.mean()),
            "median_CAR": float(car.median()),
            "tstat_mean": float((car.mean() / (car.std(ddof=1)/np.sqrt(len(car)))) if len(car) > 1 and car.std(ddof=1) > 0 else np.nan)
        }

    summ_rows = []
    for label in sorted(ev["bucket"].dropna().unique().tolist()):
        ids = ev[ev["bucket"] == label].index.tolist()
        stats = window_stat(ids)
        stats["bucket"] = label
        summ_rows.append(stats)

    # Long-short summary
    top_stats = window_stat(ev[ev["bucket"] == top].index.tolist())
    bot_stats = window_stat(ev[ev["bucket"] == bot].index.tolist())
    ls_mean = top_stats["mean_CAR"] - bot_stats["mean_CAR"]
    # Approximate t-stat for difference of means (unequal n, pooled s)
    def diff_t(car_top, car_bot):
        if len(car_top) < 2 or len(car_bot) < 2: return np.nan
        m1, m2 = car_top.mean(), car_bot.mean()
        s1, s2 = car_top.std(ddof=1), car_bot.std(ddof=1)
        n1, n2 = len(car_top), len(car_bot)
        se = np.sqrt((s1**2)/n1 + (s2**2)/n2)
        return (m1 - m2) / se if se and not np.isnan(se) and se != 0 else np.nan

    car_top = abn_panel[ev[ev["bucket"] == top].index].loc[cfg.hold_start:cfg.hold_end].sum(axis=0)
    car_bot = abn_panel[ev[ev["bucket"] == bot].index].loc[cfg.hold_start:cfg.hold_end].sum(axis=0)

    summary = pd.DataFrame(summ_rows)
    summary = pd.concat([
        summary,
        pd.DataFrame([{
            "bucket": f"{top}-{bot}",
            "N_events": float(len(car_top) + len(car_bot)),
            "mean_CAR": float(ls_mean),
            "median_CAR": float((car_top - car_bot).median() if (len(car_top) and len(car_bot)) else np.nan),
            "tstat_mean": float(diff_t(car_top, car_bot))
        }])
    ], ignore_index=True)
    summary.to_csv(os.path.join(cfg.outdir, "summary.csv"), index=False)

    # ---------------------------- Plots ----------------------------
    if cfg.plot and plt is not None:
        # CAAR by quantile
        fig1 = plt.figure(figsize=(10, 6)); ax1 = plt.gca()
        for label in sorted(ev["bucket"].dropna().unique()):
            c = aar_caar[(aar_caar["metric"] == "CAAR") & (aar_caar["bucket"] == label)].set_index("tau")["value"]
            ax1.plot(c.index, c.values, label=label)
        ax1.axvline(0, linestyle="--", color="k", alpha=0.5)
        ax1.set_title("CAAR by surprise quantile (abnormal)"); ax1.set_xlabel("Event day (τ)"); ax1.set_ylabel("CAAR")
        ax1.legend()
        plt.tight_layout(); fig1.savefig(os.path.join(cfg.outdir, "plots", "caar_by_quantile.png"), dpi=140); plt.close(fig1)

        # Long-short CAAR
        fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
        ax2.plot(ls.index, ls["LS_CAAR"].values, label=f"{top}-{bot} LS CAAR")
        ax2.axvline(0, linestyle="--", color="k", alpha=0.5)
        ax2.set_title(f"Long-short CAAR ({top} − {bot})"); ax2.set_xlabel("Event day (τ)"); ax2.set_ylabel("CAAR")
        ax2.legend(); plt.tight_layout()
        fig2.savefig(os.path.join(cfg.outdir, "plots", "long_short_caar.png"), dpi=140); plt.close(fig2)

        # Calendar-time LS equity curve (abnormal cum)
        if not cal_df.empty:
            fig3 = plt.figure(figsize=(9,5)); ax3 = plt.gca()
            ((1 + cal_df["LS_eqw"].fillna(0)).cumprod()).plot(ax=ax3)
            ax3.set_title("Calendar-time Long-Short (cum, abnormal)"); ax3.set_ylabel("Wealth (start=1)")
            plt.tight_layout(); fig3.savefig(os.path.join(cfg.outdir, "plots", "calendar_time_ls.png"), dpi=140); plt.close(fig3)

    # Console snapshot
    print("\n=== Summary (abnormal) ===")
    print(summary.round(4).to_string(index=False))
    print("\nFiles written to:", cfg.outdir)


# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Post-Earnings Announcement Drift (PEAD) analyzer")
    ap.add_argument("--events-file", required=True, help="CSV with earnings events (date,ticker[,surprise])")
    ap.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker for abnormal returns")
    ap.add_argument("--start", type=str, default=None, help="Start date for price download (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date for price download (YYYY-MM-DD)")
    ap.add_argument("--tpre", type=int, default=1, help="Days before event for window")
    ap.add_argument("--tpost", type=int, default=60, help="Days after event for window")
    ap.add_argument("--hold-start", type=int, default=2, help="Start of holding window (event days)")
    ap.add_argument("--hold-end", type=int, default=60, help="End of holding window (event days)")
    ap.add_argument("--quantiles", type=int, default=5, help="Number of surprise buckets")
    ap.add_argument("--sort-by", choices=["surprise","day0_abn"], default="surprise",
                    help="Rank by provided surprise or fallback proxy (day-0 abnormal)")
    ap.add_argument("--drop-zero-surprise", action="store_true",
                    help="If using surprise, drop events with zero/NaN surprise")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        events_file=args.events_file,
        benchmark=args.benchmark,
        start=args.start,
        end=args.end,
        tpre=int(max(0, args.tpre)),
        tpost=int(max(1, args.tpost)),
        hold_start=int(max(0, args.hold_start)),
        hold_end=int(max(1, args.hold_end)),
        quantiles=int(max(2, args.quantiles)),
        sort_by=args.sort_by,
        drop_zero_surprise=bool(args.drop_zero_surprise),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    # Validate holding window within event window
    if cfg.hold_start > cfg.tpost or cfg.hold_end > cfg.tpost:
        raise SystemExit("--hold-start/--hold-end must be within [-tpre..tpost]")

    run(cfg)


if __name__ == "__main__":
    main()