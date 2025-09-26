#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ipo_lockups.py
#
# IPO Lockup Expiration Analyzer
# ------------------------------
# What it does
# - Ingests a lockup-events CSV (one row per lockup tranche or per ticker)
# - Downloads prices with yfinance for each ticker + a benchmark (default SPY)
# - Builds event windows around each lockup date; computes AAR/CAAR
# - Benchmarks abnormal returns (stock minus benchmark)
# - Buckets events by *supply shock* (locked_shares / free_float_shares) and other columns
# - Produces long/short curves (Q5 − Q1 by supply shock)
# - Optional calendar-time portfolio of post-lockup returns (hold window)
# - Exports tidy CSVs and optional PNG plots
#
# Usage
# -----
# python ipo_lockups.py \
#   --events lockups.csv \
#   --benchmark SPY \
#   --tpre 5 --tpost 20 \
#   --hold-start 1 --hold-end 10 \
#   --quantiles 5 \
#   --bucket-by supply_shock \
#   --plot
#
# Input CSV: lockups.csv (columns; extra columns are kept if present)
# --------------------------------
# date,ticker,locked_shares,free_float_shares
# 2024-11-20,ABCD,50_000_000,80_000_000
# 2025-02-15,WXYZ,12_000_000,40_000_000
# Optional columns (if present they will be used/cleaned numerically):
# mktcap_usd,short_interest_shares,borrow_fee_pct,insider_pct,underwriter,notes
#
# Outputs
# -------
# outdir/
#   prices.csv
#   returns.csv
#   events_enriched.csv                (adds supply_shock etc.)
#   panel_raw.csv                      (event-relative returns)
#   panel_abn.csv                      (benchmark-adjusted)
#   aar_caar_by_bucket.csv
#   ls_caar.csv                        (long-short CAAR for top vs bottom bucket)
#   calendar_time_portfolios.csv
#   summary.csv                        (CAR stats per bucket & LS)
#   plots/*.png                        (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy yfinance matplotlib python-dateutil

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

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
    bucket_by: str      # supply_shock or any numeric column present in events
    drop_zero_bucket: bool
    plot: bool
    outdir: str


# ---------------------------- Helpers ----------------------------

def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"ipo_lockups_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def _to_num(x):
    # allow "50_000_000" and strings with commas
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).replace(",", "").replace("_", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"date","ticker"}
    if not need.issubset(df.columns):
        raise SystemExit("events file must contain at least: date, ticker")
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # sanitize numeric fields if present
    for c in ["locked_shares","free_float_shares","mktcap_usd","short_interest_shares","borrow_fee_pct","insider_pct"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_num)

    # compute supply shock (% of existing free float that becomes sale-eligible)
    if "locked_shares" in df.columns and "free_float_shares" in df.columns:
        df["supply_shock"] = df["locked_shares"] / df["free_float_shares"].replace(0, np.nan)
    else:
        df["supply_shock"] = np.nan

    # short interest as % float if available
    if "short_interest_shares" in df.columns and "free_float_shares" in df.columns:
        df["short_to_float"] = df["short_interest_shares"] / df["free_float_shares"].replace(0, np.nan)

    # borrow fee as decimal if given in percent
    if "borrow_fee_pct" in df.columns:
        df["borrow_fee_dec"] = df["borrow_fee_pct"] / (100.0 if df["borrow_fee_pct"].max() > 1.0 else 1.0)

    return df.dropna(subset=["date","ticker"]).reset_index(drop=True)


def yf_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, group_by="ticker", threads=True
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
        t = tickers[0]
        sub = data[["Open","High","Low","Close","Volume"]].copy()
        sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
        px = sub.sort_index()
    return px


def simple_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def abnormal_returns(stock: pd.Series, bench: pd.Series) -> pd.Series:
    a, b = stock.align(bench, join="inner")
    return a - b


def snap_to_trading_day(idx: pd.DatetimeIndex, d: pd.Timestamp) -> Optional[pd.Timestamp]:
    if d in idx:
        return d
    nxt = idx[idx >= d]
    return None if nxt.empty else nxt[0]


# ---------------------------- Event windows ----------------------------

def build_event_panel(returns: pd.DataFrame, events: pd.DataFrame,
                      tpre: int, tpost: int) -> Tuple[pd.DataFrame, Dict[int, Tuple[str, pd.Timestamp]]]:
    idx = returns.index
    panels = []
    meta = {}
    eid = 1
    for _, row in events.iterrows():
        t = row["ticker"]; d = row["date"]
        if t not in returns.columns:
            continue
        sd = snap_to_trading_day(idx, d)
        if sd is None:
            continue
        center = idx.get_indexer([sd])[0]
        start = max(0, center - tpre); end = min(len(idx)-1, center + tpost)
        rel_idx = np.arange(start - center, end - center + 1)
        win = returns[t].iloc[start:end+1].copy()
        win.index = rel_idx; win.name = eid
        panels.append(win)
        meta[eid] = (t, sd)
        eid += 1
    if not panels:
        raise RuntimeError("No event windows could be built. Check dates/tickers.")
    panel = pd.concat(panels, axis=1).reindex(np.arange(-tpre, tpost+1))
    return panel, meta


def make_aar_caar(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aar = panel.mean(axis=1).to_frame("AAR")
    caar = aar.cumsum().rename(columns={"AAR":"CAAR"})
    return aar, caar


def bucketize(series: pd.Series, q: int) -> pd.Series:
    s = series.copy()
    try:
        return pd.qcut(s, q, labels=[f"Q{i}" for i in range(1, q+1)])
    except Exception:
        ranks = s.rank(method="first", na_option="keep")
        return pd.qcut(ranks, q, labels=[f"Q{i}" for i in range(1, q+1)])


# ------------------------- Calendar-time portfolio -------------------------

def calendar_time_avg(panel: pd.DataFrame, hold_start: int, hold_end: int) -> pd.Series:
    """
    Equal-weight calendar-time average of event-day returns from τ=hold_start..hold_end.
    """
    rows = []
    for eid in panel.columns:
        for tau in range(hold_start, hold_end+1):
            if tau not in panel.index: continue
            r = panel.loc[tau, eid]
            if pd.isna(r): continue
            rows.append({"tau": tau, "ret": r})
    if not rows:
        return pd.Series(dtype=float)
    long = pd.DataFrame(rows)
    # for lack of true calendar alignment here, we simply average by τ to show typical path
    return long.groupby("tau")["ret"].mean()


# ---------------------------- Pipeline ----------------------------

def run(cfg: Config):
    # Load events and enrich
    ev = load_events(cfg.events_file)

    # Choose download window: pad pre/post around min/max event dates
    start_auto = (ev["date"].min() - pd.Timedelta(days=cfg.tpre + 15)).date().isoformat()
    start = cfg.start or start_auto
    end = cfg.end

    tickers = sorted(ev["ticker"].unique().tolist())
    universe = tickers + [cfg.benchmark]

    # Prices & returns
    px = yf_prices(universe, start, end)
    if px.empty:
        raise SystemExit("No price data downloaded. Check tickers, dates, benchmark.")
    # Save flat prices
    flat = pd.DataFrame(index=px.index)
    for t in universe:
        if t in px.columns.get_level_values(0):
            for f in ["Open","High","Low","Close","Volume"]:
                if (t,f) in px.columns:
                    flat[f"{t}_{f}"] = px[(t,f)]
    flat.to_csv(os.path.join(cfg.outdir, "prices.csv"))

    # Daily returns
    rets = {}
    for t in universe:
        if (t,"Close") in px.columns:
            rets[t] = simple_returns(px[(t,"Close")].astype(float))
    rets = pd.DataFrame(rets).dropna(how="all")
    rets.to_csv(os.path.join(cfg.outdir, "returns.csv"))

    # Abnormal returns vs benchmark
    if cfg.benchmark in rets.columns:
        abn = rets.sub(rets[cfg.benchmark], axis=0)
    else:
        abn = rets.copy()

    # Filter events to those with return coverage
    ev = ev[ev["ticker"].isin([c for c in rets.columns if c != cfg.benchmark])].copy()
    if ev.empty:
        raise SystemExit("No valid events after matching tickers to downloaded data.")

    # Build panels
    raw_panel, meta = build_event_panel(rets.drop(columns=[cfg.benchmark], errors="ignore"), ev, cfg.tpre, cfg.tpost)
    abn_panel, _ = build_event_panel(abn.drop(columns=[cfg.benchmark], errors="ignore"), ev, cfg.tpre, cfg.tpost)
    raw_panel.to_csv(os.path.join(cfg.outdir, "panel_raw.csv"))
    abn_panel.to_csv(os.path.join(cfg.outdir, "panel_abn.csv"))

    # Prepare bucket variable
    bucket_var_name = cfg.bucket_by
    if bucket_var_name not in ev.columns:
        if bucket_var_name.lower() == "supply_shock":
            raise SystemExit("Bucket variable 'supply_shock' not found. Provide locked_shares and free_float_shares.")
        else:
            raise SystemExit(f"Bucket variable '{bucket_var_name}' not in events file.")

    bucket_series = pd.to_numeric(ev[bucket_var_name], errors="coerce")
    buckets = bucketize(bucket_series, cfg.quantiles)
    ev["bucket"] = buckets

    if cfg.drop_zero_bucket:
        keep = ev[~(bucket_series == 0) & ~bucket_series.isna()].index
        ev = ev.loc[keep]
        abn_panel = abn_panel[ev.index]
        raw_panel = raw_panel[ev.index]

    # AAR/CAAR by bucket (abnormal)
    rows = []
    for lab in sorted(ev["bucket"].dropna().unique().tolist()):
        ids = ev[ev["bucket"] == lab].index.tolist()
        if not ids:
            continue
        aar, caar = make_aar_caar(abn_panel[ids])
        rows.append(pd.DataFrame({"tau": aar.index, "metric":"AAR","bucket":lab,"value":aar["AAR"].values}))
        rows.append(pd.DataFrame({"tau": caar.index, "metric":"CAAR","bucket":lab,"value":caar["CAAR"].values}))
    aar_caar = pd.concat(rows, ignore_index=True)
    aar_caar.to_csv(os.path.join(cfg.outdir, "aar_caar_by_bucket.csv"), index=False)

    # Long-short CAAR (top vs bottom bucket)
    def qlab(i): return f"Q{i}"
    top = qlab(cfg.quantiles); bot = qlab(1)
    c_top = aar_caar[(aar_caar["metric"]=="CAAR") & (aar_caar["bucket"]==top)].set_index("tau")["value"]
    c_bot = aar_caar[(aar_caar["metric"]=="CAAR") & (aar_caar["bucket"]==bot)].set_index("tau")["value"]
    ls = (c_top - c_bot).to_frame("LS_CAAR"); ls.index.name="tau"
    ls.to_csv(os.path.join(cfg.outdir, "ls_caar.csv"))

    # Calendar-time average (by τ) for top & bottom buckets
    cal_top = calendar_time_avg(abn_panel[ev[ev["bucket"]==top].index], cfg.hold_start, cfg.hold_end)
    cal_bot = calendar_time_avg(abn_panel[ev[ev["bucket"]==bot].index], cfg.hold_start, cfg.hold_end)
    cal_df = pd.DataFrame({"TOP_eqw": cal_top, "BOT_eqw": cal_bot})
    cal_df["LS_eqw"] = cal_df["TOP_eqw"] - cal_df["BOT_eqw"]
    cal_df.to_csv(os.path.join(cfg.outdir, "calendar_time_portfolios.csv"))

    # Summary (CAR over hold window)
    def car_stats(ids: List[int]) -> Dict[str, float]:
        sub = abn_panel[ids].loc[cfg.hold_start:cfg.hold_end]
        car = sub.sum(axis=0)
        return {
            "N_events": float(len(car)),
            "mean_CAR": float(car.mean()) if len(car) else np.nan,
            "median_CAR": float(car.median()) if len(car) else np.nan,
            "tstat_mean": float((car.mean() / (car.std(ddof=1)/np.sqrt(len(car)))) if len(car)>1 and car.std(ddof=1)>0 else np.nan)
        }

    summ = []
    for lab in sorted(ev["bucket"].dropna().unique().tolist()):
        ids = ev[ev["bucket"]==lab].index.tolist()
        s = car_stats(ids); s["bucket"]=lab; summ.append(s)

    car_top = abn_panel[ev[ev["bucket"]==top].index].loc[cfg.hold_start:cfg.hold_end].sum(axis=0)
    car_bot = abn_panel[ev[ev["bucket"]==bot].index].loc[cfg.hold_start:cfg.hold_end].sum(axis=0)
    def diff_t(a: pd.Series, b: pd.Series) -> float:
        if len(a)<2 or len(b)<2: return np.nan
        m1,m2 = a.mean(), b.mean(); s1,s2 = a.std(ddof=1), b.std(ddof=1)
        n1,n2 = len(a), len(b); se = np.sqrt((s1**2)/n1 + (s2**2)/n2)
        return float((m1-m2)/se) if se and not np.isnan(se) and se!=0 else np.nan
    ls_row = {"bucket": f"{top}-{bot}", "N_events": float(len(car_top)+len(car_bot)),
              "mean_CAR": float(car_top.mean()-car_bot.mean()),
              "median_CAR": float((car_top - car_bot).median() if len(car_top) and len(car_bot) else np.nan),
              "tstat_mean": diff_t(car_top, car_bot)}
    summary = pd.DataFrame(summ + [ls_row])
    summary.to_csv(os.path.join(cfg.outdir, "summary.csv"), index=False)

    # Save enriched events snapshot
    ev.to_csv(os.path.join(cfg.outdir, "events_enriched.csv"), index=False)

    # ---------------------------- Plots ----------------------------
    if cfg.plot and plt is not None:
        # CAAR by bucket
        fig1 = plt.figure(figsize=(10,6)); ax1 = plt.gca()
        for lab in sorted(ev["bucket"].dropna().unique()):
            c = aar_caar[(aar_caar["metric"]=="CAAR") & (aar_caar["bucket"]==lab)].set_index("tau")["value"]
            ax1.plot(c.index, c.values, label=str(lab))
        ax1.axvline(0, linestyle="--", color="k", alpha=0.5)
        ax1.set_title(f"Lockup CAAR by {cfg.bucket_by} quantile (abnormal)"); ax1.set_xlabel("Event day (τ)"); ax1.set_ylabel("CAAR")
        ax1.legend()
        plt.tight_layout(); fig1.savefig(os.path.join(cfg.outdir, "plots", "caar_by_bucket.png"), dpi=140); plt.close(fig1)

        # Long-short CAAR
        fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
        ax2.plot(ls.index, ls["LS_CAAR"].values, label=f"{top}-{bot} LS CAAR")
        ax2.axvline(0, linestyle="--", color="k", alpha=0.5)
        ax2.set_title("Long-short CAAR (top − bottom bucket)"); ax2.set_xlabel("Event day (τ)"); ax2.set_ylabel("CAAR")
        ax2.legend(); plt.tight_layout(); fig2.savefig(os.path.join(cfg.outdir, "plots", "ls_caar.png"), dpi=140); plt.close(fig2)

        # Average post-event (τ) returns for top vs bottom
        if not cal_df.empty:
            fig3 = plt.figure(figsize=(9,5)); ax3 = plt.gca()
            cal_df.plot(ax=ax3)
            ax3.set_title(f"Average post-lockup returns by τ (hold {cfg.hold_start}..{cfg.hold_end})")
            ax3.set_xlabel("Event day (τ)"); ax3.set_ylabel("Avg daily return")
            plt.tight_layout(); fig3.savefig(os.path.join(cfg.outdir, "plots", "calendar_time_avg.png"), dpi=140); plt.close(fig3)

    # Console snapshot
    print("\n=== Summary over hold window (abnormal returns) ===")
    print(summary.round(4).to_string(index=False))
    print("\nFiles written to:", cfg.outdir)


# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="IPO lockup expiration event-study & buckets")
    ap.add_argument("--events", required=True, help="CSV with lockup events (date,ticker[,locked_shares,free_float_shares,...])")
    ap.add_argument("--benchmark", default="SPY", help="Benchmark ETF for abnormal returns")
    ap.add_argument("--start", default=None, help="Manual start date for price download (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Manual end date for price download (YYYY-MM-DD)")
    ap.add_argument("--tpre", type=int, default=5, help="Days before event for window")
    ap.add_argument("--tpost", type=int, default=20, help="Days after event for window")
    ap.add_argument("--hold-start", type=int, default=1, help="Start of holding window (event days)")
    ap.add_argument("--hold-end", type=int, default=10, help="End of holding window (event days)")
    ap.add_argument("--quantiles", type=int, default=5, help="Number of buckets")
    ap.add_argument("--bucket-by", type=str, default="supply_shock",
                    help="Column to bucket by (default supply_shock = locked/free_float)")
    ap.add_argument("--drop-zero-bucket", action="store_true",
                    help="Drop events with zero/NaN bucket variable")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        events_file=args.events,
        benchmark=args.benchmark,
        start=args.start,
        end=args.end,
        tpre=int(max(0, args.tpre)),
        tpost=int(max(1, args.tpost)),
        hold_start=int(max(0, args.hold_start)),
        hold_end=int(max(1, args.hold_end)),
        quantiles=int(max(2, args.quantiles)),
        bucket_by=args.bucket_by,
        drop_zero_bucket=bool(args.drop_zero_bucket),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    # Validate holding window within event window
    if cfg.hold_start > cfg.tpost or cfg.hold_end > cfg.tpost:
        raise SystemExit("--hold-start/--hold-end must be within [-tpre..tpost]")

    run(cfg)


if __name__ == "__main__":
    main()