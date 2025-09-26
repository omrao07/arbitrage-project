#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nfp_surprise.py
#
# Nonfarm Payrolls (NFP) surprise analyzer & market reaction study
# ---------------------------------------------------------------
# - Ingests an NFP events CSV (actual vs consensus; optional UR, AHE)
# - Computes surprise metrics: level, %, z-score (rolling), “beat/miss” dummies
# - Downloads market prices with yfinance for a user-defined asset set
# - Event-study on daily returns around NFP (AAR/CAAR by surprise buckets)
# - Cross-asset day-0 sensitivity: ΔP (or return) ~ surprise (OLS)
# - Optional intraday (30m/60m) study for the most recent ~60 days of events
# - Exports tidy CSVs and optional PNG plots
#
# Usage
# -----
# python nfp_surprise.py \
#   --events nfp.csv \
#   --assets SPY,QQQ,IEF,TLT,UUP,GLD,HYG,USDJPY=X \
#   --benchmark SPY \
#   --tpre 5 --tpost 10 \
#   --quantiles 5 \
#   --intraday --interval 60m \
#   --plot
#
# nfp.csv (minimum columns)
# -------------------------
# date,actual,consensus
# 2024-11-01,150,180
# 2024-12-06,210,175
# Optional extra columns (kept if present): previous, revised_prev, unemployment_rate, ahe_mom, private_payrolls
#
# Outputs
# -------
# outdir/
#   events_clean.csv
#   prices_daily.csv
#   returns_daily.csv
#   panel_daily.csv               (event-relative daily returns per asset)
#   aar_caar_by_bucket.csv
#   day0_sensitivity.csv          (OLS β for each asset: return_day0 ~ surprise)
#   intraday_*.*                  (if --intraday)
#   plots/*.png                   (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy yfinance matplotlib statsmodels python-dateutil

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

import statsmodels.api as sm
from dateutil import parser as dtp


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    events_file: str
    assets: List[str]
    benchmark: Optional[str]
    start: Optional[str]
    end: Optional[str]
    tpre: int
    tpost: int
    quantiles: int
    bucket_by: str       # 'z' | 'pct' | 'level' | 'signed'
    intraday: bool
    interval: str        # '60m' | '30m' (YF constraints apply; ~60 days lookback)
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"nfp_surprise_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("_",""))
    except Exception:
        return np.nan


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"date","actual","consensus"}
    if not need.issubset(df.columns):
        raise SystemExit("events file must include: date, actual, consensus")
    df["date"] = pd.to_datetime(df["date"])
    for c in [c for c in df.columns if c not in ["date"]]:
        df[c] = df[c].apply(_num)
    # Surprise metrics
    df["surprise_level"] = df["actual"] - df["consensus"]
    df["surprise_pct"] = df["surprise_level"] / df["consensus"].replace(0, np.nan)
    # rolling z-score of level surprise (12 obs window)
    df = df.sort_values("date").reset_index(drop=True)
    r = df["surprise_level"].rolling(12)
    df["surprise_z"] = (df["surprise_level"] - r.mean()) / r.std(ddof=1)
    # signed bucket +1/-1/0
    df["surprise_signed"] = np.sign(df["surprise_level"]).astype(int)
    # Keep a clean copy
    return df


def parse_assets(s: Optional[str]) -> List[str]:
    if not s: return []
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def yf_prices(tickers: List[str], start: str, end: Optional[str], interval: str = "1d") -> pd.DataFrame:
    data = yf.download(tickers=tickers, start=start, end=end, interval=interval,
                       auto_adjust=True, progress=False, group_by="ticker", threads=True)
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                sub = data[t]
                if "Close" in sub.columns:
                    s = sub[["Close"]].rename(columns={"Close": t})
                    frames.append(s)
    else:
        # single ticker path
        t = tickers[0]
        frames.append(data[["Close"]].rename(columns={"Close": t}))
    px = pd.concat(frames, axis=1).sort_index()
    # Drop all-NaN columns
    px = px.dropna(axis=1, how="all")
    return px


def simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().replace([np.inf, -np.inf], np.nan)


def snap_to_trading_day(idx: pd.DatetimeIndex, d: pd.Timestamp) -> Optional[pd.Timestamp]:
    # For daily data: use next trading day on/after event date (events are Fridays)
    if d in idx:
        return d
    nxt = idx[idx >= d]
    return None if nxt.empty else nxt[0]


def build_event_panel(returns: pd.DataFrame, events: pd.DataFrame, tpre: int, tpost: int
) -> Tuple[Dict[str, pd.DataFrame], Dict[int, pd.Timestamp]]:
    """
    Build event-relative return panels per asset.
    Returns:
      panels: {asset -> DataFrame indexed by τ=-tpre..tpost, columns=event_id}
      e_dates: {event_id -> snapped trading date}
    """
    idx = returns.index
    assets = [c for c in returns.columns]
    panels = {a: [] for a in assets}
    e_dates = {}
    eid = 1
    for _, row in events.iterrows():
        d = row["date"]
        sd = snap_to_trading_day(idx, d)
        if sd is None:  # outside price history
            continue
        center = idx.get_indexer([sd])[0]
        start = max(0, center - tpre)
        end = min(len(idx)-1, center + tpost)
        rel_idx = np.arange(start - center, end - center + 1)
        for a in assets:
            if a not in returns.columns: 
                continue
            win = returns[a].iloc[start:end+1].copy()
            win.index = rel_idx
            win.name = eid
            panels[a].append(win)
        e_dates[eid] = sd
        eid += 1
    # Cat to wide per asset
    for a in list(panels.keys()):
        if panels[a]:
            panels[a] = pd.concat(panels[a], axis=1).reindex(np.arange(-tpre, tpost+1))
        else:
            panels[a] = pd.DataFrame()
    return panels, e_dates


def bucketize(ev: pd.DataFrame, by: str, q: int) -> pd.Series:
    if by == "z":
        series = ev["surprise_z"]
    elif by == "pct":
        series = ev["surprise_pct"]
    elif by == "level":
        series = ev["surprise_level"]
    elif by == "signed":
        # three buckets: miss / inline / beat
        s = ev["surprise_signed"]
        return pd.Categorical(np.where(s < 0, "MISS", np.where(s > 0, "BEAT", "INLINE")),
                              categories=["MISS","INLINE","BEAT"], ordered=True)
    else:
        series = ev["surprise_z"]
    try:
        return pd.qcut(series, q, labels=[f"Q{i}" for i in range(1, q+1)])
    except Exception:
        ranks = series.rank(method="first", na_option="keep")
        return pd.qcut(ranks, q, labels=[f"Q{i}" for i in range(1, q+1)])


def aar_caar(panel: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    aar = panel.mean(axis=1)
    caar = aar.cumsum()
    return aar, caar


def ols_day0_sensitivity(panels: Dict[str, pd.DataFrame], events: pd.DataFrame, var: str = "surprise_z") -> pd.DataFrame:
    """
    Regress day-0 return for each asset on chosen surprise variable.
    """
    rows = []
    for a, P in panels.items():
        if isinstance(P, pd.DataFrame) and not P.empty and 0 in P.index:
            y = P.loc[0].dropna()  # day-0 cross-section over events
            X = events.set_index(pd.Index(range(1, len(events)+1))).loc[y.index, [var]].rename(columns={var: "s"})
            X = sm.add_constant(X)
            try:
                res = sm.OLS(y.values, X.values, missing="drop").fit()
                rows.append({"asset": a, "n": int(res.nobs), "beta_surprise": float(res.params[1]),
                             "t_surprise": float(res.tvalues[1]), "alpha": float(res.params[0]),
                             "r2": float(res.rsquared)})
            except Exception:
                continue
    return pd.DataFrame(rows).sort_values("r2", ascending=False)


# ----------------------------- Intraday (optional) -----------------------------

def intraday_event_windows(assets: List[str], events: pd.DataFrame, interval: str = "60m", hours_pre: int = 6, hours_post: int = 24
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    For the last ~60 days of events (YF limitation), pull intraday bars and compute
    cumulative return from 2 hours before 8:30 ET to +X hours.
    NOTE: Simplicity: we align by wall-clock; time-zone effects may vary by asset.
    """
    if len(events) == 0:
        return pd.DataFrame(), {}
    # Filter events to last 55 days to increase chance YF returns intraday
    cutoff = pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=55)
    ev_recent = events[events["date"] >= cutoff.tz_localize(None)].copy()
    if ev_recent.empty:
        return pd.DataFrame(), {}
    # Download intraday for each asset around min/max date window
    start = (ev_recent["date"].min() - pd.Timedelta(days=3)).date().isoformat()
    end = (ev_recent["date"].max() + pd.Timedelta(days=3)).date().isoformat()
    px = yf_prices(assets, start, end, interval=interval)
    if px.empty:
        return pd.DataFrame(), {}

    # Build per-event curves: cumulative return from -pre .. +post hours
    results = {}
    for a in px.columns:
        rows = []
        for _, r in ev_recent.iterrows():
            d0 = pd.Timestamp(r["date"]).replace(hour=13, minute=30)  # 8:30 ET ≈ 13:30 UTC (no DST handling here)
            start_t = d0 - pd.Timedelta(hours=hours_pre)
            end_t = d0 + pd.Timedelta(hours=hours_post)
            sub = px.loc[(px.index >= start_t) & (px.index <= end_t), a].dropna()
            if sub.empty: 
                continue
            ret = sub.pct_change().fillna(0.0)
            cum = (1 + ret).cumprod() - 1.0
            # Map to relative hour bins
            rel_hours = ((cum.index - d0).total_seconds()/3600.0).round(1)
            tmp = pd.DataFrame({"rel_hour": rel_hours, "cumret": cum.values})
            tmp["event_date"] = r["date"]
            rows.append(tmp)
        if rows:
            results[a] = pd.concat(rows, ignore_index=True)

    # Aggregate across events (median & mean)
    agg_rows = []
    for a, df in results.items():
        g = df.groupby("rel_hour")["cumret"].agg(["mean","median","count"]).reset_index()
        g["asset"] = a
        agg_rows.append(g)
    agg = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
    return agg, results


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="NFP surprise vs market reaction (daily + optional intraday)")
    ap.add_argument("--events", required=True, help="CSV of NFP events (date,actual,consensus,...)")
    ap.add_argument("--assets", type=str, default="SPY,QQQ,IEF,TLT,UUP,GLD,HYG,USDJPY=X",
                    help="Comma-separated tickers to study")
    ap.add_argument("--benchmark", type=str, default="SPY", help="Optional benchmark for abnormal returns")
    ap.add_argument("--start", type=str, default=None, help="Start date for price history (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date")
    ap.add_argument("--tpre", type=int, default=5, help="Days before event")
    ap.add_argument("--tpost", type=int, default=10, help="Days after event")
    ap.add_argument("--quantiles", type=int, default=5, help="Number of surprise buckets (ignored for signed)")
    ap.add_argument("--bucket-by", choices=["z","pct","level","signed"], default="z",
                    help="How to bucket events")
    ap.add_argument("--intraday", action="store_true", help="Enable intraday (last ~60 days only)")
    ap.add_argument("--interval", type=str, choices=["60m","30m"], default="60m",
                    help="Intraday bar size")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        events_file=args.events,
        assets=parse_assets(args.assets),
        benchmark=(args.benchmark.strip().upper() if args.benchmark else None),
        start=args.start,
        end=args.end,
        tpre=int(max(0, args.tpre)),
        tpost=int(max(1, args.tpost)),
        quantiles=int(max(2, args.quantiles)),
        bucket_by=args.bucket_by,
        intraday=bool(args.intraday),
        interval=args.interval,
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Writing to: {cfg.outdir}")

    # ---- Load events & compute surprises ----
    ev = load_events(cfg.events_file)
    ev.to_csv(os.path.join(cfg.outdir, "events_clean.csv"), index=False)

    # ---- Download daily prices ----
    universe = cfg.assets.copy()
    if cfg.benchmark and cfg.benchmark not in universe:
        universe.append(cfg.benchmark)
    # pick start if not provided: min event date - padding
    start_auto = (ev["date"].min() - pd.Timedelta(days=cfg.tpre + 15)).date().isoformat()
    start = cfg.start or start_auto
    px = yf_prices(universe, start, cfg.end, interval="1d")
    if px.empty:
        raise SystemExit("No price data downloaded. Check tickers/dates.")
    px.to_csv(os.path.join(cfg.outdir, "prices_daily.csv"))

    rets = simple_returns(px).dropna(how="all")
    rets.to_csv(os.path.join(cfg.outdir, "returns_daily.csv"))

    # ---- Build event panel (daily) ----
    panels, e_dates = build_event_panel(rets, ev, cfg.tpre, cfg.tpost)
    # Save combined panel for convenience (stack assets)
    stacked = []
    for a, P in panels.items():
        if isinstance(P, pd.DataFrame) and not P.empty:
            Q = P.copy(); Q["asset"] = a; Q["tau"] = Q.index
            stacked.append(Q.set_index(["asset","tau"]))
    if stacked:
        panel_all = pd.concat(stacked, axis=0)
        panel_all.to_csv(os.path.join(cfg.outdir, "panel_daily.csv"))

    # ---- Bucket events by chosen metric ----
    ev = ev.copy()
    ev["bucket"] = bucketize(ev, cfg.bucket_by, cfg.quantiles)

    # ---- AAR/CAAR by bucket (per asset) ----
    rows = []
    for a, P in panels.items():
        if not isinstance(P, pd.DataFrame) or P.empty:
            continue
        # Align to events that survived panel construction
        valid_eids = [c for c in P.columns if c in e_dates]
        if not valid_eids:
            continue
        ev_sub = ev.iloc[:len(valid_eids)].copy()
        ev_sub.index = valid_eids  # event_id alignment
        for lab in sorted([x for x in ev_sub["bucket"].dropna().unique()], key=str):
            ids = ev_sub[ev_sub["bucket"] == lab].index.tolist()
            if not ids:
                continue
            aar = P[ids].mean(axis=1)
            caar = aar.cumsum()
            rows.append(pd.DataFrame({
                "asset": a, "bucket": str(lab), "tau": aar.index,
                "AAR": aar.values, "CAAR": caar.values
            }))
    if rows:
        aar_caar = pd.concat(rows, ignore_index=True)
        aar_caar.to_csv(os.path.join(cfg.outdir, "aar_caar_by_bucket.csv"), index=False)
    else:
        aar_caar = pd.DataFrame()

    # ---- Day-0 sensitivity: return ~ surprise_z (or chosen var) ----
    sens = ols_day0_sensitivity(panels, ev, var={"z":"surprise_z","pct":"surprise_pct","level":"surprise_level","signed":"surprise_signed"}[cfg.bucket_by])
    if not sens.empty:
        sens.to_csv(os.path.join(cfg.outdir, "day0_sensitivity.csv"), index=False)

    # ---- Intraday (optional, last ~60 days of events) ----
    if cfg.intraday:
        agg, per_asset = intraday_event_windows(cfg.assets, ev, interval=cfg.interval)
        if not agg.empty:
            agg.to_csv(os.path.join(cfg.outdir, f"intraday_{cfg.interval}_aggregate.csv"), index=False)
        for a, df in per_asset.items():
            df.to_csv(os.path.join(cfg.outdir, f"intraday_{a}_{cfg.interval}.csv"), index=False)

    # ---- Plots ----
    if cfg.plot and plt is not None:
        # CAAR by bucket for top assets
        if not aar_caar.empty:
            for a in aar_caar["asset"].unique()[:8]:
                sub = aar_caar[aar_caar["asset"] == a]
                fig = plt.figure(figsize=(9,5)); ax = plt.gca()
                for lab in sorted(sub["bucket"].unique(), key=str):
                    c = sub[sub["bucket"] == lab].set_index("tau")["CAAR"]
                    ax.plot(c.index, c.values, label=str(lab))
                ax.axvline(0, linestyle="--", color="k", alpha=0.5)
                ax.set_title(f"CAAR around NFP for {a} (bucket={cfg.bucket_by})")
                ax.set_xlabel("Event day (τ)"); ax.set_ylabel("CAAR")
                ax.legend()
                plt.tight_layout(); fig.savefig(os.path.join(cfg.outdir, "plots", f"caar_{a}.png"), dpi=140); plt.close(fig)

        # Day-0 scatter vs surprise for first few assets
        if not sens.empty:
            var = {"z":"surprise_z","pct":"surprise_pct","level":"surprise_level","signed":"surprise_signed"}[cfg.bucket_by]
            # Take up to 6 assets
            for a in sens["asset"].head(6):
                P = panels.get(a)
                if P is None or P.empty or 0 not in P.index:
                    continue
                y = P.loc[0].dropna()
                X = ev.set_index(pd.Index(range(1, len(ev)+1))).loc[y.index, var]
                fig2 = plt.figure(figsize=(6,5)); ax2 = plt.gca()
                ax2.scatter(X, y, s=16, alpha=0.7)
                ax2.set_title(f"{a}: day-0 return vs {var}")
                ax2.set_xlabel(var); ax2.set_ylabel("day-0 return")
                # Fit line
                try:
                    b = np.polyfit(X.values.astype(float), y.values.astype(float), 1)
                    xs = np.linspace(float(X.min()), float(X.max()), 50)
                    ax2.plot(xs, b[0]*xs + b[1])
                except Exception:
                    pass
                plt.tight_layout(); fig2.savefig(os.path.join(cfg.outdir, "plots", f"scatter_{a}.png"), dpi=140); plt.close(fig2)

        # Intraday aggregate charts
        if cfg.intraday:
            agg_path = os.path.join(cfg.outdir, f"intraday_{cfg.interval}_aggregate.csv")
            if os.path.exists(agg_path):
                agg = pd.read_csv(agg_path)
                for a in agg["asset"].unique()[:6]:
                    sub = agg[agg["asset"] == a]
                    fig3 = plt.figure(figsize=(8,4)); ax3 = plt.gca()
                    ax3.plot(sub["rel_hour"], sub["mean"], label="mean")
                    ax3.plot(sub["rel_hour"], sub["median"], label="median", alpha=0.7)
                    ax3.axvline(0, linestyle="--", color="k", alpha=0.6)
                    ax3.set_title(f"Intraday cum. return around NFP ({a}, {cfg.interval})")
                    ax3.set_xlabel("Hours from release (≈ 8:30 ET = 0)"); ax3.set_ylabel("Cum. return")
                    ax3.legend(); plt.tight_layout()
                    fig3.savefig(os.path.join(cfg.outdir, "plots", f"intraday_{a}.png"), dpi=140); plt.close(fig3)

        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # ---- Console snapshot ----
    print("\n=== Events (last 6) ===")
    cols = [c for c in ["date","actual","consensus","surprise_level","surprise_pct","surprise_z"] if c in ev.columns]
    print(ev.sort_values("date").tail(6)[cols].round(3).to_string(index=False))

    if not sens.empty:
        print("\n=== Day-0 sensitivity (return ~ surprise) ===")
        print(sens.round(4).to_string(index=False))

    print("\nFiles written to:", cfg.outdir)


if __name__ == "__main__":
    main()