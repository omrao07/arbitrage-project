#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# rail_vs_trucking.py
#
# Rail vs Trucking — equity baskets & freight proxies
# ---------------------------------------------------
# What this does
# - Builds equal-weight equity baskets for US rails and truckers (or your tickers)
# - Computes returns, relative performance (Rail − Truck), and z-scores
# - Rolling correlation/beta to a benchmark (IYT by default)
# - Optional merge with your freight fundamentals CSV (e.g., rail carloads, intermodal, ATA truck tonnage)
# - Simple lead/lag cross-correlations between equity spread and fundamentals
# - Exports tidy CSVs + optional plots
#
# Examples
# --------
# python rail_vs_trucking.py --start 2015-01-01 --plot
#
# python rail_vs_trucking.py \
#   --rails UNP,CSX,NSC,CP,CNI \
#   --trucks ODFL,JBHT,KNX,WERN,CHRW \
#   --benchmark IYT \
#   --freight freight.csv --plot
#
# freight.csv schema (flexible)
#   date, rail_carloads, intermodal_units, ata_truck_tonnage, cass_shipments, diesel_price
# (Any subset is fine; column names are case-insensitive. We'll auto-detect and use what’s present.)
#
# Dependencies
# ------------
# pip install pandas numpy yfinance matplotlib python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

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

from dateutil import parser as dtp


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    rails: List[str]
    trucks: List[str]
    benchmark: Optional[str]
    start: str
    end: Optional[str]
    freight_file: Optional[str]
    rollwin: int
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "rail_vs_trucking_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def parse_list(s: Optional[str]) -> List[str]:
    if not s: return []
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().replace([np.inf, -np.inf], np.nan)

def ann_stats(returns: pd.Series, periods_per_year: int = 252) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    mean = r.mean() * periods_per_year
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = mean / (vol if vol and not np.isnan(vol) and vol != 0 else np.nan)
    cum = (1 + r).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    return {"ann_return": float(mean), "ann_vol": float(vol), "sharpe": float(sharpe), "max_drawdown": mdd}

def rolling_beta(y: pd.Series, x: pd.Series, win: int) -> pd.Series:
    join = pd.concat([y, x], axis=1).dropna()
    if join.empty: return pd.Series(index=y.index, dtype=float)
    cov = join.iloc[:,0].rolling(win).cov(join.iloc[:,1])
    var = join.iloc[:,1].rolling(win).var()
    beta = cov / var.replace(0, np.nan)
    beta.name = f"beta_{y.name}_to_{x.name}"
    return beta.reindex(y.index)

def xcorr(a: pd.Series, b: pd.Series, max_lag: int = 12) -> pd.DataFrame:
    a, b = a.align(b, join="inner")
    out = []
    for k in range(-max_lag, max_lag+1):
        if k > 0:
            s1, s2 = a.iloc[:-k], b.iloc[k:]
        elif k < 0:
            s1, s2 = a.iloc[-k:], b.iloc[:k]
        else:
            s1, s2 = a, b
        out.append({"lag": k, "corr": s1.corr(s2)})
    df = pd.DataFrame(out)
    if not df.empty and df["corr"].notna().any():
        best = df.iloc[df["corr"].abs().idxmax()]
        df.attrs["best_lag"] = int(best["lag"])
        df.attrs["best_corr"] = float(best["corr"])
    return df


# ----------------------------- Data -----------------------------

def yf_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    data = yf.download(tickers=tickers, start=start, end=end, interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker", threads=True)
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                sub = data[t]
                if "Close" in sub.columns:
                    frames.append(sub[["Close"]].rename(columns={"Close": t}))
    else:
        # single ticker
        t = tickers[0]
        frames.append(data[["Close"]].rename(columns={"Close": t}))
    px = pd.concat(frames, axis=1).dropna(how="all").sort_index()
    return px

def load_freight_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise SystemExit("freight CSV must include a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    # sanitize numeric
    for c in df.columns:
        if c == "date": continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # resample monthly (last obs)
    monthly = df.set_index("date").resample("M").last()
    # build YoY growth where meaningful
    for c in monthly.columns:
        monthly[c + "_yoy"] = monthly[c].pct_change(12)
    return monthly


# ----------------------------- Main analysis -----------------------------

def run(cfg: Config):
    tickers = sorted(set(cfg.rails + cfg.trucks + ([cfg.benchmark] if cfg.benchmark else [])))
    if not cfg.rails or not cfg.trucks:
        raise SystemExit("Please provide at least one rail ticker and one trucking ticker.")
    px = yf_prices(tickers, cfg.start, cfg.end)
    if px.empty:
        raise SystemExit("No price data downloaded. Check tickers/dates.")
    os.makedirs(cfg.outdir, exist_ok=True)
    px.to_csv(os.path.join(cfg.outdir, "prices_close.csv"))

    # Returns
    rets = pct_change(px).dropna(how="all")
    rets.to_csv(os.path.join(cfg.outdir, "returns_daily.csv"))

    # Equal-weight baskets
    rails_valid = [t for t in cfg.rails if t in rets.columns]
    trucks_valid = [t for t in cfg.trucks if t in rets.columns]
    if not rails_valid or not trucks_valid:
        raise SystemExit("After download, at least one rail and one truck ticker must have returns.")

    baskets = pd.DataFrame({
        "RAIL_EQW": rets[rails_valid].mean(axis=1, skipna=True),
        "TRUCK_EQW": rets[trucks_valid].mean(axis=1, skipna=True),
    }).dropna(how="all")
    if cfg.benchmark and cfg.benchmark in rets.columns:
        baskets[cfg.benchmark] = rets[cfg.benchmark]
    baskets.to_csv(os.path.join(cfg.outdir, "basket_returns_eqw.csv"))

    # Relative performance & z-score (use 90d window)
    spread = baskets["RAIL_EQW"] - baskets["TRUCK_EQW"]
    spread.name = "SPREAD_RminusT"
    roll = spread.rolling(cfg.rollwin)
    z = (spread - roll.mean()) / roll.std(ddof=1)
    z.name = "SPREAD_Z"
    rel_df = pd.concat([spread, z], axis=1)
    rel_df.to_csv(os.path.join(cfg.outdir, "relative_spread.csv"))

    # Rolling correlation/beta to benchmark
    roll_df = pd.DataFrame(index=baskets.index)
    if cfg.benchmark and cfg.benchmark in baskets.columns:
        for col in ["RAIL_EQW", "TRUCK_EQW", "SPREAD_RminusT"]:
            series = spread if col == "SPREAD_RminusT" else baskets[col]
            roll_df[f"corr_{col}_to_{cfg.benchmark}"] = series.rolling(cfg.rollwin).corr(baskets[cfg.benchmark])
            roll_df[f"beta_{col}_to_{cfg.benchmark}"] = rolling_beta(series, baskets[cfg.benchmark], cfg.rollwin)
        roll_df.to_csv(os.path.join(cfg.outdir, "rolling_corr_beta.csv"))

    # Freight fundamentals (optional)
    if cfg.freight_file:
        freight = load_freight_csv(cfg.freight_file)
        freight.to_csv(os.path.join(cfg.outdir, "freight_monthly.csv"))
        # Convert baskets to monthly to compare with fundamentals
        m_baskets = (1 + baskets).resample("M").apply(lambda x: (x+1e-12).prod() - 1)  # monthly total return approx
        m_spread = m_baskets["RAIL_EQW"] - m_baskets["TRUCK_EQW"]
        m_spread.name = "SPREAD_M_RminusT"
        merged = freight.join(m_spread, how="left")
        merged.to_csv(os.path.join(cfg.outdir, "merged_monthly.csv"))

        # Cross-correlations: equity spread vs fundamentals YoY
        xcorr_rows = []
        fcols = [c for c in freight.columns if c.endswith("_yoy")]
        for c in fcols:
            X = xcorr(merged["SPREAD_M_RminusT"], merged[c], max_lag=12)
            if X.empty: continue
            X["pair"] = f"spread_vs_{c}"
            xcorr_rows.append(X)
        if xcorr_rows:
            xc = pd.concat(xcorr_rows, ignore_index=True)
            xc.to_csv(os.path.join(cfg.outdir, "xcorr_table.csv"), index=False)

    # Headline stats
    stats_rows = []
    for label, s in [("RAIL_EQW", baskets["RAIL_EQW"]),
                     ("TRUCK_EQW", baskets["TRUCK_EQW"]),
                     ("SPREAD_RminusT", spread)]:
        st = ann_stats(s)
        st["series"] = label
        stats_rows.append(st)
    stats = pd.DataFrame(stats_rows)[["series","ann_return","ann_vol","sharpe","max_drawdown"]]
    stats.to_csv(os.path.join(cfg.outdir, "stats.csv"), index=False)

    # ----------------------------- Plots -----------------------------
    if cfg.plot and plt is not None:
        # Equity curves (normalized)
        fig1 = plt.figure(figsize=(10,5)); ax1 = plt.gca()
        wealth = (1 + baskets[["RAIL_EQW","TRUCK_EQW"]]).fillna(0).add(1).cumprod()
        wealth.columns = ["Rail (EW)", "Trucking (EW)"]
        wealth.plot(ax=ax1)
        ax1.set_title("Rail vs Trucking — Equity Basket Wealth (start=1)")
        ax1.set_ylabel("Index")
        plt.tight_layout(); fig1.savefig(os.path.join(cfg.outdir, "plots", "wealth.png"), dpi=140); plt.close(fig1)

        # Spread & z-score
        fig2 = plt.figure(figsize=(10,5)); ax2 = plt.gca()
        spread.cumsum().plot(ax=ax2, label="Cum. (R−T)")
        ax2.set_title("Relative Performance: Rail − Truck (cum. return)"); ax2.legend()
        plt.tight_layout(); fig2.savefig(os.path.join(cfg.outdir, "plots", "spread_cum.png"), dpi=140); plt.close(fig2)

        fig3 = plt.figure(figsize=(10,4)); ax3 = plt.gca()
        z.plot(ax=ax3)
        ax3.axhline(0, linestyle="--", alpha=0.6); ax3.axhline(2, linestyle="--", alpha=0.4); ax3.axhline(-2, linestyle="--", alpha=0.4)
        ax3.set_title(f"Spread z-score (rolling {cfg.rollwin}d)")
        plt.tight_layout(); fig3.savefig(os.path.join(cfg.outdir, "plots", "spread_z.png"), dpi=140); plt.close(fig3)

        # Rolling corr/beta
        if not roll_df.empty:
            for kind in ["corr","beta"]:
                fig = plt.figure(figsize=(10,5)); ax = plt.gca()
                cols = [c for c in roll_df.columns if c.startswith(kind)]
                roll_df[cols].plot(ax=ax)
                ax.axhline(0, linestyle="--", alpha=0.5)
                ax.set_title(f"Rolling {cfg.rollwin}d {kind} to {cfg.benchmark}")
                plt.tight_layout(); fig.savefig(os.path.join(cfg.outdir, "plots", f"rolling_{kind}.png"), dpi=140); plt.close(fig)

        # Freight merges
        if cfg.freight_file and os.path.exists(os.path.join(cfg.outdir, "merged_monthly.csv")):
            merged = pd.read_csv(os.path.join(cfg.outdir, "merged_monthly.csv"))
            merged["date"] = pd.to_datetime(merged["Unnamed: 0"] if "Unnamed: 0" in merged.columns else merged.get("date", pd.NaT))
            merged = merged.set_index("date")
            # pick a couple of common names if present
            for col in ["rail_carloads_yoy","intermodal_units_yoy","ata_truck_tonnage_yoy","cass_shipments_yoy"]:
                if col in merged.columns:
                    figm = plt.figure(figsize=(10,5)); axm = plt.gca()
                    merged[[col,"SPREAD_M_RminusT"]].dropna().plot(ax=axm)
                    axm.axhline(0, linestyle="--", alpha=0.5)
                    axm.set_title(f"Monthly: {col} vs Rail−Truck equity spread")
                    plt.tight_layout(); figm.savefig(os.path.join(cfg.outdir, "plots", f"monthly_{col}_vs_spread.png"), dpi=140); plt.close(figm)

        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== Basket stats ===")
    print(stats.round(4).to_string(index=False))
    last = rel_df.dropna().tail(1)
    if not last.empty:
        dt = last.index[-1].date()
        print(f"\nLatest z-score ({dt}): {float(last['SPREAD_Z'].iloc[0]):.2f}")
    print("\nFiles written to:", cfg.outdir)


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Rail vs Trucking: equity baskets, relative strength, and freight proxies")
    ap.add_argument("--rails", type=str, default="UNP,CSX,NSC,CP,CNI", help="Comma-separated rail tickers")
    ap.add_argument("--trucks", type=str, default="ODFL,JBHT,KNX,WERN,CHRW", help="Comma-separated trucking/3PL tickers")
    ap.add_argument("--benchmark", type=str, default="IYT", help="Benchmark for corr/beta (e.g., IYT)")
    ap.add_argument("--start", type=str, default="2010-01-01", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--freight", dest="freight_file", type=str, default=None, help="Optional freight fundamentals CSV")
    ap.add_argument("--rollwin", type=int, default=90, help="Rolling window (days) for z-score/corr/beta")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        rails=parse_list(args.rails),
        trucks=parse_list(args.trucks),
        benchmark=(args.benchmark.strip().upper() if args.benchmark else None),
        start=args.start,
        end=args.end,
        freight_file=args.freight_file,
        rollwin=int(max(20, args.rollwin)),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Universe — Rails: {cfg.rails} | Trucks: {cfg.trucks} | Bench: {cfg.benchmark or '-'}")
    run(cfg)


if __name__ == "__main__":
    main()