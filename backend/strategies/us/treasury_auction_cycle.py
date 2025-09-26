#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# treasury_auction_cycles.py
#
# U.S. Treasury Auction Cycles — supply, seasonality, and tail analysis
# --------------------------------------------------------------------
# What this does
# - Ingests an auctions CSV (from Treasury) and optional market prices (WI yields, on-the-run)
# - Cleans security types/tenors; computes auction "tail" (stop - WI), bid-to-cover, allotment shares
# - Maps auctions to calendar features (day-of-month, week-of-month, month/quarter ends)
# - Finds cycle regularities by tenor (typical weekdays & day-of-month distributions)
# - Event study around auction day using daily yields you provide
# - Simple OLS: tail ~ size + bcov + indirect_share + month_end + (controls)
# - Exports tidy CSVs and optional PNG plots
#
# Usage
# -----
# python treasury_auction_cycles.py \
#   --auctions auctions.csv \
#   --prices prices.csv \
#   --tpre 5 --tpost 10 \
#   --plot
#
# auctions.csv (flexible, case-insensitive; extras preserved)
# ----------------------------------------------------------
# date,security_type,term,offering_amount,awarded_amount,bid_to_cover,stop_yield,wi_yield,tail_bp,high_price,off_the_run, \
# indirect_award, direct_award, dealer_award, soma_addon, cusip, reopen, issue_date
# Notes:
# - date = auction date (YYYY-MM-DD). If wi_yield absent, we compute tail as NaN or from tail_bp if given.
# - term/tenor can be strings like "2-Year Note", "13-Week Bill", "7Y", "10-year"
# - allotments can be in $ or %, we normalize to % of total awarded when possible.
#
# prices.csv (optional; daily yields you track)
# --------------------------------------------
# date,tenor,yield
# 2024-01-02,2Y,0.043
# 2024-01-03,2Y,0.0434
# (Use tickers/tenors you want—script will pivot by 'tenor' to build event windows)
#
# Outputs (./artifacts/treasury_auction_cycles/*)
# ----------------------------------------------
# auctions_clean.csv
# seasonality_by_tenor.csv
# cycle_calendar.csv
# tails_panel.csv
# event_panel.csv
# regress_tails.csv
# plots/*.png   (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib statsmodels python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import statsmodels.api as sm
from dateutil import parser as dtp


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    auctions_file: str
    prices_file: Optional[str]
    tpre: int
    tpost: int
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "treasury_auction_cycles_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def _num(x):
    if pd.isna(x): return np.nan
    try:
        return float(str(x).replace(",","").replace("$","").replace("%","").strip())
    except Exception:
        return np.nan

def _lower(s): return str(s).strip().lower() if s is not None else ""

def parse_date(x):
    if pd.isna(x) or str(x).strip()=="":
        return pd.NaT
    try:
        return pd.Timestamp(dtp.parse(str(x)).date())
    except Exception:
        return pd.NaT

def tenor_normalize(term: str, sec_type: str) -> str:
    s = (_lower(term) or _lower(sec_type))
    s = s.replace("year","y").replace("yr","y").replace("week","w").replace("mo","m").replace("month","m")
    # quick mappings
    for key in ["1y","2y","3y","5y","7y","10y","20y","30y","8w","4w","13w","17w","26w","52w","3m","6m","12m"]:
        if key in s: return key.upper()
    # FRN / TIPS guess
    if "frn" in s or "floating" in s: return "FRN"
    if "tips" in s or "inflation" in s: 
        # try to pull years
        for key in ["5y","7y","10y","20y","30y"]:
            if key in s: return f"TIPS_{key.upper()}"
        return "TIPS"
    # fallback
    return term.upper() if term else sec_type.upper()

def sec_bucket(sec_type: str) -> str:
    s = _lower(sec_type)
    if "bill" in s or any(x in s for x in ["w","wk","week","t-bill"]): return "BILL"
    if "note" in s: return "NOTE"
    if "bond" in s and "i" not in s: return "BOND"
    if "tips" in s: return "TIPS"
    if "frn" in s or "floating" in s: return "FRN"
    return "OTHER"

def week_of_month(d: pd.Timestamp) -> int:
    # 1..5
    first = d.replace(day=1)
    dom = d.day
    adj = (first.weekday() + 1) % 7  # 0=Mon alignment tweak not critical
    return int(np.ceil((dom + adj) / 7.0))

def pct(x): 
    return 100.0 * x if pd.notna(x) else np.nan


# ----------------------------- Load & clean -----------------------------

def load_auctions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise SystemExit("auctions file must include 'date' column (auction date).")
    df["date"] = df["date"].apply(parse_date)

    # Normalize numerics
    for c in ["offering_amount","awarded_amount","bid_to_cover","stop_yield","wi_yield","tail_bp",
              "indirect_award","direct_award","dealer_award","soma_addon","high_price"]:
        if c in df.columns:
            df[c] = df[c].apply(_num)

    # Security descriptors
    df["security_type"] = df.get("security_type", "").astype(str)
    df["term"] = df.get("term", df.get("tenor","")).astype(str)
    df["tenor"] = [tenor_normalize(t, s) for t,s in zip(df["term"], df["security_type"])]
    df["bucket"] = df["security_type"].apply(sec_bucket)

    # Tail in basis points (stop - WI)
    if "tail_bp" in df.columns and df["tail_bp"].notna().any():
        df["tail_bp"] = df["tail_bp"]
    else:
        df["tail_bp"] = (df["stop_yield"] - df["wi_yield"]) * 1e4  # (decimal yields)
    # Allotment shares as % awarded
    total = df["awarded_amount"].replace(0,np.nan)
    for c in ["indirect_award","direct_award","dealer_award","soma_addon"]:
        if c in df.columns:
            # If already in % (0..100), keep; else convert $ to %
            is_pct_like = (df[c].dropna().abs() <= 1.0).mean() < 0.3  # crude detection
            if not is_pct_like:
                df[c+"_pct"] = 100.0 * df[c] / total
            else:
                df[c+"_pct"] = df[c]
    # Reopen flag
    if "reopen" in df.columns:
        df["reopen"] = df["reopen"].astype(str).str.lower().isin(["1","true","yes","y","reopen"])
    else:
        df["reopen"] = False

    # Calendar features
    df["dow"] = df["date"].dt.day_name()
    df["dom"] = df["date"].dt.day
    df["wom"] = df["date"].apply(week_of_month)
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    df["month_end"] = (df["date"] == df["date"] + pd.offsets.MonthEnd(0)).astype(int)
    df["quarter_end"] = (df["date"] == df["date"] + pd.offsets.QuarterEnd(0)).astype(int)
    df["fomc_week"] = df["date"].dt.isocalendar().week.diff().abs().fillna(0)  # placeholder if you don't provide meeting dates
    # Clean bid-to-cover / size
    if "bid_to_cover" in df.columns:
        df["bcov"] = df["bid_to_cover"]
    else:
        df["bcov"] = np.nan
    df["size_usd"] = df.get("offering_amount", df.get("awarded_amount", np.nan))

    return df.sort_values("date").reset_index(drop=True)


def load_prices(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    px = pd.read_csv(path)
    px.columns = [c.strip().lower() for c in px.columns]
    if not {"date","tenor","yield"} <= set(px.columns):
        raise SystemExit("prices.csv must include: date, tenor, yield")
    px["date"] = px["date"].apply(parse_date)
    px["tenor"] = px["tenor"].astype(str).str.upper()
    px["yield"] = px["yield"].apply(_num)
    # pivot to wide (columns = tenor)
    wide = px.pivot_table(index="date", columns="tenor", values="yield", aggfunc="last").sort_index()
    return wide


# ----------------------------- Seasonality & cycles -----------------------------

def seasonality_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # For each tenor: weekday distribution and day-of-month histogram
    wk = df.groupby(["tenor","dow"]).size().rename("count").reset_index()
    wk["pct"] = wk.groupby("tenor")["count"].apply(lambda s: 100*s/s.sum())
    dom = df.groupby(["tenor","dom"]).size().rename("count").reset_index()
    dom["pct"] = dom.groupby("tenor")["count"].apply(lambda s: 100*s/s.sum())
    return wk, dom


def cycle_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact calendar: for each tenor, the most common week-of-month and typical weekday(s).
    """
    rows = []
    for ten, g in df.groupby("tenor"):
        wom_mode = g["wom"].mode().tolist()
        dow_top = g["dow"].value_counts(normalize=True).head(2)
        rows.append({
            "tenor": ten,
            "typical_wom": ",".join(str(int(x)) for x in wom_mode[:2]),
            "top_dows": ",".join(list(dow_top.index)),
            "month_end_bias_%": 100.0 * g["month_end"].mean(),
            "quarter_end_bias_%": 100.0 * g["quarter_end"].mean(),
            "avg_size_bn": (g["size_usd"].mean() or np.nan)/1e9,
            "avg_bcov": g["bcov"].mean(),
            "avg_tail_bp": g["tail_bp"].mean()
        })
    return pd.DataFrame(rows).sort_values("tenor")


# ----------------------------- Event study (daily) -----------------------------

def build_event_panel(prices_wide: pd.DataFrame, auctions: pd.DataFrame, tpre: int, tpost: int) -> pd.DataFrame:
    """
    For each auction, extract yield changes around the date for the closest matching tenor in prices.
    Returns a long table with τ ∈ [-tpre, +tpost]
    """
    if prices_wide.empty:
        return pd.DataFrame()
    # map auction tenor to a column name in prices (simple exact match)
    events = []
    for _, r in auctions.iterrows():
        ten = r["tenor"]
        if ten in prices_wide.columns:
            d = r["date"]
            if d not in prices_wide.index:
                # snap to next trading day
                idx = prices_wide.index
                nxt = idx[idx >= d]
                if len(nxt)==0: 
                    continue
                d = nxt[0]
            center = prices_wide.index.get_indexer([d])[0]
            start = max(0, center - tpre); end = min(len(prices_wide)-1, center + tpost)
            window = prices_wide.iloc[start:end+1][ten].copy()
            rets = window.diff()  # Δyields (level changes)
            # relative index
            rel = np.arange(start - center, end - center + 1)
            tmp = pd.DataFrame({
                "tenor": ten, "auction_date": r["date"], "tau": rel, "dyield": rets.values
            })
            tmp["tail_bp"] = r["tail_bp"]; tmp["size_usd"] = r["size_usd"]; tmp["bcov"] = r["bcov"]
            tmp["month_end"] = r["month_end"]; tmp["dow"] = r["dow"]; tmp["reopen"] = r["reopen"]
            events.append(tmp)
    return pd.concat(events, ignore_index=True) if events else pd.DataFrame()


# ----------------------------- Regressions -----------------------------

def regress_tails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional OLS on *auction-day* tails by tenor:
    tail_bp ~ log(size) + bcov + indirect% + direct% + dealer% + month_end + reopen
    """
    cols = ["tail_bp","size_usd","bcov","indirect_award_pct","direct_award_pct","dealer_award_pct","month_end","reopen","tenor","date"]
    use = df.copy()
    for c in ["indirect_award_pct","direct_award_pct","dealer_award_pct"]:
        if c not in use.columns: use[c] = np.nan
    use["log_size"] = np.log(use["size_usd"].replace(0,np.nan))
    out = []
    for ten, g in use.groupby("tenor"):
        sub = g[["tail_bp","log_size","bcov","indirect_award_pct","direct_award_pct","dealer_award_pct","month_end","reopen"]].dropna()
        if len(sub) < 30: 
            # not enough obs; still compute if >=10 with reduced spec
            sub = g[["tail_bp","log_size","bcov","month_end","reopen"]].dropna()
        if len(sub) < 10: 
            continue
        X = sm.add_constant(sub.drop(columns=["tail_bp"]))
        y = sub["tail_bp"]
        try:
            res = sm.OLS(y, X).fit()
            out.append({
                "tenor": ten, "nobs": int(res.nobs), "r2": float(res.rsquared),
                "b_log_size": float(res.params.get("log_size", np.nan)),
                "t_log_size": float(res.tvalues.get("log_size", np.nan)),
                "b_bcov": float(res.params.get("bcov", np.nan)),
                "t_bcov": float(res.tvalues.get("bcov", np.nan)),
                "b_month_end": float(res.params.get("month_end", np.nan)),
                "t_month_end": float(res.tvalues.get("month_end", np.nan)),
                "b_reopen": float(res.params.get("reopen", np.nan)),
                "t_reopen": float(res.tvalues.get("reopen", np.nan)),
            })
        except Exception:
            continue
    return pd.DataFrame(out)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="U.S. Treasury auction cycles: seasonality, tails, and event windows")
    ap.add_argument("--auctions", dest="auctions_file", required=True, help="CSV of auctions (see header for columns)")
    ap.add_argument("--prices", dest="prices_file", default=None, help="Optional daily yields CSV (date,tenor,yield)")
    ap.add_argument("--tpre", type=int, default=5, help="Days before auction for event study")
    ap.add_argument("--tpost", type=int, default=10, help="Days after auction for event study")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        auctions_file=args.auctions_file,
        prices_file=args.prices_file,
        tpre=int(max(1, args.tpre)),
        tpost=int(max(1, args.tpost)),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    # Load
    auctions = load_auctions(cfg.auctions_file)
    auctions.to_csv(os.path.join(cfg.outdir, "auctions_clean.csv"), index=False)

    # Seasonality
    wk, dom = seasonality_tables(auctions)
    wk.to_csv(os.path.join(cfg.outdir, "seasonality_weekday.csv"), index=False)
    dom.to_csv(os.path.join(cfg.outdir, "seasonality_day_of_month.csv"), index=False)

    cyc = cycle_calendar(auctions)
    cyc.to_csv(os.path.join(cfg.outdir, "cycle_calendar.csv"), index=False)

    # Tails panel (daily = auction-day snapshot)
    tails = auctions[["date","tenor","bucket","size_usd","bcov","tail_bp","indirect_award_pct","direct_award_pct","dealer_award_pct","month_end","quarter_end","reopen"]].copy()
    tails.to_csv(os.path.join(cfg.outdir, "tails_panel.csv"), index=False)

    # Regression on tails
    reg = regress_tails(auctions)
    if not reg.empty:
        reg.to_csv(os.path.join(cfg.outdir, "regress_tails.csv"), index=False)

    # Event study with prices
    prices = load_prices(cfg.prices_file) if cfg.prices_file else pd.DataFrame()
    panel = pd.DataFrame()
    if not prices.empty:
        panel = build_event_panel(prices, auctions, cfg.tpre, cfg.tpost)
        if not panel.empty:
            panel.to_csv(os.path.join(cfg.outdir, "event_panel.csv"), index=False)

    # ---------------- Plots ----------------
    if cfg.plot and plt is not None:
        # Weekday heatmap-like bar per tenor
        if not wk.empty:
            for ten in wk["tenor"].unique():
                sub = wk[wk["tenor"]==ten]
                fig = plt.figure(figsize=(6,3)); ax = plt.gca()
                ax.bar(sub["dow"], sub["pct"])
                ax.set_title(f"{ten}: weekday share of auctions"); ax.set_ylabel("%")
                plt.xticks(rotation=30); plt.tight_layout()
                fig.savefig(os.path.join(cfg.outdir, "plots", f"weekday_{ten}.png"), dpi=140); plt.close(fig)

        # Tail distribution by tenor
        fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
        auctions.boxplot(column="tail_bp", by="tenor", ax=ax2, rot=45, grid=False)
        ax2.set_title("Auction tails by tenor (bp)"); ax2.set_ylabel("bp"); plt.suptitle("")
        plt.tight_layout(); fig2.savefig(os.path.join(cfg.outdir, "plots", "tails_by_tenor.png"), dpi=140); plt.close(fig2)

        # Size vs tail scatter (color = reopen)
        fig3 = plt.figure(figsize=(7,5)); ax3 = plt.gca()
        s = auctions.dropna(subset=["tail_bp","size_usd"])
        sc = ax3.scatter(np.log(s["size_usd"]/1e9), s["tail_bp"], c=s["reopen"].astype(int), alpha=0.7)
        ax3.set_xlabel("log(size, $bn)"); ax3.set_ylabel("tail (bp)")
        ax3.set_title("Tail vs size"); plt.tight_layout()
        fig3.savefig(os.path.join(cfg.outdir, "plots", "tail_vs_size.png"), dpi=140); plt.close(fig3)

        # Event-study average Δy around τ by tenor (first 6 tenors)
        if not panel.empty:
            for ten in list(panel["tenor"].unique())[:6]:
                sub = panel[panel["tenor"]==ten].groupby("tau")["dyield"].mean()
                fig4 = plt.figure(figsize=(7,4)); ax4 = plt.gca()
                ax4.plot(sub.index, 1e4*sub.values)  # convert to bp
                ax4.axvline(0, linestyle="--", alpha=0.6)
                ax4.set_title(f"Avg Δyield around auction — {ten}"); ax4.set_xlabel("days from auction"); ax4.set_ylabel("bp")
                plt.tight_layout(); fig4.savefig(os.path.join(cfg.outdir, "plots", f"event_{ten}.png"), dpi=140); plt.close(fig4)

        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # ---------------- Console snapshot ----------------
    print("\n=== Cycle calendar (snippet) ===")
    snap_cols = ["tenor","typical_wom","top_dows","month_end_bias_%","avg_size_bn","avg_bcov","avg_tail_bp"]
    print(cyc[snap_cols].round(3).to_string(index=False))

    if not reg.empty:
        print("\n=== Tail regression (per tenor) ===")
        print(reg.sort_values("r2", ascending=False).round(3).to_string(index=False))

    if not panel.empty:
        print("\n=== Event study: overall average Δy (bp) by τ (first 6 τ) ===")
        m = panel.groupby("tau")["dyield"].mean().mul(1e4).round(2).head(6)
        print(m.to_string())

    print("\nFiles written to:", cfg.outdir)


if __name__ == "__main__":
    main()