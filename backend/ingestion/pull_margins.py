#!/usr/bin/env python3
"""
pull_margins.py
---------------
Builds margin datasets (gross, EBITDA, EBIT/operating, net) by company, sector, and region.

Inputs (any subset is OK; the script is resilient):

A) Unified profitability feed (preferred if available):
  data/adamodar/curated/profitability.csv
    cols (min): as_of, ticker, revenue_usd, gross_profit_usd, ebitda_usd, ebit_usd, net_income_usd,
                 sector?, region?, market_cap_usd?

B) Raw income statement feeds (long form). Provide one or more:
  - data/adamodar/curated/income_statements/us_equities.csv
  - data/adamodar/curated/income_statements/global_equities.csv
  Required cols (min): as_of, ticker, metric, value
    where metric in {"revenue","gross_profit","ebitda","ebit","net_income"}
  Optional: sector, region, market_cap_usd

C) Fundamentals “wide” feeds (per row = ticker-date):
  - data/adamodar/curated/fundamentals.csv (or year folder files)
  Must include: as_of, ticker and revenue/gross/ebitda/ebit/net income columns.

Outputs:
  data/adamodar/curated/margins_by_company.csv
  data/adamodar/curated/margins_by_sector.csv
  data/adamodar/curated/margins_by_region.csv

Each contains both the latest snapshot and (optionally) full timeseries if --timeseries is used.

Usage:
  python pull_margins.py --prefer profitability --timeseries
  python pull_margins.py --raw data/adamodar/curated/income_statements/us_equities.csv --timeseries
  python pull_margins.py --fundamentals data/adamodar/curated/fundamentals.csv
  python pull_margins.py --prefer profitability --window 4    # 4-period TTM if quarterly data
"""

import os
import argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

CURATED_DIR = "data/adamodar/curated"
OUT_COMPANY = os.path.join(CURATED_DIR, "margins_by_company.csv")
OUT_SECTOR  = os.path.join(CURATED_DIR, "margins_by_sector.csv")
OUT_REGION  = os.path.join(CURATED_DIR, "margins_by_region.csv")

# -------------------- I/O helpers --------------------

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path and os.path.exists(path) else None

# -------------------- Normalization --------------------

REQ_COLS = {
    "as_of", "ticker",
    "revenue_usd", "gross_profit_usd",
    "ebitda_usd", "ebit_usd", "net_income_usd"
}

ALT_NAMES = {
    "revenue": "revenue_usd",
    "sales": "revenue_usd",
    "total_revenue": "revenue_usd",
    "gross_profit": "gross_profit_usd",
    "operating_income": "ebit_usd",
    "ebit": "ebit_usd",
    "net_income": "net_income_usd",
    "net_income_common": "net_income_usd",
    "market_cap": "market_cap_usd"
}

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def normalize_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """Take 'profitability.csv' style and standardize columns."""
    d = df.copy()
    # standardize alt names
    d = d.rename(columns={c: ALT_NAMES.get(c, c) for c in d.columns})
    # keep optional metadata if present
    keep = ["as_of","ticker","sector","region","market_cap_usd",
            "revenue_usd","gross_profit_usd","ebitda_usd","ebit_usd","net_income_usd","frequency"]
    d = d[[c for c in keep if c in d.columns]].copy()
    d["as_of"] = pd.to_datetime(d["as_of"])
    d = coerce_numeric(d, ["market_cap_usd","revenue_usd","gross_profit_usd","ebitda_usd","ebit_usd","net_income_usd"])
    return d

def normalize_income_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect: as_of, ticker, metric, value, [sector, region, market_cap_usd]
    Metric values mapped to _usd names.
    """
    d = df.copy()
    d["as_of"] = pd.to_datetime(d["as_of"])
    d["metric"] = d["metric"].str.lower().str.strip()
    # map to target names
    metric_map = {
        "revenue":"revenue_usd", "sales":"revenue_usd", "total_revenue":"revenue_usd",
        "gross_profit":"gross_profit_usd",
        "ebitda":"ebitda_usd",
        "ebit":"ebit_usd","operating_income":"ebit_usd",
        "net_income":"net_income_usd","net_income_common":"net_income_usd"
    }
    d["metric"] = d["metric"].map(metric_map).dropna()
    d = d.rename(columns={"value":"amount"})
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    wide = d.pivot_table(index=["as_of","ticker"], columns="metric", values="amount", aggfunc="last").reset_index()
    # join optional metadata (last known per ticker)
    meta_cols = [c for c in ["sector","region","market_cap_usd"] if c in df.columns]
    if meta_cols:
        last_meta = (df.sort_values("as_of")
                       .groupby("ticker")[meta_cols]
                       .tail(1)
                       .drop_duplicates(subset=["ticker"]))
        wide = wide.merge(last_meta, on="ticker", how="left")
    return wide

def normalize_fundamentals_wide(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["as_of"] = pd.to_datetime(d["as_of"])
    # rename alt names if any
    d = d.rename(columns={c: ALT_NAMES.get(c, c) for c in d.columns})
    needed = {"as_of","ticker"}
    # pass through numeric coercion
    num_candidates = ["market_cap_usd","revenue_usd","gross_profit_usd","ebitda_usd","ebit_usd","net_income_usd"]
    d = coerce_numeric(d, num_candidates)
    # keep only required/optional cols
    keep = list(needed.union(num_candidates).union({"sector","region","frequency"}))
    keep = [c for c in keep if c in d.columns]
    return d[keep].copy()

# -------------------- Compute margins --------------------

def compute_margins(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    eps = 1e-12
    rev = d["revenue_usd"].replace(0, np.nan)

    d["gross_margin"]   = d["gross_profit_usd"] / rev
    d["ebitda_margin"]  = d["ebitda_usd"] / rev
    d["ebit_margin"]    = d["ebit_usd"] / rev
    d["net_margin"]     = d["net_income_usd"] / rev

    # Clean crazy values
    for c in ["gross_margin","ebitda_margin","ebit_margin","net_margin"]:
        d[c] = d[c].replace([np.inf,-np.inf], np.nan)

    return d

def ttm_rollup(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    If input is quarterly (or monthly) you can TTM by summing income items over 'window' periods.
    Requires evenly spaced frequency; we’ll group by ticker and roll sum.
    """
    if window <= 1:
        return df
    cols_sum = ["revenue_usd","gross_profit_usd","ebitda_usd","ebit_usd","net_income_usd"]
    d = df.sort_values(["ticker","as_of"]).copy()
    for c in cols_sum:
        if c in d.columns:
            d[c] = d.groupby("ticker")[c].transform(lambda s: s.rolling(window).sum())
    # drop rows before first complete window
    d = d.groupby("ticker").apply(lambda g: g.iloc[window-1:]).reset_index(drop=True)
    return d

# -------------------- Aggregation --------------------

def agg_bucket(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Equal-weight and cap-weight aggregates by sector or region on the latest date per ticker.
    """
    if df.empty or level not in df.columns:
        return pd.DataFrame(columns=["as_of", level, "gross_margin_eq","ebitda_margin_eq","ebit_margin_eq","net_margin_eq",
                                     "gross_margin_cap","ebitda_margin_cap","ebit_margin_cap","net_margin_cap","count","method"])

    # Use the latest observation per ticker
    latest = df.sort_values("as_of").groupby("ticker").tail(1)
    latest = latest.dropna(subset=["revenue_usd"])  # avoid 0-div artifacts

    eq = latest.groupby(level)[["gross_margin","ebitda_margin","ebit_margin","net_margin"]].mean()

    # cap weights
    if "market_cap_usd" in latest.columns and latest["market_cap_usd"].notna().any():
        tmp = latest.dropna(subset=["market_cap_usd"]).copy()
        weights = tmp.groupby(level)["market_cap_usd"].transform(lambda s: s / s.sum() if s.sum() else np.nan)
        tmp["w"] = weights
        cap = tmp.groupby(level).apply(lambda g: pd.Series({
            "gross_margin_cap":  np.nansum(g["gross_margin"]  * g["w"]),
            "ebitda_margin_cap": np.nansum(g["ebitda_margin"] * g["w"]),
            "ebit_margin_cap":   np.nansum(g["ebit_margin"]   * g["w"]),
            "net_margin_cap":    np.nansum(g["net_margin"]    * g["w"]),
        }))
    else:
        cap = pd.DataFrame(index=eq.index, data={
            "gross_margin_cap":  np.nan,
            "ebitda_margin_cap": np.nan,
            "ebit_margin_cap":   np.nan,
            "net_margin_cap":    np.nan,
        })

    out = eq.copy()
    out.columns = ["gross_margin_eq","ebitda_margin_eq","ebit_margin_eq","net_margin_eq"]
    out = out.join(cap, how="left")
    out["count"] = latest.groupby(level)["ticker"].nunique()
    out["method"] = "eq=mean(margins); cap=sum(w*margins)"
    out = out.reset_index()
    out["as_of"] = latest["as_of"].max()
    cols = ["as_of", level, "gross_margin_eq","ebitda_margin_eq","ebit_margin_eq","net_margin_eq",
            "gross_margin_cap","ebitda_margin_cap","ebit_margin_cap","net_margin_cap","count","method"]
    return out[cols].sort_values(level)

# -------------------- Pipeline --------------------

def load_and_normalize(prefer: str,
                       profitability_path: Optional[str],
                       raw_paths: List[str],
                       fundamentals_path: Optional[str]) -> pd.DataFrame:
    """
    Choose the best available source in order of preference.
    """
    if prefer == "profitability" and profitability_path and os.path.exists(profitability_path):
        pf = read_csv_safe(profitability_path)
        return normalize_profitability(pf) # type: ignore

    # else try raw long income statements
    for p in (raw_paths or []):
        if os.path.exists(p):
            d = read_csv_safe(p)
            return normalize_income_long(d) # type: ignore

    # else fundamentals wide
    if fundamentals_path and os.path.exists(fundamentals_path):
        f = read_csv_safe(fundamentals_path)
        return normalize_fundamentals_wide(f) # type: ignore

    raise FileNotFoundError("No valid inputs found. Provide --profitability OR --raw OR --fundamentals")

def run(prefer: str,
        profitability_path: Optional[str],
        raw_paths: List[str],
        fundamentals_path: Optional[str],
        timeseries: bool,
        window: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (by_company, by_sector, by_region)
    """
    df = load_and_normalize(prefer, profitability_path, raw_paths, fundamentals_path)

    # If user wants TTM from periodic data
    if window and window > 1:
        df = df.sort_values(["ticker","as_of"])
        df = ttm_rollup(df, window)

    # Compute margins
    df = compute_margins(df)

    # If not timeseries, keep latest per ticker
    company_out = df.copy()
    if not timeseries:
        company_out = company_out.sort_values("as_of").groupby("ticker").tail(1)

    # Aggregations
    sector_out = agg_bucket(df, "sector") if "sector" in df.columns else pd.DataFrame()
    region_out = agg_bucket(df, "region") if "region" in df.columns else pd.DataFrame()

    # Write
    ensure_dir(OUT_COMPANY); company_out.to_csv(OUT_COMPANY, index=False)
    if not sector_out.empty:
        ensure_dir(OUT_SECTOR); sector_out.to_csv(OUT_SECTOR, index=False)
    if not region_out.empty:
        ensure_dir(OUT_REGION); region_out.to_csv(OUT_REGION, index=False)

    print(f"✅ wrote {OUT_COMPANY} ({len(company_out)} rows{' (timeseries)' if timeseries else ''})")
    if not sector_out.empty:
        print(f"✅ wrote {OUT_SECTOR} ({len(sector_out)} rows)")
    if not region_out.empty:
        print(f"✅ wrote {OUT_REGION} ({len(region_out)} rows)")

    return company_out, sector_out, region_out

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Compute gross/EBITDA/EBIT/net margins by company/sector/region.")
    ap.add_argument("--prefer", choices=["profitability","raw","fundamentals"], default="profitability",
                    help="Which source to prefer when multiple available")
    ap.add_argument("--profitability", default=os.path.join(CURATED_DIR, "profitability.csv"),
                    help="Path to profitability.csv (unified feed)")
    ap.add_argument("--raw", nargs="*", default=[],
                    help="One or more raw income statement CSVs (long form: as_of,ticker,metric,value)")
    ap.add_argument("--fundamentals", default=os.path.join(CURATED_DIR, "fundamentals.csv"),
                    help="Path to fundamentals wide CSV")
    ap.add_argument("--timeseries", action="store_true",
                    help="If set, output full timeseries for company file (default writes latest snapshot)")
    ap.add_argument("--window", type=int, default=1,
                    help="Rolling sum window for TTM (e.g., 4 for quarterly, 12 for monthly)")
    args = ap.parse_args()

    run(args.prefer, args.profitability, args.raw, args.fundamentals, args.timeseries, args.window)

if __name__ == "__main__":
    main()