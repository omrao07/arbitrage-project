#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiples.py
------------
Compute valuation multiples (EV/EBITDA, EV/Sales, EV/FCF, P/E, P/B, P/S, etc.)
from fundamentals and market data. Designed to plug into your CSVs such as:
- fundamentals.csv (or per-year folders) with EBITDA, Revenue, FCF, NetIncome, Equity, BookValue, NetDebt, etc.
- prices.csv or quotes feed with close price and shares outstanding.

Columns expected (case-insensitive; synonyms supported):
  - Ticker / Symbol
  - Price (or Close)
  - SharesOutstanding (or Shares)
  - NetDebt (or DebtLessCash, DebtMinusCash, NetDebtTTM)
  - EBITDA (or EBITDA_TTM)
  - Revenue (or Sales, RevenueTTM)
  - FCF (or FreeCashFlow, FCF_TTM)
  - NetIncome (or EPS, Earnings)
  - BookValue (or EquityBook)

You can pass TTM or latest annual values; the script treats them generically.

Usage:
  # From a merged file containing market + fundamentals
  python multiples.py --in data/merged_fundamentals.csv --out outputs/multiples.csv

  # From separate files (join on Ticker)
  python multiples.py --market data/market.csv --fund data/fundamentals.csv --out outputs/multiples.csv

  # With sector mapping (optional)
  python multiples.py --in data/merged.csv --sectors data/sectors.csv --out outputs/multiples.csv

The output includes clean, capped multiples with NaN-safe handling.
"""

from __future__ import annotations
import argparse
import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Canonical column mapping
# -------------------------

MAP = {
    "ticker": ["ticker", "symbol"],
    "price": ["price", "close", "last", "px_last"],
    "shares": ["sharesoutstanding", "shares_outstanding", "shares", "so", "commonsharesout"],
    "net_debt": ["netdebt", "debtlesscash", "debtminuscash", "netdebt_ttm", "netdebtlatest"],
    "ebitda": ["ebitda", "ebitda_ttm"],
    "revenue": ["revenue", "sales", "revenue_ttm", "sales_ttm"],
    "fcf": ["fcf", "freecashflow", "fcf_ttm"],
    "net_income": ["netincome", "net_income", "net_income_ttm", "earnings", "ni_ttm"],
    "book_value": ["bookvalue", "book_value", "equitybook", "bvps_total_equity", "bookvalue_tangible"],
    "sector": ["sector", "gics_sector"],
    "country": ["country", "region", "geography"],
}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    def find(col: str) -> Optional[str]:
        for c in MAP[col]:
            m = [x for x in df.columns if x.lower().replace(" ", "") == c]
            if m:
                return m[0]
        return None
    out = {}
    for k in MAP.keys():
        src = find(k)
        if src is not None:
            out[k] = df[src]
    return pd.DataFrame(out)


# -------------------------
# Core calculations
# -------------------------

def market_cap(price: pd.Series, shares: pd.Series) -> pd.Series:
    return pd.to_numeric(price, errors="coerce") * pd.to_numeric(shares, errors="coerce")

def enterprise_value(mcap: pd.Series, net_debt: pd.Series) -> pd.Series:
    return pd.to_numeric(mcap, errors="coerce") + pd.to_numeric(net_debt, errors="coerce")

def safe_div(n: pd.Series, d: pd.Series, inf_cap: float = 1e6) -> pd.Series:
    """Safe division that returns NaN for zero/near-zero denominators and caps extreme results."""
    n = pd.to_numeric(n, errors="coerce")
    d = pd.to_numeric(d, errors="coerce")
    out = n / d.replace({0: np.nan})
    # cap to avoid exploding multiples
    out = out.clip(lower=-inf_cap, upper=inf_cap)
    return out

def compute_multiples(df: pd.DataFrame) -> pd.DataFrame:
    c = _norm_cols(df)
    for col in ["price", "shares", "net_debt", "ebitda", "revenue", "fcf", "net_income", "book_value"]:
        if col not in c:
            c[col] = np.nan  # create missing columns

    c["market_cap"] = market_cap(c["price"], c["shares"])
    c["enterprise_value"] = enterprise_value(c["market_cap"], c["net_debt"])

    # Per-share items if needed later
    with np.errstate(all="ignore"):
        c["eps"] = safe_div(c["net_income"], c["shares"])
        c["bvps"] = safe_div(c["book_value"], c["shares"])

    # Core multiples
    c["pe"] = safe_div(c["market_cap"], c["net_income"])
    c["pb"] = safe_div(c["market_cap"], c["book_value"])
    c["ps"] = safe_div(c["market_cap"], c["revenue"])

    c["ev_ebitda"] = safe_div(c["enterprise_value"], c["ebitda"])
    c["ev_sales"]  = safe_div(c["enterprise_value"], c["revenue"])
    c["ev_fcf"]    = safe_div(c["enterprise_value"], c["fcf"])

    # Yield-style
    c["fcy"] = safe_div(c["fcf"], c["market_cap"])       # FCF / Market Cap
    c["ey"]  = safe_div(c["net_income"], c["market_cap"])# Earnings / Market Cap

    # Carry over context
    if "ticker" in _norm_cols(df):
        c["ticker"] = _norm_cols(df)["ticker"]
    if "sector" in _norm_cols(df):
        c["sector"] = _norm_cols(df)["sector"]
    if "country" in _norm_cols(df):
        c["country"] = _norm_cols(df)["country"]

    # Order columns
    cols = ["ticker","sector","country","price","shares","market_cap","net_debt","enterprise_value",
            "ebitda","revenue","fcf","net_income","book_value","eps","bvps",
            "pe","pb","ps","ev_ebitda","ev_sales","ev_fcf","fcy","ey"]
    cols = [cname for cname in cols if cname in c.columns]
    return c[cols]


# -------------------------
# Group stats (optional)
# -------------------------

def summarize_by(df_mult: pd.DataFrame, by: str = "sector") -> pd.DataFrame:
    if by not in df_mult.columns:
        raise ValueError(f"'{by}' not in columns")
    agg = {
        "pe": "median",
        "pb": "median",
        "ps": "median",
        "ev_ebitda": "median",
        "ev_sales": "median",
        "ev_fcf": "median",
        "fcy": "median",
        "ey": "median",
        "ticker": "count",
    }
    # Use only existing columns
    agg = {k: v for k, v in agg.items() if k in df_mult.columns}
    out = df_mult.groupby(by).agg(agg).rename(columns={"ticker": "n"})
    return out.reset_index().sort_values("n", ascending=False)


# -------------------------
# File I/O helpers
# -------------------------

def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if path.lower().endswith(".feather"):
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {path}")

def join_market_fund(market_path: str, fund_path: str, on: str = "Ticker") -> pd.DataFrame:
    mk = _read_any(market_path)
    fd = _read_any(fund_path)
    # normalize join key
    mk_cols = {c.lower(): c for c in mk.columns}
    fd_cols = {c.lower(): c for c in fd.columns}
    key = None
    for cand in ["ticker", "symbol"]:
        if cand in mk_cols and cand in fd_cols:
            key = (mk_cols[cand], fd_cols[cand])
            break
    if key is None:
        # fallback to provided 'on'
        key = (on, on)
    df = mk.merge(fd, left_on=key[0], right_on=key[1], how="inner", suffixes=("", "_fund"))
    return df


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute valuation multiples from fundamentals + market data")
    ap.add_argument("--in", dest="inp", default=None, help="Single merged file (CSV/Parquet/Feather)")
    ap.add_argument("--market", default=None, help="Market file with Price/Shares/Ticker")
    ap.add_argument("--fund", default=None, help="Fundamentals file with EBITDA/Revenue/FCF/NetIncome/BookValue/NetDebt")
    ap.add_argument("--sectors", default=None, help="Optional sector mapping CSV with columns: Ticker, Sector")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--by", default=None, help="Optional summary grouping (e.g., sector, country)")
    args = ap.parse_args()

    if args.inp is None and (args.market is None or args.fund is None):
        raise SystemExit("Provide either --in merged_file OR both --market and --fund")

    if args.inp:
        df = _read_any(args.inp)
    else:
        df = join_market_fund(args.market, args.fund)

    if args.sectors:
        try:
            sec = pd.read_csv(args.sectors)
            # try to merge on Ticker (case-insensitive)
            s_key = [c for c in sec.columns if c.lower() in {"ticker","symbol"}]
            if not s_key:
                raise ValueError("Sector file must contain Ticker/Symbol column")
            s_key = s_key[0]
            d_key = [c for c in df.columns if c.lower() in {"ticker","symbol"}][0]
            df = df.merge(sec.rename(columns={s_key: d_key}), on=d_key, how="left")
        except Exception as e:
            print(f"⚠️ sector merge failed: {e}")

    mult = compute_multiples(df)
    mult.to_csv(args.out, index=False)
    print(f"✅ wrote {args.out} ({len(mult)} rows)")

    if args.by:
        try:
            summ = summarize_by(mult, by=args.by)
            path_s = args.out.replace(".csv", f".by_{args.by}.csv")
            summ.to_csv(path_s, index=False)
            print(f"✅ wrote {path_s}")
        except Exception as e:
            print(f"⚠️ summary failed: {e}")

if __name__ == "__main__":
    main()