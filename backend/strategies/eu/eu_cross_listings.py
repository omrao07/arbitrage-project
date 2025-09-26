#!/usr/bin/env python3
"""
eu_cross_listings.py — Cross-listing analytics for EU equities (premia, overlap, liquidity)

What it does
------------
Given listings, prices, and FX, this script identifies cross-listed equities (by ISIN) and computes:
- Pairwise *premia* between venues (price_A / price_B − 1) in a common base currency
- Rolling stats (mean, stdev, z-score) and residual half-life (AR(1)) of venue premia
- Liquidity metrics (ADV €; turnover proxy) per venue and for the pair
- Overlap diagnostics: timestamps when both venues are simultaneously open (optional calendar)
- Corporate-action adjustment (optional splits/recaps)

It handles both daily and intraday data. Intraday requires timestamps in UTC.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--listings listings.csv
    Columns (recommended): isin, ticker, exchange, country, currency, primary(0/1)

--prices prices.csv
    Columns: timestamp(or date), isin or (ticker+exchange), price, volume, currency(optional)
    Intraday timestamps should be UTC. Prices should be clean/close (your choice, but be consistent).
    If currency missing here, it will be taken from listings.

--fx fx.csv
    Columns: date, ccy, rate_to_EUR  (i.e., 1 unit of 'ccy' × rate_to_EUR = EUR)

--calendar calendar.csv (optional; for overlap windows)
    Columns: exchange, open_utc, close_utc
    Example: XETR, 08:00, 16:30    (24h HH:MM, UTC)

--corporate corporate.csv (optional; split/recap adjustments)
    Columns: isin, date, action, factor
    If action contains 'split' or 'reverse', multiply historical prices by cumulative factor up to each date.

Key options
-----------
--freq D                Resample frequency (Pandas offset alias like 'D', '60T', '5T')
--win 60                Rolling window for premia stats
--base EUR              Base currency for all premia/liquidity calculations
--min-overlap-mins 30   Require at least this many overlap minutes for intraday mispricing flags
--pair-limit 0          If >0, limit to first N pairs (for quick runs)
--outdir out_cross

Outputs
-------
- pairs.csv               All venue pairs per ISIN with metadata (primary, currencies)
- premia_timeseries.csv   Timestamped premia, rolling stats, z-scores, and AR(1) residuals summary
- overlap_windows.csv     Overlap summary by pair (minutes/day, coverage %), if calendar provided
- liquidity.csv           ADV (EUR), volumes, venue-level liquidity
- summary.json            Latest KPIs and extreme premia per pair
- config.json             Reproducibility dump

Usage
-----
python eu_cross_listings.py --listings listings.csv --prices prices.csv --fx fx.csv \
  --calendar calendar.csv --freq D --win 60 --outdir out_cross
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------- helpers ----------------------
def ncol(df: pd.DataFrame, name: str) -> Optional[str]:
    tl = name.lower()
    for c in df.columns:
        if c.lower() == tl:
            return c
    for c in df.columns:
        if tl in c.lower():
            return c
    return None


def to_datetime(x: pd.Series) -> pd.Series:
    if np.issubdtype(x.dtype, np.datetime64):
        return x
    return pd.to_datetime(x, errors="coerce", utc=True)


def to_date(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.normalize()


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ar1_phi(series: pd.Series) -> Tuple[float, Optional[float]]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        return (np.nan, None)
    x = s.values
    xlag = x[:-1]; y = x[1:]
    denom = float(np.dot(xlag, xlag))
    if denom <= 1e-12:
        return (np.nan, None)
    phi = float(np.dot(xlag, y) / denom)
    hl = None
    if 0 < phi < 1:
        hl = float(-np.log(2.0) / np.log(phi))
    return (phi, hl)


# ---------------------- I/O ----------------------
def read_listings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df, "isin") or "isin"): "isin",
        (ncol(df, "ticker") or "ticker"): "ticker",
        (ncol(df, "exchange") or "exchange"): "exchange",
        (ncol(df, "country") or "country"): "country",
        (ncol(df, "currency") or "currency"): "currency",
        (ncol(df, "primary") or "primary"): "primary",
    })
    df["isin"] = df["isin"].astype(str)
    df["exchange"] = df["exchange"].astype(str)
    if "primary" in df.columns:
        df["primary"] = (safe_num(df["primary"]) > 0).astype(int)
    else:
        df["primary"] = 0
    return df


def read_fx(path: str, base: str) -> pd.DataFrame:
    fx = pd.read_csv(path)
    fx = fx.rename(columns={
        (ncol(fx, "date") or fx.columns[0]): "date",
        (ncol(fx, "ccy") or ncol(fx, "currency") or "ccy"): "ccy",
        (ncol(fx, "rate_to_eur") or ncol(fx, "rate") or "rate_to_eur"): "rate_to_eur",
    })
    fx["date"] = to_date(fx["date"])
    fx["ccy"] = fx["ccy"].astype(str).str.upper()
    fx["rate_to_eur"] = safe_num(fx["rate_to_eur"])
    # Create base column rate_to_base
    if base.upper() == "EUR":
        fx["rate_to_base"] = fx["rate_to_eur"]
    else:
        # Need EUR→BASE series; assume fx includes BASE vs EUR (i.e., 1 BASE -> EUR)
        base_rows = fx[fx["ccy"] == base.upper()].copy()
        if base_rows.empty:
            raise SystemExit(f"fx.csv needs a row for base currency {base.upper()} with EUR conversion.")
        # rate_to_base(ccy) = rate_to_eur(ccy) / rate_to_eur(base)
        fx = fx.merge(base_rows[["date", "rate_to_eur"]].rename(columns={"rate_to_eur": "base_to_eur"}), on="date", how="left")
        fx["rate_to_base"] = fx["rate_to_eur"] / fx["base_to_eur"]
        fx = fx.drop(columns=["base_to_eur"])
    return fx[["date", "ccy", "rate_to_base"]]


def read_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tscol = ncol(df, "timestamp") or ncol(df, "date") or df.columns[0]
    df = df.rename(columns={
        tscol: "timestamp",
        (ncol(df, "isin") or "isin"): "isin",
        (ncol(df, "ticker") or "ticker"): "ticker",
        (ncol(df, "exchange") or "exchange"): "exchange",
        (ncol(df, "price") or "price"): "price",
        (ncol(df, "volume") or "volume"): "volume",
        (ncol(df, "currency") or "currency"): "currency",
    })
    df["timestamp"] = to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.floor("D")
    df["price"] = safe_num(df["price"])
    df["volume"] = safe_num(df["volume"])
    if "isin" in df.columns:
        df["isin"] = df["isin"].astype(str)
    if "exchange" in df.columns:
        df["exchange"] = df["exchange"].astype(str)
    return df


def read_calendar(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    cal = pd.read_csv(path)
    cal = cal.rename(columns={
        (ncol(cal, "exchange") or "exchange"): "exchange",
        (ncol(cal, "open_utc") or "open_utc"): "open_utc",
        (ncol(cal, "close_utc") or "close_utc"): "close_utc",
    })
    cal["exchange"] = cal["exchange"].astype(str)
    return cal


def read_corporate(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    ca = pd.read_csv(path)
    ca = ca.rename(columns={
        (ncol(ca, "isin") or "isin"): "isin",
        (ncol(ca, "date") or "date"): "date",
        (ncol(ca, "action") or "action"): "action",
        (ncol(ca, "factor") or "factor"): "factor",
    })
    ca["date"] = to_date(ca["date"])
    ca["isin"] = ca["isin"].astype(str)
    ca["factor"] = safe_num(ca["factor"])
    return ca


# ---------------------- core ----------------------
def apply_corporate_actions(prices: pd.DataFrame, corp: pd.DataFrame) -> pd.DataFrame:
    if corp.empty:
        return prices
    # For splits/reverse-splits, adjust historical prices by cumulative product of factors up to date
    adj = prices.copy()
    ca = corp.copy()
    ca = ca[ca["action"].astype(str).str.lower().str.contains("split")]
    if ca.empty:
        return adj
    # Build per-ISIN cumulative factor by date
    out = []
    for isin, g in adj.groupby("isin"):
        sub_ca = ca[ca["isin"] == isin].sort_values("date")
        if sub_ca.empty:
            out.append(g)
            continue
        g = g.sort_values("timestamp")
        g = g.merge(sub_ca[["date", "factor"]], left_on=g["timestamp"].dt.floor("D"), right_on="date", how="left").drop(columns=["key_0", "date"], errors="ignore")
        g["factor"] = g["factor"].fillna(1.0)
        cum = g["factor"].cumprod()
        # Adjust *prior* history: we approximate with forward fill of cumprod reversed
        # Simpler: divide price by cumprod to put all on current basis
        g["price"] = g["price"] / cum.replace(0, np.nan).fillna(1.0)
        out.append(g.drop(columns=["factor"]))
    return pd.concat(out, ignore_index=True)


def fx_convert_to_base(prices: pd.DataFrame, listings: pd.DataFrame, fx: pd.DataFrame, base: str) -> pd.DataFrame:
    df = prices.copy()
    # Attach currency: prefer prices.currency, else listings.currency by (isin, exchange)
    if "currency" not in df.columns or df["currency"].isna().all():
        cur_map = listings.set_index(["isin", "exchange"])["currency"]
        df["currency"] = [cur_map.get((i, ex), np.nan) for i, ex in zip(df["isin"], df["exchange"])]
    df["currency"] = df["currency"].astype(str).str.upper()
    # Join FX by date × ccy
    rates = fx.rename(columns={"ccy": "currency"})
    df = df.merge(rates, on=["date", "currency"], how="left")
    if base.upper() != "EUR" and "rate_to_base" not in df.columns:
        raise SystemExit("FX file missing rate_to_base for chosen base.")
    # Set EUR→base = 1 where currency==base
    df["rate_to_base"] = df["rate_to_base"].fillna(1.0).replace(0, np.nan)
    df["price_base"] = df["price"] * df["rate_to_base"]
    df["notional_eur"] = df["price_base"] * df["volume"]
    return df


def resample_prices(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample per (isin, exchange) to given frequency using last price and sum volume.
    """
    df = df.sort_values("timestamp")
    agg = (
        df.set_index("timestamp")
          .groupby(["isin", "exchange"])
          .resample(freq)
          .agg(price_base=("price_base", "last"),
               price=("price", "last"),
               volume=("volume", "sum"))
          .reset_index()
    )
    agg["date"] = agg["timestamp"].dt.floor("D")
    return agg.dropna(subset=["price_base"])


def build_pairs(listings: pd.DataFrame) -> pd.DataFrame:
    # All cross-listed ISINs with ≥ 2 exchanges
    multi = listings.groupby("isin").filter(lambda g: g["exchange"].nunique() >= 2)
    rows = []
    for isin, g in multi.groupby("isin"):
        venues = list(g["exchange"].unique())
        for a, b in combinations(sorted(venues), 2):
            pa = int(g[(g["exchange"] == a)]["primary"].max())
            pb = int(g[(g["exchange"] == b)]["primary"].max())
            rows.append({"isin": isin, "exA": a, "exB": b, "primaryA": pa, "primaryB": pb})
    return pd.DataFrame(rows)


def compute_premia(pair: dict, px: pd.DataFrame, win: int) -> pd.DataFrame:
    """
    Compute premium series for one (isin, exA, exB): prem = A/B − 1 in base.
    Adds rolling mean/sd/z, AR(1) phi/half-life (once, per series).
    """
    subA = px[(px["isin"] == pair["isin"]) & (px["exchange"] == pair["exA"])]
    subB = px[(px["isin"] == pair["isin"]) & (px["exchange"] == pair["exB"])]
    if subA.empty or subB.empty:
        return pd.DataFrame()
    merged = pd.merge_asof(
        subA.sort_values("timestamp"),
        subB.sort_values("timestamp"),
        on="timestamp", by=None, tolerance=pd.Timedelta(freq_alias(px)), direction="nearest",
        suffixes=("A", "B"))
    merged = merged.dropna(subset=["price_baseA", "price_baseB"])
    if merged.empty:
        return merged
    prem = (merged["price_baseA"] / merged["price_baseB"]) - 1.0
    merged["premium"] = prem
    # Rolling stats (by time order)
    merged = merged.sort_values("timestamp")
    merged["prem_mu"] = merged["premium"].rolling(win).mean()
    merged["prem_sd"] = merged["premium"].rolling(win).std(ddof=1)
    merged["prem_z"] = (merged["premium"] - merged["prem_mu"]) / (merged["prem_sd"] + 1e-12)
    # AR(1) on residual (premium - rolling mean) as simple proxy
    phi, hl = ar1_phi((merged["premium"] - merged["prem_mu"]).dropna())
    merged["ar1_phi"] = phi
    merged["half_life"] = hl
    merged["isin"] = pair["isin"]
    merged["exA"] = pair["exA"]
    merged["exB"] = pair["exB"]
    return merged


def freq_alias(px_resampled: pd.DataFrame) -> str:
    # Heuristic: infer the resampling frequency from median difference
    if px_resampled.empty:
        return "1D"
    t = px_resampled["timestamp"].sort_values().unique()
    if len(t) < 3:
        return "1D"
    diffs = np.diff(t).astype("timedelta64[s]").astype(float)
    med = np.median(diffs)
    if med <= 60:   # ≤1 minute
        return "60s"
    if med <= 300:  # ≤5 minutes
        return "300s"
    if med <= 3600:
        return "3600s"
    return "1D"


def overlap_minutes(timestamps: pd.Series, ex_open: str, ex_close: str) -> int:
    if timestamps.empty:
        return 0
    # Count minutes in timestamps whose time-of-day falls inside [open, close)
    open_h, open_m = map(int, ex_open.split(":"))
    close_h, close_m = map(int, ex_close.split(":"))
    tod = timestamps.dt.tz_convert("UTC").dt.time
    mins = 0
    for ts in timestamps.dt.tz_convert("UTC"):
        t = ts.time()
        if ((t.hour, t.minute) >= (open_h, open_m)) and ((t.hour, t.minute) < (close_h, close_m)):
            mins += 1  # unit: one resampled step; interpret as “one bar”
    return mins


# ---------------------- CLI ----------------------
@dataclass
class Config:
    listings: str
    prices: str
    fx: str
    calendar: Optional[str]
    corporate: Optional[str]
    freq: str
    win: int
    base: str
    min_overlap_mins: int
    pair_limit: int
    outdir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EU cross-listing analytics")
    ap.add_argument("--listings", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--fx", required=True)
    ap.add_argument("--calendar", default="")
    ap.add_argument("--corporate", default="")
    ap.add_argument("--freq", default="D")
    ap.add_argument("--win", type=int, default=60)
    ap.add_argument("--base", default="EUR")
    ap.add_argument("--min-overlap-mins", type=int, default=30)
    ap.add_argument("--pair-limit", type=int, default=0)
    ap.add_argument("--outdir", default="out_cross")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    listings = read_listings(args.listings)
    prices = read_prices(args.prices)
    fx = read_fx(args.fx, args.base)
    calendar = read_calendar(args.calendar)
    corporate = read_corporate(args.corporate)

    # Basic validation
    if listings.empty or prices.empty or fx.empty:
        raise SystemExit("Inputs appear empty. Need listings, prices, and fx.")

    # Corporate actions (optional)
    if not corporate.empty:
        prices = apply_corporate_actions(prices, corporate)

    # FX convert
    prices = fx_convert_to_base(prices, listings, fx, args.base)

    # Resample
    px = resample_prices(prices, args.freq)
    # Carry the base price column names expected by compute_premia
    px = px.rename(columns={"price_base": "price_base", "price": "price", "volume": "volume"})

    # Build pairs
    pairs_df = build_pairs(listings)
    if args.pair_limit and args.pair_limit > 0:
        pairs_df = pairs_df.head(args.pair_limit)
    pairs_df.to_csv(outdir / "pairs.csv", index=False)

    # Compute premia for each pair
    premia_rows = []
    for _, row in pairs_df.iterrows():
        prem = compute_premia(row.to_dict(), px, args.win)
        if not prem.empty:
            premia_rows.append(prem)
    premia = pd.concat(premia_rows, ignore_index=True) if premia_rows else pd.DataFrame()
    if not premia.empty:
        premia = premia[[
            "timestamp", "isin", "exA", "exB",
            "price_baseA", "price_baseB", "volumeA", "volumeB",
            "premium", "prem_mu", "prem_sd", "prem_z", "ar1_phi", "half_life"
        ]]
        premia.to_csv(outdir / "premia_timeseries.csv", index=False)

    # Liquidity metrics (ADV in base)
    if not px.empty:
        adv = (
            px.assign(notional=lambda d: d["price_base"] * d["volume"])
              .groupby(["isin", "exchange"])
              .agg(
                  obs=("timestamp", "count"),
                  days=("date", "nunique"),
                  adv_eur=("notional", lambda x: float(x.sum())/max(1, len(set(px["date"])))),
                  mean_notional=("notional", "mean"),
                  median_notional=("notional", "median"),
              )
              .reset_index()
        )
        adv.to_csv(outdir / "liquidity.csv", index=False)
    else:
        adv = pd.DataFrame()

    # Overlap diagnostics (optional)
    overlap_rows = []
    if not calendar.empty and not premia.empty:
        cal = calendar.set_index("exchange")
        for (isin, exA, exB), g in premia.groupby(["isin", "exA", "exB"]):
            if exA not in cal.index or exB not in cal.index:
                continue
            oa, ca = cal.loc[exA, ["open_utc", "close_utc"]]
            ob, cb = cal.loc[exB, ["open_utc", "close_utc"]]
            # Minutes in overlap (approx: count bars inside both opens)
            ts = g["timestamp"]
            minsA = overlap_minutes(ts, str(oa), str(ca))
            minsB = overlap_minutes(ts, str(ob), str(cb))
            overlap = min(minsA, minsB)
            overlap_rows.append({
                "isin": isin, "exA": exA, "exB": exB,
                "bars_in_overlap": int(overlap),
                "bars_exA": int(minsA),
                "bars_exB": int(minsB),
                "overlap_ok": int(overlap >= args.min_overlap_mins)
            })
        if overlap_rows:
            pd.DataFrame(overlap_rows).to_csv(outdir / "overlap_windows.csv", index=False)

    # Summary KPIs
    kpi = {}
    if not premia.empty:
        latest_ts = premia["timestamp"].max()
        last = premia[premia["timestamp"] == latest_ts]
        # extreme z by pair
        wid = (last.assign(pair=last["exA"] + "/" + last["exB"])
                    .groupby(["isin", "pair"])["prem_z"]
                    .apply(lambda s: float(s.iloc[-1]))
                    .sort_values(key=lambda s: np.abs(s), ascending=False)
                    .head(10)
                    .to_dict())
        kpi = {
            "latest_timestamp_utc": None if pd.isna(latest_ts) else str(latest_ts),
            "n_pairs": int(pairs_df.shape[0]),
            "n_pairs_with_data": int(premia.groupby(["isin", "exA", "exB"]).ngroups),
            "top_abs_z_pairs": wid,
        }
    else:
        kpi = {"latest_timestamp_utc": None, "n_pairs": int(pairs_df.shape[0]), "n_pairs_with_data": 0}

    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        listings=args.listings, prices=args.prices, fx=args.fx, calendar=args.calendar or None,
        corporate=args.corporate or None, freq=args.freq, win=args.win, base=args.base,
        min_overlap_mins=args.min_overlap_mins, pair_limit=args.pair_limit, outdir=args.outdir
    )), indent=2))

    # Console
    print("== EU Cross-Listings ==")
    print(f"Pairs: {kpi['n_pairs']}  With data: {kpi['n_pairs_with_data']}")
    if kpi.get("latest_timestamp_utc"):
        print(f"Latest: {kpi['latest_timestamp_utc']}")
    if kpi.get("top_abs_z_pairs"):
        print("Top abs z-score pairs:", list(kpi["top_abs_z_pairs"].items())[:5])


if __name__ == "__main__":
    main()
