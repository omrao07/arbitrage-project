#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eu_dividend_futures.py — Analytics for Euro-area equity dividend futures
(works for STOXX/Euro STOXX 50 dividend futures, or any annual dividend future curve)

What it does
------------
- Ingests dividend futures quotes (per expiry year), optional analyst/house forecasts,
  realized dividends, and (optionally) a risk-free curve.
- Normalizes contract metadata → dividend year, time-to-settlement, liquidity stats.
- Builds an annual dividend term structure (market-implied), computes:
    * Roll yields between adjacent years
    * Curve carry (change of “current” contract weight as year advances)
    * Implied long-run growth from a simple growth model
    * Fair-value checks vs bottom-up forecasts (with mid-year discounting option)
    * Z-scores/percentiles vs your own history (if provided as a time series)
- Produces tidy CSVs and a JSON summary.

Assumptions & notes
-------------------
- Dividend futures settle to the SUM of ordinary cash dividends paid during the calendar year.
- Price is in index points ≈ EUR, i.e., 120.5 ≈ €120.5 per index “notional”.
- Under risk-neutral pricing, the futures level is close to the expected sum (no equity carry).
  We optionally apply mid-year discounting to align *forecast* cashflows with a valuation date.
- Supports either “one row per contract per day” or latest snapshot.

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--futures futures.csv        REQUIRED
  Columns (suggested):
    date, contract, expiry, year, price, bid, ask, volume, open_interest, exchange
  - If 'year' missing, it will be inferred from 'expiry' (YYYY or YYYY-MM-DD).

--realized realized.csv      OPTIONAL (historical realized dividends by year)
  Columns: year, realized_div

--forecast forecast.csv      OPTIONAL (your/consensus dividend year forecasts)
  Columns: year, forecast_div

--curve rates.csv            OPTIONAL (OIS/risk-free, to discount forecasts mid-year)
  Columns: date, tenor_yrs, rate (decimal, e.g., 0.025 = 2.5%)

--history history.csv        OPTIONAL (historical daily close of dividend futures per year)
  Columns: date, year, price
  Used for z-scores/percentiles on the current snapshot.

Key options
-----------
--asof 2025-09-06            Valuation date; defaults to max(futures.date) if present
--midyear_discount 1         If 1 and curve given, discount forecast dividends by ~0.5y
--min_oi 0                   Filter contracts with OI below this (default keep all)
--outdir out_divs

Outputs
-------
- futures_clean.csv          Normalized latest snapshot by year with best price, bid/ask, OI, vol
- curve_by_year.csv          Market curve vs realized/forecast and fair value diffs
- rolls.csv                  Adjacent-year roll metrics (yield, calendar basis)
- growth_implied.csv         Simple growth-fit stats (CAGR between Y1 and long tail, regression)
- zscores.csv                (if history) z-score/percentiles for each active year
- summary.json               Key KPIs
- config.json                Reproducibility dump

Example
-------
python eu_dividend_futures.py --futures futures.csv --forecast forecast.csv --realized realized.csv --curve rates.csv \
  --asof 2025-09-06 --outdir out_divs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------
def ncol(df: pd.DataFrame, name: str) -> Optional[str]:
    t = name.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def midyear_df(year: int, asof: pd.Timestamp) -> float:
    """
    Approximate year-fraction from asof to mid-year of 'year' (July 1st).
    Negative if in the past.
    """
    if pd.isna(asof): return np.nan
    mid = pd.Timestamp(year=year, month=7, day=1)
    return max(0.0, (mid - asof).days / 365.25)


# ----------------------------- loaders -----------------------------
def load_futures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize columns
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"contract") or "contract"): "contract",
        (ncol(df,"expiry") or "expiry"): "expiry",
        (ncol(df,"year") or "year"): "year",
        (ncol(df,"price") or "price"): "price",
        (ncol(df,"bid") or "bid"): "bid",
        (ncol(df,"ask") or "ask"): "ask",
        (ncol(df,"volume") or "volume"): "volume",
        (ncol(df,"open_interest") or ncol(df,"oi") or "open_interest"): "open_interest",
        (ncol(df,"exchange") or "exchange"): "exchange",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    # infer year if missing
    if "year" not in df.columns or df["year"].isna().all():
        if "expiry" in df.columns:
            exp = df["expiry"].astype(str)
            # try YYYY first 4 digits
            df["year"] = exp.str.extract(r"(\d{4})")[0]
        else:
            raise SystemExit("Futures: need 'year' or 'expiry' to infer year.")
    df["year"] = num(df["year"]).astype("Int64")
    for c in ["price","bid","ask","volume","open_interest"]:
        if c in df.columns: df[c] = num(df[c])
    return df


def load_realized(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["year","realized_div"])
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df,"year") or "year"):"year",
        (ncol(df,"realized_div") or ncol(df,"realised_div") or "realized_div"):"realized_div",
    })
    df["year"] = num(df["year"]).astype("Int64")
    df["realized_div"] = num(df["realized_div"])
    return df


def load_forecast(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["year","forecast_div"])
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df,"year") or "year"):"year",
        (ncol(df,"forecast_div") or ncol(df,"forecast") or "forecast_div"):"forecast_div",
    })
    df["year"] = num(df["year"]).astype("Int64")
    df["forecast_div"] = num(df["forecast_div"])
    return df


def load_curve(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["date","tenor_yrs","rate"])
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"tenor_yrs") or ncol(df,"tenor") or "tenor_yrs"): "tenor_yrs",
        (ncol(df,"rate") or ncol(df,"yield") or "rate"): "rate",
    })
    df["date"] = to_date(df["date"])
    df["tenor_yrs"] = num(df["tenor_yrs"])
    df["rate"] = num(df["rate"])
    return df


def load_history(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"year") or "year"): "year",
        (ncol(df,"price") or "price"): "price",
    })
    df["date"] = to_date(df["date"])
    df["year"] = num(df["year"]).astype("Int64")
    df["price"] = num(df["price"])
    return df


# ----------------------------- core analytics -----------------------------
def best_quote(row) -> float:
    px = row.get("price", np.nan)
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    if np.isfinite(px): return float(px)
    if np.isfinite(bid) and np.isfinite(ask): return float((bid + ask)/2.0)
    if np.isfinite(bid): return float(bid)
    if np.isfinite(ask): return float(ask)
    return np.nan


def snapshot_by_year(futs: pd.DataFrame, asof: Optional[pd.Timestamp], min_oi: int) -> Tuple[pd.Timestamp, pd.DataFrame]:
    # pick asof = max available date if not provided
    if asof is None or pd.isna(asof):
        asof = futs["date"].max()
    snap = futs[futs["date"] == asof].copy()
    if snap.empty:
        # fallback to last for each year
        snap = futs.sort_values("date").groupby("year").tail(1)
        asof = snap["date"].max()
    # collapse duplicates per year with liquidity preference
    snap["best_px"] = snap.apply(best_quote, axis=1)
    snap["liq_score"] = snap["open_interest"].fillna(0) * 10 + snap["volume"].fillna(0)
    best = (snap.sort_values(["year","liq_score","best_px"], ascending=[True, False, False])
                .groupby("year", as_index=False)
                .first())
    if min_oi > 0:
        best = best[(best["open_interest"].fillna(0) >= min_oi)]
    best = best[["year","best_px","bid","ask","open_interest","volume","exchange","date"]].rename(columns={"date":"asof"})
    return pd.Timestamp(asof), best.sort_values("year")


def discount_factor(tenor_yrs: float, curve: pd.DataFrame, asof: pd.Timestamp) -> float:
    """
    Simple linear interpolation on curve for DF = exp(-r*T).
    If curve empty → DF=1.
    """
    if curve.empty or not np.isfinite(tenor_yrs) or tenor_yrs <= 0:
        return 1.0
    c = curve[curve["date"] == curve["date"].max()]
    if c.empty:
        c = curve
    x = c["tenor_yrs"].values
    y = c["rate"].values
    if len(x) == 0: return 1.0
    r = np.interp(tenor_yrs, x, y, left=y[0], right=y[-1])
    return float(np.exp(-r * tenor_yrs))


def implied_growth_simple(curve_df: pd.DataFrame) -> dict:
    """
    Two quick-and-dirty growth diagnostics:
    - CAGR between front year and a far year (if available)
    - OLS log(price) ~ a + b*year_index → annualized growth ≈ exp(b)-1
    """
    if curve_df.empty or curve_df.shape[0] < 2:
        return {"cagr_front_to_tail": np.nan, "ols_growth": np.nan}
    d = curve_df.dropna(subset=["best_px"]).copy()
    d = d.sort_values("year")
    if len(d) < 2: 
        return {"cagr_front_to_tail": np.nan, "ols_growth": np.nan}
    y0 = d["year"].iloc[0]; yN = d["year"].iloc[-1]
    v0 = d["best_px"].iloc[0]; vN = d["best_px"].iloc[-1]
    span = max(1, yN - y0)
    cagr = float((vN / max(1e-9, v0)) ** (1.0 / span) - 1.0)
    # OLS on logs
    X = (d["year"] - y0).values.astype(float)
    Y = np.log(d["best_px"].values.clip(min=1e-9))
    b = float(np.cov(X, Y, ddof=0)[0,1] / max(1e-12, np.var(X)))
    ols_g = float(np.exp(b) - 1.0)
    return {"cagr_front_to_tail": cagr, "ols_growth": ols_g}


def build_rolls(curve_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjacent-year roll metrics:
    - roll_yield = (Y2 - Y1) / Y1
    - calendar_basis = Y1 - realized_{Y0} (if available later via join)
    """
    d = curve_df.sort_values("year").reset_index(drop=True)
    if d.empty: return pd.DataFrame(columns=["year_from","year_to","roll_yield"])
    rows = []
    for i in range(len(d)-1):
        y1 = int(d.loc[i, "year"]); y2 = int(d.loc[i+1, "year"])
        p1 = float(d.loc[i, "best_px"]); p2 = float(d.loc[i+1, "best_px"])
        rows.append({"year_from": y1, "year_to": y2, "roll_yield": (p2 - p1) / max(1e-9, p1)})
    return pd.DataFrame(rows)


def fair_value_vs_forecast(curve_df: pd.DataFrame, fc: pd.DataFrame, realized: pd.DataFrame,
                           curve_rates: pd.DataFrame, asof: pd.Timestamp, midyear_discount: bool) -> pd.DataFrame:
    """
    Compare market-implied dividend level vs your forecast or realized:
    - If midyear_discount: fair_value_forecast = forecast_div * DF(0.5y to mid-year)
    - diff = market - fair_value, pct_diff = diff / fair_value
    """
    d = curve_df.merge(fc, on="year", how="left").merge(realized, on="year", how="left")
    d["df_mid"] = 1.0
    if midyear_discount and not curve_rates.empty:
        d["df_mid"] = [discount_factor(midyear_df(int(y), asof), curve_rates, asof) for y in d["year"]]
    d["fair_forecast"] = d["forecast_div"] * d["df_mid"]
    d["fair_realized"] = d["realized_div"]  # no discounting; already known
    d["diff_vs_forecast"] = d["best_px"] - d["fair_forecast"]
    d["pct_vs_forecast"] = d["diff_vs_forecast"] / d["fair_forecast"].replace(0, np.nan)
    d["diff_vs_realized"] = d["best_px"] - d["fair_realized"]
    d["pct_vs_realized"] = d["diff_vs_realized"] / d["fair_realized"].replace(0, np.nan)
    return d


def zscores_today(history: pd.DataFrame, asof: pd.Timestamp, today_curve: pd.DataFrame) -> pd.DataFrame:
    """
    For each year, compute z-score of today's price vs trailing distribution in 'history'.
    """
    if history.empty: 
        return pd.DataFrame(columns=["year","today","mean","std","z","pctl"])
    out = []
    for _, r in today_curve.iterrows():
        y = int(r["year"]); today = float(r["best_px"])
        h = history[history["year"] == y]["price"].dropna()
        if len(h) < 10:
            out.append({"year": y, "today": today, "mean": np.nan, "std": np.nan, "z": np.nan, "pctl": np.nan})
            continue
        mu = float(h.mean()); sd = float(h.std(ddof=1))
        z = (today - mu) / (sd + 1e-12)
        pctl = float((h < today).mean())
        out.append({"year": y, "today": today, "mean": mu, "std": sd, "z": z, "pctl": pctl})
    return pd.DataFrame(out).sort_values("year")


# ----------------------------- CLI -----------------------------
@dataclass
class Config:
    futures: str
    realized: Optional[str]
    forecast: Optional[str]
    curve: Optional[str]
    history: Optional[str]
    asof: Optional[str]
    midyear_discount: bool
    min_oi: int
    outdir: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EU dividend futures analytics")
    ap.add_argument("--futures", required=True)
    ap.add_argument("--realized", default="")
    ap.add_argument("--forecast", default="")
    ap.add_argument("--curve", default="")
    ap.add_argument("--history", default="")
    ap.add_argument("--asof", default="")
    ap.add_argument("--midyear_discount", type=int, default=1)
    ap.add_argument("--min_oi", type=int, default=0)
    ap.add_argument("--outdir", default="out_divs")
    return ap.parse_args()


# ----------------------------- main -----------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    futs = load_futures(args.futures)
    realized = load_realized(args.realized)
    forecast = load_forecast(args.forecast)
    curve_rates = load_curve(args.curve)
    hist = load_history(args.history)

    asof = pd.to_datetime(args.asof) if args.asof else pd.NaT
    asof, curve = snapshot_by_year(futs, asof, args.min_oi)

    # Growth diagnostics
    growth = implied_growth_simple(curve)

    # Rolls
    rolls = build_rolls(curve)

    # Fair value vs forecasts/realized
    fair_tbl = fair_value_vs_forecast(curve, forecast, realized, curve_rates, asof, bool(args.midyear_discount))

    # Z-scores using history
    ztbl = zscores_today(hist, asof, curve) if not hist.empty else pd.DataFrame()

    # Write outputs
    curve.to_csv(outdir / "futures_clean.csv", index=False)
    fair_tbl.to_csv(outdir / "curve_by_year.csv", index=False)
    if not rolls.empty: rolls.to_csv(outdir / "rolls.csv", index=False)
    if not ztbl.empty: ztbl.to_csv(outdir / "zscores.csv", index=False)

    # Summary
    kpi = {
        "asof": str(asof.date()) if pd.notna(asof) else None,
        "years": curve["year"].tolist(),
        "front_year": int(curve["year"].min()) if not curve.empty else None,
        "front_price": float(curve.loc[curve["year"].idxmin(), "best_px"]) if not curve.empty else None,
        "tail_year": int(curve["year"].max()) if not curve.empty else None,
        "tail_price": float(curve.loc[curve["year"].idxmax(), "best_px"]) if not curve.empty else None,
        "implied_growth_cagr": growth["cagr_front_to_tail"],
        "implied_growth_ols": growth["ols_growth"],
        "avg_roll_yield": float(rolls["roll_yield"].mean()) if not rolls.empty else None,
    }
    (outdir / "summary.json").write_text(json.dumps(kpi, indent=2))
    (outdir / "config.json").write_text(json.dumps(asdict(Config(
        futures=args.futures, realized=args.realized or None, forecast=args.forecast or None,
        curve=args.curve or None, history=args.history or None,
        asof=str(asof.date()) if pd.notna(asof) else None,
        midyear_discount=bool(args.midyear_discount), min_oi=args.min_oi, outdir=args.outdir
    )), indent=2))

    # Console
    print("== EU Dividend Futures ==")
    print(f"As of {kpi['asof']}: years {kpi['years']}")
    if kpi["front_year"] is not None:
        print(f"Front {kpi['front_year']}: {kpi['front_price']:.2f} | Tail {kpi['tail_year']}: {kpi['tail_price']:.2f}")
    if kpi["implied_growth_cagr"] is not None:
        print(f"Implied growth (CAGR): {kpi['implied_growth_cagr']*100:.2f}%  |  OLS: {kpi['implied_growth_ols']*100:.2f}%")
    if kpi["avg_roll_yield"] is not None:
        print(f"Avg roll yield: {kpi['avg_roll_yield']*100:.2f}%")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
