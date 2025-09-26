#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
liquidity_pulse.py — Cross-asset market & funding liquidity monitor

What this does
--------------
Given OHLCV (and optionally quotes) for a set of tickers, this script builds a
daily "Liquidity Pulse" for each asset and an aggregate cross-section index.

Per-ticker metrics
  • Amihud illiquidity  = 1e6 * |return| / dollar_volume
  • Roll implied spread = 2 * sqrt(max(0, -E[r_t * r_{t-1}]))      (63d window)
  • High–Low spread     = 2*(H-L)/(H+L)                            (daily)
  • Quoted spread       = (ask-bid)/mid                            (if quotes given)
  • Zero-return share   = % of days with |r| < 1bp                 (63d window)
  • Volume z-score      = z(log(volume), 252d window)

Each metric → rolling z-scores (252d); combine into a Liquidity Pulse Index (LPI):
    LPI = 0.35*z(Amihud) + 0.25*z(Roll) + 0.25*z(HiLo) + 0.10*z(ZeroRet) - 0.05*z(Volume)

Aggregate indices
  • MLI_EQW: equal-weight average of per-ticker LPI
  • MLI_CAPW: market-cap-weighted average (if meta with mcap_usd is provided)
  • Breadth: % of tickers with LPI above +1σ (stress breadth)

Optional funding stress (risk.csv: long format: date, series, value)
  • Computes per-series z-scores and their average FUNDING_Z
  • Outputs combined LIQUIDITY_STRESS = 0.7*MLI_EQW_Z + 0.3*FUNDING_Z

Inputs (CSV; case-insensitive)
------------------------------
--ohlcv ohlcv.csv      REQUIRED (long)
  Columns (min): date, ticker, close, volume
  Optional: open, high, low, dollar_volume
  Notes: If dollar_volume missing, uses volume*close.

--quotes quotes.csv    OPTIONAL (long)
  Columns: date, ticker, bid, ask

--meta meta.csv        OPTIONAL
  Columns: ticker, mcap_usd [, asset_class, region, shares_outstanding]

--risk risk.csv        OPTIONAL
  Columns: date, series, value  (e.g., TED, FRA_OIS, MOVE, FX_BASIS, GC_REPO)

CLI
---
--lookback 252 --short 63 --outdir out_liquidity --calib_start 2018-01-01 --calib_end 2019-12-31

Outputs
-------
- metrics_panel.csv     Per date×ticker metrics & z-scores incl. LPI
- aggregate.csv         Cross-asset indices (MLI_EQW, MLI_CAPW, breadth, funding_z, combined)
- latest_snapshot.csv   Latest per-ticker snapshot sorted by worst LPI
- summary.json          Headline metrics (current stress, worst tickers)
- config.json           Run configuration

DISCLAIMER: Research tooling. Estimators are proxies; validate before production use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan).astype(float)).diff()

def roll_z(s: pd.Series, window: int=252, minp: int=60) -> pd.Series:
    """Rolling z-score per series."""
    m = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std(ddof=0)
    z = (s - m) / sd.replace(0, np.nan)
    return z

def cross_z(df_row: pd.Series) -> pd.Series:
    """Cross-sectional z per date (row)."""
    mu = df_row.mean()
    sd = df_row.std(ddof=0)
    return (df_row - mu) / (sd if sd and sd>0 else np.nan)


# ----------------------------- loaders -----------------------------

def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Flexible rename
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"ticker") or "ticker"): "ticker",
        (ncol(df,"open") or "open"): "open",
        (ncol(df,"high") or "high"): "high",
        (ncol(df,"low") or "low"): "low",
        (ncol(df,"close") or ncol(df,"price") or "close"): "close",
        (ncol(df,"volume") or "volume"): "volume",
        (ncol(df,"dollar_volume") or ncol(df,"dollar_value") or "dollar_volume"): "dollar_volume",
    }
    df = df.rename(columns=ren)
    if "ticker" not in df.columns:
        raise ValueError("ohlcv must be in LONG format with a 'ticker' column.")
    df["date"] = to_date(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for c in ["open","high","low","close","volume","dollar_volume"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    # Fill dollar_volume if missing
    if "dollar_volume" in df.columns:
        missing = df["dollar_volume"].isna()
        if missing.any():
            df.loc[missing, "dollar_volume"] = df.loc[missing, "volume"] * df.loc[missing, "close"]
    else:
        df["dollar_volume"] = df["volume"] * df["close"]
    df = df.sort_values(["ticker","date"])
    return df

def load_quotes(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"ticker") or "ticker"): "ticker",
        (ncol(df,"bid") or "bid"): "bid",
        (ncol(df,"ask") or "ask"): "ask",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for c in ["bid","ask"]:
        df[c] = safe_num(df[c])
    df = df.dropna(subset=["bid","ask"])
    return df.sort_values(["ticker","date"])

def load_meta(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"ticker") or "ticker"): "ticker",
        (ncol(df,"mcap_usd") or ncol(df,"market_cap") or "mcap_usd"): "mcap_usd",
        (ncol(df,"asset_class") or "asset_class"): "asset_class",
        (ncol(df,"region") or "region"): "region",
        (ncol(df,"shares_outstanding") or "shares_outstanding"): "shares_outstanding",
    }
    df = df.rename(columns=ren)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "mcap_usd" in df.columns:
        df["mcap_usd"] = safe_num(df["mcap_usd"])
    return df

def load_risk(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"date") or df.columns[0]): "date",
        (ncol(df,"series") or "series"): "series",
        (ncol(df,"value") or "value"): "value",
    }
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["series"] = df["series"].astype(str).str.upper().str.strip()
    df["value"] = safe_num(df["value"])
    return df.sort_values(["series","date"])


# ----------------------------- metrics -----------------------------

def compute_metrics_for_ticker(g: pd.DataFrame, short: int=63, long: int=252) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    # Returns & volumes
    g["ret"] = dlog(g["close"])
    g["abs_ret"] = g["ret"].abs()
    g["dlv"] = g["dollar_volume"].replace(0, np.nan)
    g["amihud"] = 1e6 * (g["abs_ret"] / g["dlv"])  # 1e6 × (|r|/$volume)
    # Roll implied spread (use mean of r_t * r_{t-1} as simple autocov proxy)
    r = g["ret"]
    gamma = (r * r.shift(1)).rolling(short, min_periods=max(20, short//2)).mean()
    roll = 2.0 * np.sqrt(np.maximum(0.0, -gamma))
    g["roll_spread"] = roll  # in return units ~ percent (for small r)
    # High–Low relative spread
    if {"high","low"} <= set(g.columns):
        g["hl_spread"] = 2.0 * (g["high"] - g["low"]) / (g["high"] + g["low"]).replace(0, np.nan)
    else:
        g["hl_spread"] = np.nan
    # Zero-return share (|ret| < 1bp)
    zflag = (g["abs_ret"] < 0.0001).astype(float)
    g["zero_ret_share"] = zflag.rolling(short, min_periods=max(20, short//2)).mean()
    # Volume z-score (log volume)
    g["log_vol"] = np.log(g["volume"].replace(0, np.nan))
    g["volume_z"] = roll_z(g["log_vol"], window=long, minp=max(60, long//4))
    # Z-scores of spreads & amihud & zero_ret_share
    g["amihud_z"] = roll_z(g["amihud"], window=long, minp=max(60, long//4))
    g["roll_z"]   = roll_z(g["roll_spread"], window=long, minp=max(60, long//4))
    g["hl_z"]     = roll_z(g["hl_spread"], window=long, minp=max(60, long//4))
    g["zret_z"]   = roll_z(g["zero_ret_share"], window=long, minp=max(60, long//4))
    # Liquidity Pulse Index (higher = worse liquidity)
    g["LPI"] = (0.35*g["amihud_z"] + 0.25*g["roll_z"] + 0.25*g["hl_z"] + 0.10*g["zret_z"] - 0.05*g["volume_z"])
    return g

def attach_quoted_spreads(panel: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty: 
        panel["quoted_spread"] = np.nan
        panel["quoted_z"] = np.nan
        return panel
    q = quotes.copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2.0
    q["quoted_spread"] = (q["ask"] - q["bid"]) / q["mid"].replace(0, np.nan)
    q = q[["date","ticker","quoted_spread"]]
    out = panel.merge(q, on=["date","ticker"], how="left")
    # rolling z by ticker
    out["quoted_z"] = (out.groupby("ticker", group_keys=False)
                         .apply(lambda x: roll_z(x["quoted_spread"], window=252, minp=60)))
    # optionally blend quoted_z into LPI (mildly)
    out["LPI"] = out["LPI"] + 0.05 * out["quoted_z"]
    return out

def aggregate_cross_section(panel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Build cross-asset aggregates per date."""
    # Equal-weight
    piv = panel.pivot_table(index="date", columns="ticker", values="LPI", aggfunc="last")
    mli_eqw = piv.mean(axis=1)
    # Market-cap weighted if available
    if not meta.empty and "mcap_usd" in meta.columns:
        w = meta.set_index("ticker")["mcap_usd"].replace({0: np.nan})
        w = w / w.sum()
        common = [t for t in piv.columns if t in w.index and pd.notna(w[t])]
        if common:
            mli_capw = (piv[common] * w.reindex(common)).sum(axis=1)
        else:
            mli_capw = mli_eqw.copy()
    else:
        mli_capw = mli_eqw.copy()
    # Breadth: share of tickers with LPI > +1σ cross-sectionally that day
    cs_z = piv.apply(cross_z, axis=1)
    breadth = (cs_z > 1.0).sum(axis=1) / cs_z.count(axis=1)

    agg = pd.DataFrame({"MLI_EQW": mli_eqw, "MLI_CAPW": mli_capw, "BREADTH_GT1": breadth})
    return agg

def add_funding_stress(agg: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    if risk.empty:
        agg["FUNDING_Z"] = np.nan
        agg["LIQUIDITY_STRESS"] = roll_z(agg["MLI_EQW"], window=252, minp=60)  # z of MLI as fallback
        return agg
    # Per-series z on their own history
    Z = []
    for s, g in risk.groupby("series"):
        zz = g[["date","value"]].copy()
        zz["z"] = roll_z(zz["value"], window=252, minp=60)
        zz["series"] = s
        Z.append(zz)
    Z = pd.concat(Z, ignore_index=True) if Z else pd.DataFrame(columns=["date","series","z"])
    z_piv = Z.pivot_table(index="date", columns="series", values="z", aggfunc="last")
    funding_z = z_piv.mean(axis=1)
    out = agg.merge(funding_z.rename("FUNDING_Z"), left_index=True, right_index=True, how="left")
    # Z of MLI_EQW
    mli_z = roll_z(out["MLI_EQW"], window=252, minp=60)
    out["LIQUIDITY_STRESS"] = 0.7 * mli_z + 0.3 * out["FUNDING_Z"]
    return out


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    ohlcv: str
    quotes: Optional[str]
    meta: Optional[str]
    risk: Optional[str]
    lookback: int
    short: int
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Liquidity Pulse — per-asset & cross-asset liquidity monitor")
    ap.add_argument("--ohlcv", required=True, help="LONG ohlcv file: date,ticker,open,high,low,close,volume[,dollar_volume]")
    ap.add_argument("--quotes", default="", help="Quotes file: date,ticker,bid,ask")
    ap.add_argument("--meta", default="", help="Meta file: ticker,mcap_usd[,asset_class,region]")
    ap.add_argument("--risk", default="", help="Funding risk series: date,series,value")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--short", type=int, default=63)
    ap.add_argument("--outdir", default="out_liquidity")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load
    OHLCV = load_ohlcv(args.ohlcv)
    QUOTES = load_quotes(args.quotes) if args.quotes else pd.DataFrame()
    META   = load_meta(args.meta) if args.meta else pd.DataFrame()
    RISK   = load_risk(args.risk) if args.risk else pd.DataFrame()

    # Compute per-ticker metrics
    pieces = []
    for t, g in OHLCV.groupby("ticker"):
        m = compute_metrics_for_ticker(g, short=args.short, long=args.lookback)
        pieces.append(m.assign(ticker=t))
    panel = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if panel.empty:
        raise ValueError("No metrics computed. Check --ohlcv formatting.")

    # Attach quoted spreads (if available)
    panel = attach_quoted_spreads(panel, QUOTES)

    # Tidy & write panel
    keep_cols = ["date","ticker","amihud","roll_spread","hl_spread","quoted_spread",
                 "zero_ret_share","volume_z","amihud_z","roll_z","hl_z","zret_z","quoted_z","LPI"]
    for c in keep_cols:
        if c not in panel.columns: panel[c] = np.nan
    panel_out = panel[keep_cols].sort_values(["date","ticker"])
    panel_out.to_csv(outdir / "metrics_panel.csv", index=False)

    # Aggregates
    agg = aggregate_cross_section(panel, META)
    agg = add_funding_stress(agg, RISK)
    agg.to_csv(outdir / "aggregate.csv", index=True)

    # Latest snapshot
    last_date = panel["date"].max()
    snap = (panel[panel["date"]==last_date]
            .merge(META[["ticker","mcap_usd"]] if not META.empty else pd.DataFrame(columns=["ticker","mcap_usd"]),
                   on="ticker", how="left")
            .sort_values("LPI", ascending=False))
    snap_cols = ["ticker","LPI","amihud","roll_spread","hl_spread","quoted_spread","zero_ret_share","mcap_usd"]
    snap[snap_cols].to_csv(outdir / "latest_snapshot.csv", index=False)

    # Summary
    worst = snap.head(10)[["ticker","LPI","amihud","roll_spread","hl_spread","quoted_spread"]].to_dict(orient="records")
    best  = snap.tail(10)[["ticker","LPI","amihud","roll_spread","hl_spread","quoted_spread"]].to_dict(orient="records")
    summary = {
        "last_date": str(pd.to_datetime(last_date).date()) if pd.notna(last_date) else None,
        "mli_eqw_latest": float(agg["MLI_EQW"].dropna().iloc[-1]) if not agg.empty else None,
        "mli_capw_latest": float(agg["MLI_CAPW"].dropna().iloc[-1]) if not agg.empty else None,
        "funding_z_latest": float(agg["FUNDING_Z"].dropna().iloc[-1]) if "FUNDING_Z" in agg.columns and agg["FUNDING_Z"].notna().any() else None,
        "liquidity_stress_latest": float(agg["LIQUIDITY_STRESS"].dropna().iloc[-1]) if "LIQUIDITY_STRESS" in agg.columns and agg["LIQUIDITY_STRESS"].notna().any() else None,
        "worst_10": worst,
        "best_10": best
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        ohlcv=args.ohlcv, quotes=(args.quotes or None), meta=(args.meta or None), risk=(args.risk or None),
        lookback=args.lookback, short=args.short, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console logs
    print("== Liquidity Pulse ==")
    print(f"Date: {summary['last_date']} | MLI_EQW: {summary['mli_eqw_latest']:+.2f} | FundingZ: {summary['funding_z_latest'] if summary['funding_z_latest'] is not None else 'NA'}")
    if worst:
        print("Worst tickers (by LPI):", [w['ticker'] for w in worst])

if __name__ == "__main__":
    main()
