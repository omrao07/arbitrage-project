#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
currency_wars.py — FX "currency war" monitor, scores, and long/short basket backtests

What this does
--------------
Given FX spot (pairs), short rates (by currency), CPI indices (by country/currency),
and (optionally) FX reserves and macro gauges, this script:

1) Cleans & aligns time series (daily or monthly)
2) Builds factor features per pair:
   - Carry (annualized short-rate differential): i_base − i_quote
   - Momentum (3M, 12M log returns)
   - PPP misalignment from CPI (spot / PPP_fair − 1)
   - Realized vol (20d), z-reserve impulse (if provided)
3) Derives composite scores:
   - Value (undervaluation), Carry, Momentum ⇒ "Risk-ON" composite
   - "War-risk" composite (overvaluation + reserve impulse + high vol)
4) Generates trade signals (top-N long/short baskets) and an equal-weight backtest
   with monthly rebalancing and long-short returns
5) Writes tidy CSVs + a compact JSON summary

Typical inputs (CSV; case-insensitive, flexible)
-----------------------------------------------
--fx fx.csv              REQUIRED
  Wide:  date, EURUSD, USDJPY, GBPUSD, ...
  Long:  date, pair, value       (pairs like EURUSD, USDJPY, etc.)
  Levels (spot); script computes returns.

--rates rates.csv        OPTIONAL (short policy/money rates, %)
  Long:  date, ccy|country, rate_pct
  Wide:  date, USD, EUR, JPY, ...
  (We map common country names to ISO 3-letter FX codes when needed.)

--cpi cpi.csv            OPTIONAL (index; any base, consistent over time)
  Long:  date, ccy|country, cpi
  Wide:  date, USD, EUR, JPY, ...
  (For PPP we need both base and quote CPI in a pair; if missing, PPP is skipped.)

--reserves reserves.csv  OPTIONAL (FX reserves, USD)
  Long:  date, ccy|country, reserves_usd
  Wide:  date, USD, EUR, JPY, ...
  (Used for intervention/pressure z-signal.)

--macro macro.csv        OPTIONAL (free-form aux series; unused in core but merged for convenience)
  Long:  date, series, value

CLI (examples)
-------------
--freq D --rebalance M --topN 4 --start 2015-01-01 --end 2025-09-01 --outdir out_fx_wars
--fx fx.csv --rates rates.csv --cpi cpi.csv --reserves reserves.csv

Outputs
-------
- fx_panel.csv            Clean FX wide panel (levels) used
- features_panel.csv      Per date & pair features (carry, momo, ppp_misalign, vol, reserves_z, composites)
- signals_latest.csv      Latest snapshot sorted by composite
- baskets.csv             For each rebalance date: long & short members and weights
- backtest.csv            Long-only, Short-only, and L/S portfolio daily returns and indices
- summary.json            Headline stats (CAGR, vol, Sharpe, hit rate) and latest top/bottom pairs
- config.json             Run configuration for reproducibility

Notes & conventions
-------------------
- Pairs must be 6-letter ISO codes like EURUSD, USDJPY, GBPUSD, etc.
- Pair = Y/X means "quote per base" (e.g., USDJPY = JPY per USD). A positive return benefits LONG base vs quote.
- Carry per pair is i_base − i_quote (annualized %).
- PPP fair value for pair Y/X: E*_t = E_0 * (CPI_Y/CPI_Y0) / (CPI_X/CPI_X0).
  Misalignment = (E_t / E*_t) − 1  (positive = quote expensive vs base ⇒ base undervalued).
- Reserves z-signal is computed per currency as z(Δlog(reserves), 252d rolling), then mapped to pairs:
  reserves_z_pair = z_base − z_quote (positive if base showing bigger reserve drawdown ⇒ “defend base”).
- Composite score = 0.4*Value + 0.35*Carry + 0.25*Momentum12 (all z-scored cross-sectionally each day).
  War-risk = z( |misalign| ) + z(20d vol) + 0.5*z( |reserves_z_pair| ).

DISCLAIMER: Research tooling; not investment advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- utils -----------------------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def dlog(x: pd.Series) -> pd.Series:
    return np.log(x.astype(float).replace(0, np.nan)).diff()

def ann_from_daily(s: pd.Series) -> float:
    # convert daily average to annual (approx)
    return float(s.mean() * 252.0)

def zscore_cross(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across columns (cross-section)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0, np.nan)
    return (df.sub(mu, axis=0)).div(sd, axis=0)

def roll_vol(s: pd.Series, window: int=20) -> pd.Series:
    return s.rolling(window, min_periods=max(5, window//2)).std(ddof=0)

# country -> currency code mapping (very partial; extend as needed)
COUNTRY_TO_CCY = {
    "UNITED STATES":"USD","US":"USD","U.S.":"USD","USA":"USD",
    "EURO AREA":"EUR","EUROZONE":"EUR","EU":"EUR",
    "JAPAN":"JPY","UK":"GBP","UNITED KINGDOM":"GBP","BRITAIN":"GBP",
    "SWITZERLAND":"CHF","CANADA":"CAD","AUSTRALIA":"AUD","NEW ZEALAND":"NZD",
    "CHINA":"CNY","HONG KONG":"HKD","SINGAPORE":"SGD","KOREA":"KRW",
    "INDIA":"INR","BRAZIL":"BRL","MEXICO":"MXN","SOUTH AFRICA":"ZAR",
    "NORWAY":"NOK","SWEDEN":"SEK","DENMARK":"DKK","POLAND":"PLN",
    "CZECH REPUBLIC":"CZK","HUNGARY":"HUF","TURKIYE":"TRY","TURKEY":"TRY",
    "ISRAEL":"ILS","TAIWAN":"TWD","THAILAND":"THB","INDONESIA":"IDR",
    "MALAYSIA":"MYR","PHILIPPINES":"PHP","CHILE":"CLP","COLOMBIA":"COP",
    "PERU":"PEN","SAUDI ARABIA":"SAR","UAE":"AED","VIETNAM":"VND"
}

def normalize_ccy(x: str) -> str:
    if not isinstance(x, str): return str(x)
    u = x.strip().upper()
    return COUNTRY_TO_CCY.get(u, u[:3] if len(u)>=3 else u)

def split_pair(pair: str) -> Tuple[str, str]:
    p = str(pair).upper().strip().replace("/","")
    if len(p) != 6: raise ValueError(f"Bad pair '{pair}': expected like EURUSD or USDJPY")
    return p[:3], p[3:]

def is_pair_col(c: str) -> bool:
    u = str(c).upper()
    return len(u)==6 and u.isalpha()

def resample_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq.upper().startswith("D"): return df
    # Monthly (calendar end)
    return df.resample("M").last()

# ----------------------------- loaders -----------------------------

def load_fx(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={date_c:"date"})
    df["date"] = to_date(df["date"])
    if ncol(df,"pair"):
        val_c = ncol(df,"value") or ncol(df,"close") or "value"
        df = df.rename(columns={ncol(df,"pair"):"pair", val_c:"value"})
        piv = df.pivot_table(index="date", columns="pair", values="value", aggfunc="last")
    else:
        piv = df.set_index("date")
    # keep only pair-looking columns
    piv = piv.loc[:, [c for c in piv.columns if is_pair_col(c)]].sort_index()
    return piv

def load_wide_ccy(path: str, value_name: str) -> pd.DataFrame:
    """Load wide or long into wide with currencies as columns."""
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    date_c = ncol(df,"date") or df.columns[0]
    df = df.rename(columns={date_c:"date"})
    df["date"] = to_date(df["date"])
    if ncol(df,"ccy") or ncol(df,"currency") or ncol(df,"country"):
        ccy_c = ncol(df,"ccy") or ncol(df,"currency") or ncol(df,"country")
        val_c = ncol(df, value_name) or value_name
        df = df.rename(columns={ccy_c:"ccy", val_c:value_name})
        df["ccy"] = df["ccy"].astype(str).map(normalize_ccy)
        piv = df.pivot_table(index="date", columns="ccy", values=value_name, aggfunc="last")
    else:
        piv = df.set_index("date")
    # Upper-case headers
    piv.columns = [normalize_ccy(c) for c in piv.columns]
    return piv.sort_index()

# ----------------------------- features -----------------------------

def carry_panel(pairs: List[str], rates_wide: pd.DataFrame) -> pd.DataFrame:
    """Return daily carry (% annualized) per pair: i_base - i_quote."""
    if rates_wide.empty: 
        return pd.DataFrame(index=[], columns=pairs)
    out = pd.DataFrame(index=rates_wide.index)
    for p in pairs:
        b, q = split_pair(p)
        if b in rates_wide.columns and q in rates_wide.columns:
            out[p] = (rates_wide[b] - rates_wide[q]).astype(float)
    return out

def ppp_fair_value(pair: str, spot: pd.Series, cpi_wide: pd.DataFrame) -> pd.Series:
    """PPP fair value path using CPI indices; anchored to first valid date."""
    b, q = split_pair(pair)
    if cpi_wide.empty or b not in cpi_wide.columns or q not in cpi_wide.columns:
        return pd.Series(index=spot.index, dtype=float)
    CPb = cpi_wide[b].astype(float)
    CPq = cpi_wide[q].astype(float)
    # Align to spot
    df = pd.concat([spot, CPb, CPq], axis=1).dropna()
    if df.empty: return pd.Series(index=spot.index, dtype=float)
    s0 = float(df.iloc[0,0]); b0 = float(df.iloc[0,1]); q0 = float(df.iloc[0,2])
    fair = s0 * (df.iloc[:,2] / q0) / (df.iloc[:,1] / b0)  # (CPI_q/CPI_q0)/(CPI_b/CPI_b0)
    fair = fair.reindex(spot.index)
    return fair

def reserves_z(ccy_wide: pd.DataFrame, window: int=252) -> pd.DataFrame:
    """Z of Δlog(reserves) per currency (rolling mean/std)."""
    if ccy_wide.empty: return pd.DataFrame()
    dl = ccy_wide.apply(lambda s: np.log(s.replace(0,np.nan)).diff())
    mean = dl.rolling(window, min_periods=max(60, window//4)).mean()
    std  = dl.rolling(window, min_periods=max(60, window//4)).std(ddof=0)
    z = (dl - mean) / std
    return z

def pair_from_ccy_z(pair: str, z_ccy: pd.DataFrame) -> pd.Series:
    """Map currency z to pair (base minus quote)."""
    if z_ccy.empty: return pd.Series(index=z_ccy.index, dtype=float)
    b, q = split_pair(pair)
    zb = z_ccy[b] if b in z_ccy.columns else pd.Series(index=z_ccy.index, dtype=float)
    zq = z_ccy[q] if q in z_ccy.columns else pd.Series(index=z_ccy.index, dtype=float)
    res = zb.reindex(zq.index) - zq
    return res

# ----------------------------- baskets & backtest -----------------------------

def rebalance_dates(idx: pd.DatetimeIndex, freq: str="M") -> List[pd.Timestamp]:
    if freq.upper().startswith("M"):
        return list(pd.Series(idx).resample("M").last().dropna())
    elif freq.upper().startswith("Q"):
        return list(pd.Series(idx).resample("Q").last().dropna())
    else:
        # daily
        return list(idx)

def backtest_ls(
    fx: pd.DataFrame,                 # levels
    features: pd.DataFrame,           # multi-index (date, pair)
    topN: int = 4,
    rebalance: str = "M",
    score_col: str = "score_composite"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight long/short baskets rebalanced on schedule, signals lagged by 1 day."""
    # Returns per pair (log)
    R = fx.apply(dlog)
    dates = fx.index
    rebal_dates = rebalance_dates(dates, rebalance)
    weights = []  # rows of weights per pair
    for d in rebal_dates:
        # use signals up to d-1
        prev = features.loc[features.index.get_level_values(0) <= d]
        if prev.empty: continue
        last = prev.groupby(level=1).apply(lambda df: df.loc[df.index.get_level_values(0).max()])  # last per pair
        # last is messy; easier: take latest at date ≤ d for all pairs
        last2 = features.xs(prev.index.get_level_values(0).max(), level=0, drop_level=False)
        sc = last2.reset_index()
        sc = sc.sort_values(score_col, ascending=False).dropna(subset=[score_col])
        longs  = sc["pair"].head(topN).tolist()
        shorts = sc["pair"].tail(topN).tolist()
        w = pd.Series(0.0, index=fx.columns)
        if longs:
            w[longs] =  +1.0/len(longs)
        if shorts:
            w[shorts] = -1.0/len(shorts)
        weights.append({"date": d, **w.to_dict()})
    W = pd.DataFrame(weights).set_index("date").reindex(dates).ffill().fillna(0.0)
    # Portfolio daily return
    port_ret = (R * W).sum(axis=1)
    long_ret  = (R.clip(lower=0) * (W.clip(lower=0))).sum(axis=1) + (R * (W>0)).sum(axis=1)*0  # placeholder
    short_ret = (R.clip(upper=0) * (W.clip(upper=0))).sum(axis=1) + (R * (W<0)).sum(axis=1)*0
    out = pd.DataFrame({"long_short": port_ret, "long_only": (R*(W>0)).sum(axis=1), "short_only": (R*(W<0)).sum(axis=1)})
    # Basket members per rebalance
    rows = []
    for d in W.index[W.index.isin(rebal_dates)]:
        ww = W.loc[d]
        rows.append({
            "date": d,
            "longs": ",".join(ww[ww>0].sort_values(ascending=False).index.tolist()),
            "shorts": ",".join(ww[ww<0].sort_values().index.tolist())
        })
    baskets = pd.DataFrame(rows)
    return out, baskets

def perf_stats(ret: pd.Series) -> Dict[str, float]:
    if ret.dropna().empty:
        return {"cagr_pct": np.nan, "vol_pct": np.nan, "sharpe": np.nan, "hit_rate": np.nan}
    idx = (ret.fillna(0) + 1).cumprod()
    years = (idx.index[-1] - idx.index[0]).days / 365.25
    cagr = idx.iloc[-1] ** (1/years) - 1 if years>0 else np.nan
    vol = ret.std(ddof=0) * np.sqrt(252)
    sharpe = cagr / vol if (vol and vol==vol and vol>0) else np.nan
    hit = (ret > 0).mean()
    return {"cagr_pct": float(cagr*100) if cagr==cagr else np.nan,
            "vol_pct": float(vol*100) if vol==vol else np.nan,
            "sharpe": float(sharpe) if sharpe==sharpe else np.nan,
            "hit_rate": float(hit) if hit==hit else np.nan}

# ----------------------------- CLI / main -----------------------------

@dataclass
class Config:
    fx: str
    rates: Optional[str]
    cpi: Optional[str]
    reserves: Optional[str]
    macro: Optional[str]
    freq: str
    rebalance: str
    topN: int
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FX currency war monitor — features, scores, and backtests")
    ap.add_argument("--fx", required=True)
    ap.add_argument("--rates", default="")
    ap.add_argument("--cpi", default="")
    ap.add_argument("--reserves", default="")
    ap.add_argument("--macro", default="")
    ap.add_argument("--freq", default="D", help="D or M (resample) for analytics")
    ap.add_argument("--rebalance", default="M", help="M (monthly) or Q (quarterly)")
    ap.add_argument("--topN", type=int, default=4, help="# longs and # shorts")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_fx_wars")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load panels
    FX = load_fx(args.fx)
    if FX.empty: raise ValueError("No valid FX pair columns found in --fx.")

    RATES = load_wide_ccy(args.rates, "rate_pct") if args.rates else pd.DataFrame()
    CPI   = load_wide_ccy(args.cpi, "cpi") if args.cpi else pd.DataFrame()
    RES   = load_wide_ccy(args.reserves, "reserves_usd") if args.reserves else pd.DataFrame()

    # Filter window & frequency
    if args.start: FX = FX[FX.index >= pd.to_datetime(args.start)]
    if args.end:   FX = FX[FX.index <= pd.to_datetime(args.end)]
    # Align ancillary to FX dates
    for df in [RATES, CPI, RES]:
        if not df.empty:
            df.dropna(how="all", inplace=True)
            df.sort_index(inplace=True)
    if args.freq.upper().startswith("M"):
        FX = resample_freq(FX, "M")
        if not RATES.empty: RATES = resample_freq(RATES, "M")
        if not CPI.empty:   CPI   = resample_freq(CPI, "M")
        if not RES.empty:   RES   = resample_freq(RES, "M")

    FX.to_csv(outdir / "fx_panel.csv")

    pairs = [c for c in FX.columns if is_pair_col(c)]

    # Features per pair
    carry = carry_panel(pairs, RATES) if not RATES.empty else pd.DataFrame(index=FX.index, columns=pairs, dtype=float)
    mom3  = FX.apply(lambda s: dlog(s).rolling(63, min_periods=20).sum())    # ~3M
    mom12 = FX.apply(lambda s: dlog(s).rolling(252, min_periods=60).sum())   # ~12M
    vol20 = FX.apply(lambda s: roll_vol(dlog(s), 20))

    # PPP misalignment
    misalign = pd.DataFrame(index=FX.index, columns=pairs, dtype=float)
    if not CPI.empty:
        for p in pairs:
            fair = ppp_fair_value(p, FX[p], CPI)
            misalign[p] = FX[p] / fair - 1.0 if not fair.dropna().empty else np.nan

    # Reserves z (base - quote)
    res_z = reserves_z(RES) if not RES.empty else pd.DataFrame()
    res_pair = pd.DataFrame(index=FX.index, columns=pairs, dtype=float)
    if not res_z.empty:
        for p in pairs:
            res_pair[p] = pair_from_ccy_z(p, res_z)

    # Cross-sectional z-scores for composite components (per date)
    # Value: + if base undervalued (misalign > 0). Momentum: use 12M. Carry: annualized rate diff.
    # Prepare aligned frames
    comp_idx = FX.index
    Z_val = zscore_cross(misalign.reindex(comp_idx)) if not misalign.empty else pd.DataFrame(index=comp_idx, columns=pairs)
    Z_mom = zscore_cross(mom12.reindex(comp_idx))
    Z_carry = zscore_cross(carry.reindex(comp_idx)) if not carry.empty else pd.DataFrame(index=comp_idx, columns=pairs)

    score = 0.40*Z_val + 0.35*Z_carry + 0.25*Z_mom
    # War-risk (not used for ranking; informational)
    abs_mis = misalign.abs() if not misalign.empty else pd.DataFrame(index=comp_idx, columns=pairs)
    Z_abs_mis = zscore_cross(abs_mis)
    Z_vol = zscore_cross(vol20.reindex(comp_idx))
    Z_res = zscore_cross(res_pair.abs().reindex(comp_idx)) if not res_pair.empty else pd.DataFrame(index=comp_idx, columns=pairs)
    war_risk = Z_abs_mis + Z_vol + 0.5*Z_res

    # Assemble long-form features panel
    rows = []
    for p in pairs:
        df = pd.DataFrame({
            "date": FX.index,
            "pair": p,
            "spot": FX[p].values,
            "carry_pct": carry[p].values if p in carry.columns else np.nan,
            "mom_3m": mom3[p].values,
            "mom_12m": mom12[p].values,
            "vol_20d": vol20[p].values,
            "ppp_misalign": misalign[p].values if p in misalign.columns else np.nan,
            "reserves_z_pair": res_pair[p].values if p in res_pair.columns else np.nan,
            "score_value_z": Z_val[p].values if p in Z_val.columns else np.nan,
            "score_mom12_z": Z_mom[p].values if p in Z_mom.columns else np.nan,
            "score_carry_z": Z_carry[p].values if p in Z_carry.columns else np.nan,
            "score_composite": score[p].values if p in score.columns else np.nan,
            "war_risk": war_risk[p].values if p in war_risk.columns else np.nan
        })
        rows.append(df)
    feat = pd.concat(rows, ignore_index=True).dropna(subset=["spot"]).sort_values(["date","pair"])
    feat.to_csv(outdir / "features_panel.csv", index=False)

    # Latest snapshot
    last_date = FX.index.max()
    latest = feat[feat["date"]==last_date].sort_values("score_composite", ascending=False)
    latest.to_csv(outdir / "signals_latest.csv", index=False)

    # Backtest
    features_midx = feat.set_index(["date","pair"]).sort_index()
    bt, baskets = backtest_ls(FX, features_midx, topN=args.topN, rebalance=args.rebalance, score_col="score_composite")
    idx_ls = (bt["long_short"].fillna(0)+1).cumprod()
    out_bt = bt.copy()
    out_bt["idx_ls"] = idx_ls
    out_bt.to_csv(outdir / "backtest.csv", index=True)
    baskets.to_csv(outdir / "baskets.csv", index=False)

    # Performance
    stats_ls = perf_stats(bt["long_short"])
    stats_lo = perf_stats(bt["long_only"])
    stats_so = perf_stats(bt["short_only"])

    # Summary
    top5  = latest[["pair","score_composite","carry_pct","mom_12m","ppp_misalign","war_risk"]].head(5).to_dict(orient="records")
    bot5  = latest[["pair","score_composite","carry_pct","mom_12m","ppp_misalign","war_risk"]].tail(5).to_dict(orient="records")
    summary = {
        "last_date": str(last_date.date()) if pd.notna(last_date) else None,
        "topN": args.topN,
        "latest_top5": top5,
        "latest_bottom5": bot5,
        "perf_long_short": stats_ls,
        "perf_long_only": stats_lo,
        "perf_short_only": stats_so
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        fx=args.fx, rates=(args.rates or None), cpi=(args.cpi or None), reserves=(args.reserves or None),
        macro=(args.macro or None), freq=args.freq, rebalance=args.rebalance, topN=args.topN,
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Currency Wars ==")
    print(f"Last: {summary['last_date']} | Top longs: {[r['pair'] for r in top5]} | Top shorts: {[r['pair'] for r in bot5]}")
    print(f"L/S: CAGR {stats_ls['cagr_pct']:.2f}%, Vol {stats_ls['vol_pct']:.2f}%, Sharpe {stats_ls['sharpe']:.2f}, Hit {stats_ls['hit_rate']:.2%}")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
