#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exporters_vs_usdjpy.py — FX beta, event studies, hedges & backtests for JP exporters vs USDJPY
-----------------------------------------------------------------------------------------------

What this does
==============
A research toolkit to quantify how **Japanese exporters' equities** co-move with **USDJPY**,
and to test simple trading/hedging rules.

It ingests:
- equity prices (daily, adj),
- USDJPY (or JPYUSD) FX series,
- optional market benchmark (e.g., TOPIX, NKY),
- optional stock-level revenue exposure tags (exporter/importer, foreign revenue share),

…and then:

1) Cleans & aligns daily returns
   • r_fx = Δlog(USDJPY); if only JPYUSD given, it is inverted  
   • r_eq: daily pct or Δlog(adj_close)  
   • r_mkt (optional): market/sector control

2) Exposure (FX beta) estimation
   • Per-stock OLS: r_i = α + β_fx * r_fx + β_mkt * r_mkt + ε  
   • Rolling betas (lookback L), HAC/Newey–West SEs  
   • Cross-sectional summary by exporter/importer bucket

3) Event studies (big FX moves)
   • Identify days when |r_fx| exceeds a z-score threshold  
   • CARs for exporter and importer baskets ±W days, market-adjusted

4) Hedges
   • Portfolio FX beta → suggested hedge ratio using USDJPY exposure proxy  
   • With or without market neutralization

5) Backtests (illustrative)
   • Signal: FX z-score (k-day) momentum/mean-reversion  
   • Position: long exporters vs market when USDJPY strengthens; flip when it weakens  
   • Equity curve & summary stats

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--prices prices.csv              REQUIRED  (daily)
  Columns:
    date, ticker[, symbol], adj_close[, close], [weight] (weight optional)
--fx fx.csv                      REQUIRED
  Columns (any one):
    date, USDJPY[, usdjpy]   OR   date, JPYUSD[, jpyusd]
--bench bench.csv                OPTIONAL
  Columns:
    date, ticker, adj_close      # Provide a single market index ticker, or multiple (will pick first)
--map map.csv                    OPTIONAL (stock tagging)
  Columns (any subset):
    ticker, bucket[, type]    # bucket ∈ {EXPORTER, IMPORTER, MIXED}
    foreign_rev_share[, fx_share, revenue_fx_share] in [0,1]
    sector (optional)

CLI (key)
---------
--lookback 252                   Rolling window for betas
--min_obs 120                    Minimum obs for a regression window
--z_window 60                    Window for FX z-score signal
--z_threshold 1.0                Threshold in σ for events / signals
--car_window 5                   ±days for CAR in event study
--signal_mode momentum|reversion Strategy flavor for FX signal
--enter_cost_bps 5               One-way trading cost (bps) for backtest
--outdir out_usdjpy              Output directory
--start / --end                  Filters (YYYY-MM-DD)

Outputs
-------
- fx_series.csv                  Clean FX series (USDJPY, r_fx)
- returns_panel.csv              Stock & market returns (aligned)
- fx_betas.csv                   Point/rolling betas, t-stats, last-window exposures
- bucket_summary.csv             Exposure & performance by bucket
- event_study.csv               CARs around big FX moves (exporter/importer baskets)
- hedges.csv                     Hedge ratio suggestions
- backtest.csv                   Daily positions & equity curve
- backtest_summary.json          Simple performance stats
- summary.json                   Run overview
- config.json                    Echo run configuration

Notes
-----
• r_fx uses Δlog. For JPYUSD input, r_fx = -Δlog(JPYUSD).  
• Market-adjusted returns use provided benchmark if available.  
• This is research tooling, not investment advice.

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

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        lc = cand.lower()
        if lc in low: return low[lc]
    for cand in cands:
        t = cand.lower()
        for c in df.columns:
            if t in str(c).lower(): return c
    return None

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def dlog(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").replace(0, np.nan)
    return np.log(x) - np.log(x.shift(1))

def winsor(s: pd.Series, p: float=0.001) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

# OLS + HAC (Newey–West) for SEs on a single regression
def ols_beta_resid(X: np.ndarray, y: np.ndarray):
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XTX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XTX_inv: np.ndarray, L: int) -> np.ndarray:
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1) + 1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XTX_inv @ S @ XTX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return se


# ----------------------------- loaders -----------------------------

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df, "date"); t = ncol(df, "ticker","symbol"); p = ncol(df, "adj_close","close")
    w = ncol(df, "weight")
    if not (d and t and p):
        raise ValueError("prices.csv must have date, ticker, adj_close/close.")
    df = df.rename(columns={d:"date", t:"ticker", p:"adj_close"})
    df["date"] = to_dt(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    if w: df = df.rename(columns={w:"weight"})
    return df.dropna(subset=["date","ticker","adj_close"]).sort_values(["ticker","date"])

def load_fx(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df, "date")
    u = ncol(df, "USDJPY","usdjpy","USD_JPY","usd_jpy")
    j = ncol(df, "JPYUSD","jpyusd","JPY_USD","jpy_usd")
    if not d:
        raise ValueError("fx.csv needs a date column.")
    df = df.rename(columns={d:"date"})
    df["date"] = to_dt(df["date"])
    usdjpy = None
    if u:
        usdjpy = pd.to_numeric(df[u], errors="coerce")
    elif j:
        jpyusd = pd.to_numeric(df[j], errors="coerce")
        usdjpy = 1.0 / jpyusd
    else:
        # try to guess first non-date numeric column
        candidates = [c for c in df.columns if c != "date"]
        if not candidates: raise ValueError("fx.csv must include USDJPY or JPYUSD.")
        usdjpy = pd.to_numeric(df[candidates[0]], errors="coerce")
    out = pd.DataFrame({"date": df["date"], "USDJPY": usdjpy})
    out = out.dropna().sort_values("date")
    out["r_fx"] = dlog(out["USDJPY"])
    return out

def load_bench(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); t = ncol(df, "ticker","symbol"); p = ncol(df, "adj_close","close")
    if not (d and t and p): raise ValueError("bench.csv needs date, ticker, adj_close/close.")
    df = df.rename(columns={d:"date", t:"ticker", p:"adj_close"})
    df["date"] = to_dt(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna()
    # keep first index ticker as "market"
    tickers = df["ticker"].unique().tolist()
    mkt = tickers[0]
    out = df[df["ticker"]==mkt][["date","adj_close"]].sort_values("date").rename(columns={"adj_close":"mkt_px"})
    out["r_mkt"] = dlog(out["mkt_px"])
    return out

def load_map(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    t = ncol(df, "ticker","symbol"); b = ncol(df, "bucket","type"); s = ncol(df, "sector")
    fr = ncol(df, "foreign_rev_share","fx_share","revenue_fx_share")
    if not t:
        raise ValueError("map.csv needs 'ticker' column.")
    df = df.rename(columns={t:"ticker"})
    if b: df = df.rename(columns={b:"bucket"})
    if s: df = df.rename(columns={s:"sector"})
    if fr: df = df.rename(columns={fr:"foreign_rev_share"})
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "bucket" in df.columns:
        df["bucket"] = df["bucket"].astype(str).str.upper().str.strip()
    if "foreign_rev_share" in df.columns:
        df["foreign_rev_share"] = pd.to_numeric(df["foreign_rev_share"], errors="coerce").clip(0,1)
    return df


# ----------------------------- transforms -----------------------------

def build_returns_panel(PR: pd.DataFrame, FX: pd.DataFrame, BEN: pd.DataFrame) -> pd.DataFrame:
    # Stock returns
    PR = PR.sort_values(["ticker","date"]).copy()
    PR["r_eq"] = PR.groupby("ticker")["adj_close"].apply(dlog).reset_index(level=0, drop=True)
    panel = PR[["date","ticker","r_eq"]].dropna()
    # FX & Market
    panel = panel.merge(FX[["date","r_fx","USDJPY"]], on="date", how="left")
    if not BEN.empty:
        panel = panel.merge(BEN[["date","r_mkt"]], on="date", how="left")
    return panel.dropna(subset=["r_eq","r_fx"])

def market_adjusted(panel: pd.DataFrame) -> pd.DataFrame:
    if "r_mkt" in panel.columns:
        panel["r_abn"] = panel["r_eq"] - panel["r_mkt"]
    else:
        panel["r_abn"] = panel["r_eq"]
    return panel


# ----------------------------- exposures -----------------------------

def regress_exposure(df: pd.DataFrame, lookback: int, min_obs: int) -> pd.DataFrame:
    """
    Per-ticker rolling OLS: r_eq = α + β_fx * r_fx + [β_mkt * r_mkt]
    """
    rows = []
    has_mkt = "r_mkt" in df.columns and df["r_mkt"].notna().any()
    for tk, g in df.groupby("ticker"):
        g = g.dropna(subset=["r_eq","r_fx"]).sort_values("date")
        if g.shape[0] < max(min_obs, lookback):
            continue
        # design columns
        if has_mkt:
            Xfull = g[["r_fx","r_mkt"]].values
            names = ["beta_fx","beta_mkt"]
        else:
            Xfull = g[["r_fx"]].values
            names = ["beta_fx"]
        ones = np.ones((Xfull.shape[0],1))
        Xfull = np.concatenate([ones, Xfull], axis=1)
        dates = g["date"].values
        y = g["r_eq"].values.reshape(-1,1)
        for i in range(lookback, len(g)+1):
            idx = slice(i-lookback, i)
            X = Xfull[idx, :]
            Y = y[idx, :]
            beta, resid, XTX_inv = ols_beta_resid(X, Y)
            se = hac_se(X, resid, XTX_inv, L=min(20, lookback//5))
            b_dict = {"ticker": tk, "date": pd.Timestamp(dates[i-1])}
            # map coefficients
            b_dict["alpha"]   = float(beta[0,0]); b_dict["alpha_se"] = float(se[0])
            for j, nm in enumerate(names, start=1):
                b_dict[nm] = float(beta[j,0])
                b_dict[nm+"_se"] = float(se[j])
                b_dict[nm+"_t"]  = float(beta[j,0] / se[j] if se[j] > 0 else np.nan)
            rows.append(b_dict)
    return pd.DataFrame(rows).sort_values(["ticker","date"])

def latest_exposures(roll: pd.DataFrame) -> pd.DataFrame:
    if roll.empty: return roll
    idx = roll.groupby("ticker")["date"].idxmax()
    return roll.loc[idx].reset_index(drop=True)


# ----------------------------- buckets & portfolios -----------------------------

def tag_buckets(roll: pd.DataFrame, MAP: pd.DataFrame) -> pd.DataFrame:
    if MAP.empty:
        out = roll.copy()
        out["bucket"] = np.where(out["beta_fx"]>0, "EXPORTER", "IMPORTER")
        return out
    return roll.merge(MAP[["ticker","bucket","foreign_rev_share"]].drop_duplicates("ticker"), on="ticker", how="left")

def basket_returns(panel: pd.DataFrame, MAP: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight baskets by bucket (EXPORTER/IMPORTER)."""
    G = panel.copy()
    if MAP.empty or "bucket" not in MAP.columns:
        # infer by instantaneous beta sign (noisy); better to rely on map when provided
        return pd.DataFrame()
    G = G.merge(MAP[["ticker","bucket"]], on="ticker", how="left")
    G = G.dropna(subset=["bucket"])
    out = []
    for dt, gg in G.groupby("date"):
        for bkt, g in gg.groupby("bucket"):
            r = g["r_abn"].mean()
            out.append({"date": dt, "bucket": bkt, "r": r})
    BK = pd.DataFrame(out).pivot(index="date", columns="bucket", values="r").sort_index()
    if "EXPORTER" in BK.columns and "IMPORTER" in BK.columns:
        BK["EXP_MINUS_IMP"] = BK["EXPORTER"] - BK["IMPORTER"]
    BK = BK.reset_index()
    return BK


# ----------------------------- event study -----------------------------

def fx_event_days(FX: pd.DataFrame, z_window: int, z_threshold: float) -> pd.DataFrame:
    fx = FX[["date","r_fx"]].copy().sort_values("date")
    fx["mu"] = fx["r_fx"].rolling(z_window, min_periods=max(10, z_window//3)).mean()
    fx["sd"] = fx["r_fx"].rolling(z_window, min_periods=max(10, z_window//3)).std(ddof=0)
    fx["z"] = (fx["r_fx"] - fx["mu"]) / (fx["sd"] + 1e-12)
    ev = fx[(fx["z"].abs() >= z_threshold)].copy()
    ev["sign"] = np.where(ev["z"]>0, "USDJPY_UP(¥↓)", "USDJPY_DOWN(¥↑)")
    return ev[["date","z","sign"]]

def car(series: pd.Series, anchor: pd.Timestamp, pre: int, post: int) -> float:
    idx = series.index
    if anchor not in idx:
        pos = idx.searchsorted(anchor)
        if pos >= len(idx): return np.nan
        anchor = idx[pos]
    i0 = idx.searchsorted(anchor) - pre
    i1 = idx.searchsorted(anchor) + post
    i0 = max(0, i0); i1 = min(len(idx)-1, i1)
    return float(series.iloc[i0:i1+1].sum())

def event_study_baskets(BK: pd.DataFrame, FXEV: pd.DataFrame, window: int) -> pd.DataFrame:
    if BK.empty or FXEV.empty: return pd.DataFrame()
    b = BK.set_index("date").sort_index()
    out = []
    for _, e in FXEV.iterrows():
        d = pd.Timestamp(e["date"])
        for col in [c for c in b.columns if c != "date"]:
            out.append({
                "event_date": d.date(),
                "event": str(e["sign"]),
                "series": col,
                "CAR": car(b[col].dropna(), d, window, window)
            })
    return pd.DataFrame(out).sort_values(["event_date","series"])


# ----------------------------- hedging -----------------------------

def hedge_suggestions(lat: pd.DataFrame, MAP: pd.DataFrame, portfolio_weights: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Given latest exposures and (optional) portfolio weights:
    - Compute portfolio beta_fx and suggest hedge ratio using USDJPY.
    - If weights omitted, assumes equal-weight exporters basket.
    """
    if lat.empty: return pd.DataFrame()
    use = lat.copy()
    if MAP.empty or "bucket" not in MAP.columns:
        use["bucket"] = np.where(use["beta_fx"]>0, "EXPORTER", "IMPORTER")
    else:
        use = use.merge(MAP[["ticker","bucket","foreign_rev_share"]], on="ticker", how="left")
    # choose portfolio: exporters only, equal weight unless given
    port = use[use["bucket"]=="EXPORTER"].copy()
    if portfolio_weights is not None and not portfolio_weights.empty:
        w = portfolio_weights[["ticker","weight"]].copy()
        w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
        port = port.merge(w, on="ticker", how="left")
    if "weight" not in port.columns or port["weight"].isna().all():
        port["weight"] = 1.0
    port["w"] = port["weight"] / port["weight"].sum()
    beta_fx_port = float((port["w"] * port["beta_fx"]).sum())
    # Very rough notional guide: hedge notional ≈ beta_fx_port × equity notional
    # (since returns regressed on Δlog(USDJPY)). For $1 of equity, short/notional in USDJPY proportional to beta.
    out = pd.DataFrame([{
        "portfolio": "EXPORTERS_EQW" if ("weight" not in port.columns) else "EXPORTERS_WGT",
        "beta_fx_port": beta_fx_port,
        "hedge_side": "Short USDJPY (¥↑ hedge)" if beta_fx_port>0 else "Long USDJPY (¥↓ hedge)",
        "hedge_ratio_vs_equity": abs(beta_fx_port)  # per $1 equity notional
    }])
    return out


# ----------------------------- backtest -----------------------------

def backtest_fx_signal(BK: pd.DataFrame, FX: pd.DataFrame, z_window: int, z_threshold: float,
                       mode: str, cost_bps: float) -> Tuple[pd.DataFrame, Dict]:
    """
    Signal on FX; trade EXP_MINUS_IMP basket return.
    mode = 'momentum' -> go long EXP_MINUS_IMP when z > +thr, short when z < -thr
    mode = 'reversion' -> opposite
    """
    if BK.empty:
        return pd.DataFrame(), {}
    fx = FX[["date","r_fx"]].copy().sort_values("date")
    fx["mu"] = fx["r_fx"].rolling(z_window, min_periods=max(10, z_window//3)).mean()
    fx["sd"] = fx["r_fx"].rolling(z_window, min_periods=max(10, z_window//3)).std(ddof=0)
    fx["z"] = (fx["r_fx"] - fx["mu"]) / (fx["sd"] + 1e-12)
    sig = pd.Series(0.0, index=fx["date"])
    long = fx["z"] > z_threshold
    short = fx["z"] < -z_threshold
    if mode.lower().startswith("mom"):
        sig[long.index] = np.where(long, 1.0, np.where(short, -1.0, 0.0))
    else:
        sig[long.index] = np.where(long, -1.0, np.where(short, 1.0, 0.0))
    # Align with basket
    bk = BK.set_index("date").sort_index()
    if "EXP_MINUS_IMP" not in bk.columns:
        return pd.DataFrame(), {}
    r = bk["EXP_MINUS_IMP"].copy()
    S = pd.DataFrame({"signal": sig.reindex(r.index).fillna(0.0)})
    # Apply cost when signal changes
    S["signal_prev"] = S["signal"].shift(1).fillna(0.0)
    churn = (S["signal"].ne(S["signal_prev"])).astype(float)
    cost = churn * (cost_bps/10000.0)
    pnl = S["signal"] * r - cost
    eq = (1 + pnl.fillna(0)).cumprod()
    out = pd.DataFrame({
        "date": r.index,
        "r_basket": r.values,
        "signal": S["signal"].values,
        "pnl": pnl.values,
        "equity_curve": eq.values
    })
    # stats
    ann = 252
    mu = pnl.mean()*ann
    sd = pnl.std(ddof=0)*np.sqrt(ann)
    sharpe = float(mu/sd) if sd>0 else np.nan
    dd = (eq / eq.cummax() - 1.0).min()
    stats = {"ann_return": float(mu), "ann_vol": float(sd), "sharpe": float(sharpe), "max_drawdown": float(dd)}
    return out, stats


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    prices: str
    fx: str
    bench: Optional[str]
    map: Optional[str]
    lookback: int
    min_obs: int
    z_window: int
    z_threshold: float
    car_window: int
    signal_mode: str
    enter_cost_bps: float
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Exporters vs USDJPY — exposure, events, hedges & backtests")
    ap.add_argument("--prices", required=True)
    ap.add_argument("--fx", required=True)
    ap.add_argument("--bench", default="")
    ap.add_argument("--map", default="")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--min_obs", type=int, default=120)
    ap.add_argument("--z_window", type=int, default=60)
    ap.add_argument("--z_threshold", type=float, default=1.0)
    ap.add_argument("--car_window", type=int, default=5)
    ap.add_argument("--signal_mode", default="momentum", choices=["momentum","reversion"])
    ap.add_argument("--enter_cost_bps", type=float, default=5.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_usdjpy")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    PR = load_prices(args.prices)
    FX = load_fx(args.fx)
    BEN = load_bench(args.bench) if args.bench else pd.DataFrame()
    MAP = load_map(args.map) if args.map else pd.DataFrame()

    # Time filters
    if args.start:
        s = to_dt(pd.Series([args.start])).iloc[0]
        for df in [PR, FX, BEN]:
            if not df.empty: df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = to_dt(pd.Series([args.end])).iloc[0]
        for df in [PR, FX, BEN]:
            if not df.empty: df.drop(df[df["date"] > e].index, inplace=True)

    # Build returns panel
    PANEL = build_returns_panel(PR, FX, BEN)
    PANEL = market_adjusted(PANEL)
    if PANEL.empty:
        raise ValueError("No overlapping dates with returns and FX.")

    # Save basic series
    FX[["date","USDJPY","r_fx"]].to_csv(outdir / "fx_series.csv", index=False)
    PANEL.to_csv(outdir / "returns_panel.csv", index=False)

    # Rolling exposures
    ROLL = regress_exposure(PANEL, lookback=int(args.lookback), min_obs=int(args.min_obs))
    ROLL.to_csv(outdir / "fx_betas.csv", index=False)
    LAT = latest_exposures(ROLL)
    LAT = tag_buckets(LAT, MAP)
    LAT.to_csv(outdir / "fx_betas_latest.csv", index=False)

    # Bucket summary
    bsum = (LAT.groupby("bucket")[["beta_fx","alpha"]].mean().reset_index() if "bucket" in LAT.columns
            else LAT.assign(bucket=np.where(LAT["beta_fx"]>0,"EXPORTER","IMPORTER")).groupby("bucket")[["beta_fx","alpha"]].mean().reset_index())
    bsum.to_csv(outdir / "bucket_summary.csv", index=False)

    # Basket returns
    BK = basket_returns(PANEL, MAP)
    if not BK.empty:
        BK.to_csv(outdir / "baskets.csv", index=False)

    # Event study
    FXEV = fx_event_days(FX, z_window=int(args.z_window), z_threshold=float(args.z_threshold))
    ES = event_study_baskets(BK, FXEV, window=int(args.car_window)) if not BK.empty else pd.DataFrame()
    if not ES.empty:
        ES.to_csv(outdir / "event_study.csv", index=False)

    # Hedges (equal-weight exporters)
    # If user supplied weights in prices.csv (column 'weight'), pass them
    weights = None
    if "weight" in PR.columns and not PR["weight"].isna().all():
        w = (PR.sort_values(["ticker","date"])
               .groupby("ticker")["weight"].last().reset_index())
        weights = w
    HED = hedge_suggestions(LAT, MAP, weights)
    if not HED.empty:
        HED.to_csv(outdir / "hedges.csv", index=False)

    # Backtest
    BT, ST = backtest_fx_signal(BK, FX, z_window=int(args.z_window),
                                z_threshold=float(args.z_threshold),
                                mode=str(args.signal_mode),
                                cost_bps=float(args.enter_cost_bps))
    if not BT.empty:
        BT.to_csv(outdir / "backtest.csv", index=False)
        (outdir / "backtest_summary.json").write_text(json.dumps(ST, indent=2))

    # Summary
    summary = {
        "sample": {
            "start": str(PANEL["date"].min().date()),
            "end": str(PANEL["date"].max().date()),
            "n_stocks": int(PANEL["ticker"].nunique()),
            "n_days": int(PANEL["date"].nunique())
        },
        "fx": {
            "mean_daily_r": float(FX["r_fx"].mean()),
            "sd_daily_r": float(FX["r_fx"].std(ddof=0))
        },
        "betas": {
            "tickers_with_estimates": int(LAT.shape[0]),
            "exporter_avg_beta": float(bsum[bsum["bucket"]=="EXPORTER"]["beta_fx"].iloc[0]) if "EXPORTER" in bsum["bucket"].values else None,
            "importer_avg_beta": float(bsum[bsum["bucket"]=="IMPORTER"]["beta_fx"].iloc[0]) if "IMPORTER" in bsum["bucket"].values else None,
        },
        "has_baskets": bool(not BK.empty),
        "has_event_study": bool(not ES.empty),
        "has_hedges": bool(not HED.empty),
        "has_backtest": bool(not BT.empty),
        "files": {
            "fx_series": "fx_series.csv",
            "returns_panel": "returns_panel.csv",
            "fx_betas": "fx_betas.csv",
            "fx_betas_latest": "fx_betas_latest.csv",
            "bucket_summary": "bucket_summary.csv",
            "baskets": "baskets.csv" if not BK.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "hedges": "hedges.csv" if not HED.empty else None,
            "backtest": "backtest.csv" if not BT.empty else None,
            "backtest_summary": "backtest_summary.json" if not BT.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        prices=args.prices, fx=args.fx, bench=(args.bench or None), map=(args.map or None),
        lookback=int(args.lookback), min_obs=int(args.min_obs), z_window=int(args.z_window),
        z_threshold=float(args.z_threshold), car_window=int(args.car_window),
        signal_mode=args.signal_mode, enter_cost_bps=float(args.enter_cost_bps),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Exporters vs USDJPY ==")
    print(f"Sample: {summary['sample']['start']} → {summary['sample']['end']} | Stocks: {summary['sample']['n_stocks']}")
    print(f"FX daily μ/σ: {summary['fx']['mean_daily_r']:.5f} / {summary['fx']['sd_daily_r']:.5f}")
    if summary["has_baskets"]:
        print("Basket returns written (exporters/importers).")
    if summary["has_event_study"]:
        print(f"Event study (|z|≥{args.z_threshold}, ±{args.car_window}d) written.")
    if summary["has_hedges"]:
        print("Hedge ratio suggestions written.")
    if summary["has_backtest"]:
        print(f"Backtest done ({args.signal_mode}, z_window={args.z_window}, thr={args.z_threshold}).")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
