#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boj_etf_front_run.py — Detect, model, and backtest BOJ ETF front-running
-------------------------------------------------------------------------

What this does
==============
A research toolkit to study **Bank of Japan ETF purchase days** and potential
*front-run* opportunities around midday, when the BOJ historically executes ETF
buying (TOPIX / Nikkei 225 / JPX400 buckets).

It ingests intraday (preferred) or daily index/stock data and BOJ flow logs to:

1) Construct intraday features per trading day (index level)
   • Overnight, AM (open→11:30), midday gap (11:30→12:30), PM (12:30→close) returns  
   • Realized vol, drawdown, day-of-week, lagged purchase flags

2) Label purchase days from **flows.csv**; fit a **simple logistic model**
   predicting purchase probability from AM conditions (no external libraries)

3) Backtest rule-based & model-based strategies
   • Classic rule: go long index at 12:30 if AM return ≤ threshold (e.g., −0.5%), exit at close  
   • Model: go long at 12:30 if predicted P(purchase) ≥ τ, exit at close or T+1 open
   • Computes trade hit-rate, average bps, Sharpe (daily), turnover

4) Event study (abnormal returns)
   • BOJ purchase vs non-purchase days for PM (12:30→close) and close→next open

5) Cross-section flow pressure (optional)
   • Map BOJ purchase amounts into **index weights** per stock; compare to ADV  
   • Outputs flow/ADV stress to identify likely squeeze candidates on purchase days

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--prices prices.csv     REQUIRED (index prices; intraday preferred)
  Columns (any subset):
    date[, timestamp], time[, datetime], ticker[, symbol] (index symbol),
    open, high, low, close, volume
  Notes:
    • If intraday, include 'time' (HH:MM[:SS]) or a combined 'timestamp/datetime'
      column. If only daily data is supplied, AM/PM splits use proxies (less robust).

--flows flows.csv       REQUIRED (BOJ ETF operations)
  Columns:
    date, amount_jpy[, amount], bucket[, program]  # e.g., TOPIX / NIKKEI225 / JPX400
  Notes:
    • amount_jpy ≤ 0 means no purchase; positive = purchase day for that bucket.
    • If multiple buckets per day, rows can repeat the date.

--weights weights.csv   OPTIONAL (index constituent weights; for cross-section)
  Columns:
    date[, rebalance_date], bucket, ticker, weight_pct

--stocks stocks.csv     OPTIONAL (stock daily OHLCV; for ADV & flow pressure)
  Columns:
    date, ticker, close[, volume]

--events events.csv     OPTIONAL (policy changes / quirks)
  Columns:
    date, label[, type]

Key CLI
-------
--index "TOPIX"               Index name/tag to filter from prices
--am_cutoff "11:30:00"        AM session end time (local)
--pm_start  "12:30:00"        PM session start time (local)
--rule_threshold -0.5         AM return threshold (%) for rule-based entry
--prob_threshold 0.5          P(purchase) threshold for model-based entry
--exit "close"                Exit at "close" or "next_open"
--lags 3                      Lags for features (e.g., prev purchase)
--start / --end               Date filters (YYYY-MM-DD)
--outdir out_boj              Output directory
--min_obs 60                  Minimum obs to fit model
--adv_lookback 20             Days for ADV in cross-section
--bucket_map "TOPIX:TOPIX,N225:NIKKEI225,JPX:JPX400"  Normalize bucket names

Outputs
-------
- intraday_features.csv       Per-day AM/PM features and labels
- signals.csv                 Predicted probabilities and chosen signal
- backtest.csv                Trade-by-trade and equity curve
- event_study.csv             PM & overnight abnormal returns on purchase vs not
- flow_pressure.csv           Stock-level flow/ADV stress (if weights & stocks)
- summary.json                Headline metrics
- config.json                 Echo of configuration

DISCLAIMER
----------
This is research tooling with simplifying assumptions (timing, fill prices,
benchmarks). Validate against your market data & execution reality before use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

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

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.date

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def pct(a: float, b: float) -> float:
    if b is None or b == 0 or (a != a) or (b != b): return np.nan
    return (a / b) - 1.0

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def dayofweek(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d).dt.dayofweek  # 0=Mon

def parse_time_like(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has 'dt' (datetime) and 'date' columns. Accept date+time, or timestamp/datetime."""
    cols = df.columns
    # Combined timestamp?
    ts = ncol(df, "timestamp","datetime","ts")
    if ts:
        dt = to_dt(df[ts])
        df["dt"] = dt
        df["date"] = dt.dt.date
    else:
        dcol = ncol(df, "date")
        tcol = ncol(df, "time")
        if not dcol:
            raise ValueError("prices.csv needs 'date' or a combined 'timestamp/datetime'.")
        if tcol:
            dt = pd.to_datetime(df[dcol].astype(str) + " " + df[tcol].astype(str), errors="coerce")
            df["dt"] = dt.dt.tz_localize(None)
        else:
            # daily data; create 15:00 time as close
            dt = pd.to_datetime(df[dcol], errors="coerce") + pd.Timedelta(hours=15)
            df["dt"] = dt.dt.tz_localize(None)
        df["date"] = to_dt(pd.to_datetime(df[dcol], errors="coerce")).dt.date
    return df

def pick_price(df: pd.DataFrame, cutoff: str, how: str="last_le") -> pd.Series:
    """
    Pick the last price at or before cutoff time (how="last_le"), or first after (how="first_ge").
    df must have 'dt' and 'close' columns and single-day data.
    """
    tt = pd.to_datetime(str(df["date"].iloc[0]) + " " + cutoff)
    if how == "last_le":
        s = df[df["dt"] <= tt]["close"]
        return s.iloc[-1] if not s.empty else np.nan
    else:
        s = df[df["dt"] >= tt]["close"]
        return s.iloc[0] if not s.empty else np.nan

def winsor(s: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


# ----------------------------- loaders -----------------------------

def load_prices(path: str, index_tag: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize
    df = parse_time_like(df)
    sym = ncol(df, "ticker","symbol","index","name")
    if sym: df = df.rename(columns={sym:"ticker"})
    oc = ncol(df, "open","o"); hc = ncol(df, "high","h"); lc = ncol(df, "low","l"); cc = ncol(df, "close","c")
    if not cc: raise ValueError("prices.csv requires a 'close' column.")
    ren = {}
    if oc: ren[oc]="open"
    if hc: ren[hc]="high"
    if lc: ren[lc]="low"
    if cc: ren[cc]="close"
    df = df.rename(columns=ren)
    if index_tag and "ticker" in df.columns:
        df = df[df["ticker"].astype(str).str.upper().str.contains(index_tag.upper())]
        if df.empty:
            # keep all, but warn later
            pass
    df = df.sort_values("dt")
    return df[["dt","date","ticker","open","high","low","close","volume"] if "volume" in df.columns else ["dt","date","ticker","open","high","low","close"]]

def load_flows(path: str, bucket_map: Dict[str,str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    d  = ncol(df, "date") or df.columns[0]
    amt= ncol(df, "amount_jpy","amount","flow_jpy","purchase_jpy")
    buc= ncol(df, "bucket","program","index")
    if not (d and amt):
        raise ValueError("flows.csv needs date and amount_jpy.")
    df = df.rename(columns={d:"date", amt:"amount_jpy"})
    if buc: df = df.rename(columns={buc:"bucket"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.date
    df["amount_jpy"] = pd.to_numeric(df["amount_jpy"], errors="coerce").fillna(0.0)
    if "bucket" in df.columns:
        df["bucket"] = df["bucket"].astype(str).str.upper().str.strip()
        df["bucket"] = df["bucket"].map(lambda x: bucket_map.get(x, x))
    else:
        df["bucket"] = "ALL"
    # aggregate per day
    agg = df.groupby(["date"], as_index=False).agg(total_purchase_jpy=("amount_jpy","sum"))
    # bucket detail
    piv = df.pivot_table(index="date", columns="bucket", values="amount_jpy", aggfunc="sum").reset_index().rename_axis(None, axis=1)
    out = agg.merge(piv, on="date", how="left")
    out["is_purchase"] = (out["total_purchase_jpy"] > 0).astype(int)
    return out.sort_values("date")

def load_weights(path: Optional[str], bucket_map: Dict[str,str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d   = ncol(df, "date","rebalance_date","asof")
    buc = ncol(df, "bucket","index")
    tic = ncol(df, "ticker","symbol")
    w   = ncol(df, "weight_pct","weight","w")
    if not (d and buc and tic and w):
        raise ValueError("weights.csv needs date, bucket, ticker, weight_pct.")
    df = df.rename(columns={d:"date", buc:"bucket", tic:"ticker", w:"weight_pct"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.date
    df["bucket"] = df["bucket"].astype(str).str.upper().str.strip().map(lambda x: bucket_map.get(x, x))
    df["weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce")
    return df.dropna(subset=["date","bucket","ticker","weight_pct"]).sort_values(["date","bucket","ticker"])

def load_stocks(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d   = ncol(df, "date"); t = ncol(df, "ticker","symbol"); c = ncol(df, "close","c"); v = ncol(df, "volume","vol")
    if not (d and t and c):
        raise ValueError("stocks.csv needs date, ticker, close[, volume].")
    df = df.rename(columns={d:"date", t:"ticker", c:"close"})
    if v: df = df.rename(columns={v:"volume"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.date
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.sort_values(["ticker","date"])

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); lab = ncol(df, "label","event","name") or "label"
    if not d: raise ValueError("events.csv needs date.")
    df = df.rename(columns={d:"date", lab:"label"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.date
    return df[["date","label"]].dropna()


# ----------------------------- intraday features -----------------------------

def build_intraday_features(PR: pd.DataFrame, am_cutoff: str, pm_start: str) -> pd.DataFrame:
    """
    Create per-day features:
      o, am (open→11:30), md (11:30→12:30), pm (12:30→close), overnight (prev close→open)
    Accepts intraday prices or daily-only (falls back with weaker proxies).
    """
    if PR.empty: return pd.DataFrame()
    rows = []
    for d, g in PR.groupby("date"):
        g = g.sort_values("dt")
        o = g["open"].iloc[0] if "open" in g.columns and pd.notna(g["open"].iloc[0]) else g["close"].iloc[0]
        c = g["close"].iloc[-1]
        # intraday anchors
        am_px = pick_price(g, am_cutoff, "last_le")
        pm_px = pick_price(g, pm_start,  "first_ge")
        if pd.isna(am_px) or pd.isna(pm_px):
            # daily fallback: assume 'am' ~ VWAP of day? we'll approximate with mid-price
            am_px = g["close"].iloc[0]
            pm_px = g["close"].iloc[-1]
        am_ret = pct(am_px, o)
        md_ret = pct(pm_px, am_px)
        pm_ret = pct(c, pm_px)
        # realized vol proxy
        rv = np.sqrt(np.log(g["close"]).diff().pow(2).sum()) if g.shape[0] > 3 else np.nan
        rows.append({
            "date": d,
            "open": float(o), "am_px": float(am_px), "pm_px": float(pm_px), "close": float(c),
            "ret_ov_n": np.nan,  # fill after loop
            "ret_am": float(am_ret) if am_ret==am_ret else np.nan,
            "ret_md": float(md_ret) if md_ret==md_ret else np.nan,
            "ret_pm": float(pm_ret) if pm_ret==pm_ret else np.nan,
            "rv": float(rv) if rv==rv else np.nan
        })
    FE = pd.DataFrame(rows).sort_values("date")
    # overnight return (prev close → open)
    FE["ret_ov_n"] = FE["open"] / FE["close"].shift(1) - 1.0
    FE["dow"] = dayofweek(pd.to_datetime(FE["date"]))
    FE["ret_day"] = FE["close"]/FE["open"] - 1.0
    FE["drawdown_am"] = (FE[["open","am_px"]].min(axis=1) / FE["open"] - 1.0)
    FE["is_big_down_am"] = (FE["ret_am"] <= FE["ret_am"].quantile(0.15)).astype(int)
    return FE


# ----------------------------- modeling: logistic (no sklearn) -----------------------------

def add_labels(FE: pd.DataFrame, FL: pd.DataFrame) -> pd.DataFrame:
    df = FE.merge(FL[["date","is_purchase","total_purchase_jpy"]], on="date", how="left")
    df["is_purchase"] = df["is_purchase"].fillna(0).astype(int)
    df["total_purchase_jpy"] = df["total_purchase_jpy"].fillna(0.0)
    # lags
    df["lag_purchase"] = df["is_purchase"].shift(1).fillna(0)
    df["lag2_purchase"] = df["is_purchase"].shift(2).fillna(0)
    return df

def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd==0] = 1.0
    Z = (X - mu) / sd
    Z[np.isnan(Z)] = 0.0
    return Z, mu, sd

def logistic_fit(X: np.ndarray, y: np.ndarray, max_iter: int=200, tol: float=1e-6) -> Tuple[np.ndarray, float]:
    """
    Newton–Raphson for logistic regression.
    X includes a column of ones for intercept.
    """
    n,k = X.shape
    w = np.zeros((k,1))
    for _ in range(max_iter):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        W = (p * (1 - p)).flatten()
        # guard against zero weights
        W = np.clip(W, 1e-6, None)
        # IRLS step
        XTWX = X.T @ (X * W.reshape(-1,1))
        XTWZ = X.T @ ((z + (y.reshape(-1,1) - p) / W.reshape(-1,1)) * W.reshape(-1,1))
        try:
            w_new = np.linalg.pinv(XTWX) @ XTWZ
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    # logloss
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    ll = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    return w.flatten(), float(ll)

def logistic_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    z = X @ w.reshape(-1,1)
    return (1.0 / (1.0 + np.exp(-z))).flatten()

def fit_purchase_model(DF: pd.DataFrame, min_obs: int, lags: int) -> Dict:
    y = DF["is_purchase"].values.astype(float)
    # features: AM return, overnight, RV, DOW dummies, lagged purchase
    feats = ["ret_am","ret_ov_n","rv","lag_purchase"]
    if lags >= 2: feats.append("lag2_purchase")
    # dropna rows
    Z = DF[feats].copy()
    Z = Z.replace([np.inf, -np.inf], np.nan)
    mask = Z.notna().all(axis=1)
    Xraw = Z[mask].values
    y_use = y[mask.values]
    if Xraw.shape[0] < min_obs:
        return {"ok": False, "note": "not enough observations"}
    # standardize and add intercept
    Xstd, mu, sd = standardize(Xraw)
    X = np.column_stack([np.ones(Xstd.shape[0]), Xstd])
    w, ll = logistic_fit(X, y_use)
    # in-sample predictions
    pin = logistic_predict(X, w)
    # back to full DF (out-of-sample style: use same mu/sd)
    Xall_raw = DF[feats].replace([np.inf,-np.inf], np.nan).fillna(0.0).values
    Xall_std = (Xall_raw - mu) / sd
    Xall = np.column_stack([np.ones(Xall_std.shape[0]), Xall_std])
    pall = logistic_predict(Xall, w)
    out = {
        "ok": True,
        "features": feats,
        "weights": w.tolist(),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "logloss_in": ll,
        "pin_sample": float(np.mean(pin)),
        "pred": pall
    }
    return out


# ----------------------------- backtests -----------------------------

def backtest_rules(DF: pd.DataFrame, rule_threshold: float, prob_threshold: float, exit_when: str="close") -> pd.DataFrame:
    """
    Two signals evaluated each day at 12:30:
      • RULE: ret_am ≤ rule_threshold (%)
      • MODEL: p_purchase ≥ prob_threshold
    P&L measured in bps (×10,000) of index return from 12:30→exit.
    """
    df = DF.copy()
    df["signal_rule"] = (df["ret_am"]*100.0 <= float(rule_threshold)).astype(int)
    df["signal_model"] = (df["p_purchase"] >= float(prob_threshold)).astype(int) if "p_purchase" in df.columns else 0
    # returns
    if exit_when.lower()=="close":
        r = df["ret_pm"]
    else:
        # close → next open as continuation
        r = (df["close"].shift(-1) / df["open"].shift(-1) - 1.0)
    df["ret_exit"] = r
    for sig in ["signal_rule","signal_model"]:
        df[f"pnl_bps_{sig}"] = (df[sig] * df["ret_exit"] * 10000.0).fillna(0.0)
    # equity curves
    for sig in ["signal_rule","signal_model"]:
        df[f"equity_{sig}"] = df[f"pnl_bps_{sig}"].cumsum()
    return df

def summarize_backtest(df: pd.DataFrame, sig: str) -> Dict:
    sub = df[df[sig]==1].copy()
    n = int(sub.shape[0])
    hit = float((sub["ret_exit"]>0).mean()) if n>0 else np.nan
    avg = float(sub["ret_exit"].mean()*10000.0) if n>0 else np.nan
    std = float(sub["ret_exit"].std(ddof=1)*10000.0) if n>1 else np.nan
    sharpe = (avg/std*np.sqrt(252)) if (std and std==std and std>0) else np.nan
    tot = float(sub["ret_exit"].sum()*10000.0) if n>0 else 0.0
    return {"n_trades": n, "hit_rate": hit, "avg_bps": avg, "stdev_bps": std, "sharpe_annual": sharpe, "total_bps": tot}

# ----------------------------- event study -----------------------------

def event_study_pm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare PM (12:30→close) and close→next open returns on purchase vs non-purchase days.
    """
    d0 = df.dropna(subset=["ret_pm"]).copy()
    d0["overnight_next"] = (df["close"].shift(-1) / df["open"].shift(-1) - 1.0)
    rows = []
    for flag, g in d0.groupby("is_purchase"):
        rows.append({"bucket":"ALL", "group":"purchase" if flag==1 else "no_purchase",
                     "pm_mean_bps": float(g["ret_pm"].mean()*10000.0),
                     "pm_med_bps": float(g["ret_pm"].median()*10000.0),
                     "ov_next_mean_bps": float(g["overnight_next"].mean()*10000.0),
                     "n": int(g.shape[0])})
    return pd.DataFrame(rows)

# ----------------------------- flow pressure (cross-section) -----------------------------

def adv_series(ST: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if ST.empty or "volume" not in ST.columns:
        return pd.DataFrame()
    ST = ST.sort_values(["ticker","date"])
    ST["adv"] = ST.groupby("ticker")["volume"].rolling(lookback, min_periods=max(5, lookback//2)).mean().reset_index(level=0, drop=True)
    return ST[["date","ticker","adv"]]

def flow_pressure(FL: pd.DataFrame, WT: pd.DataFrame, ST: pd.DataFrame, adv_lb: int) -> pd.DataFrame:
    """
    For each purchase day and bucket, allocate amount_jpy × weight_pct to tickers.
    Compare to ADV to get flow/ADV ratio (proxy for pressure).
    """
    if FL.empty or WT.empty:
        return pd.DataFrame()
    # reshape flows to long by bucket
    flow_cols = [c for c in FL.columns if c not in ["date","total_purchase_jpy","is_purchase"]]
    F = FL.melt(id_vars=["date"], value_vars=flow_cols, var_name="bucket", value_name="amount_jpy_bucket").fillna(0.0)
    F = F[F["amount_jpy_bucket"]>0]
    if F.empty: return pd.DataFrame()
    # weights: use latest rebalance on/before date
    WT = WT.sort_values(["ticker","bucket","date"])
    # build nearest weight on/before each flow date
    rows = []
    # ADV if provided
    ADV = adv_series(ST, adv_lb) if not ST.empty else pd.DataFrame()
    for _, r in F.iterrows():
        d = r["date"]; b = r["bucket"]; amt = float(r["amount_jpy_bucket"])
        cand = WT[(WT["bucket"]==b) & (WT["date"]<=d)]
        if cand.empty:
            continue
        wb = cand.sort_values("date").groupby("ticker").tail(1)
        wb["flow_jpy"] = amt * (wb["weight_pct"]/100.0)
        wb["date"] = d
        rows.append(wb[["date","bucket","ticker","weight_pct","flow_jpy"]])
    OUT = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if OUT.empty: return OUT
    if not ADV.empty:
        OUT = OUT.merge(ADV, on=["date","ticker"], how="left")
        OUT["flow_to_adv"] = OUT["flow_jpy"] / OUT["adv"].replace(0,np.nan)
    return OUT.sort_values(["date","bucket","ticker"])


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    prices: str
    flows: str
    weights: Optional[str]
    stocks: Optional[str]
    events: Optional[str]
    index: Optional[str]
    am_cutoff: str
    pm_start: str
    rule_threshold: float
    prob_threshold: float
    exit: str
    lags: int
    start: Optional[str]
    end: Optional[str]
    outdir: str
    min_obs: int
    adv_lookback: int
    bucket_map_raw: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BOJ ETF front-run detection, modeling & backtest")
    ap.add_argument("--prices", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--stocks", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--index", default="")  # e.g., TOPIX
    ap.add_argument("--am_cutoff", default="11:30:00")
    ap.add_argument("--pm_start",  default="12:30:00")
    ap.add_argument("--rule_threshold", type=float, default=-0.5)
    ap.add_argument("--prob_threshold", type=float, default=0.5)
    ap.add_argument("--exit", default="close", choices=["close","next_open"])
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_boj")
    ap.add_argument("--min_obs", type=int, default=60)
    ap.add_argument("--adv_lookback", type=int, default=20)
    ap.add_argument("--bucket_map", dest="bucket_map_raw", default="TOPIX:TOPIX,N225:NIKKEI225,JPX:JPX400")
    return ap.parse_args()

def parse_bucket_map(s: str) -> Dict[str,str]:
    """
    "TOPIX:TOPIX,N225:NIKKEI225,JPX:JPX400" -> {"TOPIX":"TOPIX", "N225":"NIKKEI225", "JPX":"JPX400"}
    """
    if not s: return {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = {}
    for p in parts:
        if ":" in p:
            k,v = p.split(":",1)
            out[k.strip().upper()] = v.strip().upper()
        else:
            out[p.strip().upper()] = p.strip().upper()
    return out

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    bucket_map = parse_bucket_map(args.bucket_map_raw)

    PR = load_prices(args.prices, index_tag=(args.index or None))
    FL = load_flows(args.flows, bucket_map=bucket_map)
    WT = load_weights(args.weights, bucket_map=bucket_map) if args.weights else pd.DataFrame()
    ST = load_stocks(args.stocks) if args.stocks else pd.DataFrame()
    EV = load_events(args.events) if args.events else pd.DataFrame()

    # Date filters
    if args.start:
        s = pd.to_datetime(args.start).date()
        if not PR.empty: PR = PR[PR["date"] >= s]
        if not FL.empty: FL = FL[FL["date"] >= s]
        if not WT.empty: WT = WT[WT["date"] >= s]
        if not ST.empty: ST = ST[ST["date"] >= s]
    if args.end:
        e = pd.to_datetime(args.end).date()
        if not PR.empty: PR = PR[PR["date"] <= e]
        if not FL.empty: FL = FL[FL["date"] <= e]
        if not WT.empty: WT = WT[WT["date"] <= e]
        if not ST.empty: ST = ST[ST["date"] <= e]

    # Intraday features
    FE = build_intraday_features(PR, am_cutoff=args.am_cutoff, pm_start=args.pm_start)
    if FE.empty:
        raise ValueError("Could not construct features — check that prices.csv has valid dates/prices.")
    DF = add_labels(FE, FL)

    # Fit model
    MODEL = fit_purchase_model(DF, min_obs=int(args.min_obs), lags=int(args.lags))
    if MODEL.get("ok", False):
        DF["p_purchase"] = MODEL["pred"]
    else:
        DF["p_purchase"] = np.nan

    # Backtests
    BT = backtest_rules(DF, rule_threshold=float(args.rule_threshold),
                        prob_threshold=float(args.prob_threshold), exit_when=args.exit)
    # Event study
    ES = event_study_pm(DF.merge(FL[["date","is_purchase"]], on="date", how="left").fillna({"is_purchase":0}))

    # Cross-section flow pressure
    FP = flow_pressure(FL, WT, ST, adv_lb=int(args.adv_lookback)) if (not WT.empty) else pd.DataFrame()

    # Save artifacts
    FE.to_csv(outdir / "intraday_features.csv", index=False)
    sig_cols = ["date","ret_am","ret_md","ret_pm","p_purchase","is_purchase","total_purchase_jpy"]
    BT[sig_cols + [c for c in BT.columns if c.startswith("signal_")] ].to_csv(outdir / "signals.csv", index=False)
    BT.to_csv(outdir / "backtest.csv", index=False)
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)
    if not FP.empty: FP.to_csv(outdir / "flow_pressure.csv", index=False)

    # Summary
    summ_rule  = summarize_backtest(BT, "signal_rule")
    summ_model = summarize_backtest(BT, "signal_model")
    purchase_days = int(FL["is_purchase"].sum()) if "is_purchase" in FL.columns else int(DF["is_purchase"].sum())
    summary = {
        "sample": {
            "start": str(DF["date"].min()),
            "end": str(DF["date"].max()),
            "days": int(DF.shape[0]),
            "purchase_days": purchase_days
        },
        "rule_threshold_pct": float(args.rule_threshold),
        "prob_threshold": float(args.prob_threshold),
        "exit": args.exit,
        "model": {
            "ok": bool(MODEL.get("ok", False)),
            "features": MODEL.get("features"),
            "weights": MODEL.get("weights"),
            "logloss_in": MODEL.get("logloss_in"),
            "pin_sample": MODEL.get("pin_sample")
        },
        "backtest": {
            "rule": summ_rule,
            "model": summ_model
        },
        "event_study": ES.to_dict(orient="records") if not ES.empty else [],
        "has_flow_pressure": (not FP.empty)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        prices=args.prices, flows=args.flows, weights=(args.weights or None),
        stocks=(args.stocks or None), events=(args.events or None),
        index=(args.index or None), am_cutoff=args.am_cutoff, pm_start=args.pm_start,
        rule_threshold=float(args.rule_threshold), prob_threshold=float(args.prob_threshold),
        exit=args.exit, lags=int(args.lags), start=(args.start or None), end=(args.end or None),
        outdir=args.outdir, min_obs=int(args.min_obs), adv_lookback=int(args.adv_lookback),
        bucket_map_raw=args.bucket_map_raw
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== BOJ ETF Front-Run ==")
    print(f"Sample {summary['sample']['start']} → {summary['sample']['end']} | days {summary['sample']['days']} | purchase days {summary['sample']['purchase_days']}")
    if MODEL.get("ok", False):
        print(f"Model fit ok. In-sample avg P: {summary['model']['pin_sample']:.3f} | logloss {summary['model']['logloss_in']:.3f}")
    print(f"RULE (AM ≤ {args.rule_threshold:.2f}%): trades {summ_rule['n_trades']} | hit {summ_rule['hit_rate']:.2%} | avg {summ_rule['avg_bps']:.1f} bps")
    print(f"MODEL (P ≥ {args.prob_threshold:.2f}): trades {summ_model['n_trades']} | hit {summ_model['hit_rate']:.2%} | avg {summ_model['avg_bps']:.1f} bps")
    if not FP.empty:
        fp_top = FP.sort_values("flow_to_adv", ascending=False).head(5) if "flow_to_adv" in FP.columns else FP.sort_values("flow_jpy", ascending=False).head(5)
        print("Top flow/ADV pressure snapshots:")
        for _, r in fp_top.iterrows():
            val = r.get("flow_to_adv", np.nan)
            if val==val:
                print(f"  {r['date']} {r['bucket']} {r['ticker']}: flow/ADV={val:.2f}")
            else:
                print(f"  {r['date']} {r['bucket']} {r['ticker']}: flow ¥{r['flow_jpy']:,.0f}")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
