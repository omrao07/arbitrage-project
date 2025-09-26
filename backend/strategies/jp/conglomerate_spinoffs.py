#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conglomerate_spinoffs.py — Event studies, SOTP unlock, flow-pressure & backtests
--------------------------------------------------------------------------------

What this does
==============
A research toolkit for analyzing **conglomerate spin-offs**. It ingests a
corporate actions file for spin-offs, price series for parents/spinco/when-issued
shares, optional fundamentals/segments/multiples, and optional index flow events.
It then:

1) Cleans & aligns spin-off timelines
   • Key dates: announcement, record, distribution/ex-date, when-issued window  
   • Share ratio, taxable flag, region/sector tagging, aliases (WI tickers)

2) Event studies & attribution
   • CARs around announce and distribution dates (market/sector adjusted)  
   • Parent “stub” vs spinco decomposition around ex-date  
   • Beta-adjusted abnormal returns (per-name rolling beta to a benchmark)

3) Sum-of-the-parts (SOTP) unlock
   • Segment EBITDA/revenue × peer multiples → implied EV vs pre-spin EV  
   • “Unlock gap” and simple cross-section sorts

4) Index/flow pressure (optional)
   • If provided, reconstitution/add/drop events → return/volume patterns

5) Backtests (illustrative)
   • Long Spin: enter D+5 post distribution, hold H days (default 60)  
   • Stub Long: parent_ex less spinco value proxy; hold H days  
   • Paired L/S: long spinco, short parent (ratio-value neutral)

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--actions actions.csv            REQUIRED
  Columns (any subset; case-insensitive):
    parent_ticker, spin_ticker, announce_date, record_date, distribution_date[, ex_date],
    ratio  (spin shares per 1 parent share), region, sector[, industry],
    wi_ticker[, when_issued_start, when_issued_end], taxable_flag[, notes],
    shares_outstanding_parent[, shares_outstanding_spin], net_debt_parent[, net_debt_spin]

--prices prices.csv              REQUIRED (daily; adj preferred)
  Columns:
    date, ticker, adj_close[, close], volume (optional)

--bench bench.csv                OPTIONAL (benchmarks & sector series)
  Columns:
    date, ticker, adj_close
  Notes:
    Use broad market index tickers (e.g., "SPY","TOPIX","STOXX") and/or sector ETFs.

--bench_map bench_map.csv        OPTIONAL (map each equity to a benchmark)
  Columns:
    ticker, bench_ticker

--segments segments.csv          OPTIONAL (for SOTP)
  Columns:
    parent_ticker, segment, metric, value[, year]
  Where:
    metric ∈ { EBITDA, REVENUE, EBIT } (uppercase recommended)

--multiples multiples.csv        OPTIONAL (peer multiples for segments)
  Columns:
    segment[, industry], metric, multiple
  Where:
    metric ∈ { EV/EBITDA, EV/SALES, EV/EBIT }

--index_events index_events.csv  OPTIONAL (index flows)
  Columns:
    date, ticker, event_type, index_name  # event_type ∈ {ADD, DROP, REWEIGHT}

CLI (key)
---------
--car_pre 5                      CAR window pre  (trading days)
--car_post 20                    CAR window post (trading days)
--beta_lookback 252              Lookback days for rolling beta
--hold_days 60                   Holding horizon for backtests
--enter_lag 5                    Enter H days after distribution (e.g., D+5)
--min_mcap 0                     Filter spin names by market cap if available
--start / --end                  Sample filters (YYYY-MM-DD)
--outdir out_spinoffs            Output directory

Outputs
-------
- cleaned_actions.csv            Canonicalized actions with resolved dates & aliases
- event_study.csv               CARs around announce/ex for parent, spin, stub, and L/S
- abnormal_panel.csv            Daily abnormal returns & betas per name
- sotp_unlock.csv               SOTP result per parent (implied EV vs observed EV)
- flow_pressure.csv             ADD/DROP windows stats if index_events provided
- backtest.csv                  Strategy trades & equity curves
- summary.json                  Headline diagnostics
- config.json                   Echo run configuration

Assumptions & caveats
---------------------
• Prices should be adjusted for splits/dividends (adj_close preferred).  
• If ex_date missing, `distribution_date` is used.  
• Stub proxy around ex:  Parent_ex_value ≈ Parent_post − ratio × Spin_price (or WI).  
• Market/sector adjustment: r_abn = r_stock − β × r_bench; β from rolling OLS on lookback.  
• SOTP needs segments + multiples + (pre-spin EV via mkt cap & net debt).  
• This is research tooling; validate mappings/assumptions before use. Not investment advice.

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- utilities -----------------------------

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

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    s = s.replace(0, np.nan).astype(float)
    return np.log(s).diff()

def r_to_cum(r: pd.Series) -> pd.Series:
    x = (1 + r.fillna(0)).cumprod()
    return x / x.iloc[0] - 1.0 if len(x)>0 else x

def winsor(s: pd.Series, p: float=0.005) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)


# ----------------------------- loaders -----------------------------

def load_actions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {}
    for src, std in [
        ("parent_ticker","parent_ticker"), ("spin_ticker","spin_ticker"),
        ("announce_date","announce_date"), ("record_date","record_date"),
        ("distribution_date","distribution_date"), ("ex_date","ex_date"),
        ("ratio","ratio"), ("region","region"), ("sector","sector"), ("industry","industry"),
        ("wi_ticker","wi_ticker"), ("when_issued_start","wi_start"), ("when_issued_end","wi_end"),
        ("taxable_flag","taxable_flag"), ("notes","notes"),
        ("shares_outstanding_parent","shares_outstanding_parent"),
        ("shares_outstanding_spin","shares_outstanding_spin"),
        ("net_debt_parent","net_debt_parent"), ("net_debt_spin","net_debt_spin")
    ]:
        c = ncol(df, src)
        if c: ren[c]=std
    df = df.rename(columns=ren)
    req = ["parent_ticker","spin_ticker","announce_date","distribution_date","ratio"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"actions.csv missing required column: {r}")
    # normalize
    for c in ["announce_date","record_date","distribution_date","ex_date","wi_start","wi_end"]:
        if c in df.columns:
            df[c] = to_date(df[c])
    for c in ["ratio","shares_outstanding_parent","shares_outstanding_spin","net_debt_parent","net_debt_spin"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    df["ex_date"] = df["ex_date"].fillna(df["distribution_date"])
    for c in ["parent_ticker","spin_ticker","wi_ticker","region","sector","industry"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    # id
    df["spin_id"] = (df["parent_ticker"]+"→"+df["spin_ticker"]+"@"+pd.to_datetime(df["distribution_date"]).astype(str))
    return df

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df, "date"); t = ncol(df, "ticker","symbol"); p = ncol(df, "adj_close","adjclose","close")
    v = ncol(df, "volume","vol")
    if not (d and t and p):
        raise ValueError("prices.csv needs date, ticker, adj_close (or close).")
    df = df.rename(columns={d:"date", t:"ticker", p:"adj_close"})
    df["date"] = to_dt(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["adj_close"] = safe_num(df["adj_close"])
    if v: df = df.rename(columns={v:"volume"}); df["volume"] = safe_num(df["volume"])
    return df.dropna(subset=["date","ticker","adj_close"]).sort_values(["ticker","date"])

def load_bench(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); t = ncol(df, "ticker","symbol"); p = ncol(df, "adj_close","close")
    if not (d and t and p):
        raise ValueError("bench.csv needs date, ticker, adj_close.")
    df = df.rename(columns={d:"date", t:"ticker", p:"adj_close"})
    df["date"] = to_dt(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["adj_close"] = safe_num(df["adj_close"])
    return df.dropna().sort_values(["ticker","date"])

def load_bench_map(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    t = ncol(df, "ticker","symbol"); b = ncol(df, "bench_ticker","benchmark")
    if not (t and b): raise ValueError("bench_map.csv needs ticker, bench_ticker.")
    df = df.rename(columns={t:"ticker", b:"bench_ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["bench_ticker"] = df["bench_ticker"].astype(str).str.upper().str.strip()
    return df

def load_segments(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    p = ncol(df, "parent_ticker","ticker"); s = ncol(df, "segment"); m = ncol(df, "metric"); v = ncol(df, "value")
    if not (p and s and m and v): raise ValueError("segments.csv needs parent_ticker, segment, metric, value.")
    df = df.rename(columns={p:"parent_ticker", s:"segment", m:"metric", v:"value"})
    df["parent_ticker"] = df["parent_ticker"].astype(str).str.upper().str.strip()
    df["segment"] = df["segment"].astype(str).str.strip()
    df["metric"] = df["metric"].astype(str).str.upper().str.strip()
    df["value"] = safe_num(df["value"])
    return df.dropna(subset=["parent_ticker","segment","metric","value"])

def load_multiples(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    s = ncol(df, "segment","industry"); m = ncol(df, "metric"); u = ncol(df, "multiple","mult")
    if not (s and m and u): raise ValueError("multiples.csv needs segment/industry, metric, multiple.")
    df = df.rename(columns={s:"segment", m:"metric", u:"multiple"})
    df["segment"] = df["segment"].astype(str).str.strip()
    df["metric"] = df["metric"].astype(str).str.upper().str.strip()
    df["multiple"] = safe_num(df["multiple"])
    return df.dropna(subset=["segment","metric","multiple"])

def load_index_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df, "date"); t = ncol(df, "ticker"); e = ncol(df, "event_type","type"); i = ncol(df, "index_name","index")
    if not (d and t and e):
        raise ValueError("index_events.csv needs date, ticker, event_type.")
    df = df.rename(columns={d:"date", t:"ticker", e:"event_type"})
    if i: df = df.rename(columns={i:"index_name"})
    df["date"] = to_dt(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["event_type"] = df["event_type"].astype(str).str.upper().str.strip()
    if "index_name" in df.columns:
        df["index_name"] = df["index_name"].astype(str).str.upper().str.strip()
    return df.sort_values(["ticker","date"])


# ----------------------------- return plumbing -----------------------------

def pivot_prices(PR: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wide price & returns."""
    px = PR.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = px.pct_change().replace([np.inf,-np.inf], np.nan)
    return px, rets

def compute_betas(rets: pd.DataFrame, bench_rets: pd.DataFrame, bench_map: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Rolling beta per (name, date) to mapped bench ticker.
    """
    # fast covariance via rolling windows on aligned frames
    names = [c for c in rets.columns if c not in bench_rets.columns]
    if bench_map.empty:
        # Fallback: use first bench column for all
        if bench_rets.shape[1]==0:
            return pd.DataFrame(index=rets.index, columns=rets.columns)
        default_bench = bench_rets.columns[0]
        bm = {n: default_bench for n in names}
    else:
        bm = {r["ticker"]: r["bench_ticker"] for _, r in bench_map.iterrows()}
    betas = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for n in rets.columns:
        btk = bm.get(n, bench_rets.columns[0] if bench_rets.shape[1]>0 else None)
        if btk not in bench_rets.columns: 
            continue
        pair = pd.concat([rets[n], bench_rets[btk]], axis=1).dropna()
        if pair.empty: 
            continue
        cov = pair[rets.columns[:0].tolist() + [n, btk]].rolling(lookback, min_periods=lookback//2).cov()
        # Extract series: cov(n,bench) and var(bench)
        try:
            cov_nb = cov.xs(n, level=1)[btk]
            var_b  = cov.xs(btk, level=1)[btk]
        except Exception:
            # alternative extraction
            cov_nb = pair[n].rolling(lookback).cov(pair[btk])
            var_b  = pair[btk].rolling(lookback).var()
        beta = cov_nb / var_b.replace(0,np.nan)
        betas.loc[beta.index, n] = beta.values
    return betas

def abnormal_returns(rets: pd.DataFrame, bench_rets: pd.DataFrame, betas: pd.DataFrame, bench_map: pd.DataFrame) -> pd.DataFrame:
    """
    r_abn = r_stock − β × r_bench (β aligned by date)
    """
    abn = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    if bench_rets.empty:
        return rets
    bm = {r["ticker"]: r["bench_ticker"] for _, r in bench_map.iterrows()} if not bench_map.empty else {}
    default_bench = bench_rets.columns[0] if bench_rets.shape[1]>0 else None
    for n in rets.columns:
        btk = bm.get(n, default_bench)
        if btk not in bench_rets.columns:
            abn[n] = rets[n]
            continue
        b = betas[n].reindex_like(bench_rets[btk]) if n in betas.columns else 1.0
        abn[n] = rets[n] - b * bench_rets[btk]
    return abn


# ----------------------------- event study -----------------------------

def car_window(series: pd.Series, anchor_dt: pd.Timestamp, pre: int, post: int) -> Optional[float]:
    """
    Compute CAR from T−pre to T+post inclusive (excluding NaNs). Uses simple sum of daily returns.
    """
    if pd.isna(anchor_dt): return np.nan
    idx = series.index
    if anchor_dt not in idx:
        # pick nearest trading date after anchor
        pos = idx.searchsorted(anchor_dt)
        if pos >= len(idx): return np.nan
        anchor_dt = idx[pos]
    start = idx.searchsorted(anchor_dt) - pre
    end   = idx.searchsorted(anchor_dt) + post
    start = max(0, start); end = min(len(idx)-1, end)
    window = series.iloc[start:end+1]
    return float(window.sum()) if not window.empty else np.nan

def build_event_study(ACT: pd.DataFrame,
                      rets_abn: pd.DataFrame,
                      px: pd.DataFrame,
                      pre: int, post: int) -> pd.DataFrame:
    rows = []
    for _, a in ACT.iterrows():
        p = str(a["parent_ticker"]); s = str(a["spin_ticker"])
        wi = str(a["wi_ticker"]) if not pd.isna(a.get("wi_ticker", np.nan)) else None
        ad = pd.Timestamp(a["announce_date"])
        xd = pd.Timestamp(a["ex_date"])
        # choose spin series: when-issued if available during window around ex; else regular
        spin_ticker = None
        if wi and wi in px.columns:
            spin_ticker = wi
        elif s in px.columns:
            spin_ticker = s
        # parent abnormal
        if p in rets_abn.columns:
            for label, t in [("announce", ad), ("ex", xd)]:
                rows.append({"spin_id": a["spin_id"], "name": p, "role": "parent", "evt": label,
                             "CAR": car_window(rets_abn[p].dropna(), t, pre, post)})
        # spin abnormal (if exists)
        if spin_ticker and spin_ticker in rets_abn.columns:
            for label, t in [("announce", ad), ("ex", xd)]:
                rows.append({"spin_id": a["spin_id"], "name": spin_ticker, "role": "spin_or_wi", "evt": label,
                             "CAR": car_window(rets_abn[spin_ticker].dropna(), t, pre, post)})
        # stub proxy around ex: stub_ret = parent_ret − ratio*spin_ret (value approximation)
        if (p in rets_abn.columns) and (s in rets_abn.columns):
            # use spin regular after ex_date
            r_stub = (rets_abn[p] - float(a["ratio"]) * rets_abn[s]).dropna()
            rows.append({"spin_id": a["spin_id"], "name": p+"-STUB", "role": "stub", "evt": "ex",
                         "CAR": car_window(r_stub, xd, pre, post)})
        # paired long/short: long spin, short parent*ratio
        if (s in rets_abn.columns) and (p in rets_abn.columns):
            r_l_s = (rets_abn[s] - float(a["ratio"]) * rets_abn[p]).dropna()
            rows.append({"spin_id": a["spin_id"], "name": s+"−"+p, "role": "paired_LS", "evt": "ex",
                         "CAR": car_window(r_l_s, xd, pre, post)})
    return pd.DataFrame(rows)


# ----------------------------- SOTP -----------------------------

def sotp_unlock(ACT: pd.DataFrame, SEG: pd.DataFrame, MULT: pd.DataFrame) -> pd.DataFrame:
    """
    For each parent, compute implied EV = Σ (segment metric × multiple) per the first available metric.
    Observed EV = pre-spin market cap + net debt_parent.
    """
    if SEG.empty or MULT.empty:
        return pd.DataFrame()
    rows = []
    parents = sorted(SEG["parent_ticker"].unique())
    # Normalize multiples: map by segment & metric key
    mult_map = {(r["segment"].upper(), r["metric"].upper()): float(r["multiple"]) for _, r in MULT.iterrows()}
    for p in parents:
        g = SEG[SEG["parent_ticker"]==p]
        # choose metric priority
        for metric in ["EBITDA","EBIT","REVENUE"]:
            gg = g[g["metric"].str.upper()==metric]
            if not gg.empty:
                break
        if gg.empty:
            continue
        implied = 0.0
        detail = []
        for _, r in gg.iterrows():
            key = (r["segment"].upper(), metric.upper())
            m = mult_map.get(key, np.nan)
            if pd.isna(m):
                # fallback: any multiple for this metric (average)
                ms = [v for (seg, met), v in mult_map.items() if met==metric.upper()]
                m = float(np.nanmean(ms)) if ms else np.nan
            val = float(r["value"])
            ev = val * m if pd.notna(val) and pd.notna(m) else np.nan
            implied += 0.0 if pd.isna(ev) else ev
            detail.append({"parent_ticker": p, "segment": r["segment"], "metric": metric, "value": val, "multiple": m, "segment_ev": ev})
        # observed EV (needs market cap & net debt). Market cap will be inferred later if available.
        rows.append({"parent_ticker": p, "metric": metric, "implied_ev": implied, "detail": detail})
    S = pd.DataFrame(rows)
    # Attach observed EV from actions if net debt available and we can infer pre-spin market cap
    # We cannot infer market cap without price * shares; attempt from actions columns if present.
    obs_rows = []
    for _, r in S.iterrows():
        p = r["parent_ticker"]
        # pull net debt
        nd = float(ACT[ACT["parent_ticker"]==p]["net_debt_parent"].dropna().iloc[0]) if "net_debt_parent" in ACT.columns and (ACT["parent_ticker"]==p).any() and ACT["net_debt_parent"].notna().any() else np.nan
        mcp = np.nan  # will be filled in main if price & shares exist
        obs_rows.append({"parent_ticker": p, "implied_ev": r["implied_ev"], "observed_ev": np.nan, "net_debt_parent": nd, "metric": r["metric"]})
    OUT = pd.DataFrame(obs_rows)
    return OUT


# ----------------------------- Index flows -----------------------------

def flow_pressure(IE: pd.DataFrame, rets: pd.DataFrame, window_pre: int=5, window_post: int=10) -> pd.DataFrame:
    if IE.empty:
        return pd.DataFrame()
    rows = []
    for _, r in IE.iterrows():
        t = r["ticker"]; d = r["date"]
        if t not in rets.columns: 
            continue
        series = rets[t].dropna()
        car = car_window(series, d, window_pre, window_post)
        rows.append({"ticker": t, "date": d.date(), "event_type": r["event_type"], "index_name": r.get("index_name", np.nan), "CAR": car})
    return pd.DataFrame(rows)


# ----------------------------- Backtests -----------------------------

def trade_calendar_like(px: pd.DataFrame) -> List[pd.Timestamp]:
    return list(px.index)

def shift_trading_days(dates: List[pd.Timestamp], anchor: pd.Timestamp, k: int) -> Optional[pd.Timestamp]:
    if pd.isna(anchor): return None
    idx = np.searchsorted(dates, anchor)
    tgt = idx + k
    if tgt < 0 or tgt >= len(dates): return None
    return dates[tgt]

def backtests(ACT: pd.DataFrame, px: pd.DataFrame, rets_abn: pd.DataFrame,
              enter_lag: int, hold_days: int) -> pd.DataFrame:
    cal = trade_calendar_like(px)
    rows = []
    for _, a in ACT.iterrows():
        p = a["parent_ticker"]; s = a["spin_ticker"]; ratio = float(a["ratio"])
        xd = pd.Timestamp(a["ex_date"])
        # entry and exit dates
        entry = shift_trading_days(cal, xd, enter_lag)
        exit_  = shift_trading_days(cal, xd, enter_lag + hold_days)
        if (entry is None) or (exit_ is None): 
            continue
        # Long Spin (abnormal return cum)
        if s in rets_abn.columns:
            r = rets_abn[s].loc[entry:exit_]
            pnl = float((1+r).prod() - 1.0) if not r.empty else np.nan
            rows.append({"spin_id": a["spin_id"], "strategy": "LongSpin", "entry": str(entry.date()), "exit": str(exit_.date()), "pnl": pnl})
        # Stub Long: parent − ratio*spin, use simple arithmetic sum of daily abnormal returns (small r)
        if (p in rets_abn.columns) and (s in rets_abn.columns):
            r = (rets_abn[p] - ratio*rets_abn[s]).loc[entry:exit_]
            pnl = float(r.sum()) if not r.empty else np.nan
            rows.append({"spin_id": a["spin_id"], "strategy": "StubLong", "entry": str(entry.date()), "exit": str(exit_.date()), "pnl": pnl})
        # Paired L/S: spin − ratio*parent
        if (p in rets_abn.columns) and (s in rets_abn.columns):
            r = (rets_abn[s] - ratio*rets_abn[p]).loc[entry:exit_]
            pnl = float(r.sum()) if not r.empty else np.nan
            rows.append({"spin_id": a["spin_id"], "strategy": "PairedLS", "entry": str(entry.date()), "exit": str(exit_.date()), "pnl": pnl})
    return pd.DataFrame(rows)

def summarize_backtests(BT: pd.DataFrame) -> pd.DataFrame:
    if BT.empty: return pd.DataFrame()
    g = (BT.groupby("strategy")["pnl"].agg(["count","mean","median","std"])
            .rename(columns={"count":"n","mean":"avg","median":"med","std":"sd"}).reset_index())
    return g


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    actions: str
    prices: str
    bench: Optional[str]
    bench_map: Optional[str]
    segments: Optional[str]
    multiples: Optional[str]
    index_events: Optional[str]
    car_pre: int
    car_post: int
    beta_lookback: int
    hold_days: int
    enter_lag: int
    min_mcap: float
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Conglomerate spin-offs — event studies, SOTP & backtests")
    ap.add_argument("--actions", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--bench", default="")
    ap.add_argument("--bench_map", default="")
    ap.add_argument("--segments", default="")
    ap.add_argument("--multiples", default="")
    ap.add_argument("--index_events", default="")
    ap.add_argument("--car_pre", type=int, default=5)
    ap.add_argument("--car_post", type=int, default=20)
    ap.add_argument("--beta_lookback", type=int, default=252)
    ap.add_argument("--hold_days", type=int, default=60)
    ap.add_argument("--enter_lag", type=int, default=5)
    ap.add_argument("--min_mcap", type=float, default=0.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_spinoffs")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    ACT = load_actions(args.actions)
    PR  = load_prices(args.prices)
    BEN = load_bench(args.bench) if args.bench else pd.DataFrame()
    MAP = load_bench_map(args.bench_map) if args.bench_map else pd.DataFrame()
    SEG = load_segments(args.segments) if args.segments else pd.DataFrame()
    MUL = load_multiples(args.multiples) if args.multiples else pd.DataFrame()
    IE  = load_index_events(args.index_events) if args.index_events else pd.DataFrame()

    # Time filters
    if args.start:
        s = to_dt(pd.Series([args.start])).iloc[0]
        PR = PR[PR["date"] >= s]
        if not BEN.empty: BEN = BEN[BEN["date"] >= s]
        if not IE.empty:  IE  = IE[IE["date"] >= s]
    if args.end:
        e = to_dt(pd.Series([args.end])).iloc[0]
        PR = PR[PR["date"] <= e]
        if not BEN.empty: BEN = BEN[BEN["date"] <= e]
        if not IE.empty:  IE  = IE[IE["date"] <= e]

    # Wide prices & returns
    PX, RET = pivot_prices(PR)
    BENPX, BENRET = (pivot_prices(BEN) if not BEN.empty else (pd.DataFrame(index=PX.index), pd.DataFrame(index=PX.index)))
    # Align indices
    PX = PX.reindex(BENPX.index.union(PX.index)).sort_index()
    RET = PX.pct_change()
    if not BENPX.empty:
        BENPX = BENPX.reindex(PX.index)
        BENRET = BENPX.pct_change()

    # Betas & abnormal returns
    BETAS = compute_betas(RET, BENRET, MAP, lookback=int(args.beta_lookback)) if not BENRET.empty else pd.DataFrame(index=RET.index, columns=RET.columns)
    ABN = abnormal_returns(RET, BENRET, BETAS, MAP) if not BENRET.empty else RET.copy()
    ABN.to_csv(outdir / "abnormal_panel.csv")

    # Cleaned actions out
    ACT.to_csv(outdir / "cleaned_actions.csv", index=False)

    # Event studies
    ES = build_event_study(ACT, ABN, PX, pre=int(args.car_pre), post=int(args.car_post))
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # SOTP
    SOTP = sotp_unlock(ACT, SEG, MUL) if (not SEG.empty and not MUL.empty) else pd.DataFrame()
    # Try to attach observed EV if we can infer market cap at announce_date:
    if not SOTP.empty and {"shares_outstanding_parent","announce_date","parent_ticker"}.issubset(ACT.columns):
        obs = []
        for _, r in SOTP.iterrows():
            p = r["parent_ticker"]
            shares = ACT[ACT["parent_ticker"]==p]["shares_outstanding_parent"].dropna()
            if shares.empty or p not in PX.columns:
                obs.append(np.nan); continue
            ad = pd.to_datetime(ACT[ACT["parent_ticker"]==p]["announce_date"].iloc[0])
            # pick nearest trading date ≤ announce_date
            idx = PX.index.searchsorted(ad) - 1
            if idx < 0: idx = 0
            price = float(PX[p].iloc[idx])
            mcap = float(shares.iloc[0]) * price if pd.notna(price) else np.nan
            nd = float(ACT[ACT["parent_ticker"]==p]["net_debt_parent"].dropna().iloc[0]) if ACT["net_debt_parent"].notna().any() else np.nan
            obs.append(mcap + (nd if nd==nd else 0.0) if mcap==mcap else np.nan)
        SOTP["observed_ev"] = obs
        SOTP["unlock_gap_pct"] = (SOTP["implied_ev"] / SOTP["observed_ev"] - 1.0).replace([np.inf,-np.inf], np.nan)
        SOTP.to_csv(outdir / "sotp_unlock.csv", index=False)
    elif not SOTP.empty:
        SOTP.to_csv(outdir / "sotp_unlock.csv", index=False)

    # Index flow pressure
    FLOW = flow_pressure(IE, RET, window_pre=int(args.car_pre), window_post=int(args.car_post)) if not IE.empty else pd.DataFrame()
    if not FLOW.empty: FLOW.to_csv(outdir / "flow_pressure.csv", index=False)

    # Backtests
    BT = backtests(ACT, PX, ABN, enter_lag=int(args.enter_lag), hold_days=int(args.hold_days))
    if not BT.empty:
        BT.to_csv(outdir / "backtest.csv", index=False)
        SUMM = summarize_backtests(BT)
        SUMM.to_csv(outdir / "backtest_summary.csv", index=False)
    else:
        SUMM = pd.DataFrame()

    # Summary
    summary = {
        "sample": {
            "start": str(PX.index.min().date()) if len(PX.index)>0 else None,
            "end": str(PX.index.max().date()) if len(PX.index)>0 else None,
            "n_actions": int(ACT.shape[0]),
            "n_names_in_px": int(len(PX.columns)),
        },
        "car_window": {"pre": int(args.car_pre), "post": int(args.car_post)},
        "beta_lookback": int(args.beta_lookback),
        "backtest": SUMM.to_dict(orient="records") if not SUMM.empty else [],
        "files": {
            "cleaned_actions": "cleaned_actions.csv",
            "abnormal_panel": "abnormal_panel.csv",
            "event_study": "event_study.csv" if not ES.empty else None,
            "sotp_unlock": "sotp_unlock.csv" if (not SOTP.empty) else None,
            "flow_pressure": "flow_pressure.csv" if (not FLOW.empty) else None,
            "backtest": "backtest.csv" if (not BT.empty) else None,
            "backtest_summary": "backtest_summary.csv" if (not SUMM.empty) else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        actions=args.actions, prices=args.prices, bench=(args.bench or None),
        bench_map=(args.bench_map or None), segments=(args.segments or None),
        multiples=(args.multiples or None), index_events=(args.index_events or None),
        car_pre=int(args.car_pre), car_post=int(args.car_post),
        beta_lookback=int(args.beta_lookback), hold_days=int(args.hold_days),
        enter_lag=int(args.enter_lag), min_mcap=float(args.min_mcap),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Conglomerate Spin-offs ==")
    print(f"Actions: {ACT.shape[0]} | Price names: {len(PX.columns)} | Sample: {summary['sample']['start']} → {summary['sample']['end']}")
    if not ES.empty:
        print(f"Event study written (CAR −{args.car_pre}/+{args.car_post} days).")
    if not SOTP.empty:
        print("SOTP output written (check observed_ev availability).")
    if not FLOW.empty:
        print("Index flow pressure stats written.")
    if not BT.empty:
        print("Backtest results written.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
