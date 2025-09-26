#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buyback_trader.py

Simulate and analyze corporate share buyback programs:
- Parses an announcements CSV (date, ticker, size, type)
- Event study around announcement dates (AAR, CAAR)
- Execution simulation for open-market programs (ADV caps, pacing)
- Optional ASR (accelerated) and tender placeholders
- Exports tidy CSVs and optional plots
"""

import argparse, os
from dataclasses import dataclass
from datetime import datetime
import numpy as np, pandas as pd
import yfinance as yf

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------------- Config ---------------- #

@dataclass
class Config:
    announcements: str
    benchmark: str
    start: str
    end: str
    tpre: int
    tpost: int
    plot: bool
    outdir: str

# ---------------- Helpers ---------------- #

def ensure_outdir(base: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"buyback_trader_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir

def load_ann(path: str):
    df = pd.read_csv(path)
    if "date" not in df or "ticker" not in df:
        raise SystemExit("CSV must contain date and ticker columns")
    for col in ["size_usd","size_pct_mktcap","max_pct_adv","window_days","type"]:
        if col not in df: df[col] = np.nan
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["max_pct_adv"] = df["max_pct_adv"].fillna(25).astype(float)
    df["window_days"] = df["window_days"].fillna(60).astype(int)
    df["type"] = df["type"].fillna("open_market").str.lower()
    return df

# ---------------- Market data ---------------- #

def fetch_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker")
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t not in data.columns.levels[0]: continue
            sub = data[t][["Open","High","Low","Close","Volume"]].copy()
            sub.columns = pd.MultiIndex.from_product([[t], sub.columns])
            frames.append(sub)
        return pd.concat(frames, axis=1).sort_index()
    else:
        sub = data[["Open","High","Low","Close","Volume"]].copy()
        sub.columns = pd.MultiIndex.from_product([[tickers[0]], sub.columns])
        return sub

# ---------------- Event study ---------------- #

def event_windows(returns, ann, tpre, tpost):
    idx = returns.index; panels = []
    for _, row in ann.iterrows():
        t, d = row["ticker"], row["date"]
        if t not in returns.columns: continue
        if d not in idx:
            nxt = idx[idx >= d]
            if nxt.empty: continue
            d = nxt[0]
        center = idx.get_indexer([d])[0]
        start = max(0, center - tpre); end = min(len(idx)-1, center + tpost)
        win = returns[t].iloc[start:end+1].copy()
        rel = np.arange(start - center, end - center + 1)
        win.index = rel; win.name = f"{t}|{d.date()}"
        panels.append(win)
    if not panels: raise RuntimeError("No event windows built.")
    panel = pd.concat(panels, axis=1)
    AAR = panel.mean(axis=1).to_frame("AAR")
    CAAR = AAR.cumsum().rename(columns={"AAR":"CAAR"})
    return panel, pd.concat([AAR, CAAR], axis=1)

# ---------------- Execution simulation ---------------- #

def dollar_volume(close, volume): return close*volume
def adv(series, lookback=60): return series.rolling(lookback,min_periods=10).mean()

def simulate_open_market(row, px):
    t = row["ticker"]
    if t not in px.columns.levels[0]: return []
    idx = px.index; d0 = row["date"]
    if d0 not in idx:
        nxt = idx[idx>=d0]
        if nxt.empty: return []
        d0 = nxt[0]
    start = idx.get_indexer([d0])[0]; end = min(len(idx)-1, start+row["window_days"]-1)
    df = px[t].iloc[start:end+1]; 
    if df.empty: return []
    dv = dollar_volume(px[t]["Close"], px[t]["Volume"])
    adv_series = adv(dv).reindex(df.index).fillna(method="ffill")
    cap = (row["max_pct_adv"]/100.0)*adv_series
    size = row["size_usd"] if pd.notna(row["size_usd"]) else cap.mean()*row["window_days"]*0.8
    rem = size; trades=[]
    for date, sub in df.iterrows():
        close=float(sub["Close"]); hi=sub["High"]; lo=sub["Low"]
        vwap=(hi+lo+close)/3.0; daycap=cap.loc[date]
        target = rem/max(1,(df.index[-1]-date).days+1)
        notional=min(daycap,target,rem); shares=notional/close
        trades.append({"date":date,"ticker":t,"close":close,"vwap":vwap,"notional":notional,"shares":shares})
        rem-=notional; 
        if rem<=0: break
    return trades

# ---------------- Main ---------------- #

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--announcements",required=True)
    ap.add_argument("--benchmark",default="^GSPC")
    ap.add_argument("--start",default="2015-01-01")
    ap.add_argument("--end",default=None)
    ap.add_argument("--tpre",type=int,default=5)
    ap.add_argument("--tpost",type=int,default=20)
    ap.add_argument("--plot",action="store_true")
    ap.add_argument("--outdir",default="./artifacts")
    a=ap.parse_args()

    cfg=Config(a.announcements,a.benchmark,a.start,a.end,a.tpre,a.tpost,a.plot,a.outdir)
    outdir=ensure_outdir(cfg.outdir); ann=load_ann(cfg.announcements)
    tickers=ann["ticker"].unique().tolist(); universe=tickers+[cfg.benchmark]
    px=fetch_prices(universe,cfg.start,cfg.end)
    flat=pd.DataFrame({f"{t}_{f}":px[(t,f)] for t in universe for f in ["Open","High","Low","Close","Volume"] if (t,f) in px.columns})
    flat.to_csv(os.path.join(outdir,"prices.csv"))

    rets=pd.DataFrame({t:px[(t,"Close")].pct_change() for t in universe if (t,"Close") in px.columns}).dropna()
    rets.to_csv(os.path.join(outdir,"returns.csv"))

    if cfg.benchmark in rets:
        abn=rets.sub(rets[cfg.benchmark],axis=0)
    else: abn=rets
    tick_only=abn[[t for t in tickers if t in abn.columns]]
    if not tick_only.empty:
        panel,agg=event_windows(tick_only,ann,cfg.tpre,cfg.tpost)
        panel.to_csv(os.path.join(outdir,"event_windows.csv"))
        agg.to_csv(os.path.join(outdir,"AAR_CAAR.csv"))
        if cfg.plot and plt:
            plt.plot(agg.index,agg["CAAR"]); plt.axvline(0,ls="--"); 
            plt.title("CAAR"); plt.savefig(os.path.join(outdir,"plots","CAAR.png")); plt.close()

    trades=[]; 
    for _,r in ann.iterrows():
        if r["type"]=="open_market": trades+=simulate_open_market(r,px)
    pd.DataFrame(trades).to_csv(os.path.join(outdir,"executions.csv"),index=False)

    print("Done. Results in",outdir)

if __name__=="__main__":
    main()