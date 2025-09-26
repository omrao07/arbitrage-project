#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psu_disinvestment.py — PSU disinvestment analytics: events, CARs, liquidity & proceeds
-------------------------------------------------------------------------------------

What this does
==============
Given daily prices/volumes for listed PSUs, an index series, shareholding patterns,
and a table of **disinvestment events** (OFS/FPO/strategic sale/ETF rebalance, etc.),
this script:

1) Cleans & aligns everything to daily/monthly as needed.
2) Builds returns & liquidity metrics (Amihud illiquidity, turnover, value traded).
3) Runs **event studies** around disinvestment events:
   • Market-model abnormal returns and CARs over user-defined windows
   • Liquidity changes before/after (Amihud, turnover)
   • Free-float / govt-stake changes from shareholding snapshots
4) Cross-sectional regressions:
   • CAR ~ size_pct_sold + discount_pct + Δfree_float + prior run-up + valuations (P/B, ROE) + type dummies
5) Proceeds tracking vs **budget** targets (FY Apr–Mar) and ETF flow overlays
6) Scenarios:
   • Given planned stake sales (symbol, % stake, discount, ref price) → proceeds & rough impact

Inputs (CSV; flexible headers, case-insensitive)
------------------------------------------------
--prices prices.csv            REQUIRED (daily OHLC not needed; Close & Volume are enough)
  Columns: date, symbol, close[, volume, value_traded, free_float_shares, shares_out]
  Fuzzy maps: close, px, price; volume, vol; value_traded, traded_value, turnover_value

--index index.csv              REQUIRED (market or sector index for market model)
  Columns: date, close  (or return)

--events events.csv            REQUIRED (each row is an event)
  Columns (any subset):
    date, symbol, type, size_pct, stake_pct, pct_sold, discount_pct, offer_price, floor_price, proceeds_inr
  'type' examples: OFS, FPO, STRATEGIC, ETF_REBAL, GOVT_SALE, BUYBACK (ignored in CAR aggregation unless desired)

--shareholding shph.csv        OPTIONAL (quarterly)
  Columns: date, symbol[, govt_stake_pct][, free_float_pct][, shares_outstanding]

--fundamentals funda.csv       OPTIONAL (snapshot or time-series)
  Columns: date, symbol[, pb, roe, dividend_yield]

--budget budget.csv            OPTIONAL (FY tracking)
  Columns: fiscal_year, target_inr_cr[, realized_inr_cr]

--etf_flows etf.csv            OPTIONAL (CPSE/Bharat-22)
  Columns: date, etf, flow_inr[, is_rebalance]

--scenarios scn.csv            OPTIONAL (planned stake sales)
  Columns: symbol, stake_pct, discount_pct[, ref_price_inr]

Key CLI
-------
--t1 -10 --t2 10               Event window (start/end, in trading days; default -5 +5)
--est1 -120 --est2 -21         Estimation window for market model (default -120 to -21)
--min_est 60                   Min obs in estimation window (default 60)
--winsor 0.01                  Winsorize daily returns at tails (default 1%)
--outdir out_disinv            Output directory
--start / --end                Optional date filters (YYYY-MM-DD)

Outputs
-------
- daily_panel.csv              Prices/returns/liquidity aligned
- event_panel.csv              Per-event ARs/CARs, liquidity deltas, metadata
- car_by_type.csv              Mean CARs by event type & window(s)
- xsec_regression.csv          Cross-sectional OLS of CAR on drivers
- liquidity_changes.csv        Before/after liquidity metrics per event
- proceeds_by_fy.csv           FY (Apr–Mar) proceeds vs budget targets
- scenario_output.csv          Planned sales → proceeds & rough impact
- summary.json                 Headline diagnostics & configuration
- config.json                  Run configuration

DISCLAIMER
----------
This is research tooling. Event windows, mapping of stake % to free-float changes, and
price-impact approximations are simplified. Validate on your own data before use.
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

def to_day(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def winsorize(s: pd.Series, p: float=0.01) -> pd.Series:
    x = s.copy()
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

def amihud_illiq(abs_ret: pd.Series, value_traded: pd.Series) -> pd.Series:
    vt = value_traded.replace(0, np.nan).astype(float)
    return abs_ret.astype(float) / vt

def fiscal_year(dt: pd.Timestamp) -> str:
    y = dt.year
    return f"FY{y+1}" if dt.month >= 4 else f"FY{y}"


# ----------------------------- loaders -----------------------------

def load_prices(path: str, winsor_p: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    sym = ncol(df, "symbol","ticker","scrip","name","isin")
    close = ncol(df, "close","px","price","last")
    vol = ncol(df, "volume","vol")
    vval = ncol(df, "value_traded","traded_value","turnover_value","dollar_volume","rupee_volume")
    ffs  = ncol(df, "free_float_shares","freefloat_shares","ff_shares")
    sho  = ncol(df, "shares_outstanding","shares_out","so")
    if not (dt and sym and close):
        raise ValueError("prices.csv needs date, symbol, close columns.")
    df = df.rename(columns={dt:"date", sym:"symbol", close:"close"})
    df["date"] = to_day(df["date"])
    for c in [vol, vval, ffs, sho]:
        if c and c in df.columns:
            df[c] = safe_num(df[c])
    if vol: df = df.rename(columns={vol:"volume"})
    if vval: df = df.rename(columns={vval:"value_traded"})
    if ffs: df = df.rename(columns={ffs:"free_float_shares"})
    if sho: df = df.rename(columns={sho:"shares_outstanding"})
    df = df.sort_values(["symbol","date"])
    # returns
    df["ret"] = df.groupby("symbol")["close"].apply(lambda s: np.log(s) - np.log(s.shift(1)))
    if winsor_p and winsor_p>0:
        df["ret"] = df.groupby("symbol")["ret"].transform(lambda s: winsorize(s, winsor_p))
    # turnover & value traded fallback
    if "value_traded" not in df.columns or df["value_traded"].isna().all():
        if "volume" in df.columns:
            # approximate value traded using close*volume
            df["value_traded"] = df["close"] * df["volume"]
        else:
            df["value_traded"] = np.nan
    # amihud
    df["illiq"] = df.groupby("symbol").apply(lambda g: amihud_illiq(g["ret"].abs(), g["value_traded"])).reset_index(level=0, drop=True)
    # turnover (% of shares_outstanding)
    if "shares_outstanding" in df.columns:
        df["turnover"] = (df["volume"] / df["shares_outstanding"].replace(0, np.nan)) * 100.0 if "volume" in df.columns else np.nan
    else:
        df["turnover"] = np.nan
    return df

def load_index(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    close = ncol(df, "close","px","price","index")
    ret = ncol(df, "ret","return","returns")
    if not dt:
        raise ValueError("index.csv needs a date column.")
    df = df.rename(columns={dt:"date"})
    df["date"] = to_day(df["date"])
    if ret:
        df["mkt_ret"] = safe_num(df[ret])
    elif close:
        px = safe_num(df[close])
        df["mkt_ret"] = np.log(px) - np.log(px.shift(1))
    else:
        raise ValueError("index.csv needs either close or return column.")
    return df[["date","mkt_ret"]].dropna().sort_values("date")

def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    sym = ncol(df, "symbol","ticker","scrip","name","isin")
    ety = ncol(df, "type","event_type","kind")
    size = ncol(df, "size_pct","stake_pct","pct_sold","percent","stake")
    disc = ncol(df, "discount_pct","discount","ofs_discount")
    offer = ncol(df, "offer_price","floor_price","price")
    proc = ncol(df, "proceeds_inr","proceeds","amount_inr","amount")
    if not (dt and sym):
        raise ValueError("events.csv needs date and symbol columns.")
    df = df.rename(columns={dt:"date", sym:"symbol"})
    df["date"] = to_day(df["date"])
    if ety: df = df.rename(columns={ety:"type"})
    if size: df = df.rename(columns={size:"size_pct"})
    if disc: df = df.rename(columns={disc:"discount_pct"})
    if offer: df = df.rename(columns={offer:"offer_price"})
    if proc: df = df.rename(columns={proc:"proceeds_inr"})
    # normalize % fields to decimals? We'll keep them as **percent points** for readability; convert when needed.
    for c in ["size_pct","discount_pct"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    if "proceeds_inr" in df.columns:
        df["proceeds_inr"] = safe_num(df["proceeds_inr"])
    if "type" not in df.columns:
        df["type"] = "OFS"
    # unique key
    df["event_id"] = (df["symbol"].astype(str) + "_" + df["date"].dt.strftime("%Y%m%d") + "_" + df["type"].astype(str))
    return df.sort_values(["date","symbol"]).reset_index(drop=True)

def load_shareholding(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    sym = ncol(df, "symbol","ticker","scrip","name","isin")
    gov = ncol(df, "govt_stake_pct","government_pct","promoter_govt_pct","goi_pct")
    ff  = ncol(df, "free_float_pct","ff_pct")
    sho = ncol(df, "shares_outstanding","shares_out","so")
    if not (dt and sym): raise ValueError("shareholding.csv needs date and symbol.")
    df = df.rename(columns={dt:"date", sym:"symbol"})
    df["date"] = to_day(df["date"])
    if gov: df = df.rename(columns={gov:"govt_stake_pct"})
    if ff:  df = df.rename(columns={ff:"free_float_pct"})
    if sho: df = df.rename(columns={sho:"shares_outstanding"})
    for c in ["govt_stake_pct","free_float_pct","shares_outstanding"]:
        if c in df.columns: df[c] = safe_num(df[c])
    return df.sort_values(["symbol","date"])

def load_funda(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    sym = ncol(df, "symbol","ticker","scrip")
    if not (dt and sym): raise ValueError("fundamentals.csv needs date and symbol.")
    df = df.rename(columns={dt:"date", sym:"symbol"})
    df["date"] = to_day(df["date"])
    rename = {}
    for k in [("pb","pb"), ("p_b","pb"), ("price_to_book","pb"),
              ("roe","roe"), ("return_on_equity","roe"),
              ("dividend_yield","dividend_yield"), ("dy","dividend_yield")]:
        c = ncol(df, k[0])
        if c: rename[c] = k[1]
    df = df.rename(columns=rename)
    for c in df.columns:
        if c not in ["date","symbol"]:
            df[c] = safe_num(df[c])
    return df[["date","symbol"] + [c for c in ["pb","roe","dividend_yield"] if c in df.columns]].dropna(how="all", subset=["pb","roe","dividend_yield"])

def load_budget(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    fy = ncol(df, "fiscal_year","fy","year")
    tgt = ncol(df, "target_inr_cr","target","budget_target_cr")
    real = ncol(df, "realized_inr_cr","realized","achieved_cr")
    if not fy: raise ValueError("budget.csv needs fiscal_year.")
    df = df.rename(columns={fy:"fiscal_year"})
    if tgt: df = df.rename(columns={tgt:"target_inr_cr"})
    if real: df = df.rename(columns={real:"realized_inr_cr"})
    for c in ["target_inr_cr","realized_inr_cr"]:
        if c in df.columns: df[c] = safe_num(df[c])
    return df

def load_etf(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    etf = ncol(df, "etf","fund")
    flw = ncol(df, "flow_inr","flow","amount_inr")
    reb = ncol(df, "is_rebalance","rebalance")
    if not (dt and etf and flw): raise ValueError("etf_flows.csv needs date, etf, flow_inr.")
    df = df.rename(columns={dt:"date", etf:"etf", flw:"flow_inr"})
    df["date"] = to_day(df["date"])
    df["flow_inr"] = safe_num(df["flow_inr"])
    if reb: df = df.rename(columns={reb:"is_rebalance"}); df["is_rebalance"] = df["is_rebalance"].astype(bool)
    else: df["is_rebalance"] = False
    return df[["date","etf","flow_inr","is_rebalance"]].sort_values("date")

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    sym = ncol(df, "symbol","ticker","scrip")
    stake = ncol(df, "stake_pct","stake","pct","size_pct")
    disc = ncol(df, "discount_pct","discount")
    refp = ncol(df, "ref_price_inr","ref_price","price")
    if not (sym and stake): raise ValueError("scenarios.csv needs symbol and stake_pct.")
    df = df.rename(columns={sym:"symbol", stake:"stake_pct"})
    if disc: df = df.rename(columns={disc:"discount_pct"})
    if refp: df = df.rename(columns={refp:"ref_price_inr"})
    for c in ["stake_pct","discount_pct","ref_price_inr"]:
        if c in df.columns: df[c] = safe_num(df[c])
    return df[["symbol","stake_pct","discount_pct","ref_price_inr"]]


# ----------------------------- core: market model & event framing -----------------------------

def build_market_model(prices: pd.DataFrame, index: pd.DataFrame,
                       est1: int, est2: int, min_est: int) -> pd.DataFrame:
    """
    Estimate alpha/beta per (symbol, event date) using window [t+est1, t+est2] relative to event date t.
    We'll compute later on-the-fly per event to respect symbol-specific calendars.
    Here we just merge market returns into daily prices.
    """
    df = prices.merge(index, on="date", how="left")
    return df

def window_slice(df: pd.DataFrame, center: pd.Timestamp, t1: int, t2: int) -> pd.DataFrame:
    # df indexed by date
    # We'll use trading-day offsets by rank order, not calendar days
    if df.empty: return df
    srt = df.sort_index()
    if center not in srt.index:
        # find the nearest previous trading day to anchor
        prior = srt.index[srt.index <= center]
        if len(prior)==0: return pd.DataFrame(columns=srt.columns)
        center = prior.max()
    loc = srt.index.get_loc(center)
    if isinstance(loc, slice):
        # take start of slice
        loc = loc.start
    i1 = max(0, loc + t1)
    i2 = min(len(srt)-1, loc + t2)
    return srt.iloc[i1:i2+1]

def est_window(df_sym: pd.DataFrame, center: pd.Timestamp, est1: int, est2: int) -> pd.DataFrame:
    return window_slice(df_sym, center, est1, est2)

def fit_alpha_beta(est: pd.DataFrame) -> Tuple[float,float]:
    if est.empty or est["mkt_ret"].isna().all() or est["ret"].isna().all():
        return 0.0, 0.0
    x = est["mkt_ret"].values.astype(float)
    y = est["ret"].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < max(10, int(0.5*len(x))):
        return 0.0, 0.0
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y[mask])
    a, b = float(beta[0]), float(beta[1])
    return a, b

def abnormal_returns(win: pd.DataFrame, a: float, b: float) -> pd.Series:
    # AR = r_i - (a + b*r_m)
    ri = win["ret"].astype(float)
    rm = win["mkt_ret"].astype(float)
    return ri - (a + b*rm)

def car(ar: pd.Series) -> float:
    return float(np.nansum(ar.values.astype(float)))


# ----------------------------- main analytics -----------------------------

def event_study(prx_idx: pd.DataFrame, events: pd.DataFrame,
                t1: int, t2: int, est1: int, est2: int, min_est: int,
                shph: pd.DataFrame, funda: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - EV panel with AR path, CAR, liquidity deltas, stake/discount, free-float changes, prior run-up
      - Liquidity changes summary per event
    """
    out_rows = []
    liq_rows = []
    # Pre-compute symbol groups
    for sym, g in prx_idx.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        evs = events[events["symbol"]==sym]
        if evs.empty: continue
        # shareholding series for deltas
        shp = shph[shph["symbol"]==sym].sort_values("date") if not shph.empty else pd.DataFrame()
        for _, ev in evs.iterrows():
            d0 = ev["date"]
            # estimation window
            est = est_window(g, d0, est1, est2)
            if est.shape[0] < min_est:
                a, b = 0.0, 1.0  # fallback to market-adjusted
            else:
                a, b = fit_alpha_beta(est)
            # event window
            win = window_slice(g, d0, t1, t2)
            if win.empty: continue
            AR = abnormal_returns(win, a, b)
            this_car = car(AR)
            # prior run-up ([-20,-2])
            pre = window_slice(g, d0, -20, -2)
            prior_runup = float(np.nansum(pre["ret"])) if not pre.empty else np.nan
            # liquidity: Amihud & turnover before/after
            pre_liq = window_slice(g, d0, -60, -1)
            post_liq = window_slice(g, d0, +1, +60)
            illiq_pre  = float(pre_liq["illiq"].mean()) if not pre_liq.empty else np.nan
            illiq_post = float(post_liq["illiq"].mean()) if not post_liq.empty else np.nan
            turn_pre   = float(pre_liq["turnover"].mean()) if not pre_liq.empty else np.nan
            turn_post  = float(post_liq["turnover"].mean()) if not post_liq.empty else np.nan
            # free-float change around nearest shareholding snapshots
            ff_delta = np.nan; gov_delta = np.nan
            if not shp.empty:
                # pick last known before d0 and first known after
                before = shp[shp["date"]<=d0].tail(1)
                after  = shp[shp["date"]>d0].head(1)
                if not before.empty and not after.empty:
                    if "free_float_pct" in shp.columns:
                        ff_delta = float(after["free_float_pct"].iloc[0] - before["free_float_pct"].iloc[0])
                    if "govt_stake_pct" in shp.columns:
                        gov_delta = float(after["govt_stake_pct"].iloc[0] - before["govt_stake_pct"].iloc[0])
            # fundamentals (closest prior)
            pb = roe = dy = np.nan
            if not funda.empty:
                fu = funda[(funda["symbol"]==sym) & (funda["date"]<=d0)].tail(1)
                if not fu.empty:
                    pb = float(fu["pb"].iloc[0]) if "pb" in fu.columns and pd.notna(fu["pb"].iloc[0]) else np.nan
                    roe = float(fu["roe"].iloc[0]) if "roe" in fu.columns and pd.notna(fu["roe"].iloc[0]) else np.nan
                    dy = float(fu["dividend_yield"].iloc[0]) if "dividend_yield" in fu.columns and pd.notna(fu["dividend_yield"].iloc[0]) else np.nan
            # pack
            out_rows.append({
                "event_id": ev["event_id"], "date": d0, "symbol": sym, "type": ev.get("type","OFS"),
                "size_pct": float(ev.get("size_pct", np.nan)) if pd.notna(ev.get("size_pct", np.nan)) else np.nan,
                "discount_pct": float(ev.get("discount_pct", np.nan)) if pd.notna(ev.get("discount_pct", np.nan)) else np.nan,
                "offer_price": float(ev.get("offer_price", np.nan)) if pd.notna(ev.get("offer_price", np.nan)) else np.nan,
                "proceeds_inr": float(ev.get("proceeds_inr", np.nan)) if pd.notna(ev.get("proceeds_inr", np.nan)) else np.nan,
                "alpha": a, "beta": b,
                "CAR_t1_t2": this_car,
                "prior_runup": prior_runup,
                "illiq_pre": illiq_pre, "illiq_post": illiq_post, "d_illiq": (illiq_post - illiq_pre) if (pd.notna(illiq_post) and pd.notna(illiq_pre)) else np.nan,
                "turn_pre": turn_pre, "turn_post": turn_post, "d_turn": (turn_post - turn_pre) if (pd.notna(turn_post) and pd.notna(turn_pre)) else np.nan,
                "ff_delta_pct": ff_delta, "govt_stake_delta_pct": gov_delta,
            })
            liq_rows.append({
                "event_id": ev["event_id"], "date": d0, "symbol": sym, "type": ev.get("type","OFS"),
                "illiq_pre": illiq_pre, "illiq_post": illiq_post, "d_illiq": (illiq_post - illiq_pre) if (pd.notna(illiq_post) and pd.notna(illiq_pre)) else np.nan,
                "turn_pre": turn_pre, "turn_post": turn_post, "d_turn": (turn_post - turn_pre) if (pd.notna(turn_post) and pd.notna(turn_pre)) else np.nan
            })
    EV = pd.DataFrame(out_rows).sort_values(["date","symbol"])
    LIQ = pd.DataFrame(liq_rows).sort_values(["date","symbol"])
    return EV, LIQ

def car_by_type(ev_panel: pd.DataFrame) -> pd.DataFrame:
    if ev_panel.empty: return pd.DataFrame()
    g = ev_panel.groupby("type")["CAR_t1_t2"].agg(["count","mean","median","std"]).reset_index()
    g = g.rename(columns={"mean":"car_mean","median":"car_median","std":"car_sd","count":"n"})
    return g.sort_values("car_mean", ascending=False)

def xsec_regression(ev_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional OLS:
      y = CAR ~ size_pct + discount_pct + ff_delta_pct + prior_runup + pb + roe + type dummies
    Percent inputs are in **pp**; we convert to decimals where appropriate.
    """
    if ev_panel.empty or ev_panel["CAR_t1_t2"].dropna().shape[0] < 10:
        return pd.DataFrame()
    df = ev_panel.copy()
    # build design
    y = df["CAR_t1_t2"].values.reshape(-1,1)
    Xcols = []
    def add(col, transform=None):
        if col in df.columns:
            v = df[col].astype(float)
            v = transform(v) if transform else v
            Xcols.append(v.values.reshape(-1,1))
            return True
        return False
    add("size_pct", lambda s: s/100.0)
    add("discount_pct", lambda s: s/100.0)
    add("ff_delta_pct", lambda s: s/100.0)
    add("prior_runup", None)
    for c in ["pb","roe"]:
        if c in df.columns:
            add(c, None)
        else:
            df[c] = np.nan  # keep consistent
    # type dummies
    if "type" in df.columns:
        dummies = pd.get_dummies(df["type"], prefix="type", drop_first=True)
        for c in dummies.columns:
            Xcols.append(dummies[c].values.reshape(-1,1))
    if not Xcols:
        return pd.DataFrame()
    X = np.column_stack([np.ones((len(df),1))] + Xcols)
    # drop rows with any NaNs in X or y
    mask = np.isfinite(y).all(axis=1)
    for i in range(X.shape[1]):
        mask = mask & np.isfinite(X[:,i])
    X = X[mask,:]; y = y[mask,:]
    if X.shape[0] < max(20, X.shape[1]*5):
        return pd.DataFrame()
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    resid = y - X @ beta
    # simple (non-HAC) SE
    s2 = float((resid.T @ resid) / (X.shape[0] - X.shape[1]))
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov)).reshape(-1,1)
    names = ["const","size_dec","disc_dec","ff_delta_dec","prior_runup","pb","roe"]
    if "type" in df.columns:
        dnames = pd.get_dummies(df["type"], prefix="type", drop_first=True).columns.tolist()
        names += dnames
    out = []
    for i, nm in enumerate(names[:X.shape[1]]):
        out.append({"var": nm, "coef": float(beta[i,0]), "se": float(se[i,0]),
                    "t_stat": float(beta[i,0]/se[i,0] if se[i,0]>0 else np.nan), "n": int(X.shape[0])})
    return pd.DataFrame(out)

def proceeds_vs_budget(events: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    if events.empty: return pd.DataFrame()
    ev = events.copy()
    ev["fiscal_year"] = ev["date"].apply(fiscal_year)
    # realized proceeds (sum of proceeds_inr if present)
    agg = ev.groupby("fiscal_year", as_index=False)["proceeds_inr"].sum(numeric_only=True).rename(columns={"proceeds_inr":"realized_inr"})
    # if proceeds missing, try to estimate from size * price * shares_out? (insufficient without shares_out/time); we keep realized if provided.
    if budget.empty:
        return agg
    out = budget.merge(agg, on="fiscal_year", how="left")
    # normalize columns to *_inr_cr
    for c in ["target_inr_cr","realized_inr_cr"]:
        if c in out.columns:
            out[c] = safe_num(out[c])
    if "realized_inr" in out.columns:
        out["realized_inr_cr_from_events"] = out["realized_inr"] / 1e7  # 1 crore = 1e7 INR
    return out

def scenario_engine(scn: pd.DataFrame, last_px: pd.DataFrame, last_shares: pd.DataFrame,
                    xsec: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate proceeds and a rough 1–10 day CAR using cross-section beta on 'size_dec' if available.
    Inputs:
      scn: symbol, stake_pct, discount_pct[, ref_price_inr]
      last_px: latest close per symbol
      last_shares: latest shares_outstanding or free_float_shares
    """
    if scn.empty: return pd.DataFrame()
    px = last_px.copy()
    sh = last_shares.copy()
    # impact elasticity
    beta_size = np.nan
    if not xsec.empty:
        r = xsec[xsec["var"]=="size_dec"]
        if not r.empty: beta_size = float(r["coef"].iloc[0])
    rows = []
    for _, r in scn.iterrows():
        sym = r["symbol"]
        stake = float(r["stake_pct"])
        disc  = float(r.get("discount_pct", 0.0)) if pd.notna(r.get("discount_pct", np.nan)) else 0.0
        refp  = float(r.get("ref_price_inr", np.nan)) if pd.notna(r.get("ref_price_inr", np.nan)) else np.nan
        pr = px.get(sym, np.nan)
        so = sh.get(sym, np.nan)
        price = refp if np.isfinite(refp) else pr
        if not (np.isfinite(price) and np.isfinite(so) and np.isfinite(stake)):
            rows.append({"symbol": sym, "note": "missing price/shares/stake", "proceeds_inr": np.nan, "car_est": np.nan})
            continue
        sold_shares = so * (stake/100.0)
        offer_price = price * (1.0 - disc/100.0)
        proceeds = sold_shares * offer_price
        # rough CAR impact = beta_size * size_decimal
        impact = (beta_size * (stake/100.0)) if np.isfinite(beta_size) else np.nan
        rows.append({"symbol": sym, "stake_pct": stake, "discount_pct": disc, "ref_price_inr": price,
                     "offer_price_inr": offer_price, "shares_sold": sold_shares,
                     "proceeds_inr": proceeds, "car_est": impact})
    return pd.DataFrame(rows)

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    prices: str
    index: str
    events: str
    shareholding: Optional[str]
    fundamentals: Optional[str]
    budget: Optional[str]
    etf_flows: Optional[str]
    scenarios: Optional[str]
    t1: int
    t2: int
    est1: int
    est2: int
    min_est: int
    winsor: float
    outdir: str
    start: Optional[str]
    end: Optional[str]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PSU disinvestment analytics (events, CARs, liquidity & proceeds)")
    ap.add_argument("--prices", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--shareholding", default="")
    ap.add_argument("--fundamentals", default="")
    ap.add_argument("--budget", default="")
    ap.add_argument("--etf_flows", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--t1", type=int, default=-5)
    ap.add_argument("--t2", type=int, default=+5)
    ap.add_argument("--est1", type=int, default=-120)
    ap.add_argument("--est2", type=int, default=-21)
    ap.add_argument("--min_est", type=int, default=60)
    ap.add_argument("--winsor", type=float, default=0.01)
    ap.add_argument("--outdir", default="out_disinv")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    PRICES = load_prices(args.prices, winsor_p=float(args.winsor))
    INDEX  = load_index(args.index)
    EVENTS = load_events(args.events)
    SHPH   = load_shareholding(args.shareholding) if args.shareholding else pd.DataFrame()
    FUNDA  = load_funda(args.fundamentals) if args.fundamentals else pd.DataFrame()
    BUDGET = load_budget(args.budget) if args.budget else pd.DataFrame()
    ETF    = load_etf(args.etf_flows) if args.etf_flows else pd.DataFrame()
    SCN    = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Date filters
    if args.start:
        start = pd.to_datetime(args.start)
        for df in [PRICES, INDEX, EVENTS, SHPH, FUNDA, ETF]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] < start].index, inplace=True)
    if args.end:
        end = pd.to_datetime(args.end)
        for df in [PRICES, INDEX, EVENTS, SHPH, FUNDA, ETF]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] > end].index, inplace=True)

    # Merge market returns into prices
    PRX_IDX = build_market_model(PRICES, INDEX, args.est1, args.est2, args.min_est)
    PRX_IDX.to_csv(outdir / "daily_panel.csv", index=False)

    # Event study
    EV, LIQ = event_study(PRX_IDX, EVENTS, t1=int(args.t1), t2=int(args.t2),
                          est1=int(args.est1), est2=int(args.est2), min_est=int(args.min_est),
                          shph=SHPH, funda=FUNDA)
    if not EV.empty:
        EV.to_csv(outdir / "event_panel.csv", index=False)
    if not LIQ.empty:
        LIQ.to_csv(outdir / "liquidity_changes.csv", index=False)

    # CAR by type
    TYPE = car_by_type(EV) if not EV.empty else pd.DataFrame()
    if not TYPE.empty:
        TYPE.to_csv(outdir / "car_by_type.csv", index=False)

    # Cross-sectional regression
    XSEC = xsec_regression(EV) if not EV.empty else pd.DataFrame()
    if not XSEC.empty:
        XSEC.to_csv(outdir / "xsec_regression.csv", index=False)

    # Proceeds vs budget (FY)
    FY = proceeds_vs_budget(EV[["date","proceeds_inr"]].merge(EVENTS[["event_id","date","symbol","type"]], on="date", how="right").drop_duplicates(subset=["date","symbol"]),
                            BUDGET)
    if not FY.empty:
        FY.to_csv(outdir / "proceeds_by_fy.csv", index=False)

    # Scenario engine
    SCN_OUT = pd.DataFrame()
    if not SCN.empty:
        # last close & shares per symbol
        last_px = PRICES.sort_values("date").groupby("symbol")["close"].last().to_dict()
        # prefer shares_outstanding over free_float_shares for stake % math
        if "shares_outstanding" in PRICES.columns:
            last_sh = PRICES.sort_values("date").groupby("symbol")["shares_outstanding"].last().to_dict()
        else:
            last_sh = PRICES.sort_values("date").groupby("symbol")["free_float_shares"].last().to_dict() if "free_float_shares" in PRICES.columns else {}
        SCN_OUT = scenario_engine(SCN, last_px, last_sh, XSEC)
        if not SCN_OUT.empty:
            SCN_OUT.to_csv(outdir / "scenario_output.csv", index=False)

    # Summary
    summary = {
        "rows_prices": int(len(PRICES)),
        "rows_events": int(len(EVENTS)),
        "symbols": sorted(PRICES["symbol"].unique().tolist()),
        "car_window": {"t1": int(args.t1), "t2": int(args.t2)},
        "est_window": {"est1": int(args.est1), "est2": int(args.est2), "min_est": int(args.min_est)},
        "events_with_results": int(len(EV)) if not EV.empty else 0,
        "by_type": (TYPE.to_dict(orient="records") if not TYPE.empty else []),
        "xsec_terms": (XSEC.to_dict(orient="records") if not XSEC.empty else []),
        "scenarios": (SCN_OUT.to_dict(orient="records") if not SCN_OUT.empty else []),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        prices=args.prices, index=args.index, events=args.events,
        shareholding=(args.shareholding or None), fundamentals=(args.fundamentals or None),
        budget=(args.budget or None), etf_flows=(args.etf_flows or None), scenarios=(args.scenarios or None),
        t1=int(args.t1), t2=int(args.t2), est1=int(args.est1), est2=int(args.est2),
        min_est=int(args.min_est), winsor=float(args.winsor),
        outdir=args.outdir, start=(args.start or None), end=(args.end or None)
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== PSU Disinvestment Analytics ==")
    print(f"Events: {summary['rows_events']:,} | Symbols: {len(summary['symbols'])} | Results: {summary['events_with_results']:,}")
    if summary["by_type"]:
        top = max(summary["by_type"], key=lambda r: (r.get("car_mean") or -9e9))
        print(f"Top avg CAR by type: {top.get('type')} → {top.get('car_mean'):+.3%}")
    if summary["scenarios"]:
        est = summary["scenarios"][0]
        print(f"Scenario sample: {est.get('symbol')} proceeds ₹{est.get('proceeds_inr', 0):,.0f}")

if __name__ == "__main__":
    main()
