#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fii_dii_flows.py — Analyze FII vs DII equity flows & their link to market returns
---------------------------------------------------------------------------------

What this does
==============
Given daily India equity flow data for Foreign Institutional Investors (FII)
and Domestic Institutional Investors (DII), plus an index time series (e.g.
NIFTY 50), this script:

1) Cleans/aligns flow & index data (flexible headers; INR crores expected).
2) Builds derived series:
   • Net flows (FII, DII, NET = FII − DII) and rolling z-scores
   • Market returns (pct & log) and rolling betas of returns on flows
3) Computes rolling diagnostics (short/med/long windows):
   • Corr(FII, DII), Corr(Flow, Return), Flow→Return beta & R²
4) Lead–lag cross-correlations for lags −L..+L:
   • Corr(Flow_{t−k}, Return_t) for FII & DII
5) Detects “divergence days”: FII and DII net flows with opposite signs
   beyond chosen thresholds; tags who is “absorbing” supply.
6) Produces tidy CSVs + JSON summary.

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--flows flows.csv        REQUIRED
  Accepts either:
    (A) date, fii_net, dii_net        (net buy/sell in INR crores; + = buy)
    (B) date, fii_buy, fii_sell, dii_buy, dii_sell  (we compute nets)
  Column name examples it will pick up automatically:
    "fii", "fii_net", "fii_value", "fii_net_buy"
    "dii", "dii_net", "dii_value", "dii_net_buy"
    or "fii_buy"/"fii_sell", "dii_buy"/"dii_sell".

--index index.csv        OPTIONAL but recommended
  Columns: date, close   (or any price column you specify with --index_col)

CLI (example)
-------------
python fii_dii_flows.py \
  --flows flows.csv --index index.csv --index_col CLOSE \
  --wshort 20 --wmed 60 --wlong 120 --lags 10 --div_thresh 250 \
  --start 2020-01-01 --outdir out_fii_dii

Outputs
-------
- panel.csv               Date-aligned series (flows, returns, z-scores, betas)
- rolling_stats.csv       Rolling correlations & betas (short/med/long)
- leadlag_corr.csv        Lead–lag correlation table (lags × {FII,DII})
- divergences.csv         Opposite-sign flow events with stats
- summary.json            Latest snapshot & headline diagnostics
- config.json             Run configuration

Notes
-----
• Units assumed INR crores. If your file is in INR billions or ₹, convert before.
• Returns are daily simple pct returns; log returns also provided.
• Lead–lag: positive lag k means using Flow_{t−k} (flow leads returns by k days).

DISCLAIMER
----------
Research tooling. Past relationships may not persist; validate before use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    """Find a column by exact/lazy match (case-insensitive)."""
    t = target.lower()
    for c in df.columns:
        if str(c).lower() == t: return c
    for c in df.columns:
        if t in str(c).lower(): return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan).astype(float)).diff()

def roll_corr(x: pd.Series, y: pd.Series, window: int, minp: Optional[int]=None) -> pd.Series:
    minp = minp or max(10, window // 3)
    return x.rolling(window, min_periods=minp).corr(y)

def roll_beta(y: pd.Series, x: pd.Series, window: int, minp: Optional[int]=None) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS beta and R² for y ~ a + b*x (one regressor).
    Returns (beta, r2).
    """
    minp = minp or max(10, window // 3)
    x_ = x.astype(float); y_ = y.astype(float)
    mx = x_.rolling(window, min_periods=minp).mean()
    my = y_.rolling(window, min_periods=minp).mean()
    cov = (x_*y_).rolling(window, min_periods=minp).mean() - mx*my
    varx = (x_*x_).rolling(window, min_periods=minp).mean() - mx*mx
    beta = cov / varx.replace(0, np.nan)
    # R² = corr^2
    r = roll_corr(x_, y_, window, minp)
    r2 = r*r
    return beta, r2

def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(10, window//3)).mean()
    sd = s.rolling(window, min_periods=max(10, window//3)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)

def leadlag_corr(flow: pd.Series, ret: pd.Series, max_lag: int) -> pd.DataFrame:
    """
    Corr(flow_{t-k}, ret_t) for k = -max_lag..+max_lag.
    k>0 means flow leads returns by k days (use past flows).
    k<0 means returns lead flows (use future flows).
    """
    rows = []
    for k in range(-max_lag, max_lag+1):
        if k >= 0:
            f = flow.shift(k)
            r = ret
        else:
            f = flow
            r = ret.shift(-k)
        c = f.corr(r)
        rows.append({"lag": k, "corr": float(c) if c == c else np.nan})
    return pd.DataFrame(rows)

# ----------------------------- loaders -----------------------------

def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt: "date"})
    df["date"] = to_date(df["date"])
    # Try to find net columns
    fii_net_c = ncol(df, "fii_net") or ncol(df, "fii_value") or ncol(df, "fii") or ncol(df, "fii_net_buy")
    dii_net_c = ncol(df, "dii_net") or ncol(df, "dii_value") or ncol(df, "dii") or ncol(df, "dii_net_buy")
    if fii_net_c and dii_net_c:
        fii = safe_num(df[fii_net_c])
        dii = safe_num(df[dii_net_c])
    else:
        # Compute nets if buy/sell provided
        fii_buy = ncol(df, "fii_buy")
        fii_sell = ncol(df, "fii_sell")
        dii_buy = ncol(df, "dii_buy")
        dii_sell = ncol(df, "dii_sell")
        if not all([fii_buy, fii_sell, dii_buy, dii_sell]):
            raise ValueError("flows.csv must include FII/DII net columns or both buy/sell to compute nets.")
        fii = safe_num(df[fii_buy]) - safe_num(df[fii_sell])
        dii = safe_num(df[dii_buy]) - safe_num(df[dii_sell])
    out = (pd.DataFrame({"date": df["date"], "FII": fii, "DII": dii})
             .dropna(subset=["date"])
             .sort_values("date")
             .drop_duplicates(subset=["date"]))
    return out

def load_index(path: str, price_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    dt = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={dt: "date"})
    df["date"] = to_date(df["date"])
    pc = price_col or ncol(df, "close") or ncol(df, "price") or ncol(df, "px_last")
    if not pc:
        # pick first non-date numeric col
        num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("index.csv: could not find price column; pass --index_col.")
        pc = num_cols[0]
    px = safe_num(df[pc])
    return (pd.DataFrame({"date": df["date"], "PX": px})
              .dropna(subset=["date"])
              .sort_values("date")
              .drop_duplicates(subset=["date"]))

# ----------------------------- core calc -----------------------------

def build_panel(F: pd.DataFrame, I: Optional[pd.DataFrame],
                wshort: int, wmed: int, wlong: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      panel: per-date series
      roll: rolling stats (short/med/long)
    """
    df = F.copy().sort_values("date").set_index("date")
    df["NET"] = df["FII"] - df["DII"]

    # attach index returns if available
    if I is not None and not I.empty:
        I = I.sort_values("date").set_index("date")
        I["ret"] = I["PX"].pct_change()
        I["lret"] = dlog(I["PX"])
        df = df.join(I[["PX","ret","lret"]], how="left")
    else:
        df["PX"] = np.nan
        df["ret"] = np.nan
        df["lret"] = np.nan

    # z-scores (use long window)
    df["FII_z"] = zscore(df["FII"], window=wlong)
    df["DII_z"] = zscore(df["DII"], window=wlong)
    df["NET_z"] = zscore(df["NET"], window=wlong)

    # rolling stats
    R = []
    for w, lab in [(wshort, "short"), (wmed, "med"), (wlong, "long")]:
        corr_fd = roll_corr(df["FII"], df["DII"], window=w)
        corr_fr = roll_corr(df["FII"], df["ret"], window=w)
        corr_dr = roll_corr(df["DII"], df["ret"], window=w)
        beta_fr, r2_fr = roll_beta(df["ret"], df["FII"], window=w)
        beta_dr, r2_dr = roll_beta(df["ret"], df["DII"], window=w)
        beta_nr, r2_nr = roll_beta(df["ret"], df["NET"], window=w)
        R.append(pd.DataFrame({
            "date": df.index,
            f"corr_fii_dii_{lab}": corr_fd.values,
            f"corr_fii_ret_{lab}": corr_fr.values,
            f"corr_dii_ret_{lab}": corr_dr.values,
            f"beta_fii_ret_{lab}": beta_fr.values,
            f"beta_dii_ret_{lab}": beta_dr.values,
            f"beta_net_ret_{lab}": beta_nr.values,
            f"r2_fii_ret_{lab}": r2_fr.values,
            f"r2_dii_ret_{lab}": r2_dr.values,
            f"r2_net_ret_{lab}": r2_nr.values,
        }).set_index("date"))
    roll = pd.concat(R, axis=1)

    panel = df.reset_index()
    roll = roll.reset_index()
    return panel, roll

def find_divergences(panel: pd.DataFrame, div_thresh: float) -> pd.DataFrame:
    """
    Opposite-sign days with magnitudes past threshold (absolute crores).
    """
    df = panel.copy().set_index("date")
    cond = ((df["FII"] * df["DII"] < 0) &
            (df["FII"].abs() >= div_thresh) &
            (df["DII"].abs() >= div_thresh))
    div = df[cond].copy()
    if div.empty:
        return pd.DataFrame(columns=["date","FII","DII","NET","absorber","note"])
    # who is absorbing? If FII>0 & DII<0, domestics supplying; absorber=FII (taking supply)
    def tag(r):
        if r["FII"]>0 and r["DII"]<0: return "FII_absorbing"
        if r["FII"]<0 and r["DII"]>0: return "DII_absorbing"
        return "mixed"
    div["absorber"] = div.apply(tag, axis=1)
    # note intensity via z-scores if present
    if "FII_z" in div.columns and "DII_z" in div.columns:
        div["note"] = (("FII_z=" + div["FII_z"].round(2).astype(str))
                       + ", DII_z=" + div["DII_z"].round(2).astype(str))
    else:
        div["note"] = ""
    return div.reset_index()[["date","FII","DII","NET","absorber","note"]]

def leadlag_tables(panel: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    df = panel.copy().set_index("date")
    if df["ret"].dropna().empty:
        return pd.DataFrame(columns=["lag","corr_fii_ret","corr_dii_ret","corr_net_ret"])
    t_fii = leadlag_corr(df["FII"], df["ret"], max_lag)
    t_dii = leadlag_corr(df["DII"], df["ret"], max_lag)
    t_net = leadlag_corr(df["NET"], df["ret"], max_lag)
    out = pd.DataFrame({"lag": t_fii["lag"],
                        "corr_fii_ret": t_fii["corr"],
                        "corr_dii_ret": t_dii["corr"],
                        "corr_net_ret": t_net["corr"]})
    return out

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    flows: str
    index: Optional[str]
    index_col: Optional[str]
    start: Optional[str]
    end: Optional[str]
    wshort: int
    wmed: int
    wlong: int
    lags: int
    div_thresh: float
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FII vs DII flow analytics")
    ap.add_argument("--flows", required=True, help="CSV with FII/DII flows (INR crores)")
    ap.add_argument("--index", default="", help="CSV with index prices (NIFTY, etc.)")
    ap.add_argument("--index_col", default="", help="Column name for index price (default: CLOSE/PRICE)")
    ap.add_argument("--start", default="", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="", help="End date (YYYY-MM-DD)")
    ap.add_argument("--wshort", type=int, default=20, help="Short window (days)")
    ap.add_argument("--wmed", type=int, default=60, help="Medium window (days)")
    ap.add_argument("--wlong", type=int, default=120, help="Long window (days)")
    ap.add_argument("--lags", type=int, default=10, help="Max lead–lag lags (days)")
    ap.add_argument("--div_thresh", type=float, default=250.0, help="Divergence threshold in INR crores (abs)")
    ap.add_argument("--outdir", default="out_fii_dii")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load
    FLOWS = load_flows(args.flows)
    IDX = None
    if args.index:
        IDX = load_index(args.index, price_col=(args.index_col or None))

    # Filter date range
    if args.start:
        FLOWS = FLOWS[FLOWS["date"] >= pd.to_datetime(args.start)]
        if IDX is not None: IDX = IDX[IDX["date"] >= pd.to_datetime(args.start)]
    if args.end:
        FLOWS = FLOWS[FLOWS["date"] <= pd.to_datetime(args.end)]
        if IDX is not None: IDX = IDX[IDX["date"] <= pd.to_datetime(args.end)]

    # Build panel & rolling stats
    panel, rolling = build_panel(FLOWS, IDX, args.wshort, args.wmed, args.wlong)
    panel.to_csv(outdir / "panel.csv", index=False)
    rolling.to_csv(outdir / "rolling_stats.csv", index=False)

    # Lead–lag correlations
    ll = leadlag_tables(panel, args.lags)
    if not ll.empty:
        ll.to_csv(outdir / "leadlag_corr.csv", index=False)

    # Divergences
    div = find_divergences(panel, args.div_thresh)
    if not div.empty:
        div.to_csv(outdir / "divergences.csv", index=False)

    # Quick headline summary
    latest = panel.dropna(subset=["FII","DII"]).tail(1)
    last_date = None if latest.empty else str(pd.to_datetime(latest["date"].iloc[0]).date())
    summary = {
        "last_date": last_date,
        "latest": {
            "FII_inr_cr": (float(latest["FII"].iloc[0]) if not latest.empty else None),
            "DII_inr_cr": (float(latest["DII"].iloc[0]) if not latest.empty else None),
            "NET_inr_cr": (float(latest["NET"].iloc[0]) if not latest.empty else None),
            "ret_pct": (float(latest["ret"].iloc[0])*100 if ("ret" in latest and not latest.empty and pd.notna(latest["ret"].iloc[0])) else None),
        },
        "sum_5d_inr_cr": {
            "FII": float(panel.set_index("date")["FII"].tail(5).sum()) if not panel.empty else None,
            "DII": float(panel.set_index("date")["DII"].tail(5).sum()) if not panel.empty else None,
            "NET": float(panel.set_index("date")["NET"].tail(5).sum()) if not panel.empty else None,
        },
        "rolling": {}
    }
    # Append latest rolling stats
    if not rolling.empty:
        last_roll = rolling.dropna(how="all").tail(1)
        if not last_roll.empty:
            for c in last_roll.columns:
                if c != "date" and pd.notna(last_roll[c].iloc[0]):
                    summary["rolling"][c] = float(last_roll[c].iloc[0])

    # Save summary & config
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    cfg = asdict(Config(
        flows=args.flows, index=(args.index or None), index_col=(args.index_col or None),
        start=(args.start or None), end=(args.end or None),
        wshort=int(args.wshort), wmed=int(args.wmed), wlong=int(args.wlong),
        lags=int(args.lags), div_thresh=float(args.div_thresh), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== FII vs DII ==")
    print(f"Rows: {panel.shape[0]} | From {panel['date'].min().date()} to {panel['date'].max().date()}")
    if last_date:
        print(f"Latest ({last_date}): FII {summary['latest']['FII_inr_cr']:+.0f} cr, "
              f"DII {summary['latest']['DII_inr_cr']:+.0f} cr, "
              f"NET {summary['latest']['NET_inr_cr']:+.0f} cr")
    if not ll.empty:
        best_lead = ll.loc[ll["corr_fii_ret"].abs().idxmax()]
        print(f"Lead–lag: max |corr(FII,ret)| at lag {int(best_lead['lag'])} → {best_lead['corr_fii_ret']:+.2f}")
    if not div.empty:
        print(f"Divergences detected: {div.shape[0]} (threshold {args.div_thresh:.0f} cr)")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
