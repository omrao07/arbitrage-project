#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rate_convergence.py — Sovereign yield-spread convergence engine & backtester
----------------------------------------------------------------------------

What this does
==============
Given government bond yields by country & tenor, this script:
1) Builds anchor-vs-peer *convergence pairs* per tenor (e.g., IT vs DE on 10Y)
2) Computes *beta-hedged residual spreads* via rolling OLS (peer ~ a + b*anchor)
3) Estimates *mean-reversion* stats (rolling z-score, OU half-life via AR(1))
4) Generates *convergence trades* when residual z passes thresholds, with exits
   on mean reversion; P&L via duration-based DV01 approximation
5) Produces tidy CSVs + a JSON summary

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--yields yields.csv   REQUIRED
  Long  : date, country, tenor, yield_pct
  Wide  : date, DE_10Y, IT_10Y, ES_10Y, ...   (columns named CC_TENOR or CCxxY)
  Notes : yield units can be % (e.g., 2.35) or decimal (0.0235)

--pairs pairs.csv     OPTIONAL (to override auto pair-building)
  Columns: tenor, peer, anchor     (e.g., 10Y, IT, DE)

--durations durations.csv OPTIONAL (duration by country/tenor; else use defaults)
  Columns: country, tenor, duration_years

CLI
---
Example:
  python rate_convergence.py --yields yields.csv --anchor DE \
    --tenors 2,5,10 --z_entry 2.0 --z_exit 0.5 --beta_window 252 \
    --lookback 252 --max_hold 60 --outdir out_convergence

Key parameters
--------------
--anchor DE                Default anchor ISO-2 (e.g., DE, US)
--tenors 2,5,10            Comma list or leave blank to infer from data
--beta_window 252          Rolling window for regression beta (days)
--lookback 252             Rolling window for z-score (days)
--z_entry 2.0              Enter when |z| >= z_entry
--z_exit 0.5               Exit when |z| <= z_exit
--max_hold 60              Max holding days per trade
--slippage_bp 0.0          Per-leg slippage in bp (DV01-based P&L)
--outdir out_convergence   Output folder

Outputs
-------
- pairs_used.csv              Final list of tenor×(peer,anchor) pairs
- spreads_panel.csv           Per date×pair metrics: spread, beta-residual, z, half-life
- trades.csv                  Trade log: entries/exits, side, P&L
- backtest_pnl.csv            Daily P&L per pair & aggregate
- summary.json                Performance & diagnostics
- config.json                 Run configuration

Method notes
------------
• Hedged residual e_t = peer_y_t − (a_t + b_t*anchor_y_t), with a_t,b_t from rolling OLS.
• Z-score = (e_t − μ_t)/σ_t on a rolling window (lookback).
• Half-life: AR(1) on residual x_t = c + φ x_{t−1} + ε ⇒ HL = −ln(2)/ln(φ), if 0<φ<1.
• P&L (per $100 notional **on peer leg**):  ΔP ≈ −D_peer*Δy_peer  +  b_t * D_anchor * Δy_anchor.
  Direction: if z>0 (peer “too wide”), go LONG peer (receive duration) & SHORT anchor; opposite if z<0.
• Durations: default {2Y:1.9, 5Y:4.5, 10Y:8.5}. Override via durations.csv.

DISCLAIMER
----------
Research tool only. Approximations (duration-linearity, rolling OLS) ignore convexity, carry/roll, and execution frictions.
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

def pct_to_dec(s: pd.Series) -> pd.Series:
    # Treat >1 as percent (e.g., 2.3 => 0.023), else as decimals already
    out = pd.to_numeric(s, errors="coerce")
    if out.dropna().empty:
        return out
    # Heuristic: if median > 1.5, assume percent
    if np.nanmedian(out.values) > 1.5:
        out = out / 100.0
    return out

def clean_tenor(x: str) -> str:
    if not isinstance(x, str): x = str(x)
    u = x.upper().strip().replace("YR","Y").replace("YEAR","Y").replace("YEARS","Y")
    if u.endswith("Y"): return u
    # digits only like '10' -> '10Y'
    if u.isdigit(): return f"{u}Y"
    # handle like CC_10Y or CC10
    if u[-1].isdigit() and u[:-1].isalpha():
        return u[-1] + "Y" if len(u)==2 else u
    return u

def parse_cc_tenor(col: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to parse wide column like 'DE_10Y' or 'IT10Y' into (country, tenor).
    """
    u = col.upper()
    if "_" in u:
        parts = u.split("_")
        if len(parts)>=2 and parts[0].isalpha():
            cc = parts[0][:3]
            ten = clean_tenor(parts[1])
            return cc, ten
    # e.g., 'DE10Y'
    if len(u)>=4 and u[:2].isalpha() and u[-1]=="Y":
        cc = u[:2]
        ten = clean_tenor(u[2:])
        return cc, ten
    return None, None

def rolling_ols(y: pd.Series, x: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS of y on x: y = a + b*x. Returns (a_t, b_t).
    Uses rolling means/covs for efficiency.
    """
    x = x.astype(float); y = y.astype(float)
    mx = x.rolling(window, min_periods=max(20, window//4)).mean()
    my = y.rolling(window, min_periods=max(20, window//4)).mean()
    cov = (x*y).rolling(window, min_periods=max(20, window//4)).mean() - mx*my
    var = (x*x).rolling(window, min_periods=max(20, window//4)).mean() - mx*mx
    b = cov / var.replace(0, np.nan)
    a = my - b*mx
    return a, b

def rolling_z(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(20, window//4)).mean()
    sd = s.rolling(window, min_periods=max(20, window//4)).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)

def ar1_halflife(x: pd.Series, window: int) -> pd.Series:
    """
    Estimate half-life via rolling AR(1): x_t = c + φ x_{t-1} + ε.
    HL = -ln(2)/ln(φ), if 0<φ<1 else NaN.
    """
    x = x.astype(float)
    lag = x.shift(1)
    # Rolling moments
    mx = x.rolling(window, min_periods=max(20, window//4)).mean()
    ml = lag.rolling(window, min_periods=max(20, window//4)).mean()
    cov = (x*lag).rolling(window, min_periods=max(20, window//4)).mean() - mx*ml
    var = (lag*lag).rolling(window, min_periods=max(20, window//4)).mean() - ml*ml
    phi = cov / var.replace(0, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        hl = -np.log(2) / np.log(phi)
    hl[(phi<=0) | (phi>=1) | (~np.isfinite(hl))] = np.nan
    return hl

# ----------------------------- loaders -----------------------------

def load_yields(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ncol(df,"country") or ncol(df,"tenor"):
        # Long format
        ren = {
            (ncol(df,"date") or df.columns[0]): "date",
            (ncol(df,"country") or "country"): "country",
            (ncol(df,"tenor") or "tenor"): "tenor",
            (ncol(df,"yield_pct") or ncol(df,"yield") or ncol(df,"yld") or "yield"): "yield"
        }
        df = df.rename(columns=ren)
        df["date"] = to_date(df["date"])
        df["country"] = df["country"].astype(str).str.upper().str.strip()
        df["tenor"] = df["tenor"].astype(str).map(clean_tenor)
        df["yield"] = pct_to_dec(df["yield"])
        piv = df.pivot_table(index="date", columns=["tenor","country"], values="yield", aggfunc="last").sort_index()
    else:
        # Wide format: attempt to parse columns into (tenor,country)
        date_c = ncol(df,"date") or df.columns[0]
        df = df.rename(columns={date_c:"date"})
        df["date"] = to_date(df["date"])
        cols = [c for c in df.columns if c!="date"]
        recs = []
        for c in cols:
            cc, ten = parse_cc_tenor(c)
            if cc and ten:
                recs.append((c, ten, cc))
        if not recs:
            raise ValueError("Could not parse wide yield columns. Use long format or columns like 'DE_10Y', 'IT10Y'.")
        # Melt to long & pivot into tidy (tenor,country)
        m = []
        for c, ten, cc in recs:
            sub = df[["date", c]].copy()
            sub["tenor"] = ten
            sub["country"] = cc
            sub = sub.rename(columns={c:"yield"})
            m.append(sub)
        long = pd.concat(m, ignore_index=True)
        long["yield"] = pct_to_dec(long["yield"])
        piv = long.pivot_table(index="date", columns=["tenor","country"], values="yield", aggfunc="last").sort_index()
    return piv  # MultiIndex columns: (tenor, country)

def load_pairs(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"tenor") or "tenor"): "tenor",
        (ncol(df,"peer") or "peer"): "peer",
        (ncol(df,"anchor") or "anchor"): "anchor",
    }
    df = df.rename(columns=ren)
    df["tenor"] = df["tenor"].astype(str).map(clean_tenor)
    df["peer"] = df["peer"].astype(str).str.upper().str.strip()
    df["anchor"] = df["anchor"].astype(str).str.upper().str.strip()
    return df

def load_durations(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {
        (ncol(df,"country") or "country"): "country",
        (ncol(df,"tenor") or "tenor"): "tenor",
        (ncol(df,"duration_years") or ncol(df,"duration") or "duration_years"): "duration_years",
    }
    df = df.rename(columns=ren)
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["tenor"] = df["tenor"].astype(str).map(clean_tenor)
    df["duration_years"] = safe_num(df["duration_years"])
    return df

# ----------------------------- core calc -----------------------------

DEFAULT_DUR = {"2Y": 1.9, "3Y": 2.7, "5Y": 4.5, "7Y": 6.5, "10Y": 8.5, "20Y": 13.0, "30Y": 18.0}

def duration_lookup(country: str, tenor: str, DUR: pd.DataFrame) -> float:
    if not DUR.empty:
        row = DUR[(DUR["country"]==country) & (DUR["tenor"]==tenor)]
        if not row.empty and pd.notna(row["duration_years"].iloc[0]):
            return float(row["duration_years"].iloc[0])
    return DEFAULT_DUR.get(tenor, 8.0)

def build_pairs_from_panel(panel: pd.DataFrame, anchor: str, tenors: Optional[List[str]]) -> pd.DataFrame:
    all_tenors = sorted(set([t for (t, c) in panel.columns]))
    use_tenors = [clean_tenor(x) for x in tenors] if tenors else all_tenors
    rows = []
    for ten in use_tenors:
        countries = sorted([c for (t, c) in panel.columns if t==ten])
        if anchor not in countries: 
            continue
        for peer in countries:
            if peer == anchor: 
                continue
            rows.append({"tenor": ten, "peer": peer, "anchor": anchor})
    return pd.DataFrame(rows)

def compute_pair_metrics(panel: pd.DataFrame, pairs: pd.DataFrame, beta_window: int, lookback: int,
                         DUR: pd.DataFrame, slippage_bp: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      spreads_panel: per-date×pair metrics
      trades: entry/exit log with PnL
      pnl_daily: daily PnL per pair and aggregate
    """
    out_rows = []
    trade_rows = []
    pnl_cols = {}

    for _, pr in pairs.iterrows():
        ten, peer, anchor = pr["tenor"], pr["peer"], pr["anchor"]
        if (ten, peer) not in panel.columns or (ten, anchor) not in panel.columns:
            continue
        y_peer = panel[(ten, peer)].astype(float)
        y_anch = panel[(ten, anchor)].astype(float)

        # Rolling OLS (peer on anchor)
        a_t, b_t = rolling_ols(y_peer, y_anch, window=beta_window)

        # Hedged residual
        resid = y_peer - (a_t + b_t*y_anch)
        z = rolling_z(resid, lookback)
        hl = ar1_halflife(resid, lookback)

        spread = (y_peer - y_anch)  # raw spread (decimals)

        # Durations
        D_peer = duration_lookup(peer, ten, DUR)
        D_anch = duration_lookup(anchor, ten, DUR)

        # Daily P&L (per $100 notional on peer leg) for a unit *long-convergence* position (+1)
        # PnL_unit = -D_peer * Δy_peer + b_t * D_anch * Δy_anchor
        dy_peer = y_peer.diff()
        dy_anch = y_anch.diff()
        pnl_unit = (-D_peer * dy_peer + b_t.shift(1) * D_anch * dy_anch) * 100.0  # $ per 100 notional
        # Slippage in bp per leg: subtract |slip| on entry+exit days; we will apply in trade P&L.

        # Signal & position (state machine)
        z_entry = args.z_entry
        z_exit  = args.z_exit
        max_hold = int(args.max_hold)

        pos = pd.Series(0, index=panel.index)  # +1 long convergence (peer too wide), -1 short convergence
        entry_idx = None
        entry_side = 0  # +1 or -1
        hold_days = 0

        for t in panel.index:
            zt = z.loc[t]
            if entry_idx is None:
                # Check entry
                if pd.notna(zt) and abs(zt) >= z_entry:
                    entry_side = -1 if zt < 0 else +1  # if z<0, peer too tight: short convergence
                    entry_idx = t
                    hold_days = 0
                    pos.loc[t] = entry_side
                    # mark trade open
                else:
                    pos.loc[t] = 0
            else:
                # Already in a trade
                hold_days += 1
                pos.loc[t] = entry_side
                # Check exit
                exit_cond = (pd.notna(zt) and abs(zt) <= z_exit) or (hold_days >= max_hold)
                if exit_cond:
                    # record trade
                    t0 = entry_idx
                    t1 = t
                    # P&L over (t0, t1]: sum(pnl_unit * side)
                    pnl_series = pnl_unit.loc[t0:t1].copy()
                    trade_pnl = float(pnl_series.sum() * entry_side)
                    # Apply slippage (bp per leg on entry and exit): 2 legs × 2 touches × bp × $DV01 (~ duration * 1bp * 100)
                    slip_cash = 0.0
                    if slippage_bp and np.isfinite(slippage_bp):
                        dv01_peer = D_peer * 0.01  # $ per 1bp per $100 notional
                        dv01_anch = D_anch * 0.01
                        # two touches each leg
                        slip_cash = (abs(slippage_bp) / 1.0) * (dv01_peer + dv01_anch) * 2.0
                    trade_rows.append({
                        "pair": f"{ten}:{peer}_vs_{anchor}",
                        "tenor": ten, "peer": peer, "anchor": anchor,
                        "entry_date": t0, "exit_date": t1, "side": entry_side,
                        "days_held": hold_days, "gross_pnl_per_100": trade_pnl, "slippage_cost": slip_cash,
                        "net_pnl_per_100": trade_pnl - slip_cash,
                        "entry_z": float(z.loc[t0]) if pd.notna(z.loc[t0]) else np.nan,
                        "exit_z": float(zt) if pd.notna(zt) else np.nan,
                        "avg_beta": float(b_t.loc[t0:t1].mean()) if pd.notna(b_t.loc[t0:t1]).any() else np.nan
                    })
                    # reset
                    entry_idx = None
                    entry_side = 0
                    hold_days = 0

        # Build per-date rows for output
        dfp = pd.DataFrame({
            "date": panel.index,
            "pair": f"{ten}:{peer}_vs_{anchor}",
            "tenor": ten,
            "peer": peer,
            "anchor": anchor,
            "yield_peer": y_peer.values,
            "yield_anchor": y_anch.values,
            "spread_raw": spread.values,
            "beta": b_t.values,
            "alpha": a_t.values,
            "residual": resid.values,
            "zscore": z.values,
            "half_life_days": hl.values,
            "position": pos.values,
            "pnl_unit_per_100": pnl_unit.values
        }).dropna(subset=["yield_peer","yield_anchor"], how="all")
        out_rows.append(dfp)

        # Daily strategy P&L: position (carry neutralized) × unit P&L
        pnl_cols[dfp["pair"].iloc[0]] = (dfp["position"] * dfp["pnl_unit_per_100"]).fillna(0.0).values

    spreads_panel = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

    # Trades table
    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades["entry_date"] = pd.to_datetime(trades["entry_date"])
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    # Daily P&L panel
    if spreads_panel.empty:
        pnl_daily = pd.DataFrame()
    else:
        idx = sorted(spreads_panel["date"].unique())
        pnl_daily = pd.DataFrame({"date": idx}).set_index("date")
        for k, v in pnl_cols.items():
            # Align length
            s = pd.Series(v, index=idx) if len(v)==len(idx) else pd.Series(v, index=idx[:len(v)])
            pnl_daily[k] = s
        pnl_daily["aggregate"] = pnl_daily.sum(axis=1)

    return spreads_panel, trades, pnl_daily.reset_index()

def perf_stats_from_daily(pnl_daily: pd.Series) -> Dict[str, float]:
    if pnl_daily.dropna().empty:
        return {"cagr_pct": np.nan, "vol_pct": np.nan, "sharpe": np.nan, "hit_rate": np.nan, "max_dd_pct": np.nan}
    # Convert $ per 100 into return proxy by dividing by notional 100
    r = pnl_daily.fillna(0.0) / 100.0
    idx = (1 + r).cumprod()
    yrs = (idx.index[-1] - idx.index[0]).days / 365.25
    cagr = idx.iloc[-1] ** (1/yrs) - 1 if yrs>0 else np.nan
    vol = r.std(ddof=0) * np.sqrt(252)
    sharpe = (cagr / vol) if (vol and vol>0) else np.nan
    # hit rate (daily)
    hit = (r > 0).mean()
    # max drawdown
    peak = idx.cummax()
    dd = (idx / peak - 1.0).min()
    return {"cagr_pct": float(cagr*100) if cagr==cagr else np.nan,
            "vol_pct": float(vol*100) if vol==vol else np.nan,
            "sharpe": float(sharpe) if sharpe==sharpe else np.nan,
            "hit_rate": float(hit) if hit==hit else np.nan,
            "max_dd_pct": float(dd*100) if dd==dd else np.nan}

# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    yields: str
    pairs: Optional[str]
    durations: Optional[str]
    anchor: str
    tenors: Optional[str]
    beta_window: int
    lookback: int
    z_entry: float
    z_exit: float
    max_hold: int
    slippage_bp: float
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rate Convergence — beta-hedged spreads & backtest")
    ap.add_argument("--yields", required=True, help="CSV of sovereign yields (long or wide)")
    ap.add_argument("--pairs", default="", help="Optional pairs CSV: tenor,peer,anchor")
    ap.add_argument("--durations", default="", help="Optional durations CSV: country,tenor,duration_years")
    ap.add_argument("--anchor", default="DE", help="Default anchor country code (ISO-2/3-like)")
    ap.add_argument("--tenors", default="", help="Comma list like 2,5,10 (suffix Y inferred)")
    ap.add_argument("--beta_window", type=int, default=252)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--z_entry", type=float, default=2.0)
    ap.add_argument("--z_exit", type=float, default=0.5)
    ap.add_argument("--max_hold", type=int, default=60)
    ap.add_argument("--slippage_bp", type=float, default=0.0, help="Per-leg slippage (bp) applied on entry+exit")
    ap.add_argument("--outdir", default="out_convergence")
    return ap.parse_args()

# Global args handle (needed inside compute loop for thresholds)
args = None  # will be set in main()

def main():
    global args
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load data
    PANEL = load_yields(args.yields)  # columns MultiIndex (tenor, country)
    # Filter dates with at least two valid countries per tenor
    PANEL = PANEL.dropna(how="all").sort_index()

    DUR = load_durations(args.durations) if args.durations else pd.DataFrame()

    # Pairs
    PAIRS_FILE = load_pairs(args.pairs) if args.pairs else pd.DataFrame()
    if not PAIRS_FILE.empty:
        pairs = PAIRS_FILE
    else:
        tenors = [f"{t.strip()}Y" for t in args.tenors.split(",") if t.strip()] if args.tenors else None
        pairs = build_pairs_from_panel(PANEL, anchor=args.anchor.upper(), tenors=tenors)
    if pairs.empty:
        raise ValueError("No valid pairs could be constructed. Check --anchor, --tenors and input data.")
    pairs.to_csv(outdir / "pairs_used.csv", index=False)

    # Compute metrics & trades
    spreads_panel, trades, pnl_daily = compute_pair_metrics(
        panel=PANEL, pairs=pairs, beta_window=int(args.beta_window), lookback=int(args.lookback),
        DUR=DUR, slippage_bp=float(args.slippage_bp)
    )

    # Write spreads panel
    if not spreads_panel.empty:
        spreads_panel.to_csv(outdir / "spreads_panel.csv", index=False)

    # Trades
    if not trades.empty:
        trades = trades.sort_values(["pair","entry_date"])
        trades.to_csv(outdir / "trades.csv", index=False)

    # PnL
    if not pnl_daily.empty:
        pnl_daily.to_csv(outdir / "backtest_pnl.csv", index=False)
        pnl_daily["date"] = pd.to_datetime(pnl_daily["date"])
        pnl_daily = pnl_daily.set_index("date").sort_index()

    # Summary stats
    if pnl_daily.empty:
        stats = {"aggregate": perf_stats_from_daily(pd.Series(dtype=float))}
        per_pair_stats = []
    else:
        stats = {"aggregate": perf_stats_from_daily(pnl_daily["aggregate"])}
        per_pair_stats = []
        for c in pnl_daily.columns:
            if c == "aggregate": continue
            st = perf_stats_from_daily(pnl_daily[c])
            st["pair"] = c
            per_pair_stats.append(st)

    # High-level diagnostics
    if not spreads_panel.empty:
        latest = (spreads_panel.sort_values("date")
                  .groupby("pair").tail(1)[["pair","tenor","peer","anchor","zscore","half_life_days","spread_raw"]]
                  .sort_values("zscore", ascending=False))
        latest_top = latest.head(10).to_dict(orient="records")
        latest_bot = latest.tail(10).to_dict(orient="records")
    else:
        latest_top, latest_bot = [], []

    summary = {
        "pairs": pairs.to_dict(orient="records"),
        "stats_aggregate": stats["aggregate"],
        "stats_per_pair": per_pair_stats,
        "latest_top_z": latest_top,
        "latest_bottom_z": latest_bot
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        yields=args.yields, pairs=(args.pairs or None), durations=(args.durations or None),
        anchor=args.anchor, tenors=(args.tenors or None), beta_window=args.beta_window,
        lookback=args.lookback, z_entry=args.z_entry, z_exit=args.z_exit, max_hold=args.max_hold,
        slippage_bp=args.slippage_bp, outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Rate Convergence ==")
    if not pnl_daily.empty:
        a = stats["aggregate"]
        print(f"Pairs: {pairs.shape[0]} | CAGR {a['cagr_pct']:.2f}% | Vol {a['vol_pct']:.2f}% | Sharpe {a['sharpe']:.2f} | MaxDD {a['max_dd_pct']:.2f}%")
    else:
        print("No P&L produced (insufficient data/pairs). Check inputs.")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
