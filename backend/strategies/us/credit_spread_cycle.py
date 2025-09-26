#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# credit_spread_cycle.py
#
# End-to-end toolkit to analyze **credit spread cycles**:
# - Cleans and aligns spread indices (IG OAS, HY OAS, EM, CDS, etc.)
# - Computes z-scores, percentiles, drawdowns, and cycle phases
# - Detects turning points (peaks/troughs) and regime states
# - Estimates mean-reversion / widening risk via rolling stats
# - Optional overlays: policy rates, recession flags
# - Exports tidy CSVs and optional PNG plots
#
# Inputs
# ------
# --spreads FILE (CSV, required)
#   Columns (wide or long accepted):
#     If wide: date, <series1>, <series2>, ...
#     If long: date, series, value
#   Example series names: IG_OAS_bps, HY_OAS_bps, EM_HY_bps, CDX_IG_bps
#
# Optional:
# --macro FILE (CSV) with columns: date, series, value
#   (e.g., 'FFR', 'PolicyRate', 'Recession', 'CPI', 'VIX', etc.)
#   Recession should be 0/1 if present.
#
# Flags
# -----
# --freq {d,w,m}        : resample frequency (default w)
# --window N            : rolling window (default 52) for z/vol
# --hp-lambda FLOAT     : HP filter lambda (default 129600 for monthly; scaled to freq)
# --plot                : write PNG charts
# --outdir PATH         : default ./artifacts
#
# Outputs
# -------
# outdir/
#   spreads_clean.csv
#   cycles_metrics.csv          (z, pctile, vol, drawdown, carry, slope)
#   regimes.csv                 (state machine by series/date)
#   turning_points.csv          (peak/trough detection)
#   signals.csv                 (mean-reversion & breakout flags)
#   correlations.csv            (rolling corr across series)
#   plots/*.png
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib scipy statsmodels python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from statsmodels.tsa.filters.hp_filter import hpfilter
    HAVE_HP = True
except Exception:
    HAVE_HP = False

from dateutil.relativedelta import relativedelta


# ---------------------------- Config ----------------------------

@dataclass
class Config:
    spreads_file: str
    macro_file: Optional[str]
    freq: str
    window: int
    hp_lambda: float
    plot: bool
    outdir: str


# ---------------------------- IO helpers ----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "credit_spread_cycle_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def to_period_freq(freq: str) -> str:
    return {"d": "D", "w": "W-FRI", "m": "M"}.get(freq.lower(), "W-FRI")


def read_spreads(path: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise SystemExit("spreads CSV must include 'date'")
    # Wide vs long
    if "series" in df.columns and "value" in df.columns:
        # long -> wide
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot(index="date", columns="series", values="value")
    else:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        # coerce numeric
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Resample (mean) and forward-fill
    df = df.resample(to_period_freq(freq)).mean().ffill()
    # Canonicalize column names
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def read_macro(path: Optional[str], freq: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    x = pd.read_csv(path)
    if "date" not in x.columns or "series" not in x.columns or "value" not in x.columns:
        raise SystemExit("macro CSV must have columns: date, series, value")
    x["date"] = pd.to_datetime(x["date"])
    x = x.pivot(index="date", columns="series", values="value").resample(to_period_freq(freq)).mean().ffill()
    return x


# ---------------------------- Metrics ----------------------------

def rolling_stats(x: pd.Series, window: int) -> pd.DataFrame:
    df = pd.DataFrame(index=x.index)
    df["val"] = x
    df["ret_1"] = x.pct_change()             # fractional change (for bps series this may be small)
    df["diff_1"] = x.diff()                  # absolute bps change
    df["vol"] = df["diff_1"].rolling(window, min_periods=max(4, window//4)).std()
    df["mean"] = x.rolling(window, min_periods=max(4, window//4)).mean()
    df["std"]  = x.rolling(window, min_periods=max(4, window//4)).std()
    df["z"] = (x - df["mean"]) / df["std"].replace(0, np.nan)
    df["pctile"] = x.rank(pct=True)
    df["max_roll"] = x.rolling(window, min_periods=1).max()
    df["drawdown"] = (x - df["max_roll"])
    # slope via rolling OLS: regression on time index
    try:
        t = np.arange(len(x))
        df["slope"] = (
            (pd.Series(t, index=x.index) * x).rolling(window).mean()
            - pd.Series(t, index=x.index).rolling(window).mean() * x.rolling(window).mean()
        ) / (
            pd.Series(t, index=x.index).rolling(window).var()
        )
    except Exception:
        df["slope"] = np.nan
    return df


def hp_trend_cycle(x: pd.Series, hp_lambda: float) -> Tuple[pd.Series, pd.Series]:
    if not HAVE_HP:
        return x.rolling(52, min_periods=20).mean(), x - x.rolling(52, min_periods=20).mean()
    cycle, trend = hpfilter(x.dropna(), lamb=hp_lambda)
    # Reindex
    trend = trend.reindex(x.index).ffill()
    cycle = cycle.reindex(x.index).fillna(0.0)
    return trend, cycle


def detect_turns(series: pd.Series, look: int = 3) -> pd.DataFrame:
    """
    Simple local-extrema detector: marks peaks/troughs when a point is
    greater/less than 'look' neighbors on both sides.
    """
    s = series.dropna()
    flags = []
    for i in range(look, len(s) - look):
        win = s.iloc[i - look:i + look + 1]
        mid = s.iloc[i]
        if mid == win.max() and (win.idxmax() == s.index[i]):
            flags.append((s.index[i], "peak", float(mid)))
        if mid == win.min() and (win.idxmin() == s.index[i]):
            flags.append((s.index[i], "trough", float(mid)))
    return pd.DataFrame(flags, columns=["date", "type", "value"])


def regime_state(z: float, slope: float) -> str:
    """
    Heuristic state machine using z-score and slope (widening vs tightening):
    - 'Stress'         : z >= +1.5
    - 'Late-widening'  : 0.5 <= z < 1.5 and slope > 0
    - 'Early-widening' : -0.5 <= z < 0.5 and slope > 0
    - 'Normalizing'    : z > -0.5 and slope < 0
    - 'Euphoria'       : z <= -1.5
    """
    if np.isnan(z) or np.isnan(slope): return "NA"
    if z >= 1.5: return "Stress"
    if z <= -1.5: return "Euphoria"
    if slope > 0 and z >= 0.5: return "Late-widening"
    if slope > 0 and z >= -0.5: return "Early-widening"
    if slope < 0 and z > -0.5: return "Normalizing"
    return "Neutral"


def signals_from_metrics(m: pd.DataFrame, series_name: str) -> pd.DataFrame:
    df = m.copy()
    df["series"] = series_name
    # Mean-reversion: sell-widening / buy-tightening heuristics
    df["sig_widen_risk"] = ((df["z"] > 1.0) & (df["slope"] > 0)).astype(int)
    df["sig_snap_tighten"] = ((df["z"] > 1.0) & (df["slope"] < 0)).astype(int)
    df["sig_buy_beta"] = ((df["z"] < -1.0) & (df["slope"] < 0)).astype(int)
    df["sig_reduce_beta"] = ((df["z"] < -1.0) & (df["slope"] > 0)).astype(int)
    return df[["series","sig_widen_risk","sig_snap_tighten","sig_buy_beta","sig_reduce_beta"]]


def rolling_correlations(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    cols = wide.columns.tolist()
    rows = []
    for i in range(len(wide)):
        if i < window: continue
        sub = wide.iloc[i - window:i].dropna(how="all", axis=1)
        if sub.shape[1] < 2: continue
        c = sub.corr()
        for a in c.columns:
            for b in c.index:
                if a >= b:  # upper triangle once
                    continue
                rows.append({"date": wide.index[i], "a": a, "b": b, "corr": float(c.loc[b, a])})
    return pd.DataFrame(rows)


# ---------------------------- Pipeline ----------------------------

def build_pipeline(spreads: pd.DataFrame, macro: Optional[pd.DataFrame], cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      spreads_clean (wide),
      cycles_metrics (long),
      regimes (long),
      turning_points (long),
      signals (long)
    """
    spreads = spreads.copy().sort_index()
    # HP lambda scaling by freq
    if cfg.freq.lower() == "m":
        hp_lam = cfg.hp_lambda  # default monthly ~129,600
    elif cfg.freq.lower() == "w":
        # roughly 4.3 weeks per month => scale lambda by (per-month frequency)^2
        hp_lam = cfg.hp_lambda * (4.345**2)
    else:
        # daily: ~21 per month
        hp_lam = cfg.hp_lambda * (21**2)

    all_metrics = []
    all_turns = []
    all_states = []
    all_sigs = []

    for col in spreads.columns:
        x = spreads[col].astype(float)
        # Trend & cycle (deviation)
        trend, cyc = hp_trend_cycle(x, hp_lam)
        m = rolling_stats(x, cfg.window)
        m["trend"] = trend
        m["cycle"] = cyc
        m["series"] = col
        # Regime per date
        st = m[["z","slope"]].apply(lambda r: regime_state(r["z"], r["slope"]), axis=1)
        states = pd.DataFrame({"date": m.index, "series": col, "state": st})
        # Turning points on the raw series and trend-adjusted cycle
        tp_raw = detect_turns(x, look=max(2, cfg.window//8))
        tp_raw["series"] = col; tp_raw["kind"] = "raw"
        tp_cyc = detect_turns(cyc, look=max(2, cfg.window//8))
        tp_cyc["series"] = col; tp_cyc["kind"] = "cycle"
        # Signals
        sigs = signals_from_metrics(m, col)
        sigs["date"] = m.index

        all_metrics.append(m.reset_index().rename(columns={"index": "date"}))
        all_states.append(states)
        all_turns.append(pd.concat([tp_raw, tp_cyc], ignore_index=True))
        all_sigs.append(sigs)

    cycles_metrics = pd.concat(all_metrics, ignore_index=True)
    regimes = pd.concat(all_states, ignore_index=True)
    turning_points = pd.concat(all_turns, ignore_index=True).sort_values(["series","date","kind"])
    signals = pd.concat(all_sigs, ignore_index=True)

    # Merge macro overlays if present
    if macro is not None:
        cycles_metrics = cycles_metrics.merge(macro.reset_index().rename(columns={"index": "date"}), on="date", how="left")
        regimes = regimes.merge(macro.reset_index().rename(columns={"index": "date"}), on="date", how="left")

    return spreads, cycles_metrics, regimes, turning_points, signals


# ---------------------------- Plotting ----------------------------

def make_plots(spreads: pd.DataFrame, cycles: pd.DataFrame, regimes: pd.DataFrame, outdir: str):
    if plt is None:
        return

    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    # 1) Levels with trend
    for s in spreads.columns:
        d = cycles[cycles["series"] == s].set_index("date").sort_index()
        if d.empty: continue
        fig = plt.figure(figsize=(10, 5)); ax = plt.gca()
        ax.plot(d.index, d["val"], label=f"{s} (bps)")
        ax.plot(d.index, d["trend"], linestyle="--", label="HP trend")
        ax.set_title(f"{s}: level & trend"); ax.set_ylabel("bps"); ax.legend()
        plt.tight_layout(); fig.savefig(os.path.join(outdir, "plots", f"{s}_level_trend.png"), dpi=140); plt.close(fig)

        # 2) z-score & state bands
        fig2 = plt.figure(figsize=(10, 5)); ax2 = plt.gca()
        ax2.plot(d.index, d["z"], label="z-score")
        ax2.axhline(0, linestyle="--"); ax2.axhline(1.5, linestyle=":"); ax2.axhline(-1.5, linestyle=":")
        ax2.set_title(f"{s}: z-score"); ax2.set_ylabel("z")
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", f"{s}_zscore.png"), dpi=140); plt.close(fig2)

    # 3) Regime heatmap-like strip
    try:
        ser_list = spreads.columns.tolist()
        # Map states to codes
        state_rank = {"Stress":3, "Late-widening":2, "Early-widening":1, "Normalizing":0, "Euphoria":-1, "Neutral":0, "NA":np.nan}
        heat = regimes.pivot(index="date", columns="series", values="state").replace(state_rank).sort_index()
        fig3 = plt.figure(figsize=(11, max(3, len(ser_list)*0.5))); ax3 = plt.gca()
        im = ax3.imshow(heat.T.values, aspect="auto", interpolation="nearest")
        ax3.set_yticks(range(len(ser_list))); ax3.set_yticklabels(ser_list)
        ax3.set_xticks(range(0, heat.shape[0], max(1, heat.shape[0]//10)))
        ax3.set_xticklabels([str(heat.index[i].date()) for i in range(0, heat.shape[0], max(1, heat.shape[0]//10))], rotation=45, ha="right")
        ax3.set_title("Regime state map")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "regimes_map.png"), dpi=150); plt.close(fig3)
    except Exception:
        pass


# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Credit spread cycle analyzer")
    ap.add_argument("--spreads", required=True, help="CSV of spreads (wide or long)")
    ap.add_argument("--macro", default=None, help="Optional macro CSV (date, series, value)")
    ap.add_argument("--freq", choices=["d","w","m"], default="w", help="Resample frequency")
    ap.add_argument("--window", type=int, default=52, help="Rolling window for z/vol/slope")
    ap.add_argument("--hp-lambda", type=float, default=129600.0, dest="hp_lambda",
                    help="HP filter lambda (monthly baseline; scaled for freq)")
    ap.add_argument("--plot", action="store_true", help="Write PNG plots")
    ap.add_argument("--outdir", default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        spreads_file=args.spreads,
        macro_file=args.macro,
        freq=args.freq,
        window=int(max(12, args.window)),
        hp_lambda=float(max(1000.0, args.hp_lambda)),
        plot=bool(args.plot),
        outdir=args.outdir
    )

    outdir = ensure_outdir(cfg.outdir)
    print(f"[INFO] Writing artifacts to: {outdir}")

    spreads = read_spreads(cfg.spreads_file, cfg.freq)
    macro = read_macro(cfg.macro_file, cfg.freq) if cfg.macro_file else None
    spreads.to_csv(os.path.join(outdir, "spreads_clean.csv"))

    spreads_wide, cycles_metrics, regimes, turning_points, signals = build_pipeline(spreads, macro, cfg)
    cycles_metrics.to_csv(os.path.join(outdir, "cycles_metrics.csv"), index=False)
    regimes.to_csv(os.path.join(outdir, "regimes.csv"), index=False)
    turning_points.to_csv(os.path.join(outdir, "turning_points.csv"), index=False)
    signals.to_csv(os.path.join(outdir, "signals.csv"), index=False)

    # Rolling cross-correlation
    corr = rolling_correlations(spreads_wide, cfg.window)
    corr.to_csv(os.path.join(outdir, "correlations.csv"), index=False)

    if cfg.plot:
        make_plots(spreads_wide, cycles_metrics, regimes, outdir)
        print("[OK] Plots saved to:", os.path.join(outdir, "plots"))

    # Snapshot
    last = cycles_metrics["date"].max()
    snap = (cycles_metrics[cycles_metrics["date"] == last]
            .loc[:, ["series","val","z","slope","pctile","drawdown"]]
            .sort_values("z", ascending=False))
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print("\n=== Latest snapshot ===")
        print(snap.round(2).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()