#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cloud_capex_divergence.py
#
# Analyze **divergence** in cloud providers' capex (level, growth, share).
# Works off your CSVs (no scraping): compute YoY/QoQ growth, market shares,
# dispersion indices (std/iqr/spread), contribution-to-growth, rolling z-scores,
# and simple lead/lag correlations to see who’s pulling ahead.
#
# Inputs
# ------
# --capex FILE (CSV, required) columns:
#   date, provider, capex_usd         # date = YYYY-Qn or YYYY-MM-DD
# Optional columns (if present they’ll be used in extras):
#   revenue_cloud_usd, guidance_capex_usd, notes
#
# Flags
# -----
# --freq {q,m}              : sampling (default q = quarterly)
# --window N                : rolling window for z-scores & stdev (default 4)
# --share-on 'capex_usd'    : field to compute market shares on (default capex)
# --plot                    : write PNG charts
# --outdir PATH             : default ./artifacts
#
# Outputs
# -------
# outdir/
#   capex_clean.csv
#   provider_metrics.csv          (per provider × period)
#   divergence_dashboard.csv      (market shares, dispersion, spreads, HHI, Gini)
#   leaders_laggards.csv          (rankings, z-scores, inflection flags)
#   leadlag_corr.csv              (cross-provider correlations, lead/lag)
#   plots/*.png                   (optional)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib python-dateutil

import argparse
import os
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from dateutil import parser as dtp

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ------------------------- Config & IO -------------------------

@dataclass
class Config:
    capex_file: str
    freq: str
    window: int
    share_on: str
    plot: bool
    outdir: str


def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "cloud_capex_divergence_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def parse_period(x: str, freq: str) -> pd.Timestamp:
    s = str(x).strip()
    if "Q" in s.upper():  # YYYY-Qn
        year, q = s.replace(" ", "").upper().split("-Q")
        q = int(q)
        month = (q - 1) * 3 + 1
        return pd.Timestamp(int(year), month, 1)
    try:
        d = dtp.parse(s)
        if freq == "q":
            return pd.Timestamp(d.year, ((d.month - 1)//3)*3 + 1, 1)
        return pd.Timestamp(d.year, d.month, 1)
    except Exception:
        # fallback: treat as year-month
        try:
            y, m = s.split("-")[:2]
            return pd.Timestamp(int(y), int(m), 1)
        except Exception:
            raise SystemExit(f"Unparsable date '{x}'. Use YYYY-Qn or YYYY-MM or YYYY-MM-DD.")


def read_capex(path: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"date","provider","capex_usd"} <= set(df.columns):
        raise SystemExit("capex CSV must have columns: date, provider, capex_usd")
    df["provider"] = df["provider"].astype(str).str.strip()
    df["period"] = df["date"].apply(lambda x: parse_period(x, freq))
    # numeric coercions
    for c in ["capex_usd","revenue_cloud_usd","guidance_capex_usd"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["capex_usd"] = df["capex_usd"].fillna(0.0)
    # aggregate duplicates (e.g., multiple lines per provider/period)
    df = (df.groupby(["period","provider"], as_index=False)
            .agg({k:("sum" if k.endswith("_usd") else "first") for k in df.columns if k not in ["date","period","provider"]} | {"capex_usd":"sum"}))
    return df.sort_values(["period","provider"]).reset_index(drop=True)


# ------------------------- Metrics -------------------------

def yoy(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)

def qoq(series: pd.Series) -> pd.Series:
    return series.pct_change()

def rolling_z(series: pd.Series, w: int) -> pd.Series:
    m = series.rolling(w, min_periods=max(2, w//2)).mean()
    s = series.rolling(w, min_periods=max(2, w//2)).std()
    return (series - m) / s.replace(0, np.nan)

def gini(arr: np.ndarray) -> float:
    x = np.sort(np.nan_to_num(arr, nan=0.0))
    if x.sum() == 0: return 0.0
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def hhi(shares: np.ndarray) -> float:
    # shares as fractions summing to 1
    s = np.nan_to_num(shares, nan=0.0)
    if s.sum() == 0: return 0.0
    s = s / s.sum()
    return float(np.sum((s * 100) ** 2))  # in HHI points (0..10,000)

def divergence_panel(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      provider_metrics: per provider × period metrics
      div_dash: per period market shares & dispersion indices
      leaders: per period leader/laggard & inflection flags
    """
    df = df.copy()
    # align to regular period grid
    providers = sorted(df["provider"].unique().tolist())
    idx = pd.MultiIndex.from_product([sorted(df["period"].unique()), providers], names=["period","provider"])
    df = df.set_index(["period","provider"]).reindex(idx).reset_index()
    for col in ["capex_usd","revenue_cloud_usd","guidance_capex_usd"]:
        if col in df.columns: df[col] = df[col].fillna(0.0)

    # growth metrics
    perprov = []
    freq_periods = 4 if cfg.freq == "q" else 12
    for p in providers:
        sub = df[df["provider"] == p].set_index("period").sort_index()
        cap = sub["capex_usd"]
        m = pd.DataFrame(index=sub.index)
        m["provider"] = p
        m["capex_usd"] = cap
        m["qoq"] = qoq(cap)
        m["yoy"] = yoy(cap, periods=freq_periods)
        m["roll_z_capex"] = rolling_z(cap, cfg.window)
        if "revenue_cloud_usd" in sub:
            m["rev_usd"] = sub["revenue_cloud_usd"]
            m["capex_to_rev"] = m["capex_usd"] / m["rev_usd"].replace(0, np.nan)
        if "guidance_capex_usd" in sub:
            m["guidance_gap"] = (sub["guidance_capex_usd"] - m["capex_usd"]) / sub["guidance_capex_usd"].replace(0, np.nan)
        perprov.append(m.reset_index())
    provider_metrics = pd.concat(perprov, ignore_index=True)

    # market shares & dispersion
    pivot_val = provider_metrics.pivot(index="period", columns="provider", values=cfg.share_on if cfg.share_on in provider_metrics.columns else "capex_usd").fillna(0.0)
    shares = pivot_val.div(pivot_val.sum(axis=1).replace(0, np.nan), axis=0)
    # dispersion based on growth YoY (who is sprinting)
    yoy_wide = provider_metrics.pivot(index="period", columns="provider", values="yoy")
    qoq_wide = provider_metrics.pivot(index="period", columns="provider", values="qoq")
    z_wide = provider_metrics.pivot(index="period", columns="provider", values="roll_z_capex")

    def _dispersion(wide: pd.DataFrame) -> pd.Series:
        return wide.apply(lambda r: np.nanstd(r.values, ddof=0), axis=1)

    div_dash = pd.DataFrame(index=pivot_val.index)
    div_dash["capex_total_usd"] = pivot_val.sum(axis=1)
    div_dash["hhi_capex"] = shares.apply(lambda r: hhi(r.values), axis=1)
    div_dash["gini_capex"] = pivot_val.apply(lambda r: gini(r.values), axis=1)
    div_dash["yoy_dispersion"] = _dispersion(yoy_wide)
    div_dash["qoq_dispersion"] = _dispersion(qoq_wide)
    div_dash["z_median"] = z_wide.median(axis=1)
    div_dash["z_iqr"] = (z_wide.quantile(0.75, axis=1) - z_wide.quantile(0.25, axis=1))
    # leader/laggard spread
    div_dash["yoy_spread"] = yoy_wide.max(axis=1) - yoy_wide.min(axis=1)
    div_dash["top_share"] = shares.max(axis=1)
    div_dash["top_minus_median_share"] = shares.max(axis=1) - shares.median(axis=1)

    # leaders / laggards / inflection
    leaders = []
    for dt in pivot_val.index:
        row_share = shares.loc[dt]
        row_yoy = yoy_wide.loc[dt]
        if row_share.isna().all(): continue
        leader = row_share.idxmax()
        laggard = row_share.idxmin()
        sprint = row_yoy.idxmax() if not row_yoy.isna().all() else leader
        slump = row_yoy.idxmin() if not row_yoy.isna().all() else laggard
        # simple inflection: sign change of yoy for each provider
        infl = []
        for p in providers:
            series = yoy_wide[p].dropna()
            flag = False
            if dt in series.index:
                i = series.index.get_loc(dt)
                if i >= 1:
                    prev = series.iloc[i-1]
                    cur = series.iloc[i]
                    if np.sign(prev) != np.sign(cur):
                        flag = True
            infl.append((p, flag))
        leaders.append({
            "period": dt, "leader_by_share": leader, "laggard_by_share": laggard,
            "leader_by_yoy": sprint, "laggard_by_yoy": slump,
            "inflections": ";".join([f"{p}:{int(f)}" for p, f in infl])
        })
    leaders = pd.DataFrame(leaders).sort_values("period")

    # combine shares wide into long for dashboard output
    shares_long = shares.reset_index().melt(id_vars="period", var_name="provider", value_name="capex_share")
    market_levels = pivot_val.reset_index().melt(id_vars="period", var_name="provider", value_name="capex_usd")
    dash_long = shares_long.merge(market_levels, on=["period","provider"], how="left")
    div_dash_out = (dash_long.merge(div_dash.reset_index(), on="period", how="left")
                    .sort_values(["period","provider"]))

    return provider_metrics, div_dash_out, leaders


# ------------------------- Lead/Lag Correlation -------------------------

def lead_lag_corr(provider_metrics: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """
    Cross-provider correlations of YoY growth with leads/lags (e.g., does A lead B by 1–2Q?).
    Returns a long table: (p1, p2, lag, corr).
    """
    pm = provider_metrics.pivot(index="period", columns="provider", values="yoy")
    pm = pm.sort_index().dropna(how="all")
    prov = pm.columns.tolist()
    rows = []
    for i, p1 in enumerate(prov):
        for p2 in prov:
            if p1 == p2: continue
            s1 = pm[p1]
            s2 = pm[p2]
            for lag in range(-max_lag, max_lag+1):
                if lag < 0:
                    s1_shift = s1.shift(-lag)
                    s2_shift = s2
                else:
                    s1_shift = s1
                    s2_shift = s2.shift(lag)
                aligned = pd.concat([s1_shift, s2_shift], axis=1).dropna()
                if len(aligned) < 3:
                    corr = np.nan
                else:
                    corr = aligned.corr().iloc[0,1]
                rows.append({"p1": p1, "p2": p2, "lag": lag, "corr": corr})
    return pd.DataFrame(rows)


# ------------------------- Plotting -------------------------

def make_plots(provider_metrics: pd.DataFrame, div_dash: pd.DataFrame, outdir: str):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Total capex & dispersion
    agg = (div_dash[["period","capex_total_usd","hhi_capex","yoy_dispersion"]]
           .drop_duplicates().set_index("period").sort_index())
    fig1 = plt.figure(figsize=(10,6)); ax = plt.gca()
    ax.plot(agg.index, agg["capex_total_usd"]/1e9, label="Total capex (USD bn)")
    ax.set_ylabel("USD bn"); ax.set_title("Cloud capex & divergence")
    ax2 = ax.twinx()
    ax2.plot(agg.index, agg["yoy_dispersion"], linestyle="--", label="YoY dispersion")
    ax2.set_ylabel("YoY dispersion (σ)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout(); fig1.savefig(os.path.join(outdir, "plots", "capex_total_divergence.png"), dpi=140); plt.close(fig1)

    # Provider capex levels
    piv = provider_metrics.pivot(index="period", columns="provider", values="capex_usd").sort_index()
    fig2 = plt.figure(figsize=(10,6)); ax = plt.gca()
    (piv/1e9).plot(ax=ax)
    ax.set_title("Capex by provider"); ax.set_ylabel("USD bn"); plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "plots", "capex_by_provider.png"), dpi=140); plt.close(fig2)

    # YoY growth by provider
    pivg = provider_metrics.pivot(index="period", columns="provider", values="yoy").sort_index()
    fig3 = plt.figure(figsize=(10,6)); ax = plt.gca()
    (100*pivg).plot(ax=ax)
    ax.axhline(0, linestyle="--"); ax.set_ylabel("YoY %"); ax.set_title("YoY growth by provider")
    plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "yoy_by_provider.png"), dpi=140); plt.close(fig3)


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Cloud CAPEX divergence analyzer")
    ap.add_argument("--capex", required=True, help="CSV with provider capex time series")
    ap.add_argument("--freq", choices=["q","m"], default="q", help="q=quarterly, m=monthly")
    ap.add_argument("--window", type=int, default=4, help="Rolling window for z-scores & stdev")
    ap.add_argument("--share-on", default="capex_usd", dest="share_on", help="Field for share calc")
    ap.add_argument("--plot", action="store_true", help="Write PNG plots")
    ap.add_argument("--outdir", default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        capex_file=args.capex,
        freq=args.freq,
        window=int(max(2, args.window)),
        share_on=args.share_on,
        plot=bool(args.plot),
        outdir=args.outdir
    )

    outdir = ensure_outdir(cfg.outdir)
    print(f"[INFO] Writing artifacts to: {outdir}")

    df = read_capex(cfg.capex_file, cfg.freq)
    df.to_csv(os.path.join(outdir, "capex_clean.csv"), index=False)

    provider_metrics, div_dash, leaders = divergence_panel(df, cfg)
    provider_metrics.to_csv(os.path.join(outdir, "provider_metrics.csv"), index=False)
    div_dash.to_csv(os.path.join(outdir, "divergence_dashboard.csv"), index=False)
    leaders.to_csv(os.path.join(outdir, "leaders_laggards.csv"), index=False)

    ll = lead_lag_corr(provider_metrics)
    ll.to_csv(os.path.join(outdir, "leadlag_corr.csv"), index=False)

    if cfg.plot:
        make_plots(provider_metrics, div_dash, outdir)
        print("[OK] Plots saved to:", os.path.join(outdir, "plots"))

    print("\n=== Quick snapshot (last period) ===")
    last = provider_metrics["period"].max()
    snap = provider_metrics[provider_metrics["period"] == last][["provider","capex_usd","yoy","qoq","roll_z_capex"]].copy()
    snap = snap.sort_values("capex_usd", ascending=False)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(snap.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()