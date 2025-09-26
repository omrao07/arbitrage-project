#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# meme_vs_quality.py
#
# Compare “meme” stocks vs “quality” (e.g., QUAL or your tickers)
# ----------------------------------------------------------------
# - Downloads prices with yfinance
# - Builds equal-weight baskets (MEME_EQW, QUALITY_EQW)
# - Computes performance stats, max drawdown, skew/kurt, beta to SPY
# - Rolling correlations & betas (60d)
# - Simple long/shorts:
#     * Q–M (long quality, short meme), periodic rebalance
#     * Optional beta-neutral (scale legs to net beta≈0 to benchmark)
# - Episode scorecards for user-supplied windows
# - Exports tidy CSVs and PNG plots (optional)
#
# Examples
# --------
# python meme_vs_quality.py \
#   --meme GME,AMC,BBBY,BB,TLRY \
#   --quality QUAL,MSFT,ADP,CTAS \
#   --benchmark SPY --start 2020-01-01 --rebalance m --beta-neutral --plot
#
# python meme_vs_quality.py \
#   --meme GME,AMC \
#   --quality QUAL \
#   --episodes "2021-01-01:2021-03-31 Meme_Squeeze","2024-05-10:2024-06-15 RoaringKitty2" \
#   --plot
#
# Dependencies
# ------------
# pip install pandas numpy yfinance matplotlib python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    raise SystemExit("Please install yfinance: pip install yfinance")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from dateutil import parser as dtp


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    meme: List[str]
    quality: List[str]
    benchmark: Optional[str]
    start: str
    end: Optional[str]
    rf: float
    rebalance: str
    beta_neutral: bool
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp, str]]
    plot: bool
    outdir: str


# ----------------------------- Helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "meme_vs_quality_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out

def parse_list(s: Optional[str]) -> List[str]:
    if not s: return []
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def parse_episodes(items: List[str]) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    eps = []
    for it in items:
        parts = it.strip().split()
        if not parts: continue
        d1, d2 = parts[0].split(":")
        label = " ".join(parts[1:]) if len(parts) > 1 else f"Ep_{len(eps)+1}"
        eps.append((pd.Timestamp(dtp.parse(d1).date()), pd.Timestamp(dtp.parse(d2).date()), label))
    return eps

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().replace([np.inf, -np.inf], np.nan)

def max_drawdown(series: pd.Series) -> float:
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1.0
    return float(dd.min())

def ann_stats(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan,
                "skew": np.nan, "kurt": np.nan}
    mean = r.mean() * periods_per_year
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (mean - rf) / (vol if vol and not np.isnan(vol) and vol != 0 else np.nan)
    return {"ann_return": float(mean), "ann_vol": float(vol),
            "sharpe": float(sharpe), "max_drawdown": float(max_drawdown(r)),
            "skew": float(r.skew()), "kurt": float(r.kurt())}

def rolling_beta(y: pd.Series, x: pd.Series, win: int = 60) -> pd.Series:
    joined = pd.concat([y, x], axis=1).dropna()
    if joined.empty: return pd.Series(index=y.index, dtype=float)
    cov = joined[y.name].rolling(win).cov(joined[x.name])
    var = joined[x.name].rolling(win).var()
    beta = cov / var.replace(0, np.nan)
    beta.name = f"beta_{y.name}_to_{x.name}"
    return beta.reindex(y.index)

def backtest_rebalanced(returns: pd.DataFrame, weights: Dict[str, float], freq: str = "m") -> pd.Series:
    """Periodic rebalanced portfolio on simple daily returns."""
    rets = returns.dropna(how="all")
    if rets.empty: return pd.Series(dtype=float)
    periods = rets.resample({"m":"M", "q":"Q", "w":"W-FRI"}.get(freq.lower(), "M")).last().index
    w = pd.Series(weights, dtype=float); w = w / w.sum()
    wealth = pd.Series(index=rets.index, dtype=float); wealth.iloc[0] = 1.0
    cur_w = w.copy()
    for i in range(1, len(rets)):
        d = rets.index[i]
        if d in periods: cur_w = w.copy()
        r = (rets.iloc[i] * cur_w).sum(skipna=True)
        wealth.iloc[i] = wealth.iloc[i-1] * (1 + (0 if np.isnan(r) else r))
        # drift (not strictly needed since we rebalance at period boundaries)
        vals = (1 + rets.iloc[i]).fillna(1.0) * cur_w
        total = vals.sum()
        cur_w = vals / (total if total != 0 else 1.0)
    return wealth.pct_change().dropna()

def beta_neutral_weights(meme_ret: pd.Series, qual_ret: pd.Series, bench: pd.Series, base_w_long: float = 1.0) -> Tuple[float, float]:
    """Compute static weights (long quality, short meme) such that net beta to bench ≈ 0 (using full-sample betas)."""
    beta_q = rolling_beta(qual_ret, bench, win=min(60, max(20, int(len(bench)*0.25)))).dropna().mean()
    beta_m = rolling_beta(meme_ret, bench, win=min(60, max(20, int(len(bench)*0.25)))).dropna().mean()
    if pd.isna(beta_q) or pd.isna(beta_m) or beta_m == 0:
        return base_w_long, -base_w_long  # fallback equal $ weights
    # Solve w_q*beta_q + w_m*beta_m = 0 with w_q = base
    w_q = base_w_long
    w_m = -w_q * (beta_q / beta_m)
    return float(w_q), float(w_m)


# ----------------------------- Main analysis -----------------------------

def run(cfg: Config):
    tickers = sorted(set(cfg.meme + cfg.quality + ([cfg.benchmark] if cfg.benchmark else [])))
    data = yf.download(
        tickers=tickers, start=cfg.start, end=cfg.end,
        interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    closes = {}
    vols = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                if "Close" in data[t].columns:
                    closes[t] = data[t]["Close"]
                    vols[t] = data[t].get("Volume")
    else:
        t = tickers[0]
        closes[t] = data["Close"]
        vols[t] = data.get("Volume")
    px = pd.DataFrame(closes).dropna(how="all").sort_index()
    if px.empty:
        raise SystemExit("No price data downloaded. Check tickers / dates.")

    os.makedirs(cfg.outdir, exist_ok=True)
    px.to_csv(os.path.join(cfg.outdir, "prices_close.csv"))

    rets = pct_change(px).dropna(how="all")
    rets.to_csv(os.path.join(cfg.outdir, "returns_daily.csv"))

    meme = [t for t in cfg.meme if t in px.columns]
    qual = [t for t in cfg.quality if t in px.columns]
    bench = cfg.benchmark if (cfg.benchmark and cfg.benchmark in px.columns) else None

    if not meme or not qual:
        raise SystemExit("Need at least one valid ticker in both --meme and --quality after download.")

    # Equal-weight baskets
    baskets = {}
    baskets["MEME_EQW"] = rets[meme].mean(axis=1, skipna=True)
    if len(qual) == 1 and qual[0] in rets.columns:
        baskets["QUALITY_EQW"] = rets[qual[0]]
    else:
        baskets["QUALITY_EQW"] = rets[qual].mean(axis=1, skipna=True)
    if bench:
        baskets[bench] = rets[bench]
    bask = pd.DataFrame(baskets).dropna(how="all")
    bask.to_csv(os.path.join(cfg.outdir, "basket_returns_eqw.csv"))

    # Stats for baskets & components
    rows = []
    for name, ser in bask.items():
        s = ann_stats(ser, rf=cfg.rf)
        s["series"] = name
        rows.append(s)
    # component roll-up (optional snapshot)
    for t in meme + qual:
        if t in rets.columns:
            s = ann_stats(rets[t], rf=cfg.rf); s["series"] = t; rows.append(s)
    stats = pd.DataFrame(rows)[["series","ann_return","ann_vol","sharpe","max_drawdown","skew","kurt"]]
    stats.to_csv(os.path.join(cfg.outdir, "stats.csv"), index=False)

    # Rolling 60d correlation & beta to benchmark
    roll = {}
    if bench:
        for col in ["MEME_EQW","QUALITY_EQW"]:
            if col in bask.columns:
                roll[f"corr_{col}_to_{bench}"] = bask[col].rolling(60).corr(bask[bench])
                roll[f"beta_{col}_to_{bench}"] = rolling_beta(bask[col], bask[bench], 60)
        roll_df = pd.DataFrame(roll)
        roll_df.to_csv(os.path.join(cfg.outdir, "rolling_corr_beta_60d.csv"), index=False)

    # Long/short portfolios
    ls_returns = {}
    pair = bask[["QUALITY_EQW","MEME_EQW"]].dropna()
    if bench and cfg.beta_neutral:
        wq, wm = beta_neutral_weights(pair["MEME_EQW"], pair["QUALITY_EQW"], bask[bench])
    else:
        wq, wm = 1.0, -1.0
    ls_name = f"LS_QminusM_{'beta0' if cfg.beta_neutral else 'dollar'}"
    ls_series = backtest_rebalanced(pair, {"QUALITY_EQW": wq, "MEME_EQW": wm}, cfg.rebalance)
    ls_returns[ls_name] = ls_series
    # Save LS
    if ls_returns:
        ls_df = pd.DataFrame(ls_returns)
        ls_df.to_csv(os.path.join(cfg.outdir, "long_short_returns.csv"))
        # Stats
        ls_stats = []
        for c in ls_df.columns:
            s = ann_stats(ls_df[c], rf=cfg.rf); s["series"] = c; ls_stats.append(s)
        pd.DataFrame(ls_stats)[["series","ann_return","ann_vol","sharpe","max_drawdown","skew","kurt"]].to_csv(
            os.path.join(cfg.outdir, "long_short_stats.csv"), index=False
        )

    # Episodes
    if cfg.episodes:
        ep_rows = []
        for a,b,label in cfg.episodes:
            sub = bask.loc[(bask.index >= a) & (bask.index <= b)]
            if sub.empty: continue
            for col in ["MEME_EQW","QUALITY_EQW"]:
                if col not in sub.columns: continue
                ep_rows.append({
                    "episode": label, "start": str(a.date()), "end": str(b.date()),
                    "series": col,
                    "total_return": float((1 + sub[col]).prod() - 1),
                    "ann_return": float(sub[col].mean()*252),
                    "vol": float(sub[col].std(ddof=1)*np.sqrt(252)),
                    "max_drawdown": float(max_drawdown(sub[col]))
                })
            if ls_returns:
                ls_sub = ls_df.loc[(ls_df.index >= a) & (ls_df.index <= b)]
                if not ls_sub.empty:
                    col = ls_df.columns[0]
                    ep_rows.append({
                        "episode": label, "start": str(a.date()), "end": str(b.date()),
                        "series": col,
                        "total_return": float((1 + ls_sub[col]).prod() - 1),
                        "ann_return": float(ls_sub[col].mean()*252),
                        "vol": float(ls_sub[col].std(ddof=1)*np.sqrt(252)),
                        "max_drawdown": float(max_drawdown(ls_sub[col]))
                    })
        if ep_rows:
            pd.DataFrame(ep_rows).to_csv(os.path.join(cfg.outdir, "episode_scorecards.csv"), index=False)

    # ----------------------------- Plots -----------------------------
    if cfg.plot and plt is not None:
        # Normalized levels for baskets
        fig1 = plt.figure(figsize=(10,5)); ax1 = plt.gca()
        norm = (1 + bask[["MEME_EQW","QUALITY_EQW"]].dropna()).cumprod()
        norm.plot(ax=ax1)
        ax1.set_title("Baskets: MEME vs QUALITY (EW) — Wealth (start=1)")
        ax1.set_ylabel("Index")
        plt.tight_layout(); fig1.savefig(os.path.join(cfg.outdir, "plots", "baskets_wealth.png"), dpi=140); plt.close(fig1)

        # Rolling corr/beta to benchmark
        if bench and 'rolling_corr_beta_60d.csv':
            roll_df = pd.read_csv(os.path.join(cfg.outdir, "rolling_corr_beta_60d.csv"))
            roll_idx = bask.index[-len(roll_df):] if len(roll_df) <= len(bask.index) else bask.index
            roll_df.index = roll_idx
            # corr
            corr_cols = [c for c in roll_df.columns if c.startswith("corr_")]
            if corr_cols:
                fig2 = plt.figure(figsize=(10,5)); ax2 = plt.gca()
                roll_df[corr_cols].plot(ax=ax2)
                ax2.axhline(0, linestyle="--")
                ax2.set_title(f"Rolling 60D correlation to {bench}")
                plt.tight_layout(); fig2.savefig(os.path.join(cfg.outdir, "plots", "rolling_corr.png"), dpi=140); plt.close(fig2)
            # beta
            beta_cols = [c for c in roll_df.columns if c.startswith("beta_")]
            if beta_cols:
                fig3 = plt.figure(figsize=(10,5)); ax3 = plt.gca()
                roll_df[beta_cols].plot(ax=ax3)
                ax3.set_title(f"Rolling 60D beta to {bench}")
                plt.tight_layout(); fig3.savefig(os.path.join(cfg.outdir, "plots", "rolling_beta.png"), dpi=140); plt.close(fig3)

        # LS equity curve
        if ls_returns:
            fig4 = plt.figure(figsize=(9,5)); ax4 = plt.gca()
            ((1 + ls_series).cumprod()).plot(ax=ax4)
            ax4.set_title(f"Long/Short: Quality − Meme ({'beta-neutral' if cfg.beta_neutral else 'dollar-neutral'})")
            ax4.set_ylabel("Wealth (start=1)")
            plt.tight_layout(); fig4.savefig(os.path.join(cfg.outdir, "plots", "long_short_equity.png"), dpi=140); plt.close(fig4)

    # Console snapshot
    print("\n=== Basket stats ===")
    print(stats[stats["series"].isin(["MEME_EQW","QUALITY_EQW"])].round(4).to_string(index=False))
    if ls_returns:
        print("\n=== Long/Short stats ===")
        print(pd.read_csv(os.path.join(cfg.outdir, "long_short_stats.csv")).round(4).to_string(index=False))
    print("\nDone. Files written to:", cfg.outdir)


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Meme vs Quality: baskets, risk, and long/short backtests")
    ap.add_argument("--meme", type=str, required=True, help="Comma-separated list of meme tickers (e.g., GME,AMC,BBBY)")
    ap.add_argument("--quality", type=str, required=True, help="Comma-separated list or ETF (e.g., QUAL,MSFT,ADP)")
    ap.add_argument("--benchmark", type=str, default="SPY", help="Benchmark for beta/corr (optional)")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--rf", type=float, default=0.0, help="Annualized risk-free for Sharpe")
    ap.add_argument("--rebalance", choices=["m","q","w"], default="m", help="Rebalance frequency for portfolios")
    ap.add_argument("--beta-neutral", action="store_true", help="Scale long/short to net beta≈0 to benchmark")
    ap.add_argument("--episodes", type=str, nargs="*", default=[], help="Windows like 'YYYY-MM-DD:YYYY-MM-DD Label'")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        meme=parse_list(args.meme),
        quality=parse_list(args.quality),
        benchmark=(args.benchmark or None),
        start=args.start,
        end=args.end,
        rf=float(args.rf),
        rebalance=args.rebalance,
        beta_neutral=bool(args.beta_neutral),
        episodes=parse_episodes(args.episodes) if args.episodes else [],
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Universe: meme={cfg.meme} | quality={cfg.quality} | bench={[cfg.benchmark] if cfg.benchmark else []}")
    run(cfg)


if __name__ == "__main__":
    main()