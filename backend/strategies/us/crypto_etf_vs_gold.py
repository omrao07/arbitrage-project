#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# crypto_etf_vs_gold.py
#
# Compare **Crypto ETFs** vs **Gold** (GLD/IAU) across:
# - Total/annualized returns, volatility, Sharpe (rf≈0 by default), max drawdown
# - Rolling correlations (to gold and to equities), rolling beta to gold
# - Crisis/episode scorecards (user-provided windows)
# - Simple 60/40 blend backtests (e.g., 70% gold / 30% crypto ETF, monthly rebalance)
# - Exports tidy CSVs and optional PNG plots
#
# Examples
# --------
# python crypto_etf_vs_gold.py --crypto IBIT,FBTC,BITO --gold GLD,IAU --equity SPY \
#   --start 2021-01-01 --rebalance m --plot
#
# python crypto_etf_vs_gold.py --episodes "2022-11-01:2023-01-31 FTX","2024-10-01:2025-01-31 Rate_Rally"
#
# Dependencies
# ------------
# pip install pandas numpy yfinance matplotlib python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    crypto: List[str]
    gold: List[str]
    equity: Optional[str]
    start: str
    end: Optional[str]
    rf: float
    rebalance: str
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp, str]]
    plot: bool
    outdir: str


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "crypto_etf_vs_gold_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def parse_list_arg(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_episodes(items: List[str]) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    eps = []
    for it in items:
        # format: "YYYY-MM-DD:YYYY-MM-DD Label With_Underscores_orSpaces"
        # or "YYYY-MM-DD:YYYY-MM-DD Label"
        try:
            parts = it.strip().split()
            dates = parts[0]
            label = " ".join(parts[1:]) if len(parts) > 1 else f"Episode_{len(eps)+1}"
            a, b = dates.split(":")
            eps.append((pd.Timestamp(dtp.parse(a).date()), pd.Timestamp(dtp.parse(b).date()), label))
        except Exception:
            raise SystemExit(f"Unparsable episode '{it}'. Use 'YYYY-MM-DD:YYYY-MM-DD Label'.")
    return eps


# ----------------------------- Finance utils -----------------------------

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().replace([np.inf, -np.inf], np.nan)

def max_drawdown(series: pd.Series) -> float:
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())

def ann_stats(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    mean = r.mean() * periods_per_year
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (mean - rf) / (vol if vol and not np.isnan(vol) and vol != 0 else np.nan)
    return {"ann_return": float(mean), "ann_vol": float(vol), "sharpe": float(sharpe)}

def rolling_beta(y: pd.Series, x: pd.Series, win: int = 60) -> pd.Series:
    joined = pd.concat([y, x], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    cov = joined[y.name].rolling(win).cov(joined[x.name])
    var = joined[x.name].rolling(win).var()
    beta = cov / var.replace(0, np.nan)
    beta.name = f"beta_{y.name}_to_{x.name}"
    return beta.reindex(y.index)

def rebal_freq(freq: str) -> str:
    # 'm' monthly, 'q' quarterly, 'w' weekly
    m = {"m": "M", "q": "Q", "w": "W-FRI"}
    return m.get(freq.lower(), "M")

def backtest_blend(prices: pd.DataFrame, weights: dict, freq: str = "m") -> pd.Series:
    """Backtest periodic rebalanced blend; prices are Close levels."""
    rets = prices.pct_change().dropna(how="all")
    periods = rets.resample(rebal_freq(freq)).last().index
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    # initialize
    idx = rets.index
    wealth = pd.Series(index=idx, dtype=float)
    wealth.iloc[0] = 1.0
    cur_w = w.copy()

    last_rebal = periods[0] if len(periods) else idx[0]
    for i in range(1, len(idx)):
        d = idx[i]
        # periodic rebalance on boundaries
        if d in periods:
            cur_w = w.copy()
        r = (rets.iloc[i] * cur_w).sum(skipna=True)
        wealth.iloc[i] = wealth.iloc[i - 1] * (1.0 + (0 if np.isnan(r) else r))
        # drift weights
        port_val = 1.0
        asset_vals = (1.0 + rets.iloc[i]).fillna(1.0) * cur_w
        cur_w = (asset_vals / asset_vals.sum()).fillna(0.0)
    return wealth.pct_change().dropna()


# ----------------------------- Main analysis -----------------------------

def run(cfg: Config):
    universe = sorted(set(cfg.crypto + cfg.gold + ([cfg.equity] if cfg.equity else [])))
    data = yf.download(
        tickers=universe, start=cfg.start, end=cfg.end,
        interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True
    )
    # Flatten OHLCV to Close for each symbol
    closes = {}
    vols = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in universe:
            if t in data.columns.levels[0]:
                if ("Close" in data[t].columns) and ("Volume" in data[t].columns):
                    closes[t] = data[t]["Close"]
                    vols[t] = data[t]["Volume"]
    else:
        # single ticker case
        t = universe[0]
        closes[t] = data["Close"]
        vols[t] = data["Volume"]
    px = pd.DataFrame(closes).dropna(how="all").sort_index()
    if px.empty:
        raise SystemExit("No price data downloaded. Check tickers and date range.")

    px.to_csv(os.path.join(cfg.outdir, "prices_close.csv"))

    # Daily returns
    rets = pct_change(px).dropna(how="all")
    rets.to_csv(os.path.join(cfg.outdir, "returns_daily.csv"))

    # Buckets
    crypto = [t for t in cfg.crypto if t in px.columns]
    gold = [t for t in cfg.gold if t in px.columns]
    eq = cfg.equity if (cfg.equity and cfg.equity in px.columns) else None

    # Stats per asset
    stats_rows = []
    for t in px.columns:
        s = ann_stats(rets[t], periods_per_year=252, rf=cfg.rf)
        s["ticker"] = t
        s["max_drawdown"] = max_drawdown(rets[t])
        s["start"] = str(rets.index.min().date()) if len(rets) else ""
        s["end"] = str(rets.index.max().date()) if len(rets) else ""
        stats_rows.append(s)
    stats = pd.DataFrame(stats_rows)[["ticker","ann_return","ann_vol","sharpe","max_drawdown","start","end"]]
    stats.to_csv(os.path.join(cfg.outdir, "asset_stats.csv"), index=False)

    # Rolling correlations to gold proxy (use first gold, else NaN)
    gold_ref = gold[0] if gold else None
    roll = {}
    if gold_ref:
        for t in px.columns:
            roll[f"corr_{t}_to_{gold_ref}"] = rets[t].rolling(60).corr(rets[gold_ref])
            roll[f"beta_{t}_to_{gold_ref}"] = rolling_beta(rets[t], rets[gold_ref], 60)
    if eq:
        for t in px.columns:
            roll[f"corr_{t}_to_{eq}"] = rets[t].rolling(60).corr(rets[eq])
    if roll:
        roll_df = pd.DataFrame(roll).dropna(how="all")
        roll_df.to_csv(os.path.join(cfg.outdir, "rolling_corr_beta_60d.csv"))

    # Category aggregates (equal-weight baskets)
    agg = {}
    if crypto:
        agg["CRYPTO_EQW"] = rets[crypto].mean(axis=1, skipna=True)
    if gold:
        agg["GOLD_EQW"] = rets[gold].mean(axis=1, skipna=True)
    if agg:
        agg_df = pd.DataFrame(agg)
        agg_df.to_csv(os.path.join(cfg.outdir, "basket_returns_eqw.csv"))
        # basket stats
        bask_stats = []
        for col in agg_df.columns:
            s = ann_stats(agg_df[col], rf=cfg.rf)
            s["ticker"] = col
            s["max_drawdown"] = max_drawdown(agg_df[col])
            bask_stats.append(s)
        pd.DataFrame(bask_stats)[["ticker","ann_return","ann_vol","sharpe","max_drawdown"]].to_csv(
            os.path.join(cfg.outdir, "basket_stats.csv"), index=False
        )

    # Simple blends: e.g., 70% gold basket + 30% best crypto ETF (first)
    blends = {}
    if gold and crypto:
        gold_b = "GOLD_EQW"
        # if baskets missing (e.g., only one in a bucket), build them on the fly
        if gold_b not in agg:
            gold_b = gold[0]
        crypto_candidate = crypto[0]
        blend_prices = px[[gold[0], crypto_candidate]].dropna()
        blend = backtest_blend(blend_prices, {gold[0]: 0.7, crypto_candidate: 0.3}, cfg.rebalance)
        blends[f"70_{gold[0]}_30_{crypto_candidate}_{cfg.rebalance}"] = blend
        # 50/50 too
        blend2 = backtest_blend(blend_prices, {gold[0]: 0.5, crypto_candidate: 0.5}, cfg.rebalance)
        blends[f"50_{gold[0]}_50_{crypto_candidate}_{cfg.rebalance}"] = blend2
    if blends:
        blends_df = pd.DataFrame(blends)
        blends_df.to_csv(os.path.join(cfg.outdir, "blend_returns.csv"))
        # stats
        br = []
        for c in blends_df.columns:
            s = ann_stats(blends_df[c], rf=cfg.rf)
            s["ticker"] = c
            s["max_drawdown"] = max_drawdown(blends_df[c])
            br.append(s)
        pd.DataFrame(br)[["ticker","ann_return","ann_vol","sharpe","max_drawdown"]].to_csv(
            os.path.join(cfg.outdir, "blend_stats.csv"), index=False
        )

    # Episode scorecards
    if cfg.episodes:
        ep_rows = []
        for a, b, lab in cfg.episodes:
            sub = rets.loc[(rets.index >= a) & (rets.index <= b)]
            if sub.empty:
                continue
            for col in sub.columns:
                ep = {
                    "episode": lab, "start": str(a.date()), "end": str(b.date()),
                    "ticker": col,
                    "total_return": float((1 + sub[col]).prod() - 1),
                    "ann_return": float(sub[col].mean() * 252),
                    "vol": float(sub[col].std(ddof=1) * np.sqrt(252)),
                    "max_drawdown": float(max_drawdown(sub[col]))
                }
                ep_rows.append(ep)
        ep_df = pd.DataFrame(ep_rows)
        if not ep_df.empty:
            ep_df.to_csv(os.path.join(cfg.outdir, "episode_scorecards.csv"), index=False)

    # --------------- Plots ---------------
    if cfg.plot and plt is not None:
        # Levels
        fig1 = plt.figure(figsize=(10, 5)); ax = plt.gca()
        (px[crypto + gold].dropna().div(px[crypto + gold].dropna().iloc[0]) ).plot(ax=ax)
        ax.set_title("Normalized price (t0=1): Crypto ETFs vs Gold")
        ax.set_ylabel("Index")
        plt.tight_layout(); fig1.savefig(os.path.join(cfg.outdir, "plots", "normalized_levels.png"), dpi=140); plt.close(fig1)

        # Rolling 60d correlation to gold ref
        if gold_ref:
            corr_cols = [f"corr_{t}_to_{gold_ref}" for t in px.columns if f"corr_{t}_to_{gold_ref}" in roll]
            if corr_cols:
                fig2 = plt.figure(figsize=(10, 5)); ax2 = plt.gca()
                pd.DataFrame({c: roll[c] for c in corr_cols}).plot(ax=ax2)
                ax2.axhline(0, linestyle="--")
                ax2.set_title(f"Rolling 60D correlation to {gold_ref}")
                plt.tight_layout(); fig2.savefig(os.path.join(cfg.outdir, "plots", "rolling_corr_to_gold.png"), dpi=140); plt.close(fig2)

        # Blend equity curves
        if blends:
            fig3 = plt.figure(figsize=(10,5)); ax3 = plt.gca()
            w0 = (1 + pd.DataFrame(blends).iloc[1:]).cumprod()
            w0.plot(ax=ax3)
            ax3.set_title("Blend equity curves")
            ax3.set_ylabel("Wealth (start=1)")
            plt.tight_layout(); fig3.savefig(os.path.join(cfg.outdir, "plots", "blends_equity.png"), dpi=140); plt.close(fig3)

    # Console snapshot
    print("\n=== Asset stats ===")
    print(stats.round(4).to_string(index=False))
    if os.path.exists(os.path.join(cfg.outdir, "basket_stats.csv")):
        print("\n=== Basket stats ===")
        print(pd.read_csv(os.path.join(cfg.outdir, "basket_stats.csv")).round(4).to_string(index=False))
    if os.path.exists(os.path.join(cfg.outdir, "blend_stats.csv")):
        print("\n=== Blend stats ===")
        print(pd.read_csv(os.path.join(cfg.outdir, "blend_stats.csv")).round(4).to_string(index=False))


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Crypto ETF vs Gold analysis")
    ap.add_argument("--crypto", type=str, default="IBIT,FBTC,BITO",
                    help="Comma-separated crypto ETF tickers (e.g., IBIT,FBTC,BITO,ARKB,DEFI)")
    ap.add_argument("--gold", type=str, default="GLD,IAU", help="Comma-separated gold proxies")
    ap.add_argument("--equity", type=str, default="SPY", help="Benchmark equity index ETF (optional)")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--rf", type=float, default=0.0, help="Annualized risk-free rate (for Sharpe)")
    ap.add_argument("--rebalance", choices=["m","q","w"], default="m", help="Blend rebalance frequency")
    ap.add_argument("--episodes", type=str, nargs="*", default=[],
                    help="Episode windows like 'YYYY-MM-DD:YYYY-MM-DD Label'")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        crypto=parse_list_arg(args.crypto),
        gold=parse_list_arg(args.gold),
        equity=(args.equity or None),
        start=args.start,
        end=args.end,
        rf=float(args.rf),
        rebalance=args.rebalance,
        episodes=parse_episodes(args.episodes) if args.episodes else [],
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir),
    )

    print(f"[INFO] Universe: crypto={cfg.crypto} gold={cfg.gold} equity={[cfg.equity] if cfg.equity else []}")
    run(cfg)
    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()