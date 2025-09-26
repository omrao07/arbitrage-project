#!/usr/bin/env python3
"""
pull_betas.py
-------------
Compute rolling/point betas of tickers vs a benchmark from CSV prices,
then aggregate to sector and region. Writes curated outputs.

Usage:
  python pull_betas.py --prices data/prices/prices_long.csv --format long --benchmark SPY --freq D --window 252
  python pull_betas.py --prices data/prices/prices_wide.csv --format wide --benchmark SPY --freq W --window 104
  # with metadata for aggregation
  python pull_betas.py --prices data/prices/prices_long.csv --format long --benchmark SPY --metadata data/reference/metadata.csv

Outputs:
  data/adamodar/curated/betas_by_company.csv
  data/adamodar/curated/betas_by_sector.csv
  data/adamodar/curated/betas_by_region.csv
"""

import os
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd


CURATED_DIR = "data/adamodar/curated"
COMPANY_OUT = os.path.join(CURATED_DIR, "betas_by_company.csv")
SECTOR_OUT  = os.path.join(CURATED_DIR, "betas_by_sector.csv")
REGION_OUT  = os.path.join(CURATED_DIR, "betas_by_region.csv")


# ---------- I/O helpers ----------

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------- Loading & shaping ----------

def load_prices_long(path: str, price_col_candidates=("adj_close","close","price")) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # normalize price column name
    found = None
    for c in price_col_candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        raise ValueError(f"Expected one of {price_col_candidates} in long prices file")
    df = df.rename(columns={found: "price"})
    need = {"date","ticker","price"}
    if not need.issubset(df.columns):
        raise ValueError(f"Long format must include columns {need}")
    return df[["date","ticker","price"]].sort_values(["ticker","date"])

def load_prices_wide(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError("Wide format must include a 'date' column")
    return df.sort_values("date")

def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot(index="date", columns="ticker", values="price")
    wide = wide.sort_index()
    wide.columns.name = None
    return wide

def resample_freq(prices_wide: pd.DataFrame, freq_code: str) -> pd.DataFrame:
    """
    Resample to D (daily) or W (weekly, Friday close).
    """
    if freq_code.upper() == "D":
        return prices_wide.asfreq("B").ffill()  # business daily
    elif freq_code.upper() == "W":
        return prices_wide.resample("W-FRI").last().dropna(how="all")
    else:
        raise ValueError("freq must be 'D' or 'W'")


# ---------- Returns & OLS beta ----------

def pct_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    rets = prices_wide.pct_change().replace([np.inf, -np.inf], np.nan)
    return rets

def ols_beta_alpha(x: pd.Series, y: pd.Series) -> Tuple[float,float,float,float]:
    """
    OLS of y ~ alpha + beta * x.
    Returns: beta, alpha, R^2, stderr_beta.
    """
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 10:
        return np.nan, np.nan, np.nan, np.nan
    X = df.iloc[:,0].values
    Y = df.iloc[:,1].values
    X_ = np.column_stack([np.ones_like(X), X]) # type: ignore
    coef, *_ = np.linalg.lstsq(X_, Y, rcond=None) # type: ignore
    alpha, beta = coef[0], coef[1]
    y_hat = X_ @ coef
    resid = Y - y_hat
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((Y - Y.mean())**2) # type: ignore
    r2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else np.nan
    # stderr(beta)
    n, k = len(Y), 2
    sigma2 = ss_res / (n - k) if n > k else np.nan
    XtX_inv = np.linalg.inv(X_.T @ X_)
    var_beta = sigma2 * XtX_inv[1,1] if not np.isnan(sigma2) else np.nan
    se_beta = np.sqrt(var_beta) if var_beta is not None and var_beta >= 0 else np.nan
    return float(beta), float(alpha), float(r2), float(se_beta)

def rolling_beta(bench_ret: pd.Series, asset_ret: pd.Series, window: int) -> pd.Series:
    """
    Fast rolling beta via cov/var. Aligns on index.
    """
    aligned = pd.concat([bench_ret, asset_ret], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    b = aligned.iloc[:,0]
    a = aligned.iloc[:,1]
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    beta = cov / var
    beta.name = asset_ret.name
    return beta


# ---------- Company betas ----------

def compute_company_betas(rets: pd.DataFrame, benchmark: str, window: int) -> pd.DataFrame:
    if benchmark not in rets.columns:
        raise ValueError(f"Benchmark '{benchmark}' not found in returns columns")
    bench = rets[benchmark]

    rows = []
    # point-in-time using rolling window ending at last date
    end_date = rets.index.max()

    for tk in rets.columns:
        if tk == benchmark: 
            continue
        rb = rolling_beta(bench, rets[tk], window)
        beta_pt = rb.dropna().iloc[-1] if not rb.dropna().empty else np.nan

        # OLS stats on the same trailing window
        trailing = pd.concat([bench, rets[tk]], axis=1).dropna().iloc[-window:]
        beta, alpha, r2, se = (np.nan,)*4
        if len(trailing) >= max(30, window//4):
            beta, alpha, r2, se = ols_beta_alpha(trailing.iloc[:,0], trailing.iloc[:,1])

        rows.append({
            "as_of": end_date.strftime("%Y-%m-%d"),
            "ticker": tk,
            "benchmark": benchmark,
            "window": window,
            "freq": "daily" if rets.index.inferred_freq in ("B","C") else "weekly", # type: ignore
            "beta": beta_pt,
            "alpha": alpha,
            "rsq": r2,
            "stderr": se,
            "method": "rolling_cov_var+OLS_window"
        })

    return pd.DataFrame(rows)


# ---------- Aggregation ----------

def aggregate_betas(company: pd.DataFrame, meta: Optional[pd.DataFrame], level: str, benchmark: str) -> pd.DataFrame:
    """
    level: 'sector' or 'region'
    """
    if meta is None or level not in ["sector","region"]:
        return pd.DataFrame(columns=["as_of",level,"benchmark","window","freq","beta_cap_wt","beta_eq_wt","count","method"])

    df = company.merge(meta[["ticker", level, "market_cap_usd"]], on="ticker", how="left")
    df = df.dropna(subset=[level])

    # Equal-weight beta = mean of betas for tickers in bucket
    eq = df.groupby(level)["beta"].mean()

    # Cap-weight beta = sum(w_i * beta_i)
    mcap = df.groupby(level)["market_cap_usd"].sum().replace(0, np.nan)
    weights = df.set_index(level)["market_cap_usd"] / mcap.reindex(df[level].values).values
    df["w"] = weights.values
    cap = df.groupby(level).apply(lambda g: np.nansum(g["beta"] * (g["w"].fillna(0)))).replace([np.inf,-np.inf], np.nan)

    out = pd.DataFrame({
        level: eq.index,
        "beta_eq_wt": eq.values,
        "beta_cap_wt": cap.reindex(eq.index).values,
    })
    out["as_of"] = company["as_of"].iloc[0] if not company.empty else pd.NaT
    out["benchmark"] = benchmark
    out["window"] = company["window"].iloc[0] if not company.empty else np.nan
    out["freq"] = company["freq"].iloc[0] if not company.empty else ""
    out["count"] = df.groupby(level)["beta"].size().reindex(eq.index).values
    out["method"] = "cap_wt=sum(w*beta); eq_wt=mean(beta)"
    cols = ["as_of", level, "benchmark", "window", "freq", "beta_cap_wt", "beta_eq_wt", "count", "method"]
    return out[cols].sort_values(level)


# ---------- Main pipeline ----------

def run(prices_path: str,
        fmt: str,
        benchmark: str,
        metadata_path: Optional[str],
        freq_code: str,
        window: int) -> None:

    # Load prices
    if fmt == "long":
        long = load_prices_long(prices_path)
        wide = to_wide(long)
    elif fmt == "wide":
        wide = load_prices_wide(prices_path)
    else:
        raise ValueError("format must be 'long' or 'wide'")

    # Resample & compute returns
    wide = resample_freq(wide, freq_code)
    if benchmark not in wide.columns:
        raise ValueError(f"Benchmark '{benchmark}' not found in columns")
    rets = pct_returns(wide)

    # Compute company betas
    company = compute_company_betas(rets, benchmark, window)

    # Metadata for aggregation
    meta = None
    if metadata_path and os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        # Normalize sector/region if present
        for col in ["sector","region"]:
            if col in meta.columns:
                meta[col] = meta[col].fillna("Unknown")
        if "market_cap_usd" not in meta.columns:
            meta["market_cap_usd"] = np.nan

    # Aggregate
    sector = aggregate_betas(company, meta, "sector", benchmark) if meta is not None else pd.DataFrame()
    region = aggregate_betas(company, meta, "region", benchmark) if meta is not None else pd.DataFrame()

    # Write outputs
    ensure_dir(COMPANY_OUT); company.to_csv(COMPANY_OUT, index=False)
    if not sector.empty:
        ensure_dir(SECTOR_OUT); sector.to_csv(SECTOR_OUT, index=False)
    if not region.empty:
        ensure_dir(REGION_OUT); region.to_csv(REGION_OUT, index=False)

    print(f"✅ wrote {COMPANY_OUT} ({len(company)} rows)")
    if not sector.empty:
        print(f"✅ wrote {SECTOR_OUT} ({len(sector)} rows)")
    if not region.empty:
        print(f"✅ wrote {REGION_OUT} ({len(region)} rows)")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Compute betas vs benchmark and aggregate to sector/region.")
    ap.add_argument("--prices", required=True, help="Path to prices CSV (long or wide)")
    ap.add_argument("--format", dest="fmt", choices=["long","wide"], required=True, help="Input format")
    ap.add_argument("--benchmark", required=True, help="Benchmark ticker present in prices")
    ap.add_argument("--metadata", dest="metadata_path", default=None, help="Optional metadata CSV for sector/region/market_cap")
    ap.add_argument("--freq", choices=["D","W"], default="D", help="Sampling frequency: D=daily (business), W=weekly (Fri close)")
    ap.add_argument("--window", type=int, default=None, help="Rolling window length (default 252 for D, 104 for W)")
    args = ap.parse_args()

    window = args.window or (252 if args.freq == "D" else 104)
    run(args.prices, args.fmt, args.benchmark, args.metadata_path, args.freq, window)

if __name__ == "__main__":
    main()