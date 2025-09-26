# macro_inflation.py
"""
Macro inflation feature engineering without Feast.

Inputs
------
A table of inflation releases, minimally:
  date        : string or datetime (UTC or naive)
  country     : e.g. "US", "EA", "UK", "JP"
  cpi_index   : CPI level (index, NOT percent) OR headline_yoy if you lack levels
Optional columns:
  core_index        : Core CPI index level
  headline_yoy      : precomputed YoY headline inflation (%)
  core_yoy          : precomputed YoY core (%)
  exp_headline_yoy  : survey/market expectation (%)
  exp_core_yoy      : expectation for core (%)
  target_yoy        : inflation target, e.g. 2.0

Outputs
-------
DataFrame with engineered features per (country, date), including:
  headline_yoy, core_yoy, headline_mom_sa, core_mom_sa
  headline_momentum_3m, core_momentum_3m
  headline_z, core_z         (cross-country normalized by rolling 5y mean/std)
  headline_surprise, core_surprise
  gap_to_target, core_gap_to_target
  inflation_regime           ("low","moderate","high","very_high")
  sticky_flex_proxy          (sticky vs flexible inflation proxy: long MA / short MA)
  vol_12m                    (headline YoY volatility)
  ...

Usage
-----
>>> from macro_inflation import load_data, build_features, save_parquet
>>> df = load_data("data/macro/inflation/*.parquet")     # or .csv
>>> feats = build_features(df)
>>> save_parquet(feats, "data/macro/features/inflation", partition_cols=["country"])

CLI
---
python macro_inflation.py --input data/... --output data/... --write
"""

from __future__ import annotations
import io
import os
import sys
import math
import typing as T
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

try:
    import fsspec  # optional, for s3:// or gcs:// paths
    HAVE_FSSPEC = True
except Exception:
    HAVE_FSSPEC = False


# ------------------------------- IO ---------------------------------

def _open(path: str, mode: str = "rb"):
    """Open local or cloud file via fsspec if available."""
    if (path.startswith("s3://") or path.startswith("gcs://") or path.startswith("gs://")) and HAVE_FSSPEC:
        fs, _, paths = fsspec.get_fs_token_paths(path)
        return fs.open(paths[0], mode)
    return open(path, mode)


def load_data(path_or_glob: str) -> pd.DataFrame:
    """
    Load CPI data from CSV/Parquet. Supports single file or glob (local/s3/gcs).
    Required columns: ['date','country'] and either 'cpi_index' or 'headline_yoy'.
    """
    paths: list[str] = []

    # Resolve globs
    if HAVE_FSSPEC and (path_or_glob.startswith(("s3://", "gcs://", "gs://"))):
        fs, _, matches = fsspec.get_fs_token_paths(path_or_glob)
        paths = matches if isinstance(matches, list) else [matches]
    else:
        import glob
        paths = glob.glob(path_or_glob)

    if not paths:
        # maybe a single file literal
        paths = [path_or_glob]

    frames = []
    for p in paths:
        if p.lower().endswith(".parquet") or p.lower().endswith(".pq"):
            with _open(p, "rb") as f:
                frames.append(pd.read_parquet(f))
        elif p.lower().endswith(".csv"):
            with _open(p, "rb") as f:
                frames.append(pd.read_csv(f))
        else:
            raise ValueError(f"Unsupported file type: {p}")

    df = pd.concat(frames, ignore_index=True).drop_duplicates()

    # Basic cleanup
    if "date" not in df.columns or "country" not in df.columns:
        raise ValueError("Input must contain 'date' and 'country' columns.")

    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df["country"] = df["country"].astype("string").str.upper().str.strip()

    # Ensure numeric
    for col in [
        "cpi_index", "core_index", "headline_yoy", "core_yoy",
        "exp_headline_yoy", "exp_core_yoy", "target_yoy"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["country", "date"]).reset_index(drop=True)


# -------------------------- Feature logic ---------------------------

@dataclass
class FeatureConfig:
    yoy_window_months: int = 12     # for YoY calculation if index provided
    mom_sa_window: int = 3          # seasonal adjustment via 3m centered moving avg
    momentum_window: int = 3        # 3m momentum change
    zscore_lookback_months: int = 60  # 5y rolling z-score
    vol_window: int = 12            # 12m volatility of YoY
    sticky_ma_long: int = 12        # long MA for sticky proxy
    sticky_ma_short: int = 3        # short MA for flexible proxy
    regime_breaks: tuple[float, float, float] = (1.0, 3.0, 6.0)  # YoY% thresholds


def _pct_change(a: pd.Series, periods: int) -> pd.Series:
    return (a / a.shift(periods) - 1.0) * 100.0


def _seasonally_adjusted_mom(index: pd.Series, window: int = 3) -> pd.Series:
    """
    Crude MoM SA via centered moving average on index level.
    mom_sa = (MA(t) / MA(t-1) - 1)*100
    """
    ma = index.rolling(window=window, center=True, min_periods=window//2+1).mean()
    return (ma / ma.shift(1) - 1.0) * 100.0


def _rolling_zscore(s: pd.Series, lookback: int) -> pd.Series:
    m = s.rolling(lookback, min_periods=max(6, lookback//6)).mean()
    sd = s.rolling(lookback, min_periods=max(6, lookback//6)).std()
    return (s - m) / sd


def _regime(yoy: pd.Series, brks: tuple[float, float, float]) -> pd.Series:
    low, mod, high = brks
    # bins: (-inf,low], (low,mod], (mod,high], (high, inf)
    cats = pd.cut(
        yoy,
        bins=[-1e9, low, mod, high, 1e9],
        labels=["low", "moderate", "high", "very_high"],
        right=True
    ).astype("string")
    return cats


def _sticky_flex_proxy(yoy: pd.Series, long_ma: int, short_ma: int) -> pd.Series:
    long = yoy.rolling(long_ma, min_periods=max(2, long_ma//3)).mean()
    short = yoy.rolling(short_ma, min_periods=max(1, short_ma//2)).mean()
    # >1 indicates stickier inflation (long trend dominates). <1 more flexible.
    with pd.option_context("mode.use_inf_as_na", True):
        ratio = (long / short).replace([pd.NA, pd.NaT], pd.NA)#type:ignore
    return ratio


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Compute inflation features per country.
    """
    cfg = cfg or FeatureConfig()
    required_any = (("cpi_index" in df.columns) or ("headline_yoy" in df.columns))
    if not required_any:
        raise ValueError("Provide 'cpi_index' or 'headline_yoy' in input.")

    out = []
    for country, g in df.groupby("country", sort=True):
        g = g.sort_values("date").copy()

        # Headline YoY
        if "headline_yoy" not in g or g["headline_yoy"].isna().all():
            if "cpi_index" not in g:
                raise ValueError(f"{country}: need 'cpi_index' or 'headline_yoy'.")
            g["headline_yoy"] = _pct_change(g["cpi_index"], cfg.yoy_window_months)

        # Core YoY
        if "core_yoy" not in g and "core_index" in g:
            g["core_yoy"] = _pct_change(g["core_index"], cfg.yoy_window_months)

        # MoM SA (crude) from indices if available
        if "cpi_index" in g:
            g["headline_mom_sa"] = _seasonally_adjusted_mom(g["cpi_index"], cfg.mom_sa_window)
        if "core_index" in g:
            g["core_mom_sa"] = _seasonally_adjusted_mom(g["core_index"], cfg.mom_sa_window)

        # Momentum (change in YoY over 3 months)
        g["headline_momentum_3m"] = g["headline_yoy"] - g["headline_yoy"].shift(cfg.momentum_window)
        if "core_yoy" in g:
            g["core_momentum_3m"] = g["core_yoy"] - g["core_yoy"].shift(cfg.momentum_window)

        # Z-scores vs country history (5y rolling)
        g["headline_z"] = _rolling_zscore(g["headline_yoy"], cfg.zscore_lookback_months)
        if "core_yoy" in g:
            g["core_z"] = _rolling_zscore(g["core_yoy"], cfg.zscore_lookback_months)

        # Surprise vs expectations (if available)
        if "exp_headline_yoy" in g:
            g["headline_surprise"] = g["headline_yoy"] - g["exp_headline_yoy"]
        if "exp_core_yoy" in g and "core_yoy" in g:
            g["core_surprise"] = g["core_yoy"] - g["exp_core_yoy"]

        # Gap to target (default 2% if target missing)
        tgt = g["target_yoy"] if "target_yoy" in g else 2.0
        g["gap_to_target"] = g["headline_yoy"] - tgt
        if "core_yoy" in g:
            g["core_gap_to_target"] = g["core_yoy"] - (g["target_yoy"] if "target_yoy" in g else 2.0)

        # Regime classification
        g["inflation_regime"] = _regime(g["headline_yoy"], cfg.regime_breaks)

        # Sticky vs flexible proxy
        g["sticky_flex_proxy"] = _sticky_flex_proxy(g["headline_yoy"], cfg.sticky_ma_long, cfg.sticky_ma_short)

        # Volatility of YoY (12m rolling std)
        g["vol_12m"] = g["headline_yoy"].rolling(cfg.vol_window, min_periods=6).std()

        out.append(g)

    feats = pd.concat(out, ignore_index=True)
    # Order columns roughly by importance
    col_order = [
        "date", "country",
        "headline_yoy", "core_yoy",
        "headline_mom_sa", "core_mom_sa",
        "headline_momentum_3m", "core_momentum_3m",
        "headline_surprise", "core_surprise",
        "gap_to_target", "core_gap_to_target",
        "headline_z", "core_z",
        "vol_12m", "sticky_flex_proxy",
        "inflation_regime",
    ]
    existing = [c for c in col_order if c in feats.columns]
    remaining = [c for c in feats.columns if c not in existing]
    feats = feats[existing + remaining].sort_values(["country", "date"]).reset_index(drop=True)
    return feats


# ------------------------------ Save --------------------------------

def save_parquet(df: pd.DataFrame, root: str, partition_cols: list[str] | None = None) -> None:
    """
    Save DataFrame to partitioned Parquet (local or s3/gcs if fsspec is installed).
    """
    partition_cols = partition_cols or []
    if (root.startswith(("s3://", "gcs://", "gs://")) and HAVE_FSSPEC):
        # Simple multi-file write by groupby, good enough for moderate sizes.
        fs, _, _ = fsspec.get_fs_token_paths(root.rstrip("/") + "/_touch")
        for keys, g in df.groupby(partition_cols, dropna=False) if partition_cols else [((), df)]:
            subpath = root.rstrip("/")
            if partition_cols:
                keys = keys if isinstance(keys, tuple) else (keys,)
                for c, v in zip(partition_cols, keys):
                    subpath += f"/{c}={str(v)}"
            # ensure dir
            fs.mkdirs(subpath, exist_ok=True)
            with fs.open(f"{subpath}/part.parquet", "wb") as f:
                g.to_parquet(f, index=False)
    else:
        # Local filesystem via pandas’ partition-on-write approximation
        if not partition_cols:
            os.makedirs(root, exist_ok=True)
            df.to_parquet(os.path.join(root, "part.parquet"), index=False)
        else:
            for keys, g in df.groupby(partition_cols, dropna=False):
                subdir = root.rstrip("/")
                keys = keys if isinstance(keys, tuple) else (keys,)
                for c, v in zip(partition_cols, keys):
                    subdir += f"/{c}={str(v)}"
                os.makedirs(subdir, exist_ok=True)
                g.to_parquet(os.path.join(subdir, "part.parquet"), index=False)


# ------------------------------- CLI --------------------------------

def _parse_args(argv: list[str] | None = None):
    import argparse
    ap = argparse.ArgumentParser("macro_inflation features")
    ap.add_argument("--input", required=True, help="CSV/Parquet file or glob (local/s3/gcs)")
    ap.add_argument("--output", required=False, help="Output root folder for Parquet")
    ap.add_argument("--write", action="store_true", help="Write results as partitioned Parquet")
    ap.add_argument("--partition-cols", nargs="*", default=["country"])
    ap.add_argument("--regimes", nargs=3, type=float, default=(1.0, 3.0, 6.0), help="Regime breaks low/mod/high")
    args = ap.parse_args(argv)
    return args


def main(argv: list[str] | None = None):
    args = _parse_args(argv)

    df = load_data(args.input)
    cfg = FeatureConfig(regime_breaks=tuple(map(float, args.regimes))) # pyright: ignore[reportArgumentType]
    feats = build_features(df, cfg)

    if args.write:
        if not args.output:
            raise SystemExit("--output is required when --write is set")
        save_parquet(feats, args.output, partition_cols=args.partition_cols)
        print(f"[OK] wrote features to {args.output}")
    else:
        # Show a small preview
        print(feats.head(20).to_string(index=False))


if __name__ == "__main__":
    main()