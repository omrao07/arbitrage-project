# backend/analytics/corr_rollup.py
from __future__ import annotations

"""
Correlation Rollup
------------------
Compute pairwise correlations on prices or returns, with rolling windows
and point-in-time snapshots. Designed to feed the frontend corr-matrix.tsx.

Features
- Load prices or returns (plug your data_api here)
- Compute log/pct returns
- Point-in-time corr (Pearson/Spearman)
- Rolling corr windows with end timestamps
- Optional group rollups (e.g., by sector/region/strategy tag)
- Export: JSON (for UI), Parquet/CSV (for offline), Redis publish (stream)
- CLI for ad-hoc runs or cron

Typical uses
- Nightly job to refresh 1M/3M/1Y matrices
- Intraday quick snapshot on last N bars for a dashboard

Dependencies: pandas, numpy, (optional) pyarrow for parquet, redis
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

try:
    import redis  # optional
except Exception:  # pragma: no cover
    redis = None  # type: ignore

CorrMethod = Literal["pearson", "spearman"]

# --------- Config / Defaults ---------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_STREAM = os.getenv("ANALYTICS_CORR_STREAM", "analytics.corr")

# Where to write artifacts if you want files
OUT_DIR = os.getenv("AN_OUT_DIR", "data/corr")

# --------- Data Contract ----------------------------------------------------
@dataclass
class SeriesMeta:
    label: str            # label in the matrix (e.g., "AAPL" or "alpha_mean_rev")
    group: Optional[str]  # e.g., sector/region/strategy bucket


# You can swap this with your real data loader.
def load_prices(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    bar: str = "1d",
) -> pd.DataFrame:
    """
    Return a price dataframe indexed by timestamp with columns = symbols.
    Replace this stub with backend.data_api.get_prices(...)
    """
    # --- Stub: generate toy data ---
    idx = pd.date_range(end=pd.Timestamp(end) if end else pd.Timestamp.utcnow(), periods=300, freq="D")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(index=idx)
    for s in symbols:
        px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=len(idx)))
        df[s] = px
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    return df


# --------- Core Computations ----------------------------------------------
def to_returns(
    df_px: pd.DataFrame,
    kind: Literal["log", "pct"] = "log",
    dropna: bool = True,
) -> pd.DataFrame:
    if kind == "log":
        ret = np.log(df_px).diff() # type: ignore
    else:
        ret = df_px.pct_change()
    return ret.dropna(how="all") if dropna else ret


def corr_matrix(
    df: pd.DataFrame,
    method: CorrMethod = "pearson",
) -> pd.DataFrame:
    """
    df: columns = labels, rows = aligned times (returns recommended)
    """
    if method == "pearson":
        return df.corr(method="pearson")
    elif method == "spearman":
        return df.corr(method="spearman")
    else:
        raise ValueError(f"Unsupported method: {method}")


def rolling_corr(
    df: pd.DataFrame,
    window: int,
    method: CorrMethod = "pearson",
    min_periods: Optional[int] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Returns a dict of end_timestamp -> correlation matrix for each rolling window.
    window counts rows (bars). Ensure df is sorted by index ascending.
    """
    df = df.sort_index()
    mp = min_periods or window
    out: Dict[pd.Timestamp, pd.DataFrame] = {}

    # Optimize: compute corr only on valid windows
    for i in range(window - 1, len(df)):
        sl = df.iloc[i - window + 1 : i + 1]
        if sl.count().min() < mp:
            continue
        out[df.index[i]] = corr_matrix(sl, method=method)
    return out


# --------- Packaging for Frontend -----------------------------------------
def to_frontend_payload(
    labels: List[str],
    matrix_df: pd.DataFrame,
    method: CorrMethod,
    as_list: bool = False,
) -> Dict:
    """
    Build a payload consumed directly by corr-matrix.tsx:
    {
      "labels": [...],
      "matrix": [[...], ...],
      "method": "pearson"
    }
    """
    # Align order
    matrix_df = matrix_df.reindex(index=labels, columns=labels)
    mat = matrix_df.values.tolist()
    payload = {"labels": labels, "matrix": mat, "method": method}
    return payload if as_list else payload


def summarize_clusters(matrix_df: pd.DataFrame, groups: Dict[str, str]) -> pd.DataFrame:
    """
    Optional: roll correlation by group (e.g., sectors). Returns group-to-group mean corr.
    groups: map label -> group
    """
    labels = list(matrix_df.columns)
    groups_df = pd.DataFrame({"label": labels, "group": [groups.get(l, "NA") for l in labels]})
    grp = groups_df.groupby("group")["label"].apply(list)

    agg = pd.DataFrame(index=grp.index, columns=grp.index, dtype=float)
    for g1, lab1 in grp.items():
        for g2, lab2 in grp.items():
            block = matrix_df.loc[lab1, lab2].values
            agg.loc[g1, g2] = float(np.nanmean(block)) # type: ignore
    return agg


# --------- Exporters -------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    try:
        df.to_parquet(path)  # requires pyarrow or fastparquet
    except Exception:
        # fallback to csv if parquet not available
        path_csv = path.rsplit(".", 1)[0] + ".csv"
        df.to_csv(path_csv, index=True)


def _json_default(o):
    if isinstance(o, (pd.Timestamp, pd.Timedelta, np.datetime64)):
        return pd.Timestamp(o).isoformat() # type: ignore
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


# --------- Redis Publisher -------------------------------------------------
def publish_redis(payload: Dict, *, stream: str = REDIS_STREAM, host: str = REDIS_HOST, port: int = REDIS_PORT) -> None:
    if redis is None:
        print("[corr_rollup] redis not installed; skip publish")
        return
    r = redis.Redis(host=host, port=port, decode_responses=True)
    r.xadd(stream, {"ts_ms": int(time.time() * 1000), "payload": json.dumps(payload)})


# --------- Orchestrators ---------------------------------------------------
def run_point_in_time(
    labels: List[str],
    price_df: Optional[pd.DataFrame] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    returns_kind: Literal["log", "pct"] = "log",
    method: CorrMethod = "pearson",
    groups: Optional[Dict[str, str]] = None,
) -> Tuple[Dict, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns:
      - payload dict for UI (labels, matrix, method)
      - matrix DataFrame
      - optional group-summary DataFrame
    """
    if price_df is None:
        price_df = load_prices(labels, start=start, end=end)
    ret = to_returns(price_df, kind=returns_kind).dropna(how="all")
    mat = corr_matrix(ret, method=method)
    payload = to_frontend_payload(labels=labels, matrix_df=mat, method=method)
    grp_df = summarize_clusters(mat, groups) if groups else None
    return payload, mat, grp_df


def run_rolling(
    labels: List[str],
    window: int = 60,
    price_df: Optional[pd.DataFrame] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    returns_kind: Literal["log", "pct"] = "log",
    method: CorrMethod = "pearson",
) -> Dict[str, Dict]:
    """
    Produce a dict of ISO end timestamps -> {labels, matrix, method}
    Useful for time slider UIs or backtests.
    """
    if price_df is None:
        price_df = load_prices(labels, start=start, end=end)
    ret = to_returns(price_df, kind=returns_kind).dropna(how="all")
    mats = rolling_corr(ret, window=window, method=method)
    out: Dict[str, Dict] = {}
    for ts, dfm in mats.items():
        out[pd.Timestamp(ts).isoformat()] = to_frontend_payload(labels=labels, matrix_df=dfm, method=method)
    return out


# --------- CLI -------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlation rollup generator")
    p.add_argument("--labels", type=str, required=True, help="Comma-separated labels/symbols")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--returns", type=str, choices=["log", "pct"], default="log")
    p.add_argument("--method", type=str, choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--window", type=int, default=0, help="Rolling window bars; 0 = point-in-time only")
    p.add_argument("--out", type=str, default=OUT_DIR, help="Output directory for files")
    p.add_argument("--publish", action="store_true", help="Publish to Redis stream")
    p.add_argument("--name", type=str, default="default", help="Name tag for filenames/stream")
    return p.parse_args()


def main():
    args = _parse_args()
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels:
        raise SystemExit("No labels provided")

    ensure_dir(args.out)

    if args.window and args.window > 0:
        roll = run_rolling(
            labels=labels,
            window=args.window,
            start=args.start,
            end=args.end,
            returns_kind=args.returns,  # type: ignore
            method=args.method,         # type: ignore
        )
        # persist
        save_json(roll, os.path.join(args.out, f"corr_rolling_{args.name}.json"))
        if args.publish:
            publish_redis({"kind": "corr_rolling", "name": args.name, "data": roll})
        print(f"[corr_rollup] wrote rolling matrices for {len(roll)} windows")
    else:
        payload, mat, _grp = run_point_in_time(
            labels=labels,
            start=args.start,
            end=args.end,
            returns_kind=args.returns,  # type: ignore
            method=args.method,         # type: ignore
        )
        save_json(payload, os.path.join(args.out, f"corr_snapshot_{args.name}.json"))
        # Optional: store full matrix as parquet/csv
        save_parquet(mat, os.path.join(args.out, f"corr_snapshot_{args.name}.parquet"))
        if args.publish:
            publish_redis({"kind": "corr_snapshot", "name": args.name, "data": payload})
        print(f"[corr_rollup] wrote snapshot matrix for {len(payload['labels'])} labels")


if __name__ == "__main__":
    main()