# backend/liquidity/refit_liquidity_surface.py
from __future__ import annotations

"""
Refit Liquidity Surface
-----------------------
Given recent executions (and optional LOB stats), fit a parametric impact model:
    cost_bps ≈ c0
               + c_spread * spread_bps
               + temp_k * (POV ** alpha) * (H_ref / H) ** beta
               + perm_k * sqrt(Q / ADV)
               + v_k * (sigma20 * 100)

Where:
- POV = participation rate (% of market volume during execution window)
- H = execution horizon (minutes); H_ref = 30m (reference)
- Q = notional (or shares*price), ADV = average daily dollar volume (or shares)
- sigma20 = 20-day realized vol (daily, decimals)

Outputs a **surface**: grid of expected costs (bps) for horizons × POVs,
optionally per symbol and per venue.

Inputs (executions CSV/Parquet), expected columns (best-effort if missing):
  required: ts, symbol, side, qty, price, arrival_mid
  optional: spread (absolute), spread_bps, horizon_min, pov, notional, adv, venue
  optional LOB: book_imbalance, depth_1, depth_5, nbbo_bid, nbbo_ask
If 'pov' missing, we approximate using (qty / (ADV * H/390)).

CLI:
  python backend/liquidity/refit_liquidity_surface.py \
    --exec data/executions.parquet \
    --out data/liquidity/surface.json \
    --symbols AAPL,MSFT --venue-split \
    --adv-window 30 --publish

Dependencies: numpy, pandas, scipy (optimize). redis optional.
"""

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    import redis  # optional
except Exception:
    redis = None  # type: ignore


# ---------------- Defaults / Env ----------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_STREAM = os.getenv("ANALYTICS_LIQUIDITY_STREAM", "analytics.liquidity")

H_REF_MIN = 30.0  # reference horizon for scaling (minutes)
POV_GRID = [0.5, 1, 2, 5, 10, 15, 20, 30]  # %
H_GRID = [1, 5, 10, 15, 30, 60, 120, 240]  # minutes


# ---------------- Data utilities ----------------
def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _coalesce(series: pd.Series, fallback: float) -> pd.Series:
    s = series.copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.fillna(fallback)
    return s


def _to_bps(x: pd.Series) -> pd.Series:
    return 1e4 * x


def _infer_fields(d: pd.DataFrame, adv_by_symbol: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Ensure we have pov, horizon_min, spread_bps, notional, adv."""
    d = d.copy()

    # Timestamps
    if "ts" in d.columns:
        d["ts"] = pd.to_datetime(d["ts"], errors="coerce", utc=True).dt.tz_localize(None)

    # Notional
    if "notional" not in d.columns:
        d["notional"] = d.get("qty", 0) * d.get("price", 0)

    # Spread bps
    if "spread_bps" not in d.columns:
        if "spread" in d.columns and "arrival_mid" in d.columns:
            d["spread_bps"] = _to_bps(d["spread"] / d["arrival_mid"])
        elif {"nbbo_bid", "nbbo_ask", "arrival_mid"}.issubset(d.columns):
            d["spread_bps"] = _to_bps((d["nbbo_ask"] - d["nbbo_bid"]) / d["arrival_mid"])
        else:
            d["spread_bps"] = 5.0  # very rough default

    # Horizon (minutes)
    if "horizon_min" not in d.columns:
        # try duration or child order window
        if "duration_ms" in d.columns:
            d["horizon_min"] = d["duration_ms"] / 60000.0
        else:
            d["horizon_min"] = 5.0  # default

    # ADV
    if "adv" not in d.columns:
        if adv_by_symbol:
            d["adv"] = d["symbol"].map(adv_by_symbol).astype(float)
        else:
            # fallback: use 20 * median notional per symbol as pseudo ADV
            d["adv"] = d.groupby("symbol")["notional"].transform(lambda s: float(np.nanmedian(s) * 20) if len(s) else np.nan)
            d["adv"] = d["adv"].fillna(d["notional"].median() * 20 if len(d) else 1.0)

    # POV (%)
    if "pov" not in d.columns:
        # approximate: POV ≈ (Q / (ADV * H/390min)) * 100
        d["pov"] = (d["notional"] / (d["adv"] * (d["horizon_min"] / 390.0) + 1e-12)) * 100.0

    # Realized cost in bps vs arrival mid
    if "cost_bps" not in d.columns:
        if {"arrival_mid", "price", "side"}.issubset(d.columns):
            # buy: (exec - arrival)/arrival * 1e4 ; sell: (arrival - exec)/arrival * 1e4
            side = d["side"].str.lower().map({"b": "buy", "s": "sell"}).fillna(d["side"])
            buy_mask = side.isin(["buy", "b", "1", 1, True])
            sell_mask = ~buy_mask
            cost = np.zeros(len(d), dtype=float)
            arr = pd.to_numeric(d["arrival_mid"], errors="coerce")
            px = pd.to_numeric(d["price"], errors="coerce")
            cost[buy_mask] = ((px[buy_mask] - arr[buy_mask]) / arr[buy_mask]) * 1e4
            cost[sell_mask] = ((arr[sell_mask] - px[sell_mask]) / arr[sell_mask]) * 1e4
            d["cost_bps"] = cost
        else:
            raise ValueError("Need columns ['arrival_mid','price','side'] or precomputed 'cost_bps'.")

    # Sigma 20d (optional, else 0)
    if "sigma20" not in d.columns:
        d["sigma20"] = 0.0

    # Clean
    for col in ["spread_bps", "horizon_min", "pov", "notional", "adv", "cost_bps", "sigma20"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["spread_bps", "horizon_min", "pov", "notional", "adv", "cost_bps"])
    return d


# ---------------- Model ----------------
@dataclass
class LiquidityCoeffs:
    c0: float = 0.0
    c_spread: float = 0.6     # proportion of spread paid on average
    temp_k: float = 1.0
    alpha: float = 0.6        # POV exponent
    beta: float = 0.25        # horizon exponent (H_ref/H)^beta
    perm_k: float = 15.0
    v_k: float = 0.0          # vol sensitivity (bps per 1.0 daily vol)
    # metadata
    n: int = 0
    rms: float = 0.0

    def to_dict(self) -> Dict:
        d = asdict(self); return d


def _pred_cost_bps(x: np.ndarray, p: LiquidityCoeffs) -> np.ndarray:
    """
    x columns: [spread_bps, pov(%), horizon_min, q_over_adv, sigma20]
    """
    spread_bps = x[:, 0]
    pov = np.maximum(1e-6, x[:, 1]) / 100.0  # to fraction
    H = np.maximum(1e-6, x[:, 2])
    q_over_adv = np.maximum(1e-12, x[:, 3])
    sigma20 = x[:, 4]

    temp = p.temp_k * (pov ** p.alpha) * ((H_REF_MIN / H) ** p.beta)
    perm = p.perm_k * np.sqrt(q_over_adv)
    vol_term = p.v_k * (sigma20 * 100.0)

    return p.c0 + p.c_spread * spread_bps + temp + perm + vol_term


def fit_liquidity_model(df_exec: pd.DataFrame) -> LiquidityCoeffs:
    """Robust nonlinear least-squares fit for LiquidityCoeffs."""
    y = df_exec["cost_bps"].values.astype(float)
    X = np.stack([
        df_exec["spread_bps"].values.astype(float),
        df_exec["pov"].values.astype(float),
        df_exec["horizon_min"].values.astype(float),
        (df_exec["notional"] / np.maximum(1e-12, df_exec["adv"])).values.astype(float),
        df_exec["sigma20"].values.astype(float),
    ], axis=1)

    # initial guess
    p0 = LiquidityCoeffs(
        c0=np.median(y) - 0.5 * np.median(df_exec["spread_bps"]), # type: ignore
        c_spread=0.6,
        temp_k=np.percentile(np.maximum(y, 0), 60) / 2.0 if len(y) else 1.0, # type: ignore
        alpha=0.6,
        beta=0.25,
        perm_k=15.0,
        v_k=0.0,
    )

    def resid(theta: np.ndarray) -> np.ndarray:
        p = LiquidityCoeffs(
            c0=theta[0], c_spread=theta[1], temp_k=abs(theta[2]),
            alpha=np.clip(theta[3], 0.2, 1.5), beta=np.clip(theta[4], 0.0, 1.0),
            perm_k=abs(theta[5]), v_k=theta[6]
        )
        yhat = _pred_cost_bps(X, p)
        # Huber-ish: soft_l1 via least_squares(loss='soft_l1')
        return yhat - y

    theta0 = np.array([p0.c0, p0.c_spread, p0.temp_k, p0.alpha, p0.beta, p0.perm_k, p0.v_k], dtype=float)
    bounds_lo = np.array([-100.0, 0.0, 0.0, 0.1, 0.0, 0.0, -10.0])
    bounds_hi = np.array([ 100.0, 2.0, 500.0, 1.8, 1.0, 500.0,  10.0])

    res = least_squares(resid, theta0, bounds=(bounds_lo, bounds_hi), loss="soft_l1", f_scale=5.0, max_nfev=5000)
    t = res.x
    p = LiquidityCoeffs(
        c0=float(t[0]), c_spread=float(t[1]), temp_k=float(abs(t[2])),
        alpha=float(np.clip(t[3], 0.1, 1.8)), beta=float(np.clip(t[4], 0.0, 1.0)),
        perm_k=float(abs(t[5])), v_k=float(t[6]),
        n=int(len(y)), rms=float(np.sqrt(np.nanmean((res.fun) ** 2)))
    )
    return p


def expected_cost_bps(coeffs: LiquidityCoeffs, *, spread_bps: float, pov_pct: float, horizon_min: float, q_over_adv: float, sigma20: float = 0.0) -> float:
    x = np.array([[spread_bps, pov_pct, horizon_min, q_over_adv, sigma20]], dtype=float)
    return float(_pred_cost_bps(x, coeffs)[0])


# ---------------- Surface builder ----------------
@dataclass
class LiquiditySurface:
    coeffs: LiquidityCoeffs
    pov_grid: List[float]
    h_grid: List[float]
    # optional context
    spread_bps: float
    adv: float
    sigma20: float

    def to_grid(self, notional: float) -> Dict:
        """Return cost grid (bps) for given notional using coeffs."""
        q_over_adv = notional / max(1e-12, self.adv)
        G = np.zeros((len(self.h_grid), len(self.pov_grid)), dtype=float)
        for i, H in enumerate(self.h_grid):
            for j, pov in enumerate(self.pov_grid):
                G[i, j] = expected_cost_bps(
                    self.coeffs,
                    spread_bps=float(self.spread_bps),
                    pov_pct=float(pov),
                    horizon_min=float(H),
                    q_over_adv=float(q_over_adv),
                    sigma20=float(self.sigma20),
                )

        # Monotonic guards: cost should increase with POV and decrease with horizon (after scaling, still ensure non-decreasing in POV; non-increasing across H after undo scaling is tricky, use cumulative max across POV only)
        for i in range(G.shape[0]):
            G[i, :] = np.maximum.accumulate(G[i, :])

        # Clip tiny negatives (can appear from fit noise)
        G = np.maximum(G, 0.0)

        return {
            "pov": self.pov_grid,
            "horizons_min": self.h_grid,
            "cost_bps": G.tolist(),
            "coeffs": self.coeffs.to_dict(),
            "context": {
                "spread_bps": self.spread_bps,
                "adv": self.adv,
                "sigma20": self.sigma20,
                "notional": notional,
                "H_ref_min": H_REF_MIN,
            },
        }


# ---------------- Orchestration ----------------
def refit_from_exec(
    exec_df: pd.DataFrame,
    *,
    split_by: Optional[str] = None,  # "symbol", "venue", or None
    adv_by_symbol: Optional[Dict[str, float]] = None,
) -> Dict[str, LiquidityCoeffs]:
    """
    Fit coefficients globally or per group (symbol/venue).
    Returns mapping key -> coeffs.
    """
    d = _infer_fields(exec_df, adv_by_symbol=adv_by_symbol)

    if split_by in (None, ""):
        return {"ALL": fit_liquidity_model(d)}

    if split_by not in d.columns:
        raise ValueError(f"split_by '{split_by}' not in data columns")

    out: Dict[str, LiquidityCoeffs] = {}
    for key, sl in d.groupby(split_by):
        if len(sl) < 50:  # need enough data
            continue
        out[str(key)] = fit_liquidity_model(sl)
    return out


def build_surfaces(
    coeffs_map: Dict[str, LiquidityCoeffs],
    *,
    context_by_key: Dict[str, Dict],  # must include spread_bps, adv, sigma20
    pov_grid: Iterable[float] = POV_GRID,
    h_grid: Iterable[float] = H_GRID,
    notionals: Iterable[float] = (1e5, 5e5, 1e6),
) -> Dict[str, Dict]:
    """
    Return nested dict: key -> { notional_str: grid_payload }
    """
    pov = list(map(float, pov_grid))
    h = list(map(float, h_grid))
    out: Dict[str, Dict] = {}
    for key, coeffs in coeffs_map.items():
        ctx = context_by_key.get(key, {})
        spread_bps = float(ctx.get("spread_bps", 5.0))
        adv = float(ctx.get("adv", 5e7))
        sigma20 = float(ctx.get("sigma20", 0.0))
        surf = LiquiditySurface(coeffs=coeffs, pov_grid=pov, h_grid=h, spread_bps=spread_bps, adv=adv, sigma20=sigma20)
        payloads = {}
        for q in notionals:
            payloads[f"{int(q):d}"] = surf.to_grid(notional=float(q))
        out[key] = payloads
    return out


# ---------------- Persistence / Publish ----------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def publish_redis(obj: Dict, stream: str = REDIS_STREAM, host: str = REDIS_HOST, port: int = REDIS_PORT) -> None:
    if redis is None:
        print("[liquidity_surface] redis not installed; skip publish")
        return
    r = redis.Redis(host=host, port=port, decode_responses=True)
    r.xadd(stream, {"payload": json.dumps(obj)})


# ---------------- Online update (optional) ----------------
def ewma_update(old: LiquidityCoeffs, new: LiquidityCoeffs, alpha: float = 0.2) -> LiquidityCoeffs:
    """Blend coefficients for streaming recalibration."""
    def mix(a, b): return (1 - alpha) * a + alpha * b
    return LiquidityCoeffs(
        c0=mix(old.c0, new.c0),
        c_spread=mix(old.c_spread, new.c_spread),
        temp_k=mix(old.temp_k, new.temp_k),
        alpha=mix(old.alpha, new.alpha),
        beta=mix(old.beta, new.beta),
        perm_k=mix(old.perm_k, new.perm_k),
        v_k=mix(old.v_k, new.v_k),
        n=new.n, rms=new.rms
    )


# ---------------- CLI ----------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refit liquidity impact surface from executions.")
    p.add_argument("--exec", type=str, required=True, help="CSV/Parquet with executions")
    p.add_argument("--symbols", type=str, default="", help="optional comma list to filter symbols")
    p.add_argument("--split-by", type=str, default="symbol", choices=["symbol", "venue", "none"], help="fit per group or globally")
    p.add_argument("--out", type=str, default="data/liquidity/surface.json")
    p.add_argument("--publish", action="store_true", help="publish to Redis stream")
    p.add_argument("--pov-grid", type=str, default="")     # e.g., "0.5,1,2,5,10,20"
    p.add_argument("--h-grid", type=str, default="")       # e.g., "1,5,15,30,60"
    p.add_argument("--notionals", type=str, default="1e5,5e5,1e6")
    return p.parse_args()


def main():
    args = _parse_args()
    df = _read_table(args.exec)

    if args.symbols:
        keep = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper().isin(keep)]

    split = None if args.split_by == "none" else args.split_by

    coeffs_map = refit_from_exec(df, split_by=split)

    # Build minimal context per key
    context_by_key: Dict[str, Dict] = {}
    def robust_mean(x: pd.Series, default: float) -> float:
        x = pd.to_numeric(x, errors="coerce"); x = x.replace([np.inf, -np.inf], np.nan).dropna()
        return float(np.nanmedian(x)) if len(x) else default

    if split in (None, "none"):
        context_by_key["ALL"] = {
            "spread_bps": robust_mean(df.get("spread_bps", pd.Series()), 5.0),
            "adv": robust_mean(df.get("adv", pd.Series()), 5e7),
            "sigma20": robust_mean(df.get("sigma20", pd.Series()), 0.0),
        }
    else:
        for key, sl in df.groupby(split):
            context_by_key[str(key)] = {
                "spread_bps": robust_mean(sl.get("spread_bps", pd.Series()), 5.0),
                "adv": robust_mean(sl.get("adv", pd.Series()), 5e7),
                "sigma20": robust_mean(sl.get("sigma20", pd.Series()), 0.0),
            }

    pov_grid = [float(x) for x in args.pov_grid.split(",")] if args.pov_grid else POV_GRID
    h_grid = [float(x) for x in args.h_grid.split(",")] if args.h_grid else H_GRID
    notionals = [float(x) for x in args.notionals.split(",")] if args.notionals else (1e5, 5e5, 1e6)

    payload = build_surfaces(coeffs_map, context_by_key=context_by_key, pov_grid=pov_grid, h_grid=h_grid, notionals=notionals)

    ensure_dir(os.path.dirname(args.out))
    save_json(payload, args.out)
    print(f"[liquidity_surface] wrote {args.out} with {len(payload)} keys")

    if args.publish:
        publish_redis({"kind": "liquidity_surface", "data": payload})


if __name__ == "__main__":
    main()