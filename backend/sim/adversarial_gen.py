#!/usr/bin/env python3
"""
adversarial_gen.py — Generate adversarial return scenarios to stress a strategy

Goal
----
Given historical returns R[t, a] and time-varying positions/weights W[t-1, a],
construct adversarial perturbations ΔR[t, a] within constraints that *minimize*
strategy PnL (or maximize loss) over a horizon.

Why this is useful
------------------
- Finds worst-case paths under realistic bounds (per-asset caps, Lp budgets, smoothness)
- Helps identify fragility to specific assets, times, or factor mixes
- Can produce closed-form "box" worst cases, or optimized (PGD / SPSA) adversaries

Inputs
------
--returns RETURNS.csv         Wide CSV Date x Asset of base returns (decimal)
--positions POS.csv           Wide CSV Date x Asset of positions (use t-1 positions vs t returns)
                              If omitted, equal-weight (rebalanced each step) is used.
--alpha 0.05                  Optional: optimize tail (ES at alpha) instead of mean PnL
--method box|pgd|spsa         Adversary construction method
--norm linf|l2|l1             Norm constraint for ΔR (per step)
--eps 0.02                    Per-step per-asset magnitude cap (for L∞); or norm budget for L1/L2
--smooth-lambda 0.0           Temporal smoothness penalty λ * sum_t ||ΔR_t - ΔR_{t-1}||_2^2  (PGD/SPSA)
--steps -1                    Limit to first N steps (useful for speed; -1 => use all)
--seed 123                    RNG seed (SPSA)
--outdir out_adv              Output directory

Outputs
-------
- adv_returns.parquet   : adversarial ΔR[t, a] panel
- stressed.parquet      : (R + ΔR) per-step stressed returns
- pnl_timeseries.csv    : base and stressed PnL by date
- summary.json          : headline metrics (total/base PnL, stressed PnL, ES improvement, budgets, etc.)
- contribution.csv      : worst-case contribution by (t, asset)

Notes
-----
* Objective (default mean PnL):
    PnL = Σ_t Σ_a W[t-1,a] * (R[t,a] + ΔR[t,a])
  Adversary minimizes the second term under constraints. With L∞ and no smoothness, closed form exists.
* Tail objective (if --alpha provided): minimize ES_α of per-step PnL distribution (uses simple plug-in).
* Positions are interpreted as *held over the coming step* (t-1 applied to R_t). If you pass weights that sum
  to 1, units are return contributions; if you pass notional positions, units are currency PnL.

Examples
--------
# Worst-case box (closed form) with ±2% caps per asset-step
python adversarial_gen.py --returns ret.csv --positions pos.csv --method box --norm linf --eps 0.02

# Smooth PGD adversary with L2 budget 5% per step and temporal smoothness
python adversarial_gen.py --returns ret.csv --positions pos.csv --method pgd --norm l2 --eps 0.05 --smooth-lambda 10.0

# Black-box SPSA adversary to minimize ES(5%), L1 budget 5% per step
python adversarial_gen.py --returns ret.csv --positions pos.csv --method spsa --norm l1 --eps 0.05 --alpha 0.05 --seed 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# I/O helpers
# -----------------------------
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    c0 = df.columns[0].lower()
    if c0 in ("date", "time", "t"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    return df.sort_index()


def align_returns_positions(R: pd.DataFrame, W: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assets = list(R.columns)
    if W is None:
        # Equal weights each step
        W = pd.DataFrame(1.0 / len(assets), index=R.index, columns=assets)
    else:
        W = W.reindex(index=R.index, columns=assets).fillna(0.0)
    # Positions at t-1 applied to returns at t -> shift W by 1 forward (i.e., lag positions)
    W_shift = W.shift(1).fillna(0.0)
    return R, W_shift


# -----------------------------
# Objectives
# -----------------------------
def pnl_series(R: np.ndarray, Wlag: np.ndarray) -> np.ndarray:
    # per-step PnL = sum_a W_{t-1,a} * R_{t,a}
    return np.einsum("ta,ta->t", Wlag, R)


def es_alpha(x: np.ndarray, alpha: float) -> float:
    # one-sided lower tail ES
    q = np.quantile(x, alpha)
    mask = x <= q
    return float(np.mean(x[mask])) if mask.any() else float(q)


class Objective:
    def __init__(self, Wlag: np.ndarray, alpha: Optional[float] = None, smooth_lambda: float = 0.0):
        self.Wlag = Wlag  # (T, A)
        self.alpha = alpha
        self.smooth = smooth_lambda

    def value(self, base_R: np.ndarray, dR: np.ndarray) -> float:
        # Minimize mean PnL (or ES) of stressed returns plus smoothness penalty
        stressed = base_R + dR
        p = pnl_series(stressed, self.Wlag)
        primary = es_alpha(p, self.alpha) if (self.alpha is not None and self.alpha > 0) else float(np.mean(p))
        if self.smooth > 0:
            diff = np.diff(dR, axis=0)
            pen = self.smooth * float(np.sum(diff * diff))
        else:
            pen = 0.0
        return primary + pen

    def grad_mean(self, base_R: np.ndarray, dR: np.ndarray) -> np.ndarray:
        # gradient of mean PnL wrt dR is simply Wlag (broadcast across assets)
        g = self.Wlag.copy()  # (T, A)
        if self.smooth > 0:
            # ∂/∂dR: λ * Σ_t ||dR_t - dR_{t-1}||^2 -> 2λ*(2dR_t - dR_{t-1} - dR_{t+1}) with end corrections
            diff = np.zeros_like(dR)
            diff[0] = 2 * dR[0] - dR[1]
            diff[1:-1] = 2 * dR[1:-1] - dR[:-2] - dR[2:]
            diff[-1] = 2 * dR[-1] - dR[-2]
            g = g + 2.0 * self.smooth * diff
        return g

    def grad(self, base_R: np.ndarray, dR: np.ndarray) -> np.ndarray:
        if self.alpha is None or self.alpha <= 0:
            return self.grad_mean(base_R, dR)
        # Subgradient for ES: gradient equals Wlag for samples in the tail set, else 0 (plug-in)
        stressed = base_R + dR
        p = pnl_series(stressed, self.Wlag)
        q = np.quantile(p, self.alpha)
        mask = (p <= q).astype(float)[:, None]  # (T,1)
        g = mask * self.Wlag
        if self.smooth > 0:
            diff = np.zeros_like(dR)
            diff[0] = 2 * dR[0] - dR[1]
            diff[1:-1] = 2 * dR[1:-1] - dR[:-2] - dR[2:]
            diff[-1] = 2 * dR[-1] - dR[-2]
            g = g + 2.0 * self.smooth * diff
        return g


# -----------------------------
# Closed-form L∞ box adversary
# -----------------------------
def adversary_box(Wlag: np.ndarray, eps: float) -> np.ndarray:
    """
    With L∞ per-asset per-step bound: |ΔR_{t,a}| ≤ eps, and no smoothness,
    the worst-case against mean PnL is ΔR_{t,a} = -eps * sign(Wlag_{t,a}).
    For ES objective, using the same closed-form surrogate works well in practice.
    """
    return -eps * np.sign(Wlag)


# -----------------------------
# Projected Gradient Descent (PGD)
# -----------------------------
def project_norm(dR: np.ndarray, norm: str, eps: float) -> np.ndarray:
    if norm == "linf":
        return np.clip(dR, -eps, eps)
    elif norm == "l2":
        # per-time projection onto L2 ball radius eps
        norms = np.linalg.norm(dR, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, eps / norms)
        return dR * scale
    elif norm == "l1":
        # per-time projection onto L1 ball radius eps (Duchi et al.)
        X = dR.copy()
        T, A = X.shape
        for t in range(T):
            v = X[t]
            u = np.abs(v)
            if u.sum() <= eps:
                continue
            # sort
            s = np.sort(u)[::-1]
            cssv = np.cumsum(s)
            rho = np.nonzero(s * np.arange(1, A + 1) > (cssv - eps))[0][-1]
            theta = (cssv[rho] - eps) / (rho + 1.0)
            X[t] = np.sign(v) * np.maximum(u - theta, 0.0)
        return X
    else:
        raise ValueError("Unknown norm for projection.")


def adversary_pgd(
    R: np.ndarray, Wlag: np.ndarray, norm: str, eps: float, smooth_lambda: float, alpha: Optional[float],
    lr: float = 0.1, iters: int = 300,
) -> np.ndarray:
    obj = Objective(Wlag, alpha=alpha, smooth_lambda=smooth_lambda)
    T, A = R.shape
    dR = np.zeros((T, A), dtype=float)
    for k in range(iters):
        g = obj.grad(R, dR)
        dR = dR - lr * g
        dR = project_norm(dR, norm, eps)
    return dR


# -----------------------------
# SPSA (black-box)
# -----------------------------
def adversary_spsa(
    R: np.ndarray, Wlag: np.ndarray, norm: str, eps: float, smooth_lambda: float, alpha: Optional[float],
    iters: int = 400, a0: float = 0.2, c0: float = 0.1, seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obj = Objective(Wlag, alpha=alpha, smooth_lambda=smooth_lambda)
    T, A = R.shape
    dR = np.zeros((T, A), dtype=float)

    def proj(x):
        return project_norm(x, norm, eps)

    for k in range(1, iters + 1):
        ak = a0 / (k ** 0.602)   # standard SPSA decay
        ck = c0 / (k ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=(T, A))
        dR_plus = proj(dR + ck * delta)
        dR_minus = proj(dR - ck * delta)
        f_plus = obj.value(R, dR_plus)
        f_minus = obj.value(R, dR_minus)
        ghat = (f_plus - f_minus) / (2 * ck) * delta
        dR = proj(dR - ak * ghat)
    return dR


# -----------------------------
# Orchestration / CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Adversarial return generator for strategy stress testing")
    ap.add_argument("--returns", required=True, help="Wide CSV Date x Asset of returns (decimals)")
    ap.add_argument("--positions", default="", help="Wide CSV Date x Asset of positions/weights (applied at t-1)")
    ap.add_argument("--alpha", type=float, default=0.0, help="Tail ES alpha (0 => mean PnL objective)")
    ap.add_argument("--method", choices=["box", "pgd", "spsa"], default="box")
    ap.add_argument("--norm", choices=["linf", "l2", "l1"], default="linf")
    ap.add_argument("--eps", type=float, default=0.02, help="Per-step constraint: L∞ bound or L1/L2 radius")
    ap.add_argument("--smooth-lambda", type=float, default=0.0, help="Temporal smoothness penalty (PGD/SPSA)")
    ap.add_argument("--steps", type=int, default=-1, help="Use first N steps (-1 => all)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", default="out_adv")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    R_df = read_wide_csv(args.returns)
    W_df = read_wide_csv(args.positions) if args.positions else None
    R_df, W_df = align_returns_positions(R_df, W_df)

    if args.steps and args.steps > 0:
        R_df = R_df.iloc[:args.steps]
        W_df = W_df.iloc[:args.steps]

    assets = list(R_df.columns)
    dates = R_df.index

    R = R_df.values.astype(float)               # (T, A)
    Wlag = W_df.values.astype(float)           # (T, A)

    # Base metrics
    base_pnl_ts = pnl_series(R, Wlag)
    base_mean = float(np.mean(base_pnl_ts))
    base_es = es_alpha(base_pnl_ts, args.alpha) if args.alpha and args.alpha > 0 else None

    # Adversary
    if args.method == "box" and args.norm == "linf" and args.smooth_lambda == 0.0:
        dR = adversary_box(Wlag, eps=args.eps)
    elif args.method == "pgd":
        dR = adversary_pgd(R, Wlag, norm=args.norm, eps=args.eps,
                           smooth_lambda=args.smooth_lambda, alpha=(args.alpha if args.alpha > 0 else None),
                           lr=0.1, iters=400)
    elif args.method == "spsa":
        dR = adversary_spsa(R, Wlag, norm=args.norm, eps=args.eps,
                            smooth_lambda=args.smooth_lambda, alpha=(args.alpha if args.alpha > 0 else None),
                            iters=500, a0=0.3, c0=0.1, seed=args.seed)
    else:
        # fallback: project box if user chose incompatible combination
        dR = project_norm(-np.sign(Wlag) * args.eps, args.norm, args.eps)

    # Stressed outcomes
    stressed = R + dR
    stressed_pnl_ts = pnl_series(stressed, Wlag)
    stressed_mean = float(np.mean(stressed_pnl_ts))
    stressed_es = es_alpha(stressed_pnl_ts, args.alpha) if args.alpha and args.alpha > 0 else None

    # Contributions (most damaging (t, asset) under adversary)
    contrib = Wlag * dR  # (T, A)
    contrib_df = pd.DataFrame(contrib, index=dates, columns=assets)
    contrib_long = contrib_df.stack().rename("pnl_contrib").reset_index().rename(columns={"level_0": "Date", "level_1": "Asset"})
    contrib_long = contrib_long.sort_values("pnl_contrib")  # most negative first
    contrib_long.to_csv(outdir / "contribution.csv", index=False)

    # Save ΔR and stressed returns
    idx = pd.MultiIndex.from_product([dates, assets], names=["Date", "Asset"])
    dR_df = pd.DataFrame(dR.reshape(-1), index=idx, columns=["dR"]).unstack("Asset")
    dR_df.columns = dR_df.columns.droplevel(0)
    dR_df.to_parquet(outdir / "adv_returns.parquet")

    stressed_df = pd.DataFrame(stressed, index=dates, columns=assets)
    stressed_df.to_parquet(outdir / "stressed.parquet")

    # PnL time series
    pnl_df = pd.DataFrame(
        {"base_pnl": base_pnl_ts, "stressed_pnl": stressed_pnl_ts},
        index=dates,
    )
    pnl_df.to_csv(outdir / "pnl_timeseries.csv")

    # Summary
    summary = {
        "method": args.method,
        "norm": args.norm,
        "eps": float(args.eps),
        "smooth_lambda": float(args.smooth_lambda),
        "alpha": float(args.alpha) if args.alpha and args.alpha > 0 else None,
        "T": int(R.shape[0]),
        "A": int(R.shape[1]),
        "base_mean_pnl": base_mean,
        "stressed_mean_pnl": stressed_mean,
        "delta_mean_pnl": stressed_mean - base_mean,
        "base_es": base_es,
        "stressed_es": stressed_es,
        "delta_es": (None if (base_es is None or stressed_es is None) else (stressed_es - base_es)),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Console print
    print("== Adversarial Scenario Complete ==")
    for k, v in summary.items():
        if isinstance(v, float) or v is None or isinstance(v, int):
            print(f"{k:>20}: {v}")

if __name__ == "__main__":
    main()
