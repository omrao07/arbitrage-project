#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
volatility_spillover.py — Diebold–Yilmaz (generalized FEVD) spillover engine
-----------------------------------------------------------------------------

What this does
==============
Given price or return series (wide or long), this script:

1) Cleans data → log returns
2) Fits a VAR(p) (OLS) on a rolling window
3) Computes Generalized Forecast Error Variance Decomposition (GFEVD, Pesaran–Shin)
4) Builds Diebold–Yilmaz spillover tables:
   - Θ (H-step GFEVD, rows normalize to 1)
   - Total Spillover Index (TSI)
   - Directional FROM_i (row off-diagonal), TO_j (column off-diagonal), NET_j
   - Pairwise NET_ij = Θ_ij − Θ_ji
5) Writes full-sample (static) results and rolling time-varying indices

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--data data.csv     REQUIRED
  Wide:  date, AAPL, MSFT, ...  (prices or returns)
  Long:  date, ticker, value     (prices or returns)

--is_return         If provided, treat `value`/columns as returns already

Optional
--------
--tickers AAPL,MSFT,GOOG     Subset universe
--start  2015-01-01          Date filter
--end    2025-09-01
--p 2                         VAR lag order
--h 10                        Forecast horizon for FEVD
--window 250                  Rolling window length (set 0 to skip rolling)
--standardize                 Z-score returns within each window (scale-robustness)
--outdir out_spillover        Output folder

Outputs
-------
- returns_panel.csv           Clean T×N return matrix
- fevd_static.csv             Full-sample Θ (row-normalized), in %
- directional_static.csv      FROM/TO/NET (in %)
- pairwise_net_static.csv     NET_ij matrix (in %)
- rolling_indices.csv         Time series: TSI, FROM_i, TO_i, NET_i per window end
- summary.json                Latest TSI and top transmitters/receivers
- config.json                 Run configuration

Method notes
------------
• VAR(p) via multivariate OLS (one-step; intercept included but excluded from MA recursion)
• MA(∞) up to horizon H: Φ_0=I, Φ_s=Σ_{i=1..p} Φ_{s−i} A_i
• GFEVD θ_{i←j}(H) = (σ_jj^{-1} Σ_{s=0}^{H−1} [e_i' Φ_s Σ e_j]^2) / (Σ_{s=0}^{H−1} e_i' Φ_s Σ Φ_s' e_i)
  Row-normalize Θ so each row sums to 1 (Diebold–Yilmaz normalization).
• Total Spillover Index (TSI) = 100 × (sum of off-diagonal elements) / N
• FROM_i = 100 × (1 − Θ_ii); TO_j = 100 × (column sum off-diagonals); NET_j = TO_j − FROM_j

DISCLAIMER: Research tooling; validate before production use.
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

def dlog(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan).astype(float)).diff()

def zscore_cols(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean(axis=0)
    sd = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - m) / sd

def safe_nan_to_num(x: np.ndarray, repl: float=0.0) -> np.ndarray:
    y = np.array(x, dtype=float)
    y[~np.isfinite(y)] = repl
    return y

# ----------------------------- data loading -----------------------------

def load_returns(path: str, is_return: bool=False, tickers: Optional[List[str]]=None,
                 start: Optional[str]=None, end: Optional[str]=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={date_c: "date"})
    df["date"] = to_date(df["date"])
    if ncol(df, "ticker") and ncol(df, "value"):
        df = df.rename(columns={ncol(df,"ticker"): "ticker", ncol(df,"value"): "value"})
        if tickers:
            df = df[df["ticker"].astype(str).str.upper().isin([t.upper() for t in tickers])]
        if is_return:
            piv = df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
        else:
            piv = (df.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
                     .apply(dlog))
    else:
        # wide
        piv = df.set_index("date").sort_index()
        if tickers:
            keep = [c for c in piv.columns if str(c).upper() in [t.upper() for t in tickers]]
            piv = piv[keep]
        piv = piv.apply(lambda s: s if is_return else dlog(s))
    # filter dates
    if start:
        piv = piv[piv.index >= pd.to_datetime(start)]
    if end:
        piv = piv[piv.index <= pd.to_datetime(end)]
    # drop all-NaN cols & rows
    piv = piv.dropna(axis=1, how="all").dropna(how="all")
    # forward-fill small gaps (optional mild)
    piv = piv.fillna(method="ffill", limit=2).dropna()
    piv.index.name = "date"
    # enforce column order & names
    piv.columns = [str(c).upper() for c in piv.columns]
    return piv

# ----------------------------- VAR & FEVD -----------------------------

def var_fit(Y: np.ndarray, p: int) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Fit VAR(p) via OLS with intercept.
    Y: T×N
    Returns (A_list [A1..Ap], c (N,), Sigma_u (N×N))
    """
    T, N = Y.shape
    if T <= p + N + 1:
        raise ValueError("Not enough observations for VAR.")
    # Build lagged design
    X = []
    for i in range(1, p+1):
        X.append(Y[p-i:T-i, :])
    X = np.concatenate(X, axis=1)  # (T-p) × (N*p)
    X = np.column_stack([np.ones((T-p, 1)), X])  # intercept
    Yt = Y[p:, :]  # (T-p) × N
    # OLS: B = (X'X)^-1 X'Y
    XtX = X.T @ X
    XtY = X.T @ Yt
    B = np.linalg.pinv(XtX) @ XtY   # (1+Np) × N
    c = B[0, :]                     # intercept (N,)
    A_flat = B[1:, :]               # (Np) × N
    # Reshape into A_i (N×N) with convention y_t = sum A_i y_{t-i} + c + u_t
    A_list = []
    for i in range(p):
        Ai = A_flat[i*N:(i+1)*N, :].T  # transpose: columns per equation
        A_list.append(Ai)              # N×N
    # Residuals & Sigma
    U = Yt - X @ B
    dof = max(1, (T - p) - (N*p + 1))
    Sigma = (U.T @ U) / dof
    return A_list, c, Sigma

def ma_coeffs(A_list: List[np.ndarray], H: int) -> List[np.ndarray]:
    """
    Compute MA coefficients Φ_0..Φ_{H-1} for VAR(p): y_t = Σ A_i y_{t-i} + u_t.
    Recursion: Φ_0=I; Φ_s = Σ_{i=1..p} Φ_{s-i} A_i.
    """
    N = A_list[0].shape[0]
    p = len(A_list)
    Phi = [np.eye(N)]
    for s in range(1, H):
        acc = np.zeros((N, N))
        for i in range(1, p+1):
            if s - i < 0:
                continue
            acc += Phi[s-i] @ A_list[i-1]
        Phi.append(acc)
    return Phi

def gfevd(A_list: List[np.ndarray], Sigma: np.ndarray, H: int) -> np.ndarray:
    """
    Generalized FEVD matrix Θ (N×N) with rows summing to 1.
    Θ[i,j] = share of i's H-step FE variance due to shock in j.
    """
    N = Sigma.shape[0]
    Phi = ma_coeffs(A_list, H)
    Sigma = (Sigma + Sigma.T) / 2.0  # symmetrize
    sig_diag = np.diag(Sigma).copy()
    # Denominator terms per i
    denom = np.zeros(N)
    for h in range(H):
        F = Phi[h] @ Sigma @ Phi[h].T
        denom += np.diag(F)
    # Numerators
    theta = np.zeros((N, N))
    for j in range(N):
        e_j = np.zeros(N); e_j[j] = 1.0
        for i in range(N):
            num = 0.0
            for h in range(H):
                v = (Phi[h] @ Sigma @ e_j)[i]
                num += (v ** 2)
            theta[i, j] = num / (sig_diag[j] if sig_diag[j] != 0 else 1e-12)
    # Normalize rows
    row_sums = theta.sum(axis=1, keepdims=True)
    theta_normalized = theta / np.where(row_sums==0, 1.0, row_sums)
    return safe_nan_to_num(theta_normalized, 0.0)

def spillover_measures(Theta: np.ndarray, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Theta: N×N row-normalized (rows sum to 1). Returns:
      Θ% table, directional (FROM/TO/NET), pairwise NET_ij, TSI.
    """
    N = Theta.shape[0]
    Theta_pct = 100.0 * Theta
    # FROM_i (row off-diagonal)
    FROM = Theta_pct.sum(axis=1) - np.diag(Theta_pct)
    # TO_j (column off-diagonal)
    TO = Theta_pct.sum(axis=0) - np.diag(Theta_pct)
    NET = TO - FROM
    # Pairwise net
    NET_ij = Theta_pct - Theta_pct.T
    # TSI (total spillover index)
    TSI = float(Theta_pct.sum() - np.trace(Theta_pct)) / N
    # DataFrames
    Theta_df = pd.DataFrame(Theta_pct, index=cols, columns=cols)
    directional = pd.DataFrame({"FROM": FROM, "TO": TO, "NET": NET}, index=cols).reset_index().rename(columns={"index":"name"})
    pairwise = pd.DataFrame(NET_ij, index=cols, columns=cols)
    return Theta_df, directional, pairwise, TSI

# ----------------------------- rolling engine -----------------------------

def rolling_spillovers(R: pd.DataFrame, p: int, H: int, window: int, standardize: bool) -> pd.DataFrame:
    """
    Returns tidy time series with TSI and directional FROM/TO/NET per window end date.
    """
    idx = R.index
    cols = R.columns.tolist()
    rows = []
    for t_end in range(window, len(idx)+1):
        sub = R.iloc[t_end-window:t_end, :].copy()
        if standardize:
            sub = zscore_cols(sub)
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
            if sub.shape[0] <= p + len(cols) + 1:
                continue
        Y = sub.values
        try:
            A_list, _, Sigma = var_fit(Y, p=p)
            Theta = gfevd(A_list, Sigma, H=H)
        except Exception:
            continue
        Theta_df, directional, pairwise, TSI = spillover_measures(Theta, cols)
        date = idx[t_end-1]
        # Store summary row
        row = {"date": date, "TSI": TSI}
        # Add directional per name
        for _, r in directional.iterrows():
            nm = r["name"]
            row[f"FROM_{nm}"] = float(r["FROM"])
            row[f"TO_{nm}"]   = float(r["TO"])
            row[f"NET_{nm}"]  = float(r["NET"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date")

# ----------------------------- Orchestration -----------------------------

@dataclass
class Config:
    data: str
    is_return: bool
    tickers: Optional[str]
    start: Optional[str]
    end: Optional[str]
    p: int
    h: int
    window: int
    standardize: bool
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Volatility Spillover (Diebold–Yilmaz, generalized FEVD)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--is_return", action="store_true", help="Treat input values as returns already")
    ap.add_argument("--tickers", default="", help="Comma list to subset (e.g., AAPL,MSFT,GOOG)")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--p", type=int, default=2, help="VAR lag order")
    ap.add_argument("--h", type=int, default=10, help="FEVD forecast horizon")
    ap.add_argument("--window", type=int, default=250, help="Rolling window length (0 to skip)")
    ap.add_argument("--standardize", action="store_true", help="Z-score returns within each window")
    ap.add_argument("--outdir", default="out_spillover")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None

    # Load & prepare returns
    R = load_returns(args.data, is_return=args.is_return, tickers=tickers,
                     start=(args.start or None), end=(args.end or None))

    if R.shape[1] < 2:
        raise ValueError("Need at least 2 series to compute spillovers.")
    R.to_csv(outdir / "returns_panel.csv", index=True)

    # Full-sample (static)
    R_static = zscore_cols(R) if args.standardize else R
    Y = R_static.values
    A_list, _, Sigma = var_fit(Y, p=args.p)
    Theta = gfevd(A_list, Sigma, H=args.h)
    Theta_df, directional_df, pairwise_df, TSI = spillover_measures(Theta, R.columns.tolist())

    Theta_df.to_csv(outdir / "fevd_static.csv", index=True)
    directional_df.to_csv(outdir / "directional_static.csv", index=False)
    pairwise_df.to_csv(outdir / "pairwise_net_static.csv", index=True)

    # Rolling
    rolling_df = pd.DataFrame()
    if args.window and args.window > (args.p + R.shape[1] + 2):
        rolling_df = rolling_spillovers(R, p=args.p, H=args.h, window=args.window, standardize=args.standardize)
        if not rolling_df.empty:
            rolling_df.to_csv(outdir / "rolling_indices.csv", index=False)

    # Summary
    top_tx, top_rx = None, None
    try:
        top_tx = directional_df.sort_values("TO", ascending=False).iloc[0]["name"]
        top_rx = directional_df.sort_values("FROM", ascending=False).iloc[0]["name"]
    except Exception:
        pass
    summary = {
        "n_series": int(R.shape[1]),
        "var_lag_p": int(args.p),
        "fevd_horizon": int(args.h),
        "TSI_static": float(TSI),
        "top_transmitter_TO": top_tx,
        "top_receiver_FROM": top_rx,
        "rolling_available": bool(not rolling_df.empty),
        "last_rolling_TSI": float(rolling_df["TSI"].iloc[-1]) if not rolling_df.empty else None
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config dump
    cfg = asdict(Config(
        data=args.data, is_return=bool(args.is_return), tickers=(args.tickers or None),
        start=(args.start or None), end=(args.end or None), p=int(args.p), h=int(args.h),
        window=int(args.window), standardize=bool(args.standardize), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Volatility Spillover (Diebold–Yilmaz) ==")
    print(f"N={R.shape[1]} | VAR p={args.p} | H={args.h} | TSI (static)={TSI:.2f}%")
    if not rolling_df.empty:
        print(f"Rolling window {args.window}: last TSI={rolling_df['TSI'].iloc[-1]:.2f}%  (rows: {rolling_df.shape[0]})")
    print("Outputs in:", outdir.resolve())

if __name__ == "__main__":
    main()
