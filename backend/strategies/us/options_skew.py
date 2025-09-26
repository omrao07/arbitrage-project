#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# options_skew.py
#
# Options skew & smile toolkit
# ---------------------------
# - Pulls an options chain from yfinance for a given ticker/expiry OR loads your CSV
# - Computes mid prices, Black–Scholes implied vols (Newton), and Greeks (vega)
# - Maps quotes to deltas and moneyness; builds 25Δ risk-reversal & butterfly
# - Estimates ATM skew (dσ/dK and dσ/dlnK), smile curvature via quadratic fit in log-moneyness
# - Optional: fetch multiple expiries to compare term skew
# - Exports tidy CSVs and plots (smile, IV vs delta, term structure)
#
# Usage examples
# --------------
# 1) Live chain from Yahoo for nearest monthly:
#    python options_skew.py --ticker SPY --dte-target 30 --plot
#
# 2) Specific expiry (YYYY-MM-DD) from Yahoo:
#    python options_skew.py --ticker TSLA --expiry 2025-10-17 --plot
#
# 3) Multiple expiries (nearest 5):
#    python options_skew.py --ticker NVDA --n-expiries 5 --plot
#
# 4) Use your own CSV (any mix of calls/puts in one file):
#    python options_skew.py --csv chain.csv --underlier 123.45 --rate 0.045 --div 0.005 --plot
#
# CSV expected columns (case-insensitive; extra columns preserved):
#   type, strike, last, bid, ask, mid, expiry, underlyingPrice (optional), dte (optional)
#   Notes: 'type' ∈ {'C','P','CALL','PUT'}. If 'mid' missing, computed from (bid+ask)/2
#
# Outputs (./artifacts/options_skew/*)
# -----------------------------------
#   chain_raw.csv               (raw pulls merged)
#   chain_iv.csv                (per-option IVs & deltas)
#   smile_metrics.csv           (per-expiry metrics incl. RR25 and BF25)
#   term_skew.csv               (ATM skew vs DTE, if multiple expiries)
#   plots/*.png                 (if --plot)
#
# Dependencies
# ------------
# pip install pandas numpy yfinance matplotlib python-dateutil

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from dateutil import parser as dtp
from datetime import datetime, timezone


# --------------------- Config & IO helpers ---------------------

@dataclass
class Config:
    ticker: Optional[str]
    expiry: Optional[str]
    dte_target: Optional[int]
    n_expiries: int
    csv_path: Optional[str]
    underlier: Optional[float]
    rate: float
    div: float
    use_last: bool
    plot: bool
    outdir: str


def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "options_skew_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def _lower(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _coerce_num(x):
    if pd.isna(x): return np.nan
    try:
        return float(str(x).replace(",", "").replace("_", ""))
    except Exception:
        return np.nan


# --------------------- Black–Scholes utils ---------------------

from math import log, sqrt, exp
from math import erf

def _norm_cdf(x: float) -> float:
    # Φ(x) via error function
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)

def bs_price(S, K, T, r, q, vol, cp: str) -> float:
    """Black–Scholes (dividends via continuous yield q)."""
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        # intrinsic (very rough) fallback
        intrinsic = max(0.0, S - K) if cp.upper().startswith("C") else max(0.0, K - S)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if cp.upper().startswith("C"):
        return S * np.exp(-q * T) * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * _norm_cdf(-d2) - S * np.exp(-q * T) * _norm_cdf(-d1)

def bs_vega(S, K, T, r, q, vol) -> float:
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    return S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)

def bs_delta(S, K, T, r, q, vol, cp: str) -> float:
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    if cp.upper().startswith("C"):
        return np.exp(-q * T) * _norm_cdf(d1)
    else:
        return -np.exp(-q * T) * _norm_cdf(-d1)

def implied_vol_newton(price, S, K, T, r, q, cp: str, v_init=0.25, tol=1e-6, max_iter=100) -> float:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    v = max(1e-4, float(v_init))
    for _ in range(max_iter):
        model = bs_price(S, K, T, r, q, v, cp)
        diff = model - price
        if abs(diff) < tol:
            return float(v)
        veg = bs_vega(S, K, T, r, q, v)
        if veg <= 1e-12:
            # small vega — nudge vol
            v *= 1.1
            continue
        v -= diff / veg
        # clamps
        if v < 1e-6: v = 1e-6
        if v > 5.0: v = 5.0
    return float(v)  # return last iterate even if not fully converged


# --------------------- Data ingestion ---------------------

def load_chain_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [_lower(c) for c in df.columns]
    # Required basics
    need = {"type","strike"}
    if not need.issubset(df.columns):
        raise SystemExit("CSV must include at least: type (call/put), strike; bid/ask or mid or last are recommended.")
    # numerics
    for c in ["strike","last","bid","ask","mid","underlyingprice","dte"]:
        if c in df.columns:
            df[c] = df[c].apply(_coerce_num)
    # expiry
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"])
    else:
        df["expiry"] = pd.NaT
    # normalize type
    df["type"] = df["type"].astype(str).str.upper().str[0]
    # mid
    if "mid" not in df.columns or df["mid"].isna().all():
        df["mid"] = (df.get("bid", np.nan) + df.get("ask", np.nan)) / 2.0
    # drop empties
    df = df.dropna(subset=["strike","mid"], how="any")
    return df


def pick_expiries_yf(ticker: str, n: int, target_expiry: Optional[str], dte_target: Optional[int]) -> List[str]:
    tk = yf.Ticker(ticker)
    all_expiries = tk.options or []
    if not all_expiries:
        raise SystemExit("No expiries returned by Yahoo for this ticker.")
    if target_expiry:
        exp = dtp.parse(target_expiry).date().isoformat()
        if exp in all_expiries:
            return [exp]
        else:
            # if not exact, choose nearest
            dts = pd.to_datetime(all_expiries)
            tgt = pd.Timestamp(exp)
            best = dts.iloc[(dts - tgt).abs().argmin()].date().isoformat()
            return [best]
    if dte_target is not None:
        dts = pd.to_datetime(all_expiries)
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        dte = (dts - today).days
        ix = np.argmin(np.abs(dte - dte_target))
        return [dts[ix].date().isoformat()]
    # default: first n expiries
    return all_expiries[:max(1, n)]


def load_chain_from_yf(ticker: str, expiries: List[str]) -> Tuple[pd.DataFrame, float]:
    tk = yf.Ticker(ticker)
    spot = None
    try:
        spot = float(tk.fast_info.get("last_price") or tk.fast_info.get("lastPrice") or tk.history(period="1d")["Close"][-1])
    except Exception:
        # fallback
        spot = float(tk.history(period="1d")["Close"][-1])
    frames = []
    for exp in expiries:
        try:
            opt = tk.option_chain(exp)
        except Exception:
            continue
        for typ, tab in [("C", opt.calls), ("P", opt.puts)]:
            if tab is None or tab.empty: continue
            sub = tab.rename(columns=str.lower).copy()
            # yfinance usually provides 'lastPrice','bid','ask','strike' and 'impliedVolatility' (but we recompute)
            sub["type"] = typ
            sub["expiry"] = pd.Timestamp(exp)
            # mid
            sub["mid"] = (pd.to_numeric(sub.get("bid", np.nan), errors="coerce") + pd.to_numeric(sub.get("ask", np.nan), errors="coerce")) / 2.0
            if "lastprice" in sub.columns:
                sub["last"] = pd.to_numeric(sub["lastprice"], errors="coerce")
            frames.append(sub[["type","strike","last","bid","ask","mid","expiry"]])
    if not frames:
        raise SystemExit("No options data returned by Yahoo. Try a different ticker/expiry.")
    df = pd.concat(frames, ignore_index=True)
    return df, float(spot)


# --------------------- Skew & smile metrics ---------------------

def compute_iv_table(df: pd.DataFrame, S: float, r: float, q: float, price_kind: str = "mid") -> pd.DataFrame:
    d = df.copy()
    d["price"] = d[price_kind].where(d[price_kind].notna(), d.get("last", np.nan))
    d = d.dropna(subset=["price", "strike"])
    # time to expiry in years
    if "expiry" in d.columns and d["expiry"].notna().any():
        now = pd.Timestamp.utcnow().normalize()
        d["T"] = (pd.to_datetime(d["expiry"]) - now).dt.days / 365.0
    elif "dte" in d.columns:
        d["T"] = d["dte"].astype(float) / 365.0
    else:
        raise SystemExit("Cannot determine time to expiry. Provide 'expiry' or 'dte' in CSV, or use Yahoo loader.")
    # clamp
    d = d[(d["T"] > 0) & (d["price"] > 0) & (d["strike"] > 0)]
    # IV via Newton
    vols = []
    deltas = []
    for _, r0 in d.iterrows():
        cp = r0["type"]
        K = float(r0["strike"]); T = float(r0["T"]); P = float(r0["price"])
        iv = implied_vol_newton(P, S, K, T, r, q, cp, v_init=0.25)
        vols.append(iv)
        deltas.append(bs_delta(S, K, T, r, q, iv, cp) if np.isfinite(iv) else np.nan)
    d["iv"] = vols
    d["delta"] = deltas
    d["moneyness"] = d["strike"] / S
    d["ln_moneyness"] = np.log(d["moneyness"])
    return d.dropna(subset=["iv"])


def find_delta_strike(tbl: pd.DataFrame, target_delta: float, call: bool) -> Optional[float]:
    """Find strike with delta closest to target (absolute for puts negative)."""
    if call:
        sub = tbl[tbl["type"] == "C"].copy()
        if sub.empty: return None
        sub["err"] = (sub["delta"] - target_delta).abs()
    else:
        sub = tbl[tbl["type"] == "P"].copy()
        if sub.empty: return None
        sub["err"] = (sub["delta"] - (-target_delta)).abs()
    best = sub.iloc[sub["err"].argmin()] if not sub.empty else None
    return float(best["strike"]) if best is not None else None


def interpolate_iv_at_strike(tbl: pd.DataFrame, K: float, cp: str) -> Optional[float]:
    sub = tbl[(tbl["type"] == cp) & np.isfinite(tbl["iv"])].sort_values("strike")
    if sub.empty: return None
    x, y = sub["strike"].values, sub["iv"].values
    return float(np.interp(K, x, y))


def smile_metrics_for_expiry(tbl: pd.DataFrame, S: float) -> dict:
    # ATM proxy: strike closest to S
    atm_idx = (tbl["strike"] - S).abs().idxmin()
    atm_iv = float(tbl.loc[atm_idx, "iv"])
    # regress IV ~ a + b*lnK + c*(lnK)^2 around ATM window
    sub = tbl.copy()
    # focus on |lnM| <= 0.2 (≈ ±20% moneyness)
    sub = sub[sub["ln_moneyness"].between(-0.25, 0.25)]
    X = np.vstack([np.ones(len(sub)), sub["ln_moneyness"].values, (sub["ln_moneyness"].values**2)]).T
    y = sub["iv"].values
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = beta
    except Exception:
        a, b, c = np.nan, np.nan, np.nan

    # ATM slope approximations (dσ/dK and dσ/dlnK at lnM=0)
    # dσ/dlnK at ATM ≈ b (since derivative of a + b*x + c*x^2 at x=0 is b)
    d_sigma_d_lnK = float(b) if np.isfinite(b) else np.nan
    # dσ/dK = (1/K) * dσ/dlnK at ATM, with K≈S
    d_sigma_d_K = (d_sigma_d_lnK / S) if np.isfinite(d_sigma_d_lnK) and S>0 else np.nan

    # 25Δ RR and BF: find K_call(25Δ) and K_put(25Δ)
    # Many markets define 25Δ put (abs 0.25) and 25Δ call (+0.25)
    Kc = find_delta_strike(tbl, 0.25, call=True)
    Kp = find_delta_strike(tbl, 0.25, call=False)
    rr25 = bf25 = np.nan
    if Kc and Kp:
        ivc = interpolate_iv_at_strike(tbl, Kc, "C")
        ivp = interpolate_iv_at_strike(tbl, Kp, "P")
        if ivc and ivp:
            rr25 = ivc - ivp
            bf25 = 0.5 * (ivc + ivp) - atm_iv

    return {
        "atm_iv": atm_iv,
        "atm_slope_dsigma_dK": d_sigma_d_K,
        "atm_slope_dsigma_dlnK": d_sigma_d_lnK,
        "quad_a": float(a), "quad_b": float(b), "quad_c": float(c),
        "rr25": float(rr25), "bf25": float(bf25)
    }


# --------------------- Plotting ---------------------

def make_plots(per_exp: List[Tuple[str, pd.DataFrame]], S: float, outdir: str):
    if plt is None: return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Smile (IV vs strike) for each expiry
    for label, tab in per_exp:
        fig = plt.figure(figsize=(9,5)); ax = plt.gca()
        for typ, mk in [("C","o"), ("P","s")]:
            sub = tab[tab["type"] == typ]
            if sub.empty: continue
            ax.scatter(sub["strike"], 100*sub["iv"], s=14, marker=mk, label=f"{typ}")
        ax.axvline(S, linestyle="--", alpha=0.6)
        ax.set_title(f"Smile: IV vs Strike — {label}")
        ax.set_xlabel("Strike"); ax.set_ylabel("IV (%)")
        ax.legend()
        plt.tight_layout(); fig.savefig(os.path.join(outdir, "plots", f"smile_strike_{label}.png"), dpi=140); plt.close(fig)

        # IV vs delta
        fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
        ax2.scatter(np.abs(tab["delta"]), 100*tab["iv"], s=14, alpha=0.9)
        ax2.set_title(f"Smile: IV vs |Delta| — {label}")
        ax2.set_xlabel("|Delta|"); ax2.set_ylabel("IV (%)")
        plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", f"smile_delta_{label}.png"), dpi=140); plt.close(fig2)

    # Term structure at ATM (if multiple)
    if len(per_exp) > 1:
        dtes = []
        atm_ivs = []
        slopes = []
        for label, tab in per_exp:
            # DTE from first row (approx)
            T_days = max(1, int(round(float(tab["T"].iloc[0] * 365.0))))
            dtes.append(T_days)
            # ATM IV prox
            idx = (tab["strike"] - S).abs().idxmin()
            atm_ivs.append(100*float(tab.loc[idx, "iv"]))
            # slope dσ/dlnK
            # local regression window around ATM
            w = tab.copy()
            w = w[w["ln_moneyness"].between(-0.15, 0.15)]
            if len(w) >= 5:
                X = np.vstack([np.ones(len(w)), w["ln_moneyness"].values, (w["ln_moneyness"].values**2)]).T
                beta, *_ = np.linalg.lstsq(X, w["iv"].values, rcond=None)
                slopes.append(float(beta[1]))
            else:
                slopes.append(np.nan)
        fig3 = plt.figure(figsize=(9,5)); ax3 = plt.gca()
        ax3.plot(dtes, atm_ivs, marker="o")
        ax3.set_title("Term structure: ATM IV"); ax3.set_xlabel("DTE (days)"); ax3.set_ylabel("ATM IV (%)")
        plt.tight_layout(); fig3.savefig(os.path.join(outdir, "plots", "term_atm_iv.png"), dpi=140); plt.close(fig3)

        fig4 = plt.figure(figsize=(9,5)); ax4 = plt.gca()
        ax4.plot(dtes, slopes, marker="s")
        ax4.axhline(0, linestyle="--", alpha=0.6)
        ax4.set_title("Term structure: ATM skew (dσ/dlnK)"); ax4.set_xlabel("DTE (days)"); ax4.set_ylabel("slope")
        plt.tight_layout(); fig4.savefig(os.path.join(outdir, "plots", "term_atm_skew.png"), dpi=140); plt.close(fig4)


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Options skew & smile: IVs, RR25/BF25, ATM skew & term structure")
    ap.add_argument("--ticker", type=str, default=None, help="Underlying ticker (for Yahoo chain)")
    ap.add_argument("--expiry", type=str, default=None, help="Desired expiry YYYY-MM-DD")
    ap.add_argument("--dte-target", type=int, default=None, help="Pick expiry nearest this DTE")
    ap.add_argument("--n-expiries", type=int, default=1, help="If no expiry provided, fetch this many nearest expiries")
    ap.add_argument("--csv", dest="csv_path", type=str, default=None, help="Your CSV chain (see header for columns)")
    ap.add_argument("--underlier", type=float, default=None, help="Override/force underlier spot S")
    ap.add_argument("--rate", type=float, default=0.0, help="Risk-free rate (annual, decimal; e.g., 0.045)")
    ap.add_argument("--div", type=float, default=0.0, help="Dividend yield (annual, decimal)")
    ap.add_argument("--use-last", action="store_true", help="Use last price instead of mid where available")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        ticker=(args.ticker.strip().upper() if args.ticker else None),
        expiry=args.expiry,
        dte_target=args.dte_target,
        n_expiries=int(max(1, args.n_expiries)),
        csv_path=args.csv_path,
        underlier=args.underlier,
        rate=float(args.rate),
        div=float(args.div),
        use_last=bool(args.use_last),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir),
    )

    # --- Load data ---
    raw_frames = []
    spot = cfg.underlier

    if cfg.csv_path:
        raw = load_chain_from_csv(cfg.csv_path)
        raw_frames.append(raw)
        if spot is None and "underlyingprice" in raw.columns and raw["underlyingprice"].notna().any():
            spot = float(raw["underlyingprice"].dropna().iloc[0])

    if cfg.ticker and not cfg.csv_path:
        if yf is None:
            raise SystemExit("Please install yfinance (pip install yfinance) to pull chains from Yahoo.")
        expiries = pick_expiries_yf(cfg.ticker, cfg.n_expiries, cfg.expiry, cfg.dte_target)
        raw, live_spot = load_chain_from_yf(cfg.ticker, expiries)
        raw_frames.append(raw)
        if spot is None: spot = live_spot

    if not raw_frames:
        raise SystemExit("No data source provided. Use --ticker or --csv.")
    chain_raw = pd.concat(raw_frames, ignore_index=True)
    chain_raw.to_csv(os.path.join(cfg.outdir, "chain_raw.csv"), index=False)

    if spot is None or not np.isfinite(spot) or spot <= 0:
        raise SystemExit("Could not infer underlying spot. Provide --underlier explicitly.")

    print(f"[INFO] Using underlier S={spot:.4f}, r={cfg.rate:.4%}, q={cfg.div:.4%}")

    # --- Compute IVs per expiry ---
    per_expiry_tabs: List[Tuple[str, pd.DataFrame]] = []
    metrics_rows = []
    term_rows = []

    if "expiry" not in chain_raw.columns or chain_raw["expiry"].isna().all():
        # treat as single expiry snapshot
        lbl = chain_raw.get("expiry", pd.Series(["unspecified"]*len(chain_raw))).iloc[0]
        tab = compute_iv_table(chain_raw, spot, cfg.rate, cfg.div, price_kind=("last" if cfg.use_last else "mid"))
        if tab.empty:
            raise SystemExit("Empty chain after cleaning. Check price columns or try --use-last.")
        per_expiry_tabs.append((str(lbl)[:10], tab))
        sm = smile_metrics_for_expiry(tab, spot); sm["expiry"] = str(lbl)[:10]
        metrics_rows.append(sm)
        # term
        term_rows.append({"expiry": str(lbl)[:10], "dte": int(round(tab["T"].iloc[0]*365)), "atm_iv": sm["atm_iv"], "atm_slope_dsigma_dlnK": sm["atm_slope_dsigma_dlnK"]})
    else:
        for exp, sub in chain_raw.groupby(chain_raw["expiry"].astype(str)):
            tab = compute_iv_table(sub, spot, cfg.rate, cfg.div, price_kind=("last" if cfg.use_last else "mid"))
            if tab.empty: continue
            per_expiry_tabs.append((exp[:10], tab))
            sm = smile_metrics_for_expiry(tab, spot); sm["expiry"] = exp[:10]
            metrics_rows.append(sm)
            term_rows.append({"expiry": exp[:10], "dte": int(round(tab["T"].iloc[0]*365)), "atm_iv": sm["atm_iv"], "atm_slope_dsigma_dlnK": sm["atm_slope_dsigma_dlnK"]})

    # Save IV tables
    out_iv = []
    for label, tab in per_expiry_tabs:
        t = tab.copy(); t["expiry_label"] = label
        out_iv.append(t)
    iv_df = pd.concat(out_iv, ignore_index=True)
    iv_df.to_csv(os.path.join(cfg.outdir, "chain_iv.csv"), index=False)

    # Save metrics
    metrics = pd.DataFrame(metrics_rows)[["expiry","atm_iv","atm_slope_dsigma_dK","atm_slope_dsigma_dlnK","quad_a","quad_b","quad_c","rr25","bf25"]]
    metrics.to_csv(os.path.join(cfg.outdir, "smile_metrics.csv"), index=False)

    # Term skew
    term = pd.DataFrame(term_rows)
    if not term.empty:
        term = term.sort_values("dte")
        term.to_csv(os.path.join(cfg.outdir, "term_skew.csv"), index=False)

    # Plots
    if cfg.plot:
        make_plots(per_expiry_tabs, spot, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== Smile metrics ===")
    if not metrics.empty:
        snap = metrics.copy()
        snap["atm_iv_%"] = 100*snap["atm_iv"]
        snap["rr25_bps"] = 10000*snap["rr25"]
        snap["bf25_bps"] = 10000*snap["bf25"]
        print(snap[["expiry","atm_iv_%","rr25_bps","bf25_bps","atm_slope_dsigma_dlnK"]].round(3).to_string(index=False))
    if not term.empty:
        print("\n=== Term structure (ATM IV & skew) ===")
        tmp = term.copy(); tmp["atm_iv_%"] = 100*tmp["atm_iv"]
        print(tmp[["expiry","dte","atm_iv_%","atm_slope_dsigma_dlnK"]].round(3).to_string(index=False))

    print("\nFiles written to:", cfg.outdir)


if __name__ == "__main__":
    main()