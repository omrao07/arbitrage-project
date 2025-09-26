#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
japan_reit_vs_global.py — J-REIT vs Global REITs: returns, beta, rates/FX sensitivity, events & scenarios
---------------------------------------------------------------------------------------------------------

What this does
==============
Builds a **monthly (and optional daily)** panel of Japan REITs (J-REIT) and Global REITs,
then measures:
- Rolling **beta/correlation** to global REITs (hedged & unhedged to JPY)
- **Elasticities** to rate moves (JGB10y, UST10y) and **USDJPY** FX (HAC/Newey–West SEs)
- **Event studies** around policy dates (BoJ/FOMC) if daily data provided
- **Scenario engine** for REIT returns under shocks to global REITs, yields, and FX
- **Stress (VaR/ES)** via Monte Carlo using joint driver covariances

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--jpreit jpreit.csv         REQUIRED (daily or monthly)
  Columns (any subset): date, ret[, r, return], tr_index[, total_return_index, index]
  Notes: If only index is given, returns are computed as pct_change (monthly align to EOM).

--global global.csv         REQUIRED (daily or monthly)
  Columns (any subset): date, ret_usd[, r_usd], tr_index_usd[, index_usd, index]
  Notes: If returns are in local USD, script builds JPY-**unhedged** return using USDJPY.

--yields yields.csv         OPTIONAL (daily or monthly; daily preferred for events)
  Columns (any subset): date, jgb_10y[, jgb10], ust_10y[, us10], o/n optional
  Notes: Script uses **Δbp** (difference in basis points).

--fx fx.csv                 OPTIONAL (daily or monthly)
  Columns (any subset): date, USDJPY[, JPYUSD]
  Notes: r_fx = Δlog(USDJPY). Positive r_fx = USD↑/JPY↓.

--events events.csv         OPTIONAL (event list for event study)
  Columns: date, type[, label]  (e.g., BoJ, FOMC, YCC)

CLI (key)
---------
--freq monthly|daily        Frequency for core panel/alignment (default monthly; daily used if you only want high-freq)
--roll 24                   Rolling window (months for monthly, days for daily) for beta/corr
--lags 3                    Lags for HAC/NW regressions (monthly lags; daily will auto-scale)
--scenario_global -8        Global REIT return shock (%) in USD
--scenario_jgb 25           JGB10y shock (bp; + = yields up)
--scenario_ust 25           UST10y shock (bp)
--scenario_fx 5             USDJPY shock (%) (+ = JPY weaker)
--hedge_ratio 0.0           Hedge ratio for Global REIT exposure when computing JPY returns (0=unhedged, 1=fully hedged)
--n_sims 20000              Monte Carlo paths for VaR/ES
--outdir out_reit           Output directory
--start / --end             Optional inclusive date filters (YYYY-MM-DD)

Outputs
-------
panel.csv                   Aligned core panel by chosen frequency
beta_corr_roll.csv          Rolling beta/correlation (J-REIT vs Global hedged/unhedged)
regression_elasticities.csv HAC/NW regressions of J-REIT returns on drivers
event_study.csv             ±window CARs around events (if daily + events available)
scenarios.csv               One-step scenario returns for J-REIT and Global (JPY/hedge aware)
stress_vares.csv            One-step VaR/ES for J-REIT using driver covariance & estimated betas
summary.json                Headline metrics
config.json                 Echo of run configuration

DISCLAIMER: Research tooling. Validate locally for your definitions, tickers, and hedging conventions.
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

def ncol(df: pd.DataFrame, *cands: str) -> Optional[str]:
    if df is None or df.empty: return None
    low = {str(c).lower(): c for c in df.columns}
    for cand in cands:
        if cand in df.columns: return cand
        lc = cand.lower()
        if lc in low: return low[lc]
    # fuzzy contains
    for cand in cands:
        key = cand.lower()
        for c in df.columns:
            if key in str(c).lower(): return c
    return None

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def eom(s: pd.Series) -> pd.Series:
    return to_dt(s) + pd.offsets.MonthEnd(0)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def dlog(x: pd.Series) -> pd.Series:
    x = safe_num(x).replace(0, np.nan)
    return np.log(x) - np.log(x.shift(1))

def pct_change(x: pd.Series) -> pd.Series:
    x = safe_num(x)
    return x.pct_change()

def winsor(x: pd.Series, p: float=0.005) -> pd.Series:
    if x.isna().all(): return x
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

# OLS + HAC (Newey–West)
def ols_beta_resid(X: np.ndarray, y: np.ndarray):
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XTX_inv

def hac_se(X: np.ndarray, resid: np.ndarray, XTX_inv: np.ndarray, L: int) -> np.ndarray:
    n, k = X.shape
    u = resid.reshape(-1,1)
    S = (X * u).T @ (X * u)
    for l in range(1, min(L, n-1) + 1):
        w = 1.0 - l/(L+1)
        G = (X[l:,:] * u[l:]).T @ (X[:-l,:] * u[:-l])
        S += w * (G + G.T)
    cov = XTX_inv @ S @ XTX_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


# ----------------------------- loaders -----------------------------

def load_reit(path: str, side: str) -> pd.DataFrame:
    """
    side ∈ {'jp','global'}
    Accepts returns or index; returns in *native* currency (JPY for JP, USD for Global).
    """
    df = pd.read_csv(path)
    d = ncol(df, "date")
    if not d: raise ValueError(f"{side} reit file needs 'date'.")
    df = df.rename(columns={d:"date"})
    df["date"] = to_dt(df["date"])
    # try returns first
    r = ncol(df, "ret", "r", "return")
    idx = ncol(df, "tr_index", "total_return_index", "index", "px")
    if r:
        df = df.rename(columns={r:"ret"})
        df["ret"] = safe_num(df["ret"])
    elif idx:
        df = df.rename(columns={idx:"index"})
        df["ret"] = pct_change(df["index"])
    else:
        # try first numeric as index
        num = [c for c in df.columns if c != "date"]
        if not num: raise ValueError(f"{side} reit file must have returns or index.")
        df["index"] = safe_num(df[num[0]])
        df["ret"] = pct_change(df["index"])
    df = df[["date","ret"]].copy()
    df = df.rename(columns={"ret": f"ret_{side}"})
    return df.sort_values("date")

def load_yields(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); j = ncol(df,"jgb_10y","jgb10","jgb_10"); u = ncol(df,"ust_10y","us10","ust10")
    if not d: raise ValueError("yields.csv needs 'date'.")
    df = df.rename(columns={d:"date"})
    if j: df = df.rename(columns={j:"jgb_10y"})
    if u: df = df.rename(columns={u:"ust_10y"})
    df["date"] = to_dt(df["date"])
    for k in ["jgb_10y","ust_10y"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # daily/period Δbp
    for k in ["jgb_10y","ust_10y"]:
        if k in df.columns:
            df[f"d_{k}_bp"] = df[k].diff() * 100.0  # assuming yields in %
    return df.sort_values("date")

def load_fx(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); u = ncol(df,"USDJPY","usdjpy"); j = ncol(df,"JPYUSD","jpyusd")
    if not d: raise ValueError("fx.csv needs 'date'.")
    df = df.rename(columns={d:"date"})
    df["date"] = to_dt(df["date"])
    if u:
        df["USDJPY"] = safe_num(df[u])
    elif j:
        df["USDJPY"] = 1.0 / safe_num(df[j])
    else:
        num = [c for c in df.columns if c != "date"]
        if not num: raise ValueError("fx.csv needs USDJPY or JPYUSD.")
        df["USDJPY"] = safe_num(df[num[0]])
    df["r_fx"] = dlog(df["USDJPY"])
    return df.sort_values("date")

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); t = ncol(df,"type"); l = ncol(df,"label","event")
    if not d: raise ValueError("events.csv needs date.")
    df = df.rename(columns={d:"date"})
    if t: df = df.rename(columns={t:"type"})
    if l: df = df.rename(columns={l:"label"})
    df["date"] = to_dt(df["date"])
    if "type" not in df.columns: df["type"] = "EVENT"
    if "label" not in df.columns: df["label"] = df["type"]
    return df.sort_values("date")


# ----------------------------- alignment & panel -----------------------------

def to_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty: return df
    if freq.lower().startswith("month"):
        # resample to month-end sum for returns (compounding approx), last for drivers
        if "ret_jp" in df.columns or "ret_global" in df.columns:
            # For returns, use (1+r) product within month - 1
            g = df.set_index("date").copy()
            out = []
            for col in g.columns:
                if col.startswith("ret_"):
                    tmp = (1.0 + g[col]).resample("M").prod() - 1.0
                else:
                    tmp = g[col].resample("M").last()
                out.append(tmp.rename(col))
            R = pd.concat(out, axis=1).reset_index()
            R["date"] = eom(R["date"])
            return R
        else:
            g = df.set_index("date").resample("M").last().reset_index()
            g["date"] = eom(g["date"])
            return g
    else:
        return df.copy()

def build_panel(freq: str, JP: pd.DataFrame, GL: pd.DataFrame, YLD: pd.DataFrame, FX: pd.DataFrame, hedge_ratio: float) -> pd.DataFrame:
    D = JP.merge(GL, on="date", how="outer")
    if not FX.empty:
        D = D.merge(FX[["date","USDJPY","r_fx"]], on="date", how="left")
    if not YLD.empty:
        cols = ["jgb_10y","ust_10y","d_jgb_10y_bp","d_ust_10y_bp"]
        avail = [c for c in cols if c in YLD.columns]
        D = D.merge(YLD[["date"]+avail], on="date", how="left")
    # frequency alignment
    D = to_freq(D, freq=freq)
    # Global REIT JPY returns: hedged vs unhedged
    if "ret_global" in D.columns:
        if "r_fx" in D.columns:
            D["ret_global_jpy_unhedged"] = (1.0 + D["ret_global"]) * (1.0 + D["r_fx"]) - 1.0
        else:
            D["ret_global_jpy_unhedged"] = D["ret_global"]  # fallback
        h = float(max(0.0, min(1.0, hedge_ratio)))
        # blend: hedged = ret_global (USD), unhedged = USD + FX; effective JPY return given hedge ratio h
        D["ret_global_jpy_eff"] = h * D["ret_global"] + (1.0 - h) * D.get("ret_global_jpy_unhedged", D["ret_global"])
    # tidy
    D = D.sort_values("date").reset_index(drop=True)
    return D


# ----------------------------- rolling stats -----------------------------

def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    if y.isna().all() or x.isna().all(): return pd.Series(index=y.index, dtype=float)
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / (var + 1e-12)

def rolling_corr(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    return y.rolling(window).corr(x)

def make_beta_corr(D: pd.DataFrame, window: int) -> pd.DataFrame:
    if D.empty or "ret_jp" not in D.columns or "ret_global" not in D.columns:
        return pd.DataFrame()
    df = D.copy()
    out = pd.DataFrame({"date": df["date"]})
    # vs Global (USD)
    out["beta_vs_global_usd"]  = rolling_beta(df["ret_jp"], df["ret_global"], window)
    out["corr_vs_global_usd"]  = rolling_corr(df["ret_jp"], df["ret_global"], window)
    # vs Global (JPY unhedged)
    if "ret_global_jpy_unhedged" in df.columns:
        out["beta_vs_global_jpy_unhedged"] = rolling_beta(df["ret_jp"], df["ret_global_jpy_unhedged"], window)
        out["corr_vs_global_jpy_unhedged"] = rolling_corr(df["ret_jp"], df["ret_global_jpy_unhedged"], window)
    # vs Global (effective hedge)
    if "ret_global_jpy_eff" in df.columns:
        out["beta_vs_global_jpy_eff"] = rolling_beta(df["ret_jp"], df["ret_global_jpy_eff"], window)
        out["corr_vs_global_jpy_eff"] = rolling_corr(df["ret_jp"], df["ret_global_jpy_eff"], window)
    # rates betas
    for k in ["d_jgb_10y_bp","d_ust_10y_bp"]:
        if k in df.columns:
            out[f"beta_{k}"] = rolling_beta(df["ret_jp"], df[k], window)
    return out


# ----------------------------- regressions (elasticities) -----------------------------

def regress_elasticities(D: pd.DataFrame, L: int) -> pd.DataFrame:
    """
    r_jp ~ alpha + b1 * r_global_eff + b2 * ΔJGB(bp) + b3 * ΔUST(bp) + b4 * r_fx + ε
    HAC/NW SEs with lag L (months for monthly, or scaled days for daily usage upstream).
    """
    if D.empty or "ret_jp" not in D.columns: return pd.DataFrame()
    g = D.dropna(subset=["ret_jp"]).copy()
    # choose best global proxy available: eff > unhedged > usd
    xg = "ret_global_jpy_eff" if "ret_global_jpy_eff" in g.columns else ("ret_global_jpy_unhedged" if "ret_global_jpy_unhedged" in g.columns else ("ret_global" if "ret_global" in g.columns else None))
    Xlist = []
    names = ["const"]
    const = np.ones((len(g),1))
    Xlist.append(const)
    if xg:
        Xlist.append(g[[xg]].values); names.append(xg)
    for k in ["d_jgb_10y_bp","d_ust_10y_bp","r_fx"]:
        if k in g.columns:
            Xlist.append(g[[k]].values); names.append(k)
    X = np.hstack(Xlist)
    y = g[["ret_jp"]].values
    if X.shape[1] < 2 or len(g) < (10 + X.shape[1]): return pd.DataFrame()
    beta, resid, XTX_inv = ols_beta_resid(X, y)
    se = hac_se(X, resid, XTX_inv, L=max(4, L))
    rows = []
    for i, nm in enumerate(names):
        b = float(beta[i,0]); s = float(se[i]); t = b/s if s>0 else np.nan
        rows.append({"var": nm, "coef": b, "se": s, "t_stat": t, "n": int(len(g)), "lags": int(L)})
    return pd.DataFrame(rows)

# pick coefficient helper
def pick_coef(EL: pd.DataFrame, var: str, default: float) -> float:
    r = EL[EL["var"]==var]
    return float(r["coef"].iloc[0]) if not r.empty else float(default)


# ----------------------------- event study -----------------------------

def event_study_daily(R: pd.DataFrame, E: pd.DataFrame, window: int=5) -> pd.DataFrame:
    """
    CAR on J-REIT daily returns around events ±window days.
    """
    if R.empty or E.empty or "ret_jp" not in R.columns: return pd.DataFrame()
    ser = R.set_index("date")["ret_jp"].dropna()
    rows = []
    for _, ev in E.iterrows():
        dt = pd.Timestamp(ev["date"])
        if dt not in ser.index:
            idx = ser.index
            pos = idx.searchsorted(dt)
            if pos >= len(idx): continue
            dt = idx[pos]
        i = ser.index.get_loc(dt)
        L = max(0, i-window); R_ = min(len(ser)-1, i+window)
        car = float(ser.iloc[L:R_+1].sum())
        rows.append({"event_date": str(ser.index[i].date()),
                     "type": str(ev.get("type","EVENT")),
                     "label": str(ev.get("label", ev.get("type","EVENT"))),
                     "CAR_ret_jp": car})
    return pd.DataFrame(rows).sort_values(["event_date","type"])


# ----------------------------- scenarios -----------------------------

def run_scenario(D_last: pd.Series, EL: pd.DataFrame,
                 shock_global_pct: float, shock_jgb_bp: float, shock_ust_bp: float, shock_fx_pct: float) -> Dict[str,float]:
    """
    One-step expected returns from linear elasticities.
    """
    # Priors if regression missing (monthly scale)
    b_global = pick_coef(EL, "ret_global_jpy_eff", default=0.60) if "ret_global_jpy_eff" in D_last.index else \
               pick_coef(EL, "ret_global_jpy_unhedged", default=0.65) if "ret_global_jpy_unhedged" in D_last.index else \
               pick_coef(EL, "ret_global", default=0.55)
    b_jgb   = pick_coef(EL, "d_jgb_10y_bp", default=-3.0/100.0)   # per bp (convert later)
    b_ust   = pick_coef(EL, "d_ust_10y_bp", default=-1.0/100.0)   # per bp
    b_fx    = pick_coef(EL, "r_fx", default=-0.20)

    r_global_usd = shock_global_pct/100.0
    r_fx = np.log1p(shock_fx_pct/100.0)  # consistency with r_fx definition
    # choose effective global proxy (will be recomputed for display)
    if "ret_global_jpy_unhedged" in D_last.index:
        r_global_jpy_unhedged = (1.0 + r_global_usd) * (1.0 + r_fx) - 1.0
    else:
        r_global_jpy_unhedged = np.nan

    # Expected J-REIT return
    r_jp = b_global * (r_global_usd if "ret_global" in D_last.index and "ret_global_jpy_eff" not in D_last.index and "ret_global_jpy_unhedged" not in D_last.index
                       else (D_last.get("ret_global_jpy_eff", r_global_jpy_unhedged) if np.isfinite(D_last.get("ret_global_jpy_eff", np.nan)) else r_global_jpy_unhedged))
    r_jp += b_jgb * shock_jgb_bp + b_ust * shock_ust_bp + b_fx * r_fx

    # Global JPY effective return (if hedge info present)
    if "ret_global_jpy_eff" in D_last.index and np.isfinite(D_last.get("ret_global_jpy_eff", np.nan)):
        # infer hedge ratio from identity: eff = h*usd + (1-h)*unhedged
        # Using last period to recover h; if not solvable, default h=0
        try:
            h = (D_last["ret_global_jpy_eff"] - D_last.get("ret_global_jpy_unhedged", D_last.get("ret_global", 0.0)))) / (D_last.get("ret_global", 0.0) - D_last.get("ret_global_jpy_unhedged", D_last.get("ret_global", 0.0)) + 1e-12)
            h = float(np.clip(h, 0.0, 1.0))
        except Exception:
            h = 0.0
        r_gl_jpy_eff = h * r_global_usd + (1.0 - h) * r_global_jpy_unhedged
    else:
        r_gl_jpy_eff = r_global_jpy_unhedged if np.isfinite(r_global_jpy_unhedged) else r_global_usd

    return {"ret_jpreit": float(r_jp),
            "ret_global_usd": float(r_global_usd),
            "ret_global_jpy_unhedged": float(r_global_jpy_unhedged) if np.isfinite(r_global_jpy_unhedged) else np.nan,
            "ret_global_jpy_eff": float(r_gl_jpy_eff) if np.isfinite(r_gl_jpy_eff) else np.nan}


# ----------------------------- stress (VaR/ES) -----------------------------

def stress_var_es(D: pd.DataFrame, EL: pd.DataFrame, n_sims: int=20000) -> pd.DataFrame:
    """
    One-step distribution for J-REIT returns using joint shocks in:
      [ret_global_usd, d_jgb_10y_bp, d_ust_10y_bp, r_fx]
    with linear elasticities. Uses monthly panel for covariance.
    """
    need = []
    if "ret_global" in D.columns: need.append("ret_global")
    elif "ret_global_jpy_eff" in D.columns: need.append("ret_global_jpy_eff")
    else: return pd.DataFrame()

    for k in ["d_jgb_10y_bp","d_ust_10y_bp","r_fx"]:
        if k in D.columns: need.append(k)
    R = D[need].dropna()
    if R.shape[0] < 24: return pd.DataFrame()
    mu = R.mean().values
    cov = R.cov().values
    try:
        L = np.linalg.cholesky(cov + 1e-12*np.eye(len(need)))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        L = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12)))
    rng = np.random.default_rng(42)
    Z = rng.standard_normal(size=(n_sims, len(need)))
    shocks = Z @ L.T + mu

    # elasticities (fallbacks)
    b_global = pick_coef(EL, "ret_global_jpy_eff", default=0.60) if "ret_global_jpy_eff" in D.columns else \
               pick_coef(EL, "ret_global", default=0.55)
    b_jgb   = pick_coef(EL, "d_jgb_10y_bp", default=-3.0/100.0)
    b_ust   = pick_coef(EL, "d_ust_10y_bp", default=-1.0/100.0)
    b_fx    = pick_coef(EL, "r_fx", default=-0.20)

    # map columns
    idx = {nm:i for i,nm in enumerate(need)}
    x_global = shocks[:, idx[need[0]]]
    x_jgb = shocks[:, idx["d_jgb_10y_bp"]] if "d_jgb_10y_bp" in idx else 0.0
    x_ust = shocks[:, idx["d_ust_10y_bp"]] if "d_ust_10y_bp" in idx else 0.0
    x_fx  = shocks[:, idx["r_fx"]] if "r_fx" in idx else 0.0

    r = b_global * x_global + b_jgb * x_jgb + b_ust * x_ust + b_fx * x_fx
    r_sorted = np.sort(r)
    var5 = float(np.percentile(r, 5))
    es5 = float(r_sorted[:max(1,int(0.05*len(r_sorted)))].mean())
    return pd.DataFrame([{
        "VaR_5pct_ret_jp": var5,
        "ES_5pct_ret_jp": es5,
        "mean_ret_jp": float(r.mean()),
        "sd_ret_jp": float(r.std(ddof=0)),
        "n_sims": int(n_sims)
    }])


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    jpreit: str
    global_reit: str
    yields: Optional[str]
    fx: Optional[str]
    events: Optional[str]
    freq: str
    roll: int
    lags: int
    hedge_ratio: float
    scenario_global: float
    scenario_jgb: float
    scenario_ust: float
    scenario_fx: float
    n_sims: int
    start: Optional[str]
    end: Optional[str]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="J-REIT vs Global REITs — betas, rates/FX sensitivity, events & scenarios")
    ap.add_argument("--jpreit", required=True)
    ap.add_argument("--global", dest="global_reit", required=True)
    ap.add_argument("--yields", default="")
    ap.add_argument("--fx", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--freq", default="monthly", choices=["monthly","daily"])
    ap.add_argument("--roll", type=int, default=24)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--hedge_ratio", type=float, default=0.0, help="Hedge ratio for global REIT when forming JPY effective return")
    # Scenario shocks
    ap.add_argument("--scenario_global", type=float, default=-8.0, help="Global REIT shock (%) in USD")
    ap.add_argument("--scenario_jgb", type=float, default=+25.0, help="JGB10y shock (bp)")
    ap.add_argument("--scenario_ust", type=float, default=+25.0, help="UST10y shock (bp)")
    ap.add_argument("--scenario_fx", type=float, default=+5.0, help="USDJPY shock (%) (+ = JPY weaker)")
    ap.add_argument("--n_sims", type=int, default=20000)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--outdir", default="out_reit")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    JP = load_reit(args.jpreit, side="jp")
    GL = load_reit(args.global_reit, side="global")
    YLD = load_yields(args.yields) if args.yields else pd.DataFrame()
    FX  = load_fx(args.fx) if args.fx else pd.DataFrame()
    EVT = load_events(args.events) if args.events else pd.DataFrame()

    # Time filters
    if args.start:
        s = to_dt(pd.Series([args.start])).iloc[0]
        for df in [JP, GL, YLD, FX, EVT]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] < s].index, inplace=True)
    if args.end:
        e = to_dt(pd.Series([args.end])).iloc[0]
        for df in [JP, GL, YLD, FX, EVT]:
            if not df.empty and "date" in df.columns:
                df.drop(df[df["date"] > e].index, inplace=True)

    # Panel (raw daily for event study; core panel by freq)
    D_daily = build_panel(freq="daily", JP=JP, GL=GL, YLD=YLD, FX=FX, hedge_ratio=float(args.hedge_ratio))
    D = build_panel(freq=args.freq, JP=JP, GL=GL, YLD=YLD, FX=FX, hedge_ratio=float(args.hedge_ratio))
    if D.empty: raise ValueError("Panel empty after merges/filters. Check inputs.")
    D.to_csv(outdir / "panel.csv", index=False)

    # Rolling stats (use chosen frequency)
    window = int(args.roll)
    B = make_beta_corr(D, window=window)
    if not B.empty: B.to_csv(outdir / "beta_corr_roll.csv", index=False)

    # Elasticities
    L = int(args.lags)
    # For daily frequency, scale HAC lags roughly to 3 months of trading days
    if args.freq == "daily":
        L = max(L, 60)
    EL = regress_elasticities(D, L=L)
    if not EL.empty: EL.to_csv(outdir / "regression_elasticities.csv", index=False)

    # Event study (daily)
    ES = event_study_daily(D_daily, EVT, window=5) if (not D_daily.empty and not EVT.empty) else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Scenario (use last observation to infer hedge-effective composition)
    last = D.dropna(subset=["ret_jp"]).tail(1).iloc[0] if not D.empty else pd.Series(dtype=float)
    SC = run_scenario(last, EL if not EL.empty else pd.DataFrame(),
                      shock_global_pct=float(args.scenario_global),
                      shock_jgb_bp=float(args.scenario_jgb),
                      shock_ust_bp=float(args.scenario_ust),
                      shock_fx_pct=float(args.scenario_fx))
    pd.DataFrame([SC]).to_csv(outdir / "scenarios.csv", index=False)

    # Stress VaR/ES (monthly)
    ST = stress_var_es(D, EL if not EL.empty else pd.DataFrame(), n_sims=int(args.n_sims))
    if not ST.empty: ST.to_csv(outdir / "stress_vares.csv", index=False)

    # Summary
    ann_mult = 12 if args.freq=="monthly" else 252
    jpret = D["ret_jp"].dropna()
    ann_vol = float(jpret.std(ddof=0) * np.sqrt(ann_mult)) if not jpret.empty else None
    corr_g = float(D[["ret_jp","ret_global"]].dropna().corr().iloc[0,1]) if {"ret_jp","ret_global"}.issubset(D.columns) else None
    corr_gu = float(D[["ret_jp","ret_global_jpy_unhedged"]].dropna().corr().iloc[0,1]) if {"ret_jp","ret_global_jpy_unhedged"}.issubset(D.columns) else None

    summary = {
        "sample": {
            "start": str(D["date"].min().date()),
            "end": str(D["date"].max().date()),
            "n_points": int(D.shape[0]),
            "freq": args.freq
        },
        "rolling_window": int(args.roll),
        "jp_reit": {
            "ann_vol": ann_vol,
            "corr_vs_global_usd": corr_g,
            "corr_vs_global_jpy_unhedged": corr_gu
        },
        "scenario": {
            "inputs": {"global_%": float(args.scenario_global), "jgb_bp": float(args.scenario_jgb),
                       "ust_bp": float(args.scenario_ust), "usdjpy_%": float(args.scenario_fx)},
            "outputs": SC
        },
        "outputs": {
            "panel": "panel.csv",
            "beta_corr_roll": "beta_corr_roll.csv" if not B.empty else None,
            "regression_elasticities": "regression_elasticities.csv" if not EL.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "scenarios": "scenarios.csv",
            "stress_vares": "stress_vares.csv" if not ST.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Config echo
    cfg = asdict(Config(
        jpreit=args.jpreit, global_reit=args.global_reit, yields=(args.yields or None),
        fx=(args.fx or None), events=(args.events or None), freq=args.freq, roll=int(args.roll),
        lags=int(args.lags), hedge_ratio=float(args.hedge_ratio), scenario_global=float(args.scenario_global),
        scenario_jgb=float(args.scenario_jgb), scenario_ust=float(args.scenario_ust),
        scenario_fx=float(args.scenario_fx), n_sims=int(args.n_sims),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== J-REIT vs Global REITs Toolkit ==")
    print(f"Sample ({args.freq}): {summary['sample']['start']} → {summary['sample']['end']}  [{summary['sample']['n_points']} points]")
    if summary["outputs"]["beta_corr_roll"]: print("Rolling beta/correlation computed.")
    if summary["outputs"]["regression_elasticities"]: print("Elasticities (rates/FX/global) estimated (HAC/NW).")
    if summary["outputs"]["event_study"]: print("Event study (daily) written.")
    print(f"Scenario: J-REIT {SC['ret_jpreit']:.2%}, Global USD {SC['ret_global_usd']:.2%}, Global JPY eff {SC.get('ret_global_jpy_eff', np.nan):.2%}")
    if summary["outputs"]["stress_vares"]: print("Stress VaR/ES computed.")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
