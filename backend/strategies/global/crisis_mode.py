#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crisis_mode.py — Portfolio crisis monitor, trigger engine & hedge/playbook generator

What this does
--------------
Continuously (or batch) evaluates market/portfolio stress and, when thresholds are hit,
produces concrete risk-off actions and hedge sizes.

Core features
-------------
1) Cleans inputs:
   - Prices (long or wide) → returns by ticker
   - Positions (weights/$ notional, optional factor tags: asset_class, beta, dv01)
   - Risk series (VIX/MOVE/credit spreads/TED etc., optional)
   - PnL (optional) for VaR breach checks
2) Builds indicators (per-day):
   - Portfolio return, rolling vol (EWMA + realized)
   - Drawdown and max drawdown speed
   - Average absolute correlation (AAC) across tickers
   - Change-point signal (Page–Hinkley)
   - Composite Stress Index (CSI) from z-scored risk series + return/vol signals
3) Calibrates thresholds from a "normal" window (percentiles) and assigns CRISIS STAGE ∈ {0..3}
4) Generates a crisis playbook:
   - Gross/net reduction targets per stage
   - Hedge recommendations sized by minimum-variance ratio using candidate proxies
   - Simple scenario P&L table (Equity -5/-10/-20%, USD +2/+5%, Rates -20/-50bp (via TLT), Credit +2/+5% drawdowns)
5) Writes tidy outputs:
   - indicators.csv, triggers.csv, playbook_actions.csv
   - hedge_recs.csv (per stage, per hedge), scenario_pnl.csv
   - summary.json (latest stage & key metrics)
   - checklist.txt (practical steps)

Inputs (CSV; case-insensitive headers)
--------------------------------------
--prices prices.csv   REQUIRED
  EITHER wide: date, TICKER1, TICKER2, ...
  OR long : date, ticker, close

--positions positions.csv   OPTIONAL (improves sizing)
  Columns: ticker, weight  (0..1) OR position_usd
           [asset_class, beta, dv01] optional

--risk risk.csv       OPTIONAL (long)
  Columns: date, series, value    (e.g., VIX, MOVE, TED, CDX_HY, FRA_OIS, FX_VOL, LIQ_IDX)

--pnl pnl.csv         OPTIONAL (portfolio P&L)
  Columns: date, pnl_usd

CLI
---
--lookback 252                 Rolling window for stats (days)
--calib_start 2018-01-01       Optional normal-period start
--calib_end   2019-12-31       Optional normal-period end
--var_p 0.99                   VaR confidence (historical & EWMA)
--alpha_ewma 0.94              EWMA decay
--stage_pcts "80,92,97"        CSI percentiles → Stage1/2/3
--outdir out_crisis

Hedge candidates (tickers expected in prices)
---------------------------------------------
Equity:  SPY (global proxy), QQQ, ES_FUT (if you track it as a price series)
Rates :  TLT (long UST duration), IEF
Credit:  HYG (HY), LQD (IG)
USD   :  UUP (DXY proxy)
Gold  :  GLD
(You can add more; the script auto-uses any of these found in your price panel.)

DISCLAIMER: Research tooling. Not investment advice. Double-check sizes before acting.
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

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    for c in df.columns:
        if t in c.lower():
            return c
    return None

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def dlogret(x: pd.Series) -> pd.Series:
    return np.log(x.astype(float).replace(0, np.nan)).diff()

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return s*0
    return (s - mu) / sd

def ewma_vol(r: pd.Series, alpha: float=0.94) -> pd.Series:
    v = []
    s2 = 0.0
    for ret in r.fillna(0.0):
        s2 = alpha*s2 + (1-alpha)*(ret**2)
        v.append(np.sqrt(s2))
    return pd.Series(v, index=r.index)

def max_drawdown(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return drawdown (peak-to-trough) and running peak."""
    peak = series.cummax()
    dd = (series/peak) - 1.0
    return dd, peak

def page_hinkley(x: pd.Series, delta: float=0.0005, lamb: float=0.01) -> pd.Series:
    """
    Simple Page–Hinkley change detector on returns (assumes small mean under H0).
    Returns cumulative statistic; spikes suggest regime shift.
    """
    m_t = 0.0
    ph = []
    for xi in x.fillna(0.0):
        m_t = max(0.0, m_t + xi - delta)
        ph.append(m_t)
    phs = pd.Series(ph, index=x.index)
    # binary-ish signal via threshold lamb
    return (phs > lamb).astype(float)

def avg_abs_corr(R: pd.DataFrame, window: int=60) -> pd.Series:
    aac = []
    idx = R.index
    for t in range(len(R)):
        lo = max(0, t-window+1); hi = t+1
        sub = R.iloc[lo:hi].dropna(how="all", axis=1)
        if sub.shape[0] < 5 or sub.shape[1] < 2:
            aac.append(np.nan); continue
        C = sub.corr().values
        n = C.shape[0]
        v = (np.abs(C).sum() - n) / (n*(n-1))
        aac.append(v)
    return pd.Series(aac, index=idx)

def hist_var(pnl: pd.Series, p: float=0.99) -> float:
    x = pnl.dropna().values
    if x.size < 50: return np.nan
    return -np.quantile(x, 1.0-p)

def ewma_var_from_returns(r: pd.Series, alpha: float=0.94, p: float=0.99) -> float:
    # assume normal; sigma = last EWMA vol; VaR = z * sigma; z at p
    sig = float(ewma_vol(r, alpha=alpha).iloc[-1])
    # z-score for one-sided p (e.g., 99% -> 2.33)
    from math import sqrt, log, pi
    # Approx inverse-CDF via Beasley-Springer/Moro simplified (or fallback)
    try:
        import math
        # Simple approx: use scipy-like constants (hardcoded)
        # For p close to 1:
        t = np.sqrt(-2*np.log(1-p))
        z = t - (np.log(t) + np.log(2*np.pi))/(2*t)  # crude
    except Exception:
        z = 2.33
    if not np.isfinite(sig): return np.nan
    return float(z * sig)

def pct_to_mult(x: float) -> float:
    return 1.0 + x/100.0


# ----------------------------- loaders -----------------------------

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_c = ncol(df, "date") or df.columns[0]
    df = df.rename(columns={date_c: "date"})
    df["date"] = to_date(df["date"])
    if ncol(df, "ticker"):
        # long
        tik = ncol(df, "ticker")
        val = ncol(df, "close") or ncol(df, "price") or "close"
        df = df.rename(columns={tik:"ticker", val:"close"})
        piv = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    else:
        piv = df.set_index("date").sort_index()
    piv = piv.loc[:, ~piv.columns.duplicated()]
    return piv

def load_positions(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"ticker") or "ticker"):"ticker",
           (ncol(df,"weight") or "weight"):"weight",
           (ncol(df,"position_usd") or ncol(df,"notional_usd") or "position_usd"):"position_usd",
           (ncol(df,"asset_class") or "asset_class"):"asset_class",
           (ncol(df,"beta") or "beta"):"beta",
           (ncol(df,"dv01") or "dv01"):"dv01"}
    df = df.rename(columns=ren)
    if "ticker" not in df.columns: raise ValueError("positions.csv must have a 'ticker' column.")
    for c in ["weight","position_usd","beta","dv01"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "weight" not in df.columns or df["weight"].isna().all():
        tot = df["position_usd"].sum()
        if tot and np.isfinite(tot) and tot != 0:
            df["weight"] = df["position_usd"] / tot
        else:
            df["weight"] = 1.0/len(df)
    if "asset_class" not in df.columns:
        df["asset_class"] = "UNKNOWN"
    return df[["ticker","weight","position_usd","asset_class","beta","dv01"]]

def load_risk(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"series") or "series"):"series",
           (ncol(df,"value") or "value"):"value"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["series"] = df["series"].astype(str).str.upper().str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

def load_pnl(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"date") or df.columns[0]):"date",
           (ncol(df,"pnl_usd") or "pnl_usd"):"pnl_usd"}
    df = df.rename(columns=ren)
    df["date"] = to_date(df["date"])
    df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")
    return df


# ----------------------------- analytics -----------------------------

def build_returns_panel(prices: pd.DataFrame) -> pd.DataFrame:
    R = prices.apply(dlogret)
    R.index.name = "date"
    return R

def portfolio_return(R: pd.DataFrame, positions: pd.DataFrame) -> pd.Series:
    if positions.empty:
        # equal weight across available tickers
        w = pd.Series(1.0 / R.shape[1], index=R.columns)
    else:
        w = positions.set_index("ticker")["weight"].reindex(R.columns).fillna(0.0)
        s = w.sum()
        if s != 0: w = w / s
    return R.mul(w, axis=1).sum(axis=1)

def composite_stress_index(
    risk_long: pd.DataFrame,
    port_ret: pd.Series,
    R: pd.DataFrame,
    lookback: int,
    alpha_ewma: float
) -> pd.DataFrame:
    idx = port_ret.index
    out = pd.DataFrame(index=idx)

    # Vol & drawdown signals
    out["port_ret"] = port_ret
    out["vol_real"] = port_ret.rolling(lookback//21, min_periods=10).std(ddof=0) * np.sqrt(252)
    out["vol_ewma"] = ewma_vol(port_ret, alpha=alpha_ewma) * np.sqrt(252)
    # Equity-like drawdown on portfolio cum value (starting at 1)
    cum = (port_ret.fillna(0.0) + 1).cumprod()
    dd, peak = max_drawdown(cum)
    out["drawdown"] = dd
    out["dd_speed"] = dd.diff().clip(lower=0.0)  # worsening only

    # Correlation crowding
    out["aac"] = avg_abs_corr(R, window=min(60, lookback//3))

    # Change-point
    out["phinkley"] = page_hinkley(port_ret.fillna(0.0), delta=0.0005, lamb=0.01)

    # External risk series — z-score each and average
    if not risk_long.empty:
        piv = risk_long.pivot_table(index="date", columns="series", values="value", aggfunc="last").reindex(idx)
        z = piv.apply(zscore)
        out["risk_z_mean"] = z.mean(axis=1)
    else:
        out["risk_z_mean"] = 0.0

    # Normalize signals to z and combine (weights heuristic)
    sigs = pd.DataFrame({
        "z_vol": zscore(out["vol_ewma"].fillna(method="ffill")),
        "z_dd" : zscore(out["drawdown"].fillna(0.0)),
        "z_aac": zscore(out["aac"].fillna(method="ffill")),
        "z_ph" : zscore(out["phinkley"].fillna(0.0)),
        "z_risk": out["risk_z_mean"].fillna(0.0)
    })
    weights = {"z_vol": 0.30, "z_dd": 0.25, "z_aac": 0.20, "z_ph": 0.10, "z_risk": 0.15}
    out["CSI"] = sum(weights[k]*sigs[k] for k in weights)

    return out

def calibrate_stages(df: pd.DataFrame, calib_start: Optional[str], calib_end: Optional[str], stage_pcts: Tuple[float,float,float]) -> Dict[str, float]:
    CSI = df["CSI"].dropna()
    if CSI.empty:
        return {"stage1": 1.0, "stage2": 1.75, "stage3": 2.5}
    if calib_start and calib_end:
        cs = pd.to_datetime(calib_start); ce = pd.to_datetime(calib_end)
        CSI = CSI[(CSI.index>=cs) & (CSI.index<=ce)]
        if CSI.empty: CSI = df["CSI"].dropna()
    q1, q2, q3 = np.quantile(CSI, [stage_pcts[0], stage_pcts[1], stage_pcts[2]])
    return {"stage1": float(q1), "stage2": float(q2), "stage3": float(q3)}

def assign_stage(csi: float, thr: Dict[str, float], combo_flags: Dict[str, bool]) -> int:
    # Escalate if multiple red flags
    base = 0
    if csi >= thr["stage1"]: base = 1
    if csi >= thr["stage2"]: base = 2
    if csi >= thr["stage3"]: base = 3
    # additional escalation: if two or more auxiliary flags true, bump one stage
    aux = sum(1 for v in combo_flags.values() if v)
    if aux >= 2 and base < 3:
        base += 1
    return base

def var_breaches(pnl: pd.Series, port_ret: pd.Series, var_p: float, alpha_ewma: float) -> Dict[str, bool]:
    flags = {"hist_var_breach": False, "ewma_var_breach": False}
    if not pnl.dropna().empty:
        hv = hist_var(pnl, p=var_p)
        last = float(pnl.dropna().iloc[-1])
        flags["hist_var_breach"] = (last < -hv) if np.isfinite(hv) else False
    # fallback via returns scaling
    ev = ewma_var_from_returns(port_ret, alpha=alpha_ewma, p=var_p)
    last_ret = float(port_ret.dropna().iloc[-1]) if port_ret.dropna().shape[0] else 0.0
    flags["ewma_var_breach"] = (last_ret < -ev) if np.isfinite(ev) else False
    return flags

def min_var_hedge_ratio(port_ret: pd.Series, hedge_ret: pd.Series) -> float:
    a = pd.concat([port_ret, hedge_ret], axis=1, join="inner").dropna()
    if a.shape[0] < 60:
        return 0.0
    cov = np.cov(a.iloc[:,0], a.iloc[:,1], ddof=0)[0,1]
    varh = np.var(a.iloc[:,1], ddof=0)
    if varh <= 0: return 0.0
    return - cov / varh  # h* s.t. r_port + h*r_hedge min variance

def candidate_hedges_in_panel(prices: pd.DataFrame) -> List[str]:
    cands = ["SPY","QQQ","TLT","IEF","HYG","LQD","UUP","GLD"]
    return [c for c in cands if c in prices.columns]

def scenario_shocks() -> List[Dict]:
    return [
        {"name": "EQ_-5_USD+2_RATES-20bp_CRED-2", "equity_pct": -5, "usd_pct": +2, "rates_bp": -20, "credit_pct": -2},
        {"name": "EQ_-10_USD+5_RATES-50bp_CRED-5", "equity_pct": -10, "usd_pct": +5, "rates_bp": -50, "credit_pct": -5},
        {"name": "EQ_-20_USD+7_RATES-70bp_CRED-8", "equity_pct": -20, "usd_pct": +7, "rates_bp": -70, "credit_pct": -8},
    ]

def classify_proxy(ticker: str) -> str:
    t = ticker.upper()
    if t in ["SPY","QQQ"]: return "EQUITY"
    if t in ["TLT","IEF"]: return "RATES"
    if t in ["HYG","LQD"]: return "CREDIT"
    if t in ["UUP"]:       return "USD"
    if t in ["GLD"]:       return "GOLD"
    return "OTHER"

def playbook_for_stage(stage: int) -> Dict[str, str]:
    if stage <= 0:
        return {
            "gross_cut_target": "0%",
            "net_cut_target": "0%",
            "cash_min": "5%",
            "notes": "Normal ops. Keep hedges modest, refresh playbook weekly."
        }
    if stage == 1:
        return {
            "gross_cut_target": "15–20%",
            "net_cut_target": "10–15%",
            "cash_min": "10–15%",
            "notes": "Tighten stops, trim losers/crowded longs, add pilot hedges (25–35% of beta/DV01)."
        }
    if stage == 2:
        return {
            "gross_cut_target": "35–50%",
            "net_cut_target": "25–35%",
            "cash_min": "20–30%",
            "notes": "De-gross cyclicals/illiquid names, size hedges to ~50–70% beta/DV01. Reduce counterparty footprint."
        }
    return {
        "gross_cut_target": "60–80%",
        "net_cut_target": "50–70%",
        "cash_min": "30–50%",
        "notes": "Go to survival mode: unwind weak legs, close basis trades, hold liquidity, simplify book."
    }


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    prices: str
    positions: Optional[str]
    risk: Optional[str]
    pnl: Optional[str]
    lookback: int
    calib_start: Optional[str]
    calib_end: Optional[str]
    var_p: float
    alpha_ewma: float
    stage_pcts: str
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Crisis Mode — triggers & playbook generator")
    ap.add_argument("--prices", required=True)
    ap.add_argument("--positions", default="")
    ap.add_argument("--risk", default="")
    ap.add_argument("--pnl", default="")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--calib_start", default="")
    ap.add_argument("--calib_end", default="")
    ap.add_argument("--var_p", type=float, default=0.99)
    ap.add_argument("--alpha_ewma", type=float, default=0.94)
    ap.add_argument("--stage_pcts", default="0.80,0.92,0.97", help="Percentiles for Stage1,2,3 thresholds")
    ap.add_argument("--outdir", default="out_crisis")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load inputs
    PR = load_prices(args.prices)
    POS = load_positions(args.positions) if args.positions else pd.DataFrame()
    RISK = load_risk(args.risk) if args.risk else pd.DataFrame()
    PNL = load_pnl(args.pnl) if args.pnl else pd.DataFrame()

    # Returns & portfolio
    R = build_returns_panel(PR)
    port_r = portfolio_return(R, POS)
    ind = composite_stress_index(RISK, port_r, R, lookback=args.lookback, alpha_ewma=args.alpha_ewma)

    # VaR breaches
    var_flags = var_breaches(PNL.set_index("date")["pnl_usd"] if not PNL.empty else pd.Series(dtype=float),
                             port_r, var_p=args.var_p, alpha_ewma=args.alpha_ewma)

    # Stage thresholds & assignment
    pcts = tuple(float(x) for x in args.stage_pcts.split(","))
    thr = calibrate_stages(ind, args.calib_start or None, args.calib_end or None, pcts)
    # Auxiliary flags
    aux = {
        "aac_spike": bool(ind["aac"].tail(1).gt(ind["aac"].rolling(60).median().tail(1).values[0] + 0.15).iloc[-1]) if ind["aac"].notna().sum() else False,
        "dd_worsening": bool(ind["dd_speed"].tail(5).sum() > 0.05),
        "phinkley_alert": bool(ind["phinkley"].tail(5).sum() >= 1),
        "var_breach": (var_flags["hist_var_breach"] or var_flags["ewma_var_breach"])
    }
    latest_csi = float(ind["CSI"].dropna().iloc[-1]) if ind["CSI"].notna().any() else 0.0
    stage = assign_stage(latest_csi, thr, aux)

    # Indicators & triggers CSVs
    ind.reset_index().to_csv(outdir / "indicators.csv", index=False)
    trg = pd.DataFrame([{
        "date": ind.index.max(),
        "CSI": latest_csi,
        "thr_stage1": thr["stage1"], "thr_stage2": thr["stage2"], "thr_stage3": thr["stage3"],
        "stage": stage,
        **{f"flag_{k}": v for k,v in aux.items()},
        **{k: v for k,v in var_flags.items()}
    }])
    trg.to_csv(outdir / "triggers.csv", index=False)

    # Playbook actions (high level)
    pb = playbook_for_stage(stage)
    pb_df = pd.DataFrame([{**pb, "stage": stage, "date": ind.index.max()}])
    pb_df.to_csv(outdir / "playbook_actions.csv", index=False)

    # Hedge recommendations via min-variance ratio to portfolio returns
    cands = candidate_hedges_in_panel(PR)
    hedge_rows = []
    for h in cands:
        hret = R[h]
        hratio = min_var_hedge_ratio(port_r, hret)
        # Hedge fraction targets by stage
        mult = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9}[stage]
        h_stage = hratio * mult
        px = float(PR[h].dropna().iloc[-1]) if h in PR.columns else np.nan
        hedge_rows.append({
            "hedge_ticker": h,
            "proxy_class": classify_proxy(h),
            "hedge_ratio_r": hratio,
            "stage_mult": mult,
            "hedge_ratio_r_stage": h_stage,
            "last_price": px,
            "note": "Apply to current portfolio notional of 1.0 (ratios are in return space). If you track portfolio MV, multiply by MV/price for units."
        })
    hedges = pd.DataFrame(hedge_rows)
    hedges.to_csv(outdir / "hedge_recs.csv", index=False)

    # Simple scenarios P&L (portfolio + hedges, using last-window betas via covariance)
    scen_rows = []
    # Build rough factor proxies from candidate hedges if present
    fac_map = {"EQUITY":["SPY","QQQ"], "RATES":["TLT","IEF"], "CREDIT":["HYG","LQD"], "USD":["UUP"], "GOLD":["GLD"]}
    fac_ret = {}
    for f, lst in fac_map.items():
        lst = [x for x in lst if x in R.columns]
        if lst:
            fac_ret[f] = R[lst].mean(axis=1)
    # Estimate beta of portfolio to each factor
    betas = {}
    for f, rr in fac_ret.items():
        a = pd.concat([port_r, rr], axis=1, join="inner").dropna()
        if a.shape[0] >= 60:
            cov = np.cov(a.iloc[:,0], a.iloc[:,1], ddof=0)[0,1]
            varf = np.var(a.iloc[:,1], ddof=0)
            betas[f] = cov / varf if varf>0 else 0.0
        else:
            betas[f] = 0.0

    # Map shocks to proxy returns (very rough): equity % ≈ return; USD % ≈ return; rates bp via duration proxy
    # For rates: TLT ~ 18y duration ⇒ -20bp ≈ +0.20*18% ≈ +3.6% (rule of thumb: ΔP ≈ -D*Δy)
    def rates_bp_to_tlt_ret(bp: float) -> float:
        return - (-bp/10000.0) * 18.0  # +bp up → price down; we want Δy negative => positive return

    for s in scenario_shocks():
        # Portfolio factor-only estimate
        port_est = (
            betas.get("EQUITY",0.0) * (s["equity_pct"]/100.0) +
            betas.get("USD",0.0)    * (s["usd_pct"]/100.0) +
            betas.get("RATES",0.0)  * (rates_bp_to_tlt_ret(s["rates_bp"])) +
            betas.get("CREDIT",0.0) * (-s["credit_pct"]/100.0)  # credit "drawdown" negative return
        )
        # Hedged return using stage hedge ratios
        hedge_ret_total = 0.0
        for _, hr in hedges.iterrows():
            h = hr["hedge_ticker"]; hratio_stage = float(hr["hedge_ratio_r_stage"])
            if hr["proxy_class"] == "EQUITY":  hret = s["equity_pct"]/100.0
            elif hr["proxy_class"] == "USD":   hret = s["usd_pct"]/100.0
            elif hr["proxy_class"] == "RATES": hret = rates_bp_to_tlt_ret(s["rates_bp"])
            elif hr["proxy_class"] == "CREDIT":hret = -s["credit_pct"]/100.0
            elif hr["proxy_class"] == "GOLD":  hret = +0.5 * (s["equity_pct"]/100.0 * -1)  # gold up ~ 0.5*eq down (toy)
            else: hret = 0.0
            hedge_ret_total += hratio_stage * hret
        total = port_est + hedge_ret_total
        scen_rows.append({"scenario": s["name"], "stage": stage, "portfolio_ret_est": port_est, "hedge_ret_est": hedge_ret_total, "total_ret_est": total})

    scen_df = pd.DataFrame(scen_rows)
    scen_df.to_csv(outdir / "scenario_pnl.csv", index=False)

    # Checklist
    checklist = [
        "1) Confirm stage & triggers; log rationale.",
        "2) Liquidity: raise cash to target; cut illiquids/crowded names first.",
        "3) Hedge: deploy per hedge_recs.csv (staged sizing). Use liquid proxies.",
        "4) Counterparty: reduce concentration; check margin/collateral; widen haircuts on receivables.",
        "5) Basis & funding: close fragile basis trades; avoid term funding mismatch.",
        "6) Stop-loss governance: convert soft stops → hard on weakest legs.",
        "7) Communication: send note with exposures, actions, and next review time.",
        "8) Monitoring cadence: tighten (intra-day if needed).",
        "9) Post-mortem hooks: tag trades/actions for later review."
    ]
    (outdir / "checklist.txt").write_text("\n".join(checklist), encoding="utf-8")

    # Summary
    summary = {
        "date": str(ind.index.max().date()) if len(ind.index) else None,
        "stage": stage,
        "CSI_latest": latest_csi,
        "thresholds": thr,
        "aux_flags": aux,
        "available_hedges": cands,
        "betas": betas
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Console
    print("== Crisis Mode ==")
    print(f"Stage {stage} | CSI {latest_csi:+.2f} | thresholds: S1 {thr['stage1']:.2f}, S2 {thr['stage2']:.2f}, S3 {thr['stage3']:.2f}")
    print("Aux flags:", aux)
    if hedges.shape[0]:
        print("Hedge candidates & stage ratios:")
        for _, r in hedges.iterrows():
            print(f"  {r['hedge_ticker']:<4} ({r['proxy_class']:<6}) h*={r['hedge_ratio_r']:+.3f} → stage {r['hedge_ratio_r_stage']:+.3f}")
    print("Outputs in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
# EOF