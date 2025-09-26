#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keiretsu_discount.py — Cross-shareholdings & “keiretsu discount” for Japan equities
-----------------------------------------------------------------------------------

What this does
==============
Builds a **quarterly firm panel** for Japan and estimates whether firms with
high cross-shareholdings / keiretsu affiliation trade at valuation discounts
after controlling for fundamentals (size, ROE, leverage, growth) with **two-way
fixed effects (firm & year)** and firm-clustered SEs. Also:

1) Constructs a **time-varying ownership network** (issuer←holder edges),
   computes cross-holding ratios, within-group shares, free-float proxies,
   and network centralities (in-degree, eigenvector, PageRank).
2) Measures valuation: **P/B, Tobin’s Q, EV/EBITDA**; builds **log P/B** target.
3) Estimates **“keiretsu discount”**: coef on cross-holding ratio and/or keiretsu
   dummy in log(P/B) regression with 2-way FE + clustered SEs.
4) **Event study** on cross-holding unwinds/announcements (daily CAR ±N).
5) **Scenario**: re-rating if cross-holding ratio falls by X pp (firm-level uplift).

Inputs (CSV; headers flexible, case-insensitive)
------------------------------------------------
--fund fundamentals.csv   REQUIRED (quarterly or higher; we align to quarter-end)
  Columns (any subset; best-effort mapping):
    date, firm[, ticker, permno, id], sector[, industry]
    price[, close], shares_out[, shares], mkt_cap[, market_cap]
    book_equity[, be], total_assets[, assets], debt[, total_debt], cash[, cash_eq]
    ebitda, sales[, revenue]
    roe[, net_income/avg_equity], leverage[, debt/assets], sales_growth[, g]
  Notes: If mkt_cap missing, computed = price * shares_out.
        If roe/leverage/growth missing, we’ll compute simple proxies from columns available.

--ownership ownership.csv  REQUIRED (edge list; time-varying OK; daily/monthly/qtrly)
  Columns:
    date, holder, issuer, pct[, stake_pct, weight], type[, holder_type] optional
  Notes: pct in **decimal** (0.10=10%) or **%**; auto-detected.
        We compute crossholding ratios per issuer per date.

--map groups.csv           OPTIONAL (firm→group/keiretsu mapping)
  Columns:
    firm[, ticker, id], group[, keiretsu]
  Notes: Examples: Mitsubishi, Mitsui, Sumitomo, Fuyo, Sanwa, Dai-Ichi Kangyo, Independent

--prices_daily prices.csv  OPTIONAL (for event study; daily)
  Columns:
    date, firm[, ticker], ret[, return], close[, px] (ret preferred; else computed)

--events events.csv        OPTIONAL (unwind/sale/buyback announcements)
  Columns:
    date, firm[, ticker], type[, label], value[, shares_sold, stake_pp] optional
  Notes: We’ll compute simple **CAR** of daily returns around each event.

Key options (CLI)
-----------------
--freq quarterly             Output panel frequency (quarterly; monthly allowed)
--sig_pct 5                  Threshold (%) to treat a link as “significant” cross-holding
--window_car 10              Event-study ±window (trading days)
--scenario_unwind_pp 5       Shock: reduce each firm’s cross-holding ratio by N percentage points
--outdir out_keiretsu        Output directory

Outputs
-------
panel_firm.csv               Quarterly firm panel with fundamentals, valuations & ownership features
network_metrics.csv          Per-date firm network metrics (in_deg, eigenvector, PageRank, within_group share)
reg_results.csv              2-way FE regression table (coef, SE, t, p)
discount_residuals.csv       Firm-quarter residuals (implied discount vs controls)
event_study.csv              CAR around events (if provided)
scenario_rerating.csv        Predicted re-rating & value uplift under unwind shock
summary.json, config.json    Metadata & configuration echo

Dependencies
------------
pandas, numpy, networkx (for centrality). No statsmodels required; we implement OLS + cluster-robust SE.

DISCLAIMER
----------
Research toolkit. Check your column mappings, units (%, JPY), and sample filters.
This is NOT investment advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


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
    for cand in cands:
        key = cand.lower()
        for c in df.columns:
            if key in str(c).lower(): return c
    return None

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def qtr_end(s: pd.Series) -> pd.Series:
    # align to quarter-end (Japan fiscal years often Mar-end but we stick to calendar Q)
    return to_dt(s) + pd.offsets.QuarterEnd(0)

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def winsor(x: pd.Series, p: float=0.005) -> pd.Series:
    if x.isna().all(): return x
    lo, hi = x.quantile(p), x.quantile(1-p)
    return x.clip(lower=lo, upper=hi)

def zscore(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w, min_periods=max(6, w//3)).mean()
    sd = s.rolling(w, min_periods=max(6, w//3)).std(ddof=0)
    return (s - mu) / (sd + 1e-12)

def log_safe(x):
    return np.log(np.where(x>0, x, np.nan))

# OLS + clustered (by firm) SEs
def ols_beta(X: np.ndarray, y: np.ndarray):
    XTX = X.T @ X
    XTX_inv = np.linalg.pinv(XTX)
    beta = XTX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XTX_inv

def cluster_se(X: np.ndarray, resid: np.ndarray, XTX_inv: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """
    Firm-clustered (Arellano) covariance:
      (X'X)^{-1} [Σ_g X_g' u_g u_g' X_g] (X'X)^{-1}
    """
    k = X.shape[1]
    meat = np.zeros((k,k))
    # group index
    uniq = np.unique(groups)
    for g in uniq:
        mask = (groups==g)
        Xg = X[mask,:]
        ug = resid[mask,:]
        meat += (Xg.T @ ug) @ (ug.T @ Xg)
    cov = XTX_inv @ meat @ XTX_inv
    # small-sample correction (optional)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


# ----------------------------- loaders -----------------------------

def load_fundamentals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    d = ncol(df,"date"); f = ncol(df,"firm","ticker","id"); sec = ncol(df,"sector","industry")
    if not (d and f): raise ValueError("fundamentals.csv needs date and firm columns.")
    df = df.rename(columns={d:"date", f:"firm"})
    if sec: df = df.rename(columns={sec:"sector"})
    df["date"] = qtr_end(df["date"])
    # map numeric columns
    ren = {
        ncol(df,"price","close"): "price",
        ncol(df,"shares_out","shares"): "shares",
        ncol(df,"mkt_cap","market_cap"): "mkt_cap",
        ncol(df,"book_equity","be"): "book_equity",
        ncol(df,"total_assets","assets"): "assets",
        ncol(df,"debt","total_debt"): "debt",
        ncol(df,"cash","cash_eq","cash_equiv"): "cash",
        ncol(df,"ebitda"): "ebitda",
        ncol(df,"sales","revenue"): "sales",
        ncol(df,"roe"): "roe",
        ncol(df,"leverage"): "leverage",
        ncol(df,"sales_growth","g"): "sales_growth"
    }
    for src, tgt in ren.items():
        if src: df = df.rename(columns={src:tgt})
    for k in ["price","shares","mkt_cap","book_equity","assets","debt","cash","ebitda","sales","roe","leverage","sales_growth"]:
        if k in df.columns: df[k] = safe_num(df[k])
    # compute missing
    if "mkt_cap" not in df.columns and {"price","shares"}.issubset(df.columns):
        df["mkt_cap"] = df["price"] * df["shares"]
    # quick ratios if absent
    if "roe" not in df.columns and {"sales","ebitda","book_equity"}.issubset(df.columns):
        # weak proxy when NI missing
        df["roe"] = (0.7*df["ebitda"]) / (df["book_equity"].replace(0,np.nan))
    if "leverage" not in df.columns and {"debt","assets"}.issubset(df.columns):
        df["leverage"] = df["debt"] / df["assets"].replace(0,np.nan)
    # valuations
    df["pb"] = df["mkt_cap"] / df["book_equity"].replace(0,np.nan)
    if {"mkt_cap","debt","cash","assets"}.issubset(df.columns):
        df["tobins_q"] = (df["mkt_cap"] + df["debt"] - df["cash"]) / df["assets"].replace(0,np.nan)
    if {"mkt_cap","debt","cash","ebitda"}.issubset(df.columns):
        df["ev_ebitda"] = (df["mkt_cap"] + df["debt"] - df["cash"]) / df["ebitda"].replace(0,np.nan)
    # log transforms
    df["log_pb"] = winsor(log_safe(df["pb"]))
    df["log_assets"] = winsor(log_safe(df["assets"]))
    df["log_sales"] = winsor(log_safe(df["sales"])) if "sales" in df.columns else np.nan
    # year for FE
    df["year"] = df["date"].dt.year
    return df.sort_values(["date","firm"])

def load_ownership(path: str, sig_pct: float) -> pd.DataFrame:
    """Edge list; compute per-issuer crossholding ratios each date."""
    df = pd.read_csv(path)
    d = ncol(df,"date"); h = ncol(df,"holder"); i = ncol(df,"issuer","firm"); p = ncol(df,"pct","stake_pct","weight")
    typ = ncol(df,"type","holder_type")
    if not (d and h and i and p): raise ValueError("ownership.csv needs date, holder, issuer, pct.")
    df = df.rename(columns={d:"date", h:"holder", i:"issuer", p:"pct"})
    if typ: df = df.rename(columns={typ:"type"})
    df["date"] = qtr_end(df["date"])
    df["pct"] = safe_num(df["pct"])
    # auto-detect % units
    guess = df["pct"].dropna().median()
    if np.isfinite(guess) and guess > 1.0:
        df["pct"] = df["pct"] / 100.0
    # significant links
    sig = float(sig_pct)/100.0
    df["is_sig"] = (df["pct"] >= sig).astype(int)
    return df.sort_values(["date","issuer","holder"])

def load_map(path: Optional[str]) -> Dict[str,str]:
    if not path: return {}
    df = pd.read_csv(path)
    f = ncol(df,"firm","ticker","id"); g = ncol(df,"group","keiretsu")
    if not (f and g): return {}
    return {str(r[f]): str(r[g]) for _, r in df.iterrows()}

def load_prices_daily(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); f = ncol(df,"firm","ticker","id")
    if not (d and f): raise ValueError("prices_daily needs date and firm.")
    df = df.rename(columns={d:"date", f:"firm"})
    df["date"] = to_dt(df["date"])
    r = ncol(df,"ret","return")
    px = ncol(df,"close","px","price")
    if r:
        df = df.rename(columns={r:"ret"})
        df["ret"] = safe_num(df["ret"])
    elif px:
        df = df.rename(columns={px:"px"})
        df = df.sort_values(["firm","date"])
        df["ret"] = df.groupby("firm")["px"].pct_change()
    else:
        raise ValueError("prices_daily must have ret or close/px.")
    return df[["date","firm","ret"]].dropna().sort_values(["firm","date"])

def load_events(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    d = ncol(df,"date"); f = ncol(df,"firm","ticker","id"); t = ncol(df,"type"); v = ncol(df,"value","shares_sold","stake_pp"); l = ncol(df,"label","event")
    if not (d and f): raise ValueError("events.csv needs date and firm.")
    df = df.rename(columns={d:"date", f:"firm"})
    if t: df = df.rename(columns={t:"type"})
    if v: df = df.rename(columns={v:"value"})
    if l: df = df.rename(columns={l:"label"})
    df["date"] = to_dt(df["date"])
    if "type" not in df.columns: df["type"] = "UNWIND"
    if "label" not in df.columns: df["label"] = df["type"]
    return df.sort_values(["firm","date"])


# ----------------------------- ownership → features & network -----------------------------

def ownership_features(OWN: pd.DataFrame, FUND: pd.DataFrame, group_map: Dict[str,str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-issuer features each date:
      cross_ratio = Σ_j (is_sig * pct)               # strong cross-holdings
      all_ratio   = Σ_j pct                          # all cross-holdings
      within_group_ratio = Σ_{j in same group} pct
      n_sig_holders, n_all_holders
    And network metrics from significant edges.
    """
    if OWN.empty: return pd.DataFrame(), pd.DataFrame()

    # attach groups
    if group_map:
        OWN["holder_group"] = OWN["holder"].map(group_map)
        OWN["issuer_group"] = OWN["issuer"].map(group_map)
        OWN["within_group"] = (OWN["holder_group"].fillna("_") == OWN["issuer_group"].fillna("^")).astype(int)
    else:
        OWN["holder_group"] = np.nan
        OWN["issuer_group"] = np.nan
        OWN["within_group"] = 0

    # Issuer-aggregates per date
    agg_all = (OWN.groupby(["date","issuer"], as_index=False)
                 .agg(all_ratio=("pct","sum"),
                      n_all_holders=("pct","size")))
    agg_sig = (OWN[OWN["is_sig"]==1].groupby(["date","issuer"], as_index=False)
                 .agg(cross_ratio=("pct","sum"),
                      n_sig_holders=("pct","size")))
    agg_within = (OWN.groupby(["date","issuer"], as_index=False)
                    .apply(lambda g: pd.Series({"within_group_ratio": float((g["pct"] * g["within_group"]).sum())}))
                    .reset_index())

    FEAT = (agg_all.merge(agg_sig, on=["date","issuer"], how="left")
                 .merge(agg_within, on=["date","issuer"], how="left"))
    for c in ["cross_ratio","n_sig_holders","within_group_ratio"]:
        if c in FEAT.columns: FEAT[c] = FEAT[c].fillna(0.0)

    # free-float (proxy): 1 - all_ratio (bounded)
    FEAT["free_float_proxy"] = (1.0 - FEAT["all_ratio"]).clip(lower=0.0, upper=1.0)

    FEAT = FEAT.rename(columns={"issuer":"firm"})

    # Network metrics (significant edges only)
    rows = []
    for dt, g in OWN[OWN["is_sig"]==1].groupby("date"):
        G = nx.DiGraph()
        # add edges weight=pct
        for _, r in g.iterrows():
            G.add_edge(str(r["holder"]), str(r["issuer"]), weight=float(r["pct"]))
        if G.number_of_edges()==0: 
            continue
        # centralities
        try:
            ev = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            ev = {n: np.nan for n in G.nodes()}
        pr = nx.pagerank(G, weight="weight") if G.number_of_edges()>0 else {}
        indeg = dict(G.in_degree(weight="weight"))
        outdeg = dict(G.out_degree(weight="weight"))
        for n in set(G.nodes()):
            rows.append({
                "date": dt, "firm": n,
                "eigencent": float(ev.get(n, np.nan)),
                "pagerank": float(pr.get(n, np.nan)),
                "in_deg_w": float(indeg.get(n, 0.0)),
                "out_deg_w": float(outdeg.get(n, 0.0)),
            })
    NET = pd.DataFrame(rows).sort_values(["date","firm"])
    return FEAT, NET


# ----------------------------- panel build -----------------------------

def build_panel(FUND: pd.DataFrame, FEAT: pd.DataFrame, NET: pd.DataFrame, group_map: Dict[str,str]) -> pd.DataFrame:
    P = FUND.copy()
    if not FEAT.empty:
        P = P.merge(FEAT, on=["date","firm"], how="left")
    if not NET.empty:
        P = P.merge(NET, on=["date","firm"], how="left")
    if group_map:
        P["group"] = P["firm"].map(group_map)
        P["is_keiretsu"] = (~P["group"].isna()) & (P["group"].str.lower().ne("independent"))
        P["is_keiretsu"] = P["is_keiretsu"].astype(int)
    else:
        P["group"] = np.nan
        P["is_keiretsu"] = 0

    # fill zeros for missing ownership metrics
    for c in ["all_ratio","cross_ratio","within_group_ratio","n_all_holders","n_sig_holders","free_float_proxy",
              "eigencent","pagerank","in_deg_w","out_deg_w"]:
        if c in P.columns: P[c] = P[c].fillna(0.0)

    # sanity filters
    P = P[~P["log_pb"].isna()].copy()
    # winsor controls
    for c in ["roe","leverage","sales_growth","log_assets"]:
        if c in P.columns: P[c] = winsor(P[c])
    return P.sort_values(["date","firm"])


# ----------------------------- 2-way FE regression -----------------------------

def twoway_fe_regression(P: pd.DataFrame,
                         yvar: str="log_pb",
                         xvars: List[str]=["cross_ratio","is_keiretsu","free_float_proxy","roe","leverage","sales_growth","log_assets"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    y_it = β·X_it + α_i + τ_t + ε_it
    Implement via “within” transformation: de-mean by firm and year.
    Clustered SEs by firm.
    Returns (table, residuals dataframe).
    """
    df = P.dropna(subset=[yvar] + [c for c in xvars if c in P.columns]).copy()
    xvars = [c for c in xvars if c in df.columns]
    if df.empty or not xvars: 
        return pd.DataFrame(), pd.DataFrame()

    # keys
    firm = df["firm"].astype(str).values
    year = df["year"].astype(int).values

    # within transform: x* = x - x_i - x_t + x_bar ; y* similarly
    def within_transform(v: pd.Series) -> np.ndarray:
        v = v.values.astype(float)
        # firm means
        vi = pd.Series(v).groupby(pd.Series(firm)).transform("mean").values
        # year means
        vt = pd.Series(v).groupby(pd.Series(year)).transform("mean").values
        # overall mean
        vbar = np.nanmean(v)
        return v - vi - vt + vbar

    Y = within_transform(df[yvar])
    X = np.column_stack([within_transform(df[x]) for x in xvars])
    # drop rows where all X* are ~0 (non-varying within firm-year)
    keep = np.isfinite(Y) & np.isfinite(X).all(axis=1) & (np.abs(X).sum(axis=1) > 1e-12)
    Y = Y[keep].reshape(-1,1)
    X = X[keep,:]
    groups = firm[keep]

    # OLS (no intercept after within transform)
    beta, resid, XTX_inv = ols_beta(X, Y)

    # Clustered SEs (by firm)
    se = cluster_se(X, resid, XTX_inv, groups=groups)
    coefs = []
    for i, nm in enumerate(xvars):
        b = float(beta[i,0])
        s = float(se[i])
        t = (b/s) if s>0 else np.nan
        coefs.append({"var": nm, "coef": b, "se": s, "t_stat": t})
    table = pd.DataFrame(coefs)

    # residuals: implied discount unexplained by controls & FE
    # ε_it = y* - X*β (within space). For interpretability, we report ε_it (approx)
    eps = (Y - X @ beta).ravel()
    R = df.iloc[keep, :][["date","firm","group"] + ([yvar] if yvar in df.columns else [])].copy()
    R["residual"] = eps
    R["implied_discount_%"] = (np.exp(eps) - 1.0) * 100.0  # in P/B space: residual ≈ % mispricing
    return table, R.sort_values(["date","firm"])


# ----------------------------- event study (daily) -----------------------------

def event_study_daily(DAILY: pd.DataFrame, EV: pd.DataFrame, window: int=10) -> pd.DataFrame:
    if DAILY.empty or EV.empty: return pd.DataFrame()
    out = []
    dser = DAILY.set_index(["firm","date"])["ret"].sort_index()
    for _, e in EV.iterrows():
        f = str(e["firm"]); dt = pd.Timestamp(e["date"])
        if (f, dt) not in dser.index:
            # snap to next available date for that firm
            idx = dser.loc[f].index if f in dser.index.get_level_values(0) else None
            if idx is None or len(idx)==0: continue
            pos = idx.searchsorted(dt)
            if pos >= len(idx): continue
            dt = idx[pos]
        # window
        idxf = dser.loc[f].index
        i = idxf.get_loc(dt)
        L = max(0, i-window); R = min(len(idxf)-1, i+window)
        car = float(dser.loc[(f, idxf[L:R+1])].sum())
        out.append({"firm": f, "event_date": str(dt.date()),
                    "type": e.get("type","EVENT"), "label": e.get("label", e.get("type","EVENT")),
                    "CAR_ret": car})
    return pd.DataFrame(out).sort_values(["firm","event_date","type"])


# ----------------------------- scenario: unwind → re-rating -----------------------------

def scenario_rerating(P: pd.DataFrame, beta_cross: float, unwind_pp: float=5.0) -> pd.DataFrame:
    """
    Δ log(P/B) = β_cross * Δ cross_ratio
    If unwind_pp > 0, cross_ratio decreases by unwind_pp percentage points.
    Re-rating % = exp(Δ log PB) - 1
    """
    if P.empty or not np.isfinite(beta_cross):
        return pd.DataFrame()
    shock = - float(unwind_pp) / 100.0
    last = P.sort_values("date").groupby("firm").tail(1)
    last = last[["date","firm","group","mkt_cap","pb","cross_ratio","free_float_proxy"]].copy()
    last["delta_log_pb"] = beta_cross * shock
    last["re_rating_%"] = (np.exp(last["delta_log_pb"]) - 1.0) * 100.0
    last["value_uplift_jpy"] = last["mkt_cap"] * (np.exp(last["delta_log_pb"]) - 1.0)
    last["new_cross_ratio"] = (last["cross_ratio"] + shock).clip(lower=0.0)
    last["delta_free_float_pp"] = (-shock) * 100.0  # float rises when cross-holdings fall
    return last.sort_values("value_uplift_jpy", ascending=False)


# ----------------------------- CLI / Orchestration -----------------------------

@dataclass
class Config:
    fundamentals: str
    ownership: str
    mapfile: Optional[str]
    prices_daily: Optional[str]
    events: Optional[str]
    freq: str
    sig_pct: float
    window_car: int
    scenario_unwind_pp: float
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Keiretsu discount: cross-shareholdings, valuation & re-rating scenarios")
    ap.add_argument("--fund", dest="fundamentals", required=True, help="fundamentals.csv")
    ap.add_argument("--ownership", required=True, help="ownership.csv")
    ap.add_argument("--map", dest="mapfile", default="", help="groups.csv (firm→keiretsu)")
    ap.add_argument("--prices_daily", default="", help="daily prices for event study")
    ap.add_argument("--events", default="", help="events.csv for unwinds")
    ap.add_argument("--freq", default="quarterly", choices=["quarterly","monthly"])
    ap.add_argument("--sig_pct", type=float, default=5.0, help="Significant stake threshold (%)")
    ap.add_argument("--window_car", type=int, default=10, help="Event-study window ±N days")
    ap.add_argument("--scenario_unwind_pp", type=float, default=5.0, help="Reduce cross_ratio by N pp")
    ap.add_argument("--outdir", default="out_keiretsu")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    FUND = load_fundamentals(args.fundamentals)
    OWN  = load_ownership(args.ownership, sig_pct=float(args.sig_pct))
    MAP  = load_map(args.mapfile) if args.mapfile else {}
    DAILY= load_prices_daily(args.prices_daily) if args.prices_daily else pd.DataFrame()
    EV   = load_events(args.events) if args.events else pd.DataFrame()

    # Ownership → features + network
    FEAT, NET = ownership_features(OWN, FUND, MAP)

    # Panel
    P = build_panel(FUND, FEAT, NET, MAP)
    if P.empty:
        raise ValueError("Panel is empty — check inputs and column mappings.")
    P.to_csv(outdir / "panel_firm.csv", index=False)

    # Regression (2-way FE on log(P/B))
    xvars = ["cross_ratio","is_keiretsu","free_float_proxy","roe","leverage","sales_growth","log_assets","eigencent"]
    REG, RES = twoway_fe_regression(P, yvar="log_pb", xvars=xvars)
    if not REG.empty: REG.to_csv(outdir / "reg_results.csv", index=False)
    if not RES.empty: RES.to_csv(outdir / "discount_residuals.csv", index=False)

    # Pick β_cross for scenario
    if not REG.empty and "cross_ratio" in REG["var"].values:
        beta_cross = float(REG.loc[REG["var"]=="cross_ratio","coef"].iloc[0])
    else:
        beta_cross = np.nan

    # Scenario: unwind
    SC = scenario_rerating(P, beta_cross=beta_cross, unwind_pp=float(args.scenario_unwind_pp))
    if not SC.empty: SC.to_csv(outdir / "scenario_rerating.csv", index=False)

    # Event study
    ES = event_study_daily(DAILY, EV, window=int(args.window_car)) if (not DAILY.empty and not EV.empty) else pd.DataFrame()
    if not ES.empty: ES.to_csv(outdir / "event_study.csv", index=False)

    # Summary
    latest = P.sort_values("date").groupby("firm").tail(1)
    n_firms = int(latest["firm"].nunique())
    share_cross_med = float(latest["cross_ratio"].median()) if "cross_ratio" in latest.columns else None
    pb_gap = None
    if "is_keiretsu" in latest.columns and "pb" in latest.columns:
        pb_k = latest[latest["is_keiretsu"]==1]["pb"].median()
        pb_i = latest[latest["is_keiretsu"]==0]["pb"].median()
        pb_gap = float(np.log(pb_k) - np.log(pb_i)) if np.isfinite(pb_k) and np.isfinite(pb_i) and pb_k>0 and pb_i>0 else None

    summary = {
        "sample": {
            "start": str(P["date"].min().date()),
            "end": str(P["date"].max().date()),
            "periods": int(P["date"].nunique()),
            "freq": "quarterly"
        },
        "firms": n_firms,
        "median_cross_ratio": share_cross_med,
        "raw_log_pb_gap_keiretsu_minus_independent": pb_gap,
        "regression": REG.to_dict(orient="records")[:8] if not REG.empty else [],
        "beta_cross_ratio": float(beta_cross) if np.isfinite(beta_cross) else None,
        "scenario_unwind_pp": float(args.scenario_unwind_pp),
        "outputs": {
            "panel_firm": "panel_firm.csv",
            "network_metrics": "network_metrics.csv",
            "reg_results": "reg_results.csv" if not REG.empty else None,
            "discount_residuals": "discount_residuals.csv" if not RES.empty else None,
            "event_study": "event_study.csv" if not ES.empty else None,
            "scenario_rerating": "scenario_rerating.csv" if not SC.empty else None
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Save network metrics separately (already merged in panel, but export tidy too)
    if not NET.empty: NET.to_csv(outdir / "network_metrics.csv", index=False)

    # Config echo
    cfg = asdict(Config(
        fundamentals=args.fundamentals, ownership=args.ownership, mapfile=(args.mapfile or None),
        prices_daily=(args.prices_daily or None), events=(args.events or None),
        freq=args.freq, sig_pct=float(args.sig_pct), window_car=int(args.window_car),
        scenario_unwind_pp=float(args.scenario_unwind_pp), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Keiretsu Discount Toolkit ==")
    print(f"Sample: {summary['sample']['start']} → {summary['sample']['end']} | Firms: {n_firms}")
    if not REG.empty:
        b = float(REG.loc[REG['var']=='cross_ratio','coef'].iloc[0]) if 'cross_ratio' in REG['var'].values else np.nan
        print(f"β_cross_ratio (log P/B): {b:+.3f}  (neg ⇒ higher cross-holdings → lower P/B)")
    if not SC.empty:
        total_uplift = float(SC["value_uplift_jpy"].sum())
        print(f"Scenario (−{args.scenario_unwind_pp:.1f}pp cross-holdings): total value uplift ≈ ¥{total_uplift:,.0f}")
    if not ES.empty:
        print(f"Event study written ({len(ES)} rows, ±{args.window_car}d).")
    print("Artifacts in:", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
