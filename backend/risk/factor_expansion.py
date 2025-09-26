#!/usr/bin/env python3
"""
factor_expansion.py — Expand raw features into a richer factor set, orthogonalize/neutralize,
run cross-sectional regressions, compute ICs, and build long-short portfolios.

Inputs (wide CSVs; rows=time; columns=tickers):
  --returns         required  : realized returns (same frequency as features), e.g. forward 1m return
  --features        required  : base features/exposures at time t (will be expanded)
  --mcap            optional  : market cap (for WLS and portfolio weighting)
  --groups          optional  : categorical groups (industry/country) in wide CSV (string); one group per asset per date
  --universe        optional  : 0/1 tradable mask (wide CSV)

Expansion options:
  --poly-deg 2                : polynomial degree (>=1). 1 means keep as-is (with standardization)
  --interactions              : include pairwise interactions (x_i * x_j for i<j)
  --log-cols "mcap,volume"    : log-transform these base feature columns before expansion (if >0)
  --winsor q=0.01             : winsorize cross-section to given quantile (e.g., 0.01); 0 disables
  --mad-cap 5.0               : MAD cap (optional; applied after winsor if >0)
  --zscore                    : z-score per date (default on)
  --orth "gs"                 : within-date Gram-Schmidt orthogonalization of expanded factors
  --neutralize "group"        : de-mean factors within group (requires --groups)
  --nan-policy "drop|fill0"   : how to handle NaNs post expansion (default: drop asset that day)

Modeling:
  --reg "ols|wls"             : cross-sectional regression of future returns ~ expanded factors (+ intercept)
  --target-horizon 1          : if returns are concurrent, shift by +h steps to use future return
  --weights "mcap"            : use mcap as weights for WLS (ignored for OLS)
  --add-constant              : include intercept in regression (default on)

ICs & Portfolios:
  --ic-method "spearman|pearson"  : Information Coefficient method
  --quantiles 10                  : number of quantiles for portfolios
  --score "factor_name|composite" : which score to rank for portfolios
  --composite "f1:1,f2:0.5"       : weights for composite score (if score==composite)
  --long-short                    : build top-minus-bottom (QK - Q1) portfolio
  --hold 1                        : holding period in steps (no overlap netting; simple roll)

Outputs (to --outdir):
  expanded_factors.parquet    : expanded & processed factor matrix (panel index: Date, Asset)
  factor_returns.csv          : time series of cross-sectional betas (factor premia)
  ic_timeseries.csv           : per factor IC per date
  ic_summary.csv              : mean IC, IR, hit rate
  ls_timeseries.csv           : long-short portfolio return (if requested)
  exposures_snapshot.csv      : last-date snapshot (for quick inspection)
  config.json                 : run configuration

Usage:
  python factor_expansion.py --returns ret.csv --features feats.csv --mcap mcap.csv \
    --poly-deg 2 --interactions --neutralize group --groups industry.csv \
    --reg wls --weights mcap --ic-method spearman --long-short --outdir out_fx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# I/O helpers
# ----------------------------
def read_wide_csv(path: str, dtype: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].lower() in ("date", "time", "t"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    if dtype == "str":
        # ensure string dtypes in cells (groups)
        for c in df.columns:
            df[c] = df[c].astype(str)
    return df.sort_index()


def align_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Intersect on index and columns across all DataFrames
    idx = None
    cols = None
    for df in frames.values():
        idx = df.index if idx is None else idx.intersection(df.index)
        cols = set(df.columns) if cols is None else cols.intersection(df.columns)
    cols = sorted(list(cols))#type:ignore
    out = {}
    for k, df in frames.items():
        out[k] = df.loc[idx, cols].copy()#type:ignore
    return out


# ----------------------------
# Cross-sectional transforms
# ----------------------------
def winsorize_cs(x: pd.Series, q: float) -> pd.Series:
    if q <= 0:
        return x
    lo, hi = x.quantile(q), x.quantile(1 - q)
    return x.clip(lower=lo, upper=hi)


def mad_cap_cs(x: pd.Series, mad_cap: float) -> pd.Series:
    if mad_cap <= 0:
        return x
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return x
    z = (x - med) / (1.4826 * mad)
    return med + z.clip(lower=-mad_cap, upper=mad_cap) * (1.4826 * mad)


def zscore_cs(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def neutralize_group_cs(x: pd.Series, groups: pd.Series) -> pd.Series:
    # de-mean within groups
    if groups is None or groups.empty:
        return x
    gmu = x.groupby(groups).transform("mean")
    return x - gmu


def gram_schmidt(X: pd.DataFrame) -> pd.DataFrame:
    # Orthogonalize columns (within-date) via modified Gram–Schmidt; keep scale (zscore recommended beforehand)
    Q = pd.DataFrame(index=X.index)
    cols = list(X.columns)
    for i, c in enumerate(cols):
        v = X[c].copy()
        for j in range(i):
            u = Q[cols[j]]
            coeff = np.nan_to_num((v * u).sum() / (u * u).sum() if (u * u).sum() != 0 else 0.0)
            v = v - coeff * u
        Q[c] = v
    return Q


# ----------------------------
# Expansion
# ----------------------------
def expand_features(
    F: pd.DataFrame,
    poly_deg: int = 2,
    interactions: bool = True,
    log_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    # F is wide: index=date, columns=features per asset? -> Actually features are per asset per column.
    # We treat columns as assets; rows=time; each cell is base factor value for (t, asset) for each feature?
    # To support multiple base features, stack columns as MultiIndex: feature names separated by "::".
    # Expect input with columns like "mom", "value", "quality" *per asset*. But usually it's per-asset columns…
    # Therefore we assume FEATURES file is "panel-style wide by asset but multi-feature via column prefix":
    # e.g., columns: AAPL|mom, AAPL|value, MSFT|mom, ...
    # To avoid that complexity, we assume features.csv is "long-ish wide": columns are features, index are dates, and rows per asset? Not feasible.
    # Simpler approach: Expect features.csv to be a concatenation per feature group per asset isn't common.
    # We'll implement: features.csv is a panel in tidy format is not available. So:
    # We will require that feature file is a concatenation of features per column and assets in rows won't work.
    # -> New plan: We'll read FEATURES as "panel" via MultiIndex columns: top-level feature, second-level asset.
    # However CSV can't store MI easily. So we accept one of two layouts:
    # (1) Layout A (default): one CSV per feature passed via --features repeated (NOT supported here).
    # (2) Layout B (implemented): columns are of the form "<feature>::<asset>".
    #
    # We'll split on '::' to reshape into a panel (Date, Asset) x (Features).
    raise NotImplementedError(
        "This script expects the features CSV to have columns named '<feature>::<asset>'. "
        "Example: 'value::AAPL','momentum::AAPL','value::MSFT',... with Date index.\n"
        "See build_panel() for reshaping. Use --help for details."
    )


def build_panel_from_prefixed_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Convert wide features with columns '<feature>::<asset>' into a panel:
    returns DataFrame with MultiIndex (Date, Asset) rows and columns = [features].
    """
    feats = []
    assets = set()
    for c in df.columns:
        if "::" not in c:
            raise ValueError(f"Feature column '{c}' missing '::' separator (expected '<feature>::<asset>').")
        f, a = c.split("::", 1)
        feats.append(f)
        assets.add(a)
    features = sorted(set(feats))
    assets = sorted(assets)

    # build dict per feature
    frames = {}
    for f in features:
        # select columns for feature f
        sub = {c.split("::", 1)[1]: df[c] for c in df.columns if c.startswith(f + "::")}
        frames[f] = pd.DataFrame(sub, index=df.index)[assets]

    # stack into panel
    panels = []
    for f in features:
        s = frames[f].stack(dropna=False)  # (Date, Asset)
        panels.append(s.rename(f))
    panel = pd.concat(panels, axis=1).sort_index()
    return panel, features, assets


def apply_xsec_processing(
    panel: pd.DataFrame,
    winsor_q: float,
    mad_cap: float,
    do_zscore: bool,
    orth: Optional[str],
    neutralize: Optional[str],
    groups_panel: Optional[pd.Series],
) -> pd.DataFrame:
    # panel index = (Date, Asset), cols = features
    out = []
    for date, df_cs in panel.groupby(level=0, sort=True):
        x = df_cs.copy()
        # optional neutralization within group
        if neutralize and groups_panel is not None:
            g = groups_panel.loc[date]
            for c in x.columns:
                x[c] = neutralize_group_cs(x[c], g)
        # winsor + MAD cap + zscore
        for c in x.columns:
            s = x[c]
            s = winsorize_cs(s, winsor_q) if winsor_q > 0 else s
            s = mad_cap_cs(s, mad_cap) if mad_cap > 0 else s
            s = zscore_cs(s) if do_zscore else s
            x[c] = s
        # orthogonalize
        if orth in ("gs", "gram-schmidt"):
            x = gram_schmidt(x)
        out.append(x)
    return pd.concat(out, axis=0).sort_index()


def expand_poly_interact(panel: pd.DataFrame, poly_deg: int, interactions: bool) -> pd.DataFrame:
    # Build polynomial terms per date (without bias); assumes columns already processed
    cols = list(panel.columns)
    expanded_cols = cols.copy()
    expanded = panel.copy()

    if poly_deg >= 2:
        for deg in range(2, poly_deg + 1):
            for c in cols:
                cname = f"{c}^{'%d'%deg}"
                expanded[cname] = panel[c] ** deg
                expanded_cols.append(cname)

    if interactions and len(cols) >= 2:
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                cname = f"{cols[i]}*{cols[j]}"
                expanded[cname] = panel[cols[i]] * panel[cols[j]]
                expanded_cols.append(cname)

    # Optional zscore after expansion? Keep preprocessed scale; downstream regression handles intercept.
    return expanded[expanded_cols]


# ----------------------------
# Regressions & IC
# ----------------------------
def cs_regression(
    y_panel: pd.Series,  # (Date, Asset) -> return_{t+1}
    X_panel: pd.DataFrame,  # (Date, Asset) x K
    weights_panel: Optional[pd.Series],
    model: str = "wls",
    add_const: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run cross-sectional regression per date: y = X b (+ c).
    Returns:
      betas: time series of factor returns (index=Date, cols=factors + const?)
      resid: panel residuals aligned to (Date, Asset)
    """
    betas = []
    resid_records = []
    factors = list(X_panel.columns)
    for date, X in X_panel.groupby(level=0, sort=True):
        y = y_panel.loc[date]
        W = None if weights_panel is None else weights_panel.loc[date]
        # align
        idx = X.index.get_level_values(1)
        common = idx.intersection(y.index)
        if W is not None:
            common = common.intersection(W.index)
            w = W.loc[common].astype(float).values
        else:
            w = None
        Xi = X.loc[(date, common)].astype(float).values#type:ignore
        yi = y.loc[common].astype(float).values
        if add_const:
            Xi = np.column_stack([np.ones(len(common)), Xi])
        # solve
        if w is not None and model.lower() == "wls":
            ww = np.sqrt(np.maximum(w, 0.0))
            Xi_w = Xi * ww[:, None]
            yi_w = yi * ww
            coef, *_ = np.linalg.lstsq(Xi_w, yi_w, rcond=None)
        else:
            coef, *_ = np.linalg.lstsq(Xi, yi, rcond=None)

        # residuals
        yhat = Xi @ coef
        eps = yi - yhat
        resid_records.extend([(date, a, float(e)) for a, e in zip(common, eps)])

        # store betas
        row = {"Date": date}
        if add_const:
            row["const"] = float(coef[0])#type:ignore
            for k, c in enumerate(factors, start=1):
                row[c] = float(coef[k])#type:ignore
        else:
            for k, c in enumerate(factors):
                row[c] = float(coef[k])#type:ignore
        betas.append(row)

    betas_df = pd.DataFrame(betas).set_index("Date").sort_index()
    resid_df = pd.DataFrame(resid_records, columns=["Date", "Asset", "resid"]).set_index(["Date", "Asset"]).sort_index()
    return betas_df, resid_df


def information_coefficient(
    X_panel: pd.DataFrame,
    y_panel: pd.Series,
    method: str = "spearman",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (IC_t per factor per date, summary)."""
    from scipy.stats import spearmanr, pearsonr  # lightweight; if unavailable, fallback to numpy corr

    ics = []
    for date, X in X_panel.groupby(level=0, sort=True):
        y = y_panel.loc[date]
        common = X.index.get_level_values(1).intersection(y.index)
        Xi = X.loc[(date, common)]#type:ignore
        yi = y.loc[common]
        row = {"Date": date}
        for c in X.columns:
            xi = Xi[c]
            if method == "spearman":
                ic, _ = spearmanr(xi.values, yi.values, nan_policy="omit")
            else:
                ic, _ = pearsonr(xi.values, yi.values)
            row[c] = float(ic) if np.isfinite(ic) else np.nan#type:ignore
        ics.append(row)
    ic_ts = pd.DataFrame(ics).set_index("Date").sort_index()

    # summary
    summ = []
    for c in ic_ts.columns:
        s = ic_ts[c].dropna()
        if s.empty:
            m = 0.0
            ir = 0.0
            hr = 0.0
        else:
            m = float(s.mean())
            ir = float(s.mean() / (s.std(ddof=0) + 1e-12))
            hr = float((s > 0).mean())
        summ.append({"factor": c, "IC_mean": m, "IC_IR": ir, "IC_hit": hr, "N": int(s.size)})
    ic_summary = pd.DataFrame(summ).set_index("factor").sort_values("IC_mean", ascending=False)
    return ic_ts, ic_summary


# ----------------------------
# Ranking & Portfolios
# ----------------------------
def quantile_portfolios(
    score_panel: pd.Series,  # (Date, Asset)
    fwdret_panel: pd.Series,  # (Date, Asset)
    weights_panel: Optional[pd.Series],
    quantiles: int = 10,
    hold: int = 1,
    ls: bool = True,
) -> pd.DataFrame:
    """
    Build equal-weight (or cap-weight) quantile portfolios and compute next-period returns.
    No overlapping holding aggregation (simple 1-period hold repeated).
    Returns timeseries with columns Q1..Qk and optionally LS (Qk-Q1).
    """
    out = []
    for date, s in score_panel.groupby(level=0, sort=True):
        r = fwdret_panel.loc[date]
        common = s.index.get_level_values(1).intersection(r.index)
        if len(common) < quantiles * 2:
            continue
        si = s.loc[(date, common)]
        ri = r.loc[common]
        # ranks and buckets
        ranks = si.rank(method="first", na_option="keep")
        n = ranks.notna().sum()
        if n < quantiles * 2:
            continue
        q_labels = [f"Q{i+1}" for i in range(quantiles)]
        q = pd.qcut(ranks, q=quantiles, labels=q_labels, duplicates="drop")
        # weights
        if weights_panel is not None:
            w = weights_panel.loc[date].reindex(common).astype(float).clip(lower=0.0)
        else:
            w = pd.Series(1.0, index=common)
        w = w.where(~q.isna(), np.nan)
        # normalize within bucket
        ret_row = {"Date": date}
        for lab in q.unique().dropna().tolist():#type:ignore
            mask = q == lab
            ww = w[mask]
            ww = ww / (ww.sum() if ww.sum() != 0 else 1.0)
            ret_row[lab] = float((ri[mask] * ww).sum(skipna=True))
        # long-short
        if ls and ("Q1" in ret_row and f"Q{quantiles}" in ret_row):
            ret_row["LS"] = ret_row[f"Q{quantiles}"] - ret_row["Q1"]#type:ignore
        out.append(ret_row)
    ts = pd.DataFrame(out).set_index("Date").sort_index()
    ts = ts.fillna(0.0)
    # simple (no overlapping) holding extension
    if hold > 1 and not ts.empty:
        ts = ts.rolling(hold).mean().dropna()
    return ts


# ----------------------------
# CLI and orchestration
# ----------------------------
@dataclass
class Config:
    returns: str
    features: str
    outdir: str
    mcap: Optional[str] = None
    groups: Optional[str] = None
    universe: Optional[str] = None
    poly_deg: int = 2
    interactions: bool = True
    log_cols: str = ""
    winsor: float = 0.01
    mad_cap: float = 5.0
    zscore: bool = True
    orth: str = ""
    neutralize: str = ""
    nan_policy: str = "drop"
    reg: str = "wls"
    target_horizon: int = 1
    weights: str = "mcap"
    add_constant: bool = True
    ic_method: str = "spearman"
    quantiles: int = 10
    score: str = ""
    composite: str = ""
    long_short: bool = True
    hold: int = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Factor expansion, cross-sectional regressions, ICs and portfolios.")
    p.add_argument("--returns", required=True, help="CSV of realized returns (wide: Date x Assets)")
    p.add_argument("--features", required=True, help="CSV of features with '<feature>::<asset>' columns")
    p.add_argument("--mcap", default="", help="CSV of market caps (wide)")
    p.add_argument("--groups", default="", help="CSV of groups (e.g., industry) (wide, strings)")
    p.add_argument("--universe", default="", help="CSV of tradable mask 0/1 (wide)")

    p.add_argument("--poly-deg", type=int, default=2)
    p.add_argument("--interactions", action="store_true")
    p.add_argument("--log-cols", default="", help="Comma list of base columns to log-transform BEFORE expansion")
    p.add_argument("--winsor", type=float, default=0.01)
    p.add_argument("--mad-cap", type=float, default=5.0)
    p.add_argument("--zscore", action="store_true")
    p.add_argument("--orth", default="", help="Set to 'gs' to Gram–Schmidt orthogonalize factors within each date")
    p.add_argument("--neutralize", default="", help="Set to 'group' to de-mean factors within group; requires --groups")
    p.add_argument("--nan-policy", default="drop", choices=["drop", "fill0"])

    p.add_argument("--reg", default="wls", choices=["ols", "wls"])
    p.add_argument("--target-horizon", type=int, default=1, help="Future shift of returns (+h)")
    p.add_argument("--weights", default="mcap", help="Weight source for WLS (default mcap)")
    p.add_argument("--no-constant", action="store_true", help="Do not include intercept in regressions")

    p.add_argument("--ic-method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--quantiles", type=int, default=10)
    p.add_argument("--score", default="", help="Factor name for ranking; or 'composite'")
    p.add_argument("--composite", default="", help="Weights for composite 'f1:1,f2:0.5,...'")
    p.add_argument("--long-short", action="store_true")
    p.add_argument("--hold", type=int, default=1)

    p.add_argument("--outdir", default="fx_out", help="Output directory")
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    cfg = Config(
        returns=args.returns,
        features=args.features,
        outdir=args.outdir,
        mcap=args.mcap or None,
        groups=args.groups or None,
        universe=args.universe or None,
        poly_deg=max(1, args.poly_deg),
        interactions=bool(args.interactions),
        log_cols=args.log_cols,
        winsor=args.winsor,
        mad_cap=args.mad_cap,
        zscore=bool(args.zscore or (not args.interactions and args.poly_deg == 1)),  # default to True if flag given
        orth=args.orth,
        neutralize=args.neutralize,
        nan_policy=args.nan_policy,
        reg=args.reg,
        target_horizon=max(0, args.target_horizon),
        weights=args.weights,
        add_constant=not args.no_constant,
        ic_method=args.ic_method,
        quantiles=max(2, args.quantiles),
        score=args.score,
        composite=args.composite,
        long_short=bool(args.long_short),
        hold=max(1, args.hold),
    )

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load core data
    R = read_wide_csv(cfg.returns)              # realized returns at t (we will shift to future)
    FEAT_WIDE = read_wide_csv(cfg.features)     # columns feature::asset
    MC = read_wide_csv(cfg.mcap) if cfg.mcap else None
    GR = read_wide_csv(cfg.groups, dtype="str") if cfg.groups else None
    UV = read_wide_csv(cfg.universe) if cfg.universe else None

    # Reshape features to panel (Date, Asset) x features
    panel_raw, base_features, assets = build_panel_from_prefixed_columns(FEAT_WIDE)

    # Optional log-transform on base columns BEFORE expansion
    if cfg.log_cols:
        for fc in [s.strip() for s in cfg.log_cols.split(",") if s.strip()]:
            if fc in base_features:
                s = panel_raw[fc]
                s = s.where(s <= 0, np.log(s.where(s > 0)))  # note: negative/zero -> keep as-is
                panel_raw[fc] = s

    # Align with returns/mcap/groups/universe -> panel alignment
    R = R.reindex(columns=assets)
    if MC is not None:
        MC = MC.reindex(columns=assets)
    if GR is not None:
        GR = GR.reindex(columns=assets)
    if UV is not None:
        UV = UV.reindex(columns=assets)

    # Build (Date, Asset) index union and align
    idx = panel_raw.index.get_level_values(0)
    if not R.index.equals(R.index):
        pass
    # Create panel returns (future)
    R_f = R.shift(-cfg.target_horizon)  # future return realized
    # Stack to (Date, Asset)
    y_panel = R_f.stack(dropna=False).rename("ret").sort_index()#type:ignore
    if MC is not None:
        mcap_panel = MC.stack(dropna=False).rename("mcap").sort_index()#type:ignore
    else:
        mcap_panel = None
    if GR is not None:
        groups_panel = GR.stack(dropna=False).rename("group").sort_index()#type:ignore
    else:
        groups_panel = None
    if UV is not None:
        uv_panel = UV.stack(dropna=False).rename("uv").sort_index()#type:ignore
    else:
        uv_panel = None

    # Optional universe mask
    if uv_panel is not None:
        mask = uv_panel.astype(float) > 0.5
        panel_raw = panel_raw[mask.reindex(panel_raw.index).fillna(False).values]
        y_panel = y_panel[mask.reindex(y_panel.index).fillna(False).values]#type:ignore
        if mcap_panel is not None:
            mcap_panel = mcap_panel[mask.reindex(mcap_panel.index).fillna(False).values]#type:ignore
        if groups_panel is not None:
            groups_panel = groups_panel[mask.reindex(groups_panel.index).fillna(False).values]#type:ignore
#type:ignore#type:ignore
    # Cross-sectional processing (winsor/MAD/zscore/neutralize/orth)
    processed = apply_xsec_processing(
        panel_raw,
        winsor_q=cfg.winsor,
        mad_cap=cfg.mad_cap,
        do_zscore=cfg.zscore if cfg.zscore is not None else True,
        orth=cfg.orth,
        neutralize=cfg.neutralize,
        groups_panel=groups_panel,#type:ignore
    )

    # Expand polynomial & interactions
    X_panel = expand_poly_interact(processed, poly_deg=cfg.poly_deg, interactions=cfg.interactions)

    # Post-expansion NaN policy
    if cfg.nan_policy == "drop":
        keep_idx = X_panel.dropna(how="any").index.intersection(y_panel.dropna().index)
        X_panel = X_panel.loc[keep_idx]
        y_panel = y_panel.loc[keep_idx]
        if mcap_panel is not None:
            mcap_panel = mcap_panel.loc[keep_idx]
    else:
        X_panel = X_panel.fillna(0.0)
        y_panel = y_panel.fillna(0.0)
        if mcap_panel is not None:
            mcap_panel = mcap_panel.fillna(0.0)

    # Regressions
    weights_panel = None
    if cfg.reg == "wls" and cfg.weights == "mcap" and mcap_panel is not None:
        weights_panel = mcap_panel.copy()
    betas_df, resid_panel = cs_regression(y_panel, X_panel, weights_panel, model=cfg.reg, add_const=cfg.add_constant)#type:ignore

    # ICs
    ic_ts, ic_summary = information_coefficient(X_panel, y_panel, method=cfg.ic_method)#type:ignore

    # Portfolios (optional)
    ls_ts = pd.DataFrame()
    if cfg.score:
        if cfg.score == "composite":
            # parse weights
            parts = [p.strip() for p in cfg.composite.split(",") if p.strip()]
            comp = pd.Series(0.0, index=X_panel.columns, dtype=float)
            for it in parts:
                k, v = it.split(":")
                comp[k] = float(v)
            comp = comp[comp != 0]
            # reindex; missing factors -> 0 weight
            Xw = X_panel.reindex(columns=comp.index).fillna(0.0)
            score_panel = (Xw * comp.values).sum(axis=1)
        else:
            if cfg.score not in X_panel.columns:
                raise SystemExit(f"--score '{cfg.score}' not found in expanded factors.")
            score_panel = X_panel[cfg.score]
        ls_ts = quantile_portfolios(score_panel, y_panel, weights_panel, quantiles=cfg.quantiles, hold=cfg.hold, ls=cfg.long_short)#type:ignore

    # Outputs
    # Save expanded factors as parquet (compact)
    exp_path = outdir / "expanded_factors.parquet"
    X_panel.to_frame().unstack(level=1).to_parquet(exp_path, index=True)  # panels -> Date x (Factor x Asset)#type:ignore
    betas_df.to_csv(outdir / "factor_returns.csv")
    ic_ts.to_csv(outdir / "ic_timeseries.csv")
    ic_summary.to_csv(outdir / "ic_summary.csv")
    if not ls_ts.empty:
        ls_ts.to_csv(outdir / "ls_timeseries.csv")
    # snapshot last date exposures
    last_date = X_panel.index.get_level_values(0).max()
    snap = X_panel.loc[last_date]
    snap.to_csv(outdir / "exposures_snapshot.csv")
    # config
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Console summary
    print("== Factor Expansion Complete ==")
    print(f"- Expanded factors: {exp_path}")
    print(f"- Factor returns:   {outdir / 'factor_returns.csv'}")
    print(f"- IC summary:       {outdir / 'ic_summary.csv'}")
    if not ls_ts.empty:
        print(f"- Long-Short TS:    {outdir / 'ls_timeseries.csv'}")


if __name__ == "__main__":
    main()
