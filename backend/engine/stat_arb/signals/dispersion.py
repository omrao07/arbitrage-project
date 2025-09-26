# engines/stat_arb/signals/dispersion.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Literal

BetaMethod = Literal["rolling_ols", "expanding_ols", "static_ols"]

# -------------------------------------------------------------------
# Cross-sectional dispersion metrics
# -------------------------------------------------------------------

def dispersion_metrics(
    prices: pd.DataFrame,
    *,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling cross-sectional dispersion and average-correlation proxy.

    Returns DataFrame with columns:
      - dispersion: cross-sectional std of demeaned daily returns (over lookback)
      - avg_corr_proxy: 1 - (cs_var / var_of_mean) decomposition proxy (bounded [0,1])
    """
    px = prices.sort_index()
    rets = px.pct_change().fillna(0.0)

    # cross-sectional dispersion per day (demeaned across names each day)
    cs = (rets.sub(rets.mean(axis=1), axis=0)).pow(2).mean(axis=1).pow(0.5)

    # rolling stats
    disp = cs.rolling(lookback).mean()

    # Average-correlation proxy (Lopez de Prado style approximation)
    # var(mean) ≈ avg_corr * avg_var / N + (1-avg_corr) * avg_var / N ?  Use simple proxy:
    # avg pairwise corr ≈ var(mean) / (avg_var / N)
    avg_var = rets.var(axis=1)
    var_mean = rets.mean(axis=1).var()
    # More stable rolling proxy:
    avg_var_rb = rets.rolling(lookback).var().mean(axis=1)
    var_mean_rb = rets.mean(axis=1).rolling(lookback).var()
    avg_corr_proxy = (var_mean_rb / (avg_var_rb / rets.shape[1])).clip(lower=0.0, upper=1.0)

    out = pd.DataFrame({
        "dispersion": disp,
        "avg_corr_proxy": avg_corr_proxy,
    }).dropna()
    return out


# -------------------------------------------------------------------
# Group factor & residuals
# -------------------------------------------------------------------

def _group_map_series(tickers: Iterable[str], group_map: Dict[str, str]) -> pd.Series:
    return pd.Series({t: group_map.get(t, "Unknown") for t in tickers})

def group_factor_returns(returns: pd.DataFrame, group_map: Dict[str, str]) -> pd.DataFrame:
    """
    Build equal-weight group factor returns (e.g., sector averages).
    Returns DataFrame date x group_name.
    """
    gser = _group_map_series(returns.columns, group_map)
    df = {}
    for g, cols in gser.groupby(gser):
        df[g] = returns.loc[:, list(cols.index)].mean(axis=1)
    return pd.DataFrame(df, index=returns.index)

def group_residual_zscores(
    prices: pd.DataFrame,
    group_map: Dict[str, str],
    *,
    lookback: int = 60,
    delay: int = 1,
    winsor_p: float = 0.01,
) -> pd.Series:
    """
    Cross-sectional residual z-scores within each group:
      1) compute daily returns
      2) regress each stock's last-LB returns on its group's factor (EW mean)
      3) take the LAST residual (t-delay) and z-score within group
    Returns Series indexed by tickers; positive = outperformed group, negative = underperformed.
    """
    px = prices.sort_index()
    if px.shape[0] < lookback + delay + 2:
        return pd.Series(dtype=float)

    r = px.pct_change().fillna(0.0)
    group_ret = group_factor_returns(r, group_map)

    tickers = [t for t in px.columns if t in group_map]
    gser = _group_map_series(tickers, group_map)

    resids = {}
    end = -delay if delay > 0 else None
    rb = slice(-(lookback + delay), end)

    for g, cols in gser.groupby(gser):
        names = list(cols.index)
        if g not in group_ret.columns:
            continue
        gvec = group_ret[g].iloc[rb].values.reshape(-1, 1)
        gx = np.hstack([gvec, np.ones_like(gvec)])  # add intercept
        denom = float((gx.T @ gx)[0, 0])
        for t in names:
            y = r[t].iloc[rb].values.reshape(-1, 1)
            # OLS beta on group + intercept
            try:
                b = np.linalg.lstsq(gx, y, rcond=None)[0]
                y_hat = gx @ b
                resid = (y - y_hat).squeeze()[-1]  # last residual at t-delay
                resids[t] = float(resid)
            except Exception:
                resids[t] = 0.0

    s = pd.Series(resids).dropna()
    # Winsorize within each group then z-score within group
    out = []
    for g, cols in gser.groupby(gser):
        idx = list(cols.index)
        sg = s.reindex(idx)
        if sg.dropna().empty:
            continue
        lo, hi = sg.quantile([winsor_p, 1 - winsor_p])
        sg = sg.clip(lo, hi) # type: ignore
        z = (sg - sg.mean()) / (sg.std(ddof=0) + 1e-12)
        out.append(z)
    if not out:
        return pd.Series(dtype=float)
    return pd.concat(out).dropna().sort_values(ascending=False) # type: ignore


# -------------------------------------------------------------------
# Pair selection from residual extremes
# -------------------------------------------------------------------

@dataclass(frozen=True)
class PairCandidate:
    y: str          # dependent leg (underperformer if we expect mean reversion)
    x: str          # overperformer
    group: str
    score: float    # magnitude of divergence (|z_y| + |z_x|)

def select_pairs_from_residuals(
    prices: pd.DataFrame,
    group_map: Dict[str, str],
    *,
    lookback: int = 60,
    delay: int = 1,
    per_group: int = 3,
    min_abs_z: float = 0.5,
) -> List[PairCandidate]:
    """
    For each group:
      - compute residual z-scores vs. group factor
      - pick the most negative (underperformer) and most positive (overperformer)
      - form pairs (y=underperformer, x=overperformer), up to 'per_group'
    """
    z = group_residual_zscores(prices, group_map, lookback=lookback, delay=delay)
    if z.empty:
        return []

    # group tickers
    gser = _group_map_series(z.index, group_map)
    pairs: List[PairCandidate] = []

    for g, idx in gser.groupby(gser):
        names = list(idx.index)
        z_g = z.reindex(names).dropna()
        if len(z_g) < 2:
            continue
        longs = z_g.nsmallest(per_group)  # most negative
        shorts = z_g.nlargest(per_group)  # most positive
        k = min(len(longs), len(shorts))
        for i in range(k):
            y = longs.index[i]
            x = shorts.index[i]
            if abs(longs.iloc[i]) < min_abs_z or abs(shorts.iloc[i]) < min_abs_z:
                continue
            score = float(abs(longs.iloc[i]) + abs(shorts.iloc[i]))
            pairs.append(PairCandidate(y=y, x=x, group=g, score=score)) # type: ignore

    # sort by strength
    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs


# -------------------------------------------------------------------
# Units from spread z (one-shot signal snapshot)
# -------------------------------------------------------------------

def _rolling_beta(y: pd.Series, x: pd.Series, method: BetaMethod, lookback: int) -> pd.Series:
    y = y.astype(float); x = x.astype(float)
    idx = y.index.intersection(x.index) # type: ignore
    y = y.reindex(idx); x = x.reindex(idx)

    if method == "static_ols":
        X = np.vstack([x.values, np.ones(len(x))]).T # type: ignore
        beta = np.linalg.lstsq(X, y.values, rcond=None)[0][0] # type: ignore
        return pd.Series(beta, index=idx)
    betas = pd.Series(index=idx, dtype=float)
    if method == "expanding_ols":
        for i in range(2, len(x)+1):
            yw = y.iloc[:i]; xw = x.iloc[:i]
            X = np.vstack([xw.values, np.ones(len(xw))]).T # type: ignore
            b = np.linalg.lstsq(X, yw.values, rcond=None)[0][0] # type: ignore
            betas.iloc[i-1] = b
        betas.iloc[:1] = betas.iloc[1]
        return betas.ffill()
    # rolling
    lookback = max(10, int(lookback))
    for i in range(lookback, len(x)+1):
        yw = y.iloc[i-lookback:i]; xw = x.iloc[i-lookback:i]
        X = np.vstack([xw.values, np.ones(len(xw))]).T # type: ignore
        b = np.linalg.lstsq(X, yw.values, rcond=None)[0][0] # type: ignore
        betas.iloc[i-1] = b
    betas.iloc[:lookback] = betas.iloc[lookback]
    return betas.ffill()

def _zscore(s: pd.Series, lb: int) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    return (s - mu) / (sd + 1e-12)

def units_from_pairs(
    prices: pd.DataFrame,
    pairs: List[PairCandidate],
    *,
    beta_method: BetaMethod = "rolling_ols",
    beta_lookback: int = 120,
    z_lookback: int = 60,
    entry_z: float = 1.0,
    exit_z: float = 0.25,
    max_units: float = 1.0,
    risk_parity: bool = True,
    unit_risk_target: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[str, str]], pd.DataFrame]:
    """
    Produce single-timestamp units/betas for selected pairs based on spread z.
    Returns:
      - units_by_pair: {"AAA/BBB": +0.6, ...}  (sign follows mean-reversion)
      - betas_by_pair: {"AAA/BBB": 1.15, ...}
      - legs_by_pair:  {"AAA/BBB": ("AAA","BBB"), ...}
      - diag DataFrame: per pair diagnostics (z, spread, beta, risk_scale)
    """
    px = prices.sort_index()
    rets = px.pct_change()

    units_by_pair: Dict[str, float] = {}
    betas_by_pair: Dict[str, float] = {}
    legs_by_pair: Dict[str, Tuple[str, str]] = {}
    rows = []

    if not pairs:
        return units_by_pair, betas_by_pair, legs_by_pair, pd.DataFrame()

    t = px.index[-1]  # snapshot at the last date

    for p in pairs:
        if p.y not in px.columns or p.x not in px.columns:
            continue
        y = px[p.y]; x = px[p.x]
        beta = _rolling_beta(y, x, beta_method, beta_lookback)
        b = float(beta.iloc[-1])

        spread = y - b * x
        z = _zscore(spread, z_lookback).iloc[-1]
        z_val = float(z if z == z else 0.0)

        # Hysteresis entry at snapshot: if |z| >= entry_z → position; if |z| < exit_z → flat; else keep 0
        if abs(z_val) < exit_z:
            u_nom = 0.0
        else:
            u_nom = +1.0 if z_val <= -entry_z else (-1.0 if z_val >= +entry_z else 0.0)

        # Risk parity scaling using spread return vol
        if risk_parity:
            spr_ret_vol = spread.pct_change().rolling(z_lookback).std().iloc[-1]
            k = (unit_risk_target / float(spr_ret_vol if spr_ret_vol == spr_ret_vol and spr_ret_vol > 0 else 1.0))
            u = float(np.clip(u_nom * k, -max_units, +max_units))
        else:
            u = float(np.clip(u_nom, -max_units, +max_units))

        name = f"{p.y}/{p.x}"
        units_by_pair[name] = u
        betas_by_pair[name] = b
        legs_by_pair[name] = (p.y, p.x)
        rows.append({"pair": name, "z": z_val, "beta": b, "spread": float(spread.iloc[-1]), "risk_scale_units": u})

    diag = pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()
    return units_by_pair, betas_by_pair, legs_by_pair, diag


# -------------------------------------------------------------------
# Convenience: end-to-end snapshot
# -------------------------------------------------------------------

def build_dispersion_snapshot(
    prices: pd.DataFrame,
    group_map: Dict[str, str],
    *,
    lookback_disp: int = 60,
    lookback_resid: int = 60,
    delay_resid: int = 1,
    per_group: int = 3,
    min_abs_z: float = 0.5,
    beta_method: BetaMethod = "rolling_ols",
    beta_lookback: int = 120,
    z_lookback: int = 60,
    entry_z: float = 1.0,
    exit_z: float = 0.25,
    max_units: float = 1.0,
    risk_parity: bool = True,
    unit_risk_target: float = 1.0,
) -> Dict[str, object]:
    """
    One call that returns:
      - 'metrics': dispersion & avg_corr_proxy time series
      - 'pairs':  List[PairCandidate]
      - 'units_by_pair', 'betas_by_pair', 'legs_by_pair'
      - 'diag': per-pair diagnostics DataFrame
    """
    metrics = dispersion_metrics(prices, lookback=lookback_disp)
    pairs = select_pairs_from_residuals(
        prices,
        group_map,
        lookback=lookback_resid,
        delay=delay_resid,
        per_group=per_group,
        min_abs_z=min_abs_z,
    )
    u, b, legs, diag = units_from_pairs(
        prices,
        pairs,
        beta_method=beta_method,
        beta_lookback=beta_lookback,
        z_lookback=z_lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        max_units=max_units,
        risk_parity=risk_parity,
        unit_risk_target=unit_risk_target,
    )
    return {"metrics": metrics, "pairs": pairs, "units_by_pair": u, "betas_by_pair": b, "legs_by_pair": legs, "diag": diag}