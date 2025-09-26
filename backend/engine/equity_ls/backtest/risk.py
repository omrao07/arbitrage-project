# engines/equity_ls/backtest/risk.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Tuple

BPS = 1e-4
TRADING_DAYS = 252


# ------------------------- core stats -------------------------

def rolling_vol(
    returns: pd.DataFrame | pd.Series,
    window: int = 63,
    annualize: bool = True,
    ewma_lambda: float | None = None,
) -> pd.Series:
    """
    Rolling volatility. If ewma_lambda is provided, use EWMA; else simple.
    """
    r = returns if isinstance(returns, pd.Series) else returns.sum(axis=1)
    r = r.dropna()
    if ewma_lambda is None:
        vol = r.rolling(window).std()
    else:
        w = (1 - ewma_lambda) * np.power(ewma_lambda, np.arange(len(r))[::-1])
        w = w / w.sum()
        vol = pd.Series(
            np.sqrt(np.convolve((r - r.ewm(alpha=1 - ewma_lambda).mean()).pow(2), w, mode="same")),
            index=r.index,
        )
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def max_drawdown(nav: pd.Series) -> Tuple[float, pd.Series]:
    """
    Max drawdown from an equity curve (NAV).
    Returns (mdd, drawdown_series), where mdd is negative number (e.g., -0.17).
    """
    cummax = nav.cummax()
    dd = nav / cummax - 1.0
    return float(dd.min()), dd


def portfolio_beta(
    weights: pd.Series,            # weights (sum gross<=1), index=tickers
    asset_returns: pd.DataFrame,   # date x ticker
    market_returns: pd.Series,     # date
    lookback: int = 252,
) -> float:
    """
    Regress portfolio returns on market to estimate beta.
    """
    tickers = weights.index.intersection(asset_returns.columns) # type: ignore
    R = asset_returns.loc[:, tickers].tail(lookback).fillna(0.0) # type: ignore
    w = weights.reindex(tickers).fillna(0.0)
    pr = R.dot(w)
    mr = market_returns.reindex(pr.index).fillna(0.0)
    if pr.var() == 0:
        return 0.0
    cov = np.cov(pr, mr)[0, 1]
    var_m = np.var(mr)
    return float(0.0 if var_m == 0 else cov / var_m)


# ------------------------- constraints & transforms -------------------------

def clip_by_cap(weights: pd.Series, name_cap: float) -> pd.Series:
    """Per-name cap (absolute)."""
    return weights.clip(lower=-name_cap, upper=name_cap)


def sector_net_neutral(
    weights: pd.Series,
    sector_map: Dict[str, str],
    method: str = "demean",   # "demean" or "project"
) -> pd.Series:
    """
    Neutralize sector exposures: sum of weights per sector = ~0.
    - demean: subtract mean within each sector
    - project: project out sector dummies
    """
    s = pd.Series({t: sector_map.get(t, "Unknown") for t in weights.index})
    out = weights.copy().astype(float)
    if method == "demean":
        for sec, grp in s.groupby(s):
            idx = grp.index
            out.loc[idx] = out.loc[idx] - out.loc[idx].mean()
    else:
        # project-out sector betas via OLS: w = Zb + e -> use residual e
        secs = pd.get_dummies(s)
        b, _, _, _ = np.linalg.lstsq(secs.values, weights.values, rcond=None) # type: ignore
        resid = weights.values - secs.values @ b
        out = pd.Series(resid, index=weights.index)
    return out


def beta_neutralize(
    weights: pd.Series,
    betas: pd.Series | float,
    target_beta: float = 0.0,
) -> pd.Series:
    """
    Adjust weights to hit target beta. Accepts single beta or per-name betas.
    """
    if np.isscalar(betas):
        port_beta = float(betas) * weights.sum() # type: ignore
        adj = target_beta - port_beta
        # spread adjustment evenly by gross
        gross = weights.abs().sum() + 1e-12
        return weights + (adj / gross) * np.sign(weights)
    b = pd.Series(betas).reindex(weights.index).fillna(0.0)
    port_beta = float((weights * b).sum())
    if b.abs().sum() == 0:
        return weights
    # solve for scalar k such that sum((w + k*sign(w))*b) = target
    k = (target_beta - port_beta) / (np.sign(weights) * b).sum()
    return weights + k * np.sign(weights)


def gross_net_limits(
    weights: pd.Series,
    max_gross: float = 1.0,
    max_net: float = 0.10,
) -> pd.Series:
    """
    Scale weights to obey gross and net caps.
    """
    gross = weights.abs().sum()
    net = weights.sum()
    scale_gross = 1.0 if gross <= max_gross else max_gross / (gross + 1e-12)
    # Net scaling: shift by constant to reduce net exposure while keeping pattern
    shift = 0.0
    if abs(net) > max_net:
        shift = (net - np.sign(net) * max_net) / (len(weights) + 1e-12)
    w = weights * scale_gross - shift
    # Re-check gross after net shift
    g2 = w.abs().sum()
    if g2 > max_gross:
        w = w * (max_gross / (g2 + 1e-12))
    return w


def volatility_target(
    weights: pd.Series,
    returns: pd.DataFrame,      # date x ticker
    target_ann_vol: float = 0.10,
    lookback: int = 63,
    floor: float = 0.25,
    cap: float = 2.0,
) -> Tuple[pd.Series, float]:
    """
    Scale portfolio to a target annualized volatility using sample covariance.
    Returns (scaled_weights, scale_factor).
    """
    tickers = weights.index.intersection(returns.columns) # type: ignore
    R = returns.loc[:, tickers].tail(lookback).dropna(how="all") # type: ignore
    if R.empty or R.shape[0] < 2 or weights.abs().sum() == 0:
        return weights, 1.0
    w = weights.reindex(tickers).fillna(0.0).values.reshape(-1, 1) # type: ignore
    cov = np.cov(R.values, rowvar=False)
    port_vol = float(np.sqrt(max(1e-16, (w.T @ cov @ w)[0, 0])) * np.sqrt(TRADING_DAYS))
    if port_vol == 0:
        return weights, 1.0
    k = np.clip(target_ann_vol / port_vol, floor, cap)
    scaled = weights * k
    return scaled, float(k)


def enforce_caps_by_group(
    weights: pd.Series,
    group_map: Dict[str, str],
    cap_per_group: float = 0.25,
) -> pd.Series:
    """
    Limit absolute sum of weights within each group (e.g., sector <= 25% gross).
    """
    out = weights.copy().astype(float)
    groups = pd.Series({t: group_map.get(t, "Unknown") for t in weights.index})
    for g, idx in groups.groupby(groups):
        sel = out.loc[idx.index]
        gross = sel.abs().sum()
        if gross > cap_per_group and gross > 0:
            out.loc[idx.index] = sel * (cap_per_group / gross)
    return out


# ------------------------- high-level sizing pipeline -------------------------

def size_equity_ls_weights(
    raw_scores: pd.Series,                 # cross-sectional scores (+ long, - short)
    prices: pd.Series | pd.DataFrame,      # last price(s) for universe
    sector_map: Optional[Dict[str, str]] = None,
    market_betas: Optional[pd.Series | float] = None,
    asset_returns: Optional[pd.DataFrame] = None,   # for vol targeting
    market_returns: Optional[pd.Series] = None,     # for beta calc (report)
    *,
    max_name: float = 0.03,
    max_gross: float = 1.0,
    max_net: float = 0.05,
    cap_per_sector: float = 0.25,
    target_vol: float = 0.10,
    vol_lookback: int = 63,
    neutralize_sectors: bool = True,
) -> Dict[str, object]:
    """
    End-to-end conversion from scores -> portfolio weights with common constraints.
    Returns dict: {weights, scale_vol, beta, diagnostics}
    """
    # 1) Normalize raw scores -> base weights summing to zero, unit gross
    s = raw_scores.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"weights": s, "scale_vol": 1.0, "beta": 0.0, "diagnostics": {}}

    z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    w = z / z.abs().sum()  # unit gross

    # 2) Per-name cap
    w = clip_by_cap(w, max_name)

    # 3) Sector neutrality & caps
    if sector_map and neutralize_sectors:
        w = sector_net_neutral(w, sector_map, method="demean")
        w = clip_by_cap(w, max_name)  # re-clip after neutralization
        w = enforce_caps_by_group(w, sector_map, cap_per_group=cap_per_sector)

    # 4) Beta neutrality (optional)
    if market_betas is not None:
        w = beta_neutralize(w, market_betas, target_beta=0.0)

    # 5) Gross/net limits
    w = gross_net_limits(w, max_gross=max_gross, max_net=max_net)

    # 6) Vol targeting
    scale = 1.0
    if asset_returns is not None:
        w, scale = volatility_target(
            weights=w, returns=asset_returns, target_ann_vol=target_vol, lookback=vol_lookback
        )

    # Diagnostics
    beta_val = None
    if (asset_returns is not None) and (market_returns is not None):
        try:
            beta_val = portfolio_beta(w, asset_returns, market_returns, lookback=min(vol_lookback, 252))
        except Exception:
            beta_val = None

    diag = {
        "gross": float(w.abs().sum()),
        "net": float(w.sum()),
        "max_name": float(w.abs().max() if not w.empty else 0.0),
        "scale_vol": float(scale),
        "beta_est": None if beta_val is None else float(beta_val),
    }

    return {"weights": w.sort_values(ascending=False), "scale_vol": scale, "beta": beta_val, "diagnostics": diag}


# ------------------------- risk report -------------------------

def risk_report_from_positions(
    positions: pd.DataFrame,      # date x ticker in SHARES
    prices: pd.DataFrame,         # date x ticker
    factor_exposures: Optional[pd.DataFrame] = None,    # ticker x factor betas
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Build a concise daily risk report with gross/net, concentration, sector tilt,
    and optional factor exposures (weighted by notional).
    """
    # end-of-day notional
    notion = positions * prices
    gross = notion.abs().sum(axis=1)
    net = notion.sum(axis=1)
    conc = (notion.abs().div(gross.replace(0, np.nan), axis=0)).pow(2).sum(axis=1)  # Herfindahl

    df = pd.DataFrame({
        "gross$": gross,
        "net$": net,
        "concentration_HHI": conc,
    })

    # sector tilt
    if sector_map:
        s = pd.Series(sector_map)
        common = positions.columns.intersection(s.index)
        s = s.reindex(common).fillna("Unknown")
        for sec, idx in s.groupby(s):
            w_sec = notion.loc[:, idx.index].sum(axis=1) # type: ignore
            df[f"sector_{sec}_$"] = w_sec

    # factors
    if factor_exposures is not None:
        common = positions.columns.intersection(factor_exposures.index)
        W = notion.loc[:, common] # type: ignore
        B = factor_exposures.loc[common]
        # exposure_t = sum_i ( notional_{i,t} * beta_{i,f} ) / gross_t
        gross_safe = gross.replace(0, np.nan)
        expo = {}
        for f in B.columns:
            expo[f"factor_{f}"] = (W.mul(B[f], axis=1).sum(axis=1) / gross_safe)
        df = pd.concat([df, pd.DataFrame(expo)], axis=1)

    return df.fillna(0.0)