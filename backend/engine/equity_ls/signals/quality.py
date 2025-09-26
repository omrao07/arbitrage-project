# engines/equity_ls/signals/quality.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Iterable

# ----------------------------- utils -----------------------------

def _latest_nonnull(df: pd.DataFrame, delay: int = 1) -> pd.Series:
    """
    Pick the latest row (t - delay) for cross-section.
    Assumes index is chronological; delay=1 prevents lookahead.
    """
    if df.empty:
        return pd.Series(dtype=float)
    idx = max(0, len(df) - 1 - delay)
    return df.iloc[idx].astype(float)

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)

def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / (sd + 1e-12) if sd > 0 else s * 0.0

def _sector_demean(weights: pd.Series, sector_map: Optional[Dict[str, str]]) -> pd.Series:
    if not sector_map:
        return weights
    sec = pd.Series({t: sector_map.get(t, "Unknown") for t in weights.index})
    out = weights.copy().astype(float)
    for g, idx in sec.groupby(sec):
        tickers = idx.index
        out.loc[tickers] = out.loc[tickers] - out.loc[tickers].mean()
    return out

def _normalize_unit_gross(weights: pd.Series) -> pd.Series:
    gross = weights.abs().sum()
    return weights / (gross + 1e-12)

def _standardize(s: pd.Series, inverse: bool = False, winsor_p: float = 0.01) -> pd.Series:
    """Winsorize + z-score; invert if lower is better (e.g., accruals, leverage)."""
    s = _winsorize(s.replace([np.inf, -np.inf], np.nan).dropna(), p=winsor_p)
    z = _zscore(s)
    return -z if inverse else z

# ----------------------------- factor builders -----------------------------

def compute_quality_components(
    *,
    # Either pass a dict of DataFrames (preferred)...
    fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
    # ...or a single wide DataFrame with columns named below.
    wide: Optional[pd.DataFrame] = None,
    delay: int = 1,
    winsor_p: float = 0.01,
) -> Dict[str, pd.Series]:
    """
    Returns standardized (z-scored) components aligned by tickers:
      + ROE, GrossMargin, Stability    (higher is better)
      - Accruals, LeverageChange, NetIssuance (lower is better)
      + BuybackYield (higher is better)
    Accepted keys/columns (case-insensitive):
      ROE, GrossMargin, Stability, Accruals, LeverageChange, NetIssuance, BuybackYield
    """
    def pull(name: str) -> Optional[pd.Series]:
        key = name.lower()
        if fundamentals is not None:
            for k, df in fundamentals.items():
                if k.lower() == key and isinstance(df, pd.DataFrame):
                    return _latest_nonnull(df, delay)
        if wide is not None and name in wide.columns:
            return _latest_nonnull(wide[[name]], delay)
        # try case-insensitive for wide
        if wide is not None:
            for c in wide.columns:
                if c.lower() == key:
                    return _latest_nonnull(wide[[c]].rename(columns={c: name}), delay)
        return None

    comps: Dict[str, pd.Series] = {}

    # Profitability
    roe = pull("ROE")
    if roe is not None:
        comps["ROE"] = _standardize(roe, inverse=False, winsor_p=winsor_p)

    gm = pull("GrossMargin")
    if gm is not None:
        comps["GrossMargin"] = _standardize(gm, inverse=False, winsor_p=winsor_p)

    stab = pull("Stability")  # e.g., negative earnings volatility, cashflow/asset stability
    if stab is not None:
        comps["Stability"] = _standardize(stab, inverse=False, winsor_p=winsor_p)

    # Quality “penalties”
    accr = pull("Accruals")  # higher accruals = lower quality
    if accr is not None:
        comps["Accruals"] = _standardize(accr, inverse=True, winsor_p=winsor_p)

    lev = pull("LeverageChange")  # increase in leverage -> worse
    if lev is not None:
        comps["LeverageChange"] = _standardize(lev, inverse=True, winsor_p=winsor_p)

    iss = pull("NetIssuance")  # equity issuance dilutes; negative (buyback) is good
    if iss is not None:
        comps["NetIssuance"] = _standardize(iss, inverse=True, winsor_p=winsor_p)

    bb = pull("BuybackYield")  # positive buyback yield is good
    if bb is not None:
        comps["BuybackYield"] = _standardize(bb, inverse=False, winsor_p=winsor_p)

    # Align tickers (intersection) across available components
    if not comps:
        return {}
    tickers = set.intersection(*(set(s.index) for s in comps.values()))
    comps = {k: v.reindex(sorted(tickers)).dropna() for k, v in comps.items()}
    return comps

# ----------------------------- composite & weights -----------------------------

def build_quality_score(
    components: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Combine standardized components into a single cross-sectional Quality score.
    Default equal-weight; you can override per component.
    """
    if not components:
        return pd.Series(dtype=float)
    keys = list(components.keys())
    if not weights:
        weights = {k: 1.0 for k in keys}
    # align and combine
    common = set.intersection(*(set(components[k].index) for k in keys))
    if not common:
        return pd.Series(dtype=float)
    common = sorted(common)
    X = pd.DataFrame({k: components[k].reindex(common) * float(weights.get(k, 0.0)) for k in keys})
    return X.sum(axis=1)

def build_signal(
    *,
    fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
    wide: Optional[pd.DataFrame] = None,
    delay: int = 1,
    winsor_p: float = 0.01,
    component_weights: Optional[Dict[str, float]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    long_cap: float = 0.05,
    short_cap: float = -0.05,
    unit_gross: float = 1.0,
) -> pd.Series:
    """
    Main entrypoint:
      - Extract & standardize components (profitability, stability, accruals, leverage, issuance/buybacks)
      - Build composite Quality score (weighted)
      - Sector-neutralize (optional)
      - Clip per-name and normalize to unit gross
    Returns: cross-sectional weights (positive=long quality, negative=short junk)
    """
    comps = compute_quality_components(fundamentals=fundamentals, wide=wide, delay=delay, winsor_p=winsor_p)
    score = build_quality_score(comps, weights=component_weights)
    if score.empty:
        return score

    # z-score composite for robustness and convert to weights
    z = _zscore(score)

    # Optional sector neutrality
    z = _sector_demean(z, sector_map)

    w = z.clip(lower=short_cap, upper=long_cap)
    w = _normalize_unit_gross(w) * unit_gross
    return w.sort_values(ascending=False)

# ----------------------------- convenience: build from prices if no fundamentals -----------------------------

def proxy_quality_from_prices(
    prices: pd.DataFrame,           # date x ticker close
    shares_out: Optional[pd.DataFrame] = None,  # for NetIssuance proxy
    delay: int = 1,
    winsor_p: float = 0.01,
    sector_map: Optional[Dict[str, str]] = None,
    long_cap: float = 0.05,
    short_cap: float = -0.05,
    unit_gross: float = 1.0,
) -> pd.Series:
    """
    Fallback proxy using price-based estimates when fundamentals are unavailable:
      - Stability ~ negative rolling volatility (low vol = higher quality)
      - NetIssuance ~ growth in shares outstanding (if provided)
      - BuybackYield ~ -NetIssuance (if shares_out absent)
    """
    px = prices.copy().sort_index()
    if px.shape[0] < 60 + delay:
        return pd.Series(dtype=float)

    ret = px.pct_change()
    vol = ret.rolling(60).std().shift(delay).iloc[-1]
    stability = -vol  # lower vol -> higher quality
    stability_z = _standardize(stability, inverse=False, winsor_p=winsor_p)

    if shares_out is not None:
        soi = shares_out.pct_change(252).shift(delay).iloc[-1]  # YoY share growth
        net_issuance_z = _standardize(soi, inverse=True, winsor_p=winsor_p)  # more issuance worse
        buyback_z = -net_issuance_z
    else:
        net_issuance_z = pd.Series(0.0, index=stability_z.index)
        buyback_z = pd.Series(0.0, index=stability_z.index)

    comps = {
        "Stability": stability_z,
        "NetIssuance": net_issuance_z,
        "BuybackYield": buyback_z,
    }
    score = build_quality_score(comps)
    z = _zscore(score)
    z = _sector_demean(z, sector_map)
    w = z.clip(lower=short_cap, upper=long_cap)
    return _normalize_unit_gross(w) * unit_gross