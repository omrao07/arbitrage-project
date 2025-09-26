# engines/equity_ls/signals/value.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

# ----------------------------- utils -----------------------------

def _latest(df: pd.DataFrame, delay: int = 1) -> pd.Series:
    """Latest row at t-delay to avoid lookahead."""
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

def _unit_gross(w: pd.Series, gross: float = 1.0) -> pd.Series:
    g = w.abs().sum()
    return w * (gross / (g + 1e-12))

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe ratio with sign handling; avoids explode on ~0 denominator."""
    out = num.astype(float) / den.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)

# ----------------------------- component builder -----------------------------

def compute_value_components(
    *,
    fundamentals: Optional[Dict[str, pd.DataFrame]] = None,
    wide: Optional[pd.DataFrame] = None,
    prices: Optional[pd.DataFrame] = None,     # optional for P/E using market cap
    shares_out: Optional[pd.DataFrame] = None, # optional for market cap
    net_debt: Optional[pd.DataFrame] = None,   # optional (Debt - Cash)
    delay: int = 1,
    winsor_p: float = 0.01,
    prefer_ev: bool = True,
) -> Dict[str, pd.Series]:
    """
    Build standardized 'cheapness' components (higher = cheaper):
      B_P, E_P, FCF_EV, EBITDA_EV, SALES_EV
    Required inputs (by key/column; case-insensitive):
      BookEquity, Earnings, FreeCashFlow, EBITDA, Sales, Price
    Optional: SharesOut, NetDebt (for EV); or pass pre-computed EV in 'EnterpriseValue'.
    """
    def pull(name: str) -> Optional[pd.Series]:
        key = name.lower()
        if fundamentals is not None:
            for k, df in fundamentals.items():
                if k.lower() == key and isinstance(df, pd.DataFrame):
                    return _latest(df, delay)
        if wide is not None:
            # exact first
            if name in wide.columns:
                return _latest(wide[[name]], delay)
            # fallback case-insensitive
            for c in wide.columns:
                if c.lower() == key:
                    return _latest(wide[[c]].rename(columns={c: name}), delay)
        return None

    comps: Dict[str, pd.Series] = {}

    # --- Core building blocks ---
    # Price vector (last known)
    if prices is not None and not prices.empty:
        price = _latest(prices, delay)
    else:
        price = pull("Price")

    book = pull("BookEquity")
    earnings = pull("Earnings") or pull("NetIncome")
    fcf = pull("FreeCashFlow")
    ebitda = pull("EBITDA")
    sales = pull("Sales") or pull("Revenue")
    ev = pull("EnterpriseValue")

    # If EV is not provided, synthesize from MarketCap + NetDebt (if available)
    if ev is None and prefer_ev and (price is not None):
        so = _latest(shares_out, delay) if shares_out is not None else None
        nd = _latest(net_debt, delay) if net_debt is not None else None
        if so is not None:
            mcap = price * so
            ev = mcap + (nd if nd is not None else 0.0)

    # --- Ratios turned into "value yields" (higher = cheaper) ---
    if (book is not None) and (price is not None):
        b_p = _safe_div(book, price)
        comps["B_P"] = _zscore(_winsorize(b_p, winsor_p))

    if (earnings is not None) and (price is not None):
        e_p = _safe_div(earnings, price).replace(0, np.nan)
        comps["E_P"] = _zscore(_winsorize(e_p, winsor_p))

    if (fcf is not None):
        if ev is not None:
            fcf_ev = _safe_div(fcf, ev)
            comps["FCF_EV"] = _zscore(_winsorize(fcf_ev, winsor_p))
        elif price is not None:
            fcf_p = _safe_div(fcf, price)
            comps["FCF_P"] = _zscore(_winsorize(fcf_p, winsor_p))

    if (ebitda is not None):
        if ev is not None:
            ebitda_ev = _safe_div(ebitda, ev)
            comps["EBITDA_EV"] = _zscore(_winsorize(ebitda_ev, winsor_p))
        elif price is not None:
            ebitda_p = _safe_div(ebitda, price)
            comps["EBITDA_P"] = _zscore(_winsorize(ebitda_p, winsor_p))

    if (sales is not None):
        if ev is not None:
            s_ev = _safe_div(sales, ev)
            comps["SALES_EV"] = _zscore(_winsorize(s_ev, winsor_p))
        elif price is not None:
            s_p = _safe_div(sales, price)
            comps["SALES_P"] = _zscore(_winsorize(s_p, winsor_p))

    # Align on the intersection of tickers across computed components
    if not comps:
        return {}
    common = set.intersection(*(set(v.index) for v in comps.values()))
    comps = {k: v.reindex(sorted(common)).dropna() for k, v in comps.items()}
    return comps

# ----------------------------- composite & weights -----------------------------

def build_value_score(
    components: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Combine standardized value components (higher = cheaper).
    Default equal-weight across available components.
    """
    if not components:
        return pd.Series(dtype=float)
    keys = list(components.keys())
    if not weights:
        weights = {k: 1.0 for k in keys}
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
    prices: Optional[pd.DataFrame] = None,
    shares_out: Optional[pd.DataFrame] = None,
    net_debt: Optional[pd.DataFrame] = None,
    component_weights: Optional[Dict[str, float]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    delay: int = 1,
    winsor_p: float = 0.01,
    long_cap: float = 0.05,
    short_cap: float = -0.05,
    unit_gross: float = 1.0,
) -> pd.Series:
    """
    Main entrypoint:
      - Build multiple 'value yield' components (B/P, E/P, FCF/EV, EBITDA/EV, Sales/EV)
      - Winsorize + z-score each component
      - Combine into a composite value score
      - Optional sector-neutralization
      - Clip per-name and normalize to unit gross
    Returns: cross-sectional weights (positive=cheap, negative=expensive).
    """
    comps = compute_value_components(
        fundamentals=fundamentals, wide=wide, prices=prices, shares_out=shares_out,
        net_debt=net_debt, delay=delay, winsor_p=winsor_p
    )
    score = build_value_score(comps, weights=component_weights)
    if score.empty:
        return score

    z = _zscore(score)
    z = _sector_demean(z, sector_map)

    w = z.clip(lower=short_cap, upper=long_cap)
    w = _unit_gross(w, gross=unit_gross)
    return w.sort_values(ascending=False)

# ----------------------------- proxy (if fundamentals missing) -----------------------------

def proxy_book_to_price(
    prices: pd.DataFrame,
    book_value_ps: Optional[pd.DataFrame] = None,  # Book value per share
    delay: int = 1,
    winsor_p: float = 0.01,
    sector_map: Optional[Dict[str, str]] = None,
    long_cap: float = 0.05,
    short_cap: float = -0.05,
    unit_gross: float = 1.0,
) -> pd.Series:
    """
    Proxy B/P using book value per share if full fundamentals not available.
    """
    px = prices.copy().sort_index()
    if px.shape[0] < delay + 2 or book_value_ps is None or book_value_ps.empty:
        return pd.Series(dtype=float)

    bps = _latest(book_value_ps, delay)
    p = _latest(px, delay)
    b_p = _safe_div(bps, p).dropna()
    z = _zscore(_winsorize(b_p, winsor_p))
    z = _sector_demean(z, sector_map)
    w = z.clip(lower=short_cap, upper=long_cap)
    return _unit_gross(w, gross=unit_gross).sort_values(ascending=False)