# engines/equity_ls/signals/sector_rotation.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Mapping, Optional

# --- Defaults: SPDR sector ETFs ---
DEFAULT_SECTORS: Mapping[str, str] = {
    "XLK": "Tech", "XLV": "Health", "XLF": "Financials", "XLY": "ConsDisc",
    "XLP": "ConsStap", "XLI": "Industrials", "XLE": "Energy", "XLU": "Utilities",
    "XLRE": "RealEstate", "XLB": "Materials", "XLC": "CommSvcs",
}

DEFENSIVES = {"Health", "ConsStap", "Utilities", "RealEstate"}
CYCLICALS  = {"Tech", "Financials", "ConsDisc", "Industrials", "Energy", "Materials", "CommSvcs"}

# ----------------------------- utils -----------------------------

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.empty: return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)

def _z(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    return (s - mu) / (sd + 1e-12) if sd > 0 else s * 0.0

def _unit_gross(w: pd.Series, gross: float = 1.0) -> pd.Series:
    g = w.abs().sum()
    return w * (gross / (g + 1e-12))

# ----------------------------- core -----------------------------

def _multi_horizon_momentum(prices: pd.DataFrame, lookbacks: Iterable[int], delay: int) -> pd.Series:
    """
    Average of delayed horizon returns (information up to t-delay).
    """
    scores = []
    for lb in lookbacks:
        r = prices.pct_change(lb).shift(delay)
        scores.append(r.iloc[-1])
    return pd.concat(scores, axis=1).mean(axis=1)

def _vol_scale(prices: pd.DataFrame, vol_lb: int, delay: int) -> pd.Series:
    vol = prices.pct_change().rolling(vol_lb).std().shift(delay).iloc[-1]
    return vol.replace(0, np.nan)

def _apply_macro_tilts(raw: pd.Series, sectors_map: Mapping[str, str], tilts: Dict[str, float]) -> pd.Series:
    """
    tilts keys are sector family names (e.g., "Energy", "Tech", "Defensive", "Cyclical").
    """
    if not tilts:
        return raw
    out = raw.copy()
    # family tilts
    fam_bias = {t: float(tilts.get(sectors_map.get(t, ""), 0.0)) for t in raw.index}
    out = out.add(pd.Series(fam_bias), fill_value=0.0)
    # meta buckets (Defensive/Cyclical)
    if "Defensive" in tilts or "Cyclical" in tilts:
        for t in out.index:
            fam = sectors_map.get(t, "")
            if fam in DEFENSIVES:
                out.loc[t] += float(tilts.get("Defensive", 0.0))
            if fam in CYCLICALS:
                out.loc[t] += float(tilts.get("Cyclical", 0.0))
    return out

def build_signal(
    prices: pd.DataFrame,                         # date x sector ETF tickers
    *,
    sectors_map: Mapping[str, str] = DEFAULT_SECTORS,
    lookbacks: Iterable[int] = (20, 60, 120),
    vol_lb: int = 20,
    delay: int = 1,                               # use info t-1 to avoid lookahead
    winsor_p: float = 0.01,
    macro_tilts: Optional[Dict[str, float]] = None,  # e.g. {"Energy": +0.2, "Defensive": +0.1}
    regime_bias: Optional[str] = None,               # "risk_on" | "risk_off" | None
    long_cap: float = 0.20,                          # per-name cap as share of gross
    short_cap: float = -0.20,
    target_gross: float = 1.0,                       # final |w| sum
    neutralize_sum: bool = True,                     # force sum(w)=0 (market-neutral)
) -> pd.Series:
    """
    Sector rotation via vol-adjusted momentum + optional macro tilts.
    Returns capped, unit-gross weights at the last date.
    """
    px = prices.copy().sort_index()
    tickers = [t for t in sectors_map.keys() if t in px.columns]
    if not tickers:
        return pd.Series(dtype=float)

    px = px.loc[:, tickers]
    need_rows = max(max(lookbacks), vol_lb) + delay + 1
    if px.shape[0] < need_rows:
        return pd.Series(dtype=float)

    mom = _multi_horizon_momentum(px, lookbacks, delay)
    vol = _vol_scale(px, vol_lb, delay)
    raw = (mom / vol).replace([np.inf, -np.inf], np.nan).dropna()

    raw = _winsorize(raw, p=winsor_p)
    raw = _z(raw)

    # Optional macro tilts
    raw = _apply_macro_tilts(raw, sectors_map, macro_tilts or {})

    # Simple regime overlay
    if regime_bias == "risk_on":
        raw = raw.add(pd.Series({t: +0.1 for t, fam in sectors_map.items() if fam in CYCLICALS}), fill_value=0.0)
        raw = raw.add(pd.Series({t: -0.1 for t, fam in sectors_map.items() if fam in DEFENSIVES}), fill_value=0.0)
    elif regime_bias == "risk_off":
        raw = raw.add(pd.Series({t: -0.1 for t, fam in sectors_map.items() if fam in CYCLICALS}), fill_value=0.0)
        raw = raw.add(pd.Series({t: +0.1 for t, fam in sectors_map.items() if fam in DEFENSIVES}), fill_value=0.0)

    # Clip & normalize
    w = raw.clip(lower=short_cap, upper=long_cap)

    # Optional sum-to-zero (market-neutral) before unit-gross
    if neutralize_sum:
        w = w - w.mean()

    w = _unit_gross(w, gross=target_gross)
    return w.sort_values(ascending=False)

# ----------------------------- convenience -----------------------------

def example_universe() -> pd.DataFrame:
    """
    Tiny synthetic example (for tests/notebooks).
    """
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    cols = list(DEFAULT_SECTORS.keys())
    rng = np.random.default_rng(7)
    # geometric random walks per sector with slight different drifts
    drift = np.array([0.08,0.06,0.05,0.07,0.04,0.06,0.10,0.03,0.02,0.05,0.06]) / 252
    vol = np.array([0.18,0.15,0.17,0.20,0.12,0.16,0.30,0.10,0.11,0.19,0.17]) / np.sqrt(252)
    X = np.zeros((len(idx), len(cols)))
    X[0] = 100.0
    for t in range(1, len(idx)):
        ret = drift + vol * rng.standard_normal(len(cols))
        X[t] = X[t-1] * (1.0 + ret)
    return pd.DataFrame(X, index=idx, columns=cols)