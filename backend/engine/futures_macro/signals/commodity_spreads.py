# engines/commodities/signals/commodity_spreads.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Optional

# Reuse your futures contract spec (multiplier, tick, currency)
from engines.futures_macro.backtest.pnl import ContractSpec # type: ignore

SignalMode = Literal["mean_revert", "trend", "carry"]
TRADING_DAYS = 252

# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadLeg:
    symbol: str           # e.g., "CLZ5" or "CL1" for front
    weight: float         # coefficient in the spread (e.g., +1, -1, +3/2 etc.)

@dataclass(frozen=True)
class SpreadSpec:
    """
    A named spread: spread_value($) = sum_i weight_i * price_i * multiplier_i
    Examples:
      - WTI calendar: name="CL Cal", legs=[(+1*CL1), (-1*CL2)]
      - 3:2:1 crack:  name="Crack 3:2:1", legs=[(+3*RB1), (+2*HO1), (-1*CL1)]
      - WTI-Brent:    name="WTI-Brent", legs=[(+1*CL1), (-1*CO1)]
    """
    name: str
    legs: List[SpreadLeg]

@dataclass
class SignalConfig:
    lookback: int = 60           # z-score window (for mean-revert/trend)
    entry_z: float = 1.0         # used for mean_revert snapshot
    exit_z: float = 0.25
    cap_per_spread: float = 0.25 # cap absolute weight of a single spread
    unit_gross: float = 1.0      # total |weights| across spreads
    winsor_p: float = 0.01
    carry_annualize: bool = True # for 'carry' mode (calendar spreads)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)

def _zscore(s: pd.Series, lb: int) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    return (s - mu) / (sd + 1e-12)

def _to_point_values(specs: Dict[str, ContractSpec]) -> pd.Series:
    return pd.Series({k: float(v.multiplier) for k, v in specs.items()}, dtype=float)

def _synthetic_spread_value(
    prices: pd.DataFrame, specs: Dict[str, ContractSpec], spread: SpreadSpec
) -> pd.Series:
    """
    Dollar-value spread time series: Σ (w_i * Px_i * multiplier_i).
    All symbols in `spread.legs` must exist in `prices.columns`.
    """
    pv = _to_point_values(specs)
    need = [leg.symbol for leg in spread.legs]
    px = prices.loc[:, [s for s in need if s in prices.columns]].astype(float)
    if px.shape[1] != len(need):
        missing = set(need) - set(px.columns)
        raise KeyError(f"Missing prices for legs: {sorted(missing)}")

    # Σ w_i * Px_i * multiplier_i
    cols = {}
    for leg in spread.legs:
        cols[leg.symbol] = float(leg.weight) * pv[leg.symbol] * px[leg.symbol]
    df = pd.DataFrame(cols, index=prices.index)
    return df.sum(axis=1)

def _latest(ts: pd.Series) -> float:
    return float(ts.iloc[-1]) if not ts.empty else np.nan


# ---------------------------------------------------------------------
# Core: build spreads panel
# ---------------------------------------------------------------------

def build_spreads_panel(
    prices: pd.DataFrame, specs: Dict[str, ContractSpec], spreads: List[SpreadSpec]
) -> pd.DataFrame:
    """
    Returns DataFrame[date x spread_name] with $ spread values.
    """
    out = {}
    for sp in spreads:
        out[sp.name] = _synthetic_spread_value(prices, specs, sp)
    return pd.DataFrame(out).dropna(how="all")


# ---------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------

def _mean_revert_snapshot(spreads_df: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """
    Snapshot weights across spreads based on z-score mean reversion:
      - if z <= -entry_z → long spread (expect revert up)
      - if z >= +entry_z → short spread
      - if |z| < exit_z   → flat
    Then normalize to unit gross, with per-spread cap.
    """
    z = _zscore(spreads_df, cfg.lookback).iloc[-1].dropna() # type: ignore
    raw = pd.Series(0.0, index=z.index, dtype=float)
    raw[z <= -cfg.entry_z] = +1.0
    raw[z >= +cfg.entry_z] = -1.0
    raw[(z.abs() < cfg.exit_z)] = 0.0

    # winsorize tiny numerical noise, then cap and normalize
    raw = _winsorize(raw, cfg.winsor_p).clip(-cfg.cap_per_spread, cfg.cap_per_spread)
    gross = raw.abs().sum()
    if gross > 0:
        raw = raw * (cfg.unit_gross / gross)
    return raw

def _trend_snapshot(spreads_df: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """
    Trend: sign of recent change (z-scored), long rising spreads, short falling.
    """
    d = spreads_df.diff().fillna(0.0)
    z = _zscore(d, cfg.lookback).iloc[-1].dropna() # type: ignore
    w = z.clip(-cfg.cap_per_spread, cfg.cap_per_spread)
    if w.abs().sum() > 0:
        w = w * (cfg.unit_gross / w.abs().sum())
    return w

def _carry_snapshot(prices: pd.DataFrame, specs: Dict[str, ContractSpec], spreads: List[SpreadSpec], cfg: SignalConfig) -> pd.Series:
    """
    Carry proxy for **calendar** spreads (front vs next):
      - For +1*Front -1*Back spreads, carry ≈ (Back - Front) / time_to_maturity (annualized)
      - Positive carry → long spread (benefits as it 'rolls' toward spot).
    If a spread isn't a 2-leg calendar, falls back to 0.
    """
    # You can pass metadata (days to maturity) via symbol naming or an external mapping.
    # Here we try to infer tenor distance if the symbols end with numbers like "CL1", "CL2".
    last = prices.iloc[-1]
    carry_scores = {}

    for sp in spreads:
        if len(sp.legs) != 2:
            carry_scores[sp.name] = 0.0
            continue
        a, b = sp.legs[0], sp.legs[1]
        # Calendar assumed +1 front, -1 back (any scale)
        if np.sign(a.weight) == np.sign(b.weight):
            carry_scores[sp.name] = 0.0
            continue
        # Extract an approximate tenor index (e.g., 'CL1' -> 1)
        def _tenor(sym: str) -> Optional[int]:
            tail = "".join(ch for ch in sym if ch.isdigit())
            return int(tail) if tail else None

        ta, tb = _tenor(a.symbol), _tenor(b.symbol)
        if ta is None or tb is None:
            carry_scores[sp.name] = 0.0
            continue

        # Dollar spread now:
        spr_now = _latest(_synthetic_spread_value(prices, specs, sp))
        # Annualize by tenor gap (months ~ contract steps)
        steps = abs(tb - ta)
        if steps == 0:
            carry_scores[sp.name] = 0.0
            continue
        annualizer = (12.0 / steps) if cfg.carry_annualize else 1.0
        carry_scores[sp.name] = float(spr_now) * annualizer * np.sign(a.weight)  # align with +1 front, -1 back

    s = pd.Series(carry_scores, dtype=float)
    s = _winsorize(s, cfg.winsor_p)
    w = s.clip(-cfg.cap_per_spread, cfg.cap_per_spread)
    if w.abs().sum() > 0:
        w = w * (cfg.unit_gross / w.abs().sum())
    return w


def build_spread_snapshot(
    *,
    prices: pd.DataFrame,                      # date x symbol (futures prices)
    specs: Dict[str, ContractSpec],            # multipliers by symbol
    spreads: List[SpreadSpec],                 # spread definitions
    mode: SignalMode = "mean_revert",          # "mean_revert" | "trend" | "carry"
    cfg: SignalConfig = SignalConfig(),
) -> Dict[str, object]:
    """
    Returns:
      {
        'spreads': DataFrame(date x spread_name)  # $ spread series
        'weights_spread': Series (by spread)
        'weights_symbol': Series (by underlying symbol, aggregated from legs)
        'diag': DataFrame (z, d1, value) at snapshot
      }
    """
    spreads_df = build_spreads_panel(prices, specs, spreads).dropna(how="all")
    if spreads_df.empty or len(spreads_df) < max(20, cfg.lookback + 5):
        return {"spreads": spreads_df, "weights_spread": pd.Series(dtype=float),
                "weights_symbol": pd.Series(dtype=float), "diag": pd.DataFrame()}

    if mode == "mean_revert":
        w_spread = _mean_revert_snapshot(spreads_df, cfg)
    elif mode == "trend":
        w_spread = _trend_snapshot(spreads_df, cfg)
    elif mode == "carry":
        w_spread = _carry_snapshot(prices, specs, spreads, cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Spread weights → symbol weights: sum over legs of (spread_weight * leg_weight * Px*mult / |spread_value|)
    # This approximates a **dollar-neutral decomposition** of spread weight.
    last_px = prices.iloc[-1]
    pv = _to_point_values(specs)

    # Compute per-spread dollar value at snapshot
    spr_val_now = {}
    for sp in spreads:
        spr_val_now[sp.name] = _latest(_synthetic_spread_value(prices, specs, sp))
    spr_val_now = pd.Series(spr_val_now).reindex(w_spread.index).replace(0, np.nan)

    rows = []
    for sp in spreads:
        if sp.name not in w_spread.index or np.isnan(w_spread[sp.name]) or spr_val_now[sp.name] == 0:
            continue
        for leg in sp.legs:
            leg_notional = float(leg.weight) * float(last_px[leg.symbol]) * float(pv[leg.symbol])
            # proportion of spread notional carried by this leg
            frac = leg_notional / float(spr_val_now[sp.name])
            rows.append((leg.symbol, w_spread[sp.name] * frac))

    sym_weights = pd.Series(0.0)
    if rows:
        df = pd.DataFrame(rows, columns=["symbol", "w"])
        sym_weights = df.groupby("symbol")["w"].sum()
        # normalize to unit gross again in case numeric rounding drifted
        if sym_weights.abs().sum() > 0:
            sym_weights *= (cfg.unit_gross / sym_weights.abs().sum())

    # Diagnostics
    zrow = _zscore(spreads_df, cfg.lookback).iloc[-1] # type: ignore
    d1row = spreads_df.diff().iloc[-1]
    diag = pd.DataFrame({
        "value$": spreads_df.iloc[-1],
        "zscore": zrow,
        "d1$": d1row,
        "w_spread": w_spread.reindex(spreads_df.columns),
    })

    return {
        "spreads": spreads_df,
        "weights_spread": w_spread.sort_index(),
        "weights_symbol": sym_weights.sort_index(),
        "diag": diag,
    }