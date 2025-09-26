# engines/stat_arb/execution/allocator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable

# --------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------

@dataclass(frozen=True)
class PairTarget:
    """
    Target for one pair at a single time (all Series indexed by 'ticker').
    Provide either:
      - units (float): +1 unit = +1 share of Y and -β shares of X
        plus y_ticker, x_ticker, and beta (float) for this moment; OR
      - target_shares_y / target_shares_x (floats) if you precomputed per-leg shares.
    """
    name: str
    y_ticker: str
    x_ticker: str
    units: Optional[float] = None
    beta: Optional[float] = None
    target_shares_y: Optional[float] = None
    target_shares_x: Optional[float] = None

@dataclass
class AllocConfig:
    lot_size: int | Dict[str, int] = 1            # per-ticker lot sizes (board lots)
    min_notional: float = 500.0                   # skip tiny trades
    max_participation: float = 0.10               # ≤10% ADV per symbol per rebalance
    allow_shorts: bool = True
    per_symbol_gross_cap: Optional[float] = None  # e.g., 0.10 * NAV (pass as dollars if you set nav$ below)
    nav_usd: Optional[float] = None               # required only if you use per_symbol_gross_cap in % NAV
    # If per_symbol_gross_cap is a fraction (<1), it will be interpreted as % of nav_usd


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _lots_for(tickers: Iterable[str], lot_size: int | Dict[str, int]) -> pd.Series:
    if isinstance(lot_size, int):
        return pd.Series(lot_size, index=list(tickers), dtype=float).clip(lower=1)
    s = pd.Series(lot_size, dtype=float)
    return s.reindex(list(tickers)).fillna(1.0).clip(lower=1)

def _round_to_lot(shares: pd.Series, lots: pd.Series) -> pd.Series:
    return ((shares / lots).round() * lots).astype(float)

def _safe_series_map(d: Optional[Dict[str, float]], index: pd.Index, default: float = np.inf) -> pd.Series:
    if d is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.Series(d, dtype=float).reindex(index).fillna(default).astype(float)

def _resolve_per_symbol_cap(cfg: AllocConfig, tickers: pd.Index) -> Optional[pd.Series]:
    if cfg.per_symbol_gross_cap is None:
        return None
    cap = cfg.per_symbol_gross_cap
    if cap < 1.0:
        if cfg.nav_usd is None:
            raise ValueError("nav_usd must be provided when per_symbol_gross_cap is a fraction of NAV.")
        cap = float(cap) * float(cfg.nav_usd)
    return pd.Series(float(cap), index=tickers, dtype=float)

# --------------------------------------------------------------------
# Core: targets → shares (per pair, then aggregate) → orders
# --------------------------------------------------------------------

def targets_to_target_shares(
    pair_targets: Iterable[PairTarget]
) -> pd.DataFrame:
    """
    Convert a list of PairTarget into per-symbol target shares (aggregated across pairs).
    Returns DataFrame with index=tickers, columns=['target_shares'].
    """
    rows = []
    for p in pair_targets:
        if p.units is not None:
            if p.beta is None:
                raise ValueError(f"PairTarget {p.name}: 'beta' required when using 'units'.")
            y = float(p.units)
            x = -float(p.units) * float(p.beta)
            rows.append((p.y_ticker, y))
            rows.append((p.x_ticker, x))
        elif (p.target_shares_y is not None) and (p.target_shares_x is not None):
            rows.append((p.y_ticker, float(p.target_shares_y)))
            rows.append((p.x_ticker, float(p.target_shares_x)))
        else:
            raise ValueError(f"PairTarget {p.name}: provide either units+beta or both leg shares.")

    df = pd.DataFrame(rows, columns=["ticker", "target_shares"])
    agg = df.groupby("ticker")["target_shares"].sum()
    return agg.to_frame("target_shares")


def generate_orders_from_pairs(
    *,
    pair_targets: Iterable[PairTarget],
    current_pos_shares: pd.Series,        # index=ticker
    last_prices: pd.Series,                # index=ticker
    adv_usd: Optional[Dict[str, float] | pd.Series] = None,
    cfg: AllocConfig = AllocConfig(),
) -> pd.DataFrame:
    """
    Build an executable orders table from pair targets.
    Returns DataFrame indexed by ticker with columns:
      ['target_shares','current_shares','order_shares','side','price',
       'order_notional','adv_usd','participation']
    """
    # 1) Aggregate pair targets → per-symbol target shares
    tgt = targets_to_target_shares(pair_targets)

    # 2) Align inputs
    tickers = tgt.index.union(current_pos_shares.index).union(last_prices.index)
    tgt = tgt.reindex(tickers).fillna(0.0)
    cur = current_pos_shares.reindex(tickers).fillna(0.0).astype(float)
    px = last_prices.reindex(tickers).astype(float)

    # 3) Optional per-symbol gross cap in dollars
    cap_usd = _resolve_per_symbol_cap(cfg, tickers)

    # 4) Round to board lots
    lots = _lots_for(tickers, cfg.lot_size)
    # provisional order (before caps)
    raw_shares = (tgt["target_shares"] - cur)
    raw_shares = _round_to_lot(raw_shares, lots)

    # 5) Apply shorting rule (if disabled, cannot go net short)
    if not cfg.allow_shorts:
        tgt_long_only = cur.add(raw_shares, fill_value=0.0).clip(lower=0.0)
        raw_shares = _round_to_lot(tgt_long_only - cur, lots)

    # 6) Per-symbol gross cap (|target_shares| * price <= cap)
    if cap_usd is not None:
        desired_notional = (cur.add(raw_shares) .abs() * px)
        over = desired_notional > cap_usd
        if over.any():
            # scale orders toward the cap boundary (keep direction)
            room_shares = (cap_usd / px) - cur.abs()
            scaled_shares = np.sign(raw_shares) * np.maximum(0.0, room_shares)
            raw_shares = np.where(over, _round_to_lot(pd.Series(scaled_shares, index=tickers), lots), raw_shares)
            raw_shares = pd.Series(raw_shares, index=tickers).astype(float)

    # 7) ADV participation guard per symbol
    adv_series = _safe_series_map(dict(adv_usd) if isinstance(adv_usd, dict) else (adv_usd.to_dict() if isinstance(adv_usd, pd.Series) else None),
                                  tickers, default=np.inf)
    order_notional = (raw_shares.abs() * px)
    part = (order_notional / adv_series.replace(0, np.inf)).clip(0, 1.0)
    too_big = part > cfg.max_participation
    if too_big.any():
        scale = (cfg.max_participation / (part.replace(0, np.nan))).clip(upper=1.0).fillna(1.0)
        scaled_shares = raw_shares * scale
        raw_shares = _round_to_lot(pd.Series(scaled_shares, index=tickers), lots)
        order_notional = (raw_shares.abs() * px)
        part = (order_notional / adv_series.replace(0, np.inf)).clip(0, 1.0)

    # 8) Dust filter (min_notional)
    keep = order_notional >= float(cfg.min_notional)

    # 9) Build orders frame
    df = pd.DataFrame({
        "target_shares": tgt["target_shares"],
        "current_shares": cur,
        "order_shares": raw_shares,
        "side": np.where(raw_shares > 0, "BUY", np.where(raw_shares < 0, "SELL", "FLAT")),
        "price": px,
        "order_notional": order_notional,
        "adv_usd": adv_series,
        "participation": part,
    }).loc[keep].copy()

    # Remove FLAT rows
    df = df[df["order_shares"] != 0]

    # Sort by notional desc for convenience
    return df.sort_values("order_notional", ascending=False)


# --------------------------------------------------------------------
# Convenience: single-call wrapper (units + betas dicts)
# --------------------------------------------------------------------

def allocate_from_units(
    units_by_pair: Dict[str, float],                 # {"AAA/BBB": 0.7, ...}
    betas_by_pair: Dict[str, float],                 # {"AAA/BBB": 1.2, ...}
    legs_by_pair: Dict[str, Tuple[str, str]],        # {"AAA/BBB": ("AAA","BBB"), ...}
    current_pos_shares: pd.Series,
    last_prices: pd.Series,
    adv_usd: Optional[Dict[str, float] | pd.Series] = None,
    cfg: AllocConfig = AllocConfig(),
) -> pd.DataFrame:
    """
    Quick path when you have units/betas/legs in dicts (like in the simulator output).
    """
    pair_targets = []
    for name, u in units_by_pair.items():
        y, x = legs_by_pair[name]
        b = betas_by_pair[name]
        pair_targets.append(PairTarget(name=name, y_ticker=y, x_ticker=x, units=float(u), beta=float(b)))

    return generate_orders_from_pairs(
        pair_targets=pair_targets,
        current_pos_shares=current_pos_shares,
        last_prices=last_prices,
        adv_usd=adv_usd,
        cfg=cfg,
    )