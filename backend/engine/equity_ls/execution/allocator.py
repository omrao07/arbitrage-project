# engines/equity_ls/execution/allocator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

TRADING_DAYS = 252


# ---------------------------- score → weights ----------------------------

def weights_from_scores(
    scores: pd.Series,
    *,
    max_name: float = 0.03,     # ≤ 3% gross per name
    max_gross: float = 1.00,    # 1x NAV target (before vol targeting)
    max_net: float = 0.05,      # ≤ 5% net
) -> pd.Series:
    """
    Convert cross-sectional scores (+ long, - short) into portfolio weights
    with per-name cap and gross/net limits. Weights sum to ~0 and |w| sum ≤ max_gross.
    """
    s = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(dtype=float)

    # unit-gross, zero-sum base
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    w = z / (z.abs().sum() + 1e-12)

    # per-name clamp
    w = w.clip(-max_name, max_name)

    # gross control
    gross = w.abs().sum()
    if gross > 0:
        w = w * (max_gross / gross)

    # net control: shift towards zero if needed
    net = w.sum()
    if abs(net) > max_net:
        k = (net - np.sign(net) * max_net) / (len(w) + 1e-12)
        w = w - k

    # recheck gross after net shift
    g2 = w.abs().sum()
    if g2 > max_gross:
        w = w * (max_gross / (g2 + 1e-12))

    return w.sort_values(ascending=False)


# ---------------------------- weights → dollars/shares ----------------------------

def dollars_from_weights(
    weights: pd.Series,
    nav: float,
    *,
    cash_buffer: float = 0.02,  # keep 2% NAV as cash
) -> pd.Series:
    """Translate weights to $ notionals given NAV and a cash buffer."""
    investable = nav * max(0.0, 1.0 - cash_buffer)
    return (weights * investable).astype(float)


def shares_from_dollars(
    dollars: pd.Series,
    last_prices: pd.Series,
    lot_size: int | Dict[str, int] = 1,
) -> pd.Series:
    """
    Convert $ targets to shares with optional lot-size rounding (board lots).
    lot_size can be a single int (uniform) or per-ticker dict.
    """
    if isinstance(lot_size, int):
        lots = pd.Series(lot_size, index=dollars.index)
    else:
        lots = pd.Series(lot_size).reindex(dollars.index).fillna(1).astype(int).clip(lower=1)

    shares = (dollars / last_prices.replace(0, np.nan)).fillna(0.0)
    # round to nearest lot
    rounded = (shares / lots).round() * lots
    return rounded.astype(float)


# ---------------------------- positions → orders ----------------------------

def generate_orders(
    current_pos_shares: pd.Series,      # current shares by ticker
    target_pos_shares: pd.Series,       # desired shares by ticker
    last_prices: pd.Series,
    *,
    min_notional: float = 500.0,        # skip tiny trades (<$500)
    adv_usd: Optional[pd.Series] = None,
    max_participation: float = 0.10,    # ≤10% of ADV per symbol per rebalance
    allow_shorts: bool = True,
) -> pd.DataFrame:
    """
    Build an orders table from current → target positions.
    Enforces min-notional and optional %ADV participation caps.
    """
    idx = target_pos_shares.index.union(current_pos_shares.index).unique()
    cur = current_pos_shares.reindex(idx).fillna(0.0)
    tgt = target_pos_shares.reindex(idx).fillna(0.0)

    raw_shares = (tgt - cur)
    notional = (raw_shares.abs() * last_prices.reindex(idx)).fillna(0.0)

    # skip dust
    mask = notional >= float(min_notional)

    # disallow shorts if configured
    if not allow_shorts:
        tgt = tgt.clip(lower=0.0)
        raw_shares = (tgt - cur)
        notional = (raw_shares.abs() * last_prices.reindex(idx)).fillna(0.0)
        mask = notional >= float(min_notional)

    # participation guard
    if adv_usd is not None:
        adv = adv_usd.reindex(idx).fillna(np.inf)
        part = (notional / adv).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        part_cap = part.clip(upper=max_participation)
        # scale down orders that exceed cap
        scale = np.where(part > 0, (part_cap / (part + 1e-12)), 1.0)
        scaled_shares = raw_shares * scale
        scaled_notional = (scaled_shares.abs() * last_prices.reindex(idx)).fillna(0.0)
    else:
        scaled_shares = raw_shares
        scaled_notional = notional

    # final mask after scaling
    mask = mask & (scaled_notional >= float(min_notional))

    df = pd.DataFrame({
        "ticker": idx,
        "target_shares": tgt.values,
        "current_shares": cur.values,
        "order_shares": scaled_shares.values,
        "side": np.where(scaled_shares > 0, "BUY", np.where(scaled_shares < 0, "SELL", "FLAT")),
        "price": last_prices.reindex(idx).values,
        "order_notional": scaled_notional.values,
    }).set_index("ticker")

    df = df[mask]
    # add participation column if ADV provided
    if adv_usd is not None:
        df["adv_usd"] = adv_usd.reindex(df.index).values
        df["participation"] = (df["order_notional"] / df["adv_usd"]).clip(0, 1.0)
    else:
        df["participation"] = np.nan

    return df.sort_values("order_notional", ascending=False)


# ---------------------------- One-shot convenience ----------------------------

def allocate_from_scores(
    scores: pd.Series,
    last_prices: pd.Series,
    nav: float,
    *,
    max_name: float = 0.03,
    max_gross: float = 1.0,
    max_net: float = 0.05,
    cash_buffer: float = 0.02,
    lot_size: int | Dict[str, int] = 1,
) -> Tuple[pd.Series, pd.Series, pd.Series]: # type: ignore
    """
    Convenience: scores → weights → dollars → shares.
    Returns (weights, dollars, shares) aligned to `scores.index`.
    """
    w = weights_from_scores(scores, max_name=max_name, max_gross=max_gross, max_net=max_net)
  dollars_from_weights(w, nav, cash_buffer=cash_buffer) # type: ignore
    sh = shares_from_dollars( last_prices.reindex(w.index), lot_size=lot_size) # type: ignore
    return w , sh, scores # type: ignore