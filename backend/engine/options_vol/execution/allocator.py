# engines/core/allocator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

@dataclass
class AllocConfig:
    nav_usd: float = 1_000_000.0          # portfolio NAV in USD
    unit_gross: float = 1.0               # weights get scaled to sum(|w|)=unit_gross
    max_gross_leverage: float = 5.0       # cap on sum(|notional|)/NAV
    max_per_asset: float = 0.25           # cap on |w_i| AFTER normalization
    min_notional_usd: float = 5_000.0     # drop dust orders below this absolute notional
    cash_buffer: float = 0.01             # keep 1% NAV unallocated (equities path)
    round_lot: int = 1                    # equities share rounding lot (1 for most)
    allow_short: bool = True

# --- Futures-specific extras ---
@dataclass
class FuturesAllocConfig(AllocConfig):
    contract_multipliers: Optional[Dict[str, float]] = None  # symbol → $ multiplier
    # if provided, we can also size by DV01 (skip for generic)
    max_contracts_per_symbol: Optional[int] = None

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _normalize_weights(w: pd.Series, unit_gross: float, cap: float) -> pd.Series:
    w = w.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if cap is not None:
        w = w.clip(lower=-cap, upper=cap)
    gross = w.abs().sum()
    if gross > 0:
        w = w * (float(unit_gross) / gross)
    return w

def _apply_gross_cap(notional: pd.Series, nav: float, gross_cap: float) -> pd.Series:
    gross = notional.abs().sum()
    cap_val = gross_cap * nav
    if gross > cap_val and gross > 0:
        notional = notional * (cap_val / gross)
    return notional

def target_to_trades(current: pd.Series, target: pd.Series) -> pd.Series:
    """Return TRADE = target - current (aligned, NaNs → 0)."""
    cur = current.astype(float).reindex(target.index).fillna(0.0)
    tar = target.astype(float).reindex(current.index.union(target.index)).fillna(0.0)
    tar = tar.reindex(cur.index).fillna(0.0)
    return (tar - cur)

# ---------------------------------------------------------------------
# Equities / ETFs
# ---------------------------------------------------------------------

def allocate_equities_from_weights(
    *,
    weights: pd.Series,        # % of NAV per asset (can be unnormalized)
    last_prices: pd.Series,    # price per share
    cfg: AllocConfig = AllocConfig(),
) -> Dict[str, pd.Series]:
    """
    Convert weights → dollar and share targets with caps, buffer, and dust filter.
    Returns: {'weights': w, 'dollars': $, 'shares': shares}
    """
    px = last_prices.astype(float).reindex(weights.index).fillna(method="ffill") # type: ignore
    w = _normalize_weights(weights, cfg.unit_gross, cfg.max_per_asset)

    if not cfg.allow_short:
        w = w.clip(lower=0)

    # Cash buffer on NAV
    investable_nav = float(cfg.nav_usd) * max(0.0, 1.0 - cfg.cash_buffer)
    dollars = (w * investable_nav)

    # Drop dust & round to lots
    shares = (dollars / px).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if cfg.round_lot and cfg.round_lot > 1:
        shares = (shares / cfg.round_lot).round().astype(float) * cfg.round_lot
    else:
        shares = shares.round()

    # Recompute dollars after rounding
    dollars = shares * px

    # Gross cap at portfolio level
    dollars = _apply_gross_cap(dollars, cfg.nav_usd, cfg.max_gross_leverage)

    # Min notional filter (zero out tiny lines)
    mask = dollars.abs() >= float(cfg.min_notional_usd)
    shares = shares.where(mask, 0.0)
    dollars = dollars.where(mask, 0.0)

    # Final weights (post rounding)
    final_w = (dollars / max(cfg.nav_usd, 1e-9)).replace([np.inf, -np.inf], 0.0)

    return {"weights": final_w.sort_index(), "dollars": dollars.sort_index(), "shares": shares.sort_index()}

# ---------------------------------------------------------------------
# Futures / FX (contracts sizing by notional)
# ---------------------------------------------------------------------

def allocate_futures_from_weights(
    *,
    weights: pd.Series,                 # % of NAV per symbol
    last_prices: pd.Series,             # futures price per contract "price unit"
    specs: Dict[str, float] | None,     # symbol → multiplier ($ per 1.0 price)
    cfg: FuturesAllocConfig = FuturesAllocConfig(),
) -> Dict[str, pd.Series]:
    """
    Weights → target contracts (rounded), with leverage cap and dust filter.
    `specs`: dict of multipliers; if None, tries cfg.contract_multipliers.
    """
    px = last_prices.astype(float).reindex(weights.index).fillna(method="ffill") # type: ignore
    mult = pd.Series(specs or (cfg.contract_multipliers or {}), dtype=float).reindex(px.index).fillna(1.0)

    w = _normalize_weights(weights, cfg.unit_gross, cfg.max_per_asset)
    notional = w * float(cfg.nav_usd)  # $ target per symbol
    notional = _apply_gross_cap(notional, cfg.nav_usd, cfg.max_gross_leverage)

    # Convert $ notional → contracts
    contract_value = (px * mult).replace(0, np.nan)
    contracts = (notional / contract_value).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Round to nearest whole; apply optional per-symbol cap
    contracts = contracts.round()
    if cfg.max_contracts_per_symbol is not None:
        contracts = contracts.clip(lower=-cfg.max_contracts_per_symbol, upper=cfg.max_contracts_per_symbol)

    # Dust filter by notional AFTER rounding
    final_notional = (contracts * contract_value)
    mask = final_notional.abs() >= float(cfg.min_notional_usd)
    contracts = contracts.where(mask, 0.0)
    final_notional = final_notional.where(mask, 0.0)

    # Post-rounding imputed weights
    final_w = (final_notional / max(cfg.nav_usd, 1e-9)).replace([np.inf, -np.inf], 0.0)

    return {
        "weights": final_w.sort_index(),
        "notional$": final_notional.sort_index(),
        "contracts": contracts.sort_index(),
        "contract_value$": contract_value.sort_index(),
    }

# ---------------------------------------------------------------------
# Rebalance helpers
# ---------------------------------------------------------------------

def rebalance_equities(
    *,
    current_shares: pd.Series,          # current inventory (shares)
    target_shares: pd.Series,           # from allocator
    last_prices: pd.Series,
    min_notional_usd: float = 5_000.0,
) -> pd.DataFrame:
    """
    Build equity orders from target vs current. Returns DataFrame with:
    ['side','shares','notional$','px'] indexed by symbol.
    """
    cur = current_shares.astype(float).reindex(target_shares.index).fillna(0.0)
    tar = target_shares.astype(float).reindex(current_shares.index.union(target_shares.index)).fillna(0.0)
    tar = tar.reindex(cur.index).fillna(0.0)
    trades = (tar - cur)

    px = last_prices.astype(float).reindex(trades.index)
    notionals = (trades.abs() * px)

    df = pd.DataFrame({
        "side": np.where(trades > 0, "BUY", np.where(trades < 0, "SELL", "FLAT")),
        "shares": trades,
        "px": px,
        "notional$": notionals,
    })
    df = df[(df["side"] != "FLAT") & (df["notional$"] >= float(min_notional_usd))]
    return df.sort_values("notional$", ascending=False)

def rebalance_futures(
    *,
    current_contracts: pd.Series,       # current inventory (contracts)
    target_contracts: pd.Series,        # from allocator
    last_prices: pd.Series,
    multipliers: Dict[str, float],
    min_notional_usd: float = 5_000.0,
) -> pd.DataFrame:
    """
    Futures orders: returns DataFrame ['side','contracts','px','notional$'].
    """
    cur = current_contracts.astype(float).reindex(target_contracts.index).fillna(0.0)
    tar = target_contracts.astype(float).reindex(current_contracts.index.union(target_contracts.index)).fillna(0.0)
    tar = tar.reindex(cur.index).fillna(0.0)
    trades = (tar - cur)

    px = last_prices.astype(float).reindex(trades.index)
    mult = pd.Series(multipliers, dtype=float).reindex(trades.index).fillna(1.0)
    notionals = trades.abs() * px * mult

    df = pd.DataFrame({
        "side": np.where(trades > 0, "BUY", np.where(trades < 0, "SELL", "FLAT")),
        "contracts": trades,
        "px": px,
        "notional$": notionals,
    })
    df = df[(df["side"] != "FLAT") & (df["notional$"] >= float(min_notional_usd))]
    return df.sort_values("notional$", ascending=False)

# ---------------------------------------------------------------------
# Quick convenience: one-shot equities allocation
# ---------------------------------------------------------------------

def allocate_and_rebalance_equities(
    *,
    desired_weights: pd.Series,
    last_prices: pd.Series,
    current_shares: Optional[pd.Series] = None,
    cfg: AllocConfig = AllocConfig(),
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    From desired weights → (targets, orders). Great for unit tests or CLI runs.
    """
    out = allocate_equities_from_weights(weights=desired_weights, last_prices=last_prices, cfg=cfg)
    cur = (current_shares or pd.Series(0.0, index=desired_weights.index))
    orders = rebalance_equities(current_shares=cur, target_shares=out["shares"], last_prices=last_prices, min_notional_usd=cfg.min_notional_usd)
    return {**out, "orders": orders}