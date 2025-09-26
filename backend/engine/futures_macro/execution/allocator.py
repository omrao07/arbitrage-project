# engines/futures_macro/execution/allocator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Iterable

# Re-use the contract spec from your PnL engine
from engines.futures_macro.backtest.pnl import ContractSpec # type: ignore


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class AllocConfig:
    """
    Controls and risk caps when turning targets into executable futures orders.
    All caps are enforced *after* rounding to whole contracts.
    """
    min_notional_usd: float = 25_000.0          # drop "dust" orders below this notional
    max_contracts_per_symbol: Optional[int] = None  # hard cap on |target contracts| per symbol
    max_gross_leverage: Optional[float] = 10.0      # (sum |notional| / NAV) cap
    per_symbol_gross_cap_usd: Optional[float] = None  # cap |notional| per symbol
    allow_shorts: bool = True
    # Execution hints (metadata only; router can use them)
    default_order_type: str = "MKT"  # "MKT" | "LMT" | "POV" | "ICEBERG"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _point_values(specs: Dict[str, ContractSpec]) -> pd.Series:
    """$ per 1.00 price move for each symbol."""
    return pd.Series({k: float(v.multiplier) for k, v in specs.items()}, dtype=float)

def _ensure_series(x, index: pd.Index, fill=0.0) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index).fillna(fill).astype(float)
    if isinstance(x, dict):
        return pd.Series(x, index=index, dtype=float).fillna(fill)
    return pd.Series(fill, index=index, dtype=float)

def _apply_leverage_cap(
    target_notional: pd.Series, nav_usd: float, max_gross: Optional[float]
) -> pd.Series:
    if max_gross is None or max_gross <= 0:
        return target_notional
    gross = target_notional.abs().sum()
    cap = float(max_gross) * float(nav_usd)
    if gross <= cap or gross <= 0:
        return target_notional
    return target_notional * (cap / gross)

def _apply_symbol_caps(
    target_notional: pd.Series,
    per_symbol_cap_usd: Optional[float],
) -> pd.Series:
    if per_symbol_cap_usd is None or per_symbol_cap_usd <= 0:
        return target_notional
    cap = float(per_symbol_cap_usd)
    return np.sign(target_notional) * np.minimum(target_notional.abs(), cap) # type: ignore

def _round_contracts(contracts: pd.Series) -> pd.Series:
    """Futures trade in whole contracts."""
    return contracts.round().astype(float)

def _contracts_from_notional(
    notional_usd: pd.Series, prices: pd.Series, point_values: pd.Series
) -> pd.Series:
    denom = (prices * point_values).replace(0, np.nan)
    return (notional_usd / denom).fillna(0.0)

def _notional_from_contracts(
    contracts: pd.Series, prices: pd.Series, point_values: pd.Series
) -> pd.Series:
    return (contracts.abs() * prices * point_values)


# ---------------------------------------------------------------------
# Core: weights → contracts → orders
# ---------------------------------------------------------------------

def weights_to_target_contracts(
    *,
    weights: pd.Series,                 # % of NAV per symbol (can sum to >1 in gross terms)
    nav_usd: float,
    last_prices: pd.Series,
    specs: Dict[str, ContractSpec],
    cfg: AllocConfig = AllocConfig(),
) -> pd.Series:
    """
    Convert target *weights* into whole-contract targets with caps.
    Returns Series[index=symbol] of target contracts (signed).
    """
    px = last_prices.astype(float)
    pv = _point_values(specs).reindex(px.index).astype(float)

    # 1) Convert weights → dollar notionals
    tgt_notional = weights.astype(float) * float(nav_usd)

    # 2) Apply portfolio leverage cap
    tgt_notional = _apply_leverage_cap(tgt_notional, nav_usd, cfg.max_gross_leverage)

    # 3) Apply per-symbol gross cap (in USD)
    tgt_notional = _apply_symbol_caps(tgt_notional, cfg.per_symbol_gross_cap_usd)

    # 4) To contracts and round
    contracts = _contracts_from_notional(tgt_notional, px, pv)
    if not cfg.allow_shorts:
        contracts = contracts.clip(lower=0.0)
    contracts = _round_contracts(contracts)

    # 5) Optional per-symbol contracts cap (absolute)
    if cfg.max_contracts_per_symbol is not None:
        m = float(cfg.max_contracts_per_symbol)
        contracts = np.sign(contracts) * np.minimum(contracts.abs(), m)

    return contracts.astype(float) # type: ignore


def generate_orders(
    *,
    current_contracts: pd.Series,        # current positions (contracts)
    target_contracts: pd.Series,         # desired positions (contracts)
    last_prices: pd.Series,              # settlement/last prices
    specs: Dict[str, ContractSpec],
    cfg: AllocConfig = AllocConfig(),
    avg_daily_contracts: Optional[pd.Series] = None,   # optional; to enforce participation caps downstream
    max_participation: Optional[float] = None,         # e.g., 0.15 of ADV in contracts
) -> pd.DataFrame:
    """
    Build an executable orders table (per symbol, one parent order).
    Columns:
      ['current_contracts','target_contracts','order_contracts','side','price',
       'order_notional_usd','point_value','order_type','participation']
    """
    symbols = sorted(set(target_contracts.index).union(current_contracts.index).union(last_prices.index))
    cur = _ensure_series(current_contracts, symbols, fill=0.0) # type: ignore
    tgt = _ensure_series(target_contracts, symbols, fill=0.0) # type: ignore
    px  = _ensure_series(last_prices, symbols, fill=np.nan) # type: ignore
    pv  = _point_values(specs).reindex(symbols)

    # Compute raw orders in contracts (after rounding target)
    tgt = _round_contracts(tgt)
    if not cfg.allow_shorts:
        tgt = tgt.clip(lower=0.0)

    raw_orders = (tgt - cur).round()

    # Drop dust orders by notional
    notional = _notional_from_contracts(raw_orders.abs(), px, pv)
    keep = notional >= float(cfg.min_notional_usd)
    raw_orders = raw_orders.where(keep, 0.0)

    # Optional participation cap (contracts vs ADV contracts)
    participation = pd.Series(0.0, index=symbols, dtype=float)
    if avg_daily_contracts is not None and max_participation is not None and max_participation > 0:
        adv = _ensure_series(avg_daily_contracts, symbols, fill=np.inf).replace(0, np.inf) # type: ignore
        part = (raw_orders.abs() / adv).clip(0, 1.0)
        too_big = part > float(max_participation)
        if too_big.any():
            scale = (float(max_participation) / part.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
            # Scale orders and re-round to whole contracts
            raw_orders = (raw_orders * scale).round()
            notional = _notional_from_contracts(raw_orders.abs(), px, pv)
            participation = (raw_orders.abs() / adv).clip(0, 1.0)
        else:
            participation = part

    # Build orders frame
    side = np.where(raw_orders > 0, "BUY", np.where(raw_orders < 0, "SELL", "FLAT"))
    df = pd.DataFrame({
        "current_contracts": cur,
        "target_contracts": tgt,
        "order_contracts": raw_orders,
        "side": side,
        "price": px,
        "point_value": pv,
        "order_notional_usd": notional,
        "order_type": cfg.default_order_type,
        "participation": participation,
    }, index=symbols)

    # remove FLAT
    df = df[df["order_contracts"] != 0].sort_values("order_notional_usd", ascending=False)
    return df


# ---------------------------------------------------------------------
# Convenience: single-call path from weights
# ---------------------------------------------------------------------

def allocate_from_weights(
    *,
    weights: pd.Series,                 # % of NAV per symbol
    nav_usd: float,
    current_contracts: pd.Series,
    last_prices: pd.Series,
    specs: Dict[str, ContractSpec],
    cfg: AllocConfig = AllocConfig(),
    avg_daily_contracts: Optional[pd.Series] = None,
    max_participation: Optional[float] = None,
) -> pd.DataFrame:
    """
    End-to-end: weights → target contracts (with caps & rounding) → orders DataFrame.
    """
    tgt_contracts = weights_to_target_contracts(
        weights=weights,
        nav_usd=nav_usd,
        last_prices=last_prices,
        specs=specs,
        cfg=cfg,
    )
    return generate_orders(
        current_contracts=current_contracts,
        target_contracts=tgt_contracts,
        last_prices=last_prices,
        specs=specs,
        cfg=cfg,
        avg_daily_contracts=avg_daily_contracts,
        max_participation=max_participation,
    )


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke test
    symbols = ["ES", "NQ", "CL", "GC"]
    prices = pd.Series({"ES": 4800.0, "NQ": 17000.0, "CL": 85.0, "GC": 2050.0})
    specs = {
        "ES": ContractSpec(symbol="ES", multiplier=50.0, tick_size=0.25, currency="USD"),
        "NQ": ContractSpec(symbol="NQ", multiplier=20.0, tick_size=0.25, currency="USD"),
        "CL": ContractSpec(symbol="CL", multiplier=1000.0, tick_size=0.01, currency="USD"),
        "GC": ContractSpec(symbol="GC", multiplier=100.0, tick_size=0.10, currency="USD"),
    }
    weights = pd.Series({"ES": 0.8, "NQ": -0.3, "CL": 0.4, "GC": 0.1})
    cur = pd.Series(0.0, index=symbols)
    cfg = AllocConfig(
        min_notional_usd=25_000,
        max_contracts_per_symbol=50,
        max_gross_leverage=8.0,
        per_symbol_gross_cap_usd=2_000_000,
        allow_shorts=True,
        default_order_type="MKT",
    )
    orders = allocate_from_weights(
        weights=weights,
        nav_usd=10_000_000.0,
        current_contracts=cur,
        last_prices=prices,
        specs=specs,
        cfg=cfg,
        avg_daily_contracts=pd.Series({"ES": 300_000, "NQ": 200_000, "CL": 400_000, "GC": 150_000}),
        max_participation=0.10,
    )
    print(orders)