# engines/futures_macro/backtest/pnl.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

BPS = 1e-4
TRADING_DAYS = 252


# ---------------------------------------------------------------------
# Contract / Fees config
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ContractSpec:
    """
    Multiplier: $ per 1.00 price point (e.g., ES=50, CL=1000, ZN=1000 per 1.0 price)
    Tick size: minimum price increment (e.g., ES=0.25, CL=0.01, ZN=0.015625)
    Currency: quote currency for this contract's P&L before conversion (e.g., 'USD','EUR','JPY')
    """
    symbol: str
    multiplier: float
    tick_size: float
    currency: str = "USD"


@dataclass(frozen=True)
class FeeSpec:
    """
    Commission and fees per executed contract (one-way), in *quote currency* (pre-FX).
    If you only know round-turn, pass half into commission_per_contract.
    """
    commission_per_contract: float = 1.00  # $ per side
    exchange_fee_per_contract: float = 0.0


@dataclass(frozen=True)
class SlippageSpec:
    """
    Choose slippage model:
      - 'ticks': cost = |traded_contracts| * tick_value * slippage_ticks
      - 'bps':   cost = |traded_notional| * slippage_bps
    """
    model: Literal["ticks", "bps"] = "ticks"
    slippage_ticks: float = 0.25
    slippage_bps: float = 0.5


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

def _align_like(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(index=ref.index, columns=ref.columns).fillna(0.0)

def _to_series_map(d: Dict[str, float], columns: pd.Index, default: float = 1.0) -> pd.Series:
    return pd.Series(d, index=columns, dtype=float).fillna(default)

def _fx_series(
    fx_rates: Optional[pd.DataFrame],
    symbols: pd.Index,
    ccy_map: Dict[str, str],
    base_ccy: str = "USD",
) -> pd.DataFrame:
    """
    fx_rates: DataFrame[date x currency] where each column is BASECCY per 1 unit of 'currency'
              Example with base USD: USD columns are 1.0; EUR column is USD per 1 EUR; JPY column is USD per 1 JPY.
    """
    if fx_rates is None:
        return pd.DataFrame(1.0, index=[0], columns=symbols).iloc[:0]  # empty placeholder
    # Build per-symbol FX column by mapping each symbol to its currency
    out = pd.DataFrame(index=fx_rates.index, columns=symbols, dtype=float)
    for s in symbols:
        ccy = ccy_map.get(s, base_ccy)
        out[s] = 1.0 if ccy == base_ccy else fx_rates[ccy]
    return out


# ---------------------------------------------------------------------
# PnL engine
# ---------------------------------------------------------------------

def compute_futures_pnl(
    *,
    prices: pd.DataFrame,                 # date x symbol (quoted price)
    positions: pd.DataFrame,              # date x symbol (contracts, + long / - short)
    trades: pd.DataFrame,                 # date x symbol (contracts traded; diff of positions)
    contract_specs: Dict[str, ContractSpec],
    fee_spec: FeeSpec = FeeSpec(),
    slippage: SlippageSpec = SlippageSpec(model="ticks", slippage_ticks=0.25),
    fx_rates: Optional[pd.DataFrame] = None,  # date x currency (e.g., EUR, JPY) expressed in USD terms
    base_ccy: str = "USD",
    collateral_apy: float = 0.0,          # interest on NAV (e.g., SOFR); applied on prior-day NAV
    start_nav: float = 1_000_000.0,
) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      - summary: DataFrame with columns ['price_pnl$', 'fees$', 'slippage$', 'carry_collateral$', 'pnl$', 'nav', 'ret_net', 'gross_notional$']
      - per_symbol: DataFrame with per-symbol price pnl and trading costs (multiindex columns: (symbol, metric))
    Notes:
      * Futures P&L is mark-to-market: ΔPrice * Multiplier * Position_prev_day (converted to base ccy by FX).
      * 'Carry' here is collateral interest on NAV; for futures, dividend/borrow don't apply.
      * Roll costs are captured via trades (fees/slippage) + price differences between maturities that you trade.
    """
    # --- Align and sanitize ---
    px = prices.sort_index().copy()
    pos = _align_like(positions.sort_index().copy(), px)
    trd = _align_like(trades.sort_index().copy(), px)
    idx = px.index
    syms = list(px.columns)

    # maps
    mult = _to_series_map({s: contract_specs[s].multiplier for s in syms}, px.columns)
    tick = _to_series_map({s: contract_specs[s].tick_size for s in syms}, px.columns)
    ccy_map = {s: contract_specs[s].currency for s in syms}

    # FX panel per symbol → USD
    if fx_rates is not None and not fx_rates.empty:
        fx_panel = _fx_series(fx_rates.reindex(idx), px.columns, ccy_map, base_ccy=base_ccy).fillna(method="ffill").fillna(1.0) # type: ignore
    else:
        fx_panel = pd.DataFrame(1.0, index=idx, columns=px.columns)

    # --- Price P&L (mark-to-market using SOD positions) ---
    # ΔPx * Multiplier * Contracts_prev_day * FX
    dpx = px.diff().fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)
    price_pnl = (dpx * mult * pos_lag * fx_panel)

    # --- Trading costs ---
    # Commissions/fees: |trades| * (commission + exchange_fee) * FX
    fee_per_ct = fee_spec.commission_per_contract + fee_spec.exchange_fee_per_contract
    fees = (trd.abs() * fee_per_ct * fx_panel).sum(axis=1)

    # Slippage: ticks or bps
    if slippage.model == "ticks":
        tick_value = mult * tick  # $ per tick
        slip = (trd.abs() * tick_value * slippage.slippage_ticks * fx_panel).sum(axis=1)
    else:
        traded_notional = (trd.abs() * px * mult * fx_panel).sum(axis=1)
        slip = traded_notional * (slippage.slippage_bps * BPS)

    # --- Gross notional (exposure) ---
    gross_notional = (pos_lag.abs() * px * mult * fx_panel).sum(axis=1)

    # --- Collateral interest on NAV (simple daily compounding on prior NAV) ---
    daily_r = float(collateral_apy) / TRADING_DAYS
    # Compute equity NAV path iteratively to apply interest on prior-day NAV
    nav = pd.Series(index=idx, dtype=float)
    pnl_price_total = price_pnl.sum(axis=1)
    carry_collateral = pd.Series(0.0, index=idx)
    fees_slip = fees + slip

    nav_prev = start_nav
    for t in idx:
        carry = nav_prev * daily_r
        carry_collateral.loc[t] = carry
        pnl_t = float(pnl_price_total.loc[t] + carry - fees_slip.loc[t])
        nav_t = nav_prev + pnl_t
        nav.loc[t] = nav_t
        nav_prev = nav_t

    # --- Summary & per-symbol tables ---
    summary = pd.DataFrame({
        "price_pnl$": pnl_price_total,
        "fees$": fees,
        "slippage$": slip,
        "carry_collateral$": carry_collateral,
        "pnl$": pnl_price_total + carry_collateral - fees_slip,
        "nav": nav,
        "gross_notional$": gross_notional,
    })
    # Avoid divide-by-zero
    equity_base = summary["nav"].shift(1).replace(0, np.nan).fillna(start_nav)
    summary["ret_net"] = summary["pnl$"] / equity_base

    # Per-symbol detail
    per_sym_cols = {}
    for s in syms:
        per_sym_cols[(s, "price_pnl$")] = price_pnl[s]
        per_sym_cols[(s, "traded_contracts")] = trd[s].abs()
        per_sym_cols[(s, "traded_notional$")] = (trd[s].abs() * px[s] * mult[s] * fx_panel[s])
    per_symbol = pd.DataFrame(per_sym_cols)
    per_symbol.columns = pd.MultiIndex.from_tuples(per_symbol.columns) # type: ignore

    return {"summary": summary.fillna(0.0), "per_symbol": per_symbol.fillna(0.0)}


# ---------------------------------------------------------------------
# Convenience: positions/trades helpers
# ---------------------------------------------------------------------

def make_trades_from_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: compute trades as day-over-day change in contracts.
    """
    positions = positions.sort_index()
    return positions.diff().fillna(positions.iloc[0])


def target_positions_to_trades(
    *,
    current_positions: pd.Series,  # contracts at t-1 (index=symbol)
    target_positions: pd.Series,   # desired contracts at t (index=symbol)
) -> pd.Series:
    """
    One-step trades vector = target - current (per symbol).
    """
    cur = current_positions.reindex(target_positions.index).fillna(0.0)
    tgt = target_positions.astype(float)
    return (tgt - cur)


# ---------------------------------------------------------------------
# Example (synthetic)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic 4-contract portfolio: ES, NQ, CL, GC
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    rng = np.random.default_rng(7)
    def rw(mu, sig, start=100.0):
        x = np.zeros(len(idx)); x[0] = start
        for t in range(1, len(idx)):
            x[t] = x[t-1] * (1 + mu + sig * rng.standard_normal())
        return pd.Series(x, index=idx)

    prices = pd.DataFrame({
        "ES": rw(0.0002, 0.015, 4800),   # S&P 500 futures price level
        "NQ": rw(0.0003, 0.018, 17000),  # Nasdaq
        "CL": rw(0.0001, 0.020, 80),     # WTI
        "GC": rw(0.00012,0.014, 2000),   # Gold
    })

    specs = {
        "ES": ContractSpec("ES", multiplier=50.0, tick_size=0.25, currency="USD"),
        "NQ": ContractSpec("NQ", multiplier=20.0, tick_size=0.25, currency="USD"),
        "CL": ContractSpec("CL", multiplier=1000.0, tick_size=0.01, currency="USD"),
        "GC": ContractSpec("GC", multiplier=100.0, tick_size=0.10, currency="USD"),
    }

    # Toy positions: alternate long/short, rebalance weekly
    pos = pd.DataFrame(0.0, index=idx, columns=list(prices.columns))
    pos.loc[pos.index[::5], "ES"] = 2.0
    pos.loc[pos.index[::5], "NQ"] = -1.0
    pos.loc[pos.index[::5], "CL"] = 3.0
    pos.loc[pos.index[::5], "GC"] = -2.0
    pos = pos.ffill().fillna(0.0)
    trd = make_trades_from_positions(pos)

    out = compute_futures_pnl(
        prices=prices,
        positions=pos,
        trades=trd,
        contract_specs=specs,
        fee_spec=FeeSpec(commission_per_contract=1.2, exchange_fee_per_contract=0.5),
        slippage=SlippageSpec(model="ticks", slippage_ticks=0.25),
        fx_rates=None,  # all USD
        collateral_apy=0.05,
        start_nav=1_000_000.0,
    )

    print(out["summary"].tail())