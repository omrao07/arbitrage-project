# analytics/pnl_attribution.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AttributionConfig:
    base_ccy: str = "USD"
    # column names expected in input data
    col_time: str = "ts"
    col_asset: str = "asset"
    col_strategy: str = "strategy_id"
    col_ccy: str = "ccy"
    # trades
    col_side: str = "side"           # BUY / SELL or +/-
    col_qty: str = "qty"             # position units (shares/contracts) (optional if notional provided)
    col_notional: str = "notional"   # trade notional in trade_ccy (optional)
    col_price: str = "price"         # execution price in trade_ccy per unit
    col_fee: str = "fee"             # absolute fee in trade_ccy (optional)
    col_slip_bps: str = "slip_bps"   # + = cost (bps of notional) (optional)
    col_mid: str = "mid"             # mid-price at trade time (optional)
    # prices & fx
    price_col: str = "close"         # column name in prices panel
    fx_col: str = "fx"               # column name for FX (units of base_ccy per 1 unit of asset ccy)
    # carry/dividends (optional)
    carry_col: str = "carry"         # per-day accrual as % of price (e.g., dividend yield/365)
    # behavior
    assume_mid_equals_price_if_missing: bool = True


# =============================================================================
# Main API
# =============================================================================

def attribute_daily(
    *,
    date: pd.Timestamp,
    positions_prev: pd.DataFrame,
    prices: pd.DataFrame,
    fx: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    carry: Optional[pd.DataFrame] = None,
    asset_ccy: Optional[pd.Series] = None,
    cfg: AttributionConfig = AttributionConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Daily PnL attribution by asset and (optionally) by strategy.

    Inputs
    ------
    date: trading date to attribute (end-of-day timestamp or date)
    positions_prev: index=[asset] (and optionally strategy_id), columns=['qty','price','ccy','strategy_id' (if index not multi)]
        - qty: position quantity at previous close (in units)
        - price: previous close price in asset ccy
        - ccy: asset trading currency (e.g., 'USD','EUR')
    prices: wide or long table that can supply today's close price per asset in asset ccy.
        Options:
          (A) long: index=[date, asset], column=cfg.price_col
          (B) wide: index=[date], columns=assets
    fx: FX rates to base currency. Either:
          (A) long: index=[date, ccy], column=cfg.fx_col (base_ccy per 1 ccy)
          (B) wide: index=[date], columns=ccy codes
    trades: optional DataFrame of fills within 'date' with columns
          [ts, asset, side, qty?, notional?, price, fee?, slip_bps?, mid?, ccy? strategy_id?]
    carry: optional long table (index=[date, asset]) with column cfg.carry_col as daily % accrual on price
    asset_ccy: optional Series index=[asset] -> ccy. Overrides positions_prev['ccy'] if provided.

    Returns
    -------
    dict with:
      - by_asset: DataFrame index=asset, columns=[price_pnl$, fx_pnl$, carry$, trade_pnl$, slippage$, fees$, total$]
      - by_strategy: (if strategy dimension provided) DataFrame index=strategy_id, same columns
      - details_trades: per-trade PnL decomposition in base ccy (if trades provided)
    """
    date = pd.to_datetime(date).normalize()

    # ----- Normalize positions_prev
    pos = positions_prev.copy()
    # Allow strategy dimension as either index level or column
    if cfg.col_strategy in pos.columns and cfg.col_strategy not in pos.index.names:
        pos.set_index([cfg.col_asset, cfg.col_strategy], inplace=True)
    elif cfg.col_asset in pos.columns:
        pos.set_index([cfg.col_asset], inplace=True)

    # Required cols
    for req in ["qty", "price"]:
        if req not in pos.columns:
            raise ValueError(f"positions_prev must include '{req}' column")
    # Currency
    if asset_ccy is not None:
        pos["ccy"] = pos.index.get_level_values(0).map(asset_ccy.to_dict())
    elif "ccy" not in pos.columns:
        pos["ccy"] = cfg.base_ccy

    # ----- Resolve today's price per asset
    px_today = _resolve_price(prices, date, cfg)
    px_prev = pos["price"]
    # align px_today to assets present in pos
    px_today = px_today.reindex(px_prev.index.get_level_values(0))

    # ----- Resolve FX (base per ccy)
    fx_today = _resolve_fx(fx, date, cfg)
    # previous FX: use previous close (assume equal to today if not available)
    fx_prev = _resolve_fx(fx, date - pd.Timedelta(days=1), cfg)
    # Map per-asset FX
    asset_ccy_map = pos["ccy"].droplevel(1) if isinstance(pos.index, pd.MultiIndex) and pos.index.nlevels == 2 else pos["ccy"]
    fx_today_asset = asset_ccy_map.map(fx_today.to_dict()).astype(float)
    fx_prev_asset = asset_ccy_map.map(fx_prev.to_dict()).astype(float).fillna(fx_today_asset)

    # ----- Carry (optional % of price)
    carry_today = None
    if carry is not None and not carry.empty:
        carry_today = _resolve_carry(carry, date, cfg)
        carry_today = carry_today.reindex(px_today.index).fillna(0.0)
    else:
        carry_today = pd.Series(0.0, index=px_today.index)

    # ----- Price and FX PnL (on opening positions)
    qty = pos["qty"].astype(float)
    d_px = (px_today.values - px_prev.values) # type: ignore
    d_fx = (fx_today_asset.values - fx_prev_asset.values) # type: ignore

    price_pnl = qty.values * d_px * fx_today_asset.values
    fx_pnl    = qty.values * px_prev.values * d_fx # type: ignore
    carry_pnl = qty.values * (carry_today.values * px_prev.values) * fx_today_asset.values # type: ignore

    # Build asset-level frame
    idx_assets = px_today.index
    pnl_asset = pd.DataFrame({
        "price_pnl$": price_pnl,
        "fx_pnl$": fx_pnl,
        "carry$": carry_pnl,
    }, index=idx_assets)

    # ----- Trades (optional)
    details_trades = pd.DataFrame(columns=["asset","signed_qty","trade_notional_ccy","trade_pnl$","slippage$","fees$","px_mid","px_fill","fx_rate"])
    trade_pnl_asset = pd.Series(0.0, index=idx_assets)
    slip_asset = pd.Series(0.0, index=idx_assets)
    fee_asset = pd.Series(0.0, index=idx_assets)

    if trades is not None and not trades.empty:
        tr = trades.copy()
        # Restrict to this date
        tr[cfg.col_time] = pd.to_datetime(tr[cfg.col_time])
        m = tr[cfg.col_time].dt.normalize() == date
        tr = tr.loc[m].copy()
        if not tr.empty:
            # Defaults
            if cfg.col_ccy not in tr.columns:
                tr[cfg.col_ccy] = tr[cfg.col_asset].map(asset_ccy_map.to_dict())
            if cfg.col_fee not in tr.columns:
                tr[cfg.col_fee] = 0.0
            if cfg.col_slip_bps not in tr.columns:
                tr[cfg.col_slip_bps] = 0.0

            # Signed qty
            if cfg.col_qty in tr.columns and tr[cfg.col_qty].notna().any():
                sgn = tr[cfg.col_side].astype(str).str.upper().map(lambda x: 1.0 if ("BUY" in x or x in {"+","LONG"}) else -1.0)
                tr["signed_qty"] = sgn * tr[cfg.col_qty].astype(float)
                tr["trade_notional_ccy"] = tr["signed_qty"] * tr[cfg.col_price].astype(float)
            elif cfg.col_notional in tr.columns:
                sgn = tr[cfg.col_side].astype(str).str.upper().map(lambda x: 1.0 if ("BUY" in x or x in {"+","LONG"}) else -1.0)
                tr["trade_notional_ccy"] = sgn * tr[cfg.col_notional].astype(float)
                # derive qty if price provided
                if cfg.col_price in tr.columns and tr[cfg.col_price].notna().any():
                    tr["signed_qty"] = tr["trade_notional_ccy"] / tr[cfg.col_price].astype(float)
                else:
                    tr["signed_qty"] = np.nan
            else:
                raise ValueError("trades must include either qty or notional")

            # Mid for trade PnL
            if cfg.col_mid not in tr.columns or tr[cfg.col_mid].isna().all():
                if cfg.assume_mid_equals_price_if_missing:
                    tr[cfg.col_mid] = tr[cfg.col_price]
                else:
                    tr[cfg.col_mid] = np.nan

            # FX map per trade currency
            fx_map = fx_today.to_dict()
            tr_fx = tr[cfg.col_ccy].map(fx_map).astype(float)

            # Components
            # trade_pnl = (mid - fill) * signed_qty * fx_today  (benefit if you bought below mid / sold above mid)
            tr["trade_pnl$"] = (tr[cfg.col_mid].astype(float) - tr[cfg.col_price].astype(float)) * tr["signed_qty"].astype(float) * tr_fx
            # slippage cost (bps of gross notional, positive cost → negative pnl)
            tr["slippage$"] = - (tr[cfg.col_slip_bps].astype(float) / 1e4) * tr["trade_notional_ccy"].abs().astype(float) * tr_fx
            # fees (negative pnl)
            tr["fees$"] = - tr[cfg.col_fee].astype(float) * tr_fx

            # Aggregate to asset
            g = tr.groupby(cfg.col_asset, dropna=False)[["trade_pnl$","slippage$","fees$"]].sum()
            trade_pnl_asset = trade_pnl_asset.add(g["trade_pnl$"], fill_value=0.0)
            slip_asset = slip_asset.add(g["slippage$"], fill_value=0.0)
            fee_asset = fee_asset.add(g["fees$"], fill_value=0.0)

            # Details for return
            details_trades = tr[[cfg.col_asset, "signed_qty", "trade_notional_ccy", "trade_pnl$", "slippage$", "fees$", cfg.col_mid, cfg.col_price, cfg.col_ccy]].rename(columns={
                cfg.col_asset: "asset",
                cfg.col_mid: "px_mid",
                cfg.col_price: "px_fill",
                cfg.col_ccy: "ccy",
            })

    pnl_asset["trade_pnl$"] = trade_pnl_asset.reindex(pnl_asset.index).fillna(0.0)
    pnl_asset["slippage$"]  = slip_asset.reindex(pnl_asset.index).fillna(0.0)
    pnl_asset["fees$"]      = fee_asset.reindex(pnl_asset.index).fillna(0.0)
    pnl_asset["total$"]     = pnl_asset.sum(axis=1)

    # ----- By-strategy rollup if strategy dimension present
    pnl_by_strategy = pd.DataFrame()
    if isinstance(pos.index, pd.MultiIndex) and cfg.col_strategy in pos.index.names:
        # expand asset-level pnl to (asset,strategy) using qty shares
        qty_total_by_asset = pos["qty"].groupby(level=0).sum().replace(0, np.nan)
        weight = (pos["qty"] / qty_total_by_asset).fillna(0.0)
        # multiply each component by weights then sum by strategy level
        comp_cols = ["price_pnl$","fx_pnl$","carry$","trade_pnl$","slippage$","fees$","total$"]
        expanded = []
        for comp in comp_cols:
            s = pnl_asset[comp].reindex(weight.index.get_level_values(0)).values * weight.values # type: ignore
            expanded.append(pd.Series(s, index=weight.index, name=comp))
        strat = pd.concat(expanded, axis=1).groupby(level=1).sum()
        pnl_by_strategy = strat

    return {
        "by_asset": pnl_asset.sort_index(),
        "by_strategy": pnl_by_strategy.sort_index() if not pnl_by_strategy.empty else pnl_by_strategy,
        "details_trades": details_trades.sort_values("asset") if not details_trades.empty else details_trades,
    }


def attribute_range(
    *,
    dates: List[pd.Timestamp],
    positions_prev_by_day: Dict[pd.Timestamp, pd.DataFrame],
    prices: pd.DataFrame,
    fx: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    carry: Optional[pd.DataFrame] = None,
    asset_ccy: Optional[pd.Series] = None,
    cfg: AttributionConfig = AttributionConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Loop over dates and stack daily attributions. Returns panel dataframes with a 'date' column.
    """
    rows_asset = []
    rows_strat = []
    trades_rows = []

    for dt in dates:
        res = attribute_daily(
            date=dt,
            positions_prev=positions_prev_by_day[dt],
            prices=prices,
            fx=fx,
            trades=trades,
            carry=carry,
            asset_ccy=asset_ccy,
            cfg=cfg,
        )
        ba = res["by_asset"].copy()
        ba.insert(0, "date", pd.to_datetime(dt).normalize())
        rows_asset.append(ba.reset_index().rename(columns={"index": cfg.col_asset, 0: cfg.col_asset}))

        if isinstance(res["by_strategy"], pd.DataFrame) and not res["by_strategy"].empty:
            bs = res["by_strategy"].copy()
            bs.insert(0, "date", pd.to_datetime(dt).normalize())
            rows_strat.append(bs.reset_index().rename(columns={"index": cfg.col_strategy, 0: cfg.col_strategy}))

        if isinstance(res["details_trades"], pd.DataFrame) and not res["details_trades"].empty:
            dtr = res["details_trades"].copy()
            dtr.insert(0, "date", pd.to_datetime(dt).normalize())
            trades_rows.append(dtr)

    by_asset_panel = pd.concat(rows_asset, ignore_index=True) if rows_asset else pd.DataFrame()
    by_strat_panel = pd.concat(rows_strat, ignore_index=True) if rows_strat else pd.DataFrame()
    trades_panel   = pd.concat(trades_rows, ignore_index=True) if trades_rows else pd.DataFrame()

    return {
        "by_asset": by_asset_panel,
        "by_strategy": by_strat_panel,
        "details_trades": trades_panel
    }


# =============================================================================
# Helpers
# =============================================================================

def _resolve_price(prices: pd.DataFrame, date: pd.Timestamp, cfg: AttributionConfig) -> pd.Series:
    if prices.index.nlevels == 2:
        # long index [date, asset]
        if cfg.price_col not in prices.columns:
            raise ValueError(f"prices must include column '{cfg.price_col}'")
        if not isinstance(prices.index, pd.MultiIndex) or prices.index.names[:2] != ["date","asset"]:
            # try to coerce names
            prices = prices.copy()
            prices.index = prices.index.set_names(["date","asset"])
        out = prices.xs(date, level=0, drop_level=False)
        return out.reset_index("date").drop(columns="date")[cfg.price_col]
    else:
        # wide: index=date, columns=assets
        row = prices.reindex(index=[date]).ffill().iloc[-1]
        if isinstance(row, pd.Series):
            return row
        raise ValueError("prices not understood")

def _resolve_fx(fx: pd.DataFrame, date: pd.Timestamp, cfg: AttributionConfig) -> pd.Series:
    if fx.empty:
        return pd.Series(dtype=float)
    if fx.index.nlevels == 2:
        # long [date, ccy]
        if cfg.fx_col not in fx.columns:
            raise ValueError(f"fx must include column '{cfg.fx_col}'")
        if not isinstance(fx.index, pd.MultiIndex) or fx.index.names[:2] != ["date","ccy"]:
            fx = fx.copy()
            fx.index = fx.index.set_names(["date","ccy"])
        out = fx.xs(date, level=0, drop_level=False)
        return out.reset_index("date").drop(columns="date")[cfg.fx_col]
    else:
        # wide: index=date, columns=ccy
        row = fx.reindex(index=[date]).ffill().iloc[-1]
        return row if isinstance(row, pd.Series) else pd.Series(dtype=float)

def _resolve_carry(carry: pd.DataFrame, date: pd.Timestamp, cfg: AttributionConfig) -> pd.Series:
    # long [date, asset] with column cfg.carry_col as DAILY PCT of price
    if carry.index.nlevels == 2:
        if cfg.carry_col not in carry.columns:
            raise ValueError(f"carry must include column '{cfg.carry_col}'")
        if not isinstance(carry.index, pd.MultiIndex) or carry.index.names[:2] != ["date","asset"]:
            carry = carry.copy()
            carry.index = carry.index.set_names(["date","asset"])
        out = carry.xs(date, level=0, drop_level=False)
        return out.reset_index("date").drop(columns="date")[cfg.carry_col]
    else:
        # wide not supported for carry → convert before calling
        raise ValueError("carry must be indexed by [date, asset]")


# =============================================================================
# Example (synthetic)
# =============================================================================

if __name__ == "__main__":
    np.random.seed(3)
    date = pd.Timestamp("2025-09-10")

    # Positions at previous close
    pos = pd.DataFrame({
        "asset": ["AAPL","MSFT","BUND"],
        "strategy_id": ["mom","mom","rates"],
        "qty": [100, 80, 5],
        "price": [200.0, 350.0, 132.0],
        "ccy": ["USD","USD","EUR"],
    })

    # Today prices (long format)
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2025-09-10")], ["AAPL","MSFT","BUND"]], names=["date","asset"])
    prices = pd.DataFrame({"close": [203.0, 342.0, 131.2]}, index=idx)

    # FX (USD base): USD=1.0, EUR=1.08
    idx_fx = pd.MultiIndex.from_product([[pd.Timestamp("2025-09-09"), pd.Timestamp("2025-09-10")], ["USD","EUR"]], names=["date","ccy"])
    fx = pd.DataFrame({"fx":[1.0,1.08, 1.0,1.07]}, index=idx_fx)

    # Carry (daily % of price)
    idx_c = pd.MultiIndex.from_product([[pd.Timestamp("2025-09-10")], ["AAPL","MSFT","BUND"]], names=["date","asset"])
    carry = pd.DataFrame({"carry":[0.0, 0.0, 0.0001]}, index=idx_c)

    # Trades during the day
    trades = pd.DataFrame([
        {"ts":"2025-09-10 14:31", "asset":"AAPL","side":"BUY","qty":20,"price":202.5,"fee":2.0,"slip_bps":1.0,"mid":202.7,"ccy":"USD"},
        {"ts":"2025-09-10 15:05", "asset":"BUND","side":"SELL","qty":1,"price":131.25,"fee":0.5,"slip_bps":0.5,"mid":131.22,"ccy":"EUR"},
    ])

    out = attribute_daily(
        date=date,
        positions_prev=pos,
        prices=prices,
        fx=fx,
        trades=trades,
        carry=carry,
        cfg=AttributionConfig(base_ccy="USD")
    )

    print("BY ASSET\n", out["by_asset"], "\n")
    print("BY STRATEGY\n", out["by_strategy"], "\n")
    print("TRADE DETAILS\n", out["details_trades"])