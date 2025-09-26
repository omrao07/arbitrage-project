# engines/futures_macro/backtest/simulator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Literal, List, Tuple

from .pnl import compute_futures_portfolio_pnl, ContractSpec # type: ignore

Freq = Literal["D","W-FRI","M","Q"]

# --------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------

@dataclass
class StratConfig:
    rebalance: Freq = "W-FRI"
    vol_target: float = 0.12         # annualized portfolio vol target
    contract_vol_lookback: int = 60
    unit_gross: float = 1.0          # gross exposure per rebalance
    max_leverage: float = 5.0        # cap gross leverage
    min_notional: float = 50_000.0   # skip tiny trades
    fee_bps: float = 0.2
    slippage_bps: float = 0.5
    fin_bps: float = 10.0            # financing cost per annum


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _rebalance_mask(idx: pd.DatetimeIndex, freq: Freq) -> pd.Series:
    if freq == "D":
        return pd.Series(True, index=idx)
    return pd.Series(1, index=idx).resample(freq).last().reindex(idx).fillna(0).astype(bool)

def _zscore(s: pd.Series, lb: int) -> pd.Series:
    mu = s.rolling(lb).mean()
    sd = s.rolling(lb).std()
    return (s - mu) / (sd + 1e-12)

# --------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------

def simulate_futures_macro(
    prices: pd.DataFrame,
    signals: pd.DataFrame,          # date x contract, e.g. momentum z-scores
    specs: Dict[str, ContractSpec], # contract metadata (point value, etc.)
    cfg: StratConfig = StratConfig(),
    nav0: float = 10_000_000.0,
) -> Dict[str,object]:
    """
    Backtest a macro futures portfolio.

    Inputs:
      prices  : settlement prices (date x contract)
      signals : standardized signals (date x contract), e.g. momentum z
      specs   : dict of ContractSpec per contract
    Returns:
      dict with:
        - 'weights': DataFrame of weights (% NAV)
        - 'positions': contracts x dates (number of contracts)
        - 'pnl': portfolio PnL breakdown
    """
    px = prices.sort_index()
    sig = signals.reindex(px.index).fillna(0.0)

    idx = px.index
    rb_mask = _rebalance_mask(idx, cfg.rebalance) # type: ignore

    # compute vol estimates per contract
    rets = px.pct_change()
    vol = rets.rolling(cfg.contract_vol_lookback).std() * np.sqrt(252)

    weights = pd.DataFrame(0.0, index=idx, columns=px.columns)
    pos = pd.DataFrame(0.0, index=idx, columns=px.columns)

    nav = nav0
    last_pos = pd.Series(0.0, index=px.columns)

    for t in idx:
        if rb_mask.loc[t]:
            # risk-normalized signal → weights
            z = sig.loc[t]
            vol_t = vol.loc[t].replace(0,np.nan)
            inv_vol = 1.0 / vol_t
            raw = z * inv_vol
            if raw.abs().sum() > 0:
                w = raw / raw.abs().sum()
            else:
                w = pd.Series(0.0, index=px.columns)

            w = w * cfg.unit_gross
            # leverage cap
            if w.abs().sum() > cfg.max_leverage:
                w = w * (cfg.max_leverage / (w.abs().sum()+1e-12))
            weights.loc[t] = w
            # translate to contracts
            dollars = w * nav
            contracts = dollars / (px.loc[t] * pd.Series({k:specs[k].point_value for k in px.columns})) # type: ignore
            pos.loc[t] = contracts
            last_pos = contracts
        else:
            weights.loc[t] = weights.loc[t-1]
            pos.loc[t] = last_pos

    # Compute PnL
    pnl = compute_futures_portfolio_pnl(px, pos, specs, fee_bps=cfg.fee_bps,
                                        slippage_bps=cfg.slippage_bps, fin_bps=cfg.fin_bps)

    return {
        "weights": weights,
        "positions": pos,
        "pnl": pnl,
    }

# --------------------------------------------------------------------
# Example
# --------------------------------------------------------------------
if __name__ == "__main__":
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        "S&P": 3000*np.exp(np.cumsum(0.0001+0.01*rng.standard_normal(len(idx)))),
        "Gold": 1500*np.exp(np.cumsum(0.00005+0.012*rng.standard_normal(len(idx)))),
    }, index=idx)

    signals = pd.DataFrame({
        "S&P": _zscore(prices["S&P"].pct_change().cumsum(),60),
        "Gold": -_zscore(prices["Gold"].pct_change().cumsum(),60),
    }, index=idx)

    from engines.futures_macro.backtest.pnl import ContractSpec # type: ignore
    specs = {
        "S&P": ContractSpec(point_value=50.0),
        "Gold": ContractSpec(point_value=100.0),
    }

    out = simulate_futures_macro(prices, signals, specs)
    print("PnL summary:\n", out["pnl"]["summary"].tail()) # type: ignore