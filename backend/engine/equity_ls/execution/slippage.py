# engines/equity_ls/execution/slippage.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal

ModelName = Literal["flat_bps", "sqrt_impact", "amihud", "spread_vol"]

BPS = 1e-4
TRADING_DAYS = 252


# --------------------------- Core cost models ---------------------------

def cost_flat_bps(notional: float, fee_bps: float = 0.5, slip_bps: float = 2.0) -> float:
    """
    Constant fee + slippage in bps of traded notional (both sides).
    """
    return notional * ((fee_bps + slip_bps) * BPS)


def cost_sqrt_impact(
    notional: float,
    adv_usd: float,
    *,
    fee_bps: float = 0.3,
    k_impact: float = 0.10,
    floor_bps: float = 0.5,
) -> float:
    """
    Square-root impact with participation guard:
        cost_bps = fee_bps + floor_bps + k * sqrt(min(1, notional/ADV)) * 1e4
    """
    if adv_usd <= 0 or notional <= 0:
        return 0.0
    part = min(1.0, notional / adv_usd)
    impact_bps = k_impact * np.sqrt(part) * 1e4
    total_bps = fee_bps + floor_bps + impact_bps
    return notional * (total_bps * BPS)


def cost_amihud(
    notional: float,
    dollar_volume: float,
    *,
    alpha: float = 1.0,
    fee_bps: float = 0.3,
    floor_bps: float = 0.3,
) -> float:
    """
    Amihud (2002) illiquidity: impact ∝ $notional / $volume.
    cost ≈ notional * (fee_bps + floor_bps + alpha * (notional / dollar_volume) * 1e4) bps
    """
    if dollar_volume <= 0 or notional <= 0:
        return 0.0
    impact_bps = alpha * (notional / dollar_volume) * 1e4
    total_bps = fee_bps + floor_bps + impact_bps
    return notional * (total_bps * BPS)


def cost_spread_vol(
    notional: float,
    mid_price: float,
    spread_bps: float,
    *,
    daily_vol: float = 0.02,  # 2% daily vol default
    vol_coeff_bps: float = 10.0,
    fee_bps: float = 0.3,
) -> float:
    """
    Spread + volatility model:
      cost_bps = fee_bps + 0.5*spread_bps + vol_coeff_bps * daily_vol
    """
    if notional <= 0:
        return 0.0
    total_bps = fee_bps + 0.5 * max(0.0, spread_bps) + vol_coeff_bps * max(0.0, daily_vol)
    return notional * (total_bps * BPS)


# --------------------------- Dispatcher ---------------------------

def estimate_order_cost(
    *,
    model: ModelName = "sqrt_impact",
    notional: float,
    price: Optional[float] = None,
    adv_usd: Optional[float] = None,
    dollar_volume: Optional[float] = None,
    spread_bps: Optional[float] = None,
    daily_vol: Optional[float] = None,
    params: Optional[Dict] = None,
) -> float:
    """
    Generic order cost estimator (returns USD).
    """
    params = params or {}
    if model == "flat_bps":
        return cost_flat_bps(notional, **{k: params[k] for k in ["fee_bps", "slip_bps"] if k in params})
    elif model == "sqrt_impact":
        return cost_sqrt_impact(
            notional, adv_usd or 0.0,
            fee_bps=params.get("fee_bps", 0.3),
            k_impact=params.get("k_impact", 0.10),
            floor_bps=params.get("floor_bps", 0.5),
        )
    elif model == "amihud":
        return cost_amihud(
            notional, dollar_volume or (adv_usd or 0.0),
            alpha=params.get("alpha", 1.0),
            fee_bps=params.get("fee_bps", 0.3),
            floor_bps=params.get("floor_bps", 0.3),
        )
    elif model == "spread_vol":
        return cost_spread_vol(
            notional, price or 0.0, spread_bps or 0.0,
            daily_vol=(daily_vol if daily_vol is not None else 0.02),
            vol_coeff_bps=params.get("vol_coeff_bps", 10.0),
            fee_bps=params.get("fee_bps", 0.3),
        )
    else:
        raise ValueError(f"Unknown slippage model: {model}")


# --------------------------- Vectorized application ---------------------------

def apply_to_orders(
    orders: pd.DataFrame,
    *,
    model: ModelName = "sqrt_impact",
    price_col: str = "price",
    shares_col: str = "order_shares",
    adv_col: Optional[str] = "adv_usd",       # required for sqrt_impact / amihud
    volume_col: Optional[str] = None,         # alternative to adv_col for amihud
    spread_bps_col: Optional[str] = "spread_bps",
    daily_vol_col: Optional[str] = "daily_vol",
    params: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    orders columns expected:
      - price (float)
      - order_shares (signed or absolute is fine; we use abs() for cost)
      - adv_usd (optional, for sqrt_impact/amihud)
      - spread_bps, daily_vol (optional, for spread_vol)
      - dollar_volume (optional via `volume_col`)
    Returns a copy with:
      - slippage_bps (approx effective)
      - cost_usd
    """
    params = params or {}
    df = orders.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    notional = (df[shares_col].abs() * df[price_col].abs()).fillna(0.0)

    adv = df[adv_col] if (adv_col and adv_col in df.columns) else pd.Series(0.0, index=df.index)
    vol_dollar = df[volume_col] if (volume_col and volume_col in df.columns) else adv

    spread_bps = df[spread_bps_col] if (spread_bps_col and spread_bps_col in df.columns) else pd.Series(0.0, index=df.index)
    daily_vol = df[daily_vol_col] if (daily_vol_col and daily_vol_col in df.columns) else pd.Series(0.02, index=df.index)

    # Compute USD costs
    costs = []
    for i in df.index:
        c = estimate_order_cost(
            model=model,
            notional=float(notional.loc[i]),
            price=float(df[price_col].loc[i]) if price_col in df.columns else None,
            adv_usd=float(adv.loc[i]) if adv is not None is not None else None,
            dollar_volume=float(vol_dollar.loc[i]) if vol_dollar is not None else None,
            spread_bps=float(spread_bps.loc[i]) if spread_bps is not None else None,
            daily_vol=float(daily_vol.loc[i]) if daily_vol is not None else None,
            params=params,
        )
        costs.append(c)

    df["cost_usd"] = np.array(costs, dtype=float)

    # Effective bps ≈ cost / notional * 1e4
    eff_bps = np.where(notional.values > 0, df["cost_usd"].values / notional.values * 1e4, 0.0) # type: ignore
    df["slippage_bps"] = eff_bps
    return df


# --------------------------- Calibration helpers ---------------------------

def fit_sqrt_impact_alpha(
    realized_cost_bps: pd.Series,
    notional: pd.Series,
    adv_usd: pd.Series,
) -> float:
    """
    Estimate k_impact in: cost_bps ≈ fee + floor + k * sqrt(participation)*1e4
    Regress (cost_bps - fee - floor) on sqrt(participation).
    """
    part = (notional / adv_usd.replace(0, np.nan)).clip(lower=0, upper=1).replace([np.inf, -np.inf], np.nan)
    x = np.sqrt(part).fillna(0.0) # type: ignore
    y = realized_cost_bps.fillna(0.0)
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    k_hat = (x * y).sum() / denom / 1e4  # convert back from bps to alpha
    return float(max(0.0, k_hat))


def fit_amihud_alpha(
    realized_cost_bps: pd.Series,
    notional: pd.Series,
    dollar_volume: pd.Series,
) -> float:
    """
    cost_bps ≈ fee + floor + alpha * (notional / dollar_volume) * 1e4
    Regress residual on (notional/volume).
    """
    x = (notional / dollar_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = realized_cost_bps.fillna(0.0)
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    alpha_hat = (x * y).sum() / denom / 1e4
    return float(max(0.0, alpha_hat))


# --------------------------- Quick examples ---------------------------
if __name__ == "__main__":
    # demo with tiny DataFrame
    orders = pd.DataFrame({
        "ticker": ["AAPL","MSFT","XYZ"],
        "order_shares": [1500, -800, 2000],
        "price": [210.0, 420.0, 12.5],
        "adv_usd": [2_000_000_000, 2_500_000_000, 5_000_000],
        "spread_bps": [3.0, 2.0, 20.0],
        "daily_vol": [0.015, 0.012, 0.08],
    }).set_index("ticker")

    # Choose model:
    out = apply_to_orders(
        orders,
        model="sqrt_impact",
        params={"fee_bps": 0.3, "k_impact": 0.12, "floor_bps": 0.5},
    )
    print(out[["order_shares","price","cost_usd","slippage_bps"]])