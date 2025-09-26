# engines/fx/signals/fx_carry.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple

TRADING_DAYS = 252

CarrySource = Literal["rates", "forwards"]
Rebalance = Literal["D", "W-FRI", "M"]

# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

@dataclass
class SignalConfig:
    """
    Build FX carry weights vs USD base.
      - If carry_source='rates', provide short rates per currency (in decimal).
      - If carry_source='forwards', provide forward points (annualized or raw F/S - 1).
    """
    carry_source: CarrySource = "rates"
    mom_lookback_days: int = 252          # momentum filter lookback (12M)
    use_momentum_filter: bool = True
    vol_lookback_days: int = 60           # for inverse-vol sizing
    winsor_p: float = 0.01
    cap_per_ccy: float = 0.15             # |weight| cap per currency
    unit_gross: float = 1.0               # sum(|weights|) after caps
    forwards_are_annualized: bool = False # set True if forward points already annualized

@dataclass
class BacktestConfig:
    rebalance: Rebalance = "W-FRI"        # 'D', 'W-FRI', 'M'
    tc_bps: float = 5.0                   # round-trip cost per rebalance (bps of notional turnover)
    nav0: float = 1_000_000.0

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

def _rebalance_mask(idx: pd.DatetimeIndex, freq: Rebalance) -> pd.Series:
    if freq == "D":
        return pd.Series(True, index=idx)
    return pd.Series(1, index=idx).resample(freq).last().reindex(idx).fillna(0).astype(bool)

def _align_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.reindex(columns=cols).sort_index()

# ---------------------------------------------------------------------
# Core: carry & momentum
# ---------------------------------------------------------------------

def carry_from_rates(
    *,
    usd_short_rate: pd.Series,      # date index, decimal (e.g., 0.053)
    ccy_short_rates: pd.DataFrame,  # date x currency (same units)
) -> pd.DataFrame:
    """
    Carry_t(ccy) ≈ r_ccy_t - r_USD_t  (all in decimal per annum).
    Positive -> long foreign ccy vs USD.
    """
    r_usd = usd_short_rate.sort_index().rename("USD")
    R = ccy_short_rates.sort_index()
    R = _align_cols(R, [c for c in R.columns if c != "USD"])
    r_usd = r_usd.reindex(R.index).ffill()
    return (R.subtract(r_usd, axis=0)).dropna(how="all")

def carry_from_forwards(
    *,
    spot: pd.DataFrame,             # date x currency (USD per 1 unit of ccy)
    forward_points: pd.DataFrame,   # date x currency; either F/S - 1 (if forwards_are_annualized=False)
                                    # or annualized carry (if forwards_are_annualized=True)
    forwards_are_annualized: bool = False,
    annualization_days: int = 252,
) -> pd.DataFrame:
    """
    If forwards_are_annualized is False:
        carry ≈ (F/S - 1) * (TRADING_DAYS / days_to_fwd)  [requires forward_points to embed tenor!]
    In practice, pass forward_points already annualized per currency (e.g., broker data),
    or pass a 1M series and set annualization accordingly. Here we assume you're passing
    a per-day series aligned with spot; we just return it (annualized or not) as carry.
    """
    S = spot.sort_index()
    FP = forward_points.sort_index().reindex(S.index).ffill()
    # If not annualized, we treat FP as instantaneous (already aligned horizon); many feeds provide annualized.
    if forwards_are_annualized:
        return FP
    return FP  # treated as per-annum already for simplicity

def momentum_filter(spot: pd.DataFrame, lookback_days: int = 252) -> pd.Series:
    """
    Classic 12M momentum on FX spot (USD base):
      sign( price_t / price_{t-252} - 1 )
    Returns +1 for uptrend (long), -1 downtrend (short), 0 else per currency at t (last date).
    """
    px = spot.sort_index()
    if len(px) < lookback_days + 2:
        return pd.Series(0.0, index=px.columns, dtype=float)
    mom = (px.iloc[-1] / px.shift(lookback_days).iloc[-1] - 1.0).reindex(px.columns)
    filt = pd.Series(0.0, index=px.columns, dtype=float)
    filt[mom > 0] = +1.0
    filt[mom < 0] = -1.0
    return filt

# ---------------------------------------------------------------------
# Weights construction
# ---------------------------------------------------------------------

def build_fx_carry_weights(
    *,
    spot_usd: pd.DataFrame,                          # date x ccy (USD per 1 unit)
    carry_source: CarrySource,
    usd_short_rate: Optional[pd.Series] = None,      # needed if source='rates'
    ccy_short_rates: Optional[pd.DataFrame] = None,  # needed if source='rates'
    forward_points: Optional[pd.DataFrame] = None,   # needed if source='forwards'
    cfg: SignalConfig = SignalConfig(),
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns (weights at snapshot, diagnostics panel)
      weights: Series by currency (% NAV, unit-gross)
      diag: DataFrame with columns ['carry','inv_vol','mom_filter','raw_score','weight']
    """
    S = spot_usd.sort_index().ffill()
    cols = list(S.columns)

    # 1) Carry series
    if carry_source == "rates":
        if usd_short_rate is None or ccy_short_rates is None:
            raise ValueError("Provide usd_short_rate and ccy_short_rates for carry_source='rates'.")
        C = carry_from_rates(usd_short_rate=usd_short_rate, ccy_short_rates=_align_cols(ccy_short_rates, cols)).reindex(S.index).ffill()
    else:
        if forward_points is None:
            raise ValueError("Provide forward_points for carry_source='forwards'.")
        C = carry_from_forwards(spot=S, forward_points=_align_cols(forward_points, cols),
                                forwards_are_annualized=cfg.forwards_are_annualized).reindex(S.index).ffill()

    if C.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    # 2) Vol estimate for inverse-vol sizing (from daily FX returns)
    rets = S.pct_change()
    vol = (rets.rolling(cfg.vol_lookback_days).std() * np.sqrt(TRADING_DAYS)).iloc[-1].replace(0, np.nan)

    # 3) Snapshot carry (latest)
    carry_t = C.iloc[-1]

    # 4) Optional momentum side filter
    mom = momentum_filter(S, cfg.mom_lookback_days) if cfg.use_momentum_filter else pd.Series(1.0, index=cols)

    # 5) Build raw score: carry × mom_sign; winsorize; inv-vol scale
    raw = _winsorize(carry_t, cfg.winsor_p) * mom.reindex(carry_t.index).fillna(0.0)
    inv_vol = 1.0 / vol.reindex(raw.index)
    score = (raw * inv_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 6) Cap & normalize to unit gross
    w = score.clip(lower=-cfg.cap_per_ccy, upper=+cfg.cap_per_ccy)
    gross = w.abs().sum()
    if gross > 0:
        w = w * (cfg.unit_gross / gross)

    # Diagnostics
    diag = pd.DataFrame({
        "carry": carry_t.reindex(w.index),
        "inv_vol": inv_vol.reindex(w.index),
        "mom_filter": mom.reindex(w.index),
        "raw_score": score.reindex(w.index),
        "weight": w,
    }, index=w.index)

    return w.sort_index(), diag

# ---------------------------------------------------------------------
# Lightweight backtest (USD base)
# ---------------------------------------------------------------------

def backtest_fx_carry(
    *,
    spot_usd: pd.DataFrame,                    # date x ccy (USD per 1 unit)
    cfg: SignalConfig = SignalConfig(),
    bt: BacktestConfig = BacktestConfig(),
    usd_short_rate: Optional[pd.Series] = None,
    ccy_short_rates: Optional[pd.DataFrame] = None,
    forward_points: Optional[pd.DataFrame] = None,
    carry_source: CarrySource = "rates",
) -> Dict[str, pd.DataFrame]:
    """
    Simple daily backtest:
      - At each rebalance, compute weights via build_fx_carry_weights using
        information up to t (contemporaneous snapshot; can shift by 1 for strictness).
      - Daily P&L ≈ Σ w_ccy * FX return_ccy,t * NAV_{t-1}
      - Costs: turnover * tc_bps (bps of notional).
    """
    S = spot_usd.sort_index().ffill()
    idx = S.index
    rb = _rebalance_mask(idx, bt.rebalance) # type: ignore

    weights = pd.DataFrame(0.0, index=idx, columns=S.columns, dtype=float)
    w_last = pd.Series(0.0, index=S.columns, dtype=float)

    nav = bt.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    price_pnl = pd.Series(0.0, index=idx, dtype=float)
    costs = pd.Series(0.0, index=idx, dtype=float)

    # daily FX returns (USD per 1 ccy): long ccy earns + when USD price rises
    ret = S.pct_change().fillna(0.0)

    for t in idx:
        if rb.loc[t]:
            w_t, _ = build_fx_carry_weights(
                spot_usd=S.loc[:t],
                carry_source=carry_source,
                usd_short_rate=None if usd_short_rate is None else usd_short_rate.loc[:t],
                ccy_short_rates=None if ccy_short_rates is None else ccy_short_rates.loc[:t],
                forward_points=None if forward_points is None else forward_points.loc[:t],
                cfg=cfg,
            )
            # align
            w_t = w_t.reindex(S.columns).fillna(0.0)
            weights.loc[t] = w_t

            # turnover cost on rebalance
            turn = (w_t - w_last).abs().sum()
            costs.loc[t] = (bt.tc_bps * 1e-4) * turn * nav
            w_last = w_t
        else:
            weights.loc[t] = w_last

        # daily P&L
        r = ret.loc[t]
        pnl_t = float((w_last.reindex(r.index).fillna(0.0) * r).sum() * nav)
        price_pnl.loc[t] = pnl_t

        nav = nav + pnl_t - costs.loc[t]
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "price_pnl$": price_pnl,
        "costs$": costs,
        "pnl$": price_pnl - costs,
        "nav": nav_path,
    })
    equity_base = summary["nav"].shift(1).fillna(bt.nav0)
    summary["ret_net"] = summary["pnl$"] / equity_base.replace(0, np.nan)

    return {"summary": summary.fillna(0.0), "weights": weights.fillna(0.0)}

# ---------------------------------------------------------------------
# Execution stub: weights → notional orders (for forwards/spot)
# ---------------------------------------------------------------------

def weights_to_notional_orders(
    *,
    weights: pd.Series,       # % of NAV per ccy (positive = long foreign ccy vs USD)
    nav_usd: float,
    last_spot: pd.Series,     # USD per 1 ccy
    min_notional_usd: float = 25_000.0,
) -> pd.DataFrame:
    """
    Turns FX weights into USD notionals you can execute as spot/forwards.
    Output schema: ['side','ccy','usd_notional','units_ccy','px']
    """
    px = last_spot.astype(float)
    w = weights.astype(float).reindex(px.index).fillna(0.0)
    usd_notional = (w * float(nav_usd)).round(2)

    side = np.where(usd_notional > 0, "BUY_FX", np.where(usd_notional < 0, "SELL_FX", "FLAT"))
    units = (usd_notional.abs() / px).round(2)  # ccy units
    df = pd.DataFrame({
        "ccy": px.index,
        "side": side,
        "usd_notional": usd_notional,
        "px": px,
        "units_ccy": units,
    }).set_index("ccy")
    df = df[(df["usd_notional"].abs() >= float(min_notional_usd)) & (df["side"] != "FLAT")]
    # Sort largest first
    return df.sort_values("usd_notional", key=np.abs, ascending=False)


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic G10 demo
    idx = pd.date_range("2023-01-02", periods=400, freq="B")
    rng = np.random.default_rng(42)
    def rw(start):
        x = np.zeros(len(idx)); x[0] = start
        for t in range(1, len(idx)):
            x[t] = x[t-1] * (1 + 0.0 + 0.01*rng.standard_normal())
        return pd.Series(x, index=idx)

    spot = pd.DataFrame({
        "EUR": 1.10 + 0.05*np.sin(np.linspace(0, 10, len(idx))) + 0.02*rng.standard_normal(len(idx)),
        "JPY": 0.009 + 0.0005*rng.standard_normal(len(idx)),
        "GBP": 1.28 + 0.04*np.cos(np.linspace(0, 8, len(idx))) + 0.02*rng.standard_normal(len(idx)),
        "AUD": 0.70 + 0.03*rng.standard_normal(len(idx)),
        "CAD": 0.75 + 0.02*rng.standard_normal(len(idx)),
        "CHF": 1.12 + 0.02*rng.standard_normal(len(idx)),
        "NZD": 0.62 + 0.03*rng.standard_normal(len(idx)),
        "SEK": 0.095 + 0.003*rng.standard_normal(len(idx)),
        "NOK": 0.095 + 0.004*rng.standard_normal(len(idx)),
    }, index=idx).abs()

    # Toy short rates (annualized, decimal)
    r_usd = pd.Series(0.05 + 0.002*np.sin(np.linspace(0, 3, len(idx))), index=idx, name="USD")
    r_ccy = pd.DataFrame({
        "EUR": 0.03 + 0.002*np.cos(np.linspace(0, 4, len(idx))),
        "JPY": -0.001 + 0.001*np.sin(np.linspace(0, 5, len(idx))),
        "GBP": 0.045 + 0.002*np.cos(np.linspace(0, 6, len(idx))),
        "AUD": 0.04 + 0.003*np.sin(np.linspace(0, 2, len(idx))),
        "CAD": 0.042 + 0.002*np.sin(np.linspace(0, 3, len(idx))),
        "CHF": 0.02 + 0.001*np.cos(np.linspace(0, 4, len(idx))),
        "NZD": 0.048 + 0.003*np.sin(np.linspace(0, 5, len(idx))),
        "SEK": 0.038 + 0.002*np.cos(np.linspace(0, 3, len(idx))),
        "NOK": 0.041 + 0.003*np.cos(np.linspace(0, 3, len(idx))),
    }, index=idx)

    # Build weights (rates source)
    cfg = SignalConfig(carry_source="rates", use_momentum_filter=True, unit_gross=1.0, cap_per_ccy=0.2)
    w, diag = build_fx_carry_weights(
        spot_usd=spot,
        carry_source="rates",
        usd_short_rate=r_usd,
        ccy_short_rates=r_ccy,
        cfg=cfg,
    )
    print("Snapshot weights:\n", w.round(4))
    print("\nDiag tail:\n", diag.tail(3))

    # Backtest
    bt = backtest_fx_carry(
        spot_usd=spot,
        cfg=cfg,
        bt=BacktestConfig(rebalance="W-FRI", tc_bps=5.0, nav0=1_000_000),
        usd_short_rate=r_usd,
        ccy_short_rates=r_ccy,
        carry_source="rates",
    )
    print("\nPnL summary tail:\n", bt["summary"].tail())

    # Orders (to execute as forwards/spot)
    last_spot = spot.iloc[-1]
    orders = weights_to_notional_orders(weights=w, nav_usd=1_000_000, last_spot=last_spot, min_notional_usd=25_000)
    print("\nOrders:\n", orders)