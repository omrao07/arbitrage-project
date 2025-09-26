# engines/options/hedging/tail_hedges.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, Callable, List, Any
from math import erf, sqrt, log

TRADING_DAYS = 252

# =============================================================================
# Config
# =============================================================================

HedgeType = Literal["puts", "put_spread", "vix_calls", "combo"]
TriggerMode = Literal["always_on", "regime", "threshold"]

@dataclass
class SignalConfig:
    # Triggers
    trigger_mode: TriggerMode = "regime"
    dd_lookback: int = 100
    dd_trigger: float = -0.08
    rv_lb: int = 21
    rv_trigger: float = 0.30          # annualized
    corr_lb: int = 20
    corr_trigger: float = 0.6
    smooth_span: int = 3              # EMA smoothing for trigger

    # Optional regime input
    use_macro_regime: bool = False
    bearish_regimes: Tuple[str, ...] = ("Stagflation", "Deflation")

    # Sizing
    budget_bps_per_year: float = 150.0
    max_gross: float = 0.02           # cap of NAV deployable at once
    ladder_tranches: int = 3

    # Put ladder
    put_tenor_days: int = 45
    put_moneyness: Tuple[float, ...] = (0.95, 0.90, 0.85)   # K/F
    put_spread_upper_mny: float = 0.98

    # VIX calls
    vix_call_tenor_days: int = 30
    vix_call_strikes: Tuple[float, ...] = (25.0, 35.0, 45.0)

    # Execution cadence
    rebalance: str = "W-FRI"

@dataclass
class BacktestConfig:
    nav0: float = 1_000_000.0
    tc_bps: float = 5.0               # bps on paid premium
    decay_daily: float = 0.0015       # value decay per day (no shock)
    shock_beta_put: float = 6.0       # %Δ option value per -1% underlying return
    shock_beta_vix: float = 0.7       # %Δ option value per +1 VIX point
    roll_days: int = 7                # min days between ladders

# =============================================================================
# Utilities
# =============================================================================

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D", "DAILY"):
        return pd.Series(True, index=idx)
    if f.startswith("W"):
        return pd.Series(1, index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1, index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _drawdown(px: pd.Series, lb: int) -> pd.Series:
    s = px.astype(float).sort_index()
    roll_max = s.rolling(lb, min_periods=1).max()
    return s / roll_max - 1.0

def _realized_vol(px: pd.Series, lb: int) -> pd.Series:
    r = np.log(px).diff() # type: ignore
    return r.rolling(lb).std() * np.sqrt(TRADING_DAYS)

def _avg_pairwise_corr(panel: Optional[pd.DataFrame], lb: int) -> pd.Series:
    if panel is None or panel.empty:
        return pd.Series(dtype=float)
    panel = panel.sort_index()
    idx = panel.index
    out = pd.Series(index=idx, dtype=float)
    for t in range(lb, len(idx)):
        C = panel.iloc[t - lb:t].pct_change().corr()
        if C.shape[0] < 2:
            out.iloc[t] = np.nan
        else:
            a = C.values
            n = a.shape[0]
            mask = ~np.eye(n, dtype=bool)
            out.iloc[t] = np.nanmean(a[mask])
    return out

# --- Black-76 helpers (undiscounted for backtest) ---
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _bs_put(F: float, K: float, T: float, iv: float) -> float:
    if T <= 0 or iv <= 0 or F <= 0 or K <= 0:
        return 0.0
    v = iv * sqrt(T)
    if v == 0:
        return max(K - F, 0.0)
    d1 = (log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return float(K * (1 - _norm_cdf(d2)) - F * (1 - _norm_cdf(d1)))

# =============================================================================
# Signals & Budget
# =============================================================================

def build_tail_signals(
    *,
    index_px: pd.Series,
    vix_level: Optional[pd.Series] = None,
    corr_panel: Optional[pd.DataFrame] = None,
    macro_regimes: Optional[pd.Series] = None,
    cfg: SignalConfig = SignalConfig(),
) -> pd.DataFrame:
    """Return DataFrame: ['drawdown','rv','corr','trigger_raw','trigger']."""
    idx = index_px.sort_index().index
    dd = _drawdown(index_px, cfg.dd_lookback).reindex(idx)
    rv = _realized_vol(index_px, cfg.rv_lb).reindex(idx)
    corr = _avg_pairwise_corr(corr_panel, cfg.corr_lb).reindex(idx) if corr_panel is not None else pd.Series(index=idx, dtype=float)

    if cfg.trigger_mode == "always_on":
        trig = pd.Series(1.0, index=idx)
    else:
        cond_thr = (dd <= cfg.dd_trigger) | (rv >= cfg.rv_trigger) | (corr >= cfg.corr_trigger)
        if cfg.trigger_mode == "regime" and cfg.use_macro_regime and macro_regimes is not None:
            bear = macro_regimes.isin(cfg.bearish_regimes).reindex(idx).fillna(False)
            trig = (cond_thr | bear).astype(float)
        else:
            trig = cond_thr.astype(float)

    out = pd.DataFrame({"drawdown": dd, "rv": rv, "corr": corr, "trigger_raw": trig}, index=idx)
    # debounce with EMA, then hard threshold at 0.5
    ema = out["trigger_raw"].ewm(span=cfg.smooth_span, adjust=False).mean()
    out["trigger"] = (ema >= 0.5).astype(float)
    return out

def compute_premium_budget_path(idx: pd.DatetimeIndex, bps_per_year: float, nav0: float) -> pd.Series:
    daily_bps = float(bps_per_year) / TRADING_DAYS
    accrual = nav0 * (daily_bps * 1e-4)
    return pd.Series(accrual, index=idx).cumsum()

# =============================================================================
# Ladder builder (pricing-aware)
# =============================================================================

# Optional IV provider: (tenor_days:int, strike:float, moneyness:float) -> iv (decimal)
IVProvider = Callable[[int, float, float], float]

def build_hedge_ladder(
    *,
    today_px: float,
    today_vix: Optional[float],
    nav: float,
    spend_available_usd: float,        # <-- FIXED: valid identifier
    cfg: SignalConfig,
    hedge: HedgeType = "combo",
    iv_provider: Optional[IVProvider] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Decide notional to deploy across ladder tranches.
    Returns: {'orders': DataFrame[type, qty, strike, tenor_days, premium$]}
    """
    cols = ["type", "qty", "strike", "tenor_days", "premium$"]
    if spend_available_usd <= 0:
        return {"orders": pd.DataFrame(columns=cols)}

    tranches = max(1, int(cfg.ladder_tranches))
    per_tranche = min(spend_available_usd, cfg.max_gross * nav) / tranches
    legs: List[Dict[str, Any]] = []

    # --- Puts & Put Spreads ---
    if hedge in ("puts", "put_spread", "combo"):
        for m in cfg.put_moneyness[:tranches]:
            K = float(today_px * m)
            iv = iv_provider(cfg.put_tenor_days, K, m) if iv_provider else max(0.08, 0.20 + 0.30 * max(0.0, (1.0 - m)))
            prem_long = _bs_put(today_px, K, cfg.put_tenor_days / TRADING_DAYS, iv)
            qty = int(np.floor(per_tranche / max(prem_long, 1e-9)))
            if qty > 0:
                legs.append({"type": "long_put", "qty": qty, "strike": round(K, 2),
                             "tenor_days": cfg.put_tenor_days, "premium$": float(qty * prem_long)})

        if hedge in ("put_spread", "combo"):
            K_short = float(today_px * cfg.put_spread_upper_mny)
            iv_s = iv_provider(cfg.put_tenor_days, K_short, cfg.put_spread_upper_mny) if iv_provider else max(0.08, 0.20 + 0.30 * max(0.0, (1.0 - cfg.put_spread_upper_mny)))
            prem_s = _bs_put(today_px, K_short, cfg.put_tenor_days / TRADING_DAYS, iv_s)
            for m in cfg.put_moneyness[:tranches]:
                K_long = float(today_px * m)
                if K_long >= K_short:
                    continue
                iv_l = iv_provider(cfg.put_tenor_days, K_long, m) if iv_provider else max(0.08, 0.20 + 0.30 * max(0.0, (1.0 - m)))
                prem_l = _bs_put(today_px, K_long, cfg.put_tenor_days / TRADING_DAYS, iv_l)
                net = max(prem_l - prem_s, 0.0)
                qty = int(np.floor(per_tranche / max(net, 1e-9)))
                if qty > 0:
                    legs.append({"type": "put_spread", "qty": qty, "strike": (round(K_long, 2), round(K_short, 2)),
                                 "tenor_days": cfg.put_tenor_days, "premium$": float(qty * net)})

    # --- VIX Calls (proxy pricing) ---
    if hedge in ("vix_calls", "combo") and today_vix is not None:
        for K in cfg.vix_call_strikes[:tranches]:
            prem = max(0.9 + 0.11 * max(today_vix - (K - 5.0), 0.0), 0.75)
            qty = int(np.floor(per_tranche / prem))
            if qty > 0:
                legs.append({"type": "vix_call", "qty": qty, "strike": float(K),
                             "tenor_days": cfg.vix_call_tenor_days, "premium$": float(qty * prem)})

    orders = pd.DataFrame(legs, columns=cols)
    return {"orders": orders.sort_values("premium$", ascending=False)} if not orders.empty else {"orders": orders}

# =============================================================================
# Backtest (inventory MTM + budget)
# =============================================================================

def backtest_tail_hedge(
    *,
    index_px: pd.Series,
    vix_level: Optional[pd.Series] = None,
    cfg: SignalConfig = SignalConfig(),
    bt: BacktestConfig = BacktestConfig(),
    hedge: HedgeType = "combo",
    iv_provider: Optional[IVProvider] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Lightweight P&L proxy:
      - Accrue premium budget.
      - On trigger, deploy ladder using 'build_hedge_ladder'.
      - Mark inventory daily: convex shock response, otherwise decay.
      - Roll / refresh by cfg.rebalance or every bt.roll_days.
    """
    px = index_px.astype(float).sort_index()
    idx = px.index
    vix = vix_level.reindex(idx).ffill() if vix_level is not None else pd.Series(np.nan, index=idx)

    sig = build_tail_signals(index_px=px, vix_level=vix, cfg=cfg)
    budget_cum = compute_premium_budget_path(idx, cfg.budget_bps_per_year, bt.nav0) # type: ignore

    rb = _rb_mask(idx, cfg.rebalance) # type: ignore

    nav = bt.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    costs = pd.Series(0.0, index=idx, dtype=float)
    inv_value = pd.Series(0.0, index=idx, dtype=float)
    spent = pd.Series(0.0, index=idx, dtype=float)

    last_roll = idx[0]
    total_spent = 0.0
    inventory: List[Dict[str, Any]] = []
    orders_log: List[pd.DataFrame] = []

    def _mark(ret: float, dvix: float) -> float:
        nonlocal inventory
        tot = 0.0
        for leg in inventory:
            if abs(ret) > 0.02 or dvix > 2.0:
                if leg["type"] in ("long_put", "put_spread"):
                    leg["value$"] *= (1.0 + bt.shock_beta_put * max(-ret, 0.0))
                elif leg["type"] == "vix_call":
                    leg["value$"] *= (1.0 + bt.shock_beta_vix * max(dvix, 0.0))
            else:
                leg["value$"] *= (1.0 - bt.decay_daily)
            leg["tenor_days"] -= 1
            tot += leg["value$"]
        inventory = [l for l in inventory if l["tenor_days"] > 0 and l["value$"] > 1.0]
        return tot

    for i, t in enumerate(idx):
        # 1) MTM inventory
        if i == 0:
            ret, dvix = 0.0, 0.0
        else:
            ret = float(px.iloc[i] / px.iloc[i - 1] - 1.0)
            dvix = 0.0 if (pd.isna(vix.iloc[i]) or pd.isna(vix.iloc[i - 1])) else float(vix.iloc[i] - vix.iloc[i - 1])
        inv_t = _mark(ret, dvix)

        # 2) Budget pool
        pool = max(float(budget_cum.iloc[i] - total_spent), 0.0)

        # 3) Ladder placement
        cadence_ok = rb.loc[t] or ((t - last_roll).days >= bt.roll_days)
        if cadence_ok and sig.loc[t, "trigger"] >= 0.5 and pool > 0: # type: ignore
            built = build_hedge_ladder(
                today_px=float(px.loc[t]),
                today_vix=None if pd.isna(vix.iloc[i]) else float(vix.iloc[i]),
                nav=nav,
                spend_available_usd=pool,
                cfg=cfg,
                hedge=hedge,
                iv_provider=iv_provider,
            )
            od = built["orders"]
            if not od.empty:
                tc = float(od["premium$"].sum()) * (bt.tc_bps * 1e-4)
                costs.loc[t] += tc
                total_spent += float(od["premium$"].sum())
                last_roll = t
                for _, r in od.iterrows():
                    inventory.append({"type": r["type"], "value$": float(r["premium$"]), "tenor_days": int(r["tenor_days"])})
                x = od.copy(); x["date"] = t; orders_log.append(x)

        # 4) NAV update (inventory MTM minus cumulative costs)
        inv_value.loc[t] = inv_t
        spent.loc[t] = total_spent
        # NAV = NAV0 + current inventory value - cumulative costs
        nav = bt.nav0 + inv_value.loc[t] - costs.loc[:t].sum()
        nav_path.loc[t] = nav

    orders_df = pd.concat(orders_log, ignore_index=True) if orders_log else pd.DataFrame(columns=["date","type","qty","strike","tenor_days","premium$"])
    summary = pd.DataFrame({
        "nav$": nav_path,
        "inv_value$": inv_value,
        "costs$": costs.cumsum(),
        "budget_spent$": spent,
        "ret_net": nav_path.pct_change().fillna(0.0),
    }, index=idx)

    return {"summary": summary, "signals": sig, "inventory": pd.DataFrame(inventory), "orders_log": orders_df}