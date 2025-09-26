# engines/macro/signals/macro_quadrants.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal

Regime = Literal["Goldilocks", "Reflation", "Stagflation", "Deflation"]

TRADING_DAYS = 252

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class SignalConfig:
    """
    Inputs should be monthly (or weekly) macro nowcasts/surprises:
      - growth: e.g., GDP nowcast (%QoQ ann.), PMI, CESI-G growth components, etc.
      - inflation: e.g., CPI nowcast (%YoY), PPI, inflation swaps, etc.
    You can pass levels or changes. We standardize/z-score them.
    """
    lookback: int = 24                 # z-score window for macro factors
    smooth_ma: int = 3                 # EMA/MA smoothing on macro series (periods)
    use_changes: bool = True           # diff the series before z-scoring
    z_clip: float = 3.0                # cap extreme z-scores
    neutral_band: float = 0.2          # |z| below this treated as 0 (reduces flip/flop)
    hysteresis: int = 1                # require regime to persist N periods before switching (0=off)

@dataclass
class TiltConfig:
    """
    Map regimes → asset tilts (weights). Provide either futures tickers or ETF tickers.
    Weights are relative and will be rescaled to unit gross.
    Positive = long, negative = short.
    """
    # Defaults use ETFs for clarity; replace with your futures symbols (ES, ZN, CL, GC, DXY, etc.)
    regime_tilts: Dict[Regime, Dict[str, float]] = None # type: ignore
    cap_per_asset: float = 0.35
    unit_gross: float = 1.0            # sum(|weights|) after caps
    inv_vol_target: bool = True        # scale by recent asset vol
    vol_lookback: int = 60

    def __post_init__(self):
        if self.regime_tilts is None:
            self.regime_tilts = {
                # ↑growth, ↓inflation
                "Goldilocks": {
                    "SPY": +1.0, "QQQ": +0.6, "TLT": +0.4, "HYG": +0.3,
                    "GLD": -0.2, "DBC": -0.3, "UUP": -0.2,
                },
                # ↑growth, ↑inflation
                "Reflation": {
                    "SPY": +0.6, "IWM": +0.6, "XLE": +0.5, "XLF": +0.3,
                    "DBC": +0.6, "GLD": +0.2, "TLT": -0.6, "UUP": -0.2,
                },
                # ↓growth, ↑inflation
                "Stagflation": {
                    "DBC": +0.7, "GLD": +0.5, "UUP": +0.2,
                    "SPY": -0.6, "TLT": -0.4, "HYG": -0.4,
                },
                # ↓growth, ↓inflation
                "Deflation": {
                    "TLT": +0.9, "IEF": +0.5, "GLD": +0.2,
                    "SPY": -0.5, "DBC": -0.6, "HYG": -0.3, "UUP": +0.2,
                },
            }

@dataclass
class BacktestConfig:
    rebalance_freq: str = "M"          # 'M', 'W-FRI', 'D'
    tc_bps: float = 5.0                # turnover cost per rebalance (bps of notional)
    nav0: float = 1_000_000.0

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean() if span and span > 1 else s

def _standardize(s: pd.Series, lb: int, clip: float) -> pd.Series:
    m = s.rolling(lb).mean()
    v = s.rolling(lb).std()
    z = (s - m) / (v + 1e-12)
    return z.clip(-clip, clip)

def _neutralize_small(z: pd.Series, thr: float) -> pd.Series:
    return z.where(z.abs() >= thr, 0.0)

def _rb_mask(idx: pd.DatetimeIndex, freq: str) -> pd.Series:
    f = freq.upper()
    if f in ("D","DAILY"):
        return pd.Series(True, index=idx)
    if f.startswith("W"):
        return pd.Series(1, index=idx).resample(f).last().reindex(idx).fillna(0).astype(bool)
    return pd.Series(1, index=idx).resample("M").last().reindex(idx).fillna(0).astype(bool)

def _inv_vol_scale(weights: pd.Series, prices: pd.DataFrame, lb: int) -> pd.Series:
    if weights.dropna().empty:
        return weights
    rets = prices.pct_change()
    vol = rets.rolling(lb).std().iloc[-1].replace(0, np.nan)
    scaled = weights * (1.0 / vol.reindex(weights.index))
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if scaled.abs().sum() > 0:
        scaled = scaled / scaled.abs().sum()
    return scaled

# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def compute_macro_zscores(
    growth: pd.Series,
    inflation: pd.Series,
    cfg: SignalConfig = SignalConfig(),
) -> pd.DataFrame:
    """
    Returns DataFrame with columns ['z_growth','z_inflation'] (indexed like inputs).
    """
    g = growth.sort_index().astype(float)
    pi = inflation.sort_index().astype(float)

    if cfg.use_changes:
        g = g.diff()
        pi = pi.diff()

    if cfg.smooth_ma and cfg.smooth_ma > 1:
        g = _ema(g, cfg.smooth_ma)
        pi = _ema(pi, cfg.smooth_ma)

    zg = _standardize(g, cfg.lookback, cfg.z_clip)
    zp = _standardize(pi, cfg.lookback, cfg.z_clip)

    zg = _neutralize_small(zg, cfg.neutral_band)
    zp = _neutralize_small(zp, cfg.neutral_band)

    out = pd.DataFrame({"z_growth": zg, "z_inflation": zp}).dropna(how="all")
    return out

def classify_regime(z_growth: float, z_inflation: float) -> Regime:
    """
    Quadrants:
      Goldilocks :  z_g > 0, z_pi < 0
      Reflation  :  z_g > 0, z_pi > 0
      Stagflation:  z_g < 0, z_pi > 0
      Deflation  :  z_g < 0, z_pi < 0
      Ties (=0) fall to nearest nonzero or kept neutral (treated as 0 direction).
    """
    g = float(z_growth); p = float(z_inflation)
    sgn_g = 1 if g > 0 else (-1 if g < 0 else 0)
    sgn_p = 1 if p > 0 else (-1 if p < 0 else 0)
    if sgn_g >= 0 and sgn_p <= 0:
        return "Goldilocks"
    if sgn_g >= 0 and sgn_p >= 0:
        return "Reflation"
    if sgn_g <= 0 and sgn_p >= 0:
        return "Stagflation"
    return "Deflation"

def regime_series(zdf: pd.DataFrame, hysteresis: int = 1) -> pd.Series:
    """
    Convert z-scores to regime labels with optional hysteresis (persistence requirement).
    """
    labels = zdf.apply(lambda r: classify_regime(r["z_growth"], r["z_inflation"]), axis=1)
    if hysteresis <= 1:
        return labels
    out = []
    last = None; count = 0
    for lab in labels:
        if lab == last:
            count += 1
        else:
            last = lab; count = 1
        if count >= hysteresis:
            out.append(lab)
        else:
            # hold previous confirmed regime if any; otherwise use current tentative
            out.append(out[-1] if out else lab)
    return pd.Series(out, index=labels.index)

# ---------------------------------------------------------------------
# Build tilts (weights by asset) for the latest regime
# ---------------------------------------------------------------------

def build_quadrant_weights(
    regime: Regime,
    *,
    prices: Optional[pd.DataFrame] = None,  # date x asset (optional for inv-vol scaling)
    cfg: TiltConfig = TiltConfig(),
) -> pd.Series:
    """
    Returns Series of asset weights for the given regime.
    If `prices` is provided and cfg.inv_vol_target=True, applies inverse-vol scaling before capping/normalizing.
    """
    if regime not in cfg.regime_tilts:
        return pd.Series(dtype=float)
    w = pd.Series(cfg.regime_tilts[regime], dtype=float)

    if cfg.inv_vol_target and prices is not None and not prices.empty:
        w = _inv_vol_scale(w, prices, cfg.vol_lookback)

    # Cap and normalize to unit gross
    w = w.clip(lower=-cfg.cap_per_asset, upper=+cfg.cap_per_asset)
    if w.abs().sum() > 0:
        w = w * (cfg.unit_gross / w.abs().sum())
    return w.sort_index()

# ---------------------------------------------------------------------
# One-shot snapshot builder
# ---------------------------------------------------------------------

def build_macro_quadrants_snapshot(
    *,
    growth: pd.Series,
    inflation: pd.Series,
    asset_prices: Optional[pd.DataFrame] = None,  # optional for risk targeting
    sig_cfg: SignalConfig = SignalConfig(),
    tilt_cfg: TiltConfig = TiltConfig(),
) -> Dict[str, object]:
    """
    Returns:
      {
        'z': DataFrame with z-growth / z-inflation,
        'regimes': Series of discrete labels,
        'current_regime': Regime,
        'weights': Series of asset weights for the latest regime,
        'diag': DataFrame (last row) with z values & regime,
      }
    """
    z = compute_macro_zscores(growth, inflation, sig_cfg).dropna()
    if z.empty:
        return {"z": z, "regimes": pd.Series(dtype=object), "current_regime": None,
                "weights": pd.Series(dtype=float), "diag": pd.DataFrame()}

    reg = regime_series(z, sig_cfg.hysteresis)
    cur = reg.iloc[-1]
    w = build_quadrant_weights(cur, prices=asset_prices, cfg=tilt_cfg)

    diag = pd.DataFrame({
        "z_growth": [z["z_growth"].iloc[-1]],
        "z_inflation": [z["z_inflation"].iloc[-1]],
        "regime": [cur],
    }, index=[z.index[-1]])

    return {"z": z, "regimes": reg, "current_regime": cur, "weights": w, "diag": diag}

# ---------------------------------------------------------------------
# Lightweight backtest on asset price panel
# ---------------------------------------------------------------------

def backtest_quadrants(
    *,
    growth: pd.Series,
    inflation: pd.Series,
    asset_prices: pd.DataFrame,     # date x asset (e.g., ETFs or futures continuous)
    sig_cfg: SignalConfig = SignalConfig(),
    tilt_cfg: TiltConfig = TiltConfig(),
    bt_cfg: BacktestConfig = BacktestConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Regime-tilt backtest:
      - At each rebalance date, compute regime from macro z-scores (using info to t).
      - Map to asset tilts; (optionally) inverse-vol scale on past returns.
      - Daily P&L = Σ w_a * r_a,t * NAV_{t-1}
      - Costs = turnover * tc_bps on rebalance days.
    """
    z = compute_macro_zscores(growth, inflation, sig_cfg).dropna()
    if z.empty:
        return {"summary": pd.DataFrame(), "weights": pd.DataFrame(), "regimes": pd.Series(dtype=object)} # type: ignore

    # Align to asset price panel
    px = asset_prices.sort_index()
    idx = px.index.intersection(z.index)
    z = z.reindex(idx).dropna()
    px = px.reindex(idx)
    rb = _rb_mask(idx, bt_cfg.rebalance_freq) # type: ignore

    rets = px.pct_change().fillna(0.0)
    weights = pd.DataFrame(0.0, index=idx, columns=px.columns, dtype=float)
    regimes = pd.Series(index=idx, dtype=object)

    nav = bt_cfg.nav0
    nav_path = pd.Series(index=idx, dtype=float)
    price_pnl = pd.Series(0.0, index=idx, dtype=float)
    costs = pd.Series(0.0, index=idx, dtype=float)

    last_w = pd.Series(0.0, index=px.columns, dtype=float)
    last_regime: Optional[Regime] = None

    for t in idx:
        # Determine (or persist) regime with hysteresis
        reg_all = regime_series(z.loc[:t], hysteresis=sig_cfg.hysteresis)
        cur = reg_all.iloc[-1]
        regimes.loc[t] = cur

        if rb.loc[t] or cur != last_regime:
            w = build_quadrant_weights(cur, prices=px.loc[:t], cfg=tilt_cfg).reindex(px.columns).fillna(0.0)
            weights.loc[t] = w
            turn = (w - last_w).abs().sum()
            costs.loc[t] = (bt_cfg.tc_bps * 1e-4) * turn * nav
            last_w = w
            last_regime = cur
        else:
            weights.loc[t] = last_w

        pnl_t = float((last_w * rets.loc[t]).sum() * nav)
        price_pnl.loc[t] = pnl_t
        nav = nav + pnl_t - costs.loc[t]
        nav_path.loc[t] = nav

    summary = pd.DataFrame({
        "price_pnl$": price_pnl,
        "costs$": costs,
        "pnl$": price_pnl - costs,
        "nav": nav_path,
    })
    equity_base = summary["nav"].shift(1).fillna(bt_cfg.nav0)
    summary["ret_net"] = summary["pnl$"] / equity_base.replace(0, np.nan)

    return {"summary": summary.fillna(0.0), "weights": weights.fillna(0.0), "regimes": regimes} # type: ignore

# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic demo (monthly)
    idx = pd.date_range("2015-01-31", periods=120, freq="M")
    rng = np.random.default_rng(5)
    g = pd.Series(np.sin(np.linspace(0, 10, len(idx))) + 0.2*rng.standard_normal(len(idx)), index=idx)
    pi = pd.Series(np.cos(np.linspace(0, 9, len(idx))) + 0.2*rng.standard_normal(len(idx)), index=idx)

    # Toy asset prices
    def rw(start, mu=0.005, sig=0.04):
        x = [start]
        for _ in range(1, len(idx)):
            x.append(x[-1]*(1+mu+sig*rng.standard_normal()))
        return pd.Series(x, index=idx)

    prices = pd.DataFrame({
        "SPY": rw(200),
        "QQQ": rw(100, 0.006, 0.06),
        "TLT": rw(120, 0.002, 0.03),
        "IEF": rw(100, 0.0015, 0.02),
        "HYG": rw(90, 0.003, 0.03),
        "GLD": rw(120, 0.002, 0.025),
        "DBC": rw(18, 0.002, 0.05),
        "UUP": rw(25, 0.001, 0.02),
        "XLE": rw(60, 0.003, 0.07),
        "XLF": rw(30, 0.003, 0.04),
        "IWM": rw(80, 0.004, 0.05),
    })

    snap = build_macro_quadrants_snapshot(growth=g, inflation=pi, asset_prices=prices)
    print("Current regime:", snap["current_regime"])
    print("Weights snapshot:\n", snap["weights"].round(3)) # type: ignore

    bt = backtest_quadrants(growth=g, inflation=pi, asset_prices=prices)
    print("Backtest summary tail:\n", bt["summary"].tail())