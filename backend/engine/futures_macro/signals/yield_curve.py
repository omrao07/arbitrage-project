# engines/rates/signals/yield_curve.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

# ----------------------------- Types & config -----------------------------

Tenor = float  # years
SignalMode = Literal["slope", "curvature", "pca", "combo"]

@dataclass
class CurveConfig:
    tenors: List[Tenor] = None           # type: ignore # maturities in years (e.g., [0.5,2,5,10,30])
    interp_kind: Literal["linear"] = "linear"
    z_lookback: int = 252
    winsor_p: float = 0.01
    unit_gross: float = 1.0
    cap_leg: float = 0.5                 # per-leg cap when mapping to instruments

    # combo weights (for SignalMode="combo")
    w_slope: float = 0.45
    w_curve: float = 0.35
    w_pca2: float = 0.20                 # PCA slope emphasis

    def __post_init__(self):
        if self.tenors is None:
            self.tenors = [0.5, 2.0, 5.0, 10.0, 30.0]

@dataclass
class MapConfig:
    """
    Map curve tilts → tradable futures & DV01s (approx; per 1 contract).
    Replace with your latest DV01 estimates if you have them.
    """
    # keys must match symbols in your execution stack
    dv01_usd: Dict[str, float] = None     # type: ignore # $ per 1bp move (per contract)
    tenor_map: Dict[str, Tenor] = None    # type: ignore # symbol → effective maturity (years)

    def __post_init__(self):
        if self.dv01_usd is None:
            # Rough UST futures DV01 per contract (update as needed)
            self.dv01_usd = {"ZT": 45.0, "ZF": 80.0, "TY": 120.0, "US": 190.0}
        if self.tenor_map is None:
            self.tenor_map = {"ZT": 2.0, "ZF": 5.0, "TY": 10.0, "US": 30.0}

# ----------------------------- Utils -----------------------------

def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty: return s
    lo, hi = s.quantile([p, 1-p]); return s.clip(lo, hi)

def _zscore(s: pd.Series, lb: int) -> pd.Series:
    mu, sd = s.rolling(lb).mean(), s.rolling(lb).std()
    return (s - mu) / (sd + 1e-12)

def _interp_row(curve_row: pd.Series, want: List[Tenor]) -> pd.Series:
    x = np.array(curve_row.index, dtype=float); y = curve_row.values.astype(float)
    order = np.argsort(x); x = x[order]; y = y[order]
    out = {}
    for t in want:
        if t <= x[0]:
            out[t] = y[0] + (y[1]-y[0])*(t-x[0])/(x[1]-x[0]) if len(x)>1 else y[0]
        elif t >= x[-1]:
            out[t] = y[-2] + (y[-1]-y[-2])*(t-x[-2])/(x[-1]-x[-2]) if len(x)>1 else y[-1]
        else:
            j = np.searchsorted(x, t); x0,x1=x[j-1],x[j]; y0,y1=y[j-1],y[j]
            out[t] = y0 + (y1-y0)*(t-x0)/(x1-x0)
    return pd.Series(out, dtype=float)

def _align_curve_panel(yields: pd.DataFrame) -> pd.DataFrame:
    # columns should be floats (years)
    cols = [float(c) for c in yields.columns]
    df = yields.copy(); df.columns = cols
    return df.sort_index().astype(float)

# ----------------------------- Core curve features -----------------------------

def interpolate_panel(yields: pd.DataFrame, tenors: List[Tenor]) -> pd.DataFrame:
    """
    Input yields: DataFrame[date x (float years)] in decimal (e.g., 0.042)
    Output: yields at requested tenors for each date.
    """
    y = _align_curve_panel(yields)
    out = pd.DataFrame(index=y.index, columns=tenors, dtype=float)
    for dt, row in y.iterrows():
        out.loc[dt] = _interp_row(row.dropna(), tenors) # type: ignore
    return out

def slope_series(yc: pd.DataFrame, short_t: Tenor = 2.0, long_t: Tenor = 10.0) -> pd.Series:
    return (yc[long_t] - yc[short_t]).rename(f"Slope_{long_t}-{short_t}")

def curvature_series(yc: pd.DataFrame, short_t: Tenor = 2.0, mid_t: Tenor = 5.0, long_t: Tenor = 10.0) -> pd.Series:
    # 2*mid - short - long (positive when humped)
    return (2*yc[mid_t] - yc[short_t] - yc[long_t]).rename(f"Curv_{mid_t}*2-{short_t}-{long_t}")

def pca_factors(yc: pd.DataFrame, n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    PCA on standardized yields across tenors.
    Returns (scores_df, loadings_df).
    """
    X = yc.copy().dropna().values
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    scores = pd.DataFrame(U[:, :n] * s[:n], index=yc.dropna().index, columns=[f"PC{i+1}" for i in range(n)])
    loadings = pd.DataFrame(Vt[:n, :].T, index=yc.columns, columns=[f"PC{i+1}" for i in range(n)])
    return scores.reindex(yc.index), loadings

# ----------------------------- Signal builders -----------------------------

def build_signal(
    *,
    yields: pd.DataFrame,          # date x maturityYears (floats), decimal yields
    mode: SignalMode = "combo",
    cfg: CurveConfig = CurveConfig(),
) -> Dict[str, object]:
    """
    Returns dict with:
      - 'yc': interpolated curve (date x tenor)
      - 'features': DataFrame with slope/curvature/PCA scores
      - 'signal': Series (latest snapshot score)
      - 'diag': DataFrame (last row)
    """
    yc = interpolate_panel(yields, cfg.tenors).dropna(how="all")
    if yc.empty or len(yc) < max(60, cfg.z_lookback + 5):
        return {"yc": yc, "features": pd.DataFrame(), "signal": pd.Series(dtype=float), "diag": pd.DataFrame()}

    sl = slope_series(yc, 2.0, 10.0)
    cu = curvature_series(yc, 2.0, 5.0, 10.0)
    pca_scores, pca_load = pca_factors(yc, 3)
    feats = pd.concat([sl, cu, pca_scores], axis=1)

    # Snapshot scoring (higher → prefer steepener, positive curvature, positive PC2)
    z_sl = _zscore(sl, cfg.z_lookback)
    z_cu = _zscore(cu, cfg.z_lookback)
    z_pc2 = _zscore(pca_scores["PC2"], cfg.z_lookback)

    t = yc.index[-1]
    s_sl, s_cu, s_pc2 = z_sl.loc[t], z_cu.loc[t], z_pc2.loc[t]
    s_sl, s_cu, s_pc2 = [_winsorize(pd.Series([x]), cfg.winsor_p).iloc[0] for x in [s_sl, s_cu, s_pc2]]

    if mode == "slope":
        snap = pd.Series({"slope": s_sl})
    elif mode == "curvature":
        snap = pd.Series({"curvature": s_cu})
    elif mode == "pca":
        snap = pd.Series({"pc2": s_pc2})
    else:
        snap = pd.Series({
            "slope": cfg.w_slope * s_sl,
            "curvature": cfg.w_curve * s_cu,
            "pc2": cfg.w_pca2 * s_pc2,
        })

    diag = pd.DataFrame({
        "slope_z": [s_sl], "curv_z": [s_cu], "pc2_z": [s_pc2]
    }, index=[t])

    return {"yc": yc, "features": feats, "signal": snap, "diag": diag}

# ----------------------------- Map to trades (DV01-neutral) -----------------------------

def dv01_neutral_steepener(
    *,
    yc_snap: pd.Series,                # yields at tenors (last row of yc)
    target_gross_dv01_usd: float,     # e.g., 10_000  (≈ $10k / 1bp total)
    map_cfg: MapConfig = MapConfig(),
    cap_leg: float = 0.6,
) -> pd.Series:
    """
    Build a classic 2s10s steepener (long long-end DV01, short short-end DV01) DV01-neutral.
    Returns Series[symbol] of target *DV01* by leg (signed); convert to contracts separately.
    """
    # Choose closest symbols to 2y and 10y
    sym_short = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-2.0))  # ~2y
    sym_long  = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-10.0)) # ~10y

    dv = pd.Series(map_cfg.dv01_usd).astype(float)
    dv = dv[[sym_short, sym_long]]

    # DV01-neutral: w_long = +x, w_short = -x
    w_long = min(cap_leg, 1.0) * (target_gross_dv01_usd / 2.0)
    w_short = -w_long

    return pd.Series({sym_long: w_long, sym_short: w_short}, dtype=float)

def dv01_neutral_butterfly(
    *,
    target_gross_dv01_usd: float,
    map_cfg: MapConfig = MapConfig(),
    body_tenor: float = 5.0, wings: Tuple[float,float] = (2.0, 10.0),
    ratio_wing: float = 0.5,  # simple 1:-2:1 DV01 mix
) -> pd.Series:
    """
    Simple body-vs-wings fly (long body vs short wings if curvature>0, and vice versa).
    Returns DV01 targets per symbol (signed).
    """
    sym_l = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-wings[0]))
    sym_b = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-body_tenor))
    sym_r = min(map_cfg.tenor_map, key=lambda k: abs(map_cfg.tenor_map[k]-wings[1]))

    total = target_gross_dv01_usd
    dv01_leg = total / (2*ratio_wing + 2)  # allocate across 1:-2:1 roughly
    # Wings share
    w_wing = ratio_wing * dv01_leg
    # Signed as long body, short wings (flip sign if needed upstream)
    return pd.Series({sym_l: -w_wing, sym_b: +2*dv01_leg, sym_r: -w_wing}, dtype=float)

def dv01_to_contracts(
    *,
    dv01_targets: pd.Series,        # $ DV01 per symbol (signed)
    dv01_per_contract: Dict[str,float],  # $ DV01 per contract
) -> pd.Series:
    """Convert DV01 targets into whole contracts."""
    dv = pd.Series(dv01_per_contract).reindex(dv01_targets.index).astype(float)
    contracts = (dv01_targets / dv).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return contracts.round()

# ----------------------------- End-to-end snapshot helper -----------------------------

def build_yield_curve_trades(
    *,
    yields: pd.DataFrame,                 # date x maturityYears (floats), decimal
    mode: SignalMode = "combo",
    cfg: CurveConfig = CurveConfig(),
    map_cfg: MapConfig = MapConfig(),
    target_gross_dv01_usd: float = 10_000.0,
) -> Dict[str, object]:
    """
    One call: compute curve signal → propose DV01-neutral trades.
    Returns:
      {
        'yc': curve panel,
        'signal': Series,
        'steepener_dv01': Series (per symbol),
        'butterfly_dv01': Series (per symbol),
        'contracts_steepener': Series,
        'contracts_butterfly': Series,
        'diag': DataFrame (z-scores),
      }
    """
    out = build_signal(yields=yields, mode=mode, cfg=cfg)
    if out["signal"].empty or out["yc"].empty: # type: ignore
        return {**out, "steepener_dv01": pd.Series(dtype=float), "butterfly_dv01": pd.Series(dtype=float),
                "contracts_steepener": pd.Series(dtype=float), "contracts_butterfly": pd.Series(dtype=float)}

    # Direction: slope_z > 0 → prefer steepener; curvature_z > 0 → long belly fly
    z = out["diag"].iloc[-1] # type: ignore
    sign_steep = np.sign(float(z.get("slope_z", 0.0)))
    sign_fly   = np.sign(float(z.get("curv_z", 0.0)))

    st = dv01_neutral_steepener(
        yc_snap=out["yc"].iloc[-1], # type: ignore
        target_gross_dv01_usd=abs(target_gross_dv01_usd) * (1 if sign_steep >= 0 else -1),
        map_cfg=map_cfg,
        cap_leg=cfg.cap_leg,
    )
    bf = dv01_neutral_butterfly(
        target_gross_dv01_usd=abs(target_gross_dv01_usd) * (1 if sign_fly >= 0 else -1),
        map_cfg=map_cfg,
        body_tenor=5.0, wings=(2.0,10.0),
    )

    c_st = dv01_to_contracts(dv01_targets=st, dv01_per_contract=map_cfg.dv01_usd)
    c_bf = dv01_to_contracts(dv01_targets=bf, dv01_per_contract=map_cfg.dv01_usd)

    return {**out,
            "steepener_dv01": st, "butterfly_dv01": bf,
            "contracts_steepener": c_st, "contracts_butterfly": c_bf}

# ----------------------------- Example -----------------------------

if __name__ == "__main__":
    # Synthetic curve demo
    idx = pd.date_range("2023-01-02", periods=400, freq="B")
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        0.5: 0.035 + 0.001*np.sin(np.linspace(0,8,len(idx))) + 0.004*rng.standard_normal(len(idx)),
        2.0: 0.037 + 0.001*np.cos(np.linspace(0,7,len(idx))) + 0.004*rng.standard_normal(len(idx)),
        5.0: 0.038 + 0.001*np.sin(np.linspace(0,6,len(idx))) + 0.004*rng.standard_normal(len(idx)),
        10.0:0.040 + 0.001*np.cos(np.linspace(0,5,len(idx))) + 0.004*rng.standard_normal(len(idx)),
        30.0:0.041 + 0.001*np.sin(np.linspace(0,4,len(idx))) + 0.004*rng.standard_normal(len(idx)),
    }, index=idx).abs()

    out = build_yield_curve_trades(yields=base, mode="combo", target_gross_dv01_usd=10_000.0)
    print("Signal snapshot:\n", out["signal"])
    print("\nContracts (steepener):\n", out["contracts_steepener"])
    print("\nContracts (butterfly):\n", out["contracts_butterfly"])