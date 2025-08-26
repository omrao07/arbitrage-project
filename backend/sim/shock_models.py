# backend/risk/shock_models.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Reuse your PolicyShock types
from backend.risk.policy_sim import PolicyShock, RateShock, EquityShock, FXShock, VolShock # type: ignore


# ============================ Utilities ============================

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _norm_cdf_inv(p: float) -> float:
    """
    Approx inverse normal CDF (Acklam). Good enough for VaR quantiles.
    """
    # Coefficients for central region
    a = (-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    # Coefficients for tails
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00)
    p = _clamp(p, 1e-12, 1-1e-12)
    # Break-points
    pl, ph = 0.02425, 1-0.02425
    if p < pl:
        q = math.sqrt(-2*math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        x = num/den
    elif p > ph:
        q = math.sqrt(-2*math.log(1-p))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        x = num/den
    else:
        q = p - 0.5
        r = q*q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        x = num/den
    return x


# ============================ Rate shock builders ============================

def make_parallel_rate_shock(bp: float, *, note: Optional[str] = None) -> PolicyShock:
    return PolicyShock(
        name=f"Parallel {bp:+.0f}bp",
        rates=RateShock(parallel_bp=float(bp)),
        notes=[note] if note else []
    )

def make_steepen_twist(parallel_bp: float = 0.0, steepen_bp: float = 0.0, pivot_yrs: float = 5.0,
                       butterfly_bp: float = 0.0, *, note: Optional[str] = None) -> PolicyShock:
    return PolicyShock(
        name=f"Twist p={parallel_bp:+.0f}bp s={steepen_bp:+.0f}bp bfly={butterfly_bp:+.0f}bp",
        rates=RateShock(parallel_bp=float(parallel_bp),
                        steepen_bp=float(steepen_bp),
                        twist_pivot_yrs=float(pivot_yrs),
                        butterfly_bp=float(butterfly_bp)),
        notes=[note] if note else []
    )

def make_keyrate_bumps(bp_by_tenor: Dict[float, float], *, label: str = "Key-rate bump") -> PolicyShock:
    return PolicyShock(
        name=label,
        rates=RateShock(custom_tenor_bp={float(k): float(v) for k, v in bp_by_tenor.items()}),
        notes=[f"Custom tenor bumps: {bp_by_tenor}"]
    )


# ============================ Equity shock builders ============================

def make_equity_index_shock(index_return_pct: float,
                            betas: Dict[str, float] | None = None,
                            overrides: Dict[str, float] | None = None,
                            *, label: str = "Index beta shock") -> PolicyShock:
    """
    Apply index move times per-symbol beta. 'overrides' lets you pin specific symbols.
    """
    pct_map: Dict[str, float] = {}
    for sym, b in (betas or {}).items():
        pct_map[sym] = float(index_return_pct) * float(b)
    if overrides:
        pct_map.update({k: float(v) for k, v in overrides.items()})
    return PolicyShock(
        name=f"{label}: {index_return_pct:+.2f}%",
        equities=EquityShock(pct_by_symbol=pct_map, default_pct=0.0),
        notes=[f"Index move {index_return_pct:+.2f}% with betas"]
    )

def make_uniform_equity_shock(pct: float, *, label: str = "Uniform equity shock") -> PolicyShock:
    return PolicyShock(
        name=f"{label} {pct:+.2f}%",
        equities=EquityShock(default_pct=float(pct)),
        notes=[f"All symbols {pct:+.2f}%"]
    )


# ============================ FX shock builders ============================

def make_fx_shock(pair_pct: Dict[str, float], *, label: str = "FX shock") -> PolicyShock:
    return PolicyShock(
        name=label,
        fx=FXShock(pct_by_pair={k.upper(): float(v) for k, v in pair_pct.items()}),
        notes=[f"Pairs: {pair_pct}"]
    )

def make_devaluation(pair: str, pct: float, *, label: Optional[str] = None) -> PolicyShock:
    """
    +pct means the quoted pair rises. For USDINR +5% => INR weakens ~5%.
    """
    return make_fx_shock({pair.upper(): pct}, label=label or f"{pair.upper()} {pct:+.2f}%")


# ============================ Vol shock builders ============================

def make_vol_spike(vol_pts_by_symbol: Dict[str, float], *, label: str = "Vol spike") -> PolicyShock:
    return PolicyShock(
        name=label,
        vol=VolShock(vol_pts_by_symbol={k: float(v) for k, v in vol_pts_by_symbol.items()}),
        notes=[f"Abs. vol points: {vol_pts_by_symbol}"]
    )


# ============================ Combined & grid builders ============================

def combine(*shocks: PolicyShock, name: Optional[str] = None) -> PolicyShock:
    """
    Merge multiple shocks. Later shocks override earlier keys for same symbol/pair.
    Notes are concatenated.
    """
    rates = RateShock()
    eq = EquityShock()
    fx = FXShock()
    vol = VolShock()
    notes: List[str] = []
    nm = name or " + ".join(s.name for s in shocks if s)

    for s in shocks:
        if not s:
            continue
        notes.extend(s.notes)

        # rates
        if s.rates:
            r = s.rates
            rates.parallel_bp += r.parallel_bp
            rates.steepen_bp += r.steepen_bp
            rates.butterfly_bp += r.butterfly_bp
            rates.twist_pivot_yrs = r.twist_pivot_yrs or rates.twist_pivot_yrs
            rates.custom_tenor_bp.update(r.custom_tenor_bp or {})

        # equities
        if s.equities:
            e = s.equities
            # default additively combine (can change to override if you prefer)
            eq.default_pct += e.default_pct
            eq.pct_by_symbol.update(e.pct_by_symbol or {})

        # fx
        if s.fx:
            fx.pct_by_pair.update(s.fx.pct_by_pair or {})

        # vol
        if s.vol:
            v = s.vol
            for k, pts in (v.vol_pts_by_symbol or {}).items():
                vol.vol_pts_by_symbol[k] = vol.vol_pts_by_symbol.get(k, 0.0) + float(pts)

    return PolicyShock(name=nm, rates=rates, equities=eq, fx=fx, vol=vol, notes=notes)

def shock_grid(rate_bps: Sequence[float], eq_pct: Sequence[float], usd_inr_pct: Sequence[float],
               *, base_name: str = "Grid") -> Dict[str, PolicyShock]:
    """
    Cartesian grid of simple combined shocks for batch stress/backtest.
    """
    out: Dict[str, PolicyShock] = {}
    for rb in rate_bps:
        for ep in eq_pct:
            for fp in usd_inr_pct:
                key = f"{base_name}_r{rb:+.0f}bp_e{ep:+.1f}_fx{fp:+.1f}"
                out[key] = combine(
                    make_parallel_rate_shock(rb),
                    make_uniform_equity_shock(ep),
                    make_devaluation("USDINR", fp),
                    name=key,
                )
    return out


# ============================ Covariance / VaR-driven shocks ============================

def var_shock_from_cov(symbols: List[str],
                       cov: List[List[float]],
                       *,
                       quantile: float = 0.99,
                       seed: Optional[int] = None,
                       label: str = "VaR shock") -> PolicyShock:
    """
    Generate a single equity shock vector from a covariance matrix at a given quantile.
    Uses NumPy if available; otherwise falls back to a crude eigen-ish sampling.

    Returns a PolicyShock with pct_by_symbol in **percent** (e.g., -3.2).
    """
    try:
        import numpy as np  # type: ignore
        rng = np.random.default_rng(seed)
        C = np.array(cov, dtype=float)
        # Cholesky (regularize if needed)
        eps = 1e-8
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(C)
            w = np.clip(w, eps, None)
            L = V @ np.diag(np.sqrt(w))
        z = rng.standard_normal(len(symbols))
        # scale to target quantile radius
        # radius via inverse CDF (1D); for multivariate a scalar scale is a heuristic
        scale = abs(_norm_cdf_inv(1 - (1-quantile)/2))
        r = (L @ z) * scale
        pct = {sym: float(100.0 * r[i]) for i, sym in enumerate(symbols)}
        return make_equity_index_shock(0.0, betas=None, overrides=pct, label=label)
    except Exception:
        # Fallback: diagonal vol only + random signs
        if seed is not None:
            random.seed(seed)
        vols = [math.sqrt(max(cov[i][i], 0.0)) for i in range(len(symbols))]
        scale = abs(_norm_cdf_inv(quantile))
        pct = {}
        for sym, v in zip(symbols, vols):
            sgn = -1.0 if random.random() < 0.5 else 1.0
            pct[sym] = 100.0 * sgn * v * scale
        return make_equity_index_shock(0.0, overrides=pct, label=f"{label} (diag)")


# ============================ Jump / tail scenarios ============================

def jump_shock_equities(symbols: Iterable[str],
                        mean_drop_pct: float = -8.0,
                        dispersion_pct: float = 3.0,
                        *, seed: Optional[int] = None,
                        label: str = "Equity jump") -> PolicyShock:
    """
    Assigns a heavy one-day drawdown around mean_drop_pct with uniform dispersion.
    """
    if seed is not None:
        random.seed(seed)
    pct_map: Dict[str, float] = {}
    for s in symbols:
        d = mean_drop_pct + (random.random() - 0.5) * 2 * dispersion_pct
        pct_map[s] = d
    return make_equity_index_shock(0.0, overrides=pct_map, label=label)

def jump_shock_rates(parallel_bp: float = 75.0, steepen_bp: float = 25.0,
                     *, label: str = "Rate jump") -> PolicyShock:
    return make_steepen_twist(parallel_bp=parallel_bp, steepen_bp=steepen_bp, label=label) # type: ignore

def combined_jump(symbols: Iterable[str], *, eq_mean=-7.0, eq_disp=3.0,
                  rate_bp=60.0, steepen=15.0, usd_inr_pct=1.0) -> PolicyShock:
    return combine(
        jump_shock_equities(symbols, mean_drop_pct=eq_mean, dispersion_pct=eq_disp, label="Equity jump"),
        jump_shock_rates(parallel_bp=rate_bp, steepen_bp=steepen, label="Rates jump"),
        make_devaluation("USDINR", usd_inr_pct, label="FX jump"),
        name="Combined jump"
    )


# ============================ Beta helper ============================

def estimate_betas(returns_by_symbol: Dict[str, List[float]],
                   index_returns: List[float]) -> Dict[str, float]:
    """
    OLS beta estimate for each symbol vs. index: cov(x, m) / var(m).
    Inputs are simple returns (e.g., 0.01 for +1%).
    """
    def _beta(xs: List[float], m: List[float]) -> float:
        if not xs or not m or len(xs) != len(m):
            return 1.0
        mx = sum(xs)/len(xs)
        mm = sum(m)/len(m)
        cov = sum((xi-mx)*(mi-mm) for xi, mi in zip(xs, m)) / max(len(xs)-1, 1)
        var = sum((mi-mm)**2 for mi in m) / max(len(m)-1, 1)
        return 0.0 if var == 0 else cov/var
    return {sym: _beta(ret, index_returns) for sym, ret in returns_by_symbol.items()}


# ============================ Examples (optional CLI) ============================

if __name__ == "__main__":
    # Tiny smoke test
    shocks = [
        make_parallel_rate_shock(50),
        make_steepen_twist(parallel_bp=0, steepen_bp=20, pivot_yrs=7),
        make_equity_index_shock(-3.0, betas={"AAPL": 1.2, "MSFT": 1.0, "TSLA": 2.0}),
        make_devaluation("USDINR", 1.5),
        combine(
            make_parallel_rate_shock(-25),
            make_uniform_equity_shock(+1.2),
            make_vol_spike({"SPX": +8.0}),
            name="Dovish risk-on"
        ),
    ]
    for s in shocks:
        print(s.name, s.notes)