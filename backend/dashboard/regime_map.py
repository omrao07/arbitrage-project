# backend/analytics/regime_map.py
"""
Regime Map: detect and track market regimes, then project them
onto a 2x2 macro map (Growth ↑/↓  ×  Inflation ↑/↓) with a risk-on/off overlay.

Design goals
------------
- Pure Python with tiny optional deps (numpy/pandas/plotly if present).
- Works with partial inputs: missing fields default to neutral (0).
- Two modes:
    1) Rules (stable, transparent) → recommended default.
    2) Learner (optional): unsupervised k-means on standardized features.
- Emits current regime, confidence, and a tidy map payload for dashboards.

Inputs (all optional; raw levels/returns; we standardize internally):
- growth_proxies:  real activity (PMI, payrolls surprise, EPS breadth, industrials vs defensives)
- inflation_proxies: breakevens, WTI/Brent returns, commodities basket, CPI surprise
- risk_proxies: credit spreads (IG/HY), vol (VIX/RVX/realized), breadth, EM FX, crypto
- policy_proxy: rate changes or policy shock score (optional)
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional deps
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None  # type: ignore


# =============================================================================
# Data models
# =============================================================================

@dataclass
class RegimeInputs:
    # Put raw numbers here; we'll z-normalize online.
    # Growth (+ = stronger): PMI z, EPS breadth z, payrolls surprise z, cyc/def ratio z
    growth_pmi_z: Optional[float] = None
    growth_eps_breadth_z: Optional[float] = None
    growth_macro_surprise_z: Optional[float] = None
    growth_cyc_def_z: Optional[float] = None

    # Inflation (+ = higher): breakevens z, oil/commods return z, CPI surprise z
    infl_bkeven_z: Optional[float] = None
    infl_oil_ret_z: Optional[float] = None
    infl_cpi_surprise_z: Optional[float] = None

    # Risk-on (+ = risk-on): breadth z, EMFX z, crypto z; Risk-off (+) via vol/spreads (we invert)
    risk_breadth_z: Optional[float] = None
    risk_emfx_z: Optional[float] = None
    risk_crypto_z: Optional[float] = None
    risk_vol_z: Optional[float] = None           # higher vol → risk-off (we invert)
    risk_credit_spread_z: Optional[float] = None # wider spreads → risk-off (we invert)

    # Policy (optional): + = easing / supportive; − = tightening / hawkish shock
    policy_stance_z: Optional[float] = None


@dataclass
class RegimeReading:
    # Primary outputs
    growth_axis: float      # [-1, +1]
    inflation_axis: float   # [-1, +1]
    risk_axis: float        # [-1, +1]
    quadrant: str           # 'Goldilocks', 'Overheat', 'Stagflation', 'Disinflation'
    risk_label: str         # 'Risk-On' | 'Risk-Off' | 'Neutral'
    confidence: float       # [0..1], higher = stronger signal

    # Optional extras
    method: str             # 'rules' | 'learner'
    components: Dict[str, float]
    meta: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# =============================================================================
# Core engine
# =============================================================================

class RegimeMap:
    """
    Streaming regime detector with online standardization.
    """

    def __init__(
        self,
        *,
        rules_weights: Optional[Dict[str, float]] = None,
        hysteresis: float = 0.1,      # buffer for label switching
        history_len: int = 4096,
        learner_k: int = 4,           # for optional k-means
    ):
        # weights for transparent rules
        self.w = {
            # growth block
            "growth_pmi_z": 0.30,
            "growth_eps_breadth_z": 0.25,
            "growth_macro_surprise_z": 0.25,
            "growth_cyc_def_z": 0.20,
            # inflation block
            "infl_bkeven_z": 0.45,
            "infl_oil_ret_z": 0.35,
            "infl_cpi_surprise_z": 0.20,
            # risk block (risk-on positives; vol/credit inverted below)
            "risk_breadth_z": 0.30,
            "risk_emfx_z": 0.25,
            "risk_crypto_z": 0.15,
            "risk_vol_z": 0.15,
            "risk_credit_spread_z": 0.15,
            # policy
            "policy_stance_z": 0.10,
        }
        if rules_weights:
            self.w.update(rules_weights)

        self.hyst = float(hysteresis)
        self.hist_len = int(history_len)
        self.k = int(learner_k)

        # state
        self._hist_axes: List[Tuple[float, float, float]] = []  # (growth, infl, risk)
        self._hist_quadrant: List[str] = []
        self._last_risk_label = "Neutral"

        # rolling means/stds per key (for users who pass raw non-z data later)
        self._roll: Dict[str, List[float]] = {}

        # learner state
        self._centroids = None  # type: ignore

    # ----------------- Public API -----------------

    def update(self, inp: RegimeInputs, *, method: str = "rules") -> RegimeReading:
        comps = self._components(inp)

        # Map components → axes
        growth = _clip(_blend(comps, ["growth_pmi_z","growth_eps_breadth_z","growth_macro_surprise_z","growth_cyc_def_z"], self.w), -1.0, 1.0)
        infl   = _clip(_blend(comps, ["infl_bkeven_z","infl_oil_ret_z","infl_cpi_surprise_z"], self.w), -1.0, 1.0)

        # Risk: breadth/emfx/crypto positive, vol/spreads negative
        rpos = _blend(comps, ["risk_breadth_z","risk_emfx_z","risk_crypto_z"], self.w)
        rneg = _blend(comps, ["risk_vol_z","risk_credit_spread_z"], self.w)
        risk  = _clip(rpos - rneg, -1.0, 1.0)

        # Policy tilt (supportive policy nudges risk and growth a touch)
        pol = _safe(comps.get("policy_stance_z"), 0.0)
        growth = _clip(growth + 0.10 * pol, -1.0, 1.0)
        risk   = _clip(risk + 0.15 * pol, -1.0, 1.0)

        if method == "learner":
            growth, infl, risk = self._learner_adjust((growth, infl, risk))

        quad = self._quadrant(growth, infl)
        risk_label = self._risk_label(risk)

        # confidence: distance from origin in axes space (0..sqrt(3))
        conf = min(1.0, math.sqrt(max(0.0, growth**2 + infl**2 + risk**2)) / math.sqrt(3))

        # persist
        self._push_hist(growth, infl, risk, quad)

        return RegimeReading(
            growth_axis=growth,
            inflation_axis=infl,
            risk_axis=risk,
            quadrant=quad,
            risk_label=risk_label,
            confidence=conf,
            method=method,
            components=comps,
            meta={"weights": self.w, "hysteresis": self.hyst, "len_hist": len(self._hist_axes)},
        )

    def history_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas not installed")
        rows = [{"growth": g, "infl": i, "risk": r, "quadrant": q} for (g, i, r), q in zip(self._hist_axes, self._hist_quadrant)]
        return pd.DataFrame(rows)

    def figure(self, reading: Optional[RegimeReading] = None):
        if go is None:
            raise RuntimeError("plotly not installed")
        if reading is None and self._hist_axes:
            g, i, r = self._hist_axes[-1]
        elif reading is not None:
            g, i, r = reading.growth_axis, reading.inflation_axis, reading.risk_axis
        else:
            g = i = r = 0.0

        # Draw 2x2 quadrants with a marker sized by |risk|
        fig = go.Figure()

        # quadrants
        fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor="rgba(46,204,113,0.12)", line=dict(color="rgba(0,0,0,0)"))     # Goldilocks
        fig.add_shape(type="rect", x0=0, y0=-1, x1=1, y1=0, fillcolor="rgba(241,196,15,0.12)", line=dict(color="rgba(0,0,0,0)"))    # Overheat
        fig.add_shape(type="rect", x0=-1, y0=-1, x1=0, y1=0, fillcolor="rgba(231,76,60,0.12)", line=dict(color="rgba(0,0,0,0)"))    # Stagflation
        fig.add_shape(type="rect", x0=-1, y0=0, x1=0, y1=1, fillcolor="rgba(52,152,219,0.12)", line=dict(color="rgba(0,0,0,0)"))    # Disinflation

        # axes
        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="#666", width=1))
        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="#666", width=1))

        # marker
        size = 16 + 24 * abs(r)
        fig.add_trace(go.Scatter(x=[g], y=[i], mode="markers+text",
                                 marker=dict(size=size, color="#34495e"),
                                 text=[f"risk {('ON' if r>0.1 else ('OFF' if r<-0.1 else 'NEU'))}"],
                                 textposition="bottom center",
                                 hovertemplate=f"Growth={g:+.2f}<br>Inflation={i:+.2f}<br>Risk={r:+.2f}<extra></extra>"))

        fig.update_layout(
            title="Regime Map (Growth vs Inflation, dot size = |Risk|)",
            xaxis=dict(range=[-1,1], zeroline=False, title="Growth"),
            yaxis=dict(range=[-1,1], zeroline=False, title="Inflation"),
            width=650, height=520, showlegend=False, margin=dict(l=20,r=20,t=50,b=40)
        )
        return fig


    # ----------------- Internals -----------------

    def _components(self, inp: RegimeInputs) -> Dict[str, float]:
        # Values are assumed to be z-scores already; if you pass raw levels,
        # call self.standardize(key, x) for each instead.
        d = {k: _clip(_safe(getattr(inp, k), 0.0), -4.0, 4.0) for k in inp.__dataclass_fields__.keys()}
        # invert risk-off metrics so higher value = more risk-off ➜ subtract later
        d["risk_vol_z"] = max(0.0, d.get("risk_vol_z", 0.0))
        d["risk_credit_spread_z"] = max(0.0, d.get("risk_credit_spread_z", 0.0))
        return d

    def _quadrant(self, g: float, i: float) -> str:
        # Boundaries with small deadband to avoid flapping
        eps = 0.05
        if g >= +eps and i <= -eps:
            return "Goldilocks"    # Growth↑, Inflation↓
        if g >= +eps and i >= +eps:
            return "Overheat"      # Growth↑, Inflation↑
        if g <= -eps and i >= +eps:
            return "Stagflation"   # Growth↓, Inflation↑
        if g <= -eps and i <= -eps:
            return "Disinflation"  # Growth↓, Inflation↓
        # Near axes: choose closest
        if abs(g) >= abs(i):
            return "Goldilocks" if g >= 0 else "Disinflation"
        return "Overheat" if i >= 0 else "Stagflation"

    def _risk_label(self, r: float) -> str:
        # hysteresis bands
        H = self.hyst
        last = self._last_risk_label
        if last == "Risk-On":
            if r < +0.10 - H: last = "Neutral"
            if r < -0.20 - H: last = "Risk-Off"
        elif last == "Risk-Off":
            if r > -0.10 + H: last = "Neutral"
            if r > +0.20 + H: last = "Risk-On"
        else:
            if r >= +0.20 + H: last = "Risk-On"
            elif r <= -0.20 - H: last = "Risk-Off"
        self._last_risk_label = last
        return last

    def _push_hist(self, g: float, i: float, r: float, q: str) -> None:
        self._hist_axes.append((g, i, r))
        self._hist_quadrant.append(q)
        if len(self._hist_axes) > self.hist_len:
            cut = len(self._hist_axes) - self.hist_len
            del self._hist_axes[:cut]; del self._hist_quadrant[:cut]

    # --------------- Optional learner (k-means) ----------------

    def _learner_adjust(self, vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if np is None or len(self._hist_axes) < max(60, 5 * self.k):
            return vec  # not enough data or numpy missing
        X = np.array(self._hist_axes[-min(1000, len(self._hist_axes)):], dtype=float)
        # lazily fit/update centroids
        if self._centroids is None:
            self._centroids = _kmeans(X, k=self.k, n_iter=50, seed=42)[0]
        # soft pull towards nearest centroid (stabilize)
        c = self._centroids
        d = np.linalg.norm(c - np.array(vec), axis=1)
        j = int(np.argmin(d))
        w = 0.20  # pull strength
        adj = tuple(float((1 - w) * v + w * c[j, t]) for t, v in enumerate(vec))
        return adj # type: ignore

    # --------------- Standardization helper (if you want to pass raw) --------

    def standardize(self, key: str, x: float) -> float:
        """Online z-score for any metric."""
        hist = self._roll.setdefault(key, [])
        hist.append(float(x))
        if len(hist) > 5000:
            del hist[: len(hist) - 5000]
        mu = statistics.fmean(hist[:-1]) if len(hist) > 1 else float(x)
        try:
            sd = statistics.pstdev(hist[:-1]) if len(hist) > 1 else 1.0
        except Exception:
            sd = 1.0
        sd = max(sd, 1e-9)
        return _clip((float(x) - mu) / sd, -4.0, 4.0)


# =============================================================================
# Small utils
# =============================================================================

def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _blend(comps: Dict[str, float], keys: Iterable[str], weights: Dict[str, float]) -> float:
    num = den = 0.0
    for k in keys:
        if k in comps:
            w = float(weights.get(k, 1.0))
            num += w * comps[k]
            den += abs(w)
    return 0.0 if den == 0.0 else num / den

def _kmeans(X, k=4, n_iter=50, seed=42):
    rng = np.random.default_rng(seed) # type: ignore
    idx = rng.choice(X.shape[0], size=k, replace=False)
    C = X[idx].copy()
    for _ in range(n_iter):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = D.argmin(axis=1)
        C_new = np.vstack([X[lab == j].mean(axis=0) if (lab == j).any() else C[j] for j in range(k)]) # type: ignore
        if np.allclose(C_new, C): # type: ignore
            break
        C = C_new
    return C, lab


# =============================================================================
# Tiny demo
# =============================================================================

if __name__ == "__main__":
    rm = RegimeMap(hysteresis=0.08)

    # Simulate 100 steps cycling through quadrants
    import random
    for t in range(100):
        phase = math.sin(t / 12.0)
        inp = RegimeInputs(
            growth_pmi_z= 0.8 * phase + 0.2 * random.uniform(-1,1),
            growth_eps_breadth_z = 0.6 * phase + 0.1 * random.uniform(-1,1),
            infl_bkeven_z = 0.7 * (math.cos(t/11.0)) + 0.1 * random.uniform(-1,1),
            infl_oil_ret_z = 0.5 * (math.cos(t/11.0)) + 0.2 * random.uniform(-1,1),
            risk_breadth_z = 0.7 * phase + 0.2 * random.uniform(-1,1),
            risk_vol_z = 0.5 * (-phase) + 0.2 * random.uniform(-1,1),      # higher in risk-off
            risk_credit_spread_z = 0.5 * (-phase) + 0.2 * random.uniform(-1,1),
            policy_stance_z = 0.2 * phase,
        )
        reading = rm.update(inp, method="rules")
        if t % 20 == 0:
            print(f"t={t:02d} | {reading.quadrant:12s} | {reading.risk_label:8s} | g={reading.growth_axis:+.2f} i={reading.inflation_axis:+.2f} r={reading.risk_axis:+.2f}")

    if go:
        fig = rm.figure()
        # fig.show()  # enable for manual run