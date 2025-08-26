# backend/macro/economy.py
"""
Economy Gauge & Nowcaster (stdlib-only)

What this gives you
-------------------
- EconomyGauge: one-shot aggregator that ingests your flat {key: float} signals and
  returns a normalized economy snapshot:
    * growth_z, inflation_z, liquidity_z, risk_z
    * gi_quadrant (Q1..Q4), regime label
    * policy map per CB (hawkish/dovish, restrictive/loose)
    * composite 'heat' score and handy derived proxies

- OnlineNowcaster: lightweight online state with rolling mean/std to compute z-scores
  across runs (no heavy TS deps). Persist/restore to JSON.

- Feature engineering helpers: diffusion index, winsorized z, balanced composites.

Inputs (examples)
-----------------
- From real feeds/configs:
    fed.ff_upper, ECB.main_refi, RBI.repo
    FED.assets_total, ECB.assets_total, RBI.fx_reserves
    IN.cpi_yoy, EU.cpi_yoy, US.cpi_yoy (or 'infl_nowcast' from simulator)
    risk_z, liquidity_z (from sim or constructed)

- From market proxies (optional if present):
    DXY, GLD, CL, TLT, NDX, EEM ...
    term spreads like "US.y_10y - US.y_2y" if you pass them as signals.

Outputs
-------
A dict with 'eco.'-prefixed keys plus structured fields for agents:
    {
      "eco.growth_z": 0.35, "eco.inflation_z": -0.2, "eco.liquidity_z": 0.4,
      "eco.risk_z": 0.1, "eco.heat": 0.27,
      "eco.gi_quadrant": "Q1", "eco.regime": "Goldilocks",
      "eco.policy.FED.stance": "restrictive-hawkish", ...
    }

Typical usage
-------------
from backend.macro.economy import EconomyGauge, OnlineNowcaster

gauge = EconomyGauge()
eco = gauge.from_signals(signals)           # -> dict snapshot

# If you want rolling z-scores across steps:
state = OnlineNowcaster.load("runs/eco_state.json")
eco = gauge.from_signals(signals, online=state)
state.save("runs/eco_state.json")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any


Number = float
SignalMap = Dict[str, Number]


# --------------------------- tiny stats ---------------------------------

@dataclass
class OnlineStats:
    n: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        var = self._m2 / (self.n - 1)
        return math.sqrt(max(0.0, var))


@dataclass
class OnlineNowcaster:
    """
    Keeps rolling mean/std per feature key for z-scoring across runs.
    """
    stats: Dict[str, OnlineStats] = field(default_factory=dict)

    def z(self, key: str, x: Optional[float], *, eps: float = 1e-6, update: bool = True) -> Optional[float]:
        if x is None:
            return None
        st = self.stats.get(key)
        if st is None:
            st = OnlineStats(); self.stats[key] = st
        if update:
            st.update(float(x))
        sd = st.std or eps
        return (float(x) - st.mean) / (sd if sd > 0 else eps)

    # persistence
    def save(self, path: str) -> None:
        js = {k: {"n": s.n, "mean": s.mean, "m2": s._m2} for k, s in self.stats.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(js, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OnlineNowcaster":
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            stats = {}
            for k, v in (blob or {}).items():
                s = OnlineStats(n=int(v.get("n", 0)), mean=float(v.get("mean", 0.0))); s._m2 = float(v.get("m2", 0.0))
                stats[k] = s
            return cls(stats)
        except Exception:
            return cls()


# --------------------------- helpers ------------------------------------

def _get(sig: SignalMap, key: str, default: Optional[float] = None) -> Optional[float]:
    v = sig.get(key, default)
    if v is None: return None
    try:
        return float(v)
    except Exception:
        return default

def _winsor_z(x: Optional[float], mean: float = 0.0, std: float = 1.0, clip: Tuple[float, float] = (-3.0, 3.0)) -> Optional[float]:
    if x is None or std <= 0: return None
    z = (x - mean) / std
    lo, hi = clip
    return max(lo, min(hi, z))

def _safe_avg(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None

def _label_regime(growth_z: float, infl_z: float) -> Tuple[str, str]:
    """
    GI quadrant and friendly label:
      Q1: growth+, infl-  -> Goldilocks
      Q2: growth+, infl+  -> Expansion (late) / Reflation
      Q3: growth-, infl+  -> Stagflation
      Q4: growth-, infl-  -> Slowdown / Disinflation
    """
    if growth_z >= 0 and infl_z < 0:
        return "Q1", "Goldilocks"
    if growth_z >= 0 and infl_z >= 0:
        return "Q2", "Reflation"
    if growth_z < 0 and infl_z >= 0:
        return "Q3", "Stagflation"
    return "Q4", "Disinflation"

def _stance(rate: Optional[float], neutral: float, infl_z: Optional[float]) -> str:
    """
    Classify policy stance: restrictive/neutral/loose + hawkish/dovish bias.
    - Restrictive if rate - neutral > 50 bps; loose if < -50 bps.
    - Hawkish bias if infl_z > +0.5, dovish if < -0.5
    """
    if rate is None:
        return "unknown"
    spread = rate - neutral
    reg = "neutral"
    if spread > 0.005: reg = "restrictive"
    elif spread < -0.005: reg = "loose"
    bias = "balanced"
    if infl_z is not None:
        if infl_z > 0.5: bias = "hawkish"
        elif infl_z < -0.5: bias = "dovish"
    return f"{reg}-{bias}"


# --------------------------- main gauge ---------------------------------

@dataclass
class EconomyGauge:
    """
    Turns a raw signal map into an economy snapshot usable by agents.
    You can pass an OnlineNowcaster to compute persistent z-scores.
    """
    # Neutral rates (approx; adjust to your view / region models)
    neutral: Dict[str, float] = field(default_factory=lambda: {
        "FED": 0.025, "ECB": 0.020, "RBI": 0.055
    })
    # Weights for composites
    w_growth: Dict[str, float] = field(default_factory=lambda: {
        "NDX.mom": 0.20, "EEM.mom": 0.15, "US.y_10y_minus_2y": -0.25,
        "PMI": 0.25, "exports_z": 0.15
    })
    w_infl: Dict[str, float] = field(default_factory=lambda: {
        "infl_nowcast": 0.40, "CPI.yoy": 0.30, "breakeven_5y": 0.20, "wage_growth": 0.10
    })
    w_liq: Dict[str, float] = field(default_factory=lambda: {
        "liquidity_z": 0.50, "FED.assets_total": 0.25, "ECB.assets_total": 0.15, "RBI.fx_reserves": 0.10
    })
    w_risk: Dict[str, float] = field(default_factory=lambda: {
        "risk_z": 0.70, "iv_z": 0.30
    })

    # Optional key remaps to look up in signals (light normalization)
    key_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "FED.rate": ["FED.ff_upper", "fed.ff_upper", "us.ff_upper"],
        "ECB.rate": ["ECB.main_refi", "ecb.main_refi"],
        "RBI.rate": ["RBI.repo", "rbi.repo"],
        "PMI": ["US.pmi", "EU.pmi", "IN.pmi"],
        "CPI.yoy": ["US.cpi_yoy", "EU.cpi_yoy", "IN.cpi_yoy"],
        "breakeven_5y": ["US.bekeven5y", "US.breakeven5y", "breakeven5y"],
        "wage_growth": ["US.wages_yoy", "wages_yoy"],
        "iv_z": ["vix_z", "ivol_z", "vol_z"],
        "exports_z": ["IN.exports_z", "US.exports_z", "EU.exports_z"],
        "US.y_10y_minus_2y": ["US.y_10y_minus_2y", "US.y_10y-2y", "US.y_spread_10_2"],
        "NDX.mom": ["mom_z_NDX", "NDX.mom_z", "ndx_mom_z"],
        "EEM.mom": ["mom_z_EEM", "EEM.mom_z", "eem_mom_z"],
    })

    def from_signals(self, signals: SignalMap, *, online: Optional[OnlineNowcaster] = None) -> Dict[str, Any]:
        """
        Build an economy snapshot from the current flat signals dict.
        """
        s = signals
        # 1) Lookup helper with fallback list
        def v(key: str) -> Optional[float]:
            # direct
            if key in s: return _get(s, key)
            # mapped
            for alias in self.key_map.get(key, []):
                if alias in s:
                    return _get(s, alias)
            return None

        # 2) Build composites (raw then z or already z from feeds)
        def composite(weights: Dict[str, float], already_z: bool = True, z_prefix: str = "eco.") -> Optional[float]:
            acc = 0.0; wsum = 0.0
            vals: Dict[str, float] = {}
            for k, w in weights.items():
                x = v(k)
                if x is None:
                    continue
                xx = x if already_z else (online.z(f"{z_prefix}{k}", x, update=True) if online else x)
                if xx is None:
                    continue
                vals[k] = xx; acc += w * xx; wsum += abs(w)
            return (acc / max(1e-9, wsum)) if wsum > 0 else None

        growth_z = composite(self.w_growth, already_z=True)
        infl_z   = v("infl_nowcast")
        if infl_z is None:
            infl_z = composite(self.w_infl, already_z=True)
        liq_z    = composite(self.w_liq, already_z=True)
        risk_z   = composite(self.w_risk, already_z=True)
        heat     = _safe_avg([abs(x) for x in [growth_z, infl_z, risk_z] if x is not None])  # activity intensity proxy

        # 3) GI quadrant / regime label
        gi_quad, regime = _label_regime(growth_z or 0.0, infl_z or 0.0)

        # 4) Policy stance map
        fed_rate = v("FED.rate"); ecb_rate = v("ECB.rate"); rbi_rate = v("RBI.rate")
        stance_fed = _stance(fed_rate, self.neutral.get("FED", 0.025), infl_z)
        stance_ecb = _stance(ecb_rate, self.neutral.get("ECB", 0.020), infl_z)
        stance_rbi = _stance(rbi_rate, self.neutral.get("RBI", 0.055), infl_z)

        # 5) Handy derivatives / proxies if inputs exist
        # Yield curve if you passed it; else None
        curve = v("US.y_10y_minus_2y")

        out: Dict[str, Any] = {
            "eco.growth_z": growth_z,
            "eco.inflation_z": infl_z,
            "eco.liquidity_z": liq_z,
            "eco.risk_z": risk_z,
            "eco.heat": heat,
            "eco.gi_quadrant": gi_quad,
            "eco.regime": regime,
            "eco.yc_10y_2y": curve,
            "eco.policy.FED.rate": fed_rate,
            "eco.policy.ECB.rate": ecb_rate,
            "eco.policy.RBI.rate": rbi_rate,
            "eco.policy.FED.stance": stance_fed,
            "eco.policy.ECB.stance": stance_ecb,
            "eco.policy.RBI.stance": stance_rbi,
        }

        # 6) Merge back any None as omissions to keep the map clean
        return {k: v for k, v in out.items() if v is not None}

    # Convenience: diffusion index from a set of boolean/thresholded indicators
    @staticmethod
    def diffusion(signals: SignalMap, keys: List[str], *, thresh: float = 0.0) -> Optional[float]:
        vals = [_get(signals, k) for k in keys]
        xs = [1.0 if (v is not None and v > thresh) else 0.0 for v in vals if v is not None]
        return (sum(xs) / len(xs)) if xs else None