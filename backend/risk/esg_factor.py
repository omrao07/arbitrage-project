# backend/research/esg_factor.py
"""
ESG Factor Engine
-----------------
Compute composite ESG scores, neutralize by sector/region, generate tilts,
exclusions, and risk overrides. Integrates with your news sentiment + bus.

Inputs (any subset; missing fields handled gracefully)
- symbol, sector, region
- esg_env, esg_soc, esg_gov          (0-10 or 0-100 scale; auto-normalized)
- provider_score                      (optional single score)
- carbon_intensity_tco2e_per_musd    (tons CO2e / $m revenue)
- controversies (intensity 0..5) and flags (adult, tobacco, weapons, etc.)
- news_sentiment [-1..1]             (your news pipeline)
- market_cap, free_float             (optional, for scaling)

Outputs
- composite_score [0..1]
- zscores by sector/region, deciles
- tilt [-1..+1]
- exclusions (bool)
- risk_overrides (e.g., max_weight, leverage_mult)
- publish to Redis hash "esg:scores" and stream "esg.events" (optional)

CLI
  python -m backend.research.esg_factor --probe
  python -m backend.research.esg_factor --csv data/esg_sample.csv --publish
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Iterable, Tuple

# Optional deps
try:
    import pandas as pd  # pip install pandas
except Exception:
    pd = None  # type: ignore

# Optional bus (best-effort)
try:
    from backend.bus.streams import publish_stream, hset
except Exception:
    publish_stream = None  # type: ignore
    hset = None  # type: ignore


# ----------------------------- Config -----------------------------

@dataclass
class ESGWeights:
    env: float = 0.4
    soc: float = 0.3
    gov: float = 0.3
    provider: float = 0.0          # if you have a single provider score
    news: float = 0.1              # optional sentiment uplift
    carbon_penalty: float = 0.2    # penalty weight applied after normalization

@dataclass
class ESGPolicy:
    weights: ESGWeights = field(default_factory=ESGWeights)
    sector_neutral: bool = True
    region_neutral: bool = False
    carbon_floor: Optional[float] = None        # e.g., clip carbon intensity to [0, floor]
    carbon_target_pct_reduction: float = 0.30   # target vs universe median for tilt scaling
    controversies_cap: int = 4                   # >= this => exclusion
    exclude_flags: Tuple[str, ...] = ("tobacco","controversial_weapons","thermal_coal")
    tilt_scale: float = 0.6                     # scale raw z to final tilt
    max_abs_tilt: float = 0.75
    min_inclusion_score: float = 0.25           # drop if composite < this
    leverage_floor: float = 0.5                 # floor leverage multiplier for worst names
    leverage_cap: float = 1.25                  # cap for best names


# ----------------------------- Core Types -----------------------------

@dataclass
class ESGRow:
    symbol: str
    sector: str = "UNKNOWN"
    region: str = "GLOBAL"
    esg_env: Optional[float] = None
    esg_soc: Optional[float] = None
    esg_gov: Optional[float] = None
    provider_score: Optional[float] = None
    carbon_intensity_tco2e_per_musd: Optional[float] = None
    controversies: Optional[int] = None
    flags: List[str] = field(default_factory=list)
    news_sentiment: Optional[float] = None
    market_cap: Optional[float] = None
    free_float: Optional[float] = None
    # outputs
    composite: float = 0.0
    z_sector: float = 0.0
    z_region: float = 0.0
    tilt: float = 0.0
    exclude: bool = False
    decile: int = 5
    risk_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------- Utilities -----------------------------

def _minmax01(x: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        f = float(x)
        if math.isfinite(f): return f
    except Exception:
        return None
    return None

def _now_ms() -> int:
    return int(time.time() * 1000)


# ----------------------------- Engine -----------------------------

class ESGEngine:
    def __init__(self, policy: Optional[ESGPolicy] = None):
        self.policy = policy or ESGPolicy()

    # ---- Ingest ----
    def from_records(self, rows: Iterable[Dict[str, Any]]) -> List[ESGRow]:
        out: List[ESGRow] = []
        for r in rows:
            out.append(ESGRow(
                symbol=str(r.get("symbol") or r.get("ticker") or "").upper(),
                sector=str(r.get("sector") or "UNKNOWN"),
                region=str(r.get("region") or "GLOBAL"),
                esg_env=_safe_float(r.get("esg_env")),
                esg_soc=_safe_float(r.get("esg_soc")),
                esg_gov=_safe_float(r.get("esg_gov")),
                provider_score=_safe_float(r.get("provider_score")),
                carbon_intensity_tco2e_per_musd=_safe_float(r.get("carbon_intensity_tco2e_per_musd")),
                controversies=int(r.get("controversies")) if r.get("controversies") is not None else None, # type: ignore
                flags=list(r.get("flags") or []),
                news_sentiment=_safe_float(r.get("news_sentiment")),
                market_cap=_safe_float(r.get("market_cap")),
                free_float=_safe_float(r.get("free_float")),
            ))
        return out

    # ---- Scoring ----
    def score(self, data: List[ESGRow]) -> List[ESGRow]:
        W = self.policy.weights

        # Gather env/soc/gov min/max for auto-normalization if inputs look like 0-100
        def _normalize_component(v: Optional[float]) -> Optional[float]:
            if v is None: return None
            # If looks like 0-100, rescale to 0-1
            if v > 1.0: return _minmax01(v, 0.0, 100.0)
            # If already 0-10, rescale
            if 1.0 < v <= 10.0: return _minmax01(v, 0.0, 10.0)
            # Assume already 0-1
            return max(0.0, min(1.0, v))

        # Carbon normalization: transform intensity to a penalty [0..1] where low intensity → 0 penalty
        carb_vals = [r.carbon_intensity_tco2e_per_musd for r in data if r.carbon_intensity_tco2e_per_musd is not None]
        carb_med = float(pd.Series(carb_vals).median()) if (pd and len(carb_vals)>0) else (sum(carb_vals)/len(carb_vals) if carb_vals else 0.0)
        carb_floor = self.policy.carbon_floor if self.policy.carbon_floor is not None else 0.0
        carb_hi = max(carb_med * 3.0, carb_med + 1e-6) if carb_med > 0 else max(1.0, max(carb_vals) if carb_vals else 1.0)

        for r in data:
            e = _normalize_component(r.esg_env)
            s = _normalize_component(r.esg_soc)
            g = _normalize_component(r.esg_gov)
            p = _normalize_component(r.provider_score) if r.provider_score is not None else None
            ns = r.news_sentiment
            if ns is not None:
                ns = max(-1.0, min(1.0, ns))
                ns = 0.5 + 0.5 * ns  # map [-1..1] -> [0..1]

            # base mix (missing values get replaced by cohort mean later)
            parts = {"env": e, "soc": s, "gov": g, "provider": p, "news": ns}

            # fill NAs with cross-sectional means (computed lazily)
            for k in list(parts.keys()):
                if parts[k] is None:
                    parts[k] = 0.5  # temporary; we'll re-neutralize later

            # raw composite
            comp = (W.env*parts["env"] + W.soc*parts["soc"] + W.gov*parts["gov"] # type: ignore
                    + W.provider*parts["provider"] + W.news*parts["news"]) # type: ignore
            comp = comp / max(1e-9, (W.env + W.soc + W.gov + W.provider + W.news))

            # carbon penalty (higher intensity → bigger penalty)
            if r.carbon_intensity_tco2e_per_musd is not None:
                ci = max(carb_floor, r.carbon_intensity_tco2e_per_musd)
                pen = _minmax01(ci, carb_floor, carb_hi)  # 0 low → 1 very high
                comp = max(0.0, comp - W.carbon_penalty * pen)

            r.composite = float(max(0.0, min(1.0, comp)))

        # Sector/Region z-neutralization
        self._neutralize(data)

        # Deciles and tilts
        self._deciles_and_tilts(data)

        # Exclusions and overrides
        self._apply_exclusions_and_overrides(data)

        return data

    # ---- Neutralization ----
    def _neutralize(self, data: List[ESGRow]) -> None:
        if not data: return

        def zscore(vals: List[float]) -> List[float]:
            m = sum(vals)/len(vals)
            v = sum((x-m)*(x-m) for x in vals)/max(1, len(vals)-1)
            sd = math.sqrt(max(1e-9, v))
            return [(x - m) / sd for x in vals]

        # Sector z
        if self.policy.sector_neutral:
            by_sector: Dict[str, List[int]] = {}
            for i, r in enumerate(data):
                by_sector.setdefault(r.sector, []).append(i)
            for sec, idxs in by_sector.items():
                vals = [data[i].composite for i in idxs]
                zs = zscore(vals) if len(idxs) > 1 else [0.0 for _ in idxs]
                for j, i in enumerate(idxs):
                    data[i].z_sector = float(zs[j])

        # Region z
        if self.policy.region_neutral:
            by_region: Dict[str, List[int]] = {}
            for i, r in enumerate(data):
                by_region.setdefault(r.region, []).append(i)
            for reg, idxs in by_region.items():
                vals = [data[i].composite for i in idxs]
                zs = zscore(vals) if len(idxs) > 1 else [0.0 for _ in idxs]
                for j, i in enumerate(idxs):
                    data[i].z_region = float(zs[j])

    # ---- Deciles & Tilts ----
    def _deciles_and_tilts(self, data: List[ESGRow]) -> None:
        if not data: return
        # base z we’ll use for tilt
        base_z = []
        for r in data:
            z = r.z_sector if self.policy.sector_neutral else 0.0
            z += r.z_region if self.policy.region_neutral else 0.0
            base_z.append(z)

        # rank to deciles
        ranks = sorted(range(len(data)), key=lambda i: base_z[i])
        decile_size = max(1, len(data)//10)
        for rank_idx, i in enumerate(ranks):
            d = min(9, rank_idx // decile_size)
            data[i].decile = int(d)

        # tilt scaling
        for i, r in enumerate(data):
            raw = base_z[i]
            tilt = max(-self.policy.max_abs_tilt, min(self.policy.max_abs_tilt, self.policy.tilt_scale * raw))
            r.tilt = float(tilt)

    # ---- Exclusions & Overrides ----
    def _apply_exclusions_and_overrides(self, data: List[ESGRow]) -> None:
        P = self.policy
        for r in data:
            excl = False
            # controversies
            if r.controversies is not None and r.controversies >= P.controversies_cap:
                excl = True
            # flagged activities
            if any((f in P.exclude_flags) for f in (r.flags or [])):
                excl = True
            # very low composite
            if r.composite < P.min_inclusion_score:
                excl = True
            r.exclude = bool(excl)

            # risk overrides (position caps / leverage multipliers)
            # map composite -> leverage multiplier between [floor, cap]
            lev = P.leverage_floor + (P.leverage_cap - P.leverage_floor) * r.composite
            # cap position weight for excluded/severe controversy or high carbon decile
            max_w = 0.0 if r.exclude else (0.04 if r.decile <= 1 else 0.08 if r.decile <= 3 else 0.12)
            r.risk_overrides = {
                "leverage_mult": float(lev),
                "max_weight": float(max_w),
                "esg_decile": int(r.decile),
                "exclude": bool(r.exclude),
            }

    # ---- Publish / Persist ----
    def publish(self, rows: List[ESGRow], namespace: str = "esg") -> None:
        if not rows: return
        # HSET esg:scores <symbol> {composite:..., tilt:..., exclude:...}
        if hset:
            for r in rows:
                hset(f"{namespace}:scores", r.symbol, {
                    "composite": r.composite,
                    "tilt": r.tilt,
                    "exclude": r.exclude,
                    "decile": r.decile,
                    **r.risk_overrides
                })
        if publish_stream:
            try:
                publish_stream(f"{namespace}.events", {
                    "ts_ms": _now_ms(),
                    "kind": "refresh",
                    "count": len(rows)
                })
            except Exception:
                pass

    # ---- Optimizer constraints (optional helper) ----
    def to_optimizer_constraints(self, rows: List[ESGRow]) -> Dict[str, Any]:
        """
        Return simple constraints you can feed to your optimizer:
        - per-symbol max weight
        - portfolio carbon-intensity target reduction vs median
        """
        if not rows:
            return {"max_weight": {}, "portfolio_targets": {}}

        median_carbon = None
        carb_vals = [r.carbon_intensity_tco2e_per_musd for r in rows if r.carbon_intensity_tco2e_per_musd is not None]
        if carb_vals:
            if pd:
                median_carbon = float(pd.Series(carb_vals).median())
            else:
                carb_vals.sort()
                median_carbon = float(carb_vals[len(carb_vals)//2])

        max_w = {r.symbol: (0.0 if r.exclude else r.risk_overrides.get("max_weight", 0.1)) for r in rows}
        targets = {}
        if median_carbon:
            targets["carbon_intensity_max"] = (1.0 - self.policy.carbon_target_pct_reduction) * median_carbon

        return {"max_weight": max_w, "portfolio_targets": targets}


# ----------------------------- CLI -----------------------------

def _probe():
    # Minimal probe with toy data
    engine = ESGEngine()
    rows = engine.from_records([
        {"symbol":"RELIANCE.NS","sector":"Energy","region":"IN","esg_env":52,"esg_soc":61,"esg_gov":58,"carbon_intensity_tco2e_per_musd":420,"controversies":2,"flags":[],"news_sentiment":0.1},
        {"symbol":"TCS.NS","sector":"IT","region":"IN","esg_env":78,"esg_soc":82,"esg_gov":75,"carbon_intensity_tco2e_per_musd":45,"controversies":1,"flags":[],"news_sentiment":0.2},
        {"symbol":"HDFCBANK.NS","sector":"Financials","region":"IN","esg_env":65,"esg_soc":70,"esg_gov":72,"carbon_intensity_tco2e_per_musd":25,"controversies":4,"flags":["tobacco"]},
    ])
    out = engine.score(rows)
    for r in out:
        print(r.symbol, "comp=", round(r.composite,3), "tilt=", round(r.tilt,3), "excl=", r.exclude, "decile=", r.decile, r.risk_overrides)

def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="ESG Factor Engine")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--csv", type=str, help="CSV with columns: symbol,sector,region,esg_env,esg_soc,esg_gov,provider_score,carbon_intensity_tco2e_per_musd,controversies,flags,news_sentiment,market_cap,free_float")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--out", type=str, help="Write enriched CSV")
    args = ap.parse_args()

    if args.probe:
        _probe(); return

    engine = ESGEngine()
    rows: List[ESGRow] = []
    if args.csv:
        if pd is None:
            raise RuntimeError("pandas not installed. Run: pip install pandas")
        df = pd.read_csv(args.csv)
        # flags may be json-like; try to parse to list
        def _parse_flags(x):
            if isinstance(x, str):
                x = x.strip()
                if x.startswith("[") and x.endswith("]"):
                    try:
                        import json as _json
                        return list(_json.loads(x))
                    except Exception:
                        pass
                if x:
                    return [x]
            return []
        if "flags" in df.columns:
            df["flags"] = df["flags"].apply(_parse_flags)
        rows = engine.from_records(df.to_dict("records")) # type: ignore
    else:
        _probe(); return

    out = engine.score(rows)
    if args.publish:
        engine.publish(out)

    if args.out:
        if pd is None:
            raise RuntimeError("pandas not installed for saving CSV")
        pdf = pd.DataFrame([r.to_dict() for r in out])
        pdf.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps([r.to_dict() for r in out], indent=2, default=float))

if __name__ == "__main__":
    main()