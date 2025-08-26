# backend/macro/soverign.py
"""
Sovereign Risk Engine (misspelled as 'soverign' intentionally to match file name)
---------------------------------------------------------------------------------
Dynamics for country-level credit health:
- Debt/GDP, primary balance, interest costs, growth, inflation
- Rollover risk (maturity wall), external financing gap, reserve adequacy
- Reduced-form default hazard => CDS/spread
- Shocks: rates, growth, FX, commodity, fiscal; with attribution trail
- Optional publish to bus + hook into contagion graph

CLI
  python -m backend.macro.soverign --probe
  python -m backend.macro.soverign --run --yaml config/soverign.yaml --country IN

YAML (example: config/soverign.yaml)
------------------------------------
countries:
  IN:
    gdp: 3.6e12                # USD
    debt_gdp: 0.82
    avg_coupon: 0.065
    duration_yrs: 6.2
    rollover_ratio_12m: 0.18
    primary_balance_gdp: -0.03
    growth: 0.06               # real
    inflation: 0.045
    fx_reserves_usd: 6.0e11
    imports_usd_m: 55e9
    fx_pass_through: 0.25
    external_debt_usd: 6.2e11
    local_share: 0.63
    rating: "BBB-"
    recovery_rate: 0.4
    base_hazard: 0.012         # annual
    risk_beta: { rate: 0.6, growth: -0.8, fx: 0.4, pb: -1.2, reserves: -0.7 }
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List

# optional deps
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # type: ignore

# optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

_NOW_MS = lambda: int(time.time() * 1000)


# ---------------- Data Models ----------------

@dataclass
class SovereignState:
    code: str
    gdp_usd: float
    debt_gdp: float                    # D/Y
    avg_coupon: float                  # weighted avg nominal rate on debt
    duration_yrs: float
    rollover_ratio_12m: float          # share of debt maturing in next 12m
    primary_balance_gdp: float         # PB/Y (surplus +, deficit -)
    growth: float                      # real
    inflation: float
    fx_reserves_usd: float
    imports_usd_m: float               # monthly imports (USD)
    external_debt_usd: float
    local_share: float                 # share of LC debt in total
    rating: str
    recovery_rate: float = 0.4
    base_hazard: float = 0.01          # annual baseline hazard
    fx_pass_through: float = 0.3
    risk_beta: Dict[str, float] = field(default_factory=lambda: {"rate":0.5,"growth":-0.5,"fx":0.3,"pb":-1.0,"reserves":-0.5})

    # derived / evolving fields
    period: str = "M"                  # "M" monthly, "Q" quarterly
    t: int = 0
    spread_bps: float = 0.0
    hazard_annual: float = 0.0
    default_prob_12m: float = 0.0
    reserves_months: float = 0.0
    financing_gap_usd: float = 0.0
    rollover_stress: float = 0.0
    notes: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "SovereignState":
        # shallow copy with independent notes list
        c = SovereignState(**{k: v for k, v in asdict(self).items() if k != "notes"})
        c.notes = list(self.notes)
        return c


# ---------------- Core Engine ----------------

class SovereignEngine:
    def __init__(self, state: SovereignState):
        self.s = state
        self._dt = 1/12 if state.period.upper().startswith("M") else 1/4

    # ---- step dynamics ----
    def step(self, *,
             d_rate: float = 0.0,            # Δ nominal rate (level) in absolute terms (e.g., +0.01 = +100bps)
             d_growth: float = 0.0,          # Δ real growth
             d_fx: float = 0.0,              # Δ FX (depreciation fraction of local vs USD), +0.1 = 10% weaker LC
             d_pb_gdp: float = 0.0,          # Δ primary bal. (% of GDP)
             commodity_shock_gdp: float = 0.0,   # income shock as % of GDP
             publish: bool = False) -> SovereignState:
        """
        Advance one period and update:
        - Debt/GDP via debt dynamics: D_{t+1} = D_t + (r - g - π)*D_t * dt - PB*Y*dt + rollover effects + commodity shock
        - Reserves: Res_{t+1} = Res_t + current_account + flows (simplified: -financing_gap)
        - Hazard & spreads via reduced-form mapping
        """
        s = self.s
        dt = self._dt

        # 1) Update macro inputs
        s.avg_coupon = max(0.0, s.avg_coupon + d_rate)
        s.growth = max(-0.1, s.growth + d_growth)
        s.primary_balance_gdp += d_pb_gdp
        # FX pass-through -> inflation drift
        s.inflation = max(-0.05, s.inflation + s.fx_pass_through * d_fx)

        # 2) Debt dynamics
        D = s.debt_gdp * s.gdp_usd          # debt in USD terms (approx)
        r = s.avg_coupon                    # nominal rate
        g = s.growth
        pi = s.inflation
        primary = s.primary_balance_gdp * s.gdp_usd

        # interest net of growth/inflation (snowball effect)
        delta_D_snowball = (r - g - pi) * D * dt

        # primary balance reduces debt if surplus
        delta_D_primary = -primary * dt

        # rollover stress: if high maturity wall and rates jump, effective cost spikes
        roll = s.rollover_ratio_12m
        rollover_penalty = max(0.0, roll * max(0.0, d_rate)) * D * 0.5 * dt  # heuristic
        s.rollover_stress = float(rollover_penalty / max(1e-9, D))

        # commodity shock (e.g., oil importers): add to debt if negative income
        delta_D_commodity = -commodity_shock_gdp * s.gdp_usd * dt

        D_next = max(0.0, D + delta_D_snowball + delta_D_primary + rollover_penalty + delta_D_commodity)
        s.debt_gdp = float(D_next / max(1e-9, s.gdp_usd))

        # 3) External financing gap and reserves adequacy
        # crude: gap = external_debt rollovers * rate change + CA shortfall (proxy with commodity_shock)
        rollover_ext = s.external_debt_usd * min(1.0, roll) * dt
        gap = rollover_ext * max(0.0, d_rate) - commodity_shock_gdp * s.gdp_usd * dt
        s.financing_gap_usd = float(gap)

        reserves_next = max(0.0, s.fx_reserves_usd - max(0.0, gap))
        s.fx_reserves_usd = reserves_next
        s.reserves_months = float(reserves_next / max(1e-9, s.imports_usd_m))

        # 4) Reduced-form hazard & spread
        # log-hazard = base + beta_r*Δr + beta_g*Δg + beta_fx*Δfx + beta_pb*PB + beta_res*reserves_months
        beta = s.risk_beta or {}
        log_h = math.log(max(1e-6, s.base_hazard)) \
                + beta.get("rate", 0.5) * d_rate \
                + beta.get("growth", -0.5) * d_growth \
                + beta.get("fx", 0.3) * d_fx \
                + beta.get("pb", -1.0) * s.primary_balance_gdp \
                + beta.get("reserves", -0.5) * (s.reserves_months - 6) / 6.0 \
                + 3.0 * s.rollover_stress
        s.hazard_annual = max(1e-6, min(1.0, math.exp(log_h)))
        # CDS ~ hazard * (1 - R) in annualized terms, convert to bps
        s.spread_bps = float(1e4 * s.hazard_annual * (1.0 - s.recovery_rate))
        # 12m default probability (compounding)
        s.default_prob_12m = float(1.0 - math.exp(-s.hazard_annual))

        # 5) Log attribution
        s.t += 1
        s.notes.append({
            "t": s.t, "dt": dt, "d_rate": d_rate, "d_growth": d_growth, "d_fx": d_fx,
            "d_pb_gdp": d_pb_gdp, "commodity": commodity_shock_gdp,
            "delta_D_snowball": delta_D_snowball, "delta_D_primary": delta_D_primary,
            "rollover_penalty": rollover_penalty, "delta_D_commodity": delta_D_commodity,
            "reserves_m": s.reserves_months, "spread_bps": s.spread_bps, "hazard": s.hazard_annual
        })

        if publish and publish_stream:
            try:
                publish_stream("risk.sovereign", {
                    "ts_ms": _NOW_MS(), "country": s.code, "t": s.t,
                    "spread_bps": s.spread_bps, "hazard": s.hazard_annual,
                    "debt_gdp": s.debt_gdp, "reserves_m": s.reserves_months,
                    "default_prob_12m": s.default_prob_12m, "rollover_stress": s.rollover_stress
                })
            except Exception:
                pass

        return s

    # ---- shocks as a convenience API ----
    def apply_shock(self, spec: Dict[str, float], publish: bool = False) -> SovereignState:
        """
        spec keys: 'rate','growth','fx','pb','commodity'
        Example: {"rate": 0.01, "fx": 0.08}
        """
        return self.step(
            d_rate=spec.get("rate", 0.0),
            d_growth=spec.get("growth", 0.0),
            d_fx=spec.get("fx", 0.0),
            d_pb_gdp=spec.get("pb", 0.0),
            commodity_shock_gdp=spec.get("commodity", 0.0),
            publish=publish,
        )

    # ---- coupling into contagion graph (optional) ----
    def to_contagion_loss(self, bank_equity_exposure: float) -> float:
        """
        Map sovereign spread jump/hazard to an MTM loss on holders (very crude).
        loss ≈ Δspread (in decimal) * duration * exposure.
        """
        s = self.s
        # last two notes to get delta spread
        if len(s.notes) < 2:
            return 0.0
        d_spread = (s.notes[-1]["spread_bps"] - s.notes[-2]["spread_bps"]) / 1e4  # decimal
        return max(0.0, d_spread) * max(0.1, s.duration_yrs) * max(0.0, bank_equity_exposure)


# ---------------- Factory / YAML ----------------

def load_from_yaml(path: str, code: str, period: str = "M") -> SovereignState:
    if yaml is None:
        raise RuntimeError("pyyaml not installed. Run: pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    cfg = (doc.get("countries") or {}).get(code)
    if not cfg:
        raise KeyError(f"{code} not in {path}")

    return SovereignState(
        code=code,
        gdp_usd=float(cfg.get("gdp")),
        debt_gdp=float(cfg.get("debt_gdp")),
        avg_coupon=float(cfg.get("avg_coupon")),
        duration_yrs=float(cfg.get("duration_yrs")),
        rollover_ratio_12m=float(cfg.get("rollover_ratio_12m")),
        primary_balance_gdp=float(cfg.get("primary_balance_gdp")),
        growth=float(cfg.get("growth")),
        inflation=float(cfg.get("inflation")),
        fx_reserves_usd=float(cfg.get("fx_reserves_usd")),
        imports_usd_m=float(cfg.get("imports_usd_m")),
        external_debt_usd=float(cfg.get("external_debt_usd")),
        local_share=float(cfg.get("local_share")),
        rating=str(cfg.get("rating","NR")),
        recovery_rate=float(cfg.get("recovery_rate", 0.4)),
        base_hazard=float(cfg.get("base_hazard", 0.01)),
        fx_pass_through=float(cfg.get("fx_pass_through", 0.3)),
        risk_beta=dict(cfg.get("risk_beta") or {}),
        period=period,
    )


# ---------------- CLI probe ----------------

def _probe():
    # simple India-like example without YAML
    st = SovereignState(
        code="IN",
        gdp_usd=3.6e12,
        debt_gdp=0.82,
        avg_coupon=0.065,
        duration_yrs=6.0,
        rollover_ratio_12m=0.18,
        primary_balance_gdp=-0.03,
        growth=0.06,
        inflation=0.045,
        fx_reserves_usd=6.0e11,
        imports_usd_m=55e9,
        external_debt_usd=6.2e11,
        local_share=0.63,
        rating="BBB-",
        recovery_rate=0.4,
        base_hazard=0.012,
        fx_pass_through=0.25,
        risk_beta={"rate":0.6,"growth":-0.8,"fx":0.4,"pb":-1.2,"reserves":-0.7},
        period="M",
    )
    eng = SovereignEngine(st)
    # apply a rate + FX shock
    eng.apply_shock({"rate":0.01, "fx":0.07}, publish=False)
    eng.apply_shock({"growth":-0.02, "pb":-0.005}, publish=False)
    s = eng.s
    print(f"t={s.t} debt/GDP={s.debt_gdp:.2f} spread={s.spread_bps:.0f}bps reserves={s.reserves_months:.1f}m "
          f"default12m={100*s.default_prob_12m:.2f}%")

def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Sovereign Risk Engine")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--yaml", type=str, help="Path to config/soverign.yaml")
    ap.add_argument("--country", type=str, default="IN")
    args = ap.parse_args()

    if args.probe:
        _probe()
        return

    if args.yaml:
        st = load_from_yaml(args.yaml, args.country)
    else:
        # fallback to probe defaults
        _probe()
        return

    eng = SovereignEngine(st)
    # run 12 monthly steps with a mild tightening cycle
    for i in range(12):
        eng.step(d_rate=0.002 if i < 6 else 0.0, d_growth=-0.001 if i < 6 else 0.0, publish=False)
    print(json.dumps(asdict(eng.s), indent=2))

if __name__ == "__main__":
    main()