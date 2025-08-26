# backend/sim/crisis_theatre.py
"""
Crisis Theatre
--------------
Scenario orchestrator that composes your macro & market simulators into a
time-lined "play" with acts, triggers, and policy responses.

It coordinates:
- Sovereign risk (backend/macro/soverign.py)
- Bank stress (backend/risk/bank_stress.py)
- Liquidity spiral (backend/risk/liquidity_spiral.py)

Concepts
- Actor: a model instance (sovereign engine, bank engine, liquidity bucket sim)
- Act:   a time-bounded scene with scripted shocks and triggers
- Cue:   a trigger (condition) that fires mid-act and applies responses
- Bus:   optional publish to Redis streams for dashboards

CLI
  python -m backend.sim.crisis_theatre --probe
  python -m backend.sim.crisis_theatre --yaml config/crisis.yaml --run

YAML (config/crisis.yaml) — minimal sketch
------------------------------------------
timeline:
  dt_hours: 1
  steps: 72
actors:
  sovereigns:
    IN: {yaml: config/soverign.yaml, period: "M"}     # uses your soverign loader
  banks:
    BANK_A: {yaml: config/bank_stress.yaml}
  liquidity:
    IG: {price: 1.0, spread_bps: 70, haircut: 0.12, depth: 8e9, inventory: 30e9, leverage: 6.0}
    EQ: {price: 1.0, spread_bps: 25, haircut: 0.15, depth: 10e9, inventory: 25e9, leverage: 3.0}
acts:
  - name: "Opening tremor"
    t_start: 0
    t_end: 24
    shocks:
      sovereigns:
        IN: {rate: 0.005, fx: 0.03}      # +50bp rate, +3% FX depreciation per period step
      banks:
        BANK_A: "ir:+50,sov:+60,runoff:0.02"
      liquidity:
        IG: {d_funding: 0.005, d_haircut: 0.02, natural_flow: 1.5e9}
    cues:
      - when: "sov.IN.spread_bps >= 150"
        do:
          banks:
            BANK_A: "sov:+150,runoff:0.06,freeze:0.2"
          liquidity:
            IG: {d_spread_bps: 40}
  - name: "Policy response"
    t_start: 24
    t_end: 48
    shocks:
      sovereigns:
        IN: {rate: -0.002}
    policies:
      - at: 30
        type: "central_bank_liquidity"
        params: {bucket: "IG", add_depth: 5e9, cut_haircut: 0.03}

Streams (if Redis present)
- crisis.snapshots
- risk.sovereign (forwarded from sovereign engine if enabled)
- risk.bank_stress (forwarded from bank engine if enabled)
- risk.liqspiral (forwarded from spiral if enabled)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---- Optional deps / glue ----
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # type: ignore

try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

# Engines you already have
try:
    from backend.macro.soverign import SovereignEngine, SovereignState, load_from_yaml as load_sovereign # type: ignore
except Exception:
    SovereignEngine = None  # type: ignore
    load_sovereign = None   # type: ignore

try:
    from backend.risk.bank_stress import BankStressEngine, BankState, ShockSpec, load_from_yaml as load_bank, _parse_scenario # type: ignore
except Exception:
    BankStressEngine = None  # type: ignore
    load_bank = None         # type: ignore
    ShockSpec = None         # type: ignore
    def _parse_scenario(s: str): return None  # type: ignore

try:
    from backend.risk.liquidity_spiral import LiquiditySpiral, Bucket, SpiralConfig, Shock as LShock
except Exception:
    LiquiditySpiral = None  # type: ignore
    Bucket = None           # type: ignore
    SpiralConfig = None     # type: ignore
    LShock = None           # type: ignore


# ---------------- Data Model ----------------

@dataclass
class Timeline:
    dt_hours: float = 1.0
    steps: int = 72

@dataclass
class TheatreConfig:
    timeline: Timeline = field(default_factory=Timeline)
    actors: Dict[str, Any] = field(default_factory=dict)
    acts: List[Dict[str, Any]] = field(default_factory=list)
    publish_topic: str = "crisis.snapshots"


# ---------------- Orchestrator ----------------

class CrisisTheatre:
    def __init__(self, cfg: TheatreConfig):
        self.cfg = cfg
        self.t = 0  # step index
        self.sov: Dict[str, SovereignEngine] = {} # type: ignore
        self.banks: Dict[str, BankStressEngine] = {} # type: ignore
        self.liq: Optional[LiquiditySpiral] = None # type: ignore
        self._liq_buckets: Dict[str, Bucket] = {}  # type: ignore # for external policy tweaks

        self._prepare_actors(cfg.actors)

    # ---- setup actors from config dict ----
    def _prepare_actors(self, actors: Dict[str, Any]) -> None:
        # Sovereigns
        for code, spec in (actors.get("sovereigns") or {}).items():
            if load_sovereign is None:
                continue
            st = load_sovereign(spec["yaml"], code, spec.get("period","M")) if isinstance(spec, dict) and "yaml" in spec else None
            if st is None and SovereignEngine:
                # minimal fallback
                st = SovereignState(
                    code=code, gdp_usd=1e12, debt_gdp=0.6, avg_coupon=0.05, duration_yrs=6.0,
                    rollover_ratio_12m=0.2, primary_balance_gdp=-0.02, growth=0.04, inflation=0.04,
                    fx_reserves_usd=2e11, imports_usd_m=20e9, external_debt_usd=2e11, local_share=0.6, rating="BBB-"
                )
            if SovereignEngine and st:
                self.sov[code] = SovereignEngine(st)

        # Banks
        for name, spec in (actors.get("banks") or {}).items():
            if load_bank is None:
                continue
            bs, defs = load_bank(spec["yaml"], name) if isinstance(spec, dict) and "yaml" in spec else (None, None)
            if BankStressEngine and bs:
                self.banks[name] = BankStressEngine(bs, defs)

        # Liquidity buckets
        liq_cfg = actors.get("liquidity") or {}
        for bname, b in liq_cfg.items():
            if Bucket is None:
                continue
            self._liq_buckets[bname] = Bucket(
                name=bname,
                price=float(b.get("price",1.0)),
                spread_bps=float(b.get("spread_bps",30.0)),
                haircut=float(b.get("haircut",0.15)),
                depth=float(b.get("depth",5e9)),
                inventory=float(b.get("inventory",10e9)),
                leverage=float(b.get("leverage",4.0)),
                funding_rate=float(b.get("funding_rate",0.03)),
                margin_buffer=float(b.get("margin_buffer",0.15)),
                target_leverage=float(b.get("target_leverage",3.0)),
                fire_sale_elasticity=float(b.get("fire_sale_elasticity",0.08))
            )
        if LiquiditySpiral and self._liq_buckets:
            dt = (self.cfg.timeline.dt_hours or 1.0) / 24.0
            self.liq = LiquiditySpiral(self._liq_buckets, SpiralConfig(dt=dt)) # type: ignore

    # ---- run the play ----
    def run(self, steps: Optional[int] = None, publish: bool = True) -> List[Dict[str, Any]]:
        total = min(steps or self.cfg.timeline.steps, self.cfg.timeline.steps)
        snapshots: List[Dict[str, Any]] = []

        for self.t in range(total):
            # Determine current act(s)
            acts_now = [a for a in self.cfg.acts if int(a.get("t_start", 0)) <= self.t < int(a.get("t_end", total))]
            # 1) Apply scripted shocks for this step
            self._apply_shocks(acts_now)

            # 2) Advance engines one tick
            # Sovereigns
            for code, eng in list(self.sov.items()):
                eng.step(publish=False)  # act shocks already applied via step()
            # Liquidity
            liq_snap = None
            if self.liq:
                exo = getattr(self, "_pending_liq_shocks", None) or {}
                liq_snap = self.liq.step(exo, publish=False)
                self._pending_liq_shocks = {}  # reset
            # Banks
            for name, eng in list(self.banks.items()):
                spec = getattr(self, "_pending_bank_shock", {}).get(name)
                if spec is None:
                    spec = _parse_scenario("") if _parse_scenario else None
                eng.step(spec, publish=False)
            self._pending_bank_shock = {}

            # 3) Evaluate cues/triggers (after state update)
            self._apply_cues(acts_now)

            # 4) Optional policy injections at discrete times
            self._apply_policies(acts_now)

            # 5) Emit a consolidated snapshot
            snap = self._snapshot(liq_snap)
            snapshots.append(snap)
            if publish and publish_stream:
                try:
                    publish_stream(self.cfg.publish_topic, snap)
                except Exception:
                    pass

        return snapshots

    # ---- shocks helpers ----
    def _apply_shocks(self, acts_now: List[Dict[str, Any]]) -> None:
        # Sovereign shocks: call engine.step with deltas
        for act in acts_now:
            for code, s in (act.get("shocks", {}).get("sovereigns") or {}).items():
                eng = self.sov.get(code)
                if not eng: continue
                rate = float(s.get("rate", 0.0))
                growth = float(s.get("growth", 0.0))
                fx = float(s.get("fx", 0.0))
                pb = float(s.get("pb", 0.0))
                commodity = float(s.get("commodity", 0.0))
                eng.step(d_rate=rate, d_growth=growth, d_fx=fx, d_pb_gdp=pb, commodity_shock_gdp=commodity, publish=False)

        # Liquidity shocks: stage for this step; applied in liq.step
        pending_liq: Dict[str, Any] = getattr(self, "_pending_liq_shocks", {})
        for act in acts_now:
            for bname, s in (act.get("shocks", {}).get("liquidity") or {}).items():
                if LShock is None: continue
                sh = pending_liq.get(bname, LShock())
                sh.d_spread_bps += float(s.get("d_spread_bps", 0.0))
                sh.d_funding += float(s.get("d_funding", 0.0))
                sh.d_haircut += float(s.get("d_haircut", 0.0))
                sh.natural_flow += float(s.get("natural_flow", 0.0))
                pending_liq[bname] = sh
        self._pending_liq_shocks = pending_liq

        # Bank shocks: parse scenario strings and accumulate
        pending_bank: Dict[str, Any] = getattr(self, "_pending_bank_shock", {})
        for act in acts_now:
            for bname, s in (act.get("shocks", {}).get("banks") or {}).items():
                if isinstance(s, str) and _parse_scenario:
                    spec = _parse_scenario(s)
                elif isinstance(s, dict) and _parse_scenario:
                    # allow dict form "ir_bp: 100, runoff: 0.1"
                    parts = []
                    for k, v in s.items():
                        parts.append(f"{k}:{v}")
                    spec = _parse_scenario(",".join(parts))
                else:
                    spec = None
                if spec:
                    pending_bank[bname] = spec
        self._pending_bank_shock = pending_bank

    # ---- cues (triggers) ----
    def _apply_cues(self, acts_now: List[Dict[str, Any]]) -> None:
        # Build quick state for expressions
        state = {
            "t": self.t,
            "sov": {k: {"spread_bps": float(v.s.spread_bps), "dd": float(v.s.default_prob_12m)} for k, v in self.sov.items()},
            "liq": {k: {"price": float(b.price), "spread_bps": float(b.spread_bps), "haircut": float(b.haircut)} for k, b in self._liq_buckets.items()},
            "bank": {k: {"cet1": float(v.s.cet1_ratio), "lcr": float(v.s.lcr), "nsfr": float(v.s.nsfr)} for k, v in self.banks.items()},
        }

        def _eval(expr: str) -> bool:
            try:
                # super small sandbox: allow only names in 'state'
                allowed = {"t": state["t"], "sov": state["sov"], "liq": state["liq"], "bank": state["bank"]}
                return bool(eval(expr, {"__builtins__": {}}, allowed))
            except Exception:
                return False

        for act in acts_now:
            for cue in (act.get("cues") or []):
                expr = str(cue.get("when","")).strip()
                if not expr or not _eval(expr):
                    continue
                # Apply responses to banks/liquidity/sovereigns
                # Banks
                for bname, scen in (cue.get("do", {}).get("banks") or {}).items():
                    if _parse_scenario and bname in self.banks:
                        spec = _parse_scenario(scen if isinstance(scen,str) else "")
                        if spec:
                            self._pending_bank_shock[bname] = spec
                # Liquidity
                liq_do = (cue.get("do", {}).get("liquidity") or {})
                for bname, s in liq_do.items():
                    if LShock is None: continue
                    sh = self._pending_liq_shocks.get(bname, LShock())
                    sh.d_spread_bps += float(s.get("d_spread_bps", 0.0))
                    sh.d_funding += float(s.get("d_funding", 0.0))
                    sh.d_haircut += float(s.get("d_haircut", 0.0))
                    sh.natural_flow += float(s.get("natural_flow", 0.0))
                    self._pending_liq_shocks[bname] = sh
                # Sovereigns
                for code, s in (cue.get("do", {}).get("sovereigns") or {}).items():
                    eng = self.sov.get(code)
                    if not eng: continue
                    eng.step(d_rate=float(s.get("rate",0.0)),
                             d_growth=float(s.get("growth",0.0)),
                             d_fx=float(s.get("fx",0.0)),
                             d_pb_gdp=float(s.get("pb",0.0)),
                             commodity_shock_gdp=float(s.get("commodity",0.0)),
                             publish=False)

    # ---- policies ----
    def _apply_policies(self, acts_now: List[Dict[str, Any]]) -> None:
        for act in acts_now:
            for p in (act.get("policies") or []):
                at = int(p.get("at", -1))
                if at >= 0 and at != self.t:
                    continue
                ptype = p.get("type")
                params = p.get("params") or {}

                if ptype == "central_bank_liquidity" and self.liq:
                    # Increase depth and cut haircuts in a target bucket
                    bname = str(params.get("bucket") or "")
                    b = self._liq_buckets.get(bname)
                    if not b: continue
                    add_depth = float(params.get("add_depth", 0.0))
                    cut_hair = float(params.get("cut_haircut", 0.0))
                    b.depth += max(0.0, add_depth)
                    b.haircut = max(0.0, b.haircut - max(0.0, cut_hair))
                elif ptype == "swap_line" and self.sov:
                    # Improve sovereign reserves (FX swap line)
                    code = str(params.get("country") or "")
                    eng = self.sov.get(code)
                    if eng:
                        eng.s.fx_reserves_usd += float(params.get("add_reserves", 0.0))
                elif ptype == "capital_backstop" and self.banks:
                    name = str(params.get("bank") or "")
                    eng = self.banks.get(name)
                    if eng:
                        eng.s.equity_tier1 += float(params.get("amount", 0.0))

    # ---- snapshot ----
    def _snapshot(self, liq_snap: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        out = {
            "t": self.t,
            "sovereign": {k: {
                "spread_bps": float(v.s.spread_bps),
                "hazard": float(v.s.hazard_annual),
                "default_prob_12m": float(v.s.default_prob_12m),
                "debt_gdp": float(v.s.debt_gdp),
                "reserves_m": float(v.s.reserves_months)
            } for k, v in self.sov.items()},
            "bank": {k: {
                "cet1": float(v.s.cet1_ratio),
                "lcr": float(v.s.lcr),
                "nsfr": float(v.s.nsfr),
                "pnl_after_tax": float(v.s.pnl_after_tax),
                "oci_afs": float(v.s.oci_afs),
                "breaches": list(v.s.breaches)
            } for k, v in self.banks.items()},
            "liquidity": (liq_snap or {k: {
                "price": float(b.price),
                "spread_bps": float(b.spread_bps),
                "haircut": float(b.haircut),
                "depth": float(b.depth),
                "inventory": float(b.inventory),
                "leverage": float(b.leverage)
            } for k, b in self._liq_buckets.items()}),
        }
        return out


# ---------------- YAML loader ----------------

def load_yaml(path: str) -> TheatreConfig:
    if yaml is None:
        raise RuntimeError("pyyaml not installed. Run: pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    tl = doc.get("timeline") or {}
    cfg = TheatreConfig(
        timeline=Timeline(dt_hours=float(tl.get("dt_hours", 1.0)), steps=int(tl.get("steps", 72))),
        actors=doc.get("actors") or {},
        acts=doc.get("acts") or [],
        publish_topic=str(doc.get("publish_topic") or "crisis.snapshots"),
    )
    return cfg


# ---------------- CLI ----------------

def _probe():
    # Minimal self-contained demo (no YAML required)
    if SovereignEngine and BankStressEngine and LiquiditySpiral:
        # Tiny defaults: 48 steps @ 1h
        cfg = TheatreConfig(
            timeline=Timeline(dt_hours=1, steps=48),
            actors={
                "sovereigns": {"IN": {"yaml": "config/soverign.yaml", "period": "M"}},
                "banks": {"BANK_A": {"yaml": "config/bank_stress.yaml"}},
                "liquidity": {
                    "IG": {"price":1.0,"spread_bps":70,"haircut":0.12,"depth":8e9,"inventory":30e9,"leverage":6.0},
                    "EQ": {"price":1.0,"spread_bps":25,"haircut":0.15,"depth":10e9,"inventory":25e9,"leverage":3.0}
                },
            },
            acts=[
                {"name":"Shock", "t_start":0, "t_end":12,
                 "shocks":{
                    "sovereigns":{"IN":{"rate":0.01,"fx":0.05}},
                    "banks":{"BANK_A":"ir:+100,sov:+120,runoff:0.05"},
                    "liquidity":{"IG":{"d_haircut":0.03,"d_funding":0.01,"natural_flow":1.5e9}}
                 },
                 "cues":[
                    {"when":"sov['IN']['spread_bps'] >= 150",
                     "do":{"banks":{"BANK_A":"sov:+150,runoff:0.07"},
                           "liquidity":{"IG":{"d_spread_bps":50}}}}
                 ]},
                {"name":"Policy", "t_start":12, "t_end":24,
                 "policies":[{"at":12,"type":"central_bank_liquidity","params":{"bucket":"IG","add_depth":5e9,"cut_haircut":0.03}}]}
            ]
        )
        theatre = CrisisTheatre(cfg)
        out = theatre.run(publish=False)
        # print last snapshot for sanity
        print(json.dumps(out[-1], indent=2, default=float))
    else:
        print("Probe requires sovereign, bank, and liquidity modules available.")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Crisis Theatre Orchestrator")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--yaml", type=str, help="config/crisis.yaml")
    ap.add_argument("--steps", type=int, help="override steps")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    if args.probe:
        _probe(); return

    if args.yaml:
        cfg = load_yaml(args.yaml)
    else:
        _probe(); return

    theatre = CrisisTheatre(cfg)
    snaps = theatre.run(steps=args.steps, publish=not args.no_publish)
    print(json.dumps({"steps": len(snaps), "last": snaps[-1]}, indent=2, default=float))

if __name__ == "__main__":
    main()