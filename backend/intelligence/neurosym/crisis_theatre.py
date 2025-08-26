# backend/sim/crisis_theatre.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# ---------------- Soft imports (stay runnable even if parts are missing) -----
try:
    from backend.macro.central_bank_ai import ( # type: ignore
        PolicyState, EconEvent, decide_next_meeting, simulate_policy_path,
        path_to_yield_curve, curve_vs_base_to_shock, decision_to_json
    )
except Exception:
    PolicyState = object; EconEvent = object
    def decide_next_meeting(*a, **k): return None
    def simulate_policy_path(*a, **k): return []
    def path_to_yield_curve(*a, **k): return type("YC", (), {"sorted_points": lambda s: []})()
    def curve_vs_base_to_shock(*a, **k): return {"custom_tenor_bp": {}}
    def decision_to_json(x): return "{}"

try:
    from backend.risk.contagian_graph import ContagionGraph, Bank, ShockParams # type: ignore
except Exception:
    ContagionGraph = object; Bank = object
    class ShockParams: ...

try:
    from backend.risk.governor import Governor, Policy as GovPolicy # type: ignore
except Exception:
    Governor = None
    class GovPolicy: ...

try:
    from backend.risk.adversary import default_suite, GuardrailPolicy # type: ignore
except Exception:
    def default_suite(*a, **k): return None
    class GuardrailPolicy:
        def decide(self, precheck, default="CROSS"): return {"mode": "CROSS", "tox": 0.0}

# -----------------------------------------------------------------------------
# Event & scene model
# -----------------------------------------------------------------------------

@dataclass
class Cue:
    """A timed trigger in the theatre."""
    t_min: int                                  # minutes from scenario start
    action: str                                  # 'policy', 'shock', 'halt', 'resume', 'narrate', 'kill'
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Scene:
    """A sequence of cues that play over time."""
    name: str
    cues: List[Cue]

@dataclass
class Outcome:
    """Collected telemetry from a run."""
    started_at: float
    finished_at: float
    frames: List[Dict[str, Any]]                # snapshots per minute
    notes: List[str] = field(default_factory=list)

# -----------------------------------------------------------------------------
# Theatre core
# -----------------------------------------------------------------------------

class CrisisTheatre:
    """
    Orchestrates a time-stepped crisis “play” across:
      - Central bank policy engine (rates, guidance, curve)
      - Interbank contagion (defaults, fire-sales, price impact)
      - Execution guardrails (governor, adversary toxicity)
      - Symbol halts / resumes to mimic news regulators
    """
    def __init__(
        self,
        *,
        macro_state: Optional[PolicyState] = None, # type: ignore
        base_curve: Optional[Any] = None,              # YieldCurve
        cont_params: Optional[ShockParams] = None,
        governor: Optional[Governor] = None, # type: ignore
        adversaries: Optional[Any] = None,
        rng_seed: Optional[int] = 42,
    ):
        self._t0 = time.time()
        self.macro = macro_state
        self.base_curve = base_curve
        self.contagion = ContagionGraph(cont_params or ShockParams()) # type: ignore
        self.gov = governor or (Governor(GovPolicy()) if Governor else None)
        self.advs = adversaries or default_suite()
        self.guard = GuardrailPolicy()
        self.rng_seed = rng_seed

        self._minutes = 0
        self._frames: List[Dict[str, Any]] = []
        self._notes: List[str] = []

    # --------------------- Building blocks ---------------------

    def add_bank(self, bank: Bank) -> None: # type: ignore
        self.contagion.add_bank(bank) # type: ignore

    def add_exposure(self, lender: str, borrower: str, amount: float, recovery_rate: float = 0.4) -> None:
        self.contagion.add_exposure(lender, borrower, amount, recovery_rate) # type: ignore

    def narrate(self, msg: str) -> None:
        self._notes.append(f"[{self._minutes:04d}m] {msg}")

    # --------------------- Cue handlers ------------------------

    def _do_policy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a central-bank decision, update base curve, and emit a rates shock.
        payload keys:
          statement_text: str
          events: List[dict] (EconEvent-like)
          step_bps: 25 (optional)
        """
        if not self.macro:
            self.narrate("No macro state; skipping policy cue.")
            return {}
        evs = []
        for e in payload.get("events", []):
            evs.append(EconEvent(code=e.get("code",""), name=e.get("name",""), ts=datetime.utcnow(), # type: ignore
                                 actual=e.get("actual"), consensus=e.get("consensus"), unit=e.get("unit",""))) # type: ignore
        decision = decide_next_meeting(
            self.macro,
            statement_text=payload.get("statement_text",""),
            events=evs,
            hike_cut_step_bps=float(payload.get("step_bps", 25.0)),
            asof=datetime.utcnow(),
        )
        # simulate path & curve
        path = simulate_policy_path(self.macro, months=36, guidance_bps_3m=decision.guidance_bps_3m, seed=self.rng_seed) # type: ignore
        curve = path_to_yield_curve(asof=date.today(), currency=self.macro.currency, short_path=path,
                                    term_premium_bps_10y=getattr(self.macro, "term_premium_bps_10y", 60.0))
        shock = curve_vs_base_to_shock(self.base_curve or curve, curve)
        self.base_curve = curve
        self.narrate(f"Policy decision: Δ={decision.delta_bps} bps; probs={decision.probs}.") # type: ignore

        return {
            "decision": json.loads(decision_to_json(decision)),
            "curve_shock": getattr(shock, "custom_tenor_bp", getattr(shock, "get", lambda k, d=None: d)("custom_tenor_bp", {})),
        }

    def _do_shock(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply bank-level and/or asset-level shock and propagate contagion.
        payload:
          banks: [{id: 'A', equity_loss: 1e8, illiquid_haircut: 0.05}, ...]
          rounds: int
        """
        banks = payload.get("banks", [])
        for b in banks:
            try:
                self.contagion.apply_exogenous_shock( # type: ignore
                    id=b["id"],
                    equity_loss=float(b.get("equity_loss", 0.0)),
                    illiquid_haircut=float(b.get("illiquid_haircut", 0.0)),
                )
            except Exception:
                pass

        res = self.contagion.propagate(rounds=payload.get("rounds")) # type: ignore
        new_defs = res.get("system_defaults", [])
        if new_defs:
            self.narrate(f"Contagion defaults: {', '.join(new_defs)}.")
        return res

    def _do_halt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Halt symbols (e.g., news halts). payload: {symbols: ['RELIANCE.NS', ...]}
        """
        out = {"halted": []}
        if not self.gov:
            self.narrate("No governor; cannot set halts.")
            return out
        for s in payload.get("symbols", []):
            self.gov.set_symbol_halt(s, True)
            out["halted"].append(s)
        self.narrate(f"Halted: {', '.join(out['halted'])}" if out["halted"] else "No symbols to halt.")
        return out

    def _do_resume(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = {"resumed": []}
        if not self.gov:
            self.narrate("No governor; cannot clear halts.")
            return out
        for s in payload.get("symbols", []):
            self.gov.set_symbol_halt(s, False)
            out["resumed"].append(s)
        self.narrate(f"Resumed: {', '.join(out['resumed'])}" if out["resumed"] else "No symbols to resume.")
        return out

    def _do_kill(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Emulate strategy kill (e.g., from large DD). payload: {strategy: 'alpha.news'}
        """
        if not self.gov:
            self.narrate("No governor; cannot kill strategies.")
            return {}
        strat = payload.get("strategy", "")
        self.gov.set_policy(enabled=True)  # ensure live to flip a flag
        # hit the kill path by marking a big DD
        self.gov.ingest_mark(strategy=strat, pnl_delta=0.0, dd_frac=1.0)
        dec = self.gov.check_and_scale({"strategy": strat, "symbol": "XXX", "qty": 1, "price": 1.0, "side": "buy"})
        self.narrate(f"Kill request for {strat}: {dec.reason}.")
        return {"decision": asdict(dec) if hasattr(dec, "__dict__") else dec}

    # --------------------- Runner -------------------------------

    def play(self, scenes: Sequence[Scene], *, minutes: int, snapshot_fn: Optional[Callable[[Dict[str, Any]], None]] = None) -> Outcome:
        """
        Advance a virtual clock minute-by-minute, firing cues whose t_min matches.
        Collects a JSON-like snapshot per minute.
        """
        start = time.time()
        cue_map: Dict[int, List[Tuple[str, Dict[str, Any], str]]] = {}
        for sc in scenes:
            for c in sc.cues:
                cue_map.setdefault(int(c.t_min), []).append((c.action, c.payload, sc.name))

        for m in range(minutes + 1):
            self._minutes = m
            # fire cues
            results = []
            for (act, payload, sc_name) in cue_map.get(m, []):
                if act == "policy":
                    results.append({"scene": sc_name, "act": act, "result": self._do_policy(payload)})
                elif act == "shock":
                    results.append({"scene": sc_name, "act": act, "result": self._do_shock(payload)})
                elif act == "halt":
                    results.append({"scene": sc_name, "act": act, "result": self._do_halt(payload)})
                elif act == "resume":
                    results.append({"scene": sc_name, "act": act, "result": self._do_resume(payload)})
                elif act == "kill":
                    results.append({"scene": sc_name, "act": act, "result": self._do_kill(payload)})
                elif act == "narrate":
                    self.narrate(payload.get("text", ""))
                else:
                    self.narrate(f"Unknown act '{act}' in scene '{sc_name}'")

            # snapshot state for dashboards
            frame = self._snapshot(results=results)
            self._frames.append(frame)
            if snapshot_fn:
                try: snapshot_fn(frame)
                except Exception: pass

        end = time.time()
        return Outcome(started_at=start, finished_at=end, frames=self._frames, notes=self._notes)

    # --------------------- Snapshot -----------------------------

    def _snapshot(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        banks = getattr(self.contagion, "banks", {})
        bank_view = {
            bid: {
                "equity": float(getattr(b, "equity", 0.0)),
                "liq": float(getattr(b, "liquid_assets", 0.0)),
                "illq": float(getattr(b, "illiquid_assets", 0.0)),
                "dd": bool(getattr(b, "defaulted", False)),
                "cr": float(getattr(b, "capital_ratio")() if hasattr(b, "capital_ratio") else 0.0),
            }
            for bid, b in banks.items()
        }
        curve = self.base_curve
        curve_pts = []
        try:
            for p in curve.sorted_points(): # type: ignore
                curve_pts.append({"tenor_yrs": float(p.tenor_yrs), "yld": float(p.yld)})
        except Exception:
            pass

        return {
            "minute": self._minutes,
            "ts": int(time.time() * 1000),
            "banks": bank_view,
            "curve": curve_pts,
            "results": results,
        }

# -----------------------------------------------------------------------------
# Tiny demo
# -----------------------------------------------------------------------------

def _demo():
    # --- Macro setup (RBI-like) ---
    try:
        st = PolicyState(bank="RBI", currency="INR", policy_rate=0.065, r_star=0.01, # type: ignore
                         infl_target=0.04, inflation_yoy=0.051, output_gap=0.002, # type: ignore
                         term_premium_bps_10y=75.0) # type: ignore
    except Exception:
        st = None

    # --- Theatre ---
    theatre = CrisisTheatre(macro_state=st)

    # --- Banks & interbank links ---
    try:
        theatre.add_bank(Bank(id="A", name="Alpha", equity=100.0, liquid_assets=300.0, illiquid_assets=700.0, liabilities=800.0)) # type: ignore
        theatre.add_bank(Bank(id="B", name="Beta",  equity=80.0,  liquid_assets=200.0, illiquid_assets=500.0, liabilities=620.0)) # type: ignore
        theatre.add_bank(Bank(id="C", name="Gamma", equity=60.0,  liquid_assets=150.0, illiquid_assets=450.0, liabilities=540.0)) # type: ignore
        theatre.add_exposure("A","B",120.0, recovery_rate=0.5)
        theatre.add_exposure("B","C",100.0, recovery_rate=0.4)
        theatre.add_exposure("C","A",90.0,  recovery_rate=0.3)
    except Exception:
        pass

    # --- Scenes ---
    scenes = [
        Scene(
            name="Opening Bell",
            cues=[
                Cue(0, "narrate", {"text": "Normal markets; mild macro overheating."}),
                Cue(1, "policy", {
                    "statement_text": "Inflation remains elevated with upside risks; growth resilient. Further tightening may be warranted.",
                    "events": [
                        {"code":"IN_CPI_HEADLINE", "name":"India CPI YoY", "actual":5.5, "consensus":5.2, "unit":"%"},
                        {"code":"IN_IIP", "name":"India IIP YoY", "actual":4.0, "consensus":3.2, "unit":"%"},
                    ],
                    "step_bps": 25
                }),
            ],
        ),
        Scene(
            name="Liquidity Crunch",
            cues=[
                Cue(3, "shock", {"banks":[{"id":"B", "equity_loss":30.0, "illiquid_haircut":0.05}], "rounds": 4}),
                Cue(4, "halt", {"symbols": ["YESBANK.NS", "SBIN.NS"]}),
                Cue(6, "resume", {"symbols": ["SBIN.NS"]}),
                Cue(8, "kill", {"strategy": "alpha.arb"}),
            ],
        ),
    ]

    # --- Run for 10 minutes of sim-time ---
    out = theatre.play(scenes, minutes=10)
    print(json.dumps({
        "frames": out.frames[-3:],   # last few frames
        "notes": out.notes,
    }, indent=2))

if __name__ == "__main__":
    _demo()