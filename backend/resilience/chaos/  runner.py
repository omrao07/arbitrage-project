# backend/sim/runner.py
"""
Policy/Scenario Runner

What it does
------------
- Run the core PolicySimulator for N days, or execute a ScenarioSpec from JSON/YAML.
- Optional ShockEngine: build from a simple YAML and apply on-the-fly.
- Emit snapshots to JSON (list), JSONL (stream), or CSV.
- Optional live publisher (dotted path to a function), called each step with the signals map.

Examples
--------
# 1) Plain sim for 120 days, half-day steps, write JSON
python -m backend.sim.runner --days 120 --dt 0.5 --out runs/sim.json

# 2) Run a scenario file (json/yaml), stream JSONL, also publish to signal_bus.publish
python -m backend.sim.runner --scenario backend/config/scenarios/flash_crash.json \
       --jsonl runs/flash.jsonl --publish signal_bus:publish

# 3) Attach shock engine from YAML (see example at bottom)
python -m backend.sim.runner --days 60 --shocks configs/shocks.yaml --csv runs/sim.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Optional YAML
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# Project imports (guarded so this file can be imported standalone)
try:
    from .policy_sim import PolicySimConfig, PolicySimulator # type: ignore
except Exception as e:
    PolicySimConfig = None  # type: ignore
    PolicySimulator = None  # type: ignore

# Prefer the richer ScenarioRunner (with timeline blocks). Fallback to ScenarioLibrary if not present.
try:
    from .scenarios import ScenarioRunner, load_yaml as load_scen_yaml, ScenarioSpec  # type: ignore # rich API
    _SCEN_KIND = "runner"
except Exception:
    ScenarioRunner = None  # type: ignore
    ScenarioSpec = None    # type: ignore
    _SCEN_KIND = "lib"
    try:
        from .scenarios import ScenarioLibrary  # type: ignore # simpler presets API
    except Exception:
        ScenarioLibrary = None  # type: ignore

# Shock models (optional)
try:
    from .shock_models import ShockEngine, JumpDiffusion, HawkesLike, RegimeConditional, VolatilitySpike, LiquidityDrain, CrossAssetPropagator, apply_to_sim # type: ignore
    _HAVE_SHOCKS = True
except Exception:
    _HAVE_SHOCKS = False


# ----------------------- helpers -----------------------

def _die(msg: str, code: int = 2) -> None:
    print(f"[runner] {msg}", file=sys.stderr)
    sys.exit(code)

def _ensure(cond: bool, msg: str) -> None:
    if not cond:
        _die(msg)

def _ensure_dir(path: Optional[str]) -> None:
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_yaml(path: str) -> Any:
    if not _HAVE_YAML:
        _die("PyYAML not installed; cannot read YAML")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_scenario(path: str) -> Any:
    if path.endswith((".yaml", ".yml")):
        if _SCEN_KIND == "runner":
            return load_scen_yaml(path)[0]  # first scenario in file
        data = _load_yaml(path)  # library fallback expects different shape; not used here
        return data
    return _load_json(path)

def _resolve_publisher(dotted: str):
    """
    dotted can be 'module.sub:func' or 'module.sub.func'.
    Returns a callable f(payload: dict) -> None
    """
    if ":" in dotted:
        mod_name, func_name = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        mod_name, func_name = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name, None)
    _ensure(callable(fn), f"publisher '{dotted}' not found/callable")
    return fn

def _build_shock_engine_from_yaml(path: str):
    """
    YAML structure example (all fields optional):
    models:
      - kind: jump_diffusion
        key: fed
        lam: 0.02
        jump_bps: [25, 75]
      - kind: hawkes
        base: 0.004
        alpha: 0.5
        decay: 0.25
        risk_jump: 1.2
        liq_jump: -0.5
      - kind: regime_conditional
        per_regime:
          inflation: { fed: [25, 0.03] }
          crisis:    { fed: [-50, 0.05], ecb: [-25, 0.03] }
      - kind: vol_spike
        base: 0.01
        size: 0.8
        half_life: 4
      - kind: liq_drain
        fire_p: 0.006
        draw_size: -0.3
        recovery_rate: 0.03
      - kind: cross_asset

    limits:
      max_abs_rate_bps: 150
      max_abs_z_jump: 5.0
    """
    if not _HAVE_SHOCKS:
        _die("shock_models not available; cannot use --shocks")
    cfg = _load_yaml(path)
    models = []
    for m in (cfg.get("models") or []):
        kind = str(m.get("kind", "")).lower()
        if kind in ("jump", "jump_diffusion"):
            models.append(JumpDiffusion(
                key=m.get("key", "fed"),
                lam=float(m.get("lam", 0.01)),
                jump_bps=tuple(m.get("jump_bps", [25, 25])),
                normal=bool(m.get("normal", False)),
                bias=float(m.get("bias", 0.0)),
            ))
        elif kind in ("hawkes", "hawkes_like"):
            models.append(HawkesLike(
                base=float(m.get("base", 0.002)),
                alpha=float(m.get("alpha", 0.35)),
                decay=float(m.get("decay", 0.20)),
                key=m.get("key", "global"),
                risk_jump=float(m.get("risk_jump", 1.0)),
                liq_jump=float(m.get("liq_jump", -0.4)),
                infl_jump=float(m.get("infl_jump", 0.0)),
            ))
        elif kind in ("regime", "regime_conditional"):
            models.append(RegimeConditional(per_regime=m.get("per_regime", {})))
        elif kind in ("vol", "vol_spike", "volatility_spike"):
            models.append(VolatilitySpike(
                base=float(m.get("base", 0.002)),
                size=float(m.get("size", 1.0)),
                half_life=float(m.get("half_life", 3.0)),
                key=m.get("key", "global"),
            ))
        elif kind in ("liq", "liquidity_drain", "liq_drain"):
            models.append(LiquidityDrain(
                fire_p=float(m.get("fire_p", 0.004)),
                draw_size=float(m.get("draw_size", -0.4)),
                recovery_rate=float(m.get("recovery_rate", 0.05)),
                key=m.get("key", "global"),
            ))
        elif kind in ("cross", "cross_asset", "cross_asset_propagator"):
            models.append(CrossAssetPropagator())
        else:
            _die(f"unknown shock model kind '{kind}' in {path}")
    limits = cfg.get("limits") or {}
    return ShockEngine(
        models=models,
        max_abs_rate_bps=int(limits.get("max_abs_rate_bps", 150)),
        max_abs_z_jump=float(limits.get("max_abs_z_jump", 5.0)),
    )


# ----------------------- core run modes -----------------------

def run_plain(days: int, dt: float, seed: Optional[int],
              shocks_yaml: Optional[str], publisher: Optional[str],
              out_json: Optional[str], out_jsonl: Optional[str], out_csv: Optional[str]) -> None:
    _ensure(PolicySimConfig and PolicySimulator, "policy_sim not available") # type: ignore
    cfg = PolicySimConfig(seed=seed, dt_days=dt, horizon_days=days) # type: ignore
    sim = PolicySimulator(cfg) # type: ignore

    publisher_fn = _resolve_publisher(publisher) if publisher else None
    engine = _build_shock_engine_from_yaml(shocks_yaml) if shocks_yaml else None

    # Outputs
    snaps: List[Dict[str, Any]] = []
    _ensure_dir(out_json or out_jsonl or out_csv)

    jsonl_f = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
    csv_w = None
    if out_csv:
        csv_f = open(out_csv, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["t", "ts", "regime", "key", "value"])

    for _ in range(days):
        if engine:
            apply_to_sim(sim, engine)
        snap = sim.step()
        if publisher_fn:
            try:
                publisher_fn(snap["signals"])
            except Exception:
                pass

        if out_jsonl:
            jsonl_f.write(json.dumps(snap) + "\n") # type: ignore
        if out_csv and csv_w:
            s = snap["signals"]
            for k, v in s.items():
                csv_w.writerow([snap["t"], snap["ts"], snap["regime"], k, v])
        if out_json:
            snaps.append(snap)

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(snaps, f, indent=2)

    if jsonl_f:
        jsonl_f.close()
    if out_csv and 'csv_f' in locals():
        csv_f.close()


def run_scenario(path: str, days: Optional[int], dt: Optional[float], seed: Optional[int],
                 shocks_yaml: Optional[str], publisher: Optional[str],
                 out_json: Optional[str], out_jsonl: Optional[str], out_csv: Optional[str]) -> None:
    _ensure(PolicySimConfig, "policy_sim not available") # type: ignore
    _ensure_dir(out_json or out_jsonl or out_csv)
    publisher_fn = _resolve_publisher(publisher) if publisher else None
    engine = _build_shock_engine_from_yaml(shocks_yaml) if shocks_yaml else None

    if _SCEN_KIND == "runner":
        _ensure(ScenarioRunner is not None, "ScenarioRunner not available")
        cfg = PolicySimConfig(seed=seed, dt_days=(dt if dt is not None else 1.0),
                              horizon_days=(days if days is not None else 120)) # type: ignore
        runner = ScenarioRunner(cfg) # type: ignore
        spec = _load_scenario(path)
        snaps = runner.run(spec)
        # stream outputs and/or publish
        jsonl_f = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
        csv_w = None
        if out_csv:
            csv_f = open(out_csv, "w", newline="", encoding="utf-8")
            csv_w = csv.writer(csv_f)
            csv_w.writerow(["t", "ts", "regime", "key", "value"])

        out_accum = []
        for snap in snaps:
            # If engine present, you can optionally post-process (less common for scenario runs)
            if publisher_fn:
                try:
                    publisher_fn(snap["signals"])
                except Exception:
                    pass
            if jsonl_f:
                jsonl_f.write(json.dumps(snap) + "\n")
            if csv_w:
                for k, v in snap["signals"].items():
                    csv_w.writerow([snap["t"], snap["ts"], snap["regime"], k, v])
            if out_json:
                out_accum.append(snap)

        if out_json:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(out_accum, f, indent=2)
        if jsonl_f:
            jsonl_f.close()
        if csv_w:
            csv_f.close()

    else:
        _ensure(ScenarioLibrary is not None and PolicySimulator is not None, "ScenarioLibrary not available")
        lib = ScenarioLibrary() # type: ignore
        _ensure(path in lib.list(), f"scenario '{path}' not found in library presets {lib.list()}")
        sim = lib.build(path, horizon_days=(days or 120), seed=seed)
        # stream loop like plain run
        snaps = []
        jsonl_f = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
        csv_w = None
        if out_csv:
            csv_f = open(out_csv, "w", newline="", encoding="utf-8")
            csv_w = csv.writer(csv_f)
            csv_w.writerow(["t", "ts", "regime", "key", "value"])

        for _ in range(sim.cfg.horizon_days):
            if engine:
                apply_to_sim(sim, engine)
            snap = sim.step()
            if publisher_fn:
                try:
                    publisher_fn(snap["signals"])
                except Exception:
                    pass
            if jsonl_f:
                jsonl_f.write(json.dumps(snap) + "\n")
            if csv_w:
                for k, v in snap["signals"].items():
                    csv_w.writerow([snap["t"], snap["ts"], snap["regime"], k, v])
            if out_json:
                snaps.append(snap)

        if out_json:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(snaps, f, indent=2)
        if jsonl_f:
            jsonl_f.close()
        if csv_w:
            csv_f.close()


# ----------------------- CLI -----------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simrunner", description="Policy/Scenario Runner")
    p.add_argument("--days", type=int, help="Horizon in days (overrides scenario horizon if provided)")
    p.add_argument("--dt", type=float, help="Step size in days (e.g., 0.5 for 12h)")
    p.add_argument("--seed", type=int, help="RNG seed")
    p.add_argument("--scenario", help="Path to ScenarioSpec JSON/YAML (or preset name with ScenarioLibrary fallback)")
    p.add_argument("--shocks", help="YAML file describing ShockEngine models")
    p.add_argument("--out", help="Write full JSON array of snapshots")
    p.add_argument("--jsonl", help="Write line-delimited JSON snapshots")
    p.add_argument("--csv", help="Write flat CSV of (t,ts,regime,key,value)")
    p.add_argument("--publish", help="Dotted path to publisher function (e.g., 'signal_bus:publish')")
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.scenario:
        run_scenario(
            path=args.scenario,
            days=args.days, dt=args.dt, seed=args.seed,
            shocks_yaml=args.shocks, publisher=args.publish,
            out_json=args.out, out_jsonl=args.jsonl, out_csv=args.csv
        )
    else:
        _ensure(args.days is not None, "Provide --days for a plain simulation (or --scenario)")
        run_plain(
            days=args.days, dt=(args.dt or 1.0), seed=args.seed,
            shocks_yaml=args.shocks, publisher=args.publish,
            out_json=args.out, out_jsonl=args.jsonl, out_csv=args.csv
        )

if __name__ == "__main__":
    main()