# backend/cli/drivers.py
"""
Unified CLI drivers for the arbitrage project.

Subcommands
-----------
signals        : pull + mix signals once (via SignalsAdapter) and optionally persist
manager        : run Manager one step (or loop)
schedule       : start lightweight scheduler and register jobs (manager step, signals pull)
simulate       : run policy simulator (policy_sim.py) and dump snapshots
scenario       : run a named scenario (scenarios.py) and dump snapshots
pipeline       : run a named built-in pipeline (pipelines.py builders)

Examples
--------
python -m backend.cli.drivers signals --yaml configs/sentiment.yaml --yaml configs/altdata.yaml --out runs/signals_latest.json
python -m backend.cli.drivers manager --once --prices configs/examples/prices.json --save
python -m backend.cli.drivers schedule --manager-every 10 --signals-every 5 --signals-json runs/signals_latest.json
python -m backend.cli.drivers simulate --days 90 --dt 0.5 --out runs/sim.json
python -m backend.cli.drivers scenario --name stagflation --days 180 --out runs/scenario_stag.json
python -m backend.cli.drivers pipeline --name signals --out runs/signals_latest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Optional YAML
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# --- Project imports (lightweight; guarded) ---
def _die(msg: str, code: int = 2):
    print(f"[drivers] {msg}", file=sys.stderr)
    sys.exit(code)

def _ensure(cond: bool, msg: str):
    if not cond:
        _die(msg)

# Signals stack
try:
    from backend.io_utils.signals_adapter import SignalsAdapter, SourceSpec, TransformSpec, MixerHook # type: ignore
except Exception as e:
    SignalsAdapter = None  # type: ignore
    SourceSpec = None      # type: ignore
    TransformSpec = None   # type: ignore
    MixerHook = None       # type: ignore

# Mixer config (optional)
try:
    from backend.common.mixer import MixerConfig # type: ignore
except Exception:
    class MixerConfig:  # type: ignore
        def __init__(self, **kw): pass

# Pipelines / Manager / Scheduler
try:
    from backend.runtime.pipelines import build_signals_pipeline, build_manager_step_pipeline # type: ignore
except Exception:
    build_signals_pipeline = None   # type: ignore
    build_manager_step_pipeline = None  # type: ignore

try:
    from backend.runtime.manager import Manager, StaticPrices, StaticSignals # type: ignore
except Exception:
    Manager = None         # type: ignore
    StaticPrices = None    # type: ignore
    StaticSignals = None   # type: ignore

try:
    from backend.runtime.schedule import Scheduler, Job, scheduled # type: ignore
except Exception:
    Scheduler = None  # type: ignore
    Job = None        # type: ignore
    def scheduled(**kw):  # type: ignore
        def _wrap(fn): return fn
        return _wrap

# Simulators
try:
    from backend.sim.policy_sim import PolicySimConfig, PolicySimulator # type: ignore
except Exception:
    PolicySimConfig = None  # type: ignore
    PolicySimulator = None  # type: ignore

try:
    # you have two scenarios.py options; prefer the richer one if present
    from backend.sim.scenarios import ScenarioRunner, presets, ScenarioSpec  # type: ignore # richer version
    _SCEN_KIND = "runner"
except Exception:
    ScenarioRunner = None  # type: ignore
    presets = None         # type: ignore
    ScenarioSpec = None    # type: ignore
    _SCEN_KIND = "lib"
    try:
        from backend.sim.scenarios import ScenarioLibrary  # type: ignore # simpler library
    except Exception:
        ScenarioLibrary = None  # type: ignore


# ----------------------- helpers -----------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _maybe_yaml_dict(path: str) -> Dict[str, Any]:
    if not _HAVE_YAML:
        _die("PyYAML not installed; cannot read YAML")
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc or {}

def _ensure_dirs(path: Optional[str]):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _print_sample_map(name: str, m: Dict[str, Any], n: int = 10):
    items = list(m.items())[:n]
    print(f"[{name}] sample {min(n, len(items))}/{len(m)}:", items)

# ----------------------- command: signals -----------------------

def cmd_signals(args: argparse.Namespace) -> None:
    _ensure(SignalsAdapter is not None, "signals_adapter not available")
    # Build sources from yaml/json files and inline dicts
    sources = []
    transforms = {}

    # Each --yaml is a file with a dict of signals
    for y in args.yaml or []:
        sources.append(SourceSpec(name=os.path.basename(y), kind="yaml", path=y, debounce_sec=0.0)) # type: ignore
    # Each --json is a file with dict
    for j in args.json or []:
        sources.append(SourceSpec(name=os.path.basename(j), kind="json", path=j, debounce_sec=0.0)) # type: ignore
    # Inline map
    if args.inline:
        try:
            payload = json.loads(args.inline)
        except Exception:
            _die("--inline must be a JSON object string")
        sources.append(SourceSpec(name="inline", kind="dict", payload=payload)) # type: ignore

    # Default mixer (equal weights)
    weights = {s.name: 1.0 for s in sources}
    adapter = SignalsAdapter(
        sources=sources,
        transforms=transforms,
        mixer=MixerHook(MixerConfig(mode="linear", weights=weights, norm="z", clip=(-3, 3))), # type: ignore
        publisher=None,
    ) # type: ignore
    snap = adapter.snapshot(with_raw=args.verbose)
    sigs = snap["signals"]
    _print_sample_map("signals", sigs, 12)

    if args.out:
        _ensure_dirs(args.out)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(sigs, f, indent=2, sort_keys=True)
        print("[signals] wrote", args.out)

# ----------------------- command: manager -----------------------

def _prices_from_src(src: Optional[str]) -> Dict[str, float]:
    if not src:
        return {}
    if src.endswith(".json"):
        return _load_json(src)
    if src.endswith(".yaml") or src.endswith(".yml"):
        return _maybe_yaml_dict(src)
    return {}

def _signals_from_src(srcs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in srcs:
        if p.endswith(".json"):
            out.update(_load_json(p))
        elif p.endswith(".yaml") or p.endswith(".yml"):
            out.update(_maybe_yaml_dict(p))
    return out

def cmd_manager(args: argparse.Namespace) -> None:
    _ensure(Manager is not None, "runtime.manager not available")
    prices = StaticPrices(_prices_from_src(args.prices)) if StaticPrices else None
    sigs = _signals_from_src(args.signals or [])
    sigprov = StaticSignals(sigs) if StaticSignals else None

    agents = []
    if args.agents:
        # Lazy import by name; expects modules in agents/ with class <Name>Agent
        for name in args.agents:
            mod = __import__(f"agents.{name}", fromlist=[None]) # type: ignore
            AgentCls = None
            # Try exact NameAgent and TitlecaseAgent
            for cand in [f"{name.capitalize()}Agent", f"{name}Agent", "Agent"]:
                AgentCls = getattr(mod, cand, None)
                if AgentCls: break
            _ensure(AgentCls is not None, f"could not find agent class in agents.{name}")
            agents.append(AgentCls()) # type: ignore

    mgr = Manager(price_provider=prices, signal_provider=sigprov, agents=agents) # type: ignore
    if args.once:
        res = mgr.run_once(do_execute=args.execute)
        if args.save and res.get("paths"):
            print("[manager] artifacts:", res["paths"])
    else:
        print("[manager] starting loop (CTRL+C to stop)")
        mgr.run_loop(do_execute=args.execute)

# ----------------------- command: schedule -----------------------

def cmd_schedule(args: argparse.Namespace) -> None:
    _ensure(Scheduler is not None and Manager is not None, "scheduler/manager not available")
    # Build manager (static sources for now)
    prices = StaticPrices(_prices_from_src(args.prices)) if StaticPrices else None
    sigprov = StaticSignals(_signals_from_src(args.signals or [])) if StaticSignals else None
    agents = []
    if args.agents:
        for name in args.agents:
            mod = __import__(f"agents.{name}", fromlist=[None]) # type: ignore
            AgentCls = getattr(mod, f"{name.capitalize()}Agent", None) or getattr(mod, f"{name}Agent", None)
            _ensure(AgentCls is not None, f"could not find agent class in agents.{name}")
            agents.append(AgentCls()) # type: ignore
    mgr = Manager(price_provider=prices, signal_provider=sigprov, agents=agents) # type: ignore

    sch = Scheduler(persist_path=args.state) # type: ignore
    sch.set_context(manager=mgr) # type: ignore

    def _run_mgr(ctx):
        ctx["manager"].run_once(do_execute=args.execute)

    # Register jobs
    if args.manager_every:
        sch.add(Job(name="manager_step", fn=_run_mgr, every_sec=float(args.manager_every), jitter_sec=0.3)) # type: ignore
    if args.signals_every and (args.signals_json or args.signals_yaml):
        _ensure(build_signals_pipeline is not None, "pipelines not available for signals job")
        # Build a small adapter via files
        sources = []
        if args.signals_json:
            for p in args.signals_json:
                sources.append(SourceSpec(name=os.path.basename(p), kind="json", path=p)) # type: ignore
        if args.signals_yaml:
            for p in args.signals_yaml:
                sources.append(SourceSpec(name=os.path.basename(p), kind="yaml", path=p)) # type: ignore
        adapter = SignalsAdapter(
            sources=sources,
            transforms={},
            mixer=MixerHook(MixerConfig(mode="linear", weights={s.name:1 for s in sources}, norm="z")), # type: ignore
            publisher=None
        ) # type: ignore
        pipe = build_signals_pipeline(adapter=adapter, out_json=args.signals_out or "runs/signals_latest.json", verbose=False) # type: ignore
        sch.set_context(signals_pipe=pipe) # type: ignore

        def _run_signals(ctx):
            ctx["signals_pipe"].run(context={"env": "sched"})

        sch.add(Job(name="signals_pull", fn=_run_signals, every_sec=float(args.signals_every), jitter_sec=0.5)) # type: ignore

    print("[schedule] starting (CTRL+C to stop)")
    sch.start(daemon=False) # type: ignore

# ----------------------- command: simulate -----------------------

def cmd_simulate(args: argparse.Namespace) -> None:
    _ensure(PolicySimConfig is not None and PolicySimulator is not None, "simulator not available")
    cfg = PolicySimConfig(seed=args.seed, dt_days=float(args.dt), horizon_days=int(args.days)) # type: ignore
    sim = PolicySimulator(cfg) # type: ignore
    snaps = sim.run() # type: ignore
    if args.out:
        _ensure_dirs(args.out)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(snaps, f, indent=2)
        print("[simulate] wrote", args.out)
    else:
        print("[simulate] ran", len(snaps), "steps; last regime:", snaps[-1]["regime"])
        _print_sample_map("signals", snaps[-1]["signals"], 12)

# ----------------------- command: scenario -----------------------

def cmd_scenario(args: argparse.Namespace) -> None:
    if _SCEN_KIND == "runner":
        _ensure(ScenarioRunner is not None, "ScenarioRunner not available")
        cfg = PolicySimConfig(seed=args.seed, dt_days=float(args.dt), horizon_days=int(args.days)) # type: ignore
        runner = ScenarioRunner(cfg) # type: ignore
        # choose preset or ad-hoc
        if args.name:
            preset = getattr(presets, args.name, None)
            _ensure(preset is not None, f"unknown preset '{args.name}'")
            spec = preset(horizon_days=int(args.days)) if callable(preset) else preset
        else:
            # minimal inline scenario: single block in a chosen regime
            spec = ScenarioSpec(name="inline", overrides={"horizon_days": int(args.days)},
                                timeline=[{"steps": int(args.days), "regime": args.regime or "expansion"}]) # type: ignore
        snaps = runner.run(spec)
    else:
        _ensure(ScenarioLibrary is not None and PolicySimulator is not None, "ScenarioLibrary not available")
        lib = ScenarioLibrary() # type: ignore
        _ensure(args.name in lib.list(), f"unknown scenario '{args.name}', options={lib.list()}")
        sim = lib.build(args.name, horizon_days=int(args.days), seed=args.seed)
        snaps = sim.run()

    if args.out:
        _ensure_dirs(args.out)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(snaps, f, indent=2)
        print("[scenario] wrote", args.out)
    else:
        print("[scenario] ran", len(snaps), "steps; last regime:", snaps[-1]["regime"])
        _print_sample_map("signals", snaps[-1]["signals"], 12)

# ----------------------- command: pipeline -----------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    _ensure(build_signals_pipeline is not None, "pipelines not available")
    _ensure(SignalsAdapter is not None, "signals_adapter not available")

    # For now only "signals" pipeline is built-in
    if args.name != "signals":
        _die("only 'signals' pipeline is supported for now")

    sources = []
    if args.yaml:
        for y in args.yaml:
            sources.append(SourceSpec(name=os.path.basename(y), kind="yaml", path=y)) # type: ignore
    if args.json:
        for j in args.json:
            sources.append(SourceSpec(name=os.path.basename(j), kind="json", path=j)) # type: ignore
    _ensure(len(sources) > 0, "provide at least one --yaml/--json for signals pipeline")

    adapter = SignalsAdapter(
        sources=sources,
        transforms={},
        mixer=MixerHook(MixerConfig(mode="linear", weights={s.name:1 for s in sources}, norm="z")), # type: ignore
        publisher=None
    ) # type: ignore

    pipe = build_signals_pipeline(adapter=adapter, out_json=args.out, verbose=bool(args.verbose)) # type: ignore
    res = pipe.run(context={"env":"cli"})
    print("[pipeline] ok:", res["ok"], "critical(ms):", int(res["critical_ms"]))
    if args.out:
        print("[pipeline] wrote", args.out)

# ----------------------- main -----------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drivers", description="Arbitrage project CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # signals
    ps = sub.add_parser("signals", help="Pull+mix signals once")
    ps.add_argument("--yaml", action="append", help="YAML file with dict signals", default=[])
    ps.add_argument("--json", action="append", help="JSON file with dict signals", default=[])
    ps.add_argument("--inline", help='Inline JSON map, e.g. \'{"risk_z":0.2}\'')
    ps.add_argument("--out", help="Write merged signals JSON")
    ps.add_argument("--verbose", action="store_true")
    ps.set_defaults(func=cmd_signals)

    # manager
    pm = sub.add_parser("manager", help="Run Manager step or loop")
    pm.add_argument("--prices", help="JSON/YAML prices map")
    pm.add_argument("--signals", action="append", default=[], help="JSON/YAML signals map (repeatable)")
    pm.add_argument("--agents", action="append", default=[], help="Agent module names under agents/ (e.g., crypto equities fx buffett druck dalio)")
    pm.add_argument("--once", action="store_true", help="Run a single step")
    pm.add_argument("--execute", action="store_true", help="Ask coordinator to execute")
    pm.add_argument("--save", action="store_true", help="Persist reports (Manager cfg.save_reports=true)")
    pm.set_defaults(func=cmd_manager)

    # schedule
    pc = sub.add_parser("schedule", help="Start lightweight scheduler")
    pc.add_argument("--state", default="runs/scheduler.json", help="State file path")
    pc.add_argument("--prices", help="JSON/YAML prices map")
    pc.add_argument("--signals", action="append", default=[], help="JSON/YAML signals map")
    pc.add_argument("--agents", action="append", default=[], help="Agent modules")
    pc.add_argument("--execute", action="store_true")
    pc.add_argument("--manager-every", type=float, help="Seconds between manager steps")
    pc.add_argument("--signals-every", type=float, help="Seconds between signals pulls")
    pc.add_argument("--signals-json", action="append", default=[], help="Signals job: input JSON files")
    pc.add_argument("--signals-yaml", action="append", default=[], help="Signals job: input YAML files")
    pc.add_argument("--signals-out", help="Signals job: write merged JSON")
    pc.set_defaults(func=cmd_schedule)

    # simulate
    psim = sub.add_parser("simulate", help="Run policy simulator")
    psim.add_argument("--days", type=int, default=120)
    psim.add_argument("--dt", type=float, default=1.0, help="Step size in days (e.g., 0.5)")
    psim.add_argument("--seed", type=int, default=None)
    psim.add_argument("--out", help="Write snapshots JSON")
    psim.set_defaults(func=cmd_simulate)

    # scenario
    psc = sub.add_parser("scenario", help="Run a named scenario")
    psc.add_argument("--name", help="Preset scenario name (e.g., soft_landing, stagflation, hard_landing)")
    psc.add_argument("--regime", help="If no preset name, force a single-regime block (expansion/slowdown/inflation/crisis)")
    psc.add_argument("--days", type=int, default=120)
    psc.add_argument("--dt", type=float, default=1.0)
    psc.add_argument("--seed", type=int, default=None)
    psc.add_argument("--out", help="Write snapshots JSON")
    psc.set_defaults(func=cmd_scenario)

    # pipeline
    ppl = sub.add_parser("pipeline", help="Run a built-in pipeline")
    ppl.add_argument("--name", required=True, help="Pipeline name (only 'signals' supported)")
    ppl.add_argument("--yaml", action="append", default=[], help="Signals pipeline input YAML files")
    ppl.add_argument("--json", action="append", default=[], help="Signals pipeline input JSON files")
    ppl.add_argument("--out", help="Persist merged signals JSON")
    ppl.add_argument("--verbose", action="store_true")
    ppl.set_defaults(func=cmd_pipeline)

    return p

def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()