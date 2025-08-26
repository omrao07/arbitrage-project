# backend/tests/stress.py
"""
Stress & Chaos Harness for the arbitrage project (stdlib-only; YAML optional)

What it can do
--------------
- Load-Test: call Manager.run_once() N times with optional parallel workers
- Scenario Replayer: feed signals from sim/scenario per step and run manager
- Shock Engine: optional YAML to attach probabilistic shocks during runs
- Chaos Faults: latency injection, dropped signals, exceptions, value jitter
- Metrics: latency percentiles, error rates, drawdown stats on a toy PnL,
          memory usage deltas (tracemalloc), simple throughput (iter/sec)
- Reports: JSON summary + (optional) CSV with per-iteration observations

Usage
-----
python -m backend.tests.stress --mode load --iters 500 --concurrency 1 --out runs/stress.json
python -m backend.tests.stress --mode scenario --scenario backend/config/scenarios/flash_crash.json \
       --iters 60 --out runs/scenario_stress.json --csv runs/scenario_obs.csv --shocks configs/shocks.yaml

You can also import and call StressHarness programmatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics as stats
import threading
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from queue import Queue, Empty
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Optional YAML
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# --- Project imports (guarded) ---
try:
    from backend.runtime.manager import Manager, StaticPrices, StaticSignals # type: ignore
except Exception:
    Manager = None          # type: ignore
    StaticPrices = None     # type: ignore
    StaticSignals = None    # type: ignore

try:
    from backend.sim.policy_sim import PolicySimConfig, PolicySimulator # type: ignore
except Exception:
    PolicySimConfig = None  # type: ignore
    PolicySimulator = None  # type: ignore

# Prefer the richer timeline runner if present
try:
    from backend.sim.scenarios import ScenarioRunner, load_yaml as scen_load_yaml  # type: ignore
    _HAVE_SCEN_RUNNER = True
except Exception:
    _HAVE_SCEN_RUNNER = False
    try:
        from backend.sim.scenarios import ScenarioLibrary  # type: ignore # fallback presets
    except Exception:
        ScenarioLibrary = None  # type: ignore

# Shock Engine (optional)
try:
    from backend.sim.shock_models import ShockEngine, JumpDiffusion, HawkesLike, RegimeConditional, VolatilitySpike, LiquidityDrain, CrossAssetPropagator, apply_to_sim # type: ignore
    _HAVE_SHOCKS = True
except Exception:
    _HAVE_SHOCKS = False


# -------------------------- Data Models ---------------------------------

@dataclass
class ChaosConfig:
    """Fault injection knobs (all optional)."""
    sleep_ms: Tuple[int, int] = (0, 0)          # random extra latency before manager.run_once
    drop_signals_p: float = 0.0                 # prob to clear signals map for a step
    jitter_pct: float = 0.0                     # +/- percentage jitter to apply to numeric signals
    raise_p: float = 0.0                        # prob to raise an artificial exception before run
    clip_signals: Optional[Tuple[float, float]] = None  # clamp after jitter

@dataclass
class StressConfig:
    mode: str = "load"                          # "load" | "scenario"
    iters: int = 100
    concurrency: int = 1
    seed: Optional[int] = None
    # Manager wiring (if you don't pass a live Manager)
    prices_path: Optional[str] = None
    signals_paths: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    # Scenario/sim options
    scenario_path: Optional[str] = None         # JSON/YAML ScenarioSpec
    sim_days: int = 120
    sim_dt: float = 1.0
    # Shocks
    shocks_yaml: Optional[str] = None
    # Chaos
    chaos: ChaosConfig = field(default_factory=ChaosConfig)
    # Outputs
    out_json: Optional[str] = None
    out_csv: Optional[str] = None


@dataclass
class IterObs:
    t0: float
    t1: float
    ok: bool
    err: Optional[str] = None
    pnl: Optional[float] = None                 # toy PnL delta if reported by manager
    dd: Optional[float] = None                  # running drawdown
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def ms(self) -> float:
        return (self.t1 - self.t0) * 1000.0


# ---------------------------- Helpers -----------------------------------

def _now() -> float:
    return time.perf_counter()

def _ensure_dir(path: Optional[str]) -> None:
    if not path: return
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_yaml(path: str) -> Any:
    if not _HAVE_YAML:
        raise RuntimeError("PyYAML not installed; cannot read YAML")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _signals_from_files(paths: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in paths or []:
        if p.endswith(".json"):
            out.update(_load_json(p))
        elif p.endswith((".yaml", ".yml")):
            out.update(_load_yaml(p) or {})
    return out

def _try(fn: Callable[[], Any]) -> Tuple[bool, Optional[str]]:
    try:
        fn()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------- Chaos -------------------------------------

def apply_chaos(sig: Dict[str, Any], ch: ChaosConfig) -> Dict[str, Any]:
    """Return a mutated copy of signals based on ChaosConfig."""
    if ch.drop_signals_p > 0 and random.random() < ch.drop_signals_p:
        return {}
    out: Dict[str, Any] = dict(sig)
    if ch.jitter_pct and ch.jitter_pct > 0:
        for k, v in list(out.items()):
            try:
                x = float(v)
            except Exception:
                continue
            jitter = 1.0 + random.uniform(-ch.jitter_pct, ch.jitter_pct)
            x *= jitter
            if ch.clip_signals:
                lo, hi = ch.clip_signals
                x = max(lo, min(hi, x))
            out[k] = x
    return out


# ---------------------------- Stress Harness ----------------------------

class StressHarness:
    def __init__(self, cfg: StressConfig, manager: Optional[Any] = None):
        self.cfg = cfg
        self.manager = manager or self._build_manager()
        self.engine = self._build_shock_engine(cfg.shocks_yaml) if cfg.shocks_yaml else None
        self._pnl = 0.0
        self._peak = 0.0
        if cfg.seed is not None:
            random.seed(cfg.seed)

    # --- wiring ---

    def _build_manager(self):
        if Manager is None:
            raise RuntimeError("backend.runtime.manager.Manager not available")
        prices = StaticPrices(_load_yaml(self.cfg.prices_path) if self.cfg.prices_path and self.cfg.prices_path.endswith((".yaml",".yml"))
                              else (_load_json(self.cfg.prices_path) if self.cfg.prices_path else {})) if StaticPrices else None
        sigs = _signals_from_files(self.cfg.signals_paths)
        sigprov = StaticSignals(sigs) if StaticSignals else None

        agents = []
        for name in self.cfg.agents or []:
            mod = __import__(f"agents.{name}", fromlist=[None]) # type: ignore
            AgentCls = getattr(mod, f"{name.capitalize()}Agent", None) or getattr(mod, f"{name}Agent", None) or getattr(mod, "Agent", None)
            if not AgentCls:
                raise RuntimeError(f"could not find agent class in agents.{name}")
            agents.append(AgentCls())
        return Manager(price_provider=prices, signal_provider=sigprov, agents=agents)

    def _build_shock_engine(self, path: str):
        if not _HAVE_SHOCKS:
            raise RuntimeError("shock_models not available; cannot use shocks YAML")
        cfg = _load_yaml(path)
        models = []
        for m in (cfg.get("models") or []):
            kind = str(m.get("kind", "")).lower()
            if kind in ("jump", "jump_diffusion"):
                models.append(JumpDiffusion(key=m.get("key", "fed"),
                                            lam=float(m.get("lam", 0.01)),
                                            jump_bps=tuple(m.get("jump_bps", [25, 25])),
                                            normal=bool(m.get("normal", False)),
                                            bias=float(m.get("bias", 0.0))))
            elif kind in ("hawkes", "hawkes_like"):
                models.append(HawkesLike(base=float(m.get("base", 0.002)),
                                        alpha=float(m.get("alpha", 0.35)),
                                        decay=float(m.get("decay", 0.20)),
                                        key=m.get("key", "global"),
                                        risk_jump=float(m.get("risk_jump", 1.0)),
                                        liq_jump=float(m.get("liq_jump", -0.4))))
            elif kind in ("regime", "regime_conditional"):
                models.append(RegimeConditional(per_regime=m.get("per_regime", {})))
            elif kind in ("vol", "vol_spike", "volatility_spike"):
                models.append(VolatilitySpike(base=float(m.get("base", 0.002)),
                                              size=float(m.get("size", 1.0)),
                                              half_life=float(m.get("half_life", 3.0))))
            elif kind in ("liq", "liquidity_drain", "liq_drain"):
                models.append(LiquidityDrain(fire_p=float(m.get("fire_p", 0.004)),
                                             draw_size=float(m.get("draw_size", -0.3)),
                                             recovery_rate=float(m.get("recovery_rate", 0.05))))
            elif kind in ("cross", "cross_asset", "cross_asset_propagator"):
                models.append(CrossAssetPropagator())
        limits = cfg.get("limits") or {}
        return ShockEngine(models=models,
                           max_abs_rate_bps=int(limits.get("max_abs_rate_bps", 150)),
                           max_abs_z_jump=float(limits.get("max_abs_z_jump", 5.0)))

    # --- scenarios/sim source ---

    def _scenario_signals_iter(self) -> Iterable[Dict[str, float]]:
        """Yield signal maps per step from a scenario or a plain sim."""
        # Scenario runner path
        if self.cfg.scenario_path:
            if _HAVE_SCEN_RUNNER:
                base = PolicySimConfig(seed=self.cfg.seed, dt_days=self.cfg.sim_dt, horizon_days=self.cfg.sim_days) # type: ignore
                runner = ScenarioRunner(base)
                doc = self._load_scenario(self.cfg.scenario_path)
                snaps = runner.run(doc)
                for s in snaps:
                    yield s["signals"]
            else:
                # Fallback: treat scenario_path as preset name via ScenarioLibrary
                if ScenarioLibrary is None or PolicySimulator is None:
                    raise RuntimeError("Scenario support not available")
                lib = ScenarioLibrary()
                sim = lib.build(self.cfg.scenario_path, horizon_days=self.cfg.sim_days, seed=self.cfg.seed)
                for _ in range(sim.cfg.horizon_days):
                    if self.engine:
                        apply_to_sim(sim, self.engine)
                    yield sim.step()["signals"]
        else:
            # Plain simulator as generator
            if PolicySimConfig is None or PolicySimulator is None:
                # no sim; just loop with empty signals
                for _ in range(self.cfg.sim_days):
                    yield {}
            else:
                cfg = PolicySimConfig(seed=self.cfg.seed, dt_days=self.cfg.sim_dt, horizon_days=self.cfg.sim_days)
                sim = PolicySimulator(cfg) # type: ignore
                for _ in range(cfg.horizon_days):
                    if self.engine:
                        apply_to_sim(sim, self.engine)
                    yield sim.step()["signals"] # type: ignore

    def _load_scenario(self, path: str):
        if path.endswith((".yaml", ".yml")):
            if not _HAVE_YAML:
                raise RuntimeError("PyYAML not installed; cannot load YAML scenario")
            return scen_load_yaml(path)[0]
        return _load_json(path)

    # --- PnL helpers (toy) ---

    def _update_drawdown(self, pnl_delta: float) -> float:
        self._pnl += (pnl_delta or 0.0)
        self._peak = max(self._peak, self._pnl)
        dd = (self._peak - self._pnl)
        return dd

    # --- run modes ---

    def run_load(self) -> Dict[str, Any]:
        """Run Manager.run_once() iters times with optional concurrency and chaos."""
        cfg = self.cfg
        iters = max(1, int(cfg.iters))
        conc = max(1, int(cfg.concurrency))

        obs: List[IterObs] = []
        q: "Queue[int]" = Queue()
        for i in range(iters): q.put(i)

        lock = threading.Lock()

        def worker():
            while True:
                try:
                    _ = q.get_nowait()
                except Empty:
                    return
                if cfg.chaos.sleep_ms[1] > 0:
                    time.sleep(random.uniform(cfg.chaos.sleep_ms[0], cfg.chaos.sleep_ms[1]) / 1000.0)
                if cfg.chaos.raise_p > 0 and random.random() < cfg.chaos.raise_p:
                    # record artificial failure
                    with lock:
                        t0 = _now(); t1 = _now()
                        obs.append(IterObs(t0=t0, t1=t1, ok=False, err="ChaosInjectedError: synthetic"))
                    continue

                # mutate the signal provider if available
                try:
                    base = self.manager.signal_provider.signals  # type: ignore[attr-defined]
                    mutated = apply_chaos(base, cfg.chaos)
                    self.manager.signal_provider.signals = mutated  # type: ignore[attr-defined]
                except Exception:
                    pass

                t0 = _now()
                ok, err = _try(lambda: self.manager.run_once(do_execute=False))
                t1 = _now()

                # extract toy PnL delta if manager returns any metric
                pnl_delta = None
                try:
                    res = getattr(self.manager, "last_result", None)
                    if isinstance(res, dict):
                        pnl_delta = float(res.get("pnl_delta", 0.0))
                except Exception:
                    pass

                dd = self._update_drawdown(pnl_delta or 0.0)
                with lock:
                    obs.append(IterObs(t0=t0, t1=t1, ok=ok, err=err, pnl=pnl_delta, dd=dd))
                q.task_done()

        tracemalloc.start()
        t_start = _now()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(conc)]
        for t in threads: t.start()
        for t in threads: t.join()

        t_end = _now()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return self._summarize(obs, wall_s=(t_end - t_start), mem_cur=cur, mem_peak=peak)

    def run_scenario(self) -> Dict[str, Any]:
        """Replay a scenario/sim stream; call run_once each step."""
        cfg = self.cfg
        obs: List[IterObs] = []
        tracemalloc.start()
        t_start = _now()

        for i, sig in enumerate(self._scenario_signals_iter()):
            if i >= cfg.iters:
                break
            # chaos mutate scenario signals, install into manager
            sig2 = apply_chaos(sig, cfg.chaos)
            try:
                self.manager.signal_provider.signals = sig2  # type: ignore[attr-defined]
            except Exception:
                pass

            if cfg.chaos.sleep_ms[1] > 0:
                time.sleep(random.uniform(cfg.chaos.sleep_ms[0], cfg.chaos.sleep_ms[1]) / 1000.0)

            t0 = _now()
            ok, err = _try(lambda: self.manager.run_once(do_execute=False))
            t1 = _now()

            pnl_delta = None
            try:
                res = getattr(self.manager, "last_result", None)
                if isinstance(res, dict):
                    pnl_delta = float(res.get("pnl_delta", 0.0))
            except Exception:
                pass
            dd = self._update_drawdown(pnl_delta or 0.0)
            obs.append(IterObs(t0=t0, t1=t1, ok=ok, err=err, pnl=pnl_delta, dd=dd))

        t_end = _now()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return self._summarize(obs, wall_s=(t_end - t_start), mem_cur=cur, mem_peak=peak)

    # --- summary/report ---

    def _summarize(self, obs: List[IterObs], *, wall_s: float, mem_cur: int, mem_peak: int) -> Dict[str, Any]:
        lat = [o.ms for o in obs]
        errs = [o for o in obs if not o.ok]
        pnl = [o.pnl for o in obs if o.pnl is not None]
        dd = [o.dd for o in obs if o.dd is not None]

        def pct(x, p):
            if not x: return None
            xs = sorted(x); k = (len(xs)-1) * p
            f = math.floor(k); c = math.ceil(k)
            if f == c: return xs[int(k)]
            return xs[f] + (xs[c]-xs[f]) * (k-f)

        summ = {
            "mode": self.cfg.mode,
            "iters": len(obs),
            "concurrency": self.cfg.concurrency,
            "latency_ms": {
                "mean": (stats.mean(lat) if lat else None),
                "p50": pct(lat, 0.50),
                "p90": pct(lat, 0.90),
                "p99": pct(lat, 0.99),
                "min": (min(lat) if lat else None),
                "max": (max(lat) if lat else None),
            },
            "throughput_iter_per_s": (len(obs) / wall_s if wall_s > 0 else None),
            "errors": {
                "count": len(errs),
                "rate": (len(errs) / max(1, len(obs))),
                "samples": [e.err for e in errs[:5]],
            },
            "pnl": {
                "sum": (sum(x for x in pnl) if pnl else None),
                "avg": (stats.mean(pnl) if pnl else None) if pnl else None,
            },
            "drawdown": {
                "last": (dd[-1] if dd else None),
                "max": (max(dd) if dd else None),
            },
            "memory_bytes": {
                "current": mem_cur,
                "peak": mem_peak,
            },
            "chaos": asdict(self.cfg.chaos),
            "timestamp": time.time(),
        }
        return summ

    # --- persistence ---

    @staticmethod
    def write_report(report: Dict[str, Any], *, out_json: Optional[str], out_csv: Optional[str], obs: Optional[List[IterObs]] = None) -> None:
        if out_json:
            _ensure_dir(out_json)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print("[stress] wrote", out_json)
        if out_csv and obs:
            _ensure_dir(out_csv)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["i", "ms", "ok", "err", "pnl", "dd"])
                for i, o in enumerate(obs):
                    w.writerow([i, f"{o.ms:.3f}", int(o.ok), o.err or "", o.pnl if o.pnl is not None else "", o.dd if o.dd is not None else ""])


# ---------------------------- CLI ---------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stress", description="Stress & Chaos Harness")
    p.add_argument("--mode", choices=["load", "scenario"], default="load")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--seed", type=int)
    # manager wiring
    p.add_argument("--prices", help="JSON/YAML prices map")
    p.add_argument("--signals", action="append", default=[], help="JSON/YAML signals maps (repeatable)")
    p.add_argument("--agents", action="append", default=[], help="Agent modules under agents/")
    # scenario/sim
    p.add_argument("--scenario", help="Scenario JSON/YAML path or preset name (fallback library)")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--shocks", help="Shocks YAML (optional)")
    # chaos
    p.add_argument("--sleep-ms", nargs=2, type=int, metavar=("LO","HI"))
    p.add_argument("--drop-signals", type=float, default=0.0)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--raise-p", type=float, default=0.0)
    p.add_argument("--clip", nargs=2, type=float, metavar=("LO","HI"))
    # outputs
    p.add_argument("--out", help="Write JSON summary report")
    p.add_argument("--csv", help="Write per-iteration CSV (optional)")
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    chaos = ChaosConfig(
        sleep_ms=tuple(args.sleep_ms) if args.sleep_ms else (0, 0),
        drop_signals_p=float(args.drop_signals or 0.0),
        jitter_pct=float(args.jitter or 0.0),
        raise_p=float(args.raise_p or 0.0),
        clip_signals=(tuple(args.clip) if args.clip else None),
    )
    cfg = StressConfig(
        mode=args.mode,
        iters=args.iters,
        concurrency=args.concurrency,
        seed=args.seed,
        prices_path=args.prices,
        signals_paths=args.signals,
        agents=args.agents,
        scenario_path=args.scenario,
        sim_days=args.days,
        sim_dt=args.dt,
        shocks_yaml=args.shocks,
        chaos=chaos,
        out_json=args.out,
        out_csv=args.csv,
    )
    harness = StressHarness(cfg)
    if args.mode == "load":
        report = harness.run_load()
    else:
        report = harness.run_scenario()

    # Print brief summary and persist if requested
    lat = report["latency_ms"]
    print(f"[stress] iters={report['iters']} err_rate={report['errors']['rate']:.3f} "
          f"p50={lat['p50']:.1f}ms p90={lat['p90']:.1f}ms thr={report['throughput_iter_per_s']:.2f}/s "
          f"dd_max={report['drawdown']['max']}")
    if args.out:
        StressHarness.write_report(report, out_json=args.out, out_csv=args.csv, obs=None)

if __name__ == "__main__":
    main()