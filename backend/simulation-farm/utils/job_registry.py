# simulation-farm/utils/job_registry.py
"""
Job Registry
============

- Loads job definitions from config/jobs.yaml
- Builds the appropriate *Spec* dataclass per job type
- Returns either the Spec, the Job instance, or runs it via a chosen runner

Usage
-----
from simulation_farm.utils.job_registry import JobRegistry

reg = JobRegistry("simulation-farm/config/jobs.yaml")

print(reg.list_jobs())  # ['backtests.momentum_us_equities', 'stress_tests.covid_crash', ...]
print(reg.get("backtests.momentum_us_equities"))  # (job_type, spec)

# Run with LocalRunner
res = reg.run("backtests.momentum_us_equities", runner="local", runner_kwargs={"max_workers": 1})
print(res["status"], res["urls"])

# Override fields at runtime
res = reg.run(
    "backtests.momentum_us_equities",
    overrides={"start": "2021-01-01", "end": "2024-12-31", "output_prefix": "runs/momentum_override/"},
)

CLI:
python -m simulation_farm.utils.job_registry --list
python -m simulation_farm.utils.job_registry --job backtests.momentum_us_equities --runner local
python -m simulation_farm.utils.job_registry --job monte_carlo.risk_mc_us --runner ray --override start=2018-01-01 end=2022-12-31
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml  # pip install pyyaml

# Job Specs & Jobs
from simulation_farm.jobs.backtest_job import BacktestSpec, BacktestJob # type: ignore
from simulation_farm.jobs.monte_carlo_job import MonteCarloSpec, MonteCarloJob # type: ignore
from simulation_farm.jobs.replay_job import ReplaySpec, ReplayJob # type: ignore
from simulation_farm.jobs.stress_test_job import StressTestSpec, StressTestJob # type: ignore


class JobRegistry:
    def __init__(self, jobs_yaml: str = "simulation-farm/config/jobs.yaml"):
        self.path = Path(jobs_yaml)
        if not self.path.exists():
            raise FileNotFoundError(f"jobs.yaml not found at {self.path}")
        self._cfg = _load_yaml(self.path)

    # ---- listing / retrieval ----

    def list_jobs(self) -> List[str]:
        """Return dotted paths like 'backtests.momentum_us_equities'."""
        out: List[str] = []
        for section, jobs in self._iter_sections():
            for name in sorted(jobs.keys()):
                out.append(f"{section}.{name}")
        return out

    def get(self, dotted: str) -> Tuple[str, Any]:
        """
        Return (job_type, spec_dataclass) for a dotted path (e.g., 'backtests.momentum_us_equities').
        """
        section, name = _split_dotted(dotted)
        jobs = self._cfg.get(section) or {}
        if name not in jobs:
            raise KeyError(f"Job '{name}' not found under '{section}'. Available: {list(jobs.keys())}")
        job_cfg = dict(jobs[name])  # copy
        job_type = _section_to_type(section)
        spec = _build_spec(job_type, job_cfg)
        return job_type, spec

    def get_job(self, dotted: str):
        """Instantiate the proper *Job* object (BacktestJob, MonteCarloJob, etc.) with its Spec."""
        job_type, spec = self.get(dotted)
        return _make_job(job_type, spec)

    # ---- run helpers ----

    def run(
        self,
        dotted: str,
        *,
        overrides: Optional[Dict[str, Any]] = None,
        runner: str = "local",
        runner_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build Spec (with optional overrides) and run it via the chosen runner.

        Args:
          dotted: 'backtests.foo', 'monte_carlo.bar', 'replays.baz', 'stress_tests.qux'
          overrides: dict of Spec field overrides (e.g., {'start': '2021-01-01'})
          runner: 'local' | 'ray' | 'k8s'
          runner_kwargs: extra kwargs for runner constructor
        """
        section, name = _split_dotted(dotted)
        jobs = self._cfg.get(section) or {}
        if name not in jobs:
            raise KeyError(f"Job '{name}' not found under '{section}'.")
        job_cfg = dict(jobs[name])
        if overrides:
            job_cfg.update(overrides)

        job_type = _section_to_type(section)
        spec = _build_spec(job_type, job_cfg)

        rk = runner.lower()
        runner_kwargs = runner_kwargs or {}
        if rk == "local":
            from simulation_farm.runners.local_runner import LocalRunner # type: ignore
            r = LocalRunner(**runner_kwargs)
            return r.run(job_type, spec)
        elif rk == "ray":
            from simulation_farm.runners.ray_runner import RayRunner # type: ignore
            r = RayRunner(**runner_kwargs)
            return r.run(job_type, spec)
        elif rk in ("k8s", "kubernetes"):
            from simulation_farm.runners.k8s_runner import K8sRunner # type: ignore
            env = runner_kwargs.pop("k8s_env", None)  # optional env for container
            kr = K8sRunner(**runner_kwargs)
            job_name = kr.submit(job_type, spec, env=env)
            return kr.wait(job_name)
        else:
            raise ValueError("runner must be one of: local | ray | k8s")

    # ---- internal ----

    def _iter_sections(self):
        for section in ("backtests", "stress_tests", "monte_carlo", "replays"):
            jobs = self._cfg.get(section)
            if isinstance(jobs, dict):
                yield section, jobs


# ---------------- Spec/Job builders ----------------

def _section_to_type(section: str) -> str:
    if section == "backtests": return "backtest"
    if section == "monte_carlo": return "monte_carlo"
    if section == "replays": return "replay"
    if section == "stress_tests": return "stress_test"
    raise ValueError(f"Unknown section: {section}")

def _build_spec(job_type: str, cfg: Dict[str, Any]):
    """
    Convert a jobs.yaml entry into the appropriate Spec dataclass.
    Accepts common keys across job types and ignores extras.
    """
    common = {
        "run_id": cfg.get("run_id") or _default_run_id(cfg),
        "data_path": cfg.get("data") or cfg.get("data_path"),
        "start": cfg.get("start"),
        "end": cfg.get("end"),
        "output_prefix": cfg.get("output_prefix") or "runs/demo/",
        "cash": cfg.get("cash"),
    }
    # Exporter selection (optional)
    exporter = cfg.get("exporter")
    if exporter and isinstance(exporter, (list, tuple)) and len(exporter) == 2:
        common["exporter"] = (exporter[0], dict(exporter[1]))
    elif "exporter_kind" in cfg:
        common["exporter"] = (cfg.get("exporter_kind"), dict(cfg.get("exporter_args") or {}))

    if job_type == "backtest":
        return BacktestSpec(
            strategy=cfg.get("strategy", "momentum"),
            params=dict(cfg.get("params") or {}),
            universe=cfg.get("universe"),
            costs_bps=float(cfg.get("costs_bps", 5.0)),
            slippage_bps=float(cfg.get("slippage_bps", 2.0)),
            max_weight=float(cfg.get("max_weight", 0.05)),
            rebalance_days=int(cfg.get("rebalance_days", 5)),
            max_names=cfg.get("max_names"),
            title=cfg.get("title", "Backtest Report"),
            tz=cfg.get("tz", "UTC"),
            **common,
        )
    if job_type == "monte_carlo":
        return MonteCarloSpec(
            n_paths=int(cfg.get("n_paths", 2000)),
            horizon_days=int(cfg.get("horizon_days", 252)),
            weights=cfg.get("weights", "ew"),
            recalc_vol_days=int(cfg.get("recalc_vol_days", 252)),
            title=cfg.get("title", "Monte Carlo Simulation"),
            tz=cfg.get("tz", "UTC"),
            **common,
        )
    if job_type == "replay":
        return ReplaySpec(
            strategy=cfg.get("strategy", "momentum"),
            universe=cfg.get("universe"),
            speed=float(cfg.get("speed", 1.0)),
            window_minutes=int(cfg.get("window_minutes", 30)),
            title=cfg.get("title", "Replay Report"),
            tz=cfg.get("tz", "UTC"),
            **common,
        )
    if job_type == "stress_test":
        return StressTestSpec(
            weights=cfg.get("weights", "ew"),
            vol_lookback_days=int(cfg.get("vol_lookback_days", 252)),
            historical_window=cfg.get("scenario") or cfg.get("historical_window"),
            equity_drop=cfg.get("shocks", {}).get("equity_drop", cfg.get("equity_drop")),
            vol_spike=cfg.get("shocks", {}).get("vol_spike", cfg.get("vol_spike")),
            rate_shift_bps=cfg.get("rate_shift_bps"),
            duration_years=float(cfg.get("duration_years", cfg.get("duration", 0.0))),
            factor_betas=dict(cfg.get("factor_betas") or {}),
            factor_shocks=dict(cfg.get("factor_shocks") or {}),
            title=cfg.get("title", "Stress Test Report"),
            tz=cfg.get("tz", "UTC"),
            **common,
        )
    raise ValueError(f"Unsupported job_type: {job_type}")

def _make_job(job_type: str, spec):
    if job_type == "backtest": return BacktestJob(spec)
    if job_type == "monte_carlo": return MonteCarloJob(spec)
    if job_type == "replay": return ReplayJob(spec)
    if job_type == "stress_test": return StressTestJob(spec)
    raise ValueError(f"Unsupported job_type: {job_type}")


# ---------------- Utilities ----------------

def _default_run_id(cfg: Dict[str, Any]) -> str:
    base = cfg.get("strategy") or cfg.get("type") or "job"
    from datetime import datetime
    return f"{base}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

def _split_dotted(dotted: str) -> Tuple[str, str]:
    if "." not in dotted:
        raise ValueError("Use 'section.name' (e.g., 'backtests.momentum_us_equities').")
    section, name = dotted.split(".", 1)
    return section, name

def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML at {path}")
    return cfg


# ---------------- CLI ----------------

def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Job Registry CLI")
    ap.add_argument("--jobs-yaml", default="simulation-farm/config/jobs.yaml")
    ap.add_argument("--list", action="true_const", const=True, nargs="?", help="List available jobs")
    ap.add_argument("--job", help="Dotted path, e.g., backtests.momentum_us_equities")
    ap.add_argument("--runner", default="local", choices=["local", "ray", "k8s"])
    ap.add_argument("--override", action="append", default=[], help="key=value (repeatable)")
    ap.add_argument("--runner-kw", action="append", default=[], help="runnerKey=val (repeatable)")
    return ap.parse_args()

def _kv_list_to_dict(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in items:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                if v.lower() in ("true", "false"):
                    out[k] = (v.lower() == "true")
                elif "." in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except Exception:
                out[k] = v
    return out

if __name__ == "__main__":
    args = _parse_cli()
    reg = JobRegistry(args.jobs_yaml)
    if getattr(args, "list", False) or not args.job:
        print(json.dumps({"jobs": reg.list_jobs()}, indent=2))
    else:
        overrides = _kv_list_to_dict(args.override)
        runner_kwargs = _kv_list_to_dict(args.runner_kw)
        result = reg.run(args.job, overrides=overrides, runner=args.runner, runner_kwargs=runner_kwargs)
        print(json.dumps(result, indent=2))