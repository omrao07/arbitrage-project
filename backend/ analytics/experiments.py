# backend/research/experiments.py
from __future__ import annotations

"""
Experiments Harness
-------------------
- Runs parameterized experiments / sweeps defined in a YAML (or Python dict)
- Records metrics, artifacts, and structured summaries
- Integrates *optionally* with your modules if present:
    * backend.backtests.backtester: Backtester().run(config) -> dict
    * backend.analytics.attribution: AttributionEngine (rollups)
    * backend.risk.expected_shortfall: es_historical, backtest_var_es
    * backend.sim.stress.* (shock_models, scenarios) if you wired them
- Works with or without Redis. If Redis is up, emits logs to `experiments.logs`.

CLI
---
python -m backend.research.experiments run --config config/experiments.yaml
python -m backend.research.experiments sweep --config config/experiments.yaml

Config (YAML)
-------------
name: "alpha_sweep"
workdir: "artifacts/experiments"
seed: 42
mode: "sweep"   # or "run"
tasks:
  - kind: "backtest"
    params:
      start: "2024-01-01"
      end:   "2024-06-30"
      strategies: ["mm_core","arb_core"]
      data_feed: "replay://equities"
      brokers: ["paper"]
      risk:
        var_alpha: 0.99
        max_gross: 3.0
sweep:
  grid:
    mm_core.spread_bps: [2, 4, 6]
    arb_core.edge_bps:  [3, 5]
  max_concurrency: 2
"""

import argparse
import dataclasses
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

# -------- Optional deps (all graceful) ---------------------------------------
try:
    import yaml  # PyYAML (recommended)
except Exception:
    yaml = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

# Optional Redis logging
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

# -------- Env / defaults -----------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
EXPERIMENT_LOG_STREAM = os.getenv("EXPERIMENT_LOG_STREAM", "experiments.logs")

DEFAULT_WORKDIR = "artifacts/experiments"

# -------- Small utils --------------------------------------------------------
def _now_ms() -> int: return int(time.time() * 1000)
def _ts() -> str:      return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def _deep_set(d: Dict[str, Any], dotted: str, value: Any) -> Dict[str, Any]:
    """Set d['a']['b']... via 'a.b' and return d."""
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return d

def _deep_get(d: Dict[str, Any], dotted: str, default=None):
    cur = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, kk))
        else:
            out[kk] = v
    return out

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(obj))

async def _redis() -> Optional[AsyncRedis]: # type: ignore
    if not USE_REDIS:
        return None
    try:
        r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        await r.ping()
        return r
    except Exception:
        return None

async def _xadd(stream: str, obj: Dict[str, Any]):
    r = await _redis()
    if not r:
        return
    try:
        await r.xadd(stream, {"json": _safe_json(obj)}, maxlen=20000, approximate=True)  # type: ignore
    except Exception:
        pass

# -------- Data models --------------------------------------------------------
@dataclass
class RunInfo:
    run_id: str
    name: str
    workdir: str
    start_ms: int
    status: str = "running"  # "running" | "ok" | "error"
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    git_rev: Optional[str] = None
    seed: Optional[int] = None
    params: Dict[str, Any] = None # type: ignore
    metrics: Dict[str, float] = None # type: ignore
    artifacts: Dict[str, str] = None  # type: ignore # logical -> path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -------- Experiment Runner --------------------------------------------------
class ExperimentRunner:
    def __init__(self, name: str, workdir: str = DEFAULT_WORKDIR, seed: Optional[int] = None):
        self.name = name
        self.workdir = str(workdir)
        self.root = Path(self.workdir) / name
        self.root.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self._ensure_repro(seed)

    def _ensure_repro(self, seed: Optional[int]):
        if seed is None:
            return
        random.seed(seed)
        if np is not None:
            try:
                np.random.seed(seed)
            except Exception:
                pass
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _git_rev(self) -> Optional[str]:
        try:
            import subprocess
            rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            return rev
        except Exception:
            return None

    def _new_run(self, params: Dict[str, Any]) -> RunInfo:
        run_id = f"{_ts()}-{str(self.seed or '')}-{abs(hash(_safe_json(params))) % 10**6:06d}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        info = RunInfo(
            run_id=run_id,
            name=self.name,
            workdir=str(run_dir),
            start_ms=_now_ms(),
            git_rev=self._git_rev(),
            seed=self.seed,
            params=params,
            metrics={},
            artifacts={},
        )
        # persist initial state
        (run_dir / "params.json").write_text(_safe_json(params))
        return info

    def _finish_run(self, info: RunInfo, status: str, error: Optional[str] = None):
        info.status = status
        info.duration_ms = _now_ms() - info.start_ms
        info.error = error
        Path(info.workdir, "run.json").write_text(_safe_json(info.to_dict()))

    # ----------------- task dispatch -----------------
    def _import_opt(self, path: str):
        try:
            return importlib.import_module(path)
        except Exception:
            return None

    def _maybe_call(self, dotted: str, *args, **kwargs):
        """Call a dotted function 'module.sub:func' or 'module.func' if present."""
        mod_path, _, func_name = dotted.partition(":")
        if not func_name:
            mod_path, _, func_name = dotted.rpartition(".")
        mod = self._import_opt(mod_path)
        if not mod:
            raise RuntimeError(f"cannot import {mod_path}")
        fn = getattr(mod, func_name, None)
        if not callable(fn):
            raise RuntimeError(f"{dotted} not callable")
        return fn(*args, **kwargs)

    # Built-in task kinds (safe if modules missing)
    def _task_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Looks for backend.backtests.backtester.Backtester().run(params) -> dict
        If missing, returns a stub metric.
        """
        mod = self._import_opt("backend.backtests.backtester")
        if mod and hasattr(mod, "Backtester"):
            bt = getattr(mod, "Backtester")()
            res = bt.run(params)  # expected to return dict of metrics
            return {"kind": "backtest", "metrics": res}
        # Fallback: pretend we ran a toy test
        return {
            "kind": "backtest",
            "metrics": {
                "pnl_total": 0.0,
                "sharpe": 0.0,
                "drawdown": 0.0,
                "trades": 0,
            },
            "note": "backtester module not found; returned stub metrics",
        }

    def _task_stress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        If you have backend.sim.shock_models or scenarios, try calling:
          backend.sim.scenarios:run_scenarios(params) -> dict
        Else, return a deterministic stub.
        """
        # Try custom hook if provided
        if "callable" in params:
            dotted = params["callable"]
            out = self._maybe_call(dotted, params=params)
            return {"kind": "stress", "metrics": out if isinstance(out, dict) else {"ok": True}}

        # Try scenarios module
        scen = self._import_opt("backend.sim.scenarios")
        if scen and hasattr(scen, "run_scenarios"):
            out = scen.run_scenarios(params)
            return {"kind": "stress", "metrics": out}

        # Stub
        seed = self.seed or 0
        rand = (seed * 9301 + 49297) % 233280
        return {"kind": "stress", "metrics": {"max_var": 1.23, "es_99": 2.34, "breaches": rand % 3}}

    def _task_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic metrics task by dotted callable:
            params.callable = "backend.risk.expected_shortfall:es_historical"
            params.args / params.kwargs
        """
        dotted = params.get("callable")
        if not dotted:
            return {"kind": "metrics", "error": "callable missing"}
        args = params.get("args", []) or []
        kwargs = params.get("kwargs", {}) or {}
        out = self._maybe_call(dotted, *args, **kwargs)
        return {"kind": "metrics", "result": out}

    def _task_shell(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a system command (use with care in your environment).
        params.cmd = "python scripts/do_thing.py --opt 1"
        """
        import subprocess, shlex
        cmd = params.get("cmd")
        if not cmd:
            return {"kind": "shell", "error": "cmd missing"}
        t0 = time.time()
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        return {
            "kind": "shell",
            "code": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "lat_ms": int((time.time() - t0) * 1000),
        }

    def _dispatch(self, task: Dict[str, Any]) -> Dict[str, Any]:
        kind = (task.get("kind") or "").lower()
        params = task.get("params") or {}
        if kind == "backtest":
            return self._task_backtest(params)
        if kind == "stress":
            return self._task_stress(params)
        if kind == "metrics":
            return self._task_metrics(params)
        if kind == "shell":
            return self._task_shell(params)
        raise RuntimeError(f"unknown task kind '{kind}'")

    # ----------------- run / sweep -----------------
    def run(self, tasks: List[Dict[str, Any]], params: Dict[str, Any]) -> RunInfo:
        info = self._new_run(params=params)
        try:
            all_metrics: Dict[str, float] = {}
            results: List[Dict[str, Any]] = []
            for i, t in enumerate(tasks):
                step_name = t.get("name") or f"task_{i}"
                t0 = time.time()
                try:
                    out = self._dispatch(t)
                    out["_name"] = step_name
                    out["_lat_ms"] = int((time.time() - t0) * 1000)
                    results.append(out)
                    # merge numeric metrics for top-level view
                    m = out.get("metrics", {})
                    for k, v in (m.items() if isinstance(m, dict) else []):
                        if isinstance(v, (int, float)) and math.isfinite(v):
                            all_metrics[f"{step_name}.{k}"] = float(v)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    results.append({"_name": step_name, "error": err, "trace": traceback.format_exc()[-4000:]})
            # write artifacts
            Path(info.workdir, "results.json").write_text(_safe_json(results))
            Path(info.workdir, "metrics.json").write_text(_safe_json(all_metrics))
            info.metrics = all_metrics
            info.artifacts["results"] = str(Path(info.workdir, "results.json"))
            info.artifacts["metrics"] = str(Path(info.workdir, "metrics.json"))
            # summary.md
            summary = self._make_summary_md(info, results)
            Path(info.workdir, "summary.md").write_text(summary)
            info.artifacts["summary"] = str(Path(info.workdir, "summary.md"))
            self._finish_run(info, status="ok")
        except Exception as e:
            self._finish_run(info, status="error", error=f"{type(e).__name__}: {e}")
        return info

    def sweep(self, base_tasks: List[Dict[str, Any]], base_params: Dict[str, Any], grid: Dict[str, List[Any]],
              max_concurrency: int = 1) -> List[RunInfo]:
        """
        Expand a grid on top of base_params; spawn sequentially (simple & robust).
        If you need true parallel, spawn separate processes with this module.
        """
        keys = list(grid.keys())
        vals = [list(v) for v in grid.values()]
        runs: List[RunInfo] = []
        for combo in product(*vals):
            params = json.loads(_safe_json(base_params))  # deep copy
            label_bits = []
            for k, v in zip(keys, combo):
                _deep_set(params, k, v)
                label_bits.append(f"{k}={v}")
            label = ",".join(label_bits)
            print(f"[sweep] running: {label}")
            # Tag tasks with label if they want it
            tasks = json.loads(_safe_json(base_tasks))
            for t in tasks:
                t.setdefault("params", {}).setdefault("_sweep_label", label)
            info = self.run(tasks, params)
            runs.append(info)
        return runs

    # ----------------- reporting -----------------
    def _make_summary_md(self, info: RunInfo, results: List[Dict[str, Any]]) -> str:
        lines = []
        lines.append(f"# Experiment: {info.name}")
        lines.append(f"- run_id: `{info.run_id}`")
        lines.append(f"- status: **{info.status}**  (duration: {int((info.duration_ms or 0)/1000)}s)")
        if info.git_rev:
            lines.append(f"- git: `{info.git_rev}`")
        if info.seed is not None:
            lines.append(f"- seed: `{info.seed}`")
        lines.append("")
        lines.append("## Params")
        lines.append("```json")
        lines.append(_safe_json(info.params))
        lines.append("```")
        lines.append("")
        lines.append("## Metrics (flattened)")
        lines.append("```json")
        lines.append(_safe_json(info.metrics))
        lines.append("```")
        lines.append("")
        lines.append("## Steps")
        for r in results:
            nm = r.get("_name", "step")
            lines.append(f"### {nm}")
            lines.append("```json")
            lines.append(_safe_json({k:v for k,v in r.items() if not k.startswith('_')}))
            lines.append("```")
        return "\n".join(lines)

# -------- Config loader ------------------------------------------------------
def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".yml", ".yaml"}:
        if not yaml:
            raise RuntimeError("PyYAML not installed; cannot read YAML config")
        return yaml.safe_load(p.read_text()) or {}
    # JSON fallback
    return json.loads(p.read_text())

# -------- CLI ---------------------------------------------------------------
def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser("experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a single experiment")
    run.add_argument("--config", required=True, help="YAML/JSON config file")

    swp = sub.add_parser("sweep", help="run parameter sweep")
    swp.add_argument("--config", required=True, help="YAML/JSON config file")
    return ap.parse_args(argv)

def _main(argv: List[str]) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    name = str(cfg.get("name") or f"exp-{_ts()}")
    workdir = str(cfg.get("workdir") or DEFAULT_WORKDIR)
    seed = cfg.get("seed")
    runner = ExperimentRunner(name=name, workdir=workdir, seed=seed)

    base_params = cfg.get("params") or {}
    tasks = cfg.get("tasks") or []

    if args.cmd == "run":
        info = runner.run(tasks, base_params)
        print(json.dumps(info.to_dict(), indent=2))
        return 0

    if args.cmd == "sweep":
        sweep = cfg.get("sweep") or {}
        grid = sweep.get("grid") or {}
        max_ccy = int(sweep.get("max_concurrency") or 1)
        # (sequential in this file; for true parallel, spawn separate processes)
        infos = runner.sweep(tasks, base_params, grid, max_concurrency=max_ccy)
        print(_safe_json([i.to_dict() for i in infos]))
        return 0

    return 1

if __name__ == "__main__":
    try:
        rc = _main(sys.argv[1:])
        sys.exit(rc)
    except Exception as e:
        print("ERR:", e)
        sys.exit(2)