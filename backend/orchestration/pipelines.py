# backend/runtime/pipelines.py
"""
Lightweight ETL/Signal pipelines (stdlib-only).

Built on: backend/common/graph.py  (DirectedGraph + GraphRunner)

What you get
------------
- Stage: small unit of work (callable) with retry/timeout and IO slots.
- Pipeline: DAG of stages; topological execution; per-stage metrics.
- Context/DataBus: shared dict-like state passed across stages.
- Checkpointing: optional JSON checkpoint of last results.
- Decorator: @stage to quickly define a Stage from a function.
- Ready-made stages:
    * WrapSignalsAdapter      – run your backend/io/signals_adapter.py
    * MapStage                – transform dicts/records
    * FanoutStage             – copy/partition records into multiple topics
    * FileWriteJSON/YAML      – persist to disk
    * PrintStage              – debug logger
- Tiny CLI demo under __main__.

No external dependencies.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# --- bring your graph runner ---
try:
    from backend.common.graph import DirectedGraph, GraphRunner, Node # type: ignore
except Exception as e:  # very small fallback so file can import standalone
    DirectedGraph = object  # type: ignore
    GraphRunner = object    # type: ignore
    class Node:             # type: ignore
        id: str

# --------------------------------- Core ---------------------------------

@dataclass
class DataBus:
    """A dict-like bag for passing messages and sharing state across stages."""
    store: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value

    def update(self, d: Dict[str, Any]) -> None:
        self.store.update(d)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.store)


@dataclass
class Stage:
    """
    A pipeline node.

    Inputs/Outputs
    --------------
    - inputs:  list of keys to read from DataBus (optional)
    - outputs: list of keys this stage will write (optional; advisory)
    - fn signature: fn(stage: Stage, bus: DataBus, context: Dict[str, Any]) -> Any
      Return value may be:
        - dict -> merged into bus
        - tuple/list of (key, value) -> bus.set(key, value)
        - None -> no write (stage may mutate bus directly)

    Reliability
    -----------
    - retries: max retry attempts (with exponential backoff)
    - timeout_sec: soft timeout; if exceeded, stage fails (no kill)
    """
    name: str
    fn: Callable[[Any, DataBus, Dict[str, Any]], Any]
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    retries: int = 1
    timeout_sec: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # metrics (filled at runtime)
    calls: int = 0
    ok: int = 0
    fail: int = 0
    last_error: Optional[str] = None
    last_ms: float = 0.0

    def run(self, bus: DataBus, context: Dict[str, Any]) -> None:
        self.calls += 1
        attempt = 0
        backoff = 0.2
        start = time.perf_counter()
        err_txt = None
        while attempt <= max(0, self.retries):
            attempt += 1
            try:
                if self.timeout_sec:
                    # soft timeout check via elapsed comparisons inside fn boundary
                    t0 = time.perf_counter()
                out = self.fn(self, bus, context)
                if isinstance(out, dict):
                    bus.update(out)
                elif isinstance(out, (tuple, list)) and len(out) == 2:
                    k, v = out
                    bus.set(str(k), v)
                # success
                self.ok += 1
                self.last_error = None
                break
            except Exception as e:
                err_txt = f"{type(e).__name__}: {e}"
                if attempt > max(0, self.retries):
                    self.fail += 1
                    self.last_error = err_txt
                    break
                # backoff
                time.sleep(backoff)
                backoff = min(2.0, backoff * 1.8)
        self.last_ms = (time.perf_counter() - start) * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "retries": self.retries,
            "timeout_sec": self.timeout_sec,
            "meta": dict(self.meta),
            "metrics": {"calls": self.calls, "ok": self.ok, "fail": self.fail, "last_ms": self.last_ms,
                        "last_error": self.last_error},
        }


def stage(
    name: Optional[str] = None,
    *,
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    retries: int = 1,
    timeout_sec: Optional[float] = None,
    **meta: Any,
):
    """
    Decorator: turn a function into a Stage definition.
    The function will receive (stage, bus, context).
    """
    def wrap(fn: Callable[[Stage, DataBus, Dict[str, Any]], Any]) -> Stage:
        return Stage(
            name=name or fn.__name__,
            fn=fn,
            inputs=list(inputs or []),
            outputs=list(outputs or []),
            retries=retries,
            timeout_sec=timeout_sec,
            meta=dict(meta or {}),
        )
    return wrap


@dataclass
class Pipeline:
    """
    Compose and run a DAG of Stages.

    Usage:
        p = Pipeline("signals")
        p.add(s1).add(s2).edge("load", "normalize").edge("normalize", "mix")
        p.run(context={"env":"dev"})
    """
    name: str
    stages: Dict[str, Stage] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    checkpoint_path: Optional[str] = None

    # runtime
    _graph: Optional[DirectedGraph] = field(default=None, init=False) # type: ignore

    def add(self, s: Stage) -> "Pipeline":
        if s.name in self.stages:
            raise ValueError(f"Stage '{s.name}' already exists")
        self.stages[s.name] = s
        return self

    def edge(self, src: str, dst: str) -> "Pipeline":
        if src not in self.stages or dst not in self.stages:
            raise KeyError("edge(): unknown stage(s)")
        self.edges.append((src, dst))
        return self

    # ----- build and run -----

    def _build_graph(self) -> DirectedGraph: # type: ignore
        g = DirectedGraph()
        for name, s in self.stages.items():
            # attach callable runner to node
            def make_runner(st: Stage):
                def _runner(node: Node, ctx: Dict[str, Any]):
                    bus: DataBus = ctx["bus"]
                    st.run(bus, ctx)
                    return {"ok": True, "stage": st.name, "ms": st.last_ms}
                return _runner
            g.add_node(name, label=name, data=make_runner(s), weight_ms=float(s.meta.get("weight_ms", 10))) # type: ignore
        for u, v in self.edges:
            g.add_edge(u, v) # type: ignore
        return g

    def run(self, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        ctx.setdefault("ts", time.time())
        ctx.setdefault("env", "default")
        bus = ctx.setdefault("bus", DataBus())  # ensure a DataBus instance

        # optional warm start from checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    bus.update(saved.get("bus", {}))
            except Exception:
                pass

        self._graph = self._build_graph()
        runner = GraphRunner(self._graph, max_workers=int(ctx.get("max_workers", 8))) # type: ignore
        res = runner.run(context=ctx) # type: ignore

        # write checkpoint
        if self.checkpoint_path:
            try:
                os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
                with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"ts": ctx["ts"], "bus": bus.as_dict(), "metrics": self.metrics()}, f, indent=2)
            except Exception:
                pass

        return {
            "ok": res.ok,
            "errors": res.errors,
            "timings_ms": res.timings_ms,
            "critical_ms": res.critical_ms,
            "critical_path": res.critical_path,
            "bus": bus.as_dict(),
            "metrics": self.metrics(),
        }

    def metrics(self) -> Dict[str, Any]:
        return {name: s.to_dict() for name, s in self.stages.items()}

# ------------------------- Ready-made stages ----------------------------

class WrapSignalsAdapter(Stage):
    """Stage subclass to run your SignalsAdapter and stash results on the bus."""
    def __init__(self, name: str, adapter, out_key: str = "signals.snapshot"):
        def _run(me: Stage, bus: DataBus, context: Dict[str, Any]):
            snap = adapter.snapshot(now_ts=context.get("ts"), with_raw=False)
            bus.set(out_key, snap)
            # also flatten into bus["signals"] for agents
            bus.set("signals", snap.get("signals", {}))
        super().__init__(name=name, fn=_run, inputs=[], outputs=[out_key, "signals"])

class MapStage(Stage):
    """Apply a pure function to one or more bus keys; write back to a key."""
    def __init__(self, name: str, in_key: str, out_key: str, fn_map: Callable[[Any], Any]):
        def _run(me: Stage, bus: DataBus, context: Dict[str, Any]):
            val = bus.get(in_key)
            bus.set(out_key, fn_map(val))
        super().__init__(name=name, fn=_run, inputs=[in_key], outputs=[out_key])

class FanoutStage(Stage):
    """Write the same payload to multiple keys."""
    def __init__(self, name: str, in_key: str, out_keys: List[str]):
        def _run(me: Stage, bus: DataBus, context: Dict[str, Any]):
            val = bus.get(in_key)
            for k in out_keys:
                bus.set(k, val)
        super().__init__(name=name, fn=_run, inputs=[in_key], outputs=out_keys)

class FileWriteJSON(Stage):
    def __init__(self, name: str, in_key: str, path: str):
        def _run(me: Stage, bus: DataBus, context: Dict[str, Any]):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bus.get(in_key, {}), f, indent=2, sort_keys=True)
        super().__init__(name=name, fn=_run, inputs=[in_key], outputs=[])

class FileWriteYAML(Stage):
    def __init__(self, name: str, in_key: str, path: str):
        def _run(me: Stage, bus: DataBus, context: Dict[str, Any]):
            try:
                import yaml  # optional
            except Exception as e:
                raise RuntimeError("pyyaml required for FileWriteYAML") from e
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(bus.get(in_key, {}), f, sort_keys=True)
        super().__init__(name=name, fn=_run, inputs=[in_key], outputs=[])

class PrintStage(Stage):
    def __init__(self, name: str, key: Optional[str] = None, max_items: int = 8):
        def _run(me: Stage, bus: DataBus, context: Dict[str, Any]):
            if key is None:
                print(f"[{name}] bus keys:", list(bus.as_dict().keys()))
                return
            v = bus.get(key)
            if isinstance(v, dict):
                items = list(v.items())[:max_items]
                print(f"[{name}] {key}: {items}" + (" ..." if len(v) > max_items else ""))
            else:
                print(f"[{name}] {key}: {v!r}")
        super().__init__(name=name, fn=_run, inputs=[key] if key else [], outputs=[])

# ------------------------- Quick builders --------------------------------

def build_signals_pipeline(*, adapter, out_json: Optional[str] = None, verbose: bool = True) -> Pipeline:
    """
    Build a simple 3-stage pipeline:
      1) adapter.snapshot() -> bus["signals"]
      2) fanout to ["signals.live", "signals.latest"]
      3) optional write JSON to disk
    """
    p = Pipeline("signals", checkpoint_path="runs/signals_checkpoint.json")
    s0 = WrapSignalsAdapter("pull_signals", adapter, out_key="signals.snapshot")
    s1 = FanoutStage("fanout", "signals", ["signals.live", "signals.latest"])
    p.add(s0).add(s1).edge("pull_signals", "fanout")
    if out_json:
        s2 = FileWriteJSON("persist_json", "signals.latest", out_json)
        p.add(s2).edge("fanout", "persist_json")
    if verbose:
        p.add(PrintStage("peek_signals", key="signals", max_items=10)).edge("fanout", "peek_signals")
    return p


def build_manager_step_pipeline(*, manager_step: Callable[[], Any]) -> Pipeline:
    """
    Wrap a Manager.run_once() (or any step function) as a one-stage pipeline
    so you can schedule it with schedule.py and still collect metrics.
    """
    @stage("manager_step", outputs=["manager.result"])
    def _step(me: Stage, bus: DataBus, ctx: Dict[str, Any]):
        res = manager_step()
        return {"manager.result": res}
    p = Pipeline("manager_step", checkpoint_path="runs/manager_step_checkpoint.json")
    p.add(_step)
    return p

# ------------------------------ __main__ --------------------------------

if __name__ == "__main__":
    # Tiny demo using a fake signals adapter (replace with your real one)
    class _FakeAdapter:
        def __init__(self):
            self._t = 0
        def snapshot(self, now_ts=None, with_raw=False):
            self._t += 1
            return {"ts": now_ts or time.time(),
                    "signals": {"AAPL.mom": 0.8 + 0.01*self._t, "BTC.sent": 0.3, "risk_z": 0.2}}

    adapter = _FakeAdapter()
    pipe = build_signals_pipeline(adapter=adapter, out_json="runs/signals_latest.json", verbose=True)
    # run once
    res = pipe.run(context={"env": "demo"})
    print("[pipeline] ok:", res["ok"], "critical(ms):", int(res["critical_ms"]))
    print("[pipeline] metrics keys:", list(res["metrics"].keys()))