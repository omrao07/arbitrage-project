# backend/risk/dynamic_governor.py
"""
Dynamic Governor
----------------
Continuously evaluates live telemetry and enforces risk/latency/sentiment
policies by issuing commands to strategies/OMS:

- Pause / resume a strategy
- Adjust default order qty / max leverage / participation caps
- Flip router TIF (e.g., POST_ONLY under stress to avoid taker fees/impact)
- Publish alerts + policy decisions to the bus

Inputs (best-effort; all optional)
- HGET strategy:drawdown <name>   -> {"dd": 0.08}
- HGET strategy:vol <name>        -> {"vol": 0.25}
- HGET strategy:signal <name>     -> {"score": 0.12}
- HGET engine:health engine       -> {...}
- Streams you already emit: risk.metrics, engine.metrics, news.sentiment, oms.router, tca.metrics

Outputs
- XADD strategy.cmd.<name> * {"cmd":"pause"|"resume"} or {"cmd":"set","key":"default_qty","value":...}
- HSET strategy:enabled <name> true|false
- XADD risk.governor * {decision...}
- HSET risk:limits <name> {max_leverage:..., max_participation:...}

CLI
- python -m backend.risk.dynamic_governor --probe
- python -m backend.risk.dynamic_governor --run --registry config/governor.yaml

Example YAML (config/governor.yaml)
-----------------------------------
strategies:
  - name: momo_in
    min_qty: 1
    max_qty: 500
    cool_off_s: 300
    rules:
      - when: "dd >= 0.08"                   # 8% drawdown
        actions:
          - pause
          - set: { key: "default_qty", value: 1 }
      - when: "vol >= 0.35 or slip_bps >= 25"
        actions:
          - set: { key: "default_qty", value: "max(min_qty, 0.25 * default_qty)" }
          - limits: { max_participation: 0.05 }
      - when: "sentiment <= -0.6 and news_heat >= 0.7"
        actions:
          - set: { key: "tif_override", value: "POST_ONLY" }
      - when: "dd < 0.04 and vol < 0.28"
        actions:
          - resume
          - limits: { max_participation: 0.12 }
defaults:
  polling_ms: 1000
  global_max_slip_bps: 35
"""

from __future__ import annotations

import ast
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

# ---- Optional deps ----
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # type: ignore

try:
    import redis  # pip install redis
except Exception:
    redis = None  # type: ignore

# bus (best-effort)
try:
    from backend.bus.streams import publish_stream, hset
except Exception:
    publish_stream = None  # type: ignore
    hset = None  # type: ignore


# -------------------- small utils --------------------

def _now_ms() -> int: return int(time.time() * 1000)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if math.isfinite(f): return f
    except Exception:
        pass
    return float(default)

def _safe_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in ("true","1","yes","y","on"): return True
    if s in ("false","0","no","n","off"): return False
    return default

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -------------------- config models --------------------

@dataclass
class Rule:
    expr: str               # e.g., "dd >= 0.08 and vol > 0.30"
    actions: List[Dict[str, Any]]

@dataclass
class StratCfg:
    name: str
    min_qty: float = 1.0
    max_qty: float = 1_000_000.0
    cool_off_s: int = 120
    rules: List[Rule] = field(default_factory=list)

@dataclass
class GovConfig:
    strategies: List[StratCfg] = field(default_factory=list)
    polling_ms: int = 1000
    global_max_slip_bps: float = 35.0

    @staticmethod
    def from_yaml(path: str) -> "GovConfig":
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot load governor.yaml")
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        defs = doc.get("defaults") or {}
        cfg = GovConfig(
            polling_ms = int(defs.get("polling_ms", 1000)),
            global_max_slip_bps = float(defs.get("global_max_slip_bps", 35.0)),
        )
        for s in (doc.get("strategies") or []):
            rules = []
            for r in (s.get("rules") or []):
                rules.append(Rule(expr=str(r.get("when","")), actions=list(r.get("actions") or [])))
            cfg.strategies.append(StratCfg(
                name=s["name"],
                min_qty=float(s.get("min_qty", 1.0)),
                max_qty=float(s.get("max_qty", 1_000_000.0)),
                cool_off_s=int(s.get("cool_off_s", 120)),
                rules=rules
            ))
        return cfg


# -------------------- telemetry provider --------------------

class Telemetry:
    """
    Pulls the minimal metrics the governor needs, best-effort.
    If Redis is available, it hits HGETs. Else, accept injected provider function.
    """
    def __init__(self, r: Optional["redis.Redis"] = None, provider: Optional[Callable[[str], Dict[str, Any]]] = None): # type: ignore
        self._r = r
        self._provider = provider

    def read(self, strat_name: str) -> Dict[str, Any]:
        """
        Returns a dict of normalized signals:
        {
          "dd": 0.08, "vol": 0.32, "slip_bps": 12, "lat_ms": 18,
          "sentiment": -0.4, "news_heat": 0.7,
          "enabled": True, "default_qty": 10, "max_participation": 0.1
        }
        """
        if self._provider:
            try:
                return dict(self._provider(strat_name) or {})
            except Exception:
                pass

        out: Dict[str, Any] = {}
        if not self._r:
            return out

        try:
            dd = self._r.hget("strategy:drawdown", strat_name)
            vol = self._r.hget("strategy:vol", strat_name)
            sig = self._r.hget("strategy:signal", strat_name)
            en  = self._r.hget("strategy:enabled", strat_name)
            # optional: recent OMS/TCA metrics
            slip = self._r.hget("router:slippage_bps", strat_name)
            lat  = self._r.hget("router:latency_ms", strat_name)
            # optional: sentiment aggregator (your news pipeline can write these)
            sent = self._r.hget("news:sentiment", strat_name)  # or global key "news:sentiment:*"
            heat = self._r.hget("news:heat", strat_name)

            # optional runtime knobs the strat might expose in ctx
            dq = self._r.hget("strategy:ctx:default_qty", strat_name)
            mp = self._r.hget("risk:limits:max_participation", strat_name)

            # parse
            if dd: 
                # value may be a JSON-like or just number; handle both
                try:
                    import json
                    v = json.loads(dd); out["dd"] = _safe_float(v.get("dd", v))
                except Exception:
                    out["dd"] = _safe_float(dd)
            if vol:
                try:
                    import json
                    v = json.loads(vol); out["vol"] = _safe_float(v.get("vol", v))
                except Exception:
                    out["vol"] = _safe_float(vol)
            if sig:
                try:
                    import json
                    v = json.loads(sig); out["signal"] = _safe_float(v.get("score", v))
                except Exception:
                    out["signal"] = _safe_float(sig)
            if en is not None:
                out["enabled"] = _safe_bool(en, True)
            if slip is not None:
                out["slip_bps"] = _safe_float(slip)
            if lat is not None:
                out["lat_ms"] = _safe_float(lat)
            if sent is not None:
                out["sentiment"] = _safe_float(sent)
            if heat is not None:
                out["news_heat"] = _safe_float(heat)
            if dq is not None:
                out["default_qty"] = _safe_float(dq)
            if mp is not None:
                out["max_participation"] = _safe_float(mp)
        except Exception:
            pass

        return out


# -------------------- safe expression evaluator --------------------

_ALLOWED_NAMES = {"dd","vol","slip_bps","lat_ms","sentiment","news_heat","enabled","default_qty","max_participation"}

class _ExprEvaluator(ast.NodeVisitor):
    def __init__(self, vars: Dict[str, Any]):
        self.vars = vars

    def visit(self, node):  # type: ignore
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.BoolOp):
            vals = [self.visit(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.BinOp):
            left, right = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Mod): return left % right
            raise ValueError("op not allowed")
        if isinstance(node, ast.UnaryOp):
            val = self.visit(node.operand)
            if isinstance(node.op, ast.USub): return -val
            if isinstance(node.op, ast.UAdd): return +val
            if isinstance(node.op, ast.Not): return not val
            raise ValueError("uop not allowed")
        if isinstance(node, ast.Compare):
            left = self.visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self.visit(comp)
                if isinstance(op, ast.Lt): ok = left < right
                elif isinstance(op, ast.LtE): ok = left <= right
                elif isinstance(op, ast.Gt): ok = left > right
                elif isinstance(op, ast.GtE): ok = left >= right
                elif isinstance(op, ast.Eq): ok = left == right
                elif isinstance(op, ast.NotEq): ok = left != right
                else: raise ValueError("cmp not allowed")
                if not ok: return False
                left = right
            return True
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                raise ValueError(f"name '{node.id}' not allowed")
            return self.vars.get(node.id, 0.0)
        if isinstance(node, ast.Constant):
            return node.value
        raise ValueError("expr not allowed")

def _eval_expr(expr: str, vars: Dict[str, Any]) -> bool:
    if not expr.strip():
        return False
    tree = ast.parse(expr, mode="eval")
    return bool(_ExprEvaluator(vars).visit(tree))


# -------------------- governor core --------------------

class DynamicGovernor:
    def __init__(self, cfg: GovConfig, telemetry: Telemetry):
        self.cfg = cfg
        self.tel = telemetry
        self._r = telemetry._r
        self._running = False
        self._cool_until: Dict[str, int] = {}  # strat -> ts_ms until which it's in cooloff

    # ---- main loop ----
    def tick(self):
        for scfg in self.cfg.strategies:
            name = scfg.name
            metrics = self.tel.read(name)
            now = _now_ms()

            # derived fallbacks
            dd         = _safe_float(metrics.get("dd"))
            vol        = _safe_float(metrics.get("vol"))
            slip_bps   = _safe_float(metrics.get("slip_bps"))
            lat_ms     = _safe_float(metrics.get("lat_ms"))
            sentiment  = _safe_float(metrics.get("sentiment"))
            news_heat  = _safe_float(metrics.get("news_heat"))
            default_qty= _safe_float(metrics.get("default_qty"), 1.0)

            vars = {
                "dd": dd, "vol": vol, "slip_bps": slip_bps, "lat_ms": lat_ms,
                "sentiment": sentiment, "news_heat": news_heat,
                "enabled": bool(metrics.get("enabled", True)),
                "default_qty": default_qty,
                "max_participation": _safe_float(metrics.get("max_participation"), 0.10),
            }

            # enforce global guards (slippage hard cap)
            if slip_bps and slip_bps >= self.cfg.global_max_slip_bps:
                self._apply(name, [{"pause": True}, {"limits": {"max_participation": 0.02}}], vars, reason="hard_slippage_cap")
                self._cool_until[name] = now + scfg.cool_off_s * 1000
                continue

            # still cooling off?
            if self._cool_until.get(name, 0) > now:
                self._emit("cooloff", name, {"until_ms": self._cool_until[name], "vars": vars})
                continue

            # evaluate rules in order
            acted = False
            for rule in scfg.rules:
                try:
                    if _eval_expr(rule.expr, vars):
                        self._apply(name, rule.actions, vars, reason=f"rule:{rule.expr}")
                        acted = True
                        # start cool-off if any action paused/resized risk
                        if any(("pause" in a) or ("limits" in a) or ("set" in a) for a in rule.actions):
                            self._cool_until[name] = now + scfg.cool_off_s * 1000
                        break  # first match wins
                except Exception as e:
                    self._emit("error", name, {"err": str(e), "rule": rule.expr})
            if not acted:
                # write a light heartbeat
                self._emit("noop", name, {"vars": vars})

    def start(self):
        if self._running: return
        self._running = True
        th = threading.Thread(target=self._loop, daemon=True)
        th.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                self.tick()
            except Exception as e:
                self._emit("error", "governor", {"err": str(e)})
            time.sleep(max(0.05, self.cfg.polling_ms / 1000.0))

    # ---- action application ----
    def _apply(self, strat: str, actions: List[Dict[str, Any]], vars: Dict[str, Any], *, reason: str):
        self._emit("decision", strat, {"reason": reason, "actions": actions, "vars": vars})

        for a in actions:
            # pause / resume
            if "pause" in a and _safe_bool(a.get("pause"), True):
                self._cmd(strat, {"cmd": "pause"})
                self._hset("strategy:enabled", strat, "false")
            if "resume" in a and _safe_bool(a.get("resume"), True):
                self._cmd(strat, {"cmd": "resume"})
                self._hset("strategy:enabled", strat, "true")

            # set param on strategy ctx
            if "set" in a and isinstance(a["set"], dict):
                key = str(a["set"].get("key"))
                val_spec = a["set"].get("value")
                val = self._eval_value(val_spec, vars)
                self._cmd(strat, {"cmd": "set", "key": key, "value": val})
                # Mirror to Redis for dashboards if it's a “known” knob
                if key in ("default_qty","tif_override","pov","max_children"):
                    self._hset(f"strategy:ctx:{key}", strat, val)

            # set risk limits (for router/risk_manager to read)
            if "limits" in a and isinstance(a["limits"], dict):
                for k, v in a["limits"].items():
                    val = self._eval_value(v, vars)
                    if k == "max_participation":
                        val = float(_clamp(float(val), 0.01, 0.50))
                    self._hset(f"risk:limits:{k}", strat, val)

    def _cmd(self, strat: str, payload: Dict[str, Any]) -> None:
        if publish_stream:
            try:
                publish_stream(f"strategy.cmd.{strat}", {"ts_ms": _now_ms(), **payload})
            except Exception:
                pass

    def _hset(self, key: str, field: str, value: Any) -> None:
        if hset:
            try:
                hset(key, field, value)
            except Exception:
                pass
        elif self._r:
            try:
                self._r.hset(key, field, value)
            except Exception:
                pass

    def _emit(self, kind: str, strat: str, payload: Dict[str, Any]) -> None:
        if publish_stream:
            try:
                publish_stream("risk.governor", {"ts_ms": _now_ms(), "kind": kind, "strategy": strat, **payload})
            except Exception:
                pass

    # value can be a literal, or a tiny expression referencing current vars:
    # e.g., "max(min_qty, 0.25 * default_qty)"
    def _eval_value(self, val: Any, vars: Dict[str, Any]) -> Any:
        if isinstance(val, (int, float, bool)) or val is None:
            return val
        s = str(val).strip()
        try:
            tree = ast.parse(s, mode="eval")
            # only allow +,-,*,/,numbers,names,max,min
            return _eval_value_expr(tree, vars)
        except Exception:
            return val  # fallback as string


# Safe value expression evaluator allowing max/min & arithmetic
_ALLOWED_FUNCS = {"max": max, "min": min}

class _ValueEval(ast.NodeVisitor):
    def __init__(self, vars: Dict[str, Any]):
        self.vars = vars

    def visit(self, node):  # type: ignore
        if isinstance(node, ast.Expression): return self.visit(node.body)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("func not allowed")
            args = [self.visit(a) for a in node.args]
            return _ALLOWED_FUNCS[node.func.id](*args)
        if isinstance(node, ast.BinOp):
            a, b = self.visit(node.left), self.visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
            raise ValueError("op not allowed")
        if isinstance(node, ast.UnaryOp):
            v = self.visit(node.operand)
            if isinstance(node.op, ast.USub): return -v
            if isinstance(node.op, ast.UAdd): return +v
            raise ValueError("uop not allowed")
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES and node.id not in ("min_qty", "max_qty"):
                raise ValueError("name not allowed")
            return self.vars.get(node.id, 0.0)
        if isinstance(node, ast.Constant): return node.value
        raise ValueError("expr not allowed")

def _eval_value_expr(tree: ast.AST, vars: Dict[str, Any]) -> Any:
    return _ValueEval(vars).visit(tree)


# -------------------- CLI --------------------

def _probe():
    # Minimal, Redis-free probe: simulate metrics across a few ticks.
    cfg = GovConfig(
        strategies=[
            StratCfg(
                name="momo_in",
                min_qty=1, max_qty=500, cool_off_s=5,
                rules=[
                    Rule("dd >= 0.08", [{"pause": True}, {"set": {"key":"default_qty","value":1}}]),
                    Rule("vol >= 0.35 or slip_bps >= 25", [{"set":{"key":"default_qty","value":"max(min_qty, 0.5*default_qty)"}}, {"limits":{"max_participation":0.05}}]),
                    Rule("dd < 0.04 and vol < 0.28", [{"resume": True}, {"limits":{"max_participation":0.12}}]),
                ]
            )
        ],
        polling_ms=300,
    )

    class FakeProvider:
        def __init__(self):
            self.t=0
        def __call__(self, name):
            # oscillate dd/vol to trigger rules
            self.t+=1
            return {
                "dd": 0.02 if self.t<6 else 0.09 if self.t<12 else 0.03,
                "vol": 0.26 if self.t<6 else 0.36 if self.t<12 else 0.24,
                "slip_bps": 12,
                "default_qty": 10,
                "enabled": True,
                "max_participation": 0.10
            }

    gov = DynamicGovernor(cfg, Telemetry(provider=FakeProvider()))
    # run a short demo
    for _ in range(20):
        gov.tick()
        time.sleep(cfg.polling_ms/1000.0)
    print("Probe complete (see bus events if wired).")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dynamic Governor")
    ap.add_argument("--probe", action="store_true", help="Run a local probe (no Redis needed)")
    ap.add_argument("--run", action="store_true", help="Run with real telemetry")
    ap.add_argument("--registry", type=str, help="Path to config/governor.yaml")
    args = ap.parse_args()

    if args.probe:
        _probe()
        return

    # Build telemetry (Redis if available)
    r = None
    if redis:
        try:
            host=os.getenv("REDIS_HOST","localhost"); port=int(os.getenv("REDIS_PORT","6379"))
            r=redis.Redis(host=host, port=port, decode_responses=True)
        except Exception:
            r=None
    tel = Telemetry(r=r)

    # Load config
    if args.registry:
        cfg = GovConfig.from_yaml(args.registry)
    else:
        # tiny default
        cfg = GovConfig(
            strategies=[StratCfg(name="example", rules=[Rule("dd >= 0.08", [{"pause": True}])])],
            polling_ms=1000
        )

    gov = DynamicGovernor(cfg, tel)
    if args.run:
        gov.start()
        try:
            while True: time.sleep(1.0)
        except KeyboardInterrupt:
            gov.stop()
    else:
        _probe()

if __name__ == "__main__":
    main()