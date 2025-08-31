# backend/engine/strategy_dsl.py
from __future__ import annotations

import ast, math, time, json, threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, DefaultDict
from collections import defaultdict, deque

# ---------- Optional YAML (graceful) ----------
HAVE_YAML = True
try:
    import yaml  # type: ignore
except Exception:
    HAVE_YAML = False
    yaml = None  # type: ignore

Number = Union[int, float]

# ===================== Rolling storage & indicators ==========================
class SeriesStore:
    """Per-symbol rolling stores for price/volume and arbitrary series."""
    def __init__(self, maxlen: int = 10_000):
        self.maxlen = maxlen
        self.price: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.vol:   Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.custom: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def add_tick(self, symbol: str, price: Number, volume: Number | None = None, ts_ms: Optional[int] = None):
        if price and price > 0:
            self.price[symbol].append(float(price))
        if volume is not None:
            self.vol[symbol].append(float(volume))

    def push(self, symbol: str, name: str, value: Number):
        self.custom[(symbol, name)].append(float(value))

    # ---- indicators ----
    def sma(self, symbol: str, n: int) -> float:
        q = self.price[symbol]
        if len(q) < n or n <= 0: return float('nan')
        return sum(list(q)[-n:]) / n

    def ema(self, symbol: str, n: int) -> float:
        q = self.price[symbol]
        if len(q) < n or n <= 0: return float('nan')
        alpha = 2.0 / (n + 1.0)
        e = list(q)[-n]
        for v in list(q)[-n+1:]:
            e = alpha * v + (1 - alpha) * e
        return e

    def rsi(self, symbol: str, n: int = 14) -> float:
        q = self.price[symbol]
        if len(q) < n + 1 or n <= 0: return float('nan')
        gains = losses = 0.0
        arr = list(q)[-n-1:]
        for i in range(1, len(arr)):
            d = arr[i] - arr[i-1]
            if d >= 0: gains += d
            else: losses -= d
        if losses == 0: return 100.0
        rs = (gains / n) / (losses / n)
        return 100.0 - (100.0 / (1.0 + rs))

    def zscore(self, symbol: str, n: int = 50) -> float:
        q = self.price[symbol]
        if len(q) < n or n <= 1: return float('nan')
        window = list(q)[-n:]
        mu = sum(window) / n
        var = sum((x - mu) ** 2 for x in window) / (n - 1)
        sd = math.sqrt(max(var, 1e-12))
        return (window[-1] - mu) / sd

    def pct_chg(self, symbol: str, n: int = 1) -> float:
        q = self.price[symbol]
        if len(q) < n + 1 or n <= 0: return float('nan')
        prev = list(q)[-n-1]
        cur = q[-1]
        return (cur - prev) / prev if prev else float('nan')

    def vwap_win(self, symbol: str, n: int = 20) -> float:
        p = self.price[symbol]; v = self.vol[symbol]
        if len(p) < n or len(v) < n: return float('nan')
        ps = list(p)[-n:]; vs = list(v)[-n:]
        tv = sum(vs)
        if tv <= 0: return float('nan')
        return sum(pi * vi for pi, vi in zip(ps, vs)) / tv

# ===================== Safe expression evaluator ============================
class SafeEval(ast.NodeVisitor):
    """
    Evaluate a restricted expression tree against a context dict.
    Allowed: literals, names (from ctx), bool ops, compare, math ops,
             function calls to whitelisted callables (from ctx['__funcs__']).
    """
    ALLOWED_NODES = {
        ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
        ast.Name, ast.Load, ast.Call, ast.Constant, ast.IfExp,
        ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
    }

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    def visit(self, node):
        if type(node) not in self.ALLOWED_NODES:
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        return super().visit(node)

    def eval(self, expr: str):
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    # nodes
    def visit_Constant(self, node: ast.Constant): return node.value
    def visit_Name(self, node: ast.Name):
        if node.id in self.ctx: return self.ctx[node.id]
        raise NameError(f"name '{node.id}' is not defined")
    def visit_UnaryOp(self, node: ast.UnaryOp):
        v = self.visit(node.operand)
        return -v if isinstance(node.op, ast.USub) else (+v if isinstance(node.op, ast.UAdd) else (not v))
    def visit_BoolOp(self, node: ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not self.visit(v): return False
            return True
        elif isinstance(node.op, ast.Or):
            for v in node.values:
                if self.visit(v): return True
            return False
        raise ValueError("unsupported BoolOp")
    def visit_BinOp(self, node: ast.BinOp):
        a, b = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.FloorDiv): return a // b
        if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node.op, ast.Pow): return a ** b
        raise ValueError("unsupported BinOp")
    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            ok = (isinstance(op, ast.Lt) and left < right) or \
                 (isinstance(op, ast.Gt) and left > right) or \
                 (isinstance(op, ast.LtE) and left <= right) or \
                 (isinstance(op, ast.GtE) and left >= right) or \
                 (isinstance(op, ast.Eq) and left == right) or \
                 (isinstance(op, ast.NotEq) and left != right)
            if not ok: return False
            left = right
        return True
    def visit_IfExp(self, node: ast.IfExp):
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)
    def visit_Call(self, node: ast.Call):
        fn_name = getattr(node.func, "id", None)
        if not fn_name: raise ValueError("only simple function calls allowed")
        funcs = self.ctx.get("__funcs__", {})
        if fn_name not in funcs: raise NameError(f"function '{fn_name}' not allowed")
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return funcs[fn_name](*args, **kwargs) # type: ignore

# ===================== Rule model & engine ==================================
@dataclass
class Rule:
    name: str
    when: str
    then: List[Dict[str, Any]] = field(default_factory=list)  # list of actions
    otherwise: List[Dict[str, Any]] = field(default_factory=list)
    min_interval_ms: int = 0
    cooldown_ms: int = 0
    once: bool = False
    enabled: bool = True
    _last_fire_ms: int = 0
    _fired_once: bool = False

class StrategyDSLEngine:
    """
    Tick → evaluate rules → execute actions via injected callbacks.

    Callbacks you provide:
      - order_cb(symbol, side, qty=None, order_type="market", limit_price=None, extra=None)
      - emit_signal_cb(score: float)
      - log_cb(level, message, **kwargs)  (optional)
    """
    def __init__(
        self,
        *,
        name: str,
        default_qty: float = 1.0,
        series_maxlen: int = 20_000,
        order_cb: Callable[..., None],
        emit_signal_cb: Callable[[float], None],
        log_cb: Optional[Callable[..., None]] = None,
    ):
        self.name = name
        self.default_qty = float(default_qty)
        self.series = SeriesStore(maxlen=series_maxlen)
        self.order_cb = order_cb
        self.emit_signal_cb = emit_signal_cb
        self.log = log_cb or (lambda lvl, msg, **kw: None)
        self.state: Dict[str, Any] = {}    # user variables across rules
        self.clock_skew_ms = 0

        # compiled rules
        self.rules: List[Rule] = []

    # ---- loading rules ------------------------------------------------------
    def load(self, rules: Union[str, List[Dict[str, Any]]]) -> None:
        """
        Accepts:
          - YAML/JSON string
          - List[dict] already parsed
        Each rule dict:
          {name, when, then:[{action: "order"|"emit_signal"|"set"|"log", ...}],
           otherwise:[...], min_interval_ms?, cooldown_ms?, once?, enabled?}
        """
        if isinstance(rules, str):
            parsed = None
            # try YAML then JSON
            if HAVE_YAML:
                try:
                    parsed = yaml.safe_load(rules)  # type: ignore
                except Exception:
                    parsed = None
            if parsed is None:
                parsed = json.loads(rules)
            rules_list = parsed
        else:
            rules_list = rules

        self.rules.clear()
        for r in rules_list or []:
            self.rules.append(Rule(
                name=str(r.get("name") or f"rule_{len(self.rules)+1}"),
                when=str(r.get("when") or "False"),
                then=list(r.get("then") or []),
                otherwise=list(r.get("otherwise") or []),
                min_interval_ms=int(r.get("min_interval_ms") or 0),
                cooldown_ms=int(r.get("cooldown_ms") or 0),
                once=bool(r.get("once") or False),
                enabled=bool(r.get("enabled", True)),
            ))

    # ---- evaluation context -------------------------------------------------
    def _ctx(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        px  = float(tick.get("price") or tick.get("p") or 0.0)
        vol = float(tick.get("volume") or tick.get("v") or 0.0)
        ts  = int(tick.get("ts_ms") or tick.get("t") or time.time()*1000)

        # keep series
        if sym and px > 0:
            self.series.add_tick(sym, px, vol, ts)

        # math helpers
        funcs = {
            "sma": lambda n: self.series.sma(sym, int(n)),
            "ema": lambda n: self.series.ema(sym, int(n)),
            "rsi": lambda n=14: self.series.rsi(sym, int(n)),
            "zscore": lambda n=50: self.series.zscore(sym, int(n)),
            "pct_chg": lambda n=1: self.series.pct_chg(sym, int(n)),
            "vwap_win": lambda n=20: self.series.vwap_win(sym, int(n)),
            "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        }

        # commonly used vars
        ctx = {
            # tick fields
            "symbol": sym, "price": px, "volume": vol, "ts_ms": ts,
            # state bag (read/write via set)
            **self.state,
            # consts
            "default_qty": self.default_qty,
            # function table (for SafeEval)
            "__funcs__": funcs,
            # allow direct calls (e.g., sma(20))
            **funcs,
        }
        return ctx

    # ---- execute actions ----------------------------------------------------
    def _exec_actions(self, actions: List[Dict[str, Any]], ctx: Dict[str, Any]) -> None:
        for a in actions or []:
            act = str(a.get("action") or "").lower()
            if act == "order":
                symbol = str(a.get("symbol") or ctx.get("symbol"))
                side   = str(a.get("side") or "buy").lower()
                qty_expr = a.get("qty")
                qty = float(qty_expr if isinstance(qty_expr, (int,float)) else ctx.get("default_qty")) # type: ignore
                order_type = str(a.get("order_type") or "market")
                limit_price = a.get("limit_price")
                if isinstance(limit_price, str):
                    limit_price = float(SafeEval(ctx).eval(limit_price))
                extra = a.get("extra") or {}
                self.order_cb(symbol, side, qty=qty, order_type=order_type, limit_price=limit_price, extra=extra)
                self.log("info", f"order {symbol} {side} {qty}", rule=a)

            elif act == "emit_signal":
                score_expr = a.get("score", 0.0)
                score = float(score_expr if isinstance(score_expr, (int,float)) else SafeEval(ctx).eval(str(score_expr)))
                self.emit_signal_cb(max(-1.0, min(1.0, score)))
                self.log("info", f"emit_signal {score}", rule=a)

            elif act == "set":
                # set a state variable to an expression
                name = str(a.get("name"))
                val  = a.get("value")
                if isinstance(val, str):
                    val = SafeEval(ctx).eval(val)
                self.state[name] = val
                self.log("debug", f"set {name}={val}", rule=a)

            elif act == "log":
                msg = str(a.get("message") or "")
                try:
                    msg = str(SafeEval(ctx).eval(msg)) if a.get("eval", False) else msg
                except Exception:
                    pass
                self.log("info", msg, rule=a)

            else:
                self.log("warn", f"unknown action: {act}", rule=a)

    # ---- main hook ----------------------------------------------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        ctx = self._ctx(tick)
        now = int(ctx["ts_ms"])
        for r in self.rules:
            if not r.enabled: 
                continue
            if r.once and r._fired_once:
                continue
            # rate limit
            if r.min_interval_ms > 0 and (now - r._last_fire_ms) < r.min_interval_ms:
                continue
            # cooldown
            if r.cooldown_ms > 0 and (now - r._last_fire_ms) < r.cooldown_ms:
                continue

            try:
                ok = bool(SafeEval(ctx).eval(r.when))
            except Exception as e:
                self.log("error", f"eval error in rule '{r.name}': {e}")
                continue

            if ok:
                self._exec_actions(r.then, ctx)
                r._last_fire_ms = now
                if r.once:
                    r._fired_once = True
            else:
                if r.otherwise:
                    self._exec_actions(r.otherwise, ctx)

# ===================== Convenience: Strategy adapter =========================
class DSLStrategyAdapter:
    """
    Glue to your Strategy base:
      dsls = DSLStrategyAdapter(strategy=self_strategy_instance, name="my_dsl", default_qty=1)
      dsls.load(yaml_or_list)
      dsls.on_tick(tick)
    """
    def __init__(self, *, strategy, name: str, default_qty: float = 1.0):
        self.strategy = strategy
        self.engine = StrategyDSLEngine(
            name=name,
            default_qty=default_qty,
            order_cb=self._order,
            emit_signal_cb=self._emit_signal,
            log_cb=self._log
        )

    def load(self, rules: Union[str, List[Dict[str, Any]]]): self.engine.load(rules)
    def on_tick(self, tick: Dict[str, Any]): self.engine.on_tick(tick)

    # callbacks map to your Strategy API
    def _order(self, symbol, side, qty=None, order_type="market", limit_price=None, extra=None):
        self.strategy.order(symbol, side, qty, order_type=order_type, limit_price=limit_price, extra=extra)

    def _emit_signal(self, score: float):
        try:
            self.strategy.emit_signal(score)
        except Exception:
            pass

    def _log(self, level: str, message: str, **kw):
        # optionally push to redis/logs; here we just print
        # print(f"[DSL][{level}] {message}")
        pass

# ===================== Example rules (YAML/JSON) ============================
EXAMPLE_YAML = """
- name: buy_the_dip
  when: "price < sma(50) * 0.99 and rsi(14) < 30"
  min_interval_ms: 2000
  then:
    - { action: "order", side: "buy" }
    - { action: "emit_signal", score: "-(price - sma(50))/sma(50) * 5" }
  otherwise:
    - { action: "emit_signal", score: "0" }

- name: take_profit
  when: "price > ema(50) * 1.01"
  cooldown_ms: 5000
  then:
    - { action: "order", side: "sell", qty: 1 }
    - { action: "log", message: "TP fired for {symbol} @ {price}" }

- name: momentum_entry_once
  when: "zscore(100) > 2.0 and pct_chg(5) > 0.01"
  once: true
  then:
    - { action: "order", side: "buy", qty: 2 }
"""

# ===================== Quick self-test ======================================
if __name__ == "__main__":
    # Tiny dry run
    class _DummyStrat:
        def order(self, *a, **k): print("ORDER", a, k)
        def emit_signal(self, s): print("SIGNAL", s)

    d = DSLStrategyAdapter(strategy=_DummyStrat(), name="demo", default_qty=1.0)
    d.load(EXAMPLE_YAML)

    # fake ticks
    import random, time
    px = 100.0
    for i in range(300):
        px += random.uniform(-0.5, 0.7)
        tick = {"symbol": "AAPL", "price": px, "volume": 1000+random.randint(-50,50), "ts_ms": int(time.time()*1000)}
        d.on_tick(tick)
        time.sleep(0.01)