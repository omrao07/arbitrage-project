# backend/engine/strategy_dsl.py
from __future__ import annotations

import ast, math, time, json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque

Number = Union[int, float]

# ===================== Rolling storage & indicators ==========================
class SeriesStore:
    """Per-symbol rolling stores for price/volume and notional (for exact VWAP)."""
    def __init__(self, maxlen: int = 50_000):
        self.maxlen = maxlen
        self.price: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.vol:   Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self.notional: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))  # price*size

    def add_tick(self, symbol: str, price: Number, size: Number | None = None):
        if price and price > 0:
            self.price[symbol].append(float(price))
        v = float(size or 0.0)
        self.vol[symbol].append(v)
        self.notional[symbol].append(v * float(price or 0.0))

    # ---- helpers ----
    def _tail(self, q: deque, n: int) -> List[float]:
        if n <= 0: return []
        if len(q) < n: return list(q)
        return list(q)[-n:]

    # ---- indicators ----
    def sma(self, s: str, n: int) -> float:
        w = self._tail(self.price[s], n)
        return float("nan") if not w or len(w) < n else sum(w)/n

    def ema(self, s: str, n: int) -> float:
        w = self._tail(self.price[s], n)
        if len(w) < n or n <= 0: return float("nan")
        a = 2.0/(n+1.0); e = w[0]
        for x in w[1:]: e = a*x + (1-a)*e
        return e

    def rsi(self, s: str, n: int = 14) -> float:
        w = self._tail(self.price[s], n+1)
        if len(w) < n+1: return float("nan")
        gains = losses = 0.0
        for i in range(1, len(w)):
            d = w[i]-w[i-1]
            gains += max(0.0, d)
            losses += max(0.0, -d)
        if losses == 0: return 100.0
        rs = (gains/n) / (losses/n)
        return 100.0 - 100.0/(1.0+rs)

    def zscore(self, s: str, n: int = 50) -> float:
        w = self._tail(self.price[s], n)
        if len(w) < n or n <= 1: return float("nan")
        mu = sum(w)/n
        var = sum((x-mu)**2 for x in w)/(n-1)
        sd = math.sqrt(max(var, 1e-12))
        return (w[-1]-mu)/sd

    def vwap_win(self, s: str, n: int = 20) -> float:
        v = self._tail(self.vol[s], n)
        no = self._tail(self.notional[s], n)
        tv = sum(v)
        return float("nan") if tv <= 0 else sum(no)/tv

    def atr(self, s: str, n: int = 14) -> float:
        # simple proxy ATR using close-to-close absolute change
        w = self._tail(self.price[s], n+1)
        if len(w) < n+1: return float("nan")
        tr = [abs(w[i]-w[i-1]) for i in range(1, len(w))]
        return sum(tr)/len(tr)

    def pct_chg(self, s: str, n: int = 1) -> float:
        w = self._tail(self.price[s], n+1)
        if len(w) < n+1: return float("nan")
        return (w[-1]-w[0]) / (w[0] if w[0] else float("nan"))

# ===================== Safe expression evaluator ============================
class SafeEval(ast.NodeVisitor):
    """Restricted evaluator for rule expressions."""
    ALLOWED = {
        ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
        ast.Name, ast.Load, ast.Call, ast.Constant, ast.IfExp,
        ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
    }
    def __init__(self, ctx: Dict[str, Any]): self.ctx = ctx
    def visit(self, node):
        if type(node) not in self.ALLOWED:
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        return super().visit(node)
    def eval(self, expr: str): return self.visit(ast.parse(expr, mode="eval").body)
    def visit_Constant(self, n): return n.value
    def visit_Name(self, n):
        if n.id in self.ctx: return self.ctx[n.id]
        raise NameError(f"name '{n.id}' not defined")
    def visit_UnaryOp(self, n):
        v = self.visit(n.operand)
        if isinstance(n.op, ast.USub): return -v
        if isinstance(n.op, ast.UAdd): return +v
        if isinstance(n.op, ast.Not):  return not v
        raise ValueError("bad UnaryOp")
    def visit_BoolOp(self, n):
        if isinstance(n.op, ast.And):
            for v in n.values:
                if not self.visit(v): return False
            return True
        if isinstance(n.op, ast.Or):
            for v in n.values:
                if self.visit(v): return True
            return False
        raise ValueError("bad BoolOp")
    def visit_BinOp(self, n):
        a, b = self.visit(n.left), self.visit(n.right)
        if isinstance(n.op, ast.Add): return a+b
        if isinstance(n.op, ast.Sub): return a-b
        if isinstance(n.op, ast.Mult): return a*b
        if isinstance(n.op, ast.Div): return a/b
        if isinstance(n.op, ast.FloorDiv): return a//b
        if isinstance(n.op, ast.Mod): return a % b
        if isinstance(n.op, ast.Pow): return a ** b
        raise ValueError("bad BinOp")
    def visit_Compare(self, n):
        left = self.visit(n.left)
        for op, comp in zip(n.ops, n.comparators):
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
    def visit_IfExp(self, n): return self.visit(n.body) if self.visit(n.test) else self.visit(n.orelse)
    def visit_Call(self, n):
        fn = getattr(n.func, "id", None)
        funcs = self.ctx.get("__funcs__", {})
        if not fn or fn not in funcs: raise NameError(f"function '{fn}' not allowed")
        args = [self.visit(a) for a in n.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in n.keywords}
        return funcs[fn](*args, **kwargs)

# ===================== Rule model & engine ==================================
@dataclass
class Rule:
    name: str
    when: str
    then: List[Dict[str, Any]] = field(default_factory=list)
    otherwise: List[Dict[str, Any]] = field(default_factory=list)
    min_interval_ms: int = 0      # rate limit
    cooldown_ms: int = 0          # wait after fire
    once: bool = False
    enabled: bool = True
    _last_ms: int = 0
    _fired_once: bool = False

class StrategyDSLEngine:
    """
    Tick → evaluate rules → execute actions via callbacks you pass in:
      - order_cb(symbol, side, qty=None, order_type="market", limit_price=None, extra=None)
      - emit_signal_cb(score: float)
      - alert_cb(level, message, **meta)  (optional; no-op if not provided)
      - position_cb(symbol) -> dict(qty, avg_px) (optional; returns 0 if missing)
    """
    def __init__(
        self,
        *,
        name: str,
        default_qty: float = 1.0,
        series_maxlen: int = 50_000,
        order_cb: Callable[..., None],
        emit_signal_cb: Callable[[float], None],
        alert_cb: Optional[Callable[[str, str], None]] = None,
        position_cb: Optional[Callable[[str], Dict[str, float]]] = None,
        logger: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ):
        self.name = name
        self.default_qty = float(default_qty)
        self.series = SeriesStore(series_maxlen)
        self.order_cb = order_cb
        self.emit_signal_cb = emit_signal_cb
        self.alert = alert_cb or (lambda lvl, msg, **kw: None)
        self.position_cb = position_cb or (lambda s: {"qty": 0.0, "avg_px": float("nan")})
        self.log = logger or (lambda lvl, msg, extra=None: None)
        self.rules: List[Rule] = []
        self.state: Dict[str, Any] = {}

    # ---- load rules (YAML/JSON/list[dict]) ----------------------------------
    def load(self, rules: Union[str, List[Dict[str, Any]]]) -> None:
        parsed = rules
        if isinstance(rules, str):
            try:
                import yaml  # type: ignore
                parsed = yaml.safe_load(rules)
            except Exception:
                parsed = json.loads(rules)
        self.rules.clear()
        for r in (parsed or []):
            self.rules.append(Rule(
                name=str(r.get("name") or f"rule_{len(self.rules)+1}"), # type: ignore
                when=str(r.get("when") or "False"), # type: ignore
                then=list(r.get("then") or []), # type: ignore
                otherwise=list(r.get("otherwise") or []), # type: ignore
                min_interval_ms=int(r.get("min_interval_ms") or 0), # type: ignore
                cooldown_ms=int(r.get("cooldown_ms") or 0), # type: ignore
                once=bool(r.get("once") or False), # type: ignore
                enabled=bool(r.get("enabled", True)), # type: ignore
            ))

    # ---- evaluation context -------------------------------------------------
    def _ctx(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        s = (tick.get("symbol") or tick.get("s") or "").upper()
        px = float(tick.get("price") or tick.get("p") or 0.0)
        sz = float(tick.get("size") or tick.get("q") or 0.0)
        ts = int(tick.get("ts_ms") or tick.get("t") or time.time()*1000)
        if s and px > 0:
            self.series.add_tick(s, px, sz)
        pos = self.position_cb(s) or {"qty": 0.0, "avg_px": float("nan")}
        qty = float(pos.get("qty") or 0.0)
        avg = float(pos.get("avg_px") or float("nan"))

        funcs = {
            # indicators
            "sma": lambda n: self.series.sma(s, int(n)),
            "ema": lambda n: self.series.ema(s, int(n)),
            "rsi": lambda n=14: self.series.rsi(s, int(n)),
            "zscore": lambda n=50: self.series.zscore(s, int(n)),
            "vwap_win": lambda n=20: self.series.vwap_win(s, int(n)),
            "atr": lambda n=14: self.series.atr(s, int(n)),
            "pct_chg": lambda n=1: self.series.pct_chg(s, int(n)),
            # math
            "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        }
        ctx = {
            "symbol": s, "price": px, "size": sz, "ts_ms": ts,
            "default_qty": self.default_qty,
            "pos_qty": qty, "pos_avg_px": avg,
            "pnl_unreal": (px-avg)*qty if qty and avg==avg else 0.0,
            "__funcs__": funcs, **funcs,
            **self.state,
        }
        return ctx

    # ---- actions ------------------------------------------------------------
    def _exec(self, actions: List[Dict[str, Any]], ctx: Dict[str, Any]) -> None:
        for a in actions or []:
            act = str(a.get("action") or "").lower()
            if act == "order":
                symbol = str(a.get("symbol") or ctx["symbol"])
                side   = str(a.get("side") or "buy").lower()
                qty    = a.get("qty")
                qty_f  = float(qty) if isinstance(qty, (int,float)) else float(ctx.get("default_qty", 1.0))
                typ    = str(a.get("order_type") or "market")
                lp     = a.get("limit_price")
                if isinstance(lp, str): lp = float(SafeEval(ctx).eval(lp))
                extra  = a.get("extra") or {}
                self.order_cb(symbol, side, qty=qty_f, order_type=typ, limit_price=lp, extra=extra)
                self.log("info", f"order({symbol},{side},{qty_f},{typ},{lp})") # type: ignore

            elif act == "emit_signal":
                val = a.get("score", 0.0)
                score = float(val if isinstance(val, (int,float)) else SafeEval(ctx).eval(str(val)))
                self.emit_signal_cb(max(-1.0, min(1.0, score)))

            elif act == "set":
                name = str(a.get("name")); val = a.get("value")
                if isinstance(val, str):
                    val = SafeEval(ctx).eval(val)
                self.state[name] = val

            elif act == "log":
                msg = str(a.get("message") or "")
                try:
                    msg = str(SafeEval(ctx).eval(msg)) if a.get("eval") else msg
                except Exception: pass
                self.log("info", msg) # type: ignore

            elif act == "alert":
                lvl = str(a.get("level") or "info")
                msg = str(a.get("message") or "")
                self.alert(lvl, msg, rule=a, ctx=ctx) # type: ignore

            else:
                self.log("warn", f"unknown action: {act}") # type: ignore

    # ---- main loop hook -----------------------------------------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        ctx = self._ctx(tick)
        now = int(ctx["ts_ms"])
        for r in self.rules:
            if not r.enabled: continue
            if r.once and r._fired_once: continue
            if r.min_interval_ms and (now - r._last_ms) < r.min_interval_ms: continue
            if r.cooldown_ms and (now - r._last_ms) < r.cooldown_ms: continue
            try:
                ok = bool(SafeEval(ctx).eval(r.when))
            except Exception as e:
                self.log("error", f"eval error in '{r.name}': {e}") # type: ignore
                continue
            if ok:
                self._exec(r.then, ctx)
                r._last_ms = now
                if r.once: r._fired_once = True
            else:
                if r.otherwise: self._exec(r.otherwise, ctx)

# ===================== Adapter to your Strategy base =========================
class DSLStrategyAdapter:
    """
    Glue to your Strategy base (from strategy_base.py):
        dsl = DSLStrategyAdapter(strategy=self, name="alpha_dsl", default_qty=1)
        dsl.load(open("configs/strategies/alpha.yaml").read())
        dsl.on_tick(tick)
    """
    def __init__(self, *, strategy, name: str, default_qty: float = 1.0):
        self.strategy = strategy
        self.engine = StrategyDSLEngine(
            name=name,
            default_qty=default_qty,
            order_cb=self._order,
            emit_signal_cb=self._emit_signal,
            alert_cb=self._alert,
            position_cb=self._position,
            logger=lambda lvl, msg, extra=None: None,
        )

    def load(self, rules): self.engine.load(rules)
    def on_tick(self, tick): self.engine.on_tick(tick)

    # ---- callbacks mapping to Strategy API ----
    def _order(self, *a, **k): self.strategy.order(*a, **k)
    def _emit_signal(self, s: float): 
        try: self.strategy.emit_signal(s)
        except Exception: pass
    def _alert(self, level: str, message: str, **meta):
        # plug your alerts stream here if you like
        # e.g., publish_stream("alerts.events", {...})
        pass
    def _position(self, symbol: str) -> Dict[str, float]:
        # If your strategy stores positions in Redis/DB, fetch them here.
        # Default to flat.
        return {"qty": 0.0, "avg_px": float("nan")}

# ===================== Example rules ========================================
EXAMPLE_YAML = """
- name: buy_dip_rsi
  when: "price < sma(50) * 0.99 and rsi(14) < 30"
  min_interval_ms: 2000
  then:
    - { action: "order", side: "buy", qty: 1 }
    - { action: "emit_signal", score: "-(price - sma(50))/sma(50) * 5" }
  otherwise:
    - { action: "emit_signal", score: "0" }

- name: stop_loss_take_profit
  when: "pos_qty != 0 and ( (pos_qty > 0 and (price <= pos_avg_px * 0.98 or price >= pos_avg_px * 1.02)) or (pos_qty < 0 and (price >= pos_avg_px * 1.02 or price <= pos_avg_px * 0.98)) )"
  cooldown_ms: 5000
  then:
    - action: order
      side: "{{ 'sell' if pos_qty > 0 else 'buy' }}"
      qty: 1
"""

# Tiny self-test (optional)
if __name__ == "__main__":
    class _DummyStrat:
        def order(self, *a, **k): print("ORDER", a, k)
        def emit_signal(self, s): print("SIGNAL", s)
    d = DSLStrategyAdapter(strategy=_DummyStrat(), name="demo", default_qty=1.0)
    # Load rules from EXAMPLE_YAML but fix Jinja-like side:
    rules = [
        {"name":"buy","when":"price < sma(10) * 0.99","then":[{"action":"order","side":"buy","qty":1}]},
        {"name":"tp","when":"pct_chg(5) > 0.01","then":[{"action":"order","side":"sell","qty":1}]},
    ]
    d.load(rules)
    import random
    px=100.0
    for i in range(200):
        px += random.uniform(-0.5,0.6)
        d.on_tick({"symbol":"AAPL","price":px,"size":random.randint(1,5),"ts_ms":int(time.time()*1000)})
        time.sleep(0.01)