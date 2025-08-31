# backend/ai/agents/core/quality.py
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# Optional helpers (safe shims if not present)
try:
    from .toolbelt import now_ms, percentile # type: ignore
except Exception:
    def now_ms() -> int: return int(time.time()*1000)
    def percentile(values: List[float], q: float) -> float:
        if not values: return float("nan")
        v = sorted(values); k = (len(v)-1)*(q/100.0); f = int(k); c = min(f+1,len(v)-1)
        return v[f] if f==c else v[f] + (v[c]-v[f])*(k-f)

# ----------------------------- Data models -----------------------------
@dataclass
class QualityItem:
    name: str
    ok: bool
    category: str                 # "data" | "execution" | "risk" | "model" | "infra"
    took_ms: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    trace: Optional[str] = None
    severity: str = "warn"        # "info" | "warn" | "crit"

@dataclass
class QualityReport:
    suite: str
    started_at: int
    finished_at: int
    items: List[QualityItem]
    ok: bool
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

# ----------------------------- Registry -----------------------------
_CheckFn = Callable[[], Tuple[bool, Dict[str, Any], str]]

class QualityRegistry:
    """
    Register lightweight checks and run them as suites.
    A check returns (ok, metrics, message).
    """
    def __init__(self):
        self._checks: Dict[str, Tuple[str, str, _CheckFn]] = {}
        # name -> (category, severity, fn)

    def register(self, name: str, fn: _CheckFn, *, category: str, severity: str = "warn") -> None:
        self._checks[name] = (category, severity, fn)

    def list(self, category: Optional[str] = None) -> List[str]:
        if not category:
            return list(self._checks.keys())
        return [n for n,(c,_,_) in self._checks.items() if c == category]

    def run(self, names: Iterable[str]) -> List[QualityItem]:
        items: List[QualityItem] = []
        for n in names:
            cat, sev, fn = self._checks[n]
            t0 = now_ms()
            ok = False; metrics: Dict[str, Any] = {}; msg = ""; tr = None
            try:
                ok, metrics, msg = fn()
            except Exception as e:
                ok = False
                msg = f"{e.__class__.__name__}: {e}"
                tr = traceback.format_exc()
            items.append(QualityItem(
                name=n, ok=ok, category=cat, took_ms=now_ms()-t0,
                metrics=metrics, message=msg, trace=tr, severity=sev
            ))
        return items

    def run_suite(self, *, suite: str, include: Optional[List[str]] = None,
                  category: Optional[str] = None) -> QualityReport:
        start = now_ms()
        if include:
            names = include
        else:
            names = self.list(category)
        items = self.run(names)
        end = now_ms()
        ok_all = all(i.ok or i.severity=="info" for i in items)
        summary = {
            "total": len(items),
            "passed": sum(1 for i in items if i.ok),
            "failed": sum(1 for i in items if not i.ok and i.severity!="info"),
            "median_ms": percentile([i.took_ms for i in items], 50) if items else 0,
            "by_category": _rollup_by(items, key="category"),
            "by_severity": _rollup_by(items, key="severity"),
        }
        return QualityReport(suite=suite, started_at=start, finished_at=end, items=items, ok=ok_all, summary=summary)

def _rollup_by(items: List[QualityItem], *, key: str) -> Dict[str, Any]:
    agg: Dict[str, Dict[str, int]] = {}
    for i in items:
        k = getattr(i, key)
        d = agg.setdefault(k, {"total":0, "pass":0, "fail":0})
        d["total"] += 1
        d["pass"] += 1 if i.ok else 0
        d["fail"] += 0 if i.ok else 1
    return agg

# ----------------------------- Built-in checks -----------------------------
def builtins(reg: QualityRegistry) -> None:
    """
    Register a sensible starter set for your stack. All checks are dependency-free
    and accept data via provider callables you can override below.
    """

    # ---- Providers (override these lambdas from your app as you wire things) ----
    # MARKET DATA
    reg.price_provider = lambda sym="AAPL": {"symbol": sym, "ts_ms": now_ms(), "last": 100.0}  # type: ignore
    reg.candles_provider = lambda sym="AAPL", n=100: [{"t": now_ms()-i*60_000, "o":100,"h":101,"l":99,"c":100,"v":1000} for i in range(n)][::-1]  # type: ignore

    # EXECUTION / BROKERS
    reg.broker_health = lambda: {"connected": True, "name": "paper"}  # type: ignore
    reg.order_roundtrip = lambda: {"ok": True, "lat_ms": 15, "slip_bps": 0.0}  # type: ignore

    # RISK / MODEL
    reg.var_provider = lambda: {"VaR": 0.02, "stdev": 0.15, "mean": 0.001}  # type: ignore
    reg.hrp_weights = lambda: {"sum": 1.0, "negatives": 0, "n": 15}  # type: ignore

    # ---------------- Data quality ----------------
    def q_price_staleness(max_stale_ms: int = 5_000) -> _CheckFn:
        def _fn():
            p = reg.price_provider()  # type: ignore
            age = now_ms() - int(p.get("ts_ms", 0))
            ok = age <= max_stale_ms
            return ok, {"age_ms": age, "symbol": p.get("symbol")}, f"age={age}ms (≤{max_stale_ms})"
        return _fn

    def q_candle_sanity() -> _CheckFn:
        def _fn():
            cd = reg.candles_provider()  # type: ignore
            if not cd: return False, {}, "no candles"
            ok = True; issues = 0
            for k in ("o","h","l","c"):
                if any((not math.isfinite(x.get(k, float("nan")))) for x in cd):
                    ok = False; issues += 1
            # OHLC logic: l ≤ {o,c} ≤ h
            for x in cd:
                l,h,o,c = x["l"],x["h"],x["o"],x["c"]
                if not (l <= o <= h and l <= c <= h and l <= h):
                    ok=False; issues+=1; break
            # time monotonic increasing
            t_prev = -1
            for x in cd:
                if x["t"] <= t_prev: ok=False; issues+=1; break
                t_prev = x["t"]
            return ok, {"n": len(cd), "issues": issues}, f"{len(cd)} candles, issues={issues}"
        return _fn

    # ---------------- Execution quality ----------------
    def q_broker_connectivity() -> _CheckFn:
        def _fn():
            h = reg.broker_health()  # type: ignore
            ok = bool(h.get("connected"))
            return ok, h, f"broker={h.get('name')} connected={ok}"
        return _fn

    def q_order_roundtrip(max_lat_ms: int = 250, max_slip_bps: float = 15.0) -> _CheckFn:
        def _fn():
            r = reg.order_roundtrip()  # type: ignore
            lat = float(r.get("lat_ms", 9e9)); slip = float(r.get("slip_bps", 9e9))
            ok = (lat <= max_lat_ms) and (slip <= max_slip_bps) and bool(r.get("ok", False))
            return ok, {"lat_ms": lat, "slip_bps": slip}, f"lat={lat}ms (≤{max_lat_ms}), slip={slip}bps (≤{max_slip_bps})"
        return _fn

    # ---------------- Risk / Model quality ----------------
    def q_var_sanity(max_var: float = 0.25, max_sigma: float = 1.0) -> _CheckFn:
        def _fn():
            r = reg.var_provider()  # type: ignore
            var = float(r.get("VaR", 0)); sigma = float(r.get("stdev", 0))
            ok = (0 <= var <= max_var) and (0 <= sigma <= max_sigma)
            return ok, {"VaR": var, "stdev": sigma}, f"VaR={var:.4f} σ={sigma:.4f}"
        return _fn

    def q_hrp_weights() -> _CheckFn:
        def _fn():
            w = reg.hrp_weights()  # type: ignore
            s = float(w.get("sum", 0)); neg = int(w.get("negatives", 0)); n=int(w.get("n",0))
            ok = abs(s - 1.0) < 1e-6 and neg == 0 and n > 0
            return ok, {"sum": s, "negatives": neg, "n": n}, f"sum≈{s:.6f} negatives={neg} n={n}"
        return _fn

    # Register all defaults
    reg.register("data:price_staleness",   q_price_staleness(), category="data",      severity="crit")
    reg.register("data:candles_sanity",    q_candle_sanity(),   category="data",      severity="warn")
    reg.register("exec:broker_connect",    q_broker_connectivity(), category="execution", severity="crit")
    reg.register("exec:order_roundtrip",   q_order_roundtrip(), category="execution", severity="crit")
    reg.register("risk:var_sanity",        q_var_sanity(),      category="risk",      severity="warn")
    reg.register("model:hrp_weights",      q_hrp_weights(),     category="model",     severity="warn")

# ----------------------------- Public API -----------------------------
def new_registry(with_builtins: bool = True) -> QualityRegistry:
    reg = QualityRegistry()
    if with_builtins:
        builtins(reg)
    return reg

# ----------------------------- How to wire (examples) -----------------------------
"""
# 1) Create registry and override providers with your real functions
reg = new_registry()

# Example: wire to your adapters
from backend.ai.agents.connectors.brokers import paperbroker
reg.broker_health = lambda: {"connected": paperbroker.is_connected(), "name": "paper"}  # type: ignore
reg.order_roundtrip = lambda: _probe_roundtrip(paperbroker)  # type: ignore

from backend.data.quotes import latest_price, get_candles
reg.price_provider = lambda sym="AAPL": latest_price(sym)    # must return {"symbol","ts_ms","last"}
reg.candles_provider = lambda sym="AAPL", n=200: get_candles(sym, n)

from backend.risk.var_engine import snapshot_var
reg.var_provider = lambda: snapshot_var()                    # -> {"VaR","stdev","mean"}

from backend.research.portfolio import hrp_snapshot
reg.hrp_weights = lambda: hrp_snapshot()                     # -> {"sum","negatives","n"}

# 2) Run a suite
report = reg.run_suite(suite="pre-market")
print(report.to_json())

# 3) Export to file or emit on your bus
with open("quality_report.json","w") as f:
    f.write(report.to_json())

# Optional helper for order round-trip (example):
def _probe_roundtrip(broker):
    import random
    sym = "AAPL"; px = broker.last_price(sym) or 100.0
    qty = 1
    oid = broker.submit_order(sym, "buy", qty, order_type="limit", limit_price=px)
    t0 = now_ms()
    # In paperbroker, touching price fills immediately; in live, poll a bit (pseudo):
    time.sleep(0.05)
    lat = now_ms()-t0
    slip_bps = 0.0
    # If broker exposes fills, compute slippage here
    broker.cancel_order(oid)  # noop if already filled
    return {"ok": True, "lat_ms": lat, "slip_bps": slip_bps}
"""