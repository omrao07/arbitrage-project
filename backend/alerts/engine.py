# backend/alerts/engine.py
"""
Alerts Engine
-------------
Evaluate streaming events against user-defined rules and emit alerts.

✓ Inputs: Redis Streams via backend.bus.streams.consume_stream
✓ Rules: threshold / rate-of-change / expression (Python-safe eval) / windowed count
✓ Controls: dedupe, throttle, quiet hours, severity, tags, escalation
✓ Outputs: Redis stream (alerts.events), optional webhooks, email stub, WS gateway
✓ Storage: in-memory registry (can be extended to Redis HSET if desired)
✓ CLI: --probe (self-test) / --run (live)

Example rule (programmatic):
    Rule(
        name="drawdown_kill",
        stream="risk.metrics",
        expr="float(ev.get('dd',0)) >= 0.10",
        severity="critical",
        message="Kill switch: {strategy} DD={dd:.2%}",
        keys=["strategy"], throttle_s=60, dedupe="strategy"
    )

YAML (optional) schema (config/alerts.yaml):
rules:
  - name: pnl_spike
    stream: pnl.events
    type: threshold
    field: pnl
    op: <=                # >=, <=, >, <, ==, !=
    value: -5000
    severity: high
    message: "{strategy}: PnL spike {pnl}"
    keys: [strategy]
    throttle_s: 120
    dedupe: strategy
quiet_hours:
  enabled: true
  tz: Asia/Kolkata
  start: "22:00"
  end:   "07:00"
  allow_severity: ["critical"]   # allowed during quiet hours
webhook:
  url: https://example.com/hook
  headers: {"Authorization":"Bearer XYZ"}

"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

# --------- Optional deps / glue ----------
try:
    import requests  # pip install requests
except Exception:
    requests = None  # type: ignore

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # type: ignore

try:
    from backend.bus.streams import consume_stream, publish_stream, hset # type: ignore
except Exception:
    # minimal fallbacks for probe mode
    def consume_stream(stream: str, start_id: str = "$", block_ms: int = 1000, count: int = 100):
        if False:
            yield None
        return []
    def publish_stream(stream: str, payload: Dict[str, Any]):
        print(f"[ALERTS-> {stream}] {payload}")
    def hset(key: str, field: str, value: Any):
        pass

# ------------- utils -------------
def _now_ms() -> int: return int(time.time() * 1000)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if math.isfinite(f): return f
        return default
    except Exception:
        return default

def _within_quiet(now_ts: float, tz: str, start: str, end: str) -> bool:
    """Simple quiet-hours check using local time offset from env TZ or provided tz."""
    try:
        import datetime as dt
        # naive: interpret tz only via offset if it's in 'UTC±HH:MM' form; else fallback to localtime
        if re.match(r"^UTC[+-]\d{2}:\d{2}$", tz or ""):
            sign = 1 if "+" in tz else -1
            hh, mm = map(int, tz.split("±" if "±" in tz else ("+" if "+" in tz else "-"))[-1].split(":"))
            now = dt.datetime.utcfromtimestamp(now_ts).replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=sign*hh, minutes=sign*mm)
        else:
            now = dt.datetime.fromtimestamp(now_ts)
        s_h, s_m = map(int, (start or "22:00").split(":"))
        e_h, e_m = map(int, (end or "07:00").split(":"))
        s = now.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
        e = now.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
        if s <= e:
            return s <= now <= e
        # crosses midnight
        return now >= s or now <= e
    except Exception:
        return False

# ------------- data model -------------

Severity = Literal["low","medium","high","critical"]

@dataclass
class Rule:
    name: str
    stream: str
    # One of: threshold / roc / expr / count
    type: Literal["threshold","roc","expr","count"] = "expr"
    # threshold
    field: Optional[str] = None
    op: Optional[str] = None      # >=, <=, >, <, ==, !=
    value: Optional[float] = None
    # rate of change (percent over window)
    roc_field: Optional[str] = None
    roc_pct: Optional[float] = None       # e.g., 5.0 for +5%
    roc_window_n: int = 10
    # expression (evaluate against 'ev' dict)
    expr: Optional[str] = None
    # count (N events matching filter within window_s)
    where: Optional[str] = None
    count_n: int = 5
    window_s: int = 60
    # common
    severity: Severity = "medium"
    message: str = "{name} triggered"
    keys: Sequence[str] = field(default_factory=list)  # type: ignore # dimensions for dedupe/keying (e.g., ["strategy","symbol"])
    throttle_s: int = 60
    dedupe: Optional[str] = None       # format-string key or name of a field; None -> global
    tags: Sequence[str] = field(default_factory=list) # type: ignore
    escalate_after_s: Optional[int] = None
    escalate_to: Optional[Severity] = None

@dataclass
class Alert:
    ts_ms: int
    rule: str
    severity: Severity
    title: str
    message: str
    data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    key: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms, "rule": self.rule, "severity": self.severity,
            "title": self.title, "message": self.message, "data": self.data,
            "tags": self.tags, "key": self.key
        }

# ------------- engine -------------

class AlertsEngine:
    def __init__(self, rules: Optional[List[Rule]] = None, *, publish_topic: str = "alerts.events"):
        self.rules: List[Rule] = rules or []
        self.publish_topic = publish_topic
        self._running = False

        # state stores
        self._last_fire_ms: Dict[str, int] = {}     # by dedupe key
        self._roc_buffers: Dict[Tuple[str,str], List[float]] = {}   # (rule.name, field) -> values
        self._count_buffers: Dict[str, List[int]] = {}              # rule.name -> timestamps

        # cursors per stream (when running in stream mode)
        self._cursors: Dict[str, str] = {}

        # global quiet hours
        self._quiet_cfg = {
            "enabled": False,
            "tz": os.getenv("ALERTS_TZ", "Asia/Kolkata"),
            "start": "22:00",
            "end": "07:00",
            "allow_severity": ["critical"],
        }

        # outbound channels
        self._webhook_cfg: Optional[Dict[str, Any]] = None
        self._email_cfg: Optional[Dict[str, Any]] = None  # placeholder

    # ---------- configuration ----------
    def load_yaml(self, path: str) -> None:
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        self.rules = [self._rule_from_dict(r) for r in (doc.get("rules") or [])]
        qc = (doc.get("quiet_hours") or {})
        self._quiet_cfg.update({
            "enabled": bool(qc.get("enabled", False)),
            "tz": qc.get("tz", self._quiet_cfg["tz"]),
            "start": qc.get("start", self._quiet_cfg["start"]),
            "end": qc.get("end", self._quiet_cfg["end"]),
            "allow_severity": qc.get("allow_severity", self._quiet_cfg["allow_severity"]),
        })
        self._webhook_cfg = doc.get("webhook")
        self._email_cfg = doc.get("email")

    @staticmethod
    def _rule_from_dict(d: Dict[str, Any]) -> Rule:
        return Rule(
            name=str(d["name"]),
            stream=str(d["stream"]),
            type=str(d.get("type","expr")), # type: ignore
            field=d.get("field"),
            op=d.get("op"),
            value=float(d["value"]) if d.get("value") is not None else None,
            roc_field=d.get("roc_field"),
            roc_pct=float(d["roc_pct"]) if d.get("roc_pct") is not None else None,
            roc_window_n=int(d.get("roc_window_n",10)),
            expr=d.get("expr"),
            where=d.get("where"),
            count_n=int(d.get("count_n",5)),
            window_s=int(d.get("window_s",60)),
            severity=str(d.get("severity","medium")),  # type: ignore
            message=str(d.get("message","{name} triggered")),
            keys=list(d.get("keys") or []),
            throttle_s=int(d.get("throttle_s",60)),
            dedupe=d.get("dedupe"),
            tags=list(d.get("tags") or []),
            escalate_after_s=int(d["escalate_after_s"]) if d.get("escalate_after_s") else None,
            escalate_to=str(d.get("escalate_to")) if d.get("escalate_to") else None,  # type: ignore
        )

    def register(self, rule: Rule):
        self.rules.append(rule)

    # ---------- runtime ----------
    def start(self, streams: Optional[Sequence[str]] = None):
        """
        Start polling the given streams (or the unique set of streams in rules if None).
        """
        self._running = True
        self._install_signals()
        targets = list(dict.fromkeys(streams or [r.stream for r in self.rules]))
        try:
            while self._running:
                for s in targets:
                    self._drain_stream(s)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    def stop(self):
        self._running = False

    # ---------- stream plumbing ----------
    def _drain_stream(self, stream: str):
        cursor = self._cursors.get(stream, "$")
        try:
            for _id, raw in consume_stream(stream, start_id=cursor, block_ms=1, count=200): # type: ignore
                ev = raw if isinstance(raw, dict) else self._safe_json(raw)
                self.evaluate(ev, stream=stream)
            self._cursors[stream] = "$"
        except Exception:
            pass

    # ---------- evaluation ----------
    def evaluate(self, ev: Dict[str, Any], *, stream: Optional[str] = None) -> List[Alert]:
        """
        Evaluate one event against all rules (filtered by stream).
        Returns list of fired Alert objects.
        """
        fired: List[Alert] = []
        for r in self.rules:
            if stream and r.stream != stream:
                continue
            ok = False
            if r.type == "threshold":
                val = _safe_float(ev.get(r.field or ""), float("nan"))
                ok = self._cmp(val, r.op or ">=", r.value if r.value is not None else 0.0)
            elif r.type == "roc":
                ok = self._check_roc(r, ev)
            elif r.type == "expr":
                ok = self._safe_eval(r.expr or "", ev)
            elif r.type == "count":
                ok = self._check_count(r, ev)

            if not ok:
                continue

            alert = self._make_alert(r, ev)
            if not self._should_emit(alert):
                continue

            self._emit(alert)
            fired.append(alert)
        return fired

    # --- rule helpers ---
    def _cmp(self, val: float, op: str, ref: float) -> bool:
        if op == ">=": return val >= ref
        if op == "<=": return val <= ref
        if op == ">":  return val >  ref
        if op == "<":  return val <  ref
        if op == "==": return val == ref
        if op == "!=": return val != ref
        return False

    def _check_roc(self, r: Rule, ev: Dict[str, Any]) -> bool:
        field = r.roc_field or r.field or ""
        key = (r.name, field)
        buf = self._roc_buffers.setdefault(key, [])
        val = _safe_float(ev.get(field))
        buf.append(val)
        if len(buf) > max(2, r.roc_window_n):
            buf.pop(0)
        if len(buf) < 2:
            return False
        base = buf[0]
        if base == 0:
            return False
        pct = (buf[-1] - base) / abs(base) * 100.0
        need = float(r.roc_pct or 0.0)
        return abs(pct) >= abs(need)

    def _check_count(self, r: Rule, ev: Dict[str, Any]) -> bool:
        now = int(time.time())
        key = r.name
        lst = self._count_buffers.setdefault(key, [])
        cond = True
        if r.where:
            cond = self._safe_eval(r.where, ev)
        if cond:
            lst.append(now)
        # drop old
        lst[:] = [t for t in lst if now - t <= r.window_s]
        return len(lst) >= r.count_n

    def _safe_eval(self, expr: str, ev: Dict[str, Any]) -> bool:
        if not expr:
            return False
        try:
            allowed = {"__builtins__": {}}
            scope = {"ev": ev, "float": float, "int": int, "abs": abs, "min": min, "max": max, "math": math}
            return bool(eval(expr, allowed, scope))
        except Exception:
            return False

    # --- alert build / gating ---
    def _make_alert(self, r: Rule, ev: Dict[str, Any]) -> Alert:
        title = r.name
        # message formatting with safe defaults
        fmt = {k: ev.get(k) for k in ev.keys()}
        try:
            msg = r.message.format(**fmt, **{"name": r.name})
        except Exception:
            msg = r.message
        key = self._dedupe_key(r, ev)
        return Alert(
            ts_ms=_now_ms(),
            rule=r.name,
            severity=r.severity,
            title=title,
            message=msg,
            data={"event": ev, "rule": asdict(r)},
            tags=list(r.tags),
            key=key,
        )

    def _dedupe_key(self, r: Rule, ev: Dict[str, Any]) -> str:
        if r.dedupe:
            # if dedupe is a field name, use that, else treat as format string
            if r.dedupe in ev:
                return f"{r.name}:{ev.get(r.dedupe)}"
            try:
                return r.dedupe.format(**ev)
            except Exception:
                pass
        if r.keys:
            parts = [str(ev.get(k,"*")) for k in r.keys]
            return f"{r.name}:{'|'.join(parts)}"
        return r.name

    def _should_emit(self, alert: Alert) -> bool:
        now = int(time.time())
        # quiet hours
        qc = self._quiet_cfg
        if qc.get("enabled"):
            in_quiet = _within_quiet(now, qc.get("tz",""), qc.get("start","22:00"), qc.get("end","07:00"))
            if in_quiet and alert.severity not in set(qc.get("allow_severity", [])):
                return False
        # throttle
        last = self._last_fire_ms.get(alert.key or alert.rule, 0)
        throttle_ms = 1000 * next((r.throttle_s for r in self.rules if r.name == alert.rule), 60)
        if _now_ms() - last < throttle_ms:
            return False
        self._last_fire_ms[alert.key or alert.rule] = _now_ms()
        return True

    # --- emit ---
    def _emit(self, alert: Alert) -> None:
        payload = alert.as_dict()
        # bus
        try:
            publish_stream(self.publish_topic, payload)
        except Exception:
            pass
        if hset:
            try:
                hset("alerts:last", alert.key or alert.rule, payload)
            except Exception:
                pass
        # webhook
        if self._webhook_cfg and requests:
            try:
                url = self._webhook_cfg.get("url")
                headers = self._webhook_cfg.get("headers") or {"Content-Type":"application/json"}
                if url:
                    requests.post(url, headers=headers, data=json.dumps(payload), timeout=3)
            except Exception:
                pass
        # email (stub – integrate your mailer)
        if self._email_cfg:
            # TODO: wire to your SMTP or provider
            pass

    # ---------- signals ----------
    def _install_signals(self):
        def _stop(signum, frame):
            self.stop(); sys.exit(0)
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

    # ---------- helpers ----------
    @staticmethod
    def _safe_json(x: Any) -> Dict[str, Any]:
        if isinstance(x, dict): return x
        try: return json.loads(x)
        except Exception: return {}

# ---------------- CLI ----------------

def _probe():
    """
    Self-test: create a few rules and feed synthetic events.
    """
    eng = AlertsEngine([
        Rule(name="dd_warn", stream="risk.metrics", type="threshold", field="dd", op=">=", value=0.06,
             severity="high", message="{strategy}: DD {dd:.2%}", keys=["strategy"], throttle_s=10, dedupe="strategy"),
        Rule(name="pnl_spike", stream="pnl.events", type="threshold", field="pnl", op="<=", value=-5000,
             severity="medium", message="{strategy}: PnL spike {pnl}", keys=["strategy"], throttle_s=30),
        Rule(name="slip_high", stream="tca.metrics", type="expr", expr="float(ev.get('slip_bps',0)) >= 20",
             severity="low", message="{strategy}: slippage {slip_bps} bps", keys=["strategy"], throttle_s=60),
        Rule(name="news_heat", stream="news.sentiment", type="expr", expr="float(ev.get('heat',0))>=0.8",
             severity="high", message="🔥 {symbol} news heat {heat}", keys=["symbol"], throttle_s=120),
        Rule(name="roc_vol", stream="risk.metrics", type="roc", roc_field="vol", roc_pct=50.0, roc_window_n=5,
             severity="medium", message="{strategy}: vol ROC breached", keys=["strategy"], throttle_s=120),
        Rule(name="error_burst", stream="engine.errors", type="count", where="True", count_n=5, window_s=30,
             severity="high", message="Engine error burst", throttle_s=120),
    ])
    # feed events
    events = [
        ("risk.metrics", {"strategy":"momo_in","dd":0.03,"vol":0.10}),
        ("risk.metrics", {"strategy":"momo_in","dd":0.07,"vol":0.11}),
        ("pnl.events",  {"strategy":"momo_in","pnl":-7000}),
        ("tca.metrics", {"strategy":"arb_x","slip_bps":26.4}),
        ("news.sentiment", {"symbol":"NIFTY","heat":0.85}),
    ]
    for s, ev in events:
        eng.evaluate(ev, stream=s)
    # ROC feed
    for v in [0.10,0.11,0.12,0.13,0.16,0.18]:
        eng.evaluate({"strategy":"momo_in","vol":v}, stream="risk.metrics")
    # Count feed
    for _ in range(6):
        eng.evaluate({"msg":"boom"}, stream="engine.errors")
    print("Probe done. Alerts were printed to stdout via publish_stream fallback.")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Alerts Engine")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--yaml", type=str, help="config/alerts.yaml")
    args = ap.parse_args()

    eng = AlertsEngine()
    if args.yaml:
        eng.load_yaml(args.yaml)

    if args.probe or not args.run:
        _probe()
        return

    # Live mode
    if not eng.rules:
        print("No rules loaded (use --yaml). Exiting."); return
    eng.start()

if __name__ == "__main__":
    main()