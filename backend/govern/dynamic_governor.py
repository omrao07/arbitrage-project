# backend/risk/dynamic_governor.py
from __future__ import annotations

import os, time, json, math, signal, sys, traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# ---------- optional deps (graceful) ----------
HAVE_YAML = True
try:
    import yaml  # type: ignore
except Exception:
    HAVE_YAML = False
    yaml = None  # type: ignore

HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

# ---------- env / streams ----------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_METRICS  = os.getenv("RISK_METRICS_STREAM", "risk.metrics")     # incoming snapshots
STREAM_COMMANDS = os.getenv("RISK_COMMANDS_STREAM", "risk.commands")   # outgoing decisions
HASH_LIMITS     = os.getenv("RISK_LIMITS_HASH", "risk:limits")         # where we store latest caps
KEY_KILLSWITCH  = os.getenv("RISK_KILLSWITCH_KEY", "risk:killswitch")  # "on"/"off"

# ---------- datastructs ----------
@dataclass
class MetricSnapshot:
    ts_ms: int
    # global / book
    gross_lev: float = 0.0          # |long|+|short| / equity
    net_lev: float = 0.0            # |net| / equity
    book_var: float = 0.0           # VaR (currency)
    book_es: float = 0.0            # Expected shortfall
    book_dd_frac: float = 0.0       # trailing drawdown 0..1
    book_liq_score: float = 1.0     # 0..1, 1=very liquid
    pnl_1d: float = 0.0
    pnl_5d: float = 0.0
    # per strategy (optional)
    per_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g., {"alpha_dip":{"vol":0.22,"dd":0.05,"pnl_5d":1200,"signal":0.7}}

@dataclass
class BandRule:
    name: str
    when: Dict[str, Any]            # condition dict (see _cond_ok)
    set: Dict[str, Any]             # limits to apply if matched
    hysteresis_bps: float = 30.0    # to avoid thrash (applied where relevant)
    priority: int = 0               # larger wins if multiple match

@dataclass
class Decision:
    ts_ms: int
    scope: str                      # "global" or "strategy:<name>"
    applied_rule: Optional[str]
    limits: Dict[str, Any]
    quarantined: bool = False
    kill_switch: bool = False
    reason: Optional[str] = None

# ---------- governor core ----------
class DynamicGovernor:
    """
    Rule-based governor with hysteresis and safe fallbacks.

    Config YAML (configs/dynamic_governor.yaml) example:

    global:
      defaults:
        max_gross_lev: 3.0
        max_single_name_w: 0.10
        max_order_val_usd: 5_000_000
      bands:
        - name: NORMAL
          when: { dd_frac_lte: 0.05, vol_lte: 0.25 }
          set:  { max_gross_lev: 3.0, risk_on: true }
        - name: CAUTION
          when: { dd_frac_gt: 0.05, dd_frac_lte: 0.10 }
          set:  { max_gross_lev: 2.0, throttle_bps: 20 }
        - name: STRESS
          when: { dd_frac_gt: 0.10 }
          set:  { max_gross_lev: 1.0, halt_new: true }
          hysteresis_bps: 50

    per_strategy:
      defaults:
        max_strategy_lev: 1.5
        max_drawdown: 0.15
      bands:
        - name: COOL_OFF
          when: { dd_frac_gt: 0.08 }             # strategy dd
          set:  { disabled: true }
        - name: VOLY
          when: { vol_gt: 0.35 }
          set:  { max_strategy_lev: 0.7, throttle_bps: 50 }

    kill:
      trigger:
        dd_frac_gt: 0.20
        es_gt: 1_000_000
      action: { kill_switch: true }

    quarantine:
      when:
        liq_score_lt: 0.2
      action: { quarantined: true }
    """

    def __init__(self, cfg_path: str = "configs/dynamic_governor.yaml", *, redis_url: Optional[str] = None):
        self.cfg_path = cfg_path
        self.cfg = self._load_cfg(cfg_path)
        self.redis_url = redis_url or REDIS_URL
        self.r = None
        if HAVE_REDIS:
            try:
                self.r = Redis.from_url(self.redis_url, decode_responses=True)  # type: ignore
            except Exception:
                self.r = None
        # keep previous decisions for hysteresis
        self._last: Dict[str, Decision] = {}

    # ----- config loading -----
    def _load_cfg(self, path: str) -> Dict[str, Any]:
        if HAVE_YAML and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {} # type: ignore
        # minimal default
        return {
            "global": {
                "defaults": {"max_gross_lev": 2.0, "max_single_name_w": 0.1},
                "bands": [
                    {"name":"NORMAL","when":{"dd_frac_lte":0.05},"set":{"max_gross_lev":2.0}},
                    {"name":"STRESS","when":{"dd_frac_gt":0.10},"set":{"max_gross_lev":1.0,"halt_new":True},"hysteresis_bps":50}
                ]
            },
            "per_strategy": {
                "defaults": {"max_strategy_lev":1.0},
                "bands": [{"name":"COOL_OFF","when":{"dd_frac_gt":0.08},"set":{"disabled":True}}]
            },
            "kill": {"trigger":{"dd_frac_gt":0.20}, "action":{"kill_switch":True}},
            "quarantine": {"when":{"liq_score_lt":0.15},"action":{"quarantined":True}}
        }

    # ----- evaluation helpers -----
    def _hyst(self, prev: Optional[Decision], key: str, new_val: float, bps: float) -> float:
        """Apply hysteresis around previous numeric limit to reduce oscillation (bps = 0.01% units)."""
        if not prev or key not in prev.limits or bps <= 0: 
            return new_val
        old = float(prev.limits[key])
        band = abs(old) * (bps / 1e4)
        if new_val > old and new_val - old < band:   # raising cap slightly → keep old until clear
            return old
        if new_val < old and old - new_val < band:   # lowering cap slightly → keep old until clear
            return old
        return new_val

    def _cond_ok(self, when: Dict[str, Any], scope_metrics: Dict[str, float]) -> bool:
        """
        Evaluate simple threshold conditions against provided metrics.
        Supported ops (suffix): _gt, _gte, _lt, _lte, _eq, e.g., dd_frac_gt: 0.05
        Metric keys: "dd_frac","vol","var","es","liq_score","pnl_1d","pnl_5d","gross_lev","net_lev"
        """
        m = scope_metrics
        for k, v in (when or {}).items():
            if "_" not in k:
                # boolean flag present in metrics must equal v
                if bool(m.get(k)) != bool(v):
                    return False
                continue
            name, op = k.rsplit("_", 1)
            x = m.get(name)
            if x is None:
                return False
            if op == "gt"  and not (x >  v): return False
            if op == "gte" and not (x >= v): return False
            if op == "lt"  and not (x <  v): return False
            if op == "lte" and not (x <= v): return False
            if op == "eq"  and not (x == v): return False
        return True

    def _pick_band(self, bands: List[Dict[str, Any]], metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        matched = []
        for b in bands or []:
            if self._cond_ok(b.get("when") or {}, metrics):
                matched.append(b)
        if not matched:
            return None
        # pick highest priority then most restrictive (lowest max_gross_lev if present)
        matched.sort(key=lambda b: (int(b.get("priority", 0)), -float(b.get("set", {}).get("max_gross_lev", 1e9))), reverse=True)
        return matched[0]

    # ----- main evaluation -----
    def evaluate(self, snap: MetricSnapshot) -> List[Decision]:
        """Return a list of decisions (global + per-strategy)."""
        now = int(time.time() * 1000)
        out: List[Decision] = []

        # ---- kill-switch & quarantine at book level
        kill_cfg = self.cfg.get("kill", {})
        kill_trig = kill_cfg.get("trigger", {})
        book = {
            "dd_frac": snap.book_dd_frac,
            "es": snap.book_es,
            "var": snap.book_var,
            "vol": max(1e-9, snap.per_strategy.get("_book", {}).get("vol", 0.0)),
            "liq_score": snap.book_liq_score,
            "pnl_1d": snap.pnl_1d,
            "pnl_5d": snap.pnl_5d,
            "gross_lev": snap.gross_lev,
            "net_lev": snap.net_lev,
        }
        kill = self._cond_ok(kill_trig, book)
        quarantined = False
        q_cfg = self.cfg.get("quarantine", {})
        if self._cond_ok(q_cfg.get("when", {}), book):
            quarantined = True

        # ---- global limits via bands
        g = self.cfg.get("global", {})
        g_defaults = g.get("defaults", {}) or {}
        g_bands = g.get("bands", []) or []
        g_band = self._pick_band(g_bands, book)

        limits = dict(g_defaults)
        applied_name = None
        if g_band:
            applied_name = g_band.get("name")
            limits.update(g_band.get("set", {}))
            # hysteresis on numeric caps
            prev = self._last.get("global")
            for k, v in list(limits.items()):
                if isinstance(v, (int, float)):
                    limits[k] = self._hyst(prev, k, float(v), float(g_band.get("hysteresis_bps", 0.0)))

        dec_g = Decision(ts_ms=now, scope="global", applied_rule=applied_name,
                         limits=limits, quarantined=quarantined, kill_switch=kill,
                         reason=("kill" if kill else applied_name or "defaults"))
        out.append(dec_g)
        self._last["global"] = dec_g

        # ---- per-strategy limits
        ps = self.cfg.get("per_strategy", {})
        s_defaults = ps.get("defaults", {}) or {}
        s_bands = ps.get("bands", []) or []
        for name, m in (snap.per_strategy or {}).items():
            if name == "_book":  # reserved
                continue
            metrics = {
                "dd_frac": float(m.get("dd", 0.0)),
                "vol": float(m.get("vol", 0.0)),
                "pnl_5d": float(m.get("pnl_5d", 0.0)),
                "liq_score": float(m.get("liq", 1.0)),
                "signal": float(m.get("signal", 0.0)),
            }
            band = self._pick_band(s_bands, metrics)
            slims = dict(s_defaults)
            applied = None
            if band:
                applied = band.get("name")
                slims.update(band.get("set", {}))
                prev = self._last.get(f"strategy:{name}")
                for k, v in list(slims.items()):
                    if isinstance(v, (int, float)):
                        slims[k] = self._hyst(prev, k, float(v), float(band.get("hysteresis_bps", 0.0)))
            dec_s = Decision(ts_ms=now, scope=f"strategy:{name}", applied_rule=applied,
                             limits=slims, quarantined=quarantined, kill_switch=kill,
                             reason=("kill" if kill else applied or "defaults"))
            out.append(dec_s)
            self._last[f"strategy:{name}"] = dec_s

        return out

    # ----- side effects (Redis) -----
    def apply(self, decisions: List[Decision]) -> None:
        """Persist decisions to Redis (hash + stream). No-op if Redis not available."""
        if not HAVE_REDIS or not self.r:
            return
        for d in decisions:
            # flatten limits with metadata and write to hash
            key = HASH_LIMITS if d.scope == "global" else f"{HASH_LIMITS}:{d.scope.split(':',1)[1]}"
            payload = {
                "ts_ms": d.ts_ms,
                "applied_rule": d.applied_rule or "",
                "quarantined": "true" if d.quarantined else "false",
                "kill_switch": "true" if d.kill_switch else "false",
                "reason": d.reason or "",
                **{k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in d.limits.items()},
            }
            try:
                self.r.hset(key, mapping=payload)  # type: ignore
            except Exception:
                pass
            # publish command to stream for runtime components (router, executor, etc.)
            cmd = {"scope": d.scope, "limits": d.limits, "quarantined": d.quarantined, "kill_switch": d.kill_switch, "reason": d.reason, "ts_ms": d.ts_ms}
            try:
                self.r.xadd(STREAM_COMMANDS, {"json": json.dumps(cmd)}, maxlen=50_000, approximate=True)  # type: ignore
            except Exception:
                pass
        # set kill key for global consumers
        try:
            ks = any(d.kill_switch for d in decisions if d.scope == "global")
            self.r.set(KEY_KILLSWITCH, "on" if ks else "off")  # type: ignore
        except Exception:
            pass

# ---------- streaming runner (optional) ----------
def run_loop(governor: DynamicGovernor):
    """
    Listens to STREAM_METRICS for MetricSnapshot JSON and emits decisions.
    Metrics producer should XADD like:
      XADD risk.metrics * json '{"ts_ms":..., "gross_lev":..., "book_dd_frac":..., "per_strategy":{"alpha":{"vol":0.2,"dd":0.03}}}'
    """
    if not HAVE_REDIS or governor.r is None:
        print("[governor] Redis not available; run_loop() will idle. Use CLI for file-based evaluation.")
        while True:
            time.sleep(60)

    r = governor.r
    last_id = "$"  # only new metrics
    print("[governor] listening on", STREAM_METRICS)
    while True:
        try:
            resp = r.xread({STREAM_METRICS: last_id}, count=200, block=1000)  # type: ignore
            if not resp:
                continue
            _, entries = resp[0] # type: ignore
            last_id = entries[-1][0]
            for _id, fields in entries:
                snap = _parse_snapshot(fields)
                decisions = governor.evaluate(snap)
                governor.apply(decisions)
        except KeyboardInterrupt:
            print("\n[governor] stopping...")
            break
        except Exception as e:
            print("[governor] error:", e)
            traceback.print_exc()
            time.sleep(0.5)

def _parse_snapshot(fields: Dict[str, Any]) -> MetricSnapshot:
    if "json" in fields:
        try:
            obj = json.loads(fields["json"])
            return MetricSnapshot(
                ts_ms=int(obj.get("ts_ms") or int(time.time()*1000)),
                gross_lev=float(obj.get("gross_lev", 0.0)),
                net_lev=float(obj.get("net_lev", 0.0)),
                book_var=float(obj.get("book_var", 0.0)),
                book_es=float(obj.get("book_es", 0.0)),
                book_dd_frac=float(obj.get("book_dd_frac", 0.0)),
                book_liq_score=float(obj.get("book_liq_score", 1.0)),
                pnl_1d=float(obj.get("pnl_1d", 0.0)),
                pnl_5d=float(obj.get("pnl_5d", 0.0)),
                per_strategy=obj.get("per_strategy") or {},
            )
        except Exception:
            pass
    # fall back to raw fields
    def f(x, d=0.0):
        try: return float(x)
        except Exception: return d
    return MetricSnapshot(
        ts_ms=int(time.time()*1000),
        gross_lev=f(fields.get("gross_lev")),
        net_lev=f(fields.get("net_lev")),
        book_var=f(fields.get("book_var")),
        book_es=f(fields.get("book_es")),
        book_dd_frac=f(fields.get("book_dd_frac")),
        book_liq_score=f(fields.get("book_liq_score"), 1.0),
        pnl_1d=f(fields.get("pnl_1d")),
        pnl_5d=f(fields.get("pnl_5d")),
        per_strategy={},
    )

# ---------- CLI ----------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("dynamic_governor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="Evaluate a single metrics JSON file")
    e.add_argument("--cfg", default="configs/dynamic_governor.yaml")
    e.add_argument("--metrics-json", required=True, help="Path to metrics JSON")
    e.add_argument("--out", default="-", help="Write decisions JSON (- for stdout)")

    r = sub.add_parser("run", help="Run streaming governor against Redis (risk.metrics → risk.commands)")
    r.add_argument("--cfg", default="configs/dynamic_governor.yaml")
    r.add_argument("--redis-url", default=REDIS_URL)

    args = ap.parse_args()
    if args.cmd == "eval":
        gov = DynamicGovernor(cfg_path=args.cfg)
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        snap = MetricSnapshot(
            ts_ms=int(obj.get("ts_ms") or int(time.time()*1000)),
            gross_lev=float(obj.get("gross_lev", 0.0)),
            net_lev=float(obj.get("net_lev", 0.0)),
            book_var=float(obj.get("book_var", 0.0)),
            book_es=float(obj.get("book_es", 0.0)),
            book_dd_frac=float(obj.get("book_dd_frac", 0.0)),
            book_liq_score=float(obj.get("book_liq_score", 1.0)),
            pnl_1d=float(obj.get("pnl_1d", 0.0)),
            pnl_5d=float(obj.get("pnl_5d", 0.0)),
            per_strategy=obj.get("per_strategy") or {},
        )
        decisions = gov.evaluate(snap)
        out = [asdict(d) for d in decisions]
        if args.out == "-":
            print(json.dumps(out, indent=2))
        else:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"[governor] wrote {args.out}")

    elif args.cmd == "run":
        gov = DynamicGovernor(cfg_path=args.cfg, redis_url=args.redis_url)
        run_loop(gov)

if __name__ == "__main__":
    _cli()