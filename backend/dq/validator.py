# backend/safety/validator.py
from __future__ import annotations

import os, sys, json, time, glob, traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# ---- Optional deps (graceful) ----------------------------------------------
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

# ---- Helpers ----------------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def _coerce_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try: return float(x)
    except Exception: return default

def _coerce_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try: return int(x)
    except Exception: return default

# ---- Reporting ---------------------------------------------------------------
@dataclass
class Finding:
    level: str     # "ok" | "warn" | "err"
    scope: str     # e.g. "config:latency.yaml" or "stream:orders.filled"
    code: str      # short id
    msg: str
    hint: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    started_ms: int = field(default_factory=now_ms)
    finished_ms: Optional[int] = None

    def add(self, level: str, scope: str, code: str, msg: str, hint: Optional[str] = None, **data):
        self.findings.append(Finding(level, scope, code, msg, hint, data))

    def done(self):
        self.finished_ms = now_ms()

    def ok(self) -> bool:
        return not any(f.level == "err" for f in self.findings)

    def to_json(self) -> str:
        return json.dumps({
            "started_ms": self.started_ms,
            "finished_ms": self.finished_ms or now_ms(),
            "ok": self.ok(),
            "findings": [asdict(f) for f in self.findings],
        }, indent=2)

# ---- Config schemas (minimal, extend as needed) -----------------------------
# We intentionally use *hand checks* (no jsonschema dependency).
# Each schema: key -> ("type", required:bool)
Schema = Dict[str, Tuple[str, bool]]

SCHEMAS: Dict[str, Schema] = {
    # risk / stress configs
    "bank_stress.yaml": {
        "shock_scenarios": ("list", True),
        "capital_threshold": ("float", True),
        "liquidity_haircut_bps": ("float", False),
    },
    "capital_stress.yaml": {
        "var_limit": ("float", True),
        "es_limit": ("float", False),
        "drawdown_limit": ("float", False),
    },
    "governor.yaml": {
        "max_gross_leverage": ("float", True),
        "max_single_name_weight": ("float", True),
        "kill_switch_drawdown": ("float", True),
    },
    "dynamic_governor.yaml": {
        "bands": ("list", True),  # [{metric: 'vol', low:0.1, high:0.3, target_lev:2.0}, ...]
    },
    "latency.yaml": {
        "target_ms": ("int", True),
        "warn_ms": ("int", True),
        "crit_ms": ("int", True),
    },
    "registry.yaml": {
        "brokers": ("dict", False),
        "venues": ("dict", False),
        "symbols": ("dict", False),  # {AAPL: {isin:..., mic:...}}
    },
    "liquidity_surface.yaml": {
        "grid": ("list", True),       # e.g., tenors/qty grid
        "fit": ("dict", False),
    },
    "vol_surface.yaml": {
        "smile": ("list", True),      # list of {k, iv}
        "asof": ("str", False),
    },
    "microstructure.yaml": {
        "tick_size": ("float", False),
        "lot_size": ("float", False),
        "slippage_bps": ("float", False),
    },
    "rivals.yaml": {
        "competitors": ("list", False),
    },
    "shocks.yaml": {
        "shocks": ("list", True),     # [{name, type, magnitude, ...}]
    },
    "soverign.yaml": {
        "countries": ("list", True),
        "default_prob_floor": ("float", False),
    },
    "alerts.yaml": {
        "routes": ("list", False),    # where to send alerts
        "thresholds": ("dict", False)
    },
    "altdata.yaml": {
        "sources": ("list", False),
        "refresh_minutes": ("int", False),
    },
}

# ---- Message validators -----------------------------------------------------
def validate_order(obj: Dict[str, Any]) -> List[str]:
    req = ("strategy","symbol","side","qty")
    errs = [f"missing:{k}" for k in req if obj.get(k) in (None,"",0)]
    if obj.get("side") not in ("buy","sell"): errs.append("side:must be 'buy'/'sell'")
    if _coerce_float(obj.get("qty"), -1) <= 0: errs.append("qty:must be > 0") # type: ignore
    return errs

def validate_fill(obj: Dict[str, Any]) -> List[str]:
    req = ("order_id","symbol","side","qty","price")
    errs = [f"missing:{k}" for k in req if obj.get(k) in (None,"",0)]
    if _coerce_float(obj.get("price"), -1) <= 0: errs.append("price:must be > 0") # type: ignore
    if _coerce_float(obj.get("qty"), -1) <= 0: errs.append("qty:must be > 0") # type: ignore
    return errs

def validate_bar(obj: Dict[str, Any]) -> List[str]:
    req = ("symbol","open","high","low","close")
    errs = [f"missing:{k}" for k in req if _coerce_float(obj.get(k)) is None]
    o = _coerce_float(obj.get("open")); h = _coerce_float(obj.get("high")); l = _coerce_float(obj.get("low")); c = _coerce_float(obj.get("close"))
    if None not in (o,h,l,c) and not (h >= max(o,c) and l <= min(o,c)): # type: ignore
        errs.append("ohlc:inconsistent (expect low<=open/close<=high)")
    return errs

def validate_l2(obj: Dict[str, Any]) -> List[str]:
    # minimal: require symbol and at least one side level with (px,qty)
    sym = obj.get("symbol"); bids = obj.get("bids"); asks = obj.get("asks")
    errs = []
    if not sym: errs.append("missing:symbol")
    def _ok_side(side):
        if not isinstance(side, list) or not side: return False
        a = side[0]
        return isinstance(a, (list,tuple)) and len(a) >= 2 and _coerce_float(a[0]) and _coerce_float(a[1])
    if not (_ok_side(bids) or _ok_side(asks)): errs.append("orderbook:need bids/asks as [[px,qty],...]")
    return errs

def validate_greeks(obj: Dict[str, Any]) -> List[str]:
    req = ("symbol","delta","gamma","vega","theta")
    errs = [f"missing:{k}" for k in req if _coerce_float(obj.get(k)) is None]
    return errs

# ---- Core validator ---------------------------------------------------------
class Validator:
    def __init__(self, configs_dir: str = "configs", redis_url: Optional[str] = None):
        self.configs_dir = configs_dir
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.report = Report()

    # -- CONFIGS --------------------------------------------------------------
    def validate_configs(self):
        if not os.path.isdir(self.configs_dir):
            self.report.add("warn", f"configs:{self.configs_dir}", "NO_DIR", "configs directory not found", hint="Create configs/ or pass --configs DIR")
            return

        # For each known schema, try to load and validate if file exists
        for fname, schema in SCHEMAS.items():
            path = os.path.join(self.configs_dir, fname)
            if not os.path.exists(path):
                self.report.add("warn", f"config:{fname}", "MISSING", "file not found", hint=f"Create {fname} under {self.configs_dir}/")
                continue
            try:
                if HAVE_YAML and (path.endswith(".yaml") or path.endswith(".yml")):
                    with open(path, "r", encoding="utf-8") as f: cfg = yaml.safe_load(f) or {} # type: ignore
                else:
                    with open(path, "r", encoding="utf-8") as f: cfg = json.load(f)
            except Exception as e:
                self.report.add("err", f"config:{fname}", "PARSE_ERR", f"failed to parse: {e}")
                continue

            # schema check
            errs, warns = self._check_schema(cfg, schema)
            if errs:
                for e in errs:
                    self.report.add("err", f"config:{fname}", "SCHEMA_ERR", e)
            else:
                self.report.add("ok", f"config:{fname}", "SCHEMA_OK", "schema validated")
            for w in warns:
                self.report.add("warn", f"config:{fname}", "SCHEMA_WARN", w)

    def _check_schema(self, cfg: Dict[str, Any], schema: Schema) -> Tuple[List[str], List[str]]:
        errs, warns = [], []
        # required keys
        for k, (typ, req) in schema.items():
            if req and k not in cfg:
                errs.append(f"missing required key '{k}'")
        # type checks
        for k, v in cfg.items():
            spec = schema.get(k)
            if not spec:
                warns.append(f"unknown key '{k}' (not in schema)")
                continue
            typ, _ = spec
            if typ == "float" and _coerce_float(v) is None: errs.append(f"'{k}' should be float")
            elif typ == "int" and _coerce_int(v) is None: errs.append(f"'{k}' should be int")
            elif typ == "list" and not isinstance(v, list): errs.append(f"'{k}' should be list")
            elif typ == "dict" and not isinstance(v, dict): errs.append(f"'{k}' should be dict")
            elif typ == "str" and not isinstance(v, str): errs.append(f"'{k}' should be str")
        return errs, warns

    # -- ENV / CONNECTIVITY ---------------------------------------------------
    def validate_env(self):
        # Basic must-haves for your stack (tweak as needed)
        keys = ["REDIS_URL"]
        for k in keys:
            val = os.getenv(k)
            if not val:
                self.report.add("warn", "env", "ENV_MISSING", f"{k} not set", hint=f"export {k}=...")
            else:
                self.report.add("ok", "env", "ENV_OK", f"{k} present", kv=k)

        if HAVE_REDIS:
            try:
                r = Redis.from_url(self.redis_url, decode_responses=True)  # type: ignore
                pong = r.ping()
                if pong:
                    self.report.add("ok", "redis", "PING_OK", f"connected to {self.redis_url}")
                else:
                    self.report.add("err", "redis", "PING_FAIL", "ping returned False")
            except Exception as e:
                self.report.add("err", "redis", "CONNECT_ERR", f"cannot connect: {e}", hint="Is Redis running and REDIS_URL correct?")
        else:
            self.report.add("warn", "redis", "CLIENT_MISSING", "redis python package not installed", hint="pip install redis")

    # -- SAMPLE MESSAGE SHAPES ------------------------------------------------
    def validate_samples(self):
        # Provide small sample dicts; users can modify to their payloads
        samples = [
            ("stream:orders.incoming", validate_order, {"strategy":"demo","symbol":"AAPL","side":"buy","qty":1}),
            ("stream:orders.filled",   validate_fill,  {"order_id":"o1","symbol":"AAPL","side":"buy","qty":1,"price":200.0}),
            ("stream:prices.bars",     validate_bar,   {"symbol":"AAPL","open":199.0,"high":201.0,"low":198.5,"close":200.5}),
            ("stream:ws.orderbook",    validate_l2,    {"symbol":"AAPL","bids":[[200.4,100]],"asks":[[200.6,120]]}),
            ("stream:ws.greeks",       validate_greeks,{"symbol":"AAPL2309C200","delta":0.52,"gamma":0.01,"vega":0.12,"theta":-0.02}),
        ]
        for scope, fn, obj in samples:
            errs = fn(obj)
            if errs:
                self.report.add("err", scope, "SCHEMA_ERR", "; ".join(errs), hint="fix your producer/normalizer to match spec", sample=obj)
            else:
                self.report.add("ok", scope, "SCHEMA_OK", "shape validated")

    # -- Public entry ---------------------------------------------------------
    def run_all(self) -> Report:
        self.validate_configs()
        self.validate_env()
        self.validate_samples()
        self.report.done()
        return self.report

# ---- CLI --------------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("validator")
    ap.add_argument("--configs", type=str, default="configs", help="Configs dir")
    ap.add_argument("--redis-url", type=str, default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--json", action="store_true", help="Print JSON report")
    args = ap.parse_args()

    v = Validator(configs_dir=args.configs, redis_url=args.redis_url)
    rep = v.run_all()
    if args.json:
        print(rep.to_json())
    else:
        # pretty text
        print(f"VALIDATOR: ok={rep.ok()} findings={len(rep.findings)}")
        for f in rep.findings:
            badge = {"ok":"✅","warn":"⚠️","err":"❌"}.get(f.level, "•")
            hint = f" — hint: {f.hint}" if f.hint else ""
            print(f"{badge} [{f.level}] {f.scope} :: {f.code} :: {f.msg}{hint}")
    sys.exit(0 if rep.ok() else 2)

if __name__ == "__main__":
    _cli()