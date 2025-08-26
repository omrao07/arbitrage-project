# backend/tools/validator.py
"""
Project Validator / Doctor
--------------------------
Run targeted health checks across your trading stack:

- Python/runtime & env variables
- Optional dependencies present (redis, feedparser, yfinance, nsepython, pandas/numpy, plotly, rasterio/shapely)
- Redis bus connectivity + basic R/W on expected keys/streams
- SQLite caches exist and contain expected tables/schemas:
    - runtime/options.db -> options, fetch_log, vol_surface
    - runtime/altdata.db -> spend_index, card_tx, regions, lights_index, shipping_index
- Config files load & shape check: config/*.yaml (latency.yaml, registry.yaml, bank_stress.yaml, capital_stress.yaml,
  governor.yaml, liquidity_surface.yaml, rivals.yaml, shocks.yaml, soverign*.yaml)
- Imports for key modules you added (best-effort):
    backend/engine/{ensemble, strategy_base}.py
    backend/data/{option_chain, opendataset}.py
    backend/analytics/{vol_surface, hrp}.py
    backend/oms/{market_maker, risk_manager, router, reconciler}.py
    backend/broker/{broker_base, broker_interface}.py
    backend/news/{news_yahoo, news_moneycontrol}.py
    backend/ai/{sentiment_ai, insight_agent, query_agent}.py
    backend/risk/{risk_manager, risk_explainer}.py
    backend/dash/{risk_dashboard, hedge_dashboard, literacy_dashboard}.py
    backend/altdata/{card_spend, satellite_lights, shipping_traffic}.py

CLI:
  python -m backend.tools.validator
  python -m backend.tools.validator --json > health.json
  python -m backend.tools.validator --only redis,sqlite --fail-fast

Exit code:
  0 = all good (no critical failures)
  1 = one or more critical failures
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------- optional deps (we test their presence, not require them) ----------
def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

redis = _try_import("redis")
yaml  = _try_import("yaml")

# ---------- small helpers ----------
ANSI = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}

def _color(txt: str, c: str) -> str:
    return f"{ANSI.get(c,'')}{txt}{ANSI['reset']}"

def _exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False

def _read_yaml(path: str) -> Tuple[Optional[dict], Optional[str]]:
    if yaml is None:
        return None, "pyyaml not installed"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, None
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:
        return None, f"parse_error: {e}"

def _connect_sqlite(path: str) -> Tuple[Optional[sqlite3.Connection], Optional[str]]:
    try:
        cx = sqlite3.connect(path, timeout=5.0)
        cx.row_factory = sqlite3.Row
        return cx, None
    except Exception as e:
        return None, str(e)

def _sqlite_has_tables(cx: sqlite3.Connection, names: List[str]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    try:
        cur = cx.execute("SELECT name FROM sqlite_master WHERE type='table'")
        present = {r[0] for r in cur.fetchall()}
        for n in names:
            out[n] = n in present
        return out
    except Exception:
        return {n: False for n in names}

def _redis_ping(host: str, port: int, db: int = 0) -> Tuple[bool, Optional[str]]:
    if redis is None:
        return False, "redis-py not installed"
    try:
        r = redis.Redis(host=host, port=port, db=db, decode_responses=True, socket_connect_timeout=1.5, socket_timeout=2.0)
        ok = r.ping()
        if not ok:
            return False, "ping=false"
        # RW smoke
        key = f"validator:ping:{int(time.time()*1000)}"
        r.set(key, "ok", ex=5)
        v = r.get(key)
        return (v == "ok"), None if v == "ok" else "rw_mismatch"
    except Exception as e:
        return False, str(e)

# ---------- result struct ----------
@dataclass
class Check:
    name: str
    critical: bool
    passed: bool
    details: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    fix: Optional[str] = None

# ---------- main validator ----------
class Validator:
    def __init__(self, only: Optional[List[str]] = None, fail_fast: bool = False):
        self.only = {x.strip().lower() for x in (only or []) if x.strip()}
        self.fail_fast = fail_fast
        self.results: List[Check] = []

    # ---- dispatcher ----
    def run_all(self) -> List[Check]:
        checks = [
            self.check_python,
            self.check_env_defaults,
            self.check_optional_deps,
            self.check_imports,
            self.check_configs,
            self.check_redis,
            self.check_sqlite_options,
            self.check_sqlite_altdata,
            self.check_bus_topics_optional,
        ]
        for fn in checks:
            tag = fn.__name__.replace("check_", "")
            if self.only and not any(tag.startswith(x) or x in tag for x in self.only):
                continue
            res = fn()
            self.results.extend(res if isinstance(res, list) else [res])
            if self.fail_fast and any((c.critical and not c.passed) for c in (res if isinstance(res, list) else [res])):
                break
        return self.results

    # ---- individual checks ----
    def check_python(self) -> List[Check]:
        ver_ok = sys.version_info >= (3, 10)
        return [Check(
            name="python.version",
            critical=True,
            passed=ver_ok,
            details=f"Running Python {sys.version.split()[0]}, need >= 3.10",
            fix="Use pyenv/conda to switch to Python 3.10+"
        )]

    def check_env_defaults(self) -> List[Check]:
        envs = {
            "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
            "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
            "RISK_INCOMING_STREAM": os.getenv("RISK_INCOMING_STREAM", "orders.incoming"),
        }
        return [Check(
            name="env.core",
            critical=False,
            passed=True,
            details=f"REDIS_HOST={envs['REDIS_HOST']} REDIS_PORT={envs['REDIS_PORT']} RISK_INCOMING_STREAM={envs['RISK_INCOMING_STREAM']}",
        )]

    def check_optional_deps(self) -> List[Check]:
        want = [
            ("redis", "pip install redis", True),
            ("feedparser", "pip install feedparser", False),
            ("yfinance", "pip install yfinance", False),
            ("nsepython", "pip install nsepython", False),
            ("nsetools", "pip install nsetools", False),
            ("pandas", "pip install pandas", False),
            ("numpy", "pip install numpy", False),
            ("plotly.express", "pip install plotly", False),
            ("rasterio", "pip install rasterio", False),
            ("shapely", "pip install shapely", False),
            ("pyarrow", "pip install pyarrow", False),
            ("scipy", "pip install scipy", False),
        ]
        out: List[Check] = []
        for mod, fix, critical in want:
            ok = _try_import(mod) is not None
            out.append(Check(
                name=f"dep.{mod}",
                critical=critical,
                passed=ok,
                details="present" if ok else "missing",
                fix=fix if not ok else None
            ))
        return out

    def check_imports(self) -> List[Check]:
        modules = [
            "backend.bus.streams",
            "backend.engine.strategy_base",
            "backend.engine.ensemble",
            "backend.data.option_chain",
            "backend.analytics.vol_surface",
            "backend.analytics.hrp",
            "backend.oms.market_maker",
            "backend.oms.risk_manager",
            "backend.oms.router",
            "backend.oms.reconciler",
            "backend.broker.broker_base",
            "backend.broker.broker_interface",
            "backend.news.news_yahoo",
            "backend.news.news_moneycontrol",
            "backend.ai.sentiment_ai",
            "backend.ai.insight_agent",
            "backend.ai.query_agent",
            "backend.risk.risk_explainer",
            "backend.altdata.card_spend",
            "backend.altdata.satellite_lights",
            "backend.altdata.shipping_traffic",
        ]
        out: List[Check] = []
        for m in modules:
            try:
                importlib.import_module(m)
                out.append(Check(name=f"import.{m}", critical=False, passed=True, details="ok"))
            except Exception as e:
                out.append(Check(name=f"import.{m}", critical=False, passed=False, details=str(e), fix="Verify path/module or __init__.py"))
        return out

    def check_configs(self) -> List[Check]:
        # best-effort presence + shallow schema
        cfg_dir = "config"
        cfgs = [
            "latency.yaml",
            "registry.yaml",
            "bank_stress.yaml",
            "capital_stress.yaml",
            "governor.yaml",
            "liquidity_surface.yaml",
            "rivals.yaml",
            "shocks.yaml",
            "soverign.yaml",
            "sovereign.yaml",  # in case of alt spelling
            "brand_map.yaml",
        ]
        out: List[Check] = []
        for f in cfgs:
            path = os.path.join(cfg_dir, f)
            doc, err = _read_yaml(path)
            if err == "missing":
                out.append(Check(name=f"config.{f}", critical=False, passed=False, details="missing", fix=f"Create {path}"))
                continue
            if err and err != "missing":
                out.append(Check(name=f"config.{f}", critical=True, passed=False, details=err, fix="Fix YAML syntax"))
                continue
            # shallow checks
            ok = isinstance(doc, (dict, list))
            details = "loaded" if ok else "unexpected_type"
            out.append(Check(name=f"config.{f}", critical=False, passed=ok, details=details))
        return out

    def check_redis(self) -> List[Check]:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        ok, err = _redis_ping(host, port)
        return [Check(
            name="redis.ping",
            critical=True,
            passed=ok,
            details="ok" if ok else f"failed: {err}",
            fix="Ensure Redis is running and REDIS_HOST/REDIS_PORT are correct"
        )]

    def check_sqlite_options(self) -> List[Check]:
        db = "runtime/options.db"
        cx, err = _connect_sqlite(db) if _exists(db) else (None, "missing")
        if not cx:
            return [Check(name="sqlite.options", critical=False, passed=False, details=err or "open_failed", fix="Will be created after first run of option_chain/vol_surface")]
        tables = _sqlite_has_tables(cx, ["options", "fetch_log", "vol_surface"])
        all_ok = all(tables.values())
        return [Check(
            name="sqlite.options.tables",
            critical=False,
            passed=all_ok,
            details=", ".join([f"{k}={'ok' if v else 'missing'}" for k, v in tables.items()]),
            fix="Run: python -m backend.data.option_chain ... then python -m backend.analytics.vol_surface ..."
        )]

    def check_sqlite_altdata(self) -> List[Check]:
        db = "runtime/altdata.db"
        cx, err = _connect_sqlite(db) if _exists(db) else (None, "missing")
        if not cx:
            return [Check(name="sqlite.altdata", critical=False, passed=False, details=err or "open_failed", fix="Will be created by altdata modules on first run")]
        tables = _sqlite_has_tables(cx, [
            "spend_index", "card_tx",
            "regions", "lights_index",
            "shipping_index"
        ])
        ok_any = any(tables.values())
        return [Check(
            name="sqlite.altdata.tables",
            critical=False,
            passed=ok_any,
            details=", ".join([f"{k}={'ok' if v else 'missing'}" for k, v in tables.items()]),
            fix="Run: backend/altdata modules (card_spend, satellite_lights, shipping_traffic) with --probe or real data"
        )]

    def check_bus_topics_optional(self) -> List[Check]:
        # Best-effort: if redis available, check we can LPUSH to a few streams/keys used across the project
        if redis is None:
            return [Check(name="bus.streams", critical=False, passed=False, details="redis-py missing", fix="pip install redis")]
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        r = redis.Redis(host=host, port=port, decode_responses=True)
        out: List[Check] = []
        topics = ["orders.incoming", "alpha.ensemble", "derivs.option_chain", "derivs.vol_surface", "ai.insight"]
        try:
            for t in topics:
                # Streams: XADD if available; else just a key set as proof
                try:
                    r.xadd(t, {"ts": int(time.time()*1000), "msg": "validator"}, maxlen=1000, approximate=True)
                    out.append(Check(name=f"bus.stream.{t}", critical=False, passed=True, details="xadd ok"))
                except Exception:
                    # fallback HSET proof
                    r.hset("validator:bus", t, int(time.time()*1000))
                    out.append(Check(name=f"bus.key.{t}", critical=False, passed=True, details="hset ok"))
        except Exception as e:
            out.append(Check(name="bus.streams", critical=False, passed=False, details=str(e)))
        return out


# ---------- pretty print ----------
def _print_summary(checks: List[Check], *, use_color: bool = True) -> None:
    crit_fail = any(c.critical and not c.passed for c in checks)
    ok = sum(1 for c in checks if c.passed)
    fail = sum(1 for c in checks if not c.passed)
    print("")
    hdr = "PROJECT VALIDATOR SUMMARY"
    print(_color(hdr, "bold"))
    print("-" * len(hdr))
    print(("Status: " + (_color("OK", "green") if not crit_fail else _color("FAIL", "red"))))
    print(f"Checks passed: {ok} | failed: {fail}")
    print("")
    for c in checks:
        status = _color("✓", "green") if c.passed else _color("✗", "red")
        name = c.name
        crit = _color("[CRIT] ", "red") if c.critical and not c.passed else ""
        details = c.details
        print(f"{status} {crit}{name}: {details}")
        if c.fix and not c.passed:
            print(f"    → Fix: {c.fix}")
    print("")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Project Validator / Doctor")
    ap.add_argument("--json", action="store_true", help="Emit JSON results to stdout")
    ap.add_argument("--only", type=str, help="Comma-separated prefixes (e.g., redis,sqlite,configs,dep,import)")
    ap.add_argument("--fail-fast", action="store_true", help="Stop at first critical failure")
    args = ap.parse_args()

    only = [x.strip() for x in (args.only or "").split(",") if x.strip()]
    v = Validator(only=only, fail_fast=args.fail_fast)
    checks = v.run_all()

    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=2))
    else:
        _print_summary(checks)

    # Exit with 1 if any critical failure
    if any(c.critical and not c.passed for c in checks):
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()