#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit.py
--------
End-to-end audit helpers for your quant stack.

Commands
- validate : schema checks for config/registry + data sanity (no backtest)
- run      : everything in validate + smoke backtest slice
- runfile  : audit an existing history parquet

Examples
python audit.py validate --config configs/portfolio.yaml --registry configs/registry.yaml --data data/spx.csv --out runs/audit
python audit.py run --config configs/portfolio.yaml --registry configs/registry.yaml --out runs/audit
python audit.py runfile --history runs/bt_xxx/history_XXXXXXXX.parquet --out runs/audit_from_history
"""

from __future__ import annotations
import os, re, json, math, uuid, time, importlib, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yaml  # optional
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# Optional orchestrator import (graceful fallback)
try:
    from orchestrator import Orchestrator, load_yaml_or_json as _orc_load#type:ignore
except Exception:
    Orchestrator = None
    _orc_load = None


# ------------------------------- utils ---------------------------------------

def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.lower().endswith((".yaml", ".yml")) and _HAS_YAML:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except Exception:
        if not _HAS_YAML:
            raise
        return yaml.safe_load(text)

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _pct(x: float) -> str: return f"{100*x:.2f}%"


# ------------------------------- metrics -------------------------------------

def perf_metrics(equity: pd.Series, freq: str = "D") -> Dict[str, float]:
    equity = equity.dropna()
    if equity.empty: return {}
    rets = equity.pct_change().dropna()
    ann = {"D":252, "H":24*252, "W":52, "M":12}.get(freq.upper(), 252)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ann / max(1, len(rets))) - 1.0
    vol = rets.std(ddof=1) * math.sqrt(ann) if len(rets)>1 else float("nan")
    sharpe = rets.mean()/(rets.std(ddof=1)+1e-12)*math.sqrt(ann) if len(rets)>1 else float("nan")
    dr = rets[rets<0]
    sortino = rets.mean()/(dr.std(ddof=1)+1e-12)*math.sqrt(ann) if len(dr)>1 else float("nan")
    roll_max = equity.cummax()
    maxdd = (equity/(roll_max+1e-12) - 1.0).min()
    return {"cagr":float(cagr), "vol":float(vol), "sharpe":float(sharpe),
            "sortino":float(sortino), "max_drawdown":float(maxdd)}

def turnover_from_fills(hist: pd.DataFrame) -> float:
    if not {"fills","price","equity"} <= set(hist.columns): return float("nan")
    acc, n = 0.0, 0
    for _, r in hist.iterrows():
        fills = r["fills"] or {}
        px = float(r["price"]); eq = max(1e-12, float(r["equity"]))
        notion = sum(abs(q)*px for q in fills.values())
        acc += notion/eq; n += 1
    return float(acc/max(1,n))


# ------------------------------ results --------------------------------------

@dataclass
class AuditResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Auditor:
    out_dir: Path
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("audit"))

    # ---------- config & registry ----------

    def validate_config(self, cfg: Dict[str, Any]) -> AuditResult:
        errs, warns = [], []
        req = ["starting_equity", "risk_limits", "execution", "data"]
        for k in req:
            if k not in cfg: errs.append(f"Config missing '{k}'")
        rl = cfg.get("risk_limits", {})
        for k in ["gross","net","max_drawdown"]:
            if k not in rl: warns.append(f"risk_limits missing '{k}'")
            else:
                try: float(rl[k])
                except Exception: errs.append(f"risk_limits['{k}'] must be numeric")
        ex = cfg.get("execution", {})
        for k in ["slippage_bps","fee_bps"]:
            v = ex.get(k)
            if v is None: warns.append(f"execution missing '{k}'")
            else:
                try: float(v)
                except Exception: errs.append(f"execution['{k}'] must be numeric")
        if "csv" not in cfg.get("data", {}):
            warns.append("data.csv not specified (ok if API feeds are used)")
        return AuditResult(ok=not errs, errors=errs, warnings=warns, info={"config_keys": list(cfg.keys())})

    def validate_registry(self, reg: Dict[str, Any]) -> AuditResult:
        errs, warns = [], []
        strategies = reg.get("strategies") or reg
        if not isinstance(strategies, dict) or not strategies:
            return AuditResult(ok=False, errors=["Registry must contain non-empty 'strategies' dict."])
        for name, meta in strategies.items():
            if not isinstance(meta, dict):
                errs.append(f"{name}: metadata must be mapping"); continue
            if not (meta.get("module") or meta.get("function")):
                errs.append(f"{name}: need 'module'+'class' OR 'function'")
            if meta.get("module") and not meta.get("class"):
                warns.append(f"{name}: module without class (ok if function-based)")
            if "params" in meta and not isinstance(meta["params"], dict):
                errs.append(f"{name}: 'params' must be dict")
        return AuditResult(ok=not errs, errors=errs, warnings=warns, info={"count": len(strategies)})

    def importability_check(self, reg: Dict[str, Any]) -> AuditResult:
        errs, warns = [], []
        strategies = reg.get("strategies") or reg
        for name, meta in strategies.items():
            mod, cls, fn = meta.get("module"), meta.get("class"), meta.get("function")
            try:
                if mod and cls:
                    m = importlib.import_module(mod); getattr(m, cls)
                elif fn and "." in fn:
                    pkg, attr = fn.rsplit(".",1); m = importlib.import_module(pkg); getattr(m, attr)
                else:
                    warns.append(f"{name}: unresolved target (noop fallback)")
            except Exception as e:
                errs.append(f"{name}: import failed -> {e}")
        return AuditResult(ok=not errs, errors=errs, warnings=warns)

    # ------------------------------ data --------------------------------

    def validate_data(self, df: pd.DataFrame) -> AuditResult:
        errs, warns = [], []
        if "price" not in df.columns:
            errs.append("Data missing 'price' column (or rename close/px/last -> price)")
        if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            if not df.index.is_monotonic_increasing:
                errs.append("Datetime index must be monotonic increasing")
            if df.index.duplicated().any():
                errs.append(f"Duplicate timestamps: {int(df.index.duplicated().sum())}")
        na_cols = [c for c in df.columns if df[c].isna().any()]
        if na_cols: warns.append(f"NaNs in: {', '.join(na_cols[:10])}" + ("..." if len(na_cols)>10 else ""))
        if "price" in df.columns:
            p = df["price"].astype(float)
            if (p <= 0).any(): errs.append("Non-positive prices detected")
            jumps = p.pct_change().abs().dropna()
            if (jumps > 0.5).mean() > 0.02: warns.append("Huge price jumps >50% in >2% rows (splits/rolls?)")
        return AuditResult(ok=not errs, errors=errs, warnings=warns, info={"rows": int(len(df))})

    # --------------------------- backtest --------------------------------

    def smoke_backtest(self, cfg: Dict[str, Any], reg: Dict[str, Any]) -> AuditResult:
        if Orchestrator is None:
            return AuditResult(ok=False, errors=["orchestrator.py not available on PYTHONPATH"])
        # load data (use orchestrator helper if present)
        df = None
        if _orc_load:
            try:
                from orchestrator import _load_data_for_backtest  # type: ignore
                df = _load_data_for_backtest(cfg)
            except Exception as e:
                return AuditResult(ok=False, errors=[f"data load failed: {e}"])
        if df is None:
            path = cfg.get("data", {}).get("csv")
            if not path:
                return AuditResult(ok=False, errors=["No data csv to backtest"])
            df = pd.read_csv(path)
            for c in ["timestamp","date","time","datetime"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c]); df = df.set_index(c).sort_index(); break
            if "price" not in df.columns:
                for c in ["close","px","last"]:
                    if c in df.columns: df = df.rename(columns={c:"price"}); break

        vd = self.validate_data(df)
        if not vd.ok: return vd

        steps = cfg.get("audit", {}).get("smoke_steps", 252)
        df_slice = df.iloc[-min(steps, len(df)):].copy()

        run_dir = _ensure_dir(self.out_dir / f"smoke_{uuid.uuid4().hex[:8]}")
        orch = Orchestrator(config=cfg, registry=reg, out_dir=run_dir, mode="backtest", paper=True)
        res = orch.run_backtest(df_slice)
        hist_path = Path(res["history_path"])
        hist = pd.read_parquet(hist_path)

        met = perf_metrics(hist["equity"])
        tov = turnover_from_fills(hist)

        alerts = []
        rl = cfg.get("risk_limits", {})
        if rl:
            if "gross" in rl and "weights" in hist.columns:
                gross = hist["weights"].apply(lambda d: sum(abs(v) for v in (d or {}).values())).max()
                if gross - float(rl["gross"]) > 1e-12:
                    alerts.append(f"Gross exceeded: max {gross:.3f} > cap {float(rl['gross']):.3f}")
            if "net" in rl and "weights" in hist.columns:
                net_max = abs(hist["weights"].apply(lambda d: sum((d or {}).values()))).max()
                if net_max - float(rl["net"]) > 1e-12:
                    alerts.append(f"Net exceeded: max |net| {net_max:.3f} > cap {float(rl['net']):.3f}")
            if "max_drawdown" in rl and met.get("max_drawdown", 0.0) < -abs(float(rl["max_drawdown"])):
                alerts.append(f"MaxDD breached: {met['max_drawdown']:.3f} < -{abs(float(rl['max_drawdown'])):.3f}")

        info = {"history_path": str(hist_path), "metrics": met | {"turnover": float(tov)},
                "n_steps": int(len(hist)), "alerts": alerts}
        return AuditResult(ok=(len(alerts)==0), errors=[], warnings=[] if len(alerts)==0 else alerts, info=info)

    # ---------------------------- reporting --------------------------------

    def write_report(self, name: str, results: Dict[str, AuditResult]) -> Dict[str, str]:
        out = _ensure_dir(self.out_dir)
        jpath, mpath = out / f"{name}.json", out / f"{name}.md"
        data = {k: {"ok":v.ok, "errors":v.errors, "warnings":v.warnings, "info":v.info} for k,v in results.items()}
        jpath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        lines = [f"# Audit Report: {name}", ""]
        for sec, res in results.items():
            lines += [f"## {sec}",
                      f"- **Status**: {'✅ OK' if res.ok else '❌ FAIL'}"]
            if res.errors:
                lines.append("- **Errors**:"); lines += [f"  - {e}" for e in res.errors]
            if res.warnings:
                lines.append("- **Warnings**:"); lines += [f"  - {w}" for w in res.warnings]
            if res.info:
                lines.append("- **Info**:"); lines += [f"  - **{k}**: {v}" for k,v in res.info.items()]
            lines.append("")
        mpath.write_text("\n".join(lines), encoding="utf-8")
        return {"json": str(jpath), "markdown": str(mpath)}


# --------------------------------- CLI ---------------------------------------

def _setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Audit utilities for quant stack")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate config/registry/data")
    p_val.add_argument("--config", required=True)
    p_val.add_argument("--registry", required=True)
    p_val.add_argument("--data", required=False, help="CSV override for data sanity")
    p_val.add_argument("--out", required=True)
    p_val.add_argument("--log", default="INFO")

    p_run = sub.add_parser("run", help="Validate + smoke backtest")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--registry", required=True)
    p_run.add_argument("--data", required=False, help="CSV override for backtest")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--log", default="INFO")

    p_file = sub.add_parser("runfile", help="Audit an existing history parquet")
    p_file.add_argument("--history", required=True)
    p_file.add_argument("--out", required=True)
    p_file.add_argument("--log", default="INFO")

    args = ap.parse_args()
    _setup_logging(args.log)

    out_dir = _ensure_dir(args.out)
    auditor = Auditor(out_dir=out_dir)

    if args.cmd in ("validate", "run"):
        cfg = _load_yaml_or_json(args.config)
        reg = _load_yaml_or_json(args.registry)

        if args.data:
            cfg = dict(cfg); cfg.setdefault("data", {}); cfg["data"]["csv"] = args.data

        res_cfg = auditor.validate_config(cfg)
        res_reg = auditor.validate_registry(reg)
        res_imp = auditor.importability_check(reg)

        # data sanity
        data_path = cfg.get("data", {}).get("csv")
        if data_path:
            try:
                df = pd.read_csv(data_path)
                for c in ["timestamp","date","time","datetime"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c]); df = df.set_index(c).sort_index(); break
                if "price" not in df.columns:
                    for c in ["close","px","last"]:
                        if c in df.columns: df = df.rename(columns={c:"price"}); break
                res_data = auditor.validate_data(df)
            except Exception as e:
                res_data = AuditResult(ok=False, errors=[f"Failed to load/validate data: {e}"])
        else:
            res_data = AuditResult(ok=False, errors=["No data.csv provided; cannot validate data."])

        results = {"config":res_cfg, "registry":res_reg, "importability":res_imp, "data":res_data}

        if args.cmd == "run":
            results["smoke_backtest"] = auditor.smoke_backtest(cfg, reg)

        paths = auditor.write_report("audit", results)
        print(json.dumps({"status":"ok", "report":paths}, indent=2))
        return

    if args.cmd == "runfile":
        hist = pd.read_parquet(args.history)
        if "equity" not in hist.columns:
            raise SystemExit("History parquet missing 'equity' column.")
        met = perf_metrics(hist["equity"])
        tov = turnover_from_fills(hist)
        res = AuditResult(ok=True, info={"metrics": met | {"turnover": float(tov)}, "n_steps": int(len(hist))})
        auditor.write_report("audit_from_history", {"runfile": res})
        print(json.dumps({"status":"ok", "metrics":res.info["metrics"]}, indent=2))
        return


if __name__ == "__main__":
    main()