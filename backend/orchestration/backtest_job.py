#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import importlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import pandas as pd
import yaml

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "strategies" / "registry"
CONFIGS_DIR = REPO_ROOT / "strategies" / "configs"
CACHE_PROCESSED = REPO_ROOT / "data" / "cache" / "processed" / "daily" / "signals"
REPORTS_DIR = REPO_ROOT / "reports"

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("backtest_job")

# ---------------------------------------------------------------------
# Types & adapters
# ---------------------------------------------------------------------

@dataclass
class JobSpec:
    sid: str                      # strategy id (e.g., BW-0042)
    name: str                     # human name
    family: str                   # e.g., equity_ls / stat_arb / futures_macro / options_vol / credit_cds / cap_struct
    engine: str                   # import path to engine adapter (module:function OR module.runner)
    yaml_path: Path               # config file for this strategy
    tags: List[str]               # tags from registry
    meta: Dict[str, Any]          # any extra registry columns


# Engine adapter signature: Callable[[Path, Dict[str, Any]], Dict[str, Any]]
# It must return a dict with at least {"summary": DataFrame} and (optionally) {"signals": {...}}
Adapter = Callable[[Path, Dict[str, Any]], Dict[str, Any]]

# Registry columns we expect (add more as you like)
REGISTRY_CSV = "all_strategies_master_fullnames.csv"
REGISTRY_JSONL = "all_strategies_master_fullnames.jsonl"

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def _safe_read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

def _ensure_dirs():
    CACHE_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _load_registry() -> pd.DataFrame:
    csv_p = REGISTRY_DIR / REGISTRY_CSV
    jsl_p = REGISTRY_DIR / REGISTRY_JSONL
    if jsl_p.exists():
        rows = [json.loads(line) for line in jsl_p.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        if not df.empty:
            return df
    if csv_p.exists():
        return pd.read_csv(csv_p)
    raise FileNotFoundError(f"Registry not found: {csv_p} or {jsl_p}")

def _filter_registry(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    if args.id:
        ids = [s.strip() for s in args.id.split(",")]
        out = out[out["id"].isin(ids)]
    if args.family:
        fams = [s.strip().lower() for s in args.family.split(",")]
        out = out[out["family"].str.lower().isin(fams)]
    if args.tag:
        tags = {t.strip().lower() for t in args.tag.split(",")}
        def has_tags(x: Any) -> bool:
            if pd.isna(x): return False
            if isinstance(x, str): toks = {t.strip().lower() for t in x.split("|")}
            elif isinstance(x, list): toks = {str(t).lower() for t in x}
            else: toks = {str(x).lower()}
            return len(tags & toks) > 0
        out = out[out["tags"].apply(has_tags)]
    if args.limit:
        out = out.iloc[:int(args.limit)]
    return out

def _to_jobs(df: pd.DataFrame) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for _, r in df.iterrows():
        sid = str(r["id"])
        name = str(r.get("name", sid))
        family = str(r.get("family", "unknown")).lower()
        engine = str(r.get("engine", "")).strip()
        yaml_rel = r.get("yaml", "")
        # default YAML location: strategies/configs/{FIRM}/{SID}.yaml if registry not explicit
        if yaml_rel:
            ypath = CONFIGS_DIR / yaml_rel
        else:
            # try to infer from id prefix BW-/CIT-/P72-
            prefix = sid.split("-")[0]
            ypath = CONFIGS_DIR / prefix / f"{sid}.yaml"
        tags = r.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split("|") if t.strip()]
        meta = {k: r[k] for k in r.index if k not in {"id", "name", "family", "engine", "yaml", "tags"}}
        jobs.append(JobSpec(sid=sid, name=name, family=family, engine=engine, yaml_path=Path(ypath), tags=tags, meta=meta))
    return jobs

def _resolve_adapter(engine_spec: str, family: str) -> Adapter:
    """
    engine_spec can be:
      - empty: we map by family to a default adapter
      - 'package.module:func' → import func
      - 'package.module' → module must expose 'run_from_yaml(config_path, overrides) -> dict'
    """
    # Built-in defaults by family
    default_map: Dict[str, str] = {
        "equity_ls": "orchestrator.eng_adapters.equity_ls:run_from_yaml",
        "stat_arb": "orchestrator.eng_adapters.stat_arb:run_from_yaml",
        "futures_macro": "orchestrator.eng_adapters.futures_macro:run_from_yaml",
        "options_vol": "orchestrator.eng_adapters.options_vol:run_from_yaml",
        "credit_cds": "orchestrator.eng_adapters.credit_cds:run_from_yaml",
        "cap_struct": "orchestrator.eng_adapters.cap_struct:run_from_yaml",
    }
    target = engine_spec or default_map.get(family, "")
    if not target:
        raise ValueError(f"No engine adapter for family={family}; set 'engine' in registry.")
    if ":" in target:
        mod_name, func_name = target.split(":", 1)
        func = getattr(importlib.import_module(mod_name), func_name)
        return func  # type: ignore
    # else expect module with run_from_yaml
    mod = importlib.import_module(target)
    func = getattr(mod, "run_from_yaml")
    return func  # type: ignore

def _write_outputs(sid: str, result: Dict[str, Any], write_csv: bool, write_html: bool):
    _ensure_dirs()
    # Persist signal (if provided)
    signals = result.get("signals")
    if isinstance(signals, dict):
        for key, obj in signals.items():
            # Expect a DataFrame or Series
            if hasattr(obj, "to_csv"):
                out = CACHE_PROCESSED / f"{sid.lower()}__{key}.csv"
                obj.to_csv(out)
                log.info(f"  → signals[{key}] → {out.relative_to(REPO_ROOT)}")
    # Persist summary
    summary = result.get("summary")
    if isinstance(summary, pd.DataFrame):
        path_csv = CACHE_PROCESSED / f"{sid.lower()}__summary.csv"
        summary.to_csv(path_csv)
        log.info(f"  → summary → {path_csv.relative_to(REPO_ROOT)}")
        if write_html:
            html = REPORTS_DIR / f"{sid.lower()}__report.html"
            try:
                summary.tail(200).to_html(html, border=0)
                log.info(f"  → report → {html.relative_to(REPO_ROOT)}")
            except Exception as e:
                log.warning(f"  (html failed) {e}")
    elif isinstance(summary, dict):
        path_json = CACHE_PROCESSED / f"{sid.lower()}__summary.json"
        with path_json.open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info(f"  → summary(json) → {path_json.relative_to(REPO_ROOT)}")

# ---------------------------------------------------------------------
# Engine Adapters (thin wrappers)
# Each adapter must accept a YAML config and return a dict with results.
# You already have several engines; here are default implementations that
# call the modules you built earlier.
# ---------------------------------------------------------------------

# You can keep these thin wrappers in this file or in orchestrator/eng_adapters/*.py

def _adapter_tail_hedge_from_yaml(cfg_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Options tail-hedge engine (engines/options/hedging/tail_hedges.py)
    YAML should include data loader params or paths.
    """
    from engines.options.hedging import tail_hedges as th # type: ignore

    cfg_yaml = _safe_read_yaml(cfg_path)
    cfg_yaml.update(overrides or {})

    # Minimal synthetic loader if user didn't wire data
    # Expect 'index_px' series path; else synth
    idx_px_path = cfg_yaml.get("index_px_path")
    vix_path = cfg_yaml.get("vix_path")

    if idx_px_path and Path(idx_px_path).exists():
        index_px = pd.read_parquet(idx_px_path)["close"]
    else:
        idx = pd.date_range("2020-01-02", periods=750, freq="B")
        rng = pd.Series(pd.Series(1 + 0.0003).reindex(idx)).fillna(1.0003)
        index_px = pd.Series(4_000.0, index=idx) * (1 + 0.0005 * pd.Series(range(len(idx)), index=idx)).astype(float)
        index_px += pd.Series(pd.Series(pd.np.random.normal(0, 5, len(idx))), index=idx)  # type: ignore # noqa

    vix_series = None
    if vix_path and Path(vix_path).exists():
        vix_series = pd.read_parquet(vix_path)["vix"]

    sig = th.build_tail_signals(index_px=index_px, vix_level=vix_series, cfg=th.SignalConfig())
    bt = th.backtest_tail_hedge(index_px=index_px, vix_level=vix_series)
    return {"signals": {"tail_signals": sig}, "summary": bt["summary"]}

def _adapter_hy_vs_ig_from_yaml(cfg_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    from engines.credit import hy_vs_ig as hyig # type: ignore
    cfg_yaml = _safe_read_yaml(cfg_path); cfg_yaml.update(overrides or {})
    hy = _load_series(cfg_yaml.get("hy_path"), default_level=400, noise=0.8)
    ig = _load_series(cfg_yaml.get("ig_path"), default_level=100, noise=0.2)
    sig = hyig.build_signal(hy, ig)
    bt = hyig.backtest(hy, ig)
    return {"signals": {"hy_vs_ig": sig["features"]}, "summary": bt["summary"]}

def _adapter_cds_basis_from_yaml(cfg_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    from engines.cap_struct import cds_basis # type: ignore
    cfg_yaml = _safe_read_yaml(cfg_path); cfg_yaml.update(overrides or {})
    cds = _load_series(cfg_yaml.get("cds_path"), default_level=120, noise=0.4)
    gy  = _load_series(cfg_yaml.get("gov_yield_path"), default_level=0.02, noise=0.00005)
    by  = (gy + 0.015 + _load_series(None, default_level=0.0, noise=0.001, index_like=cds.index))
    sig = cds_basis.build_basis_signal(cds_spread_bps=cds, bond_yield=by, gov_yield=gy)
    bt = cds_basis.backtest(cds_spread_bps=cds, bond_yield=by, gov_yield=gy)
    return {"signals": {"cds_basis": sig["features"]}, "summary": bt["summary"]}

# Map families to built-in adapters (you can override with 'engine' field)
BUILTIN_FAMILY_ADAPTERS: Dict[str, Adapter] = {
    "options_vol": _adapter_tail_hedge_from_yaml,
    "credit": _adapter_hy_vs_ig_from_yaml,
    "cap_struct": _adapter_cds_basis_from_yaml,
}

def _load_series(path: Optional[str], default_level: float, noise: float, index_like: Optional[pd.Index] = None) -> pd.Series:
    if path and Path(path).exists():
        s = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, parse_dates=[0], index_col=0).iloc[:, 0]
        return s.astype(float) # type: ignore
    idx = index_like if index_like is not None else pd.date_range("2020-01-02", periods=600, freq="B")
    rng = pd.Series(0.0, index=idx)
    vals = default_level + pd.Series(pd.np.random.normal(0, noise, len(idx)), index=idx)  # type: ignore # noqa
    return vals.astype(float)

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_one(job: JobSpec, overrides: Dict[str, Any], write_csv: bool, write_html: bool) -> Tuple[str, bool, Optional[str]]:
    try:
        if not job.yaml_path.exists():
            raise FileNotFoundError(f"YAML not found: {job.yaml_path}")
        # Resolve adapter
        if job.family in BUILTIN_FAMILY_ADAPTERS and not job.engine:
            adapter = BUILTIN_FAMILY_ADAPTERS[job.family]
        else:
            adapter = _resolve_adapter(job.engine, job.family)
        log.info(f"[{job.sid}] {job.name} | family={job.family} | yaml={job.yaml_path.relative_to(REPO_ROOT)}")
        result = adapter(job.yaml_path, overrides or {})
        _write_outputs(job.sid, result, write_csv=write_csv, write_html=write_html)
        return (job.sid, True, None)
    except Exception as e:
        log.exception(f"[{job.sid}] failed: {e}")
        return (job.sid, False, str(e))

def main():
    p = argparse.ArgumentParser(description="Run backtests for strategies from the registry.")
    p.add_argument("--id", help="Comma-separated strategy IDs (e.g., BW-0001,CIT-0123)")
    p.add_argument("--family", help="Filter by family (equity_ls, stat_arb, futures_macro, options_vol, credit, cap_struct)")
    p.add_argument("--tag", help="Filter by tags (comma-separated; any match)")
    p.add_argument("--limit", type=int, help="Run only first N after filters")
    p.add_argument("--parallel", type=int, default=1, help="Workers for parallel backtests")
    p.add_argument("--dryrun", action="store_true", help="List jobs and exit")
    p.add_argument("--write-html", action="store_true", help="Emit simple HTML tail report")
    p.add_argument("--override", action="append", default=[], help="Override YAML with key=val (repeatable)")
    p.add_argument("--list", action="store_true", help="List filtered strategies and exit")
    args = p.parse_args()

    df = _load_registry()
    if "id" not in df.columns:
        raise ValueError("Registry must contain an 'id' column.")
    filt = _filter_registry(df, args)
    if filt.empty:
        log.warning("No strategies matched filters.")
        return

    if args.list or args.dryrun:
        cols = ["id", "name", "family", "engine", "yaml", "tags"]
        print(filt[cols].to_string(index=False))
        return

    # Parse overrides (key=val)
    overrides: Dict[str, Any] = {}
    for item in args.override:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        # naive type cast
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        overrides[k] = v

    jobs = _to_jobs(filt)
    _ensure_dirs()

    if args.parallel and args.parallel > 1:
        with futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            results = list(ex.map(lambda j: run_one(j, overrides, True, args.write_html), jobs))
    else:
        results = [run_one(j, overrides, True, args.write_html) for j in jobs]

    ok = sum(1 for _, s, _ in results if s)
    fail = [(sid, err) for sid, s, err in results if not s]
    log.info(f"Done. Success={ok} Fail={len(fail)}")
    if fail:
        for sid, msg in fail:
            log.error(f"  {sid}: {msg}")

if __name__ == "__main__":
    main()