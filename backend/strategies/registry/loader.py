# strategies/registry/loader.py
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# ---- Optional import from types.py (your Pydantic or dataclasses). Fallback if missing.
try:
    from .types import StrategyRow, StrategyConfig  # type: ignore
except Exception:  # lightweight fallback types
    @dataclass
    class StrategyRow:
        id: str
        firm: str
        discipline: str
        family: str
        region: str
        horizon: str
        status: str
        name: str
        description: str
        risk_budget: float
        engine: str
        owner: str
        created_at: str = ""
        updated_at: str = ""

    @dataclass
    class StrategyConfig:
        id: str
        firm: str
        name: str
        description: str
        discipline: str
        family: str
        region: str
        horizon: str
        status: str
        engine: str
        params: dict
        risk: dict
        data: dict
        owner: str = ""
        created_at: str = ""
        updated_at: str = ""


ALLOWED_ENGINES = {"equity_ls", "stat_arb", "futures_macro", "options_vol", "credit_cds"}
REQUIRED_REGISTRY_COLS = [
    "id", "firm", "discipline", "family", "region", "horizon", "status",
    "name", "description", "risk_budget", "engine", "owner", "created_at", "updated_at",
]
REQUIRED_YAML_KEYS = [
    "id", "firm", "name", "discipline", "family", "region",
    "horizon", "status", "engine", "params", "risk"
]

# ----------------------------- tiny YAML reader (no dependency) -----------------------------
_YAML_KEYVAL = re.compile(r"^([A-Za-z0-9_\-]+):(?:\s+(.*))?$")

def _read_yaml_simple(path: Path) -> dict:
    """
    Minimal YAML parser for our simple 'key: value' and nested dict blocks.
    Supports lists as:
      key:
        - item
        - item
    For complex YAML, install PyYAML and replace with yaml.safe_load.
    """
    root: dict = {}
    stack: List[Tuple[int, dict]] = [(-1, root)]
    current_list_key: Optional[str] = None

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            # list item?
            if line.lstrip().startswith("- "):
                if current_list_key is None:
                    # try to attach to the most recent dict key
                    pass
                item = line.lstrip()[2:].strip()
                # auto-parse scalars
                try:
                    val = json.loads(item)
                except Exception:
                    val = item
                parent = stack[-1][1]
                if current_list_key is None:
                    # try to find last key in parent
                    if parent:
                        current_list_key = list(parent.keys())[-1]
                        if not isinstance(parent[current_list_key], list):
                            parent[current_list_key] = []
                if current_list_key is not None:
                    parent[current_list_key].append(val)
                continue

            # key/value
            m = _YAML_KEYVAL.match(line.strip())
            if not m:
                continue
            key, val = m.group(1), (m.group(2) or "")

            # unwind stack by indent
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]

            if val == "":
                parent[key] = {}
                stack.append((indent, parent[key]))
                current_list_key = None
            else:
                # parse JSON-like scalars (numbers, dicts)
                try:
                    parent[key] = json.loads(val)
                except Exception:
                    parent[key] = val
                current_list_key = key if isinstance(parent[key], list) else None

    return root


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return _read_yaml_simple(path)

# --------------------------------- Registry loader ---------------------------------

def _csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in REQUIRED_REGISTRY_COLS if c not in rdr.fieldnames]  # type: ignore
        if missing:
            raise ValueError(f"Registry CSV missing columns: {missing}")
        return list(rdr)

def _jsonl_ids(jsonl_path: Path) -> List[str]:
    ids: List[str] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(str(obj.get("id")))
    return ids

def _validate_registry_rows(rows: List[dict]) -> None:
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dup = [k for k, v in Counter(ids).items() if v > 1]
        raise ValueError(f"Duplicate IDs in registry: {dup[:10]} …")
    for r in rows:
        if r["engine"] not in ALLOWED_ENGINES:
            raise ValueError(f"Unknown engine '{r['engine']}' for {r['id']}")
        try:
            rb = float(r["risk_budget"])
        except Exception:
            raise ValueError(f"Non-numeric risk_budget for {r['id']}: {r['risk_budget']}")
        if not (0 <= rb <= 0.20):
            raise ValueError(f"risk_budget out of range (0–0.20) for {r['id']}: {rb}")

@lru_cache(maxsize=2)
def load_registry(registry_dir: Path) -> Tuple[List[StrategyRow], Dict[str, StrategyRow]]:
    """
    Load registry CSV (and cross-check JSONL if present).
    Returns (rows_list, id_index).
    """
    registry_dir = Path(registry_dir)
    csv_path = registry_dir / "all_strategies_master_fullnames.csv"
    jsonl_path = registry_dir / "all_strategies_master_fullnames.jsonl"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = _csv_rows(csv_path)
    _validate_registry_rows(rows)

    if jsonl_path.exists():
        csv_ids = {r["id"] for r in rows}
        jsonl_ids = set(_jsonl_ids(jsonl_path))
        if csv_ids != jsonl_ids:
            raise ValueError("Mismatch between CSV and JSONL IDs")

    typed: List[StrategyRow] = []
    id_index: Dict[str, StrategyRow] = {}
    for r in rows:
        row = StrategyRow(
            id=r["id"], firm=r["firm"], discipline=r["discipline"], family=r["family"],
            region=r["region"], horizon=r["horizon"], status=r["status"],
            name=r["name"], description=r["description"], risk_budget=float(r["risk_budget"]),
            engine=r["engine"], owner=r.get("owner", ""),
            created_at=r.get("created_at", ""), updated_at=r.get("updated_at", "")
        )
        typed.append(row)
        id_index[row.id] = row
    return typed, id_index

# --------------------------------- Config loader ---------------------------------

def _expected_folder_for_firm(firm: str) -> str:
    if firm == "Bridgewater": return "BW"
    if firm == "Citadel": return "CIT"
    if firm == "Point72": return "P72"
    return "GLB"  # Global / new pack

def _validate_yaml_dict(d: dict, src: Path) -> None:
    missing = [k for k in REQUIRED_YAML_KEYS if k not in d]
    if missing:
        raise ValueError(f"YAML missing keys {missing} in {src}")
    if d.get("engine") not in ALLOWED_ENGINES:
        raise ValueError(f"Bad engine '{d.get('engine')}' in {src}")

def _to_config(d: dict) -> StrategyConfig:
    return StrategyConfig(
        id=str(d["id"]), firm=str(d["firm"]), name=str(d["name"]),
        description=str(d.get("description", "")),
        discipline=str(d["discipline"]), family=str(d["family"]),
        region=str(d["region"]), horizon=str(d["horizon"]),
        status=str(d["status"]), engine=str(d["engine"]),
        params=d.get("params", {}) or {},
        risk=d.get("risk", {}) or {},
        data=d.get("data", {}) or {},
        owner=str(d.get("owner", "")),
        created_at=str(d.get("created_at", "")),
        updated_at=str(d.get("updated_at", "")),
    )

@lru_cache(maxsize=2)
def load_all_configs(configs_dir: Path) -> Dict[str, StrategyConfig]:
    """
    Scan configs/{BW,CIT,P72,GLB}/*.yaml and return a map id -> StrategyConfig
    """
    configs_dir = Path(configs_dir)
    out: Dict[str, StrategyConfig] = {}

    for sub in ["BW", "CIT", "P72", "GLB"]:
        d = configs_dir / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.yaml")):
            raw = _load_yaml(p)
            _validate_yaml_dict(raw, p)
            cfg = _to_config(raw)
            out[cfg.id] = cfg

    if not out:
        raise FileNotFoundError(f"No YAMLs found under {configs_dir} (expected BW/CIT/P72/GLB)")
    return out

# --------------------------------- High-level API ---------------------------------

@dataclass
class StrategyIndex:
    rows: List[StrategyRow]
    by_id: Dict[str, StrategyRow]
    cfg_by_id: Dict[str, StrategyConfig]
    by_firm: Dict[str, List[str]]
    by_engine: Dict[str, List[str]]
    by_family: Dict[str, List[str]]
    by_discipline: Dict[str, List[str]]

def _index(rows: List[StrategyRow], cfg_by_id: Dict[str, StrategyConfig]) -> StrategyIndex:
    by_firm: Dict[str, List[str]] = {}
    by_engine: Dict[str, List[str]] = {}
    by_family: Dict[str, List[str]] = {}
    by_discipline: Dict[str, List[str]] = {}
    by_id = {r.id: r for r in rows}

    for r in rows:
        if r.id not in cfg_by_id:
            raise ValueError(f"Missing YAML for registry id {r.id}")
        by_firm.setdefault(r.firm, []).append(r.id)
        by_engine.setdefault(r.engine, []).append(r.id)
        by_family.setdefault(r.family, []).append(r.id)
        by_discipline.setdefault(r.discipline, []).append(r.id)

    return StrategyIndex(rows, by_id, cfg_by_id, by_firm, by_engine, by_family, by_discipline)

@lru_cache(maxsize=1)
def load_all(strategies_root: Path | str) -> StrategyIndex:
    """
    Load everything from strategies/ root.
    Expected layout:
      strategies/
        registry/all_strategies_master_fullnames.csv
        configs/{BW,CIT,P72,GLB}/*.yaml
    """
    root = Path(strategies_root)
    reg_rows, reg_by_id = load_registry(root / "registry")
    cfg_by_id = load_all_configs(root / "configs")
    return _index(reg_rows, cfg_by_id)

# Convenience helpers
def get(strategies_root: Path | str, strategy_id: str) -> Tuple[StrategyRow, StrategyConfig]:
    idx = load_all(strategies_root)
    if strategy_id not in idx.by_id:
        raise KeyError(f"Unknown strategy id: {strategy_id}")
    return idx.by_id[strategy_id], idx.cfg_by_id[strategy_id]

def list_ids(strategies_root: Path | str,
             firm: Optional[str] = None,
             engine: Optional[str] = None,
             discipline: Optional[str] = None,
             family: Optional[str] = None) -> List[str]:
    idx = load_all(strategies_root)
    ids = [r.id for r in idx.rows]
    if firm:
        ids = [i for i in ids if idx.by_id[i].firm == firm]
    if engine:
        ids = [i for i in ids if idx.by_id[i].engine == engine]
    if discipline:
        ids = [i for i in ids if idx.by_id[i].discipline == discipline]
    if family:
        ids = [i for i in ids if idx.by_id[i].family == family]
    return ids

def iter_configs(strategies_root: Path | str) -> Iterator[StrategyConfig]:
    return iter(load_all(strategies_root).cfg_by_id.values())

def load_bundle(strategies_root: Path | str, strategy_id: str) -> dict:
    """Return a dict with registry row + YAML config merged, ready for engines."""
    row, cfg = get(strategies_root, strategy_id)
    out = asdict(row)
    out.update({
        "params": cfg.params,
        "risk": cfg.risk,
        "data": cfg.data,
    })
    return out

# --------------------------------- CLI ---------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Load/validate strategy registry + configs")
    ap.add_argument("root", help="Path to strategies/ folder")
    ap.add_argument("--list", action="store_true", help="List first 10 IDs")
    ap.add_argument("--check", action="store_true", help="Validate counts & mapping")
    args = ap.parse_args()

    idx = load_all(args.root)
    print(f"Loaded rows: {len(idx.rows)}; YAMLs: {len(idx.cfg_by_id)}")
    if args.list:
        print("First 10:", [r.id for r in idx.rows[:10]])
    if args.check:
        # sanity: ids parity
        missing = [r.id for r in idx.rows if r.id not in idx.cfg_by_id]
        extra = [k for k in idx.cfg_by_id.keys() if k not in {r.id for r in idx.rows}]
        if missing:
            print("[ERROR] Missing YAML for:", missing[:5], "…")
        if extra:
            print("[WARN] Extra YAML not in registry:", extra[:5], "…")
        if not missing:
            print("Registry ↔ YAML parity OK.")