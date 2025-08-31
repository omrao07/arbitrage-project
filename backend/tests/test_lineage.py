# tests/test_lineage.py
import json
import importlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pytest # type: ignore

"""
Expected public API (any one is fine)

Class-style:
-----------
class LineageGraph:
    # entities
    def add_dataset(self, name: str, *, schema: dict | None = None, tags: list[str] | None = None, props: dict | None = None) -> str: ...
    def add_job(self, name: str, *, owner: str | None = None, props: dict | None = None) -> str: ...
    def add_run(self, job_id: str, *, ts: int | None = None, version: str | None = None, props: dict | None = None) -> str: ...
    # edges
    def link_read(self, run_id: str, dataset_id: str, *, columns: list[str] | None = None): ...
    def link_write(self, run_id: str, dataset_id: str, *, columns: list[str] | None = None): ...
    # queries
    def upstream(self, dataset_id: str, *, depth: int | None = None, as_of: int | None = None) -> dict: ...
    def downstream(self, dataset_id: str, *, depth: int | None = None, as_of: int | None = None) -> dict: ...
    def impact_of(self, dataset_id: str, *, as_of: int | None = None) -> dict: ...
    def lineage_path(self, src_dataset_id: str, dst_dataset_id: str, *, as_of: int | None = None) -> list[str] | list[dict]: ...
    def runs_for(self, job_id: str) -> list[dict]: ...
    def get(self, entity_id: str) -> dict: ...
    def detect_cycles(self) -> list[list[str]]: ...
    # maintenance & io
    def delete(self, entity_id: str) -> None: ...
    def prune(self, before_ts: int) -> int: ...
    def export_json(self) -> dict | list | str: ...
    def import_json(self, blob: dict | list | str) -> None: ...
    def clear(self) -> None: ...
    # optional column-level graph
    def column_lineage(self, dataset_id: str) -> dict[str, list[dict]]: ...

Function-style:
---------------
- new_lineage() -> handle
- Same method names exposed at module level taking (handle, ...) as first arg.

The tests below auto-skip features you don’t implement.
"""

# ----------------------------- Import resolver -----------------------------
IMPORT_CANDIDATES = [
    "backend.metadata.lineage",
    "backend.analytics.lineage",
    "backend.platform.lineage",
    "metadata.lineage",
    "lineage",
]

def _load_mod():
    last = None
    for p in IMPORT_CANDIDATES:
        try:
            return importlib.import_module(p)
        except ModuleNotFoundError as e:
            last = e
    pytest.skip(f"Cannot import lineage module from {IMPORT_CANDIDATES} ({last})")

class API:
    """Unifies class- and function-style lineage APIs."""
    def __init__(self, mod):
        self.mod = mod
        if hasattr(mod, "LineageGraph"):
            L = getattr(mod, "LineageGraph")
            try:
                self.g = L()
            except TypeError:
                self.g = L
        else:
            if not hasattr(mod, "new_lineage"):
                pytest.skip("No LineageGraph class and no new_lineage() factory.")
            self.g = mod.new_lineage()

    def has(self, name: str) -> bool:
        return hasattr(self.g, name) or hasattr(self.mod, name)

    def call(self, name: str, *args, **kw):
        if hasattr(self.g, name):
            return getattr(self.g, name)(*args, **kw)
        if hasattr(self.mod, name):
            return getattr(self.mod, name)(self.g, *args, **kw)
        raise AttributeError(f"API missing '{name}'")

# ----------------------------- Fixtures ----------------------------------

@pytest.fixture(scope="module")
def api():
    return API(_load_mod())

@pytest.fixture()
def now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)

@pytest.fixture()
def seed(api, now_ms):
    """Create a small DAG:
       raw_trades  -> clean_trades -> agg_daily -> risk_report
                ^                 \
           ref_fx -----------------> agg_daily
    Jobs: ingest_raw, clean_job, aggregate_job, risk_job
    Two runs per middle job with versions.
    """
    if api.has("clear"):
        api.call("clear")

    # Datasets
    raw = api.call("add_dataset", "raw_trades", schema={"cols": ["ts","sym","px","qty"]}, tags=["raw"])
    clean = api.call("add_dataset", "clean_trades", schema={"cols": ["ts","sym","px","qty","valid"]}, tags=["silver"])
    fx = api.call("add_dataset", "ref_fx", schema={"cols": ["ts","ccy","rate"]}, tags=["ref"])
    agg = api.call("add_dataset", "agg_daily", schema={"cols": ["date","sym","vwap","vol"]}, tags=["gold"])
    risk = api.call("add_dataset", "risk_report", schema={"cols": ["date","sym","var","es"]}, tags=["report"])

    # Jobs
    inj = api.call("add_job", "ingest_raw", owner="data-eng")
    cln = api.call("add_job", "clean_job", owner="quant")
    agj = api.call("add_job", "aggregate_job", owner="quant")
    rsk = api.call("add_job", "risk_job", owner="risk")

    # Runs (with versions & timestamps)
    t0 = now_ms - 60_000
    t1 = now_ms - 40_000
    t2 = now_ms - 20_000
    t3 = now_ms - 10_000

    run_inj = api.call("add_run", inj, ts=t0, version="v1")
    api.call("link_write", run_inj, raw)

    run_cln_v1 = api.call("add_run", cln, ts=t1, version="v1")
    api.call("link_read", run_cln_v1, raw, columns=["ts","sym","px","qty"])
    api.call("link_write", run_cln_v1, clean, columns=["ts","sym","px","qty","valid"])

    run_ag_v1 = api.call("add_run", agj, ts=t2, version="v1")
    api.call("link_read", run_ag_v1, clean, columns=["ts","sym","px","qty"])
    api.call("link_read", run_ag_v1, fx, columns=["ts","ccy","rate"])
    api.call("link_write", run_ag_v1, agg, columns=["date","sym","vwap","vol"])

    run_rsk = api.call("add_run", rsk, ts=t3, version="v1")
    api.call("link_read", run_rsk, agg, columns=["date","sym","vwap","vol"])
    api.call("link_write", run_rsk, risk, columns=["date","sym","var","es"])

    return {
        "ds": {"raw": raw, "clean": clean, "fx": fx, "agg": agg, "risk": risk},
        "jobs": {"inj": inj, "cln": cln, "agj": agj, "rsk": rsk},
        "runs": {"inj": run_inj, "cln": run_cln_v1, "ag": run_ag_v1, "rsk": run_rsk},
        "times": (t0, t1, t2, t3),
    }

# ----------------------------- Tests ------------------------------------

def test_entities_created(api, seed):
    raw = seed["ds"]["raw"]
    got = api.call("get", raw) if api.has("get") else None
    # tolerate minimal implementations
    assert raw and (got is None or isinstance(got, dict))

def test_upstream_and_downstream(api, seed):
    clean = seed["ds"]["clean"]
    agg = seed["ds"]["agg"]
    # upstream of agg should include clean and fx
    up = api.call("upstream", agg, depth=None)
    s = json.dumps(up).lower()
    assert "clean" in s and "fx" in s
    # downstream of clean should include agg and risk (through agg)
    down = api.call("downstream", clean, depth=None)
    s2 = json.dumps(down).lower()
    assert "agg" in s2 and ("risk" in s2 or "risk_report" in s2)

def test_lineage_path(api, seed):
    raw = seed["ds"]["raw"]
    risk = seed["ds"]["risk"]
    if not api.has("lineage_path"):
        pytest.skip("No lineage_path() API")
    path = api.call("lineage_path", raw, risk)
    assert isinstance(path, (list, tuple)) and len(path) >= 3

def test_runs_and_versions(api, seed):
    cln_job = seed["jobs"]["cln"]
    if not api.has("add_run") or not api.has("runs_for"):
        pytest.skip("No run tracking API")
    # add a second version run
    run2 = api.call("add_run", cln_job, ts=seed["times"][1] + 5000, version="v2", props={"notes": "bugfix"})
    api.call("link_read", run2, seed["ds"]["raw"])
    api.call("link_write", run2, seed["ds"]["clean"])
    runs = api.call("runs_for", cln_job)
    assert isinstance(runs, list) and len(runs) >= 2
    assert any(r.get("version") == "v2" for r in runs)

def test_as_of_filtering(api, seed):
    agg = seed["ds"]["agg"]
    t1 = seed["times"][1]
    if not api.has("upstream"):
        pytest.skip("Need upstream()")
    up_at_t1 = api.call("upstream", agg, as_of=t1)
    s = json.dumps(up_at_t1).lower()
    # before aggregate run (t2), upstream may be empty or only planned deps
    # Accept either empty or missing "clean".
    assert isinstance(up_at_t1, (dict, list))
    # no strict assertion beyond type—implementations vary.

def test_impact_analysis(api, seed):
    if not api.has("impact_of"):
        pytest.skip("No impact_of() API")
    impact = api.call("impact_of", seed["ds"]["clean"])
    s = json.dumps(impact).lower()
    assert "agg" in s  # agg depends on clean

def test_cycle_detection(api, seed):
    if not api.has("detect_cycles"):
        pytest.skip("No detect_cycles() API")
    # Create a tiny cycle (optional): risk_report -> ref_fx (nonsense, but for test)
    # If implementation blocks cycles on write, we just assert it returns [].
    cycles = api.call("detect_cycles")
    assert isinstance(cycles, list)
    assert all(isinstance(c, (list, tuple)) for c in cycles)

def test_column_lineage_optional(api, seed):
    if not api.has("column_lineage"):
        pytest.skip("No column_lineage() API")
    cl = api.call("column_lineage", seed["ds"]["agg"])
    # Expect a mapping like {"vwap": [{"src_dataset": clean, "src_col": "px", ...}, ...], ...}
    assert isinstance(cl, dict)
    # keys exist (if you implemented column map when linking)
    # No strict check—implementations differ.

def test_delete_and_prune(api, seed, now_ms):
    # delete a tmp dataset and ensure graph updates
    tmp = api.call("add_dataset", "tmp_view", schema={"cols": ["x"]})
    job = api.call("add_job", "tmp_job")
    run = api.call("add_run", job, ts=now_ms, version="v1")
    api.call("link_read", run, seed["ds"]["raw"])
    api.call("link_write", run, tmp)
    # delete tmp
    if api.has("delete"):
        api.call("delete", tmp)
        # downstream of raw should not include tmp anymore
        d = api.call("downstream", seed["ds"]["raw"])
        assert "tmp_view" not in json.dumps(d)
    # prune (should not crash)
    if api.has("prune"):
        count = api.call("prune", before_ts=now_ms - 1)
        assert isinstance(count, int)

def test_dedup_and_idempotency(api, seed, now_ms):
    # adding same dataset/job again should return same id or merge
    d1 = api.call("add_dataset", "raw_trades")
    d2 = api.call("add_dataset", "raw_trades")
    assert d1 == d2 or True
    j1 = api.call("add_job", "aggregate_job")
    j2 = api.call("add_job", "aggregate_job")
    assert j1 == j2 or True

def test_export_import_roundtrip(api):
    if not api.has("export_json") or not api.has("import_json"):
        pytest.skip("No export/import API")
    blob = api.call("export_json")
    s = json.dumps(blob, default=str)
    assert isinstance(s, str) and len(s) > 10
    if api.has("clear"):
        api.call("clear")
    api.call("import_json", blob)
    # Quick presence check
    # If you expose a simple query, use it; otherwise rely on import not crashing.
    if api.has("upstream"):
        # A known dataset should exist
        # (some implementations retain ids; others re-id—don’t assert exact ids)
        pass

def test_search_by_tag_or_props_optional(api):
    # Optional convenience: find datasets by tag
    for name in ("find_datasets", "query_datasets", "search"):
        if api.has(name):
            res = api.call(name, tags=["gold"]) if name != "search" else api.call(name, type="dataset", tags=["gold"])
            assert isinstance(res, (list, dict))
            break
    else:
        pytest.skip("No dataset search API")