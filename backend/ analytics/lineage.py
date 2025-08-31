# backend/infra/lineage.py
from __future__ import annotations

import os, io, json, time, hashlib, inspect, typing as T
from dataclasses import dataclass, asdict, field
from pathlib import Path
from contextlib import contextmanager

# -------- optional Redis (graceful) ------------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    USE_REDIS = False
    AsyncRedis = None  # type: ignore

REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LINEAGE_STREAM   = os.getenv("LINEAGE_STREAM", "lineage.events")
LINEAGE_DIR      = Path(os.getenv("LINEAGE_DIR", "artifacts/lineage"))
LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

# -------- small utils --------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(obj))

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(chunk)
            if not c: break
            h.update(c)
    return h.hexdigest()

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# -------- core entities ------------------------------------------------------
@dataclass
class Asset:
    kind: str                      # "dataset" | "model" | "feature" | "config" | "artifact" | "metric"
    name: str
    version: str | None = None
    uri: str | None = None         # path, s3://, db://, redis key, etc.
    hash: str | None = None        # SHA256 or logical hash
    meta: dict = field(default_factory=dict)

@dataclass
class Run:
    run_id: str
    ts_ms: int
    actor: str                     # e.g., "experiments.sweep", "backtester", "feature_store"
    purpose: str                   # "train" | "backtest" | "materialize" | "ingest" | "analysis" | ...
    params: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

@dataclass
class Edge:
    src: str
    dst: str
    relation: str                  # "consumes" | "produces" | "derives" | "evaluates" | ...
    meta: dict = field(default_factory=dict)

# -------- in-memory graph ----------------------------------------------------
class LineageGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}  # id -> node dict
        self.edges: list[Edge] = []

    def add_node(self, nid: str, payload: dict):
        self.nodes[nid] = {**payload, "id": nid}

    def add_edge(self, e: Edge):
        self.edges.append(e)

    def to_json(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": [asdict(e) for e in self.edges]}

    def to_dot(self) -> str:
        def esc(s: str) -> str: return s.replace('"', r'\"')
        lines = ["digraph lineage {", '  rankdir=LR; node [shape=box,fontname="Inter"];']
        for nid, n in self.nodes.items():
            label = n.get("label") or n.get("name") or nid
            lines.append(f'  "{esc(nid)}" [label="{esc(label)}\\n({esc(n.get("type",""))})"];')
        for e in self.edges:
            lines.append(f'  "{esc(e.src)}" -> "{esc(e.dst)}" [label="{esc(e.relation)}"];')
        lines.append("}")
        return "\n".join(lines)

# -------- store & broker -----------------------------------------------------
class LineageStore:
    """
    Persists lineage graphs per run as JSON sidecars; can also publish events to Redis.
    """
    def __init__(self, root: Path = LINEAGE_DIR, redis_url: str = REDIS_URL):
        self.root = Path(root)
        self.redis_url = redis_url
        self._r: AsyncRedis | None = None # type: ignore

    async def connect(self):
        if not USE_REDIS: return
        try:
            self._r = AsyncRedis.from_url(self.redis_url, decode_responses=True)  # type: ignore
            await self._r.ping() # type: ignore
        except Exception:
            self._r = None

    async def emit(self, obj: dict):
        if self._r:
            try:
                await self._r.xadd(LINEAGE_STREAM, {"json": _safe_json(obj)}, maxlen=20000, approximate=True)  # type: ignore
            except Exception:
                pass

    def save_graph(self, run_id: str, graph: LineageGraph):
        p = self.root / run_id / "lineage.json"
        _ensure_parent(p)
        p.write_text(_safe_json(graph.to_json()))

    def save_dot(self, run_id: str, graph: LineageGraph):
        p = self.root / run_id / "lineage.dot"
        _ensure_parent(p)
        p.write_text(graph.to_dot())

# -------- public API ---------------------------------------------------------
class Lineage:
    """
    High-level façade to record lineage for a single run.
    """
    def __init__(self, run: Run, store: LineageStore | None = None):
        self.run = run
        self.graph = LineageGraph()
        self.store = store or LineageStore()
        # seed nodes
        self._add_node(self._run_node_id(), {
            "type": "run", "label": f"{run.actor}::{run.purpose}", "name": run.run_id,
            "ts_ms": run.ts_ms, "params": run.params, "tags": run.tags
        })

    def _run_node_id(self) -> str:
        return f"run::{self.run.run_id}"

    def _asset_node_id(self, a: Asset) -> str:
        v = a.version or "v0"
        return f"asset::{a.kind}::{a.name}::{v}"

    def _add_node(self, nid: str, payload: dict): self.graph.add_node(nid, payload)
    def _add_edge(self, src: str, dst: str, relation: str, meta: dict | None = None):
        self.graph.add_edge(Edge(src=src, dst=dst, relation=relation, meta=meta or {}))

    # ---- asset helpers ----
    def register_asset(self, a: Asset) -> str:
        nid = self._asset_node_id(a)
        self._add_node(nid, {
            "type": "asset", "name": a.name, "kind": a.kind, "version": a.version or "v0",
            "uri": a.uri, "hash": a.hash, "meta": a.meta
        })
        return nid

    def link(self, src_asset: Asset, dst_asset: Asset, relation: str, meta: dict | None = None):
        s = self.register_asset(src_asset)
        d = self.register_asset(dst_asset)
        self._add_edge(s, d, relation, meta)

    def consumes(self, asset: Asset, meta: dict | None = None):
        a = self.register_asset(asset)
        self._add_edge(a, self._run_node_id(), "consumed_by", meta)

    def produces(self, asset: Asset, meta: dict | None = None):
        a = self.register_asset(asset)
        self._add_edge(self._run_node_id(), a, "produces", meta)

    def derives(self, src: Asset, dst: Asset, meta: dict | None = None):
        self.link(src, dst, "derives", meta)

    # ---- annotations ----
    def record_metric(self, name: str, value: float, meta: dict | None = None):
        mid = f"metric::{self.run.run_id}::{name}"
        self._add_node(mid, {"type":"metric","name":name,"value":float(value), "meta": meta or {}})
        self._add_edge(self._run_node_id(), mid, "emits")

    def note(self, text: str, meta: dict | None = None):
        nid = f"note::{self.run.run_id}::{len([n for n in self.graph.nodes if n.startswith('note::')])}"
        self._add_node(nid, {"type":"note","text":text,"meta":meta or {}})
        self._add_edge(self._run_node_id(), nid, "annotates")

    # ---- persistence ----
    async def flush(self, also_dot: bool = True):
        self.store.save_graph(self.run.run_id, self.graph)
        if also_dot:
            self.store.save_dot(self.run.run_id, self.graph)
        await self.store.emit({"ts_ms": now_ms(), "kind":"flush", "run_id": self.run.run_id})

# -------- convenience: context & decorator -----------------------------------
@contextmanager
def lineage_run(actor: str, purpose: str, *, params: dict | None = None, tags: list[str] | None = None) -> T.Iterator[Lineage]:
    """
    Context manager that creates a run, yields a Lineage object, and flushes on exit.
    """
    run_id = f"{int(time.time())}-{hash(actor+purpose) & 0xFFFF:04x}"
    run = Run(run_id=run_id, ts_ms=now_ms(), actor=actor, purpose=purpose, params=params or {}, tags=tags or [])
    lin = Lineage(run)
    try:
        yield lin
    finally:
        # flush synchronously
        import asyncio
        try:
            asyncio.run(lin.flush())
        except Exception:
            pass

def capture_lineage(actor: str, purpose: str):
    """
    Decorator that wraps a function, auto-capturing inputs, outputs and exceptions as lineage.
    - Captures: function name, module, file, args/kwargs (repr), return type/size (repr)
    - If the return value looks like a file/path, it will be recorded as an artifact with hash.
    """
    def deco(fn):
        def _hash_if_path(x) -> tuple[str|None, str|None]:
            try:
                p = Path(str(x))
                if p.exists() and p.is_file():
                    return str(p), sha256_file(p)
            except Exception:
                pass
            return None, None

        def wrap(*args, **kwargs):
            with lineage_run(actor, purpose, params={"fn": fn.__name__}) as lin:
                # inputs
                lin.note(f"call {fn.__module__}.{fn.__name__}", {
                    "file": inspect.getsourcefile(fn),
                    "args": [repr(a)[:400] for a in args],
                    "kwargs": {k: repr(v)[:200] for k,v in kwargs.items()}
                })
                try:
                    out = fn(*args, **kwargs)
                    # try to recognize outputs
                    path, h = _hash_if_path(out)
                    if path:
                        lin.produces(Asset(kind="artifact", name=Path(path).name, uri=path, hash=h, meta={"from":"decorator"}))
                    else:
                        # primitive metric? list/len?
                        try:
                            if isinstance(out, (int, float)) and not isinstance(out, bool):
                                lin.record_metric("return_value", float(out))
                            elif hasattr(out, "__len__"):
                                lin.record_metric("return_len", float(len(out)))  # type: ignore
                        except Exception:
                            pass
                    return out
                except Exception as e:
                    lin.note("exception", {"type": type(e).__name__, "msg": str(e)})
                    raise
        return wrap
    return deco

# -------- helpers to fingerprint and build assets ----------------------------
def asset_from_path(kind: str, path: str | Path, name: str | None = None, version: str | None = None, meta: dict | None = None) -> Asset:
    p = Path(path)
    h = sha256_file(p) if p.exists() and p.is_file() else None
    return Asset(kind=kind, name=(name or p.name), version=version, uri=str(p), hash=h, meta=meta or {})

def asset_from_bytes(kind: str, name: str, payload: bytes, version: str | None = None, meta: dict | None = None) -> Asset:
    h = sha256_bytes(payload)
    # optionally persist
    out = LINEAGE_DIR / "blobs" / name
    _ensure_parent(out)
    out.write_bytes(payload)
    return Asset(kind=kind, name=name, version=version, uri=str(out), hash=h, meta=meta or {})

# -------- quick examples -----------------------------------------------------
def _demo():
    # 1) Context manager
    with lineage_run("experiments", "backtest", params={"strategy":"mm_core","start":"2024-01-01"}) as lin:
        raw = asset_from_path("dataset", "data/minute/AAPL.parquet", version="v1")
        cfg = Asset(kind="config", name="bt_cfg", version="v2", uri="config/experiments.yaml", hash=None, meta={"alpha":0.99})
        lin.consumes(raw, {"why":"source bars"})
        lin.consumes(cfg)
        # produce PnL curve artifact
        png = asset_from_bytes("artifact", "pnl.png", b"fake-png", version="v1")
        lin.produces(png, {"format":"png"})
        lin.record_metric("sharpe", 1.42)
        lin.note("Good spread at open")

    # 2) Decorator
    @capture_lineage("analytics", "compute_corr")
    def compute_corr(a, b):  # toy
        import math
        return (sum(a)*sum(b)) / max(1, len(a)*len(b)) + math.pi

    compute_corr([1,2,3], [4,5,6])

if __name__ == "__main__":
    _demo()