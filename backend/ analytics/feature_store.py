# backend/data/feature_store.py
from __future__ import annotations

import os, time, json, math, hashlib, threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

# ---------- Optional deps (all graceful) -------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None  # type: ignore
    pq = None  # type: ignore

# ---------- Environment / Defaults ------------------------------------------
REDIS_URL            = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FEATURE_REGISTRY_KEY = os.getenv("FEATURE_REGISTRY_KEY", "features.registry")   # HSET name -> json(spec)
FEATURE_NS           = os.getenv("FEATURE_NS", "feature")                       # key namespace
FEATURE_STREAM       = os.getenv("FEATURE_STREAM", "features.signals")          # ingest
AI_SIGNALS_STREAM    = os.getenv("AI_SIGNALS_STREAM", "ai.signals")             # optional ingest
PARQUET_DIR          = os.getenv("FEATURE_PARQUET_DIR", "artifacts/features")   # offline store root
PARQUET_ROLL_BYTES   = int(os.getenv("FEATURE_PARQUET_ROLL_BYTES", "134217728")) # ~128MB

# ---------- Small utils ------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

def _safe_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, default=float_if_needed)
    except Exception:
        return json.dumps(str(x))

def float_if_needed(x: Any):
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
    except Exception:
        pass
    return x

def _ensure_dir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

# ---------- Registry ---------------------------------------------------------
@dataclass
class FeatureSpec:
    """
    name: unique name (e.g., "analyst_confidence")
    dtype: "float","int","bool","str","json","vector"
    version: int for evolution (breaking change)
    entity: primary entity key ("symbol","account","strategy", etc.)
    freshness_ms: how often you expect this to update (SLO)
    ttl_ms: expiry for online cache
    desc: human text
    """
    name: str
    dtype: str
    version: int = 1
    entity: str = "symbol"
    freshness_ms: int = 60_000
    ttl_ms: int = 300_000
    desc: str = ""

# ---------- FeatureStore -----------------------------------------------------
class FeatureStore:
    """
    Online (Redis) + offline (Parquet) feature store.

    Online layout (per latest value):
      HSET feature:{entity_key} -> { "<name>@v<version>": JSON({ts_ms,value,meta}) }
      TTL applied via EXPIRE (whole entity hash) or per-key shadow

    Offline layout (append-only):
      artifacts/features/<name>/v<version>/YYYYMMDD/part-*.parquet
      Columns: ts_ms, entity_key, value, meta (JSON string)
    """
    def __init__(self, redis_url: str = REDIS_URL, parquet_dir: str = PARQUET_DIR):
        self.redis_url = redis_url
        self.parquet_dir = parquet_dir
        self.r: Optional[AsyncRedis] = None # type: ignore

    # ---- connections --------------------------------------------------------
    async def connect(self):
        if not HAVE_REDIS:
            self.r = None
            return
        try:
            self.r = AsyncRedis.from_url(self.redis_url, decode_responses=True) # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    # ---- registry CRUD ------------------------------------------------------
    async def register(self, spec: FeatureSpec) -> None:
        if self.r:
            await self.r.hset(FEATURE_REGISTRY_KEY, spec.name, _safe_json(asdict(spec)))
        else:
            print("[feature_store] registry (no-redis):", asdict(spec))

    async def describe(self, name: str) -> Optional[FeatureSpec]:
        if self.r:
            raw = await self.r.hget(FEATURE_REGISTRY_KEY, name)
            if raw:
                try:
                    return FeatureSpec(**json.loads(raw))
                except Exception:
                    return None
        return None

    async def list_features(self) -> List[FeatureSpec]:
        out: List[FeatureSpec] = []
        if self.r:
            h = await self.r.hgetall(FEATURE_REGISTRY_KEY)
            for _, v in (h or {}).items():
                try:
                    out.append(FeatureSpec(**json.loads(v)))
                except Exception:
                    continue
        return out

    # ---- type guards --------------------------------------------------------
    @staticmethod
    def _coerce(dtype: str, value: Any) -> Any:
        try:
            if value is None: return None
            if dtype == "float": return float(value)
            if dtype == "int":   return int(value)
            if dtype == "bool":  return bool(value)
            if dtype == "str":   return str(value)
            if dtype == "json":  return value  # expect dict/list/primitive
            if dtype == "vector":
                if isinstance(value, (list, tuple)): return [float(v) for v in value]
                raise TypeError("vector requires list/tuple")
        except Exception as e:
            raise TypeError(f"cannot coerce value to {dtype}: {e}")
        return value

    # ---- keys ----------------------------------------------------------------
    @staticmethod
    def _key_entity(entity_key: str) -> str:
        return f"{FEATURE_NS}:{entity_key}"

    @staticmethod
    def _field_name(name: str, version: int) -> str:
        return f"{name}@v{version}"

    # ---- online writes ------------------------------------------------------
    async def put(
        self,
        *,
        name: str,
        entity_key: str,
        value: Any,
        version: int = 1,
        dtype: Optional[str] = None,
        ts_ms: Optional[int] = None,
        ttl_ms: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Upsert latest value online; append offline parquet row (best-effort).
        """
        ts = int(ts_ms or now_ms())
        spec = await self.describe(name)
        dtyp = dtype or (spec.dtype if spec else "json")
        use_ttl = int(ttl_ms or (spec.ttl_ms if spec else 0))
        v = self._coerce(dtyp, value)

        payload = {"ts_ms": ts, "value": v, "meta": meta or {}}
        field = self._field_name(name, version if version else (spec.version if spec else 1))
        hash_key = self._key_entity(entity_key)

        # Online write
        if self.r:
            await self.r.hset(hash_key, field, _safe_json(payload))
            if use_ttl > 0:
                await self.r.expire(hash_key, use_ttl // 1000)
        else:
            print("[feature_store] put (no-redis):", hash_key, field, payload)

        # Offline append (best-effort)
        self._append_parquet_row(name, version if version else 1, ts, entity_key, v, meta or {})

        return {"ok": True, "ts_ms": ts}

    # ---- online reads -------------------------------------------------------
    async def get(
        self,
        *,
        name: str,
        entity_key: str,
        version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Return {"ts_ms":..., "value":..., "meta":...} or None
        """
        field = self._field_name(name, version or (await self._resolve_version(name)))
        key = self._key_entity(entity_key)
        if self.r:
            raw = await self.r.hget(key, field)
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None

    async def mget_latest(self, *, entity_key: str) -> Dict[str, Dict[str, Any]]:
        """
        Return all features for an entity (latest snapshot), mapped by "<name>@vX".
        """
        key = self._key_entity(entity_key)
        if self.r:
            h = await self.r.hgetall(key)
            out: Dict[str, Dict[str, Any]] = {}
            for f, raw in (h or {}).items():
                try:
                    out[f] = json.loads(raw)
                except Exception:
                    continue
            return out
        return {}

    async def _resolve_version(self, name: str) -> int:
        spec = await self.describe(name)
        return spec.version if spec else 1

    # ---- offline (Parquet) --------------------------------------------------
    def _parquet_path(self, name: str, version: int, ts_ms: int) -> str:
        day = time.strftime("%Y%m%d", time.gmtime(ts_ms/1000))
        base = os.path.join(self.parquet_dir, name, f"v{version}", day)
        _ensure_dir(base)
        fname = f"part-{sha1(f'{name}|{version}|{day}|{(ts_ms//3600000)}')}.parquet"
        return os.path.join(base, fname)

    def _append_parquet_row(self, name: str, version: int, ts_ms: int, entity_key: str, value: Any, meta: Dict[str, Any]):
        if pd is None or pq is None:
            return  # offline disabled
        try:
            df = pd.DataFrame([{
                "ts_ms": int(ts_ms),
                "entity_key": str(entity_key),
                "value": json.dumps(value) if isinstance(value, (dict, list)) else value,
                "meta": json.dumps(meta, ensure_ascii=False),
            }])
            path = self._parquet_path(name, version, ts_ms)
            # append-or-create
            if not os.path.exists(path):
                table = pa.Table.from_pandas(df) # type: ignore
                pq.write_table(table, path)
            else:
                # small row-group append
                with pq.ParquetWriter(path, pa.Table.from_pandas(df).schema, use_dictionary=True) as writer: # type: ignore
                    writer.write_table(pa.Table.from_pandas(df)) # type: ignore
            # naive roll control by size
            if os.path.getsize(path) > PARQUET_ROLL_BYTES:
                os.rename(path, path.replace(".parquet", f".{int(ts_ms)}.parquet"))
        except Exception:
            pass

    # ---- windows / aggregations --------------------------------------------
    async def materialize_window(
        self,
        *,
        name: str,
        version: int,
        entity_key: str,
        source_name: str,
        source_version: int,
        window_ms: int,
        agg: str = "mean",
    ) -> Optional[Dict[str, Any]]:
        """
        Compute a windowed aggregation from offline rows for `source_name`
        and publish online as `name`.
        Requires pandas/pyarrow present.
        """
        if pd is None or pq is None:
            return None
        ts = now_ms()
        since = ts - window_ms
        # scan day folders (simple; for production use a catalog)
        rows: List[pd.DataFrame] = [] # type: ignore
        root = os.path.join(self.parquet_dir, source_name, f"v{source_version}")
        if not os.path.isdir(root):
            return None
        for day in sorted(os.listdir(root)):
            try:
                day_int = int(day)
            except Exception:
                continue
            # quick filter by day; read if overlaps window
            # (treat day as UTC day, crude but fine)
            for fname in os.listdir(os.path.join(root, day)):
                if not fname.endswith(".parquet"): continue
                path = os.path.join(root, day, fname)
                try:
                    tbl = pq.read_table(path, columns=["ts_ms","entity_key","value"])
                    df = tbl.to_pandas()
                    df = df[(df["entity_key"] == entity_key) & (df["ts_ms"] >= since)]
                    if not df.empty:
                        rows.append(df)
                except Exception:
                    continue
        if not rows:
            return None
        df = pd.concat(rows, ignore_index=True)
        # coerce value
        try:
            vals = pd.to_numeric(df["value"], errors="coerce")
        except Exception:
            return None
        vals = vals.dropna()
        if vals.empty:
            return None
        if agg == "mean":
            out = float(vals.mean())
        elif agg == "sum":
            out = float(vals.sum())
        elif agg == "max":
            out = float(vals.max())
        elif agg == "min":
            out = float(vals.min())
        else:
            raise ValueError("unsupported agg")
        # publish
        return await self.put(name=name, entity_key=entity_key, value=out, version=version, dtype="float", ts_ms=ts)

    # ---- stream ingestion worker -------------------------------------------
    async def run_ingest(self, streams: Optional[List[str]] = None, maxlen_log: int = 20000):
        """
        Tail feature streams and upsert online & offline.
        Expected records:
          XADD <stream> * json '{"ts_ms":..., "symbol":"TSLA", "features":{"analyst_confidence":0.77,...}}'
        OR:
          XADD ai.signals * json '{"ts_ms":..., "symbol":"TSLA", "score":0.42, "confidence":0.71, ...}'
        """
        if not HAVE_REDIS:
            raise RuntimeError("Redis not available")
        streams = streams or [FEATURE_STREAM, AI_SIGNALS_STREAM]
        r: AsyncRedis = AsyncRedis.from_url(self.redis_url, decode_responses=True)  # type: ignore
        await r.ping()

        last_ids = {s: "$" for s in streams}
        while True:
            try:
                resp = await r.xread({s: last_ids[s] for s in streams}, count=200, block=5000)
                if not resp:
                    continue
                for stream_key, entries in resp:
                    for _id, fields in entries:
                        last_ids[stream_key] = _id
                        try:
                            obj = json.loads(fields.get("json", "{}"))
                        except Exception:
                            continue
                        await self._ingest_record(obj, stream_key)
            except Exception:
                await _sleep(0.5)

    async def _ingest_record(self, obj: Dict[str, Any], source: str):
        ts = int(obj.get("ts_ms") or now_ms())
        symbol = (obj.get("symbol") or obj.get("entity_key") or "").upper()
        if not symbol:
            return
        # Case A: generic feature bundle
        feats = obj.get("features")
        if isinstance(feats, dict):
            for k, v in feats.items():
                # try registry; if unknown default json type
                spec = await self.describe(k)
                dtype = spec.dtype if spec else ("float" if isinstance(v, (int, float)) else "json")
                await self.put(name=k, entity_key=symbol, value=v, version=(spec.version if spec else 1),
                               dtype=dtype, ts_ms=ts, meta={"source": source})
            return
        # Case B: ai.signals shorthand
        if "score" in obj or "confidence" in obj:
            # derive two features
            if "score" in obj:
                await self.put(name="analyst_score", entity_key=symbol, value=float(obj["score"]),
                               version=1, dtype="float", ts_ms=ts, meta={"source": source})
            if "confidence" in obj:
                await self.put(name="analyst_confidence", entity_key=symbol, value=float(obj["confidence"]),
                               version=1, dtype="float", ts_ms=ts, meta={"source": source})

# ---------- helpers ----------------------------------------------------------
async def _sleep(sec: float):
    try:
        import asyncio
        await asyncio.sleep(sec)
    except Exception:
        time.sleep(sec)

# ---------- quick demo -------------------------------------------------------
async def _demo():
    fs = FeatureStore()
    await fs.connect()
    # register a couple features
    await fs.register(FeatureSpec(name="analyst_confidence", dtype="float", desc="Analyst agent confidence 0..1", ttl_ms=600_000))
    await fs.register(FeatureSpec(name="sentiment_mc", dtype="float", desc="Moneycontrol sentiment 0..1", ttl_ms=600_000))
    # put
    await fs.put(name="analyst_confidence", entity_key="TSLA", value=0.76, version=1, dtype="float", meta={"src":"demo"})
    await fs.put(name="sentiment_mc", entity_key="RELIANCE", value=0.61, version=1, dtype="float")
    # get
    print(await fs.get(name="analyst_confidence", entity_key="TSLA"))
    # windowed materialize (requires parquet/pandas)
    await fs.materialize_window(name="analyst_conf_1h", version=1, entity_key="TSLA",
                                source_name="analyst_confidence", source_version=1, window_ms=3600_000, agg="mean")

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass