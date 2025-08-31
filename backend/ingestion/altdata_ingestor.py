# backend/altdata/altdata_ingestor.py
from __future__ import annotations

import os, sys, json, csv, time, math, hashlib, threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Iterable, Tuple, Callable

# -------- optional deps (graceful) -------------------------------------------
HAVE_YAML = True
try:
    import yaml  # type: ignore
except Exception:
    HAVE_YAML = False
    yaml = None  # type: ignore

HAVE_PD = True
try:
    import pandas as pd  # type: ignore
except Exception:
    HAVE_PD = False
    pd = None  # type: ignore

HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

# -------- env / defaults -----------------------------------------------------
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STREAM_ALT     = os.getenv("ALT_STREAM", "altdata.events")           # unified stream
OUT_DIR        = os.getenv("ALT_OUT_DIR", "artifacts/altdata")
CFG_PATH       = os.getenv("ALT_CFG", "configs/altdata.yaml")
DEDUPE_TTL_SEC = int(os.getenv("ALT_DEDUPE_TTL_SEC", "86400"))       # 1 day
BATCH_SIZE     = int(os.getenv("ALT_BATCH_SIZE", "500"))
FLUSH_SEC      = int(os.getenv("ALT_FLUSH_SEC", "2"))

# -------- schemas (lightweight) ---------------------------------------------
BASE_FIELDS = {
    "ts_ms": int,
    "source": str,      # e.g., "card_spend", "sat_lights", "ais_shipping", "social_sent"
    "entity": str,      # e.g., ticker, port_code, region_id, store_id
    "value": (int, float, str),
    "metric": str,      # e.g., "spend_index", "luminosity", "vessels", "sentiment"
}
GEO_FIELDS = {"lat": float, "lon": float, "bbox": list}  # optional

def _type_ok(key: str, val: Any, t) -> bool:
    if val is None: return False
    if isinstance(t, tuple): return isinstance(val, t)
    return isinstance(val, t)

# -------- data models --------------------------------------------------------
@dataclass
class AltRecord:
    ts_ms: int
    source: str
    entity: str
    value: Any
    metric: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    _id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# -------- persistence backend -----------------------------------------------
class _Backend:
    def __init__(self, redis_url: Optional[str] = None):
        self.r = None
        if HAVE_REDIS:
            try:
                self.r = Redis.from_url(redis_url or REDIS_URL, decode_responses=True)  # type: ignore
                self.r.ping()
            except Exception:
                self.r = None
        os.makedirs(OUT_DIR, exist_ok=True)
        self._buf: List[Dict[str, Any]] = []
        self._last_flush = time.time()

    def xadd(self, stream: str, obj: Dict[str, Any]) -> None:
        if self.r:
            try:
                self.r.xadd(stream, {"json": json.dumps(obj)}, maxlen=100_000, approximate=True)  # type: ignore
                return
            except Exception:
                pass
        # fallback: buffer to JSONL
        self._buf.append(obj)
        now = time.time()
        if len(self._buf) >= BATCH_SIZE or (now - self._last_flush) >= FLUSH_SEC:
            path = os.path.join(OUT_DIR, f"{int(now)}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                for row in self._buf:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._buf.clear()
            self._last_flush = now

# -------- throttle & dedupe --------------------------------------------------
class _Throttle:
    def __init__(self):
        self._last: Dict[str, int] = {}  # key -> last ts_sec

    def hit(self, key: str, per_seconds: int) -> bool:
        now = int(time.time())
        last = self._last.get(key, 0)
        if now - last >= max(1, per_seconds):
            self._last[key] = now
            return True
        return False

class _Dedupe:
    def __init__(self):
        self._seen: Dict[str, int] = {}   # id -> ts_sec

    def make_id(self, rec: AltRecord) -> str:
        base = {
            "ts_ms": rec.ts_ms,
            "source": rec.source,
            "entity": rec.entity,
            "metric": rec.metric,
            "value": rec.value,
            "lat": rec.lat, "lon": rec.lon, "bbox": rec.bbox,
            "meta": rec.meta,
        }
        blob = json.dumps(base, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def seen(self, _id: str) -> bool:
        tnow = int(time.time())
        # purge stale
        dead = [k for k, t in self._seen.items() if (tnow - t) > DEDUPE_TTL_SEC]
        for k in dead: self._seen.pop(k, None)
        if _id in self._seen:
            return True
        self._seen[_id] = tnow
        return False

# -------- adapters (plugin interface) ---------------------------------------
class BaseAdapter:
    name: str = "base"
    metric: str = "value"

    def iterate(self) -> Iterable[AltRecord]:  # pull mode (optional)
        return []

    def parse_row(self, row: Dict[str, Any]) -> AltRecord:
        ts_ms = int(row.get("ts_ms") or row.get("timestamp_ms") or int(time.time()*1000))
        entity = str(row.get("entity") or row.get("ticker") or row.get("region") or "UNKNOWN")
        value = row.get("value")
        metric = str(row.get("metric") or self.metric or "value")
        lat = row.get("lat"); lon = row.get("lon"); bbox = row.get("bbox")
        meta = {k:v for k,v in row.items() if k not in ("ts_ms","entity","value","metric","lat","lon","bbox")}
        return AltRecord(ts_ms=ts_ms, source=self.name, entity=entity, value=value, metric=metric,
                         lat=lat, lon=lon, bbox=bbox, meta=meta)

    def validate(self, rec: AltRecord) -> Tuple[bool, str]:
        for k, t in BASE_FIELDS.items():
            if not hasattr(rec, k): return False, f"missing {k}"
            if not _type_ok(k, getattr(rec, k), t): return False, f"type {k}"
        if rec.lat is not None and not (-90.0 <= float(rec.lat) <= 90.0): return False, "lat_out_of_range"
        if rec.lon is not None and not (-180.0 <= float(rec.lon) <= 180.0): return False, "lon_out_of_range"
        if rec.bbox:
            try:
                mnL, mnB, mxR, mxT = rec.bbox
                if not (-180<=mnL<=mxR<=180 and -90<=mnB<=mxT<=90): return False, "bbox_invalid"
            except Exception:
                return False, "bbox_invalid"
        return True, ""

    def enrich(self, rec: AltRecord) -> AltRecord:
        if rec.lat is not None: rec.lat = round(float(rec.lat), 6)
        if rec.lon is not None: rec.lon = round(float(rec.lon), 6)
        rec.entity = rec.entity.upper()
        return rec

class CardSpendAdapter(BaseAdapter):
    name = "card_spend"
    metric = "spend_index"

class SatLightsAdapter(BaseAdapter):
    name = "sat_lights"
    metric = "luminosity"

class ShippingAISAdapter(BaseAdapter):
    name = "ais_shipping"
    metric = "vessels"

class SocialSentAdapter(BaseAdapter):
    name = "social_sent"
    metric = "sentiment"

# -------- Ingestor -----------------------------------------------------------
class AltDataIngestor:
    def __init__(self, *, redis_url: Optional[str] = None):
        self.backend = _Backend(redis_url)
        self.adapters: Dict[str, BaseAdapter] = {}
        self.throttle = _Throttle()
        self.dedupe = _Dedupe()

    def register(self, adapter: BaseAdapter):
        self.adapters[adapter.name] = adapter

    def ingest_records(self, records: Iterable[AltRecord], *, throttle_sec: int = 0) -> Dict[str, int]:
        stats = {"ok":0, "dupe":0, "bad":0}
        for rec in records:
            ok, reason = self._validate_and_enrich(rec)
            if not ok:
                stats["bad"] += 1
                continue
            key = f"{rec.source}:{rec.entity}:{rec.metric}"
            if throttle_sec>0 and not self.throttle.hit(key, throttle_sec):
                continue
            rec._id = rec._id or self.dedupe.make_id(rec)
            if self.dedupe.seen(rec._id):
                stats["dupe"] += 1
                continue
            self.backend.xadd(STREAM_ALT, rec.to_dict())
            stats["ok"] += 1
        return stats

    def ingest_file(self, path: str, *, adapter: Optional[str] = None, throttle_sec: int = 0) -> Dict[str,int]:
        ad = self._pick_adapter(adapter or self._infer_adapter(path))
        rows = self._read_rows(path)
        recs = (ad.parse_row(row) for row in rows)
        return self.ingest_records(recs, throttle_sec=throttle_sec)

    def pull_once(self, adapter: str, *, throttle_sec: int = 0) -> Dict[str,int]:
        ad = self._pick_adapter(adapter)
        return self.ingest_records(ad.iterate(), throttle_sec=throttle_sec)

    # ---- helpers
    def _validate_and_enrich(self, rec: AltRecord) -> Tuple[bool, str]:
        ad = self.adapters.get(rec.source)
        if ad:
            ok, why = ad.validate(rec)
            if not ok: return False, why
            _ = ad.enrich(rec)
        else:
            for k, t in BASE_FIELDS.items():
                if not hasattr(rec, k): return False, f"missing {k}"
                if not _type_ok(k, getattr(rec, k), t): return False, f"type {k}"
        return True, ""

    def _pick_adapter(self, name: str) -> BaseAdapter:
        if name not in self.adapters:
            for b in [CardSpendAdapter(), SatLightsAdapter(), ShippingAISAdapter(), SocialSentAdapter()]:
                self.adapters.setdefault(b.name, b)
        ad = self.adapters.get(name)
        if not ad:
            raise ValueError(f"unknown adapter '{name}'")
        return ad

    @staticmethod
    def _infer_adapter(path: str) -> str:
        base = os.path.basename(path).lower()
        if base.startswith("card_"): return "card_spend"
        if base.startswith("sat_"):  return "sat_lights"
        if base.startswith("ais_"):  return "ais_shipping"
        if base.startswith("social_"): return "social_sent"
        return "card_spend"

    @staticmethod
    def _read_rows(path: str) -> Iterable[Dict[str, Any]]:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jsonl", ".json"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        try:
                            arr = json.loads(line)
                            if isinstance(arr, list):
                                for obj in arr: yield obj
                        except Exception:
                            continue
        elif ext in (".csv",):
            with open(path, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r: yield row
        elif HAVE_PD and ext in (".parquet", ".pq"):
            df = pd.read_parquet(path)  # type: ignore
            for rec in df.to_dict("records"): yield rec # type: ignore
        else:
            raise ValueError(f"unsupported file type: {ext}")

# -------- CLI ----------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("altdata_ingestor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest-file", help="Ingest a CSV/JSON(L)/Parquet file")
    ing.add_argument("--path", required=True)
    ing.add_argument("--adapter", default=None, help="card_spend | sat_lights | ais_shipping | social_sent")
    ing.add_argument("--throttle-sec", type=int, default=0)

    pull = sub.add_parser("pull-once", help="Call adapter.iterate() once and ingest results")
    pull.add_argument("--adapter", required=True)
    pull.add_argument("--throttle-sec", type=int, default=0)

    mock = sub.add_parser("mock", help="Emit N mock events of a given type")
    mock.add_argument("--adapter", default="card_spend")
    mock.add_argument("--n", type=int, default=10)
    mock.add_argument("--entity", default="AAPL")
    mock.add_argument("--metric", default=None)

    args = ap.parse_args()
    ingestor = AltDataIngestor()

    if args.cmd == "ingest-file":
        stats = ingestor.ingest_file(args.path, adapter=args.adapter, throttle_sec=args.throttle_sec)
        print(json.dumps({"ok":True, "stats":stats}, indent=2))

    elif args.cmd == "pull-once":
        stats = ingestor.pull_once(args.adapter, throttle_sec=args.throttle_sec)
        print(json.dumps({"ok":True, "stats":stats}, indent=2))

    elif args.cmd == "mock":
        ad = ingestor._pick_adapter(args.adapter)
        metric = (args.metric or getattr(ad, "metric", "value"))
        now = int(time.time()*1000)
        recs = []
        for i in range(args.n):
            recs.append(AltRecord(
                ts_ms=now + i*1000,
                source=ad.name,
                entity=args.entity,
                value=round(100 + 5*math.sin(i/5.0), 3),
                metric=metric,
                lat=37.77, lon=-122.41,
                meta={"mock":True, "i":i}
            ))
        stats = ingestor.ingest_records(recs, throttle_sec=0)
        print(json.dumps({"ok":True, "stats":stats}, indent=2))

if __name__ == "__main__":
    _cli()