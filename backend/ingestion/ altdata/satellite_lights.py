# backend/altdata/satellite_lights.py
"""
Satellite Night Lights (VIIRS/DMSP) -> Economic Activity Index
--------------------------------------------------------------
Features
- Ingest night-lights GeoTIFFs (e.g., VIIRS monthly VNP46A3, NPP VCMCFG, DMSP OLS*)
- Aggregate brightness (radiance / digital number) over:
    * Regions from GeoJSON polygons (admin-1/2, custom AOIs)
    * Facility points (CSV lat/lon -> company/site), with radius buffer
- Build daily/weekly indices, compute DoD/WoW/MoM/YoY, seasonality-lite trend, anomaly z-score
- Persist to SQLite runtime/altdata.db; export CSV; publish insights to bus

Notes
- Raster download is out-of-scope; point this at already-downloaded GeoTIFFs.
- Units vary by product; we store raw mean/sum + meta.source for clarity.
- This is engineering scaffolding, not scientific ground-truth. Calibrate per dataset.

CLI examples
  # Register regions & ingest a VIIRS tile for 2025-08-01
  python -m backend.altdata.satellite_lights --regions config/regions.geojson
  python -m backend.altdata.satellite_lights --load VNP46A3_A2025213.tif --date 2025-08-01 --source viirs_m --index --publish

  # Facilities (points) -> company tickers
  python -m backend.altdata.satellite_lights --facilities config/facilities.csv --lat LAT --lon LON --id company --ticker ticker
  python -m backend.altdata.satellite_lights --load tile.tif --date 2025-08-01 --source viirs_m --radius 1500 --index

File formats
- GeoTIFF: projected or WGS84; we reproject if rasterio/shapely present
- GeoJSON: FeatureCollection; each feature needs: {"properties": {"id": "<REGION_ID>", "name": "..."}}
- Facilities CSV: columns for id (e.g., company/site), lat, lon; optional ticker column

Optional deps (auto-graceful)
  rasterio, shapely, geopandas, numpy, pandas
"""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import statistics as stats
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional stack
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import rasterio  # type: ignore
    from rasterio import features as rio_features  # type: ignore
    from rasterio.warp import transform_geom  # type: ignore
except Exception:
    rasterio = None  # type: ignore
    rio_features = None  # type: ignore
    transform_geom = None  # type: ignore

try:
    from shapely.geometry import shape, Point, mapping  # type: ignore
    from shapely.ops import unary_union  # type: ignore
except Exception:
    shape = Point = mapping = unary_union = None  # type: ignore

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

DB_PATH = "runtime/altdata.db"

# ----------------------------- utils -----------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _parse_date(d: str) -> str:
    # Accept YYYY-MM-DD or YYYYMMDD
    d = str(d).strip()
    if len(d) == 8 and d.isdigit():
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    return d

def _week_tag(day: str) -> str:
    from datetime import datetime
    dt = datetime.strptime(day, "%Y-%m-%d")
    y, w, _ = dt.isocalendar()
    return f"{y:04d}-W{int(w):02d}"

def _growth_metrics(days: List[str], vals: List[float]) -> Dict[str, Dict[str, float]]:
    from datetime import datetime, timedelta
    idx = {d: i for i, d in enumerate(days)}
    out: Dict[str, Dict[str, float]] = {}
    for i, d in enumerate(days):
        v = vals[i]
        dt = datetime.strptime(d, "%Y-%m-%d")
        def pct(prev_val):
            if prev_val is None or prev_val <= 0:
                return None
            return (v - prev_val) / prev_val * 100.0
        d1  = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        d7  = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        d30 = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
        d365= (dt - timedelta(days=365)).strftime("%Y-%m-%d")
        out[d] = { # type: ignore
            "dod": pct(vals[idx[d1]])  if d1  in idx else None,
            "wow": pct(vals[idx[d7]])  if d7  in idx else None,
            "mom": pct(vals[idx[d30]]) if d30 in idx else None,
            "yoy": pct(vals[idx[d365]])if d365 in idx else None,
        }
    return out

def _trend_anomaly(vals: List[float]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    if not vals:
        return [], []
    if pd is not None:
        s = pd.Series(vals, dtype="float64")
        trend = s.ewm(span=10, adjust=False).mean().tolist()
    else:
        # simple rolling mean
        q = []; acc = 0.0; trend = []
        for v in vals:
            q.append(v); acc += v
            if len(q) > 10: acc -= q.pop(0)
            trend.append(acc/len(q))
    med = stats.median(vals)
    mad = stats.median([abs(x-med) for x in vals]) or 1e-6
    z = [max(-6.0, min(6.0, (x-med)/(1.4826*mad))) for x in vals]
    return trend, z # type: ignore

# ----------------------------- storage -----------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS regions (
  id TEXT PRIMARY KEY,
  name TEXT,
  type TEXT,         -- 'polygon'|'point'
  props TEXT,        -- JSON
  geom_geojson TEXT  -- GeoJSON geometry (WGS84)
);
CREATE TABLE IF NOT EXISTS lights_obs (
  day TEXT,                  -- YYYY-MM-DD
  source TEXT,               -- viirs_m | npp_vcmcfg | dmsp_ols | custom
  level TEXT,                -- 'region'|'facility'
  key TEXT,                  -- region_id or facility_id
  mean_dn REAL,
  sum_dn REAL,
  pct_area_lit REAL,         -- % pixels above threshold
  pixels INTEGER,
  PRIMARY KEY(day, source, level, key)
);
CREATE INDEX IF NOT EXISTS ix_lights_obs_key ON lights_obs(level, key);

CREATE TABLE IF NOT EXISTS lights_index (
  level TEXT,
  key TEXT,
  day TEXT,
  mean_dn REAL,
  sum_dn REAL,
  pct_area_lit REAL,
  dod REAL, wow REAL, mom REAL, yoy REAL,
  trend REAL,
  anomaly REAL,
  meta TEXT,
  PRIMARY KEY(level, key, day)
);
"""

class LightsStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        _ensure_dir(db_path)
        with self._cx() as cx:
            cx.executescript(_SCHEMA)

    def _cx(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path, timeout=60.0)
        cx.row_factory = sqlite3.Row
        return cx

    # -------- regions/facilities --------
    def upsert_regions_geojson(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features") or []
        n = 0
        with self._cx() as cx:
            for ft in feats:
                props = ft.get("properties") or {}
                rid   = str(props.get("id") or props.get("ID") or props.get("name")).strip()
                name  = str(props.get("name") or rid).strip()
                geom  = ft.get("geometry")
                if not rid or not geom: continue
                cx.execute("INSERT OR REPLACE INTO regions(id,name,type,props,geom_geojson) VALUES(?,?,?,?,?)",
                           (rid, name, "polygon", json.dumps(props), json.dumps(geom)))
                n += 1
            cx.commit()
        return n

    def upsert_facilities_csv(self, path: str, id_col: str = "id", lat_col: str = "lat",
                              lon_col: str = "lon", ticker_col: Optional[str] = None) -> int:
        n = 0
        with self._cx() as cx, open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    rid = str(r[id_col]).strip()
                    lat = float(r[lat_col]); lon = float(r[lon_col])
                except Exception:
                    continue
                props = {k: r[k] for k in r.keys()}
                if ticker_col and ticker_col in r:
                    props["ticker"] = r[ticker_col]
                pt = {"type":"Point","coordinates":[lon, lat]}
                cx.execute("INSERT OR REPLACE INTO regions(id,name,type,props,geom_geojson) VALUES(?,?,?,?,?)",
                           (rid, r.get("name", rid), "point", json.dumps(props), json.dumps(pt)))
                n += 1
            cx.commit()
        return n

    def list_regions(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM regions"
        args: List[Any] = []
        if kind:
            q += " WHERE type=?"; args.append(kind)
        with self._cx() as cx:
            return [dict(r) for r in cx.execute(q, args).fetchall()]

    # -------- write obs --------
    def write_obs(self, day: str, source: str, level: str, key: str,
                  mean_dn: float, sum_dn: float, pct_area_lit: float, pixels: int) -> None:
        with self._cx() as cx:
            cx.execute("""
                INSERT OR REPLACE INTO lights_obs(day, source, level, key, mean_dn, sum_dn, pct_area_lit, pixels)
                VALUES(?,?,?,?,?,?,?,?)
            """, (day, source, level, key, float(mean_dn or 0.0), float(sum_dn or 0.0), float(pct_area_lit or 0.0), int(pixels or 0)))
            cx.commit()

    # -------- index (growth/trend) --------
    def build_index(self) -> int:
        written = 0
        with self._cx() as cx:
            keys = cx.execute("SELECT DISTINCT level, key FROM lights_obs").fetchall()
            for row in keys:
                level, key = row["level"], row["key"]
                series = cx.execute("""
                    SELECT day, mean_dn, sum_dn, pct_area_lit
                    FROM lights_obs WHERE level=? AND key=? ORDER BY day
                """, (level, key)).fetchall()
                days = [r["day"] for r in series]
                vals = [float(r["mean_dn"] or 0.0) for r in series]  # focus on mean brightness
                growth = _growth_metrics(days, vals)
                trend, anom = _trend_anomaly(vals)
                for i, d in enumerate(days):
                    g = growth.get(d, {})
                    cx.execute("""
                        INSERT OR REPLACE INTO lights_index(level,key,day,mean_dn,sum_dn,pct_area_lit,
                                                            dod,wow,mom,yoy,trend,anomaly,meta)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (level, key, d, series[i]["mean_dn"], series[i]["sum_dn"], series[i]["pct_area_lit"],
                          g.get("dod"), g.get("wow"), g.get("mom"), g.get("yoy"),
                          trend[i] if i < len(trend) else None,
                          anom[i] if i < len(anom) else None,
                          json.dumps({})))
                    written += 1
            cx.commit()
        return written

    def latest(self, level: str, key: str, n: int = 30) -> List[Dict[str, Any]]:
        with self._cx() as cx:
            rows = cx.execute("""
                SELECT * FROM lights_index WHERE level=? AND key=? ORDER BY day DESC LIMIT ?
            """, (level, key, n)).fetchall()
        return [dict(r) for r in rows]

    def export_csv(self, path: str) -> str:
        _ensure_dir(path)
        with self._cx() as cx, open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = [c[1] for c in cx.execute("PRAGMA table_info(lights_index)").fetchall()]
            w.writerow(cols)
            for r in cx.execute("SELECT * FROM lights_index ORDER BY level,key,day").fetchall():
                w.writerow([r[c] for c in cols])
        return path

# ----------------------------- raster ingest -----------------------------

@dataclass
class RasterMeta:
    path: str
    day: str
    source: str = "viirs_m"   # viirs_m | npp_vcmcfg | dmsp_ols | custom
    nodata: Optional[float] = None
    lit_threshold: Optional[float] = None  # pixel considered "lit" if value >= threshold
    band: int = 1

def _read_raster_stats_over_geom(meta: RasterMeta, geom_geojson: Dict[str, Any]) -> Tuple[float, float, float, int]:
    """
    Returns (mean_dn, sum_dn, pct_area_lit, pixels) for given geometry.
    Requires rasterio & shapely; if unavailable -> raises with clear message.
    """
    assert rasterio is not None and shape is not None and rio_features is not None, \
        "Please install rasterio and shapely for geospatial aggregation: pip install rasterio shapely"
    geom = geom_geojson
    with rasterio.open(meta.path) as ds:
        # Reproject geometry to raster CRS if needed
        g = geom
        if transform_geom and ds.crs and ds.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            try:
                g = transform_geom("EPSG:4326", ds.crs.to_string(), geom, precision=6)
            except Exception:
                g = geom
        mask = rio_features.geometry_mask([g], out_shape=(ds.height, ds.width), transform=ds.transform, invert=True)
        arr = ds.read(meta.band, masked=False).astype("float64")
        nd = meta.nodata if meta.nodata is not None else (ds.nodata if ds.nodata not in (None, 0) else None)
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr) if np is not None else arr
        # Apply mask
        if np is not None:
            vals = arr[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return (0.0, 0.0, 0.0, 0)
            mean_dn = float(np.mean(vals))
            sum_dn  = float(np.sum(vals))
            thr = meta.lit_threshold if meta.lit_threshold is not None else np.nanpercentile(vals, 60.0)
            pct_lit = float((vals >= thr).sum() / max(1, vals.size)) * 100.0
            return (mean_dn, sum_dn, pct_lit, int(vals.size))
        else:
            # Slow fallback (no numpy) — iterate row/col
            h, w = mask.shape
            cnt = 0; s = 0.0; s2 = 0.0; lit = 0
            # simple percentile proxy for threshold later
            vals_tmp: List[float] = []
            for r in range(h):
                for c in range(w):
                    if not mask[r, c]:
                        continue
                    v = float(arr[r, c])
                    if nd is not None and v == nd:
                        continue
                    vals_tmp.append(v); s += v; cnt += 1
            if cnt == 0: return (0.0, 0.0, 0.0, 0)
            mean_dn = s / cnt
            vals_tmp.sort()
            thr = vals_tmp[int(0.6 * (len(vals_tmp)-1))]
            for v in vals_tmp:
                if v >= thr: lit += 1
                s2 += v
            pct_lit = lit / len(vals_tmp) * 100.0
            return (mean_dn, s2, pct_lit, cnt)

def ingest_tile(path: str, *, day: str, source: str, store: LightsStore,
                lit_threshold: Optional[float] = None, nodata: Optional[float] = None,
                radius_m: int = 1500) -> int:
    """
    For each registered region/facility, compute stats over raster and write to DB.
    If a region is a point (facility), we buffer by radius_m (requires shapely).
    """
    meta = RasterMeta(path=path, day=_parse_date(day), source=str(source), nodata=nodata, lit_threshold=lit_threshold)
    regs = store.list_regions()
    if not regs:
        raise RuntimeError("No regions/facilities registered. Use --regions or --facilities first.")
    n = 0
    for r in regs:
        geom = json.loads(r["geom_geojson"])
        if r["type"] == "point":
            if shape is None:
                raise RuntimeError("shapely required for point buffering (facilities). pip install shapely")
            pt = shape(geom)
            # crude buffer in degrees if raster CRS is lat/lon; otherwise we rely on transform_geom
            # approximate meters->degrees (at equator); acceptable for small radii
            deg = radius_m / 111_000.0
            poly = mapping(pt.buffer(deg)) # type: ignore
            g = poly
        else:
            g = geom
        mean_dn, sum_dn, pct_lit, pixels = _read_raster_stats_over_geom(meta, g)
        store.write_obs(meta.day, meta.source, "facility" if r["type"] == "point" else "region",
                        r["id"], mean_dn, sum_dn, pct_lit, pixels)
        n += 1
    return n

# ----------------------------- publisher -----------------------------

def publish_latest(store: LightsStore, *, top_k: int = 10) -> None:
    if not publish_stream:
        return
    with store._cx() as cx:
        row = cx.execute("SELECT day FROM lights_index ORDER BY day DESC LIMIT 1").fetchone()
        if not row: return
        day = row["day"]
        movers = cx.execute("""
          SELECT level, key, mean_dn, dod, wow, mom, yoy, anomaly
          FROM lights_index
          WHERE day=? ORDER BY COALESCE(yoy,0) DESC LIMIT ?
        """, (day, top_k)).fetchall()
        payload = {
            "ts_ms": _utc_ms(),
            "day": day,
            "top": [dict(r) for r in movers],
        }
        publish_stream("altdata.satellite_lights", payload)
        if movers:
            m0 = dict(movers[0])
            publish_stream("ai.insight", {
                "ts_ms": payload["ts_ms"],
                "kind": "night_lights",
                "summary": f"{m0['level']} {m0['key']}: YoY {m0.get('yoy') or 0:+.1f}%, anomaly {m0.get('anomaly') or 0:+.2f}",
                "details": [f"DoD {m0.get('dod') or 0:+.1f}%, WoW {m0.get('wow') or 0:+.1f}%, MoM {m0.get('mom') or 0:+.1f}%"],
                "tags": ["altdata","night-lights", m0["key"]]
            })

# ----------------------------- CLI -----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Satellite Night Lights -> Economic Activity Index")
    ap.add_argument("--db", type=str, default=DB_PATH)
    ap.add_argument("--regions", type=str, help="Register GeoJSON regions (FeatureCollection with properties.id)")
    ap.add_argument("--facilities", type=str, help="Register facilities CSV (id,lat,lon[,ticker])")
    ap.add_argument("--id", type=str, default="id", help="Facilities id column")
    ap.add_argument("--lat", type=str, default="lat", help="Facilities lat column")
    ap.add_argument("--lon", type=str, default="lon", help="Facilities lon column")
    ap.add_argument("--ticker", type=str, help="Optional facilities ticker column")
    ap.add_argument("--load", type=str, help="Path to GeoTIFF tile")
    ap.add_argument("--date", type=str, help="Observation date YYYY-MM-DD")
    ap.add_argument("--source", type=str, default="viirs_m", help="viirs_m|npp_vcmcfg|dmsp_ols|custom")
    ap.add_argument("--nodata", type=float, help="Override nodata value")
    ap.add_argument("--thr", type=float, help="Lit pixel threshold (otherwise auto-percentile)")
    ap.add_argument("--radius", type=int, default=1500, help="Facility buffer radius (meters approx)")
    ap.add_argument("--index", action="store_true", help="Rebuild growth/trend index after ingest")
    ap.add_argument("--export", type=str, help="Export lights_index to CSV")
    ap.add_argument("--publish", action="store_true", help="Publish latest movers/insights")
    ap.add_argument("--probe", action="store_true", help="Create a tiny synthetic demo (no raster)")
    args = ap.parse_args()

    store = LightsStore(db_path=args.db)

    if args.regions:
        n = store.upsert_regions_geojson(args.regions)
        print(f"Registered {n} regions from {args.regions}")

    if args.facilities:
        n = store.upsert_facilities_csv(args.facilities, id_col=args.id, lat_col=args.lat, lon_col=args.lon, ticker_col=args.ticker)
        print(f"Registered {n} facilities from {args.facilities}")

    if args.probe:
        # synthetic series (no raster): write fake obs for 3 regions to test index/publish
        import random
        regs = store.list_regions() or [
            {"id":"R1","name":"Region 1","type":"polygon","props":"{}", "geom_geojson": json.dumps({"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]})},
            {"id":"R2","name":"Region 2","type":"polygon","props":"{}", "geom_geojson": json.dumps({"type":"Polygon","coordinates":[[[1,1],[2,1],[2,2],[1,2],[1,1]]]})},
            {"id":"PLANT_A","name":"Plant A","type":"point","props":"{}", "geom_geojson": json.dumps({"type":"Point","coordinates":[77.6,12.9]})},
        ]
        if not store.list_regions():
            with store._cx() as cx:
                for r in regs:
                    cx.execute("INSERT OR REPLACE INTO regions(id,name,type,props,geom_geojson) VALUES(?,?,?,?,?)",
                               (r["id"], r["name"], r["type"], r["props"], r["geom_geojson"]))
                cx.commit()
        base_day = "2025-07-01"
        from datetime import datetime, timedelta
        dt0 = datetime.strptime(base_day, "%Y-%m-%d")
        for i in range(40):
            d = (dt0 + timedelta(days=i)).strftime("%Y-%m-%d")
            for r in store.list_regions():
                mean = 10 + math.sin(i/5.0)*2 + random.uniform(-0.5,0.5)
                store.write_obs(d, "synthetic", "facility" if r["type"]=="point" else "region",
                                r["id"], mean, mean*1000, 50+random.uniform(-5,5), 500)
        store.build_index()
        if args.publish:
            publish_latest(store)
        if args.export:
            p = store.export_csv(args.export)
            print(f"Wrote {p}")
        return

    if args.load:
        assert args.date, "--date is required when using --load"
        n = ingest_tile(args.load, day=args.date, source=args.source, store=store,
                        lit_threshold=args.thr, nodata=args.nodata, radius_m=args.radius)
        print(f"Ingested stats for {n} regions/facilities from {args.load}")

    if args.index:
        rows = store.build_index()
        print(f"Indexed {rows} rows")

    if args.publish:
        publish_latest(store)
        print("Published latest night-lights movers")

    if args.export:
        p = store.export_csv(args.export)
        print(f"Wrote {p}")

if __name__ == "__main__":
    main()